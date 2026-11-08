"""Run reproducible Phase 5 PyTorch experiments without changing the serving model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from deep_learning import (  # noqa: E402
    DeepLearningConfig,
    FeatureNormalizer,
    LSTMSequenceClassifier,
    MLPClassifier,
    NormalBehaviorAutoencoder,
    TCNSequenceClassifier,
    build_sequence_dataset,
    classifier_scores,
    measure_latency_ms,
    parameter_count,
    reconstruction_errors,
    set_deterministic,
    train_autoencoder,
    train_classifier,
)
from experiments import fit_and_score, get_classical_candidates  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402
from split import time_split  # noqa: E402

from metrics import ranking_metrics  # noqa: E402


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "deep_learning_experiments.json")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")


def _ranking_any(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "rows": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "pr_auc": round(float(average_precision_score(labels, scores)), 6),
    }


def _classifier_report(
    name: str,
    model: torch.nn.Module,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    validation_values: np.ndarray,
    validation_labels: np.ndarray,
    config: DeepLearningConfig,
    checkpoint_path: str,
    hypothesis: str,
    architecture: str,
) -> dict[str, Any]:
    training = _portable_checkpoint_evidence(train_classifier(
        model, train_values, train_labels, validation_values, validation_labels, config, checkpoint_path
    ))
    scores = classifier_scores(model, validation_values)
    latency_values = validation_values[: min(512, len(validation_values))]
    return {
        "name": name,
        "hypothesis": hypothesis,
        "architecture": architecture,
        "parameter_count": parameter_count(model),
        "training": training,
        "validation_ranking": ranking_metrics(validation_labels, scores),
        "validation_latency_cpu": measure_latency_ms(model, latency_values),
    }


def _portable_checkpoint_evidence(training: dict[str, Any]) -> dict[str, Any]:
    """Retain checkpoint proof without committing a machine-specific temp path."""
    output = dict(training)
    output["checkpoint_filename"] = os.path.basename(output.pop("checkpoint_path"))
    output["checkpoint_retention"] = "transient_local_execution_artifact; state hash records the loaded best checkpoint"
    return output


def _candidate_by_name(name: str):
    return next(candidate for candidate in get_classical_candidates() if candidate.name == name)


def _classical_comparators(train: pd.DataFrame, validation: pd.DataFrame) -> dict[str, Any]:
    labels = validation["label_fraud_spike"].to_numpy(dtype=int)
    output = {}
    for name in ("heuristic_baseline", "isolation_forest", "hist_gradient_boosting"):
        candidate = _candidate_by_name(name)
        _, scores = fit_and_score(
            candidate,
            train[FEATURE_COLUMNS],
            train["label_fraud_spike"].to_numpy(dtype=int),
            validation[FEATURE_COLUMNS],
        )
        output[name] = {
            "score_kind": candidate.score_kind,
            "validation_ranking": _ranking_any(labels, scores),
        }
    return output


def run(output_path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    config = DeepLearningConfig(max_epochs=25, patience=4)
    set_deterministic(config.seed)
    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    train, validation, _ = time_split(featured)
    train_values = train[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    validation_values = validation[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    train_labels = train["label_fraud_spike"].to_numpy(dtype=np.int64)
    validation_labels = validation["label_fraud_spike"].to_numpy(dtype=np.int64)
    normalizer = FeatureNormalizer().fit(train_values)
    normalized_train = normalizer.transform(train_values)
    normalized_validation = normalizer.transform(validation_values)
    checkpoint_dir = os.path.join(tempfile.gettempdir(), "ai-risk-manager-phase5-checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    set_deterministic(config.seed)
    mlp = MLPClassifier(len(FEATURE_COLUMNS), config.hidden_dim, config.dropout)
    mlp_report = _classifier_report(
        "mlp",
        mlp,
        normalized_train,
        train_labels,
        normalized_validation,
        validation_labels,
        config,
        os.path.join(checkpoint_dir, "mlp.pt"),
        "A compact non-linear tabular model may match classical ranking while remaining operationally simple.",
        "17 normalized end-of-day features -> 48 ReLU -> 24 ReLU -> binary logit.",
    )

    normal_train = normalized_train[train_labels == 0]
    normal_validation = normalized_validation[validation_labels == 0]
    set_deterministic(config.seed)
    autoencoder = NormalBehaviorAutoencoder(len(FEATURE_COLUMNS), config.hidden_dim)
    autoencoder_training = _portable_checkpoint_evidence(train_autoencoder(
        autoencoder,
        normal_train,
        normal_validation,
        config,
        os.path.join(checkpoint_dir, "autoencoder.pt"),
    ))
    autoencoder_scores = reconstruction_errors(autoencoder, normalized_validation)
    autoencoder_report = {
        "name": "normal_behavior_autoencoder",
        "hypothesis": "Reconstruction error after normal-only training can rank unusual merchant-days without fraud labels during fitting.",
        "architecture": "17 normalized features -> 48 -> 12 latent dimensions -> 48 -> reconstruction.",
        "parameter_count": parameter_count(autoencoder),
        "training": autoencoder_training,
        "validation_ranking": _ranking_any(validation_labels, autoencoder_scores),
        "validation_latency_cpu": measure_latency_ms(autoencoder, normalized_validation[: min(512, len(normalized_validation))]),
        "score_semantics": "reconstruction_error_anomaly_score_not_probability",
    }

    sequence_data = build_sequence_dataset(featured, FEATURE_COLUMNS, sequence_length=config.sequence_length)
    sequence_dates = pd.to_datetime(sequence_data.target_dates)
    train_end = pd.Timestamp(train.date.max())
    validation_end = pd.Timestamp(validation.date.max())
    sequence_train_mask = sequence_dates <= train_end
    sequence_validation_mask = (sequence_dates > train_end) & (sequence_dates <= validation_end)
    sequence_train = normalizer.transform(sequence_data.values[sequence_train_mask])
    sequence_validation = normalizer.transform(sequence_data.values[sequence_validation_mask])
    sequence_train_labels = sequence_data.labels[sequence_train_mask]
    sequence_validation_labels = sequence_data.labels[sequence_validation_mask]
    if set(np.unique(sequence_train_labels)) != {0, 1} or set(np.unique(sequence_validation_labels)) != {0, 1}:
        raise RuntimeError("Sequence experiment needs both classes in train and validation target windows.")

    set_deterministic(config.seed)
    lstm = LSTMSequenceClassifier(len(FEATURE_COLUMNS), config.hidden_dim, config.dropout)
    lstm_report = _classifier_report(
        "lstm",
        lstm,
        sequence_train,
        sequence_train_labels,
        sequence_validation,
        sequence_validation_labels,
        config,
        os.path.join(checkpoint_dir, "lstm.pt"),
        "A recurrent model may detect developing multi-day merchant patterns missed by a single end-of-day feature row.",
        "14 observed merchant-day feature vectors -> 48-unit LSTM -> binary logit; every target window ends at its prediction day.",
    )

    set_deterministic(config.seed)
    tcn = TCNSequenceClassifier(len(FEATURE_COLUMNS), config.hidden_dim, config.dropout)
    tcn_report = _classifier_report(
        "tcn",
        tcn,
        sequence_train,
        sequence_train_labels,
        sequence_validation,
        sequence_validation_labels,
        config,
        os.path.join(checkpoint_dir, "tcn.pt"),
        "A causal convolution may match recurrent temporal detection with fewer sequential operations and lower latency.",
        "14 observed merchant-day feature vectors -> causal 1x1 projection -> dilated residual convolutions (1,2) -> binary logit.",
    )

    report = {
        "report_kind": "phase-5-deep-learning-experiments",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "library_versions": {"torch": torch.__version__, "torch_cuda_available": bool(torch.cuda.is_available())},
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "configuration": config.as_dict(),
        "protocol": {
            "train_date_range": [str(train.date.min().date()), str(train.date.max().date())],
            "validation_date_range": [str(validation.date.min().date()), str(validation.date.max().date())],
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_split_loaded": False,
            "normalization": "FeatureNormalizer is fitted only on tabular training rows and then applied unchanged to validation rows and sequences.",
            "sequence_training_targets": int(sequence_train_mask.sum()),
            "sequence_validation_targets": int(sequence_validation_mask.sum()),
            "sequence_information_boundary": "A sequence ends on its target merchant-day and contains no later feature rows or labels.",
        },
        "models": {
            "mlp": mlp_report,
            "normal_behavior_autoencoder": autoencoder_report,
            "lstm": lstm_report,
            "tcn": tcn_report,
        },
        "classical_comparators": _classical_comparators(train, validation),
        "conditional_models": {
            "transformer": "not_implemented: 180-day synthetic histories and the observed LSTM/TCN comparison do not yet justify additional sequence complexity.",
            "graph_neural_network": "not_implemented: merchant-day aggregates contain no defensible merchant-device-account graph relationships.",
        },
        "operational_decision": (
            "The serving HistGradientBoosting model remains operational. These synthetic validation experiments alone do not "
            "justify promotion of a neural model, especially given Phase 4 robustness failures."
        ),
        "selection_boundary": "No deep-learning experiment loaded, scored, or tuned on the declared final temporal test split.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 5 deterministic CPU deep-learning experiments.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
