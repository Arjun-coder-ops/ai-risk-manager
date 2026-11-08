"""Execute Phase 4.6 feature, calibration, and model-family ablations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from calibration import ScoreCalibrator  # noqa: E402
from experiments import fit_and_score, get_classical_candidates  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402

from ablations import (  # noqa: E402
    ABLATION_CALIBRATION_END,
    ABLATION_EVALUATION_END,
    ABLATION_TRAIN_END,
    GLOBAL_FEATURE_COLUMNS,
    REDUCED_FEATURE_COLUMNS,
    ablation_partitions,
    global_feature_frame,
)
from metrics import calibration_metrics, ranking_metrics  # noqa: E402


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "ablation_experiments.json")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "raw", "simulation_manifest.json")


def _events() -> list[dict[str, Any]]:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle).get("events", [])


def _ranking(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Ranking comparison that accepts heuristic/anomaly scores of any finite scale."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Ablation scores must be finite.")
    return {
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "pr_auc": round(float(average_precision_score(labels, scores)), 6),
    }


def _hist_gradient_spec():
    return next(candidate for candidate in get_classical_candidates() if candidate.name == "hist_gradient_boosting")


def _feature_ablation(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Any]:
    candidate = _hist_gradient_spec()
    labels = evaluation["label_fraud_spike"].to_numpy(dtype=int)
    global_train, global_evaluation = global_feature_frame(train), global_feature_frame(evaluation)
    feature_sets = {
        "full_merchant_relative_contract": (train[FEATURE_COLUMNS], evaluation[FEATURE_COLUMNS], list(FEATURE_COLUMNS)),
        "reduced_core_merchant_relative": (train[list(REDUCED_FEATURE_COLUMNS)], evaluation[list(REDUCED_FEATURE_COLUMNS)], list(REDUCED_FEATURE_COLUMNS)),
        "global_raw_baseline": (global_train, global_evaluation, list(GLOBAL_FEATURE_COLUMNS)),
    }
    results = {}
    for name, (X_train, X_evaluation, columns) in feature_sets.items():
        _, scores = fit_and_score(candidate, X_train, train["label_fraud_spike"].to_numpy(dtype=int), X_evaluation)
        results[name] = {
            "feature_count": len(columns),
            "feature_columns": columns,
            "ranking": _ranking(labels, scores),
        }
    return results


def _model_family_ablation(train: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Any]:
    labels = evaluation["label_fraud_spike"].to_numpy(dtype=int)
    results = []
    for candidate in get_classical_candidates():
        _, scores = fit_and_score(
            candidate,
            train[FEATURE_COLUMNS],
            train["label_fraud_spike"].to_numpy(dtype=int),
            evaluation[FEATURE_COLUMNS],
        )
        results.append({
            "name": candidate.name,
            "score_kind": candidate.score_kind,
            "eligible_for_production_selection": candidate.eligible_for_production_selection,
            "ranking": _ranking(labels, scores),
        })
    return {"common_feature_columns": list(FEATURE_COLUMNS), "results": results}


def _calibration_ablation(train: pd.DataFrame, calibration: pd.DataFrame, evaluation: pd.DataFrame) -> dict[str, Any]:
    candidate = _hist_gradient_spec()
    model, calibration_scores = fit_and_score(
        candidate,
        train[FEATURE_COLUMNS],
        train["label_fraud_spike"].to_numpy(dtype=int),
        calibration[FEATURE_COLUMNS],
    )
    raw_evaluation_scores = model.predict_proba(evaluation[FEATURE_COLUMNS])[:, 1]
    calibration_labels = calibration["label_fraud_spike"].to_numpy(dtype=int)
    evaluation_labels = evaluation["label_fraud_spike"].to_numpy(dtype=int)
    score_sets: dict[str, np.ndarray] = {"raw": raw_evaluation_scores}
    for method in ("sigmoid", "isotonic"):
        mapping = ScoreCalibrator(method).fit(calibration_scores, calibration_labels)
        score_sets[method] = mapping.predict(raw_evaluation_scores)
    return {
        name: {
            "ranking": ranking_metrics(evaluation_labels, scores),
            "calibration": calibration_metrics(
                evaluation_labels,
                scores,
                score_semantics=(
                    "uncalibrated_model_probability" if name == "raw" else f"{name}_calibration_adjusted_probability"
                ),
            ),
        }
        for name, scores in score_sets.items()
    }


def run(output_path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    partitions = ablation_partitions(featured, _events())
    report = {
        "report_kind": "phase-4.6-ablation-experiments",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "protocol": {
            "train_date_range": [str(partitions.train.date.min()), str(partitions.train.date.max())],
            "calibration_date_range": [str(partitions.calibration.date.min()), str(partitions.calibration.date.max())],
            "common_evaluation_date_range": [str(partitions.evaluation.date.min()), str(partitions.evaluation.date.max())],
            "train_rows": len(partitions.train),
            "calibration_rows": len(partitions.calibration),
            "evaluation_rows": len(partitions.evaluation),
            "evaluation_positive_rate": round(float(partitions.evaluation.label_fraud_spike.mean()), 6),
            "boundaries": {
                "train_end": ABLATION_TRAIN_END,
                "calibration_end": ABLATION_CALIBRATION_END,
                "evaluation_end": ABLATION_EVALUATION_END,
            },
            "purged_boundary_spanning_fraud_events": partitions.purged_event_count,
            "purged_rows": partitions.purged_rows,
            "later_calibration_selection_period": ["2026-04-01", "2026-04-15"],
            "threshold_selection_period": ["2026-04-16", "2026-05-15"],
            "final_test_period": ["2026-05-16", "2026-06-29"],
        },
        "feature_ablations": _feature_ablation(partitions.train, partitions.evaluation),
        "calibration_ablation": _calibration_ablation(
            partitions.train, partitions.calibration, partitions.evaluation
        ),
        "model_family_comparison": _model_family_ablation(partitions.train, partitions.evaluation),
        "selection_boundary": (
            "This is a reproducible offline comparison only. It does not alter the frozen serving model, thresholds, "
            "calibration decision, or final test report. Heuristic and Isolation Forest remain non-probability baselines."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4.6 leakage-conscious ablation experiments.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
