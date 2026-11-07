"""Execute Phase 4.2 generalization protocols with a frozen model artifact."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any

import joblib
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

try:  # Supports both ``python evaluation/...py`` and package-style imports.
    from evaluation.ood_benchmark import run_ood_benchmark  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised by the CLI invocation.
    from ood_benchmark import run_ood_benchmark  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_features  # noqa: E402
from simulator import SimulationConfig, generate_simulation  # noqa: E402
from split import TRAIN_END, VAL_END  # noqa: E402

from protocols import (  # noqa: E402
    cold_start_protocol,
    merchant_generalization_split,
    merchant_split_metadata,
    rank_metrics,
    simulation_stress_configs,
    stress_config_manifest,
)


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "generalization_protocols.json")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model.joblib")
META_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model_meta.json")
TEMPORAL_REPORT_PATH = os.path.join(REPO_ROOT, "evaluation", "test_results.json")


def _score(model: Any, frame: pd.DataFrame) -> dict[str, float]:
    scores = model.predict_proba(frame[FEATURE_COLUMNS])[:, 1]
    return rank_metrics(frame["label_fraud_spike"].to_numpy(), scores)


def _merchant_protocol(featured: pd.DataFrame) -> dict[str, Any]:
    split = merchant_generalization_split(featured, train_end=TRAIN_END, validation_end=VAL_END)
    # Hyperparameters are frozen to the already-selected deployed family. This
    # protocol does not compare candidates or choose a model on its test cohort.
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_depth=4, learning_rate=0.08, max_iter=200, l2_regularization=1.0, random_state=42
    )
    model.fit(split.train[FEATURE_COLUMNS], split.train["label_fraud_spike"])
    metadata = merchant_split_metadata(split)
    metadata.update({
        "protocol": "merchant_disjoint_and_chronological",
        "frozen_model_family": "hist_gradient_boosting",
        "validation_ranking": _score(model, split.validation),
        "test_ranking": _score(model, split.test),
        "selection_boundary": "No hyperparameter, model-family, threshold, or calibration choice used the merchant-test cohort.",
    })
    return metadata


def _stress_protocol(model: Any) -> dict[str, Any]:
    scenarios = {}
    for name, overrides in simulation_stress_configs().items():
        # These are independently seeded evaluation populations, separate from
        # the committed training data. They remain synthetic evidence only.
        config = replace(SimulationConfig(n_merchants=100, n_days=180), **overrides)
        result = generate_simulation(config)
        featured = build_features(result.observations)
        scored = featured[featured["has_sufficient_history"]].copy()
        scenarios[name] = {
            "config": stress_config_manifest(config),
            "raw_merchant_days": int(len(result.observations)),
            "scored_merchant_days": int(len(scored)),
            "ranking": _score(model, scored),
            "disclosure": "Synthetic stress evidence only; it is not real-world fraud validation.",
        }
    return scenarios


def run(output_path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Generate all Phase 4.2 protocol evidence without artifact mutation."""
    with open(META_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise RuntimeError("Frozen model artifact does not match the current feature schema.")
    if metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Frozen model artifact does not match the current feature columns.")
    model = joblib.load(MODEL_PATH)
    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    with open(TEMPORAL_REPORT_PATH, encoding="utf-8") as handle:
        temporal_report = json.load(handle)
    report = {
        "report_kind": "phase-4.2-generalization-protocols",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "temporal_generalization": {
            "status": "reuses_frozen_held_out_evaluation",
            "source": "evaluation/test_results.json",
            "test_split_rescored": False,
            "date_range": temporal_report["test_date_range"],
            "rows": temporal_report["test_set_size"],
            "positive_rate": temporal_report["test_positive_rate"],
            "roc_auc": temporal_report["roc_auc"],
            "pr_auc": temporal_report["pr_auc"],
            "disclosure": "The declared final temporal test was not rescored or retuned by this protocol runner.",
        },
        "merchant_generalization": _merchant_protocol(featured),
        "ood_generalization": run_ood_benchmark(),
        "cold_start": cold_start_protocol(featured),
        "stress_tests": _stress_protocol(model),
        "selection_boundary": (
            "All non-temporal populations are synthetic. This report is evaluation evidence only and does not "
            "promote, calibrate, tune, or replace a model."
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4.2 generalization protocols.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON report path.")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
