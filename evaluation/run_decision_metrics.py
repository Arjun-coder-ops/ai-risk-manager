"""Compute Phase 4.3 decision and uncertainty metrics for the frozen model."""

from __future__ import annotations

import argparse
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

from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402
from split import time_split  # noqa: E402

from metrics import (  # noqa: E402
    DEFAULT_REVIEW_BUDGET_PER_1000,
    calibration_metrics,
    cost_sensitivity,
    decision_metrics,
    event_bootstrap_intervals,
    event_metrics,
    merchant_bootstrap_intervals,
    ranking_metrics,
    review_budget_metrics,
)


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "decision_metrics.json")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model.joblib")
META_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model_meta.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "raw", "simulation_manifest.json")


def _event_windows() -> list[dict[str, Any]]:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle).get("events", [])


def _bootstrap_summary(review_threshold: float):
    def summary(labels, scores) -> dict[str, float]:
        ranking = ranking_metrics(labels, scores)
        decision = decision_metrics(labels, scores, review_threshold)
        calibration = calibration_metrics(labels, scores)
        return {
            "roc_auc": ranking["roc_auc"],
            "pr_auc": ranking["pr_auc"],
            "review_precision": decision["precision"],
            "review_recall": decision["recall"],
            "review_false_positive_rate": decision["false_positive_rate"],
            "brier_score": calibration["brier_score"],
            "expected_calibration_error": calibration["expected_calibration_error"],
        }
    return summary


def run(output_path: str = DEFAULT_OUTPUT, *, bootstrap_resamples: int = 200) -> dict[str, Any]:
    with open(META_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Frozen model artifact is incompatible with the current feature contract.")
    model = joblib.load(MODEL_PATH)
    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    _, _, test = time_split(featured)
    labels = test["label_fraud_spike"].to_numpy(dtype=int)
    scores = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    scored = test[["merchant_id", "date", "label_fraud_spike"]].copy()
    scored["score"] = scores
    review_event_metrics = event_metrics(scored, _event_windows(), float(metadata["review_threshold"]))
    high_event_metrics = event_metrics(scored, _event_windows(), float(metadata["high_threshold"]))
    report = {
        "report_kind": "phase-4.3-decision-metrics",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "evaluation_population": {
            "name": "frozen_declared_temporal_test",
            "rows": int(len(test)),
            "date_range": [str(test.date.min().date()), str(test.date.max().date())],
            "positive_rate": round(float(labels.mean()), 6),
            "disclosure": "Scores are recomputed only to expand metrics for the frozen artifact; this run performs no selection, tuning, calibration, or promotion.",
        },
        "no_information_baseline": {
            "roc_auc": 0.5,
            "pr_auc": round(float(labels.mean()), 6),
            "description": "Random ranking / prevalence baseline.",
        },
        "ranking": ranking_metrics(labels, scores),
        "review_tier": decision_metrics(labels, scores, float(metadata["review_threshold"])),
        "high_tier": decision_metrics(labels, scores, float(metadata["high_threshold"])),
        "fixed_review_budget": review_budget_metrics(labels, scores, DEFAULT_REVIEW_BUDGET_PER_1000),
        "raw_probability_calibration": calibration_metrics(labels, scores),
        "business_cost_sensitivity": {
            "review_tier": cost_sensitivity(labels, scores, float(metadata["review_threshold"])),
            "high_tier": cost_sensitivity(labels, scores, float(metadata["high_threshold"])),
        },
        "event_metrics": {"review_tier": review_event_metrics, "high_tier": high_event_metrics},
        "uncertainty_intervals": {
            "merchant_aware": merchant_bootstrap_intervals(scored, _bootstrap_summary(float(metadata["review_threshold"])), resamples=bootstrap_resamples),
            "event_aware": {
                "review_tier": event_bootstrap_intervals(review_event_metrics, resamples=bootstrap_resamples),
                "high_tier": event_bootstrap_intervals(high_event_metrics, resamples=bootstrap_resamples),
            },
        },
        "selection_boundary": "This report evaluates the frozen artifact. Probability calibration methods are evaluated separately in Phase 4.4.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4.3 decision metrics on the frozen temporal test artifact.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.output, bootstrap_resamples=args.bootstrap_resamples), indent=2))


if __name__ == "__main__":
    main()
