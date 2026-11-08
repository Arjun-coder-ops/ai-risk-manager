"""Evaluate versioned three-action policies on validation data only.

This runner is evidence for policy discussion, not an optimizer or deployment
mechanism.  It never changes the active policy artifact or uses final-test
labels to choose a policy.
"""

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

from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402
from policy import ThreeActionPolicy, evaluate_three_action_policy, load_policy  # noqa: E402
from split import time_split  # noqa: E402


FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model.joblib")
MODEL_META_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model_meta.json")
POLICY_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "policy.json")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "policy_analysis.json")


def _candidate_policy(active: ThreeActionPolicy, *, version: str, review: float, high: float, tolerance: str) -> ThreeActionPolicy:
    return replace(
        active,
        policy_version=version,
        review_threshold=review,
        high_threshold=high,
        risk_tolerance=tolerance,
    )


def run(output_path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    with open(MODEL_META_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Model metadata is incompatible with the current feature contract.")
    active = load_policy(POLICY_PATH)
    if active.model_version != metadata["model_version"]:
        raise RuntimeError("Active policy and model metadata refer to different model versions.")

    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    _, validation, _ = time_split(featured)
    model = joblib.load(MODEL_PATH)
    scores = model.predict_proba(validation[FEATURE_COLUMNS])[:, 1]
    labels = validation["label_fraud_spike"].to_numpy(dtype=int)

    candidates = [
        active,
        _candidate_policy(
            active,
            version="three-action-candidate-recall-v1",
            review=0.05,
            high=0.55,
            tolerance="candidate_more_recall_with_higher_review_volume",
        ),
        _candidate_policy(
            active,
            version="three-action-candidate-conservative-v1",
            review=0.20,
            high=0.80,
            tolerance="candidate_lower_alert_volume_with_more_missed_risk",
        ),
    ]
    report = {
        "report_kind": "phase-6-three-action-policy-analysis",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata["model_version"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "evaluation_population": {
            "name": "validation_only",
            "date_range": [str(validation.date.min().date()), str(validation.date.max().date())],
            "rows": int(len(validation)),
            "positive_rate": round(float(labels.mean()), 6),
            "final_test_loaded": False,
        },
        "active_policy_artifact": active.to_dict(),
        "candidate_policies": {
            policy.policy_version: {
                "configuration": policy.to_dict(),
                "offline_evaluation": evaluate_three_action_policy(labels, scores, policy),
            }
            for policy in candidates
        },
        "operational_decision": (
            "three-action-v1 remains the active policy because it mirrors the existing validation-selected thresholds. "
            "Candidate rows are transparent sensitivity analysis only; no candidate is selected or deployed by this runner."
        ),
        "separation_of_concerns": {
            "model_estimated_risk": "The serialized model produces an uncalibrated class-1 score.",
            "policy_decision": "The versioned policy maps the score to one exclusive LOW, REVIEW, or HIGH action.",
            "human_investigation_outcome": "Analyst feedback and case resolution are stored separately and never overwrite the model score or policy action.",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate transparent three-action risk policies on validation data.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
