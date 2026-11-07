"""Run the Phase 4.1 classical candidate comparison on the validation period.

The test split is intentionally not loaded.  This report is evidence for later
model-selection and calibration work, not a production-promotion decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from experiments import fit_and_score, get_classical_candidates  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402
from split import time_split  # noqa: E402


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "classical_model_comparison.json")
FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")


def _library_versions() -> dict[str, str]:
    import lightgbm
    import xgboost

    return {"scikit_learn": sklearn_version, "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__}


def run(output_path: str = DEFAULT_OUTPUT, random_state: int = 42) -> dict:
    """Fit each required candidate against the common temporal validation split."""
    frame = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    train, validation, _ = time_split(frame)
    X_train = train[FEATURE_COLUMNS]
    y_train = train["label_fraud_spike"].to_numpy(dtype=int)
    X_validation = validation[FEATURE_COLUMNS]
    y_validation = validation["label_fraud_spike"].to_numpy(dtype=int)

    candidates = []
    for candidate in get_classical_candidates(random_state):
        _, scores = fit_and_score(candidate, X_train, y_train, X_validation)
        candidates.append({
            "name": candidate.name,
            "description": candidate.description,
            "score_kind": candidate.score_kind,
            "eligible_for_production_selection": candidate.eligible_for_production_selection,
            "validation_roc_auc": round(float(roc_auc_score(y_validation, scores)), 6),
            "validation_pr_auc": round(float(average_precision_score(y_validation, scores)), 6),
        })
    eligible = [candidate for candidate in candidates if candidate["eligible_for_production_selection"]]
    selected = max(eligible, key=lambda candidate: candidate["validation_pr_auc"])
    report = {
        "report_kind": "phase-4.1-classical-model-comparison",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "random_state": random_state,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "protocol": {
            "name": "temporal_validation_only",
            "train_date_range": [str(train.date.min().date()), str(train.date.max().date())],
            "validation_date_range": [str(validation.date.min().date()), str(validation.date.max().date())],
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "validation_positive_rate": round(float(y_validation.mean()), 6),
            "test_split_loaded": False,
        },
        "library_versions": _library_versions(),
        "candidates": candidates,
        "validation_leader_among_probability_models": selected["name"],
        "selection_boundary": (
            "This report does not promote or replace the deployed model. Heuristic and Isolation Forest scores "
            "are ranking baselines, not calibrated fraud probabilities. Promotion requires the remaining Phase 4 "
            "protocols, calibration, robustness, and ablation evidence."
        ),
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the classical validation-model comparison.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="JSON report path.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.random_state), indent=2))


if __name__ == "__main__":
    main()
