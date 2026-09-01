"""
FINAL HELD-OUT TEST EVALUATION.

This script touches the TEST split exactly once, using the model and
thresholds that were already selected using TRAIN/VALIDATION only (see
train.py). Nothing here is re-tuned against the test set. Numbers printed
and saved by this script are the actual numbers reported in the README --
none of them are hand-written or illustrative.
"""

import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))
from features import FEATURE_COLUMNS  # noqa: E402
from split import time_split  # noqa: E402
from cost_model import CostAssumptions, evaluate_cost  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts", "model.joblib")
META_PATH = os.path.join(os.path.dirname(__file__), "..", "model", "artifacts", "model_meta.json")
OUT_PATH = os.path.join(os.path.dirname(__file__), "test_results.json")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv")


def evaluate_at_threshold(y_true, scores, threshold, label):
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
    cost = evaluate_cost(int(tp), int(fp), int(tn), int(fn), CostAssumptions())
    return {
        "tier_label": label,
        "threshold": threshold,
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(float(fp) / (fp + tn), 4) if (fp + tn) else None,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
        "cost": cost,
    }


def main():
    with open(META_PATH) as f:
        meta = json.load(f)
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    _, _, test = time_split(df)

    X_test = test[FEATURE_COLUMNS]
    y_test = test["label_fraud_spike"].values
    scores = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    review_eval = evaluate_at_threshold(y_test, scores, meta["review_threshold"], "REVIEW (>=) — flag for analyst")
    high_eval = evaluate_at_threshold(y_test, scores, meta["high_threshold"], "HIGH (>=) — direct merchant alert")

    results = {
        "model_version": meta["model_version"],
        "model_type": meta["model_type"],
        "test_set_size": int(len(test)),
        "test_positive_rate": round(float(y_test.mean()), 4),
        "test_date_range": [str(test.date.min().date()), str(test.date.max().date())],
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "review_tier": review_eval,
        "high_tier": high_eval,
        "note": (
            "These are the actual measured results of running the trained model on the "
            "held-out test split (dates after the validation window, never used for "
            "training or threshold selection). Cost figures use the illustrative, "
            "configurable assumptions in cost_model.py, not real Razorpay financial data."
        ),
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
