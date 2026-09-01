"""
Train and compare candidate models on TRAIN, select the final model and
decision threshold using VALIDATION ONLY. The held-out TEST set is never
touched in this file -- see evaluate.py for the one-time final measurement.

Models compared:
  - Logistic Regression (balanced class weights) - interpretable baseline
  - Random Forest (balanced class weights)
  - Gradient Boosting (via HistGradientBoostingClassifier)

Model selection criterion: PR-AUC on validation (appropriate for a rare
positive class; ROC-AUC is overly optimistic under imbalance). We don't
default to the most complex model -- we report all three and only pick the
most complex one if it actually wins on validation.

Threshold selection: for the winning model, we sweep thresholds on
VALIDATION predictions and pick the threshold that minimizes total
estimated business cost (see cost_model.py), not simply the one that
maximizes F1. This directly operationalizes the "false-positive cost"
requirement into the policy, rather than treating it as an afterthought
metric computed after the fact.
"""

import json
import joblib
import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, confusion_matrix

from features import FEATURE_COLUMNS
from split import time_split
from cost_model import CostAssumptions, evaluate_cost

MODEL_VERSION = "fraud-spike-v1"
MODEL_OUT_PATH = os.path.join(REPO_ROOT, "model/artifacts/model.joblib")
META_OUT_PATH = os.path.join(REPO_ROOT, "model/artifacts/model_meta.json")


def get_candidates():
    return {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, C=1.0, random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.08, max_iter=200,
            l2_regularization=1.0, random_state=42
        ),
    }


def sweep_thresholds(y_true, y_scores, assumptions: CostAssumptions):
    best = None
    for t in np.arange(0.05, 0.96, 0.01):
        preds = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = evaluate_cost(tp, fp, tn, fn, assumptions)
        total_cost = cost["total_estimated_cost_inr"]
        p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="binary", zero_division=0)
        candidate = {
            "threshold": round(float(t), 2), "total_cost": total_cost,
            "precision": p, "recall": r, "f1": f1, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        }
        if best is None or total_cost < best["total_cost"]:
            best = candidate
    return best


def main():
    df = pd.read_csv(os.path.join(REPO_ROOT, "data/processed/features.csv"), parse_dates=["date"])
    train, val, test = time_split(df)

    X_train, y_train = train[FEATURE_COLUMNS], train["label_fraud_spike"]
    X_val, y_val = val[FEATURE_COLUMNS], val["label_fraud_spike"]

    candidates = get_candidates()
    results = {}
    fitted = {}
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        val_scores = clf.predict_proba(X_val)[:, 1]
        pr_auc = average_precision_score(y_val, val_scores)
        results[name] = {"val_pr_auc": round(float(pr_auc), 4)}
        fitted[name] = clf
        print(f"{name}: validation PR-AUC = {pr_auc:.4f}")

    best_name = max(results, key=lambda k: results[k]["val_pr_auc"])
    best_model = fitted[best_name]
    print(f"\nSelected model: {best_name} (highest validation PR-AUC)")

    val_scores = best_model.predict_proba(X_val)[:, 1]
    assumptions = CostAssumptions()
    # REVIEW threshold: the cost-optimal single cutoff found by sweeping validation
    # predictions against the business cost model (minimizes manual-review + friction
    # + undetected-fraud cost together). Because undetected fraud is assumed far more
    # costly than a manual review, this cutoff is deliberately recall-leaning -- a
    # human still reviews these, so lower precision here is an acceptable trade.
    review_threshold_info = sweep_thresholds(y_val.values, val_scores, assumptions)

    # HIGH threshold: a stricter, higher-precision cutoff used for direct
    # merchant-facing alerts (no human in the loop before the merchant is notified),
    # so we require precision >= 0.95 on validation before recommending immediate
    # investigation without review.
    high_candidate = None
    for t in np.arange(0.20, 0.96, 0.01):
        preds = (val_scores >= t).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_val.values, preds, average="binary", zero_division=0)
        if p >= 0.95:
            high_candidate = {"threshold": round(float(t), 2), "precision": p, "recall": r, "f1": f1}
            break
    if high_candidate is None:
        high_candidate = {"threshold": 0.5, "precision": None, "recall": None, "f1": None}

    print(f"REVIEW threshold (min validation cost): {review_threshold_info['threshold']}")
    print(json.dumps(review_threshold_info, indent=2))
    print(f"HIGH threshold (precision>=0.95 on validation): {high_candidate['threshold']}")
    print(json.dumps(high_candidate, indent=2))

    os.makedirs(os.path.join(REPO_ROOT, "model", "artifacts"), exist_ok=True)
    joblib.dump(best_model, MODEL_OUT_PATH)

    meta = {
        "model_version": MODEL_VERSION,
        "model_type": best_name,
        "feature_columns": FEATURE_COLUMNS,
        "review_threshold": review_threshold_info["threshold"],
        "high_threshold": high_candidate["threshold"],
        "model_comparison_val_pr_auc": {k: v["val_pr_auc"] for k, v in results.items()},
        "review_threshold_val_metrics": review_threshold_info,
        "high_threshold_val_metrics": high_candidate,
        "cost_assumptions": {
            "manual_review_cost_inr": assumptions.manual_review_cost_inr,
            "merchant_friction_cost_inr": assumptions.merchant_friction_cost_inr,
            "undetected_fraud_loss_inr": assumptions.undetected_fraud_loss_inr,
        },
    }
    with open(META_OUT_PATH, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nSaved model to {MODEL_OUT_PATH}")
    print(f"Saved metadata to {META_OUT_PATH}")


if __name__ == "__main__":
    main()
