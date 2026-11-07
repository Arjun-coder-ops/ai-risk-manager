"""Regression tests for Phase 4.3 decision, event, and bootstrap metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "model"))

from metrics import (  # noqa: E402
    calibration_metrics,
    decision_metrics,
    event_bootstrap_intervals,
    event_metrics,
    merchant_bootstrap_intervals,
    ranking_metrics,
    review_budget_metrics,
)


def labels_and_scores():
    return np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.4, 0.1])


def test_decision_and_fixed_budget_metrics_have_exact_count_semantics():
    labels, scores = labels_and_scores()
    decision = decision_metrics(labels, scores, 0.5)
    assert decision["confusion_matrix"] == {"tp": 1, "fp": 1, "tn": 1, "fn": 1}
    assert decision["precision"] == 0.5
    assert decision["recall"] == 0.5
    assert decision["false_alerts_per_1000_merchant_days"] == 250.0
    budget = review_budget_metrics(labels, scores, review_budget_per_1000=500)
    assert budget["review_budget_rows"] == 2
    assert budget["precision_at_k"] == 0.5
    assert budget["recall_at_fixed_review_budget"] == 0.5


def test_calibration_metrics_and_merchant_bootstrap_are_deterministic():
    labels, scores = labels_and_scores()
    calibration = calibration_metrics(labels, scores, bins=2)
    assert calibration["brier_score"] == 0.255
    assert calibration["expected_calibration_error"] >= 0
    assert calibration["score_semantics"] == "uncalibrated_model_probability"
    scored = pd.DataFrame({
        "merchant_id": ["A", "A", "B", "B"],
        "label_fraud_spike": labels,
        "score": scores,
    })
    first = merchant_bootstrap_intervals(scored, ranking_metrics, resamples=20, random_state=9)
    second = merchant_bootstrap_intervals(scored, ranking_metrics, resamples=20, random_state=9)
    assert first == second
    assert first["intervals"]["roc_auc"]["lower_95"] <= first["intervals"]["roc_auc"]["upper_95"]


def test_event_recall_time_to_detect_and_event_bootstrap_only_use_overlapping_days():
    scored = pd.DataFrame({
        "merchant_id": ["A", "A", "A", "B", "B"],
        "date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-01", "2026-01-02"],
        "score": [0.2, 0.9, 0.1, 0.1, 0.2],
    })
    events = [
        {"kind": "fraud_spike", "merchant_id": "A", "affected_dates": ["2025-12-31", "2026-01-01", "2026-01-02"]},
        {"kind": "fraud_spike", "merchant_id": "B", "affected_dates": ["2026-01-01", "2026-01-02"]},
    ]
    report = event_metrics(scored, events, threshold=0.5)
    assert report["events_evaluated"] == 2
    assert report["events_detected"] == 1
    assert report["event_level_recall"] == 0.5
    assert report["mean_time_to_detect_days"] == 1.0
    intervals = event_bootstrap_intervals(report, resamples=20, random_state=7)
    assert intervals["intervals"]["event_level_recall"]["lower_95"] <= 0.5 <= intervals["intervals"]["event_level_recall"]["upper_95"]
