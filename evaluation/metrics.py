"""Reusable, explicit Phase 4.3 offline classification and event metrics."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from cost_model import CostAssumptions, evaluate_cost


DEFAULT_REVIEW_BUDGET_PER_1000 = 50
DEFAULT_CALIBRATION_BINS = 10


def _binary_inputs(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be aligned one-dimensional arrays.")
    if len(labels) == 0 or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Metrics require both binary label classes.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("Probability metrics require finite scores in [0, 1].")
    return labels, scores


def ranking_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels, scores = _binary_inputs(labels, scores)
    return {
        "rows": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "pr_auc": round(float(average_precision_score(labels, scores)), 6),
    }


def decision_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    labels, scores = _binary_inputs(labels, scores)
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {
        "threshold": float(threshold),
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "f1": round(float(f1), 6),
        "false_positive_rate": round(float(fp / (fp + tn)), 6) if fp + tn else None,
        "false_alerts_per_1000_merchant_days": round(float(fp * 1000 / len(labels)), 6),
        "alerts": int(predictions.sum()),
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def review_budget_metrics(labels: np.ndarray, scores: np.ndarray, review_budget_per_1000: int = DEFAULT_REVIEW_BUDGET_PER_1000) -> dict[str, Any]:
    """Exact top-K selection for a fixed analyst capacity, stable under ties."""
    labels, scores = _binary_inputs(labels, scores)
    if review_budget_per_1000 < 1:
        raise ValueError("review_budget_per_1000 must be at least one.")
    k = min(len(labels), max(1, int(np.ceil(len(labels) * review_budget_per_1000 / 1000))))
    selected = np.argsort(-scores, kind="mergesort")[:k]
    selected_labels = labels[selected]
    selected_positives = int(selected_labels.sum())
    return {
        "review_budget_per_1000_merchant_days": review_budget_per_1000,
        "review_budget_rows": k,
        "precision_at_k": round(float(selected_positives / k), 6),
        "recall_at_fixed_review_budget": round(float(selected_positives / labels.sum()), 6),
        "score_cutoff_at_k": round(float(scores[selected[-1]]), 6),
    }


def calibration_metrics(labels: np.ndarray, scores: np.ndarray, bins: int = DEFAULT_CALIBRATION_BINS) -> dict[str, Any]:
    """Report raw-probability Brier score and uniform-bin expected calibration error."""
    labels, scores = _binary_inputs(labels, scores)
    if bins < 2:
        raise ValueError("bins must be at least two.")
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    details = []
    ece = 0.0
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        mask = (scores >= lower) & ((scores < upper) if index < bins - 1 else (scores <= upper))
        count = int(mask.sum())
        if not count:
            continue
        observed = float(labels[mask].mean())
        mean_score = float(scores[mask].mean())
        ece += count / len(labels) * abs(observed - mean_score)
        details.append({
            "lower": round(float(lower), 4),
            "upper": round(float(upper), 4),
            "count": count,
            "observed_positive_rate": round(observed, 6),
            "mean_score": round(mean_score, 6),
        })
    return {
        "score_semantics": "uncalibrated_model_probability",
        "brier_score": round(float(brier_score_loss(labels, scores)), 6),
        "expected_calibration_error": round(float(ece), 6),
        "bins": details,
    }


def cost_sensitivity(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    """Show decision-cost sensitivity without optimizing a threshold on this data."""
    labels, scores = _binary_inputs(labels, scores)
    confusion = decision_metrics(labels, scores, threshold)["confusion_matrix"]
    baseline = CostAssumptions()
    scenarios = {
        "default_illustrative": baseline,
        "false_negative_heavy": replace(baseline, undetected_fraud_loss_inr=baseline.undetected_fraud_loss_inr * 2),
        "false_positive_heavy": replace(
            baseline,
            manual_review_cost_inr=baseline.manual_review_cost_inr * 2,
            merchant_friction_cost_inr=baseline.merchant_friction_cost_inr * 2,
        ),
    }
    return {
        name: evaluate_cost(
            confusion["tp"], confusion["fp"], confusion["tn"], confusion["fn"], assumptions
        )
        for name, assumptions in scenarios.items()
    }


def event_metrics(scored: pd.DataFrame, events: Iterable[dict[str, Any]], threshold: float) -> dict[str, Any]:
    """Compute event-level recall and within-evaluation time to first alert."""
    required = {"merchant_id", "date", "score"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Event metrics require score frame fields: {missing}")
    working = scored.copy()
    working["date"] = pd.to_datetime(working["date"], errors="raise").dt.strftime("%Y-%m-%d")
    score_map = {(row.merchant_id, row.date): float(row.score) for row in working.itertuples(index=False)}
    outcomes: list[dict[str, Any]] = []
    for event in events:
        if event.get("kind") != "fraud_spike":
            continue
        merchant_id = str(event["merchant_id"])
        available_dates = sorted(
            date_value for date_value in event.get("affected_dates", []) if (merchant_id, date_value) in score_map
        )
        if not available_dates:
            continue
        detected_dates = [date_value for date_value in available_dates if score_map[(merchant_id, date_value)] >= threshold]
        start = pd.Timestamp(available_dates[0])
        first_detected = pd.Timestamp(detected_dates[0]) if detected_dates else None
        outcomes.append({
            "detected": bool(detected_dates),
            "time_to_detect_days": int((first_detected - start).days) if first_detected is not None else None,
        })
    if not outcomes:
        raise ValueError("No fraud events overlap the scored evaluation slice.")
    detected = [item for item in outcomes if item["detected"]]
    detection_times = [item["time_to_detect_days"] for item in detected]
    return {
        "events_evaluated": len(outcomes),
        "events_detected": len(detected),
        "event_level_recall": round(float(len(detected) / len(outcomes)), 6),
        "undetected_events": len(outcomes) - len(detected),
        "mean_time_to_detect_days": round(float(np.mean(detection_times)), 6) if detection_times else None,
        "median_time_to_detect_days": round(float(np.median(detection_times)), 6) if detection_times else None,
        "outcomes": outcomes,
    }


def merchant_bootstrap_intervals(
    scored: pd.DataFrame,
    summary: Callable[[np.ndarray, np.ndarray], dict[str, float]],
    *,
    resamples: int = 200,
    random_state: int = 42,
) -> dict[str, Any]:
    """Merchant-cluster bootstrap intervals, preserving within-merchant rows."""
    if resamples < 20:
        raise ValueError("resamples must be at least 20 for a bootstrap interval.")
    required = {"merchant_id", "label_fraud_spike", "score"}
    missing = sorted(required - set(scored.columns))
    if missing:
        raise ValueError(f"Bootstrap requires score frame fields: {missing}")
    groups = [group for _, group in scored.groupby("merchant_id", sort=True)]
    rng = np.random.default_rng(random_state)
    collected: dict[str, list[float]] = {}
    effective = 0
    for _ in range(resamples):
        sample = pd.concat([groups[index] for index in rng.integers(0, len(groups), size=len(groups))], ignore_index=True)
        try:
            values = summary(sample["label_fraud_spike"].to_numpy(), sample["score"].to_numpy())
        except ValueError:
            continue
        effective += 1
        for key, value in values.items():
            if value is not None and np.isfinite(value):
                collected.setdefault(key, []).append(float(value))
    if effective < 20:
        raise RuntimeError("Too few valid merchant bootstrap replicates.")
    return {
        "method": "merchant_cluster_bootstrap_percentile",
        "requested_resamples": resamples,
        "effective_resamples": effective,
        "random_state": random_state,
        "intervals": {
            key: {"lower_95": round(float(np.percentile(values, 2.5)), 6), "upper_95": round(float(np.percentile(values, 97.5)), 6)}
            for key, values in collected.items()
        },
    }


def event_bootstrap_intervals(event_summary: dict[str, Any], *, resamples: int = 200, random_state: int = 42) -> dict[str, Any]:
    """Event-cluster bootstrap for event recall and time-to-detect uncertainty."""
    outcomes = event_summary.get("outcomes", [])
    if len(outcomes) < 2:
        raise ValueError("At least two evaluated events are required for event bootstrap intervals.")
    rng = np.random.default_rng(random_state)
    recalls: list[float] = []
    medians: list[float] = []
    for _ in range(resamples):
        sample = [outcomes[index] for index in rng.integers(0, len(outcomes), size=len(outcomes))]
        detected = [item for item in sample if item["detected"]]
        recalls.append(len(detected) / len(sample))
        times = [item["time_to_detect_days"] for item in detected]
        if times:
            medians.append(float(np.median(times)))
    intervals: dict[str, dict[str, float | None]] = {
        "event_level_recall": {
            "lower_95": round(float(np.percentile(recalls, 2.5)), 6),
            "upper_95": round(float(np.percentile(recalls, 97.5)), 6),
        },
        "median_time_to_detect_days": {
            "lower_95": round(float(np.percentile(medians, 2.5)), 6) if medians else None,
            "upper_95": round(float(np.percentile(medians, 97.5)), 6) if medians else None,
        },
    }
    return {
        "method": "event_cluster_bootstrap_percentile",
        "resamples": resamples,
        "random_state": random_state,
        "intervals": intervals,
    }
