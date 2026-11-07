"""Leakage-conscious Phase 4.2 generalization and stress protocols.

The functions in this module are offline-only.  They never mutate model
artifacts, tune a threshold, or expose synthetic labels through an API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


@dataclass(frozen=True)
class MerchantGeneralizationSplit:
    """Disjoint merchant cohorts with chronological boundaries per cohort."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_merchants: tuple[str, ...]
    validation_merchants: tuple[str, ...]
    test_merchants: tuple[str, ...]
    random_state: int


def _require_binary_labels(frame: pd.DataFrame, context: str) -> None:
    if "label_fraud_spike" not in frame or frame["label_fraud_spike"].nunique() != 2:
        raise ValueError(f"{context} must contain both offline synthetic label classes.")


def rank_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Ranking metrics only; decision metrics are added in Phase 4.3."""
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    _require_binary_labels(pd.DataFrame({"label_fraud_spike": labels}), "Evaluation slice")
    return {
        "rows": int(len(labels)),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 6),
        "pr_auc": round(float(average_precision_score(labels, scores)), 6),
    }


def merchant_generalization_split(
    featured: pd.DataFrame,
    *,
    train_end: str,
    validation_end: str,
    random_state: int = 42,
) -> MerchantGeneralizationSplit:
    """Create group-disjoint train/validation/test cohorts with future dates.

    Training uses only early dates from one merchant cohort.  Validation and
    test use wholly different merchants and progressively later date windows,
    so neither merchant identity nor future observations cross a boundary.
    """
    required = {"merchant_id", "date", "has_sufficient_history", "label_fraud_spike"}
    missing = sorted(required - set(featured.columns))
    if missing:
        raise ValueError(f"Featured frame is missing required protocol fields: {missing}")
    usable = featured[featured["has_sufficient_history"]].copy()
    usable["date"] = pd.to_datetime(usable["date"], errors="raise")
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be before validation_end.")

    merchants = np.array(sorted(usable["merchant_id"].unique()))
    if len(merchants) < 15:
        raise ValueError("Merchant generalization requires at least 15 merchants.")
    shuffled = np.random.default_rng(random_state).permutation(merchants)
    train_count = int(np.floor(len(shuffled) * 0.60))
    validation_count = int(np.floor(len(shuffled) * 0.20))
    train_ids = tuple(sorted(map(str, shuffled[:train_count])))
    validation_ids = tuple(sorted(map(str, shuffled[train_count:train_count + validation_count])))
    test_ids = tuple(sorted(map(str, shuffled[train_count + validation_count:])))

    train = usable[(usable["merchant_id"].isin(train_ids)) & (usable["date"] <= train_cutoff)].copy()
    validation = usable[
        (usable["merchant_id"].isin(validation_ids))
        & (usable["date"] > train_cutoff)
        & (usable["date"] <= validation_cutoff)
    ].copy()
    test = usable[(usable["merchant_id"].isin(test_ids)) & (usable["date"] > validation_cutoff)].copy()
    for name, frame in (("merchant-train", train), ("merchant-validation", validation), ("merchant-test", test)):
        _require_binary_labels(frame, name)
    return MerchantGeneralizationSplit(train, validation, test, train_ids, validation_ids, test_ids, random_state)


def merchant_split_metadata(split: MerchantGeneralizationSplit) -> dict[str, Any]:
    """Return serializable proof of cohort and chronological separation."""
    def describe(frame: pd.DataFrame, merchants: tuple[str, ...]) -> dict[str, Any]:
        return {
            "rows": int(len(frame)),
            "merchants": int(len(merchants)),
            "positive_rate": round(float(frame["label_fraud_spike"].mean()), 6),
            "date_range": [str(frame["date"].min().date()), str(frame["date"].max().date())],
        }

    train_ids, validation_ids, test_ids = map(set, (split.train_merchants, split.validation_merchants, split.test_merchants))
    if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
        raise AssertionError("Merchant generalization cohorts overlap.")
    return {
        "random_state": split.random_state,
        "merchant_sets_disjoint": True,
        "train": describe(split.train, split.train_merchants),
        "validation": describe(split.validation, split.validation_merchants),
        "test": describe(split.test, split.test_merchants),
    }


def cold_start_protocol(featured: pd.DataFrame) -> dict[str, Any]:
    """Measure the operational policy for rows without sufficient history."""
    if "has_sufficient_history" not in featured or "label_fraud_spike" not in featured:
        raise ValueError("Cold-start protocol requires feature readiness and offline labels.")
    cold_start = featured[~featured["has_sufficient_history"]]
    if cold_start.empty:
        raise ValueError("Cold-start protocol requires at least one insufficient-history row.")
    return {
        "rows": int(len(cold_start)),
        "merchants": int(cold_start["merchant_id"].nunique()),
        "positive_rate": round(float(cold_start["label_fraud_spike"].mean()), 6),
        "operational_policy": "insufficient_evidence_manual_review",
        "model_scored_rows": 0,
        "model_score_coverage": 0.0,
        "disclosure": (
            "The operational engine intentionally does not score merchant-days without sufficient prior history. "
            "This reports coverage and label prevalence, not an artificial cold-start probability metric. "
            "A zero positive rate is possible because this simulator requires prior observed days before injecting fraud."
        ),
    }


def simulation_stress_configs() -> dict[str, dict[str, Any]]:
    """Fixed, labelled stress configurations; no scenario participates in tuning."""
    return {
        "mild_anomalies": {"seed": 3101, "fraud_severity_min": 0.10, "fraud_severity_max": 0.35},
        "low_prevalence": {"seed": 3102, "fraud_day_rate": 0.025},
        "high_prevalence": {"seed": 3103, "fraud_day_rate": 0.15, "max_fraud_windows_per_merchant": 6},
        "high_missingness": {"seed": 3104, "missing_observation_rate": 0.12},
        "campaign_heavy": {"seed": 3105, "campaign_fraction": 0.80, "quiet_period_fraction": 0.55},
    }


def stress_config_manifest(config: Any) -> dict[str, Any]:
    """Make the exact synthetic stress parameters auditable in JSON reports."""
    return asdict(config)
