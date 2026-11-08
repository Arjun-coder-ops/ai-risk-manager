"""Leakage-conscious partitions and feature views for Phase 4.6 ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


ABLATION_TRAIN_END = "2026-02-28"
ABLATION_CALIBRATION_END = "2026-03-07"
ABLATION_EVALUATION_END = "2026-03-15"

REDUCED_FEATURE_COLUMNS = (
    "txn_volume_zscore",
    "failed_rate_delta",
    "amount_zscore",
    "amount_volatility_ratio",
    "new_user_ratio_delta",
    "device_diversity_ratio",
    "region_spread_zscore",
    "rolling_volume_trend",
)

GLOBAL_FEATURE_COLUMNS = (
    "global_n_transactions",
    "global_failed_rate",
    "global_avg_amount",
    "global_amount_std",
    "global_new_user_ratio",
    "global_device_per_user",
    "global_distinct_regions_seen",
)


@dataclass(frozen=True)
class AblationPartitions:
    train: pd.DataFrame
    calibration: pd.DataFrame
    evaluation: pd.DataFrame
    purged_event_count: int
    purged_rows: int


def ablation_partitions(
    featured: pd.DataFrame,
    events: Iterable[dict[str, Any]],
    *,
    train_end: str = ABLATION_TRAIN_END,
    calibration_end: str = ABLATION_CALIBRATION_END,
    evaluation_end: str = ABLATION_EVALUATION_END,
) -> AblationPartitions:
    """Build train/calibration/evaluation roles and purge crossing fraud events."""
    if not (pd.Timestamp(train_end) < pd.Timestamp(calibration_end) < pd.Timestamp(evaluation_end)):
        raise ValueError("Ablation boundaries must be strictly increasing.")
    required = {"merchant_id", "date", "has_sufficient_history"}
    missing = sorted(required - set(featured.columns))
    if missing:
        raise ValueError(f"Feature frame lacks required fields: {missing}")
    usable = featured[featured["has_sufficient_history"]].copy()
    usable["date"] = pd.to_datetime(usable["date"], errors="raise").dt.strftime("%Y-%m-%d")
    boundaries = tuple(str(pd.Timestamp(boundary).date()) for boundary in (train_end, calibration_end))
    purge_keys: set[tuple[str, str]] = set()
    purged_event_count = 0
    for event in events:
        if event.get("kind") != "fraud_spike":
            continue
        dates = sorted(str(value) for value in event.get("affected_dates", []))
        if not dates or not any(dates[0] <= boundary < dates[-1] for boundary in boundaries):
            continue
        purged_event_count += 1
        purge_keys.update((str(event["merchant_id"]), date_value) for date_value in dates)
    keep = [key not in purge_keys for key in zip(usable["merchant_id"].astype(str), usable["date"])]
    cleaned = usable.loc[keep].copy()
    train_end, calibration_end, evaluation_end = boundaries[0], boundaries[1], str(pd.Timestamp(evaluation_end).date())
    train = cleaned[cleaned["date"] <= train_end].copy()
    calibration = cleaned[(cleaned["date"] > train_end) & (cleaned["date"] <= calibration_end)].copy()
    evaluation = cleaned[(cleaned["date"] > calibration_end) & (cleaned["date"] <= evaluation_end)].copy()
    for role, frame in (("train", train), ("calibration", calibration), ("evaluation", evaluation)):
        if frame.empty or frame["label_fraud_spike"].nunique() != 2:
            raise ValueError(f"Ablation {role} role must contain both classes.")
    return AblationPartitions(
        train=train,
        calibration=calibration,
        evaluation=evaluation,
        purged_event_count=purged_event_count,
        purged_rows=len(usable) - len(cleaned),
    )


def global_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw/global feature baseline with no merchant-specific historical reference."""
    required = {
        "n_transactions", "n_failed", "avg_amount", "amount_std", "n_unique_users",
        "n_new_users", "n_devices", "distinct_regions_seen",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Global ablation lacks raw fields: {missing}")
    transactions = frame["n_transactions"].to_numpy(dtype=float)
    users = frame["n_unique_users"].to_numpy(dtype=float)
    def ratio(numerator: str, denominator: np.ndarray, neutral: float = 0.0) -> np.ndarray:
        output = np.full(len(frame), neutral, dtype=float)
        np.divide(frame[numerator].to_numpy(dtype=float), denominator, out=output, where=denominator > 0)
        return output
    return pd.DataFrame({
        "global_n_transactions": transactions,
        "global_failed_rate": ratio("n_failed", transactions),
        "global_avg_amount": frame["avg_amount"].to_numpy(dtype=float),
        "global_amount_std": frame["amount_std"].to_numpy(dtype=float),
        "global_new_user_ratio": ratio("n_new_users", users),
        "global_device_per_user": ratio("n_devices", users, neutral=1.0),
        "global_distinct_regions_seen": frame["distinct_regions_seen"].to_numpy(dtype=float),
    }, index=frame.index)
