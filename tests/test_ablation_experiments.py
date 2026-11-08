"""Regression tests for the Phase 4.6 ablation protocol."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "model"))

from ablations import (  # noqa: E402
    ablation_partitions,
    global_feature_frame,
)


def test_ablation_partition_purges_events_crossing_any_experiment_boundary():
    frame = pd.DataFrame({
        "merchant_id": ["A", "A", "A", "B", "B", "B", "B", "B", "B", "B"],
        "date": ["2026-02-28", "2026-03-01", "2026-03-08", "2026-02-27", "2026-02-28", "2026-03-01", "2026-03-02", "2026-03-08", "2026-03-09", "2026-03-10"],
        "has_sufficient_history": [True] * 10,
        "label_fraud_spike": [1, 1, 0, 0, 1, 0, 1, 0, 1, 0],
    })
    events = [{
        "kind": "fraud_spike",
        "merchant_id": "A",
        "affected_dates": ["2026-02-28", "2026-03-01"],
    }]
    partitions = ablation_partitions(
        frame,
        events,
        train_end="2026-02-28",
        calibration_end="2026-03-07",
        evaluation_end="2026-03-10",
    )
    assert partitions.purged_event_count == 1
    assert partitions.purged_rows == 2
    assert set(partitions.train["merchant_id"]) == {"B"}
    assert set(partitions.calibration["merchant_id"]) == {"B"}
    assert set(partitions.evaluation["merchant_id"]) == {"A", "B"}


def test_global_feature_baseline_has_no_merchant_history_columns_or_nan_ratios():
    frame = pd.DataFrame({
        "n_transactions": [10, 0],
        "n_failed": [2, 0],
        "avg_amount": [100.0, 0.0],
        "amount_std": [20.0, 0.0],
        "n_unique_users": [5, 0],
        "n_new_users": [3, 0],
        "n_devices": [4, 0],
        "distinct_regions_seen": [2, 0],
    })
    global_features = global_feature_frame(frame)
    assert global_features.columns.tolist() == [
        "global_n_transactions", "global_failed_rate", "global_avg_amount", "global_amount_std",
        "global_new_user_ratio", "global_device_per_user", "global_distinct_regions_seen",
    ]
    assert global_features.isna().sum().sum() == 0
    assert global_features.loc[0, "global_failed_rate"] == 0.2
    assert global_features.loc[1, "global_device_per_user"] == 1.0
