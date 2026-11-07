"""Tests for Phase 4.5 synthetic robustness perturbations."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "model"))

from robustness import (  # noqa: E402
    apply_gradual_temporal_shift,
    availability_on_event_day,
    campaign_keys,
    robustness_simulation_configs,
    symmetric_label_noise,
)


def observation_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "merchant_id": ["A", "A", "A", "A"],
        "date": ["2028-01-01", "2028-01-02", "2028-01-03", "2028-01-04"],
        "record_available_date": ["2028-01-01", "2028-01-04", "2028-01-03", "2028-01-04"],
        "n_transactions": [10, 20, 0, 30],
        "n_failed": [1, 2, 0, 3],
        "avg_amount": [100.0, 150.0, 0.0, 200.0],
        "amount_std": [25.0, 35.0, 0.0, 45.0],
        "n_unique_users": [8, 15, 0, 20],
        "n_new_users": [2, 4, 0, 5],
        "n_devices": [7, 12, 0, 16],
        "distinct_regions_seen": [1, 2, 0, 3],
        "label_fraud_spike": [0, 1, 0, 1],
    })


def test_robustness_config_matrix_covers_required_simulator_axes():
    configs = robustness_simulation_configs()
    assert {"unseen_high_severity", "missing_records", "delayed_records", "low_prevalence", "high_prevalence", "temporal_degradation"} <= set(configs)
    assert configs["unseen_high_severity"]["fraud_severity_min"] > 1.0
    assert configs["missing_records"]["missing_observation_rate"] > 0.015
    assert configs["delayed_records"]["delayed_record_rate"] > 0.05
    assert configs["low_prevalence"]["fraud_day_rate"] < configs["high_prevalence"]["fraud_day_rate"]


def test_delayed_availability_keeps_only_same_day_records():
    available = availability_on_event_day(observation_frame())
    assert available["date"].tolist() == ["2028-01-01", "2028-01-03", "2028-01-04"]


def test_campaign_keys_expand_provenance_window_to_merchant_days():
    keys = campaign_keys(
        [{"kind": "legitimate_campaign", "merchant_id": "A", "start_day": 2, "duration_days": 3}],
        "2028-01-01",
    )
    assert keys == {("A", "2028-01-03"), ("A", "2028-01-04"), ("A", "2028-01-05")}


def test_symmetric_noise_is_reproducible_and_temporal_drift_preserves_integrity():
    labels = np.array([0, 1] * 50)
    noisy_one, count_one = symmetric_label_noise(labels, flip_probability=0.1, random_state=9)
    noisy_two, count_two = symmetric_label_noise(labels, flip_probability=0.1, random_state=9)
    assert count_one == count_two
    assert np.array_equal(noisy_one, noisy_two)
    shifted = apply_gradual_temporal_shift(observation_frame(), shift_start_day=1, daily_volume_growth=0.2)
    assert shifted.loc[0, "n_transactions"] == 10
    assert shifted.loc[3, "n_transactions"] > 30
    assert shifted.loc[2, "n_transactions"] == 0
    assert (shifted["n_new_users"] <= shifted["n_unique_users"]).all()
