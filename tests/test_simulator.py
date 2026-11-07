"""Regression coverage for the offline synthetic-data simulator."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from simulator import SimulationConfig, generate_simulation  # noqa: E402
from validation import validate_observations  # noqa: E402


def scenario_config() -> SimulationConfig:
    """Small but event-rich config that keeps simulator tests fast and stable."""
    return SimulationConfig(
        seed=17,
        n_merchants=50,
        n_days=100,
        min_active_days=35,
        fraud_day_rate=0.08,
        missing_observation_rate=0.12,
        delayed_record_rate=0.30,
        cold_start_fraction=0.55,
        churn_fraction=0.45,
        quiet_period_fraction=0.85,
        campaign_fraction=0.85,
    )


def test_simulation_is_deterministic_and_has_valid_unique_observations():
    first = generate_simulation(scenario_config())
    second = generate_simulation(scenario_config())

    pd.testing.assert_frame_equal(first.observations, second.observations)
    assert first.events == second.events
    validated = validate_observations(first.observations)
    assert not validated.duplicated(["merchant_id", "date"]).any()
    assert set(validated["archetype"]) == {"small", "medium", "large"}
    assert set(validated["label_fraud_spike"].unique()) == {0, 1}


def test_simulation_covers_lifecycle_missing_and_delayed_record_scenarios():
    result = generate_simulation(scenario_config())
    kinds = {event["kind"] for event in result.events}
    assert {"cold_start", "merchant_churn", "quiet_period", "legitimate_campaign", "missing_observation", "delayed_record", "fraud_spike"} <= kinds

    delayed = result.observations[result.observations["record_delay_days"] > 0]
    assert not delayed.empty
    assert (pd.to_datetime(delayed["record_available_date"]) > pd.to_datetime(delayed["date"])).all()

    observed_keys = set(zip(result.observations["merchant_id"], result.observations["date"]))
    missing = next(event for event in result.events if event["kind"] == "missing_observation")
    assert (missing["merchant_id"], missing["date"]) not in observed_keys


def test_only_injected_fraud_windows_receive_positive_ground_truth_labels():
    result = generate_simulation(scenario_config())
    fraud_events = [event for event in result.events if event["kind"] == "fraud_spike"]
    asserted_positive_keys = {
        (event["merchant_id"], event_date)
        for event in fraud_events
        for event_date in event["affected_dates"]
    }
    actual_positive_keys = {
        (row.merchant_id, row.date)
        for row in result.observations[result.observations["label_fraud_spike"] == 1].itertuples()
    }

    assert actual_positive_keys == asserted_positive_keys
    assert {event["fraud_type"] for event in fraud_events} <= {
        "volume_burst", "failed_payment_spike", "new_user_device_burst",
        "device_anomaly", "region_anomaly", "amount_anomaly",
    }
    assert result.manifest()["ground_truth"]["positive_label"].startswith("label_fraud_spike=1")


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_merchants": 0},
        {"n_days": 29},
        {"fraud_day_rate": 0.0},
        {"fraud_severity_min": 0.0},
        {"fraud_severity_min": 0.8, "fraud_severity_max": 0.5},
        {"missing_observation_rate": 1.1},
        {"start_date": "01-01-2026"},
    ],
)
def test_invalid_simulation_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        generate_simulation(SimulationConfig(**overrides))
