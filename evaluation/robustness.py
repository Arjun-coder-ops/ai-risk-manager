"""Deterministic, offline data perturbations for Phase 4.5 robustness evidence."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from simulator import SimulationConfig
from validation import validate_observations


def robustness_simulation_configs() -> dict[str, dict[str, Any]]:
    """Independent configurations for axes that use the in-distribution simulator."""
    return {
        "unseen_high_severity": {
            "seed": 4501,
            "fraud_severity_min": 1.05,
            "fraud_severity_max": 1.60,
            "max_fraud_windows_per_merchant": 5,
        },
        "merchant_size_and_campaigns": {
            "seed": 4502,
            "campaign_fraction": 1.0,
            "quiet_period_fraction": 0.0,
            "max_fraud_windows_per_merchant": 5,
        },
        "missing_records": {
            "seed": 4503,
            "missing_observation_rate": 0.20,
            "max_fraud_windows_per_merchant": 5,
        },
        "delayed_records": {
            "seed": 4504,
            "delayed_record_rate": 0.45,
            "max_record_delay_days": 7,
            "max_fraud_windows_per_merchant": 5,
        },
        "low_prevalence": {
            "seed": 4505,
            "fraud_day_rate": 0.025,
            "max_fraud_windows_per_merchant": 4,
        },
        "high_prevalence": {
            "seed": 4506,
            "fraud_day_rate": 0.15,
            "max_fraud_windows_per_merchant": 8,
        },
        "temporal_degradation": {
            "seed": 4507,
            "n_days": 360,
            "fraud_day_rate": 0.035,
            "max_fraud_windows_per_merchant": 7,
            "start_date": "2028-01-01",
        },
    }


def availability_on_event_day(observations: pd.DataFrame) -> pd.DataFrame:
    """Keep only records available by their merchant-day's end-of-day scoring time."""
    required = {"date", "record_available_date"}
    missing = sorted(required - set(observations.columns))
    if missing:
        raise ValueError(f"Delayed-record simulation lacks fields: {missing}")
    available = observations[
        pd.to_datetime(observations["record_available_date"], errors="raise")
        <= pd.to_datetime(observations["date"], errors="raise")
    ].copy()
    return available.sort_values(["merchant_id", "date"]).reset_index(drop=True)


def campaign_keys(events: list[dict[str, Any]], start_date: str) -> set[tuple[str, str]]:
    """Return offline-only campaign merchant-day keys from simulator provenance."""
    start = pd.Timestamp(start_date)
    keys: set[tuple[str, str]] = set()
    for event in events:
        if event.get("kind") != "legitimate_campaign":
            continue
        merchant_id = str(event["merchant_id"])
        for day in range(int(event["start_day"]), int(event["start_day"]) + int(event["duration_days"])):
            keys.add((merchant_id, str((start + timedelta(days=day)).date())))
    return keys


def symmetric_label_noise(labels, *, flip_probability: float, random_state: int) -> tuple[np.ndarray, int]:
    """Create evaluation-only noisy labels while keeping latent labels unchanged elsewhere."""
    if not 0 < flip_probability < 0.5:
        raise ValueError("flip_probability must be between zero and 0.5.")
    output = np.asarray(labels, dtype=int).copy()
    if output.ndim != 1 or set(np.unique(output)) != {0, 1}:
        raise ValueError("labels must be a binary vector containing both classes.")
    flips = np.random.default_rng(random_state).random(len(output)) < flip_probability
    output[flips] = 1 - output[flips]
    return output, int(flips.sum())


def apply_gradual_temporal_shift(
    observations: pd.DataFrame,
    *,
    shift_start_day: int = 150,
    daily_volume_growth: float = 0.003,
    daily_amount_growth: float = 0.0015,
) -> pd.DataFrame:
    """Add a label-negative global product-mix drift after a calendar boundary.

    Counts preserve the original row's ratios and integrity constraints. The
    mutation deliberately affects fraud and non-fraud rows alike: it measures
    sensitivity to a changing observation distribution, not a new label rule.
    """
    if shift_start_day < 1 or daily_volume_growth < 0 or daily_amount_growth < 0:
        raise ValueError("Temporal-shift parameters must be non-negative with a positive start day.")
    output = observations.copy()
    dates = pd.to_datetime(output["date"], errors="raise")
    day_index = (dates - dates.min()).dt.days.to_numpy()
    elapsed = np.maximum(day_index - shift_start_day, 0)
    volume_factor = 1.0 + elapsed * daily_volume_growth
    amount_factor = 1.0 + elapsed * daily_amount_growth
    old_transactions = output["n_transactions"].to_numpy(dtype=int)
    new_transactions = np.maximum(0, np.rint(old_transactions * volume_factor).astype(int))
    for column, upper in (
        ("n_failed", new_transactions),
        ("n_unique_users", new_transactions),
        ("n_devices", new_transactions),
        ("distinct_regions_seen", new_transactions),
    ):
        old = output[column].to_numpy(dtype=float)
        scaled_float = np.zeros(len(output), dtype=float)
        np.divide(old * new_transactions, old_transactions, out=scaled_float, where=old_transactions > 0)
        scaled = np.rint(scaled_float).astype(int)
        output[column] = np.minimum(np.maximum(scaled, 0), upper)
    old_new_users = output["n_new_users"].to_numpy(dtype=float)
    scaled_new_users_float = np.zeros(len(output), dtype=float)
    np.divide(
        old_new_users * new_transactions,
        old_transactions,
        out=scaled_new_users_float,
        where=old_transactions > 0,
    )
    scaled_new_users = np.rint(scaled_new_users_float).astype(int)
    output["n_new_users"] = np.minimum(np.maximum(scaled_new_users, 0), output["n_unique_users"])
    output["n_transactions"] = new_transactions
    output["avg_amount"] = np.where(new_transactions > 0, output["avg_amount"].to_numpy(dtype=float) * amount_factor, 0.0)
    output["amount_std"] = np.where(new_transactions > 0, output["amount_std"].to_numpy(dtype=float) * amount_factor, 0.0)
    return validate_observations(output).sort_values(["merchant_id", "date"]).reset_index(drop=True)
