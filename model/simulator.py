"""Deterministic synthetic merchant-day simulator.

This module is deliberately an *offline evaluation* tool.  It models
plausible aggregate merchant behaviour and exposes simulator-only ground
truth so that the training pipeline can be exercised.  It is not evidence of
real-world fraud performance, and the event manifest must never be served by
the production-facing APIs.

The implementation has two independent stages:

``generate_merchant_behavior``
    creates lifecycle-aware, seasonal normal behaviour plus legitimate demand
    events, missing observations, and delayed-record metadata;

``inject_fraud_spikes``
    selects observed merchant-day windows and applies fraud-like perturbations
    while assigning the only positive ground-truth label.

Keeping those stages separate makes the label semantics reviewable and avoids
the former generator's accidental double/random amount mutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ARCHETYPES = (
    # name, weight, base transaction mean, transaction std, failure rate,
    # average order value, order-value std, new-user ratio
    ("small", 0.55, 40, 15, 0.04, 650, 300, 0.15),
    ("medium", 0.32, 220, 60, 0.05, 1400, 700, 0.20),
    ("large", 0.13, 900, 220, 0.06, 2600, 1500, 0.25),
)

INDIAN_REGIONS = (
    "Maharashtra", "Karnataka", "Delhi-NCR", "Gujarat", "Tamil Nadu",
    "Telangana", "West Bengal", "Uttar Pradesh", "Rajasthan", "Kerala",
)

FRAUD_TYPES = (
    "volume_burst",
    "failed_payment_spike",
    "new_user_device_burst",
    "device_anomaly",
    "region_anomaly",
    "amount_anomaly",
)


@dataclass(frozen=True)
class SimulationConfig:
    """All defaults needed for a reproducible, reviewable simulation run."""

    seed: int = 42
    n_merchants: int = 150
    n_days: int = 180
    start_date: str = "2026-01-01"
    fraud_day_rate: float = 0.07
    max_fraud_windows_per_merchant: int = 3
    fraud_severity_min: float = 0.35
    fraud_severity_max: float = 1.0
    missing_observation_rate: float = 0.015
    delayed_record_rate: float = 0.05
    max_record_delay_days: int = 5
    cold_start_fraction: float = 0.18
    cold_start_max_day: int = 75
    churn_fraction: float = 0.10
    quiet_period_fraction: float = 0.22
    campaign_fraction: float = 0.20
    min_active_days: int = 60
    holiday_day_offsets: tuple[int, ...] = (26, 75, 126, 160)

    def validate(self) -> None:
        if self.n_merchants < 1:
            raise ValueError("n_merchants must be at least 1.")
        if self.n_days < 30:
            raise ValueError("n_days must be at least 30.")
        if not 0 < self.fraud_day_rate < 0.5:
            raise ValueError("fraud_day_rate must be between 0 and 0.5.")
        if not 0 < self.fraud_severity_min <= self.fraud_severity_max:
            raise ValueError("fraud_severity_min must be positive and no greater than fraud_severity_max.")
        for field in (
            "missing_observation_rate", "delayed_record_rate", "cold_start_fraction",
            "churn_fraction", "quiet_period_fraction", "campaign_fraction",
        ):
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} must be between 0 and 1.")
        if self.max_record_delay_days < 1:
            raise ValueError("max_record_delay_days must be at least 1.")
        if self.max_fraud_windows_per_merchant < 1:
            raise ValueError("max_fraud_windows_per_merchant must be at least 1.")
        if not 7 <= self.min_active_days < self.n_days:
            raise ValueError("min_active_days must be at least 7 and less than n_days.")
        try:
            date.fromisoformat(self.start_date)
        except ValueError as exc:
            raise ValueError("start_date must use YYYY-MM-DD format.") from exc


@dataclass(frozen=True)
class MerchantProfile:
    merchant_id: str
    archetype: str
    base_txn_mean: float
    base_fail_rate: float
    base_aov: float
    aov_std: float
    base_new_user_ratio: float
    home_region: str
    activation_day: int
    deactivation_day: int
    daily_growth_rate: float
    weekly_amplitude: float
    seasonal_amplitude: float
    seasonal_phase: float


@dataclass
class SimulationResult:
    """Offline observations plus auditable generator metadata."""

    observations: pd.DataFrame
    events: list[dict[str, Any]]
    config: SimulationConfig

    def manifest(self) -> dict[str, Any]:
        positive_count = int(self.observations["label_fraud_spike"].sum())
        return {
            "schema_version": "synthetic-simulator-v2",
            "purpose": (
                "Offline synthetic-data provenance only. This manifest and simulator labels "
                "must not be exposed by production-facing APIs."
            ),
            "config": asdict(self.config),
            "ground_truth": {
                "positive_label": "label_fraud_spike=1 only for a row altered by inject_fraud_spikes",
                "negative_lookalikes": (
                    "campaign and holiday demand can alter volume, users, devices, regions, and "
                    "amounts but retain label_fraud_spike=0."
                ),
                "delayed_record_metadata": (
                    "record_available_date describes simulated arrival delay and is not a model feature."
                ),
            },
            "summary": {
                "observed_merchant_days": int(len(self.observations)),
                "merchants_observed": int(self.observations["merchant_id"].nunique()),
                "positive_merchant_days": positive_count,
                "positive_rate": round(float(positive_count / len(self.observations)), 6),
                "delayed_records": int((self.observations["record_delay_days"] > 0).sum()),
            },
            "events": self.events,
        }


def _window_days(rng: np.random.Generator, profile: MerchantProfile, length: int) -> list[int]:
    """Pick a calendar window strictly inside a merchant's active lifetime."""
    latest_start = profile.deactivation_day - length
    if latest_start <= profile.activation_day:
        return []
    start = int(rng.integers(profile.activation_day, latest_start + 1))
    return list(range(start, start + length))


def make_merchants(config: SimulationConfig, rng: np.random.Generator) -> list[MerchantProfile]:
    """Sample merchant-specific scale, lifecycle, and seasonal parameters."""
    names, weights = zip(*[(item[0], item[1]) for item in DEFAULT_ARCHETYPES])
    archetypes = {item[0]: item for item in DEFAULT_ARCHETYPES}
    profiles: list[MerchantProfile] = []
    max_cold_start = min(config.cold_start_max_day, config.n_days - config.min_active_days - 1)
    for index in range(config.n_merchants):
        archetype_name = str(rng.choice(names, p=weights))
        _, _, txn_mean, txn_std, fail_rate, aov, aov_std, new_user_ratio = archetypes[archetype_name]
        activation_day = (
            int(rng.integers(1, max_cold_start + 1))
            if max_cold_start > 0 and rng.random() < config.cold_start_fraction
            else 0
        )
        earliest_churn = activation_day + config.min_active_days
        if rng.random() < config.churn_fraction and earliest_churn < config.n_days:
            deactivation_day = int(rng.integers(earliest_churn, config.n_days))
        else:
            deactivation_day = config.n_days
        profiles.append(MerchantProfile(
            merchant_id=f"M{index:04d}",
            archetype=archetype_name,
            base_txn_mean=max(5.0, float(rng.normal(txn_mean, txn_std * 0.30))),
            base_fail_rate=float(np.clip(rng.normal(fail_rate, 0.01), 0.01, 0.25)),
            base_aov=max(50.0, float(rng.normal(aov, aov_std * 0.30))),
            aov_std=float(aov_std),
            base_new_user_ratio=float(np.clip(rng.normal(new_user_ratio, 0.03), 0.03, 0.60)),
            home_region=str(rng.choice(INDIAN_REGIONS)),
            activation_day=activation_day,
            deactivation_day=deactivation_day,
            # Negative and positive trends are both intentional: a merchant's
            # long-term trajectory must not itself imply fraud.
            daily_growth_rate=float(rng.uniform(-0.0020, 0.0035)),
            weekly_amplitude=float(rng.uniform(0.08, 0.24)),
            seasonal_amplitude=float(rng.uniform(0.04, 0.18)),
            seasonal_phase=float(rng.uniform(0, 2 * math.pi)),
        ))
    return profiles


def _build_behavior_windows(
    profiles: list[MerchantProfile], config: SimulationConfig, rng: np.random.Generator
) -> tuple[dict[str, dict[int, str]], list[dict[str, Any]]]:
    """Schedule legitimate quiet/campaign events before labels are assigned."""
    events_by_day: dict[str, dict[int, str]] = {profile.merchant_id: {} for profile in profiles}
    manifest_events: list[dict[str, Any]] = []
    for profile in profiles:
        if profile.activation_day:
            manifest_events.append({
                "kind": "cold_start", "merchant_id": profile.merchant_id,
                "day": profile.activation_day,
            })
        if profile.deactivation_day < config.n_days:
            manifest_events.append({
                "kind": "merchant_churn", "merchant_id": profile.merchant_id,
                "day": profile.deactivation_day,
            })
        for kind, fraction, min_length, max_length in (
            ("quiet_period", config.quiet_period_fraction, 2, 6),
            ("legitimate_campaign", config.campaign_fraction, 2, 6),
        ):
            if rng.random() >= fraction:
                continue
            length = int(rng.integers(min_length, max_length + 1))
            days = _window_days(rng, profile, length)
            if not days:
                continue
            # A campaign wins over a quiet-period event on the rare overlap;
            # this keeps normal-behaviour causes mutually interpretable.
            for day_index in days:
                previous = events_by_day[profile.merchant_id].get(day_index)
                if previous is None or kind == "legitimate_campaign":
                    events_by_day[profile.merchant_id][day_index] = kind
            manifest_events.append({
                "kind": kind, "merchant_id": profile.merchant_id,
                "start_day": days[0], "duration_days": len(days),
                "label_fraud_spike": 0,
            })
    return events_by_day, manifest_events


def _holiday_factor(day_index: int, config: SimulationConfig) -> float:
    """Smooth, legitimate demand uplift around configurable holiday offsets."""
    factor = 1.0
    for holiday in config.holiday_day_offsets:
        distance = abs(day_index - holiday)
        if distance <= 2:
            factor *= 1.0 + 0.35 * (1.0 - distance / 3.0)
    return factor


def _normal_day(
    profile: MerchantProfile,
    day_index: int,
    day: date,
    config: SimulationConfig,
    rng: np.random.Generator,
    behavior_event: str | None,
) -> dict[str, Any]:
    """Generate one normal (potentially legitimate-event) merchant-day."""
    weekly = 1.0 + profile.weekly_amplitude * math.sin(2 * math.pi * day.weekday() / 7 + profile.seasonal_phase)
    monthly = 1.0 + profile.seasonal_amplitude * math.sin(2 * math.pi * day_index / 30.4 + profile.seasonal_phase)
    trend = max(0.35, 1.0 + profile.daily_growth_rate * day_index)
    txn_mean = profile.base_txn_mean * weekly * monthly * trend * _holiday_factor(day_index, config)
    fail_rate = profile.base_fail_rate
    aov = profile.base_aov * (0.96 + 0.08 * monthly)
    new_user_ratio = profile.base_new_user_ratio
    device_ratio = float(np.clip(rng.normal(0.84, 0.05), 0.55, 1.0))
    region_count = 1

    # These are intentionally normal/legitimate events.  They can resemble a
    # detector's fraud signals but always retain ground-truth label 0.
    if behavior_event == "quiet_period":
        txn_mean *= float(rng.uniform(0.05, 0.35))
        new_user_ratio *= 0.65
    elif behavior_event == "legitimate_campaign":
        txn_mean *= float(rng.uniform(1.8, 3.8))
        new_user_ratio = float(np.clip(new_user_ratio + rng.uniform(0.12, 0.30), 0, 0.90))
        device_ratio = float(np.clip(device_ratio + rng.uniform(0.04, 0.14), 0, 1.0))
        region_count = int(rng.integers(2, 5))
        aov *= float(rng.uniform(0.85, 1.20))

    n_transactions = int(rng.poisson(max(0.05, txn_mean)))
    delay_days = (
        int(rng.integers(1, config.max_record_delay_days + 1))
        if rng.random() < config.delayed_record_rate else 0
    )
    row: dict[str, Any] = {
        "merchant_id": profile.merchant_id,
        "date": day.isoformat(),
        "archetype": profile.archetype,
        "home_region": profile.home_region,
        "record_delay_days": delay_days,
        "record_available_date": (day + timedelta(days=delay_days)).isoformat(),
        "label_fraud_spike": 0,
        "_behavior_event": behavior_event or "baseline",
    }
    if n_transactions == 0:
        row.update({
            "n_transactions": 0, "n_failed": 0, "avg_amount": 0.0, "amount_std": 0.0,
            "n_unique_users": 0, "n_new_users": 0, "n_devices": 0,
            "distinct_regions_seen": 0,
        })
        return row

    n_failed = int(rng.binomial(n_transactions, fail_rate))
    amounts = rng.gamma(shape=2.0, scale=max(10.0, aov / 2.0), size=n_transactions)
    n_unique_users = max(1, int(n_transactions * rng.uniform(0.60, 0.90)))
    n_new_users = int(rng.binomial(n_unique_users, new_user_ratio))
    n_devices = max(1, int(n_unique_users * device_ratio))
    row.update({
        "n_transactions": n_transactions,
        "n_failed": n_failed,
        "avg_amount": round(float(np.mean(amounts)), 2),
        "amount_std": round(float(np.std(amounts)), 2),
        "n_unique_users": n_unique_users,
        "n_new_users": n_new_users,
        "n_devices": n_devices,
        "distinct_regions_seen": min(n_transactions, region_count),
    })
    return row


def generate_merchant_behavior(
    config: SimulationConfig, rng: np.random.Generator
) -> tuple[pd.DataFrame, list[MerchantProfile], list[dict[str, Any]]]:
    """Generate unlabeled behaviour, delivery metadata, and lifecycle events."""
    profiles = make_merchants(config, rng)
    scheduled_events, manifest_events = _build_behavior_windows(profiles, config, rng)
    start = date.fromisoformat(config.start_date)
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for day_index in range(profile.activation_day, profile.deactivation_day):
            day = start + timedelta(days=day_index)
            if rng.random() < config.missing_observation_rate:
                manifest_events.append({
                    "kind": "missing_observation", "merchant_id": profile.merchant_id,
                    "day": day_index, "date": day.isoformat(),
                })
                continue
            event = scheduled_events[profile.merchant_id].get(day_index)
            row = _normal_day(profile, day_index, day, config, rng, event)
            rows.append(row)
            if row["record_delay_days"]:
                manifest_events.append({
                    "kind": "delayed_record", "merchant_id": profile.merchant_id,
                    "day": day_index, "date": day.isoformat(),
                    "record_available_date": row["record_available_date"],
                    "delay_days": row["record_delay_days"],
                })
    observations = pd.DataFrame(rows).sort_values(["merchant_id", "date"]).reset_index(drop=True)
    return observations, profiles, manifest_events


def _bounded_counts(row: pd.Series) -> None:
    """Normalize a mutated row to the merchant-day integrity contract."""
    transactions = max(0, int(round(float(row["n_transactions"]))))
    if transactions == 0:
        for field in ("n_failed", "n_unique_users", "n_new_users", "n_devices", "distinct_regions_seen"):
            row[field] = 0
        row["avg_amount"] = 0.0
        row["amount_std"] = 0.0
        return
    row["n_transactions"] = transactions
    row["n_failed"] = min(transactions, max(0, int(round(float(row["n_failed"])))) )
    users = min(transactions, max(1, int(round(float(row["n_unique_users"])))) )
    row["n_unique_users"] = users
    row["n_new_users"] = min(users, max(0, int(round(float(row["n_new_users"])))) )
    row["n_devices"] = min(transactions, max(1, int(round(float(row["n_devices"])))) )
    row["distinct_regions_seen"] = min(transactions, max(1, int(round(float(row["distinct_regions_seen"])))) )
    row["avg_amount"] = round(max(0.0, float(row["avg_amount"])), 2)
    row["amount_std"] = round(max(0.0, float(row["amount_std"])), 2)


def _scale_volume(row: pd.Series, factor: float) -> None:
    """Scale counts while retaining the row's normal composition ratios."""
    old_transactions = max(1, int(row["n_transactions"]))
    new_transactions = max(1, int(round(old_transactions * factor)))
    for field in ("n_failed", "n_unique_users", "n_new_users", "n_devices", "distinct_regions_seen"):
        row[field] = float(row[field]) * new_transactions / old_transactions
    row["n_transactions"] = new_transactions


def _apply_fraud_type(row: pd.Series, fraud_type: str, severity: float, rng: np.random.Generator) -> None:
    """Apply one fraud-like perturbation; this is the only label-positive path."""
    if int(row["n_transactions"]) == 0:
        row["n_transactions"] = 1
        row["n_unique_users"] = 1
        row["n_devices"] = 1
        row["distinct_regions_seen"] = 1
        row["avg_amount"] = max(20.0, float(row["avg_amount"]))
    if fraud_type == "volume_burst":
        _scale_volume(row, 1.0 + severity * float(rng.uniform(2.0, 5.0)))
    elif fraud_type == "failed_payment_spike":
        _scale_volume(row, 1.0 + severity * float(rng.uniform(0.25, 1.00)))
        failure_rate = min(0.95, float(row["n_failed"]) / max(1, int(row["n_transactions"])) + severity * rng.uniform(0.18, 0.55))
        row["n_failed"] = round(float(row["n_transactions"]) * failure_rate)
    elif fraud_type == "new_user_device_burst":
        _scale_volume(row, 1.0 + severity * float(rng.uniform(0.8, 2.5)))
        row["n_new_users"] = float(row["n_unique_users"]) * min(0.95, 0.35 + severity * rng.uniform(0.35, 0.65))
        row["n_devices"] = float(row["n_unique_users"]) * min(1.0, 0.82 + severity * 0.25)
    elif fraud_type == "device_anomaly":
        # Device reuse is deliberately directional: many identities appear on
        # too few devices, a stronger aggregate bot/credential-stuffing cue.
        row["n_devices"] = float(row["n_unique_users"]) * max(0.08, 0.65 - severity * rng.uniform(0.25, 0.50))
        row["n_new_users"] = float(row["n_unique_users"]) * min(0.92, 0.20 + severity * 0.45)
    elif fraud_type == "region_anomaly":
        row["distinct_regions_seen"] = float(row["distinct_regions_seen"]) + 1 + round(severity * rng.integers(3, 9))
        _scale_volume(row, 1.0 + severity * float(rng.uniform(0.25, 1.2)))
    elif fraud_type == "amount_anomaly":
        # One signed, bounded shift corrects the old double-random mutation.
        direction = -1.0 if rng.random() < 0.5 else 1.0
        amount_factor = max(0.15, 1.0 + direction * severity * float(rng.uniform(0.55, 1.45)))
        row["avg_amount"] = float(row["avg_amount"]) * amount_factor
        row["amount_std"] = max(1.0, float(row["amount_std"]) * (1.0 + severity * rng.uniform(0.40, 1.80)))
    else:  # pragma: no cover - protects future callers, not random sampling.
        raise ValueError(f"Unsupported fraud type: {fraud_type}")
    _bounded_counts(row)


def inject_fraud_spikes(
    observations: pd.DataFrame,
    profiles: list[MerchantProfile],
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Inject non-overlapping fraud windows into already-observed behaviour."""
    output = observations.copy()
    output["label_fraud_spike"] = 0
    target_positive_days = int(round(len(output) * config.fraud_day_rate))
    positive_rows: set[int] = set()
    windows_per_merchant = {profile.merchant_id: 0 for profile in profiles}
    # Campaigns are deliberately label-negative fraud lookalikes.  Do not
    # overwrite them with a fraud label; this preserves the stated simulator
    # contract even when the random schedules would otherwise overlap.
    by_merchant = {
        merchant_id: group.loc[group["_behavior_event"] != "legitimate_campaign"].index.to_list()
        for merchant_id, group in output.groupby("merchant_id", sort=False)
    }
    manifest_events: list[dict[str, Any]] = []
    merchant_ids = list(by_merchant)
    attempts = 0
    max_attempts = max(500, config.n_merchants * config.max_fraud_windows_per_merchant * 30)

    while len(positive_rows) < target_positive_days and attempts < max_attempts:
        attempts += 1
        merchant_id = str(rng.choice(merchant_ids))
        indices = by_merchant[merchant_id]
        if windows_per_merchant[merchant_id] >= config.max_fraud_windows_per_merchant:
            continue
        max_duration = min(7, len(indices) - 7)
        if max_duration < 2:
            continue
        duration = int(rng.integers(2, max_duration + 1))
        # Reserve an observed history before the attack window so the primary
        # in-distribution training task has a meaningful merchant baseline.
        start_position = int(rng.integers(7, len(indices) - duration + 1))
        selected = indices[start_position:start_position + duration]
        if len(selected) != duration or any(index in positive_rows for index in selected):
            continue
        fraud_type = str(rng.choice(FRAUD_TYPES))
        severity = float(rng.uniform(config.fraud_severity_min, config.fraud_severity_max))
        for index in selected:
            mutated = output.loc[index].copy()
            _apply_fraud_type(mutated, fraud_type, severity, rng)
            output.loc[index, mutated.index] = mutated
            output.loc[index, "label_fraud_spike"] = 1
            positive_rows.add(index)
        windows_per_merchant[merchant_id] += 1
        manifest_events.append({
            "kind": "fraud_spike", "merchant_id": merchant_id,
            "fraud_type": fraud_type, "severity": round(severity, 4),
            "start_date": output.loc[selected[0], "date"],
            "affected_dates": output.loc[selected, "date"].to_list(),
            "duration_observed_days": duration,
            "label_fraud_spike": 1,
        })

    if len(positive_rows) < target_positive_days:
        raise RuntimeError("Unable to schedule enough non-overlapping fraud windows for this configuration.")
    return (
        output.drop(columns=["_behavior_event"]).sort_values(["merchant_id", "date"]).reset_index(drop=True),
        manifest_events,
    )


def generate_simulation(config: SimulationConfig | None = None) -> SimulationResult:
    """Generate a deterministic simulation result from a validated config."""
    config = config or SimulationConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    observations, profiles, behavior_events = generate_merchant_behavior(config, rng)
    observations, fraud_events = inject_fraud_spikes(observations, profiles, config, rng)
    return SimulationResult(observations=observations, events=behavior_events + fraud_events, config=config)
