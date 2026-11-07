"""Tests for Phase 4.2 offline generalization protocol boundaries."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "model"))

from protocols import cold_start_protocol, merchant_generalization_split, merchant_split_metadata, simulation_stress_configs  # noqa: E402
from split import TRAIN_END, VAL_END  # noqa: E402


def featured_frame() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "processed" / "features.csv", parse_dates=["date"])


def test_merchant_protocol_has_disjoint_cohorts_and_time_boundaries():
    split = merchant_generalization_split(featured_frame(), train_end=TRAIN_END, validation_end=VAL_END)
    metadata = merchant_split_metadata(split)
    assert metadata["random_state"] == 42
    assert metadata["merchant_sets_disjoint"] is True
    assert set(split.train_merchants).isdisjoint(split.validation_merchants)
    assert set(split.train_merchants).isdisjoint(split.test_merchants)
    assert set(split.validation_merchants).isdisjoint(split.test_merchants)
    assert split.train["date"].max() <= pd.Timestamp(TRAIN_END)
    assert split.validation["date"].min() > pd.Timestamp(TRAIN_END)
    assert split.validation["date"].max() <= pd.Timestamp(VAL_END)
    assert split.test["date"].min() > pd.Timestamp(VAL_END)
    assert split.train["label_fraud_spike"].nunique() == 2
    assert split.validation["label_fraud_spike"].nunique() == 2
    assert split.test["label_fraud_spike"].nunique() == 2


def test_cold_start_protocol_reports_intentional_no_score_policy():
    report = cold_start_protocol(featured_frame())
    assert report["rows"] > 0
    assert report["merchants"] > 0
    assert report["model_scored_rows"] == 0
    assert report["model_score_coverage"] == 0.0
    assert report["operational_policy"] == "insufficient_evidence_manual_review"


def test_stress_matrix_covers_all_required_change_axes():
    configs = simulation_stress_configs()
    assert set(configs) == {
        "mild_anomalies", "low_prevalence", "high_prevalence", "high_missingness", "campaign_heavy"
    }
    assert configs["mild_anomalies"]["fraud_severity_max"] < 1.0
    assert configs["low_prevalence"]["fraud_day_rate"] < configs["high_prevalence"]["fraud_day_rate"]
    assert configs["high_missingness"]["missing_observation_rate"] > 0.015
    assert configs["campaign_heavy"]["campaign_fraction"] > 0.2
