"""Regression coverage for the isolated synthetic OOD benchmark."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "evaluation"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from ood_benchmark import OODConfig, generate_ood_observations, run_ood_benchmark  # noqa: E402
from validation import validate_observations  # noqa: E402


def compact_config() -> OODConfig:
    return OODConfig(seed=29, n_merchants=20, n_days=60, fraud_day_rate=0.08)


def test_ood_generation_is_deterministic_and_obeys_observation_contract():
    first = generate_ood_observations(compact_config())
    second = generate_ood_observations(compact_config())

    pd.testing.assert_frame_equal(first, second)
    validated = validate_observations(first)
    assert not validated.duplicated(["merchant_id", "date"]).any()
    assert set(validated["label_fraud_spike"].unique()) == {0, 1}
    assert validated["merchant_id"].str.startswith("OOD").all()


def test_ood_benchmark_scores_frozen_model_without_claiming_external_validation():
    report = run_ood_benchmark(compact_config())

    assert report["benchmark_name"] == "independent-synthetic-ood-v1"
    assert report["provenance"]["real_world_validation"] is False
    assert report["provenance"]["training_data_used"] is False
    assert report["raw_merchant_days"] == 1200
    assert 0 < report["positive_rate"] < 1
    assert 0 <= report["roc_auc"] <= 1
    assert 0 <= report["pr_auc"] <= 1
    assert set(report["review_tier"]["confusion_matrix"]) == {"tp", "fp", "tn", "fn"}


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_merchants": 4},
        {"n_days": 39},
        {"fraud_day_rate": 0.0},
        {"max_attack_windows_per_merchant": 0},
        {"start_date": "2027/01/01"},
    ],
)
def test_invalid_ood_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        generate_ood_observations(OODConfig(**overrides))
