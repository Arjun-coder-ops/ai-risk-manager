"""Regression coverage for Phase 6 policy and explanation boundaries."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "model"))

from explanations import (  # noqa: E402
    neural_integrated_gradients,
    normalize_tree_shap_values,
    tree_attribution_signal,
)
from policy import PolicyValidationError, ThreeActionPolicy, evaluate_three_action_policy, load_policy  # noqa: E402


POLICY_PATH = ROOT / "model" / "artifacts" / "policy.json"


def test_active_policy_maps_scores_to_one_exclusive_versioned_action():
    policy = load_policy(POLICY_PATH)
    low, review, high = policy.decisions_for_scores([0.099, 0.10, 0.67])
    assert (low.risk_level, low.action_code) == ("LOW", "CONTINUE_MONITORING")
    assert (review.risk_level, review.action_code) == ("REVIEW", "ANALYST_REVIEW")
    assert (high.risk_level, high.action_code) == ("HIGH", "PRIORITY_INVESTIGATION")
    assert all(decision.policy_version == "three-action-v1" for decision in (low, review, high))


def test_policy_rejects_invalid_threshold_ordering_and_scores():
    policy = load_policy(POLICY_PATH)
    with pytest.raises(PolicyValidationError, match="between 0 and 1"):
        policy.decide(float("nan"))
    payload = policy.to_dict()
    payload["review_threshold"] = 0.8
    payload["high_threshold"] = 0.7
    with pytest.raises(PolicyValidationError, match="review <= high"):
        ThreeActionPolicy.from_dict(payload)


def test_three_action_costs_are_exclusive_and_include_capacity_evidence():
    policy = load_policy(POLICY_PATH)
    result = evaluate_three_action_policy([1, 0, 1], [0.05, 0.2, 0.9], policy)
    assert result["action_counts"] == {
        "LOW": {"total": 1, "positive": 1, "negative": 0},
        "REVIEW": {"total": 1, "positive": 0, "negative": 1},
        "HIGH": {"total": 1, "positive": 1, "negative": 0},
    }
    assert result["cost_components_inr"] == {
        "analyst_review_cost_inr": 150.0,
        "reviewed_true_positive_response_cost_inr": 0.0,
        "reviewed_false_alert_cost_inr": 300.0,
        "high_escalation_cost_inr": 800.0,
        "high_true_positive_response_cost_inr": 500.0,
        "high_false_alert_cost_inr": 0.0,
        "missed_fraud_loss_inr": 8000.0,
    }
    assert result["total_estimated_cost_inr"] == 9750.0
    assert result["review_capacity"]["review_rows"] == 1


def test_shap_direction_is_not_conflated_with_observed_value_direction():
    feature_row = pd.Series({
        "txn_volume_zscore": 2.5,
        "n_transactions": 100,
        "txn_baseline_mean": 50,
    })
    signal = tree_attribution_signal(
        "txn_volume_zscore",
        feature_row,
        -1.2,
        model_version="model-v1",
        feature_schema_version="schema-v1",
    )
    assert signal["observed_value_direction"] == "above_historical_baseline"
    assert signal["attribution_direction"] == "decreases_model_estimated_risk"
    assert signal["observed_value"] == 100.0
    assert signal["baseline_value"] == 50.0
    assert signal["explanation_provenance"]["method"] == "tree_shap"


def test_tree_shap_shape_normalization_handles_class_axis():
    values = np.array([[[1.0, 2.0], [3.0, 4.0]]])
    normalized = normalize_tree_shap_values(values, expected_rows=1, expected_features=2)
    assert normalized.shape == (1, 2)
    assert normalized.tolist() == [[2.0, 4.0]]


def test_integrated_gradients_matches_a_linear_logit_from_zero_baseline():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[2.0, -3.0]]))
    values = np.array([[1.0, 2.0]], dtype=np.float32)
    attribution = neural_integrated_gradients(model, values, baseline=np.zeros_like(values), steps=16)
    assert attribution.shape == (1, 2)
    assert np.allclose(attribution[0], [2.0, -6.0], atol=1e-6)
