"""Versioned, three-action decision policy for merchant-day risk scores.

The model estimates risk; this module decides what the organization should do
with that estimate.  It deliberately contains no model inference, feedback,
or case-state mutation so all four concerns remain independently auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class PolicyValidationError(ValueError):
    """Raised when a policy artifact is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class ActionCostAssumptions:
    """Illustrative, configurable retrospective costs for exclusive actions.

    These values are not Razorpay pricing, loss rates, or authorization for
    automated merchant action.  They make each action's trade-off explicit
    while real operational assumptions are unavailable.
    """

    analyst_review_cost_inr: float
    reviewed_true_positive_response_cost_inr: float
    reviewed_false_alert_cost_inr: float
    high_escalation_cost_inr: float
    high_true_positive_response_cost_inr: float
    high_false_alert_cost_inr: float
    missed_fraud_loss_inr: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionCostAssumptions":
        expected = {
            "analyst_review_cost_inr",
            "reviewed_true_positive_response_cost_inr",
            "reviewed_false_alert_cost_inr",
            "high_escalation_cost_inr",
            "high_true_positive_response_cost_inr",
            "high_false_alert_cost_inr",
            "missed_fraud_loss_inr",
        }
        missing = sorted(expected - set(payload))
        if missing:
            raise PolicyValidationError(f"Policy cost assumptions are missing fields: {missing}")
        values: dict[str, float] = {}
        for field in expected:
            try:
                value = float(payload[field])
            except (TypeError, ValueError) as exc:
                raise PolicyValidationError(f"Policy cost {field} must be numeric.") from exc
            if not np.isfinite(value) or value < 0:
                raise PolicyValidationError(f"Policy cost {field} must be finite and non-negative.")
            values[field] = value
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "analyst_review_cost_inr": self.analyst_review_cost_inr,
            "reviewed_true_positive_response_cost_inr": self.reviewed_true_positive_response_cost_inr,
            "reviewed_false_alert_cost_inr": self.reviewed_false_alert_cost_inr,
            "high_escalation_cost_inr": self.high_escalation_cost_inr,
            "high_true_positive_response_cost_inr": self.high_true_positive_response_cost_inr,
            "high_false_alert_cost_inr": self.high_false_alert_cost_inr,
            "missed_fraud_loss_inr": self.missed_fraud_loss_inr,
            "disclosure": (
                "Illustrative configurable prototype assumptions, not real merchant, analyst, "
                "or fraud-loss financial data."
            ),
        }


@dataclass(frozen=True)
class PolicyDecision:
    """A policy outcome; it is not a statement of human investigation outcome."""

    risk_level: str
    action_code: str
    action_description: str
    policy_version: str
    threshold_band: str

    def to_dict(self) -> dict[str, str]:
        return {
            "risk_level": self.risk_level,
            "action_code": self.action_code,
            "action_description": self.action_description,
            "policy_version": self.policy_version,
            "threshold_band": self.threshold_band,
        }


@dataclass(frozen=True)
class ThreeActionPolicy:
    """Frozen configuration that maps a score to one exclusive action."""

    policy_version: str
    model_version: str
    review_threshold: float
    high_threshold: float
    review_capacity_per_1000_merchant_days: int
    risk_tolerance: str
    actions: dict[str, dict[str, str]]
    costs: ActionCostAssumptions

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ThreeActionPolicy":
        required = {
            "policy_version",
            "model_version",
            "review_threshold",
            "high_threshold",
            "review_capacity_per_1000_merchant_days",
            "risk_tolerance",
            "actions",
            "cost_assumptions",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise PolicyValidationError(f"Policy artifact is missing fields: {missing}")
        try:
            review_threshold = float(payload["review_threshold"])
            high_threshold = float(payload["high_threshold"])
        except (TypeError, ValueError) as exc:
            raise PolicyValidationError("Policy thresholds must be numeric.") from exc
        if not (np.isfinite(review_threshold) and np.isfinite(high_threshold)):
            raise PolicyValidationError("Policy thresholds must be finite.")
        if not 0 <= review_threshold <= high_threshold <= 1:
            raise PolicyValidationError("Policy thresholds must satisfy 0 <= review <= high <= 1.")
        capacity = payload["review_capacity_per_1000_merchant_days"]
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
            raise PolicyValidationError("review_capacity_per_1000_merchant_days must be a non-negative integer.")
        actions = payload["actions"]
        if not isinstance(actions, dict) or set(actions) != {"LOW", "REVIEW", "HIGH"}:
            raise PolicyValidationError("Policy actions must define exactly LOW, REVIEW, and HIGH.")
        normalized_actions: dict[str, dict[str, str]] = {}
        for level, action in actions.items():
            if not isinstance(action, dict) or not all(isinstance(action.get(field), str) and action[field] for field in ("code", "description")):
                raise PolicyValidationError(f"Policy action {level} requires non-empty code and description.")
            normalized_actions[level] = {"code": action["code"], "description": action["description"]}
        policy_version = str(payload["policy_version"])
        model_version = str(payload["model_version"])
        risk_tolerance = str(payload["risk_tolerance"])
        if not policy_version or not model_version or not risk_tolerance:
            raise PolicyValidationError("Policy version, model version, and risk tolerance must be non-empty.")
        return cls(
            policy_version=policy_version,
            model_version=model_version,
            review_threshold=review_threshold,
            high_threshold=high_threshold,
            review_capacity_per_1000_merchant_days=capacity,
            risk_tolerance=risk_tolerance,
            actions=normalized_actions,
            costs=ActionCostAssumptions.from_dict(payload["cost_assumptions"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "model_version": self.model_version,
            "review_threshold": self.review_threshold,
            "high_threshold": self.high_threshold,
            "review_capacity_per_1000_merchant_days": self.review_capacity_per_1000_merchant_days,
            "risk_tolerance": self.risk_tolerance,
            "actions": self.actions,
            "cost_assumptions": self.costs.to_dict(),
        }

    def decide(self, risk_score: float) -> PolicyDecision:
        score = float(risk_score)
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise PolicyValidationError("Risk score must be a finite value between 0 and 1.")
        if score >= self.high_threshold:
            level, band = "HIGH", f"score >= {self.high_threshold:.2f}"
        elif score >= self.review_threshold:
            level, band = "REVIEW", f"{self.review_threshold:.2f} <= score < {self.high_threshold:.2f}"
        else:
            level, band = "LOW", f"score < {self.review_threshold:.2f}"
        action = self.actions[level]
        return PolicyDecision(
            risk_level=level,
            action_code=action["code"],
            action_description=action["description"],
            policy_version=self.policy_version,
            threshold_band=band,
        )

    def decisions_for_scores(self, scores: Iterable[float]) -> list[PolicyDecision]:
        return [self.decide(float(score)) for score in scores]


def load_policy(path: str | Path) -> ThreeActionPolicy:
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise PolicyValidationError(f"Policy artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise PolicyValidationError("Policy artifact is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise PolicyValidationError("Policy artifact must be a JSON object.")
    return ThreeActionPolicy.from_dict(payload)


def evaluate_three_action_policy(
    labels: Iterable[int], scores: Iterable[float], policy: ThreeActionPolicy
) -> dict[str, Any]:
    """Evaluate mutually exclusive LOW, REVIEW, HIGH actions offline.

    Labels are only appropriate for offline evaluation.  At runtime, policy
    decisions do not contain or infer a fraud outcome.
    """
    y = np.asarray(list(labels), dtype=int)
    score_values = np.asarray(list(scores), dtype=float)
    if y.ndim != 1 or score_values.ndim != 1 or len(y) != len(score_values) or not len(y):
        raise ValueError("Labels and scores must be non-empty one-dimensional arrays of equal length.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Policy evaluation labels must be binary 0/1 values.")
    decisions = policy.decisions_for_scores(score_values)
    action_levels = np.asarray([decision.risk_level for decision in decisions])

    counts = {}
    for level in ("LOW", "REVIEW", "HIGH"):
        selected = action_levels == level
        counts[level] = {
            "total": int(selected.sum()),
            "positive": int(((y == 1) & selected).sum()),
            "negative": int(((y == 0) & selected).sum()),
        }

    review_count = counts["REVIEW"]
    high_count = counts["HIGH"]
    low_count = counts["LOW"]
    cost_components = {
        "analyst_review_cost_inr": review_count["total"] * policy.costs.analyst_review_cost_inr,
        "reviewed_true_positive_response_cost_inr": review_count["positive"] * policy.costs.reviewed_true_positive_response_cost_inr,
        "reviewed_false_alert_cost_inr": review_count["negative"] * policy.costs.reviewed_false_alert_cost_inr,
        "high_escalation_cost_inr": high_count["total"] * policy.costs.high_escalation_cost_inr,
        "high_true_positive_response_cost_inr": high_count["positive"] * policy.costs.high_true_positive_response_cost_inr,
        "high_false_alert_cost_inr": high_count["negative"] * policy.costs.high_false_alert_cost_inr,
        "missed_fraud_loss_inr": low_count["positive"] * policy.costs.missed_fraud_loss_inr,
    }
    total_cost = sum(cost_components.values())
    review_capacity_rows = int(np.floor(len(y) * policy.review_capacity_per_1000_merchant_days / 1000))
    return {
        "policy_version": policy.policy_version,
        "model_version": policy.model_version,
        "rows": int(len(y)),
        "positive_rate": round(float(y.mean()), 6),
        "action_counts": counts,
        "review_capacity": {
            "configured_per_1000_merchant_days": policy.review_capacity_per_1000_merchant_days,
            "capacity_rows": review_capacity_rows,
            "review_rows": review_count["total"],
            "within_capacity": review_count["total"] <= review_capacity_rows,
        },
        "alert_volume": {"high_rows": high_count["total"], "review_or_high_rows": review_count["total"] + high_count["total"]},
        "cost_assumptions": policy.costs.to_dict(),
        "cost_components_inr": {key: round(value, 2) for key, value in cost_components.items()},
        "total_estimated_cost_inr": round(total_cost, 2),
        "disclosure": (
            "Offline synthetic-label evaluation under illustrative assumptions. It does not establish real-world "
            "financial impact or change the active policy."
        ),
    }
