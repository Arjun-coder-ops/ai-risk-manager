"""Explanation contracts shared by serving and offline evaluation.

An attribution answers how a model input changed the model estimate.  It does
not answer whether a raw observed value rose or fell versus a merchant's
history.  This module keeps those two statements separate in every record.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


EXPLANATION_SCHEMA_VERSION = "explanation-v1"

# Where available, expose the actual end-of-day observation and its strictly
# prior baseline.  The fallback reference is the semantic neutral point of the
# engineered model feature, not a fabricated historical baseline.
FEATURE_CONTEXT: dict[str, dict[str, Any]] = {
    "txn_volume_zscore": {"observed": "n_transactions", "baseline": "txn_baseline_mean", "neutral": 0.0, "label": "transaction volume"},
    "failed_rate_delta": {"observed": "failed_rate", "baseline": "fail_baseline_mean", "neutral": 0.0, "label": "failed-payment rate"},
    "amount_zscore": {"observed": "avg_amount", "baseline": "amt_baseline_mean", "neutral": 0.0, "label": "average transaction amount"},
    "amount_volatility_ratio": {"observed": "amount_std", "baseline": "amtstd_baseline_mean", "neutral": 1.0, "label": "transaction-amount variability"},
    "new_user_ratio_delta": {"observed": "new_user_ratio", "baseline": "newuser_baseline_mean", "neutral": 0.0, "label": "new-user ratio"},
    "device_diversity_ratio": {"observed": "device_per_user", "baseline": "device_baseline_mean", "neutral": 1.0, "label": "devices per user"},
    "region_spread_zscore": {"observed": "distinct_regions_seen", "baseline": "region_baseline_mean", "neutral": 0.0, "label": "distinct regions"},
    "rolling_volume_trend": {"neutral": 1.0, "label": "prior short-versus-long volume trend"},
    "weekday_volume_ratio": {"observed": "n_transactions", "neutral": 1.0, "label": "same-weekday volume ratio"},
    "txn_volume_percentile": {"observed": "n_transactions", "neutral": 0.5, "label": "prior-volume percentile"},
    "ewm_volume_zscore": {"observed": "n_transactions", "neutral": 0.0, "label": "exponentially weighted volume deviation"},
    "recent_volume_volatility_ratio": {"neutral": 1.0, "label": "prior short-versus-long volume variability"},
    "txn_acceleration_ratio": {"observed": "n_transactions", "neutral": 1.0, "label": "volume acceleration"},
    "trend_adjusted_volume_zscore": {"observed": "n_transactions", "neutral": 0.0, "label": "trend-adjusted volume deviation"},
    "volume_change_point_score": {"neutral": 0.0, "label": "prior volume change-point score"},
    "device_diversity_delta": {"observed": "device_per_user", "baseline": "device_baseline_mean", "neutral": 0.0, "label": "devices-per-user change"},
    "historical_anomaly_rate": {"neutral": 0.0, "label": "prior anomaly frequency"},
}


def _finite_value(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _rounded(value: float | None, digits: int = 4) -> float | None:
    return round(value, digits) if value is not None else None


def observed_value_context(feature: str, feature_row: pd.Series) -> dict[str, Any]:
    """Describe an observed value versus a prior baseline without using SHAP."""
    spec = FEATURE_CONTEXT.get(feature, {"neutral": 0.0, "label": feature})
    feature_value = _finite_value(feature_row.get(feature))
    observed_value = _finite_value(feature_row.get(spec.get("observed"))) if spec.get("observed") else None
    baseline_value = _finite_value(feature_row.get(spec.get("baseline"))) if spec.get("baseline") else None
    neutral = float(spec["neutral"])

    if observed_value is not None and baseline_value is not None:
        if np.isclose(observed_value, baseline_value, rtol=0, atol=1e-9):
            observed_direction = "at_historical_baseline"
        elif observed_value > baseline_value:
            observed_direction = "above_historical_baseline"
        else:
            observed_direction = "below_historical_baseline"
        reference_kind = "strictly_prior_merchant_baseline"
    elif feature_value is not None:
        if np.isclose(feature_value, neutral, rtol=0, atol=1e-9):
            observed_direction = "at_feature_neutral_reference"
        elif feature_value > neutral:
            observed_direction = "above_feature_neutral_reference"
        else:
            observed_direction = "below_feature_neutral_reference"
        reference_kind = "engineered_feature_neutral_reference"
    else:
        observed_direction = "unavailable"
        reference_kind = "unavailable"

    return {
        "feature_label": spec["label"],
        "feature_value": _rounded(feature_value),
        "feature_neutral_reference": _rounded(neutral),
        "observed_value": _rounded(observed_value),
        "baseline_value": _rounded(baseline_value),
        "observed_value_direction": observed_direction,
        "reference_kind": reference_kind,
    }


def attribution_direction(value: float) -> str:
    if value > 0:
        return "increases_model_estimated_risk"
    if value < 0:
        return "decreases_model_estimated_risk"
    return "no_model_estimated_risk_change"


def attribution_impact(value: float) -> str:
    magnitude = abs(value)
    direction = attribution_direction(value)
    if direction == "no_model_estimated_risk_change":
        return "No material model-estimated risk change"
    strength = "Strongly" if magnitude > 2.0 else "Moderately" if magnitude > 0.5 else "Slightly"
    noun = "increased" if value > 0 else "reduced"
    return f"{strength} {noun} model-estimated risk"


def tree_attribution_signal(
    feature: str,
    feature_row: pd.Series,
    shap_value: float,
    *,
    model_version: str,
    feature_schema_version: str,
) -> dict[str, Any]:
    """Build one SHAP record with non-conflated observed/attribution directions."""
    shap_numeric = float(shap_value)
    context = observed_value_context(feature, feature_row)
    return {
        "feature": feature,
        "value": context["feature_value"],  # backwards-compatible alias for the model input value
        **context,
        "attribution_value": _rounded(shap_numeric),
        "shap_value": _rounded(shap_numeric),  # retained for existing clients
        "attribution_direction": attribution_direction(shap_numeric),
        "impact": attribution_impact(shap_numeric),
        "description": (
            f"Observed {context['feature_label']} is {context['observed_value_direction']}; "
            f"its SHAP attribution {attribution_direction(shap_numeric).replace('_', ' ')}."
        ),
        "explanation_provenance": {
            "schema_version": EXPLANATION_SCHEMA_VERSION,
            "method": "tree_shap",
            "attribution_space": "model_raw_margin_log_odds",
            "model_version": model_version,
            "feature_schema_version": feature_schema_version,
            "is_model_attribution": True,
        },
    }


def heuristic_deviation_signal(
    feature: str,
    feature_row: pd.Series,
    *,
    model_version: str,
    feature_schema_version: str,
) -> dict[str, Any]:
    """Build an honest non-attribution fallback record."""
    context = observed_value_context(feature, feature_row)
    return {
        "feature": feature,
        "value": context["feature_value"],
        **context,
        "attribution_value": None,
        "attribution_direction": None,
        "impact": None,
        "description": (
            f"Observed {context['feature_label']} is {context['observed_value_direction']}. "
            "This is a feature-deviation fallback, not a model attribution."
        ),
        "explanation_provenance": {
            "schema_version": EXPLANATION_SCHEMA_VERSION,
            "method": "heuristic_deviation_fallback",
            "attribution_space": None,
            "model_version": model_version,
            "feature_schema_version": feature_schema_version,
            "is_model_attribution": False,
        },
    }


def normalize_tree_shap_values(values: Any, *, expected_rows: int, expected_features: int) -> np.ndarray:
    """Normalize SHAP return conventions to ``(rows, features)`` for class 1."""
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        if array.shape[0] == expected_rows and array.shape[1] == expected_features:
            array = array[:, :, -1]
        elif array.shape[0] >= 2:
            array = array[-1]
    if array.shape != (expected_rows, expected_features):
        raise ValueError(
            f"Unexpected SHAP shape {array.shape}; expected ({expected_rows}, {expected_features})."
        )
    if not np.isfinite(array).all():
        raise ValueError("SHAP values must be finite.")
    return array


def neural_integrated_gradients(
    model: Any,
    values: np.ndarray,
    *,
    baseline: np.ndarray | None = None,
    steps: int = 32,
) -> np.ndarray:
    """Compute a deterministic integrated-gradients attribution for a torch model.

    The model must return one logit per sample.  The caller supplies a
    meaningful baseline (Phase 5 uses the zero vector in train-normalized
    feature space), so the baseline's semantics remain visible in evidence.
    """
    if steps < 2:
        raise ValueError("Integrated gradients requires at least two interpolation steps.")
    import torch

    inputs = torch.as_tensor(np.asarray(values, dtype=np.float32))
    if inputs.ndim < 2:
        raise ValueError("Integrated gradients expects a batch of feature vectors or sequences.")
    reference = torch.zeros_like(inputs) if baseline is None else torch.as_tensor(np.asarray(baseline, dtype=np.float32))
    if reference.shape != inputs.shape:
        raise ValueError("Integrated-gradients baseline must match input shape.")
    was_training = bool(model.training)
    model.eval()
    total_gradients = torch.zeros_like(inputs)
    try:
        for alpha in torch.linspace(0.0, 1.0, steps, dtype=inputs.dtype):
            interpolated = (reference + alpha * (inputs - reference)).detach().requires_grad_(True)
            logits = model(interpolated)
            gradients = torch.autograd.grad(logits.sum(), interpolated)[0]
            total_gradients += gradients.detach()
    finally:
        model.train(was_training)
    return ((inputs - reference) * (total_gradients / steps)).detach().cpu().numpy().astype(float)


def top_neural_attributions(
    attributions: np.ndarray, feature_columns: Iterable[str], *, top_k: int = 3
) -> list[dict[str, Any]]:
    """Summarize one vector of neural attributions without implying probability units."""
    vector = np.asarray(attributions, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Neural attribution summary requires one feature vector.")
    features = list(feature_columns)
    if len(vector) != len(features):
        raise ValueError("Feature columns do not match neural attribution length.")
    order = np.argsort(np.abs(vector))[::-1][:top_k]
    return [
        {
            "feature": features[index],
            "attribution_value": _rounded(float(vector[index])),
            "attribution_direction": attribution_direction(float(vector[index])),
            "attribution_space": "integrated_gradients_logit",
        }
        for index in order
    ]
