"""Unit tests for the Phase 4.1 candidate-model experiment contract."""

from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model"))

from experiments import fit_and_score, get_classical_candidates  # noqa: E402
from features import FEATURE_COLUMNS  # noqa: E402


def feature_frame(rows: int = 40) -> pd.DataFrame:
    values = np.linspace(-2.0, 3.0, rows)
    return pd.DataFrame({column: values + offset for offset, column in enumerate(FEATURE_COLUMNS)})


def test_required_candidate_set_and_score_semantics_are_explicit():
    candidates = {candidate.name: candidate for candidate in get_classical_candidates()}
    assert set(candidates) == {
        "heuristic_baseline",
        "logistic_regression",
        "random_forest",
        "hist_gradient_boosting",
        "xgboost",
        "lightgbm",
        "isolation_forest",
    }
    assert candidates["heuristic_baseline"].eligible_for_production_selection is False
    assert candidates["isolation_forest"].score_kind == "anomaly_score"
    assert candidates["xgboost"].score_kind == "probability"


@pytest.mark.parametrize("candidate_name", ["xgboost", "lightgbm"])
def test_boosting_factories_honor_the_experiment_random_state(candidate_name):
    package = "xgboost" if candidate_name == "xgboost" else "lightgbm"
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is installed through requirements.lock in the isolated project environment.")
    candidate = next(item for item in get_classical_candidates(random_state=19) if item.name == candidate_name)
    model = candidate.factory(np.array([0] * 30 + [1] * 10))
    assert model.get_params()["random_state"] == 19


def test_heuristic_baseline_is_deterministic_and_responds_to_stronger_signals():
    candidate = next(item for item in get_classical_candidates() if item.name == "heuristic_baseline")
    X = feature_frame(2)
    X.loc[0, FEATURE_COLUMNS] = 0.0
    X.loc[1, FEATURE_COLUMNS] = 0.0
    X.loc[1, ["txn_volume_zscore", "failed_rate_delta", "amount_zscore", "new_user_ratio_delta", "region_spread_zscore"]] = [6.0, 1.0, 6.0, 1.0, 6.0]
    _, scores_a = fit_and_score(candidate, X, np.array([0, 1]), X)
    _, scores_b = fit_and_score(candidate, X, np.array([0, 1]), X)
    assert np.array_equal(scores_a, scores_b)
    assert scores_a[1] > scores_a[0]
    assert np.all((scores_a >= 0) & (scores_a <= 1))


def test_isolation_forest_trains_only_on_normal_rows_and_returns_finite_ranking_scores():
    candidate = next(item for item in get_classical_candidates() if item.name == "isolation_forest")
    X = feature_frame()
    y = np.array([0] * 30 + [1] * 10)
    _, scores = fit_and_score(candidate, X, y, X.iloc[-5:])
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("candidate_name", ["xgboost", "lightgbm"])
def test_optional_boosting_candidates_fit_when_locked_dependencies_are_installed(candidate_name):
    package = "xgboost" if candidate_name == "xgboost" else "lightgbm"
    if importlib.util.find_spec(package) is None:
        pytest.skip(f"{package} is installed through requirements.lock in the isolated project environment.")
    candidate = next(item for item in get_classical_candidates() if item.name == candidate_name)
    X = feature_frame(50)
    y = np.array([0] * 35 + [1] * 15)
    _, scores = fit_and_score(candidate, X, y, X.iloc[-5:])
    assert scores.shape == (5,)
    assert np.all((scores >= 0) & (scores <= 1))
