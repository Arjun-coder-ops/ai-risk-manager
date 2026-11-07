"""Reproducible classical-model candidates for offline fraud-spike experiments.

This module deliberately separates *ranking baselines* from classifiers that
emit class-one scores.  The heuristic and Isolation Forest are useful
comparators, but their scores are not calibrated fraud probabilities and are
therefore ineligible for production threshold selection or deployment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ScoreKind = Literal["probability", "anomaly_score", "heuristic_score"]


@dataclass(frozen=True)
class CandidateSpec:
    """One auditable experiment candidate and its score semantics."""

    name: str
    description: str
    score_kind: ScoreKind
    eligible_for_production_selection: bool
    factory: Callable[[np.ndarray], Any]


class HeuristicRiskBaseline:
    """Fixed transparent anomaly rule used only as a non-probabilistic baseline."""

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "HeuristicRiskBaseline":
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        def column(name: str) -> np.ndarray:
            return X[name].to_numpy(dtype=float) if name in X else np.zeros(len(X), dtype=float)

        # Each term is intentionally bounded.  This is not trained, calibrated,
        # or suitable for an operational probability threshold.
        volume = np.clip(np.abs(column("txn_volume_zscore")), 0.0, 6.0) / 6.0
        failures = np.clip(column("failed_rate_delta"), 0.0, 1.0)
        amount = np.clip(np.abs(column("amount_zscore")), 0.0, 6.0) / 6.0
        new_users = np.clip(column("new_user_ratio_delta"), 0.0, 1.0)
        regions = np.clip(np.abs(column("region_spread_zscore")), 0.0, 6.0) / 6.0
        momentum = np.clip(column("rolling_volume_trend") - 1.0, 0.0, 4.0) / 4.0
        return (0.28 * volume + 0.25 * failures + 0.12 * amount + 0.15 * new_users + 0.12 * regions + 0.08 * momentum)


class IsolationForestBaseline:
    """Normal-behaviour anomaly detector with an explicit anomaly-score API."""

    def __init__(self, random_state: int) -> None:
        self.pipeline = Pipeline([
            ("scale", StandardScaler()),
            ("model", IsolationForest(
                n_estimators=300,
                contamination="auto",
                random_state=random_state,
                n_jobs=1,
            )),
        ])

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "IsolationForestBaseline":
        normal_rows = X[np.asarray(y) == 0]
        if len(normal_rows) < 2:
            raise ValueError("Isolation Forest requires at least two normal training observations.")
        self.pipeline.fit(normal_rows)
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        # IsolationForest has a larger decision function for normal points.
        # Negating it establishes the shared convention: larger = riskier.
        return -self.pipeline.decision_function(X)


def _xgboost_factory(class_weight: np.ndarray, random_state: int) -> Any:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:  # pragma: no cover - exercised in dependency setup.
        raise RuntimeError("xgboost is required for the classical experiment suite. Install requirements.lock.") from exc
    negatives, positives = class_weight
    return XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        scale_pos_weight=float(negatives / positives),
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=1,
    )


def _lightgbm_factory(class_weight: np.ndarray, random_state: int) -> Any:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as exc:  # pragma: no cover - exercised in dependency setup.
        raise RuntimeError("lightgbm is required for the classical experiment suite. Install requirements.lock.") from exc
    negatives, positives = class_weight
    return LGBMClassifier(
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=25,
        reg_lambda=1.0,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=float(negatives / positives),
        random_state=random_state,
        n_jobs=1,
        verbosity=-1,
    )


def get_classical_candidates(random_state: int = 42) -> tuple[CandidateSpec, ...]:
    """Return the Phase 4.1 candidate set in a stable reporting order."""
    def class_balance(y: np.ndarray) -> np.ndarray:
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if counts[0] == 0 or counts[1] == 0:
            raise ValueError("Supervised candidate training requires both classes.")
        return counts.astype(float)

    return (
        CandidateSpec(
            "heuristic_baseline",
            "Fixed merchant-relative signal rule; ranking baseline only.",
            "heuristic_score",
            False,
            lambda _: HeuristicRiskBaseline(),
        ),
        CandidateSpec(
            "logistic_regression",
            "Standardized, class-balanced linear probability baseline.",
            "probability",
            True,
            lambda _: Pipeline([
                ("scale", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=random_state)),
            ]),
        ),
        CandidateSpec(
            "random_forest",
            "Class-balanced non-linear bagging baseline.",
            "probability",
            True,
            lambda _: RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=1,
            ),
        ),
        CandidateSpec(
            "hist_gradient_boosting",
            "Histogram gradient boosting probability model.",
            "probability",
            True,
            lambda _: HistGradientBoostingClassifier(
                max_depth=4,
                learning_rate=0.08,
                max_iter=200,
                l2_regularization=1.0,
                random_state=random_state,
            ),
        ),
        CandidateSpec(
            "xgboost",
            "XGBoost tree-boosting probability model with train-only class weighting.",
            "probability",
            True,
            lambda y: _xgboost_factory(class_balance(y), random_state),
        ),
        CandidateSpec(
            "lightgbm",
            "LightGBM leaf-wise boosting probability model with train-only class weighting.",
            "probability",
            True,
            lambda y: _lightgbm_factory(class_balance(y), random_state),
        ),
        CandidateSpec(
            "isolation_forest",
            "Normal-only unsupervised anomaly-score baseline; not a probability.",
            "anomaly_score",
            False,
            lambda _: IsolationForestBaseline(random_state=random_state),
        ),
    )


def fit_and_score(candidate: CandidateSpec, X_train: pd.DataFrame, y_train: np.ndarray, X_evaluate: pd.DataFrame) -> tuple[Any, np.ndarray]:
    """Fit one candidate on training data and score a separate evaluation set."""
    model = candidate.factory(np.asarray(y_train))
    if candidate.score_kind == "probability":
        model.fit(X_train, y_train)
        scores = model.predict_proba(X_evaluate)[:, 1]
    elif candidate.score_kind == "anomaly_score":
        model.fit(X_train, y_train)
        scores = model.score(X_evaluate)
    else:
        model.fit(X_train, y_train)
        scores = model.score(X_evaluate)
    return model, np.asarray(scores, dtype=float)
