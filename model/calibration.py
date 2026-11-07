"""Probability-calibration primitives for offline, versioned model candidates.

The active serving artifact remains separate from calibration candidates until a
release decision explicitly promotes one.  This prevents a newly calibrated
score scale from being applied with thresholds chosen for a different scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


CALIBRATION_METHODS = ("sigmoid", "isotonic")


def _inputs(raw_scores, labels=None) -> tuple[np.ndarray, np.ndarray | None]:
    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("raw_scores must be a non-empty one-dimensional array.")
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError("raw_scores must contain finite probabilities in [0, 1].")
    if labels is None:
        return scores, None
    labels = np.asarray(labels, dtype=int)
    if labels.ndim != 1 or len(labels) != len(scores) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("labels must align with raw_scores and contain both binary classes.")
    return scores, labels


@dataclass
class ScoreCalibrator:
    """A fitted sigmoid or isotonic mapping from raw to calibrated probability."""

    method: str
    estimator: object | None = None

    def fit(self, raw_scores, labels) -> "ScoreCalibrator":
        scores, labels = _inputs(raw_scores, labels)
        if self.method == "sigmoid":
            self.estimator = LogisticRegression(C=1_000_000.0, solver="lbfgs", random_state=42).fit(
                scores.reshape(-1, 1), labels
            )
        elif self.method == "isotonic":
            self.estimator = IsotonicRegression(out_of_bounds="clip").fit(scores, labels)
        else:
            raise ValueError(f"Unsupported calibration method: {self.method}")
        return self

    def predict(self, raw_scores) -> np.ndarray:
        scores, _ = _inputs(raw_scores)
        if self.estimator is None:
            raise RuntimeError("Fit the calibrator before predicting.")
        if self.method == "sigmoid":
            calibrated = self.estimator.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.estimator.predict(scores)
        return np.clip(np.asarray(calibrated, dtype=float), 0.0, 1.0)


@dataclass
class CalibratedProbabilityModel:
    """Candidate model wrapper retaining the exact base estimator and mapping."""

    base_model: object
    calibrator: ScoreCalibrator

    def predict_proba(self, features) -> np.ndarray:
        raw = self.base_model.predict_proba(features)[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack((1.0 - calibrated, calibrated))
