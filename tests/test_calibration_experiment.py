"""Regression tests for the Phase 4.4 chronological calibration experiment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "model"))

from calibration import CalibratedProbabilityModel, ScoreCalibrator  # noqa: E402
from calibration_protocol import calibration_partitions, reliability_diagram_svg  # noqa: E402


def test_sigmoid_and_isotonic_calibrators_return_bounded_probabilities():
    labels = np.array([0, 0, 0, 1, 1, 1])
    raw_scores = np.array([0.02, 0.15, 0.40, 0.45, 0.72, 0.95])
    for method in ("sigmoid", "isotonic"):
        calibrator = ScoreCalibrator(method).fit(raw_scores, labels)
        calibrated = calibrator.predict(np.array([0.0, 0.5, 1.0]))
        assert calibrated.shape == (3,)
        assert np.all((0.0 <= calibrated) & (calibrated <= 1.0))


def test_candidate_model_calibrates_only_the_positive_probability_column():
    class BaseModel:
        def predict_proba(self, features):
            return np.column_stack((np.array([0.8, 0.2]), np.array([0.2, 0.8])))

    calibrator = ScoreCalibrator("isotonic").fit(
        np.array([0.1, 0.3, 0.7, 0.9]), np.array([0, 0, 1, 1])
    )
    result = CalibratedProbabilityModel(BaseModel(), calibrator).predict_proba(pd.DataFrame({"x": [1, 2]}))
    assert result.shape == (2, 2)
    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.all((0.0 <= result[:, 1]) & (result[:, 1] <= 1.0))


def test_partitioning_purges_entire_fraud_event_that_crosses_fit_calibration_boundary():
    frame = pd.DataFrame({
        "merchant_id": ["A", "A", "A", "B", "B", "B", "B"],
        "date": ["2026-03-14", "2026-03-15", "2026-03-16", "2026-03-14", "2026-03-15", "2026-03-16", "2026-03-17"],
        "has_sufficient_history": [True] * 7,
    })
    events = [{
        "kind": "fraud_spike",
        "merchant_id": "A",
        "affected_dates": ["2026-03-14", "2026-03-15", "2026-03-16"],
    }]
    partitions = calibration_partitions(
        frame,
        events,
        fit_end="2026-03-15",
        calibration_fit_end="2026-03-16",
        calibration_method_selection_end="2026-03-17",
    )
    assert partitions.purged_event_count == 1
    assert partitions.purged_rows == 3
    assert set(partitions.fit["merchant_id"]) == {"B"}
    assert set(partitions.calibration_fit["merchant_id"]) == {"B"}
    assert set(partitions.calibration_selection["merchant_id"]) == {"B"}


def test_reliability_svg_contains_series_and_escapes_title():
    svg = reliability_diagram_svg(
        {
            "raw": {"brier_score": 0.1, "expected_calibration_error": 0.02, "bins": [{"mean_score": 0.2, "observed_positive_rate": 0.3}]},
            "isotonic": {"brier_score": 0.08, "expected_calibration_error": 0.01, "bins": [{"mean_score": 0.2, "observed_positive_rate": 0.2}]},
        },
        title="Reliability <report>",
    )
    assert "<svg" in svg
    assert "Reliability &lt;report&gt;" in svg
    assert "raw: Brier" in svg
    assert "isotonic: Brier" in svg
