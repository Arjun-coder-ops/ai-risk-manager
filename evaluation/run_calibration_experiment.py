"""Evaluate sigmoid and isotonic calibration on a dedicated chronological slice."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any

import joblib
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from calibration import CALIBRATION_METHODS, CalibratedProbabilityModel, ScoreCalibrator  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402
from train import get_candidates  # noqa: E402

from calibration_protocol import (  # noqa: E402
    CALIBRATION_FIT_END,
    CALIBRATION_METHOD_SELECTION_END,
    MODEL_FIT_END,
    calibration_partitions,
    reliability_diagram_svg,
)
from metrics import calibration_metrics, ranking_metrics  # noqa: E402


FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "raw", "simulation_manifest.json")
DEFAULT_REPORT = os.path.join(REPO_ROOT, "evaluation", "calibration_report.json")
DEFAULT_DIAGRAM = os.path.join(REPO_ROOT, "evaluation", "calibration_reliability.svg")
DEFAULT_CANDIDATE = os.path.join(REPO_ROOT, "model", "artifacts", "calibrated_candidate.joblib")
DEFAULT_CANDIDATE_META = os.path.join(REPO_ROOT, "model", "artifacts", "calibrated_candidate_meta.json")


def _events() -> list[dict[str, Any]]:
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        return json.load(handle).get("events", [])


def _winner(candidate_reports: dict[str, dict[str, Any]]) -> str:
    """Select strictly by calibration-set Brier, then ECE and a fixed name tie-break."""
    return min(
        candidate_reports,
        key=lambda name: (
            candidate_reports[name]["calibration"]["brier_score"],
            candidate_reports[name]["calibration"]["expected_calibration_error"],
            name,
        ),
    )


def run(
    report_path: str = DEFAULT_REPORT,
    diagram_path: str = DEFAULT_DIAGRAM,
    candidate_path: str = DEFAULT_CANDIDATE,
    candidate_meta_path: str = DEFAULT_CANDIDATE_META,
) -> dict[str, Any]:
    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    partitions = calibration_partitions(featured, _events())
    fit = partitions.fit
    calibration_fit = partitions.calibration_fit
    calibration_selection = partitions.calibration_selection
    base_model = get_candidates()["gradient_boosting"].fit(
        fit[FEATURE_COLUMNS], fit["label_fraud_spike"]
    )
    calibration_fit_labels = calibration_fit["label_fraud_spike"].to_numpy(dtype=int)
    calibration_fit_scores = base_model.predict_proba(calibration_fit[FEATURE_COLUMNS])[:, 1]
    labels = calibration_selection["label_fraud_spike"].to_numpy(dtype=int)
    raw_scores = base_model.predict_proba(calibration_selection[FEATURE_COLUMNS])[:, 1]
    fitted = {
        method: ScoreCalibrator(method).fit(calibration_fit_scores, calibration_fit_labels)
        for method in CALIBRATION_METHODS
    }
    score_sets = {
        "raw": raw_scores,
        **{method: calibrator.predict(raw_scores) for method, calibrator in fitted.items()},
    }
    candidate_reports = {
        name: {
            "calibration": calibration_metrics(
                labels,
                scores,
                score_semantics=(
                    "uncalibrated_model_probability"
                    if name == "raw"
                    else f"{name}_calibration_adjusted_probability"
                ),
            ),
            "ranking": ranking_metrics(labels, scores),
        }
        for name, scores in score_sets.items()
    }
    calibration_winner = _winner(candidate_reports)
    selected_method = calibration_winner if calibration_winner != "raw" else None
    report = {
        "report_kind": "phase-4.4-calibration-experiment",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "base_model": "hist_gradient_boosting",
        "partitions": {
            "fit": {
                "date_range": [str(fit.date.min()), str(fit.date.max())],
                "rows": len(fit),
                "positive_rate": round(float(fit.label_fraud_spike.mean()), 6),
            },
            "calibration_fit": {
                "date_range": [str(calibration_fit.date.min()), str(calibration_fit.date.max())],
                "rows": len(calibration_fit),
                "positive_rate": round(float(calibration_fit.label_fraud_spike.mean()), 6),
            },
            "calibration_method_selection": {
                "date_range": [str(calibration_selection.date.min()), str(calibration_selection.date.max())],
                "rows": len(calibration_selection),
                "positive_rate": round(float(calibration_selection.label_fraud_spike.mean()), 6),
            },
            "fit_end": MODEL_FIT_END,
            "calibration_fit_end": CALIBRATION_FIT_END,
            "calibration_method_selection_end": CALIBRATION_METHOD_SELECTION_END,
            "purged_boundary_spanning_fraud_events": partitions.purged_event_count,
            "purged_rows": partitions.purged_rows,
            "selection_validation_period": ["2026-04-16", "2026-05-15"],
            "final_test_period": ["2026-05-16", "2026-06-29"],
        },
        "selection_rule": (
            "A calibrated method must strictly beat raw on untouched calibration-method-selection Brier score; "
            "ECE then method name are deterministic tie-breaks among equal-Brier calibrated candidates."
        ),
        "candidates": candidate_reports,
        "selected_method": selected_method,
        "retained_candidate_artifact": None,
        "candidate_status": "no_calibrated_candidate_retained",
        "serving_boundary": (
            "The active serving model and thresholds remain unchanged. No calibrated score scale is available for "
            "serving because neither calibration method beat raw scores on the untouched calibration-selection period."
        ),
        "probability_disclosure": (
            "No adjusted probability estimate is retained. The active raw score is not a reliable real-world fraud probability."
        ),
    }
    for path in (report_path, diagram_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with open(diagram_path, "w", encoding="utf-8") as handle:
        handle.write(reliability_diagram_svg(
            {name: data["calibration"] for name, data in candidate_reports.items()},
            title="Calibration-method-selection reliability: raw vs sigmoid vs isotonic",
        ))
    if selected_method is not None:
        retained_calibrator = ScoreCalibrator(selected_method).fit(
            list(calibration_fit_scores) + list(raw_scores),
            list(calibration_fit_labels) + list(labels),
        )
        candidate = CalibratedProbabilityModel(base_model=base_model, calibrator=retained_calibrator)
        joblib.dump(candidate, candidate_path)
        with open(candidate_meta_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "status": "evaluation_candidate_not_deployed",
                    "selected_method": selected_method,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION,
                    "source_report": os.path.relpath(report_path, REPO_ROOT).replace("\\", "/"),
                    "serving_boundary": "Candidate is not deployed pending independent threshold, robustness, and release-gate evidence.",
                },
                handle,
                indent=2,
            )
            handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Phase 4.4 chronological calibration experiment.")
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--diagram", default=DEFAULT_DIAGRAM)
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--candidate-meta", default=DEFAULT_CANDIDATE_META)
    args = parser.parse_args()
    print(json.dumps(run(args.report, args.diagram, args.candidate, args.candidate_meta), indent=2))


if __name__ == "__main__":
    main()
