"""Run Phase 4.5 robustness tests against the frozen serving model only."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

try:  # Supports CLI execution and package-style imports.
    from evaluation.ood_benchmark import run_ood_benchmark  # noqa: E402
except ModuleNotFoundError:  # pragma: no cover - exercised by CLI invocation.
    from ood_benchmark import run_ood_benchmark  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION, build_features  # noqa: E402
from simulator import SimulationConfig, generate_simulation  # noqa: E402

from metrics import decision_metrics, ranking_metrics  # noqa: E402
from robustness import (  # noqa: E402
    apply_gradual_temporal_shift,
    availability_on_event_day,
    campaign_keys,
    robustness_simulation_configs,
    symmetric_label_noise,
)


DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "robustness_tests.json")
MODEL_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model.joblib")
META_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model_meta.json")


def _score_observations(model: Any, observations: pd.DataFrame, metadata: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    featured = build_features(observations)
    scored = featured[featured["has_sufficient_history"]].copy()
    labels = scored["label_fraud_spike"].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise RuntimeError("Robustness scenario must retain both synthetic label classes after scoring eligibility.")
    scored["score"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    scores = scored["score"].to_numpy(dtype=float)
    return {
        "raw_merchant_days": int(len(observations)),
        "scored_merchant_days": int(len(scored)),
        "scored_positive_rate": round(float(labels.mean()), 6),
        "ranking": ranking_metrics(labels, scores),
        "review_tier": decision_metrics(labels, scores, float(metadata["review_threshold"])),
        "high_tier": decision_metrics(labels, scores, float(metadata["high_threshold"])),
    }, scored


def _metric_slice(scored: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    labels = scored["label_fraud_spike"].to_numpy(dtype=int)
    scores = scored["score"].to_numpy(dtype=float)
    if set(np.unique(labels)) != {0, 1}:
        return {
            "rows": int(len(scored)),
            "positive_rate": round(float(labels.mean()), 6) if len(labels) else None,
            "metric_status": "not_reported_single_class",
        }
    return {
        "rows": int(len(scored)),
        "positive_rate": round(float(labels.mean()), 6),
        "ranking": ranking_metrics(labels, scores),
        "review_tier": decision_metrics(labels, scores, float(metadata["review_threshold"])),
        "high_tier": decision_metrics(labels, scores, float(metadata["high_threshold"])),
    }


def _negative_campaign_alerts(scored: pd.DataFrame, keys: set[tuple[str, str]], metadata: dict[str, Any]) -> dict[str, Any]:
    candidate = scored[
        [
            (str(row.merchant_id), str(pd.Timestamp(row.date).date())) in keys
            for row in scored[["merchant_id", "date"]].itertuples(index=False)
        ]
    ].copy()
    if candidate.empty:
        raise RuntimeError("Campaign scenario produced no scoreable campaign merchant-days.")
    if candidate["label_fraud_spike"].any():
        raise RuntimeError("Campaign contract violated: legitimate campaign rows must remain label-negative.")
    return {
        "scoreable_negative_campaign_days": int(len(candidate)),
        "review_false_alert_rate": round(float((candidate["score"] >= float(metadata["review_threshold"])).mean()), 6),
        "high_false_alert_rate": round(float((candidate["score"] >= float(metadata["high_threshold"])).mean()), 6),
        "review_false_alerts_per_1000_campaign_days": round(
            float((candidate["score"] >= float(metadata["review_threshold"])).mean() * 1000), 6
        ),
        "high_false_alerts_per_1000_campaign_days": round(
            float((candidate["score"] >= float(metadata["high_threshold"])).mean() * 1000), 6
        ),
    }


def _temporal_windows(scored: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    dates = pd.to_datetime(scored["date"], errors="raise")
    start = dates.min()
    offset = ((dates - start).dt.days // 90).clip(upper=3)
    output: dict[str, Any] = {}
    for period in range(4):
        frame = scored[offset == period]
        output[f"days_{period * 90 + 1}_to_{(period + 1) * 90}"] = {
            "date_range": [str(pd.Timestamp(frame["date"].min()).date()), str(pd.Timestamp(frame["date"].max()).date())],
            **_metric_slice(frame, metadata),
        }
    return output


def _simulation_config(name: str) -> SimulationConfig:
    defaults = SimulationConfig(n_merchants=100, n_days=180)
    return replace(defaults, **robustness_simulation_configs()[name])


def run(output_path: str = DEFAULT_OUTPUT) -> dict[str, Any]:
    with open(META_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Frozen model artifact is incompatible with the current feature contract.")
    model = joblib.load(MODEL_PATH)

    severity_config = _simulation_config("unseen_high_severity")
    severity_result = generate_simulation(severity_config)
    severity_report, _ = _score_observations(model, severity_result.observations, metadata)

    size_campaign_config = _simulation_config("merchant_size_and_campaigns")
    size_campaign_result = generate_simulation(size_campaign_config)
    size_campaign_report, size_campaign_scored = _score_observations(model, size_campaign_result.observations, metadata)
    size_slices = {
        str(archetype): _metric_slice(group, metadata)
        for archetype, group in size_campaign_scored.groupby("archetype", sort=True)
    }

    missing_config = _simulation_config("missing_records")
    missing_result = generate_simulation(missing_config)
    missing_report, _ = _score_observations(model, missing_result.observations, metadata)

    delayed_config = _simulation_config("delayed_records")
    delayed_result = generate_simulation(delayed_config)
    timely = availability_on_event_day(delayed_result.observations)
    delayed_report, _ = _score_observations(model, timely, metadata)

    low_config = _simulation_config("low_prevalence")
    high_config = _simulation_config("high_prevalence")
    low_result = generate_simulation(low_config)
    high_result = generate_simulation(high_config)
    low_report, _ = _score_observations(model, low_result.observations, metadata)
    high_report, _ = _score_observations(model, high_result.observations, metadata)

    noise_labels, flipped_rows = symmetric_label_noise(
        size_campaign_scored["label_fraud_spike"].to_numpy(dtype=int), flip_probability=0.10, random_state=4508
    )
    latent_labels = size_campaign_scored["label_fraud_spike"].to_numpy(dtype=int)
    scenario_scores = size_campaign_scored["score"].to_numpy(dtype=float)

    drift_config = _simulation_config("temporal_degradation")
    drift_result = generate_simulation(drift_config)
    shifted_observations = apply_gradual_temporal_shift(drift_result.observations)
    drift_report, drift_scored = _score_observations(model, shifted_observations, metadata)

    report = {
        "report_kind": "phase-4.5-robustness-tests",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "scenarios": {
            "previously_unseen_anomaly_severity": {
                "config": asdict(severity_config),
                "definition": "Fraud severities 1.05-1.60, outside the training simulator's 0.35-1.00 range.",
                "result": severity_report,
            },
            "merchant_size_variation": {
                "config": asdict(size_campaign_config),
                "overall": size_campaign_report,
                "slices_by_simulator_archetype": size_slices,
                "disclosure": "Archetypes are synthetic merchant-size proxies, not real merchant segments.",
            },
            "legitimate_high_volume_events": {
                "config": asdict(size_campaign_config),
                "definition": "Every eligible merchant is offered a label-negative legitimate campaign event.",
                "negative_campaign_alerts": _negative_campaign_alerts(
                    size_campaign_scored, campaign_keys(size_campaign_result.events, size_campaign_config.start_date), metadata
                ),
            },
            "missing_records": {
                "config": asdict(missing_config),
                "definition": "20% missing-observation rate versus the baseline simulator's 1.5%.",
                "result": missing_report,
            },
            "delayed_records": {
                "config": asdict(delayed_config),
                "definition": "45% delayed-record rate; only rows available by their own end-of-day are scoreable.",
                "original_observations": int(len(delayed_result.observations)),
                "same_day_available_observations": int(len(timely)),
                "same_day_availability_coverage": round(float(len(timely) / len(delayed_result.observations)), 6),
                "result": delayed_report,
            },
            "distribution_shift": run_ood_benchmark(),
            "noisy_labels": {
                "definition": "10% symmetric evaluation-label flips; latent labels and model scores are unchanged.",
                "rows": int(len(latent_labels)),
                "flipped_rows": flipped_rows,
                "latent_label_metrics": _metric_slice(size_campaign_scored, metadata),
                "noisy_observed_label_metrics": {
                    "ranking": ranking_metrics(noise_labels, scenario_scores),
                    "review_tier": decision_metrics(noise_labels, scenario_scores, float(metadata["review_threshold"])),
                    "high_tier": decision_metrics(noise_labels, scenario_scores, float(metadata["high_threshold"])),
                },
            },
            "changing_class_prevalence": {
                "low_prevalence": {"config": asdict(low_config), "result": low_report},
                "high_prevalence": {"config": asdict(high_config), "result": high_report},
            },
            "model_degradation_over_time": {
                "config": asdict(drift_config),
                "induced_shift": {
                    "definition": "After day 150, label-negative and label-positive rows receive gradual global product-mix volume and amount drift.",
                    "daily_volume_growth": 0.003,
                    "daily_amount_growth": 0.0015,
                },
                "overall": drift_report,
                "chronological_90_day_windows": _temporal_windows(drift_scored, metadata),
            },
        },
        "selection_boundary": (
            "Every scenario is synthetic, independently seeded evaluation evidence. The frozen artifact and existing "
            "thresholds are scored without retraining, tuning, calibration selection, or promotion."
        ),
        "real_world_disclosure": "These robustness tests do not establish real-world fraud performance or fairness across real merchant populations.",
    }
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 4.5 synthetic robustness tests.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
