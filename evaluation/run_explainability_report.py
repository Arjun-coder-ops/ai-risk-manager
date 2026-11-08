"""Produce reproducible local/global explanation evidence for the serving tree model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from explanations import normalize_tree_shap_values, tree_attribution_signal  # noqa: E402
from features import FEATURE_COLUMNS, FEATURE_SCHEMA_VERSION  # noqa: E402
from split import time_split  # noqa: E402


FEATURES_PATH = os.path.join(REPO_ROOT, "data", "processed", "features.csv")
MODEL_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model.joblib")
MODEL_META_PATH = os.path.join(REPO_ROOT, "model", "artifacts", "model_meta.json")
DEEP_LEARNING_REPORT_PATH = os.path.join(REPO_ROOT, "evaluation", "deep_learning_experiments.json")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "evaluation", "explainability_report.json")
DEFAULT_PLOT = os.path.join(REPO_ROOT, "evaluation", "global_feature_importance.svg")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    source_paths = [
        "model/explanations.py",
        "model/features.py",
        "model/risk_engine.py",
        "model/policy.py",
        "evaluation/run_explainability_report.py",
    ]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.run(
            ["git", "diff", "--quiet", "--", *source_paths], cwd=REPO_ROOT, check=False
        ).returncode != 0
        return {"commit": commit, "source_paths_clean": not dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "source_paths_clean": None}


def _write_importance_svg(rows: list[dict[str, Any]], output_path: str) -> None:
    """Write a dependency-free contribution plot from measured global SHAP means."""
    plotted = rows[:10]
    width, row_height, left, right, top = 900, 34, 310, 80, 60
    height = top + row_height * len(plotted) + 50
    maximum = max((row["mean_absolute_shap"] for row in plotted), default=1.0) or 1.0
    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" role="img" aria-label="Global mean absolute SHAP feature importance">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="30" font-family="Arial, sans-serif" font-size="18" fill="#18212f">Global feature importance — mean absolute SHAP (validation sample)</text>',
    ]
    usable_width = width - left - right
    for index, row in enumerate(plotted):
        y = top + index * row_height
        bar_width = usable_width * row["mean_absolute_shap"] / maximum
        label = html.escape(row["feature"])
        value = html.escape(f"{row['mean_absolute_shap']:.4f}")
        pieces.extend([
            f'<text x="{left - 12}" y="{y + 19}" text-anchor="end" font-family="Arial, sans-serif" font-size="13" fill="#263445">{label}</text>',
            f'<rect x="{left}" y="{y + 5}" width="{bar_width:.2f}" height="18" fill="#287a65"/>',
            f'<text x="{left + bar_width + 8:.2f}" y="{y + 19}" font-family="Arial, sans-serif" font-size="12" fill="#263445">{value}</text>',
        ])
    pieces.append("</svg>")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(pieces))


def _neural_support() -> dict[str, Any]:
    if not os.path.exists(DEEP_LEARNING_REPORT_PATH):
        return {"status": "not_run", "method": "integrated_gradients", "reason": "Phase 5 report is absent."}
    with open(DEEP_LEARNING_REPORT_PATH, encoding="utf-8") as handle:
        report = json.load(handle)
    example = report.get("models", {}).get("mlp", {}).get("neural_attribution_example")
    if not example:
        return {"status": "not_yet_regenerated", "method": "integrated_gradients"}
    return {"status": "available_offline_only", "method": "integrated_gradients", "example": example}


def run(output_path: str = DEFAULT_OUTPUT, plot_path: str = DEFAULT_PLOT) -> dict[str, Any]:
    with open(MODEL_META_PATH, encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or metadata.get("feature_columns") != FEATURE_COLUMNS:
        raise RuntimeError("Model metadata is incompatible with the current feature contract.")
    featured = pd.read_csv(FEATURES_PATH, parse_dates=["date"])
    _, validation, _ = time_split(featured)
    sample = validation.iloc[: min(1024, len(validation))].copy()
    model = joblib.load(MODEL_PATH)
    explainer = shap.TreeExplainer(model)
    shap_values = normalize_tree_shap_values(
        explainer.shap_values(sample[FEATURE_COLUMNS]),
        expected_rows=len(sample),
        expected_features=len(FEATURE_COLUMNS),
    )
    scores = model.predict_proba(sample[FEATURE_COLUMNS])[:, 1]
    mean_absolute = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    global_rows = sorted(
        [
            {
                "feature": feature,
                "mean_absolute_shap": round(float(mean_absolute[index]), 6),
                "mean_signed_shap": round(float(mean_signed[index]), 6),
                "attribution_space": "model_raw_margin_log_odds",
            }
            for index, feature in enumerate(FEATURE_COLUMNS)
        ],
        key=lambda row: row["mean_absolute_shap"],
        reverse=True,
    )
    local_examples = []
    for position in np.argsort(scores)[::-1][:3]:
        row = sample.iloc[int(position)]
        shap_row = shap_values[int(position)]
        feature_order = np.argsort(np.abs(shap_row))[::-1][:3]
        local_examples.append({
            "merchant_id": str(row["merchant_id"]),
            "date": str(pd.Timestamp(row["date"]).date()),
            "risk_score": round(float(scores[int(position)]), 6),
            "signals": [
                tree_attribution_signal(
                    FEATURE_COLUMNS[index],
                    row,
                    float(shap_row[index]),
                    model_version=metadata["model_version"],
                    feature_schema_version=FEATURE_SCHEMA_VERSION,
                )
                for index in feature_order
            ],
        })
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    _write_importance_svg(global_rows, plot_path)
    report = {
        "report_kind": "phase-6-explainability-evidence",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_provenance": _git_provenance(),
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "model_artifact_sha256": _sha256(MODEL_PATH),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "explanation_provenance": {
            "method": "TreeExplainer",
            "shap_version": shap.__version__,
            "attribution_space": "model_raw_margin_log_odds",
            "sample_selection": "first 1,024 chronological validation rows",
            "validation_date_range": [str(sample.date.min().date()), str(sample.date.max().date())],
            "final_test_loaded": False,
        },
        "global_feature_importance": global_rows,
        "global_feature_importance_plot": os.path.basename(plot_path),
        "local_explanations": local_examples,
        "neural_attribution": _neural_support(),
        "interpretation_guardrail": (
            "observed_value_direction describes the feature observation against its prior baseline or neutral reference; "
            "attribution_direction independently describes whether the model attribution increased or decreased estimated risk."
        ),
        "operational_boundary": (
            "Tree SHAP is attached only to the compatible serving tree model. Integrated gradients is retained as offline "
            "evidence for the non-serving Phase 5 MLP; it cannot alter a score, policy decision, or human case outcome."
        ),
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Phase 6 SHAP explanation evidence.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--plot", default=DEFAULT_PLOT)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.plot), indent=2))


if __name__ == "__main__":
    main()
