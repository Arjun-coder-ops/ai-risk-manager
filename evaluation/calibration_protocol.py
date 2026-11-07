"""Chronological calibration partitioning and dependency-free reliability SVGs."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Iterable

import pandas as pd


MODEL_FIT_END = "2026-03-15"
CALIBRATION_METHOD_SELECTION_END = "2026-04-15"
CALIBRATION_FIT_END = "2026-03-31"


@dataclass(frozen=True)
class CalibrationPartitions:
    fit: pd.DataFrame
    calibration_fit: pd.DataFrame
    calibration_selection: pd.DataFrame
    purged_event_count: int
    purged_rows: int


def calibration_partitions(
    featured: pd.DataFrame,
    events: Iterable[dict[str, Any]],
    *,
    fit_end: str = MODEL_FIT_END,
    calibration_fit_end: str = CALIBRATION_FIT_END,
    calibration_method_selection_end: str = CALIBRATION_METHOD_SELECTION_END,
) -> CalibrationPartitions:
    """Create disjoint fit/calibration roles and purge boundary-spanning events."""
    if not (
        pd.Timestamp(fit_end)
        < pd.Timestamp(calibration_fit_end)
        < pd.Timestamp(calibration_method_selection_end)
    ):
        raise ValueError("fit, calibration-fit, and calibration-selection boundaries must increase.")
    required = {"merchant_id", "date", "has_sufficient_history"}
    missing = sorted(required - set(featured.columns))
    if missing:
        raise ValueError(f"Feature frame lacks required fields: {missing}")
    eligible = featured[featured["has_sufficient_history"]].copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="raise").dt.strftime("%Y-%m-%d")
    fit_end = str(pd.Timestamp(fit_end).date())
    calibration_fit_end = str(pd.Timestamp(calibration_fit_end).date())
    calibration_method_selection_end = str(pd.Timestamp(calibration_method_selection_end).date())
    purge_keys: set[tuple[str, str]] = set()
    purged_event_count = 0
    for event in events:
        if event.get("kind") != "fraud_spike":
            continue
        dates = sorted(str(date_value) for date_value in event.get("affected_dates", []))
        boundaries = (fit_end, calibration_fit_end)
        if not dates or not any(dates[0] <= boundary < dates[-1] for boundary in boundaries):
            continue
        purged_event_count += 1
        merchant_id = str(event["merchant_id"])
        purge_keys.update((merchant_id, date_value) for date_value in dates)
    keys = list(zip(eligible["merchant_id"].astype(str), eligible["date"]))
    keep = [key not in purge_keys for key in keys]
    cleaned = eligible.loc[keep].copy()
    purged_rows = len(eligible) - len(cleaned)
    fit = cleaned[cleaned["date"] <= fit_end].copy()
    calibration_fit = cleaned[(cleaned["date"] > fit_end) & (cleaned["date"] <= calibration_fit_end)].copy()
    calibration_selection = cleaned[
        (cleaned["date"] > calibration_fit_end) & (cleaned["date"] <= calibration_method_selection_end)
    ].copy()
    if fit.empty or calibration_fit.empty or calibration_selection.empty:
        raise ValueError("Calibration partitioning produced an empty role.")
    return CalibrationPartitions(
        fit=fit,
        calibration_fit=calibration_fit,
        calibration_selection=calibration_selection,
        purged_event_count=purged_event_count,
        purged_rows=purged_rows,
    )


def reliability_diagram_svg(series: dict[str, dict[str, Any]], *, title: str) -> str:
    """Render calibration-bin points and an identity line without a plotting dependency."""
    width, height, left, top, chart = 760, 490, 92, 58, 340
    colors = {"raw": "#6b7280", "sigmoid": "#2563eb", "isotonic": "#059669"}
    x = lambda value: left + float(value) * chart
    y = lambda value: top + (1.0 - float(value)) * chart
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-family="sans-serif" font-size="18" font-weight="700">{escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart}" x2="{left + chart}" y2="{top + chart}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart}" stroke="#111827"/>',
        f'<line x1="{x(0)}" y1="{y(0)}" x2="{x(1)}" y2="{y(1)}" stroke="#9ca3af" stroke-dasharray="6 6"/>',
        f'<text x="{left + chart / 2 - 45}" y="{top + chart + 48}" font-family="sans-serif" font-size="13">Mean predicted score</text>',
        f'<text x="20" y="{top + chart / 2}" transform="rotate(-90 20 {top + chart / 2})" font-family="sans-serif" font-size="13">Observed positive rate</text>',
    ]
    for tick in range(6):
        value = tick / 5
        elements.extend((
            f'<text x="{x(value) - 8:.1f}" y="{top + chart + 20}" font-family="sans-serif" font-size="11">{value:.1f}</text>',
            f'<text x="{left - 32}" y="{y(value) + 4:.1f}" font-family="sans-serif" font-size="11">{value:.1f}</text>',
        ))
    legend_y = 92
    for index, (name, report) in enumerate(series.items()):
        color = colors.get(name, "#7c3aed")
        bins = report.get("bins", [])
        points = " ".join(
            f"{x(item['mean_score']):.2f},{y(item['observed_positive_rate']):.2f}" for item in bins
        )
        if points:
            elements.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
        for item in bins:
            elements.append(
                f'<circle cx="{x(item["mean_score"]):.2f}" cy="{y(item["observed_positive_rate"]):.2f}" r="3.5" fill="{color}"/>'
            )
        label = f"{name}: Brier {report['brier_score']:.6f}, ECE {report['expected_calibration_error']:.6f}"
        elements.extend((
            f'<line x1="470" y1="{legend_y + index * 30}" x2="495" y2="{legend_y + index * 30}" stroke="{color}" stroke-width="3"/>',
            f'<text x="505" y="{legend_y + 5 + index * 30}" font-family="sans-serif" font-size="12">{escape(label)}</text>',
        ))
    elements.append('</svg>')
    return "\n".join(elements) + "\n"
