# Calibration experiment (Phase 4.4)

`evaluation/run_calibration_experiment.py` evaluates sigmoid (Platt-style) and
isotonic score calibration without changing the active serving artifact. It
writes `evaluation/calibration_report.json` and the dependency-free SVG
reliability diagram at `evaluation/calibration_reliability.svg`.

## Chronological roles and anti-leakage boundary

| Role | Dates | Rows | Use |
|---|---|---:|---|
| Base-model fit | 2026-01-08 to 2026-03-15 | 8,742 | Fits a fresh HistGradientBoosting calibration-experiment base model. |
| Calibration fit | 2026-03-16 to 2026-03-31 | 2,304 | Fits each score-mapping method. |
| Calibration-method selection | 2026-04-01 to 2026-04-15 | 2,179 | Compares raw, sigmoid, and isotonic maps. |
| Existing threshold/model validation | 2026-04-16 to 2026-05-15 | not consumed | Remains logically separate. |
| Declared final temporal test | 2026-05-16 to 2026-06-29 | not consumed | Remains untouched by this experiment. |

Fifteen simulated fraud events that crossed a fit/calibration boundary were
purged in full (78 available merchant-day rows), so a single fraud event does
not appear in two calibration roles. This remains synthetic offline evidence.

## Result and decision

Methods were ranked first by Brier score on the **untouched
calibration-method-selection** rows; ECE and method name are deterministic
tie-breakers. A calibrated method must strictly beat raw Brier to be retained.

| Score mapping | Brier | Ten-bin ECE | Result |
|---|---:|---:|---|
| Raw model score | **0.009472** | 0.005844 | Baseline; not a reliable fraud probability. |
| Sigmoid | 0.010276 | 0.006022 | Rejected: worse Brier and ECE. |
| Isotonic | 0.009968 | **0.004585** | Rejected: better ECE but worse primary Brier. |

No calibrated candidate artifact is retained or deployed. This is intentional:
the evidence does not justify changing score semantics. The active model still
returns an uncalibrated model score, and neither score nor this synthetic
experiment should be represented as a reliable real-world fraud probability.

The reliability SVG plots bin-level mean score against observed rate alongside
the identity line. It is a diagnostic for method comparison, not a production
calibration guarantee.

## Reproduce

Run in the locked environment:

```powershell
python evaluation/run_calibration_experiment.py
python -m pytest tests/test_calibration_experiment.py -q
```

If future data or methodology makes a calibrated method win, the runner will
serialize exactly one non-deployed candidate. It must still clear independent
threshold, robustness, and release-gate evidence before serving.
