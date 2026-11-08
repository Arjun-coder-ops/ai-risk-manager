# Ablation experiments (Phase 4.6)

`evaluation/run_ablation_experiments.py` compares feature sets, calibration
mappings, and model families on one common, chronological synthetic evaluation
window. It writes `evaluation/ablation_experiments.json` and makes no serving
or promotion change.

## Protocol

| Role | Dates | Rows | Purpose |
|---|---|---:|---|
| Model fit | 2026-01-08 to 2026-02-28 | 6,601 | Fits every comparable model. |
| Calibration-map fit | 2026-03-01 to 2026-03-07 | 894 | Fits sigmoid/isotonic mappings only. |
| Common experiment evaluation | 2026-03-08 to 2026-03-15 | 1,099 | Scores every ablation; 4.8226% positive. |

Twenty-nine fraud events crossing either experiment boundary were removed in
full (164 merchant-day rows). The later calibration-selection window
(2026-04-01 to 2026-04-15), threshold-selection window, and declared final
test are untouched.

## Reproduced comparison

### Feature ablations

All feature variants use the same HistGradientBoosting parameters and common
evaluation rows.

| Feature set | Features | ROC-AUC | PR-AUC |
|---|---:|---:|---:|
| Full merchant-relative contract | 17 | 0.995653 | 0.946726 |
| Reduced core merchant-relative | 8 | **0.996663** | **0.947519** |
| Global/raw baseline | 7 | 0.979545 | 0.928373 |

The narrow reduced-vs-full difference on one synthetic slice is not evidence
to remove features. The global baseline is worse, supporting the use of
merchant-relative context, but it is also synthetic-only evidence.

### Calibration ablation

The mappings fit only on the calibration-map window and are compared on the
common evaluation window.

| Mapping | Brier | ECE | PR-AUC |
|---|---:|---:|---:|
| Raw | 0.008812 | 0.005329 | **0.946726** |
| Sigmoid | 0.008806 | **0.003412** | **0.946726** |
| Isotonic | **0.008533** | 0.005628 | 0.934008 |

This is a metric trade-off, not a reversal of Phase 4.4’s decision to retain
no calibration candidate. The later, dedicated calibration-method-selection
experiment remains authoritative for serving. None of these synthetic scores
is a reliable real-world fraud probability.

### Model-family comparison

Every family uses the full 17-feature contract, the same fit rows, and the
same evaluation rows.

| Family | Score semantics | PR-AUC |
|---|---|---:|
| Heuristic baseline | Ranking-only | 0.471237 |
| Logistic regression | Model score | 0.711398 |
| Random forest | Model score | 0.825346 |
| HistGradientBoosting | Model score | 0.946726 |
| XGBoost | Model score | 0.935068 |
| LightGBM | Model score | **0.952794** |
| Isolation Forest | Ranking-only | 0.309280 |

LightGBM’s lead on this small, early synthetic slice does not promote it or
replace the active HistGradientBoosting artifact. Heuristic and Isolation
Forest scores are not probabilities and remain ineligible for threshold or
deployment selection.

## Reproduce

```powershell
python evaluation/run_ablation_experiments.py
python -m pytest tests/test_ablation_experiments.py -q
```

This report completes the Phase 4 experimental comparison requirement. Its
results must be read together with the documented OOD, campaign, delayed-data,
and drift failures in the robustness report.
