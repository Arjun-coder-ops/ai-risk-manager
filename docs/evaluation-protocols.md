# Generalization protocol evidence (Phase 4.2)

`evaluation/run_generalization_protocols.py` evaluates the frozen
`fraud-spike-v2-eod` artifact without threshold tuning, calibration, or model
promotion. It writes `evaluation/generalization_protocols.json`.

## Protocols and boundaries

| Protocol | Design | Evidence boundary |
|---|---|---|
| Temporal | Reuses the declared frozen report in `evaluation/test_results.json` rather than rescoring or tuning its final temporal test set. | The 6,157 merchant-days from 2026-05-16 to 2026-06-29 remain the held-out temporal result. |
| Merchant generalization | Deterministic 90/30/30 merchant cohorts; training uses early dates, validation uses new merchants in the next period, and test uses another new merchant cohort in the final period. | No merchant overlaps; no hyperparameter, model, threshold, or calibration choice uses the merchant-test cohort. |
| OOD | Reuses the separately designed synthetic OOD generator. | Explicitly simulated; never external or real-world validation. |
| Cold start | Counts rows with fewer than seven prior observed days. | The operational engine intentionally returns `insufficient_evidence`, so no artificial probability or classification score is reported. |
| Stress | Independently seeded synthetic populations vary anomaly severity, prevalence, missingness, and legitimate campaign/quiet behaviour. | Ranking evidence only; no scenario is used for tuning or promotion. |

## Reproduced ranking evidence

| Slice | ROC-AUC | PR-AUC | Notes |
|---|---:|---:|---|
| Frozen temporal held-out report | 0.990300 | 0.950000 | Existing temporal result; not rerun or retuned by this command. |
| Merchant-disjoint future test | 0.980936 | 0.930540 | 30 merchant test cohort; synthetic data. |
| Independent synthetic OOD | 0.939000 | 0.668100 | Material degradation; do not use the in-distribution figure as an OOD proxy. |
| Mild anomalies | 0.897847 | 0.697251 | Lower-severity attacks are substantially harder. |
| Low prevalence | 0.984084 | 0.901963 | 2.62% scored positive rate. |
| High prevalence | 0.986352 | 0.968384 | 15.71% scored positive rate; not representative of a real base rate. |
| High missingness | 0.979623 | 0.922574 | 12% simulated missing observations. |
| Campaign-heavy | 0.979391 | 0.902282 | 80% of merchants may receive legitimate campaign events. |

The cold-start slice contains 1,050 rows across all 150 merchants. Its label
prevalence is 0 in this simulation because fraud injection is deliberately
scheduled after a baseline exists. That is a simulator property, not proof
that real new merchants are safe.

Run the protocol in the pinned environment:

```powershell
python evaluation/run_generalization_protocols.py
```

Phase 4.3 now supplies the complete decision, review-budget, event, raw
calibration, cost, and cluster-bootstrap metric contract in
[`decision-metrics.md`](decision-metrics.md). Phase 4.4 separately tests
calibration methods.
