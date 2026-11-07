# Decision, capacity, and uncertainty evidence (Phase 4.3)

`evaluation/run_decision_metrics.py` expands the evidence for the frozen
`fraud-spike-v2-eod` artifact. It scores the declared temporal holdout with
the already-selected model and thresholds, then writes
`evaluation/decision_metrics.json`. It performs no model selection, threshold
tuning, probability calibration, or promotion.

## Reproduced result

The committed report was generated with 200 deterministic resamples
(`random_state=42`) on the 6,157 merchant-day temporal holdout from
2026-05-16 through 2026-06-29. Its fraud prevalence is 7.3250%; a random
ranking baseline therefore has ROC-AUC 0.5 and PR-AUC 0.073250.

| Measure | REVIEW (score >= 0.10) | HIGH (score >= 0.67) |
|---|---:|---:|
| Precision | 0.815385 | 0.959900 |
| Recall | 0.940133 | 0.849224 |
| F1 | 0.873326 | 0.901176 |
| False-positive rate | 0.016824 | 0.002804 |
| False alerts per 1,000 merchant-days | 15.592009 | 2.598668 |
| Alerts | 520 | 399 |

The frozen model's ranking metrics are ROC-AUC **0.990291** and PR-AUC
**0.950014**. These results are synthetic in-distribution evidence, not a
claim about live fraud performance.

### Analyst capacity

At a fixed review capacity of 50 merchant-days per 1,000, the exact stable
top-K policy reviews 308 of 6,157 days. It yields precision@K 0.977273,
recall 0.667406, and a score cutoff of 0.933039. This capacity calculation is
separate from the validation-selected score thresholds above; it is a planning
metric, not a replacement policy.

### Raw probability calibration

The artifact currently emits **uncalibrated model probabilities**. On the
frozen test slice, the raw Brier score is 0.010649 and the ten-uniform-bin ECE
is 0.003681. These descriptive values must not be treated as proof that scores
are calibrated in production. Phase 4.4 evaluates calibration methods without
using the temporal test set for method selection.

### Cost sensitivity

The report evaluates the existing configurable, illustrative INR cost model
under three scenarios rather than optimizing on the test data: default,
false-negative-heavy (2x missed-fraud loss), and false-positive-heavy (2x
review and merchant-friction costs). At REVIEW, the default estimate is
INR 259,200 (INR 43,200 false-positive plus INR 216,000 false-negative cost);
the false-negative-heavy case is INR 475,200. These are prototype assumptions,
not Razorpay prices, loss rates, or a recommendation to automate action.

### Event and uncertainty metrics

Fraud-spike event windows are read only from the offline simulation manifest.
Within the temporal test window, REVIEW detects 107 of 107 simulated events;
HIGH detects 106 of 107. Time-to-first-detection is measured from the first
event day that overlaps the scored slice, avoiding an artificial pre-slice
delay. This is simulation-manifest evidence, not incident-recovery evidence.

The report supplies percentile intervals from a merchant-cluster bootstrap
(200 resamples): ROC-AUC [0.984332, 0.995165], PR-AUC [0.914826, 0.972391],
and REVIEW precision [0.767959, 0.859018]. Event recall and detection-time
intervals use an event-cluster bootstrap. Cluster resampling keeps all rows
from a selected merchant or event together, rather than pretending merchant
days are independent.

## Reproduce

Use the locked project environment and retain the committed report unless a
deliberate evaluation refresh is intended:

```powershell
python evaluation/run_decision_metrics.py
python -m pytest tests/test_evaluation_metrics.py -q
```

The JSON report retains confusion matrices, bin-level raw calibration data,
cost assumptions, and bootstrap configuration so a reviewer can independently
recompute or challenge each summary.
