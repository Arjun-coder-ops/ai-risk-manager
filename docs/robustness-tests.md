# Robustness evidence (Phase 4.5)

`evaluation/run_robustness_tests.py` scores the frozen
`fraud-spike-v2-eod` artifact and its existing thresholds on independently
seeded synthetic scenarios. It never retrains, tunes thresholds, selects a
calibrator, or promotes an artifact. The complete machine-readable evidence is
`evaluation/robustness_tests.json`.

## Required-axis coverage and results

| Axis | Reproduced evidence | Interpretation |
|---|---|---|
| Unseen anomaly severity | Severity 1.05–1.60 (outside the 0.35–1.00 training range): ROC-AUC 0.995663, PR-AUC 0.977429. | Larger, obvious synthetic attacks remain easy; this says nothing about subtle novel attacks. |
| Merchant-size variation | PR-AUC: small 0.889420, medium 0.940687, large 0.945585. | Small synthetic merchants have the weakest ranking and more REVIEW false alerts (41.17 per 1,000). |
| Legitimate high-volume events | On 376 scoreable, label-negative campaign days: REVIEW false-alert rate 0.933511; HIGH 0.289894. | Severe synthetic false-positive failure. Do not auto-act on campaign-like volume changes. |
| Missing records | 20% missing-observation rate: PR-AUC 0.922201; REVIEW recall 0.880042. | Ranking and recall degrade from the frozen in-distribution result. |
| Delayed records | 45% delayed rate leaves only 55.3787% same-day availability; PR-AUC 0.897316 on 6,340 scoreable days. | Delayed arrival is both a coverage and quality issue. |
| Distribution shift | Independent OOD: PR-AUC 0.6681; REVIEW precision 0.1410 and FPR 0.3976. | The frozen policy is unsafe under this synthetic shifted population. |
| Noisy labels | 10% symmetric evaluation-label flips reduce PR-AUC from 0.914145 (latent) to 0.507064. | Reported quality is highly sensitive to label reliability. |
| Changing prevalence | REVIEW precision: 0.644670 at 2.63% prevalence; 0.898266 at 15.66%. | Precision and alert volume are prevalence-dependent; a fixed threshold needs monitoring. |
| Degradation over time | Induced late product-mix drift: PR-AUC falls from 0.941588 in days 1–90 to 0.618582 in days 271–360; REVIEW false alerts rise from 13.25 to 170.32 per 1,000. | Strong synthetic drift failure; a future release gate must detect and contain it. |

The OOD, campaign, and late-drift results are not defects hidden by aggregate
ROC-AUC. They are evidence that this prototype cannot claim production
robustness or deploy automated merchant actions. Thresholds are intentionally
left unchanged in this phase so the failure modes remain independently
verifiable.

## Method boundaries

- Merchant size uses simulator archetypes only; they are not demographic,
  geographic, or real customer segments.
- Campaign, noise, delayed-arrival, and drift definitions are fully recorded in
  the report config; all labels remain offline-only synthetic ground truth.
- The delayed-record test scores only rows available by that event day’s end,
  rather than incorrectly treating future-arriving records as known history.
- Time degradation introduces label-neutral volume and amount drift after day
  150 and reports four chronological 90-day windows.
- This suite is robustness evidence, not real-world validation, fairness
  analysis, or a promotion decision.

## Reproduce

```powershell
python evaluation/run_robustness_tests.py
python -m pytest tests/test_robustness.py -q
```

The release work that follows must define monitoring, alerting, and rollback
criteria before any real deployment claim can be made.
