# Explainability and decision-science evidence (Phase 6)

Phase 6 makes the explanation and action boundary explicit. It does not
change the selected `fraud-spike-v2-eod` HistGradientBoosting model, its
validation-selected score thresholds, or the fact that all reported labels are
synthetic.

## Evidence and reproducibility

```powershell
python evaluation/run_explainability_report.py
python evaluation/run_policy_analysis.py
python evaluation/run_deep_learning_experiments.py
python -m pytest tests/test_policy_explanations.py tests/test_risk_engine.py tests/test_tenant_api.py -q
```

The committed evidence was generated on the 4,346-row validation window
(2026-04-16 through 2026-05-15); the final temporal test is not loaded by
either Phase 6 runner.

`evaluation/explainability_report.json` records the serving-model artifact
SHA-256, feature-schema version, Git provenance (including whether the
explanation source paths were clean), SHAP version, and the exact
validation sample. `evaluation/global_feature_importance.svg` is a
dependency-free plot of the measured global mean absolute SHAP values.

On the first 1,024 chronological validation rows, the leading global
mean-absolute SHAP features are `historical_anomaly_rate` (0.371827),
`amount_volatility_ratio` (0.328086), `amount_zscore` (0.308832), and
`region_spread_zscore` (0.303486). These are descriptive local-model
attribution results on synthetic validation data, not causal explanations or
real-world fraud evidence.

## Explanation contract

Each tree-SHAP signal includes all of the following:

- The model input feature value and its neutral reference.
- Where the feature supports it, the actual observed end-of-day value and its
  strictly-prior merchant baseline.
- `observed_value_direction`, such as `below_historical_baseline`.
- `attribution_direction`, such as `increases_model_estimated_risk`, in the
  tree model's raw log-odds margin space.
- Provenance: explanation schema, method, model version, and feature-schema
  version.

These directions are intentionally independent. For example, a lower
devices-per-user observation can still have a positive SHAP contribution if
the tree learned that a decrease from this merchant's own baseline is risky.
The system no longer describes a positive SHAP value as an observed increase.

When Tree SHAP is unavailable or incompatible, the result is explicitly marked
as `heuristic_deviation_fallback`; it is not presented as model attribution.

The non-serving Phase 5 MLP has a reproducible Integrated Gradients example
with a zero vector in training-normalized feature space as its stated
baseline. It is offline evidence only: no neural model was promoted, and this
attribution cannot modify a score, a policy decision, or an investigation.

## Three separate records

| Concern | Source of truth | Can change the others? |
|---|---|---|
| Model-estimated risk | Serialized model score | No |
| Policy decision | `model/artifacts/policy.json` | No |
| Human investigation outcome | Case event and verified analyst feedback | No |

Every new durable `RiskAssessment` persists both `model_version` and
`policy_version`, along with the policy-decision trace. The database migration
`20261108_04` adds this trace safely; older assessments retain `{}` rather
than a fabricated historical decision.

The active `three-action-v1` policy maps scores below 0.10 to LOW, 0.10-0.67
to REVIEW, and scores at least 0.67 to HIGH. Its actions are recommendations:
LOW continues monitoring, REVIEW recommends analyst review, and HIGH creates a
priority investigation recommendation. Human confirmation is required before
irreversible merchant-facing action.

## Transparent policy sensitivity

The active policy and two deliberately unselected candidates are evaluated on
the validation labels under disclosed illustrative INR assumptions. The
framework charges exclusive actions separately: analyst review for every
reviewed row, true-positive response and false-alert cost by review outcome,
escalation and outcome-specific cost for HIGH, and missed-fraud loss only for
positive LOW rows.

| Policy | REVIEW / HIGH thresholds | REVIEW rows (capacity 217) | HIGH rows | Estimated cost |
|---|---:|---:|---:|---:|
| Active `three-action-v1` | 0.10 / 0.67 | 99 | 251 | INR 558,450 |
| Recall candidate | 0.05 / 0.55 | 159 | 264 | INR 587,750 |
| Conservative candidate | 0.20 / 0.80 | 75 | 238 | INR 599,950 |

The active policy remains unchanged because it mirrors the pre-existing
validation-selected thresholds. The table is sensitivity analysis, not a
new policy optimization or a financial-impact claim. Review capacity is a
planning check on REVIEW rows; HIGH priority investigations require their own
operational capacity planning.

## Remaining limits

- The score is an uncalibrated class-1 model output; Phase 4.4 did not retain
  a calibration candidate. It is not a guaranteed fraud probability.
- SHAP explains model behavior, not causality or intent.
- Cost assumptions, capacity, and risk tolerance are placeholders pending
  real operating data and governance approval.
- Synthetic labels support only offline evaluation; analyst feedback must be
  verified before future real-world evaluation use.
