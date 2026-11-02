# Merchant-day end-of-day feature contract

Schema version: `merchant-day-eod-v2`.

## Availability and consistency

All features in this contract are calculated **after a merchant-day closes**
and its daily aggregates are available. They are not intraday-monitoring
features: an intraday product would need separately versioned partial-day
counts, watermarks, and an explicit availability policy.

`model/features.py:build_features` is the sole implementation for offline
training/evaluation and `model/risk_engine.py` inference. The risk engine
rejects a model artifact whose feature schema/version does not match this
contract. All historical windows are calendar-time, left-closed windows:
they use rows strictly before the merchant-day being scored. Labels and model
scores are never feature inputs.

When history is too short or a denominator has no variation, the documented
neutral value is used rather than imputing future data. `has_sufficient_history`
remains false until at least seven prior observations exist in the preceding
14 calendar days, and the policy engine returns insufficient evidence instead
of a risk decision.

| Feature | Business meaning and calculation | Input fields / lookback | Missing or thin-history treatment | End-of-day availability, leakage control, expected benefit |
|---|---|---|---|---|
| `txn_volume_zscore` | Today’s transaction-count deviation from the merchant’s own mean. | `n_transactions`; prior 14 calendar days, >=7 observations. | `0`. | EOD; left-closed baseline; catches volume bursts. |
| `failed_rate_delta` | Change in declined/failed share vs normal. | `n_failed`, `n_transactions`; prior 14 days. | Zero rate and delta `0`. | EOD; no same/future row in baseline; catches card testing or checkout failure shifts. |
| `amount_zscore` | Today’s average ticket deviation. | `avg_amount`; prior 14 days. | `0`. | EOD; prior-only baseline; catches unusually large or small tickets. |
| `amount_volatility_ratio` | Today’s amount dispersion vs normal dispersion. | `amount_std`; prior 14 days. | `1`. | EOD; prior-only baseline; catches spread changes. |
| `new_user_ratio_delta` | Change in new users as a share of unique users. | `n_new_users`, `n_unique_users`; prior 14 days. | Zero ratio and delta `0`. | EOD; prior-only baseline; catches bursts of new identities. |
| `device_diversity_ratio` | Current devices per unique user. | `n_devices`, `n_unique_users`; current closed-day aggregate. | `1` when users are zero. | EOD; no history needed; exposes reuse or device proliferation. |
| `region_spread_zscore` | Geographic footprint deviation. | `distinct_regions_seen`; prior 14 days. | `0`. | EOD; prior-only baseline; catches distributed activity. |
| `rolling_volume_trend` | Short-horizon momentum state. | `n_transactions`; prior 3-day mean / prior 14-day mean. | `1`. | EOD; both windows end before today; detects developing trends. |
| `weekday_volume_ratio` | Seasonal/weekday-normalized volume. | `n_transactions`; matching weekday in prior 56 days, >=2 observations. | `1`. | EOD; prior matching weekdays only; prevents weekend/weekday demand from looking anomalous. |
| `txn_volume_percentile` | Position of today’s volume in recent distribution. | `n_transactions`; prior 28 days, >=7 observations. | `0.5`. | EOD; explicit strict-prior loop; robust to non-normal merchant volume. |
| `ewm_volume_zscore` | Fast-reacting deviation from an exponentially weighted state. | `n_transactions`; prior observation-order EWM span 14, >=7 observations. | `0`. | EOD; input shifted before EWM; adapts to sustained shifts without future leakage. |
| `recent_volume_volatility_ratio` | Recent three-day volume volatility vs long baseline volatility. | `n_transactions`; prior 3-day std / prior 14-day std. | `1`. | EOD; both are historical state; detects instability before/around attacks. |
| `txn_acceleration_ratio` | Today’s volume relative to immediately preceding volume. | `n_transactions`; current value / prior 3-day mean. | `1`. | EOD; denominator is prior-only; catches sudden acceleration. |
| `trend_adjusted_volume_zscore` | Today’s deviation after allowing recent trend. | `n_transactions`; current value minus prior 3-day mean, scaled by prior 14-day std. | `0`. | EOD; all expected-state terms are prior-only; reduces false alerts during gradual growth/decline. |
| `volume_change_point_score` | Strength of a prior short-vs-long volume regime break. | `n_transactions`; prior 3-day vs prior 14-day means/std. | `0`. | EOD; entirely historical; identifies baseline instability. |
| `device_diversity_delta` | Merchant-relative device mix change. | `n_devices`, `n_unique_users`; current ratio minus prior 14-day ratio mean. | `0`. | EOD; baseline is prior-only; gives context to current device diversity. |
| `historical_anomaly_rate` | Frequency of prior raw-signal anomalies. | Prior `txn_volume_zscore`, `failed_rate_delta`, `amount_zscore`, `region_spread_zscore`; prior 28 days. | `0`. | EOD; current indicator is excluded with a left-closed window; summarizes repeat anomalous behaviour without using labels. |

Z-score, ratio, percentile, delta, and historical-rate features are clipped
to documented bounded ranges after calculation to limit the effect of
corrupted but valid extreme aggregates. The clip is not a substitute for input
validation, which occurs first in `model/validation.py`.
