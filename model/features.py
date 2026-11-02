"""
Feature engineering for end-of-day merchant-day fraud-spike detection.

DESIGN PRINCIPLE: every feature is a *merchant-relative* deviation, not a
global threshold. A momentum-only global rule ("flag any merchant doing
>500 txns/day") would misfire on merchants that are simply large. What
matters is whether a merchant is behaving abnormally *for itself*. So each
feature compares today's behaviour to that merchant's own trailing 14-day
baseline (excluding the current day, to avoid leakage).

Every feature has a stated reason for existing (per project requirement):

- txn_volume_zscore: sudden transaction-volume increase relative to the
  merchant's own recent baseline. Core signal for volume-burst fraud rings.
- failed_rate_delta: sudden rise in failed-payment rate. Classic signal of
  card-testing / credential-stuffing attacks against a merchant's checkout.
- amount_zscore: unusual average-transaction-amount relative to baseline.
  Catches both "unusually large ticket" fraud and "unusually small ticket"
  card-testing fraud.
- amount_volatility_ratio: change in the spread (std) of transaction
  amounts. A sudden widening of amount variance often precedes/accompanies
  a spike, even when the mean amount hasn't moved much yet.
- new_user_ratio_delta: sudden increase in the share of transactions from
  new users. Abuse rings and account-takeover waves show up as bursts of
  "new" identities transacting.
- device_diversity_ratio: unique devices per unique user. A drop toward 1
  device shared across many "different" users (device reuse) or a spike in
  devices per user can both indicate scripted/bot activity.
- region_spread_zscore: sudden increase in the number of distinct
  transaction regions seen for a merchant that normally has a tight
  geographic footprint. Signals geographically distributed fraud/bot
  traffic.
- rolling_volume_trend: 3-day vs 14-day volume ratio, a short-horizon
  momentum signal that catches spikes still building, independent of the
  z-score which is more sensitive to already-large deviations.

The advanced features below add seasonal normalization, robust historical
positioning, exponentially weighted state, volatility, acceleration, trend
breaks, and *prior* anomaly frequency.  Their full calculation and online
availability contract is in ``docs/feature-contract.md``.

Leakage control: all rolling statistics use `.shift(1)` before rolling, so
the baseline for day T only ever uses days < T. No feature ever looks at
day T's own label or at future days.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from validation import ObservationValidationError, validate_observations

BASELINE_WINDOW = 14
SHORT_WINDOW = 3
SEASONAL_WINDOW = 56
QUANTILE_WINDOW = 28
ANOMALY_HISTORY_WINDOW = 28
EWM_SPAN = 14
MIN_HISTORY_DAYS = 7  # below this we don't have enough baseline to trust a score
FEATURE_SCHEMA_VERSION = "merchant-day-eod-v2"

FEATURE_COLUMNS = [
    "txn_volume_zscore",
    "failed_rate_delta",
    "amount_zscore",
    "amount_volatility_ratio",
    "new_user_ratio_delta",
    "device_diversity_ratio",
    "region_spread_zscore",
    "rolling_volume_trend",
    "weekday_volume_ratio",
    "txn_volume_percentile",
    "ewm_volume_zscore",
    "recent_volume_volatility_ratio",
    "txn_acceleration_ratio",
    "trend_adjusted_volume_zscore",
    "volume_change_point_score",
    "device_diversity_delta",
    "historical_anomaly_rate",
]


def _safe_div(a, b, default=0.0):
    return np.where(b > 0, a / b.replace(0, np.nan), default)


def _prior_volume_percentile(g: pd.DataFrame) -> pd.Series:
    """Position today's volume in the preceding calendar-window distribution.

    This deliberately uses a small explicit loop per merchant.  Merchant
    histories are short daily sequences, and this form makes the strict
    ``date < current_date`` boundary easy to audit instead of relying on a
    rank operation whose window semantics are unclear.
    """
    dates = g["_feature_date"].to_numpy(dtype="datetime64[ns]")
    values = g["n_transactions"].to_numpy(dtype=float)
    output = np.full(len(g), 0.5, dtype=float)  # neutral when history is thin
    left = 0
    window = np.timedelta64(QUANTILE_WINDOW, "D")
    for current in range(len(g)):
        while left < current and dates[left] < dates[current] - window:
            left += 1
        history = values[left:current]
        if len(history) >= MIN_HISTORY_DAYS:
            output[current] = float(np.mean(history <= values[current]))
    return pd.Series(output, index=g.index)


def _weekday_volume_baseline(g: pd.DataFrame) -> pd.Series:
    """Return a same-weekday, prior-only seasonal volume baseline."""
    baseline = pd.Series(np.nan, index=g.index, dtype=float)
    for _, weekday_rows in g.groupby("_weekday", sort=False):
        rolling = weekday_rows.rolling(
            f"{SEASONAL_WINDOW}D",
            on="_feature_date",
            closed="left",
            min_periods=2,
        )["n_transactions"].mean()
        baseline.loc[weekday_rows.index] = rolling.to_numpy()
    return baseline


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe merchant-day features from validated observations.

    The output always retains the canonical ``(merchant_id, date)`` key.
    Calendar-time rolling windows avoid treating a missing calendar day as if
    it were an observed day of merchant history.
    """
    df = validate_observations(df).sort_values(["merchant_id", "date"]).copy()

    df["failed_rate"] = df["n_failed"] / df["n_transactions"].replace(0, np.nan)
    df["failed_rate"] = df["failed_rate"].fillna(0.0)
    df["new_user_ratio"] = df["n_new_users"] / df["n_unique_users"].replace(0, np.nan)
    df["new_user_ratio"] = df["new_user_ratio"].fillna(0.0)
    df["device_per_user"] = df["n_devices"] / df["n_unique_users"].replace(0, np.nan)
    df["device_per_user"] = df["device_per_user"].fillna(1.0)

    def per_merchant(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["_feature_date"] = pd.to_datetime(g["date"], format="%Y-%m-%d")
        g["_weekday"] = g["_feature_date"].dt.weekday
        prior_observations = g.rolling(
            f"{BASELINE_WINDOW}D", on="_feature_date", closed="left", min_periods=1
        )["n_transactions"].count()
        g["history_days"] = prior_observations.fillna(0).astype(int)

        for col, out_prefix in [
            ("n_transactions", "txn"),
            ("failed_rate", "fail"),
            ("avg_amount", "amt"),
            ("amount_std", "amtstd"),
            ("new_user_ratio", "newuser"),
            ("device_per_user", "device"),
            ("distinct_regions_seen", "region"),
        ]:
            rolling = g.rolling(
                f"{BASELINE_WINDOW}D",
                on="_feature_date",
                closed="left",
                min_periods=MIN_HISTORY_DAYS,
            )[col]
            roll_mean = rolling.mean()
            roll_std = rolling.std()
            g[f"{out_prefix}_baseline_mean"] = roll_mean
            g[f"{out_prefix}_baseline_std"] = roll_std

        # txn volume z-score vs merchant's own baseline
        std_safe = g["txn_baseline_std"].replace(0, np.nan)
        g["txn_volume_zscore"] = (g["n_transactions"] - g["txn_baseline_mean"]) / std_safe
        g["txn_volume_zscore"] = g["txn_volume_zscore"].fillna(0.0)

        # failed-rate delta vs baseline (absolute percentage-point shift)
        g["failed_rate_delta"] = g["failed_rate"] - g["fail_baseline_mean"]
        g["failed_rate_delta"] = g["failed_rate_delta"].fillna(0.0)

        # amount z-score
        amt_std_safe = g["amt_baseline_std"].replace(0, np.nan)
        g["amount_zscore"] = (g["avg_amount"] - g["amt_baseline_mean"]) / amt_std_safe
        g["amount_zscore"] = g["amount_zscore"].fillna(0.0)

        # amount volatility ratio: today's within-day std vs merchant's baseline avg within-day std
        amtstd_base_safe = g["amtstd_baseline_mean"].replace(0, np.nan)
        g["amount_volatility_ratio"] = g["amount_std"] / amtstd_base_safe
        g["amount_volatility_ratio"] = g["amount_volatility_ratio"].fillna(1.0)

        # new-user ratio delta
        g["new_user_ratio_delta"] = g["new_user_ratio"] - g["newuser_baseline_mean"]
        g["new_user_ratio_delta"] = g["new_user_ratio_delta"].fillna(0.0)

        # Current device diversity remains independently useful for bot/device
        # reuse patterns; the delta below adds merchant-relative context.
        g["device_diversity_ratio"] = g["device_per_user"]
        g["device_diversity_delta"] = (
            g["device_per_user"] - g["device_baseline_mean"]
        ).fillna(0.0)

        # region spread z-score
        region_std_safe = g["region_baseline_std"].replace(0, np.nan)
        g["region_spread_zscore"] = (g["distinct_regions_seen"] - g["region_baseline_mean"]) / region_std_safe
        g["region_spread_zscore"] = g["region_spread_zscore"].fillna(0.0)

        # rolling short vs long volume trend
        short_mean = g.rolling(
            f"{SHORT_WINDOW}D", on="_feature_date", closed="left", min_periods=2
        )["n_transactions"].mean()
        long_mean = g.rolling(
            f"{BASELINE_WINDOW}D",
            on="_feature_date",
            closed="left",
            min_periods=MIN_HISTORY_DAYS,
        )["n_transactions"].mean()
        g["rolling_volume_trend"] = (short_mean / long_mean.replace(0, np.nan)).fillna(1.0)

        # Same-weekday volume gives both weekday/weekend normalization and a
        # merchant-specific seasonal expectation.  It requires two prior
        # matching weekdays; until then a neutral ratio prevents cold-start
        # rows from being over-interpreted.
        weekday_baseline = _weekday_volume_baseline(g)
        g["weekday_volume_ratio"] = (
            g["n_transactions"] / weekday_baseline.replace(0, np.nan)
        ).fillna(1.0)

        # Robust distributional position versus the *preceding* four weeks.
        g["txn_volume_percentile"] = _prior_volume_percentile(g)

        # EWM state deliberately uses only prior observations.  It complements
        # calendar-window statistics by reacting faster to sustained shifts.
        prior_volume = g["n_transactions"].shift(1)
        ewm_mean = prior_volume.ewm(
            span=EWM_SPAN, adjust=False, min_periods=MIN_HISTORY_DAYS
        ).mean()
        ewm_std = prior_volume.ewm(
            span=EWM_SPAN, adjust=False, min_periods=MIN_HISTORY_DAYS
        ).std()
        g["ewm_volume_zscore"] = (
            (g["n_transactions"] - ewm_mean) / ewm_std.replace(0, np.nan)
        ).fillna(0.0)

        short_std = g.rolling(
            f"{SHORT_WINDOW}D", on="_feature_date", closed="left", min_periods=2
        )["n_transactions"].std()
        long_std = g["txn_baseline_std"]
        g["recent_volume_volatility_ratio"] = (
            short_std / long_std.replace(0, np.nan)
        ).fillna(1.0)

        # Acceleration is current volume vs recent state.  The trend-adjusted
        # z-score asks whether today's level exceeds a short-horizon expected
        # trend, rather than treating legitimate growth as a full anomaly.
        g["txn_acceleration_ratio"] = (
            g["n_transactions"] / short_mean.replace(0, np.nan)
        ).fillna(1.0)
        g["trend_adjusted_volume_zscore"] = (
            (g["n_transactions"] - short_mean) / long_std.replace(0, np.nan)
        ).fillna(0.0)

        # A change-point state is historical: it compares two *prior* windows
        # and does not use today's value.  It detects a regime break that may
        # make simple long-baseline features temporarily less reliable.
        g["volume_change_point_score"] = (
            (short_mean - long_mean).abs() / long_std.replace(0, np.nan)
        ).fillna(0.0)

        # Historical anomaly frequency is based only on prior raw-signal
        # anomalies, never labels or model scores.  The current indicator is
        # excluded by the left-closed rolling window below.
        current_signal_anomaly = (
            (g["txn_volume_zscore"].abs() >= 2.5)
            | (g["failed_rate_delta"].abs() >= 0.10)
            | (g["amount_zscore"].abs() >= 2.5)
            | (g["region_spread_zscore"].abs() >= 2.5)
        ).astype(float)
        # ``current_signal_anomaly`` is a Series without a stable column name;
        # assign it explicitly before rolling to keep pandas column selection
        # deterministic across supported versions.
        g["_current_signal_anomaly"] = current_signal_anomaly.to_numpy()
        g["historical_anomaly_rate"] = g.rolling(
            f"{ANOMALY_HISTORY_WINDOW}D", on="_feature_date", closed="left", min_periods=1
        )["_current_signal_anomaly"].mean().fillna(0.0)

        g["has_sufficient_history"] = g["history_days"] >= MIN_HISTORY_DAYS
        g["feature_schema_version"] = FEATURE_SCHEMA_VERSION
        return g.drop(columns=["_feature_date", "_weekday", "_current_signal_anomaly"])

    # Avoid DataFrameGroupBy.apply: pandas' grouping-column behavior has
    # changed across releases.  Explicit concatenation preserves the key.
    feature_groups = [per_merchant(group) for _, group in df.groupby("merchant_id", sort=False)]
    df = pd.concat(feature_groups, ignore_index=True)
    if df.duplicated(["merchant_id", "date"]).any():
        raise ObservationValidationError("Feature output contains duplicate merchant-day keys.")

    # Clip extreme values to keep the model robust to rare aggregate outliers
    # without changing the neutral fallbacks used for sparse history.
    for col in [
        "txn_volume_zscore", "amount_zscore", "region_spread_zscore",
        "ewm_volume_zscore", "trend_adjusted_volume_zscore",
    ]:
        df[col] = df[col].clip(-10, 10)
    for col in [
        "amount_volatility_ratio", "rolling_volume_trend", "weekday_volume_ratio",
        "recent_volume_volatility_ratio", "txn_acceleration_ratio",
        "volume_change_point_score",
    ]:
        df[col] = df[col].clip(0, 10)
    df["txn_volume_percentile"] = df["txn_volume_percentile"].clip(0, 1)
    df["device_diversity_delta"] = df["device_diversity_delta"].clip(-1, 1)
    df["historical_anomaly_rate"] = df["historical_anomaly_rate"].clip(0, 1)

    return df


if __name__ == "__main__":
    raw = pd.read_csv(os.path.join(REPO_ROOT, "data/raw/merchant_daily_transactions.csv"), parse_dates=["date"])
    featured = build_features(raw)
    out_path = os.path.join(REPO_ROOT, "data/processed/features.csv")
    featured.to_csv(out_path, index=False)
    print(f"Built features for {len(featured)} rows")
    print(f"Rows with sufficient history: {featured.has_sufficient_history.sum()} / {len(featured)}")
    print(featured[FEATURE_COLUMNS + ['label_fraud_spike']].describe())
    print(f"Saved to {out_path}")
