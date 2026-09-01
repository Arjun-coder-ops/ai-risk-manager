"""
Feature engineering for merchant-day fraud-spike detection.

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

Leakage control: all rolling statistics use `.shift(1)` before rolling, so
the baseline for day T only ever uses days < T. No feature ever looks at
day T's own label or at future days.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_WINDOW = 14
SHORT_WINDOW = 3
MIN_HISTORY_DAYS = 7  # below this we don't have enough baseline to trust a score

FEATURE_COLUMNS = [
    "txn_volume_zscore",
    "failed_rate_delta",
    "amount_zscore",
    "amount_volatility_ratio",
    "new_user_ratio_delta",
    "device_diversity_ratio",
    "region_spread_zscore",
    "rolling_volume_trend",
]


def _safe_div(a, b, default=0.0):
    return np.where(b > 0, a / b.replace(0, np.nan), default)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["merchant_id", "date"]).copy()

    df["failed_rate"] = df["n_failed"] / df["n_transactions"].replace(0, np.nan)
    df["failed_rate"] = df["failed_rate"].fillna(0.0)
    df["new_user_ratio"] = df["n_new_users"] / df["n_unique_users"].replace(0, np.nan)
    df["new_user_ratio"] = df["new_user_ratio"].fillna(0.0)
    df["device_per_user"] = df["n_devices"] / df["n_unique_users"].replace(0, np.nan)
    df["device_per_user"] = df["device_per_user"].fillna(1.0)

    grouped = df.groupby("merchant_id", group_keys=False)

    def per_merchant(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("date").copy()
        g["history_days"] = np.arange(len(g))  # 0-indexed count of prior days available

        for col, out_prefix in [
            ("n_transactions", "txn"),
            ("failed_rate", "fail"),
            ("avg_amount", "amt"),
            ("amount_std", "amtstd"),
            ("new_user_ratio", "newuser"),
            ("distinct_regions_seen", "region"),
        ]:
            shifted = g[col].shift(1)
            roll_mean = shifted.rolling(BASELINE_WINDOW, min_periods=MIN_HISTORY_DAYS).mean()
            roll_std = shifted.rolling(BASELINE_WINDOW, min_periods=MIN_HISTORY_DAYS).std()
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

        # device diversity ratio (devices per user vs its own recent norm ~0.85-1.0, so raw value is informative)
        g["device_diversity_ratio"] = g["device_per_user"]

        # region spread z-score
        region_std_safe = g["region_baseline_std"].replace(0, np.nan)
        g["region_spread_zscore"] = (g["distinct_regions_seen"] - g["region_baseline_mean"]) / region_std_safe
        g["region_spread_zscore"] = g["region_spread_zscore"].fillna(0.0)

        # rolling short vs long volume trend
        shifted_txn = g["n_transactions"].shift(1)
        short_mean = shifted_txn.rolling(SHORT_WINDOW, min_periods=2).mean()
        long_mean = shifted_txn.rolling(BASELINE_WINDOW, min_periods=MIN_HISTORY_DAYS).mean()
        g["rolling_volume_trend"] = (short_mean / long_mean.replace(0, np.nan)).fillna(1.0)

        g["has_sufficient_history"] = g["history_days"] >= MIN_HISTORY_DAYS
        return g

    df = grouped.apply(per_merchant, include_groups=False).reset_index(level=0)
    if "merchant_id" not in df.columns:
        df = df.rename(columns={"level_0": "merchant_id"})

    # clip extreme z-scores to keep the model robust to rare outliers
    for col in ["txn_volume_zscore", "amount_zscore", "region_spread_zscore"]:
        df[col] = df[col].clip(-10, 10)
    df["amount_volatility_ratio"] = df["amount_volatility_ratio"].clip(0, 10)
    df["rolling_volume_trend"] = df["rolling_volume_trend"].clip(0, 10)

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
