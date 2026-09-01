"""
Synthetic merchant transaction data generator.

WHY SYNTHETIC DATA (disclosed honestly, per project requirements):
Razorpay does not expose a public dataset of real merchant transactions or
fraud-spike labels, and no Razorpay test-mode API returns historical
merchant-level transaction streams. Public payment-fraud datasets that do
exist (e.g. card-level Kaggle sets) are transaction-level and unlabeled for
"merchant fraud spikes" specifically, so they don't fit this problem shape.
We therefore generate synthetic merchant-day transaction data designed to
resemble realistic merchant behaviour, and we are explicit about this in the
README. No evaluation result in this project is based on real Razorpay data.

METHODOLOGY:
- We simulate N merchants over T days.
- Each merchant has a baseline daily transaction volume, failure rate,
  average order value, and device/user churn rate, drawn from merchant-
  category-informed distributions (small/medium/large merchant archetypes).
- Normal days are generated with Poisson/Gamma noise around the baseline.
- A subset of (merchant, day-window) pairs are injected with a "fraud spike"
  event: a multi-day period where one or more signals deviate sharply from
  that merchant's own baseline (not from a global threshold), e.g. a burst
  of low-value high-velocity transactions from new devices, or a burst of
  failed authentication attempts. This mirrors how real fraud spikes show up
  as *merchant-relative* anomalies rather than one-off transaction outliers.
- Spike magnitude and duration are randomized so spikes are NOT trivially
  separable: some spikes are subtle (small deviation, short duration) and
  some are more obvious, producing a realistic classification difficulty.
- Class imbalance is enforced: only ~6-9% of merchant-days are labeled as
  part of a fraud-spike window, which is representative of real anomaly
  detection settings (fraud spikes are rare events, not the norm).
- Labels are attached at the (merchant, day) grain: label=1 if that day
  falls inside an injected spike window for that merchant.

This is a simulation, not a claim about real-world Razorpay fraud rates.
"""

import os
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from dataclasses import dataclass
from datetime import datetime, timedelta

RNG_SEED = 42
N_MERCHANTS = 150
N_DAYS = 180  # ~6 months of daily activity per merchant
START_DATE = datetime(2026, 1, 1)

MERCHANT_ARCHETYPES = [
    # (name, weight, base_txn_mean, base_txn_std, base_fail_rate, base_aov, aov_std, base_new_user_ratio)
    ("small", 0.55, 40, 15, 0.04, 650, 300, 0.15),
    ("medium", 0.32, 220, 60, 0.05, 1400, 700, 0.20),
    ("large", 0.13, 900, 220, 0.06, 2600, 1500, 0.25),
]

INDIAN_REGIONS = [
    "Maharashtra", "Karnataka", "Delhi-NCR", "Gujarat", "Tamil Nadu",
    "Telangana", "West Bengal", "Uttar Pradesh", "Rajasthan", "Kerala",
]


@dataclass
class Merchant:
    merchant_id: str
    archetype: str
    base_txn_mean: float
    base_fail_rate: float
    base_aov: float
    aov_std: float
    base_new_user_ratio: float
    home_region: str


def make_merchants(rng: np.random.Generator) -> list[Merchant]:
    archetypes, weights = zip(*[(a[0], a[1]) for a in MERCHANT_ARCHETYPES])
    lookup = {a[0]: a for a in MERCHANT_ARCHETYPES}
    merchants = []
    for i in range(N_MERCHANTS):
        arch_name = rng.choice(archetypes, p=weights)
        _, _, mean, std, fail_rate, aov, aov_std, new_user_ratio = lookup[arch_name]
        merchants.append(
            Merchant(
                merchant_id=f"M{i:04d}",
                archetype=arch_name,
                base_txn_mean=max(5, rng.normal(mean, std * 0.3)),
                base_fail_rate=float(np.clip(rng.normal(fail_rate, 0.01), 0.01, 0.25)),
                base_aov=max(50, rng.normal(aov, aov_std * 0.3)),
                aov_std=aov_std,
                base_new_user_ratio=float(np.clip(rng.normal(new_user_ratio, 0.03), 0.03, 0.6)),
                home_region=rng.choice(INDIAN_REGIONS),
            )
        )
    return merchants


def pick_spike_windows(rng: np.random.Generator, merchants: list[Merchant]) -> dict:
    """Assign each merchant 0-2 fraud-spike windows across the timeline.
    Target ~7% of merchant-days labeled positive overall."""
    spikes = {m.merchant_id: [] for m in merchants}
    target_positive_days = int(N_MERCHANTS * N_DAYS * 0.07)
    max_windows_per_merchant = 3
    assigned = 0
    safety_iterations = 0
    merchant_pool = list(merchants)
    while assigned < target_positive_days and safety_iterations < 20:
        rng.shuffle(merchant_pool)
        for m in merchant_pool:
            if assigned >= target_positive_days:
                break
            if len(spikes[m.merchant_id]) >= max_windows_per_merchant:
                continue
            if rng.random() < 0.5:  # not every merchant gets a spike each pass
                continue
            duration = int(rng.integers(2, 8))  # 2-7 day spike window
            start_day = int(rng.integers(10, N_DAYS - duration - 5))
            # skip if it overlaps an existing window for this merchant
            existing = spikes[m.merchant_id]
            if any(not (start_day + duration <= s["start"] or start_day >= s["start"] + s["duration"]) for s in existing):
                continue
            severity = rng.uniform(0.35, 1.0)  # subtle -> obvious
            spike_type = rng.choice(
                ["volume_burst", "failed_payment_burst", "velocity_new_device", "amount_anomaly"]
            )
            spikes[m.merchant_id].append(
                {"start": start_day, "duration": duration, "severity": severity, "type": spike_type}
            )
            assigned += duration
        safety_iterations += 1
    return spikes


def in_spike(spikes_for_merchant, day):
    for s in spikes_for_merchant:
        if s["start"] <= day < s["start"] + s["duration"]:
            return s
    return None


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    merchants = make_merchants(rng)
    spikes = pick_spike_windows(rng, merchants)

    rows = []
    for m in merchants:
        # mild weekly seasonality + slow organic growth trend per merchant
        growth = rng.uniform(-0.0005, 0.0015)
        for day in range(N_DAYS):
            date = START_DATE + timedelta(days=day)
            dow_factor = 1.15 if date.weekday() in (5, 6) else 1.0
            trend_factor = 1 + growth * day

            spike = in_spike(spikes[m.merchant_id], day)

            txn_mean = m.base_txn_mean * dow_factor * trend_factor
            fail_rate = m.base_fail_rate
            aov = m.base_aov
            new_user_ratio = m.base_new_user_ratio
            unique_devices_ratio = 0.85  # devices per user, normal
            distinct_regions = 1  # normally transactions cluster in home region

            label = 0
            if spike is not None:
                label = 1
                sev = spike["severity"]
                if spike["type"] == "volume_burst":
                    txn_mean *= (1 + sev * rng.uniform(2.0, 5.0))
                elif spike["type"] == "failed_payment_burst":
                    fail_rate = float(np.clip(fail_rate + sev * rng.uniform(0.15, 0.45), 0, 0.9))
                    txn_mean *= (1 + sev * 0.6)
                elif spike["type"] == "velocity_new_device":
                    new_user_ratio = float(np.clip(new_user_ratio + sev * rng.uniform(0.3, 0.6), 0, 0.95))
                    txn_mean *= (1 + sev * rng.uniform(0.8, 2.2))
                    distinct_regions = 1 + int(sev * rng.integers(2, 6))
                elif spike["type"] == "amount_anomaly":
                    aov *= rng.choice([1, -1]) * sev * rng.uniform(0.6, 2.5) + aov  # will normalize below
                    aov = max(20, m.base_aov * (1 + rng.choice([-1, 1]) * sev * rng.uniform(0.7, 2.0)))

            n_txns = max(1, int(rng.poisson(max(1, txn_mean))))
            n_failed = int(np.clip(rng.binomial(n_txns, fail_rate), 0, n_txns))
            amounts = rng.gamma(shape=2.0, scale=max(10, aov / 2), size=n_txns)
            avg_amount = float(np.mean(amounts)) if n_txns else 0.0
            amount_std = float(np.std(amounts)) if n_txns else 0.0
            n_unique_users = max(1, int(n_txns * rng.uniform(0.6, 0.9)))
            n_new_users = int(np.clip(rng.binomial(max(n_unique_users, 1), new_user_ratio), 0, n_unique_users))
            n_devices = max(1, int(n_unique_users * unique_devices_ratio))

            rows.append(
                {
                    "merchant_id": m.merchant_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "archetype": m.archetype,
                    "home_region": m.home_region,
                    "n_transactions": n_txns,
                    "n_failed": n_failed,
                    "avg_amount": round(avg_amount, 2),
                    "amount_std": round(amount_std, 2),
                    "n_unique_users": n_unique_users,
                    "n_new_users": n_new_users,
                    "n_devices": n_devices,
                    "distinct_regions_seen": distinct_regions,
                    "label_fraud_spike": label,
                }
            )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["merchant_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    df = generate()
    out_path = os.path.join(REPO_ROOT, "data/raw/merchant_daily_transactions.csv")
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} merchant-day rows for {df.merchant_id.nunique()} merchants")
    print(f"Positive (fraud-spike) rate: {df.label_fraud_spike.mean():.4f}")
    print(f"Date range: {df.date.min().date()} to {df.date.max().date()}")
    print(f"Saved to {out_path}")
