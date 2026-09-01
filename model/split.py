"""
Leakage-resistant time-based split.

We split by DATE, not by row, so no future information ever leaks into
training or validation. Every merchant appears in all three splits (this is
"new time period for a known merchant", the realistic production setting:
a merchant's history keeps accumulating and we score its new days).

Timeline (180 days total, 2026-01-01 to 2026-06-29):
  Train:      2026-01-01 -> 2026-04-15  (~105 days, ~58%)
  Validation: 2026-04-16 -> 2026-05-15  (~30 days,  ~17%)  <- used ONLY to pick threshold + compare models
  Test:       2026-05-16 -> 2026-06-29  (~45 days,  ~25%)  <- touched exactly once, for final reporting

The first MIN_HISTORY_DAYS of each merchant's timeline are dropped from all
splits because their rolling-baseline features are not yet reliable
(insufficient history) -- see features.py `has_sufficient_history`.
"""

import os
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_END = "2026-04-15"
VAL_END = "2026-05-15"


def time_split(df: pd.DataFrame):
    df = df[df["has_sufficient_history"]].copy()
    train = df[df["date"] <= TRAIN_END]
    val = df[(df["date"] > TRAIN_END) & (df["date"] <= VAL_END)]
    test = df[df["date"] > VAL_END]
    return train, val, test


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(REPO_ROOT, "data/processed/features.csv"), parse_dates=["date"])
    train, val, test = time_split(df)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        print(f"{name}: {len(split)} rows, {split.date.min().date()} -> {split.date.max().date()}, "
              f"positive rate={split.label_fraud_spike.mean():.4f}")
