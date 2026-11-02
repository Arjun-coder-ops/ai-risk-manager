"""Independent synthetic out-of-distribution (OOD) evaluation benchmark.

This module intentionally does *not* import ``model.simulator``.  It uses a
separate merchant population, heavy-tailed count/amount generation, structural
breaks, and different attack mechanisms to challenge the already-trained
model.  It is an offline synthetic benchmark, not external or real-world
validation, and must never be presented as such.

The benchmark scores the committed model without retraining it.  Its only
purpose is to measure distributional sensitivity beyond the in-distribution
simulator that supplied training data.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import date, timedelta
import json
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(REPO_ROOT, "model")
if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from features import FEATURE_COLUMNS, build_features  # noqa: E402
from validation import validate_observations  # noqa: E402


DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "evaluation", "ood_benchmark_results.json")
OOD_ATTACK_TYPES = ("credential_stuffing", "mule_network", "account_takeover")


@dataclass(frozen=True)
class OODConfig:
    """Configuration for a deterministic, deliberately independent benchmark."""

    seed: int = 20261102
    n_merchants: int = 110
    n_days: int = 150
    start_date: str = "2027-01-01"
    fraud_day_rate: float = 0.06
    max_attack_windows_per_merchant: int = 3

    def validate(self) -> None:
        if self.n_merchants < 5:
            raise ValueError("n_merchants must be at least 5.")
        if self.n_days < 40:
            raise ValueError("n_days must be at least 40.")
        if not 0 < self.fraud_day_rate < 0.4:
            raise ValueError("fraud_day_rate must be between 0 and 0.4.")
        if self.max_attack_windows_per_merchant < 1:
            raise ValueError("max_attack_windows_per_merchant must be at least 1.")
        try:
            date.fromisoformat(self.start_date)
        except ValueError as exc:
            raise ValueError("start_date must use YYYY-MM-DD format.") from exc


@dataclass(frozen=True)
class OODMerchant:
    merchant_id: str
    transaction_scale: float
    average_amount: float
    failure_rate: float
    user_ratio: float
    new_user_ratio: float
    device_ratio: float
    weekly_phase: float
    break_day: int
    break_factor: float


def _make_merchants(config: OODConfig, rng: np.random.Generator) -> list[OODMerchant]:
    """Create a heavy-tailed merchant population unlike simulator-v2 profiles."""
    merchants = []
    for index in range(config.n_merchants):
        merchants.append(OODMerchant(
            merchant_id=f"OOD{index:04d}",
            transaction_scale=float(np.clip(rng.lognormal(mean=4.7, sigma=1.05), 8, 3500)),
            average_amount=float(np.clip(rng.lognormal(mean=6.8, sigma=0.85), 30, 50000)),
            failure_rate=float(np.clip(rng.beta(1.5, 30), 0.002, 0.25)),
            user_ratio=float(rng.uniform(0.42, 0.96)),
            new_user_ratio=float(rng.uniform(0.03, 0.45)),
            device_ratio=float(rng.uniform(0.45, 1.0)),
            weekly_phase=float(rng.uniform(0, 2 * np.pi)),
            break_day=int(rng.integers(25, config.n_days - 20)),
            # A structural expansion or contraction is normal business drift,
            # not a fraud label; it is an intentional covariate shift.
            break_factor=float(rng.choice([rng.uniform(0.45, 0.80), rng.uniform(1.25, 2.20)])),
        ))
    return merchants


def _bound_row(row: dict[str, Any]) -> dict[str, Any]:
    """Guarantee the same raw-observation integrity contract as production."""
    transactions = max(0, int(round(float(row["n_transactions"]))))
    if transactions == 0:
        row.update({
            "n_transactions": 0, "n_failed": 0, "avg_amount": 0.0, "amount_std": 0.0,
            "n_unique_users": 0, "n_new_users": 0, "n_devices": 0,
            "distinct_regions_seen": 0,
        })
        return row
    users = min(transactions, max(1, int(round(float(row["n_unique_users"])))))
    row.update({
        "n_transactions": transactions,
        "n_failed": min(transactions, max(0, int(round(float(row["n_failed"]))))),
        "n_unique_users": users,
        "n_new_users": min(users, max(0, int(round(float(row["n_new_users"]))))),
        "n_devices": min(transactions, max(1, int(round(float(row["n_devices"]))))),
        "distinct_regions_seen": min(transactions, max(1, int(round(float(row["distinct_regions_seen"]))))),
        "avg_amount": round(max(0.0, float(row["avg_amount"])), 2),
        "amount_std": round(max(0.0, float(row["amount_std"])), 2),
    })
    return row


def _normal_ood_row(merchant: OODMerchant, day_index: int, current_day: date, rng: np.random.Generator) -> dict[str, Any]:
    """Generate normal OOD behaviour using negative-binomial, not Poisson, counts."""
    weekday_cycle = 1.0 + 0.32 * np.cos(2 * np.pi * current_day.weekday() / 7 + merchant.weekly_phase)
    monthly_cycle = 1.0 + 0.18 * np.sin(2 * np.pi * day_index / 29.5 + merchant.weekly_phase / 2)
    trend = max(0.25, 1.0 + 0.0015 * day_index)
    structural_shift = merchant.break_factor if day_index >= merchant.break_day else 1.0
    demand = merchant.transaction_scale * weekday_cycle * monthly_cycle * trend * structural_shift

    # Distinct, legitimate demand patterns serve as label-negative OOD
    # lookalikes.  They are not represented in the training simulator.
    if day_index % 37 in (0, 1, 2):
        demand *= float(rng.uniform(1.5, 3.2))  # recurring marketplace promotion
    if day_index % 53 in (15, 16):
        demand *= float(rng.uniform(0.10, 0.45))  # operational outage/quiet period

    dispersion = 9
    probability = dispersion / (dispersion + max(demand, 0.01))
    transactions = int(rng.negative_binomial(dispersion, probability))
    row: dict[str, Any] = {
        "merchant_id": merchant.merchant_id,
        "date": current_day.isoformat(),
        "label_fraud_spike": 0,
        "n_transactions": transactions,
        "n_failed": 0,
        "avg_amount": 0.0,
        "amount_std": 0.0,
        "n_unique_users": 0,
        "n_new_users": 0,
        "n_devices": 0,
        "distinct_regions_seen": 0,
    }
    if transactions == 0:
        return _bound_row(row)

    failure_rate = min(0.75, merchant.failure_rate * rng.lognormal(mean=0, sigma=0.18))
    users = max(1, int(transactions * np.clip(rng.normal(merchant.user_ratio, 0.08), 0.15, 1.0)))
    new_users = int(rng.binomial(users, np.clip(rng.normal(merchant.new_user_ratio, 0.06), 0.0, 0.95)))
    devices = max(1, int(users * np.clip(rng.normal(merchant.device_ratio, 0.10), 0.08, 1.0)))
    average_amount = merchant.average_amount * rng.lognormal(mean=0, sigma=0.12)
    row.update({
        "n_failed": int(rng.binomial(transactions, failure_rate)),
        "avg_amount": average_amount,
        "amount_std": average_amount * abs(rng.normal(0.72, 0.18)),
        "n_unique_users": users,
        "n_new_users": new_users,
        "n_devices": devices,
        "distinct_regions_seen": 1 + int(rng.poisson(0.25)),
    })
    return _bound_row(row)


def _apply_attack(row: dict[str, Any], attack_type: str, severity: float, rng: np.random.Generator) -> dict[str, Any]:
    """Apply independent OOD attack mechanics to one already-normal row."""
    transactions = max(1, int(row["n_transactions"]))
    if attack_type == "credential_stuffing":
        transactions = int(round(transactions * (1.0 + severity * rng.uniform(1.2, 3.5))))
        row["n_failed"] = transactions * min(0.98, 0.15 + severity * rng.uniform(0.45, 0.75))
        row["n_unique_users"] = transactions * min(0.90, 0.25 + severity * 0.45)
        row["n_new_users"] = row["n_unique_users"] * min(0.98, 0.50 + severity * 0.40)
        row["n_devices"] = row["n_unique_users"] * max(0.06, 0.40 - severity * 0.25)
    elif attack_type == "mule_network":
        transactions = int(round(transactions * (1.0 + severity * rng.uniform(0.5, 2.0))))
        row["n_unique_users"] = transactions * min(0.95, 0.35 + severity * 0.45)
        row["n_new_users"] = row["n_unique_users"] * min(0.98, 0.35 + severity * 0.55)
        row["n_devices"] = row["n_unique_users"] * max(0.08, 0.32 - severity * 0.20)
        row["distinct_regions_seen"] = 2 + severity * rng.integers(4, 12)
        row["avg_amount"] *= 1.0 + severity * rng.uniform(0.6, 1.6)
        row["amount_std"] *= 1.0 + severity * rng.uniform(0.8, 2.2)
    elif attack_type == "account_takeover":
        transactions = int(round(transactions * (1.0 + severity * rng.uniform(0.25, 1.4))))
        row["n_failed"] = transactions * min(0.85, 0.05 + severity * rng.uniform(0.15, 0.45))
        row["n_unique_users"] = transactions * min(0.90, 0.35 + severity * 0.35)
        row["n_new_users"] = row["n_unique_users"] * min(0.96, 0.20 + severity * 0.45)
        row["n_devices"] = row["n_unique_users"] * min(1.0, 0.75 + severity * 0.20)
        row["avg_amount"] *= 1.0 + severity * rng.uniform(0.5, 1.9)
    else:  # pragma: no cover - protects future callers, not random sampling.
        raise ValueError(f"Unsupported OOD attack type: {attack_type}")
    row["n_transactions"] = transactions
    return _bound_row(row)


def generate_ood_observations(config: OODConfig | None = None) -> pd.DataFrame:
    """Generate independent OOD observations with offline synthetic labels."""
    config = config or OODConfig()
    config.validate()
    rng = np.random.default_rng(config.seed)
    merchants = _make_merchants(config, rng)
    first_day = date.fromisoformat(config.start_date)
    rows = [
        _normal_ood_row(merchant, day_index, first_day + timedelta(days=day_index), rng)
        for merchant in merchants
        for day_index in range(config.n_days)
    ]
    output = pd.DataFrame(rows).sort_values(["merchant_id", "date"]).reset_index(drop=True)

    target = int(round(len(output) * config.fraud_day_rate))
    selected_indices: set[int] = set()
    windows_per_merchant = {merchant.merchant_id: 0 for merchant in merchants}
    grouped_indices = {
        merchant_id: group.index.to_list()
        for merchant_id, group in output.groupby("merchant_id", sort=False)
    }
    attempts = 0
    max_attempts = config.n_merchants * config.max_attack_windows_per_merchant * 40
    while len(selected_indices) < target and attempts < max_attempts:
        attempts += 1
        merchant_id = str(rng.choice(list(grouped_indices)))
        indices = grouped_indices[merchant_id]
        if windows_per_merchant[merchant_id] >= config.max_attack_windows_per_merchant:
            continue
        duration = int(rng.integers(2, min(7, len(indices) - 10) + 1))
        start = int(rng.integers(10, len(indices) - duration + 1))
        attack_indices = indices[start:start + duration]
        if any(index in selected_indices for index in attack_indices):
            continue
        attack_type = str(rng.choice(OOD_ATTACK_TYPES))
        severity = float(rng.uniform(0.30, 1.0))
        for index in attack_indices:
            row = _apply_attack(output.loc[index].to_dict(), attack_type, severity, rng)
            output.loc[index, list(row)] = list(row.values())
            output.loc[index, "label_fraud_spike"] = 1
            selected_indices.add(index)
        windows_per_merchant[merchant_id] += 1

    if len(selected_indices) < target:
        raise RuntimeError("Unable to schedule enough OOD attack windows for this configuration.")
    return validate_observations(output).sort_values(["merchant_id", "date"]).reset_index(drop=True)


def _threshold_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "threshold": threshold,
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if fp + tn else None,
        "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
    }


def run_ood_benchmark(config: OODConfig | None = None) -> dict[str, Any]:
    """Score the frozen in-distribution model on a separately generated OOD set."""
    config = config or OODConfig()
    observations = generate_ood_observations(config)
    featured = build_features(observations)
    scored = featured[featured["has_sufficient_history"]].copy()
    if scored["label_fraud_spike"].nunique() != 2:
        raise RuntimeError("OOD scored set must contain both positive and negative labels.")

    model_path = os.path.join(REPO_ROOT, "model", "artifacts", "model.joblib")
    metadata_path = os.path.join(REPO_ROOT, "model", "artifacts", "model_meta.json")
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = joblib.load(model_path)
    scores = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    labels = scored["label_fraud_spike"].to_numpy()

    return {
        "benchmark_name": "independent-synthetic-ood-v1",
        "model_version": metadata["model_version"],
        "model_type": metadata["model_type"],
        "provenance": {
            "source": "evaluation/ood_benchmark.py",
            "type": "independently generated synthetic OOD benchmark",
            "real_world_validation": False,
            "training_data_used": False,
            "disclosure": (
                "This is a separately designed synthetic stress test. It is not an external "
                "dataset and must not be interpreted as real-world fraud performance."
            ),
        },
        "config": asdict(config),
        "raw_merchant_days": int(len(observations)),
        "scored_merchant_days": int(len(scored)),
        "positive_rate": round(float(labels.mean()), 6),
        "roc_auc": round(float(roc_auc_score(labels, scores)), 4),
        "pr_auc": round(float(average_precision_score(labels, scores)), 4),
        "review_tier": _threshold_metrics(labels, scores, float(metadata["review_threshold"])),
        "high_tier": _threshold_metrics(labels, scores, float(metadata["high_threshold"])),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the isolated synthetic OOD benchmark.")
    parser.add_argument("--seed", type=int, help="Override the benchmark seed.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="JSON report output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = OODConfig()
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    report = run_ood_benchmark(config)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved OOD report to {args.output}")


if __name__ == "__main__":
    main()
