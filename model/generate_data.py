"""CLI wrapper for the configurable synthetic merchant-day simulator.

Run from either the repository root (``python model/generate_data.py``) or
the ``model`` directory (``python generate_data.py``).  The default output is
the training CSV plus an offline-only provenance manifest; neither represents
real payment behaviour or real fraud outcomes.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import os

from simulator import SimulationConfig, generate_simulation


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "raw", "merchant_daily_transactions.csv")
DEFAULT_MANIFEST_PATH = os.path.join(REPO_ROOT, "data", "raw", "simulation_manifest.json")


def load_config(path: str | None) -> SimulationConfig:
    """Load a JSON config while rejecting unknown parameters early."""
    if path is None:
        return SimulationConfig()
    with open(path, encoding="utf-8") as handle:
        supplied = json.load(handle)
    if not isinstance(supplied, dict):
        raise ValueError("Simulation config JSON must be an object.")
    allowed = set(asdict(SimulationConfig()))
    unknown = sorted(set(supplied) - allowed)
    if unknown:
        raise ValueError(f"Unknown simulation config fields: {unknown}")
    if "holiday_day_offsets" in supplied:
        supplied["holiday_day_offsets"] = tuple(supplied["holiday_day_offsets"])
    return SimulationConfig(**supplied)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate offline synthetic merchant-day data.")
    parser.add_argument("--config", help="Path to a JSON object matching SimulationConfig.")
    parser.add_argument("--seed", type=int, help="Override the config seed.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="CSV output path.")
    parser.add_argument(
        "--manifest-output", default=DEFAULT_MANIFEST_PATH,
        help="Offline-only simulator provenance JSON output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    result = generate_simulation(config)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_output)), exist_ok=True)
    result.observations.to_csv(args.output, index=False)
    with open(args.manifest_output, "w", encoding="utf-8") as handle:
        json.dump(result.manifest(), handle, indent=2)

    summary = result.manifest()["summary"]
    print(f"Generated {summary['observed_merchant_days']} observed merchant-day rows for "
          f"{summary['merchants_observed']} merchants")
    print(f"Positive (fraud-spike) rate: {summary['positive_rate']:.4f}")
    print(f"Delayed records: {summary['delayed_records']}")
    print(f"Date range: {result.observations.date.min()} to {result.observations.date.max()}")
    print(f"Saved data to {args.output}")
    print(f"Saved offline provenance manifest to {args.manifest_output}")


# Backwards-compatible import surface for notebooks and existing callers.
def generate(config: SimulationConfig | None = None):
    return generate_simulation(config).observations


if __name__ == "__main__":
    main()
