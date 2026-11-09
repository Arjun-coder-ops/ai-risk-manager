"""Reproducible in-process baseline for selected authenticated API paths.

This is intentionally not a production load test: it uses Flask's test client
and temporary SQLite to make the route/auth/query cost repeatable without a
running network server. Its report labels that limitation explicitly, so it
cannot be mistaken for a Gunicorn/PostgreSQL capacity claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import tempfile
import time
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Match the application's established import path so benchmark/test execution
# does not create a second Flask app object with divergent configuration.
import app as application  # noqa: E402
from backend.database import configure_database, create_schema, get_session  # noqa: E402
from backend.models import Merchant  # noqa: E402


def percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile without a NumPy dependency."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("At least one latency sample is required.")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def measure(name: str, call: Callable[[], object], *, warmup: int, requests: int) -> dict:
    for _ in range(warmup):
        response = call()
        assert getattr(response, "status_code") == 200, f"{name} warmup failed: {response!r}"
    samples_ms: list[float] = []
    for _ in range(requests):
        started = time.perf_counter()
        response = call()
        elapsed_ms = (time.perf_counter() - started) * 1000
        assert getattr(response, "status_code") == 200, f"{name} failed: {response!r}"
        samples_ms.append(elapsed_ms)
    total_seconds = sum(samples_ms) / 1000
    return {
        "requests": requests,
        "latency_ms": {
            "min": round(min(samples_ms), 4),
            "p50": round(percentile(samples_ms, 0.50), 4),
            "p95": round(percentile(samples_ms, 0.95), 4),
            "max": round(max(samples_ms), 4),
        },
        "sequential_throughput_requests_per_second": round(requests / total_seconds, 4),
    }


def run(*, requests: int, warmup: int) -> dict:
    """Exercise health, contract, and an authenticated tenant DB list route."""
    if requests < 1 or warmup < 0:
        raise ValueError("requests must be positive and warmup must not be negative.")
    with tempfile.TemporaryDirectory(prefix="ai-risk-api-benchmark-") as directory:
        database_path = Path(directory) / "benchmark.db"
        configure_database(f"sqlite:///{database_path.as_posix()}")
        create_schema()
        config_keys = ("TESTING", "JWT_SECRET", "ALLOW_BOOTSTRAP_REGISTRATION")
        previous_config = {key: application.app.config.get(key) for key in config_keys}
        application.app.config.update(
            TESTING=True,
            JWT_SECRET="benchmark-secret-that-is-longer-than-thirty-two-characters",
            ALLOW_BOOTSTRAP_REGISTRATION=True,
        )
        application.app.extensions["rate_limiter"].clear()
        try:
            with application.app.test_client() as client:
                registration = client.post("/api/v1/auth/register", json={
                    "email": "benchmark-admin@example.test",
                    "password": "benchmark-password-123",
                    "organization_name": "Benchmark Organization",
                    "organization_slug": "benchmark-organization",
                })
                assert registration.status_code == 201, registration.get_json()
                principal = registration.get_json()
                session = get_session()
                try:
                    session.add_all([
                        Merchant(
                            organization_id=principal["organization"]["id"],
                            external_id=f"benchmark-{index:03d}",
                            display_name=f"Benchmark {index:03d}",
                        )
                        for index in range(50)
                    ])
                    session.commit()
                finally:
                    session.close()
                headers = {"Authorization": f"Bearer {principal['access_token']}"}
                endpoints = {
                    "liveness_health": measure("liveness_health", lambda: client.get("/health"), warmup=warmup, requests=requests),
                    "openapi_contract": measure("openapi_contract", lambda: client.get("/api/v1/openapi.json"), warmup=warmup, requests=requests),
                    "tenant_merchants_page": measure("tenant_merchants_page", lambda: client.get("/api/v1/merchants?page=1&limit=50", headers=headers), warmup=warmup, requests=requests),
                }
        finally:
            application.app.extensions["rate_limiter"].clear()
            application.app.config.update(previous_config)
            configure_database()
    return {
        "benchmark_version": "phase7-api-inprocess-v1",
        "execution_scope": "Flask test client with temporary SQLite; no TCP, Gunicorn, PostgreSQL, concurrent clients, or network latency.",
        "requests_per_endpoint": requests,
        "warmup_requests_per_endpoint": warmup,
        "seeded_merchants": 50,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "endpoints": endpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation" / "api_benchmark.json")
    args = parser.parse_args()
    result = run(requests=args.requests, warmup=args.warmup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
