"""Smoke coverage for the reproducible API benchmark harness."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.run_api_benchmark import main, run


def test_api_benchmark_is_explicitly_in_process_and_measures_tenant_reads():
    report = run(requests=2, warmup=1)
    assert report["benchmark_version"] == "phase7-api-inprocess-v1"
    assert "no TCP" in report["execution_scope"]
    assert report["endpoints"]["tenant_merchants_page"]["requests"] == 2
    assert report["endpoints"]["tenant_merchants_page"]["latency_ms"]["p95"] >= 0


def test_api_benchmark_cli_writes_a_json_report(tmp_path, monkeypatch):
    output = Path(tmp_path) / "api-benchmark.json"
    monkeypatch.setattr("sys.argv", ["run_api_benchmark.py", "--requests", "1", "--warmup", "0", "--output", str(output)])
    main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["requests_per_endpoint"] == 1
