"""Focused operational HTTP controls for Phase 7."""

from __future__ import annotations

import os
import re
import sys

import pytest


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app as application  # noqa: E402
from backend.request_controls import FixedWindowRateLimiter, request_id_from_header  # noqa: E402
from backend.database import configure_database, create_schema  # noqa: E402


@pytest.fixture
def client(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'operations.db').as_posix()}")
    create_schema()
    config_keys = ("TESTING", "JWT_SECRET", "ALLOW_BOOTSTRAP_REGISTRATION")
    previous_config = {key: application.app.config.get(key) for key in config_keys}
    application.app.config.update(
        TESTING=True,
        JWT_SECRET="test-secret-that-is-longer-than-thirty-two-characters",
    )
    application.app.extensions["rate_limiter"].clear()
    with application.app.test_client() as test_client:
        yield test_client
    application.app.extensions["rate_limiter"].clear()
    application.app.config.update(previous_config)
    configure_database()


def test_request_id_accepts_safe_client_value_and_replaces_unsafe_value(client):
    safe = client.get("/health", headers={"X-Request-ID": "trace-12345678"})
    assert safe.status_code in {200, 503}
    assert safe.headers["X-Request-ID"] == "trace-12345678"

    unsafe = client.get("/health", headers={"X-Request-ID": "<script>"})
    assert unsafe.status_code in {200, 503}
    assert unsafe.headers["X-Request-ID"] != "<script>"
    assert request_id_from_header("trace-12345678") == "trace-12345678"


def test_fixed_window_limiter_has_stable_retry_boundary():
    limiter = FixedWindowRateLimiter()
    assert limiter.check("key", limit=2, window_seconds=60, now=120.0).allowed
    assert limiter.check("key", limit=2, window_seconds=60, now=120.5).remaining == 0
    blocked = limiter.check("key", limit=2, window_seconds=60, now=121.0)
    assert not blocked.allowed
    assert blocked.retry_after_seconds == 59
    assert limiter.check("key", limit=2, window_seconds=60, now=180.0).allowed


def test_auth_rate_limit_returns_controlled_correlation_aware_429(client, monkeypatch):
    monkeypatch.setitem(application.app.config, "AUTH_RATE_LIMIT_PER_MINUTE", 2)
    for _ in range(2):
        response = client.post("/api/v1/auth/login", json={})
        assert response.status_code in {400, 401}
    blocked = client.post("/api/v1/auth/login", json={})
    assert blocked.status_code == 429
    assert blocked.get_json()["status"] == "error"
    assert blocked.get_json()["request_id"] == blocked.headers["X-Request-ID"]
    assert blocked.headers["Retry-After"]


def test_authentication_error_includes_the_request_correlation_id(client):
    response = client.get("/api/v1/auth/me", headers={"X-Request-ID": "request-12345678"})
    assert response.status_code == 401
    assert response.get_json()["request_id"] == "request-12345678"


def test_unknown_api_endpoint_returns_a_controlled_correlation_aware_404(client):
    response = client.get("/api/v1/not-a-real-endpoint", headers={"X-Request-ID": "request-87654321"})
    assert response.status_code == 404
    assert response.get_json() == {
        "status": "error",
        "reason": "API endpoint was not found.",
        "request_id": "request-87654321",
    }


def test_versioned_openapi_contract_excludes_the_synthetic_demo_routes(client):
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    contract = response.get_json()
    assert contract["openapi"] == "3.1.0"
    assert contract["info"]["version"] == "1.0.0"
    assert "/api/v1/risk-assessments" in contract["paths"]
    assert "/api/risk/analyze" not in contract["paths"]
    assert contract["components"]["schemas"]["Principal"]["properties"]["role"]["enum"] == [
        "admin", "analyst", "investigator", "viewer",
    ]
    documented_paths = set(contract["paths"])
    registered_tenant_paths = {
        re.sub(r"<([^>]+)>", r"{\1}", rule.rule) for rule in application.app.url_map.iter_rules()
        if rule.rule.startswith("/api/v1/") and rule.rule != "/api/v1/openapi.json"
    }
    assert documented_paths == registered_tenant_paths
    registered_methods_by_path: dict[str, set[str]] = {}
    for rule in application.app.url_map.iter_rules():
        if not rule.rule.startswith("/api/v1/") or rule.rule == "/api/v1/openapi.json":
            continue
        path = re.sub(r"<([^>]+)>", r"{\1}", rule.rule)
        registered_methods_by_path.setdefault(path, set()).update(
            method.lower() for method in rule.methods - {"HEAD", "OPTIONS"}
        )
    for path, registered_methods in registered_methods_by_path.items():
        documented_methods = set(contract["paths"][path])
        assert documented_methods == registered_methods


def test_readiness_requires_the_risk_engine_even_when_database_state_is_unknown(client, monkeypatch):
    monkeypatch.setattr(application, "engine", None)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.get_json()["status"] == "not_ready"
    assert response.get_json()["checks"]["risk_engine"] is False
