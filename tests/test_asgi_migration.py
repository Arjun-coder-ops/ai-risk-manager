"""Compatibility tests for the incremental FastAPI/ASGI gateway."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from backend.asgi import app, legacy_app_module
from backend.schemas import ErrorResponse, HealthResponse, ReadinessChecks, ReadinessResponse


def asgi_get(path: str) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get(path)

    return asyncio.run(request())


def test_fastapi_gateway_serves_swagger_for_the_existing_tenant_contract():
    docs = asgi_get("/api/v1/docs")
    assert docs.status_code == 200
    assert "/api/v1/openapi.json" in docs.text

    contract = asgi_get("/api/v1/openapi.json")
    assert contract.status_code == 200
    assert contract.json()["openapi"] == "3.1.0"


def test_fastapi_gateway_preserves_flask_liveness_route():
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.get("/health", headers={"X-Request-ID": "native-asgi-12345678"})

    response = asyncio.run(request())
    assert response.status_code in {200, 503}
    health = HealthResponse.model_validate(response.json())
    assert health.status in {"ok", "degraded"}
    if health.status == "degraded":
        assert "model_version" not in response.json()
    assert response.headers["X-Request-ID"] == "native-asgi-12345678"


def test_native_fastapi_readiness_uses_the_strict_response_contract():
    response = asgi_get("/ready")
    assert response.status_code in {200, 503}
    validated = ReadinessResponse.model_validate(response.json())
    assert isinstance(validated.checks.risk_engine, bool)
    assert isinstance(validated.checks.database, bool)


def test_native_degraded_health_preserves_the_legacy_response_shape(monkeypatch):
    monkeypatch.setattr(legacy_app_module, "engine", None)

    response = asgi_get("/health")

    assert response.status_code == 503
    assert response.json() == {"status": "degraded"}


def test_migration_response_schemas_reject_undocumented_fields():
    readiness = ReadinessResponse(
        status="ready",
        checks=ReadinessChecks(risk_engine=True, database=True),
    )
    assert readiness.model_dump() == {
        "status": "ready",
        "checks": {"risk_engine": True, "database": True},
    }
    with pytest.raises(ValidationError):
        ErrorResponse(status="error", reason="Bad request.", request_id="request-12345678", internal_trace="no")
