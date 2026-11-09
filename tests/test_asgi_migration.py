"""Compatibility tests for the incremental FastAPI/ASGI gateway."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError

from backend.asgi import app
from backend.schemas import ErrorResponse, ReadinessChecks, ReadinessResponse


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
    response = asgi_get("/health")
    assert response.status_code in {200, 503}
    assert response.json()["status"] in {"ok", "degraded"}


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
