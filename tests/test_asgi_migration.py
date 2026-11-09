"""Compatibility tests for the incremental FastAPI/ASGI gateway."""

from __future__ import annotations

import asyncio

import httpx
import pytest
from pydantic import ValidationError
from werkzeug.security import generate_password_hash

from backend.asgi import app, legacy_app_module
from backend.auth import issue_access_token
from backend.database import configure_database, create_schema, get_session
from backend.models import Membership, Organization, User
from backend.schemas import ErrorResponse, HealthResponse, PrincipalResponse, ReadinessChecks, ReadinessResponse, TokenResponse


def asgi_request(
    path: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    json: dict | None = None,
) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, path, headers=headers, json=json)

    return asyncio.run(request())


def asgi_get(path: str, *, headers: dict[str, str] | None = None) -> httpx.Response:
    return asgi_request(path, headers=headers)


@pytest.fixture
def native_principal(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'native-auth.db').as_posix()}")
    create_schema()
    config_keys = ("JWT_SECRET", "AUTH_RATE_LIMIT_PER_MINUTE", "RATE_LIMIT_WINDOW_SECONDS", "ALLOW_BOOTSTRAP_REGISTRATION")
    previous_config = {key: legacy_app_module.app.config.get(key) for key in config_keys}
    legacy_app_module.app.config.update(
        JWT_SECRET="native-asgi-test-secret-that-is-longer-than-thirty-two-characters",
        AUTH_RATE_LIMIT_PER_MINUTE=30,
        RATE_LIMIT_WINDOW_SECONDS=60,
        ALLOW_BOOTSTRAP_REGISTRATION=True,
    )
    legacy_app_module.app.extensions["rate_limiter"].clear()
    session = get_session()
    try:
        organization = Organization(name="Native Test Organization", slug="native-test")
        user = User(email="native-admin@example.test", password_hash=generate_password_hash("test-password"))
        session.add_all([organization, user])
        session.flush()
        membership = Membership(organization_id=organization.id, user_id=user.id, role="admin")
        session.add(membership)
        session.commit()
        with legacy_app_module.app.app_context():
            token = issue_access_token(user, membership)
        yield {"token": token, "organization_id": organization.id, "user_id": user.id}
    finally:
        session.close()
        legacy_app_module.app.extensions["rate_limiter"].clear()
        legacy_app_module.app.config.update(previous_config)
        configure_database()


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


def test_native_auth_me_revalidates_membership_and_uses_the_error_envelope(native_principal):
    anonymous = asgi_get("/api/v1/auth/me", headers={"X-Request-ID": "native-auth-request-1"})
    assert anonymous.status_code == 401
    assert ErrorResponse.model_validate(anonymous.json()).reason == "A Bearer access token is required."
    assert anonymous.headers["X-Request-ID"] == "native-auth-request-1"

    authenticated = asgi_get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {native_principal['token']}", "X-Request-ID": "native-auth-request-2"},
    )
    assert authenticated.status_code == 200
    principal = PrincipalResponse.model_validate(authenticated.json())
    assert principal.user_id == native_principal["user_id"]
    assert principal.organization_id == native_principal["organization_id"]
    assert principal.role == "admin"
    assert authenticated.headers["X-RateLimit-Remaining"] == "28"

    session = get_session()
    try:
        session.get(User, native_principal["user_id"]).is_active = False
        session.commit()
    finally:
        session.close()
    revoked = asgi_get("/api/v1/auth/me", headers={"Authorization": f"Bearer {native_principal['token']}"})
    assert revoked.status_code == 401
    assert ErrorResponse.model_validate(revoked.json()).reason == "Access token no longer has an active membership."


def test_native_registration_and_login_use_strict_typed_contracts(native_principal):
    payload = {
        "email": "new-admin@example.test",
        "password": "a-long-native-test-password",
        "organization_name": "New Native Organization",
        "organization_slug": "new-native-org",
    }
    created = asgi_request("/api/v1/auth/register", method="POST", json=payload)
    assert created.status_code == 201
    created_token = TokenResponse.model_validate(created.json())
    assert created_token.token_type == "Bearer"
    assert created_token.user.role == "admin"

    logged_in = asgi_request(
        "/api/v1/auth/login",
        method="POST",
        json={
            "email": payload["email"],
            "password": payload["password"],
            "organization_slug": payload["organization_slug"],
        },
    )
    assert logged_in.status_code == 200
    assert TokenResponse.model_validate(logged_in.json()).organization.slug == payload["organization_slug"]

    malformed = asgi_request(
        "/api/v1/auth/login",
        method="POST",
        json={"email": payload["email"], "password": payload["password"], "organization_slug": payload["organization_slug"], "extra": "no"},
    )
    assert malformed.status_code == 422
    assert ErrorResponse.model_validate(malformed.json()).reason == "Request validation failed."


def test_native_auth_cors_preflight_remains_allowlisted():
    allowed = asgi_request(
        "/api/v1/auth/me",
        method="OPTIONS",
        headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET"},
    )
    assert allowed.status_code == 200
    assert allowed.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"

    rejected = asgi_request(
        "/api/v1/auth/me",
        method="OPTIONS",
        headers={"Origin": "https://untrusted.example", "Access-Control-Request-Method": "GET"},
    )
    assert rejected.status_code == 400
    assert "Access-Control-Allow-Origin" not in rejected.headers


def test_native_auth_rate_limit_returns_a_controlled_error(native_principal, monkeypatch):
    monkeypatch.setitem(legacy_app_module.app.config, "AUTH_RATE_LIMIT_PER_MINUTE", 1)
    legacy_app_module.app.extensions["rate_limiter"].clear()

    first = asgi_get("/api/v1/auth/me")
    blocked = asgi_get("/api/v1/auth/me")

    assert first.status_code == 401
    assert blocked.status_code == 429
    assert ErrorResponse.model_validate(blocked.json()).reason == "Rate limit exceeded. Retry later."
    assert int(blocked.headers["Retry-After"]) >= 1
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


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
