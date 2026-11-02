"""Integration tests for the authenticated, tenant-scoped durable API."""

from __future__ import annotations

from datetime import date, timedelta
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app as application  # noqa: E402
from backend.database import configure_database, create_schema  # noqa: E402


@pytest.fixture
def client(tmp_path):
    configure_database(f"sqlite:///{(tmp_path / 'tenant-api.db').as_posix()}")
    create_schema()
    application.app.config.update(TESTING=True, JWT_SECRET="test-secret-that-is-longer-than-thirty-two-characters")
    with application.app.test_client() as test_client:
        yield test_client
    configure_database()


def register(client, suffix: str):
    response = client.post("/api/v1/auth/register", json={
        "email": f"admin-{suffix}@example.test",
        "password": "a-long-test-password-123",
        "organization_name": f"Test Organization {suffix}",
        "organization_slug": f"test-org-{suffix}",
    })
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def auth_headers(token: str, idempotency_key: str | None = None):
    headers = {"Authorization": f"Bearer {token}"}
    if idempotency_key:
        headers["Idempotency-Key"] = idempotency_key
    return headers


def observations(merchant_id: str = "M0001", days: int = 8):
    start = date(2026, 1, 1)
    return [{
        "merchant_id": merchant_id,
        "date": (start + timedelta(days=offset)).isoformat(),
        "n_transactions": 100 + offset,
        "n_failed": 3,
        "avg_amount": 125.0,
        "amount_std": 30.0,
        "n_unique_users": 80,
        "n_new_users": 10,
        "n_devices": 70,
        "distinct_regions_seen": 2,
    } for offset in range(days)]


class TestTenantAuthAndIngestion:
    def test_tenant_routes_require_a_valid_bearer_token(self, client):
        response = client.get("/api/v1/merchants")
        assert response.status_code == 401
        assert "Bearer" in response.get_json()["reason"]

    def test_allowed_cors_preflight_permits_authentication_headers(self, client):
        response = client.options("/api/v1/ingestions/merchant-days", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, Content-Type, Idempotency-Key",
        })
        assert response.status_code == 200
        assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
        allowed_headers = response.headers["Access-Control-Allow-Headers"].lower()
        assert "authorization" in allowed_headers
        assert "idempotency-key" in allowed_headers

    def test_register_login_and_me(self, client):
        created = register(client, "one")
        token = created["access_token"]
        me = client.get("/api/v1/auth/me", headers=auth_headers(token))
        assert me.status_code == 200
        assert me.get_json()["role"] == "admin"

        login = client.post("/api/v1/auth/login", json={
            "email": "admin-one@example.test",
            "password": "a-long-test-password-123",
            "organization_slug": "test-org-one",
        })
        assert login.status_code == 200
        assert login.get_json()["access_token"]

    def test_batch_ingestion_is_validated_atomic_and_idempotent(self, client):
        token = register(client, "ingest")["access_token"]
        payload = {"observations": observations()}
        headers = auth_headers(token, "batch-ingest-0001")
        first = client.post("/api/v1/ingestions/merchant-days", json=payload, headers=headers)
        assert first.status_code == 201, first.get_json()
        batch = first.get_json()
        assert batch["accepted_count"] == 8

        replay = client.post("/api/v1/ingestions/merchant-days", json=payload, headers=headers)
        assert replay.status_code == 200
        assert replay.get_json()["id"] == batch["id"]
        assert replay.get_json()["idempotent_replay"] is True

        merchant_list = client.get("/api/v1/merchants", headers=auth_headers(token))
        assert merchant_list.status_code == 200
        assert [merchant["merchant_id"] for merchant in merchant_list.get_json()["merchants"]] == ["M0001"]

        changed_payload = {"observations": observations("M0002")}
        conflict = client.post("/api/v1/ingestions/merchant-days", json=changed_payload, headers=headers)
        assert conflict.status_code == 409
        assert "different payload" in conflict.get_json()["reason"]

    def test_invalid_batch_is_rejected_without_persisting_partial_data(self, client):
        token = register(client, "invalid")["access_token"]
        invalid = observations(days=2)
        invalid[1]["date"] = invalid[0]["date"]
        response = client.post(
            "/api/v1/ingestions/merchant-days",
            json={"observations": invalid},
            headers=auth_headers(token, "batch-invalid-0001"),
        )
        assert response.status_code == 422
        assert response.get_json()["accepted_count"] == 0
        merchants = client.get("/api/v1/merchants", headers=auth_headers(token))
        assert merchants.get_json()["merchants"] == []

    def test_organization_isolation_hides_foreign_batches_and_merchants(self, client):
        org_a_token = register(client, "org-a")["access_token"]
        org_b_token = register(client, "org-b")["access_token"]
        created = client.post(
            "/api/v1/ingestions/merchant-days",
            json={"observations": observations("M9001")},
            headers=auth_headers(org_a_token, "batch-org-a-0001"),
        )
        assert created.status_code == 201
        foreign_batch = client.get(f"/api/v1/ingestions/{created.get_json()['id']}", headers=auth_headers(org_b_token))
        assert foreign_batch.status_code == 404
        foreign_merchants = client.get("/api/v1/merchants", headers=auth_headers(org_b_token))
        assert foreign_merchants.status_code == 200
        assert foreign_merchants.get_json()["merchants"] == []
