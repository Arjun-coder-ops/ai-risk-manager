"""Integration tests for the authenticated, tenant-scoped durable API."""

from __future__ import annotations

from datetime import date, timedelta
import os
import sys
from types import SimpleNamespace

import pytest
from werkzeug.security import generate_password_hash

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

import app as application  # noqa: E402
from backend.auth import issue_access_token  # noqa: E402
from backend.database import configure_database, create_schema, get_session  # noqa: E402
from backend.models import Membership, User  # noqa: E402


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


def create_member_token(organization_id: str, email: str, role: str) -> str:
    """Provision a test-only organization member without exposing a public API."""
    session = get_session()
    try:
        user = User(email=email, password_hash=generate_password_hash("test-only-password"))
        session.add(user)
        session.flush()
        membership = Membership(organization_id=organization_id, user_id=user.id, role=role)
        session.add(membership)
        session.commit()
        with application.app.app_context():
            return issue_access_token(user, membership)
    finally:
        session.close()


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


class DeterministicRiskEngine:
    """Keep tenant API tests independent from serialized model artifacts."""

    model_version = "test-model-v1"

    def score_merchant_day(self, history, target_date, merchant_id):
        assert len(history) >= 1
        return SimpleNamespace(
            status="ok",
            merchant_id=merchant_id,
            date=target_date,
            risk_score=0.91,
            risk_level="HIGH",
            signals=[{"feature": "txn_volume_zscore", "impact": "Strongly increased risk"}],
            recommended_action="Open an investigation.",
            model_version=self.model_version,
            reason=None,
        )


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
        allowed_methods = response.headers["Access-Control-Allow-Methods"]
        assert "PATCH" in allowed_methods
        assert "DELETE" in allowed_methods

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


class TestTenantMemberAdministration:
    def test_admin_can_manage_existing_accounts_without_losing_last_admin(self, client):
        owner = register(client, "member-owner")
        owner_token = owner["access_token"]
        candidate = register(client, "member-candidate")
        candidate_own_token = candidate["access_token"]
        candidate_user = candidate["user"]
        owner_user = owner["user"]

        initial_members = client.get("/api/v1/members", headers=auth_headers(owner_token))
        assert initial_members.status_code == 200
        assert [member["email"] for member in initial_members.get_json()["members"]] == [owner_user["email"]]

        grant = client.post(
            "/api/v1/members",
            json={"email": candidate_user["email"], "role": "analyst"},
            headers=auth_headers(owner_token),
        )
        assert grant.status_code == 201, grant.get_json()
        assert grant.get_json()["user_id"] == candidate_user["id"]
        assert grant.get_json()["role"] == "analyst"

        duplicate = client.post(
            "/api/v1/members",
            json={"email": candidate_user["email"], "role": "analyst"},
            headers=auth_headers(owner_token),
        )
        assert duplicate.status_code == 409

        candidate_owner_login = client.post("/api/v1/auth/login", json={
            "email": candidate_user["email"],
            "password": "a-long-test-password-123",
            "organization_slug": "test-org-member-owner",
        })
        assert candidate_owner_login.status_code == 200
        candidate_owner_token = candidate_owner_login.get_json()["access_token"]
        assert client.get("/api/v1/members", headers=auth_headers(candidate_owner_token)).status_code == 403

        last_admin = client.patch(
            f"/api/v1/members/{owner_user['id']}",
            json={"role": "viewer"},
            headers=auth_headers(owner_token),
        )
        assert last_admin.status_code == 409

        promote = client.patch(
            f"/api/v1/members/{candidate_user['id']}",
            json={"role": "admin"},
            headers=auth_headers(owner_token),
        )
        assert promote.status_code == 200
        assert promote.get_json()["role"] == "admin"
        # Membership is checked against the database on each request, so a
        # pre-promotion token receives the new permission immediately.
        assert client.get("/api/v1/members", headers=auth_headers(candidate_owner_token)).status_code == 200

        foreign_update = client.patch(
            f"/api/v1/members/{owner_user['id']}",
            json={"role": "viewer"},
            headers=auth_headers(candidate_own_token),
        )
        assert foreign_update.status_code == 404

        revoke = client.delete(
            f"/api/v1/members/{candidate_user['id']}",
            headers=auth_headers(owner_token),
        )
        assert revoke.status_code == 204
        assert client.get("/api/v1/auth/me", headers=auth_headers(candidate_owner_token)).status_code == 401


class TestTenantRiskAssessmentAndCases:
    def _create_case(self, client, monkeypatch, suffix: str = "case"):
        monkeypatch.setitem(application.app.extensions, "risk_engine", DeterministicRiskEngine())
        token = register(client, suffix)["access_token"]
        ingestion = client.post(
            "/api/v1/ingestions/merchant-days",
            json={"observations": observations("M7001")},
            headers=auth_headers(token, f"batch-{suffix}-0001"),
        )
        assert ingestion.status_code == 201, ingestion.get_json()
        score = client.post(
            "/api/v1/risk-assessments",
            json={"merchant_id": "M7001", "date": "2026-01-08"},
            headers=auth_headers(token),
        )
        assert score.status_code == 201, score.get_json()
        return token, score.get_json()

    def test_scoring_is_tenant_scoped_persistent_and_replay_safe(self, client, monkeypatch):
        token, assessment = self._create_case(client, monkeypatch, "score")
        assert assessment["risk_level"] == "HIGH"
        assert assessment["risk_score"] == 0.91
        assert assessment["case_id"]
        assert assessment["evidence"][0]["feature"] == "txn_volume_zscore"

        replay = client.post(
            "/api/v1/risk-assessments",
            json={"merchant_id": "M7001", "date": "2026-01-08"},
            headers=auth_headers(token),
        )
        assert replay.status_code == 200
        assert replay.get_json()["id"] == assessment["id"]
        assert replay.get_json()["idempotent_replay"] is True

        fetched = client.get(f"/api/v1/risk-assessments/{assessment['id']}", headers=auth_headers(token))
        assert fetched.status_code == 200
        assert fetched.get_json()["case_id"] == assessment["case_id"]
        cases = client.get("/api/v1/cases?status=OPEN", headers=auth_headers(token))
        assert cases.status_code == 200
        assert [item["id"] for item in cases.get_json()["cases"]] == [assessment["case_id"]]

        other_token = register(client, "score-other")["access_token"]
        assert client.get(f"/api/v1/risk-assessments/{assessment['id']}", headers=auth_headers(other_token)).status_code == 404
        assert client.get(f"/api/v1/cases/{assessment['case_id']}", headers=auth_headers(other_token)).status_code == 404

    def test_case_assignment_transitions_and_event_history(self, client, monkeypatch):
        token, assessment = self._create_case(client, monkeypatch, "workflow")
        me = client.get("/api/v1/auth/me", headers=auth_headers(token)).get_json()
        case_id = assessment["case_id"]

        assignment = client.post(
            f"/api/v1/cases/{case_id}/assignment",
            json={"assigned_user_id": me["user_id"]},
            headers=auth_headers(token),
        )
        assert assignment.status_code == 200
        assert assignment.get_json()["assigned_user_id"] == me["user_id"]

        in_review = client.post(
            f"/api/v1/cases/{case_id}/transitions",
            json={"status": "IN_REVIEW", "note": "Examining the unexpected volume."},
            headers=auth_headers(token),
        )
        assert in_review.status_code == 200
        assert in_review.get_json()["status"] == "IN_REVIEW"

        missing_evidence = client.post(
            f"/api/v1/cases/{case_id}/transitions",
            json={"status": "RESOLVED"},
            headers=auth_headers(token),
        )
        assert missing_evidence.status_code == 422

        resolved = client.post(
            f"/api/v1/cases/{case_id}/transitions",
            json={"status": "RESOLVED", "final_decision": "confirmed_fraud", "note": "Verified by the analyst."},
            headers=auth_headers(token),
        )
        assert resolved.status_code == 200
        assert resolved.get_json()["final_decision"] == "confirmed_fraud"
        assert resolved.get_json()["resolved_at"]

        events = client.get(f"/api/v1/cases/{case_id}/events", headers=auth_headers(token))
        assert events.status_code == 200
        assert [event["event_type"] for event in events.get_json()["events"]] == [
            "case.created_from_risk_assessment",
            "case.assignment_changed",
            "case.transitioned",
            "case.transitioned",
        ]

    def test_risk_assessment_rejects_unknown_fields_and_unknown_merchant(self, client, monkeypatch):
        monkeypatch.setitem(application.app.extensions, "risk_engine", DeterministicRiskEngine())
        token = register(client, "assessment-validation")["access_token"]
        extra_field = client.post(
            "/api/v1/risk-assessments",
            json={"merchant_id": "M0001", "date": "2026-01-01", "ignore": True},
            headers=auth_headers(token),
        )
        assert extra_field.status_code == 422
        missing_merchant = client.post(
            "/api/v1/risk-assessments",
            json={"merchant_id": "M0001", "date": "2026-01-01"},
            headers=auth_headers(token),
        )
        assert missing_merchant.status_code == 404

    def test_real_engine_persists_insufficient_evidence_without_opening_a_case(self, client):
        token = register(client, "insufficient-evidence")["access_token"]
        ingestion = client.post(
            "/api/v1/ingestions/merchant-days",
            json={"observations": observations("M8001", days=1)},
            headers=auth_headers(token, "batch-insufficient-0001"),
        )
        assert ingestion.status_code == 201
        assessment = client.post(
            "/api/v1/risk-assessments",
            json={"merchant_id": "M8001", "date": "2026-01-01"},
            headers=auth_headers(token),
        )
        assert assessment.status_code == 201, assessment.get_json()
        assert assessment.get_json()["status"] == "insufficient_evidence"
        assert assessment.get_json()["case_id"] is None

    def test_feedback_is_scoped_audited_and_requires_admin_verification(self, client, monkeypatch):
        admin_token, assessment = self._create_case(client, monkeypatch, "feedback")
        organization_id = client.get("/api/v1/auth/me", headers=auth_headers(admin_token)).get_json()["organization_id"]
        analyst_token = create_member_token(organization_id, "analyst-feedback@example.test", "analyst")

        created = client.post(
            f"/api/v1/risk-assessments/{assessment['id']}/feedback",
            json={"label": "false_positive", "notes": "The merchant supplied valid supporting invoices."},
            headers=auth_headers(analyst_token),
        )
        assert created.status_code == 201, created.get_json()
        feedback = created.get_json()
        assert feedback["label"] == "FALSE_POSITIVE"
        assert feedback["verified"] is False
        assert feedback["submitted_by_user_id"]

        duplicate = client.post(
            f"/api/v1/risk-assessments/{assessment['id']}/feedback",
            json={"label": "FALSE_POSITIVE", "notes": "The merchant supplied valid supporting invoices."},
            headers=auth_headers(analyst_token),
        )
        assert duplicate.status_code == 200
        assert duplicate.get_json()["id"] == feedback["id"]

        analyst_verify = client.post(
            f"/api/v1/feedback/{feedback['id']}/verify",
            json={},
            headers=auth_headers(analyst_token),
        )
        assert analyst_verify.status_code == 403

        verified = client.post(
            f"/api/v1/feedback/{feedback['id']}/verify",
            json={},
            headers=auth_headers(admin_token),
        )
        assert verified.status_code == 200
        assert verified.get_json()["verified"] is True
        assert verified.get_json()["verified_by_user_id"]
        assert verified.get_json()["verified_at"]

        events = client.get(f"/api/v1/cases/{assessment['case_id']}/events", headers=auth_headers(admin_token))
        assert [event["event_type"] for event in events.get_json()["events"]][-2:] == [
            "case.feedback_submitted",
            "case.feedback_verified",
        ]

        other_token = register(client, "feedback-other")["access_token"]
        foreign = client.get(
            f"/api/v1/risk-assessments/{assessment['id']}/feedback",
            headers=auth_headers(other_token),
        )
        assert foreign.status_code == 404

    @pytest.mark.parametrize("payload", [
        {"label": "NOT_A_LABEL", "notes": "A complete note."},
        {"label": "FALSE_POSITIVE", "notes": ""},
        {"label": "FALSE_POSITIVE", "notes": "A complete note.", "unexpected": True},
    ])
    def test_feedback_validation_rejects_malformed_payloads(self, client, monkeypatch, payload):
        suffix = f"feedback-validation-{payload['label'].lower().replace('_', '-')}"
        token, assessment = self._create_case(client, monkeypatch, suffix)
        response = client.post(
            f"/api/v1/risk-assessments/{assessment['id']}/feedback",
            json=payload,
            headers=auth_headers(token),
        )
        assert response.status_code == 422
