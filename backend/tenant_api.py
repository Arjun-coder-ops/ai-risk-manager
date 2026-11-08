"""Authenticated, durable tenant API.

Legacy ``/api/risk`` routes remain a synthetic-demo surface for backwards
compatibility. New tenant data must use this authenticated ``/api/v1`` API.
"""

from __future__ import annotations

import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
from io import StringIO
import json
import os
import re

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import select
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth import issue_access_token, jwt_secret, require_auth
from backend.database import get_session
from backend.models import (
    AnalystFeedback,
    AuditEvent,
    IngestionBatch,
    IngestionJob,
    InvestigationCase,
    InvestigationEvent,
    Membership,
    Merchant,
    MerchantDayObservation,
    Organization,
    RiskAssessment,
    TransactionRecord,
    User,
)
from backend.transaction_validation import TransactionValidationError, validate_transactions
from model.validation import (
    ObservationValidationError,
    REQUIRED_INPUT_FIELDS,
    validate_observations,
    validate_request_identifiers,
)


tenant_api = Blueprint("tenant_api", __name__, url_prefix="/api/v1")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ORG_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")
CASE_STATUSES = frozenset({"OPEN", "IN_REVIEW", "ESCALATED", "RESOLVED", "FALSE_POSITIVE"})
CASE_TRANSITIONS = {
    "OPEN": frozenset({"IN_REVIEW", "ESCALATED", "RESOLVED", "FALSE_POSITIVE"}),
    "IN_REVIEW": frozenset({"ESCALATED", "RESOLVED", "FALSE_POSITIVE"}),
    "ESCALATED": frozenset({"IN_REVIEW", "RESOLVED", "FALSE_POSITIVE"}),
    "RESOLVED": frozenset(),
    "FALSE_POSITIVE": frozenset(),
}
AUTO_CASE_RISK_LEVELS = frozenset({"HIGH", "REVIEW"})
FEEDBACK_LABELS = frozenset({"CONFIRMED_FRAUD", "FALSE_POSITIVE", "INCONCLUSIVE"})
MEMBERSHIP_ROLES = frozenset({"admin", "analyst", "viewer"})
MAX_REPORTED_INGESTION_ERRORS = 100


class CsvUploadError(ValueError):
    """A controlled structural CSV upload error."""

    def __init__(self, reason: str, status: int = 422):
        super().__init__(reason)
        self.status = status


def api_error(reason: str, status: int):
    return jsonify({"status": "error", "reason": reason}), status


def request_object():
    body = request.get_json(silent=True)
    if body is None or not isinstance(body, dict):
        return None, api_error("Request body must be a valid JSON object.", 400)
    return body, None


def write_audit(session, *, organization_id: str | None, actor_user_id: str | None, action: str, entity_type: str, entity_id: str | None, payload: dict) -> None:
    session.add(AuditEvent(
        organization_id=organization_id,
        actor_user_id=actor_user_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        payload=payload,
    ))


def serialize_batch(batch: IngestionBatch, replayed: bool = False) -> dict:
    return {
        "id": batch.id,
        "status": batch.status,
        "received_count": batch.received_count,
        "accepted_count": batch.accepted_count,
        "rejected_count": batch.rejected_count,
        "error_summary": batch.error_summary,
        "created_at": batch.created_at.isoformat(),
        "idempotent_replay": replayed,
    }


def parse_csv_upload(upload) -> list[tuple[int, dict]]:
    """Return CSV rows with their one-based source line numbers.

    Parsing is intentionally strict about UTF-8 and the aggregate schema so
    clients do not get a successful batch with silently ignored columns.
    """
    if upload is None or not upload.filename:
        raise CsvUploadError("A CSV file is required in the file form field.", 400)
    if not upload.filename.lower().endswith(".csv"):
        raise CsvUploadError("Uploaded file must use a .csv filename.", 415)
    try:
        content = upload.read()
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise CsvUploadError("CSV file must be UTF-8 encoded.") from exc
    if not text.strip():
        raise CsvUploadError("CSV file must not be empty.")

    try:
        reader = csv.DictReader(StringIO(text))
        headers = reader.fieldnames
        if headers is None:
            raise CsvUploadError("CSV file must include a header row.")
        if len(headers) != len(set(headers)):
            raise CsvUploadError("CSV header contains duplicate column names.")
        missing = sorted(set(REQUIRED_INPUT_FIELDS) - set(headers))
        unexpected = sorted(set(headers) - set(REQUIRED_INPUT_FIELDS))
        if missing or unexpected:
            raise CsvUploadError(
                f"CSV header must contain exactly the required fields; missing={missing}, unexpected={unexpected}."
            )
        rows = []
        maximum_rows = int(current_app.config.get("MAX_CSV_INGESTION_ROWS", 1000))
        for line_number, row in enumerate(reader, start=2):
            if len(rows) >= maximum_rows:
                raise CsvUploadError(f"CSV file exceeds the {maximum_rows}-row ingestion limit.", 413)
            if None in row:
                raise CsvUploadError(f"CSV row {line_number} has more columns than the header.")
            rows.append((line_number, row))
    except csv.Error as exc:
        raise CsvUploadError("CSV file could not be parsed.") from exc
    if not rows:
        raise CsvUploadError("CSV file must contain at least one data row.")
    return rows


def append_ingestion_error(errors: list[dict], line_number: int, reason: str) -> None:
    if len(errors) < MAX_REPORTED_INGESTION_ERRORS:
        errors.append({"line": line_number, "reason": reason})


def transaction_days_for_merchants(session, merchant_ids: list[str], candidate_dates: set[date]) -> set[tuple[str, date]]:
    """Find raw transaction days that cannot be mixed with direct aggregates."""
    if not merchant_ids or not candidate_dates:
        return set()
    start = datetime.combine(min(candidate_dates), datetime.min.time(), tzinfo=timezone.utc)
    end = datetime.combine(max(candidate_dates), datetime.min.time(), tzinfo=timezone.utc) + timedelta(days=1)
    records = session.execute(
        select(TransactionRecord.merchant_id, TransactionRecord.occurred_at).where(
            TransactionRecord.merchant_id.in_(merchant_ids),
            TransactionRecord.occurred_at >= start,
            TransactionRecord.occurred_at < end,
        )
    ).all()
    result: set[tuple[str, date]] = set()
    for merchant_id, occurred_at in records:
        if occurred_at.tzinfo is None:
            occurred_at = occurred_at.replace(tzinfo=timezone.utc)
        record_date = occurred_at.astimezone(timezone.utc).date()
        if record_date in candidate_dates:
            result.add((merchant_id, record_date))
    return result


def validate_csv_rows(rows: list[tuple[int, dict]]) -> tuple[list[tuple[int, dict]], list[dict]]:
    """Validate independently so one bad CSV row does not discard good rows."""
    errors: list[dict] = []
    accepted_by_key: dict[tuple[str, str], tuple[int, dict]] = {}
    first_seen_line: dict[tuple[str, str], int] = {}
    duplicate_keys: set[tuple[str, str]] = set()

    for line_number, raw_row in rows:
        try:
            record = validate_observations(pd.DataFrame([raw_row])).to_dict(orient="records")[0]
        except ObservationValidationError as exc:
            append_ingestion_error(errors, line_number, str(exc).removeprefix("Row 0: "))
            continue
        key = (record["merchant_id"], record["date"])
        if key in first_seen_line:
            if key not in duplicate_keys:
                original_line = first_seen_line[key]
                accepted_by_key.pop(key, None)
                append_ingestion_error(errors, original_line, "Duplicate merchant-day observation in CSV file.")
                duplicate_keys.add(key)
            append_ingestion_error(errors, line_number, "Duplicate merchant-day observation in CSV file.")
            continue
        first_seen_line[key] = line_number
        accepted_by_key[key] = (line_number, record)
    return list(accepted_by_key.values()), errors


def serialize_member(membership: Membership, user: User) -> dict:
    return {
        "user_id": user.id,
        "email": user.email,
        "role": membership.role,
        "is_active": user.is_active,
        "created_at": membership.created_at.isoformat(),
    }


def serialize_assessment(
    assessment: RiskAssessment,
    merchant: Merchant,
    observation: MerchantDayObservation,
    *,
    case: InvestigationCase | None = None,
    replayed: bool = False,
) -> dict:
    return {
        "id": assessment.id,
        "merchant_id": merchant.external_id,
        "date": observation.observation_date.isoformat(),
        "status": assessment.status,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "model_version": assessment.model_version,
        "policy_version": assessment.policy_version,
        "evidence": assessment.evidence,
        "policy_decision": assessment.policy_decision,
        "case_id": case.id if case is not None else None,
        "idempotent_replay": replayed,
        "created_at": assessment.created_at.isoformat(),
    }


def serialize_feedback(feedback: AnalystFeedback) -> dict:
    return {
        "id": feedback.id,
        "risk_assessment_id": feedback.risk_assessment_id,
        "label": feedback.label,
        "notes": feedback.notes,
        "submitted_by_user_id": feedback.submitted_by_user_id,
        "verified": feedback.verified,
        "verified_by_user_id": feedback.verified_by_user_id,
        "verified_at": feedback.verified_at.isoformat() if feedback.verified_at else None,
        "created_at": feedback.created_at.isoformat(),
    }


def serialize_case(case: InvestigationCase, merchant: Merchant, assessment: RiskAssessment) -> dict:
    return {
        "id": case.id,
        "merchant_id": merchant.external_id,
        "risk_assessment_id": assessment.id,
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "model_version": assessment.model_version,
        "policy_version": assessment.policy_version,
        "status": case.status,
        "priority": case.priority,
        "assigned_user_id": case.assigned_user_id,
        "final_decision": case.final_decision,
        "resolved_at": case.resolved_at.isoformat() if case.resolved_at else None,
        "created_at": case.created_at.isoformat(),
    }


def case_context(session, case_id: str, organization_id: str):
    """Fetch a case and all display fields through organization-scoped joins."""
    return session.execute(
        select(InvestigationCase, Merchant, RiskAssessment)
        .join(Merchant, Merchant.id == InvestigationCase.merchant_id)
        .join(RiskAssessment, RiskAssessment.id == InvestigationCase.risk_assessment_id)
        .where(
            InvestigationCase.id == case_id,
            InvestigationCase.organization_id == organization_id,
            Merchant.organization_id == organization_id,
            RiskAssessment.organization_id == organization_id,
        )
    ).one_or_none()


def assessment_context(session, assessment_id: str, organization_id: str):
    return session.execute(
        select(RiskAssessment, Merchant, MerchantDayObservation)
        .join(MerchantDayObservation, MerchantDayObservation.id == RiskAssessment.merchant_day_observation_id)
        .join(Merchant, Merchant.id == MerchantDayObservation.merchant_id)
        .where(
            RiskAssessment.id == assessment_id,
            RiskAssessment.organization_id == organization_id,
            MerchantDayObservation.organization_id == organization_id,
            Merchant.organization_id == organization_id,
        )
    ).one_or_none()


def feedback_context(session, feedback_id: str, organization_id: str):
    return session.execute(
        select(AnalystFeedback, RiskAssessment)
        .join(RiskAssessment, RiskAssessment.id == AnalystFeedback.risk_assessment_id)
        .where(
            AnalystFeedback.id == feedback_id,
            AnalystFeedback.organization_id == organization_id,
            RiskAssessment.organization_id == organization_id,
        )
    ).one_or_none()


def membership_context(session, user_id: str, organization_id: str):
    return session.execute(
        select(Membership, User)
        .join(User, User.id == Membership.user_id)
        .where(
            Membership.organization_id == organization_id,
            Membership.user_id == user_id,
        )
    ).one_or_none()


def is_last_admin(session, membership: Membership, next_role: str | None = None) -> bool:
    """Prevent a tenant from becoming inaccessible through admin removal."""
    if membership.role != "admin" or next_role == "admin":
        return False
    admin_memberships = session.scalars(select(Membership.id).where(
        Membership.organization_id == membership.organization_id,
        Membership.role == "admin",
    )).all()
    return len(admin_memberships) <= 1


@tenant_api.route("/auth/register", methods=["POST"])
def register():
    """Create the first admin user for a new organization.

    This is a development bootstrap route. Production provisioning should move
    behind an invitation or SSO workflow before public deployment.
    """
    if not current_app.config.get("ALLOW_BOOTSTRAP_REGISTRATION", False):
        return api_error("Bootstrap registration is disabled.", 404)
    body, error = request_object()
    if error:
        return error
    try:
        jwt_secret()
    except RuntimeError:
        return api_error("Authentication is not configured.", 503)
    email = str(body.get("email", "")).strip().lower()
    password = body.get("password")
    organization_name = str(body.get("organization_name", "")).strip()
    organization_slug = str(body.get("organization_slug", "")).strip().lower()
    if not EMAIL_PATTERN.fullmatch(email):
        return api_error("email must be a valid address.", 422)
    if not isinstance(password, str) or len(password) < 12:
        return api_error("password must contain at least 12 characters.", 422)
    if not organization_name or len(organization_name) > 160:
        return api_error("organization_name must be between 1 and 160 characters.", 422)
    if not ORG_SLUG_PATTERN.fullmatch(organization_slug):
        return api_error("organization_slug must use lowercase letters, digits, and internal hyphens.", 422)

    session = get_session()
    try:
        if session.scalar(select(User).where(User.email == email)) is not None:
            return api_error("email is already registered.", 409)
        if session.scalar(select(Organization).where(Organization.slug == organization_slug)) is not None:
            return api_error("organization_slug is already registered.", 409)
        organization = Organization(name=organization_name, slug=organization_slug)
        user = User(email=email, password_hash=generate_password_hash(password))
        session.add_all([organization, user])
        session.flush()
        membership = Membership(organization_id=organization.id, user_id=user.id, role="admin")
        session.add(membership)
        write_audit(
            session,
            organization_id=organization.id,
            actor_user_id=user.id,
            action="organization.bootstrap",
            entity_type="organization",
            entity_id=organization.id,
            payload={"email": user.email},
        )
        session.commit()
        return jsonify({
            "access_token": issue_access_token(user, membership),
            "token_type": "Bearer",
            "organization": {"id": organization.id, "slug": organization.slug, "name": organization.name},
            "user": {"id": user.id, "email": user.email, "role": membership.role},
        }), 201
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@tenant_api.route("/auth/login", methods=["POST"])
def login():
    body, error = request_object()
    if error:
        return error
    try:
        jwt_secret()
    except RuntimeError:
        return api_error("Authentication is not configured.", 503)
    email = str(body.get("email", "")).strip().lower()
    password = body.get("password")
    organization_slug = str(body.get("organization_slug", "")).strip().lower()
    if not isinstance(password, str):
        return api_error("Invalid email, password, or organization.", 401)
    session = get_session()
    try:
        user = session.scalar(select(User).where(User.email == email))
        organization = session.scalar(select(Organization).where(Organization.slug == organization_slug))
        membership = None if user is None or organization is None else session.scalar(
            select(Membership).where(Membership.user_id == user.id, Membership.organization_id == organization.id)
        )
        if user is None or membership is None or not user.is_active or not check_password_hash(user.password_hash, password):
            return api_error("Invalid email, password, or organization.", 401)
        return jsonify({
            "access_token": issue_access_token(user, membership),
            "token_type": "Bearer",
            "user": {"id": user.id, "email": user.email, "role": membership.role},
            "organization": {"id": organization.id, "slug": organization.slug, "name": organization.name},
        })
    finally:
        session.close()


@tenant_api.route("/auth/me", methods=["GET"])
@require_auth()
def me():
    return jsonify({"user_id": g.principal.user_id, "organization_id": g.principal.organization_id, "role": g.principal.role})


@tenant_api.route("/members", methods=["GET"])
@require_auth({"admin"})
def list_members():
    members = g.db.execute(
        select(Membership, User)
        .join(User, User.id == Membership.user_id)
        .where(Membership.organization_id == g.principal.organization_id)
        .order_by(User.email)
    ).all()
    return jsonify({"members": [serialize_member(membership, user) for membership, user in members]})


@tenant_api.route("/members", methods=["POST"])
@require_auth({"admin"})
def grant_membership():
    """Grant an existing active account access to this organization.

    Account creation remains an explicit external identity/onboarding concern;
    this route never accepts another person's password or exposes invite tokens.
    """
    body, error = request_object()
    if error:
        return error
    if set(body) != {"email", "role"}:
        return api_error("Request body must contain only email and role.", 422)
    email = str(body["email"]).strip().lower() if isinstance(body["email"], str) else ""
    role = body["role"]
    if not EMAIL_PATTERN.fullmatch(email):
        return api_error("email must be a valid address.", 422)
    if not isinstance(role, str) or role not in MEMBERSHIP_ROLES:
        return api_error(f"role must be one of {sorted(MEMBERSHIP_ROLES)}.", 422)
    user = g.db.scalar(select(User).where(User.email == email))
    if user is None:
        return api_error("No active account exists for this email.", 404)
    if not user.is_active:
        return api_error("The account is inactive and cannot be granted access.", 422)
    existing = g.db.scalar(select(Membership).where(
        Membership.organization_id == g.principal.organization_id,
        Membership.user_id == user.id,
    ))
    if existing is not None:
        return api_error("This account already has a membership in the organization.", 409)
    membership = Membership(
        organization_id=g.principal.organization_id,
        user_id=user.id,
        role=role,
    )
    g.db.add(membership)
    g.db.flush()
    write_audit(
        g.db,
        organization_id=g.principal.organization_id,
        actor_user_id=g.principal.user_id,
        action="membership.granted",
        entity_type="membership",
        entity_id=membership.id,
        payload={"user_id": user.id, "role": role},
    )
    try:
        g.db.commit()
    except Exception:
        g.db.rollback()
        raise
    return jsonify(serialize_member(membership, user)), 201


@tenant_api.route("/members/<user_id>", methods=["PATCH"])
@require_auth({"admin"})
def change_membership_role(user_id: str):
    body, error = request_object()
    if error:
        return error
    if set(body) != {"role"} or not isinstance(body["role"], str):
        return api_error("Request body must contain only a string role.", 422)
    role = body["role"]
    if role not in MEMBERSHIP_ROLES:
        return api_error(f"role must be one of {sorted(MEMBERSHIP_ROLES)}.", 422)
    context = membership_context(g.db, user_id, g.principal.organization_id)
    if context is None:
        return api_error("Organization member was not found.", 404)
    membership, user = context
    if is_last_admin(g.db, membership, role):
        return api_error("The organization must retain at least one admin.", 409)
    previous_role = membership.role
    membership.role = role
    if previous_role != role:
        write_audit(
            g.db,
            organization_id=g.principal.organization_id,
            actor_user_id=g.principal.user_id,
            action="membership.role_changed",
            entity_type="membership",
            entity_id=membership.id,
            payload={"user_id": user.id, "previous_role": previous_role, "role": role},
        )
        try:
            g.db.commit()
        except Exception:
            g.db.rollback()
            raise
    return jsonify(serialize_member(membership, user))


@tenant_api.route("/members/<user_id>", methods=["DELETE"])
@require_auth({"admin"})
def revoke_membership(user_id: str):
    context = membership_context(g.db, user_id, g.principal.organization_id)
    if context is None:
        return api_error("Organization member was not found.", 404)
    membership, user = context
    if is_last_admin(g.db, membership):
        return api_error("The organization must retain at least one admin.", 409)
    membership_id = membership.id
    g.db.delete(membership)
    write_audit(
        g.db,
        organization_id=g.principal.organization_id,
        actor_user_id=g.principal.user_id,
        action="membership.revoked",
        entity_type="membership",
        entity_id=membership_id,
        payload={"user_id": user.id, "email": user.email},
    )
    try:
        g.db.commit()
    except Exception:
        g.db.rollback()
        raise
    return "", 204


@tenant_api.route("/ingestions/merchant-days", methods=["POST"])
@require_auth({"admin", "analyst"})
def ingest_merchant_days():
    body, error = request_object()
    if error:
        return error
    idempotency_key = request.headers.get("Idempotency-Key", "")
    if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
        return api_error("Idempotency-Key header must contain 8-128 safe characters.", 400)
    observations = body.get("observations")
    if not isinstance(observations, list) or not observations:
        return api_error("observations must be a non-empty JSON array.", 422)
    if not all(isinstance(row, dict) for row in observations):
        return api_error("Each observation must be a JSON object.", 422)
    try:
        validated = validate_observations(pd.DataFrame(observations))
    except ObservationValidationError as exc:
        return jsonify({"status": "error", "reason": str(exc), "accepted_count": 0, "rejected_count": len(observations)}), 422

    normalized_records = validated.to_dict(orient="records")
    payload_hash = hashlib.sha256(json.dumps(normalized_records, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()
    session = g.db
    try:
        existing_batch = session.scalar(select(IngestionBatch).where(
            IngestionBatch.organization_id == g.principal.organization_id,
            IngestionBatch.idempotency_key == idempotency_key,
        ))
        if existing_batch is not None:
            if existing_batch.payload_hash != payload_hash:
                return api_error("Idempotency-Key has already been used with a different payload.", 409)
            return jsonify(serialize_batch(existing_batch, replayed=True))

        merchant_ids = sorted(validated["merchant_id"].unique().tolist())
        existing_merchants = {
            merchant.external_id: merchant
            for merchant in session.scalars(select(Merchant).where(
                Merchant.organization_id == g.principal.organization_id,
                Merchant.external_id.in_(merchant_ids),
            ))
        }
        for external_id in merchant_ids:
            if external_id not in existing_merchants:
                merchant = Merchant(organization_id=g.principal.organization_id, external_id=external_id)
                session.add(merchant)
                existing_merchants[external_id] = merchant
        session.flush()

        incoming_keys = {(existing_merchants[row["merchant_id"]].id, date.fromisoformat(row["date"])) for row in normalized_records}
        existing_rows = session.execute(
            select(MerchantDayObservation.merchant_id, MerchantDayObservation.observation_date).where(
                MerchantDayObservation.merchant_id.in_([merchant.id for merchant in existing_merchants.values()]),
                MerchantDayObservation.observation_date.in_([key[1] for key in incoming_keys]),
            )
        ).all()
        if any((merchant_id, observation_date) in incoming_keys for merchant_id, observation_date in existing_rows):
            session.rollback()
            return api_error("One or more merchant-day observations already exist for this organization.", 409)
        transaction_keys = transaction_days_for_merchants(
            session,
            [merchant.id for merchant in existing_merchants.values()],
            {key[1] for key in incoming_keys},
        )
        if any(key in transaction_keys for key in incoming_keys):
            session.rollback()
            return api_error("One or more merchant-days already contain transaction-level records and cannot accept a direct aggregate.", 409)

        batch = IngestionBatch(
            organization_id=g.principal.organization_id,
            created_by_user_id=g.principal.user_id,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            received_count=len(normalized_records),
            accepted_count=len(normalized_records),
            rejected_count=0,
            status="SUCCEEDED",
        )
        session.add(batch)
        session.flush()
        for row in normalized_records:
            merchant = existing_merchants[row["merchant_id"]]
            session.add(MerchantDayObservation(
                organization_id=g.principal.organization_id,
                merchant_id=merchant.id,
                ingestion_batch_id=batch.id,
                observation_date=date.fromisoformat(row["date"]),
                source_kind="merchant_day",
                n_transactions=row["n_transactions"],
                n_failed=row["n_failed"],
                avg_amount=row["avg_amount"],
                amount_std=row["amount_std"],
                n_unique_users=row["n_unique_users"],
                n_new_users=row["n_new_users"],
                n_devices=row["n_devices"],
                distinct_regions_seen=row["distinct_regions_seen"],
            ))
        write_audit(session, organization_id=g.principal.organization_id, actor_user_id=g.principal.user_id,
                    action="merchant_day.ingested", entity_type="ingestion_batch", entity_id=batch.id,
                    payload={"accepted_count": batch.accepted_count, "idempotency_key": idempotency_key})
        session.commit()
        return jsonify(serialize_batch(batch)), 201
    except Exception:
        session.rollback()
        raise


@tenant_api.route("/ingestions/merchant-days/csv", methods=["POST"])
@require_auth({"admin", "analyst"})
def ingest_merchant_days_csv():
    """Synchronously ingest one CSV while retaining bounded row error evidence."""
    idempotency_key = request.headers.get("Idempotency-Key", "")
    if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
        return api_error("Idempotency-Key header must contain 8-128 safe characters.", 400)
    upload = request.files.get("file")
    try:
        rows = parse_csv_upload(upload)
    except CsvUploadError as exc:
        return api_error(str(exc), exc.status)
    assert upload is not None  # parse_csv_upload guarantees this for type checkers.
    upload.stream.seek(0)
    payload_hash = hashlib.sha256(upload.read()).hexdigest()

    session = g.db
    try:
        existing_batch = session.scalar(select(IngestionBatch).where(
            IngestionBatch.organization_id == g.principal.organization_id,
            IngestionBatch.idempotency_key == idempotency_key,
        ))
        if existing_batch is not None:
            if existing_batch.payload_hash != payload_hash:
                return api_error("Idempotency-Key has already been used with a different payload.", 409)
            return jsonify(serialize_batch(existing_batch, replayed=True))

        candidates, errors = validate_csv_rows(rows)
        candidate_merchant_ids = sorted({record["merchant_id"] for _, record in candidates})
        existing_merchants = {
            merchant.external_id: merchant
            for merchant in session.scalars(select(Merchant).where(
                Merchant.organization_id == g.principal.organization_id,
                Merchant.external_id.in_(candidate_merchant_ids),
            ))
        } if candidate_merchant_ids else {}
        existing_keys = set()
        if existing_merchants:
            candidate_dates = sorted({date.fromisoformat(record["date"]) for _, record in candidates})
            existing_keys = {
                (merchant_id, observation_date.isoformat())
                for merchant_id, observation_date in session.execute(
                    select(Merchant.external_id, MerchantDayObservation.observation_date)
                    .join(Merchant, Merchant.id == MerchantDayObservation.merchant_id)
                    .where(
                        Merchant.organization_id == g.principal.organization_id,
                        Merchant.external_id.in_(candidate_merchant_ids),
                        MerchantDayObservation.observation_date.in_(candidate_dates),
                    )
                )
            }

        transaction_keys = transaction_days_for_merchants(
            session,
            [merchant.id for merchant in existing_merchants.values()],
            {date.fromisoformat(record["date"]) for _, record in candidates},
        )
        transaction_external_keys = {
            (external_id, observation_date.isoformat())
            for external_id, merchant in existing_merchants.items()
            for merchant_id, observation_date in transaction_keys
            if merchant_id == merchant.id
        }

        persistable: list[tuple[int, dict]] = []
        for line_number, record in candidates:
            key = (record["merchant_id"], record["date"])
            if key in existing_keys:
                append_ingestion_error(errors, line_number, "Merchant-day observation already exists for this organization.")
                continue
            if key in transaction_external_keys:
                append_ingestion_error(errors, line_number, "Merchant-day already contains transaction-level records and cannot accept a direct aggregate.")
                continue
            persistable.append((line_number, record))

        for external_id in sorted({record["merchant_id"] for _, record in persistable}):
            if external_id not in existing_merchants:
                merchant = Merchant(organization_id=g.principal.organization_id, external_id=external_id)
                session.add(merchant)
                existing_merchants[external_id] = merchant
        session.flush()

        accepted_count = len(persistable)
        rejected_count = len(rows) - accepted_count
        batch = IngestionBatch(
            organization_id=g.principal.organization_id,
            created_by_user_id=g.principal.user_id,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            status="SUCCEEDED" if rejected_count == 0 else "PARTIAL" if accepted_count else "REJECTED",
            received_count=len(rows),
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            error_summary={
                "source": "csv",
                "error_count": rejected_count,
                "reported_errors": errors,
                "errors_truncated": rejected_count > len(errors),
            },
        )
        session.add(batch)
        session.flush()
        for _, row in persistable:
            merchant = existing_merchants[row["merchant_id"]]
            session.add(MerchantDayObservation(
                organization_id=g.principal.organization_id,
                merchant_id=merchant.id,
                ingestion_batch_id=batch.id,
                observation_date=date.fromisoformat(row["date"]),
                source_kind="merchant_day",
                n_transactions=row["n_transactions"],
                n_failed=row["n_failed"],
                avg_amount=row["avg_amount"],
                amount_std=row["amount_std"],
                n_unique_users=row["n_unique_users"],
                n_new_users=row["n_new_users"],
                n_devices=row["n_devices"],
                distinct_regions_seen=row["distinct_regions_seen"],
            ))
        write_audit(
            session,
            organization_id=g.principal.organization_id,
            actor_user_id=g.principal.user_id,
            action="merchant_day.csv_ingested",
            entity_type="ingestion_batch",
            entity_id=batch.id,
            payload={
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "status": batch.status,
                "idempotency_key": idempotency_key,
            },
        )
        session.commit()
        return jsonify(serialize_batch(batch)), 201
    except Exception:
        session.rollback()
        raise


@tenant_api.route("/ingestions/transactions", methods=["POST"])
@require_auth({"admin", "analyst"})
def ingest_transactions():
    """Durably accept pseudonymous transactions and queue daily aggregation.

    The 202 response means immutable raw records are committed, not that their
    merchant-day aggregate is already available.  A database-backed worker
    performs the aggregation and clients can poll the returned ingestion batch.
    """
    body, error = request_object()
    if error:
        return error
    idempotency_key = request.headers.get("Idempotency-Key", "")
    if not IDEMPOTENCY_PATTERN.fullmatch(idempotency_key):
        return api_error("Idempotency-Key header must contain 8-128 safe characters.", 400)
    transactions = body.get("transactions")
    maximum_rows = int(current_app.config.get("MAX_TRANSACTION_INGESTION_ROWS", 1000))
    if isinstance(transactions, list) and len(transactions) > maximum_rows:
        return api_error(f"transactions exceeds the {maximum_rows}-record ingestion limit.", 413)
    try:
        validated = validate_transactions(transactions)
    except TransactionValidationError as exc:
        return api_error(str(exc), 422)

    normalized_records = sorted(validated, key=lambda record: record["transaction_id"])
    payload_hash = hashlib.sha256(
        json.dumps(normalized_records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    session = g.db
    try:
        existing_batch = session.scalar(select(IngestionBatch).where(
            IngestionBatch.organization_id == g.principal.organization_id,
            IngestionBatch.idempotency_key == idempotency_key,
        ))
        if existing_batch is not None:
            if existing_batch.payload_hash != payload_hash:
                return api_error("Idempotency-Key has already been used with a different payload.", 409)
            return jsonify(serialize_batch(existing_batch, replayed=True))

        transaction_ids = [record["transaction_id"] for record in normalized_records]
        existing_transaction_ids = set(session.scalars(select(TransactionRecord.external_id).where(
            TransactionRecord.organization_id == g.principal.organization_id,
            TransactionRecord.external_id.in_(transaction_ids),
        )))
        if existing_transaction_ids:
            return api_error("One or more transaction_id values already exist for this organization.", 409)

        merchant_external_ids = sorted({record["merchant_id"] for record in normalized_records})
        merchants = {
            merchant.external_id: merchant
            for merchant in session.scalars(select(Merchant).where(
                Merchant.organization_id == g.principal.organization_id,
                Merchant.external_id.in_(merchant_external_ids),
            ))
        }
        for external_id in merchant_external_ids:
            if external_id not in merchants:
                merchant = Merchant(organization_id=g.principal.organization_id, external_id=external_id)
                session.add(merchant)
                merchants[external_id] = merchant
        session.flush()

        incoming_days = {
            (merchants[record["merchant_id"]].id, date.fromisoformat(record["date"]))
            for record in normalized_records
        }
        existing_observations = session.scalars(select(MerchantDayObservation).where(
            MerchantDayObservation.organization_id == g.principal.organization_id,
            MerchantDayObservation.merchant_id.in_([merchant.id for merchant in merchants.values()]),
            MerchantDayObservation.observation_date.in_([day for _, day in incoming_days]),
        )).all()
        direct_keys = {
            (observation.merchant_id, observation.observation_date)
            for observation in existing_observations
            if observation.source_kind != "transaction_aggregate"
        }
        if incoming_days & direct_keys:
            session.rollback()
            return api_error("One or more transaction days already have a direct merchant-day aggregate.", 409)

        batch = IngestionBatch(
            organization_id=g.principal.organization_id,
            created_by_user_id=g.principal.user_id,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            status="QUEUED",
            received_count=len(normalized_records),
            accepted_count=len(normalized_records),
            rejected_count=0,
            error_summary={"source": "transaction_api", "processing_state": "queued"},
        )
        session.add(batch)
        session.flush()
        for record in normalized_records:
            session.add(TransactionRecord(
                organization_id=g.principal.organization_id,
                merchant_id=merchants[record["merchant_id"]].id,
                ingestion_batch_id=batch.id,
                external_id=record["transaction_id"],
                occurred_at=datetime.fromisoformat(record["occurred_at"].replace("Z", "+00:00")),
                amount=record["amount"],
                outcome=record["outcome"],
                user_token=record["user_token"],
                device_token=record["device_token"],
                region_code=record["region_code"],
            ))
        session.add(IngestionJob(
            organization_id=g.principal.organization_id,
            ingestion_batch_id=batch.id,
            payload={
                "source": "transaction_api",
                "merchant_day_count": len(incoming_days),
            },
        ))
        write_audit(
            session,
            organization_id=g.principal.organization_id,
            actor_user_id=g.principal.user_id,
            action="transaction_ingestion.queued",
            entity_type="ingestion_batch",
            entity_id=batch.id,
            payload={
                "transaction_count": len(normalized_records),
                "merchant_day_count": len(incoming_days),
                "idempotency_key": idempotency_key,
            },
        )
        session.commit()
        return jsonify(serialize_batch(batch)), 202
    except Exception:
        session.rollback()
        raise


@tenant_api.route("/ingestions/<batch_id>", methods=["GET"])
@require_auth()
def get_ingestion(batch_id: str):
    batch = g.db.scalar(select(IngestionBatch).where(
        IngestionBatch.id == batch_id,
        IngestionBatch.organization_id == g.principal.organization_id,
    ))
    if batch is None:
        return api_error("Ingestion batch was not found.", 404)
    return jsonify(serialize_batch(batch))


@tenant_api.route("/merchants", methods=["GET"])
@require_auth()
def list_merchants():
    merchants = g.db.scalars(select(Merchant).where(
        Merchant.organization_id == g.principal.organization_id
    ).order_by(Merchant.external_id)).all()
    return jsonify({"merchants": [
        {"id": merchant.id, "merchant_id": merchant.external_id, "display_name": merchant.display_name}
        for merchant in merchants
    ]})


@tenant_api.route("/risk-assessments", methods=["POST"])
@require_auth({"admin", "analyst"})
def create_risk_assessment():
    """Score a persisted merchant-day observation and retain the decision trace.

    This endpoint deliberately reads only tenant-scoped database observations;
    it never falls back to the synthetic CSV used by legacy demo endpoints.
    """
    body, error = request_object()
    if error:
        return error
    unexpected = sorted(set(body) - {"merchant_id", "date"})
    if unexpected:
        return api_error(f"Unsupported request fields: {unexpected}", 422)
    try:
        merchant_id, target_date = validate_request_identifiers(body.get("merchant_id"), body.get("date"))
    except ObservationValidationError as exc:
        return api_error(f"Invalid request: {exc}", 422)

    session = g.db
    merchant = session.scalar(select(Merchant).where(
        Merchant.organization_id == g.principal.organization_id,
        Merchant.external_id == merchant_id,
    ))
    if merchant is None:
        return api_error("Merchant was not found.", 404)
    observation = session.scalar(select(MerchantDayObservation).where(
        MerchantDayObservation.organization_id == g.principal.organization_id,
        MerchantDayObservation.merchant_id == merchant.id,
        MerchantDayObservation.observation_date == date.fromisoformat(target_date),
    ))
    if observation is None:
        return api_error("Merchant-day observation was not found.", 404)

    engine = current_app.extensions.get("risk_engine")
    if engine is None:
        return api_error("Risk engine is unavailable.", 503)

    history_rows = session.scalars(select(MerchantDayObservation).where(
        MerchantDayObservation.organization_id == g.principal.organization_id,
        MerchantDayObservation.merchant_id == merchant.id,
    ).order_by(MerchantDayObservation.observation_date)).all()
    history = pd.DataFrame([{
        "merchant_id": merchant.external_id,
        "date": row.observation_date.isoformat(),
        "n_transactions": row.n_transactions,
        "n_failed": row.n_failed,
        "avg_amount": row.avg_amount,
        "amount_std": row.amount_std,
        "n_unique_users": row.n_unique_users,
        "n_new_users": row.n_new_users,
        "n_devices": row.n_devices,
        "distinct_regions_seen": row.distinct_regions_seen,
    } for row in history_rows])
    try:
        result = engine.score_merchant_day(history, target_date, merchant_id)
    except Exception:
        current_app.logger.exception("Tenant-scoped risk scoring failed")
        return api_error("Risk scoring could not be completed.", 503)

    model_version = str(getattr(result, "model_version", None) or getattr(engine, "model_version", "unknown"))
    # The engine owns the frozen policy artifact.  Do not reconstruct a policy
    # version from an environment variable after the score was produced.
    policy_version = str(getattr(result, "policy_version", None) or "unknown-policy")
    existing = session.scalar(select(RiskAssessment).where(
        RiskAssessment.organization_id == g.principal.organization_id,
        RiskAssessment.merchant_day_observation_id == observation.id,
        RiskAssessment.model_version == model_version,
        RiskAssessment.policy_version == policy_version,
    ))
    if existing is not None:
        existing_case = session.scalar(select(InvestigationCase).where(
            InvestigationCase.organization_id == g.principal.organization_id,
            InvestigationCase.risk_assessment_id == existing.id,
        ))
        payload = serialize_assessment(existing, merchant, observation, case=existing_case, replayed=True)
        return jsonify(payload)

    status = str(getattr(result, "status", "error"))
    assessment = RiskAssessment(
        organization_id=g.principal.organization_id,
        merchant_day_observation_id=observation.id,
        model_version=model_version,
        policy_version=policy_version,
        risk_score=getattr(result, "risk_score", None),
        risk_level=getattr(result, "risk_level", None),
        status=status,
        evidence=list(getattr(result, "signals", []) or []),
        policy_decision=dict(getattr(result, "policy_decision", None) or {}),
    )
    session.add(assessment)
    session.flush()

    case = None
    if status == "ok" and assessment.risk_level in AUTO_CASE_RISK_LEVELS:
        case = InvestigationCase(
            organization_id=g.principal.organization_id,
            merchant_id=merchant.id,
            risk_assessment_id=assessment.id,
            priority="HIGH" if assessment.risk_level == "HIGH" else "NORMAL",
        )
        session.add(case)
        session.flush()
        session.add(InvestigationEvent(
            organization_id=g.principal.organization_id,
            case_id=case.id,
            actor_user_id=g.principal.user_id,
            event_type="case.created_from_risk_assessment",
            payload={"risk_level": assessment.risk_level, "risk_score": assessment.risk_score},
        ))

    write_audit(
        session,
        organization_id=g.principal.organization_id,
        actor_user_id=g.principal.user_id,
        action="risk_assessment.created",
        entity_type="risk_assessment",
        entity_id=assessment.id,
        payload={
            "merchant_id": merchant.external_id,
            "date": target_date,
            "status": assessment.status,
            "risk_level": assessment.risk_level,
            "model_version": assessment.model_version,
            "policy_version": assessment.policy_version,
            "case_id": case.id if case else None,
        },
    )
    try:
        session.commit()
    except Exception:
        session.rollback()
        raise
    payload = serialize_assessment(assessment, merchant, observation, case=case)
    payload["recommended_action"] = getattr(result, "recommended_action", None)
    payload["reason"] = getattr(result, "reason", None)
    return jsonify(payload), 201


@tenant_api.route("/risk-assessments/<assessment_id>", methods=["GET"])
@require_auth()
def get_risk_assessment(assessment_id: str):
    context = assessment_context(g.db, assessment_id, g.principal.organization_id)
    if context is None:
        return api_error("Risk assessment was not found.", 404)
    assessment, merchant, observation = context
    case = g.db.scalar(select(InvestigationCase).where(
        InvestigationCase.organization_id == g.principal.organization_id,
        InvestigationCase.risk_assessment_id == assessment.id,
    ))
    return jsonify(serialize_assessment(assessment, merchant, observation, case=case))


@tenant_api.route("/risk-assessments/<assessment_id>/feedback", methods=["POST"])
@require_auth({"admin", "analyst"})
def submit_feedback(assessment_id: str):
    """Record a human outcome without changing a production model decision."""
    body, error = request_object()
    if error:
        return error
    if set(body) != {"label", "notes"}:
        return api_error("Request body must contain only label and notes.", 422)
    if not isinstance(body["label"], str):
        return api_error("label must be a string.", 422)
    label = body["label"].strip().upper()
    if label not in FEEDBACK_LABELS:
        return api_error(f"label must be one of {sorted(FEEDBACK_LABELS)}.", 422)
    notes = body["notes"]
    if not isinstance(notes, str) or not 1 <= len(notes.strip()) <= 2000:
        return api_error("notes must be a non-empty string of at most 2000 characters.", 422)

    context = assessment_context(g.db, assessment_id, g.principal.organization_id)
    if context is None:
        return api_error("Risk assessment was not found.", 404)
    assessment, _, _ = context
    normalized_notes = notes.strip()
    # A correction is represented by a new feedback record with a different
    # outcome or note; exact retries return the existing immutable record.
    existing = g.db.scalar(select(AnalystFeedback).where(
        AnalystFeedback.organization_id == g.principal.organization_id,
        AnalystFeedback.risk_assessment_id == assessment.id,
        AnalystFeedback.submitted_by_user_id == g.principal.user_id,
        AnalystFeedback.label == label,
        AnalystFeedback.notes == normalized_notes,
    ))
    if existing is not None:
        return jsonify(serialize_feedback(existing))

    feedback = AnalystFeedback(
        organization_id=g.principal.organization_id,
        risk_assessment_id=assessment.id,
        submitted_by_user_id=g.principal.user_id,
        label=label,
        notes=normalized_notes,
    )
    g.db.add(feedback)
    g.db.flush()
    linked_cases = g.db.scalars(select(InvestigationCase).where(
        InvestigationCase.organization_id == g.principal.organization_id,
        InvestigationCase.risk_assessment_id == assessment.id,
    )).all()
    for case in linked_cases:
        g.db.add(InvestigationEvent(
            organization_id=g.principal.organization_id,
            case_id=case.id,
            actor_user_id=g.principal.user_id,
            event_type="case.feedback_submitted",
            payload={"feedback_id": feedback.id, "label": label},
        ))
    write_audit(
        g.db,
        organization_id=g.principal.organization_id,
        actor_user_id=g.principal.user_id,
        action="feedback.submitted",
        entity_type="analyst_feedback",
        entity_id=feedback.id,
        payload={"risk_assessment_id": assessment.id, "label": label},
    )
    try:
        g.db.commit()
    except Exception:
        g.db.rollback()
        raise
    return jsonify(serialize_feedback(feedback)), 201


@tenant_api.route("/risk-assessments/<assessment_id>/feedback", methods=["GET"])
@require_auth({"admin", "analyst"})
def list_feedback(assessment_id: str):
    if assessment_context(g.db, assessment_id, g.principal.organization_id) is None:
        return api_error("Risk assessment was not found.", 404)
    feedback = g.db.scalars(select(AnalystFeedback).where(
        AnalystFeedback.organization_id == g.principal.organization_id,
        AnalystFeedback.risk_assessment_id == assessment_id,
    ).order_by(AnalystFeedback.created_at)).all()
    return jsonify({"feedback": [serialize_feedback(item) for item in feedback]})


@tenant_api.route("/feedback/<feedback_id>/verify", methods=["POST"])
@require_auth({"admin"})
def verify_feedback(feedback_id: str):
    body, error = request_object()
    if error:
        return error
    if body:
        return api_error("Feedback verification does not accept request fields.", 422)
    context = feedback_context(g.db, feedback_id, g.principal.organization_id)
    if context is None:
        return api_error("Analyst feedback was not found.", 404)
    feedback, assessment = context
    if feedback.verified:
        return api_error("Analyst feedback has already been verified.", 409)
    feedback.verified = True
    feedback.verified_by_user_id = g.principal.user_id
    feedback.verified_at = datetime.now(timezone.utc)
    linked_cases = g.db.scalars(select(InvestigationCase).where(
        InvestigationCase.organization_id == g.principal.organization_id,
        InvestigationCase.risk_assessment_id == assessment.id,
    )).all()
    for case in linked_cases:
        g.db.add(InvestigationEvent(
            organization_id=g.principal.organization_id,
            case_id=case.id,
            actor_user_id=g.principal.user_id,
            event_type="case.feedback_verified",
            payload={"feedback_id": feedback.id, "label": feedback.label},
        ))
    write_audit(
        g.db,
        organization_id=g.principal.organization_id,
        actor_user_id=g.principal.user_id,
        action="feedback.verified",
        entity_type="analyst_feedback",
        entity_id=feedback.id,
        payload={"risk_assessment_id": assessment.id, "label": feedback.label},
    )
    try:
        g.db.commit()
    except Exception:
        g.db.rollback()
        raise
    return jsonify(serialize_feedback(feedback))


@tenant_api.route("/cases", methods=["GET"])
@require_auth()
def list_cases():
    requested_status = request.args.get("status")
    if requested_status is not None:
        requested_status = requested_status.upper()
        if requested_status not in CASE_STATUSES:
            return api_error("status must be a valid case status.", 400)
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 100)
    except ValueError:
        return api_error("limit must be an integer between 1 and 100.", 400)
    query = (
        select(InvestigationCase, Merchant, RiskAssessment)
        .join(Merchant, Merchant.id == InvestigationCase.merchant_id)
        .join(RiskAssessment, RiskAssessment.id == InvestigationCase.risk_assessment_id)
        .where(
            InvestigationCase.organization_id == g.principal.organization_id,
            Merchant.organization_id == g.principal.organization_id,
            RiskAssessment.organization_id == g.principal.organization_id,
        )
        .order_by(InvestigationCase.created_at.desc())
        .limit(limit)
    )
    if requested_status:
        query = query.where(InvestigationCase.status == requested_status)
    contexts = g.db.execute(query).all()
    return jsonify({"cases": [serialize_case(case, merchant, assessment) for case, merchant, assessment in contexts]})


@tenant_api.route("/cases/<case_id>", methods=["GET"])
@require_auth()
def get_case(case_id: str):
    context = case_context(g.db, case_id, g.principal.organization_id)
    if context is None:
        return api_error("Investigation case was not found.", 404)
    case, merchant, assessment = context
    return jsonify(serialize_case(case, merchant, assessment))


@tenant_api.route("/cases/<case_id>/events", methods=["GET"])
@require_auth()
def list_case_events(case_id: str):
    if case_context(g.db, case_id, g.principal.organization_id) is None:
        return api_error("Investigation case was not found.", 404)
    events = g.db.scalars(select(InvestigationEvent).where(
        InvestigationEvent.organization_id == g.principal.organization_id,
        InvestigationEvent.case_id == case_id,
    ).order_by(InvestigationEvent.created_at)).all()
    return jsonify({"events": [{
        "id": event.id,
        "event_type": event.event_type,
        "actor_user_id": event.actor_user_id,
        "payload": event.payload,
        "created_at": event.created_at.isoformat(),
    } for event in events]})


@tenant_api.route("/cases/<case_id>/assignment", methods=["POST"])
@require_auth({"admin", "analyst"})
def assign_case(case_id: str):
    body, error = request_object()
    if error:
        return error
    if set(body) != {"assigned_user_id"}:
        return api_error("Request body must contain only assigned_user_id.", 422)
    assigned_user_id = body["assigned_user_id"]
    if assigned_user_id is not None and (not isinstance(assigned_user_id, str) or not assigned_user_id.strip()):
        return api_error("assigned_user_id must be a non-empty string or null.", 422)
    context = case_context(g.db, case_id, g.principal.organization_id)
    if context is None:
        return api_error("Investigation case was not found.", 404)
    case, merchant, assessment = context
    if g.principal.role != "admin" and assigned_user_id not in (None, g.principal.user_id):
        return api_error("Analysts may only assign a case to themselves or unassign it.", 403)
    if assigned_user_id is not None:
        assignee = g.db.scalar(select(Membership).where(
            Membership.organization_id == g.principal.organization_id,
            Membership.user_id == assigned_user_id,
        ))
        if assignee is None:
            return api_error("Assignee must have an active membership in this organization.", 422)
    previous_assignee = case.assigned_user_id
    case.assigned_user_id = assigned_user_id
    g.db.add(InvestigationEvent(
        organization_id=g.principal.organization_id,
        case_id=case.id,
        actor_user_id=g.principal.user_id,
        event_type="case.assignment_changed",
        payload={"previous_assigned_user_id": previous_assignee, "assigned_user_id": assigned_user_id},
    ))
    write_audit(g.db, organization_id=g.principal.organization_id, actor_user_id=g.principal.user_id,
                action="case.assignment_changed", entity_type="investigation_case", entity_id=case.id,
                payload={"assigned_user_id": assigned_user_id})
    try:
        g.db.commit()
    except Exception:
        g.db.rollback()
        raise
    return jsonify(serialize_case(case, merchant, assessment))


@tenant_api.route("/cases/<case_id>/transitions", methods=["POST"])
@require_auth({"admin", "analyst"})
def transition_case(case_id: str):
    body, error = request_object()
    if error:
        return error
    unexpected = sorted(set(body) - {"status", "note", "final_decision"})
    if unexpected or "status" not in body:
        return api_error("Request body must contain status and may include note and final_decision only.", 422)
    if not isinstance(body["status"], str):
        return api_error("status must be a string.", 422)
    next_status = body["status"].upper()
    if next_status not in CASE_STATUSES:
        return api_error("status must be a valid case status.", 422)
    note = body.get("note")
    if note is not None and (not isinstance(note, str) or len(note.strip()) > 2000):
        return api_error("note must be a string of at most 2000 characters.", 422)
    final_decision = body.get("final_decision")
    terminal = next_status in {"RESOLVED", "FALSE_POSITIVE"}
    if terminal:
        if not isinstance(final_decision, str) or not 1 <= len(final_decision.strip()) <= 64:
            return api_error("Terminal transitions require final_decision of 1-64 characters.", 422)
        if not isinstance(note, str) or not note.strip():
            return api_error("Terminal transitions require a non-empty note.", 422)
    elif final_decision is not None:
        return api_error("final_decision is allowed only for terminal transitions.", 422)

    context = case_context(g.db, case_id, g.principal.organization_id)
    if context is None:
        return api_error("Investigation case was not found.", 404)
    case, merchant, assessment = context
    if next_status not in CASE_TRANSITIONS.get(case.status, frozenset()):
        return api_error(f"Cannot transition a {case.status} case to {next_status}.", 409)
    previous_status = case.status
    case.status = next_status
    if terminal:
        case.final_decision = final_decision.strip()
        case.resolved_at = datetime.now(timezone.utc)
    g.db.add(InvestigationEvent(
        organization_id=g.principal.organization_id,
        case_id=case.id,
        actor_user_id=g.principal.user_id,
        event_type="case.transitioned",
        payload={
            "previous_status": previous_status,
            "status": next_status,
            "note": note.strip() if isinstance(note, str) else None,
            "final_decision": case.final_decision if terminal else None,
        },
    ))
    write_audit(g.db, organization_id=g.principal.organization_id, actor_user_id=g.principal.user_id,
                action="case.transitioned", entity_type="investigation_case", entity_id=case.id,
                payload={"previous_status": previous_status, "status": next_status})
    try:
        g.db.commit()
    except Exception:
        g.db.rollback()
        raise
    return jsonify(serialize_case(case, merchant, assessment))


@tenant_api.route("/audit-events", methods=["GET"])
@require_auth({"admin", "analyst"})
def list_audit_events():
    try:
        limit = min(max(int(request.args.get("limit", 50)), 1), 100)
    except ValueError:
        return api_error("limit must be an integer between 1 and 100.", 400)
    events = g.db.scalars(select(AuditEvent).where(
        AuditEvent.organization_id == g.principal.organization_id
    ).order_by(AuditEvent.created_at.desc()).limit(limit)).all()
    return jsonify({"events": [{
        "id": event.id, "action": event.action, "entity_type": event.entity_type,
        "entity_id": event.entity_id, "payload": event.payload, "created_at": event.created_at.isoformat(),
    } for event in events]})
