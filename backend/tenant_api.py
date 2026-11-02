"""Authenticated, durable tenant API.

Legacy ``/api/risk`` routes remain a synthetic-demo surface for backwards
compatibility. New tenant data must use this authenticated ``/api/v1`` API.
"""

from __future__ import annotations

from datetime import date
import hashlib
import json
import re

import pandas as pd
from flask import Blueprint, current_app, g, jsonify, request
from sqlalchemy import select
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth import issue_access_token, jwt_secret, require_auth
from backend.database import get_session
from backend.models import (
    AuditEvent,
    IngestionBatch,
    Membership,
    Merchant,
    MerchantDayObservation,
    Organization,
    RiskAssessment,
    User,
)
from model.validation import ObservationValidationError, normalize_date, validate_observations


tenant_api = Blueprint("tenant_api", __name__, url_prefix="/api/v1")
EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ORG_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
IDEMPOTENCY_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{8,128}$")


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
        "created_at": batch.created_at.isoformat(),
        "idempotent_replay": replayed,
    }


@tenant_api.route("/auth/register", methods=["POST"])
def register():
    """Create the first admin user for a new organization.

    This is a development bootstrap route. Production provisioning should move
    behind an invitation or SSO workflow before public deployment.
    """
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
