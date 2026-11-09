"""JWT authentication and organization-scoped authorization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
import os
from typing import Callable, Iterable

import jwt
from flask import current_app, g, jsonify, request
from sqlalchemy import select

from backend.database import get_session
from backend.models import Membership, User


JWT_ALGORITHM = "HS256"
JWT_AUDIENCE = "ai-risk-intelligence-api"


@dataclass(frozen=True)
class Principal:
    user_id: str
    organization_id: str
    role: str


def _error_payload(reason: str) -> dict:
    """Return the stable API error envelope without importing the app module."""
    payload = {"status": "error", "reason": reason}
    request_id = getattr(g, "request_id", None)
    if request_id:
        payload["request_id"] = request_id
    return payload


def jwt_secret() -> str:
    secret = current_app.config.get("JWT_SECRET") or os.environ.get("JWT_SECRET")
    if not secret or len(secret) < 32:
        raise RuntimeError("JWT_SECRET must be configured with at least 32 characters.")
    return secret


def issue_access_token(user: User, membership: Membership) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.id,
        "org": membership.organization_id,
        "role": membership.role,
        "aud": JWT_AUDIENCE,
        "iat": now,
        "exp": now + timedelta(minutes=60),
    }
    return jwt.encode(payload, jwt_secret(), algorithm=JWT_ALGORITHM)


def _unauthorized(reason: str):
    return jsonify(_error_payload(reason)), 401


def require_auth(roles: Iterable[str] | None = None) -> Callable:
    """Require a valid token and a currently active organization membership."""
    permitted_roles = set(roles or [])

    def decorator(view: Callable) -> Callable:
        @wraps(view)
        def wrapped(*args, **kwargs):
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return _unauthorized("A Bearer access token is required.")
            try:
                claims = jwt.decode(
                    header.removeprefix("Bearer "),
                    jwt_secret(),
                    algorithms=[JWT_ALGORITHM],
                    audience=JWT_AUDIENCE,
                )
            except RuntimeError:
                return jsonify(_error_payload("Authentication is not configured.")), 503
            except jwt.PyJWTError:
                return _unauthorized("Access token is invalid or expired.")

            session = get_session()
            try:
                membership = session.scalar(
                    select(Membership).where(
                        Membership.organization_id == claims.get("org"),
                        Membership.user_id == claims.get("sub"),
                    )
                )
                user = session.get(User, claims.get("sub"))
                if membership is None or user is None or not user.is_active:
                    return _unauthorized("Access token no longer has an active membership.")
                if permitted_roles and membership.role not in permitted_roles:
                    return jsonify(_error_payload("Insufficient role for this action.")), 403
                g.db = session
                g.principal = Principal(user_id=user.id, organization_id=membership.organization_id, role=membership.role)
                return view(*args, **kwargs)
            finally:
                session.close()

        return wrapped

    return decorator
