"""Shared tenant-authentication service used by Flask and FastAPI routes.

Keeping validation, persistence, audit recording, and token issuance in one
framework-neutral module prevents the incremental ASGI migration from creating
two security-sensitive registration or login implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth import issue_access_token, jwt_secret
from backend.database import get_session
from backend.models import AuditEvent, Membership, Organization, User


EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ORG_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}[a-z0-9]$")
REGISTRATION_FIELDS = frozenset({"email", "password", "organization_name", "organization_slug"})
LOGIN_FIELDS = frozenset({"email", "password", "organization_slug"})


class TenantAuthError(Exception):
    """A deliberately non-sensitive tenant-authentication error."""

    def __init__(self, reason: str, status_code: int) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


@dataclass(frozen=True)
class TokenBundle:
    """Stable, framework-neutral token response contract."""

    access_token: str
    organization_id: str
    organization_slug: str
    organization_name: str
    user_id: str
    email: str
    role: str

    def to_dict(self) -> dict:
        return {
            "access_token": self.access_token,
            "token_type": "Bearer",
            "organization": {
                "id": self.organization_id,
                "slug": self.organization_slug,
                "name": self.organization_name,
            },
            "user": {"id": self.user_id, "email": self.email, "role": self.role},
        }


def _exact_object(payload: object, *, fields: frozenset[str], operation: str) -> dict:
    if not isinstance(payload, dict) or set(payload) != fields:
        raise TenantAuthError(f"{operation} request must contain exactly {', '.join(sorted(fields))}.", 422)
    return payload


def _registration_values(payload: object) -> tuple[str, str, str, str]:
    body = _exact_object(payload, fields=REGISTRATION_FIELDS, operation="Registration")
    email = body["email"].strip().lower() if isinstance(body["email"], str) else ""
    password = body["password"]
    organization_name = body["organization_name"].strip() if isinstance(body["organization_name"], str) else ""
    organization_slug = body["organization_slug"].strip().lower() if isinstance(body["organization_slug"], str) else ""
    if not EMAIL_PATTERN.fullmatch(email):
        raise TenantAuthError("email must be a valid address.", 422)
    if not isinstance(password, str) or len(password) < 12:
        raise TenantAuthError("password must contain at least 12 characters.", 422)
    if not organization_name or len(organization_name) > 160:
        raise TenantAuthError("organization_name must be between 1 and 160 characters.", 422)
    if not ORG_SLUG_PATTERN.fullmatch(organization_slug):
        raise TenantAuthError("organization_slug must use lowercase letters, digits, and internal hyphens.", 422)
    return email, password, organization_name, organization_slug


def _login_values(payload: object) -> tuple[str, str, str]:
    # Preserve the legacy anti-enumeration behavior for the Flask endpoint:
    # incomplete credentials look the same as incorrect credentials (401).
    # The native FastAPI route validates this schema before reaching the
    # service and returns its controlled 422 malformed-request response.
    if not isinstance(payload, dict) or set(payload) != LOGIN_FIELDS:
        raise TenantAuthError("Invalid email, password, or organization.", 401)
    body = payload
    email = body["email"].strip().lower() if isinstance(body["email"], str) else ""
    password = body["password"]
    organization_slug = body["organization_slug"].strip().lower() if isinstance(body["organization_slug"], str) else ""
    if not isinstance(password, str):
        raise TenantAuthError("Invalid email, password, or organization.", 401)
    return email, password, organization_slug


def _token_bundle(user: User, membership: Membership, organization: Organization, *, configured_secret: str | None) -> TokenBundle:
    return TokenBundle(
        access_token=issue_access_token(user, membership, configured_secret=configured_secret),
        organization_id=organization.id,
        organization_slug=organization.slug,
        organization_name=organization.name,
        user_id=user.id,
        email=user.email,
        role=membership.role,
    )


def register_bootstrap(
    payload: object,
    *,
    bootstrap_enabled: bool,
    configured_secret: str | None,
) -> TokenBundle:
    """Create a locally bootstrapped tenant administrator exactly once per slug."""
    if not bootstrap_enabled:
        raise TenantAuthError("Bootstrap registration is disabled.", 404)
    try:
        jwt_secret(configured_secret=configured_secret)
    except RuntimeError:
        raise TenantAuthError("Authentication is not configured.", 503) from None
    email, password, organization_name, organization_slug = _registration_values(payload)

    session = None
    try:
        session = get_session()
        if session.scalar(select(User).where(User.email == email)) is not None:
            raise TenantAuthError("email is already registered.", 409)
        if session.scalar(select(Organization).where(Organization.slug == organization_slug)) is not None:
            raise TenantAuthError("organization_slug is already registered.", 409)
        organization = Organization(name=organization_name, slug=organization_slug)
        user = User(email=email, password_hash=generate_password_hash(password))
        session.add_all([organization, user])
        session.flush()
        membership = Membership(organization_id=organization.id, user_id=user.id, role="admin")
        session.add(membership)
        session.add(AuditEvent(
            organization_id=organization.id,
            actor_user_id=user.id,
            action="organization.bootstrap",
            entity_type="organization",
            entity_id=organization.id,
            payload={"email": user.email},
        ))
        session.commit()
        return _token_bundle(user, membership, organization, configured_secret=configured_secret)
    except TenantAuthError:
        if session is not None:
            session.rollback()
        raise
    except SQLAlchemyError:
        if session is not None:
            session.rollback()
        raise TenantAuthError("Authentication dependency is unavailable.", 503) from None
    except Exception:
        if session is not None:
            session.rollback()
        raise
    finally:
        if session is not None:
            session.close()


def login(payload: object, *, configured_secret: str | None) -> TokenBundle:
    """Authenticate locally without disclosing which credential component failed."""
    try:
        jwt_secret(configured_secret=configured_secret)
    except RuntimeError:
        raise TenantAuthError("Authentication is not configured.", 503) from None
    email, password, organization_slug = _login_values(payload)

    session = None
    try:
        session = get_session()
        user = session.scalar(select(User).where(User.email == email))
        organization = session.scalar(select(Organization).where(Organization.slug == organization_slug))
        membership = None if user is None or organization is None else session.scalar(
            select(Membership).where(Membership.user_id == user.id, Membership.organization_id == organization.id)
        )
        if user is None or membership is None or not user.is_active or not check_password_hash(user.password_hash, password):
            raise TenantAuthError("Invalid email, password, or organization.", 401)
        return _token_bundle(user, membership, organization, configured_secret=configured_secret)
    except SQLAlchemyError:
        raise TenantAuthError("Authentication dependency is unavailable.", 503) from None
    finally:
        if session is not None:
            session.close()
