"""Typed contracts shared by the incremental ASGI migration.

The established Flask tenant API remains the compatibility implementation while
route groups are migrated behind these Pydantic contracts.  Keeping the shared
response primitives here prevents the FastAPI surface from silently drifting
from the versioned OpenAPI document.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictResponseModel(BaseModel):
    """Forbid undocumented response fields on newly migrated endpoints."""

    model_config = ConfigDict(extra="forbid")


class StrictRequestModel(BaseModel):
    """Reject undocumented fields and implicit type coercion on native input."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ErrorResponse(StrictResponseModel):
    status: Literal["error"]
    reason: str = Field(min_length=1)
    request_id: str = Field(min_length=8, max_length=128)


class PrincipalResponse(StrictResponseModel):
    user_id: str = Field(min_length=1)
    organization_id: str = Field(min_length=1)
    role: Literal["admin", "analyst", "investigator", "viewer"]


class RegistrationRequest(StrictRequestModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=12, max_length=4096)
    organization_name: str = Field(min_length=1, max_length=160)
    organization_slug: str = Field(min_length=1, max_length=64)


class LoginRequest(StrictRequestModel):
    email: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=4096)
    organization_slug: str = Field(min_length=1, max_length=64)


class OrganizationResponse(StrictResponseModel):
    id: str = Field(min_length=1)
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)


class TokenUserResponse(StrictResponseModel):
    id: str = Field(min_length=1)
    email: str = Field(min_length=1, max_length=254)
    role: Literal["admin", "analyst", "investigator", "viewer"]


class TokenResponse(StrictResponseModel):
    access_token: str = Field(min_length=1)
    token_type: Literal["Bearer"]
    organization: OrganizationResponse
    user: TokenUserResponse


class HealthResponse(StrictResponseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None = None


class ReadinessChecks(StrictResponseModel):
    risk_engine: bool
    database: bool


class ReadinessResponse(StrictResponseModel):
    status: Literal["ready", "not_ready"]
    checks: ReadinessChecks
