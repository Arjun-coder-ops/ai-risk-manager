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


class ErrorResponse(StrictResponseModel):
    status: Literal["error"]
    reason: str = Field(min_length=1)
    request_id: str = Field(min_length=8, max_length=128)


class HealthResponse(StrictResponseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None = None


class ReadinessChecks(StrictResponseModel):
    risk_engine: bool
    database: bool


class ReadinessResponse(StrictResponseModel):
    status: Literal["ready", "not_ready"]
    checks: ReadinessChecks

