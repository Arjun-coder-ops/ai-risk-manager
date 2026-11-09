"""ASGI entrypoint for the incremental FastAPI migration.

FastAPI owns documentation delivery and future typed route groups.  The
well-tested Flask application is mounted unchanged for the current API
contract, which preserves compatibility during a deliberate route-by-route
migration instead of creating a second, divergent tenant API.
"""

from __future__ import annotations

import importlib

from fastapi import FastAPI, Request, Response, status
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from a2wsgi import WSGIMiddleware
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.request_controls import request_id_from_header
from backend.schemas import HealthResponse, ReadinessChecks, ReadinessResponse


legacy_app_module = importlib.import_module("backend.app")
flask_app = legacy_app_module.app


app = FastAPI(
    title="AI Risk Intelligence Platform",
    version="1.0.0",
    description=(
        "Incremental ASGI gateway. The published tenant contract remains at "
        "`/api/v1/openapi.json`; Flask serves legacy-compatible routes while "
        "typed FastAPI route groups are introduced with compatibility tests."
    ),
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


@app.middleware("http")
async def attach_request_correlation_id(request: Request, call_next):
    """Provide a safe correlation ID for native FastAPI routes.

    Mounted Flask routes already attach their own response ID.  ``setdefault``
    preserves that compatibility behavior during the incremental migration.
    """
    request_id = request_id_from_header(request.headers.get("X-Request-ID"))
    response: Response = await call_next(request)
    response.headers.setdefault("X-Request-ID", request_id)
    return response


@app.get("/health", response_model=HealthResponse, responses={503: {"model": HealthResponse}})
def health() -> HealthResponse | JSONResponse:
    """Typed ASGI liveness endpoint with legacy-compatible semantics."""
    engine = legacy_app_module.engine
    if engine is None:
        return JSONResponse(
            HealthResponse(status="degraded").model_dump(exclude_none=True),
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )
    return HealthResponse(status="ok", model_version=engine.model_version)


@app.get("/ready", response_model=ReadinessResponse, responses={503: {"model": ReadinessResponse}})
def ready() -> ReadinessResponse | JSONResponse:
    """Typed ASGI readiness endpoint requiring model and database access."""
    checks = ReadinessChecks(risk_engine=legacy_app_module.engine is not None, database=False)
    try:
        with legacy_app_module.get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        checks = checks.model_copy(update={"database": True})
    except SQLAlchemyError:
        pass
    payload = ReadinessResponse(status="ready" if checks.risk_engine and checks.database else "not_ready", checks=checks)
    if payload.status == "ready":
        return payload
    return JSONResponse(payload.model_dump(), status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.get("/api/v1/docs", include_in_schema=False)
def tenant_api_docs():
    """Serve Swagger UI for the source-controlled tenant OpenAPI contract."""
    return get_swagger_ui_html(
        openapi_url="/api/v1/openapi.json",
        title="AI Risk Intelligence Platform tenant API",
    )


# This catch-all mount must remain after native FastAPI routes so the docs
# endpoint above is served by ASGI while all existing Flask endpoints retain
# their behavior and route precedence.
app.mount("/", WSGIMiddleware(flask_app))
