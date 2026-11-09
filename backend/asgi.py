"""ASGI entrypoint for the incremental FastAPI migration.

FastAPI owns documentation delivery and future typed route groups.  The
well-tested Flask application is mounted unchanged for the current API
contract, which preserves compatibility during a deliberate route-by-route
migration instead of creating a second, divergent tenant API.
"""

from __future__ import annotations

import importlib

from a2wsgi import WSGIMiddleware
from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.auth import AuthenticationError, Principal, authenticate_access_token
from backend.request_controls import request_id_from_header
from backend.schemas import ErrorResponse, HealthResponse, PrincipalResponse, ReadinessChecks, ReadinessResponse


legacy_app_module = importlib.import_module("backend.app")
flask_app = legacy_app_module.app
NATIVE_AUTH_PATHS = frozenset({"/api/v1/auth/me"})


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


def native_error_response(reason: str, status_code: int, request_id: str) -> JSONResponse:
    """Return the stable error envelope for a route owned by FastAPI."""
    return JSONResponse(
        ErrorResponse(status="error", reason=reason, request_id=request_id).model_dump(),
        status_code=status_code,
    )


@app.exception_handler(HTTPException)
async def native_http_exception(request: Request, exc: HTTPException) -> JSONResponse:
    reason = exc.detail if isinstance(exc.detail, str) else "Request could not be completed."
    request_id = getattr(request.state, "request_id", request_id_from_header(None))
    return native_error_response(reason, exc.status_code, request_id)


@app.exception_handler(RequestValidationError)
async def native_validation_exception(request: Request, _: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", request_id_from_header(None))
    return native_error_response("Request validation failed.", status.HTTP_422_UNPROCESSABLE_ENTITY, request_id)


@app.middleware("http")
async def attach_request_correlation_id(request: Request, call_next):
    """Provide a safe correlation ID for native FastAPI routes.

    Mounted Flask routes already attach their own response ID.  ``setdefault``
    preserves that compatibility behavior during the incremental migration.
    """
    request_id = request_id_from_header(request.headers.get("X-Request-ID"))
    request.state.request_id = request_id
    rate_limit = None
    if request.method != "OPTIONS" and request.url.path in NATIVE_AUTH_PATHS:
        limiter = flask_app.extensions["rate_limiter"]
        client = request.client.host if request.client else "unknown"
        rate_limit = limiter.check(
            f"auth:{client}",
            limit=flask_app.config["AUTH_RATE_LIMIT_PER_MINUTE"],
            window_seconds=flask_app.config["RATE_LIMIT_WINDOW_SECONDS"],
        )
        if not rate_limit.allowed:
            response = native_error_response("Rate limit exceeded. Retry later.", 429, request_id)
            response.headers["Retry-After"] = str(rate_limit.retry_after_seconds)
        else:
            response = await call_next(request)
    else:
        response = await call_next(request)
    response.headers.setdefault("X-Request-ID", request_id)
    if rate_limit is not None:
        response.headers["X-RateLimit-Limit"] = str(rate_limit.limit)
        response.headers["X-RateLimit-Remaining"] = str(rate_limit.remaining)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(legacy_app_module.ALLOWED_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "Idempotency-Key"],
)


def native_principal(request: Request) -> Principal:
    """FastAPI dependency that preserves the active-membership auth invariant."""
    try:
        return authenticate_access_token(
            request.headers.get("Authorization"),
            configured_secret=flask_app.config.get("JWT_SECRET"),
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.reason) from None


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


@app.get(
    "/api/v1/auth/me",
    response_model=PrincipalResponse,
    responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def me(principal: Principal = Depends(native_principal)) -> PrincipalResponse:
    """Return a tenant principal after current membership revalidation."""
    return PrincipalResponse(
        user_id=principal.user_id,
        organization_id=principal.organization_id,
        role=principal.role,
    )


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
