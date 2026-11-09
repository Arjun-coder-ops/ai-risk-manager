"""ASGI entrypoint for the incremental FastAPI migration.

FastAPI owns documentation delivery and future typed route groups.  The
well-tested Flask application is mounted unchanged for the current API
contract, which preserves compatibility during a deliberate route-by-route
migration instead of creating a second, divergent tenant API.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html
from a2wsgi import WSGIMiddleware

from backend.app import app as flask_app


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
