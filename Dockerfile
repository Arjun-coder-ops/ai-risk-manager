# Multi-stage build for AI Risk Manager application
# Stage 1: Builder
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# requirements.txt includes the reviewed lock file, so both must be present
# before pip resolves dependencies in the build stage.
COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Set PATH to include local pip installations
ENV PATH=/root/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY . .

# Readiness verifies both the loaded inference artifact and database
# connectivity; liveness remains available at /health for process checks.
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/ready')"

# Apply schema migrations before starting the application. Production delivery
# should run this as a dedicated one-off release job; this command keeps local
# Docker Compose startup reproducible. Gunicorn replaces Flask's development
# server while retaining the existing Flask WSGI application.
CMD ["sh", "-c", "alembic upgrade head && exec gunicorn --bind 0.0.0.0:${PORT:-8000} --workers ${WEB_CONCURRENCY:-1} --timeout ${GUNICORN_TIMEOUT:-60} --access-logfile - --error-logfile - backend.app:app"]
