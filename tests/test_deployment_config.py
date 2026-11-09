from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_build_copies_the_lockfile_required_by_requirements_txt():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "-r requirements.lock" in requirements
    assert "COPY requirements.txt requirements.lock ./" in dockerfile


def test_container_uses_pinned_uvicorn_as_the_asgi_runtime():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lockfile = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "uvicorn==0.34.0" in lockfile
    assert "exec uvicorn backend.asgi:app" in dockerfile
    assert "flask --app" not in dockerfile


def test_compose_defaults_bootstrap_registration_to_disabled():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    env_example = (ROOT / ".env.example").read_text(encoding="utf-8")

    assert "ALLOW_BOOTSTRAP_REGISTRATION=${ALLOW_BOOTSTRAP_REGISTRATION:-false}" in compose
    assert "ALLOW_BOOTSTRAP_REGISTRATION=false" in env_example
    assert "MAX_CSV_INGESTION_ROWS=${MAX_CSV_INGESTION_ROWS:-1000}" in compose
    assert "MAX_CSV_INGESTION_ROWS=1000" in env_example
