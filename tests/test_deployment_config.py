from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_docker_build_copies_the_lockfile_required_by_requirements_txt():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "-r requirements.lock" in requirements
    assert "COPY requirements.txt requirements.lock ./" in dockerfile


def test_container_uses_pinned_gunicorn_instead_of_flask_development_server():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lockfile = (ROOT / "requirements.lock").read_text(encoding="utf-8")

    assert "gunicorn==23.0.0" in lockfile
    assert "exec gunicorn" in dockerfile
    assert "flask --app" not in dockerfile
