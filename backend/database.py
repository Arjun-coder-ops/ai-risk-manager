"""Database lifecycle helpers shared by the Flask application and Alembic."""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    """Declarative base for durable tenant-scoped entities."""


_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None
_configured_url: str | None = None


def database_url() -> str:
    """Return the configured URL, normalizing common PostgreSQL aliases."""
    default_path = Path(__file__).resolve().parents[1] / "data" / "ai_risk_manager.db"
    url = os.environ.get("DATABASE_URL", f"sqlite:///{default_path.as_posix()}")
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def configure_database(url: str | None = None) -> None:
    """Reset lazy database state; used by isolated integration tests."""
    global _engine, _session_factory, _configured_url
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _session_factory = None
    _configured_url = url


def get_engine() -> Engine:
    global _engine, _session_factory
    if _engine is None:
        url = _configured_url or database_url()
        options: dict = {"future": True, "pool_pre_ping": not url.startswith("sqlite")}
        if url.startswith("sqlite"):
            options["connect_args"] = {"check_same_thread": False}
        _engine = create_engine(url, **options)
        _session_factory = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)
    return _engine


def get_session() -> Session:
    get_engine()
    assert _session_factory is not None
    return _session_factory()


def create_schema() -> None:
    """Create schema only for local development and isolated tests.

    Production environments must apply Alembic revisions instead.
    """
    # Import registers the models with Base.metadata without creating an
    # import cycle for callers that only need the database URL.
    from backend import models  # noqa: F401

    Base.metadata.create_all(get_engine())
