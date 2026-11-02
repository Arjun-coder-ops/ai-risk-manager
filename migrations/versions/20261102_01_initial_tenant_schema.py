"""Create the initial durable tenant schema.

Revision ID: 20261102_01
Revises: None
"""

from alembic import op

from backend import models  # noqa: F401 - registers all metadata
from backend.database import Base


revision = "20261102_01"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind())


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())
