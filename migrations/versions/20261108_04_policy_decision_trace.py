"""Persist the versioned policy decision with each risk assessment.

Revision ID: 20261108_04
Revises: 20261102_03
"""

from alembic import op
import sqlalchemy as sa


revision = "20261108_04"
down_revision = "20261102_03"
branch_labels = None
depends_on = None


def _columns(bind) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns("risk_assessments")}


def upgrade() -> None:
    if "policy_decision" not in _columns(op.get_bind()):
        # Existing rows predate the policy trace.  The empty object explicitly
        # denotes unavailable historic decision detail rather than inventing it.
        with op.batch_alter_table("risk_assessments") as batch:
            batch.add_column(sa.Column("policy_decision", sa.JSON(), nullable=False, server_default="{}"))
            batch.alter_column("policy_decision", server_default=None)


def downgrade() -> None:
    if "policy_decision" in _columns(op.get_bind()):
        with op.batch_alter_table("risk_assessments") as batch:
            batch.drop_column("policy_decision")
