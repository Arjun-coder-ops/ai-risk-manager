"""Add the explicit investigator membership role.

Revision ID: 20261108_05
Revises: 20261108_04
"""

from alembic import op


revision = "20261108_05"
down_revision = "20261108_04"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("memberships") as batch:
        batch.drop_constraint("ck_membership_role", type_="check")
        batch.create_check_constraint(
            "ck_membership_role", "role IN ('admin', 'analyst', 'investigator', 'viewer')"
        )


def downgrade() -> None:
    with op.batch_alter_table("memberships") as batch:
        batch.drop_constraint("ck_membership_role", type_="check")
        batch.create_check_constraint("ck_membership_role", "role IN ('admin', 'analyst', 'viewer')")
