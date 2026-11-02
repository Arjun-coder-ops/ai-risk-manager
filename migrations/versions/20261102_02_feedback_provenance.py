"""Add analyst-feedback note and verification provenance.

Revision ID: 20261102_02
Revises: 20261102_01
"""

from alembic import op
import sqlalchemy as sa


revision = "20261102_02"
down_revision = "20261102_01"
branch_labels = None
depends_on = None


def _columns(bind) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns("analyst_feedback")}


def upgrade() -> None:
    bind = op.get_bind()
    existing = _columns(bind)
    added_notes = "notes" not in existing
    # Revision 20261102_01 historically constructs metadata dynamically. On a
    # clean installation it will already have these columns, while an existing
    # 2A/2B database needs this upgrade. Inspecting first makes both paths
    # repeatable without mutating a committed migration history.
    with op.batch_alter_table("analyst_feedback") as batch:
        if added_notes:
            batch.add_column(sa.Column("notes", sa.String(length=2000), nullable=False, server_default=""))
        if "verified_by_user_id" not in existing:
            batch.add_column(sa.Column("verified_by_user_id", sa.String(length=36), nullable=True))
            batch.create_foreign_key(
                "fk_analyst_feedback_verified_by_user",
                "users",
                ["verified_by_user_id"],
                ["id"],
                ondelete="SET NULL",
            )
        if "verified_at" not in existing:
            batch.add_column(sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True))
        if added_notes:
            # The temporary default backfills historical rows only; the API
            # requires a real note for every new feedback submission.
            batch.alter_column("notes", server_default=None)


def downgrade() -> None:
    bind = op.get_bind()
    existing = _columns(bind)
    with op.batch_alter_table("analyst_feedback") as batch:
        if "verified_by_user_id" in existing:
            batch.drop_constraint("fk_analyst_feedback_verified_by_user", type_="foreignkey")
            batch.drop_column("verified_by_user_id")
        if "verified_at" in existing:
            batch.drop_column("verified_at")
        if "notes" in existing:
            batch.drop_column("notes")
