"""Add transaction facts and retry-safe aggregation jobs.

Revision ID: 20261102_03
Revises: 20261102_02
"""

from alembic import op
import sqlalchemy as sa


revision = "20261102_03"
down_revision = "20261102_02"
branch_labels = None
depends_on = None


def _tables(bind) -> set[str]:
    return set(sa.inspect(bind).get_table_names())


def _columns(bind, table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(bind).get_columns(table)}


def _indexes(bind, table: str) -> set[str]:
    return {index["name"] for index in sa.inspect(bind).get_indexes(table)}


def upgrade() -> None:
    bind = op.get_bind()
    tables = _tables(bind)
    if "source_kind" not in _columns(bind, "merchant_day_observations"):
        # Existing aggregate uploads remain direct merchant-day records.  A
        # temporary default only backfills these historical rows.
        with op.batch_alter_table("merchant_day_observations") as batch:
            batch.add_column(sa.Column("source_kind", sa.String(length=32), nullable=False, server_default="merchant_day"))
            batch.alter_column("source_kind", server_default=None)

    if "transaction_records" not in tables:
        op.create_table(
            "transaction_records",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("organization_id", sa.String(length=36), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("merchant_id", sa.String(length=36), sa.ForeignKey("merchants.id", ondelete="CASCADE"), nullable=False),
            sa.Column("ingestion_batch_id", sa.String(length=36), sa.ForeignKey("ingestion_batches.id", ondelete="RESTRICT"), nullable=False),
            sa.Column("external_id", sa.String(length=128), nullable=False),
            sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("amount", sa.Float(), nullable=False),
            sa.Column("outcome", sa.String(length=16), nullable=False),
            sa.Column("user_token", sa.String(length=128), nullable=False),
            sa.Column("device_token", sa.String(length=128), nullable=False),
            sa.Column("region_code", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("organization_id", "external_id", name="uq_transaction_organization_external"),
        )
        tables.add("transaction_records")
    transaction_indexes = _indexes(bind, "transaction_records")
    for name, columns in (
        ("ix_transaction_records_organization_id", ["organization_id"]),
        ("ix_transaction_records_merchant_id", ["merchant_id"]),
        ("ix_transaction_records_ingestion_batch_id", ["ingestion_batch_id"]),
        ("ix_transaction_records_occurred_at", ["occurred_at"]),
    ):
        if name not in transaction_indexes:
            op.create_index(name, "transaction_records", columns)

    if "ingestion_jobs" not in tables:
        op.create_table(
            "ingestion_jobs",
            sa.Column("id", sa.String(length=36), primary_key=True),
            sa.Column("organization_id", sa.String(length=36), sa.ForeignKey("organizations.id", ondelete="CASCADE"), nullable=False),
            sa.Column("ingestion_batch_id", sa.String(length=36), sa.ForeignKey("ingestion_batches.id", ondelete="CASCADE"), nullable=False),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("status", sa.String(length=16), nullable=False),
            sa.Column("attempt_count", sa.Integer(), nullable=False),
            sa.Column("max_attempts", sa.Integer(), nullable=False),
            sa.Column("next_attempt_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
            sa.Column("last_error", sa.String(length=512), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
            sa.UniqueConstraint("ingestion_batch_id", name="uq_ingestion_job_batch"),
        )
        tables.add("ingestion_jobs")
    job_indexes = _indexes(bind, "ingestion_jobs")
    for name, columns in (
        ("ix_ingestion_jobs_organization_id", ["organization_id"]),
        ("ix_ingestion_jobs_ingestion_batch_id", ["ingestion_batch_id"]),
        ("ix_ingestion_jobs_status", ["status"]),
        ("ix_ingestion_jobs_next_attempt_at", ["next_attempt_at"]),
    ):
        if name not in job_indexes:
            op.create_index(name, "ingestion_jobs", columns)


def downgrade() -> None:
    bind = op.get_bind()
    tables = _tables(bind)
    if "ingestion_jobs" in tables:
        op.drop_table("ingestion_jobs")
    if "transaction_records" in tables:
        op.drop_table("transaction_records")
    if "merchant_day_observations" in tables and "source_kind" in _columns(bind, "merchant_day_observations"):
        with op.batch_alter_table("merchant_day_observations") as batch:
            batch.drop_column("source_kind")
