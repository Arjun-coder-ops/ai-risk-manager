"""Database-backed worker for retry-safe transaction aggregation.

This is deliberately a bounded batch worker, not a real-time streaming claim.
Raw transactions commit with their ingestion batch before a job is queued.  A
retry recomputes each affected merchant-day from those immutable records, so a
crash between aggregation and acknowledgement cannot double-count data.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta, timezone
import math
import time as wall_time

from sqlalchemy import select

from backend.database import get_session
from backend.models import (
    AuditEvent,
    IngestionBatch,
    IngestionJob,
    MerchantDayObservation,
    TransactionRecord,
    utcnow,
)


def _as_utc(value: datetime) -> datetime:
    """SQLite may return naive values despite a timezone-aware model column."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _day_bounds(observation_date: date) -> tuple[datetime, datetime]:
    start = datetime.combine(observation_date, time.min, tzinfo=timezone.utc)
    return start, start + timedelta(days=1)


def _write_audit(session, job: IngestionJob, action: str, payload: dict) -> None:
    session.add(AuditEvent(
        organization_id=job.organization_id,
        actor_user_id=None,
        action=action,
        entity_type="ingestion_job",
        entity_id=job.id,
        payload=payload,
    ))


def claim_next_job() -> str | None:
    """Atomically claim one due job; PostgreSQL workers use SKIP LOCKED."""
    session = get_session()
    try:
        now = utcnow()
        statement = (
            select(IngestionJob)
            .where(
                IngestionJob.status.in_(("PENDING", "RETRY")),
                IngestionJob.next_attempt_at <= now,
            )
            .order_by(IngestionJob.created_at, IngestionJob.id)
            .limit(1)
        )
        if session.bind is not None and session.bind.dialect.name == "postgresql":
            statement = statement.with_for_update(skip_locked=True)
        job = session.scalar(statement)
        if job is None:
            return None
        batch = session.get(IngestionBatch, job.ingestion_batch_id)
        if batch is None:
            raise RuntimeError("Ingestion job references a missing batch.")
        job.status = "PROCESSING"
        job.attempt_count += 1
        job.locked_at = now
        batch.status = "PROCESSING"
        session.commit()
        return job.id
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def _aggregate_day(session, *, organization_id: str, merchant_id: str, observation_date: date, batch_id: str) -> None:
    """Rebuild one merchant-day from durable transaction facts.

    Rebuilding is idempotent.  It also correctly revises ``n_new_users`` when
    a backfilled transaction establishes an earlier first-seen date.
    """
    start, end = _day_bounds(observation_date)
    history = session.scalars(
        select(TransactionRecord).where(
            TransactionRecord.organization_id == organization_id,
            TransactionRecord.merchant_id == merchant_id,
            TransactionRecord.occurred_at < end,
        )
    ).all()
    day_records = [record for record in history if start <= _as_utc(record.occurred_at) < end]
    if not day_records:
        raise RuntimeError("A queued transaction job has no matching records to aggregate.")

    existing = session.scalar(select(MerchantDayObservation).where(
        MerchantDayObservation.organization_id == organization_id,
        MerchantDayObservation.merchant_id == merchant_id,
        MerchantDayObservation.observation_date == observation_date,
    ))
    if existing is not None and existing.source_kind != "transaction_aggregate":
        raise RuntimeError("Transaction aggregate conflicts with an existing direct merchant-day observation.")

    first_seen: dict[str, datetime] = {}
    for record in history:
        occurred_at = _as_utc(record.occurred_at)
        prior = first_seen.get(record.user_token)
        if prior is None or occurred_at < prior:
            first_seen[record.user_token] = occurred_at

    amounts = [record.amount for record in day_records]
    mean_amount = sum(amounts) / len(amounts)
    amount_std = math.sqrt(sum((amount - mean_amount) ** 2 for amount in amounts) / len(amounts))
    daily_users = {record.user_token for record in day_records}
    values = {
        "n_transactions": len(day_records),
        "n_failed": sum(record.outcome == "FAILED" for record in day_records),
        "avg_amount": mean_amount,
        "amount_std": amount_std,
        "n_unique_users": len(daily_users),
        "n_new_users": sum(start <= first_seen[user] < end for user in daily_users),
        "n_devices": len({record.device_token for record in day_records}),
        "distinct_regions_seen": len({record.region_code for record in day_records}),
    }
    if existing is None:
        session.add(MerchantDayObservation(
            organization_id=organization_id,
            merchant_id=merchant_id,
            ingestion_batch_id=batch_id,
            observation_date=observation_date,
            source_kind="transaction_aggregate",
            **values,
        ))
    else:
        for field, value in values.items():
            setattr(existing, field, value)


def process_claimed_job(job_id: str) -> None:
    """Process a claimed job and commit all affected merchant-day aggregates."""
    session = get_session()
    try:
        job = session.get(IngestionJob, job_id)
        if job is None or job.status != "PROCESSING":
            return
        batch = session.get(IngestionBatch, job.ingestion_batch_id)
        if batch is None:
            raise RuntimeError("Ingestion job references a missing batch.")
        transaction_records = session.scalars(select(TransactionRecord).where(
            TransactionRecord.ingestion_batch_id == batch.id,
        )).all()
        affected = {(record.merchant_id, _as_utc(record.occurred_at).date()) for record in transaction_records}
        if not affected:
            raise RuntimeError("Ingestion batch contains no durable transaction records.")
        for merchant_id, observation_date in sorted(affected, key=lambda item: (item[0], item[1])):
            _aggregate_day(
                session,
                organization_id=job.organization_id,
                merchant_id=merchant_id,
                observation_date=observation_date,
                batch_id=batch.id,
            )
        job.status = "SUCCEEDED"
        job.completed_at = utcnow()
        job.last_error = None
        batch.status = "SUCCEEDED"
        batch.error_summary = None
        _write_audit(session, job, "transaction_ingestion.processed", {
            "batch_id": batch.id,
            "transaction_count": len(transaction_records),
            "merchant_day_count": len(affected),
            "attempt_count": job.attempt_count,
        })
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def mark_retry_or_failure(job_id: str, error: Exception) -> None:
    """Persist retry state separately so a processing rollback cannot lose it."""
    session = get_session()
    try:
        job = session.get(IngestionJob, job_id)
        if job is None or job.status != "PROCESSING":
            return
        batch = session.get(IngestionBatch, job.ingestion_batch_id)
        if batch is None:
            raise RuntimeError("Ingestion job references a missing batch.")
        error_type = type(error).__name__
        job.last_error = error_type
        if job.attempt_count >= job.max_attempts:
            job.status = "FAILED"
            job.completed_at = utcnow()
            batch.status = "FAILED"
            batch.error_summary = {
                "source": "transaction_api",
                "processing_state": "failed",
                "attempt_count": job.attempt_count,
                "reason": "Transaction aggregation failed after the configured retry limit.",
            }
            action = "transaction_ingestion.failed"
        else:
            delay_seconds = min(300, 5 * (2 ** (job.attempt_count - 1)))
            job.status = "RETRY"
            job.next_attempt_at = utcnow() + timedelta(seconds=delay_seconds)
            batch.status = "QUEUED"
            batch.error_summary = {
                "source": "transaction_api",
                "processing_state": "retry_scheduled",
                "attempt_count": job.attempt_count,
            }
            action = "transaction_ingestion.retry_scheduled"
        _write_audit(session, job, action, {"batch_id": batch.id, "error_type": error_type, "attempt_count": job.attempt_count})
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def run_once() -> bool:
    """Claim and execute one job.  Returns ``True`` only when work was found."""
    job_id = claim_next_job()
    if job_id is None:
        return False
    try:
        process_claimed_job(job_id)
    except Exception as exc:
        mark_retry_or_failure(job_id, exc)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Process durable transaction-ingestion jobs.")
    parser.add_argument("--once", action="store_true", help="Process at most one currently due job, then exit.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Idle delay for continuous mode (default: 5).")
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be greater than zero.")
    if args.once:
        run_once()
        return
    while True:
        if not run_once():
            wall_time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
