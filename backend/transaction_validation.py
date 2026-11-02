"""Strict, privacy-minimising validation for transaction ingestion.

Transactions are accepted only to derive the existing merchant-day aggregate
contract.  ``user_token`` and ``device_token`` must be opaque provider tokens;
callers must never send names, e-mail addresses, card data, or other PII.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
import re
from typing import Any

from model.validation import ObservationValidationError, normalize_merchant_id


TRANSACTION_REQUIRED_FIELDS = (
    "transaction_id",
    "merchant_id",
    "occurred_at",
    "amount",
    "outcome",
    "user_token",
    "device_token",
    "region_code",
)
TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
REGION_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9-]{1,31}$")
OUTCOMES = frozenset({"SUCCEEDED", "FAILED"})


class TransactionValidationError(ValueError):
    """Raised when a transaction cannot safely enter the durable queue."""


def _safe_identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TransactionValidationError(f"{field} must be a non-empty opaque identifier string.")
    normalized = value.strip()
    if not TOKEN_PATTERN.fullmatch(normalized):
        raise TransactionValidationError(
            f"{field} must be 1-128 characters using letters, digits, '.', '_', ':', or '-'."
        )
    return normalized


def _finite_amount(value: object) -> float:
    if isinstance(value, bool):
        raise TransactionValidationError("amount must be numeric, not boolean.")
    try:
        amount = float(value)
    except (TypeError, ValueError) as exc:
        raise TransactionValidationError("amount must be numeric.") from exc
    if not math.isfinite(amount):
        raise TransactionValidationError("amount must be finite.")
    if amount < 0:
        raise TransactionValidationError("amount cannot be negative.")
    return amount


def normalize_occurred_at(value: object) -> datetime:
    """Require a timezone-aware RFC 3339 timestamp and convert it to UTC."""
    if not isinstance(value, str) or not value.strip():
        raise TransactionValidationError("occurred_at must be a timezone-aware RFC 3339 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise TransactionValidationError("occurred_at must be a valid RFC 3339 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TransactionValidationError("occurred_at must include an explicit timezone offset.")
    return parsed.astimezone(timezone.utc)


def _canonical_timestamp(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def validate_transactions(rows: object) -> list[dict[str, Any]]:
    """Return canonical records and reject malformed or conflicting inputs."""
    if not isinstance(rows, list) or not rows:
        raise TransactionValidationError("transactions must be a non-empty JSON array.")

    normalized: list[dict[str, Any]] = []
    seen_transaction_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TransactionValidationError(f"Row {index}: each transaction must be a JSON object.")
        missing = [field for field in TRANSACTION_REQUIRED_FIELDS if field not in row]
        unexpected = sorted(set(row) - set(TRANSACTION_REQUIRED_FIELDS))
        if missing or unexpected:
            raise TransactionValidationError(
                f"Row {index}: transaction must contain exactly the required fields; "
                f"missing={missing}, unexpected={unexpected}."
            )
        try:
            transaction_id = _safe_identifier(row["transaction_id"], "transaction_id")
            merchant_id = normalize_merchant_id(row["merchant_id"])
            occurred_at = normalize_occurred_at(row["occurred_at"])
            amount = _finite_amount(row["amount"])
            outcome = row["outcome"].strip().upper() if isinstance(row["outcome"], str) else ""
            if outcome not in OUTCOMES:
                raise TransactionValidationError("outcome must be either SUCCEEDED or FAILED.")
            user_token = _safe_identifier(row["user_token"], "user_token")
            device_token = _safe_identifier(row["device_token"], "device_token")
            region_code = row["region_code"].strip().upper() if isinstance(row["region_code"], str) else ""
            if not REGION_PATTERN.fullmatch(region_code):
                raise TransactionValidationError(
                    "region_code must be 2-32 uppercase letters, digits, or hyphens."
                )
        except (TransactionValidationError, ObservationValidationError) as exc:
            raise TransactionValidationError(f"Row {index}: {exc}") from exc
        if transaction_id in seen_transaction_ids:
            raise TransactionValidationError(f"Row {index}: duplicate transaction_id in request.")
        seen_transaction_ids.add(transaction_id)
        normalized.append({
            "transaction_id": transaction_id,
            "merchant_id": merchant_id,
            "occurred_at": _canonical_timestamp(occurred_at),
            "date": occurred_at.date().isoformat(),
            "amount": amount,
            "outcome": outcome,
            "user_token": user_token,
            "device_token": device_token,
            "region_code": region_code,
        })
    return normalized
