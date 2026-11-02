"""Regression coverage for strict transaction-ingestion input validation."""

import pytest

from backend.transaction_validation import TransactionValidationError, validate_transactions


def valid_transaction():
    return {
        "transaction_id": "txn-1001",
        "merchant_id": "merchant-1001",
        "occurred_at": "2026-01-01T12:00:00+05:30",
        "amount": 42.5,
        "outcome": "succeeded",
        "user_token": "user-token-1001",
        "device_token": "device-token-1001",
        "region_code": "in-ka",
    }


def test_transaction_validation_canonicalizes_safe_fields_and_utc_timestamp():
    record = validate_transactions([valid_transaction()])[0]
    assert record["occurred_at"] == "2026-01-01T06:30:00.000000Z"
    assert record["date"] == "2026-01-01"
    assert record["outcome"] == "SUCCEEDED"
    assert record["region_code"] == "IN-KA"


@pytest.mark.parametrize("field,value,reason", [
    ("amount", float("nan"), "amount must be finite"),
    ("amount", float("inf"), "amount must be finite"),
    ("amount", -1, "amount cannot be negative"),
    ("occurred_at", "2026-01-01T12:00:00", "explicit timezone"),
    ("outcome", "PENDING", "outcome must be either"),
    ("user_token", "user@example.test", "user_token must be"),
    ("region_code", "not a region", "region_code must be"),
])
def test_transaction_validation_rejects_malformed_fields(field, value, reason):
    transaction = valid_transaction()
    transaction[field] = value
    with pytest.raises(TransactionValidationError, match=reason):
        validate_transactions([transaction])


def test_transaction_validation_rejects_duplicate_ids_and_unexpected_fields():
    first = valid_transaction()
    second = valid_transaction()
    second["merchant_id"] = "merchant-1002"
    with pytest.raises(TransactionValidationError, match="duplicate transaction_id"):
        validate_transactions([first, second])
    first["unsafe_extra"] = "value"
    with pytest.raises(TransactionValidationError, match="unexpected"):
        validate_transactions([first])
