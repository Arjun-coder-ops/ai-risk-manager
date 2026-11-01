"""Validation utilities for merchant-day observations.

The model operates on one record per merchant per UTC calendar day.  Keeping
this contract in one small dependency-free module ensures feature engineering,
single-merchant inference, and portfolio inference apply the same rules.
"""

from __future__ import annotations

from datetime import date, datetime
import re

import numpy as np
import pandas as pd


DATE_FORMAT = "%Y-%m-%d"
MERCHANT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")

REQUIRED_INPUT_FIELDS = [
    "merchant_id",
    "date",
    "n_transactions",
    "n_failed",
    "avg_amount",
    "amount_std",
    "n_unique_users",
    "n_new_users",
    "n_devices",
    "distinct_regions_seen",
]

INTEGER_COUNT_FIELDS = [
    "n_transactions",
    "n_failed",
    "n_unique_users",
    "n_new_users",
    "n_devices",
    "distinct_regions_seen",
]
NON_NEGATIVE_NUMERIC_FIELDS = ["avg_amount", "amount_std"]


class ObservationValidationError(ValueError):
    """Raised when an observation does not satisfy the model input contract."""


def normalize_date(value: object) -> str:
    """Return a strict ISO calendar date or raise a stable validation error."""
    if isinstance(value, str):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ObservationValidationError("date must use YYYY-MM-DD format.")
        try:
            datetime.strptime(value, DATE_FORMAT)
        except ValueError as exc:
            raise ObservationValidationError("date must be a valid calendar date.") from exc
        return value

    if isinstance(value, (pd.Timestamp, datetime, date)):
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp != timestamp.normalize():
            raise ObservationValidationError("date must be a calendar date without a time component.")
        return timestamp.strftime(DATE_FORMAT)

    raise ObservationValidationError("date must use YYYY-MM-DD format.")


def normalize_merchant_id(value: object) -> str:
    if not isinstance(value, str):
        raise ObservationValidationError("merchant_id must be a non-empty string.")
    merchant_id = value.strip()
    if not MERCHANT_ID_PATTERN.fullmatch(merchant_id):
        raise ObservationValidationError(
            "merchant_id must be 1-128 characters using letters, digits, '.', '_', ':', or '-'."
        )
    return merchant_id


def validate_request_identifiers(merchant_id: object, date_value: object) -> tuple[str, str]:
    """Validate identifiers received at the HTTP boundary."""
    return normalize_merchant_id(merchant_id), normalize_date(date_value)


def _as_finite_number(value: object, field: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ObservationValidationError(f"{field} must be numeric, not boolean.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationValidationError(f"{field} must be numeric.") from exc
    if not np.isfinite(numeric):
        raise ObservationValidationError(f"{field} must be finite.")
    return numeric


def validate_observations(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a merchant-day frame.

    Dates and merchant identifiers are canonicalized so joins have a stable
    key.  The returned frame is a copy; callers never mutate their input.
    """
    if not isinstance(df, pd.DataFrame):
        raise ObservationValidationError("Observations must be a tabular DataFrame.")
    if df.empty:
        raise ObservationValidationError("No transaction history provided.")

    missing = [field for field in REQUIRED_INPUT_FIELDS if field not in df.columns]
    if missing:
        raise ObservationValidationError(f"Missing required fields: {missing}")

    output = df.copy()
    normalized_ids: list[str] = []
    normalized_dates: list[str] = []
    for index, row in output.iterrows():
        try:
            normalized_ids.append(normalize_merchant_id(row["merchant_id"]))
            normalized_dates.append(normalize_date(row["date"]))
        except ObservationValidationError as exc:
            raise ObservationValidationError(f"Row {index}: {exc}") from exc
    output["merchant_id"] = normalized_ids
    output["date"] = normalized_dates

    for field in INTEGER_COUNT_FIELDS:
        values = []
        for index, value in output[field].items():
            try:
                numeric = _as_finite_number(value, field)
            except ObservationValidationError as exc:
                raise ObservationValidationError(f"Row {index}: {exc}") from exc
            if numeric < 0:
                raise ObservationValidationError(f"Row {index}: {field} cannot be negative.")
            if not numeric.is_integer():
                raise ObservationValidationError(f"Row {index}: {field} must be an integer count.")
            values.append(int(numeric))
        output[field] = values

    for field in NON_NEGATIVE_NUMERIC_FIELDS:
        values = []
        for index, value in output[field].items():
            try:
                numeric = _as_finite_number(value, field)
            except ObservationValidationError as exc:
                raise ObservationValidationError(f"Row {index}: {exc}") from exc
            if numeric < 0:
                raise ObservationValidationError(f"Row {index}: {field} cannot be negative.")
            values.append(numeric)
        output[field] = values

    invalid_counts = output[
        (output["n_failed"] > output["n_transactions"])
        | (output["n_unique_users"] > output["n_transactions"])
        | (output["n_new_users"] > output["n_unique_users"])
        | (output["n_devices"] > output["n_transactions"])
    ]
    if not invalid_counts.empty:
        raise ObservationValidationError(
            "Impossible count relationship: require failed <= transactions, unique_users <= "
            "transactions, new_users <= unique_users, and devices <= transactions."
        )

    zero_transaction_rows = output[output["n_transactions"] == 0]
    if not zero_transaction_rows.empty:
        dependent = [
            "n_failed",
            "n_unique_users",
            "n_new_users",
            "n_devices",
            "distinct_regions_seen",
            "avg_amount",
            "amount_std",
        ]
        if (zero_transaction_rows[dependent] != 0).any(axis=None):
            raise ObservationValidationError(
                "Zero-transaction observations must have zero dependent counts and amounts."
            )

    duplicate_mask = output.duplicated(["merchant_id", "date"], keep=False)
    if duplicate_mask.any():
        duplicate_keys = output.loc[duplicate_mask, ["merchant_id", "date"]].drop_duplicates()
        examples = duplicate_keys.head(3).to_dict(orient="records")
        raise ObservationValidationError(
            f"Duplicate merchant-day observations are not allowed: {examples}"
        )

    return output
