"""Tests for ticket-text and dataset validation."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import LABELS, MAX_TICKET_CHARS
from src.validation import (
    DatasetValidationError,
    InvalidTicketTextError,
    validate_dataset,
    validate_ticket_text,
)


def _dataset(per_class: int = 6) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        for n in range(per_class):
            rows.append({"ticket_id": f"V-{len(rows):03d}", "text": f"{label} sample {n}", "category": label})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ ticket text
def test_valid_text_is_trimmed():
    assert validate_ticket_text("  I cannot log in  ") == "I cannot log in"


@pytest.mark.parametrize("value", ["", None, float("nan")])
def test_empty_or_missing_text_is_rejected(value):
    with pytest.raises(InvalidTicketTextError):
        validate_ticket_text(value)


def test_non_string_text_is_rejected():
    with pytest.raises(InvalidTicketTextError, match="string"):
        validate_ticket_text(12345)


def test_over_long_text_is_rejected_with_the_limit_in_the_message():
    with pytest.raises(InvalidTicketTextError, match=str(MAX_TICKET_CHARS)):
        validate_ticket_text("a" * (MAX_TICKET_CHARS + 1))


def test_text_at_the_limit_is_accepted():
    assert len(validate_ticket_text("a" * MAX_TICKET_CHARS)) == MAX_TICKET_CHARS


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Known defect A (see README): whitespace-only ticket text is accepted instead of rejected.",
)
def test_whitespace_only_text_is_rejected_with_a_helpful_message():
    with pytest.raises(InvalidTicketTextError, match="empty"):
        validate_ticket_text("   \n\t  ")


# ---------------------------------------------------------------------- dataset
def test_valid_dataset_passes():
    df = _dataset()
    assert validate_dataset(df) is df


def test_invalid_category_label_is_rejected():
    df = _dataset()
    df.loc[0, "category"] = "refund"
    with pytest.raises(DatasetValidationError, match="refund"):
        validate_dataset(df)


def test_missing_values_are_rejected():
    df = _dataset()
    df.loc[1, "text"] = None
    with pytest.raises(DatasetValidationError, match="missing values"):
        validate_dataset(df)


def test_blank_text_rows_are_rejected():
    df = _dataset()
    df.loc[2, "text"] = "   "
    with pytest.raises(DatasetValidationError, match="blank"):
        validate_dataset(df)


def test_too_few_examples_per_category_is_rejected():
    with pytest.raises(DatasetValidationError, match="at least"):
        validate_dataset(_dataset(per_class=2))


def test_exact_duplicate_rows_are_rejected():
    df = _dataset()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        validate_dataset(df)


@pytest.mark.xfail(
    strict=True,
    raises=pytest.fail.Exception,
    reason="Known defect D (see README): duplicate ticket_id values are not detected "
    "unless the entire row is identical.",
)
def test_duplicate_ticket_ids_with_different_text_are_rejected():
    df = _dataset()
    df.loc[1, "ticket_id"] = df.loc[0, "ticket_id"]
    with pytest.raises(DatasetValidationError, match="Duplicate"):
        validate_dataset(df)
