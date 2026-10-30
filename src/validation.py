"""Input and dataset validation with clear, actionable error messages."""

from __future__ import annotations

import pandas as pd

from src.config import LABELS, MAX_TICKET_CHARS, MIN_EXAMPLES_PER_CLASS, REQUIRED_COLUMNS


class ValidationError(ValueError):
    """Base class for all validation problems raised by this project."""


class InvalidTicketTextError(ValidationError):
    """Raised when a single ticket text cannot be classified."""


class DatasetValidationError(ValidationError):
    """Raised when a dataset file or DataFrame violates the expected schema."""


def validate_ticket_text(text: object, max_chars: int = MAX_TICKET_CHARS) -> str:
    """Return the cleaned ticket text or raise :class:`InvalidTicketTextError`.

    Rejects missing values, non-strings, empty or whitespace-only text, and
    text longer than ``max_chars`` characters (after trimming). Leading and
    trailing whitespace is removed from accepted text.
    """
    if text is None or (isinstance(text, float) and text != text):
        raise InvalidTicketTextError("Ticket text is missing. Describe the problem in a sentence or two.")
    if not isinstance(text, str):
        raise InvalidTicketTextError(f"Ticket text must be a string, got {type(text).__name__}.")
    cleaned = text.strip()
    if not cleaned:
        raise InvalidTicketTextError(
            "Ticket text is empty. Describe the problem in a sentence or two."
        )

    if len(cleaned) > max_chars:
        raise InvalidTicketTextError(
            f"Ticket text is too long ({len(cleaned)} characters). "
            f"Please shorten it to at most {max_chars} characters."
        )
    return cleaned


def validate_dataset(
    df: pd.DataFrame,
    labels: tuple[str, ...] = LABELS,
    min_per_class: int = MIN_EXAMPLES_PER_CLASS,
) -> pd.DataFrame:
    """Check that a ticket DataFrame is safe to train on and return it unchanged.

    Verifies, in order: required columns, non-empty data, no missing values,
    no blank ticket text, only known category labels, unique ``ticket_id``
    values, and at least ``min_per_class`` examples for every label.
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DatasetValidationError(
            f"Dataset is missing required column(s): {', '.join(missing_columns)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )
    if df.empty:
        raise DatasetValidationError("Dataset contains no rows.")

    required = df[list(REQUIRED_COLUMNS)]
    columns_with_nulls = [col for col in REQUIRED_COLUMNS if required[col].isna().any()]
    if columns_with_nulls:
        raise DatasetValidationError(
            f"Dataset has missing values in column(s): {', '.join(columns_with_nulls)}."
        )

    blank = df["text"].str.strip() == ""
    if blank.any():
        ids = df.loc[blank, "ticket_id"].tolist()[:5]
        raise DatasetValidationError(f"Dataset has blank ticket text for ticket_id(s): {ids}.")

    unknown = sorted(set(df["category"]) - set(labels))
    if unknown:
        raise DatasetValidationError(
            f"Invalid category label(s): {unknown}. Allowed labels: {list(labels)}."
        )

    if df.duplicated().any():
        duplicate_ids = df.loc[df.duplicated(), "ticket_id"].tolist()[:5]
        raise DatasetValidationError(f"Duplicate ticket_id value(s) found: {duplicate_ids}.")

    counts = df["category"].value_counts()
    too_small = {label: int(counts.get(label, 0)) for label in labels if counts.get(label, 0) < min_per_class}
    if too_small:
        raise DatasetValidationError(
            f"Each category needs at least {min_per_class} examples; too few for: {too_small}."
        )
    return df
