"""Load, tidy, split and fingerprint the support-ticket dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_SEED, REQUIRED_COLUMNS, TEST_SIZE
from src.validation import DatasetValidationError, validate_dataset


def _tidy(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, trim whitespace and lower-case category labels."""
    df = df.rename(columns=lambda name: str(name).strip().lower()).copy()
    for column in ("ticket_id", "text"):
        if column in df.columns:
            df[column] = df[column].str.strip()
    if "category" in df.columns:
        df["category"] = df["category"].str.strip().str.lower()
    return df


def load_dataset(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Read a ticket CSV, tidy it and validate it.

    Raises:
        DatasetValidationError: if the file is missing, unreadable or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetValidationError(f"Dataset file not found: {path}")
    try:
        raw = pd.read_csv(path, dtype=str)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise DatasetValidationError(f"Could not read dataset {path}: {exc}") from exc
    return validate_dataset(_tidy(raw))


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a reproducible, label-stratified ``(train, test)`` split."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["category"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """SHA-256 of the dataset content, independent of row order and file formatting."""
    canonical = df[list(REQUIRED_COLUMNS)].sort_values("ticket_id").to_csv(index=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
