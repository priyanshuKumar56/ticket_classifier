"""Tests for dataset loading, splitting and fingerprinting."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import LABELS, REQUIRED_COLUMNS
from src.data_loader import dataset_fingerprint, load_dataset, split_dataset
from src.validation import DatasetValidationError


def _write_csv(path, rows, header=("ticket_id", "text", "category")):
    pd.DataFrame(rows, columns=list(header)).to_csv(path, index=False)


def _valid_rows(per_class: int = 6):
    return [
        (f"X-{i:03d}", f"{label} example number {n}", label)
        for i, (label, n) in enumerate((l, n) for l in LABELS for n in range(per_class))
    ]


def test_real_dataset_has_required_columns(full_df):
    assert list(full_df.columns) == list(REQUIRED_COLUMNS)


def test_real_dataset_size_and_labels(full_df):
    assert 80 <= len(full_df) <= 120
    counts = full_df["category"].value_counts()
    assert set(counts.index) == set(LABELS)
    assert counts.max() / counts.min() <= 1.5, "labels should be reasonably balanced"


def test_real_dataset_ids_and_texts_are_unique(full_df):
    assert full_df["ticket_id"].is_unique
    assert full_df["text"].is_unique


def test_missing_file_gives_clear_error(tmp_path):
    with pytest.raises(DatasetValidationError, match="not found"):
        load_dataset(tmp_path / "nope.csv")


def test_missing_required_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"ticket_id": ["a"], "text": ["hello"]}).to_csv(path, index=False)
    with pytest.raises(DatasetValidationError, match="category"):
        load_dataset(path)


def test_empty_file_gives_clear_error(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="Could not read"):
        load_dataset(path)


def test_loader_trims_whitespace_and_lowercases_labels(tmp_path):
    rows = [(i, t.upper() if n % 2 else f"  {t}  ", f" {c.title()} ") for n, (i, t, c) in enumerate(_valid_rows())]
    path = tmp_path / "messy.csv"
    _write_csv(path, rows)
    loaded = load_dataset(path)
    assert set(loaded["category"]) == set(LABELS)
    assert not loaded["text"].str.startswith(" ").any()


def test_split_is_reproducible(full_df):
    first_train, first_test = split_dataset(full_df, seed=7)
    second_train, second_test = split_dataset(full_df, seed=7)
    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)


def test_split_changes_with_seed(full_df):
    _, test_a = split_dataset(full_df, seed=1)
    _, test_b = split_dataset(full_df, seed=2)
    assert set(test_a["ticket_id"]) != set(test_b["ticket_id"])


def test_split_has_no_overlap_and_keeps_every_row(full_df):
    train_df, test_df = split_dataset(full_df)
    assert set(train_df["ticket_id"]).isdisjoint(test_df["ticket_id"])
    assert len(train_df) + len(test_df) == len(full_df)


def test_split_is_stratified(full_df):
    _, test_df = split_dataset(full_df)
    expected = full_df["category"].value_counts() * len(test_df) / len(full_df)
    actual = test_df["category"].value_counts()
    for label in LABELS:
        assert abs(actual[label] - expected[label]) <= 1


def test_fingerprint_is_stable_and_content_sensitive(full_df):
    assert dataset_fingerprint(full_df) == dataset_fingerprint(full_df.sample(frac=1, random_state=3))
    changed = full_df.copy()
    changed.loc[0, "text"] = changed.loc[0, "text"] + " changed"
    assert dataset_fingerprint(changed) != dataset_fingerprint(full_df)
