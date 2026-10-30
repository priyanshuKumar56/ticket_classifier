"""Shared fixtures: a tiny in-memory dataset, a trained pipeline and a results folder."""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import DATA_PATH
from src.data_loader import load_dataset
from src.model import build_pipeline, train_model
from src.training import run_training

_TINY_TEXTS = {
    "appointment": [
        "Please reschedule my appointment to next week",
        "The calendar shows a double booked slot for the doctor",
        "Reminder messages for the visit are not sent",
        "I want to book a follow up appointment",
        "Cancel my booking for Friday morning",
        "The doctor schedule has no free slots",
        "Patient missed the appointment and needs a new slot",
        "Waitlist should offer the cancelled booking",
    ],
    "billing": [
        "I was charged twice on my invoice",
        "Please refund the payment for the visit",
        "The invoice total is wrong",
        "Insurance claim was rejected for the bill",
        "Update the credit card for our subscription",
        "Charged a late fee on the statement",
        "Receipt shows the wrong amount paid",
        "Need a discount applied to the balance due",
    ],
    "login": [
        "I forgot my password and cannot sign in",
        "My account is locked after failed login attempts",
        "The verification code is rejected",
        "Password reset email never arrives",
        "Cannot log in with single sign on",
        "Invitation link to create an account expired",
        "Two factor authentication code is not accepted",
        "Please unlock my user account",
    ],
    "technical": [
        "The app crashes when I open the report",
        "Error 500 appears on export",
        "The page is very slow and times out",
        "Sync with the lab system stopped working",
        "Printed document is cut off at the edge",
        "The mobile app freezes on startup",
        "Upload fails with an unexpected error",
        "Blank screen after saving the form",
    ],
}


@pytest.fixture(scope="session")
def tiny_df() -> pd.DataFrame:
    """32 short labelled tickets (8 per category) for fast, deterministic tests."""
    rows = []
    for label, texts in _TINY_TEXTS.items():
        for text in texts:
            rows.append({"ticket_id": f"F-{len(rows) + 1:03d}", "text": text, "category": label})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_pipeline(tiny_df):
    """A pipeline trained on the tiny dataset."""
    return train_model(tiny_df, build_pipeline())


@pytest.fixture(scope="session")
def full_df() -> pd.DataFrame:
    """The real 120-ticket synthetic dataset, loaded and validated."""
    return load_dataset(DATA_PATH)


@pytest.fixture(scope="session")
def trained_results_dir(tmp_path_factory):
    """Train once on the full dataset into a temporary folder and return it."""
    directory = tmp_path_factory.mktemp("results")
    run_training(DATA_PATH, directory)
    return directory
