"""Central configuration for the Clinic Support Ticket Classifier.

Every tunable constant lives here so that the rest of the code base contains
no magic numbers. Values are chosen up front and are *not* adjusted by looking
at held-out test results.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tickets.csv"
DEMO_BATCH_PATH = PROJECT_ROOT / "data" / "demo_batch.csv"

#: Environment variable that overrides where model artifacts are written/read.
RESULTS_ENV_VAR = "CLINIC_RESULTS_DIR"

# --- Data schema -----------------------------------------------------------
LABELS: tuple[str, ...] = ("appointment", "billing", "login", "technical")
REQUIRED_COLUMNS: tuple[str, ...] = ("ticket_id", "text", "category")
MIN_EXAMPLES_PER_CLASS = 6  # needed for a stratified split plus cross-validation

# --- Reproducibility and splitting -----------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.25
CV_FOLDS = 5

# --- Model hyper-parameters (defaults; --tune searches TUNING_GRID) ---------
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
LOGREG_C = 10.0
LOGREG_MAX_ITER = 1000
TUNING_GRID: dict[str, list] = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [1.0, 3.0, 10.0, 30.0],
}

# --- Input limits ------------------------------------------------------------
MAX_TICKET_CHARS = 1000
MAX_BATCH_ROWS = 1000

# --- Triage policy ---------------------------------------------------------
# A ticket is flagged for human review when the top estimated probability is
# below REVIEW_MIN_PROBABILITY, when the gap to the runner-up is below
# REVIEW_MIN_MARGIN, or when none of its words are known to the model.
REVIEW_MIN_PROBABILITY = 0.45
REVIEW_MIN_MARGIN = 0.10

# --- Artifact file names ---------------------------------------------------
MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "model_metadata.json"
EVALUATION_FILENAME = "evaluation.json"
TEST_PREDICTIONS_FILENAME = "test_predictions.csv"
MISCLASSIFIED_FILENAME = "misclassified.csv"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"


def results_dir() -> Path:
    """Return the directory that holds trained-model artifacts.

    Defaults to ``<project>/results`` but can be redirected with the
    ``CLINIC_RESULTS_DIR`` environment variable (used by the tests).
    """
    override = os.environ.get(RESULTS_ENV_VAR)
    return Path(override) if override else PROJECT_ROOT / "results"
