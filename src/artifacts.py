"""Save and load the trained pipeline with integrity checks.

joblib files are pickles: loading one can execute arbitrary code. The checks
here catch accidental corruption and a swapped or edited model file, but they
cannot protect you from an attacker who can rewrite *both* the model and its
metadata. Only load artifacts you trained yourself.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.pipeline import Pipeline

from src.config import METADATA_FILENAME, MODEL_FILENAME


class ModelArtifactError(RuntimeError):
    """Base class for problems with saved model files."""


class ModelNotFoundError(ModelArtifactError):
    """No trained model exists in the requested directory."""


class ModelIntegrityError(ModelArtifactError):
    """The saved model or its metadata failed an integrity check."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(pipeline: Pipeline, metadata: dict[str, Any], directory: Path) -> Path:
    """Write ``model.joblib`` and ``model_metadata.json`` (with the model checksum).

    The model is written to a temporary file first and then moved into place,
    so a crash cannot leave a half-written model behind.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    temp_path = model_path.with_name(model_path.name + ".tmp")
    joblib.dump(pipeline, temp_path, compress=3)
    os.replace(temp_path, model_path)

    record = {
        **metadata,
        "model_file": MODEL_FILENAME,
        "model_sha256": sha256_file(model_path),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "saved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (directory / METADATA_FILENAME).write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    return model_path


def load_metadata(directory: Path) -> dict[str, Any]:
    """Read the metadata JSON written by :func:`save_model`."""
    meta_path = Path(directory) / METADATA_FILENAME
    if not meta_path.is_file():
        raise ModelNotFoundError(
            f"No trained model found in {directory}. Run `python train.py` first."
        )
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModelIntegrityError(f"Model metadata is not valid JSON: {exc}") from exc


def load_model(directory: Path) -> tuple[Pipeline, dict[str, Any]]:
    """Verify the checksum, then load the pipeline and its metadata.

    Raises:
        ModelNotFoundError: if the model or metadata file is missing.
        ModelIntegrityError: if the checksum does not match or the file does
            not contain a scikit-learn pipeline with ``predict_proba``.
    """
    directory = Path(directory)
    model_path = directory / MODEL_FILENAME
    metadata = load_metadata(directory)
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"No trained model found in {directory}. Run `python train.py` first."
        )

    if sha256_file(model_path) != metadata.get("model_sha256"):
        raise ModelIntegrityError(
            f"{model_path} does not match the checksum recorded in {METADATA_FILENAME}. "
            "The file may be corrupted or replaced. Retrain with `python train.py`."
        )
    saved_version = metadata.get("sklearn_version")
    if saved_version != sklearn.__version__:
        warnings.warn(
            f"Model was trained with scikit-learn {saved_version} but {sklearn.__version__} is "
            "installed. Retrain with `python train.py` if predictions look wrong.",
            UserWarning,
            stacklevel=2,
        )

    pipeline = joblib.load(model_path)
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "predict_proba"):
        raise ModelIntegrityError(f"{model_path} does not contain a scikit-learn Pipeline.")
    return pipeline, metadata
