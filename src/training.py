"""End-to-end training run: load, split, fit, cross-validate, evaluate, save.

Used by ``train.py`` and by the Streamlit app when no model exists yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.artifacts import save_model
from src.config import (
    CV_FOLDS,
    DATA_PATH,
    LOGREG_C,
    RANDOM_SEED,
    TEST_SIZE,
    TFIDF_NGRAM_RANGE,
    results_dir,
)
from src.data_loader import dataset_fingerprint, load_dataset, split_dataset
from src.evaluation import (
    EvaluationResult,
    evaluate_baseline,
    evaluate_pipeline,
    save_evaluation_artifacts,
    triage_summary,
)
from src.model import build_pipeline, cross_validate_macro_f1, train_model, tune_model


@dataclass
class TrainingSummary:
    """What a training run produced."""

    output_dir: Path
    metadata: dict[str, Any]
    evaluation: dict[str, Any]
    result: EvaluationResult
    saved_files: list[Path]


def run_training(
    data_path: str | Path = DATA_PATH,
    output_dir: str | Path | None = None,
    tune: bool = False,
    seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
) -> TrainingSummary:
    """Train on the training split, evaluate once on the test split, save artifacts.

    The test split is used for a single final evaluation. With ``tune=True``
    hyper-parameters are chosen by cross-validation on the training split only.
    """
    output_dir = Path(output_dir) if output_dir is not None else results_dir()
    data_path = Path(data_path)

    df = load_dataset(data_path)
    train_df, test_df = split_dataset(df, test_size=test_size, seed=seed)

    if tune:
        pipeline, tuning = tune_model(train_df, seed=seed)
    else:
        pipeline, tuning = train_model(train_df, build_pipeline(seed=seed)), None

    cross_validation = cross_validate_macro_f1(train_df, folds=CV_FOLDS, seed=seed, estimator=pipeline)
    result = evaluate_pipeline(pipeline, test_df)
    baseline = evaluate_baseline(train_df, test_df)

    metadata: dict[str, Any] = {
        "dataset_file": data_path.name,
        "dataset_sha256": dataset_fingerprint(df),
        "n_examples": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "seed": seed,
        "test_size": test_size,
        "labels": result.labels,
        "test_ticket_ids": test_df["ticket_id"].tolist(),
        "tuned": bool(tune),
        "hyperparameters": {
            "ngram_range": list(pipeline.named_steps["tfidf"].ngram_range),
            "C": float(pipeline.named_steps["clf"].C),
            "defaults": {"ngram_range": list(TFIDF_NGRAM_RANGE), "C": LOGREG_C},
        },
    }
    payload: dict[str, Any] = {
        "n_test": int(len(test_df)),
        "baseline": baseline,
        "cross_validation": cross_validation,
        "tuning": tuning,
        "triage": triage_summary(result.predictions),
    }

    saved = [save_model(pipeline, metadata, output_dir)]
    saved += save_evaluation_artifacts(result, payload, output_dir)
    evaluation = {**payload, "metrics": result.metrics}
    return TrainingSummary(output_dir, metadata, evaluation, result, saved)
