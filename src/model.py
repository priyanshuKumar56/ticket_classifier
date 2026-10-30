"""Pipeline construction, training, cross-validation and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    LOGREG_C,
    LOGREG_MAX_ITER,
    RANDOM_SEED,
    REVIEW_MIN_MARGIN,
    REVIEW_MIN_PROBABILITY,
    TFIDF_NGRAM_RANGE,
    TUNING_GRID,
)
from src.preprocessing import normalize_text
from src.validation import DatasetValidationError, InvalidTicketTextError, validate_ticket_text


@dataclass(frozen=True)
class Prediction:
    """Result of classifying one ticket.

    ``probabilities`` come from ``predict_proba``. They are model estimates,
    not guaranteed calibrated confidence values.
    """

    label: str
    probabilities: dict[str, float]
    confidence: float  # estimated probability of the predicted label
    margin: float  # gap between the top two estimated probabilities
    known_terms: int  # number of distinct vocabulary terms found in the ticket
    needs_review: bool
    review_reasons: tuple[str, ...]


def build_pipeline(
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
    c: float = LOGREG_C,
    seed: int = RANDOM_SEED,
) -> Pipeline:
    """Create an unfitted TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=normalize_text,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                ),
            ),
            ("clf", LogisticRegression(C=c, max_iter=LOGREG_MAX_ITER, random_state=seed)),
        ]
    )


def train_model(train_df: pd.DataFrame, pipeline: Pipeline | None = None) -> Pipeline:
    """Fit the pipeline on the *training* split only.

    Because the vectoriser is a pipeline step, its vocabulary and IDF weights
    are learned exclusively from ``train_df`` - nothing from the test split
    can leak into feature extraction.
    """
    for column in ("text", "category"):
        if column not in train_df.columns:
            raise DatasetValidationError(f"Training data needs a '{column}' column.")
    pipeline = pipeline if pipeline is not None else build_pipeline()
    return pipeline.fit(train_df["text"].tolist(), train_df["category"].tolist())


def _stratified_folds(train_df: pd.DataFrame, folds: int, seed: int) -> StratifiedKFold:
    smallest_class = int(train_df["category"].value_counts().min())
    n_splits = min(folds, smallest_class)
    if n_splits < 2:
        raise DatasetValidationError("Need at least 2 examples per category for cross-validation.")
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def cross_validate_macro_f1(
    train_df: pd.DataFrame,
    folds: int = CV_FOLDS,
    seed: int = RANDOM_SEED,
    estimator: Pipeline | None = None,
) -> dict[str, Any]:
    """Stratified cross-validated macro F1 on the training split only.

    ``estimator`` supplies the hyper-parameters to evaluate (default: the
    standard pipeline). scikit-learn clones it for every fold, so the
    vectoriser is re-fitted on each fold's training portion (no leakage
    between folds) and a pipeline that is already fitted is never reused.
    """
    splitter = _stratified_folds(train_df, folds, seed)
    scores = cross_val_score(
        estimator if estimator is not None else build_pipeline(seed=seed),
        train_df["text"].tolist(),
        train_df["category"].tolist(),
        cv=splitter,
        scoring="f1_macro",
    )
    return {
        "folds": int(splitter.get_n_splits()),
        "scores": [float(s) for s in scores],
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
    }


def tune_model(
    train_df: pd.DataFrame, folds: int = CV_FOLDS, seed: int = RANDOM_SEED
) -> tuple[Pipeline, dict[str, Any]]:
    """Grid-search hyper-parameters with cross-validation on the training split.

    The test split is never touched. Returns the refitted best pipeline and a
    JSON-friendly summary of the search.
    """
    search = GridSearchCV(
        build_pipeline(seed=seed),
        param_grid=TUNING_GRID,
        scoring="f1_macro",
        cv=_stratified_folds(train_df, folds, seed),
        refit=True,
    )
    search.fit(train_df["text"].tolist(), train_df["category"].tolist())
    best = {k: (list(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()}
    summary = {
        "best_params": best,
        "best_cv_macro_f1": float(search.best_score_),
        "candidates_evaluated": int(len(search.cv_results_["params"])),
    }
    return search.best_estimator_, summary


def _review_reasons(confidence: float, margin: float, known_terms: int) -> tuple[str, ...]:
    reasons: list[str] = []
    if known_terms == 0:
        reasons.append("none of the words in this ticket were seen during training")
    if confidence < REVIEW_MIN_PROBABILITY:
        reasons.append(f"top probability estimate is below {REVIEW_MIN_PROBABILITY:.0%}")
    if margin < REVIEW_MIN_MARGIN:
        reasons.append(f"top two categories are within {REVIEW_MIN_MARGIN:.0%} of each other")
    return tuple(reasons)


def predict_texts(pipeline: Pipeline, texts: Sequence[str]) -> list[Prediction]:
    """Classify texts that have *already* been validated (see ``validate_ticket_text``)."""
    if len(texts) == 0:
        return []
    vectorizer, classifier = pipeline.named_steps["tfidf"], pipeline.named_steps["clf"]
    features = vectorizer.transform(list(texts))
    probabilities = classifier.predict_proba(features)
    known_terms = features.getnnz(axis=1)
    classes = [str(c) for c in classifier.classes_]

    predictions: list[Prediction] = []
    for row, known in zip(probabilities, known_terms):
        order = np.argsort(row)[::-1]
        confidence = float(row[order[0]])
        margin = confidence - float(row[order[1]])
        reasons = _review_reasons(confidence, margin, int(known))
        predictions.append(
            Prediction(
                label=classes[int(order[0])],
                probabilities={cls: float(p) for cls, p in zip(classes, row)},
                confidence=confidence,
                margin=margin,
                known_terms=int(known),
                needs_review=bool(reasons),
                review_reasons=reasons,
            )
        )
    return predictions


def predict_ticket(pipeline: Pipeline, text: object) -> Prediction:
    """Validate and classify a single ticket.

    Raises:
        InvalidTicketTextError: if the text fails validation.
    """
    return predict_texts(pipeline, [validate_ticket_text(text)])[0]


def predict_dataframe(pipeline: Pipeline, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
    """Classify every row of ``df``; invalid rows get an ``error`` message instead of crashing.

    Returns a copy of ``df`` with ``predicted_category``, ``confidence``,
    ``needs_review``, ``review_reason``, one ``prob_<label>`` column per class
    and ``error`` columns appended.
    """
    if text_column not in df.columns:
        raise DatasetValidationError(f"Input needs a '{text_column}' column. Found: {list(df.columns)}.")

    valid_positions: list[int] = []
    valid_texts: list[str] = []
    errors: dict[int, str] = {}
    for position, raw in enumerate(df[text_column].tolist()):
        try:
            valid_texts.append(validate_ticket_text(raw))
            valid_positions.append(position)
        except InvalidTicketTextError as exc:
            errors[position] = str(exc)

    by_position = dict(zip(valid_positions, predict_texts(pipeline, valid_texts)))
    classes = [str(c) for c in pipeline.named_steps["clf"].classes_]
    records: list[dict[str, Any]] = []
    for position in range(len(df)):
        prediction = by_position.get(position)
        record: dict[str, Any] = {
            "predicted_category": prediction.label if prediction else None,
            "confidence": round(prediction.confidence, 4) if prediction else None,
            "needs_review": prediction.needs_review if prediction else True,
            "review_reason": "; ".join(prediction.review_reasons) if prediction else None,
            "error": errors.get(position),
        }
        for cls in classes:
            record[f"prob_{cls}"] = round(prediction.probabilities[cls], 4) if prediction else None
        records.append(record)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(records)], axis=1)
