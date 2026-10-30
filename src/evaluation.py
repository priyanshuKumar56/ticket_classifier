"""Metrics, confusion matrix, baseline comparison and misclassification review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import (
    CONFUSION_MATRIX_FILENAME,
    EVALUATION_FILENAME,
    LABELS,
    MISCLASSIFIED_FILENAME,
    TEST_PREDICTIONS_FILENAME,
)
from src.model import predict_texts


@dataclass
class EvaluationResult:
    """Everything produced by evaluating a pipeline on a held-out test set."""

    metrics: dict[str, Any]
    report: dict[str, Any]
    report_text: str
    confusion: list[list[int]]
    labels: list[str]
    predictions: pd.DataFrame


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = LABELS
) -> dict[str, Any]:
    """Compute accuracy, macro precision/recall/F1 and per-class scores.

    Macro averages weight every category equally, regardless of how many
    test tickets it has.
    """
    labels = list(labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
    }


def evaluate_pipeline(
    pipeline: Pipeline, test_df: pd.DataFrame, labels: Sequence[str] = LABELS
) -> EvaluationResult:
    """Evaluate a fitted pipeline on ``test_df`` (columns: ticket_id, text, category)."""
    labels = list(labels)
    predictions = predict_texts(pipeline, test_df["text"].tolist())
    y_true = test_df["category"].tolist()
    y_pred = [p.label for p in predictions]

    table = pd.DataFrame(
        {
            "ticket_id": test_df["ticket_id"].tolist(),
            "text": test_df["text"].tolist(),
            "true_category": y_true,
            "predicted_category": y_pred,
            "confidence": [round(p.confidence, 4) for p in predictions],
            "margin": [round(p.margin, 4) for p in predictions],
            "needs_review": [p.needs_review for p in predictions],
            "correct": [t == p for t, p in zip(y_true, y_pred)],
        }
    )
    for label in labels:
        table[f"prob_{label}"] = [round(p.probabilities.get(label, 0.0), 4) for p in predictions]

    return EvaluationResult(
        metrics=compute_metrics(y_true, y_pred, labels),
        report=classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        report_text=classification_report(y_true, y_pred, labels=labels, zero_division=0),
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        labels=labels,
        predictions=table,
    )


def evaluate_baseline(
    train_df: pd.DataFrame, test_df: pd.DataFrame, labels: Sequence[str] = LABELS
) -> dict[str, Any]:
    """Score a 'always predict the most common category' baseline for context."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(train_df["text"].tolist(), train_df["category"].tolist())
    y_pred = dummy.predict(test_df["text"].tolist())
    return {
        "strategy": "most_frequent",
        "predicted_category": str(y_pred[0]) if len(y_pred) else None,
        "metrics": compute_metrics(test_df["category"].tolist(), list(y_pred), labels),
    }


def find_misclassified(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return wrongly classified rows, most confidently wrong first."""
    wrong = predictions.loc[~predictions["correct"]]
    return wrong.sort_values("confidence", ascending=False).reset_index(drop=True)


def triage_summary(predictions: pd.DataFrame) -> dict[str, Any]:
    """Summarise how the human-review flag relates to correctness.

    Groups can be empty (for example when nothing is flagged), in which case
    the matching accuracy is ``None``.
    """

    def accuracy_of(mask: pd.Series) -> float | None:
        return float(predictions.loc[mask, "correct"].mean()) if mask.any() else None

    def mean_confidence(mask: pd.Series) -> float | None:
        return float(predictions.loc[mask, "confidence"].mean()) if mask.any() else None

    flagged = predictions["needs_review"].astype(bool)
    correct = predictions["correct"].astype(bool)
    total = len(predictions)
    return {
        "n_flagged": int(flagged.sum()),
        "flag_rate": float(flagged.mean()) if total else None,
        "accuracy_flagged": accuracy_of(flagged),
        "accuracy_not_flagged": accuracy_of(~flagged),
        "mean_confidence_correct": mean_confidence(correct),
        "mean_confidence_incorrect": mean_confidence(~correct),
    }


def plot_confusion_matrix(
    confusion: Sequence[Sequence[int]], labels: Sequence[str] = LABELS
) -> Figure:
    """Draw a confusion matrix (rows = true category, columns = predicted)."""
    figure = Figure(figsize=(5.2, 4.4), layout="constrained")
    axes = figure.subplots()
    display = ConfusionMatrixDisplay(np.asarray(confusion), display_labels=list(labels))
    display.plot(ax=axes, cmap="Blues", colorbar=False, values_format="d")
    axes.set_xlabel("Predicted category")
    axes.set_ylabel("True category")
    axes.tick_params(axis="x", rotation=30)
    return figure


def save_evaluation_artifacts(
    result: EvaluationResult, payload: dict[str, Any], directory: Path
) -> list[Path]:
    """Write evaluation JSON, prediction CSVs and the confusion-matrix PNG."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    evaluation_path = directory / EVALUATION_FILENAME
    document = {
        **payload,
        "metrics": result.metrics,
        "classification_report": result.report,
        "confusion_matrix": {"labels": result.labels, "matrix": result.confusion},
    }
    evaluation_path.write_text(json.dumps(document, indent=2), encoding="utf-8")

    predictions_path = directory / TEST_PREDICTIONS_FILENAME
    result.predictions.to_csv(predictions_path, index=False)
    misclassified_path = directory / MISCLASSIFIED_FILENAME
    find_misclassified(result.predictions).to_csv(misclassified_path, index=False)

    image_path = directory / CONFUSION_MATRIX_FILENAME
    plot_confusion_matrix(result.confusion, result.labels).savefig(image_path, dpi=150)
    return [evaluation_path, predictions_path, misclassified_path, image_path]
