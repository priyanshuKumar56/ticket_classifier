"""Tests for metrics, confusion matrix, baseline, triage summary and saved artifacts."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.config import (
    CONFUSION_MATRIX_FILENAME,
    EVALUATION_FILENAME,
    LABELS,
    MISCLASSIFIED_FILENAME,
    TEST_PREDICTIONS_FILENAME,
)
from src.data_loader import split_dataset
from src.evaluation import (
    compute_metrics,
    evaluate_baseline,
    evaluate_pipeline,
    find_misclassified,
    plot_confusion_matrix,
    save_evaluation_artifacts,
    triage_summary,
)

# Imbalanced fixture where macro F1 (0.5641) and weighted F1 (0.6838) clearly differ.
Y_TRUE = ["appointment"] * 6 + ["billing"] * 2 + ["login"] * 2 + ["technical"] * 2
Y_PRED = ["appointment"] * 6 + ["billing", "appointment"] + ["login"] * 2 + ["login"] * 2
EXPECTED_MACRO_F1 = 0.5641


def test_metrics_have_the_expected_structure():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert {"accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class"} <= set(metrics)
    assert set(metrics["per_class"]) == set(LABELS)
    for scores in metrics["per_class"].values():
        assert set(scores) == {"precision", "recall", "f1", "support"}
        assert all(0.0 <= scores[key] <= 1.0 for key in ("precision", "recall", "f1"))
    assert metrics["per_class"]["appointment"]["support"] == 6
    assert sum(item["support"] for item in metrics["per_class"].values()) == len(Y_TRUE)


def test_accuracy_and_per_class_values_are_correct():
    metrics = compute_metrics(Y_TRUE, Y_PRED)
    assert metrics["accuracy"] == pytest.approx(9 / 12)  # 6 + 1 + 2 + 0 correct
    assert metrics["per_class"]["technical"]["recall"] == 0.0
    assert metrics["per_class"]["login"]["precision"] == pytest.approx(0.5)


def test_absent_class_does_not_crash_metrics():
    metrics = compute_metrics(["login", "login"], ["login", "billing"])
    assert metrics["per_class"]["technical"]["support"] == 0


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Known defect B (see README): macro_f1 is calculated with the weighted average.",
)
def test_macro_f1_uses_the_macro_average():
    reference = f1_score(Y_TRUE, Y_PRED, labels=list(LABELS), average="macro", zero_division=0)
    assert reference == pytest.approx(EXPECTED_MACRO_F1, abs=1e-4)
    assert compute_metrics(Y_TRUE, Y_PRED)["macro_f1"] == pytest.approx(reference)


def test_evaluate_pipeline_output_structure(tiny_pipeline, tiny_df):
    result = evaluate_pipeline(tiny_pipeline, tiny_df)
    assert result.labels == list(LABELS)
    assert len(result.confusion) == len(LABELS)
    assert all(len(row) == len(LABELS) for row in result.confusion)
    assert sum(sum(row) for row in result.confusion) == len(tiny_df)
    for i, label in enumerate(LABELS):
        assert sum(result.confusion[i]) == result.metrics["per_class"][label]["support"]
    assert {"macro avg", "weighted avg", "accuracy", *LABELS} <= set(result.report)
    assert "precision" in result.report_text
    expected_columns = {"ticket_id", "text", "true_category", "predicted_category", "confidence", "needs_review", "correct"}
    assert expected_columns <= set(result.predictions.columns)
    assert len(result.predictions) == len(tiny_df)


def test_correct_column_matches_predictions(tiny_pipeline, tiny_df):
    table = evaluate_pipeline(tiny_pipeline, tiny_df).predictions
    assert (table["correct"] == (table["true_category"] == table["predicted_category"])).all()


def test_misclassified_rows_are_wrong_and_sorted_most_confident_first():
    table = pd.DataFrame(
        {
            "ticket_id": ["a", "b", "c", "d"],
            "true_category": ["login", "login", "billing", "billing"],
            "predicted_category": ["login", "billing", "login", "billing"],
            "confidence": [0.9, 0.4, 0.8, 0.7],
            "correct": [True, False, False, True],
        }
    )
    wrong = find_misclassified(table)
    assert wrong["ticket_id"].tolist() == ["c", "b"]
    assert not wrong["correct"].any()


def test_baseline_always_predicts_the_most_common_class(tiny_df):
    train_df, test_df = split_dataset(tiny_df, test_size=0.25, seed=1)
    baseline = evaluate_baseline(train_df, test_df)
    assert baseline["strategy"] == "most_frequent"
    assert baseline["predicted_category"] in LABELS
    assert baseline["metrics"]["per_class"][baseline["predicted_category"]]["recall"] == 1.0


def test_triage_summary_handles_empty_groups():
    table = pd.DataFrame(
        {"needs_review": [False, False], "correct": [True, False], "confidence": [0.9, 0.6]}
    )
    summary = triage_summary(table)
    assert summary["n_flagged"] == 0
    assert summary["accuracy_flagged"] is None
    assert summary["accuracy_not_flagged"] == pytest.approx(0.5)
    assert summary["mean_confidence_correct"] == pytest.approx(0.9)


def test_confusion_matrix_figure_can_be_drawn(tiny_pipeline, tiny_df):
    result = evaluate_pipeline(tiny_pipeline, tiny_df)
    figure = plot_confusion_matrix(result.confusion, result.labels)
    assert len(figure.axes) == 1


def test_evaluation_artifacts_are_written(tiny_pipeline, tiny_df, tmp_path):
    result = evaluate_pipeline(tiny_pipeline, tiny_df)
    paths = save_evaluation_artifacts(result, {"n_test": len(tiny_df)}, tmp_path)
    assert {p.name for p in paths} == {
        EVALUATION_FILENAME,
        TEST_PREDICTIONS_FILENAME,
        MISCLASSIFIED_FILENAME,
        CONFUSION_MATRIX_FILENAME,
    }
    document = json.loads((tmp_path / EVALUATION_FILENAME).read_text(encoding="utf-8"))
    assert document["n_test"] == len(tiny_df)
    assert document["confusion_matrix"]["labels"] == list(LABELS)
    assert (tmp_path / CONFUSION_MATRIX_FILENAME).stat().st_size > 0


def test_training_run_writes_a_complete_results_folder(trained_results_dir):
    names = {p.name for p in trained_results_dir.iterdir()}
    assert {"model.joblib", "model_metadata.json", EVALUATION_FILENAME, MISCLASSIFIED_FILENAME} <= names
    metadata = json.loads((trained_results_dir / "model_metadata.json").read_text(encoding="utf-8"))
    assert metadata["n_train"] + metadata["n_test"] == metadata["n_examples"]
    assert len(metadata["test_ticket_ids"]) == metadata["n_test"]
