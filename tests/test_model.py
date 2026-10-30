"""Tests for training, prediction, explanation, tuning and model persistence."""

from __future__ import annotations

import json
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.artifacts import (
    ModelIntegrityError,
    ModelNotFoundError,
    load_model,
    save_model,
    sha256_file,
)
from src.config import LABELS, METADATA_FILENAME, MODEL_FILENAME, RANDOM_SEED
from src.data_loader import split_dataset
from src.explain import explain_prediction
from src.model import (
    build_pipeline,
    cross_validate_macro_f1,
    predict_dataframe,
    predict_ticket,
    train_model,
    tune_model,
)
from src.preprocessing import normalize_text
from src.validation import DatasetValidationError, InvalidTicketTextError


# --------------------------------------------------------------------- training
def test_pipeline_has_tfidf_and_logistic_regression_steps():
    pipeline = build_pipeline()
    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(pipeline.named_steps["clf"], LogisticRegression)


def test_training_succeeds_and_learns_all_four_labels(tiny_pipeline):
    assert list(tiny_pipeline.named_steps["clf"].classes_) == list(LABELS)
    assert len(tiny_pipeline.named_steps["tfidf"].vocabulary_) > 0


def test_training_requires_text_and_category_columns():
    with pytest.raises(DatasetValidationError, match="category"):
        train_model(pd.DataFrame({"text": ["a"]}))


def test_vectorizer_is_fitted_on_the_training_split_only(tiny_df):
    train_df = tiny_df.groupby("category").head(6).copy()
    probe = "zzqxtestonlyterm"
    assert probe not in " ".join(train_df["text"])
    pipeline = train_model(train_df, build_pipeline())
    assert probe not in pipeline.named_steps["tfidf"].vocabulary_

    leaky = pd.concat([train_df, pd.DataFrame([{"ticket_id": "L", "text": probe + " login", "category": "login"}])])
    assert probe in train_model(leaky, build_pipeline()).named_steps["tfidf"].vocabulary_


def test_training_is_reproducible(tiny_df):
    first = train_model(tiny_df, build_pipeline())
    second = train_model(tiny_df, build_pipeline())
    texts = ["reset my password", "invoice is wrong", "app crashes", "move my booking"]
    np.testing.assert_allclose(first.predict_proba(texts), second.predict_proba(texts))


def test_model_beats_the_majority_class_baseline_on_real_data(full_df):
    train_df, test_df = split_dataset(full_df)
    pipeline = train_model(train_df, build_pipeline())
    accuracy = float(np.mean(pipeline.predict(test_df["text"].tolist()) == test_df["category"].to_numpy()))
    majority = float(test_df["category"].value_counts(normalize=True).max())
    assert accuracy > majority + 0.1


def test_cross_validation_uses_the_training_split_and_reports_scores(tiny_df):
    summary = cross_validate_macro_f1(tiny_df, folds=4)
    assert summary["folds"] == 4
    assert len(summary["scores"]) == 4
    assert all(0.0 <= score <= 1.0 for score in summary["scores"])


def test_cross_validation_uses_the_supplied_estimator(tiny_df):
    weak = build_pipeline(c=0.001)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    expected = cross_val_score(weak, tiny_df["text"].tolist(), tiny_df["category"].tolist(), cv=splitter, scoring="f1_macro")
    summary = cross_validate_macro_f1(tiny_df, folds=4, estimator=weak)
    assert summary["scores"] == pytest.approx(list(expected))


def test_tuning_returns_a_fitted_pipeline_and_summary(tiny_df):
    pipeline, summary = tune_model(tiny_df, folds=3)
    assert set(summary["best_params"]) == {"tfidf__ngram_range", "clf__C"}
    assert summary["candidates_evaluated"] == 8
    assert pipeline.predict(["reset my password"]).shape == (1,)


# ------------------------------------------------------------------- prediction
def test_prediction_label_is_valid(tiny_pipeline):
    prediction = predict_ticket(tiny_pipeline, "I forgot my password and cannot sign in")
    assert prediction.label in LABELS


def test_probabilities_are_in_range_and_sum_to_one(tiny_pipeline):
    prediction = predict_ticket(tiny_pipeline, "The invoice total is wrong")
    values = list(prediction.probabilities.values())
    assert set(prediction.probabilities) == set(LABELS)
    assert all(0.0 <= value <= 1.0 for value in values)
    assert sum(values) == pytest.approx(1.0)
    assert prediction.confidence == pytest.approx(max(values))
    assert prediction.probabilities[prediction.label] == pytest.approx(prediction.confidence)


def test_clear_ticket_is_classified_sensibly(tiny_pipeline):
    assert predict_ticket(tiny_pipeline, "password reset email never arrives").label == "login"


def test_unknown_vocabulary_is_handled_and_flagged_for_review(tiny_pipeline):
    prediction = predict_ticket(tiny_pipeline, "zxqv blorf wibble")
    assert prediction.label in LABELS
    assert prediction.known_terms == 0
    assert prediction.needs_review
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_mixed_known_and_unknown_words_still_work(tiny_pipeline):
    prediction = predict_ticket(tiny_pipeline, "zxqv blorf my password reset")
    assert prediction.known_terms > 0
    assert prediction.label in LABELS


@pytest.mark.parametrize("value", ["", None])
def test_invalid_text_is_rejected_by_the_prediction_layer(tiny_pipeline, value):
    with pytest.raises(InvalidTicketTextError):
        predict_ticket(tiny_pipeline, value)


def test_batch_prediction_reports_bad_rows_without_crashing(tiny_pipeline):
    frame = pd.DataFrame({"ticket_id": ["a", "b", "c"], "text": ["reset my password", "", None]})
    result = predict_dataframe(tiny_pipeline, frame)
    assert list(result["ticket_id"]) == ["a", "b", "c"]
    assert result.loc[0, "predicted_category"] in LABELS
    assert result.loc[1:, "error"].notna().all()
    assert result.loc[1:, "needs_review"].all()
    assert {f"prob_{label}" for label in LABELS} <= set(result.columns)


def test_batch_prediction_requires_a_text_column(tiny_pipeline):
    with pytest.raises(DatasetValidationError, match="text"):
        predict_dataframe(tiny_pipeline, pd.DataFrame({"body": ["hello"]}))


# ----------------------------------------------------------------- explanation
def test_explanation_lists_known_terms_that_support_the_prediction(tiny_pipeline):
    terms = explain_prediction(tiny_pipeline, "password reset email never arrives", "login", top_k=5)
    assert 0 < len(terms) <= 5
    assert terms[0].contribution > 0
    assert all(abs(a.contribution) >= abs(b.contribution) for a, b in zip(terms, terms[1:]))


def test_explanation_is_empty_for_unknown_words(tiny_pipeline):
    assert explain_prediction(tiny_pipeline, "zxqv blorf wibble") == []


def test_explanation_rejects_unknown_category(tiny_pipeline):
    with pytest.raises(ValueError, match="Unknown category"):
        explain_prediction(tiny_pipeline, "reset my password", "refunds")


# ---------------------------------------------------------------- preprocessing
def test_normalize_text_masks_numbers_urls_and_emails():
    cleaned = normalize_text("Error 500 at https://x.example/r, mail me@clinic.example!  ")
    assert cleaned == "error numtoken at urltoken mail emailtoken !"


# ------------------------------------------------------------------ persistence
def test_save_and_load_round_trip_gives_identical_probabilities(tiny_pipeline, tmp_path):
    save_model(tiny_pipeline, {"seed": 42}, tmp_path)
    loaded, metadata = load_model(tmp_path)
    texts = ["reset my password", "invoice is wrong"]
    np.testing.assert_allclose(loaded.predict_proba(texts), tiny_pipeline.predict_proba(texts))
    assert metadata["seed"] == 42
    assert metadata["model_sha256"] == sha256_file(tmp_path / MODEL_FILENAME)


def test_loading_without_a_model_gives_a_helpful_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="python train.py"):
        load_model(tmp_path)


def test_tampered_model_file_is_detected(tiny_pipeline, tmp_path):
    save_model(tiny_pipeline, {}, tmp_path)
    with (tmp_path / MODEL_FILENAME).open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ModelIntegrityError, match="checksum"):
        load_model(tmp_path)


def test_file_that_is_not_a_pipeline_is_rejected(tmp_path):
    model_path = tmp_path / MODEL_FILENAME
    joblib.dump({"not": "a pipeline"}, model_path)
    (tmp_path / METADATA_FILENAME).write_text(
        json.dumps({"model_sha256": sha256_file(model_path), "sklearn_version": "x"}), encoding="utf-8"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ModelIntegrityError, match="Pipeline"):
            load_model(tmp_path)


def test_corrupt_metadata_is_reported(tiny_pipeline, tmp_path):
    save_model(tiny_pipeline, {}, tmp_path)
    (tmp_path / METADATA_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(ModelIntegrityError, match="JSON"):
        load_model(tmp_path)


def test_version_mismatch_warns_but_loads(tiny_pipeline, tmp_path):
    save_model(tiny_pipeline, {}, tmp_path)
    meta_path = tmp_path / METADATA_FILENAME
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata["sklearn_version"] = "0.0.1"
    meta_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.warns(UserWarning, match="0.0.1"):
        pipeline, _ = load_model(tmp_path)
    assert isinstance(pipeline, Pipeline)
