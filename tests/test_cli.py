"""Tests for the command-line entry points (train.py, predict.py, evaluate.py)."""

from __future__ import annotations

import json

import pandas as pd

import evaluate
import predict
import train
from src.config import DATA_PATH, DEMO_BATCH_PATH, LABELS


def test_train_cli_writes_artifacts_and_prints_a_report(tmp_path, capsys):
    exit_code = train.main(["--output-dir", str(tmp_path)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert (tmp_path / "model.joblib").is_file()
    assert "macro F1" in output
    assert "Confusion matrix" in output
    assert "cannot establish production readiness" in output


def test_train_cli_reports_a_missing_dataset(tmp_path, capsys):
    exit_code = train.main(["--data", str(tmp_path / "missing.csv"), "--output-dir", str(tmp_path)])
    assert exit_code == 2
    assert "not found" in capsys.readouterr().err


def test_predict_cli_single_ticket(trained_results_dir, capsys):
    exit_code = predict.main(["--text", "I forgot my password", "--model-dir", str(trained_results_dir)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Predicted category" in output
    assert "not guaranteed calibrated confidence" in output


def test_predict_cli_json_output_is_valid(trained_results_dir, capsys):
    exit_code = predict.main(["--text", "The invoice is wrong", "--json", "--explain", "--model-dir", str(trained_results_dir)])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["label"] in LABELS
    assert abs(sum(payload["probabilities"].values()) - 1.0) < 1e-9
    assert "top_terms" in payload


def test_predict_cli_rejects_empty_text(trained_results_dir, capsys):
    exit_code = predict.main(["--text", "", "--model-dir", str(trained_results_dir)])
    assert exit_code == 2
    assert "empty" in capsys.readouterr().err


def test_predict_cli_without_a_model_exits_with_code_1(tmp_path, capsys):
    exit_code = predict.main(["--text", "hello", "--model-dir", str(tmp_path)])
    assert exit_code == 1
    assert "python train.py" in capsys.readouterr().err


def test_predict_cli_batch_writes_output_csv(trained_results_dir, tmp_path, capsys):
    destination = tmp_path / "out" / "batch.csv"
    exit_code = predict.main(
        ["--file", str(DEMO_BATCH_PATH), "--output", str(destination), "--model-dir", str(trained_results_dir)]
    )
    assert exit_code == 0
    result = pd.read_csv(destination)
    assert len(result) == len(pd.read_csv(DEMO_BATCH_PATH))
    assert set(result["predicted_category"]) <= set(LABELS)
    assert "flagged for human review" in capsys.readouterr().out


def test_predict_cli_batch_missing_text_column(trained_results_dir, tmp_path, capsys):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"body": ["hello"]}).to_csv(path, index=False)
    exit_code = predict.main(["--file", str(path), "--model-dir", str(trained_results_dir)])
    assert exit_code == 2
    assert "text" in capsys.readouterr().err


def test_evaluate_cli_matches_the_saved_evaluation(trained_results_dir, capsys):
    exit_code = evaluate.main(["--model-dir", str(trained_results_dir)])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Matches the evaluation saved at training time: yes" in output


def test_evaluate_cli_refuses_a_changed_dataset(trained_results_dir, tmp_path, capsys):
    frame = pd.read_csv(DATA_PATH, dtype=str)
    frame.loc[0, "text"] = frame.loc[0, "text"] + " (edited)"
    edited = tmp_path / "edited.csv"
    frame.to_csv(edited, index=False)
    exit_code = evaluate.main(["--model-dir", str(trained_results_dir), "--data", str(edited)])
    assert exit_code == 1
    assert "dataset changed" in capsys.readouterr().err
