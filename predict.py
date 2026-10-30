"""Classify one ticket or a CSV of tickets with the trained model.

Usage:
    python predict.py --text "I forgot my password and the reset email never arrives"
    python predict.py --text "..." --explain          # show the terms that drove the prediction
    python predict.py --text "..." --json             # machine-readable output
    python predict.py --file data/demo_batch.csv --output results/batch_predictions.csv

Exit codes: 0 success, 1 model problem, 2 invalid input.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.artifacts import ModelArtifactError, load_model
from src.config import MAX_BATCH_ROWS, results_dir
from src.explain import explain_prediction
from src.model import Prediction, predict_dataframe, predict_ticket
from src.validation import ValidationError

PROBABILITY_NOTE = "Probabilities are model estimates, not guaranteed calibrated confidence."


def format_prediction(prediction: Prediction) -> str:
    """Render a prediction as plain ASCII text (safe for any terminal)."""
    lines = [
        f"Predicted category : {prediction.label}",
        f"Top probability    : {prediction.confidence:.3f} (estimate)",
        f"Needs human review : {'yes - ' + '; '.join(prediction.review_reasons) if prediction.needs_review else 'no'}",
        "Probabilities:",
    ]
    for label, value in sorted(prediction.probabilities.items(), key=lambda item: -item[1]):
        lines.append(f"  {label:<12} {value:.3f} {'#' * round(value * 30)}")
    lines.append(PROBABILITY_NOTE)
    return "\n".join(lines)


def _predict_single(pipeline, text: str, explain: bool, as_json: bool) -> int:
    try:
        prediction = predict_ticket(pipeline, text)
        terms = explain_prediction(pipeline, text, prediction.label) if explain else []
    except ValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    if as_json:
        payload = asdict(prediction)
        if explain:
            payload["top_terms"] = [asdict(term) for term in terms]
        print(json.dumps(payload, indent=2))
        return 0
    print(format_prediction(prediction))
    if explain:
        print("Top terms for this category (+ supports, - opposes):")
        for term in terms or []:
            print(f"  {term.contribution:+.3f}  {term.term}")
        if not terms:
            print("  (no known terms)")
    return 0


def _predict_file(pipeline, file: str, output: str | None) -> int:
    path = Path(file)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 2
    try:
        frame = pd.read_csv(path, dtype=str)
        if len(frame) > MAX_BATCH_ROWS:
            print(f"Error: {len(frame)} rows exceeds the limit of {MAX_BATCH_ROWS}.", file=sys.stderr)
            return 2
        result = predict_dataframe(pipeline, frame)
    except (ValidationError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    shown = [c for c in ("ticket_id", "predicted_category", "confidence", "needs_review", "error") if c in result.columns]
    print(result[shown].to_string(index=False))
    print(
        f"\n{len(result)} tickets, {int(result['needs_review'].sum())} flagged for human review, "
        f"{int(result['error'].notna().sum())} rejected as invalid."
    )
    print(PROBABILITY_NOTE)
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)
        print(f"Saved: {output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="a single ticket to classify")
    source.add_argument("--file", help="CSV with a 'text' column (and optionally 'ticket_id')")
    parser.add_argument("--output", help="write batch predictions to this CSV (with --file)")
    parser.add_argument("--model-dir", default=str(results_dir()), help="directory containing model.joblib")
    parser.add_argument("--explain", action="store_true", help="show top contributing terms (with --text)")
    parser.add_argument("--json", action="store_true", help="print JSON (with --text)")
    args = parser.parse_args(argv)

    try:
        pipeline, _ = load_model(Path(args.model_dir))
    except ModelArtifactError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.text is not None:
        return _predict_single(pipeline, args.text, args.explain, args.json)
    return _predict_file(pipeline, args.file, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
