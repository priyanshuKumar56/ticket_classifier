"""Re-evaluate the saved model on the exact held-out tickets recorded at training time.

Usage:
    python evaluate.py
    python evaluate.py --model-dir results --data data/tickets.csv

The test tickets are looked up by the ``ticket_id`` values stored in the model
metadata, and the dataset checksum must match the one used for training. This
guards against evaluating on rows the model has seen.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from src.artifacts import ModelArtifactError, load_model
from src.config import DATA_PATH, EVALUATION_FILENAME, results_dir
from src.data_loader import dataset_fingerprint, load_dataset
from src.evaluation import evaluate_pipeline, find_misclassified
from src.validation import ValidationError


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", default=str(results_dir()), help="directory containing model.joblib")
    parser.add_argument("--data", default=str(DATA_PATH), help="labelled ticket CSV used for training")
    parser.add_argument("--show-errors", type=int, default=5, help="how many misclassified tickets to print")
    args = parser.parse_args(argv)

    try:
        pipeline, metadata = load_model(Path(args.model_dir))
        df = load_dataset(args.data)
    except (ModelArtifactError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if dataset_fingerprint(df) != metadata.get("dataset_sha256"):
        print(
            "Error: the dataset changed since this model was trained, so the recorded test split is "
            "no longer trustworthy. Run `python train.py` again.",
            file=sys.stderr,
        )
        return 1

    test_ids = metadata["test_ticket_ids"]
    test_df = df.set_index("ticket_id").loc[test_ids].reset_index()
    result = evaluate_pipeline(pipeline, test_df)
    metrics = result.metrics

    print(f"Evaluated {len(test_df)} held-out tickets (seed {metadata['seed']}).")
    print(f"accuracy        {metrics['accuracy']:.3f}")
    print(f"macro precision {metrics['macro_precision']:.3f}")
    print(f"macro recall    {metrics['macro_recall']:.3f}")
    print(f"macro F1        {metrics['macro_f1']:.3f}\n")
    print(result.report_text)
    print("Confusion matrix (rows = true, columns = predicted; order: " + ", ".join(result.labels) + "):")
    for row in result.confusion:
        print("  " + " ".join(f"{value:3d}" for value in row))

    wrong = find_misclassified(result.predictions)
    print(f"\nMisclassified: {len(wrong)} of {len(test_df)}")
    for _, row in wrong.head(args.show_errors).iterrows():
        print(f"  {row['ticket_id']}  true={row['true_category']}  predicted={row['predicted_category']}  p={row['confidence']:.2f}")
        print(f"    {row['text']}")

    saved_path = Path(args.model_dir) / EVALUATION_FILENAME
    if saved_path.is_file():
        saved = json.loads(saved_path.read_text(encoding="utf-8"))
        same = saved["confusion_matrix"]["matrix"] == result.confusion
        print(f"\nMatches the evaluation saved at training time: {'yes' if same else 'NO - investigate'}")
    print("\nNote: small synthetic educational dataset; these results cannot establish production readiness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
