"""Train the classifier, evaluate it once on the held-out test split and save artifacts.

Usage:
    python train.py                  # fixed, documented hyper-parameters
    python train.py --tune           # choose hyper-parameters by cross-validation on the training split
    python train.py --output-dir out # write artifacts somewhere else
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from src.config import DATA_PATH, RANDOM_SEED, TEST_SIZE, results_dir
from src.training import TrainingSummary, run_training
from src.validation import ValidationError

DISCLAIMER = (
    "Note: this is a small synthetic educational dataset. These numbers cannot establish "
    "production readiness, and predicted probabilities are estimates, not calibrated confidence."
)


def format_summary(summary: TrainingSummary) -> str:
    """Build the human-readable text printed after training."""
    meta, ev = summary.metadata, summary.evaluation
    metrics, baseline, cv = ev["metrics"], ev["baseline"]["metrics"], ev["cross_validation"]
    lines = [
        f"Dataset      : {meta['dataset_file']} ({meta['n_examples']} tickets, "
        f"{meta['n_train']} train / {meta['n_test']} test, seed {meta['seed']})",
        f"Hyper-params : ngram_range={tuple(meta['hyperparameters']['ngram_range'])}, "
        f"C={meta['hyperparameters']['C']} ({'tuned on training split' if meta['tuned'] else 'fixed defaults'})",
        f"Train CV     : macro F1 {cv['mean']:.3f} +/- {cv['std']:.3f} ({cv['folds']}-fold, training split only)",
        "",
        "Held-out test results (evaluated once):",
        f"  accuracy        {metrics['accuracy']:.3f}   (baseline '{ev['baseline']['predicted_category']}': {baseline['accuracy']:.3f})",
        f"  macro precision {metrics['macro_precision']:.3f}",
        f"  macro recall    {metrics['macro_recall']:.3f}",
        f"  macro F1        {metrics['macro_f1']:.3f}   (baseline: {baseline['macro_f1']:.3f})",
        "",
        summary.result.report_text.rstrip(),
        "",
        "Confusion matrix (rows = true, columns = predicted; order: "
        + ", ".join(summary.result.labels)
        + "):",
    ]
    lines += ["  " + " ".join(f"{value:3d}" for value in row) for row in summary.result.confusion]
    wrong = int((~summary.result.predictions["correct"]).sum())
    lines += ["", f"Misclassified test tickets: {wrong} (see misclassified.csv)", "", "Saved files:"]
    lines += [f"  {path}" for path in summary.saved_files]
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(DATA_PATH), help="path to the labelled ticket CSV")
    parser.add_argument("--output-dir", default=str(results_dir()), help="where to write the model and results")
    parser.add_argument("--tune", action="store_true", help="grid-search hyper-parameters (training split only)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed (default: %(default)s)")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE, help="test fraction (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        summary = run_training(args.data, args.output_dir, tune=args.tune, seed=args.seed, test_size=args.test_size)
    except ValidationError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(format_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
