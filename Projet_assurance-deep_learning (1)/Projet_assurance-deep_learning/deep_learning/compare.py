"""Compare all training runs stored in the results/ directory.

Usage:
  python compare.py                        # all runs
  python compare.py --model tabm           # filter by model
  python compare.py --dataset train        # filter by dataset
  python compare.py --sort val_qwk_offset  # sort by a different metric
"""
from __future__ import annotations
import argparse
import csv
import json
import os


RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

SORT_KEYS = [
    "val_qwk_raw", "val_qwk_offset",
    "val_acc_raw", "val_acc_offset",
]


def load_runs(model_filter: str | None, dataset_filter: str | None) -> list[dict]:
    runs = []
    if not os.path.isdir(RESULTS_DIR):
        return runs
    for name in sorted(os.listdir(RESULTS_DIR)):
        path = os.path.join(RESULTS_DIR, name, "hyperparameters.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data["_run"] = name
        if model_filter and data.get("model") != model_filter:
            continue
        if dataset_filter and data.get("dataset") != dataset_filter:
            continue
        runs.append(data)
    return runs


COLS = [
    ("Run", "_run", 40),
    ("Dataset", "dataset", 16),
    ("Model", "model", 8),
    ("Ep", "epochs_trained", 4),
    ("Val QWK raw", "val_qwk_raw", 12),
    ("Val QWK+off", "val_qwk_offset", 12),
    ("Val Acc+off", "val_acc_offset", 12),
]


def print_table(runs: list[dict], sort_key: str) -> None:
    header = "  ".join(label.ljust(width) for label, _, width in COLS)
    sep = "  ".join("-" * width for _, _, width in COLS)
    print(header)
    print(sep)
    for r in runs:
        row_parts = []
        for label, key, width in COLS:
            val = r.get(key, "-")
            cell = f"{val:.4f}" if isinstance(val, float) else str(val)
            row_parts.append(cell.ljust(width)[:width])
        print("  ".join(row_parts))
    print(f"\n{len(runs)} run(s) - sorted by {sort_key} desc")


def export_csv(runs: list[dict], path: str) -> None:
    fieldnames = [key for _, key, _ in COLS]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    print(f"Exported to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare training runs.")
    parser.add_argument("--model", default=None, help="Filter by model name.")
    parser.add_argument("--dataset", default=None, help="Filter by dataset name.")
    parser.add_argument("--sort", default="val_qwk_offset", choices=SORT_KEYS,
                        help="Metric to sort by (default: test_qwk_offset).")
    parser.add_argument("--export", default=None, metavar="FILE",
                        help="Export table to a CSV file (e.g. comparison.csv).")
    args = parser.parse_args()

    runs = load_runs(args.model, args.dataset)
    if not runs:
        print("No runs found.")
        return

    runs = sorted(runs, key=lambda r: r.get(args.sort, 0.0), reverse=True)
    print_table(runs, args.sort)

    out = args.export or os.path.join(RESULTS_DIR, "comparison.csv")
    export_csv(runs, out)


if __name__ == "__main__":
    main()
