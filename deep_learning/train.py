from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime

# Force unbuffered output regardless of how the script is launched.
if os.environ.get("PYTHONUNBUFFERED") != "1":
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.execv(sys.executable, [sys.executable, "-u"] + sys.argv)
from sklearn.model_selection import train_test_split
from config import DataConfig, MODEL_CONFIGS
from logging_utils import setup_logging, get_logger
from metrics import accuracy, qwk, apply_offsets
from models import MODEL_REGISTRY
from preprocessing import load_data
from results_utils import save_results
from trainer import run_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement d'un modèle de deep learning tabulaire."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture du modèle à entraîner.",
    )
    return parser.parse_args()


def train_one(model_name: str, data_path: str, data_cfg: DataConfig) -> None:
    model_cfg = MODEL_CONFIGS[model_name]
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(data_cfg.results_dir, f"{dataset_name}_{model_name}_{timestamp}")

    setup_logging(log_dir=run_dir, model_name=model_name)
    logger = get_logger(__name__)

    print(f"Model: {model_name} | Dataset: {dataset_name} | Config: {model_cfg}")
    print(f"Run dir: {run_dir}")

    print("Loading data...")
    X, y = load_data(
        data_path=data_path,
        use_cached=data_cfg.use_cached,
        random_state=data_cfg.random_state,
    )
    print(f"Full dataset: {X.shape}")
    input_dim = X.shape[1]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=data_cfg.test_size,
        random_state=data_cfg.random_state,
        stratify=y.astype(int),
    )
    val_fraction = data_cfg.val_size / (1.0 - data_cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=data_cfg.random_state,
        stratify=y_trainval.astype(int),
    )
    print(
        f"Split — train: {X_train.shape[0]:,} | val: {X_val.shape[0]:,} | test: {X_test.shape[0]:,}",
        flush=True,
    )

    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(input_dim=input_dim, config=model_cfg)
    qwk_raw, qwk_offset, acc_raw, acc_offset, val_offsets, history = run_training(
        model, X_train, y_train, X_val, y_val
    )

    test_preds = model.predict(X_test)
    test_qwk_raw = qwk(test_preds, y_test)
    test_acc_raw = accuracy(test_preds, y_test)
    test_preds_offset = apply_offsets(test_preds, val_offsets)
    test_qwk_offset = qwk(test_preds_offset, y_test)
    test_acc_offset = accuracy(test_preds_offset, y_test)

    summary_lines = [
        "=" * 55,
        f"  RESULTS — {model_name} on {dataset_name}",
        "=" * 55,
        f"  {'Metric':<30} {'Val':>8}  {'Test':>8}",
        f"  {'-' * 49}",
        f"  {'QWK (raw)':<30} {qwk_raw:>8.4f}  {test_qwk_raw:>8.4f}",
        f"  {'QWK (offsets)':<30} {qwk_offset:>8.4f}  {test_qwk_offset:>8.4f}",
        f"  {'Accuracy (raw)':<30} {acc_raw:>8.4f}  {test_acc_raw:>8.4f}",
        f"  {'Accuracy (offsets)':<30} {acc_offset:>8.4f}  {test_acc_offset:>8.4f}",
        "=" * 55,
    ]
    summary = "\n".join(summary_lines)
    print(summary, flush=True)
    logger.info(summary)

    save_results(
        run_dir, model_name, dataset_name, model_cfg, data_cfg, history,
        qwk_raw, qwk_offset, acc_raw, acc_offset,
        test_qwk_raw, test_qwk_offset, test_acc_raw, test_acc_offset,
    )


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig()

    if not data_cfg.datasets:
        raise SystemExit("No datasets configured. Add CSV paths to DataConfig.datasets in config.py.")

    total = len(data_cfg.datasets)
    for i, data_path in enumerate(data_cfg.datasets, start=1):
        print(f"\n{'=' * 55}")
        print(f"  [{i}/{total}] {os.path.basename(data_path)}")
        print(f"{'=' * 55}\n", flush=True)
        train_one(args.model, data_path, data_cfg)


if __name__ == "__main__":
    main()
