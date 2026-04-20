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
from models import MODEL_REGISTRY
from preprocessing import load_data
from results_utils import save_results
from trainer import run_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement d'un modèle de deep learning tabulaire sur le dataset Prudential."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY.keys()),
        help="Architecture du modèle à entraîner.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig()
    model_cfg = MODEL_CONFIGS[args.model]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(data_cfg.results_dir, f"{args.model}_{timestamp}")

    setup_logging(log_dir=run_dir, model_name=args.model)
    logger = get_logger(__name__)

    print(f"Model: {args.model} | Config: {model_cfg}")
    print(f"Run dir: {run_dir}")

    print("Loading data...")
    X_all, y_all, _, _ = load_data(
        train_raw=data_cfg.train_raw,
        test_raw=data_cfg.test_raw,
        train_clean=data_cfg.train_clean,
        test_clean=data_cfg.test_clean,
        use_cached=data_cfg.use_cached,
        random_state=data_cfg.random_state,
    )
    print(f"Train: {X_all.shape}")
    input_dim = X_all.shape[1]

    X_train, X_val, y_train, y_val = train_test_split(
        X_all,
        y_all,
        test_size=data_cfg.val_size,
        random_state=data_cfg.random_state,
        stratify=y_all.astype(int),
    )
    print(f"Split — train: {X_train.shape[0]:,} | val: {X_val.shape[0]:,}")

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(input_dim=input_dim, config=model_cfg)
    qwk_raw, qwk_offset, history = run_training(model, X_train, y_train, X_val, y_val)

    summary_lines = [
        "=" * 50,
        f"  RESULTS — {args.model}",
        "=" * 50,
        f"  QWK val raw:                 {qwk_raw:>8.4f}",
        f"  QWK val+offsets:             {qwk_offset:>8.4f}",
        "=" * 50,
    ]
    summary = "\n".join(summary_lines)
    print(summary, flush=True)
    logger.info(summary)

    save_results(run_dir, args.model, model_cfg, data_cfg, history, qwk_raw, qwk_offset)


if __name__ == "__main__":
    main()
