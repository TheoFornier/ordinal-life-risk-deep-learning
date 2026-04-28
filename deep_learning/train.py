from __future__ import annotations
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DataConfig, MODEL_CONFIGS
from logging_utils import setup_logging, get_logger
from models import MODEL_REGISTRY
from results_utils import save_results, save_distribution_plot, save_per_class_accuracy_plot
from trainer import run_training

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c != "Response"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Response"].values.astype(np.float32)
    return X, y



def load_test_data(test_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(test_path)
    ids = df["Id"]
    feature_cols = [c for c in df.columns if c != "Response"]
    X = df[feature_cols].values.astype(np.float32)
    return ids, X


def generate_submission(
    model,
    offsets: np.ndarray,
    test_path: str,
    run_dir: str,
    dataset_name: str,
    model_name: str,
) -> None:
    from metrics import apply_offsets
    ids, X_test = load_test_data(test_path)
    raw_preds = model.predict(X_test)
    preds = apply_offsets(raw_preds, offsets)
    preds_int = np.clip(np.round(preds).astype(int), 1, 8)
    submission = pd.DataFrame({"Id": ids, "Response": preds_int})
    out_path = os.path.join(run_dir, f"submission_{dataset_name}_{model_name}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}", flush=True)


def train_one(
    model_name: str,
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    data_cfg: DataConfig,
    X_fit: np.ndarray | None = None,
    y_fit: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    test_path: str | None = None,
    n_val_real: int | None = None,
) -> None:
    model_cfg = MODEL_CONFIGS[model_name]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(data_cfg.results_dir, f"{dataset_name}_{model_name}_{timestamp}")

    setup_logging(log_dir=run_dir, model_name=model_name)
    logger = get_logger(__name__)

    print(f"Model: {model_name} | Dataset: {dataset_name} | Config: {model_cfg}")
    print(f"Run dir: {run_dir}")
    print(
        f"Split — train: {X_train.shape[0]:,} | val: {X_val.shape[0]:,}",
        flush=True,
    )

    input_dim = X_train.shape[1]
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(input_dim=input_dim, config=model_cfg)
    qwk_raw, qwk_offset, acc_raw, acc_offset, offsets, history = run_training(
        model, X_train, y_train, X_val, y_val, X_fit=X_fit, y_fit=y_fit, sample_weight=sample_weight
    )

    summary_lines = [
        "=" * 55,
        f"  RESULTS — {model_name} on {dataset_name}",
        "=" * 55,
        f"  {'Metric':<30} {'Val':>8}",
        f"  {'-' * 40}",
        f"  {'QWK (raw)':<30} {qwk_raw:>8.4f}",
        f"  {'QWK (offsets)':<30} {qwk_offset:>8.4f}",
        f"  {'Accuracy (raw)':<30} {acc_raw:>8.4f}",
        f"  {'Accuracy (offsets)':<30} {acc_offset:>8.4f}",
        "=" * 55,
    ]
    summary = "\n".join(summary_lines)
    print(summary, flush=True)
    logger.info(summary)

    save_results(
        run_dir, model_name, dataset_name, model_cfg, data_cfg, history,
        qwk_raw, qwk_offset, acc_raw, acc_offset,
    )

    y_real_train = y_fit if y_fit is not None else y_train
    y_synth_train = y_train[len(y_fit):] if y_fit is not None else None
    X_synth_train = X_train[len(X_fit):] if X_fit is not None else None

    save_distribution_plot(run_dir, y_real_train, y_synth_train)

    n_real_val = n_val_real if n_val_real is not None else len(y_val)
    y_val_real_plot = y_val[:n_real_val]
    y_val_synth_plot = y_val[n_real_val:] if n_real_val < len(y_val) else None
    save_distribution_plot(
        run_dir, y_val_real_plot, y_val_synth_plot,
        title="Validation data — Response distribution",
        filename="distribution_val.png",
    )

    val_preds = model.predict(X_val)
    preds_val_real = val_preds[:n_real_val]
    preds_val_synth = val_preds[n_real_val:] if n_real_val < len(val_preds) else None
    save_per_class_accuracy_plot(run_dir, preds_val_real, y_val_real_plot, preds_val_synth, y_val_synth_plot)

    generate_submission(model, offsets, test_path or data_cfg.test_path, run_dir, dataset_name, model_name)


def main() -> None:
    data_cfg = DataConfig()

    print("Loading base dataset...")
    X, y = load_data(data_cfg.train_path)
    print(f"Full dataset: {X.shape}")

    X_train_base, X_val, y_train_base, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y.astype(int),
    )

    total = int(data_cfg.run_base_training) + len(data_cfg.synthetic_datasets)
    run_idx = 0

    if data_cfg.run_base_training:
        run_idx += 1
        print(f"\n{'=' * 55}")
        print(f"  [{run_idx}/{total}] train_clean (base only)")
        print(f"{'=' * 55}\n", flush=True)
        train_one(data_cfg.model, "train_clean", X_train_base, y_train_base, X_val, y_val, data_cfg)

    for synth_folder in data_cfg.synthetic_datasets:
        run_idx += 1
        dataset_name = os.path.basename(synth_folder.rstrip("/\\"))
        print(f"\n{'=' * 55}")
        print(f"  [{run_idx}/{total}] {dataset_name}")
        print(f"{'=' * 55}\n", flush=True)

        X_folder, y_folder = load_data(os.path.join(synth_folder, "train_clean.csv"))
        X_train_folder, X_val_folder, y_train_folder, y_val_folder = train_test_split(
            X_folder, y_folder,
            test_size=0.2,
            random_state=data_cfg.random_state,
            stratify=y_folder.astype(int),
        )

        synth_csvs = [
            f for f in os.listdir(synth_folder)
            if f.endswith(".csv") and f not in ("train_clean.csv", "test_clean.csv")
        ]
        X_synth_parts, y_synth_parts = [], []
        for fname in synth_csvs:
            Xs, ys = load_data(os.path.join(synth_folder, fname))
            X_synth_parts.append(Xs)
            y_synth_parts.append(ys)

        n_val_real = len(X_val_folder)
        X_synth_all = np.concatenate(X_synth_parts, axis=0)
        y_synth_all = np.concatenate(y_synth_parts, axis=0)
        val_synth_mask = np.zeros(len(X_synth_all), dtype=bool)
        if data_cfg.max_synth_val_ratio > 0 and len(X_synth_all) > 0:
            n_val_synth = min(len(X_synth_all), int(n_val_real * data_cfg.max_synth_val_ratio))
            rng_val = np.random.default_rng(data_cfg.random_state)
            val_idx = rng_val.choice(len(X_synth_all), size=n_val_synth, replace=False)
            val_synth_mask[val_idx] = True
            X_val_folder = np.concatenate([X_val_folder, X_synth_all[val_idx]], axis=0)
            y_val_folder = np.concatenate([y_val_folder, y_synth_all[val_idx]], axis=0)
            print(f"Val set augmented with {n_val_synth:,} synthetic rows ({data_cfg.max_synth_val_ratio}× real val)", flush=True)
        # Exclude rows already in val from the training pool to prevent data leak
        X_synth_all = X_synth_all[~val_synth_mask]
        y_synth_all = y_synth_all[~val_synth_mask]
        rng = np.random.default_rng(data_cfg.random_state)
        kept = []
        for c in range(1, data_cfg.num_classes + 1):
            real_count = int(np.sum(y_train_folder.astype(int) == c))
            max_c = int(real_count * data_cfg.max_synth_ratio)
            idx_c = np.where(y_synth_all.astype(int) == c)[0]
            if len(idx_c) > max_c:
                idx_c = rng.choice(idx_c, size=max_c, replace=False)
            kept.append(idx_c)
        kept_idx = np.concatenate(kept)
        X_synth_parts = [X_synth_all[kept_idx]]
        y_synth_parts = [y_synth_all[kept_idx]]
        n_synth = len(kept_idx)
        print(
            f"Synthetic rows after per-class cap ({data_cfg.max_synth_ratio}× real per class): "
            f"{n_synth:,} kept of {len(y_synth_all):,}",
            flush=True,
        )
        n_real = len(X_train_folder)
        sample_weight = np.concatenate([
            np.ones(n_real, dtype=np.float32),
            np.full(n_synth, data_cfg.synth_sample_weight, dtype=np.float32),
        ])

        X_train_aug = np.concatenate([X_train_folder] + X_synth_parts, axis=0)
        y_train_aug = np.concatenate([y_train_folder] + y_synth_parts, axis=0)

        synth_test_path = os.path.join(synth_folder, "test_clean.csv")
        train_one(data_cfg.model, dataset_name, X_train_aug, y_train_aug, X_val_folder, y_val_folder, data_cfg,
                  X_fit=X_train_folder, y_fit=y_train_folder, sample_weight=sample_weight,
                  test_path=synth_test_path, n_val_real=n_val_real)


if __name__ == "__main__":
    main()
