from __future__ import annotations
import json
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from config import DataConfig, MODEL_CONFIGS
from logging_utils import setup_logging, get_logger
from models import MODEL_REGISTRY
from results_utils import save_results, save_distribution_plot, save_per_class_accuracy_plot, save_confusion_matrix_plot
from trainer import run_training
from metrics import apply_offsets, optimize_offsets, qwk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_data(data_path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    feature_cols = [c for c in df.columns if c not in ["Id", "Response"]]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Response"].values.astype(np.float32)
    return X, y



def class_weights(y: np.ndarray, num_classes: int, power: float = 0.5) -> np.ndarray:
    classes, counts = np.unique(y.astype(int), return_counts=True)
    freq = dict(zip(classes, counts))
    n_total = len(y)
    w = np.array(
        [(n_total / (num_classes * freq.get(int(yi), 1))) ** power for yi in y],
        dtype=np.float32,
    )
    w /= w.mean()
    return w


def load_test_data(test_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(test_path)
    ids = df["Id"]
    feature_cols = [c for c in df.columns if c not in ["Id", "Response"]]
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
        f"Split - train: {X_train.shape[0]:,} | val: {X_val.shape[0]:,}",
        flush=True,
    )

    input_dim = X_train.shape[1]
    model_cls = MODEL_REGISTRY[model_name]
    model = model_cls(input_dim=input_dim, config=model_cfg)
    qwk_raw, qwk_offset, acc_raw, acc_offset, offsets, history = run_training(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        X_fit=X_fit,
        y_fit=y_fit,
        sample_weight=sample_weight,
        run_dir=run_dir,
    )

    summary_lines = [
        "=" * 55,
        f"  RESULTS - {model_name} on {dataset_name}",
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
        title="Validation data - Response distribution",
        filename="distribution_val.png",
    )

    val_preds = model.predict(X_val)
    preds_val_real = val_preds[:n_real_val]
    preds_val_synth = val_preds[n_real_val:] if n_real_val < len(val_preds) else None
    save_per_class_accuracy_plot(run_dir, preds_val_real, y_val_real_plot, preds_val_synth, y_val_synth_plot,
                                 filename="per_class_accuracy_raw.png",
                                 title="Per-class accuracy - validation (raw)")

    save_confusion_matrix_plot(run_dir, preds_val_real, y_val_real_plot,
                               filename="confusion_matrix_raw.png",
                               title="Validation confusion matrix (raw)")

    val_preds_offset = apply_offsets(val_preds, offsets)
    preds_offset_real = val_preds_offset[:n_real_val]
    preds_offset_synth = val_preds_offset[n_real_val:] if n_real_val < len(val_preds_offset) else None
    save_per_class_accuracy_plot(run_dir, preds_offset_real, y_val_real_plot, preds_offset_synth, y_val_synth_plot,
                                 filename="per_class_accuracy_offset.png",
                                 title="Per-class accuracy - validation (with offsets)")
    save_confusion_matrix_plot(run_dir, preds_offset_real, y_val_real_plot,
                               filename="confusion_matrix_offset.png",
                               title="Validation confusion matrix (with offsets)")

    generate_submission(model, offsets, test_path or data_cfg.test_path, run_dir, dataset_name, model_name)


def train_kfold(
    model_name: str,
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    data_cfg: DataConfig,
    X_synth: np.ndarray | None = None,
    y_synth: np.ndarray | None = None,
    test_path: str | None = None,
) -> None:
    model_cfg = MODEL_CONFIGS[model_name]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(data_cfg.results_dir, f"{dataset_name}_{model_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    setup_logging(log_dir=run_dir, model_name=model_name)
    logger = get_logger(__name__)

    ids, X_test = load_test_data(test_path or data_cfg.test_path)

    n_folds = data_cfg.n_folds
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=data_cfg.random_state)

    oof_preds = np.zeros(len(y), dtype=np.float32)
    test_preds = np.zeros(len(X_test), dtype=np.float32)
    fold_qwk_raw: list[float] = []
    fold_qwk_offset: list[float] = []

    has_synth = X_synth is not None and y_synth is not None
    print(f"\nModel: {model_name} | Dataset: {dataset_name} | {n_folds}-fold CV", flush=True)
    if has_synth:
        print(f"Synthetic pool: {len(y_synth):,} rows (capped per fold at {data_cfg.max_synth_ratio}× real per class)", flush=True)
    print(f"Run dir: {run_dir}\n", flush=True)

    for fold_i, (train_idx, val_idx) in enumerate(skf.split(X, y.astype(int)), start=1):
        print(f"{'=' * 55}", flush=True)
        print(f"  Fold {fold_i}/{n_folds}  (train={len(train_idx):,}  val={len(val_idx):,})", flush=True)
        print(f"{'=' * 55}", flush=True)

        X_train_f, X_val_f = X[train_idx], X[val_idx]
        y_train_f, y_val_f = y[train_idx], y[val_idx]

        X_train_final, y_train_final = X_train_f, y_train_f
        X_fit_fold: np.ndarray | None = None
        y_fit_fold: np.ndarray | None = None

        if has_synth:
            rng = np.random.default_rng(data_cfg.random_state + fold_i)
            kept = []
            for c in range(1, data_cfg.num_classes + 1):
                real_count = int(np.sum(y_train_f.astype(int) == c))
                max_c = int(real_count * data_cfg.max_synth_ratio)
                idx_c = np.where(y_synth.astype(int) == c)[0]
                if len(idx_c) > max_c:
                    idx_c = rng.choice(idx_c, size=max_c, replace=False)
                kept.append(idx_c)
            kept_idx = np.concatenate(kept)
            X_synth_fold, y_synth_fold = X_synth[kept_idx], y_synth[kept_idx]
            n_synth_fold = len(kept_idx)
            print(f"  Synthetic rows this fold: {n_synth_fold:,}", flush=True)

            real_weights = (
                class_weights(y_train_f, data_cfg.num_classes, data_cfg.class_weight_power)
                if data_cfg.use_class_weights
                else np.ones(len(y_train_f), dtype=np.float32)
            )
            fold_weights = np.concatenate([
                real_weights,
                np.full(n_synth_fold, data_cfg.synth_sample_weight, dtype=np.float32),
            ])
            X_train_final = np.concatenate([X_train_f, X_synth_fold])
            y_train_final = np.concatenate([y_train_f, y_synth_fold])
            X_fit_fold = X_train_f
            y_fit_fold = y_train_f
        else:
            fold_weights = (
                class_weights(y_train_f, data_cfg.num_classes, data_cfg.class_weight_power)
                if data_cfg.use_class_weights
                else None
            )

        fold_dir = os.path.join(run_dir, f"fold_{fold_i}")
        os.makedirs(fold_dir, exist_ok=True)
        model_cls = MODEL_REGISTRY[model_name]
        model = model_cls(input_dim=X.shape[1], config=model_cfg)

        qwk_raw, qwk_off, acc_raw, acc_off, fold_offsets, history = run_training(
            model,
            X_train_final, y_train_final,
            X_val_f, y_val_f,
            X_fit=X_fit_fold,
            y_fit=y_fit_fold,
            sample_weight=fold_weights,
            run_dir=fold_dir,
        )
        fold_qwk_raw.append(qwk_raw)
        fold_qwk_offset.append(qwk_off)

        oof_preds[val_idx] = model.predict(X_val_f)
        test_preds += model.predict(X_test) / n_folds

        val_preds_fold = oof_preds[val_idx]
        val_preds_offset_fold = apply_offsets(val_preds_fold, fold_offsets)
        save_confusion_matrix_plot(
            fold_dir, val_preds_fold, y_val_f,
            filename="confusion_matrix_raw.png",
            title=f"Fold {fold_i} confusion matrix (raw) QWK={qwk_raw:.4f}",
        )
        save_confusion_matrix_plot(
            fold_dir, val_preds_offset_fold, y_val_f,
            filename="confusion_matrix_offset.png",
            title=f"Fold {fold_i} confusion matrix (offset) QWK={qwk_off:.4f}",
        )
        save_per_class_accuracy_plot(
            fold_dir, val_preds_fold, y_val_f,
            filename="per_class_accuracy_raw.png",
            title=f"Fold {fold_i} per-class accuracy (raw)",
        )
        save_per_class_accuracy_plot(
            fold_dir, val_preds_offset_fold, y_val_f,
            filename="per_class_accuracy_offset.png",
            title=f"Fold {fold_i} per-class accuracy (offset)",
        )
        save_results(fold_dir, model_name, dataset_name, model_cfg, data_cfg, history,
                     qwk_raw, qwk_off, acc_raw, acc_off)

        logger.info(f"Fold {fold_i}/{n_folds} done. QWK_raw={qwk_raw:.4f} QWK_off={qwk_off:.4f}")

    # OOF metrics on full dataset
    final_offsets = optimize_offsets(oof_preds, y)
    oof_offset_preds = apply_offsets(oof_preds, final_offsets)
    oof_qwk_raw = qwk(oof_preds, y)
    oof_qwk_off = qwk(oof_offset_preds, y)

    print(f"\n{'=' * 55}", flush=True)
    print(f"  OOF QWK (raw):    {oof_qwk_raw:.4f}", flush=True)
    print(f"  OOF QWK (offset): {oof_qwk_off:.4f}", flush=True)
    print(f"  Per-fold QWK raw:    {[round(v, 4) for v in fold_qwk_raw]}", flush=True)
    print(f"  Per-fold QWK offset: {[round(v, 4) for v in fold_qwk_offset]}", flush=True)
    print(f"{'=' * 55}\n", flush=True)
    logger.info(f"OOF QWK raw={oof_qwk_raw:.4f} offset={oof_qwk_off:.4f}")

    save_confusion_matrix_plot(
        run_dir, oof_preds, y,
        filename="oof_confusion_matrix_raw.png",
        title=f"OOF confusion matrix (raw) QWK={oof_qwk_raw:.4f}",
    )
    save_confusion_matrix_plot(
        run_dir, oof_offset_preds, y,
        filename="oof_confusion_matrix_offset.png",
        title=f"OOF confusion matrix (offset) QWK={oof_qwk_off:.4f}",
    )
    save_per_class_accuracy_plot(
        run_dir, oof_preds, y,
        filename="oof_per_class_accuracy_raw.png",
        title="OOF per-class accuracy (raw)",
    )
    save_per_class_accuracy_plot(
        run_dir, oof_offset_preds, y,
        filename="oof_per_class_accuracy_offset.png",
        title="OOF per-class accuracy (offset)",
    )

    summary = {
        "model": model_name,
        "dataset": dataset_name,
        "n_folds": n_folds,
        "oof_qwk_raw": round(float(oof_qwk_raw), 6),
        "oof_qwk_offset": round(float(oof_qwk_off), 6),
        "fold_qwk_raw": [round(v, 6) for v in fold_qwk_raw],
        "fold_qwk_offset": [round(v, 6) for v in fold_qwk_offset],
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Submission from averaged test predictions
    final_test_preds = apply_offsets(test_preds, final_offsets)
    final_test_preds_int = np.clip(np.round(final_test_preds).astype(int), 1, 8)
    submission = pd.DataFrame({"Id": ids, "Response": final_test_preds_int})
    out_path = os.path.join(run_dir, f"submission_{dataset_name}_{model_name}.csv")
    submission.to_csv(out_path, index=False)
    print(f"Submission saved: {out_path}", flush=True)


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
        if data_cfg.use_kfold:
            train_kfold(data_cfg.model, "train_clean", X, y, data_cfg)
        else:
            base_weights = (
                class_weights(y_train_base, data_cfg.num_classes, data_cfg.class_weight_power)
                if data_cfg.use_class_weights
                else None
            )
            train_one(data_cfg.model, "train_clean", X_train_base, y_train_base, X_val, y_val, data_cfg,
                      sample_weight=base_weights)

    for synth_folder in data_cfg.synthetic_datasets:
        run_idx += 1
        dataset_name = os.path.basename(synth_folder.rstrip("/\\"))
        print(f"\n{'=' * 55}")
        print(f"  [{run_idx}/{total}] {dataset_name}")
        print(f"{'=' * 55}\n", flush=True)

        X_folder, y_folder = load_data(os.path.join(synth_folder, "train_clean.csv"))

        synth_csvs = [
            f for f in os.listdir(synth_folder)
            if f.endswith(".csv") and f not in ("train_clean.csv", "test_clean.csv")
        ]
        X_synth_parts, y_synth_parts = [], []
        for fname in synth_csvs:
            Xs, ys = load_data(os.path.join(synth_folder, fname))
            X_synth_parts.append(Xs)
            y_synth_parts.append(ys)
        X_synth_all = np.concatenate(X_synth_parts, axis=0)
        y_synth_all = np.concatenate(y_synth_parts, axis=0)

        synth_test_path = os.path.join(synth_folder, "test_clean.csv")

        X_train_folder, X_val_folder, y_train_folder, y_val_folder = train_test_split(
            X_folder, y_folder,
            test_size=0.2,
            random_state=data_cfg.random_state,
            stratify=y_folder.astype(int),
        )
        n_val_real = len(X_val_folder)
        val_synth_mask = np.zeros(len(X_synth_all), dtype=bool)
        if data_cfg.max_synth_val_ratio > 0 and len(X_synth_all) > 0:
            n_val_synth = min(len(X_synth_all), int(n_val_real * data_cfg.max_synth_val_ratio))
            rng_val = np.random.default_rng(data_cfg.random_state)
            val_idx = rng_val.choice(len(X_synth_all), size=n_val_synth, replace=False)
            val_synth_mask[val_idx] = True
            X_val_folder = np.concatenate([X_val_folder, X_synth_all[val_idx]], axis=0)
            y_val_folder = np.concatenate([y_val_folder, y_synth_all[val_idx]], axis=0)
            print(f"Val set augmented with {n_val_synth:,} synthetic rows ({data_cfg.max_synth_val_ratio}× real val)", flush=True)
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
        n_synth = len(kept_idx)
        print(
            f"Synthetic rows after per-class cap ({data_cfg.max_synth_ratio}× real per class): "
            f"{n_synth:,} kept of {len(y_synth_all):,}",
            flush=True,
        )
        n_real = len(X_train_folder)
        real_weights = (
            class_weights(y_train_folder, data_cfg.num_classes, data_cfg.class_weight_power)
            if data_cfg.use_class_weights
            else np.ones(n_real, dtype=np.float32)
        )
        sample_weight = np.concatenate([
            real_weights,
            np.full(n_synth, data_cfg.synth_sample_weight, dtype=np.float32),
        ])
        X_train_aug = np.concatenate([X_train_folder, X_synth_all[kept_idx]], axis=0)
        y_train_aug = np.concatenate([y_train_folder, y_synth_all[kept_idx]], axis=0)
        train_one(data_cfg.model, dataset_name, X_train_aug, y_train_aug, X_val_folder, y_val_folder, data_cfg,
                  X_fit=X_train_folder, y_fit=y_train_folder, sample_weight=sample_weight,
                  test_path=synth_test_path, n_val_real=n_val_real)


if __name__ == "__main__":
    main()
