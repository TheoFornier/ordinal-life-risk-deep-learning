from __future__ import annotations
import dataclasses
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from config import NUM_CLASSES


def save_results(
    run_dir: str,
    model_name: str,
    dataset_name: str,
    model_cfg: object,
    data_cfg: object,
    history: list[dict],
    val_qwk_raw: float,
    val_qwk_offset: float,
    val_acc_raw: float,
    val_acc_offset: float,
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    params = {
        "model": model_name,
        "dataset": dataset_name,
        "model_config": dataclasses.asdict(model_cfg),
        "data_config": dataclasses.asdict(data_cfg),
        "val_qwk_raw": round(val_qwk_raw, 6),
        "val_qwk_offset": round(val_qwk_offset, 6),
        "val_acc_raw": round(val_acc_raw, 6),
        "val_acc_offset": round(val_acc_offset, 6),
        "epochs_trained": len(history),
    }
    with open(os.path.join(run_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    if not history:
        print(f"Results saved: {run_dir}", flush=True)
        return

    epochs = [r["epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["val_loss"] for r in history]
    val_qwks = [r["val_qwk"] for r in history]

    fig, (ax_loss, ax_qwk) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} on {dataset_name} — training curves", fontsize=13)

    ax_loss.plot(epochs, train_losses, label="train loss")
    ax_loss.plot(epochs, val_losses, label="val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_qwk.plot(epochs, val_qwks, color="tab:green", label="val QWK")
    ax_qwk.axhline(val_qwk_offset, color="tab:orange", linestyle="--", label=f"val QWK+off ({val_qwk_offset:.4f})")
    ax_qwk.set_xlabel("Epoch")
    ax_qwk.set_ylabel("QWK")
    ax_qwk.set_title("Validation QWK")
    ax_qwk.legend()
    ax_qwk.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots.png"), dpi=150)
    plt.close(fig)

    print(f"Results saved: {run_dir}", flush=True)


def save_distribution_plot(
    run_dir: str,
    y_real: np.ndarray,
    y_synth: np.ndarray | None = None,
) -> None:
    classes = list(range(1, NUM_CLASSES + 1))
    x = np.arange(len(classes))
    to_int = lambda y: np.clip(np.round(y), 1, NUM_CLASSES).astype(int)
    real_counts = [int(np.sum(to_int(y_real) == c)) for c in classes]

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Training data — Response distribution", fontsize=13)
    if y_synth is not None and len(y_synth) > 0:
        synth_counts = [int(np.sum(to_int(y_synth) == c)) for c in classes]
        w = 0.4
        ax.bar(x - w / 2, real_counts, width=w, label="Real", color="steelblue")
        ax.bar(x + w / 2, synth_counts, width=w, label="Synthetic", color="coral")
    else:
        ax.bar(x, real_counts, color="steelblue", label="Real")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Response")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "distribution.png"), dpi=150)
    plt.close(fig)


def save_per_class_accuracy_plot(
    run_dir: str,
    preds_real: np.ndarray,
    y_real: np.ndarray,
    preds_synth: np.ndarray | None = None,
    y_synth: np.ndarray | None = None,
) -> None:
    classes = list(range(1, NUM_CLASSES + 1))
    x = np.arange(len(classes))

    def per_class_acc(preds: np.ndarray, y_true: np.ndarray) -> list[float]:
        p = np.clip(np.round(preds), 1, NUM_CLASSES).astype(int)
        t = np.clip(np.round(y_true), 1, NUM_CLASSES).astype(int)
        return [float(np.mean(p[t == c] == c)) if np.any(t == c) else float("nan") for c in classes]

    real_accs = per_class_acc(preds_real, y_real)

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("Per-class accuracy — validation", fontsize=13)
    if preds_synth is not None and y_synth is not None and len(y_synth) > 0:
        synth_accs = per_class_acc(preds_synth, y_synth)
        w = 0.4
        ax.bar(x - w / 2, real_accs, width=w, label="Real val", color="steelblue")
        ax.bar(x + w / 2, synth_accs, width=w, label="Synthetic", color="coral")
    else:
        ax.bar(x, real_accs, color="steelblue", label="Real val")
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_xlabel("Response")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "per_class_accuracy.png"), dpi=150)
    plt.close(fig)
