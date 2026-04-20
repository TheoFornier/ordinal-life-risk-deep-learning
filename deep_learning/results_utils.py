from __future__ import annotations
import dataclasses
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
