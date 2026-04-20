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
    model_cfg: object,
    data_cfg: object,
    history: list[dict],
    qwk_raw: float,
    qwk_offset: float,
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    # --- hyperparameters.json ---
    params = {
        "model": model_name,
        "model_config": dataclasses.asdict(model_cfg),
        "data_config": dataclasses.asdict(data_cfg),
        "final_qwk_raw": round(qwk_raw, 6),
        "final_qwk_offset": round(qwk_offset, 6),
    }
    with open(os.path.join(run_dir, "hyperparameters.json"), "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    # --- plots.png ---
    if not history:
        return

    epochs = [r["epoch"] for r in history]
    train_losses = [r["train_loss"] for r in history]
    val_losses = [r["val_loss"] for r in history]
    val_qwks = [r["val_qwk"] for r in history]

    fig, (ax_loss, ax_qwk) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{model_name} — training curves", fontsize=13)

    ax_loss.plot(epochs, train_losses, label="train loss")
    ax_loss.plot(epochs, val_losses, label="val loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_qwk.plot(epochs, val_qwks, color="tab:green", label="val QWK")
    ax_qwk.set_xlabel("Epoch")
    ax_qwk.set_ylabel("QWK")
    ax_qwk.set_title("Validation QWK")
    ax_qwk.legend()
    ax_qwk.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "plots.png"), dpi=150)
    plt.close(fig)

    print(f"Results saved: {run_dir}", flush=True)
