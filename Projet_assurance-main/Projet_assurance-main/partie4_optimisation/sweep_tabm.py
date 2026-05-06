"""
Partie 4 — W&B Sweep : recherche bayésienne d'hyperparamètres pour TabM.

Ce script crée un sweep W&B qui explore :
  - arch_type : tabm, tabm-mini
  - k         : taille de l'ensemble
  - n_blocks  : profondeur du MLP
  - d_block   : largeur des couches
  - dropout   : régularisation
  - lr        : taux d'apprentissage
  - weight_decay
  - use_embeddings + n_bins + d_embedding

Usage :
    cd partie4_optimisation
    python sweep_tabm.py --count 30      # lance 30 runs bayésiens
    python sweep_tabm.py --count 50 --arch tabm_embed   # focus sur TabM†

Variables d'environnement (alternative à la saisie interactive) :
    WANDB_API_KEY=<votre_cle>
"""
from __future__ import annotations

import argparse
import dataclasses
import getpass
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import wandb  # type: ignore[import]

from config_ablation import GlobalConfig, AblationVariantConfig, NUM_CLASSES
from metrics_ablation import qwk, optimize_offsets, apply_offsets
from model_ablation import build_model


# ============================================================
# Configurations des sweeps disponibles
# ============================================================

# --- Sweep A : Toutes architectures (exploration large) ---
SWEEP_CONFIG_ALL = {
    "method": "bayes",
    "metric": {
        "name": "val_qwk_offset",
        "goal": "maximize",
    },
    "parameters": {
        "arch_type": {
            "values": ["tabm", "tabm-mini"],
        },
        "k": {
            "values": [8, 16, 32, 64],
        },
        "n_blocks": {
            "values": [2, 3, 4],
        },
        "d_block": {
            "values": [256, 384, 512, 768],
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.5,
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2,
        },
        "use_embeddings": {
            "values": [False, True],
        },
        "n_bins": {
            "values": [32, 48, 64],
        },
        "d_embedding": {
            "values": [8, 16, 32],
        },
    },
}

# --- Sweep B : Focus TabM† (avec embeddings seulement, recherche fine) ---
SWEEP_CONFIG_EMBED = {
    "method": "bayes",
    "metric": {
        "name": "val_qwk_offset",
        "goal": "maximize",
    },
    "parameters": {
        "arch_type": {
            "values": ["tabm", "tabm-mini"],
        },
        "k": {
            "values": [16, 32, 64],
        },
        "n_blocks": {
            "values": [2, 3],
        },
        "d_block": {
            "values": [256, 384, 512, 768, 1024],
        },
        "dropout": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.4,
        },
        "lr": {
            "distribution": "log_uniform_values",
            "min": 5e-4,
            "max": 3e-3,
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 5e-3,
        },
        "use_embeddings": {"value": True},
        "n_bins": {
            "values": [32, 48, 64, 96],
        },
        "d_embedding": {
            "values": [8, 16, 32],
        },
    },
}

SWEEP_CONFIGS = {
    "all":        SWEEP_CONFIG_ALL,
    "tabm_embed": SWEEP_CONFIG_EMBED,
}


# ============================================================
# Login W&B
# ============================================================
def _wandb_login() -> None:
    if os.environ.get("WANDB_API_KEY"):
        print("✓ Clé API W&B trouvée dans WANDB_API_KEY.")
        wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
        return

    print("\n" + "=" * 60)
    print("  Authentification Weights & Biases (W&B)")
    print("  Votre clé API : https://wandb.ai/authorize")
    print("=" * 60)
    email    = input("  Email W&B        : ").strip()
    password = getpass.getpass("  Mot de passe     : ")
    api_key  = getpass.getpass("  Clé API W&B      : ")
    print("=" * 60 + "\n")

    if not api_key:
        raise ValueError("Aucune clé API fournie. Abandon.")

    wandb.login(key=api_key, relogin=True)
    print(f"✓ Connecté à W&B (email : {email})\n")


# ============================================================
# Chargement données (une seule fois avant le sweep)
# ============================================================
_X_TRAIN: np.ndarray | None = None
_X_VAL:   np.ndarray | None = None
_Y_TRAIN: np.ndarray | None = None
_Y_VAL:   np.ndarray | None = None


def _load_split(cfg: GlobalConfig) -> None:
    global _X_TRAIN, _X_VAL, _Y_TRAIN, _Y_VAL
    df = pd.read_csv(cfg.train_path)
    feature_cols = [c for c in df.columns if c != "Response"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Response"].values.astype(np.float32)
    _X_TRAIN, _X_VAL, _Y_TRAIN, _Y_VAL = train_test_split(
        X, y, test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y.astype(int),
    )
    print(f"Données chargées : train={len(_X_TRAIN):,}  val={len(_X_VAL):,}")


# ============================================================
# Fonction de train appelée par l'agent de sweep
# ============================================================
def _sweep_train() -> None:
    """Appelée automatiquement par wandb.agent pour chaque combinaison."""
    run = wandb.init()
    cfg_wb = wandb.config

    # Construire la config variante depuis les paramètres W&B
    variant = AblationVariantConfig(
        variant_id=f"sweep_{cfg_wb.arch_type}",
        arch_type=cfg_wb.arch_type,
        k=cfg_wb.k,
        n_blocks=cfg_wb.n_blocks,
        d_block=cfg_wb.d_block,
        dropout=cfg_wb.dropout,
        lr=cfg_wb.lr,
        weight_decay=cfg_wb.weight_decay,
        batch_size=1024,
        epochs=50,
        early_stopping_patience=10,
        use_embeddings=cfg_wb.use_embeddings,
        n_bins=getattr(cfg_wb, "n_bins", 48),
        d_embedding=getattr(cfg_wb, "d_embedding", 16),
        label=f"sweep {cfg_wb.arch_type} k={cfg_wb.k}",
    )

    input_dim = _X_TRAIN.shape[1]
    model = build_model(input_dim, variant)

    try:
        t0 = time.time()
        history = model.fit(_X_TRAIN, _Y_TRAIN, _X_VAL, _Y_VAL)
        elapsed = time.time() - t0

        # Log courbes époque par époque
        for h in history:
            wandb.log({
                "train_loss": h["train_loss"],
                "val_loss":   h["val_loss"],
                "val_qwk":    h["val_qwk"],
                "epoch":      h["epoch"],
            })

        # Métriques finales
        val_preds = model.predict(_X_VAL)
        qwk_raw = qwk(val_preds, _Y_VAL)
        offsets = optimize_offsets(val_preds, _Y_VAL)
        preds_off = apply_offsets(val_preds, offsets)
        qwk_off = qwk(preds_off, _Y_VAL)

        wandb.log({
            "val_qwk_raw":    qwk_raw,
            "val_qwk_offset": qwk_off,
            "epochs_trained": len(history),
            "elapsed_sec":    round(elapsed, 1),
        })
        wandb.summary["val_qwk_offset"] = qwk_off
        wandb.summary["val_qwk_raw"]    = qwk_raw

        print(
            f"[Sweep] arch={variant.arch_type} k={variant.k} "
            f"blocks={variant.n_blocks} d={variant.d_block} "
            f"embed={variant.use_embeddings} "
            f"→ QWK+off={qwk_off:.4f}",
            flush=True,
        )

    except Exception as e:
        print(f"✗ Erreur sweep run : {e}", flush=True)
        wandb.finish(exit_code=1)
        return

    wandb.finish()


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W&B Sweep hyperparamètres — TabM sur Prudential"
    )
    parser.add_argument(
        "--count", type=int, default=30,
        help="Nombre de runs du sweep (défaut: 30)",
    )
    parser.add_argument(
        "--arch", choices=list(SWEEP_CONFIGS.keys()), default="all",
        help="Configuration de sweep : 'all' (exploration large) ou 'tabm_embed' (focus embeddings)",
    )
    parser.add_argument(
        "--project", default="prudential-tabm-sweep",
        help="Nom du projet W&B",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global_cfg = GlobalConfig()

    # Login W&B
    _wandb_login()

    # Charger les données
    _load_split(global_cfg)

    # Créer le sweep
    sweep_config = SWEEP_CONFIGS[args.arch]
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    print(f"\n✓ Sweep créé : {sweep_id}")
    print(f"  Projet W&B  : {args.project}")
    print(f"  Config      : {args.arch}")
    print(f"  Runs prévus : {args.count}")
    print(f"\n  Suivi en temps réel : https://wandb.ai/home → {args.project}\n")

    # Lancer l'agent
    wandb.agent(sweep_id, function=_sweep_train, count=args.count)

    print("\n✓ Sweep terminé.")
    print(f"  Consultez les résultats sur : https://wandb.ai/home → {args.project}")


if __name__ == "__main__":
    main()
