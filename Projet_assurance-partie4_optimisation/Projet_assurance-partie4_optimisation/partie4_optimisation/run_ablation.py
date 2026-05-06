"""
Partie 4 — Script principal d'ablation.

Lance les 6 variantes TabM sur le dataset Prudential, log toutes les métriques
sur Weights & Biases et sauvegarde les résultats localement.

Usage :
    cd partie4_optimisation
    python run_ablation.py                    # toutes les variantes
    python run_ablation.py --variants tabm tabm_embed   # variantes spécifiques
    python run_ablation.py --no-wandb         # sans W&B (résultats locaux seulement)

Variables d'environnement (alternative à la saisie interactive) :
    WANDB_API_KEY=<votre_cle>
"""
from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Modules locaux
from config_ablation import (
    ABLATION_VARIANTS,
    VARIANTS_BY_ID,
    GlobalConfig,
    AblationVariantConfig,
    NUM_CLASSES,
)
from metrics_ablation import qwk, accuracy, optimize_offsets, apply_offsets
from model_ablation import build_model

# ============================================================
# Résultats attendus de l'article TabM (dataset Prudential non inclus
# directement mais valeurs de référence générales sur les benchmarks tabular)
# Référence article TabM : https://arxiv.org/abs/2410.00203
# Ces valeurs sont là pour fournir un contexte de comparaison qualitatif.
# ============================================================
ARTICLE_REFERENCE = {
    "mlp_plain":      "Baseline MLP — référence inférieure",
    "tabm_packed":    "TabM-packed — plus lent, moins performant que TabM",
    "tabm_mini":      "TabM-mini — bon rapport perf/vitesse",
    "tabm":           "TabM — meilleur que MLP et packed sur la plupart des benchmarks",
    "tabm_mini_embed":"TabM†-mini — amélioration notable avec embeddings",
    "tabm_embed":     "TabM† — meilleur résultat global dans l'article",
}


# ============================================================
# W&B login
# ============================================================
def _wandb_login() -> bool:
    """
    Authentification W&B.
    Demande l'email, le mot de passe et la clé API à l'utilisateur.
    Retourne True si la connexion a réussi, False sinon.
    """
    import wandb  # type: ignore[import]

    # Vérifier si la clé est déjà dans l'environnement
    if os.environ.get("WANDB_API_KEY"):
        print("✓ Clé API W&B trouvée dans WANDB_API_KEY (variable d'environnement).")
        try:
            wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
            return True
        except Exception as e:
            print(f"Erreur lors du login W&B : {e}")
            return False

    print("\n" + "=" * 60)
    print("  Authentification Weights & Biases (W&B)")
    print("  Votre clé API se trouve sur : https://wandb.ai/authorize")
    print("=" * 60)
    email   = input("  Email W&B        : ").strip()
    password = getpass.getpass("  Mot de passe     : ")
    api_key  = getpass.getpass("  Clé API W&B      : ")
    print("=" * 60 + "\n")

    if not api_key:
        print("⚠ Aucune clé API fournie. W&B désactivé.")
        return False

    try:
        wandb.login(key=api_key, relogin=True)
        print(f"✓ Connecté à W&B (email : {email})")
        return True
    except Exception as e:
        print(f"✗ Erreur de connexion W&B : {e}")
        return False


# ============================================================
# Chargement des données
# ============================================================
def load_data(cfg: GlobalConfig) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(cfg.train_path):
        raise FileNotFoundError(
            f"Fichier introuvable : {cfg.train_path}\n"
            "Vérifiez que la branche deep_learning est bien présente dans branches/."
        )
    df = pd.read_csv(cfg.train_path)
    feature_cols = [c for c in df.columns if c != "Response"]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Response"].values.astype(np.float32)
    return X, y


# ============================================================
# Sauvegarde des résultats
# ============================================================
def save_run_results(
    run_dir: str,
    variant: AblationVariantConfig,
    history: list[dict],
    qwk_raw: float,
    qwk_off: float,
    acc_raw: float,
    acc_off: float,
) -> None:
    os.makedirs(run_dir, exist_ok=True)

    # JSON
    result = {
        "variant_id": variant.variant_id,
        "arch_type": variant.arch_type,
        "label": variant.label,
        "description": variant.description,
        "config": dataclasses.asdict(variant),
        "val_qwk_raw": round(qwk_raw, 6),
        "val_qwk_offset": round(qwk_off, 6),
        "val_acc_raw": round(acc_raw, 6),
        "val_acc_offset": round(acc_off, 6),
        "epochs_trained": len(history),
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Courbes
    if history:
        epochs      = [r["epoch"]      for r in history]
        tr_losses   = [r["train_loss"] for r in history]
        va_losses   = [r["val_loss"]   for r in history]
        va_qwks     = [r["val_qwk"]    for r in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{variant.label} — courbes d'entraînement", fontsize=12)

        ax1.plot(epochs, tr_losses, label="train loss")
        ax1.plot(epochs, va_losses, label="val loss")
        ax1.set_xlabel("Époque"); ax1.set_ylabel("MSE loss")
        ax1.set_title("Loss"); ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, va_qwks, color="tab:green", label="val QWK")
        ax2.axhline(qwk_off, color="tab:orange", linestyle="--",
                    label=f"QWK+offsets ({qwk_off:.4f})")
        ax2.set_xlabel("Époque"); ax2.set_ylabel("QWK")
        ax2.set_title("Validation QWK"); ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(run_dir, "curves.png"), dpi=150)
        plt.close(fig)

    print(f"  → Résultats sauvegardés : {run_dir}", flush=True)


# ============================================================
# Tableau de synthèse
# ============================================================
def print_summary(all_results: list[dict]) -> None:
    if not all_results:
        return
    print("\n" + "=" * 80)
    print("  RÉSULTATS D'ABLATION — Prudential Life Insurance (QWK)")
    print("=" * 80)
    header = f"  {'Variante':<35} {'QWK raw':>9} {'QWK +off':>9} {'Acc raw':>8} {'Acc +off':>9}"
    print(header)
    print("  " + "-" * 75)
    for r in sorted(all_results, key=lambda x: x["val_qwk_offset"], reverse=True):
        print(
            f"  {r['label']:<35} "
            f"{r['val_qwk_raw']:>9.4f} "
            f"{r['val_qwk_offset']:>9.4f} "
            f"{r['val_acc_raw']:>8.4f} "
            f"{r['val_acc_offset']:>9.4f}"
        )
    print("=" * 80)


def save_summary_csv(all_results: list[dict], out_dir: str) -> None:
    df = pd.DataFrame(all_results)
    path = os.path.join(out_dir, "ablation_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n  Tableau de synthèse CSV : {path}")


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablation study TabM — Prudential")
    parser.add_argument(
        "--variants", nargs="+",
        choices=list(VARIANTS_BY_ID.keys()),
        default=None,
        help="Sous-ensemble de variantes à lancer (défaut: toutes)",
    )
    parser.add_argument(
        "--no-wandb", action="store_true",
        help="Désactiver le logging W&B",
    )
    parser.add_argument(
        "--wandb-project", default="prudential-tabm-ablation",
        help="Nom du projet W&B",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Surcharger le nombre d'époques pour tous les runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    global_cfg = GlobalConfig()
    os.makedirs(global_cfg.results_dir, exist_ok=True)

    # Sélection des variantes
    variants_to_run = (
        [VARIANTS_BY_ID[v] for v in args.variants]
        if args.variants
        else ABLATION_VARIANTS
    )

    if args.epochs is not None:
        for v in variants_to_run:
            v.epochs = args.epochs

    # Données
    print("\nChargement des données...")
    X, y = load_data(global_cfg)
    print(f"Dataset : {X.shape[0]:,} lignes × {X.shape[1]} features")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=global_cfg.test_size,
        random_state=global_cfg.random_state,
        stratify=y.astype(int),
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # W&B
    use_wandb = not args.no_wandb
    wandb_ok = False
    if use_wandb:
        try:
            import wandb  # type: ignore[import]
            wandb_ok = _wandb_login()
        except ImportError:
            print("⚠ wandb non installé. Lancez : pip install wandb")
            use_wandb = False

    # ---- Boucle ablation ----
    all_results: list[dict] = []
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, variant in enumerate(variants_to_run, 1):
        print(f"\n{'='*70}")
        print(f"  [{i}/{len(variants_to_run)}] {variant.label}")
        print(f"  {variant.description}")
        print(f"{'='*70}")

        run_dir = os.path.join(
            global_cfg.results_dir,
            f"{session_ts}_{variant.variant_id}",
        )

        # Init W&B run
        wb_run = None
        if wandb_ok:
            import wandb  # type: ignore[import]
            wb_run = wandb.init(
                project=args.wandb_project,
                name=f"{variant.variant_id}_{session_ts}",
                config=dataclasses.asdict(variant),
                notes=variant.description,
                tags=[variant.arch_type, "ablation", "prudential"],
                reinit=True,
            )

        t0 = time.time()
        model = build_model(X_train.shape[1], variant)

        try:
            history = model.fit(X_train, y_train, X_val, y_val)
        except Exception as e:
            print(f"\n✗ Erreur pendant l'entraînement de {variant.variant_id} : {e}")
            if wb_run:
                wb_run.finish(exit_code=1)
            continue

        elapsed = time.time() - t0

        # Métriques finales
        val_preds = model.predict(X_val)
        qwk_raw = qwk(val_preds, y_val)
        acc_raw = accuracy(val_preds, y_val)
        offsets = optimize_offsets(val_preds, y_val)
        preds_off = apply_offsets(val_preds, offsets)
        qwk_off = qwk(preds_off, y_val)
        acc_off = accuracy(preds_off, y_val)

        # Affichage
        print(f"\n  ┌─ Résultats {variant.label}")
        print(f"  │  QWK (raw)         : {qwk_raw:.4f}")
        print(f"  │  QWK (+ offsets)   : {qwk_off:.4f}  ← métrique principale")
        print(f"  │  Accuracy (raw)    : {acc_raw:.4f}")
        print(f"  │  Accuracy (offsets): {acc_off:.4f}")
        print(f"  │  Durée             : {elapsed:.0f}s")
        print(f"  └─ Référence article : {ARTICLE_REFERENCE.get(variant.variant_id, 'N/A')}")

        # Sauvegarde locale
        save_run_results(run_dir, variant, history, qwk_raw, qwk_off, acc_raw, acc_off)

        row = {
            "variant_id": variant.variant_id,
            "label": variant.label,
            "arch_type": variant.arch_type,
            "k": variant.k,
            "n_blocks": variant.n_blocks,
            "d_block": variant.d_block,
            "use_embeddings": variant.use_embeddings,
            "val_qwk_raw": round(qwk_raw, 6),
            "val_qwk_offset": round(qwk_off, 6),
            "val_acc_raw": round(acc_raw, 6),
            "val_acc_offset": round(acc_off, 6),
            "epochs_trained": len(history),
            "elapsed_sec": round(elapsed, 1),
        }
        all_results.append(row)

        # Log W&B
        if wb_run:
            import wandb  # type: ignore[import]
            # Log courbes époque par époque
            for h in history:
                wb_run.log({
                    "train_loss": h["train_loss"],
                    "val_loss": h["val_loss"],
                    "val_qwk": h["val_qwk"],
                    "epoch": h["epoch"],
                })
            # Métriques finales
            wb_run.summary.update({
                "val_qwk_raw": qwk_raw,
                "val_qwk_offset": qwk_off,
                "val_acc_raw": acc_raw,
                "val_acc_offset": acc_off,
                "epochs_trained": len(history),
                "elapsed_sec": round(elapsed, 1),
                "reference_article": ARTICLE_REFERENCE.get(variant.variant_id, ""),
            })
            # Sauvegarder les courbes comme artefact W&B
            curves_path = os.path.join(run_dir, "curves.png")
            if os.path.exists(curves_path):
                wb_run.log({"curves": wandb.Image(curves_path)})
            wb_run.finish()

    # ---- Résumé global ----
    print_summary(all_results)
    save_summary_csv(all_results, global_cfg.results_dir)

    # Log tableau comparatif sur W&B
    if wandb_ok and all_results:
        import wandb  # type: ignore[import]
        summary_run = wandb.init(
            project=args.wandb_project,
            name=f"ablation_summary_{session_ts}",
            job_type="summary",
            reinit=True,
        )
        table = wandb.Table(
            columns=list(all_results[0].keys()),
            data=[list(r.values()) for r in all_results],
        )
        summary_run.log({"ablation_results": table})

        # Bar chart QWK
        labels = [r["label"] for r in all_results]
        qwks   = [r["val_qwk_offset"] for r in all_results]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3" if "embed" not in r["variant_id"] else "#4CAF50"
                  for r in all_results]
        bars = ax.barh(labels, qwks, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3)
        ax.set_xlabel("QWK (avec offsets)")
        ax.set_title("Ablation TabM — Prudential Life Insurance")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
        plt.tight_layout()
        bar_path = os.path.join(global_cfg.results_dir, "ablation_barplot.png")
        plt.savefig(bar_path, dpi=150)
        plt.close(fig)
        summary_run.log({"qwk_comparison": wandb.Image(bar_path)})
        summary_run.finish()
        print(f"\n  Tableau et graphique comparatif envoyés sur W&B.")

    print("\n✓ Ablation terminée.")


if __name__ == "__main__":
    main()
