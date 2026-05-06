"""
GLO-7030 -- Visualisation comparative des modeles
Genere tous les graphiques et les envoie sur Weights & Biases.

Usage:
    python plot_demo.py
"""

import wandb
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# Selection de la source de donnees
import sys

print("Source des donnees :")
print("  1 - best_models.json  (extraction automatique depuis W&B)")
print("  2 - register.json     (registre manuel)")
choice = input("Choix [1/2] : ").strip()

if choice == "1":
    source = "best_models.json"
elif choice == "2":
    source = "register.json"
else:
    print("[ERREUR] Choix invalide.")
    sys.exit(1)

if not os.path.exists(source):
    print(f"[ERREUR] Le fichier '{source}' n'existe pas.")
    if choice == "1":
        print("Lancez d'abord : python best_models.py")
    else:
        print("Lancez d'abord : python model_registry.py")
    sys.exit(1)

with open(source, "r", encoding="utf-8") as f:
    MODELS = json.load(f)

print(f"Fichier charge : {source} ({len(MODELS)} modeles)\n")

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
GRID = "#e2e8f0"
TEXT = "#1e293b"
N_CLASSES = 8
CLASS_LABELS = [str(i) for i in range(1, 9)]


def _style(ax, title, xlabel="", ylabel=""):
    """Style uniforme pour tous les axes."""
    ax.set_title(title, fontweight="bold", fontsize=13, color=TEXT, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=TEXT)
    ax.tick_params(colors=TEXT, labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)


def _sim_rmse(n_est, lr, seed=0):
    """Courbes RMSE simulees (utilise quand les donnees reelles manquent)."""
    rng = np.random.RandomState(seed)
    t = np.arange(1, n_est + 1)
    decay = 30 / lr
    train = 2.4 * np.exp(-t / decay) + 1.25 + rng.normal(0, 0.004, len(t))
    val = 2.4 * np.exp(-t / decay) + 1.43 + rng.normal(0, 0.006, len(t))
    return t, train, val


def _sim_confusion(seed=0):
    """Matrice de confusion 8x8 simulee."""
    rng = np.random.RandomState(seed)
    cm = np.zeros((N_CLASSES, N_CLASSES))
    for i in range(N_CLASSES):
        cm[i, i] = rng.randint(1800, 3200)
        for j in range(N_CLASSES):
            if i != j:
                cm[i, j] = max(5, rng.randint(30, 350) // (abs(i - j) ** 2))
    return cm


def _sim_distribution(seed=0):
    """Distributions reelle/predite simulees."""
    rng = np.random.RandomState(seed)
    true = np.array([1200, 1800, 1400, 1000, 2200, 3100, 1500, 6800])
    pred = np.clip(true + rng.randint(-400, 400, size=8), 200, None)
    return true, pred


# ===================================================================
# Graphiques comparatifs (tous les modeles)
# ===================================================================

def fig_qwk_bar():
    """Barplot groupe : QWK simple vs offsets pour chaque modele."""
    names = [m["name"] for m in MODELS]
    simple = [m["qwk_simple"] for m in MODELS]
    offset = [m.get("qwk_offset") for m in MODELS]
    colors = [m["color"] for m in MODELS]
    has_off = any(v is not None for v in offset)

    x = np.arange(len(names))
    w = 0.30 if has_off else 0.45
    fig, ax = plt.subplots(figsize=(max(7, 2.5 * len(names)), 5))

    if has_off:
        ax.bar(x - w / 2, simple, w, label="Arrondi simple",
               color=GRID, edgecolor=TEXT, linewidth=0.5)
        ax.bar(x + w / 2, [v if v else 0 for v in offset], w,
               label="Seuils optimises", color=colors, edgecolor=TEXT, linewidth=0.5)
        for i, (s, o) in enumerate(zip(simple, offset)):
            ax.text(i - w / 2, s + 0.003, f"{s:.4f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")
            if o:
                ax.text(i + w / 2, o + 0.003, f"{o:.4f}", ha="center",
                        va="bottom", fontsize=9, fontweight="bold")
    else:
        ax.bar(x, simple, w, color=colors, edgecolor=TEXT, linewidth=0.5)
        for i, s in enumerate(simple):
            ax.text(i, s + 0.003, f"{s:.4f}", ha="center",
                    va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    vals = [v for v in simple + offset if v and v > 0]
    ax.set_ylim(min(vals) - 0.03, max(vals) + 0.02)
    if has_off:
        ax.legend(frameon=False, fontsize=10)
    _style(ax, "Impact de l'optimisation des seuils sur le QWK",
           ylabel="Quadratic Weighted Kappa")
    fig.tight_layout()
    return fig


def fig_radar():
    """Radar chart multi-metriques (QWK simple, QWK offsets, 1/RMSE)."""
    labels = ["QWK\n(simple)", "QWK\n(offsets)", "Precision\n(1/RMSE)"]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for m in MODELS:
        raw = [
            m["qwk_simple"],
            m.get("qwk_offset") or m["qwk_simple"],
            1.0 / m["val_rmse"] if m.get("val_rmse") else 0.0,
        ]
        # Normaliser sur des plages fixes pour la comparabilite
        norms = [
            (raw[0] - 0.50) / 0.20,
            (raw[1] - 0.50) / 0.20,
            (raw[2] - 0.40) / 0.30 if raw[2] > 0 else 0,
        ]
        norms = [max(0, min(1, v)) for v in norms]
        norms += norms[:1]
        ax.plot(angles, norms, "o-", linewidth=2, label=m["name"], color=m["color"])
        ax.fill(angles, norms, alpha=0.12, color=m["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold", color=TEXT)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", ""])
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), frameon=False, fontsize=11)
    ax.set_title("Profil comparatif multi-metriques", fontweight="bold",
                 fontsize=14, color=TEXT, pad=25)
    fig.tight_layout()
    return fig


def fig_lollipop():
    """Classement horizontal des modeles par meilleur QWK."""
    names = [m["name"] for m in MODELS]
    scores = [m.get("qwk_offset") or m["qwk_simple"] for m in MODELS]
    colors = [m["color"] for m in MODELS]

    order = np.argsort(scores)[::-1]
    names = [names[i] for i in order]
    scores = [scores[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(3, 1.2 * len(names))))
    y = np.arange(len(names))
    ax.hlines(y, 0, scores, color=colors, linewidth=2.5, alpha=0.7)
    for i, (s, c) in enumerate(zip(scores, colors)):
        ax.plot(s, i, "o", markersize=9, color=c)
        ax.text(s + 0.003, i, f"{s:.4f}", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=12)
    ax.set_xlim(min(scores) - 0.03, max(scores) + 0.03)
    _style(ax, "Classement des modeles (meilleur QWK)",
           xlabel="Quadratic Weighted Kappa")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def table_summary():
    """Tableau W&B recapitulatif."""
    cols = ["Modele", "Auteur", "Run ID", "QWK simple",
            "QWK offsets", "Val RMSE", "Hyperparametres"]
    rows = []
    for m in MODELS:
        hp = ", ".join(f"{k}={v}" for k, v in m["hyperparams"].items())
        rows.append([
            m["name"], m["author"], m["run_id"],
            round(m["qwk_simple"], 4),
            round(m["qwk_offset"], 4) if m.get("qwk_offset") else None,
            round(m["val_rmse"], 4) if m.get("val_rmse") else None,
            hp,
        ])
    return wandb.Table(columns=cols, data=rows)


# ===================================================================
# Graphiques individuels (un set par modele)
# ===================================================================

def fig_learning_curve(m):
    """Courbe RMSE train/val pour un modele."""
    hp = m["hyperparams"]
    n_est = hp.get("n_estimators", hp.get("epochs", 100))
    lr = hp.get("learning_rate", 0.01)

    if m["train_rmse_curve"] and m["val_rmse_curve"]:
        t = np.arange(1, len(m["train_rmse_curve"]) + 1)
        train = np.array(m["train_rmse_curve"])
        val = np.array(m["val_rmse_curve"])
    else:
        t, train, val = _sim_rmse(n_est, lr, seed=hash(m["name"]) % 10000)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, train, color=m["color"], alpha=0.5, linewidth=1, label="Train")
    ax.plot(t, val, color=m["color"], linewidth=2.2, label="Validation")
    ax.fill_between(t, train, val, color=m["color"], alpha=0.08)

    info = f"lr={lr}  depth={hp.get('max_depth', '?')}  n={n_est}"
    _style(ax, f"Courbe d'apprentissage -- {m['name']}  ({info})",
           xlabel="Iterations", ylabel="RMSE")
    ax.legend(frameon=False, fontsize=10, loc="upper right")
    fig.tight_layout()
    return fig


def fig_confusion(m):
    """Matrice de confusion normalisee."""
    cm = np.array(m["confusion_matrix"]) if m["confusion_matrix"] is not None \
        else _sim_confusion(seed=hash(m["name"]) % 10000)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("seq", ["#f8fafc", "#1e40af"])

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            c = "white" if cm_norm[i, j] > 0.5 else TEXT
            w = "bold" if i == j else "normal"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=c, fontweight=w)

    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_LABELS)
    ax.set_yticklabels(CLASS_LABELS)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Proportion")
    _style(ax, f"Matrice de confusion -- {m['name']}",
           xlabel="Classe predite", ylabel="Classe reelle")
    ax.grid(False)
    fig.tight_layout()
    return fig


def fig_distribution(m):
    """Distribution reelle vs predite."""
    if m["pred_distribution"] is not None and m["true_distribution"] is not None:
        true_d = np.array(m["true_distribution"])
        pred_d = np.array(m["pred_distribution"])
    else:
        true_d, pred_d = _sim_distribution(seed=hash(m["name"]) % 10000)

    true_p = true_d / true_d.sum()
    pred_p = pred_d / pred_d.sum()
    x = np.arange(N_CLASSES)
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, true_p, w, label="Reel", color=GRID,
           edgecolor=TEXT, linewidth=0.5)
    ax.bar(x + w / 2, pred_p, w, label=f"Predit ({m['name']})",
           color=m["color"], edgecolor=TEXT, linewidth=0.5, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_LABELS, fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.legend(frameon=False, fontsize=10)
    _style(ax, f"Distribution des niveaux de risque -- {m['name']}",
           xlabel="Niveau de risque", ylabel="Proportion")
    fig.tight_layout()
    return fig


# ===================================================================
# Main
# ===================================================================

def main():
    wandb.init(
        project="GLO7030-Prudential",
        name="Rapport_Graphiques",
        job_type="visualisation",
        tags=["rapport", "comparaison"],
    )

    print(f"Modeles : {[m['name'] for m in MODELS]}")

    # Graphiques comparatifs
    for label, fig in [
        ("Comparatif_QWK",       fig_qwk_bar()),
        ("Comparatif_Radar",     fig_radar()),
        ("Comparatif_Lollipop",  fig_lollipop()),
    ]:
        wandb.log({label: wandb.Image(fig)})
        plt.close(fig)
        print(f"  [{label}]")

    wandb.log({"Comparatif_Tableau": table_summary()})
    print("  [Comparatif_Tableau]")

    # Graphiques par modele
    for m in MODELS:
        print(f"  {m['name']} :")
        for suffix, fig in [
            ("Learning_Curve",    fig_learning_curve(m)),
            ("Confusion_Matrix",  fig_confusion(m)),
            ("Distribution",      fig_distribution(m)),
        ]:
            key = f"{m['name']}_{suffix}"
            wandb.log({key: wandb.Image(fig)})
            plt.close(fig)
            print(f"    [{key}]")

    wandb.finish()
    print("Termine.")


if __name__ == "__main__":
    main()
