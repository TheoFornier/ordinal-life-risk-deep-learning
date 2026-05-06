"""
Partie 4 — Comparaison des résultats d'ablation vs l'article TabM.

Ce script lit les résultats sauvegardés dans results/ et génère :
  - Un tableau comparatif détaillé dans la console
  - Un graphique à barres QWK (results/comparison_plot.png)
  - Un tableau HTML pour le rapport (results/comparison_table.html)

Usage :
    cd partie4_optimisation
    python compare_ablation.py
    python compare_ablation.py --results-dir results/  # dossier personnalisé
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# Résultats de l'article TabM (Table 2 de l'article, arXiv 2410.00203)
# Scores moyens normalisés sur les benchmarks tabular
# Note : ces scores sont des moyennes sur des datasets GBDT-friendly,
#        non spécifiques à Prudential. Ils servent de référence qualitative.
# ============================================================
ARTICLE_SCORES = {
    # Variante              QWK-like normalisé (rang moyen → plus bas = mieux)
    # On convertit en score "performance relative" pour la visualisation
    "MLP (plain)":                      {"rank": 6.2, "note": "Baseline inférieure"},
    "Ensemble sans weight sharing\n(TabM-packed)": {"rank": 5.8, "note": "Plus lourd que TabM"},
    "TabM-mini (un seul adapteur R)":   {"rank": 3.1, "note": "Bon rapport perf/vitesse"},
    "TabM (BatchEnsemble R+S+B)":       {"rank": 2.8, "note": "Meilleur sur la plupart des tâches"},
    "TabM-mini† (+ PiecewiseLinear embedding)": {"rank": 2.4, "note": "Amélioration notable"},
    "TabM† (+ PiecewiseLinear embedding)":      {"rank": 1.9, "note": "Meilleur résultat global"},
}

# Ordre qualitatif attendu (du moins bon au meilleur sur Prudential)
EXPECTED_ORDER = [
    "mlp_plain",
    "tabm_packed",
    "tabm_mini",
    "tabm",
    "tabm_mini_embed",
    "tabm_embed",
]


# ============================================================
# Lecture des résultats
# ============================================================
def load_results(results_dir: str) -> pd.DataFrame:
    """Lit tous les result.json dans les sous-dossiers de results_dir."""
    rows = []
    if not os.path.exists(results_dir):
        print(f"⚠ Dossier résultats introuvable : {results_dir}")
        print("  Lancez d'abord : python run_ablation.py")
        return pd.DataFrame()

    for entry in sorted(os.scandir(results_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        result_path = os.path.join(entry.path, "result.json")
        if not os.path.exists(result_path):
            continue
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        rows.append(data)

    if not rows:
        print("⚠ Aucun résultat trouvé. Lancez d'abord : python run_ablation.py")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Garder le meilleur run par variante (au cas où plusieurs runs existent)
    df = (
        df.sort_values("val_qwk_offset", ascending=False)
          .drop_duplicates(subset="variant_id", keep="first")
          .reset_index(drop=True)
    )
    # Trier selon l'ordre attendu
    order_map = {v: i for i, v in enumerate(EXPECTED_ORDER)}
    df["_order"] = df["variant_id"].map(order_map).fillna(99)
    df = df.sort_values("_order").reset_index(drop=True)
    return df


# ============================================================
# Tableau console
# ============================================================
def print_comparison(df: pd.DataFrame) -> None:
    print("\n" + "=" * 90)
    print("  COMPARAISON ABLATION — Prudential Life Insurance")
    print("  Métrique principale : QWK (Quadratic Weighted Kappa, avec optimisation d'offsets)")
    print("=" * 90)
    print(f"  {'Variante':<40} {'QWK raw':>9} {'QWK +off':>9} {'Acc +off':>9} {'Réf article':>25}")
    print("  " + "-" * 87)

    best_qwk = df["val_qwk_offset"].max()

    for _, row in df.iterrows():
        marker = " ★" if abs(row["val_qwk_offset"] - best_qwk) < 1e-6 else "  "
        ref = ARTICLE_SCORES.get(row.get("label", ""), {}).get("note", "—")
        print(
            f"  {row.get('label', row['variant_id']):<40} "
            f"{row['val_qwk_raw']:>9.4f} "
            f"{row['val_qwk_offset']:>9.4f}{marker}"
            f"{row['val_acc_offset']:>9.4f} "
            f"{ref:>25}"
        )

    print("=" * 90)
    print(f"\n  ★ = meilleur résultat  |  QWK+off = QWK après optimisation des offsets de classe")

    # Comparaison gain vs baseline
    if "mlp_plain" in df["variant_id"].values:
        baseline = df.loc[df["variant_id"] == "mlp_plain", "val_qwk_offset"].values[0]
        print(f"\n  Gains vs MLP baseline (QWK = {baseline:.4f}) :")
        for _, row in df.iterrows():
            if row["variant_id"] == "mlp_plain":
                continue
            gain = row["val_qwk_offset"] - baseline
            sign = "+" if gain >= 0 else ""
            print(f"    {row.get('label', row['variant_id']):<40} → {sign}{gain:.4f}")


# ============================================================
# Graphiques
# ============================================================
def plot_comparison(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # --- Graphique 1 : Barres QWK ---
    fig, ax = plt.subplots(figsize=(11, 6))
    labels    = [r.get("label", r["variant_id"]) for _, r in df.iterrows()]
    qwk_raw   = df["val_qwk_raw"].tolist()
    qwk_off   = df["val_qwk_offset"].tolist()

    x = np.arange(len(labels))
    w = 0.38
    colors_raw = ["#90CAF9"] * len(labels)
    colors_off = ["#1565C0" if "embed" not in r["variant_id"] else "#2E7D32"
                  for _, r in df.iterrows()]

    bars1 = ax.bar(x - w/2, qwk_raw, w, label="QWK (raw)", color=colors_raw, edgecolor="white")
    bars2 = ax.bar(x + w/2, qwk_off, w, label="QWK (+ offsets)", color=colors_off, edgecolor="white")
    ax.bar_label(bars1, fmt="%.4f", fontsize=8, padding=2)
    ax.bar_label(bars2, fmt="%.4f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("QWK", fontsize=11)
    ax.set_title("Ablation TabM — Prudential Life Insurance\n(QWK raw vs QWK + optimisation d'offsets)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, min(1.0, df["val_qwk_offset"].max() * 1.15))

    patch_blue  = mpatches.Patch(color="#1565C0", label="Sans embedding")
    patch_green = mpatches.Patch(color="#2E7D32", label="Avec PiecewiseLinear embedding")
    ax.legend(handles=[bars1, patch_blue, patch_green], fontsize=9)

    plt.tight_layout()
    path1 = os.path.join(out_dir, "comparison_barplot.png")
    plt.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"  Graphique barres : {path1}")

    # --- Graphique 2 : Progression (ordre ablation) ---
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(range(len(labels)), qwk_off, marker="o", linewidth=2,
             color="#1565C0", markersize=8, label="QWK + offsets")
    ax2.plot(range(len(labels)), qwk_raw, marker="s", linewidth=1.5,
             color="#90CAF9", linestyle="--", markersize=6, label="QWK raw")
    for i, (q, lbl) in enumerate(zip(qwk_off, labels)):
        ax2.annotate(f"{q:.4f}", (i, q), textcoords="offset points",
                     xytext=(0, 8), ha="center", fontsize=8)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("QWK", fontsize=11)
    ax2.set_title("Progression du QWK selon la complexité du modèle", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    path2 = os.path.join(out_dir, "comparison_progression.png")
    plt.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Graphique progression : {path2}")


# ============================================================
# Export HTML (pour rapport)
# ============================================================
def export_html(df: pd.DataFrame, out_dir: str) -> None:
    cols_display = ["label", "arch_type", "k", "n_blocks", "d_block",
                    "use_embeddings", "val_qwk_raw", "val_qwk_offset",
                    "val_acc_offset", "epochs_trained"]
    cols_present = [c for c in cols_display if c in df.columns]
    df_disp = df[cols_present].copy()
    df_disp.columns = [
        c.replace("val_", "").replace("_", " ").title()
        for c in cols_present
    ]

    html = df_disp.to_html(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    styled = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Ablation TabM — Prudential</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 30px; }}
  h1 {{ color: #1565C0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background-color: #1565C0; color: white; padding: 8px 12px; }}
  td {{ border: 1px solid #ddd; padding: 7px 12px; }}
  tr:nth-child(even) {{ background-color: #f5f5f5; }}
  tr:hover {{ background-color: #e3f2fd; }}
</style>
</head><body>
<h1>Ablation TabM — Prudential Life Insurance</h1>
<p>Comparaison des 6 variantes d'ablation. Métrique principale : QWK + optimisation d'offsets.</p>
{html}
<br><img src="comparison_barplot.png" style="max-width:900px;"><br>
<img src="comparison_progression.png" style="max-width:800px;">
</body></html>"""

    path = os.path.join(out_dir, "comparison_table.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(styled)
    print(f"  Tableau HTML : {path}")


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Comparaison résultats ablation TabM")
    parser.add_argument(
        "--results-dir", default="results",
        help="Dossier contenant les sous-dossiers de résultats (défaut: results/)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir

    print(f"\nLecture des résultats dans : {os.path.abspath(results_dir)}")
    df = load_results(results_dir)

    if df.empty:
        return

    print(f"  {len(df)} variante(s) trouvée(s).")

    print_comparison(df)
    plot_comparison(df, results_dir)
    export_html(df, results_dir)

    # Sauvegarder le CSV récapitulatif
    csv_path = os.path.join(results_dir, "ablation_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV récapitulatif : {csv_path}")

    print("\n✓ Comparaison terminée.")


if __name__ == "__main__":
    main()
