from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Script EDA orienté dataset Kaggle "Prudential Life Insurance Assessment"
# - train.csv contient la cible Response (8 niveaux)
# - test.csv n'a pas la cible
# - Product_Info_2 est une variable catégorielle texte
# - de nombreuses colonnes sont normalisées / déjà numériques
# -----------------------------------------------------------------------------

DATA_PATH = Path("data/train.csv")
OUTPUT_DIR = Path("graph_data/eda_plots")
MAX_HIST_NUMERIC = 24
MAX_BOXPLOT_NUMERIC = 16
MAX_CATEGORICAL_PLOTS = 12
MAX_BINARY_PLOTS = 16
CORR_TOP_N = 25


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Impossible de trouver {path.resolve()}. Mets train.csv à côté du script "
            "ou modifie DATA_PATH."
        )
    return pd.read_csv(path)


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Retourne :
    - numeric_cols : colonnes numériques hors Id et Response
    - categorical_cols : colonnes object/category
    - binary_cols : sous-ensemble numérique quasi binaire {0,1}
    - ordinal_small_int_cols : entiers avec petit nombre de modalités (>2)
    """
    excluded = {"Id", "Response"}
    feature_cols = [c for c in df.columns if c not in excluded]

    categorical_cols = [
        c for c in feature_cols
        if pd.api.types.is_object_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])
    ]

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]

    binary_cols: list[str] = []
    ordinal_small_int_cols: list[str] = []

    for c in numeric_cols:
        vals = df[c].dropna().unique()
        if len(vals) == 0:
            continue

        # Colonnes 0/1 typiques du dataset (ex: Medical_Keyword_*).
        if set(pd.Series(vals).astype(float).round(10).tolist()).issubset({0.0, 1.0}):
            binary_cols.append(c)
            continue

        # Colonnes entières avec peu de modalités.
        s = df[c].dropna()
        if len(s) and np.all(np.isclose(s, np.round(s))) and s.nunique() <= 20:
            ordinal_small_int_cols.append(c)

    return numeric_cols, categorical_cols, binary_cols, ordinal_small_int_cols


def save_dataset_overview(df: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for c in df.columns:
        rows.append(
            {
                "column": c,
                "dtype": str(df[c].dtype),
                "missing_count": int(df[c].isna().sum()),
                "missing_pct": round(float(df[c].isna().mean() * 100), 3),
                "n_unique": int(df[c].nunique(dropna=True)),
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        by=["missing_pct", "n_unique"], ascending=[False, False]
    )
    summary.to_csv(out_dir / "dataset_overview.csv", index=False)


def plot_response_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    if "Response" not in df.columns:
        return

    counts = df["Response"].value_counts().sort_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Distribution de la cible Response")
    ax.set_xlabel("Classe de risque")
    ax.set_ylabel("Nombre d'observations")

    total = counts.sum()
    for i, val in enumerate(counts.values):
        ax.text(i, val, f"{val}\n({val/total:.1%})", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "01_response_distribution.png", dpi=180)
    plt.close(fig)


def plot_missingness(df: pd.DataFrame, out_dir: Path) -> None:
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]

    if miss.empty:
        return

    fig_h = max(6, min(18, 0.28 * len(miss)))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    miss.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Part de valeurs manquantes par colonne")
    ax.set_xlabel("Proportion manquante")
    ax.set_ylabel("Colonnes")
    fig.tight_layout()
    fig.savefig(out_dir / "02_missingness_by_column.png", dpi=180)
    plt.close(fig)


def _choose_numeric_columns_for_hist(df: pd.DataFrame, numeric_cols: list[str], max_cols: int) -> list[str]:
    # On privilégie les colonnes les plus variées et non binaires.
    candidates = []
    for c in numeric_cols:
        nunique = df[c].nunique(dropna=True)
        if nunique <= 2:
            continue
        candidates.append((c, nunique, df[c].isna().mean()))

    candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
    return [c for c, _, _ in candidates[:max_cols]]


def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path) -> None:
    cols = _choose_numeric_columns_for_hist(df, numeric_cols, MAX_HIST_NUMERIC)
    if not cols:
        return

    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        ax.hist(s, bins=30)
        ax.set_title(col)
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Histogrammes des principales variables numériques", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "03_numeric_histograms.png", dpi=180)
    plt.close(fig)


def plot_numeric_boxplots(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path) -> None:
    cols = _choose_numeric_columns_for_hist(df, numeric_cols, MAX_BOXPLOT_NUMERIC)
    if not cols:
        return

    n = len(cols)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        s = df[col].dropna()
        ax.boxplot(s, vert=True)
        ax.set_title(col)
        ax.set_xticks([])

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Boxplots des principales variables numériques", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "04_numeric_boxplots.png", dpi=180)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path) -> None:
    usable = [c for c in numeric_cols if df[c].nunique(dropna=True) > 2]
    if len(usable) < 2:
        return

    # Sélectionne les colonnes les plus corrélées globalement pour garder une heatmap lisible.
    corr_full = df[usable].corr(numeric_only=True)
    strength = corr_full.abs().sum().sort_values(ascending=False)
    top_cols = strength.head(min(CORR_TOP_N, len(strength))).index.tolist()
    corr = df[top_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr.values, aspect="auto")
    ax.set_xticks(range(len(top_cols)))
    ax.set_yticks(range(len(top_cols)))
    ax.set_xticklabels(top_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(top_cols, fontsize=8)
    ax.set_title("Heatmap des corrélations (variables numériques principales)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Corrélation")
    fig.tight_layout()
    fig.savefig(out_dir / "05_correlation_heatmap.png", dpi=180)
    plt.close(fig)


def plot_product_info_2(df: pd.DataFrame, out_dir: Path) -> None:
    col = "Product_Info_2"
    if col not in df.columns:
        return

    counts = df[col].fillna("<NA>").value_counts()

    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Fréquences de Product_Info_2")
    ax.set_xlabel("Modalité")
    ax.set_ylabel("Nombre d'observations")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(out_dir / "06_product_info_2_counts.png", dpi=180)
    plt.close(fig)

    if "Response" in df.columns:
        ordered = counts.index.tolist()
        data = [df.loc[df[col].fillna("<NA>") == cat, "Response"].dropna() for cat in ordered]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data, labels=ordered, showfliers=False)
        ax.set_title("Response par modalité de Product_Info_2")
        ax.set_xlabel("Product_Info_2")
        ax.set_ylabel("Response")
        ax.tick_params(axis="x", labelrotation=45)
        fig.tight_layout()
        fig.savefig(out_dir / "07_product_info_2_vs_response_boxplot.png", dpi=180)
        plt.close(fig)


def plot_top_categorical_counts(df: pd.DataFrame, categorical_cols: list[str], out_dir: Path) -> None:
    if not categorical_cols:
        return

    cols = sorted(
        categorical_cols,
        key=lambda c: (df[c].nunique(dropna=True), df[c].isna().mean(), c)
    )[:MAX_CATEGORICAL_PLOTS]

    n = len(cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, cols):
        vc = df[col].fillna("<NA>").value_counts().head(20)
        vc.plot(kind="bar", ax=ax)
        ax.set_title(f"{col} (top 20)")
        ax.tick_params(axis="x", labelrotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Variables catégorielles texte", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "08_categorical_counts.png", dpi=180)
    plt.close(fig)


def plot_binary_prevalence(df: pd.DataFrame, binary_cols: list[str], out_dir: Path) -> None:
    if not binary_cols:
        return

    prevalence = df[binary_cols].mean(skipna=True).sort_values(ascending=False).head(MAX_BINARY_PLOTS)

    fig_h = max(6, 0.4 * len(prevalence))
    fig, ax = plt.subplots(figsize=(12, fig_h))
    prevalence.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Prévalence des principales colonnes binaires (ex: Medical_Keyword_*)")
    ax.set_xlabel("Proportion de 1")
    ax.set_ylabel("Colonnes")
    fig.tight_layout()
    fig.savefig(out_dir / "09_binary_prevalence.png", dpi=180)
    plt.close(fig)


def plot_response_by_top_numeric(df: pd.DataFrame, numeric_cols: list[str], out_dir: Path) -> None:
    if "Response" not in df.columns:
        return

    usable = [c for c in numeric_cols if c != "Response" and df[c].nunique(dropna=True) > 2]
    if not usable:
        return

    # Corrélation absolue avec Response, purement exploratoire.
    corr_to_target = (
        df[usable + ["Response"]]
        .corr(numeric_only=True)["Response"]
        .drop(labels=["Response"])
        .abs()
        .sort_values(ascending=False)
    )
    top_cols = corr_to_target.head(8).index.tolist()

    n = len(top_cols)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, top_cols):
        x = df[col]
        y = df["Response"]
        ax.scatter(x, y, alpha=0.15, s=8)
        ax.set_title(f"{col} vs Response")
        ax.set_xlabel(col)
        ax.set_ylabel("Response")

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle("Variables numériques les plus liées à Response", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "10_top_numeric_vs_response.png", dpi=180)
    plt.close(fig)


def save_brief_report(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    binary_cols: list[str],
    ordinal_small_int_cols: list[str],
    out_dir: Path,
) -> None:
    lines = []
    lines.append(f"Nombre de lignes : {len(df)}")
    lines.append(f"Nombre de colonnes : {df.shape[1]}")
    lines.append(f"Colonnes numériques : {len(numeric_cols)}")
    lines.append(f"Colonnes catégorielles texte : {len(categorical_cols)}")
    lines.append(f"Colonnes binaires {0,1} : {len(binary_cols)}")
    lines.append(f"Colonnes entières à peu de modalités : {len(ordinal_small_int_cols)}")
    lines.append("")

    if "Response" in df.columns:
        counts = df["Response"].value_counts().sort_index()
        lines.append("Distribution de Response :")
        for cls, cnt in counts.items():
            lines.append(f"  - {cls}: {cnt} ({cnt / len(df):.2%})")
        lines.append("")

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(20)
    if not miss.empty:
        lines.append("Top 20 colonnes les plus manquantes :")
        for col, pct in miss.items():
            lines.append(f"  - {col}: {pct:.2%}")
        lines.append("")

    if categorical_cols:
        lines.append("Colonnes catégorielles texte détectées :")
        for c in categorical_cols:
            lines.append(f"  - {c} ({df[c].nunique(dropna=True)} modalités)")
        lines.append("")

    with open(out_dir / "00_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ensure_output_dir(OUTPUT_DIR)
    df = load_data(DATA_PATH)

    numeric_cols, categorical_cols, binary_cols, ordinal_small_int_cols = split_columns(df)

    save_dataset_overview(df, OUTPUT_DIR)
    save_brief_report(
        df,
        numeric_cols,
        categorical_cols,
        binary_cols,
        ordinal_small_int_cols,
        OUTPUT_DIR,
    )

    plot_response_distribution(df, OUTPUT_DIR)
    plot_missingness(df, OUTPUT_DIR)
    plot_numeric_histograms(df, numeric_cols, OUTPUT_DIR)
    plot_numeric_boxplots(df, numeric_cols, OUTPUT_DIR)
    plot_correlation_heatmap(df, numeric_cols, OUTPUT_DIR)
    plot_product_info_2(df, OUTPUT_DIR)
    plot_top_categorical_counts(df, categorical_cols, OUTPUT_DIR)
    plot_binary_prevalence(df, binary_cols, OUTPUT_DIR)
    plot_response_by_top_numeric(df, numeric_cols, OUTPUT_DIR)

    print(f"Graphiques enregistrés dans : {OUTPUT_DIR.resolve()}")
    print("Fichiers principaux générés :")
    for p in sorted(OUTPUT_DIR.glob("*")):
        print(f" - {p.name}")


if __name__ == "__main__":
    main()
