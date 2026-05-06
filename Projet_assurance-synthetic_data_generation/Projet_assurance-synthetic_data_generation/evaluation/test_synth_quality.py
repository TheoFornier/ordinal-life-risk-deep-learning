from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def align_columns(real_df: pd.DataFrame, synth_df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    real_cols = list(real_df.columns)
    synth_cols = list(synth_df.columns)

    if target_col not in real_cols:
        raise ValueError(f"Colonne cible '{target_col}' absente du réel.")
    if target_col not in synth_cols:
        raise ValueError(f"Colonne cible '{target_col}' absente du synthétique.")

    common_cols = [c for c in real_cols if c in synth_cols]
    missing_in_synth = [c for c in real_cols if c not in synth_cols]
    extra_in_synth = [c for c in synth_cols if c not in real_cols]

    if missing_in_synth:
        print("\n[ATTENTION] Colonnes absentes du synthétique :", missing_in_synth)
    if extra_in_synth:
        print("\n[INFO] Colonnes supplémentaires dans le synthétique :", extra_in_synth)

    real_df = real_df[common_cols].copy()
    synth_df = synth_df[common_cols].copy()
    return real_df, synth_df


def exact_duplicates(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> tuple[int, float]:
    real_tuples = set(map(tuple, real_df.to_numpy()))
    synth_tuples = list(map(tuple, synth_df.to_numpy()))
    dup_count = sum(row in real_tuples for row in synth_tuples)
    dup_ratio = dup_count / len(synth_df) if len(synth_df) > 0 else 0.0
    return dup_count, dup_ratio


def rounded_duplicates(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    numeric_cols: list[str],
    decimals: int = 3,
) -> tuple[int, float]:
    real_copy = real_df.copy()
    synth_copy = synth_df.copy()

    for col in numeric_cols:
        real_copy[col] = pd.to_numeric(real_copy[col], errors="coerce").round(decimals)
        synth_copy[col] = pd.to_numeric(synth_copy[col], errors="coerce").round(decimals)

    real_tuples = set(map(tuple, real_copy.to_numpy()))
    synth_tuples = list(map(tuple, synth_copy.to_numpy()))
    dup_count = sum(row in real_tuples for row in synth_tuples)
    dup_ratio = dup_count / len(synth_df) if len(synth_df) > 0 else 0.0
    return dup_count, dup_ratio


def nearest_neighbor_analysis(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> dict:
    numeric_cols = real_x.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return {"available": False}

    real_num = real_x[numeric_cols].copy()
    synth_num = synth_x[numeric_cols].copy()

    real_num = real_num.fillna(real_num.median())
    synth_num = synth_num.fillna(real_num.median())

    # normalisation simple pour éviter qu'une seule colonne domine
    means = real_num.mean()
    stds = real_num.std().replace(0, 1)

    real_scaled = (real_num - means) / stds
    synth_scaled = (synth_num - means) / stds

    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(real_scaled)
    distances, indices = nn.kneighbors(synth_scaled)

    distances = distances.ravel()

    return {
        "available": True,
        "mean_distance": float(np.mean(distances)),
        "median_distance": float(np.median(distances)),
        "p01_distance": float(np.quantile(distances, 0.01)),
        "p05_distance": float(np.quantile(distances, 0.05)),
        "p10_distance": float(np.quantile(distances, 0.10)),
        "very_close_ratio_d_lt_0_01": float(np.mean(distances < 0.01)),
        "very_close_ratio_d_lt_0_05": float(np.mean(distances < 0.05)),
        "very_close_ratio_d_lt_0_10": float(np.mean(distances < 0.10)),
    }


def response_distribution(real_df: pd.DataFrame, synth_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    real_dist = real_df[target_col].value_counts(normalize=True).sort_index()
    synth_dist = synth_df[target_col].value_counts(normalize=True).sort_index()

    all_idx = sorted(set(real_dist.index).union(set(synth_dist.index)))
    table = pd.DataFrame({
        "real_pct": real_dist.reindex(all_idx, fill_value=0.0),
        "synth_pct": synth_dist.reindex(all_idx, fill_value=0.0),
    })
    table["abs_diff"] = (table["real_pct"] - table["synth_pct"]).abs()
    return table


def build_real_vs_synth_auc(real_x: pd.DataFrame, synth_x: pd.DataFrame) -> float:
    real_x = real_x.copy()
    synth_x = synth_x.copy()

    real_x["__label__"] = 0
    synth_x["__label__"] = 1

    full = pd.concat([real_x, synth_x], axis=0, ignore_index=True)
    y = full["__label__"]
    X = full.drop(columns="__label__")

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                categorical_features,
            ),
        ]
    )

    clf = Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    return float(auc)


def heuristic_judgment(
    exact_dup_ratio: float,
    rounded_dup_ratio: float,
    nn_stats: dict,
    auc: float,
) -> str:
    flags = []

    if exact_dup_ratio > 0.01:
        flags.append("trop de doublons exacts")
    if rounded_dup_ratio > 0.05:
        flags.append("trop de quasi-doublons")
    if nn_stats.get("available", False):
        if nn_stats["very_close_ratio_d_lt_0_01"] > 0.05:
            flags.append("beaucoup de lignes quasi identiques au réel")
        if nn_stats["median_distance"] < 0.10:
            flags.append("distance médiane anormalement faible")
    if auc > 0.90:
        flags.append("réel vs synthétique trop facile à distinguer")
    elif auc < 0.60:
        flags.append("synthétique très proche du réel ou test peu informatif")

    if not flags:
        return "Aucun signal majeur d’artefact évident."
    return " / ".join(flags)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", required=True, help="Chemin vers train_clean.csv")
    parser.add_argument("--synth", required=True, help="Chemin vers gpt_b.csv")
    parser.add_argument("--target", default="Response", help="Nom de la colonne cible")
    parser.add_argument("--rounded-decimals", type=int, default=3, help="Arrondi pour quasi-doublons")
    args = parser.parse_args()

    real_df = load_csv(args.real)
    synth_df = load_csv(args.synth)

    real_df, synth_df = align_columns(real_df, synth_df, args.target)

    numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()

    print("\n==============================")
    print("TAILLE DES DATASETS")
    print("==============================")
    print(f"Réal       : {real_df.shape}")
    print(f"Synthétique: {synth_df.shape}")

    exact_dup_count, exact_dup_ratio = exact_duplicates(real_df, synth_df)
    rounded_dup_count, rounded_dup_ratio = rounded_duplicates(
        real_df, synth_df, numeric_cols=numeric_cols, decimals=args.rounded_decimals
    )

    print("\n==============================")
    print("DOUBLONS")
    print("==============================")
    print(f"Doublons exacts                : {exact_dup_count} ({exact_dup_ratio:.4%})")
    print(f"Quasi-doublons arrondis x{args.rounded_decimals}: {rounded_dup_count} ({rounded_dup_ratio:.4%})")

    feature_cols = [c for c in real_df.columns if c != args.target]
    nn_stats = nearest_neighbor_analysis(real_df[feature_cols], synth_df[feature_cols])

    print("\n==============================")
    print("PLUS PROCHE VOISIN (NUMERIQUE)")
    print("==============================")
    if nn_stats["available"]:
        for k, v in nn_stats.items():
            if k != "available":
                print(f"{k:30s}: {v:.6f}")
    else:
        print("Aucune colonne numérique exploitable.")

    dist_table = response_distribution(real_df, synth_df, args.target)
    print("\n==============================")
    print("DISTRIBUTION DE Response")
    print("==============================")
    print(dist_table)

    auc = build_real_vs_synth_auc(real_df[feature_cols], synth_df[feature_cols])
    print("\n==============================")
    print("REAL VS SYNTH AUC")
    print("==============================")
    print(f"AUC = {auc:.6f}")

    verdict = heuristic_judgment(
        exact_dup_ratio=exact_dup_ratio,
        rounded_dup_ratio=rounded_dup_ratio,
        nn_stats=nn_stats,
        auc=auc,
    )

    print("\n==============================")
    print("VERDICT HEURISTIQUE")
    print("==============================")
    print(verdict)

    print("\n==============================")
    print("INTERPRETATION RAPIDE")
    print("==============================")
    print("- Beaucoup de doublons exacts/quasi-doublons -> dataset suspect.")
    print("- Distances très faibles -> probable copie ou mémorisation.")
    print("- AUC très haute (>0.90) -> synthétique très différent du réel.")
    print("- AUC très basse (<0.60) + distances très faibles -> possible contamination/copie.")
    print("- Le bon cas est plutôt : peu de doublons, distances non nulles, AUC modérée.")


if __name__ == "__main__":
    main()