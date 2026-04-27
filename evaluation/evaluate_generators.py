from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from sdv.metadata import SingleTableMetadata
from sdmetrics.reports.single_table import QualityReport


TARGET_COL = "Response"
ID_COL = "Id"
RANDOM_STATE = 42


# Charge le dataset réel nettoyé qui sert de référence pour toutes les évaluations.
def load_real_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


# Liste tous les fichiers CSV synthétiques présents dans un dossier donné.
def list_synthetic_files(folder: Path) -> list[Path]:
    return sorted([p for p in folder.glob("*.csv") if p.is_file()])


# Aligne les colonnes du dataset synthétique sur celles du dataset réel.
def align_synthetic_to_real(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in real_df.columns if c not in syn_df.columns]
    extra = [c for c in syn_df.columns if c not in real_df.columns]

    if missing:
        raise ValueError(f"Colonnes manquantes dans le synthétique : {missing}")
    if extra:
        syn_df = syn_df.drop(columns=extra)

    return syn_df[real_df.columns].copy()


# Construit automatiquement le metadata SDV à partir du dataset réel.
def build_metadata(real_df: pd.DataFrame) -> dict:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_df)
    return metadata.to_dict()


# Fait un split train/validation stratifié sur le dataset réel pour évaluer l’utilité downstream.
def make_real_train_val_split(real_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    y = real_df[TARGET_COL]

    for train_idx, val_idx in splitter.split(real_df, y):
        real_train = real_df.iloc[train_idx].reset_index(drop=True)
        real_val = real_df.iloc[val_idx].reset_index(drop=True)
        return real_train, real_val

    raise RuntimeError("Impossible de créer le split train/validation.")


# Sépare X et y en retirant la cible et l’identifiant si présent.
def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    drop_cols = [TARGET_COL]
    if ID_COL in df.columns:
        drop_cols.append(ID_COL)

    X = df.drop(columns=drop_cols).copy()
    y = df[TARGET_COL].astype(int).copy()
    return X, y


# Construit un préprocesseur simple pour gérer variables numériques et catégorielles.
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


# Entraîne un modèle downstream simple et retourne le QWK sur la validation réelle.
def compute_qwk_train_on_train_eval_on_val(train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    preprocessor = build_preprocessor(X_train)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_val)

    return float(cohen_kappa_score(y_val, preds, weights="quadratic"))


# Mesure à quel point un classifieur peut distinguer les lignes réelles des lignes synthétiques.
def compute_real_vs_syn_auc(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> float:
    real_tagged = real_df.copy()
    syn_tagged = syn_df.copy()

    real_tagged["_is_synth"] = 0
    syn_tagged["_is_synth"] = 1

    combined = pd.concat([real_tagged, syn_tagged], axis=0, ignore_index=True)

    drop_cols = ["_is_synth"]
    if TARGET_COL in combined.columns:
        drop_cols.append(TARGET_COL)
    if ID_COL in combined.columns:
        drop_cols.append(ID_COL)

    X = combined.drop(columns=drop_cols).copy()
    y = combined["_is_synth"].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    clf = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", clf),
        ]
    )

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    return float(roc_auc_score(y_test, proba))


# Calcule le Quality Report SDMetrics entre le réel et le synthétique.
def compute_quality_report(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> dict:
    metadata = build_metadata(real_df)

    report = QualityReport()
    report.generate(real_data=real_df, synthetic_data=syn_df, metadata=metadata, verbose=False)

    overall_score = float(report.get_score())
    properties = report.get_properties()

    result = {
        "quality_overall": overall_score,
        "quality_column_shapes": np.nan,
        "quality_column_pair_trends": np.nan,
    }

    if "Property" in properties.columns and "Score" in properties.columns:
        for _, row in properties.iterrows():
            prop = row["Property"]
            score = float(row["Score"])

            if prop == "Column Shapes":
                result["quality_column_shapes"] = score
            elif prop == "Column Pair Trends":
                result["quality_column_pair_trends"] = score

    return result

# Extrait automatiquement le nom du modèle et de la stratégie à partir du chemin du fichier synthétique.
def parse_model_and_strategy(syn_path: Path) -> tuple[str, str, str]:
    path_str = str(syn_path).replace("\\", "/").lower()

    if "ctgan/" in path_str:
        model_name = "ctgan"
    elif "tvae/" in path_str:
        model_name = "tvae"
    elif "tabsyn/" in path_str:
        model_name = "tabsyn"
    else:
        model_name = "unknown_model"

    if "synthetic_a/" in path_str or "strategy_a" in syn_path.stem.lower():
        strategy_name = "A"
    elif "synthetic_b/" in path_str or "strategy_b" in syn_path.stem.lower():
        strategy_name = "B"
    else:
        strategy_name = "unknown_strategy"

    dataset_name = f"{model_name}_strategy_{strategy_name}"
    return model_name, strategy_name, dataset_name

# Évalue un dataset synthétique unique avec plusieurs métriques et renvoie une ligne de résultats.
def evaluate_one_synthetic_dataset(
    real_train: pd.DataFrame,
    real_val: pd.DataFrame,
    baseline_qwk: float,
    syn_path: Path,
) -> dict:
    syn_df = pd.read_csv(syn_path)
    syn_df = align_synthetic_to_real(real_train, syn_df)
    model_name, strategy_name, dataset_name = parse_model_and_strategy(syn_path)

    quality_scores = compute_quality_report(real_train, syn_df)

    augmented_train = pd.concat([real_train, syn_df], axis=0, ignore_index=True)
    augmented_qwk = compute_qwk_train_on_train_eval_on_val(augmented_train, real_val)

    detection_auc = compute_real_vs_syn_auc(real_train, syn_df)

    result = {
    "dataset_name": dataset_name,
    "model_name": model_name,
    "strategy_name": strategy_name,
    "dataset_path": str(syn_path),
    "n_synthetic_rows": int(len(syn_df)),
    "baseline_qwk": baseline_qwk,
    "augmented_qwk": augmented_qwk,
    "delta_qwk": augmented_qwk - baseline_qwk,
    "real_vs_syn_auc": detection_auc,
}
    result.update(quality_scores)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Évalue plusieurs datasets synthétiques tabulaires.")
    parser.add_argument(
        "--real_path",
        type=str,
        default="data_clean/train_clean.csv",
        help="Chemin vers le dataset réel nettoyé",
    )
    parser.add_argument(
        "--synthetic_paths",
        type=str,
        nargs="+",
        required=True,
        help="Liste des fichiers CSV synthétiques à comparer",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="ctgan/outputs/evaluation_summary.csv",
        help="Chemin du tableau de sortie",
    )
    args = parser.parse_args()

    real_path = Path(args.real_path)
    synthetic_files = [Path(p) for p in args.synthetic_paths]
    output_csv = Path(args.output_csv)

    real_df = load_real_data(real_path)
    real_train, real_val = make_real_train_val_split(real_df)

    baseline_qwk = compute_qwk_train_on_train_eval_on_val(real_train, real_val)
    print(f"Baseline QWK (réel uniquement) : {baseline_qwk:.6f}")

    for path in synthetic_files:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    rows = []
    for syn_path in synthetic_files:
        print(f"Évaluation de {syn_path.name} ...")
        row = evaluate_one_synthetic_dataset(
            real_train=real_train,
            real_val=real_val,
            baseline_qwk=baseline_qwk,
            syn_path=syn_path,
        )
        rows.append(row)

    results_df = pd.DataFrame(rows).sort_values(
        by=["delta_qwk", "quality_overall"],
        ascending=[False, False],
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    print("\nRésumé :")
    print(results_df)
    print(f"\nTableau sauvegardé ici : {output_csv}")


if __name__ == "__main__":
    main()