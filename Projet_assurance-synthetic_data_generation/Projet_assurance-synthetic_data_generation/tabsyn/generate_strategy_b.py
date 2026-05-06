from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
REAL_TRAIN_PATH = BASE_DIR / "data_clean" / "train_clean.csv"
EXTERNAL_TABSYN_DIR = BASE_DIR / "external" / "tabsyn"

OUTPUT_BASE_DIR = BASE_DIR / "tabsyn" / "outputs"
SYNTHETIC_B_DIR = OUTPUT_BASE_DIR / "synthetic_b"
AUGMENTED_DIR = OUTPUT_BASE_DIR / "augmented"

TARGET_COL = "Response"
ID_COL = "Id"


# Charge le dataset réel nettoyé qui sert de référence pour la structure finale.
def load_real_train() -> pd.DataFrame:
    df = pd.read_csv(REAL_TRAIN_PATH)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


# Calcule le nombre d'exemples par classe.
def get_class_counts(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL].value_counts().sort_index()


# Stratégie B : on remonte toutes les classes jusqu'à la classe majoritaire.
def get_strategy_b_plan(class_counts: pd.Series) -> dict[int, int]:
    max_size = int(class_counts.max())

    plan = {}
    for cls, count in class_counts.items():
        plan[int(cls)] = max(0, max_size - int(count))
    return plan


# Supprime et recrée proprement les dossiers de sortie pour éviter de garder d'anciens fichiers.
def reset_output_dirs() -> None:
    if SYNTHETIC_B_DIR.exists():
        shutil.rmtree(SYNTHETIC_B_DIR)
    SYNTHETIC_B_DIR.mkdir(parents=True, exist_ok=True)

    AUGMENTED_DIR.mkdir(parents=True, exist_ok=True)


# Lance le sampling TabSyn pour une classe donnée.
def run_tabsyn_sample(class_value: int) -> None:
    dataname = f"prudential_class_{class_value}"
    cmd = [
        sys.executable,
        "main.py",
        "--dataname",
        dataname,
        "--method",
        "tabsyn",
        "--mode",
        "sample",
    ]

    subprocess.run(cmd, cwd=EXTERNAL_TABSYN_DIR, check=True)


# Lit le CSV synthétique produit par TabSyn pour une classe.
def read_tabsyn_sample(class_value: int) -> pd.DataFrame:
    sample_path = (
        EXTERNAL_TABSYN_DIR
        / "synthetic"
        / f"prudential_class_{class_value}"
        / "tabsyn.csv"
    )

    if not sample_path.exists():
        raise FileNotFoundError(f"Fichier synthétique introuvable : {sample_path}")

    return pd.read_csv(sample_path)


# Génère suffisamment de lignes pour une classe en relançant le sampling autant de fois que nécessaire.
def sample_until_n(class_value: int, n_rows: int) -> pd.DataFrame:
    if n_rows <= 0:
        return pd.DataFrame()

    collected = []
    total_rows = 0

    while total_rows < n_rows:
        run_tabsyn_sample(class_value)
        batch = read_tabsyn_sample(class_value)
        collected.append(batch)
        total_rows += len(batch)

    synthetic_df = pd.concat(collected, axis=0, ignore_index=True).iloc[:n_rows].copy()
    return synthetic_df


# Ajoute la cible correspondant à la classe générée.
def add_response_column(df: pd.DataFrame, class_value: int) -> pd.DataFrame:
    df = df.copy()
    df[TARGET_COL] = class_value
    return df


# Ajoute de nouveaux Id pour éviter les doublons avec le dataset réel.
def assign_new_ids(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    synthetic_df = synthetic_df.copy()

    if ID_COL in real_df.columns:
        start_id = int(real_df[ID_COL].max()) + 1
        synthetic_df[ID_COL] = range(start_id, start_id + len(synthetic_df))

    return synthetic_df


# Réordonne les colonnes pour retrouver exactement la structure du dataset réel.
def reorder_like_real(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    missing_cols = [c for c in real_df.columns if c not in synthetic_df.columns]
    extra_cols = [c for c in synthetic_df.columns if c not in real_df.columns]

    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le synthétique : {missing_cols}")

    if extra_cols:
        synthetic_df = synthetic_df.drop(columns=extra_cols)

    return synthetic_df[real_df.columns].copy()


def main() -> None:
    reset_output_dirs()

    real_df = load_real_train()
    class_counts = get_class_counts(real_df)
    plan = get_strategy_b_plan(class_counts)

    print("Plan stratégie B :")
    for cls, n in plan.items():
        print(f"Classe {cls} -> {n} lignes à générer")

    synthetic_parts = []

    for class_value, n_rows in plan.items():
        if n_rows <= 0:
            continue

        print(f"\nSampling classe {class_value} pour {n_rows} lignes...")
        synthetic_df = sample_until_n(class_value, n_rows)
        synthetic_df = add_response_column(synthetic_df, class_value)
        synthetic_df = assign_new_ids(real_df, synthetic_df)
        synthetic_df = reorder_like_real(real_df, synthetic_df)

        out_path = SYNTHETIC_B_DIR / f"synthetic_class_{class_value}_B.csv"
        synthetic_df.to_csv(out_path, index=False)
        print(f"Sauvegardé : {out_path}")

        synthetic_parts.append(synthetic_df)

    if synthetic_parts:
        synthetic_all = pd.concat(synthetic_parts, axis=0, ignore_index=True)
    else:
        synthetic_all = pd.DataFrame(columns=real_df.columns)

    synthetic_all_path = SYNTHETIC_B_DIR / "synthetic_all_strategy_b.csv"
    synthetic_all.to_csv(synthetic_all_path, index=False)

    augmented_df = pd.concat([real_df, synthetic_all], axis=0, ignore_index=True)
    augmented_path = AUGMENTED_DIR / "train_augmented_strategy_b.csv"
    augmented_df.to_csv(augmented_path, index=False)

    print(f"\nDonnées synthétiques B : {synthetic_all_path}")
    print(f"Dataset augmenté B : {augmented_path}")
    print(f"Taille finale : {augmented_df.shape}")


if __name__ == "__main__":
    main()