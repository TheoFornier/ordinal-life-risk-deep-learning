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
SYNTHETIC_C_DIR = OUTPUT_BASE_DIR / "synthetic_c"

TARGET_COL = "Response"
ID_COL = "Id"


# Charge le dataset réel nettoyé de référence.
def load_real_train() -> pd.DataFrame:
    df = pd.read_csv(REAL_TRAIN_PATH)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Le fichier réel doit contenir la colonne cible '{TARGET_COL}'.")
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


# Calcule le nombre d'exemples par classe dans le vrai train.
def get_class_counts(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL].value_counts().sort_index()


# Stratégie C :
# on génère pour chaque classe jusqu'à atteindre la taille de la classe majoritaire.
# Cela donne un gros ensemble plus équilibré et varié, utile avant pseudo-labellisation.
def get_strategy_c_plan(class_counts: pd.Series, target_ratio: float = 1.0) -> dict[int, int]:
    max_size = int(class_counts.max())
    target_size = int(max_size * target_ratio)

    plan = {}
    for cls, count in class_counts.items():
        # ici on génère target_size lignes par classe, pas seulement le complément
        plan[int(cls)] = target_size
    return plan


# Supprime et recrée proprement le dossier de sortie.
def reset_output_dirs() -> None:
    if SYNTHETIC_C_DIR.exists():
        shutil.rmtree(SYNTHETIC_C_DIR)
    SYNTHETIC_C_DIR.mkdir(parents=True, exist_ok=True)


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


# Relance le sampling autant de fois que nécessaire pour atteindre n_rows.
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
        print(f"Classe {class_value} : {total_rows}/{n_rows} lignes collectées")

    synthetic_df = pd.concat(collected, axis=0, ignore_index=True).iloc[:n_rows].copy()
    return synthetic_df


# Ajoute temporairement la classe source générée.
# Cette colonne est utile pour debug / traçabilité, mais ne sera pas gardée dans le fichier final X-only.
def add_source_class_column(df: pd.DataFrame, class_value: int) -> pd.DataFrame:
    df = df.copy()
    df["source_class"] = class_value
    return df


# Supprime Id, Response, et toute colonne de debug éventuelle.
def keep_features_only(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    real_feature_cols = [c for c in real_df.columns if c not in (ID_COL, TARGET_COL)]

    synthetic_df = synthetic_df.copy()

    # On retire explicitement les colonnes qui ne doivent pas partir au teacher.
    cols_to_drop = [c for c in [ID_COL, TARGET_COL, "source_class"] if c in synthetic_df.columns]
    if cols_to_drop:
        synthetic_df = synthetic_df.drop(columns=cols_to_drop)

    missing_cols = [c for c in real_feature_cols if c not in synthetic_df.columns]
    extra_cols = [c for c in synthetic_df.columns if c not in real_feature_cols]

    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans le synthétique : {missing_cols}")

    if extra_cols:
        synthetic_df = synthetic_df.drop(columns=extra_cols)

    return synthetic_df[real_feature_cols].copy()


def main() -> None:
    reset_output_dirs()

    real_df = load_real_train()
    class_counts = get_class_counts(real_df)

    # target_ratio=1.0 : taille de la classe majoritaire pour chaque classe
    # Tu peux augmenter à 1.5 ou 2.0 si tu veux encore plus de volume.
    plan = get_strategy_c_plan(class_counts, target_ratio=1.0)

    print("Plan stratégie C :")
    for cls, n in plan.items():
        print(f"Classe {cls} -> {n} lignes à générer")

    synthetic_parts = []

    for class_value, n_rows in plan.items():
        if n_rows <= 0:
            continue

        print(f"\nSampling classe {class_value} pour {n_rows} lignes...")
        synthetic_df = sample_until_n(class_value, n_rows)
        synthetic_df = add_source_class_column(synthetic_df, class_value)

        out_debug_path = SYNTHETIC_C_DIR / f"synthetic_class_{class_value}_C_debug.csv"
        synthetic_df.to_csv(out_debug_path, index=False)
        print(f"Sauvegardé (debug) : {out_debug_path}")

        synthetic_parts.append(synthetic_df)

    if synthetic_parts:
        synthetic_all = pd.concat(synthetic_parts, axis=0, ignore_index=True)
    else:
        synthetic_all = pd.DataFrame()

    # Fichier final pour teacher-student : features uniquement
    synthetic_x_only = keep_features_only(real_df, synthetic_all)

    synthetic_all_path = SYNTHETIC_C_DIR / "synthetic_all_strategy_c_xonly.csv"
    synthetic_x_only.to_csv(synthetic_all_path, index=False)

    print(f"\nDonnées synthétiques C (X-only) : {synthetic_all_path}")
    print(f"Taille finale : {synthetic_x_only.shape}")
    print(f"Colonnes finales : {len(synthetic_x_only.columns)}")
    print("\nÉtape suivante : utiliser ce fichier dans pseudo_label.py")


if __name__ == "__main__":
    main()