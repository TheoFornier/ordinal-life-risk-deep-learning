from __future__ import annotations

from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_CLEAN_PATH = BASE_DIR / "data_clean" / "train_clean.csv"

OUTPUTS_DIR = BASE_DIR / "ctgan" / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
SYNTHETIC_A_DIR = OUTPUTS_DIR / "synthetic_a"
SYNTHETIC_B_DIR = OUTPUTS_DIR / "synthetic_b"
AUGMENTED_DIR = OUTPUTS_DIR / "augmented"

TARGET_COL = "Response"
ID_COL = "Id"

# Crée automatiquement tous les dossiers de sortie nécessaires pour stocker les modèles, les données synthétiques et les datasets augmentés.
def ensure_directories() -> None:
    for path in [OUTPUTS_DIR, MODELS_DIR, SYNTHETIC_A_DIR, SYNTHETIC_B_DIR, AUGMENTED_DIR]:
        path.mkdir(parents=True, exist_ok=True)

# Charge le fichier train_clean.csv pour récupérer le dataset nettoyé qui servira à l'entraînement du générateur.
def load_train_clean() -> pd.DataFrame:
    return pd.read_csv(DATA_CLEAN_PATH)

# Calcule le nombre d'exemples présents dans chaque classe de Response pour connaître la répartition actuelle du dataset.
def get_class_counts(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COL].value_counts().sort_index()

# Construit le plan de génération de la stratégie A en ajoutant des données seulement pour les classes jugées rares.
def get_strategy_a_plan(class_counts: pd.Series, rare_ratio: float = 0.5) -> dict[int, int]:
    max_size = int(class_counts.max())
    target_floor = int(max_size * rare_ratio)

    plan = {}
    for cls, count in class_counts.items():
        plan[int(cls)] = max(0, target_floor - int(count))
    return plan

# Construit le plan de génération de la stratégie B en ajoutant assez de données pour ramener toutes les classes au niveau de la classe majoritaire.
def get_strategy_b_plan(class_counts: pd.Series) -> dict[int, int]:
    max_size = int(class_counts.max())

    plan = {}
    for cls, count in class_counts.items():
        plan[int(cls)] = max(0, int(max_size) - int(count))
    return plan

# Identifie les colonnes discrètes ou catégorielles du dataset pour les déclarer correctement lors de l'entraînement du modèle CTGAN.
def get_discrete_columns(df: pd.DataFrame) -> list[str]:
    discrete_cols = []

    for col in df.columns:
        if col == ID_COL:
            continue

        if pd.api.types.is_object_dtype(df[col]):
            discrete_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() <= 100:
            discrete_cols.append(col)

    if TARGET_COL in df.columns and TARGET_COL not in discrete_cols:
        discrete_cols.append(TARGET_COL)

    return sorted(set(discrete_cols))