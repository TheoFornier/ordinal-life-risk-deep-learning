from __future__ import annotations

from pathlib import Path

import pandas as pd
from sdv.single_table import CTGANSynthesizer

from utils import (
    AUGMENTED_DIR,
    MODELS_DIR,
    SYNTHETIC_A_DIR,
    TARGET_COL,
    ID_COL,
    ensure_directories,
    load_train_clean,
    get_class_counts,
    get_strategy_a_plan,
)


# Charge un modèle CTGAN déjà entraîné pour une classe donnée.
def load_model_for_class(class_value: int) -> CTGANSynthesizer:
    model_path = MODELS_DIR / f"ctgan_class_{class_value}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Modèle introuvable pour la classe {class_value} : {model_path}")
    return CTGANSynthesizer.load(filepath=str(model_path))


# Génère le nombre demandé de lignes synthétiques pour une classe donnée.
def generate_samples_for_class(class_value: int, num_rows: int) -> pd.DataFrame:
    if num_rows <= 0:
        return pd.DataFrame()

    synthesizer = load_model_for_class(class_value)
    synthetic_df = synthesizer.sample(num_rows=num_rows)

    synthetic_df[TARGET_COL] = class_value
    return synthetic_df


# Ajoute de nouveaux identifiants aux lignes synthétiques pour éviter les doublons avec le dataset réel.
def assign_new_ids(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    if synthetic_df.empty:
        return synthetic_df

    if ID_COL not in real_df.columns:
        return synthetic_df

    synthetic_df = synthetic_df.copy()
    start_id = int(real_df[ID_COL].max()) + 1
    synthetic_df[ID_COL] = range(start_id, start_id + len(synthetic_df))
    return synthetic_df


# Réordonne les colonnes synthétiques pour qu'elles aient exactement le même ordre que le dataset réel.
def reorder_like_real(real_df: pd.DataFrame, synthetic_df: pd.DataFrame) -> pd.DataFrame:
    return synthetic_df[real_df.columns]


def main() -> None:
    ensure_directories()

    df_real = load_train_clean()
    class_counts = get_class_counts(df_real)
    plan_a = get_strategy_a_plan(class_counts, rare_ratio=0.5)

    print("Plan de génération stratégie A :")
    for cls, n in plan_a.items():
        print(f"Classe {cls} -> {n} lignes à générer")

    synthetic_parts = []

    for class_value, num_rows in plan_a.items():
        if num_rows <= 0:
            continue

        print(f"\nGénération pour la classe {class_value} ({num_rows} lignes)...")
        synthetic_df = generate_samples_for_class(class_value, num_rows)
        synthetic_df = assign_new_ids(df_real, synthetic_df)
        synthetic_df = reorder_like_real(df_real, synthetic_df)

        output_class_path = SYNTHETIC_A_DIR / f"synthetic_class_{class_value}_A.csv"
        synthetic_df.to_csv(output_class_path, index=False)
        print(f"Sauvegardé : {output_class_path}")

        synthetic_parts.append(synthetic_df)

    if synthetic_parts:
        df_synthetic_all = pd.concat(synthetic_parts, axis=0, ignore_index=True)
    else:
        df_synthetic_all = pd.DataFrame(columns=df_real.columns)

    synthetic_all_path = SYNTHETIC_A_DIR / "synthetic_ctgan_strategy_a.csv"
    df_synthetic_all.to_csv(synthetic_all_path, index=False)

    df_augmented = pd.concat([df_real, df_synthetic_all], axis=0, ignore_index=True)
    augmented_path = AUGMENTED_DIR / "train_augmented_ctgan_strategy_a.csv"
    df_augmented.to_csv(augmented_path, index=False)

    print(f"\nDonnées synthétiques A sauvegardées : {synthetic_all_path}")
    print(f"Dataset augmenté A sauvegardé : {augmented_path}")
    print(f"Taille finale dataset augmenté A : {df_augmented.shape}")


if __name__ == "__main__":
    main()