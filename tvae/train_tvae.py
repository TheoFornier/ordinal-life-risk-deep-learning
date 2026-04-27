from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer

from utils import (
    MODELS_DIR,
    TARGET_COL,
    ID_COL,
    ensure_directories,
    load_train_clean,
)


# Charge uniquement les lignes correspondant à une classe donnée de Response.
def subset_class(df: pd.DataFrame, class_value: int) -> pd.DataFrame:
    return df[df[TARGET_COL] == class_value].copy()


# Supprime la colonne Id avant l'entraînement car elle identifie les lignes sans apporter de structure utile au générateur.
def drop_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if ID_COL in df.columns:
        return df.drop(columns=[ID_COL]).copy()
    return df.copy()


# Construit automatiquement le metadata SDV à partir du DataFrame d'entraînement.
def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    return metadata


# Entraîne un modèle TVAE sur les données d'une seule classe.
def train_tvae_for_class(
    df_class: pd.DataFrame,
    epochs: int = 300,
    batch_size: int = 500,
    verbose: bool = True,
) -> TVAESynthesizer:
    metadata = build_metadata(df_class)

    synthesizer = TVAESynthesizer(
        metadata=metadata,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )

    synthesizer.fit(df_class)
    return synthesizer


# Sauvegarde le modèle entraîné dans outputs/models pour pouvoir le recharger ensuite pendant la génération.
def save_model(synthesizer: TVAESynthesizer, class_value: int) -> Path:
    model_path = MODELS_DIR / f"tvae_class_{class_value}.pkl"
    synthesizer.save(filepath=str(model_path))
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne un TVAE pour une classe donnée.")
    parser.add_argument("--class_value", type=int, required=True, help="Classe Response à entraîner")
    parser.add_argument("--epochs", type=int, default=300, help="Nombre d'epochs pour TVAE")
    parser.add_argument("--batch_size", type=int, default=500, help="Batch size pour TVAE")
    args = parser.parse_args()

    ensure_directories()

    df = load_train_clean()
    df_class = subset_class(df, args.class_value)
    df_class = drop_id_column(df_class)

    if df_class.empty:
        raise ValueError(f"Aucune ligne trouvée pour la classe {args.class_value}.")

    print(f"Classe {args.class_value} : {len(df_class)} lignes")
    print(f"Dimensions utilisées pour l'entraînement : {df_class.shape}")

    synthesizer = train_tvae_for_class(
        df_class=df_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=True,
    )

    model_path = save_model(synthesizer, args.class_value)
    print(f"Modèle sauvegardé ici : {model_path}")


if __name__ == "__main__":
    main()