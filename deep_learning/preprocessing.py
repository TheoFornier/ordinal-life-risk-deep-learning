from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Définition des types de colonnes (tirée du notebook de nettoyage)
# ---------------------------------------------------------------------------

CONTINUOUS_COLS = [
    "Product_Info_4", "Ins_Age", "Ht", "Wt", "BMI",
    "Employment_Info_1", "Employment_Info_4", "Employment_Info_6",
    "Insurance_History_5",
    "Family_Hist_2", "Family_Hist_3", "Family_Hist_4", "Family_Hist_5",
]

DISCRETE_COLS = [
    "Medical_History_1", "Medical_History_10", "Medical_History_15",
    "Medical_History_24", "Medical_History_32",
]

DUMMY_COLS = [f"Medical_Keyword_{i}" for i in range(1, 49)]

TARGET_COL = "Response"
ID_COL = "Id"


def _drop_sparse_columns(
    df: pd.DataFrame,
    train_mask: pd.Series,
    missing_threshold: float = 0.95,
    signal_threshold: float = 1.0,
) -> list[str]:
    # On retire les colonnes quasi-vides qui n'apportent aucun signal prédictif
    train_df = df[train_mask]
    n_train = len(train_df)
    cols_to_drop = []

    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL, "is_train")]
    for col in feature_cols:
        missing_rate = train_df[col].isna().sum() / n_train
        if missing_rate <= missing_threshold:
            continue

        non_null = train_df[col].notna()
        n_non_null = non_null.sum()

        if n_non_null < 10:
            cols_to_drop.append(col)
            continue

        mean_present = train_df.loc[non_null, TARGET_COL].mean()
        mean_absent = train_df.loc[~non_null, TARGET_COL].mean()
        if abs(mean_present - mean_absent) < signal_threshold:
            cols_to_drop.append(col)

    return cols_to_drop


def _target_encode_kfold(
    df: pd.DataFrame,
    col: str,
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> str:
    # Encodage cible par K-Fold pour éviter la fuite de données
    train_mask = df["is_train"] == 1
    test_mask = df["is_train"] == 0
    encoded_col = f"{col}_target_enc"
    df[encoded_col] = np.nan

    train_idx = df[train_mask].index
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_fold_idx, val_fold_idx in kf.split(train_idx):
        actual_train_idx = train_idx[train_fold_idx]
        actual_val_idx = train_idx[val_fold_idx]
        means = df.loc[actual_train_idx].groupby(col)[target].mean()
        df.loc[actual_val_idx, encoded_col] = df.loc[actual_val_idx, col].map(means)

    global_means = df[train_mask].groupby(col)[target].mean()
    global_mean = df.loc[train_mask, target].mean()
    df.loc[test_mask, encoded_col] = df.loc[test_mask, col].map(global_means)
    df[encoded_col].fillna(global_mean, inplace=True)

    return encoded_col


def clean(
    train_path: str,
    test_path: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(train_path)
    test_raw = pd.read_csv(test_path)

    # Fusion train/test pour des transformations cohérentes
    test_raw[TARGET_COL] = np.nan
    df = pd.concat(
        [train_raw.assign(is_train=1), test_raw.assign(is_train=0)],
        ignore_index=True,
    )

    train_mask = df["is_train"] == 1

    # 1. Suppression des colonnes creuses sans signal
    cols_to_drop = _drop_sparse_columns(df, train_mask)
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"{len(cols_to_drop)} sparse columns dropped.")

    # 2. Imputation des valeurs manquantes
    remaining = [c for c in df.columns if c not in (ID_COL, TARGET_COL, "is_train")]

    for col in remaining:
        if col in CONTINUOUS_COLS and col in df.columns:
            median = df.loc[train_mask, col].median()
            df[col].fillna(median, inplace=True)
        elif df[col].dtype == object:
            df[col].fillna("Missing", inplace=True)
        else:
            df[col].fillna(-1, inplace=True)

    # 3. Encodage label des colonnes textuelles
    text_cols = df[remaining].select_dtypes(include=["object"]).columns.tolist()
    for col in text_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Encodage cible pour les colonnes à forte cardinalité (> 20 valeurs uniques)
    remaining = [c for c in df.columns if c not in (ID_COL, TARGET_COL, "is_train")]
    high_card_cols = [col for col in remaining if df.loc[train_mask, col].nunique() > 20]
    for col in high_card_cols:
        _target_encode_kfold(df, col, TARGET_COL, random_state=random_state)
    df.drop(columns=high_card_cols, inplace=True)
    print(f"{len(high_card_cols)} high-cardinality columns target-encoded.")

    # 5. Feature engineering
    med_kw_cols = [c for c in DUMMY_COLS if c in df.columns]
    if med_kw_cols:
        df["Medical_Keyword_Count"] = df[med_kw_cols].sum(axis=1)

    if "BMI" in df.columns and "Ins_Age" in df.columns:
        df["BMI_Age"] = df["BMI"] * df["Ins_Age"]

    if "Wt" in df.columns and "Ht" in df.columns:
        df["Wt_Ht_ratio"] = df["Wt"] / (df["Ht"] + 1e-8)

    # 6. Séparation en train / test
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL, "is_train")]
    train_df = df[train_mask][feature_cols + [TARGET_COL]].reset_index(drop=True)
    test_df = df[~train_mask][feature_cols].reset_index(drop=True)
    test_ids = df[~train_mask][ID_COL].reset_index(drop=True)

    test_df.insert(0, ID_COL, test_ids.values)

    print(f"Clean done — train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


def load_data(
    train_raw: str,
    test_raw: str,
    train_clean: str,
    test_clean: str,
    use_cached: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if use_cached and os.path.exists(train_clean) and os.path.exists(test_clean):
        print("Loading from cache...")
        train_df = pd.read_csv(train_clean)
        test_df = pd.read_csv(test_clean)
        if ID_COL not in test_df.columns:
            raise ValueError(f"Le CSV de test en cache n'a pas la colonne '{ID_COL}'.")
    else:
        print("Running cleaning pipeline...")
        train_df, test_df = clean(train_raw, test_raw, random_state=random_state)
        train_df.to_csv(train_clean, index=False)
        test_df.to_csv(test_clean, index=False)
        print(f"Cleaned data saved to {train_clean}")

    feature_cols = [c for c in train_df.columns if c != TARGET_COL]
    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df[TARGET_COL].values.astype(np.float32)

    ids_test = test_df[ID_COL].values.astype(int)
    test_feature_cols = [c for c in test_df.columns if c != ID_COL]
    X_test = test_df[test_feature_cols].values.astype(np.float32)

    return X_train, y_train, X_test, ids_test
