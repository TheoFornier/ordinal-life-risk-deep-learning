from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


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
    missing_threshold: float = 0.95,
    signal_threshold: float = 1.0,
) -> list[str]:
    n = len(df)
    cols_to_drop = []
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    for col in feature_cols:
        missing_rate = df[col].isna().sum() / n
        if missing_rate <= missing_threshold:
            continue
        non_null = df[col].notna()
        if non_null.sum() < 10:
            cols_to_drop.append(col)
            continue
        mean_present = df.loc[non_null, TARGET_COL].mean()
        mean_absent = df.loc[~non_null, TARGET_COL].mean()
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
    encoded_col = f"{col}_target_enc"
    df[encoded_col] = np.nan
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    idx = df.index

    for train_fold_idx, val_fold_idx in kf.split(idx):
        actual_train = idx[train_fold_idx]
        actual_val = idx[val_fold_idx]
        means = df.loc[actual_train].groupby(col)[target].mean()
        df.loc[actual_val, encoded_col] = df.loc[actual_val, col].map(means)

    global_mean = df[target].mean()
    df[encoded_col].fillna(global_mean, inplace=True)
    return encoded_col


def clean(train_path: str, random_state: int = 42) -> pd.DataFrame:
    df = pd.read_csv(train_path).drop(columns=[ID_COL])

    # 1. Drop sparse columns with no predictive signal
    cols_to_drop = _drop_sparse_columns(df)
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"{len(cols_to_drop)} sparse columns dropped.")

    # 2. Impute missing values
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    for col in feature_cols:
        if col in CONTINUOUS_COLS and col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
        elif df[col].dtype == object:
            df[col].fillna("Missing", inplace=True)
        else:
            df[col].fillna(-1, inplace=True)

    # 3. Label-encode categorical columns
    text_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    for col in text_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 4. Target-encode high-cardinality columns (K-Fold to avoid leakage)
    feature_cols = [c for c in df.columns if c != TARGET_COL]
    high_card_cols = [col for col in feature_cols if df[col].nunique() > 20]
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

    print(f"Clean done — shape: {df.shape}")
    return df


def load_data(
    train_raw: str,
    train_clean: str,
    use_cached: bool = True,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if use_cached and os.path.exists(train_clean):
        print("Loading from cache...")
        df = pd.read_csv(train_clean)
    else:
        print("Running cleaning pipeline...")
        df = clean(train_raw, random_state=random_state)
        df.to_csv(train_clean, index=False)
        print(f"Cleaned data saved to {train_clean}")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y
