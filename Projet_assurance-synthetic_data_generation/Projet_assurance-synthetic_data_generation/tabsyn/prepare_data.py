from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data_clean" / "train_clean.csv"

EXTERNAL_TABSYN_DIR = BASE_DIR / "external" / "tabsyn"
DATA_DIR = EXTERNAL_TABSYN_DIR / "data"
INFO_DIR = EXTERNAL_TABSYN_DIR / "data" / "Info"

TARGET_COL = "Response"
ID_COL = "Id"

CONTINUOUS_COLS = {
    "Id",
    "Product_Info_4",
    "Ins_Age",
    "Ht",
    "Wt",
    "BMI",
    "Employment_Info_1",
    "Employment_Info_4",
    "Employment_Info_6",
    "Insurance_History_5",
    "Family_Hist_2",
    "Family_Hist_3",
    "Family_Hist_4",
    "Family_Hist_5",
}
FORCE_CATEGORICAL_COLS = {
    "Product_Info_2",
}

# Charge le dataset nettoyé qui servira de base à la préparation TabSyn.
def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


# Garde uniquement les lignes d'une classe donnée.
def subset_one_class(df: pd.DataFrame, class_value: int) -> pd.DataFrame:
    return df[df[TARGET_COL] == class_value].copy().reset_index(drop=True)



# Détecte les indices de colonnes numériques et catégorielles pour le JSON TabSyn.
def get_column_indices(df: pd.DataFrame) -> tuple[list[int], list[int], list[int]]:
    num_col_idx = []
    cat_col_idx = []
    target_col_idx = []

    for i, col in enumerate(df.columns):
        if col == TARGET_COL:
            target_col_idx.append(i)
        elif col in FORCE_CATEGORICAL_COLS:
            cat_col_idx.append(i)
        else:
            num_col_idx.append(i)

    return num_col_idx, cat_col_idx, target_col_idx


# Construit le JSON de métadonnées attendu par TabSyn pour un dataset perso.
def build_info(dataset_name: str, df: pd.DataFrame) -> dict:
    num_col_idx, cat_col_idx, target_col_idx = get_column_indices(df)

    return {
        "name": dataset_name,
        "task_type": "regression",
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"data/{dataset_name}/{dataset_name}.csv",
        "test_path": None,
    }

# Sauvegarde le CSV et le JSON associés à une classe.
def save_dataset_and_info(dataset_name: str, df: pd.DataFrame, info: dict) -> None:
    dataset_dir = DATA_DIR / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    INFO_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(dataset_dir / f"{dataset_name}.csv", index=False)

    with open(INFO_DIR / f"{dataset_name}.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

def cast_categorical_columns_to_string(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    _, cat_col_idx, _ = get_column_indices(df)

    for i in cat_col_idx:
        col = df.columns[i]
        df[col] = df[col].astype(str)

    return df

def main() -> None:
    df = load_data()
    classes = sorted(df[TARGET_COL].unique())

    print(f"Classes trouvées : {classes}")

    for class_value in classes:
        dataset_name = f"prudential_class_{class_value}"

        df_class = subset_one_class(df, class_value)
        df_class = df_class.drop(columns=[ID_COL])
        df_class = cast_categorical_columns_to_string(df_class)

        info = build_info(dataset_name, df_class)
        save_dataset_and_info(dataset_name, df_class, info)

        print(f"{dataset_name} préparé | shape = {df_class.shape}")



if __name__ == "__main__":
    main()