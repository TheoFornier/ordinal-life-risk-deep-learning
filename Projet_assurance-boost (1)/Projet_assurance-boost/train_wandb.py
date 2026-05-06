"""
GLO-7030 — Entraînement Boosting avec Weights & Biases (Méthode Gábor)
Script principal implémentant la stratégie gagnante de Gábor (7 classifieurs binaires, Variable Split Calibration).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import wandb
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

NUM_CLASSES = 8
DATA_DIR = "prudential-life-insurance-assessment"
TARGET = "Response"

def qwk(y_pred, y_true):
    y_true = np.array(y_true).astype(int)
    y_pred = np.clip(np.round(np.array(y_pred)), 1, 8).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")

def apply_boundaries(preds, boundaries):
    res = np.ones_like(preds, dtype=int)
    for b in boundaries:
        res += (preds > b).astype(int)
    return res

def optimize_split_boundaries(preds, y_true, split_var):
    mask = (split_var == 1)
    
    b1 = [np.percentile(preds[mask], np.mean(y_true[mask] <= i)*100) if np.sum(mask)>0 else 0 for i in range(1, 8)]
    b2 = [np.percentile(preds[~mask], np.mean(y_true[~mask] <= i)*100) if np.sum(~mask)>0 else 0 for i in range(1, 8)]
    
    def eval_kappa(b1, b2):
        final_preds = np.zeros_like(preds, dtype=int)
        if np.sum(mask) > 0: final_preds[mask] = apply_boundaries(preds[mask], b1)
        if np.sum(~mask) > 0: final_preds[~mask] = apply_boundaries(preds[~mask], b2)
        return cohen_kappa_score(y_true, final_preds, weights='quadratic')
        
    best_kappa = eval_kappa(b1, b2)
    
    changed = True
    while changed:
        changed = False
        for i in range(7):
            best_b_val = b1[i]
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                temp_b = b1.copy()
                temp_b[i] += delta
                if i > 0 and temp_b[i] <= temp_b[i-1]: continue
                if i < 6 and temp_b[i] >= temp_b[i+1]: continue
                k = eval_kappa(temp_b, b2)
                if k > best_kappa:
                    best_kappa = k
                    best_b_val = temp_b[i]
                    changed = True
            b1[i] = best_b_val
            
        for i in range(7):
            best_b_val = b2[i]
            for delta in [-0.05, -0.02, -0.01, 0.01, 0.02, 0.05]:
                temp_b = b2.copy()
                temp_b[i] += delta
                if i > 0 and temp_b[i] <= temp_b[i-1]: continue
                if i < 6 and temp_b[i] >= temp_b[i+1]: continue
                k = eval_kappa(b1, temp_b)
                if k > best_kappa:
                    best_kappa = k
                    best_b_val = temp_b[i]
                    changed = True
            b2[i] = best_b_val
            
    return b1, b2

def apply_split_boundaries_full(preds, split_var, b1, b2):
    mask = (split_var == 1)
    final_preds = np.zeros_like(preds, dtype=int)
    if np.sum(mask) > 0: final_preds[mask] = apply_boundaries(preds[mask], b1)
    if np.sum(~mask) > 0: final_preds[~mask] = apply_boundaries(preds[~mask], b2)
    return final_preds

# Keep original functions for compatibility with sweep_wandb.py imports
def optimize_offsets(*args, **kwargs): pass
def apply_offsets(*args, **kwargs): pass

def get_keyword_features_vectorized(X_df, y_bin, kw_cols, X_val_df=None):
    kw_means = []
    for col in kw_cols:
        mask = (X_df[col] == 1)
        kw_means.append(y_bin[mask].mean() if mask.sum() > 0 else 0)
    kw_means = np.array(kw_means)
    
    def apply_kw(df):
        mat = df[kw_cols].values
        masked_mat = np.where(mat == 1, kw_means, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_means = np.nanmean(masked_mat, axis=1)
            row_mins = np.nanmin(masked_mat, axis=1)
        row_means = np.nan_to_num(row_means, nan=0.0)
        row_mins = np.nan_to_num(row_mins, nan=0.0)
        
        df_new = df.copy()
        df_new['kw_mean'] = row_means
        df_new['kw_min'] = row_mins
        return df_new

    if X_val_df is not None:
        return apply_kw(X_df), apply_kw(X_val_df)
    return apply_kw(X_df)

def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_clean.csv"))

    # Nettoyage de sécurité : retirer toute observation où la cible (Response) est NaN
    # (Cela prévient l'erreur 'Input y contains NaN' lors du train_test_split)
    train = train.dropna(subset=[TARGET])

    if 'Product_Info_2' in train.columns:
        train = pd.get_dummies(train, columns=['Product_Info_2'])
        test = pd.get_dummies(test, columns=['Product_Info_2'])
        train, test = train.align(test, join='left', axis=1, fill_value=0)
        
    kw_cols = [c for c in train.columns if c.startswith('Medical_Keyword_')]
    train = train.copy()
    test = test.copy()
    train['Medical_Keyword_Sum'] = train[kw_cols].sum(axis=1)
    test['Medical_Keyword_Sum'] = test[kw_cols].sum(axis=1)

    feature_cols = [c for c in train.columns if c not in ["Id", TARGET]]
    
    X = train[feature_cols]
    y = train[TARGET]
    X_test = test[feature_cols]
    test_ids = test["Id"]

    return X, y, X_test, test_ids, feature_cols

def train_xgboost(config, X_train, y_train, X_val, y_val):
    preds_train = np.zeros(X_train.shape[0])
    preds_val = np.zeros(X_val.shape[0])
    
    kw_cols = [c for c in X_train.columns if c.startswith('Medical_Keyword_')]
    
    xgb_params = {
        "objective": "binary:logistic",
        "eta": config.get("learning_rate", 0.05),
        "min_child_weight": config.get("min_child_weight", 1),
        "subsample": config.get("subsample", 0.8),
        "colsample_bytree": config.get("colsample_bytree", 0.8),
        "max_depth": config.get("max_depth", 5),
        "verbosity": 0,
    }
    n_rounds = config.get("n_estimators", 150)

    for i in range(1, 8):
        y_train_bin = (y_train > i).astype(int)
        y_val_bin = (y_val > i).astype(int)
        
        X_tr_kw, X_val_kw = get_keyword_features_vectorized(X_train, y_train_bin, kw_cols, X_val)
        
        dtrain = xgb.DMatrix(X_tr_kw, label=y_train_bin)
        dval = xgb.DMatrix(X_val_kw, label=y_val_bin)
        
        model = xgb.train(xgb_params, dtrain, n_rounds, evals=[(dval, "val")], verbose_eval=False)
        
        preds_train += model.predict(dtrain)
        preds_val += model.predict(dval)
        
    preds_train += 1
    preds_val += 1
    
    split_var_train = X_train['Medical_History_23'].values
    split_var_val = X_val['Medical_History_23'].values
    
    preds_train = np.nan_to_num(preds_train, nan=1.0)
    preds_val = np.nan_to_num(preds_val, nan=1.0)
    
    b1, b2 = optimize_split_boundaries(preds_train, y_train.values, split_var_train)
    val_offset = apply_split_boundaries_full(preds_val, split_var_val, b1, b2)
    score_offset = cohen_kappa_score(y_val, val_offset, weights='quadratic')
    
    score_simple = cohen_kappa_score(y_val, np.clip(np.round(preds_val), 1, 8).astype(int), weights='quadratic')
    
    return None, score_simple, score_offset, (b1, b2), preds_val, val_offset

def train_catboost(config, X_train, y_train, X_val, y_val):
    preds_train = np.zeros(X_train.shape[0])
    preds_val = np.zeros(X_val.shape[0])
    
    kw_cols = [c for c in X_train.columns if c.startswith('Medical_Keyword_')]
    
    for i in range(1, 8):
        y_train_bin = (y_train > i).astype(int)
        y_val_bin = (y_val > i).astype(int)
        
        X_tr_kw, X_val_kw = get_keyword_features_vectorized(X_train, y_train_bin, kw_cols, X_val)
        
        model = CatBoostClassifier(
            iterations=config.get("n_estimators", 150),
            depth=config.get("max_depth", 5),
            learning_rate=config.get("learning_rate", 0.05),
            l2_leaf_reg=config.get("l2_leaf_reg", 3.0),
            random_strength=config.get("random_strength", 1.0),
            loss_function="Logloss",
            random_seed=42,
            verbose=0,
            task_type="CPU",
        )
        model.fit(X_tr_kw, y_train_bin, eval_set=(X_val_kw, y_val_bin), early_stopping_rounds=50)
        
        preds_train += model.predict_proba(X_tr_kw)[:, 1]
        preds_val += model.predict_proba(X_val_kw)[:, 1]
        
    preds_train += 1
    preds_val += 1
    
    split_var_train = X_train['Medical_History_23'].values
    split_var_val = X_val['Medical_History_23'].values
    
    preds_train = np.nan_to_num(preds_train, nan=1.0)
    preds_val = np.nan_to_num(preds_val, nan=1.0)
    
    b1, b2 = optimize_split_boundaries(preds_train, y_train.values, split_var_train)
    val_offset = apply_split_boundaries_full(preds_val, split_var_val, b1, b2)
    score_offset = cohen_kappa_score(y_val, val_offset, weights='quadratic')
    
    score_simple = cohen_kappa_score(y_val, np.clip(np.round(preds_val), 1, 8).astype(int), weights='quadratic')
    
    return None, score_simple, score_offset, (b1, b2), preds_val, val_offset

def main():
    config = {
        "model_type": "CatBoost",
        "learning_rate": 0.04716813821393108,
        "max_depth": 5,
        "n_estimators": 1200,
        "min_child_weight": 200,
        "subsample": 0.6037648584029301,
        "colsample_bytree": 0.5331073139717044,
        "use_offsets": True,
        "clean_method": "v1_impute_median_target_enc",
        "test_size": 0.2,
    }

    wandb.init(
        project="GLO7030-Prudential",
        config=config,
        name=f"Gabor_{config['model_type']}_lr{config['learning_rate']}_d{config['max_depth']}",
    )
    config = wandb.config

    print("=" * 60)
    print(f"ENTRAÎNEMENT (MÉTHODE GÁBOR) : {config['model_type']}")
    print("=" * 60)

    X, y, X_test, test_ids, feature_cols = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.get("test_size", 0.2), random_state=42, stratify=y
    )

    if config["model_type"] == "XGBoost":
        _, score_simple, score_offset, boundaries, val_preds, val_offset = train_xgboost(
            config, X_train, y_train, X_val, y_val
        )
    else:
        _, score_simple, score_offset, boundaries, val_preds, val_offset = train_catboost(
            config, X_train, y_train, X_val, y_val
        )

    best_score = score_offset if config["use_offsets"] else score_simple
    wandb.log({
        "val_qwk_simple": score_simple,
        "val_qwk_offset": score_offset,
        "qwk_gain_offset": score_offset - score_simple,
        "best_qwk": best_score,
    })

    wandb.summary["val_qwk_simple"] = score_simple
    wandb.summary["val_qwk_offset"] = score_offset
    wandb.summary["best_qwk"] = best_score

    print(f"  QWK arrondi simple : {score_simple:.4f}")
    print(f"  QWK avec offsets Gábor : {score_offset:.4f} (gain: +{score_offset - score_simple:.4f})")

    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=[int(p) for p in val_offset], 
            y_true=[int(y) for y in y_val.values]
        )
    })

    plt.figure(figsize=(10, 5))
    sns.kdeplot(val_preds, fill=True, color="blue", alpha=0.3, label="Distribution Gábor (1 à 8)")
    plt.title("Distribution des prédictions brutes Gábor")
    wandb.log({"Prediction_Distribution": wandb.Image(plt)})
    plt.close()

    print("\nGénération de la soumission complète...")
    
    preds_full = np.zeros(X.shape[0])
    preds_test = np.zeros(X_test.shape[0])
    kw_cols = [c for c in X.columns if c.startswith('Medical_Keyword_')]
    
    if config["model_type"] == "XGBoost":
        xgb_params = {
            "objective": "binary:logistic", "eta": config["learning_rate"],
            "min_child_weight": config.get("min_child_weight", 1),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "max_depth": config["max_depth"], "verbosity": 0,
        }
        for i in range(1, 8):
            y_bin = (y > i).astype(int)
            X_tr_kw, X_te_kw = get_keyword_features_vectorized(X, y_bin, kw_cols, X_test)
            dtrain = xgb.DMatrix(X_tr_kw, label=y_bin)
            dtest = xgb.DMatrix(X_te_kw)
            model = xgb.train(xgb_params, dtrain, config.get("n_estimators", 150), verbose_eval=False)
            preds_full += model.predict(dtrain)
            preds_test += model.predict(dtest)
    else:
        for i in range(1, 8):
            y_bin = (y > i).astype(int)
            X_tr_kw, X_te_kw = get_keyword_features_vectorized(X, y_bin, kw_cols, X_test)
            model = CatBoostClassifier(
                iterations=config.get("n_estimators", 150), depth=config["max_depth"],
                learning_rate=config["learning_rate"],
                l2_leaf_reg=config.get("l2_leaf_reg", 3.0),
                random_strength=config.get("random_strength", 1.0),
                loss_function="Logloss",
                random_seed=42, verbose=0, task_type="CPU",
            )
            model.fit(X_tr_kw, y_bin)
            preds_full += model.predict_proba(X_tr_kw)[:, 1]
            preds_test += model.predict_proba(X_te_kw)[:, 1]

    preds_full += 1
    preds_test += 1
    preds_full = np.nan_to_num(preds_full, nan=1.0)
    preds_test = np.nan_to_num(preds_test, nan=1.0)
    
    split_var_full = X['Medical_History_23'].values
    b1, b2 = optimize_split_boundaries(preds_full, y.values, split_var_full)
    
    split_var_test = X_test['Medical_History_23'].values
    final_predictions = apply_split_boundaries_full(preds_test, split_var_test, b1, b2)

    submission = pd.DataFrame({"Id": test_ids, "Response": final_predictions})
    submission_path = os.path.join(DATA_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Soumission sauvegardée : {submission_path}")

    artifact = wandb.Artifact("submission", type="predictions")
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    main()
