"""
GLO-7030 — Entraînement Boosting avec Weights & Biases
Script principal pour XGBoost et CatBoost avec logging W&B.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
import wandb
import warnings
import os
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# CONSTANTES
# ============================================================
NUM_CLASSES = 8
DATA_DIR = "prudential-life-insurance-assessment"
TARGET = "Response"


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def qwk(y_pred, y_true):
    """Quadratic Weighted Kappa."""
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred)
    y_pred = np.clip(np.round(y_pred), np.min(y_true), np.max(y_true)).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def optimize_offsets(preds, labels):
    """Optimise un offset par classe pour maximiser le QWK."""
    preds = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)
    adjusted = preds.copy()
    offsets = np.zeros(NUM_CLASSES)

    for j in [6, 4, 5, 3, 2, 1, 7, 0]:
        def objective(x, cls=j):
            adjusted[preds.astype(int) == cls] = preds[preds.astype(int) == cls] + x
            return -qwk(adjusted, labels)

        result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
        offsets[j] = result.x
        adjusted[preds.astype(int) == j] = preds[preds.astype(int) == j] + offsets[j]

    return offsets


def apply_offsets(preds, offsets):
    """Applique les offsets et retourne les predictions finales (1-8)."""
    preds = np.array(preds, dtype=float)
    adjusted = preds.copy()
    for j in range(NUM_CLASSES):
        mask = preds.astype(int) == j
        adjusted[mask] = preds[mask] + offsets[j]
    return np.clip(np.round(adjusted), 1, 8).astype(int)


# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train_clean.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_clean.csv"))

    feature_cols = [c for c in train.columns if c not in ["Id", TARGET]]
    X = train[feature_cols].values
    y = train[TARGET].values
    X_test = test[feature_cols].values
    test_ids = test["Id"].values

    return X, y, X_test, test_ids, feature_cols


# ============================================================
# ENTRAÎNEMENT XGBOOST
# ============================================================
def train_xgboost(config, X_train, y_train, X_val, y_val):
    """Entraîne XGBoost et logge les métriques sur W&B."""
    xgb_params = {
        "objective": "reg:squarederror",
        "eta": config["learning_rate"],
        "min_child_weight": config["min_child_weight"],
        "subsample": config["subsample"],
        "colsample_bytree": config["colsample_bytree"],
        "max_depth": config["max_depth"],
        "verbosity": 0,
    }
    n_rounds = config["n_estimators"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Callback pour logger chaque 10 rounds sur W&B
    evals_result = {}
    model = xgb.train(
        xgb_params,
        dtrain,
        n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        evals_result=evals_result,
        verbose_eval=False,
    )

    # Logger la progression sur W&B
    for i in range(n_rounds):
        log_data = {
            "round": i + 1,
            "train_rmse": evals_result["train"]["rmse"][i],
            "val_rmse": evals_result["val"]["rmse"][i],
        }
        # Calculer le QWK toutes les 50 rounds
        if (i + 1) % 50 == 0 or i == n_rounds - 1:
            dtrain_tmp = xgb.DMatrix(X_train)
            dval_tmp = xgb.DMatrix(X_val)
            model_tmp = xgb.train(xgb_params, dtrain, i + 1, verbose_eval=False)
            val_preds = model_tmp.predict(dval_tmp)
            train_preds = model_tmp.predict(dtrain_tmp)
            log_data["val_qwk_simple"] = qwk(val_preds, y_val)
            log_data["train_qwk_simple"] = qwk(train_preds, y_train)
            print(f"  Round {i+1}/{n_rounds} | val_rmse={log_data['val_rmse']:.4f} | val_qwk={log_data['val_qwk_simple']:.4f}")

        wandb.log(log_data)

    # Scores finaux
    val_preds = model.predict(dval)
    train_preds = model.predict(dtrain)

    score_simple = qwk(val_preds, y_val)

    offsets = optimize_offsets(train_preds, y_train)
    val_offset = apply_offsets(val_preds, offsets)
    score_offset = qwk(val_offset, y_val)

    return model, score_simple, score_offset, offsets, val_preds, val_offset


# ============================================================
# ENTRAÎNEMENT CATBOOST
# ============================================================
def train_catboost(config, X_train, y_train, X_val, y_val):
    """Entraîne CatBoost et logge les métriques sur W&B."""
    model = CatBoostRegressor(
        iterations=config["n_estimators"],
        depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        loss_function="RMSE",
        random_seed=42,
        verbose=0,
        early_stopping_rounds=50,
        task_type="CPU",
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val))

    # Logger les métriques depuis l'historique CatBoost
    best_iteration = model.get_best_iteration() if model.get_best_iteration() else config["n_estimators"]
    wandb.log({"catboost_best_iteration": best_iteration})

    # Scores finaux
    val_preds = model.predict(X_val)
    train_preds = model.predict(X_train)

    score_simple = qwk(val_preds, y_val)

    offsets = optimize_offsets(train_preds, y_train)
    val_offset = apply_offsets(val_preds, offsets)
    score_offset = qwk(val_offset, y_val)

    return model, score_simple, score_offset, offsets, val_preds, val_offset


# ============================================================
# FONCTION PRINCIPALE
# ============================================================
def main():
    # --- Configuration ---
    config = {
        "model_type": "XGBoost",        # "XGBoost" ou "CatBoost"
        "learning_rate": 0.05,
        "max_depth": 7,
        "n_estimators": 720,
        "min_child_weight": 360,         # XGBoost seulement
        "subsample": 0.85,               # XGBoost seulement
        "colsample_bytree": 0.3,         # XGBoost seulement
        "use_offsets": True,
        "clean_method": "v1_impute_median_target_enc",
        "test_size": 0.2,
    }

    # --- Init W&B ---
    wandb.init(
        project="GLO7030-Prudential",
        config=config,
        name=f"{config['model_type']}_lr{config['learning_rate']}_d{config['max_depth']}_n{config['n_estimators']}",
    )
    config = wandb.config  # Permet au sweep de remplacer les valeurs

    print("=" * 60)
    print(f"ENTRAÎNEMENT : {config['model_type']}")
    print(f"Config : lr={config['learning_rate']}, depth={config['max_depth']}, n={config['n_estimators']}")
    print("=" * 60)

    # --- Charger les données ---
    X, y, X_test, test_ids, feature_cols = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config["test_size"], random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape} | Val: {X_val.shape}")

    # --- Entraînement ---
    if config["model_type"] == "XGBoost":
        model, score_simple, score_offset, offsets, val_preds, val_offset = train_xgboost(
            config, X_train, y_train, X_val, y_val
        )
    else:
        model, score_simple, score_offset, offsets, val_preds, val_offset = train_catboost(
            config, X_train, y_train, X_val, y_val
        )

    # --- Logger les scores finaux ---
    best_score = score_offset if config["use_offsets"] else score_simple
    wandb.log({
        "val_qwk_simple": score_simple,
        "val_qwk_offset": score_offset,
        "qwk_gain_offset": score_offset - score_simple,
        "best_qwk": best_score,
    })

    # Résumé dans W&B (valeurs finales visibles dans le tableau)
    wandb.summary["val_qwk_simple"] = score_simple
    wandb.summary["val_qwk_offset"] = score_offset
    wandb.summary["best_qwk"] = best_score

    print(f"\n{'=' * 60}")
    print(f"RÉSULTATS — {config['model_type']}")
    print(f"{'=' * 60}")
    print(f"  QWK arrondi simple : {score_simple:.4f}")
    print(f"  QWK avec offsets   : {score_offset:.4f}  (gain: +{score_offset - score_simple:.4f})")
    print(f"  Meilleur QWK       : {best_score:.4f}")
    print(f"{'=' * 60}")

    print("\nGénération des graphiques W&B...")
    # 1. Matrice de Confusion
    wandb.log({
        "confusion_matrix": wandb.plot.confusion_matrix(
            preds=val_offset,
            y_true=y_val,
            class_names=[str(i) for i in range(1, 9)]
        )
    })

    # 2. Importance des Variables
    if config["model_type"] == "XGBoost":
        importance_dict = model.get_score(importance_type='gain')
        features = list(importance_dict.keys())
        importances = list(importance_dict.values())
    else:
        importances = model.get_feature_importance()
        features = feature_cols

    df_imp = pd.DataFrame({"Feature": features, "Importance": importances})
    df_imp = df_imp.sort_values(by="Importance", ascending=False).head(15)
    
    table = wandb.Table(data=df_imp, columns=["Feature", "Importance"])
    wandb.log({
        "Feature_Importance": wandb.plot.bar(
            table, "Feature", "Importance", title="Top 15 Feature Importance"
        )
    })

    # 3. Density Plot (Offsets)
    plt.figure(figsize=(10, 5))
    sns.kdeplot(val_preds, fill=True, color="blue", alpha=0.3, label="Distribution des prédictions continues")

    # Ajouter les lignes verticales pour les offsets optimisés
    for j in range(NUM_CLASSES):
        seuil = (j + 0.5) - offsets[j]
        plt.axvline(x=seuil, color='red', linestyle='--', alpha=0.7)

    plt.title("Distribution des prédictions et seuils optimisés")
    plt.xlabel("Valeur Prédite Continue")
    plt.ylabel("Densité")
    plt.legend()

    # Logger l'image directement dans W&B
    wandb.log({"Threshold_Optimization": wandb.Image(plt)})
    plt.close() # Nettoyer la mémoire

    # --- Générer la soumission ---
    print("\nGénération de la soumission...")
    if config["model_type"] == "XGBoost":
        dfull = xgb.DMatrix(X, label=y)
        dtest = xgb.DMatrix(X_test)
        xgb_params = {
            "objective": "reg:squarederror",
            "eta": config["learning_rate"],
            "min_child_weight": config["min_child_weight"],
            "subsample": config["subsample"],
            "colsample_bytree": config["colsample_bytree"],
            "max_depth": config["max_depth"],
            "verbosity": 0,
        }
        final_model = xgb.train(xgb_params, dfull, config["n_estimators"], verbose_eval=False)
        full_preds = final_model.predict(dfull)
        test_preds_raw = final_model.predict(dtest)
    else:
        final_model = CatBoostRegressor(
            iterations=config["n_estimators"], depth=config["max_depth"],
            learning_rate=config["learning_rate"], loss_function="RMSE",
            random_seed=42, verbose=0, task_type="CPU",
        )
        final_model.fit(X, y)
        full_preds = final_model.predict(X)
        test_preds_raw = final_model.predict(X_test)

    if config["use_offsets"]:
        final_offsets = optimize_offsets(full_preds, y)
        final_predictions = apply_offsets(test_preds_raw, final_offsets)
    else:
        final_predictions = np.clip(np.round(test_preds_raw), 1, 8).astype(int)

    submission = pd.DataFrame({"Id": test_ids, "Response": final_predictions})
    submission_path = os.path.join(DATA_DIR, "submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Soumission sauvegardée : {submission_path}")
    print(f"Distribution :\n{pd.Series(final_predictions).value_counts().sort_index()}")

    # Sauvegarder la soumission comme artifact W&B
    artifact = wandb.Artifact("submission", type="predictions")
    artifact.add_file(submission_path)
    wandb.log_artifact(artifact)

    wandb.finish()
    print("\n✅ Run terminée et synchronisée sur W&B !")


if __name__ == "__main__":
    main()
