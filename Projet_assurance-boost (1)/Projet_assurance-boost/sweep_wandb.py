"""
GLO-7030 — Sweep W&B (Recherche automatique d'hyperparamètres)
Méthode Gábor : 7 classifieurs binaires + Variable Split Calibration.
Sweep élargi : 100 runs, plages étendues, focus CatBoost.
"""

import wandb

# ============================================================
# CONFIGURATION DU SWEEP (ÉLARGI)
# ============================================================
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "best_qwk",
        "goal": "maximize",
    },
    "parameters": {
        "model_type": {
            "values": ["CatBoost", "CatBoost", "CatBoost", "CatBoost", "CatBoost",
                        "CatBoost", "CatBoost", "XGBoost", "XGBoost", "XGBoost"],
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.005,
            "max": 0.15,
        },
        "max_depth": {
            "values": [3, 4, 5, 6, 7, 8, 9, 10],
        },
        "n_estimators": {
            "values": [500, 750, 1000, 1200, 1500, 2000, 2500, 3000],
        },
        "min_child_weight": {
            "values": [10, 30, 50, 100, 150, 200, 300, 500],
        },
        "subsample": {
            "distribution": "uniform",
            "min": 0.5,
            "max": 1.0,
        },
        "colsample_bytree": {
            "distribution": "uniform",
            "min": 0.3,
            "max": 1.0,
        },
        "l2_leaf_reg": {
            "values": [1.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0],
        },
        "random_strength": {
            "values": [0.0, 0.5, 1.0, 2.0, 3.0, 5.0],
        },
        "use_offsets": {
            "value": True,
        },
        "clean_method": {
            "value": "v1_impute_median_target_enc",
        },
        "test_size": {
            "value": 0.2,
        },
    },
}


# ============================================================
# FONCTION D'ENTRAÎNEMENT POUR LE SWEEP
# ============================================================
def sweep_train():
    """Fonction appelée par l'agent de sweep pour chaque combinaison."""
    from train_wandb import load_data, train_xgboost, train_catboost, qwk
    from sklearn.model_selection import train_test_split
    import numpy as np
    import traceback

    wandb.init()
    config = wandb.config

    try:
        # Charger les données
        X, y, X_test, test_ids, feature_cols = load_data()
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=config.test_size, random_state=42, stratify=y
        )

        # Entraîner (méthode Gábor : 7 classifieurs binaires + calibration)
        if config.model_type == "XGBoost":
            _, score_simple, score_offset, boundaries, val_preds, val_offset = train_xgboost(
                config, X_train, y_train, X_val, y_val
            )
        else:
            _, score_simple, score_offset, boundaries, val_preds, val_offset = train_catboost(
                config, X_train, y_train, X_val, y_val
            )

        best_score = score_offset if config.use_offsets else score_simple

        wandb.log({
            "val_qwk_simple": score_simple,
            "val_qwk_offset": score_offset,
            "qwk_gain_calibration": score_offset - score_simple,
            "best_qwk": best_score,
        })
        wandb.summary["best_qwk"] = best_score
        wandb.summary["val_qwk_simple"] = score_simple
        wandb.summary["val_qwk_offset"] = score_offset

        print(f"[{config.model_type}] lr={config.learning_rate:.4f} d={config.max_depth} "
              f"n={config.n_estimators} -> QWK={best_score:.4f} "
              f"(simple={score_simple:.4f}, calibré={score_offset:.4f})")

    except Exception as e:
        print(f"[ERREUR] Run échoué : {e}")
        traceback.print_exc()
        wandb.log({"best_qwk": 0.0, "error": str(e)})
        wandb.summary["best_qwk"] = 0.0

    wandb.finish()


# ============================================================
# LANCEMENT DU SWEEP
# ============================================================
if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep_config, project="GLO7030-Prudential")

    N_RUNS = 100

    print(f"\n{'='*60}")
    print(f"SWEEP ÉLARGI — {N_RUNS} combinaisons")
    print(f"{'='*60}")
    print(f"  Méthode     : Bayésienne (W&B)")
    print(f"  Modèles     : CatBoost (70%) + XGBoost (30%)")
    print(f"  n_estimators: 500 à 3000")
    print(f"  lr          : 0.005 à 0.15")
    print(f"  max_depth   : 3 à 10")
    print(f"  + l2_leaf_reg, random_strength (CatBoost)")
    print(f"{'='*60}\n")

    wandb.agent(sweep_id, function=sweep_train, count=N_RUNS)
