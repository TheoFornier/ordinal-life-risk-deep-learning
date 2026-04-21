"""
GLO-7030 — Sweep W&B (Recherche automatique d'hyperparamètres)
Lance une recherche bayésienne pour trouver la meilleure combinaison.
"""

import wandb

# ============================================================
# CONFIGURATION DU SWEEP
# ============================================================
sweep_config = {
    "method": "bayes",  # Recherche bayésienne (plus intelligent que random/grid)
    "metric": {
        "name": "best_qwk",
        "goal": "maximize",
    },
    "parameters": {
        "model_type": {
            "values": ["XGBoost", "CatBoost"],
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 0.01,
            "max": 0.3,
        },
        "max_depth": {
            "values": [3, 4, 5, 6, 7, 8, 9, 10],
        },
        "n_estimators": {
            "values": [300, 500, 750, 1000, 1200],
        },
        "min_child_weight": {
            "values": [50, 100, 200, 360, 500],
        },
        "subsample": {
            "distribution": "uniform",
            "min": 0.6,
            "max": 1.0,
        },
        "colsample_bytree": {
            "distribution": "uniform",
            "min": 0.2,
            "max": 0.8,
        },
        "use_offsets": {
            "values": [True, False],
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
    # Import ici pour éviter les problèmes de scope
    from train_wandb import load_data, train_xgboost, train_catboost
    from train_wandb import optimize_offsets, apply_offsets, qwk
    from sklearn.model_selection import train_test_split
    import numpy as np

    wandb.init()
    config = wandb.config

    # Charger les données
    X, y, X_test, test_ids, feature_cols = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.test_size, random_state=42, stratify=y
    )

    # Entraîner
    if config.model_type == "XGBoost":
        model, score_simple, score_offset, offsets, val_preds, val_offset = train_xgboost(
            config, X_train, y_train, X_val, y_val
        )
    else:
        model, score_simple, score_offset, offsets, val_preds, val_offset = train_catboost(
            config, X_train, y_train, X_val, y_val
        )

    best_score = score_offset if config.use_offsets else score_simple

    wandb.log({
        "val_qwk_simple": score_simple,
        "val_qwk_offset": score_offset,
        "best_qwk": best_score,
    })
    wandb.summary["best_qwk"] = best_score

    print(f"[{config.model_type}] lr={config.learning_rate:.4f} d={config.max_depth} "
          f"-> QWK={best_score:.4f}")

    wandb.finish()


# ============================================================
# LANCEMENT DU SWEEP
# ============================================================
if __name__ == "__main__":
    # Créer le sweep
    sweep_id = wandb.sweep(sweep_config, project="GLO7030-Prudential")

    # Lancer l'agent — 30 combinaisons
    print(f"\nLancement du sweep : {sweep_id}")
    print("30 combinaisons vont être testées automatiquement (devrait prendre ~15-20 min)...")
    print("Tu peux suivre l'avancée en temps réel sur W&B !\n")

    wandb.agent(sweep_id, function=sweep_train, count=30)
