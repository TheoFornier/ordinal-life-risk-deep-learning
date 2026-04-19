"""
GLO-7030 -- Registre des modeles et palette de couleurs
Chaque membre ajoute son modele a la liste MODELS.

INSTRUCTIONS :
  1. Choisir une couleur dans COLOR_PALETTE (index 0 a 14)
  2. Copier le template en bas du fichier
  3. Remplir les champs, ajouter le dict a MODELS
  4. Lancer : python plot_demo.py
"""

# ---------------------------------------------------------------------------
# 15 couleurs distinctes -- choisir un index libre
# ---------------------------------------------------------------------------
COLOR_PALETTE = [
    "#2563eb",  # 0  bleu roi
    "#f59e0b",  # 1  ambre
    "#10b981",  # 2  emeraude
    "#ef4444",  # 3  rouge vif
    "#8b5cf6",  # 4  violet
    "#ec4899",  # 5  rose
    "#06b6d4",  # 6  cyan
    "#84cc16",  # 7  lime
    "#f97316",  # 8  orange
    "#6366f1",  # 9  indigo
    "#14b8a6",  # 10 teal
    "#e11d48",  # 11 framboise
    "#a855f7",  # 12 pourpre
    "#0ea5e9",  # 13 bleu ciel
    "#d97706",  # 14 moutarde
]


# ---------------------------------------------------------------------------
# Modeles enregistres
# ---------------------------------------------------------------------------
MODELS = [

    # -- XGBoost (Bilal, sweep 14 avril) -- couleur 0
    {
        "name": "XGBoost",
        "author": "Bilal",
        "color": COLOR_PALETTE[0],
        "run_id": "0g2qb1ud",
        "qwk_simple": 0.5990,
        "qwk_offset": 0.6535,
        "val_rmse": 1.8419,
        "train_rmse": 1.7587,
        "hyperparams": {
            "learning_rate": 0.0421,
            "max_depth": 4,
            "n_estimators": 750,
            "min_child_weight": 100,
            "subsample": 0.744,
            "colsample_bytree": 0.645,
        },
        "train_rmse_curve": None,
        "val_rmse_curve": None,
        "confusion_matrix": None,
        "pred_distribution": None,
        "true_distribution": None,
    },

    # -- CatBoost (Bilal, sweep 14 avril) -- couleur 1
    {
        "name": "CatBoost",
        "author": "Bilal",
        "color": COLOR_PALETTE[1],
        "run_id": "sno5wrbj",
        "qwk_simple": 0.6013,
        "qwk_offset": 0.6538,
        "val_rmse": None,
        "train_rmse": None,
        "hyperparams": {
            "learning_rate": 0.1935,
            "max_depth": 7,
            "n_estimators": 300,
            "best_iteration": 213,
        },
        "train_rmse_curve": None,
        "val_rmse_curve": None,
        "confusion_matrix": None,
        "pred_distribution": None,
        "true_distribution": None,
    },

    # -- TEMPLATE (decommenter et remplir) -- couleur 2
    # {
    #     "name": "MonModele",
    #     "author": "Prenom",
    #     "color": COLOR_PALETTE[2],
    #     "run_id": "xxx",
    #     "qwk_simple": 0.0,
    #     "qwk_offset": None,       # None si non applicable
    #     "val_rmse": None,
    #     "train_rmse": None,
    #     "hyperparams": {},
    #     "train_rmse_curve": None,  # liste de float
    #     "val_rmse_curve": None,
    #     "confusion_matrix": None,  # numpy array 8x8
    #     "pred_distribution": None, # liste de 8 comptages
    # },
]

if __name__ == "__main__":
    import json
    with open("register.json", "w", encoding="utf-8") as f:
        json.dump(MODELS, f, indent=4)
    print(f"Registre manuel exporte dans 'register.json' ({len(MODELS)} modeles).")

