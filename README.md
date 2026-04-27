# Adaptation d’un jeu de données tabulaire pour modèles deep learning (synthèse de données)

Ce projet explore la génération de données synthétiques pour adapter un dataset tabulaire aux contraintes des modèles de deep learning. L’objectif principal est de réduire le déséquilibre de la variable cible `Response` et d’évaluer l’impact des données synthétiques sur la qualité des modèles.

Principales approches implémentées
- Stratégie A — génération ciblée pour les classes rares : augmentation uniquement des classes sous-représentées.
- Stratégie B — génération pour équilibrage complet : augmentation jusqu'à l'équilibre entre classes.
- Stratégie C - génération pour les classe 1 et 2 qui sont les moins bien prédites.
- Expérimentations avec plusieurs générateurs (CTGAN, TVAE, Tabsyn, LLM) et évaluation comparative.

Deux environnements virtuels : un pour TabSyn et un pour le reste du code, car TabSyn nécessite des versions précises et antérieures de certains outils.

Structure du dépôt
- `ctgan/` : scripts d'entraînement et de génération avec CTGAN
    - `train_ctgan.py` — entraînement du modèle CTGAN
    - `generate_strategy_a.py`, `generate_strategy_b.py` — génération selon les stratégies A et B
    - `utils.py` — utilitaires partagés
    - `outputs/` — datasets synthétiques et modèles entraînés
- `tvae/` : scripts pour TVAE (mêmes usages que `ctgan/`)
- `tabsyn/`, `external/tabsyn/` : implémentations et dépendances pour Tabsyn
- `LLM/` : dataset générés par LLM.
- `data/` : données brutes 
- `data_clean/` : données nettoyées prêtes pour entraînement
 - `LLM/` : jeux de données produits par LLM (CSV)
 - `data/` : données brutes
 - `data_clean/` : données nettoyées prêtes pour entraînement
- `dataset_cleaning.ipynb`, `data_repartition.ipynb` : notebooks d'exploration et préparation
- `evaluation/` : scripts d'évaluation quantitatives des générateurs (`test_synth_quality.py`, `evaluate_generators.py`)
- `external/` clonage du code de Tabsyn : https://github.com/amazon-science/tabsyn



Exemples d'utilisation :


- Entraîner un CTGAN :

```bash
python ctgan/train_ctgan.py --data data_clean/train_clean.csv --out ctgan/outputs/models
```

- Générer des données selon la Stratégie A :

```bash
python ctgan/generate_strategy_a.py --model ctgan/outputs/models/last_model.pt --out ctgan/outputs/synthetic_a
```

- Évaluer la qualité synthétique :

```bash
python evaluation/test_synth_quality.py --real data_clean/train_clean.csv --synthetic ctgan/outputs/synthetic_a/synthetic_all_strategy_a.csv
```

Sorties
- `ctgan/outputs/` et `tvae/outputs/` contiennent les fichiers `synthetic_*` et `models/`.
- `evaluation/outputs/` contient les rapports d'évaluation comparant générateurs et stratégies.

Note sur les CSV
- Les CSV présents dans `LLM/` ne sont pas inclus dans le dépôt par défaut (ils peuvent être reproduits en relançant les prompts décrits dans les fichiers de `LLM/`).
- Les CSV de `data/` et `data_clean/` doivent être obtenus en téléchargeant le jeu de données depuis Kaggle (https://www.kaggle.com/competitions/prudential-life-insurance-assessment/data) puis en exécutant le nettoyage (voir `dataset_cleaning.ipynb` / scripts de préparation). Ils ne sont pas fournis ici pour alléger le dépôt.


