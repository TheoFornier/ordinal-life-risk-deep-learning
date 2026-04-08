# Deep Learning — Prudential Life Insurance

Pipeline d'entraînement de modèles de deep learning tabulaire sur le dataset Kaggle [Prudential Life Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment).

La métrique cible est le **Quadratic Weighted Kappa (QWK)**.

---

## Lancer un entraînement

```bash
cd deep_learning
python train.py --model <nom_du_modèle>
```

### Modèles disponibles

| Argument        | Modèle       |
|-----------------|--------------|
| `--model tabnet` | TabNet       |
| `--model tabm`   | TabM         |

### Exemples

```bash
python train.py --model tabnet
python train.py --model tabm
```

Le script enchaîne automatiquement :

1. Chargement (ou nettoyage) des données
2. Séparation train / validation (80/20)
3. Entraînement et évaluation sur la validation
4. Réentraînement sur l'ensemble complet (si `retrain=True`)
5. Génération du fichier de soumission

---

## Configuration

Tous les hyperparamètres sont dans `config.py`. Il n'y a aucun argument CLI à part `--model`.

### DataConfig

Paramètres globaux du pipeline de données et d'entraînement.

| Paramètre       | Défaut  | Description |
|-----------------|---------|-------------|
| `use_cached`    | `True`  | Réutilise les CSV nettoyés si disponibles ; mettre à `False` pour relancer le nettoyage |
| `retrain`       | `True`  | Réentraîne sur l'ensemble complet après la phase de validation |
| `val_size`      | `0.2`   | Proportion du jeu d'entraînement réservée à la validation |
| `random_state`  | `42`    | Graine aléatoire (split, encodage cible, etc.) |
| `log_dir`       | `logs/` | Répertoire où sont écrits les fichiers de log |
| `output_dir`    | `../prudential-life-insurance-assessment/` | Répertoire de sortie pour les soumissions |

---

### TabNetConfig

Paramètres de l'architecture TabNet (via `pytorch-tabnet`).

| Paramètre                | Défaut  | Description |
|--------------------------|---------|-------------|
| `n_d`                    | `32`    | Dimension de l'espace de décision |
| `n_a`                    | `32`    | Dimension de l'espace d'attention (typiquement égal à `n_d`) |
| `n_steps`                | `5`     | Nombre d'étapes séquentielles |
| `gamma`                  | `1.5`   | Coefficient de régularisation de l'attention |
| `n_independent`          | `2`     | Couches GLU indépendantes par étape |
| `n_shared`               | `2`     | Couches GLU partagées entre les étapes |
| `momentum`               | `0.02`  | Momentum de la batch normalization |
| `lr`                     | `2e-3`  | Taux d'apprentissage (Adam) |
| `batch_size`             | `1024`  | Taille de batch |
| `virtual_batch_size`     | `256`   | Taille de batch virtuel pour la Ghost Batch Normalization |
| `epochs`                 | `100`   | Nombre maximal d'époques |
| `early_stopping_patience`| `15`    | Époques sans amélioration avant arrêt anticipé |

---

### TabMConfig

Paramètres de l'architecture TabM avec embeddings numériques (PiecewiseLinear).

| Paramètre                | Défaut  | Description |
|--------------------------|---------|-------------|
| `k`                      | `32`    | Nombre de sous-modèles ensemblés |
| `n_blocks`               | `2`     | Nombre de blocs MLP |
| `d_block`                | `512`   | Dimension cachée de chaque bloc |
| `dropout`                | `0.1`   | Taux de dropout |
| `lr`                     | `2e-3`  | Taux d'apprentissage (AdamW) |
| `weight_decay`           | `3e-4`  | Régularisation L2 |
| `batch_size`             | `1024`  | Taille de batch |
| `epochs`                 | `100`   | Nombre maximal d'époques |
| `early_stopping_patience`| `15`    | Époques sans amélioration avant arrêt anticipé |
| `use_embeddings`         | `True`  | Active les embeddings PiecewiseLinear sur les features numériques |
| `n_bins`                 | `48`    | Nombre de bins pour les embeddings PiecewiseLinear |
| `d_embedding`            | `16`    | Dimension des embeddings numériques |
