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

| Argument         | Modèle |
|------------------|--------|
| `--model tabnet` | TabNet |
| `--model tabm`   | TabM   |

### Exemples

```bash
python train.py --model tabnet
python train.py --model tabm
```

Les datasets à utiliser sont définis dans `DataConfig.datasets` dans `config.py`. Si plusieurs fichiers sont listés, le script enchaîne automatiquement un entraînement complet pour chacun.

Le script enchaîne automatiquement pour chaque dataset :

1. Chargement (ou nettoyage) des données — le CSV nettoyé est mis en cache automatiquement
2. Séparation stratifiée train / validation / test (70 / 15 / 15)
3. Entraînement avec arrêt anticipé sur la validation
4. Évaluation finale sur le jeu de test avec les offsets ajustés sur la validation
5. Sauvegarde des résultats dans `results/<dataset>_<modèle>_<timestamp>/`

---

## Comparer les résultats

```bash
python compare.py                              # tous les runs, triés par QWK test+offsets
python compare.py --model tabm                 # filtrer par modèle
python compare.py --dataset train              # filtrer par dataset
python compare.py --sort val_qwk_offset        # trier par une autre métrique
python compare.py --export mon_fichier.csv     # exporter vers un fichier CSV personnalisé
```

Le tableau est toujours exporté automatiquement dans `results/comparison.csv`.

### Métriques de tri disponibles

`val_qwk_raw`, `val_qwk_offset`, `test_qwk_raw`, `test_qwk_offset`, `val_acc_raw`, `val_acc_offset`, `test_acc_raw`, `test_acc_offset`

---

## Résultats d'un run

Chaque entraînement crée un dossier `results/<dataset>_<modèle>_<timestamp>/` contenant :

| Fichier                | Contenu |
|------------------------|---------|
| `training.log`         | Log complet de l'entraînement (perte et QWK par époque) |
| `hyperparameters.json` | Config complète + métriques finales (val et test) |
| `plots.png`            | Courbes de perte et de QWK par époque |

---

## Configuration

Tous les hyperparamètres sont dans `config.py`. Il n'y a aucun argument CLI à part `--model`.

### DataConfig

Paramètres globaux du pipeline de données et d'entraînement.

| Paramètre      | Défaut                 | Description |
|----------------|------------------------|-------------|
| `datasets`     | `[]`                   | Liste des chemins vers les fichiers CSV à entraîner (un run par fichier) |
| `results_dir`  | `results/`             | Répertoire racine pour les résultats de chaque run |
| `use_cached`   | `True`                 | Réutilise le CSV nettoyé (`<nom>_clean.csv`) si disponible ; mettre à `False` pour relancer le nettoyage |
| `val_size`     | `0.15`                 | Proportion des données réservée à la validation |
| `test_size`    | `0.15`                 | Proportion des données réservée au test final |
| `random_state` | `42`                   | Graine aléatoire (split, encodage cible, etc.) |

> Le cache est dérivé automatiquement du chemin du CSV : `train.csv` → `train_clean.csv` dans le même répertoire.

---

### TabNetConfig

Paramètres de l'architecture TabNet (via `pytorch-tabnet`).

| Paramètre                 | Défaut | Description |
|---------------------------|--------|-------------|
| `n_d`                     | `32`   | Dimension de l'espace de décision |
| `n_a`                     | `32`   | Dimension de l'espace d'attention (typiquement égal à `n_d`) |
| `n_steps`                 | `5`    | Nombre d'étapes séquentielles |
| `gamma`                   | `1.5`  | Coefficient de régularisation de l'attention |
| `n_independent`           | `2`    | Couches GLU indépendantes par étape |
| `n_shared`                | `2`    | Couches GLU partagées entre les étapes |
| `momentum`                | `0.02` | Momentum de la batch normalization |
| `lr`                      | `2e-3` | Taux d'apprentissage (Adam) |
| `batch_size`              | `1024` | Taille de batch |
| `virtual_batch_size`      | `256`  | Taille de batch virtuel pour la Ghost Batch Normalization |
| `epochs`                  | `50`   | Nombre maximal d'époques |
| `early_stopping_patience` | `15`   | Époques sans amélioration avant arrêt anticipé |

---

### TabMConfig

Paramètres de l'architecture TabM avec embeddings numériques (PiecewiseLinear).

| Paramètre                 | Défaut  | Description |
|---------------------------|---------|-------------|
| `k`                       | `32`    | Nombre de sous-modèles ensemblés |
| `n_blocks`                | `2`     | Nombre de blocs MLP |
| `d_block`                 | `512`   | Dimension cachée de chaque bloc |
| `dropout`                 | `0.1`   | Taux de dropout |
| `lr`                      | `2e-3`  | Taux d'apprentissage (AdamW) |
| `weight_decay`            | `3e-4`  | Régularisation L2 |
| `batch_size`              | `1024`  | Taille de batch |
| `epochs`                  | `50`    | Nombre maximal d'époques |
| `early_stopping_patience` | `15`    | Époques sans amélioration avant arrêt anticipé |
| `use_embeddings`          | `True`  | Active les embeddings PiecewiseLinear sur les features numériques |
| `n_bins`                  | `48`    | Nombre de bins pour les embeddings PiecewiseLinear |
| `d_embedding`             | `16`    | Dimension des embeddings numériques |
