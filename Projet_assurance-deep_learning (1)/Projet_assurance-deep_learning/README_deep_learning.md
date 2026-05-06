# Deep Learning — Prudential Life Insurance

Pipeline d'entraînement de modèles de deep learning tabulaire sur le dataset Kaggle [Prudential Life Insurance Assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment).

La métrique cible est le **Quadratic Weighted Kappa (QWK)**.

---

## Lancer un entraînement

Ouvrir `deep_learning/train.py` dans VSCode et cliquer sur le bouton **Run** (▶), ou depuis un terminal :

```bash
cd deep_learning
python train.py
```

Il n'y a aucun argument CLI. Le modèle et tous les paramètres se configurent directement dans `config.py`.

Le script enchaîne automatiquement :

1. Un entraînement sur le dataset de base (`train_path`), si `run_base_training = True`
2. Un entraînement par dossier listé dans `synthetic_datasets`, où :
   - Le split train/val est effectué sur le `train_clean.csv` du dossier
   - Les fichiers CSV synthétiques du dossier sont ajoutés à l'ensemble d'entraînement
   - Le prétraitement (scaler, bins, normalisation y) est calibré sur les données réelles uniquement
   - Un poids de loss réduit est appliqué aux lignes synthétiques (`synth_sample_weight`)

---

## Structure d'un dossier synthétique

Chaque dossier listé dans `synthetic_datasets` doit contenir :

| Fichier | Rôle |
|---|---|
| `train_clean.csv` | Données réelles nettoyées — utilisées pour le split train/val |
| `test_clean.csv` | Ignoré lors de l'entraînement |
| `*_clean.csv` (autres) | Données synthétiques ajoutées à l'ensemble d'entraînement |

---

## Comparer les résultats

```bash
python compare.py                              # tous les runs, triés par QWK val+offsets
python compare.py --model tabm                 # filtrer par modèle
python compare.py --dataset train              # filtrer par dataset
python compare.py --sort val_qwk_raw           # trier par une autre métrique
python compare.py --export mon_fichier.csv     # exporter vers un fichier CSV personnalisé
```

Le tableau est toujours exporté automatiquement dans `results/comparison.csv`.

### Métriques de tri disponibles

`val_qwk_raw`, `val_qwk_offset`, `val_acc_raw`, `val_acc_offset`

---

## Résultats d'un run

Chaque entraînement crée un dossier `results/<dataset>_<modèle>_<timestamp>/` contenant :

| Fichier | Contenu |
|---|---|
| `training.log` | Log complet de l'entraînement (perte et QWK par époque) |
| `hyperparameters.json` | Config complète + métriques finales |
| `plots.png` | Courbes de perte et de QWK par époque |

---

## Configuration

Tous les paramètres sont dans `config.py`.

### DataConfig

Paramètres globaux du pipeline de données et d'entraînement.

| Paramètre | Défaut | Description |
|---|---|---|
| `model` | `"tabm"` | Modèle à entraîner : `"tabm"` ou `"tabnet"` |
| `run_base_training` | `True` | Si `False`, ignore le run sur `train_path` et enchaîne directement les dossiers synthétiques |
| `synth_sample_weight` | `0.5` | Poids de loss appliqué aux lignes synthétiques (1.0 = identique aux lignes réelles) |
| `max_synth_ratio` | `1.0` | Nombre maximum de lignes synthétiques en multiple des lignes réelles (ex. `0.5` = moitié moins) |
| `train_path` | `../prudential-.../train_clean.csv` | Dataset réel de base |
| `synthetic_datasets` | liste de dossiers | Dossiers contenant les données d'augmentation synthétique |
| `results_dir` | `deep_learning/results/` | Répertoire de sortie des runs |
| `random_state` | `42` | Graine pour les splits et l'échantillonnage synthétique |

---

### TabNetConfig

Paramètres de l'architecture TabNet (via `pytorch-tabnet`).

| Paramètre | Défaut | Description |
|---|---|---|
| `n_d` | `64` | Dimension de l'espace de décision |
| `n_a` | `64` | Dimension de l'espace d'attention (typiquement égal à `n_d`) |
| `n_steps` | `6` | Nombre d'étapes séquentielles |
| `gamma` | `1.5` | Coefficient de régularisation de l'attention |
| `n_independent` | `2` | Couches GLU indépendantes par étape |
| `n_shared` | `2` | Couches GLU partagées entre les étapes |
| `momentum` | `0.02` | Momentum de la batch normalization |
| `lr` | `1e-3` | Taux d'apprentissage (Adam) |
| `batch_size` | `1024` | Taille de batch |
| `virtual_batch_size` | `256` | Taille de batch virtuel pour la Ghost Batch Normalization |
| `epochs` | `200` | Nombre maximal d'époques |
| `early_stopping_patience` | `15` | Époques sans amélioration avant arrêt anticipé |

---

### TabMConfig

Paramètres de l'architecture TabM avec embeddings numériques (PiecewiseLinear).

| Paramètre | Défaut | Description |
|---|---|---|
| `k` | `32` | Nombre de sous-modèles ensemblés |
| `n_blocks` | `3` | Nombre de blocs MLP |
| `d_block` | `512` | Dimension cachée de chaque bloc |
| `dropout` | `0.1` | Taux de dropout |
| `lr` | `1e-3` | Taux d'apprentissage (AdamW) |
| `weight_decay` | `3e-4` | Régularisation L2 |
| `batch_size` | `1024` | Taille de batch |
| `epochs` | `100` | Nombre maximal d'époques |
| `early_stopping_patience` | `15` | Époques sans amélioration avant arrêt anticipé |
| `use_embeddings` | `True` | Active les embeddings PiecewiseLinear sur les features numériques |
| `n_bins` | `48` | Nombre de bins pour les embeddings PiecewiseLinear |
| `d_embedding` | `16` | Dimension des embeddings numériques |

---

## Dépendances

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128  # PyTorch avec CUDA 12.8
pip install -r requirements.txt
```