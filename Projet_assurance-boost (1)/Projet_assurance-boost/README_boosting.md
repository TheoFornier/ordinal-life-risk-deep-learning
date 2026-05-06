# README — Notebook 2 : Entraînement des Modèles de Boosting

## Fichier : `notebook_boosting.ipynb` / `02_boosting.ipynb`

## Compétition Kaggle

**Prudential Life Insurance Assessment**  
🔗 [https://www.kaggle.com/c/prudential-life-insurance-assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment)

**Objectif** : Entraîner des modèles XGBoost et CatBoost pour prédire le niveau de risque (`Response`, 1 à 8) et générer un fichier de soumission pour Kaggle.

**Métrique d'évaluation** : **Quadratic Weighted Kappa (QWK)** — mesure l'accord entre les prédictions et la réalité, en tenant compte de la nature ordinale de la variable cible.

---

## Plan du notebook

| Étape | Section | Description |
|:------|:--------|:------------|
| 0 | Installation & Imports | Dépendances, configuration |
| — | Fonctions utilitaires | QWK, offset optimization, application des offsets |
| 1 | Chargement des données | Lecture des CSV nettoyés (sortie de la Partie 1) |
| 2 | Split train/validation | 80% entraînement / 20% validation |
| 3 | Entraînement XGBoost | Arrondi simple + avec offsets |
| 4 | Entraînement CatBoost | Arrondi simple + avec offsets |
| 5 | Comparaison | Tableau récapitulatif des 4 configurations |
| 6 | Soumission | Re-entraînement sur tout le train + prédiction test |

---

## 4 configurations testées

| # | Modèle | Méthode de post-traitement | Description |
|:--|:-------|:---------------------------|:------------|
| 1 | **XGBoost** | Arrondi simple | La valeur continue prédite est arrondie à l'entier le plus proche, puis clippée entre 1 et 8 |
| 2 | **XGBoost** | Offset optimization | Un offset par classe est optimisé pour maximiser le QWK |
| 3 | **CatBoost** | Arrondi simple | Même logique que #1 |
| 4 | **CatBoost** | Offset optimization | Même logique que #2 |

---

## Détail de chaque étape

### 0. Installation et Imports

**Dépendances installées** :
```
pip install pandas numpy scikit-learn xgboost catboost scipy
```

**Librairies utilisées** :
- `pandas` — manipulation de données
- `numpy` — calcul numérique
- `xgboost` — modèle XGBoost (gradient boosting)
- `catboost.CatBoostRegressor` — modèle CatBoost (gradient boosting)
- `scipy.optimize.minimize_scalar` — optimisation unidimensionnelle pour les offsets
- `sklearn.metrics.cohen_kappa_score` — calcul du Quadratic Weighted Kappa
- `sklearn.model_selection.train_test_split` — split train/validation
- `warnings` — suppression des warnings
- `os` — gestion des chemins de fichiers

**Configuration** :
- `NUM_CLASSES = 8` — nombre de classes (Response va de 1 à 8)

---

### Fonctions utilitaires

Trois fonctions clés sont définies :

#### `qwk(y_pred, y_true)` — Quadratic Weighted Kappa

```python
def qwk(y_pred, y_true):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred)
    y_pred = np.clip(np.round(y_pred), np.min(y_true), np.max(y_true)).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")
```

**Fonctionnement** :
1. Convertir `y_true` en entiers
2. Arrondir `y_pred` à l'entier le plus proche
3. Clipper les prédictions dans la plage `[min(y_true), max(y_true)]`
4. Calculer le Cohen's Kappa avec pondération quadratique

**Interprétation du QWK** :
- `1.0` = accord parfait
- `0.0` = accord aléatoire
- `< 0` = pire que le hasard

---

#### `optimize_offsets(preds, labels)` — Optimisation des offsets par classe

**Concept** : Plutôt qu'arrondir simplement, on ajoute un décalage (offset) à chaque classe pour « pousser » les prédictions dans la bonne direction. Chaque classe `j` (0–7) reçoit un offset `δ_j` ∈ [-3, +3].

```python
def optimize_offsets(preds, labels):
    for j in [6, 4, 5, 3, 2, 1, 7, 0]:   # ordre spécifique
        result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
        offsets[j] = result.x
```

**Détail de l'algorithme** :
1. **Ordre d'optimisation** : `[6, 4, 5, 3, 2, 1, 7, 0]` — les classes sont optimisées dans un ordre choisi de manière empirique (souvent les classes du milieu d'abord, car elles ont le plus d'impact sur le QWK)
2. **Pour chaque classe `j`** :
   - Définir une fonction objectif qui ajoute un offset `x` à toutes les prédictions correspondant à la classe `j`
   - Minimiser `-QWK` (car `minimize_scalar` minimise, mais on veut maximiser le QWK)
   - Bornes de recherche : `[-3, +3]`
   - Méthode : `"bounded"` (optimisation bornée)
3. **Résultat** : un tableau de 8 offsets, un par classe

**Pourquoi cette approche fonctionne** : Le modèle de régression produit des valeurs continues (ex: 3.7). L'arrondi simple donne 4, mais peut-être que pour les Response autour de 3.7, la vraie réponse est plus souvent 3. L'offset corrige ce biais systématique par classe.

---

#### `apply_offsets(preds, offsets)` — Application des offsets

```python
def apply_offsets(preds, offsets):
    for j in range(NUM_CLASSES):
        mask = preds.astype(int) == j
        adjusted[mask] = preds[mask] + offsets[j]
    return np.clip(np.round(adjusted), 1, 8).astype(int)
```

**Fonctionnement** :
1. Pour chaque classe `j`, sélectionner les prédictions dont la partie entière = `j`
2. Ajouter l'offset correspondant
3. Arrondir et clipper entre 1 et 8

---

### 1. Chargement des données

**Fichiers en entrée** (sortie de la Partie 1 — nettoyage) :
```python
train = pd.read_csv("prudential-life-insurance-assessment/train_clean.csv")
test  = pd.read_csv("prudential-life-insurance-assessment/test_clean.csv")
```

**Préparation** :
- `feature_cols` = toutes les colonnes sauf `"Id"` et `"Response"`
- `X` = matrice de features du train (numpy array)
- `y` = vecteur cible (numpy array)
- `X_test` = matrice de features du test
- `test_ids` = identifiants du test (pour la soumission)

---

### 2. Split train / validation (80/20)

```python
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Paramètres** :
- `test_size=0.2` → 80% entraînement, 20% validation
- `random_state=42` → reproductibilité
- `stratify=y` → **stratification** : chaque fold contient la même proportion de chaque classe Response que le dataset complet (important car les classes sont déséquilibrées)

---

### 3. Entraînement XGBoost

#### Hyperparamètres

```python
xgb_params = {
    "objective": "reg:squarederror",   # Régression (MSE)
    "eta": 0.05,                        # Learning rate (faible = meilleure généralisation)
    "min_child_weight": 360,            # Régularisation forte
    "subsample": 0.85,                  # 85% des lignes par arbre (bagging)
    "colsample_bytree": 0.3,            # 30% des features par arbre
    "max_depth": 7,                     # Profondeur maximale des arbres
    "verbosity": 0,                     # Pas de logs
}
XGB_ROUNDS = 720                        # Nombre de boosting rounds
```

**Choix importants** :
- **`reg:squarederror`** : On traite le problème comme une régression (pas une classification), car Response est ordinal. La valeur 3 est « plus proche » de 4 que de 8, ce qui est capturé par la régression.
- **`eta = 0.05`** : Learning rate faible pour une convergence lente mais précise, compensée par un grand nombre de rounds (720).
- **`min_child_weight = 360`** : Valeur élevée qui force chaque feuille à contenir au moins 360 observations → forte régularisation contre l'overfitting.
- **`colsample_bytree = 0.3`** : Seulement 30% des features par arbre → diversité des arbres et régularisation.

#### Entraînement

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val, label=y_val)

xgb_model = xgb.train(
    xgb_params, dtrain, XGB_ROUNDS,
    evals=[(dtrain, "train"), (dval, "val")],
    verbose_eval=100,
)
```

- Utilisation de `xgb.DMatrix` (format optimisé de XGBoost)
- Évaluation sur train et val tous les 100 rounds
- Pas d'early stopping (on fait les 720 rounds complets)

#### Résultats

1. **Score [1] — XGBoost arrondi simple** : QWK calculé directement en arrondissant les prédictions continues
2. **Score [2] — XGBoost avec offsets** :
   - Optimisation des offsets sur les prédictions du **train** (`xgb_train_preds`)
   - Application des offsets sur les prédictions de **validation** (`xgb_val_preds`)
   - Affichage du gain apporté par les offsets

---

### 4. Entraînement CatBoost

#### Hyperparamètres

```python
cat_model = CatBoostRegressor(
    iterations=720,              # Nombre de boosting rounds
    depth=7,                     # Profondeur maximale des arbres
    learning_rate=0.05,          # Learning rate
    loss_function="RMSE",        # Root Mean Square Error
    random_seed=42,              # Reproductibilité
    verbose=100,                 # Log tous les 100 rounds
    early_stopping_rounds=50,    # Arrêt si pas d'amélioration pendant 50 rounds
    task_type="CPU",             # Pas de GPU
)
```

**Différences avec XGBoost** :
- **Early stopping** : CatBoost s'arrête si la validation ne s'améliore pas pendant 50 rounds consécutifs (évite l'overfitting)
- **`loss_function="RMSE"`** : Équivalent à `reg:squarederror` de XGBoost

#### Entraînement

```python
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
```

- Interface scikit-learn style (plus simple que XGBoost)
- Évaluation sur le set de validation pour l'early stopping

#### Résultats

1. **Score [3] — CatBoost arrondi simple** : QWK avec arrondi simple
2. **Score [4] — CatBoost avec offsets** : QWK avec offset optimization (même technique que XGBoost)

---

### 5. Comparaison des 4 configurations

Tableau récapitulatif affiché dans le notebook :

```
==================================================
RESULTATS (Quadratic Weighted Kappa sur validation)
==================================================
  [1] XGBoost simple         : X.XXXX
  [2] XGBoost offset         : X.XXXX
  [3] CatBoost simple        : X.XXXX
  [4] CatBoost offset        : X.XXXX
──────────────────────────────────────────────────
  Meilleur : [X] ... (QWK = X.XXXX)
```

Le meilleur modèle est automatiquement sélectionné via :
```python
best_name = max(results, key=results.get)
```

---

### 6. Soumission (meilleur modèle sur le test)

#### Logique automatique

Le notebook sélectionne automatiquement le meilleur modèle et sa méthode :

```python
use_offsets = "offset" in best_name    # True si la meilleure config utilise les offsets
use_catboost = "CatBoost" in best_name # True si le meilleur est CatBoost
```

#### Re-entraînement sur TOUT le train

**Pourquoi re-entraîner ?** : Le modèle a été entraîné sur 80% des données. Pour la soumission finale, on utilise 100% des données d'entraînement pour maximiser la performance.

**Si CatBoost est le meilleur** :
```python
final_model = CatBoostRegressor(iterations=720, depth=7, learning_rate=0.05, ...)
final_model.fit(X, y)  # X = tout le train
```
Note : pas d'early stopping ici (pas de validation set), on fait les 720 iterations complètes.

**Si XGBoost est le meilleur** :
```python
dfull = xgb.DMatrix(X, label=y)
final_model = xgb.train(xgb_params, dfull, XGB_ROUNDS, verbose_eval=False)
```

#### Application des offsets (si applicable)

Si la meilleure config utilise les offsets :
1. Calcul des offsets sur les prédictions du train complet
2. Application sur les prédictions du test

Sinon : arrondi simple + clip [1, 8]

#### Génération du fichier de soumission

```python
submission = pd.DataFrame({"Id": test_ids, "Response": final_predictions})
submission.to_csv("prudential-life-insurance-assessment/submission.csv", index=False)
```

**Affichage** :
- Nombre de prédictions
- Distribution des classes prédites (pour vérifier que le modèle ne prédit pas une seule classe)
- Aperçu des 10 premières lignes

---

## Fichiers en entrée / sortie

| Direction | Fichier | Description |
|:----------|:--------|:------------|
| **Entrée** | `prudential-life-insurance-assessment/train_clean.csv` | Données nettoyées (sortie de Partie 1) |
| **Entrée** | `prudential-life-insurance-assessment/test_clean.csv` | Données nettoyées (sortie de Partie 1) |
| **Sortie** | `prudential-life-insurance-assessment/submission.csv` | Prédictions finales pour Kaggle |

---

## Résumé technique

| Aspect | Détail |
|:-------|:-------|
| **Approche** | Régression (pas classification) — car la cible Response est ordinale |
| **Modèles** | XGBoost et CatBoost |
| **Post-traitement** | Arrondi simple **ou** offset optimization par classe |
| **Validation** | Train/Val split 80/20 stratifié |
| **Métrique** | Quadratic Weighted Kappa (QWK) |
| **Soumission** | Re-entraînement sur 100% du train avec le meilleur modèle |

---

## Pistes d'amélioration

- **Optuna** : Tuner automatiquement les hyperparamètres (eta, max_depth, subsample, etc.)
- **Ensemble** : Moyenner les prédictions XGBoost + CatBoost pour plus de robustesse
- **Plus de features** : Interactions supplémentaires, agrégations statistiques, etc.
- **Cross-validation** : Remplacer le simple split 80/20 par un K-Fold complet
- **LightGBM** : Ajouter un 3ème modèle de boosting
