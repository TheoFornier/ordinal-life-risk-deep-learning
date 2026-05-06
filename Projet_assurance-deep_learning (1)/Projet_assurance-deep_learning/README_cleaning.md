# README — Notebook 1 : Nettoyage et Préparation des Données

## Fichier : `notebook_cleaning.ipynb`

## Compétition Kaggle

**Prudential Life Insurance Assessment**  
🔗 [https://www.kaggle.com/c/prudential-life-insurance-assessment](https://www.kaggle.com/c/prudential-life-insurance-assessment)

**Objectif** : Prédire le niveau de risque d'un candidat à une assurance vie (`Response`, variable ordinale de 1 à 8) à partir de plus de 100 variables décrivant son profil (informations médicales, démographiques, habitudes de vie, etc.).

---

## Plan du notebook

| Étape | Section | Description |
|:------|:--------|:------------|
| 0 | Installation & Imports | Installation des dépendances, imports, configuration |
| 1 | Chargement des données | Lecture des CSV train/test, combinaison en un seul DataFrame |
| 2 | Suppression des colonnes vides | Colonnes avec >95% de valeurs manquantes |
| 3 | Imputation des valeurs manquantes | Stratégie différenciée selon le type de variable |
| 4 | Encodage des variables textuelles | Label Encoding |
| 5 | Target Encoding | K-Fold Target Encoding pour les variables à haute cardinalité |
| 6 | Feature Engineering | Création de nouvelles features métier |
| 7 | Sauvegarde | Export des DataFrames nettoyés en CSV |

---

## Détail de chaque étape

### 0. Installation et Imports

**Dépendances installées** :
```
pip install pandas numpy scikit-learn
```

**Librairies utilisées** :
- `pandas` — manipulation de données
- `numpy` — calcul numérique
- `sklearn.preprocessing.LabelEncoder` — encodage des variables catégorielles
- `sklearn.model_selection.KFold` — validation croisée pour le target encoding
- `warnings` — suppression des warnings
- `os` — gestion des chemins de fichiers

**Configuration** :
- `warnings.filterwarnings("ignore")` — masquer tous les warnings
- `pd.set_option("display.max_columns", 50)` — afficher jusqu'à 50 colonnes dans les prints

---

### 0bis. Définition des types de variables

Les colonnes du dataset sont classifiées en 4 catégories selon la documentation Kaggle :

| Type | Nombre | Exemples |
|:-----|:-------|:---------|
| **Catégorielles** (`CATEGORICAL_COLS`) | ~55 | `Product_Info_1`, `Product_Info_2`, `Medical_History_2`, ... |
| **Continues** (`CONTINUOUS_COLS`) | 13 | `Product_Info_4`, `Ins_Age`, `Ht`, `Wt`, `BMI`, `Insurance_History_5`, ... |
| **Discrètes** (`DISCRETE_COLS`) | 5 | `Medical_History_1`, `Medical_History_10`, `Medical_History_15`, `Medical_History_24`, `Medical_History_32` |
| **Dummy** (`DUMMY_COLS`) | 48 | `Medical_Keyword_1` à `Medical_Keyword_48` |

**Variable cible** : `TARGET = "Response"`

---

### 1. Chargement des données

**Chemins des fichiers** :
```python
TRAIN_PATH = "prudential-life-insurance-assessment/train.csv/train.csv"
TEST_PATH  = "prudential-life-insurance-assessment/test.csv/test.csv"
```

**Opérations** :
1. Lecture des fichiers CSV (`pd.read_csv`)
2. Affichage des dimensions : `(nb_lignes, nb_colonnes)` pour train et test
3. **Combinaison train + test** en un seul DataFrame `df` pour appliquer les transformations de manière cohérente :
   - Ajout d'une colonne `is_train` (1 = train, 0 = test)
   - Ajout d'une colonne `Response = NaN` au test (il n'a pas de cible)
   - Concaténation verticale avec `pd.concat`
4. Affichage des dimensions du jeu combiné

**Justification de la combinaison** : Toutes les transformations (imputation, encodage) doivent être cohérentes entre train et test. Si on les fait séparément, on risque des incohérences (ex : une catégorie présente dans le test mais pas dans le train).

---

### 2. Suppression des colonnes avec >95% de valeurs manquantes

**Seuil** : `MISSING_THRESHOLD = 0.95` (95%)

#### Étape 2a — Identification des candidates
- Calcul du pourcentage de valeurs manquantes pour chaque feature
- Sélection des colonnes dépassant le seuil de 95%

#### Étape 2b — Vérification avant suppression (analyse du signal)

Avant de supprimer, on vérifie si les quelques valeurs **présentes** dans ces colonnes ont un lien significatif avec `Response`. L'idée : même si une colonne est 97% vide, les 3% restants peuvent contenir un signal prédictif fort.

**Méthode** :
1. Pour chaque colonne candidate, on sépare le train en deux groupes :
   - Lignes **avec** une valeur → calcul de la moyenne de `Response`
   - Lignes **sans** valeur (NaN) → calcul de la moyenne de `Response`
2. On calcule l'écart entre ces deux moyennes
3. **Règle de décision** :
   - Si **< 10 lignes** avec valeur → `DROP` (pas assez de données pour juger)
   - Si **écart > 1.0** point de Response → `KEEP` (signal fort)
   - Sinon → `DROP`

**Affichage** : Tableau récapitulatif avec le % de NaN, le nombre de valeurs présentes, les moyennes de Response (présent/absent), et la décision (KEEP/DROP).

#### Étape 2c — Suppression effective
- Suppression des colonnes marquées `DROP`
- Mise à jour des listes de colonnes (`CATEGORICAL_COLS`, `CONTINUOUS_COLS`, `DISCRETE_COLS`, `DUMMY_COLS`)
- Affichage du nombre de colonnes supprimées et des dimensions après suppression

---

### 3. Imputation des valeurs manquantes

**Stratégie différenciée selon le type** :

| Type de variable | Stratégie d'imputation | Justification |
|:-----------------|:-----------------------|:--------------|
| **Continue** (BMI, Age, Wt, Ht...) | **Médiane** | Robuste aux valeurs extrêmes (mieux que la moyenne pour des distributions asymétriques) |
| **Discrète** (Medical History) | **-1** | L'absence d'information est un signal en soi — le modèle apprend que « manquant » est une catégorie distincte |
| **Catégorielle** (codes médicaux) | **-1** | Même logique : « manquant » = catégorie à part |

**Imputation de sécurité** : Pour tout ce qui reste (colonnes non classifiées) :
- Si type `object` → `"Missing"`
- Sinon → médiane

**Vérification finale** : Comptage des NaN restants (doit être 0).

---

### 4. Encodage des variables textuelles (Label Encoding)

**Problème** : La colonne `Product_Info_2` contient des codes textuels ("A1", "D3", "B2"...). Les modèles de boosting ne peuvent pas traiter directement du texte.

**Solution** : `LabelEncoder` de scikit-learn
- Chaque catégorie textuelle unique reçoit un entier : `"A1" → 0`, `"A2" → 1`, etc.
- Application sur toutes les colonnes de type `object`

**Affichage** : Pour chaque colonne encodée, on affiche le nombre de catégories et la plage de valeurs `[min, max]`.

---

### 5. Target Encoding pour les colonnes à haute cardinalité

**Seuil** : `HIGH_CARDINALITY_THRESHOLD = 20` — une colonne est considérée « haute cardinalité » si elle a plus de 20 catégories uniques.

#### Pourquoi le Target Encoding ?

Le Label Encoding simple assigne des entiers arbitraires (0, 1, 2...) qui n'ont aucun lien avec la variable cible. Pour les colonnes avec beaucoup de catégories, le Target Encoding remplace chaque catégorie par la **moyenne de `Response`** pour cette catégorie, ce qui injecte directement un signal prédictif.

#### Implémentation K-Fold (anti-leakage)

**Problème** : Si on calcule les moyennes sur tout le train, on introduit du data leakage — le modèle « verrait » indirectement la réponse de chaque ligne à travers sa propre catégorie.

**Solution** : Target Encoding avec K-Fold (5 folds) :
1. Découper le train en 5 folds
2. Pour chaque fold de validation : calculer les moyennes sur les 4 autres folds uniquement
3. Pour le test : utiliser les moyennes globales du train entier

```python
def target_encode_kfold(df, col, target, n_splits=5):
```

**Paramètres** :
- `n_splits = 5` folds
- `shuffle = True`, `random_state = 42`
- En cas de catégories absentes → remplacement par la moyenne globale de Response

**Après le target encoding** :
- Les colonnes originales à haute cardinalité sont **supprimées**
- Les nouvelles colonnes `{col}_target_enc` les remplacent

---

### 6. Feature Engineering

Trois nouvelles features sont créées à partir des colonnes existantes :

| Feature | Formule | Justification |
|:--------|:--------|:--------------|
| `Medical_Keyword_Count` | `sum(Medical_Keyword_1 ... Medical_Keyword_48)` | Nombre total de mots-clés médicaux flaggés pour un candidat — proxy de la complexité médicale |
| `BMI_Age` | `BMI × Ins_Age` | Interaction entre l'IMC et l'âge — un IMC élevé est plus risqué chez une personne âgée |
| `Wt_Ht_ratio` | `Wt / (Ht + ε)` | Ratio poids/taille — mesure alternative de la corpulence, différente du BMI |

**Note** : `ε = 1e-8` pour éviter la division par zéro.

---

### 7. Sauvegarde

**Séparation** du DataFrame combiné en train et test :
```python
train_clean = df[df["is_train"] == 1].drop(columns=["is_train"])
test_clean  = df[df["is_train"] == 0].drop(columns=["is_train", TARGET])
```

**Fichiers de sortie** :
- `prudential-life-insurance-assessment/train_clean.csv`
- `prudential-life-insurance-assessment/test_clean.csv`

**Vérifications finales** :
- Dimensions de chaque fichier
- Nombre de NaN restants (doit être 0)
- Nombre de features
- Aperçu des premières lignes (`train_clean.head()`)

---

## Résumé des transformations

| Opération | Détail |
|:----------|:-------|
| Colonnes poubelles | Vérifiées puis supprimées (sauf si signal fort avec Response) |
| Imputation | Médiane (continu), -1 (catégoriel/discret) |
| Encodage | Label Encoding sur `Product_Info_2` |
| Haute cardinalité | Target Encoding K-Fold (5 folds) |
| Feature Engineering | `Medical_Keyword_Count`, `BMI_Age`, `Wt_Ht_ratio` |

---

## Fichiers en entrée / sortie

| Direction | Fichier | Description |
|:----------|:--------|:------------|
| **Entrée** | `prudential-life-insurance-assessment/train.csv/train.csv` | Données brutes d'entraînement |
| **Entrée** | `prudential-life-insurance-assessment/test.csv/test.csv` | Données brutes de test |
| **Sortie** | `prudential-life-insurance-assessment/train_clean.csv` | Données nettoyées d'entraînement |
| **Sortie** | `prudential-life-insurance-assessment/test_clean.csv` | Données nettoyées de test |

---

## Prochaine étape

→ **Partie 2** : Entraînement des modèles XGBoost et CatBoost (`notebook_boosting.ipynb` / `02_boosting.ipynb`)
