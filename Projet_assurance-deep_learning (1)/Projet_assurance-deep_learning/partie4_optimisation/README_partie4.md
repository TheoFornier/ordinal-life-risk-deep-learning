# Partie 4 — Optimisation & Métriques

## Vue d'ensemble

Cette partie implémente l'**étude d'ablation** et l'**optimisation d'hyperparamètres** pour les modèles TabM sur le dataset Prudential Life Insurance.

### Ce qui est fait ici

| Composant | Fichier | Description |
|---|---|---|
| Configuration | `config_ablation.py` | 6 variantes d'ablation définies comme dataclasses |
| Modèles | `model_ablation.py` | `PlainMLPModel` + `AblationTabMModel` (wrappers avec fit/predict) |
| Métriques | `metrics_ablation.py` | QWK, accuracy, optimisation d'offsets |
| Ablation | `run_ablation.py` | Lance les 6 variantes + log W&B |
| Sweep | `sweep_tabm.py` | Recherche bayésienne d'hyperparamètres W&B |
| Comparaison | `compare_ablation.py` | Graphiques + tableau HTML vs article |

---

## Installation

```bash
cd partie4_optimisation
pip install -r requirements_partie4.txt
```

---

## Utilisation

### 1. Lancer l'étude d'ablation complète

```bash
python run_ablation.py
```

Cela lance les **6 variantes** dans l'ordre suivant :
1. `mlp_plain` — MLP de base (baseline inférieure)
2. `tabm_packed` — Ensemble Simple sans weight sharing
3. `tabm_mini` — Mini Ensemble (1 adapteur R)
4. `tabm` — Batch Ensemble complet (R + S + B)
5. `tabm_mini_embed` — TabM-mini + PiecewiseLinear embeddings
6. `tabm_embed` — TabM + PiecewiseLinear embeddings ← **meilleur attendu**

**Options utiles :**
```bash
# Lancer uniquement certaines variantes
python run_ablation.py --variants tabm tabm_embed

# Sans W&B (résultats locaux uniquement)
python run_ablation.py --no-wandb

# Réduire les époques pour un test rapide
python run_ablation.py --epochs 5 --no-wandb

# Spécifier le projet W&B
python run_ablation.py --wandb-project mon-projet
```

Lors du premier lancement avec W&B, le script demande :
```
Email W&B        : votre@email.com
Mot de passe     : ****
Clé API W&B      : ****  (disponible sur https://wandb.ai/authorize)
```

Vous pouvez aussi pré-définir la clé :
```bash
export WANDB_API_KEY=votre_cle_api
python run_ablation.py
```

### 2. Comparer les résultats

```bash
python compare_ablation.py
```

Génère dans `results/` :
- `comparison_barplot.png` — barres QWK raw vs QWK+offsets
- `comparison_progression.png` — courbe de progression
- `comparison_table.html` — tableau HTML pour le rapport

### 3. Lancer le sweep W&B (recherche d'hyperparamètres)

```bash
# Exploration large (tabm + tabm-mini, 30 runs bayésiens)
python sweep_tabm.py --count 30

# Focus TabM† avec embeddings (50 runs)
python sweep_tabm.py --arch tabm_embed --count 50

# Projet W&B personnalisé
python sweep_tabm.py --project prudential-tabm --count 40
```

---

## Les 6 variantes en détail

### 1. MLP plain — `mlp_plain`
MLP standard PyTorch sans aucune approche d'ensemble.
- Architecture : Linear → ReLU → Dropout × n_blocks → Linear
- Sert de **baseline inférieure** pour mesurer l'apport de l'ensembling.
- Équivalent au "plain" de l'article TabM.

### 2. Ensemble Simple sans weight sharing — `tabm_packed`
`TabM.make(..., arch_type='tabm-packed')` — k MLPs complètement indépendants (Packed Ensemble).
- Correspondance article : **TabM-packed**
- ⚠ Plus lourd en mémoire et généralement moins performant que TabM.
- Équivalent au PackedEnsemble(alpha=k, M=k, gamma=1) du papier Packed Ensembles.

### 3. Mini Ensemble — `tabm_mini`
`TabM.make(..., arch_type='tabm-mini')` — un seul adapteur **R** (scaling input).
- Correspondance article : **TabM-mini**
- Toutes les couches sauf la première sont partagées → très efficace.
- Bon rapport performance / vitesse d'entraînement.

### 4. Batch Ensemble — `tabm`
`TabM.make(..., arch_type='tabm')` — adapteurs **R** (input), **S** (output), **B** (bias).
- Correspondance article : **TabM**
- Meilleur que MLP et packed sur la plupart des benchmarks de l'article.
- Formule BatchEnsemble : `out = (x × R) @ W × S + B`

### 5. TabM-mini + PiecewiseLinear — `tabm_mini_embed`
TabM-mini avec feature embeddings PiecewiseLinear.
- Correspondance article : **TabM†-mini**
- n_blocks réduit à 2 (recommandé dans l'article quand on utilise des embeddings).
- Amélioration notable par rapport à TabM-mini sans embeddings.

### 6. TabM + PiecewiseLinear — `tabm_embed`
TabM complet avec feature embeddings PiecewiseLinear.
- Correspondance article : **TabM†**
- **Meilleur résultat attendu** — variante recommandée pour la production.
- n_blocks réduit à 2 (recommandé dans l'article).

---

## Métriques

### QWK — Quadratic Weighted Kappa
Métrique principale du concours Kaggle Prudential Life Insurance Assessment.

$$\text{QWK} = 1 - \frac{\sum_{i,j} W_{ij} O_{ij}}{\sum_{i,j} W_{ij} E_{ij}}$$

où $W_{ij} = \frac{(i-j)^2}{(N-1)^2}$ est la matrice de pondération quadratique.

### Optimisation d'offsets
Après entraînement, un offset par classe est optimisé pour maximiser le QWK sur la validation. Cette technique est standard sur ce dataset (utilisée dans toutes les solutions du top Kaggle).

---

## Structure des résultats

Chaque run crée un sous-dossier dans `results/` :
```
results/
├── 20260428_143000_mlp_plain/
│   ├── result.json       ← métriques finales
│   └── curves.png        ← courbes loss + QWK
├── 20260428_143500_tabm_packed/
│   ├── result.json
│   └── curves.png
├── ...
├── ablation_summary.csv  ← tableau récapitulatif
├── comparison_barplot.png
├── comparison_progression.png
└── comparison_table.html
```

---

## Correspondance avec l'article TabM

| Notre variante | Notation article | Rang moyen (benchmarks article) |
|---|---|---|
| `mlp_plain` | MLP | ~6e sur 7 méthodes |
| `tabm_packed` | TabM-packed | ~5e |
| `tabm_mini` | TabM-mini | ~3e |
| `tabm` | TabM | ~3e (souvent meilleur que Mini) |
| `tabm_mini_embed` | TabM†-mini | ~2e |
| `tabm_embed` | TabM† | **~1er** |

Source : Table 2, arXiv:2410.00203 — *TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling*

---

## Notes importantes

- **Prudential n'est pas dans les benchmarks de l'article** → les rangs ci-dessus sont indicatifs.
- **Durée estimée par run** (GPU) : ~5–15 min pour tabm_embed avec 50 époques.
- **Durée estimée par run** (CPU) : ~30–90 min — réduire les époques avec `--epochs 10`.
- Le dataset `train_clean.csv` est lu depuis la branche `deep_learning` (dossier `branches/`).
