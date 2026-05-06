"""
Partie 4 — Optimisation & Métriques
Configuration des variantes d'ablation TabM sur le dataset Prudential Life Insurance.

Variantes testées (inspirées de l'article TabM et du dépôt yandex-research/tabm) :
  1. mlp_plain         — MLP de base, aucune approche d'ensemble
  2. tabm_packed       — Ensemble Simple sans weight sharing (TabM-packed)
  3. tabm_mini         — Mini Ensemble : un seul adapteur R (TabM-mini)
  4. tabm              — Batch Ensemble complet : adapteurs R, S et B (TabM)
  5. tabm_mini_embed   — TabM-mini + feature embedding PiecewiseLinear
  6. tabm_embed        — TabM      + feature embedding PiecewiseLinear  (meilleur attendu)
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DL_BRANCH = os.path.join(
    os.path.dirname(_HERE),
    "branches", "deep_learning",
    "prudential-life-insurance-assessment",
)

NUM_CLASSES = 8

TRAIN_PATH = os.path.join(_DL_BRANCH, "train_clean.csv")
TEST_PATH  = os.path.join(_DL_BRANCH, "test_clean.csv")
RESULTS_DIR = os.path.join(_HERE, "results")


# ---------------------------------------------------------------------------
# Config globale
# ---------------------------------------------------------------------------
@dataclass
class GlobalConfig:
    train_path: str = TRAIN_PATH
    test_path: str  = TEST_PATH
    results_dir: str = RESULTS_DIR
    random_state: int = 42
    test_size: float = 0.2
    num_classes: int = NUM_CLASSES


# ---------------------------------------------------------------------------
# Config commune à toutes les variantes TabM / MLP
# ---------------------------------------------------------------------------
@dataclass
class AblationVariantConfig:
    # Identifiant de la variante
    variant_id: str = "tabm"

    # Architecture TabM : 'plain' | 'tabm-packed' | 'tabm-mini' | 'tabm'
    # 'plain' déclenche l'utilisation du PlainMLP (pas de TabM)
    arch_type: str = "tabm"

    # Taille de l'ensemble  (ignoré pour 'plain')
    k: int = 32

    # Hyperparamètres MLP partagés
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.25

    # Optimisation
    lr: float = 1e-3
    weight_decay: float = 1e-3
    batch_size: int = 1024
    epochs: int = 50
    early_stopping_patience: int = 10

    # Feature embeddings PiecewiseLinear (activé uniquement pour les variantes _embed)
    use_embeddings: bool = False
    n_bins: int = 48
    d_embedding: int = 16

    # Label pour les graphiques
    label: str = ""
    # Description narrative (pour README / W&B notes)
    description: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.variant_id


# ---------------------------------------------------------------------------
# Définition des 6 variantes d'ablation
# ---------------------------------------------------------------------------
ABLATION_VARIANTS: list[AblationVariantConfig] = [

    # 1 — MLP de base (sans ensemble)
    AblationVariantConfig(
        variant_id="mlp_plain",
        arch_type="plain",
        k=1,            # Sans objet pour plain, mis à 1 pour la cohérence de l'API
        n_blocks=3,
        d_block=512,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        use_embeddings=False,
        label="MLP (plain)",
        description=(
            "MLP standard sans aucune approche d'ensemble. "
            "Sert de baseline inférieure. "
            "Équivalent au 'plain' de l'article TabM."
        ),
    ),

    # 2 — Ensemble Simple sans weight sharing (TabM-packed)
    AblationVariantConfig(
        variant_id="tabm_packed",
        arch_type="tabm-packed",
        k=32,
        n_blocks=3,
        d_block=512,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        use_embeddings=False,
        label="Ensemble sans weight sharing (TabM-packed)",
        description=(
            "Ensemble de k MLPs complètement indépendants (Packed Ensemble). "
            "Correspond au TabM-packed de l'article. "
            "Aucun partage de poids entre les membres → plus lourd, souvent moins bon."
        ),
    ),

    # 3 — Mini Ensemble (TabM-mini) : un seul adapteur R
    AblationVariantConfig(
        variant_id="tabm_mini",
        arch_type="tabm-mini",
        k=32,
        n_blocks=3,
        d_block=512,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        use_embeddings=False,
        label="TabM-mini (un seul adapteur R)",
        description=(
            "MiniEnsemble : k représentations différentes créées par k transformations "
            "affines non partagées (adapteur R uniquement) puis propagées dans un MLP partagé. "
            "Correspond à TabM_mini de l'article."
        ),
    ),

    # 4 — Batch Ensemble complet (TabM) : adapteurs R, S et B
    AblationVariantConfig(
        variant_id="tabm",
        arch_type="tabm",
        k=32,
        n_blocks=3,
        d_block=512,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        use_embeddings=False,
        label="TabM (BatchEnsemble R+S+B)",
        description=(
            "TabM complet : adapteurs R (scaling input), S (scaling output) et B (bias) "
            "appliqués à chaque couche linéaire, avec weight sharing. "
            "Correspond à TabM de l'article."
        ),
    ),

    # 5 — TabM-mini + PiecewiseLinear embeddings
    AblationVariantConfig(
        variant_id="tabm_mini_embed",
        arch_type="tabm-mini",
        k=32,
        n_blocks=2,     # L'article recommande n_blocks=2 avec embeddings
        d_block=512,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        use_embeddings=True,
        n_bins=48,
        d_embedding=16,
        label="TabM-mini† (+ PiecewiseLinear embedding)",
        description=(
            "TabM-mini avec feature embeddings PiecewiseLinear. "
            "Noté TabM†_mini dans l'article. "
            "Le nombre de blocs est réduit à 2 comme recommandé dans l'article."
        ),
    ),

    # 6 — TabM + PiecewiseLinear embeddings (meilleur résultat attendu)
    AblationVariantConfig(
        variant_id="tabm_embed",
        arch_type="tabm",
        k=32,
        n_blocks=2,     # L'article recommande n_blocks=2 avec embeddings
        d_block=512,
        dropout=0.25,
        lr=1e-3,
        weight_decay=1e-3,
        use_embeddings=True,
        n_bins=48,
        d_embedding=16,
        label="TabM† (+ PiecewiseLinear embedding)",
        description=(
            "TabM complet avec feature embeddings PiecewiseLinear. "
            "Noté TabM† dans l'article, c'est la variante qui obtient les meilleurs résultats. "
            "Le nombre de blocs est réduit à 2 comme recommandé dans l'article."
        ),
    ),
]

# Dictionnaire pour accès par id
VARIANTS_BY_ID: dict[str, AblationVariantConfig] = {v.variant_id: v for v in ABLATION_VARIANTS}
