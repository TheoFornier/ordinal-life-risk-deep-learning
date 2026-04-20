from __future__ import annotations
from dataclasses import dataclass
import os

_CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
_DATASET_PATH = os.path.join(_CURRENT_PATH, "..", "prudential-life-insurance-assessment")

NUM_CLASSES = 8


@dataclass
class DataConfig:
    train_raw: str = os.path.join(_DATASET_PATH, "train.csv", "train.csv")
    test_raw: str = os.path.join(_DATASET_PATH, "test.csv", "test.csv")
    train_clean: str = os.path.join(_DATASET_PATH, "train_clean.csv")
    test_clean: str = os.path.join(_DATASET_PATH, "test_clean.csv")
    results_dir: str = os.path.join(_CURRENT_PATH, "results")
    use_cached: bool = True
    val_size: float = 0.2
    random_state: int = 42
    num_classes: int = NUM_CLASSES


@dataclass
class TabNetConfig:
    n_d: int = 32
    n_a: int = 32
    n_steps: int = 5
    gamma: float = 1.5
    n_independent: int = 2
    n_shared: int = 2
    momentum: float = 0.02
    lr: float = 2e-3
    batch_size: int = 1024
    virtual_batch_size: int = 256
    epochs: int = 5
    early_stopping_patience: int = 15


@dataclass
class TabMConfig:
    k: int = 32
    n_blocks: int = 2
    d_block: int = 512
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 3e-4
    batch_size: int = 1024
    epochs: int = 5
    early_stopping_patience: int = 15
    use_embeddings: bool = True
    n_bins: int = 48
    d_embedding: int = 16


MODEL_CONFIGS: dict[str, object] = {
    "tabnet": TabNetConfig(),
    "tabm": TabMConfig(),
}
