from __future__ import annotations
from dataclasses import dataclass, field
import os

_CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 8


@dataclass
class DataConfig:
    # datasets: list[str] = field(default_factory=lambda: [
    #     "../prudential-life-insurance-assessment/train.csv/train.csv",
    #     "../prudential-life-insurance-assessment/synthetic/train_augmented_strategy_a.csv",
    #     "../prudential-life-insurance-assessment/synthetic/train_augmented_strategy_b.csv",
    #     "../prudential-life-insurance-assessment/synthetic/train_augmented_ctgan_strategy_a.csv",
    #     "../prudential-life-insurance-assessment/synthetic/train_augmented_ctgan_strategy_b.csv",
    #     "../prudential-life-insurance-assessment/synthetic/gptband0eal.csv",
    #     "../prudential-life-insurance-assessment/synthetic/train_augmented_tvae_strategy_a.csv",
    #     "../prudential-life-insurance-assessment/synthetic/train_augmented_tvae_strategy_b.csv",
    # ])
    datasets: list[str] = field(default_factory=lambda: [
        "../prudential-life-insurance-assessment/train.csv/train.csv",
        "../prudential-life-insurance-assessment/synthetic/train_augmented_strategy_b.csv",
    ])
    results_dir: str = os.path.join(_CURRENT_PATH, "results")
    use_cached: bool = True
    val_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    num_classes: int = NUM_CLASSES


@dataclass
class TabNetConfig:
    n_d: int = 64
    n_a: int = 64
    n_steps: int = 6
    gamma: float = 1.5
    n_independent: int = 2
    n_shared: int = 2
    momentum: float = 0.02
    lr: float = 1e-3
    batch_size: int = 1024
    virtual_batch_size: int = 256
    epochs: int = 200
    early_stopping_patience: int = 20


@dataclass
class TabMConfig:
    k: int = 32
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 3e-4
    batch_size: int = 1024
    epochs: int = 100
    early_stopping_patience: int = 20
    use_embeddings: bool = True
    n_bins: int = 48
    d_embedding: int = 16


MODEL_CONFIGS: dict[str, object] = {
    "tabnet": TabNetConfig(),
    "tabm": TabMConfig(),
}
