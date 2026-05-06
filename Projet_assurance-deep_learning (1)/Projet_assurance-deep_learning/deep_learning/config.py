from __future__ import annotations
from dataclasses import dataclass, field
import os

_CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

NUM_CLASSES = 8


@dataclass
class DataConfig:
    model: str = "tabm"  # "tabm" or "tabnet"
    run_base_training: bool = False
    use_class_weights: bool = False  # inverse-frequency weight real samples by class (redundant with soft_qwk)
    class_weight_power: float = 0.7  # exponent on inverse-freq weights: 0=none, 0.5=sqrt, 1.0=full
    synth_sample_weight: float = 0.2  # loss weight for synthetic rows (1.0 = same as real)
    max_synth_ratio: float = 1.0  # max synthetic rows as a multiple of real train rows (e.g. 0.5 = half as many)
    max_synth_val_ratio: float = 0.0  # max synthetic rows added to val set as a multiple of real val rows (0.0 = disabled)
    use_kfold: bool = True
    n_folds: int = 5
    train_path: str = "../prudential-life-insurance-assessment/train_clean.csv"
    # synthetic_datasets: list[str] = field(default_factory=lambda: [
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/train_augmented_ctgan_strategy_a",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/train_augmented_ctgan_strategy_b",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/train_augmented_strategy_a",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/train_augmented_strategy_b",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/train_augmented_tvae_strategy_a",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/train_augmented_tvae_strategy_b",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/gptband0eal",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/gpt_1",
    #     "../prudential-life-insurance-assessment/synthetic/cleaned/gpt_2",
    # ])
    synthetic_datasets: list[str] = field(default_factory=lambda: [
        "../prudential-life-insurance-assessment/synthetic/cleaned/synthetic_pseudo_labeled_filtered_balanced",
    ])
    test_path: str = "../prudential-life-insurance-assessment/test_clean.csv"
    results_dir: str = os.path.join(_CURRENT_PATH, "results")
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
    early_stopping_patience: int = 7


@dataclass
class TabMConfig:
    k: int = 32
    n_blocks: int = 3
    d_block: int = 512
    dropout: float = 0.25
    lr: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 512
    epochs: int = 100
    early_stopping_patience: int = 7
    use_embeddings: bool = True
    n_bins: int = 48
    d_embedding: int = 16
    loss_fn: str = "soft_qwk"  # "mse" or "soft_qwk"
    qwk_temperature: float = 0.5


MODEL_CONFIGS: dict[str, object] = {
    "tabnet": TabNetConfig(),
    "tabm": TabMConfig(),
}
