from __future__ import annotations

import numpy as np

from config import TabNetConfig
from logging_utils import get_logger
from models.base import BaseTabularModel

logger = get_logger(__name__)


class TabNetModel(BaseTabularModel):
    def __init__(self, input_dim: int, config: TabNetConfig) -> None:
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor

        self.config = config
        self.input_dim = input_dim
        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = TabNetRegressor(
            n_d=config.n_d,
            n_a=config.n_a,
            n_steps=config.n_steps,
            gamma=config.gamma,
            n_independent=config.n_independent,
            n_shared=config.n_shared,
            momentum=config.momentum,
            optimizer_fn=torch.optim.Adam,
            optimizer_params={"lr": config.lr},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            mask_type="entmax",
            device_name=device,
            # On désactive les prints internes; on logue l'historique après fit()
            verbose=0,
        )
        logger.info(
            f"TabNetModel | device={device} | n_d={config.n_d} | "
            f"n_steps={config.n_steps} | gamma={config.gamma}"
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        cfg = self.config
        logger.info(
            f"Entraînement | max_epochs={cfg.epochs} | batch={cfg.batch_size} | "
            f"lr={cfg.lr} | patience={cfg.early_stopping_patience}"
        )

        self.model.fit(
            X_train=X_train,
            y_train=y_train.reshape(-1, 1),
            eval_set=[(X_val, y_val.reshape(-1, 1))],
            eval_name=["val"],
            eval_metric=["mse"],
            max_epochs=cfg.epochs,
            patience=cfg.early_stopping_patience,
            batch_size=cfg.batch_size,
            virtual_batch_size=cfg.virtual_batch_size,
            drop_last=False,
        )

        try:
            train_losses = list(self.model.history["loss"])
            val_losses = list(self.model.history["val_mse"])
        except (KeyError, TypeError):
            train_losses, val_losses = [], []

        best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else 0

        for epoch_idx, (tl, vl) in enumerate(zip(train_losses, val_losses), start=1):
            marker = " ★" if epoch_idx == best_epoch else ""
            logger.info(
                f"Époque {epoch_idx:4d}/{cfg.epochs} | "
                f"train_loss={tl:.4f} | "
                f"val_mse={vl:.4f}"
                f"{marker}"
            )

        if val_losses:
            logger.info(f"Entraînement terminé. Meilleure époque={best_epoch} | val_mse={min(val_losses):.4f}")
        else:
            logger.info("Entraînement terminé.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).squeeze(1)

    def save(self, path: str) -> None:
        # pytorch-tabnet sauvegarde une archive zip; le chemin ne doit pas inclure l'extension
        self.model.save_model(path)
        logger.info(f"Modèle sauvegardé dans {path}.zip")

    def load(self, path: str) -> None:
        self.model.load_model(path + ".zip")
        logger.info(f"Modèle chargé depuis {path}.zip")
