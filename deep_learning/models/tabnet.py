from __future__ import annotations

import sys
import numpy as np
from tqdm import tqdm

from config import TabNetConfig
from logging_utils import get_logger
from metrics import qwk
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
            verbose=0,
        )
        print(f"TabNet | device={device} | n_d={config.n_d} | n_steps={config.n_steps} | gamma={config.gamma}", flush=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        from pytorch_tabnet.callbacks import Callback

        cfg = self.config
        outer_model = self.model  # TabNetRegressor, captured for use inside callback
        print(f"TabNet | training max_epochs={cfg.epochs} batch={cfg.batch_size} patience={cfg.early_stopping_patience}...", flush=True)

        ep_width = len(str(cfg.epochs))

        class _TQDMCallback(Callback):
            def on_train_begin(self, logs=None):
                self.ep = 0
                self.pbar = tqdm(
                    total=cfg.epochs,
                    desc="TabNet batches",
                    unit="batch",
                    dynamic_ncols=True,
                    file=sys.stdout,
                    leave=False,
                )

            def on_batch_end(self, batch, logs=None):
                loss = (logs or {}).get("loss", 0.0)
                self.pbar.set_postfix({"loss": f"{loss:.4f}"})
                self.pbar.update(1)

            def on_epoch_end(self, epoch, logs=None):
                self.ep += 1
                self.pbar.reset()
                tr_loss = (logs or {}).get("loss", float("nan"))
                val_loss = (logs or {}).get("val_mse", float("nan"))
                val_preds = outer_model.predict(X_val).squeeze(1)
                val_qwk = qwk(val_preds, y_val)
                epoch_line = (
                    f"Ep {self.ep:>{ep_width}}/{cfg.epochs} | "
                    f"tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"QWK={val_qwk:.4f}"
                )
                tqdm.write(epoch_line, file=sys.stdout)
                logger.info(epoch_line)

            def on_train_end(self, logs=None):
                self.pbar.close()

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
            callbacks=[_TQDMCallback()],
        )

        try:
            val_losses = list(self.model.history["val_mse"])
        except (KeyError, TypeError):
            val_losses = []

        if val_losses:
            best_epoch = int(np.argmin(val_losses)) + 1
            best_val_preds = self.model.predict(X_val).squeeze(1)
            best_val_qwk = qwk(best_val_preds, y_val)
            print(f"TabNet | best epoch={best_epoch} val_loss={min(val_losses):.4f} QWK={best_val_qwk:.4f}", flush=True)
            logger.info(f"Training done. best_epoch={best_epoch} val_loss={min(val_losses):.4f} QWK={best_val_qwk:.4f}")
        else:
            print("TabNet | training done.", flush=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X).squeeze(1)

    def save(self, path: str) -> None:
        self.model.save_model(path)
        print(f"Model saved: {path}.zip", flush=True)

    def load(self, path: str) -> None:
        self.model.load_model(path + ".zip")
        print(f"Model loaded: {path}.zip", flush=True)
