from __future__ import annotations

import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from config import TabNetConfig
from logging_utils import get_logger
from metrics import qwk
from models.base import BaseTabularModel
from results_utils import save_confusion_matrix_plot

logger = get_logger(__name__)


class TabNetModel(BaseTabularModel):
    def __init__(self, input_dim: int, config: TabNetConfig) -> None:
        import torch
        from pytorch_tabnet.tab_model import TabNetRegressor

        self.config = config
        self.input_dim = input_dim
        self._feature_scaler = StandardScaler()
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._has_preprocessing = False
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
        X_fit: np.ndarray | None = None,
        y_fit: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        run_dir: str | None = None,
    ) -> list[dict]:
        from pytorch_tabnet.callbacks import Callback

        cfg = self.config
        X_stats = X_fit if X_fit is not None else X_train
        y_stats = y_fit if y_fit is not None else y_train

        self._feature_scaler.fit(X_stats)
        self._has_preprocessing = True
        X_train_s = self._feature_scaler.transform(X_train).astype(np.float32)
        X_val_s = self._feature_scaler.transform(X_val).astype(np.float32)

        self._y_mean = float(y_stats.mean())
        self._y_std = float(y_stats.std()) or 1.0
        y_train_s = ((y_train - self._y_mean) / self._y_std).astype(np.float32)
        y_val_s = ((y_val - self._y_mean) / self._y_std).astype(np.float32)

        if sample_weight is not None and len(sample_weight) != len(X_train):
            raise ValueError("sample_weight must have the same length as X_train")
        train_weights = sample_weight.astype(np.float32) if sample_weight is not None else 0

        outer_model = self.model  # TabNetRegressor, captured for use inside callback
        outer_wrapper = self
        print(f"TabNet | training max_epochs={cfg.epochs} batch={cfg.batch_size} patience={cfg.early_stopping_patience}...", flush=True)

        ep_width = len(str(cfg.epochs))
        history: list[dict] = []

        class _TQDMCallback(Callback):
            def on_train_begin(self, logs=None):
                self.ep = 0
                self.best_val_qwk = -float("inf")
                self.best_qwk_epoch = 0
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
                val_preds_s = outer_model.predict(X_val_s).squeeze(1)
                val_preds = val_preds_s * outer_wrapper._y_std + outer_wrapper._y_mean
                val_qwk = qwk(val_preds, y_val)
                is_best = val_qwk > self.best_val_qwk
                marker = " *" if is_best else ""
                epoch_line = (
                    f"Ep {self.ep:>{ep_width}}/{cfg.epochs} | "
                    f"tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | "
                    f"QWK={val_qwk:.4f}{marker}"
                )
                tqdm.write(epoch_line, file=sys.stdout)
                logger.info(epoch_line)
                history.append({"epoch": self.ep, "train_loss": tr_loss, "val_loss": val_loss, "val_qwk": val_qwk})

                if is_best:
                    self.best_val_qwk = val_qwk
                    self.best_qwk_epoch = self.ep
                    if run_dir is not None:
                        checkpoint_path = os.path.join(run_dir, "best_model")
                        outer_wrapper.save(checkpoint_path)
                        save_confusion_matrix_plot(
                            run_dir,
                            val_preds,
                            y_val,
                            title=f"Validation confusion matrix - epoch {self.ep} QWK {val_qwk:.4f}",
                        )
                        logger.info(f"Best checkpoint saved. epoch={self.ep} val_qwk={val_qwk:.4f}")

            def on_train_end(self, logs=None):
                self.pbar.close()

        checkpoint_callback = _TQDMCallback()
        self.model.fit(
            X_train=X_train_s,
            y_train=y_train_s.reshape(-1, 1),
            eval_set=[(X_val_s, y_val_s.reshape(-1, 1))],
            eval_name=["val"],
            eval_metric=["mse"],
            weights=train_weights,
            max_epochs=cfg.epochs,
            patience=cfg.early_stopping_patience,
            batch_size=cfg.batch_size,
            virtual_batch_size=cfg.virtual_batch_size,
            drop_last=False,
            callbacks=[checkpoint_callback],
        )

        try:
            val_losses = list(self.model.history["val_mse"])
        except (KeyError, TypeError):
            val_losses = []

        if val_losses:
            best_epoch = int(np.argmin(val_losses)) + 1
            if run_dir is not None and os.path.exists(os.path.join(run_dir, "best_model.zip")):
                self.load(os.path.join(run_dir, "best_model"))
            best_val_preds = self.predict(X_val)
            best_val_qwk = qwk(best_val_preds, y_val)
            best_qwk_epoch = getattr(checkpoint_callback, "best_qwk_epoch", 0)
            best_saved_qwk = getattr(checkpoint_callback, "best_val_qwk", float("nan"))
            print(
                f"TabNet | best loss epoch={best_epoch} val_loss={min(val_losses):.4f} | "
                f"best QWK epoch={best_qwk_epoch} QWK={best_saved_qwk:.4f}",
                flush=True,
            )
            logger.info(
                f"Training done. best_loss_epoch={best_epoch} val_loss={min(val_losses):.4f} "
                f"best_qwk_epoch={best_qwk_epoch} best_val_qwk={best_saved_qwk:.4f} "
                f"final_model_qwk={best_val_qwk:.4f}"
            )
        else:
            print("TabNet | training done.", flush=True)
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_in = X.astype(np.float32)
        if self._has_preprocessing:
            X_in = self._feature_scaler.transform(X).astype(np.float32)
        preds = self.model.predict(X_in).squeeze(1)
        return preds * self._y_std + self._y_mean if self._has_preprocessing else preds

    def save(self, path: str) -> None:
        checkpoint_zip = f"{path}.zip"
        if os.path.exists(checkpoint_zip):
            os.remove(checkpoint_zip)
        if self._has_preprocessing:
            np.savez(
                f"{path}_preprocess.npz",
                feature_mean=self._feature_scaler.mean_,
                feature_scale=self._feature_scaler.scale_,
                feature_var=self._feature_scaler.var_,
                n_features_in=np.array(self._feature_scaler.n_features_in_),
                n_samples_seen=np.asarray(self._feature_scaler.n_samples_seen_),
                y_mean=np.array(self._y_mean),
                y_std=np.array(self._y_std),
            )
        self.model.save_model(path)
        print(f"Model saved: {path}.zip", flush=True)

    def load(self, path: str) -> None:
        base_path = path[:-4] if path.endswith(".zip") else path
        self.model.load_model(base_path + ".zip")
        preprocess_path = f"{base_path}_preprocess.npz"
        if os.path.exists(preprocess_path):
            data = np.load(preprocess_path)
            scaler = StandardScaler()
            scaler.mean_ = data["feature_mean"]
            scaler.scale_ = data["feature_scale"]
            scaler.var_ = data["feature_var"]
            scaler.n_features_in_ = int(data["n_features_in"])
            n_samples_seen = data["n_samples_seen"]
            scaler.n_samples_seen_ = int(n_samples_seen) if n_samples_seen.ndim == 0 else n_samples_seen
            self._feature_scaler = scaler
            self._y_mean = float(data["y_mean"])
            self._y_std = float(data["y_std"])
            self._has_preprocessing = True
        else:
            self._has_preprocessing = False
            self._y_mean = 0.0
            self._y_std = 1.0
        print(f"Model loaded: {base_path}.zip", flush=True)
