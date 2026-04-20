from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tabm import TabM  # type: ignore[import]
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader
import sys
import warnings
from tqdm import tqdm
from config import TabMConfig
from logging_utils import get_logger
from metrics import qwk
from models.base import BaseTabularModel

warnings.filterwarnings("ignore", message=".*just two bin edges.*", category=UserWarning)

logger = get_logger(__name__)


class _TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray | None = None) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def _make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        _TabularDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        _TabularDataset(X_val, y_val),
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader


def _make_predict_loader(X: np.ndarray, batch_size: int = 2048) -> DataLoader:
    return DataLoader(
        _TabularDataset(X),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


class TabMModel(BaseTabularModel):
    def __init__(self, input_dim: int, config: TabMConfig) -> None:
        self.config = config
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._feature_scaler = StandardScaler()
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self.model: nn.Module | None = None

        print(f"TabM | device={self.device} | k={config.k} | n_blocks={config.n_blocks} | d_block={config.d_block}", flush=True)

    def _build_model(self, X_train_scaled: np.ndarray) -> None:
        cfg = self.config

        num_embeddings = None
        if cfg.use_embeddings:
            X_t = torch.from_numpy(X_train_scaled).float()
            bins = compute_bins(X_t, n_bins=cfg.n_bins)
            num_embeddings = PiecewiseLinearEmbeddings(
                bins,
                d_embedding=cfg.d_embedding,
                activation=False,
                version="B",
            )

        self.model = TabM.make(
            n_num_features=self.input_dim,
            d_out=1,
            num_embeddings=num_embeddings,
            k=cfg.k,
            n_blocks=cfg.n_blocks,
            d_block=cfg.d_block,
            dropout=cfg.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"TabM | params={n_params:,}", flush=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_fit: np.ndarray | None = None,
        y_fit: np.ndarray | None = None,
    ) -> list[dict]:
        cfg = self.config
        X_stats = X_fit if X_fit is not None else X_train
        y_stats = y_fit if y_fit is not None else y_train

        self._feature_scaler.fit(X_stats)
        X_train_s = self._feature_scaler.transform(X_train).astype(np.float32)
        X_val_s = self._feature_scaler.transform(X_val).astype(np.float32)

        self._y_mean = float(y_stats.mean())
        self._y_std = float(y_stats.std()) or 1.0
        y_train_s = ((y_train - self._y_mean) / self._y_std).astype(np.float32)
        y_val_s = ((y_val - self._y_mean) / self._y_std).astype(np.float32)

        X_stats_s = self._feature_scaler.transform(X_stats).astype(np.float32)
        self._build_model(X_stats_s)

        train_loader, val_loader = _make_loaders(
            X_train_s, y_train_s, X_val_s, y_val_s, batch_size=cfg.batch_size
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict = {}
        k = self.model.k
        history: list[dict] = []

        for epoch in range(1, cfg.epochs + 1):
            self.model.train()
            train_loss = 0.0
            batch_pbar = tqdm(
                train_loader,
                desc=f"Ep {epoch:>{len(str(cfg.epochs))}}/{cfg.epochs}",
                unit="batch",
                dynamic_ncols=True,
                file=sys.stdout,
                leave=False,
            )
            for X_batch, y_batch in batch_pbar:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                pred_flat = out.squeeze(-1).flatten(0, 1)
                true_flat = y_batch.repeat_interleave(k)
                loss = loss_fn(pred_flat, true_flat)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            batch_pbar.close()
            train_loss /= len(X_train)
            scheduler.step()

            self.model.eval()
            val_loss = 0.0
            val_preds_list: list[np.ndarray] = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    out = self.model(X_batch)
                    pred_flat = out.squeeze(-1).flatten(0, 1)
                    true_flat = y_batch.repeat_interleave(k)
                    val_loss += loss_fn(pred_flat, true_flat).item() * len(X_batch)
                    preds_mean = out.mean(dim=1).squeeze(-1).cpu().numpy()
                    val_preds_list.append(preds_mean * self._y_std + self._y_mean)
            val_loss /= len(X_val)

            val_preds_denorm = np.concatenate(val_preds_list)
            val_qwk = qwk(val_preds_denorm, y_val)
            lr_cur = scheduler.get_last_lr()[0]
            is_best = val_loss < best_val_loss
            marker = " ★" if is_best else ""

            epoch_line = (
                f"Ep {epoch:>{len(str(cfg.epochs))}}/{cfg.epochs} | "
                f"tr_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"QWK={val_qwk:.4f} | lr={lr_cur:.2e} | "
                f"pat={patience_counter}/{cfg.early_stopping_patience}{marker}"
            )
            tqdm.write(epoch_line, file=sys.stdout)
            logger.info(epoch_line)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_qwk": val_qwk})

            if is_best:
                best_val_loss = val_loss
                best_state = {k_: v.cpu().clone() for k_, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    tqdm.write(f"Early stopping at epoch {epoch}.", file=sys.stdout)
                    logger.info(f"Early stopping at epoch {epoch}.")
                    break

        self.model.load_state_dict(best_state)
        print(f"TabM | best val_loss={best_val_loss:.4f}", flush=True)
        logger.info(f"Training done. best_val_loss={best_val_loss:.4f}")
        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._feature_scaler.transform(X).astype(np.float32)
        loader = _make_predict_loader(X_s)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch)
                preds.append(out.mean(dim=1).squeeze(-1).cpu().numpy())
        raw = np.concatenate(preds)
        return raw * self._y_std + self._y_mean

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "feature_scaler": self._feature_scaler,
                "y_mean": self._y_mean,
                "y_std": self._y_std,
            },
            path,
        )
        print(f"Model saved: {path}", flush=True)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._feature_scaler = ckpt["feature_scaler"]
        self._y_mean = ckpt["y_mean"]
        self._y_std = ckpt["y_std"]
        self.model.load_state_dict(ckpt["state_dict"])
        print(f"Model loaded: {path}", flush=True)
