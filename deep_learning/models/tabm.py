from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tabm import TabM  # type: ignore[import]
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader
from config import TabMConfig
from logging_utils import get_logger
from metrics import qwk
from models.base import BaseTabularModel

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

        # Construction différée : les bins PLE nécessitent X_train
        self.model: nn.Module | None = None

        logger.info(
            f"TabMModel | device={self.device} | k={config.k} | "
            f"n_blocks={config.n_blocks} | d_block={config.d_block} | "
            f"dropout={config.dropout} | embeddings={config.use_embeddings}"
        )

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
        logger.info(f"Modèle construit | paramètres={n_params:,}")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> None:
        cfg = self.config

        # Normalisation des features
        X_train_s = self._feature_scaler.fit_transform(X_train).astype(np.float32)
        X_val_s = self._feature_scaler.transform(X_val).astype(np.float32)

        # Normalisation de la cible
        self._y_mean = float(y_train.mean())
        self._y_std = float(y_train.std()) or 1.0
        y_train_s = ((y_train - self._y_mean) / self._y_std).astype(np.float32)
        y_val_s = ((y_val - self._y_mean) / self._y_std).astype(np.float32)

        # Construction du modèle maintenant qu'on a les données pour les bins PLE
        self._build_model(X_train_s)

        train_loader, val_loader = _make_loaders(
            X_train_s, y_train_s, X_val_s, y_val_s, batch_size=cfg.batch_size
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs
        )
        loss_fn = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict = {}
        k = self.model.k

        logger.info(
            f"Entraînement | epochs={cfg.epochs} | batch={cfg.batch_size} | "
            f"lr={cfg.lr} | weight_decay={cfg.weight_decay} | "
            f"patience={cfg.early_stopping_patience}"
        )

        for epoch in range(1, cfg.epochs + 1):
            # Entraînement
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                out = self.model(X_batch)                        # (B, k, 1)
                pred_flat = out.squeeze(-1).flatten(0, 1)        # (B*k,)
                true_flat = y_batch.repeat_interleave(k)         # (B*k,)
                loss = loss_fn(pred_flat, true_flat)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item() * len(X_batch)
            train_loss /= len(X_train)
            scheduler.step()

            # Validation
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
                    # Moyenne des k sous-modèles, dénormalisée
                    preds_mean = out.mean(dim=1).squeeze(-1).cpu().numpy()
                    val_preds_list.append(preds_mean * self._y_std + self._y_mean)
            val_loss /= len(X_val)

            val_preds_denorm = np.concatenate(val_preds_list)
            val_qwk = qwk(val_preds_denorm, y_val)

            is_best = val_loss < best_val_loss
            marker = " ★" if is_best else ""
            lr_cur = scheduler.get_last_lr()[0]

            logger.info(
                f"Époque {epoch:4d}/{cfg.epochs} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"val_QWK={val_qwk:.4f} | "
                f"lr={lr_cur:.2e} | "
                f"patience={patience_counter}/{cfg.early_stopping_patience}"
                f"{marker}"
            )

            if is_best:
                best_val_loss = val_loss
                best_state = {k_: v.cpu().clone() for k_, v in self.model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.early_stopping_patience:
                    logger.info(f"Arrêt anticipé déclenché à l'époque {epoch}.")
                    break

        self.model.load_state_dict(best_state)
        logger.info(f"Entraînement terminé. Meilleure val_loss={best_val_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._feature_scaler.transform(X).astype(np.float32)
        loader = _make_predict_loader(X_s)
        self.model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_batch in loader:
                X_batch = X_batch.to(self.device)
                out = self.model(X_batch)                        # (B, k, 1)
                preds.append(out.mean(dim=1).squeeze(-1).cpu().numpy())
        raw = np.concatenate(preds)
        return raw * self._y_std + self._y_mean                  # dénormalisation

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
        logger.info(f"Modèle sauvegardé dans {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._feature_scaler = ckpt["feature_scaler"]
        self._y_mean = ckpt["y_mean"]
        self._y_std = ckpt["y_std"]
        self.model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Modèle chargé depuis {path}")
