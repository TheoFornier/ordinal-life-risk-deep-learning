"""
Partie 4 — Modèles d'ablation.

Deux wrappers avec la même interface (fit / predict) :

  PlainMLPModel        → MLP standard PyTorch, aucun ensemble
  AblationTabMModel    → TabM.make(..., arch_type=...)
                         Couvre tabm-packed, tabm-mini, tabm

Les deux héritent de la même boucle d'entraînement interne et normalisent
features (StandardScaler) et cible (mean/std), exactement comme le fait le
code existant dans la branche deep_learning.
"""
from __future__ import annotations

import sys
import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from config_ablation import AblationVariantConfig, NUM_CLASSES
from metrics_ablation import qwk

warnings.filterwarnings("ignore", message=".*just two bin edges.*", category=UserWarning)


# ===========================================================================
# Dataset interne
# ===========================================================================
class _TabDS(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def _loaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    pin = torch.cuda.is_available()
    train_dl = DataLoader(
        _TabDS(X_tr, y_tr), batch_size=batch_size, shuffle=True, pin_memory=pin
    )
    val_dl = DataLoader(
        _TabDS(X_va, y_va), batch_size=batch_size * 2, shuffle=False, pin_memory=pin
    )
    return train_dl, val_dl


def _pred_loader(X: np.ndarray, batch_size: int = 2048) -> DataLoader:
    return DataLoader(
        _TabDS(X), batch_size=batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(),
    )


# ===========================================================================
# Boucle d'entraînement partagée
# ===========================================================================
def _train_loop(
    model_nn: nn.Module,
    train_dl: DataLoader,
    val_dl: DataLoader,
    cfg: AblationVariantConfig,
    device: torch.device,
    y_val_original: np.ndarray,
    y_std: float,
    y_mean: float,
    k: int,
) -> list[dict]:
    """
    Entraîne model_nn et retourne l'historique [{epoch, train_loss, val_loss, val_qwk}].
    Le modèle doit produire un tensor de shape (B, k, 1) ou (B, 1) pour plain (k=1).
    """
    optimizer = torch.optim.AdamW(
        model_nn.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.MSELoss(reduction="mean")

    best_val_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0
    history: list[dict] = []
    ep_w = len(str(cfg.epochs))

    for epoch in range(1, cfg.epochs + 1):
        # ---- train ----
        model_nn.train()
        tr_loss = 0.0
        for X_b, y_b in train_dl:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model_nn(X_b)               # (B, k, 1)  ou (B, 1) pour plain
            # Aplatir sur les k membres pour la loss
            pred_flat = out.reshape(-1)
            true_flat = y_b.repeat_interleave(k)
            loss = loss_fn(pred_flat, true_flat)
            loss.backward()
            nn.utils.clip_grad_norm_(model_nn.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(X_b)
        tr_loss /= len(train_dl.dataset)
        scheduler.step()

        # ---- val ----
        model_nn.eval()
        va_loss = 0.0
        preds_list: list[np.ndarray] = []
        with torch.no_grad():
            for X_b, y_b in val_dl:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model_nn(X_b)           # (B, k, 1)
                pred_flat = out.reshape(-1)
                true_flat = y_b.repeat_interleave(k)
                va_loss += loss_fn(pred_flat, true_flat).item() * len(X_b)
                # Moyenne sur les k membres → (B,)
                preds_list.append(out.mean(dim=1).squeeze(-1).cpu().numpy())
        va_loss /= len(val_dl.dataset)

        val_preds_denorm = np.concatenate(preds_list) * y_std + y_mean
        val_qwk = qwk(val_preds_denorm, y_val_original)

        lr_cur = scheduler.get_last_lr()[0]
        is_best = va_loss < best_val_loss
        marker = " ★" if is_best else ""
        line = (
            f"Ep {epoch:>{ep_w}}/{cfg.epochs} | "
            f"tr={tr_loss:.4f} | va={va_loss:.4f} | "
            f"QWK={val_qwk:.4f} | lr={lr_cur:.2e} | "
            f"pat={patience_counter}/{cfg.early_stopping_patience}{marker}"
        )
        tqdm.write(line, file=sys.stdout)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_qwk": val_qwk})

        if is_best:
            best_val_loss = va_loss
            best_state = {k_: v.cpu().clone() for k_, v in model_nn.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                tqdm.write(f"Early stopping à l'époque {epoch}.", file=sys.stdout)
                break

    model_nn.load_state_dict(best_state)
    print(f"Best val_loss={best_val_loss:.4f}", flush=True)
    return history


# ===========================================================================
# 1. PlainMLPModel
# ===========================================================================
class _PlainMLP(nn.Module):
    """MLP standard PyTorch.  Sortie : (B, 1, 1) pour cohérence avec l'API k."""

    def __init__(self, d_in: int, n_blocks: int, d_block: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = d_in
        for _ in range(n_blocks):
            layers += [nn.Linear(in_dim, d_block), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d_block
        layers.append(nn.Linear(d_block, 1))
        self.net = nn.Sequential(*layers)

    @property
    def k(self) -> int:
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).unsqueeze(1)   # (B, 1, 1)


class PlainMLPModel:
    """Wrapper sklearn-like pour PlainMLP."""

    def __init__(self, input_dim: int, config: AblationVariantConfig) -> None:
        self.config = config
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._scaler = StandardScaler()
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._model: _PlainMLP | None = None
        print(f"PlainMLP | device={self.device} | blocks={config.n_blocks} | d={config.d_block}", flush=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict]:
        cfg = self.config
        self._scaler.fit(X_train)
        X_tr_s = self._scaler.transform(X_train).astype(np.float32)
        X_va_s = self._scaler.transform(X_val).astype(np.float32)

        self._y_mean = float(y_train.mean())
        self._y_std  = float(y_train.std()) or 1.0
        y_tr_s = ((y_train - self._y_mean) / self._y_std).astype(np.float32)
        y_va_s = ((y_val   - self._y_mean) / self._y_std).astype(np.float32)

        self._model = _PlainMLP(self.input_dim, cfg.n_blocks, cfg.d_block, cfg.dropout).to(self.device)
        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"PlainMLP | params={n_params:,}", flush=True)

        train_dl, val_dl = _loaders(X_tr_s, y_tr_s, X_va_s, y_va_s, cfg.batch_size)
        return _train_loop(
            self._model, train_dl, val_dl, cfg, self.device,
            y_val, self._y_std, self._y_mean, k=1,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X).astype(np.float32)
        loader = _pred_loader(X_s)
        self._model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_b in loader:
                out = self._model(X_b.to(self.device))   # (B, 1, 1)
                preds.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(preds) * self._y_std + self._y_mean


# ===========================================================================
# 2. AblationTabMModel  (tabm-packed | tabm-mini | tabm)
# ===========================================================================
class AblationTabMModel:
    """
    Wrapper pour TabM.make(..., arch_type=arch_type).
    Supporte : 'tabm-packed', 'tabm-mini', 'tabm'
    avec ou sans feature embeddings PiecewiseLinear.
    """

    def __init__(self, input_dim: int, config: AblationVariantConfig) -> None:
        self.config = config
        self.input_dim = input_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._scaler = StandardScaler()
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._model: nn.Module | None = None
        print(
            f"{config.variant_id} | arch={config.arch_type} | k={config.k} | "
            f"blocks={config.n_blocks} | d={config.d_block} | "
            f"embed={config.use_embeddings} | device={self.device}",
            flush=True,
        )

    def _build_model(self, X_scaled: np.ndarray) -> None:
        from tabm import TabM  # type: ignore[import]
        cfg = self.config

        num_embeddings = None
        if cfg.use_embeddings:
            from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins  # type: ignore[import]
            X_t = torch.from_numpy(X_scaled).float()
            bins = compute_bins(X_t, n_bins=cfg.n_bins)
            num_embeddings = PiecewiseLinearEmbeddings(
                bins,
                d_embedding=cfg.d_embedding,
                activation=False,
                version="B",
            )

        self._model = TabM.make(
            n_num_features=self.input_dim,
            d_out=1,
            num_embeddings=num_embeddings,
            k=cfg.k,
            arch_type=cfg.arch_type,
            n_blocks=cfg.n_blocks,
            d_block=cfg.d_block,
            dropout=cfg.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        print(f"{cfg.variant_id} | params={n_params:,}", flush=True)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict]:
        cfg = self.config
        self._scaler.fit(X_train)
        X_tr_s = self._scaler.transform(X_train).astype(np.float32)
        X_va_s = self._scaler.transform(X_val).astype(np.float32)

        self._y_mean = float(y_train.mean())
        self._y_std  = float(y_train.std()) or 1.0
        y_tr_s = ((y_train - self._y_mean) / self._y_std).astype(np.float32)
        y_va_s = ((y_val   - self._y_mean) / self._y_std).astype(np.float32)

        self._build_model(X_tr_s)   # bins calculés sur le train scalé

        train_dl, val_dl = _loaders(X_tr_s, y_tr_s, X_va_s, y_va_s, cfg.batch_size)
        return _train_loop(
            self._model, train_dl, val_dl, cfg, self.device,
            y_val, self._y_std, self._y_mean, k=cfg.k,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_s = self._scaler.transform(X).astype(np.float32)
        loader = _pred_loader(X_s)
        self._model.eval()
        preds: list[np.ndarray] = []
        with torch.no_grad():
            for X_b in loader:
                out = self._model(X_b.to(self.device))   # (B, k, 1)
                preds.append(out.mean(dim=1).squeeze(-1).cpu().numpy())
        return np.concatenate(preds) * self._y_std + self._y_mean


# ===========================================================================
# Factory
# ===========================================================================
def build_model(input_dim: int, cfg: AblationVariantConfig):
    """Instancie le bon wrapper selon cfg.arch_type."""
    if cfg.arch_type == "plain":
        return PlainMLPModel(input_dim, cfg)
    return AblationTabMModel(input_dim, cfg)
