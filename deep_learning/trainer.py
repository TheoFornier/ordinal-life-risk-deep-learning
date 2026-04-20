from __future__ import annotations
import numpy as np
from logging_utils import get_logger
from metrics import accuracy, qwk, optimize_offsets, apply_offsets
from models.base import BaseTabularModel

logger = get_logger(__name__)


def run_training(
    model: BaseTabularModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_fit: np.ndarray | None = None,
    y_fit: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> tuple[float, float, float, float, np.ndarray, list[dict]]:
    history = model.fit(X_train, y_train, X_val, y_val, X_fit=X_fit, y_fit=y_fit, sample_weight=sample_weight)

    val_preds = model.predict(X_val)
    qwk_raw = qwk(val_preds, y_val)
    acc_raw = accuracy(val_preds, y_val)
    print(f"QWK (raw):     {qwk_raw:.4f}  |  Accuracy (raw):     {acc_raw:.4f}", flush=True)
    logger.info(f"QWK (raw)={qwk_raw:.4f} | accuracy (raw)={acc_raw:.4f}")

    offsets = optimize_offsets(val_preds, y_val)
    preds_offset = apply_offsets(val_preds, offsets)
    qwk_offset = qwk(preds_offset, y_val)
    acc_offset = accuracy(preds_offset, y_val)
    print(f"QWK (offsets): {qwk_offset:.4f}  |  Accuracy (offsets): {acc_offset:.4f}", flush=True)
    logger.info(f"QWK (offsets)={qwk_offset:.4f} | accuracy (offsets)={acc_offset:.4f}")

    return qwk_raw, qwk_offset, acc_raw, acc_offset, offsets, history
