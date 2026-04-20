from __future__ import annotations
import numpy as np
from logging_utils import get_logger
from metrics import qwk, optimize_offsets, apply_offsets
from models.base import BaseTabularModel

logger = get_logger(__name__)


def run_training(
    model: BaseTabularModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[float, float]:
    model.fit(X_train, y_train, X_val, y_val)

    val_preds = model.predict(X_val)
    qwk_raw = qwk(val_preds, y_val)
    print(f"QWK (raw):     {qwk_raw:.4f}", flush=True)
    logger.info(f"QWK (raw)={qwk_raw:.4f}")

    offsets = optimize_offsets(val_preds, y_val)
    qwk_offset = qwk(apply_offsets(val_preds, offsets), y_val)
    print(f"QWK (offsets): {qwk_offset:.4f}", flush=True)
    logger.info(f"QWK (offsets)={qwk_offset:.4f}")

    return qwk_raw, qwk_offset
