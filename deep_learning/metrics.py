from __future__ import annotations
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import cohen_kappa_score
from config import NUM_CLASSES


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.array(y_true).astype(int)
    y_pred = np.clip(np.round(np.array(y_pred)), 1, NUM_CLASSES).astype(int)
    return float(np.mean(y_pred == y_true))


# Quadratic Weighted Kappa — métrique principale du concours
def qwk(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.array(y_true).astype(int)
    y_pred = np.clip(np.round(np.array(y_pred)), 1, NUM_CLASSES).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# Recherche des décalages par classe qui maximisent le QWK
def optimize_offsets(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    preds = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)
    adjusted = preds.copy()
    offsets = np.zeros(NUM_CLASSES)

    for j in [6, 4, 5, 3, 2, 1, 7, 0]:
        def objective(x: float, cls: int = j) -> float:
            adjusted[preds.astype(int) == cls] = preds[preds.astype(int) == cls] + x
            return -qwk(adjusted, labels)

        result = minimize_scalar(objective, bounds=(-3, 3), method="bounded")
        offsets[j] = result.x
        adjusted[preds.astype(int) == j] = preds[preds.astype(int) == j] + offsets[j]

    return offsets


def apply_offsets(preds: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    preds = np.array(preds, dtype=float)
    adjusted = preds.copy()
    for j in range(NUM_CLASSES):
        mask = preds.astype(int) == j
        adjusted[mask] = preds[mask] + offsets[j]
    return np.clip(np.round(adjusted), 1, NUM_CLASSES).astype(int)
