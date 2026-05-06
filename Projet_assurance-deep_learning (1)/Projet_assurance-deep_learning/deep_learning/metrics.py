from __future__ import annotations
import numpy as np
from scipy.optimize import differential_evolution
from sklearn.metrics import cohen_kappa_score
from config import NUM_CLASSES


def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.array(y_true).astype(int)
    y_pred = np.clip(np.round(np.array(y_pred)), 1, NUM_CLASSES).astype(int)
    return float(np.mean(y_pred == y_true))


# Quadratic Weighted Kappa
def qwk(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.array(y_true).astype(int)
    y_pred = np.clip(np.round(np.array(y_pred)), 1, NUM_CLASSES).astype(int)
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


# Recherche des décalages par classe qui maximisent le QWK
def optimize_offsets(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    preds = np.array(preds, dtype=float)
    labels = np.array(labels, dtype=float)

    def objective(offsets: np.ndarray) -> float:
        adjusted = preds.copy()
        for j in range(NUM_CLASSES):
            mask = preds.astype(int) == j
            adjusted[mask] = preds[mask] + offsets[j]
        return -qwk(adjusted, labels)

    bounds = [(-3, 3)] * NUM_CLASSES
    result = differential_evolution(objective, bounds, seed=42, tol=1e-4, maxiter=1000)
    return result.x


def apply_offsets(preds: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    preds = np.array(preds, dtype=float)
    adjusted = preds.copy()
    for j in range(NUM_CLASSES):
        mask = preds.astype(int) == j
        adjusted[mask] = preds[mask] + offsets[j]
    return np.clip(np.round(adjusted), 1, NUM_CLASSES).astype(int)
