from __future__ import annotations
import os
import numpy as np
import pandas as pd
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
) -> tuple[float, float, np.ndarray]:
    logger.info("--- Début de l'entraînement ---")
    model.fit(X_train, y_train, X_val, y_val)

    logger.info("Évaluation sur la validation...")
    val_preds = model.predict(X_val)

    qwk_raw = qwk(val_preds, y_val)
    logger.info(f"QWK (arrondi simple) :  {qwk_raw:.4f}")

    logger.info("Optimisation des offsets par classe...")
    offsets = optimize_offsets(val_preds, y_val)
    qwk_offset = qwk(apply_offsets(val_preds, offsets), y_val)
    logger.info(f"QWK (avec offsets) :    {qwk_offset:.4f}")
    logger.debug(f"Offsets : {np.round(offsets, 4)}")

    return qwk_raw, qwk_offset, offsets


def generate_submission(
    model: BaseTabularModel,
    X_test: np.ndarray,
    ids_test: np.ndarray,
    offsets: np.ndarray,
    output_path: str,
) -> None:
    logger.info("Génération des prédictions sur le jeu de test...")
    test_preds = model.predict(X_test)
    final_preds = apply_offsets(test_preds, offsets)

    submission = pd.DataFrame({"Id": ids_test, "Response": final_preds})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False)

    dist = submission["Response"].value_counts().sort_index()
    logger.info(f"Soumission sauvegardée : {output_path}")
    logger.info(f"Distribution des prédictions :\n{dist.to_string()}")
