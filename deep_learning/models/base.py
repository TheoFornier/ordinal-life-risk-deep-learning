from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np


class BaseTabularModel(ABC):
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> list[dict]:
        """Entraîne le modèle sur les données. Retourne l'historique par époque."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne les prédictions continues pour le jeu X."""

    @abstractmethod
    def save(self, path: str) -> None:
        """Sauvegarde les poids du modèle sur disque."""

    @abstractmethod
    def load(self, path: str) -> None:
        """Charge les poids depuis un fichier sauvegardé."""
