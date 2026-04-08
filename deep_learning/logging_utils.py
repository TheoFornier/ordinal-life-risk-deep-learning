from __future__ import annotations

import logging
import os
import sys
from datetime import datetime


_FMT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
_DATE_FMT = "%H:%M:%S"


def setup_logging(log_dir: str | None = None, model_name: str = "") -> None:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # On repart d'une ardoise vierge pour éviter les doublons de handlers
    root.handlers.clear()

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console : niveau INFO et plus
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root.addHandler(console)

    # Fichier : tout (DEBUG inclus)
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.log" if model_name else f"{timestamp}.log"
        fh = logging.FileHandler(os.path.join(log_dir, filename), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        root.addHandler(fh)
        logging.getLogger(__name__).info(f"Logs écrits dans : {os.path.join(log_dir, filename)}")

    # On réduit le bruit des bibliothèques tierces
    for lib in ("pytorch_tabnet", "torch", "urllib3", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
