from __future__ import annotations
import logging
import os
from datetime import datetime


def setup_logging(log_dir: str, model_name: str = "") -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.log" if model_name else f"{timestamp}.log"
    log_path = os.path.join(log_dir, filename)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(fh)

    for lib in ("pytorch_tabnet", "torch", "urllib3", "PIL"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    print(f"Log: {log_path}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
