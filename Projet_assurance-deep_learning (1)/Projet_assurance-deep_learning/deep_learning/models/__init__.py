from __future__ import annotations
from models.tabnet import TabNetModel
from models.tabm_model import TabMModel

MODEL_REGISTRY: dict[str, type] = {
    "tabnet": TabNetModel,
    "tabm": TabMModel,
}
