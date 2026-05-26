"""Value network / regressor stub."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "value_model.pt"


def load_value_model(path: Optional[Path] = None):
    """TODO: torch.load or joblib for LightGBM."""
    return None


def predict_value(features: List[float], model=None) -> float:
    _ = (features, model)
    return 0.0
