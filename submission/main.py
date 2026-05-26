"""Kaggle submission entry — production uses v2 (set before hybrid import)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["ORBIT_AGENT_MODE"] = "v2"

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.policy.hybrid_agent import agent  # noqa: E402

__all__ = ["agent"]
