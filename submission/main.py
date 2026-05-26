"""Kaggle submission entry — thin wrapper over src.policy.hybrid_agent."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.policy.hybrid_agent import agent  # noqa: E402

__all__ = ["agent"]
