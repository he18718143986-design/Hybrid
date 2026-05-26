"""Load submission_v2.py from repo root (SSOT) without duplicating logic."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path
from types import ModuleType

_REPO_ROOT = Path(__file__).resolve().parents[2]
_V2_PATH = _REPO_ROOT / "submission_v2.py"


@lru_cache(maxsize=1)
def get_v2_module() -> ModuleType:
    if not _V2_PATH.is_file():
        raise FileNotFoundError(f"submission_v2.py not found at {_V2_PATH}")
    spec = importlib.util.spec_from_file_location("orbit_submission_v2", _V2_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {_V2_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def v2_agent(obs, config=None):
    return get_v2_module().agent(obs, config)
