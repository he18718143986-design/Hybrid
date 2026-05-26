"""Act-step time budget (shared with policy layer)."""

from __future__ import annotations

import time
from enum import Enum
from typing import Optional

from src.world.constants import DEFAULT_ACT_TIMEOUT_S, SOFT_ACT_CAP, SOFT_ACT_FLOOR


class ComputeTier(Enum):
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


class TimeBudget:
    @classmethod
    def from_config(cls, config: Optional[dict] = None) -> "TimeBudget":
        config = config or {}
        act_timeout = float(config.get("actTimeout", DEFAULT_ACT_TIMEOUT_S))
        soft = min(SOFT_ACT_CAP, max(SOFT_ACT_FLOOR, act_timeout * SOFT_ACT_CAP))
        return cls(soft)

    def __init__(self, soft: float):
        self._start = time.perf_counter()
        self.deadline = self._start + soft
        self.tier = ComputeTier.LIGHT

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.perf_counter())

    def expired(self) -> bool:
        return time.perf_counter() >= self.deadline

    def refresh_tier(self) -> None:
        r = self.remaining()
        if r > 0.5:
            self.tier = ComputeTier.HEAVY
        elif r > 0.2:
            self.tier = ComputeTier.MEDIUM
        else:
            self.tier = ComputeTier.LIGHT
