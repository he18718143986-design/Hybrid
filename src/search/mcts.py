"""Optional MCTS — Phase 5 only; keep tree small."""

from __future__ import annotations

from typing import Any, List

from src.search.budget import TimeBudget
from src.world.state import World


def mcts(
    world: World,
    candidates: List[Any],
    budget: TimeBudget,
) -> Any:
    """TODO: small MCTS over top-k candidates only."""
    raise NotImplementedError("Phase 5")
