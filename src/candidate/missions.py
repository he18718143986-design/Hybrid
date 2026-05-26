"""Mission types — extract builders from submission_v2 build_*_missions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from src.world.state import World


@dataclass
class Mission:
    kind: str
    target_id: int
    priority: float = 0.0
    meta: dict = field(default_factory=dict)


def generate_missions(world: World, modes: dict, budget) -> List[Mission]:
    """Return prioritized missions (capture / reinforce / snipe / …).

    TODO: call submission_v2 mission builders or reimplement incrementally.
    """
    from src.candidate.scoring import target_value

    missions: List[Mission] = []
    for planet in world.inner.planets:
        if planet.owner == world.player:
            continue
        score = target_value(world, planet, modes)
        if score <= -1e8:
            continue
        kind = "capture" if planet.owner == -1 else "attack"
        missions.append(Mission(kind=kind, target_id=planet.id, priority=score))
    missions.sort(key=lambda m: m.priority, reverse=True)
    return missions[:16]
