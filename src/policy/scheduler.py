"""Conflict resolution and per-turn ship budget — extract from submission_v2.plan_moves."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence

from src.world.state import PlannedCommitments

Move = List[Any]


class MoveScheduler:
    def __init__(self, world, player: int):
        self.world = world
        self.player = player
        self.reserved_by_planet: Dict[int, int] = defaultdict(int)
        self.planned_commitments: PlannedCommitments = defaultdict(list)

    def source_inventory_left(self, source_id: int) -> int:
        planet = self.world.inner.planet_by_id.get(source_id)
        if planet is None:
            return 0
        return max(0, int(planet.ships) - self.reserved_by_planet[source_id])

    def can_spend(self, source_id: int, ships: int) -> bool:
        return self.source_inventory_left(source_id) >= int(ships)

    def commit(self, source_id: int, target_id: int, ships: int, eta: int) -> None:
        ships = int(ships)
        self.reserved_by_planet[source_id] += ships
        self.planned_commitments[target_id].append((int(eta), self.player, ships))

    def to_moves(self, plans: Sequence) -> List[Move]:
        """plans: objects with src_id, angle, ships attributes."""
        moves: List[Move] = []
        for plan in sorted(plans, key=lambda p: p.score, reverse=True):
            if not self.can_spend(plan.src_id, plan.ships):
                continue
            self.commit(plan.src_id, plan.target_id, plan.ships, plan.eta)
            moves.append([plan.src_id, float(plan.angle), int(plan.ships)])
        return moves
