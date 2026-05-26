"""World facade — native context + v2 bridge for plan_shot / missions (transitional)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from src.world.context import WorldContext, build_world_context
from src.world.need import min_ships_to_own_at as _min_ships_to_own_at
from src.world.need import min_ships_to_own_by as _min_ships_to_own_by
from src.world.need import projected_state as _projected_state


class World:
    """Hybrid world: native physics in `.ctx`, shooting still via v2 WorldModel."""

    def __init__(self, obs: Any, inferred_step: Optional[int] = None):
        self.ctx = build_world_context(obs, inferred_step=inferred_step)
        self.player = self.ctx.player
        self.step = self.ctx.step
        self.planets = self.ctx.planets
        self.fleets = self.ctx.fleets

        from src.policy.v2_bridge import get_v2_module

        self._v2 = get_v2_module().build_world(obs, inferred_step=inferred_step)

    @classmethod
    def from_obs(cls, obs: Any, inferred_step: Optional[int] = None) -> "World":
        return cls(obs, inferred_step=inferred_step)

    @property
    def inner(self):
        """v2 WorldModel — transitional for plan_shot, settle_plan, missions."""
        return self._v2

    def plan_shot(self, src_id: int, target_id: int, ships: int):
        return self._v2.plan_shot(src_id, target_id, ships)

    def projected_state(self, target_id: int, eval_turn: int, **kwargs) -> Tuple[int, float]:
        return _projected_state(self.ctx, target_id, eval_turn, **kwargs)

    def min_ships_to_own_at(self, target_id: int, arrival_turn: int, **kwargs) -> int:
        return _min_ships_to_own_at(self.ctx, target_id, arrival_turn, **kwargs)

    def min_ships_to_own_by(self, target_id: int, eval_turn: int, attacker_owner: int, **kwargs) -> int:
        return _min_ships_to_own_by(self.ctx, target_id, eval_turn, attacker_owner, **kwargs)
