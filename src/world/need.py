"""Ownership need search — ported from submission_v2.WorldModel.min_ships_to_own_by."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.world.context import WorldContext
from src.world.timeline import simulate_planet_timeline, state_at_timeline

PlannedCommitments = Dict[int, List[Tuple[int, int, int]]]
Arrival = Tuple[int, int, int]


def projected_timeline(
    ctx: WorldContext,
    target_id: int,
    horizon: int,
    planned_commitments: Optional[PlannedCommitments] = None,
    extra_arrivals: Sequence[Arrival] = (),
) -> dict:
    planned_commitments = planned_commitments or {}
    horizon = max(1, int(math.ceil(horizon)))
    arrivals = [
        item for item in ctx.arrivals_by_planet.get(target_id, []) if item[0] <= horizon
    ]
    arrivals.extend(
        item for item in planned_commitments.get(target_id, []) if item[0] <= horizon
    )
    arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
    target = ctx.planet_by_id[target_id]
    return simulate_planet_timeline(target, arrivals, ctx.player, horizon)


def projected_state(
    ctx: WorldContext,
    target_id: int,
    arrival_turn: int,
    planned_commitments: Optional[PlannedCommitments] = None,
    extra_arrivals: Sequence[Arrival] = (),
) -> Tuple[int, float]:
    planned_commitments = planned_commitments or {}
    cutoff = max(1, int(math.ceil(arrival_turn)))
    if not planned_commitments.get(target_id) and not extra_arrivals:
        return state_at_timeline(ctx.base_timeline[target_id], cutoff)

    dyn = projected_timeline(
        ctx, target_id, cutoff, planned_commitments, extra_arrivals,
    )
    return state_at_timeline(dyn, cutoff)


def ownership_search_cap(ctx: "WorldContext", eval_turn: int) -> int:
    productive_cap = ctx.total_production * max(2, eval_turn + 2)
    return max(32, int(ctx.total_visible_ships + productive_cap + 32))


def min_ships_to_own_by(
    ctx: "WorldContext",
    target_id: int,
    eval_turn: int,
    attacker_owner: int,
    arrival_turn: Optional[int] = None,
    planned_commitments: Optional[PlannedCommitments] = None,
    extra_arrivals: Sequence[Arrival] = (),
    upper_bound: Optional[int] = None,
    model_hostile_reinforce: bool = False,
) -> int:
    if model_hostile_reinforce:
        # B1 still uses v2 plan_shot / hostile_reinforcement_arrivals until extracted.
        from src.policy.v2_bridge import get_v2_module

        return get_v2_module().min_ships_to_own_by(
            target_id,
            eval_turn,
            attacker_owner,
            arrival_turn=arrival_turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
            model_hostile_reinforce=True,
        )

    planned_commitments = planned_commitments or {}
    eval_turn = max(1, int(math.ceil(eval_turn)))
    arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))

    if arrival_turn > eval_turn:
        if upper_bound is not None:
            return max(1, int(upper_bound)) + 1
        return ownership_search_cap(ctx, eval_turn) + 1

    normalized_extra = tuple(
        (max(1, int(math.ceil(turns))), owner, int(ships))
        for turns, owner, ships in extra_arrivals
        if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
    )

    owner_before, ships_before = projected_state(
        ctx, target_id, eval_turn, planned_commitments, normalized_extra,
    )
    if owner_before == attacker_owner:
        return 0

    def owns_at(ships: int) -> bool:
        owner_after, _ = projected_state(
            ctx,
            target_id,
            eval_turn,
            planned_commitments,
            normalized_extra + ((arrival_turn, attacker_owner, int(ships)),),
        )
        return owner_after == attacker_owner

    if upper_bound is not None:
        hi = max(1, int(upper_bound))
        if not owns_at(hi):
            return hi + 1
    else:
        hi = max(1, int(math.ceil(ships_before)) + 1)
        search_cap = ownership_search_cap(ctx, eval_turn)
        while hi <= search_cap and not owns_at(hi):
            hi *= 2
        if hi > search_cap:
            hi = search_cap
            if not owns_at(hi):
                return hi + 1

    lo = 1
    while lo < hi:
        mid = (lo + hi) // 2
        if owns_at(mid):
            hi = mid
        else:
            lo = mid + 1
    return lo


def min_ships_to_own_at(
    ctx: WorldContext,
    target_id: int,
    arrival_turn: int,
    attacker_owner: Optional[int] = None,
    planned_commitments: Optional[PlannedCommitments] = None,
    extra_arrivals: Sequence[Arrival] = (),
    upper_bound: Optional[int] = None,
    model_hostile_reinforce: bool = False,
) -> int:
    attacker_owner = ctx.player if attacker_owner is None else attacker_owner
    turn = max(1, int(math.ceil(arrival_turn)))
    return min_ships_to_own_by(
        ctx,
        target_id,
        turn,
        attacker_owner,
        arrival_turn=turn,
        planned_commitments=planned_commitments,
        extra_arrivals=extra_arrivals,
        upper_bound=upper_bound,
        model_hostile_reinforce=model_hostile_reinforce,
    )
