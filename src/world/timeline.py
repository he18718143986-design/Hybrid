"""Planet timeline — ported from submission_v2."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Sequence, Set, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from src.world.combat import resolve_arrival_event
from src.world.fleet import fleet_target_planet

Arrival = Tuple[int, int, int]


def build_arrival_ledger(
    fleets: Sequence[Fleet],
    planets: Sequence[Planet],
    initial_by_id: Dict[int, Planet],
    angular_velocity: float,
    comets,
    comet_ids: Set[int],
) -> Dict[int, List[Arrival]]:
    arrivals_by_planet: Dict[int, List[Arrival]] = {planet.id: [] for planet in planets}
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet, list(planets), initial_by_id, angular_velocity, comets, comet_ids,
        )
        if target is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def normalize_arrivals(arrivals: Sequence[Arrival], horizon: int) -> List[Arrival]:
    events: List[Arrival] = []
    for turns, owner, ships in arrivals:
        if ships <= 0:
            continue
        eta = max(1, int(math.ceil(turns)))
        if eta > horizon:
            continue
        events.append((eta, owner, int(ships)))
    events.sort(key=lambda item: item[0])
    return events


def simulate_planet_timeline(
    planet: Planet,
    arrivals: Sequence[Arrival],
    player: int,
    horizon: int,
) -> dict:
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn: Dict[int, List[Arrival]] = defaultdict(list)
    for item in events:
        by_turn[item[0]].append(item)

    owner = planet.owner
    garrison = float(planet.ships)
    owner_at = {0: owner}
    ships_at = {0: max(0.0, garrison)}
    min_owned = garrison if owner == player else 0.0
    first_enemy = None
    fall_turn = None

    for turn in range(1, horizon + 1):
        if owner != -1:
            garrison += planet.production

        group = by_turn.get(turn, [])
        prev_owner = owner
        if group:
            if prev_owner == player and first_enemy is None:
                if any(item[1] not in (-1, player) for item in group):
                    first_enemy = turn
            owner, garrison = resolve_arrival_event(owner, garrison, group)
            if prev_owner == player and owner != player and fall_turn is None:
                fall_turn = turn

        owner_at[turn] = owner
        ships_at[turn] = max(0.0, garrison)
        if owner == player:
            min_owned = min(min_owned, garrison)

    keep_needed = 0
    holds_full = True

    if planet.owner == player:

        def survives_with_keep(keep: int) -> bool:
            sim_owner = planet.owner
            sim_garrison = float(keep)
            for turn in range(1, horizon + 1):
                if sim_owner != -1:
                    sim_garrison += planet.production
                group = by_turn.get(turn, [])
                if group:
                    sim_owner, sim_garrison = resolve_arrival_event(
                        sim_owner, sim_garrison, group,
                    )
                    if sim_owner != player:
                        return False
            return sim_owner == player

        if survives_with_keep(int(planet.ships)):
            lo, hi = 0, int(planet.ships)
            while lo < hi:
                mid = (lo + hi) // 2
                if survives_with_keep(mid):
                    hi = mid
                else:
                    lo = mid + 1
            keep_needed = lo
        else:
            holds_full = False
            keep_needed = int(planet.ships)

    return {
        "owner_at": owner_at,
        "ships_at": ships_at,
        "keep_needed": keep_needed,
        "min_owned": max(0, int(math.floor(min_owned))) if planet.owner == player else 0,
        "first_enemy": first_enemy,
        "fall_turn": fall_turn,
        "holds_full": holds_full,
        "horizon": horizon,
    }


def state_at_timeline(timeline: dict, arrival_turn: int) -> Tuple[int, float]:
    turn = max(0, int(math.ceil(arrival_turn)))
    turn = min(turn, timeline["horizon"])
    owner = timeline["owner_at"].get(turn, timeline["owner_at"][timeline["horizon"]])
    ships = timeline["ships_at"].get(turn, timeline["ships_at"][timeline["horizon"]])
    return owner, max(0.0, ships)
