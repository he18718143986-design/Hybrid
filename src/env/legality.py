"""Move legality checks against current observation."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Sequence

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet

from src.env.parser import parse_observation
from src.world.fleet import fleet_target_planet

Move = List[Any]  # [from_planet_id, angle, num_ships]


def validate_move(
    move: Move,
    owned_planet_ids: set,
    available_ships: dict,
) -> bool:
    """Return True if move is structurally valid and affordable."""
    if len(move) != 3:
        return False
    src_id, angle, ships = move
    if int(src_id) not in owned_planet_ids:
        return False
    ships = int(ships)
    if ships < 1:
        return False
    if ships > available_ships.get(int(src_id), 0):
        return False
    try:
        float(angle)
    except (TypeError, ValueError):
        return False
    return True


def validate_move_for_obs(move: Move, obs: Any) -> bool:
    """Structural + reachable-target check (hybrid safety gate)."""
    parsed = parse_observation(obs)
    player = parsed["player"]
    owned = {p.id for p in parsed["planets"] if p.owner == player}
    available = {p.id: int(p.ships) for p in parsed["planets"] if p.owner == player}
    if not validate_move(move, owned, available):
        return False

    from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
    if not math.isfinite(angle):
        return False

    src = next((p for p in parsed["planets"] if p.id == from_id), None)
    if src is None:
        return False

    from src.world.geometry import launch_point

    lx, ly = launch_point(src.x, src.y, src.radius, angle)
    fleet = Fleet(-1, player, lx, ly, angle, from_id, ships)
    target, eta = fleet_target_planet(
        fleet,
        parsed["planets"],
        {p.id: p for p in parsed["initial_planets"]},
        parsed["angular_velocity"],
        parsed["comets"],
        set(parsed["comet_planet_ids"]),
    )
    return target is not None and eta is not None and int(eta) >= 1


def pick_legal_or_fallback(
    preferred: Sequence[Move],
    fallback: Sequence[Move],
    obs: Any,
) -> List[Move]:
    """Prefer preferred moves if legal and reachable; else v2/fallback."""
    legal_pref = filter_legal_moves(preferred, obs)
    if legal_pref and all(validate_move_for_obs(m, obs) for m in legal_pref):
        return legal_pref
    return filter_legal_moves(fallback, obs)


def filter_legal_moves(moves: Sequence[Move], obs: Any) -> List[Move]:
    """Drop illegal moves; dedupe by (src, angle, ships) not applied here."""
    parsed = parse_observation(obs)
    player = parsed["player"]
    owned = {p.id for p in parsed["planets"] if p.owner == player}
    available = {p.id: int(p.ships) for p in parsed["planets"] if p.owner == player}
    spent: dict = {}
    legal: List[Move] = []
    for move in moves:
        src_id = int(move[0])
        ships = int(move[2])
        left = available.get(src_id, 0) - spent.get(src_id, 0)
        if validate_move(move, owned, {src_id: left}):
            legal.append(move)
            spent[src_id] = spent.get(src_id, 0) + ships
    return legal
