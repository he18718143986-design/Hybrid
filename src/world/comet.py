"""Comet path prediction from obs['comets']."""

from __future__ import annotations

import copy
from typing import List, Optional, Set, Tuple

Point = Tuple[float, float]


def predict_comet_position(
    planet_id: int,
    comets,
    turn_offset: int,
) -> Optional[Point]:
    turn_offset = int(turn_offset)
    for group in comets or []:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + turn_offset
        if 0 <= future_idx < len(path):
            return float(path[future_idx][0]), float(path[future_idx][1])
    return None


def advance_comets_one_turn(comets: list) -> list:
    """Return comet groups with path_index incremented by 1 (env end-of-turn)."""
    out: list = []
    for group in comets or []:
        g = copy.copy(group)
        g["path_index"] = int(g.get("path_index", 0)) + 1
        out.append(g)
    return out


def comet_position_at_current_index(
    planet_id: int,
    comets: list,
) -> Optional[Point]:
    """Position at group's current path_index (after advance)."""
    for group in comets or []:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        if idx >= len(paths):
            return None
        path = paths[idx]
        path_index = int(group.get("path_index", 0))
        if 0 <= path_index < len(path):
            return float(path[path_index][0]), float(path[path_index][1])
        return None
    return None


def expired_comet_ids(comets: list) -> Set[int]:
    """Planet IDs whose comet path has been exhausted."""
    expired: Set[int] = set()
    for group in comets or []:
        idx = int(group.get("path_index", 0))
        for i, pid in enumerate(group.get("planet_ids", [])):
            paths = group.get("paths", [])
            if i < len(paths) and idx >= len(paths[i]):
                expired.add(int(pid))
    return expired


def comet_remaining_life(planet_id: int, comets) -> int:
    for group in comets or []:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0
