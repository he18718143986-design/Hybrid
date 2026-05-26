"""Orbiting planet position prediction."""

from __future__ import annotations

import math
from typing import Dict, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.world.constants import CENTER_X, CENTER_Y, ROTATION_LIMIT

Point = Tuple[float, float]


def orbital_radius_from_initial(init: Planet) -> float:
    return math.hypot(init.x - CENTER_X, init.y - CENTER_Y)


def is_orbiting_initial(init: Planet) -> bool:
    return orbital_radius_from_initial(init) + init.radius < ROTATION_LIMIT


def predict_orbiting_position(
    planet: Planet,
    orbital_r: float,
    angular_velocity: float,
    turn_offset: int,
) -> Point:
    """Current observation angle + omega * offset (robust if step is missing)."""
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    ang = cur_ang + float(angular_velocity) * int(turn_offset)
    return CENTER_X + orbital_r * math.cos(ang), CENTER_Y + orbital_r * math.sin(ang)


def build_orbital_metadata(
    initial_by_id: Dict[int, Planet],
    comet_ids: set,
) -> Dict[int, dict]:
    """Per-planet {is_orbiting, orbital_radius} from initial_planets only."""
    meta: Dict[int, dict] = {}
    for pid, init in initial_by_id.items():
        if pid in comet_ids:
            continue
        r = orbital_radius_from_initial(init)
        meta[pid] = {
            "is_orbiting": r + init.radius < ROTATION_LIMIT,
            "orbital_radius": r,
        }
    return meta
