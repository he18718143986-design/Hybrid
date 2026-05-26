"""Position prediction — ported from submission_v2."""

from __future__ import annotations

import math
from typing import Dict, Optional, Set, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.world.comet import comet_remaining_life, predict_comet_position
from src.world.constants import CENTER_X, CENTER_Y, ROTATION_LIMIT
from src.world.geometry import dist

Point = Tuple[float, float]


def predict_planet_position(
    planet: Planet,
    initial_by_id: Dict[int, Planet],
    angular_velocity: float,
    turns: int,
) -> Point:
    init = initial_by_id.get(planet.id)
    if init is None:
        return planet.x, planet.y
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    if r + init.radius >= ROTATION_LIMIT:
        return planet.x, planet.y
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + angular_velocity * turns
    return (
        CENTER_X + r * math.cos(new_ang),
        CENTER_Y + r * math.sin(new_ang),
    )


def predict_target_position(
    target: Planet,
    turns: int,
    initial_by_id: Dict[int, Planet],
    ang_vel: float,
    comets,
    comet_ids: Set[int],
) -> Optional[Point]:
    if target.id in comet_ids:
        return predict_comet_position(target.id, comets, turns)
    return predict_planet_position(target, initial_by_id, ang_vel, turns)


def target_can_move(target: Planet, initial_by_id: Dict[int, Planet], comet_ids: Set[int]) -> bool:
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    r = dist(init.x, init.y, CENTER_X, CENTER_Y)
    return r + init.radius < ROTATION_LIMIT
