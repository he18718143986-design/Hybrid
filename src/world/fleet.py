"""Fleet target inference (A1) — ported from submission_v2.fleet_target_planet."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from src.world.constants import CENTER_X, CENTER_Y, HORIZON, ROTATION_LIMIT
from src.world.geometry import dist, fleet_speed
from src.world.prediction import predict_target_position


def fleet_target_planet(
    fleet: Fleet,
    planets: List[Planet],
    initial_by_id: Dict[int, Planet],
    angular_velocity: float,
    comets,
    comet_ids: Set[int],
) -> Tuple[Optional[Planet], Optional[int]]:
    dir_x = math.cos(fleet.angle)
    dir_y = math.sin(fleet.angle)
    speed = fleet_speed(fleet.ships)

    static_candidates: List[Planet] = []
    rotating_candidates: List[Planet] = []
    for planet in planets:
        if planet.id in comet_ids:
            rotating_candidates.append(planet)
            continue
        init = initial_by_id.get(planet.id)
        if init is None:
            static_candidates.append(planet)
            continue
        if dist(init.x, init.y, CENTER_X, CENTER_Y) + init.radius >= ROTATION_LIMIT:
            static_candidates.append(planet)
        else:
            rotating_candidates.append(planet)

    best_eta: Optional[int] = None
    best_planet: Optional[Planet] = None

    for planet in static_candidates:
        dx = planet.x - fleet.x
        dy = planet.y - fleet.y
        proj = dx * dir_x + dy * dir_y
        if proj < 0:
            continue
        perp_sq = dx * dx + dy * dy - proj * proj
        radius_sq = planet.radius * planet.radius
        if perp_sq >= radius_sq:
            continue
        hit_d = max(0.0, proj - math.sqrt(max(0.0, radius_sq - perp_sq)))
        turns = hit_d / speed
        if turns > HORIZON:
            continue
        eta = max(1, int(math.ceil(turns)))
        if best_eta is None or eta < best_eta:
            best_eta = eta
            best_planet = planet

    if rotating_candidates:
        radius_sq_cache = {p.id: p.radius * p.radius for p in rotating_candidates}
        upper_t = best_eta if best_eta is not None else HORIZON
        for t in range(1, upper_t + 1):
            if best_eta is not None and t >= best_eta:
                break
            fx = fleet.x + dir_x * speed * t
            fy = fleet.y + dir_y * speed * t
            for planet in rotating_candidates:
                pos = predict_target_position(
                    planet, t, initial_by_id, angular_velocity, comets, comet_ids,
                )
                if pos is None:
                    continue
                px, py = pos
                dxp = px - fx
                dyp = py - fy
                if dxp * dxp + dyp * dyp <= radius_sq_cache[planet.id]:
                    if best_eta is None or t < best_eta:
                        best_eta = t
                        best_planet = planet
                    break

    if best_planet is None:
        return None, None
    return best_planet, best_eta
