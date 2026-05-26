"""Per-turn world context built from observation (native physics)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from src.env.parser import parse_observation
from src.world.constants import SIM_HORIZON
from src.world.timeline import Arrival, build_arrival_ledger, simulate_planet_timeline


@dataclass
class WorldContext:
    player: int
    step: int
    planets: List[Planet]
    fleets: List[Fleet]
    initial_by_id: Dict[int, Planet]
    comets: list
    comet_ids: Set[int]
    angular_velocity: float
    planet_by_id: Dict[int, Planet] = field(default_factory=dict)
    arrivals_by_planet: Dict[int, List[Arrival]] = field(default_factory=dict)
    base_timeline: Dict[int, dict] = field(default_factory=dict)
    total_visible_ships: int = 0
    total_production: int = 0

    def __post_init__(self):
        self.planet_by_id = {p.id: p for p in self.planets}
        self.total_visible_ships = sum(int(p.ships) for p in self.planets) + sum(
            int(f.ships) for f in self.fleets
        )
        self.total_production = sum(int(p.production) for p in self.planets)


def build_world_context(obs: Any, inferred_step: int | None = None) -> WorldContext:
    parsed = parse_observation(obs)
    step = max(parsed["step"], inferred_step or 0)
    initial_by_id = {p.id: p for p in parsed["initial_planets"]}
    comet_ids = set(parsed["comet_planet_ids"])

    arrivals = build_arrival_ledger(
        parsed["fleets"],
        parsed["planets"],
        initial_by_id,
        parsed["angular_velocity"],
        parsed["comets"],
        comet_ids,
    )
    base_timeline = {
        planet.id: simulate_planet_timeline(
            planet, arrivals[planet.id], parsed["player"], SIM_HORIZON,
        )
        for planet in parsed["planets"]
    }

    return WorldContext(
        player=parsed["player"],
        step=step,
        planets=parsed["planets"],
        fleets=parsed["fleets"],
        initial_by_id=initial_by_id,
        comets=parsed["comets"],
        comet_ids=comet_ids,
        angular_velocity=parsed["angular_velocity"],
        arrivals_by_planet=arrivals,
        base_timeline=base_timeline,
    )
