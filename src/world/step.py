"""One-turn environment step: state + actions -> next state (no agent calls)."""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet

from src.world.combat import resolve_arrival_event
from src.world.comet import (
    advance_comets_one_turn,
    comet_position_at_current_index,
    expired_comet_ids,
)
from src.world.prediction import predict_planet_position
from src.world.constants import SIM_HORIZON, TOTAL_STEPS
from src.world.context import WorldContext, build_world_context
from src.world.fleet import fleet_target_planet
from src.world.geometry import estimate_arrival, launch_point
from src.world.timeline import Arrival, build_arrival_ledger, simulate_planet_timeline

Move = List[Any]  # [from_planet_id, angle, num_ships]
ScheduledArrival = Tuple[int, int, int, int]  # abs_step, planet_id, owner, ships


@dataclass
class StepState:
    """Mutable discrete-event state for rollout (arrival-based, not full continuous fleets)."""

    player: int
    step: int
    planets: List[Planet]
    initial_by_id: Dict[int, Planet]
    comets: list
    comet_ids: Set[int]
    angular_velocity: float
    scheduled_arrivals: List[ScheduledArrival] = field(default_factory=list)
    planet_by_id: Dict[int, Planet] = field(default_factory=dict)

    def __post_init__(self):
        self.planet_by_id = {p.id: p for p in self.planets}

    @classmethod
    def from_context(cls, ctx: WorldContext) -> "StepState":
        scheduled: List[ScheduledArrival] = []
        for planet_id, events in ctx.arrivals_by_planet.items():
            for eta, owner, ships in events:
                if ships > 0 and eta > 0:
                    scheduled.append((ctx.step + int(eta), planet_id, owner, int(ships)))

        return cls(
            player=ctx.player,
            step=ctx.step,
            planets=list(ctx.planets),
            initial_by_id=dict(ctx.initial_by_id),
            comets=ctx.comets,
            comet_ids=set(ctx.comet_ids),
            angular_velocity=ctx.angular_velocity,
            scheduled_arrivals=scheduled,
        )

    def copy(self) -> "StepState":
        return StepState(
            player=self.player,
            step=self.step,
            planets=list(self.planets),
            initial_by_id=dict(self.initial_by_id),
            comets=copy.deepcopy(self.comets),
            comet_ids=set(self.comet_ids),
            angular_velocity=self.angular_velocity,
            scheduled_arrivals=list(self.scheduled_arrivals),
        )


def _player_owners(state: StepState) -> Set[int]:
    owners = {p.owner for p in state.planets if p.owner != -1}
    for _, _, owner, _ in state.scheduled_arrivals:
        owners.add(owner)
    return owners


def passive_enemy_moves(state: StepState) -> Dict[int, List[Move]]:
    """Pessimistic opponent: no launches."""
    enemies = {o for o in _player_owners(state) if o != state.player}
    return {owner: [] for owner in enemies}


def _apply_launches(state: StepState, moves: Sequence[Move], owner: int) -> None:
    if not moves:
        return

    for move in moves:
        if len(move) != 3:
            continue
        from_id, angle, ships = int(move[0]), float(move[1]), int(move[2])
        src = state.planet_by_id.get(from_id)
        if src is None or src.owner != owner:
            continue
        ships = min(ships, int(src.ships))
        if ships < 1:
            continue

        lx, ly = launch_point(src.x, src.y, src.radius, angle)
        fleet = Fleet(-1, owner, lx, ly, angle, from_id, ships)
        target, eta = fleet_target_planet(
            fleet,
            state.planets,
            state.initial_by_id,
            state.angular_velocity,
            state.comets,
            state.comet_ids,
        )
        if target is None or eta is None:
            continue

        abs_step = state.step + int(eta)
        if abs_step > state.step + SIM_HORIZON:
            continue

        state.scheduled_arrivals.append((abs_step, target.id, owner, ships))
        state.planets = [
            p._replace(ships=int(p.ships) - ships) if p.id == from_id else p
            for p in state.planets
        ]
        state.planet_by_id = {p.id: p for p in state.planets}


def _production(state: StepState) -> None:
    state.planets = [
        p._replace(ships=int(p.ships) + int(p.production))
        if p.owner != -1
        else p
        for p in state.planets
    ]
    state.planet_by_id = {p.id: p for p in state.planets}


def _resolve_arrivals_for_step(state: StepState, arrival_step: int) -> None:
    by_planet: Dict[int, List[Arrival]] = defaultdict(list)
    remaining: List[ScheduledArrival] = []

    for abs_step, planet_id, owner, ships in state.scheduled_arrivals:
        if abs_step == arrival_step:
            by_planet[planet_id].append((1, owner, ships))
        elif abs_step > arrival_step:
            remaining.append((abs_step, planet_id, owner, ships))

    for planet_id, arrivals in by_planet.items():
        planet = state.planet_by_id[planet_id]
        new_owner, garrison = resolve_arrival_event(
            planet.owner, float(planet.ships), arrivals,
        )
        state.planets = [
            p._replace(owner=new_owner, ships=int(max(0, math.floor(garrison))))
            if p.id == planet_id
            else p
            for p in state.planets
        ]

    state.scheduled_arrivals = remaining
    state.planet_by_id = {p.id: p for p in state.planets}


def _advance_positions(state: StepState) -> None:
    """Orbiting planets + comet path_index (B2 fidelity)."""
    state.comets = advance_comets_one_turn(state.comets)
    expired = expired_comet_ids(state.comets)

    updated: List[Planet] = []
    for p in state.planets:
        if p.id in expired:
            continue
        if p.id in state.comet_ids:
            pos = comet_position_at_current_index(p.id, state.comets)
            if pos is None:
                updated.append(p)
            else:
                updated.append(p._replace(x=pos[0], y=pos[1]))
        else:
            nx, ny = predict_planet_position(
                p, state.initial_by_id, state.angular_velocity, 1,
            )
            updated.append(p._replace(x=nx, y=ny))

    state.comet_ids = {pid for pid in state.comet_ids if pid not in expired}
    state.planets = updated
    state.planet_by_id = {p.id: p for p in state.planets}


def simulate_one_turn(
    state: StepState,
    moves_by_player: Dict[int, Sequence[Move]],
) -> StepState:
    """Advance one turn: launches -> production -> arrivals -> orbit/comets.

    Does not call any agent. Only uses src/world physics.
    """
    if state.step >= TOTAL_STEPS:
        return state

    next_state = state.copy()

    for owner, moves in moves_by_player.items():
        _apply_launches(next_state, moves, owner)

    _production(next_state)

    arrival_step = next_state.step + 1
    _resolve_arrivals_for_step(next_state, arrival_step)
    _advance_positions(next_state)

    next_state.step = arrival_step
    return next_state


def step_state_to_context(state: StepState) -> WorldContext:
    """Rebuild WorldContext for need/timeline queries after simulation."""
    fleets: List[Fleet] = []
    arrivals_by_planet: Dict[int, List[Arrival]] = {p.id: [] for p in state.planets}

    for abs_step, planet_id, owner, ships in state.scheduled_arrivals:
        rel = abs_step - state.step
        if rel > 0:
            arrivals_by_planet[planet_id].append((rel, owner, ships))

    if fleets:
        ledger = build_arrival_ledger(
            fleets,
            state.planets,
            state.initial_by_id,
            state.angular_velocity,
            state.comets,
            state.comet_ids,
        )
        for pid, events in ledger.items():
            arrivals_by_planet[pid].extend(events)

    base_timeline = {
        planet.id: simulate_planet_timeline(
            planet, arrivals_by_planet[planet.id], state.player, SIM_HORIZON,
        )
        for planet in state.planets
    }

    total_visible = sum(int(p.ships) for p in state.planets) + sum(
        s for _, _, _, s in state.scheduled_arrivals
    )

    return WorldContext(
        player=state.player,
        step=state.step,
        planets=state.planets,
        fleets=fleets,
        initial_by_id=state.initial_by_id,
        comets=state.comets,
        comet_ids=state.comet_ids,
        angular_velocity=state.angular_velocity,
        arrivals_by_planet=arrivals_by_planet,
        base_timeline=base_timeline,
        total_visible_ships=total_visible,
        total_production=sum(int(p.production) for p in state.planets),
    )


def apply_candidate_launch(
    state: StepState,
    src_id: int,
    target_id: int,
    angle: float,
    ships: int,
    owner: Optional[int] = None,
) -> None:
    """Schedule one launch (used by rollout to inject the candidate move)."""
    owner = state.player if owner is None else owner
    src = state.planet_by_id.get(src_id)
    target = state.planet_by_id.get(target_id)
    if src is None or target is None or src.owner != owner:
        return

    ships = min(int(ships), int(src.ships))
    if ships < 1:
        return

    est = estimate_arrival(
        src.x, src.y, src.radius, target.x, target.y, target.radius, ships,
    )
    if est is None:
        return
    _, eta = est

    abs_step = state.step + int(eta)
    state.scheduled_arrivals.append((abs_step, target_id, owner, ships))
    state.planets = [
        p._replace(ships=int(p.ships) - ships) if p.id == src_id else p
        for p in state.planets
    ]
    state.planet_by_id = {p.id: p for p in state.planets}
