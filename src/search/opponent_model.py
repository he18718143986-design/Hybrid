"""B2.5 light opponent for rollout — reinforce threatened + punish exposed (not passive)."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.world.constants import SIM_HORIZON
from src.world.geometry import estimate_arrival
from src.world.orbit import is_orbiting_initial
from src.world.step import Move, StepState, passive_enemy_moves
from src.world.timeline import Arrival, simulate_planet_timeline

# One launch per enemy per rollout turn keeps O(planets^2) and avoids naive spam.
MAX_MOVES_PER_OWNER = 1
REINFORCE_MARGIN = 2
EXPOSED_SHIP_THRESHOLD = 18
THREAT_HORIZON = 12


def opponent_mode() -> str:
    """ORBIT_ROLLOUT_OPPONENT: passive | frontier (default frontier)."""
    return os.environ.get("ORBIT_ROLLOUT_OPPONENT", "frontier").lower()


def _enemy_owners(state: StepState, perspective_player: int) -> List[int]:
    owners = {
        p.owner
        for p in state.planets
        if p.owner not in (-1, perspective_player)
    }
    for _, _, owner, _ in state.scheduled_arrivals:
        if owner not in (-1, perspective_player):
            owners.add(owner)
    return sorted(owners)


def _arrivals_for_planet(state: StepState, planet_id: int) -> List[Arrival]:
    out: List[Arrival] = []
    for abs_step, pid, owner, ships in state.scheduled_arrivals:
        if pid == planet_id:
            rel = abs_step - state.step
            if rel > 0:
                out.append((rel, owner, int(ships)))
    return out


def _planet_lost_to_aggressor(
    planet: Planet,
    events: Sequence[Arrival],
    defender: int,
    aggressor: int,
) -> bool:
    horizon = min(THREAT_HORIZON, SIM_HORIZON)
    tl = simulate_planet_timeline(planet, events, defender, horizon)
    if tl.get("fall_turn") is not None:
        ft = int(tl["fall_turn"])
        if tl["owner_at"].get(ft) == aggressor:
            return True
    for turn, owner in tl.get("owner_at", {}).items():
        if owner == aggressor:
            return True
    for rel, owner, ships in events:
        if owner == aggressor and ships > 0:
            return True
    return False


def _nearest_source(
    state: StepState,
    owner: int,
    target: Planet,
    min_ships: int,
) -> Optional[Tuple[Planet, float, int]]:
    best: Optional[Tuple[float, Planet, float, int]] = None
    for src in state.planets:
        if src.owner != owner or src.id == target.id:
            continue
        cap = int(src.ships)
        if cap < min_ships:
            continue
        send = min(cap, min_ships)
        est = estimate_arrival(
            src.x, src.y, src.radius, target.x, target.y, target.radius, send,
        )
        if est is None:
            continue
        angle, eta = est
        if best is None or eta < best[0]:
            best = (float(eta), src, float(angle), int(send))
    if best is None:
        return None
    _, src, angle, send = best
    return src, angle, send


def _reinforce_move(
    state: StepState,
    defender: int,
    aggressor: int,
) -> Optional[Move]:
    for planet in state.planets:
        if planet.owner != defender:
            continue
        events = _arrivals_for_planet(state, planet.id)
        if not _planet_lost_to_aggressor(planet, events, defender, aggressor):
            continue
        incoming = sum(s for _, o, s in events if o == aggressor)
        need = max(int(planet.ships), incoming) + REINFORCE_MARGIN
        pick = _nearest_source(state, defender, planet, need)
        if pick is None:
            continue
        src, angle, send = pick
        return [src.id, angle, send]
    return None


def _punish_move(
    state: StepState,
    defender: int,
    aggressor: int,
) -> Optional[Move]:
    """Single counter-strike on a thin aggressor planet if reachable."""
    thin = [
        p
        for p in state.planets
        if p.owner == aggressor and int(p.ships) <= EXPOSED_SHIP_THRESHOLD
    ]
    thin.sort(key=lambda p: int(p.ships))
    for target in thin:
        need = int(target.ships) + REINFORCE_MARGIN
        pick = _nearest_source(state, defender, target, need)
        if pick is None:
            continue
        src, angle, send = pick
        return [src.id, angle, send]
    return None


def generate_enemy_moves(state: StepState, perspective_player: int) -> Dict[int, List[Move]]:
    """Frontier punish: threatened reinforce + exposed counter (max 1 move / enemy / turn)."""
    if opponent_mode() == "passive":
        return passive_enemy_moves(state)

    out: Dict[int, List[Move]] = {}
    for enemy in _enemy_owners(state, perspective_player):
        moves: List[Move] = []
        rein = _reinforce_move(state, enemy, perspective_player)
        if rein is not None:
            moves.append(rein)
        if len(moves) < MAX_MOVES_PER_OWNER:
            pun = _punish_move(state, enemy, perspective_player)
            if pun is not None and pun not in moves:
                moves.append(pun)
        out[enemy] = moves[:MAX_MOVES_PER_OWNER]
    return out
