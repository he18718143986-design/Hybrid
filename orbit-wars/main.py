"""Orbit Wars agent main.py skeleton.

Modular single-file scaffold for iterative development.

Status:
  [x] predict_position (static / orbit / comet paths)
  [x] segment_hits_sun, compute_intercept (iterative + degrade)
  [x] arrival_ledger, timeline, resolve_arrival_event
  [x] min_ships_to_own_by, settle_plan, planned_commitments
  [ ] build_modes / mission builders (reinforce, snipe, …)

No full-match trajectory table — orbit uses current-angle + angular_velocity.

See agents.md and ORBIT_WARS_ARCHITECTURE.md in the parent repo for production reference.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple

from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, Fleet, Planet

# =============================================================================
# Global constants / knobs
# =============================================================================

BOARD_SIZE: float = 100.0
SUN_RADIUS: float = 10.0
ROTATION_LIMIT: float = 50.0
MAX_TURNS: int = 500
DEFAULT_ACT_TIMEOUT_S: float = 1.0
SOFT_ACT_CAP: float = 0.82
SOFT_ACT_FLOOR: float = 0.55
INTERCEPT_TOLERANCE: int = 1
INTERCEPT_ITERATIONS: int = 5
SUN_SAFETY: float = 1.5
LAUNCH_CLEARANCE: float = 0.1

DEFENSE_MARGIN: float = 0.20
SIM_HORIZON: int = 110
ROUTE_SEARCH_HORIZON: int = 60
PARTIAL_SOURCE_MIN_SHIPS: int = 6
SETTLE_MAX_ITER: int = 4

NEUTRAL_MARGIN_BASE: int = 2
NEUTRAL_MARGIN_PROD_WEIGHT: int = 2
NEUTRAL_MARGIN_CAP: int = 8
HOSTILE_MARGIN_BASE: int = 3
HOSTILE_MARGIN_PROD_WEIGHT: int = 2
HOSTILE_MARGIN_CAP: int = 12
STATIC_TARGET_MARGIN: int = 4
LONG_TRAVEL_MARGIN_START: int = 18
LONG_TRAVEL_MARGIN_DIVISOR: int = 3
LONG_TRAVEL_MARGIN_CAP: int = 8

PlannedCommitments = Dict[int, List[Tuple[int, int, int]]]
SettledPlan = Tuple[float, int, int, int, int]  # angle, eta, eval_turn, need, send
SettleEval = Tuple[float, int, int, int, int, int]  # + desired (internal)

# Per-match cache (invalidated when fingerprint changes)
_MATCH_FP: Optional[tuple] = None
_ORBITAL_METADATA: Optional[Dict[int, "OrbitMeta"]] = None

# =============================================================================
# Types
# =============================================================================

Move = List[object]  # [from_planet_id, direction_angle, num_ships]
Point = Tuple[float, float]


class ComputeTier(Enum):
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


@dataclass(frozen=True)
class ShotOption:
    src_id: int
    target_id: int
    angle: float
    ships: int
    eta: int
    score: float


@dataclass
class Mission:
    kind: str
    target_id: int
    priority: float = 0.0
    options: List[ShotOption] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OrbitMeta:
    is_orbiting: bool
    orbital_radius: float = 0.0


# =============================================================================
# Observation helpers
# =============================================================================


def _read(obs, key: str, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _game_fingerprint(obs) -> tuple:
    init = tuple(tuple(p) for p in (_read(obs, "initial_planets", []) or []))
    comet_ids = tuple(sorted(_read(obs, "comet_planet_ids", []) or []))
    ang = float(_read(obs, "angular_velocity", 0.0) or 0.0)
    return (init, round(ang, 8), comet_ids)


def _build_orbital_metadata(obs, comet_ids: set) -> Dict[int, OrbitMeta]:
    meta: Dict[int, OrbitMeta] = {}
    initial_by_id = {
        p.id: p for p in [Planet(*row) for row in (_read(obs, "initial_planets", []) or [])]
    }
    for planet_id, init_p in initial_by_id.items():
        if planet_id in comet_ids:
            continue
        dx = init_p.x - CENTER[0]
        dy = init_p.y - CENTER[1]
        r = math.hypot(dx, dy)
        orbiting = (r + init_p.radius) < ROTATION_LIMIT
        meta[planet_id] = OrbitMeta(is_orbiting=orbiting, orbital_radius=r)
    return meta


def _ensure_orbital_metadata(obs, comet_ids: set) -> Dict[int, OrbitMeta]:
    global _MATCH_FP, _ORBITAL_METADATA
    fp = _game_fingerprint(obs)
    if _MATCH_FP == fp and _ORBITAL_METADATA is not None:
        return _ORBITAL_METADATA
    _MATCH_FP = fp
    _ORBITAL_METADATA = _build_orbital_metadata(obs, comet_ids)
    return _ORBITAL_METADATA


# =============================================================================
# Position prediction (L1)
# =============================================================================


def predict_comet_position(planet_id: int, comets, turn_offset: int) -> Optional[Point]:
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


def predict_orbiting_planet_position(
    planet: Planet,
    meta: OrbitMeta,
    angular_velocity: float,
    turn_offset: int,
) -> Point:
    if not meta.is_orbiting:
        return (planet.x, planet.y)
    r = meta.orbital_radius
    cur_ang = math.atan2(planet.y - CENTER[1], planet.x - CENTER[0])
    new_ang = cur_ang + float(angular_velocity) * int(turn_offset)
    return (
        CENTER[0] + r * math.cos(new_ang),
        CENTER[1] + r * math.sin(new_ang),
    )


def predict_target_position(
    planet: Planet,
    turn_offset: int,
    comet_ids: set,
    comets,
    orbital_metadata: Dict[int, OrbitMeta],
    angular_velocity: float,
) -> Optional[Point]:
    if planet.id in comet_ids:
        return predict_comet_position(planet.id, comets, turn_offset)
    meta = orbital_metadata.get(planet.id, OrbitMeta(is_orbiting=False))
    return predict_orbiting_planet_position(planet, meta, angular_velocity, turn_offset)


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


# =============================================================================
# Arrival ledger / combat timeline (L1)
# =============================================================================


def _dist_xy(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def fleet_target_planet(
    fleet: Fleet,
    planets: Sequence[Planet],
    initial_by_id: Dict[int, Planet],
    angular_velocity: float,
    comets,
    comet_ids: set,
    orbital_metadata: Dict[int, OrbitMeta],
    horizon: int = SIM_HORIZON,
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
        r0 = _dist_xy(init.x, init.y, CENTER[0], CENTER[1])
        if r0 + init.radius >= ROTATION_LIMIT:
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
        if turns > horizon:
            continue
        eta = max(1, int(math.ceil(turns)))
        if best_eta is None or eta < best_eta:
            best_eta = eta
            best_planet = planet

    if rotating_candidates:
        radius_sq_cache = {p.id: p.radius * p.radius for p in rotating_candidates}
        upper_t = best_eta if best_eta is not None else horizon
        for t in range(1, upper_t + 1):
            if best_eta is not None and t >= best_eta:
                break
            fx = fleet.x + dir_x * speed * t
            fy = fleet.y + dir_y * speed * t
            for planet in rotating_candidates:
                pos = predict_target_position(
                    planet, t, comet_ids, comets, orbital_metadata, angular_velocity,
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


def build_arrival_ledger(
    fleets: Sequence[Fleet],
    planets: Sequence[Planet],
    initial_by_id: Dict[int, Planet],
    angular_velocity: float,
    comets,
    comet_ids: set,
    orbital_metadata: Dict[int, OrbitMeta],
) -> Dict[int, List[Tuple[int, int, int]]]:
    arrivals_by_planet: Dict[int, List[Tuple[int, int, int]]] = {
        planet.id: [] for planet in planets
    }
    for fleet in fleets:
        target, eta = fleet_target_planet(
            fleet,
            planets,
            initial_by_id,
            angular_velocity,
            comets,
            comet_ids,
            orbital_metadata,
        )
        if target is None or eta is None:
            continue
        arrivals_by_planet[target.id].append((eta, fleet.owner, int(fleet.ships)))
    return arrivals_by_planet


def resolve_arrival_event(owner: int, garrison: float, arrivals: Sequence[Tuple[int, int, int]]):
    by_owner: Dict[int, int] = {}
    for _, attacker_owner, ships in arrivals:
        by_owner[attacker_owner] = by_owner.get(attacker_owner, 0) + ships

    if not by_owner:
        return owner, max(0.0, garrison)

    sorted_players = sorted(by_owner.items(), key=lambda item: item[1], reverse=True)
    top_owner, top_ships = sorted_players[0]

    if len(sorted_players) > 1:
        second_ships = sorted_players[1][1]
        if top_ships == second_ships:
            survivor_owner = -1
            survivor_ships = 0
        else:
            survivor_owner = top_owner
            survivor_ships = top_ships - second_ships
    else:
        survivor_owner = top_owner
        survivor_ships = top_ships

    if survivor_ships <= 0:
        return owner, max(0.0, garrison)

    if owner == survivor_owner:
        return owner, garrison + survivor_ships

    garrison -= survivor_ships
    if garrison < 0:
        return survivor_owner, -garrison
    return owner, garrison


def normalize_arrivals(arrivals: Sequence[Tuple[int, int, int]], horizon: int):
    events = []
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
    arrivals: Sequence[Tuple[int, int, int]],
    player: int,
    horizon: int,
) -> dict:
    horizon = max(0, int(math.ceil(horizon)))
    events = normalize_arrivals(arrivals, horizon)
    by_turn: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
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


# =============================================================================
# Time budget
# =============================================================================


class TimeBudget:
    """Soft deadline inside actTimeout; heavy loops should call expired()."""

    @classmethod
    def from_config(cls, config=None) -> "TimeBudget":
        config = config or {}
        act_timeout = float(config.get("actTimeout", DEFAULT_ACT_TIMEOUT_S))
        soft = min(SOFT_ACT_CAP, max(SOFT_ACT_FLOOR, act_timeout * SOFT_ACT_CAP))
        return cls(soft_seconds=soft)

    def __init__(self, soft_seconds: float):
        self._start = time.perf_counter()
        self.deadline = self._start + soft_seconds
        self.tier = ComputeTier.LIGHT

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.perf_counter())

    def expired(self) -> bool:
        return time.perf_counter() >= self.deadline

    def should_downgrade(self) -> bool:
        return self.remaining() < 0.10

    def refresh_tier(self) -> None:
        r = self.remaining()
        if r > 0.50:
            self.tier = ComputeTier.HEAVY
        elif r > 0.20:
            self.tier = ComputeTier.MEDIUM
        else:
            self.tier = ComputeTier.LIGHT


# =============================================================================
# Utility / geometry (L2)
# =============================================================================


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def angle_between(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def point_to_segment_distance(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return dist(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
    t = clamp(t, 0.0, 1.0)
    proj = (ax + t * dx, ay + t * dy)
    return dist(p, proj)


def segment_hits_sun(a: Point, b: Point, safety: float = SUN_SAFETY) -> bool:
    return point_to_segment_distance(CENTER, a, b) < SUN_RADIUS + safety


def launch_point(sx: float, sy: float, sr: float, angle: float) -> Point:
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def path_end(sx: float, sy: float, sr: float, tx: float, ty: float, tr: float) -> Tuple[float, Point, Point]:
    angle = math.atan2(ty - sy, tx - sx)
    start = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist((sx, sy), (tx, ty)) - (sr + LAUNCH_CLEARANCE) - tr)
    end = (
        start[0] + math.cos(angle) * hit_distance,
        start[1] + math.sin(angle) * hit_distance,
    )
    return angle, start, end


def safe_angle_and_distance(
    sx: float, sy: float, sr: float, tx: float, ty: float, tr: float,
) -> Optional[Tuple[float, float]]:
    angle, start, end = path_end(sx, sy, sr, tx, ty, tr)
    if segment_hits_sun(start, end):
        return None
    return angle, dist((sx, sy), (tx, ty)) - (sr + LAUNCH_CLEARANCE) - tr


@lru_cache(maxsize=4096)
def fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    ships = max(1, int(ships))
    ratio = max(0.0, min(1.0, math.log(ships) / math.log(1000.0)))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def estimate_arrival(sx: float, sy: float, sr: float, tx: float, ty: float, tr: float, ships: int) -> Optional[Tuple[float, int]]:
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    total_d = max(0.0, total_d)
    turns = max(1, int(math.ceil(total_d / fleet_speed(ships))))
    return angle, turns


def target_can_move(
    target: Planet,
    initial_by_id: Dict[int, Planet],
    comet_ids: set,
    orbital_metadata: Dict[int, OrbitMeta],
) -> bool:
    if target.id in comet_ids:
        return True
    init = initial_by_id.get(target.id)
    if init is None:
        return False
    meta = orbital_metadata.get(target.id)
    if meta is not None:
        return meta.is_orbiting
    r = _dist_xy(init.x, init.y, CENTER[0], CENTER[1])
    return r + init.radius < ROTATION_LIMIT


def search_safe_intercept(
    src: Planet,
    target: Planet,
    ships: int,
    state: "GameState",
) -> Optional[Tuple[float, int, float, float]]:
    best = None
    best_score = None
    max_turns = min(SIM_HORIZON, ROUTE_SEARCH_HORIZON)
    if target.id in state.comet_ids:
        max_turns = min(max_turns, max(0, comet_remaining_life(target.id, state.comets) - 1))

    for candidate_turns in range(1, max_turns + 1):
        pos = predict_target_position(
            target,
            candidate_turns,
            state.comet_ids,
            state.comets,
            state.orbital_metadata,
            state.angular_velocity,
        )
        if pos is None:
            continue
        est = estimate_arrival(src.x, src.y, src.radius, pos[0], pos[1], target.radius, ships)
        if est is None:
            continue
        _, turns = est
        if abs(turns - candidate_turns) > INTERCEPT_TOLERANCE:
            continue

        actual_turns = max(turns, candidate_turns)
        actual_pos = predict_target_position(
            target,
            actual_turns,
            state.comet_ids,
            state.comets,
            state.orbital_metadata,
            state.angular_velocity,
        )
        if actual_pos is None:
            continue

        confirm = estimate_arrival(
            src.x, src.y, src.radius, actual_pos[0], actual_pos[1], target.radius, ships,
        )
        if confirm is None:
            continue

        delta = abs(confirm[1] - actual_turns)
        if delta > INTERCEPT_TOLERANCE:
            continue

        score = (delta, confirm[1], candidate_turns)
        if best is None or score < best_score:
            best_score = score
            best = (confirm[0], confirm[1], actual_pos[0], actual_pos[1])

    return best


def aim_with_prediction(
    src: Planet,
    target: Planet,
    ships: int,
    state: "GameState",
) -> Optional[Tuple[float, int, float, float]]:
    est = estimate_arrival(src.x, src.y, src.radius, target.x, target.y, target.radius, ships)
    if est is None:
        if not target_can_move(
            target, state.initial_by_id, state.comet_ids, state.orbital_metadata,
        ):
            return None
        return search_safe_intercept(src, target, ships, state)

    tx, ty = target.x, target.y
    for _ in range(INTERCEPT_ITERATIONS):
        _, turns = est
        pos = predict_target_position(
            target,
            turns,
            state.comet_ids,
            state.comets,
            state.orbital_metadata,
            state.angular_velocity,
        )
        if pos is None:
            return None
        ntx, nty = pos
        next_est = estimate_arrival(src.x, src.y, src.radius, ntx, nty, target.radius, ships)
        if next_est is None:
            if not target_can_move(
                target, state.initial_by_id, state.comet_ids, state.orbital_metadata,
            ):
                return None
            return search_safe_intercept(src, target, ships, state)
        if (
            abs(ntx - tx) < 0.3
            and abs(nty - ty) < 0.3
            and abs(next_est[1] - turns) <= INTERCEPT_TOLERANCE
        ):
            return next_est[0], next_est[1], ntx, nty
        tx, ty = ntx, nty
        est = next_est

    final_est = estimate_arrival(src.x, src.y, src.radius, tx, ty, target.radius, ships)
    if final_est is None:
        return search_safe_intercept(src, target, ships, state)
    return final_est[0], final_est[1], tx, ty


def preferred_send(
    target: Planet,
    base_needed: int,
    arrival_turns: int,
    src_available: int,
    state: "GameState",
    modes: dict,
) -> int:
    send = max(base_needed, int(math.ceil(base_needed * modes.get("attack_margin_mult", 1.0))))
    margin = 0
    if target.owner == -1:
        margin += min(
            NEUTRAL_MARGIN_CAP,
            NEUTRAL_MARGIN_BASE + target.production * NEUTRAL_MARGIN_PROD_WEIGHT,
        )
    else:
        margin += min(
            HOSTILE_MARGIN_CAP,
            HOSTILE_MARGIN_BASE + target.production * HOSTILE_MARGIN_PROD_WEIGHT,
        )
    meta = state.orbital_metadata.get(target.id)
    if meta is not None and not meta.is_orbiting:
        margin += STATIC_TARGET_MARGIN
    if arrival_turns > LONG_TRAVEL_MARGIN_START:
        margin += min(LONG_TRAVEL_MARGIN_CAP, arrival_turns // LONG_TRAVEL_MARGIN_DIVISOR)
    return min(src_available, send + margin)


def settle_plan(
    src: Planet,
    target: Planet,
    src_cap: int,
    send_guess: int,
    state: "GameState",
    planned_commitments: PlannedCommitments,
    modes: dict,
    mission: str = "capture",
    eval_turn_fn=None,
    anchor_turn: Optional[int] = None,
    anchor_tolerance: Optional[int] = None,
    max_iter: int = SETTLE_MAX_ITER,
) -> Optional[SettledPlan]:
    if src_cap < 1:
        return None

    seed_hint = max(1, min(src_cap, int(send_guess)))
    eval_turn_fn = eval_turn_fn or (lambda turns: turns)
    anchor_tolerance = anchor_tolerance if anchor_tolerance is not None else (
        1 if mission == "snipe" else None
    )
    tested: Dict[int, Optional[SettleEval]] = {}
    tested_order: List[int] = []

    def evaluate(send: int) -> Optional[SettleEval]:
        send = max(1, min(src_cap, int(send)))
        if send in tested:
            return tested[send]

        aim = state.plan_shot(src.id, target.id, send)
        if aim is None:
            tested[send] = None
            return None

        angle, turns, _, _ = aim
        if mission == "crash_exploit" and anchor_turn is not None and turns < anchor_turn:
            tested[send] = None
            return None

        raw_eval_turn = int(math.ceil(eval_turn_fn(turns)))
        if raw_eval_turn < turns:
            tested[send] = None
            return None
        eval_turn = raw_eval_turn

        need = state.min_ships_to_own_by(
            target.id,
            eval_turn,
            state.player,
            arrival_turn=turns,
            planned_commitments=planned_commitments,
            upper_bound=src_cap,
        )
        if need <= 0 or need > src_cap:
            tested[send] = None
            return None

        if mission in ("snipe", "crash_exploit"):
            desired = need
        else:
            desired = min(
                src_cap,
                max(need, preferred_send(target, need, turns, src_cap, state, modes)),
            )

        result: SettleEval = (angle, turns, eval_turn, need, send, desired)
        tested[send] = result
        tested_order.append(send)
        return result

    initial_candidates = sorted(
        state.probe_ship_candidates(src.id, target.id, src_cap, hints=(seed_hint,)),
        key=lambda s: (abs(s - seed_hint), s),
    )

    current_send = None
    for seed in initial_candidates:
        result = evaluate(seed)
        if result is None:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(result[1] - anchor_turn) > anchor_tolerance
        ):
            continue
        current_send = seed
        break

    if current_send is None:
        return None

    for _ in range(max_iter):
        result = evaluate(current_send)
        if result is None:
            break

        angle, turns, eval_turn, need, actual_send, desired = result
        if desired == actual_send:
            if (
                anchor_turn is not None
                and anchor_tolerance is not None
                and abs(turns - anchor_turn) > anchor_tolerance
            ):
                return None
            return (angle, turns, eval_turn, need, actual_send)

        next_send = max(1, min(src_cap, int(desired)))
        if next_send in tested:
            current_send = next_send
            break
        current_send = next_send

    candidate_sends = sorted(
        [send for send in tested_order if tested.get(send) is not None],
        key=lambda send: (
            0 if mission != "snipe" or anchor_turn is None else abs(tested[send][1] - anchor_turn),
            abs(send - seed_hint),
            tested[send][1],
            send,
        ),
    )

    seen = set()
    for send in candidate_sends:
        if send in seen:
            continue
        seen.add(send)
        result = tested.get(send)
        if result is None:
            continue
        angle, turns, eval_turn, need, actual_send, desired = result
        if actual_send < need:
            continue
        if (
            anchor_turn is not None
            and anchor_tolerance is not None
            and abs(turns - anchor_turn) > anchor_tolerance
        ):
            continue
        return (angle, turns, eval_turn, need, actual_send)

    return None


# =============================================================================
# Match-state / world model
# =============================================================================


class GameState:
    """Per-turn view; orbit metadata cached at module scope per match fingerprint."""

    def __init__(self, obs):
        self.raw = obs
        self.step: int = int(_read(obs, "step", 0) or 0)
        self.player: int = int(_read(obs, "player", 0))
        self.angular_velocity: float = float(_read(obs, "angular_velocity", 0.0) or 0.0)
        self.comets = _read(obs, "comets", []) or []
        self.comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
        self.planets: List[Planet] = [Planet(*p) for p in (_read(obs, "planets", []) or [])]
        self.fleets: List[Fleet] = [Fleet(*f) for f in (_read(obs, "fleets", []) or [])]
        self.initial_planets: List[Planet] = [Planet(*p) for p in (_read(obs, "initial_planets", []) or [])]

        self.planet_by_id: Dict[int, Planet] = {p.id: p for p in self.planets}
        self.initial_by_id: Dict[int, Planet] = {p.id: p for p in self.initial_planets}
        self.orbital_metadata: Dict[int, OrbitMeta] = _ensure_orbital_metadata(obs, self.comet_ids)

        self.arrivals_by_planet = build_arrival_ledger(
            self.fleets,
            self.planets,
            self.initial_by_id,
            self.angular_velocity,
            self.comets,
            self.comet_ids,
            self.orbital_metadata,
        )
        self.base_timeline = {
            planet.id: simulate_planet_timeline(
                planet,
                self.arrivals_by_planet[planet.id],
                self.player,
                SIM_HORIZON,
            )
            for planet in self.planets
        }
        self.shot_cache: Dict[Tuple[int, int, int], Optional[Tuple[float, int, float, float]]] = {}
        self.probe_candidate_cache: Dict[tuple, List[int]] = {}

    def is_static(self, planet_id: int) -> bool:
        meta = self.orbital_metadata.get(planet_id)
        return meta is None or not meta.is_orbiting

    def plan_shot(
        self, src_id: int, target_id: int, ships: int,
    ) -> Optional[Tuple[float, int, float, float]]:
        ships = int(ships)
        key = (src_id, target_id, ships)
        if key in self.shot_cache:
            return self.shot_cache[key]
        src = self.planet_by_id[src_id]
        target = self.planet_by_id[target_id]
        result = aim_with_prediction(src, target, ships, self)
        self.shot_cache[key] = result
        return result

    def probe_ship_candidates(
        self,
        src_id: int,
        target_id: int,
        source_cap: int,
        hints: Sequence[int] = (),
    ) -> List[int]:
        source_cap = max(1, int(source_cap))
        normalized_hints = tuple(int(math.ceil(h)) for h in hints if h is not None)
        cache_key = (src_id, target_id, source_cap, normalized_hints)
        cached = self.probe_candidate_cache.get(cache_key)
        if cached is not None:
            return cached

        target = self.planet_by_id[target_id]
        target_ships = max(1, int(math.ceil(target.ships)))
        values = set(range(1, min(6, source_cap) + 1))
        values.update({
            source_cap,
            max(1, source_cap // 2),
            max(1, source_cap // 3),
            min(source_cap, PARTIAL_SOURCE_MIN_SHIPS),
            min(source_cap, target_ships + 1),
            min(source_cap, target_ships + 2),
            min(source_cap, target_ships + 4),
            min(source_cap, target_ships + 8),
        })
        for hint in normalized_hints:
            base = max(1, min(source_cap, hint))
            for delta in (-2, -1, 0, 1, 2):
                candidate = base + delta
                if 1 <= candidate <= source_cap:
                    values.add(candidate)
        result = sorted(values)
        self.probe_candidate_cache[cache_key] = result
        return result

    def predict_position(self, planet_id: int, turn_offset: int = 0) -> Point:
        planet = self.planet_by_id.get(planet_id)
        if planet is None:
            return CENTER

        if planet_id in self.comet_ids:
            pos = predict_comet_position(planet_id, self.comets, turn_offset)
            return pos if pos is not None else (planet.x, planet.y)

        meta = self.orbital_metadata.get(planet_id, OrbitMeta(is_orbiting=False))
        return predict_orbiting_planet_position(
            planet, meta, self.angular_velocity, turn_offset,
        )

    def my_planets(self) -> List[Planet]:
        return [p for p in self.planets if p.owner == self.player]

    def enemy_planets(self) -> List[Planet]:
        return [p for p in self.planets if p.owner not in (-1, self.player)]

    def neutral_planets(self) -> List[Planet]:
        return [p for p in self.planets if p.owner == -1]

    def current_planet(self, planet_id: int) -> Optional[Planet]:
        return self.planet_by_id.get(planet_id)

    def projected_timeline(
        self,
        target_id: int,
        horizon: int,
        planned_commitments: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
    ) -> dict:
        planned_commitments = planned_commitments or {}
        horizon = max(1, int(math.ceil(horizon)))
        arrivals = [
            item
            for item in self.arrivals_by_planet.get(target_id, [])
            if item[0] <= horizon
        ]
        arrivals.extend(
            item
            for item in planned_commitments.get(target_id, [])
            if item[0] <= horizon
        )
        arrivals.extend(item for item in extra_arrivals if item[0] <= horizon)
        target = self.planet_by_id[target_id]
        return simulate_planet_timeline(target, arrivals, self.player, horizon)

    def projected_state(
        self,
        target_id: int,
        eval_turn: int,
        planned_commitments: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
    ) -> Tuple[int, float]:
        cutoff = max(1, int(math.ceil(eval_turn)))
        timeline = self.projected_timeline(
            target_id, cutoff, planned_commitments, extra_arrivals,
        )
        return state_at_timeline(timeline, cutoff)

    def min_ships_to_own_by(
        self,
        target_id: int,
        eval_turn: int,
        attacker_owner: int,
        arrival_turn: Optional[int] = None,
        planned_commitments: Optional[PlannedCommitments] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
        upper_bound: Optional[int] = None,
    ) -> int:
        """Ships for attacker_owner to own target by eval_turn (fleet arrives at arrival_turn)."""
        planned_commitments = planned_commitments or {}
        eval_turn = max(1, int(math.ceil(eval_turn)))
        arrival_turn = eval_turn if arrival_turn is None else max(1, int(math.ceil(arrival_turn)))

        if arrival_turn > eval_turn:
            if upper_bound is not None:
                return max(1, int(upper_bound)) + 1
            return 10**6

        normalized_extra = tuple(
            (max(1, int(math.ceil(turns))), owner, int(ships))
            for turns, owner, ships in extra_arrivals
            if ships > 0 and max(1, int(math.ceil(turns))) <= eval_turn
        )

        owner_before, ships_before = self.projected_state(
            target_id, eval_turn, planned_commitments, normalized_extra,
        )
        if owner_before == attacker_owner:
            return 0

        def owns_at(ships: int) -> bool:
            owner_after, _ = self.projected_state(
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
            search_cap = max(32, int(sum(p.ships for p in self.planets) + sum(f.ships for f in self.fleets) + 32))
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
        self,
        target_id: int,
        arrival_turn: int,
        attacker_owner: Optional[int] = None,
        planned_commitments: Optional[PlannedCommitments] = None,
        extra_arrivals: Sequence[Tuple[int, int, int]] = (),
        upper_bound: Optional[int] = None,
    ) -> int:
        attacker_owner = self.player if attacker_owner is None else attacker_owner
        turn = max(1, int(math.ceil(arrival_turn)))
        return self.min_ships_to_own_by(
            target_id,
            turn,
            attacker_owner,
            arrival_turn=turn,
            planned_commitments=planned_commitments,
            extra_arrivals=extra_arrivals,
            upper_bound=upper_bound,
        )


# =============================================================================
# Layer 3: strategic planner
# =============================================================================


class StrategicPlanner:
    def __init__(self, state: GameState, budget: TimeBudget):
        self.state = state
        self.budget = budget

    def build_modes(self) -> dict:
        # TODO: domination / finishing / opening_filter hooks
        step = self.state.step
        return {
            "opening": step < 40,
            "midgame": 40 <= step < 430,
            "endgame": step >= 430,
            "is_four_player": len({p.owner for p in self.state.planets if p.owner != -1}) >= 3,
            "attack_margin_mult": 1.0,
            "is_finishing": False,
        }

    def target_value(self, planet: Planet) -> float:
        if planet.owner == self.state.player:
            return -1e9
        production = float(planet.production)
        ships = float(max(1, planet.ships))
        nearest = self._nearest_my_distance(planet)
        return production * 10.0 + (20.0 / (1.0 + nearest)) + (1.0 / (1.0 + ships)) * 3.0

    def _nearest_my_distance(self, target: Planet) -> float:
        mines = self.state.my_planets()
        if not mines:
            return 1e9
        return min(dist((m.x, m.y), (target.x, target.y)) for m in mines)

    def select_objectives(self) -> List[Mission]:
        missions: List[Mission] = []
        for p in self.state.planets:
            if p.owner == self.state.player:
                continue
            score = self.target_value(p)
            if score <= -1e8:
                continue
            kind = "capture" if p.owner == -1 else "attack"
            missions.append(Mission(kind=kind, target_id=p.id, priority=score))
        missions.sort(key=lambda m: m.priority, reverse=True)
        return missions[:8]


# =============================================================================
# Layer 4: tactical execution
# =============================================================================


class TacticalExecutor:
    def __init__(self, state: GameState, budget: TimeBudget):
        self.state = state
        self.budget = budget

    def compute_intercept(self, src: Planet, dst: Planet, ships: int) -> Tuple[float, int]:
        """Return (angle, eta). MEDIUM; degrades to current-position aim."""
        if self.budget.tier == ComputeTier.LIGHT or self.budget.should_downgrade():
            return angle_between((src.x, src.y), (dst.x, dst.y)), max(
                1,
                int(math.ceil(dist((src.x, src.y), (dst.x, dst.y)) / fleet_speed(ships))),
            )

        tx, ty = dst.x, dst.y
        for _ in range(INTERCEPT_ITERATIONS):
            if self.budget.expired():
                break
            est = estimate_arrival(src.x, src.y, src.radius, tx, ty, dst.radius, ships)
            if est is None:
                return angle_between((src.x, src.y), (dst.x, dst.y)), 10**9
            angle, turns = est
            future = self.state.predict_position(dst.id, turn_offset=turns)
            ntx, nty = future
            confirm = estimate_arrival(src.x, src.y, src.radius, ntx, nty, dst.radius, ships)
            if confirm is None:
                return angle_between((src.x, src.y), (dst.x, dst.y)), turns
            if (
                abs(ntx - tx) < 0.3
                and abs(nty - ty) < 0.3
                and abs(confirm[1] - turns) <= INTERCEPT_TOLERANCE
            ):
                return confirm[0], confirm[1]
            tx, ty = ntx, nty

        est = estimate_arrival(src.x, src.y, src.radius, tx, ty, dst.radius, ships)
        if est is None:
            return angle_between((src.x, src.y), (dst.x, dst.y)), 10**9
        return est

    def simulate_combat(
        self,
        planet_id: int,
        incoming_fleets: Sequence[Fleet],
        arrival_turn: int = 1,
    ) -> dict:
        """Single-turn combat resolution via resolve_arrival_event (rules-aligned)."""
        planet = self.state.current_planet(planet_id)
        if planet is None:
            return {"owner": -1, "ships": 0}

        arrivals = [
            (max(1, int(arrival_turn)), f.owner, int(f.ships))
            for f in incoming_fleets
            if f.ships > 0
        ]
        owner, ships = resolve_arrival_event(planet.owner, float(planet.ships), arrivals)
        return {"owner": owner, "ships": int(max(0, ships))}

    def plan_attack(
        self,
        objective: Mission,
        planned_commitments: PlannedCommitments,
        modes: dict,
        source_cap_fn,
    ) -> List[ShotOption]:
        """Build shot options via settle_plan (need + margin iteration)."""
        target = self.state.current_planet(objective.target_id)
        if target is None:
            return []

        mission = objective.kind if objective.kind in ("capture", "attack", "snipe") else "capture"
        send_guess = int(target.ships) + 1
        candidates: List[ShotOption] = []

        for src in self.state.my_planets():
            if self.budget.expired():
                break
            src_cap = source_cap_fn(src.id)
            if src_cap < 1:
                continue
            reserve = int(max(0, src_cap * DEFENSE_MARGIN))
            src_cap_attack = max(0, src_cap - reserve)
            if src_cap_attack < 1:
                continue

            settled = settle_plan(
                src,
                target,
                src_cap_attack,
                send_guess,
                self.state,
                planned_commitments,
                modes,
                mission=mission,
            )
            if settled is None:
                continue

            angle, eta, _eval_turn, need, send = settled
            score = objective.priority - eta * 0.5 - send * 0.02 - max(0, send - need) * 0.01
            candidates.append(
                ShotOption(src.id, target.id, angle, send, eta, score),
            )

        candidates.sort(key=lambda s: s.score, reverse=True)
        return candidates[:3]


# =============================================================================
# Layer 5: scheduling
# =============================================================================


class Scheduler:
    def __init__(
        self,
        state: GameState,
        planner: StrategicPlanner,
        executor: TacticalExecutor,
        budget: TimeBudget,
    ):
        self.state = state
        self.planner = planner
        self.executor = executor
        self.budget = budget
        self.reserved_by_planet: Dict[int, int] = {}
        self.planned_commitments: PlannedCommitments = defaultdict(list)
        self.moves: List[Move] = []

    def source_inventory_left(self, source_id: int) -> int:
        planet = self.state.current_planet(source_id)
        if planet is None:
            return 0
        spent = self.reserved_by_planet.get(source_id, 0)
        return max(0, int(planet.ships) - spent)

    def commit_plan(self, plan: ShotOption) -> None:
        """Reserve ships and record in-flight commitment for timeline projection."""
        self.reserve(plan.src_id, plan.ships)
        self.planned_commitments[plan.target_id].append(
            (plan.eta, self.state.player, plan.ships),
        )

    def build_policy_state(self) -> dict:
        return {
            "budget_left": self.budget.remaining(),
            "tier": self.budget.tier.name,
            "my_planet_count": len(self.state.my_planets()),
        }

    def can_spend(self, planet_id: int, ships: int) -> bool:
        planet = self.state.current_planet(planet_id)
        if planet is None:
            return False
        spent = self.reserved_by_planet.get(planet_id, 0)
        return (planet.ships - spent) >= ships

    def reserve(self, planet_id: int, ships: int) -> None:
        self.reserved_by_planet[planet_id] = self.reserved_by_planet.get(planet_id, 0) + ships

    def resolve_conflicts(self, plans: List[ShotOption]) -> List[ShotOption]:
        chosen: List[ShotOption] = []
        for plan in sorted(plans, key=lambda p: p.score, reverse=True):
            if self.can_spend(plan.src_id, plan.ships):
                self.commit_plan(plan)
                chosen.append(plan)
        return chosen

    def generate_orders(self) -> List[Move]:
        """Plan objectives in priority order so later settle_plan sees commitments."""
        self.budget.refresh_tier()
        modes = self.planner.build_modes()

        for obj in self.planner.select_objectives():
            if self.budget.expired():
                break
            candidates = self.executor.plan_attack(
                obj,
                self.planned_commitments,
                modes,
                self.source_inventory_left,
            )
            for plan in sorted(candidates, key=lambda p: p.score, reverse=True):
                if not self.can_spend(plan.src_id, plan.ships):
                    continue
                self.commit_plan(plan)
                self.moves.append([plan.src_id, plan.angle, plan.ships])
                break

        return self.moves


# =============================================================================
# Fallback
# =============================================================================


def fallback_moves(state: GameState) -> List[Move]:
    moves: List[Move] = []
    my_planets = state.my_planets()
    targets = state.neutral_planets() + state.enemy_planets()
    if not my_planets or not targets:
        return moves

    src = max(my_planets, key=lambda p: p.ships)
    dst = min(targets, key=lambda p: dist((p.x, p.y), (src.x, src.y)))
    ships = max(1, src.ships // 2)
    moves.append([src.id, angle_between((src.x, src.y), (dst.x, dst.y)), ships])
    return moves


# =============================================================================
# Agent entrypoint
# =============================================================================


def agent(obs, config=None) -> List[Move]:
    budget = TimeBudget.from_config(config)

    try:
        state = GameState(obs)
    except Exception:
        return []

    if not state.my_planets():
        return []

    try:
        planner = StrategicPlanner(state, budget)
        executor = TacticalExecutor(state, budget)
        scheduler = Scheduler(state, planner, executor, budget)
        moves = scheduler.generate_orders()
        return moves if moves else fallback_moves(state)
    except Exception:
        return fallback_moves(state)


if __name__ == "__main__":
    # Lightweight sanity checks (no kaggle_environments required for resolve_arrival_event)
    owner, ships = resolve_arrival_event(0, 10.0, [(1, 1, 15)])
    assert owner == 1 and ships == 5.0, (owner, ships)
    tied_owner, tied_ships = resolve_arrival_event(0, 10.0, [(1, 1, 8), (1, 2, 8)])
    assert tied_owner == 0 and tied_ships == 10.0, (tied_owner, tied_ships)
    print("Orbit Wars main.py: resolve_arrival_event OK")
