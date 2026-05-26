#!/usr/bin/env python3
"""
Orbit Wars regression suite — fixed seeds + layer unit checks.

Verifies (on production baseline submission_v2.py by default):
  L4  combat resolution (resolve_arrival_event)
  L2  sun collision geometry
  L1  rotating planet prediction, fleet_target_planet (A1)
  L2  plan_shot paths avoid sun (sampled from env)
  L1  build_arrival_ledger smoke on live obs
  L4  detect_enemy_planet_battles / B1 smoke (no crash)
  Games: fixed-seed 1v1 / 4P — no INVALID, completes

Usage:
  .venv/bin/python regression_test.py
  .venv/bin/python regression_test.py --baseline submission_v2.py
  .venv/bin/python regression_test.py --candidate submission_ablation_x.py
  .venv/bin/python regression_test.py --layers-only
  .venv/bin/python regression_test.py --games-only --kaggle-timeout

Exit code 0 = all checks passed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

logging.disable(logging.CRITICAL)

ROOT = Path(__file__).resolve().parent
SEEDS_PATH = ROOT / "regression" / "seeds.json"
RESULTS_DIR = ROOT / "regression" / "results"
LOCAL_ACT_TIMEOUT = 999.0


@dataclass
class CheckResult:
    name: str
    layer: str
    passed: bool
    detail: str = ""


@dataclass
class SuiteReport:
    module_path: Path
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, layer: str, passed: bool, detail: str = "") -> None:
        self.checks.append(CheckResult(name, layer, passed, detail))

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def load_seeds() -> dict:
    with open(SEEDS_PATH, encoding="utf-8") as f:
        return json.load(f)


def obs_from_env(seed: int, step: int, num_players: int = 2):
    from kaggle_environments import make

    agents = ["random"] * num_players
    env = make(
        "orbit_wars",
        debug=False,
        configuration={"seed": seed, "actTimeout": LOCAL_ACT_TIMEOUT},
    )
    env.reset(num_players)
    while env.state[0].observation.step < step and not env.done:
        env.step(["[]"] * num_players)
    return env.state[0].observation


# --- Layer tests ---


def test_combat_resolution(mod, report: SuiteReport) -> None:
    owner, ships = mod.resolve_arrival_event(
        -1,
        10.0,
        [
            (1, 0, 41),
            (1, 1, 20),
            (1, 1, 20),
            (1, 2, 42),
        ],
    )
    ok = owner == -1 and int(ships) == 9
    report.add(
        "combat: multi-attacker tie-break",
        "L4",
        ok,
        f"owner={owner} ships={ships} (expect -1, 9)",
    )


def test_sun_geometry(mod, report: SuiteReport) -> None:
    through = mod.segment_hits_sun(5.0, 50.0, 95.0, 50.0)
    tangent = mod.segment_hits_sun(5.0, 5.0, 95.0, 5.0)
    ok = through and not tangent
    report.add(
        "geometry: segment_hits_sun",
        "L2",
        ok,
        f"through_center={through} low_path={tangent}",
    )


def test_rotating_prediction(mod, report: SuiteReport) -> None:
    Planet = mod.Planet
    init = Planet(0, -1, 60.0, 50.0, 2.0, 5, 1)
    planet = Planet(0, -1, 60.0, 50.0, 2.0, 5, 1)
    initial_by_id = {0: init}
    ang = 0.05
    x0, y0 = mod.predict_planet_position(planet, initial_by_id, ang, 0)
    x1, y1 = mod.predict_planet_position(planet, initial_by_id, ang, 10)
    moved = (x0 - x1) ** 2 + (y0 - y1) ** 2 > 1e-6
    report.add(
        "world: predict_planet_position rotates",
        "L1",
        moved,
        f"delta=({x1-x0:.4f},{y1-y0:.4f})",
    )


def test_fleet_target_static(mod, report: SuiteReport) -> None:
    Planet = mod.Planet
    Fleet = mod.Fleet
    target = Planet(1, -1, 80.0, 50.0, 2.0, 10, 2)
    src_x, src_y = 20.0, 50.0
    angle = math.atan2(target.y - src_y, target.x - src_x)
    fleet = Fleet(0, 0, src_x, src_y, angle, 0, 20)
    hit, eta = mod.fleet_target_planet(
        fleet, [target], {1: target}, 0.0, [], set(),
    )
    ok = hit is not None and hit.id == 1 and eta is not None and eta >= 1
    report.add(
        "world: fleet_target_planet static (A1)",
        "L1",
        ok,
        f"hit={getattr(hit, 'id', None)} eta={eta}",
    )


def test_plan_shot_no_sun(mod, report: SuiteReport, seed: int) -> None:
    obs = obs_from_env(seed, step=30, num_players=2)
    world = mod.build_world(obs)
    static_bad = 0
    static_n = 0
    moving_n = 0
    for src in world.my_planets:
        for target in world.planets:
            if target.id == src.id:
                continue
            for ships in (1, max(1, int(src.ships // 2)), int(src.ships)):
                aim = world.plan_shot(src.id, target.id, ships)
                if aim is None:
                    continue
                angle, turns, tx, ty = aim
                if world.is_static(target.id):
                    static_n += 1
                    safe = mod.safe_angle_and_distance(
                        src.x, src.y, src.radius, tx, ty, target.radius,
                    )
                    if safe is None:
                        static_bad += 1
                else:
                    moving_n += 1
                if static_n + moving_n >= 80:
                    break
        if static_n + moving_n >= 80:
            break
    ok = static_n > 0 and static_bad == 0
    report.add(
        f"geometry: plan_shot safe on static targets (seed={seed})",
        "L2",
        ok,
        f"static={static_n} bad={static_bad} moving_smoke={moving_n}",
    )


def test_arrival_ledger_smoke(mod, report: SuiteReport, seed: int) -> None:
    obs = obs_from_env(seed, step=40, num_players=2)
    world = mod.build_world(obs)
    n_fleets = len(world.fleets)
    n_entries = sum(len(v) for v in world.arrivals_by_planet.values())
    ok = isinstance(world.arrivals_by_planet, dict)
    report.add(
        f"world: arrival ledger built (seed={seed})",
        "L1",
        ok,
        f"fleets={n_fleets} ledger_entries={n_entries}",
    )


def test_gang_up_and_b1_smoke(mod, report: SuiteReport, seed: int) -> None:
    obs = obs_from_env(seed, step=60, num_players=4)
    world = mod.build_world(obs)
    try:
        battles = mod.detect_enemy_planet_battles(world)
        ok_b = isinstance(battles, list)
        detail_b = f"battles={len(battles)}"
    except Exception as exc:
        ok_b = False
        detail_b = str(exc)
    report.add(f"combat: detect_enemy_planet_battles (A2 seed={seed})", "L4", ok_b, detail_b)

    try:
        if world.enemy_planets:
            tid = world.enemy_planets[0].id
            extras = world.hostile_reinforcement_arrivals(tid, 5)
            ok_r = isinstance(extras, tuple)
            detail_r = f"extras={len(extras)}"
        else:
            ok_r = True
            detail_r = "no enemy planets"
    except Exception as exc:
        ok_r = False
        detail_r = str(exc)
    report.add(f"combat: hostile_reinforcement_arrivals (B1 seed={seed})", "L4", ok_r, detail_r)


def run_layer_suite(mod, report: SuiteReport, seeds: dict) -> None:
    test_combat_resolution(mod, report)
    test_sun_geometry(mod, report)
    test_rotating_prediction(mod, report)
    test_fleet_target_static(mod, report)
    for seed in seeds.get("replay_seeds_for_layer_tests", [42]):
        test_plan_shot_no_sun(mod, report, seed)
        test_arrival_ledger_smoke(mod, report, seed)
        test_gang_up_and_b1_smoke(mod, report, seed)


# --- Fixed-seed games ---


def run_fixed_games(
    agent,
    agent_label: str,
    opponent,
    seeds: list[int],
    num_players: int,
    new_idx: int,
    act_timeout: float,
    report: SuiteReport,
) -> dict:
    from kaggle_environments import make

    invalid = 0
    done = 0
    rewards_sum = [0.0] * num_players
    times: list[float] = []

    for seed in seeds:
        agents = [opponent] * num_players
        agents[new_idx] = agent
        env = make(
            "orbit_wars",
            debug=False,
            configuration={"seed": seed, "actTimeout": act_timeout},
        )
        t0 = time.perf_counter()
        result = env.run(agents)
        dt = time.perf_counter() - t0
        final = result[-1]
        statuses = [s.status for s in final]
        rewards = [s.reward for s in final]
        if any(s == "INVALID" for s in statuses):
            invalid += 1
        else:
            done += 1
            for i, r in enumerate(rewards):
                rewards_sum[i] += r
        times.append(dt)

    label = f"games:{agent_label} {num_players}P×{len(seeds)}"
    ok = invalid == 0 and done == len(seeds)
    avg_agent_reward = rewards_sum[new_idx] / max(1, done)
    report.add(
        label,
        "L5",
        ok,
        f"invalid={invalid}/{len(seeds)} avg_reward[{new_idx}]={avg_agent_reward:.2f} "
        f"avg_time={sum(times)/len(times):.1f}s",
    )
    return {
        "seeds": seeds,
        "invalid": invalid,
        "completed": done,
        "avg_reward_agent": avg_agent_reward,
        "avg_time_s": sum(times) / len(times),
    }


def verify_baseline_lock(baseline_path: Path) -> CheckResult | None:
    lock_path = ROOT / "BASELINE.lock"
    if not lock_path.is_file():
        return CheckResult(
            "baseline: BASELINE.lock present",
            "—",
            False,
            "run: .venv/bin/python scripts/lock_baseline.py",
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    import hashlib

    sha = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    ok = lock.get("sha256") == sha and lock.get("production_file") == baseline_path.name
    return CheckResult(
        "baseline: SHA256 matches BASELINE.lock",
        "—",
        ok,
        f"lock={lock.get('sha256_short')} file={sha[:12]}",
    )


def print_report(report: SuiteReport) -> None:
    print(f"\n=== Regression: {report.module_path.name} ===")
    width = max(len(c.name) for c in report.checks) if report.checks else 10
    for c in report.checks:
        mark = "PASS" if c.passed else "FAIL"
        print(f"  [{mark}] [{c.layer:>2}] {c.name:<{width}}  {c.detail}")
    n_pass = sum(1 for c in report.checks if c.passed)
    print(f"\n  {n_pass}/{len(report.checks)} passed")


def main() -> int:
    parser = argparse.ArgumentParser(description="Orbit Wars fixed-seed regression suite")
    parser.add_argument(
        "--baseline",
        type=str,
        default=str(ROOT / "submission_v2.py"),
        help="Production baseline module (default: submission_v2.py)",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Optional candidate module; layer tests also run on this file",
    )
    parser.add_argument("--layers-only", action="store_true")
    parser.add_argument("--games-only", action="store_true")
    parser.add_argument(
        "--kaggle-timeout",
        action="store_true",
        help="Use actTimeout=1.0 for game smoke tests",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=str(ROOT / "submission_baseline.py"),
        help="Opponent for fixed-seed games",
    )
    args = parser.parse_args()

    try:
        from kaggle_environments import make  # noqa: F401
    except ImportError:
        print("ERROR: pip install kaggle-environments")
        return 2

    baseline_path = Path(args.baseline).resolve()
    if not baseline_path.is_file():
        print(f"baseline not found: {baseline_path}")
        return 2

    seeds = load_seeds()
    act_timeout = 1.0 if args.kaggle_timeout else LOCAL_ACT_TIMEOUT
    all_ok = True
    run_results: dict = {"generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    lock_check = verify_baseline_lock(baseline_path)
    if lock_check:
        print(f"[{'PASS' if lock_check.passed else 'FAIL'}] {lock_check.name} — {lock_check.detail}")
        all_ok &= lock_check.passed

    modules_to_test: list[tuple[Path, str]] = [(baseline_path, "baseline")]
    if args.candidate:
        cand = Path(args.candidate).resolve()
        if cand.is_file():
            modules_to_test.append((cand, "candidate"))

    opponent_path = Path(args.opponent).resolve()
    if opponent_path.is_file():
        opponent_agent = load_module(opponent_path, "opp").agent
    else:
        opponent_agent = "random"

    for path, label in modules_to_test:
        mod = load_module(path, f"reg_{label}")
        report = SuiteReport(path)

        if not args.games_only:
            run_layer_suite(mod, report, seeds)

        if not args.layers_only:
            agent = mod.agent
            opp = opponent_agent
            game_stats = {}
            s1 = seeds.get("seeds_1v1_vs_baseline", [1100])
            game_stats["1v1"] = run_fixed_games(
                agent, label, opp, s1, 2, 0, act_timeout, report,
            )
            s4 = seeds.get("seeds_4p_vs_baseline", [2000])
            game_stats["4p"] = run_fixed_games(
                agent, label, opp, s4, 4, 0, act_timeout, report,
            )
            run_results[label] = game_stats

        print_report(report)
        all_ok &= report.ok

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "latest.json"
    out.write_text(json.dumps(run_results, indent=2) + "\n", encoding="utf-8")
    print(f"\nResults written: {out}")

    if all_ok:
        print("\nREGRESSION: ALL PASSED")
        return 0
    print("\nREGRESSION: FAILURES (see above)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
