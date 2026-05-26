"""
Self-play parameter sweep for Orbit Wars.

Lets you A/B-test parameter changes WITHOUT manually editing submission.py.
For each parameter value in your sweep, it:
  1. Loads submission_972.py (the frozen baseline) as a fresh module
  2. Monkey-patches the chosen parameter to the new value
  3. Plays N games of variant vs frozen baseline
  4. Reports win rate

Usage examples:
    # Sweep one parameter with a few values
    python sweep_params.py \
        --param HOSTILE_REINFORCE_FRACTION \
        --values 0.15 0.20 0.25 0.30 \
        --games 30

    # Sweep with 4P games too
    python sweep_params.py \
        --param EARLY_TURN_LIMIT \
        --values 35 40 45 50 \
        --games 30 --ffa 10

    # Quick sanity check (10 games per value)
    python sweep_params.py \
        --param ROUTE_SEARCH_HORIZON \
        --values 60 80 100 \
        --games 10

If you want to sweep a parameter that doesn't exist as a module-level constant
in submission.py, this script will refuse — that's intentional, it forces you
to add the constant explicitly first.

The baseline this compares against is always `submission_972.py`. Variants need
to BEAT 55% to be considered an improvement (sample-size aware).

NOTE: Each game takes 30-90 seconds. 30 games × 4 values = 60-180 minutes.
Plan accordingly. Use `--games 10` for quick exploration.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "submission_972.py"


def _ensure_env():
    try:
        from kaggle_environments import make  # noqa: F401
    except ImportError:
        print("ERROR: kaggle-environments is not installed.")
        print("Run: pip install 'kaggle-environments>=1.28.0'")
        sys.exit(2)


def _fresh_load(path: Path, name: str):
    """Load a module fresh (no cache). Returns the module object."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_variant_agent(param: str, value: Any, name: str):
    """Load a fresh copy of the baseline and patch one constant."""
    module = _fresh_load(BASELINE_PATH, name)
    if not hasattr(module, param):
        raise AttributeError(
            f"submission_972.py has no module-level constant named {param!r}. "
            f"Edit submission_972.py to expose it (or pick a different param).",
        )
    original = getattr(module, param)
    setattr(module, param, value)
    print(f"  [{name}] patched {param}: {original!r} -> {value!r}")
    return module.agent


def _winner(rewards):
    best = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == best]
    return winners[0] if len(winners) == 1 else None


def _play(agents, seed):
    from kaggle_environments import make
    env = make("orbit_wars", debug=False, configuration={"seed": seed})
    t0 = time.perf_counter()
    result = env.run(agents)
    dt = time.perf_counter() - t0
    rewards = [step.reward for step in result[-1]]
    statuses = [step.status for step in result[-1]]
    return rewards, statuses, dt


def _run_pair(variant_agent, baseline_agent, num_games, label):
    print(f"\n--- 1v1 ({label}): {num_games} games ---")
    wins = losses = ties = 0
    times = []
    invalids = 0
    for i in range(num_games):
        if i % 2 == 0:
            agents = [variant_agent, baseline_agent]
            variant_idx = 0
        else:
            agents = [baseline_agent, variant_agent]
            variant_idx = 1
        seed = 1000 + i
        rewards, statuses, dt = _play(agents, seed)
        invalids += sum(1 for s in statuses if s == "INVALID")
        winner = _winner(rewards)
        if winner == variant_idx:
            wins += 1
            tag = "WIN"
        elif winner is None:
            ties += 1
            tag = "TIE"
        else:
            losses += 1
            tag = "LOSS"
        times.append(dt)
        print(f"  [{i+1:3d}/{num_games}] seed={seed} {tag:4s} rewards={rewards} t={dt:.1f}s")
    avg_time = statistics.mean(times) if times else 0.0
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "n": num_games, "win_rate": wins / num_games if num_games else 0,
        "avg_time": avg_time, "invalids": invalids,
    }


def _run_ffa(variant_agent, baseline_agent, num_games, label):
    print(f"\n--- 4P FFA ({label}): {num_games} games (variant vs 3x baseline) ---")
    wins = losses = ties = 0
    invalids = 0
    times = []
    for i in range(num_games):
        variant_idx = i % 4
        agents = [baseline_agent] * 4
        agents[variant_idx] = variant_agent
        seed = 2000 + i
        rewards, statuses, dt = _play(agents, seed)
        invalids += sum(1 for s in statuses if s == "INVALID")
        winner = _winner(rewards)
        if winner == variant_idx:
            wins += 1
            tag = "WIN"
        elif winner is None:
            ties += 1
            tag = "TIE"
        else:
            losses += 1
            tag = "LOSS"
        times.append(dt)
        print(f"  [{i+1:3d}/{num_games}] seed={seed} {tag:4s} rewards={rewards} t={dt:.1f}s")
    avg_time = statistics.mean(times) if times else 0.0
    return {
        "wins": wins, "losses": losses, "ties": ties,
        "n": num_games, "win_rate": wins / num_games if num_games else 0,
        "avg_time": avg_time, "invalids": invalids,
    }


def _verdict(win_rate, n):
    """Sample-size aware verdict."""
    # Wilson 95% CI lower bound; if even the lower bound > 0.5 we're confident
    if n == 0:
        return "NO DATA"
    z = 1.96
    p = win_rate
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denominator
    lo = max(0.0, centre - margin)
    hi = min(1.0, centre + margin)
    if lo > 0.55:
        return f"IMPROVED (CI [{lo*100:.0f}%, {hi*100:.0f}%])"
    if hi < 0.45:
        return f"REGRESSED (CI [{lo*100:.0f}%, {hi*100:.0f}%])"
    if win_rate > 0.55:
        return f"likely improved (CI [{lo*100:.0f}%, {hi*100:.0f}%]) — need more games"
    if win_rate < 0.45:
        return f"likely regressed (CI [{lo*100:.0f}%, {hi*100:.0f}%]) — need more games"
    return f"INCONCLUSIVE (CI [{lo*100:.0f}%, {hi*100:.0f}%])"


def _coerce(value: str):
    """Try to coerce a CLI string into the right Python type."""
    lower = value.lower()
    if lower in ("true", "false"):
        return lower == "true"
    try:
        if "." in value or "e" in lower:
            return float(value)
        return int(value)
    except ValueError:
        return value


def main():
    parser = argparse.ArgumentParser(
        description="Sweep a single Orbit Wars parameter and report win rates.",
    )
    parser.add_argument("--param", required=True, help="Constant name in submission_972.py")
    parser.add_argument("--values", nargs="+", required=True, help="Values to test")
    parser.add_argument("--games", type=int, default=20, help="1v1 games per value")
    parser.add_argument("--ffa", type=int, default=0, help="4P FFA games per value")
    args = parser.parse_args()

    _ensure_env()

    if not BASELINE_PATH.exists():
        print(f"ERROR: baseline {BASELINE_PATH} not found.")
        sys.exit(2)

    baseline_agent = _fresh_load(BASELINE_PATH, "baseline_972").agent

    values = [_coerce(v) for v in args.values]
    print(f"\nParameter sweep: {args.param}")
    print(f"Values:    {values}")
    print(f"Baseline:  {BASELINE_PATH.name} (the locked 972-rated agent)")
    print(f"Plan:      {args.games} 1v1 + {args.ffa} 4P FFA per value")
    if args.games + args.ffa > 0:
        est_min = (args.games + args.ffa) * len(values) * 60 / 60
        print(f"Estimated: ~{est_min:.0f} minutes total\n")

    results = []
    for value in values:
        label = f"{args.param}={value}"
        print(f"\n{'=' * 60}")
        print(f"Testing {label}")
        print("=" * 60)
        try:
            variant_agent = _make_variant_agent(args.param, value, f"variant_{value}")
        except AttributeError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)

        pair = _run_pair(variant_agent, baseline_agent, args.games, label) if args.games else None
        ffa = _run_ffa(variant_agent, baseline_agent, args.ffa, label) if args.ffa else None
        results.append((value, pair, ffa))

    print("\n" + "=" * 70)
    print("FINAL SWEEP RESULTS")
    print("=" * 70)
    print(f"Parameter: {args.param}")
    print(f"{'Value':<15s} {'1v1 win%':<10s} {'1v1 verdict':<40s} {'4P win%':<10s} {'invalids':<10s}")
    print("-" * 95)
    for value, pair, ffa in results:
        pair_str = f"{pair['win_rate']*100:.1f}%" if pair else "N/A"
        pair_verdict = _verdict(pair["win_rate"], pair["n"]) if pair else "—"
        ffa_str = f"{ffa['win_rate']*100:.1f}%" if ffa else "N/A"
        invalids = (pair["invalids"] if pair else 0) + (ffa["invalids"] if ffa else 0)
        invalids_str = f"{invalids}" if invalids == 0 else f"{invalids} ⚠️"
        print(f"{str(value):<15s} {pair_str:<10s} {pair_verdict:<40s} {ffa_str:<10s} {invalids_str:<10s}")

    print(
        "\nDecision rule: only adopt a value if its 1v1 win rate is IMPROVED "
        "(lower CI bound > 55%) AND has zero INVALIDs."
    )


if __name__ == "__main__":
    main()
