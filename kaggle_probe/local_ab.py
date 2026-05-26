"""
Local A/B test: submission.py (new) vs submission_baseline.py (old).

Usage:
    python local_ab.py                        # default: 20 1v1 + 10 4P
    python local_ab.py --pair 50 --ffa 20     # custom counts
    python local_ab.py --quick                # 10 1v1 + 4 4P sanity check
    python local_ab.py --pair-only 40         # skip FFA
    python local_ab.py --warmup 1             # discard first game (cold-start JIT)
    python local_ab.py --kaggle-timeout       # simulate Kaggle 1s/turn budget

Requires `kaggle-environments` (pip install kaggle-environments).
Use the project venv: `.venv/bin/python local_ab.py ...`
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import logging
import math
import os
import statistics
import sys
import time
from pathlib import Path

# Force unbuffered output so progress is visible under pipes/tee.
print = functools.partial(print, flush=True)  # noqa: A001

# Silence kaggle_environments init noise.
logging.disable(logging.CRITICAL)


ROOT = Path(__file__).resolve().parent

# Kaggle default actTimeout=1.0; agents derive a soft wall-clock budget from it.
# Local runs are slower than Kaggle, so use a large timeout for stable A/B.
LOCAL_ACT_TIMEOUT = 999.0


def load_agent(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.agent


def ensure_env():
    try:
        from kaggle_environments import make  # noqa: F401
    except ImportError:
        print("ERROR: kaggle-environments is not installed.")
        print("Run: pip install kaggle-environments")
        sys.exit(2)


def run_game(agents, seed, act_timeout=1.0):
    from kaggle_environments import make
    env = make(
        "orbit_wars",
        debug=False,
        configuration={"seed": seed, "actTimeout": act_timeout},
    )
    t0 = time.perf_counter()
    result = env.run(agents)
    dt = time.perf_counter() - t0
    rewards = [step.reward for step in result[-1]]
    statuses = [step.status for step in result[-1]]
    return rewards, statuses, dt


def winner_index(rewards):
    """Return index of the player with the highest reward, or None on tie."""
    best = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == best]
    if len(winners) == 1:
        return winners[0]
    return None


def summarize_pair(label, new_idx_history, rewards_history, time_history, status_history):
    wins = sum(1 for r, idx in zip(rewards_history, new_idx_history)
               if winner_index(r) == idx)
    losses = sum(1 for r, idx in zip(rewards_history, new_idx_history)
                 if winner_index(r) is not None and winner_index(r) != idx)
    ties = len(rewards_history) - wins - losses
    n = len(rewards_history)
    win_rate = wins / n if n else 0.0
    times_avg = statistics.mean(time_history) if time_history else 0.0
    timeouts = sum(1 for s in status_history if any(x not in ("DONE", "INVALID") for x in s))
    invalids = sum(1 for s in status_history for x in s if x == "INVALID")

    elo_diff = None
    if 0 < win_rate < 1:
        elo_diff = -400 * math.log10((1 - win_rate) / win_rate)

    print(f"\n=== {label} ===")
    print(f"  games:       {n}")
    print(f"  wins:        {wins}")
    print(f"  losses:      {losses}")
    print(f"  ties:        {ties}")
    print(f"  win rate:    {win_rate*100:.1f}%")
    if elo_diff is not None:
        sign = "+" if elo_diff > 0 else ""
        print(f"  elo delta:   {sign}{elo_diff:.0f} (estimated)")
    print(f"  avg time:    {times_avg:.1f}s per game")
    print(f"  invalid:     {invalids}")
    return win_rate, elo_diff, wins, losses, ties


def run_pair(new_agent, old_agent, num_games, label_prefix="", act_timeout=1.0):
    print(f"\n--- {label_prefix}1v1: {num_games} games ---")
    new_idx_history, rewards_history, time_history, status_history = [], [], [], []
    for i in range(num_games):
        if i % 2 == 0:
            agents = [new_agent, old_agent]
            new_idx = 0
        else:
            agents = [old_agent, new_agent]
            new_idx = 1
        seed = 1000 + i
        rewards, statuses, dt = run_game(agents, seed=seed, act_timeout=act_timeout)
        winner = winner_index(rewards)
        winner_label = "NEW" if winner == new_idx else ("OLD" if winner is not None else "TIE")
        print(f"  [{i+1:3d}/{num_games}] seed={seed} new_idx={new_idx} "
              f"rewards={rewards} winner={winner_label} time={dt:.1f}s")
        new_idx_history.append(new_idx)
        rewards_history.append(rewards)
        time_history.append(dt)
        status_history.append(statuses)
    return summarize_pair(f"{label_prefix}1v1 RESULTS",
                          new_idx_history, rewards_history, time_history, status_history)


def run_ffa(new_agent, old_agent, num_games, act_timeout=1.0):
    print(f"\n--- 4P FFA: {num_games} games (NEW vs 3x OLD) ---")
    new_idx_history, rewards_history, time_history, status_history = [], [], [], []
    for i in range(num_games):
        new_idx = i % 4
        agents = [old_agent] * 4
        agents[new_idx] = new_agent
        seed = 2000 + i
        rewards, statuses, dt = run_game(agents, seed=seed, act_timeout=act_timeout)
        winner = winner_index(rewards)
        winner_label = (
            "NEW" if winner == new_idx
            else (f"OLD#{winner}" if winner is not None else "TIE")
        )
        print(f"  [{i+1:3d}/{num_games}] seed={seed} new_idx={new_idx} "
              f"rewards={rewards} winner={winner_label} time={dt:.1f}s")
        new_idx_history.append(new_idx)
        rewards_history.append(rewards)
        time_history.append(dt)
        status_history.append(statuses)
    return summarize_pair("4P FFA RESULTS",
                          new_idx_history, rewards_history, time_history, status_history)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=int, default=20, help="1v1 game count")
    parser.add_argument("--ffa", type=int, default=10, help="4P FFA game count")
    parser.add_argument("--quick", action="store_true", help="Override to 10 pair + 4 FFA")
    parser.add_argument("--pair-only", type=int, default=None,
                        help="Run only 1v1 with this many games")
    parser.add_argument("--new", type=str, default=str(ROOT / "submission.py"))
    parser.add_argument("--old", type=str, default=str(ROOT / "submission_baseline.py"))
    parser.add_argument(
        "--kaggle-timeout",
        action="store_true",
        help="Use actTimeout=1.0 like Kaggle (default: generous local timeout for stable A/B)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Run N warmup games (discarded) to absorb cold-start JIT before scoring",
    )
    args = parser.parse_args()

    ensure_env()

    pair_n = args.pair
    ffa_n = args.ffa
    if args.quick:
        pair_n, ffa_n = 10, 4
    if args.pair_only is not None:
        pair_n, ffa_n = args.pair_only, 0

    new_path = Path(args.new).resolve()
    old_path = Path(args.old).resolve()
    assert new_path.exists(), f"New agent not found: {new_path}"
    assert old_path.exists(), f"Baseline not found: {old_path}"

    act_timeout = 1.0 if args.kaggle_timeout else LOCAL_ACT_TIMEOUT
    timeout_label = "Kaggle (1.0s)" if args.kaggle_timeout else f"local ({act_timeout}s)"

    print(f"NEW agent: {new_path}")
    print(f"OLD agent: {old_path}")
    print(f"Plan: {pair_n} 1v1 games + {ffa_n} 4P games")
    print(f"actTimeout: {timeout_label}")

    new_agent = load_agent(new_path, "sub_new")
    old_agent = load_agent(old_path, "sub_old")

    if args.warmup > 0:
        print(f"\n--- Warmup: {args.warmup} game(s), results discarded ---")
        run_pair(new_agent, old_agent, args.warmup, label_prefix="warmup ", act_timeout=act_timeout)

    summaries = []
    if pair_n > 0:
        summaries.append(("1v1", run_pair(new_agent, old_agent, pair_n, act_timeout=act_timeout)))
    if ffa_n > 0:
        summaries.append(("4P", run_ffa(new_agent, old_agent, ffa_n, act_timeout=act_timeout)))

    print("\n" + "=" * 56)
    print("FINAL VERDICT")
    print("=" * 56)
    for label, (win_rate, elo_diff, wins, losses, ties) in summaries:
        if wins == 0 and losses == 0 and ties > 0:
            verdict = "EVEN (all ties)"
        elif win_rate > 0.55:
            verdict = "IMPROVED"
        elif win_rate < 0.45 and losses > 0:
            verdict = "REGRESSED"
        else:
            verdict = "INCONCLUSIVE"
        elo_str = f"  (~{elo_diff:+.0f} elo)" if elo_diff is not None else ""
        print(f"  {label:>4s}: {win_rate*100:.1f}% win rate  "
              f"({wins}W/{losses}L/{ties}T)  -> {verdict}{elo_str}")
    print("\nThresholds: >55% win = IMPROVED, <45% win with losses = REGRESSED.")
    print("All ties = evenly matched in this matchup (not a regression).")


if __name__ == "__main__":
    main()
