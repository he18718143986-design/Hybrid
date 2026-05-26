#!/usr/bin/env python3
"""Batch probe: v2 vs baseline, hybrid vs baseline, hybrid vs v2."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

logging.disable(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search import probe_run as pr

AGENT_V2 = pr.AGENT_V2
AGENT_HYBRID = pr.AGENT_HYBRID
RESULTS_DIR = pr.RESULTS_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid vs v2 batch probe")
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument(
        "--match",
        choices=("all", "v2_vs_random", "hybrid_vs_random", "hybrid_vs_v2", "hybrid_vs_v2_swap"),
        default="all",
    )
    parser.add_argument("--opponent", default="random")
    parser.add_argument("--kaggle-timeout", action="store_true")
    parser.add_argument("--decision-trace", action="store_true")
    parser.add_argument("--regret", action="store_true")
    parser.add_argument("--regret-horizon", type=int, default=15)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "latest_probe.json")
    args = parser.parse_args()

    act_timeout = pr.KAGGLE_ACT_TIMEOUT if args.kaggle_timeout else pr.LOCAL_ACT_TIMEOUT
    seeds = (
        list(range(args.seed_start, args.seed_start + args.seeds))
        if args.seed_start is not None
        else pr.load_probe_seeds(args.seeds)
    )

    if args.opponent != "random" and not Path(args.opponent).is_file():
        print(f"opponent not found: {args.opponent}")
        return 2

    try:
        from kaggle_environments import make  # noqa: F401
    except ImportError:
        print("ERROR: pip install kaggle-environments")
        return 2

    summaries = []
    if args.match in ("all", "v2_vs_random"):
        summaries.append(pr.run_match("v2_vs_random", AGENT_V2, args.opponent, seeds, act_timeout))
    if args.match in ("all", "hybrid_vs_random"):
        summaries.append(pr.run_match("hybrid_vs_random", AGENT_HYBRID, args.opponent, seeds, act_timeout))
    if args.match in ("all", "hybrid_vs_v2"):
        summaries.append(pr.run_match("hybrid_vs_v2", AGENT_HYBRID, AGENT_V2, seeds, act_timeout))
    if args.match in ("all", "hybrid_vs_v2_swap"):
        summaries.append(pr.run_match("hybrid_vs_v2_swap", AGENT_V2, AGENT_HYBRID, seeds, act_timeout))

    aggs = [s.aggregate() for s in summaries]
    for agg in aggs:
        pr.print_summary(agg)

    payload = {
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "act_timeout": act_timeout,
        "seeds": seeds,
        "summaries": aggs,
        "games": {s.name: [asdict(g) for g in s.games] for s in summaries},
    }
    if args.decision_trace:
        payload["rollout_divergence"] = pr.rollout_divergence_probe(
            seeds[: min(10, len(seeds))], act_timeout,
        )
    if args.regret:
        payload["regret"] = pr.regret_probe(
            seeds[: min(30, len(seeds))], act_timeout, horizon=args.regret_horizon,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
