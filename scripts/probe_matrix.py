#!/usr/bin/env python3
"""Frozen probe matrix v1 — comparable hybrid experiments across rollout opponents."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

logging.disable(logging.CRITICAL)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search import probe_run as pr
from src.search.decision_delta import collect_decision_deltas

MATRIX_VERSION = "v1"
RESULTS_DIR = pr.RESULTS_DIR

MATRIX_CELLS: List[Dict[str, Any]] = [
    {"name": "v2_vs_random", "p0": pr.AGENT_V2, "p1": "random", "rollout_opponent": None},
    {
        "name": "hybrid_passive",
        "p0": pr.AGENT_HYBRID,
        "p1": pr.AGENT_V2,
        "rollout_opponent": "passive",
    },
    {
        "name": "hybrid_frontier",
        "p0": pr.AGENT_HYBRID,
        "p1": pr.AGENT_V2,
        "rollout_opponent": "frontier",
    },
]


def run_cell(
    cell: Dict[str, Any],
    seeds: Sequence[int],
    act_timeout: float,
    regret_horizon: int,
    decision_steps: Sequence[int],
    collect_delta: bool,
) -> Dict[str, Any]:
    name = cell["name"]
    rollout_opp = cell.get("rollout_opponent")

    print(f"\n{'='*60}\nCell: {name}  seeds={len(seeds)}  rollout_opponent={rollout_opp}\n{'='*60}")

    if rollout_opp:
        os.environ["ORBIT_ROLLOUT_OPPONENT"] = rollout_opp

    summary = pr.run_match(name, cell["p0"], cell["p1"], seeds, act_timeout)
    agg = summary.aggregate()
    pr.print_summary(agg)

    regret = None
    divergence: dict = {}
    delta_count = 0
    if rollout_opp in ("passive", "frontier"):
        regret = pr.regret_probe(
            seeds, act_timeout, decision_steps=decision_steps, horizon=regret_horizon,
        )
        divergence = pr.rollout_divergence_probe(
            seeds[: min(10, len(seeds))], act_timeout, steps_to_sample=decision_steps,
        )

        print(f"\n--- regret ({name}, horizon={regret_horizon}) ---")
        if regret.get("n"):
            traj = regret.get("replay_trajectory", "v2")
            print(f"  replay_trajectory={traj}")
            print(
                f"  n={regret['n']}  avg_ship_diff={regret['avg_ship_diff']:.2f}  "
                f"hybrid_worse_rate={regret['hybrid_worse_ships_rate']:.1%}"
            )
            print(
                f"  same_decision_rate={regret.get('same_decision_rate', 0):.1%}  "
                f"decision_reversal_rate={regret.get('decision_reversal_rate', 0):.1%}  "
                f"reversal|diverged={regret.get('decision_reversal_given_diverged', 0):.1%}"
            )
            dash = regret.get("override_precision_dashboard", {})
            if dash.get("n"):
                print(
                    f"  override_precision={dash.get('override_precision', 0):.1%}  "
                    f"override_recall={dash.get('override_recall', 0):.1%}  "
                    f"reversal_severity={dash.get('reversal_severity_mean_ships', 0):.1f} ships"
                )
                cal = dash.get("confidence_calibration")
                if cal:
                    print(
                        f"  margin↔outcome corr={cal.get('margin_outcome_correlation', 0):.2f}"
                    )
            div = divergence.get("divergence", {})
            print(
                f"  divergence_sample same_decision={div.get('same_decision_rate', 0):.1%} "
                f"(n={divergence.get('sample_count', 0)})"
            )
            print("  by_bucket (read order: neutral_capture, aggressive_expansion, comet_chase):")
            for bucket, stats in sorted(regret.get("by_bucket", {}).items()):
                print(
                    f"    {bucket}: n={stats['n']}  worse={stats['hybrid_worse_ships_rate']:.0%}  "
                    f"avg_ship_diff={stats['avg_ship_diff']:.1f}  "
                    f"reversal={stats.get('decision_reversal_rate', 0):.0%}  "
                    f"reversal|div={stats.get('decision_reversal_given_diverged', 0):.0%}"
                )
        else:
            divergence = {"divergence": {}, "sample_count": 0}

        if collect_delta:
            use_cf = name == "hybrid_frontier" and len(seeds) > 5
            deltas = collect_decision_deltas(
                seeds,
                act_timeout,
                rollout_opp,
                decision_steps=decision_steps,
                regret_horizon=regret_horizon,
                counterfactual=use_cf,
            )
            delta_count = len(deltas)
            diverged = sum(1 for r in deltas if r.get("differ"))
            delta_path = RESULTS_DIR / f"decision_delta_{name}_{MATRIX_VERSION}.jsonl"
            with delta_path.open("w", encoding="utf-8") as f:
                for row in deltas:
                    f.write(json.dumps(row, separators=(",", ":")) + "\n")
            print(f"  decision_delta: {delta_count} records ({diverged} diverged) -> {delta_path}")

        report_path = RESULTS_DIR / f"regret_report_{name}_{MATRIX_VERSION}.json"
        report_path.write_text(
            json.dumps(
                {
                    "matrix_version": MATRIX_VERSION,
                    "cell": name,
                    "rollout_opponent": rollout_opp,
                    "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "seeds": list(seeds),
                    "act_timeout": act_timeout,
                    "regret_horizon": regret_horizon,
                    "decision_steps": list(decision_steps),
                    "head_to_head": agg,
                    "regret": {
                        k: v
                        for k, v in regret.items()
                        if k != "rows"
                    },
                    "rollout_divergence": divergence,
                    "decision_delta_count": delta_count,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"  wrote {report_path}")

    return {
        "cell": name,
        "rollout_opponent": rollout_opp,
        "head_to_head": agg,
        "regret": regret,
        "decision_delta_count": delta_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Orbit Hybrid probe matrix v1")
    parser.add_argument("--quick", action="store_true", help="First 5 seeds only")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--cell", choices=[c["name"] for c in MATRIX_CELLS] + ["all"], default="all")
    parser.add_argument("--kaggle-timeout", action="store_true")
    parser.add_argument("--regret-horizon", type=int, default=15)
    parser.add_argument("--decision-steps", default="0,10,25,50,100")
    parser.add_argument("--no-delta", action="store_true")
    args = parser.parse_args()

    n = 5 if args.quick else (args.seeds or 30)
    seeds = pr.load_seeds_probe_v1(n)
    act_timeout = pr.KAGGLE_ACT_TIMEOUT if args.kaggle_timeout else pr.LOCAL_ACT_TIMEOUT
    if args.quick:
        decision_steps = (10, 25)
        regret_horizon = min(args.regret_horizon, 8)
    else:
        decision_steps = tuple(int(x) for x in args.decision_steps.split(",") if x.strip())
        if args.decision_steps == "0,10,25,50,100":
            decision_steps = (0, 10, 25, 50, 75, 100, 150)
        regret_horizon = args.regret_horizon

    try:
        from kaggle_environments import make  # noqa: F401
    except ImportError:
        print("ERROR: pip install kaggle-environments")
        return 2

    cells = MATRIX_CELLS if args.cell == "all" else [c for c in MATRIX_CELLS if c["name"] == args.cell]
    print(f"Probe matrix {MATRIX_VERSION}: {len(seeds)} seeds, actTimeout={act_timeout}")

    results = [
        run_cell(
            cell, seeds, act_timeout, regret_horizon, decision_steps, not args.no_delta,
        )
        for cell in cells
    ]

    matrix_path = RESULTS_DIR / f"matrix_report_{MATRIX_VERSION}.json"
    matrix_path.write_text(
        json.dumps(
            {
                "matrix_version": MATRIX_VERSION,
                "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "seeds": seeds,
                "cells": results,
                "success_criteria": {
                    "hybrid_worse_ships_rate_max": 0.55,
                    "avg_ship_diff_target": 0.0,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nMatrix report: {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
