#!/usr/bin/env python3
"""Run a local orbit_wars game."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=("submission", "v2", "hybrid"),
        default="submission",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--opponent", default="random")
    args = parser.parse_args()

    if args.agent == "v2":
        agent_path = str(ROOT / "submission_v2.py")
        os.environ["ORBIT_AGENT_MODE"] = "v2"
    elif args.agent == "hybrid":
        agent_path = str(ROOT / "submission" / "main.py")
        os.environ["ORBIT_AGENT_MODE"] = "hybrid"
    else:
        agent_path = str(ROOT / "submission" / "main.py")
        os.environ.setdefault("ORBIT_AGENT_MODE", "v2")

    from kaggle_environments import make

    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run([agent_path, args.opponent])
    final = env.steps[-1]
    for i, s in enumerate(final):
        print(f"player {i}: reward={s.reward} status={s.status}")


if __name__ == "__main__":
    main()
