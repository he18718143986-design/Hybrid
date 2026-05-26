#!/usr/bin/env python3
"""Classify Orbit Wars Kaggle replay loss patterns for the Tina / v2 agent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def ship_totals(obs: dict, player: int) -> tuple[int, int, int, int]:
    """Return (planet_ships, fleet_ships, owned_planets, neutral_owned)."""
    ps = fs = owned = neutrals = 0
    for p in obs.get("planets", []):
        owner, ships = p[1], p[5]
        if owner == player:
            ps += ships
            owned += 1
        elif owner == -1:
            neutrals += 1
    for f in obs.get("fleets", []):
        if f[1] == player:
            fs += f[6]
    return ps, fs, owned, neutrals


def production_sum(obs: dict, player: int) -> float:
    return sum(p[6] for p in obs.get("planets", []) if p[1] == player)


def comet_owned_by(obs: dict, player: int) -> int:
    ids = set(obs.get("comet_planet_ids") or [])
    return sum(1 for p in obs.get("planets", []) if p[0] in ids and p[1] == player)


def analyze(path: Path, my_name: str = "Tina") -> dict:
    data = json.loads(path.read_text())
    names = data["info"].get("TeamNames") or []
    if my_name not in names:
        raise ValueError(f"{my_name} not in TeamNames {names}")
    me = names.index(my_name)
    rewards = data["rewards"]
    statuses = data["statuses"]
    steps = data["steps"]
    n_players = len(names)

    # Winner by reward
    win_idx = max(range(len(rewards)), key=lambda i: rewards[i])
    winner = names[win_idx]

    checkpoints = [0, 10, 20, 30, 50, 100, 200, min(300, len(steps) - 1), len(steps) - 1]

    timeline = []
    for st in checkpoints:
        if st >= len(steps):
            continue
        agents = steps[st]
        my_obs = agents[me]["observation"]
        win_obs = agents[win_idx]["observation"]
        mp, mf, mo, _ = ship_totals(my_obs, me)
        wp, wf, wo, _ = ship_totals(win_obs, win_idx)
        my_total = mp + mf
        win_total = wp + wf
        timeline.append(
            {
                "step": st,
                "my_ships": my_total,
                "win_ships": win_total,
                "ratio": round(my_total / win_total, 3) if win_total else None,
                "my_planets": mo,
                "win_planets": wo,
                "my_prod": production_sum(my_obs, me),
                "comet_mine": comet_owned_by(my_obs, me),
                "comet_win": comet_owned_by(win_obs, win_idx),
                "comets_active": len(my_obs.get("comets") or []),
            }
        )

    t20 = next((t for t in timeline if t["step"] == 20), timeline[1] if len(timeline) > 1 else timeline[0])
    t30 = next((t for t in timeline if t["step"] == 30), t20)
    t100 = next((t for t in timeline if t["step"] == 100), timeline[-1])
    final = timeline[-1]

    tags = []
    if n_players >= 4:
        tags.append("4P")
    if any(s == "INVALID" for s in statuses):
        tags.append("超时/INVALID")
    if len(steps) < 500:
        tags.append("提前终局")
    if (t30.get("ratio") or 1) < 0.75:
        tags.append("开局落后")
    elif (t20.get("ratio") or 1) < 0.85 and (t100.get("ratio") or 1) < 0.8:
        tags.append("开局落后")
    if (t30.get("ratio") or 0) >= 0.85 and (final.get("ratio") or 1) < 0.6:
        tags.append("中后期崩盘")
    if max(t.get("comets_active", 0) for t in timeline) > 0:
        if final.get("comet_win", 0) > final.get("comet_mine", 0):
            tags.append("彗星劣势")
        else:
            tags.append("有彗星局")

    overage_min = None
    for agents in steps[::25]:
        obs = agents[me]["observation"]
        if "remainingOverageTime" in obs:
            v = obs["remainingOverageTime"]
            overage_min = v if overage_min is None else min(overage_min, v)
    if overage_min is not None and overage_min < 5:
        tags.append("时间预算紧")

    return {
        "episode_id": data["info"].get("EpisodeId"),
        "file": path.name,
        "players": n_players,
        "my_index": me,
        "winner": winner,
        "my_reward": rewards[me],
        "steps_played": len(steps),
        "statuses": statuses,
        "tags": tags or ["未分类"],
        "timeline": timeline,
        "seed": data["info"].get("seed"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("replays", nargs="+", type=Path)
    ap.add_argument("--name", default="Tina")
    args = ap.parse_args()
    for p in args.replays:
        r = analyze(p, args.name)
        print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
