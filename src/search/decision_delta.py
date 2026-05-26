"""Decision delta replay — v2 vs hybrid openings with real-env regret (gold debug data)."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence

from src.env.parser import parse_observation
from src.search.probe_metrics import choices_at_obs, measure_regret

Move = List[Any]


def state_fingerprint(obs: Any) -> str:
    """Stable id for comparable world states (seed+step+planet hash)."""
    parsed = parse_observation(obs)
    rows = []
    for p in sorted(parsed["planets"], key=lambda x: x.id):
        rows.append((p.id, p.owner, int(p.ships), round(p.x, 2), round(p.y, 2)))
    blob = json.dumps(
        {
            "step": parsed["step"],
            "player": parsed["player"],
            "planets": rows,
            "fleets": len(parsed["fleets"]),
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def build_delta_record(
    seed: int,
    step: int,
    obs: Any,
    config: Optional[dict],
    regret_horizon: int,
    act_timeout: float,
    follow_agent,
    rollout_opponent: str,
    *,
    counterfactual: bool = False,
) -> Optional[Dict[str, Any]]:
    """One decision point: v2 vs hybrid move + N-turn real-env regret."""
    ch = choices_at_obs(obs, config, counterfactual=counterfactual)
    v2_moves = ch.get("v2_moves") or []
    hy_moves = ch.get("hybrid_moves") or []

    differ = ch.get("differ", False)
    vc, hc = ch.get("v2_candidate"), ch.get("hybrid_candidate")
    margin = ch.get("rollout_margin")
    if margin is None and hc:
        v2_rollout = float(vc.get("rollout_score", 0)) if vc else 0.0
        margin = float(hc.get("rollout_score", 0)) - v2_rollout
    reg = measure_regret(
        seed,
        step,
        v2_moves,
        hy_moves,
        regret_horizon,
        act_timeout,
        follow_agent,
        action_bucket=ch.get("hybrid_action_bucket", "unknown"),
        differ=differ,
        rollout_margin=margin,
    )
    ship_diff = reg["ship_diff_hybrid_minus_v2"]
    decision_reversal = reg.get("decision_reversal", differ and ship_diff < 0)

    winner = "tie"
    if ship_diff > 0:
        winner = "hybrid"
    elif ship_diff < 0:
        winner = "v2"

    return {
        "state_id": f"{seed}:{step}:{state_fingerprint(obs)}",
        "seed": seed,
        "step": step,
        "rollout_opponent": rollout_opponent,
        "v2_move": v2_moves[0] if v2_moves else None,
        "hybrid_move": hy_moves[0] if hy_moves else None,
        "v2_moves_full": v2_moves,
        "hybrid_moves_full": hy_moves,
        "bucket": ch.get("hybrid_action_bucket"),
        "v2_bucket": ch.get("v2_action_bucket"),
        "divergence_kind": ch.get("divergence_kind"),
        "differ": differ,
        "decision_reversal": decision_reversal,
        "horizon_turns": regret_horizon,
        "ship_diff_hybrid_minus_v2": reg["ship_diff_hybrid_minus_v2"],
        "planet_diff_hybrid_minus_v2": reg["planet_diff_hybrid_minus_v2"],
        "hybrid_worse_ships": reg["hybrid_worse_ships"],
        "v2_ships_after": reg["v2"]["ships"],
        "hybrid_ships_after": reg["hybrid"]["ships"],
        "v2_planets_after": reg["v2"]["planets"],
        "hybrid_planets_after": reg["hybrid"]["planets"],
        "winner_by_ships": winner,
        "v2_candidate": ch.get("v2_candidate"),
        "hybrid_candidate": ch.get("hybrid_candidate"),
        "rollout_margin": margin,
    }


def collect_decision_deltas(
    seeds: Sequence[int],
    act_timeout: float,
    rollout_opponent: str,
    decision_steps: Sequence[int] = (0, 10, 25, 50, 100),
    regret_horizon: int = 15,
    only_differ: bool = False,
    *,
    counterfactual: bool = False,
) -> List[Dict[str, Any]]:
    """Walk seeds with v2 replay; record delta at each decision step."""
    import os

    from kaggle_environments import make
    from src.policy.v2_bridge import v2_agent

    os.environ["ORBIT_ROLLOUT_OPPONENT"] = rollout_opponent
    cfg = {"actTimeout": act_timeout}
    records: List[Dict[str, Any]] = []

    for seed in seeds:
        env = make(
            "orbit_wars",
            debug=False,
            configuration={"seed": seed, "actTimeout": act_timeout},
        )
        env.reset(2)
        while not env.done:
            step = int(env.state[0].observation.step)
            if step in decision_steps:
                obs0 = env.state[0].observation
                rec = build_delta_record(
                    seed,
                    step,
                    obs0,
                    cfg,
                    regret_horizon,
                    act_timeout,
                    v2_agent,
                    rollout_opponent,
                    counterfactual=counterfactual,
                )
                if rec is not None:
                    if only_differ and not rec.get("differ"):
                        pass
                    else:
                        records.append(rec)
            env.step([v2_agent(env.state[0].observation, cfg), []])

    return records
