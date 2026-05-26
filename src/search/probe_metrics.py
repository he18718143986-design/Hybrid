"""Rollout divergence and real-env regret metrics for hybrid probes."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.candidate.generate import Candidate
from src.env.parser import parse_observation
from src.search.evaluator import score_candidates
from src.search.budget import TimeBudget
from src.world.orbit import is_orbiting_initial
from src.world.state import World

Move = List[Any]

REGRET_BUCKETS = (
    "pass",
    "neutral_capture",
    "aggressive_expansion",
    "reinforcement",
    "comet_chase",
    "rotating_intercept",
    "unknown",
    "other",
)


def _primary_move(moves: Sequence[Move]) -> Optional[Move]:
    return list(moves[0]) if moves else None


def _match_scored_candidate(
    scored: Sequence[Candidate], move: Optional[Move],
) -> Optional[Candidate]:
    """Map an executed move to its scored candidate (pass → None)."""
    if not move:
        return None
    m = list(move)
    if len(m) < 3:
        return None
    src, angle, ships = int(m[0]), float(m[1]), int(m[2])
    for c in scored:
        if (
            c.src_id == src
            and abs(c.angle - angle) < 1e-6
            and c.ships == ships
        ):
            return c
    return None


def rollout_margin_vs_v2(
    hybrid_cand: Optional[Candidate],
    v2_cand: Optional[Candidate],
) -> Optional[float]:
    """Same-scale margin: hybrid rollout minus v2-executed rollout (pass → 0)."""
    if hybrid_cand is None:
        return None
    v2_rollout = float(v2_cand.rollout_score) if v2_cand else 0.0
    return float(hybrid_cand.rollout_score) - v2_rollout


def _candidate_dict(c: Candidate, *, include_rollout: bool = True) -> dict:
    d = {
        "src_id": c.src_id,
        "target_id": c.target_id,
        "ships": c.ships,
        "eta": c.eta,
        "heuristic_score": c.heuristic_score,
    }
    if include_rollout:
        d["rollout_score"] = c.rollout_score
        d["final_score"] = c.final_score
    return d


def classify_rollout_divergence(
    v2_move: Optional[Move],
    hybrid_move: Optional[Move],
    v2_candidate: Optional[Candidate],
    hybrid_candidate: Optional[Candidate],
) -> str:
    """How hybrid rerank differs from v2's first move / top heuristic candidate."""
    if v2_move is None and hybrid_move is None:
        return "same_empty"
    if v2_move is not None and hybrid_move is not None and v2_move == hybrid_move:
        return "same_decision"

    if v2_candidate and hybrid_candidate:
        if v2_candidate.target_id != hybrid_candidate.target_id:
            return "different_target"
        if v2_candidate.src_id != hybrid_candidate.src_id:
            return "different_source"
        if v2_candidate.ships != hybrid_candidate.ships:
            return "different_ships"
        if v2_candidate.eta != hybrid_candidate.eta:
            return "different_eta"
        if abs(v2_candidate.angle - hybrid_candidate.angle) > 1e-6:
            return "different_angle"

    if v2_move is None or hybrid_move is None:
        return "different_presence"
    if int(v2_move[0]) != int(hybrid_move[0]):
        return "different_source"
    if int(v2_move[2]) != int(hybrid_move[2]):
        return "different_ships"
    if abs(float(v2_move[1]) - float(hybrid_move[1])) > 1e-6:
        return "different_angle"
    return "different_other"


def aggregate_divergence(samples: List[dict]) -> dict[str, Any]:
    n = len(samples)
    if n == 0:
        return {"n": 0}
    counts: Dict[str, int] = {}
    for s in samples:
        kind = s.get("divergence_kind", "unknown")
        counts[kind] = counts.get(kind, 0) + 1
    return {
        "n": n,
        "counts": counts,
        "rates": {k: v / n for k, v in counts.items()},
        "same_decision_rate": counts.get("same_decision", 0) / n
        + counts.get("same_empty", 0) / n,
    }


def classify_decision_bucket(
    obs: Any,
    candidate: Optional[Candidate] = None,
    move: Optional[Move] = None,
) -> str:
    """Regret breakdown category for a single opening decision."""
    if candidate is None and (move is None or len(move) < 3):
        return "pass"

    parsed = parse_observation(obs)
    player = parsed["player"]
    comet_ids = set(parsed["comet_planet_ids"])
    initial = {p.id: p for p in parsed["initial_planets"]}
    planets = {p.id: p for p in parsed["planets"]}

    if candidate is not None:
        src_id, target_id = candidate.src_id, candidate.target_id
    else:
        src_id = int(move[0])  # type: ignore[index]
        target_id = None
        if move and len(move) >= 3:
            from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet
            from src.world.fleet import fleet_target_planet
            from src.world.geometry import launch_point

            src = planets.get(src_id)
            if src is None:
                return "unknown"
            angle = float(move[1])
            ships = int(move[2])
            lx, ly = launch_point(src.x, src.y, src.radius, angle)
            fleet = Fleet(-1, player, lx, ly, angle, src_id, ships)
            target, _ = fleet_target_planet(
                fleet,
                parsed["planets"],
                initial,
                parsed["angular_velocity"],
                parsed["comets"],
                comet_ids,
            )
            if target is None:
                return "unknown"
            target_id = target.id

    if target_id is None:
        return "unknown"

    target = planets.get(target_id)
    src = planets.get(src_id)
    if target is None:
        return "unknown"

    if target.id in comet_ids:
        return "comet_chase"
    if target.owner == -1:
        return "neutral_capture"
    if target.owner == player:
        return "reinforcement"
    init = initial.get(target.id)
    if init is not None and is_orbiting_initial(init):
        return "rotating_intercept"
    if target.owner not in (-1, player):
        return "aggressive_expansion"
    return "other"


def moves_differ(v2_moves: Sequence[Move], hybrid_moves: Sequence[Move]) -> bool:
    """True if hybrid opening differs from v2 (first move or presence)."""
    v2m = _primary_move(v2_moves)
    hym = _primary_move(hybrid_moves)
    if v2m is None and hym is None:
        return False
    return v2m != hym


def is_decision_reversal(row: dict) -> bool:
    """Hybrid changed v2's choice and real-env regret is negative for hybrid."""
    if not row.get("differ"):
        return False
    return float(row.get("ship_diff_hybrid_minus_v2", 0)) < 0


def is_positive_override(row: dict) -> bool:
    """Hybrid diverged from v2 and real-env outcome improved."""
    if not row.get("differ"):
        return False
    return float(row.get("ship_diff_hybrid_minus_v2", 0)) > 0


def summarize_override_precision(rows: List[dict]) -> dict[str, Any]:
    """Override precision/recall dashboard (paired counterfactual at each state)."""
    if not rows:
        return {"n": 0}

    n = len(rows)
    diverged = [r for r in rows if r.get("differ")]
    nd = len(diverged)
    tp = sum(1 for r in diverged if is_positive_override(r))
    fp = sum(1 for r in diverged if is_decision_reversal(r))
    opportunities = [r for r in rows if float(r.get("ship_diff_hybrid_minus_v2", 0)) > 0]
    no = len(opportunities)

    reversals = [r for r in diverged if is_decision_reversal(r)]
    reversal_severity = (
        sum(abs(float(r["ship_diff_hybrid_minus_v2"])) for r in reversals) / len(reversals)
        if reversals
        else 0.0
    )

    # margin vs outcome (when rollout_margin recorded)
    paired = [
        (float(r.get("rollout_margin", 0)), float(r["ship_diff_hybrid_minus_v2"]))
        for r in diverged
        if r.get("rollout_margin") is not None
    ]
    calibration = None
    if len(paired) >= 3:
        margins = [p[0] for p in paired]
        outcomes = [p[1] for p in paired]
        mean_m = sum(margins) / len(margins)
        mean_o = sum(outcomes) / len(outcomes)
        if sum((m - mean_m) ** 2 for m in margins) > 1e-9:
            cov = sum((m - mean_m) * (o - mean_o) for m, o in zip(margins, outcomes)) / len(
                paired,
            )
            var_m = sum((m - mean_m) ** 2 for m in margins) / len(paired)
            calibration = {"margin_outcome_correlation": cov / (var_m ** 0.5 + 1e-9)}

    return {
        "n": n,
        "override_precision": tp / max(1, nd),
        "override_recall": tp / max(1, no),
        "positive_override_rate": tp / max(1, n),
        "diverged_count": nd,
        "opportunity_count": no,
        "true_positive_overrides": tp,
        "false_positive_overrides": fp,
        "decision_reversal_given_diverged": fp / max(1, nd),
        "reversal_severity_mean_ships": reversal_severity,
        "confidence_calibration": calibration,
    }


def aggregate_regret_by_bucket(rows: List[dict]) -> dict[str, Any]:
    """Summarize regret rows by classify_decision_bucket (hybrid opening)."""
    by_bucket: Dict[str, List[dict]] = {}
    for row in rows:
        bucket = row.get("action_bucket", "unknown")
        by_bucket.setdefault(bucket, []).append(row)

    summary: Dict[str, Any] = {}
    for bucket, items in sorted(by_bucket.items()):
        n = len(items)
        worse = sum(1 for r in items if r.get("hybrid_worse_ships"))
        diverged = sum(1 for r in items if r.get("differ"))
        reversals = sum(1 for r in items if is_decision_reversal(r))
        summary[bucket] = {
            "n": n,
            "hybrid_worse_ships_rate": worse / max(1, n),
            "avg_ship_diff": sum(r["ship_diff_hybrid_minus_v2"] for r in items) / max(1, n),
            "avg_planet_diff": sum(r["planet_diff_hybrid_minus_v2"] for r in items) / max(1, n),
            "diverged_rate": diverged / max(1, n),
            "decision_reversal_rate": reversals / max(1, n),
            "decision_reversal_given_diverged": reversals / max(1, diverged),
        }
    return summary


def summarize_regret(rows: List[dict]) -> dict[str, Any]:
    """Overall regret + decision_reversal_rate + same_decision complement."""
    if not rows:
        return {"n": 0, "rows": []}

    n = len(rows)
    worse = sum(1 for r in rows if r.get("hybrid_worse_ships"))
    diverged_count = sum(1 for r in rows if r.get("differ"))
    reversals = sum(1 for r in rows if is_decision_reversal(r))
    same = n - diverged_count

    out = {
        "n": n,
        "avg_ship_diff": sum(r["ship_diff_hybrid_minus_v2"] for r in rows) / n,
        "avg_planet_diff": sum(r["planet_diff_hybrid_minus_v2"] for r in rows) / n,
        "hybrid_worse_ships_rate": worse / n,
        "same_decision_rate": same / n,
        "diverged_rate": diverged_count / n,
        "decision_reversal_rate": reversals / n,
        "decision_reversal_given_diverged": reversals / max(1, diverged_count),
        "by_bucket": aggregate_regret_by_bucket(rows),
        "override_precision_dashboard": summarize_override_precision(rows),
        "rows": rows,
    }
    return out


def choices_at_obs(obs: Any, config: Optional[dict]) -> dict[str, Any]:
    """v2 vs hybrid decisions + candidate metadata at one observation."""
    from src.candidate.generate import generate_all_candidates
    from src.env.legality import pick_legal_or_fallback
    from src.policy.v2_bridge import v2_agent

    world = World.from_obs(obs)
    v2_moves = v2_agent(obs, config)

    scored: List[Candidate] = []
    hy_top = None
    try:
        score_budget = TimeBudget.from_config(config)
        candidates = generate_all_candidates(world, score_budget)
        scored = score_candidates(
            world, candidates, score_budget, rollout_depth=4, use_rollout=True,
        )
        hy_top = scored[0] if scored else None
    except Exception:
        hy_top = None

    if hy_top is not None:
        hy_moves = pick_legal_or_fallback(
            [[hy_top.src_id, hy_top.angle, hy_top.ships]], v2_moves, obs,
        )
    else:
        hy_moves = v2_moves

    v2m = _primary_move(v2_moves)
    hym = _primary_move(hy_moves)
    v2_top = _match_scored_candidate(scored, v2m) if scored else None
    margin = rollout_margin_vs_v2(hy_top, v2_top)
    kind = classify_rollout_divergence(v2m, hym, v2_top, hy_top)
    v2_bucket = classify_decision_bucket(obs, candidate=v2_top, move=v2m)
    hy_bucket = classify_decision_bucket(obs, candidate=hy_top, move=hym)

    return {
        "v2_moves": v2_moves,
        "hybrid_moves": hy_moves,
        "v2_candidate": None if v2_top is None else _candidate_dict(v2_top),
        "hybrid_candidate": None if hy_top is None else _candidate_dict(hy_top),
        "rollout_margin": margin,
        "divergence_kind": kind,
        "differ": kind not in ("same_decision", "same_empty"),
        "v2_action_bucket": v2_bucket,
        "hybrid_action_bucket": hy_bucket,
    }


def _player_stats(obs: Any, player: int) -> Tuple[int, int]:
    parsed = parse_observation(obs)
    ships = 0
    planets = 0
    for p in parsed["planets"]:
        if p.owner == player:
            planets += 1
            ships += int(p.ships)
    for f in parsed["fleets"]:
        if f.owner == player:
            ships += int(f.ships)
    return ships, planets


def real_env_playout(
    seed: int,
    decision_step: int,
    forced_move: Sequence[Move],
    horizon: int,
    act_timeout: float,
    follow_agent,
) -> dict[str, int]:
    """Replay with follow_agent until decision_step, force P0 move, continue vs passive P1."""
    from kaggle_environments import make

    cfg = {"actTimeout": act_timeout}
    env = make(
        "orbit_wars",
        debug=False,
        configuration={"seed": seed, "actTimeout": act_timeout},
    )
    env.reset(2)
    player = 0
    end_step = decision_step + horizon

    while not env.done:
        step = int(env.state[0].observation.step)
        if step > end_step:
            break
        if step < decision_step:
            obs0 = env.state[0].observation
            env.step([follow_agent(obs0, cfg), []])
        elif step == decision_step:
            p0 = [list(m) for m in forced_move] if forced_move else []
            env.step([p0, []])
        else:
            obs0 = env.state[0].observation
            env.step([follow_agent(obs0, cfg), []])

    obs = env.state[0].observation
    ships, planets = _player_stats(obs, player)
    return {"ships": ships, "planets": planets, "step": int(obs.step)}


def measure_regret(
    seed: int,
    decision_step: int,
    v2_move: Sequence[Move],
    hybrid_move: Sequence[Move],
    horizon: int,
    act_timeout: float,
    follow_agent,
    action_bucket: str = "unknown",
    differ: Optional[bool] = None,
    rollout_margin: Optional[float] = None,
) -> dict[str, Any]:
    """Compare real-env outcomes after v2 vs hybrid opening at one step."""
    v2_out = real_env_playout(
        seed, decision_step, v2_move, horizon, act_timeout, follow_agent,
    )
    hy_out = real_env_playout(
        seed, decision_step, hybrid_move, horizon, act_timeout, follow_agent,
    )
    ship_diff = hy_out["ships"] - v2_out["ships"]
    if differ is None:
        differ = moves_differ(v2_move, hybrid_move)
    row = {
        "seed": seed,
        "decision_step": decision_step,
        "action_bucket": action_bucket,
        "differ": differ,
        "v2": v2_out,
        "hybrid": hy_out,
        "ship_diff_hybrid_minus_v2": ship_diff,
        "planet_diff_hybrid_minus_v2": hy_out["planets"] - v2_out["planets"],
        "hybrid_worse_ships": hy_out["ships"] < v2_out["ships"],
    }
    row["decision_reversal"] = is_decision_reversal(row)
    if rollout_margin is not None:
        row["rollout_margin"] = float(rollout_margin)
    return row
