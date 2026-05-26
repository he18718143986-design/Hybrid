"""Hybrid agent — v2 plan by default; gated single-fleet override only."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.env.legality import filter_legal_moves, merge_override_move, validate_move_for_obs
from src.policy.fallback import fallback_moves
from src.policy.v2_bridge import v2_agent

Move = List[Any]

# ORBIT_AGENT_MODE: "v2" (default) | "hybrid"
# Set ORBIT_HYBRID_DISABLE_GATE=1 to always return v2 (smoke / ablation)
_DISABLE_GATE = os.environ.get("ORBIT_HYBRID_DISABLE_GATE", "").lower() in ("1", "true", "yes")


def _agent_mode() -> str:
    return os.environ.get("ORBIT_AGENT_MODE", "v2").lower()


def agent(obs: Any, config: Optional[dict] = None) -> List[Move]:
    """Kaggle entry: production defaults to v2; hybrid is opt-in via env."""
    if _agent_mode() == "v2":
        moves = v2_agent(obs, config)
        return filter_legal_moves(moves, obs)

    try:
        moves = _hybrid_agent(obs, config)
        moves = filter_legal_moves(moves, obs)
        if not moves:
            # v2 pass is intentional — do not inject fallback fleet
            if not filter_legal_moves(v2_agent(obs, config), obs):
                return []
            return fallback_moves(obs)
        return moves
    except Exception:
        moves = v2_agent(obs, config)
        return filter_legal_moves(moves, obs) or fallback_moves(obs)


def _hybrid_agent(obs: Any, config: Optional[dict] = None) -> List[Move]:
    """Full v2 move list + at most one gated fleet override (never Top-1 replace)."""
    from src.candidate.generate import generate_all_candidates
    from src.search.evaluator import score_candidates
    from src.search.budget import TimeBudget
    from src.search.probe_metrics import (
        classify_decision_bucket,
        rollout_margin_vs_v2,
        _match_scored_candidate,
    )
    from src.world.state import World

    v2_moves = filter_legal_moves(v2_agent(obs, config), obs)
    # v2 pass → never inject a fleet; avoids diverge when baseline chooses to wait
    if not v2_moves:
        return v2_moves

    if _DISABLE_GATE:
        return v2_moves

    budget = TimeBudget.from_config(config)
    world = World.from_obs(obs)
    candidates = generate_all_candidates(world, budget)
    if not candidates or budget.expired():
        return v2_moves

    scored = score_candidates(
        world, candidates, budget, rollout_depth=8, use_rollout=True,
    )
    if not scored or budget.expired():
        return v2_moves

    best = scored[0]
    v2_primary = v2_moves[0] if v2_moves else None
    v2_top = _match_scored_candidate(scored, v2_primary)
    margin = rollout_margin_vs_v2(best, v2_top)
    if margin is None:
        return v2_moves

    bucket = classify_decision_bucket(obs, candidate=best, move=v2_primary)
    # Case A / gate_policy: neutral_capture — never override (overextended_neutral)
    if bucket == "neutral_capture":
        return v2_moves

    override_move: Move = [best.src_id, best.angle, best.ships]

    if not validate_move_for_obs(override_move, obs):
        return v2_moves

    # Same as v2 top fleet — no change
    if v2_primary and list(v2_primary) == override_move:
        return v2_moves

    try:
        from src.policy.gate_policy_v1 import decide_override

        if not decide_override(bucket, margin):
            return v2_moves
    except ImportError:
        return v2_moves

    return merge_override_move(v2_moves, override_move, obs)
