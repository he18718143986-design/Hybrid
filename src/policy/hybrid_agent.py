"""Hybrid agent entry — phases: rules (v2) → search → model."""

from __future__ import annotations

import os
from typing import Any, List, Optional

from src.env.legality import filter_legal_moves, pick_legal_or_fallback
from src.policy.fallback import fallback_moves
from src.policy.v2_bridge import v2_agent

Move = List[Any]

# ORBIT_AGENT_MODE: "v2" (default) | "hybrid" (partial pipeline)
_AGENT_MODE = os.environ.get("ORBIT_AGENT_MODE", "v2").lower()


def agent(obs: Any, config: Optional[dict] = None) -> List[Move]:
    """Kaggle-compatible entry point.

    Phase 0: delegate to submission_v2 (production SSOT).
    Phase 1+: ORBIT_AGENT_MODE=hybrid enables modular pipeline pieces.
    """
    if _AGENT_MODE == "v2":
        moves = v2_agent(obs, config)
        return filter_legal_moves(moves, obs)

    try:
        moves = _hybrid_agent(obs, config)
        moves = filter_legal_moves(moves, obs)
        return moves if moves else fallback_moves(obs)
    except Exception:
        moves = v2_agent(obs, config)
        return filter_legal_moves(moves, obs) or fallback_moves(obs)


def _hybrid_agent(obs: Any, config: Optional[dict] = None) -> List[Move]:
    """World → candidates (v2 settle) → rollout re-rank → legality gate → else v2."""
    from src.candidate.generate import generate_all_candidates
    from src.search.evaluator import score_candidates
    from src.search.budget import TimeBudget
    from src.world.state import World

    v2_moves = v2_agent(obs, config)
    budget = TimeBudget.from_config(config)
    world = World.from_obs(obs)
    candidates = generate_all_candidates(world, budget)
    if not candidates:
        return v2_moves

    scored = score_candidates(
        world, candidates, budget, rollout_depth=4, use_rollout=True,
    )
    if not scored:
        return v2_moves

    best = scored[0]
    hybrid_moves: List[Move] = [[best.src_id, best.angle, best.ships]]
    return pick_legal_or_fallback(hybrid_moves, v2_moves, obs)
