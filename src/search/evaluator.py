"""Combine heuristic, rollout, and model scores."""

from __future__ import annotations

from typing import List

from src.candidate.generate import Candidate
from src.model.infer import value_predict
from src.search.budget import TimeBudget
from src.search.rollout import rollout
from src.world.state import World

WEIGHT_HEURISTIC = 0.4
WEIGHT_ROLLOUT = 0.4
WEIGHT_MODEL = 0.2
ROLLOUT_DEPTH = 12


def score_candidates(
    world: World,
    candidates: List[Candidate],
    budget: TimeBudget,
    rollout_depth: int = ROLLOUT_DEPTH,
    use_rollout: bool = True,
) -> List[Candidate]:
    for cand in candidates:
        if budget.expired():
            break
        if use_rollout:
            cand.rollout_score = rollout(world, cand, depth=rollout_depth, budget=budget)
        else:
            cand.rollout_score = 0.0
        cand.model_score = value_predict(world, cand)
    candidates.sort(key=lambda c: c.final_score, reverse=True)
    return candidates
