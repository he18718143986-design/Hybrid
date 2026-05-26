"""Vectorize world + candidate for ML models."""

from __future__ import annotations

from typing import List, Optional

from src.candidate.features import candidate_features, global_features
from src.candidate.generate import Candidate
from src.world.state import World


def build_state_vector(world: World) -> List[float]:
    return global_features(world)


def build_candidate_vector(world: World, candidate: Optional[Candidate] = None) -> List[float]:
    if candidate is None:
        return build_state_vector(world)
    feats = candidate_features(
        world,
        candidate.src_id,
        candidate.target_id,
        candidate.ships,
        candidate.eta,
        candidate.heuristic_score,
    )
    return list(feats.values())
