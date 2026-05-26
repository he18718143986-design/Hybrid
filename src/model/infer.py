"""Online model inference — must stay sub-millisecond."""

from __future__ import annotations

from src.candidate.generate import Candidate
from src.model.feature_builder import build_candidate_vector
from src.model.value_net import predict_value
from src.world.state import World


def value_predict(world: World, candidate: Candidate | None = None) -> float:
    vec = build_candidate_vector(world, candidate)
    return predict_value(vec)
