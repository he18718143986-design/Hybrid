"""Top-K candidate moves from missions via settle_plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.candidate.missions import generate_missions
from src.search.budget import TimeBudget
from src.world.state import World


@dataclass
class Candidate:
    src_id: int
    target_id: int
    angle: float
    ships: int
    eta: int
    heuristic_score: float
    rollout_score: float = 0.0
    model_score: float = 0.0

    @property
    def final_score(self) -> float:
        return (
            0.4 * self.heuristic_score
            + 0.4 * self.rollout_score
            + 0.2 * self.model_score
        )


def generate_candidates_for_mission(
    world: World,
    mission,
    modes: dict,
    policy: dict,
    planned_commitments,
    budget: TimeBudget,
    top_k: int = 3,
) -> List[Candidate]:
    from src.policy.v2_bridge import get_v2_module

    v2 = get_v2_module()
    w = world.inner
    target = w.planet_by_id.get(mission.target_id)
    if target is None:
        return []

    out: List[Candidate] = []
    send_guess = int(target.ships) + 1
    for src in w.my_planets:
        if budget.expired():
            break
        src_cap = int(src.ships)
        if src_cap < 1:
            continue
        settled = v2.settle_plan(
            src,
            target,
            src_cap,
            send_guess,
            w,
            planned_commitments,
            modes,
            policy,
            mission=mission.kind,
        )
        if settled is None:
            continue
        angle, eta, _eval_turn, need, send = settled
        score = mission.priority - eta * 0.5 - send * 0.02
        out.append(
            Candidate(
                src_id=src.id,
                target_id=target.id,
                angle=angle,
                ships=send,
                eta=eta,
                heuristic_score=score,
            ),
        )
    out.sort(key=lambda c: c.heuristic_score, reverse=True)
    return out[:top_k]


def generate_all_candidates(world: World, budget: TimeBudget) -> List[Candidate]:
    from src.policy.v2_bridge import get_v2_module

    v2 = get_v2_module()
    w = world.inner
    modes = v2.build_modes(w)
    policy = v2.build_policy_state(w)
    planned = {}
    candidates: List[Candidate] = []
    for mission in generate_missions(world, modes, budget, policy=policy):
        if budget.expired():
            break
        for cand in generate_candidates_for_mission(
            world, mission, modes, policy, planned, budget,
        ):
            candidates.append(cand)
    candidates.sort(key=lambda c: c.heuristic_score, reverse=True)
    return candidates[:8]
