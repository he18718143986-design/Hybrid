"""Heuristic target/candidate scoring — delegate to submission_v2.target_value."""

from __future__ import annotations

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.world.state import World


def target_value(
    world: World,
    planet: Planet,
    modes: dict,
    mission: str = "capture",
    arrival_turns: int = 10,
) -> float:
    from src.policy.v2_bridge import get_v2_module

    v2 = get_v2_module()
    policy = v2.build_policy_state(world.inner)
    return v2.target_value(
        planet, arrival_turns, mission, world.inner, modes, policy,
    )


def apply_score_modifiers(base: float, target: Planet, mission: str, world: World) -> float:
    from src.policy.v2_bridge import get_v2_module

    return get_v2_module().apply_score_modifiers(base, target, mission, world.inner)
