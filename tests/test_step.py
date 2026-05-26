"""One-turn step simulator tests."""

from __future__ import annotations

import math

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.world.constants import CENTER_X, CENTER_Y
from src.world.step import StepState, passive_enemy_moves, simulate_one_turn


def _planet(pid, owner, ships, prod=2, x=10.0, y=50.0, r=2.0):
    return Planet(pid, owner, x, y, r, ships, prod)


def test_simulate_production():
    state = StepState(
        player=0,
        step=5,
        planets=[_planet(1, 0, 10, prod=3)],
        initial_by_id={1: _planet(1, 0, 10)},
        comets=[],
        comet_ids=set(),
        angular_velocity=0.0,
    )
    next_s = simulate_one_turn(state, {0: []})
    assert next_s.step == 6
    assert next_s.planet_by_id[1].ships == 13


def test_scheduled_arrival_captures():
    state = StepState(
        player=0,
        step=1,
        planets=[_planet(1, -1, 10, prod=1)],
        initial_by_id={1: _planet(1, -1, 10)},
        comets=[],
        comet_ids=set(),
        angular_velocity=0.0,
        scheduled_arrivals=[(2, 1, 0, 12)],
    )
    next_s = simulate_one_turn(state, {0: []})
    assert next_s.planet_by_id[1].owner == 0
    assert next_s.planet_by_id[1].ships == 2  # 12-10


def test_orbiting_planet_advances():
    # Initial on rotating ring (r < 50); should move after one turn.
    init = Planet(1, 0, 60.0, 50.0, 2.0, 10, 2)
    state = StepState(
        player=0,
        step=0,
        planets=[init],
        initial_by_id={1: init},
        comets=[],
        comet_ids=set(),
        angular_velocity=0.05,
    )
    next_s = simulate_one_turn(state, {0: []})
    p = next_s.planet_by_id[1]
    moved = math.hypot(p.x - init.x, p.y - init.y) > 1e-6
    assert moved


def test_passive_enemy_empty():
    state = StepState(
        player=0,
        step=0,
        planets=[_planet(1, 0, 5), _planet(2, 1, 5)],
        initial_by_id={1: _planet(1, 0, 5), 2: _planet(2, 1, 5)},
        comets=[],
        comet_ids=set(),
        angular_velocity=0.0,
    )
    moves = passive_enemy_moves(state)
    assert moves[1] == []


def test_rollout_smoke():
    """Rollout returns a finite score on a real env step (no agent in loop)."""
    try:
        from kaggle_environments import make
    except ImportError:
        return

    from src.candidate.generate import Candidate
    from src.search.rollout import rollout
    from src.world.state import World

    env = make("orbit_wars", configuration={"seed": 7}, debug=False)
    world = World.from_obs(env.reset()[0])
    my = next(p for p in world.ctx.planets if p.owner == world.player)
    cand = Candidate(
        src_id=my.id,
        target_id=my.id,
        angle=0.0,
        ships=1,
        eta=1,
        heuristic_score=0.0,
    )
    score = rollout(world, cand, depth=2)
    assert -1.0 <= score <= 1.0
