"""Light opponent model for rollout."""

from __future__ import annotations

import os

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from src.search.opponent_model import generate_enemy_moves
from src.world.step import StepState


def _planet(pid, owner, x, y, ships, r=2.0, prod=2):
    return Planet(pid, owner, x, y, r, ships, prod)


def test_passive_mode_empty():
    os.environ["ORBIT_ROLLOUT_OPPONENT"] = "passive"
    state = StepState(
        player=0,
        step=1,
        planets=[_planet(1, 0, 10.0, 50.0, 10), _planet(2, 1, 60.0, 50.0, 10)],
        initial_by_id={
            1: _planet(1, 0, 10.0, 50.0, 10),
            2: _planet(2, 1, 60.0, 50.0, 10),
        },
        comets=[],
        comet_ids=set(),
        angular_velocity=0.0,
    )
    moves = generate_enemy_moves(state, 0)
    assert moves[1] == []


def test_reinforce_under_incoming_attack():
    os.environ["ORBIT_ROLLOUT_OPPONENT"] = "frontier"
    # P1 owns planet 2; P0 scheduled to capture at step 3.
    state = StepState(
        player=0,
        step=1,
        planets=[_planet(1, 1, 10.0, 8.0, 30), _planet(2, 1, 90.0, 8.0, 8)],
        initial_by_id={
            1: _planet(1, 1, 10.0, 8.0, 30),
            2: _planet(2, 1, 90.0, 8.0, 8),
        },
        comets=[],
        comet_ids=set(),
        angular_velocity=0.0,
        scheduled_arrivals=[(3, 2, 0, 20)],
    )
    moves = generate_enemy_moves(state, 0)
    assert 1 in moves
    assert len(moves[1]) >= 1
    m = moves[1][0]
    assert m[2] >= 1
