"""Shallow forward simulation for candidate evaluation."""

from __future__ import annotations

from typing import Optional

from src.candidate.generate import Candidate
from src.search.budget import TimeBudget
from src.world.state import World
from src.search.opponent_model import generate_enemy_moves
from src.world.step import StepState, simulate_one_turn


def evaluate_step_state(state: StepState, player: int) -> float:
    """Leaf value: ship/prod advantage minus incoming threat and fragile holdings."""
    my_ships = sum(int(p.ships) for p in state.planets if p.owner == player)
    my_ships += sum(
        ships for _, _, owner, ships in state.scheduled_arrivals if owner == player
    )
    my_prod = sum(int(p.production) for p in state.planets if p.owner == player)

    enemy_ships = 0
    enemy_prod = 0
    for p in state.planets:
        if p.owner not in (-1, player):
            enemy_ships += int(p.ships)
            enemy_prod += int(p.production)
    for _, _, owner, ships in state.scheduled_arrivals:
        if owner not in (-1, player):
            enemy_ships += ships

    total = my_ships + enemy_ships + 1
    ship_adv = (my_ships - enemy_ships) / total
    prod_adv = (my_prod - enemy_prod) / (my_prod + enemy_prod + 1)

    incoming_threat = 0.0
    for _abs_step, planet_id, owner, ships in state.scheduled_arrivals:
        planet = state.planet_by_id.get(planet_id)
        if planet and planet.owner == player and owner not in (-1, player):
            incoming_threat += ships / (total + 1)

    fragile_penalty = 0.0
    for p in state.planets:
        if p.owner == player and int(p.ships) < 5 and int(p.production) <= 2:
            fragile_penalty += 0.02

    return (
        0.6 * ship_adv
        + 0.25 * prod_adv
        - 0.10 * incoming_threat
        - fragile_penalty
    )


def rollout(
    world: World,
    candidate: Candidate,
    depth: int = 6,
    budget: Optional[TimeBudget] = None,
) -> float:
    """Simulate `depth` turns; enemies use light frontier model (see opponent_model)."""
    if budget is not None and budget.expired():
        return 0.0

    state = StepState.from_context(world.ctx)

    for turn_idx in range(max(0, depth)):
        if budget is not None and budget.expired():
            break

        my_moves = []
        if turn_idx == 0:
            my_moves = [[candidate.src_id, candidate.angle, candidate.ships]]

        moves_by_player = {world.player: my_moves, **generate_enemy_moves(state, world.player)}
        state = simulate_one_turn(state, moves_by_player)

    return evaluate_step_state(state, world.player)
