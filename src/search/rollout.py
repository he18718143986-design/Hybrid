"""Shallow forward simulation for candidate evaluation."""

from __future__ import annotations

from typing import Optional

from src.candidate.generate import Candidate
from src.search.budget import TimeBudget
from src.world.state import World
from src.search.opponent_model import generate_enemy_moves
from src.world.step import StepState, simulate_one_turn


def evaluate_step_state(state: StepState, player: int) -> float:
    """Simple rollout leaf value: ship advantage + production weight."""
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
    return 0.7 * ship_adv + 0.3 * prod_adv


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
