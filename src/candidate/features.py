"""Feature extraction for candidates and value model."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.world.state import World


def global_features(world: World) -> List[float]:
    w = world.inner
    player = world.player
    my_ships = sum(int(p.ships) for p in w.planets if p.owner == player)
    my_ships += sum(int(f.ships) for f in w.fleets if f.owner == player)
    enemy_ships = sum(
        int(p.ships) for p in w.planets if p.owner not in (-1, player)
    )
    return [
        float(world.step),
        float(w.is_four_player),
        float(my_ships),
        float(enemy_ships),
        float(len(w.my_planets)),
        float(len(w.enemy_planets)),
        float(len(w.neutral_planets)),
    ]


def candidate_features(
    world: World,
    src_id: int,
    target_id: int,
    ships: int,
    eta: int,
    heuristic_score: float,
) -> Dict[str, float]:
    return {
        "ships": float(ships),
        "eta": float(eta),
        "heuristic_score": heuristic_score,
        **{f"g_{i}": v for i, v in enumerate(global_features(world))},
    }
