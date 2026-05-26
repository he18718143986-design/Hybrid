"""Timeline behavior tests (native src/world)."""

from __future__ import annotations

from types import SimpleNamespace

from src.world.timeline import simulate_planet_timeline, state_at_timeline


def test_production_before_combat():
    planet = SimpleNamespace(id=1, owner=0, ships=5, production=2, radius=2.0)
    tl = simulate_planet_timeline(planet, [(1, 1, 3)], player=0, horizon=3)
    # Turn 1: 5+2 prod = 7, then fight 3 attackers
    owner, ships = state_at_timeline(tl, 1)
    assert owner == 0
    assert ships == 4.0


def test_fall_turn_recorded():
    planet = SimpleNamespace(id=1, owner=0, ships=5, production=1, radius=2.0)
    tl = simulate_planet_timeline(planet, [(2, 1, 20)], player=0, horizon=10)
    assert tl["fall_turn"] == 2
    assert tl["owner_at"][2] == 1
