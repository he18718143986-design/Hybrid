"""Combat SSOT checks — via v2 bridge until full extract."""

from __future__ import annotations

from src.world.combat import resolve_arrival_event


def test_resolve_single_attacker_captures():
    owner, ships = resolve_arrival_event(0, 10.0, [(1, 1, 15)])
    assert owner == 1
    assert ships == 5.0


def test_resolve_tie_destroys_attackers():
    owner, ships = resolve_arrival_event(0, 10.0, [(1, 1, 8), (1, 2, 8)])
    assert owner == 0
    assert ships == 10.0
