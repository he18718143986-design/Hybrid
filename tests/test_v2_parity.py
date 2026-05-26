"""Native src/world must match submission_v2 on core physics."""

from __future__ import annotations

from types import SimpleNamespace

from src.policy.v2_bridge import get_v2_module
from src.world.combat import resolve_arrival_event as src_resolve
from src.world.context import build_world_context
from src.world.fleet import fleet_target_planet as src_fleet_target
from src.world.need import min_ships_to_own_at as src_need
from src.world.timeline import simulate_planet_timeline as src_sim


def test_resolve_matches_v2():
    v2 = get_v2_module()
    cases = [
        (0, 10.0, [(1, 1, 15)]),
        (0, 10.0, [(1, 1, 8), (1, 2, 8)]),
        (1, 5.0, [(2, 0, 3), (2, 2, 10)]),
    ]
    for owner, garrison, arrivals in cases:
        assert src_resolve(owner, garrison, arrivals) == v2.resolve_arrival_event(
            owner, garrison, arrivals,
        )


def test_timeline_neutral_capture_need():
    v2 = get_v2_module()
    planet = SimpleNamespace(id=1, owner=-1, ships=10, production=3, radius=2.0)
    arrivals = [(4, 0, 12)]
    src_tl = src_sim(planet, arrivals, player=0, horizon=20)
    v2_tl = v2.simulate_planet_timeline(planet, arrivals, 0, 20)
    assert src_tl["owner_at"][4] == v2_tl["owner_at"][4]
    assert src_tl["ships_at"][4] == v2_tl["ships_at"][4]


def test_min_ships_parity_on_synthetic_obs():
    """Smoke: native need matches v2 on a minimal env step if kaggle_environments available."""
    try:
        from kaggle_environments import make
    except ImportError:
        return

    env = make("orbit_wars", configuration={"seed": 42}, debug=False)
    obs = env.reset()[0]
    ctx = build_world_context(obs)
    v2w = get_v2_module().build_world(obs, inferred_step=0)

    for planet in ctx.planets:
        if planet.owner == -1 and planet.ships < 30:
            eta = 8
            src_n = src_need(ctx, planet.id, eta, upper_bound=50)
            v2_n = v2w.min_ships_to_own_at(planet.id, eta, upper_bound=50)
            assert src_n == v2_n, (planet.id, src_n, v2_n)
            break
