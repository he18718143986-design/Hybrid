"""Observation parsing — thin wrapper until extracted from submission_v2.build_world."""

from __future__ import annotations

from typing import Any, Dict, Optional

from kaggle_environments.envs.orbit_wars.orbit_wars import Fleet, Planet


def read_obs(obs: Any, key: str, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def parse_observation(obs: Any) -> Dict[str, Any]:
    """Normalize raw env observation into a plain dict of typed rows.

    Returns:
        player, step, planets, fleets, initial_planets, comets, comet_planet_ids,
        angular_velocity, raw (original obs).
    """
    # Kaggle agent() receives Agent.state; unwrap nested game observation.
    if isinstance(obs, dict) and "observation" in obs and "planets" not in obs:
        obs = obs["observation"]
    elif read_obs(obs, "observation", None) is not None and read_obs(obs, "planets", None) is None:
        obs = read_obs(obs, "observation")

    return {
        "player": int(read_obs(obs, "player", 0)),
        "step": int(read_obs(obs, "step", 0) or 0),
        "planets": [Planet(*row) for row in (read_obs(obs, "planets", []) or [])],
        "fleets": [Fleet(*row) for row in (read_obs(obs, "fleets", []) or [])],
        "initial_planets": [
            Planet(*row) for row in (read_obs(obs, "initial_planets", []) or [])
        ],
        "comets": read_obs(obs, "comets", []) or [],
        "comet_planet_ids": list(read_obs(obs, "comet_planet_ids", []) or []),
        "angular_velocity": float(read_obs(obs, "angular_velocity", 0.0) or 0.0),
        "raw": obs,
    }
