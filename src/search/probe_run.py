"""Shared probe runners — games, regret, divergence (imported by scripts/)."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Sequence, Tuple, Union

ROOT = Path(__file__).resolve().parents[2]
AGENT_V2 = str(ROOT / "submission_v2.py")
AGENT_HYBRID = str(ROOT / "submission" / "hybrid_main.py")
SEEDS_FILE = ROOT / "kaggle_probe" / "regression" / "seeds.json"
SEEDS_PROBE_V1 = ROOT / "probe_results" / "seeds_probe_v1.json"
RESULTS_DIR = ROOT / "probe_results"
LOCAL_ACT_TIMEOUT = 999.0
KAGGLE_ACT_TIMEOUT = 1.0

AgentRef = Union[str, Callable[..., Any]]


def resolve_agent(ref: AgentRef) -> Callable[..., Any]:
    """Load probe agents as callables (kaggle path strings break hybrid PYTHONPATH)."""
    if callable(ref):
        return ref
    path = Path(ref).resolve()
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if path.name == "hybrid_main.py":
        import submission.hybrid_main as mod

        return mod.agent
    if path.name == "submission_v2.py":
        import submission_v2 as mod

        return mod.agent
    spec = importlib.util.spec_from_file_location(f"probe_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load agent from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "agent"):
        raise AttributeError(f"{path} has no agent()")
    return mod.agent


@dataclass
class EndStats:
    ships: int = 0
    planets: int = 0
    reward: float = 0.0
    status: str = ""


@dataclass
class GameResult:
    seed: int
    p0: EndStats = field(default_factory=EndStats)
    p1: EndStats = field(default_factory=EndStats)
    timeout_events: int = 0
    invalid_events: int = 0
    episode_invalid: bool = False
    duration_s: float = 0.0


@dataclass
class MatchSummary:
    name: str
    agent_p0: str
    agent_p1: str
    seeds: List[int]
    games: List[GameResult] = field(default_factory=list)

    def aggregate(self) -> dict[str, Any]:
        n = len(self.games)
        if n == 0:
            return {"name": self.name, "n": 0}

        completed = [
            g for g in self.games if g.p0.status == "DONE" and g.p1.status == "DONE"
        ]
        nc = len(completed)
        p0_wins = sum(1 for g in completed if g.p0.reward > g.p1.reward)
        p1_wins = sum(1 for g in completed if g.p1.reward > g.p0.reward)
        draws = nc - p0_wins - p1_wins

        return {
            "name": self.name,
            "agent_p0": self.agent_p0,
            "agent_p1": self.agent_p1,
            "n_seeds": n,
            "completed": nc,
            "episode_invalid": sum(1 for g in self.games if g.episode_invalid),
            "p0_winrate": p0_wins / max(1, nc),
            "p1_winrate": p1_wins / max(1, nc),
            "draw_rate": draws / max(1, nc),
            "avg_final_ships": [
                sum(g.p0.ships for g in completed) / max(1, nc),
                sum(g.p1.ships for g in completed) / max(1, nc),
            ],
            "avg_planets_owned": [
                sum(g.p0.planets for g in completed) / max(1, nc),
                sum(g.p1.planets for g in completed) / max(1, nc),
            ],
            "total_timeout_events": sum(g.timeout_events for g in self.games),
            "total_invalid_events": sum(g.invalid_events for g in self.games),
            "avg_duration_s": sum(g.duration_s for g in self.games) / n,
        }


def _row_stats(obs: Any, player: int) -> Tuple[int, int]:
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    ships = owned = 0
    for row in planets:
        owner = int(row[1]) if isinstance(row, (list, tuple)) else int(row.owner)
        nships = int(row[5]) if isinstance(row, (list, tuple)) else int(row.ships)
        if owner == player:
            owned += 1
            ships += nships
    for row in fleets:
        owner = int(row[1]) if isinstance(row, (list, tuple)) else int(row.owner)
        nships = int(row[6]) if isinstance(row, (list, tuple)) else int(row.ships)
        if owner == player:
            ships += nships
    return ships, owned


def _unwrap_obs(agent_state: Any) -> Any:
    if isinstance(agent_state, dict) and "observation" in agent_state:
        return agent_state["observation"]
    obs = getattr(agent_state, "observation", None)
    return obs if obs is not None else agent_state


def _scan_step_statuses(env) -> Tuple[int, int]:
    timeouts = invalids = 0
    for step in env.steps:
        for s in step:
            st = getattr(s, "status", "")
            if st == "TIMEOUT":
                timeouts += 1
            elif st in ("INVALID", "ERROR"):
                invalids += 1
    return timeouts, invalids


def run_game(
    agents: Sequence[AgentRef],
    seed: int,
    act_timeout: float,
    num_players: int = 2,
) -> GameResult:
    from kaggle_environments import make

    env = make(
        "orbit_wars",
        debug=False,
        configuration={"seed": seed, "actTimeout": act_timeout},
    )
    t0 = time.perf_counter()
    env.run([resolve_agent(a) for a in agents])
    dt = time.perf_counter() - t0
    final = env.steps[-1]
    timeouts, invalids = _scan_step_statuses(env)
    gr = GameResult(seed=seed, duration_s=dt, timeout_events=timeouts, invalid_events=invalids)
    for i, st in enumerate(final[:num_players]):
        stats = EndStats(reward=float(st.reward), status=str(st.status))
        obs = _unwrap_obs(st)
        if obs is not None:
            stats.ships, stats.planets = _row_stats(obs, i)
        if i == 0:
            gr.p0 = stats
        else:
            gr.p1 = stats
    gr.episode_invalid = any(
        s.status in ("INVALID", "ERROR") for s in final[:num_players]
    )
    return gr


def run_match(
    name: str,
    agent_p0: AgentRef,
    agent_p1: AgentRef,
    seeds: Sequence[int],
    act_timeout: float,
) -> MatchSummary:
    summary = MatchSummary(
        name=name,
        agent_p0=str(agent_p0),
        agent_p1=str(agent_p1),
        seeds=list(seeds),
    )
    for seed in seeds:
        summary.games.append(run_game([agent_p0, agent_p1], seed, act_timeout))
    return summary


def default_seeds(n: int, start: int = 5000) -> List[int]:
    return list(range(start, start + n))


def load_probe_seeds(n: int) -> List[int]:
    if SEEDS_PROBE_V1.is_file():
        data = json.loads(SEEDS_PROBE_V1.read_text(encoding="utf-8"))
        seeds = list(data.get("seeds", []))
        if len(seeds) >= n:
            return seeds[:n]
    if SEEDS_FILE.is_file():
        data = json.loads(SEEDS_FILE.read_text(encoding="utf-8"))
        base = list(data.get("seeds_1v1_vs_random", []))
        base += list(data.get("seeds_1v1_vs_baseline", []))
        if len(base) >= n:
            return base[:n]
        return base + default_seeds(n - len(base), start=5000 + len(base))
    return default_seeds(n)


def load_seeds_probe_v1(n: int | None = None) -> List[int]:
    data = json.loads(SEEDS_PROBE_V1.read_text(encoding="utf-8"))
    seeds = list(data["seeds"])
    return seeds[:n] if n is not None else seeds


def rollout_divergence_probe(
    seeds: Sequence[int],
    act_timeout: float,
    steps_to_sample: Sequence[int] = (0, 10, 25, 50),
) -> dict[str, Any]:
    import importlib.util

    from src.policy.v2_bridge import v2_agent as v2_fn
    from src.search.probe_metrics import aggregate_divergence, choices_at_obs

    def load_agent(path: Path):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        return mod.agent

    from kaggle_environments import make

    hybrid_fn = load_agent(Path(AGENT_HYBRID))
    cfg = {"actTimeout": act_timeout}
    samples: list = []
    for seed in seeds:
        env = make(
            "orbit_wars",
            debug=False,
            configuration={"seed": seed, "actTimeout": act_timeout},
        )
        env.reset(2)
        while not env.done:
            step_num = env.state[0].observation.step
            if step_num in steps_to_sample:
                obs0 = env.state[0].observation
                detail = choices_at_obs(obs0, cfg)
                detail["seed"] = seed
                detail["step"] = step_num
                samples.append(detail)
            env.step([v2_fn(env.state[0].observation, cfg), []])
    agg = aggregate_divergence(samples)
    return {
        "seeds": list(seeds),
        "samples": samples,
        "divergence": agg,
        "differ_count": sum(1 for s in samples if s.get("differ")),
        "sample_count": len(samples),
    }


def regret_probe(
    seeds: Sequence[int],
    act_timeout: float,
    decision_steps: Sequence[int] = (0, 10, 25, 50, 100),
    horizon: int = 15,
) -> dict[str, Any]:
    from kaggle_environments import make
    from src.policy.v2_bridge import v2_agent
    from src.search.probe_metrics import choices_at_obs, measure_regret, summarize_regret

    cfg = {"actTimeout": act_timeout}
    rows: list = []
    for seed in seeds:
        env = make(
            "orbit_wars",
            debug=False,
            configuration={"seed": seed, "actTimeout": act_timeout},
        )
        env.reset(2)
        while not env.done:
            step = int(env.state[0].observation.step)
            if step in decision_steps:
                obs0 = env.state[0].observation
                ch = choices_at_obs(obs0, cfg)
                margin = ch.get("rollout_margin")
                rows.append(
                    measure_regret(
                        seed,
                        step,
                        ch.get("v2_moves") or [],
                        ch.get("hybrid_moves") or [],
                        horizon,
                        act_timeout,
                        v2_agent,
                        action_bucket=ch.get("hybrid_action_bucket", "unknown"),
                        differ=ch.get("differ"),
                        rollout_margin=margin,
                    )
                )
            env.step([v2_agent(env.state[0].observation, cfg), []])

    if not rows:
        return {"n": 0, "horizon": horizon, "rows": []}
    out = summarize_regret(rows)
    out["horizon"] = horizon
    return out


def print_summary(agg: dict[str, Any]) -> None:
    if agg.get("n_seeds", 0) == 0:
        print(f"  {agg.get('name', '?')}: no games")
        return
    print(f"\n=== {agg['name']} ===")
    print(
        f"  P0 ({Path(agg['agent_p0']).name}): winrate={agg['p0_winrate']:.1%}  "
        f"avg_ships={agg['avg_final_ships'][0]:.0f}  avg_planets={agg['avg_planets_owned'][0]:.1f}"
    )
    print(
        f"  P1 ({Path(agg['agent_p1']).name}): winrate={agg['p1_winrate']:.1%}  "
        f"avg_ships={agg['avg_final_ships'][1]:.0f}  avg_planets={agg['avg_planets_owned'][1]:.1f}"
    )
    print(
        f"  completed={agg['completed']}/{agg['n_seeds']}  invalid_episodes={agg['episode_invalid']}  "
        f"timeout_events={agg['total_timeout_events']}  invalid_events={agg['total_invalid_events']}"
    )
    print(f"  draws={agg['draw_rate']:.1%}  avg_game_s={agg['avg_duration_s']:.1f}")
