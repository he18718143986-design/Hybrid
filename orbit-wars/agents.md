# Orbit Wars: Getting Started

This guide walks you through building an agent, testing it locally, and submitting it to the Orbit Wars competition on Kaggle.

## Game Overview

Orbit Wars is a real-time strategy game on a 100x100 board with a sun at the center. Players conquer planets by sending fleets of ships between them.

- **Planets** produce ships each turn (proportional to their radius)
- **Inner planets** rotate around the central sun; outer planets are static
- **Fleets** fly in straight lines at a given angle from their source planet
- **Fleet speed** scales with fleet size (1 ship = 1/turn, larger fleets up to 6/turn)
- **Combat**: arriving fleet ships are subtracted from the planet's garrison. If the garrison drops below 0, ownership flips
- **Sun**: fleets that hit the sun are destroyed
- **Comets**: temporary planets that fly through the board on elliptical paths
- **Win condition**: highest ship count (planets + fleets) when time runs out, or last player standing

See [README.md](README.md) for full rules and configuration defaults.

## Your Agent

Your agent is a function that receives an observation and returns a list of moves.

**Observation fields:**
- `player` — your player ID (0-3)
- `planets` — list of `[id, owner, x, y, radius, ships, production]` (owner -1 = neutral)
- `fleets` — list of `[id, owner, x, y, angle, from_planet_id, ships]`
- `angular_velocity` — rotation speed of inner planets (radians/turn)

**Action format:**
Each move is `[from_planet_id, angle_in_radians, num_ships]`.

Also useful: `initial_planets`, `comets` (with `paths` / `path_index`), `comet_planet_ids`, `step`, `remainingOverageTime`. See [README.md](README.md) for full combat and turn order (multi-attacker rules are richer than a simple garrison subtraction).

**Example — Nearest Planet Sniper (minimal baseline):**

```python
import math
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

def agent(obs):
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        # Find nearest planet we don't own
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))

        # Send exactly enough ships to capture it
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves
```

### Modular agent skeleton (engineering-ready)

The skeleton below is still a **teaching baseline** (not a leaderboard agent). It encodes four practices that matter on Kaggle:

1. **Derive orbit data from `initial_planets`** — `Planet` has no `orbital_radius` field; compute `r` and `is_orbiting` once per match.
2. **Invalidate module-level caches per match** — use a fingerprint of `initial_planets` + `angular_velocity` + comet ids so a new episode in the same process does not reuse stale geometry.
3. **Predict comets from `obs["comets"]` paths** — do not apply planet rotation logic to comets.
4. **Soft time budget from `config["actTimeout"]`** — use `time.perf_counter()` and stop heavy work before the hard cutoff; do not plan on `remainingOverageTime`.

Orbiting planets use **current observation angle + `angular_velocity * turn_offset`** (robust if `obs["step"]` is missing). In-flight fleets and combat timelines are **not** cached across turns — rebuild those every step when you add real tactics.

```python
import math
import time
from enum import Enum

from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

CENTER_X = 50.0
CENTER_Y = 50.0
ROTATION_LIMIT = 50.0
TOTAL_STEPS = 500

# Per-match cache (invalidated when fingerprint changes)
_MATCH_FP = None
_ORBITAL_METADATA = None  # planet_id -> dict


def _read(obs, key, default=None):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _game_fingerprint(obs):
    init = tuple(tuple(p) for p in (_read(obs, "initial_planets", []) or []))
    comet_ids = tuple(sorted(_read(obs, "comet_planet_ids", []) or []))
    return (init, float(_read(obs, "angular_velocity", 0.0) or 0.0), comet_ids)


def _ensure_orbital_metadata(obs):
    global _MATCH_FP, _ORBITAL_METADATA
    fp = _game_fingerprint(obs)
    if _MATCH_FP == fp and _ORBITAL_METADATA is not None:
        return _ORBITAL_METADATA

    _MATCH_FP = fp
    metadata = {}
    initial_by_id = {p.id: p for p in [Planet(*row) for row in (_read(obs, "initial_planets", []) or [])]}
    comet_ids = set(fp[2])

    for planet_id, init_p in initial_by_id.items():
        if planet_id in comet_ids:
            continue
        dx = init_p.x - CENTER_X
        dy = init_p.y - CENTER_Y
        r = math.hypot(dx, dy)
        is_orbiting = (r + init_p.radius) < ROTATION_LIMIT
        if is_orbiting:
            metadata[planet_id] = {"orbital_radius": r, "is_orbiting": True}
        else:
            metadata[planet_id] = {"is_orbiting": False}

    _ORBITAL_METADATA = metadata
    return metadata


def predict_comet_position(planet_id, comets, turn_offset):
    turn_offset = int(turn_offset)
    for group in comets or []:
        pids = group.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = group.get("paths", [])
        path_index = int(group.get("path_index", 0))
        if idx >= len(paths):
            return None
        path = paths[idx]
        future_idx = path_index + turn_offset
        if 0 <= future_idx < len(path):
            return float(path[future_idx][0]), float(path[future_idx][1])
    return None


def predict_planet_position(planet, orbital_meta, angular_velocity, turn_offset):
    if not orbital_meta.get("is_orbiting"):
        return planet.x, planet.y
    r = orbital_meta["orbital_radius"]
    cur_ang = math.atan2(planet.y - CENTER_Y, planet.x - CENTER_X)
    new_ang = cur_ang + float(angular_velocity) * int(turn_offset)
    return CENTER_X + r * math.cos(new_ang), CENTER_Y + r * math.sin(new_ang)


def predict_position(planet, comet_ids, comets, orbital_metadata, angular_velocity, turn_offset):
    if planet.id in comet_ids:
        pos = predict_comet_position(planet.id, comets, turn_offset)
        if pos is not None:
            return pos
        return planet.x, planet.y
    meta = orbital_metadata.get(planet.id, {"is_orbiting": False})
    return predict_planet_position(planet, meta, angular_velocity, turn_offset)


class ComputeTier(Enum):
    LIGHT = 1
    MEDIUM = 2
    HEAVY = 3


class TimeBudget:
    """Soft deadline inside actTimeout; heavy work checks expired() in loops."""

    def __init__(self, config=None):
        config = config or {}
        act_timeout = float(config.get("actTimeout", 1.0))
        self._start = time.perf_counter()
        self.deadline = self._start + min(0.82, max(0.55, act_timeout * 0.82))
        self.tier = ComputeTier.MEDIUM

    def remaining(self):
        return max(0.0, self.deadline - time.perf_counter())

    def expired(self):
        return time.perf_counter() >= self.deadline

    def refresh_tier(self):
        if self.remaining() > 0.5:
            self.tier = ComputeTier.HEAVY
        elif self.remaining() > 0.2:
            self.tier = ComputeTier.MEDIUM
        else:
            self.tier = ComputeTier.LIGHT


class GameState:
    def __init__(self, obs):
        self.player = int(_read(obs, "player", 0))
        self.step = int(_read(obs, "step", 0) or 0)
        self.angular_velocity = float(_read(obs, "angular_velocity", 0.0) or 0.0)
        self.comets = _read(obs, "comets", []) or []
        self.comet_ids = set(_read(obs, "comet_planet_ids", []) or [])
        self.planets = [Planet(*row) for row in (_read(obs, "planets", []) or [])]
        self.fleets = [Fleet(*row) for row in (_read(obs, "fleets", []) or [])]
        self.planet_by_id = {p.id: p for p in self.planets}
        self.orbital_metadata = _ensure_orbital_metadata(obs)

    def predict_position(self, planet_id, turn_offset=0):
        planet = self.planet_by_id[planet_id]
        return predict_position(
            planet,
            self.comet_ids,
            self.comets,
            self.orbital_metadata,
            self.angular_velocity,
            turn_offset,
        )


class TacticalExecutor:
    def __init__(self, state, budget):
        self.state = state
        self.budget = budget

    def compute_intercept(self, src, target, ships=1):
        """MEDIUM: iterative aim; LIGHT fallback = straight atan2 at current position."""
        if self.budget.tier == ComputeTier.LIGHT or self.budget.expired():
            return math.atan2(target.y - src.y, target.x - src.x)

        tx, ty = target.x, target.y
        for _ in range(5):
            if self.budget.expired():
                break
            pos = self.state.predict_position(target.id, turn_offset=0)
            tx, ty = pos
            angle = math.atan2(ty - src.y, tx - src.x)
            dist = math.hypot(tx - src.x, ty - src.y) - (src.radius + target.radius)
            dist = max(0.0, dist)
            turns = max(1, int(math.ceil(dist / max(1.0, fleet_speed(ships)))))
            future = self.state.predict_position(target.id, turn_offset=turns)
            if abs(future[0] - tx) < 0.3 and abs(future[1] - ty) < 0.3:
                return angle
            tx, ty = future
        return math.atan2(ty - src.y, tx - src.x)


def fleet_speed(ships):
    if ships <= 1:
        return 1.0
    ratio = max(0.0, min(1.0, math.log(ships) / math.log(1000.0)))
    return 1.0 + (6.0 - 1.0) * (ratio ** 1.5)


class StrategicPlanner:
    def select_targets(self, state, max_targets=5):
        my = [p for p in state.planets if p.owner == state.player]
        if not my:
            return []
        candidates = [p for p in state.planets if p.owner != state.player]
        candidates.sort(key=lambda t: (-t.production, math.hypot(my[0].x - t.x, my[0].y - t.y)))
        return candidates[:max_targets]


class Scheduler:
    def __init__(self, state, planner, executor, budget):
        self.state = state
        self.planner = planner
        self.executor = executor
        self.budget = budget

    def generate_orders(self):
        moves = []
        spent = {}

        for target in self.planner.select_targets(self.state):
            if self.budget.expired():
                break
            sources = [p for p in self.state.planets if p.owner == self.state.player]
            if not sources:
                break
            src = min(sources, key=lambda s: math.hypot(s.x - target.x, s.y - target.y))
            left = src.ships - spent.get(src.id, 0)
            need = int(target.ships) + 1
            if left < need:
                continue
            angle = self.executor.compute_intercept(src, target, ships=need)
            moves.append([src.id, float(angle), need])
            spent[src.id] = spent.get(src.id, 0) + need

        return moves


def agent(obs, config=None):
    budget = TimeBudget(config)
    state = GameState(obs)
    budget.refresh_tier()

    if not [p for p in state.planets if p.owner == state.player]:
        return []

    planner = StrategicPlanner()
    executor = TacticalExecutor(state, budget)
    scheduler = Scheduler(state, planner, executor, budget)
    return scheduler.generate_orders()
```

**Compute tiers (when you grow beyond this skeleton):**

| Function | Tier | On `budget.expired()` |
|----------|------|-------------------------|
| `predict_*` / parsing | LIGHT | Always run |
| `compute_intercept` | MEDIUM | `atan2` to current position |
| `simulate_combat` (future) | MEDIUM–HEAVY | Fewer arrivals / shorter horizon |
| `plan_attack` / mission loop | HEAVY | Return moves collected so far |

Do not rely on `remainingOverageTime` as part of your normal budget.

## Test Locally

Install the environment from PyPI (Orbit Wars requires version 1.28.0 or later):

```bash
pip install "kaggle-environments>=1.28.0"
```

Run a game from Python or a notebook:

```python
from kaggle_environments import make

env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["main.py", "random"])

# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Render in a notebook
env.render(mode="ipython", width=800, height=600)
```

## Set Up the Kaggle CLI

Install the CLI:

```bash
pip install kaggle
```

You'll need a Kaggle account — sign up at https://www.kaggle.com if you don't have one. Then download your API credentials at https://www.kaggle.com/settings/api by clicking **"Generate New Token"** under the "API" section.

**Recommended: API token file.** Save the token string to `~/.kaggle/access_token`:

```bash
mkdir -p ~/.kaggle
# Paste the token from the Kaggle settings UI into this file
nano ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

Alternative auth methods:
- **OAuth (browser flow):** `kaggle auth login`
- **Environment variable:** `export KAGGLE_API_TOKEN=xxxxxxxxxxxxxx`

Verify the CLI is wired up:

```bash
kaggle competitions list -s "orbit wars"
```

## Find the Competition

```bash
kaggle competitions list -s "orbit wars"
kaggle competitions pages orbit-wars
kaggle competitions pages orbit-wars --content
```

## Accept the Competition Rules

Before submitting, you **must** accept the rules on the Kaggle website. Navigate to `https://www.kaggle.com/competitions/orbit-wars` and click **"Join Competition"**.

Verify you've joined:

```bash
kaggle competitions list --group entered
```

## Download Competition Data

```bash
kaggle competitions download orbit-wars -p orbit-wars-data
```

## Submit Your Agent

Your submission must have a `main.py` at the root with an `agent` function.

**Single file agent:**

```bash
kaggle competitions submit orbit-wars -f main.py -m "Nearest planet sniper v1"
```

**Multi-file agent** — bundle into a tar.gz with `main.py` at the root:

```bash
tar -czf submission.tar.gz main.py helper.py model_weights.pkl
kaggle competitions submit orbit-wars -f submission.tar.gz -m "Multi-file agent v1"
```

**Notebook submission:**

```bash
kaggle competitions submit orbit-wars -k YOUR_USERNAME/orbit-wars-agent -f submission.tar.gz -v 1 -m "Notebook agent v1"
```

## Monitor Your Submission

Check submission status:

```bash
kaggle competitions submissions orbit-wars
```

Note the submission ID from the output — you'll need it for episodes.

## List Episodes

Once your submission has played some games:

```bash
kaggle competitions episodes <SUBMISSION_ID>
```

CSV output for scripting:

```bash
kaggle competitions episodes <SUBMISSION_ID> -v
```

## Download Replays and Logs

Download the replay JSON for an episode (for visualization or analysis):

```bash
kaggle competitions replay <EPISODE_ID>
kaggle competitions replay <EPISODE_ID> -p ./replays
```

Download agent logs to debug your agent's behavior:

```bash
# Logs for the first agent (index 0)
kaggle competitions logs <EPISODE_ID> 0

# Logs for the second agent (index 1)
kaggle competitions logs <EPISODE_ID> 1 -p ./logs
```

## Check the Leaderboard

```bash
kaggle competitions leaderboard orbit-wars -s
```

## Typical Workflow

```bash
# Test locally
python -c "
from kaggle_environments import make
env = make('orbit_wars', debug=True)
env.run(['main.py', 'random'])
print([(i, s.reward) for i, s in enumerate(env.steps[-1])])
"

# Submit
kaggle competitions submit orbit-wars -f main.py -m "v1"

# Check status
kaggle competitions submissions orbit-wars

# Review episodes
kaggle competitions episodes <SUBMISSION_ID>

# Download replay and logs
kaggle competitions replay <EPISODE_ID>
kaggle competitions logs <EPISODE_ID> 0

# Check leaderboard
kaggle competitions leaderboard orbit-wars -s
```
