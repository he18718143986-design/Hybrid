# Orbit Hybrid `src/`

Modular layout for rule + search + learning. **World/combat SSOT** must stay aligned with `submission_v2.py` until extracted and covered by `tests/`.

## Extraction order (from `submission_v2.py`)

**Phase A (done in tree):**

- [x] `world/combat.py` — `resolve_arrival_event`
- [x] `world/timeline.py` — `simulate_planet_timeline`, `build_arrival_ledger`
- [x] `world/fleet.py` — `fleet_target_planet` (A1)
- [x] `world/need.py` — `min_ships_to_own_*` (B1 hostile path still v2 bridge)
- [x] `world/context.py` — `build_world_context`
- [x] `world/geometry.py`, `prediction.py`, `comet.py`

**Still on v2 bridge:** `plan_shot`, `settle_plan`, `hostile_reinforcement`, full `plan_moves`

**Phase B (started):**

- [x] `world/step.py` — `simulate_one_turn`, `StepState`, `passive_enemy_moves`
- [x] `search/rollout.py` — shallow rollout + leaf evaluation
- [x] `search/evaluator.py` — heuristic + rollout + model weights
- [x] `policy/hybrid_agent.py` — `ORBIT_AGENT_MODE=hybrid` (Top-1 rollout pick)
- [x] B2: orbit + comet `path_index` in `simulate_one_turn`
- [x] B2.5: `search/opponent_model.py` frontier reinforce/punish (`ORBIT_ROLLOUT_OPPONENT=frontier`)
- [x] legality gate + v2 fallback in hybrid
- [x] `scripts/probe_hybrid.py --decision-trace --regret`
- [ ] continuous fleet positions / full env turn order

1. ~~`world/constants.py`, `geometry.py`, `orbit.py`, `comet.py`~~
2. ~~`world/combat.py`, `timeline.py`~~ → `world/state.py` uses native `ctx`
3. `candidate/scoring.py`, `missions.py`, `generate.py`
4. `policy/scheduler.py` (`plan_moves`)
5. `search/rollout.py` (new)
6. `model/*` (new)

## Local run

```bash
cd /path/to/Orbit
PYTHONPATH=. .venv/bin/python scripts/run_local.py --agent submission
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

## Hybrid probe (Phase B1 discipline)

**Probe matrix v1** (frozen seeds + passive/frontier ablation):

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py          # 30 seeds, all cells
PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py --quick    # 5 seeds smoke
PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py --cell hybrid_frontier
```

Outputs: `probe_results/matrix_report_v1.json`, `regret_report_hybrid_frontier_v1.json`, `decision_delta_hybrid_frontier_v1.jsonl`.

Legacy single probe:

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_hybrid.py --seeds 30 --regret --decision-trace
```

Rollout opponent: `ORBIT_ROLLOUT_OPPONENT=frontier` (default) or `passive` for ablation.

Research logs: `research_logs/` (start with `2026-05-26_frontier_v1.md`).

Key regret metrics: `by_bucket`, `decision_reversal_rate`, `override_precision_dashboard`, `same_decision_rate`.

Phase C (after matrix):

```bash
PYTHONPATH=. .venv/bin/python scripts/calibration_report.py --tag frontier_v1_baseline
PYTHONPATH=. .venv/bin/python scripts/ev_quality_report.py   # EV rankable? before gate
PYTHONPATH=. .venv/bin/python scripts/signal_existence_report.py  # Phase C.5 verdict
PYTHONPATH=. .venv/bin/python scripts/calibration_policy_spec.py
```

Outputs: `calibration_report_v1.json`, `ev_quality_report_v1.json`, `gate_policy_v1.json`.

Match matrix: `v2_vs_random`, `hybrid_vs_random`, `hybrid_vs_v2` (+ swap). Results → `probe_results/latest_probe.json`.
Use `submission/hybrid_main.py` for hybrid (separate from v2 env var).

## Kaggle submit

```bash
tar -czf hybrid.tar.gz submission/main.py src/
kaggle competitions submit orbit-wars -f hybrid.tar.gz -m "hybrid scaffold"
```
