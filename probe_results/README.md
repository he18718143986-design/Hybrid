# Probe results (Hybrid experiment discipline)

## Frozen artifacts (v1)

| File | Purpose |
|------|---------|
| `seeds_probe_v1.json` | Fixed 30 seeds — do not reorder |
| `matrix_report_v1.json` | Full matrix run summary |
| `regret_report_hybrid_frontier_v1.json` | Frontier cell: h2h + regret + by_bucket |
| `regret_report_hybrid_passive_v1.json` | Passive ablation |
| `decision_delta_hybrid_frontier_v1.jsonl` | Decision delta replay (one JSON per line) |
| `calibration_report_v1.json` | Margin × bucket reliability curves |

## Run matrix

```bash
cd /path/to/Orbit
PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py
```

## Success criteria (before scheduler/MCTS)

- `hybrid_worse_ships_rate` < 55%
- `avg_ship_diff` ≈ 0
- `episode_invalid` = 0
- Compare `by_bucket` — especially `neutral_capture`, `comet_chase`
- `decision_reversal_rate` — hybrid changed v2 but real-env ships lost
- `same_decision_rate` — rollout agrees with v2 (should rise vs passive)

## Frontier v1 frozen

Do not extend `opponent_model.py` until matrix v1 baseline is recorded.
