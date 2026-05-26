# Orbit Wars — Hybrid Decision Research Stack

Kaggle [Orbit Wars](https://www.kaggle.com/) agent with a modular **v2 baseline + rollout hybrid** pipeline, regret probes, and calibration / EV signal diagnostics.

Production planner SSOT: `submission_v2.py` (~814 score baseline). Experimental hybrid lives under `src/` and `submission/hybrid_main.py`.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

PYTHONPATH=. python scripts/run_local.py --agent submission
PYTHONPATH=. python -m pytest tests/ -q
```

## Hybrid probe & calibration (Phase C)

```bash
PYTHONPATH=. python scripts/probe_matrix.py --cell hybrid_frontier
PYTHONPATH=. python scripts/calibration_report.py --tag frontier_v1_baseline
PYTHONPATH=. python scripts/ev_quality_report.py
PYTHONPATH=. python scripts/signal_existence_report.py
```

See `src/README.md` and `research_logs/` for architecture and experiment discipline.

## Layout

| Path | Role |
|------|------|
| `submission_v2.py` | v2 planner (do not replace lightly) |
| `src/` | world model, rollout, hybrid policy, metrics |
| `submission/` | Kaggle entrypoints |
| `scripts/` | probes, calibration, ablations |
| `tests/` | parity and unit tests |
| `probe_results/` | frozen seeds + reports (large jsonl gitignored) |
| `research_logs/` | hypotheses and phase notes |
| `kaggle_probe/` | **local only** (gitignored) — old ablations / replays |

## License

Private research repository unless otherwise noted.

