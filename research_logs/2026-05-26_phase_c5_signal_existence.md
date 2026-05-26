# Phase C.5 — Signal Validity Collapse Test

**Prerequisite:** 30-seed `decision_delta_hybrid_frontier_v1.jsonl`

## One command

```bash
PYTHONPATH=. .venv/bin/python scripts/signal_existence_report.py
```

## Three metrics (per bucket + global)

| Metric | Meaning |
|--------|---------|
| Spearman(margin, ship_diff) | EV rankable? |
| sign_accuracy_directional | P(sign pred = sign actual) |
| noise_ratio = Var(error)/Var(actual) | signal vs noise floor |

## Verdict cases

| Case | Condition | Route |
|------|-----------|-------|
| **A** | all bucket ρ ≈ 0 | `gate_veto_only` — hybrid never plans, only veto |
| **B** | few buckets ρ ≥ 0.25 | `bucket_local_gate` — router + local calibrator |
| **C** | global ρ + monotone gain | `value_learning_candidate` — distillation OK |
| insufficient | n < min_support | re-run matrix |

## Do not proceed until answered

- ❌ deeper rollout / MCTS / scheduler
- ❌ wire gate into production
- ✅ only if Case B/C with 30-seed population

## Frozen pipeline

```text
probe_matrix → calibration_report → ev_quality_report → signal_existence_report → decision
```
