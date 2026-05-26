# Phase B2.6 (hypothesis): Bucket-conditioned Confidence Gate

**Prerequisite:** 30-seed matrix + `decision_delta_hybrid_frontier_v1.jsonl` frozen.  
**See also:** `2026-05-26_failure_taxonomy.md`

## Research question

**When should hybrid stay silent?** Rollout = veto, not replacement planner.

## Primary metrics

| Metric | Target |
|--------|--------|
| `decision_reversal_given_diverged` | ↓ (main) |
| `avg_ship_diff` | ≈ 0 |
| `same_decision_rate` | high OK if reversal low |

Do not optimize divergence ↑. Target: ~90% agree v2; ~10% override; overrides mostly win.

## Gate: per-bucket τ (calibrated from regret)

```python
# Fit τ per bucket: minimal margin with reversal_given_diverged < 35%

TAU_BY_BUCKET = {
    "reinforcement": 0.05,
    "pass": 1.0,
    "aggressive_expansion": 0.15,
    "neutral_capture": 0.25,
    "comet_chase": 0.40,
    "rotating_intercept": 0.35,
}

def should_override(v2_cand, hy_cand, bucket):
    margin = hy_cand.final_score - v2_cand.heuristic_score
    if margin <= TAU_BY_BUCKET.get(bucket, 0.30):
        return False
    return validate_move_for_obs(...)
```

Calibrate from matrix stats + tag `decision_reversal: true` rows (failure taxonomy).

## Implementation order

1. `probe_matrix.py` (30 seeds) — freeze frontier v1.
2. Tag reversals in decision_delta → failure_taxonomy.
3. Fit `TAU_BY_BUCKET`.
4. One PR: gate in `_hybrid_agent`.
5. Re-run matrix only.

## Deferred

MCTS, value net, scheduler, self-play, RL, frontier v2+ until step 5.
