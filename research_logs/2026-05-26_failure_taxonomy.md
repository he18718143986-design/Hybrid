# Failure taxonomy (beyond action buckets)

**Status:** Design only — tag `decision_delta_*.jsonl` after 30-seed matrix.  
**Prerequisite:** `regret_report_hybrid_frontier_v1.json` frozen.

## Why bucket alone is insufficient

| bucket | surface | same bucket, different root cause |
|--------|---------|-----------------------------------|
| `neutral_capture` | capture neutral | A: countered → frontier too weak |
| `neutral_capture` | capture neutral | B: production good but horizon too short |
| `comet_chase` | chase comet | C: false reachability (geometry) |
| `comet_chase` | chase comet | D: long-horizon ROI, 6-step rollout blind |

**Bucket = symptom. Taxonomy = mechanism.**

## Proposed failure tags (manual or semi-auto later)

| tag | mechanism | typical fix direction |
|-----|-----------|------------------------|
| `overextended_neutral` | rollout misses post-capture frontier collapse | stronger opponent / longer horizon / veto neutral |
| `delayed_counterattack` | opponent model too passive | frontier punish (done in v1) |
| `comet_false_positive` | intercept/ETA wrong in rollout | comet motion fidelity / hard veto |
| `comet_horizon_blind` | true long-term comet value, short rollout | don't override comet without high τ |
| `rotating_eta_miss` | orbit step error accumulates | geometry fidelity |
| `defensive_overreaction` | frontier punish too aggressive | tune EXPOSED_THRESHOLD / one move cap |
| `confident_wrong_override` | high rollout margin, negative regret | confidence gate (bucket τ) |
| `spurious_divergence` | ships/angle only change, same strategic intent | ignore small divergences |

## Tagging workflow (after matrix)

1. Filter `decision_delta_hybrid_frontier_v1.jsonl` where `decision_reversal == true`.
2. For each row: read `v2_move`, `hybrid_move`, `bucket`, `ship_diff`, candidates.
3. Assign one primary `failure_tag` (table above).
4. Aggregate: `P(regret<0 | tag)` — drives bucket-conditioned τ.

## Bucket-conditioned confidence gate (calibration target)

Per bucket, choose minimal `τ_score` such that:

```text
decision_reversal_given_diverged < 35%
```

(on held-out seeds or cross-validated within matrix)

Sketch:

```python
TAU_BY_BUCKET = {
    "reinforcement": 0.05,
    "pass": 1.0,              # never override
    "aggressive_expansion": 0.15,
    "neutral_capture": 0.25,
    "comet_chase": 0.40,
    "rotating_intercept": 0.35,
    "unknown": 0.30,
}

def should_override(v2_top, hy_top, bucket):
    margin = hy_top.final_score - v2_top.heuristic_score  # or rollout-only margin
    return margin > TAU_BY_BUCKET.get(bucket, 0.30)
```

**Calibrate τ from regret stats**, not intuition.

## Research loop (current)

```text
Hypothesis → one change → probe matrix → regret + reversal
→ decision_delta → failure_tag → freeze → next hypothesis
```

## Primary metric shift

Optimize:

```text
decision_reversal_given_diverged ↓
```

Not:

```text
divergence ↑
```

Target hybrid shape: ~90% agree v2; ~10% override; most overrides win in real env.

## Deferred (still)

MCTS, value net, scheduler, self-play, RL, frontier v2+ until matrix + taxonomy on reversals.
