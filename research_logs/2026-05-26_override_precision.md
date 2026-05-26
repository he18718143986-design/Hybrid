# Override precision dashboard

**Status:** Implemented in `summarize_override_precision()` — reported in matrix regret output.

## Definitions (paired counterfactual per state)

At each sampled `(seed, step)` we force v2 opening vs hybrid opening, same replay thereafter.

| Metric | Formula | Meaning |
|--------|---------|---------|
| `override_precision` | TP / diverged | When hybrid **changed** v2, how often ships improved |
| `override_recall` | TP / opportunities | When hybrid line is better, did we **actually** diverge |
| `decision_reversal_given_diverged` | FP / diverged | Confident wrong override rate |
| `reversal_severity_mean_ships` | mean \|ship_diff\| on reversals | How bad are bad overrides |
| `margin_outcome_correlation` | corr(rollout_margin, ship_diff) on diverged | Is higher rollout margin predictive |

Where:

- TP = `differ` ∧ `ship_diff > 0`
- FP = `differ` ∧ `ship_diff < 0` (reversal)
- opportunities = all states with `ship_diff > 0`

## Target shape (mature hybrid)

```text
override_precision  high   (~0.6+ after gate)
override_recall     moderate (precision/recall tradeoff after gate)
reversal|diverged    low    (<0.35)
same_decision_rate  high   (~0.9+)
```

## Not the goal

- High `diverged_rate` alone
- High divergence without precision

## Gate calibration uses

1. Matrix `override_precision_dashboard` per bucket (extend later)
2. `decision_reversal: true` in jsonl → failure_taxonomy
3. Fit `TAU_BY_BUCKET` to hit reversal|diverged target
