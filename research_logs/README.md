# Hybrid research logs

Chronological experiment memory — **more important than code diffs** once probe matrix is frozen.

## Format

Each entry: `YYYY-MM-DD_<topic>.md`

Include:

- Hypothesis
- Code / config changes (with frozen versions)
- Probe matrix command + artifact paths
- `regret.by_bucket` and `decision_reversal_rate` (not just winrate)
- Next hypothesis (falsifiable)

## Frozen baselines

| Artifact | Role |
|----------|------|
| `probe_results/seeds_probe_v1.json` | 30 fixed seeds |
| `opponent_model.py` frontier v1 | Rollout opponent (do not extend before matrix baseline) |
| `submission_v2.py` | Production SSOT |

## Key metrics

- **regret** (`avg_ship_diff`, `hybrid_worse_ships_rate`) — primary
- **decision_reversal_rate** — hybrid diverged from v2 and real-env regret &lt; 0 (“confidently wrong”)
- **decision_reversal_given_diverged** — main toxicity metric for confidence gate
- **same_decision_rate** — rollout agrees with v2 wisdom
- **by_bucket** — symptom (`neutral_capture`, …)
- **failure_tag** — mechanism (see `2026-05-26_failure_taxonomy.md`)

## Hypothesis docs

| File | Topic |
|------|--------|
| `2026-05-26_frontier_v1.md` | Frozen opponent + matrix |
| `2026-05-26_confidence_gate_hypothesis.md` | Bucket-conditioned override |
| `2026-05-26_failure_taxonomy.md` | Beyond buckets |
| `2026-05-26_override_precision.md` | Precision/recall dashboard metrics |
