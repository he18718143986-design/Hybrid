# Phase C — Reliability mapping & calibration

**Prerequisite:** `decision_delta_hybrid_frontier_v1.jsonl` from 30-seed matrix.

## C1 — Reliability map

```bash
PYTHONPATH=. .venv/bin/python scripts/calibration_report.py
```

Outputs:

- `probe_results/calibration_report_v1.json`
- Margin → `reversal_rate` curve (diverged states)
- `bucket × margin_bin` table
- Suggested `τ` bin per bucket (target reversal ≤ 35%)

## C2 — Failure taxonomy

Tag `decision_reversal: true` rows in jsonl → `failure_tag` (see `2026-05-26_failure_taxonomy.md`).

## C3 — Margin calibration

Question: does higher `rollout_margin` predict lower reversal?

Read `margin_curve_diverged` in calibration report:

- `support` per bin (low n marked `*`)
- Wilson 95% CI on `reversal_rate`
- `margin_monotonicity.monotone_decreasing` — must be YES for global τ
- `expected_calibration_error` (ECE) — trust vs actual success

## C3b — EV signal validity (before gate)

```bash
PYTHONPATH=. .venv/bin/python scripts/ev_quality_report.py
```

Outputs `ev_quality_report_v1.json`:

- `ranking_global` / `ranking_by_bucket` — Spearman(predicted EV, actual ship_diff)
- `ev_calibration_curve` — predicted bin → mean actual gain
- `ev_monotonicity` — gain rises with margin bin?
- `error_decomposition` — bias, MAE, reversal vs non-reversal bias, noise proxy

**Gate only if `ev_signal_valid=True`** (ρ>0 and monotone gain curve).

```bash
PYTHONPATH=. .venv/bin/python scripts/calibration_report.py --tag frontier_v1
```

Gate logic (after C1 validated):

```python
if bucket in DISABLED_BUCKETS:
    use_v2()
elif margin > TAU_BY_BUCKET[bucket] and reversal_ci_high(bucket, bin) <= 0.35:
    override()
```

## C4 — Bucket-conditioned gate

After baseline imaging:

```bash
PYTHONPATH=. .venv/bin/python scripts/calibration_report.py --tag frontier_v1_baseline
# also writes gate_policy_v1.json + src/policy/gate_policy_v1.py

PYTHONPATH=. .venv/bin/python scripts/calibration_policy_spec.py  # standalone regen
```

Wire `decide_override(bucket, rollout_margin, rng)` in hybrid when `CALIBRATION_READY=True`.

Gate v2 combines three layers:

```text
reliability (G/Y/R) + utility (ev_ship_diff) + calibration (ECE/monotone)
→ G: deterministic override if EV>0
→ Y: stochastic override with p=p_trust
→ R/gray: fallback v2
```

## Phase order (frozen)

```text
Matrix v1 → calibration_report → ev_quality_report → taxonomy → policy compiler → matrix v2
```

Do not add MCTS / scheduler / value net before gate validated.
