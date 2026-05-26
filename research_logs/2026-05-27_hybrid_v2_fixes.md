# Hybrid 止血 + v2 调参（2026-05-27）

## 数据更正

- **Smoke (n=10)**：Case A，margin 无方差
- **30-seed (n=141 diverged)**：Case C，ρ≈0.46，gate `calibration_ready=True`
- 勿用 smoke 结论否决 rollout；用 30-seed 报告为准

## 已实施

### Hybrid（阶段一 + 四）

- 默认 **完整 v2_moves**，不再 Top-1 替换整盘
- `merge_override_move`：同 src 替换一支舰队，否则 prepend
- `decide_override(bucket, margin)` 来自 `gate_policy_v1`
- `submission/main.py`：`ORBIT_AGENT_MODE=v2` 默认
- 探针仍用 `submission/hybrid_main.py`（`ORBIT_AGENT_MODE=hybrid`）

### v2（阶段二）

- 开局：`SAFE_OPENING_PROD_THRESHOLD=3`，`ROTATING_OPENING_MAX_TURNS=16`，`OPENING_TURN_LIMIT=60`
- 敌方援军建模：`HOSTILE_REINFORCE_*` 加强
- `preferred_send`：后期缩小 neutral margin
- total_war：每颗我方星球可打一发，去掉过早 `break`

### Rollout（阶段三）

- 叶评估：incoming_threat + fragile_penalty
- `ROLLOUT_DEPTH=12`

## 验证

```bash
PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py --quick --cell hybrid_frontier
PYTHONPATH=. .venv/bin/python scripts/signal_existence_report.py
```

成功标准：h2h invalid↓、regret worse_rate < 50%、episode 仍看 v2 提交分。
