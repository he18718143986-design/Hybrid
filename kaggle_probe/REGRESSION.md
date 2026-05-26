# Orbit Wars 回归测试清单

> **流程**：文档 → 固定基线 → 本清单 → 单变量 ablation → 小流量验证 → Kaggle  
> **架构**：[`ORBIT_WARS_ARCHITECTURE.md`](ORBIT_WARS_ARCHITECTURE.md)  
> **实验记录**：[`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md)

**当前基线（2026-05-25）**：`submission_v2.py` · `BASELINE.lock` `710d3ff438a2` · 最近全量回归 **18/18 PASS** → `regression/results/latest.json`

---

## 0. 锁定基线（每次大改前）

```bash
# 勿直接改 submission_v2.py；实验请复制为 submission_ablation_*.py
.venv/bin/python scripts/lock_baseline.py
.venv/bin/python scripts/dump_ssot.py
```

确认 `BASELINE.lock` 中 SHA256 与 `submission_v2.py` 一致。

---

## 1. 一键回归（固定 seed）

```bash
.venv/bin/python regression_test.py --layers-only          # 快：L1–L4 单元检查 (~30s)
.venv/bin/python regression_test.py --kaggle-timeout       # 含固定 seed 对局 + 1s 预算
.venv/bin/python regression_test.py --candidate submission_ablation_x.py --baseline submission_v2.py
```

### 1b. 自动化 ablation（推荐）

实验配方：`regression/ablations.json`（**新实验只加一条 JSON**）

```bash
.venv/bin/python scripts/run_ablation.py --list
.venv/bin/python scripts/run_ablation.py --id v6p3                    # 生成 + 回归 + local_ab
.venv/bin/python scripts/run_ablation.py --id v6p3 --generate-only    # 仅生成 .py
.venv/bin/python scripts/run_ablation.py --id v6p3 --skip-ab          # 跳过 local_ab
.venv/bin/python scripts/run_ablation.py --all --generate-only        # 批量生成全部 7 项
.venv/bin/python scripts/run_ablation.py --id v6p3 --pair 10 --ffa 10 --kaggle-timeout
```

结果：`regression/results/ablation_<id>_<timestamp>.json` + `regression/results/latest_ablation.json`

### 1c. Kaggle 探针打包（提交前门禁）

```bash
.venv/bin/python scripts/prepare_kaggle_probe.py --id v6p4
# → kaggle_probe/v6p4/submission.py + PROBE_README.md（含 kaggle submit 命令）
```

一次只探一个 id；回归 **16/16 PASS** 后才应提交。判据见 `kaggle_probe/<id>/PROBE_README.md`。4P 批次下一发：**v6p7**（`GANG_UP_ETA_WINDOW` 4→5）。

```bash
.venv/bin/python scripts/prepare_kaggle_probe.py --id v6p7
```

固定 seed 配置：`regression/seeds.json`（**勿改已有 seed 顺序**，新 seed 追加在末尾）。

---

## 2. 分层检查项（与脚本对应）

| 层 | 检查项 | 脚本中的 test | 失败时改 |
|----|--------|---------------|----------|
| L4 | 多方到达战斗结算 | `combat: multi-attacker tie-break` | `resolve_arrival_event` |
| L2 | 太阳碰撞几何 | `geometry: segment_hits_sun` | `segment_hits_sun` |
| L1 | 公转位置预测 | `world: predict_planet_position rotates` | `predict_planet_position` |
| L1 | 舰队目标推断 A1 | `world: fleet_target_planet static` | `fleet_target_planet` |
| L2 | 静态目标 plan_shot 安全 | `geometry: plan_shot safe on static targets` | `aim_with_prediction` / `plan_shot` |
| L1 | 到达账本 | `world: arrival ledger built` | `build_arrival_ledger` |
| L4 | Gang-up 检测 A2 | `combat: detect_enemy_planet_battles` | A2 + `projected_state` |
| L4 | 敌增援 B1 | `combat: hostile_reinforcement_arrivals` | B1 |
| L5 | 固定 seed 整局 | `games:*` | `plan_moves` / 超时 |

---

## 3. Ablation 顺序（一次只改一项）

1. **L1** 目标预测 / arrival ledger  
2. **L2** 几何与安全路径  
3. **L4** 战斗 / 增援 need  
4. **L3** `target_value` / margin  
5. **L5** opening / total war / 时间预算  

每项流程：

```bash
cp submission_v2.py submission_ablation_<name>.py
# 只改一层的一项
.venv/bin/python regression_test.py --candidate submission_ablation_<name>.py --layers-only
.venv/bin/python local_ab.py --new submission_ablation_<name>.py --old submission_v2.py --pair-only 10 --kaggle-timeout
# 通过后再 Kaggle；记录到 ORBIT_WARS_EXPERIMENTS.md
```

---

## 4. 优先验证（失败率最低 / 收益最明显）

- [x] 公转拦截（L1/L2）— 回归 smoke 已覆盖  
- [x] 太阳碰撞（L2）— `segment_hits_sun` + 静态 `plan_shot`  
- [x] arrival ledger / A1（L1）  
- [x] 敌增援 B1（L4）  
- [x] 4P gang-up A2（L4）  

**下一阶段 ablation**（见 EXPERIMENTS §4）：L4 `HOSTILE_SWARM_ETA_TOLERANCE`、L4 `MULTI_ENEMY_PROACTIVE_RATIO`，再 L3 三条。

---

## 5. 提交前门禁

- [x] `regression_test.py` 全绿（2026-05-25：`--kaggle-timeout` 18/18）  
- [ ] `scripts/dump_ssot.py` 已跑，SSOT 元数据新鲜（改常量后必跑）  
- [ ] EXPERIMENTS §1.2 有 submission id（若已上 Kaggle）  
- [x] 仅 **一个** 生产主版本（v2）；其它均为实验分支  

---

## 6. 工程化流水线（摘要）

```
锁定 v2 (BASELINE.lock)
    → regression_test.py (固定 seed + 分层单测)
    → 单变量 ablation (local_ab vs v2)
    → 小批量 Kaggle 提交
    → 更新 EXPERIMENTS + SSOT
```
