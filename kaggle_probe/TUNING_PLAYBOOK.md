# Orbit Wars 调参手册

> **生产参考**：[`submission_v2.py`](submission_v2.py)（仓库内最高记录 814，见 [`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md) §1.2）  
> **实验分支**：[`submission_v6.py`](submission_v6.py)  
> **常量 SSOT**：[`ORBIT_WARS_SSOT.md`](ORBIT_WARS_SSOT.md)（`scripts/dump_ssot.py` 生成）  
> **架构**：[`ORBIT_WARS_ARCHITECTURE.md`](ORBIT_WARS_ARCHITECTURE.md)

## 重要说明

1. **生产常量以 SSOT 为准**（由 v2 生成）；改代码后运行 `scripts/dump_ssot.py`
2. **实验从 v2 复制**；分数 claims 写入 EXPERIMENTS §1.2（含 submission id）
3. **永不直接改 `submission_v2.py`**
4. 本地 A/B 对照 `--old submission_v2.py`；镜像平局不能替代 Kaggle rating

---

## 工作流

```
1. 从下面 P0/P1 选一个参数（核对 ORBIT_WARS_ARCHITECTURE.md §11.2 当前值）
2. cp submission_v6.py submission_ablation_X.py
3. 只改该参数
4. python sweep_params.py --param XXX --values ... --games 30
   或 local_ab.py --new submission_ablation_X.py --old submission_v6.py ...
5. IMPROVED / INCONCLUSIVE / REGRESSED → 记录到「已确认改动」表
6. 攒 1–3 项有效改动 → 合并回 submission_v6.py → Kaggle 提交
7. 更新 ORBIT_WARS_ARCHITECTURE.md §11
```

---

## P0 优先调参（相对 v6 当前值）

### 1. `HOSTILE_REINFORCE_FRACTION`

- **当前值（v6）**：`0.5`（见代码，**不是** 0.25）
- **层级**：L4 / B1
- **为什么调**：控制进攻 need 是否高估敌增援；0.25–0.5 均可 sweep

```bash
python sweep_params.py \
  --param HOSTILE_REINFORCE_FRACTION \
  --values 0.25 0.35 0.45 0.5 \
  --games 30 --ffa 10
```

### 2. `EARLY_TURN_LIMIT`

- **当前值**：`40`

```bash
python sweep_params.py --param EARLY_TURN_LIMIT --values 35 40 45 50 --games 30
```

### 3. `LATE_REMAINING_TURNS`

- **当前值**：`70`

```bash
python sweep_params.py --param LATE_REMAINING_TURNS --values 55 65 70 80 --games 30
```

### 4. `PROACTIVE_DEFENSE_RATIO`

- **当前值**：`0.28`

```bash
python sweep_params.py --param PROACTIVE_DEFENSE_RATIO --values 0.20 0.24 0.28 0.32 --games 30
```

### 5. `HOSTILE_TARGET_VALUE_MULT`

- **当前值**：`2.05`

```bash
python sweep_params.py --param HOSTILE_TARGET_VALUE_MULT --values 1.85 1.95 2.05 2.20 --games 30
```

---

## P1 高价值（4P，需 `--ffa`）

### 6. `FOUR_PLAYER_ROTATING_SEND_RATIO`

- **当前值（v6）**：`0.62`（v2 为 0.55；**不是**手册旧写的 0.55）

```bash
python sweep_params.py --param FOUR_PLAYER_ROTATING_SEND_RATIO \
  --values 0.55 0.62 0.65 --games 20 --ffa 15
```

### 7. `WEAKEST_ENEMY_VALUE_MULT_4P`

- **当前值**：`1.5`

```bash
python sweep_params.py --param WEAKEST_ENEMY_VALUE_MULT_4P \
  --values 1.3 1.5 1.7 1.9 --games 15 --ffa 20
```

### 8. `ELIMINATION_BONUS`

- **当前值**：`55.0`

```bash
python sweep_params.py --param ELIMINATION_BONUS \
  --values 35 45 55 70 --games 15 --ffa 20
```

### 9. v6 已锁定参数（ablation 时作对照，勿与 P0 同时改）

| 参数 | v6 当前值 | v2 原值 | 说明 |
|------|-----------|---------|------|
| `HOSTILE_SWARM_ETA_TOLERANCE` | 2 | 1 | 协同攻击窗口 |
| `MULTI_ENEMY_PROACTIVE_RATIO` | 0.28 | 0.35 | 4P 防守锁兵 |

---

## P2 微调（P0/P1 完成后）

| 参数 | v6 当前值 | 建议 sweep 范围 |
|------|-----------|-----------------|
| `ATTACK_COST_TURN_WEIGHT` | 0.50 | 0.44, 0.50, 0.56 |
| `STATIC_NEUTRAL_VALUE_MULT` | 1.4 | 1.3, 1.4, 1.6 |
| `FOLLOWUP_MIN_SHIPS` | 8 | 6, 8, 12 |
| `REAR_SEND_RATIO_TWO_PLAYER` | 0.62 | 0.55, 0.62, 0.70 |
| `ROUTE_SEARCH_HORIZON` | **60** | 60, 75, 90 |

> `ROUTE_SEARCH_HORIZON` 当前为 **60**（不是 80）。增大同时关注 `SOFT_ACT_DEADLINE` 与超时。

---

## L3 逻辑 ablation（v6 新增，逐项测）

| 代号 | 改法 | 层级 |
|------|------|------|
| v6-1 | 注释/关闭 `COMPETITION_PRESSURE_*` | L3 `target_value` |
| v6-2 | 注释/关闭 `HIGH_PROD_BONUS_*` | L3 `target_value` |
| v6-3 | 注释/关闭 `PROD_BEHIND_NEUTRAL_MULT` | L3 `target_value` |

每次只关一条，对比 `submission_v2.py` 或 v6 自身。

---

## ❌ 高风险（历史回归）

- 同时改多个 `*_VALUE_MULT` + `SIM_HORIZON` + `SOFT_ACT_DEADLINE`（v7 路径）
- `FOUR_SOURCE_SWARM`、B1 双层 margin、`_perf_tight`、`very_late reserve=0`
- 在未保留 A1/A2/B1 的 baseline 上堆参数
- 改 `BOARD` / `SUN_R` / `MAX_SPEED` 等物理常量

---

## 已确认改动（待合并）

| 日期 | 参数/逻辑 | 旧值 | 新值 | 1v1 | 4P | Kaggle ELO | 备注 |
|------|-----------|------|------|-----|-----|------------|------|
| | | | | | | | |

合并后：更新 `submission_v6.py`（或 v2）、运行 `scripts/dump_ssot.py`、填写 [`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md) §1.2 / §6。

---

## 提交检查清单

- [ ] `python -m py_compile submission_v6.py` 无 error
- [ ] `local_ab.py` vs v6 或 v2：0 INVALID，记录 W/L/T
- [ ] 架构文档 §11 当前值已更新
- [ ] 仅当 ELO 确认上涨时，才替换 Kaggle `submission.py`
