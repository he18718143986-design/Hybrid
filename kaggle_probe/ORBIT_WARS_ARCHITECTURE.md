# Orbit Wars Agent 架构文档

> **版本**：2026-05-25  
> **性质**：五层架构 + 函数对照 + ablation 定位（**非**分数/常量/实验记录的 SSOT）

## 相关文档

| 文档 | 内容 |
|------|------|
| **本文档** | 分层架构、函数 ↔ 层级、跨层函数主归属 |
| [`ORBIT_WARS_SSOT.md`](ORBIT_WARS_SSOT.md) | 生产常量（`scripts/dump_ssot.py` 生成） |
| [`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md) | 分数证据、版本族谱、ablation、回归 |
| [`TUNING_PLAYBOOK.md`](TUNING_PLAYBOOK.md) | sweep 命令与流程 |
| [`REGRESSION.md`](REGRESSION.md) | **固定 seed 回归清单 + ablation 门禁** |

**代码参考实现**：`submission_v2.py` 为 **唯一生产基线**（`# 814`，`BASELINE.lock` `710d3ff438a2`；策略见 [`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md) §0）。  
`submission_v6.py` 等为实验分支，不得替代主提交直至 Kaggle 证据超 v2。v2↔v6 结构差异可复现核对：

```bash
diff -u submission_v2.py submission_v6.py | grep -E '^\+|^-' | grep -vE '^\+\+\+|^---' | wc -l
wc -l submission_v2.py submission_v6.py
```

生成 SSOT 时会自动记录上述 diff 行数、文件 SHA256 与 git commit → [`ORBIT_WARS_SSOT.md`](ORBIT_WARS_SSOT.md)「生成元数据」节。

---

## 1. 问题本质

Orbit Wars（[Kaggle 竞赛页](https://www.kaggle.com/competitions/orbit-wars)）是 2/4 人、500 回合、连续空间 RTS；终局分数为己方星球 + 在途舰队总船数；agent rating 按 episode 更新。

| 维度 | 具体表现 |
|------|----------|
| 连续空间几何 | 射线-圆、太阳避让 |
| 动态系统 | 公转、彗星 path |
| 资源增长 | production |
| 多智能体 | 2/4 人、第三方抢战斗 |
| 实时决策 | `actTimeout = 1.0s` |
| 未来推演 | 需 agent 自行 forward sim |

**方法论**：拆成可独立开发/测试的子模块，再组合为决策流水线。

---

## 2. 五层架构

```
L5 战略   阶段 / 编排 / 时间预算
L4 战斗   need、增援、任务解算
L3 经济   目标价值、score、margin
L2 几何   angle、路径、太阳
L1 世界   位置预测、arrival、timeline
```

### 2.1 跨层函数：主归属层

函数可**调用**其他层，但 ablation 时按 **主归属** 改代码，避免「到底改哪层」争议。

| 函数 | 主归属 | 也涉及 | ablation 时… |
|------|--------|--------|--------------|
| `build_modes` | **L5** | L3 输入（`target_value` 读 modes） | 改阶段/ domination 阈值 → L5 |
| `build_policy_state` | **L4** | L5 预算后果 | 改 reserve/attack_budget → L4 |
| `target_value` | **L3** | L5 modes 作乘子 | 改星球值/优先级 → L3 |
| `preferred_send` | **L3** | L4 need 输出 | 改 margin/发送量 → L3 |
| `opening_filter` | **L5** | L3 竞争分类 | 改 opening 激进度 → L5 |
| `plan_moves` | **L5** | L2–L4 全流程 | 改任务顺序/兜底阶段 → L5；改 need → L4 |
| `WorldModel.plan_shot` | **L2** | L1 位置预测 | 改拦截/太阳 → L2 |
| `WorldModel.projected_state` | **L1** | L4 战斗输入 | 改 timeline → L1 |
| `settle_plan` | **L4** | L2 shot、L3 send | 改打赢判定 → L4 |

### 2.2 每回合流水线

```
agent → build_world → WorldModel
     → plan_moves
          build_modes()           # L5
          build_policy_state()    # L4
          mission builders        # L3+L4
          missions.sort()         # L3 score
          mission execution       # L4 settle_plan
          fallback phases         # L5 rear / total war / evac
     → [[src, angle, ships], ...]
```

---

## 3. 架构 ↔ 代码对照

> 锚点：`submission_v2.py`。v6 独有逻辑见 [`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md) §4。

### L1 — 世界建模

| 函数 | 说明 | 改这里若… |
|------|------|-----------|
| `predict_planet_position`, `predict_comet_position`, `predict_target_position` | 位置预测 | 公转/彗星不准 |
| **`fleet_target_planet` (A1)** | 舰队目标推断 | arrival ledger 错 |
| `build_arrival_ledger` | 在途 → (planet, eta) | 到达预判错 |
| `simulate_planet_timeline`, `resolve_arrival_event`, `state_at_timeline` | 时间线 / 战斗规则 | 战斗解算错 |
| `detect_exposed_enemy_planets`, `_compute_weakest_enemy` | 薄弱敌星 / 最弱玩家 | snipe / 4P 集火 |
| `WorldModel.__init__`, `projected_state`, `projected_timeline`, `hold_status`, `reaction_times` | 状态容器与投影 | 全局 forward sim |

常量：`SIM_HORIZON`, `ROTATION_LIMIT` → [`ORBIT_WARS_SSOT.md`](ORBIT_WARS_SSOT.md)

### L2 — 几何

| 函数 | 说明 | 改这里若… |
|------|------|-----------|
| `segment_hits_sun`, `point_to_segment_distance` | 太阳 / 距离 | 舰队莫名消失 |
| `search_safe_intercept`, **`aim_with_prediction`** | 拦截 angle | 打不中移动目标 |
| `WorldModel.plan_shot`, `probe_ship_candidates`, `best_probe_aim` | 射击入口 | ETA/路径系统偏差 |
| `planet_distance`, `nearest_sources_to_target`, `min_legal_reaction_time` | 距离 / 反应时间 | 源选择 |

常量：`SUN_SAFETY`, `ROUTE_SEARCH_HORIZON`, `MAX_SPEED` → SSOT

### L3 — 经济 / 评分

| 函数 | 说明 | 改这里若… |
|------|------|-----------|
| `indirect_features` | 间接财富 | 位置价值 |
| **`target_value`** | 核心目标价值 | 优先级 / 抢点 |
| `reinforce_value`, `preferred_send`, `apply_score_modifiers` | 增援值 / 发送量 / score 修饰 | 过度投入 |
| `is_safe_neutral`, `is_contested_neutral`, `policy_reaction_times` | 中立分类 | 竞争判断 |
| `candidate_time_valid` | 时间窗 | 末期浪费 |

常量：`*_VALUE_MULT`, `*_MARGIN_*`, `ELIMINATION_BONUS` → SSOT

### L4 — 战斗

| 函数 | 说明 | 改这里若… |
|------|------|-----------|
| **`hostile_reinforcement_arrivals` (B1)** | 敌增援 | need 高低 |
| `min_ships_to_own_by`, `min_ships_to_own_at` | 占领 need | 打输/发太多 |
| **`detect_enemy_planet_battles` (A2)** | gang-up 窗口 | 4P 捡漏 |
| `detect_enemy_crashes`, `stacked_enemy_proactive_keep`, `swarm_eta_tolerance` | 坠毁 / 防守 / swarm | 投机 / 锁兵 |
| **`build_policy_state`** | reserve, attack_budget | 可进攻兵力 |
| `settle_plan`, `settle_reinforce_plan` | 方案收敛 | 单任务质量 |
| `build_snipe/rescue/recapture/reinforce/crash/gang_up/elimination_missions` | 任务生成 | 战术种类 |

常量：`HOSTILE_REINFORCE_*`, `HOSTILE_SWARM_*`, `GANG_UP_*` → SSOT

### L5 — 战略

| 函数 | 说明 | 改这里若… |
|------|------|-----------|
| `WorldModel.is_early/opening/late/total_war/...` | 阶段标志 | 阶段切换 |
| **`build_modes`** | domination / finishing | 进攻风格 |
| **`opening_filter`** | opening 过滤 | 过早扩张 |
| **`plan_moves`** | 编排 + 时间预算 + fallback | 整体行为 |
| `agent`, `build_world` | 入口 / 解析 | 超时 / obs |

常量：`EARLY_*`, `TOTAL_WAR_*`, `SOFT_ACT_DEADLINE`, `FOUR_PLAYER_ROTATING_*` → SSOT

---

## 4. 数据流与类型

| 类型 | 用途 |
|------|------|
| `Planet`, `Fleet` | 观测解析 |
| `ShotOption`, `Mission` | 候选动作 / 任务 |
| `WorldModel` | L1 状态 + L2 方法 + L4 投影 |

```
obs → WorldModel(arrivals, base_timeline, keep_needed, exposed, weakest)
    → plan_moves(modes, policy, missions, planned_commitments, moves)
```

---

## 5. Mission 速查

| kind | Builder | 主层 |
|------|---------|------|
| `capture` | `plan_moves` 主循环 | L3+L4 |
| `swarm` | 多源组合 | L3+L4 |
| `snipe` | `build_snipe_mission` | L3+L4 |
| `reinforce` | `build_reinforce_missions` | L3+L4 |
| `rescue` / `recapture` | 对应 builder | L4 (+L5) |
| `crash_exploit` | `build_crash_exploit_missions` | L1+L4 |
| `gang_up` | `build_gang_up_missions` | L1+L4 |
| `elimination` | `build_elimination_missions` | L3+L5 |

---

## 6. Ablation 定位（摘要）

| 症状 | 主查层 / 函数 |
|------|----------------|
| 公转打不中 | L2 `aim_with_prediction` |
| fleet 目标错 | L1 `fleet_target_planet` |
| 不该打的中立星 | L3 `target_value`；L5 `opening_filter` |
| need 不准 | L4 `min_ships_to_own_*`, B1 |
| 4P 不捡漏 | L4 A2 + `build_gang_up_missions` |
| 超时 | L5 `SOFT_ACT_*`, `HEAVY_*` |

流程、禁止事项、分数记录 → [`ORBIT_WARS_EXPERIMENTS.md`](ORBIT_WARS_EXPERIMENTS.md)、[`TUNING_PLAYBOOK.md`](TUNING_PLAYBOOK.md)

---

## 7. 代码结构（submission_v2.py，约 3424 行）

行号为 **2026-05-25** 统计，合并 v6 后可能偏移；以 `grep "^def "` 为准。

| 行号（约） | 内容 |
|------------|------|
| 1–191 | CONFIG |
| 193–455 | L2 几何 + L1 预测 |
| 456–737 | L1 arrival / timeline |
| 738–1264 | `WorldModel` |
| 1266–1699 | L4 检测 + L3/L5 价值 |
| 1701–2539 | L4 mission builders |
| 2541–3395 | L5 `plan_moves` |
| 3397–3425 | `agent` |

v2↔v6 行数与 diff 统计见 SSOT 元数据，或运行：

```bash
wc -l submission_v2.py submission_v6.py
diff -u submission_v2.py submission_v6.py | grep -E '^\+|^-' | grep -v '^\+\+\+\|^---' | wc -l
```

---

*函数表随 `submission_v2.py` 结构更新；常量勿写在本文件，见 SSOT；分数勿写在本文件，见 EXPERIMENTS。*
