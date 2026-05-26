# Orbit Wars 实验记录

> **性质**：版本族谱、Kaggle 分数锚点、ablation 结果、回归教训。  
> **架构**：见 [`ORBIT_WARS_ARCHITECTURE.md`](ORBIT_WARS_ARCHITECTURE.md)  
> **常量**：见 [`ORBIT_WARS_SSOT.md`](ORBIT_WARS_SSOT.md)（机器生成）  
> **调参流程**：见 [`TUNING_PLAYBOOK.md`](TUNING_PLAYBOOK.md)  
> **回归门禁**：见 [`REGRESSION.md`](REGRESSION.md)

---

## 0. 生产策略（2026-05-25 起生效）

| # | 规则 |
|---|------|
| 1 | **`submission_v2.py` 为唯一生产基线（SSOT）**；常量/文档以 v2 为准 |
| 2 | 所有实验版本必须从 v2 **复制分支**（`submission_ablation_*.py`），禁止在原文件上混改 |
| 3 | 遵循 **单变量 ablation**（一次只改 L1–L5 中一项） |
| 4 | **未超过 v2 记录分数前，不替换 Kaggle 主提交** |
| 5 | 回归失败或 Kaggle 回归的实验 **仅记入本文档**，不合并、不污染基线 |

**基线锁定**：`BASELINE.lock` → SHA256 short `710d3ff438a2`（814，3424 行）  
**回归产物**：`regression/results/latest.json`（2026-05-25：**18/18 PASS，0 INVALID**）

**当前阶段**：L3/L4 小步 ablation → Kaggle 小流量验证 → 固定 seed replay 分析。  
**远期**：稳定超 v2 后再考虑 RL / Monte Carlo / 更深 forward sim。

---

## 1. 分数与证据锚点

### 1.1 表述规范

- 写 **「截至 YYYY-MM-DD，仓库内记录的最高 Kaggle rating estimate」**，不写「绝对最高分」。
- Kaggle rating 按 episode 更新，早期样本少时波动大；需结合 **submission 列表 + 对局数** 一起看。
- 本表 ELO/rating 来自：**代码文件注释**、**本地实验笔记**、**Kaggle CLI**（需人工补全 submission id）。

### 1.2 版本分数记录（待补全 submission id）

| 文件 | 记录分数 | 证据来源 | Submission ID | 记录日期 | 备注 |
|------|----------|----------|---------------|----------|------|
| `submission_baseline.py` | ~652 | 文件注释 / 历史笔记 | *待填* | *待填* | 无 A1/A2/B1 |
| **`submission_v2.py`** | **814** | 文件第 2 行 `# 814` | *待填* | 2026-05-25 | **生产基线**；`BASELINE.lock` `710d3ff438a2`；回归 18/18 |
| `submission_ablation_v6p3.py` | **786** | Kaggle 探针 | *待填* | 2026-05-26 | L4 单变量；**低于 v2，勿作主提交** |
| `submission_ablation_v6p4.py` | **804**（Kaggle 803.6） | ref **`53039641`** · Notebook [orbit-wars0526](https://www.kaggle.com/code/tinahe1995/orbit-wars0526) V3 · episode **77750881** | 53039641 | 2026-05-26 | `GANG_UP_VALUE_MULT` 1.55；**REGRESSED** vs v2 814 |
| `submission_ablation_v6p5.py` | **~682**（681.9） | ref **53039881** · [orbit-wars0526](https://www.kaggle.com/code/tinahe1995/orbit-wars0526?scriptVersionId=322191136) V4 | 53039881 | 2026-05-26 | `WEAKEST_ENEMY_VALUE_MULT_4P` 1.7；**REGRESSED** |
| `submission_ablation_v6p6.py` | **~673**（672.8） | ref **53040527** · [orbit-wars0526](https://www.kaggle.com/code/tinahe1995/orbit-wars0526?scriptVersionId=322195466) V5 | 53040527 | 2026-05-26 | `FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT` 0.95；**REGRESSED** |
| `submission_ablation_v6p7.py` | 未提交 | 探针包就绪 | — | 2026-05-26 | `GANG_UP_ETA_WINDOW` 4→5；`kaggle_probe/v6p7/` |
| `submission_v3.py` | 772 | 历史笔记 | *待填* | *待填* | 回归 |
| `submission_v4.py` | 750 | 历史笔记 | *待填* | *待填* | 回归 |
| `submission_v5.py` | 783 | 历史笔记 | *待填* | *待填* | v5-safe 仍 < v2 |
| `submission_v6.py` | 未提交 / 待测 | — | — | — | v2 + 实验增量 |
| `submission_v7.py` | 未提交 / 待测 | — | — | — | 不推荐 |

### 1.3 本地回放资产（非 rating，仅供调试）

| 目录 | Episode ID | 说明 |
|------|------------|------|
| `v2/` | 77598585 | v2 时代下载的回放 JSON |
| `v3/` | 77603079 | v3 时代回放 |
| `v4/` | 77621576 | v4 时代回放 |

补全 Kaggle 证据示例：

```bash
kaggle competitions submissions orbit-wars -v   # 第一列 ref = 提交 ID
kaggle competitions episodes 53039641 -v        # 用 ref 拉该次提交的对局列表
kaggle competitions replay 77750881 -p ./replays
```

**ref 与 Notebook**：**ref** = 每次「Submit to Competition」的 ID（CLI 第一列）。**scriptVersionId** = Notebook 某一版代码（URL 参数）。同一 notebook [orbit-wars0526](https://www.kaggle.com/code/tinahe1995/orbit-wars0526) 三次探针对应：

| 探针 | ref | Notebook 版本 | scriptVersionId（约） | 公开分 |
|------|-----|---------------|---------------------|--------|
| v6p4 | 53039641 | V3 | 322189452（你提供的 v6p4 页） | 804 |
| v6p5 | 53039881 | V4 | 322191136 | 682 |
| v6p6 | 53040527 | V5 | 322195466 | 673 |

---

## 2. 版本族谱

| 文件 | 相对 v2 | 架构角色 |
|------|---------|----------|
| `orbit-wars/main.py` | — | 官方 starter |
| `submission_baseline.py` | 缺 A1/A2/B1 | 反面参考 |
| **`submission_v2.py`** | 基线 | 生产参考实现 |
| `submission_v3/v4/v5.py` | 回归实验 | 勿作生产参考 |
| `submission_v6.py` | +3 逻辑 + 3 参数 | 实验 fork |
| `submission_v7.py` | +大量参数 | 反例 |

**提交策略**：见 §0。Kaggle 主提交保持 v2，直至 §1.2 出现更高分数且回归/证据齐全。

---

## 3. 已验证能力（v2，相对 baseline）

| 代号 | 主层 | 函数 | 相对 baseline ~652 的假设作用 |
|------|------|------|-------------------------------|
| **A1** | L1 | `fleet_target_planet` | 公转感知舰队目标 |
| **A2** | L4 | `detect_enemy_planet_battles` + `projected_state` | 4P gang-up 窗口 |
| **B1** | L4 | `hostile_reinforcement_arrivals` + `model_hostile_reinforce=True` | 进攻 need 含敌增援 |

*「已验证」= 与 v2 分数提升同期引入；严格因果需逐项 ablation + Kaggle。*

---

## 4. 待验证增量（下一阶段：L3/L4 优先）

按 **单变量** 顺序从 v2 分支；每项：`regression_test.py --layers-only` → `local_ab` → 小流量 Kaggle → 更新本表。

| 优先级 | 代号 | 主层 | 改动 | 回归 | Kaggle |
|--------|------|------|------|------|--------|
| 1 | v6-p1 | L4 | `HOSTILE_SWARM_ETA_TOLERANCE` 1→2 | ✅ layers 16/16 | ❌ 不上线 |
| 2 | v6-p3 | L4 | `MULTI_ENEMY_PROACTIVE_RATIO` 0.35→0.28 | ✅ layers 16/16 | ❌ Kaggle **786**（< v2 814） |
| 3 | v6-1 | L3 | `COMPETITION_PRESSURE_*` | ✅ layers 16/16 | ⏳ 本地 INCONCLUSIVE |
| 4 | v6-2 | L3 | `HIGH_PROD_BONUS_*` | ✅ layers 16/16 | ❌ 不上线 |
| 5 | v6-3 | L3 | `PROD_BEHIND_NEUTRAL_MULT` + `is_prod_behind` | ✅ layers 16/16 | ❌ 不上线 |
| — | v6-p2 | L5 | `FOUR_PLAYER_ROTATING_SEND_RATIO` 0.55→0.62 | ⏳ 可 `--id v6p2` | 暂缓 |
| 6 | v6-p4 | L4 | `GANG_UP_VALUE_MULT` 1.4→1.55 | ✅ layers 16/16 | ❌ Kaggle **804**（< v2 814） |
| 7 | v6-p5 | L4 | `WEAKEST_ENEMY_VALUE_MULT_4P` 1.5→1.7 | ✅ layers 16/16 | ❌ Kaggle **682**（ref 53039881） |
| 8 | v6-p6 | L3 | `FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT` 0.84→0.95 | ✅ layers 16/16 | ❌ Kaggle **673**（ref 53040527） |
| 9 | v6-p7 | L4 | `GANG_UP_ETA_WINDOW` 4→5 | ⏳ `--id v6p7` | 4P 批次下一发（时序，非加权） |
| combo | **v6p3+v6-1** | L4+L3 | `MULTI_ENEMY_PROACTIVE` 0.28 + `COMPETITION_PRESSURE_*` | ✅ layers 16/16 | ❌ 不上线 |

差异常量见 [`ORBIT_WARS_SSOT.md`](ORBIT_WARS_SSOT.md)。**自动化**：`regression/ablations.json` + `scripts/run_ablation.py`。

---

## 5. 回归教训

| 版本 | 记录分数 | 主要嫌疑改动 |
|------|----------|--------------|
| v3 | 772 | B1 双层 margin、参数堆叠 |
| v4 | 750 | `_perf_tight`、FOUR_SOURCE |
| v5 | 783 | ENDGAME dispatcher 等（仍 < v2） |
| v7 | 待测 | 50+ 参数 + 撤回 v6 低风险项 |
| 早期 v6 | — | 基于 baseline，丢失 A1/A2/B1 |

---

## 6. 本地 A/B 记录

| 日期 | 对比 | 结果 | 结论 |
|------|------|------|------|
| 2026-05-25 | v6 vs v2，10 局 1v1 镜像 | 1W / 0L / 9T | INCONCLUSIVE；不能预测线上涨跌 |
| 2026-05-26 | **v6p1 vs v2**，10 局 1v1，`--kaggle-timeout` | **0W / 1L / 9T** | 脚本判 REGRESSED；9 平局≈镜像相当，1 负局 seed=1000；**不替换 Kaggle**；日志 `regression/results/local_ab_v6p1_vs_v2.log` |
| 2026-05-26 | **v6p3 vs v2**，10 局 1v1，`--kaggle-timeout` | **1W / 0L / 9T** | INCONCLUSIVE（10% 胜率）；seed=1000 胜；**暂不替换主提交**；可小流量 Kaggle 探针 |
| 2026-05-26 | `submission_ablation_v6p3.py` `--layers-only` | candidate **16/16 PASS** | 单变量 L4：`MULTI_ENEMY_PROACTIVE_RATIO` 0.35→0.28 |
| 2026-05-26 | `submission_ablation_v6_1.py` `--layers-only` | candidate **16/16 PASS** | 单变量 L3：`COMPETITION_PRESSURE_HARD/SOFT` + `target_value` 块 |
| 2026-05-26 | **v6-1 vs v2**，10 局 1v1，`--kaggle-timeout` | **1W / 0L / 9T** | INCONCLUSIVE；seed=1000 胜（与 v6p3 同型）；**暂不替换主提交** |
| 2026-05-26 | `submission_ablation_v6_2.py` `--layers-only` | candidate **16/16 PASS** | 单变量 L3：`HIGH_PROD_BONUS_THRESHOLD/PER_POINT` + `target_value` 块 |
| 2026-05-26 | **v6-2 vs v2**，10 局 1v1，`--kaggle-timeout` | **0W / 1L / 9T** | REGRESSED；seed=1000 负（同 v6p1）；**不替换 Kaggle**；`regression/results/local_ab_v6_2_vs_v2.log` |
| 2026-05-26 | `submission_ablation_v6_3.py` `--layers-only` | candidate **16/16 PASS** | 单变量 L3：`PROD_BEHIND_NEUTRAL_MULT` + `build_modes.is_prod_behind` |
| 2026-05-26 | **v6-3 vs v2**，10 局 1v1，`--kaggle-timeout` | **0W / 1L / 9T** | REGRESSED；seed=1000 负；**不替换 Kaggle** |
| 2026-05-26 | `submission_ablation_v6p3_v6_1.py` `--layers-only` | candidate **16/16 PASS** | 组合 L4+L3：v6p3 + v6-1 |
| 2026-05-26 | **v6p3+v6-1 vs v2**，10 局 1v1 | **0W / 1L / 9T** | REGRESSED；seed=1000 **负**（单改 v6p3/v6-1 时该 seed 均胜）；**组合未叠加收益**；`local_ab_v6p3_v6_1_vs_v2.log` |

镜像大量平局时，**不能**替代 Kaggle rating 判断。

| 2026-05-26 | **v6p3 Kaggle 探针** | rating **786** | **REGRESSED** vs v2 814；撤回「晋升 v6p3 为 SSOT」 |
| 2026-05-26 | **v6p4 Kaggle 探针** | rating **804** | **REGRESSED** vs 814；`GANG_UP_VALUE_MULT` 1.55 关闭；episode 77750881 |
| 2026-05-26 | **v6p4** 生成 + 探针包 | — | `kaggle_probe/v6p4/` |
| 2026-05-26 | **4P 探针批次** v6p4→v6p5→v6p6 | 804 / 682 / 673 | **全部 REGRESSED**；见 §6.1 |
| 2026-05-26 | **v6p5 Kaggle 探针** | **682** | ref 53039881；`WEAKEST_ENEMY_VALUE_MULT_4P` 1.7 关闭 |
| 2026-05-26 | **v6p6 Kaggle 探针** | **673** | ref 53040527；`FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT` 0.95 关闭 |

### 6.1 探针批次 v6p4 / v6p5 / v6p6（因果分开，可连交）

| ID | 单变量 | Submission ref | 公开分 | vs 814 | 结论 |
|----|--------|----------------|--------|--------|------|
| v6p4 | `GANG_UP_VALUE_MULT` 1.4→1.55 | **53039641**（V3） | **804** | **−10** | **REGRESSED**；replay `77750881` |
| v6p5 | `WEAKEST_ENEMY_VALUE_MULT_4P` 1.5→1.7 | **53039881**（V4）· [script](https://www.kaggle.com/code/tinahe1995/orbit-wars0526?scriptVersionId=322191136) | **682** | **−132** | **REGRESSED** |
| v6p6 | `FOUR_PLAYER_ROTATING_NEUTRAL_SCORE_MULT` 0.84→0.95 | **53040527**（V5）· [script](https://www.kaggle.com/code/tinahe1995/orbit-wars0526?scriptVersionId=322195466) | **673** | **−141** | **REGRESSED** |

**批次结论（2026-05-26）**：4P 价值类三探针 **全部未超 v2 814**；p5/p6 **大幅低于** p4，说明「加权重 / 加中立分」方向在真实池里整体有害，**勿合并三者为新 agent**。**主提交仍 `submission_v2.py`**（建议用 **0524 V2 ref 52990182** 的 notebook 再 submit，勿让 orbit-wars0526 最新低分版本占「最后 2 次提交」槽位）。

| v6p7 | `GANG_UP_ETA_WINDOW` 4→5 | *待交* | *待填* | | 📦 `kaggle_probe/v6p7/` |

**下一步**：v6p4–p6 价值类已关闭；**v6p7** 时序探针见 `kaggle_probe/v6p7/PROBE_README.md`。提交前请用 **v2 代码的 notebook** 复制探针文件，避免 orbit-wars0526 低分版本污染对比。

**结论**：本地 1v1 镜像 **不能** 预测 Kaggle；v6p3 线上已证伪。**主提交保持 v2**。勿在 v6p3 上继续叠 ablation。

---

## 7. Ablation 工作流（摘要）

```bash
# 自动化（推荐）：配方 regression/ablations.json，新实验只加一条
.venv/bin/python scripts/run_ablation.py --id v6p3
.venv/bin/python scripts/run_ablation.py --list

# 手动
cp submission_v2.py submission_ablation_X.py
# 只改一项 → local_ab vs v2 → 有效则写 §4/§6 → Kaggle 探针 → 更新 §1.2
```

完整命令见 [`REGRESSION.md`](REGRESSION.md) §1b、[`TUNING_PLAYBOOK.md`](TUNING_PLAYBOOK.md)。

---

## 8. 固定 seed 回归记录

| 日期 | 命令 | 结果 | 产物 |
|------|------|------|------|
| 2026-05-25 | `regression_test.py --layers-only` | 16/16 PASS | — |
| 2026-05-25 | `regression_test.py --kaggle-timeout` | **18/18 PASS**，0 INVALID | `regression/results/latest.json` |
| 2026-05-26 | `submission_ablation_v6p1.py` `--layers-only` | candidate **16/16 PASS** | 单变量 L4：`HOSTILE_SWARM_ETA_TOLERANCE` 1→2 |

对局 smoke（v2 vs baseline）：1v1 seeds 1100–1102，`avg_reward[0]=0.33`；4P 2000–2002，`avg_reward[0]=1.00`（多平局，不替代 Kaggle rating）。

---

## 9. 变更日志（文档）

| 日期 | 变更 |
|------|------|
| 2026-05-25 | 从 `ORBIT_WARS_ARCHITECTURE.md` 拆出实验/证据/族谱 |
| 2026-05-25 | §0 生产策略；§8 回归 18/18；§4 L3/L4 ablation 队列 |
