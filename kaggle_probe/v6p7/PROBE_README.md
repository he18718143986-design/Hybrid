# Kaggle 探针：v6p7

> 自动生成。提交前请确认 regression 已通过。

## 改动（单变量）

- **GANG_UP_ETA_WINDOW 4→5**
- 源文件：`submission_ablation_v6p7.py`
- 候选 SHA256（短）：`2dfc3e147be9`
- **对照基线**：`submission_v2.py` 记录分 **~814**（submission `52990182`）

## 判据

| 结果 | 动作 |
|------|------|
| 公开分 **> 814** 且稳定 | 记入 `ORBIT_WARS_EXPERIMENTS.md`，考虑换主提交 |
| **≤ 814** 或明显下跌 | 仅记笔记，**主提交仍用 v2** |

本地 `local_ab` 大量平局 **不能** 代替本探针。

---

## 方式 A：CLI 直接提交（推荐）

在仓库根目录执行：

```bash
cd /Users/97jiyu/Downloads/Orbit
.venv/bin/kaggle competitions submit orbit-wars \
  -f kaggle_probe/v6p7/submission.py \
  -m "probe v6p7: GANG_UP_ETA_WINDOW 4→5 (base v2 @ 710d3ff438a2)"
```

提交后查看：

```bash
.venv/bin/kaggle competitions submissions orbit-wars -v | head -5
# 记下 ref → 对局列表
.venv/bin/kaggle competitions episodes <SUBMISSION_REF> -v
```

---

## 方式 B：Notebook 提交

1. 打开你常用的 Orbit Wars Notebook  
2. 将 `kaggle_probe/v6p7/submission.py` 全文复制进一个 cell（保留首行 `# %%writefile submission.py` 若存在）  
3. Run → **Save Version** → **Submit to Competition**  
4. Description 填：`probe v6p7: GANG_UP_ETA_WINDOW 4→5 (base v2 @ 710d3ff438a2)`

---

## 提交后记录（请手动填）

| 字段 | 值 |
|------|-----|
| submission ref | |
| 公开分 | |
| 日期 | |
| 结论 | IMPROVED / REGRESSED / INCONCLUSIVE |

写入 `ORBIT_WARS_EXPERIMENTS.md` §1.2 与 §6。

---

## 配额提醒

- 每天最多 **5** 次提交；**一次只探一个 id**（先 v6p7，再 v6p5、v6p6）。
- 勿在未确认结果前连续交 v6p5/v6p6。
