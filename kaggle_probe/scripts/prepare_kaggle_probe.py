#!/usr/bin/env python3
"""
Prepare a Kaggle probe bundle: generate ablation, run layers-only regression, copy submission.py.

Usage:
  .venv/bin/python scripts/prepare_kaggle_probe.py --id v6p4
  .venv/bin/python scripts/prepare_kaggle_probe.py --id v6p4 --skip-regression

Output: kaggle_probe/<id>/submission.py + PROBE_README.md + manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE_ROOT = ROOT / "kaggle_probe"
PYTHON = sys.executable


def sha256_short(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h[:12]


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Kaggle probe bundle for one ablation id")
    parser.add_argument("--id", required=True, help="Experiment id in regression/ablations.json")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-generate", action="store_true", help="Use existing ablation .py")
    args = parser.parse_args()
    exp_id = args.id

    if not args.skip_generate:
        code = subprocess.call(
            [PYTHON, str(ROOT / "scripts" / "run_ablation.py"), "--id", exp_id, "--generate-only"],
            cwd=ROOT,
        )
        if code != 0:
            return code

    registry = json.loads((ROOT / "regression" / "ablations.json").read_text(encoding="utf-8"))
    exp = registry["experiments"].get(exp_id)
    if not exp:
        print(f"ERROR: unknown experiment {exp_id!r}")
        return 2

    src = ROOT / exp["output_file"]
    if not src.is_file():
        print(f"ERROR: missing {src}")
        return 2

    regression = {"skipped": True}
    if not args.skip_regression:
        print(f"\n--- regression layers-only ({exp_id}) ---")
        proc = subprocess.run(
            [
                PYTHON,
                str(ROOT / "regression_test.py"),
                "--baseline",
                registry["baseline_file"],
                "--candidate",
                str(src),
                "--layers-only",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        print(out[-4000:] if len(out) > 4000 else out)
        regression = {"exit_code": proc.returncode, "passed": proc.returncode == 0}
        if proc.returncode != 0:
            print(f"\nPROBE BLOCKED: regression failed for {exp_id}")
            return 1

    out_dir = PROBE_ROOT / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "submission.py"
    dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "experiment_id": exp_id,
        "layer": exp.get("layer"),
        "description": exp.get("description"),
        "source_file": src.name,
        "baseline_file": registry["baseline_file"],
        "baseline_sha256_short": registry.get("baseline_sha256_short"),
        "candidate_sha256_short": sha256_short(dest),
        "baseline_score_target": 814,
        "prepared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "regression": regression,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme = _readme(exp_id, exp, dest, manifest)
    (out_dir / "PROBE_README.md").write_text(readme, encoding="utf-8")

    print(f"\nOK: probe bundle ready at {out_dir.relative_to(ROOT)}/")
    print(f"  submission.py  ({dest.stat().st_size} bytes)")
    print(f"  PROBE_README.md")
    print(f"\nNext: open PROBE_README.md and submit (one probe per day recommended).")
    return 0


def _readme(exp_id: str, exp: dict, dest: Path, manifest: dict) -> str:
    desc = exp.get("description", exp_id)
    msg = f"probe {exp_id}: {desc} (base v2 @ {manifest['baseline_sha256_short']})"
    return f"""# Kaggle 探针：{exp_id}

> 自动生成。提交前请确认 regression 已通过。

## 改动（单变量）

- **{desc}**
- 源文件：`{manifest['source_file']}`
- 候选 SHA256（短）：`{manifest['candidate_sha256_short']}`
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
cd {ROOT.as_posix()}
.venv/bin/kaggle competitions submit orbit-wars \\
  -f kaggle_probe/{exp_id}/submission.py \\
  -m "{msg}"
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
2. 将 `kaggle_probe/{exp_id}/submission.py` 全文复制进一个 cell（保留首行 `# %%writefile submission.py` 若存在）  
3. Run → **Save Version** → **Submit to Competition**  
4. Description 填：`{msg}`

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

- 每天最多 **5** 次提交；**一次只探一个 id**（先 {exp_id}，再 v6p5、v6p6）。
- 勿在未确认结果前连续交 v6p5/v6p6。
"""


if __name__ == "__main__":
    raise SystemExit(main())
