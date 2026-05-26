#!/usr/bin/env python3
"""Generate ORBIT_WARS_SSOT.md from submission_v2.py (production constants)."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "submission_v2.py"
OUTPUT = ROOT / "ORBIT_WARS_SSOT.md"
EXPERIMENTAL = ROOT / "submission_v6.py"

LAYER_HINTS: dict[str, str] = {
    "SIM_HORIZON": "L1",
    "HORIZON": "L1",
    "ROUTE_SEARCH_HORIZON": "L2",
    "HOSTILE_REINFORCE": "L4",
    "HOSTILE_SWARM": "L4",
    "MULTI_ENEMY": "L4",
    "PROACTIVE_DEFENSE": "L4",
    "FOUR_PLAYER": "L5",
    "EARLY_TURN": "L5",
    "OPENING_TURN": "L5",
    "LATE_REMAINING": "L5",
    "TOTAL_WAR": "L5",
    "SOFT_ACT": "L5",
    "HEAVY_": "L5",
    "REAR_": "L5",
    "DOOMED_": "L5",
    "VALUE_MULT": "L3",
    "SCORE_MULT": "L3",
    "MARGIN": "L3",
    "ELIMINATION": "L3",
    "ATTACK_COST": "L3",
    "SNIPE_COST": "L3",
}


def guess_layer(name: str) -> str:
    for prefix, layer in LAYER_HINTS.items():
        if name.startswith(prefix) or prefix.rstrip("_") in name:
            return layer
    return "—"


def parse_constants(path: Path) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    rows: list[tuple[str, str]] = []
    for m in re.finditer(r"^([A-Z][A-Z0-9_]*) = (.+?)(?:\s+#.*)?$", text, re.M):
        name, val = m.group(1), m.group(2).strip()
        if name in ("Planet", "Fleet"):
            break
        rows.append((name, val))
    return rows


def file_fingerprint(path: Path) -> dict[str, str | int]:
    data = path.read_bytes()
    lines = path.read_text(encoding="utf-8").splitlines()
    version_tag = ""
    for line in lines[:5]:
        m = re.match(r"^#\s*(.+)$", line.strip())
        if m and "writefile" not in m.group(1).lower():
            version_tag = m.group(1).strip()
            break
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha256_short": hashlib.sha256(data).hexdigest()[:12],
        "lines": len(lines),
        "version_tag": version_tag or "—",
    }


def git_info(root: Path) -> dict[str, str]:
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(
                args,
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
            )
            return out.stdout.strip() or None
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            return None

    commit = run("git", "rev-parse", "HEAD")
    short = run("git", "rev-parse", "--short", "HEAD")
    dirty = run("git", "status", "--porcelain")
    branch = run("git", "rev-parse", "--abbrev-ref", "HEAD")
    if commit is None:
        return {
            "commit": "N/A",
            "commit_short": "N/A",
            "branch": "N/A",
            "dirty": "unknown (not a git repo or git unavailable)",
        }
    return {
        "commit": commit,
        "commit_short": short or commit[:7],
        "branch": branch or "N/A",
        "dirty": "yes (uncommitted changes)" if dirty else "no",
    }


def diff_stat(v2: Path, v6: Path) -> str:
    if not v6.is_file():
        return "submission_v6.py missing"
    try:
        out = subprocess.run(
            ["diff", "-u", str(v2), str(v6)],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        changed = sum(1 for line in out.stdout.splitlines() if line.startswith("+") or line.startswith("-"))
        changed -= 2  # --- +++ headers
        return str(max(0, changed))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "N/A"


def main() -> int:
    if not SOURCE.is_file():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    git = git_info(ROOT)
    prod_fp = file_fingerprint(SOURCE)
    exp_fp = file_fingerprint(EXPERIMENTAL) if EXPERIMENTAL.is_file() else None

    prod = dict(parse_constants(SOURCE))
    exp_rows: list[tuple[str, str, str]] = []
    if EXPERIMENTAL.is_file():
        exp = dict(parse_constants(EXPERIMENTAL))
        for name in sorted(set(prod) | set(exp)):
            if prod.get(name) != exp.get(name):
                exp_rows.append((name, prod.get(name, "—"), exp.get(name, "—")))

    diff_lines = diff_stat(SOURCE, EXPERIMENTAL)

    lines = [
        "# Orbit Wars 常量 SSOT",
        "",
        "> **自动生成 — 勿手改**",
        "",
        "## 生成元数据",
        "",
        "| 字段 | 值 |",
        "|------|-----|",
        f"| 生成时间 (UTC) | `{generated_at}` |",
        f"| Git commit | `{git['commit']}` |",
        f"| Git commit (short) | `{git['commit_short']}` |",
        f"| Git branch | `{git['branch']}` |",
        f"| 工作区 dirty | {git['dirty']} |",
        f"| 生产源文件 | `{prod_fp['path']}` |",
        f"| 生产 version 注释 | `{prod_fp['version_tag']}` |",
        f"| 生产 SHA256 | `{prod_fp['sha256']}` |",
        f"| 生产 SHA256 (short) | `{prod_fp['sha256_short']}` |",
        f"| 生产行数 | {prod_fp['lines']} |",
    ]
    if exp_fp:
        lines.extend([
            f"| 实验源文件 | `{exp_fp['path']}` |",
            f"| 实验 version 注释 | `{exp_fp['version_tag']}` |",
            f"| 实验 SHA256 (short) | `{exp_fp['sha256_short']}` |",
            f"| 实验行数 | {exp_fp['lines']} |",
            f"| v2↔v6 diff 行数 (+/-) | {diff_lines} |",
        ])

    lines.extend([
        "",
        "重新生成：",
        "",
        "```bash",
        ".venv/bin/python scripts/dump_ssot.py",
        "```",
        "",
        "复核 v2↔v6 diff：",
        "",
        "```bash",
        "diff -u submission_v2.py submission_v6.py | grep -E '^\\+|^-' | grep -vE '^\\+\\+\\+|^---' | wc -l",
        "wc -l submission_v2.py submission_v6.py",
        "```",
        "",
        "## 生产常量（submission_v2.py）",
        "",
        "| 常量 | 值 | 层级（启发式） |",
        "|------|-----|----------------|",
    ])
    for name, val in parse_constants(SOURCE):
        lines.append(f"| `{name}` | `{val}` | {guess_layer(name)} |")

    lines.extend([
        "",
        "## 实验分支差异（v6 ≠ v2）",
        "",
        "| 常量 | v2（生产） | v6（实验） |",
        "|------|------------|------------|",
    ])
    if exp_rows:
        for name, v2, v6 in exp_rows:
            lines.append(f"| `{name}` | `{v2}` | `{v6}` |")
    else:
        lines.append("| *(无差异或 submission_v6.py 不存在)* | | |")

    lines.append("")
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"wrote {OUTPUT} @ {generated_at} "
        f"commit={git['commit_short']} "
        f"prod={prod_fp['sha256_short']} "
        f"({len(prod)} constants, {len(exp_rows)} diffs)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
