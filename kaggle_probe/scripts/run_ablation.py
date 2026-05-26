#!/usr/bin/env python3
"""
Generate ablation agents from regression/ablations.json and run regression + local A/B.

Usage:
  .venv/bin/python scripts/run_ablation.py --list
  .venv/bin/python scripts/run_ablation.py --id v6p3
  .venv/bin/python scripts/run_ablation.py --id v6p3 --generate-only
  .venv/bin/python scripts/run_ablation.py --all --skip-ab
  .venv/bin/python scripts/run_ablation.py --id v6p3 --pair 10 --ffa 10 --kaggle-timeout

New experiment: add one entry to regression/ablations.json, then --id <name>.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "regression" / "ablations.json"
RESULTS_DIR = ROOT / "regression" / "results"
PYTHON = sys.executable


class PatchError(Exception):
    pass


def load_registry() -> dict:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if "experiments" not in data:
        raise PatchError("ablations.json missing 'experiments'")
    return data


def resolve_patch_ops(registry: dict, exp_id: str, _stack: list[str] | None = None) -> list[dict]:
    if _stack is None:
        _stack = []
    if exp_id in _stack:
        raise PatchError(f"extends cycle: {' -> '.join(_stack + [exp_id])}")
    exp = registry["experiments"].get(exp_id)
    if exp is None:
        raise PatchError(f"unknown experiment id: {exp_id!r}")
    ops: list[dict] = []
    for parent in exp.get("extends", []):
        ops.extend(resolve_patch_ops(registry, parent, _stack + [exp_id]))
    ops.extend(exp.get("patch_ops", []))
    return ops


def default_header(exp_id: str, exp: dict, registry: dict) -> list[str]:
    sha = registry.get("baseline_sha256_short", "N/A")
    layer = exp.get("layer", "?")
    desc = exp.get("description", exp_id)
    return [
        "# %%writefile submission.py",
        f"# ablation {exp_id} ({layer}): {desc}; base {registry.get('baseline_file', 'submission_v2.py')} @ {sha}",
        "# 814 (parent score — not validated on Kaggle)",
    ]


def apply_header(source: str, header_lines: list[str]) -> str:
    lines = source.splitlines()
    if lines and lines[0].startswith("# %%writefile"):
        idx = 0
        while idx < len(lines) and (lines[idx].startswith("#") or lines[idx].strip() == ""):
            idx += 1
        body = lines[idx:]
    else:
        body = lines
    return "\n".join(header_lines + [""] + body) + "\n"


def apply_patch_ops(source: str, ops: list[dict]) -> str:
    for i, op in enumerate(ops):
        kind = op["op"]
        if kind == "replace_line_contains":
            source, n = _replace_line_contains(source, op["contains"], op["replacement"])
            if n != 1:
                raise PatchError(f"patch {i} ({kind}): expected 1 match for {op['contains']!r}, got {n}")
        elif kind == "insert_after_line_contains":
            source, n = _insert_after_line_contains(source, op["contains"], op["text"])
            if n != 1:
                raise PatchError(f"patch {i} ({kind}): expected 1 match for {op['contains']!r}, got {n}")
        elif kind == "replace_block":
            old = op["old"]
            new = op["new"]
            if old not in source:
                raise PatchError(f"patch {i} ({kind}): block not found")
            count = source.count(old)
            if count != 1:
                raise PatchError(f"patch {i} ({kind}): expected 1 block match, got {count}")
            source = source.replace(old, new, 1)
        else:
            raise PatchError(f"patch {i}: unknown op {kind!r}")
    return source


def _replace_line_contains(source: str, needle: str, replacement: str) -> tuple[str, int]:
    lines = source.splitlines(keepends=True)
    count = 0
    out: list[str] = []
    for line in lines:
        if needle in line:
            count += 1
            if not line.endswith("\n"):
                out.append(replacement)
            else:
                out.append(replacement + "\n")
        else:
            out.append(line)
    return "".join(out), count


def _insert_after_line_contains(source: str, needle: str, text: str) -> tuple[str, int]:
    lines = source.splitlines(keepends=True)
    count = 0
    out: list[str] = []
    for line in lines:
        out.append(line)
        if needle in line:
            count += 1
            insert = text
            if not insert.startswith("\n") and not line.endswith("\n"):
                insert = "\n" + insert
            if insert and not insert.endswith("\n") and not insert.endswith("\n\n"):
                insert = insert + "\n"
            out.append(insert if insert.endswith("\n") else insert + "\n")
    return "".join(out), count


def generate_ablation(registry: dict, exp_id: str) -> tuple[Path, str]:
    exp = registry["experiments"][exp_id]
    baseline_path = ROOT / registry["baseline_file"]
    if not baseline_path.is_file():
        raise PatchError(f"baseline not found: {baseline_path}")
    source = baseline_path.read_text(encoding="utf-8")
    ops = resolve_patch_ops(registry, exp_id)
    source = apply_patch_ops(source, ops)
    header = exp.get("header") or default_header(exp_id, exp, registry)
    source = apply_header(source, header)
    out_path = ROOT / exp.get("output_file", f"submission_ablation_{exp_id}.py")
    out_path.write_text(source, encoding="utf-8")
    return out_path, source


def run_cmd(cmd: list[str], label: str) -> tuple[int, str]:
    print(f"\n--- {label} ---")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    print(out[-8000:] if len(out) > 8000 else out)
    return proc.returncode, out


def parse_local_ab(output: str) -> dict:
    result = {"raw_verdict": None, "wins": None, "losses": None, "ties": None}
    m = re.search(
        r"1v1:\s+([\d.]+)% win rate\s+\((\d+)W/(\d+)L/(\d+)T\)\s+->\s+(\S+)",
        output,
    )
    if m:
        result.update(
            win_rate_pct=float(m.group(1)),
            wins=int(m.group(2)),
            losses=int(m.group(3)),
            ties=int(m.group(4)),
            raw_verdict=m.group(5),
        )
    m4 = re.search(
        r"4P:\s+([\d.]+)% win rate\s+\((\d+)W/(\d+)L/(\d+)T\)\s+->\s+(\S+)",
        output,
    )
    if m4:
        result["ffa"] = {
            "win_rate_pct": float(m4.group(1)),
            "wins": int(m4.group(2)),
            "losses": int(m4.group(3)),
            "ties": int(m4.group(4)),
            "verdict": m4.group(5),
        }
    if re.search(r"REGRESSION: ALL PASSED", output):
        result["regression_passed"] = True
    elif re.search(r"REGRESSION: FAILURES", output):
        result["regression_passed"] = False
    return result


def run_experiment(
    registry: dict,
    exp_id: str,
    *,
    generate_only: bool,
    skip_regression: bool,
    skip_ab: bool,
    pair: int,
    ffa: int,
    kaggle_timeout: bool,
    opponent: str,
) -> dict:
    exp = registry["experiments"][exp_id]
    t0 = time.perf_counter()
    out_path, _ = generate_ablation(registry, exp_id)
    print(f"Generated: {out_path} ({out_path.stat().st_size} bytes)")

    record: dict = {
        "id": exp_id,
        "layer": exp.get("layer"),
        "description": exp.get("description"),
        "output_file": str(out_path.name),
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline": registry["baseline_file"],
    }

    if generate_only:
        record["status"] = "generated_only"
        record["elapsed_s"] = time.perf_counter() - t0
        return record

    if not skip_regression:
        code, out = run_cmd(
            [
                PYTHON,
                str(ROOT / "regression_test.py"),
                "--baseline",
                registry["baseline_file"],
                "--candidate",
                str(out_path),
                "--layers-only",
            ],
            f"regression layers-only ({exp_id})",
        )
        record["regression"] = {"exit_code": code, "passed": code == 0}
        if code != 0:
            record["status"] = "regression_failed"
            record["elapsed_s"] = time.perf_counter() - t0
            _write_result(exp_id, record)
            return record

    if not skip_ab:
        ab_cmd = [
            PYTHON,
            str(ROOT / "local_ab.py"),
            "--new",
            str(out_path),
            "--old",
            opponent,
            "--pair",
            str(pair),
            "--ffa",
            str(ffa),
        ]
        if kaggle_timeout:
            ab_cmd.append("--kaggle-timeout")
        code, out = run_cmd(ab_cmd, f"local_ab ({exp_id})")
        ab_stats = parse_local_ab(out)
        record["local_ab"] = {
            "exit_code": code,
            "pair": pair,
            "ffa": ffa,
            "kaggle_timeout": kaggle_timeout,
            **ab_stats,
        }
        log_path = RESULTS_DIR / f"local_ab_{exp_id}_vs_{Path(opponent).stem}.log"
        log_path.write_text(out, encoding="utf-8")
        record["local_ab"]["log"] = str(log_path.relative_to(ROOT))

    record["status"] = "completed"
    record["elapsed_s"] = round(time.perf_counter() - t0, 1)
    _write_result(exp_id, record)
    return record


def _write_result(exp_id: str, record: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = RESULTS_DIR / f"ablation_{exp_id}_{ts}.json"
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    latest = RESULTS_DIR / "latest_ablation.json"
    bundle = {}
    if latest.is_file():
        try:
            bundle = json.loads(latest.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            bundle = {}
    bundle[exp_id] = record
    bundle["updated_at_utc"] = record.get("generated_at_utc")
    latest.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(f"\nResult: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Orbit Wars ablation from ablations.json")
    parser.add_argument("--id", action="append", dest="ids", help="Experiment id (repeatable)")
    parser.add_argument("--all", action="store_true", help="Run all registered experiments")
    parser.add_argument("--list", action="store_true", help="List experiment ids")
    parser.add_argument("--generate-only", action="store_true", help="Only write ablation .py files")
    parser.add_argument("--skip-regression", action="store_true")
    parser.add_argument("--skip-ab", action="store_true")
    parser.add_argument("--pair", type=int, default=None, help="1v1 games (default from JSON)")
    parser.add_argument("--ffa", type=int, default=None, help="4P FFA games (default from JSON)")
    parser.add_argument(
        "--kaggle-timeout",
        action="store_true",
        default=None,
        help="Use actTimeout=1.0 for local_ab",
    )
    parser.add_argument(
        "--no-kaggle-timeout",
        action="store_true",
        help="Disable kaggle timeout for local_ab",
    )
    parser.add_argument("--opponent", type=str, default=None, help="Override opponent file")
    args = parser.parse_args()

    if not REGISTRY_PATH.is_file():
        print(f"ERROR: missing {REGISTRY_PATH}")
        return 2

    registry = load_registry()
    exp_ids = sorted(registry["experiments"].keys())

    if args.list:
        print(f"Baseline: {registry['baseline_file']} @ {registry.get('baseline_sha256_short', '?')}")
        for eid in exp_ids:
            exp = registry["experiments"][eid]
            extends = f" extends={exp['extends']}" if exp.get("extends") else ""
            print(f"  {eid:<12} [{exp.get('layer', '?'):>5}] {exp.get('description', '')}{extends}")
        return 0

    if args.all:
        ids = exp_ids
    elif args.ids:
        ids = args.ids
        for eid in ids:
            if eid not in registry["experiments"]:
                print(f"ERROR: unknown id {eid!r}; use --list")
                return 2
    else:
        parser.print_help()
        return 2

    defaults = registry.get("default_ab", {})
    pair = args.pair if args.pair is not None else defaults.get("pair", 10)
    ffa = args.ffa if args.ffa is not None else defaults.get("ffa", 0)
    if args.no_kaggle_timeout:
        kaggle_timeout = False
    elif args.kaggle_timeout:
        kaggle_timeout = True
    else:
        kaggle_timeout = defaults.get("kaggle_timeout", True)
    opponent = args.opponent or registry.get("default_opponent", "submission_v2.py")

    failed = 0
    for exp_id in ids:
        print(f"\n{'=' * 60}\nExperiment: {exp_id}\n{'=' * 60}")
        try:
            record = run_experiment(
                registry,
                exp_id,
                generate_only=args.generate_only,
                skip_regression=args.skip_regression,
                skip_ab=args.skip_ab,
                pair=pair,
                ffa=ffa,
                kaggle_timeout=kaggle_timeout,
                opponent=opponent,
            )
            print(f"Status: {record.get('status')} ({record.get('elapsed_s', '?')}s)")
            if record.get("status") in ("regression_failed",):
                failed += 1
        except PatchError as exc:
            print(f"ERROR [{exp_id}]: {exc}")
            failed += 1

    if failed:
        print(f"\n{failed} experiment(s) failed.")
        return 1
    print(f"\nAll {len(ids)} experiment(s) OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
