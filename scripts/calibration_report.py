#!/usr/bin/env python3
"""Trust infrastructure: P(reversal|bucket,margin), ECE, Brier, heatmap, drift."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search.calibration import (
    append_calibration_drift,
    build_calibration_report,
    build_policy_spec,
    build_reliability_heatmap,
    format_heatmap_ascii,
    format_policy_spec_python,
    load_decision_delta,
)

DEFAULT_INPUT = ROOT / "probe_results" / "decision_delta_hybrid_frontier_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "probe_results" / "calibration_report_v1.json"
DEFAULT_DRIFT = ROOT / "probe_results" / "calibration_drift.jsonl"
DEFAULT_POLICY_JSON = ROOT / "probe_results" / "gate_policy_v1.json"
DEFAULT_POLICY_PY = ROOT / "src/policy/gate_policy_v1.py"


def print_margin_curve(curve: list, mono: dict) -> None:
    print("\n=== Margin → P(reversal) [diverged, pooled] ===")
    print(f"{'margin_bin':<10} {'support':>7} {'reversal':>10} {'95% CI':>18} {'avg_diff':>10}")
    for row in curve:
        sup = row.get("support", row.get("n", 0))
        flag = " *" if row.get("low_support") else ""
        ci = f"[{row['reversal_ci_low']:.0%}, {row['reversal_ci_high']:.0%}]"
        print(
            f"{row['margin_bin']:<10} {sup:>7}{flag} "
            f"{row['reversal_rate']:>9.1%} {ci:>18} {row['avg_ship_diff']:>10.1f}"
        )
    if mono.get("bins_compared", 0) >= 2:
        status = "YES" if mono.get("monotone_decreasing") else "NO"
        print(f"\n  Ranking monotonicity (reversal↓ as margin↑)? {status}")
        for v in mono.get("violations", []):
            print(
                f"    violation: {v['from_bin']} ({v['from_rate']:.1%}) → "
                f"{v['to_bin']} ({v['to_rate']:.1%})"
            )


def print_scores(report: dict) -> None:
    ece = report.get("expected_calibration_error") or {}
    brier = report.get("brier_score") or {}
    rank = report.get("ranking_calibration") or {}
    print("\n=== Global scores (diverged) ===")
    if ece.get("ece") is not None:
        print(f"  ECE={ece['ece']:.3f}  Brier={brier.get('brier', 0):.3f}  "
              f"ρ(margin,success)={rank.get('spearman_margin_vs_success')}")
    else:
        print("  (no samples)")

    print("\n=== Per-bucket ECE / Brier / Spearman ===")
    ece_map = {r["bucket"]: r for r in report.get("ece_by_bucket", [])}
    brier_map = {r["bucket"]: r for r in report.get("brier_by_bucket", [])}
    rank_map = {r["bucket"]: r for r in report.get("ranking_by_bucket", [])}
    buckets = sorted(set(ece_map) | set(brier_map) | set(rank_map))
    print(f"{'bucket':<22} {'n':>5} {'ECE':>8} {'Brier':>8} {'ρ':>8}")
    for b in buckets:
        ne = ece_map.get(b, {}).get("n", 0)
        e = ece_map.get(b, {}).get("ece")
        br = brier_map.get(b, {}).get("brier")
        rho = rank_map.get(b, {}).get("spearman_margin_vs_success")
        es = f"{e:.3f}" if e is not None else "—"
        bs = f"{br:.3f}" if br is not None else "—"
        rs = f"{rho:.2f}" if rho is not None else "—"
        print(f"{b:<22} {ne:>5} {es:>8} {bs:>8} {rs:>8}")


def print_heatmap(report: dict) -> None:
    hm = report.get("reliability_heatmap")
    if hm:
        print("\n" + format_heatmap_ascii(hm))


def suggest_tau(table: list, target_reversal: float, min_support: int) -> None:
    print(f"\n=== Suggested τ (reversal_ci_high ≤ {target_reversal:.0%}, support≥{min_support}) ===")
    by_bucket: dict = {}
    for row in table:
        by_bucket.setdefault(row["bucket"], []).append(row)
    bin_order = ["0-2", "2-5", "5-10", "10+"]
    disable: list[str] = []
    for bucket in sorted(by_bucket):
        rows = sorted(
            by_bucket[bucket],
            key=lambda r: bin_order.index(r["margin_bin"])
            if r["margin_bin"] in bin_order
            else 99,
        )
        tau = None
        all_red = True
        for r in rows:
            sup = r.get("support", r.get("n", 0))
            tier = "red"
            if sup >= min_support:
                if r["reversal_ci_high"] <= target_reversal:
                    tier = "green"
                    if tau is None:
                        tau = r["margin_bin"]
                elif r["reversal_ci_low"] < target_reversal:
                    tier = "yellow"
                    all_red = False
                else:
                    all_red = all_red and tier == "red"
            else:
                all_red = False
        if tau is None and all_red and rows:
            disable.append(bucket)
        print(f"  {bucket:<22}  τ_bin≥{tau or '—'}"
              f"{'  [DISABLE_OVERRIDE?]' if bucket in disable else ''}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration trust dashboard")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-reversal", type=float, default=0.35)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument(
        "--tag",
        default="",
        help="Label for drift history (e.g. frontier_v1)",
    )
    parser.add_argument(
        "--drift-history",
        type=Path,
        default=DEFAULT_DRIFT,
        help="Append ECE/Brier snapshot JSONL",
    )
    parser.add_argument("--no-drift", action="store_true", help="Skip drift append")
    parser.add_argument("--no-policy", action="store_true", help="Skip gate policy spec")
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY_JSON)
    parser.add_argument("--policy-py", type=Path, default=DEFAULT_POLICY_PY)
    parser.add_argument("--min-diverged", type=int, default=30)
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing {args.input}")
        print("Run: PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py")
        return 2

    rows = load_decision_delta(args.input)
    report = build_calibration_report(
        rows, source=str(args.input), min_support=args.min_support,
    )
    report["reliability_heatmap"] = build_reliability_heatmap(
        report["by_bucket_margin"],
        target_reversal=args.target_reversal,
        min_support=args.min_support,
    )

    print_margin_curve(
        report["margin_curve_diverged"],
        report.get("margin_monotonicity", {}),
    )
    print_heatmap(report)
    print_scores(report)
    suggest_tau(report["by_bucket_margin"], args.target_reversal, args.min_support)

    if not args.no_policy:
        spec = build_policy_spec(
            report,
            rows=rows,
            target_reversal=args.target_reversal,
            min_support=args.min_support,
            min_diverged_total=args.min_diverged,
        )
        report["policy_spec"] = spec
        args.policy_json.parent.mkdir(parents=True, exist_ok=True)
        args.policy_json.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
        args.policy_py.write_text(format_policy_spec_python(spec), encoding="utf-8")
        print(f"\nPolicy spec: calibration_ready={spec['calibration_ready']}")
        print(f"  → {args.policy_json}")
        print(f"  → {args.policy_py}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")

    if not args.no_drift:
        entry = append_calibration_drift(report, args.drift_history, tag=args.tag)
        print(f"Drift snapshot → {args.drift_history}  tag={entry['tag']!r}  "
              f"ECE={entry['ece']}  Brier={entry['brier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
