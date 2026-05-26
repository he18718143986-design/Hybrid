#!/usr/bin/env python3
"""Decision policy compiler: calibration → EV-gated override policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search.calibration import (
    build_calibration_report,
    build_policy_spec,
    build_reliability_heatmap,
    format_policy_spec_python,
    load_decision_delta,
)

DEFAULT_REPORT = ROOT / "probe_results" / "calibration_report_v1.json"
DEFAULT_DELTA = ROOT / "probe_results" / "decision_delta_hybrid_frontier_v1.jsonl"
DEFAULT_SPEC_JSON = ROOT / "probe_results" / "gate_policy_v1.json"
DEFAULT_SPEC_PY = ROOT / "src/policy/gate_policy_v1.py"


def print_policy_summary(spec: dict) -> None:
    print(f"\n=== Decision policy compiler (ready={spec.get('calibration_ready')}) ===")
    print(f"  version={spec.get('version')}  n_diverged={spec.get('n_diverged')}  "
          f"ECE={spec.get('ece')}  monotone={spec.get('margin_monotone_decreasing')}")
    print(f"  DISABLE={spec.get('disable_override_buckets')}")
    print(f"  STOCHASTIC={spec.get('stochastic_override_buckets')}")
    for bucket, pol in sorted((spec.get("bucket_policy") or {}).items()):
        tau = pol.get("tau_margin")
        ts = f"{tau:.2f}" if tau is not None else "—"
        ev = pol.get("best_ev_ship_diff", 0)
        print(
            f"  {bucket:<22} {pol.get('mode', pol.get('action')):<14} "
            f"τ≥{ts}  EV={ev:+.1f}  ({pol['rationale']})"
        )
    surface = spec.get("decision_surface", {}).get("cells") or {}
    if surface:
        print("\n  decision surface (tier / p_trust / EV):")
        for bucket in sorted(surface):
            for mb, cell in sorted(surface[bucket].items()):
                print(
                    f"    {bucket:<18} {mb:<6} {cell.get('tier','?'):<6} "
                    f"p={cell.get('p_trust', 0):.2f}  ev={cell.get('ev_ship_diff', 0):+.1f}  "
                    f"n={cell.get('support', 0)}"
                )
    rev = spec.get("reversal_by_bucket_divergence_kind") or []
    if rev:
        print("\n  reversal structure (bucket × divergence_kind):")
        for row in rev[:12]:
            print(f"    {row['bucket']:<20} {row['divergence_kind']:<22} n={row['n_reversal']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibration → gate policy spec")
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Existing calibration_report JSON (else build from --delta)",
    )
    parser.add_argument("--delta", type=Path, default=DEFAULT_DELTA)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_SPEC_JSON)
    parser.add_argument("--output-py", type=Path, default=DEFAULT_SPEC_PY)
    parser.add_argument("--target-reversal", type=float, default=0.35)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument("--min-diverged", type=int, default=30)
    args = parser.parse_args()

    rows = None
    if args.report and args.report.is_file():
        report = json.loads(args.report.read_text(encoding="utf-8"))
    elif args.delta.is_file():
        rows = load_decision_delta(args.delta)
        report = build_calibration_report(
            rows, source=str(args.delta), min_support=args.min_support,
        )
        report["reliability_heatmap"] = build_reliability_heatmap(
            report["by_bucket_margin"],
            target_reversal=args.target_reversal,
            min_support=args.min_support,
        )
    else:
        print("Need --report or decision_delta JSONL")
        return 2

    spec = build_policy_spec(
        report,
        rows=rows,
        target_reversal=args.target_reversal,
        min_support=args.min_support,
        min_diverged_total=args.min_diverged,
    )
    print_policy_summary(spec)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output_json}")

    py_text = format_policy_spec_python(spec)
    args.output_py.parent.mkdir(parents=True, exist_ok=True)
    args.output_py.write_text(py_text, encoding="utf-8")
    print(f"Wrote {args.output_py}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
