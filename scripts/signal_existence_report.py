#!/usr/bin/env python3
"""Phase C.5 — Signal existence proof (30-seed decision_delta).

Answers: does a learnable value structure exist, or is hybrid veto-only?

Usage:
  PYTHONPATH=. .venv/bin/python scripts/signal_existence_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search.calibration import load_decision_delta
from src.search.ev_inspector import build_signal_existence_report

DEFAULT_INPUT = ROOT / "probe_results" / "decision_delta_hybrid_frontier_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "probe_results" / "signal_existence_v1.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Signal existence proof (Phase C.5)")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-support", type=int, default=5)
    parser.add_argument(
        "--signal",
        choices=("rollout_margin", "final_score_delta"),
        default="rollout_margin",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing {args.input}")
        return 2

    rows = load_decision_delta(args.input)
    rep = build_signal_existence_report(
        rows, source=str(args.input), min_support=args.min_support, signal=args.signal,
    )
    ex = rep["existence_verdict"]
    g = rep["global"]

    print("\n=== Phase C.5: Signal Existence Proof ===")
    print(f"  n_diverged={rep['n_diverged']}  signal={rep['signal']}")
    print(f"  global ρ={g.get('spearman')}  sign_dir={g.get('sign_accuracy_directional')}  "
          f"noise_ratio={g.get('noise_ratio')}")
    print(f"\n  CASE {ex['case']}: {ex['verdict']}")
    print(f"  route → {ex['recommended_route']}")
    if ex.get("buckets_with_structure"):
        print(f"  structured buckets: {ex['buckets_with_structure']}")

    print(f"\n  {'bucket':<22} {'n':>5} {'ρ':>8} {'sign':>8} {'noise':>8} {'struct':>7}")
    for row in rep["by_bucket"]:
        rho = row.get("spearman")
        rs = f"{rho:.2f}" if rho is not None else "—"
        sg = row.get("sign_accuracy_directional")
        ss = f"{sg:.0%}" if sg is not None else "—"
        nr = row.get("noise_ratio")
        ns = f"{nr:.2f}" if nr is not None else "—"
        st = "YES" if row.get("has_local_structure") else "no"
        print(f"  {row['bucket']:<22} {row['n']:>5} {rs:>8} {ss:>8} {ns:>8} {st:>7}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")

    return 0 if ex["case"] in ("B", "C") else 1


if __name__ == "__main__":
    raise SystemExit(main())
