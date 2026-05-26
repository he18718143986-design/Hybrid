#!/usr/bin/env python3
"""EV Quality Inspector — is predicted utility rankable vs actual regret?

Run after decision_delta, before wiring gate_v2 into hybrid.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/ev_quality_report.py
  PYTHONPATH=. .venv/bin/python scripts/ev_quality_report.py \\
    probe_results/decision_delta_hybrid_frontier_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search.calibration import load_decision_delta
from src.search.ev_inspector import build_ev_quality_report

DEFAULT_INPUT = ROOT / "probe_results" / "decision_delta_hybrid_frontier_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "probe_results" / "ev_quality_report_v1.json"


def print_report(rep: dict) -> None:
    sig = rep.get("signal", "rollout_margin")
    print(f"\n=== EV Quality Inspector (signal={sig}) ===")
    print(f"  ev_signal_valid={rep.get('ev_signal_valid')}  "
          f"n_diverged={rep.get('n_diverged')}")

    g = rep.get("ranking_global") or {}
    sp = g.get("spearman")
    print(f"\n  Global ranking: ρ={sp}  pearson={g.get('pearson')}  n={g.get('n')}")
    print(f"    mean_pred={g.get('mean_predicted')}  mean_actual={g.get('mean_actual')}")

    print("\n  Per-bucket Spearman(pred, actual_gain):")
    print(f"  {'bucket':<22} {'n':>5} {'ρ':>8} {'bias':>10}")
    decomp = rep.get("error_decomposition") or {}
    by_b = decomp.get("by_bucket") or {}
    for row in rep.get("ranking_by_bucket") or []:
        b = row["bucket"]
        rho = row.get("spearman")
        rs = f"{rho:.3f}" if rho is not None else "—"
        bias = by_b.get(b, {}).get("bias")
        bs = f"{bias:+.1f}" if bias is not None else "—"
        print(f"  {b:<22} {row.get('n', 0):>5} {rs:>8} {bs:>10}")

    print("\n  EV calibration curve (predicted bin → mean actual gain):")
    print(f"  {'bin':<8} {'support':>7} {'mean_pred':>10} {'mean_gain':>10}")
    for row in rep.get("ev_calibration_curve") or []:
        flag = " *" if row.get("low_support") else ""
        print(
            f"  {row['margin_bin']:<8} {row['support']:>7}{flag} "
            f"{row['mean_predicted_ev']:>10.2f} {row['mean_actual_gain']:>+10.1f}"
        )

    mono = rep.get("ev_monotonicity") or {}
    if mono.get("bins_compared", 0) >= 2:
        ok = "YES" if mono.get("monotone_increasing") else "NO"
        print(f"\n  Gain monotone in margin bin? {ok}")

    d = rep.get("error_decomposition") or {}
    if d.get("n"):
        print(f"\n  Error decomposition: bias={d.get('bias'):+.2f}  "
              f"MAE={d.get('mae'):.2f}  RMSE={d.get('rmse'):.2f}")
        print(f"    reversal_bias={d.get('reversal_bias')}  "
              f"non_reversal_bias={d.get('non_reversal_bias')}")
        print(f"    rollout_noise_proxy ρ(|margin|,|error|)={d.get('rollout_noise_proxy')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="EV signal validity diagnostics")
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
        print("Run: PYTHONPATH=. .venv/bin/python scripts/probe_matrix.py --cell hybrid_frontier")
        return 2

    rows = load_decision_delta(args.input)
    report = build_ev_quality_report(
        rows, source=str(args.input), min_support=args.min_support, signal=args.signal,
    )
    print_report(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0 if report.get("ev_signal_valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
