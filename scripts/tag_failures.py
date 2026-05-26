#!/usr/bin/env python3
"""Tag decision_reversal rows with failure mechanisms (Phase C2).

Heuristic v0 tags from bucket + margin + regret signals.
Manual review still required for gold labels.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/tag_failures.py
  PYTHONPATH=. .venv/bin/python scripts/tag_failures.py probe_results/decision_delta_hybrid_frontier_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search.calibration import load_decision_delta, record_margin

DEFAULT_INPUT = ROOT / "probe_results" / "decision_delta_hybrid_frontier_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "probe_results" / "failure_taxonomy_v1.json"


def infer_failure_tag(row: dict) -> str:
    """Rule-based v0; replace with manual labels over time."""
    bucket = row.get("bucket") or "unknown"
    margin = record_margin(row) or 0.0
    kind = row.get("divergence_kind") or ""
    ship_diff = float(row.get("ship_diff_hybrid_minus_v2", 0))

    if not row.get("decision_reversal"):
        return "none"

    if margin >= 5.0:
        return "confident_wrong_override"

    if kind == "spurious" or (row.get("differ") and abs(ship_diff) < 3):
        return "spurious_divergence"

    if bucket == "comet_chase":
        if margin >= 2.0:
            return "comet_false_positive"
        return "comet_horizon_blind"

    if bucket == "rotating_intercept":
        return "rotating_eta_miss"

    if bucket == "neutral_capture":
        if ship_diff <= -10:
            return "overextended_neutral"
        return "delayed_counterattack"

    if bucket == "reinforcement":
        return "defensive_overreaction"

    if bucket == "aggressive_expansion":
        return "overextended_neutral"

    return "confident_wrong_override" if margin >= 3 else "unknown_reversal"


def tag_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        tagged = dict(row)
        tagged["failure_tag"] = infer_failure_tag(row)
        out.append(tagged)
    return out


def summarize_tagged(rows: list[dict]) -> dict:
    reversals = [r for r in rows if r.get("decision_reversal")]
    by_tag: dict[str, list] = defaultdict(list)
    for r in reversals:
        by_tag[r["failure_tag"]].append(r)

    tags = {}
    for tag, items in sorted(by_tag.items()):
        diffs = [float(x.get("ship_diff_hybrid_minus_v2", 0)) for x in items]
        margins = [record_margin(x) or 0 for x in items]
        tags[tag] = {
            "n": len(items),
            "avg_ship_diff": sum(diffs) / max(1, len(diffs)),
            "avg_margin": sum(margins) / max(1, len(margins)),
            "buckets": dict(Counter(x.get("bucket") or "unknown" for x in items)),
        }
    return {
        "n_total": len(rows),
        "n_reversal": len(reversals),
        "by_failure_tag": tags,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Failure taxonomy tagger v0")
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write-tagged",
        type=Path,
        default=None,
        help="Optional JSONL with failure_tag field",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing {args.input}")
        return 2

    rows = load_decision_delta(args.input)
    tagged = tag_rows(rows)
    summary = summarize_tagged(tagged)
    summary["source"] = str(args.input)

    print(f"\n=== Failure taxonomy (reversals={summary['n_reversal']}/{summary['n_total']}) ===")
    for tag, stats in summary["by_failure_tag"].items():
        print(
            f"  {tag:<28} n={stats['n']:>4}  "
            f"avg_diff={stats['avg_ship_diff']:>6.1f}  margin={stats['avg_margin']:>5.1f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")

    if args.write_tagged:
        with args.write_tagged.open("w", encoding="utf-8") as f:
            for row in tagged:
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        print(f"Tagged JSONL → {args.write_tagged}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
