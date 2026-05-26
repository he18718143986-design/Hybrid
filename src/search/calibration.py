"""Reliability / calibration curves from decision_delta records."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DEFAULT_MARGIN_BINS: List[Tuple[str, float, float]] = [
    ("0-2", 0.0, 2.0),
    ("2-5", 2.0, 5.0),
    ("5-10", 5.0, 10.0),
    ("10+", 10.0, 1e9),
]

MIN_SUPPORT_DEFAULT = 5


def wilson_ci(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion (95% default)."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def load_decision_delta(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def record_margin(row: dict) -> Optional[float]:
    if row.get("rollout_margin") is not None:
        return float(row["rollout_margin"])
    hc = row.get("hybrid_candidate") or {}
    vc = row.get("v2_candidate") or {}
    if hc:
        v2_rollout = float(vc.get("rollout_score", 0)) if vc else 0.0
        return float(hc.get("rollout_score", 0)) - v2_rollout
    return None


def margin_bin_label(margin: float, bins: Sequence[Tuple[str, float, float]]) -> str:
    for label, lo, hi in bins:
        if lo <= margin < hi:
            return label
    return bins[-1][0] if bins else "unknown"


def margin_to_confidence(margin: float, scale: float = 5.0) -> float:
    """Map rollout margin → predicted P(override is good); monotone in margin."""
    x = -margin / max(1e-6, scale)
    if x >= 20:
        return 0.0
    if x <= -20:
        return 1.0
    return 1.0 / (1.0 + math.exp(x))


def _cell_stats(items: List[dict]) -> dict[str, Any]:
    n = len(items)
    reversals = sum(1 for x in items if x.get("decision_reversal"))
    ship_diffs = [float(x.get("ship_diff_hybrid_minus_v2", 0)) for x in items]
    lo, hi = wilson_ci(reversals, n)
    return {
        "support": n,
        "n": n,
        "reversal_rate": reversals / max(1, n),
        "reversal_ci_low": lo,
        "reversal_ci_high": hi,
        "avg_ship_diff": sum(ship_diffs) / max(1, n),
        "positive_override_rate": sum(1 for d in ship_diffs if d > 0) / max(1, n),
        "low_support": n < MIN_SUPPORT_DEFAULT,
    }


def calibration_table(
    rows: Sequence[dict],
    margin_bins: Sequence[Tuple[str, float, float]] = DEFAULT_MARGIN_BINS,
    diverged_only: bool = True,
) -> List[dict[str, Any]]:
    """Rows: bucket × margin_bin with reversal_rate, Wilson CI, support."""
    filtered = list(rows)
    if diverged_only:
        filtered = [r for r in filtered if r.get("differ")]

    cells: Dict[Tuple[str, str], List[dict]] = {}
    for r in filtered:
        margin = record_margin(r)
        if margin is None:
            continue
        bucket = r.get("bucket") or "unknown"
        mb = margin_bin_label(margin, margin_bins)
        cells.setdefault((bucket, mb), []).append(r)

    table: List[dict[str, Any]] = []
    for (bucket, mb), items in sorted(cells.items()):
        row = _cell_stats(items)
        row["bucket"] = bucket
        row["margin_bin"] = mb
        table.append(row)
    return table


def margin_only_curve(
    rows: Sequence[dict],
    margin_bins: Sequence[Tuple[str, float, float]] = DEFAULT_MARGIN_BINS,
    diverged_only: bool = True,
) -> List[dict[str, Any]]:
    """Aggregate reversal by margin bin (all buckets pooled)."""
    filtered = list(rows)
    if diverged_only:
        filtered = [r for r in rows if r.get("differ")]

    by_bin: Dict[str, List[dict]] = {}
    for r in filtered:
        margin = record_margin(r)
        if margin is None:
            continue
        mb = margin_bin_label(margin, margin_bins)
        by_bin.setdefault(mb, []).append(r)

    order = [b[0] for b in margin_bins]
    out: List[dict[str, Any]] = []
    for mb in order:
        items = by_bin.get(mb, [])
        if not items:
            continue
        row = _cell_stats(items)
        row["margin_bin"] = mb
        out.append(row)
    return out


def check_monotonicity(
    curve: Sequence[dict],
    rate_key: str = "reversal_rate",
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> dict[str, Any]:
    """Check if rate decreases as margin_bin increases (bins with enough support only)."""
    bin_order = [b[0] for b in DEFAULT_MARGIN_BINS]
    rated = [
        r
        for r in curve
        if r.get("support", r.get("n", 0)) >= min_support
        and r.get("margin_bin") in bin_order
    ]
    rated.sort(key=lambda r: bin_order.index(r["margin_bin"]))

    violations = []
    for i in range(len(rated) - 1):
        lo_bin, hi_bin = rated[i], rated[i + 1]
        # Higher margin should mean lower reversal; violation if rate rises with margin.
        if float(hi_bin[rate_key]) > float(lo_bin[rate_key]) + 1e-9:
            violations.append(
                {
                    "from_bin": lo_bin["margin_bin"],
                    "to_bin": hi_bin["margin_bin"],
                    "from_rate": lo_bin[rate_key],
                    "to_rate": hi_bin[rate_key],
                }
            )

    return {
        "bins_compared": len(rated),
        "monotone_decreasing": len(violations) == 0 and len(rated) >= 2,
        "violations": violations,
        "rates_by_bin": [
            {
                "margin_bin": r["margin_bin"],
                "support": r.get("support", r.get("n")),
                rate_key: r[rate_key],
            }
            for r in rated
        ],
    }


def _prediction_pairs(
    rows: Sequence[dict],
    confidence_scale: float = 5.0,
    diverged_only: bool = True,
    bucket: Optional[str] = None,
) -> List[Tuple[float, float]]:
    filtered = [r for r in rows if r.get("differ")] if diverged_only else list(rows)
    pairs: List[Tuple[float, float]] = []
    for r in filtered:
        if bucket is not None and (r.get("bucket") or "unknown") != bucket:
            continue
        margin = record_margin(r)
        if margin is None:
            continue
        pred = margin_to_confidence(margin, confidence_scale)
        actual = 0.0 if r.get("decision_reversal") else 1.0
        pairs.append((pred, actual))
    return pairs


def brier_score(
    rows: Sequence[dict],
    confidence_scale: float = 5.0,
    diverged_only: bool = True,
    bucket: Optional[str] = None,
) -> dict[str, Any]:
    """Brier: mean (pred_trust - actual_success)^2; lower is better."""
    pairs = _prediction_pairs(rows, confidence_scale, diverged_only, bucket)
    if not pairs:
        return {"brier": None, "n": 0}
    sq = sum((p - a) ** 2 for p, a in pairs)
    return {"brier": sq / len(pairs), "n": len(pairs), "confidence_scale": confidence_scale}


def _spearman_rho(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None

    def ranks(vals: Sequence[float]) -> List[float]:
        order = sorted(range(n), key=lambda i: vals[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg_rank
            i = j + 1
        return out

    rx, ry = ranks(list(xs)), ranks(list(ys))
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if den_x < 1e-12 or den_y < 1e-12:
        return None
    return num / (den_x * den_y)


def ranking_calibration(
    rows: Sequence[dict],
    diverged_only: bool = True,
    bucket: Optional[str] = None,
) -> dict[str, Any]:
    """Spearman(margin, success): higher margin should rank above successful overrides."""
    filtered = [r for r in rows if r.get("differ")] if diverged_only else list(rows)
    margins: List[float] = []
    successes: List[float] = []
    for r in filtered:
        if bucket is not None and (r.get("bucket") or "unknown") != bucket:
            continue
        margin = record_margin(r)
        if margin is None:
            continue
        margins.append(margin)
        successes.append(0.0 if r.get("decision_reversal") else 1.0)
    rho = _spearman_rho(margins, successes)
    return {
        "spearman_margin_vs_success": rho,
        "n": len(margins),
        "bucket": bucket,
    }


def expected_calibration_error(
    rows: Sequence[dict],
    confidence_scale: float = 5.0,
    n_bins: int = 10,
    diverged_only: bool = True,
    bucket: Optional[str] = None,
) -> dict[str, Any]:
    """ECE: predicted trust (from margin) vs actual P(override good). Diverged samples only."""
    pairs = _prediction_pairs(rows, confidence_scale, diverged_only, bucket)
    if not pairs:
        return {"ece": None, "n": 0, "bucket": bucket}

    pairs.sort(key=lambda x: x[0])
    bin_size = max(1, len(pairs) // n_bins)
    ece = 0.0
    bin_rows: List[dict] = []
    for i in range(0, len(pairs), bin_size):
        chunk = pairs[i : i + bin_size]
        if not chunk:
            continue
        avg_pred = sum(p for p, _ in chunk) / len(chunk)
        avg_actual = sum(a for _, a in chunk) / len(chunk)
        weight = len(chunk) / len(pairs)
        ece += abs(avg_pred - avg_actual) * weight
        bin_rows.append(
            {
                "support": len(chunk),
                "avg_predicted_trust": avg_pred,
                "avg_actual_success": avg_actual,
                "gap": abs(avg_pred - avg_actual),
            }
        )

    return {
        "ece": ece,
        "n": len(pairs),
        "confidence_scale": confidence_scale,
        "bucket": bucket,
        "bins": bin_rows,
    }


def ece_by_bucket(
    rows: Sequence[dict],
    confidence_scale: float = 5.0,
    n_bins: int = 10,
) -> List[dict[str, Any]]:
    buckets = sorted({r.get("bucket") or "unknown" for r in rows if r.get("differ")})
    out: List[dict[str, Any]] = []
    for bucket in buckets:
        block = expected_calibration_error(
            rows, confidence_scale, n_bins, diverged_only=True, bucket=bucket,
        )
        block["bucket"] = bucket
        out.append(block)
    return out


def reliability_tier(
    row: dict,
    target_reversal: float = 0.35,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> str:
    """green = safe override zone; red = toxic; yellow = uncertain; gray = low support."""
    sup = row.get("support", row.get("n", 0))
    if sup < min_support:
        return "gray"
    if row.get("reversal_ci_high", 1.0) <= target_reversal:
        return "green"
    if row.get("reversal_ci_low", 0.0) >= target_reversal:
        return "red"
    return "yellow"


def build_reliability_heatmap(
    table: Sequence[dict],
    target_reversal: float = 0.35,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> dict[str, Any]:
    """bucket × margin_bin → tier + stats for routing decisions."""
    bin_order = [b[0] for b in DEFAULT_MARGIN_BINS]
    buckets = sorted({r["bucket"] for r in table})
    cells: Dict[str, Dict[str, dict]] = {}
    for r in table:
        tier = reliability_tier(r, target_reversal, min_support)
        cells.setdefault(r["bucket"], {})[r["margin_bin"]] = {
            "tier": tier,
            "support": r.get("support", r.get("n", 0)),
            "reversal_rate": r["reversal_rate"],
            "reversal_ci_low": r["reversal_ci_low"],
            "reversal_ci_high": r["reversal_ci_high"],
        }
    return {
        "margin_bins": bin_order,
        "buckets": buckets,
        "cells": cells,
        "target_reversal": target_reversal,
        "min_support": min_support,
    }


TIER_SYMBOL = {"green": "G", "yellow": "Y", "red": "R", "gray": "."}


def format_heatmap_ascii(heatmap: dict) -> str:
    bins = heatmap["margin_bins"]
    lines = ["Reliability heatmap (G=safe Y=uncertain R=toxic .=low n):"]
    header = f"{'bucket':<22}" + "".join(f"{b:>8}" for b in bins)
    lines.append(header)
    for bucket in heatmap["buckets"]:
        row_cells = heatmap["cells"].get(bucket, {})
        chars = []
        for mb in bins:
            cell = row_cells.get(mb)
            chars.append(TIER_SYMBOL.get(cell["tier"], "?") if cell else " ")
        lines.append(f"{bucket:<22}" + "".join(f"{c:>8}" for c in chars))
    return "\n".join(lines)


def append_calibration_drift(
    report: dict[str, Any],
    history_path: Path,
    tag: str = "",
) -> dict[str, Any]:
    """Append compact snapshot for ECE/Brier drift across frontier/rollout versions."""
    from datetime import datetime, timezone

    ece = report.get("expected_calibration_error") or {}
    brier = report.get("brier_score") or {}
    mono = report.get("margin_monotonicity") or {}
    ranking = report.get("ranking_calibration") or {}
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tag": tag or Path(report.get("source", "")).stem,
        "source": report.get("source"),
        "n_diverged": report.get("n_diverged"),
        "n_reversal": report.get("n_reversal"),
        "ece": ece.get("ece"),
        "brier": brier.get("brier"),
        "spearman": ranking.get("spearman_margin_vs_success"),
        "monotone_decreasing": mono.get("monotone_decreasing"),
    }
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def margin_bin_floor(label: str) -> float:
    for lbl, lo, _hi in DEFAULT_MARGIN_BINS:
        if lbl == label:
            return lo
    return float("inf")


def divergence_structure(rows: Sequence[dict]) -> List[dict[str, Any]]:
    """bucket × divergence_kind counts for reversals."""
    cells: Dict[Tuple[str, str], int] = {}
    for r in rows:
        if not r.get("decision_reversal"):
            continue
        bucket = r.get("bucket") or "unknown"
        kind = r.get("divergence_kind") or "unknown"
        cells[(bucket, kind)] = cells.get((bucket, kind), 0) + 1
    out = [
        {"bucket": b, "divergence_kind": k, "n_reversal": n}
        for (b, k), n in sorted(cells.items())
    ]
    return out


def _cell_utility(items: Sequence[dict]) -> dict[str, float]:
    """EV layer: P(improve) and expected ship delta from diverged samples."""
    n = len(items)
    if n == 0:
        return {
            "p_improve": 0.0,
            "p_improve_ci_low": 0.0,
            "ev_ship_diff": 0.0,
            "avg_gain": 0.0,
            "avg_loss": 0.0,
        }
    diffs = [float(x.get("ship_diff_hybrid_minus_v2", 0)) for x in items]
    reversals = sum(1 for x in items if x.get("decision_reversal"))
    wins = [d for d in diffs if d > 0]
    losses = [d for d in diffs if d < 0]
    p_improve = (n - reversals) / n
    p_lo, _hi = wilson_ci(n - reversals, n)
    return {
        "p_improve": p_improve,
        "p_improve_ci_low": p_lo,
        "ev_ship_diff": sum(diffs) / n,
        "avg_gain": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def build_decision_surface(
    table: Sequence[dict],
    rows: Optional[Sequence[dict]] = None,
    target_reversal: float = 0.35,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> dict[str, Any]:
    """bucket × margin_bin → tier + trust + EV (decision surface reconstruction)."""
    bin_order = [b[0] for b in DEFAULT_MARGIN_BINS]
    raw_cells: Dict[Tuple[str, str], List[dict]] = {}
    if rows is not None:
        filtered = [r for r in rows if r.get("differ")]
        for r in filtered:
            margin = record_margin(r)
            if margin is None:
                continue
            bucket = r.get("bucket") or "unknown"
            mb = margin_bin_label(margin, DEFAULT_MARGIN_BINS)
            raw_cells.setdefault((bucket, mb), []).append(r)

    cells: Dict[str, Dict[str, dict]] = {}
    for row in table:
        bucket = row["bucket"]
        mb = row["margin_bin"]
        tier = reliability_tier(row, target_reversal, min_support)
        util = _cell_utility(raw_cells.get((bucket, mb), []))
        p_trust = max(0.0, 1.0 - row["reversal_rate"])
        if tier == "yellow":
            p_trust = util["p_improve_ci_low"]
        elif tier == "green":
            p_trust = max(p_trust, util["p_improve_ci_low"])
        cells.setdefault(bucket, {})[mb] = {
            "tier": tier,
            "support": row.get("support", row.get("n", 0)),
            "reversal_rate": row["reversal_rate"],
            "reversal_ci_high": row["reversal_ci_high"],
            "p_trust": p_trust,
            "ev_ship_diff": util["ev_ship_diff"] if raw_cells.get((bucket, mb)) else row.get("avg_ship_diff", 0),
            "p_improve": util["p_improve"],
            "avg_gain": util["avg_gain"],
            "avg_loss": util["avg_loss"],
        }

    return {
        "margin_bins": bin_order,
        "buckets": sorted(cells.keys()),
        "cells": cells,
        "target_reversal": target_reversal,
        "min_support": min_support,
    }


def margin_to_trust(
    bucket: str,
    margin: float,
    surface: dict[str, Any],
) -> Tuple[str, float, float]:
    """Lookup tier, calibrated trust, EV from decision surface."""
    mb = margin_bin_label(margin, DEFAULT_MARGIN_BINS)
    cell = (surface.get("cells") or {}).get(bucket, {}).get(mb)
    if not cell:
        return "gray", 0.0, 0.0
    return cell["tier"], float(cell["p_trust"]), float(cell["ev_ship_diff"])


def build_policy_spec(
    report: dict[str, Any],
    rows: Optional[Sequence[dict]] = None,
    target_reversal: float = 0.35,
    min_support: int = MIN_SUPPORT_DEFAULT,
    min_diverged_total: int = 30,
) -> dict[str, Any]:
    """Compile empirical trust surface → EV-gated override policy."""
    table = report.get("by_bucket_margin") or []
    mono = report.get("margin_monotonicity") or {}
    bin_order = [b[0] for b in DEFAULT_MARGIN_BINS]

    surface = build_decision_surface(
        table, rows=rows, target_reversal=target_reversal, min_support=min_support,
    )

    by_bucket: Dict[str, List[dict]] = {}
    for row in table:
        by_bucket.setdefault(row["bucket"], []).append(row)

    bucket_specs: Dict[str, dict[str, Any]] = {}
    disable_buckets: List[str] = []
    tau_by_bucket: Dict[str, float] = {}
    prob_override_buckets: List[str] = []

    for bucket in sorted(by_bucket.keys()):
        rows_b = sorted(
            by_bucket[bucket],
            key=lambda r: bin_order.index(r["margin_bin"])
            if r["margin_bin"] in bin_order
            else 99,
        )
        cell_map = surface["cells"].get(bucket, {})
        diverged_support = sum(r.get("support", r.get("n", 0)) for r in rows_b)
        tau_bin = None
        has_green = False
        has_yellow = False
        all_supported_red = True
        has_supported = False
        best_ev = float("-inf")

        for r in rows_b:
            sup = r.get("support", r.get("n", 0))
            mb = r["margin_bin"]
            cell = cell_map.get(mb, {})
            tier = cell.get("tier", reliability_tier(r, target_reversal, min_support))
            ev = float(cell.get("ev_ship_diff", r.get("avg_ship_diff", 0)))
            if sup >= min_support:
                has_supported = True
                if tier == "green" and tau_bin is None and ev > 0:
                    tau_bin = mb
                    has_green = True
                if tier == "yellow":
                    has_yellow = True
                if tier != "red":
                    all_supported_red = False
                best_ev = max(best_ev, ev)
            else:
                all_supported_red = False

        if not has_supported or diverged_support < min_support:
            action = "fallback_v2"
            mode = "pending_data"
            rationale = f"insufficient support (n={diverged_support})"
        elif has_green and tau_bin is not None:
            action = "override"
            mode = "deterministic"
            tau_by_bucket[bucket] = margin_bin_floor(tau_bin)
            rationale = f"green+EV>0 from bin {tau_bin}"
        elif all_supported_red:
            action = "fallback_v2"
            mode = "veto"
            disable_buckets.append(bucket)
            rationale = "all supported bins toxic (red)"
        elif has_yellow and best_ev > 0:
            action = "override"
            mode = "stochastic"
            prob_override_buckets.append(bucket)
            rationale = "yellow: stochastic override when p_trust×EV>0"
        else:
            action = "fallback_v2"
            mode = "conservative"
            rationale = "no green EV>0; yellow absent or EV≤0"

        bucket_specs[bucket] = {
            "action": action,
            "mode": mode,
            "tau_margin": tau_by_bucket.get(bucket),
            "tau_bin": tau_bin,
            "rationale": rationale,
            "support_diverged": diverged_support,
            "best_ev_ship_diff": best_ev if best_ev > float("-inf") else 0.0,
        }

    n_diverged = int(report.get("n_diverged") or 0)
    ece = (report.get("expected_calibration_error") or {}).get("ece")
    calibration_ready = (
        n_diverged >= min_diverged_total
        and bool(mono.get("monotone_decreasing"))
        and ece is not None
        and ece < 0.45
        and (len(disable_buckets) + len(tau_by_bucket) + len(prob_override_buckets) > 0)
    )

    spec: dict[str, Any] = {
        "version": "gate_v2",
        "source": report.get("source"),
        "target_reversal": target_reversal,
        "min_support": min_support,
        "calibration_ready": calibration_ready,
        "n_diverged": n_diverged,
        "ece": ece,
        "margin_monotone_decreasing": mono.get("monotone_decreasing"),
        "decision_surface": surface,
        "tau_margin_by_bucket": tau_by_bucket,
        "disable_override_buckets": sorted(disable_buckets),
        "stochastic_override_buckets": sorted(prob_override_buckets),
        "bucket_policy": bucket_specs,
        "tier_semantics": {
            "green": "deterministic override if EV>0",
            "yellow": "stochastic override with p=p_trust when EV>0",
            "red": "veto → fallback v2",
            "gray": "insufficient data → fallback v2",
        },
    }
    if rows is not None:
        spec["reversal_by_bucket_divergence_kind"] = divergence_structure(rows)
    return spec


def format_policy_spec_python(spec: dict[str, Any]) -> str:
    """Decision policy compiler: safety + EV + calibrated trust."""
    disable = spec.get("disable_override_buckets") or []
    stochastic = spec.get("stochastic_override_buckets") or []
    tau = spec.get("tau_margin_by_bucket") or {}
    surface = spec.get("decision_surface") or {}
    cells = surface.get("cells") or {}

    lines = [
        '"""Auto-generated decision policy compiler output — do not edit by hand."""',
        "",
        "from __future__ import annotations",
        "",
        "import random",
        "from typing import Optional",
        "",
        f"CALIBRATION_READY = {spec.get('calibration_ready', False)}",
        "",
        "TAU_MARGIN_BY_BUCKET = {",
    ]
    for k in sorted(tau):
        lines.append(f'    "{k}": {tau[k]},')
    lines.append("}")
    lines.append("")
    lines.append(f"DISABLE_OVERRIDE_BUCKETS = frozenset({disable!r})")
    lines.append(f"STOCHASTIC_OVERRIDE_BUCKETS = frozenset({stochastic!r})")
    lines.append("")
    lines.append(f"DECISION_CELLS = {json.dumps(cells, indent=4)}")
    lines.append("")
    lines.append("")
    lines.append("def _margin_bin(margin: float) -> str:")
    lines.append("    if margin < 2.0:")
    lines.append('        return "0-2"')
    lines.append("    if margin < 5.0:")
    lines.append('        return "2-5"')
    lines.append("    if margin < 10.0:")
    lines.append('        return "5-10"')
    lines.append('    return "10+"')
    lines.append("")
    lines.append("")
    lines.append("def lookup_cell(bucket: str, margin: float) -> dict:")
    lines.append('    """Tier + p_trust + ev_ship_diff for (bucket, margin_bin)."""')
    lines.append("    return dict((DECISION_CELLS.get(bucket) or {}).get(_margin_bin(margin), {}))")
    lines.append("")
    lines.append("")
    lines.append("def expected_override_ev(bucket: str, margin: float) -> float:")
    lines.append('    """EV(override) − EV(v2) ≈ E[ship_diff | bucket, margin_bin]."""')
    lines.append("    cell = lookup_cell(bucket, margin)")
    lines.append('    return float(cell.get("ev_ship_diff", 0.0))')
    lines.append("")
    lines.append("")
    lines.append("def calibrated_trust(bucket: str, margin: float) -> float:")
    lines.append('    """P(override helps) from empirical cell."""')
    lines.append("    cell = lookup_cell(bucket, margin)")
    lines.append('    return float(cell.get("p_trust", 0.0))')
    lines.append("")
    lines.append("")
    lines.append("def decide_override(")
    lines.append("    bucket: str,")
    lines.append("    rollout_margin: float,")
    lines.append("    rng: Optional[random.Random] = None,")
    lines.append(") -> bool:")
    lines.append('    """G→override if EV>0; Y→stochastic(p_trust); R/gray→v2."""')
    lines.append("    if not CALIBRATION_READY:")
    lines.append("        return False")
    lines.append("    if bucket in DISABLE_OVERRIDE_BUCKETS:")
    lines.append("        return False")
    lines.append("    cell = lookup_cell(bucket, rollout_margin)")
    lines.append('    tier = cell.get("tier", "gray")')
    lines.append("    ev = expected_override_ev(bucket, rollout_margin)")
    lines.append("    if tier in ('red', 'gray') or ev <= 0:")
    lines.append("        return False")
    lines.append("    if tier == 'green':")
    lines.append("        tau = TAU_MARGIN_BY_BUCKET.get(bucket)")
    lines.append("        if tau is not None and rollout_margin < tau:")
    lines.append("            return False")
    lines.append("        return True")
    lines.append("    if tier == 'yellow' and bucket in STOCHASTIC_OVERRIDE_BUCKETS:")
    lines.append("        r = rng or random")
    lines.append("        return r.random() < calibrated_trust(bucket, rollout_margin)")
    lines.append("    return False")
    lines.append("")
    lines.append("")
    lines.append("def should_override(bucket: str, rollout_margin: float) -> bool:")
    lines.append('    """Deterministic green-only alias."""')
    lines.append("    return decide_override(bucket, rollout_margin) and (")
    lines.append('        lookup_cell(bucket, rollout_margin).get("tier") == "green"')
    lines.append("    )")
    return "\n".join(lines) + "\n"


def build_calibration_report(
    rows: Sequence[dict],
    source: str = "",
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> dict[str, Any]:
    diverged = [r for r in rows if r.get("differ")]
    curve = margin_only_curve(rows, diverged_only=True)
    by_bucket_margin = calibration_table(rows, diverged_only=True)
    return {
        "source": source,
        "n_total": len(rows),
        "n_diverged": len(diverged),
        "n_reversal": sum(1 for r in diverged if r.get("decision_reversal")),
        "min_support_threshold": min_support,
        "margin_curve_diverged": curve,
        "margin_monotonicity": check_monotonicity(curve, min_support=min_support),
        "by_bucket_margin": by_bucket_margin,
        "by_bucket_all": _bucket_summary(rows, diverged_only=False),
        "expected_calibration_error": expected_calibration_error(rows),
        "ece_by_bucket": ece_by_bucket(rows),
        "brier_score": brier_score(rows),
        "brier_by_bucket": [
            {**brier_score(rows, bucket=b), "bucket": b}
            for b in sorted({r.get("bucket") or "unknown" for r in diverged})
        ],
        "ranking_calibration": ranking_calibration(rows),
        "ranking_by_bucket": [
            {**ranking_calibration(rows, bucket=b), "bucket": b}
            for b in sorted({r.get("bucket") or "unknown" for r in diverged})
        ],
        "reliability_heatmap": build_reliability_heatmap(
            by_bucket_margin, min_support=min_support,
        ),
    }


def _bucket_summary(rows: Sequence[dict], diverged_only: bool) -> List[dict]:
    filtered = [r for r in rows if r.get("differ")] if diverged_only else list(rows)
    by_bucket: Dict[str, List[dict]] = {}
    for r in filtered:
        by_bucket.setdefault(r.get("bucket") or "unknown", []).append(r)
    out = []
    for bucket, items in sorted(by_bucket.items()):
        row = _cell_stats(items)
        row["bucket"] = bucket
        out.append(row)
    return out
