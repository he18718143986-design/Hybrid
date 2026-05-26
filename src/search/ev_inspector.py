"""EV signal quality diagnostics — is predicted utility rankable vs actual regret?"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.search.calibration import (
    DEFAULT_MARGIN_BINS,
    MIN_SUPPORT_DEFAULT,
    margin_bin_label,
    record_margin,
)


SPEARMAN_STRUCTURE_THRESHOLD = 0.25
SIGN_BASELINE = 0.5  # random sign match


def predicted_ev(row: dict, signal: str = "rollout_margin") -> Optional[float]:
    """Proxy for E[ship_diff | override]: rollout margin or final_score delta."""
    if signal == "final_score_delta":
        hc = row.get("hybrid_candidate") or {}
        vc = row.get("v2_candidate") or {}
        if not hc:
            return None
        v2_final = float(vc.get("final_score", 0)) if vc else 0.0
        return float(hc.get("final_score", 0)) - v2_final
    return record_margin(row)


def actual_gain(row: dict) -> float:
    return float(row.get("ship_diff_hybrid_minus_v2", 0))


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
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


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den_x = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    if den_x < 1e-12 or den_y < 1e-12:
        return None
    return num / (den_x * den_y)


def _pairs(
    rows: Sequence[dict],
    diverged_only: bool = True,
    bucket: Optional[str] = None,
    signal: str = "rollout_margin",
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for r in rows:
        if diverged_only and not r.get("differ"):
            continue
        if bucket is not None and (r.get("bucket") or "unknown") != bucket:
            continue
        pred = predicted_ev(r, signal)
        if pred is None:
            continue
        out.append((pred, actual_gain(r)))
    return out


def ev_ranking_stats(
    rows: Sequence[dict],
    diverged_only: bool = True,
    bucket: Optional[str] = None,
    signal: str = "rollout_margin",
) -> dict[str, Any]:
    pairs = _pairs(rows, diverged_only, bucket, signal)
    if not pairs:
        return {"n": 0, "spearman": None, "pearson": None}
    preds = [p for p, _ in pairs]
    acts = [a for _, a in pairs]
    return {
        "n": len(pairs),
        "spearman": _spearman(preds, acts),
        "pearson": _pearson(preds, acts),
        "mean_predicted": sum(preds) / len(preds),
        "mean_actual": sum(acts) / len(acts),
    }


def ev_calibration_curve(
    rows: Sequence[dict],
    margin_bins: Sequence[Tuple[str, float, float]] = DEFAULT_MARGIN_BINS,
    diverged_only: bool = True,
    signal: str = "rollout_margin",
) -> List[dict[str, Any]]:
    """Predicted EV bin → mean actual ship_diff (utility calibration curve)."""
    by_bin: Dict[str, List[float]] = {}
    preds_by_bin: Dict[str, List[float]] = {}
    for r in rows:
        if diverged_only and not r.get("differ"):
            continue
        pred = predicted_ev(r, signal)
        if pred is None:
            continue
        mb = margin_bin_label(pred, margin_bins)
        by_bin.setdefault(mb, []).append(actual_gain(r))
        preds_by_bin.setdefault(mb, []).append(pred)

    order = [b[0] for b in margin_bins]
    out: List[dict[str, Any]] = []
    for mb in order:
        gains = by_bin.get(mb, [])
        if not gains:
            continue
        n = len(gains)
        mean_g = sum(gains) / n
        var_g = sum((g - mean_g) ** 2 for g in gains) / max(1, n - 1) if n > 1 else 0.0
        out.append(
            {
                "margin_bin": mb,
                "support": n,
                "mean_predicted_ev": sum(preds_by_bin[mb]) / n,
                "mean_actual_gain": mean_g,
                "std_actual_gain": math.sqrt(var_g),
                "low_support": n < MIN_SUPPORT_DEFAULT,
            }
        )
    return out


def check_ev_monotonicity(
    curve: Sequence[dict],
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> dict[str, Any]:
    """Higher predicted margin bin should mean higher mean actual gain."""
    bin_order = [b[0] for b in DEFAULT_MARGIN_BINS]
    rated = [
        r for r in curve
        if r.get("support", 0) >= min_support and r.get("margin_bin") in bin_order
    ]
    rated.sort(key=lambda r: bin_order.index(r["margin_bin"]))
    violations = []
    for i in range(len(rated) - 1):
        lo, hi = rated[i], rated[i + 1]
        if float(hi["mean_actual_gain"]) < float(lo["mean_actual_gain"]) - 1e-9:
            violations.append(
                {
                    "from_bin": lo["margin_bin"],
                    "to_bin": hi["margin_bin"],
                    "from_gain": lo["mean_actual_gain"],
                    "to_gain": hi["mean_actual_gain"],
                }
            )
    return {
        "bins_compared": len(rated),
        "monotone_increasing": len(violations) == 0 and len(rated) >= 2,
        "violations": violations,
    }


def ev_error_decomposition(
    rows: Sequence[dict],
    diverged_only: bool = True,
    signal: str = "rollout_margin",
) -> dict[str, Any]:
    """Bias / variance / noise proxy for predicted vs actual utility."""
    pairs = _pairs(rows, diverged_only, None, signal)
    if not pairs:
        return {"n": 0}

    errors = [p - a for p, a in pairs]
    abs_errors = [abs(e) for e in errors]
    n = len(errors)
    bias = sum(errors) / n
    mae = sum(abs_errors) / n
    rmse = math.sqrt(sum(e * e for e in errors) / n)

    rev_errors = []
    non_rev_errors = []
    for r in rows:
        if diverged_only and not r.get("differ"):
            continue
        pred = predicted_ev(r, signal)
        if pred is None:
            continue
        err = pred - actual_gain(r)
        if r.get("decision_reversal"):
            rev_errors.append(err)
        else:
            non_rev_errors.append(err)

    margins = [abs(p) for p, _ in pairs]
    noise_proxy = _spearman(margins, abs_errors)

    by_bucket: Dict[str, dict] = {}
    for bucket in sorted({r.get("bucket") or "unknown" for r in rows if r.get("differ")}):
        bp = _pairs(rows, diverged_only, bucket, signal)
        if not bp:
            continue
        be = [p - a for p, a in bp]
        by_bucket[bucket] = {
            "n": len(be),
            "bias": sum(be) / len(be),
            "mae": sum(abs(x) for x in be) / len(be),
            "spearman": _spearman([p for p, _ in bp], [a for _, a in bp]),
        }

    return {
        "n": n,
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "reversal_bias": sum(rev_errors) / len(rev_errors) if rev_errors else None,
        "non_reversal_bias": sum(non_rev_errors) / len(non_rev_errors) if non_rev_errors else None,
        "rollout_noise_proxy": noise_proxy,
        "by_bucket": by_bucket,
    }


def ev_ranking_by_bucket(
    rows: Sequence[dict],
    signal: str = "rollout_margin",
) -> List[dict[str, Any]]:
    buckets = sorted({r.get("bucket") or "unknown" for r in rows if r.get("differ")})
    out = []
    for b in buckets:
        stats = ev_ranking_stats(rows, bucket=b, signal=signal)
        stats["bucket"] = b
        out.append(stats)
    return out


def _variance(vals: Sequence[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    m = sum(vals) / n
    return sum((x - m) ** 2 for x in vals) / (n - 1)


def sign_consistency(pairs: Sequence[Tuple[float, float]]) -> dict[str, Any]:
    """P(sign(pred)==sign(actual)); excludes pred≈0 from directional test."""
    if not pairs:
        return {"n": 0, "sign_accuracy": None, "n_directional": 0}
    all_match = sum(1 for p, a in pairs if (p > 0) == (a > 0) or (p < 0) == (a < 0) or (p == 0 and a == 0))
    directional = [(p, a) for p, a in pairs if abs(p) > 1e-9]
    dir_acc = None
    if directional:
        dir_match = sum(
            1 for p, a in directional
            if (p > 0 and a > 0) or (p < 0 and a < 0)
        )
        dir_acc = dir_match / len(directional)
    return {
        "n": len(pairs),
        "sign_accuracy_all": all_match / len(pairs),
        "sign_accuracy_directional": dir_acc,
        "n_directional": len(directional),
    }


def variance_ratio(pairs: Sequence[Tuple[float, float]]) -> dict[str, Any]:
    """Noise floor: Var(error)/Var(actual); pred flat → ratio≈0."""
    if len(pairs) < 2:
        return {"var_actual": 0.0, "var_error": 0.0, "var_predicted": 0.0, "noise_ratio": None}
    preds = [p for p, _ in pairs]
    acts = [a for _, a in pairs]
    errors = [p - a for p, a in pairs]
    va = _variance(acts)
    ve = _variance(errors)
    vp = _variance(preds)
    return {
        "var_actual": va,
        "var_predicted": vp,
        "var_error": ve,
        "noise_ratio": ve / va if va > 1e-9 else None,
        "signal_ratio": vp / va if va > 1e-9 else None,
    }


def bucket_structure_row(
    rows: Sequence[dict],
    bucket: str,
    min_support: int = MIN_SUPPORT_DEFAULT,
    signal: str = "rollout_margin",
) -> dict[str, Any]:
    pairs = _pairs(rows, True, bucket, signal)
    rank = ev_ranking_stats(rows, bucket=bucket, signal=signal)
    signs = sign_consistency(pairs)
    vr = variance_ratio(pairs)
    rho = rank.get("spearman")
    has_structure = (
        rank.get("n", 0) >= min_support
        and rho is not None
        and abs(rho) >= SPEARMAN_STRUCTURE_THRESHOLD
    )
    return {
        "bucket": bucket,
        "n": rank.get("n", 0),
        "spearman": rho,
        "sign_accuracy_directional": signs.get("sign_accuracy_directional"),
        "noise_ratio": vr.get("noise_ratio"),
        "signal_ratio": vr.get("signal_ratio"),
        "has_local_structure": has_structure,
        "low_support": rank.get("n", 0) < min_support,
    }


def classify_signal_existence(
    bucket_rows: Sequence[dict],
    global_spearman: Optional[float],
    monotone: bool,
    min_support: int = MIN_SUPPORT_DEFAULT,
) -> dict[str, Any]:
    """Case A/B/C verdict for Phase C.5."""
    supported = [r for r in bucket_rows if not r.get("low_support")]
    structured = [r for r in supported if r.get("has_local_structure")]
    n_supported = len(supported)
    n_structured = len(structured)

    if n_supported == 0:
        case = "insufficient_data"
        verdict = "Need more diverged samples before signal existence test."
        route = "run_full_matrix"
    elif n_structured == 0:
        case = "A"
        verdict = "No bucket shows rankable EV structure — rollout is not a value proxy."
        route = "gate_veto_only"  # hybrid = regulated fallback, not planner
    elif n_structured < n_supported and not (
        global_spearman is not None and global_spearman >= SPEARMAN_STRUCTURE_THRESHOLD and monotone
    ):
        case = "B"
        verdict = "Partial local structure — bucket router + conservative calibrator only."
        route = "bucket_local_gate"
        structured_buckets = [r["bucket"] for r in structured]
    elif global_spearman is not None and global_spearman >= SPEARMAN_STRUCTURE_THRESHOLD and monotone:
        case = "C"
        verdict = "Global monotonic value structure — value head / distillation may be viable."
        route = "value_learning_candidate"
        structured_buckets = [r["bucket"] for r in structured]
    else:
        case = "B"
        verdict = "Sparse local structure without global monotonicity — partial gate only."
        route = "bucket_local_gate"
        structured_buckets = [r["bucket"] for r in structured]

    out: dict[str, Any] = {
        "case": case,
        "verdict": verdict,
        "recommended_route": route,
        "buckets_with_structure": [r["bucket"] for r in structured],
        "n_buckets_supported": n_supported,
        "n_buckets_structured": n_structured,
    }
    return out


def build_signal_existence_report(
    rows: Sequence[dict],
    source: str = "",
    min_support: int = MIN_SUPPORT_DEFAULT,
    signal: str = "rollout_margin",
) -> dict[str, Any]:
    """Phase C.5 — does a learnable decision surface exist at all?"""
    buckets = sorted({r.get("bucket") or "unknown" for r in rows if r.get("differ")})
    bucket_rows = [
        bucket_structure_row(rows, b, min_support=min_support, signal=signal) for b in buckets
    ]
    ranking = ev_ranking_stats(rows, signal=signal)
    curve = ev_calibration_curve(rows, signal=signal)
    mono = check_ev_monotonicity(curve, min_support=min_support)
    pairs = _pairs(rows, True, None, signal)
    existence = classify_signal_existence(
        bucket_rows,
        ranking.get("spearman"),
        mono.get("monotone_increasing", False),
        min_support=min_support,
    )

    return {
        "phase": "C.5_signal_existence",
        "source": source,
        "signal": signal,
        "n_diverged": sum(1 for r in rows if r.get("differ")),
        "min_support": min_support,
        "thresholds": {
            "spearman_structure": SPEARMAN_STRUCTURE_THRESHOLD,
            "sign_baseline": SIGN_BASELINE,
        },
        "global": {
            **ranking,
            **sign_consistency(pairs),
            **variance_ratio(pairs),
            "monotone_gain": mono.get("monotone_increasing"),
        },
        "by_bucket": bucket_rows,
        "existence_verdict": existence,
    }


def build_ev_quality_report(
    rows: Sequence[dict],
    source: str = "",
    min_support: int = MIN_SUPPORT_DEFAULT,
    signal: str = "rollout_margin",
) -> dict[str, Any]:
    """Full EV validity diagnostic — run before trusting gate_v2."""
    curve = ev_calibration_curve(rows, diverged_only=True, signal=signal)
    ranking = ev_ranking_stats(rows, signal=signal)
    mono = check_ev_monotonicity(curve, min_support=min_support)
    decomp = ev_error_decomposition(rows, signal=signal)

    ev_valid = (
        ranking.get("n", 0) >= min_support * 2
        and ranking.get("spearman") is not None
        and ranking.get("spearman", 0) > 0
        and mono.get("monotone_increasing", False)
    )

    return {
        "source": source,
        "signal": signal,
        "n_total": len(rows),
        "n_diverged": sum(1 for r in rows if r.get("differ")),
        "ev_signal_valid": ev_valid,
        "ranking_global": ranking,
        "ranking_by_bucket": ev_ranking_by_bucket(rows, signal=signal),
        "ev_calibration_curve": curve,
        "ev_monotonicity": mono,
        "error_decomposition": decomp,
        "interpretation": {
            "ev_valid_means": "Spearman(pred,actual)>0 and mean gain rises with margin bin",
            "if_invalid": "gate structure OK but EV signal biased/non-monotone — refine rollout not τ",
        },
    }
