"""EV signal quality diagnostics."""

from src.search.ev_inspector import (
    build_ev_quality_report,
    check_ev_monotonicity,
    ev_calibration_curve,
    ev_error_decomposition,
    ev_ranking_stats,
)


def test_ev_ranking():
    rows = [
        {"differ": True, "rollout_margin": 1.0, "ship_diff_hybrid_minus_v2": -5},
        {"differ": True, "rollout_margin": 3.0, "ship_diff_hybrid_minus_v2": 0},
        {"differ": True, "rollout_margin": 6.0, "ship_diff_hybrid_minus_v2": 4},
        {"differ": True, "rollout_margin": 9.0, "ship_diff_hybrid_minus_v2": 8},
    ]
    r = ev_ranking_stats(rows)
    assert r["spearman"] is not None
    assert r["spearman"] > 0.9


def test_ev_report():
    rows = [
        {
            "differ": True,
            "bucket": "reinforcement",
            "rollout_margin": float(i),
            "ship_diff_hybrid_minus_v2": i - 2,
        }
        for i in range(8)
    ]
    rep = build_ev_quality_report(rows, min_support=3)
    assert rep["n_diverged"] == 8
    assert "error_decomposition" in rep
    assert rep["ranking_global"]["spearman"] is not None
