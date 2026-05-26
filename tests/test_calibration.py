"""Calibration curve aggregation."""

from pathlib import Path

from src.search.calibration import (
    append_calibration_drift,
    brier_score,
    build_calibration_report,
    build_reliability_heatmap,
    check_monotonicity,
    format_heatmap_ascii,
    margin_bin_label,
    ranking_calibration,
    reliability_tier,
    wilson_ci,
)


def test_margin_bin():
    assert margin_bin_label(1.5, [("0-2", 0, 2), ("2-5", 2, 5)]) == "0-2"


def test_wilson_ci():
    lo, hi = wilson_ci(2, 10)
    assert 0 <= lo <= hi <= 1


def test_monotonicity():
    curve = [
        {"margin_bin": "0-2", "support": 20, "reversal_rate": 0.45},
        {"margin_bin": "2-5", "support": 20, "reversal_rate": 0.30},
        {"margin_bin": "5-10", "support": 20, "reversal_rate": 0.15},
        {"margin_bin": "10+", "support": 20, "reversal_rate": 0.05},
    ]
    m = check_monotonicity(curve, min_support=5)
    assert m["monotone_decreasing"] is True


def test_calibration_report():
    rows = [
        {
            "differ": True,
            "decision_reversal": True,
            "bucket": "neutral_capture",
            "rollout_margin": 1.0,
            "ship_diff_hybrid_minus_v2": -5,
        },
        {
            "differ": True,
            "decision_reversal": False,
            "bucket": "neutral_capture",
            "rollout_margin": 8.0,
            "ship_diff_hybrid_minus_v2": 3,
        },
    ]
    rep = build_calibration_report(rows, min_support=1)
    assert rep["n_diverged"] == 2
    assert "expected_calibration_error" in rep
    assert rep["margin_curve_diverged"][0]["support"] == 1
    assert rep["brier_score"]["n"] == 2
    assert rep["ece_by_bucket"]


def test_reliability_tier():
    assert reliability_tier({"support": 1, "reversal_ci_high": 0.2}, min_support=5) == "gray"
    assert (
        reliability_tier(
            {"support": 10, "reversal_ci_high": 0.2, "reversal_ci_low": 0.1},
            min_support=5,
        )
        == "green"
    )


def test_heatmap_and_drift(tmp_path: Path):
    table = [
        {
            "bucket": "reinforcement",
            "margin_bin": "5-10",
            "support": 10,
            "reversal_rate": 0.1,
            "reversal_ci_low": 0.05,
            "reversal_ci_high": 0.2,
        },
    ]
    hm = build_reliability_heatmap(table, min_support=5)
    assert "G" in format_heatmap_ascii(hm)
    hist = tmp_path / "drift.jsonl"
    append_calibration_drift({"source": "x", "n_diverged": 1, "expected_calibration_error": {"ece": 0.1}, "brier_score": {"brier": 0.2}, "margin_monotonicity": {}, "ranking_calibration": {}}, hist, tag="t1")
    assert hist.read_text().strip()


def test_policy_spec():
    table = [
        {
            "bucket": "reinforcement",
            "margin_bin": "5-10",
            "support": 20,
            "reversal_rate": 0.1,
            "reversal_ci_low": 0.05,
            "reversal_ci_high": 0.2,
            "avg_ship_diff": 8.0,
        },
        {
            "bucket": "comet_chase",
            "margin_bin": "0-2",
            "support": 15,
            "reversal_rate": 0.6,
            "reversal_ci_low": 0.4,
            "reversal_ci_high": 0.75,
        },
    ]
    from src.search.calibration import build_policy_spec, format_policy_spec_python

    report = {
        "source": "test",
        "n_diverged": 50,
        "margin_monotonicity": {"monotone_decreasing": True},
        "by_bucket_margin": table,
        "reliability_heatmap": build_reliability_heatmap(table, min_support=5),
    }
    spec = build_policy_spec(report, min_support=5, min_diverged_total=30)
    assert spec["bucket_policy"]["reinforcement"]["mode"] == "deterministic"
    assert "comet_chase" in spec["disable_override_buckets"]
    assert "decide_override" in format_policy_spec_python(spec)


def test_ranking():
    rows = [
        {"differ": True, "decision_reversal": True, "rollout_margin": 1.0},
        {"differ": True, "decision_reversal": False, "rollout_margin": 9.0},
        {"differ": True, "decision_reversal": False, "rollout_margin": 8.0},
    ]
    r = ranking_calibration(rows)
    assert r["spearman_margin_vs_success"] is not None
    assert brier_score(rows)["brier"] is not None
