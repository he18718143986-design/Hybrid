"""Probe metric helpers."""

from src.search.probe_metrics import (
    is_decision_reversal,
    is_positive_override,
    summarize_override_precision,
    summarize_regret,
)


def test_decision_reversal():
    assert is_decision_reversal({"differ": True, "ship_diff_hybrid_minus_v2": -5})
    assert not is_decision_reversal({"differ": False, "ship_diff_hybrid_minus_v2": -5})
    assert not is_decision_reversal({"differ": True, "ship_diff_hybrid_minus_v2": 3})


def test_summarize_regret_reversal_rate():
    rows = [
        {"differ": True, "ship_diff_hybrid_minus_v2": -1, "hybrid_worse_ships": True,
         "planet_diff_hybrid_minus_v2": 0, "action_bucket": "neutral_capture"},
        {"differ": False, "ship_diff_hybrid_minus_v2": 0, "hybrid_worse_ships": False,
         "planet_diff_hybrid_minus_v2": 0, "action_bucket": "pass"},
    ]
    for r in rows:
        r["decision_reversal"] = is_decision_reversal(r)
    s = summarize_regret(rows)
    assert s["decision_reversal_rate"] == 0.5
    assert s["same_decision_rate"] == 0.5


def test_override_precision_dashboard():
    rows = [
        {"differ": True, "ship_diff_hybrid_minus_v2": 5, "hybrid_worse_ships": False},
        {"differ": True, "ship_diff_hybrid_minus_v2": -3, "hybrid_worse_ships": True},
        {"differ": False, "ship_diff_hybrid_minus_v2": 10, "hybrid_worse_ships": False},
    ]
    for r in rows:
        r["decision_reversal"] = is_decision_reversal(r)
    d = summarize_override_precision(rows)
    assert d["override_precision"] == 0.5
    assert d["true_positive_overrides"] == 1
    assert d["opportunity_count"] == 2
    assert d["override_recall"] == 0.5
