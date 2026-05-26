"""Auto-generated decision policy compiler output — do not edit by hand."""

from __future__ import annotations

import os
import random
from typing import Optional

CALIBRATION_READY = True

# Uncalibrated buckets: per-bucket margin floor (non-neutral exploration)
TAU_UNCALIBRATED_BY_BUCKET = {
    "aggressive_expansion": 0.7,
    "comet_chase": 0.5,
    "reinforcement": 1.0,
    "rotating_intercept": 1.5,
    "pass": 999.0,
}
DEFAULT_TAU_UNCALIBRATED = float(os.environ.get("ORBIT_GATE_DEFAULT_TAU", "2.5"))

TAU_MARGIN_BY_BUCKET = {
    "neutral_capture": 0.0,
}

DISABLE_OVERRIDE_BUCKETS = frozenset(["neutral_capture"])
STOCHASTIC_OVERRIDE_BUCKETS = frozenset([])

DECISION_CELLS = {
    "neutral_capture": {
        "0-2": {
            "tier": "green",
            "support": 136,
            "reversal_rate": 0.25735294117647056,
            "reversal_ci_high": 0.3367839006382682,
            "p_trust": 0.7426470588235294,
            "ev_ship_diff": 11.154411764705882,
            "p_improve": 0.7426470588235294,
            "avg_gain": 23.023255813953487,
            "avg_loss": -13.228571428571428
        },
        "10+": {
            "tier": "yellow",
            "support": 5,
            "reversal_rate": 0.2,
            "reversal_ci_high": 0.6244717358814613,
            "p_trust": 0.3755282641185388,
            "ev_ship_diff": 23.2,
            "p_improve": 0.8,
            "avg_gain": 29.25,
            "avg_loss": -1.0
        }
    }
}


def _margin_bin(margin: float) -> str:
    if margin < 2.0:
        return "0-2"
    if margin < 5.0:
        return "2-5"
    if margin < 10.0:
        return "5-10"
    return "10+"


def lookup_cell(bucket: str, margin: float) -> dict:
    """Tier + p_trust + ev_ship_diff for (bucket, margin_bin)."""
    return dict((DECISION_CELLS.get(bucket) or {}).get(_margin_bin(margin), {}))


def expected_override_ev(bucket: str, margin: float) -> float:
    """EV(override) − EV(v2) ≈ E[ship_diff | bucket, margin_bin]."""
    cell = lookup_cell(bucket, margin)
    return float(cell.get("ev_ship_diff", 0.0))


def calibrated_trust(bucket: str, margin: float) -> float:
    """P(override helps) from empirical cell."""
    cell = lookup_cell(bucket, margin)
    return float(cell.get("p_trust", 0.0))


def decide_override(
    bucket: str,
    rollout_margin: float,
    rng: Optional[random.Random] = None,
) -> bool:
    """G→override if EV>0; Y→stochastic(p_trust); R/gray→v2."""
    if not CALIBRATION_READY:
        return False
    if bucket in DISABLE_OVERRIDE_BUCKETS:
        return False
    cell = lookup_cell(bucket, rollout_margin)
    if not cell:
        tau = TAU_UNCALIBRATED_BY_BUCKET.get(bucket, DEFAULT_TAU_UNCALIBRATED)
        return rollout_margin >= tau
    tier = cell.get("tier", "gray")
    ev = expected_override_ev(bucket, rollout_margin)
    if tier in ('red', 'gray') or ev <= 0:
        return False
    if tier == 'green':
        tau = TAU_MARGIN_BY_BUCKET.get(bucket)
        if tau is not None and rollout_margin < tau:
            return False
        return True
    if tier == 'yellow' and bucket in STOCHASTIC_OVERRIDE_BUCKETS:
        r = rng or random
        return r.random() < calibrated_trust(bucket, rollout_margin)
    return False


def should_override(bucket: str, rollout_margin: float) -> bool:
    """Deterministic green-only alias."""
    return decide_override(bucket, rollout_margin) and (
        lookup_cell(bucket, rollout_margin).get("tier") == "green"
    )
