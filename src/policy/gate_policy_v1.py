"""Auto-generated from calibration_report — do not edit by hand."""

CALIBRATION_READY = False

TAU_MARGIN_BY_BUCKET = {
}

DISABLE_OVERRIDE_BUCKETS = frozenset([])


def should_override(bucket: str, rollout_margin: float) -> bool:
    """Conservative gate: override only in empirical trust regions."""
    if not CALIBRATION_READY:
        return False
    if bucket in DISABLE_OVERRIDE_BUCKETS:
        return False
    tau = TAU_MARGIN_BY_BUCKET.get(bucket)
    if tau is None:
        return False
    return rollout_margin >= tau
