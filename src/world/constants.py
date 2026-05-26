"""Physical and tuning constants — mirror submission_v2 CONFIG until fully extracted."""

from __future__ import annotations

BOARD: float = 100.0
CENTER_X: float = 50.0
CENTER_Y: float = 50.0
SUN_R: float = 10.0
MAX_SPEED: float = 6.0
SUN_SAFETY: float = 1.5
ROTATION_LIMIT: float = 50.0
TOTAL_STEPS: int = 500
SIM_HORIZON: int = 110
HORIZON: int = SIM_HORIZON
ROUTE_SEARCH_HORIZON: int = 60
LAUNCH_CLEARANCE: float = 0.1
INTERCEPT_TOLERANCE: int = 1
DEFAULT_ACT_TIMEOUT_S: float = 1.0
SOFT_ACT_CAP: float = 0.82
SOFT_ACT_FLOOR: float = 0.55

DEFENSE_MARGIN: float = 0.20
