"""L2 geometry — ported from submission_v2 (physics section)."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from src.world.constants import (
    CENTER_X,
    CENTER_Y,
    LAUNCH_CLEARANCE,
    MAX_SPEED,
    SUN_R,
    SUN_SAFETY,
)

Point = Tuple[float, float]


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def dist_points(a: Point, b: Point) -> float:
    return dist(a[0], a[1], b[0], b[1])


def angle_between(a: Point, b: Point) -> float:
    return math.atan2(b[1] - a[1], b[0] - a[0])


def fleet_speed(ships: int, max_speed: float = MAX_SPEED) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(1, ships)) / math.log(1000.0)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq <= 1e-9:
        return dist(px, py, x1, y1)
    t = ((px - x1) * dx + (py - y1) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return dist(px, py, proj_x, proj_y)


def segment_hits_sun(x1: float, y1: float, x2: float, y2: float, safety: float = SUN_SAFETY) -> bool:
    return point_to_segment_distance(CENTER_X, CENTER_Y, x1, y1, x2, y2) < SUN_R + safety


def segment_hits_sun_pts(a: Point, b: Point, safety: float = SUN_SAFETY) -> bool:
    return segment_hits_sun(a[0], a[1], b[0], b[1], safety)


def launch_point(sx: float, sy: float, sr: float, angle: float) -> Tuple[float, float]:
    clearance = sr + LAUNCH_CLEARANCE
    return sx + math.cos(angle) * clearance, sy + math.sin(angle) * clearance


def actual_path_geometry(sx: float, sy: float, sr: float, tx: float, ty: float, tr: float):
    angle = math.atan2(ty - sy, tx - sx)
    start_x, start_y = launch_point(sx, sy, sr, angle)
    hit_distance = max(0.0, dist(sx, sy, tx, ty) - (sr + LAUNCH_CLEARANCE) - tr)
    end_x = start_x + math.cos(angle) * hit_distance
    end_y = start_y + math.sin(angle) * hit_distance
    return angle, start_x, start_y, end_x, end_y, hit_distance


def safe_angle_and_distance(
    sx: float, sy: float, sr: float, tx: float, ty: float, tr: float,
) -> Optional[Tuple[float, float]]:
    angle, start_x, start_y, end_x, end_y, hit_distance = actual_path_geometry(
        sx, sy, sr, tx, ty, tr,
    )
    if segment_hits_sun(start_x, start_y, end_x, end_y):
        return None
    return angle, hit_distance


def estimate_arrival(
    sx: float, sy: float, sr: float, tx: float, ty: float, tr: float, ships: int,
) -> Optional[Tuple[float, int]]:
    safe = safe_angle_and_distance(sx, sy, sr, tx, ty, tr)
    if safe is None:
        return None
    angle, total_d = safe
    turns = max(1, int(math.ceil(total_d / fleet_speed(max(1, ships)))))
    return angle, turns


def travel_time(sx: float, sy: float, sr: float, tx: float, ty: float, tr: float, ships: int) -> int:
    est = estimate_arrival(sx, sy, sr, tx, ty, tr, ships)
    if est is None:
        return 10**9
    return est[1]
