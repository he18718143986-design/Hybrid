"""Safe fallback when hybrid pipeline fails or times out."""

from __future__ import annotations

import math
from typing import Any, List

from src.env.parser import parse_observation

Move = List[Any]


def fallback_moves(obs: Any) -> List[Move]:
    parsed = parse_observation(obs)
    player = parsed["player"]
    my = [p for p in parsed["planets"] if p.owner == player]
    targets = [p for p in parsed["planets"] if p.owner != player]
    if not my or not targets:
        return []
    src = max(my, key=lambda p: p.ships)
    dst = min(targets, key=lambda p: math.hypot(p.x - src.x, p.y - src.y))
    ships = max(1, src.ships // 2)
    angle = math.atan2(dst.y - src.y, dst.x - src.x)
    return [[src.id, angle, ships]]
