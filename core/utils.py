from __future__ import annotations
import math, random
from typing import List, Sequence, Tuple, Any

def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def mean(xs: Sequence[float]) -> float:
    return (sum(xs) / len(xs)) if xs else 0.0

def weighted_pick(opts: Sequence[Any], weights: Sequence[float], rng: random.Random) -> Any:
    total = sum(weights)
    if total <= 0:
        return rng.choice(list(opts))
    r = rng.random() * total
    acc = 0.0
    for o, w in zip(opts, weights):
        acc += w
        if r <= acc:
            return o
    return opts[-1]

def stage_of_grade(g: int) -> str:
    if g <= 4:  return "LP"
    if g <= 8:  return "UP"
    if g <= 10: return "LS"
    return "US"

def dist_point_to_rect(px: int, py: int, left: int, right: int, bottom: int, top: int) -> float:
    dx = 0 if left <= px <= right else (left - px if px < left else px - right)
    dy = 0 if bottom <= py <= top else (bottom - py if py < bottom else py - top)
    return math.hypot(dx, dy)
