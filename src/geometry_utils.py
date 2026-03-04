from typing import List, Tuple
import numpy as np

def point_to_line_segment_dist(
    p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]
) -> float:
    """Distance from point p to line segment ab."""
    px, py = p
    ax, ay = a
    bx, by = b

    dx = bx - ax
    dy = by - ay

    if dx == 0 and dy == 0:
        return float(np.hypot(px - ax, py - ay))

    # Projection factor t
    t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)

    if t < 0:
        return float(np.hypot(px - ax, py - ay))
    elif t > 1:
        return float(np.hypot(px - bx, py - by))
    else:
        cx = ax + t * dx
        cy = ay + t * dy
        return float(np.hypot(px - cx, py - cy))

def min_dist_to_grain_boundaries(
    p: Tuple[float, float],
    gbs: List[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> float:
    """Min distance from point p to nearest segment in grain_boundaries."""
    if not gbs:
        return float("nan")
    return min(point_to_line_segment_dist(p, gb[0], gb[1]) for gb in gbs)
