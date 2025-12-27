"""
nav_utils.py

Small navigation/math utilities used across the project:

- Angle wrapping
- Clamping
- Map size loading (ASCII)
- Waypoint simplification / downsampling
- Pure pursuit target selection
- Polyline length and point-to-polyline distance (lateral deviation)
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple


Point2D = Tuple[float, float]


# =============================================================================
# Basic math helpers
# =============================================================================

def wrap_angle(a: float) -> float:
    """
    Wrap an angle to [-pi, pi] using atan2(sin, cos).
    This is the standard robust way to avoid discontinuities.
    """
    return math.atan2(math.sin(a), math.cos(a))


def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp a value into [lo, hi]."""
    return max(lo, min(hi, v))


# =============================================================================
# Map helpers
# =============================================================================

def load_map_size(path: str) -> Tuple[int, int]:
    """
    Read an ASCII map file and return (width, height) in characters/cells.
    Empty lines are ignored.

    Returns
    -------
    w : int
        Max line length (after stripping)
    h : int
        Number of non-empty lines
    """
    with open(path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip("\n")]
    h = len(lines)
    w = max((len(ln.strip()) for ln in lines), default=0)
    return w, h


# =============================================================================
# Waypoint post-processing
# =============================================================================

def simplify_waypoints_collinear(wps: Sequence[Point2D], eps: float = 1e-9) -> List[Point2D]:
    """
    Remove collinear intermediate waypoints.

    Why:
    A* returns a "staircase" polyline. Many points are redundant when three consecutive
    points lie on the same line. Removing collinear points makes paths cleaner and
    reduces controller workload.

    eps:
        Numerical tolerance for cross product check.
    """
    if not wps or len(wps) < 3:
        return list(wps)

    out = [wps[0]]
    for i in range(1, len(wps) - 1):
        x0, y0 = out[-1]
        x1, y1 = wps[i]
        x2, y2 = wps[i + 1]

        # Cross product of vectors (p0->p1) x (p0->p2)
        cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

        # If not collinear, keep the point
        if abs(cross) > eps:
            out.append((x1, y1))

    out.append(wps[-1])
    return out


def downsample_waypoints(wps: Sequence[Point2D], min_dist: float = 1.0) -> List[Point2D]:
    """
    Downsample waypoints so that consecutive kept points are at least min_dist apart.

    Why:
    - Reduces waypoint density, which reduces oscillations for simple controllers.
    - Keeps the same overall path shape.

    Note:
    This uses an "accumulated distance" rule: it keeps the next point when the
    traveled distance since the last kept point exceeds min_dist.
    """
    if not wps:
        return list(wps)

    out = [wps[0]]
    last_x, last_y = wps[0]
    acc = 0.0

    for x, y in wps[1:]:
        d = math.hypot(x - last_x, y - last_y)
        acc += d
        if acc >= min_dist:
            out.append((x, y))
            acc = 0.0
            last_x, last_y = x, y

    if out[-1] != wps[-1]:
        out.append(wps[-1])

    return out


# =============================================================================
# Pure pursuit helper
# =============================================================================

def pure_pursuit_target(
    pose: Tuple[float, float, float],
    wps: Sequence[Point2D],
    idx: int,
    lookahead: float = 1.8,
) -> Tuple[float, float, int]:
    """
    Pick a "lookahead" target point for pure pursuit.

    Parameters
    ----------
    pose : (x, y, theta)
        Current robot pose estimate (typically PF pose).
    wps :
        List of waypoints to follow (world coordinates).
    idx :
        Current waypoint index (progress state).
    lookahead :
        Desired distance ahead along the waypoint list.

    Returns
    -------
    (tx, ty, new_idx)
        Target point coordinates + updated waypoint index.

    Why it avoids "flip-flop":
    - It monotonically advances idx when close enough.
    - It searches forward from idx to find the first point at >= lookahead distance.
    """
    x, y, _ = pose
    if not wps:
        return x, y, idx

    idx = max(0, min(len(wps) - 1, idx))

    # Advance the waypoint index when I am already close to it.
    # Using lookahead*0.5 makes it less jittery.
    while idx < len(wps) - 1 and math.hypot(wps[idx][0] - x, wps[idx][1] - y) < lookahead * 0.5:
        idx += 1

    # Find first waypoint at least lookahead away
    for j in range(idx, len(wps)):
        if math.hypot(wps[j][0] - x, wps[j][1] - y) >= lookahead:
            return wps[j][0], wps[j][1], j

    # If none found, target the last waypoint
    j_last = len(wps) - 1
    return wps[j_last][0], wps[j_last][1], j_last


# =============================================================================
# Polyline metrics (used for plotting and evaluation)
# =============================================================================

def polyline_length(poly: Sequence[Point2D]) -> float:
    """Return the length of a polyline (sum of segment lengths)."""
    if not poly or len(poly) < 2:
        return 0.0
    return sum(
        math.hypot(poly[i + 1][0] - poly[i][0], poly[i + 1][1] - poly[i][1])
        for i in range(len(poly) - 1)
    )


def _dist_point_to_segment(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    """
    Distance from point P to segment AB (2D).
    Standard projection + clamping approach.
    """
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    ab2 = abx * abx + aby * aby
    if ab2 <= 1e-12:
        # A and B are the same point
        return math.hypot(px - ax, py - ay)

    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))

    cx = ax + t * abx
    cy = ay + t * aby
    return math.hypot(px - cx, py - cy)


def point_to_polyline_distance(px: float, py: float, poly: Sequence[Point2D]) -> float:
    """
    Compute the shortest distance from a point P(px,py) to a polyline.

    Used as "lateral deviation" to evaluate how far the robot deviates from the plan.
    """
    if not poly:
        return float("inf")
    if len(poly) == 1:
        return math.hypot(px - poly[0][0], py - poly[0][1])

    best = float("inf")
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i + 1]
        d = _dist_point_to_segment(px, py, ax, ay, bx, by)
        if d < best:
            best = d

    return best