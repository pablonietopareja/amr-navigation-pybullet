from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Tuple, Set

Cell = Tuple[int, int]          # (x, y) grid cell indices
WayPoint = Tuple[float, float]  # (x, y) in meters (cell centers)


# =============================================================================
# Grid helpers
# =============================================================================

def _in_bounds(grid: List[List[int]], x: int, y: int) -> bool:
    """Return True if (x, y) is within the grid bounds."""
    return 0 <= y < len(grid) and 0 <= x < len(grid[0])


def _is_free(grid: List[List[int]], x: int, y: int) -> bool:
    """Return True if the cell is inside bounds and not an obstacle."""
    return _in_bounds(grid, x, y) and grid[y][x] == 0


def make_occupancy_from_scan_map(scan_map) -> List[List[int]]:
    """
    Convert ScanableMap.grid into a Python occupancy grid.

    Convention:
      - 0 = free
      - 1 = obstacle

    ScanableMap.grid stores 1 for obstacles (grid[y, x]).
    """
    grid = scan_map.grid
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    # Build as a list-of-lists (kept compatible with the rest of the module)
    occ: List[List[int]] = []
    for y in range(rows):
        # Convert any positive value to 1, else 0
        occ.append([1 if grid[y][x] > 0 else 0 for x in range(cols)])
    return occ


def inflate_grid(occ: List[List[int]], radius_cells: int) -> List[List[int]]:
    """
    Inflate obstacles by 'radius_cells' cells.

    This is a classic way to add safety margin so paths do not graze walls.
    """
    if radius_cells <= 0:
        return [row[:] for row in occ]

    rows = len(occ)
    cols = len(occ[0]) if rows else 0
    out = [row[:] for row in occ]

    r = radius_cells

    # Precompute square offsets (dx, dy) once (micro-optimization)
    offsets = [(dx, dy) for dy in range(-r, r + 1) for dx in range(-r, r + 1)]

    for y in range(rows):
        for x in range(cols):
            if occ[y][x] != 1:
                continue

            # Mark all cells in the square window as obstacle
            for dx, dy in offsets:
                xx = x + dx
                yy = y + dy
                if 0 <= xx < cols and 0 <= yy < rows:
                    out[yy][xx] = 1

    return out


def world_to_cell(x: float, y: float, cols: int, rows: int) -> Cell:
    """
    Convert a world coordinate (meters) into a grid cell (integer indices).

    I use floor() so:
      5.99 -> 5
      5.00 -> 5
    Then clamp inside [0..cols-1] / [0..rows-1].
    """
    cx = int(math.floor(x))
    cy = int(math.floor(y))
    cx = max(0, min(cols - 1, cx))
    cy = max(0, min(rows - 1, cy))
    return (cx, cy)


def cell_to_world_center(cell: Cell) -> WayPoint:
    """Convert a grid cell index to the corresponding world-space cell center."""
    cx, cy = cell
    return (cx + 0.5, cy + 0.5)


def cells_to_waypoints(cells: List[Cell]) -> List[WayPoint]:
    """Convert a list of cells into world-space waypoint centers."""
    return [cell_to_world_center(c) for c in cells]


def _heuristic(a: Cell, b: Cell) -> float:
    """
    A* heuristic (Euclidean distance).
    Works well for grid movement (especially if diagonals are allowed).
    """
    return math.hypot(b[0] - a[0], b[1] - a[1])


# =============================================================================
# Start/Goal snapping
# =============================================================================

def _snap_to_nearest_free(occ: List[List[int]], cell: Cell, max_radius: int = 50) -> Optional[Cell]:
    """
    If 'cell' is inside an obstacle, find the nearest free cell using BFS.

    Why this is important:
    - The robot or target might fall inside an inflated wall cell.
    - Snapping makes planning more robust and avoids returning "no path".

    BFS is done in a 4-neighborhood (up/down/left/right), which is enough for snapping.
    """
    if _is_free(occ, cell[0], cell[1]):
        return cell

    q = deque([cell])
    visited: Set[Cell] = set([cell])

    # 4-connected BFS neighbors
    moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        cx, cy = q.popleft()

        # Stop runaway searches (Manhattan radius)
        if abs(cx - cell[0]) + abs(cy - cell[1]) > max_radius:
            continue

        for dx, dy in moves:
            nx, ny = cx + dx, cy + dy
            nxt = (nx, ny)
            if nxt in visited:
                continue
            visited.add(nxt)

            if not _in_bounds(occ, nx, ny):
                continue

            if _is_free(occ, nx, ny):
                return nxt

            q.append(nxt)

    return None


# =============================================================================
# A* search
# =============================================================================

def astar(
    occ: List[List[int]],
    start: Cell,
    goal: Cell,
    allow_diagonal: bool = True,
) -> Optional[List[Cell]]:
    """
    Run A* on a binary occupancy grid.

    Parameters
    ----------
    occ:
        2D list where 0=free, 1=obstacle
    start, goal:
        (x, y) integer cell coordinates
    allow_diagonal:
        If True, use 8-connected neighbors. If False, 4-connected neighbors.

    Returns
    -------
    path (list of Cell) or None if no path exists.
    """
    if not _is_free(occ, start[0], start[1]) or not _is_free(occ, goal[0], goal[1]):
        return None

    if allow_diagonal:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                     (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Priority queue items are (f_cost, cell)
    open_heap: List[Tuple[float, Cell]] = []
    heapq.heappush(open_heap, (_heuristic(start, goal), start))

    came_from: Dict[Cell, Cell] = {}
    g_cost: Dict[Cell, float] = {start: 0.0}

    # Closed set prevents re-expanding the same node many times
    closed: Set[Cell] = set()

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            # Reconstruct path by walking backward through came_from
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            if not _is_free(occ, nx, ny):
                continue

            step_cost = math.hypot(dx, dy)
            new_g = g_cost[current] + step_cost
            nxt = (nx, ny)

            # Standard A* relaxation
            if nxt not in g_cost or new_g < g_cost[nxt]:
                g_cost[nxt] = new_g
                came_from[nxt] = current
                f = new_g + _heuristic(nxt, goal)
                heapq.heappush(open_heap, (f, nxt))

    return None


# =============================================================================
# Project wrapper: plan in meters and return waypoints
# =============================================================================

def plan_path_waypoints(
    scan_map,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    inflation_cells: int = 1,
    allow_diagonal: bool = True,
) -> Optional[List[WayPoint]]:
    """
    Project wrapper used by MissionRunner:

    - Reads obstacles from ScanableMap.grid
    - Inflates obstacles (safety margin)
    - Converts start/goal from meters -> grid cells
    - If start/goal are inside obstacles, snap to nearest free cell
    - Runs A*
    - Returns waypoints as world-space cell centers

    Assumption:
      1 grid cell == 1 meter (same as ScanableMap).
    """
    occ = make_occupancy_from_scan_map(scan_map)
    rows = len(occ)
    cols = len(occ[0]) if rows else 0
    if cols == 0 or rows == 0:
        return None

    if inflation_cells > 0:
        occ = inflate_grid(occ, inflation_cells)

    start = world_to_cell(start_xy[0], start_xy[1], cols, rows)
    goal = world_to_cell(goal_xy[0], goal_xy[1], cols, rows)

    # Snapping prevents "no path" when the robot/goal is inside inflated obstacles.
    start2 = _snap_to_nearest_free(occ, start, max_radius=60)
    goal2 = _snap_to_nearest_free(occ, goal, max_radius=120)
    if start2 is None or goal2 is None:
        return None

    path_cells = astar(occ, start2, goal2, allow_diagonal=allow_diagonal)
    if path_cells is None:
        return None

    return cells_to_waypoints(path_cells)