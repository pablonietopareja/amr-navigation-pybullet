"""
plot_path.py

Visualizes:
- The ASCII map as a background overlay (obstacles):
    '1' -> dark/black wall blocks
    '2' -> "cardboard" colored blocks
- True robot trajectory
- Particle Filter (PF) estimated trajectory
- Planned A* paths (per segment)
- Start position, targets, and reached markers

IMPORTANT:
If a target lies inside a wall cell, that specific wall cell is NOT drawn so the target marker remains visible.
"""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


# =============================================================================
# Configuration
# =============================================================================
LOG_DIR = "logs"
SLAM_LOG_CSV = os.path.join(LOG_DIR, "slam_log.csv")
TARGETS_SUMMARY_CSV = os.path.join(LOG_DIR, "targets_summary.csv")
PLANNED_PATHS_CSV = os.path.join(LOG_DIR, "planned_paths.csv")

MAP_FILE = os.path.join("maps", "myMap")   # ASCII map file (your grid map)
CELL_SIZE = 1.0                            # 1 cell = 1 meter (map/world units)


# =============================================================================
# CSV helpers
# =============================================================================
def read_csv_as_dicts(path: str):
    """Read a CSV file into a list of dict rows."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def closest_time_index(times, t):
    """
    Return the index i such that times[i] is closest to t.
    Used to find a 'reached' point in the recorded time series.
    """
    if not times:
        return None
    best_i = 0
    best_d = abs(times[0] - t)
    for i, ti in enumerate(times):
        d = abs(ti - t)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


# =============================================================================
# Map helpers
# =============================================================================
def load_ascii_map(map_path: str):
    """
    Load an ASCII map as a list of equal-length strings.
    I normalize line lengths by padding missing cells with '0' (free space).
    Returns: (map_lines, width, height)
    """
    with open(map_path, "r") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip("\n")]

    h = len(lines)
    w = max((len(ln.strip()) for ln in lines), default=0)

    norm = []
    for ln in lines:
        s = ln.strip()
        if len(s) < w:
            s = s + ("0" * (w - len(s)))
        norm.append(s)

    return norm, w, h


def point_inside_cell(tx, ty, col, row, cell_size=1.0, eps=1e-9):
    """
    Check whether a point (tx, ty) lies inside the cell [col,row].
    This is used to hide the wall cell underneath a target marker.
    """
    x0 = col * cell_size
    y0 = row * cell_size
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    return (tx >= x0 - eps) and (tx <= x1 + eps) and (ty >= y0 - eps) and (ty <= y1 + eps)


def build_obstacle_patches(map_lines, targets_xy, cell_size=1.0):
    """
    Build two lists of matplotlib Rectangle patches for obstacles:
      - patches_1 for '1' cells
      - patches_2 for '2' cells

    If any target falls inside a wall cell, I skip drawing that cell so
    the target marker stays visible.
    """
    patches_1 = []
    patches_2 = []

    h = len(map_lines)
    w = len(map_lines[0]) if h else 0

    for row in range(h):
        ln = map_lines[row]
        for col in range(w):
            c = ln[col]
            if c not in ("1", "2"):
                continue

            # If a target lies in this cell, do NOT draw the wall block
            skip_cell = False
            for (tx, ty) in targets_xy:
                if point_inside_cell(tx, ty, col, row, cell_size=cell_size):
                    skip_cell = True
                    break
            if skip_cell:
                continue

            x0 = col * cell_size
            y0 = row * cell_size
            rect = Rectangle((x0, y0), cell_size, cell_size)

            if c == "1":
                patches_1.append(rect)
            else:
                patches_2.append(rect)

    return patches_1, patches_2


# =============================================================================
# Load required files
# =============================================================================
if not os.path.exists(SLAM_LOG_CSV):
    raise FileNotFoundError(f"Missing {SLAM_LOG_CSV}. Run the simulation first.")

if not os.path.exists(MAP_FILE):
    raise FileNotFoundError(f"Missing {MAP_FILE}. Make sure the map exists in maps/.")

rows = read_csv_as_dicts(SLAM_LOG_CSV)
if not rows:
    raise RuntimeError(f"{SLAM_LOG_CSV} exists but is empty.")


# =============================================================================
# Parse slam_log.csv (trajectories)
# =============================================================================
time_s = []
x_true, y_true = [], []
x_pf, y_pf = [], []

# Optional: slam_log may store target_x/target_y each sample (useful for fallback)
target_x_samples, target_y_samples = [], []

for r in rows:
    time_s.append(float(r["time_s"]))
    x_true.append(float(r["x_true"]))
    y_true.append(float(r["y_true"]))
    x_pf.append(float(r["x_pf"]))
    y_pf.append(float(r["y_pf"]))

    if "target_x" in r and "target_y" in r:
        target_x_samples.append(float(r["target_x"]))
        target_y_samples.append(float(r["target_y"]))


# =============================================================================
# Parse targets summary (preferred) OR fallback to unique targets in slam_log
# =============================================================================
targets = []
if os.path.exists(TARGETS_SUMMARY_CSV):
    trows = read_csv_as_dicts(TARGETS_SUMMARY_CSV)
    for tr in trows:
        targets.append(
            {
                "segment": int(float(tr["segment"])),
                "target_x": float(tr["target_x"]),
                "target_y": float(tr["target_y"]),
                "t_start_s": float(tr["t_start_s"]),
                "t_reached_s": float(tr["t_reached_s"]),
            }
        )
else:
    # Fallback: infer unique targets from slam_log.csv (less reliable)
    if target_x_samples and target_y_samples:
        seen = set()
        for sx, sy in zip(target_x_samples, target_y_samples):
            key = (round(sx, 3), round(sy, 3))
            if key not in seen:
                seen.add(key)
                targets.append(
                    {
                        "segment": len(targets) + 1,
                        "target_x": sx,
                        "target_y": sy,
                        "t_start_s": 0.0,
                        "t_reached_s": None,
                    }
                )

targets_xy = [(t["target_x"], t["target_y"]) for t in targets]


# =============================================================================
# Parse planned paths (optional)
# =============================================================================
paths_by_segment = defaultdict(list)  # seg -> [(x,y), ...]
if os.path.exists(PLANNED_PATHS_CSV):
    prows = read_csv_as_dicts(PLANNED_PATHS_CSV)
    for pr in prows:
        seg = int(float(pr["segment"]))
        wx = float(pr["x"])
        wy = float(pr["y"])
        paths_by_segment[seg].append((wx, wy))


# =============================================================================
# Build map overlay (obstacle patches)
# =============================================================================
map_lines, map_w, map_h = load_ascii_map(MAP_FILE)
patches_1, patches_2 = build_obstacle_patches(map_lines, targets_xy, cell_size=CELL_SIZE)


# =============================================================================
# Plot
# =============================================================================
plt.figure()
ax = plt.gca()

# --- MAP overlay ---
# '1' cells -> darker black walls
if patches_1:
    pc1 = PatchCollection(
        patches_1,
        facecolor="black",
        edgecolor="black",
        linewidths=0.0,
        alpha=0.40,   # darker visibility
        zorder=0,
    )
    ax.add_collection(pc1)

# '2' cells -> cardboard color
if patches_2:
    pc2 = PatchCollection(
        patches_2,
        facecolor="#C8A26A",  # cardboard tone
        edgecolor="#8B6B3E",  # subtle border
        linewidths=0.0,
        alpha=0.35,
        zorder=1,
    )
    ax.add_collection(pc2)

# --- Trajectories ---
plt.plot(x_true, y_true, linewidth=2, label="True trajectory", zorder=4)
plt.plot(x_pf, y_pf, linewidth=2, linestyle="--", label="PF estimated trajectory", zorder=4)

# --- Planned paths per segment ---
for seg, pts in sorted(paths_by_segment.items()):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    plt.plot(xs, ys, linewidth=1, linestyle=":", label=f"Planned path (seg {seg})", zorder=3)

# --- Start marker ---
plt.scatter(x_true[0], y_true[0], s=80, marker="o", label="Start", zorder=6)

# --- Targets + reached markers ---
for t in targets:
    seg = t["segment"]
    tx, ty = t["target_x"], t["target_y"]

    plt.scatter(tx, ty, s=110, marker="X", label=f"Target {seg}", zorder=7)
    plt.text(tx + 0.2, ty + 0.2, str(seg), zorder=8)

    # Plot the true pose at reach time (if available)
    if t.get("t_reached_s") is not None:
        idx = closest_time_index(time_s, t["t_reached_s"])
        if idx is not None:
            plt.scatter(x_true[idx], y_true[idx], s=70, marker="s", label=f"Reached (seg {seg})", zorder=7)

plt.xlabel("X position [m]")
plt.ylabel("Y position [m]")
plt.title("Trajectory + targets (True vs PF) + planned paths + map overlay")

# Show full map bounds (so map overlay is not cropped)
plt.xlim(0, map_w * CELL_SIZE)
plt.ylim(0, map_h * CELL_SIZE)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()