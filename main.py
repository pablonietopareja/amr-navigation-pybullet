"""
main.py — Project entry point

This script boots the PyBullet simulation, loads the ASCII map, spawns the robot,
initializes odometry + particle filter localization, optionally spawns dynamic
"worker" obstacles, and runs a multi-target mission using the MissionRunner.

Key idea:
- The robot CONTROL uses the PF estimate.
- Ground truth pose is only used inside the simulation for:
  (1) generating simulated lidar measurements,
  (2) logging / evaluation / plotting.
"""

import math
from pathlib import Path

import pybullet as p

import utils.world as W
import utils.demo_robot as R
from utils.demo_odometry import DifferentialDriveOdometry
from utils.particle_filter import ParticleFilter
from utils.ScannableMap import ScanableMap

from utils.nav_utils import load_map_size
from utils.mission_runner import MissionRunner
from utils.workers import Workers


# =============================================================================
# Configuration
# =============================================================================

# Map file (ASCII grid). "1" and "2" are obstacles with different textures.
MAP_FILE = "maps/myMap"

# Lidar simulation parameters for the particle filter update step
FOV_DEG = 360.0
MAX_RANGE = 10.0
NUM_RAYS = 36
MAP_SCALE = 1  # map-to-world scaling factor used by ScanableMap

# Particle Filter parameters
N_PARTICLES = 1000
MOTION_STD_LIN = 0.03               # motion noise (linear)
MOTION_STD_ROT = math.radians(2)    # motion noise (angular)
SCAN_STD = 0.3                      # measurement noise for lidar matching

# Robot initial position
SPAWN = (5.0, 5.0)

# A* planner configuration
INFLATION_CELLS = 2            # obstacle inflation in grid cells
ALLOW_DIAGONAL = False         # A* 4-connected grid when False

# Path tracking / control tuning
WP_REACHED_TOL = 0.7           # how close to a waypoint/goal is considered "reached"
LOOKAHEAD = 3                  # pure-pursuit lookahead distance (old: 1.8)
DOWNSAMPLE_DIST = 1.0          # waypoint downsampling spacing in meters
LOG_INTERVAL_SEC = 0.33        # log interval (old: 1.0)
V_MAX_MPS = 1.8                # max linear speed command (old: 0.9)

# Dynamic workers (moving spheres) safety behavior
WORKER_STOP_DIST_M = 2.5       # robot stops if a worker is detected closer than this
WORKER_CHECK_EVERY_STEPS = 1   # how often to check worker detection (in sim steps)

# Logs folder (CSV outputs used by plot scripts)
LOGS_DIR = "logs"


# =============================================================================
# Helpers
# =============================================================================

def build_targets(map_w: int, map_h: int):
    """
    Define the 3 mission targets in world coordinates.
    Here I place them relative to the map size so targets adapt if the map changes.
    """
    return [
        (map_w - 10.0, 10.0),
        (map_w - 10.0, map_h - 10.0),
        (10.0, map_h - 10.0),
    ]


# =============================================================================
# Main entry point
# =============================================================================

def main() -> None:
    # --- Basic file/folder sanity checks (safe, no behavior change) ---
    map_path = Path(MAP_FILE)
    if not map_path.exists():
        raise FileNotFoundError(f"Map file not found: {MAP_FILE}")

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)

    # Load map size (used for targets and ScanableMap dimensions)
    map_w, map_h = load_map_size(MAP_FILE)

    # Mission targets
    targets = build_targets(map_w, map_h)

    # =========================================================================
    # WORLD + MAP (PyBullet environment)
    # =========================================================================
    wld = W.BaseWorld()
    wld.map_from_file(MAP_FILE)
    wld.texture_walls()

    # =========================================================================
    # ROBOT (Differential drive demo robot)
    # =========================================================================
    # NOTE: DemoRobot internally handles wheel joints and applies motor commands
    # when its step_action() callback is executed every simulation tick.
    robot = R.DemoRobot(x=SPAWN[0], y=SPAWN[1])
    wld.add_step_callback(robot.step_action)  # critical: applies motor commands

    # =========================================================================
    # WORKERS (Dynamic obstacles)
    # =========================================================================
    # Two red textured spheres moving back-and-forth.
    # You can edit their start coordinates in utils/workers.py.
    workers = Workers()
    wld.add_step_callback(workers.step)

    # =========================================================================
    # ODOMETRY
    # =========================================================================
    # Uses wheel joint states to estimate (x, y, theta) over time.
    odom = DifferentialDriveOdometry(
        wheel_radius=robot.wheel_radius,
        wheel_base=robot.wheel_base,
        left_wheel_joints=[robot.left_joint],
        right_wheel_joints=[robot.right_joint],
        # Calibration constants (already tuned in the previous project)
        k_dist=1.060,
        k_rot=0.190,
    )

    # Initialize odometry pose from the simulation start pose
    init_pos, init_orn = p.getBasePositionAndOrientation(robot.agv_id)
    init_theta = p.getEulerFromQuaternion(init_orn)[2]
    odom.reset(x=init_pos[0], y=init_pos[1], theta=init_theta)

    # =========================================================================
    # SCANNABLE MAP (for simulated lidar rays and PF measurement model)
    # =========================================================================
    scan_map = ScanableMap(
        map_dimensions=(map_w, map_h),
        fov=FOV_DEG,
        max_distance=MAX_RANGE,
        num_rays=NUM_RAYS,
        scale=MAP_SCALE,
    )
    scan_map.load_map(MAP_FILE)

    # =========================================================================
    # PARTICLE FILTER (Localization)
    # =========================================================================
    # PF state estimate is what the controller uses for navigation.
    pf = ParticleFilter(
        scan_map=scan_map,
        n_particles=N_PARTICLES,
        init_x_range=(0.0, map_w),
        init_y_range=(0.0, map_h),
        init_theta_range=(-math.pi, math.pi),
        motion_std_lin=MOTION_STD_LIN,
        motion_std_rot=MOTION_STD_ROT,
        scan_std=SCAN_STD,
    )
    pf.reset(init_pose=(init_pos[0], init_pos[1], init_theta))

    # =========================================================================
    # MISSION RUNNER (Planning + Control + Logging)
    # =========================================================================
    runner = MissionRunner(
        world=wld,
        robot=robot,
        odom=odom,
        pf=pf,
        scan_map=scan_map,
        spawn_xy=SPAWN,
        targets=targets,
        logs_dir=LOGS_DIR,
        # planning
        inflation_cells=INFLATION_CELLS,
        allow_diagonal=ALLOW_DIAGONAL,
        # control / waypoint following
        wp_reached_tol=WP_REACHED_TOL,
        lookahead=LOOKAHEAD,
        downsample_dist=DOWNSAMPLE_DIST,
        v_max_mps=V_MAX_MPS,
        # logging cadence
        log_interval_sec=LOG_INTERVAL_SEC,
        # dynamic obstacles
        workers=workers,
        worker_stop_dist_m=WORKER_STOP_DIST_M,
        worker_check_every_steps=WORKER_CHECK_EVERY_STEPS,
    )

    print(f"Simulation running. SPAWN={SPAWN}, TARGETS={targets}")

    try:
        runner.run()
    except KeyboardInterrupt:
        print("Simulation interrupted by user (Ctrl+C).")
    finally:
        # Always persist logs even if the simulation is interrupted
        runner.save_logs()
        try:
            wld.end()
        except Exception as e:
            print(f"[WORLD] end() warning: {e}")


if __name__ == "__main__":
    main()