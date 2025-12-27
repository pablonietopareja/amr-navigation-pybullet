import os
import csv
import time
import math
from typing import List, Optional, Tuple

import pybullet as p

from .planner_astar import plan_path_waypoints
from .nav_utils import (
    wrap_angle,
    simplify_waypoints_collinear,
    downsample_waypoints,
    pure_pursuit_target,
    polyline_length,
    point_to_polyline_distance,
)
from .controller import drive_to_target_continuous, hard_stop


class MissionRunner:
    """
    MissionRunner orchestrates a full navigation mission over multiple targets.

    Features:
    - A* planning per segment (start -> target_i)
    - Particle Filter localization (predict with odometry, update with simulated scan)
    - Pure pursuit waypoint following + low-level controller
    - Logging for plots
    - Dynamic obstacle handling via 'Workers' object:
        * Detect workers via ray tests (lidar-like)
        * If a worker is too close -> stop & wait
        * Resume once the path is clear

    IMPORTANT DESIGN CHOICE:
    - PF prediction uses odometry increments (delta_d, delta_theta).
    - PF update uses scan_map.scan(...), which is a *map-based simulated lidar*.
      In this project, the measurement is generated from the *true pose* in simulation
      (true_x, true_y, true_theta) to emulate what a real lidar would measure.
      This is NOT used directly for control. Control uses PF estimated pose.
    """

    def __init__(
        self,
        world,
        robot,
        odom,
        pf,
        scan_map,
        spawn_xy: Tuple[float, float],
        targets: List[Tuple[float, float]],
        logs_dir: str = "logs",
        inflation_cells: int = 2,
        allow_diagonal: bool = False,
        wp_reached_tol: float = 0.7,
        lookahead: float = 1.8,
        downsample_dist: float = 1.0,
        log_interval_sec: float = 1.0,
        v_max_mps: float = 0.9,
        # ---- workers / humans ----
        workers=None,
        worker_stop_dist_m: float = 1.6,
        worker_check_every_steps: int = 1,
        worker_ray_z: float = 0.25,
    ):
        # Core objects
        self.world = world
        self.robot = robot
        self.odom = odom
        self.pf = pf
        self.scan_map = scan_map

        # Mission definition
        self.spawn_xy = tuple(spawn_xy)
        self.targets = list(targets)

        # Logging
        self.logs_dir = logs_dir
        os.makedirs(self.logs_dir, exist_ok=True)

        # Planner / controller parameters
        self.inflation_cells = int(inflation_cells)
        self.allow_diagonal = bool(allow_diagonal)
        self.wp_reached_tol = float(wp_reached_tol)
        self.lookahead = float(lookahead)
        self.downsample_dist = float(downsample_dist)
        self.log_interval_sec = float(log_interval_sec)
        self.v_max_mps = float(v_max_mps)

        # Workers (dynamic obstacles)
        self.workers = workers
        self.worker_stop_dist_m = float(worker_stop_dist_m)
        self.worker_check_every_steps = max(1, int(worker_check_every_steps))
        self.worker_ray_z = float(worker_ray_z)

        # In-memory logs for plots
        self.planned_paths_rows = []   # [segment, wp_index, x, y]
        self.segment_summary_rows = [] # per target/segment summary
        self.nav_metrics_rows = []     # [time_s, segment, true_x, true_y, pf_x, pf_y, dist_path_true, dist_path_pf, dist_goal_pf]
        self.slam_rows = []            # slam_log.csv

        # Timing
        self.step_count = 0
        self.dt = p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0)
        self.steps_per_log = max(1, int(self.log_interval_sec / self.dt))

        # State used for worker stop/resume prints
        self._blocked = False
        self._last_block_print_t = -999.0

    # -------------------------------------------------------------------------
    # Planning
    # -------------------------------------------------------------------------

    def plan_segment(self, start_xy: Tuple[float, float], goal_xy: Tuple[float, float]):
        """
        Plan a segment path with A* and post-process waypoints.

        Returns:
            waypoints (list[(x,y)] or None),
            effective_goal (x,y),
            plan_time_s (float)
        """
        t0 = time.perf_counter()
        wps = plan_path_waypoints(
            scan_map=self.scan_map,
            start_xy=start_xy,
            goal_xy=goal_xy,
            inflation_cells=self.inflation_cells,
            allow_diagonal=self.allow_diagonal,
        )
        t_plan = time.perf_counter() - t0

        if not wps:
            return None, goal_xy, t_plan

        # Path post-processing:
        # - remove redundant collinear points
        # - downsample to reduce oscillations and make controller smoother
        wps = simplify_waypoints_collinear(wps)
        wps = downsample_waypoints(wps, min_dist=self.downsample_dist)

        effective_goal = wps[-1] if wps else goal_xy
        return wps, effective_goal, t_plan

    # -------------------------------------------------------------------------
    # Worker detection (dynamic obstacles)
    # -------------------------------------------------------------------------

    def _check_workers_blocking(self, x: float, y: float, theta: float):
        """
        Detect workers via PyBullet ray tests (lidar-like).

        NOTE:
        - PF and A* are still based on ScanableMap (static map), so workers do NOT
          directly modify PF weights or A* occupancy (unless you implement that later).
        - Here I only use workers detection to stop the robot safely.

        Returns:
            blocked (bool), worker_id (int|None), distance (float)
        """
        if self.workers is None:
            return False, None, float("inf")

        # Reduce CPU by checking only every N steps
        if self.step_count % self.worker_check_every_steps != 0:
            return self._blocked, None, float("inf")

        # ScanableMap stores fov in radians in your implementation
        fov_deg = math.degrees(getattr(self.scan_map, "fov", math.radians(360.0)))
        num_rays = int(getattr(self.scan_map, "num_rays", 36))
        max_range = float(getattr(self.scan_map, "max_distance", 10.0))

        hit, d, wid = self.workers.lidar_detect(
            robot_x=x,
            robot_y=y,
            robot_theta=theta,
            fov_deg=fov_deg,
            num_rays=num_rays,
            max_range=max_range,
            ray_z=self.worker_ray_z,
        )

        if hit and d <= self.worker_stop_dist_m:
            return True, wid, d
        return False, None, d

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    def run(self):
        """
        Execute the full mission:
        - Plan segment 1
        - Loop simulation steps:
            * odom update -> PF predict
            * (periodic) PF update + logging
            * check workers -> stop/wait if needed
            * waypoint following (pure pursuit + controller)
            * when goal reached -> log summary, plan next segment, continue
        """
        if not self.targets:
            print("[NAV] No targets provided. Nothing to do.")
            return

        segment_idx = 0
        active_goal = self.targets[segment_idx]

        # ---- initial planning ----
        waypoints, goal_effective, plan_time_s = self.plan_segment(self.spawn_xy, active_goal)
        if not waypoints:
            print(f"[A*] ERROR: No path found to target 1: {active_goal}")
            return

        # Store planned waypoints for plotting
        for i_wp, (wx, wy) in enumerate(waypoints):
            self.planned_paths_rows.append([segment_idx + 1, i_wp, wx, wy])

        print(
            f"[A*] Segment 1 planned. target={active_goal} effective_goal={goal_effective} "
            f"wps={len(waypoints)} plan_time={plan_time_s*1000.0:.1f} ms path_len={polyline_length(waypoints):.2f} m"
        )

        # Waypoint index initialization:
        wp_index = 0
        if waypoints and math.hypot(waypoints[0][0] - self.spawn_xy[0], waypoints[0][1] - self.spawn_xy[1]) < 0.6:
            wp_index = 1

        # Segment timing bookkeeping
        segment_started = False
        segment_start_time_s = None
        segment_start_xy = self.spawn_xy
        segment_plan_time_s = plan_time_s

        # PF current estimate
        pf_x, pf_y, pf_theta = self.pf.estimate()

        # ---- simulation loop ----
        while p.isConnected():
            self.world.simStep()

            # Mark segment start time as soon as I begin moving
            if (not segment_started) and waypoints:
                segment_started = True
                segment_start_time_s = self.step_count * self.dt
                segment_start_xy = (pf_x, pf_y)

            # =========================
            # ODOMETRY UPDATE
            # =========================
            prev_x, prev_y, prev_theta = self.odom.get_pose()
            self.odom.update(self.robot.agv_id)
            odom_x, odom_y, odom_theta = self.odom.get_pose()

            delta_d = math.hypot(odom_x - prev_x, odom_y - prev_y)
            delta_theta = wrap_angle(odom_theta - prev_theta)

            # =========================
            # PF PREDICTION (motion)
            # =========================
            self.pf.predict(delta_d, delta_theta)
            pf_x, pf_y, pf_theta = self.pf.estimate()

            # =========================
            # TRUE POSE (simulation)
            # =========================
            # Used for:
            #   1) logging (True vs PF)
            #   2) generating simulated scan measurement (see PF update below)
            pos, orn = p.getBasePositionAndOrientation(self.robot.agv_id)
            true_x, true_y = pos[0], pos[1]
            true_theta = p.getEulerFromQuaternion(orn)[2]

            # =========================
            # PF UPDATE + LOGGING (periodic)
            # =========================
            if self.step_count % self.steps_per_log == 0:
                # In a real robot, I'd use the real lidar measurement here.
                # In this simulation, ScanableMap.scan() gives a *map-based scan*.
                # I generate the measurement from the ground-truth pose to emulate a sensor.
                scan_distances, _ = self.scan_map.scan(true_x, true_y, true_theta)

                self.pf.update(scan_distances)
                pf_x, pf_y, pf_theta = self.pf.estimate()

                t_sim = self.step_count * self.dt

                # Main SLAM log (time, true pose, odom pose, pf pose, segment, target)
                self.slam_rows.append([
                    t_sim,
                    true_x, true_y, true_theta,
                    odom_x, odom_y, odom_theta,
                    pf_x, pf_y, pf_theta,
                    segment_idx + 1,
                    active_goal[0], active_goal[1],
                ])

                # Navigation metrics for evaluation
                dist_path_true = point_to_polyline_distance(true_x, true_y, waypoints)
                dist_path_pf = point_to_polyline_distance(pf_x, pf_y, waypoints)
                dist_goal_pf = math.hypot(goal_effective[0] - pf_x, goal_effective[1] - pf_y)

                self.nav_metrics_rows.append([
                    t_sim, segment_idx + 1,
                    true_x, true_y,
                    pf_x, pf_y,
                    dist_path_true, dist_path_pf,
                    dist_goal_pf,
                ])

            # =========================
            # WORKERS: STOP & WAIT
            # =========================
            blocked, wid, d = self._check_workers_blocking(pf_x, pf_y, pf_theta)
            if blocked:
                hard_stop(self.robot)
                self._blocked = True

                t_now = self.step_count * self.dt
                if t_now - self._last_block_print_t >= 1.0:
                    print(f"[HUMAN] ⛔ Worker detected at ~{d:.2f} m. Waiting...")
                    self._last_block_print_t = t_now

                self.step_count += 1
                continue
            else:
                if self._blocked:
                    print("[HUMAN] ✅ Path clear. Continuing.")
                self._blocked = False

            # =========================
            # PATH FOLLOWING
            # =========================
            if not waypoints:
                print("[NAV] No path. Stopping.")
                hard_stop(self.robot)
                break

            wp_index = max(0, min(len(waypoints) - 1, wp_index))

            # If close enough to current waypoint, advance to next
            if math.hypot(waypoints[wp_index][0] - pf_x, waypoints[wp_index][1] - pf_y) < self.wp_reached_tol:
                if wp_index < len(waypoints) - 1:
                    wp_index += 1

            # Pure pursuit target selection (lookahead point)
            tx, ty, new_idx = pure_pursuit_target(
                (pf_x, pf_y, pf_theta),
                waypoints,
                wp_index,
                lookahead=self.lookahead
            )
            wp_index = new_idx

            # Low-level control (uses PF pose only)
            drive_to_target_continuous(
                self.robot,
                pose=(pf_x, pf_y, pf_theta),
                target_xy=(tx, ty),
                wp_reached_tol=self.wp_reached_tol,
                v_max_mps=self.v_max_mps,
                k_w=3.0,
                w_max=1.8,
                turn_in_place_deg=35.0,
            )

            # =========================
            # SEGMENT COMPLETION
            # =========================
            if math.hypot(goal_effective[0] - pf_x, goal_effective[1] - pf_y) < self.wp_reached_tol:
                t_reached = self.step_count * self.dt
                hard_stop(self.robot)

                seg_id = segment_idx + 1

                # Extract metrics rows for this segment (simple, readable; not the most optimal, but fine)
                seg_metrics = [row for row in self.nav_metrics_rows if row[1] == seg_id]
                lat_errs_true = [row[6] for row in seg_metrics] if seg_metrics else []
                lat_errs_pf = [row[7] for row in seg_metrics] if seg_metrics else []

                mean_lat_true = (sum(lat_errs_true) / len(lat_errs_true)) if lat_errs_true else float("nan")
                max_lat_true = (max(lat_errs_true)) if lat_errs_true else float("nan")
                mean_lat_pf = (sum(lat_errs_pf) / len(lat_errs_pf)) if lat_errs_pf else float("nan")
                max_lat_pf = (max(lat_errs_pf)) if lat_errs_pf else float("nan")

                travel_time_s = (t_reached - segment_start_time_s) if (segment_start_time_s is not None) else float("nan")
                path_len_m = polyline_length(waypoints)

                # Save per-segment summary
                self.segment_summary_rows.append([
                    seg_id,
                    active_goal[0], active_goal[1],
                    segment_start_xy[0], segment_start_xy[1],
                    segment_plan_time_s,
                    path_len_m,
                    segment_start_time_s if segment_start_time_s is not None else 0.0,
                    t_reached,
                    travel_time_s,
                    mean_lat_true, max_lat_true,
                    mean_lat_pf, max_lat_pf,
                ])

                print(
                    f"[NAV] ✅ Target {seg_id} reached! "
                    f"travel_time={travel_time_s:.2f}s plan_time={segment_plan_time_s*1000.0:.1f}ms "
                    f"path_len={path_len_m:.2f}m mean_lat_err_true={mean_lat_true:.3f}m"
                )

                # Mission finished?
                if segment_idx >= len(self.targets) - 1:
                    print("[NAV] ✅ All targets completed.")
                    break

                # =========================
                # NEXT SEGMENT PLANNING
                # =========================
                segment_idx += 1
                active_goal = self.targets[segment_idx]
                start_xy = (pf_x, pf_y)

                waypoints, goal_effective, segment_plan_time_s = self.plan_segment(start_xy, active_goal)
                if not waypoints:
                    print(f"[A*] ERROR: No path found to target {segment_idx+1}: {active_goal}")
                    hard_stop(self.robot)
                    break

                # Append waypoints for plotting
                self.planned_paths_rows.extend([
                    [segment_idx + 1, i_wp, wx, wy] for i_wp, (wx, wy) in enumerate(waypoints)
                ])

                # Reset waypoint index for the new segment
                wp_index = 0
                if waypoints and math.hypot(waypoints[0][0] - start_xy[0], waypoints[0][1] - start_xy[1]) < 0.6:
                    wp_index = 1

                # Reset segment timers
                segment_started = False
                segment_start_time_s = None
                segment_start_xy = start_xy

                print(
                    f"[A*] Segment {segment_idx+1} planned. target={active_goal} effective_goal={goal_effective} "
                    f"wps={len(waypoints)} plan_time={segment_plan_time_s*1000.0:.1f} ms path_len={polyline_length(waypoints):.2f} m"
                )

            self.step_count += 1

    # -------------------------------------------------------------------------
    # Log saving
    # -------------------------------------------------------------------------

    def save_logs(self):
        """
        Persist all logs to CSV in logs_dir.
        These files are consumed by plot_path.py and plot_nav_metrics.py.
        """
        if self.slam_rows:
            path = os.path.join(self.logs_dir, "slam_log.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time_s",
                    "x_true", "y_true", "theta_true",
                    "x_odom", "y_odom", "theta_odom",
                    "x_pf", "y_pf", "theta_pf",
                    "segment",
                    "target_x", "target_y",
                ])
                w.writerows(self.slam_rows)
            print(f"{len(self.slam_rows)} samples saved to {path}")

        if self.planned_paths_rows:
            path = os.path.join(self.logs_dir, "planned_paths.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["segment", "wp_index", "x", "y"])
                w.writerows(self.planned_paths_rows)
            print(f"{len(self.planned_paths_rows)} waypoint samples saved to {path}")

        if self.nav_metrics_rows:
            path = os.path.join(self.logs_dir, "nav_metrics.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "time_s",
                    "segment",
                    "x_true", "y_true",
                    "x_pf", "y_pf",
                    "dist_to_path_true_m",
                    "dist_to_path_pf_m",
                    "dist_to_goal_pf_m",
                ])
                w.writerows(self.nav_metrics_rows)
            print(f"{len(self.nav_metrics_rows)} metric samples saved to {path}")

        if self.segment_summary_rows:
            path = os.path.join(self.logs_dir, "targets_summary.csv")
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "segment",
                    "target_x", "target_y",
                    "start_x", "start_y",
                    "plan_time_s",
                    "path_length_m",
                    "t_start_s",
                    "t_reached_s",
                    "travel_time_s",
                    "mean_lat_err_true_m",
                    "max_lat_err_true_m",
                    "mean_lat_err_pf_m",
                    "max_lat_err_pf_m",
                ])
                w.writerows(self.segment_summary_rows)
            print(f"{len(self.segment_summary_rows)} rows saved to {path}")