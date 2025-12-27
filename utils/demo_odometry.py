"""
demo_odometry.py

Differential-drive odometry for the DemoRobot (PyBullet).

What this module does:
- Reads wheel joint angles (encoders) from PyBullet.
- Computes incremental motion (delta_d, delta_theta).
- Integrates pose (x, y, theta) over time.

IMPORTANT (kept as-is):
- This project uses a sign convention where "forward" motion corresponds
  to NEGATIVE wheel angular velocity. Therefore, in odometry I flip the
  wheel angle increments by -1 to keep the estimated motion consistent.
"""

import os
import numpy as np
import pybullet as p
import pygame

# Optional: force pygame window position (only relevant if you enable 2D viz)
os.environ["SDL_VIDEO_WINDOW_POS"] = "%d,%d" % (0, 0)


# =============================================================================
# OPTIONAL 2D VISUALIZATION (not used in the current SLAM pipeline)
# =============================================================================

class OdometryVisualisation:
    """
    Optional pygame-based 2D visualisation for the ASCII map.

    This is NOT required for the SLAM / navigation setup.
    It can be used for quick debugging if you want a simple top-down map view.
    """

    def __init__(self, odo, dimensions, robotID):
        """
        Parameters
        ----------
        odo : DifferentialDriveOdometry
            Odometry instance.
        dimensions : (width, height)
            ASCII map size in cells.
        robotID : int
            PyBullet body id of the robot (not actively used in this class).
        """
        self.odo = odo
        self.robotID = robotID
        self.draw_boxes = []

        pygame.init()
        self.UNIT = 10  # pixels per cell

        self.displaysurface = pygame.display.set_mode(
            (dimensions[0] * self.UNIT, dimensions[1] * self.UNIT)
        )
        self.displaysurface.fill((0, 0, 0))
        pygame.display.flip()
        self.clock = pygame.time.Clock()

    def load_map(self, fname: str) -> None:
        """
        Load an ASCII map (e.g., maps/myMap) and store wall cells.

        Walls are any '1' or '2' characters.
        """
        xs = open(fname, "r").read().strip()
        row = 0
        for line in xs.split("\n"):
            col = 0
            for c in line.strip():
                if c == "1" or c == "2":
                    self.draw_boxes.append(
                        (col * self.UNIT, row * self.UNIT, self.UNIT, self.UNIT)
                    )
                col += 1
            row += 1

    def redraw(self) -> None:
        """
        Draw the wall cells.

        NOTE:
        This class currently does not draw the robot pose
        """
        self.displaysurface.fill((0, 0, 0))

        for rect in self.draw_boxes:
            pygame.draw.rect(self.displaysurface, (255, 255, 255), rect)

        pygame.display.flip()
        self.clock.tick(20)

    def pygame_event_clean(self) -> None:
        """Prevent pygame from freezing by processing internal events."""
        pygame.event.pump()


# =============================================================================
# DIFFERENTIAL DRIVE ODOMETRY
# =============================================================================

class DifferentialDriveOdometry:
    """
    Lightweight differential-drive odometry.

    Pipeline:
    1) Read wheel joint angles from PyBullet (left/right).
    2) Compute encoder increments (delta_left, delta_right).
    3) Convert to wheel travel distances:
          dist = delta_angle_rad * wheel_radius
       Apply calibration factor k_dist.
    4) Compute robot motion:
          delta_d = (dl + dr)/2
          delta_theta = (dr - dl)/wheel_base
       Apply calibration factor k_rot.
    5) Integrate pose (x, y, theta).

    Calibration:
    - k_dist compensates for systematic distance scaling error.
    - k_rot compensates for systematic rotation scaling error.
    """

    def __init__(
        self,
        wheel_radius: float,
        wheel_base: float,
        left_wheel_joints,
        right_wheel_joints,
        k_dist: float = 1.0,
        k_rot: float = 1.0,
    ):
        self.wheel_radius = float(wheel_radius)
        self.wheel_base = float(wheel_base)

        # Joints are passed as lists (even if single wheel per side)
        self.left_joints = list(left_wheel_joints)
        self.right_joints = list(right_wheel_joints)

        self.k_dist = float(k_dist)
        self.k_rot = float(k_rot)

        # Estimated pose
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # Last encoder readings (mean wheel angle per side)
        self.prev_left_pos = None
        self.prev_right_pos = None

        # Last motion increments (useful for PF motion model)
        self.last_delta_d = 0.0
        self.last_delta_theta = 0.0

    # -------------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------------

    def reset(self, x: float = 0.0, y: float = 0.0, theta: float = 0.0) -> None:
        """Reset odometry pose and clear encoder memory."""
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)

        self.prev_left_pos = None
        self.prev_right_pos = None

        self.last_delta_d = 0.0
        self.last_delta_theta = 0.0

    # -------------------------------------------------------------------------
    # Update
    # -------------------------------------------------------------------------

    def update(self, robot_id: int):
        """
        Update odometry by reading wheel joint angles from PyBullet.

        Returns (for debugging/analysis):
            left_wheels, right_wheels,
            delta_left, delta_right,
            delta_distance, delta_theta
        """
        # Read wheel angles (radians) from PyBullet joint states
        left_wheels = [p.getJointState(robot_id, j)[0] for j in self.left_joints]
        right_wheels = [p.getJointState(robot_id, j)[0] for j in self.right_joints]

        left_pos = float(np.mean(left_wheels))
        right_pos = float(np.mean(right_wheels))

        # First call: just store reference encoder values
        if self.prev_left_pos is None:
            self.prev_left_pos = left_pos
            self.prev_right_pos = right_pos
            return left_wheels, right_wheels, 0.0, 0.0, 0.0, 0.0

        # Encoder increments (radians)
        delta_left = left_pos - self.prev_left_pos
        delta_right = right_pos - self.prev_right_pos

        # IMPORTANT sign convention:
        # In this project, "forward" wheel velocity is NEGATIVE.
        # So I flip encoder increments here to keep odometry consistent.
        delta_left *= -1.0
        delta_right *= -1.0

        # Convert wheel rotation to linear distance (meters)
        left_distance = delta_left * self.wheel_radius * self.k_dist
        right_distance = delta_right * self.wheel_radius * self.k_dist

        # Differential-drive kinematics
        delta_distance = 0.5 * (left_distance + right_distance)
        delta_theta = (right_distance - left_distance) / self.wheel_base
        delta_theta *= self.k_rot

        # Integrate pose:
        # If rotation is tiny -> straight-line approx
        if abs(delta_theta) < 1e-9:
            delta_x = delta_distance * np.cos(self.theta)
            delta_y = delta_distance * np.sin(self.theta)
        else:
            # Circular arc integration
            R = delta_distance / delta_theta
            delta_x = R * (np.sin(self.theta + delta_theta) - np.sin(self.theta))
            delta_y = -R * (np.cos(self.theta + delta_theta) - np.cos(self.theta))

        self.x += float(delta_x)
        self.y += float(delta_y)
        self.theta += float(delta_theta)

        # Normalize theta to [-pi, pi]
        self.theta = float(np.arctan2(np.sin(self.theta), np.cos(self.theta)))

        # Store increments for external use (PF motion model)
        self.last_delta_d = float(delta_distance)
        self.last_delta_theta = float(delta_theta)

        # Update previous encoders
        self.prev_left_pos = left_pos
        self.prev_right_pos = right_pos

        return (
            left_wheels,
            right_wheels,
            float(delta_left),
            float(delta_right),
            self.last_delta_d,
            self.last_delta_theta,
        )

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------

    def get_pose(self):
        """Return current odometry estimate as (x, y, theta)."""
        return self.x, self.y, self.theta

    def get_pose_2d_transform(self):
        """
        Return 2D homogeneous transform matrix (3x3) for the odometry pose.
        Useful if wanted to transform points between frames.
        """
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        return np.array(
            [
                [c, -s, self.x],
                [s,  c, self.y],
                [0,  0, 1],
            ]
        )


# =============================================================================
# Debug helper
# =============================================================================

def debug_text(odo, r, wld=None):
    """
    Print odometry pose vs. ground-truth pose from PyBullet.

    NOTE:
    This is only for debugging prints and is NOT used by the controller.
    """
    x, y, theta = odo.get_pose()

    pos, orn = p.getBasePositionAndOrientation(r.agv_id)
    true_theta = p.getEulerFromQuaternion(orn)[2]

    print(
        f"Odom: x={x:.3f}, y={y:.3f}, θ={np.degrees(theta):.1f}°   "
        f"True: x={pos[0]:.3f}, y={pos[1]:.3f}, θ={np.degrees(true_theta):.1f}°"
    )

    return theta, true_theta