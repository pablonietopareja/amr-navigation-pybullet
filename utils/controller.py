"""
controller.py

Low-level velocity controller for the differential-drive robot.

This module is intentionally simple:
- It turns a pose estimate, target point, into left/right wheel angular velocities.
- It does NOT use ground-truth pose. Whatever I pass in as `pose` is what it uses

IMPORTANT (kept as-is):
- In this DemoRobot model, forward motion corresponds to NEGATIVE wheel velocity.
  This is why I flip the sign of wl/wr at the end.
"""

import math
import pybullet as p

from .nav_utils import wrap_angle, clamp


def drive_to_target_continuous(
    robot,
    pose,
    target_xy,
    wp_reached_tol: float = 0.7,
    v_max_mps: float = 0.9,
    k_w: float = 3.0,
    w_max: float = 1.8,
    turn_in_place_deg: float = 35.0,
):
    """
    Proportional heading controller for waypoint following.

    Inputs
    ------
    robot:
        DemoRobot instance. I will only set:
          - robot.left_speed  (rad/s)
          - robot.right_speed (rad/s)

    pose:
        (x, y, theta) pose estimate. In the main it is PF pose.

    target_xy:
        (tx, ty) target point (pure pursuit lookahead waypoint).

    Behavior
    --------
    - If close to target point -> stop.
    - Else compute desired heading to target.
    - Angular velocity w is proportional to heading error, clamped to w_max.
    - Linear speed v is reduced when heading error is large.
    - Convert (v, w) to wheel angular velocities (wl, wr).
    - Flip sign so that forward in this model is negative wheel velocity.
    """
    x, y, th = pose
    tx, ty = target_xy

    # Vector to target
    dx, dy = tx - x, ty - y
    dist = math.hypot(dx, dy)

    # Stop if I am close enough to the current target point
    if dist < wp_reached_tol:
        robot.left_speed = 0.0
        robot.right_speed = 0.0
        return

    # Desired heading to face the target point
    desired = math.atan2(dy, dx)
    err = wrap_angle(desired - th)

    # Proportional heading correction (angular velocity)
    w = clamp(k_w * err, -w_max, w_max)

    # If I am too misaligned, rotate in place (v=0)
    if abs(err) > math.radians(turn_in_place_deg):
        v = 0.0
    else:
        # Otherwise move forward, but reduce speed when slightly misaligned
        align = max(0.0, 1.0 - abs(err) / math.radians(55.0))
        v = v_max_mps * align

        # Also reduce speed when very close to the waypoint to avoid overshoot
        v = min(v, 0.7 * dist)

    # Robot geometry
    r_w = robot.wheel_radius
    L = robot.wheel_base

    # Convert (v, w) -> wheel angular velocities (rad/s)
    wl = (v - (w * L / 2.0)) / r_w
    wr = (v + (w * L / 2.0)) / r_w

    # IMPORTANT: forward motion is NEGATIVE wheel velocity for this model
    wl = -wl
    wr = -wr

    # Clamp wheel speeds to allowed max (robot.v_forward is your rad/s limit)
    wl = clamp(wl, -robot.v_forward, robot.v_forward)
    wr = clamp(wr, -robot.v_forward, robot.v_forward)

    # Store commands. They are applied each tick by robot.step_action().
    robot.left_speed = wl
    robot.right_speed = wr


def hard_stop(robot):
    """
    Immediately stop the robot in a "hard" way:
    - Set internal commands to zero.
    - Send zero target velocity with zero force to the motors.

    This is used when:
    - A worker is detected too close
    - A target is reached
    - No path is available / mission ends
    """
    robot.left_speed = 0.0
    robot.right_speed = 0.0

    # Also force motors to stop (no force)
    p.setJointMotorControl2(
        robot.agv_id,
        robot.left_joint,
        p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0,
    )
    p.setJointMotorControl2(
        robot.agv_id,
        robot.right_joint,
        p.VELOCITY_CONTROL,
        targetVelocity=0.0,
        force=0.0,
    )