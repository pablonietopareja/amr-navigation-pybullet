"""
demo_robot.py

Defines a simple differential-drive robot model built directly in PyBullet
(no URDF needed), used for the lab simulations.

Robot model:
- Box chassis (base)
- Two revolute wheel joints (left/right) driven in VELOCITY_CONTROL
- One fixed "caster" sphere in front for stability

IMPORTANT CONVENTION (kept as-is):
- Forward motion corresponds to NEGATIVE wheel angular velocity.
  This is due to how the wheel cylinders are oriented in this model.
  The controller uses this same convention, so do NOT change it unless
  you also update the controller logic.
"""

import pybullet as p


def motor_noise(x: float) -> float:
    """
    Optional motor noise injection.

    Currently returns the input unchanged (no noise).
    If you want to simulate imperfect motors, you can add noise here.
    """
    return x


class DemoRobot:
    """
    Differential-drive robot model used in the PyBullet project.

    Coordinate convention (as documented by the original lab):
    - +X: robot forward direction
    - +Y: robot left side
    - Wheels rotate around the Z axis (after applying a visual rotation)
    """

    def __init__(self, x: float = 0.0, y: float = 0.0):
        # ============================================================
        # GEOMETRY / PHYSICAL PARAMETERS
        # ============================================================
        body_length = 1.2
        body_width = 0.8
        body_height = 0.3

        self.body_length = body_length
        self.body_width = body_width
        self.body_height = body_height

        # Driving wheels
        wheel_radius = 0.15
        wheel_width = 0.1

        self.wheel_radius = wheel_radius
        self.wheel_width = wheel_width

        # Effective wheel-to-wheel distance (center-to-center)
        # Note: wheel_base is used in your controller/odometry
        self.wheel_base = self.body_width + self.wheel_width  # ≈ 0.9 m

        # ============================================================
        # CHASSIS: COLLISION + VISUAL SHAPE
        # ============================================================
        chassis_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[body_length / 2, body_width / 2, body_height / 2],
        )

        chassis_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[body_length / 2, body_width / 2, body_height / 2],
            rgbaColor=[0.2, 0.4, 0.8, 1.0],
        )

        # ============================================================
        # WHEELS: COLLISION + VISUAL SHAPE
        # ============================================================
        # Wheels are cylinders. In PyBullet, a cylinder is aligned along its local Y axis
        # by default for the visual shape, so I rotate it later (link_orientations).
        wheel_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            height=wheel_width,
        )

        wheel_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=wheel_radius,
            length=wheel_width,
            rgbaColor=[0.3, 0.3, 0.3, 1.0],
        )

        # ============================================================
        # FRONT CASTER (passive sphere, fixed joint)
        # ============================================================
        caster_radius = 0.08
        caster_collision = p.createCollisionShape(p.GEOM_SPHERE, radius=caster_radius)
        caster_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=caster_radius,
            rgbaColor=[0.1, 0.1, 0.1, 1.0],
        )

        # ============================================================
        # MULTIBODY LINK CONFIG
        # I create 3 links:
        #   0) left wheel (revolute)
        #   1) right wheel (revolute)
        #   2) caster wheel (fixed)
        # ============================================================
        link_masses = []
        link_collision_shapes = []
        link_visual_shapes = []
        link_positions = []
        link_orientations = []
        link_inertial_positions = []
        link_inertial_orientations = []
        link_parent_indices = []
        link_joint_types = []
        link_joint_axes = []

        # ------------------------------------------------------------
        # LEFT WHEEL (positive Y side)
        # ------------------------------------------------------------
        link_masses.append(1.0)
        link_collision_shapes.append(wheel_collision)
        link_visual_shapes.append(wheel_visual)

        # Link position is relative to the chassis frame
        link_positions.append([
            -body_length / 3,                 # a bit towards the rear
            body_width / 2 + wheel_width / 2, # left side
            -body_height / 2                  # at the bottom of chassis
        ])

        # Rotate cylinder so its axis aligns with Z (so it spins properly)
        link_orientations.append(p.getQuaternionFromEuler([1.57, 0, 0]))

        link_inertial_positions.append([0, 0, 0])
        link_inertial_orientations.append([0, 0, 0, 1])
        link_parent_indices.append(0)

        link_joint_types.append(p.JOINT_REVOLUTE)
        link_joint_axes.append([0, 0, 1])

        # ------------------------------------------------------------
        # RIGHT WHEEL (negative Y side)
        # ------------------------------------------------------------
        link_masses.append(1.0)
        link_collision_shapes.append(wheel_collision)
        link_visual_shapes.append(wheel_visual)

        link_positions.append([
            -body_length / 3,
            -body_width / 2 - wheel_width / 2,
            -body_height / 2
        ])

        link_orientations.append(p.getQuaternionFromEuler([1.57, 0, 0]))
        link_inertial_positions.append([0, 0, 0])
        link_inertial_orientations.append([0, 0, 0, 1])
        link_parent_indices.append(0)

        link_joint_types.append(p.JOINT_REVOLUTE)
        link_joint_axes.append([0, 0, 1])

        # ------------------------------------------------------------
        # FRONT CASTER (passive, fixed)
        # ------------------------------------------------------------
        link_masses.append(0.5)
        link_collision_shapes.append(caster_collision)
        link_visual_shapes.append(caster_visual)

        # This caster is placed at the front center
        # (You had an old commented offset here; kept the effective value as-is)
        link_positions.append([
            body_length / 2,
            0.0,
            -body_height / 2
        ])

        link_orientations.append([0, 0, 0, 1])
        link_inertial_positions.append([0, 0, 0])
        link_inertial_orientations.append([0, 0, 0, 1])
        link_parent_indices.append(0)

        link_joint_types.append(p.JOINT_FIXED)
        link_joint_axes.append([0, 0, 0])

        # ============================================================
        # CREATE THE MULTIBODY
        # ============================================================
        # Base position: z placed so wheels touch ground (wheel_radius) plus half chassis height.
        self.agv_id = p.createMultiBody(
            baseMass=50,
            baseCollisionShapeIndex=chassis_collision,
            baseVisualShapeIndex=chassis_visual,
            basePosition=[x, y, wheel_radius + body_height / 2],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shapes,
            linkVisualShapeIndices=link_visual_shapes,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_inertial_positions,
            linkInertialFrameOrientations=link_inertial_orientations,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axes,
        )

        # PyBullet assigns link joint indices in the creation order
        self.left_joint = 0
        self.right_joint = 1
        self.caster_joint = 2
        self.num_joints = p.getNumJoints(self.agv_id)

        # Caster: passive (no motor force)
        p.setJointMotorControl2(
            self.agv_id,
            self.caster_joint,
            p.VELOCITY_CONTROL,
            force=0,
        )

        # ============================================================
        # DYNAMICS TUNING
        # ============================================================
        # These values affect stability and slipping.
        p.changeDynamics(
            self.agv_id,
            -1,
            lateralFriction=1.0,
            rollingFriction=0.0,
            spinningFriction=0.0,
            linearDamping=0.3,
            angularDamping=0.3,
        )

        # Wheels: slightly higher friction, small rolling/spinning friction
        for j in (self.left_joint, self.right_joint):
            p.changeDynamics(
                self.agv_id,
                j,
                lateralFriction=1.2,
                rollingFriction=0.01,
                spinningFriction=0.01,
            )

        # Internal wheel speed commands (rad/s)
        self.left_speed = 0.0
        self.right_speed = 0.0

        # Max reference velocities used for manual keyboard teleop
        self.v_forward = 15.0
        self.v_reverse = 15.0
        self.v_turn = 10.0

    # ============================================================
    # KEYBOARD TELEOP (I/J/K/L)
    # ============================================================

    def user_control(self, keys):
        """
        Manual teleoperation:

        i: forward  (NEGATIVE wheel velocity in this robot model)
        k: backward
        j: rotate left
        l: rotate right

        'keys' is the dictionary returned by p.getKeyboardEvents().
        """
        key_i = ord("i")
        key_k = ord("k")
        key_j = ord("j")
        key_l = ord("l")

        # Forward → negative wheel velocity (kept as-is!)
        if key_i in keys and keys[key_i] & p.KEY_IS_DOWN:
            self.left_speed = -self.v_forward
            self.right_speed = -self.v_forward

        # Backward
        elif key_k in keys and keys[key_k] & p.KEY_IS_DOWN:
            self.left_speed = self.v_reverse
            self.right_speed = self.v_reverse

        # Rotate left (counter-clockwise)
        elif key_j in keys and keys[key_j] & p.KEY_IS_DOWN:
            self.left_speed = self.v_turn
            self.right_speed = -self.v_turn

        # Rotate right (clockwise)
        elif key_l in keys and keys[key_l] & p.KEY_IS_DOWN:
            self.left_speed = -self.v_turn
            self.right_speed = self.v_turn

        # No input → stop
        else:
            self.left_speed = 0.0
            self.right_speed = 0.0

    # ============================================================
    # APPLY MOTOR COMMANDS EACH SIMULATION STEP
    # ============================================================

    def step_action(self):
        """
        Called every simulation tick (typically via world.add_step_callback()).
        Applies wheel angular velocities using PyBullet VELOCITY_CONTROL.

        This is the only place where wheel commands are actually sent to PyBullet.
        """
        if not p.isConnected():
            return

        # Left wheel motor command
        p.setJointMotorControl2(
            self.agv_id,
            self.left_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=motor_noise(self.left_speed),
            force=40.0,   # motor torque limit (N*m equivalent)
        )

        # Right wheel motor command
        p.setJointMotorControl2(
            self.agv_id,
            self.right_joint,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=motor_noise(self.right_speed),
            force=40.0,
        )