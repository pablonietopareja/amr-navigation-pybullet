"""
workers.py

Creates two moving spherical objects ("workers") inside the PyBullet simulation.

Goal:
- Provide simple dynamic obstacles that are NOT part of the static ASCII map.
- The robot can "detect" them using lidar-like ray tests (p.rayTestBatch).
- Mission logic can then stop/wait when a worker is too close.

Implementation details:
- I DO NOT need an URDF for these workers:
  A sphere collision + visual shape created via createMultiBody is enough.
- Workers move in a ping-pong (back-and-forth) pattern:
  Worker 1 moves along Y between [Y_max - WORKER1_Y_DELTA, Y_max]
  Worker 2 moves along X between [X_min, X_min + WORKER2_X_DELTA]
"""

import os
import math
import pybullet as p


# =============================================================================
# Global configuration
# =============================================================================

# Start poses (x, y) for each worker sphere
WORKER1_START_XY = (24.5, 29.0)   # Worker 1 moves on Y
WORKER2_START_XY = (33.0, 16.0)   # Worker 2 moves on X

# Sphere geometry
WORKER_RADIUS = 0.45
WORKER_Z = WORKER_RADIUS  # place the sphere on the ground (z = radius)

# Linear movement speed (m/s)
WORKER_SPEED_MPS = 1.6

# Movement ranges (meters)
# NOTE: these deltas define the ping-pong boundaries.
WORKER1_Y_DELTA = 7.0   # Worker 1 goes down by 7m from its start Y, then returns
WORKER2_X_DELTA = 8.0   # Worker 2 goes up by 8m from its start X, then returns


class Workers:
    """
    Two moving spheres representing workers.

    Public API:
    - step(): move workers one simulation tick (call every sim step)
    - get_ids(): list of PyBullet body IDs
    - lidar_detect(...): ray-based detection of worker bodies

    NOTE:
    Detection here is done by checking if rayTestBatch hits one of the worker body IDs.
    This is "perfect classification" in simulation; in a real robot you would classify
    obstacles differently (e.g., compare expected map scan vs measured scan).
    """

    def __init__(
        self,
        worker1_xy=WORKER1_START_XY,
        worker2_xy=WORKER2_START_XY,
        radius=WORKER_RADIUS,
        speed_mps=WORKER_SPEED_MPS,
        texture_candidates=("textures/red.png", "../pictures/red.png", "pictures/red.png", "red.png"),
    ):
        # Store initial coordinates
        self.worker1_xy0 = tuple(worker1_xy)
        self.worker2_xy0 = tuple(worker2_xy)

        self.radius = float(radius)
        self.speed = float(speed_mps)

        # Ping-pong limits for worker 1 (Y axis)
        self.worker1_y_max = self.worker1_xy0[1]
        self.worker1_y_min = self.worker1_y_max - WORKER1_Y_DELTA

        # Ping-pong limits for worker 2 (X axis)
        self.worker2_x_min = self.worker2_xy0[0]
        self.worker2_x_max = self.worker2_x_min + WORKER2_X_DELTA

        # Movement direction (+1 or -1)
        # Worker 1 starts moving "down" (negative Y direction)
        self._dir1 = -1
        # Worker 2 starts moving "up" in X (positive X direction)
        self._dir2 = +1

        # Spawn bodies
        self.worker_ids = []
        self._tex_id = self._load_texture(texture_candidates)
        self._spawn()

        # Cache simulation dt (fallback to 1/240 if not available)
        self._dt = p.getPhysicsEngineParameters().get("fixedTimeStep", 1.0 / 240.0)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _load_texture(self, candidates):
        """
        Try multiple texture paths and return a textureUniqueId, or -1 if not found.
        We search both:
          - relative to this file (utils/...),
          - and as-is (current working directory).
        """
        base_dir = os.path.dirname(__file__)
        for rel in candidates:
            for pth in (os.path.join(base_dir, rel), rel):
                if os.path.exists(pth):
                    try:
                        return p.loadTexture(pth)
                    except Exception:
                        pass
        return -1

    def _spawn_one(self, x, y):
        """
        Spawn a static sphere (mass=0) with collision + visual shape.
        """
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=self.radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.radius, rgbaColor=[1, 1, 1, 1])

        body = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, WORKER_Z],
        )

        # Apply texture if loaded
        if self._tex_id != -1:
            try:
                p.changeVisualShape(body, -1, textureUniqueId=self._tex_id)
            except Exception:
                pass

        return body

    def _spawn(self):
        """
        Spawn both workers and keep their body IDs.
        """
        self.worker_ids = [
            self._spawn_one(self.worker1_xy0[0], self.worker1_xy0[1]),
            self._spawn_one(self.worker2_xy0[0], self.worker2_xy0[1]),
        ]

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def get_ids(self):
        """Return the PyBullet body IDs for the workers."""
        return list(self.worker_ids)

    def step(self):
        """
        Move workers one simulation tick.

        Worker 1: moves on Y in [y_min, y_max]
        Worker 2: moves on X in [x_min, x_max]
        """
        # If PyBullet is not connected (e.g., during shutdown), do nothing safely.
        if not p.isConnected():
            return

        # -------------------------
        # Worker 1 (Y ping-pong)
        # -------------------------
        wid1 = self.worker_ids[0]
        pos1, orn1 = p.getBasePositionAndOrientation(wid1)
        x1, y1, z1 = pos1

        y1 += self._dir1 * self.speed * self._dt

        # Bounce at limits
        if y1 <= self.worker1_y_min:
            y1 = self.worker1_y_min
            self._dir1 = +1
        elif y1 >= self.worker1_y_max:
            y1 = self.worker1_y_max
            self._dir1 = -1

        p.resetBasePositionAndOrientation(wid1, [x1, y1, z1], orn1)

        # -------------------------
        # Worker 2 (X ping-pong)
        # -------------------------
        wid2 = self.worker_ids[1]
        pos2, orn2 = p.getBasePositionAndOrientation(wid2)
        x2, y2, z2 = pos2

        x2 += self._dir2 * self.speed * self._dt

        # Bounce at limits
        if x2 <= self.worker2_x_min:
            x2 = self.worker2_x_min
            self._dir2 = +1
        elif x2 >= self.worker2_x_max:
            x2 = self.worker2_x_max
            self._dir2 = -1

        p.resetBasePositionAndOrientation(wid2, [x2, y2, z2], orn2)

    def lidar_detect(
        self,
        robot_x,
        robot_y,
        robot_theta,
        fov_deg=360.0,
        num_rays=36,
        max_range=10.0,
        ray_z=0.25,
    ):
        """
        Lidar-like detection via ray casting.

        I cast 'num_rays' rays over 'fov_deg' centered around robot_theta,
        and check if any ray hits a worker sphere.

        Returns:
            hit (bool): True if a worker is hit by any ray
            min_dist (float): minimum hit distance among worker hits (meters)
            hit_worker_id (int|None): body id of the closest worker hit
        """
        if not self.worker_ids:
            return False, float("inf"), None

        # Build ray angles
        half = math.radians(fov_deg) / 2.0
        if num_rays <= 1:
            angles = [robot_theta]
        else:
            step = (2.0 * half) / (num_rays - 1)
            angles = [robot_theta - half + i * step for i in range(num_rays)]

        # Build ray endpoints
        ray_from = [[robot_x, robot_y, ray_z] for _ in angles]
        ray_to = [
            [robot_x + max_range * math.cos(a), robot_y + max_range * math.sin(a), ray_z]
            for a in angles
        ]

        results = p.rayTestBatch(ray_from, ray_to, numThreads=0)

        best_dist = float("inf")
        best_id = None

        # PyBullet ray result tuple:
        # (hitBodyUniqueId, hitLinkIndex, hitFraction, hitPosition, hitNormal)
        for hit in results:
            body_id = hit[0]
            hit_frac = hit[2]
            if body_id in self.worker_ids and hit_frac >= 0.0:
                d = hit_frac * max_range
                if d < best_dist:
                    best_dist = d
                    best_id = body_id

        return (best_id is not None), best_dist, best_id