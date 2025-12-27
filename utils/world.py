import os
import time
import math

import pybullet as p
import pybullet_data

# Disable profiling output from internal Bullet components (keeps console clean)
os.environ["B3_NO_PROFILE"] = "1"
os.environ["BT_DISABLE_PROFILE"] = "1"
os.environ["BT_NO_PROFILE"] = "1"
os.environ["BULLET_NO_PROFILE"] = "1"


# =====================================================================
# CAMERA UTILITY FUNCTIONS
# =====================================================================

def get_forward_vector(yaw_deg: float, pitch_deg: float):
    """
    Compute the forward (look) direction vector for the PyBullet debug camera.

    PyBullet's camera uses yaw/pitch in degrees. I convert to radians and build a
    unit direction vector to move the camera target in a "FPS-like" way.

    Notes:
    - Add +90° to yaw to align with PyBullet's camera convention.
    """
    yaw = math.radians(yaw_deg + 90.0)
    pitch = math.radians(pitch_deg)

    return [
        math.cos(pitch) * math.cos(yaw),
        math.cos(pitch) * math.sin(yaw),
        math.sin(pitch),
    ]


def get_right_vector(yaw_deg: float, pitch_deg: float):
    """
    Compute the right vector for the debug camera (used for strafing).
    Pitch does not matter for horizontal strafing, so I ignore it.
    """
    yaw = math.radians(yaw_deg)
    return [
        math.cos(yaw),
        math.sin(yaw),
        0.0,
    ]


# =====================================================================
# BASE WORLD CLASS (LAB ENVIRONMENT)
# =====================================================================

class BaseWorld:
    """
    BaseWorld encapsulates the PyBullet simulation environment:

    Responsibilities:
    - Connect to PyBullet GUI
    - Load ground plane
    - Load wall blocks from an ASCII map file:
        '1' -> wall block (static)
        '2' -> box block (dynamic mass in this implementation)
    - Apply textures to blocks (wall.png for '1', box.png for '2')
    - Provide a free camera (arrow keys) to navigate the scene
    - Provide step callbacks (executed each simulation tick), used by robot and workers
    """

    def __init__(self):
        print("PyBullet version:", p.getAPIVersion())

        # Connect to PyBullet using GUI. Use p.DIRECT for headless mode.
        # The options disable timer/file caching for more deterministic behavior.
        self.client = p.connect(
            p.GUI,
            options="--disable_timer --disable_file_caching"
        )

        # Allow loading standard URDFs included with pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Gravity settings
        p.setGravity(0, 0, -9.81)

        # Enable basic visualizer features
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 1)

        # Load the ground plane (simple infinite plane)
        self.planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(self.planeId, -1, lateralFriction=1.0)

        # Simulation helpers / registries
        self.move_speed = 0.02                 # camera movement speed
        self.cuboids = []                      # IDs for ASCII '1' blocks
        self.box_cuboids = []                  # IDs for ASCII '2' blocks
        self.step_callbacks = []               # called each simulation tick
        self.additional_key_callbacks = []     # called with key events

        # Initial camera view (this is the view to see the map immediately)
        p.resetDebugVisualizerCamera(
            cameraDistance=1,
            cameraYaw=0,
            cameraPitch=-52.8,
            cameraTargetPosition=[24, -2.5, 14],
        )

    # ------------------------------------------------------------------
    # CALLBACK REGISTRATION
    # ------------------------------------------------------------------

    def add_keyboard_callback(self, f):
        """Register a callback that receives the raw PyBullet keyboard event dict."""
        self.additional_key_callbacks.append(f)

    def add_step_callback(self, f):
        """Register a function to be executed every simulation tick."""
        self.step_callbacks.append(f)

    # ------------------------------------------------------------------
    # WORLD OBJECTS (CUBOIDS / WALL BLOCKS)
    # ------------------------------------------------------------------

    def add_cuboid(self, dims, position, color, mass=0.0):
        """
        Create a cuboid (box) rigid body.

        Parameters
        ----------
        dims : (half_x, half_y, half_z)
            Half extents of the box (PyBullet uses half extents).
        position : (x, y, z)
            World position of the body.
        color : [r,g,b]
            RGB in [0,1].
        mass : float
            mass=0 -> static body. mass>0 -> dynamic body.

        Returns
        -------
        body_id : int
            PyBullet body unique id.
        """
        collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=dims)
        visual_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=dims,
            rgbaColor=color + [1.0],
        )

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
        )

        # Track static walls (ASCII '1') in self.cuboids by convention.
        # Note: dynamic blocks (mass>0) are handled separately.
        if mass == 0.0:
            self.cuboids.append(body_id)

        return body_id

    # ------------------------------------------------------------------
    # LOAD MAP FROM ASCII FILE
    # ------------------------------------------------------------------

    def map_from_file(self, fname, unitwidth=1.0):
        """
        Convert an ASCII map into 3D blocks in PyBullet.

        Characters:
            '1' -> static wall block
            '2' -> box block (currently spawned with mass=0.5 in this file)

        unitwidth:
            Size of one ASCII cell in meters.
        """
        with open(fname, "r") as f:
            txt = f.read().strip()

        row = 0
        for line in txt.split("\n"):
            col = 0
            for c in line.strip():
                # Convert grid coordinate -> world coordinate (center of the cell)
                x = col * unitwidth + unitwidth / 2.0
                y = row * unitwidth + unitwidth / 2.0
                z = unitwidth / 2.0

                if c == "1":
                    # Static wall
                    self.add_cuboid(
                        dims=(unitwidth / 2.0, unitwidth / 2.0, unitwidth / 2.0),
                        position=(x, y, z),
                        color=[0.7, 0.7, 0.7],
                        mass=0.0,
                    )

                elif c == "2":
                    # "Box" blocks: your project textures them with box.png.
                    # Keep as in your current implementation (mass=0.5).
                    cube_id = self.add_cuboid(
                        dims=(unitwidth / 2.0, unitwidth / 2.0, unitwidth / 2.0),
                        position=(x, y, z),
                        color=[0.7, 0.7, 0.7],
                        mass=0.5,
                    )
                    self.box_cuboids.append(cube_id)

                col += 1
            row += 1

    # ------------------------------------------------------------------
    # APPLY TEXTURES
    # ------------------------------------------------------------------

    def texture_walls(self):
        """
        Apply textures to map blocks:
          - ASCII '1' blocks (self.cuboids)     -> wall.png (fallback: silly.png)
          - ASCII '2' blocks (self.box_cuboids) -> box.png
        """

        def first_existing_path(candidates):
            """
            Try several candidate paths and return the first one that exists.
            I test:
              1) relative to this file (utils/...)
              2) as provided (current working directory)
            """
            base_dir = os.path.dirname(__file__)
            for rel in candidates:
                for pth in (os.path.join(base_dir, rel), rel):
                    if os.path.exists(pth):
                        return pth
            return None

        # --- Texture for walls ('1') ---
        wall_path = first_existing_path([
            "textures/wall.png",
            "textures/silly.png",
            "wall.png",
            "silly.png",
        ])
        if wall_path is not None:
            try:
                wall_tex = p.loadTexture(wall_path)
                for cube_id in self.cuboids:
                    p.changeVisualShape(cube_id, -1, textureUniqueId=wall_tex)
            except Exception as e:
                print(f"[WORLD] Warning: could not load/apply wall texture ({wall_path}): {e}")

        # --- Texture for boxes ('2') ---
        box_path = first_existing_path([
            "textures/box.png",
            "box.png",
        ])
        if box_path is not None:
            try:
                box_tex = p.loadTexture(box_path)
                for cube_id in self.box_cuboids:
                    p.changeVisualShape(cube_id, -1, textureUniqueId=box_tex)
            except Exception as e:
                print(f"[WORLD] Warning: could not load/apply box texture ({box_path}): {e}")

    # ------------------------------------------------------------------
    # CAMERA MOVEMENT (ARROW KEYS)
    # ------------------------------------------------------------------

    def camera_movement(self):
        """
        FPS-like free camera movement using arrow keys:

        - UP/DOWN: move forward/backward along camera forward vector
        - LEFT/RIGHT: strafe left/right along camera right vector

        This moves the "camera target position" so you can quickly navigate the scene.
        """
        keys = p.getKeyboardEvents()
        cam_info = p.getDebugVisualizerCamera()

        yaw = cam_info[8]
        pitch = cam_info[9]
        camera_target = list(cam_info[11])  # camera target position (x,y,z)

        moved = False

        # Forward
        if p.B3G_UP_ARROW in keys and (keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN):
            fwd = get_forward_vector(yaw, pitch)
            camera_target[0] += fwd[0] * self.move_speed
            camera_target[1] += fwd[1] * self.move_speed
            camera_target[2] += fwd[2] * self.move_speed
            moved = True

        # Backward
        if p.B3G_DOWN_ARROW in keys and (keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN):
            fwd = get_forward_vector(yaw, pitch)
            camera_target[0] -= fwd[0] * self.move_speed
            camera_target[1] -= fwd[1] * self.move_speed
            camera_target[2] -= fwd[2] * self.move_speed
            moved = True

        # Strafe left/right (based on right vector)
        if p.B3G_LEFT_ARROW in keys and (keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN):
            right = get_right_vector(yaw, pitch)
            camera_target[0] -= right[0] * self.move_speed
            camera_target[1] -= right[1] * self.move_speed
            moved = True

        if p.B3G_RIGHT_ARROW in keys and (keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN):
            right = get_right_vector(yaw, pitch)
            camera_target[0] += right[0] * self.move_speed
            camera_target[1] += right[1] * self.move_speed
            moved = True

        # Apply camera changes only if we actually moved
        if moved:
            p.resetDebugVisualizerCamera(
                cameraDistance=0.01,
                cameraYaw=yaw,
                cameraPitch=pitch,
                cameraTargetPosition=camera_target,
            )

        # Forward keyboard state to any registered callbacks (robot UI, debug hotkeys, etc.)
        for f in self.additional_key_callbacks:
            f(keys)

    # ------------------------------------------------------------------
    # SIMULATION STEP
    # ------------------------------------------------------------------

    def simStep(self):
        """
        Execute one simulation tick at 240 Hz.

        Order matters:
        1) Run registered step callbacks (robot motor update, workers movement, etc.)
        2) Handle camera movement
        3) Step physics
        4) Sleep to keep real-time pacing
        """
        for f in self.step_callbacks:
            f()

        self.camera_movement()
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    # ------------------------------------------------------------------
    # SHUTDOWN
    # ------------------------------------------------------------------

    def end(self):
        """
        Clean shutdown.

        IMPORTANT:
        p.connect() returns a client id (int). The correct way to close is p.disconnect(client_id).
        """
        try:
            if p.isConnected(self.client):
                p.disconnect(self.client)
        except Exception as e:
            print(f"[WORLD] end() warning: {e}")