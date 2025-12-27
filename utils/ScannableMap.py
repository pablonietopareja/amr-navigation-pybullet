import numpy as np
import pygame


class ScanableMap:
    """
    ScanableMap — 2D occupancy grid + simple simulated 2D lidar.

    Concept:
    - The ASCII map is loaded into an occupancy grid: grid[y, x] = 1 means obstacle.
    - I assume 1 map cell ≈ 1 meter in the PyBullet world.
    - The scan() function simulates a lidar by ray-marching in small increments
      until it hits an obstacle or leaves the map boundaries.

    Notes:
    - This lidar is "map-based": it detects obstacles that exist in the loaded map.
      Dynamic objects (e.g., Workers spheres) are not part of this map-based scan.
    - The method is vectorized with NumPy for simplicity and speed.
    """

    def __init__(
        self,
        map_dimensions=(100, 100),   # (width, height) in ASCII cells
        fov=360,                     # field of view in degrees
        max_distance=100.0,          # max lidar range (meters ~= cells)
        num_rays=360,
        scale=1,                     # kept for compatibility (unused)
        draw_unit=10,                # pixel size of each cell when drawing with pygame
        step_size=0.1,               # ray-marching resolution in meters
    ):
        """
        Parameters
        ----------
        map_dimensions:
            (width, height) of the occupancy grid, in cells (≈ meters).
        fov:
            Lidar field of view in degrees.
        max_distance:
            Max lidar distance in meters (≈ cells).
        num_rays:
            Number of rays (angular samples).
        draw_unit:
            Pixel size used by draw_map().
        step_size:
            Distance between ray-marching samples along each ray.
            Smaller -> more accurate but more expensive.
        """
        width_cells, height_cells = int(map_dimensions[0]), int(map_dimensions[1])

        # Grid is indexed as grid[row=y, col=x]
        self.cols = width_cells
        self.rows = height_cells
        self.grid = np.zeros((self.rows, self.cols), dtype=np.uint8)

        # For pygame visualization only (optional)
        self.draw_boxes = []

        self.max_distance = float(max_distance)
        self.num_rays = int(num_rays)
        self.fov = float(np.radians(fov))  # store in radians internally
        self.draw_unit = int(draw_unit)

        # Ray marching resolution (meters)
        self.step_size = float(step_size)

        # Precompute "steps along the ray" once (saves allocations per scan call).
        # Example: [0.0, 0.1, 0.2, ..., max_distance)
        self._steps = np.arange(0.0, self.max_distance, self.step_size, dtype=float)
        if self._steps.size == 0:
            # Ensure I always have at least one sample
            self._steps = np.array([0.0], dtype=float)

    # ------------------------------------------------------------------
    # MAP BUILDING
    # ------------------------------------------------------------------

    def set_solid(self, x: int, y: int) -> None:
        """
        Mark cell (x, y) as an obstacle (occupied).
        """
        if 0 <= x < self.cols and 0 <= y < self.rows:
            self.grid[y, x] = 1
            self.draw_boxes.append(
                (x * self.draw_unit, y * self.draw_unit, self.draw_unit, self.draw_unit)
            )

    def load_map(self, fname: str) -> None:
        """
        Load an ASCII map from file.

        Characters interpreted as obstacles:
            '1' or '2'
        Everything else is treated as free space.

        Important:
        - Map coordinates are interpreted as:
            col -> x
            row -> y
        - If the file has more rows/cols than the configured grid size,
          extra cells are ignored (clipped).
        """
        with open(fname, "r") as f:
            txt = f.read().strip()

        # Reset previous map content
        self.grid.fill(0)
        self.draw_boxes.clear()

        row = 0
        for line in txt.split("\n"):
            if row >= self.rows:
                break
            col = 0
            for c in line.strip():
                if col >= self.cols:
                    break
                if c == "1" or c == "2":
                    self.set_solid(col, row)
                col += 1
            row += 1

    def draw_map(self, display_surface) -> None:
        """
        Draw the map to a pygame surface (white rectangles = obstacles).
        Used only for debugging/visualization.
        """
        for rect in self.draw_boxes:
            pygame.draw.rect(display_surface, (255, 255, 255), rect)

    # ------------------------------------------------------------------
    # LIDAR SIMULATION
    # ------------------------------------------------------------------

    def scan(self, start_x: float, start_y: float, start_a: float):
        """
        Simulate a 2D lidar scan by ray-marching on the occupancy grid.

        Parameters
        ----------
        start_x, start_y:
            Robot position in meters (aligned with map cells).
        start_a:
            Robot yaw (heading) in radians.

        Returns
        -------
        distances : np.ndarray, shape (num_rays,)
            Distance in meters to the first obstacle (or boundary), per ray.
        angles : np.ndarray, shape (num_rays,)
            Absolute ray angles in radians.
        """
        if self.num_rays <= 0:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        # Robot pose in "cell space" (1 cell ≈ 1 meter)
        gx = float(start_x)
        gy = float(start_y)

        # Ray angles uniformly distributed across FOV (endpoint excluded to avoid duplication)
        angles = np.linspace(
            start_a - self.fov / 2.0,
            start_a + self.fov / 2.0,
            self.num_rays,
            endpoint=False,
            dtype=float,
        )

        # Unit direction vectors for each ray
        dx = np.cos(angles)
        dy = np.sin(angles)

        steps = self._steps  # precomputed
        n_steps = steps.size

        # Compute all ray sample positions:
        # x_positions shape: (num_rays, n_steps)
        x_positions = gx + dx[:, None] * steps[None, :]
        y_positions = gy + dy[:, None] * steps[None, :]

        # Convert positions to grid indices (cell coordinates)
        x_idx = np.floor(x_positions).astype(int)
        y_idx = np.floor(y_positions).astype(int)

        # Valid samples are those inside the map boundaries
        valid = (
            (x_idx >= 0) & (x_idx < self.cols) &
            (y_idx >= 0) & (y_idx < self.rows)
        )

        # Avoid index errors by clipping (I will mask invalid samples later)
        x_safe = np.clip(x_idx, 0, self.cols - 1)
        y_safe = np.clip(y_idx, 0, self.rows - 1)

        # Sample occupancy grid: 0 = free, 1 = obstacle
        samples = self.grid[y_safe, x_safe]

        # Mask out-of-bounds samples (treat them as free for the sample array,
        # but I still count boundary crossing as a "hit" below)
        samples = samples * valid

        # A ray "hits" if:
        # - it encounters an obstacle cell (samples == 1), OR
        # - it goes outside the map (valid == False)
        hit_mask = (samples == 1) | (~valid)

        # Index of the first hit along each ray.
        # np.argmax returns 0 if all values are False, so I handle "no hit" separately.
        first_hit_idx = np.argmax(hit_mask, axis=1)

        # Convert hit index to physical distance
        distances = steps[first_hit_idx].copy()

        # Rays that never hit anything and never leave the map
        no_hit = ~np.any(hit_mask, axis=1)
        distances[no_hit] = self.max_distance

        return distances, angles