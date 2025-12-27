import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence


# =============================================================================
# Simple particle data structure
# =============================================================================

@dataclass
class Particle:
    """
    Represents a single hypothesis of the robot pose.

    Attributes
    ----------
    x, y : float
        Position in meters (world/map coordinates).
    theta : float
        Heading angle in radians.
    w : float
        Particle weight (probability) after measurement update.
    """
    x: float
    y: float
    theta: float
    w: float = 1.0


# =============================================================================
# 2D Particle Filter for localization in a known map
# =============================================================================

class ParticleFilter:
    """
    2D Particle Filter (PF) for robot localization.

    Core idea:
    - Maintain N particles, each one is a possible robot pose (x, y, theta).
    - Prediction step (motion model):
        Apply odometry increments (delta_d, delta_theta) + noise to every particle.
    - Update step (measurement model):
        For each particle, simulate a lidar scan from that pose using ScanableMap.scan(),
        compare with measured scan -> update particle weight.
    - Resampling:
        When particles degenerate (low Neff), resample systematically.

    Expected ScanableMap API:
        expected_distances, angles = scan_map.scan(x, y, theta)
    """

    def __init__(
        self,
        scan_map,
        n_particles: int,
        init_x_range: Tuple[float, float],
        init_y_range: Tuple[float, float],
        init_theta_range: Tuple[float, float],
        motion_std_lin: float = 0.01,
        motion_std_rot: float = math.radians(1.0),
        scan_std: float = 0.4,
    ):
        self.scan_map = scan_map
        self.n = int(n_particles)

        self.init_x_range = init_x_range
        self.init_y_range = init_y_range
        self.init_theta_range = init_theta_range

        self.motion_std_lin = float(motion_std_lin)
        self.motion_std_rot = float(motion_std_rot)
        self.scan_std = float(scan_std)

        self.particles: List[Particle] = []
        self._init_uniform()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------

    @staticmethod
    def _wrap_angle(a: float) -> float:
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(a), math.cos(a))

    # -------------------------------------------------------------------------
    # Initialization / Reset
    # -------------------------------------------------------------------------

    def _init_uniform(self) -> None:
        """Initialize particles uniformly across the configured ranges."""
        self.particles = []
        xmin, xmax = self.init_x_range
        ymin, ymax = self.init_y_range
        tmin, tmax = self.init_theta_range

        if self.n <= 0:
            return

        w0 = 1.0 / self.n
        for _ in range(self.n):
            x = random.uniform(xmin, xmax)
            y = random.uniform(ymin, ymax)
            theta = random.uniform(tmin, tmax)
            self.particles.append(Particle(x=x, y=y, theta=theta, w=w0))

    def reset(self, init_pose: Optional[Tuple[float, float, float]] = None) -> None:
        """
        Reset the PF.

        If init_pose is None:
            Re-initialize uniformly in the configured ranges.
        If init_pose = (x0, y0, theta0):
            Initialize near that pose using a small Gaussian spread.
        """
        if init_pose is None:
            self._init_uniform()
            return

        if self.n <= 0:
            return

        x0, y0, t0 = init_pose
        self.particles = []
        w0 = 1.0 / self.n

        for _ in range(self.n):
            x = random.gauss(x0, 0.5)
            y = random.gauss(y0, 0.5)
            theta = self._wrap_angle(random.gauss(t0, math.radians(10.0)))
            self.particles.append(Particle(x=x, y=y, theta=theta, w=w0))

    # -------------------------------------------------------------------------
    # Prediction (motion update)
    # -------------------------------------------------------------------------

    def predict(self, delta_d: float, delta_theta: float) -> None:
        """Apply the odometry motion increment to each particle with noise."""
        if self.n <= 0 or not self.particles:
            return

        std_d = self.motion_std_lin
        std_th = self.motion_std_rot

        for p in self.particles:
            dd = random.gauss(delta_d, std_d)
            dth = random.gauss(delta_theta, std_th)

            p.theta = self._wrap_angle(p.theta + dth)
            p.x += dd * math.cos(p.theta)
            p.y += dd * math.sin(p.theta)

    # -------------------------------------------------------------------------
    # Update (measurement / lidar)
    # -------------------------------------------------------------------------

    def update(self, measured_distances: Sequence[float]) -> None:
        """
        Update particle weights using a lidar measurement vector.

        measured_distances:
            Can be a list OR a numpy array. We only require:
            - len(measured_distances) works
            - indexing works
        """
        if self.n <= 0 or not self.particles:
            return
        if measured_distances is None:
            return

        # IMPORTANT: do NOT do `if not measured_distances` (numpy arrays break that).
        try:
            m = len(measured_distances)
        except TypeError:
            # If something non-iterable comes in, just ignore the update safely.
            return

        if m == 0:
            return

        sigma = self.scan_std
        if sigma <= 1e-12:
            sigma = 1e-12

        inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)

        # 1) Compute log-weights (unnormalized)
        log_ws: List[float] = []
        max_log_w = -float("inf")

        for p in self.particles:
            expected, _ = self.scan_map.scan(p.x, p.y, p.theta)

            L = min(len(expected), m)
            if L <= 0:
                lw = -1e9
            else:
                lw = 0.0
                for i in range(L):
                    e = float(measured_distances[i]) - float(expected[i])
                    lw += -(e * e) * inv_2sigma2

            log_ws.append(lw)
            if lw > max_log_w:
                max_log_w = lw

        # 2) Convert to normal weights in a numerically stable way
        total_w = 0.0
        for p, lw in zip(self.particles, log_ws):
            w = math.exp(lw - max_log_w)
            p.w = w
            total_w += w

        # 3) Normalize weights
        if total_w <= 1e-30:
            self._init_uniform()
            return

        inv_total = 1.0 / total_w
        for p in self.particles:
            p.w *= inv_total

        # 4) Resample if needed
        self._maybe_resample()

    # -------------------------------------------------------------------------
    # Resampling
    # -------------------------------------------------------------------------

    def _effective_sample_size(self) -> float:
        """Neff = 1 / sum(w_i^2)."""
        sum_w2 = 0.0
        for p in self.particles:
            sum_w2 += p.w * p.w
        if sum_w2 <= 1e-12:
            return 0.0
        return 1.0 / sum_w2

    def _maybe_resample(self) -> None:
        """Systematic resampling when Neff < N/2."""
        Neff = self._effective_sample_size()
        if Neff > self.n / 2.0:
            return

        if self.n <= 0:
            return

        # Build CDF
        cdf = []
        cumsum = 0.0
        for p in self.particles:
            cumsum += p.w
            cdf.append(cumsum)

        u0 = random.random() / self.n
        new_particles: List[Particle] = []
        idx = 0

        for i in range(self.n):
            u = u0 + i / self.n
            while idx < self.n - 1 and u > cdf[idx]:
                idx += 1
            sel = self.particles[idx]
            new_particles.append(Particle(x=sel.x, y=sel.y, theta=sel.theta, w=1.0 / self.n))

        self.particles = new_particles

    # -------------------------------------------------------------------------
    # Estimation
    # -------------------------------------------------------------------------

    def estimate(self) -> Tuple[float, float, float]:
        """Return estimated pose (x, y, theta) as the weighted mean."""
        if self.n <= 0 or not self.particles:
            return 0.0, 0.0, 0.0

        sum_w = 0.0
        for p in self.particles:
            sum_w += p.w

        if sum_w <= 1e-12:
            self._init_uniform()
            return 0.0, 0.0, 0.0

        inv_sum_w = 1.0 / sum_w

        x_est = 0.0
        y_est = 0.0
        cos_sum = 0.0
        sin_sum = 0.0

        for p in self.particles:
            w = p.w * inv_sum_w
            x_est += p.x * w
            y_est += p.y * w
            cos_sum += math.cos(p.theta) * w
            sin_sum += math.sin(p.theta) * w

        theta_est = math.atan2(sin_sum, cos_sum)
        return x_est, y_est, theta_est