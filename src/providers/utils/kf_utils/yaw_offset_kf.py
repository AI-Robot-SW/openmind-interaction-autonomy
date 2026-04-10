# yaw_offset_kf.py
"""
1D Kalman filter for estimating the yaw offset between odometry frame and
global (GNSS/ENU) frame.

State   : x = yaw_offset (degrees)
Model   : constant offset with small random walk
          x_{k+1} = x_k + w_k
Measure : z = gnss_heading_deg - odom_heading_deg
"""

import math

from ..geo_utils import wrap_deg


class YawOffsetKF:
    """Scalar Kalman filter for odom-to-GNSS yaw offset estimation."""

    def __init__(
        self,
        p_init_deg2: float = 180.0 ** 2,
        q_deg2_per_sec: float = 0.05,
        r_deg2: float = 25.0,
        converged_std_deg: float = 10.0,
        min_updates_for_convergence: int = 3,
    ) -> None:
        self.x: float = 0.0
        self.P: float = float(p_init_deg2)

        self.q_deg2_per_sec: float = float(q_deg2_per_sec)
        self.r_deg2: float = float(r_deg2)

        self.converged_std_deg: float = float(converged_std_deg)
        self.min_updates_for_convergence: int = int(min_updates_for_convergence)

        self._n_updates: int = 0

    def predict(self, dt_sec: float) -> None:
        """Time update."""
        dt_sec = max(float(dt_sec), 1e-3)
        self.P += self.q_deg2_per_sec * dt_sec

    def update(self, z_deg: float, r_deg2: float | None = None) -> None:
        """Measurement update.

        Args:
            z_deg: measured yaw offset in degrees
            r_deg2: optional measurement variance override
        """
        measurement_variance_deg2 = self.r_deg2 if r_deg2 is None else float(r_deg2)
        measurement_variance_deg2 = max(measurement_variance_deg2, 1e-9)

        innovation_deg = wrap_deg(float(z_deg) - self.x)
        S = self.P + measurement_variance_deg2
        K = self.P / S

        self.x = wrap_deg(self.x + K * innovation_deg)
        self.P = max((1.0 - K) * self.P, 1e-9)
        self._n_updates += 1

    @property
    def std_deg(self) -> float:
        return math.sqrt(max(0.0, self.P))

    @property
    def converged(self) -> bool:
        return (
            self._n_updates >= self.min_updates_for_convergence
            and self.std_deg < self.converged_std_deg
        )

    @property
    def n_updates(self) -> int:
        return self._n_updates
