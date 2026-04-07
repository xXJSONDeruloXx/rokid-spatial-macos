"""IMU sensor fusion — Madgwick AHRS filter.

Fuses accelerometer, gyroscope, and (optionally) magnetometer data
into a quaternion orientation estimate.

Reference: S. Madgwick, "An efficient orientation filter for inertial
and inertial/magnetic sensor arrays" (2010).

Rokid Max axis mapping notes:
  The Rokid Max's IMU has a non-standard axis orientation. From empirical
  testing and XRLinuxDriver reference code, the raw sensor axes need
  remapping for correct pitch/yaw/roll behavior:
    - Gyro/Accel X axis → maps to head pitch (nodding)
    - Gyro/Accel Y axis → maps to head yaw (turning left/right)
    - Gyro/Accel Z axis → maps to head roll (tilting)
  The Madgwick filter expects:
    - X = roll, Y = pitch, Z = yaw
  So we remap: sensor(x,y,z) → filter(z, x, y) with sign adjustments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from rokid_spatial.spatial import Quaternion, normalize_quaternion


def remap_rokid_axes(
    gx: float, gy: float, gz: float,
    ax: float, ay: float, az: float,
) -> tuple[float, float, float, float, float, float]:
    """Remap Rokid Max sensor axes to Madgwick-expected NWU convention.

    The Rokid's IMU axes are oriented differently from the NED/NWU frame
    that Madgwick expects. This remapping was derived empirically:
      - Negate gyro X and Z for correct pitch/yaw direction
      - Negate accel X and Z for correct gravity alignment

    Returns: (gx, gy, gz, ax, ay, az) in Madgwick convention.
    """
    return (-gx, gy, -gz, -ax, ay, -az)


@dataclass
class MadgwickFilter:
    """Madgwick AHRS orientation filter.

    Attributes:
        beta: Filter gain (higher = more accel trust, less gyro drift).
              Default 0.1 is a good balance for head tracking.
        sample_period: Expected time between samples in seconds.
        quaternion: Current orientation estimate.
        remap_axes: Apply Rokid-specific axis remapping before fusion.
    """

    beta: float = 0.1
    sample_period: float = 1.0 / 90.0  # 90 Hz default for Rokid Max
    quaternion: Quaternion = field(
        default_factory=lambda: Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    )
    remap_axes: bool = True  # Set False for pre-remapped or simulated data

    def update_imu(
        self,
        gx: float,
        gy: float,
        gz: float,
        ax: float,
        ay: float,
        az: float,
        dt: float | None = None,
    ) -> Quaternion:
        """Update orientation using gyroscope and accelerometer (6-axis).

        Args:
            gx, gy, gz: Gyroscope readings in rad/s.
            ax, ay, az: Accelerometer readings in m/s².
            dt: Time step override (seconds). Uses sample_period if None.

        Returns:
            Updated quaternion orientation.
        """
        if dt is None:
            dt = self.sample_period

        if self.remap_axes:
            gx, gy, gz, ax, ay, az = remap_rokid_axes(gx, gy, gz, ax, ay, az)

        q = self.quaternion
        qw, qx, qy, qz = q.w, q.x, q.y, q.z

        # Normalize accelerometer
        a_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if a_norm < 1e-10:
            # Can't determine orientation from zero accel
            return self._integrate_gyro(gx, gy, gz, dt)

        ax /= a_norm
        ay /= a_norm
        az /= a_norm

        # Gradient descent corrective step
        _2qw = 2.0 * qw
        _2qx = 2.0 * qx
        _2qy = 2.0 * qy
        _2qz = 2.0 * qz
        _4qw = 4.0 * qw
        _4qx = 4.0 * qx
        _4qy = 4.0 * qy
        _8qx = 8.0 * qx
        _8qy = 8.0 * qy
        qw2 = qw * qw
        qx2 = qx * qx
        qy2 = qy * qy
        qz2 = qz * qz

        # Objective function and Jacobian
        s0 = _4qw * qy2 + _2qy * ax + _4qw * qx2 - _2qx * ay
        s1 = _4qx * qz2 - _2qz * ax + 4.0 * qw2 * qx - _2qw * ay - _4qx + _8qx * qx2 + _8qx * qy2 + _4qx * az
        s2 = 4.0 * qw2 * qy + _2qw * ax + _4qy * qz2 - _2qz * ay - _4qy + _8qy * qx2 + _8qy * qy2 + _4qy * az
        s3 = 4.0 * qx2 * qz - _2qx * ax + 4.0 * qy2 * qz - _2qy * ay

        # Normalize step
        s_norm = math.sqrt(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3)
        if s_norm > 1e-10:
            s0 /= s_norm
            s1 /= s_norm
            s2 /= s_norm
            s3 /= s_norm

        # Compute rate of change of quaternion
        q_dot_w = 0.5 * (-qx * gx - qy * gy - qz * gz) - self.beta * s0
        q_dot_x = 0.5 * (qw * gx + qy * gz - qz * gy) - self.beta * s1
        q_dot_y = 0.5 * (qw * gy - qx * gz + qz * gx) - self.beta * s2
        q_dot_z = 0.5 * (qw * gz + qx * gy - qy * gx) - self.beta * s3

        # Integrate
        qw += q_dot_w * dt
        qx += q_dot_x * dt
        qy += q_dot_y * dt
        qz += q_dot_z * dt

        self.quaternion = normalize_quaternion(
            Quaternion(w=qw, x=qx, y=qy, z=qz)
        )
        return self.quaternion

    def _integrate_gyro(
        self, gx: float, gy: float, gz: float, dt: float
    ) -> Quaternion:
        """Pure gyroscope integration (no correction)."""
        q = self.quaternion
        qw, qx, qy, qz = q.w, q.x, q.y, q.z

        q_dot_w = 0.5 * (-qx * gx - qy * gy - qz * gz)
        q_dot_x = 0.5 * (qw * gx + qy * gz - qz * gy)
        q_dot_y = 0.5 * (qw * gy - qx * gz + qz * gx)
        q_dot_z = 0.5 * (qw * gz + qx * gy - qy * gx)

        qw += q_dot_w * dt
        qx += q_dot_x * dt
        qy += q_dot_y * dt
        qz += q_dot_z * dt

        self.quaternion = normalize_quaternion(
            Quaternion(w=qw, x=qx, y=qy, z=qz)
        )
        return self.quaternion

    def reset(self) -> None:
        """Reset to identity orientation."""
        self.quaternion = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
