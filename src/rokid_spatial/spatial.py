"""Spatial math: quaternion operations, head pose, screen projection.

Coordinate system conventions:
- Rokid SDK native: East-Up-South
- Our target: North-West-Up (NWU)
- Adjustment quaternion converts between them
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rokid_spatial.constants import ADJUSTMENT_QUAT


@dataclass(frozen=True, slots=True)
class Quaternion:
    """Unit quaternion representing a 3D rotation.

    Convention: (w, x, y, z) where w is the scalar part.
    """

    w: float
    x: float
    y: float
    z: float

    @property
    def norm(self) -> float:
        """Euclidean norm (magnitude) of the quaternion."""
        return math.sqrt(self.w**2 + self.x**2 + self.y**2 + self.z**2)

    def as_tuple(self) -> tuple[float, float, float, float]:
        """Return (w, x, y, z) tuple."""
        return (self.w, self.x, self.y, self.z)


def normalize_quaternion(q: Quaternion) -> Quaternion:
    """Normalize a quaternion to unit length.

    Args:
        q: Input quaternion (any magnitude).

    Returns:
        Unit quaternion pointing in the same direction.
    """
    n = q.norm
    if n < 1e-10:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
    return Quaternion(w=q.w / n, x=q.x / n, y=q.y / n, z=q.z / n)


def quaternion_multiply(q1: Quaternion, q2: Quaternion) -> Quaternion:
    """Hamilton product of two quaternions: q1 * q2.

    This composes rotations: first q2, then q1 (right-to-left).
    """
    w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    return Quaternion(w=w, x=x, y=y, z=z)


def euler_from_quaternion(q: Quaternion) -> tuple[float, float, float]:
    """Convert quaternion to Euler angles (ZYX intrinsic / Tait-Bryan).

    Args:
        q: Unit quaternion.

    Returns:
        (roll, pitch, yaw) in radians.
        - roll:  rotation around X axis
        - pitch: rotation around Y axis
        - yaw:   rotation around Z axis
    """
    # Roll (X)
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    sinp = max(-1.0, min(1.0, sinp))  # Clamp for numerical safety
    pitch = math.asin(sinp)

    # Yaw (Z)
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (roll, pitch, yaw)


def apply_coordinate_adjustment(q: Quaternion) -> Quaternion:
    """Apply the Rokid east-up-south → north-west-up coordinate adjustment.

    Multiplies the input quaternion by the factory calibration adjustment
    quaternion from the XRLinuxDriver reference implementation.
    """
    adj = Quaternion(
        w=ADJUSTMENT_QUAT[0],
        x=ADJUSTMENT_QUAT[1],
        y=ADJUSTMENT_QUAT[2],
        z=ADJUSTMENT_QUAT[3],
    )
    result = quaternion_multiply(q, adj)
    return normalize_quaternion(result)


def smooth_pose(
    samples: list[Quaternion],
    alpha: float = 0.3,
) -> Quaternion:
    """Smooth a sequence of orientation samples using exponential moving average.

    Uses a simple weighted-average approach in quaternion space, which works
    well for small angular differences typical in head tracking jitter.

    Args:
        samples: List of quaternion samples (most recent last).
        alpha: Smoothing factor (0 = full smoothing, 1 = no smoothing).

    Returns:
        Smoothed quaternion (normalized).
    """
    if not samples:
        return Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)

    if len(samples) == 1:
        return normalize_quaternion(samples[0])

    # Simple EMA in quaternion component space
    # This is valid for small angular differences between consecutive samples
    result = samples[0]
    for s in samples[1:]:
        # Ensure quaternion hemisphere consistency (shortest path)
        dot = result.w * s.w + result.x * s.x + result.y * s.y + result.z * s.z
        if dot < 0:
            s = Quaternion(w=-s.w, x=-s.x, y=-s.y, z=-s.z)

        result = Quaternion(
            w=result.w * (1 - alpha) + s.w * alpha,
            x=result.x * (1 - alpha) + s.x * alpha,
            y=result.y * (1 - alpha) + s.y * alpha,
            z=result.z * (1 - alpha) + s.z * alpha,
        )

    return normalize_quaternion(result)
