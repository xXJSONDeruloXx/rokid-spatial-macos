"""RED tests for spatial math — quaternion operations, head pose, screen projection."""

import math

import numpy as np
import pytest

from rokid_spatial.spatial import (
    Quaternion,
    apply_coordinate_adjustment,
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_multiply,
    smooth_pose,
)


class TestQuaternion:
    """Test the Quaternion dataclass."""

    def test_quaternion_creation(self):
        """Quaternion stores w, x, y, z components."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        assert q.w == 1.0
        assert q.x == 0.0
        assert q.y == 0.0
        assert q.z == 0.0

    def test_quaternion_norm(self):
        """Quaternion exposes a norm property."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        assert pytest.approx(q.norm, abs=1e-6) == 1.0

    def test_quaternion_as_tuple(self):
        """Quaternion can be converted to (w, x, y, z) tuple."""
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        assert q.as_tuple() == (0.5, 0.5, 0.5, 0.5)


class TestNormalizeQuaternion:
    """Test quaternion normalization."""

    def test_normalize_unit_quaternion_is_noop(self):
        """Normalizing a unit quaternion returns the same values."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        result = normalize_quaternion(q)
        assert pytest.approx(result.norm, abs=1e-6) == 1.0

    def test_normalize_non_unit_quaternion(self):
        """Non-unit quaternion is scaled to unit length."""
        q = Quaternion(w=2.0, x=0.0, y=0.0, z=0.0)
        result = normalize_quaternion(q)
        assert pytest.approx(result.w, abs=1e-6) == 1.0
        assert pytest.approx(result.norm, abs=1e-6) == 1.0

    def test_normalize_preserves_direction(self):
        """Normalization preserves the rotation direction."""
        q = Quaternion(w=3.0, x=3.0, y=3.0, z=3.0)
        result = normalize_quaternion(q)
        assert pytest.approx(result.norm, abs=1e-6) == 1.0
        # All components should be equal (same direction)
        assert pytest.approx(result.w, abs=1e-6) == result.x
        assert pytest.approx(result.x, abs=1e-6) == result.y


class TestQuaternionMultiply:
    """Test quaternion multiplication (Hamilton product)."""

    def test_multiply_identity_left(self):
        """q * identity = q."""
        identity = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        q = Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)
        result = quaternion_multiply(q, identity)
        assert pytest.approx(result.w, abs=1e-3) == q.w
        assert pytest.approx(result.x, abs=1e-3) == q.x

    def test_multiply_identity_right(self):
        """identity * q = q."""
        identity = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        q = Quaternion(w=0.707, x=0.707, y=0.0, z=0.0)
        result = quaternion_multiply(identity, q)
        assert pytest.approx(result.w, abs=1e-3) == q.w
        assert pytest.approx(result.x, abs=1e-3) == q.x

    def test_multiply_90deg_rotations(self):
        """Two 90° rotations around the same axis = 180° rotation."""
        # 90° around Z axis: w=cos(45°), z=sin(45°)
        half_sqrt2 = math.sqrt(2) / 2
        q90z = Quaternion(w=half_sqrt2, x=0.0, y=0.0, z=half_sqrt2)
        result = quaternion_multiply(q90z, q90z)
        # Should be 180° around Z: w≈0, z≈1
        assert pytest.approx(result.w, abs=1e-3) == 0.0
        assert pytest.approx(abs(result.z), abs=1e-3) == 1.0


class TestEulerFromQuaternion:
    """Test quaternion → Euler angle (roll, pitch, yaw) conversion."""

    def test_identity_gives_zero_angles(self):
        """Identity quaternion → (0, 0, 0) Euler angles."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        roll, pitch, yaw = euler_from_quaternion(q)
        assert pytest.approx(roll, abs=1e-6) == 0.0
        assert pytest.approx(pitch, abs=1e-6) == 0.0
        assert pytest.approx(yaw, abs=1e-6) == 0.0

    def test_90deg_yaw(self):
        """90° yaw rotation is correctly extracted."""
        half_sqrt2 = math.sqrt(2) / 2
        q = Quaternion(w=half_sqrt2, x=0.0, y=0.0, z=half_sqrt2)
        roll, pitch, yaw = euler_from_quaternion(q)
        assert pytest.approx(yaw, abs=0.01) == math.pi / 2

    def test_euler_angles_in_radians(self):
        """Euler angles are returned in radians."""
        half_sqrt2 = math.sqrt(2) / 2
        q = Quaternion(w=half_sqrt2, x=half_sqrt2, y=0.0, z=0.0)
        roll, pitch, yaw = euler_from_quaternion(q)
        # 90° roll
        assert pytest.approx(roll, abs=0.01) == math.pi / 2


class TestCoordinateAdjustment:
    """Test east-up-south → north-west-up conversion."""

    def test_adjustment_returns_quaternion(self):
        """apply_coordinate_adjustment returns a Quaternion."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        result = apply_coordinate_adjustment(q)
        assert isinstance(result, Quaternion)

    def test_adjustment_changes_orientation(self):
        """Adjustment quaternion modifies the input (not identity)."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        result = apply_coordinate_adjustment(q)
        # The adjustment quat is not identity, so result should differ
        assert not (
            pytest.approx(result.w, abs=1e-6) == 1.0
            and pytest.approx(result.x, abs=1e-6) == 0.0
        )

    def test_adjustment_preserves_unit_length(self):
        """Adjusted quaternion remains unit length."""
        q = Quaternion(w=0.5, x=0.5, y=0.5, z=0.5)
        result = apply_coordinate_adjustment(q)
        assert pytest.approx(result.norm, abs=1e-4) == 1.0


class TestSmoothPose:
    """Test head pose smoothing/filtering."""

    def test_smooth_single_sample_returns_same(self):
        """A single sample passed to smooth_pose returns it unchanged."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        result = smooth_pose([q])
        assert pytest.approx(result.w, abs=1e-6) == q.w

    def test_smooth_identical_samples(self):
        """Smoothing identical samples returns that sample."""
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        result = smooth_pose([q, q, q, q, q])
        assert pytest.approx(result.w, abs=1e-4) == 1.0
        assert pytest.approx(result.x, abs=1e-4) == 0.0

    def test_smooth_reduces_jitter(self):
        """Smoothing noisy samples produces a result closer to the mean."""
        samples = [
            Quaternion(w=0.999, x=0.01, y=0.01, z=0.01),
            Quaternion(w=0.998, x=-0.01, y=0.02, z=-0.01),
            Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
            Quaternion(w=0.999, x=0.005, y=-0.01, z=0.005),
        ]
        result = smooth_pose(samples)
        # Result should be close to identity (the mean of near-identity samples)
        assert result.w > 0.99
        assert abs(result.x) < 0.02
