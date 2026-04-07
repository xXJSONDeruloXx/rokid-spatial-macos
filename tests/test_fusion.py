"""Tests for Madgwick AHRS sensor fusion filter."""

import math

import pytest

from rokid_spatial.fusion import MadgwickFilter
from rokid_spatial.spatial import Quaternion, euler_from_quaternion


class TestMadgwickFilter:
    """Test the Madgwick AHRS orientation filter."""

    def test_initial_orientation_is_identity(self):
        """Filter starts at identity quaternion (no rotation)."""
        f = MadgwickFilter()
        assert pytest.approx(f.quaternion.w, abs=1e-6) == 1.0
        assert pytest.approx(f.quaternion.x, abs=1e-6) == 0.0

    def test_stationary_converges_to_gravity(self):
        """With no rotation and gravity on -Z, filter should stay near identity."""
        f = MadgwickFilter(beta=0.5, sample_period=1 / 90)
        # Simulate 90 samples (1 second) of stationary data
        for _ in range(90):
            f.update_imu(
                gx=0.0, gy=0.0, gz=0.0,  # no rotation
                ax=0.0, ay=0.0, az=-9.81,  # gravity down
            )
        roll, pitch, yaw = euler_from_quaternion(f.quaternion)
        # Should be near zero rotation
        assert abs(math.degrees(roll)) < 5.0
        assert abs(math.degrees(pitch)) < 5.0

    def test_gyro_rotation_detected(self):
        """Constant gyroscope rotation around Z should produce yaw change."""
        f = MadgwickFilter(beta=0.01, sample_period=1 / 90)
        # 1 rad/s around Z for 1 second
        for _ in range(90):
            f.update_imu(
                gx=0.0, gy=0.0, gz=1.0,  # rotating around Z
                ax=0.0, ay=0.0, az=-9.81,
            )
        _, _, yaw = euler_from_quaternion(f.quaternion)
        # Should have accumulated ~1 radian of yaw (57°)
        assert abs(yaw) > 0.5, f"Expected yaw > 0.5 rad, got {yaw:.3f}"

    def test_output_is_unit_quaternion(self):
        """Filter output is always a unit quaternion."""
        f = MadgwickFilter()
        for _ in range(100):
            q = f.update_imu(
                gx=0.1, gy=-0.05, gz=0.2,
                ax=-0.5, ay=9.5, az=1.5,
            )
            assert pytest.approx(q.norm, abs=1e-4) == 1.0

    def test_reset_returns_to_identity(self):
        """reset() restores identity quaternion."""
        f = MadgwickFilter()
        for _ in range(50):
            f.update_imu(gx=1.0, gy=0.0, gz=0.0, ax=0.0, ay=0.0, az=-9.81)
        f.reset()
        assert pytest.approx(f.quaternion.w, abs=1e-6) == 1.0
        assert pytest.approx(f.quaternion.x, abs=1e-6) == 0.0

    def test_custom_dt_overrides_sample_period(self):
        """Passing dt to update_imu overrides the default sample_period."""
        f1 = MadgwickFilter(beta=0.01, sample_period=1 / 90)
        f2 = MadgwickFilter(beta=0.01, sample_period=1 / 90)

        # Same inputs but different time steps
        f1.update_imu(gx=0.0, gy=0.0, gz=1.0, ax=0.0, ay=0.0, az=-9.81, dt=0.1)
        f2.update_imu(gx=0.0, gy=0.0, gz=1.0, ax=0.0, ay=0.0, az=-9.81, dt=0.001)

        # Larger dt → more rotation accumulated
        _, _, yaw1 = euler_from_quaternion(f1.quaternion)
        _, _, yaw2 = euler_from_quaternion(f2.quaternion)
        assert abs(yaw1) > abs(yaw2)

    def test_beta_controls_correction_strength(self):
        """Higher beta trusts accelerometer more, lower beta trusts gyro more."""
        # With wrong initial accel direction, high beta corrects faster
        f_high = MadgwickFilter(beta=0.5, sample_period=1 / 90)
        f_low = MadgwickFilter(beta=0.01, sample_period=1 / 90)

        # Gravity along Y instead of Z (tilted 90°)
        for _ in range(45):
            f_high.update_imu(gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=-9.81, az=0.0)
            f_low.update_imu(gx=0.0, gy=0.0, gz=0.0, ax=0.0, ay=-9.81, az=0.0)

        # High beta should have corrected more toward the accel direction
        r_high, p_high, _ = euler_from_quaternion(f_high.quaternion)
        r_low, p_low, _ = euler_from_quaternion(f_low.quaternion)

        total_rot_high = abs(r_high) + abs(p_high)
        total_rot_low = abs(r_low) + abs(p_low)
        assert total_rot_high > total_rot_low

    def test_real_rokid_data(self):
        """Feed real captured Rokid Max sensor data through the filter."""
        f = MadgwickFilter(beta=0.5, sample_period=1 / 90)

        # Real data from captured packets (stationary glasses on desk)
        # Repeated to give filter time to converge
        real_sample = (-0.1003, 9.6876, 1.9794, 0.0088, 0.0034, 0.0020)
        ax, ay, az, gx, gy, gz = real_sample

        for _ in range(180):  # 2 seconds at 90Hz
            q = f.update_imu(gx=gx, gy=gy, gz=gz, ax=ax, ay=ay, az=az)

        # Should produce a valid unit quaternion
        assert pytest.approx(q.norm, abs=1e-4) == 1.0
        # Filter should detect the tilt (gravity not purely on Z)
        roll, pitch, yaw = euler_from_quaternion(q)
        # With accel_y ≈ 9.7 and accel_z ≈ 2.0, glasses are tilted significantly
        # (gravity mostly on Y, some on Z — ~80° tilt)
        assert abs(math.degrees(roll)) + abs(math.degrees(pitch)) > 10.0
