"""RED tests for Rokid Max HID report parser."""

import struct

import numpy as np
import pytest

from rokid_spatial.parser import IMUReport, parse_imu_report


def _make_fake_imu_packet(
    report_id: int = 0x02,
    timestamp_ns: int = 1_000_000_000,
    qx: float = 0.0,
    qy: float = 0.0,
    qz: float = 0.0,
    qw: float = 1.0,
    ax: float = 0.0,
    ay: float = 0.0,
    az: float = -9.81,
    gx: float = 0.0,
    gy: float = 0.0,
    gz: float = 0.0,
) -> bytes:
    """Build a synthetic IMU packet matching the expected Rokid HID layout.

    Layout hypothesis (64 bytes total):
      [0]       report_id (uint8)
      [1:9]     sensor_timestamp_ns (uint64 LE)
      [9:13]    qx (float32 LE)
      [13:17]   qy (float32 LE)
      [17:21]   qz (float32 LE)
      [21:25]   qw (float32 LE)
      [25:29]   ax (float32 LE) - accelerometer
      [29:33]   ay (float32 LE)
      [33:37]   az (float32 LE)
      [37:41]   gx (float32 LE) - gyroscope
      [41:45]   gy (float32 LE)
      [45:49]   gz (float32 LE)
      [49:64]   padding/reserved
    """
    header = struct.pack("<BQ", report_id, timestamp_ns)
    quat = struct.pack("<ffff", qx, qy, qz, qw)
    accel = struct.pack("<fff", ax, ay, az)
    gyro = struct.pack("<fff", gx, gy, gz)
    padding = bytes(64 - len(header) - len(quat) - len(accel) - len(gyro))
    return header + quat + accel + gyro + padding


class TestParseIMUReport:
    """Test raw bytes → IMUReport conversion."""

    def test_parse_returns_imu_report(self):
        """parse_imu_report returns an IMUReport dataclass."""
        packet = _make_fake_imu_packet()
        report = parse_imu_report(packet)
        assert isinstance(report, IMUReport)

    def test_parse_extracts_timestamp(self):
        """Timestamp is correctly extracted from the packet."""
        ts = 5_000_000_000
        packet = _make_fake_imu_packet(timestamp_ns=ts)
        report = parse_imu_report(packet)
        assert report.timestamp_ns == ts

    def test_parse_extracts_quaternion(self):
        """Quaternion components are extracted correctly."""
        packet = _make_fake_imu_packet(qx=0.1, qy=0.2, qz=0.3, qw=0.9)
        report = parse_imu_report(packet)
        assert pytest.approx(report.qx, abs=1e-5) == 0.1
        assert pytest.approx(report.qy, abs=1e-5) == 0.2
        assert pytest.approx(report.qz, abs=1e-5) == 0.3
        assert pytest.approx(report.qw, abs=1e-5) == 0.9

    def test_parse_extracts_accelerometer(self):
        """Accelerometer data is parsed from the packet."""
        packet = _make_fake_imu_packet(ax=1.5, ay=-2.0, az=-9.81)
        report = parse_imu_report(packet)
        assert pytest.approx(report.accel_x, abs=1e-4) == 1.5
        assert pytest.approx(report.accel_y, abs=1e-4) == -2.0
        assert pytest.approx(report.accel_z, abs=1e-4) == -9.81

    def test_parse_extracts_gyroscope(self):
        """Gyroscope data is parsed from the packet."""
        packet = _make_fake_imu_packet(gx=0.01, gy=-0.02, gz=0.03)
        report = parse_imu_report(packet)
        assert pytest.approx(report.gyro_x, abs=1e-5) == 0.01
        assert pytest.approx(report.gyro_y, abs=1e-5) == -0.02
        assert pytest.approx(report.gyro_z, abs=1e-5) == 0.03

    def test_parse_rejects_wrong_report_id(self):
        """Packets with non-IMU report IDs raise ValueError."""
        packet = _make_fake_imu_packet(report_id=0x11)  # Display report, not IMU
        with pytest.raises(ValueError, match="report ID"):
            parse_imu_report(packet)

    def test_parse_rejects_short_packet(self):
        """Packets shorter than expected raise ValueError."""
        short = bytes(10)
        with pytest.raises(ValueError, match="length"):
            parse_imu_report(short)

    def test_parse_identity_quaternion(self):
        """Identity quaternion (0,0,0,1) represents no rotation."""
        packet = _make_fake_imu_packet(qx=0.0, qy=0.0, qz=0.0, qw=1.0)
        report = parse_imu_report(packet)
        # Identity quaternion norm should be 1.0
        norm = np.sqrt(report.qx**2 + report.qy**2 + report.qz**2 + report.qw**2)
        assert pytest.approx(norm, abs=1e-6) == 1.0
