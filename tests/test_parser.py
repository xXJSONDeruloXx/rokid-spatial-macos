"""RED → GREEN tests for Rokid Max HID report parser.

Uses the REAL packet format confirmed from live capture:
  Report ID 0x11, raw accel/gyro/mag (no pre-fused quaternion).
"""

import math
import struct

import numpy as np
import pytest

from rokid_spatial.constants import REPORT_ID_IMU_DATA
from rokid_spatial.parser import IMURawReport, parse_imu_report


def _make_imu_packet(
    report_id: int = REPORT_ID_IMU_DATA,
    timestamp_ns: int = 1_000_000_000,
    accel: tuple[float, float, float] = (0.0, 0.0, -9.81),
    gyro: tuple[float, float, float] = (0.0, 0.0, 0.0),
    mag: tuple[float, float, float] = (-23.55, -36.3, -31.2),
    host_timestamp_ns: int = 2_000_000_000,
) -> bytes:
    """Build a synthetic IMU packet matching the confirmed Rokid HID layout."""
    header = struct.pack("<BQ", report_id, timestamp_ns)
    sensors = struct.pack(
        "<3f3f3f",
        *accel, *gyro, *mag,
    )
    reserved = struct.pack("<f", 0.0)
    host_ts = struct.pack("<Q", host_timestamp_ns)
    # Pad to 64 bytes total
    so_far = header + sensors + reserved + host_ts
    padding = bytes(64 - len(so_far))
    return so_far + padding


# Actual packet captured from the Rokid Max
REAL_PACKET = bytes.fromhex(
    "11304fa221b50100"
    "00a26ecdbd52001b"
    "41bc5bfd3f11df10"
    "3c20eb5e3bc49902"
    "3b6666bcc1333311"
    "c29a99f9c1000000"
    "80d27e18ab010000"
    "000000643c000000"
)


class TestParseIMUReport:
    """Test raw bytes → IMURawReport conversion."""

    def test_parse_returns_imu_report(self):
        """parse_imu_report returns an IMURawReport dataclass."""
        packet = _make_imu_packet()
        report = parse_imu_report(packet)
        assert isinstance(report, IMURawReport)

    def test_parse_extracts_timestamp(self):
        """Timestamp is correctly extracted from the packet."""
        ts = 5_000_000_000
        packet = _make_imu_packet(timestamp_ns=ts)
        report = parse_imu_report(packet)
        assert report.timestamp_ns == ts

    def test_parse_extracts_accelerometer(self):
        """Accelerometer data is parsed correctly."""
        packet = _make_imu_packet(accel=(1.5, -2.0, -9.81))
        report = parse_imu_report(packet)
        assert pytest.approx(report.accel_x, abs=1e-4) == 1.5
        assert pytest.approx(report.accel_y, abs=1e-4) == -2.0
        assert pytest.approx(report.accel_z, abs=1e-4) == -9.81

    def test_parse_extracts_gyroscope(self):
        """Gyroscope data is parsed correctly."""
        packet = _make_imu_packet(gyro=(0.01, -0.02, 0.03))
        report = parse_imu_report(packet)
        assert pytest.approx(report.gyro_x, abs=1e-5) == 0.01
        assert pytest.approx(report.gyro_y, abs=1e-5) == -0.02
        assert pytest.approx(report.gyro_z, abs=1e-5) == 0.03

    def test_parse_extracts_magnetometer(self):
        """Magnetometer data is parsed correctly."""
        packet = _make_imu_packet(mag=(-23.55, -36.3, -31.2))
        report = parse_imu_report(packet)
        assert pytest.approx(report.mag_x, abs=0.1) == -23.55
        assert pytest.approx(report.mag_y, abs=0.1) == -36.3
        assert pytest.approx(report.mag_z, abs=0.1) == -31.2

    def test_parse_extracts_host_timestamp(self):
        """Host timestamp is extracted from bytes 49-57."""
        packet = _make_imu_packet(host_timestamp_ns=9_876_543_210)
        report = parse_imu_report(packet)
        assert report.host_timestamp_ns == 9_876_543_210

    def test_parse_rejects_wrong_report_id(self):
        """Packets with non-IMU report IDs raise ValueError."""
        packet = _make_imu_packet(report_id=0x02)
        with pytest.raises(ValueError, match="report ID"):
            parse_imu_report(packet)

    def test_parse_rejects_short_packet(self):
        """Packets shorter than expected raise ValueError."""
        short = bytes(10)
        with pytest.raises(ValueError, match="length"):
            parse_imu_report(short)

    def test_accel_norm_is_gravity(self):
        """Accelerometer norm should be approximately 9.81 m/s² when stationary."""
        packet = _make_imu_packet(accel=(-0.1, 9.69, 1.98))
        report = parse_imu_report(packet)
        norm = math.sqrt(report.accel_x**2 + report.accel_y**2 + report.accel_z**2)
        assert pytest.approx(norm, abs=0.5) == 9.81

    def test_parse_real_captured_packet(self):
        """Parse an actual packet captured from the Rokid Max hardware."""
        report = parse_imu_report(REAL_PACKET)
        assert isinstance(report, IMURawReport)
        # Accelerometer should show gravity (~9.8 m/s²)
        accel_norm = math.sqrt(
            report.accel_x**2 + report.accel_y**2 + report.accel_z**2
        )
        assert 9.0 < accel_norm < 11.0, f"Accel norm {accel_norm} not near gravity"
        # Gyro should be small when stationary
        gyro_norm = math.sqrt(
            report.gyro_x**2 + report.gyro_y**2 + report.gyro_z**2
        )
        assert gyro_norm < 1.0, f"Gyro norm {gyro_norm} too large for stationary"
        # Timestamp should be reasonable (positive, non-zero)
        assert report.timestamp_ns > 0
