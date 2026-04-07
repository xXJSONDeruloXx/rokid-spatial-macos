"""Rokid Max raw HID report parser — decodes IMU packets into structured data.

Confirmed packet layout from live capture (64 bytes, little-endian):
  [0]       report_id       (uint8)  — always 0x11 for IMU data
  [1:9]     timestamp_ns    (uint64) — sensor clock, nanoseconds
  [9:13]    accel_x         (float32) — accelerometer x (m/s²)
  [13:17]   accel_y         (float32) — accelerometer y
  [17:21]   accel_z         (float32) — accelerometer z
  [21:25]   gyro_x          (float32) — gyroscope x (rad/s)
  [25:29]   gyro_y          (float32) — gyroscope y
  [29:33]   gyro_z          (float32) — gyroscope z
  [33:37]   mag_x           (float32) — magnetometer x (µT)
  [37:41]   mag_y           (float32) — magnetometer y
  [41:45]   mag_z           (float32) — magnetometer z
  [45:49]   reserved        (float32) — always -0.0
  [49:57]   host_timestamp  (uint64) — host/system clock nanoseconds
  [57:61]   unknown         (float32) — small value ~0.0039
  [61:64]   padding         (3 bytes of 0x00)

Key finding: Rokid Max sends RAW sensor data, not fused quaternions.
Quaternion fusion must be done in software (Madgwick, Mahony, or complementary filter).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from rokid_spatial.constants import REPORT_ID_IMU_DATA

# Minimum packet length we expect
MIN_PACKET_LENGTH = 49

# Struct format for the IMU payload (after report_id byte):
#   Q    = uint64 timestamp_ns
#   3f   = accel x,y,z
#   3f   = gyro x,y,z
#   3f   = mag x,y,z
#   f    = reserved
#   Q    = host_timestamp
_IMU_STRUCT = struct.Struct("<Q3f3f3ffQ")


@dataclass(frozen=True, slots=True)
class IMURawReport:
    """Parsed raw IMU report from the Rokid Max.

    Contains accelerometer, gyroscope, and magnetometer data.
    NO quaternion — fusion must be done in software.
    """

    timestamp_ns: int
    accel_x: float  # m/s²
    accel_y: float
    accel_z: float
    gyro_x: float   # rad/s
    gyro_y: float
    gyro_z: float
    mag_x: float    # µT (microtesla)
    mag_y: float
    mag_z: float
    host_timestamp_ns: int


# Keep backward compat alias
IMUReport = IMURawReport


def parse_imu_report(data: bytes) -> IMURawReport:
    """Parse a raw HID packet into an IMURawReport.

    Args:
        data: Raw bytes from the HID device (64 bytes expected).

    Returns:
        Parsed IMURawReport with timestamp, accel, gyro, and magnetometer.

    Raises:
        ValueError: If the packet is too short or has an unexpected report ID.
    """
    if len(data) < MIN_PACKET_LENGTH:
        raise ValueError(
            f"Packet length {len(data)} is below minimum {MIN_PACKET_LENGTH}"
        )

    report_id = data[0]
    if report_id != REPORT_ID_IMU_DATA:
        raise ValueError(
            f"Unexpected report ID 0x{report_id:02X}; "
            f"expected IMU report ID 0x{REPORT_ID_IMU_DATA:02X}"
        )

    payload = data[1 : 1 + _IMU_STRUCT.size]
    (
        timestamp_ns,
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
        mag_x, mag_y, mag_z,
        _reserved,
        host_timestamp_ns,
    ) = _IMU_STRUCT.unpack(payload)

    return IMURawReport(
        timestamp_ns=timestamp_ns,
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
        mag_x=mag_x,
        mag_y=mag_y,
        mag_z=mag_z,
        host_timestamp_ns=host_timestamp_ns,
    )
