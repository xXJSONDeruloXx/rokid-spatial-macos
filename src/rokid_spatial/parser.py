"""Rokid Max raw HID report parser — decodes IMU packets into structured data.

Packet layout hypothesis (64 bytes total, little-endian):
  [0]       report_id       (uint8)
  [1:9]     timestamp_ns    (uint64)  — sensor clock, nanoseconds
  [9:13]    qx              (float32) — quaternion x
  [13:17]   qy              (float32) — quaternion y
  [17:21]   qz              (float32) — quaternion z
  [21:25]   qw              (float32) — quaternion w
  [25:29]   accel_x         (float32) — accelerometer x (m/s²)
  [29:33]   accel_y         (float32) — accelerometer y
  [33:37]   accel_z         (float32) — accelerometer z
  [37:41]   gyro_x          (float32) — gyroscope x (rad/s)
  [41:45]   gyro_y          (float32) — gyroscope y
  [45:49]   gyro_z          (float32) — gyroscope z
  [49:64]   reserved/padding

This layout will be validated and refined once we capture real packets from the device.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from rokid_spatial.constants import REPORT_ID_IMU_EXTRA, REPORT_ID_IMU_SENSOR

# Minimum packet length we expect
MIN_PACKET_LENGTH = 49  # Up through gyro_z

# Valid IMU report IDs (we accept report 2 and 3 as IMU data)
VALID_IMU_REPORT_IDS = {REPORT_ID_IMU_SENSOR, REPORT_ID_IMU_EXTRA}

# Struct format for the IMU payload (after report_id byte):
#   Q  = uint64  timestamp_ns
#   10f = 10 x float32 (qx, qy, qz, qw, ax, ay, az, gx, gy, gz)
_IMU_STRUCT = struct.Struct("<Q10f")


@dataclass(frozen=True, slots=True)
class IMUReport:
    """Parsed IMU report from the Rokid Max."""

    timestamp_ns: int
    qx: float
    qy: float
    qz: float
    qw: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float


def parse_imu_report(data: bytes) -> IMUReport:
    """Parse a raw HID packet into an IMUReport.

    Args:
        data: Raw bytes from the HID device (64 bytes expected).

    Returns:
        Parsed IMUReport with timestamp, quaternion, accelerometer, and gyroscope data.

    Raises:
        ValueError: If the packet is too short or has an unexpected report ID.
    """
    if len(data) < MIN_PACKET_LENGTH:
        raise ValueError(
            f"Packet length {len(data)} is below minimum {MIN_PACKET_LENGTH}"
        )

    report_id = data[0]
    if report_id not in VALID_IMU_REPORT_IDS:
        raise ValueError(
            f"Unexpected report ID 0x{report_id:02X}; "
            f"expected IMU report ID in {[f'0x{r:02X}' for r in VALID_IMU_REPORT_IDS]}"
        )

    payload = data[1 : 1 + _IMU_STRUCT.size]
    (
        timestamp_ns,
        qx, qy, qz, qw,
        accel_x, accel_y, accel_z,
        gyro_x, gyro_y, gyro_z,
    ) = _IMU_STRUCT.unpack(payload)

    return IMUReport(
        timestamp_ns=timestamp_ns,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
        accel_x=accel_x,
        accel_y=accel_y,
        accel_z=accel_z,
        gyro_x=gyro_x,
        gyro_y=gyro_y,
        gyro_z=gyro_z,
    )
