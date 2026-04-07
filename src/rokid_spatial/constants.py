"""Rokid Max USB HID constants derived from protocol analysis."""

# USB Vendor/Product IDs
ROKID_VENDOR_ID = 0x04D2  # 1234 decimal

# Known Rokid product IDs
ROKID_PRODUCT_IDS = {
    0x162B: "Rokid Air",
    0x162C: "Rokid Air Plus",
    0x162D: "Rokid Unknown 162D",
    0x162E: "Rokid Unknown 162E",
    0x162F: "Rokid Max",
    0x2002: "Rokid Max 2",
    0x2180: "Rokid Max Pro",
}

# HID Report IDs
REPORT_ID_OUTPUT = 0x01        # Host → Device (63 bytes payload)
REPORT_ID_IMU_SENSOR = 0x02   # Device → Host: IMU sensor data (64 bytes)
REPORT_ID_IMU_EXTRA = 0x03    # Device → Host: additional IMU data
REPORT_ID_CONTROL = 0x04      # Device → Host: control/status reports
REPORT_ID_DISPLAY = 0x11      # Device → Host: display status (report ID 17)

# HID Report sizes (bytes, excluding report ID)
REPORT_PAYLOAD_SIZE = 63  # 504 bits

# IMU characteristics
IMU_RATE_HZ = 90  # Effective IMU update rate after SDK decimation
IMU_RAW_INTERVAL_US = 2000  # Raw report interval from HID descriptor

# Display modes
DISPLAY_MODE_2D_1080_60 = 0
DISPLAY_MODE_3D_SBS_1080_60 = 1
DISPLAY_MODE_3D_SBS_1200_90 = 4
DISPLAY_MODE_3D_SBS_1200_60 = 5

# Device specs
RESOLUTION_W = 1920
RESOLUTION_H = 1080
FOV_DEGREES = 45.0
LENS_DISTANCE_RATIO = 0.03125

# Coordinate system adjustment quaternion
# Converts Rokid east-up-south → north-west-up with ~5° factory calibration offset
ADJUSTMENT_QUAT = (0.521, -0.478, 0.478, 0.521)  # (w, x, y, z)
