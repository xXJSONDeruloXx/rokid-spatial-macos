# Rokid Max USB HID Protocol

## Device Identification

| Field | Value |
|-------|-------|
| **Vendor ID** | `0x04D2` (1234 decimal, registered as "Altec Lansing Technologies") |
| **Product ID (Max)** | `0x162F` |
| **USB Speed** | Full Speed (12 Mb/s) |
| **Manufacturer** | Rokid Corporation Ltd. |
| **Serial** | Unique per device |

### All Known Rokid Product IDs
From XRLinuxDriver `rokid.c`:
- `0x162B` — Rokid Air
- `0x162C` — Rokid Air Plus (?)
- `0x162D` — Unknown variant
- `0x162E` — Unknown variant
- `0x162F` — **Rokid Max** ✅
- `0x2002` — Rokid Max 2 (?)
- `0x2180` — Rokid Max Pro (?)

## IMU Data

### Coordinate System
- Rokid SDK returns rotations in **east-up-south** coordinate system
- XRLinuxDriver converts to **north-west-up** using an adjustment quaternion:
  ```
  adjustment_quat = {w: 0.521, x: -0.478, y: 0.478, z: 0.521}
  ```
- Factory calibration includes a ~5-degree pitch offset baked into this quaternion

### Event Types (from SDK header)
| Event | Bit | Description |
|-------|-----|-------------|
| `ACC_EVENT` | 0x001 | Accelerometer |
| `GYRO_EVENT` | 0x002 | Gyroscope |
| `FUSION_EVENT` | 0x004 | Fused IMU |
| `KEY_EVENT` | 0x008 | Button press |
| `P_SENSOR_EVENT` | 0x010 | Proximity sensor |
| `ROTATION_EVENT` | 0x040 | Rotation vector |
| `GAME_ROTATION_EVENT` | 0x080 | Game rotation (no mag) ← **used by driver** |
| `L_SENSOR_EVENT` | 0x100 | Light sensor |
| `MAGNET_EVENT` | 0x200 | Magnetometer |
| `VSYNC_EVENT` | 0x400 | Display vsync |

### Rotation Data Structure
```c
typedef struct RotationData {
    uint64_t system_timestamp;     // host clock, nanoseconds
    uint64_t sensor_timestamp_ns;  // glass clock, nanoseconds
    uint32_t sequence;             // reserved
    float Q[4];                    // quaternion [x, y, z, w]
    int status;                    // reserved
} RotationData;
```

**Quaternion mapping** (SDK → standard):
- `Q[0]` → x
- `Q[1]` → y
- `Q[2]` → z
- `Q[3]` → w

### IMU Rate
- **90 Hz** (cycles per second) for the Rokid Max

### Raw Data
- 64-byte raw packets available via `RAW_EVENT` (bit 31)
- SDK handles parsing internally; for direct HID access we need to decode these

## Display Modes
| Mode | Resolution | Refresh |
|------|-----------|---------|
| 0 | 3840×1080 2D | 60 Hz |
| 1 | 3840×1080 3D (SBS) | 60 Hz |
| 4 | 3840×1200 3D (SBS) | 90 Hz |
| 5 | 3840×1200 3D (SBS) | 60 Hz |

## Device Properties (from XRLinuxDriver)
| Property | Value |
|----------|-------|
| Resolution | 1920×1080 per eye |
| FOV | 45° diagonal |
| Lens distance ratio | 0.03125 |
| IMU buffer size | 1 |
| Look-ahead constant | 20.0 ms |
| Look-ahead FT multiplier | 0.6 |
| SBS mode | Supported |
| Provides orientation | Yes (3DoF) |
| Provides position | No (no 6DoF natively) |

## Architecture: Linux vs macOS

### Linux (XRLinuxDriver)
- Uses **libusb** for hotplug detection and device enumeration
- Rokid provides a **closed-source `libGlassSDK.so`** that wraps USB communication
- The SDK opens the USB device, registers event listeners, and delivers parsed quaternions
- Linux inotify used for filesystem-based IPC

### macOS (our approach)
- **IOKit** or **hidapi** for USB HID device access
- Must **reverse-engineer the raw HID reports** since libGlassSDK.so is Linux-only (ELF)
- Alternative: use `libusb` (available via Homebrew) for cross-platform USB access
- macOS has no libusb hotplug support — must poll or use IOKit notifications

## Raw HID Protocol (needs reverse engineering)
The Rokid Max HID interface likely sends 64-byte interrupt reports containing:
1. Report ID (1 byte)
2. Sensor type indicator
3. Timestamp data
4. Quaternion/accelerometer/gyroscope payload

**Next step**: Use `hidapi` to enumerate HID interfaces and dump raw reports to decode the packet format.
