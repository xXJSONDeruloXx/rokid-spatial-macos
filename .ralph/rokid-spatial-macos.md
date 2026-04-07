# Rokid Max macOS Spatial Computing Driver

## Goal
Build an open-source macOS driver/framework that reads the Rokid Max IMU via USB HID and enables spatial computing features (head tracking, floating windows, 6DoF experiments).

## Checklist

### Phase 1: Repo & Research Setup
- [x] Create `~/Developer/personal/rokid-spatial-macos` directory
- [x] Init git repo with personal identity (xXJSONDeruloXx)
- [x] Create GitHub repo on personal account and push
- [x] Clone prior art into `research/` dir: XRLinuxDriver, breezy-desktop, VARDIAN
- [x] Analyze XRLinuxDriver HID protocol (device IDs, packet format, IMU data parsing)
- [x] Document findings in `docs/PROTOCOL.md`

### Phase 2: Project Scaffold & TDD Setup
- [x] Choose language/stack (Python + hidapi for rapid prototyping, pytest for TDD)
- [x] Set up pyproject.toml, dev dependencies (pytest, hidapi, numpy)
- [x] Create project structure (src/, tests/, docs/)
- [x] Write README.md with project vision and setup instructions

### Phase 3: Red/Green TDD — HID Device Layer
- [x] RED/GREEN: USB HID device discovery (find Rokid Max by VID/PID) — 8 tests
- [x] RED/GREEN: Raw IMU packet parsing (accel/gyro/mag from 64-byte HID reports) — 10 tests
- [x] RED/GREEN: IMU data stream (continuous reading with callbacks) — 5 tests

### Phase 4: Red/Green TDD — Spatial Math Layer
- [x] RED/GREEN: Quaternion math (normalize, multiply, euler conversion) — 14 tests
- [x] RED/GREEN: Head pose stabilization (Madgwick AHRS filter + EMA smoothing) — 8 tests
- [x] RED/GREEN: Virtual screen positioning from head pose — 8 tests

### Phase 5: Integration
- [x] Wire up end-to-end: HID → parse → Madgwick fusion → projection → output
- [x] Add CLI entry point (discover, dump, track subcommands)
- [x] Push to GitHub with MIT license

### Key Discovery
Rokid Max sends **raw sensor data** (accel/gyro/mag) via Report ID 0x11, NOT pre-fused quaternions. Sensor fusion done with Madgwick AHRS filter in software.

### Stats
- **57 tests, all passing**
- 7 source modules: constants, device, parser, fusion, spatial, projection, stream, cli
- Confirmed live HID data capture from hardware
