# rokid-spatial-macos

**macOS spatial computing driver for Rokid Max AR glasses.**

Turn your Rokid Max from a dumb secondary monitor into a head-tracked spatial display with floating virtual windows.

## What This Does

- 🎯 Reads IMU data from the Rokid Max via USB HID on macOS (no Linux required)
- 🔄 Parses quaternion orientation data at 90Hz
- 🧮 Converts head pose to screen-space coordinates
- 🪟 (Future) Enables floating multi-window spatial computing

## Status

🔴 **Early development** — reverse-engineering the HID protocol from the Rokid Max on macOS.

## Supported Devices

| Device | VID | PID | Status |
|--------|-----|-----|--------|
| Rokid Max | 0x04D2 | 0x162F | 🟡 In Progress |
| Rokid Air | 0x04D2 | 0x162B | Untested |
| Rokid Max Pro | 0x04D2 | 0x2180 | Untested |

## Requirements

- macOS 13+ (Ventura or later)
- Python 3.11+
- Rokid Max glasses connected via USB-C

## Install

```bash
git clone https://github.com/xXJSONDeruloXx/rokid-spatial-macos.git
cd rokid-spatial-macos
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
# Live head tracking output
rokid-track

# Run tests
pytest
```

## Architecture

```
src/rokid_spatial/
├── constants.py   # USB IDs, HID report IDs, device specs
├── device.py      # HID device discovery and connection
├── parser.py      # Raw HID report → structured IMU data
├── spatial.py     # Quaternion math, head pose, screen projection
└── cli.py         # CLI entry point
```

## Prior Art & Credits

This project builds on research from:
- [XRLinuxDriver](https://github.com/wheaney/XRLinuxDriver) — Linux XR driver with Rokid support
- [breezy-desktop](https://github.com/wheaney/breezy-desktop) — Linux virtual workspace for XR glasses
- [VARDIAN](https://github.com/github-nico-code/Vardian) — Windows multi-window for Rokid

## License

MIT
