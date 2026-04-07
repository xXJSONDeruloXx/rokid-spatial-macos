"""CLI entry point for live head tracking visualization."""

from __future__ import annotations

import argparse
import sys
import time

from rokid_spatial.constants import ROKID_VENDOR_ID
from rokid_spatial.device import RokidDevice, discover_rokid_devices
from rokid_spatial.parser import IMUReport, parse_imu_report
from rokid_spatial.spatial import Quaternion, apply_coordinate_adjustment, euler_from_quaternion


def cmd_discover(args: argparse.Namespace) -> None:
    """List connected Rokid devices."""
    devices = discover_rokid_devices()
    if not devices:
        print("No Rokid devices found.")
        print(f"  Expected VID: 0x{ROKID_VENDOR_ID:04X}")
        print("  Make sure glasses are connected via USB-C.")
        return

    for i, dev in enumerate(devices):
        print(f"[{i}] {dev.name}")
        print(f"    VID: 0x{dev.vendor_id:04X}  PID: 0x{dev.product_id:04X}")
        print(f"    Serial: {dev.serial}")
        print(f"    Path: {dev.path}")


def cmd_dump(args: argparse.Namespace) -> None:
    """Dump raw HID packets from the device for protocol analysis."""
    devices = discover_rokid_devices()
    if not devices:
        print("No Rokid devices found.", file=sys.stderr)
        sys.exit(1)

    dev = devices[0]
    print(f"Opening {dev.name} (serial: {dev.serial})...")

    count = 0
    max_packets = args.count
    try:
        with dev:
            print(f"Reading raw HID reports (max {max_packets})...\n")
            while count < max_packets:
                data = dev.read(size=64, timeout_ms=1000)
                if data:
                    report_id = data[0] if data else 0
                    hex_str = data.hex(" ")
                    print(f"[{count:04d}] ID=0x{report_id:02X} len={len(data):2d} | {hex_str}")
                    count += 1
                else:
                    print(".", end="", flush=True)
    except KeyboardInterrupt:
        pass

    print(f"\nCaptured {count} packets.")


def cmd_track(args: argparse.Namespace) -> None:
    """Live head tracking — print Euler angles from IMU data."""
    devices = discover_rokid_devices()
    if not devices:
        print("No Rokid devices found.", file=sys.stderr)
        sys.exit(1)

    dev = devices[0]
    print(f"Tracking {dev.name} — press Ctrl+C to stop\n")
    print(f"{'Roll':>8s}  {'Pitch':>8s}  {'Yaw':>8s}  {'ts_ms':>10s}")
    print("-" * 42)

    try:
        with dev:
            while True:
                data = dev.read(size=64, timeout_ms=100)
                if not data:
                    continue
                try:
                    report = parse_imu_report(data)
                except ValueError:
                    continue  # Skip non-IMU reports

                q = Quaternion(w=report.qw, x=report.qx, y=report.qy, z=report.qz)
                q_nwu = apply_coordinate_adjustment(q)
                roll, pitch, yaw = euler_from_quaternion(q_nwu)

                import math
                print(
                    f"{math.degrees(roll):8.2f}° "
                    f"{math.degrees(pitch):8.2f}° "
                    f"{math.degrees(yaw):8.2f}° "
                    f"{report.timestamp_ns // 1_000_000:10d}ms",
                    end="\r",
                )
    except KeyboardInterrupt:
        print("\nStopped.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rokid-track",
        description="Rokid Max spatial computing tools for macOS",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("discover", help="List connected Rokid devices")

    dump_p = sub.add_parser("dump", help="Dump raw HID packets for analysis")
    dump_p.add_argument("-n", "--count", type=int, default=100, help="Number of packets to capture")

    sub.add_parser("track", help="Live head tracking output")

    args = parser.parse_args()
    if args.command is None:
        # Default to discover
        cmd_discover(args)
    elif args.command == "discover":
        cmd_discover(args)
    elif args.command == "dump":
        cmd_dump(args)
    elif args.command == "track":
        cmd_track(args)


if __name__ == "__main__":
    main()
