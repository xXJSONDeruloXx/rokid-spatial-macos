"""Continuous IMU data stream — reads packets and dispatches parsed reports.

Wraps a RokidDevice (or any object with a .read() method) and provides
a high-level streaming interface with:
  - Automatic parsing of raw HID packets
  - Timestamp-based dt computation
  - Optional callbacks for each report and dt
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import Any

from rokid_spatial.parser import IMURawReport, parse_imu_report


class IMUStream:
    """Streaming reader for Rokid Max IMU data.

    Args:
        device: Any object with a `.read(size=64)` method that returns bytes.
        on_report: Optional callback invoked for each valid IMURawReport.
        on_dt: Optional callback invoked with the dt (seconds) between packets.
               First packet receives None since there's no previous timestamp.
    """

    def __init__(
        self,
        device: Any,
        on_report: Callable[[IMURawReport], None] | None = None,
        on_dt: Callable[[float | None], None] | None = None,
    ) -> None:
        self._device = device
        self._on_report = on_report
        self._on_dt = on_dt
        self._last_ts: int | None = None

    def read_batch(self, max_packets: int = 100) -> Generator[IMURawReport, None, None]:
        """Read up to max_packets valid IMU reports from the device.

        Yields parsed IMURawReport instances. Skips empty reads and
        packets with invalid report IDs.

        Args:
            max_packets: Maximum number of valid reports to yield.

        Yields:
            IMURawReport for each successfully parsed packet.
        """
        count = 0
        max_attempts = max_packets * 10  # Avoid infinite loop on bad streams
        attempts = 0

        while count < max_packets and attempts < max_attempts:
            attempts += 1
            data = self._device.read(size=64)
            if not data:
                continue

            try:
                report = parse_imu_report(data)
            except ValueError:
                continue  # Skip non-IMU or malformed packets

            # Compute dt
            dt: float | None = None
            if self._last_ts is not None and report.timestamp_ns > self._last_ts:
                dt = (report.timestamp_ns - self._last_ts) / 1e9
            self._last_ts = report.timestamp_ns

            if self._on_dt is not None:
                self._on_dt(dt)

            if self._on_report is not None:
                self._on_report(report)

            count += 1
            yield report
