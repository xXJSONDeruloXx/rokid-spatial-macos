"""Tests for IMU data stream — continuous reading and event dispatch."""

import struct
import time

import pytest

from rokid_spatial.constants import REPORT_ID_IMU_DATA
from rokid_spatial.stream import IMUStream


def _make_packet(timestamp_ns: int, accel_y: float = 9.8) -> bytes:
    """Build a minimal valid IMU packet."""
    header = struct.pack("<BQ", REPORT_ID_IMU_DATA, timestamp_ns)
    sensors = struct.pack("<3f3f3ff", 0.0, accel_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    host_ts = struct.pack("<Q", timestamp_ns + 1000)
    so_far = header + sensors + host_ts
    return so_far + bytes(64 - len(so_far))


class TestIMUStream:
    """Test the high-level IMU streaming reader."""

    def test_stream_processes_packets(self, mocker):
        """IMUStream reads and parses packets from a device."""
        packets = [_make_packet(ts) for ts in [1_000_000, 2_000_000, 3_000_000]]
        mock_device = mocker.MagicMock()
        mock_device.read.side_effect = packets + [b""]  # End with empty

        stream = IMUStream(mock_device)
        reports = list(stream.read_batch(max_packets=3))
        assert len(reports) == 3
        assert reports[0].timestamp_ns == 1_000_000
        assert reports[2].timestamp_ns == 3_000_000

    def test_stream_skips_empty_reads(self, mocker):
        """Empty reads (timeouts) are skipped, not returned."""
        mock_device = mocker.MagicMock()
        mock_device.read.side_effect = [
            b"",
            _make_packet(1_000_000),
            b"",
            b"",
            _make_packet(2_000_000),
            b"",
        ]

        stream = IMUStream(mock_device)
        reports = list(stream.read_batch(max_packets=2))
        assert len(reports) == 2

    def test_stream_skips_invalid_reports(self, mocker):
        """Packets with wrong report IDs are silently skipped."""
        bad_packet = bytes(64)  # Report ID 0x00, not 0x11
        good_packet = _make_packet(5_000_000)
        mock_device = mocker.MagicMock()
        mock_device.read.side_effect = [bad_packet, good_packet, b""]

        stream = IMUStream(mock_device)
        reports = list(stream.read_batch(max_packets=1))
        assert len(reports) == 1
        assert reports[0].timestamp_ns == 5_000_000

    def test_stream_callback(self, mocker):
        """on_report callback is called for each valid report."""
        mock_device = mocker.MagicMock()
        mock_device.read.side_effect = [
            _make_packet(1_000_000),
            _make_packet(2_000_000),
            b"",
        ]

        received = []
        stream = IMUStream(mock_device, on_report=lambda r: received.append(r))
        list(stream.read_batch(max_packets=2))  # Must consume the generator
        assert len(received) == 2

    def test_stream_computes_dt(self, mocker):
        """Stream computes dt between consecutive packets."""
        t1 = 1_000_000_000  # 1 second
        t2 = 1_011_111_111  # +11.1ms
        mock_device = mocker.MagicMock()
        mock_device.read.side_effect = [_make_packet(t1), _make_packet(t2), b""]

        dts = []
        stream = IMUStream(mock_device, on_dt=lambda dt: dts.append(dt))
        list(stream.read_batch(max_packets=2))  # Must consume the generator
        # First packet has no previous, dt should be None or default
        # Second packet dt should be ~0.0111s
        assert len(dts) == 2
        assert dts[0] is None  # No previous timestamp
        assert pytest.approx(dts[1], abs=0.001) == 0.01111
