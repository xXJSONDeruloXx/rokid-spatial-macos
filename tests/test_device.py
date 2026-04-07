"""RED tests for Rokid Max HID device discovery and connection."""

import pytest

from rokid_spatial.constants import ROKID_PRODUCT_IDS, ROKID_VENDOR_ID
from rokid_spatial.device import RokidDevice, discover_rokid_devices


class TestDiscoverDevices:
    """Test USB HID device discovery."""

    def test_discover_returns_list(self, mocker):
        """discover_rokid_devices() returns a list (possibly empty)."""
        # Mock hidapi so we don't need hardware
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.enumerate.return_value = []
        result = discover_rokid_devices()
        assert isinstance(result, list)

    def test_discover_finds_rokid_max(self, mocker):
        """A connected Rokid Max (VID=0x04D2, PID=0x162F) is discovered."""
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.enumerate.return_value = [
            {
                "vendor_id": 0x04D2,
                "product_id": 0x162F,
                "path": b"/dev/hidraw0",
                "serial_number": "1501082327000865",
                "product_string": "Rokid Max",
                "manufacturer_string": "Rokid Corporation Ltd.",
                "interface_number": 3,
            }
        ]
        devices = discover_rokid_devices()
        assert len(devices) == 1
        assert devices[0].vendor_id == ROKID_VENDOR_ID
        assert devices[0].product_id == 0x162F
        assert devices[0].name == "Rokid Max"

    def test_discover_ignores_non_rokid(self, mocker):
        """Non-Rokid HID devices are filtered out."""
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.enumerate.return_value = [
            {
                "vendor_id": 0x1050,  # Yubico
                "product_id": 0x0407,
                "path": b"/dev/hidraw1",
                "serial_number": "123456",
                "product_string": "YubiKey",
                "manufacturer_string": "Yubico",
                "interface_number": 0,
            }
        ]
        devices = discover_rokid_devices()
        assert len(devices) == 0

    def test_discover_filters_by_known_product_ids(self, mocker):
        """Only known Rokid product IDs are accepted."""
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.enumerate.return_value = [
            {
                "vendor_id": 0x04D2,
                "product_id": 0xFFFF,  # Unknown PID
                "path": b"/dev/hidraw2",
                "serial_number": "unknown",
                "product_string": "Unknown Rokid",
                "manufacturer_string": "Rokid Corporation Ltd.",
                "interface_number": 3,
            }
        ]
        devices = discover_rokid_devices()
        assert len(devices) == 0


class TestRokidDevice:
    """Test RokidDevice wrapper."""

    def test_device_has_required_attributes(self, mocker):
        """RokidDevice exposes vendor_id, product_id, name, serial, path."""
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.enumerate.return_value = [
            {
                "vendor_id": 0x04D2,
                "product_id": 0x162F,
                "path": b"/dev/hidraw0",
                "serial_number": "SN123",
                "product_string": "Rokid Max",
                "manufacturer_string": "Rokid Corporation Ltd.",
                "interface_number": 3,
            }
        ]
        devices = discover_rokid_devices()
        dev = devices[0]
        assert dev.vendor_id == 0x04D2
        assert dev.product_id == 0x162F
        assert dev.serial == "SN123"
        assert dev.path == b"/dev/hidraw0"
        assert dev.name == "Rokid Max"

    def test_device_open_close(self, mocker):
        """RokidDevice can open and close a HID connection."""
        mock_device_obj = mocker.MagicMock()
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.Device.return_value = mock_device_obj

        dev = RokidDevice(
            vendor_id=0x04D2,
            product_id=0x162F,
            path=b"/dev/hidraw0",
            serial="SN123",
            name="Rokid Max",
        )
        dev.open()
        assert dev.is_open
        dev.close()
        assert not dev.is_open

    def test_device_context_manager(self, mocker):
        """RokidDevice works as a context manager."""
        mock_device_obj = mocker.MagicMock()
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.Device.return_value = mock_device_obj

        dev = RokidDevice(
            vendor_id=0x04D2,
            product_id=0x162F,
            path=b"/dev/hidraw0",
            serial="SN123",
            name="Rokid Max",
        )
        with dev:
            assert dev.is_open
        assert not dev.is_open

    def test_device_read_returns_bytes(self, mocker):
        """read() returns raw bytes from the HID device."""
        mock_device_obj = mocker.MagicMock()
        mock_device_obj.read.return_value = bytes(64)
        mock_hid = mocker.patch("rokid_spatial.device.hid")
        mock_hid.Device.return_value = mock_device_obj

        dev = RokidDevice(
            vendor_id=0x04D2,
            product_id=0x162F,
            path=b"/dev/hidraw0",
            serial="SN123",
            name="Rokid Max",
        )
        dev.open()
        data = dev.read()
        assert isinstance(data, bytes)
        assert len(data) == 64
