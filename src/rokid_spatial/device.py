"""Rokid Max USB HID device discovery and connection."""

from __future__ import annotations

from dataclasses import dataclass, field

import hid

from rokid_spatial.constants import ROKID_PRODUCT_IDS, ROKID_VENDOR_ID


@dataclass
class RokidDevice:
    """Wrapper around a Rokid HID device."""

    vendor_id: int
    product_id: int
    path: bytes
    serial: str
    name: str
    _hid_device: hid.Device | None = field(default=None, repr=False, init=False)

    @property
    def is_open(self) -> bool:
        return self._hid_device is not None

    def open(self) -> None:
        """Open the HID connection to the device."""
        if self._hid_device is not None:
            return
        dev = hid.device()
        dev.open_path(self.path)
        dev.set_nonblocking(1)
        self._hid_device = dev

    def close(self) -> None:
        """Close the HID connection."""
        if self._hid_device is not None:
            self._hid_device.close()
            self._hid_device = None

    def read(self, size: int = 64, timeout_ms: int = 100) -> bytes:
        """Read a raw HID report from the device.

        Returns bytes of the requested size. Raises RuntimeError if not open.
        """
        if self._hid_device is None:
            raise RuntimeError("Device is not open. Call open() first.")
        data = self._hid_device.read(size)
        if not data:
            return b""
        return bytes(data)

    def __enter__(self) -> RokidDevice:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def discover_rokid_devices() -> list[RokidDevice]:
    """Enumerate USB HID devices and return any recognized Rokid glasses.

    Filters by ROKID_VENDOR_ID and known product IDs.
    """
    devices: list[RokidDevice] = []
    for info in hid.enumerate(vendor_id=ROKID_VENDOR_ID):
        pid = info.get("product_id", 0)
        if pid not in ROKID_PRODUCT_IDS:
            continue
        devices.append(
            RokidDevice(
                vendor_id=info["vendor_id"],
                product_id=pid,
                path=info.get("path", b""),
                serial=info.get("serial_number", ""),
                name=ROKID_PRODUCT_IDS.get(pid, f"Rokid Unknown ({pid:#06x})"),
            )
        )
    return devices
