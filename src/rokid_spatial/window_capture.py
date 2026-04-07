"""macOS window capture via Quartz/CoreGraphics.

Enumerates real macOS windows, captures their contents as images,
and converts them to pygame surfaces for rendering in the spatial display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

try:
    import Quartz
    from Quartz import (
        CGDataProviderCopyData,
        CGImageGetBytesPerRow,
        CGImageGetDataProvider,
        CGImageGetHeight,
        CGImageGetWidth,
        CGRectNull,
        CGWindowListCopyWindowInfo,
        CGWindowListCreateImage,
        kCGNullWindowID,
        kCGWindowImageBoundsIgnoreFraming,
        kCGWindowListExcludeDesktopElements,
        kCGWindowListOptionIncludingWindow,
        kCGWindowListOptionOnScreenOnly,
    )

    HAS_QUARTZ = True
except ImportError:
    HAS_QUARTZ = False


# Minimum window size to show in the picker (filters menu bar items, etc.)
MIN_WINDOW_WIDTH = 200
MIN_WINDOW_HEIGHT = 100

# How often to re-capture windows (seconds)
DEFAULT_CAPTURE_INTERVAL = 0.1  # 10 fps capture


@dataclass(frozen=True)
class WindowInfo:
    """Info about a macOS window available for capture."""

    window_id: int
    owner_name: str  # App name (e.g., "Safari", "Terminal")
    window_name: str  # Window title
    x: int
    y: int
    width: int
    height: int
    layer: int  # Window layer (0 = normal)
    owner_pid: int

    @property
    def display_name(self) -> str:
        title = self.window_name or "(untitled)"
        return f"{self.owner_name}: {title}"


@dataclass
class CapturedWindow:
    """A window capture with its pixel data."""

    info: WindowInfo
    pixels: np.ndarray | None = None  # RGB uint8 array (H, W, 3)
    captured_at: float = 0.0
    _pygame_surface: Any = None  # Cached pygame.Surface

    @property
    def is_stale(self) -> bool:
        return (time.time() - self.captured_at) > DEFAULT_CAPTURE_INTERVAL * 3

    def get_pygame_surface(self):
        """Get or create a pygame surface from the captured pixels."""
        import pygame

        if self._pygame_surface is not None and not self.is_stale:
            return self._pygame_surface

        if self.pixels is None:
            return None

        # pygame.surfarray expects (width, height, 3) — transpose from (H, W, 3)
        try:
            surface = pygame.surfarray.make_surface(
                np.ascontiguousarray(self.pixels.transpose(1, 0, 2))
            )
            self._pygame_surface = surface
            return surface
        except Exception:
            return None

    def invalidate_surface(self):
        """Force surface recreation on next access."""
        self._pygame_surface = None


def list_windows(
    min_width: int = MIN_WINDOW_WIDTH,
    min_height: int = MIN_WINDOW_HEIGHT,
    on_screen_only: bool = True,
) -> list[WindowInfo]:
    """List macOS windows available for capture.

    Returns windows sorted by app name, filtered to reasonable sizes.
    Excludes desktop elements, menu bar items, and tiny windows.
    """
    if not HAS_QUARTZ:
        return []

    opts = kCGWindowListExcludeDesktopElements
    if on_screen_only:
        opts |= kCGWindowListOptionOnScreenOnly

    window_list = CGWindowListCopyWindowInfo(opts, kCGNullWindowID)
    if not window_list:
        return []

    windows: list[WindowInfo] = []
    for w in window_list:
        bounds = w.get("kCGWindowBounds", {})
        width = int(bounds.get("Width", 0))
        height = int(bounds.get("Height", 0))

        if width < min_width or height < min_height:
            continue

        layer = int(w.get("kCGWindowLayer", 0))
        # Skip system overlay windows (layer != 0 is usually system UI)
        if layer != 0:
            continue

        owner = w.get("kCGWindowOwnerName", "")
        # Skip certain system processes
        if owner in ("Window Server", "Dock", "SystemUIServer"):
            continue

        windows.append(
            WindowInfo(
                window_id=int(w.get("kCGWindowNumber", 0)),
                owner_name=owner,
                window_name=w.get("kCGWindowName", "") or "",
                x=int(bounds.get("X", 0)),
                y=int(bounds.get("Y", 0)),
                width=width,
                height=height,
                layer=layer,
                owner_pid=int(w.get("kCGWindowOwnerPID", 0)),
            )
        )

    # Sort by app name, then window title
    windows.sort(key=lambda w: (w.owner_name.lower(), w.window_name.lower()))
    return windows


def capture_window(window_id: int, max_dimension: int = 800) -> np.ndarray | None:
    """Capture a single window's pixels as an RGB numpy array.

    Args:
        window_id: The CGWindowNumber to capture.
        max_dimension: Scale down if either dimension exceeds this. Keeps aspect ratio.

    Returns:
        numpy array of shape (H, W, 3) with RGB uint8 values, or None on failure.
    """
    if not HAS_QUARTZ:
        return None

    img = CGWindowListCreateImage(
        CGRectNull,
        kCGWindowListOptionIncludingWindow,
        window_id,
        kCGWindowImageBoundsIgnoreFraming,
    )
    if img is None:
        return None

    width = CGImageGetWidth(img)
    height = CGImageGetHeight(img)
    bpr = CGImageGetBytesPerRow(img)

    if width == 0 or height == 0:
        return None

    data = CGDataProviderCopyData(CGImageGetDataProvider(img))
    if data is None:
        return None

    # Raw data is BGRA with potential row padding
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, bpr)
    # Take only the pixel columns (4 bytes per pixel), ignore padding
    arr = arr[:, : width * 4].reshape(height, width, 4)

    # BGRA → RGB
    rgb = arr[:, :, [2, 1, 0]]

    # Scale down for performance if needed
    if max_dimension and (width > max_dimension or height > max_dimension):
        scale = max_dimension / max(width, height)
        new_w = max(1, int(width * scale))
        new_h = max(1, int(height * scale))
        # Simple nearest-neighbor downscale using numpy
        row_indices = np.linspace(0, height - 1, new_h, dtype=int)
        col_indices = np.linspace(0, width - 1, new_w, dtype=int)
        rgb = rgb[row_indices][:, col_indices]

    return np.ascontiguousarray(rgb)


@dataclass
class WindowCaptureManager:
    """Manages periodic capture of selected macOS windows.

    Designed to be called from the main render loop — captures one window
    per call in a round-robin fashion to spread the work across frames.
    """

    captures: dict[int, CapturedWindow] = field(default_factory=dict)
    _capture_order: list[int] = field(default_factory=list)
    _next_capture_idx: int = 0
    _last_capture_time: float = 0.0
    capture_interval: float = DEFAULT_CAPTURE_INTERVAL
    max_dimension: int = 800  # Max texture dimension per window

    def add_window(self, info: WindowInfo) -> CapturedWindow:
        """Start capturing a window."""
        cap = CapturedWindow(info=info)
        self.captures[info.window_id] = cap
        self._capture_order.append(info.window_id)
        # Immediately capture first frame
        self._capture_one(info.window_id)
        return cap

    def remove_window(self, window_id: int) -> None:
        """Stop capturing a window."""
        self.captures.pop(window_id, None)
        if window_id in self._capture_order:
            self._capture_order.remove(window_id)

    def update(self) -> None:
        """Call each frame. Captures one window per interval in round-robin."""
        if not self._capture_order:
            return

        now = time.time()
        if (now - self._last_capture_time) < self.capture_interval:
            return

        self._last_capture_time = now

        # Round-robin through windows
        if self._next_capture_idx >= len(self._capture_order):
            self._next_capture_idx = 0

        wid = self._capture_order[self._next_capture_idx]
        self._capture_one(wid)
        self._next_capture_idx += 1

    def _capture_one(self, window_id: int) -> None:
        """Capture a single window and update its CapturedWindow."""
        cap = self.captures.get(window_id)
        if cap is None:
            return

        pixels = capture_window(window_id, max_dimension=self.max_dimension)
        if pixels is not None:
            cap.pixels = pixels
            cap.captured_at = time.time()
            cap.invalidate_surface()

    def get_surface(self, window_id: int):
        """Get the pygame surface for a captured window, or None."""
        cap = self.captures.get(window_id)
        if cap is None:
            return None
        return cap.get_pygame_surface()
