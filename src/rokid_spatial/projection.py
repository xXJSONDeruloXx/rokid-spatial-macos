"""Virtual screen projection — map head orientation to screen-space coordinates.

Given a quaternion head pose, computes where on a virtual screen the user's
gaze intersects. Used for head-tracked cursor positioning and virtual window
placement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rokid_spatial.constants import FOV_DEGREES, RESOLUTION_H, RESOLUTION_W
from rokid_spatial.spatial import Quaternion, euler_from_quaternion


@dataclass(frozen=True, slots=True)
class VirtualScreen:
    """Configuration for the virtual display surface.

    Attributes:
        width_px: Screen width in pixels.
        height_px: Screen height in pixels.
        fov_h_deg: Horizontal field of view in degrees.
    """

    width_px: int = RESOLUTION_W
    height_px: int = RESOLUTION_H
    fov_h_deg: float = FOV_DEGREES

    @property
    def fov_v_deg(self) -> float:
        """Vertical FOV derived from horizontal FOV and aspect ratio."""
        aspect = self.height_px / self.width_px
        return self.fov_h_deg * aspect

    @property
    def px_per_deg_h(self) -> float:
        """Pixels per degree of horizontal rotation."""
        return self.width_px / self.fov_h_deg

    @property
    def px_per_deg_v(self) -> float:
        """Pixels per degree of vertical rotation."""
        return self.height_px / self.fov_v_deg


@dataclass(frozen=True, slots=True)
class ScreenProjection:
    """Result of projecting head pose onto a virtual screen.

    Attributes:
        x: Horizontal pixel coordinate (0 = left, width = right).
        y: Vertical pixel coordinate (0 = top, height = bottom).
        roll_deg: Head roll in degrees.
        pitch_deg: Head pitch in degrees.
        yaw_deg: Head yaw in degrees.
    """

    x: float
    y: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


def project_head_to_screen(
    orientation: Quaternion,
    screen: VirtualScreen,
) -> ScreenProjection:
    """Project a head orientation quaternion onto screen-space pixel coordinates.

    Maps yaw to horizontal position and pitch to vertical position.
    Roll is reported but does not affect the x/y mapping.

    The screen center corresponds to identity quaternion (looking straight ahead).
    Positive yaw (turning left) moves the cursor left; negative yaw (right) moves right.
    Positive pitch (looking up) moves the cursor up; negative pitch (down) moves down.

    Args:
        orientation: Unit quaternion representing head orientation.
        screen: Virtual screen configuration.

    Returns:
        ScreenProjection with pixel coordinates and Euler angles.
    """
    roll, pitch, yaw = euler_from_quaternion(orientation)

    roll_deg = math.degrees(roll)
    pitch_deg = math.degrees(pitch)
    yaw_deg = math.degrees(yaw)

    # Map yaw → x:  yaw=0 → center, negative yaw (right) → larger x
    center_x = screen.width_px / 2.0
    x = center_x - yaw_deg * screen.px_per_deg_h

    # Map pitch → y: pitch=0 → center, negative pitch (down) → larger y
    center_y = screen.height_px / 2.0
    y = center_y - pitch_deg * screen.px_per_deg_v

    # Clamp to screen bounds
    x = max(0.0, min(float(screen.width_px), x))
    y = max(0.0, min(float(screen.height_px), y))

    return ScreenProjection(
        x=x,
        y=y,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
    )
