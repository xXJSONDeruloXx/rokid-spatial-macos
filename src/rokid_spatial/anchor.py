"""World-space anchor engine for spatial window placement.

Anchors panels at a fixed orientation in world space. As the user rotates
their head, the viewport shifts so anchored content stays "pinned" in place.

Core idea:
  - Each anchor stores the HEAD ORIENTATION at the moment it was placed.
  - Every frame, we compute the angular delta between current head pose
    and each anchor's stored pose.
  - That delta maps to pixel offsets on a virtual canvas larger than the
    physical display, creating the illusion that panels are fixed in space.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from rokid_spatial.constants import FOV_DEGREES, RESOLUTION_H, RESOLUTION_W
from rokid_spatial.spatial import Quaternion, euler_from_quaternion, quaternion_multiply, normalize_quaternion


@dataclass
class AnchoredPanel:
    """A panel pinned to a world-space orientation.

    Attributes:
        panel_id: Unique identifier.
        yaw_deg: World-space yaw where the panel center lives.
        pitch_deg: World-space pitch where the panel center lives.
        width: Panel width in virtual pixels (mutable for resize).
        height: Panel height in virtual pixels (mutable for resize).
        color: RGB tuple for the panel background.
        title: Display label.
        created_at: Timestamp of creation.
    """

    panel_id: str
    yaw_deg: float
    pitch_deg: float
    width: int = 600
    height: int = 400
    color: tuple[int, int, int] = (40, 44, 52)
    title: str = "Panel"
    created_at: float = field(default_factory=time.time)


def quaternion_conjugate(q: Quaternion) -> Quaternion:
    """Return the conjugate (inverse for unit quaternions)."""
    return Quaternion(w=q.w, x=-q.x, y=-q.y, z=-q.z)


def relative_euler(current: Quaternion, reference: Quaternion) -> tuple[float, float, float]:
    """Compute the Euler angle delta from reference to current orientation.

    Returns (roll_delta, pitch_delta, yaw_delta) in degrees.
    """
    # q_delta = q_ref_inv * q_current  →  rotation FROM reference TO current
    ref_inv = quaternion_conjugate(reference)
    delta = quaternion_multiply(ref_inv, current)
    delta = normalize_quaternion(delta)
    roll, pitch, yaw = euler_from_quaternion(delta)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


@dataclass
class SpatialAnchorEngine:
    """Manages anchored panels in world space and computes viewport projections.

    The virtual canvas is conceptually infinite (or very large). The physical
    display is a viewport into this canvas. Head rotation pans the viewport.

    Attributes:
        display_w: Physical display width in pixels.
        display_h: Physical display height in pixels.
        fov_h: Horizontal field of view in degrees.
        panels: Dict of panel_id → AnchoredPanel.
        reference_orientation: Head orientation when display was "centered" (recenter pose).
        sensitivity: Degrees of head rotation per degree of viewport shift. 1.0 = 1:1 mapping.
    """

    display_w: int = RESOLUTION_W
    display_h: int = RESOLUTION_H
    fov_h: float = FOV_DEGREES
    panels: dict[str, AnchoredPanel] = field(default_factory=dict)
    reference_orientation: Quaternion | None = None
    sensitivity: float = 1.0

    @property
    def fov_v(self) -> float:
        """Vertical FOV derived from aspect ratio."""
        return self.fov_h * (self.display_h / self.display_w)

    @property
    def px_per_deg_h(self) -> float:
        return self.display_w / self.fov_h

    @property
    def px_per_deg_v(self) -> float:
        return self.display_h / self.fov_v

    def recenter(self, current_orientation: Quaternion) -> None:
        """Set current head position as the viewport center (recenter)."""
        self.reference_orientation = current_orientation

    def place_panel(
        self,
        panel_id: str,
        current_orientation: Quaternion,
        width: int = 600,
        height: int = 400,
        color: tuple[int, int, int] = (40, 44, 52),
        title: str = "Panel",
    ) -> AnchoredPanel:
        """Place a new panel at the user's current gaze direction.

        The panel's world-space position is recorded as the Euler angles
        of the current head orientation relative to the reference pose.
        """
        if self.reference_orientation is None:
            self.recenter(current_orientation)

        _, pitch_delta, yaw_delta = relative_euler(
            current_orientation, self.reference_orientation
        )

        panel = AnchoredPanel(
            panel_id=panel_id,
            yaw_deg=yaw_delta,
            pitch_deg=pitch_delta,
            width=width,
            height=height,
            color=color,
            title=title,
        )
        self.panels[panel_id] = panel
        return panel

    def remove_panel(self, panel_id: str) -> None:
        self.panels.pop(panel_id, None)

    def get_visible_panels(
        self, current_orientation: Quaternion
    ) -> list[tuple[AnchoredPanel, float, float]]:
        """Compute screen-space positions for all panels given current head pose.

        Returns a list of (panel, screen_x, screen_y) where screen_x/y are
        the CENTER of the panel in display pixel coordinates. Panels partially
        or fully off-screen are still returned (the renderer clips them).
        """
        if self.reference_orientation is None:
            return []

        # How far has the head moved from reference?
        _, head_pitch, head_yaw = relative_euler(
            current_orientation, self.reference_orientation
        )

        results: list[tuple[AnchoredPanel, float, float]] = []
        for panel in self.panels.values():
            # Panel's world position relative to reference (in degrees)
            # Subtract head movement to get screen-space offset
            delta_yaw = panel.yaw_deg - head_yaw * self.sensitivity
            delta_pitch = panel.pitch_deg - head_pitch * self.sensitivity

            # Convert angular offset to pixel position
            # Positive delta_yaw = panel is to the left → screen x decreases
            screen_x = self.display_w / 2.0 - delta_yaw * self.px_per_deg_h
            # Positive delta_pitch = panel is above → screen y decreases
            screen_y = self.display_h / 2.0 - delta_pitch * self.px_per_deg_v

            results.append((panel, screen_x, screen_y))

        return results

    def is_on_screen(self, screen_x: float, screen_y: float, panel: AnchoredPanel) -> bool:
        """Check if any part of the panel is visible on the display."""
        half_w = panel.width / 2
        half_h = panel.height / 2
        return (
            screen_x + half_w > 0
            and screen_x - half_w < self.display_w
            and screen_y + half_h > 0
            and screen_y - half_h < self.display_h
        )
