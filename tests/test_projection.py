"""RED → GREEN tests for virtual screen projection from head pose."""

import math

import pytest

from rokid_spatial.projection import ScreenProjection, VirtualScreen, project_head_to_screen
from rokid_spatial.spatial import Quaternion


class TestVirtualScreen:
    """Test the VirtualScreen configuration dataclass."""

    def test_default_screen_matches_rokid_specs(self):
        """Default VirtualScreen uses Rokid Max specs."""
        s = VirtualScreen()
        assert s.width_px == 1920
        assert s.height_px == 1080
        assert s.fov_h_deg == 45.0

    def test_custom_screen(self):
        """VirtualScreen accepts custom parameters."""
        s = VirtualScreen(width_px=2560, height_px=1440, fov_h_deg=60.0)
        assert s.width_px == 2560
        assert s.fov_h_deg == 60.0


class TestProjectHeadToScreen:
    """Test mapping head orientation → screen-space pixel coordinates."""

    def test_identity_maps_to_center(self):
        """No rotation → cursor at screen center."""
        screen = VirtualScreen()
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        proj = project_head_to_screen(q, screen)
        assert isinstance(proj, ScreenProjection)
        assert pytest.approx(proj.x, abs=1.0) == screen.width_px / 2
        assert pytest.approx(proj.y, abs=1.0) == screen.height_px / 2

    def test_yaw_right_moves_cursor_right(self):
        """Yaw rotation to the right moves the x coordinate rightward."""
        screen = VirtualScreen()
        identity = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        center = project_head_to_screen(identity, screen)

        # 5° yaw right (negative Z rotation in our convention)
        angle = math.radians(-5)
        q_right = Quaternion(
            w=math.cos(angle / 2), x=0.0, y=0.0, z=math.sin(angle / 2)
        )
        proj = project_head_to_screen(q_right, screen)
        assert proj.x > center.x

    def test_pitch_down_moves_cursor_down(self):
        """Pitch downward moves the y coordinate downward (larger y)."""
        screen = VirtualScreen()
        identity = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        center = project_head_to_screen(identity, screen)

        # Negative pitch (looking down) → larger y in screen space
        angle = math.radians(-5)
        q_down = Quaternion(
            w=math.cos(angle / 2), x=0.0, y=math.sin(angle / 2), z=0.0
        )
        proj = project_head_to_screen(q_down, screen)
        assert proj.y > center.y

    def test_projection_clamps_to_screen(self):
        """Large rotations are clamped to screen boundaries."""
        screen = VirtualScreen()
        # 90° yaw — way beyond the 45° FOV
        angle = math.radians(90)
        q_extreme = Quaternion(
            w=math.cos(angle / 2), x=0.0, y=0.0, z=math.sin(angle / 2)
        )
        proj = project_head_to_screen(q_extreme, screen)
        assert 0.0 <= proj.x <= screen.width_px
        assert 0.0 <= proj.y <= screen.height_px

    def test_projection_includes_euler_angles(self):
        """ScreenProjection includes the Euler angles used for mapping."""
        screen = VirtualScreen()
        q = Quaternion(w=1.0, x=0.0, y=0.0, z=0.0)
        proj = project_head_to_screen(q, screen)
        assert hasattr(proj, "roll_deg")
        assert hasattr(proj, "pitch_deg")
        assert hasattr(proj, "yaw_deg")
        assert pytest.approx(proj.yaw_deg, abs=0.1) == 0.0

    def test_small_yaw_proportional(self):
        """Small yaw angles produce proportional pixel offsets."""
        screen = VirtualScreen()
        angle_5 = math.radians(-5)
        angle_10 = math.radians(-10)

        q5 = Quaternion(w=math.cos(angle_5 / 2), x=0.0, y=0.0, z=math.sin(angle_5 / 2))
        q10 = Quaternion(w=math.cos(angle_10 / 2), x=0.0, y=0.0, z=math.sin(angle_10 / 2))

        center_x = screen.width_px / 2
        dx5 = project_head_to_screen(q5, screen).x - center_x
        dx10 = project_head_to_screen(q10, screen).x - center_x

        # 10° should produce roughly 2× the offset of 5°
        assert pytest.approx(dx10 / dx5, abs=0.2) == 2.0
