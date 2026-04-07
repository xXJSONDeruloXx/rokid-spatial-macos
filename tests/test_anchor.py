"""Tests for the spatial anchor engine."""

import math

import pytest

from rokid_spatial.anchor import (
    AnchoredPanel,
    SpatialAnchorEngine,
    quaternion_conjugate,
    relative_euler,
)
from rokid_spatial.spatial import Quaternion, normalize_quaternion


def euler_to_quat(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quaternion:
    """Helper: Euler angles (degrees) → quaternion."""
    r = math.radians(roll_deg) / 2
    p = math.radians(pitch_deg) / 2
    y = math.radians(yaw_deg) / 2
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return Quaternion(
        w=cr * cp * cy + sr * sp * sy,
        x=sr * cp * cy - cr * sp * sy,
        y=cr * sp * cy + sr * cp * sy,
        z=cr * cp * sy - sr * sp * cy,
    )


class TestQuaternionConjugate:
    def test_conjugate_identity(self):
        q = Quaternion(1, 0, 0, 0)
        c = quaternion_conjugate(q)
        assert c.w == 1.0
        assert c.x == 0.0
        assert c.y == 0.0
        assert c.z == 0.0

    def test_conjugate_negates_vector(self):
        q = Quaternion(0.5, 0.1, 0.2, 0.3)
        c = quaternion_conjugate(q)
        assert c.w == 0.5
        assert c.x == pytest.approx(-0.1)
        assert c.y == pytest.approx(-0.2)
        assert c.z == pytest.approx(-0.3)


class TestRelativeEuler:
    def test_same_orientation_gives_zero_delta(self):
        q = euler_to_quat(0, 0, 30)
        roll, pitch, yaw = relative_euler(q, q)
        assert abs(roll) < 0.5
        assert abs(pitch) < 0.5
        assert abs(yaw) < 0.5

    def test_yaw_delta(self):
        ref = euler_to_quat(0, 0, 0)
        cur = euler_to_quat(0, 0, 15)
        _, _, yaw = relative_euler(cur, ref)
        assert yaw == pytest.approx(15, abs=1.0)

    def test_pitch_delta(self):
        ref = euler_to_quat(0, 0, 0)
        cur = euler_to_quat(0, 10, 0)
        _, pitch, _ = relative_euler(cur, ref)
        assert pitch == pytest.approx(10, abs=1.0)


class TestSpatialAnchorEngine:
    def test_recenter_sets_reference(self):
        engine = SpatialAnchorEngine()
        q = euler_to_quat(0, 0, 0)
        engine.recenter(q)
        assert engine.reference_orientation is not None
        assert engine.reference_orientation.w == pytest.approx(q.w)

    def test_place_panel_stores_world_position(self):
        engine = SpatialAnchorEngine()
        origin = euler_to_quat(0, 0, 0)
        engine.recenter(origin)

        # Look 20° to the right, then place
        look_right = euler_to_quat(0, 0, -20)
        panel = engine.place_panel(
            "test1", look_right, title="Test"
        )
        assert panel.yaw_deg == pytest.approx(-20, abs=2.0)

    def test_panel_stays_fixed_when_head_turns(self):
        engine = SpatialAnchorEngine()
        origin = euler_to_quat(0, 0, 0)
        engine.recenter(origin)

        # Place panel straight ahead
        panel = engine.place_panel("center", origin, width=600, height=400, title="Center")

        # Panel should be at screen center when looking straight
        visible = engine.get_visible_panels(origin)
        assert len(visible) == 1
        _, sx, sy = visible[0]
        assert sx == pytest.approx(engine.display_w / 2, abs=5)
        assert sy == pytest.approx(engine.display_h / 2, abs=5)

        # Turn head right (negative yaw) — panel should move LEFT on screen
        look_right = euler_to_quat(0, 0, -10)
        visible = engine.get_visible_panels(look_right)
        _, sx2, _ = visible[0]
        # Panel at yaw=0, head at yaw=-10, delta=+10 → screen shifts left
        assert sx2 < engine.display_w / 2  # Moved left (off to the left)

    def test_remove_panel(self):
        engine = SpatialAnchorEngine()
        origin = euler_to_quat(0, 0, 0)
        engine.recenter(origin)
        engine.place_panel("p1", origin, title="P1")
        assert "p1" in engine.panels
        engine.remove_panel("p1")
        assert "p1" not in engine.panels

    def test_remove_nonexistent_is_safe(self):
        engine = SpatialAnchorEngine()
        engine.remove_panel("nope")  # Should not raise

    def test_multiple_panels_at_different_positions(self):
        engine = SpatialAnchorEngine()
        origin = euler_to_quat(0, 0, 0)
        engine.recenter(origin)

        # Place panels at different yaw angles
        engine.place_panel("left", euler_to_quat(0, 0, 30), title="Left")
        engine.place_panel("right", euler_to_quat(0, 0, -30), title="Right")
        engine.place_panel("center", origin, title="Center")

        visible = engine.get_visible_panels(origin)
        assert len(visible) == 3

        # Sort by screen x to verify ordering
        visible.sort(key=lambda v: v[1])
        # Left panel (yaw=30) should have lowest screen_x (leftmost)
        # Right panel (yaw=-30) should have highest screen_x (rightmost)
        assert visible[0][0].title == "Left"
        assert visible[2][0].title == "Right"

    def test_is_on_screen(self):
        engine = SpatialAnchorEngine()
        panel = AnchoredPanel(
            panel_id="test", yaw_deg=0, pitch_deg=0, width=600, height=400
        )
        # Center of screen — visible
        assert engine.is_on_screen(960, 540, panel) is True
        # Way off screen
        assert engine.is_on_screen(-5000, 540, panel) is False
        # Partially visible (edge)
        assert engine.is_on_screen(200, 540, panel) is True

    def test_get_visible_without_reference_returns_empty(self):
        engine = SpatialAnchorEngine()
        q = euler_to_quat(0, 0, 0)
        assert engine.get_visible_panels(q) == []

    def test_sensitivity_scaling(self):
        engine = SpatialAnchorEngine(sensitivity=2.0)
        origin = euler_to_quat(0, 0, 0)
        engine.recenter(origin)
        engine.place_panel("p", origin, title="P")

        # Turn head 10° right
        head = euler_to_quat(0, 0, -10)

        visible = engine.get_visible_panels(head)
        _, sx, _ = visible[0]

        # With 2x sensitivity, the panel should shift MORE than with 1x
        engine2 = SpatialAnchorEngine(sensitivity=1.0)
        engine2.recenter(origin)
        engine2.place_panel("p", origin, title="P")
        visible2 = engine2.get_visible_panels(head)
        _, sx2, _ = visible2[0]

        # Higher sensitivity = more viewport shift = panel appears further from center
        shift_2x = abs(sx - engine.display_w / 2)
        shift_1x = abs(sx2 - engine2.display_w / 2)
        assert shift_2x > shift_1x
