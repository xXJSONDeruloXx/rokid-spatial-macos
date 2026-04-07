"""Spatial display — fullscreen pygame app for head-tracked anchored panels.

Renders on the Rokid Max's secondary display. Panels are pinned in world
space; head rotation pans the viewport so panels stay fixed.

Supports two panel types:
  - Mock panels with fake content (keys 1-5)
  - REAL macOS window captures anchored in space (key W to open picker)

Controls:
  SPACE     — Recenter (set current pose as origin)
  1-5       — Place a mock panel at current gaze direction
  W         — Open window picker — anchor a real macOS window
  D + 1-5   — Delete panel by number
  Q / ESC   — Quit (or close picker if open)
  +/-       — Adjust display size (sensitivity)
  R         — Reset all panels

Works in two modes:
  --mock    — Simulated head tracking via mouse (for development without glasses)
  default   — Live IMU from Rokid Max via HID
"""

from __future__ import annotations

import math
import sys
import time

import pygame

from rokid_spatial.anchor import AnchoredPanel, SpatialAnchorEngine
from rokid_spatial.spatial import Quaternion, euler_from_quaternion
from rokid_spatial.window_capture import (
    CapturedWindow,
    WindowCaptureManager,
    WindowInfo,
    list_windows,
)

# Colors
BG_COLOR = (10, 10, 15)
GRID_COLOR = (30, 30, 40)
TEXT_COLOR = (200, 200, 210)
ACCENT_COLORS = [
    (59, 130, 246),   # Blue
    (16, 185, 129),   # Green
    (245, 158, 11),   # Amber
    (239, 68, 68),    # Red
    (139, 92, 246),   # Purple
]
HUD_BG = (20, 20, 30, 180)
CROSSHAIR_COLOR = (100, 100, 120)

PANEL_TITLES = [
    "Terminal",
    "Browser",
    "Code Editor",
    "Slack",
    "Notes",
]

# Panel content mockups
PANEL_CONTENTS = {
    "Terminal": [
        "$ rokid-track anchor",
        "Tracking Rokid Max (Madgwick AHRS)",
        "Spatial anchoring active",
        "  Roll:   0.42°",
        "  Pitch: -1.23°",
        "  Yaw:    5.67°",
        "$ _",
    ],
    "Browser": [
        "╔═══════════════════════════════╗",
        "║  github.com/xXJSONDeruloXx   ║",
        "╠═══════════════════════════════╣",
        "║                               ║",
        "║  rokid-spatial-macos           ║",
        "║  ★ 42  🍴 7                   ║",
        "║                               ║",
        "╚═══════════════════════════════╝",
    ],
    "Code Editor": [
        'def anchor_panel(self, pose):',
        '    """Pin panel to world."""',
        '    yaw = pose.yaw_deg',
        '    pitch = pose.pitch_deg',
        '    self.panels[id] = Panel(',
        '        world_yaw=yaw,',
        '        world_pitch=pitch,',
        '    )',
    ],
    "Slack": [
        "#general",
        "",
        "kurt: spatial computing on mac!",
        "bot:  build passing ✅",
        "kurt: head tracking works 🎯",
        "",
        "Type a message...",
    ],
    "Notes": [
        "## Spatial Computing TODO",
        "",
        "✅ HID protocol reverse-eng",
        "✅ IMU parsing (accel/gyro/mag)",
        "✅ Madgwick AHRS fusion",
        "✅ Head tracking output",
        "⬜ Multi-window anchoring  ← HERE",
        "⬜ Window capture/mirroring",
    ],
}


def euler_to_quaternion(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Quaternion:
    """Create a quaternion from Euler angles in degrees (for mock mode)."""
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


class MockIMU:
    """Mouse-driven fake IMU for development without hardware."""

    def __init__(self, sensitivity: float = 0.15):
        self.yaw = 0.0
        self.pitch = 0.0
        self.sensitivity = sensitivity
        self._last_mouse = None

    def update(self, mouse_rel: tuple[int, int]) -> Quaternion:
        dx, dy = mouse_rel
        self.yaw -= dx * self.sensitivity
        self.pitch += dy * self.sensitivity
        self.pitch = max(-60, min(60, self.pitch))
        return euler_to_quaternion(0, self.pitch, self.yaw)

    @property
    def quaternion(self) -> Quaternion:
        return euler_to_quaternion(0, self.pitch, self.yaw)


class LiveIMU:
    """Real Rokid Max IMU via HID + Madgwick fusion."""

    def __init__(self):
        from rokid_spatial.device import discover_rokid_devices
        from rokid_spatial.fusion import MadgwickFilter

        devices = discover_rokid_devices()
        if not devices:
            raise RuntimeError(
                "No Rokid device found. Use --mock for mouse simulation."
            )
        self.device = devices[0]
        self.device.open()
        self.ahrs = MadgwickFilter(beta=0.1, sample_period=1 / 90)
        self._last_ts: int | None = None

    def update(self) -> Quaternion:
        """Read all available IMU packets and return latest fused orientation."""
        from rokid_spatial.parser import parse_imu_report

        latest_q = self.ahrs.quaternion
        # Drain all available packets for lowest latency
        for _ in range(20):
            data = self.device.read(size=64, timeout_ms=1)
            if not data:
                break
            try:
                report = parse_imu_report(data)
            except ValueError:
                continue

            dt = 1.0 / 90.0
            if self._last_ts is not None and report.timestamp_ns > self._last_ts:
                dt = (report.timestamp_ns - self._last_ts) / 1e9
                dt = max(0.001, min(dt, 0.1))
            self._last_ts = report.timestamp_ns

            latest_q = self.ahrs.update_imu(
                gx=report.gyro_x,
                gy=report.gyro_y,
                gz=report.gyro_z,
                ax=report.accel_x,
                ay=report.accel_y,
                az=report.accel_z,
                dt=dt,
            )
        return latest_q

    @property
    def quaternion(self) -> Quaternion:
        return self.ahrs.quaternion

    def close(self):
        if self.device.is_open:
            self.device.close()


def draw_panel(
    surface: pygame.Surface,
    panel: AnchoredPanel,
    cx: float,
    cy: float,
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    is_visible: bool,
    panel_index: int,
    capture_surface: pygame.Surface | None = None,
):
    """Draw an anchored panel at the given screen center coordinates.

    If capture_surface is provided, renders the live window capture.
    Otherwise falls back to mock content.
    """
    left = int(cx - panel.width / 2)
    top = int(cy - panel.height / 2)
    rect = pygame.Rect(left, top, panel.width, panel.height)

    # Panel shadow
    shadow_rect = rect.move(4, 4)
    shadow_surf = pygame.Surface((panel.width, panel.height), pygame.SRCALPHA)
    shadow_surf.fill((0, 0, 0, 80))
    surface.blit(shadow_surf, shadow_rect.topleft)

    # Panel background
    pygame.draw.rect(surface, panel.color, rect, border_radius=8)

    # Title bar
    title_bar = pygame.Rect(left, top, panel.width, 32)
    accent = ACCENT_COLORS[panel_index % len(ACCENT_COLORS)]
    pygame.draw.rect(surface, accent, title_bar, border_radius=8)
    # Flatten bottom corners of title bar
    pygame.draw.rect(
        surface, accent, pygame.Rect(left, top + 16, panel.width, 16)
    )

    # Title text
    label = f"  {panel_index + 1}  {panel.title}"
    title_text = title_font.render(label, True, (255, 255, 255))
    surface.blit(title_text, (left + 8, top + 6))

    # Window control dots
    for i, dot_color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        pygame.draw.circle(
            surface,
            dot_color,
            (left + panel.width - 20 - i * 22, top + 16),
            6,
        )

    # Content area
    content_rect = pygame.Rect(left + 2, top + 32, panel.width - 4, panel.height - 34)

    if capture_surface is not None:
        # Render live window capture — scale to fit content area
        scaled = pygame.transform.smoothscale(capture_surface, (content_rect.width, content_rect.height))
        surface.blit(scaled, content_rect.topleft)
    else:
        # Fallback: mock content
        content_lines = PANEL_CONTENTS.get(panel.title, ["(empty)"])
        mono_font = font
        y_offset = top + 42
        for line in content_lines:
            if y_offset > top + panel.height - 20:
                break
            line_surf = mono_font.render(line, True, (180, 180, 190))
            surface.blit(line_surf, (left + 14, y_offset))
            y_offset += 22

    # Border
    pygame.draw.rect(surface, accent, rect, width=2, border_radius=8)


def draw_window_picker(
    surface: pygame.Surface,
    font: pygame.font.Font,
    title_font: pygame.font.Font,
    windows: list[WindowInfo],
    selected_idx: int,
    scroll_offset: int,
):
    """Draw a window picker overlay for selecting real macOS windows."""
    picker_w = 700
    visible_rows = 12
    row_h = 28
    header_h = 50
    picker_h = header_h + visible_rows * row_h + 20
    px = (1920 - picker_w) // 2
    py = (1080 - picker_h) // 2

    # Background
    bg_surf = pygame.Surface((picker_w, picker_h), pygame.SRCALPHA)
    bg_surf.fill((20, 22, 30, 240))
    surface.blit(bg_surf, (px, py))
    pygame.draw.rect(surface, (59, 130, 246), (px, py, picker_w, picker_h), 2, border_radius=8)

    # Header
    header = title_font.render("Select a window to anchor in space", True, (255, 255, 255))
    surface.blit(header, (px + 20, py + 14))
    hint = font.render("Up/Down=navigate  Enter=select  Esc=cancel", True, (140, 140, 160))
    surface.blit(hint, (px + 20, py + 34))

    # Window list
    for i in range(visible_rows):
        idx = scroll_offset + i
        if idx >= len(windows):
            break
        w = windows[idx]
        row_y = py + header_h + i * row_h

        if idx == selected_idx:
            sel_surf = pygame.Surface((picker_w - 8, row_h - 2), pygame.SRCALPHA)
            sel_surf.fill((59, 130, 246, 80))
            surface.blit(sel_surf, (px + 4, row_y))

        # App name in accent color, window title in gray
        app_text = font.render(w.owner_name, True, (100, 180, 255))
        surface.blit(app_text, (px + 20, row_y + 5))

        title_text = font.render(
            f"  {w.window_name[:50]}" if w.window_name else "  (untitled)",
            True,
            (180, 180, 190),
        )
        surface.blit(title_text, (px + 20 + app_text.get_width(), row_y + 5))

        # Dimensions
        dim_text = font.render(f"{w.width}x{w.height}", True, (100, 100, 120))
        surface.blit(dim_text, (px + picker_w - dim_text.get_width() - 20, row_y + 5))


def draw_hud(
    surface: pygame.Surface,
    font: pygame.font.Font,
    engine: SpatialAnchorEngine,
    head_yaw: float,
    head_pitch: float,
    fps: float,
    mock_mode: bool,
):
    """Draw the heads-up display overlay."""
    lines = [
        f"FPS: {fps:.0f}  |  Panels: {len(engine.panels)}",
        f"Head: yaw={head_yaw:+.1f}°  pitch={head_pitch:+.1f}°",
        "",
        "SPACE=recenter  1-5=mock panel  W=capture window",
        "D+1-5=delete  +/-=sensitivity  R=reset  Q=quit",
    ]
    if mock_mode:
        lines.insert(0, "🖱 MOCK MODE — move mouse to simulate head tracking")

    hud_w = 540
    hud_h = len(lines) * 22 + 16
    hud_surf = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
    hud_surf.fill((15, 15, 25, 200))

    for i, line in enumerate(lines):
        text = font.render(line, True, TEXT_COLOR)
        hud_surf.blit(text, (10, 8 + i * 22))

    surface.blit(hud_surf, (10, 10))


def draw_minimap(
    surface: pygame.Surface,
    engine: SpatialAnchorEngine,
    head_yaw: float,
    head_pitch: float,
):
    """Draw a minimap showing panel positions in world space."""
    map_w, map_h = 200, 120
    map_x = engine.display_w - map_w - 10
    map_y = engine.display_h - map_h - 10

    # Background
    map_surf = pygame.Surface((map_w, map_h), pygame.SRCALPHA)
    map_surf.fill((15, 15, 25, 180))

    # Scale: map covers ±90° yaw, ±45° pitch
    scale_h = map_w / 180.0  # px per degree
    scale_v = map_h / 90.0

    # Draw panels as dots
    for i, panel in enumerate(engine.panels.values()):
        px = int(map_w / 2 - panel.yaw_deg * scale_h)
        py = int(map_h / 2 - panel.pitch_deg * scale_v)
        color = ACCENT_COLORS[i % len(ACCENT_COLORS)]
        pygame.draw.circle(map_surf, color, (px, py), 5)

    # Draw FOV rectangle (current viewport)
    fov_w = engine.fov_h * scale_h
    fov_h = engine.fov_v * scale_v
    fov_x = map_w / 2 - head_yaw * scale_h - fov_w / 2
    fov_y = map_h / 2 - head_pitch * scale_v - fov_h / 2
    pygame.draw.rect(
        map_surf, (100, 100, 120), (int(fov_x), int(fov_y), int(fov_w), int(fov_h)), 1
    )

    # Crosshair (head center)
    cx = int(map_w / 2 - head_yaw * scale_h)
    cy = int(map_h / 2 - head_pitch * scale_v)
    pygame.draw.line(map_surf, (200, 200, 200), (cx - 4, cy), (cx + 4, cy))
    pygame.draw.line(map_surf, (200, 200, 200), (cx, cy - 4), (cx, cy + 4))

    surface.blit(map_surf, (map_x, map_y))


def draw_crosshair(surface: pygame.Surface, w: int, h: int):
    """Draw a subtle center crosshair."""
    cx, cy = w // 2, h // 2
    length = 20
    gap = 5
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        start = (cx + dx * gap, cy + dy * gap)
        end = (cx + dx * length, cy + dy * length)
        pygame.draw.line(surface, CROSSHAIR_COLOR, start, end, 1)


def run_spatial_display(mock: bool = False, fullscreen: bool = True):
    """Main loop for the spatial anchoring display.

    Args:
        mock: Use mouse simulation instead of real IMU.
        fullscreen: Run fullscreen (for Rokid display). False = windowed for dev.
    """
    pygame.init()
    pygame.font.init()

    # Display setup
    if fullscreen:
        # Try to use the Rokid's display (second monitor)
        num_displays = pygame.display.get_num_displays() if hasattr(pygame.display, 'get_num_displays') else 1
        if num_displays > 1:
            # Use second display
            import os
            os.environ['SDL_VIDEO_WINDOW_POS'] = '1920,0'  # Offset to second display
        screen = pygame.display.set_mode((1920, 1080), pygame.NOFRAME)
    else:
        screen = pygame.display.set_mode((1920, 1080), pygame.RESIZABLE)

    pygame.display.set_caption("Rokid Spatial Anchor")

    # Fonts
    try:
        mono_font = pygame.font.SysFont("menlo", 14)
        title_font = pygame.font.SysFont("menlo", 14, bold=True)
        hud_font = pygame.font.SysFont("menlo", 13)
    except Exception:
        mono_font = pygame.font.Font(None, 16)
        title_font = pygame.font.Font(None, 16)
        hud_font = pygame.font.Font(None, 15)

    # Engine
    engine = SpatialAnchorEngine()

    # IMU source
    imu: MockIMU | LiveIMU
    if mock:
        imu = MockIMU(sensitivity=0.15)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)
    else:
        try:
            imu = LiveIMU()
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            pygame.quit()
            sys.exit(1)

    clock = pygame.time.Clock()
    running = True
    panel_counter = 0
    d_held = False
    head_yaw = 0.0
    head_pitch = 0.0

    # Window capture state
    capture_mgr = WindowCaptureManager()
    # Maps panel_id → window_id for panels that are real window captures
    panel_window_map: dict[str, int] = {}

    # Window picker state
    picker_open = False
    picker_windows: list[WindowInfo] = []
    picker_selected = 0
    picker_scroll = 0

    # Auto-recenter after first few frames
    frames_until_recenter = 30

    while running:
        dt = clock.tick(90) / 1000.0  # Target 90fps to match IMU rate

        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if picker_open:
                    # Window picker handles its own keys
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        picker_open = False
                        if mock:
                            pygame.event.set_grab(True)
                            pygame.mouse.set_visible(False)
                    elif event.key == pygame.K_UP:
                        picker_selected = max(0, picker_selected - 1)
                        if picker_selected < picker_scroll:
                            picker_scroll = picker_selected
                    elif event.key == pygame.K_DOWN:
                        picker_selected = min(len(picker_windows) - 1, picker_selected + 1)
                        if picker_selected >= picker_scroll + 12:
                            picker_scroll = picker_selected - 11
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if picker_windows and picker_selected < len(picker_windows):
                            win_info = picker_windows[picker_selected]
                            # Anchor this window
                            panel_counter += 1
                            pid = f"win_{panel_counter}"
                            orientation = imu.quaternion
                            # Size the panel to match window aspect ratio
                            aspect = win_info.width / max(win_info.height, 1)
                            pw = min(700, int(500 * aspect))
                            ph = min(500, int(pw / aspect) + 32)  # +32 for title bar
                            color_idx = panel_counter % len(ACCENT_COLORS)
                            panel = engine.place_panel(
                                panel_id=pid,
                                current_orientation=orientation,
                                width=pw,
                                height=ph,
                                color=(35 + color_idx * 5, 39 + color_idx * 3, 47 + color_idx * 4),
                                title=win_info.display_name[:40],
                            )
                            # Start capturing
                            capture_mgr.add_window(win_info)
                            panel_window_map[pid] = win_info.window_id
                            print(
                                f"Anchored [{win_info.owner_name}: {win_info.window_name}] "
                                f"at yaw={panel.yaw_deg:.1f}° pitch={panel.pitch_deg:.1f}°"
                            )
                        picker_open = False
                        if mock:
                            pygame.event.set_grab(True)
                            pygame.mouse.set_visible(False)
                else:
                    # Normal mode keys
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        engine.recenter(imu.quaternion)
                        print("Recentered!")
                    elif event.key == pygame.K_w:
                        # Open window picker
                        picker_windows = list_windows()
                        if picker_windows:
                            picker_open = True
                            picker_selected = 0
                            picker_scroll = 0
                            if mock:
                                pygame.event.set_grab(False)
                                pygame.mouse.set_visible(True)
                            print(f"Window picker: {len(picker_windows)} windows available")
                        else:
                            print("No capturable windows found")
                    elif event.key == pygame.K_d:
                        d_held = True
                    elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
                        idx = event.key - pygame.K_1  # 0-based
                        if d_held:
                            # Delete panel
                            panel_ids = list(engine.panels.keys())
                            if idx < len(panel_ids):
                                pid = panel_ids[idx]
                                # Clean up capture if it's a real window
                                if pid in panel_window_map:
                                    capture_mgr.remove_window(panel_window_map.pop(pid))
                                engine.remove_panel(pid)
                                print(f"Deleted panel {idx + 1}")
                        else:
                            # Place mock panel
                            orientation = imu.quaternion
                            panel_counter += 1
                            title = PANEL_TITLES[idx % len(PANEL_TITLES)]
                            color_idx = idx % len(ACCENT_COLORS)
                            panel = engine.place_panel(
                                panel_id=f"panel_{panel_counter}",
                                current_orientation=orientation,
                                width=520,
                                height=340,
                                color=(35 + color_idx * 5, 39 + color_idx * 3, 47 + color_idx * 4),
                                title=title,
                            )
                            print(f"Placed '{title}' at yaw={panel.yaw_deg:.1f}° pitch={panel.pitch_deg:.1f}°")
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        engine.sensitivity = min(2.0, engine.sensitivity + 0.1)
                        print(f"Sensitivity: {engine.sensitivity:.1f}")
                    elif event.key == pygame.K_MINUS:
                        engine.sensitivity = max(0.3, engine.sensitivity - 0.1)
                        print(f"Sensitivity: {engine.sensitivity:.1f}")
                    elif event.key == pygame.K_r:
                        # Reset everything
                        for pid in list(panel_window_map.keys()):
                            capture_mgr.remove_window(panel_window_map[pid])
                        panel_window_map.clear()
                        engine.panels.clear()
                        panel_counter = 0
                        print("Reset all panels")
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_d:
                    d_held = False

        # --- Update IMU ---
        if mock:
            if not picker_open:
                mouse_rel = pygame.mouse.get_rel()
                orientation = imu.update(mouse_rel)
            else:
                orientation = imu.quaternion
        else:
            orientation = imu.update()

        # Auto-recenter
        if frames_until_recenter > 0:
            frames_until_recenter -= 1
            if frames_until_recenter == 0:
                engine.recenter(orientation)

        # Update window captures (round-robin, one per interval)
        capture_mgr.update()

        # Compute head angles for HUD
        if engine.reference_orientation is not None:
            from rokid_spatial.anchor import relative_euler
            _, head_pitch, head_yaw = relative_euler(
                orientation, engine.reference_orientation
            )
        else:
            head_yaw = 0.0
            head_pitch = 0.0

        # --- Render ---
        screen.fill(BG_COLOR)

        # Subtle grid (shifts with head for parallax depth cue)
        grid_spacing = 80
        offset_x = int(head_yaw * engine.px_per_deg_h * 0.3) % grid_spacing
        offset_y = int(head_pitch * engine.px_per_deg_v * 0.3) % grid_spacing
        for gx in range(-grid_spacing, 1920 + grid_spacing, grid_spacing):
            pygame.draw.line(screen, GRID_COLOR, (gx + offset_x, 0), (gx + offset_x, 1080), 1)
        for gy in range(-grid_spacing, 1080 + grid_spacing, grid_spacing):
            pygame.draw.line(screen, GRID_COLOR, (0, gy + offset_y), (1920, gy + offset_y), 1)

        # Draw panels
        visible_panels = engine.get_visible_panels(orientation)
        for i, (panel, sx, sy) in enumerate(visible_panels):
            is_vis = engine.is_on_screen(sx, sy, panel)
            # Get live capture surface if this is a real window panel
            cap_surface = None
            pid = panel.panel_id
            if pid in panel_window_map:
                cap_surface = capture_mgr.get_surface(panel_window_map[pid])
            draw_panel(screen, panel, sx, sy, mono_font, title_font, is_vis, i, cap_surface)

        # Crosshair
        draw_crosshair(screen, 1920, 1080)

        # HUD
        fps = clock.get_fps()
        draw_hud(screen, hud_font, engine, head_yaw, head_pitch, fps, mock)

        # Minimap
        draw_minimap(screen, engine, head_yaw, head_pitch)

        # Window picker overlay (on top of everything)
        if picker_open:
            draw_window_picker(
                screen, hud_font, title_font, picker_windows,
                picker_selected, picker_scroll,
            )

        pygame.display.flip()

    # Cleanup
    if not mock and isinstance(imu, LiveIMU):
        imu.close()
    if mock:
        pygame.event.set_grab(False)
        pygame.mouse.set_visible(True)
    pygame.quit()
