# -*- coding: utf-8 -*-
"""
Unit tests for Map/Radar Worker 9:
1. Live Signal Synchronization:
   - Signals between MapViewerWindow, CartographyRadarDialog, and bot runtime threads (BotWorker, AutoCartographer).
   - page_created pyqtSignal(str, QWidget) emission on page creation.
   - get_page_if_created(key) lookup.
2. Tab Switching Refresh & Non-Freezing Telemetry:
   - Switching between Map, Radar, Home guarantees canvases reload and update without pausing telemetry.
3. Region & Route Selection Synchronization:
   - Changing active region dropdown in app_shell propagates simultaneously to MapViewerWindow, CartographyRadarWindow, and RouteManager.
   - Canonical subregion resolution ('Dredgion Crash Site' -> 'Altgard').
4. Overlay HUD Radar Projection:
   - RadarProjectionWidget renders concentric range rings, sweep beam, blips, and target direction arrow.
   - RadarProjectionOverlay does not steal focus (WindowDoesNotAcceptFocus, WA_ShowWithoutActivating, MA_NOACTIVATE).
   - OverlayHUD and GameOverlay pass radar projection data and maintain click-through.
"""

import sys
import unittest

import numpy as np
from PyQt6.QtCore import QObject, Qt, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QImage, QColor
from PyQt6.QtWidgets import QApplication, QWidget, QLabel

from core.route_manager import RouteManager
from ui.app_shell import GatherionShell, CANONICAL_REGION_MAP
from ui.cartography_radar import CartographyRadarDialog
from ui.map_viewer import MapViewerWindow
from ui.overlay import GameOverlay
from ui.overlay_hud import (
    OverlayHUD,
    RadarProjectionWidget,
    RadarProjectionOverlay,
)

# Ensure QApplication instance for GUI tests
app = QApplication.instance()
if app is None:
    app = QApplication(sys.argv)


class TestOverlayHUDRadarProjection(unittest.TestCase):
    """Verifies HUD overlay radar projection, target arrow, and non-focus-stealing behavior."""

    def setUp(self):
        self.hud = OverlayHUD()
        self.overlay = GameOverlay()

    def tearDown(self):
        self.hud.close()
        self.overlay.close()

    def test_radar_projection_instantiation_and_properties(self):
        """RadarProjectionWidget initializes with blip tracking and target metrics."""
        proj = RadarProjectionWidget()
        self.assertIsNotNone(proj)
        self.assertEqual(proj.radar_radius, 75)
        self.assertEqual(len(proj.blips), 0)

        # Set blips and target direction
        blips = [
            {"dx": 15.0, "dy": -20.0, "type": "gather", "label": "Iron Ore"},
            {"dx": -30.0, "dy": 40.0, "type": "monster", "label": "Wolf"},
            {"dx": 0.0, "dy": 50.0, "type": "waypoint", "label": "#2"},
        ]
        proj.set_blips(blips)
        self.assertEqual(len(proj.blips), 3)

        proj.set_target_direction(bearing_deg=45.0, distance_m=28.5, label="Platinum Node")
        self.assertAlmostEqual(proj.target_bearing_deg, 45.0)
        self.assertAlmostEqual(proj.target_distance_m, 28.5)
        self.assertEqual(proj.target_name, "Platinum Node")

    @staticmethod
    def _render_pixels(widget):
        """Render `widget` onto a magenta QImage and return its pixels as an (H, W, 4) array."""
        size = widget.size()
        image = QImage(size, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(255, 0, 255, 255))
        widget.render(image)
        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        # Copy: the array must outlive the QImage whose buffer it views.
        return np.frombuffer(ptr, dtype=np.uint8).reshape(
            image.height(), image.width(), 4
        ).copy()

    def test_radar_projection_actually_paints_rings_blips_and_target_arrow(self):
        """paintEvent must really draw: rings/background, blips and the target arrow.

        Renders the widget to an offscreen image (magenta backing, so every
        painted pixel is detectable) and compares the renders: an empty widget
        already paints the radar disc/rings, adding blips must change pixels,
        and setting a target direction must change pixels again (the arrow).
        The sweep timer is frozen so renders are comparable.
        """
        proj = RadarProjectionWidget()
        proj._sweep_timer.stop()
        proj._sweep_angle = 0.0

        def painted_pixels(pixels):
            # Magenta backing is (B=255, G=0, R=255, A=255) in premultiplied ARGB32.
            not_magenta = ~((pixels[:, :, 0] == 255) & (pixels[:, :, 1] == 0) & (pixels[:, :, 2] == 255))
            return int(np.count_nonzero(not_magenta))

        empty = self._render_pixels(proj)
        self.assertGreater(
            painted_pixels(empty), 1000,
            "Empty radar widget painted almost nothing: rings/disc/text missing?",
        )

        proj.set_blips([
            {"distance_m": 10.0, "bearing_deg": 45.0, "type": "gather", "label": "Iron Ore"},
            {"distance_m": 25.0, "bearing_deg": 180.0, "type": "monster", "label": "Wolf"},
            {"distance_m": 40.0, "bearing_deg": 270.0, "type": "waypoint", "label": "#2"},
        ])
        proj._sweep_angle = 0.0
        with_blips = self._render_pixels(proj)
        blip_diff = int(np.count_nonzero(np.any(empty != with_blips, axis=2)))
        self.assertGreater(
            blip_diff, 50,
            "Setting blips did not change the rendered image: blips are not painted?",
        )

        proj.set_target_direction(bearing_deg=45.0, distance_m=28.5, label="Platinum Node")
        proj._sweep_angle = 0.0
        with_target = self._render_pixels(proj)
        arrow_diff = int(np.count_nonzero(np.any(with_blips != with_target, axis=2)))
        self.assertGreater(
            arrow_diff, 50,
            "Setting a target direction did not change the rendered image: arrow is not painted?",
        )

    def test_radar_projection_overlay_focus_and_window_flags(self):
        """RadarProjectionOverlay must not steal focus and must support transparent click-through."""
        overlay = RadarProjectionOverlay()
        flags = overlay.windowFlags()

        # Frameless, StaysOnTop, Tool, WindowDoesNotAcceptFocus
        self.assertTrue(bool(flags & Qt.WindowType.FramelessWindowHint))
        self.assertTrue(bool(flags & Qt.WindowType.WindowStaysOnTopHint))
        self.assertTrue(bool(flags & Qt.WindowType.Tool))
        self.assertTrue(bool(flags & Qt.WindowType.WindowDoesNotAcceptFocus))

        # Qt Attributes
        self.assertTrue(overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
        self.assertTrue(overlay.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating))
        self.assertTrue(overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))

        # Click-through toggle
        overlay.set_click_through(True)
        self.assertTrue(overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))
        overlay.set_click_through(False)
        self.assertFalse(overlay.testAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents))

    def test_overlay_hud_radar_projection_integration(self):
        """OverlayHUD integrates RadarProjectionOverlay and forwards telemetry data."""
        self.assertIsNotNone(self.hud.radar_projection)
        self.assertTrue(hasattr(self.hud, "show_radar_projection"))
        self.assertTrue(hasattr(self.hud, "hide_radar_projection"))
        self.assertTrue(hasattr(self.hud, "toggle_radar_projection"))

        # Test state update forwarding
        telemetry_data = {
            "radar_blips": [{"dx": 10.0, "dy": -10.0, "type": "gather"}],
            "target_direction_deg": 120.0,
            "target_distance_m": 42.0,
            "target_name": "Titanium",
        }
        self.hud.update_state(telemetry_data)
        self.assertEqual(len(self.hud.radar_projection.widget.blips), 1)
        self.assertAlmostEqual(self.hud.radar_projection.widget.target_bearing_deg, 120.0)
        self.assertAlmostEqual(self.hud.radar_projection.widget.target_distance_m, 42.0)
        self.assertEqual(self.hud.radar_projection.widget.target_name, "Titanium")

    def test_game_overlay_radar_projection_integration(self):
        """GameOverlay integrates RadarProjectionOverlay and forwards radar updates."""
        self.assertIsNotNone(self.overlay.radar_projection)
        self.assertTrue(hasattr(self.overlay, "set_radar_blips"))
        self.assertTrue(hasattr(self.overlay, "set_target_direction"))

        self.overlay.set_radar_blips([{"dx": -5.0, "dy": 15.0, "type": "waypoint"}])
        self.assertEqual(len(self.overlay.radar_projection.widget.blips), 1)

        self.overlay.set_target_direction(90.0, 15.0, "Node 5")
        self.assertAlmostEqual(self.overlay.radar_projection.widget.target_bearing_deg, 90.0)


class _RealWorkerSignals(QObject):
    """Minimal runtime worker exposing REAL Qt signals (no MagicMock).

    Mirrors the signals GatherionShell._bind_worker_signals connects:
    node_recorded, hud_update, state_changed, status_updated.
    """

    node_recorded = pyqtSignal(dict)
    hud_update = pyqtSignal(dict)
    state_changed = pyqtSignal(str, str)
    status_updated = pyqtSignal(str)


class TestAppShellMapRadarSignalSync(unittest.TestCase):
    """Verifies signal wiring between shell, Map, Radar, and runtime threads.

    One shell for the whole class (setUpClass): constructing GatherionShell
    costs ~23s, and every test below used to pay it again in setUp
    (7 shells ≈ 160s for the file). The shell is shared and never closed
    between tests — close() sets _closing=True and runs the shutdown
    handler, which would strand the shared instance — and is closed once
    in tearDownClass.

    Sharing is safe because every test below is order-independent by
    construction: the region test adapts to the current region, the
    canonical/route tests set explicit values, the tab test seeds its node
    in whatever region is active, and the runtime test filters recorded
    nodes by its own unique coordinates (510, 340). The probe-page test
    disconnects the listener it connects.

    Lazy pages (startup-hiz): only "home" (+ f9) is built eagerly; the
    other pages are mounted on first open. Every test below therefore
    opens the pages it reads via open_tool() before touching them —
    get_page_if_created() returns None for a page that was never opened,
    and region/route/signal propagation only reaches created pages.
    """

    @classmethod
    def setUpClass(cls):
        cls._route_mgr = RouteManager()
        cls._shell = GatherionShell(route_mgr=cls._route_mgr)

    @classmethod
    def tearDownClass(cls):
        cls._shell.close()

    def setUp(self):
        self.route_mgr = type(self)._route_mgr
        self.shell = type(self)._shell

    def test_page_created_signal_and_get_page_if_created(self):
        """page_created signal is defined and emits when pages are mounted.

        Lazy contract (startup-hiz): only "home" is built eagerly, so a
        never-opened page ("settings" — no test in this file opens it)
        must report None, and "map"/"radar" only become non-None after
        they are opened via open_tool().
        """
        self.assertTrue(hasattr(self.shell, "page_created"))
        self.assertTrue(hasattr(self.shell, "get_page_if_created"))

        # Test lookup of eager pages
        home_page = self.shell.get_page_if_created("home")
        self.assertIsNotNone(home_page)

        # Lazy contract: a page that was never opened reports None.
        self.assertIsNone(
            self.shell.get_page_if_created("settings"),
            "Never-opened lazy page must report None (placeholder is no page)",
        )

        # Open the lazy pages first; only then are they mounted.
        self.shell.open_tool("map")
        self.shell.open_tool("radar")

        map_page = self.shell.get_page_if_created("map")
        self.assertIsNotNone(map_page)

        radar_page = self.shell.get_page_if_created("radar")
        self.assertIsNotNone(radar_page)

        # Aliases lookup
        self.assertIs(self.shell.get_page_if_created("cartography_radar"), radar_page)
        self.assertIs(self.shell.get_page_if_created("cartography_radar_dialog"), radar_page)
        self.assertIs(self.shell.get_page_if_created("map_viewer"), map_page)

    def test_page_created_signal_actually_emits_on_lazy_page_mount(self):
        """page_created must really fire (with key + widget) when a page is mounted.

        The eager pages are already mounted by __init__, so the emission is
        observed through the shell's own lazy path: a factory is registered
        for a new key and _ensure_page mounts it. The emission, the payload
        and get_page_if_created are all asserted; a second _ensure_page call
        must NOT re-emit (the page is already mounted).
        """
        emitted = []
        slot = lambda key, widget: emitted.append((key, widget))
        self.shell.page_created.connect(slot)

        try:
            probe = QWidget()
            self.shell._factory_probe_page = lambda: probe
            page = self.shell._ensure_page("probe_page")

            self.assertIs(page, probe)
            self.assertEqual(len(emitted), 1, f"page_created must fire exactly once, got: {emitted}")
            self.assertEqual(emitted[0][0], "probe_page")
            self.assertIs(emitted[0][1], probe)
            self.assertIs(self.shell.get_page_if_created("probe_page"), probe)

            self.shell._ensure_page("probe_page")
            self.assertEqual(len(emitted), 1, "Already-mounted page must not re-emit page_created")
        finally:
            # Shared shell (setUpClass): the listener must not leak into later tests.
            try:
                self.shell.page_created.disconnect(slot)
            except (TypeError, RuntimeError):
                pass

    def test_region_dropdown_synchronization_to_map_and_radar(self):
        """Changing active region dropdown in shell propagates to Map and Radar simultaneously.

        Lazy pages: open map/radar first — region propagation only
        reaches created pages.
        """
        self.shell.open_tool("map")
        self.shell.open_tool("radar")
        map_win = self.shell.get_page_if_created("map")
        radar_win = self.shell.get_page_if_created("radar")

        initial_reg = self.shell.get_active_region()
        target_reg = "Altgard" if initial_reg != "Altgard" else "Verteron"

        # Change region to target
        self.shell.set_active_region(target_reg)
        self.assertEqual(self.shell.get_active_region(), target_reg)
        self.assertEqual(self.shell.region_combo.currentText(), target_reg)
        self.assertEqual(self.route_mgr.active_region, target_reg)

        # Check simultaneous propagation to Map and Radar
        self.assertEqual(map_win._region_name, target_reg)
        self.assertEqual(radar_win._region_name, target_reg)

    def test_canonical_subregion_resolution(self):
        """Selecting a subregion ('Dredgion Crash Site') resolves to canonical zone ('Altgard')."""
        self.assertEqual(self.shell._resolve_canonical_region("Dredgion Crash Site"), "Altgard")
        self.assertEqual(self.shell._resolve_canonical_region("Tolbas Forest"), "Verteron")

        # Set subregion in shell
        self.shell.set_active_region("Dredgion Crash Site")
        self.assertEqual(self.shell.get_active_region(), "Altgard")

        # Lazy pages: opening them syncs the shell's current region
        # into the page (_after_page -> _sync_region_route_to_page).
        self.shell.open_tool("map")
        self.shell.open_tool("radar")
        map_win = self.shell.get_page_if_created("map")
        radar_win = self.shell.get_page_if_created("radar")
        self.assertEqual(map_win._region_name, "Altgard")
        self.assertEqual(radar_win._region_name, "Altgard")

    def test_route_dropdown_synchronization(self):
        """Changing active route dropdown propagates to Map, Radar, and RouteManager.

        Lazy pages: open map/radar first — route propagation only
        reaches created pages.
        """
        self.shell.open_tool("map")
        self.shell.open_tool("radar")
        map_win = self.shell.get_page_if_created("map")
        radar_win = self.shell.get_page_if_created("radar")

        self.shell.set_active_route("IronLoop")
        self.assertEqual(self.shell.get_active_route(), "IronLoop")
        self.assertEqual(self.route_mgr.active_route, "IronLoop")
        self.assertEqual(map_win._route_name, "IronLoop")
        self.assertEqual(radar_win._route_name, "IronLoop")

    def test_tab_switching_refreshes_canvases_without_pausing_telemetry(self):
        """Switching between Map, Radar, and Home runs the REAL reloads and leaves timers active.

        A node is seeded into the RouteManager first; after each tab switch
        the canvases must actually contain it (map canvas._nodes / radar
        _ordered_nodes) — no reload mock, the reload itself is the assertion.

        Lazy pages: the page widgets are fetched AFTER open_tool() mounts
        them — get_page_if_created() returns None before first open.
        """
        region = self.shell.get_active_region()

        seeded = self.route_mgr.add_node(
            region=region, map_x=500, map_y=400, resource_type="Gatherable Material"
        )
        seeded_id = int(getattr(seeded, "id", seeded["id"]))

        # Switch to Map tab: real reload must load the seeded node into the canvas
        self.shell.open_tool("map")
        self.assertEqual(self.shell.current_key(), "map")
        map_win = self.shell.get_page_if_created("map")
        map_node_ids = [int(n.get("id", 0)) for n in getattr(map_win.canvas, "_nodes", [])]
        self.assertIn(seeded_id, map_node_ids)

        # Switch to Radar tab: real reload must load the seeded node into _ordered_nodes
        self.shell.open_tool("radar")
        self.assertEqual(self.shell.current_key(), "radar")
        radar_win = self.shell.get_page_if_created("radar")
        radar_node_ids = [int(n.get("id", 0)) for n in getattr(radar_win, "_ordered_nodes", [])]
        self.assertIn(seeded_id, radar_node_ids)

        # Switch to Home tab
        self.shell.open_tool("home")
        self.assertEqual(self.shell.current_key(), "home")

        # Verify background telemetry timers are still active
        self.assertTrue(self.shell._home_timer.isActive())
        self.assertTrue(self.shell._log_timer.isActive())

    def test_runtime_threads_binding_and_signal_reception(self):
        """bind_runtime_threads connects REAL worker signals to Map and Radar.

        Lazy pages: open map/radar BEFORE binding/emitting — worker
        signal handlers only forward to created pages.
        """
        self.shell.open_tool("map")
        self.shell.open_tool("radar")

        worker = _RealWorkerSignals()

        self.shell.bind_runtime_threads(worker=worker)
        self.assertIs(self.shell._worker, worker)

        # Emit a real hud_update signal: the radar canvas must receive the
        # player position through the connected slot (no direct handler call).
        radar_win = self.shell.get_page_if_created("radar")
        map_win = self.shell.get_page_if_created("map")

        hud_data = {
            "player_x": 480.0,
            "player_y": 320.0,
            "heading_deg": 135.0,
        }
        worker.hud_update.emit(hud_data)
        self.assertIsNotNone(radar_win.canvas._player_pos)
        self.assertEqual(radar_win.canvas._player_pos[0], 480.0)
        self.assertEqual(radar_win.canvas._player_pos[1], 320.0)
        self.assertEqual(radar_win.canvas._player_pos[2], 135.0)

        # Emit a real node_recorded signal: the radar must gain the node
        # through its connected slot (add_discovered_node).
        worker.node_recorded.emit({"map_x": 510, "map_y": 340, "name": "Adamantium"})
        recorded = [
            n for n in radar_win._ordered_nodes
            if int(n.get("map_x", -1)) == 510 and int(n.get("map_y", -1)) == 340
        ]
        self.assertEqual(len(recorded), 1)


if __name__ == "__main__":
    unittest.main()
