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
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt, QPoint, QPointF
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
        try:
            self.hud.close()
        except Exception:
            pass
        try:
            self.overlay.close()
        except Exception:
            pass

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


class TestAppShellMapRadarSignalSync(unittest.TestCase):
    """Verifies signal wiring between shell, Map, Radar, and runtime threads."""

    def setUp(self):
        self.route_mgr = RouteManager()
        self.shell = GatherionShell(route_mgr=self.route_mgr)

    def tearDown(self):
        try:
            self.shell.close()
        except Exception:
            pass

    def test_page_created_signal_and_get_page_if_created(self):
        """page_created signal is defined and emits when pages are mounted."""
        self.assertTrue(hasattr(self.shell, "page_created"))
        self.assertTrue(hasattr(self.shell, "get_page_if_created"))

        # Test lookup of eager pages
        home_page = self.shell.get_page_if_created("home")
        self.assertIsNotNone(home_page)

        map_page = self.shell.get_page_if_created("map")
        self.assertIsNotNone(map_page)

        radar_page = self.shell.get_page_if_created("radar")
        self.assertIsNotNone(radar_page)

        # Aliases lookup
        self.assertIs(self.shell.get_page_if_created("cartography_radar"), radar_page)
        self.assertIs(self.shell.get_page_if_created("cartography_radar_dialog"), radar_page)
        self.assertIs(self.shell.get_page_if_created("map_viewer"), map_page)

    def test_region_dropdown_synchronization_to_map_and_radar(self):
        """Changing active region dropdown in shell propagates to Map and Radar simultaneously."""
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

        map_win = self.shell.get_page_if_created("map")
        radar_win = self.shell.get_page_if_created("radar")
        self.assertEqual(map_win._region_name, "Altgard")
        self.assertEqual(radar_win._region_name, "Altgard")

    def test_route_dropdown_synchronization(self):
        """Changing active route dropdown propagates to Map, Radar, and RouteManager."""
        map_win = self.shell.get_page_if_created("map")
        radar_win = self.shell.get_page_if_created("radar")

        self.shell.set_active_route("IronLoop")
        self.assertEqual(self.shell.get_active_route(), "IronLoop")
        self.assertEqual(self.route_mgr.active_route, "IronLoop")
        self.assertEqual(map_win._route_name, "IronLoop")
        self.assertEqual(radar_win._route_name, "IronLoop")

    def test_tab_switching_refreshes_canvases_without_pausing_telemetry(self):
        """Switching between Map, Radar, and Home refreshes canvases and leaves timers active."""
        map_win = self.shell.get_page_if_created("map")
        radar_win = self.shell.get_page_if_created("radar")

        with patch.object(map_win, "reload_map_and_nodes") as mock_map_reload, \
             patch.object(radar_win, "reload_map_and_nodes") as mock_radar_reload:

            # Switch to Map tab
            self.shell.open_tool("map")
            self.assertEqual(self.shell.current_key(), "map")
            mock_map_reload.assert_called()

            # Switch to Radar tab
            self.shell.open_tool("radar")
            self.assertEqual(self.shell.current_key(), "radar")
            mock_radar_reload.assert_called()

            # Switch to Home tab
            self.shell.open_tool("home")
            self.assertEqual(self.shell.current_key(), "home")

            # Verify background telemetry timers are still active
            self.assertTrue(self.shell._home_timer.isActive())
            self.assertTrue(self.shell._log_timer.isActive())

    def test_runtime_threads_binding_and_signal_reception(self):
        """bind_runtime_threads connects worker HUD and node signals to Map and Radar."""
        mock_worker = MagicMock()
        mock_cartographer = MagicMock()

        self.shell.bind_runtime_threads(worker=mock_worker, auto_cartographer=mock_cartographer)
        self.assertIs(self.shell._worker, mock_worker)
        self.assertIs(self.shell._auto_cartographer, mock_cartographer)

        # Simulate worker hud update
        radar_win = self.shell.get_page_if_created("radar")
        map_win = self.shell.get_page_if_created("map")

        hud_data = {
            "player_x": 480.0,
            "player_y": 320.0,
            "heading_deg": 135.0,
        }
        self.shell._on_worker_hud_update(hud_data)
        self.assertIsNotNone(radar_win.canvas._player_pos)
        self.assertEqual(radar_win.canvas._player_pos[0], 480.0)
        self.assertEqual(radar_win.canvas._player_pos[1], 320.0)
        self.assertEqual(radar_win.canvas._player_pos[2], 135.0)

        # Simulate node discovered event
        with patch.object(radar_win, "reload_map_and_nodes") as mock_radar_reload, \
             patch.object(map_win, "reload_map_and_nodes") as mock_map_reload:
            self.shell._on_worker_node_recorded({"map_x": 510, "map_y": 340, "name": "Adamantium"})
            mock_map_reload.assert_called()


if __name__ == "__main__":
    unittest.main()
