# -*- coding: utf-8 -*-
"""
Test suite for F9Menu Live Route & Gathering UI Synchronization (Agent G9).
Validates multi-directional synchronization across F9Menu, RouteManager,
OverlayHUD, BotWorker, and GatherionShell.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

# Ensure headless Qt support
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication

# Ensure single QApplication instance
app = QApplication.instance() or QApplication(sys.argv)

from ui.f9_menu import (
    F9Menu,
    get_f9_menu_capabilities,
    validate_f9_menu_contract,
    get_f9_menu_readiness_status,
)


class DummyRouteManager:
    """Mock RouteManager simulating core/route_manager.py functionality."""

    def __init__(self):
        self.active_region = "Altgard"
        self.active_route = "Dredgion Crash Site"
        self.data = {
            "active_region": "Altgard",
            "active_route": "Dredgion Crash Site",
            "regions": {
                "Altgard": {
                    "routes": ["All", "Dredgion Crash Site", "Safe Haven", "Quai Campsite"],
                    "nodes": [],
                },
                "Verteron": {
                    "routes": ["All", "Tolbas Forest", "Cantas Coast"],
                    "nodes": [],
                },
            },
        }
        self.saved = False

    def get_regions(self):
        return list(self.data["regions"].keys())

    def get_routes(self, region):
        reg = self.data["regions"].get(region, {})
        return list(reg.get("routes", ["All"]))

    def load_region(self, region):
        self.active_region = region
        return True

    def save_data(self):
        self.saved = True
        return True


class DummyShell:
    """Mock GatherionShell to verify live UI synchronization."""

    def __init__(self):
        self.region = ""
        self.route = ""
        self.published = False

    def get_active_region(self):
        return self.region or "Altgard"

    def get_active_route(self):
        return self.route or "All"

    def set_active_region(self, region, update_combo=True):
        self.region = region

    def set_active_route(self, route, update_combo=True):
        self.route = route

    def _publish_home(self):
        self.published = True


class DummyOverlay:
    """Mock OverlayHUD to verify live status, logs, and harvest stage synchronization."""

    def __init__(self):
        self.logs = []
        self.state_data = {}
        self.harvest_stage = (0, 3)

    def add_log(self, msg):
        self.logs.append(msg)

    def update_state(self, data):
        self.state_data.update(data)

    def update_harvest_stage(self, stage, max_s):
        self.harvest_stage = (stage, max_s)


class DummyWorker:
    """Mock BotWorker to verify mode, auto-cartography, and snapshot synchronization."""

    def __init__(self):
        self.mode = "route"
        self.auto_cartography_enabled = False
        self.snapshots_emitted = 0
        self.route_manager = DummyRouteManager()
        self.hud_update = MagicMock()

    def set_mode(self, mode):
        self.mode = mode

    def set_minimap_auto_record(self, enabled):
        self.auto_cartography_enabled = bool(enabled)

    def _auto_sort_sequential_route(self, mode="smooth_loop"):
        pass

    def _emit_hud_snapshot(self):
        self.snapshots_emitted += 1


class TestF9MenuLiveSync(unittest.TestCase):
    """Verifies live route and gathering synchronization in F9Menu."""

    def setUp(self):
        self.route_mgr = DummyRouteManager()
        self.menu = F9Menu(route_manager=self.route_mgr)

    def test_sync_from_route_manager_discovers_routes(self):
        """Routes from RouteManager must be discovered correctly into _regions_routes."""
        self.assertIn("Altgard", self.menu._regions_routes)
        self.assertIn("Dredgion Crash Site", self.menu._regions_routes["Altgard"])
        self.assertIn("Safe Haven", self.menu._regions_routes["Altgard"])
        self.assertIn("Verteron", self.menu._regions_routes)
        self.assertEqual(self.menu.get_active_region(), "Altgard")
        self.assertEqual(self.menu.get_active_route(), "Dredgion Crash Site")

    def test_route_changed_signal_and_external_slot(self):
        """Calling on_route_changed should update active region/route and emit route_changed."""
        received = []
        self.menu.route_changed.connect(lambda reg, rte: received.append((reg, rte)))

        self.menu.on_route_changed("Verteron", "Tolbas Forest")

        self.assertEqual(self.menu.get_active_region(), "Verteron")
        self.assertEqual(self.menu.get_active_route(), "Tolbas Forest")
        self.assertEqual(len(received), 1)
        self.assertEqual(received[0], ("Verteron", "Tolbas Forest"))
        self.assertIn("Verteron", self.menu._format_status_line())
        self.assertIn("Tolbas Forest", self.menu._format_status_line())

    def test_select_route_propagates_to_route_manager_and_targets(self):
        """Selecting a route updates RouteManager and bound Shell, Overlay, and Worker."""
        shell = DummyShell()
        overlay = DummyOverlay()
        worker = DummyWorker()

        self.menu.bind_targets(shell=shell, overlay=overlay, worker=worker)

        # Trigger route selection
        self.menu._select_route("Altgard", "Quai Campsite")

        # 1. Internal menu state
        self.assertEqual(self.menu.get_active_region(), "Altgard")
        self.assertEqual(self.menu.get_active_route(), "Quai Campsite")

        # 2. RouteManager state
        self.assertEqual(self.route_mgr.active_region, "Altgard")
        self.assertEqual(self.route_mgr.active_route, "Quai Campsite")

        # 3. Shell state
        self.assertEqual(shell.region, "Altgard")
        self.assertEqual(shell.route, "Quai Campsite")
        self.assertTrue(shell.published)

        # 4. Overlay state
        self.assertTrue(any("Quai Campsite" in msg for msg in overlay.logs))
        self.assertEqual(overlay.state_data.get("region"), "Altgard")
        self.assertEqual(overlay.state_data.get("route"), "Quai Campsite")

        # 5. Worker state
        self.assertEqual(worker.route_manager.active_region, "Altgard")
        self.assertEqual(worker.route_manager.active_route, "Quai Campsite")
        self.assertGreaterEqual(worker.snapshots_emitted, 1)

    def test_gathering_state_synchronization(self):
        """Setting gathering telemetry updates stage, badge, status line, and emits signals."""
        harvest_signals = []
        state_signals = []

        self.menu.harvest_stage_changed.connect(lambda s, m: harvest_signals.append((s, m)))
        self.menu.gathering_state_changed.connect(lambda d: state_signals.append(d))

        payload = {
            "harvest_stage": 2,
            "max_harvest_stages": 3,
            "current_state": "GATHERING",
            "resource_type": "Orichalcum Ore",
            "gathered_count": 14,
            "active_node_id": 105,
            "mode": "route",
            "paused": False,
        }

        self.menu.set_gathering_state(payload)

        # Verify state attributes
        self.assertEqual(self.menu.get_current_harvest_stage(), 2)
        self.assertEqual(self.menu.get_harvests_per_node(), 3)
        self.assertTrue(self.menu.is_gathering())
        self.assertEqual(self.menu._current_resource, "Orichalcum Ore")
        self.assertEqual(self.menu._gathered_count, 14)

        # Verify harvest badge label
        self.assertEqual(self.menu.harvest_badge_label.text(), "2/3")

        # Verify status line formatting
        status = self.menu._format_status_line()
        self.assertIn("Topluyor", status)
        self.assertIn("[2/3]", status)
        self.assertIn("Orichalcum Ore", status)

        # Verify emitted signals
        self.assertEqual(len(harvest_signals), 1)
        self.assertEqual(harvest_signals[0], (2, 3))
        self.assertEqual(len(state_signals), 1)

    def test_update_harvest_stage_forwards_to_overlay(self):
        """Calling update_harvest_stage must update labels and forward to bound OverlayHUD."""
        overlay = DummyOverlay()
        self.menu.bind_targets(overlay=overlay)

        self.menu.update_harvest_stage(1, 3)
        self.assertEqual(overlay.harvest_stage, (1, 3))
        self.assertEqual(self.menu.get_current_harvest_stage(), 1)
        self.assertEqual(self.menu.harvest_badge_label.text(), "1/3")

        self.menu.update_harvest_stage(3, 3)
        self.assertEqual(overlay.harvest_stage, (3, 3))
        self.assertEqual(self.menu.get_current_harvest_stage(), 3)
        self.assertEqual(self.menu.harvest_badge_label.text(), "3/3")

    def test_mode_selection_and_action_propagation(self):
        """Clicking mode rows updates active mode and propagates to worker and overlay."""
        overlay = DummyOverlay()
        worker = DummyWorker()
        shell = DummyShell()

        self.menu.bind_targets(shell=shell, overlay=overlay, worker=worker)

        # Test Combat mode
        self.menu._trigger_action("combat")
        self.assertEqual(self.menu.get_active_mode(), "combat")
        self.assertEqual(worker.mode, "combat")
        self.assertTrue(any("Savaş" in log for log in overlay.logs))
        self.assertTrue(shell.published)

        # Test Explorer mode
        self.menu._trigger_action("explorer")
        self.assertEqual(self.menu.get_active_mode(), "explorer")
        self.assertEqual(worker.mode, "explorer")

    def test_minimap_auto_record_toggle_propagation(self):
        """Toggling minimap auto record propagates to worker, overlay, and signals."""
        toggled_signals = []
        self.menu.minimap_auto_record_toggled.connect(lambda state: toggled_signals.append(state))

        overlay = DummyOverlay()
        worker = DummyWorker()
        self.menu.bind_targets(overlay=overlay, worker=worker)

        initial = self.menu.is_minimap_auto_record_enabled()
        self.menu._trigger_action("minimap_auto_record")

        self.assertEqual(self.menu.is_minimap_auto_record_enabled(), not initial)
        self.assertEqual(worker.auto_cartography_enabled, not initial)
        self.assertTrue(any("Minimap" in log for log in overlay.logs))
        self.assertEqual(len(toggled_signals), 1)

    def test_contract_integrity_and_summary(self):
        """Validate F9Menu health snapshot, contract functions, and summary digest."""
        summary = self.menu.get_gathering_summary()
        self.assertEqual(summary["active_region"], "Altgard")
        self.assertEqual(summary["active_route"], "Dredgion Crash Site")
        self.assertIn("harvests_per_node", summary)
        self.assertIn("current_harvest_stage", summary)

        health = self.menu.get_f9_menu_health_summary()
        self.assertTrue(health.get("f9_menu_ready"))

        self.assertTrue(validate_f9_menu_contract())
        readiness = get_f9_menu_readiness_status()
        self.assertTrue(readiness.get("ready"))


if __name__ == "__main__":
    unittest.main()
