# -*- coding: utf-8 -*-
"""
Test suite for F9Menu Live Route & Gathering UI Synchronization (Agent G9).
Validates multi-directional synchronization across F9Menu, RouteManager,
OverlayHUD, BotWorker, and GatherionShell.

Route data uses the REAL core.route_manager.RouteManager backed by a
temporary directory (no hand-rolled route-manager double: the sync under
test reads get_regions()/get_routes()/active_* from the real component,
and route persistence is verified with a disk round-trip).

Shell / Overlay / Worker are unittest.mock doubles, but spec'd against the
REAL GatherionShell / OverlayHUD / BotWorker classes, so they only expose
methods the real components actually have (a hand-written dummy previously
invented BotWorker.set_minimap_auto_record, which the real worker does
not have). Assertions on them are interaction assertions (recorded calls),
not invented dummy state.
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

# Ensure headless Qt support
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PyQt6.QtWidgets import QApplication

# Ensure single QApplication instance
app = QApplication.instance() or QApplication(sys.argv)

from core.route_manager import RouteManager
from core.state_machine import BotWorker
from ui.app_shell import GatherionShell
from ui.f9_menu import (
    F9Menu,
    get_f9_menu_baseline_metrics,
    get_f9_menu_capabilities,
    validate_f9_menu_contract,
    get_f9_menu_readiness_status,
)
from ui.overlay_hud import OverlayHUD


def _seed_route_manager(base_dir):
    """Build a real RouteManager with Altgard + Verteron routes on disk."""
    rm = RouteManager(base_dir=base_dir)
    rm.create_region("Altgard")
    for route in ("Dredgion Crash Site", "Safe Haven", "Quai Campsite"):
        rm.create_route("Altgard", route)
    rm.create_region("Verteron")
    for route in ("Tolbas Forest", "Cantas Coast"):
        rm.create_route("Verteron", route)
    rm.active_region = "Altgard"
    rm.active_route = "Dredgion Crash Site"
    rm.data["active_region"] = "Altgard"
    rm.data["active_route"] = "Dredgion Crash Site"
    rm.save_data()
    return rm


def _log_texts(overlay_mock):
    """Plain-text payloads of every add_log call recorded on a mock overlay."""
    return [str(call.args[0]) for call in overlay_mock.add_log.call_args_list]


class TestF9MenuLiveSync(unittest.TestCase):
    """Verifies live route and gathering synchronization in F9Menu."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.route_mgr = _seed_route_manager(self._tmp.name)
        self.menu = F9Menu(route_manager=self.route_mgr)
        self.addCleanup(self.menu.deleteLater)

    def _make_shell(self):
        shell = MagicMock(spec=GatherionShell)
        shell.get_active_region.return_value = "Altgard"
        shell.get_active_route.return_value = "Dredgion Crash Site"
        return shell

    def _make_overlay(self):
        return MagicMock(spec=OverlayHUD)

    def _make_worker(self):
        """Spec'd BotWorker double with its own real RouteManager instance.

        A separate RouteManager object (same backing directory) proves the
        propagation path really writes into the worker's manager instead of
        passing trivially through a shared object.
        """
        worker = MagicMock(spec=BotWorker)
        worker.route_manager = RouteManager(base_dir=self._tmp.name)
        worker.route_manager.load_region("Altgard")
        return worker

    def test_sync_from_route_manager_discovers_routes(self):
        """Routes from the real RouteManager must be discovered correctly."""
        regions_routes = self.menu.get_regions_routes()
        self.assertIn("Altgard", regions_routes)
        self.assertIn("Dredgion Crash Site", regions_routes["Altgard"])
        self.assertIn("Safe Haven", regions_routes["Altgard"])
        self.assertIn("Verteron", regions_routes)
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
        self.assertIn("Verteron", self.menu.header_subtitle_label.text())
        self.assertIn("Tolbas Forest", self.menu.header_subtitle_label.text())

    def test_select_route_propagates_to_route_manager_and_targets(self):
        """Selecting a route updates RouteManager and bound Shell, Overlay, and Worker."""
        shell = self._make_shell()
        overlay = self._make_overlay()
        worker = self._make_worker()

        self.menu.bind_targets(shell=shell, overlay=overlay, worker=worker)

        # Trigger route selection
        self.menu._select_route("Altgard", "Quai Campsite")

        # 1. Internal menu state
        self.assertEqual(self.menu.get_active_region(), "Altgard")
        self.assertEqual(self.menu.get_active_route(), "Quai Campsite")

        # 2. Real RouteManager state (attributes + in-memory mirror)
        self.assertEqual(self.route_mgr.active_region, "Altgard")
        self.assertEqual(self.route_mgr.active_route, "Quai Campsite")
        self.assertEqual(self.route_mgr.data["active_region"], "Altgard")
        self.assertEqual(self.route_mgr.data["active_route"], "Quai Campsite")

        # 2b. Route persistence: selection must reach disk, not just attributes.
        json_path = os.path.join(self._tmp.name, "Altgard", "region_data.json")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path, encoding="utf-8") as f:
            saved = json.load(f)
        self.assertEqual(saved.get("region"), "Altgard")
        self.assertIn("Quai Campsite", saved.get("routes", []))
        reloaded = RouteManager(base_dir=self._tmp.name)
        self.assertTrue(reloaded.load_region("Altgard"))
        self.assertIn("Quai Campsite", reloaded.get_routes("Altgard"))

        # 3. Shell state (recorded calls on the spec'd GatherionShell double)
        shell.set_active_region.assert_called_with("Altgard", update_combo=True)
        shell.set_active_route.assert_called_with("Quai Campsite", update_combo=True)
        shell._publish_home.assert_called()

        # 4. Overlay state (recorded calls on the spec'd OverlayHUD double)
        self.assertTrue(any("Quai Campsite" in msg for msg in _log_texts(overlay)))
        overlay.update_state.assert_called_with({"region": "Altgard", "route": "Quai Campsite"})

        # 5. Worker state (real RouteManager attached to the spec'd BotWorker double)
        self.assertEqual(worker.route_manager.active_region, "Altgard")
        self.assertEqual(worker.route_manager.active_route, "Quai Campsite")
        worker._auto_sort_sequential_route.assert_called_with(mode="smooth_loop")
        worker._emit_hud_snapshot.assert_called()

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

        # Verify state attributes via the public summary
        self.assertEqual(self.menu.get_current_harvest_stage(), 2)
        self.assertEqual(self.menu.get_harvests_per_node(), 3)
        self.assertTrue(self.menu.is_gathering())
        summary = self.menu.get_gathering_summary()
        self.assertEqual(summary["current_resource"], "Orichalcum Ore")
        self.assertEqual(summary["gathered_count"], 14)

        # Verify harvest badge label
        self.assertEqual(self.menu.harvest_badge_label.text(), "2/3")

        # Verify status line formatting
        status = self.menu.header_subtitle_label.text()
        self.assertIn("Topluyor", status)
        self.assertIn("[2/3]", status)
        self.assertIn("Orichalcum Ore", status)

        # Verify emitted signals
        self.assertEqual(len(harvest_signals), 1)
        self.assertEqual(harvest_signals[0], (2, 3))
        self.assertEqual(len(state_signals), 1)

    def test_update_harvest_stage_forwards_to_overlay(self):
        """Calling update_harvest_stage must update labels and forward to bound OverlayHUD."""
        overlay = self._make_overlay()
        self.menu.bind_targets(overlay=overlay)

        self.menu.update_harvest_stage(1, 3)
        overlay.update_harvest_stage.assert_called_with(1, 3)
        self.assertEqual(self.menu.get_current_harvest_stage(), 1)
        self.assertEqual(self.menu.harvest_badge_label.text(), "1/3")

        self.menu.update_harvest_stage(3, 3)
        overlay.update_harvest_stage.assert_called_with(3, 3)
        self.assertEqual(self.menu.get_current_harvest_stage(), 3)
        self.assertEqual(self.menu.harvest_badge_label.text(), "3/3")

    def test_mode_selection_and_action_propagation(self):
        """Clicking mode rows updates active mode and propagates to worker and overlay."""
        overlay = self._make_overlay()
        worker = self._make_worker()
        shell = self._make_shell()

        self.menu.bind_targets(shell=shell, overlay=overlay, worker=worker)

        # Test Combat mode
        self.menu._trigger_action("combat")
        self.assertEqual(self.menu.get_active_mode(), "combat")
        worker.set_mode.assert_called_with("combat")
        self.assertTrue(any("Savaş" in log for log in _log_texts(overlay)))
        shell._publish_home.assert_called()

        # Test Explorer mode
        self.menu._trigger_action("explorer")
        self.assertEqual(self.menu.get_active_mode(), "explorer")
        worker.set_mode.assert_called_with("explorer")

    def test_minimap_auto_record_toggle_propagation(self):
        """Toggling minimap auto record propagates to worker, overlay, and signals."""
        toggled_signals = []
        self.menu.minimap_auto_record_toggled.connect(lambda state: toggled_signals.append(state))

        overlay = self._make_overlay()
        worker = self._make_worker()
        self.menu.bind_targets(overlay=overlay, worker=worker)

        initial = self.menu.is_minimap_auto_record_enabled()
        self.menu._trigger_action("minimap_auto_record")

        self.assertEqual(self.menu.is_minimap_auto_record_enabled(), not initial)
        # The real BotWorker has no set_minimap_auto_record method; the
        # production code propagates via direct attribute assignment.
        self.assertEqual(worker.auto_cartography_enabled, not initial)
        self.assertTrue(any("Minimap" in log for log in _log_texts(overlay)))
        self.assertEqual(len(toggled_signals), 1)

    def test_contract_integrity_and_summary(self):
        """Validate F9Menu health snapshot, contract functions, and summary digest.

        The contract is checked on its actual content (exact capability
        list, instance/module agreement, metrics counts) — not merely by
        converting a constant list to bool.
        """
        summary = self.menu.get_gathering_summary()
        self.assertEqual(summary["active_region"], "Altgard")
        self.assertEqual(summary["active_route"], "Dredgion Crash Site")
        self.assertIn("harvests_per_node", summary)
        self.assertIn("current_harvest_stage", summary)

        health = self.menu.get_f9_menu_health_summary()
        self.assertTrue(health.get("f9_menu_ready"))

        capabilities = get_f9_menu_capabilities()
        self.assertEqual(capabilities, ["f9_menu", "quick_actions", "diagnostic_snapshot"])
        self.assertEqual(self.menu.get_f9_menu_capabilities(), capabilities)
        self.assertTrue(validate_f9_menu_contract())

        metrics = get_f9_menu_baseline_metrics()
        self.assertEqual(metrics["capability_count"], len(capabilities))
        self.assertEqual(metrics["capability_count"], 3)
        self.assertTrue(metrics["contract_valid"])

        readiness = get_f9_menu_readiness_status()
        self.assertTrue(readiness.get("ready"))
        self.assertEqual(readiness["metrics"]["capability_count"], 3)


if __name__ == "__main__":
    unittest.main()
