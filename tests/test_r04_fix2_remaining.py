# -*- coding: utf-8 -*-
"""
tests/test_r04_fix2_remaining.py
================================
R04 (fix2) regression tests for the remaining-225 findings:

- S-0601: advance_route_waypoint must NOT reset partial harvest progress
  (current_harvests, e.g. 2/3) when unlocking a falsely "resting" node.
- S-0602: GlideWaypointController ready/health/telemetry methods must
  reflect a real probe (thresholds + synthetic glide decision), not a
  hardcoded True.
- S-0603: ExplorerMode wrapper ready/version/diagnostic methods must
  delegate to the real module-level contract probe, not hardcoded True.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.depth_glide_navigator import (
    GlideWaypointController,
    validate_depth_glide_navigator_contract,
)
from core.explorer_mode import (
    ExplorerMode,
    ExplorerModeController,
    validate_explorer_mode_contract,
)
from core.route_manager import NodeData, RouteManager


def _make_explorer():
    rm = RouteManager()
    region = "test_r04_region"
    route = "test_r04_route"
    rm.create_region(region)
    rm.create_route(region, route)
    rm.active_region = region
    rm.active_route = route
    exp = ExplorerModeController(
        route_manager=rm, region=region, route=route, test_mode=True
    )
    return rm, region, route, exp


class TestS0601AdvancePreservesPartialHarvests(unittest.TestCase):
    def test_advance_to_resting_node_keeps_partial_harvests(self):
        _, region, route, exp = _make_explorer()
        n1 = NodeData(id=1, map_x=100, map_y=100, region=region, resource_type="Aria")
        n2 = NodeData(id=2, map_x=200, map_y=200, region=region, resource_type="Mela")
        n3 = NodeData(id=3, map_x=300, map_y=300, region=region, resource_type="Gem")
        # n2 was falsely locked "resting" mid-harvest (2/3 done).
        n2.state = "resting"
        n2.current_harvests = 2
        exp._ordered_nodes = [n1, n2, n3]
        exp._active_route_index = 0
        exp.auto_cartographer.set_position(105.0, 105.0)

        advanced = exp.advance_route_waypoint(arrival_radius_px=22.0)

        self.assertEqual(advanced.id, 2)
        self.assertEqual(n2.state, "active")
        # S-0601: partial progress must survive the unlock (was reset to 0).
        self.assertEqual(n2.current_harvests, 2)

    def test_advance_to_legit_resting_node_still_works(self):
        # A legitimately depleted node (3/3 done) already has
        # current_harvests == 0, so unlocking it must stay a no-op on the counter.
        _, region, route, exp = _make_explorer()
        n1 = NodeData(id=1, map_x=100, map_y=100, region=region, resource_type="Aria")
        n2 = NodeData(id=2, map_x=200, map_y=200, region=region, resource_type="Mela")
        n2.state = "resting"
        n2.current_harvests = 0
        exp._ordered_nodes = [n1, n2]
        exp._active_route_index = 0
        exp.auto_cartographer.set_position(105.0, 105.0)

        advanced = exp.advance_route_waypoint(arrival_radius_px=22.0)

        self.assertEqual(advanced.id, 2)
        self.assertEqual(n2.state, "active")
        self.assertEqual(n2.current_harvests, 0)


class TestS0602GlideReadyProbe(unittest.TestCase):
    def test_default_controller_reports_ready(self):
        ctrl = GlideWaypointController()
        self.assertTrue(ctrl.get_depth_glide_navigator_ready_flag()["ready"])
        self.assertTrue(ctrl.get_depth_glide_navigator_version_info()["ready"])
        snap = ctrl.get_depth_glide_navigator_diagnostic_snapshot()
        self.assertTrue(snap["ready"])
        self.assertEqual(snap["status"], "ok")
        self.assertTrue(ctrl.get_depth_glide_health_summary()["depth_glide_ready"])
        self.assertTrue(ctrl.get_navigator_telemetry_summary()["navigator_ready"])
        self.assertTrue(validate_depth_glide_navigator_contract())

    def test_degenerate_thresholds_report_not_ready(self):
        # A controller whose thresholds cannot describe a real glide must
        # NOT report ready — this is what the hardcoded True used to hide.
        ctrl = GlideWaypointController(min_elevation_drop=-5.0)
        self.assertFalse(ctrl.get_depth_glide_navigator_ready_flag()["ready"])
        self.assertFalse(ctrl.get_depth_glide_navigator_version_info()["ready"])
        snap = ctrl.get_depth_glide_navigator_diagnostic_snapshot()
        self.assertFalse(snap["ready"])
        self.assertEqual(snap["status"], "not_ready")
        self.assertFalse(ctrl.get_depth_glide_health_summary()["depth_glide_ready"])
        self.assertFalse(ctrl.get_navigator_telemetry_summary()["navigator_ready"])

    def test_probe_has_positive_and_negative_controls(self):
        probe = GlideWaypointController()._probe_glide_readiness()
        self.assertTrue(probe["passed"])
        self.assertTrue(probe["thresholds_sane"])
        self.assertTrue(probe["synthetic_glide_decision"])
        self.assertTrue(probe["flat_pair_rejected"])
        self.assertTrue(probe["glide_estimate_sane"])


class TestS0603ExplorerReadyDelegatesToContract(unittest.TestCase):
    def _facade(self):
        rm = RouteManager()
        region = "test_r04_region"
        route = "test_r04_route"
        rm.create_region(region)
        rm.create_route(region, route)
        rm.active_region = region
        rm.active_route = route
        return ExplorerMode(
            route_manager=rm, region=region, route=route, test_mode=True
        )

    def test_ready_flag_matches_contract_probe(self):
        facade = self._facade()
        expected = validate_explorer_mode_contract()
        self.assertEqual(facade.get_explorer_mode_ready_flag()["ready"], expected)
        self.assertEqual(facade.get_explorer_mode_version_info()["ready"], expected)
        snap = facade.get_explorer_mode_diagnostic_snapshot()
        self.assertEqual(snap["ready"], expected)
        self.assertEqual(snap["status"], "ok" if expected else "not_ready")

    def test_ready_flag_follows_contract_when_contract_fails(self):
        # Proves delegation: if the real probe fails, the wrapper must
        # report not-ready instead of a hardcoded True.
        facade = self._facade()
        with mock.patch(
            "core.explorer_mode.validate_explorer_mode_contract", return_value=False
        ):
            self.assertFalse(facade.get_explorer_mode_ready_flag()["ready"])
            self.assertFalse(facade.get_explorer_mode_version_info()["ready"])
            snap = facade.get_explorer_mode_diagnostic_snapshot()
            self.assertFalse(snap["ready"])
            self.assertEqual(snap["status"], "not_ready")


if __name__ == "__main__":
    unittest.main()
