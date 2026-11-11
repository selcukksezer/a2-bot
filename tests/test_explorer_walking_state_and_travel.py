# -*- coding: utf-8 -*-
"""
tests/test_explorer_walking_state_and_travel.py
================================================
Comprehensive test suite verifying Agent G6 walking state machine,
false 3s obstacle alert elimination, 5m BEKLEMEDE lock elimination,
real movement coordinate tracking, and continuous smooth route waypoint progression.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from core.explorer_mode import ExplorerMode, ExplorerModeController
from core.route_manager import NodeData, RouteManager
from dataset_factory.exemplar_memory import ExemplarMemory


def _add_near_wall_stripes(
    frame: np.ndarray,
    y0: float = 0.26,
    y1: float = 0.76,
    x0: float = 0.37,
    x1: float = 0.63,
) -> np.ndarray:
    """Paint a near vertical obstacle (tree trunk / wall) into ``frame`` in place.

    Moderate-contrast vertical stripes: vertical structure + dark mass,
    which the real geometric depth engine
    (``MonocularDepthObstacleAvoidance``) reads as a close obstacle
    (clearance_center ~0.26 over the full corridor, below the 0.35 obstacle
    threshold but above the 0.18 critical-blockage floor, so first-frame
    debounce can still suppress it). Returns the frame for chaining.

    F-1824 kalibrasyonu: eski 60/130 yüksek kontrastlı çizgiler, D11'de
    engel tepkisi baskın hâle gelince koridoru kritik tıkalı (~0.0)
    okutmaya başladı; 106/122 orta kontrastı aynı senaryoyu (orta
    şiddette duvar -> ilk kare debounce) bugünkü motorla verir.
    """
    h, w = frame.shape[:2]
    ya, yb, xa, xb = int(h * y0), int(h * y1), int(w * x0), int(w * x1)
    band = np.zeros((yb - ya, xb - xa, 3), dtype=np.uint8)
    xs = np.arange(xb - xa)
    band[:, (xs // 16) % 2 == 0] = 106
    band[:, (xs // 16) % 2 == 1] = 122
    frame[ya:yb, xa:xb] = band
    return frame


class TestExplorerWalkingStateAndTravel(unittest.TestCase):
    """Test suite for Agent G6: Walking State Machine & Pause/Lock Elimination."""

    def setUp(self) -> None:
        self.rm = RouteManager()
        self.region = "test_g6_region"
        self.route = "test_g6_route"
        self.rm.create_region(self.region)
        self.rm.create_route(self.region, self.route)
        self.rm.active_region = self.region
        self.rm.active_route = self.route

        self.explorer = ExplorerModeController(
            route_manager=self.rm,
            region=self.region,
            route=self.route,
            test_mode=True,
        )

    def test_01_walking_state_machine_accuracy_and_anti_beklemede(self) -> None:
        """
        Verify that Walking State Machine guarantees YÜRÜYOR / ROTA TAKİBİ / KEŞİFTE
        state accuracy during player-walk and bot-walk, and eliminates false BEKLEMEDE locks.
        """
        exp = self.explorer

        # 1. Active walking motion -> YÜRÜYOR
        state = exp.update_walking_state(is_moving=True)
        self.assertEqual(state, "YÜRÜYOR")
        self.assertEqual(exp.get_walking_state(), "YÜRÜYOR")

        # 2. Exploration mode with sector waypoints -> KEŞİFTE
        exp.generate_sector_sweep_waypoints((100, 100, 400, 400), lane_spacing_px=100)
        state = exp.update_walking_state(is_moving=True, in_exploration=True)
        self.assertEqual(state, "KEŞİFTE")
        self.assertEqual(exp.get_walking_state(), "KEŞİFTE")

        # 3. Route mode with ordered nodes -> ROTA TAKİBİ
        n1 = NodeData(id=1, map_x=200, map_y=200, region=self.region, resource_type="Aria")
        exp._ordered_nodes = [n1]
        state = exp.update_walking_state(is_moving=True, in_route=True)
        self.assertEqual(state, "ROTA TAKİBİ")
        self.assertEqual(exp.get_walking_state(), "ROTA TAKİBİ")

        # 4. Anti-BEKLEMEDE lock: setting BEKLEMEDE while movement or route is active
        # MUST NOT downgrade to BEKLEMEDE
        state = exp.set_walking_state("BEKLEMEDE", reason="test attempt to lock")
        self.assertNotEqual(state, "BEKLEMEDE")
        self.assertEqual(state, "ROTA TAKİBİ")

        # 5. True idle stop: when not moving and no active tasks, can enter BEKLEMEDE
        exp._ordered_nodes = []
        exp._sector_waypoints = []
        exp._movement_active_override = False
        state = exp.update_walking_state(is_moving=False)
        self.assertEqual(state, "BEKLEMEDE")

        # 6. Stuck detection & force resume
        exp._movement_active_override = True
        self.assertTrue(exp.is_stuck_in_beklemede())
        resumed = exp.force_resume_from_beklemede()
        self.assertIn(resumed, ("YÜRÜYOR", "ROTA TAKİBİ", "KEŞİFTE"))
        self.assertFalse(exp.is_stuck_in_beklemede())

    def test_02_real_movement_coords_and_continuous_odometry(self) -> None:
        """
        Verify that ExplorerModeController passes real movement coordinates and
        advances dead reckoning continuously when walking, preventing false 3s stuck alerts.
        """
        exp = self.explorer
        exp.auto_cartographer.set_position(500.0, 500.0)

        # 1. Initial coordinates are valid and non-zero
        coords1 = exp.get_player_movement_coords()
        self.assertIsInstance(coords1, tuple)
        self.assertEqual(len(coords1), 2)
        self.assertGreater(coords1[0], 0.0)
        self.assertGreater(coords1[1], 0.0)

        # 2. Aliases return identical coordinates
        self.assertEqual(exp.get_movement_coords(), coords1)
        self.assertEqual(exp.get_current_position(), coords1)
        self.assertEqual(exp.auto_cartographer.get_current_position(), coords1)

        # 3. Real movement over real time -> coordinates advance smoothly.
        # The odometry clock is read from the live clock: sync it, let a genuine
        # 250 ms elapse, then read again. No forged internal timestamp.
        exp.set_walking_state("YÜRÜYOR")
        exp._movement_active_override = True
        coords_sync = exp.get_player_movement_coords()
        time.sleep(0.25)
        coords2 = exp.get_player_movement_coords()

        # Displacement must be non-zero (smooth travel progression)
        dx = abs(coords2[0] - coords_sync[0])
        dy = abs(coords2[1] - coords_sync[1])
        displacement = math.hypot(dx, dy)
        self.assertGreater(displacement, 0.5, "Moving player coordinates must advance over time")
        # Walking speed is 85 px/s, so ~21 px in 0.25 s; a step near the 0.5 s
        # dead-reckoning clamp (42.5 px) would mean the clock was forged.
        self.assertLess(displacement, 40.0, "Odometry step must match real elapsed time")

        # 4. Manual odometry delta update
        exp.update_movement_odometry(dx=15.0, dy=-10.0, dt=0.1)
        coords3 = exp.get_player_movement_coords()
        self.assertAlmostEqual(coords3[0], coords2[0] + 15.0, delta=2.0)

    def test_03_false_3s_obstacle_alert_elimination(self) -> None:
        """
        Verify elimination of false 3s obstacle alerts:
        - Transient single-frame depth anomalies are suppressed / debounced.
        - Ground path confirmation (class 15: path_ground) suppresses depth alerts.
        - Genuine monster (10: monster_3d) or obstacle (14: obstacle) in ExemplarMemory is alerted.
        """
        # test_03 kendi kendini kirletmesin: ExemplarMemory(json_path=None)
        # reponun gerçek dataset/learned_exemplars.json dosyasına bağlanır ve
        # add_positive_exemplar her eklemede save() ile dosyaya yazar. İlk
        # koşuda yazılan class-15 exemplar, ikinci koşuda near-duplicate
        # dalında (sim 1.0 >= 0.96) add_positive_exemplar'ı False döndürüp
        # testi kırar. Her bellek ayrı bir geçici JSON kullanır; testin
        # niyeti (false 3s obstacle alert eliminasyonu) aynen korunur.
        _tmp = tempfile.TemporaryDirectory()
        _mem_path = lambda name: str(Path(_tmp.name) / name)  # noqa: E731
        exp = self.explorer

        # 0. The REAL depth engine (no mock) must see the near wall in a
        # synthetic frame: obstacle detected, clearance below the 0.35
        # threshold but above the 0.18 critical-blockage floor.
        obstacle_frame = _add_near_wall_stripes(
            np.full((720, 1280, 3), 120, dtype=np.uint8)
        )
        raw = exp.depth_obstacle_avoidance.analyze_forward_corridor(obstacle_frame)
        self.assertTrue(raw["obstacle_detected"], "Real depth engine must see the near wall")
        self.assertNotEqual(raw["obstacle_type"], "none")
        self.assertLess(raw["clearance_center"], 0.35)
        self.assertGreater(raw["clearance_center"], 0.18)

        # 1. First transient depth blip -> suppressed by debounce
        res1 = exp.check_path_obstacle(
            obstacle_frame,
            exemplar_memory=ExemplarMemory(json_path=_mem_path("exemplar_blip.json")),
            suppress_transient_depth=True,
        )
        self.assertFalse(res1["obstacle_detected"], "Transient 1-frame depth blip must be suppressed")
        self.assertTrue(res1["false_alert_suppressed"])

        # 2. Walkable ground path present -> suppresses depth alert even on consecutive frames
        mem = ExemplarMemory(json_path=_mem_path("exemplar_path.json"))
        path_crop = np.full((80, 80, 3), 140, dtype=np.uint8)
        # K-3: düz tek renk kare crop dejenere sayılıp exemplar olarak
        # reddedilir; gerçek zemin gibi hafif 8 px dama dokusu ver
        # (140/152). Koridora yapıştırılan büyütülmüş kopya aynı desen
        # olduğu için eşleşme skoru korunur.
        yy, xx = np.mgrid[0:80, 0:80]
        path_crop[((yy // 8) + (xx // 8)) % 2 == 0] = 152
        self.assertTrue(
            mem.add_positive_exemplar(path_crop, class_id=15, class_name="path_ground"),
            "path_ground exemplar K-3 dejenere kapısına takıldı",
        )

        # Composite frame: the near-wall stripes stay ABOVE the exemplar
        # corridor (rows 26-52%) so the real depth engine still reports a raw
        # obstacle, while the corridor itself (rows 52-82%) shows walkable
        # path ground. The suppression below is then suppressing a REAL raw
        # alert, not a vacuous one.
        path_frame = np.full((720, 1280, 3), 120, dtype=np.uint8)
        _add_near_wall_stripes(path_frame, y0=0.26, y1=0.52)
        h, w = path_frame.shape[:2]
        path_frame[int(h * 0.52) : int(h * 0.82), int(w * 0.40) : int(w * 0.60)] = cv2.resize(
            path_crop, (int(w * 0.20), int(h * 0.30))
        )
        raw2 = exp.depth_obstacle_avoidance.analyze_forward_corridor(path_frame)
        self.assertTrue(
            raw2["obstacle_detected"],
            "Composite frame must still trigger the real depth engine",
        )

        res2 = exp.check_path_obstacle(path_frame, exemplar_memory=mem)
        self.assertFalse(res2["obstacle_detected"], "Walkable path ground must suppress false obstacle alerts")
        self.assertTrue(res2["walkable_path_detected"])
        self.assertTrue(res2["false_alert_suppressed"])

        # 3. Genuine monster obstacle (class 10: monster_3d) -> must trigger immediately
        monster_mem = ExemplarMemory(json_path=_mem_path("exemplar_monster.json"))
        monster_crop = np.zeros((80, 80, 3), dtype=np.uint8)
        monster_crop[:, :] = (20, 30, 210)  # Red hostile mob
        # K-3: düz tek renk kare crop dejenere sayılıp exemplar olarak
        # reddedilir; 8 px dama dokusu ver (koyu/açık kırmızı).
        # Koridordaki büyütülmüş kopya aynı desen, eşleşme korunur.
        yy, xx = np.mgrid[0:80, 0:80]
        monster_crop[((yy // 8) + (xx // 8)) % 2 == 0] = (40, 50, 180)
        self.assertTrue(
            monster_mem.add_positive_exemplar(monster_crop, class_id=10, class_name="monster_3d"),
            "monster_3d exemplar K-3 dejenere kapısına takıldı",
        )

        monster_frame = np.full((720, 1280, 3), 100, dtype=np.uint8)
        monster_frame[int(h * 0.52) : int(h * 0.82), int(w * 0.40) : int(w * 0.60)] = cv2.resize(
            monster_crop, (int(w * 0.20), int(h * 0.30))
        )

        res3 = exp.check_path_obstacle(monster_frame, exemplar_memory=monster_mem)
        self.assertTrue(res3["obstacle_detected"], "Real monster must be detected as an obstacle")
        self.assertTrue(res3["exemplar_obstacle"])
        self.assertNotEqual(res3["recommended_turn"], "none")

    def test_04_5m_resting_lock_elimination(self) -> None:
        """
        Verify elimination of 5m BEKLEMEDE locks:
        Falsely resting nodes are automatically cleared and restored to active.
        """
        exp = self.explorer

        # Add 3 nodes to route
        n1 = NodeData(id=1, map_x=100, map_y=100, region=self.region, resource_type="Aria")
        n2 = NodeData(id=2, map_x=200, map_y=200, region=self.region, resource_type="Mela")
        n3 = NodeData(id=3, map_x=300, map_y=300, region=self.region, resource_type="Gem")
        for n in (n1, n2, n3):
            n.region = self.region
            n.routes = [self.route]
            self.rm.add_node(n)
        exp._ordered_nodes = [n1, n2, n3]

        # Simulate nodes being falsely marked as "resting" (5-minute cooldown)
        n1.state = "resting"
        n2.state = "resting"
        self.rm.save_region(self.region)

        # Clear false locks
        unlocked = exp.clear_false_resting_locks()
        self.assertEqual(unlocked, 2)
        self.assertEqual(n1.state, "active")
        self.assertEqual(n2.state, "active")
        self.assertEqual(n1.current_harvests, 0)
        self.assertEqual(n2.current_harvests, 0)

    def test_05_continuous_smooth_route_waypoint_progression(self) -> None:
        """
        Verify continuous smooth travel across route waypoints without stuttering or stopping.
        """
        exp = self.explorer
        n1 = NodeData(id=1, map_x=100, map_y=100, region=self.region, resource_type="Aria")
        n2 = NodeData(id=2, map_x=200, map_y=200, region=self.region, resource_type="Mela")
        n3 = NodeData(id=3, map_x=300, map_y=300, region=self.region, resource_type="Gem")
        exp._ordered_nodes = [n1, n2, n3]
        exp._active_route_index = 0

        # Position bot far from n1 via the public cartographer API only.
        exp.auto_cartographer.set_position(50.0, 50.0)

        # Waypoint heading and steering
        steer = exp.compute_waypoint_heading_and_steer(current_heading_deg=0.0)
        self.assertTrue(steer["has_target"])
        self.assertFalse(steer["reached"])
        self.assertEqual(steer["target_coords"], (100.0, 100.0))
        self.assertGreater(steer["distance_px"], 0.0)

        # Bot arrives at waypoint 1 (n1) via the public cartographer API only.
        exp.auto_cartographer.set_position(105.0, 105.0)

        advanced_node = exp.advance_route_waypoint(arrival_radius_px=22.0)
        # Must advance to n2 (index 1)
        self.assertEqual(exp._active_route_index, 1)
        self.assertEqual(advanced_node.id, 2)
        self.assertEqual(exp.get_walking_state(), "ROTA TAKİBİ")

        # Bot arrives at waypoint 2 (n2) via the public cartographer API only.
        exp.auto_cartographer.set_position(202.0, 198.0)
        advanced_node2 = exp.advance_route_waypoint(arrival_radius_px=22.0)
        self.assertEqual(exp._active_route_index, 2)
        self.assertEqual(advanced_node2.id, 3)

        # Bot arrives at waypoint 3 (n3) -> closed-loop loopback to n1 (index 0)
        exp.auto_cartographer.set_position(301.0, 301.0)
        advanced_node3 = exp.advance_route_waypoint(arrival_radius_px=22.0, closed_loop=True)
        self.assertEqual(exp._active_route_index, 0)
        self.assertEqual(advanced_node3.id, 1)
        self.assertEqual(exp.get_walking_state(), "ROTA TAKİBİ")

    def test_06_explorer_mode_facade_delegation(self) -> None:
        """Verify that ExplorerMode facade correctly delegates to internal controller."""
        facade = ExplorerMode(
            route_manager=self.rm,
            region=self.region,
            route=self.route,
            test_mode=True,
        )

        state = facade.set_walking_state("YÜRÜYOR")
        self.assertEqual(state, "YÜRÜYOR")
        self.assertEqual(facade.get_walking_state(), "YÜRÜYOR")

        coords = facade.get_player_movement_coords()
        self.assertIsInstance(coords, tuple)
        self.assertEqual(len(coords), 2)

        snap = facade.get_explorer_mode_diagnostic_snapshot()
        self.assertEqual(snap["status"], "ok")
        self.assertTrue(snap["ready"])
        self.assertEqual(snap["walking_state"], "YÜRÜYOR")
        self.assertTrue(snap["anti_beklemede_lock"])
        # Snapshot must mirror live controller state, not constants.
        self.assertEqual(snap["module"], "explorer_mode")
        self.assertEqual(snap["walking_state"], facade.get_walking_state())
        # Back-to-back reads may drift by one dead-reckoning step while
        # walking, so compare coordinates within a small tolerance.
        live_coords = facade.get_player_movement_coords()
        self.assertAlmostEqual(snap["movement_coords"][0], live_coords[0], delta=5.0)
        self.assertAlmostEqual(snap["movement_coords"][1], live_coords[1], delta=5.0)

        # ...and it must track a genuine state change, not a frozen value.
        facade.update_walking_state(is_moving=False)
        snap2 = facade.get_explorer_mode_diagnostic_snapshot()
        self.assertEqual(snap2["walking_state"], "BEKLEMEDE")
        self.assertNotEqual(snap2["walking_state"], snap["walking_state"])
        live_coords2 = facade.get_player_movement_coords()
        self.assertAlmostEqual(snap2["movement_coords"][0], live_coords2[0], delta=5.0)
        self.assertAlmostEqual(snap2["movement_coords"][1], live_coords2[1], delta=5.0)


if __name__ == "__main__":
    unittest.main()
