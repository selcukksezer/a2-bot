# -*- coding: utf-8 -*-
"""
Comprehensive 10-Agent Gather, Walk & Route Test Suite (Agent G10)
==================================================================

Covers all 6 core subsystems required by the Gather/Walk/Route architecture:
1. 3x Harvest Cycle (ThreeStageHarvestController, 1/3 -> 2/3 -> 3/3, 400ms re-prompt,
   anchoring, partial node early depletion, essence extraction failure retry).
2. F8 Waypoint Recording & Persistence (RouteManager, hybrid minimap + full map,
   80x80 patch extraction, duplicate node cleanup, save/load roundtrip).
3. 2-Opt TSP Route Optimization (RouteOptimizer in path_optimizer.py and
   SequentialRoutePlanner in sequential_router.py: distance reduction, crossing elimination).
4. Anti-Inversion & Heading Alignment (compute_bearing, compute_angle_diff,
   TurnDampingController: North=0°, East=90°, eliminating 180° Cartesian inversion,
   deadband ±5°..±8° oscillation prevention).
5. 3D Obstacle Avoidance & Corridor Steering (CorridorNavigator: 3 sub-lanes
   Left/Center/Right, obstacle/monster/path classification, steering bias and turn offset).
6. Unstick Maneuvers (PhaseCorrelateUnstuckManeuver & RadarStuckDetector: shift < 1.2px
   over 2.5s, Space jump -> A+W / D+W alternating strafe -> realign).
7. Full 10-Agent End-to-End Integration Scenario (Record -> Optimize -> Steer -> Avoid ->
   Unstick -> 3x Harvest -> Next Waypoint).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np
import pytest

# Core imports
import core.navigator as nav
import core.corridor_navigator as cor_nav
import core.navigation_intel as nav_intel
from core.corridor_navigator import (
    CentroidDriftCorrector,
    ContestedNodeDetector,
    CorridorNavigator,
    PhaseCorrelateUnstuckManeuver,
    ZAxisCliffFilter,
)
from core.navigation_intel import (
    AutoMountManager,
    MaplessMicroNavigator,
    PartialNodeDetector,
    RadarStuckDetector,
    ThreeStageHarvestController,
    WaypointArrivalDetector,
    handle_extraction_failure,
)
from core.path_optimizer import RouteOptimizer
from core.route_manager import (
    DEFAULT_MAX_HARVESTS,
    DEFAULT_REGION,
    DEFAULT_RESOURCE_TYPE,
    DEFAULT_REST_SEC,
    DEFAULT_ROUTE,
    NodeData,
    RouteManager,
)
from core.sequential_router import SequentialRoutePlanner


# ==============================================================================
# TEST FIXTURES & MOCK OBJECTS
# ==============================================================================
class MockInputController:
    """Mock hardware/game input controller recording all invoked actions."""

    def __init__(self) -> None:
        self.keys_pressed: List[str] = []
        self.keys_held: List[Tuple[List[str], float]] = []
        self.tapped_keys: List[str] = []
        self.turned_angles: List[Tuple[float, Optional[float]]] = []
        self.realigned_targets: List[float] = []

    def tap_key(self, key: str) -> None:
        clean = str(key).lower()
        self.tapped_keys.append(clean)
        self.keys_pressed.append(clean)

    def press_key(self, key: str) -> None:
        clean = str(key).lower()
        self.keys_pressed.append(clean)

    def hold_keys(self, keys: List[str], duration: float = 1.0) -> None:
        self.keys_held.append(([str(k).lower() for k in keys], float(duration)))
        for k in keys:
            self.keys_pressed.append(str(k).lower())

    def key_down(self, key: str) -> None:
        self.keys_pressed.append(f"{str(key).lower()}_down")

    def key_up(self, key: str) -> None:
        self.keys_pressed.append(f"{str(key).lower()}_up")

    def turn_by_angle(self, angle: float, duration: Optional[float] = None) -> None:
        self.turned_angles.append((float(angle), duration))

    def realign_to_target(self, target: float) -> None:
        self.realigned_targets.append(float(target))

    def send_gather_key(self) -> None:
        self.keys_pressed.append("f")


@pytest.fixture
def mock_input() -> MockInputController:
    return MockInputController()


@pytest.fixture
def temp_routes_dir() -> str:
    temp_dir = tempfile.mkdtemp(prefix="test_routes_g10_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_game_frame() -> np.ndarray:
    """Generates a synthetic 1920x1080 game frame with minimap and ground corridor."""
    frame = np.full((1080, 1920, 3), 45, dtype=np.uint8)
    # Ground walking corridor: lower-middle (y: 560..885, x: 537..1382)
    cv2.rectangle(frame, (537, 560), (1382, 885), (60, 90, 60), -1)
    # Minimap area: upper-right (x: 1640..1900, y: 70..275)
    cv2.circle(frame, (1770, 175), 90, (120, 120, 120), -1)
    # Minimap player arrow
    pts = np.array([[1770, 160], [1760, 185], [1770, 180], [1780, 185]], dtype=np.int32)
    cv2.fillPoly(frame, [pts], (0, 255, 255))
    return frame


# ==============================================================================
# 1. 3X HARVEST CYCLE VERIFICATION (Agent G1 / G10)
# ==============================================================================
class TestThreeStageHarvestCycle:
    """
    Verifies the 3-Stage Harvest Loop (1/3 -> 2/3 -> 3/3):
    - 400ms re-prompt wait window.
    - Node anchoring (`is_anchored = True`).
    - Partial node early depletion (another player harvested earlier).
    - Essence Extraction failure handling (rights preserved, immediate retry).
    """

    def test_full_three_stage_harvest_progression(self, mock_input: MockInputController) -> None:
        """
        Tests complete 1/3 -> 2/3 -> 3/3 cycle:
        Bot remains anchored during stages 1 & 2, releases anchor and marks resting on stage 3.
        """
        slept_intervals: List[float] = []

        def mock_sleep(s: float) -> None:
            slept_intervals.append(float(s))

        node = {
            "id": 101,
            "current_harvests": 0,
            "max_harvests": 3,
            "state": "active",
        }

        controller = ThreeStageHarvestController(
            input_controller=mock_input,
            reprompt_wait_sec=0.40,
            sleep_fn=mock_sleep,
            test_mode=True,
        )

        # 1. Start node
        start_res = controller.start_node(node, max_harvests=3, gather_key="f")
        assert start_res["node_id"] == 101
        assert controller.is_anchored is True
        assert controller.current_harvests == 0
        assert controller.state == ThreeStageHarvestController.STATE_ANCHORED_GATHERING

        # 2. Stage 1 completed (prompt reappears after 400ms)
        stage1_res = controller.on_cast_completed(
            node_or_id=node,
            prompt_reappeared=True,
            gather_key="f",
            auto_press_next_stage=True,
        )
        assert stage1_res["current_harvests"] == 1
        assert stage1_res["stage"] == "1/3"
        assert stage1_res["is_anchored"] is True
        assert controller.is_anchored is True
        assert controller.current_harvests == 1
        assert stage1_res["status"] == "CONTINUE_NEXT_STAGE"
        assert 0.40 in slept_intervals

        # 3. Stage 2 completed (prompt reappears after 400ms)
        stage2_res = controller.on_cast_completed(
            node_or_id=node,
            prompt_reappeared=True,
            gather_key="f",
            auto_press_next_stage=True,
        )
        assert stage2_res["current_harvests"] == 2
        assert stage2_res["stage"] == "2/3"
        assert stage2_res["is_anchored"] is True
        assert controller.is_anchored is True
        assert controller.current_harvests == 2
        assert stage2_res["status"] == "CONTINUE_NEXT_STAGE"

        # 4. Stage 3 completed (final stage -> completed, anchor released)
        stage3_res = controller.on_cast_completed(
            node_or_id=node,
            prompt_reappeared=False,
            gather_key="f",
        )
        assert stage3_res["current_harvests"] == 3
        assert stage3_res["stage"] == "3/3"
        assert stage3_res["is_anchored"] is False
        assert controller.is_anchored is False
        assert controller.state == ThreeStageHarvestController.STATE_COMPLETED
        assert stage3_res["status"] == "COMPLETED"
        assert node["state"] == "resting"
        assert node["current_harvests"] == 0

    def test_reprompt_wait_interval_is_exact_400ms(self, mock_input: MockInputController) -> None:
        """
        Verifies that the 400ms re-prompt wait is not just configured but actually
        exercised: after stage 1/3 completes and the prompt reappears, the
        controller must really sleep 0.40s before pressing the gather key again.
        """
        controller = ThreeStageHarvestController()
        assert controller.reprompt_wait_sec == 0.40
        assert controller.reprompt_wait_ms == 400

        slept_intervals: List[float] = []
        live = ThreeStageHarvestController(
            input_controller=mock_input,
            reprompt_wait_sec=0.40,
            sleep_fn=lambda s: slept_intervals.append(s),
            test_mode=True,
        )
        node = {"id": 101, "map_x": 100, "map_y": 100, "max_harvests": 3, "current_harvests": 0, "state": "active"}
        live.start_node(node, max_harvests=3, gather_key="f")

        res = live.on_cast_completed(
            node_or_id=node,
            prompt_reappeared=True,
            gather_key="f",
            auto_press_next_stage=True,
        )

        assert res["status"] == "CONTINUE_NEXT_STAGE"
        assert res["current_harvests"] == 1
        assert res["waited_sec"] == 0.40
        # The 400ms wait must have really happened (recorded sleep call), not
        # merely been stored as a config attribute.
        assert slept_intervals == [0.40]
        assert live.reprompt_waits_sec == [0.40]
        assert live.last_reprompt_wait_sec == 0.40
        assert "f" in mock_input.keys_pressed

    def test_partial_node_early_depletion_after_stage_one(
        self, mock_input: MockInputController, temp_routes_dir: str
    ) -> None:
        """
        When another player harvested a node previously, the prompt does NOT reappear
        after stage 1/3. PartialNodeDetector must mark the node RESTING (5 min) immediately
        without waiting or attempting stage 2 and 3.
        """
        slept_intervals: List[float] = []
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("TestRegion")
        rm.add_node(region="TestRegion", node_id=1, map_x=100, map_y=100, max_harvests=3)

        detector = PartialNodeDetector(
            route_manager=rm,
            reprompt_wait_sec=0.40,
            sleep_fn=lambda s: slept_intervals.append(s),
        )

        controller = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=mock_input,
            partial_detector=detector,
            reprompt_wait_sec=0.40,
            sleep_fn=lambda s: slept_intervals.append(s),
            test_mode=True,
        )

        target_node = rm.get_node(1, region="TestRegion")
        assert target_node is not None
        controller.start_node(target_node, max_harvests=3)

        # Stage 1 finishes, but prompt does NOT reappear (source depleted)
        res = controller.on_cast_completed(
            node_or_id=target_node,
            prompt_reappeared=False,
            region="TestRegion",
        )

        assert res["partial_node"] is True
        assert res["node_depleted"] is True
        assert res["status"] == "PARTIAL_DEPLETED"
        assert res["is_anchored"] is False
        assert controller.is_anchored is False
        assert controller.state == ThreeStageHarvestController.STATE_PARTIAL_DEPLETED

        # Verify node transitioned to resting on RouteManager
        fresh_node = rm.get_node(1, region="TestRegion")
        assert fresh_node is not None
        assert fresh_node.state == "resting"

    def test_essence_extraction_failure_preserves_harvest_rights(
        self, mock_input: MockInputController
    ) -> None:
        """
        When chat feedback detects 'Essence Extraction failed':
        1. current_harvests is NOT incremented (rights preserved).
        2. Node state remains active.
        3. F/E gather key is immediately re-pressed to retry the stage.
        """
        node = {
            "id": 42,
            "current_harvests": 1,
            "max_harvests": 3,
            "state": "active",
        }

        controller = ThreeStageHarvestController(
            input_controller=mock_input,
            test_mode=True,
        )
        controller.start_node(node, max_harvests=3)
        controller.current_harvests = 1

        fail_result = controller.on_cast_completed(
            node_or_id=node,
            extraction_failed=True,
            gather_key="f",
        )

        assert fail_result["extraction_failed"] is True
        assert fail_result["status"] == "RETRY_STAGE"
        assert fail_result["current_harvests"] == 1  # Kept at 1, NOT incremented to 2!
        assert fail_result["retry_stage"] == "2/3"
        assert fail_result["re_pressed"] is True
        assert controller.is_anchored is True  # Bot stays anchored at the node
        assert "f" in mock_input.keys_pressed

        # Direct handle_extraction_failure verification
        direct_res = handle_extraction_failure(
            node_or_id=node,
            input_ctrl=mock_input,
            gather_key="f",
            extraction_failed=True,
            retry_press=True,
            test_mode=True,
        )
        assert direct_res["extraction_failed"] is True
        assert direct_res["rights_preserved"] is True
        assert direct_res["current_harvests"] == 1

    def test_gather_key_support_f_and_e(self, mock_input: MockInputController) -> None:
        """Verifies both 'F' (ground interact) and 'E' (floating/odyle interact) are dispatched correctly."""
        node = {"id": 10, "current_harvests": 0, "max_harvests": 3}
        controller = ThreeStageHarvestController(input_controller=mock_input, test_mode=True)

        # Test with 'e' key
        controller.start_node(node, gather_key="e", press_initial_key=True)
        assert "e" in mock_input.keys_pressed

        # Handle failure with 'e' key retry
        res = handle_extraction_failure(
            node_or_id=node,
            input_ctrl=mock_input,
            gather_key="e",
            extraction_failed=True,
            retry_press=True,
            test_mode=True,
        )
        assert res["gather_key"] == "e"
        assert res["re_pressed"] is True
        assert mock_input.keys_pressed.count("e") >= 2

    def test_route_manager_record_harvest_stage_progression(self, temp_routes_dir: str) -> None:
        """Verifies RouteManager.record_harvest_stage tracks incremental stages to resting state."""
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("TestRegion")
        rm.add_node(region="TestRegion", node_id=1, map_x=250, map_y=300, max_harvests=3)

        # Stage 1/3 -> returns (current_h, max_h, is_depleted)
        c1, m1, dep1 = rm.record_harvest_stage(1, region="TestRegion")
        assert c1 == 1
        assert m1 == 3
        assert dep1 is False
        node1 = rm.get_node(1, region="TestRegion")
        assert node1.state == "active"

        # Stage 2/3
        c2, m2, dep2 = rm.record_harvest_stage(1, region="TestRegion")
        assert c2 == 2
        assert m2 == 3
        assert dep2 is False
        node2 = rm.get_node(1, region="TestRegion")
        assert node2.state == "active"

        # Stage 3/3 (depletion)
        c3, m3, dep3 = rm.record_harvest_stage(1, region="TestRegion", rest_minutes=5.0)
        assert c3 == 3
        assert dep3 is True
        node3 = rm.get_node(1, region="TestRegion")
        assert node3.state == "resting"


# ==============================================================================
# 2. F8 WAYPOINT RECORDING & PERSISTENCE VERIFICATION (Agent G3 / G10)
# ==============================================================================
class TestF8WaypointRecordingAndPersistence:
    """
    Verifies F8 Waypoint Recording, Snap-to-Path, Deduplication & JSON Persistence:
    - Normal gameplay minimap hybrid recording (extracts clean 80x80 patch, inpainting).
    - Full map mode recording.
    - Resilient fallback (never returns None).
    - Save / Load roundtrip to region_data.json.
    - Duplicate route cleanup.
    """

    def test_f8_record_minimap_hybrid_mode(
        self, temp_routes_dir: str, sample_game_frame: np.ndarray
    ) -> None:
        """
        Verifies that pressing F8 in normal gameplay extracts player minimap position,
        cleans the player arrow with inpaint, saves an 80x80 patch image,
        and creates a NodeData with max_harvests=3.
        """
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("Altgard")

        node = rm.record_current_node(
            input_ctrl=sample_game_frame,
            region="Altgard",
            route="All",
            use_minimap_for_f8=True,
            resource_type="Orichalcum Ore",
        )

        assert node is not None
        assert isinstance(node, NodeData)
        assert node.id == 1
        assert node.region == "Altgard"
        assert node.max_harvests == 3
        assert node.resource_type == "Orichalcum Ore"
        assert node.norm_x > 0.0 and node.norm_y > 0.0

        # Verify patch file was created on disk
        patch_path = os.path.join(temp_routes_dir, "Altgard", "patches", f"node_{node.id}.png")
        assert os.path.isfile(patch_path), f"Patch image not found at {patch_path}"

        img = cv2.imread(patch_path, cv2.IMREAD_GRAYSCALE)
        assert img is not None
        assert img.shape == (80, 80), f"Expected 80x80 patch, got {img.shape}"

    def test_f8_fallback_safety_returns_none_on_capture_failure(self, temp_routes_dir: str) -> None:
        """
        Capture failure must NOT fabricate a node (F-1617 [O3]).

        record_current_node returns None when no frame can be captured
        (no frame, no vision engine) — the old behaviour recorded a
        synthetic viewport-center node from a blank frame and called it
        a successful recording. No patch file is written for the
        failed capture.
        """
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("Altgard")

        # Pass None input and None vision
        node = rm.record_current_node(input_ctrl=None, vision_engine=None, region="Altgard")
        assert node is None

        # No fabricated node was persisted and no black patch was saved.
        assert rm.get_node(1, region="Altgard") is None
        patch_path = os.path.join(temp_routes_dir, "Altgard", "patches", "node_1.png")
        assert not os.path.isfile(patch_path)

    def test_route_persistence_save_load_roundtrip(self, temp_routes_dir: str) -> None:
        """
        Verifies saving multiple recorded nodes to region_data.json and reloading
        faithfully restores all node coordinates, harvest limits, and attributes.
        """
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("Verteron")

        # Add 3 nodes via add_node
        n1 = rm.add_node(region="Verteron", node_id=1, map_x=150, map_y=200, resource_type="Aria Herb", max_harvests=3)
        n2 = rm.add_node(region="Verteron", node_id=2, map_x=350, map_y=400, resource_type="Iron Ore", max_harvests=3)
        n3 = rm.add_node(region="Verteron", node_id=3, map_x=550, map_y=600, resource_type="Odyle", max_harvests=3)

        # Save to disk
        save_ok = rm.save_region("Verteron")
        assert save_ok is True

        json_file = os.path.join(temp_routes_dir, "Verteron", "region_data.json")
        assert os.path.isfile(json_file)

        # Create a fresh RouteManager and load the saved data
        fresh_rm = RouteManager(base_dir=temp_routes_dir)
        fresh_rm.load_region("Verteron")

        loaded_nodes = fresh_rm.get_route_nodes(region_name="Verteron")
        assert len(loaded_nodes) == 3

        loaded_n1 = fresh_rm.get_node(1, region="Verteron")
        assert loaded_n1 is not None
        assert loaded_n1.map_x == 150
        assert loaded_n1.map_y == 200
        assert loaded_n1.resource_type == "Aria Herb"
        assert loaded_n1.max_harvests == 3

    def test_node_deduplication_and_route_membership(self, temp_routes_dir: str) -> None:
        """Verifies duplicate custom route names on a node are cleanly deduplicated."""
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("Altgard")
        rm.create_route("Altgard", "MiningLoop")

        node = rm.add_node(region="Altgard", node_id=1, map_x=100, map_y=100, routes=["MiningLoop", "SpecialLoop", "MiningLoop"])
        # Duplicate custom routes must be pruned
        assert node.routes.count("MiningLoop") == 1


# ==============================================================================
# 3. 2-OPT TSP ROUTE OPTIMIZATION VERIFICATION (Agent G3 / G5 / G10)
# ==============================================================================
class TestTwoOptTSPRouteOptimization:
    """
    Verifies Traveling Salesperson (TSP) 2-Opt local search optimization:
    - RouteOptimizer (path_optimizer.py): 2-opt swaps untangle crossed edges.
    - optimize_node_sequence reduces total tour distance.
    - SequentialRoutePlanner (sequential_router.py): smooth_loop produces forward progression.
    """

    def test_two_opt_untangles_crossing_edges(self) -> None:
        """
        Given a simple 4-node square:
        Node 0: (0, 0), Node 1: (0, 10), Node 2: (10, 10), Node 3: (10, 0)
        Uncrossed perimeter tour: 0 -> 1 -> 2 -> 3 -> 0 (distance = 40.0).
        Tangled 'hour-glass' tour: 0 -> 2 -> 1 -> 3 -> 0 (distance = 14.14 + 10.0 + 14.14 + 10.0 = 48.28).
        2-Opt swap must untangle the crossing and return distance 40.0.
        """
        coords = [(0.0, 0.0), (0.0, 10.0), (10.0, 10.0), (10.0, 0.0)]
        optimizer = RouteOptimizer()

        dist_matrix = optimizer._build_distance_matrix(coords)
        tangled_order = [0, 2, 1, 3]

        unopt_dist = optimizer._compute_tour_distance(tangled_order, dist_matrix, closed_loop=True)
        assert unopt_dist > 45.0  # Around ~48.28

        improved_order = optimizer._two_opt_improve(tangled_order, dist_matrix, closed_loop=True)
        opt_dist = optimizer._compute_tour_distance(improved_order, dist_matrix, closed_loop=True)

        assert abs(opt_dist - 40.0) < 1e-4, f"Expected 40.0, got {opt_dist}"
        # Crossed lines eliminated
        crossings = optimizer._count_coords_crossings([coords[i] for i in improved_order], closed_loop=True)
        assert crossings == 0

    def test_optimize_node_sequence_reduces_tour_distance(self) -> None:
        """
        Takes 8 nodes arranged around a circle in a randomized zigzag order,
        verifying optimize_node_sequence achieves a strictly shorter tour distance.
        """
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        radius = 100.0
        coords = [(radius * math.cos(a), radius * math.sin(a)) for a in angles]

        # Tangled order with zigzags across circle
        scrambled_order = [0, 4, 1, 5, 2, 6, 3, 7]
        nodes = [{"id": i + 1, "map_x": coords[idx][0], "map_y": coords[idx][1]} for i, idx in enumerate(scrambled_order)]

        optimizer = RouteOptimizer()
        ordered_nodes = optimizer.optimize_node_sequence(nodes)

        assert len(ordered_nodes) == 8
        stats = optimizer.calculate_route_distance_stats(nodes)
        assert stats["optimized_distance"] < stats["unoptimized_distance"]
        assert stats["improvement_pct"] > 15.0

    def test_sequential_route_planner_smooth_loop(self) -> None:
        """
        Verifies SequentialRoutePlanner builds a continuous smooth loop without
        jagged turns: the output must be a true permutation (no node dropped or
        repeated), the closed-loop tour must be strictly shorter than the
        scrambled input tour, no two tour edges may cross, and consecutive
        legs must not reverse direction (smooth heading progression).
        """
        coords = [
            (0.0, 0.0),
            (20.0, 5.0),
            (40.0, 25.0),
            (45.0, 50.0),
            (30.0, 75.0),
            (10.0, 70.0),
            (-10.0, 45.0),
            (-5.0, 15.0),
        ]
        # Scramble coordinates
        scrambled_coords = [coords[i] for i in [0, 4, 1, 5, 2, 6, 3, 7]]
        nodes = [NodeData(id=i + 1, map_x=int(c[0]), map_y=int(c[1])) for i, c in enumerate(scrambled_coords)]

        planner = SequentialRoutePlanner()
        ordered_nodes = planner.build_sequential_order(nodes, mode="smooth_loop")

        assert len(ordered_nodes) == len(nodes)
        visited_ids = [n.id for n in ordered_nodes]
        assert set(visited_ids) == set(range(1, 9))
        # Exact permutation: every id exactly once, and the order really changed.
        assert sorted(visited_ids) == list(range(1, 9))
        assert len(set(visited_ids)) == 8
        assert visited_ids != [n.id for n in nodes]

        def closed_tour_distance(pts: List[Tuple[float, float]]) -> float:
            total = sum(
                math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                for i in range(len(pts) - 1)
            )
            return total + math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])

        ordered_pts = [(float(n.map_x), float(n.map_y)) for n in ordered_nodes]
        scrambled_dist = closed_tour_distance([(float(x), float(y)) for x, y in scrambled_coords])
        ordered_dist = closed_tour_distance(ordered_pts)
        assert ordered_dist < scrambled_dist
        assert ordered_dist < 0.7 * scrambled_dist

        # No edge crossings in the closed loop (smooth loop must untangle).
        def _segments_cross(a1, a2, b1, b2) -> bool:
            def _ccw(a, b, c):
                return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

            return _ccw(a1, b1, b2) != _ccw(a2, b1, b2) and _ccw(a1, a2, b1) != _ccw(a1, a2, b2)

        n = len(ordered_pts)
        crossings = 0
        for i in range(n):
            for j in range(i + 1, n):
                if len({i, (i + 1) % n, j, (j + 1) % n}) < 4:
                    continue
                if _segments_cross(ordered_pts[i], ordered_pts[(i + 1) % n], ordered_pts[j], ordered_pts[(j + 1) % n]):
                    crossings += 1
        assert crossings == 0, f"Smooth loop tour has {crossings} edge crossing(s)"

        # Heading continuity: no leg reverses direction (max bearing change < 120°).
        def _bearing(a, b) -> float:
            return math.degrees(math.atan2(b[0] - a[0], -(b[1] - a[1]))) % 360.0

        legs = [_bearing(ordered_pts[i], ordered_pts[(i + 1) % n]) for i in range(n)]
        max_turn = max(abs((legs[(i + 1) % n] - legs[i] + 180.0) % 360.0 - 180.0) for i in range(n))
        assert max_turn < 120.0, f"Jagged turn in smooth loop: max bearing change {max_turn:.1f}°"

    def test_sequential_route_planner_apply_to_route_manager(self, temp_routes_dir: str) -> None:
        """Verifies reordering is committed to RouteManager."""
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("Altgard")

        for i in range(1, 6):
            rm.add_node(region="Altgard", node_id=i, map_x=i * 50, map_y=i * 30)

        # Expected order computed independently BEFORE apply mutates the manager.
        pre_apply_nodes = rm.get_route_nodes(region_name="Altgard", route_name="All")
        pre_apply_ids = [n.id for n in pre_apply_nodes]
        assert pre_apply_ids == [1, 2, 3, 4, 5]

        planner = SequentialRoutePlanner()
        expected_nodes = planner.build_sequential_order(list(pre_apply_nodes), mode="smooth_loop")
        expected_ids = [n.id for n in expected_nodes]
        assert len(expected_ids) == 5
        assert set(expected_ids) == set(pre_apply_ids)
        # The smooth loop must actually reorder this diagonal line, not echo it.
        assert expected_ids != pre_apply_ids

        ordered_nodes = planner.apply_sequential_order_to_route(rm, region="Altgard", route="All", mode="smooth_loop")

        assert len(ordered_nodes) == 5
        # Returned order matches the independently computed sequential order.
        assert [n.id for n in ordered_nodes] == expected_ids
        # seq_index is assigned 1..N in the returned order.
        assert [n.seq_index for n in ordered_nodes] == [1, 2, 3, 4, 5]
        # RouteManager reads back the new order (committed, not just returned).
        assert [n.id for n in rm.get_route_nodes(region_name="Altgard", route_name="All")] == expected_ids
        # ... and the order survives a reload from disk.
        fresh_rm = RouteManager(base_dir=temp_routes_dir)
        assert [n.id for n in fresh_rm.get_route_nodes(region_name="Altgard", route_name="All")] == expected_ids


# ==============================================================================
# 4. ANTI-INVERSION & HEADING ALIGNMENT VERIFICATION (Agent G5 / G10)
# ==============================================================================
class TestAntiInversionAndHeadingAlignment:
    """
    Verifies compass bearing calculation and shortest rotational turn:
    - 0° = North, 90° = East, 180° = South, 270° = West.
    - Eliminates 180° Cartesian inversion bug.
    - Shortest rotational path [-180°, +180°].
    - Turn damping controller deadband (±5°..±8°) oscillation prevention.
    """

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_cardinal_and_intercardinal_bearings(self, module: Any) -> None:
        """
        Minimap screen coords: North is -Y.
        Confirms bearing follows compass convention:
        (0, -Y) -> 0° North, (+X, 0) -> 90° East, (0, +Y) -> 180° South, (-X, 0) -> 270° West.
        """
        origin = (100.0, 100.0)

        # North (0, -dy)
        assert module.compute_bearing(origin, (100.0, 50.0)) == 0.0
        # North-East (+dx, -dy)
        assert abs(module.compute_bearing(origin, (150.0, 50.0)) - 45.0) < 0.01
        # East (+dx, 0)
        assert module.compute_bearing(origin, (150.0, 100.0)) == 90.0
        # South-East (+dx, +dy)
        assert abs(module.compute_bearing(origin, (150.0, 150.0)) - 135.0) < 0.01
        # South (0, +dy)
        assert module.compute_bearing(origin, (100.0, 150.0)) == 180.0
        # South-West (-dx, +dy)
        assert abs(module.compute_bearing(origin, (50.0, 150.0)) - 225.0) < 0.01
        # West (-dx, 0)
        assert module.compute_bearing(origin, (50.0, 100.0)) == 270.0
        # North-West (-dx, -dy)
        assert abs(module.compute_bearing(origin, (50.0, 50.0)) - 315.0) < 0.01

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_shortest_rotational_path_across_north_boundary(self, module: Any) -> None:
        """
        Turning across the 0°/360° North seam:
        Heading 350° to target 10° -> +20.0° (turn right, NOT -340°).
        Heading 10° to target 350° -> -20.0° (turn left, NOT +340°).
        """
        diff_right = module.compute_angle_diff(target_bearing=10.0, player_heading=350.0)
        assert diff_right == 20.0

        diff_left = module.compute_angle_diff(target_bearing=350.0, player_heading=10.0)
        assert diff_left == -20.0

    def test_turn_damping_controller_deadband_zone(self) -> None:
        """
        Inside deadband (±5°..±8°), micro-turns must be suppressed
        to prevent rapid steering jitter / left-right oscillation.
        """
        damper = nav.TurnDampingController(deadband_deg=6.0)

        # 1. Inside deadband (+4° error)
        damped_in = damper.calculate_turn(target_bearing=4.0, player_heading=0.0)
        assert damped_in["in_deadband"] is True
        assert damped_in["turn_needed"] is False
        assert damped_in["turn_direction"] == "straight"
        assert damped_in["turn_duration_sec"] == 0.0

        # 2. Outside deadband (+35° error) -> turn right
        damped_out = damper.calculate_turn(target_bearing=35.0, player_heading=0.0)
        assert damped_out["in_deadband"] is False
        assert damped_out["turn_needed"] is True
        assert damped_out["turn_direction"] == "right"
        assert damped_out["turn_duration_sec"] > 0.04

        # 3. Negative outside deadband (-45° error) -> turn left
        damped_left = damper.calculate_turn(target_bearing=315.0, player_heading=0.0)
        assert damped_left["in_deadband"] is False
        assert damped_left["turn_direction"] == "left"
        assert damped_left["turn_duration_sec"] > 0.04

    def test_navigator_steer_step_end_to_end(self, temp_routes_dir: str, mock_input: MockInputController) -> None:
        """Verifies steer_step computes bearing and damped turn command in one call."""
        rm = RouteManager(base_dir=temp_routes_dir)
        navigator = nav.MapNavigator(route_manager=rm, vision=None, driver=mock_input)

        step = navigator.steer_step(
            current_pos=(100.0, 100.0),
            target_pos=(100.0, 40.0),  # Straight North
            player_heading=345.0,       # Pointing slightly left of North
        )

        assert step["target_bearing"] == 0.0
        assert step["player_heading"] == 345.0
        assert step["angle_diff"] == 15.0  # Needs +15° right turn
        assert step["turn_needed"] is True
        assert step["turn_direction"] == "right"


# ==============================================================================
# 5. 3D OBSTACLE AVOIDANCE & CORRIDOR STEERING VERIFICATION (Agent G7 / G10)
# ==============================================================================
class TestObstacleAvoidanceAndCorridorSteering:
    """
    Verifies 3D Forward Corridor Navigation & Obstacle Avoidance:
    - Lower-middle viewport 3-lane evaluation (Left, Center, Right).
    - Steering bias: steer_left, steer_right, clear.
    - Adjusted bearing application during walking.
    - ZAxisCliffFilter elevation rejection.
    """

    def test_corridor_path_clear_ground(self) -> None:
        """When the walking corridor is clear, steering bias is 'clear' with 0.0° offset."""
        frame = np.full((1080, 1920, 3), 50, dtype=np.uint8)
        cnav = CorridorNavigator()

        eval_res = cnav.evaluate_corridor_path(frame)
        assert eval_res["steering_bias"] == "clear"
        assert eval_res["recommended_turn_offset_deg"] == 0.0
        assert eval_res["obstacle_detected"] is False

    def test_corridor_steer_step_applies_offset_bearing(self) -> None:
        """
        When corridor evaluation suggests a turn offset (e.g. -15.0° left to avoid obstacle),
        steer_step must adjust target bearing accordingly.

        A live obstacle detection in the center lane (class 14, cx=0.5) must
        produce a real, non-zero offset: the adjusted bearing is the target
        bearing shifted by exactly the recommended offset, and the resulting
        turn command must steer left around the obstacle.
        """
        cnav = CorridorNavigator()
        frame = np.full((1080, 1920, 3), 50, dtype=np.uint8)

        step = cnav.steer_step(
            current_pos=(500.0, 500.0),
            target_pos=(500.0, 200.0),  # North (bearing 0.0°)
            player_heading=0.0,
            frame=frame,
            yolo_detections=[
                {"class_id": 14, "class_name": "obstacle", "confidence": 0.9, "cx": 0.5, "cy": 0.6}
            ],
        )

        assert step["target_bearing"] == 0.0
        assert "adjusted_bearing" in step
        assert "corridor_eval" in step

        corridor_eval = step["corridor_eval"]
        assert corridor_eval["obstacle_detected"] is True
        assert corridor_eval["steering_bias"] == "steer_left"
        assert corridor_eval["recommended_turn_offset_deg"] == -18.0
        # Center lane is blocked, side lanes stay clear.
        assert corridor_eval["lane_scores"]["center"] < 0.42
        assert corridor_eval["lane_scores"]["left"] > corridor_eval["lane_scores"]["center"]
        assert corridor_eval["lane_scores"]["right"] > corridor_eval["lane_scores"]["center"]

        # Offset is really applied: adjusted = (target + offset) mod 360.
        expected_adjusted = (0.0 + corridor_eval["recommended_turn_offset_deg"] + 360.0) % 360.0
        assert step["adjusted_bearing"] == expected_adjusted == 342.0
        assert step["adjusted_bearing"] != step["target_bearing"]
        # The damped turn follows the adjusted bearing, not the raw target.
        assert step["angle_diff"] == -18.0
        assert step["turn_direction"] == "left"
        assert step["turn_needed"] is True

    def test_z_axis_cliff_filter_marks_unreachable_node(self) -> None:
        """
        When a node is close on 2D minimap (< 25px) but interaction fails for 2 consecutive
        attempts (cliff elevation mismatch), it must be flagged unreachable_cliff = True.
        """
        node = {"id": 88, "x": 100.0, "y": 100.0}
        cliff_filter = ZAxisCliffFilter(close_minimap_radius_px=25.0, max_failed_approaches=2)

        # Attempt 1: Failed within 15px (< 25px threshold)
        flag1 = cliff_filter.record_approach_attempt(node, minimap_dist_px=15.0, interaction_succeeded=False)
        assert flag1 is False
        assert node.get("unreachable_cliff") is not True

        # Attempt 2: Failed again -> threshold reached
        flag2 = cliff_filter.record_approach_attempt(node, minimap_dist_px=14.0, interaction_succeeded=False)
        assert flag2 is True
        assert node.get("unreachable_cliff") is True
        assert node.get("altitude_band") == "cliff_mismatch"


# ==============================================================================
# 6. UNSTICK MANEUVERS VERIFICATION (Agent G7 / G10)
# ==============================================================================
class TestUnstickManeuvers:
    """
    Verifies stuck detection and escape maneuvers:
    - PhaseCorrelateUnstuckManeuver: shift < 1.2px over 2.5s.
    - Action sequence: Space (Jump) -> A+W / D+W (Strafe) -> Realign.
    - Alternating strafe directions.
    - RadarStuckDetector: distance < threshold over timeout.
    """

    def test_phase_correlate_stuck_detection_timing(self) -> None:
        """
        When minimap image shift remains < 1.2px for >= 2.5 seconds with movement keys active,
        is_stuck must transition to True.
        """
        unstuck = PhaseCorrelateUnstuckManeuver(stuck_threshold_px=1.2, stuck_duration_sec=2.5)
        minimap_patch = np.full((120, 120), 100, dtype=np.uint8)

        t0 = 1000.0
        # Frame 1 at t0
        r1 = unstuck.update_walking_frame(minimap_patch, is_walking=True, now=t0)
        assert r1["is_stuck"] is False

        # Frame 2 at t0 + 1.0s (not stuck yet)
        r2 = unstuck.update_walking_frame(minimap_patch, is_walking=True, now=t0 + 1.0)
        assert r2["is_stuck"] is False

        # Frame 3 at t0 + 2.6s (stuck timeout exceeded)
        r3 = unstuck.update_walking_frame(minimap_patch, is_walking=True, now=t0 + 2.6)
        assert r3["is_stuck"] is True
        assert r3["stuck_elapsed_sec"] >= 2.5

    def test_phase_correlate_unstuck_sequence_actions_and_alternation(
        self, mock_input: MockInputController
    ) -> None:
        """
        Verifies unstick sequence:
        1. SPACE (Jump).
        2. A+W (Strafe for 1.2s) on attempt 1.
        3. D+W (Strafe for 1.2s) on attempt 2 (alternating).
        4. Realign to target bearing.
        """
        unstuck = PhaseCorrelateUnstuckManeuver(strafe_duration_sec=1.2)

        # Maneuver #1
        res1 = unstuck.execute_unstuck_sequence(
            input_controller=mock_input,
            target_angle_deg=90.0,
            current_angle_deg=70.0,
        )
        assert res1["executed"] is True
        assert res1["sequence"] == ["SPACE", "A+W", "REALIGN"]
        assert res1["strafe_key"] == "a"
        assert res1["unstuck_count"] == 1
        assert "space" in mock_input.keys_pressed
        assert "a" in mock_input.keys_pressed
        assert "w" in mock_input.keys_pressed

        # Maneuver #2 (must alternate to 'D+W')
        res2 = unstuck.execute_unstuck_sequence(
            input_controller=mock_input,
            target_angle_deg=90.0,
            current_angle_deg=70.0,
        )
        assert res2["sequence"] == ["SPACE", "D+W", "REALIGN"]
        assert res2["strafe_key"] == "d"
        assert res2["unstuck_count"] == 2
        assert "d" in mock_input.keys_pressed

    def test_radar_stuck_detector_and_recovery(self, mock_input: MockInputController) -> None:
        """Verifies RadarStuckDetector triggers escape maneuver when distance < threshold."""
        detector = RadarStuckDetector(stuck_timeout_sec=3.0, strafe_duration_sec=1.2)

        # Initial check at (100, 100)
        c1 = detector.update_position(player_pos=(100.0, 100.0), now=100.0, movement_keys_held=True)
        assert c1["is_stuck"] is False

        # No movement for 3.2 seconds
        c2 = detector.update_position(player_pos=(100.1, 100.1), now=103.2, movement_keys_held=True)
        assert c2["is_stuck"] is True

        # Execute unstick
        maneuver_res = detector.execute_unstick_maneuver(
            input_controller=mock_input,
            target_bearing=45.0,
            player_heading=30.0,
        )
        assert maneuver_res["executed"] is True
        assert maneuver_res["sequence"] == ["SPACE", "A+W", "REALIGN"]
        assert detector.is_stuck is False


# ==============================================================================
# 7. FULL 10-AGENT GATHER, WALK & ROUTE INTEGRATION SCENARIO
# ==============================================================================
class TestFullGatherWalkRouteIntegrationSuite:
    """
    End-to-end integration test linking all G1..G10 capabilities into a cohesive flow:
    1. Route Creation & Persistence with F8 (G3).
    2. 2-Opt TSP Route Optimization (G3/G5).
    3. Waypoint Navigation with Anti-Inversion Heading (G5).
    4. Corridor Obstacle Avoidance (G7).
    5. Unstick Jump+Strafe Recovery (G7).
    6. Waypoint Arrival Detection (G5).
    7. 3x Harvest Cycle with Anchoring & 400ms Re-prompt (G1).
    """

    def test_complete_gather_walk_route_lifecycle(
        self, temp_routes_dir: str, mock_input: MockInputController, sample_game_frame: np.ndarray
    ) -> None:
        # ------------------------------------------------------------------
        # Step 1: Initialize Route Manager and Record 4 Waypoints
        # ------------------------------------------------------------------
        rm = RouteManager(base_dir=temp_routes_dir)
        rm.create_region("Altgard")

        coords = [
            (100, 100),
            (300, 500),
            (150, 450),
            (400, 150),
        ]
        nodes: List[NodeData] = []
        for i, (cx, cy) in enumerate(coords, start=1):
            n = rm.add_node(
                region="Altgard",
                node_id=i,
                map_x=cx,
                map_y=cy,
                resource_type="Orichalcum Ore",
                max_harvests=3,
            )
            nodes.append(n)

        assert len(nodes) == 4
        save_ok = rm.save_region("Altgard")
        assert save_ok is True

        # ------------------------------------------------------------------
        # Step 2: 2-Opt TSP Route Optimization (Minimize Tour Distance)
        # ------------------------------------------------------------------
        optimizer = RouteOptimizer()
        ordered_nodes = optimizer.optimize_node_sequence(nodes)
        assert len(ordered_nodes) == 4

        # Reorder route based on TSP
        planner = SequentialRoutePlanner()
        apply_res = planner.apply_sequential_order_to_route(rm, region="Altgard", route="All", mode="smooth_loop")
        assert len(apply_res) == 4

        # ------------------------------------------------------------------
        # Step 3: Waypoint Navigation with Anti-Inversion Compass Heading
        # ------------------------------------------------------------------
        navigator = nav.MapNavigator(route_manager=rm, vision=None, driver=mock_input)
        target_node = rm.get_node(1, region="Altgard")
        assert target_node is not None

        player_pos = (100.0, 200.0)
        target_pos = (float(target_node.map_x), float(target_node.map_y))

        steer = navigator.steer_step(
            current_pos=player_pos,
            target_pos=target_pos,
            player_heading=350.0,  # Heading nearly North
        )
        assert steer["target_bearing"] == 0.0  # Straight North (-Y)
        assert steer["turn_direction"] == "right"
        assert steer["angle_diff"] == 10.0

        # ------------------------------------------------------------------
        # Step 4: Corridor Navigation & Obstacle Avoidance
        # ------------------------------------------------------------------
        cnav = CorridorNavigator()
        step_eval = cnav.steer_step(
            current_pos=player_pos,
            target_pos=target_pos,
            player_heading=0.0,
            frame=sample_game_frame,
        )
        # Target is straight North of the player (0, -Y): bearing must be 0.0,
        # not just any number under the "target_bearing" key.
        assert step_eval["target_bearing"] == 0.0
        assert step_eval["player_heading"] == 0.0
        # Synthetic frame has a clear corridor: no obstacle, no turn offset.
        corridor_eval = step_eval["corridor_eval"]
        assert corridor_eval is not None
        assert corridor_eval["obstacle_detected"] is False
        assert corridor_eval["recommended_turn_offset_deg"] == 0.0
        # Adjusted bearing is the target bearing shifted by the corridor offset.
        assert step_eval["adjusted_bearing"] == pytest.approx(
            (step_eval["target_bearing"] + corridor_eval["recommended_turn_offset_deg"]) % 360.0
        )
        assert step_eval["adjusted_bearing"] == 0.0
        # Heading 0.0 already points at the 0.0 target: zero angular error,
        # inside the deadband, no turn needed, direction "straight".
        assert step_eval["angle_diff"] == 0.0
        assert step_eval["angle_diff"] == pytest.approx(
            cor_nav.compute_angle_diff(step_eval["adjusted_bearing"], 0.0)
        )
        assert step_eval["abs_error"] == 0.0
        assert step_eval["turn_direction"] == "straight"
        assert step_eval["in_deadband"] is True
        assert step_eval["turn_needed"] is False
        assert step_eval["turn_duration_sec"] == 0.0

        # ------------------------------------------------------------------
        # Step 5: Unstick Recovery on Obstacle Collision
        # ------------------------------------------------------------------
        unstuck_ctrl = PhaseCorrelateUnstuckManeuver()
        unstuck_res = unstuck_ctrl.execute_unstuck_sequence(
            input_controller=mock_input,
            target_angle_deg=0.0,
            current_angle_deg=345.0,
        )
        assert unstuck_res["executed"] is True
        assert "SPACE" in unstuck_res["sequence"]
        assert "A+W" in unstuck_res["sequence"]
        assert "REALIGN" in unstuck_res["sequence"]

        # ------------------------------------------------------------------
        # Step 6: Waypoint Arrival
        # ------------------------------------------------------------------
        arrival_detector = WaypointArrivalDetector(waypoint_radius_px=20.0, waypoint_radius_m=3.0)
        arr_info = arrival_detector.is_arrival_reached(
            player_pos=(100.0, 101.5),
            waypoint_pos=target_pos,
        )
        assert arr_info["arrived"] is True

        # ------------------------------------------------------------------
        # Step 7: 3x Harvest Cycle with Re-prompt & Anchoring
        # ------------------------------------------------------------------
        harvest_controller = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=mock_input,
            reprompt_wait_sec=0.40,
            sleep_fn=lambda s: None,  # Instant in-memory simulation
            test_mode=True,
        )

        harvest_controller.start_node(target_node, max_harvests=3, gather_key="f")
        assert harvest_controller.is_anchored is True

        # Stage 1 (1/3)
        h1 = harvest_controller.on_cast_completed(target_node, prompt_reappeared=True)
        assert h1["current_harvests"] == 1
        assert harvest_controller.is_anchored is True

        # Stage 2 (2/3)
        h2 = harvest_controller.on_cast_completed(target_node, prompt_reappeared=True)
        assert h2["current_harvests"] == 2
        assert harvest_controller.is_anchored is True

        # Stage 3 (3/3 -> complete)
        h3 = harvest_controller.on_cast_completed(target_node, prompt_reappeared=False)
        assert h3["current_harvests"] == 3
        assert harvest_controller.is_anchored is False
        assert target_node.state == "resting"

        # Verify next node transition is ready
        next_node = rm.get_next_ready_node(region="Altgard")
        assert next_node is not None
        assert next_node.id != target_node.id
