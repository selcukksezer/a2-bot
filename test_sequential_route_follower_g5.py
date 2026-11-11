# -*- coding: utf-8 -*-
"""
Agent G5 Verification Suite:
1. Cyclical 3+ Node Route Following by Sequence Index
2. 180° Compass Heading Anti-Inversion
3. ±6° Turn Damping & Oscillation Prevention
4. Arrival Thresholds (<=20px / <=3m)
5. Nearest-Waypoint Resume after F10 Pause / Combat Interrupt
6. Interoperability between SequentialRoutePlanner, MapNavigator, and CyclicalRouteFollower
"""

import math
import time
import pytest
from typing import Any, Dict, List, Tuple

import core.navigator as nav
from core.navigator import (
    DEFAULT_DEADBAND_DEG,
    DEFAULT_WAYPOINT_RADIUS_M,
    DEFAULT_WAYPOINT_RADIUS_PX,
    CyclicalRouteFollower,
    MapNavigator,
    TurnDampingController,
    WaypointArrivalDetector,
    anti_inversion_compass_heading,
    apply_turn_damping,
    cartesian_to_compass,
    compass_to_cartesian,
    compute_angle_diff,
    compute_bearing,
    find_nearest_waypoint,
    normalize_compass_heading,
    resume_route_after_interrupt,
)
import core.sequential_router as router
from core.sequential_router import SequentialRoutePlanner


# ==============================================================================
# TEST 1: CYCLICAL 3+ NODE ROUTE FOLLOWING BY SEQUENCE INDEX
# ==============================================================================
class TestCyclicalRouteFollowing:
    """Verifies cyclical 3+ node traversal by sequence index without ping-pong or stalling."""

    def test_three_node_cyclical_progression(self):
        """Confirms 0 -> 1 -> 2 -> 0 -> 1 -> 2 sequence looping for 3 nodes."""
        nodes = [
            {"id": 1, "name": "Node_1", "map_x": 100.0, "map_y": 100.0, "seq_index": 1},
            {"id": 2, "name": "Node_2", "map_x": 200.0, "map_y": 100.0, "seq_index": 2},
            {"id": 3, "name": "Node_3", "map_x": 200.0, "map_y": 200.0, "seq_index": 3},
        ]
        planner = SequentialRoutePlanner()

        # Step 0 -> 1
        idx1, n1 = planner.get_next_cyclical_waypoint(nodes, current_index=0)
        assert idx1 == 1
        assert n1["id"] == 2

        # Step 1 -> 2
        idx2, n2 = planner.get_next_cyclical_waypoint(nodes, current_index=1)
        assert idx2 == 2
        assert n2["id"] == 3

        # Step 2 -> 0 (loop back to start)
        idx0, n0 = planner.get_next_cyclical_waypoint(nodes, current_index=2)
        assert idx0 == 0
        assert n0["id"] == 1

        # Second lap continuation 0 -> 1
        idx1_b, n1_b = planner.get_next_cyclical_waypoint(nodes, current_index=0)
        assert idx1_b == 1
        assert n1_b["id"] == 2

    def test_cyclical_route_follower_lap_tracking(self):
        """Verifies CyclicalRouteFollower completes full laps and increments lap counter."""
        nodes = [
            {"id": 10, "map_x": 100.0, "map_y": 100.0},
            {"id": 20, "map_x": 200.0, "map_y": 100.0},
            {"id": 30, "map_x": 150.0, "map_y": 200.0},
        ]
        follower = CyclicalRouteFollower(nodes=nodes, waypoint_radius_px=20.0, start_index=0)
        assert follower.current_index == 0
        assert follower.total_completed_laps == 0

        # Reach node 0 (player at 105, 105 -> dist ~7px <= 20px)
        s1 = follower.step(player_pos=(105.0, 105.0), player_heading=90.0)
        assert s1["arrived"] is True
        assert s1["transitioned"] is True
        assert s1["current_index"] == 1
        assert follower.total_completed_laps == 0

        # Reach node 1 (player at 195, 102 -> dist ~5px <= 20px)
        s2 = follower.step(player_pos=(195.0, 102.0), player_heading=180.0)
        assert s2["arrived"] is True
        assert s2["transitioned"] is True
        assert s2["current_index"] == 2
        assert follower.total_completed_laps == 0

        # Reach node 2 (player at 150, 198 -> dist ~2px <= 20px) -> wrap to 0, lap completed!
        s3 = follower.step(player_pos=(150.0, 198.0), player_heading=0.0)
        assert s3["arrived"] is True
        assert s3["transitioned"] is True
        assert s3["current_index"] == 0
        assert follower.total_completed_laps == 1

    def test_four_plus_nodes_cyclical_traversal(self):
        """Ensures routes with 4+ nodes advance strictly by sequence index."""
        nodes = [{"id": i, "map_x": i * 50.0, "map_y": i * 50.0} for i in range(5)]
        follower = CyclicalRouteFollower(nodes=nodes, waypoint_radius_px=20.0)

        # Traverse all 5 nodes and loop back
        for expected_idx in range(5):
            assert follower.current_index == expected_idx
            pos = (nodes[expected_idx]["map_x"], nodes[expected_idx]["map_y"])
            res = follower.step(player_pos=pos, player_heading=45.0)
            assert res["arrived"] is True
            assert res["transitioned"] is True

        # After visiting index 4, should be back at index 0 with 1 lap
        assert follower.current_index == 0
        assert follower.total_completed_laps == 1


# ==============================================================================
# TEST 2: 180° COMPASS HEADING ANTI-INVERSION
# ==============================================================================
class TestHeadingAntiInversion:
    """Verifies compass bearing calculation and 180° anti-inversion logic."""

    def test_compass_bearing_cardinal_directions(self):
        """Ensures North=0°, East=90°, South=180°, West=270° in screen coordinates."""
        center = (200.0, 200.0)
        # North (-Y)
        assert compute_bearing(center, (200.0, 100.0)) == 0.0
        # East (+X)
        assert compute_bearing(center, (300.0, 200.0)) == 90.0
        # South (+Y)
        assert compute_bearing(center, (200.0, 300.0)) == 180.0
        # West (-X)
        assert compute_bearing(center, (100.0, 200.0)) == 270.0

    def test_shortest_rotational_angle_difference(self):
        """Verifies [-180°, +180°] shortest path wrapping across 0°/360°."""
        # 355° to 5° -> +10° (right turn)
        assert compute_angle_diff(5.0, 355.0) == 10.0
        # 5° to 355° -> -10° (left turn)
        assert compute_angle_diff(355.0, 5.0) == -10.0
        # 90° to 270° -> 180°
        assert abs(compute_angle_diff(270.0, 90.0)) == 180.0

    def test_anti_inversion_heading_correction(self):
        """
        When heading detection erroneously flips by 180° opposite to forward bearing,
        anti_inversion_compass_heading corrects it back to forward alignment.
        """
        forward_bearing = 90.0  # Heading East towards waypoint
        inverted_heading = 270.0  # Erroneous 180° inverted heading (pointing West)

        corrected = anti_inversion_compass_heading(
            heading=inverted_heading,
            reference_bearing=forward_bearing,
            threshold_deg=150.0,
        )
        assert corrected == 90.0, f"Expected 90°, got {corrected}"

        # If heading is already aligned (e.g. 85° vs 90°), do not invert
        aligned = anti_inversion_compass_heading(
            heading=85.0,
            reference_bearing=forward_bearing,
            threshold_deg=150.0,
        )
        assert aligned == 85.0

    def test_cartesian_and_compass_conversions(self):
        """Verifies conversion between Cartesian (0°=East, CCW) and Compass (0°=North, CW)."""
        # Cartesian 90° (North, +Y in math) -> Compass 0° (North)
        assert cartesian_to_compass(90.0) == 0.0
        # Cartesian 0° (East) -> Compass 90° (East)
        assert cartesian_to_compass(0.0) == 90.0
        # Cartesian 270° (South) -> Compass 180° (South)
        assert cartesian_to_compass(270.0) == 180.0
        # Cartesian 180° (West) -> Compass 270° (West)
        assert cartesian_to_compass(180.0) == 270.0

        # Roundtrip check
        for deg in [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]:
            compass = cartesian_to_compass(deg)
            cartesian = compass_to_cartesian(compass)
            assert abs(cartesian - deg) < 1e-4, f"Roundtrip failed for {deg}"


# ==============================================================================
# TEST 3: ±6° TURN DAMPING & OSCILLATION PREVENTION
# ==============================================================================
class TestTurnDamping:
    """Verifies ±6° deadband filtering and smooth turn duration interpolation."""

    def test_deadband_rejection_within_plus_minus_6_degrees(self):
        """Errors within ±6.0° must not trigger turns (in_deadband=True, straight)."""
        for err in [0.0, 1.2, -1.8, 4.5, -5.9, 6.0, -6.0]:
            res = apply_turn_damping(angle_diff=err, deadband_deg=6.0)
            assert res["in_deadband"] is True
            assert res["turn_needed"] is False
            assert res["turn_direction"] == "straight"
            assert res["turn_duration_sec"] == 0.0

    def test_turn_trigger_outside_deadband(self):
        """Errors outside ±6.0° must trigger right or left turns."""
        res_r = apply_turn_damping(angle_diff=12.0, deadband_deg=6.0)
        assert res_r["in_deadband"] is False
        assert res_r["turn_needed"] is True
        assert res_r["turn_direction"] == "right"
        assert res_r["turn_duration_sec"] > 0.0

        res_l = apply_turn_damping(angle_diff=-20.0, deadband_deg=6.0)
        assert res_l["in_deadband"] is False
        assert res_l["turn_needed"] is True
        assert res_l["turn_direction"] == "left"
        assert res_l["turn_duration_sec"] > 0.0

    def test_turn_controller_reversal_smoothing(self):
        """Verifies TurnDampingController damps rapid direction reversals."""
        ctrl = TurnDampingController(deadband_deg=6.0, reversal_cooldown_sec=0.2)
        t0 = 100.0

        # First turn: Right
        turn1 = ctrl.calculate_turn(target_bearing=45.0, player_heading=0.0, now=t0)
        assert turn1["turn_direction"] == "right"
        assert turn1["reversal_damped"] is False
        dur1 = turn1["turn_duration_sec"]

        # Rapid overshoot within 0.05s (< 0.2s): Left
        turn2 = ctrl.calculate_turn(target_bearing=0.0, player_heading=45.0, now=t0 + 0.05)
        assert turn2["turn_direction"] == "left"
        assert turn2["reversal_damped"] is True
        assert turn2["turn_duration_sec"] < dur1


# ==============================================================================
# TEST 4: ARRIVAL THRESHOLDS (<=20PX / <=3M)
# ==============================================================================
class TestArrivalThresholds:
    """Verifies waypoint arrival detection at <=20px and <=3m thresholds."""

    def test_pixel_arrival_threshold(self):
        """Distances <= 20.0px trigger arrival; > 20.0px do not."""
        detector = WaypointArrivalDetector(waypoint_radius_px=20.0, waypoint_radius_m=3.0)

        # Distance = 15.0px <= 20.0px -> arrived
        r_in = detector.is_arrival_reached(
            player_pos=(100.0, 100.0),
            waypoint_pos=(115.0, 100.0),
            distance_m=5.0,  # meter distance > 3m, but pixel <= 20px
        )
        assert r_in["arrived"] is True
        assert r_in["reached_px"] is True

        # Distance = 28.0px > 20.0px -> not arrived
        r_out = detector.is_arrival_reached(
            player_pos=(100.0, 100.0),
            waypoint_pos=(128.0, 100.0),
            distance_m=5.0,
        )
        assert r_out["arrived"] is False
        assert r_out["reached_px"] is False

    def test_meter_arrival_threshold(self):
        """Distances <= 3.0m trigger arrival; > 3.0m do not."""
        detector = WaypointArrivalDetector(waypoint_radius_px=20.0, waypoint_radius_m=3.0)

        # 2.2m <= 3.0m -> arrived
        r_m_in = detector.is_arrival_reached(
            player_pos=(100.0, 100.0),
            waypoint_pos=(150.0, 150.0),  # pixel dist ~70px > 20px
            distance_m=2.2,
        )
        assert r_m_in["arrived"] is True
        assert r_m_in["reached_m"] is True

        # 4.5m > 3.0m and 70px > 20px -> not arrived
        r_m_out = detector.is_arrival_reached(
            player_pos=(100.0, 100.0),
            waypoint_pos=(150.0, 150.0),
            distance_m=4.5,
        )
        assert r_m_out["arrived"] is False
        assert r_m_out["reached_m"] is False

    def test_transition_on_arrival(self):
        """Confirms check_and_transition triggers transition when arrived."""
        detector = WaypointArrivalDetector(waypoint_radius_px=20.0)
        nodes = [
            {"id": 1, "map_x": 50.0, "map_y": 50.0},
            {"id": 2, "map_x": 150.0, "map_y": 50.0},
        ]
        # Player at (55, 50) -> dist = 5.0px <= 20px -> transition to index 1
        t_res = detector.check_and_transition(
            player_pos=(55.0, 50.0),
            waypoint_pos=(50.0, 50.0),
            nodes=nodes,
            current_index=0,
        )
        assert t_res["arrived"] is True
        assert t_res["transitioned"] is True
        assert t_res["current_index"] == 1
        assert t_res["active_node"]["id"] == 2


# ==============================================================================
# TEST 5: NEAREST-WAYPOINT RESUME AFTER INTERRUPT
# ==============================================================================
class TestNearestWaypointResume:
    """Verifies route resumption after F10 pause or combat interrupt."""

    def test_find_nearest_waypoint(self):
        """Confirms find_nearest_waypoint correctly identifies closest node."""
        nodes = [
            {"id": 100, "map_x": 100.0, "map_y": 100.0},
            {"id": 200, "map_x": 300.0, "map_y": 300.0},
            {"id": 300, "map_x": 500.0, "map_y": 500.0},
        ]
        # Player at (290, 305) -> closest to Node 200
        idx, node, dist = find_nearest_waypoint((290.0, 305.0), nodes)
        assert idx == 1
        assert node["id"] == 200
        assert dist < 15.0

    def test_resume_route_after_interrupt_snapping(self):
        """
        After being pushed away by combat or pausing, resume_route_after_interrupt
        resumes from the nearest waypoint.

        Default (advance_if_reached=False): the player at (190, 195) is
        ~11px from Node 2, so resume targets Node 2 itself (index 1),
        not the following node.
        """
        nodes = [
            {"id": 1, "map_x": 100.0, "map_y": 100.0},
            {"id": 2, "map_x": 200.0, "map_y": 200.0},
            {"id": 3, "map_x": 300.0, "map_y": 100.0},
        ]
        # Player got pulled to (190, 195) during combat
        resume_info = resume_route_after_interrupt(
            player_pos=(190.0, 195.0),
            nodes=nodes,
            last_index=0,
            waypoint_radius_px=20.0,
        )
        assert resume_info["cyclical"] is True
        assert resume_info["resumed_index"] == 1
        assert resume_info["resumed_from"] == "nearest"
        assert resume_info["active_node"]["id"] == 2
        assert resume_info["distance_px"] <= 20.0

    def test_resume_route_after_interrupt_advance_if_reached(self):
        """
        With advance_if_reached=True and the player already within the
        waypoint radius of the nearest node, resume advances to the next
        sequential node in cyclical order (Node 2 -> Node 3).
        """
        nodes = [
            {"id": 1, "map_x": 100.0, "map_y": 100.0},
            {"id": 2, "map_x": 200.0, "map_y": 200.0},
            {"id": 3, "map_x": 300.0, "map_y": 100.0},
        ]
        resume_info = resume_route_after_interrupt(
            player_pos=(190.0, 195.0),
            nodes=nodes,
            last_index=0,
            waypoint_radius_px=20.0,
            advance_if_reached=True,
        )
        assert resume_info["cyclical"] is True
        assert resume_info["resumed_index"] == 2
        assert resume_info["resumed_from"] == "advance_from_nearest"
        assert resume_info["active_node"]["id"] == 3

    def test_planner_resume_after_interrupt(self):
        """Tests SequentialRoutePlanner.resume_after_interrupt updates internal seq pos."""
        nodes = [
            {"id": 1, "map_x": 10.0, "map_y": 10.0},
            {"id": 2, "map_x": 500.0, "map_y": 500.0},
            {"id": 3, "map_x": 900.0, "map_y": 900.0},
        ]
        planner = SequentialRoutePlanner()
        node, idx = planner.resume_after_interrupt(nodes, player_pos=(490.0, 505.0))
        assert idx == 1
        assert node["id"] == 2
        assert planner._current_seq_pos == 1


# ==============================================================================
# TEST 6: SEQUENTIAL ROUTE PLANNER & MAP NAVIGATOR INTEROPERABILITY
# ==============================================================================
class TestPlannerAndNavigatorInteroperability:
    """Verifies that SequentialRoutePlanner and MapNavigator seamlessly interoperate."""

    def test_planner_steer_step_integration(self):
        """Confirms planner.steer_step returns bearing, heading, and turn damping."""
        planner = SequentialRoutePlanner()
        step_res = planner.steer_step(
            player_pos=(100.0, 100.0),
            target_pos=(100.0, 50.0),  # North -> 0.0°
            player_heading=0.0,         # Aligned -> in deadband
        )
        assert step_res["target_bearing"] == 0.0
        assert step_res["player_heading"] == 0.0
        assert step_res["in_deadband"] is True
        assert step_res["turn_needed"] is False

    def test_planner_calculate_route_metrics(self):
        """Confirms calculate_route_metrics returns full suite of standardized metrics.

        Uses a genuinely closed route (rectangle returning to the start
        edge): 300 + 400 + 300 + 400 = 1400px. An open L-shaped route must
        NOT be asserted as closed here — production still reports
        closed_loop for any route with >= 3 nodes (see
        SequentialRoutePlanner.calculate_route_metrics in
        core/sequential_router.py); fixing that heuristic is outside this
        test slice.
        """
        planner = SequentialRoutePlanner()
        pts = [
            {"map_x": 0.0, "map_y": 0.0},
            {"map_x": 300.0, "map_y": 0.0},
            {"map_x": 300.0, "map_y": 400.0},
            {"map_x": 0.0, "map_y": 400.0},
        ]
        metrics = planner.calculate_route_metrics(pts)
        assert metrics["node_count"] == 4
        assert metrics["discovered_nodes"] == 4
        assert metrics["closed_loop"] is True
        assert abs(metrics["total_distance_px"] - 1400.0) < 1.0
        assert 0.0 <= metrics["smoothness_score"] <= 100.0

    def test_advance_cyclical_sequence_on_planner(self):
        """Verifies planner.advance_cyclical_sequence performs atomic arrival check & step."""
        planner = SequentialRoutePlanner()
        nodes = [
            {"id": 1, "map_x": 100.0, "map_y": 100.0},
            {"id": 2, "map_x": 200.0, "map_y": 100.0},
            {"id": 3, "map_x": 200.0, "map_y": 200.0},
        ]
        # Arrived at Node 1 (player at 102, 100 -> dist 2px <= 20px)
        adv = planner.advance_cyclical_sequence(nodes, current_index=0, player_pos=(102.0, 100.0))
        assert adv["arrived"] is True
        assert adv["transitioned"] is True
        assert adv["current_index"] == 1
        assert planner._current_seq_pos == 1
