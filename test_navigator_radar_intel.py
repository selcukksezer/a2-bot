# -*- coding: utf-8 -*-
"""
Unit tests for Navigator, Corridor Navigator, and Navigation Intel:
1. Heading vs Bearing Alignment & 180° Inversion Fix
2. Turn Damping & Oscillation Prevention (Deadband ±5°..±8° & Smooth Interpolation)
3. Arrival & Stuck Detection with Radar Intel (Waypoint Radius <=20px/<=3m & Stuck >3.0s)
"""

import math
import pytest
from typing import List, Tuple

import core.navigator as nav
import core.corridor_navigator as cor_nav
import core.navigation_intel as nav_intel


# ==============================================================================
# TASK 1: HEADING VS BEARING ALIGNMENT & 180° INVERSION FIX
# ==============================================================================
class TestHeadingBearingAlignment:
    """Tests compass bearing calculation and shortest rotational path."""

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_compass_bearing_cardinal_directions(self, module):
        """
        Confirms bearing follows compass convention:
        North is 0°, East is 90°, South is 180°, West is 270°.
        Eliminates the 180° inversion bug between Cartesian (0° = East) and
        Minimap arrow heading (0° = North).
        """
        origin = (100.0, 100.0)

        # North (-Y in screen/minimap coords) -> 0.0°
        bearing_north = module.compute_bearing(origin, (100.0, 50.0))
        assert bearing_north == 0.0, f"North must be 0°, got {bearing_north}"

        # North-East (+X, -Y) -> 45.0°
        bearing_ne = module.compute_bearing(origin, (150.0, 50.0))
        assert abs(bearing_ne - 45.0) < 0.01, f"North-East must be 45°, got {bearing_ne}"

        # East (+X, 0) -> 90.0°
        bearing_east = module.compute_bearing(origin, (150.0, 100.0))
        assert bearing_east == 90.0, f"East must be 90°, got {bearing_east}"

        # South-East (+X, +Y) -> 135.0°
        bearing_se = module.compute_bearing(origin, (150.0, 150.0))
        assert abs(bearing_se - 135.0) < 0.01, f"South-East must be 135°, got {bearing_se}"

        # South (0, +Y) -> 180.0°
        bearing_south = module.compute_bearing(origin, (100.0, 150.0))
        assert bearing_south == 180.0, f"South must be 180°, got {bearing_south}"

        # South-West (-X, +Y) -> 225.0°
        bearing_sw = module.compute_bearing(origin, (50.0, 150.0))
        assert abs(bearing_sw - 225.0) < 0.01, f"South-West must be 225°, got {bearing_sw}"

        # West (-X, 0) -> 270.0°
        bearing_west = module.compute_bearing(origin, (50.0, 100.0))
        assert bearing_west == 270.0, f"West must be 270°, got {bearing_west}"

        # North-West (-X, -Y) -> 315.0°
        bearing_nw = module.compute_bearing(origin, (50.0, 50.0))
        assert abs(bearing_nw - 315.0) < 0.01, f"North-West must be 315°, got {bearing_nw}"

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_shortest_rotational_path(self, module):
        """
        Ensures angle_diff = (target_bearing - player_heading + 180) % 360 - 180
        always chooses the shortest rotational path in range [-180°, +180°].
        """
        # Crossing the 0°/360° North boundary from left: 350° heading to 10° target
        # Should turn right by +20°, NOT left by -340°
        diff1 = module.compute_angle_diff(target_bearing=10.0, player_heading=350.0)
        assert diff1 == 20.0, f"Expected +20.0°, got {diff1}"

        # Crossing the 0°/360° North boundary from right: 10° heading to 350° target
        # Should turn left by -20°, NOT right by +340°
        diff2 = module.compute_angle_diff(target_bearing=350.0, player_heading=10.0)
        assert diff2 == -20.0, f"Expected -20.0°, got {diff2}"

        # Heading 90° (East) to target 180° (South) -> +90° (right)
        diff3 = module.compute_angle_diff(target_bearing=180.0, player_heading=90.0)
        assert diff3 == 90.0, f"Expected +90.0°, got {diff3}"

        # Heading 180° (South) to target 90° (East) -> -90° (left)
        diff4 = module.compute_angle_diff(target_bearing=90.0, player_heading=180.0)
        assert diff4 == -90.0, f"Expected -90.0°, got {diff4}"

        # Exact opposite (180° diff)
        diff5 = module.compute_angle_diff(target_bearing=180.0, player_heading=0.0)
        assert abs(diff5) == 180.0


# ==============================================================================
# TASK 2: TURN DAMPING & OSCILLATION PREVENTION
# ==============================================================================
class TestTurnDampingAndOscillation:
    """Tests deadband filtering (±5°..±8°) and smooth turn duration interpolation."""

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_deadband_zone_prevents_micro_turns(self, module):
        """
        Confirms deadband zone (default 6.0°, within ±5° to ±8°) rejects micro-turns
        to prevent left-right rapid oscillation near target bearing.
        """
        # Angle errors within deadband (±5° to ±6°)
        for err in [0.0, 2.5, -3.0, 5.0, -5.5, 6.0, -6.0]:
            res = module.apply_turn_damping(angle_diff=err, deadband_deg=6.0)
            assert res["in_deadband"] is True, f"Error {err} should be inside deadband"
            assert res["turn_needed"] is False, f"Error {err} must not trigger turn"
            assert res["turn_duration_sec"] == 0.0
            assert res["turn_direction"] == "straight"

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_turn_triggers_outside_deadband(self, module):
        """
        Confirms angular errors exceeding deadband trigger turn with correct direction.
        """
        # Right turn (> +6.0°)
        res_right = module.apply_turn_damping(angle_diff=15.0, deadband_deg=6.0)
        assert res_right["in_deadband"] is False
        assert res_right["turn_needed"] is True
        assert res_right["turn_direction"] == "right"
        assert res_right["turn_duration_sec"] > 0.0

        # Left turn (< -6.0°)
        res_left = module.apply_turn_damping(angle_diff=-25.0, deadband_deg=6.0)
        assert res_left["in_deadband"] is False
        assert res_left["turn_needed"] is True
        assert res_left["turn_direction"] == "left"
        assert res_left["turn_duration_sec"] > 0.0

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_smooth_turn_duration_interpolation(self, module):
        """
        Ensures turn duration smoothly scales with angular error:
        duration(8°) < duration(25°) < duration(60°) < duration(120°) < duration(180°)
        """
        d8 = module.apply_turn_damping(8.0, deadband_deg=6.0)["turn_duration_sec"]
        d25 = module.apply_turn_damping(25.0, deadband_deg=6.0)["turn_duration_sec"]
        d60 = module.apply_turn_damping(60.0, deadband_deg=6.0)["turn_duration_sec"]
        d120 = module.apply_turn_damping(120.0, deadband_deg=6.0)["turn_duration_sec"]
        d180 = module.apply_turn_damping(180.0, deadband_deg=6.0)["turn_duration_sec"]

        assert 0.03 <= d8 < d25 < d60 < d120 < d180 <= 0.40, (
            f"Durations must be monotonically increasing: {d8}, {d25}, {d60}, {d120}, {d180}"
        )

    @pytest.mark.parametrize(
        "module",
        [nav, cor_nav, nav_intel],
        ids=["navigator", "corridor_navigator", "navigation_intel"],
    )
    def test_turn_damping_controller_reversal_suppression(self, module):
        """
        Confirms TurnDampingController suppresses rapid direction reversal chattering.
        """
        controller = module.TurnDampingController(deadband_deg=6.0, reversal_cooldown_sec=0.2)
        t0 = 1000.0

        # Turn right at t0
        r1 = controller.calculate_turn(target_bearing=45.0, player_heading=0.0, now=t0)
        assert r1["turn_direction"] == "right"
        assert r1["reversal_damped"] is False
        dur_initial = r1["turn_duration_sec"]

        # Rapid overshoot reversal to left within 0.05s (< 0.2s cooldown)
        r2 = controller.calculate_turn(target_bearing=0.0, player_heading=45.0, now=t0 + 0.05)
        assert r2["turn_direction"] == "left"
        assert r2["reversal_damped"] is True
        assert r2["turn_duration_sec"] < dur_initial, "Reversal duration must be damped"


# ==============================================================================
# TASK 3: ARRIVAL & STUCK DETECTION WITH RADAR INTEL
# ==============================================================================
class TestRadarIntelArrivalAndStuck:
    """Tests waypoint arrival detection (<=20px / <=3m) and stuck detection (>3.0s)."""

    def test_waypoint_arrival_pixel_radius(self):
        """
        Confirms waypoint arrival is detected when player is within 20px radius.
        """
        detector = nav_intel.WaypointArrivalDetector(waypoint_radius_px=20.0, waypoint_radius_m=3.0)

        # Inside radius (distance = 10px <= 20px)
        res_in = detector.is_arrival_reached(
            player_pos=(100.0, 100.0),
            waypoint_pos=(108.0, 106.0),  # dist = 10.0px
        )
        assert res_in["arrived"] is True
        assert res_in["reached_px"] is True

        # Outside radius (distance = 30px > 20px, and distance_m > 3m)
        res_out = detector.is_arrival_reached(
            player_pos=(100.0, 100.0),
            waypoint_pos=(124.0, 118.0),  # dist = 30.0px -> 16.5m
        )
        assert res_out["arrived"] is False

    def test_waypoint_arrival_meter_radius(self):
        """
        Confirms waypoint arrival is detected when world distance <= 3.0m.
        """
        detector = nav_intel.WaypointArrivalDetector(waypoint_radius_px=20.0, waypoint_radius_m=3.0)

        # 2.5m <= 3.0m -> arrived
        res_in = detector.is_arrival_reached(
            player_pos=(0, 0),
            waypoint_pos=(50, 50),
            distance_m=2.5,
        )
        assert res_in["arrived"] is True
        assert res_in["reached_m"] is True

        # 4.2m > 3.0m -> not arrived
        res_out = detector.is_arrival_reached(
            player_pos=(0, 0),
            waypoint_pos=(50, 50),
            distance_m=4.2,
            pixel_distance=35.0,
        )
        assert res_out["arrived"] is False

    def test_clean_sequential_node_transition(self):
        """
        Confirms arrival at waypoint triggers clean transition to next sequential node.
        """
        nodes = [
            {"id": 101, "name": "Node A", "map_x": 100, "map_y": 100},
            {"id": 102, "name": "Node B", "map_x": 200, "map_y": 200},
            {"id": 103, "name": "Node C", "map_x": 300, "map_y": 300},
        ]
        detector = nav_intel.WaypointArrivalDetector(waypoint_radius_px=20.0)

        # Player at (105, 105) approaches Node A at (100, 100) -> distance ~7.07px <= 20px
        step1 = detector.check_and_transition(
            player_pos=(105.0, 105.0),
            waypoint_pos=(100.0, 100.0),
            nodes=nodes,
            current_index=0,
        )
        assert step1["arrived"] is True
        assert step1["transitioned"] is True
        assert step1["current_index"] == 1
        assert step1["active_node"]["id"] == 102

        # Approaching Node B but still far away (distance = 80px > 20px)
        step2 = detector.check_and_transition(
            player_pos=(120.0, 120.0),
            waypoint_pos=(200.0, 200.0),
            nodes=nodes,
            current_index=1,
        )
        assert step2["arrived"] is False
        assert step2["transitioned"] is False
        assert step2["current_index"] == 1
        assert step2["active_node"]["id"] == 102

    def test_stuck_detection_timeout_greater_than_3_seconds(self):
        """
        Confirms stuck is triggered ONLY when player position does not change
        for > 3.0s despite movement keys being held.
        """
        stuck_det = nav_intel.RadarStuckDetector(stuck_timeout_sec=3.0, movement_threshold_px=2.0)
        t0 = 500.0

        # t=0s: initial position
        s0 = stuck_det.update_position((100.0, 100.0), movement_keys_held=True, now=t0)
        assert s0["is_stuck"] is False

        # t=1.5s: position barely changed (< 2.0px threshold)
        s1 = stuck_det.update_position((100.8, 100.6), movement_keys_held=True, now=t0 + 1.5)
        assert s1["is_stuck"] is False
        assert s1["stuck_elapsed_sec"] >= 1.5

        # t=2.9s: still stationary (< 3.0s)
        s2 = stuck_det.update_position((100.9, 100.7), movement_keys_held=True, now=t0 + 2.9)
        assert s2["is_stuck"] is False

        # t=3.2s: position unchanged for > 3.0s while movement keys held -> STUCK!
        s3 = stuck_det.update_position((101.0, 100.8), movement_keys_held=True, now=t0 + 3.2)
        assert s3["is_stuck"] is True
        assert s3["stuck_elapsed_sec"] >= 3.0

    def test_stuck_not_triggered_when_player_moving(self):
        """
        Confirms stuck is NOT triggered when player is making forward progress (> threshold).
        """
        stuck_det = nav_intel.RadarStuckDetector(stuck_timeout_sec=3.0, movement_threshold_px=2.0)
        t0 = 500.0

        # Player walking forward: position updates by ~10px every second
        for i in range(6):
            res = stuck_det.update_position(
                player_pos=(100.0 + i * 10.0, 100.0),
                movement_keys_held=True,
                now=t0 + i * 1.0,
            )
            assert res["is_stuck"] is False, f"Step {i} falsely triggered stuck"

    def test_stuck_not_triggered_when_idle(self):
        """
        Confirms stuck is NOT triggered when player is intentionally idle (movement_keys_held=False).
        """
        stuck_det = nav_intel.RadarStuckDetector(stuck_timeout_sec=3.0)
        t0 = 500.0

        stuck_det.update_position((100.0, 100.0), movement_keys_held=False, now=t0)
        res = stuck_det.update_position((100.0, 100.0), movement_keys_held=False, now=t0 + 5.0)
        assert res["is_stuck"] is False

    def test_unstick_maneuver_execution(self):
        """
        Confirms unstick maneuver executes SPACE -> A+W/D+W -> REALIGN and resets stuck timer.
        """
        class MockInputController:
            def __init__(self):
                self.tapped_keys = []
                self.held_keys = []
                self.realigned_angles = []

            def tap_key(self, k):
                self.tapped_keys.append(k)

            def hold_keys(self, keys, duration=1.0):
                self.held_keys.append((list(keys), duration))

            def turn_by_angle(self, angle, duration=None):
                self.realigned_angles.append((angle, duration))

        mock_input = MockInputController()
        stuck_det = nav_intel.RadarStuckDetector(stuck_timeout_sec=3.0, strafe_duration_sec=0.5)

        # Trigger stuck
        t0 = 100.0
        stuck_det.update_position((50.0, 50.0), movement_keys_held=True, now=t0)
        stuck_det.update_position((50.0, 50.0), movement_keys_held=True, now=t0 + 3.5)
        assert stuck_det.is_stuck is True

        # Execute unstick #1 (should use A+W)
        m1 = stuck_det.execute_unstick_maneuver(
            input_controller=mock_input,
            target_bearing=90.0,
            player_heading=0.0,
        )
        assert m1["executed"] is True
        assert m1["strafe_key"] == "a"
        assert m1["sequence"] == ["SPACE", "A+W", "REALIGN"]
        assert "space" in mock_input.tapped_keys
        assert mock_input.held_keys[0][0] == ["a", "w"]
        assert stuck_det.is_stuck is False

        # Trigger stuck again and execute unstick #2 (should alternate to D+W)
        stuck_det.update_position((50.0, 50.0), movement_keys_held=True, now=t0 + 4.0)
        stuck_det.update_position((50.0, 50.0), movement_keys_held=True, now=t0 + 7.5)
        assert stuck_det.is_stuck is True

        m2 = stuck_det.execute_unstick_maneuver(
            input_controller=mock_input,
            target_bearing=180.0,
            player_heading=90.0,
        )
        assert m2["strafe_key"] == "d"
        assert m2["sequence"] == ["SPACE", "D+W", "REALIGN"]
        assert mock_input.held_keys[1][0] == ["d", "w"]


# ==============================================================================
# INTEGRATION: NAVIGATION INTEL COORDINATION
# ==============================================================================
class TestNavigationIntelIntegration:
    """Tests the unified NavigationIntel coordinator class."""

    def test_navigation_intel_full_cycle(self):
        """
        Verifies NavigationIntel.step_navigation_cycle:
        - Stuck monitoring
        - Waypoint arrival and transition
        - Bearing and turn damping
        """
        intel = nav_intel.NavigationIntel(
            waypoint_radius_px=20.0,
            waypoint_radius_m=3.0,
            stuck_timeout_sec=3.0,
            deadband_deg=6.0,
        )

        nodes = [
            {"id": 1, "map_x": 100, "map_y": 50},  # North of (100, 100) -> bearing 0°
            {"id": 2, "map_x": 150, "map_y": 100}, # East of (100, 100) -> bearing 90°
        ]

        # Player at (100, 100), heading 0° (North), target Node 1 at (100, 50)
        cycle1 = intel.step_navigation_cycle(
            player_pos=(100.0, 100.0),
            player_heading=0.0,
            waypoint_pos=(100.0, 50.0),
            nodes=nodes,
            current_index=0,
            movement_keys_held=True,
            now=100.0,
        )
        assert cycle1["stuck_detected"] is False
        assert cycle1["target_bearing"] == 0.0
        assert cycle1["turn_data"]["in_deadband"] is True  # 0° vs 0° -> in deadband (no turn)
        assert cycle1["arrival_info"]["arrived"] is False

        # Player arrives within 15px of Node 1
        cycle2 = intel.step_navigation_cycle(
            player_pos=(100.0, 65.0),
            player_heading=0.0,
            waypoint_pos=(100.0, 50.0),  # dist = 15px <= 20px
            nodes=nodes,
            current_index=0,
            movement_keys_held=True,
            now=101.5,
        )
        assert cycle2["arrival_info"]["arrived"] is True
        assert cycle2["arrival_info"]["transitioned"] is True
        assert cycle2["arrival_info"]["current_index"] == 1
        assert cycle2["arrival_info"]["active_node"]["id"] == 2
