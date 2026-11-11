# -*- coding: utf-8 -*-
"""
W55 Unit Test Suite:
Verifies Navigator compass bearing (North=0°, East=90°) and TurnDampingController
turn damping calculations do not exhibit 180° inversion bugs.
"""

import os
import sys
from unittest.mock import MagicMock

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.navigator import (
    CyclicalRouteFollower,
    MapNavigator,
    TurnDampingController,
    anti_inversion_compass_heading,
    apply_turn_damping,
    cartesian_to_compass,
    compass_to_cartesian,
    compute_angle_diff,
    compute_bearing,
)


def _create_navigator() -> MapNavigator:
    return MapNavigator(
        route_manager=MagicMock(),
        vision=MagicMock(),
        driver=MagicMock(),
    )


def test_compass_bearing_north_0_east_90_no_180_inversion():
    """
    Test 1: Verifies Navigator compass bearing convention (North=0°, East=90°,
    South=180°, West=270°) in screen/minimap coordinates without 180° inversion.
    """
    nav = _create_navigator()
    origin = (500.0, 500.0)

    # Cardinal directions: North (dy < 0) = 0°, East (dx > 0) = 90°, South (dy > 0) = 180°, West (dx < 0) = 270°
    north_bearing = nav.compute_bearing(origin, (500.0, 350.0))
    east_bearing = nav.compute_bearing(origin, (650.0, 500.0))
    south_bearing = nav.compute_bearing(origin, (500.0, 650.0))
    west_bearing = nav.compute_bearing(origin, (350.0, 500.0))

    assert north_bearing == 0.0
    assert east_bearing == 90.0
    assert south_bearing == 180.0
    assert west_bearing == 270.0

    # Verify non-inversion: North is never 180° and East is never 270°
    assert north_bearing != 180.0
    assert east_bearing != 270.0

    # Intercardinal directions: NE=45°, SE=135°, SW=225°, NW=315°
    assert compute_bearing(origin, (600.0, 400.0)) == 45.0
    assert compute_bearing(origin, (600.0, 600.0)) == 135.0
    assert compute_bearing(origin, (400.0, 600.0)) == 225.0
    assert compute_bearing(origin, (400.0, 400.0)) == 315.0


def test_anti_inversion_compass_heading_and_cartesian_conversion():
    """
    Test 2: Verifies 180° anti-inversion heading correction and Cartesian <-> Compass
    conversions (Cartesian 90° -> Compass 0° North; Cartesian 0° -> Compass 90° East).
    """
    assert cartesian_to_compass(90.0) == 0.0
    assert cartesian_to_compass(0.0) == 90.0
    assert cartesian_to_compass(270.0) == 180.0
    assert cartesian_to_compass(180.0) == 270.0

    assert compass_to_cartesian(0.0) == 90.0
    assert compass_to_cartesian(90.0) == 0.0

    # When arrow heading is inverted by 180° relative to true forward travel bearing:
    # Target is North (0°), but raw arrow reading is 180° (South) -> corrected to 0°
    assert anti_inversion_compass_heading(180.0, reference_bearing=0.0, threshold_deg=150.0) == 0.0
    # Target is East (90°), but raw arrow reading is 270° (West) -> corrected to 90°
    assert anti_inversion_compass_heading(270.0, reference_bearing=90.0, threshold_deg=150.0) == 90.0

    # Normal forward headings within threshold must NOT be inverted
    assert anti_inversion_compass_heading(12.0, reference_bearing=0.0, threshold_deg=150.0) == 12.0
    assert anti_inversion_compass_heading(82.0, reference_bearing=90.0, threshold_deg=150.0) == 82.0


def test_navigator_steer_step_shortest_angle_without_180_flip():
    """
    Test 3: Verifies MapNavigator.steer_step and compute_angle_diff compute the
    shortest rotational turn in [-180°, +180°] without 180° reverse steering.
    """
    nav = _create_navigator()
    current_pos = (100.0, 100.0)

    # Target is due East (90°), player faces North (0°) -> should turn right (+90°), never left (-90°)
    step_east = nav.steer_step(
        current_pos=current_pos,
        target_pos=(200.0, 100.0),
        player_heading=0.0,
        deadband_deg=6.0,
    )
    assert step_east["target_bearing"] == 90.0
    assert step_east["angle_diff"] == 90.0
    assert step_east["turn_needed"] is True
    assert step_east["turn_direction"] == "right"

    # Target is due North (0°), player faces East (90°) -> should turn left (-90°), never right (+90°)
    step_north = nav.steer_step(
        current_pos=current_pos,
        target_pos=(100.0, 0.0),
        player_heading=90.0,
        deadband_deg=6.0,
    )
    assert step_north["target_bearing"] == 0.0
    assert step_north["angle_diff"] == -90.0
    assert step_north["turn_needed"] is True
    assert step_north["turn_direction"] == "left"

    # Wrapping across 0°/360° North boundary: 350° -> 10° is +20° (right), 10° -> 350° is -20° (left)
    assert compute_angle_diff(target_bearing=10.0, player_heading=350.0) == 20.0
    assert compute_angle_diff(target_bearing=350.0, player_heading=10.0) == -20.0


def test_turn_damping_controller_deadband_and_monotonic_scaling():
    """
    Test 4: Verifies TurnDampingController deadband suppression (±6°) and
    smooth monotonic turn duration scaling without sign or 180° inversion.
    """
    controller = TurnDampingController(deadband_deg=6.0, min_turn_sec=0.04, max_turn_sec=0.35)

    # Inside deadband around North (0°) and East (90°)
    in_db_north = controller.calculate_turn(target_bearing=0.0, player_heading=356.0, now=10.0)
    assert in_db_north["in_deadband"] is True
    assert in_db_north["turn_needed"] is False
    assert in_db_north["turn_direction"] == "straight"
    assert in_db_north["turn_duration_sec"] == 0.0

    in_db_east = controller.calculate_turn(target_bearing=90.0, player_heading=95.5, now=11.0)
    assert in_db_east["in_deadband"] is True
    assert in_db_east["turn_needed"] is False
    assert in_db_east["turn_direction"] == "straight"
    assert in_db_east["turn_duration_sec"] == 0.0

    # Outside deadband: duration increases smoothly as angular error grows from 15° to 90° to 170°
    d_small = apply_turn_damping(angle_diff=15.0, deadband_deg=6.0)
    d_mid = apply_turn_damping(angle_diff=90.0, deadband_deg=6.0)
    d_large = apply_turn_damping(angle_diff=170.0, deadband_deg=6.0)

    assert d_small["turn_direction"] == "right"
    assert d_mid["turn_direction"] == "right"
    assert d_large["turn_direction"] == "right"
    assert 0.04 <= d_small["turn_duration_sec"] < d_mid["turn_duration_sec"] < d_large["turn_duration_sec"] <= 0.35


def test_turn_damping_controller_reversal_damping_and_follower_integration():
    """
    Test 5: Verifies TurnDampingController damps rapid left-right direction reversals
    within reversal_cooldown_sec and integrates cleanly with CyclicalRouteFollower.
    """
    controller = TurnDampingController(
        deadband_deg=6.0,
        min_turn_sec=0.04,
        max_turn_sec=0.35,
        reversal_cooldown_sec=0.15,
    )

    # Step 1: Turn right toward East (90°) from North (0°) at t=50.0
    first_turn = controller.calculate_turn(target_bearing=90.0, player_heading=0.0, now=50.0)
    assert first_turn["turn_direction"] == "right"
    assert first_turn["reversal_damped"] is False
    base_duration = first_turn["turn_duration_sec"]

    # Step 2: Immediate overshoot reversal to left within cooldown (t=50.05, dt=0.05 < 0.15)
    reversed_turn = controller.calculate_turn(target_bearing=0.0, player_heading=90.0, now=50.05)
    assert reversed_turn["turn_direction"] == "left"
    assert reversed_turn["reversal_damped"] is True
    assert reversed_turn["turn_duration_sec"] == round(base_duration * 0.65, 4)

    # Step 3: After cooldown expires (t=51.0, dt=0.95 >= 0.15), full turn duration is restored
    post_cooldown_turn = controller.calculate_turn(target_bearing=90.0, player_heading=0.0, now=51.0)
    assert post_cooldown_turn["turn_direction"] == "right"
    assert post_cooldown_turn["reversal_damped"] is False
    assert post_cooldown_turn["turn_duration_sec"] == base_duration

    # Step 4: CyclicalRouteFollower integration check (moving North then East without 180° flip)
    nodes = [
        {"id": 1, "map_x": 100.0, "map_y": 50.0},   # Due North of (100, 100) -> 0°
        {"id": 2, "map_x": 200.0, "map_y": 50.0},   # Due East of (100, 50) -> 90°
        {"id": 3, "map_x": 200.0, "map_y": 150.0},  # Due South of (200, 50) -> 180°
    ]
    follower = CyclicalRouteFollower(nodes=nodes, waypoint_radius_px=20.0, deadband_deg=6.0)
    state_north = follower.step(player_pos=(100.0, 100.0), player_heading=0.0, now=60.0)
    assert state_north["current_index"] == 0
    assert state_north["active_node"]["id"] == 1
    assert state_north["arrived"] is False
    assert state_north["transitioned"] is False
    assert state_north["closed_loop"] is True
    assert state_north["target_bearing"] == 0.0
    assert state_north["angle_diff"] == 0.0
    assert state_north["in_deadband"] is True
    assert state_north["turn_needed"] is False

    # Step 5: Player arrives at node 1 (100, 50) -> advances to node 2 (East, 90°)
    state_east = follower.step(player_pos=(100.0, 50.0), player_heading=0.0, now=61.0)
    assert state_east["arrived"] is True
    assert state_east["transitioned"] is True
    assert state_east["current_index"] == 1
    assert state_east["active_node"]["id"] == 2
    assert state_east["target_bearing"] == 90.0
    assert state_east["angle_diff"] == 90.0
    assert state_east["in_deadband"] is False
    assert state_east["turn_needed"] is True
    assert state_east["turn_direction"] == "right"

    # Step 6: Player arrives at node 2 (200, 50) -> advances to node 3 (South, 180°)
    state_south = follower.step(player_pos=(200.0, 50.0), player_heading=90.0, now=62.0)
    assert state_south["arrived"] is True
    assert state_south["transitioned"] is True
    assert state_south["current_index"] == 2
    assert state_south["active_node"]["id"] == 3
    assert state_south["target_bearing"] == 180.0
    assert state_south["angle_diff"] == 90.0
    assert state_south["in_deadband"] is False
    assert state_south["turn_needed"] is True
    assert state_south["turn_direction"] == "right"

    # Step 7: Player arrives at node 3 (200, 150) -> wraps cyclically to node 1,
    # completing one lap; bearing back to node 1 is NW (315°), +135° right turn.
    state_wrap = follower.step(player_pos=(200.0, 150.0), player_heading=180.0, now=63.0)
    assert state_wrap["arrived"] is True
    assert state_wrap["transitioned"] is True
    assert state_wrap["current_index"] == 0
    assert state_wrap["active_node"]["id"] == 1
    assert state_wrap["total_completed_laps"] == 1
    assert state_wrap["target_bearing"] == 315.0
    assert state_wrap["angle_diff"] == 135.0
    assert state_wrap["in_deadband"] is False
    assert state_wrap["turn_needed"] is True
    assert state_wrap["turn_direction"] == "right"
