# -*- coding: utf-8 -*-
"""
Agent G7 Comprehensive Test Suite:
1. 3D Obstacle & Monster Avoidance in Travel Corridor (ExemplarMemory & live detections)
2. Clear Path Terrain Steering (prioritizing path_ground / clear lanes)
3. Optical Flow & Minimap Delta Stuck Detection (< 2.5s timeout)
4. Physical Unstick Recovery Maneuvers:
   - Jump (SPACE)
   - Lateral Strafe (A+W and D+W)
   - Reverse-Backout (STOP_W -> REVERSE_S -> REVERSE_STRAFE -> SPACE -> REALIGN)
   - Escalating Multi-Tier Recovery
   - Repath Request on persistent blockage
5. Full CorridorNavigator step_navigation integration
"""

import math
import cv2
import numpy as np
import pytest

from core.corridor_navigator import (
    CorridorNavigator,
    PhaseCorrelateUnstuckManeuver,
    CorridorStuckDetector,
    compute_bearing,
    compute_angle_diff,
    apply_turn_damping,
)
from dataset_factory.exemplar_memory import get_exemplar_memory


class MockInputController:
    """Mock hardware key driver for validating unstick sequences."""
    def __init__(self):
        self.tapped_keys = []
        self.pressed_keys = []
        self.released_keys = []
        self.held_keys = []
        self.turned_angles = []
        self.realigned_targets = []

    def tap_key(self, key):
        self.tapped_keys.append(str(key).lower())

    def press_key(self, key):
        self.pressed_keys.append(str(key).lower())

    def release_key(self, key):
        self.released_keys.append(str(key).lower())

    def key_down(self, key):
        self.pressed_keys.append(str(key).lower())

    def key_up(self, key):
        self.released_keys.append(str(key).lower())

    def hold_keys(self, keys, duration=1.0):
        self.held_keys.append(([str(k).lower() for k in keys], float(duration)))

    def turn_by_angle(self, angle, duration=None):
        self.turned_angles.append((float(angle), duration))

    def realign_to_target(self, target_bearing):
        if target_bearing is not None:
            self.realigned_targets.append(float(target_bearing))


class TestCorridorObstacleAndPathAvoidance:
    """Tests 3D obstacle, monster avoidance, and clear path terrain steering."""

    def test_center_obstacle_steers_to_clear_lane(self):
        """Obstacle in center lane causes steering_bias='steer_left' or 'steer_right'."""
        nav = CorridorNavigator()
        mem = get_exemplar_memory()

        # Learn obstacle patch (class 14)
        obs_patch = np.zeros((60, 60, 3), dtype=np.uint8)
        obs_patch[:, :] = (30, 35, 45)  # dark rock color
        mem.add_positive_exemplar(obs_patch, class_id=14, class_name="obstacle")

        # Base frame is outdoor green ground
        frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)
        y1, y2 = int(720 * 0.52), int(720 * 0.82)
        cx1, cx2 = int(1280 * 0.44), int(1280 * 0.56)
        frame[y1:y2, cx1:cx2] = (30, 35, 45)

        eval_res = nav.evaluate_corridor_path(frame)
        assert eval_res["obstacle_detected"] is True
        assert eval_res["steering_bias"] in ("steer_left", "steer_right")
        assert eval_res["recommended_turn_offset_deg"] != 0.0
        assert eval_res["lane_scores"]["center"] < eval_res["lane_scores"]["left"]

    def test_monster_in_corridor_steers_away(self):
        """Monster (class 10) in travel corridor triggers monster_detected and steers to clear side."""
        nav = CorridorNavigator()
        mem = get_exemplar_memory()

        # Learn monster patch (class 10)
        mob_patch = np.zeros((60, 60, 3), dtype=np.uint8)
        mob_patch[:, :] = (15, 20, 180)  # red hostile mob
        mem.add_positive_exemplar(mob_patch, class_id=10, class_name="monster_3d")

        # Base frame is outdoor green ground
        frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)
        y1, y2 = int(720 * 0.52), int(720 * 0.82)
        cx1, cx2 = int(1280 * 0.44), int(1280 * 0.56)
        frame[y1:y2, cx1:cx2] = (15, 20, 180)

        eval_res = nav.evaluate_corridor_path(frame)
        assert eval_res["monster_detected"] is True
        assert eval_res["steering_bias"] in ("steer_left", "steer_right")
        assert abs(eval_res["recommended_turn_offset_deg"]) >= 15.0

    def test_path_ground_lane_prioritized(self):
        """Walkable ground path (class 15) increases lane score and draws steering."""
        nav = CorridorNavigator()
        mem = get_exemplar_memory()

        path_patch = np.zeros((60, 60, 3), dtype=np.uint8)
        path_patch[:, :] = (90, 140, 120)  # dirt path
        mem.add_positive_exemplar(path_patch, class_id=15, class_name="path_ground")

        # Base frame is outdoor green ground
        frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)
        y1, y2 = int(720 * 0.52), int(720 * 0.82)
        # Place path in right lane
        rx1, rx2 = int(1280 * 0.56), int(1280 * 0.72)
        frame[y1:y2, rx1:rx2] = (90, 140, 120)

        eval_res = nav.evaluate_corridor_path(frame)
        assert eval_res["path_detected"] is True
        assert eval_res["lane_scores"]["right"] > eval_res["lane_scores"]["left"]
        assert eval_res["steering_bias"] == "steer_right"
        assert eval_res["recommended_turn_offset_deg"] > 0.0

    def test_live_yolo_detections_steer_away(self):
        """Live YOLO detection of obstacle or monster steers vehicle away."""
        nav = CorridorNavigator()
        frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)

        # Obstacle in center lane (x: 0.50, y: 0.65)
        yolo_dets = [
            {
                "class_id": 14,
                "class_name": "obstacle",
                "confidence": 0.92,
                "bbox": [0.46, 0.58, 0.54, 0.72],
            }
        ]

        eval_res = nav.evaluate_corridor_path(frame, yolo_detections=yolo_dets)
        assert eval_res["obstacle_detected"] is True
        assert eval_res["steering_bias"] in ("steer_left", "steer_right")

    def test_all_lanes_blocked_triggers_reverse_needed(self):
        """When left, center, and right lanes are all severely blocked, reverse_needed is flagged."""
        nav = CorridorNavigator()
        frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)

        # Block all 3 lanes with obstacles
        yolo_dets = [
            {"class_id": 14, "class_name": "obstacle", "confidence": 0.95, "cx": 0.35, "cy": 0.65},
            {"class_id": 14, "class_name": "obstacle", "confidence": 0.95, "cx": 0.50, "cy": 0.65},
            {"class_id": 14, "class_name": "obstacle", "confidence": 0.95, "cx": 0.65, "cy": 0.65},
        ]

        eval_res = nav.evaluate_corridor_path(frame, yolo_detections=yolo_dets)
        assert eval_res["obstacle_detected"] is True
        assert eval_res["steering_bias"] == "reverse_needed"
        assert eval_res["recommended_turn_offset_deg"] == 180.0


class TestStuckDetectionAndOpticalFlow:
    """Tests stuck detection using minimap shift and optical flow (< 2.5s)."""

    def test_stuck_detection_timeout_under_2_5_seconds(self):
        """Stuck detector triggers when position doesn't shift for 2.0s (< 2.5s)."""
        detector = PhaseCorrelateUnstuckManeuver(stuck_threshold_px=1.2, stuck_duration_sec=2.0)
        t0 = 100.0

        # Create static minimap crop
        static_minimap = np.full((240, 240, 3), 120, dtype=np.uint8)
        static_minimap[60:180, 60:180] = 160

        # t=0s
        s0 = detector.update_walking_frame(static_minimap, is_walking=True, now=t0)
        assert s0["is_stuck"] is False

        # t=1.0s: still static
        s1 = detector.update_walking_frame(static_minimap, is_walking=True, now=t0 + 1.0)
        assert s1["is_stuck"] is False
        assert s1["stuck_elapsed_sec"] >= 1.0

        # t=2.1s: static for >= 2.0s (< 2.5s threshold) -> STUCK!
        s2 = detector.update_walking_frame(static_minimap, is_walking=True, now=t0 + 2.1)
        assert s2["is_stuck"] is True
        assert s2["shift_px"] < 1.2

    def test_stuck_not_triggered_when_moving(self):
        """When minimap shifts significantly, stuck timer resets."""
        detector = PhaseCorrelateUnstuckManeuver(stuck_threshold_px=1.2, stuck_duration_sec=2.0)
        t0 = 100.0

        # Shift the terrain patch 20px each frame away from center mask
        for i in range(5):
            terrain = np.zeros((240, 240, 3), dtype=np.uint8)
            terrain[40:80, 40 + i * 20 : 80 + i * 20] = 200
            res = detector.update_walking_frame(terrain, is_walking=True, now=t0 + i * 0.6)
            assert res["is_stuck"] is False

    def test_stuck_not_triggered_when_not_walking(self):
        """Idle player (is_walking=False) never triggers stuck."""
        detector = PhaseCorrelateUnstuckManeuver(stuck_duration_sec=2.0)
        static_minimap = np.full((240, 240, 3), 120, dtype=np.uint8)

        res = detector.update_walking_frame(static_minimap, is_walking=False, now=100.0)
        assert res["is_stuck"] is False
        res = detector.update_walking_frame(static_minimap, is_walking=False, now=110.0)
        assert res["is_stuck"] is False

    def test_optical_flow_measurement_on_viewport(self):
        """Optical flow calculation measures movement across viewport frames."""
        detector = PhaseCorrelateUnstuckManeuver()
        frame1 = np.zeros((300, 400, 3), dtype=np.uint8)
        frame2 = np.zeros((300, 400, 3), dtype=np.uint8)

        # Place texture in viewport crop region (y: 165..255, x: 140..260)
        cv2.rectangle(frame1, (150, 170), (230, 240), (200, 200, 200), -1)
        cv2.rectangle(frame2, (150, 185), (230, 255), (200, 200, 200), -1)

        flow1 = detector.compute_viewport_optical_flow(frame1)
        flow2 = detector.compute_viewport_optical_flow(frame2)
        assert flow2 > 0.0


class TestPhysicalUnstickManeuvers:
    """Tests jump, lateral strafe, reverse-backout, and escalating stages."""

    def test_unstick_tier_0_jump_and_strafe_left(self):
        """Attempt 1 executes Space -> A+W -> Realign."""
        mock_input = MockInputController()
        maneuver = PhaseCorrelateUnstuckManeuver(strafe_duration_sec=0.8)

        res = maneuver.execute_unstuck_sequence(
            input_controller=mock_input,
            target_angle_deg=90.0,
            current_angle_deg=0.0,
        )
        assert res["executed"] is True
        assert res["strafe_key"] == "a"
        assert res["sequence"] == ["SPACE", "A+W", "REALIGN"]
        assert "space" in mock_input.tapped_keys or "space" in mock_input.pressed_keys
        assert mock_input.held_keys[0][0] == ["a", "w"]
        assert res["realigned"] is True

    def test_unstick_tier_1_jump_and_strafe_right(self):
        """Attempt 2 executes Space -> D+W -> Realign."""
        mock_input = MockInputController()
        maneuver = PhaseCorrelateUnstuckManeuver(strafe_duration_sec=0.8)

        # 1st attempt (tier 0)
        maneuver.execute_unstuck_sequence(input_controller=mock_input)

        # 2nd attempt (tier 1)
        mock_input.held_keys.clear()
        res = maneuver.execute_unstuck_sequence(
            input_controller=mock_input,
            target_angle_deg=90.0,
            current_angle_deg=0.0,
        )
        assert res["executed"] is True
        assert res["strafe_key"] == "d"
        assert res["sequence"] == ["SPACE", "D+W", "REALIGN"]
        assert mock_input.held_keys[0][0] == ["d", "w"]

    def test_unstick_tier_2_physical_reverse_backout(self):
        """Attempt 3 executes physical reverse backout (STOP_W -> REVERSE_S -> REVERSE_STRAFE -> SPACE -> REALIGN)."""
        mock_input = MockInputController()
        maneuver = PhaseCorrelateUnstuckManeuver(reverse_duration_sec=1.0)

        # 1st and 2nd attempts
        maneuver.execute_unstuck_sequence(input_controller=mock_input)
        maneuver.execute_unstuck_sequence(input_controller=mock_input)

        # 3rd attempt: Tier 2 (Reverse-Backout)
        mock_input.held_keys.clear()
        mock_input.released_keys.clear()
        res = maneuver.execute_unstuck_sequence(
            input_controller=mock_input,
            target_angle_deg=180.0,
            current_angle_deg=90.0,
        )
        assert res["executed"] is True
        assert res["maneuver_type"] == "reverse_backout"
        assert "REVERSE_S" in res["sequence"]
        assert "SPACE" in res["sequence"]
        assert "w" in mock_input.released_keys
        held_key_names = [k[0] for k in mock_input.held_keys]
        assert ["s"] in held_key_names
        assert res["reverse_duration_sec"] > 0.0

    def test_unstick_tier_3_repath_recovery(self):
        """Attempt 4 executes extended reverse, pivot, and sets request_repath=True."""
        mock_input = MockInputController()
        maneuver = PhaseCorrelateUnstuckManeuver()

        # Run 3 attempts
        for _ in range(3):
            maneuver.execute_unstuck_sequence(input_controller=mock_input)

        # 4th attempt: Repath request
        res = maneuver.execute_unstuck_sequence(input_controller=mock_input)
        assert res["executed"] is True
        assert res["request_repath"] is True
        assert "REQUEST_REPATH" in res["sequence"]

    def test_forced_reverse_backout_maneuver(self):
        """Calling with maneuver_type='reverse_backout' directly executes reverse maneuver."""
        mock_input = MockInputController()
        maneuver = PhaseCorrelateUnstuckManeuver()

        res = maneuver.execute_unstuck_sequence(
            input_controller=mock_input,
            maneuver_type="reverse_backout",
        )
        assert res["executed"] is True
        assert res["maneuver_type"] == "reverse_backout"
        assert "REVERSE_S" in res["sequence"]
        assert res["reverse_duration_sec"] > 0.0

    def test_direct_execute_jump_and_strafe_methods(self):
        """Tests individual helper methods execute_jump, execute_lateral_strafe, execute_reverse_backout."""
        mock_input = MockInputController()
        maneuver = PhaseCorrelateUnstuckManeuver()

        # Jump
        assert maneuver.execute_jump(input_controller=mock_input) is True
        assert "space" in mock_input.tapped_keys or "space" in mock_input.pressed_keys

        # Lateral strafe
        assert maneuver.execute_lateral_strafe("d", duration=0.5, input_controller=mock_input) is True
        assert mock_input.held_keys[-1][0] == ["d", "w"]

        # Reverse backout
        assert maneuver.execute_reverse_backout(duration=0.8, input_controller=mock_input) is True
        assert "w" in mock_input.released_keys


class TestCorridorNavigatorIntegration:
    """Tests CorridorNavigator end-to-end integration."""

    def test_steer_step_with_corridor_evaluation(self):
        """steer_step incorporates corridor obstacle turn offset into adjusted bearing."""
        nav = CorridorNavigator()
        frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)

        # Obstacle in center
        yolo_dets = [{"class_id": 14, "class_name": "obstacle", "confidence": 0.9, "cx": 0.50, "cy": 0.65}]

        steer = nav.steer_step(
            current_pos=(100.0, 100.0),
            target_pos=(100.0, 50.0),  # North (0°)
            player_heading=0.0,
            frame=frame,
            live_detections=yolo_dets,
        )
        assert steer["target_bearing"] == 0.0
        assert steer["adjusted_bearing"] != 0.0
        assert steer["corridor_eval"]["obstacle_detected"] is True
        assert steer["corridor_eval"]["steering_bias"] in ("steer_left", "steer_right")

    def test_step_navigation_forced_unstick(self):
        """step_navigation executes forced unstick maneuver and returns status."""
        nav = CorridorNavigator()
        mock_input = MockInputController()
        nodes = [{"id": 1, "x": 100, "y": 100}, {"id": 2, "x": 200, "y": 200}]

        res = nav.step_navigation(
            frame=None,
            nodes=nodes,
            current_seq_pos=0,
            input_controller=mock_input,
            force_unstick_maneuver="reverse_backout",
        )
        assert res["status"] == "unstuck_executed"
        assert res["unstuck_triggered"] is True
        assert res["unstuck_info"]["maneuver_type"] == "reverse_backout"

    def test_corridor_convenience_methods(self):
        """is_corridor_clear and execute_physical_unstick work reliably."""
        nav = CorridorNavigator()
        mock_input = MockInputController()
        clear_frame = np.full((720, 1280, 3), (45, 120, 60), dtype=np.uint8)

        assert nav.is_corridor_clear(clear_frame) is True

        res = nav.execute_physical_unstick(
            maneuver_type="reverse_backout",
            input_controller=mock_input,
        )
        assert res["executed"] is True
        assert res["maneuver_type"] == "reverse_backout"
