# -*- coding: utf-8 -*-
"""
Comprehensive Unit & Integration Test Suite for Combat Positioning & Line-of-Sight Recovery
Agent C9 Verification Suite — AION 2 Gatherion Pixel Bot
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import cv2
import numpy as np
import pytest

from core.combat import (
    CombatPositioner,
    CombatPositioningConfig,
    CombatSystem,
    EngagementProfile,
    LineOfSightResult,
    LineOfSightTracker,
    ObstructionType,
    RangeEvaluation,
    RangeStatus,
    RecoveryActionResult,
    RecoveryTactic,
    analyze_corridor_occlusion,
    get_combat_capabilities,
    segment_intersects_rect,
    validate_combat_contract,
)


class MockInputDriver:
    """Mock Input Driver recording key presses, holds, and mouse clicks."""

    def __init__(self):
        self.pressed_keys: list[tuple[str, float]] = []
        self.clicked_buttons: list[str] = []
        self.down_keys: list[str] = []
        self.up_keys: list[str] = []

    def press_key(self, key: str, hold_sec: float = 0.05) -> bool:
        self.pressed_keys.append((key.lower(), float(hold_sec)))
        return True

    def key_down(self, key: str) -> bool:
        self.down_keys.append(key.lower())
        return True

    def key_up(self, key: str) -> bool:
        self.up_keys.append(key.lower())
        return True

    def human_click(self, button: str = "left") -> bool:
        self.clicked_buttons.append(button.lower())
        return True


class MockVision:
    """Mock Vision engine returning custom frames or player vitals."""

    def __init__(self, frame: np.ndarray | None = None, vitals: dict | None = None):
        self.frame = frame
        self.vitals = vitals or {}

    def grab_screen(self, grayscale: bool = False) -> np.ndarray | None:
        return self.frame

    def read_player_hp_mp_bars(self, frame=None) -> dict:
        return self.vitals


# =============================================================================
# 1. Geometry & Intersection Tests
# =============================================================================

class TestGeometricIntersection:
    """Tests for Liang-Barsky line segment to AABB intersection."""

    def test_direct_horizontal_intersection(self):
        # Ray from (100, 500) to (900, 500) passes through box at (400, 450, 100, 100)
        p1 = (100.0, 500.0)
        p2 = (900.0, 500.0)
        rect = (400.0, 450.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, rect) is True

    def test_direct_vertical_intersection(self):
        # Ray from (500, 800) to (500, 200) passes through box at (450, 400, 100, 100)
        p1 = (500.0, 800.0)
        p2 = (500.0, 200.0)
        rect = (450.0, 400.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, rect) is True

    def test_diagonal_intersection(self):
        # Ray from (0, 0) to (1000, 1000) passes through box at (400, 400, 100, 100)
        p1 = (0.0, 0.0)
        p2 = (1000.0, 1000.0)
        rect = (400.0, 400.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, rect) is True

    def test_ray_misses_box(self):
        # Ray from (100, 100) to (100, 900) misses box at (400, 400, 100, 100)
        p1 = (100.0, 100.0)
        p2 = (100.0, 900.0)
        rect = (400.0, 400.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, rect) is False

    def test_ray_short_of_box(self):
        # Ray ends before reaching the box
        p1 = (100.0, 500.0)
        p2 = (300.0, 500.0)
        rect = (400.0, 450.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, rect) is False


# =============================================================================
# 2. Computer Vision Corridor Occlusion Tests
# =============================================================================

class TestCorridorOcclusion:
    """Tests for CV edge and barrier occlusion detection along sightline."""

    def test_clear_corridor(self):
        # Uniform ground image with low edge density
        canvas = np.full((1080, 1920, 3), (80, 120, 90), dtype=np.uint8)
        p1 = (960, 800)
        p2 = (960, 350)
        is_obs, score, center, flank = analyze_corridor_occlusion(canvas, p1, p2)
        assert is_obs is False
        assert score < 0.25

    def test_occluded_corridor_with_rock_barrier(self):
        # Canvas with a large textured high-edge obstacle in the corridor
        canvas = np.full((1080, 1920, 3), (80, 120, 90), dtype=np.uint8)
        # Draw dark high-contrast rock barrier right in the sightline at y=500..600, x=900..1020
        cv2.rectangle(canvas, (900, 500), (1020, 600), (20, 20, 25), -1)
        # Add high-frequency edge texture inside obstacle
        for y in range(510, 590, 8):
            cv2.line(canvas, (910, y), (1010, y), (240, 240, 240), 2)

        p1 = (960, 800)
        p2 = (960, 350)
        is_obs, score, center, flank = analyze_corridor_occlusion(canvas, p1, p2, threshold=0.25)
        assert is_obs is True
        assert score >= 0.25
        assert center is not None
        assert flank in ("left", "right")


# =============================================================================
# 3. Combat Positioner Tests (Melee vs Ranged vs Hybrid)
# =============================================================================

class TestCombatPositioner:
    """Tests for engagement range evaluation and gap-close / kiting execution."""

    def test_melee_profile_range_evaluation(self):
        config = CombatPositioningConfig(profile=EngagementProfile.MELEE, melee_min_norm_h=0.12)
        pos = CombatPositioner(config)

        # Target too far (small bbox height: 8% screen height < 12%)
        far_eval = pos.evaluate_target_range(target_bbox=(800, 300, 80, 86), screen_shape=(1080, 1920))
        assert far_eval.status == RangeStatus.TOO_FAR
        assert far_eval.recommended_action == "step_forward"

        # Target in range (large bbox height: 25% screen height)
        in_range_eval = pos.evaluate_target_range(target_bbox=(800, 300, 150, 270), screen_shape=(1080, 1920))
        assert in_range_eval.status == RangeStatus.IN_RANGE
        assert in_range_eval.recommended_action == "none"

    def test_melee_distance_meters_evaluation(self):
        pos = CombatPositioner(CombatPositioningConfig(profile=EngagementProfile.MELEE))
        # 6.0 meters is too far for melee (limit 3.8m)
        eval_far = pos.evaluate_target_range(estimated_distance_m=6.0)
        assert eval_far.status == RangeStatus.TOO_FAR
        assert eval_far.recommended_action == "step_forward"

        # 2.5 meters is in melee range
        eval_close = pos.evaluate_target_range(estimated_distance_m=2.5)
        assert eval_close.status == RangeStatus.IN_RANGE
        assert eval_close.recommended_action == "none"

    def test_ranged_profile_range_evaluation(self):
        config = CombatPositioningConfig(
            profile=EngagementProfile.RANGED,
            ranged_min_norm_h=0.035,
            ranged_max_norm_h=0.160,
        )
        pos = CombatPositioner(config)

        # Target too far (> 25m, norm_h = 2% < 3.5%)
        far_eval = pos.evaluate_target_range(target_bbox=(800, 300, 25, 25), screen_shape=(1080, 1920))
        assert far_eval.status == RangeStatus.TOO_FAR
        assert far_eval.recommended_action == "step_forward"

        # Target in optimal standoff range (norm_h = 8%)
        in_range_eval = pos.evaluate_target_range(target_bbox=(800, 300, 80, 90), screen_shape=(1080, 1920))
        assert in_range_eval.status == RangeStatus.IN_RANGE
        assert in_range_eval.recommended_action == "none"

        # Target too close / danger zone (norm_h = 25% > 16%) -> triggers kiting
        close_eval = pos.evaluate_target_range(target_bbox=(800, 300, 200, 270), screen_shape=(1080, 1920))
        assert close_eval.status == RangeStatus.TOO_CLOSE
        assert close_eval.recommended_action == "backpedal"

    def test_gap_close_movement_dispatch(self):
        driver = MockInputDriver()
        pos = CombatPositioner()
        ok = pos.execute_gap_close(driver, duration_sec=0.30)
        assert ok is True
        assert pos.consecutive_gap_closes == 1
        assert len(driver.pressed_keys) == 1
        assert driver.pressed_keys[0][0] == "w"
        assert driver.pressed_keys[0][1] == 0.30

    def test_kiting_movement_dispatch(self):
        driver = MockInputDriver()
        pos = CombatPositioner(CombatPositioningConfig(profile=EngagementProfile.RANGED))
        ok = pos.execute_kiting(driver, duration_sec=0.25)
        assert ok is True
        assert pos.consecutive_kites == 1
        assert len(driver.pressed_keys) == 1
        assert driver.pressed_keys[0][0] == "s"
        assert driver.pressed_keys[0][1] == 0.25


# =============================================================================
# 4. Line of Sight Tracker Tests & Obstruction Recovery
# =============================================================================

class TestLineOfSightTracker:
    """Tests for LoS checking against geometric obstacles, damage stagnation, and recovery."""

    def test_los_clear_by_default(self):
        tracker = LineOfSightTracker()
        res = tracker.check_line_of_sight(player_pos=(960, 800), target_pos=(960, 350))
        assert res.has_los is True
        assert res.obstruction_type == ObstructionType.NONE

    def test_los_blocked_by_geometric_obstacle(self):
        tracker = LineOfSightTracker()
        player_pos = (960, 800)
        target_pos = (960, 350)
        # Obstacle box directly on the path between (960, 800) and (960, 350)
        obs_box = (900, 500, 120, 80)
        res = tracker.check_line_of_sight(
            player_pos=player_pos,
            target_pos=target_pos,
            obstacle_bboxes=[obs_box],
        )
        assert res.has_los is False
        assert res.obstruction_type == ObstructionType.GEOMETRIC_OBSTACLE
        assert res.flank_direction in ("left", "right")

    def test_damage_stagnation_triggers_los_obstruction(self):
        config = CombatPositioningConfig(damage_stagnation_threshold=3)
        tracker = LineOfSightTracker(config)

        # 1st cast dealing no damage
        tracker.record_cast_outcome(target_hp_pct=100.0)
        assert tracker.consecutive_stagnant_casts == 1
        assert tracker.check_line_of_sight().has_los is True

        # 2nd cast dealing no damage
        tracker.record_cast_outcome(target_hp_pct=100.0)
        assert tracker.consecutive_stagnant_casts == 2
        assert tracker.check_line_of_sight().has_los is True

        # 3rd cast dealing no damage -> STAGNATION THRESHOLD MET!
        tracker.record_cast_outcome(target_hp_pct=100.0)
        assert tracker.consecutive_stagnant_casts == 3
        res = tracker.check_line_of_sight()
        assert res.has_los is False
        assert res.obstruction_type == ObstructionType.DAMAGE_STAGNATION

    def test_damage_dealt_resets_stagnation(self):
        config = CombatPositioningConfig(damage_stagnation_threshold=3)
        tracker = LineOfSightTracker(config)

        tracker.record_cast_outcome(target_hp_pct=100.0)
        tracker.record_cast_outcome(target_hp_pct=100.0)
        assert tracker.consecutive_stagnant_casts == 2

        # Target took 10% damage!
        dealt = tracker.record_cast_outcome(target_hp_pct=90.0)
        assert dealt is True
        assert tracker.consecutive_stagnant_casts == 0
        assert tracker.check_line_of_sight().has_los is True

    def test_multi_phase_recovery_progression(self):
        driver = MockInputDriver()
        tracker = LineOfSightTracker(CombatPositioningConfig(max_recovery_attempts=5))
        los_res = LineOfSightResult(has_los=False, obstruction_type=ObstructionType.DAMAGE_STAGNATION, flank_direction="left")

        # Phase 1: Lateral strafe (A)
        act1 = tracker.recover_line_of_sight(driver, los_res)
        assert act1.tactic == RecoveryTactic.STRAFE_LEFT
        assert "a" in act1.keys_dispatched

        # Phase 2: Diagonal flank (A+W)
        act2 = tracker.recover_line_of_sight(driver, los_res)
        assert act2.tactic == RecoveryTactic.FLANK_LEFT
        assert "a" in act2.keys_dispatched
        assert "w" in act2.keys_dispatched

        # Phase 3: Obstacle clearing hop / jump forward (Space + W)
        act3 = tracker.recover_line_of_sight(driver, los_res)
        assert act3.tactic == RecoveryTactic.JUMP_FORWARD
        assert "space" in act3.keys_dispatched

        # Phase 4: Backpedal-flank (S then A)
        act4 = tracker.recover_line_of_sight(driver, los_res)
        assert act4.tactic == RecoveryTactic.BACKPEDAL_FLANK
        assert "s" in act4.keys_dispatched

        # Phase 5: Exceeded max attempts -> Tab cycle target
        act5 = tracker.recover_line_of_sight(driver, los_res)
        assert act5.tactic == RecoveryTactic.TAB_RESET
        assert act5.target_switched is True
        assert "tab" in act5.keys_dispatched
        # Resets tracker
        assert tracker.recovery_attempts == 0


# =============================================================================
# 5. Full CombatSystem Integration Tests
# =============================================================================

class TestCombatSystemIntegration:
    """Tests for integrated CombatSystem positioning, LoS checking, and combat rotation."""

    def test_contract_and_capabilities(self):
        caps = get_combat_capabilities()
        assert "engagement_range_optimization" in caps
        assert "line_of_sight_obstacle_check" in caps
        assert "adaptive_obstruction_recovery" in caps
        assert validate_combat_contract() is True

    def test_profile_configuration(self):
        combat = CombatSystem(profile=EngagementProfile.RANGED)
        assert combat.get_engagement_profile() == EngagementProfile.RANGED

        combat.set_engagement_profile("melee")
        assert combat.get_engagement_profile() == EngagementProfile.MELEE

    def test_check_under_attack_with_synthetic_frame(self):
        combat = CombatSystem()

        # Plain frame: not under attack
        plain = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        assert combat.check_under_attack(plain) is False

        # Frame with red enemy target health bar at top-center (y=40..70, x=700..900)
        hostile_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        # Red in BGR is (0, 0, 220)
        cv2.rectangle(hostile_frame, (700, 40), (900, 70), (0, 0, 220), -1)
        assert combat.check_under_attack(hostile_frame) is True

    def test_fight_back_step_gap_close_when_out_of_range(self):
        driver = MockInputDriver()
        combat = CombatSystem(driver=driver, profile=EngagementProfile.MELEE)

        # Hostile target present, but target bbox height is tiny (far away)
        hostile_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.rectangle(hostile_frame, (700, 40), (900, 70), (0, 0, 220), -1)

        small_bbox = (900, 400, 40, 40)  # height=40 (3.7% screen height < 11% melee min)
        result = combat.fight_back_step(
            input_ctrl=driver,
            frame=hostile_frame,
            target_bbox=small_bbox,
        )
        assert result is True
        assert combat.total_gap_closes == 1
        # 'w' key dispatched to close gap
        assert any(k[0] == "w" for k in driver.pressed_keys)

    def test_fight_back_step_kiting_when_ranged_too_close(self):
        driver = MockInputDriver()
        combat = CombatSystem(driver=driver, profile=EngagementProfile.RANGED)

        hostile_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.rectangle(hostile_frame, (700, 40), (900, 70), (0, 0, 220), -1)

        huge_bbox = (800, 300, 300, 350)  # height=350 (32% screen height > 16% ranged max)
        result = combat.fight_back_step(
            input_ctrl=driver,
            frame=hostile_frame,
            target_bbox=huge_bbox,
        )
        assert result is True
        assert combat.total_kites == 1
        # 's' key dispatched for tactical kiting
        assert any(k[0] == "s" for k in driver.pressed_keys)

    def test_fight_back_step_recovers_when_obstructed_by_box(self):
        driver = MockInputDriver()
        combat = CombatSystem(driver=driver, profile=EngagementProfile.MELEE)

        hostile_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.rectangle(hostile_frame, (700, 40), (900, 70), (0, 0, 220), -1)

        target_bbox = (900, 350, 120, 150)  # in melee range
        obstacle_bbox = (920, 500, 100, 100)  # sits directly between player (960, 777) and target (960, 425)

        result = combat.fight_back_step(
            input_ctrl=driver,
            frame=hostile_frame,
            target_bbox=target_bbox,
            obstacle_bboxes=[obstacle_bbox],
        )
        assert result is True
        assert combat.total_los_recoveries == 1
        # Strafe maneuver executed
        assert any(k[0] in ("a", "d") for k in driver.pressed_keys)

    def test_fight_back_step_attacks_when_in_range_and_clear_los(self):
        driver = MockInputDriver()
        combat = CombatSystem(driver=driver, profile=EngagementProfile.MELEE)

        hostile_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
        cv2.rectangle(hostile_frame, (700, 40), (900, 70), (0, 0, 220), -1)

        target_bbox = (900, 350, 150, 200)  # in melee range, no obstacles
        result = combat.fight_back_step(
            input_ctrl=driver,
            frame=hostile_frame,
            target_bbox=target_bbox,
        )
        assert result is True
        assert combat.total_attacks == 1
        # Either an attack skill key was pressed or a human click was performed
        has_skill_key = any(k[0] in ("1", "2", "3", "4", "q", "e", "r") for k in driver.pressed_keys)
        has_click = len(driver.clicked_buttons) > 0
        assert has_skill_key or has_click

    def test_enemy_defeat_loot_and_tab_cycle(self):
        driver = MockInputDriver()
        combat = CombatSystem(driver=driver)

        # Set fighting state
        combat.was_fighting = True
        combat.in_combat = True

        # Plain frame -> no enemy
        plain_frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)

        # 1st frame with no enemy increments streak
        combat.fight_back_step(input_ctrl=driver, frame=plain_frame)
        assert combat.no_target_streak == 1

        # 2nd frame with no enemy confirms enemy defeat -> loots (F) & tabs next
        combat.fight_back_step(input_ctrl=driver, frame=plain_frame)
        assert combat.in_combat is False
        assert any(k[0] == "f" for k in driver.pressed_keys)
        assert any(k[0] == "tab" for k in driver.pressed_keys)

    def test_diagnostic_and_health_summaries(self):
        combat = CombatSystem(profile=EngagementProfile.RANGED)
        health = combat.get_combat_health_summary()
        assert health["module"] == "combat"
        assert health["ready"] is True
        assert health["profile"] == "ranged"

        detector = combat.get_combat_detector_summary()
        assert detector["combat_detector_ready"] is True
        assert detector["profile"] == "ranged"

        diag = combat.get_combat_diagnostic_snapshot()
        assert diag["status"] == "ok"
        assert "positioning" in diag
