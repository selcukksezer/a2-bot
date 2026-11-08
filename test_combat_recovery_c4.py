# -*- coding: utf-8 -*-
"""
Agent C4 Verification Test Suite:
Target Acquisition, Fast TAB Cycling, Olive-Green Mob Detection & 1-Frame Flicker Grace.

Covers:
1. Fast TAB Cycling: responsive cadence, loot-to-TAB transition, no idle stalls.
2. Olive-Green Mob Detection Mask: HSV thresholding, aspect ratio filtering, self-bar masking.
3. 1-Frame Targeting Flicker Grace Ticks: preventing premature combat aborts on transient frame drops.
"""

import time
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

from core.combat_recovery import CombatRecovery


class MockInputController:
    """Mock input controller capturing key presses and mouse clicks."""

    def __init__(self, test_mode: bool = False):
        self.pressed = []
        self.clicks = []
        self.test_mode = test_mode

    def press_key(self, key, min_hold=0.045, max_hold=0.095):
        self.pressed.append(str(key).lower())
        return True

    def send_key(self, key):
        self.pressed.append(str(key).lower())
        return True

    def human_click(self, x=None, y=None, button="left", min_hold=0.045, max_hold=0.095):
        self.clicks.append(button)
        return True


def create_synthetic_frame(
    width: int = 1280,
    height: int = 720,
    olive_bar: bool = False,
    red_bar: bool = False,
    combat_zone_olive: bool = False,
    self_overhead_only: bool = False,
) -> np.ndarray:
    """
    Creates a synthetic 1280x720 BGR frame with realistic HUD regions
    and optional target HP bars for deterministic unit testing.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Add bottom HUD (Player HP/MP) so HUD visibility checks pass
    # Alt-orta HP bar: x: 0.345..0.478, y: 0.885..0.902 (Red: H=0, S=180, V=200)
    hp_x1, hp_x2 = int(width * 0.365), int(width * 0.475)
    hp_y1, hp_y2 = int(height * 0.885), int(height * 0.902)
    hsv_hp = np.zeros((hp_y2 - hp_y1, hp_x2 - hp_x1, 3), dtype=np.uint8)
    hsv_hp[:, :, 0] = 0    # Red
    hsv_hp[:, :, 1] = 200  # Sat
    hsv_hp[:, :, 2] = 200  # Val
    frame[hp_y1:hp_y2, hp_x1:hp_x2] = cv2.cvtColor(hsv_hp, cv2.COLOR_HSV2BGR)

    # Alt-orta MP bar: x: 0.519..0.636, y: 0.885..0.902 (Blue: H=100, S=180, V=180)
    mp_x1, mp_x2 = int(width * 0.520), int(width * 0.630)
    mp_y1, mp_y2 = int(height * 0.885), int(height * 0.902)
    hsv_mp = np.zeros((mp_y2 - mp_y1, mp_x2 - mp_x1, 3), dtype=np.uint8)
    hsv_mp[:, :, 0] = 100  # Blue
    hsv_mp[:, :, 1] = 180
    hsv_mp[:, :, 2] = 180
    frame[mp_y1:mp_y2, mp_x1:mp_x2] = cv2.cvtColor(hsv_mp, cv2.COLOR_HSV2BGR)

    # 2. Self overhead bar (if self_overhead_only is requested)
    if self_overhead_only:
        so_x1 = int(width * CombatRecovery.SELF_OVERHEAD_X1_NORM)
        so_x2 = int(width * CombatRecovery.SELF_OVERHEAD_X2_NORM)
        so_y1 = int(height * CombatRecovery.SELF_OVERHEAD_Y1_NORM)
        so_y2 = int(height * CombatRecovery.SELF_OVERHEAD_Y2_NORM)
        hsv_so = np.zeros((so_y2 - so_y1, so_x2 - so_x1, 3), dtype=np.uint8)
        hsv_so[:, :, 0] = 40   # Olive Hue in self zone
        hsv_so[:, :, 1] = 150
        hsv_so[:, :, 2] = 160
        frame[so_y1:so_y2, so_x1:so_x2] = cv2.cvtColor(hsv_so, cv2.COLOR_HSV2BGR)
        return frame

    # 3. Upper target frame: x: 0.35..0.55, y: 0.04..0.06
    if red_bar:
        tb_x1, tb_x2 = int(width * 0.38), int(width * 0.52)
        tb_y1, tb_y2 = int(height * 0.04), int(height * 0.055)
        hsv_tb = np.zeros((tb_y2 - tb_y1, tb_x2 - tb_x1, 3), dtype=np.uint8)
        hsv_tb[:, :, 0] = 0    # Red hostile target bar
        hsv_tb[:, :, 1] = 220
        hsv_tb[:, :, 2] = 220
        frame[tb_y1:tb_y2, tb_x1:tb_x2] = cv2.cvtColor(hsv_tb, cv2.COLOR_HSV2BGR)

    if olive_bar:
        # Olive-green monster target bar (Yellow-Maned Zaif: H=45, S=160, V=180)
        ob_x1, ob_x2 = int(width * 0.38), int(width * 0.52)
        ob_y1, ob_y2 = int(height * 0.04), int(height * 0.055)
        hsv_ob = np.zeros((ob_y2 - ob_y1, ob_x2 - ob_x1, 3), dtype=np.uint8)
        hsv_ob[:, :, 0] = 45   # Olive green Hue
        hsv_ob[:, :, 1] = 160  # Sat >= 80
        hsv_ob[:, :, 2] = 180  # Val >= 80
        frame[ob_y1:ob_y2, ob_x1:ob_x2] = cv2.cvtColor(hsv_ob, cv2.COLOR_HSV2BGR)

    if combat_zone_olive:
        # Mid viewport olive-green mob nameplate/bar (outside self-overhead zone)
        cz_x1, cz_x2 = int(width * 0.28), int(width * 0.42)
        cz_y1, cz_y2 = int(height * 0.25), int(height * 0.27)
        hsv_cz = np.zeros((cz_y2 - cz_y1, cz_x2 - cz_x1, 3), dtype=np.uint8)
        hsv_cz[:, :, 0] = 42   # Olive green
        hsv_cz[:, :, 1] = 170
        hsv_cz[:, :, 2] = 190
        frame[cz_y1:cz_y2, cz_x1:cz_x2] = cv2.cvtColor(hsv_cz, cv2.COLOR_HSV2BGR)

    return frame


class TestCombatRecoveryAgentC4(unittest.TestCase):
    """Rigorous tests for Agent C4 features."""

    def setUp(self):
        self.cr = CombatRecovery()
        self.input_ctrl = MockInputController()

    # ========================================================================
    # 1. FAST TAB CYCLING TESTS
    # ========================================================================

    def test_fast_tab_cycle_cadence_enforced(self):
        """fast_tab_cycle should respect min_interval_sec and not spam within cadence window."""
        self.cr.last_tab_time = 0.0
        # First press must succeed
        pressed_first = self.cr.fast_tab_cycle(self.input_ctrl, min_interval_sec=0.55)
        self.assertTrue(pressed_first)
        self.assertEqual(len(self.input_ctrl.pressed), 1)
        self.assertEqual(self.input_ctrl.pressed[-1], "tab")

        # Second press immediately after (< 0.55s) must be throttled
        pressed_immediate = self.cr.fast_tab_cycle(self.input_ctrl, min_interval_sec=0.55)
        self.assertFalse(pressed_immediate)
        self.assertEqual(len(self.input_ctrl.pressed), 1)

        # Force press overrides interval
        pressed_forced = self.cr.fast_tab_cycle(self.input_ctrl, force=True)
        self.assertTrue(pressed_forced)
        self.assertEqual(len(self.input_ctrl.pressed), 2)

    def test_fast_tab_cycle_test_mode_acceleration(self):
        """When input_ctrl.test_mode=True, fast_tab_cycle accelerates interval for headless tests."""
        test_ctrl = MockInputController(test_mode=True)
        self.cr.last_tab_time = 0.0
        self.assertTrue(self.cr.fast_tab_cycle(test_ctrl))
        # After 0.06s in test mode, next cycle is allowed
        self.cr.last_tab_time = time.monotonic() - 0.06
        self.assertTrue(self.cr.fast_tab_cycle(test_ctrl))
        self.assertEqual(len(test_ctrl.pressed), 2)

    def test_fast_tab_counter_increments(self):
        """tab_cycle_count increments accurately on every successful TAB press."""
        initial_count = self.cr.tab_cycle_count
        self.cr.fast_tab_cycle(self.input_ctrl, force=True)
        self.cr.fast_tab_cycle(self.input_ctrl, force=True)
        self.assertEqual(self.cr.tab_cycle_count, initial_count + 2)

    # ========================================================================
    # 2. OLIVE-GREEN MOB DETECTION MASK TESTS
    # ========================================================================

    def test_olive_mob_mask_color_bounds(self):
        """get_olive_mob_mask correctly extracts olive-green HSV range and rejects red/blue/black."""
        # 1. Pure olive-green patch (H=45, S=160, V=180)
        olive_bgr = cv2.cvtColor(np.array([[[45, 160, 180]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
        mask = self.cr.get_olive_mob_mask(olive_bgr)
        self.assertEqual(mask[0, 0], 1)

        # 2. Pure red patch (H=0, S=200, V=200) -> must be 0 in olive mask
        red_bgr = cv2.cvtColor(np.array([[[0, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
        mask_red = self.cr.get_olive_mob_mask(red_bgr)
        self.assertEqual(mask_red[0, 0], 0)

        # 3. Pure blue patch (H=110, S=200, V=200) -> must be 0
        blue_bgr = cv2.cvtColor(np.array([[[110, 200, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
        mask_blue = self.cr.get_olive_mob_mask(blue_bgr)
        self.assertEqual(mask_blue[0, 0], 0)

        # 4. Low saturation dark grey patch -> must be 0
        dark_bgr = cv2.cvtColor(np.array([[[45, 20, 40]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
        mask_dark = self.cr.get_olive_mob_mask(dark_bgr)
        self.assertEqual(mask_dark[0, 0], 0)

    def test_detect_upper_target_bar_olive_mob(self):
        """_raw_detect_enemy_target_bar detects upper target frame with olive-green mob bar."""
        frame = create_synthetic_frame(olive_bar=True)
        detected = self.cr._raw_detect_enemy_target_bar(frame)
        self.assertTrue(detected, "Olive-green target bar in upper frame must be detected")

    def test_detect_combat_zone_olive_mob_bar(self):
        """detect_olive_mob_bar detects horizontal olive nameplates in the combat zone."""
        frame = create_synthetic_frame(combat_zone_olive=True)
        detected = self.cr.detect_olive_mob_bar(frame)
        self.assertTrue(detected, "Olive-green mob nameplate in combat zone must be detected")

        # Full target bar detection should also report True
        self.assertTrue(self.cr._raw_detect_enemy_target_bar(frame))

    def test_self_overhead_bar_rejection(self):
        """Olive pixels inside the player's self-overhead zone must NOT trigger enemy detection."""
        frame = create_synthetic_frame(self_overhead_only=True)
        detected = self.cr.detect_olive_mob_bar(frame)
        self.assertFalse(detected, "Player self-overhead zone must be masked out to prevent false enemy lock")

    # ========================================================================
    # 3. 1-FRAME TARGETING FLICKER GRACE TICKS TESTS
    # ========================================================================

    def test_single_frame_flicker_grace_sustains_combat(self):
        """
        Verify 1-frame targeting flicker grace:
        Tick 1: Target present -> True
        Tick 2: Target dropped for 1 frame -> sustained as True via flicker grace!
        Tick 3: Target restored -> True, grace counter reset to 0
        """
        frame_present = create_synthetic_frame(olive_bar=True)
        frame_empty = create_synthetic_frame(olive_bar=False)

        self.cr.reset_flicker_grace()
        self.cr.in_combat = True

        # Tick 1: Target present
        t1 = self.cr._detect_enemy_target_bar(frame_present)
        self.assertTrue(t1)
        self.assertFalse(self.cr.is_targeting_flicker_grace())
        self.assertEqual(self.cr._flicker_grace_counter, 0)

        # Advance timestamp slightly so tick is distinct
        time.sleep(0.09)

        # Tick 2: Target flicker (missing for 1 frame)
        t2 = self.cr._detect_enemy_target_bar(frame_empty)
        self.assertTrue(t2, "1-frame targeting flicker must be sustained via grace tick")
        self.assertTrue(self.cr.is_targeting_flicker_grace())
        self.assertEqual(self.cr._flicker_grace_counter, 1)

        time.sleep(0.09)

        # Tick 3: Target restored
        t3 = self.cr._detect_enemy_target_bar(frame_present)
        self.assertTrue(t3)
        self.assertFalse(self.cr.is_targeting_flicker_grace(), "Grace should deactivate when target returns")
        self.assertEqual(self.cr._flicker_grace_counter, 0)

    def test_confirmed_target_loss_after_grace_expires(self):
        """
        When target is absent for >= 2 consecutive frames, grace expires on tick 2 and returns False.
        """
        frame_present = create_synthetic_frame(red_bar=True)
        frame_empty = create_synthetic_frame()

        self.cr.reset_flicker_grace()
        self.cr.in_combat = True

        # Tick 1: Target present
        self.assertTrue(self.cr._detect_enemy_target_bar(frame_present))

        time.sleep(0.09)

        # Tick 2: 1-frame grace active
        self.assertTrue(self.cr._detect_enemy_target_bar(frame_empty))
        self.assertTrue(self.cr.is_flicker_grace_active)

        time.sleep(0.09)

        # Tick 3: Second consecutive frame of target absence -> grace expired, must return False!
        t3 = self.cr._detect_enemy_target_bar(frame_empty)
        self.assertFalse(t3, "Second frame without target must confirm loss and return False")
        self.assertFalse(self.cr.is_flicker_grace_active)

    def test_check_under_attack_sustained_during_flicker(self):
        """check_under_attack remains True during a 1-frame targeting flicker, preventing abort."""
        frame_present = create_synthetic_frame(olive_bar=True)
        frame_flicker = create_synthetic_frame(olive_bar=False)

        self.cr.reset_flicker_grace()
        # Tick 1: under attack
        under_atk1, _ = self.cr.check_under_attack(frame_present)
        self.assertTrue(under_atk1)

        time.sleep(0.09)

        # Tick 2: flicker frame
        under_atk2, _ = self.cr.check_under_attack(frame_flicker)
        self.assertTrue(under_atk2, "check_under_attack must not abort on 1-frame target flicker")

    def test_fight_back_step_grace_tick_does_not_abort(self):
        """fight_back_step does not prematurely declare target dead on a 1-frame flicker."""
        frame_present = create_synthetic_frame(red_bar=True)
        frame_flicker = create_synthetic_frame(red_bar=False)

        self.cr.reset_flicker_grace()
        self.cr.in_combat = True

        # Step 1: Real target present
        self.cr.fight_back_step(self.input_ctrl, frame=frame_present)
        self.assertEqual(getattr(self.cr, "_no_target_steps", 0), 0)

        time.sleep(0.09)

        # Step 2: Flicker frame -> sustained via grace, no_target_steps remains 0
        self.cr.fight_back_step(self.input_ctrl, frame=frame_flicker)
        self.assertEqual(
            getattr(self.cr, "_no_target_steps", 0),
            0,
            "1-frame flicker must not increment no_target_steps",
        )

    def test_execute_combat_loop_fast_tab_first_action(self):
        """execute_combat_loop executes fast TAB cycling as the very first action."""
        frame_present = create_synthetic_frame(olive_bar=True)
        frames = [frame_present, None]

        def mock_capture():
            return frames.pop(0) if frames else None

        self.cr.execute_combat_loop(
            self.input_ctrl,
            capture=mock_capture,
            max_fight_sec=0.15,
            stop_gather_sec=0.01,
        )
        self.assertIn("tab", self.input_ctrl.pressed)
        self.assertEqual(self.input_ctrl.pressed[0], "tab")

    def test_execute_combat_loop_flicker_grace_sustains_fight(self):
        """execute_combat_loop does not abort or loot when target drops for 1 frame."""
        frame_present = create_synthetic_frame(olive_bar=True)
        frame_flicker = create_synthetic_frame(olive_bar=False)

        # Sequence: Present -> Flicker -> Present -> Present
        frame_seq = [frame_present, frame_flicker, frame_present, frame_present]

        def mock_capture():
            return frame_seq.pop(0) if frame_seq else frame_present

        step_count = [0]

        def is_cancelled():
            step_count[0] += 1
            return step_count[0] > 4

        res = self.cr.execute_combat_loop(
            self.input_ctrl,
            capture=mock_capture,
            is_cancelled_fn=is_cancelled,
            max_fight_sec=2.0,
            stop_gather_sec=0.01,
        )
        self.assertEqual(res, "CANCELLED")
        self.assertNotIn("f", self.input_ctrl.pressed)

    def test_execute_combat_loop_cleared_on_real_death(self):
        """execute_combat_loop loots (F) and returns CLEARED when mob is dead across consecutive frames."""
        frame_present = create_synthetic_frame(red_bar=True)
        frame_dead = create_synthetic_frame(red_bar=False)

        # Sequence: Present -> Dead -> Dead -> Dead (beyond flicker grace)
        frame_seq = [frame_present, frame_dead, frame_dead, frame_dead, frame_dead]

        def mock_capture():
            return frame_seq.pop(0) if frame_seq else frame_dead

        res = self.cr.execute_combat_loop(
            self.input_ctrl,
            capture=mock_capture,
            max_fight_sec=3.0,
            stop_gather_sec=0.05,
        )
        self.assertEqual(res, "CLEARED")
        self.assertIn("f", self.input_ctrl.pressed, "Loot (F) must be pressed upon monster death")

    def test_reset_flicker_grace_clears_all_counters(self):
        """reset_flicker_grace completely resets internal tracking."""
        self.cr._flicker_grace_counter = 1
        self.cr.is_flicker_grace_active = True
        self.cr._last_raw_target_detected = True
        self.cr.reset_flicker_grace()
        self.assertEqual(self.cr._flicker_grace_counter, 0)
        self.assertFalse(self.cr.is_flicker_grace_active)
        self.assertFalse(self.cr._last_raw_target_detected)


if __name__ == "__main__":
    unittest.main()


