# -*- coding: utf-8 -*-
"""
Unit test suite for ManaPhaseCombat — Sub-100ms Burst Engine & 8-Key Round-Robin Rotation.
Agent C1 Verification Suite.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.mana_phase_combat import (
    ManaPhaseCombat,
    ManaPhase,
)


class TestManaPhaseCombatRotation:
    """Tests for 8-key round-robin rotation, combo chains, sub-100ms burst, and anti-sticky '2'."""

    def test_default_constants(self):
        """Verify default keys, burst interval, and GCD delay."""
        combat = ManaPhaseCombat()
        assert combat.combat_skill_keys == ["1", "2", "3", "4", "q", "e", "r", "t"]
        assert combat.burst_skill_interval_sec == 0.08
        assert combat.gcd_delay_sec == 0.12

    def test_sub_100ms_burst_delay_configuration(self):
        """Verify burst delay can be configured to 0.08s and 0.15s cleanly."""
        combat = ManaPhaseCombat(burst_skill_interval_sec=0.08, gcd_delay_sec=0.12)
        assert combat.burst_skill_interval_sec == 0.08
        assert combat.gcd_delay_sec == 0.12

        combat.set_burst_delays(burst_interval_sec=0.15, gcd_delay_sec=0.10)
        assert combat.burst_skill_interval_sec == 0.15
        assert combat.gcd_delay_sec == 0.10

    def test_should_cast_skill_deterministic_no_hidden_gaussian(self):
        """
        Verify should_cast_skill operates deterministically at sub-100ms intervals
        without hidden Gaussian delays or jitter.
        """
        combat = ManaPhaseCombat(burst_skill_interval_sec=0.08)
        now = 1000.0

        # First cast should succeed
        assert combat.should_cast_skill(now=now) is True

        # At 50ms later (< 80ms default), should be gated
        assert combat.should_cast_skill(now=now + 0.050) is False

        # At 79ms later (< 80ms), should still be gated
        assert combat.should_cast_skill(now=now + 0.079) is False

        # At 81ms later (>= 80ms), should pass cleanly
        assert combat.should_cast_skill(now=now + 0.081) is True

        # Now test custom min_gap e.g. 0.15s
        now2 = 2000.0
        assert combat.should_cast_skill(min_gap=0.15, now=now2) is True
        assert combat.should_cast_skill(min_gap=0.15, now=now2 + 0.10) is False
        assert combat.should_cast_skill(min_gap=0.15, now=now2 + 0.151) is True

    def test_is_gcd_ready(self):
        """Verify GCD delay check (~0.12s)."""
        combat = ManaPhaseCombat(gcd_delay_sec=0.12)
        now = 100.0

        combat.record_skill_cast("1", now=now)
        assert combat.is_gcd_ready(now=now + 0.06) is False
        assert combat.is_gcd_ready(now=now + 0.119) is False
        assert combat.is_gcd_ready(now=now + 0.121) is True

    def test_8_key_round_robin_burst_progression(self):
        """
        Verify that get_round_robin_burst_slice produces fair, contiguous
        round-robin slices across all 8 keys ('1', '2', '3', '4', 'q', 'e', 'r', 't').
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        # Slice 1 (3 keys) -> 1, 2, 3
        s1 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s1 == ["1", "2", "3"]

        # Slice 2 (3 keys) -> 4, q, e
        s2 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s2 == ["4", "q", "e"]

        # Slice 3 (3 keys) -> r, t, 1
        s3 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s3 == ["r", "t", "1"]

        # Slice 4 (3 keys) -> 2, 3, 4
        s4 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s4 == ["2", "3", "4"]

        # Slice 5 (3 keys) -> q, e, r
        s5 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s5 == ["q", "e", "r"]

    def test_eliminate_sticky_2_key_bias_in_decide(self):
        """
        Simulate 40 combat decisions in BURST phase.
        Verify that all 8 keys are cast evenly and key '2' has no sticky bias.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        counts = {k: 0 for k in combat.combat_skill_keys}
        t = 1000.0

        for _ in range(40):
            decision = combat.decide(player_mp=1.0, player_hp=1.0, now=t)
            assert decision.phase == ManaPhase.BURST
            assert len(decision.skill_keys) == 3

            # Record primary skill cast
            primary = decision.skill_keys[0]
            counts[primary] += 1
            combat.record_skill_cast(primary, now=t)
            t += 0.08

        # Each of the 8 keys should have been selected as primary ~5 times (uniform round-robin)
        for key in combat.combat_skill_keys:
            assert counts[key] == 5, f"Key '{key}' expected count 5, got {counts[key]}"

        # Verify key '2' specifically does not monopolize
        assert counts["2"] == 5
        assert counts["2"] < 10

    def test_sticky_2_prevention_when_vision_reports_only_key_2_ready(self):
        """
        If CV/SkillBarReader reports ready_skills=['2'], ManaPhaseCombat must NOT lock onto '2'.
        It must blend with fallback hotbar keys and cycle smoothly.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        chosen_primaries = []
        t = 500.0

        # Simulate 14 calls where ready_skills is always ['2']
        for _ in range(14):
            decision = combat.decide(player_mp=1.0, player_hp=1.0, ready_skills=["2"], now=t)
            primary = decision.skill_keys[0]
            chosen_primaries.append(primary)
            combat.record_skill_cast(primary, now=t)
            t += 0.08

        # Key '2' should NOT be repeated consecutively
        for i in range(1, len(chosen_primaries)):
            if chosen_primaries[i] == "2":
                assert chosen_primaries[i - 1] != "2", f"Sticky '2' detected at index {i}!"

        # Multiple other keys ('1', '3', '4', 'q', 'e', 'r', 't') must have been cast
        other_keys = [k for k in chosen_primaries if k != "2"]
        assert len(set(other_keys)) >= 3
        assert len(other_keys) > 0

    def test_sticky_2_prevention_in_regen_when_vision_reports_only_key_2_ready(self):
        """
        v20/madde 2: the BURST-phase blend above does not cover REGEN —
        decide() used `pool = ready if ready else fallback`, so with
        ready_skills=['2'] and low mana the slice was ['2'] every tick
        (REGEN is sticky: exit threshold is 0.99, so the bot can sit in
        REGEN for long stretches). REGEN must blend the same way.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        chosen_primaries = []
        t = 500.0

        # Simulate 14 calls in REGEN (mp < 0.95) where ready_skills is always ['2']
        for _ in range(14):
            decision = combat.decide(player_mp=0.5, player_hp=1.0, ready_skills=["2"], now=t)
            assert decision.phase == ManaPhase.REGEN
            primary = decision.skill_keys[0]
            chosen_primaries.append(primary)
            combat.record_skill_cast(primary, now=t)
            t += 0.08

        # Key '2' should NOT be repeated consecutively
        for i in range(1, len(chosen_primaries)):
            if chosen_primaries[i] == "2":
                assert chosen_primaries[i - 1] != "2", f"Sticky '2' detected at index {i}!"

        # Multiple other keys must have been cast
        other_keys = [k for k in chosen_primaries if k != "2"]
        assert len(set(other_keys)) >= 3
        assert len(other_keys) > 0

    def test_combo_chain_progression_and_expiry(self):
        """
        Verify combo chain: '1' -> '2' -> '3' and 'q' -> 'e' -> 'r'.
        When '1' is cast, '2' should be prioritized in next burst slice.
        """
        combat = ManaPhaseCombat()
        combat.register_combo("1", ["2", "3"])
        combat.register_combo("q", ["e", "r"])
        combat.reset_rotation()

        now = 100.0
        # Cast '1'
        combat.record_skill_cast("1", now=now)
        assert combat.get_next_combo_skill(now=now + 0.1) == "2"

        # Next burst decision should include '2' as primary
        dec1 = combat.decide(player_mp=1.0, now=now + 0.1)
        assert dec1.skill_keys[0] == "2"
        assert dec1.combo_active is True

        # Cast '2'
        combat.record_skill_cast("2", now=now + 0.1)
        assert combat.get_next_combo_skill(now=now + 0.2) == "3"

        # Next burst decision should include '3' as primary
        dec2 = combat.decide(player_mp=1.0, now=now + 0.2)
        assert dec2.skill_keys[0] == "3"
        assert dec2.combo_active is True

        # Cast '3' (finishes chain)
        combat.record_skill_cast("3", now=now + 0.3)
        assert combat.get_next_combo_skill(now=now + 0.4) is None

        # Next burst resumes normal round-robin
        dec3 = combat.decide(player_mp=1.0, now=now + 0.4)
        assert dec3.combo_active is False

    def test_combo_chain_timeout_resets_cleanly(self):
        """Verify that if combo timeout passes without casting follow-up, combo resets."""
        combat = ManaPhaseCombat()
        combat.register_combo("1", ["2"])
        combat.reset_rotation()

        now = 100.0
        combat.record_skill_cast("1", now=now)
        assert combat.get_next_combo_skill(now=now + 1.0) == "2"

        # At now + 3.0s (> 2.0s timeout), combo has expired
        assert combat.get_next_combo_skill(now=now + 3.0) is None

    def test_regen_phase_round_robin(self):
        """Verify that REGEN phase also rotates through low-mana skills without sticking."""
        combat = ManaPhaseCombat(mp_full_threshold=0.90)
        combat.set_lexicon({
            "1": "basic",
            "2": "attack_low_mana",
            "3": "attack",
            "4": "burst_high_mana",
        })
        combat.reset_rotation()

        # mp = 0.40 -> REGEN phase
        dec1 = combat.decide(player_mp=0.40)
        assert dec1.phase == ManaPhase.REGEN
        assert dec1.hold_left is True
        assert dec1.hold_right is False
        assert len(dec1.skill_keys) <= 2
        # Confirm skills are low-mana / basic
        for k in dec1.skill_keys:
            assert k in ["1", "2", "3"]

    def test_custom_combat_skill_keys_and_case_normalization(self):
        """Verify custom keys e.g. uppercase 'Q', 'E', 'R' are normalized and rotated cleanly."""
        combat = ManaPhaseCombat()
        combat.set_combat_skill_keys(["1", "2", "Q", "E"])
        assert combat.combat_skill_keys == ["1", "2", "q", "e"]
        combat.reset_rotation()

        slice1 = combat.get_round_robin_burst_slice(count=2)
        assert slice1 == ["1", "2"]
        slice2 = combat.get_round_robin_burst_slice(count=2)
        assert slice2 == ["q", "e"]
        slice3 = combat.get_round_robin_burst_slice(count=2)
        assert slice3 == ["1", "2"]

    def test_panic_hp_and_cc_defense_priority(self):
        """Verify that when HP < panic_threshold or CC is active, defense skill is prioritized first."""
        combat = ManaPhaseCombat(hp_panic_threshold=0.50)
        combat.set_lexicon({
            "r": "defense_heal",
            "1": "attack",
            "2": "attack",
            "3": "attack",
        })
        combat.reset_rotation()

        # Normal HP -> regular burst
        normal_dec = combat.decide(player_mp=1.0, player_hp=0.90)
        assert normal_dec.skill_keys[0] != "r"

        # Low HP -> panic defense prioritized
        panic_dec = combat.decide(player_mp=1.0, player_hp=0.30)
        assert panic_dec.skill_keys[0] == "r"
        assert "panic" in panic_dec.reason

        # CC active -> defense prioritized
        cc_dec = combat.decide(player_mp=1.0, player_hp=0.90, cc_active=True)
        assert cc_dec.skill_keys[0] == "r"
        assert "cc" in cc_dec.reason

    def test_high_frequency_burst_stress_no_deadlock(self):
        """Stress test: 1000 burst decisions at 0.08s intervals. Zero deadlocks or NaN."""
        combat = ManaPhaseCombat(burst_skill_interval_sec=0.08)
        combat.reset_rotation()

        t = 0.0
        casts = 0
        for _ in range(1000):
            if combat.should_cast_skill(now=t):
                dec = combat.decide(player_mp=1.0, now=t)
                primary = dec.skill_keys[0]
                combat.record_skill_cast(primary, now=t)
                casts += 1
            t += 0.04  # Tick every 40ms; should cast every other tick (80ms)

        # 1000 ticks * 40ms = 40 seconds. At 80ms interval, ~500 casts should succeed.
        assert 490 <= casts <= 510, f"Expected ~500 casts, got {casts}"


class TestManaPhaseCombatVitalsRoi:
    """D13: read_vitals_from_frame converts only the ROI crops to HSV
    (was: whole-frame cvtColor every tick) and the log callback is funnelled
    through one helper. Ratios must be bit-identical to the full-frame path."""

    @staticmethod
    def _frame_with_bars(hp_fill=0.5, mp_fill=0.7, target_fill=0.0, target_rect=(0.350, 0.030, 0.650, 0.075)):
        import numpy as np

        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        h, w = 1080, 1920
        # Default ROIs: hp (0.345,0.880,0.485,0.915), mp (0.515,0.880,0.655,0.915).
        # Fill bars from the left in BGR.
        if hp_fill > 0:
            x2 = int(w * 0.345 + (w * 0.485 - w * 0.345) * hp_fill)
            frame[int(h * 0.880):int(h * 0.915), int(w * 0.345):x2] = (0, 0, 255)
        if mp_fill > 0:
            x2 = int(w * 0.515 + (w * 0.655 - w * 0.515) * mp_fill)
            frame[int(h * 0.880):int(h * 0.915), int(w * 0.515):x2] = (255, 0, 0)
        if target_fill > 0:
            x2 = int(w * target_rect[0] + (w * target_rect[2] - w * target_rect[0]) * target_fill)
            frame[int(h * target_rect[1]):int(h * target_rect[3]), int(w * target_rect[0]):x2] = (0, 0, 255)
        return frame

    def test_read_vitals_roi_crop_ratios_match_full_frame_baseline(self):
        """Crop-first HSV must give the same ratios as the old whole-frame
        conversion on the same frame (reference computed inline below)."""
        import cv2
        import numpy as np

        combat = ManaPhaseCombat()
        frame = self._frame_with_bars()
        vitals = combat.read_vitals_from_frame(frame)
        assert vitals["vitals_valid"] is True
        assert vitals["player_hp"] == 0.498
        assert vitals["player_mp"] == 0.699
        assert vitals["has_target"] == 0.0

        # Reference: convert the whole frame, then mask the same ROIs.
        rois = combat.get_calibrated_roi_bounds()
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hp_rect = rois["player_hp_bar"]
        hp_crop = hsv[int(h * hp_rect[1]):int(h * hp_rect[3]), int(w * hp_rect[0]):int(w * hp_rect[2])]
        red_mask = ((hp_crop[:, :, 0] < 12) | (hp_crop[:, :, 0] > 168)) & (hp_crop[:, :, 1] > 110) & (hp_crop[:, :, 2] > 90)
        ref_hp = round(float(np.sum(np.mean(red_mask, axis=0) > 0.25)) / float(max(1, hp_crop.shape[1])), 3)
        assert vitals["player_hp"] == ref_hp

    def test_read_vitals_full_and_empty_bars(self):
        combat = ManaPhaseCombat()
        full = combat.read_vitals_from_frame(self._frame_with_bars(hp_fill=1.0, mp_fill=1.0))
        assert full["player_hp"] == 1.0
        assert full["player_mp"] == 1.0
        assert full["hp_potion_needed"] is False
        assert full["mp_potion_needed"] is False

        empty = combat.read_vitals_from_frame(self._frame_with_bars(hp_fill=0.0, mp_fill=0.0))
        assert empty["player_hp"] == 0.0
        assert empty["player_mp"] == 0.0
        assert empty["hp_potion_needed"] is True
        assert empty["is_vitality_critical"] is True

    def test_read_vitals_target_bar_detected(self):
        combat = ManaPhaseCombat()
        target_rect = combat.get_calibrated_roi_bounds()["target_hp_bar"]
        vitals = combat.read_vitals_from_frame(
            self._frame_with_bars(target_fill=1.0, target_rect=target_rect)
        )
        assert vitals["has_target"] == 1.0
        assert vitals["target_hp"] == 1.0

    def test_read_vitals_tiny_frame_empty_crop_keeps_defaults(self):
        """A 1x1 frame makes every ROI crop empty: ratios stay at the
        1.0 defaults instead of crashing or dividing by zero."""
        import numpy as np

        combat = ManaPhaseCombat()
        vitals = combat.read_vitals_from_frame(np.zeros((1, 1, 3), dtype=np.uint8))
        assert vitals["vitals_valid"] is True
        assert vitals["player_hp"] == 1.0
        assert vitals["player_mp"] == 1.0

    def test_read_vitals_none_frame_fail_closed(self):
        combat = ManaPhaseCombat()
        vitals = combat.read_vitals_from_frame(None)
        assert vitals["vitals_valid"] is False
        assert vitals["player_hp"] == 0.0
        assert vitals["player_mp"] == 0.0
        assert vitals["hp_potion_needed"] is True

    def test_log_callback_exception_never_breaks_potion_decision(self):
        """_log_info must swallow a raising on_log callback (same contract
        as the old inline try/except at every log site)."""
        def _boom(_msg):
            raise RuntimeError("callback exploded")

        combat = ManaPhaseCombat(on_log=_boom)
        res = combat.check_auto_potions(player_hp=0.30, player_mp=1.0, now=1000.0)
        assert res.hp_potion_triggered is True
        assert res.keys_to_press == ["F1"]

    def test_forbidden_consumable_keys_never_trigger(self):
        """X / F6 / B are automatic client slots; the shared module
        constant must block them on both potion paths."""
        from core.mana_phase_combat import FORBIDDEN_CONSUMABLE_KEYS

        assert FORBIDDEN_CONSUMABLE_KEYS == frozenset({"x", "f6", "b"})
        combat = ManaPhaseCombat(hp_potion_key="X", mp_potion_key="F6")
        res = combat.check_auto_potions(player_hp=0.10, player_mp=0.10, now=1000.0)
        assert res.hp_potion_triggered is False
        assert res.mp_potion_triggered is False
        assert res.keys_to_press == []

