# -*- coding: utf-8 -*-
"""
Unit test suite for ManaPhaseCombat — Sub-100ms Burst Engine & 7-Key Round-Robin Rotation.
Agent C1 Verification Suite.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import pytest
from core.mana_phase_combat import (
    ManaPhaseCombat,
    ManaPhase,
    ManaPhaseDecision,
    DEFAULT_COMBAT_SKILL_KEYS,
    DEFAULT_BURST_INTERVAL_SEC,
    DEFAULT_GCD_DELAY_SEC,
)


class TestManaPhaseCombatRotation:
    """Tests for 7-key round-robin rotation, combo chains, sub-100ms burst, and anti-sticky '2'."""

    def test_default_constants(self):
        """Verify default keys, burst interval, and GCD delay."""
        combat = ManaPhaseCombat()
        assert combat.combat_skill_keys == ["1", "2", "3", "4", "q", "e", "r"]
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

    def test_7_key_round_robin_burst_progression(self):
        """
        Verify that get_round_robin_burst_slice produces fair, contiguous
        round-robin slices across all 7 keys ('1', '2', '3', '4', 'q', 'e', 'r').
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        # Slice 1 (3 keys) -> 1, 2, 3
        s1 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s1 == ["1", "2", "3"]

        # Slice 2 (3 keys) -> 4, q, e
        s2 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s2 == ["4", "q", "e"]

        # Slice 3 (3 keys) -> r, 1, 2
        s3 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s3 == ["r", "1", "2"]

        # Slice 4 (3 keys) -> 3, 4, q
        s4 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s4 == ["3", "4", "q"]

        # Slice 5 (3 keys) -> e, r, 1
        s5 = combat.get_round_robin_burst_slice(count=3, allow_combo=False)
        assert s5 == ["e", "r", "1"]

    def test_eliminate_sticky_2_key_bias_in_decide(self):
        """
        Simulate 35 combat decisions in BURST phase.
        Verify that all 7 keys are cast evenly and key '2' has no sticky bias.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        counts = {k: 0 for k in combat.combat_skill_keys}
        t = 1000.0

        for _ in range(35):
            decision = combat.decide(player_mp=1.0, player_hp=1.0, now=t)
            assert decision.phase == ManaPhase.BURST
            assert len(decision.skill_keys) == 3

            # Record primary skill cast
            primary = decision.skill_keys[0]
            counts[primary] += 1
            combat.record_skill_cast(primary, now=t)
            t += 0.08

        # Each of the 7 keys should have been selected as primary ~5 times (uniform round-robin)
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

        # Multiple other keys ('1', '3', '4', 'q', 'e', 'r') must have been cast
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

