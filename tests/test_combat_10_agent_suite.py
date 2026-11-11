# -*- coding: utf-8 -*-
"""
Comprehensive 10-Agent Combat Test Suite (Agent C10).
File: tests/test_combat_10_agent_suite.py

Verifies the 6 Core Combat Pillars across Unit & Integration levels:
1. Sub-100ms Burst Rotation (burst timing, GCD delays, deterministic pacing, no hidden humanizer lag)
2. 7-Key Round-Robin Rotation (1, 2, 3, 4, q, e, r hotbar, anti-sticky '2' bias, combo chains)
3. Target Acquisition & Validation (TAB cycling, soft-aim, monster death loot+TAB, ExemplarMemory validation)
4. Auto-Potions & Vitals Management (calibrated ROIs, HP/MP reading, potion triggers, panic defense)
5. Combat Isolation & Mode Gate (zero skill leakage in Toplama when fight_back=False, clean combat defense)
6. Line-of-Sight (LoS) & Range Recovery (damage stagnation, raycast obstruction, gap-close W, strafe A/D)
"""

from __future__ import annotations

import collections
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    segment_intersects_rect,
)
from core.combat_recovery import CombatRecovery
from core.game_action_bus import GameActionBus
from core.mana_phase_combat import (
    AutoPotionResult,
    DEFAULT_BURST_INTERVAL_SEC,
    DEFAULT_COMBAT_SKILL_KEYS,
    DEFAULT_GCD_DELAY_SEC,
    ManaPhase,
    ManaPhaseCombat,
    ManaPhaseDecision,
)
from core.skill_bar_reader import SkillBarReader
from core.state_machine import BotMode, BotState, BotWorker


# =============================================================================
# Mock Helpers for Hardware Input, Vision, and Action Bus
# =============================================================================

class MockInputDriver:
    """Mock input driver tracking key presses, holds, and mouse clicks."""

    def __init__(self, trusted: bool = True):
        self._trusted = bool(trusted)
        self.pressed_keys: List[Tuple[str, float]] = []
        self.pressed: List[str] = []
        self.mouse_clicks: List[str] = []
        self.held_mouse_buttons: Dict[str, bool] = {"left": False, "right": False}
        self.down_keys: List[str] = []
        self.up_keys: List[str] = []

    def is_keyboard_trusted(self) -> bool:
        return self._trusted

    def keyboard_inject_trusted(self) -> bool:
        return self._trusted

    def press_key(self, key: str, min_hold: float = 0.03, max_hold: float = 0.05, **kwargs) -> bool:
        k = str(key).lower()
        hold = float(kwargs.get("hold_sec", (min_hold + max_hold) / 2.0))
        self.pressed_keys.append((k, hold))
        self.pressed.append(k)
        return True

    def send_key(self, key: str) -> bool:
        return self.press_key(key)

    def human_click(self, button: str = "left", **kwargs) -> bool:
        b = str(button).lower()
        self.mouse_clicks.append(b)
        return True

    def _hw_mouse_down(self, button: str) -> None:
        self.held_mouse_buttons[str(button).lower()] = True

    def _hw_mouse_up(self, button: str) -> None:
        self.held_mouse_buttons[str(button).lower()] = False

    def clear(self) -> None:
        self.pressed_keys.clear()
        self.pressed.clear()
        self.mouse_clicks.clear()
        self.down_keys.clear()
        self.up_keys.clear()


class MockVisionEngine:
    """Mock vision engine for frame grabbing and vital detection."""

    def __init__(self, frame: Optional[np.ndarray] = None):
        self.frame = frame

    def grab_screen(self, grayscale: bool = False) -> Optional[np.ndarray]:
        return self.frame

    def read_player_hp_mp_bars(self, frame: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return {"hp_pct": 100.0, "mp_pct": 100.0, "target_hp_pct": 0.0, "has_target": False}


# =============================================================================
# 1. SUB-100MS BURST ROTATION TESTS
# =============================================================================

class TestSub100msBurstRotation:
    """Verifies sub-100ms skill burst interval, GCD delays, and deterministic pacing."""

    def test_default_sub_100ms_timing_constants(self):
        """Default burst interval must be <= 0.08s (80ms) and GCD <= 0.12s (120ms)."""
        combat = ManaPhaseCombat()
        assert combat.burst_skill_interval_sec <= 0.08
        assert combat.gcd_delay_sec <= 0.12
        assert DEFAULT_BURST_INTERVAL_SEC <= 0.08
        assert DEFAULT_GCD_DELAY_SEC <= 0.12

    def test_burst_gate_timing_precision(self):
        """
        Gate must permit casts at >= 80ms and block casts < 80ms.
        Must operate without hidden random Gaussian lag.
        """
        combat = ManaPhaseCombat(burst_skill_interval_sec=0.08)
        t_zero = 500.0

        # First cast should be permitted
        assert combat.should_cast_skill(now=t_zero) is True

        # At 30ms later (too fast), must be blocked
        assert combat.should_cast_skill(now=t_zero + 0.030) is False

        # At 75ms later (still sub-80ms), must be blocked
        assert combat.should_cast_skill(now=t_zero + 0.075) is False

        # At 81ms later (>= 80ms), must pass cleanly
        assert combat.should_cast_skill(now=t_zero + 0.081) is True

    def test_state_machine_burst_knobs_trusted_hardware(self):
        """
        In BotWorker, trusted Interception/HID input must configure burst interval
        to 0.08s and per_burst to 4 (multi-key ms burst).
        """
        worker = BotWorker.__new__(BotWorker)
        driver = MockInputDriver(trusted=True)
        worker.input_driver = driver
        # F-1458: no lambda stand-in — the real _keyboard_trusted_for_combat()
        # must read trust from the driver (keyboard_inject_trusted()).
        assert worker._keyboard_trusted_for_combat() is True
        worker.config = {}

        assert worker._combat_skill_burst_sec() == 0.08
        assert worker._combat_skills_per_burst() == 4
        assert worker._combat_skill_cooldown_sec("1") == 0.12

    def test_state_machine_burst_knobs_untrusted_driver(self):
        """
        S-7101: the UNTRUSTED branch of the safety gate must be pinned too.
        With MockInputDriver(trusted=False) the gate must report False and
        the slow (soft) knobs must apply: burst 0.15s, per_burst 2, hold
        0.06-0.10s, cooldown 0.18s (core/state_machine.py:2069, :2084,
        :2087-2088, :2107-2110). Previously every driver in this file was
        trusted=True, so a gate stuck on True (or a driver always reporting
        trusted) was invisible here.
        """
        worker = BotWorker.__new__(BotWorker)
        driver = MockInputDriver(trusted=False)
        worker.input_driver = driver
        # No lambda stand-in — the real _keyboard_trusted_for_combat()
        # must read trust from the driver (keyboard_inject_trusted()).
        assert worker._keyboard_trusted_for_combat() is False
        worker.config = {}

        assert worker._combat_skill_burst_sec() == 0.15
        assert worker._combat_skills_per_burst() == 2
        assert worker._combat_skill_hold_range() == (0.06, 0.10)
        assert worker._combat_skill_cooldown_sec("1") == 0.18

    def test_deterministic_rapid_burst_sequence_without_stall(self):
        """
        Simulate 50 burst cycles at 0.08s (80ms) intervals.
        Ensure every cast succeeds without deadlock or exponential stall.
        """
        combat = ManaPhaseCombat(burst_skill_interval_sec=0.08)
        combat.reset_rotation()

        simulated_t = 1000.0
        success_casts = 0

        for _ in range(50):
            can_cast = combat.should_cast_skill(now=simulated_t)
            assert can_cast is True, f"Failed to cast at timestamp {simulated_t}"
            decision = combat.decide(player_mp=1.0, player_hp=1.0, now=simulated_t)
            assert decision.phase == ManaPhase.BURST
            assert len(decision.skill_keys) >= 2
            combat.record_skill_cast(decision.skill_keys[0], now=simulated_t)
            success_casts += 1
            simulated_t += 0.085  # Advance 85ms (> 80ms)

        assert success_casts == 50

    def test_custom_burst_delay_reconfiguration(self):
        """Dynamic reconfiguration of burst delay (e.g. 0.05s / 0.15s)."""
        combat = ManaPhaseCombat(burst_skill_interval_sec=0.08)
        combat.set_burst_delays(burst_interval_sec=0.05, gcd_delay_sec=0.10)
        assert combat.burst_skill_interval_sec == 0.05
        assert combat.gcd_delay_sec == 0.10

        t = 2000.0
        assert combat.should_cast_skill(now=t) is True
        assert combat.should_cast_skill(now=t + 0.04) is False
        assert combat.should_cast_skill(now=t + 0.052) is True


# =============================================================================
# 2. 8-KEY ROUND-ROBIN ROTATION TESTS
# =============================================================================

class Test8KeyRoundRobinRotation:
    """Verifies round-robin rotation across 8 keys (1, 2, 3, 4, q, e, r, t) and anti-sticky '2'."""

    def test_all_8_hotbar_keys_configured(self):
        """Default hotbar keys must encompass all 8 keys."""
        expected = ["1", "2", "3", "4", "q", "e", "r", "t"]
        combat = ManaPhaseCombat()
        assert combat.combat_skill_keys == expected
        assert list(DEFAULT_COMBAT_SKILL_KEYS) == expected

        worker = BotWorker.__new__(BotWorker)
        worker.config = {}
        worker.combat_recovery = None
        worker.skill_bar_monitor = None
        assert worker._configured_combat_skill_keys() == expected

    def test_round_robin_slice_progression_across_all_keys(self):
        """
        Slices of size 3 must advance cleanly across all 8 keys:
        Slice 1: ['1', '2', '3']
        Slice 2: ['4', 'q', 'e']
        Slice 3: ['r', 't', '1']
        Slice 4: ['2', '3', '4']
        Slice 5: ['q', 'e', 'r']
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        assert combat.get_round_robin_burst_slice(count=3, allow_combo=False) == ["1", "2", "3"]
        assert combat.get_round_robin_burst_slice(count=3, allow_combo=False) == ["4", "q", "e"]
        assert combat.get_round_robin_burst_slice(count=3, allow_combo=False) == ["r", "t", "1"]
        assert combat.get_round_robin_burst_slice(count=3, allow_combo=False) == ["2", "3", "4"]
        assert combat.get_round_robin_burst_slice(count=3, allow_combo=False) == ["q", "e", "r"]

    def test_zero_sticky_2_bias_empirical_distribution(self):
        """
        Simulate 80 combat decisions in BURST phase.
        Every key in ('1', '2', '3', '4', 'q', 'e', 'r', 't') must be primary exactly 10 times.
        Key '2' must NEVER monopolize or exceed fair share.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        counts = collections.Counter()
        t = 100.0

        for _ in range(80):
            decision = combat.decide(player_mp=1.0, player_hp=1.0, now=t)
            primary = decision.skill_keys[0]
            counts[primary] += 1
            combat.record_skill_cast(primary, now=t)
            t += 0.085

        for k in combat.combat_skill_keys:
            assert counts[k] == 10, f"Key '{k}' expected 10 casts, got {counts[k]}"
        assert counts["2"] == 10

    def test_single_key_ready_vision_does_not_trap_rotation(self):
        """
        If vision reports only ready_skills=['2'], combat must NOT stick on '2' endlessly.
        It must blend with fallback hotbar keys and cycle.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()

        primaries = []
        t = 50.0

        for _ in range(14):
            dec = combat.decide(player_mp=1.0, player_hp=1.0, ready_skills=["2"], now=t)
            prim = dec.skill_keys[0]
            primaries.append(prim)
            combat.record_skill_cast(prim, now=t)
            t += 0.085

        distinct = set(primaries)
        assert len(distinct) >= 5, f"Expected varied skills, got {distinct}"
        assert primaries.count("2") <= 4, f"Key '2' was cast too often: {primaries}"

    def test_combo_chain_integration_and_expiry(self):
        """
        Registered combo chain (e.g. '1' -> ['2', '3']) takes precedence when unexpired,
        then smoothly returns to normal round-robin sequence upon completion or timeout.
        """
        combat = ManaPhaseCombat()
        combat.reset_rotation()
        combat.register_combo("1", ["2", "3"])

        t = 100.0
        # Cast trigger '1'
        combat.record_skill_cast("1", now=t)

        # Next slice should prioritize combo step '2'
        s1 = combat.get_round_robin_burst_slice(count=3, allow_combo=True, now=t + 0.1)
        assert s1[0] == "2"

        # Cast '2' -> should advance combo to '3'
        combat.record_skill_cast("2", now=t + 0.1)
        s2 = combat.get_round_robin_burst_slice(count=3, allow_combo=True, now=t + 0.2)
        assert s2[0] == "3"

        # Cast '3' -> finishes combo
        combat.record_skill_cast("3", now=t + 0.2)
        assert combat.get_next_combo_skill(now=t + 0.3) is None

    def test_state_machine_send_configured_combat_skill_cycles(self):
        """
        Tests BotWorker._send_configured_combat_skill() cycling keys into driver.
        """
        from PyQt6.QtCore import QThread
        worker = BotWorker.__new__(BotWorker)
        QThread.__init__(worker)
        worker._lock = threading.RLock()
        driver = MockInputDriver(trusted=True)
        worker.input_driver = driver
        # S-2901: no lambda stand-in — the real _keyboard_trusted_for_combat()
        # must read trust from the driver (keyboard_inject_trusted()).
        assert worker._keyboard_trusted_for_combat() is True
        worker._mode = BotMode.COMBAT_MODE.value
        worker._state = BotState.FIGHTING.value
        worker._combat_was_targeting = True
        worker.config = {"test_mode": False}
        worker._paused = False
        worker._combat_skill_used_at = {}
        worker._combat_skill_idx = 0
        worker._combat_last_skill_burst_ts = 0.0

        fired_keys = []
        now = 1000.0

        # Speed (was ~2.5s): each burst burns 3 x 0.035s inter-key settle
        # sleeps (time.sleep inside _send_configured_combat_skill). Key
        # selection is driven by the patched time.time, not the sleeps —
        # verified: the fired sequence stays exactly 4 x
        # ['1','2','3','4','q','e','r','t'] in order.
        with patch("time.sleep", return_value=None):
            for _ in range(8):
                now += 0.150  # Advance past burst gap (0.08s) and GCD floor (0.12s)
                with patch("core.state_machine.time.time", return_value=now):
                    ok = worker._send_configured_combat_skill()
                    assert ok is True
                    fired_keys.extend(list(driver.pressed))
                    driver.clear()

        # All 8 keys should be represented across bursts
        assert set(fired_keys) == {"1", "2", "3", "4", "q", "e", "r", "t"}


# =============================================================================
# 3. TARGET ACQUISITION & VALIDATION TESTS
# =============================================================================

class TestTargetAcquisitionAndValidation:
    """Verifies TAB search cadence, soft-aim, loot+TAB upon death, and ExemplarMemory validation."""

    def test_tab_search_cadence_when_no_target(self):
        """
        When no target is detected, TAB must be pressed to acquire hostiles.
        Consecutive search steps increment no_target_streak and respect cadence.
        """
        combat_sys = CombatSystem(vision=MockVisionEngine(), driver=MockInputDriver())
        driver = combat_sys.driver

        # Frame with no enemy (black canvas)
        empty_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        t_start = 1000.0
        with patch("core.combat.time.time", return_value=t_start):
            res = combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)
            assert res is False
            assert "tab" in driver.pressed
            driver.clear()

        # Second step within 0.5s: must NOT spam TAB (cadence check >= 1.2s in combat.py)
        with patch("core.combat.time.time", return_value=t_start + 0.4):
            res2 = combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)
            assert res2 is False
            assert "tab" not in driver.pressed

        # Step after 1.3s: TAB should fire again
        with patch("core.combat.time.time", return_value=t_start + 1.3):
            combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)
            assert "tab" in driver.pressed

    def test_target_death_triggers_loot_f_and_instant_tab(self):
        """
        When fighting was active and target dies (enemy bar gone for 2+ steps),
        the system must immediately press 'F' (loot) and 'TAB' (next monster).
        """
        combat_sys = CombatSystem(vision=MockVisionEngine(), driver=MockInputDriver())
        driver = combat_sys.driver
        combat_sys.was_fighting = True

        empty_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Step 1: streak = 1
        combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)
        assert combat_sys.no_target_streak == 1
        driver.clear()

        # Step 2: streak = 2 -> Target dead! Loot F and TAB
        with patch("core.combat.time.sleep", return_value=None):
            combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)

        assert combat_sys.was_fighting is False
        assert "f" in driver.pressed
        assert "tab" in driver.pressed

    def test_monster_target_validation_with_real_exemplar_memory(self):
        """
        is_valid_monster_target() verified against a REAL ExemplarMemory
        (tmp JSON store, real crops -> real feature vectors), not a hand-made
        mock: monster_3d exemplar validates (conf >= 0.85), a negative
        exemplar rejects, and a resource exemplar (herb) is not a monster.
        """
        from dataset_factory.exemplar_memory import ExemplarMemory

        combat = ManaPhaseCombat()
        rng = np.random.default_rng(7)
        patch_crop = rng.integers(0, 256, size=(50, 50, 3), dtype=np.uint8)
        fake_frame = np.full((200, 200, 3), 120, dtype=np.uint8)
        fake_frame[10:60, 10:60] = patch_crop

        def _real_memory(tmpdir, kind):
            mem = ExemplarMemory(json_path=Path(tmpdir) / "exemplars.json")
            if kind == "monster":
                assert mem.add_positive_exemplar(patch_crop, 10, "monster_3d") is True
            elif kind == "negative":
                assert mem.add_negative_exemplar(patch_crop, 10, "monster_3d") is True
            elif kind == "herb":
                assert mem.add_positive_exemplar(patch_crop, 0, "herb_aria") is True
            return mem

        # Case 1: real positive monster exemplar -> valid, conf >= 0.85
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = _real_memory(tmpdir, "monster")
            with patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=mem):
                is_valid, conf = combat.is_valid_monster_target(fake_frame, bbox=(10, 10, 50, 50))
                assert is_valid is True
                assert conf >= 0.85

        # Case 2: real negative exemplar (rejected rock/false target) -> invalid
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = _real_memory(tmpdir, "negative")
            with patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=mem):
                is_valid, conf = combat.is_valid_monster_target(fake_frame, bbox=(10, 10, 50, 50))
                assert is_valid is False
                assert conf <= 0.20

        # Case 3: real resource exemplar (herb), not a monster -> invalid
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = _real_memory(tmpdir, "herb")
            with patch("dataset_factory.exemplar_memory.get_exemplar_memory", return_value=mem):
                is_valid, conf = combat.is_valid_monster_target(fake_frame, bbox=(10, 10, 50, 50))
                assert is_valid is False

    def test_filter_monster_targets_ranks_by_confidence(self):
        """filter_monster_targets ranks candidate bounding boxes descending by confidence."""
        combat = ManaPhaseCombat()
        frame = np.full((300, 300, 3), 100, dtype=np.uint8)

        cands = [
            {"bbox": (10, 10, 30, 30), "id": "m1"},
            {"bbox": (50, 50, 30, 30), "id": "m2"},
        ]

        def fake_valid(f, bbox=None, **kw):
            b = bbox.get("bbox") if isinstance(bbox, dict) else bbox
            if b == (10, 10, 30, 30):
                return True, 0.78
            return True, 0.94

        with patch.object(combat, "is_valid_monster_target", side_effect=fake_valid):
            ranked = combat.filter_monster_targets(frame, cands)
            assert len(ranked) == 2
            assert ranked[0]["id"] == "m2"
            assert ranked[1]["id"] == "m1"

    def test_combat_has_hostile_lock_on_rendered_target_bar(self):
        """
        Verify hostile target bar detection from rendered HSV frame with red/olive target bar.
        """
        recovery = CombatRecovery()
        # Generate 720p frame with a bright red enemy target bar at top-center (x: 40%..60%, y: 5%..8%)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Red bar in BGR: (30, 30, 220) -> high Red, low Blue/Green
        cv2.rectangle(frame, (int(1280 * 0.40), int(720 * 0.05)), (int(1280 * 0.60), int(720 * 0.08)), (30, 30, 220), -1)

        # Tek karelik bar girişi sayılmaz — kilit 2. ardışık karede doğrulanır.
        assert recovery._detect_enemy_target_bar(frame.copy()) is False
        # İkinci kare gerçekten farklı içerikli olmalı — .copy() içerik
        # parmak iziyle "aynı kare" sayılır ve doğrulama ilerlemez.
        frame2 = frame.copy()
        frame2[0, 0] = (1, 0, 0)
        has_bar = recovery._detect_enemy_target_bar(frame2)
        assert has_bar is True


# =============================================================================
# 4. AUTO-POTIONS & VITALS MANAGEMENT TESTS
# =============================================================================

class TestAutoPotionsAndVitalsManagement:
    """Verifies calibrated vitals reading, ManaPhase switching, and auto-potions."""

    def test_calibrated_roi_bounds_retrieval(self):
        """get_calibrated_roi_bounds returns all required vitals ROIs."""
        combat = ManaPhaseCombat()
        rois = combat.get_calibrated_roi_bounds()
        assert "player_hp_bar" in rois
        assert "player_mp_bar" in rois
        assert "target_hp_bar" in rois
        assert "skill_bar" in rois

        for name in ("player_hp_bar", "player_mp_bar", "target_hp_bar"):
            rect = rois[name]
            assert rect is not None
            assert len(rect) == 4
            assert 0.0 <= rect[0] < rect[2] <= 1.0
            assert 0.0 <= rect[1] < rect[3] <= 1.0

    def test_read_vitals_from_frame(self):
        """read_vitals_from_frame computes HP/MP and target HP from synthetic frame."""
        combat = ManaPhaseCombat()
        h, w = 720, 1280
        frame = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw full player HP bar (red in BGR) at default ROI (x: 0.345..0.485, y: 0.880..0.915)
        x1, x2 = int(w * 0.345), int(w * 0.485)
        y1, y2 = int(h * 0.880), int(h * 0.915)
        frame[y1:y2, x1:x2] = (30, 30, 220)

        # Draw half player MP bar (blue in BGR) at default ROI (x: 0.515..0.585, y: 0.880..0.915)
        mx1, mx2 = int(w * 0.515), int(w * 0.585)
        my1, my2 = int(h * 0.880), int(h * 0.915)
        frame[my1:my2, mx1:mx2] = (220, 180, 20)

        vitals = combat.read_vitals_from_frame(frame)
        assert vitals["player_hp"] > 0.85
        assert 0.40 <= vitals["player_mp"] <= 0.65

    def test_mana_phase_transitions_regen_and_burst(self):
        """
        When MP < 0.95 -> REGEN phase (hold_left=True, hold_right=False).
        REGEN is sticky: exit needs MP >= 0.99 (D3 hysteresis), so 0.96
        stays in REGEN; (nearly) full MP -> BURST phase
        (hold_left=False, hold_right=True).
        """
        combat = ManaPhaseCombat(mp_full_threshold=0.95)

        # Initial state full MP -> BURST
        d1 = combat.decide(player_mp=1.0, player_hp=1.0)
        assert d1.phase == ManaPhase.BURST
        assert d1.hold_right is True
        assert d1.hold_left is False

        # Drop MP to 0.40 -> switches to REGEN
        d2 = combat.decide(player_mp=0.40, player_hp=1.0)
        assert d2.phase == ManaPhase.REGEN
        assert d2.hold_left is True
        assert d2.hold_right is False
        assert "mp=" in d2.reason

        # MP recovers to 0.80 -> stays in REGEN (sticky threshold)
        d3 = combat.decide(player_mp=0.80, player_hp=1.0)
        assert d3.phase == ManaPhase.REGEN

        # MP reaches 0.96 -> STILL in REGEN: hysteresis exit is 0.99 (D3),
        # so the entry threshold (0.95) alone does not re-enter BURST.
        d4 = combat.decide(player_mp=0.96, player_hp=1.0)
        assert d4.phase == ManaPhase.REGEN
        assert d4.hold_left is True

        # MP reaches (nearly) full -> returns to BURST
        d5 = combat.decide(player_mp=1.0, player_hp=1.0)
        assert d5.phase == ManaPhase.BURST
        assert d5.hold_right is True

    def test_panic_defense_priority_on_low_hp(self):
        """
        When player HP < hp_panic_threshold (e.g. 0.50), defense/heal skill is placed first.
        """
        combat = ManaPhaseCombat(hp_panic_threshold=0.50)
        combat.set_lexicon({"r": "defense_heal", "1": "attack", "2": "attack"})
        combat.reset_rotation()

        # Low HP (0.30)
        dec = combat.decide(player_mp=1.0, player_hp=0.30)
        assert dec.phase == ManaPhase.BURST
        assert dec.skill_keys[0] == "r"
        assert "panic" in dec.reason

    def test_auto_potion_thresholds_and_cooldown(self):
        """
        HP potion triggered when HP < 0.50.
        MP potion triggered when MP < 0.30.
        Subsequent calls within cooldown period are blocked.
        """
        combat = ManaPhaseCombat(
            hp_potion_threshold=0.50,
            mp_potion_threshold=0.30,
            hp_potion_key="7",
            mp_potion_key="8",
            hp_potion_cd_sec=10.0,
            mp_potion_cd_sec=10.0,
        )

        t = 1000.0
        # F-1356: no hasattr/if guard — if check_auto_potions is ever removed
        # this test must fail loudly instead of passing with zero assertions.
        assert hasattr(combat, "check_auto_potions")
        # Normal vitals: no potions
        p0 = combat.check_auto_potions(player_hp=0.80, player_mp=0.80, now=t)
        assert p0.hp_potion_triggered is False
        assert p0.mp_potion_triggered is False

        # Low HP -> triggers HP potion '7', but only once the drop is
        # confirmed (v20 madde 6, `_hp_low_reading_confirmed`): the
        # first low sample after a healthy one is an unconfirmed cliff
        # (single-frame misreads read 0.00 while full) and must not
        # trigger; the second consecutive low sample confirms it.
        p1 = combat.check_auto_potions(player_hp=0.40, player_mp=0.80, now=t)
        assert p1.hp_potion_triggered is False
        assert p1.hp_potion_key is None

        # P0-6: karar tek başına damgalamaz — gönderim onayı olmadan
        # hemen yeniden karar da tetikler. Bu ikinci düşük örnek aynı
        # zamanda düşüşü doğrulayan örnektir: iksir burada tetiklenir.
        p_confirmed = combat.check_auto_potions(player_hp=0.30, player_mp=0.80, now=t + 1.0)
        assert p_confirmed.hp_potion_triggered is True
        assert "7" in p_confirmed.keys_to_press
        assert p_confirmed.hp_potion_key == "7"

        # t + 1.0'daki basım onaylanınca cooldown başlar.
        combat.mark_potions_sent([p_confirmed.hp_potion_key], hp_key=p_confirmed.hp_potion_key, now=t + 1.0)

        # Immediate repeat at t + 1.0s -> blocked by cooldown
        p2 = combat.check_auto_potions(player_hp=0.30, player_mp=0.80, now=t + 1.0)
        assert p2.hp_potion_triggered is False

        # After 11.0s -> permitted again
        p3 = combat.check_auto_potions(player_hp=0.30, player_mp=0.80, now=t + 11.0)
        assert p3.hp_potion_triggered is True


# =============================================================================
# 5. COMBAT ISOLATION & MODE GATE TESTS
# =============================================================================

class TestCombatIsolationAndModeGate:
    """Verifies mode isolation: zero skill key leakage in Toplama when fight_back=False."""

    @staticmethod
    def _create_worker(mode: str = BotMode.ROUTE_MODE.value, fight_back: bool = False, driver: Any = None) -> BotWorker:
        from PyQt6.QtCore import QThread
        worker = BotWorker.__new__(BotWorker)
        QThread.__init__(worker)
        worker._lock = threading.RLock()
        worker._paused = False
        worker._mode = mode
        worker._state = BotState.WALKING.value
        worker._in_step_combat_mode = False
        worker._is_gathering_active = False
        worker._combat_was_targeting = False
        worker._combat_no_target_streak = 0
        worker._route_prev_hp_ratio = 1.0
        worker._route_combat_streak = 0
        worker._route_saw_target_bar = False
        worker._combat_skill_used_at = {}
        worker._combat_skill_idx = 0
        worker._combat_last_skill_burst_ts = 0.0
        worker.config = {
            "test_mode": False,
            "fight_back": fight_back,
            "combat_skill_keys": ["1", "2", "3", "4", "q", "e", "r"],
        }
        worker.input_driver = driver or MockInputDriver(trusted=True)
        worker._get_cfg = lambda k, d=None: worker.config.get(k, d)
        # S-2902: no lambda stand-in — the real _keyboard_trusted_for_combat()
        # reads trust from the driver (keyboard_inject_trusted()); the helper
        # installs a trusted MockInputDriver by default, so the gate stays True.
        worker._grab_frame = lambda grayscale=False: np.zeros((720, 1280, 3), dtype=np.uint8)
        worker._sync_combat_left_button = MagicMock()
        worker._sync_combat_right_button = MagicMock()
        worker._release_combat_left_button = MagicMock()
        worker._release_combat_right_button = MagicMock()
        worker._emit_hud_snapshot = MagicMock()
        worker._emit_log = MagicMock()
        worker._set_state_internal = MagicMock()
        worker._zk_post_combat_tick = MagicMock()
        worker._maybe_run_skill_discovery = MagicMock()
        worker._player_driven_mapping_mode = lambda: False
        worker.combat_recovery = MagicMock()
        worker.combat_recovery.check_character_dead.return_value = False
        worker.combat_recovery.check_under_attack.return_value = False
        return worker

    def test_zero_skill_leakage_in_route_mode_when_fight_back_disabled(self):
        """
        In ROUTE_MODE (Toplama) with fight_back=False:
        Even when taking damage or near an enemy, NO combat skill keys (1,2,3,4,q,e,r) are pressed.
        """
        driver = MockInputDriver(trusted=True)
        worker = self._create_worker(mode=BotMode.ROUTE_MODE.value, fight_back=False, driver=driver)

        mock_cr = MagicMock()
        mock_cr.check_character_dead.return_value = False
        mock_cr.check_under_attack.return_value = (True, 0.70)
        worker.combat_recovery = mock_cr

        # Phase A — fight_back=False: gate short-circuits, nothing handled.
        handled = worker._handle_combat_and_death()
        assert handled is False

        # Zero combat keys must have leaked into driver
        combat_keys = {"1", "2", "3", "4", "q", "e", "r"}
        pressed_set = set(driver.pressed)
        assert not (pressed_set & combat_keys), f"Combat skills leaked into Toplama: {pressed_set & combat_keys}"

        # Phase B (F-1459) — the SAME (True, 0.70) damage signal must actually
        # reach production: with fight_back=True the worker reads it (handled
        # True) yet route mode still suppresses every combat skill key.
        # 2-kare kuralı (core/state_machine.py _route_combat_signal_confirmed,
        # route_fight_confirm_frames=2): tick 1 only primes the streak,
        # tick 2 confirms — handled is True only on the 2nd tick.
        worker.config["fight_back"] = True
        driver.clear()
        mock_cr.reset_mock()
        mock_cr.check_character_dead.return_value = False
        mock_cr.check_under_attack.return_value = (True, 0.70)
        worker.combat_recovery = mock_cr

        handled_b_first = worker._handle_combat_and_death()
        assert handled_b_first is False
        handled_b = worker._handle_combat_and_death()
        assert handled_b is True
        assert mock_cr.check_under_attack.call_count >= 2, "damage signal never reached production"
        assert mock_cr.fight_back_step.call_count == 0, "fight_back_step must stay suppressed in Toplama"
        pressed_b = set(driver.pressed)
        assert not (pressed_b & combat_keys), f"Combat skills leaked into Toplama: {pressed_b & combat_keys}"

    def test_fight_back_engages_cleanly_when_enabled_in_route_mode(self):
        """
        In ROUTE_MODE with fight_back=True:
        When attacked and damage is detected, fight state is acknowledged.
        """
        driver = MockInputDriver(trusted=True)
        worker = self._create_worker(mode=BotMode.ROUTE_MODE.value, fight_back=True, driver=driver)

        mock_cr = MagicMock()
        mock_cr.check_character_dead.return_value = False
        # Under attack with HP drop (from 1.0 to 0.70).
        # 2-kare kuralı (core/state_machine.py _route_combat_signal_confirmed,
        # route_fight_confirm_frames=2): tick 1 only primes the streak,
        # tick 2 confirms — handled is True only on the 2nd tick.
        mock_cr.check_under_attack.return_value = (True, 0.70)
        worker.combat_recovery = mock_cr

        handled_first = worker._handle_combat_and_death()
        assert handled_first is False
        handled = worker._handle_combat_and_death()
        assert handled is True

    def test_dedicated_combat_mode_active_execution(self):
        """
        In COMBAT_MODE, combat step executes targeting and rotation actively.
        """
        driver = MockInputDriver(trusted=True)
        worker = self._create_worker(mode=BotMode.COMBAT_MODE.value, fight_back=True, driver=driver)
        worker._state = BotState.FIGHTING.value

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        worker._grab_frame = lambda grayscale=False: frame
        worker._combat_has_hostile_lock = lambda f: True
        worker._combat_was_targeting = True

        # F-1304: no stub — the real _send_configured_combat_skill() must run
        # and put real keys into the driver. COMBAT_MODE + hostile lock means
        # mouse spam fires 't' first, then the first 4-key burst 1,2,3,4.
        worker._step_combat_mode()
        assert driver.pressed == ["t", "1", "2", "3", "4"]
        assert worker._combat_was_targeting is True

    def test_action_bus_exclusive_ui_protection(self):
        """
        GameActionBus exclusive lock releases mouse buttons and prevents input interference.
        """
        bus = GameActionBus(input_driver=MockInputDriver(), mouse_look_mode=True)
        assert bus._exclusive_depth == 0

        bus.begin_exclusive("test_interaction")
        assert bus._exclusive_depth == 1
        assert bus._exclusive_reason == "test_interaction"

        bus.end_exclusive()
        assert bus._exclusive_depth == 0
        assert bus._exclusive_reason == ""


# =============================================================================
# 6. LINE-OF-SIGHT (LOS) & RANGE RECOVERY TESTS
# =============================================================================

class TestLineOfSightAndRangeRecovery:
    """Verifies geometric raycast obstruction, damage stagnation, gap-close W, and strafe A/D."""

    def test_geometric_raycast_intersection_detection(self):
        """
        segment_intersects_rect correctly detects line segment crossing obstacle bounding box.
        """
        # Segment from (100, 500) to (900, 500) passes through rect at (400, 450, 100, 100)
        p1 = (100.0, 500.0)
        p2 = (900.0, 500.0)
        obstructing_rect = (400.0, 450.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, obstructing_rect) is True

        # Non-intersecting rect far above segment
        clear_rect = (400.0, 200.0, 100.0, 100.0)
        assert segment_intersects_rect(p1, p2, clear_rect) is False

    def test_damage_stagnation_triggers_los_recovery(self):
        """
        When 3+ consecutive casts deal 0 target HP damage, LineOfSightTracker detects
        damage stagnation and recommends tactical recovery (flank/strafe).
        """
        tracker = LineOfSightTracker(CombatPositioningConfig(damage_stagnation_threshold=3))

        # S-1303: first reading is the baseline only — no damage verdict is
        # possible without a previous reading, so it neither deals damage
        # nor moves the stagnation counter.
        damage_dealt = tracker.record_cast_outcome(target_hp_pct=95.0, now=100.0)
        assert damage_dealt is False
        assert tracker.consecutive_stagnant_casts == 0

        # Cast 1 with no HP drop (still 95%) -> stagnant = 1
        tracker.record_cast_outcome(target_hp_pct=95.0, now=100.2)
        assert tracker.consecutive_stagnant_casts == 1

        # Cast 2 with no HP drop (still 95%) -> stagnant = 2
        tracker.record_cast_outcome(target_hp_pct=95.0, now=100.4)
        assert tracker.consecutive_stagnant_casts == 2

        # Cast 3 with no HP drop (still 95%) -> reaches threshold 3!
        tracker.record_cast_outcome(target_hp_pct=95.0, now=100.6)
        assert tracker.consecutive_stagnant_casts == 3

        los = tracker.check_line_of_sight()
        assert los.has_los is False
        assert los.obstruction_type == ObstructionType.DAMAGE_STAGNATION
        assert los.flank_direction in ("left", "right")

    def test_range_evaluation_too_far_recommends_step_forward(self):
        """
        When target is beyond melee range (> 3.8m), CombatPositioner evaluates TOO_FAR
        and recommends 'step_forward' (key 'W').
        """
        pos = CombatPositioner()
        pos.set_profile(EngagementProfile.MELEE)

        eval_res = pos.evaluate_target_range(estimated_distance_m=6.5)
        assert eval_res.status == RangeStatus.TOO_FAR
        assert eval_res.recommended_action == "step_forward"

    def test_range_evaluation_too_close_recommends_backpedal(self):
        """
        When ranged character has target too close (< 4.5m), recommends 'backpedal' (key 'S').
        """
        pos = CombatPositioner()
        pos.set_profile(EngagementProfile.RANGED)

        eval_res = pos.evaluate_target_range(estimated_distance_m=2.5)
        assert eval_res.status == RangeStatus.TOO_CLOSE
        assert eval_res.recommended_action == "backpedal"

    def test_combat_recovery_stuck_maneuvers_execute(self):
        """
        CombatRecovery.recover_from_stuck executes tactical jump + strafe maneuvers.
        """
        recovery = CombatRecovery()
        driver = MockInputDriver(trusted=True)

        # F-1460: attempt=1 is deterministic in production — exactly
        # "JUMP_AND_STRAFE" with space, strafe 'a' + forward 'w', space.
        # A loose membership assertion would also pass ERROR/COMPLETED.
        res = recovery.recover_from_stuck(input_ctrl=driver, attempt=1, fly_unreachable=False)
        assert res == "JUMP_AND_STRAFE"
        assert driver.pressed == ["space", "a", "w", "space"]


# =============================================================================
# 7. END-TO-END 10-AGENT COMBAT BENCHMARK
# =============================================================================

class TestEndToEnd10AgentCombatBenchmark:
    """Integrated scenario testing all 10 agents working together."""

    def test_full_combat_encounter_cycle(self):
        """
        Simulates complete combat lifecycle:
        1. Target search with TAB
        2. Hostile target acquired
        3. Sub-100ms burst rotation across 8 keys
        4. Target damaged from 100% to 0%
        5. Loot F + instant next target TAB
        """
        combat_sys = CombatSystem(vision=MockVisionEngine(), driver=MockInputDriver())
        driver = combat_sys.driver
        mana_brain = ManaPhaseCombat(burst_skill_interval_sec=0.08)
        mana_brain.reset_rotation()

        # Step 1: No target -> TAB
        empty_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)
        assert "tab" in driver.pressed
        driver.clear()

        # Step 2: Target acquired!
        # F-1357 [DOĞRULA]: this frame was dead code in the base (built but
        # never fed to the combat system). It is now actually consumed: the
        # red target bar must drive a real attack step, deterministically
        # (random.choice pinned to the first pool entry, 't').
        enemy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        # Add target bar
        cv2.rectangle(enemy_frame, (int(1280 * 0.4), int(720 * 0.05)), (int(1280 * 0.6), int(720 * 0.08)), (30, 30, 220), -1)

        with patch("core.combat.random.choice", side_effect=lambda pool: pool[0]):
            acquired = combat_sys.fight_back_step(input_ctrl=driver, frame=enemy_frame)
        assert acquired is True
        assert combat_sys.in_combat is True
        assert combat_sys.total_attacks == 1
        assert driver.pressed == ["t"]
        driver.clear()

        sim_t = 100.0
        fired_keys = []

        # Step 3: Run 16 burst ticks (2 full 8-key cycles)
        for _ in range(16):
            if mana_brain.should_cast_skill(now=sim_t):
                dec = mana_brain.decide(player_mp=1.0, player_hp=1.0, now=sim_t)
                prim = dec.skill_keys[0]
                driver.press_key(prim)
                mana_brain.record_skill_cast(prim, now=sim_t)
                fired_keys.append(prim)
            sim_t += 0.085

        # All 8 keys should have fired
        assert set(fired_keys) == {"1", "2", "3", "4", "q", "e", "r", "t"}
        driver.clear()

        # Step 4: Target dies -> Loot and TAB next
        combat_sys.was_fighting = True
        combat_sys.no_target_streak = 1
        with patch("core.combat.time.sleep", return_value=None):
            combat_sys.fight_back_step(input_ctrl=driver, frame=empty_frame)

        assert "f" in driver.pressed
        assert "tab" in driver.pressed


# =============================================================================
# 8. FAILURE MODES: MOCKS THAT CAN FAIL (F-1568)
# =============================================================================

class TestFailureModesDriverAndVision:
    """Mock driver/vision must be able to fail — a mock that always succeeds
    cannot prove production handles failure."""

    def test_vision_blackout_returns_no_target_without_crash(self):
        """Vision returns no frame at all: no crash, no target, no skill keys."""
        combat_sys = CombatSystem(vision=MockVisionEngine(frame=None), driver=MockInputDriver())
        driver = combat_sys.driver
        res = combat_sys.fight_back_step(input_ctrl=driver, frame=None)
        assert res is False
        combat_keys = {"1", "2", "3", "4", "q", "e", "r", "t"}
        assert not (set(driver.pressed) & combat_keys)

    def test_failing_input_driver_error_is_not_silently_swallowed(self):
        """A raising driver must surface, not be logged away as success.

        NOTE: core/combat.py currently has no driver-failure handling, so the
        error propagates out of fight_back_step. This test pins that contract
        explicitly (D22 scope if graceful handling is ever added) instead of
        letting a never-failing mock pretend every press succeeded.
        """

        class FailingDriver(MockInputDriver):
            def press_key(self, key: str, min_hold: float = 0.03, max_hold: float = 0.05, **kwargs) -> bool:
                raise RuntimeError("simulated HID failure")

        combat_sys = CombatSystem(vision=MockVisionEngine(), driver=FailingDriver())
        empty_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="simulated HID failure"):
            combat_sys.fight_back_step(input_ctrl=combat_sys.driver, frame=empty_frame)
