# -*- coding: utf-8 -*-
"""
R06: post-cast HP re-read tests (CombatSystem level).

After the first cast on a (re)targeted enemy, fight_back_step settles briefly
and re-reads target HP through vision, so the first cast can be judged instead
of counting only as a baseline. S-1303 stays: without a previous reading no
verdict is possible — the re-read supplies the second reading.

No real sleeps: core.combat.time.sleep is monkeypatched in every test.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pytest

from core.combat import (
    POSTCAST_HP_SETTLE_SEC,
    CombatSystem,
    EngagementProfile,
)


class ScriptedVision:
    """Returns queued vitals dicts, one per read_player_hp_mp_bars call.

    Call order inside one fight_back_step: check_under_attack consumes the
    first reading, the post-cast re-read (first cast only) the second.
    """

    def __init__(self, readings):
        self._readings = list(readings)
        self.calls = 0

    def read_player_hp_mp_bars(self, frame=None):
        self.calls += 1
        if not self._readings:
            raise AssertionError("ScriptedVision: unexpected extra read")
        item = self._readings.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class RaisingVision:
    """Every vitals read raises (vision present but broken)."""

    def __init__(self):
        self.calls = 0

    def read_player_hp_mp_bars(self, frame=None):
        self.calls += 1
        raise RuntimeError("vision unavailable")


class MockInputDriver:
    """Minimal driver: every press dispatches (returns True)."""

    def __init__(self):
        self.pressed_keys = []

    def press_key(self, key, hold_sec=0.05):
        self.pressed_keys.append((key.lower(), float(hold_sec)))
        return True


def _hostile_frame():
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    cv2.rectangle(frame, (700, 40), (900, 70), (0, 0, 220), -1)
    return frame


# P1-12 migrasyonu: tek formülde melee IN_RANGE için bh >= 273 (1080p);
# bh=300 (norm_h 0.278 -> 3.45 m) menzildedir.
# (Eski bh=200, 5.18 m ile TOO_FAR'dı — gap-close saldırıyı engelliyordu.)
TARGET_BBOX = (900, 350, 150, 300)  # melee range, no obstacles


def _record_sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr("core.combat.time.sleep", lambda sec: calls.append(sec))
    return calls


def _step(combat, driver, frame, target_hp_pct):
    # Pin the skill choice (first pool entry) like the fight_back_step tests.
    with patch("core.combat.random.choice", side_effect=lambda pool: pool[0]):
        return combat.fight_back_step(
            input_ctrl=driver,
            frame=frame,
            target_bbox=TARGET_BBOX,
            target_hp_pct=target_hp_pct,
        )


class TestPostCastHpReRead:
    """R06: first cast after (re)target is judged via a post-cast HP re-read."""

    def test_first_cast_damage_judged_via_postcast_read(self, monkeypatch):
        vision = ScriptedVision(
            [
                {"has_target": True, "target_hp_pct": 100.0},  # check_under_attack
                {"has_target": True, "target_hp_pct": 95.0},  # post-cast re-read
            ]
        )
        driver = MockInputDriver()
        combat = CombatSystem(
            vision=vision, driver=driver, profile=EngagementProfile.MELEE
        )
        sleep_calls = _record_sleeps(monkeypatch)

        result = _step(combat, driver, _hostile_frame(), 100.0)

        assert result is True
        assert combat.total_attacks == 1
        # Damage dealt on the FIRST cast: counter stays 0, latest HP is the
        # post-cast reading, not the step-start one.
        assert combat.los_tracker.consecutive_stagnant_casts == 0
        assert combat.los_tracker.last_target_hp_pct == 95.0
        assert combat.last_target_hp == 95.0
        assert sleep_calls == [POSTCAST_HP_SETTLE_SEC]
        assert vision.calls == 2

    def test_first_cast_stagnation_counted_via_postcast_read(self, monkeypatch):
        vision = ScriptedVision(
            [
                {"has_target": True, "target_hp_pct": 100.0},  # check_under_attack
                {"has_target": True, "target_hp_pct": 100.0},  # post-cast: no damage
            ]
        )
        driver = MockInputDriver()
        combat = CombatSystem(
            vision=vision, driver=driver, profile=EngagementProfile.MELEE
        )
        sleep_calls = _record_sleeps(monkeypatch)

        result = _step(combat, driver, _hostile_frame(), 100.0)

        assert result is True
        assert combat.total_attacks == 1
        # No HP drop after the cast: the first stagnant cast is counted
        # immediately instead of waiting for the second cast.
        assert combat.los_tracker.consecutive_stagnant_casts == 1
        assert combat.last_target_hp == 100.0
        assert sleep_calls == [POSTCAST_HP_SETTLE_SEC]
        assert vision.calls == 2

    def test_postcast_reread_skipped_without_vision(self, monkeypatch):
        driver = MockInputDriver()
        combat = CombatSystem(driver=driver, profile=EngagementProfile.MELEE)
        sleep_calls = _record_sleeps(monkeypatch)

        result = _step(combat, driver, _hostile_frame(), 100.0)

        assert result is True
        assert combat.total_attacks == 1
        # No vision: first cast is the baseline only, no settle paid.
        assert combat.los_tracker.consecutive_stagnant_casts == 0
        assert combat.last_target_hp == 100.0
        assert sleep_calls == []

    def test_postcast_reread_skipped_on_vision_error(self, monkeypatch):
        vision = RaisingVision()
        driver = MockInputDriver()
        combat = CombatSystem(
            vision=vision, driver=driver, profile=EngagementProfile.MELEE
        )
        sleep_calls = _record_sleeps(monkeypatch)

        result = _step(combat, driver, _hostile_frame(), 100.0)

        assert result is True
        assert combat.total_attacks == 1
        # Both reads raise (check falls back to the frame path, post-cast is
        # caught): no second record, counter stays 0. The settle is paid
        # once because the vision guard holds and the sleep precedes the try.
        assert combat.los_tracker.consecutive_stagnant_casts == 0
        assert combat.last_target_hp == 100.0
        assert sleep_calls == [POSTCAST_HP_SETTLE_SEC]
        assert vision.calls == 2

    def test_postcast_reread_skipped_when_no_target(self, monkeypatch):
        vision = ScriptedVision(
            [
                {"has_target": True, "target_hp_pct": 100.0},  # check_under_attack
                {"has_target": False},  # post-cast: target gone
            ]
        )
        driver = MockInputDriver()
        combat = CombatSystem(
            vision=vision, driver=driver, profile=EngagementProfile.MELEE
        )
        sleep_calls = _record_sleeps(monkeypatch)

        result = _step(combat, driver, _hostile_frame(), 100.0)

        assert result is True
        assert combat.total_attacks == 1
        # No target in the post-cast read: no second record, counter stays 0.
        assert combat.los_tracker.consecutive_stagnant_casts == 0
        assert combat.last_target_hp == 100.0
        assert sleep_calls == [POSTCAST_HP_SETTLE_SEC]
        assert vision.calls == 2

    def test_postcast_reread_only_once_per_target(self, monkeypatch):
        vision = ScriptedVision(
            [
                {"has_target": True, "target_hp_pct": 100.0},  # step 1 check
                {"has_target": True, "target_hp_pct": 100.0},  # step 1 post-cast
                {"has_target": True, "target_hp_pct": 100.0},  # step 2 check
                # No 4th reading: step 2 must NOT re-read (would raise).
            ]
        )
        driver = MockInputDriver()
        combat = CombatSystem(
            vision=vision, driver=driver, profile=EngagementProfile.MELEE
        )
        sleep_calls = _record_sleeps(monkeypatch)
        frame = _hostile_frame()

        assert _step(combat, driver, frame, 100.0) is True
        assert combat.los_tracker.consecutive_stagnant_casts == 1
        # GCD guard (skill_gcd_sec = 0.28): pretend enough time passed so the
        # second step casts instead of returning early.
        combat.last_attack_time = 0.0
        assert _step(combat, driver, frame, 100.0) is True
        assert combat.los_tracker.consecutive_stagnant_casts == 2

        assert combat.total_attacks == 2
        # Settle paid exactly once (first cast); second cast re-reads nothing.
        assert sleep_calls == [POSTCAST_HP_SETTLE_SEC]
        assert vision.calls == 3

    def test_late_damage_self_corrects_before_threshold(self, monkeypatch):
        vision = ScriptedVision(
            [
                {"has_target": True, "target_hp_pct": 100.0},  # step 1 check
                {"has_target": True, "target_hp_pct": 100.0},  # step 1 post-cast
                {"has_target": True, "target_hp_pct": 90.0},  # step 2 check
            ]
        )
        driver = MockInputDriver()
        combat = CombatSystem(
            vision=vision, driver=driver, profile=EngagementProfile.MELEE
        )
        sleep_calls = _record_sleeps(monkeypatch)
        frame = _hostile_frame()

        # Step 1: damage not yet reflected in the bar — counted stagnant.
        assert _step(combat, driver, frame, 100.0) is True
        assert combat.los_tracker.consecutive_stagnant_casts == 1

        # Step 2: the step-start reading shows the damage — counter resets
        # before the threshold (3) is reached, no recovery is triggered.
        # GCD guard (skill_gcd_sec = 0.28): pretend enough time passed.
        combat.last_attack_time = 0.0
        assert _step(combat, driver, frame, 90.0) is True
        assert combat.los_tracker.consecutive_stagnant_casts == 0
        assert combat.los_tracker.check_line_of_sight().has_los is True

        assert combat.total_attacks == 2
        assert sleep_calls == [POSTCAST_HP_SETTLE_SEC]
        assert vision.calls == 3
