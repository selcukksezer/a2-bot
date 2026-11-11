# -*- coding: utf-8 -*-
"""
tests/test_combat_continuous_skill_spam.py
=========================================
Unit tests for continuous combat skill spamming and target flicker persistence grace:
1. Verifies that transient target occlusions (floating damage numbers, VFX, screen shake)
   do NOT trigger Tab presses, soft-aim camera sweeps, or 5-second hesitations.
2. Verifies that hotbar skill rotation continues uninterrupted through visual flickers.
3. Verifies that sustained target loss (> 2.5s) correctly transitions to loot ('f') and next-target Tab.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.mana_phase_combat import ManaPhaseCombat
from core.state_machine import BotMode, BotState, BotWorker


@pytest.fixture(autouse=True)
def _isolated_lifetime_stats(tmp_path):
    """DEV-D088: the sustained-target-loss path records a monster kill via
    get_lifetime_stats() (module singleton, default
    config/lifetime_stats.json) — each run incremented the REAL repo file.
    Redirect the lookup to a tmp-backed store; no test here asserts on
    lifetime stats."""
    from core.lifetime_stats import LifetimeStatsStore

    store = LifetimeStatsStore(path=str(tmp_path / "lifetime_stats.json"))
    with patch("core.lifetime_stats.get_lifetime_stats", return_value=store):
        yield


class _MockDriver:
    def __init__(self, trusted: bool = True):
        self.trusted = trusted
        self.pressed: list[str] = []
        self.mouse_events: list[str] = []
        self.held_keys: set[str] = set()
        self.key_events: list[tuple[str, str]] = []

    def keyboard_inject_trusted(self) -> bool:
        return self.trusted

    def press_key(self, key: str, **kwargs) -> bool:
        self.pressed.append(str(key).lower())
        return True

    def key_down(self, key: str, **kwargs) -> bool:
        name = str(key).lower()
        self.held_keys.add(name)
        self.key_events.append(("down", name))
        return True

    def key_up(self, key: str, **kwargs) -> bool:
        name = str(key).lower()
        self.held_keys.discard(name)
        self.key_events.append(("up", name))
        return True

    def send_key(self, key: str, **kwargs) -> bool:
        return self.press_key(key, **kwargs)

    def _hw_mouse_down(self, btn: str = "left") -> None:
        self.mouse_events.append(f"down_{btn}")

    def _hw_mouse_up(self, btn: str = "left") -> None:
        self.mouse_events.append(f"up_{btn}")

    def nudge_camera(self, dx: int, dy: int) -> None:
        self.mouse_events.append(f"nudge_{dx}_{dy}")


def _create_combat_worker(driver: _MockDriver) -> BotWorker:
    import threading
    from PyQt6.QtCore import QThread
    worker = BotWorker.__new__(BotWorker)
    QThread.__init__(worker)
    worker._lock = threading.RLock()
    worker._paused = False
    worker._mode = BotMode.COMBAT_MODE.value
    worker._state = BotState.FIGHTING.value
    worker._running = True
    worker._lock = MagicMock()
    worker._lock.__enter__ = MagicMock(return_value=None)
    worker._lock.__exit__ = MagicMock(return_value=None)

    worker.input_driver = driver
    worker.combat_recovery = MagicMock()
    worker.combat_recovery.check_under_attack.return_value = (False, 1.0)
    worker.combat_recovery.get_player_hp_ratio.return_value = 1.0
    worker.game_learner = None
    worker.skill_discovery = None
    worker.perception_hub = None
    worker.focus_monitor = None
    worker.action_bus = None
    worker.mana_phase_combat = ManaPhaseCombat(burst_skill_interval_sec=0.08)

    worker._combat_was_targeting = True
    worker._combat_no_target_streak = 0
    worker._combat_last_target_ts = time.time()
    worker._combat_tab_miss_count = 0
    worker._combat_last_tab_ts = 0.0
    worker._combat_last_soft_aim_ts = 0.0
    worker._combat_last_skill_burst_ts = 0.0
    worker._combat_skill_idx = 0
    worker._combat_skill_used_at = {}

    worker._emit_log = MagicMock()
    worker._emit_hud_snapshot = MagicMock()
    worker._set_state_internal = MagicMock()
    worker._handle_combat_and_death = MagicMock(return_value=False)
    worker._check_and_trigger_auto_potions = MagicMock()
    # NOTE (F-2048): _spam_combat_mouse / _sync_combat_*_button are deliberately
    # NOT mocked here — they are the production attack path and must be proven
    # to use T/R keys with zero mouse events (see the assertions below).
    worker._zk_post_combat_tick = MagicMock()
    worker._maybe_run_skill_discovery = MagicMock()
    worker._log_combat_skill_knobs_once = MagicMock()
    worker.msleep = MagicMock()

    worker._get_cfg = MagicMock(side_effect=lambda k, d=None: {
        "test_mode": False,
        "combat_skill_keys": ["1", "2", "3", "4", "q", "e", "r"],
        "combat_skills": ["1", "2", "3", "4", "q", "e", "r"],
        "combat_skill_burst_sec": 0.08,
        "skills_per_burst": 4,
        "combat_skill_gcd_sec": 0.12,
    }.get(k, d))

    return worker


def test_transient_target_flicker_sustains_combat_without_tab_or_soft_aim():
    """
    During active combat, if target detection flickers (misses 1-5 frames due to damage text/VFX),
    target persistence grace MUST keep has_enemy=True, NOT press Tab, NOT sweep camera,
    and allow skills to continue firing.
    """
    driver = _MockDriver(trusted=True)
    worker = _create_combat_worker(driver)

    t0 = 1000.0
    worker._combat_last_target_ts = t0

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    worker._grab_frame = lambda grayscale=False: frame
    # Target occlusion on this tick: OpenCV raw detector returns False
    worker._combat_has_hostile_lock = MagicMock(return_value=False)
    worker._press_combat_tab = MagicMock()

    skills_fired: list[str] = []
    worker._cast_one_combat_skill = lambda k, **kw: skills_fired.append(k)

    # Tick 1: 50ms after lock was active (damage number popup)
    with patch("core.state_machine.time.time", return_value=t0 + 0.05):
        worker._step_combat_mode()

    # Tab should NOT be pressed during flicker grace!
    worker._press_combat_tab.assert_not_called()
    assert worker._combat_was_targeting is True
    assert worker._combat_no_target_streak == 1
    # Skills should fire uninterrupted!
    assert len(skills_fired) == 4
    assert skills_fired == ["1", "2", "3", "4"]
    # F-2048: the attack itself must go through the T key (right-click attack
    # = T), never through the mouse — no mouse down/up or camera nudge at all.
    assert "t" in driver.pressed
    assert driver.mouse_events == []

    # Tick 2: 150ms after lock was active (particles still exploding)
    skills_fired.clear()
    with patch("core.state_machine.time.time", return_value=t0 + 0.15):
        worker._step_combat_mode()

    worker._press_combat_tab.assert_not_called()
    assert worker._combat_was_targeting is True
    assert worker._combat_no_target_streak == 2
    # Next burst of skills in round-robin sequence fires immediately ('q', 'e', 'r')!
    assert skills_fired == ["q", "e", "r"]

    # Tick 3: 250ms (GCD for hotbar slot 1-4 elapsed) -> next burst cycles back to hotbar start
    skills_fired.clear()
    with patch("core.state_machine.time.time", return_value=t0 + 0.25):
        worker._step_combat_mode()

    worker._press_combat_tab.assert_not_called()
    assert worker._combat_was_targeting is True
    assert worker._combat_no_target_streak == 3
    assert skills_fired == ["1", "2", "3", "4"]
    # F-2048: still zero mouse activity after three full combat ticks.
    assert driver.mouse_events == []


def test_sustained_target_loss_triggers_clean_death_loot_and_tab():
    """
    When target is absent for sustained period (> 2.5s / 25 ticks) and character is not taking damage,
    the monster is confirmed dead: loots with 'f' and presses Tab for the next target.
    """
    driver = _MockDriver(trusted=True)
    worker = _create_combat_worker(driver)

    t0 = 1000.0
    worker._combat_last_target_ts = t0
    worker._combat_no_target_streak = 24  # Will reach 25 on this tick

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    worker._grab_frame = lambda grayscale=False: frame
    worker._combat_has_hostile_lock = MagicMock(return_value=False)
    worker._press_combat_tab = MagicMock()
    worker._press_key = MagicMock()

    # 3.0 seconds since target was seen -> grace expired, confirmed dead!
    with patch("core.state_machine.time.time", return_value=t0 + 3.0):
        worker._step_combat_mode()

    # Must loot 'f' and press Tab for next target
    worker._press_key.assert_called_with("f")
    worker._press_combat_tab.assert_called_with(soft_aim=False)
    assert worker._combat_was_targeting is False
    assert worker._combat_no_target_streak == 0
    # F-2048: the death/loot transition also issues no mouse input.
    assert driver.mouse_events == []


def test_combat_attack_uses_T_R_keys_not_mouse():
    """Direct proof of the R/T rule (F-2048): the production attack path
    presses T for the right-click attack and R for the left-click attack,
    holds/releases them via key_down/key_up, and never touches the mouse."""
    driver = _MockDriver(trusted=True)
    worker = _create_combat_worker(driver)

    # Right-click attack = T key pulse, no mouse event.
    worker._last_combat_mouse_spam_ts = 0.0
    worker._spam_combat_mouse("right")
    assert "t" in driver.pressed
    assert driver.mouse_events == []

    # Left-click attack = R key pulse, no mouse event.
    worker._last_combat_mouse_spam_ts = 0.0
    worker._spam_combat_mouse("left")
    assert "r" in driver.pressed
    assert driver.mouse_events == []

    # T hold/release for a selected target (right button sync).
    worker._sync_combat_right_button(True)
    assert "t" in driver.held_keys
    assert ("down", "t") in driver.key_events
    assert driver.mouse_events == []
    worker._release_combat_right_button()
    assert "t" not in driver.held_keys
    assert ("up", "t") in driver.key_events
    assert driver.mouse_events == []

    # R hold/release for the mana-regen phase (left button sync).
    worker._sync_combat_left_button(True)
    assert "r" in driver.held_keys
    assert driver.mouse_events == []
    worker._release_combat_left_button()
    assert "r" not in driver.held_keys
    assert driver.mouse_events == []
