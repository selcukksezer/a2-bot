# -*- coding: utf-8 -*-
"""
Tests for:
1. Continuous combat attack spam via keys (R on low mana / regen phase, T on high mana / burst phase).
   No mouse clicks: right-click attack = T, left-click attack = R.
2. Auto-potion usage when HP drops.
3. Strict suppression of automated keys X, F6, and B from being pressed by the bot or chosen by AI.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.mana_phase_combat import ManaPhaseCombat, ManaPhase
from core.skill_bar_reader import SmartConsumableManager
from core.game_action_bus import GameActionBus
from core.state_machine import BotWorker, BotState, BotMode


class _RecordingDriver:
    """Recording input driver: real press/key_up/key_down surface, no hardware."""

    def __init__(self):
        self.pressed = []
        self.released = []
        self.holds = []
        self.clicks = []
        self._mouse_buttons_down = set()
        self._active_bot_keys = set()

    def keyboard_inject_trusted(self) -> bool:
        return True

    def is_keyboard_trusted(self) -> bool:
        return True

    def press_key(self, key, **kwargs) -> bool:
        self.pressed.append(key)
        return True

    def send_key(self, key, **kwargs) -> bool:
        self.pressed.append(key)
        return True

    def key_down(self, key, **kwargs) -> bool:
        self.holds.append(key)
        self._active_bot_keys.add(str(key).lower())
        return True

    def key_up(self, key, **kwargs) -> bool:
        self.released.append(key)
        self._active_bot_keys.discard(str(key).lower())
        return True

    def release_key(self, key, **kwargs) -> bool:
        return self.key_up(key)

    def human_click(self, button="left", **kwargs) -> bool:
        self.clicks.append(button)
        return True

    def _hw_mouse_down(self, button) -> None:
        self._mouse_buttons_down.add(button)

    def _hw_mouse_up(self, button) -> None:
        self._mouse_buttons_down.discard(button)


def _create_lightweight_worker(mode: str = BotMode.COMBAT_MODE.value, state: str = BotState.FIGHTING.value) -> BotWorker:
    """F-1463: build the worker through the REAL BotWorker.__init__ (deps
    injected) instead of BotWorker.__new__ + hand-set attributes, and use a
    recording driver — no _press_key lambda, no mocked is_combat_authorized.
    COMBAT_MODE + FIGHTING authorizes combat through the real gate."""
    driver = _RecordingDriver()
    # DEV-D088: same stub as test_combat_gather_isolation_c7.create_c7_worker
    # — BotWorker.__init__ builds ~40 optional subsystems (ONNX YOLO,
    # dataset factory, ...) at ~7s per worker; None is the production
    # load-failure fallback, and the gates under test never touch them.
    # The real __init__ path (locks, action bus, learner) still runs.
    with patch.object(BotWorker, "_init_optional_subsystem", return_value=None):
        worker = BotWorker(
            config={"test_mode": False},
            vision=MagicMock(),
            input_driver=driver,
            route_manager=MagicMock(),
            combat_recovery=MagicMock(),
            navigator=None,
            recorder=None,
        )
    worker.driver = driver
    worker._mode = mode
    worker._state = state
    worker._paused = False
    worker._emit_log = MagicMock()
    return worker


def test_forbidden_keys_suppression_in_state_machine():
    """Verify that BotWorker._press_key and _cast_one_combat_skill strictly block X, F6, and B."""
    sm = _create_lightweight_worker()
    driver = sm.input_driver
    assert isinstance(driver, _RecordingDriver)

    # Attempt to press forbidden keys
    for k in ("x", "X", "f6", "F6", "b", "B"):
        sm._press_key(k)
        sm._cast_one_combat_skill(k)

    # The recording driver must not have received any press for these keys.
    assert driver.pressed == []
    assert driver.holds == []


def test_game_action_bus_suppresses_x_f6_b():
    """Verify that GameActionBus.cast_skill rejects X, F6, and B immediately."""
    bus = GameActionBus(input_driver=MagicMock(), mouse_look_mode=True)
    for k in ("x", "X", "f6", "F6", "b", "B"):
        res = bus.cast_skill(k)
        assert res["ok"] is False
        assert res["method"] == "suppressed_automatic"


def test_smart_consumable_manager_forbidden_slots_default():
    """Verify SmartConsumableManager defaults to treating X, F6, and B as forbidden/automatic."""
    mgr = SmartConsumableManager(seed_defaults=True)
    assert "X" in mgr.forbidden_slots
    assert "F6" in mgr.forbidden_slots
    assert "B" in mgr.forbidden_slots

    # Default slot assignments
    assert mgr.get_slot_assignment("F6") == "automatic"
    assert mgr.get_slot_assignment("X") == "automatic"
    assert mgr.get_slot_assignment("B") == "automatic"

    # Best potion / scroll should never return forbidden keys
    hp = mgr.get_best_hp_potion()
    assert hp != "X"
    scrolls = mgr.get_expired_buff_scrolls()
    assert "F6" not in scrolls
    refreshed = mgr.check_and_refresh_scroll_buffs()
    assert "F6" not in refreshed

    # Decision should not contain forbidden keys
    dec = mgr.decide_consumable_action(player_hp_pct=0.30, player_mp_pct=0.30)
    for k in dec.get("keys_to_press", []):
        assert k.upper() not in ("X", "F6", "B")


def test_auto_potion_triggers_when_hp_drops():
    """Verify BotWorker._check_and_trigger_auto_potions drinks HP potion when HP < 65%."""
    sm = _create_lightweight_worker()
    sm.__dict__["_last_hp_potion_ts"] = 0.0
    sm.__dict__["_last_mp_potion_ts"] = 0.0
    driver = sm.input_driver

    # F-1463: no _press_key lambda — the real _press_key must drive the
    # recording driver.
    mana_brain = ManaPhaseCombat(hp_potion_key="F1", hp_potion_cd_sec=5.0)
    sm.mana_phase_combat = mana_brain

    # HP Healthy (90%) -> no potion
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.90, player_mp_ratio=0.90)
    assert triggered is False
    assert driver.pressed == []

    # HP Low (50% < 65%) -> triggers F1
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.50, player_mp_ratio=0.90)
    assert triggered is True
    assert "F1" in driver.pressed

    # Immediate second call is blocked by cooldown
    triggered_again = sm._check_and_trigger_auto_potions(player_hp_ratio=0.45, player_mp_ratio=0.90)
    assert triggered_again is False


def test_combat_mouse_spam_cadence_and_button_selection():
    """Verify _spam_combat_mouse presses R (left/low-mana) / T (right/high-mana) keys,
    releases the opposite attack key, and never clicks the mouse."""
    sm = _create_lightweight_worker()
    sm.__dict__["_last_combat_mouse_spam_ts"] = 0.0
    sm.__dict__["_combat_mouse_spam_gap"] = 0.05
    driver = sm.input_driver

    # F-1463: no stubbed sync/release lambdas — the real release paths run
    # against the recording driver. Pre-hold the opposite attack key on each
    # side so the release is observable through key_up().
    sm._mark_right_button(True)

    # Spam left attack (low mana) -> R key, T released
    sm._spam_combat_mouse("left")
    assert "r" in driver.pressed
    assert "t" in driver.released
    assert driver.clicks == []

    # Advance time and spam right attack (high mana) -> T key, R released
    sm._last_combat_mouse_spam_ts = time.time() - 1.0
    if sm.action_bus is not None:
        sm.action_bus._left_held = True
    sm._spam_combat_mouse("right")
    assert "t" in driver.pressed
    assert "r" in driver.released
    assert driver.clicks == []


def test_core_wiring_survives_stubbed_optional_subsystems():
    """DEV-D088 guard: _create_lightweight_worker stubs
    _init_optional_subsystem to None for speed. The real __init__
    invariants must still hold: real RLock, real action bus (the spam
    path dispatches through it), real learner wiring, stubbed heavy
    subsystems recorded as None."""
    sm = _create_lightweight_worker()
    assert isinstance(sm._lock, type(threading.RLock()))
    assert not isinstance(sm._lock, MagicMock)
    assert sm.action_bus is not None
    assert sm.game_learner is not None
    assert sm.dataset_factory is None
    assert sm.yolo_detector is None
    assert sm.input_driver is not None
