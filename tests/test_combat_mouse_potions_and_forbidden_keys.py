# -*- coding: utf-8 -*-
"""
Tests for:
1. Continuous combat mouse spam (left click on low mana / regen phase, right click on high mana / burst phase).
2. Auto-potion usage when HP drops.
3. Strict suppression of automated keys X, F6, and B from being pressed by the bot or chosen by AI.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
import pytest
from unittest.mock import MagicMock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.mana_phase_combat import ManaPhaseCombat, ManaPhase
from core.skill_bar_reader import SmartConsumableManager
from core.game_action_bus import GameActionBus
from core.state_machine import BotWorker, BotState, BotMode


from PyQt6.QtCore import QThread
import threading

def _create_lightweight_worker(mode: str = BotMode.COMBAT_MODE.value, state: str = BotState.FIGHTING.value) -> BotWorker:
    worker = BotWorker.__new__(BotWorker)
    QThread.__init__(worker)
    worker._lock = threading.RLock()
    worker._mode = mode
    worker._state = state
    worker._paused = False
    worker.config = {"test_mode": False}
    worker.is_combat_authorized = MagicMock(return_value=True)
    return worker


def test_forbidden_keys_suppression_in_state_machine():
    """Verify that BotWorker._press_key and _cast_one_combat_skill strictly block X, F6, and B."""
    sm = _create_lightweight_worker()
    driver = MagicMock()
    sm.input_driver = driver
    sm.is_combat_authorized = MagicMock(return_value=True)

    # Attempt to press forbidden keys
    for k in ("x", "X", "f6", "F6", "b", "B"):
        sm._press_key(k)
        sm._cast_one_combat_skill(k)

    # input_driver should NOT have received press_key or send_key for any of these
    driver.press_key.assert_not_called()
    driver.send_key.assert_not_called()


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

    pressed_keys = []
    sm._press_key = lambda k, **kw: pressed_keys.append(k)
    sm._emit_log = MagicMock()

    mana_brain = ManaPhaseCombat(hp_potion_key="F1", hp_potion_cd_sec=5.0)
    sm.mana_phase_combat = mana_brain

    # HP Healthy (90%) -> no potion
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.90, player_mp_ratio=0.90)
    assert triggered is False
    assert len(pressed_keys) == 0

    # HP Low (50% < 65%) -> triggers F1
    triggered = sm._check_and_trigger_auto_potions(player_hp_ratio=0.50, player_mp_ratio=0.90)
    assert triggered is True
    assert "F1" in pressed_keys

    # Immediate second call is blocked by cooldown
    triggered_again = sm._check_and_trigger_auto_potions(player_hp_ratio=0.45, player_mp_ratio=0.90)
    assert triggered_again is False


def test_combat_mouse_spam_cadence_and_button_selection():
    """Verify _spam_combat_mouse clicks target button and releases opposite button."""
    sm = _create_lightweight_worker()
    sm.__dict__["_last_combat_mouse_spam_ts"] = 0.0
    sm.__dict__["_combat_mouse_spam_gap"] = 0.05

    clicked_buttons = []
    released_buttons = []

    driver = MagicMock()
    driver.human_click = lambda button, **kw: clicked_buttons.append(button)
    sm.input_driver = driver
    sm._sync_combat_right_button = lambda on: released_buttons.append("right_off") if not on else None
    sm._release_combat_left_button = lambda: released_buttons.append("left_off")

    # Spam left button (low mana)
    sm._spam_combat_mouse("left")
    assert "left" in clicked_buttons
    assert "right_off" in released_buttons

    # Advance time and spam right button (high mana)
    sm._last_combat_mouse_spam_ts = time.time() - 1.0
    sm._spam_combat_mouse("right")
    assert "right" in clicked_buttons
    assert "left_off" in released_buttons
