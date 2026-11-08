# -*- coding: utf-8 -*-
"""
Agent C7 Verification Suite: Combat State Machine Isolation & Skill Leak Prevention
Tests:
1. Zero skill key leakage during Toplama (ROUTE_MODE) when route.fight_back=False under hostile attack.
2. Mode Isolation Gate: direct _press_key('1'), _cast_one_combat_skill(), _send_configured_combat_skill() suppression.
3. Active harvest (GATHERING) strict isolation: zero combat skills even with fight_back enabled.
4. Clean aggro defense in ROUTE_MODE when route.fight_back=True: fights back when attacked.
5. Clean post-combat return-to-route: after mob dies, loots with F, does NOT press TAB, releases all combat keys, and resumes route walking.
6. Anti-stuck key release guard: mode transitions and state changes guarantee all combat keys and mouse buttons released.
"""

import sys
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from PyQt6.QtCore import QThread

from core.state_machine import BotMode, BotState, BotWorker


class MockInputDriver:
    """Mock input driver tracking all key and mouse events."""

    def __init__(self, trusted: bool = True):
        self.pressed = []
        self.released = []
        self.holds = []
        self.mouse_events = []
        self._mouse_buttons_down = set()
        self._active_bot_keys = set()
        self._held_movement = set()
        self._trusted = trusted
        self._motion_abort_count = 0

    def keyboard_inject_trusted(self) -> bool:
        return self._trusted

    def is_keyboard_trusted(self) -> bool:
        return self._trusted

    def movement_active(self) -> bool:
        return bool(self._held_movement) or getattr(self, "moving", True)

    def press_key(self, key: str, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.pressed.append(k)
        return True

    def send_key(self, key: str) -> bool:
        k = str(key).lower().strip()
        self.pressed.append(k)
        return True

    def key_down(self, key: str, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.holds.append(k)
        self._active_bot_keys.add(k)
        return True

    def key_up(self, key: str, **kwargs) -> bool:
        k = str(key).lower().strip()
        self.released.append(k)
        self._active_bot_keys.discard(k)
        return True

    def release_key(self, key: str) -> bool:
        return self.key_up(key)

    def _hw_key_up(self, key: str) -> bool:
        return self.key_up(key)

    def _hw_mouse_down(self, button: str) -> None:
        self.mouse_events.append(("down", button))
        self._mouse_buttons_down.add(button)

    def _hw_mouse_up(self, button: str) -> None:
        self.mouse_events.append(("up", button))
        self._mouse_buttons_down.discard(button)

    def abort_motion(self) -> None:
        self._motion_abort_count += 1
        self._mouse_buttons_down.clear()

    def release_all_movement_keys(self) -> None:
        self._held_movement.clear()

    def clear(self):
        self.pressed.clear()
        self.released.clear()
        self.holds.clear()
        self.mouse_events.clear()


def create_c7_worker(mode=BotMode.ROUTE_MODE.value, state=BotState.WALKING.value, config=None):
    """Helper to instantiate a lightweight BotWorker for isolation testing."""
    worker = BotWorker.__new__(BotWorker)
    QThread.__init__(worker)
    worker._lock = threading.RLock()
    worker._mode = mode
    worker._state = state
    worker._paused = False
    worker._state_enter_time = time.time()
    worker.config = {
        "test_mode": False,
        "fight_back": False,
        "combat_skill_keys": ["1", "2", "3", "4", "q", "e", "r"],
    }
    if config:
        worker.config.update(config)

    driver = MockInputDriver(trusted=True)
    worker.input_driver = driver
    worker.driver = driver

    worker.combat_recovery = MagicMock()
    worker.combat_recovery.is_dead.return_value = False
    worker.combat_recovery.check_character_dead.return_value = False
    worker.combat_recovery.check_under_attack.return_value = (False, 1.0)
    worker.combat_recovery.in_combat = False

    worker.route_manager = MagicMock()
    worker.active_node = {"id": 101, "distance_m": 15}
    worker.logs = []
    worker._emit_log = lambda msg: worker.logs.append(str(msg))
    worker._emit_hud_snapshot = lambda **kwargs: {}
    worker._grab_frame = lambda grayscale=False: np.zeros((720, 1280, 3), dtype=np.uint8)

    worker._route_was_fighting = False
    worker._route_fight_had_signal = False
    worker._route_combat_confirmed = False
    worker._route_saw_target_bar = False
    worker._route_saw_hp_drop = False
    worker._route_prev_hp_ratio = None
    worker._route_last_loot_ts = 0.0
    worker._route_bar_since_loot = False
    worker._combat_was_targeting = False
    worker._combat_no_target_streak = 0
    worker._combat_last_tab_ts = 0.0
    worker._combat_right_button_down = False
    worker._combat_skill_used_at = {}
    worker._combat_skill_idx = 0
    worker._last_bot_key_time = 0.0
    worker._is_gathering_active = False

    return worker, driver


class TestCombatGatherIsolationC7(unittest.TestCase):
    """Rigorous verification of mode isolation and zero key leaks."""

    def test_pure_gathering_zero_skill_leakage_when_fight_back_false(self):
        """Zero SKILL KEY leakage during Toplama when route.fight_back=False under simulated attack."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.WALKING.value,
            config={"fight_back": False, "route": {"fight_back": False}},
        )

        worker.combat_recovery.check_under_attack.return_value = (True, 0.40)
        worker._enemy_target_bar_visible = MagicMock(return_value=True)
        worker._note_player_hp_drop = MagicMock(return_value=True)

        for _ in range(30):
            worker._handle_combat_and_death()

        combat_keys = {"1", "2", "3", "4", "5", "q", "e", "r", "tab"}
        leaked_keys = [k for k in driver.pressed if k in combat_keys]
        self.assertEqual(len(leaked_keys), 0, f"Combat keys leaked during Toplama with fight_back=False: {leaked_keys}")
        self.assertEqual(worker.combat_recovery.fight_back_step.call_count, 0)
        self.assertNotIn("right", driver._mouse_buttons_down)

    def test_mode_isolation_gate_blocks_direct_combat_calls_in_route_mode(self):
        """Mode Isolation Gate directly suppresses _cast_one_combat_skill, _send_configured_combat_skill, _press_key."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.WALKING.value,
            config={"fight_back": False},
        )

        # 1. is_combat_authorized must return False
        self.assertFalse(worker.is_combat_authorized("skill"))
        self.assertFalse(worker.is_combat_authorized("target"))
        self.assertFalse(worker.is_combat_authorized("attack"))

        # 2. _send_configured_combat_skill must return False with 0 keys pressed
        res = worker._send_configured_combat_skill()
        self.assertFalse(res)
        self.assertEqual(len(driver.pressed), 0)

        # 3. _cast_one_combat_skill must be suppressed with 0 keys pressed
        worker._cast_one_combat_skill("1")
        worker._cast_one_combat_skill("2")
        worker._cast_one_combat_skill("q")
        self.assertEqual(len(driver.pressed), 0)

        # 4. Direct _press_key with pure combat skill keys must be blocked by the hardware gate
        worker._press_key("1")
        worker._press_key("3")
        worker._press_key("r")
        worker._press_key("tab")
        self.assertEqual(len(driver.pressed), 0)

        # 5. Non-combat keys (movement, interaction, map, esc) must still pass through
        worker._press_key("f")
        worker._press_key("esc")
        self.assertEqual(driver.pressed, ["f", "esc"])

    def test_active_harvest_state_strictly_suppresses_combat(self):
        """When in GATHERING state or _is_gathering_active=True, all combat actions are suppressed even with fight_back=True."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.GATHERING.value,
            config={"fight_back": True, "route": {"fight_back": True}},
        )
        worker._is_gathering_active = True

        self.assertFalse(worker.is_combat_authorized("skill"))
        self.assertFalse(worker.is_combat_authorized("attack"))

        worker._cast_one_combat_skill("1")
        self.assertEqual(len(driver.pressed), 0)

        worker._press_key("2")
        self.assertEqual(len(driver.pressed), 0)

    def test_clean_aggro_defense_when_route_fight_back_true(self):
        """When route.fight_back=True, character attacked executes clean aggro defense."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.WALKING.value,
            config={"fight_back": True, "route": {"fight_back": True}},
        )

        worker.combat_recovery.check_under_attack.return_value = (True, 0.60)
        worker._enemy_target_bar_visible = MagicMock(return_value=True)
        worker._note_player_hp_drop = MagicMock(return_value=True)

        res = worker._handle_combat_and_death()
        self.assertTrue(res)
        self.assertEqual(worker._state, BotState.FIGHTING.value)
        self.assertTrue(worker._route_combat_confirmed)
        self.assertGreater(worker.combat_recovery.fight_back_step.call_count, 0)

    def test_clean_post_combat_return_to_route(self):
        """When mob dies, loots with F, does NOT press TAB, releases all combat inputs, and resumes route."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.FIGHTING.value,
            config={"fight_back": True, "route": {"fight_back": True}},
        )
        worker._route_combat_confirmed = True
        worker._route_was_fighting = True
        worker._route_fight_had_signal = True
        worker._route_saw_target_bar = True
        worker.active_node = {"id": 105, "distance_m": 12}

        # Simulate right button down during combat
        worker._mark_right_button(True)
        driver._mouse_buttons_down.add("right")

        # Mob dead: bar gone, hp stable
        worker.combat_recovery.check_under_attack.return_value = (False, 0.60)
        worker._enemy_target_bar_visible = MagicMock(return_value=False)
        worker._note_player_hp_drop = MagicMock(return_value=False)

        worker._handle_combat_and_death()

        # 1. F was pressed for loot
        self.assertIn("f", driver.pressed)
        # 2. TAB was NOT pressed (prevent targeting other innocent mobs in route mode)
        self.assertNotIn("tab", driver.pressed)
        # 3. Right mouse button was released cleanly
        self.assertNotIn("right", driver._mouse_buttons_down)
        # 4. Combat flags cleared
        self.assertFalse(worker._route_combat_confirmed)
        self.assertFalse(worker._route_was_fighting)
        # 5. Clean return to WALKING state with active_node preserved
        self.assertEqual(worker._state, BotState.WALKING.value)
        self.assertIsNotNone(worker.active_node)

    def test_mode_switch_releases_all_combat_inputs(self):
        """Switching mode away from COMBAT_MODE releases all mouse buttons and keys."""
        worker, driver = create_c7_worker(
            mode=BotMode.COMBAT_MODE.value,
            state=BotState.FIGHTING.value,
        )
        worker._mark_right_button(True)
        driver._mouse_buttons_down.add("right")
        driver._active_bot_keys.add("1")
        driver._active_bot_keys.add("q")

        # Switch to ROUTE_MODE
        worker.set_mode(BotMode.ROUTE_MODE.value)

        self.assertEqual(worker._mode, BotMode.ROUTE_MODE.value)
        self.assertFalse(worker._right_button_is_tracked())
        self.assertNotIn("right", driver._mouse_buttons_down)
        self.assertIn("1", driver.released)
        self.assertIn("q", driver.released)

    def test_state_change_out_of_fighting_releases_combat_inputs(self):
        """Transitioning from FIGHTING to WAITING or GATHERING automatically triggers key release guard."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.FIGHTING.value,
        )
        worker._mark_right_button(True)
        driver._mouse_buttons_down.add("right")
        driver._active_bot_keys.add("2")

        worker._set_state_internal(BotState.WAITING.value)

        self.assertEqual(worker._state, BotState.WAITING.value)
        self.assertFalse(worker._right_button_is_tracked())
        self.assertNotIn("right", driver._mouse_buttons_down)
        self.assertIn("2", driver.released)


if __name__ == "__main__":
    unittest.main()
