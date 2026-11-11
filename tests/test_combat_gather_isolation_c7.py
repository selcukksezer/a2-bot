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
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest
from PyQt6.QtCore import QThread

from core.state_machine import BotMode, BotState, BotWorker


@pytest.fixture(autouse=True)
def _isolated_lifetime_stats(tmp_path):
    """DEV-D088: combat-death paths call get_lifetime_stats() (module
    singleton, default config/lifetime_stats.json) and record_monster_kill
    increments the REAL repo file (+6 kills per full slice run). Redirect
    the singleton lookup to a tmp-backed store for every test in this file;
    no test here asserts on lifetime stats, so nothing is weakened."""
    from core.lifetime_stats import LifetimeStatsStore

    store = LifetimeStatsStore(path=str(tmp_path / "lifetime_stats.json"))
    with patch("core.lifetime_stats.get_lifetime_stats", return_value=store):
        yield


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
    """Helper instantiating a BotWorker for isolation testing.

    F-1461: the worker is built through the REAL BotWorker.__init__ (all
    dependencies injected as mocks/recording stubs) instead of
    BotWorker.__new__ + hand-set attributes, so __init__-established
    invariants (locks, flags, action bus, learner wiring) are real. Only
    test-arranged state that __init__ does not establish is set afterwards.
    """
    base_config = {
        "test_mode": False,
        "fight_back": False,
        "combat_skill_keys": ["1", "2", "3", "4", "q", "e", "r", "t"],
    }
    if config:
        base_config.update(config)

    driver = MockInputDriver(trusted=True)
    combat_recovery = MagicMock()
    combat_recovery.is_dead.return_value = False
    combat_recovery.check_character_dead.return_value = False
    combat_recovery.check_under_attack.return_value = (False, 1.0)
    combat_recovery.in_combat = False

    # DEV-D088: BotWorker.__init__ builds ~40 optional subsystems through
    # _init_optional_subsystem (ONNX YOLO detector, audio analyzer, dataset
    # factory, ...), ~7s per worker on this VM — 8 tests x fresh worker.
    # None is exactly the production fallback when a subsystem fails to
    # load, so stubbing it to None keeps the real __init__ path (locks,
    # flags, action bus, learner wiring) while skipping unrelated heavy
    # construction. The isolation gates under test never touch these
    # subsystems; test_core_wiring_survives_stubbed_optional_subsystems
    # locks that in.
    with patch.object(BotWorker, "_init_optional_subsystem", return_value=None):
        worker = BotWorker(
            config=base_config,
            vision=MagicMock(),
            input_driver=driver,
            route_manager=MagicMock(),
            combat_recovery=combat_recovery,
            navigator=None,
            recorder=None,
        )
    worker.driver = driver
    worker._mode = mode
    worker._state = state
    worker._paused = False
    worker._state_enter_time = time.time()
    worker.active_node = {"id": 101, "distance_m": 15}
    worker.logs = []
    worker._emit_log = lambda msg: worker.logs.append(str(msg))
    worker._emit_hud_snapshot = lambda **kwargs: {}
    worker._grab_frame = lambda grayscale=False: np.zeros((720, 1280, 3), dtype=np.uint8)

    # Per-test arrangement (attributes __init__ leaves to callers/lazy init).
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

        combat_keys = {"1", "2", "3", "4", "5", "q", "e", "r", "t", "tab"}
        leaked_keys = [k for k in driver.pressed if k in combat_keys]
        self.assertEqual(len(leaked_keys), 0, f"Combat keys leaked during Toplama with fight_back=False: {leaked_keys}")
        self.assertEqual(worker.combat_recovery.fight_back_step.call_count, 0)
        self.assertNotIn("right", driver._mouse_buttons_down)

        # F-1462 control phase: the SAME (True, 0.40) attack signal must reach
        # production and engage when fight_back is enabled — proving the
        # zero-leakage phase above is a live gate, not an unread signal.
        # 2-kare kuralı (core/state_machine.py _route_combat_signal_confirmed,
        # route_fight_confirm_frames=2): the fight_back=False phase clears
        # the confirmation streaks on every tick, so the control phase must
        # prime for 2 ticks — tick 1 only primes, tick 2 engages.
        worker.config["fight_back"] = True
        worker.config["route"]["fight_back"] = True
        driver.clear()
        res_first = worker._handle_combat_and_death()
        self.assertFalse(res_first)
        res = worker._handle_combat_and_death()
        self.assertTrue(res)
        self.assertEqual(worker._state, BotState.FIGHTING.value)
        self.assertTrue(worker._route_combat_confirmed)
        self.assertGreater(worker.combat_recovery.fight_back_step.call_count, 0)

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

    def test_press_key_gate_blocks_e_and_t_combat_keys(self):
        """F-1358: 'e' and 't' are combat skill keys (hotbar 1..4/q/e/r/t) and
        must be blocked by the _press_key hardware gate in ROUTE_MODE with
        fight_back=False — exactly like '1', '3' and 'r' above.

        Regression protection (P3-4: the old "KNOWN FAILURE" note was
        stale): the F-1358 fix is IN v17 — the production gate tuple in
        core/state_machine.py::_press_key (F-1358 [O1]) already covers
        "e"/"t". This test pins that behaviour; it must stay green.
        """
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.WALKING.value,
            config={"fight_back": False},
        )

        worker._press_key("e")
        worker._press_key("t")
        self.assertEqual(driver.pressed, [], f"Combat keys leaked via _press_key gate: {driver.pressed}")

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

        # 2-kare kuralı (core/state_machine.py _route_combat_signal_confirmed,
        # route_fight_confirm_frames=2): tick 1 only primes the streak,
        # tick 2 confirms the engagement.
        res_first = worker._handle_combat_and_death()
        self.assertFalse(res_first)
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

        # 2-kare kuralı (core/state_machine.py _route_combat_signal_confirmed,
        # route_fight_confirm_frames=2): the fight must really start first —
        # two consecutive attack ticks. Tick 1 only primes the streak
        # (route_bar comes from the recovery's bar detector, which is a
        # truthy MagicMock here); tick 2 confirms and enters FIGHTING.
        # fix/v20r1-combat-gate (madde 4, commit 106bdde): bar + SABİT HP
        # artık doğrulamaz — bar penceresinde HP düşüşü veya ToT aggro
        # gerekir (hayalet-savaş: pasif mob barı + sabit HP; bkz.
        # test_route_fight_confirmation.py
        # test_two_consecutive_bar_frames_alone_do_not_enter_fighting —
        # bar + (True, 1.0) sabit HP 10 tick'te bile FIGHTING'e sokmuyor).
        # O yüzden tick 1 (True, 0.65) tabanı kurar, tick 2 (True, 0.60)
        # düşüşü verir (test_bar_plus_hp_drop_enters_fighting deseni);
        # loot / TAB yok / temiz dönüş iddiaları değişmedi.
        worker.combat_recovery.check_under_attack.return_value = (True, 0.65)
        self.assertFalse(worker._handle_combat_and_death())
        worker.combat_recovery.check_under_attack.return_value = (True, 0.60)
        self.assertTrue(worker._handle_combat_and_death())
        self.assertEqual(worker._state, BotState.FIGHTING.value)

        # Simulate right button down during combat
        worker._mark_right_button(True)
        driver._mouse_buttons_down.add("right")

        # Mob dead: bar gone, hp stable. NOTE: the route gate reads the
        # bar from the recovery's _detect_enemy_target_bar /
        # detect_enemy_target_bar (core/state_machine.py
        # _route_enemy_bar_visible), NOT from _enemy_target_bar_visible —
        # and a plain MagicMock auto-creates TRUTHY detectors, so without
        # stubbing BOTH names the bar would stay "visible", the fight
        # would never end, and no loot F would fire (v10 failure:
        # 'f' not found in ['f1'] — only the auto-potion F1 was pressed).
        worker.combat_recovery.check_under_attack.return_value = (False, 0.60)
        worker.combat_recovery._detect_enemy_target_bar = MagicMock(return_value=False)
        worker.combat_recovery.detect_enemy_target_bar = MagicMock(return_value=False)
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

    def test_core_wiring_survives_stubbed_optional_subsystems(self):
        """DEV-D088 guard: create_c7_worker stubs _init_optional_subsystem
        to None for speed. The real __init__ invariants the isolation
        tests rely on must still hold: a real RLock, the real action bus,
        the real learner wiring, and the stubbed heavy subsystems recorded
        as None (the production load-failure fallback)."""
        worker, driver = create_c7_worker(
            mode=BotMode.ROUTE_MODE.value,
            state=BotState.WALKING.value,
        )
        self.assertIsInstance(worker._lock, type(threading.RLock()))
        self.assertNotIsInstance(worker._lock, MagicMock)
        self.assertIsNotNone(worker.action_bus)
        self.assertIsNotNone(worker.game_learner)
        self.assertIsNone(worker.dataset_factory)
        self.assertIsNone(worker.yolo_detector)
        self.assertIsNone(worker.telemetry)
        self.assertEqual(worker.input_driver, driver)
        self.assertEqual(worker._mode, BotMode.ROUTE_MODE.value)


if __name__ == "__main__":
    unittest.main()
