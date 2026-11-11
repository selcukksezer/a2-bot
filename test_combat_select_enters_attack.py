# -*- coding: utf-8 -*-
"""After a monster select, combat must enter the attack step via keys (T + hotbar).

The attack entry of ``_step_combat_mode`` is the key-based rotation: the T key
(right-click attack = T) is held via ``_sync_combat_right_button(True)`` and
hotbar skills are cast via ``_cast_one_combat_skill``. ``fight_back_step`` is
NOT this path's entry — it is the legacy route-mode entry
(``core/state_machine.py`` calls it from the route loop only), so a test that
claims "attack entry called" must not point at ``fight_back_step``.
"""

import threading
import time
import unittest

import numpy as np
from PyQt6.QtCore import QThread

from core.state_machine import BotWorker


class _Keys:
    def __init__(self):
        self.pressed = []

    def press_key(self, key, *args, **kwargs):
        self.pressed.append(str(key).lower())


class _Recovery:
    def __init__(self, bar_after_tab):
        self.bar_after_tab = bar_after_tab
        self.fight_calls = 0
        self.under = False
        self.hp = 1.0
        self.in_combat = False

    def check_character_dead(self, frame=None):
        return False

    def check_under_attack(self, frame=None):
        return (self.under, self.hp)

    def _detect_enemy_target_bar(self, frame):
        return False

    def fight_back_step(self, input_ctrl=None, frame=None):
        self.fight_calls += 1

    def fight_back(self, input_ctrl=None, frame=None):
        raise AssertionError("fight_back_step must not be called on the combat path")

    def execute_combat_step(self, input_ctrl=None, frame=None):
        raise AssertionError("fight_back_step must not be called on the combat path")


def _worker(bar_after_tab):
    worker = BotWorker.__new__(BotWorker)
    # QThread.__init__ is required: BotWorker properties (is_paused, ...) and
    # _check_game_focus() raise RuntimeError without it.
    try:
        QThread.__init__(worker)
    except Exception:
        pass
    worker._lock = threading.RLock()
    worker._mode = "combat"
    worker._state = "WAITING"
    worker._paused = False
    worker._state_enter_time = time.time()
    worker.config = {"test_mode": False, "fight_back": True}
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
    worker.logs = []
    worker.input_driver = _Keys()
    worker.vision = None
    worker.combat = None
    worker.game_learner = None
    worker.skill_bar_monitor = None
    worker.behavioral_guard = None
    worker.combat_recovery = _Recovery(bar_after_tab)
    worker.msleep = lambda _ms: None
    worker._emit_log = lambda message: worker.logs.append(str(message))
    worker._emit_hud_snapshot = lambda **_kwargs: {}
    worker._set_state_internal = lambda new_state: setattr(worker, "_state", new_state)
    worker._grab_frame = lambda grayscale=True: np.zeros((16, 16, 3), dtype=np.uint8)

    recovery = worker.combat_recovery

    def _detect(frame):
        return bool(recovery.bar_after_tab and "tab" in worker.input_driver.pressed)

    recovery._detect_enemy_target_bar = _detect
    return worker


class CombatSelectEntersAttackTests(unittest.TestCase):
    def _run(self, bar_after_tab):
        worker = _worker(bar_after_tab)
        right = []
        real_sync = worker._sync_combat_right_button

        def _sync(target_selected):
            right.append(bool(target_selected))
            return real_sync(target_selected)

        worker._sync_combat_right_button = _sync
        worker._step_combat_mode()
        return worker, right

    def test_successful_monster_select_enters_key_attack(self):
        worker, right = self._run(True)
        skills = [key for key in worker.input_driver.pressed if key not in ("tab", "f")]
        self.assertIn("tab", worker.input_driver.pressed)
        # Attack entry is the key rotation: T held + hotbar skills cast.
        self.assertIn(True, right, "T-key attack hold was never requested")
        self.assertIn("t", worker.input_driver.pressed)
        self.assertTrue(skills)
        self.assertEqual(worker._state, "FIGHTING")
        # fight_back_step is the legacy route-mode entry, not this path's:
        # it must NOT be called by _step_combat_mode.
        self.assertEqual(worker.combat_recovery.fight_calls, 0)

    def test_failed_select_does_not_enter_attack(self):
        worker, right = self._run(False)
        skills = [key for key in worker.input_driver.pressed if key not in ("tab", "f")]
        self.assertIn("tab", worker.input_driver.pressed)
        self.assertNotIn(True, right, "no target, so no T-key attack hold")
        self.assertNotIn("t", worker.input_driver.pressed)
        self.assertEqual(worker.combat_recovery.fight_calls, 0)
        self.assertEqual(skills, [])
        self.assertNotEqual(worker._state, "FIGHTING")


if __name__ == "__main__":
    unittest.main()
