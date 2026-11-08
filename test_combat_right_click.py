# -*- coding: utf-8 -*-
"""Right-click basic attack during one combat tick. No GPU and no game."""

import unittest

from test_false_action_gates import _worker


class _Mouse:
    def __init__(self):
        self._mouse_buttons_down = set()
        self.right_events = []
        self.pressed = []

    def press_key(self, key, *args, **kwargs):
        self.pressed.append(str(key).lower())

    def _hw_mouse_down(self, button="left"):
        name = str(button).lower()
        self._mouse_buttons_down.add(name)
        if name == "right":
            self.right_events.append("down")

    def _hw_mouse_up(self, button="left"):
        name = str(button).lower()
        self._mouse_buttons_down.discard(name)
        if name == "right":
            self.right_events.append("up")


def _combat_worker():
    worker = _worker()
    worker._mode = "combat"
    worker._paused = False
    worker._combat_was_targeting = False
    worker._combat_no_target_streak = 0
    worker._combat_last_tab_ts = 0.0
    worker.behavioral_guard = None
    worker.skill_bar_monitor = None
    return worker


class CombatRightClickTests(unittest.TestCase):
    def test_target_present_one_tick_holds_right_button(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker.combat_recovery.bar = True
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0

        worker._step_combat_mode()

        self.assertTrue("right" in mouse._mouse_buttons_down or "down" in mouse.right_events)

    def test_no_target_and_no_hp_drop_does_not_right_click(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker.combat_recovery.bar = False
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0

        worker._step_combat_mode()

        self.assertNotIn("right", mouse._mouse_buttons_down)
        self.assertNotIn("down", mouse.right_events)
        self.assertEqual(worker.combat_recovery.fight_calls, 0)

    def test_pause_with_target_does_not_right_click(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker._paused = True
        worker.combat_recovery.bar = True
        worker.combat_recovery.hp = 1.0

        worker._step_combat_mode()

        self.assertNotIn("right", mouse._mouse_buttons_down)
        self.assertNotIn("down", mouse.right_events)

    def test_second_tick_keeps_right_without_skill_spam(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker.combat_recovery.bar = True
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0
        worker._combat_last_skill_burst_ts = __import__("time").time()

        worker._step_combat_mode()
        worker._step_combat_mode()

        self.assertIn("right", mouse._mouse_buttons_down)
        self.assertEqual(mouse.pressed, [])


if __name__ == "__main__":
    unittest.main()
