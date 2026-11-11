# -*- coding: utf-8 -*-
"""Basic attack during one combat tick uses the T key, never the mouse. No GPU and no game.

R/T rule (user mapping): right-click attack = T key, left-click attack = R key.
``_sync_combat_right_button`` holds T via ``key_down("t")`` (or a ``press_key("t")``
pulse when the driver has no ``key_down``); the right mouse button must never
be pressed or held on the attack path.
"""

import unittest

from test_false_action_gates import _worker


class _Mouse:
    """Fake driver that records keys AND mouse-button activity separately,
    so the test can tell a T-key attack apart from a mouse click."""

    def __init__(self):
        self._mouse_buttons_down = set()
        self.right_events = []
        self.pressed = []
        self.held_keys = set()
        self.key_events = []

    def press_key(self, key, *args, **kwargs):
        self.pressed.append(str(key).lower())

    def key_down(self, key, *args, **kwargs):
        name = str(key).lower()
        self.held_keys.add(name)
        self.key_events.append(("down", name))
        return True

    def key_up(self, key, *args, **kwargs):
        name = str(key).lower()
        self.held_keys.discard(name)
        self.key_events.append(("up", name))
        return True

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


def _assert_no_right_mouse_click(testcase, mouse):
    testcase.assertNotIn("right", mouse._mouse_buttons_down)
    testcase.assertNotIn("down", mouse.right_events)
    testcase.assertEqual(mouse.right_events, [])


class CombatRightClickTests(unittest.TestCase):
    def test_target_present_one_tick_uses_T_key_not_mouse(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker.combat_recovery.bar = True
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0

        worker._step_combat_mode()

        # Attack went through the T key (hold + pulse), never through the mouse.
        self.assertIn("t", mouse.pressed)
        self.assertIn(("down", "t"), mouse.key_events)
        _assert_no_right_mouse_click(self, mouse)
        # Skills also went through keys (hotbar rotation), not mouse clicks.
        self.assertIn("1", mouse.pressed)

    def test_no_target_and_no_hp_drop_does_not_attack(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker.combat_recovery.bar = False
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0

        worker._step_combat_mode()

        _assert_no_right_mouse_click(self, mouse)
        self.assertNotIn("t", mouse.pressed)
        self.assertEqual(mouse.held_keys, set())
        self.assertEqual(worker.combat_recovery.fight_calls, 0)

    def test_pause_with_target_does_not_attack(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker._paused = True
        worker.combat_recovery.bar = True
        worker.combat_recovery.hp = 1.0

        worker._step_combat_mode()

        _assert_no_right_mouse_click(self, mouse)
        self.assertNotIn("t", mouse.pressed)
        self.assertEqual(mouse.held_keys, set())

    def test_second_tick_keeps_T_held_without_skill_spam(self):
        worker = _combat_worker()
        mouse = _Mouse()
        worker.input_driver = mouse
        worker.combat_recovery.bar = True
        worker.combat_recovery.under = True
        worker.combat_recovery.hp = 1.0
        worker._combat_last_skill_burst_ts = __import__("time").time()

        worker._step_combat_mode()
        worker._step_combat_mode()

        # T stays held across ticks (key_down, no release); no hotbar skill
        # keys were spammed — only the T attack key was pressed.
        self.assertIn("t", mouse.held_keys)
        self.assertEqual(mouse.pressed, ["t"])
        _assert_no_right_mouse_click(self, mouse)


if __name__ == "__main__":
    unittest.main()
