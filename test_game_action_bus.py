# -*- coding: utf-8 -*-
"""GameActionBus: skill UI click primary, gather prompt click, exclusive lock order."""

import unittest

import numpy as np

from core.game_action_bus import GameActionBus
from core.skill_bar_reader import SkillBarReader


class _Driver:
    def __init__(self, trusted_keyboard: bool = False):
        self.clicks = []
        self.keys = []
        self.moves = []
        self.held = []
        self._mouse_buttons_down = set()
        self._trusted_keyboard = bool(trusted_keyboard)

    def focus_game_window(self):
        return True

    def human_click(self, x, y, button="left", **kwargs):
        self.clicks.append((int(x), int(y), str(button)))

    def press_key(self, key, **kwargs):
        self.keys.append((str(key).lower(), dict(kwargs)))

    def keyboard_inject_trusted(self):
        """SerialHID/NetworkHID mock — soft SendInput is not trusted for AION2."""
        return bool(self._trusted_keyboard)

    def _hw_key_down(self, key, post_to_window=False):
        self.held.append(("down", str(key).lower(), bool(post_to_window)))
        return True

    def _hw_key_up(self, key, post_to_window=False):
        self.held.append(("up", str(key).lower(), bool(post_to_window)))

    def _hw_move_to(self, x, y):
        self.moves.append((int(x), int(y)))

    def _release_mouse_buttons(self):
        self._mouse_buttons_down.clear()


class _Vision:
    def __init__(self, bbox=None, key="f"):
        self.bbox = bbox
        self.last_detected_gather_key = key

    def detect_gather_prompt(self, frame=None, **kwargs):
        if self.bbox is None:
            return False, 0, None
        return True, 90, self.bbox


class GameActionBusTests(unittest.TestCase):
    def test_hold_left_attack_toggles(self):
        driver = _Driver()
        downs = []

        def _down(btn):
            downs.append(("down", btn))
            driver._mouse_buttons_down.add(btn)

        def _up(btn):
            downs.append(("up", btn))
            driver._mouse_buttons_down.discard(btn)

        driver._hw_mouse_down = _down
        driver._hw_mouse_up = _up
        bus = GameActionBus(input_driver=driver)
        bus.hold_left_attack(True)
        bus.hold_left_attack(True)  # idempotent
        bus.hold_left_attack(False)
        self.assertEqual(downs, [("down", "left"), ("up", "left")])

    def test_slot_center_splits_bar_into_eight(self):
        reader = SkillBarReader()
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        p1 = reader.slot_center_px(frame, "1", jitter_px=0)
        pq = reader.slot_center_px(frame, "q", jitter_px=0)
        pr = reader.slot_center_px(frame, "R", jitter_px=0)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(pq)
        self.assertIsNotNone(pr)
        self.assertLess(p1[0], pq[0])
        self.assertLess(pq[0], pr[0])
        # y inside skill bar band
        self.assertGreaterEqual(p1[1], int(1080 * reader.SKILL_BAR_ROI[1]))
        self.assertLessEqual(p1[1], int(1080 * reader.SKILL_BAR_ROI[3]))

    def test_cast_skill_mouse_look_uses_key_not_ui_click(self):
        """AION2 mouse-look: trusted SerialHID/NetworkHID → hardware key only."""
        driver = _Driver(trusted_keyboard=True)
        reader = SkillBarReader()
        released = []
        restored = []
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=reader,
            release_basic_attack=lambda: released.append("up"),
            restore_basic_attack=lambda: restored.append("down"),
            mouse_look_mode=True,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = bus.cast_skill("2", frame=frame)
        self.assertTrue(out["ok"])
        self.assertEqual(out["method"], "key")
        self.assertEqual(driver.clicks, [])
        # Trusted path uses _hw_key_down/up (no soft press_key / no PostMessage).
        downs = [h for h in driver.held if h[0] == "down" and h[1] == "2"]
        self.assertGreaterEqual(len(downs), 1)
        self.assertTrue(all(h[2] is False for h in downs))
        # Mouse-look keeps LMB AA; exclusive path must not release/restore RMB attack.
        self.assertEqual(released, [])
        self.assertEqual(restored, [])  # no RMB restore in mouse-look

    def test_cast_skill_cursor_mode_clicks_slot(self):
        driver = _Driver()
        reader = SkillBarReader()
        released = []
        restored = []
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=reader,
            release_basic_attack=lambda: released.append("up"),
            restore_basic_attack=lambda: restored.append("down"),
            mouse_look_mode=False,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = bus.cast_skill("2", frame=frame)
        self.assertTrue(out["ok"])
        self.assertEqual(out["method"], "ui_click")
        self.assertEqual(len(driver.clicks), 1)
        self.assertEqual(driver.keys, [])

    def test_cast_skill_key_fallback_without_frame(self):
        driver = _Driver(trusted_keyboard=True)
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=SkillBarReader(),
            mouse_look_mode=True,
        )
        out = bus.cast_skill("3", frame=None)
        self.assertTrue(out["ok"])
        self.assertEqual(out["method"], "key")
        downs = [h for h in driver.held if h[0] == "down" and h[1] == "3"]
        self.assertGreaterEqual(len(downs), 1)
        self.assertTrue(all(h[2] is False for h in downs))

    def test_gather_interact_clicks_prompt_bbox(self):
        driver = _Driver()
        vision = _Vision(bbox=(100, 200, 40, 20), key="f")
        bus = GameActionBus(input_driver=driver, vision=vision)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        out = bus.gather_interact(frame=frame, preferred_key="f")
        self.assertTrue(out["ok"])
        self.assertEqual(out["method"], "ui_click")
        self.assertEqual(len(driver.clicks), 1)
        x, y, btn = driver.clicks[0]
        self.assertEqual(btn, "left")
        self.assertGreaterEqual(x, 100)
        self.assertLessEqual(x, 140)
        self.assertGreaterEqual(y, 200)
        self.assertLessEqual(y, 220)

    def test_exclusive_lock_order_release_then_restore(self):
        order = []
        driver = _Driver()
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=SkillBarReader(),
            release_basic_attack=lambda: order.append("release"),
            restore_basic_attack=lambda: order.append("restore"),
            mouse_look_mode=False,  # cursor mode restores RMB after UI skill click
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        bus.cast_skill("1", frame=frame)
        self.assertEqual(order[0], "release")
        self.assertEqual(order[-1], "restore")

    def test_hover_skill_with_alt_holds_alt_and_moves(self):
        driver = _Driver()
        seen = []
        bus = GameActionBus(
            input_driver=driver,
            skill_bar_reader=SkillBarReader(),
            mouse_look_mode=True,
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

        def on_hover(x, y):
            seen.append((x, y))
            return "deals damage"

        out = bus.hover_skill_with_alt("2", frame=frame, hold_sec=0.05, on_hover=on_hover)
        self.assertTrue(out["ok"])
        self.assertEqual(out["method"], "alt_hover")
        self.assertEqual(out["hover_payload"], "deals damage")
        self.assertEqual(driver.held[0][:2], ("down", "alt"))
        self.assertEqual(driver.held[-1][:2], ("up", "alt"))
        self.assertTrue(driver.moves)
        self.assertTrue(seen)
        self.assertEqual(driver.clicks, [])  # no cast click


if __name__ == "__main__":
    unittest.main()
