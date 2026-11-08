# -*- coding: utf-8 -*-
"""Keyboard and left click are posted to the game window, not only SendInput."""

import unittest

from core.input_driver import (
    WM_KEYDOWN,
    WM_KEYUP,
    WM_LBUTTONDOWN,
    WM_LBUTTONUP,
    WM_SYSKEYDOWN,
    InputController,
    key_window_message,
    mouse_button_window_message,
)


class GameWindowMessageTests(unittest.TestCase):
    def test_tab_and_map_key_use_virtual_key_messages(self):
        tab_down = key_window_message("tab", True)
        tab_up = key_window_message("tab", False)
        map_down = key_window_message("m", True)

        self.assertEqual(tab_down, (WM_KEYDOWN, 0x09, 1 | (0x0F << 16)))
        self.assertEqual(tab_up[0], WM_KEYUP)
        self.assertEqual(tab_up[1], 0x09)
        self.assertNotEqual(tab_up[2] & (1 << 31), 0)
        self.assertEqual(map_down[0], WM_KEYDOWN)
        self.assertEqual(map_down[1], 0x4D)
        self.assertEqual((map_down[2] >> 16) & 0xFF, 0x32)

    def test_alt_is_a_system_key_and_right_click_is_not_posted(self):
        alt_down = key_window_message("alt", True)
        self.assertEqual(alt_down[0], WM_SYSKEYDOWN)
        self.assertIsNone(mouse_button_window_message("right", True, 1, 1))

    def test_left_click_uses_client_point(self):
        down = mouse_button_window_message("left", True, 10, 20)
        up = mouse_button_window_message("left", False, 10, 20)
        self.assertEqual(down, (WM_LBUTTONDOWN, 0x0001, (20 << 16) | 10))
        self.assertEqual(up, (WM_LBUTTONUP, 0, (20 << 16) | 10))

    def test_key_down_skips_postmessage_by_default_but_fires_scancode_paths(self):
        driver = InputController(use_faker_input=False, test_mode=True)
        posted = []
        keybd = []
        driver._post_key_to_game = lambda key, is_down: posted.append((key, is_down)) or True
        driver._sendinput_key = lambda key, is_down: True
        driver._keybd_event_key = lambda key, is_down: keybd.append((key, is_down)) or True

        self.assertTrue(driver._hw_key_down("tab"))
        driver._hw_key_up("1")
        self.assertEqual(posted, [])
        self.assertEqual(keybd, [("tab", True), ("1", False)])

        self.assertTrue(driver._hw_key_down("m", post_to_window=True))
        driver._hw_key_up("m", post_to_window=True)
        self.assertEqual(posted, [("m", True), ("m", False)])

    def test_left_click_is_posted_and_right_click_is_not(self):
        driver = InputController(use_faker_input=False, test_mode=False)
        posted = []
        driver._post_mouse_button_to_game = lambda button, is_down: posted.append((button, is_down)) or True
        driver._sendinput_mouse_button = lambda button, is_down: True

        driver._hw_mouse_down("left")
        driver._hw_mouse_down("right")
        driver._hw_mouse_up("left")
        driver._hw_mouse_up("right")

        self.assertEqual(posted, [("left", True), ("left", False)])

    def test_open_chat_is_closed_and_skill_key_still_sent(self):
        driver = InputController(use_faker_input=False, test_mode=True, chat_box_guard_enabled=True)
        driver.set_chat_box_open(True)

        hold = driver.press_key("1")

        self.assertGreater(hold, 0.0)
        self.assertEqual(driver._chat_guard_esc_count, 1)


if __name__ == "__main__":
    unittest.main()
