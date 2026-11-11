# -*- coding: utf-8 -*-
"""Keyboard and left click are posted to the game window, not only SendInput."""

import ctypes
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

from core.input_driver import (
    HARDWARE_SCANCODES,
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
        sentinput = []
        keybd = []

        def _spy(recorder, original):
            # F-1410: OS sink'i lambda ile değiştirmek yerine gerçek metoda
            # çağrıyı kaydeden şeffaf bir casus sarılır; gerçek kod çalışır.
            def _call(key, is_down):
                recorder.append((key, is_down))
                return original(key, is_down)

            return _call

        driver._post_key_to_game = _spy(posted, driver._post_key_to_game)
        driver._sendinput_key = _spy(sentinput, driver._sendinput_key)
        driver._keybd_event_key = _spy(keybd, driver._keybd_event_key)

        driver._hw_key_down("tab")
        driver._hw_key_up("1")
        self.assertEqual(posted, [])
        self.assertEqual(keybd, [("tab", True), ("1", False)])

        driver._hw_key_down("m", post_to_window=True)
        driver._hw_key_up("m", post_to_window=True)
        self.assertEqual(posted, [("m", True), ("m", False)])
        self.assertEqual(keybd, [("tab", True), ("1", False), ("m", True), ("m", False)])

    def test_post_key_to_game_posts_real_window_message(self):
        """F-1410: gerçek _post_key_to_game uçtan uca çalışır.

        Yalnızca OS sınırı sahtedir (PostMessageW + oyun hwnd'si); mesajı
        kuran ve gönderen kod gerçektir. Linux'ta Win32 olmadığı için
        platform ve ctypes.windll bu test süresince sahteyle değiştirilir.
        """
        driver = InputController(use_faker_input=False, test_mode=False)
        user32 = mock.MagicMock()
        user32.MapVirtualKeyW.return_value = 0
        user32.PostMessageW.return_value = 1
        user32.GetCursorPos.return_value = 0
        user32.ScreenToClient.return_value = 1
        driver._game_message_hwnd = lambda: 0xABCD
        with mock.patch.object(sys, "platform", "win32"):
            with mock.patch.object(ctypes, "windll", SimpleNamespace(user32=user32), create=True):
                self.assertTrue(driver._post_key_to_game("tab", True))
                self.assertTrue(driver._post_key_to_game("tab", False))
        self.assertEqual(user32.PostMessageW.call_count, 2)
        posted = [tuple(arg.value for arg in call.args) for call in user32.PostMessageW.call_args_list]
        self.assertEqual(posted[0][0], 0xABCD)
        self.assertEqual(posted[0][1:], key_window_message("tab", True))
        self.assertEqual(posted[1][0], 0xABCD)
        self.assertEqual(posted[1][1:], key_window_message("tab", False))

    def test_left_click_fails_without_interception_and_no_postmessage_sent(self):
        """N-3101: baypas kaldırıldı, Interception yoksa girdi başarısız sayılır.

        F-1410 dikişi silindi: `_hw_mouse_down/up` güvenilir yol
        (Interception/Faker) yokken sol tıkı pencereye PostMessage ile de
        göndermez; False döner. Sağ tık zaten hiçbir yola uğramıyordu."""
        driver = InputController(use_faker_input=False, test_mode=False)
        driver._interception = None
        driver._faker = None
        user32 = mock.MagicMock()
        user32.PostMessageW.return_value = 1
        with mock.patch.object(sys, "platform", "win32"):
            with mock.patch.object(ctypes, "windll", SimpleNamespace(user32=user32), create=True):
                self.assertFalse(driver._hw_mouse_down("left"))
                self.assertFalse(driver._hw_mouse_down("right"))
                self.assertFalse(driver._hw_mouse_up("left"))
                self.assertFalse(driver._hw_mouse_up("right"))
        user32.PostMessageW.assert_not_called()

    def test_open_chat_is_closed_and_skill_key_not_dispatched_in_test_mode(self):
        """F-1325: eski test adı 'skill_key_still_sent' diyordu ama test
        modunda tuş hiçbir yola gönderilmez — yalnızca sohbet kapanır.
        Ad ve assertion artık gerçek sözleşmeyi söyler: sohbet kapanır,
        skill tuşu `_hw_key_down`'a hiç ulaşmaz."""
        driver = InputController(use_faker_input=False, test_mode=True, chat_box_guard_enabled=True)
        driver.set_chat_box_open(True)

        dispatched = []
        original_hw_key_down = driver._hw_key_down

        def _spy(key, post_to_window=False):
            dispatched.append((key, post_to_window))
            return original_hw_key_down(key, post_to_window=post_to_window)

        driver._hw_key_down = _spy

        hold = driver.press_key("1")

        self.assertGreater(hold, 0.0)
        self.assertEqual(driver._chat_guard_esc_count, 1)
        self.assertFalse(driver._chat_open_flag)
        self.assertEqual(dispatched, [])

    def test_open_chat_is_closed_and_skill_key_sent_via_trusted_path(self):
        """F-1325 (tamamlayıcı): test_mode kapalıyken ve güvenilir
        Interception yolu varken sohbet kapanır VE skill tuşu donanım
        yoluna gerçekten gönderilir (scancode down/up). ESC de aynı
        güvenilir yoldan gönderilir (D05, fee6e3e), o yüzden `sent`
        önce ESC çiftini, sonra skill tuşunu içerir."""
        driver = InputController(use_faker_input=False, test_mode=False, chat_box_guard_enabled=True)

        sent = []

        class _FakeInterception:
            available = True

            def send_key(self, scancode, is_down, extended=False):
                sent.append((scancode, is_down))
                return True

        driver._interception = _FakeInterception()
        driver.set_chat_box_open(True)

        hold = driver.press_key("1")

        self.assertGreater(hold, 0.0)
        self.assertEqual(driver._chat_guard_esc_count, 1)
        # D05 öncesi close_chat_box_if_open ESC'yi yalnızca sayıyor,
        # göndermiyordu (d3aec9f); o yüzden eski beklenti `sent`'te
        # yalnızca skill tuşunu görüyordu. D05'ten beri ESC gerçek
        # gönderimdir ve güvenilir yoldan önce gelir.
        self.assertEqual(
            sent,
            [
                (HARDWARE_SCANCODES["esc"], True),
                (HARDWARE_SCANCODES["esc"], False),
                (HARDWARE_SCANCODES["1"], True),
                (HARDWARE_SCANCODES["1"], False),
            ],
        )


if __name__ == "__main__":
    unittest.main()
