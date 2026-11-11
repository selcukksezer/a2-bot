# -*- coding: utf-8 -*-
"""Bare F8/F9/F10 must not call bot handlers. Ctrl+Alt chords must."""

import unittest
from types import SimpleNamespace
from unittest import mock

from config.schema import (
    DEFAULT_MENU_CHORD,
    DEFAULT_NODE_CHORD,
    DEFAULT_PAUSE_CHORD,
    HotkeysSettings,
)
from core.input_driver import HotkeyListener
from core.opsec_crypto import NativeWin32HotkeyManager


class _Controller:
    def __init__(self) -> None:
        self._sending_virtual_key = False
        self._last_virtual_key_ts = 0.0
        self._last_f8_ts = 0.0
        self._last_f9_ts = 0.0
        self._last_f10_ts = 0.0
        self._last_resume_ts = 0.0
        self.route_active = False
        self.pause_on_any_key = False
        self.freed = 0
        self.released = 0

    def is_bot_pressing(self, _key: str) -> bool:
        return False

    def free_mouse_cursor(self) -> None:
        self.freed += 1

    def release_all_movement_keys(self) -> None:
        self.released += 1


class HotkeyChordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = _Controller()
        self.listener = HotkeyListener(self.controller)
        self.listener._async_invoke = False
        self.hits = []
        self.listener.on_f8 = lambda: self.hits.append("f8")
        self.listener.on_f9 = lambda: self.hits.append("f9")
        self.listener.on_f10 = lambda: self.hits.append("f10")

    def test_config_defaults_are_chords(self) -> None:
        settings = HotkeysSettings().validate()
        self.assertEqual(settings.record_node_f8, DEFAULT_NODE_CHORD)
        self.assertEqual(settings.toggle_menu_f9, DEFAULT_MENU_CHORD)
        self.assertEqual(settings.pause_resume_f10, DEFAULT_PAUSE_CHORD)
        loaded = HotkeysSettings.from_dict({})
        self.assertEqual(loaded.record_node_f8, "ctrl+alt+f8")
        self.assertEqual(loaded.toggle_menu_f9, "ctrl+alt+f9")
        self.assertEqual(loaded.pause_resume_f10, "ctrl+alt+f10")
        upgraded = HotkeysSettings.from_dict(
            {"record_node_f8": "f8", "toggle_menu_f9": "f9", "pause_resume_f10": "f10"}
        )
        self.assertEqual(upgraded.record_node_f8, "ctrl+alt+f8")
        self.assertEqual(upgraded.toggle_menu_f9, "ctrl+alt+f9")
        self.assertEqual(upgraded.pause_resume_f10, "ctrl+alt+f10")
        custom = HotkeysSettings.from_dict({"toggle_menu_f9": "ctrl+shift+f9"})
        self.assertEqual(custom.toggle_menu_f9, "ctrl+shift+f9")

    def test_bare_keys_do_not_call_handlers(self) -> None:
        for key in ("f8", "f9", "f10"):
            decision = self.listener.handle_bot_chord(key, frozenset())
            self.assertEqual(decision, "ignored")
        self.assertEqual(self.hits, [])
        self.assertEqual(self.controller.freed, 0)
        self.assertEqual(self.controller.released, 0)

    def test_ctrl_alt_chords_call_handlers(self) -> None:
        mods = frozenset({"ctrl", "alt"})
        self.assertEqual(self.listener.handle_bot_chord("f8", mods), "handled")
        self.assertEqual(self.listener.handle_bot_chord("f9", mods), "handled")
        self.assertEqual(self.listener.handle_bot_chord("f10", mods), "handled")
        self.assertEqual(self.hits, ["f8", "f9", "f10"])
        self.assertEqual(self.controller.freed, 2)
        self.assertEqual(self.controller.released, 1)

    def test_native_binding_uses_norepeat_and_refuses_bare_keys(self) -> None:
        vk, mods = NativeWin32HotkeyManager.chord_binding("ctrl+alt+f10")
        self.assertEqual(vk, NativeWin32HotkeyManager.VK_F10)
        self.assertTrue(mods & NativeWin32HotkeyManager.MOD_CONTROL)
        self.assertTrue(mods & NativeWin32HotkeyManager.MOD_ALT)
        self.assertTrue(mods & NativeWin32HotkeyManager.MOD_NOREPEAT)
        self.assertEqual(NativeWin32HotkeyManager.chord_binding("f8"), (0, 0))
        self.assertEqual(NativeWin32HotkeyManager.chord_binding("f9"), (0, 0))
        self.assertEqual(NativeWin32HotkeyManager.chord_binding("f10"), (0, 0))

    def test_register_and_unregister_do_not_raise(self) -> None:
        # Exercises the real register()/unregister() hook-install path
        # (returns False when the `keyboard` module is unavailable, True
        # when a hook was actually installed) and always cleans up.
        result = self.listener.register()
        self.assertIsInstance(result, bool)
        self.listener.unregister()
        self.assertIsNone(self.listener._hook_handle)
        self.assertIsNone(self.listener._release_hook_handle)
        self.listener.unregister()  # double unregister must stay safe

    def test_hook_callback_dispatches_bare_keys_without_handlers(self) -> None:
        # _on_key_event is the real keyboard-hook callback: a bare F8/F9/F10
        # arriving through the hook must not call any bot handler.
        for key in ("f8", "f9", "f10"):
            self.listener._on_key_event(SimpleNamespace(name=key, scan_code=0))
        self.assertEqual(self.hits, [])
        self.assertEqual(self.controller.freed, 0)
        self.assertEqual(self.controller.released, 0)

    def test_hook_callback_dispatches_ctrl_alt_chord(self) -> None:
        # Same real hook callback, but with Ctrl+Alt held. Only the OS
        # modifier query is stubbed (keyboard.is_pressed has no real keys
        # in CI); dispatch, debounce, accept and invoke are all real.
        with mock.patch.object(
            HotkeyListener, "_held_modifiers", return_value={"ctrl", "alt"}
        ):
            for key in ("f8", "f9", "f10"):
                self.listener._on_key_event(SimpleNamespace(name=key, scan_code=0))
        self.assertEqual(self.hits, ["f8", "f9", "f10"])

    def test_hook_callback_scan_code_fallback_names_key(self) -> None:
        # Windows sometimes delivers F10 without a name; the hook must
        # recover it from the scan code (0x44) instead of dropping it.
        self.assertEqual(
            HotkeyListener._normalize_hotkey_name(
                SimpleNamespace(name=None, scan_code=0x44)
            ),
            "f10",
        )
        self.listener._on_key_event(SimpleNamespace(name=None, scan_code=0x44))
        self.assertEqual(self.hits, [])  # bare (no modifiers) -> ignored
        self.listener._on_key_release(SimpleNamespace(name="f10", scan_code=0x44))
        self.assertFalse(self.listener._f10_physically_down)


if __name__ == "__main__":
    unittest.main()
