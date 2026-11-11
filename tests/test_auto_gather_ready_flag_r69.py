# -*- coding: utf-8 -*-
"""
R69 — AutoGather readiness probe tests (S-2703, S-2704).

Windows-only stdlib (winsound) and the `keyboard` package are stubbed
before import so the suite runs on Linux CI. The bot instance is built
with object.__new__ so the real Vision/InputDriver constructors (heavy,
Windows-oriented) are never needed here; fake vision/driver objects
exercise the probe logic directly.
"""

import os
import sys
import types
import unittest
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if "winsound" not in sys.modules:
    _winsound = types.ModuleType("winsound")
    _winsound.Beep = lambda *a, **k: None
    sys.modules["winsound"] = _winsound
if "keyboard" not in sys.modules:
    _keyboard = types.ModuleType("keyboard")
    _keyboard.add_hotkey = lambda *a, **k: None
    _keyboard.remove_hotkey = lambda *a, **k: None
    _keyboard.is_pressed = lambda *a, **k: False
    _keyboard.unhook_all = MagicMock(name="unhook_all")
    sys.modules["keyboard"] = _keyboard

import keyboard  # noqa: E402  (stubbed above on Linux)
from modules import auto_gather as ag  # noqa: E402
from modules.auto_gather import AutoGatherBot, validate_auto_gather_contract  # noqa: E402


class _FakeVision:
    def __init__(self, ready=True):
        self._ready = ready

    def check_gather_prompt(self, *args, **kwargs):
        return (False, 0, None)

    def get_vision_engine_ready_flag(self):
        return {"module": "vision", "ready": self._ready}


class _FakeDriver:
    def __init__(self, trusted=True, test_mode=False):
        self.keyboard_inject_trusted = trusted
        self.test_mode = test_mode
        self.active_backend_name = (
            "Interception" if trusted else "ctypes.SendInput"
        )

    def send_gather_key(self, *args, **kwargs):
        return 0.0


def _bot(vision=None, driver=None, threshold=0.75, f_min=1.5, f_max=3.0):
    bot = object.__new__(AutoGatherBot)
    bot.vision = vision if vision is not None else _FakeVision(True)
    bot.driver = driver if driver is not None else _FakeDriver(True)
    bot.match_threshold = threshold
    bot.min_f_interval = f_min
    bot.max_f_interval = f_max
    bot._hotkey_names = []
    return bot


class TestAutoGatherReadyFlag(unittest.TestCase):
    def test_ready_when_vision_and_trusted_driver_ok(self):
        flag = _bot().get_auto_gather_ready_flag()
        self.assertTrue(flag["ready"], flag)
        self.assertEqual(flag["problems"], [])
        self.assertTrue(flag["vision_ready"])
        self.assertTrue(flag["driver_ready"])

    def test_not_ready_when_vision_not_ready(self):
        flag = _bot(vision=_FakeVision(False)).get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertIn("vision_not_ready", flag["problems"])
        self.assertFalse(flag["vision_ready"])

    def test_not_ready_when_driver_has_no_trusted_backend(self):
        # Soft SendInput is dropped by AION 2: no trusted backend = not ready.
        flag = _bot(driver=_FakeDriver(trusted=False)).get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertTrue(
            any(p.startswith("no_trusted_keyboard_backend") for p in flag["problems"]),
            flag["problems"],
        )
        self.assertFalse(flag["driver_ready"])

    def test_ready_in_test_mode_despite_untrusted_driver(self):
        flag = _bot(
            driver=_FakeDriver(trusted=False, test_mode=True)
        ).get_auto_gather_ready_flag()
        self.assertTrue(flag["ready"], flag)
        self.assertTrue(flag["driver_ready"])

    def test_not_ready_on_bad_threshold_and_bad_interval(self):
        flag = _bot(threshold=1.5).get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertTrue(
            any("match_threshold" in p for p in flag["problems"]), flag["problems"]
        )
        flag = _bot(f_min=3.0, f_max=1.0).get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertTrue(
            any("f_interval" in p for p in flag["problems"]), flag["problems"]
        )

    def test_not_ready_when_vision_or_driver_missing(self):
        bot = _bot()
        bot.vision = None
        flag = bot.get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertIn("vision_missing", flag["problems"])
        bot = _bot()
        bot.driver = None
        flag = bot.get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertIn("driver_missing", flag["problems"])

    def test_probe_never_raises(self):
        bot = _bot()
        bot.vision = object()  # no methods at all
        bot.driver = object()
        flag = bot.get_auto_gather_ready_flag()
        self.assertFalse(flag["ready"])
        self.assertIn("problems", flag)


class TestValidateAutoGatherContract(unittest.TestCase):
    def test_contract_true_with_real_classes(self):
        self.assertTrue(validate_auto_gather_contract())

    def test_contract_false_when_capability_missing(self):
        orig = ag.get_auto_gather_capabilities
        ag.get_auto_gather_capabilities = lambda: []
        try:
            self.assertFalse(validate_auto_gather_contract())
        finally:
            ag.get_auto_gather_capabilities = orig


class TestUnregisterHotkeys(unittest.TestCase):
    def test_failed_removal_never_calls_unhook_all(self):
        # S-2704: a failed remove_hotkey must not rip out other modules'
        # hotkeys via keyboard.unhook_all().
        removed = []

        def remove_hotkey(name):
            removed.append(name)
            if name == "ctrl+alt+f9":
                raise RuntimeError("hook gone")

        keyboard.remove_hotkey = remove_hotkey
        keyboard.unhook_all = MagicMock(name="unhook_all")
        try:
            bot = _bot()
            bot._hotkey_names = ["ctrl+alt+f9", "ctrl+alt+f10"]
            bot._unregister_hotkeys()
            self.assertEqual(removed, ["ctrl+alt+f9", "ctrl+alt+f10"])
            keyboard.unhook_all.assert_not_called()
            self.assertEqual(bot._hotkey_names, [])
        finally:
            keyboard.remove_hotkey = lambda *a, **k: None


if __name__ == "__main__":
    unittest.main()
