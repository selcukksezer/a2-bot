# -*- coding: utf-8 -*-
"""D78 dev-improvement tests.

Covers the three D78 improvements (see dev-results/D78.json):
1. parse_hotkey_chord is cached (config/schema.py).
2. SettingsManager.save() validates exactly once per save (config/schema.py).
3. RouteRunner / AutoGatherBot long sleeps are ESC-interruptible
   (modules/route_runner.py, modules/auto_gather.py).

Windows-only modules (winsound, keyboard) do not exist on this Linux VM,
so they are stubbed in sys.modules before import. Temp files go under
~/workspace/bench-d78 (never /tmp: the tmpfs there is full).
"""

import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest import mock

# --- Windows-only stubs (must precede the modules.* imports) ---
if "winsound" not in sys.modules:
    sys.modules["winsound"] = types.ModuleType("winsound")
_keyboard_stub = types.ModuleType("keyboard")
_keyboard_stub.add_hotkey = lambda *a, **k: None
_keyboard_stub.remove_hotkey = lambda *a, **k: None
_keyboard_stub.is_pressed = lambda *a, **k: False
if "keyboard" not in sys.modules:
    sys.modules["keyboard"] = _keyboard_stub

from config.schema import (  # noqa: E402
    BotSettings,
    SettingsManager,
    canonical_hotkey_chord,
    format_hotkey_label,
    match_hotkey_action,
    parse_hotkey_chord,
)

import modules.auto_gather as auto_gather_mod  # noqa: E402
import modules.route_runner as route_runner_mod  # noqa: E402

_TMP_ROOT = Path.home() / "workspace" / "bench-d78" / "pytest-tmp"
_TMP_ROOT.mkdir(parents=True, exist_ok=True)


class TestParseHotkeyChordCache(unittest.TestCase):
    def test_values_unchanged(self):
        self.assertEqual(parse_hotkey_chord("ctrl+alt+f10"), ("f10", frozenset({"ctrl", "alt"})))
        self.assertEqual(parse_hotkey_chord("ctrl+alt+f8"), ("f8", frozenset({"ctrl", "alt"})))
        self.assertEqual(parse_hotkey_chord("f10"), ("f10", frozenset()))
        self.assertEqual(parse_hotkey_chord(""), ("", frozenset()))
        self.assertEqual(parse_hotkey_chord("CONTROL+Alt+F9"), ("f9", frozenset({"ctrl", "alt"})))
        # Downstream helpers keep their contracts.
        self.assertEqual(canonical_hotkey_chord("f10", "ctrl+alt+f10"), "ctrl+alt+f10")
        self.assertEqual(format_hotkey_label("ctrl+alt+f10"), "Ctrl+Alt+F10")
        chords = {"node": "ctrl+alt+f8", "menu": "ctrl+alt+f9"}
        self.assertEqual(match_hotkey_action("f8", ["ctrl", "alt"], chords), "node")
        self.assertIsNone(match_hotkey_action("f8", [], chords))  # bare F8 never matches

    def test_repeat_calls_hit_cache(self):
        info_before = parse_hotkey_chord.cache_info()
        parse_hotkey_chord("ctrl+shift+f12")
        parse_hotkey_chord("ctrl+shift+f12")
        parse_hotkey_chord("ctrl+shift+f12")
        info_after = parse_hotkey_chord.cache_info()
        self.assertEqual(info_after.misses - info_before.misses, 1)
        self.assertEqual(info_after.hits - info_before.hits, 2)


class TestSaveValidatesOnce(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(dir=str(_TMP_ROOT))
        self.settings_path = Path(self._tmp.name) / "settings.json"
        self.mgr = SettingsManager(settings_path=self.settings_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_save_calls_validate_once(self):
        count = {"n": 0}
        orig = BotSettings.validate

        def counting(self):
            count["n"] += 1
            return orig(self)

        with mock.patch.object(BotSettings, "validate", counting):
            count["n"] = 0
            self.mgr.settings.gathering.f_interval_min = 1.7
            self.assertTrue(self.mgr.save())
        self.assertEqual(count["n"], 1)

    def test_save_roundtrip_unchanged(self):
        self.mgr.settings.gathering.f_interval_min = 1.7
        self.mgr.settings.route.rest_minutes = 7
        self.assertTrue(self.mgr.save())
        reloaded = SettingsManager(settings_path=self.settings_path).settings
        self.assertEqual(reloaded.gathering.f_interval_min, 1.7)
        self.assertEqual(reloaded.route.rest_minutes, 7)

    def test_export_payload_matches_to_dict(self):
        # Unknown file keys survive the merge, in-memory values win.
        self.settings_path.write_text(
            '{"keep_me": 1, "gathering": {"f_interval_min": 9.9}}',
            encoding="utf-8",
        )
        mgr = SettingsManager(settings_path=self.settings_path)
        mgr.settings.gathering.f_interval_min = 1.7
        payload = mgr.export_payload()
        self.assertEqual(payload["keep_me"], 1)
        self.assertEqual(payload["gathering"]["f_interval_min"], 1.7)
        self.assertEqual(payload["gathering"]["f_interval_max"], mgr.settings.to_dict()["gathering"]["f_interval_max"])


def _esc_after(deadline_s):
    t0 = time.monotonic()

    def _pressed(key):
        return time.monotonic() - t0 > deadline_s

    return _pressed


class TestInterruptibleSleep(unittest.TestCase):
    def test_route_runner_sleep_returns_false_on_esc(self):
        with mock.patch.object(route_runner_mod, "_load_route_settings", return_value={}), \
             mock.patch.object(route_runner_mod, "RouteManager", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "Vision", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "InputDriver", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "MapNavigator", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "NodeRecorder", return_value=mock.MagicMock()):
            runner = route_runner_mod.RouteRunner()
            runner.running = True
            with mock.patch.object(route_runner_mod.keyboard, "is_pressed", return_value=True):
                t0 = time.perf_counter()
                self.assertFalse(runner._sleep_interruptible(30))
                self.assertLess(time.perf_counter() - t0, 1.0)
            with mock.patch.object(route_runner_mod.keyboard, "is_pressed", return_value=False):
                t0 = time.perf_counter()
                self.assertTrue(runner._sleep_interruptible(0.3))
                self.assertGreaterEqual(time.perf_counter() - t0, 0.3)

    def test_route_runner_start_returns_fast_on_esc_during_rest_wait(self):
        fake_routes = mock.MagicMock()
        fake_routes.get_next_ready_node.return_value = None
        with mock.patch.object(route_runner_mod, "_load_route_settings", return_value={}), \
             mock.patch.object(route_runner_mod, "RouteManager", return_value=fake_routes), \
             mock.patch.object(route_runner_mod, "Vision", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "InputDriver", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "MapNavigator", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod, "NodeRecorder", return_value=mock.MagicMock()), \
             mock.patch.object(route_runner_mod.keyboard, "is_pressed", _esc_after(1.0)):
            runner = route_runner_mod.RouteRunner()
            t0 = time.perf_counter()
            runner.start()
            elapsed = time.perf_counter() - t0
            # Before the fix this took ~30 s (single time.sleep(30)).
            self.assertLess(elapsed, 5.0)
            self.assertFalse(runner.running)

    def test_auto_gather_sleep_returns_false_on_esc(self):
        with mock.patch.object(auto_gather_mod, "_load_gather_settings", return_value={}), \
             mock.patch.object(auto_gather_mod, "Vision", return_value=mock.MagicMock()), \
             mock.patch.object(auto_gather_mod, "InputDriver", return_value=mock.MagicMock()):
            bot = auto_gather_mod.AutoGatherBot()
            bot.running = True
            with mock.patch.object(auto_gather_mod.keyboard, "is_pressed", return_value=True):
                t0 = time.perf_counter()
                self.assertFalse(bot._sleep_interruptible(1.5))
                self.assertLess(time.perf_counter() - t0, 1.0)
            with mock.patch.object(auto_gather_mod.keyboard, "is_pressed", return_value=False):
                t0 = time.perf_counter()
                self.assertTrue(bot._sleep_interruptible(0.3))
                self.assertGreaterEqual(time.perf_counter() - t0, 0.3)


if __name__ == "__main__":
    unittest.main()
