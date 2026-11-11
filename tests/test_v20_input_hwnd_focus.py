# -*- coding: utf-8 -*-
"""
v20 input_driver HWND / EnumWindows focus detection.

Live evidence (session_2026-11-11_192331.log):
  repeated GameActionBus "key '…' sent while game window NOT focused
  (focus_game_window False)" while user was on the game screen.

Root causes addressed (code):
1) `_find_aion_hwnd` used WINFUNCTYPE(ctypes.c_bool, …) — BOOL is 4 bytes,
   c_bool is 1; watchdog uses WINFUNCTYPE(BOOL, HWND, LPARAM).
2) GetForegroundWindow / FindWindowExW lacked restype=HWND → Win64
   truncates to c_long (32-bit), so fg != game even when focused.

These unit tests mock user32; they do NOT claim live "NOT focused" is gone.
"""
from __future__ import annotations

import ctypes
import sys
import types
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import input_driver as id_mod  # noqa: E402
from core.input_driver import InputDriver  # noqa: E402


def test_wndenumproc_uses_bool_not_c_bool():
    """Module callback type must match watchdog: BOOL return, HWND args."""
    if not sys.platform.startswith("win"):
        assert id_mod._WNDENUMPROC is None
        return
    assert id_mod._WNDENUMPROC is not None
    # ctypes exposes argtypes/restype on the function prototype factory result
    # via a probe callback instance.
    probe = id_mod._WNDENUMPROC(lambda hwnd, lp: True)
    # restype of the callback type is BOOL-compatible (size 4), not c_bool (1).
    assert ctypes.sizeof(probe.restype) == ctypes.sizeof(ctypes.wintypes.BOOL)
    assert ctypes.sizeof(probe.restype) != ctypes.sizeof(ctypes.c_bool)


def test_ensure_prototypes_sets_hwnd_restypes():
    if not sys.platform.startswith("win"):
        pytest.skip("Windows-only")
    driver = InputDriver.__new__(InputDriver)
    driver.test_mode = True
    fake = MagicMock(name="user32")
    # Start without the flag so ensure runs.
    fake._gatherion_hwnd_proto = False
    with mock.patch.object(ctypes, "windll", types.SimpleNamespace(user32=fake), create=True):
        # Re-bind windll.user32 for the method under test
        with mock.patch.object(ctypes.windll, "user32", fake, create=True):
            out = driver._ensure_user32_hwnd_prototypes()
    assert out is fake
    assert fake.GetForegroundWindow.restype is ctypes.wintypes.HWND
    assert fake.FindWindowExW.restype is ctypes.wintypes.HWND
    assert fake.FindWindowW.restype is ctypes.wintypes.HWND
    assert fake.EnumWindows.restype is ctypes.wintypes.BOOL


def test_find_aion_hwnd_uses_persistent_bool_callback():
    if not sys.platform.startswith("win"):
        pytest.skip("Windows-only")
    driver = InputDriver.__new__(InputDriver)
    driver.test_mode = True
    driver._cached_game_hwnd = 0
    driver._game_hwnd_cache_ts = 0.0

    titles = {0x10001: "Discord", 0x10002: "AION 2", 0x10003: "Chrome"}
    fake = MagicMock(name="user32")
    fake._gatherion_hwnd_proto = False
    fake.IsWindowVisible.side_effect = lambda hwnd: True
    fake.IsWindow.side_effect = lambda hwnd: True

    def _len(hwnd):
        return len(titles.get(int(getattr(hwnd, "value", hwnd) or int(hwnd)), ""))

    def _text(hwnd, buf, _n):
        key = int(getattr(hwnd, "value", hwnd) or int(hwnd))
        buf.value = titles.get(key, "")
        return len(buf.value)

    fake.GetWindowTextLengthW.side_effect = _len
    fake.GetWindowTextW.side_effect = _text

    def _enum(callback, _lp):
        for hwnd in titles:
            if callback(hwnd, 0) is False:
                return True
        return True

    fake.EnumWindows.side_effect = _enum
    fake.FindWindowW.return_value = 0

    with mock.patch.object(sys, "platform", "win32"):
        with mock.patch.object(id_mod, "_WNDENUMPROC", lambda fn: fn):
            with mock.patch.object(ctypes, "windll", types.SimpleNamespace(user32=fake), create=True):
                with mock.patch.object(ctypes.windll, "user32", fake, create=True):
                    hwnd = driver._find_aion_hwnd()

    assert hwnd == 0x10002
    assert driver._enum_aion_cb_ref is not None
    assert fake.EnumWindows.call_count == 1


def test_focus_game_window_true_when_fg_equals_game():
    if not sys.platform.startswith("win"):
        pytest.skip("Windows-only")
    driver = InputDriver.__new__(InputDriver)
    driver.test_mode = False
    driver._cached_game_hwnd = 0xABCDE
    driver._game_hwnd_cache_ts = 10**12  # fresh cache
    driver.motion_aborted = lambda: False  # type: ignore[method-assign]

    fake = MagicMock(name="user32")
    fake._gatherion_hwnd_proto = False
    fake.IsWindow.return_value = True
    fake.GetForegroundWindow.return_value = 0xABCDE

    with mock.patch.object(sys, "platform", "win32"):
        with mock.patch.object(ctypes, "windll", types.SimpleNamespace(user32=fake), create=True):
            with mock.patch.object(ctypes.windll, "user32", fake, create=True):
                assert driver.focus_game_window() is True
    # Must not steal focus when already on game.
    fake.SetForegroundWindow.assert_not_called()


def test_focus_game_window_true_when_same_process_different_hwnd():
    """Child vs top-level HWND mismatch must not report NOT focused."""
    if not sys.platform.startswith("win"):
        pytest.skip("Windows-only")
    driver = InputDriver.__new__(InputDriver)
    driver.test_mode = False
    driver._cached_game_hwnd = 0x1000
    driver._game_hwnd_cache_ts = 10**12
    driver.motion_aborted = lambda: False  # type: ignore[method-assign]

    fake = MagicMock(name="user32")
    fake._gatherion_hwnd_proto = False
    fake.IsWindow.return_value = True
    fake.GetForegroundWindow.return_value = 0x1001  # different hwnd, same PID

    def _gtp(hwnd, pid_ptr):
        if pid_ptr:
            pid_ptr._obj.value = 4242
        return 1

    fake.GetWindowThreadProcessId.side_effect = _gtp

    with mock.patch.object(sys, "platform", "win32"):
        with mock.patch.object(ctypes, "windll", types.SimpleNamespace(user32=fake), create=True):
            with mock.patch.object(ctypes.windll, "user32", fake, create=True):
                assert driver.focus_game_window() is True
    fake.SetForegroundWindow.assert_not_called()


def test_hwnd_to_int_preserves_high_bits():
    driver = InputDriver.__new__(InputDriver)
    # c_void_p round-trip must not collapse to 32-bit.
    high = 0x0000000123456789
    wrapped = ctypes.c_void_p(high)
    assert driver._hwnd_to_int(wrapped) == high
