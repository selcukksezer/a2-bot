# -*- coding: utf-8 -*-
"""
Unit tests for automatic bot pause on Alt-Tab / game focus loss.
Verifies ForegroundFocusGuard, BotWorker integration, and InputController focus protection.
"""

import os
import time
import pytest
from core.dpi_focus_guard import ForegroundFocusGuard
from core.state_machine import BotMode, BotState, BotWorker
from core.input_driver import InputController
from config.schema import SettingsManager, BotSettings


def _wait_until(predicate, timeout_sec=2.0, poll_sec=0.01):
    """Poll ``predicate`` until true or ``timeout_sec`` elapses.

    The focus-monitor test used to sleep fixed 0.04/0.06/0.06 s and then
    assert on whatever the background thread had managed to observe — a
    slow CI tick (check interval is 0.02 s) could miss the event and fail
    the test flakily. Polling waits until the event actually arrives
    (usually ~one interval) and only fails after a real 2 s timeout.
    """
    deadline = time.monotonic() + timeout_sec
    while True:
        if predicate():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(poll_sec)


_SHARED_WORKER = None


def _shared_worker():
    """One real BotWorker for the whole module.

    BotWorker() costs ~2 s after warm-up (~8-14 s cold: lazy ONNX model
    load in _init_optional_subsystem) and both worker tests below only
    need a live worker they resume/pause — each test calls resume() first,
    which resets pause state. Sharing saves one construction per run;
    the worker stays the real production worker (no __new__ bypass —
    see test_watchdog_stall_resume.py D89/F-1554 note).
    """
    global _SHARED_WORKER
    if _SHARED_WORKER is None:
        _SHARED_WORKER = BotWorker()
    return _SHARED_WORKER


def test_foreground_focus_guard_game_recognition():
    guard = ForegroundFocusGuard()

    # Allowed game titles
    guard.set_simulated_foreground_window("AION 2", 9096)
    focused, reason = guard.is_game_window_focused()
    assert focused is True, f"Expected AION 2 to be focused, got: {reason}"

    guard.set_simulated_foreground_window("AionClient - UnrealWindow", 1234)
    focused, reason = guard.is_game_window_focused()
    assert focused is True

    guard.set_simulated_foreground_window("NCSoft Game Client", 5678)
    focused, reason = guard.is_game_window_focused()
    assert focused is True

    guard.set_simulated_foreground_window("아이온 2", 9999)
    focused, reason = guard.is_game_window_focused()
    assert focused is True


def test_foreground_focus_guard_alt_tab_blocked_apps():
    guard = ForegroundFocusGuard()

    # User Alt-Tabs to Chrome
    guard.set_simulated_foreground_window("Google Chrome - YouTube", 5001)
    focused, reason = guard.is_game_window_focused()
    assert focused is False
    assert "Focus stolen" in reason or "Google Chrome" in reason

    # User Alt-Tabs to Discord
    guard.set_simulated_foreground_window("Discord #general", 5002)
    focused, reason = guard.is_game_window_focused()
    assert focused is False

    # User Alt-Tabs to Desktop / Explorer
    guard.set_simulated_foreground_window("Program Manager", 5003)
    focused, reason = guard.is_game_window_focused()
    assert focused is False

    # User Alt-Tabs to arbitrary foreign window
    guard.set_simulated_foreground_window("Calculator", 5004)
    focused, reason = guard.is_game_window_focused()
    assert focused is False


def test_foreground_focus_guard_gatherion_ui_exemption():
    guard = ForegroundFocusGuard()

    # Interacting with Gatherion HUD or Menu should not be flagged as foreign app stealing focus
    guard.set_simulated_foreground_window("Gatherion-Local v2.0 (Türkçe) — Oyun İçi Durum Paneli", 1000)
    focused, reason = guard.is_game_window_focused(allow_gatherion_ui=True)
    assert focused is True
    assert "Gatherion UI" in reason


def test_foreground_focus_guard_monitoring_thread():
    guard = ForegroundFocusGuard()
    events = []

    guard.set_simulated_foreground_window("AION 2", 9096)
    guard.start_monitoring(
        on_focus_lost=lambda reason: events.append(("lost", reason)),
        on_focus_gained=lambda info: events.append(("gained", info)),
        check_interval_sec=0.02,
        allow_gatherion_ui=True,
    )
    try:
        # The monitor loop starts with last_focused=True, so no
        # baseline-establishment sleep is needed: the first check after
        # the switch fires "lost". Poll instead of sleeping a fixed
        # 0.06 s (3 intervals) and hoping the thread got scheduled.
        guard.set_simulated_foreground_window("Discord", 7777)
        assert _wait_until(lambda: len(events) >= 1), f"focus-lost event never arrived: {events}"
        assert events[0][0] == "lost"

        # Simulate Alt-Tab back to Aion
        guard.set_simulated_foreground_window("AION 2", 9096)
        assert _wait_until(lambda: len(events) >= 2), f"focus-gained event never arrived: {events}"
        assert events[1][0] == "gained"
    finally:
        # Previously a mid-test assert failure leaked the daemon monitor
        # thread; stop it even on failure.
        guard.stop_monitoring()


def test_worker_pause_on_alt_tab():
    guard = ForegroundFocusGuard()
    guard.set_simulated_foreground_window("AION 2", 9096)

    worker = _shared_worker()
    worker.focus_monitor = guard
    worker.pause_on_alt_tab = True

    # Start worker in route mode (simulate resumed)
    worker.resume()
    assert not worker.is_paused

    # Simulate focus check while game is focused
    assert worker._check_game_focus() is True
    assert not worker.is_paused

    # Simulate Alt-Tab to Chrome
    guard.set_simulated_foreground_window("Google Chrome", 8888)
    assert worker._check_game_focus() is False

    # Worker must now be paused!
    assert worker.is_paused is True
    assert "Alt-Tab" in worker.pause_reason
    assert worker._motion_should_stop() is True


def test_worker_pause_on_alt_tab_disabled():
    guard = ForegroundFocusGuard()
    guard.set_simulated_foreground_window("Google Chrome", 8888)

    worker = _shared_worker()
    worker.focus_monitor = guard
    worker.pause_on_alt_tab = False  # Disabled by user

    worker.resume()
    assert not worker.is_paused

    # Focus check should return True because pause_on_alt_tab is disabled
    assert worker._check_game_focus() is True
    assert not worker.is_paused


def test_input_driver_focus_stealing_protection():
    driver = InputController(test_mode=True)
    # When motion_aborted is set, focus_game_window returns False
    driver.abort_motion()
    assert driver.focus_game_window() is False


def _assert_real_foreground_query(guard):
    """Assertions for the real (non-simulated) Win32 foreground query path.

    On Linux/macOS there is no Win32: the real path must report "no active
    foreground window" instead of pretending the game is focused. On
    Windows the real foreground window is genuinely unknown here, so only
    the (bool, str) shape and no-exception behaviour is asserted.
    """
    hwnd, pid, title = guard.query_foreground_window_info()
    assert isinstance(hwnd, int) and isinstance(pid, int) and isinstance(title, str)
    focused, reason = guard.is_game_window_focused()
    assert isinstance(focused, bool) and isinstance(reason, str)
    if not guard._has_win32:
        assert (hwnd, pid, title) == (0, 0, "")
        assert focused is False
        assert "No active foreground window" in reason
    return focused, reason


def test_real_foreground_query_without_simulation():
    # Fresh guard: no simulated window set, so is_game_window_focused()
    # must go through the real Win32 query path (GetForegroundWindow on
    # Windows, the no-Win32 fallback elsewhere) — not the simulation.
    guard = ForegroundFocusGuard()
    assert guard._simulated_foreground_title is None
    assert guard._has_win32 == (os.name == "nt")
    _assert_real_foreground_query(guard)


def test_clearing_simulation_returns_to_real_query():
    guard = ForegroundFocusGuard()
    guard.set_simulated_foreground_window("AION 2", 9096)
    focused, _reason = guard.is_game_window_focused()
    assert focused is True

    # Clearing the simulation must return to the real query path.
    guard.set_simulated_foreground_window(None)
    assert guard._simulated_foreground_title is None
    _assert_real_foreground_query(guard)
