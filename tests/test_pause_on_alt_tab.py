# -*- coding: utf-8 -*-
"""
Unit tests for automatic bot pause on Alt-Tab / game focus loss.
Verifies ForegroundFocusGuard, BotWorker integration, and InputController focus protection.
"""

import time
import pytest
from core.dpi_focus_guard import ForegroundFocusGuard
from core.state_machine import BotMode, BotState, BotWorker
from core.input_driver import InputController
from config.schema import SettingsManager, BotSettings


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
    time.sleep(0.04)

    # Simulate Alt-Tab to Discord
    guard.set_simulated_foreground_window("Discord", 7777)
    time.sleep(0.06)

    assert len(events) >= 1
    assert events[0][0] == "lost"

    # Simulate Alt-Tab back to Aion
    guard.set_simulated_foreground_window("AION 2", 9096)
    time.sleep(0.06)

    assert len(events) >= 2
    assert events[1][0] == "gained"

    guard.stop_monitoring()


def test_worker_pause_on_alt_tab():
    guard = ForegroundFocusGuard()
    guard.set_simulated_foreground_window("AION 2", 9096)

    worker = BotWorker()
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

    worker = BotWorker()
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
