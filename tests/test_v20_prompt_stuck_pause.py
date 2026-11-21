# -*- coding: utf-8 -*-
"""v20 PROMPT_STUCK_10S — duraklatmada yanlış pozitif + soft-pause / grace."""

from __future__ import annotations

import time

from core.opsec_crypto import PanicAlertSystem


def _fire_prompt_stuck(panic: PanicAlertSystem, *, appear_age: float = 12.0):
    """Grace kapalı + 2 confirm poll ile soft PROMPT_STUCK üretir."""
    panic.arm_prompt_stuck_grace(0.0)
    with panic._lock:
        panic._prompt_stuck_grace_until = 0.0
        panic._prompt_stuck_confirm_streak = 0
        panic._prompt_appear_ts = time.monotonic() - appear_age
    # Confirm streak: iki poll gerekir.
    assert panic.check_watchdog_conditions(play_siren=False, game_window_present=True) is None
    return panic.check_watchdog_conditions(play_siren=False, game_window_present=True)


def test_prompt_stuck_does_not_fire_while_paused():
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    panic.set_interact_prompt_visible(True)
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 20.0
    panic.set_paused(True)
    event = panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
    assert event is None
    assert panic.panic_triggered is False


def test_prompt_stuck_soft_pause_not_hard_lock():
    """F-0049: PROMPT_STUCK panik kilidi değil — soft_pause + unlock_hint."""
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    pauses = []
    panic.set_prompt_stuck_pause_callback(lambda r: pauses.append(r))
    event = _fire_prompt_stuck(panic)
    assert event is not None
    assert event["reason"] == "PROMPT_STUCK_10S"
    assert event.get("soft_pause") is True
    assert "Başlat" in str(event.get("unlock_hint") or "")
    assert panic.panic_triggered is False
    assert panic.inputs_halted is False
    assert pauses, "worker.pause soft callback çağrılmalı"


def test_prompt_stuck_grace_after_resume_blocks_immediate_retrigger():
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    panic.set_interact_prompt_visible(True)
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 20.0
    panic.set_paused(True)
    assert panic.check_watchdog_conditions(play_siren=False) is None
    panic.set_paused(False)  # grace arm
    panic.set_interact_prompt_visible(True)
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 12.0
        panic._prompt_stuck_confirm_streak = 99
    event = panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
    assert event is None
    assert panic.panic_triggered is False


def test_prompt_stuck_fires_after_grace_expires():
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    panic.set_paused(False)
    event = _fire_prompt_stuck(panic, appear_age=12.0)
    assert event is not None
    assert event["reason"] == "PROMPT_STUCK_10S"
    assert event.get("soft_pause") is True


def test_prompt_stuck_auto_recover_clears_hard_lock():
    """Eski hard PROMPT_STUCK kilidi varken istem kaybolursa unlock."""
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    with panic._lock:
        panic._panic_triggered = True
        panic._inputs_halted = True
        panic._last_panic_event = {
            "reason": "PROMPT_STUCK_10S",
            "details": "test",
            "timestamp": time.time(),
        }
    assert panic.maybe_auto_recover_prompt_stuck(False) is True
    assert panic.panic_triggered is False
