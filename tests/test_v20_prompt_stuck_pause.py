# -*- coding: utf-8 -*-
"""v20 PROMPT_STUCK_10S — duraklatmada yanlış pozitif (canlı 19:26:11)."""

from __future__ import annotations

import time

from core.opsec_crypto import PanicAlertSystem


def test_prompt_stuck_does_not_fire_while_paused():
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    panic.set_interact_prompt_visible(True)
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 20.0
    panic.set_paused(True)
    event = panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
    assert event is None
    assert panic.panic_triggered is False


def test_prompt_stuck_fires_after_resume_when_still_visible():
    panic = PanicAlertSystem(prompt_stuck_sec=10.0, enable_sound=False)
    panic.set_interact_prompt_visible(True)
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 20.0
    panic.set_paused(True)
    assert panic.check_watchdog_conditions(play_siren=False) is None
    panic.set_paused(False)
    panic.set_interact_prompt_visible(True)
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 12.0
    event = panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
    assert event is not None
    assert event["reason"] == "PROMPT_STUCK_10S"
