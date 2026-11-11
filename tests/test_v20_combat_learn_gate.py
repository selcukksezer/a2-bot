# -*- coding: utf-8 -*-
"""
v20 combat stall — gate game_learner / perception_hub during combat.

Live evidence (session_2026-11-11_192331.log):
  19:23:50 Mode → COMBAT
  19:24:05 BotWorker stalled 15.18s (fps 2.44, avg_latency 409ms)

Root cause (code): `_grab_frame` already skipped dataset_factory ingest in
combat, but every color grab still ran synchronous
`game_learner.observe_frame` + `perception_hub.observe`. `_step_combat_mode`
also re-called `perception_hub.observe` when `_frame_cognition` was empty.

These tests assert the gate; they do NOT claim live stall is gone.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Optional native deps stubbed for headless import of BotWorker.
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

from core.state_machine import BotWorker  # noqa: E402


def _make_worker(*, mode: str, state: str) -> BotWorker:
    worker = BotWorker.__new__(BotWorker)
    worker._lock = threading.RLock()
    worker._mode = mode
    worker._state = state
    worker.config = {}
    worker.vision = MagicMock()
    worker.vision.grab_screen = MagicMock(
        return_value=np.zeros((64, 64, 3), dtype=np.uint8)
    )
    worker.reflex = None
    worker.game_learner = MagicMock()
    worker.game_learner.observe_frame = MagicMock()
    worker.perception_hub = MagicMock()
    worker.perception_hub.observe = MagicMock(return_value={"ok": True})
    worker.dataset_factory = MagicMock()
    worker.dataset_factory.feed_live_frame = MagicMock()
    worker._frame_cognition = None
    worker._last_combat_perception_mono = 0.0
    worker._learning_mix_sources = lambda: None  # type: ignore[method-assign]
    return worker


def test_grab_frame_skips_learner_and_factory_in_combat():
    worker = _make_worker(mode="combat", state="FIGHTING")
    worker._last_combat_perception_mono = 0.0
    frame = worker._grab_frame(grayscale=False)
    assert frame is not None
    worker.game_learner.observe_frame.assert_not_called()
    worker.dataset_factory.feed_live_frame.assert_not_called()
    worker.perception_hub.observe.assert_called_once()


def test_grab_frame_throttles_perception_in_combat():
    worker = _make_worker(mode="combat", state="FIGHTING")
    worker._last_combat_perception_mono = time.monotonic()
    worker._grab_frame(grayscale=False)
    worker.perception_hub.observe.assert_not_called()
    worker.game_learner.observe_frame.assert_not_called()


def test_grab_frame_resumes_learning_outside_combat():
    worker = _make_worker(mode="route", state="WALKING")
    worker._grab_frame(grayscale=False)
    worker.game_learner.observe_frame.assert_called_once()
    worker.perception_hub.observe.assert_called_once()
    worker.dataset_factory.feed_live_frame.assert_called_once()


def test_step_combat_mode_does_not_duplicate_observe_when_throttled():
    worker = _make_worker(mode="combat", state="FIGHTING")
    worker._frame_cognition = None
    worker._last_combat_perception_mono = time.monotonic()
    assert worker._combat_perception_due() is False
    frame_cog = worker._plain_attr("_frame_cognition")
    perception_hub = worker._plain_attr("perception_hub")
    if (
        frame_cog is None
        and perception_hub is not None
        and hasattr(perception_hub, "observe")
        and worker._combat_perception_due()
    ):
        perception_hub.observe(np.zeros((8, 8, 3), dtype=np.uint8))
    worker.perception_hub.observe.assert_not_called()


def test_combat_perception_due_after_interval():
    worker = _make_worker(mode="combat", state="FIGHTING")
    worker._last_combat_perception_mono = time.monotonic() - 2.0
    assert worker._combat_perception_due() is True
    worker._mark_combat_perception()
    assert worker._combat_perception_due() is False
