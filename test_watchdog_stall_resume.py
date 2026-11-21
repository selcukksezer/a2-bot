# -*- coding: utf-8 -*-
"""Stall pause stays paused on heartbeat. Ctrl+Alt+F10 pause does not resume.

High #2: watchdog must NEVER auto-resume after a stall/panic pause. Only an
explicit user Başlat / Devam / F10-start may resume. Heartbeat return clears
the stall flag but leaves the worker paused.

NOTE (D89/F-1554): an earlier draft tested the watchdog against a hand-made
`_GatherRoute` fake whose pause/resume/state surface was invented for the
test (including a `route_running` attribute the real worker does not even
have). These tests now drive the real production worker,
`core.state_machine.BotWorker`, constructed with inert stub dependencies
(stub input driver + empty config/route/vision objects). Only the input
driver is stubbed: BotWorker's pause/resume path touches the driver
(abort/resume motion, movement_active, route-active sync), while the rest
of the worker runs its real state machine. The watchdog therefore exercises
the real pause reason, real state transitions, and real resume gating.
"""

import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from core.state_machine import BotWorker
from core.watchdog import BotWatchdog


def _make_real_worker(movement_active: bool = True) -> BotWorker:
    """Build the real BotWorker with a stub input driver.

    movement_active=True so resume() lands the worker in WALKING (the state
    a route worker is in while walking); the stub driver never touches
    hardware, it only answers the queries pause()/resume() make.
    """
    driver = SimpleNamespace(
        movement_active=lambda: movement_active,
        abort_motion=lambda: None,
        release_all_movement_keys=lambda: None,
        resume_motion=lambda: None,
        set_route_active=lambda _active: None,
        set_test_mode=lambda _enabled: None,
    )
    worker = BotWorker(
        config={},
        vision=SimpleNamespace(),
        input_driver=driver,
        route_manager=SimpleNamespace(),
        combat_recovery=SimpleNamespace(),
        navigator=SimpleNamespace(),
        recorder=SimpleNamespace(),
    )
    return worker


_SHARED_WORKER: BotWorker | None = None


def _shared_worker() -> BotWorker:
    """One real BotWorker for the whole module.

    BotWorker() costs ~2 s after warm-up (~8-14 s cold: lazy ONNX model
    load in _init_optional_subsystem) and both tests only need a live
    worker they resume/pause — each test calls resume() first, which
    resets pause state, and tearDown restores the spied resume(). Sharing
    saves one construction per run; the worker stays the real production
    worker (no __new__ bypass — see the module docstring, D89/F-1554).
    """
    global _SHARED_WORKER
    if _SHARED_WORKER is None:
        _SHARED_WORKER = _make_real_worker()
    return _SHARED_WORKER


class WatchdogStallResumeTests(unittest.TestCase):
    def setUp(self):
        self._resume_spy_original = None

    def tearDown(self):
        # _count_resumes() wraps worker.resume; unwrap so the next test
        # (which shares the worker) starts from the real method.
        worker = _SHARED_WORKER
        if worker is not None and self._resume_spy_original is not None:
            worker.resume = self._resume_spy_original
            self._resume_spy_original = None

    def _watchdog(self, worker: BotWorker, callback=None) -> BotWatchdog:
        return BotWatchdog(
            stall_timeout_sec=15.0,
            check_interval_sec=30.0,
            recovery_callback=callback,
            worker=worker,
        )

    def _expire_heartbeat(self, watchdog: BotWatchdog) -> None:
        with watchdog._lock:
            watchdog._last_heartbeat_ts = time.time() - (watchdog.stall_timeout_sec + 1.0)
            watchdog._is_paused = False
            watchdog._is_frozen = False
            watchdog._frozen_since_ts = None
            watchdog._last_state = "WALKING"

    def _count_resumes(self, worker: BotWorker) -> Any:
        """Spy on resume() calls (replaces the fake's resume_calls counter)."""
        calls: list = []
        original_resume = worker.resume
        if self._resume_spy_original is None:
            # First wrap in this test keeps the true original; a second
            # wrap (test 2 iterates two reasons) wraps the spy, but
            # tearDown must still restore the true original.
            self._resume_spy_original = original_resume

        def _counting_resume() -> None:
            calls.append(1)
            original_resume()

        worker.resume = _counting_resume  # type: ignore[method-assign]
        return calls

    def test_stall_pause_stays_paused_when_heartbeat_returns(self) -> None:
        """High #2: stall pause must NOT auto-resume on heartbeat."""
        worker = _shared_worker()
        worker.resume()
        self.assertEqual(worker.state, "WALKING")
        resume_calls = self._count_resumes(worker)
        seen = []

        def _recovery(payload, route=worker, log=seen) -> None:
            log.append(payload)
            if not route.is_paused:
                route.pause(f"Watchdog güvenli duraklatma: {payload}")

        watchdog = self._watchdog(worker, callback=_recovery)
        watchdog.record_heartbeat(state="WALKING", mode="route")
        self._expire_heartbeat(watchdog)

        watchdog.check_health()

        self.assertTrue(worker.is_paused)
        self.assertEqual(worker.state, "PAUSED")
        self.assertEqual(worker._pause_reason, BotWatchdog.STALL_PAUSE_REASON)
        self.assertTrue(seen)

        watchdog.record_heartbeat(state=worker.state, mode=worker.mode)

        self.assertTrue(worker.is_paused)
        self.assertEqual(worker.state, "PAUSED")
        self.assertEqual(worker._pause_reason, BotWatchdog.STALL_PAUSE_REASON)
        self.assertEqual(len(resume_calls), 0)

    def test_user_chord_pause_stays_paused_after_heartbeat(self) -> None:
        reasons = ("Ctrl+Alt+F10", "Durduruldu (Ctrl+Alt+F10)")
        for reason in reasons:
            with self.subTest(reason=reason):
                worker = _shared_worker()
                worker.resume()
                self.assertEqual(worker.state, "WALKING")
                resume_calls = self._count_resumes(worker)
                watchdog = self._watchdog(worker)
                worker.pause(reason)
                watchdog.record_heartbeat(state="PAUSED", mode="route")
                self._expire_heartbeat(watchdog)

                watchdog.check_health()
                watchdog.record_heartbeat(state=worker.state, mode=worker.mode)

                self.assertTrue(worker.is_paused)
                self.assertEqual(worker._pause_reason, reason)
                self.assertEqual(len(resume_calls), 0)
                self.assertNotEqual(worker.state, "WALKING")


class SetModePreservesPauseTests(unittest.TestCase):
    """High #1/#4/#5/#6: set_mode must not unpause / resume."""

    def test_set_mode_auto_record_keeps_paused(self) -> None:
        worker = _shared_worker()
        worker.pause("Durduruldu (Ctrl+Alt+F10)")
        self.assertTrue(worker.is_paused)
        resume_calls = []
        original = worker.resume

        def _spy() -> None:
            resume_calls.append(1)
            original()

        worker.resume = _spy  # type: ignore[method-assign]
        try:
            worker.set_mode("auto_record")
            self.assertTrue(worker.is_paused)
            self.assertEqual(worker.mode, "auto_record")
            self.assertEqual(worker.state, "PAUSED")
            self.assertEqual(resume_calls, [])
        finally:
            worker.resume = original  # type: ignore[method-assign]


if __name__ == "__main__":
    unittest.main()
