# -*- coding: utf-8 -*-
"""Stall pause resumes on heartbeat. Ctrl+Alt+F10 pause does not."""

import time
import unittest

from core.watchdog import BotWatchdog


class _GatherRoute:
    """Minimal route worker: pause stops the route, resume walks again."""

    def __init__(self) -> None:
        self._paused = False
        self._pause_reason = ""
        self.state = "WALKING"
        self.mode = "route"
        self.route_running = True
        self.resume_calls = 0

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self, reason: str = "Paused by user") -> None:
        self._paused = True
        self._pause_reason = reason or "Paused by user"
        self.state = "PAUSED"
        self.route_running = False

    def resume(self) -> None:
        self.resume_calls += 1
        self._paused = False
        self._pause_reason = ""
        self.state = "WALKING"
        self.mode = "route"
        self.route_running = True


class WatchdogStallResumeTests(unittest.TestCase):
    def _watchdog(self, worker: _GatherRoute, callback=None) -> BotWatchdog:
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

    def test_stall_pause_resumes_when_heartbeat_returns(self) -> None:
        worker = _GatherRoute()
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
        self.assertFalse(worker.route_running)
        self.assertEqual(worker._pause_reason, BotWatchdog.STALL_PAUSE_REASON)
        self.assertTrue(seen)

        watchdog.record_heartbeat(state=worker.state, mode=worker.mode)

        self.assertFalse(worker.is_paused)
        self.assertTrue(worker.route_running)
        self.assertEqual(worker.state, "WALKING")
        self.assertEqual(worker._pause_reason, "")
        self.assertEqual(worker.resume_calls, 1)

    def test_user_chord_pause_stays_paused_after_heartbeat(self) -> None:
        reasons = ("Ctrl+Alt+F10", "Durduruldu (Ctrl+Alt+F10)")
        for reason in reasons:
            with self.subTest(reason=reason):
                worker = _GatherRoute()
                watchdog = self._watchdog(worker)
                worker.pause(reason)
                watchdog.record_heartbeat(state="PAUSED", mode="route")
                self._expire_heartbeat(watchdog)

                watchdog.check_health()
                watchdog.record_heartbeat(state=worker.state, mode=worker.mode)

                self.assertTrue(worker.is_paused)
                self.assertFalse(worker.route_running)
                self.assertEqual(worker._pause_reason, reason)
                self.assertEqual(worker.resume_calls, 0)
                self.assertNotEqual(worker.state, "WALKING")


if __name__ == "__main__":
    unittest.main()
