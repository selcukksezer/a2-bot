# -*- coding: utf-8 -*-
"""D24 PerceptionHub improvements: lock-free perception, vitals dedup.

New files only — existing test_perception_hub.py is NOT modified.
"""

import threading
import time
import unittest
from unittest import mock

import numpy as np

from core.perception_hub import FrameCognition, PerceptionHub


class _SlowYolo:
    def detect(self, frame):
        time.sleep(0.25)  # simulate YOLO inference
        return [{"label": "mob", "conf": 0.9}]


class _SlowVision:
    def detect_gather_prompt(self, frame=None):
        time.sleep(0.05)
        return (False, 0, None)


class _LegacyHUD:
    """Only the legacy read_hp_mp API — no read_player_hp_mp_bars."""

    def read_hp_mp(self, frame):
        return {"hp_pct": 60.0, "mp_pct": 25.0}


class PerceptionHubD24Tests(unittest.TestCase):
    def test_last_not_blocked_by_slow_observe(self):
        # D24: observe() no longer holds the lock during YOLO/vision work,
        # so last() from another thread must return promptly (<100 ms)
        # instead of blocking ~300 ms behind the slow perception.
        hub = PerceptionHub(yolo_detector=_SlowYolo(), vision=_SlowVision(),
                            min_interval_sec=0.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        hub.observe(frame, force=True)  # warm-up publish
        done = threading.Event()

        def observer():
            hub.observe(frame, force=True)
            done.set()

        t = threading.Thread(target=observer)
        t.start()
        time.sleep(0.05)  # let observe() enter slow perception
        t0 = time.perf_counter()
        snap = hub.last()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        t.join()
        self.assertIsNotNone(snap)
        self.assertLess(latency_ms, 100.0, f"last() blocked {latency_ms:.1f} ms")

    def test_stale_observe_does_not_overwrite_newer_snapshot(self):
        # D24: a slow observe() that started earlier (ts=999) must not
        # clobber a newer published snapshot (ts=1000).
        hub = PerceptionHub(min_interval_sec=0.0)
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        with mock.patch("core.perception_hub.time.time", return_value=1000.0):
            first = hub.observe(frame, force=True)
        with mock.patch("core.perception_hub.time.time", return_value=999.0):
            second = hub.observe(frame, force=True)
        self.assertEqual(second.ts, 999.0)  # it computed, but...
        self.assertIs(hub.last(), first)  # ...did not publish over newer

    def test_legacy_read_hp_mp_path(self):
        # D24: _apply_hp_mp dedup must keep the legacy read_hp_mp branch
        # working (hp/mp fused, no target info from this API).
        hub = PerceptionHub(hud_reader=_LegacyHUD(), min_interval_sec=0.0)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cog = hub.observe(frame, force=True)
        self.assertAlmostEqual(cog.player_hp, 0.6, places=2)
        self.assertAlmostEqual(cog.player_mp, 0.25, places=2)
        self.assertFalse(cog.has_target)
        self.assertAlmostEqual(cog.monster_hp, 0.0, places=2)

    def test_apply_hp_mp_zero_is_valid_and_missing_is_unknown(self):
        cog = FrameCognition()
        filled = PerceptionHub._apply_hp_mp({"hp_pct": 0.0, "mp_pct": 0.0}, cog)
        self.assertTrue(filled)
        self.assertEqual(cog.player_hp, 0.0)  # 0% is dead/empty, not "full"
        self.assertEqual(cog.player_mp, 0.0)
        cog2 = FrameCognition()
        filled2 = PerceptionHub._apply_hp_mp({}, cog2)
        self.assertFalse(filled2)
        self.assertEqual(cog2.player_hp, 0.0)  # unknown, untouched


if __name__ == "__main__":
    unittest.main()
