# -*- coding: utf-8 -*-
"""R07 regression tests: ready/health flags must be probed for real (S-0301, S-0302).

The flags used to be constant ``True``. These tests pin the probe wiring:
a broken pipeline/contract must flip the flags to False, and a diagnostic
probe must not inject a stale audio event into ``poll_latest_audio_events``.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from core.material_scanner import DetectedMaterial, MaterialScanner
from core.neural_audio_perception import (
    AudioLoopbackFFTAnalyzer,
    validate_neural_audio_perception_contract,
)


class MockYoloDetector:
    def detect_3d_objects(self, frame: np.ndarray) -> list:
        return []


def _candidate(**overrides):
    kw = dict(
        material_type="Orichalcum Ore",
        source="3d_world",
        screen_x=640,
        screen_y=400,
        bbox=(620, 380, 40, 40),
        rel_map_dx=2.5,
        rel_map_dy=5.0,
        estimated_distance_m=5.6,
        confidence=0.88,
        harvests_remaining=3,
    )
    kw.update(overrides)
    return DetectedMaterial(**kw)


class TestR07MaterialScannerProbes(unittest.TestCase):
    def setUp(self) -> None:
        self.scanner = MaterialScanner()
        self.scanner.set_yolo_detector(MockYoloDetector())

    def test_health_summary_all_probed_true(self) -> None:
        health = self.scanner.get_material_scanner_health_summary()
        self.assertTrue(health["material_scanner_ready"])
        self.assertTrue(health["yolo_detector_ready"])
        self.assertTrue(health["hybrid_scoring_ready"])
        self.assertTrue(health["prioritization_ready"])

    def test_health_summary_no_detector(self) -> None:
        self.scanner.set_yolo_detector(None)
        health = self.scanner.get_material_scanner_health_summary()
        self.assertFalse(health["yolo_detector_ready"])
        self.assertTrue(health["material_scanner_ready"])
        self.assertTrue(health["hybrid_scoring_ready"])
        self.assertTrue(health["prioritization_ready"])

    def test_ready_flag_and_diagnostic_follow_contract(self) -> None:
        self.assertTrue(self.scanner.get_material_scanner_ready_flag()["ready"])
        snap = self.scanner.get_material_scanner_diagnostic_snapshot()
        self.assertEqual(snap["status"], "ok")
        self.assertTrue(snap["ready"])
        with patch(
            "core.material_scanner.validate_material_scanner_contract",
            return_value=False,
        ):
            self.assertFalse(self.scanner.get_material_scanner_ready_flag()["ready"])
            snap = self.scanner.get_material_scanner_diagnostic_snapshot()
            self.assertEqual(snap["status"], "degraded")
            self.assertFalse(snap["ready"])
            health = self.scanner.get_material_scanner_health_summary()
            self.assertFalse(health["material_scanner_ready"])

    def test_hybrid_probe_fails_without_fusion(self) -> None:
        with patch.object(
            MaterialScanner, "_fuse_hybrid_detections", return_value=[]
        ):
            self.assertFalse(self.scanner._probe_hybrid_scoring())
            health = self.scanner.get_material_scanner_health_summary()
            self.assertFalse(health["hybrid_scoring_ready"])


class TestR07NeuralAudioProbes(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = AudioLoopbackFFTAnalyzer()

    def test_ready_flag_health_and_diagnostic_probed_true(self) -> None:
        self.assertTrue(self.analyzer.get_neural_audio_ready_flag()["ready"])
        health = self.analyzer.get_neural_audio_health_summary()
        self.assertTrue(health["neural_audio_ready"])
        self.assertFalse(health["loopback_stream_running"])
        self.assertIn("capture_backend", health)
        snap = self.analyzer.get_neural_audio_diagnostic_snapshot()
        self.assertEqual(snap["status"], "ok")
        self.assertTrue(snap["ready"])
        self.assertTrue(self.analyzer.get_neural_audio_version_info()["ready"])

    def test_contract_validator_is_real(self) -> None:
        self.assertTrue(validate_neural_audio_perception_contract())
        with patch.object(
            AudioLoopbackFFTAnalyzer, "_probe_audio_pipeline", return_value=False
        ):
            self.assertFalse(validate_neural_audio_perception_contract())

    def test_probe_fails_when_analyzer_cannot_detect_chime(self) -> None:
        with patch.object(
            AudioLoopbackFFTAnalyzer,
            "analyze_pcm_buffer",
            return_value={"gather_success_chime": False},
        ):
            self.assertFalse(self.analyzer._probe_audio_pipeline())
            self.assertFalse(self.analyzer.get_neural_audio_ready_flag()["ready"])
            snap = self.analyzer.get_neural_audio_diagnostic_snapshot()
            self.assertEqual(snap["status"], "degraded")

    def test_probe_does_not_mutate_last_analysis(self) -> None:
        sentinel = {"dominant_event": "sentinel", "confidence": 0.123}
        self.analyzer._last_analysis = dict(sentinel)
        self.assertTrue(self.analyzer._probe_audio_pipeline())
        self.assertEqual(self.analyzer._last_analysis, sentinel)


if __name__ == "__main__":
    unittest.main()
