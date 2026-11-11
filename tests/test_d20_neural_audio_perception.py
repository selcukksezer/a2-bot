# -*- coding: utf-8 -*-
"""D20 (neural_audio_perception) improvements — regression / lock tests.

Covers:
  1. Audio spectral caches (Hann window + rfftfreq/band masks): identical
     analysis results, cache reuse, bounded size, no cross-talk between
     alternating buffer lengths.
  2. Visual fallback masks converted to cv2.inRange: bit-exact equivalence
     with the old boolean chains, and identical detect_3d_objects outputs
     on golden synthetic frames (golden values recorded pre-change).
  3. _tensor_bbox_to_abs: the single helper replacing 5 inline conversions.

NOTE: synthesize_failure_shatter() is deliberately NOT asserted here:
its output currently classifies as "idle" (global spectral flatness
~1e-11 fails the >= 0.015 gate although the in-band evidence passes).
That is a real bug, reported in dev-results/D20.json as found_bugs —
not fixed by this slice, so it must not be locked in by a test.
"""
import sys
import os

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.neural_audio_perception import (  # noqa: E402
    AudioLoopbackFFTAnalyzer,
    ONNXYolo3DDetector,
)


# ----------------------------------------------------------------------
# helpers: golden synthetic frames (same builders as the pre-change run)
# ----------------------------------------------------------------------
def _smooth_blobs_frame():
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    f[:, :] = (60, 90, 40)
    cv2.rectangle(f, (300, 200), (500, 400), (180, 40, 220), -1)
    cv2.rectangle(f, (700, 300), (850, 420), (40, 40, 200), -1)
    cv2.circle(f, (640, 360), 80, (30, 30, 30), -1)
    return f


def _aria_field_frame():
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    f[:, :] = (40, 120, 40)
    cv2.rectangle(f, (400, 250), (470, 330), (200, 60, 180), -1)
    cv2.rectangle(f, (405, 255), (465, 325), (220, 80, 200), -1)
    return f


def _cyan_crest_frame():
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    f[:, :] = (50, 80, 50)
    cv2.rectangle(f, (500, 200), (620, 214), (220, 200, 0), -1)
    cv2.rectangle(f, (505, 186), (615, 200), (240, 240, 240), -1)
    cv2.rectangle(f, (500, 216), (620, 280), (200, 60, 180), -1)
    return f


def _strip(dets):
    return [
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in d.items() if k != "inference_ms"}
        for d in dets
    ]


# ----------------------------------------------------------------------
# 1. Audio spectral caches
# ----------------------------------------------------------------------
class TestAudioSpectralCache:
    def test_chime_and_footstep_classification_unchanged(self):
        a = AudioLoopbackFFTAnalyzer()
        chime = a.analyze_pcm_buffer(
            AudioLoopbackFFTAnalyzer.synthesize_crystal_chime()
        )
        assert chime["dominant_event"] == "gather_success_chime"
        assert chime["confidence"] == 0.99
        foot = a.analyze_pcm_buffer(
            AudioLoopbackFFTAnalyzer.synthesize_monster_footstep()
        )
        assert foot["dominant_event"] == "danger_audio_alert"
        assert foot["confidence"] == 0.98

    def test_silence_short_int16_stereo_unchanged(self):
        a = AudioLoopbackFFTAnalyzer()
        assert a.analyze_pcm_buffer(np.zeros(2048, dtype=np.float32))["dominant_event"] == "idle"
        assert a.analyze_pcm_buffer(np.zeros(10, dtype=np.float32))["dominant_event"] == "idle"
        ch = AudioLoopbackFFTAnalyzer.synthesize_crystal_chime()
        assert (
            a.analyze_pcm_buffer((ch * 32767).astype(np.int16))["dominant_event"]
            == "gather_success_chime"
        )
        assert (
            a.analyze_pcm_buffer(np.stack([ch, ch], axis=1))["dominant_event"]
            == "gather_success_chime"
        )

    def test_alternating_lengths_do_not_mix_windows_or_bands(self):
        a = AudioLoopbackFFTAnalyzer()
        chime = AudioLoopbackFFTAnalyzer.synthesize_crystal_chime()
        noise = np.random.default_rng(0).standard_normal(2048).astype(np.float32) * 0.3
        first_chime = a.analyze_pcm_buffer(chime)
        first_noise = a.analyze_pcm_buffer(noise)
        # Alternate several times; every repeat must reproduce the first
        # result. spectral_flux is exempt: it is stateful by design (0.25
        # on the first frame after a shape change, 0.0 once the previous
        # spectrum has the same length), not a cache value.
        for _ in range(3):
            rep_noise = a.analyze_pcm_buffer(noise)
            rep_chime = a.analyze_pcm_buffer(chime)
            for rep, first in ((rep_noise, first_noise), (rep_chime, first_chime)):
                assert rep["spectral_flux"] in (0.0, 0.25)
                assert {k: v for k, v in rep.items() if k != "spectral_flux"} == {
                    k: v for k, v in first.items() if k != "spectral_flux"
                }

    def test_cache_reuse_and_bounded_size(self):
        a = AudioLoopbackFFTAnalyzer()
        assert a._hann_window_for(2048) is a._hann_window_for(2048)
        assert a._hann_window_for(2048) is a._hann_window
        bands1 = a._spectral_bands_for(6615, 44100)
        bands2 = a._spectral_bands_for(6615, 44100)
        assert bands1 is bands2
        assert bands1[0].shape == (6615 // 2 + 1,)
        # 12 distinct lengths must not grow either cache past the cap of 8.
        for n in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]:
            a.analyze_pcm_buffer(
                np.random.default_rng(n).standard_normal(n).astype(np.float32)
            )
        assert len(a._hann_cache) <= 8
        assert len(a._spectral_cache) <= 8

    def test_spectral_bands_match_plain_rfftfreq(self):
        a = AudioLoopbackFFTAnalyzer()
        freqs, m_danger, m_fail, m_chime = a._spectral_bands_for(2048, 44100)
        expect = np.fft.rfftfreq(2048, d=1.0 / 44100)
        assert np.array_equal(freqs, expect)
        assert np.array_equal(m_danger, (expect >= 80.0) & (expect <= 320.0))
        assert np.array_equal(m_fail, (expect >= 180.0) & (expect <= 650.0))
        assert np.array_equal(m_chime, (expect >= 2200.0) & (expect <= 4800.0))


# ----------------------------------------------------------------------
# 2. inRange mask equivalence (old boolean formula vs new inRange)
# ----------------------------------------------------------------------
class TestInRangeMaskEquivalence:
    @pytest.fixture()
    def hsv_grad(self):
        rng = np.random.default_rng(1)
        hsv = (rng.random((180, 320, 3)) * np.array([180, 255, 255])).astype(np.uint8)
        grad = np.clip(rng.random((180, 320)) * 80, 0, 255).astype(np.uint8)
        return hsv, grad

    def test_converted_masks_bit_exact(self, hsv_grad):
        hsv, grad = hsv_grad
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        cases = [
            # (old boolean mask, new inRange mask)
            (
                ((h >= 60) & (h <= 102) & (s >= 55) & (v >= 115)).astype(np.uint8) * 255,
                cv2.inRange(hsv, (60, 55, 115), (102, 255, 255)),
            ),
            (
                ((h >= 126) & (h <= 164) & (s >= 75) & (v >= 80) & (grad >= 18)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (126, 75, 80), (164, 255, 255)), cv2.inRange(grad, 18, 255)),
            ),
            (
                ((h >= 36) & (h <= 76) & (s >= 65) & (v >= 45) & (v <= 225)).astype(np.uint8) * 255,
                cv2.inRange(hsv, (36, 65, 45), (76, 255, 225)),
            ),
            (
                ((h >= 20) & (h <= 34) & (s >= 115) & (v >= 130) & (grad >= 18)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (20, 115, 130), (34, 255, 255)), cv2.inRange(grad, 18, 255)),
            ),
            (
                ((h >= 104) & (h <= 132) & (s >= 145) & (v >= 140) & (grad >= 35)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (104, 145, 140), (132, 255, 255)), cv2.inRange(grad, 35, 255)),
            ),
            (
                ((s <= 58) & (v >= 20) & (v <= 78) & (grad >= 30)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 20), (180, 58, 78)), cv2.inRange(grad, 30, 255)),
            ),
            (
                ((h >= 10) & (h <= 23) & (s >= 145) & (v >= 145) & (grad >= 42)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (10, 145, 145), (23, 255, 255)), cv2.inRange(grad, 42, 255)),
            ),
            (
                ((s <= 45) & (v >= 210) & (grad >= 45)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 210), (180, 45, 255)), cv2.inRange(grad, 45, 255)),
            ),
            (
                ((s <= 40) & (v >= 215) & (grad >= 45)).astype(np.uint8) * 255,
                cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 215), (180, 40, 255)), cv2.inRange(grad, 45, 255)),
            ),
            (
                ((h >= 15) & (h <= 34) & (s >= 115) & (v >= 135)).astype(np.uint8) * 255,
                cv2.inRange(hsv, (15, 115, 135), (34, 255, 255)),
            ),
        ]
        for i, (old, new) in enumerate(cases):
            assert np.array_equal(old, new), f"mask case {i} differs"

    def test_wrap_around_red_masks_kept_boolean(self, hsv_grad):
        # Red wraps hue 0, so inRange needs two calls and measured slower
        # (0.79 ms vs 0.60 ms); the module keeps the boolean chain there.
        # This test pins that the boolean form itself is still what the
        # module's behaviour relies on (golden outputs below cover it).
        hsv, _ = hsv_grad
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        boolean = ((((h <= 9) | (h >= 169)) & (s >= 115) & (v >= 95))).astype(np.uint8) * 255
        two_range = cv2.bitwise_or(
            cv2.inRange(hsv, (0, 115, 95), (9, 255, 255)),
            cv2.inRange(hsv, (169, 115, 95), (180, 255, 255)),
        )
        assert np.array_equal(boolean, two_range)


# ----------------------------------------------------------------------
# 3. Golden visual outputs (recorded pre-change, inference_ms excluded)
# ----------------------------------------------------------------------
class TestVisualGoldenOutputs:
    def test_smooth_blobs(self):
        d = ONNXYolo3DDetector()
        got = _strip(d.detect_3d_objects(_smooth_blobs_frame()))
        assert got == [
            {"class_id": 1, "class_name": "herb_mela", "fallback_label": "herb_mela",
             "category_tr": "Mela (Kırmızı Meyveli Çalı)",
             "confidence": 0.93, "bbox": [695, 297, 164, 130], "center": [777, 362],
             "backend": "fast_3d_mesh_tensor_fallback", "sub_type": "Mela"},
            {"class_id": 4, "class_name": "gem_crystal", "fallback_label": "gem_crystal",
             "category_tr": "Gem/Kristal (Safir/Yakut/Elmas)",
             "confidence": 0.9, "bbox": [701, 302, 152, 120], "center": [777, 362],
             "backend": "fast_3d_mesh_tensor_fallback", "sub_type": "Ruby Gemstone"},
        ]

    def test_aria_field(self):
        d = ONNXYolo3DDetector()
        got = _strip(d.detect_3d_objects(_aria_field_frame()))
        assert got == [
            {"class_id": 0, "class_name": "herb_aria", "fallback_label": "herb_aria",
             "category_tr": "Aria (Mor Kristal Bitki)",
             "confidence": 0.92, "bbox": [400, 252, 73, 82], "center": [436, 293],
             "backend": "fast_3d_mesh_tensor_fallback", "sub_type": "Aria"},
        ]

    def test_cyan_crest(self):
        d = ONNXYolo3DDetector()
        got = _strip(d.detect_3d_objects(_cyan_crest_frame()))
        assert len(got) == 2
        assert got[0]["class_name"] == "herb_aria" and got[0]["sub_type"] == "Aria"
        assert got[0]["confidence"] == 0.96
        assert got[1]["class_name"] == "gem_crystal" and got[1]["sub_type"] == "Diamond Gemstone"

    def test_noise_frame_detection_count(self):
        d = ONNXYolo3DDetector()
        noise = (np.random.default_rng(0).random((720, 1280, 3)) * 255).astype(np.uint8)
        assert len(d.detect_3d_objects(noise)) == 5

    def test_blank_slate_tiny_frames_empty(self):
        d = ONNXYolo3DDetector()
        assert d.detect_3d_objects(np.zeros((720, 1280, 3), dtype=np.uint8)) == []
        slate = np.zeros((720, 1280, 3), dtype=np.uint8)
        slate[:, :] = (30, 30, 30)
        assert d.detect_3d_objects(slate) == []
        assert d.detect_3d_objects(np.zeros((50, 80, 3), dtype=np.uint8)) == []
        assert d.detect_3d_objects(None) == []


# ----------------------------------------------------------------------
# 4. bbox helper
# ----------------------------------------------------------------------
class TestTensorBboxHelper:
    def test_basic_mapping(self):
        got = ONNXYolo3DDetector._tensor_bbox_to_abs(102, 86, 2.4, 2.4, 10, 20, 30, 40, 24, 24)
        assert got == (126, 134, 72, 96)

    def test_minimum_size_clamp(self):
        got = ONNXYolo3DDetector._tensor_bbox_to_abs(0, 0, 2.4, 2.4, 0, 0, 5, 5, 28, 26)
        assert got == (0, 0, 28, 26)

    def test_no_inline_copies_remain(self):
        src = open(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "core", "neural_audio_perception.py",
            )
        ).read()
        assert src.count("abs_x = int(round(vx1 + x_t * sx))") == 0
        assert src.count("self._tensor_bbox_to_abs(") >= 5
