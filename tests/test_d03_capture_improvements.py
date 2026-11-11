# -*- coding: utf-8 -*-
"""D03 capture improvements: sampled validation stats, direct-gray CLAHE,
sampled-std template guard (core/capture.py, core/stealth_capture.py).

Timing bounds are deliberately generous (this VM is slow and shared); the
before/after numbers in dev-results/D03.json are the measurement evidence.
"""
import time

import numpy as np
import pytest

from core.capture import ScreenCapture, _frame_sample_stride
from core.stealth_capture import (
    DXGIDesktopCapture,
    _sampled_std,
    _get_clahe,
)


def _capture_no_init():
    cap = ScreenCapture.__new__(ScreenCapture)
    cap.screen_w, cap.screen_h = 2560, 1440
    cap._last_valid = True
    cap._last_validation_msg = "OK"
    return cap


def _dxgi_no_init():
    cap = DXGIDesktopCapture.__new__(DXGIDesktopCapture)
    cap.min_threshold, cap.max_threshold = 0.75, 0.85
    return cap


# --- I1: validate_frame sampled statistics -------------------------------

class TestValidateFrameSampled:
    def test_black_1440p_is_invalid(self):
        cap = _capture_no_init()
        valid, msg = cap.validate_frame(np.zeros((1440, 2560, 3), dtype=np.uint8))
        assert valid is False
        assert "black" in msg.lower()

    def test_uniform_dark_value_2_is_invalid(self):
        cap = _capture_no_init()
        valid, _ = cap.validate_frame(np.full((1440, 2560, 3), 2, dtype=np.uint8))
        assert valid is False

    def test_dark_scene_with_bright_content_is_valid(self):
        # mean < 3 but max >> 16: dark scene, must stay valid (F-0228).
        cap = _capture_no_init()
        frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        frame[100:120, 100:200] = 200  # small bright UI text block
        valid, msg = cap.validate_frame(frame)
        assert valid is True, msg

    def test_bright_frame_is_valid(self):
        cap = _capture_no_init()
        rng = np.random.default_rng(0)
        valid, _ = cap.validate_frame(rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8))
        assert valid is True

    def test_empty_frame_is_invalid(self):
        cap = _capture_no_init()
        valid, _ = cap.validate_frame(np.zeros((0, 0, 3), dtype=np.uint8))
        assert valid is False

    def test_small_frames_use_exact_stride_one(self):
        assert _frame_sample_stride(100, 100, 3) == 1
        assert _frame_sample_stride(1440, 2560, 3) > 1

    def test_1440p_validation_is_fast(self):
        cap = _capture_no_init()
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, size=(1440, 2560, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(5):
            valid, _ = cap.validate_frame(frame)
        dt_ms = (time.perf_counter() - t0) / 5 * 1000
        assert valid is True
        # Before: ~54 ms/call on this VM. Generous bound; sampled path is ~2 ms.
        assert dt_ms < 20.0, f"validate_frame too slow: {dt_ms:.1f} ms/call"


# --- I2: normalize_illumination direct-gray CLAHE ------------------------

class TestNormalizeIllumination:
    def test_bgr_output_shape_dtype(self):
        rng = np.random.default_rng(0)
        out = DXGIDesktopCapture.normalize_illumination(
            rng.integers(0, 256, size=(108, 192, 3), dtype=np.uint8)
        )
        assert out.ndim == 2 and out.shape == (108, 192)
        assert out.dtype == np.uint8

    def test_gray_input_unchanged_shape(self):
        rng = np.random.default_rng(0)
        out = DXGIDesktopCapture.normalize_illumination(
            rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        )
        assert out.shape == (64, 64) and out.dtype == np.uint8

    def test_deterministic(self):
        rng = np.random.default_rng(1)
        img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        a = DXGIDesktopCapture.normalize_illumination(img)
        b = DXGIDesktopCapture.normalize_illumination(img)
        assert np.array_equal(a, b)

    def test_low_contrast_gradient_gains_contrast(self):
        # Narrow-range gradient: CLAHE must spread it out.
        row = np.linspace(100, 120, 192, dtype=np.uint8)
        img = np.tile(row, (108, 1))
        img3 = np.stack([img, img, img], axis=2)
        out = DXGIDesktopCapture.normalize_illumination(img3)
        assert float(np.std(out)) > float(np.std(img))

    def test_empty_input_returns_1x1(self):
        out = DXGIDesktopCapture.normalize_illumination(np.zeros((0, 0, 3), dtype=np.uint8))
        assert out.shape == (1, 1)

    def test_clahe_singleton_is_shared(self):
        assert _get_clahe() is _get_clahe()

    def test_bgr_1080p_is_fast(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        t0 = time.perf_counter()
        for _ in range(3):
            DXGIDesktopCapture.normalize_illumination(img)
        dt_ms = (time.perf_counter() - t0) / 3 * 1000
        # Before: ~450 ms/call on this VM. Generous bound.
        assert dt_ms < 300.0, f"normalize_illumination too slow: {dt_ms:.1f} ms/call"


# --- I3: sampled std in template matching / dynamic threshold -----------

class TestSampledStd:
    def test_constant_array_std_is_zero_at_any_stride(self):
        assert _sampled_std(np.zeros((1080, 1920, 3), dtype=np.uint8)) == 0.0
        assert _sampled_std(np.full((1080, 1920), 7, dtype=np.uint8)) == 0.0

    def test_small_array_std_is_exact(self):
        arr = np.arange(12, dtype=np.uint8).reshape(3, 4)
        assert _sampled_std(arr) == pytest.approx(float(np.std(arr)))

    def test_random_array_std_close_to_exact(self):
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        assert _sampled_std(arr) == pytest.approx(float(np.std(arr)), rel=0.05)


class TestMatchTemplateTolerant:
    def _frame_with_template(self):
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
        tmpl = frame[100:164, 200:264, :].copy()
        return frame, tmpl

    def test_embedded_template_still_found(self):
        cap = _dxgi_no_init()
        frame, tmpl = self._frame_with_template()
        matched, score, loc, thr = cap.match_template_tolerant(frame, tmpl)
        assert matched is True
        assert loc == (200, 100)
        assert score >= thr

    def test_black_frame_black_template_no_match(self):
        # F-1840 guard must survive the sampled-std change.
        cap = _dxgi_no_init()
        black = np.zeros((480, 640, 3), dtype=np.uint8)
        matched, score, loc, _ = cap.match_template_tolerant(black, black[0:64, 0:64])
        assert matched is False
        assert score == 0.0 and loc == (-1, -1)

    def test_oversized_template_no_match(self):
        cap = _dxgi_no_init()
        frame, tmpl = self._frame_with_template()
        matched, _, loc, _ = cap.match_template_tolerant(tmpl, frame)
        assert matched is False and loc == (-1, -1)

    def test_1080p_match_is_fast(self):
        cap = _dxgi_no_init()
        rng = np.random.default_rng(1)
        frame = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        tmpl = frame[100:164, 200:264, :].copy()
        cap.match_template_tolerant(frame, tmpl)  # warm-up
        t0 = time.perf_counter()
        matched, _, _, _ = cap.match_template_tolerant(frame, tmpl)
        dt_ms = (time.perf_counter() - t0) * 1000
        assert matched is True
        # Before: ~1981 ms/call on this VM. Generous bound.
        assert dt_ms < 1000.0, f"match_template_tolerant too slow: {dt_ms:.1f} ms/call"

    def test_dynamic_threshold_range(self):
        cap = _dxgi_no_init()
        rng = np.random.default_rng(0)
        low = np.full((100, 100), 128, dtype=np.uint8)
        high = rng.integers(0, 256, size=(100, 100), dtype=np.uint8)
        t_low = cap.compute_dynamic_threshold(low)
        t_high = cap.compute_dynamic_threshold(high)
        assert 0.75 <= t_low <= 0.85
        assert 0.75 <= t_high <= 0.85
        assert t_low <= t_high
