# -*- coding: utf-8 -*-
"""R51 regression tests for dataset_factory/monster_diamond_detector.py.

- S-2502: integer dtype frames must be clipped to 0..255 before the uint8
  cast (uint16 value 300 wrapped to 44 and silently broke HSV thresholds).
- S-2505: inspect_nameplate_roi confidence must scale with evidence
  (diamond tips / red agro / yellow stroke), not sit at a flat 0.90 when
  has_agro comes only from the diamond tips (red_px == 0).
- F-1999 (recorded, not fixed here): the module has no production caller;
  this file at least pins that the public API imports and works.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import cv2

from dataset_factory.monster_diamond_detector import (
    DiamondTipMonsterDetector,
    detect_diamond_tip_monsters,
    get_monster_diamond_detector,
)


# ----------------------------------------------------------------------
# S-2502: integer dtype clipping in _coerce_bgr_frame
# ----------------------------------------------------------------------
def test_coerce_uint16_clips_instead_of_wrapping():
    frame = np.full((4, 4, 3), 300, dtype=np.uint16)
    out = DiamondTipMonsterDetector._coerce_bgr_frame(frame)
    assert out is not None
    assert out.dtype == np.uint8
    assert np.all(out == 255)  # was 44 (300 % 256) before the fix


def test_coerce_int16_clips_negative_and_overflow():
    frame = np.zeros((4, 4, 3), dtype=np.int16)
    frame[:, :, 0] = -5
    frame[:, :, 1] = 128
    frame[:, :, 2] = 300
    out = DiamondTipMonsterDetector._coerce_bgr_frame(frame)
    assert out is not None
    assert out.dtype == np.uint8
    assert np.all(out[:, :, 0] == 0)
    assert np.all(out[:, :, 1] == 128)
    assert np.all(out[:, :, 2] == 255)


def test_coerce_uint8_passthrough_unchanged():
    frame = np.full((4, 4, 3), 77, dtype=np.uint8)
    out = DiamondTipMonsterDetector._coerce_bgr_frame(frame)
    assert out is not None
    assert np.all(out == 77)


# ----------------------------------------------------------------------
# S-2505: evidence-scaled confidence in inspect_nameplate_roi
# ----------------------------------------------------------------------
def _make_roi(n_yellow_stripes: int, with_red: bool = False) -> np.ndarray:
    """Synthetic nameplate ROI (BGR): thin yellow strokes in the middle,
    cyan diamond-tip blocks in the left/right 25% bands, optional red block.
    Built in HSV space so hues are exact, then converted to BGR."""
    w, h = 200, 40
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    # Cyan tips: solid blocks inside the left/right bands.
    hsv[15:25, 5:25] = (90, 180, 200)  # left band
    hsv[15:25, 175:195] = (90, 180, 200)  # right band
    # Yellow strokes: 2px-wide vertical stripes (thin => strong top-hat).
    for i in range(n_yellow_stripes):
        x = 60 + i * 6
        hsv[:, x : x + 2] = (25, 200, 200)
    if with_red:
        hsv[28:38, 90:110] = (0, 200, 200)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_roi_confidence_scales_with_yellow_stroke_when_tips_only():
    det = DiamondTipMonsterDetector()
    few = det.inspect_nameplate_roi(_make_roi(3))
    many = det.inspect_nameplate_roi(_make_roi(10))
    assert few["is_monster_nameplate"] is True
    assert many["is_monster_nameplate"] is True
    assert few["has_diamond_tips"] is True
    assert many["has_diamond_tips"] is True
    assert few["red_agro_px"] == 0 and many["red_agro_px"] == 0
    # Before the fix both were a flat 0.90 (red_px == 0); now the stronger
    # stroke evidence must score higher.
    assert many["confidence"] > few["confidence"]
    assert 0.90 < few["confidence"] <= 0.96
    assert 0.90 < many["confidence"] <= 0.96


def test_roi_confidence_higher_with_both_tips_and_red():
    det = DiamondTipMonsterDetector()
    tips_only = det.inspect_nameplate_roi(_make_roi(5, with_red=False))
    both = det.inspect_nameplate_roi(_make_roi(5, with_red=True))
    assert tips_only["is_monster_nameplate"] is True
    assert both["is_monster_nameplate"] is True
    assert both["has_agro_badge"] is True
    assert both["red_agro_px"] > 0
    assert both["confidence"] > tips_only["confidence"]
    assert both["confidence"] <= 0.96


# ----------------------------------------------------------------------
# F-1999: public API is importable and usable (wiring is cross-slice)
# ----------------------------------------------------------------------
def test_public_api_singleton_and_detect_on_blank_frame():
    det = get_monster_diamond_detector()
    assert isinstance(det, DiamondTipMonsterDetector)
    assert get_monster_diamond_detector() is det
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert detect_diamond_tip_monsters(blank) == []
    assert det.inspect_nameplate_roi(blank)["confidence"] == 0.0
