# -*- coding: utf-8 -*-
"""
tests/test_d62_calibration_wizard_dev.py
=========================================
D62 (calibration_wizard) DEV improvements — verification suite.

NOTE (2026-11-08 21:12 Selçuk rule): coding agents do NOT run pytest.
These tests are written for the dedicated test agents to run, in queue:
    QT_QPA_PLATFORM=offscreen <venv>/bin/python -m pytest \\
        tests/test_d62_calibration_wizard_dev.py -v
plus the pre-existing calibration suites:
    QT_QPA_PLATFORM=offscreen <venv>/bin/python -m pytest \\
        tests/test_w48_calibration_wizard_exemplar_bridge.py \\
        tests/test_all_bot_screens_and_tabs_suite.py::test_tab05_calibration_wizard_consumable_hud_and_completeness -v

Covers:
  1. _frame_gray reuses the shared gray frame (no redundant full-frame
     BGR→GRAY conversion) and _locate_resource_node returns identical
     results with / without the cached gray.
  2. _best_yolo_ann picks the highest-confidence annotation.
  3. _locate_obstacle / _locate_path_ground keep their exact behaviour
     after consolidation into _locate_yolo_or_template (YOLO hit,
     below-gate YOLO + template fallback, plain miss).
  4. auto_calibrate_consumables_and_hud reports the lower skill bar
     (skill_slots, 12 entries) that its docstring always promised:
     flat frames report no ready slot, a textured blue-dominant skill
     bar reports ready slots and persists them as class 20 ROIs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest

from ui import calibration_wizard as cw


# ---------------------------------------------------------------- helpers

def _noise_frame(h: int = 480, w: int = 640, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _template_dir_with(frame: np.ndarray, specs) -> Path:
    """Write templates cut from `frame`; specs = [(filename, (x, y, w, h))]."""
    tmp = Path(tempfile.mkdtemp(prefix="test_d62_tpl_"))
    for name, (x, y, w, h) in specs:
        cv2.imwrite(str(tmp / name), frame[y : y + h, x : x + w])
    return tmp


class _CountingCvt:
    """Wraps cv2.cvtColor, counting full-frame BGR→GRAY conversions."""

    def __init__(self, orig, shape):
        self.orig = orig
        self.shape = shape
        self.full_gray = 0

    def __call__(self, src, code, *args, **kwargs):
        if (
            code == cv2.COLOR_BGR2GRAY
            and isinstance(src, np.ndarray)
            and src.shape[:2] == self.shape
        ):
            self.full_gray += 1
        return self.orig(src, code, *args, **kwargs)


# ---------------------------------------------------------------- 1. gray cache

def test_frame_gray_reuses_cached_gray_and_matches_direct_conversion():
    frame = _noise_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    assert np.array_equal(cw._frame_gray(frame, {"frame_gray": gray}), gray)
    # Shape mismatch / missing cache falls back to a real conversion.
    assert np.array_equal(
        cw._frame_gray(frame, {"frame_gray": np.zeros((10, 10), dtype=np.uint8)}),
        gray,
    )
    assert np.array_equal(cw._frame_gray(frame, {}), gray)
    assert np.array_equal(cw._frame_gray(frame, None), gray)


def test_locate_resource_node_uses_cached_gray_without_changing_result(monkeypatch):
    frame = _noise_frame()
    tdir = _template_dir_with(frame, [("herb_aria_template.png", (100, 100, 60, 60))])
    counting = _CountingCvt(cv2.cvtColor, frame.shape[:2])
    monkeypatch.setattr(cv2, "cvtColor", counting)

    counting.full_gray = 0
    res_direct = cw._locate_resource_node(
        frame, 0, "herb_aria", 0.60, "herb_aria_template.png",
        yolo_anns=[], templates_dir=tdir,
    )
    direct_calls = counting.full_gray

    counting.full_gray = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    counting.full_gray = 0  # the precompute itself is not the locator's cost
    res_cached = cw._locate_resource_node(
        frame, 0, "herb_aria", 0.60, "herb_aria_template.png",
        yolo_anns=[], templates_dir=tdir, frame_gray=gray,
    )
    cached_calls = counting.full_gray

    assert res_direct["ok"] is True and res_cached["ok"] is True
    assert res_direct["bbox"] == res_cached["bbox"]
    assert res_direct["score"] == pytest.approx(res_cached["score"])
    assert direct_calls == 1
    assert cached_calls == 0


# ---------------------------------------------------------------- 2. best ann

def test_best_yolo_ann_picks_highest_confidence():
    anns = [
        {"class_id": 14, "confidence": 0.5, "bbox": (0, 0, 10, 10)},
        {"class_id": 14, "confidence": 0.9, "bbox": (5, 5, 10, 10)},
        {"class_id": 15, "confidence": 0.99, "bbox": (9, 9, 10, 10)},
    ]
    best = cw._best_yolo_ann(anns, 14)
    assert best is not None and best["confidence"] == 0.9
    assert cw._best_yolo_ann(anns, 99) is None
    assert cw._best_yolo_ann([], 14) is None
    assert cw._best_yolo_ann(None, 14) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------- 3. obstacle/path

def test_obstacle_and_path_yolo_hit_and_plain_miss():
    frame = _noise_frame()
    anns = [{"class_id": 14, "confidence": 0.85, "bbox": (50, 60, 40, 30)}]
    res = cw._locate_obstacle(frame, 0.60, yolo_anns=anns, templates_dir=None)
    assert res["ok"] is True
    assert res["bbox"] == (50, 60, 40, 30)
    assert res["score"] == pytest.approx(0.85)

    # No YOLO hit, no template dir -> plain miss with the locator's reason.
    assert cw._locate_obstacle(frame, 0.60, yolo_anns=[], templates_dir=None)[
        "reason"
    ] == "engel bu karede yok"
    assert cw._locate_path_ground(frame, 0.60, yolo_anns=[], templates_dir=None)[
        "reason"
    ] == "yol/zemin bu karede yok"


def test_obstacle_below_gate_yolo_still_falls_through_to_template():
    """Below-gate YOLO must NOT short-circuit the template fallback."""
    frame = _noise_frame()
    tdir = _template_dir_with(frame, [("obstacle_template.png", (200, 150, 80, 80))])
    low = [{"class_id": 14, "confidence": 0.30, "bbox": (10, 10, 20, 20)}]
    res = cw._locate_obstacle(frame, 0.60, yolo_anns=low, templates_dir=tdir)
    assert res["ok"] is True  # template (cut from this frame) matches ~1.0
    assert res["bbox"] == (200, 150, 80, 80)

    tdir2 = _template_dir_with(frame, [("path_ground_template.png", (300, 200, 70, 70))])
    res2 = cw._locate_path_ground(frame, 0.60, yolo_anns=low, templates_dir=tdir2)
    assert res2["ok"] is True
    assert res2["bbox"] == (300, 200, 70, 70)


# ---------------------------------------------------------------- 4. skill slots

def _flat_frame(h: int = 720, w: int = 1280) -> np.ndarray:
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :] = (40, 60, 180)  # BGR: red-dominant flat -> no mana, no contrast
    return frame


def _skill_bar_noisy_frame() -> np.ndarray:
    """720x1280 frame whose whole bottom HUD band is blue-dominant noise:
    bright, saturated, high contrast, blue mana indicator present, so the
    SkillBarReader measures ready slots."""
    frame = _flat_frame()
    rng = np.random.default_rng(7)
    # FIX (R13b): int(0.88 * 720) == 633, so the slice below is 87 rows
    # tall, but the band was built with 86 rows -> ValueError (broadcast).
    # Derive the height from the slice so they can never disagree again.
    # G spans the full range: with G in [90, 161) the gray std of the band
    # is only ~13.8, below SkillBarReader.min_contrast_std (22.0, D03
    # gray-std contrast, which predates this test), so no slot ever
    # measured ready. Full-range G raises gray std to ~44 while the band
    # stays blue-dominant (blue_ratio ~0.79), bright (V ~230) and
    # saturated (S ~190), so the reader measures ready slots.
    top = int(0.88 * 720)
    band_h = 720 - top
    band = np.zeros((band_h, 1280, 3), dtype=np.uint8)
    band[:, :, 0] = rng.integers(170, 256, (band_h, 1280), dtype=np.uint8)  # B high
    band[:, :, 1] = rng.integers(0, 256, (band_h, 1280), dtype=np.uint8)  # G full-range
    band[:, :, 2] = rng.integers(40, 111, (band_h, 1280), dtype=np.uint8)  # R low
    frame[top:, :] = band
    return frame


def _wizard(base_dir=None):
    wiz = cw.CalibrationWizard.__new__(cw.CalibrationWizard)
    wiz.templates_dir = Path("templates")
    if base_dir is not None:
        wiz.base_dir = Path(base_dir)
    return wiz


def test_consumables_and_hud_reports_skill_slots_on_flat_frame():
    wiz = _wizard()
    res = wiz.auto_calibrate_consumables_and_hud(frame_bgr=_flat_frame())
    # 12-slot bar (1-8, Q, E, R, T) — SkillBarReader.SKILL_SLOTS (G2).
    assert len(res["skill_slots"]) == 12
    assert [s["slot"] for s in res["skill_slots"]] == [
        "1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T",
    ]
    for idx, slot in enumerate(res["skill_slots"], start=1):
        assert slot["slot_index"] == idx
        bx, by, bw, bh = slot["bbox"]
        assert 0 <= bx and 0 <= by and bx + bw <= 1280 and by + bh <= 720
        assert bw > 0 and bh > 0
        assert 0.0 <= slot["score"] <= 1.0
        # Flat red-dominant frame: no mana indicator, zero contrast ->
        # the reader measures every slot not-ready (class 21, valid False).
        assert slot["valid"] is False
        assert slot["class_id"] == 21
        assert slot["class_name"] == "skill_slot_cooldown"
    assert res["skill_count"] == 0
    assert res["skill_persisted_count"] == 0
    # Consumables still carry `calibrated` on this frame (as before).
    assert res["calibrated"] is True


def test_skill_slots_ready_and_persisted_on_textured_frame(tmp_path):
    wiz = _wizard(base_dir=tmp_path)
    res = wiz.auto_calibrate_consumables_and_hud(frame_bgr=_skill_bar_noisy_frame())
    assert len(res["skill_slots"]) == 12
    assert res["skill_count"] >= 1
    assert res["calibrated"] is True
    assert res["skill_persisted_count"] == res["skill_count"]

    rois_path = tmp_path / "config" / "calibrated_rois.json"
    assert rois_path.exists()
    data = json.loads(rois_path.read_text(encoding="utf-8"))
    saved_keys = [k for k in data.get("rois", {}) if k.startswith("skill_slot_")]
    assert len(saved_keys) == res["skill_count"]
    for key in saved_keys:
        assert data["rois"][key]["yolo_class_id"] == 20
        assert data["rois"][key]["class_name"] == "skill_slot_ready"


def test_consumables_and_hud_no_frame_reports_empty_skill_slots():
    wiz = _wizard()
    res = wiz.auto_calibrate_consumables_and_hud(frame_bgr=None)
    assert res["calibrated"] is False
    assert res["skill_slots"] == []
    assert res["reason"] == "no_frame"
