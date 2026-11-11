# -*- coding: utf-8 -*-
r"""
D45 tests: improvements to `dataset_factory/crest_nameplate_verifier.py`.

1. Template-match search restriction (`detect_winged_crests`): matching only
   inside expanded white-contour rects must return IDENTICAL detections to
   the old full-mask `matchTemplate` (proved by a test-side reimplementation
   of the old algorithm) while being much faster on a 1080p frame.
2. Batch HSV reuse (`verify_annotations` / `verify_annotation` `hsv=`):
   shared HSV must give identical results, and the parameter must be live
   (a wrong HSV changes the outcome).
3. Dict-like accessors on `CrestNameplateVerificationResult` must not run
   `asdict` for key-only questions (`in` / `keys()` / `iter()`).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import dataset_factory.crest_nameplate_verifier as verifier_mod
from dataset_factory.crest_nameplate_verifier import CrestNameplateVerifier

VERIFIER = CrestNameplateVerifier()


# ----------------------------------------------------------------------
# Reference: the OLD full-mask template-match algorithm (pre-D45), kept
# here so the new search restriction is locked to identical results.
# ----------------------------------------------------------------------
def _old_detect_winged_crests(frame_bgr, hsv=None, roi=None):
    """Pre-D45 `detect_winged_crests`: full-mask matchTemplate + fallback."""
    hsv_full = VERIFIER._ensure_hsv(frame_bgr, hsv)
    if hsv_full is None:
        return []
    h, w = hsv_full.shape[:2]
    if roi is not None:
        rx1 = max(0, min(w - 1, int(roi[0])))
        ry1 = max(0, min(h - 1, int(roi[1])))
        rx2 = max(rx1 + 1, min(w, int(roi[2])))
        ry2 = max(ry1 + 1, min(h, int(roi[3])))
    else:
        rx1, ry1, rx2, ry2 = 0, 0, w, h
    sub_hsv = hsv_full[ry1:ry2, rx1:rx2]
    if sub_hsv.size == 0:
        return []
    white_mask = cv2.inRange(sub_hsv, VERIFIER.WHITE_CREST_LOWER, VERIFIER.WHITE_CREST_UPPER)
    crests = []
    th, tw = VERIFIER._crest_template_mask.shape[:2]
    if white_mask.shape[0] >= th and white_mask.shape[1] >= tw:
        res = cv2.matchTemplate(white_mask, VERIFIER._crest_template_mask, cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.45)
        candidates = sorted(
            ((float(res[y_pt, x_pt]), int(x_pt), int(y_pt)) for y_pt, x_pt in zip(ys, xs)),
            reverse=True,
        )
        accepted_centers = []
        for score, x_pt, y_pt in candidates:
            cand_cx = rx1 + x_pt + 0.5 * tw
            cand_cy = ry1 + y_pt + 0.5 * th
            if any(math.hypot(cand_cx - ax, cand_cy - ay) <= 16.0 for ax, ay in accepted_centers):
                continue
            accepted_centers.append((cand_cx, cand_cy))
            abs_x = rx1 + int(x_pt)
            abs_y = ry1 + int(y_pt)
            crests.append(
                {
                    "cx": float(abs_x + 0.5 * tw),
                    "cy": float(abs_y + 0.5 * th),
                    "bbox": (abs_x, abs_y, tw, th),
                    "score": round(min(0.99, 0.75 + 0.24 * score), 4),
                }
            )
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        if cw < 8 or ch < 8 or cw > 64 or ch > 64:
            continue
        sub_wm = white_mask[y : y + ch, x : x + cw]
        white_px = int(np.count_nonzero(sub_wm))
        if white_px < 10:
            continue
        flipped = np.fliplr(sub_wm)
        inter = int(np.count_nonzero(sub_wm & flipped))
        union = int(np.count_nonzero(sub_wm | flipped))
        sym = inter / float(max(1, union))
        if sym < VERIFIER.min_crest_symmetry:
            continue
        abs_x = rx1 + x
        abs_y = ry1 + y
        if any(
            math.hypot((abs_x + 0.5 * cw) - c["cx"], (abs_y + 0.5 * ch) - c["cy"]) <= 16.0
            for c in crests
        ):
            continue
        crests.append(
            {
                "cx": float(abs_x + 0.5 * cw),
                "cy": float(abs_y + 0.5 * ch),
                "bbox": (int(abs_x), int(abs_y), int(cw), int(ch)),
                "score": round(min(0.95, 0.72 + 0.25 * sym), 4),
            }
        )
    crests.sort(key=lambda d: d["score"], reverse=True)
    return crests


def _paint_crest(frame, cx=960, cy=400):
    cv2.line(frame, (cx - 13, cy - 13), (cx, cy - 1), (250, 250, 250), 5)
    cv2.line(frame, (cx + 13, cy - 13), (cx, cy - 1), (250, 250, 250), 5)
    cv2.line(frame, (cx, cy - 1), (cx, cy + 13), (250, 250, 250), 5)


def _paint_nameplate(frame, cx=964, cy=434):
    cv2.rectangle(frame, (cx - 36, cy - 9), (cx + 36, cy + 9), (20, 20, 20), -1)
    for gx in range(cx - 32, cx + 30, 9):
        cv2.rectangle(frame, (gx, cy - 6), (gx + 5, cy + 4), (235, 225, 30), -1)


def _clean_frame():
    frame = np.full((1080, 1920, 3), 85, dtype=np.uint8)
    _paint_crest(frame)
    _paint_nameplate(frame)
    return frame


def _cluttered_frame():
    frame = _clean_frame()
    # White clutter far from the crest: rectangles + text-like strokes.
    cv2.rectangle(frame, (60, 60), (300, 110), (245, 245, 245), -1)
    cv2.rectangle(frame, (1500, 900), (1850, 960), (240, 240, 240), -1)
    for x in range(100, 900, 40):
        cv2.line(frame, (x, 1000), (x + 25, 1000), (250, 250, 250), 4)
    cv2.circle(frame, (1800, 120), 26, (248, 248, 248), -1)
    return frame


def _sparse_noise_frame():
    rng = np.random.default_rng(7)
    frame = np.full((1080, 1920, 3), 40, dtype=np.uint8)
    _paint_crest(frame, cx=500, cy=300)
    ys = rng.integers(0, 1080, size=200)
    xs = rng.integers(0, 1920, size=200)
    frame[ys, xs] = (250, 250, 250)
    return frame


def _detections_key(dets):
    return sorted(((tuple(d["bbox"]), round(d["score"], 3)) for d in dets))


def _assert_same_detections(new, old):
    assert len(new) == len(old), f"count differs: new={new} old={old}"
    for (nb, ns), (ob, os_) in zip(_detections_key(new), _detections_key(old)):
        assert nb == ob, f"bbox differs: new={nb} old={ob}"
        assert abs(ns - os_) <= 0.001, f"score differs: new={ns} old={os_}"


# ----------------------------------------------------------------------
# 1. Template-match search restriction: identical results, faster.
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "frame_fn",
    [_clean_frame, _cluttered_frame, _sparse_noise_frame],
    ids=["clean", "cluttered", "sparse-noise"],
)
def test_d45_template_search_matches_full_frame_detection(frame_fn):
    frame = frame_fn()
    new = VERIFIER.detect_winged_crests(frame)
    old = _old_detect_winged_crests(frame)
    assert len(new) >= 1, "painted crest was not detected"
    _assert_same_detections(new, old)


def test_d45_template_search_matches_full_frame_detection_with_roi():
    frame = _cluttered_frame()
    roi = (800, 250, 1150, 550)
    new = VERIFIER.detect_winged_crests(frame, roi=roi)
    old = _old_detect_winged_crests(frame, roi=roi)
    _assert_same_detections(new, old)


def test_d45_template_search_empty_and_all_white_frames():
    empty = np.full((1080, 1920, 3), 85, dtype=np.uint8)
    assert VERIFIER.detect_winged_crests(empty) == []
    assert _old_detect_winged_crests(empty) == []
    white = np.full((1080, 1920, 3), 250, dtype=np.uint8)
    _assert_same_detections(
        VERIFIER.detect_winged_crests(white), _old_detect_winged_crests(white)
    )


def test_d45_template_search_rects_unit():
    th, tw = 28, 29
    assert VERIFIER._template_search_rects([], 1920, 1080, th, tw) == []
    # One contour: rect = bbox expanded by (tw-1, th-1), clamped.
    cnt = np.array([[[100, 100]], [[120, 100]], [[120, 130]], [[100, 130]]], dtype=np.int32)
    rects = VERIFIER._template_search_rects([cnt], 1920, 1080, th, tw)
    assert rects == [(100 - 28, 100 - 27, 21 + 2 * 28, 31 + 2 * 27)]
    # >64 fragments collapse to a single union rect.
    frags = [
        np.array([[[10 * i, 10 * i]], [[10 * i + 1, 10 * i]], [[10 * i + 1, 10 * i + 1]]], dtype=np.int32)
        for i in range(80)
    ]
    rects = VERIFIER._template_search_rects(frags, 1920, 1080, th, tw)
    assert len(rects) == 1


def test_d45_detect_winged_crests_fast_on_clean_1080p_frame():
    frame = _clean_frame()
    best = min(
        (lambda t0=time.perf_counter(): (VERIFIER.detect_winged_crests(frame), time.perf_counter() - t0)[1])()
        for _ in range(3)
    )
    # Old full-mask matchTemplate measured 237-474 ms (best) on this VM;
    # the restricted search must stay well under that (floor is the HSV
    # conversion of the full frame, ~50 ms under load).
    assert best < 0.200, f"detect_winged_crests too slow: best={best:.3f}s"


# ----------------------------------------------------------------------
# 2. Batch HSV reuse.
# ----------------------------------------------------------------------
def _batch_annotations():
    return [
        {"class_id": 6, "bbox": (946, 386, 29, 28), "confidence": 0.84, "source": "vision_3d_crest"},
        {"class_id": 0, "bbox": (930, 427, 68, 14), "confidence": 0.82, "source": "3d_world_nameplate"},
        {"class_id": 0, "bbox": (260, 500, 52, 48), "confidence": 0.91, "source": "chromatic_salient_3d_resource"},
        {"class_id": 0, "bbox": (1500, 800, 52, 48), "confidence": 0.88, "source": "chromatic_salient_3d_resource"},
    ]


def test_d45_batch_hsv_reuse_identical_results():
    frame = _clean_frame()
    default = VERIFIER.verify_annotations(_batch_annotations(), frame_bgr=frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    shared = VERIFIER.verify_annotations(_batch_annotations(), frame_bgr=frame, hsv=hsv)
    assert shared == default
    # Single-annotation path: same result with and without precomputed HSV.
    ann = _batch_annotations()[2]
    a = VERIFIER.verify_annotation(dict(ann), frame_bgr=frame, all_annotations=_batch_annotations())
    b = VERIFIER.verify_annotation(dict(ann), frame_bgr=frame, all_annotations=_batch_annotations(), hsv=hsv)
    assert a == b


def test_d45_batch_hsv_parameter_is_live_not_placebo():
    """A wrong HSV must change the outcome — the parameter is really read.

    The crest annotation below has no geometric partner in `all_annotations`,
    so its verification must go through the frame path (turquoise search),
    which reads the supplied HSV.
    """
    frame = _clean_frame()
    crest_only = [
        {"class_id": 6, "bbox": (946, 386, 29, 28), "confidence": 0.84, "source": "vision_3d_crest"},
    ]
    real = VERIFIER.verify_annotations(crest_only, frame_bgr=frame)
    assert real[0]["verification_status"] == "Emin", real
    assert real[0]["crest_nameplate_verified"] is True
    black_hsv = cv2.cvtColor(np.zeros((1080, 1920, 3), dtype=np.uint8), cv2.COLOR_BGR2HSV)
    wrong = VERIFIER.verify_annotations(crest_only, frame_bgr=frame, hsv=black_hsv)
    assert wrong[0]["verification_status"] == "Şüpheli", wrong
    assert wrong[0]["crest_nameplate_verified"] is False


# ----------------------------------------------------------------------
# 3. Dict-like accessors without asdict.
# ----------------------------------------------------------------------
def test_d45_result_key_accessors_match_to_dict():
    res = VERIFIER.check_vertical_alignment((946, 386, 29, 28), (930, 420, 60, 14))
    assert res.keys() == list(res.to_dict().keys())
    assert list(res) == res.keys()
    assert "confidence" in res
    assert "is_verified" in res
    assert "no_such_key" not in res
    assert "_DERIVED_KEYS" not in res
    assert res["confidence"] == res.confidence
    assert res.get("missing", "dflt") == "dflt"
    assert dict(res.items()) == res.to_dict()
    assert "_DERIVED_KEYS" not in res.to_dict()
    assert len(res.to_dict()) == 17


def test_d45_result_key_accessors_do_not_call_asdict(monkeypatch):
    res = VERIFIER.check_vertical_alignment((946, 386, 29, 28), (930, 420, 60, 14))
    monkeypatch.setattr(
        verifier_mod, "asdict", lambda *a, **k: (_ for _ in ()).throw(AssertionError("asdict must not run"))
    )
    assert "confidence" in res
    assert "is_verified" in res
    assert "no_such_key" not in res
    assert res.keys() == [
        "verified", "trusted", "weak", "confidence", "status", "status_label",
        "dx", "dy", "crest_bbox", "nameplate_bbox", "crest_score",
        "turquoise_ratio", "turquoise_pixels", "reason",
        "is_verified", "aligned", "has_evidence",
    ]
    assert list(res) == res.keys()
    # Value access still goes through asdict, by design.
    with pytest.raises(AssertionError):
        res.to_dict()
