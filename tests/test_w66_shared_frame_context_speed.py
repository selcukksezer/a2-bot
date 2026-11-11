# -*- coding: utf-8 -*-
"""
Worker W66 Unit Tests: Shared Frame Context Single-Pass Matrix Caching & Sharing
================================================================================
Validates that `YOLOAutoLabeler.auto_label_frame` computes `bgr`, `hsv`, `gray`,
and `edges` once in `shared_frame_context` and shares/reuses them across calls
and internal sub-scanners without redundant full-frame conversions.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

import dataset_factory.yolo_auto_labeler as auto_labeler_mod
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler


def _make_lightweight_labeler(crests: list | None = None) -> YOLOAutoLabeler:
    """Create a YOLOAutoLabeler with lightweight mock sub-engines for fast deterministic tests.

    `crests` overrides what the mock vision engine reports from
    `detect_3d_gather_crests` (default: no crests), so crest-present paths
    can reuse this factory instead of rebuilding the same three mocks.
    """
    vision_mock = MagicMock()
    vision_mock.detect_3d_gather_crests.return_value = list(crests) if crests is not None else []
    vision_mock.detect_floating_resource_icon.return_value = (False, 0.0, None, None)
    vision_mock.detect_gather_prompt.return_value = (False, 0.0, None)
    vision_mock.last_detected_gather_key = "f"

    material_mock = MagicMock()
    material_mock.scan_frame_for_materials.return_value = []

    minimap_mock = MagicMock()
    minimap_mock.scan_gatherable_materials.return_value = {
        "gatherable_nodes": [],
        "monster_dots": [],
    }

    return YOLOAutoLabeler(
        vision_engine=vision_mock,
        material_scanner=material_mock,
        minimap_scanner=minimap_mock,
        exemplar_memory=None,
    )


def _create_sample_frame(height: int = 720, width: int = 1280) -> np.ndarray:
    """Create a synthetic game frame with edges and chromatic regions."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 350), (40, 180, 240), -1)
    cv2.circle(frame, (640, 360), 60, (200, 90, 50), 3)
    cv2.line(frame, (100, 600), (1100, 600), (255, 255, 255), 2)
    return frame


def test_1_shared_frame_context_populates_all_four_matrices_on_first_call():
    """
    Test 1: Passing an empty `shared_frame_context` dict to `auto_label_frame`
    populates `bgr`, `hsv`, `gray`, and `edges` with valid numpy matrices of expected shapes.
    """
    labeler = _make_lightweight_labeler()
    frame = _create_sample_frame(720, 1280)
    shared_ctx: Dict[str, Any] = {}

    labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=shared_ctx)

    for key in ("bgr", "hsv", "gray", "edges"):
        assert key in shared_ctx, f"Missing expected key '{key}' in shared_frame_context"
        assert isinstance(shared_ctx[key], np.ndarray), f"'{key}' must be a numpy.ndarray"

    assert shared_ctx["bgr"].shape == (720, 1280, 3)
    assert shared_ctx["hsv"].shape == (720, 1280, 3)
    assert shared_ctx["gray"].shape == (720, 1280)
    assert shared_ctx["edges"].shape == (720, 1280)
    assert shared_ctx["edges"].dtype == np.uint8


def test_2_shared_frame_context_preserves_matrix_identities_across_repeated_calls():
    """
    Test 2: Calling `auto_label_frame` multiple times with the same `shared_frame_context`
    reuses the exact same in-memory matrix objects (`bgr`, `hsv`, `gray`, `edges`).
    """
    labeler = _make_lightweight_labeler()
    frame = _create_sample_frame(720, 1280)
    shared_ctx: Dict[str, Any] = {}

    labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=shared_ctx)
    first_ids = {k: id(shared_ctx[k]) for k in ("bgr", "hsv", "gray", "edges")}

    # Second and third calls with the same shared_frame_context
    labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=shared_ctx)
    labeler.auto_label_frame(frame, fast_mode=False, shared_frame_context=shared_ctx)

    for k in ("bgr", "hsv", "gray", "edges"):
        assert id(shared_ctx[k]) == first_ids[k], f"Matrix '{k}' was reallocated instead of reused"


def test_3_shared_frame_context_computes_cvtcolor_and_canny_only_once(monkeypatch):
    """
    Test 3: Verify via spies on `cv2.cvtColor` and `cv2.Canny` that full-frame HSV,
    GRAY, and Canny edge conversions happen exactly once across multiple `auto_label_frame`
    invocations sharing `shared_frame_context` (even on `learn_skills_*` stems where HUD Canny is used).

    The spy counts EVERY conversion, not only full-frame ones: per-crop
    conversions inside the sub-scanners still repeat on every call (they are
    not cached by shared_frame_context), and the test pins that fact instead of
    hiding it -- the per-call crop-conversion count must stay constant (shared
    context adds no new work per call), while full-frame conversions happen once.
    """
    labeler = _make_lightweight_labeler()
    frame = _create_sample_frame(720, 1280)
    shared_ctx: Dict[str, Any] = {}

    real_cvt_color = cv2.cvtColor
    real_canny = cv2.Canny

    full_frame_cvt_calls: List[int] = []
    crop_cvt_calls: List[int] = []
    canny_calls: List[Any] = []

    def spy_cvt_color(src, code, *args, **kwargs):
        if isinstance(src, np.ndarray) and src.shape[:2] == (720, 1280):
            full_frame_cvt_calls.append(code)
        else:
            crop_cvt_calls.append(code)
        return real_cvt_color(src, code, *args, **kwargs)

    def spy_canny(image, threshold1, threshold2, *args, **kwargs):
        canny_calls.append(image.shape)
        return real_canny(image, threshold1, threshold2, *args, **kwargs)

    monkeypatch.setattr(auto_labeler_mod.cv2, "cvtColor", spy_cvt_color)
    monkeypatch.setattr(auto_labeler_mod.cv2, "Canny", spy_canny)

    # First call: should compute HSV (1), GRAY (1), and Canny (1) at full-frame size.
    labeler.auto_label_frame(
        frame,
        fast_mode=True,
        source_path="learn_skills_test_01.png",
        shared_frame_context=shared_ctx,
    )
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2HSV) == 1
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2GRAY) == 1
    assert len(canny_calls) == 1
    assert canny_calls[0] == (720, 1280)
    crop_after_first = len(crop_cvt_calls)

    # Second call with same shared_frame_context: 0 additional full-frame cvtColor or Canny calls.
    labeler.auto_label_frame(
        frame,
        fast_mode=True,
        source_path="learn_skills_test_02.png",
        shared_frame_context=shared_ctx,
    )
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2HSV) == 1
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2GRAY) == 1
    assert len(canny_calls) == 1
    # Per-crop conversions repeat every call (not cached) -- but the per-call
    # count must not grow: shared context must not add conversions.
    assert len(crop_cvt_calls) - crop_after_first <= crop_after_first


def test_4_shared_frame_context_forwarded_to_all_internal_subscanners(monkeypatch):
    """
    Test 4: Verify that `auto_label_frame` forwards the same `shared_frame_context`
    and precomputed `bgr`, `hsv`, `gray`, `edges` matrices to `_scan_3d_monsters`,
    `_scan_salient_3d_resources`, `_scan_hud_and_skills`, and `_scan_path_and_obstacles`.
    """
    labeler = _make_lightweight_labeler()
    frame = _create_sample_frame(720, 1280)
    shared_ctx: Dict[str, Any] = {}

    observed_contexts: Dict[str, Dict[str, Any]] = {}

    orig_monsters = labeler._scan_3d_monsters
    orig_salient = labeler._scan_salient_3d_resources
    orig_hud = labeler._scan_hud_and_skills
    orig_path = labeler._scan_path_and_obstacles

    # One wrapper factory for all four sub-scanners: each records whether it
    # received the same shared context object and the same precomputed
    # matrices. The salient scanner never receives `gray`, so it is the only
    # one whose checklist omits `same_gray`.
    _CHECKS = {
        "monsters": ("same_ctx", "same_bgr", "same_hsv", "same_gray", "has_edges"),
        "salient": ("same_ctx", "same_bgr", "same_hsv", "has_edges"),
        "hud": ("same_ctx", "same_bgr", "same_hsv", "same_gray", "has_edges"),
        "path": ("same_ctx", "same_bgr", "same_hsv", "same_gray", "has_edges"),
    }

    def _make_wrapper(name, fn):
        wanted = _CHECKS[name]

        def wrapped(*args, **kwargs):
            ctx = kwargs.get("shared_frame_context")
            checks = {
                "same_ctx": ctx is shared_ctx,
                "same_bgr": kwargs.get("frame_bgr") is shared_ctx.get("bgr"),
                "same_hsv": kwargs.get("hsv") is shared_ctx.get("hsv"),
                "same_gray": kwargs.get("gray") is shared_ctx.get("gray"),
                "has_edges": isinstance(ctx.get("edges"), np.ndarray)
                if isinstance(ctx, dict)
                else False,
            }
            observed_contexts[name] = {k: checks[k] for k in wanted}
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(labeler, "_scan_3d_monsters", _make_wrapper("monsters", orig_monsters))
    monkeypatch.setattr(labeler, "_scan_salient_3d_resources", _make_wrapper("salient", orig_salient))
    monkeypatch.setattr(labeler, "_scan_hud_and_skills", _make_wrapper("hud", orig_hud))
    monkeypatch.setattr(labeler, "_scan_path_and_obstacles", _make_wrapper("path", orig_path))

    labeler.auto_label_frame(
        frame,
        fast_mode=True,
        source_path="learn_paths_and_learn_skills_combo.png",
        shared_frame_context=shared_ctx,
    )

    for scanner_name in ("monsters", "salient", "hud", "path"):
        assert scanner_name in observed_contexts, f"{scanner_name} was not called"
        for check_name, passed in observed_contexts[scanner_name].items():
            assert passed is True, f"{scanner_name} failed check '{check_name}'"


def test_5_prepopulated_shared_frame_context_bypasses_all_matrix_computations(monkeypatch):
    """
    Test 5: When `shared_frame_context` is pre-populated with `bgr`, `hsv`, `gray`, and `edges`,
    `auto_label_frame` uses the provided matrices directly without invoking `_coerce_bgr_frame`,
    `cv2.cvtColor`, or `cv2.Canny`.
    """
    labeler = _make_lightweight_labeler()
    bgr = _create_sample_frame(720, 1280)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    shared_ctx: Dict[str, Any] = {
        "bgr": bgr,
        "hsv": hsv,
        "gray": gray,
        "edges": edges,
    }

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Should not recompute matrix when already in shared_frame_context")

    monkeypatch.setattr(labeler, "_coerce_bgr_frame", fail_if_called)
    monkeypatch.setattr(auto_labeler_mod.cv2, "cvtColor", fail_if_called)
    monkeypatch.setattr(auto_labeler_mod.cv2, "Canny", fail_if_called)

    # Pass None as frame because bgr is already inside shared_frame_context
    anns = labeler.auto_label_frame(
        None,
        fast_mode=True,
        source_path="learn_skills_cached.png",
        shared_frame_context=shared_ctx,
    )

    assert isinstance(anns, list)
    assert shared_ctx["bgr"] is bgr
    assert shared_ctx["hsv"] is hsv
    assert shared_ctx["gray"] is gray
    assert shared_ctx["edges"] is edges


def test_6_real_engines_share_full_frame_matrices_across_calls(monkeypatch):
    """
    Test 6: The single-pass caching claim proven with REAL sub-engines, not
    mocks, as an A/B comparison: two `auto_label_frame` calls on the same frame
    WITH a shared_frame_context must perform strictly fewer full-frame
    conversions than the same two calls WITHOUT one. The spy counts EVERY
    conversion (full-frame and per-crop).

    Measured reality (pinned, not hidden): sharing saves one HSV, one GRAY and
    one Canny conversion across the two calls (HSV 4->3, GRAY 2->1, Canny 2->1);
    per-crop conversions are NOT cached (equal in both arms) and one sub-scanner
    still recomputes a full-frame HSV on the second call.
    The same frame must also yield the same annotations on both shared calls.

    The frame is 360x640 instead of full 720p: this test pins conversion
    *counts*, not pixels, and the counts were verified identical at both
    sizes, while full-frame cvtColor/Canny cost scales with area.
    """
    frame = _create_sample_frame(360, 640)

    real_cvt_color = cv2.cvtColor
    real_canny = cv2.Canny

    def _counted_run(labeler, shared_ctx):
        counts = {"hsv": 0, "gray": 0, "canny": 0, "crop": 0}

        def spy_cvt_color(src, code, *args, **kwargs):
            if isinstance(src, np.ndarray) and src.shape[:2] == frame.shape[:2]:
                if code == cv2.COLOR_BGR2HSV:
                    counts["hsv"] += 1
                elif code == cv2.COLOR_BGR2GRAY:
                    counts["gray"] += 1
            else:
                counts["crop"] += 1
            return real_cvt_color(src, code, *args, **kwargs)

        def spy_canny(image, threshold1, threshold2, *args, **kwargs):
            counts["canny"] += 1
            return real_canny(image, threshold1, threshold2, *args, **kwargs)

        monkeypatch.setattr(auto_labeler_mod.cv2, "cvtColor", spy_cvt_color)
        monkeypatch.setattr(auto_labeler_mod.cv2, "Canny", spy_canny)
        try:
            anns = [
                labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=shared_ctx),
                labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=shared_ctx),
            ]
        finally:
            monkeypatch.undo()
        return counts, anns

    # Arm A: no shared context (fresh dict per call) -- the baseline.
    baseline_counts, _ = _counted_run(YOLOAutoLabeler(), None)
    # Arm B: one shared context across both calls.
    shared_ctx: Dict[str, Any] = {}
    shared_counts, shared_anns = _counted_run(YOLOAutoLabeler(), shared_ctx)

    assert shared_counts["hsv"] < baseline_counts["hsv"]
    assert shared_counts["gray"] < baseline_counts["gray"]
    assert shared_counts["canny"] < baseline_counts["canny"]
    # GRAY and Canny are fully cached (exactly one each across both calls);
    # HSV is only partially cached (one sub-scanner recomputes it per call).
    assert shared_counts["gray"] == 1
    assert shared_counts["canny"] == 1
    # Per-crop conversions are not cached by shared_frame_context at all.
    assert shared_counts["crop"] == baseline_counts["crop"]

    # Same frame + same shared context must label identically on both calls.
    assert [a["class_id"] for a in shared_anns[0]] == [a["class_id"] for a in shared_anns[1]]


def test_7_crest_present_path_skips_salient_scan_but_shares_context(monkeypatch):
    """
    Test 7: The crest-present branch (`has_crest_or_prompt == True`) must skip
    `_scan_salient_3d_resources` (Stage A runs only when NO crest/prompt was
    found) while still forwarding the shared frame context to the remaining
    sub-scanners. Test 4 only proves forwarding on the crestless path.
    The crest annotation carries the canonical id 13 (`skill_ui_prompt`,
    material `world_gather_crest`), not the legacy dataset id 6.
    """
    labeler = _make_lightweight_labeler(
        crests=[{"bbox": (300, 200, 60, 40), "x": 330, "y": 220, "confidence": 0.92}]
    )
    frame = _create_sample_frame(720, 1280)
    shared_ctx: Dict[str, Any] = {}

    observed: Dict[str, Any] = {}

    def record_salient(*args, **kwargs):
        observed["salient_called"] = True
        return []

    orig_monsters = labeler._scan_3d_monsters
    orig_hud = labeler._scan_hud_and_skills
    orig_path = labeler._scan_path_and_obstacles

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            ctx = kwargs.get("shared_frame_context")
            observed[name] = {
                "same_ctx": ctx is shared_ctx,
                "same_bgr": kwargs.get("frame_bgr") is shared_ctx.get("bgr"),
                "same_hsv": kwargs.get("hsv") is shared_ctx.get("hsv"),
            }
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr(labeler, "_scan_salient_3d_resources", record_salient)
    monkeypatch.setattr(labeler, "_scan_3d_monsters", wrap("monsters", orig_monsters))
    monkeypatch.setattr(labeler, "_scan_hud_and_skills", wrap("hud", orig_hud))
    monkeypatch.setattr(labeler, "_scan_path_and_obstacles", wrap("path", orig_path))

    # fast_mode=False so the vision crest detection actually runs.
    anns = labeler.auto_label_frame(frame, fast_mode=False, shared_frame_context=shared_ctx)

    # A class-13 crest annotation with the vision source must be present...
    # (canonical id: world_gather_crest -> 13 skill_ui_prompt,
    # dataset_factory/class_taxonomy.py; the legacy dataset id 6 is
    # herb_edana in the canonical taxonomy and must never be expected here.)
    crest_anns = [a for a in anns if a["class_id"] == 13]
    assert len(crest_anns) == 1
    assert crest_anns[0]["source"] == "vision_3d_crest"
    assert crest_anns[0]["material_name"] == "world_gather_crest"
    # ...and the crestless-only salient scan must NOT have run.
    assert observed.get("salient_called") is not True
    # ...while the remaining sub-scanners still received the shared context.
    for scanner_name in ("monsters", "hud", "path"):
        assert scanner_name in observed, f"{scanner_name} was not called"
        for check_name, passed in observed[scanner_name].items():
            assert passed is True, f"{scanner_name} failed check '{check_name}'"
