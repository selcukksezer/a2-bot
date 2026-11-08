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


def _make_lightweight_labeler() -> YOLOAutoLabeler:
    """Create a YOLOAutoLabeler with lightweight mock sub-engines for fast deterministic tests."""
    vision_mock = MagicMock()
    vision_mock.detect_3d_gather_crests.return_value = []
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
    """
    labeler = _make_lightweight_labeler()
    frame = _create_sample_frame(720, 1280)
    shared_ctx: Dict[str, Any] = {}

    real_cvt_color = cv2.cvtColor
    real_canny = cv2.Canny

    full_frame_cvt_calls: List[int] = []
    canny_calls: List[ Any ] = []

    def spy_cvt_color(src, code, *args, **kwargs):
        if isinstance(src, np.ndarray) and src.shape[:2] == (720, 1280):
            full_frame_cvt_calls.append(code)
        return real_cvt_color(src, code, *args, **kwargs)

    def spy_canny(image, threshold1, threshold2, *args, **kwargs):
        canny_calls.append(image.shape)
        return real_canny(image, threshold1, threshold2, *args, **kwargs)

    monkeypatch.setattr(auto_labeler_mod.cv2, "cvtColor", spy_cvt_color)
    monkeypatch.setattr(auto_labeler_mod.cv2, "Canny", spy_canny)

    # First call: should compute HSV (1), GRAY (1), and Canny (1)
    labeler.auto_label_frame(
        frame,
        fast_mode=True,
        source_path="learn_skills_test_01.png",
        shared_frame_context=shared_ctx,
    )
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2HSV) == 1
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2GRAY) == 1
    assert len(canny_calls) == 1

    # Second call with same shared_frame_context: 0 additional cvtColor or Canny calls
    labeler.auto_label_frame(
        frame,
        fast_mode=True,
        source_path="learn_skills_test_02.png",
        shared_frame_context=shared_ctx,
    )
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2HSV) == 1
    assert full_frame_cvt_calls.count(cv2.COLOR_BGR2GRAY) == 1
    assert len(canny_calls) == 1


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

    def wrap_monsters(*args, **kwargs):
        ctx = kwargs.get("shared_frame_context")
        observed_contexts["monsters"] = {
            "same_ctx": ctx is shared_ctx,
            "same_bgr": kwargs.get("frame_bgr") is shared_ctx.get("bgr"),
            "same_hsv": kwargs.get("hsv") is shared_ctx.get("hsv"),
            "same_gray": kwargs.get("gray") is shared_ctx.get("gray"),
            "has_edges": isinstance(ctx.get("edges"), np.ndarray) if isinstance(ctx, dict) else False,
        }
        return orig_monsters(*args, **kwargs)

    def wrap_salient(*args, **kwargs):
        ctx = kwargs.get("shared_frame_context")
        observed_contexts["salient"] = {
            "same_ctx": ctx is shared_ctx,
            "same_bgr": kwargs.get("frame_bgr") is shared_ctx.get("bgr"),
            "same_hsv": kwargs.get("hsv") is shared_ctx.get("hsv"),
            "has_edges": isinstance(ctx.get("edges"), np.ndarray) if isinstance(ctx, dict) else False,
        }
        return orig_salient(*args, **kwargs)

    def wrap_hud(*args, **kwargs):
        ctx = kwargs.get("shared_frame_context")
        observed_contexts["hud"] = {
            "same_ctx": ctx is shared_ctx,
            "same_bgr": kwargs.get("frame_bgr") is shared_ctx.get("bgr"),
            "same_hsv": kwargs.get("hsv") is shared_ctx.get("hsv"),
            "same_gray": kwargs.get("gray") is shared_ctx.get("gray"),
            "has_edges": isinstance(ctx.get("edges"), np.ndarray) if isinstance(ctx, dict) else False,
        }
        return orig_hud(*args, **kwargs)

    def wrap_path(*args, **kwargs):
        ctx = kwargs.get("shared_frame_context")
        observed_contexts["path"] = {
            "same_ctx": ctx is shared_ctx,
            "same_bgr": kwargs.get("frame_bgr") is shared_ctx.get("bgr"),
            "same_hsv": kwargs.get("hsv") is shared_ctx.get("hsv"),
            "same_gray": kwargs.get("gray") is shared_ctx.get("gray"),
            "has_edges": isinstance(ctx.get("edges"), np.ndarray) if isinstance(ctx, dict) else False,
        }
        return orig_path(*args, **kwargs)

    monkeypatch.setattr(labeler, "_scan_3d_monsters", wrap_monsters)
    monkeypatch.setattr(labeler, "_scan_salient_3d_resources", wrap_salient)
    monkeypatch.setattr(labeler, "_scan_hud_and_skills", wrap_hud)
    monkeypatch.setattr(labeler, "_scan_path_and_obstacles", wrap_path)

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
