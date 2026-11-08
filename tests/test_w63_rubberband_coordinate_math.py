# -*- coding: utf-8 -*-
"""
Unit tests for W63: RubberBandBoxConverter coordinate math.
Verifies reverse-direction drags (bottom-right to top-left), off-screen/out-of-bounds
clipping, and the <8px click-vs-drag threshold distinction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.rubberband_box_converter import (
    DEFAULT_MIN_DRAG_PX,
    RubberBandBoxConverter,
)


def test_reverse_drag_bottom_right_to_top_left_matches_forward() -> None:
    """
    Dragging from bottom-right to top-left (and other reverse diagonals) must
    produce the exact same normalized YOLO (cx, cy, w, h) box as top-left to bottom-right.
    """
    converter = RubberBandBoxConverter()
    # Canvas viewport offset at (100, 50) with size 800x600
    target_rect = (100.0, 50.0, 800.0, 600.0)

    # Top-left (260, 170) -> Bottom-right (500, 350)
    # Relative to rect: x in [160, 400] (cx=280/800=0.35, w=240/800=0.30)
    #                   y in [120, 300] (cy=210/600=0.35, h=180/600=0.30)
    forward = converter.convert_result(260.0, 170.0, 500.0, 350.0, target_rect=target_rect)
    reverse_br_tl = converter.convert_result(500.0, 350.0, 260.0, 170.0, target_rect=target_rect)
    reverse_bl_tr = converter.convert_result(260.0, 350.0, 500.0, 170.0, target_rect=target_rect)
    reverse_tr_bl = converter.convert_result(500.0, 170.0, 260.0, 350.0, target_rect=target_rect)

    assert forward.valid is True
    assert reverse_br_tl.valid is True
    assert reverse_bl_tr.valid is True
    assert reverse_tr_bl.valid is True

    expected_yolo = (0.35, 0.35, 0.30, 0.30)
    for res in (forward, reverse_br_tl, reverse_bl_tr, reverse_tr_bl):
        assert res.cx == pytest.approx(expected_yolo[0], abs=1e-6)
        assert res.cy == pytest.approx(expected_yolo[1], abs=1e-6)
        assert res.w == pytest.approx(expected_yolo[2], abs=1e-6)
        assert res.h == pytest.approx(expected_yolo[3], abs=1e-6)
        assert res.clipped_pixel_rect == pytest.approx((260.0, 170.0, 240.0, 180.0), abs=1e-6)


def test_offscreen_overflow_clamped_to_target_rect_bounds() -> None:
    """
    Drags that start or end outside the target_rect (negative coordinates or beyond
    width/height, including reverse drags) must be clipped to target_rect and clamped to [0.0, 1.0].
    """
    converter = RubberBandBoxConverter()
    # Target rect: left=50, top=40, width=200, height=100 -> right=250, bottom=140
    target_rect = {"left": 50.0, "top": 40.0, "width": 200.0, "height": 100.0}

    # Reverse drag starting way past bottom-right (350, 200) and ending inside at (150, 90)
    # Clipped pixel rect should be x in [150, 250], y in [90, 140] -> width=100, height=50
    # Normalized: nx1=(150-50)/200=0.5, nx2=1.0 -> cx=0.75, w=0.50
    #             ny1=(90-40)/100=0.5,  ny2=1.0 -> cy=0.75, h=0.50
    res = converter.convert_result(350.0, 200.0, 150.0, 90.0, target_rect=target_rect)
    assert res.valid is True
    assert res.reason == "ok"
    assert (res.x1_px, res.y1_px, res.x2_px, res.y2_px) == pytest.approx((150.0, 90.0, 250.0, 140.0))
    assert res.yolo_box == pytest.approx((0.75, 0.75, 0.50, 0.50), abs=1e-6)

    # Drag spanning across the entire viewport from negative off-screen (-20, -10) to (500, 400)
    full_span = converter.convert_result(-20.0, -10.0, 500.0, 400.0, target_rect=target_rect)
    assert full_span.valid is True
    assert full_span.yolo_box == pytest.approx((0.5, 0.5, 1.0, 1.0), abs=1e-6)
    assert all(0.0 <= v <= 1.0 for v in full_span.yolo_box)


def test_completely_offscreen_or_clipped_sliver_rejected() -> None:
    """
    Drags that occur entirely outside target_rect or whose clipped intersection with
    target_rect is smaller than 8px must be rejected.
    """
    converter = RubberBandBoxConverter()
    target_rect = (100.0, 100.0, 400.0, 300.0)  # x in [100, 500], y in [100, 400]

    # Completely to the left/top of target_rect (large raw drag, zero intersection)
    res_outside = converter.convert_result(10.0, 10.0, 80.0, 80.0, target_rect=target_rect)
    assert res_outside.valid is False
    assert res_outside.reason == "outside_target_rect"
    assert converter.convert(10.0, 10.0, 80.0, 80.0, target_rect=target_rect) is None

    # Starts outside at x=50 and barely enters target_rect up to x=105 (clipped width = 5px < 8px)
    res_clipped_short = converter.convert_result(50.0, 120.0, 105.0, 200.0, target_rect=target_rect)
    assert res_clipped_short.valid is False
    assert res_clipped_short.reason == "clipped_drag_too_short"
    assert res_clipped_short.clipped_w_px == pytest.approx(5.0)


def test_click_vs_drag_under_8px_threshold_rejected() -> None:
    """
    Accidental clicks (0px), micro-drags (<8px), and single-axis slivers where one
    axis is <8px must be rejected as 'drag_too_short'.
    """
    converter = RubberBandBoxConverter()
    assert converter.min_drag_px == pytest.approx(DEFAULT_MIN_DRAG_PX)
    target_rect = (0.0, 0.0, 1000.0, 1000.0)

    # Zero-distance click
    res_click = converter.convert_result(200.0, 200.0, 200.0, 200.0, target_rect=target_rect)
    assert res_click.valid is False
    assert res_click.reason == "drag_too_short"

    # 7.9px drag on both axes (even though Euclidean distance is ~11.17px, axis span < 8px)
    res_micro = converter.convert_result(200.0, 200.0, 207.9, 207.9, target_rect=target_rect)
    assert res_micro.valid is False
    assert res_micro.reason == "drag_too_short"
    assert converter.convert(200.0, 200.0, 207.9, 207.9, target_rect=target_rect) is None

    # Reverse micro-drag from bottom-right to top-left (-5px, -6px)
    res_rev_micro = converter.convert_result(200.0, 200.0, 195.0, 194.0, target_rect=target_rect)
    assert res_rev_micro.valid is False
    assert res_rev_micro.reason == "drag_too_short"

    # Single-axis sliver: dx=50px (>=8px), dy=4px (<8px)
    res_sliver = converter.convert_result(100.0, 100.0, 150.0, 104.0, target_rect=target_rect)
    assert res_sliver.valid is False
    assert res_sliver.reason == "drag_too_short"


def test_exact_8px_boundary_drag_accepted() -> None:
    """
    A drag of exactly 8.0px in both axes (forward or reverse) must pass the >=8px gate
    and produce valid normalized coordinates.
    """
    converter = RubberBandBoxConverter()
    target_rect = (0.0, 0.0, 800.0, 400.0)

    # Reverse drag of exactly (-8px, -8px): from (408, 208) to (400, 200)
    res = converter.convert_result(408.0, 208.0, 400.0, 200.0, target_rect=target_rect)
    assert res.valid is True
    assert res.reason == "ok"
    assert res.clipped_w_px == pytest.approx(8.0)
    assert res.clipped_h_px == pytest.approx(8.0)
    assert res.w == pytest.approx(8.0 / 800.0, abs=1e-6)
    assert res.h == pytest.approx(8.0 / 400.0, abs=1e-6)
    assert res.cx == pytest.approx(404.0 / 800.0, abs=1e-6)
    assert res.cy == pytest.approx(204.0 / 400.0, abs=1e-6)
