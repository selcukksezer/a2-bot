# -*- coding: utf-8 -*-
"""tests/test_w27_three_tier_box_colors.py
==========================================
Worker W27 unit tests verifying the 3-tier bounding box confidence color coding:
1. Tier 1 (Emin, >= 85% trusted or hand_label/human_approved): Green (#22c55e)
2. Tier 2 (Şüpheli, 70% - 84%): Orange (#fb923c)
3. Tier 3 (Düşük Güven, < 70%): Red (#ef4444)
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QApplication

from dataset_factory.confidence_tier_coloring import (
    CONFIDENT_COLOR_HEX,
    CONFIDENT_TEXT_COLOR_HEX,
    LOW_CONFIDENCE_COLOR_HEX,
    LOW_CONFIDENCE_TEXT_COLOR_HEX,
    SUSPICIOUS_COLOR_HEX,
    SUSPICIOUS_TEXT_COLOR_HEX,
    TIER_CONFIDENT,
    TIER_LOW_CONFIDENCE,
    TIER_SUSPICIOUS,
    ConfidenceTierColoring,
    classify_confidence_tier,
    get_box_visual_style,
    hex_to_bgr,
    hex_to_rgb,
)
from ui.yolo_factory_dialog import YOLOPreviewCanvas

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)


@pytest.mark.parametrize(
    "conf_val,expected_pct",
    [(0.85, 85), (0.92, 92), (0.99, 99), (88.0, 88), (100.0, 100)],
)
def test_tier1_confident_boxes_classified_as_green_22c55e(conf_val: float, expected_pct: int) -> None:
    """Verify >= 85% trusted boxes are classified as Confident ('confident') with Green (#22c55e).

    Parametrized (was a for-loop inside one test): each confidence value is
    now its own test case, so a failure at one value no longer masks the
    remaining values and the failing value is named in the test id.
    """
    ann = {
        "class_name": "herb_aria",
        "conf": conf_val,
        "trusted": True,
    }
    style = get_box_visual_style(ann)
    assert classify_confidence_tier(ann) == TIER_CONFIDENT
    assert style["tier"] == TIER_CONFIDENT
    assert style["is_confident"] is True
    assert style["is_suspicious"] is False
    assert style["is_low_confidence"] is False
    assert style["box_color"].lower() == "#22c55e"
    assert style["line_color"].lower() == CONFIDENT_COLOR_HEX
    assert style["badge_bg_color"].lower() == "#22c55e"
    assert style["text_color"].upper() == CONFIDENT_TEXT_COLOR_HEX
    assert style["line_style"] == "solid"
    assert style["pen_style"] == "SolidLine"
    assert style["line_width"] == pytest.approx(2.2)
    assert style["badge_text"] == f"✓ herb_aria %{expected_pct}"
    assert style["status_text"] == f"Emin · %{expected_pct} güven"
    assert style["color_rgb"] == (34, 197, 94)
    assert style["color_bgr"] == (94, 197, 34)


@pytest.mark.parametrize(
    "conf_val,expected_pct",
    [(0.70, 70), (0.75, 75), (0.80, 80), (0.84, 84), (77.0, 77), (84.0, 84)],
)
def test_tier2_suspicious_70_to_84_classified_as_orange_fb923c(conf_val: float, expected_pct: int) -> None:
    """Verify 70%-84% boxes are classified as Suspicious ('suspicious') with Orange (#fb923c).

    Parametrized (was a for-loop inside one test): same failure-isolation
    reason as tier 1. Note the "confidence" key (not "conf") is kept
    deliberately — both keys must resolve to the same tier.
    """
    ann = {
        "class_name": "ore_orichalcum",
        "confidence": conf_val,
        "trusted": False,
    }
    style = get_box_visual_style(ann)
    assert classify_confidence_tier(ann) == TIER_SUSPICIOUS
    assert style["tier"] == TIER_SUSPICIOUS
    assert style["is_confident"] is False
    assert style["is_suspicious"] is True
    assert style["is_low_confidence"] is False
    assert style["box_color"].lower() == "#fb923c"
    assert style["line_color"].lower() == SUSPICIOUS_COLOR_HEX
    assert style["badge_bg_color"].lower() == "#fb923c"
    assert style["text_color"].lower() == SUSPICIOUS_TEXT_COLOR_HEX
    assert style["line_style"] == "dashed"
    assert style["pen_style"] == "DashLine"
    assert style["line_width"] == pytest.approx(1.8)
    assert style["badge_text"] == f"🟠 şüpheli ore_orichalcum %{expected_pct}"
    assert style["status_text"] == f"Şüpheli · %{expected_pct} güven"
    assert style["color_rgb"] == (251, 146, 60)
    assert style["color_bgr"] == (60, 146, 251)


@pytest.mark.parametrize(
    "conf_val,expected_pct",
    [(0.69, 69), (0.55, 55), (0.42, 42), (0.15, 15), (69.0, 69), (12.0, 12)],
)
def test_tier3_low_confidence_below_70_classified_as_red_ef4444(conf_val: float, expected_pct: int) -> None:
    """Verify < 70% boxes are classified as Low Confidence ('low_confidence') with Red (#ef4444).

    Parametrized (was a for-loop inside one test): same failure-isolation
    reason as tier 1.
    """
    ann = {
        "class_name": "herb_mela",
        "conf": conf_val,
        "trusted": False,
    }
    style = get_box_visual_style(ann)
    assert classify_confidence_tier(ann) == TIER_LOW_CONFIDENCE
    assert style["tier"] == TIER_LOW_CONFIDENCE
    assert style["is_confident"] is False
    assert style["is_suspicious"] is False
    assert style["is_low_confidence"] is True
    assert style["box_color"].lower() == "#ef4444"
    assert style["line_color"].lower() == LOW_CONFIDENCE_COLOR_HEX
    assert style["badge_bg_color"].lower() == "#ef4444"
    assert style["text_color"].lower() == LOW_CONFIDENCE_TEXT_COLOR_HEX
    assert style["line_style"] == "dashed"
    assert style["pen_style"] == "DashLine"
    assert style["line_width"] == pytest.approx(1.8)
    assert style["badge_text"] == f"şüpheli herb_mela %{expected_pct}"
    assert style["status_text"] == f"Düşük Güven (Şüpheli) · %{expected_pct} güven"
    assert style["color_rgb"] == (239, 68, 68)
    assert style["color_bgr"] == (68, 68, 239)


def test_exact_boundary_transitions_69_70_84_85() -> None:
    """Verify exact threshold boundaries at 69% -> 70% and 84% -> 85%."""
    box_69 = {"class_name": "minimap_sprout", "conf": 0.69, "trusted": True}
    box_70 = {"class_name": "minimap_sprout", "conf": 0.70, "trusted": True}
    box_84 = {"class_name": "minimap_sprout", "conf": 0.84, "trusted": True}
    box_85 = {"class_name": "minimap_sprout", "conf": 0.85, "trusted": True}

    style_69 = ConfidenceTierColoring.get_box_visual_style(box_69)
    style_70 = ConfidenceTierColoring.get_box_visual_style(box_70)
    style_84 = ConfidenceTierColoring.get_box_visual_style(box_84)
    style_85 = ConfidenceTierColoring.get_box_visual_style(box_85)

    assert style_69["box_color"] == "#ef4444"
    assert style_69["tier"] == TIER_LOW_CONFIDENCE

    assert style_70["box_color"] == "#fb923c"
    assert style_70["tier"] == TIER_SUSPICIOUS

    assert style_84["box_color"] == "#fb923c"
    assert style_84["tier"] == TIER_SUSPICIOUS

    assert style_85["box_color"] == "#22c55e"
    assert style_85["tier"] == TIER_CONFIDENT


def test_human_approved_and_hand_label_override_to_green_22c55e() -> None:
    """Verify hand_label and human_approved promote boxes to Green (#22c55e) while untrusted >=85% stays Orange (#fb923c)."""
    untrusted_high = {
        "class_name": "herb_aria",
        "conf": 0.91,
        "trusted": False,
        "hand_label": False,
        "human_approved": False,
    }
    style_untrusted = get_box_visual_style(untrusted_high)
    assert style_untrusted["tier"] == TIER_SUSPICIOUS
    assert style_untrusted["box_color"] == "#fb923c"

    human_approved_box = {
        "class_name": "herb_aria",
        "conf": 0.62,
        "trusted": False,
        "human_approved": True,
    }
    style_approved = get_box_visual_style(human_approved_box)
    assert style_approved["tier"] == TIER_CONFIDENT
    assert style_approved["box_color"] == "#22c55e"

    hand_labeled_box = {
        "class_name": "hp_potion",
        "conf": 1.0,
        "trusted": False,
        "hand_label": True,
    }
    style_hand = get_box_visual_style(hand_labeled_box)
    assert style_hand["tier"] == TIER_CONFIDENT
    assert style_hand["box_color"] == "#22c55e"
    assert hex_to_rgb("#22c55e") == (34, 197, 94)
    assert hex_to_bgr("#ef4444") == (68, 68, 239)


def test_yolo_preview_canvas_paints_three_tier_colors() -> None:
    """Verify YOLOPreviewCanvas renders Green (#22c55e), Orange (#fb923c), and Red (#ef4444) pens for the 3 tiers."""
    import ui.yolo_factory_dialog as yfd

    canvas = YOLOPreviewCanvas()
    canvas.resize(640, 360)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    annotations = [
        {"class_name": "herb_aria", "conf": 0.93, "trusted": True, "cx": 0.2, "cy": 0.5, "w": 0.12, "h": 0.16},
        {"class_name": "ore_orichalcum", "conf": 0.76, "trusted": False, "cx": 0.5, "cy": 0.5, "w": 0.12, "h": 0.16},
        {"class_name": "herb_mela", "conf": 0.48, "trusted": False, "cx": 0.8, "cy": 0.5, "w": 0.12, "h": 0.16},
    ]
    assert canvas.set_live_frame(dummy_frame, "w27_test.png", annotations) is True

    recorded_pens: list[tuple[str, float]] = []
    orig_qpen = yfd.QPen

    def spy_qpen(*args: object, **kwargs: object) -> QPen:
        pen = orig_qpen(*args, **kwargs)
        recorded_pens.append((pen.color().name().lower(), round(float(pen.widthF()), 1)))
        return pen

    yfd.QPen = spy_qpen  # type: ignore[assignment]
    try:
        pixmap = canvas.grab()
    finally:
        yfd.QPen = orig_qpen  # type: ignore[assignment]

    assert not pixmap.isNull()
    recorded_colors = [c for c, _w in recorded_pens]
    assert "#22c55e" in recorded_colors
    assert "#fb923c" in recorded_colors
    assert "#ef4444" in recorded_colors
    assert ("#22c55e", 2.2) in recorded_pens
    assert ("#fb923c", 1.8) in recorded_pens
    assert ("#ef4444", 1.8) in recorded_pens
    assert len(canvas._close_hit_rects) == 3

