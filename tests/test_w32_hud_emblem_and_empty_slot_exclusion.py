# -*- coding: utf-8 -*-
"""
Unit tests for W32: Bottom HUD Center Level '45' Emblem (`0.44 <= norm_cx <= 0.56`)
and Empty Quickbar Slot (`F8` low saturation / low contrast) Exclusion.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.scroll_potion_classifier import (
    CENTER_EMBLEM_X_MAX,
    CENTER_EMBLEM_X_MIN,
    CENTER_EMBLEM_Y_MAX,
    CENTER_EMBLEM_Y_MIN,
    HUDConsumableClassifier,
)
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler


def _make_level_45_emblem_crop(size: int = 38) -> np.ndarray:
    """Create a high-contrast, high-saturation BGR crop resembling the center '45' level emblem."""
    crop = np.full((size, size, 3), (35, 95, 210), dtype=np.uint8)
    cv2.rectangle(crop, (2, 2), (size - 3, size - 3), (80, 215, 255), 2)
    cv2.putText(
        crop,
        "45",
        (5, size - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 255),
        2,
    )
    return crop


def _make_empty_f8_slot_crop(size: int = 36) -> np.ndarray:
    """
    Create a low-saturation, low-contrast BGR crop representing an empty 'F8' HUD slot:
    dark flat interior (BGR 30,30,30) with a subtle 1px frame bevel (BGR 72,72,72)
    so Canny detects the slot contour while saturation (< 28) and value std (< 16) remain low.
    """
    crop = np.full((size, size, 3), (30, 30, 30), dtype=np.uint8)
    cv2.rectangle(crop, (1, 1), (size - 2, size - 2), (72, 72, 72), 1)
    return crop


def _build_synthetic_hud_frame_with_emblem_and_empty_f8(
    height: int = 1080,
    width: int = 1920,
) -> np.ndarray:
    """
    Build a 1920x1080 synthetic frame containing:
    1. Valid HP potion slot in upper consumable row (`norm_cx ~ 0.38`, `norm_cy ~ 0.896`)
    2. Center '45' level emblem (`norm_cx = 0.50`, `norm_cy ~ 0.896` inside `0.44..0.56`)
    3. Empty 'F8' slot in upper consumable row (`norm_cx ~ 0.62`, `norm_cy ~ 0.896`)
    4. Valid lower combat skill slot (`norm_cx ~ 0.68`, `norm_cy ~ 0.955`)
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    slot_sz = 36
    upper_y = int(round(height * 0.896)) - slot_sz // 2  # norm_cy ~ 0.896 (in 0.865..0.938)
    lower_y = int(round(height * 0.955)) - slot_sz // 2  # norm_cy ~ 0.955 (>= 0.939)

    # 1. Valid red HP potion slot at norm_cx ~ 0.38
    hp_x = int(round(width * 0.38)) - slot_sz // 2
    cv2.rectangle(frame, (hp_x, upper_y), (hp_x + slot_sz, upper_y + slot_sz), (180, 180, 180), 2)
    cv2.circle(
        frame,
        (hp_x + slot_sz // 2, upper_y + slot_sz // 2),
        12,
        (20, 25, 230),  # Bright red in BGR -> H~0, S>200, V>200
        -1,
    )

    # 2. Center '45' level emblem at norm_cx = 0.50 (inside 0.44 <= norm_cx <= 0.56)
    emblem_x = int(round(width * 0.50)) - slot_sz // 2
    frame[upper_y : upper_y + slot_sz, emblem_x : emblem_x + slot_sz] = _make_level_45_emblem_crop(slot_sz)

    # 3. Empty 'F8' slot at norm_cx ~ 0.62 (low saturation & low contrast)
    f8_x = int(round(width * 0.62)) - slot_sz // 2
    frame[upper_y : upper_y + slot_sz, f8_x : f8_x + slot_sz] = _make_empty_f8_slot_crop(slot_sz)

    # 4. Valid lower combat skill slot at norm_cx ~ 0.68, norm_cy ~ 0.955
    skill_x = int(round(width * 0.68)) - slot_sz // 2
    cv2.rectangle(frame, (skill_x, lower_y), (skill_x + slot_sz, lower_y + slot_sz), (200, 200, 200), 2)
    cv2.circle(frame, (skill_x + slot_sz // 2, lower_y + slot_sz // 2), 11, (240, 150, 40), -1)
    cv2.line(frame, (skill_x + 7, lower_y + 7), (skill_x + slot_sz - 7, lower_y + slot_sz - 7), (255, 255, 255), 2)

    return frame


def test_w32_01_center_level_45_emblem_spatial_exclusion() -> None:
    """
    Test 1: Verify that any slot candidate inside the center character level '45' emblem band
    (`0.44 <= norm_cx <= 0.56`, `0.85 <= norm_cy <= 0.95`) is excluded (`should_label=False`,
    `class_id=None`, `category='center_emblem'`, `is_center_emblem=True`), even when its
    visual appearance has strong edges and potion/scroll colors.
    """
    classifier = HUDConsumableClassifier()
    emblem_crop = _make_level_45_emblem_crop(38)

    assert CENTER_EMBLEM_X_MIN == 0.44
    assert CENTER_EMBLEM_X_MAX == 0.56
    assert CENTER_EMBLEM_Y_MIN == 0.85
    assert CENTER_EMBLEM_Y_MAX == 0.95

    for norm_cx in (0.44, 0.47, 0.50, 0.53, 0.56):
        for norm_cy in (0.86, 0.90, 0.94):
            assert classifier.is_center_level_emblem(norm_cx, norm_cy) is True
            res = classifier.classify_slot(emblem_crop, norm_cx=norm_cx, norm_cy=norm_cy)
            assert res.is_center_emblem is True
            assert res.category == "center_emblem"
            assert res.should_label is False
            assert res.class_id is None
            assert res.canonical_class_id is None

    # Coordinates outside 0.44..0.56 X are not treated as the center level emblem
    assert classifier.is_center_level_emblem(0.439, 0.90) is False
    assert classifier.is_center_level_emblem(0.561, 0.90) is False

    # Y boundaries of the emblem band (0.85..0.95, inclusive — D89 coverage):
    # previously only in-band Y (0.86/0.90/0.94) was exercised, so an
    # off-by-one on the Y edge would have passed silently.
    assert classifier.is_center_level_emblem(0.50, 0.85) is True
    assert classifier.is_center_level_emblem(0.50, 0.95) is True
    assert classifier.is_center_level_emblem(0.50, 0.849) is False
    assert classifier.is_center_level_emblem(0.50, 0.951) is False

    # An emblem-looking crop just above the band is NOT the center emblem.
    res_above = classifier.classify_slot(emblem_crop, norm_cx=0.50, norm_cy=0.849)
    assert res_above.is_center_emblem is False
    assert res_above.category != "center_emblem"


def test_w32_02_empty_f8_slot_low_saturation_and_contrast_exclusion() -> None:
    """
    Test 2: Verify that an empty 'F8' quickbar slot with low saturation (`sat_mean < 28.0`)
    and low value variance (`val_std < 16.0`) is identified as empty (`is_empty=True`) and
    excluded from labeling (`should_label=False`, `class_id=None`) in both HUD rows.
    """
    classifier = HUDConsumableClassifier()
    empty_f8_bgr = _make_empty_f8_slot_crop(36)
    empty_f8_hsv = cv2.cvtColor(empty_f8_bgr, cv2.COLOR_BGR2HSV)

    is_empty, metrics = classifier.is_empty_slot(empty_f8_hsv)
    assert is_empty is True
    assert metrics["sat_mean"] < 28.0
    assert metrics["val_std"] < 16.0

    # Upper consumable row (e.g. F8 slot at norm_cx=0.62, norm_cy=0.895)
    upper_res = classifier.classify_slot(empty_f8_bgr, norm_cx=0.62, norm_cy=0.895)
    assert upper_res.is_empty is True
    assert upper_res.category == "empty_slot"
    assert upper_res.should_label is False
    assert upper_res.class_id is None
    assert upper_res.row_type == "upper_consumable_row"

    # Lower combat skill row (norm_cx=0.62, norm_cy=0.955)
    lower_res = classifier.classify_slot(empty_f8_bgr, norm_cx=0.62, norm_cy=0.955)
    assert lower_res.is_empty is True
    assert lower_res.category == "empty_slot"
    assert lower_res.should_label is False
    assert lower_res.class_id is None
    assert lower_res.row_type == "lower_skill_row"


def test_w32_03_empty_f8_slot_with_outer_bevel_and_hotkey_text_still_excluded() -> None:
    """
    Test 3: Verify that an empty 'F8' slot whose outer 15% border has a bright frame bevel
    leaving a dark, low-saturation inner 70% core is still excluded as `empty_slot`.
    """
    classifier = HUDConsumableClassifier()
    slot_bgr = np.full((40, 40, 3), (26, 26, 28), dtype=np.uint8)
    # Bright outer frame bevel confined to outer border (trimmed by 15% inner ROI guard)
    cv2.rectangle(slot_bgr, (0, 0), (39, 39), (165, 165, 170), 2)

    slot_hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    slot_edges = cv2.Canny(cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY), 50, 150)

    is_empty, metrics = classifier.is_empty_slot(slot_hsv, slot_edges=slot_edges)
    assert is_empty is True
    assert metrics["sat_mean"] < 28.0
    assert metrics["val_std"] < 16.0

    res = classifier.classify_slot(
        slot_bgr,
        norm_cx=0.64,
        norm_cy=0.90,
        slot_edges=slot_edges,
    )
    assert res.is_empty is True
    assert res.should_label is False
    assert res.class_id is None
    assert res.category == "empty_slot"


def test_w32_04_hud_consumable_classifier_scan_excludes_emblem_and_empty_f8() -> None:
    """
    Test 4: Verify `HUDConsumableClassifier.scan_hud_slots` on a full 1920x1080 frame
    excludes both the center '45' emblem (`0.44 <= cx <= 0.56`) and the empty 'F8' slot
    (`cx ~ 0.62`), while emitting valid `hp_potion` and `skill_slot` annotations.
    """
    classifier = HUDConsumableClassifier()
    frame = _build_synthetic_hud_frame_with_emblem_and_empty_f8()

    detections = classifier.scan_hud_slots(frame)
    assert len(detections) >= 2

    # 1. No detection inside center level '45' emblem X band (0.44..0.56)
    for det in detections:
        assert not (0.44 <= float(det["cx"]) <= 0.56), (
            f"Center '45' emblem should never be labeled, got: {det}"
        )

    # 2. No detection at the empty 'F8' slot coordinate (norm_cx ~ 0.62, norm_cy ~ 0.896)
    for det in detections:
        assert not (abs(float(det["cx"]) - 0.62) < 0.025 and abs(float(det["cy"]) - 0.896) < 0.025), (
            f"Empty F8 slot should never be labeled, got: {det}"
        )

    # 3. Valid HP potion and skill slot are detected
    categories = {det["class_name"] for det in detections}
    assert "hp_potion" in categories
    assert "skill_slot" in categories


def test_w32_05_yolo_auto_labeler_scan_hud_and_skills_excludes_emblem_and_empty_f8() -> None:
    """
    Test 5: Verify `YOLOAutoLabeler._scan_hud_and_skills` on a `learn_skills_*` stem
    excludes the center '45' level emblem (`0.44 <= norm_cx <= 0.56`) and the empty 'F8'
    slot (`sat_mean < 28.0` and `val_std < 16.0`), while labeling valid HUD slots.
    """
    labeler = YOLOAutoLabeler()
    frame = _build_synthetic_hud_frame_with_emblem_and_empty_f8()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    hud_anns = labeler._scan_hud_and_skills(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="learn_skills_w32_emblem_and_empty_slot_check",
    )

    assert len(hud_anns) >= 2

    # Ensure center '45' emblem (0.44 <= cx <= 0.56, 0.85 <= cy <= 0.95) is excluded
    for ann in hud_anns:
        cx, cy = float(ann["cx"]), float(ann["cy"])
        assert not (0.44 <= cx <= 0.56 and 0.85 <= cy <= 0.95), (
            f"YOLOAutoLabeler must exclude center '45' emblem, got: {ann}"
        )

    # Ensure empty 'F8' slot (cx ~ 0.62, cy ~ 0.896) is excluded
    for ann in hud_anns:
        cx, cy = float(ann["cx"]), float(ann["cy"])
        assert not (abs(cx - 0.62) < 0.025 and abs(cy - 0.896) < 0.025), (
            f"YOLOAutoLabeler must exclude empty F8 slot, got: {ann}"
        )

    # Ensure valid HP potion and combat skill slot are present. Assert by
    # class NAME, not by hard-coded class id (D89/F-2065): the labeler emits
    # the canonical ids 22 (hp_potion) / 20 (skill_slot_ready), which do NOT
    # match the legacy HUD ids 18/21 (dataset_factory/class_taxonomy.py:
    # 18 = target_hp_bar, 20 = skill_slot_ready, 21 = skill_slot_cooldown,
    # 22 = hp_potion). Pinning 18/21 here would lock in the legacy scheme;
    # the id alignment belongs to the labeler (D43).
    # NOTE: the name check reads `material_name`, not `class_name` — the
    # canonical taxonomy name for id 20 is "skill_slot_ready"
    # (YOLO_CLASSES[20]), while the slot's material_name is "skill_slot".
    # `class_name` would therefore never contain "skill_slot" even though
    # the skill slot IS emitted (with canonical id 20).
    emitted_names = {str(ann["material_name"]) for ann in hud_anns}
    assert "hp_potion" in emitted_names, f"Expected hp_potion in {hud_anns}"
    assert "skill_slot" in emitted_names, f"Expected skill_slot in {hud_anns}"
