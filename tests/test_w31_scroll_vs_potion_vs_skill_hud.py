# -*- coding: utf-8 -*-
"""
Worker W31 Unit Tests: Bottom HUD Upper Consumable Row (`0.865 <= norm_cy <= 0.938`)
(`buff_scroll`, `hp_potion`, `mp_potion`) vs Lower Skill Row (`norm_cy >= 0.939`)
(`skill_slot`) Separation and Zero `skill_ui_prompt` Mislabeling Guarantee.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from dataset_factory.scroll_potion_classifier import (
    CANONICAL_HUD_CLASS_IDS,
    HUDConsumableClassifier,
    LOWER_SKILL_ROW_Y_MIN,
    UPPER_CONSUMABLE_ROW_Y_MAX,
    UPPER_CONSUMABLE_ROW_Y_MIN,
    YOLO_HUD_CLASS_IDS,
)
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler, YOLO_CLASSES


def _make_hsv_slot(
    h_val: int,
    s_val: int,
    v_val: int,
    size: int = 36,
    with_border_variation: bool = True,
) -> np.ndarray:
    """Create a synthetic slot crop in HSV with realistic inner icon and frame edges."""
    slot_hsv = np.zeros((size, size, 3), dtype=np.uint8)
    slot_hsv[:, :] = (h_val, s_val, v_val)
    if with_border_variation:
        # Add subtle bright/dark bevel so val_std > 16.0 (not treated as flat empty slot)
        slot_hsv[:4, :, 2] = np.clip(int(v_val) + 45, 0, 255)
        slot_hsv[-4:, :, 2] = max(10, int(v_val) - 55)
        slot_hsv[8:14, 8:14, 2] = max(15, int(v_val) - 60)
        slot_hsv[18:26, 18:26, 2] = min(255, int(v_val) + 40)
    return slot_hsv


def test_upper_row_buff_scroll_classification_and_no_skill_mislabel():
    """
    Test 1: Upper row (`0.865 <= norm_cy <= 0.938`) parchment slot must be classified
    as `buff_scroll` and never as `skill_slot` or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    # Parchment HSV: H=20 (in [12, 28]), S=110 (in [30, 175]), V=180 (in [85, 235])
    # Note: keep S < 95 or H=16 so it matches scroll_mask ([12..28, 30..175, 85..235])
    # without triggering yellow MP potion mask ([20..35, 95..255, 95..255]).
    scroll_hsv = _make_hsv_slot(h_val=16, s_val=110, v_val=180)

    for norm_cy in (UPPER_CONSUMABLE_ROW_Y_MIN, 0.900, UPPER_CONSUMABLE_ROW_Y_MAX):
        res = classifier.classify_slot(
            scroll_hsv,
            norm_cx=0.38,
            norm_cy=norm_cy,
            is_hsv=True,
        )
        assert res.should_label is True
        assert res.row_type == "upper_consumable_row"
        assert res.category == "buff_scroll"
        assert res.class_id == YOLO_HUD_CLASS_IDS["buff_scroll"] == 22
        assert res.canonical_class_id == CANONICAL_HUD_CLASS_IDS["buff_scroll"] == 24
        assert res.category not in ("skill_slot", "skill_ui_prompt")
        assert res.class_id not in (13, 21)


def test_upper_row_hp_potion_classification_and_no_skill_mislabel():
    """
    Test 2: Upper row (`0.865 <= norm_cy <= 0.938`) red potion slot must be classified
    as `hp_potion` and never as `skill_slot` or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    # Red HP potion liquid: H=5 (in [0, 12]), S=200 (>= 105), V=200 (>= 75)
    hp_hsv = _make_hsv_slot(h_val=5, s_val=200, v_val=200)

    for norm_cy in (0.865, 0.895, 0.938):
        res = classifier.classify_slot(
            hp_hsv,
            norm_cx=0.34,
            norm_cy=norm_cy,
            is_hsv=True,
        )
        assert res.should_label is True
        assert res.row_type == "upper_consumable_row"
        assert res.category == "hp_potion"
        assert res.class_id == YOLO_HUD_CLASS_IDS["hp_potion"] == 18
        assert res.canonical_class_id == CANONICAL_HUD_CLASS_IDS["hp_potion"] == 22
        assert res.category not in ("skill_slot", "skill_ui_prompt")
        assert res.class_id not in (13, 21)


def test_upper_row_mp_potion_blue_and_yellow_classification():
    """
    Test 3: Upper row (`0.865 <= norm_cy <= 0.938`) blue mana potion and yellow serum
    slots must both be classified as `mp_potion` and never as `skill_slot` or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    # Blue MP potion: H=110 (in [95, 135]), S=180 (>= 85), V=190 (>= 75)
    mp_blue_hsv = _make_hsv_slot(h_val=110, s_val=180, v_val=190)
    # Yellow MP potion: H=30 (in [20, 35], outside scroll H<=28), S=200 (>= 95), V=200 (>= 95)
    mp_yellow_hsv = _make_hsv_slot(h_val=30, s_val=200, v_val=200)

    for slot_hsv in (mp_blue_hsv, mp_yellow_hsv):
        res = classifier.classify_slot(
            slot_hsv,
            norm_cx=0.36,
            norm_cy=0.912,
            is_hsv=True,
        )
        assert res.should_label is True
        assert res.row_type == "upper_consumable_row"
        assert res.category == "mp_potion"
        assert res.class_id == YOLO_HUD_CLASS_IDS["mp_potion"] == 19
        assert res.canonical_class_id == CANONICAL_HUD_CLASS_IDS["mp_potion"] == 23
        assert res.category not in ("skill_slot", "skill_ui_prompt")
        assert res.class_id not in (13, 21)


def test_lower_row_skill_slot_separation_at_boundary():
    """
    Test 4: Verify strict Y-boundary separation between upper consumable row (`norm_cy <= 0.938`)
    and lower combat skill row (`norm_cy >= 0.939`).
    Slots at `norm_cy >= 0.939` with active icon edges must be classified strictly as
    `skill_slot` and never as `buff_scroll`, `hp_potion`, `mp_potion`, or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    assert UPPER_CONSUMABLE_ROW_Y_MIN == pytest.approx(0.865)
    assert UPPER_CONSUMABLE_ROW_Y_MAX == pytest.approx(0.938)
    assert LOWER_SKILL_ROW_Y_MIN == pytest.approx(0.939)

    assert classifier.is_upper_consumable_row(0.865) is True
    assert classifier.is_upper_consumable_row(0.938) is True
    assert classifier.is_upper_consumable_row(0.939) is False
    assert classifier.is_lower_skill_row(0.938) is False
    assert classifier.is_lower_skill_row(0.939) is True

    # Create a high-contrast combat skill icon (even if it has red/gold hues, in lower row it is skill_slot)
    skill_hsv = _make_hsv_slot(h_val=5, s_val=200, v_val=200)
    edges = np.zeros((36, 36), dtype=np.uint8)
    cv2.rectangle(edges, (6, 6), (30, 30), 255, 2)

    # At norm_cy = 0.938 -> upper consumable row -> hp_potion
    upper_res = classifier.classify_slot(
        skill_hsv,
        norm_cx=0.40,
        norm_cy=0.938,
        is_hsv=True,
        slot_edges=edges,
    )
    assert upper_res.row_type == "upper_consumable_row"
    assert upper_res.category == "hp_potion"
    assert upper_res.category != "skill_slot"

    # At norm_cy = 0.939 and 0.960 -> lower skill row -> skill_slot
    for norm_cy in (0.939, 0.960):
        lower_res = classifier.classify_slot(
            skill_hsv,
            norm_cx=0.40,
            norm_cy=norm_cy,
            is_hsv=True,
            slot_edges=edges,
        )
        assert lower_res.should_label is True
        assert lower_res.row_type == "lower_skill_row"
        assert lower_res.category == "skill_slot"
        assert lower_res.class_id == YOLO_HUD_CLASS_IDS["skill_slot"] == 21
        assert lower_res.canonical_class_id == CANONICAL_HUD_CLASS_IDS["skill_slot"] == 20
        assert lower_res.category not in ("buff_scroll", "hp_potion", "mp_potion", "skill_ui_prompt")
        assert lower_res.class_id != 13


def test_center_emblem_and_empty_f8_slot_exclusion_never_skill_ui_prompt():
    """
    Test 5: Center character level emblem (`0.44 <= norm_cx <= 0.56`, `0.85 <= norm_cy <= 0.95`)
    and empty dark quickbar slots (`F8`) must be excluded (`should_label=False`) and never
    labeled as `skill_slot` or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    scroll_hsv = _make_hsv_slot(h_val=16, s_val=110, v_val=180)

    # Center level emblem in upper consumable row Y range
    emblem_res = classifier.classify_slot(
        scroll_hsv,
        norm_cx=0.50,
        norm_cy=0.900,
        is_hsv=True,
    )
    assert emblem_res.is_center_emblem is True
    assert emblem_res.should_label is False
    assert emblem_res.class_id is None
    assert emblem_res.category == "center_emblem"
    assert emblem_res.category not in ("skill_slot", "skill_ui_prompt")

    # Empty dark F8 slot in both upper consumable row and lower skill row
    empty_hsv = np.zeros((36, 36, 3), dtype=np.uint8)
    empty_hsv[:, :] = (0, 10, 35)  # Low saturation (<28) and flat value (std < 16)

    for norm_cy in (0.900, 0.955):
        empty_res = classifier.classify_slot(
            empty_hsv,
            norm_cx=0.62,
            norm_cy=norm_cy,
            is_hsv=True,
        )
        assert empty_res.is_empty is True
        assert empty_res.should_label is False
        assert empty_res.class_id is None
        assert empty_res.category == "empty_slot"
        assert empty_res.category not in ("skill_slot", "skill_ui_prompt")


def test_full_frame_hud_scan_and_yolo_auto_labeler_never_emits_skill_ui_prompt():
    """
    Test 6: Full-frame HUD scan via both `HUDConsumableClassifier.scan_hud_slots` and
    `YOLOAutoLabeler._scan_hud_and_skills` on a 1920x1080 synthetic HUD frame must:
    - Classify upper row (`0.865 <= cy <= 0.938`) slots into `hp_potion` (18), `mp_potion` (19),
      and `buff_scroll` (22), never `skill_slot` (21) or `skill_ui_prompt` (13).
    - Classify lower row (`cy >= 0.939`) slots into `skill_slot` (21), never consumables or
      `skill_ui_prompt` (13).
    """
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def _draw_hud_slot(x1: int, y1: int, size: int, hsv_color: tuple[int, int, int]) -> None:
        bgr_px = cv2.cvtColor(np.uint8([[list(hsv_color)]]), cv2.COLOR_HSV2BGR)[0][0]
        bgr_color = (int(bgr_px[0]), int(bgr_px[1]), int(bgr_px[2]))
        # Outer metallic slot frame for contour detection
        cv2.rectangle(frame, (x1, y1), (x1 + size, y1 + size), (190, 190, 190), 2)
        # Inner colored consumable / skill fill
        cv2.rectangle(frame, (x1 + 4, y1 + 4), (x1 + size - 4, y1 + size - 4), bgr_color, -1)
        # Inner detail for realistic value variance & Canny edges
        cv2.rectangle(frame, (x1 + 10, y1 + 10), (x1 + size - 10, y1 + size - 10), (245, 245, 245), 1)

    # Upper row slots: y1=956, size=36 -> cy = 974 / 1080 = 0.90185 (in [0.865, 0.938])
    _draw_hud_slot(660, 956, 36, (5, 210, 210))    # Red HP Potion
    _draw_hud_slot(710, 956, 36, (110, 200, 210))  # Blue MP Potion
    _draw_hud_slot(760, 956, 36, (16, 120, 195))   # Parchment Buff Scroll

    # Lower row slot: y1=1004, size=36 -> cy = 1022 / 1080 = 0.94630 (>= 0.939)
    _draw_hud_slot(810, 1004, 36, (140, 180, 210))  # Combat Skill Slot

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Verify HUDConsumableClassifier.scan_hud_slots
    classifier = HUDConsumableClassifier()
    scanned = classifier.scan_hud_slots(frame, hsv=hsv, gray=gray)
    assert len(scanned) == 4

    upper_scanned = [s for s in scanned if 0.865 <= s["cy"] <= 0.938]
    lower_scanned = [s for s in scanned if s["cy"] >= 0.939]

    assert {s["class_name"] for s in upper_scanned} == {"hp_potion", "mp_potion", "buff_scroll"}
    for s in upper_scanned:
        assert s["class_name"] not in ("skill_slot", "skill_ui_prompt")
        assert s["class_id"] in (18, 19, 22)
        assert s["class_id"] not in (13, 21)

    assert len(lower_scanned) == 1
    assert lower_scanned[0]["class_name"] == "skill_slot"
    assert lower_scanned[0]["class_id"] == 21

    # 2. Verify YOLOAutoLabeler._scan_hud_and_skills
    labeler = YOLOAutoLabeler()
    yolo_anns = labeler._scan_hud_and_skills(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="learn_skills_w31_hud_test",
    )
    assert len(yolo_anns) >= 4

    for ann in yolo_anns:
        assert ann["class_id"] != 13, "HUD scanner must NEVER emit class_id 13 (skill_ui_prompt)"
        assert ann.get("material_name") != "skill_ui_prompt"
        assert YOLO_CLASSES.get(ann["class_id"]) != "skill_ui_prompt"
        if 0.865 <= ann["cy"] <= 0.938 and ann["class_id"] in (18, 19, 21, 22):
            assert ann["class_id"] in (18, 19, 22)
            assert ann["class_id"] != 21
            assert ann["material_name"] in ("hp_potion", "mp_potion", "buff_scroll")
        elif ann["cy"] >= 0.939 and ann["class_id"] in (18, 19, 21, 22):
            assert ann["class_id"] == 21
            assert ann["material_name"] == "skill_slot"
