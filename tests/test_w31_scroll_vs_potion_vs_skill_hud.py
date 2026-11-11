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


def _hsv_of_rgb(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Real-world RGB reference color -> OpenCV HSV triple (via cv2, not hand-tuned)."""
    px = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return tuple(int(v) for v in cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0][0])


def _make_rgb_slot(
    rgb: tuple[int, int, int],
    size: int = 36,
    with_border_variation: bool = True,
) -> np.ndarray:
    """Create a synthetic slot crop in BGR from a real-world RGB reference color.

    The color is specified as an RGB pigment (e.g. parchment, potion red) and
    converted with cv2 -- never hand-picked from the classifier's own HSV
    thresholds. The classifier receives BGR (is_hsv=False) and does its own
    conversion, so the test exercises the production conversion path too.
    """
    h_val, s_val, v_val = _hsv_of_rgb(rgb)
    slot_hsv = np.zeros((size, size, 3), dtype=np.uint8)
    slot_hsv[:, :] = (h_val, s_val, v_val)
    if with_border_variation:
        # Add subtle bright/dark bevel so val_std > 16.0 (not treated as flat empty slot)
        slot_hsv[:4, :, 2] = np.clip(int(v_val) + 45, 0, 255)
        slot_hsv[-4:, :, 2] = max(10, int(v_val) - 55)
        slot_hsv[8:14, 8:14, 2] = max(15, int(v_val) - 60)
        slot_hsv[18:26, 18:26, 2] = min(255, int(v_val) + 40)
    return cv2.cvtColor(slot_hsv, cv2.COLOR_HSV2BGR)


# Real-world RGB reference colors (pigment names, not classifier thresholds).
PARCHMENT_RGB = (230, 211, 163)  # rolled parchment / buff scroll (#E6D3A3)
HP_POTION_RGB = (200, 50, 50)  # red potion liquid (#C83232)
MP_BLUE_RGB = (46, 111, 216)  # blue mana potion (#2E6FD8)
MP_YELLOW_RGB = (229, 193, 0)  # golden-yellow serum (#E5C100)
GOLDEN_SERUM_RGB = (242, 194, 0)  # saturated golden serum (#F2C200)
SKILL_VIOLET_RGB = (150, 90, 220)  # violet-blue combat skill icon (#965ADC)


def test_upper_row_buff_scroll_classification_and_no_skill_mislabel():
    """
    Test 1: Upper row (`0.865 <= norm_cy <= 0.938`) parchment slot must be classified
    as `buff_scroll` and never as `skill_slot` or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    # Parchment RGB reference (#E6D3A3) converts to HSV ~(21, 74, 230): a tan
    # parchment tone. It must read as a scroll, never as a yellow MP serum --
    # the serum reference below (high saturation) is the boundary control.
    scroll_bgr = _make_rgb_slot(PARCHMENT_RGB)

    for norm_cy in (UPPER_CONSUMABLE_ROW_Y_MIN, 0.900, UPPER_CONSUMABLE_ROW_Y_MAX):
        res = classifier.classify_slot(
            scroll_bgr,
            norm_cx=0.38,
            norm_cy=norm_cy,
            is_hsv=False,
        )
        assert res.should_label is True
        assert res.row_type == "upper_consumable_row"
        assert res.category == "buff_scroll"
        assert res.class_id == YOLO_HUD_CLASS_IDS["buff_scroll"] == 22
        assert res.canonical_class_id == CANONICAL_HUD_CLASS_IDS["buff_scroll"] == 24
        assert res.category not in ("skill_slot", "skill_ui_prompt")
        assert res.class_id not in (13, 21)

    # The same parchment slot given as HSV must classify identically -- the
    # classifier's own BGR->HSV conversion must agree with the pre-converted input.
    scroll_hsv = cv2.cvtColor(scroll_bgr, cv2.COLOR_BGR2HSV)
    res_hsv = classifier.classify_slot(scroll_hsv, norm_cx=0.38, norm_cy=0.900, is_hsv=True)
    assert res_hsv.category == "buff_scroll"
    assert res_hsv.class_id == 22

    # Boundary control: a saturated golden serum (same hue family as parchment,
    # but high saturation) must be an MP potion, never a scroll.
    serum_bgr = _make_rgb_slot(GOLDEN_SERUM_RGB)
    serum_res = classifier.classify_slot(serum_bgr, norm_cx=0.38, norm_cy=0.900, is_hsv=False)
    assert serum_res.category == "mp_potion"
    assert serum_res.category != "buff_scroll"


def test_upper_row_hp_potion_classification_and_no_skill_mislabel():
    """
    Test 2: Upper row (`0.865 <= norm_cy <= 0.938`) red potion slot must be classified
    as `hp_potion` and never as `skill_slot` or `skill_ui_prompt`.
    """
    classifier = HUDConsumableClassifier()
    # Red potion RGB reference (#C83232) converts to HSV ~(0, 191, 200).
    hp_bgr = _make_rgb_slot(HP_POTION_RGB)

    for norm_cy in (0.865, 0.895, 0.938):
        res = classifier.classify_slot(
            hp_bgr,
            norm_cx=0.34,
            norm_cy=norm_cy,
            is_hsv=False,
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
    # Blue mana potion (#2E6FD8) and golden-yellow serum (#E5C100) RGB references.
    mp_blue_bgr = _make_rgb_slot(MP_BLUE_RGB)
    mp_yellow_bgr = _make_rgb_slot(MP_YELLOW_RGB)

    for slot_bgr in (mp_blue_bgr, mp_yellow_bgr):
        res = classifier.classify_slot(
            slot_bgr,
            norm_cx=0.36,
            norm_cy=0.912,
            is_hsv=False,
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
    skill_bgr = _make_rgb_slot(HP_POTION_RGB)
    edges = np.zeros((36, 36), dtype=np.uint8)
    cv2.rectangle(edges, (6, 6), (30, 30), 255, 2)

    # At norm_cy = 0.938 -> upper consumable row -> hp_potion
    upper_res = classifier.classify_slot(
        skill_bgr,
        norm_cx=0.40,
        norm_cy=0.938,
        is_hsv=False,
        slot_edges=edges,
    )
    assert upper_res.row_type == "upper_consumable_row"
    assert upper_res.category == "hp_potion"
    assert upper_res.category != "skill_slot"

    # At norm_cy = 0.939 and 0.960 -> lower skill row -> skill_slot
    for norm_cy in (0.939, 0.960):
        lower_res = classifier.classify_slot(
            skill_bgr,
            norm_cx=0.40,
            norm_cy=norm_cy,
            is_hsv=False,
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
    scroll_bgr = _make_rgb_slot(PARCHMENT_RGB)

    # Center level emblem in upper consumable row Y range
    emblem_res = classifier.classify_slot(
        scroll_bgr,
        norm_cx=0.50,
        norm_cy=0.900,
        is_hsv=False,
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
    `YOLOAutoLabeler._scan_hud_and_skills` on a 960x540 synthetic HUD frame must:
    - Classify upper row (`0.865 <= cy <= 0.938`) slots into `hp_potion` (22), `mp_potion` (23),
      and `buff_scroll` (24), never `skill_slot` (20) or `skill_ui_prompt` (13).
    - Classify lower row (`cy >= 0.939`) slots into `skill_slot` (20), never consumables or
      `skill_ui_prompt` (13).

    NOTE (canonical ids): the classifier's `scan_hud_slots` still emits the
    legacy HUD ids (18/19/22/21, `YOLO_HUD_CLASS_IDS`) — that is its own
    pinned contract (tests 1-5 above). The labeler's `_scan_hud_and_skills`
    emits the CANONICAL taxonomy ids (22/23/24/20,
    `dataset_factory/class_taxonomy.py`; `yolo_auto_labeler.py` header:
    "legacy ids must never be reintroduced here"), so part 2 below pins
    the canonical ids, not the legacy ones.
    """
    # Half-resolution frame (960x540) with half-size slots: slot centres land
    # on the same normalized cy (0.90185 upper / 0.94630 lower), so the row
    # separation and every class assertion below are identical to 1080p input.
    frame = np.zeros((540, 960, 3), dtype=np.uint8)

    def _draw_hud_slot(x1: int, y1: int, size: int, rgb: tuple[int, int, int]) -> None:
        # RGB reference color -> BGR fill (no hand-tuned HSV).
        bgr_color = (rgb[2], rgb[1], rgb[0])
        # Outer metallic slot frame for contour detection
        cv2.rectangle(frame, (x1, y1), (x1 + size, y1 + size), (190, 190, 190), 2)
        # Inner colored consumable / skill fill
        cv2.rectangle(frame, (x1 + 4, y1 + 4), (x1 + size - 4, y1 + size - 4), bgr_color, -1)
        # Inner detail for realistic value variance & Canny edges
        cv2.rectangle(frame, (x1 + 10, y1 + 10), (x1 + size - 10, y1 + size - 10), (245, 245, 245), 1)

    # Upper row slots: y1=478, size=18 -> cy = 487 / 540 = 0.90185 (in [0.865, 0.938])
    # Colors are RGB references (see module constants), not hand-tuned HSV.
    _draw_hud_slot(330, 478, 18, HP_POTION_RGB)    # Red HP Potion
    _draw_hud_slot(355, 478, 18, MP_BLUE_RGB)      # Blue MP Potion
    _draw_hud_slot(380, 478, 18, PARCHMENT_RGB)    # Parchment Buff Scroll

    # Lower row slot: y1=502, size=18 -> cy = 511 / 540 = 0.94630 (>= 0.939)
    _draw_hud_slot(405, 502, 18, SKILL_VIOLET_RGB)  # Combat Skill Slot

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
    # Exact count and exact class multiset: the labeler must emit one annotation
    # per drawn slot and nothing else (no extras, no missing slots, no duplicates).
    # Canonical ids (22/23/24/20) — the labeler never emits the legacy HUD ids
    # (18/19/22/21) that the classifier's scan_hud_slots still returns in part 1.
    assert len(yolo_anns) == 4, f"Expected exactly 4 HUD annotations, got: {yolo_anns}"
    assert {ann["class_id"] for ann in yolo_anns} == {20, 22, 23, 24}

    # Exact per-slot mapping: upper-row slots at cy ~= 0.90185 are the three
    # consumables, the lower-row slot at cy ~= 0.94630 is the skill slot.
    by_class = {ann["class_id"]: ann for ann in yolo_anns}
    assert by_class[22]["material_name"] == "hp_potion"
    assert abs(by_class[22]["cy"] - 487 / 540) < 0.005
    assert by_class[23]["material_name"] == "mp_potion"
    assert abs(by_class[23]["cy"] - 487 / 540) < 0.005
    assert by_class[24]["material_name"] == "buff_scroll"
    assert abs(by_class[24]["cy"] - 487 / 540) < 0.005
    assert by_class[20]["material_name"] == "skill_slot"
    assert abs(by_class[20]["cy"] - 511 / 540) < 0.005

    for ann in yolo_anns:
        assert ann["class_id"] != 13, "HUD scanner must NEVER emit class_id 13 (skill_ui_prompt)"
        assert ann.get("material_name") != "skill_ui_prompt"
        assert YOLO_CLASSES.get(ann["class_id"]) != "skill_ui_prompt"
        if 0.865 <= ann["cy"] <= 0.938 and ann["class_id"] in (20, 22, 23, 24):
            assert ann["class_id"] in (22, 23, 24)
            assert ann["class_id"] != 20
            assert ann["material_name"] in ("hp_potion", "mp_potion", "buff_scroll")
        elif ann["cy"] >= 0.939 and ann["class_id"] in (20, 22, 23, 24):
            assert ann["class_id"] == 20
            assert ann["material_name"] == "skill_slot"
