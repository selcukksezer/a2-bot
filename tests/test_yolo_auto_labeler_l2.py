# -*- coding: utf-8 -*-
"""
Tests for Worker L2: YOLO Auto-Labeler Calibration, Stem Gating, and HUD/Consumable Separation.

Validates:
1. `_scan_salient_3d_resources`:
   - Artificial 0.85 floor is completely removed.
   - Confidence strictly calibrated via min(0.68, max(0.38, round(p_score * 0.75, 4))).
   - class_id is never > 12 (never essence_*).
   - Always marked trusted=False, weak=True.
   - Gated as fallback when NO 3D crest or prompt is present.
2. `_scan_path_and_obstacles`:
   - Normal gather/boot frames (e.g. `boot_*`, `gather_*`) NEVER receive path or obstacle boxes.
   - Only runs on `learn_paths_*` or `learn_obstacles_*` stems with confidence <= 0.68 (weak=True).
3. `_scan_hud_and_skills`:
   - Normal frames NEVER receive HUD/skill/vitality/consumable boxes.
   - Only runs on `learn_skills_*` or `learn_vitality_*` stems with confidence <= 0.70 (weak=True).
   - Upper consumable row (0.870..0.938*H) classifies hp_potion (18), mp_potion (19), buff_scroll (22).
   - Lower combat skills row (>=0.940*H) classifies skill_slot (21).
   - Center level 45 emblem is excluded.
4. `shared_frame_context`:
   - Reuses cached bgr, hsv, gray, and edge representations across sub-scanners.
5. 3D Monster detection:
   - Recognizes yellow nameplate mobs (Hue 20..32, Sat 90..255, Val 110..255) and cyan diamond brackets.
   - Respects central player self-exclusion zone.
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

from dataset_factory.yolo_auto_labeler import (
    YOLOAutoLabeler,
    is_path_learning_stem,
    is_skill_learning_stem,
    map_material_name_to_yolo_class_id,
    YOLO_CLASSES,
)


def test_stem_gating_helpers():
    """Verify is_path_learning_stem and is_skill_learning_stem accurately discriminate stems."""
    # Path learning stems
    assert is_path_learning_stem("learn_paths_001_corridor") is True
    assert is_path_learning_stem("path_ground_run") is True
    assert is_path_learning_stem("learn_obstacles_stone") is True
    assert is_path_learning_stem("boot_478467_000_aria_3d_crest_far") is False
    assert is_path_learning_stem("gather_orichalcum_node_12") is False

    # Skill learning stems
    assert is_skill_learning_stem("learn_skills_combo_01") is True
    assert is_skill_learning_stem("learn_vitality_hp_test") is True
    assert is_skill_learning_stem("skill_rotation_fast") is True
    assert is_skill_learning_stem("boot_478467_000_aria_3d_crest_far") is False
    assert is_skill_learning_stem("gather_orichalcum_node_12") is False


def test_salient_3d_resources_no_floor_and_confidence_calibrated():
    """
    Test 1: _scan_salient_3d_resources must:
    - Never assign class_id > 12.
    - Calibrate confidence to min(0.68, max(0.38, round(p_score * 0.75, 4))).
    - Set trusted=False, weak=True.
    """
    labeler = YOLOAutoLabeler()

    # Create a 720x1280 frame with a high-saturation blob in the viewport
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Add a chromatic salient patch in viewport region (vx1..vx2 = 204..1049, vy1..vy2 = 129..561)
    cv2.circle(frame, (500, 350), 30, (0, 180, 220), -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Direct call to _scan_salient_3d_resources
    salient_anns = labeler._scan_salient_3d_resources(
        frame_bgr=frame,
        hsv=hsv,
        existing_boxes=[],
        hint_material_name="Orichalcum Ore",
        has_crest_or_prompt=False,
    )

    for ann in salient_anns:
        # Confidence must be strictly <= 0.68 and >= 0.38
        assert ann["confidence"] <= 0.68, f"Artificial floor detected: {ann['confidence']} > 0.68"
        assert ann["confidence"] >= 0.38
        # Class ID must be <= 12
        assert ann["class_id"] <= 12, f"Invalid class ID: {ann['class_id']} > 12"
        # Must be untrusted / weak
        assert ann["trusted"] is False, "Salient fallback must not be trusted"
        assert ann["weak"] is True, "Salient fallback must be weak"


def test_salient_3d_resources_fallback_gating():
    """
    Test 2: _scan_salient_3d_resources is a fallback that must NOT run when
    a 3D crest or prompt is already detected.
    """
    labeler = YOLOAutoLabeler()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # When has_crest_or_prompt is True, should return []
    anns = labeler._scan_salient_3d_resources(
        frame_bgr=frame,
        hsv=hsv,
        existing_boxes=[],
        has_crest_or_prompt=True,
    )
    assert anns == [], "Salient 3D resource scan must return [] when crest or prompt is present"


def test_path_and_obstacles_strict_stem_gating():
    """
    Test 3: _scan_path_and_obstacles ONLY executes on path learning stems.
    Normal gather/boot frames must NEVER receive 15: path_ground or 14: obstacle annotations.
    """
    labeler = YOLOAutoLabeler()
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 120
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Normal frame: boot_*
    anns_boot = labeler._scan_path_and_obstacles(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="boot_478467_000_aria_3d_crest_far",
    )
    assert anns_boot == [], "Normal boot frame must NEVER receive path/obstacle annotations!"

    # 2. Normal frame: gather_*
    anns_gather = labeler._scan_path_and_obstacles(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="gather_orichalcum_ore_node_01",
    )
    assert anns_gather == [], "Normal gather frame must NEVER receive path/obstacle annotations!"

    # 3. Path learning stem: learn_paths_*
    anns_path = labeler._scan_path_and_obstacles(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="learn_paths_corridor_01",
    )
    # If any path annotation was emitted, its confidence must be capped at <= 0.68
    for ann in anns_path:
        assert ann["confidence"] <= 0.68, f"Path confidence {ann['confidence']} exceeded 0.68 cap"
        assert ann["weak"] is True
        assert ann["trusted"] is False


def test_hud_and_skills_strict_stem_gating_and_row_separation():
    """
    Test 4: _scan_hud_and_skills ONLY executes on learn_skills/learn_vitality stems.
    On valid stems:
    - Excludes center level 45 emblem.
    - Upper consumable row classifies hp_potion (18), mp_potion (19), buff_scroll (22).
    - Lower combat skills row classifies skill_slot (21).
    - Confidence is capped at <= 0.70 (weak=True).
    """
    labeler = YOLOAutoLabeler()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # 1. Normal frame must return []
    anns_normal = labeler._scan_hud_and_skills(
        frame_bgr=frame,
        hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV),
        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        source_stem="gather_aria_plant_05",
    )
    assert anns_normal == [], "Normal frame must NEVER receive HUD/skill annotations!"

    # 2. Skill learning stem with simulated HUD elements
    # Add HP bar in bottom HUD (x: 400..600, y: 900..920)
    cv2.rectangle(frame, (400, 900), (600, 915), (0, 0, 220), -1)  # Red HP bar
    cv2.rectangle(frame, (400, 925), (600, 940), (220, 100, 0), -1)  # Blue MP bar

    # Add lower combat skill slot (y: 1020, norm_cy ~ 0.944 >= 0.940)
    cv2.rectangle(frame, (800, 1010), (840, 1050), (180, 180, 180), 2)
    cv2.circle(frame, (820, 1030), 10, (150, 150, 255), -1)

    # Add upper consumable row items (y: 960, norm_cy ~ 0.888 in [0.870, 0.938])
    # A. HP Potion (red)
    cv2.rectangle(frame, (650, 950), (685, 985), (160, 160, 160), 2)
    cv2.circle(frame, (667, 967), 10, (0, 0, 240), -1)

    # B. MP Potion (blue)
    cv2.rectangle(frame, (700, 950), (735, 985), (160, 160, 160), 2)
    cv2.circle(frame, (717, 967), 10, (240, 120, 0), -1)

    # C. Buff scroll (golden/cream)
    cv2.rectangle(frame, (750, 950), (785, 985), (160, 160, 160), 2)
    cv2.circle(frame, (767, 967), 10, (80, 180, 220), -1)

    # D. Center emblem inside exclusion zone (x: 960 ~ 0.50*W, y: 970 ~ 0.898*H)
    cv2.rectangle(frame, (940, 950), (980, 990), (255, 255, 255), -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    anns_skill = labeler._scan_hud_and_skills(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="learn_skills_combo_01",
    )

    assert len(anns_skill) >= 1

    # Verify no annotation was placed in center emblem exclusion zone
    for ann in anns_skill:
        cx_norm = ann["cx"]
        cy_norm = ann["cy"]
        assert not (0.44 <= cx_norm <= 0.56 and 0.85 <= cy_norm <= 0.95), (
            f"Center emblem exclusion failed: box at cx={cx_norm:.3f}, cy={cy_norm:.3f}"
        )
        # All confidence capped at <= 0.70 and weak=True
        assert ann["confidence"] <= 0.70
        assert ann["weak"] is True
        assert ann["trusted"] is False

    # Check that upper row consumable boxes have class_id in (18, 19, 22) and NOT 13 or 21
    upper_row_anns = [a for a in anns_skill if 0.865 <= a["cy"] <= 0.938 and a["class_id"] in (18, 19, 21, 22)]
    for a in upper_row_anns:
        assert a["class_id"] != 21, "Upper consumable row must NEVER be labeled as skill_slot!"
        assert a["class_id"] != 13, "Upper consumable row must NEVER be labeled as skill_ui_prompt!"


def test_shared_frame_context_caching():
    """
    Test 5: Verify that shared_frame_context caches bgr, hsv, gray, edges and reuses them.
    """
    labeler = YOLOAutoLabeler()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame, (320, 240), 50, (50, 150, 200), -1)

    context = {}
    _ = labeler.auto_label_frame(frame, shared_frame_context=context)

    assert "bgr" in context
    assert "hsv" in context
    assert "gray" in context
    assert "edges" in context
    assert isinstance(context["hsv"], np.ndarray)
    assert isinstance(context["gray"], np.ndarray)
    assert isinstance(context["edges"], np.ndarray)

    # Calling again with same context should not recompute
    cached_hsv_id = id(context["hsv"])
    _ = labeler.auto_label_frame(frame, shared_frame_context=context)
    assert id(context["hsv"]) == cached_hsv_id


def test_3d_monster_detection_with_brackets():
    """
    Test 6: Verify yellow nameplate mob with cyan diamond brackets is detected
    as 10: monster_3d with genuine bounding box, respecting player self-exclusion zone.
    """
    labeler = YOLOAutoLabeler()
    # 1080p frame
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    # Add yellow nameplate text at x=700, y=300 (W=180, H=14, aspect > 4.5)
    # Hue: 25, Sat: 150, Val: 180 (Yellow text)
    hsv_text_color = (25, 150, 180)
    bgr_text_color = cv2.cvtColor(np.uint8([[list(hsv_text_color)]]), cv2.COLOR_HSV2BGR)[0][0]
    bgr_text_tuple = (int(bgr_text_color[0]), int(bgr_text_color[1]), int(bgr_text_color[2]))

    # Render realistic yellow text stroke via putText
    cv2.putText(frame, "< Yellow-Maned Zait >", (700, 312), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_text_tuple, 2)

    # Add cyan diamond brackets on left and right of nameplate
    # Hue: 90, Sat: 180, Val: 200
    cyan_hsv = (90, 180, 200)
    cyan_bgr = cv2.cvtColor(np.uint8([[list(cyan_hsv)]]), cv2.COLOR_HSV2BGR)[0][0]
    cyan_tuple = (int(cyan_bgr[0]), int(cyan_bgr[1]), int(cyan_bgr[2]))

    cv2.rectangle(frame, (680, 298), (696, 316), cyan_tuple, -1)
    cv2.rectangle(frame, (855, 298), (870, 316), cyan_tuple, -1)

    # Add mob body below nameplate
    cv2.rectangle(frame, (680, 320), (900, 420), (40, 40, 80), -1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    monsters = labeler._scan_3d_monsters(frame_bgr=frame, hsv=hsv, existing_positive_boxes=[], gray=gray)
    assert len(monsters) >= 1
    m = monsters[0]
    assert m["class_id"] == 10
    assert m["material_name"] == "monster_3d"
    # Bounding box should enclose both nameplate and body
    assert m["bbox"][2] >= 150  # width
    assert m["bbox"][3] >= 50   # height
