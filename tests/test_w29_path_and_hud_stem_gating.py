# -*- coding: utf-8 -*-
"""
Unit tests for W29: Path & HUD Stem Gating in YOLOAutoLabeler.

Verifies that on normal gather frames (`boot_*_aria_*`), grass regions never receive
`15: path_ground` annotations and HUD regions never receive fake `13: skill_ui_prompt`
(or unrequested HUD/skill boxes) due to `is_path_learning_stem` and `is_skill_learning_stem` locks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory.yolo_auto_labeler import (
    YOLOAutoLabeler,
    is_path_learning_stem,
    is_skill_learning_stem,
)


def _build_synthetic_aria_grass_and_hud_frame(height: int = 1080, width: int = 1920) -> np.ndarray:
    """Create a synthetic 1080p frame with grass in the path ROI and HUD elements at the bottom."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Fill ground/path ROI (y: 0.52..0.85, x: 0.28..0.72) with natural green grass
    # BGR (60, 115, 65) -> HSV Saturation ~122 (<= 150), Value 115 (>= 35), smooth Sobel gradient
    # This triggers 15: path_ground in _scan_path_and_obstacles if not locked by is_path_learning_stem
    py1, py2 = int(height * 0.50), int(height * 0.86)
    px1, px2 = int(width * 0.25), int(width * 0.75)
    frame[py1:py2, px1:px2] = (60, 115, 65)

    # Add fake HUD bars and skill/consumable slots in bottom HUD (y: 0.86..0.98)
    cv2.rectangle(frame, (400, 900), (600, 916), (0, 0, 225), -1)   # Red HP bar
    cv2.rectangle(frame, (400, 924), (600, 940), (225, 105, 0), -1)  # Blue MP bar

    # Upper consumable slots (y ~ 965 -> norm_cy ~ 0.893)
    for sx in (660, 710, 760, 810):
        cv2.rectangle(frame, (sx, 950), (sx + 36, 986), (175, 175, 175), 2)
        cv2.circle(frame, (sx + 18, 968), 10, (90, 190, 230), -1)

    # Lower combat skill slots (y ~ 1030 -> norm_cy ~ 0.954)
    for sx in (660, 710, 760, 810):
        cv2.rectangle(frame, (sx, 1012), (sx + 38, 1050), (190, 190, 190), 2)
        cv2.line(frame, (sx + 6, 1020), (sx + 32, 1042), (140, 160, 250), 2)

    return frame


def test_w29_01_stem_gating_locks_reject_boot_aria_stems():
    """
    Test 1: Verify `is_path_learning_stem` and `is_skill_learning_stem` strictly return False
    for normal gather frames (`boot_*_aria_*`) while returning True for dedicated learning stems.
    """
    boot_aria_stems = [
        "boot_478467_000_aria_3d_crest_far",
        "boot_478467_001_aria_gather_prompt",
        "boot_912345_012_aria_herb_node",
        "boot_001_aria_field_grass",
        "",
    ]
    for stem in boot_aria_stems:
        assert is_path_learning_stem(stem) is False, f"Expected False for path stem lock on {stem!r}"
        assert is_skill_learning_stem(stem) is False, f"Expected False for skill stem lock on {stem!r}"

    assert is_path_learning_stem("learn_paths_001_grass") is True
    assert is_skill_learning_stem("learn_skills_001_hud") is True


def test_w29_02_scan_path_and_obstacles_blocks_grass_path_ground_on_boot_aria():
    """
    Test 2: Verify `_scan_path_and_obstacles` does NOT emit `15: path_ground` (or `14: obstacle`)
    over smooth grass on `boot_*_aria_*` frames, even when the grass texture satisfies ground criteria.
    """
    labeler = YOLOAutoLabeler()
    frame = _build_synthetic_aria_grass_and_hud_frame()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Confirm the synthetic grass DOES trigger 15: path_ground when unlocked via learn_paths_*
    unlocked_anns = labeler._scan_path_and_obstacles(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="learn_paths_synthetic_grass_check",
    )
    assert any(a["class_id"] == 15 for a in unlocked_anns), (
        "Synthetic grass should produce class 15 when stem is learn_paths_*"
    )

    # Now verify strict lock on boot_*_aria_* stems
    for stem in ("boot_478467_000_aria_3d_crest_far", "boot_102_aria_meadow"):
        locked_anns = labeler._scan_path_and_obstacles(
            frame_bgr=frame,
            hsv=hsv,
            gray=gray,
            source_stem=stem,
        )
        assert locked_anns == [], f"Expected empty list for {stem}, got {locked_anns}"
        assert not any(a["class_id"] == 15 for a in locked_anns)


def test_w29_03_scan_hud_and_skills_blocks_fake_skill_ui_prompt_on_boot_aria():
    """
    Test 3: Verify `_scan_hud_and_skills` returns no annotations on `boot_*_aria_*` stems
    and never emits fake `13: skill_ui_prompt` over the HUD even on `learn_skills_*` stems.
    """
    labeler = YOLOAutoLabeler()
    frame = _build_synthetic_aria_grass_and_hud_frame()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Normal gather frame (`boot_*_aria_*`) must be completely blocked by is_skill_learning_stem
    for stem in ("boot_478467_000_aria_3d_crest_far", "boot_555_003_aria_gather"):
        hud_anns = labeler._scan_hud_and_skills(
            frame_bgr=frame,
            hsv=hsv,
            gray=gray,
            source_stem=stem,
        )
        assert hud_anns == [], f"HUD scanner must return [] on {stem}, got {hud_anns}"

    # Even when unlocked with learn_skills_*, fake 13: skill_ui_prompt must never be emitted on HUD
    unlocked_hud_anns = labeler._scan_hud_and_skills(
        frame_bgr=frame,
        hsv=hsv,
        gray=gray,
        source_stem="learn_skills_hud_verification",
    )
    assert len(unlocked_hud_anns) >= 1
    assert all(a["class_id"] != 13 for a in unlocked_hud_anns), (
        "HUD scanner must never emit fake class 13 (skill_ui_prompt) on HUD slots!"
    )


def test_w29_04_auto_label_frame_end_to_end_boot_aria_no_path_ground_or_skill_ui_prompt():
    """
    Test 4: End-to-end `auto_label_frame` check on a `boot_*_aria_*` source path.
    Ensures neither `15: path_ground` on grass nor `13: skill_ui_prompt` on HUD is emitted.
    """
    labeler = YOLOAutoLabeler(exemplar_memory=False)
    frame = _build_synthetic_aria_grass_and_hud_frame()

    annotations = labeler.auto_label_frame(
        frame=frame,
        known_material_name="Aria",
        fast_mode=True,
        source_path="dataset/images/boot_478467_000_aria_3d_crest_far.jpg",
    )

    emitted_class_ids = {int(a["class_id"]) for a in annotations}
    assert 15 not in emitted_class_ids, f"15: path_ground must not be emitted on boot_*_aria_*: {annotations}"
    assert 13 not in emitted_class_ids, f"13: skill_ui_prompt must not be emitted on boot_*_aria_*: {annotations}"
    assert 14 not in emitted_class_ids, f"14: obstacle must not be emitted on boot_*_aria_*: {annotations}"


def test_w29_05_auto_label_frame_preserves_aria_and_filters_minimap_hud_poison_on_boot_aria():
    """
    Test 5: Verify `auto_label_frame` on a `boot_*_aria_*` frame preserves valid 3D `0: herb_aria`
    target detections while rejecting grass `15: path_ground` and dropping any illegal `13: skill_ui_prompt`
    or `15: path_ground` injected into the HUD minimap region.
    """
    labeler = YOLOAutoLabeler(exemplar_memory=False)
    frame = _build_synthetic_aria_grass_and_hud_frame()

    # Draw a vibrant purple Aria crystal cluster around (960, 420)
    cv2.circle(frame, (960, 420), 28, (215, 70, 235), -1)

    extra = [
        # Fake 13: skill_ui_prompt inside HUD minimap region (cx=0.90, cy=0.15)
        {
            "class_id": 13,
            "cx": 0.90,
            "cy": 0.15,
            "w": 0.03,
            "h": 0.04,
            "confidence": 0.80,
            "source": "hud_skill_slot",
            "material_name": "skill_ui_prompt",
        },
        # Fake 15: path_ground inside HUD minimap region (cx=0.91, cy=0.18)
        {
            "class_id": 15,
            "cx": 0.91,
            "cy": 0.18,
            "w": 0.04,
            "h": 0.04,
            "confidence": 0.65,
            "source": "visual_ground_path_segmenter",
            "material_name": "path_ground",
        },
    ]

    annotations = labeler.auto_label_frame(
        frame=frame,
        extra_detections=extra,
        known_target_xy=(960, 420),
        known_material_name="Aria",
        fast_mode=True,
        source_path="boot_478467_002_aria_close.png",
    )

    class_ids = [int(a["class_id"]) for a in annotations]
    assert 0 in class_ids, "Valid 0: herb_aria detection must be preserved on boot_*_aria_*"
    assert 13 not in class_ids, "Fake 13: skill_ui_prompt must be filtered out on boot_*_aria_*"
    assert 15 not in class_ids, "15: path_ground must never appear on boot_*_aria_*"
