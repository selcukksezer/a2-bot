# -*- coding: utf-8 -*-
"""Unit tests for W37: Minimap 3-leaf sprout (9) vs circular red/white monster dot (11) vs center player arrow separation."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.minimap_scanner import MinimapScanner


def _make_base_minimap_crop(width: int = 332, height: int = 257) -> np.ndarray:
    """Create a clean dark-slate minimap background crop (332x257 reference scale)."""
    return np.full((height, width, 3), (52, 62, 56), dtype=np.uint8)


def _paste_green_3_leaf_sprout(crop_bgr: np.ndarray, top_left_x: int, top_left_y: int) -> Tuple[int, int]:
    """
    Paste the canonical 16x16 3-leaf sprout template onto crop_bgr with green/bright-leaf
    chromaticity so both template NCC (>= 0.88) and green/white HSV chromatic checks pass.
    Returns the expected center (cx, cy).
    """
    tpl = MinimapScanner._CANONICAL_SPROUT_TPL
    patch_bgr = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR)
    # Tint bright leaf pixels (tpl >= 175) with a natural bright green-white sprout hue
    # while keeping the luminance profile matched to _CANONICAL_SPROUT_TPL for high NCC.
    leaf_pixels = tpl >= 175
    patch_bgr[leaf_pixels, 0] = np.clip(tpl[leaf_pixels].astype(np.int32) - 45, 130, 255).astype(np.uint8)  # B
    patch_bgr[leaf_pixels, 1] = tpl[leaf_pixels]  # G (highest)
    patch_bgr[leaf_pixels, 2] = np.clip(tpl[leaf_pixels].astype(np.int32) - 35, 140, 255).astype(np.uint8)  # R

    crop_bgr[top_left_y : top_left_y + 16, top_left_x : top_left_x + 16] = patch_bgr
    return top_left_x + 8, top_left_y + 8


def _draw_red_monster_dot(crop_bgr: np.ndarray, cx: int, cy: int, radius: int = 4) -> None:
    """Draw a circular hostile red monster dot (HSV red, circularity >= 0.65)."""
    cv2.circle(crop_bgr, (cx, cy), radius + 2, (25, 25, 25), -1)
    cv2.circle(crop_bgr, (cx, cy), radius, (20, 20, 245), -1)


def _draw_white_monster_dot(crop_bgr: np.ndarray, cx: int, cy: int, radius: int = 4) -> None:
    """Draw a circular white monster dot ('•') with dark outer ring contrast."""
    cv2.circle(crop_bgr, (cx, cy), radius + 3, (30, 35, 32), -1)
    cv2.circle(crop_bgr, (cx, cy), radius, (245, 245, 245), -1)


def _draw_center_player_arrow(crop_bgr: np.ndarray) -> Tuple[int, int]:
    """Draw a bright North-pointing player arrow chevron at the canonical player center."""
    h, w = crop_bgr.shape[:2]
    cx = int(round(w * MinimapScanner.PLAYER_CENTER_REL[0]))
    cy = int(round(h * MinimapScanner.PLAYER_CENTER_REL[1]))
    pts = np.array(
        [
            [cx, cy - 6],      # North apex
            [cx + 5, cy + 5],  # Right wing
            [cx, cy + 2],      # Rear notch
            [cx - 5, cy + 5],  # Left wing
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(crop_bgr, [pts], (255, 255, 255))
    return cx, cy


def test_green_3_leaf_sprout_detected_as_class_9_only(tmp_path: Path) -> None:
    """1. Verify a 3-leaf green/canonical sprout on the minimap is classified strictly as class 9 (minimap_sprout)."""
    scanner = MinimapScanner(templates_dir=str(tmp_path), config_dir=str(tmp_path))
    crop = _make_base_minimap_crop()
    expected_cx, expected_cy = _paste_green_3_leaf_sprout(crop, 70, 96)

    result = scanner.scan_minimap_crop(crop)
    gatherables = result["gatherable_nodes"]
    monster_dots = result["monster_dots"]
    hostile_dots = result["hostile_dots"]

    assert len(gatherables) == 1, f"Expected exactly 1 gatherable sprout, got: {gatherables}"
    assert monster_dots == [], f"Sprout must not be classified as white monster_dot: {monster_dots}"
    assert hostile_dots == [], f"Sprout must not be classified as red hostile_dot: {hostile_dots}"

    sprout = gatherables[0]
    assert sprout["class_id"] == 9
    assert sprout["class_name"] == "minimap_sprout"
    assert sprout["kind"] == "sprout_material"
    assert sprout["is_gatherable"] is True
    assert sprout["is_hostile"] is False
    assert sprout["is_minimap"] is True
    assert sprout["is_world_3d"] is False
    assert sprout["ncc_score"] >= MinimapScanner.SPROUT_NCC_MIN_THRESHOLD
    assert sprout["chromatic_density"] >= 0.06
    assert math.hypot(sprout["pos"][0] - expected_cx, sprout["pos"][1] - expected_cy) <= 3.0


def test_circular_red_and_white_monster_dots_detected_as_class_11_never_class_9(tmp_path: Path) -> None:
    """2. Verify circular red hostile dots and white monster dots are classified strictly as class 11 and rejected as sprouts (9)."""
    scanner = MinimapScanner(templates_dir=str(tmp_path), config_dir=str(tmp_path))
    crop = _make_base_minimap_crop()

    red_x, red_y = 95, 110
    white_x, white_y = 235, 120
    _draw_red_monster_dot(crop, red_x, red_y, radius=4)
    _draw_white_monster_dot(crop, white_x, white_y, radius=4)

    result = scanner.scan_minimap_crop(crop)
    assert result["gatherable_nodes"] == [], "Circular monster dots must never be classified as class 9 sprouts"
    assert len(result["hostile_dots"]) == 1
    assert len(result["monster_dots"]) == 1

    h_dot = result["hostile_dots"][0]
    assert h_dot["class_id"] == 11
    assert h_dot["class_name"] == "minimap_monster_negative"
    assert h_dot["is_hostile"] is True
    assert h_dot["is_gatherable"] is False
    assert h_dot["circularity"] >= MinimapScanner.HOSTILE_RED_CIRCULARITY_MIN
    assert math.hypot(h_dot["pos"][0] - red_x, h_dot["pos"][1] - red_y) <= 2.5

    m_dot = result["monster_dots"][0]
    assert m_dot["class_id"] == 11
    assert m_dot["class_name"] == "monster_dot"
    assert m_dot["is_hostile"] is False
    assert m_dot["is_gatherable"] is False
    assert m_dot["circularity"] >= 0.62
    assert math.hypot(m_dot["pos"][0] - white_x, m_dot["pos"][1] - white_y) <= 2.5

    # Also verify direct patch structural verification rejects both red and circular white dots
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    assert scanner._verify_sprout_patch_structure(gray, hsv, red_x, red_y, scale=1.0) is False
    assert scanner._verify_sprout_patch_structure(gray, hsv, white_x, white_y, scale=1.0) is False


def test_center_player_arrow_isolated_and_excluded_from_sprout_and_monster_detections(tmp_path: Path) -> None:
    """3. Verify the center player arrow is accurately localized/oriented and never misclassified as class 9 or class 11."""
    scanner = MinimapScanner(templates_dir=str(tmp_path), config_dir=str(tmp_path))
    crop = _make_base_minimap_crop()
    arrow_cx, arrow_cy = _draw_center_player_arrow(crop)

    # Player center and heading detection should succeed on the center arrow
    detected_cx, detected_cy = scanner.get_minimap_player_center(crop.shape, crop_bgr=crop)
    assert math.hypot(detected_cx - arrow_cx, detected_cy - arrow_cy) <= 4.0

    ok_heading, heading_deg = scanner.detect_player_heading(crop)
    assert ok_heading is True
    assert heading_deg <= 25.0 or heading_deg >= 335.0, f"Expected North (~0 deg) heading, got {heading_deg}"

    # Center arrow must be excluded from sprout (9) and monster (11) outputs
    scan_res = scanner.scan_minimap_crop(crop)
    assert scan_res["gatherable_nodes"] == []
    assert scan_res["monster_dots"] == []
    assert scan_res["hostile_dots"] == []

    # Even if a red dot or sprout pattern falls inside the center exclusion zone (<= 14px), it must be suppressed
    crop_with_center_decoys = crop.copy()
    _draw_red_monster_dot(crop_with_center_decoys, arrow_cx + 5, arrow_cy - 4, radius=3)
    _paste_green_3_leaf_sprout(crop_with_center_decoys, arrow_cx - 8, arrow_cy - 8)
    decoy_res = scanner.scan_minimap_crop(crop_with_center_decoys)
    assert decoy_res["gatherable_nodes"] == []
    assert decoy_res["monster_dots"] == []
    assert decoy_res["hostile_dots"] == []


def test_mixed_minimap_scene_cleanly_separates_sprout_9_monster_11_and_player_arrow(tmp_path: Path) -> None:
    """4. Verify a composite minimap with sprout (9), red/white monster dots (11), and player arrow separates all three without error."""
    scanner = MinimapScanner(templates_dir=str(tmp_path), config_dir=str(tmp_path))
    crop = _make_base_minimap_crop()

    arrow_cx, arrow_cy = _draw_center_player_arrow(crop)
    sprout_cx, sprout_cy = _paste_green_3_leaf_sprout(crop, 64, 88)
    red_cx, red_cy = 246, 96
    white_cx, white_cy = 104, 198
    _draw_red_monster_dot(crop, red_cx, red_cy, radius=4)
    _draw_white_monster_dot(crop, white_cx, white_cy, radius=4)

    classified = scanner.classify_minimap_targets(crop)
    sprouts = classified["sprouts"]
    hostiles = classified["hostile_dots"]
    monsters = classified["monster_dots"]
    p_cx, p_cy = classified["player_center"]

    assert math.hypot(p_cx - arrow_cx, p_cy - arrow_cy) <= 4.0
    assert len(sprouts) == 1
    assert len(hostiles) == 1
    assert len(monsters) == 1

    # Class 9 check
    s_node = sprouts[0]
    assert s_node["class_id"] == 9
    assert s_node["class_name"] == "minimap_sprout"
    assert s_node["is_gatherable"] is True
    assert s_node["is_hostile"] is False
    assert math.hypot(s_node["pos"][0] - sprout_cx, s_node["pos"][1] - sprout_cy) <= 3.0
    assert s_node["dist_px"] > MinimapScanner.CENTER_EXCLUSION_RADIUS_PX

    # Class 11 hostile red dot check
    h_node = hostiles[0]
    assert h_node["class_id"] == 11
    assert h_node["class_name"] == "minimap_monster_negative"
    assert h_node["is_gatherable"] is False
    assert h_node["is_hostile"] is True
    assert math.hypot(h_node["pos"][0] - red_cx, h_node["pos"][1] - red_cy) <= 2.5
    assert h_node["dist_px"] > MinimapScanner.CENTER_EXCLUSION_RADIUS_PX

    # Class 11 white monster dot check
    m_node = monsters[0]
    assert m_node["class_id"] == 11
    assert m_node["class_name"] == "monster_dot"
    assert m_node["is_gatherable"] is False
    assert m_node["is_hostile"] is False
    assert math.hypot(m_node["pos"][0] - white_cx, m_node["pos"][1] - white_cy) <= 2.5
    assert m_node["dist_px"] > MinimapScanner.CENTER_EXCLUSION_RADIUS_PX


def test_minimap_taxonomy_and_spatial_suppression_prevent_cross_classification(tmp_path: Path) -> None:
    """5. Verify taxonomy guardrails and red-monster chromatic/spatial suppression prevent class 9 vs 11 cross-contamination."""
    scanner = MinimapScanner(templates_dir=str(tmp_path), config_dir=str(tmp_path))

    # Class taxonomy isolation checks
    assert MinimapScanner.is_minimap_class(9) is True
    assert MinimapScanner.is_minimap_class(11) is True
    assert MinimapScanner.is_minimap_class(0) is False
    assert MinimapScanner.is_world_3d_class(0) is True
    assert MinimapScanner.is_world_3d_class(10) is True
    assert MinimapScanner.is_world_3d_class(9) is False
    assert MinimapScanner.is_world_3d_class(11) is False

    s_sanitized = MinimapScanner.sanitize_minimap_detection({"class_id": 9})
    assert s_sanitized["class_id"] == 9
    assert s_sanitized["class_name"] == "minimap_sprout"
    assert s_sanitized["is_gatherable"] is True
    assert s_sanitized["is_hostile"] is False

    h_sanitized = MinimapScanner.sanitize_minimap_detection({"class_id": 11, "is_hostile": True})
    assert h_sanitized["class_id"] == 11
    assert h_sanitized["class_name"] == "minimap_monster_negative"
    assert h_sanitized["is_gatherable"] is False

    m_sanitized = MinimapScanner.sanitize_minimap_detection({"class_id": 11, "is_hostile": False})
    assert m_sanitized["class_id"] == 11
    assert m_sanitized["class_name"] == "monster_dot"
    assert m_sanitized["is_gatherable"] is False

    with pytest.raises(ValueError, match="Taksonomi"):
        MinimapScanner.sanitize_minimap_detection({"class_id": 10})

    # If a sprout patch is contaminated by hostile red monster pixels, _check_sprout_chromatic_density must reject it
    crop = _make_base_minimap_crop()
    sx, sy = _paste_green_3_leaf_sprout(crop, 120, 90)
    _draw_red_monster_dot(crop, sx, sy, radius=3)
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    patch_hsv = hsv_crop[sy - 8 : sy + 9, sx - 8 : sx + 9]
    ok_chrom, _ = scanner._check_sprout_chromatic_density(patch_hsv, scale=1.0)
    assert ok_chrom is False, "Patch containing hostile red monster pixels must be rejected as a sprout"
