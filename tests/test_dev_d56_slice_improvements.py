# -*- coding: utf-8 -*-
"""D56 DEV tests: performance improvements in the monster-discriminator slice.

Locks in the behaviour of:
- `MinimapSproutMonsterDiscriminator`: shared morphology kernel, cached
  annular minimap mask, lazy per-ROI white measurement in
  `_compute_local_ratios` (no full-frame white inRange per call).
- `DiamondTipMonsterDetector`: hoisted inRange threshold arrays, shared 3x3
  dilate kernel, cached rectangular structuring elements.

All tests assert identical *outputs* to the direct (unoptimised) computation,
so a caching/hoisting regression fails here first.
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset_factory.minimap_sprout_monster_discriminator import (  # noqa: E402
    MinimapSproutMonsterDiscriminator,
)
from dataset_factory.monster_diamond_detector import (  # noqa: E402
    DiamondTipMonsterDetector,
    _rect_struct_elem,
)


# ----------------------------------------------------------------------
# Shared synthetic fixtures
# ----------------------------------------------------------------------
def _minimap_frame() -> np.ndarray:
    """400x400 dark minimap frame with one red monster dot and one green blob."""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    img[:] = (20, 20, 25)
    cv2.circle(img, (200, 200), 90, (40, 40, 45), -1)
    cv2.circle(img, (300, 120), 7, (255, 255, 255), -1)
    cv2.circle(img, (300, 120), 5, (0, 0, 255), -1)
    cv2.circle(img, (110, 280), 8, (255, 255, 255), -1)
    cv2.circle(img, (110, 280), 6, (0, 180, 0), -1)
    cv2.circle(img, (120, 110), 6, (255, 255, 255), -1)
    cv2.circle(img, (120, 110), 4, (0, 0, 255), -1)
    return img


def _nameplate_frame() -> np.ndarray:
    """1920x1080 dark frame with a yellow nameplate + cyan tips + red badge."""
    img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    img[:] = (25, 25, 30)
    cv2.putText(img, "Yellow-Maned Zait", (760, 300), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "<", (700, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 220, 0), 3, cv2.LINE_AA)
    cv2.putText(img, ">", (1130, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 220, 0), 3, cv2.LINE_AA)
    cv2.rectangle(img, (940, 315), (980, 340), (0, 0, 255), -1)
    return img


def _nameplate_roi() -> np.ndarray:
    roi = np.zeros((40, 260, 3), dtype=np.uint8)
    roi[:] = (25, 25, 30)
    cv2.putText(roi, "Zait", (60, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 215, 255), 2, cv2.LINE_AA)
    roi[:, :12] = (255, 220, 0)
    roi[:, -12:] = (255, 220, 0)
    roi[30:, 110:150] = (0, 0, 255)
    return roi


# ----------------------------------------------------------------------
# Discriminator: annular-mask cache
# ----------------------------------------------------------------------
def test_cached_annular_mask_matches_fresh_build() -> None:
    disc = MinimapSproutMonsterDiscriminator()
    for shape, center, radius in [
        ((400, 400), (200, 200), 90.0),
        ((400, 400), (200, 200), 60.0),
        ((480, 640), (320, 240), 110.0),
    ]:
        fresh = disc.build_annular_minimap_mask(
            shape, center_xy=center, minimap_radius_px=radius
        )
        cached = disc._cached_annular_mask(
            shape[0], shape[1], center[0], center[1], radius, 14
        )
        assert np.array_equal(cached, fresh), f"cache mismatch for {shape}/{radius}"
        # Second lookup must return the identical (shared) array.
        assert disc._cached_annular_mask(
            shape[0], shape[1], center[0], center[1], radius, 14
        ) is cached


def test_public_mask_mutation_cannot_poison_cache() -> None:
    disc = MinimapSproutMonsterDiscriminator()
    frame = _minimap_frame()
    before = disc.discriminate(frame).to_dict()
    # Mutate the fresh array the public API returns; the internal cache must
    # be unaffected because build_annular_minimap_mask() never returns it.
    public = disc.build_annular_minimap_mask(frame.shape)
    public[:, :] = 0
    after = disc.discriminate(frame).to_dict()
    assert before == after
    assert len(after["monsters"]) >= 1
    assert len(after["sprouts"]) >= 1


def test_discriminate_stable_across_repeats_and_geometries() -> None:
    disc = MinimapSproutMonsterDiscriminator()
    frame = _minimap_frame()
    first = disc.discriminate(frame).to_dict()
    for _ in range(3):
        assert disc.discriminate(frame).to_dict() == first
    # Alternating geometries must not leak a stale cached mask into results.
    for center, radius in [((200, 200), 60.0), ((200, 200), 90.0), ((200, 200), 60.0)]:
        again = disc.discriminate(
            frame, center_xy=center, minimap_radius_px=radius
        ).to_dict()
        assert disc.discriminate(
            frame, center_xy=center, minimap_radius_px=radius
        ).to_dict() == again
    assert disc.discriminate(frame).to_dict() == first


# ----------------------------------------------------------------------
# Discriminator: lazy white-ratio measurement
# ----------------------------------------------------------------------
def test_compute_local_ratios_white_matches_full_frame_mask() -> None:
    disc = MinimapSproutMonsterDiscriminator()
    frame = _minimap_frame()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, disc.sprout_hsv_lower, disc.sprout_hsv_upper)
    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, disc.monster_red_lower1, disc.monster_red_upper1),
        cv2.inRange(hsv, disc.monster_red_lower2, disc.monster_red_upper2),
    )
    white_mask = cv2.inRange(hsv, disc.white_hsv_lower, disc.white_hsv_upper)
    for bbox in [(293, 113, 14, 14), (102, 272, 16, 16), (0, 0, 20, 20), (390, 390, 10, 10)]:
        green_r, red_r, white_r = disc._compute_local_ratios(
            green_mask, red_mask, hsv, bbox, pad=2
        )
        x, y, bw, bh = bbox
        x0, y0 = max(0, x - 2), max(0, y - 2)
        x1, y1 = min(400, x + bw + 2), min(400, y + bh + 2)
        total = max(1, (x1 - x0) * (y1 - y0))
        core = max(1, bw * bh)
        assert green_r == float(np.count_nonzero(green_mask[y:y + bh, x:x + bw])) / core
        assert red_r == float(np.count_nonzero(red_mask[y:y + bh, x:x + bw])) / core
        # Byte-identical white count to slicing the full-frame white mask.
        assert white_r == float(np.count_nonzero(white_mask[y0:y1, x0:x1])) / total


# ----------------------------------------------------------------------
# Diamond detector: hoisted thresholds / kernels / structuring elements
# ----------------------------------------------------------------------
def test_precomputed_cyan_bounds_match_constructor_params() -> None:
    det = DiamondTipMonsterDetector()
    assert det._cyan_low.tolist() == [75, 65, 115]
    assert det._cyan_high.tolist() == [108, 255, 255]
    custom = DiamondTipMonsterDetector(cyan_hue_range=(80, 100), cyan_min_sat=70, cyan_min_val=120)
    assert custom._cyan_low.tolist() == [80, 70, 120]
    assert custom._cyan_high.tolist() == [100, 255, 255]


def test_rect_struct_elem_cache_returns_identical_element() -> None:
    elem = _rect_struct_elem(15, 7)
    assert _rect_struct_elem(15, 7) is elem
    assert np.array_equal(
        elem, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 7))
    )
    assert not np.array_equal(elem, _rect_struct_elem(9, 3))


def test_diamond_helpers_agree_with_direct_computation() -> None:
    det = DiamondTipMonsterDetector()
    # Hand-built HSV image: cyan blocks placed exactly inside the helper's
    # left/right windows and a red block inside its below-nameplate window,
    # so the expected outcome is computable by hand (no rendered-text
    # geometry involved). x=100, y=10, cw=100, ch=20, scale 1.0:
    #   tips windows: left x 44..110, right x 190..256, y 2..38
    #   badge window: x 102..198, y 28..40
    hsv = np.zeros((40, 300, 3), dtype=np.uint8)
    hsv[:, :] = (0, 0, 30)  # dark, non-cyan, non-red background
    hsv[2:7, 44:49] = (90, 200, 200)    # cyan left tip (25 px)
    hsv[2:7, 190:195] = (90, 200, 200)  # cyan right tip (25 px)
    hsv[28:38, 120:180] = (0, 200, 200)  # red badge (600 px)
    has_tips, left_px, right_px = det._check_flanking_diamond_tips(
        hsv, 100, 10, 100, 20, 1.0, 1.0
    )
    assert has_tips is True
    assert left_px == 25 and right_px == 25
    # The hoisted bounds must agree with a direct inRange on the same ROIs.
    c_low = np.array([75, 65, 115], dtype=np.uint8)
    c_high = np.array([108, 255, 255], dtype=np.uint8)
    assert int(np.count_nonzero(cv2.inRange(hsv[2:38, 44:110], c_low, c_high))) == left_px
    assert int(np.count_nonzero(cv2.inRange(hsv[2:38, 190:256], c_low, c_high))) == right_px
    has_badge, red_px = det._check_red_agro_badge_or_bar(
        hsv, 100, 10, 100, 20, 1.0, 1.0
    )
    assert has_badge is True
    assert red_px == 600


def test_diamond_detect_deterministic_across_repeated_calls() -> None:
    det = DiamondTipMonsterDetector()
    frame = _nameplate_frame()
    first = det.detect(frame)
    assert len(first) == 1
    assert first[0]["class_id"] == 10
    for _ in range(3):
        assert det.detect(frame) == first
    assert det.detect(frame, verify_with_hflip=True) == first


def test_inspect_nameplate_roi_stable_and_detects_tips() -> None:
    det = DiamondTipMonsterDetector()
    roi = _nameplate_roi()
    first = det.inspect_nameplate_roi(roi)
    assert first["is_monster_nameplate"] is True
    assert first["has_diamond_tips"] is True
    assert first["has_agro_badge"] is True
    for _ in range(3):
        assert det.inspect_nameplate_roi(roi) == first
