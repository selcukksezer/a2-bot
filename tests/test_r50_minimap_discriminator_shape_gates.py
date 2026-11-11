# -*- coding: utf-8 -*-
"""R50 regression tests for dataset_factory/minimap_sprout_monster_discriminator.py.

Pins the three R50 fixes (S-2301, S-2302, S-2303) with synthetic patches only:

- S-2301: classify_patch must not accept a shapeless green square as a
  3-leaf sprout (class 9); a genuine saturated-green 3-leaf shape still
  classifies as 9.
- S-2302: discriminate_patch monster confidence must use the calibrated
  0.40 base (same structure as discriminate()), not 0.55 + 0.35*circ.
- S-2303: compute_circularity must score a filled square below the 0.65
  monster gate (a square fills exactly 2/pi of its enclosing circle)
  while real rasterized circles (r>=3) still clear it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.minimap_sprout_monster_discriminator import (
    CLASS_ID_MINIMAP_MONSTER,
    CLASS_ID_MINIMAP_SPROUT,
    MinimapSproutMonsterDiscriminator,
)

_GREEN = (40, 200, 40)  # saturated green, BGR
_RED = (20, 20, 245)  # saturated red, BGR
_BG = (52, 62, 56)  # dark slate minimap background, BGR


def _disc() -> MinimapSproutMonsterDiscriminator:
    return MinimapSproutMonsterDiscriminator()


def _green_square_with_white_outline(size: int = 20) -> np.ndarray:
    patch = np.full((size, size, 3), _BG, dtype=np.uint8)
    patch[2 : size - 2, 2 : size - 2] = _GREEN
    patch[2, 2 : size - 2] = (255, 255, 255)
    return patch


def _saturated_three_leaf_sprout(size: int = 20) -> np.ndarray:
    img = np.full((size, size, 3), _BG, dtype=np.uint8)
    leaf = (0, 210, 0)
    cv2.ellipse(img, (10, 6), (3, 5), 0, 0, 360, leaf, -1)
    cv2.ellipse(img, (6, 11), (4, 3), -30, 0, 360, leaf, -1)
    cv2.ellipse(img, (14, 11), (4, 3), 30, 0, 360, leaf, -1)
    cv2.line(img, (10, 11), (10, 16), (0, 130, 0), 2)
    img[1, 10] = (255, 255, 255)
    img[10, 1] = (255, 255, 255)
    return img


def _red_shape_patch(draw, size: int = 20) -> np.ndarray:
    patch = np.full((size, size, 3), _BG, dtype=np.uint8)
    draw(patch)
    return patch


def _contour_of(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert contours, "expected at least one contour"
    return max(contours, key=cv2.contourArea)


def test_s2301_green_square_is_not_a_sprout_but_three_leaf_shape_is() -> None:
    d = _disc()
    # S-2301 probe: 20x20 green square classified as 9 before the fix.
    assert d.classify_patch(_green_square_with_white_outline()) is None
    assert d.discriminate_patch(_green_square_with_white_outline())["label"] is None
    # Positive control: a genuine saturated-green 3-leaf shape still passes.
    assert d.classify_patch(_saturated_three_leaf_sprout()) == CLASS_ID_MINIMAP_SPROUT
    assert (
        d.discriminate_patch(_saturated_three_leaf_sprout())["label"]
        == "minimap_sprout"
    )


@pytest.mark.parametrize("side", [3, 6, 10, 20])
def test_s2303_filled_square_circularity_below_monster_gate(side: int) -> None:
    d = _disc()
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(mask, (5, 5), (5 + side - 1, 5 + side - 1), 255, -1)
    circ = d.compute_circularity(_contour_of(mask))
    assert circ < d.min_monster_circularity, f"{side}x{side} square circ={circ}"


@pytest.mark.parametrize("radius", [3, 4, 5, 8])
def test_s2303_real_circle_circularity_still_clears_monster_gate(radius: int) -> None:
    d = _disc()
    mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.circle(mask, (20, 20), radius, 255, -1)
    circ = d.compute_circularity(_contour_of(mask))
    assert circ >= d.min_monster_circularity, f"r={radius} circle circ={circ}"
    # A real circle must score strictly above a square of similar size.
    square_mask = np.zeros((40, 40), dtype=np.uint8)
    cv2.rectangle(square_mask, (5, 5), (5 + 2 * radius - 1, 5 + 2 * radius - 1), 255, -1)
    assert circ > d.compute_circularity(_contour_of(square_mask))


def test_s2302_and_s2303_red_square_is_not_a_monster_dot() -> None:
    d = _disc()
    patch = _red_shape_patch(
        lambda p: p.__setitem__((slice(7, 13), slice(7, 13)), _RED)
    )
    out = d.discriminate_patch(patch)
    assert out["label"] is None
    assert out["reason"] == "red_non_circular"
    assert out["confidence"] == 0.0


def test_s2302_red_circle_monster_confidence_is_calibrated() -> None:
    d = _disc()
    patch = _red_shape_patch(lambda p: cv2.circle(p, (10, 10), 4, _RED, -1))
    out = d.discriminate_patch(patch)
    assert out["label"] == "minimap_monster_dot"
    assert out["class_id"] == CLASS_ID_MINIMAP_MONSTER
    # Old formula gave 0.818 for this patch; calibrated base-0.40 formula
    # must stay clearly below 0.70 for an ordinary r=4 dot.
    assert 0.50 <= out["confidence"] < 0.70, out["confidence"]
