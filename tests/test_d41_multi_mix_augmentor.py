# -*- coding: utf-8 -*-
"""D41 (augment-mix) regression tests.

Locks the D41 performance rework in dataset_factory/synthetic_augmentor.py
(LUT-baked photometrics, cv2 fog blend, half-resolution bloom, scalar
_clamp) and the _one_box trust simplification in
dataset_factory/multi_mix.py. The photometric tests compare against the
pre-D41 formulas reproduced inline, so a bit-exact (or, for bloom,
tightly bounded) rework cannot silently drift the training pixels.
"""

from __future__ import annotations

import math
import random

import cv2
import numpy as np
import pytest

from dataset_factory.multi_mix import _one_box, fuse_sources
from dataset_factory.synthetic_augmentor import SyntheticYOLOAugmentor


@pytest.fixture(scope="module")
def frame():
    return np.random.default_rng(0).integers(0, 255, (72, 96, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def anns():
    return [
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.9},
        {"class_id": 4, "cx": 0.25, "cy": 0.63, "w": 0.05, "h": 0.09, "confidence": 0.89},
    ]


def _old_day_night(frame, gamma, gains):
    """Pre-D41 formula: uint8 gamma LUT, float32 frame, broadcast gains."""
    inv_gamma = 1.0 / max(1e-4, gamma)
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    adjusted = cv2.LUT(frame, lut).astype(np.float32)
    g = np.array(gains, dtype=np.float32).reshape((1, 1, 3))
    return np.clip(adjusted * g, 0.0, 255.0).astype(np.uint8)


def test_day_night_gain_path_matches_old_math(frame):
    """Gain-baked LUT == old float32 broadcast path, bit for bit."""
    aug = SyntheticYOLOAugmentor(seed=42)
    out = aug.augment_day_night_lighting(frame, gamma=1.2, color_temp="night_cool")
    ref = random.Random(42)
    gains = (ref.uniform(1.08, 1.18), ref.uniform(0.95, 1.02), ref.uniform(0.84, 0.93))
    assert out.dtype == np.uint8 and out.shape == frame.shape
    np.testing.assert_array_equal(out, _old_day_night(frame, 1.2, gains))


def test_day_night_neutral_matches_gamma_lut(frame):
    aug = SyntheticYOLOAugmentor(seed=7)
    out = aug.augment_day_night_lighting(frame, gamma=1.3, color_temp="neutral")
    inv_gamma = 1.0 / 1.3
    lut = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    np.testing.assert_array_equal(out, cv2.LUT(frame, lut))


def _old_hsv(frame, h_delta, s_scale, v_scale):
    """Pre-D41 formula: int32 HSV frame, per-channel arithmetic."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + h_delta) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_scale, 0, 255).astype(np.int32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_scale, 0, 255).astype(np.int32)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def test_hsv_jitter_matches_old_math(frame):
    """LUT HSV shift == old int32 path, bit for bit."""
    aug = SyntheticYOLOAugmentor(seed=7)
    out = aug.augment_hsv_jitter(frame)
    ref = random.Random(7)
    h_delta = ref.randint(-8, 8)
    s_scale = ref.uniform(0.80, 1.25)
    v_scale = ref.uniform(0.90, 1.12)
    np.testing.assert_array_equal(out, _old_hsv(frame, h_delta, s_scale, v_scale))


def _old_fog_bloom(frame, fog_intensity, bloom_strength, fog_color):
    """Pre-D41 formula: numpy broadcast fog + full-resolution bloom."""
    h, w = frame.shape[:2]
    vertical_profile = np.linspace(1.0, 0.35, h, dtype=np.float32).reshape((h, 1, 1))
    alpha_map = vertical_profile * fog_intensity
    img_f = frame.astype(np.float32)
    fogged = img_f * (1.0 - alpha_map) + fog_color.reshape((1, 1, 3)) * alpha_map
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, bright_mask = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
    bright_regions = cv2.bitwise_and(frame, frame, mask=bright_mask)
    bloom_layer = cv2.GaussianBlur(bright_regions, (0, 0), sigmaX=15.0, sigmaY=15.0).astype(np.float32)
    return np.clip(fogged + bloom_layer * bloom_strength, 0.0, 255.0).astype(np.uint8)


def test_fog_bloom_within_tolerance_of_full_res_reference(frame):
    """Half-res bloom + cv2 fog blend stay within 2 LSB of the old formula."""
    aug = SyntheticYOLOAugmentor(seed=123)
    out = aug.augment_atmospheric_fog_and_bloom(frame, fog_intensity=0.2, bloom_strength=0.15)
    fog_colors = [
        np.array([228.0, 222.0, 212.0], dtype=np.float32),
        np.array([235.0, 232.0, 225.0], dtype=np.float32),
        np.array([218.0, 225.0, 232.0], dtype=np.float32),
    ]
    fog_color = fog_colors[random.Random(123).randrange(len(fog_colors))]
    expected = _old_fog_bloom(frame, 0.2, 0.15, fog_color)
    assert out.dtype == np.uint8 and out.shape == frame.shape
    diff = np.abs(out.astype(np.int16) - expected.astype(np.int16))
    assert diff.max() <= 2, f"max abs diff {diff.max()}"
    assert diff.mean() <= 0.5, f"mean abs diff {diff.mean()}"


def test_clamp_yolo_box_values():
    clamp = SyntheticYOLOAugmentor._clamp_yolo_box
    assert clamp(0.5, 0.5, 0.2, 0.2) == pytest.approx((0.5, 0.5, 0.2, 0.2))
    # Box hanging off the left edge is clipped to [0, 0.1], not dropped.
    assert clamp(-0.1, 0.5, 0.4, 0.2) == pytest.approx((0.05, 0.5, 0.1, 0.2))
    # Box larger than the frame collapses to the full frame.
    assert clamp(0.5, 0.5, 2.0, 2.0) == pytest.approx((0.5, 0.5, 1.0, 1.0))
    assert clamp(0.0, 0.0, 0.0, 0.0) == pytest.approx((0.0, 0.0, 0.0, 0.0))
    # NaN propagates exactly like the old np.clip path (unordered input):
    # the NaN centre/width stay NaN, the unaffected axis stays numeric.
    out = clamp(float("nan"), 0.5, 0.2, 0.2)
    assert math.isnan(out[0]) and math.isnan(out[2])
    assert out[1] == pytest.approx(0.5) and out[3] == pytest.approx(0.2)


def test_extract_yolo_cxcywh_schemas():
    extract = SyntheticYOLOAugmentor._extract_yolo_cxcywh
    assert extract({"cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}) == pytest.approx((0.5, 0.5, 0.2, 0.2))
    assert extract({"yolo_bbox": [0.5, 0.5, 0.2, 0.2]}) == pytest.approx((0.5, 0.5, 0.2, 0.2))
    # Normalized xyxy -> cxcywh.
    assert extract({"bbox_xyxy": [0.4, 0.4, 0.6, 0.6]}) == pytest.approx((0.5, 0.5, 0.2, 0.2))
    # Pixel xyxy on a 100x100 frame.
    assert extract({"bbox_xyxy": [40, 40, 60, 60]}, img_w=100, img_h=100) == pytest.approx(
        (0.5, 0.5, 0.2, 0.2)
    )
    # Per-coordinate unit detection (F6): only a coordinate that is itself
    # > 1.0 is divided by the image size -- a pixel cx (960) no longer drags
    # the other coordinates down with it.
    assert extract({"cx": 960, "cy": 0.5, "w": 0.2, "h": 0.2}) == pytest.approx(
        (0.5, 0.5, 0.2, 0.2)
    )
    # ...and a coordinate that is STILL out of [0, 1] afterwards is
    # REJECTED, never clamped: cx=1.5 -> 1.5/1920, but cy=-0.5 stays
    # -0.5, so _validate_normalized_box raises ValueError (the old
    # all-or-nothing heuristic collapsed the whole box to ~0 instead).
    with pytest.raises(ValueError):
        extract({"cx": 1.5, "cy": -0.5, "w": 0.2, "h": 0.2})
    with pytest.raises(ValueError):
        extract({"class_id": 0})


def test_update_annotation_coords_pixel_math():
    """Pixel xyxy round-trip through the flip still lands on exact pixels."""
    aug = SyntheticYOLOAugmentor(seed=3)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    ann = {"class_id": 0, "bbox_xyxy": [10, 20, 30, 40], "confidence": 0.9}
    _, out = aug.augment_horizontal_flip(frame, [ann])
    assert out[0]["bbox_xyxy"] == [70, 20, 90, 40]
    assert out[0]["cx"] == pytest.approx(0.8)
    assert out[0]["synthetic"] is True


def _mk(**kw):
    d = {"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.9}
    d.update(kw)
    return d


# (raw, default_class, source, expected_trusted) -- characterization of the
# pre-D41 _one_box trust decision; the D41 simplification must reproduce
# every row (the old block double-called is_trusted_annotation and its
# NOTE (Y1 tarama, v11): the 5 legacy 7/8 gather-seed rows below no longer
# expect True -- multi_mix._one_box + label_trust._is_hand_or_gather_seed
# seed only canonical cid 13 (skill_ui_prompt); a 7/8 box with an F/E trace
# is a material box, not a prompt. Their cid-13 mirrors expected True until
# P1-12 (K-34): a seed alone is no longer a human decision, so the cid-13
# mirrors expect False too (see the rows at the bottom of the matrix).
# second OR-branch could never change the outcome). Note the two False
# rows for approved/human_approved: _one_box does NOT copy those flags
# into the box dict (only hand_label is copied), so they cannot make a
# box trusted here -- that is the pre-existing behaviour, locked as-is.
TRUST_MATRIX = [
    (_mk(class_id=7, gather_key="f", trusted=True), None, "yolo3d", False),  # Y1: legacy 7 seed, no longer trusted
    (_mk(class_id=8, gather_key="e", trusted=True), None, "yolo3d", False),  # Y1: legacy 8 seed, no longer trusted
    (_mk(class_id=9, gather_key="f", trusted=True), 9, "minimap_sprout", False),
    (_mk(class_id=9, gather_key="f", trusted=True), None, "yolo3d", False),
    (_mk(class_id=0, hand_label=True, confidence=0.3), None, "yolo3d", True),
    (_mk(class_id=0, trusted=True, confidence=0.9), None, "yolo3d", False),
    (_mk(class_id=7, trusted=True, confidence=0.9, source="f_gather_prompt"), None, "yolo3d", False),  # Y1: _real_prompt needs cid 13
    (_mk(class_id=7, trusted=True, confidence=0.9, source="prompt_candidate"), None, "yolo3d", False),
    (_mk(class_id=0, approved=True), None, "yolo3d", False),
    (_mk(class_id=0, human_approved=True), None, "yolo3d", False),
    (_mk(class_id=0, trusted=True, source="chromatic_salient_3d_resource"), None, "yolo3d", False),
    (_mk(class_id=0, trusted=True, source="omni_foo"), None, "yolo3d", False),
    (_mk(class_id=0, trusted=True, hand_label=True, gather_key="e"), None, "yolo3d", True),
    (_mk(class_id=11, gather_key="e", trusted=True), 11, "minimap_monster", False),
    (_mk(class_id=15, gather_key="f", trusted=True), None, "yolo3d", False),
    (_mk(class_id=7, key="F", trusted=True), None, "yolo3d", False),  # Y1: legacy 7 seed, no longer trusted
    (_mk(class_id=7, key="x", trusted=True), None, "yolo3d", False),
    (_mk(class_id=0, confidence=0.2), None, "yolo3d", False),
    (_mk(gather_key="f", trusted=True), 9, "minimap_sprout", False),
    (_mk(class_id=8, source="e_floating", trusted=True), None, "yolo3d", False),  # Y1: legacy 8 seed, no longer trusted
    # P1-12 (K-34): a seed alone is NOT a human decision anymore — the
    # same five cid-13 prompts are untrusted without hand_label /
    # human_approved. (Pre-P1-12 they were trusted via the seed rule.)
    (_mk(class_id=13, gather_key="f", trusted=True), None, "yolo3d", False),
    (_mk(class_id=13, gather_key="e", trusted=True), None, "yolo3d", False),
    (_mk(class_id=13, trusted=True, confidence=0.9, source="f_gather_prompt"), None, "yolo3d", False),
    (_mk(class_id=13, key="F", trusted=True), None, "yolo3d", False),
    (_mk(class_id=13, source="e_floating", trusted=True), None, "yolo3d", False),
]


@pytest.mark.parametrize("raw,default_class,source,expected", TRUST_MATRIX)
def test_one_box_trust_matrix(raw, default_class, source, expected):
    box = _one_box(dict(raw), default_class, source)
    assert box is not None
    assert box["trusted"] is expected, f"{raw} -> {box['trusted']}, expected {expected}"


def test_fuse_sources_prompt_without_coords_creates_no_box():
    fused = fuse_sources({"gather_prompt": {"gather_key": "f", "real_prompt": True}})
    assert fused["annotations"] == []
    assert fused["sources"] == []


def test_fuse_sources_basic():
    # NOTE (F-0882, R59 commit dbcc6f8): F/E gather prompts are class 13
    # (skill_ui_prompt), not legacy 7 (herb_lumina) / 8 (ore_obsidian).
    # This test was written when multi_mix._CLASS_F was 7 (dev D41,
    # 2a632a3); the canonical remap superseded that value — the same
    # scenario is locked to 13 by
    # tests/test_multi_mix_canonical.py::test_real_f_prompt_is_class_13_and_trusted
    # (passes in v2), and class_taxonomy.py maps f_gather_prompt 7 -> 13.
    fused = fuse_sources(
        {
            "yolo": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.9}],
            "gather_prompt": {"cx": 0.5, "cy": 0.6, "gather_key": "f", "real_prompt": True},
        }
    )
    assert fused["sources"] == ["yolo", "gather_prompt"]
    assert len(fused["annotations"]) == 2
    prompt = fused["annotations"][1]
    assert prompt["class_id"] == 13 and prompt["trusted"] is True and prompt["gather_key"] == "f"


def test_no_unused_np_rng_state():
    """The numpy RNG created in __init__ was never read anywhere in the repo."""
    assert not hasattr(SyntheticYOLOAugmentor(seed=1), "np_rng")


def test_generate_variants_end_to_end(frame, anns):
    aug = SyntheticYOLOAugmentor(seed=42)
    variants = aug.generate_augmented_variants(frame, anns, count=5)
    assert len(variants) == 5
    suffixes = set()
    for suffix, v_frame, v_anns in variants:
        assert v_frame.shape == frame.shape and v_frame.dtype == np.uint8
        assert suffix not in suffixes
        suffixes.add(suffix)
        assert len(v_anns) > 0
        assert SyntheticYOLOAugmentor.validate_annotations_in_bounds(v_anns, img_w=96, img_h=72)
        assert all(a.get("synthetic") is True for a in v_anns)
        assert all(a.get("source") == "synthetic_augmentor" for a in v_anns)
