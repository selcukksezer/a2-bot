# -*- coding: utf-8 -*-
"""D59 dev-improvement tests: stem-material validator batch performance
paths and six-material discriminator constructor/histogram improvements.

These tests lock behaviour, not speed: the speed gains are measured with
~/workspace/bench_d59.py (before/after numbers in dev-results/D59.json).
Every expected value below was recorded from the pre-change code, so a
regression in the refactored paths fails here first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.six_material_discriminator import SixMaterialDiscriminator
from dataset_factory.stem_material_validator import (
    StemMaterialValidator,
    _cached_stem_materials,
)


# ---------------------------------------------------------------------------
# StemMaterialValidator: normalize_stem fast path
# ---------------------------------------------------------------------------

NORMALIZE_CASES = [
    ("boot_001_aria_node", "boot_001_aria_node"),
    ("  MELA  ", "mela"),
    ("img.png.json", "img"),
    ("x.png.json.txt", "x"),
    ("UPPER.PNG", "upper"),
    ("a/b/c_aria.png", "c_aria"),
    ("foo/", "foo"),  # trailing slash: Path drops the empty tail
    ("a/b/", "b"),
    ("a\\b.png", "a\\b"),  # on Linux "\\" is NOT a separator for Path
    ("C:\\x\\aria.png", "c:\\x\\aria"),
    ("", ""),
    ("   ", ""),
]


@pytest.mark.parametrize("raw,expected", NORMALIZE_CASES)
def test_normalize_stem_matches_pathlib_behaviour(raw: str, expected: str) -> None:
    assert StemMaterialValidator.normalize_stem(raw) == expected


def test_normalize_stem_none_and_path_input() -> None:
    assert StemMaterialValidator.normalize_stem(None) == ""
    assert (
        StemMaterialValidator.normalize_stem(Path("/tmp/boot_01_aria.png"))
        == "boot_01_aria"
    )


# ---------------------------------------------------------------------------
# StemMaterialValidator: cached stem extraction
# ---------------------------------------------------------------------------

STEM_CASES = [
    ("boot_001_aria_node", {"aria"}),
    ("ref_mela_02", {"mela"}),
    ("targena", {"targena"}),
    ("boot01aria", {"aria"}),
    ("barista", set()),  # "aria" inside "barista" must NOT match
    ("orichalcum02", {"orichalcum"}),
    ("crystal_blood", {"blood"}),
    ("blood_crystal", {"blood"}),
    ("boot_03_aria_gather", {"aria"}),
    ("img.png.json", set()),
    ("a/b/c_aria.png", {"aria"}),
    ("  MELA  ", {"mela"}),
    ("", set()),
    ("wood_log_yggdrasil", {"wood"}),
]


@pytest.mark.parametrize("stem,expected", STEM_CASES)
def test_extract_stem_materials_values(stem: str, expected: set) -> None:
    assert StemMaterialValidator.extract_stem_materials(stem) == expected


def test_extract_stem_materials_uses_cache_and_returns_copies() -> None:
    _cached_stem_materials.cache_clear()
    stem = "boot_001_aria_node"
    first = StemMaterialValidator.extract_stem_materials(stem)
    assert first == {"aria"}
    info_after_first = _cached_stem_materials.cache_info()
    assert info_after_first.misses >= 1
    second = StemMaterialValidator.extract_stem_materials(stem)
    assert second == {"aria"}
    info_after_second = _cached_stem_materials.cache_info()
    assert info_after_second.hits > info_after_first.hits
    # Mutating the returned set must not poison the cache.
    second.add("mela")
    third = StemMaterialValidator.extract_stem_materials(stem)
    assert third == {"aria"}


# ---------------------------------------------------------------------------
# StemMaterialValidator: box fallback (single compiled alternation)
# ---------------------------------------------------------------------------

BOX_CASES = [
    ("monster_3d", set()),
    ("orichalcum02", {"orichalcum"}),
    ("crystal_blood", {"blood"}),
    ("gem_sapphire", {"crystal"}),
    ("unknown_thing", set()),
    ("ironwood", set()),  # "wood" mid-token must NOT match
    ("hardwood", set()),
    ("crystalblood", set()),
    ("bloodcrystal", set()),
    ("orichalcum02wood", {"orichalcum", "wood"}),
    ("wood02", {"wood"}),
    ("ariax", set()),
    ("xaria", set()),
    ("boot01aria", {"aria"}),
    ("sapphire_gem", {"crystal"}),
    ("mithril_ore", {"orichalcum"}),
    ("titanium02", {"orichalcum"}),
    ("edana", {"edana"}),
    ("lumina_x", {"lumina"}),
    ("obsidian02", {"obsidian"}),
    ("tree", {"wood"}),
    ("log", {"wood"}),
    ("pine", {"wood"}),
    ("oak_tree", {"wood"}),
    ("crystal", {"crystal"}),
    ("blood", {"blood"}),
    ("crystal_blood_02", {"blood"}),
    ("aether_crystal", {"crystal"}),
    ("moonstone", {"crystal"}),
]


@pytest.mark.parametrize("class_name,expected", BOX_CASES)
def test_extract_box_materials_values(class_name: str, expected: set) -> None:
    assert StemMaterialValidator.extract_box_materials(class_name) == expected


def test_extract_box_materials_crystal_blood_beats_crystal_prefix() -> None:
    # "crystal_blood" must resolve to blood only: the longer token wins
    # over its "crystal" prefix (the old per-token loop skipped "crystal"
    # via the has_blood_crystal guard; the alternation must do the same).
    assert StemMaterialValidator.extract_box_materials("crystal_blood") == {"blood"}
    assert StemMaterialValidator.extract_box_materials("blood_crystal") == {"blood"}


# ---------------------------------------------------------------------------
# StemMaterialValidator: batch paths == per-box paths
# ---------------------------------------------------------------------------

_BOX_NAMES = [
    "herb_aria",
    "ore_orichalcum",
    "herb_mela",
    "gem_crystal",
    "tree_log",
    "herb_targena",
    "ore_obsidian",
    "herb_edana",
    "monster_3d",
    "unknown_thing",
]


def _batch_boxes(n: int = 200) -> list:
    boxes = []
    for i in range(n):
        name = _BOX_NAMES[i % len(_BOX_NAMES)]
        box = {
            "class_id": i % 13,
            "class_name": name,
            "confidence": 0.37 + (i % 7) * 0.09,
            "trusted": True,
            "weak": False,
        }
        if i % 17 == 0:  # protected hand label: must never be demoted
            box["hand_label"] = True
        if i % 23 == 0:  # zero confidence stays zero, never inflated
            box["confidence"] = 0.0
        boxes.append(box)
    return boxes


def test_validate_boxes_matches_per_box_validation() -> None:
    validator = StemMaterialValidator()
    stem = "boot_001_aria_node"
    batch = validator.validate_boxes(stem, [dict(b) for b in _batch_boxes()])
    per_box = [
        validator.validate_box(stem, dict(b), in_place=False)
        for b in _batch_boxes()
    ]
    assert len(batch) == len(per_box)
    for got, want in zip(batch, per_box):
        assert got == want


def test_validate_boxes_empty_stem_falls_back_to_box_stem() -> None:
    validator = StemMaterialValidator()
    boxes = [
        {"stem": "boot_001_aria_node", "class_id": 3, "class_name": "ore_orichalcum",
         "confidence": 0.9, "trusted": True, "weak": False},
        {"stem": "ref_mela_02", "class_id": 3, "class_name": "ore_orichalcum",
         "confidence": 0.9, "trusted": True, "weak": False},
    ]
    batch = validator.validate_boxes("", [dict(b) for b in boxes])
    per_box = [validator.validate_box("", dict(b), in_place=False) for b in boxes]
    assert batch == per_box
    assert batch[0]["trusted"] is False  # aria stem vs orichalcum: conflict
    assert batch[1]["trusted"] is False  # mela stem vs orichalcum: conflict


def test_validate_review_payload_matches_per_box_evaluation() -> None:
    validator = StemMaterialValidator()
    stem = "boot_04_aria_field"
    payload = {
        "stem": stem,
        "trusted": [
            {"class_id": 0, "class_name": "herb_aria", "confidence": 0.92,
             "trusted": True, "weak": False},
            {"class_id": 3, "class_name": "ore_orichalcum", "confidence": 0.89,
             "trusted": True, "weak": False},
            {"class_id": 10, "class_name": "monster_3d", "confidence": 0.8,
             "trusted": True, "weak": False},
            "not-a-box",
        ],
        "weak": [
            {"class_id": 3, "class_name": "ore_orichalcum", "confidence": 0.9,
             "trusted": False, "weak": True},
            42,
        ],
    }
    updated = validator.validate_review_payload(payload, stem=stem, in_place=False)
    assert [b["class_name"] for b in updated["trusted"] if isinstance(b, dict)] == [
        "herb_aria",
        "monster_3d",
    ]
    assert "not-a-box" in updated["trusted"]
    weak_names = [b["class_name"] for b in updated["weak"] if isinstance(b, dict)]
    assert weak_names == ["ore_orichalcum", "ore_orichalcum"]
    assert all(b["confidence"] <= 0.45 for b in updated["weak"] if isinstance(b, dict))
    assert 42 in updated["weak"]


def test_unresolved_class_ids_still_conflict_on_material_stem() -> None:
    validator = StemMaterialValidator()
    # class_id 99 is outside the canonical taxonomy -> conflict on a material stem.
    assert validator.is_conflict("boot_001_aria_node", {"class_id": 99}) is True
    # class_id 10 (monster_3d) is a known non-material canonical class -> no conflict.
    assert validator.is_conflict("boot_001_aria_node", {"class_id": 10}) is False


# ---------------------------------------------------------------------------
# SixMaterialDiscriminator: constructor validation
# ---------------------------------------------------------------------------

INVALID_CTOR_KWARGS = [
    {"h_bins": 0},
    {"s_bins": -1},
    {"v_bins": 0},
    {"min_confidence": 5.0},
    {"min_confidence": -0.1},
    {"canny_low": 200, "canny_high": 50},
    {"canny_low": -1},
    {"canny_high": 300},
    {"noise_margin": 2.0},
    {"noise_margin": -0.1},
    {"specular_v_threshold": 999},
    {"specular_s_max": -5},
]


@pytest.mark.parametrize("kwargs", INVALID_CTOR_KWARGS)
def test_constructor_rejects_nonsense_params(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SixMaterialDiscriminator(**kwargs)


VALID_CTOR_KWARGS = [
    {"h_bins": 1, "s_bins": 1, "v_bins": 1},
    {"min_confidence": 1.0},
    {"min_confidence": 0.0},
    {"noise_margin": 0.0},
    {"noise_margin": 1.0},
    {"canny_low": 0, "canny_high": 255},
]


@pytest.mark.parametrize("kwargs", VALID_CTOR_KWARGS)
def test_constructor_accepts_boundary_values(kwargs: dict) -> None:
    SixMaterialDiscriminator(**kwargs)


# ---------------------------------------------------------------------------
# SixMaterialDiscriminator: vectorized reference histograms == loop oracle
# ---------------------------------------------------------------------------

def _loop_oracle_hist(
    h_bins: int,
    s_bins: int,
    v_bins: int,
    hue_range: tuple,
    sat_range: tuple,
    val_range: tuple,
) -> np.ndarray:
    """Pre-change per-bin Python loop, kept as the equivalence oracle."""

    def _channel(n: int, span: float, rng: tuple, sigma: float) -> np.ndarray:
        hist = np.zeros(n, dtype=np.float32)
        for i in range(n):
            center = (i + 0.5) * (span / n)
            lo, hi = rng
            if lo <= center <= hi:
                hist[i] = 1.0
            else:
                dist = min(abs(center - lo), abs(center - hi))
                hist[i] = float(np.exp(-0.5 * (dist / sigma) ** 2))
        hist /= max(float(np.sum(hist)), 1e-6)
        return hist

    h = _channel(h_bins, 180.0, hue_range, 12.0)
    s = _channel(s_bins, 256.0, sat_range, 25.0)
    v = _channel(v_bins, 256.0, val_range, 25.0)
    combined = np.concatenate([h, s, v]).astype(np.float32)
    return combined / max(float(np.linalg.norm(combined)), 1e-6)


@pytest.mark.parametrize("bins", [(18, 8, 8), (30, 10, 10), (1, 1, 1), (6, 4, 4)])
def test_reference_histograms_match_loop_oracle(bins: tuple) -> None:
    disc = SixMaterialDiscriminator(h_bins=bins[0], s_bins=bins[1], v_bins=bins[2])
    for name, profile in disc.profiles.items():
        oracle = _loop_oracle_hist(
            bins[0],
            bins[1],
            bins[2],
            profile.hue_range,
            profile.saturation_range,
            profile.value_range,
        )
        assert np.array_equal(oracle, profile.reference_histogram), name


# ---------------------------------------------------------------------------
# SixMaterialDiscriminator: classification behaviour unchanged
# ---------------------------------------------------------------------------

def _solid_hsv_patch(h: int, s: int, v: int, height: int = 80, width: int = 80) -> np.ndarray:
    hsv = np.zeros((height, width, 3), dtype=np.uint8)
    hsv[:, :, 0] = h
    hsv[:, :, 1] = s
    hsv[:, :, 2] = v
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


@pytest.mark.parametrize(
    "patch,expected",
    [
        ((85, 120, 200, 80, 80), "herb_aria"),
        # STALE-EXPECTATION FIX (R13b; G2-05 commit 62f17ce, F-1009):
        # D59 wrote these cases when the discriminator still carried the
        # old six (herb_mela / tree_log profiles). F-1009 aligned
        # MATERIAL_CLASSES with production MAIN_MATERIALS_6
        # (aria/edana/lumina/orichalcum/obsidian/crystal_blood), so
        # herb_mela / tree_log can no longer be returned at all.
        # Under the current scorer both solid patches score highest as
        # rock_noise (v2 log: got 'rock_noise'; hand-computed scores
        # ~0.696 / ~0.659, both >= 0.6; best material herb_lumina /
        # ore_orichalcum stays below at ~0.626 / ~0.530).
        ((25, 180, 200, 80, 80), "rock_noise"),
        ((25, 100, 110, 120, 50), "rock_noise"),  # tall aspect, bark brown
        ((0, 20, 90, 80, 80), "rock_noise"),
    ],
)
def test_discriminate_solid_patches(patch: tuple, expected: str) -> None:
    disc = SixMaterialDiscriminator()
    result = disc.discriminate(_solid_hsv_patch(*patch))
    assert result.predicted_class == expected
    assert result.confidence >= 0.6


def test_discriminate_unknown_when_nothing_can_match() -> None:
    # min_confidence=1.0 is unreachable: every input must be "unknown".
    disc = SixMaterialDiscriminator(min_confidence=1.0)
    result = disc.discriminate(_solid_hsv_patch(85, 120, 200))
    assert result.predicted_class == "unknown"
    assert result.is_valid_material is False
    assert result.is_noise is False


def test_filter_detections_keeps_material_drops_noise() -> None:
    disc = SixMaterialDiscriminator()
    img = _solid_hsv_patch(85, 120, 200, 120, 120)
    detections = [
        {"bbox": (10, 10, 60, 60), "id": "aria"},
        {"bbox": "not-a-bbox", "id": "bad"},
        {"bbox": (0, 0, 5, 5), "id": "tiny"},
    ]
    kept = disc.filter_detections(img, detections)
    kept_ids = [d["id"] for d in kept]
    assert "aria" in kept_ids
    assert "bad" not in kept_ids
    assert all(d["material_class"] == "herb_aria" for d in kept if d["id"] == "aria")
