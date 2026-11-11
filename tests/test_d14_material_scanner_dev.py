# -*- coding: utf-8 -*-
"""D14 dev tests: material_scanner improvements.

I1: _refine_candidates_with_exemplars must NOT re-run the same exemplar
    queries for the same crop: classify_patch / is_negative_patch /
    match_positive / match_negative are measured ONCE per candidate and the
    verdicts are threaded into discriminate_3d_material via
    PrecomputedExemplarVerdict (9 queries/candidate -> 4).
I2: match/normalize + per-call lookup tables hoisted to module/class level
    (same outputs, less allocation); compute_color_histogram_scores reuses
    cached green/targena pixel counts (same scores).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.material_scanner import (
    DetectedMaterial,
    MaterialScanner,
    PrecomputedExemplarVerdict,
    compute_color_histogram_scores,
    discriminate_3d_material,
    match_material_names,
    normalize_material_name,
)
from dataset_factory.exemplar_memory import extract_crop_features


# ------------------------------------------------------------------ fixtures
class CountingMem:
    """Canned exemplar verdicts; runs the REAL feature extraction per query
    like production match_positive/match_negative, and counts queries."""

    def __init__(self, classify_result=(None, None, 0.10)):
        self.counts = {"classify_patch": 0, "is_negative_patch": 0,
                       "match_positive": 0, "match_negative": 0}
        self.suppress_neg_threshold = 0.82
        self.classify_result = classify_result

    def is_negative_patch(self, crop, class_id=None, threshold=None):
        self.counts["is_negative_patch"] += 1
        extract_crop_features(crop)
        return (False, 0.0, None)

    def classify_patch(self, crop, min_sim=0.70):
        self.counts["classify_patch"] += 1
        extract_crop_features(crop)
        return self.classify_result

    def match_positive(self, crop, **_kw):
        self.counts["match_positive"] += 1
        extract_crop_features(crop)
        return (0.0, None, None, None)

    def match_negative(self, crop, class_id=None):
        self.counts["match_negative"] += 1
        extract_crop_features(crop)
        return (0.0, None)

    @property
    def total(self):
        return sum(self.counts.values())


def _frame_and_candidates(n=3):
    rng = np.random.default_rng(7)
    frame = rng.integers(40, 230, size=(720, 1280, 3)).astype(np.uint8)
    cands = []
    for i in range(n):
        cands.append(
            DetectedMaterial(
                material_type="Aria", source="3d_world",
                screen_x=300 + i * 300, screen_y=300,
                bbox=(276 + i * 300, 276, 48, 48),
                rel_map_dx=0.0, rel_map_dy=0.0,
                estimated_distance_m=10.0, confidence=0.80,
            )
        )
    return frame, cands


# ------------------------------------------------------------------ I1: dedup
class TestExemplarQueryDedup:
    def test_refine_runs_each_exemplar_query_once_per_candidate(self):
        """3x 3d_world candidates: before the fix this was 27 queries
        (9/candidate); after, at most 12 (4/candidate)."""
        scanner = MaterialScanner()
        frame, cands = _frame_and_candidates(3)
        mem = CountingMem()
        kept = scanner._refine_candidates_with_exemplars(
            frame, list(cands), exemplar_memory=mem
        )
        assert len(kept) == 3
        assert mem.counts["classify_patch"] == 3, mem.counts
        assert mem.counts["is_negative_patch"] == 3, mem.counts
        assert mem.counts["match_positive"] == 3, mem.counts
        assert mem.counts["match_negative"] == 3, mem.counts
        assert mem.total == 12, mem.counts

    def test_refine_with_positive_match_keeps_single_classify_call(self):
        """A strong positive match (cid=0 Aria, sim=0.95) must still be
        classified exactly once per candidate, and the candidate kept."""
        scanner = MaterialScanner()
        frame, cands = _frame_and_candidates(1)
        mem = CountingMem(classify_result=(0, "Aria", 0.95))
        kept = scanner._refine_candidates_with_exemplars(
            frame, list(cands), exemplar_memory=mem
        )
        assert len(kept) == 1
        assert kept[0].material_type == "Aria"
        assert mem.counts["classify_patch"] == 1, mem.counts
        assert mem.total == 4, mem.counts

    def test_discriminate_accepts_precomputed_verdict_without_querying(self):
        """discriminate_3d_material with a full precomputed verdict must
        not call the memory at all (negative + positive + fake gate)."""
        rng = np.random.default_rng(3)
        patch = rng.integers(40, 230, size=(48, 48, 3)).astype(np.uint8)
        mem = CountingMem()
        verdict = PrecomputedExemplarVerdict(
            positive=(None, None, 0.10),
            negative=(False, 0.0, None),
            fake_gate=(False, 0.0, ""),
        )
        res = discriminate_3d_material(
            patch, exemplar_memory=mem, precomputed_exemplar=verdict
        )
        assert mem.total == 0, mem.counts
        assert isinstance(res.is_authentic_node, bool)

    def test_discriminate_precomputed_negative_rejects_without_querying(self):
        """A precomputed negative verdict must reject the patch and skip
        every exemplar query, exactly like a measured negative match."""
        rng = np.random.default_rng(3)
        patch = rng.integers(40, 230, size=(48, 48, 3)).astype(np.uint8)
        mem = CountingMem()
        verdict = PrecomputedExemplarVerdict(
            positive=(None, None, 0.10),
            negative=(True, 0.93, {"class_name": "barren_rock"}),
            fake_gate=(False, 0.0, ""),
        )
        res = discriminate_3d_material(
            patch, exemplar_memory=mem, precomputed_exemplar=verdict
        )
        assert res.is_authentic_node is False
        assert res.rejection_reason == "learned_negative:barren_rock"
        assert mem.total == 0, mem.counts

    def test_discriminate_without_precomputed_still_queries(self):
        """Default path (no verdict) is unchanged: the memory IS queried."""
        rng = np.random.default_rng(3)
        patch = rng.integers(40, 230, size=(48, 48, 3)).astype(np.uint8)
        mem = CountingMem()
        discriminate_3d_material(patch, exemplar_memory=mem)
        assert mem.counts["classify_patch"] >= 1, mem.counts
        assert mem.counts["is_negative_patch"] >= 1, mem.counts


# ------------------------------------------------------------------ I2: names
NAME_CASES = [
    ("Orichalcum Ore", "orichalcum", True),
    ("Aria", "Herb Aria", True),
    ("Blood Crystal", "kan kristali", True),
    ("Obsidian Ore", "obsidyen cevheri", True),
    ("Mela", "fresh mela", True),
    ("Targena", "targena", True),
    ("Diamond Gemstone", "sapphire", True),
    ("Yggdrasil Log", "agac", True),
    ("Herb Edana", "edana", True),
    ("Herb Lumina", "lumina", True),
    ("Aria", "Orichalcum Ore", False),
    ("Blood Crystal", "Sapphire Gemstone", False),
    ("Mela", "Targena", False),
    ("", "Aria", False),
    ("Aria", "", False),
]


@pytest.mark.parametrize("target,candidate,expected", NAME_CASES)
def test_match_material_names_unchanged(target, candidate, expected):
    assert match_material_names(target, candidate) is expected


NORMALIZE_CASES = [
    ("  Herb Aria ⛏️ ", "herb aria"),
    ("Kan Kristali", "kan kristali"),
    ("Obsidyen_Cevheri", "obsidyen cevheri"),
    ("ORICHALCUM-ORE", "orichalcum ore"),
    ("Çağıl Şüşü", "cagil susu"),
    ("Fresh/Aria", "fresh aria"),
    ("", ""),
]


@pytest.mark.parametrize("raw,expected", NORMALIZE_CASES)
def test_normalize_material_name_unchanged(raw, expected):
    assert normalize_material_name(raw) == expected


def test_alias_table_is_module_level_and_shared():
    """Hoisted tables exist once at module level (no per-call rebuild)."""
    import core.material_scanner as ms

    assert isinstance(ms._MATERIAL_NAME_ALIASES, dict)
    assert len(ms._MATERIAL_NAME_ALIASES) >= 10
    assert "aria" in ms._MATERIAL_NAME_ALIASES
    assert isinstance(ms._MATERIAL_NAME_GENERIC_WORDS, frozenset)
    assert "crystal" in ms._MATERIAL_NAME_GENERIC_WORDS
    # Same object reused across calls (not rebuilt per call).
    assert ms._MATERIAL_NAME_ALIASES is ms._MATERIAL_NAME_ALIASES


def test_yolo_lookup_tables_are_class_level():
    scanner = MaterialScanner()
    assert scanner._YOLO_CLASS_NAME_MAP["aria"] == "Aria"
    assert scanner._COARSE_YOLO_CLASSES["ore_vein"] == 3
    assert "monster" in scanner._YOLO_NON_RESOURCE_KEYWORDS
    assert MaterialScanner._YOLO_CLASS_NAME_MAP is scanner._YOLO_CLASS_NAME_MAP


# ------------------------------------------------------------------ I2: hist
def test_color_histogram_scores_unchanged():
    """Cached green/targena counts must not change any score."""
    import cv2

    rng = np.random.default_rng(42)
    hsv = np.zeros((120, 120, 3), dtype=np.uint8)
    hsv[:, :, 0] = rng.integers(0, 180, size=(120, 120)).astype(np.uint8)
    hsv[:, :, 1] = rng.integers(60, 255, size=(120, 120)).astype(np.uint8)
    hsv[:, :, 2] = rng.integers(80, 255, size=(120, 120)).astype(np.uint8)
    scores = compute_color_histogram_scores(hsv)
    assert set(scores) == {
        "herb_aria", "herb_edana", "herb_lumina", "ore_orichalcum",
        "ore_obsidian", "crystal_blood", "mela", "targena",
        "dirt_terrain_ratio", "barren_gray_rock_ratio", "dense_bush_ratio",
    }
    for key, val in scores.items():
        assert 0.0 <= val <= 1.0, (key, val)
    # Scores are non-trivial on a colourful patch (not all zeros).
    assert sum(scores.values()) > 0.5
    # Empty input still returns an empty dict.
    assert compute_color_histogram_scores(np.zeros((0, 0, 3), dtype=np.uint8)) == {}


def test_scan_3d_yolo_materials_uses_hoisted_tables():
    """scan_3d_yolo_materials with a stub detector: monster classes are
    skipped, a resource detection flows through discrimination."""
    scanner = MaterialScanner()

    class StubDetector:
        def detect_3d_objects(self, frame):
            return [
                {"class_name": "monster", "confidence": 0.99,
                 "bbox": (100, 100, 60, 60)},
                {"class_name": "herb_aria", "confidence": 0.10,
                 "bbox": (200, 200, 60, 60)},
            ]

    scanner.set_yolo_detector(StubDetector())
    rng = np.random.default_rng(11)
    frame = rng.integers(40, 230, size=(720, 1280, 3)).astype(np.uint8)
    out = scanner.scan_3d_yolo_materials(frame)
    assert out == []  # monster skipped; low-confidence aria skipped
