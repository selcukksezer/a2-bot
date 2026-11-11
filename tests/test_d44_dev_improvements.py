# -*- coding: utf-8 -*-
"""D44 DEV-improvement locks.

Pins the behaviour of the two D44 hot-path refactors so a future change
cannot silently alter decisions while chasing speed:

1. player_self_exclusion_guard: single mapping extraction per candidate
   (evaluate_candidate -> evaluate -> has_*_evidence) + lru_cache on
   is_protected_3d_category + precompiled token splitter.
2. scroll_potion_classifier: module-level HSV bounds / morph kernel +
   is_empty_slot full-frame short-circuit.
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

from dataset_factory.player_self_exclusion_guard import (
    PlayerSelfExclusionGuard,
    is_protected_3d_category,
)
from dataset_factory.scroll_potion_classifier import HUDConsumableClassifier


# ----------------------------------------------------------------------
# Guard: decision matrix (behaviour pinned, speed is the only change)
# ----------------------------------------------------------------------
def test_guard_decision_matrix_unchanged():
    g = PlayerSelfExclusionGuard()
    # in-zone 3D material, no evidence -> excluded
    d = g.evaluate(0.50, 0.51, category="3d_material")
    assert d.excluded is True
    assert d.reason == "excluded_player_mount_center_zone_without_evidence"
    # token-exact gate: "wooden_chest" is NOT protected (F-0983)
    d = g.evaluate(0.50, 0.51, category="wooden_chest")
    assert d.excluded is False and d.reason == "non_3d_category_exempt"
    # "ore_orichalcum" splits to the protected token "ore"
    d = g.evaluate(0.50, 0.51, category="ore_orichalcum")
    assert d.excluded is True
    # outside the zone -> never excluded
    d = g.evaluate(0.10, 0.10, category="3d_material")
    assert d.excluded is False and d.reason == "outside_self_exclusion_zone"
    # gather evidence via OCR frees the candidate; low confidence does not
    d = g.evaluate(0.50, 0.51, category="3d_material", ocr_text="[F Gather]")
    assert d.excluded is False and d.reason == "allowed_by_f_gather_evidence"
    d = g.evaluate(0.50, 0.51, category="3d_material",
                   candidate={"has_f_gather": True, "gather_confidence": 0.3})
    assert d.excluded is True  # bare flag below min confidence is not evidence
    d = g.evaluate(0.50, 0.51, category="3d_material",
                   candidate={"has_f_gather": True, "gather_confidence": 0.9})
    assert d.excluded is False
    # crest evidence via tag frees the candidate
    d = g.evaluate(0.50, 0.51, category="monster_3d",
                   candidate={"evidence_tags": ["target_crest"]})
    assert d.excluded is False and d.reason == "allowed_by_crest_evidence"


def test_guard_filter_candidates_and_yolo_lines():
    g = PlayerSelfExclusionGuard()
    cands = [
        {"cx": 0.50, "cy": 0.51, "category": "3d_material"},            # dropped
        {"cx": 0.10, "cy": 0.10, "category": "3d_material"},            # kept
        {"cx": 0.50, "cy": 0.51, "category": "wooden_chest"},          # kept
        {"cx": 0.50, "cy": 0.51, "category": "3d_material",
         "ocr_text": "[F Gather]"},                                    # kept
        (0.50, 0.51),                                                  # tuple, default 3d -> dropped
    ]
    kept = g.filter_candidates(cands)
    assert len(kept) == 3
    assert g.get_diagnostics()["total_excluded"] == 2

    lines = [
        "3 0.500000 0.510000 0.050000 0.050000",   # ore_orichalcum, in zone -> dropped
        "26 0.500000 0.510000 0.050000 0.050000",  # player_character -> kept
        "13 0.500000 0.510000 0.050000 0.050000",  # skill_ui_prompt -> kept
        "3 0.100000 0.100000 0.050000 0.050000",   # outside zone -> kept
        "bad line",                               # malformed -> counted, dropped
        "x 0.500000 0.510000 0.050000 0.050000",   # unparsable class -> counted
    ]
    out = g.filter_yolo_lines(lines)
    assert len(out) == 3
    diag = g.get_diagnostics()
    assert diag["yolo_lines_skipped_malformed"] == 1
    assert diag["yolo_lines_skipped_unparsable"] == 1


def test_guard_single_mapping_extraction_per_candidate():
    """evaluate_candidate used to extract the candidate mapping up to 4x
    (center + category + both evidence checks); now it extracts once."""
    g = PlayerSelfExclusionGuard()
    calls = {"n": 0}
    orig = PlayerSelfExclusionGuard._extract_mapping

    def counting(candidate):
        calls["n"] += 1
        return orig(candidate)

    g._extract_mapping = counting  # instance-level spy (staticmethod via self)
    cands = [{"cx": 0.50, "cy": 0.51, "category": "3d_material",
              "evidence_tags": ["node"]} for _ in range(50)]
    g.filter_candidates(cands)
    assert calls["n"] <= 55, f"expected ~1 extraction/candidate, got {calls['n']}"
    assert calls["n"] >= 50


def test_guard_protected_category_cache():
    is_protected_3d_category.cache_clear()
    assert is_protected_3d_category("ore_orichalcum") is True
    assert is_protected_3d_category("wooden_chest") is False
    assert is_protected_3d_category("monster_3d") is True
    assert is_protected_3d_category("monster_hunter") is True  # whole token
    info = is_protected_3d_category.cache_info()
    assert info.misses == 4
    for _ in range(5):
        is_protected_3d_category("ore_orichalcum")
    assert is_protected_3d_category.cache_info().hits >= 5


def test_guard_pixel_coords_still_need_frame_size():
    g = PlayerSelfExclusionGuard()
    with pytest.raises(ValueError):
        g.evaluate(960.0, 550.0, category="3d_material")
    d = g.evaluate(960.0, 550.0, category="3d_material",
                   frame_width=1920, frame_height=1080)
    assert d.excluded is True  # (0.5, ~0.509) is in the zone


# ----------------------------------------------------------------------
# Classifier: pinned counts / categories (HSV bounds hoisted, not changed)
# ----------------------------------------------------------------------
def _solid_hsv(h, s, v, size=48):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    arr[:, :, 0] = h
    arr[:, :, 1] = s
    arr[:, :, 2] = v
    return arr


def test_classifier_pixel_counts_match_reference_ranges():
    clf = HUDConsumableClassifier()
    slot = _solid_hsv(5, 200, 200)  # red
    counts = clf.compute_consumable_pixel_counts(slot)
    # reference: same ranges, freshly built arrays
    r1 = cv2.inRange(slot, np.array([0, 105, 75], dtype=np.uint8),
                     np.array([12, 255, 255], dtype=np.uint8))
    assert counts["hp_red_px"] == int(np.count_nonzero(r1))
    assert counts["hp_red_px"] == 48 * 48
    assert counts["mp_blue_yellow_px"] == 0
    assert counts["slot_area"] == 48 * 48

    scroll = _solid_hsv(15, 100, 180)  # parchment, outside yellow range
    counts = clf.compute_consumable_pixel_counts(scroll)
    assert counts["scroll_px"] == 48 * 48
    assert counts["mp_blue_yellow_px"] == 0


def test_classifier_categories_pinned():
    clf = HUDConsumableClassifier()
    assert clf.classify_slot(_solid_hsv(5, 200, 200), 0.30, 0.90,
                             is_hsv=True).category == "hp_potion"
    assert clf.classify_slot(_solid_hsv(110, 200, 200), 0.30, 0.90,
                             is_hsv=True).category == "mp_potion"
    assert clf.classify_slot(_solid_hsv(25, 200, 200), 0.30, 0.90,
                             is_hsv=True).category == "mp_potion"  # yellow
    assert clf.classify_slot(_solid_hsv(15, 100, 180), 0.30, 0.90,
                             is_hsv=True).category == "buff_scroll"
    # flat dark slot -> empty; centre emblem -> excluded; gap cy -> routed
    assert clf.classify_slot(_solid_hsv(0, 0, 30), 0.30, 0.90,
                             is_hsv=True).category == "empty_slot"
    assert clf.classify_slot(_solid_hsv(5, 200, 200), 0.50, 0.90,
                             is_hsv=True).category == "center_emblem"
    res = clf.classify_slot(_solid_hsv(5, 200, 200), 0.30, 0.9384, is_hsv=True)
    assert res.should_label is True and res.category == "hp_potion"


def test_classifier_is_empty_slot_metrics_unchanged():
    clf = HUDConsumableClassifier()
    flat = _solid_hsv(0, 0, 30)
    empty, metrics = clf.is_empty_slot(flat)
    assert empty is True
    assert metrics == {"sat_mean": 0.0, "val_std": 0.0, "val_mean": 30.0}
    # chromatic slot is not empty; with edges the edge ratio is reported
    chroma = _solid_hsv(5, 200, 200)
    empty, metrics = clf.is_empty_slot(chroma)
    assert empty is False
    edges = np.zeros((48, 48), dtype=np.uint8)
    edges[10:40, 10:40] = 255
    empty, metrics = clf.is_empty_slot(chroma, slot_edges=edges)
    assert empty is False and "inner_edge_ratio" in metrics
    # flat + edges still reports the ratio and stays empty
    empty, metrics = clf.is_empty_slot(flat, slot_edges=np.zeros((48, 48), dtype=np.uint8))
    assert empty is True and metrics["inner_edge_ratio"] == 0.0


def test_classifier_bbox_out_of_frame():
    clf = HUDConsumableClassifier()
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    res = clf.classify_bbox(frame, (5000, 5000, 40, 40))
    assert res.category == "unclassified" and res.row_type == "out_of_frame"
