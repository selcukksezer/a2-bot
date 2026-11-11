# -*- coding: utf-8 -*-
"""
D60 DEV tests: measurable, behaviour-preserving optimizations in
dataset_factory/yolo_auto_labeler.py.

1. `_deduplicate_annotations` compares a candidate only against kept boxes
   of the SAME class (duplicates can only occur within a class) instead of
   scanning the whole kept list -> O(n^2) over all classes becomes
   O(sum over classes n_c^2). Output (order + content) must be identical
   to the old flat-scan algorithm.
2. Scalar `np.clip` in `_pixel_xywh_to_yolo_norm` / `_make_annotation` /
   `format_yolo_txt` replaced by plain-Python `_clamp` (np.clip builds a
   0-d array per scalar call). Normalized outputs must be bit-identical.
3. `map_material_name_to_yolo_class_id` memoized on (norm, default_id):
   same outputs, repeated names served from cache.
"""

from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from dataset_factory.yolo_auto_labeler import (
    YOLOAutoLabeler,
    _clamp,
    _map_material_name_to_yolo_class_id_cached,
    map_material_name_to_yolo_class_id,
)


# ----------------------------------------------------------------------
# _clamp
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,lo,hi",
    [
        (-5.0, 0.0, 1.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (2.5, 0.0, 1.0),
        (-0.001, 0.0005, 1.0),
        (0.0004, 0.0005, 1.0),
        (150.0, 0.01, 1.0),
        (0.005, 0.01, 1.0),
    ],
)
def test_clamp_matches_np_clip(value, lo, hi):
    assert _clamp(value, lo, hi) == float(np.clip(value, lo, hi))


def test_clamp_nan_and_inf_match_np_clip():
    assert math.isnan(_clamp(float("nan"), 0.0, 1.0))
    assert _clamp(float("inf"), 0.0, 1.0) == 1.0
    assert _clamp(float("-inf"), 0.0, 1.0) == 0.0


# ----------------------------------------------------------------------
# _pixel_xywh_to_yolo_norm
# ----------------------------------------------------------------------
def test_pixel_xywh_to_yolo_norm_edge_cases():
    # In-bounds box: exact hand-computed values.
    cx, cy, w, h, px_xywh, px_xyxy = YOLOAutoLabeler._pixel_xywh_to_yolo_norm(
        100, 200, 80, 60, 1920, 1080
    )
    assert (cx, cy, w, h) == (0.072917, 0.212963, 0.041667, 0.055556)
    assert px_xywh == (100, 200, 80, 60)
    assert px_xyxy == (100, 200, 180, 260)

    # Negative origin clamps to the image edge.
    cx, cy, w, h, _, _ = YOLOAutoLabeler._pixel_xywh_to_yolo_norm(
        -50, -30, 100, 80, 1920, 1080
    )
    assert 0.0 <= cx - w / 2 and cy - h / 2 >= 0.0
    assert (cx, cy) == (0.013021, 0.023148)

    # Overflow clamps to the far edge. NOTE: outputs are rounded to 6
    # decimals, so cx + w/2 may exceed 1.0 by up to 5e-7 (pre-existing:
    # the base version returns the identical values); allow that epsilon.
    cx, cy, w, h, _, _ = YOLOAutoLabeler._pixel_xywh_to_yolo_norm(
        1900, 1060, 200, 100, 1920, 1080
    )
    assert cx + w / 2 <= 1.0 + 1e-6 and cy + h / 2 <= 1.0 + 1e-6

    # Degenerate / zero-size box still yields the minimum 0.0005 size.
    _, _, w, h, _, _ = YOLOAutoLabeler._pixel_xywh_to_yolo_norm(500, 500, 0, 0, 1920, 1080)
    assert w >= 0.0005 and h >= 0.0005

    # Every output stays inside [0, 1] on a sweep of adversarial inputs.
    rng = random.Random(60)
    for _ in range(500):
        x = rng.uniform(-500, 2500)
        y = rng.uniform(-500, 1500)
        bw = rng.uniform(0, 800)
        bh = rng.uniform(0, 800)
        cx, cy, w, h, _, _ = YOLOAutoLabeler._pixel_xywh_to_yolo_norm(x, y, bw, bh, 1920, 1080)
        assert 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0
        assert 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0
        assert 0.0 - 1e-6 <= cx - w / 2 and cx + w / 2 <= 1.0 + 1e-6
        assert 0.0 - 1e-6 <= cy - h / 2 and cy + h / 2 <= 1.0 + 1e-6


def test_make_annotation_confidence_clamped():
    labeler = YOLOAutoLabeler.__new__(YOLOAutoLabeler)
    assert labeler._make_annotation(0, 0, 0, 10, 10, 100, 100, confidence=2.0)["confidence"] == 1.0
    assert labeler._make_annotation(0, 0, 0, 10, 10, 100, 100, confidence=-3.0)["confidence"] == 0.01
    assert labeler._make_annotation(0, 0, 0, 10, 10, 100, 100, confidence=0.5)["confidence"] == 0.5


# ----------------------------------------------------------------------
# _compute_iou_xyxy: plain-arithmetic version, identical values
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "a,b,want",
    [
        ((100, 100, 180, 180), (105, 105, 185, 185), 0.7839721253262756),
        ((0, 0, 50, 50), (500, 500, 600, 600), 0.0),
        ((10, 10, 200, 120), (50, 40, 150, 100), 0.2870813396991827),
        ((0, 0, 100, 100), (0, 0, 100, 100), 1.0),  # identical boxes (up to the 1e-6 epsilon)
        ((0, 0, 10, 10), (10, 0, 20, 10), 0.0),  # touching edges, no overlap
        ((50, 50, 50, 50), (0, 0, 100, 100), 0.0),  # degenerate zero-area box
    ],
)
def test_compute_iou_xyxy_values(a, b, want):
    got = YOLOAutoLabeler._compute_iou_xyxy(a, b)
    assert isinstance(got, float)
    assert got == pytest.approx(want, abs=1e-9)
    # Symmetric.
    assert YOLOAutoLabeler._compute_iou_xyxy(b, a) == pytest.approx(want, abs=1e-9)


# ----------------------------------------------------------------------
# _deduplicate_annotations: identical output to the old flat-scan NMS
# ----------------------------------------------------------------------
def _reference_dedup(labeler, annotations, iou_threshold=0.45):
    """Naive copy of the pre-D60 algorithm: scan the whole kept list."""
    if not annotations:
        return []
    ordered = sorted(
        annotations,
        key=lambda a: (int(a["class_id"]), -float(a.get("confidence", 0.0))),
    )
    kept = []
    for cand in ordered:
        cid = int(cand["class_id"])
        c_xyxy = cand["bbox_xyxy"]
        c_cx = 0.5 * (c_xyxy[0] + c_xyxy[2])
        c_cy = 0.5 * (c_xyxy[1] + c_xyxy[3])
        duplicate = False
        for existing in kept:
            if int(existing["class_id"]) != cid:
                continue
            e_xyxy = existing["bbox_xyxy"]
            iou = labeler._compute_iou_xyxy(c_xyxy, e_xyxy)
            if iou >= iou_threshold:
                duplicate = True
                break
            e_cx = 0.5 * (e_xyxy[0] + e_xyxy[2])
            e_cy = 0.5 * (e_xyxy[1] + e_xyxy[3])
            min_dim = max(8.0, min(c_xyxy[2] - c_xyxy[0], c_xyxy[3] - c_xyxy[1]) * 0.35)
            if math.hypot(c_cx - e_cx, c_cy - e_cy) <= min_dim:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def _random_anns(labeler, n, seed):
    rng = random.Random(seed)
    anns = []
    for _ in range(n):
        cid = rng.randrange(23)
        x = rng.randrange(0, 1800)
        y = rng.randrange(0, 900)
        w = rng.randrange(20, 120)
        h = rng.randrange(20, 120)
        anns.append(
            labeler._make_annotation(
                class_id=cid, x=x, y=y, w=w, h=h, img_w=1920, img_h=1080,
                confidence=round(rng.random(), 3), source="d60_test",
            )
        )
    return anns


@pytest.mark.parametrize("seed", [7, 11, 23, 42])
def test_dedup_identical_to_reference_algorithm(seed):
    labeler = YOLOAutoLabeler.__new__(YOLOAutoLabeler)
    anns = _random_anns(labeler, 600, seed)
    # Deliberate near-duplicates: same box twice, and same-class boxes
    # whose centers are within the 0.35 merge radius.
    anns.extend([dict(anns[0]), dict(anns[5])])
    got = labeler._deduplicate_annotations(list(anns))
    want = _reference_dedup(labeler, list(anns))
    assert got == want


def test_dedup_merges_same_class_overlap_keeps_other_classes():
    labeler = YOLOAutoLabeler.__new__(YOLOAutoLabeler)
    a = labeler._make_annotation(3, 100, 100, 80, 80, 1920, 1080, confidence=0.9, source="d60")
    b = labeler._make_annotation(3, 105, 105, 80, 80, 1920, 1080, confidence=0.8, source="d60")
    c = labeler._make_annotation(4, 100, 100, 80, 80, 1920, 1080, confidence=0.8, source="d60")
    out = labeler._deduplicate_annotations([b, c, a])
    assert [o["class_id"] for o in out] == [3, 4]  # same-class dup merged, other class kept
    assert out[0]["confidence"] == 0.9  # highest-confidence box wins


def test_dedup_2000_is_fast():
    """Regression lock for the D60 NMS speedup (was ~5.5s for 2000 boxes
    on this VM, best of 3, before the per-class comparison narrowing)."""
    labeler = YOLOAutoLabeler.__new__(YOLOAutoLabeler)
    anns = _random_anns(labeler, 2000, seed=11)
    t0 = time.perf_counter()
    out = labeler._deduplicate_annotations(list(anns))
    assert time.perf_counter() - t0 < 2.0
    assert len(out) > 0


# ----------------------------------------------------------------------
# map_material_name_to_yolo_class_id memoization
# ----------------------------------------------------------------------
# NOTE (R13a, bayat-test düzeltmesi — F-0824/R60): D60 bu parametreleri
# legacy 0..22 düzenine göre pinlemişti (crest 6, hp_potion 18,
# buff_scroll 22, boş/None -> uydurma varsayılan cevher 3). R60
# (6479b48, v1 ve v2'de de var) YOLO_CLASSES'ı kanonik 27 sınıfa
# çevirdi: crest -> 13 skill_ui_prompt, hp_potion -> 22, buff_scroll
# -> 24 (dataset_factory/class_taxonomy.py LEGACY_TO_CANONICAL);
# boş/None -> 12 world_material, "never fabricate the default ore"
# (8f31462 + G2-01). Üretim kodu kanonik sözleşmeye uyuyor (w31 de
# buff_scroll==24, hp_potion==22 assert ediyor); legacy id'leri geri
# getirmek 6=herb_edana / 22=hp_potion çakışmasını diriltirdi.
# Aşağıdaki beklentiler kanonik düzene güncellendi; memoize davranışı
# (aynı girdi -> aynı çıktı) değişmedi.
@pytest.mark.parametrize(
    "name,default_id,want",
    [
        ("aria", 3, 0),
        ("Aria", 3, 0),
        ("  Mela Berry ", 3, 1),
        ("targena", 3, 2),
        ("Orichalcum Ore", 3, 3),
        ("Crystal", 3, 4),
        ("Wood", 3, 5),
        ("world_gather_crest", 3, 13),
        ("monster", 3, 10),
        ("hp_potion", 3, 22),
        ("buff_scroll", 3, 24),
        ("zzzqqq_unknown", 3, 12),
        ("zzzqqq_unknown", 2, 2),  # unknown name honours the caller's default_id (contract)
        ("", 3, 12),
        (None, 3, 12),
    ],
)
def test_map_material_name_outputs_unchanged(name, default_id, want):
    assert map_material_name_to_yolo_class_id(name, default_id=default_id) == want


def test_map_material_name_uses_cache():
    _map_material_name_to_yolo_class_id_cached.cache_clear()
    map_material_name_to_yolo_class_id("Orichalcum Ore")
    map_material_name_to_yolo_class_id("Orichalcum Ore")
    map_material_name_to_yolo_class_id("orichalcum ore")  # same normalized key
    info = _map_material_name_to_yolo_class_id_cached.cache_info()
    assert info.misses == 1
    assert info.hits == 2
    # Different default_id is a different cache key.
    map_material_name_to_yolo_class_id("Orichalcum Ore", default_id=2)
    assert _map_material_name_to_yolo_class_id_cached.cache_info().misses == 2


# ----------------------------------------------------------------------
# format_yolo_txt clamping (via _clamp)
# ----------------------------------------------------------------------
def test_format_yolo_txt_clamps_out_of_range():
    labeler = YOLOAutoLabeler.__new__(YOLOAutoLabeler)
    anns = [
        {"class_id": 3, "cx": -0.2, "cy": 1.4, "w": 2.0, "h": -0.5},
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
    ]
    assert labeler.format_yolo_txt(anns) == "3 0.000000 1.000000 1.000000 0.000000\n0 0.500000 0.500000 0.100000 0.100000"
