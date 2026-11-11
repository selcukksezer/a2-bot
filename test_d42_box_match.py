# -*- coding: utf-8 -*-
"""
tests for dev slice D42 (dataset_factory/box-match):
- BoxGeometryGuard.filter_yolo_arrays: vectorized filter, same rules/stats
  as the scalar filter_yolo_annotations (incl. reason precedence + NaN).
- VectorizedExemplarMatcher incremental indexing: one-by-one adds must
  leave the index identical to a full rebuild_matrices() over the same
  exemplars (matrices, class ids, super-categories, match results).
- LRUFrameDecodeCache: memoized abspath key (str == Path, chdir-safe for
  relative paths), one stat per decode miss, symlink spelling separate.
- RubberBandBoxConverter.convert_result fast path: identical results to
  the public helpers on random drags, NaN still rejected.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_factory.box_geometry_guard import BoxGeometryGuard
from dataset_factory.lru_frame_cache import LRUFrameDecodeCache, _abspath_cache
from dataset_factory.rubberband_box_converter import RubberBandBoxConverter
from dataset_factory.vectorized_exemplar_matcher import (
    FEATURE_VECTOR_DIM,
    VectorizedExemplarMatcher,
    feature_dict_to_normalized_vector,
    normalize_l2_vector,
)


# ------------------------------------------------------------------ guard

def _rand_boxes(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    boxes = np.column_stack([
        rng.uniform(-0.05, 1.05, n), rng.uniform(-0.05, 1.05, n),
        rng.uniform(0.0, 0.8, n), rng.uniform(0.0, 0.8, n),
    ])
    boxes[:50, 0] = np.nan
    boxes[50:100, 1] = np.inf
    boxes[100:150, 2] = 0.0
    boxes[150:200, 2] = 0.002
    boxes[200:250, 3] = 0.9
    return boxes, rng.integers(0, 25, n)


def test_filter_yolo_arrays_matches_scalar_random():
    guard = BoxGeometryGuard()
    boxes, cids = _rand_boxes(5000)
    tuples = [(int(c), float(r[0]), float(r[1]), float(r[2]), float(r[3]))
              for c, r in zip(cids, boxes)]
    acc_scalar, stats_scalar = guard.filter_yolo_annotations(tuples)
    acc_vec, ids_vec, stats_vec = guard.filter_yolo_arrays(boxes, cids)
    acc_vec_t = [(int(c), float(r[0]), float(r[1]), float(r[2]), float(r[3]))
                 for c, r in zip(ids_vec, acc_vec)]
    assert acc_vec_t == acc_scalar
    for key, val in stats_scalar.items():
        assert stats_vec[key] == val, key
    assert stats_vec["accepted"] == len(acc_scalar)
    assert acc_vec.dtype == np.float64
    assert ids_vec.dtype == np.int64


def test_filter_yolo_arrays_reason_precedence_and_edges():
    guard = BoxGeometryGuard()
    # rows: (box, expected_reason) — NaN + micro together must report
    # non_finite (precedence), zero-width + giant must report non_positive.
    cases = [
        ((0.5, 0.5, 0.1, 0.1), "ok"),
        ((float("nan"), 0.5, 0.001, 0.1), "non_finite_coordinate"),
        ((0.5, 0.5, 0.0, 0.1), "non_positive_dimension"),
        ((0.5, 0.5, -0.1, 0.9), "non_positive_dimension"),
        ((0.5, 0.5, 0.002, 0.1), "micro_noise_box"),
        ((0.5, 0.5, 0.1, 0.9), "giant_false_box"),
        ((1.5, 0.5, 0.1, 0.1), "center_out_of_screen"),
        ((0.98, 0.5, 0.1, 0.1), "corners_out_of_screen"),
        ((0.5, 0.5, 0.004, 0.004), "ok"),  # exactly at min size
        ((0.5, 0.5, 0.65, 0.65), "ok"),  # exactly at max size
    ]
    boxes = np.array([c[0] for c in cases], dtype=np.float64)
    acc, _, stats = guard.filter_yolo_arrays(boxes)
    assert len(acc) == 3
    for (box, expected), (_, reason) in zip(
        cases, [guard.is_valid_yolo_box(*c[0]) for c in cases]
    ):
        assert reason == expected, (box, reason)
    assert stats["non_finite_coordinate"] == 1
    assert stats["non_positive_dimension"] == 2
    assert stats["micro_noise_box"] == 1
    assert stats["giant_false_box"] == 1
    assert stats["center_out_of_screen"] == 1
    assert stats["corners_out_of_screen"] == 1
    assert stats["accepted"] == 3
    # corner eps: x2 = 1.0 + 0.5e-6 is tolerated, x2 = 1.0 + 2e-6 is not
    edge = np.array([[0.9500005, 0.5, 0.1, 0.1],
                     [0.950002, 0.5, 0.1, 0.1]], dtype=np.float64)
    acc2, _, stats2 = guard.filter_yolo_arrays(edge)
    assert len(acc2) == 1 and stats2["corners_out_of_screen"] == 1


def test_filter_yolo_arrays_empty_single_and_errors():
    guard = BoxGeometryGuard()
    acc, ids, stats = guard.filter_yolo_arrays(np.zeros((0, 4)))
    assert acc.shape == (0, 4) and ids is None and stats["total"] == 0
    acc, ids, stats = guard.filter_yolo_arrays((0.5, 0.5, 0.1, 0.1), [7])
    assert acc.shape == (1, 4) and list(ids) == [7] and stats["accepted"] == 1
    # float class ids truncate toward zero, like int(...)
    _, ids, _ = guard.filter_yolo_arrays(
        [[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.2, 0.2]], [2.9, -1.9])
    assert list(ids) == [2, -1]
    with pytest.raises(ValueError):
        guard.filter_yolo_arrays(np.zeros((3, 3)))
    with pytest.raises(ValueError):
        guard.filter_yolo_arrays([["a", "b", "c", "d"]])
    with pytest.raises(ValueError):
        guard.filter_yolo_arrays(np.zeros((2, 4)), [1])
    with pytest.raises(ValueError):
        guard.filter_yolo_arrays(np.zeros((2, 4)), [1, "x"])


# ------------------------------------------------------------------ matcher

def _rand_vec(rng):
    v = rng.normal(size=FEATURE_VECTOR_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _exemplars(count, seed=7):
    rng = np.random.default_rng(seed)
    return [{"class_id": int(i % 25), "class_name": f"class_{i % 25}",
             "feature_vector": _rand_vec(rng), "timestamp": float(i)}
            for i in range(count)]


def test_incremental_adds_match_full_rebuild():
    exemplars = _exemplars(200)
    inc = VectorizedExemplarMatcher()
    for ex in exemplars:
        assert inc.add_positive_exemplar(ex) is True
    ref = VectorizedExemplarMatcher(positive_exemplars=exemplars)
    assert inc._pos_matrix.shape == ref._pos_matrix.shape == (200, FEATURE_VECTOR_DIM)
    assert np.array_equal(inc._pos_cids, ref._pos_cids)
    assert np.array_equal(inc._pos_super_cats, ref._pos_super_cats)
    assert np.allclose(inc._pos_matrix, ref._pos_matrix, atol=1e-6)
    assert len(inc._positive) == len(ref._positive) == 200
    # match results agree too (incl. a category-masked query)
    rng = np.random.default_rng(123)
    for _ in range(5):
        q = _rand_vec(rng)
        assert inc.match_positive(q) == ref.match_positive(q) or (
            abs(inc.match_positive(q)[0] - ref.match_positive(q)[0]) < 1e-5
            and inc.match_positive(q)[1:] == ref.match_positive(q)[1:])
        a = inc.match_positive(q, same_category_as=3)
        b = ref.match_positive(q, same_category_as=3)
        assert a[1:] == b[1:] and abs(a[0] - b[0]) < 1e-5


def test_dedup_hit_updates_indexed_row_in_place():
    rng = np.random.default_rng(99)
    v1 = _rand_vec(rng)
    ex1 = {"class_id": 3, "class_name": "class_3", "feature_vector": v1,
           "timestamp": 1.0}
    v2 = v1 + 0.001 * _rand_vec(rng)
    v2 = (v2 / np.linalg.norm(v2)).astype(np.float32)
    ex2 = {"class_id": 3, "class_name": "class_3", "feature_vector": v2,
           "timestamp": 2.0}
    m = VectorizedExemplarMatcher()
    assert m.add_positive_exemplar(ex1) is True
    assert m.add_positive_exemplar(ex2) is True  # dedup hit (cos ~1.0)
    assert m._pos_matrix.shape[0] == 1
    assert np.allclose(m._pos_matrix[0], normalize_l2_vector(v2), atol=1e-6)
    assert m._positive[0]["timestamp"] == 2.0


def test_positive_quota_eviction_keeps_newest():
    rng = np.random.default_rng(11)
    m = VectorizedExemplarMatcher()
    for i in range(8):
        ex = {"class_id": 3, "class_name": "class_3",
              "feature_vector": _rand_vec(rng), "timestamp": float(i)}
        assert m.add_positive_exemplar(ex, max_per_class=5) is True
    assert m._pos_matrix.shape[0] == 5
    assert [e["timestamp"] for e in m._positive] == [3.0, 4.0, 5.0, 6.0, 7.0]
    assert set(m._pos_cids.tolist()) == {3}
    for e, row in zip(m._positive, m._pos_matrix):
        assert np.allclose(row, feature_dict_to_normalized_vector(e), atol=1e-6)


def test_negative_fifo_eviction_keeps_newest():
    rng = np.random.default_rng(13)
    m = VectorizedExemplarMatcher()
    for i in range(8):
        ex = {"class_id": int(i % 5), "class_name": f"class_{i % 5}",
              "feature_vector": _rand_vec(rng), "timestamp": float(i)}
        assert m.add_negative_exemplar(ex, max_negatives=5) is True
    assert m._neg_matrix.shape[0] == 5
    assert [e["timestamp"] for e in m._negative] == [3.0, 4.0, 5.0, 6.0, 7.0]
    for e, row in zip(m._negative, m._neg_matrix):
        assert np.allclose(row, feature_dict_to_normalized_vector(e), atol=1e-6)


def test_sqrt_weight_constants_unchanged():
    # the pre-computed constants must equal sqrt() of the design weights
    import dataset_factory.vectorized_exemplar_matcher as mod
    assert float(mod._SQRT_WEIGHT_HSV) == pytest.approx(math.sqrt(0.70), rel=1e-6)
    assert float(mod._SQRT_WEIGHT_ASPECT_RATIO) == pytest.approx(
        math.sqrt(0.15), rel=1e-6)
    assert float(mod._SQRT_WEIGHT_TEXTURE) == pytest.approx(
        math.sqrt(0.15), rel=1e-6)


# ------------------------------------------------------------------ lru cache

def _write_png(path: Path, fill: int = 77) -> None:
    import cv2
    img = np.full((32, 48, 3), fill, dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(enc.tobytes())


def test_normalize_key_str_path_equal_and_chdir_safe(tmp_path, monkeypatch):
    _abspath_cache.clear()
    p = tmp_path / "frame.png"
    _write_png(p)
    key_abs = LRUFrameDecodeCache._normalize_key(str(p))
    assert key_abs == LRUFrameDecodeCache._normalize_key(p)
    assert key_abs == os.path.abspath(str(p))
    # relative path must follow the cwd after a chdir (not a stale memo)
    monkeypatch.chdir(tmp_path)
    assert LRUFrameDecodeCache._normalize_key("frame.png") == os.path.abspath(
        "frame.png")
    assert LRUFrameDecodeCache._normalize_key("frame.png") == key_abs
    # memoized absolute key survives a chdir unchanged
    assert LRUFrameDecodeCache._normalize_key(str(p)) == key_abs


def test_get_or_decode_stats_file_once_on_miss(tmp_path):
    p = tmp_path / "frame.png"
    _write_png(p)
    cache = LRUFrameDecodeCache(capacity=8)
    orig = LRUFrameDecodeCache._stat_file
    calls = {"n": 0}

    def counting(path):
        calls["n"] += 1
        return orig(path)

    LRUFrameDecodeCache._stat_file = staticmethod(counting)
    try:
        assert cache.get_or_decode(p) is not None
        assert calls["n"] == 1, f"miss statted the file {calls['n']}x, want 1"
        calls["n"] = 0
        assert cache.get_or_decode(p) is not None  # hit path
        assert calls["n"] == 1
    finally:
        # restore as a staticmethod: assigning the bare function back
        # would rebind it as an instance method and break later calls
        LRUFrameDecodeCache._stat_file = staticmethod(orig)


def test_symlink_spelling_is_separate_key_but_decodes(tmp_path):
    target = tmp_path / "frame.png"
    _write_png(target)
    link = tmp_path / "link.png"
    link.symlink_to(target)
    assert (LRUFrameDecodeCache._normalize_key(str(link))
            != LRUFrameDecodeCache._normalize_key(str(target)))
    cache = LRUFrameDecodeCache(capacity=8)
    assert cache.get_or_decode(link) is not None
    assert cache.get_or_decode(target) is not None
    assert cache.get_stats()["size"] == 2


# ------------------------------------------------------------------ rubberband

def test_convert_result_matches_public_helpers_random():
    conv = RubberBandBoxConverter()
    rect = (100.0, 80.0, 800.0, 600.0)
    rng = np.random.default_rng(5)
    for _ in range(300):
        x1, y1 = float(rng.uniform(0, 900)), float(rng.uniform(0, 680))
        x2, y2 = x1 + float(rng.uniform(-300, 300)), y1 + float(rng.uniform(-300, 300))
        res = conv.convert_result(x1, y1, x2, y2, rect)
        drag_ok, drag_reason = conv.validate_drag_distance(x1, y1, x2, y2)
        if not drag_ok:
            assert res.valid is False and res.reason == drag_reason
            continue
        clipped = conv.clip_to_target_rect(x1, y1, x2, y2, rect)
        assert clipped is not None
        cx1, cy1, cx2, cy2 = clipped
        if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
            assert res.valid is False and res.reason == "outside_target_rect"
            continue
        clipped_ok, _ = conv.validate_drag_distance(cx1, cy1, cx2, cy2)
        if conv.config.require_clipped_min_drag and not clipped_ok:
            assert res.valid is False and res.reason == "clipped_drag_too_short"
            continue
        assert res.valid is True
        assert res.cx == pytest.approx(
            conv.clamp_normalized_coord(((cx1 + cx2) * 0.5 - 100.0) / 800.0, 0.0, 1.0))
        assert res.w == pytest.approx(
            conv.clamp_normalized_coord((cx2 - cx1) / 800.0, 0.0, 1.0))


def test_convert_result_rejects_nonfinite_and_click():
    conv = RubberBandBoxConverter()
    rect = (0.0, 0.0, 800.0, 600.0)
    assert conv.convert_result(float("nan"), 10, 100, 100, rect).valid is False
    assert conv.convert_result(float("nan"), 10, 100, 100, rect).reason == \
        "invalid_coordinates"
    assert conv.validate_drag_distance(float("nan"), 0, 10, 10) == (
        False, "invalid_coordinates")
    click = conv.convert_result(100, 100, 105, 105, rect)
    assert click.valid is False and click.reason == "drag_too_short"
    exact = conv.convert_result(100, 100, 108, 108, rect)
    assert exact.valid is True  # exactly 8px on both axes
    rev = conv.convert_result(108, 108, 100, 100, rect)
    assert (rev.cx, rev.cy, rev.w, rev.h) == (exact.cx, exact.cy, exact.w, exact.h)
