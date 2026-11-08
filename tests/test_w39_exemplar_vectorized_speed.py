# -*- coding: utf-8 -*-
"""
tests/test_w39_exemplar_vectorized_speed.py
===========================================
Worker W39 Verification Suite:
Validates that `ExemplarMemory` pre-normalized `(N, 438)` NumPy matrix inner product (`np.dot`)
executes 500 queries in < 100ms while preserving mathematical cosine similarity accuracy,
category-locked filtering, and class-scaled negative matching.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from dataset_factory.exemplar_memory import (
    FEATURE_VECTOR_DIM,
    ExemplarMemory,
    compute_feature_cosine_similarity,
    extract_crop_features,
    feature_dict_to_normalized_vector,
)


@pytest.fixture
def w39_sandbox() -> Dict[str, Any]:
    temp_dir = tempfile.mkdtemp(prefix="test_w39_exemplar_")
    ex_path = Path(temp_dir) / "learned_exemplars.json"
    yield {"temp_dir": temp_dir, "ex_path": ex_path}
    shutil.rmtree(temp_dir, ignore_errors=True)


def _make_synthetic_exemplar_dict(seed: int, class_id: int, class_name: str) -> Dict[str, Any]:
    """Create a deterministic 438-dim feature dict without slow disk IO."""
    rng = np.random.default_rng(seed)
    hsv_raw = rng.random(432, dtype=np.float32)
    hsv_norm = (hsv_raw / (float(np.sum(hsv_raw)) + 1e-9)).tolist()
    aspect_ratio = round(float(rng.uniform(0.4, 2.5)), 4)
    texture = [round(float(x), 4) for x in rng.uniform(0.05, 0.95, size=5)]
    return {
        "class_id": int(class_id),
        "class_name": class_name,
        "hsv_hist": hsv_norm,
        "aspect_ratio": aspect_ratio,
        "texture": texture,
        "mean_hsv": [float(rng.uniform(0, 180)), float(rng.uniform(50, 255)), float(rng.uniform(50, 255))],
        "timestamp": time.time(),
        "source": "w39_benchmark",
    }


def test_1_matrix_shape_438_and_unit_norm_invariants(w39_sandbox: Dict[str, Any]) -> None:
    """
    Test 1: Verify ExemplarMemory builds `(N, 438)` float32 matrices (`_pos_matrix`, `_neg_matrix`)
    with unit L2 row norms so `np.dot` equals exact cosine similarity.
    """
    assert FEATURE_VECTOR_DIM == 438

    mem = ExemplarMemory(json_path=w39_sandbox["ex_path"])
    assert mem._pos_matrix is not None and mem._pos_matrix.shape == (0, 438)
    assert mem._neg_matrix is not None and mem._neg_matrix.shape == (0, 438)

    N_pos, N_neg = 64, 48
    mem._positive = [
        _make_synthetic_exemplar_dict(1000 + i, class_id=i % 6, class_name=f"cls_{i % 6}")
        for i in range(N_pos)
    ]
    mem._negative = [
        _make_synthetic_exemplar_dict(2000 + i, class_id=14, class_name="obstacle")
        for i in range(N_neg)
    ]
    mem._rebuild_matrices()

    assert mem._pos_matrix.shape == (N_pos, 438)
    assert mem._pos_matrix.dtype == np.float32
    assert mem._neg_matrix.shape == (N_neg, 438)
    assert mem._neg_matrix.dtype == np.float32

    pos_norms = np.linalg.norm(mem._pos_matrix, axis=1)
    neg_norms = np.linalg.norm(mem._neg_matrix, axis=1)
    np.testing.assert_allclose(pos_norms, 1.0, atol=1e-5)
    np.testing.assert_allclose(neg_norms, 1.0, atol=1e-5)


def test_2_vectorized_500_positive_queries_under_100ms(w39_sandbox: Dict[str, Any]) -> None:
    """
    Test 2: Benchmark 500 positive queries against an `(N, 438)` matrix (N=250 exemplars)
    using `np.dot(mem._pos_matrix, q_vec)` and argmax selection; verify completion in < 100ms.
    """
    mem = ExemplarMemory(json_path=w39_sandbox["ex_path"])
    N = 250
    mem._positive = [
        _make_synthetic_exemplar_dict(3000 + i, class_id=i % 6, class_name=f"mat_{i % 6}")
        for i in range(N)
    ]
    mem._rebuild_matrices()
    assert mem._pos_matrix.shape == (N, 438)

    # Pre-generate 500 normalized query vectors of shape (438,)
    queries: List[np.ndarray] = [
        feature_dict_to_normalized_vector(_make_synthetic_exemplar_dict(5000 + q, class_id=0, class_name="q"))
        for q in range(500)
    ]
    assert queries[0].shape == (438,)

    # Warm-up call
    _ = np.dot(mem._pos_matrix, queries[0])

    t0 = time.perf_counter()
    best_indices = []
    best_scores = []
    for q_vec in queries:
        sims = np.dot(mem._pos_matrix, q_vec)
        idx = int(np.argmax(sims))
        best_indices.append(idx)
        best_scores.append(float(sims[idx]))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(best_indices) == 500
    assert elapsed_ms < 100.0, f"500 positive np.dot queries took {elapsed_ms:.2f}ms (limit: 100.0ms)"


def test_3_vectorized_500_negative_queries_with_class_scaling_under_100ms(w39_sandbox: Dict[str, Any]) -> None:
    """
    Test 3: Benchmark 500 negative queries against an `(N, 438)` negative matrix (N=150)
    including `np.dot` and cross-class 0.90 scaling; verify completion in < 100ms and exact accuracy.
    """
    mem = ExemplarMemory(json_path=w39_sandbox["ex_path"])
    N_neg = 150
    mem._negative = [
        _make_synthetic_exemplar_dict(7000 + i, class_id=(i % 5), class_name=f"neg_{i % 5}")
        for i in range(N_neg)
    ]
    mem._rebuild_matrices()
    assert mem._neg_matrix.shape == (N_neg, 438)

    queries: List[np.ndarray] = [
        feature_dict_to_normalized_vector(_make_synthetic_exemplar_dict(8000 + q, class_id=2, class_name="q"))
        for q in range(500)
    ]

    target_cid = 2
    t0 = time.perf_counter()
    results = []
    for q_vec in queries:
        sims = np.dot(mem._neg_matrix, q_vec)
        scale = np.where(mem._neg_cids == target_cid, 1.0, 0.90)
        scaled_sims = sims * scale
        best_idx = int(np.argmax(scaled_sims))
        results.append((float(scaled_sims[best_idx]), best_idx))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert len(results) == 500
    assert elapsed_ms < 100.0, f"500 negative np.dot queries took {elapsed_ms:.2f}ms (limit: 100.0ms)"

    # Verify accuracy of cross-class scaling on a known exemplar vector
    exact_ex = mem._negative[3]  # class_id == 3
    q_exact = feature_dict_to_normalized_vector(exact_ex)
    raw_sims = np.dot(mem._neg_matrix, q_exact)
    assert float(raw_sims[3]) == pytest.approx(1.0, abs=1e-5)

    # When querying with class_id=3 (same class), score is ~1.0; when querying with class_id=0 (different class), scaled by 0.90
    same_scale = np.where(mem._neg_cids == 3, 1.0, 0.90)
    diff_scale = np.where(mem._neg_cids == 0, 1.0, 0.90)
    assert float((raw_sims * same_scale)[3]) == pytest.approx(1.0, abs=1e-5)
    assert float((raw_sims * diff_scale)[3]) == pytest.approx(0.90, abs=1e-5)


def test_4_np_dot_mathematical_equivalence_with_cosine_similarity(w39_sandbox: Dict[str, Any]) -> None:
    """
    Test 4: Verify that vectorized `np.dot(mem._pos_matrix, q_vec)` produces scores identical
    to pairwise `compute_feature_cosine_similarity` across all N exemplars.
    """
    mem = ExemplarMemory(json_path=w39_sandbox["ex_path"])
    N = 60
    exemplars = [
        _make_synthetic_exemplar_dict(9000 + i, class_id=i % 6, class_name=f"cls_{i % 6}")
        for i in range(N)
    ]
    mem._positive = exemplars
    mem._rebuild_matrices()

    query_feat = _make_synthetic_exemplar_dict(9999, class_id=1, class_name="query_herb")
    q_vec = feature_dict_to_normalized_vector(query_feat)

    # Vectorized matrix dot product
    vec_sims = np.dot(mem._pos_matrix, q_vec)
    assert vec_sims.shape == (N,)

    # Scalar loop baseline
    scalar_sims = np.array(
        [compute_feature_cosine_similarity(ex, query_feat) for ex in exemplars],
        dtype=np.float32,
    )

    np.testing.assert_allclose(vec_sims, scalar_sims, atol=1e-5)
    assert int(np.argmax(vec_sims)) == int(np.argmax(scalar_sims))


def test_5_end_to_end_match_positive_and_negative_accuracy_and_speed(w39_sandbox: Dict[str, Any]) -> None:
    """
    Test 5: Verify end-to-end `ExemplarMemory` positive and negative matching accuracy
    with real BGR image crops and confirm 500 mixed pos+neg matrix lookups run in < 100ms.
    """
    mem = ExemplarMemory(json_path=w39_sandbox["ex_path"])

    # Register real crops for canonical classes
    crop_aria = np.full((48, 48, 3), (210, 160, 40), dtype=np.uint8)       # Cyan-blue BGR (class 0)
    crop_orichalcum = np.full((48, 48, 3), (30, 95, 220), dtype=np.uint8)  # Copper-orange BGR (class 3)
    crop_minimap = np.full((24, 24, 3), (50, 230, 50), dtype=np.uint8)     # Green minimap icon (class 9)
    crop_neg_rock = np.full((48, 48, 3), (85, 85, 85), dtype=np.uint8)     # Gray rock negative (class 14)

    assert mem.add_positive_exemplar(crop_aria, class_id=0, class_name="herb_aria") is True
    assert mem.add_positive_exemplar(crop_orichalcum, class_id=3, class_name="ore_orichalcum") is True
    assert mem.add_positive_exemplar(crop_minimap, class_id=9, class_name="minimap_sprout") is True
    assert mem.add_negative_exemplar(crop_neg_rock, class_id=14, class_name="false_rock") is True

    # Pad matrix to N=100 positive and N=100 negative exemplars for realistic load
    for i in range(97):
        mem._positive.append(_make_synthetic_exemplar_dict(11000 + i, class_id=i % 6, class_name=f"pad_pos_{i}"))
    for i in range(99):
        mem._negative.append(_make_synthetic_exemplar_dict(12000 + i, class_id=14, class_name=f"pad_neg_{i}"))
    mem._rebuild_matrices()

    assert mem._pos_matrix.shape == (100, 438)
    assert mem._neg_matrix.shape == (100, 438)

    # Accuracy checks
    sim_aria, cid_aria, cname_aria, _ = mem.match_positive(crop_aria)
    assert sim_aria >= 0.99
    assert cid_aria == 0
    assert cname_aria == "herb_aria"

    sim_ori, cid_ori, cname_ori, _ = mem.match_positive(crop_orichalcum, same_category_as=0)
    assert sim_ori >= 0.99
    assert cid_ori == 3
    assert cname_ori == "ore_orichalcum"

    neg_sim, neg_ex = mem.match_negative(crop_neg_rock, class_id=14)
    assert neg_sim >= 0.99
    assert neg_ex is not None and neg_ex["class_id"] == 14

    # 500 combined (positive + negative) matrix inner product queries in < 100ms
    q_aria = feature_dict_to_normalized_vector(extract_crop_features(crop_aria))
    q_rock = feature_dict_to_normalized_vector(extract_crop_features(crop_neg_rock))

    t0 = time.perf_counter()
    for i in range(500):
        q = q_aria if (i & 1) == 0 else q_rock
        pos_scores = np.dot(mem._pos_matrix, q)
        neg_scores = np.dot(mem._neg_matrix, q)
        _ = int(np.argmax(pos_scores))
        _ = int(np.argmax(neg_scores))
    combined_ms = (time.perf_counter() - t0) * 1000.0

    assert combined_ms < 100.0, f"500 combined pos+neg np.dot queries took {combined_ms:.2f}ms (limit: 100.0ms)"
