# -*- coding: utf-8 -*-
"""
tests/test_d44_vectorized_exemplar_matcher.py
=============================================
Regression tests for fix slice D44 (dataset_factory/vectorized_exemplar_matcher.py):
F-0818 consumable classes 22..24 accepted, F-0819 dedup rebuilds the matrix,
F-0903/F-1018 consumables super-category, F-0904 no phantom negative in
match_batch, F-1013 wrong-size vector rejected, F-1014 aspect-ratio weight,
F-1015 repair_pos_threshold read, F-1016 zero-vector exemplar dropped,
F-1017 thresholded match_positive, F-1019 FIFO quota eviction,
F-1078 non-numeric class_id rejected without raising.
"""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from dataset_factory.vectorized_exemplar_matcher import (
    CANONICAL_CLASSES,
    FEATURE_VECTOR_DIM,
    VectorizedExemplarMatcher,
    feature_dict_to_normalized_vector,
    get_super_category,
    normalize_l2_vector,
)


def _vec(seed: int) -> np.ndarray:
    # Deterministic unit vector on a 20-dim block; distinct seeds (mod 20)
    # give disjoint support, hence cosine 0.0, while the same seed gives 1.0.
    v = np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
    start = (seed % 20) * 20
    v[start:start + 20] = 1.0
    return (v / np.linalg.norm(v)).astype(np.float32)


def _ex(cid: int, name: str, seed: int, **extra):
    d = {"class_id": cid, "class_name": name, "feature_vector": _vec(seed)}
    d.update(extra)
    return d


def test_consumable_classes_accepted_and_categorized():
    # F-0818 / F-0903 / F-1018
    assert {22, 23, 24} <= CANONICAL_CLASSES
    assert get_super_category(22) == "consumables"
    assert get_super_category(23) == "consumables"
    assert get_super_category(24) == "consumables"
    m = VectorizedExemplarMatcher()
    assert m.add_positive_exemplar(_ex(22, "hp_potion", 1)) is True
    assert m.add_positive_exemplar(_ex(24, "buff_scroll", 2)) is True
    assert m.positive_count == 2
    sim, cid, _, _ = m.match_positive(_vec(1), min_similarity=0.0)
    assert cid == 22 and sim > 0.99


def test_dedup_hit_rebuilds_matrix_row():
    # F-0819: updating a near-duplicate must refresh the indexed vector.
    m = VectorizedExemplarMatcher()
    assert m.add_positive_exemplar(_ex(0, "herb_aria", 10)) is True
    updated = _ex(0, "herb_aria", 99)
    assert m.add_positive_exemplar(updated, dedup_threshold=0.0) is True
    assert m.positive_count == 1
    sim, cid, _, _ = m.match_positive(_vec(99), min_similarity=0.0)
    assert cid == 0 and sim > 0.99
    sim_old, _, _, _ = m.match_positive(_vec(10), min_similarity=0.0)
    assert sim_old < 0.99


def test_match_batch_no_phantom_negative():
    # F-0904: an all-negative column must not fabricate a negative exemplar.
    m = VectorizedExemplarMatcher(
        negative_exemplars=[_ex(0, "herb_aria", 20)],
    )
    # Query exactly opposite to the negative: cosine -1.0, so no negative match exists.
    (res,) = m.match_batch([-_vec(20)])
    assert res["negative_similarity"] == 0.0
    assert res["negative_exemplar"] is None
    assert res["is_suppressed"] is False


def test_wrong_size_vector_rejected():
    # F-1013
    with pytest.raises(ValueError):
        normalize_l2_vector(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError):
        normalize_l2_vector(np.zeros(439, dtype=np.float32))
    assert normalize_l2_vector(np.zeros(0, dtype=np.float32)).shape == (FEATURE_VECTOR_DIM,)
    assert normalize_l2_vector(_vec(5)).shape == (FEATURE_VECTOR_DIM,)


def test_aspect_ratio_component_within_design_weight():
    # F-1014: with every component at full strength, the aspect-ratio bin
    # contributes at most its 0.15 design weight of the squared norm
    # (old / 2.0 scaling let it reach ~79%).
    feat = {"hsv_hist": [1.0] * 432, "aspect_ratio": 10.0, "texture": [1.0] * 5}
    v = feature_dict_to_normalized_vector(feat)
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5
    assert float(v[432]) ** 2 <= 0.15 + 1e-6
    assert float(v[432]) <= np.sqrt(0.15) + 1e-6


def test_repair_threshold_is_read():
    # F-1015
    m = VectorizedExemplarMatcher(repair_pos_threshold=0.88)
    m.add_positive_exemplar(_ex(0, "herb_aria", 30))
    weak = m.match_positive_for_repair(_vec(31))
    assert weak.matched is False
    strong = m.match_positive_for_repair(_vec(30))
    assert strong.matched is True and strong.similarity > 0.99


def test_zero_vector_exemplar_dropped_from_index():
    # F-1016
    m = VectorizedExemplarMatcher(
        positive_exemplars=[
            {"class_id": 0, "class_name": "herb_aria", "feature_vector": np.zeros(FEATURE_VECTOR_DIM)},
            _ex(1, "herb_mela", 40),
        ]
    )
    assert m.positive_count == 1
    sim, cid, _, _ = m.match_positive(_vec(40), min_similarity=0.0)
    assert cid == 1


def test_match_positive_is_thresholded():
    # F-1017
    m = VectorizedExemplarMatcher()
    m.add_positive_exemplar(_ex(0, "herb_aria", 50))
    sim, cid, _, _ = m.match_positive(_vec(51))
    assert (sim, cid) == (0.0, None)
    sim, cid, _, _ = m.match_positive(_vec(50))
    assert cid == 0 and sim > 0.99


def test_quota_evicts_oldest_insertion_order_on_timestamp_tie():
    # F-1019: timestamp-less exemplars tie at 0.0; eviction must be FIFO.
    m = VectorizedExemplarMatcher()
    seeds = [60, 61, 62]
    for s in seeds:
        assert m.add_positive_exemplar(_ex(0, "herb_aria", s), max_per_class=2) is True
    assert m.positive_count == 2
    sim, cid, _, _ = m.match_positive(_vec(60), min_similarity=0.0)
    assert sim < 0.99  # seed 60 (first inserted) was evicted
    sim, _, _, _ = m.match_positive(_vec(62), min_similarity=0.0)
    assert sim > 0.99


def test_non_numeric_class_id_returns_false():
    # F-1078
    m = VectorizedExemplarMatcher()
    assert m.add_positive_exemplar({"class_id": "hp_potion", "class_name": "hp_potion", "feature_vector": _vec(70)}) is False
    assert m.add_negative_exemplar({"class_id": None, "class_name": "x", "feature_vector": _vec(71)}) is False
    assert m.positive_count == 0 and m.negative_count == 0
