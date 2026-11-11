# -*- coding: utf-8 -*-
"""
tests/test_w68_learned_exemplars_integrity.py
=============================================
Worker W68: Integrity verification suite for `dataset/learned_exemplars.json`.
Verifies that:
1. No poisoned `class_id >= 27` (`essence_*`) exemplars exist in positive, negative, or class dicts.
2. All feature vectors extracted from `dataset/learned_exemplars.json` are 438-dimensional and L2-normalized.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from dataset_factory.exemplar_memory import (
    CANONICAL_CLASSES,
    DEFAULT_EXEMPLAR_PATH,
    FEATURE_VECTOR_DIM,
    ExemplarMemory,
    feature_dict_to_normalized_vector,
)


@pytest.fixture(scope="module")
def learned_exemplars_data() -> Dict[str, Any]:
    """Load `dataset/learned_exemplars.json` in read-only mode."""
    assert DEFAULT_EXEMPLAR_PATH.is_file(), (
        f"Expected learned exemplars file at {DEFAULT_EXEMPLAR_PATH}"
    )
    with open(DEFAULT_EXEMPLAR_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict), "learned_exemplars.json root must be a JSON object"
    return data


def test_learned_exemplars_no_poisoned_essence_positive_entries(
    learned_exemplars_data: Dict[str, Any],
) -> None:
    """
    Test 1: Verify that no positive exemplar in `dataset/learned_exemplars.json`
    has `class_id >= 27`, non-canonical class ID, or `essence_*` class name.
    (22..26 are canonical now: hp/mp_potion, buff_scroll, kinah, player —
    Tarama K onarımı `ee3c631` ile dosyada 6 adet kanonik (22, hp_potion) var.)
    """
    positives: List[Dict[str, Any]] = learned_exemplars_data.get("positive", [])
    assert isinstance(positives, list) and len(positives) > 0, (
        "Expected non-empty positive exemplars list"
    )

    for idx, ex in enumerate(positives):
        cid = int(ex.get("class_id", -1))
        cname = str(ex.get("class_name", "")).strip().lower()
        assert cid < 27, f"Positive exemplar #{idx} has poisoned class_id={cid} (>= 27)"
        assert cid in CANONICAL_CLASSES, (
            f"Positive exemplar #{idx} has non-canonical class_id={cid}"
        )
        assert not cname.startswith("essence_") and "essence" not in cname, (
            f"Positive exemplar #{idx} has poisoned essence class_name='{cname}'"
        )


def test_learned_exemplars_no_poisoned_essence_negative_and_clean_sanitize(
    learned_exemplars_data: Dict[str, Any],
) -> None:
    """
    Test 2: Verify that no negative exemplar (or optional `classes` section) in
    `dataset/learned_exemplars.json` has `class_id >= 27` or `essence_*`, and
    `ExemplarMemory.sanitize(persist=False)` finds 0 poisoned records to purge.
    """
    negatives: List[Dict[str, Any]] = learned_exemplars_data.get("negative", [])
    assert isinstance(negatives, list) and len(negatives) > 0, (
        "Expected non-empty negative exemplars list"
    )

    for idx, ex in enumerate(negatives):
        cid = int(ex.get("class_id", -1))
        cname = str(ex.get("class_name", "")).strip().lower()
        assert cid < 27, f"Negative exemplar #{idx} has poisoned class_id={cid} (>= 27)"
        assert cid in CANONICAL_CLASSES, (
            f"Negative exemplar #{idx} has non-canonical class_id={cid}"
        )
        assert not cname.startswith("essence_") and "essence" not in cname, (
            f"Negative exemplar #{idx} has poisoned essence class_name='{cname}'"
        )

    classes_dict = learned_exemplars_data.get("classes", {})
    if isinstance(classes_dict, dict):
        for key, cinfo in classes_dict.items():
            cid = int(key) if str(key).isdigit() else int(cinfo.get("class_id", -1))
            cname = str(cinfo.get("class_name", "")).strip().lower()
            assert 0 <= cid < 27, f"classes[{key}] has poisoned class_id={cid}"
            assert "essence" not in cname, f"classes[{key}] has poisoned class_name='{cname}'"

    mem = ExemplarMemory(json_path=DEFAULT_EXEMPLAR_PATH)
    report = mem.sanitize(persist=False)
    assert report["purged_positive"] == 0, (
        f"Expected 0 purged positive exemplars, got {report['purged_positive']}"
    )
    assert report["purged_negative"] == 0, (
        f"Expected 0 purged negative exemplars, got {report['purged_negative']}"
    )


def test_raw_feature_components_match_438_dim_schema(
    learned_exemplars_data: Dict[str, Any],
) -> None:
    """
    Test 3: Verify that every exemplar's raw feature components in
    `dataset/learned_exemplars.json` strictly match the 438-dimensional layout:
    432 HSV histogram bins + 1 aspect ratio scalar + 5 texture descriptors = 438.
    """
    assert FEATURE_VECTOR_DIM == 438
    all_exemplars = (
        learned_exemplars_data.get("positive", [])
        + learned_exemplars_data.get("negative", [])
    )
    assert len(all_exemplars) > 0

    for idx, ex in enumerate(all_exemplars):
        hsv_hist = ex.get("hsv_hist")
        aspect_ratio = ex.get("aspect_ratio")
        texture = ex.get("texture")

        assert isinstance(hsv_hist, list) and len(hsv_hist) == 432, (
            f"Exemplar #{idx} hsv_hist must have 432 bins, got {len(hsv_hist) if isinstance(hsv_hist, list) else type(hsv_hist)}"
        )
        assert isinstance(aspect_ratio, (int, float)) and np.isfinite(aspect_ratio) and aspect_ratio > 0.0, (
            f"Exemplar #{idx} aspect_ratio must be a positive finite number, got {aspect_ratio}"
        )
        assert isinstance(texture, list) and len(texture) == 5, (
            f"Exemplar #{idx} texture must have 5 bins, got {len(texture) if isinstance(texture, list) else type(texture)}"
        )
        assert len(hsv_hist) + 1 + len(texture) == FEATURE_VECTOR_DIM


def test_feature_dict_to_normalized_vector_438d_and_unit_l2_norm(
    learned_exemplars_data: Dict[str, Any],
) -> None:
    """
    Test 4: Verify that converting every exemplar in `dataset/learned_exemplars.json`
    via `feature_dict_to_normalized_vector` produces a 438-dimensional float32 vector
    that is L2-normalized (`||v||_2 == 1.0`).
    """
    all_exemplars = (
        learned_exemplars_data.get("positive", [])
        + learned_exemplars_data.get("negative", [])
    )
    assert len(all_exemplars) > 0

    for idx, ex in enumerate(all_exemplars):
        vec = feature_dict_to_normalized_vector(ex)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        assert vec.shape == (438,), (
            f"Exemplar #{idx} vector shape mismatch: expected (438,), got {vec.shape}"
        )
        assert np.all(np.isfinite(vec)), f"Exemplar #{idx} vector contains NaN or Inf"

        l2_norm = float(np.linalg.norm(vec))
        assert np.isclose(l2_norm, 1.0, atol=1e-5), (
            f"Exemplar #{idx} vector is not L2-normalized: ||v||_2 = {l2_norm}"
        )


def test_exemplar_memory_pre_normalized_matrices_438d_l2_integrity() -> None:
    """
    Test 5: Verify that `ExemplarMemory` builds `(N_pos, 438)` and `(N_neg, 438)`
    feature matrices where every row is L2-normalized (`||row||_2 == 1.0`) and
    every class ID in `_pos_cids` and `_neg_cids` is strictly `< 27`.
    """
    mem = ExemplarMemory(json_path=DEFAULT_EXEMPLAR_PATH)

    assert mem._pos_matrix is not None and mem._pos_cids is not None
    assert mem._neg_matrix is not None and mem._neg_cids is not None

    n_pos = len(mem._positive)
    n_neg = len(mem._negative)
    assert n_pos > 0 and n_neg > 0

    assert mem._pos_matrix.shape == (n_pos, 438)
    assert mem._neg_matrix.shape == (n_neg, 438)

    # All class IDs strictly in canonical range [0, 26] (< 27)
    assert np.all((mem._pos_cids >= 0) & (mem._pos_cids < 27))
    assert np.all((mem._neg_cids >= 0) & (mem._neg_cids < 27))

    # Row-wise L2 norms of both positive and negative matrices must be 1.0
    pos_norms = np.linalg.norm(mem._pos_matrix, axis=1)
    neg_norms = np.linalg.norm(mem._neg_matrix, axis=1)
    assert np.allclose(pos_norms, 1.0, atol=1e-5), (
        f"Positive matrix rows not L2-normalized: min={pos_norms.min()}, max={pos_norms.max()}"
    )
    assert np.allclose(neg_norms, 1.0, atol=1e-5), (
        f"Negative matrix rows not L2-normalized: min={neg_norms.min()}, max={neg_norms.max()}"
    )
