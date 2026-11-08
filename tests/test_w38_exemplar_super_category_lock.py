# -*- coding: utf-8 -*-
"""
Unit tests for W38: ExemplarMemory Super-Category Lock.
Verifies that a 3D material bounding box is never converted into
`class_id >= 22` (`essence_*`) or any HUD class (`13, 16..21`).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pytest

from dataset_factory.exemplar_memory import (
    ExemplarMemory,
    SUPER_CATEGORY_3D_MATERIALS,
    SUPER_CATEGORY_HUD,
    extract_crop_features,
    get_super_category,
    is_same_super_category,
)


def _make_textured_frame(seed: int = 42, h: int = 240, w: int = 320) -> np.ndarray:
    """Create a deterministic BGR frame with a distinct chromatic textured region."""
    rng = np.random.default_rng(seed)
    frame = rng.integers(20, 60, size=(h, w, 3), dtype=np.uint8)
    # Draw a vivid textured patch inside [60:140, 80:180]
    patch = frame[60:140, 80:180]
    patch[:, :, 0] = 35   # Blue
    patch[:, :, 1] = 190  # Green
    patch[:, :, 2] = 95   # Red
    noise = rng.integers(0, 35, size=patch.shape, dtype=np.uint8)
    frame[60:140, 80:180] = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


def test_super_category_mapping_and_isolation() -> None:
    """
    Test 1: Verify get_super_category and is_same_super_category strictly isolate
    3D materials (0..5, 12) from HUD classes (13, 16..21) and essence_* (class_id >= 22).
    """
    for mat_cid in SUPER_CATEGORY_3D_MATERIALS:
        assert get_super_category(mat_cid) == "3d_materials"
        # Same category among 3D materials
        for other_mat_cid in SUPER_CATEGORY_3D_MATERIALS:
            assert is_same_super_category(mat_cid, other_mat_cid) is True

        # Never same category as HUD classes
        for hud_cid in SUPER_CATEGORY_HUD:
            assert get_super_category(hud_cid) == "hud"
            assert is_same_super_category(mat_cid, hud_cid) is False
            assert is_same_super_category(hud_cid, mat_cid) is False

        # Never same category as essence_* (class_id >= 22)
        for essence_cid in (22, 23, 35, 64, 105):
            assert get_super_category(essence_cid) is None
            assert is_same_super_category(mat_cid, essence_cid) is False
            assert is_same_super_category(essence_cid, mat_cid) is False


def test_refine_candidate_never_converts_3d_material_to_essence_class(tmp_path: Path) -> None:
    """
    Test 2: Even if a poisoned essence_* exemplar (class_id >= 22) with identical
    visual features is present in memory, refine_candidate and refine_annotations
    must never convert a 3D material box into class_id >= 22.
    """
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    frame = _make_textured_frame(seed=101)
    crop = frame[60:140, 80:180]
    feats = extract_crop_features(crop)
    assert feats is not None

    # Inject poisoned essence_* exemplars directly into memory and rebuild matrices
    for essence_cid in (22, 27, 50):
        mem._positive.append({
            "class_id": essence_cid,
            "class_name": f"essence_item_{essence_cid}",
            "hsv_hist": feats["hsv_hist"],
            "aspect_ratio": feats["aspect_ratio"],
            "texture": feats["texture"],
            "mean_hsv": feats["mean_hsv"],
            "timestamp": 1700000000.0,
            "source": "poisoned_test",
        })
    mem._rebuild_matrices()

    for mat_cid in SUPER_CATEGORY_3D_MATERIALS:
        candidate = {
            "class_id": mat_cid,
            "class_name": f"material_{mat_cid}",
            "bbox_xyxy": [80, 60, 180, 140],
            "confidence": 0.78,
        }
        keep, refined = mem.refine_candidate(frame, candidate)
        assert keep is True
        assert refined["class_id"] == mat_cid
        assert refined["class_id"] < 22
        assert not str(refined.get("class_name", "")).startswith("essence_")
        assert refined.get("repaired_by_exemplar") is False


def test_refine_candidate_never_converts_3d_material_to_hud_class(tmp_path: Path) -> None:
    """
    Test 3: When a HUD positive exemplar (13, 16..21) has a 100% identical crop
    to a 3D material box, refine_candidate must block the cross-category conversion.
    """
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    frame = _make_textured_frame(seed=202)
    crop = frame[60:140, 80:180]

    # Add identical crop as a HUD positive exemplar (e.g., class_id=18 skill_icon)
    added = mem.add_positive_exemplar(
        crop=crop,
        class_id=18,
        class_name="hud_skill_icon",
        source="ui_approval",
    )
    assert added is True

    for mat_cid in SUPER_CATEGORY_3D_MATERIALS:
        candidate = {
            "class_id": mat_cid,
            "class_name": f"world_mat_{mat_cid}",
            "bbox_xyxy": [80, 60, 180, 140],
            "confidence": 0.81,
        }
        keep, refined = mem.refine_candidate(frame, candidate)
        assert keep is True
        assert refined["class_id"] == mat_cid
        assert refined["class_id"] not in SUPER_CATEGORY_HUD
        assert refined.get("repaired_by_exemplar") is False

        # Also verify via refine_annotations batch call
        batch_out = mem.refine_annotations(frame, [candidate])
        assert len(batch_out) == 1
        assert batch_out[0]["class_id"] == mat_cid
        assert batch_out[0]["class_id"] not in SUPER_CATEGORY_HUD


def test_match_positive_with_same_category_lock_excludes_hud_and_essence(tmp_path: Path) -> None:
    """
    Test 4: match_positive(..., same_category_as=<3D material>) must ignore HUD
    and essence_* exemplars even when their similarity is 1.0, and only match
    within SUPER_CATEGORY_3D_MATERIALS.
    """
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    frame = _make_textured_frame(seed=303)
    crop = frame[60:140, 80:180]
    feats = extract_crop_features(crop)
    assert feats is not None

    # Register HUD exemplar (cid=17) and inject poisoned essence exemplar (cid=24)
    mem.add_positive_exemplar(crop, class_id=17, class_name="hud_mp_bar", source="ui_approval")
    mem._positive.append({
        "class_id": 24,
        "class_name": "essence_dragon_tooth",
        "hsv_hist": feats["hsv_hist"],
        "aspect_ratio": feats["aspect_ratio"],
        "texture": feats["texture"],
        "mean_hsv": feats["mean_hsv"],
        "timestamp": 1700000001.0,
        "source": "poisoned_test",
    })
    mem._rebuild_matrices()

    # Query with same_category_as=0 (mine_ore, 3D material): must return no match
    sim, best_cid, best_cname, best_ex = mem.match_positive(crop, same_category_as=0)
    assert best_cid is None
    assert best_cname is None
    assert best_ex is None
    assert sim == 0.0

    # Now add a legitimate 3D material exemplar (class_id=3) and verify intra-category match works
    mem.add_positive_exemplar(crop, class_id=3, class_name="herb_flower", source="3d_world_approval")
    sim2, best_cid2, best_cname2, _ = mem.match_positive(crop, same_category_as=0)
    assert best_cid2 == 3
    assert best_cid2 in SUPER_CATEGORY_3D_MATERIALS
    assert best_cname2 == "herb_flower"
    assert sim2 >= 0.99


def test_add_positive_and_sanitize_block_essence_and_3d_hud_poisoning(tmp_path: Path) -> None:
    """
    Test 5: Verify add_positive_exemplar rejects essence_* (class_id >= 22) and
    3D-sourced HUD classes, while sanitize() purges any disk-injected essence_* or
    3D-world HUD exemplars and preserves valid intra-category 3D material repair.
    """
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    frame = _make_textured_frame(seed=404)
    crop = frame[60:140, 80:180]
    feats = extract_crop_features(crop)
    assert feats is not None

    # 1. Direct add of essence_* (class_id >= 22) is rejected
    assert mem.add_positive_exemplar(crop, class_id=22, class_name="essence_ruby", source="ui_approval") is False
    assert mem.add_positive_exemplar(crop, class_id=5, class_name="essence_spoofed", source="ui_approval") is False

    # 2. Direct add of HUD class with 3D world source is rejected
    for hud_cid in (13, 16, 19, 21):
        assert mem.add_positive_exemplar(crop, class_id=hud_cid, class_name=f"hud_{hud_cid}", source="3d_world_harvest") is False

    # 3. Direct add of valid 3D material (class_id=4) succeeds
    assert mem.add_positive_exemplar(crop, class_id=4, class_name="wood_tree", source="3d_world_harvest") is True

    # 4. Simulate poisoned entries on disk/memory and run sanitize()
    mem._positive.append({
        "class_id": 25,
        "class_name": "essence_void_pearl",
        "hsv_hist": feats["hsv_hist"],
        "aspect_ratio": feats["aspect_ratio"],
        "texture": feats["texture"],
        "mean_hsv": feats["mean_hsv"],
        "timestamp": 1700000002.0,
        "source": "ui_approval",
    })
    mem._positive.append({
        "class_id": 19,
        "class_name": "hud_cooldown_slot",
        "hsv_hist": feats["hsv_hist"],
        "aspect_ratio": feats["aspect_ratio"],
        "texture": feats["texture"],
        "mean_hsv": feats["mean_hsv"],
        "timestamp": 1700000003.0,
        "source": "3d_world_resource",
    })
    mem._rebuild_matrices()

    summary = mem.sanitize(persist=True)
    assert summary["purged_positive"] == 2
    assert summary["remaining_positive"] == 1

    # 5. Candidate 3D material box (class_id=1) can only repair to valid 3D material (class_id=4)
    candidate = {
        "class_id": 1,
        "class_name": "stone_rock",
        "bbox_xyxy": [80, 60, 180, 140],
        "confidence": 0.75,
    }
    keep, refined = mem.refine_candidate(frame, candidate)
    assert keep is True
    assert refined["class_id"] == 4
    assert refined["class_id"] in SUPER_CATEGORY_3D_MATERIALS
    assert refined.get("repaired_by_exemplar") is True
