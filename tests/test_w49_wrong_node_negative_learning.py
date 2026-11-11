# -*- coding: utf-8 -*-
"""Unit tests for W49: Wrong node deletion negative exemplar active learning loop."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

from core.node_quality_gate import evaluate_node_candidate
from core.wrong_node_scan import (
    WrongNode,
    classify_wrong_nodes,
    purge_wrong_nodes,
    register_deleted_nodes_as_negative_exemplars,
    save_wrong_nodes_as_negative_exemplars,
)
from dataset_factory.exemplar_memory import ExemplarMemory


def _make_distinct_patch(hue: int, sat: int = 200, val: int = 180, seed: int = 42) -> np.ndarray:
    """Create a rich, textured 80x80 BGR patch that does not trigger flat-gray terrain filters by default."""
    rng = np.random.default_rng(seed)
    hsv = np.zeros((80, 80, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = sat
    # Structured gradient + noise for rich texture
    grad = np.linspace(max(20, val - 50), min(255, val + 40), 80, dtype=np.float32)
    base_v = np.tile(grad, (80, 1))
    noise = rng.normal(0.0, 18.0, (80, 80))
    hsv[:, :, 2] = np.clip(base_v + noise, 20, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Draw colorful non-circular, non-scroll, non-feather blocks for rich texture
    cv2.rectangle(bgr, (15, 20), (65, 60), (40, 200, 120), 2)
    cv2.rectangle(bgr, (28, 30), (52, 50), (180, 90, 40), -1)
    return bgr


class _DummyRouteManager:
    def __init__(self, nodes: List[dict]) -> None:
        self.nodes = list(nodes)
        self.deleted_ids: List[int] = []
        self.deleted_region: str = ""

    def get_region_nodes(self, region: str) -> List[dict]:
        return [n for n in self.nodes if n.get("region", region) == region]

    def delete_nodes(self, ids: List[int], region: str = "") -> bool:
        self.deleted_ids.extend(ids)
        self.deleted_region = region
        self.nodes = [n for n in self.nodes if n.get("id") not in ids]
        return True


def test_save_wrong_nodes_records_negative_exemplars(tmp_path: Path) -> None:
    """1. WrongNode objects passed to save_wrong_nodes_as_negative_exemplars are stored in ExemplarMemory."""
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    patch_a = _make_distinct_patch(hue=15, sat=210, val=190, seed=101)
    patch_b = _make_distinct_patch(hue=110, sat=190, val=170, seed=202)

    wrong_list = [
        WrongNode(node_id=10, reasons=["model"], class_id=10, confidence=0.88),
        WrongNode(node_id=20, reasons=["scroll"], class_id=13, confidence=0.91),
    ]
    images: Dict[int, np.ndarray] = {10: patch_a, 20: patch_b}

    saved_count = save_wrong_nodes_as_negative_exemplars(
        wrong_list,
        images_by_id=images,
        exemplar_memory=mem,
        region="Altgard",
    )

    assert saved_count == 2
    assert mem.has_negative_exemplars(class_id=10)
    assert mem.has_negative_exemplars(class_id=13)
    assert json_path.is_file()

    is_neg_a, sim_a, _ = mem.is_negative_patch(patch_a)
    is_neg_b, sim_b, _ = mem.is_negative_patch(patch_b)
    assert is_neg_a and sim_a >= 0.95
    assert is_neg_b and sim_b >= 0.95

    # Near-duplicate generalisation: the learned negative must still fire on
    # the SAME visual patch seen again with small real-world perturbations
    # (flip, 2px shift, slight brightening, shifted crop, 95% rescale) —
    # querying only the byte-identical patch proves nothing about recall.
    near_duplicates = {
        "horizontal_flip": cv2.flip(patch_a, 1),
        "shifted_2px": np.roll(patch_a, 2, axis=1),
        "brighter_+8": np.clip(patch_a.astype(np.int16) + 8, 0, 255).astype(np.uint8),
        "shifted_crop": patch_a[2:78, 2:78],
        "resized_95pct": cv2.resize(patch_a, (76, 76), interpolation=cv2.INTER_AREA),
    }
    for name, variant in near_duplicates.items():
        is_neg, sim, ex = mem.is_negative_patch(variant)
        assert is_neg and sim >= 0.95, (
            f"{name}: near-duplicate of the learned wrong patch must still "
            f"match the negative exemplar (sim={sim})"
        )
        assert ex is not None and ex["class_id"] == 10

    # Negative control: a visually different patch that was never taught as
    # wrong must NOT be flagged as a negative exemplar.
    unseen_patch = _make_distinct_patch(hue=60, sat=200, val=180, seed=999)
    is_neg_unseen, sim_unseen, _ = mem.is_negative_patch(unseen_patch)
    assert is_neg_unseen is False, f"Unseen patch must not match negatives (sim={sim_unseen})"


def test_register_deleted_nodes_maps_reasons_to_canonical_negative_classes(tmp_path: Path) -> None:
    """2. register_deleted_nodes_as_negative_exemplars handles dicts/IDs and maps reasons to canonical classes."""
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    patch_monster = _make_distinct_patch(hue=5, sat=230, val=180, seed=301)
    patch_terrain = _make_distinct_patch(hue=35, sat=160, val=150, seed=302)

    nodes_to_delete = [
        {"id": 101, "reason": "monster"},
        {"id": 102, "primary_reason": "terrain"},
    ]
    images = {101: patch_monster, 102: patch_terrain}

    saved = register_deleted_nodes_as_negative_exemplars(
        nodes_to_delete,
        region="Verteron",
        images_by_id=images,
        exemplar_memory=mem,
    )

    assert saved == 2
    neg_monster = mem.get_negative_exemplars(class_id=11)
    neg_terrain = mem.get_negative_exemplars(class_id=15)
    assert len(neg_monster) == 1
    assert neg_monster[0]["class_name"] == "minimap_monster_negative"
    assert len(neg_terrain) == 1
    assert neg_terrain[0]["class_name"] == "path_ground"


def test_purge_wrong_nodes_deletes_and_persists_negative_memory(tmp_path: Path) -> None:
    """3. purge_wrong_nodes deletes nodes via route_manager and registers them in ExemplarMemory."""
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    patch_wrong = _make_distinct_patch(hue=140, sat=220, val=175, seed=404)
    nodes = [
        {"id": 1, "map_x": 100.0, "map_y": 100.0, "region": "Altgard"},
        {"id": 2, "map_x": 300.0, "map_y": 300.0, "region": "Altgard"},
        {"id": 3, "map_x": 500.0, "map_y": 500.0, "region": "Altgard"},
    ]
    rm = _DummyRouteManager(nodes)
    wrong_nodes = [WrongNode(node_id=2, reasons=["model"], class_id=7, confidence=0.75)]

    ok, msg, count = purge_wrong_nodes(
        region="Altgard",
        wrong_nodes=wrong_nodes,
        route_manager=rm,
        images_by_id={2: patch_wrong},
        exemplar_memory=mem,
        total_nodes_count=3,
        auto_backup=False,
        save_negative_exemplars=True,
    )

    assert ok is True
    assert count == 1
    assert rm.deleted_ids == [2]
    assert "1 nokta sildi" in msg
    is_neg, sim, ex = mem.is_negative_patch(patch_wrong)
    assert is_neg is True
    assert sim >= 0.95
    # Merge-fix (O8): D42's add_negative_exemplar remaps rejected
    # 3D-material classes to ground/terrain (class 7 is a material class,
    # so the purged node is stored as path_ground/15, not as class 7).
    assert ex is not None and ex["class_id"] == 15
    assert ex["class_name"] == "path_ground"


def test_purged_node_is_flagged_by_classify_wrong_nodes_on_rescan(tmp_path: Path) -> None:
    """4. Once a wrong node is purged into ExemplarMemory, classify_wrong_nodes flags it as negative_exemplar."""
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    bad_patch = _make_distinct_patch(hue=85, sat=200, val=185, seed=505)
    good_patch = _make_distinct_patch(hue=25, sat=210, val=200, seed=606)

    nodes = [
        {"id": 11, "map_x": 100.0, "map_y": 100.0, "source_type": "3d_crest", "resource_type": "Aria"},
        {"id": 12, "map_x": 400.0, "map_y": 400.0, "source_type": "3d_crest", "resource_type": "Aria"},
    ]
    images = {11: bad_patch, 12: good_patch}

    # Before learning: without a detector firing, neither patch is rejected
    initial_wrong = classify_wrong_nodes(
        nodes,
        images_by_id=images,
        detector=None,
        exemplar_memory=mem,
        check_exemplars=True,
    )
    assert [w.node_id for w in initial_wrong] == []

    # User / scan purges node 11 as a wrong node
    save_wrong_nodes_as_negative_exemplars(
        [WrongNode(node_id=11, reasons=["model"], class_id=10, confidence=0.80)],
        images_by_id=images,
        exemplar_memory=mem,
    )

    # Rescan: node 11 is now automatically caught via ExemplarMemory negative_exemplar, while node 12 stays kept
    rescan_wrong = classify_wrong_nodes(
        nodes,
        images_by_id=images,
        detector=None,
        exemplar_memory=mem,
        check_exemplars=True,
    )
    assert len(rescan_wrong) == 1
    assert rescan_wrong[0].node_id == 11
    assert "negative_exemplar" in rescan_wrong[0].reasons


def test_evaluate_node_candidate_blocks_re_marking_deleted_wrong_node(tmp_path: Path) -> None:
    """5. Candidate gate prevents a deleted wrong node from being marked again during cartography/auto-add."""
    json_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=json_path)

    false_positive_patch = _make_distinct_patch(hue=160, sat=195, val=170, seed=707)

    # Initially accepted by quality gate when no negative exemplar exists
    verdict_before = evaluate_node_candidate(
        false_positive_patch,
        source_type="3d_world",
        resource_type="Orichalcum",
        exemplar_memory=mem,
        check_exemplars=True,
    )
    assert verdict_before.accept is True

    # Delete node and record into negative exemplar memory
    register_deleted_nodes_as_negative_exemplars(
        [{"id": 99, "class_id": 14, "reason": "model"}],
        images_by_id={99: false_positive_patch},
        exemplar_memory=mem,
    )

    # Attempting to mark/add the same visual target again is rejected by the gate
    verdict_after = evaluate_node_candidate(
        false_positive_patch,
        source_type="3d_world",
        resource_type="Orichalcum",
        exemplar_memory=mem,
        check_exemplars=True,
    )
    assert verdict_after.accept is False
    assert verdict_after.reject is True
    assert "negative_exemplar" in verdict_after.reasons
    assert verdict_after.class_id == 14
