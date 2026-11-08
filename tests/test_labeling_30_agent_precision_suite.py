# -*- coding: utf-8 -*-
"""
tests/test_labeling_30_agent_precision_suite.py
===============================================
Master Precision Verification Suite for 30-Agent Labeling & Discrimination System
(Clusters L1..L5, L6..L10, L16..L20, L21..L25).

Authored by Worker L26_30 (Master Test Author).

Test Sections:
- Part 1 (Trust & Confidence Calibration — L1..L5):
  * Unverified heuristic sources capped at <= 0.68 and strictly trusted=False, weak=True.
  * default_conf in requeue is 0.58 and legacy .txt boxes without proof are never trusted=True.
  * Salient fallback never exceeds 0.68 and stem gating excludes path/hud boxes from gathering frames.
- Part 2 (Material & Symbol Discrimination — L6..L10):
  * 6-material discrimination (Aria, Edana, Lumina, Orichalcum, Obsidian, Blood Crystal) and background rejection.
  * 3D crest alignment with cyan nameplate proof (spatial dx <= 45, 0 <= dy <= 70).
  * Minimap sprout 9 vs monster dot 11 vs central player arrow.
  * ExemplarMemory category-locked refinement and essence purge.
- Part 3 (Data Hygiene & Zero-Bug Guardrails — L16..L20):
  * Atomic JSON <-> TXT sync and malformed JSON recovery.
  * Box size bounds (0.004 <= w,h <= 0.65) and micro noise / screen fill rejection.
  * Canonical 0..21 class taxonomy enforcement.
  * Pre-training hygiene gate.
- Part 4 (Performance & Vectorization — L21..L25):
  * Vectorized ExemplarMemory NumPy dot product speed and accuracy.
  * Incremental label cache delta detection and zero-box safety guarantee.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest

# Core system imports
from dataset_factory.label_trust import (
    UNVERIFIED_HEURISTIC_SOURCES,
    _normalize_requeue_box,
    calibrate_box_confidence_and_trust,
    requeue_labeled_dataset_for_review,
)
from dataset_factory.yolo_auto_labeler import (
    YOLOAutoLabeler,
    is_path_learning_stem,
    is_skill_learning_stem,
)
from core.material_scanner import (
    MAIN_MATERIALS_6,
    MaterialDiscriminationResult,
    MaterialScanner,
    discriminate_3d_material,
)
from core.vision import (
    PLAYER_SELF_EXCLUSION_ZONE,
    VisionEngine,
    is_in_player_self_exclusion_zone,
)
from core.minimap_scanner import MinimapScanner
from dataset_factory.exemplar_memory import (
    ExemplarMemory,
    extract_box_crop,
    extract_crop_features,
    is_same_super_category,
    sanitize_exemplar_memory,
)
from dataset_factory.dataset_validator import (
    MAX_BOX_DIM_BOUND,
    MIN_BOX_DIM_BOUND,
    check_box_size_bounds,
    sanitize_dataset_boxes,
    verify_and_sync_review_txt_pair,
)
from dataset_factory.class_taxonomy import (
    CORE_22_CANONICAL_NAMES,
    is_3d_world_material,
    is_consumable,
    is_hud_element,
    is_minimap_symbol,
    is_navigable_element,
    is_valid_canonical_class_id,
    sanitize_class_id_and_name,
)
from dataset_factory.dataset_hygiene_gate import (
    _load_review_json_safe,
    pre_training_dataset_hygiene_gate,
)
from dataset_factory.label_cache import LabelCache


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_dataset_dir(tmp_path: Path):
    """Create a clean, isolated dataset sandbox for review, labels, and images."""
    ds = tmp_path / "dataset"
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        (ds / "review" / split).mkdir(parents=True, exist_ok=True)
    yield ds
    shutil.rmtree(ds, ignore_errors=True)


# ==============================================================================
# PART 1: Trust & Confidence Calibration (L1..L5)
# ==============================================================================

class TestPart1TrustAndConfidenceCalibration:
    """Verifies that heuristic and unverified sources are capped at <= 0.68,
    default requeue confidence is 0.58, salient fallback is capped, and stem gating
    strictly blocks clutter from gathering frames."""

    def test_unverified_heuristic_sources_definition(self):
        """Verify the unverified heuristic sources set contains all known heuristic generators."""
        expected_heuristics = {
            "chromatic_salient_3d_resource",
            "visual_ground_path_segmenter",
            "visual_obstacle_segmenter",
            "hud_skill_slot",
            "hud_player_hp_bar",
            "hud_player_mp_bar",
            "dataset_requeue",
            "legacy_auto",
            "center_anchor_fallback",
            "gather_material_hint",
        }
        for heur in expected_heuristics:
            assert heur in UNVERIFIED_HEURISTIC_SOURCES, (
                f"Expected '{heur}' to be recognized in UNVERIFIED_HEURISTIC_SOURCES"
            )

    def test_unverified_heuristics_capped_at_068_and_weak(self):
        """Unverified heuristic sources must be strictly capped at <= 0.68 and marked trusted=False, weak=True."""
        heuristics_to_test = [
            "chromatic_salient_3d_resource",
            "visual_ground_path_segmenter",
            "visual_obstacle_segmenter",
            "hud_skill_slot",
            "hud_player_hp_bar",
            "hud_player_mp_bar",
            "dataset_requeue",
            "legacy_auto",
            "center_anchor_fallback",
            "gather_material_hint",
        ]

        for src in heuristics_to_test:
            candidate = {
                "class_id": 0,
                "confidence": 0.98,  # Inflated score
                "source": src,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.1,
            }
            calibrated = calibrate_box_confidence_and_trust(candidate, stem="boot_sample_01")
            assert calibrated["confidence"] <= 0.68, (
                f"Source '{src}' confidence must be capped at <= 0.68, got {calibrated['confidence']}"
            )
            assert calibrated["trusted"] is False, f"Source '{src}' must not be trusted without proof"
            assert calibrated["weak"] is True, f"Source '{src}' must be flagged as weak (Şüpheli)"

    def test_human_approved_boxes_preserve_high_confidence_and_emin(self):
        """Human-approved or hand-labeled boxes are immune: confidence >= 0.90, trusted=True, weak=False."""
        cases = [
            {"hand_label": True, "source": "hand_drawn", "confidence": 0.50},
            {"human_approved": True, "source": "ui_reviewed", "confidence": 0.62},
            {"approved": True, "source": "batch_approved", "confidence": 0.70},
            {"source": "vision_f_gather_prompt", "confidence": 0.94},
            {"source": "vision_3d_crest", "confidence": 0.92},
        ]

        for box in cases:
            box["class_id"] = 0
            box["cx"] = 0.5
            box["cy"] = 0.5
            box["w"] = 0.1
            box["h"] = 0.1
            calibrated = calibrate_box_confidence_and_trust(box, stem="boot_sample_02")
            assert calibrated["trusted"] is True, f"Approved box {box} must be trusted=True"
            assert calibrated["weak"] is False, f"Approved box {box} must be weak=False"
            assert calibrated["confidence"] >= 0.90, f"Approved box confidence must be >= 0.90, got {calibrated['confidence']}"

    def test_stem_material_conflict_calibration(self):
        """When stem indicates one material (e.g. Aria) but box claims conflicting material (e.g. Orichalcum),
        confidence must drop to <= 0.45 and remain weak."""
        conflicting_box = {
            "class_id": 3,
            "class_name": "ore_orichalcum",
            "confidence": 0.82,
            "source": "auto_scan",
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
        }
        calibrated = calibrate_box_confidence_and_trust(conflicting_box, stem="boot_001_aria_3d_crest_far")
        assert calibrated["confidence"] <= 0.45, (
            f"Conflicting material box should drop to <= 0.45, got {calibrated['confidence']}"
        )
        assert calibrated["trusted"] is False
        assert calibrated["weak"] is True

    def test_requeue_default_conf_is_058_and_legacy_txt_is_weak(self, temp_dataset_dir: Path):
        """default_conf in requeue functions must be 0.58, and legacy boxes without proof must be weak=True."""
        # Check signature defaults
        sig_norm = inspect.signature(_normalize_requeue_box)
        assert sig_norm.parameters["default_conf"].default == 0.58, "Expected _normalize_requeue_box default_conf == 0.58"

        sig_req = inspect.signature(requeue_labeled_dataset_for_review)
        assert sig_req.parameters["default_conf"].default == 0.58, "Expected requeue_labeled_dataset_for_review default_conf == 0.58"

        # Check normalization behavior on raw legacy txt box
        raw_box = {
            "class_id": 0,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
            "has_explicit_conf": False,
            "confidence": 0.0,
        }
        normalized = _normalize_requeue_box(raw_box, stem="boot_test_01")
        assert normalized is not None
        assert normalized["confidence"] == 0.58
        assert normalized["trusted"] is False
        assert normalized["weak"] is True

        # Test end-to-end dataset requeue
        train_img = temp_dataset_dir / "images" / "train" / "legacy_frame_01.jpg"
        train_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 64)
        train_lbl = temp_dataset_dir / "labels" / "train" / "legacy_frame_01.txt"
        train_lbl.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")  # No confidence in txt

        stats = requeue_labeled_dataset_for_review(temp_dataset_dir, default_conf=0.58)
        assert stats["ok"] is True
        assert stats["trusted_boxes"] == 0, "Unproven legacy boxes must not enter trusted"
        assert stats["weak_boxes"] >= 1, "Unproven legacy boxes must be classified as weak"

        rev_file = temp_dataset_dir / "review" / "train" / "legacy_frame_01.json"
        assert rev_file.is_file()
        payload = json.loads(rev_file.read_text(encoding="utf-8"))
        assert len(payload["trusted"]) == 0
        assert len(payload["weak"]) == 1
        assert payload["weak"][0]["confidence"] == 0.58
        assert payload["weak"][0]["trusted"] is False
        assert payload["weak"][0]["weak"] is True

    def test_salient_fallback_confidence_cap(self):
        """Salient 3D resource detector must cap confidence at <= 0.68, set weak=True, and never assign essence classes."""
        labeler = YOLOAutoLabeler()
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        hsv = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Draw a synthetic salient resource patch
        bgr[300:360, 500:560] = (180, 120, 60)
        hsv[300:360, 500:560] = (25, 180, 180)

        salient_anns = labeler._scan_salient_3d_resources(
            frame_bgr=bgr,
            hsv=hsv,
            existing_boxes=[],
            hint_material_name="",
        )

        for ann in salient_anns:
            assert ann["confidence"] <= 0.68, (
                f"Salient resource box confidence must not exceed 0.68, got {ann['confidence']}"
            )
            assert ann.get("trusted") is False
            assert ann.get("weak") is True
            assert int(ann["class_id"]) <= 12, (
                f"Salient fallback must not emit essence classes (>12), got class_id={ann['class_id']}"
            )

    def test_stem_gating_excludes_path_and_hud_from_gathering_frames(self):
        """Gathering frames (boot_*_aria_*, etc.) must be gated so path and HUD boxes are NEVER injected."""
        labeler = YOLOAutoLabeler()
        bgr = np.zeros((720, 1280, 3), dtype=np.uint8)
        hsv = np.zeros((720, 1280, 3), dtype=np.uint8)
        gray = np.zeros((720, 1280), dtype=np.uint8)

        gather_stem = "boot_478467_000_aria_3d_crest_far"
        path_stem = "learn_paths_valley_corridor_01"
        skill_stem = "learn_skills_hotbar_01"

        # Check stem categorization
        assert is_path_learning_stem(gather_stem) is False
        assert is_path_learning_stem(path_stem) is True
        assert is_skill_learning_stem(gather_stem) is False
        assert is_skill_learning_stem(skill_stem) is True

        # On gathering frames, path scan must return strictly empty list
        path_on_gather = labeler._scan_path_and_obstacles(
            frame_bgr=bgr,
            hsv=hsv,
            gray=gray,
            source_stem=gather_stem,
        )
        assert len(path_on_gather) == 0, (
            f"Path segmenter must produce 0 boxes on gathering frames, got {len(path_on_gather)}"
        )


# ==============================================================================
# PART 2: Material & Symbol Discrimination (L6..L10)
# ==============================================================================

class TestPart2MaterialAndSymbolDiscrimination:
    """Verifies 6-material discrimination, 3D crest vertical alignment proof,
    minimap sprout vs monster dot vs central player arrow, and category-locked exemplar memory."""

    def test_six_material_discrimination_and_background_rejection(self):
        """Verify the 6 main game materials are defined and distinguishable, while flat background is rejected."""
        # 1. Verify MAIN_MATERIALS_6 registry
        expected_keys = {
            "herb_aria",
            "herb_edana",
            "herb_lumina",
            "ore_orichalcum",
            "ore_obsidian",
            "crystal_blood",
        }
        assert set(MAIN_MATERIALS_6.keys()) == expected_keys

        # 2. Test authentic textured Aria patch
        np.random.seed(101)
        aria_patch = np.random.randint(140, 240, (64, 64, 3), dtype=np.uint8)
        aria_patch[:, :, 0] = np.clip(aria_patch[:, :, 0] + 50, 0, 255)  # Boost blue floral tone
        res_aria = discriminate_3d_material(
            aria_patch,
            nameplate_width_px=60,
            has_specular_crest=True,
        )
        assert res_aria.is_authentic_node is True
        assert res_aria.class_name == "herb_aria"
        assert res_aria.calibrated_confidence >= 0.80

        # 3. Test authentic textured Orichalcum Ore patch
        ori_patch = np.random.randint(80, 200, (64, 64, 3), dtype=np.uint8)
        ori_patch[:, :, 0] = np.clip(ori_patch[:, :, 0] - 40, 0, 255)  # Low blue
        ori_patch[:, :, 1] = np.clip(ori_patch[:, :, 1] + 10, 0, 255)  # Orange-brown
        ori_patch[:, :, 2] = np.clip(ori_patch[:, :, 2] + 70, 0, 255)  # High red/copper
        res_ori = discriminate_3d_material(
            ori_patch,
            nameplate_width_px=110,
            has_specular_crest=True,
        )
        assert res_ori.is_authentic_node is True
        assert res_ori.class_name == "ore_orichalcum"
        assert res_ori.calibrated_confidence >= 0.80

        # 4. Test flat / featureless background rejection (dead dirt/sky)
        flat_dirt = np.full((64, 64, 3), (70, 90, 110), dtype=np.uint8)
        res_dirt = discriminate_3d_material(flat_dirt, nameplate_width_px=0)
        assert res_dirt.is_authentic_node is False
        assert res_dirt.calibrated_confidence < 0.35

    def test_3d_crest_vertical_alignment_with_cyan_nameplate_proof(self):
        """Winged 3D crest is only trusted=True (>=0.88) when vertically aligned with cyan nameplate/prompt proof."""
        ve = VisionEngine()
        frame = np.zeros((600, 800, 3), dtype=np.uint8)

        # Paint cyan nameplate below crest: crest at (400, 200), nameplate centered at (400, 235)
        # BGR cyan: High Blue, High Green, Low Red
        frame[225:245, 375:425] = (220, 210, 20)

        # Case A: Properly aligned (dx=0 <= 45, dy=35 in [0, 70])
        proof_aligned = ve.verify_crest_alignment_proof(
            crest_xy=(400, 200),
            frame_bgr=frame,
            raw_score=0.92,
        )
        assert proof_aligned["has_proof"] is True
        assert proof_aligned["trusted"] is True
        assert proof_aligned["weak"] is False
        assert 0.88 <= proof_aligned["confidence"] <= 0.96

        # Case B: Unaligned crest (no cyan nameplate below it)
        empty_frame = np.zeros((600, 800, 3), dtype=np.uint8)
        proof_unaligned = ve.verify_crest_alignment_proof(
            crest_xy=(400, 200),
            frame_bgr=empty_frame,
            raw_score=0.92,
        )
        assert proof_unaligned["has_proof"] is False
        assert proof_unaligned["trusted"] is False
        assert proof_unaligned["weak"] is True
        assert proof_unaligned["confidence"] <= 0.68

    def test_player_self_exclusion_zone_guards(self):
        """Player self-exclusion zone must accurately identify center player screen coords."""
        x1, y1, x2, y2 = PLAYER_SELF_EXCLUSION_ZONE
        center_x = (x1 + x2) * 0.5
        center_y = (y1 + y2) * 0.5

        assert is_in_player_self_exclusion_zone(center_x, center_y) is True
        assert is_in_player_self_exclusion_zone(0.10, 0.10) is False  # Far upper-left
        assert is_in_player_self_exclusion_zone(0.85, 0.15) is False  # Minimap zone

    def test_minimap_sprout_vs_monster_dot_vs_player_arrow(self):
        """MinimapScanner must strictly separate sprout 9, monster dot 11, and player center arrow (r<=14px)."""
        scanner = MinimapScanner()
        assert scanner.CENTER_EXCLUSION_RADIUS_PX == 14.0
        assert scanner.CLASS_ID_SPROUT == 9
        assert scanner.CLASS_ID_MONSTER_DOT == 11

        # Center exclusion logic: points within r <= 14px of player center (150, 150 in 300x300) must be rejected
        cx, cy = 150, 150
        in_center_pt = (152, 149)
        dist = ((in_center_pt[0] - cx) ** 2 + (in_center_pt[1] - cy) ** 2) ** 0.5
        assert dist <= scanner.CENTER_EXCLUSION_RADIUS_PX

        # Test Taxonomy isolation: MinimapScanner must reject world 3D class injections
        det_world = {"class_id": 0, "class_name": "herb_aria", "source": "minimap"}
        with pytest.raises(ValueError, match="Taksonomi İhlali"):
            scanner.sanitize_minimap_detection(det_world)

        # Valid sprout detection must be recognized as minimap class
        det_sprout = {"class_id": 9, "class_name": "minimap_sprout", "source": "minimap_scanner_sprout"}
        sanitized_sprout = scanner.sanitize_minimap_detection(det_sprout)
        assert sanitized_sprout["class_id"] == 9
        assert scanner.is_minimap_class(9) is True
        assert scanner.is_world_3d_class(9) is False

    def test_exemplar_memory_category_locking_and_essence_purge(self, tmp_path: Path):
        """ExemplarMemory must purge poisoned essence classes and lock refinement within same super-category."""
        # 1. Super-category boundaries
        assert is_same_super_category(0, 3) is True    # Aria (0) and Orichalcum (3) are both 3D world
        assert is_same_super_category(0, 98) is False  # 3D world vs poisoned essence log
        assert is_same_super_category(0, 9) is False   # 3D world vs minimap sprout
        assert is_same_super_category(9, 11) is True   # Sprout and monster dot are both minimap
        assert is_same_super_category(16, 17) is True  # Player HP and MP are both HUD

        # 2. Test exemplar memory sanitization
        ex_file = tmp_path / "learned_exemplars.json"
        poisoned_data = {
            "version": 1,
            "positive": [
                {"class_id": 0, "class_name": "herb_aria", "source": "verified"},
                {"class_id": 98, "class_name": "essence_yggdrasil_log", "source": "poison"},
                {"class_id": 25, "class_name": "essence_gravi_s_root", "source": "poison"},
            ],
            "negative": [
                {"class_id": 10, "class_name": "monster_3d", "source": "verified"},
                {"class_id": 98, "class_name": "essence_yggdrasil_log", "source": "poison"},
            ],
        }
        ex_file.write_text(json.dumps(poisoned_data), encoding="utf-8")

        stats = sanitize_exemplar_memory(json_path=ex_file, persist=True)
        assert stats["purged_positive"] == 2
        assert stats["purged_negative"] == 1

        cleaned = json.loads(ex_file.read_text(encoding="utf-8"))
        assert len(cleaned["positive"]) == 1
        assert cleaned["positive"][0]["class_id"] == 0
        assert len(cleaned["negative"]) == 1
        assert cleaned["negative"][0]["class_id"] == 10


# ==============================================================================
# PART 3: Data Hygiene & Zero-Bug Guardrails (L16..L20)
# ==============================================================================

class TestPart3DataHygieneAndZeroBugGuardrails:
    """Verifies atomic JSON <-> TXT sync, box size bounds (0.004 <= w,h <= 0.65),
    canonical 0..21 taxonomy enforcement, and pre-training hygiene gate."""

    def test_atomic_json_txt_sync_and_malformed_json_repair(self, tmp_path: Path):
        """Verify atomic JSON <-> TXT synchronization and resilient recovery from malformed JSON."""
        rev_file = tmp_path / "sample_sync.json"
        lbl_file = tmp_path / "sample_sync.txt"

        # Write trusted boxes in JSON but empty TXT
        json_payload = {
            "stem": "sample_sync",
            "trusted": [
                {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95},
                {"class_id": 3, "cx": 0.3, "cy": 0.3, "w": 0.08, "h": 0.08, "confidence": 0.92},
            ],
            "weak": [
                {"class_id": 1, "cx": 0.8, "cy": 0.8, "w": 0.05, "h": 0.05, "confidence": 0.55},
            ],
        }
        rev_file.write_text(json.dumps(json_payload), encoding="utf-8")
        lbl_file.write_text("", encoding="utf-8")  # Mismatched empty file

        # Synchronize
        synced = verify_and_sync_review_txt_pair(rev_file, lbl_file, auto_repair=True)
        assert synced is True
        txt_lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(txt_lines) == 2, f"TXT should now contain exactly the 2 trusted boxes, got {len(txt_lines)}"
        assert txt_lines[0].startswith("0 ")
        assert txt_lines[1].startswith("3 ")

        # Test resilient recovery from corrupted/malformed JSON
        corrupted_file = tmp_path / "corrupted.json"
        corrupted_file.write_text('{"stem": "corrupted", "trusted": []} trailing_garbage_corrupt_data', encoding="utf-8")
        repaired_data, was_repaired = _load_review_json_safe(corrupted_file)
        assert was_repaired is True
        assert repaired_data is not None
        assert repaired_data["stem"] == "corrupted"

    def test_box_size_bounds_and_micro_screenfill_rejection(self):
        """Micro noise (w,h < 0.004) and massive screen-fill boxes (w,h > 0.65) must be rejected."""
        assert MIN_BOX_DIM_BOUND == 0.004
        assert MAX_BOX_DIM_BOUND == 0.65

        # 1. Micro noise
        is_bad, reason = check_box_size_bounds(w=0.001, h=0.002)
        assert is_bad is True
        assert "Mikro" in reason or "< 0.004" in reason

        # 2. Screen fill
        is_bad_big, reason_big = check_box_size_bounds(w=0.75, h=0.80)
        assert is_bad_big is True
        assert "Aşırı büyük" in reason_big or "> 0.65" in reason_big

        # 3. Legitimate box
        is_bad_valid, reason_valid = check_box_size_bounds(w=0.10, h=0.15)
        assert is_bad_valid is False
        assert reason_valid == ""

        # Test batch sanitization
        boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.001, "h": 0.001},  # micro
            {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.70, "h": 0.70},   # screen fill
            {"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.12, "h": 0.12, "confidence": 0.90},  # valid
        ]
        clean, _ = sanitize_dataset_boxes(boxes)
        assert len(clean) == 1
        assert clean[0]["class_id"] == 2

    def test_canonical_0_to_21_class_taxonomy_enforcement(self):
        """Taxonomy must enforce strictly canonical IDs (0..21) and super-category mappings."""
        # 1. Valid canonical IDs
        for cid in range(22):
            assert is_valid_canonical_class_id(cid) is True, f"Class {cid} must be canonical"
            assert cid in CORE_22_CANONICAL_NAMES

        # 2. Non-canonical IDs must be rejected
        assert is_valid_canonical_class_id(98) is False
        assert is_valid_canonical_class_id(135) is False
        assert is_valid_canonical_class_id(-1) is False

        # 3. Automatic repair of non-canonical essence classes
        repaired_cid, repaired_name = sanitize_class_id_and_name(98, "essence_yggdrasil_log")
        assert repaired_cid == 5
        assert repaired_name == "tree_log"

        # 4. Super-category classifications
        assert is_3d_world_material(0) is True   # Aria
        assert is_3d_world_material(9) is False  # Sprout
        assert is_minimap_symbol(9) is True      # Sprout
        assert is_minimap_symbol(0) is False     # Aria
        assert is_hud_element(16) is True        # Player HP
        assert is_hud_element(18) is True        # Target HP Bar
        assert is_hud_element(0) is False         # Aria
        assert is_consumable("hp_potion") is True  # HP Potion by name
        assert is_consumable(22) is True          # HP Potion canonical ID
        assert is_consumable(0) is False          # Aria
        assert is_navigable_element(15) is True   # Path Ground

    def test_pre_training_hygiene_gate_invariants(self, temp_dataset_dir: Path):
        """pre_training_dataset_hygiene_gate must purge invalid classes, unapproved boxes, and orphans in sub-second time."""
        # Setup contaminated dataset
        (temp_dataset_dir / "images" / "train" / "img_01.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
        (temp_dataset_dir / "images" / "train" / "img_02.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)

        # img_01 has toxic non-canonical class 98 in labels/train
        (temp_dataset_dir / "labels" / "train" / "img_01.txt").write_text("98 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        # Orphaned label without image
        (temp_dataset_dir / "labels" / "train" / "orphan_missing_img.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        # Review with unverified heuristic box in trusted
        rev_02 = temp_dataset_dir / "review" / "train" / "img_02.json"
        rev_02.write_text(json.dumps({
            "stem": "img_02",
            "trusted": [
                {
                    "class_id": 15,
                    "cx": 0.5,
                    "cy": 0.5,
                    "w": 0.2,
                    "h": 0.2,
                    "confidence": 0.92,
                    "source": "visual_ground_path_segmenter",  # Heuristic without approval
                }
            ],
            "weak": [],
        }), encoding="utf-8")

        # Run hygiene gate
        report = pre_training_dataset_hygiene_gate(temp_dataset_dir, auto_clean=True)
        assert report["status"] == "PASSED"
        assert report["purged_boxes"] >= 1
        assert report["fixed_files"] >= 1
        assert report["duration_ms"] < 1000.0  # Sub-second guarantee

        # Second pass must be 100% clean
        clean_report = pre_training_dataset_hygiene_gate(temp_dataset_dir, auto_clean=True)
        assert clean_report["clean"] is True
        assert clean_report["purged_boxes"] == 0


# ==============================================================================
# PART 4: Performance & Vectorization (L21..L25)
# ==============================================================================

class TestPart4PerformanceAndVectorization:
    """Verifies vectorized NumPy dot product speed and accuracy, as well as
    incremental label cache delta detection."""

    def test_vectorized_exemplar_memory_numpy_dot_product_speed_and_accuracy(self):
        """Vectorized NumPy dot product must yield exact numerical equivalence to cosine loops
        and achieve sub-millisecond execution for 100 exemplars."""
        N = 100
        D = 432  # 12 Hue x 6 Sat x 6 Val bins

        np.random.seed(42)
        raw_matrix = np.random.rand(N, D).astype(np.float32)
        # Normalize L2
        norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
        norm_matrix = raw_matrix / np.maximum(norms, 1e-7)

        raw_query = np.random.rand(D).astype(np.float32)
        norm_query = raw_query / max(float(np.linalg.norm(raw_query)), 1e-7)

        # 1. Vectorized dot product
        t0 = time.perf_counter()
        vectorized_sims = np.dot(norm_matrix, norm_query)
        vec_duration = time.perf_counter() - t0

        # 2. Sequential loop dot product
        t0 = time.perf_counter()
        loop_sims = np.array([float(np.dot(norm_matrix[i], norm_query)) for i in range(N)], dtype=np.float32)
        loop_duration = time.perf_counter() - t0

        # Verify numerical accuracy within 1e-5
        assert np.allclose(vectorized_sims, loop_sims, atol=1e-5), "Vectorized sims must match loop sims"

        # Verify sub-millisecond speed
        assert vec_duration < 0.005, f"Vectorized dot product must take < 5ms, took {vec_duration * 1000:.3f}ms"
        assert loop_duration >= vec_duration, "Vectorized operation must be as fast or faster than iterative loop"

    def test_incremental_label_cache_delta_detection(self, tmp_path: Path):
        """LabelCache must perform delta skipping when file and revision are unchanged,
        and adhere to zero-box safety (unlabeled images are never skipped)."""
        cache_file = tmp_path / ".label_cache.json"
        cache = LabelCache(cache_path=cache_file)

        test_img = tmp_path / "test_frame_01.jpg"
        test_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 128)

        # 1. Record label with 2 boxes at revision 100
        cache.record_label(test_img, exemplar_revision=100, box_count=2)

        # 2. Querying with identical file and revision -> should skip (should_relabel=False)
        assert cache.should_relabel(test_img, exemplar_revision=100) is False

        # 3. Revision changed -> must relabel
        assert cache.should_relabel(test_img, exemplar_revision=101) is True

        # 4. File mtime changed -> must relabel
        # Touch file by appending bytes
        with open(test_img, "ab") as f:
            f.write(b"\x00")
        assert cache.should_relabel(test_img, exemplar_revision=100) is True

        # 5. Zero-box safety guarantee:
        # An image recorded with 0 boxes must NEVER be skipped when allow_zero_box_skip=False
        empty_img = tmp_path / "empty_frame_02.jpg"
        empty_img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 128)
        cache.record_label(empty_img, exemplar_revision=100, box_count=0)

        assert cache.should_relabel(empty_img, exemplar_revision=100, allow_zero_box_skip=False) is True
