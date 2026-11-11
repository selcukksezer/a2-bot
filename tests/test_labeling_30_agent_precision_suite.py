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
    feature_dict_to_normalized_vector,
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


def _hsv_patch(h: int, s: int, v: int, seed: int, size: int = 64) -> np.ndarray:
    """Deterministic BGR patch built from a real HSV colour: base hue with
    seeded per-pixel jitter plus a darker stripe every 8 rows for texture, so
    discriminate_3d_material sees genuine colour + edge evidence, not noise."""
    rng = np.random.default_rng(seed)
    H = np.clip(h + rng.integers(-4, 5, (size, size)), 0, 179).astype(np.uint8)
    S = np.clip(s + rng.integers(-12, 13, (size, size)), 0, 255).astype(np.uint8)
    V = np.clip(v + rng.integers(-12, 13, (size, size)), 0, 255).astype(np.uint8)
    hsv = np.stack([H, S, V], axis=2)
    hsv[::8, :, :] = np.clip(hsv[::8, :, :].astype(int) - 25, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _empty_exemplar_memory(tmp_path: Path) -> ExemplarMemory:
    """A real, empty ExemplarMemory on a tmp JSON path, so material
    discrimination runs deterministically without the global learned file."""
    return ExemplarMemory(json_path=tmp_path / "empty_exemplars.json")


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
        """P1-12 (K-32/K-34, F-1212 never inflate): human-approved or
        hand-labeled boxes are immune (trusted=True, weak=False) and keep
        their MEASURED confidence — no 0.90 inflation. A bare "approved"
        flag and a gather seed are not human decisions: weak/untrusted."""
        human_cases = [
            ({"hand_label": True, "source": "hand_drawn", "confidence": 0.50}, 0.50),
            ({"human_approved": True, "source": "ui_reviewed", "confidence": 0.62}, 0.62),
        ]
        for box, expected_conf in human_cases:
            box.setdefault("class_id", 0)
            box["cx"] = 0.5
            box["cy"] = 0.5
            box["w"] = 0.1
            box["h"] = 0.1
            calibrated = calibrate_box_confidence_and_trust(box, stem="boot_sample_02")
            assert calibrated["trusted"] is True, f"Approved box {box} must be trusted=True"
            assert calibrated["weak"] is False, f"Approved box {box} must be weak=False"
            assert calibrated["confidence"] == expected_conf, (
                f"Human box confidence must be kept as measured ({expected_conf}), "
                f"got {calibrated['confidence']}"
            )

        # Bare "approved" (pipeline flag) and a gather seed (cid 13) are
        # NOT human decisions: they stay weak with confidence kept as-is.
        for box in [
            {"approved": True, "source": "batch_approved", "confidence": 0.70},
            # Y1: the F/E gather seed is cid 13 (skill_ui_prompt) + F/E trace.
            {"class_id": 13, "source": "vision_f_gather_prompt", "confidence": 0.94},
        ]:
            box.setdefault("class_id", 0)
            box["cx"] = 0.5
            box["cy"] = 0.5
            box["w"] = 0.1
            box["h"] = 0.1
            expected_conf = box["confidence"]
            calibrated = calibrate_box_confidence_and_trust(box, stem="boot_sample_02")
            assert calibrated["trusted"] is False, f"Non-human box {box} must be trusted=False"
            assert calibrated["weak"] is True, f"Non-human box {box} must be weak=True"
            assert calibrated["confidence"] == expected_conf

        # E-07: a high-confidence vision box with no human decision is NOT
        # immune — it stays weak (flagged as a high-confidence candidate).
        # The old matrix trusted these two on confidence alone.
        for box in [
            {"source": "vision_f_gather_prompt", "confidence": 0.94},
            {"source": "vision_3d_crest", "confidence": 0.92},
        ]:
            box.update({"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1})
            calibrated = calibrate_box_confidence_and_trust(box, stem="boot_sample_02")
            assert calibrated["trusted"] is False, f"Unapproved box {box} must be trusted=False"
            assert calibrated["weak"] is True, f"Unapproved box {box} must be weak=True"
            assert calibrated.get("high_confidence_candidate") is True

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
        """default_conf in requeue functions must be 0.58; a bare legacy box
        without the requeue stamp stays weak=True in direct normalization.

        §7 P0-8 requeue kararı: requeue TXT-only kareye hand_label damgası
        vurur (diskteki 5-alanlı TXT satırı el etiketiyle ayırt edilemez)
        ve damgalı kare korumalı sayılır — kare düzeyinde atlanır, review
        JSON yazılmaz. Uçtan uca requeue artık bu kareyi kuyruğa almaz;
        0.58 weak sözleşmesi doğrudan normalize geçişinde denetlenir.
        """
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
        # §7 P0-8: TXT-only kare hand_label damgasıyla korumalı atlanır —
        # kuyruğa girmez, review JSON yazılmaz.
        assert stats["samples"] == 0, "TXT-only legacy frame is skipped as protected, never requeued"
        assert stats["boxes"] == 0
        assert stats["trusted_boxes"] == 0, "No trusted boxes: a stamp grants no trust"
        assert stats["weak_boxes"] == 0
        assert stats["skipped_protected"] == 1

        rev_file = temp_dataset_dir / "review" / "train" / "legacy_frame_01.json"
        assert not rev_file.is_file()
        assert train_lbl.read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.1\n"

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

        # The synthetic patch must actually produce annotations — otherwise the
        # loop below would pass vacuously on an empty list.
        assert len(salient_anns) >= 1, (
            "Salient scan must return >= 1 annotation for the synthetic resource patch"
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

    def test_six_material_discrimination_and_background_rejection(self, tmp_path: Path):
        """Verify ALL 6 main game materials are discriminated by the REAL
        discriminate_3d_material on genuine HSV-colour patches (one per
        material, each with its own nameplate width), while flat background
        is rejected."""
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

        # 2. One genuine colour patch per material: (key, hue, sat, val, nameplate_width_px)
        #    Hues sit inside the production HSV masks in compute_color_histogram_scores.
        memory = _empty_exemplar_memory(tmp_path)
        cases = [
            ("herb_aria", 105, 150, 220, 60),      # blue petals
            ("herb_edana", 155, 180, 220, 70),     # purple/pink petals
            ("herb_lumina", 28, 200, 230, 70),     # golden/yellow glow
            ("ore_orichalcum", 15, 180, 200, 110),  # copper/orange-brown vein
            ("ore_obsidian", 140, 100, 70, 110),   # dark violet glassy vein
            ("crystal_blood", 0, 220, 200, 100),   # crimson red crystal
        ]
        for i, (key, h, s, v, width) in enumerate(cases):
            patch = _hsv_patch(h, s, v, seed=100 + i)
            res = discriminate_3d_material(
                patch,
                nameplate_width_px=width,
                has_specular_crest=True,
                exemplar_memory=memory,
            )
            assert res.is_authentic_node is True, f"{key}: patch rejected ({res.rejection_reason})"
            assert res.class_name == key, f"Expected {key}, got {res.class_name}"
            assert res.calibrated_confidence >= 0.80, (
                f"{key}: confidence {res.calibrated_confidence} < 0.80"
            )

        # 3. Test flat / featureless background rejection (dead dirt/sky)
        flat_dirt = np.full((64, 64, 3), (70, 90, 110), dtype=np.uint8)
        res_dirt = discriminate_3d_material(flat_dirt, nameplate_width_px=0, exemplar_memory=memory)
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
        """Player self-exclusion zone must accurately identify center player screen coords.

        Points are LITERAL coordinates of the documented zone
        (x 0.43..0.57, y 0.38..0.64) — never derived from the constant under
        test, so the test cannot pass tautologically."""
        assert PLAYER_SELF_EXCLUSION_ZONE == (0.43, 0.38, 0.57, 0.64)

        # Inside: documented centre and inclusive boundaries
        assert is_in_player_self_exclusion_zone(0.50, 0.51) is True
        assert is_in_player_self_exclusion_zone(0.43, 0.38) is True
        assert is_in_player_self_exclusion_zone(0.57, 0.64) is True
        # Just outside each edge
        assert is_in_player_self_exclusion_zone(0.42, 0.51) is False
        assert is_in_player_self_exclusion_zone(0.58, 0.51) is False
        assert is_in_player_self_exclusion_zone(0.50, 0.37) is False
        assert is_in_player_self_exclusion_zone(0.50, 0.65) is False
        # Far away: upper-left and minimap zone
        assert is_in_player_self_exclusion_zone(0.10, 0.10) is False  # Far upper-left
        assert is_in_player_self_exclusion_zone(0.85, 0.15) is False  # Minimap zone

    def test_minimap_sprout_vs_monster_dot_vs_player_arrow(self):
        """MinimapScanner must strictly separate sprout 9, monster dot 11, and player center arrow (r<=14px).

        The centre exclusion is exercised through the REAL production scan
        (MinimapScanner.scan_minimap_crop) on synthetic 300x300 crops: a white
        dot ~2px from the production player centre must be swallowed by the
        centre exclusion, while an identical dot 50px away must be reported
        as a monster dot with dist_px > 14."""
        scanner = MinimapScanner()
        assert scanner.CENTER_EXCLUSION_RADIUS_PX == 14.0
        assert scanner.CLASS_ID_SPROUT == 9
        assert scanner.CLASS_ID_MONSTER_DOT == 11

        # Production player centre for a blank 300x300 crop (no arrow drawn yet).
        blank = np.full((300, 300, 3), (25, 30, 22), dtype=np.uint8)
        pcx, pcy = scanner.get_minimap_player_center(blank.shape, crop_bgr=blank)
        cx, cy = int(round(pcx)), int(round(pcy))

        def _crop_with_dot(dot_dx: int, dot_dy: int) -> np.ndarray:
            crop = np.full((300, 300, 3), (25, 30, 22), dtype=np.uint8)
            cv2.circle(crop, (cx, cy), 6, (255, 255, 255), -1)  # player arrow
            cv2.circle(crop, (cx + dot_dx, cy + dot_dy), 4, (255, 255, 255), -1)
            return crop

        # Dot ~2px from centre: inside r<=14 -> must NOT appear as a monster dot.
        near = scanner.scan_minimap_crop(_crop_with_dot(2, -1))
        near_positions = [d["pos"] for d in near["monster_dots"]]
        assert near_positions == [], (
            f"Dot at r~2px from player centre must be excluded, got {near_positions}"
        )
        assert all(d["dist_px"] > scanner.CENTER_EXCLUSION_RADIUS_PX for d in near["monster_dots"])

        # Identical dot 50px away: outside r<=14 -> must be reported as monster dot 11.
        far = scanner.scan_minimap_crop(_crop_with_dot(50, 0))
        assert len(far["monster_dots"]) == 1, f"Expected 1 far monster dot, got {far['monster_dots']}"
        far_dot = far["monster_dots"][0]
        assert far_dot["class_id"] == 11
        assert far_dot["dist_px"] > scanner.CENTER_EXCLUSION_RADIUS_PX
        assert abs(far_dot["pos"][0] - (cx + 50)) <= 2
        assert abs(far_dot["pos"][1] - cy) <= 2

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
        """pre_training_dataset_hygiene_gate must purge invalid classes, unapproved boxes, and orphans."""
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
        # duration_ms is reported for observability only — it is a wall-clock
        # measurement, so no threshold is asserted here (timing assertions
        # are flaky under load and prove nothing about correctness).
        assert isinstance(report["duration_ms"], (int, float))
        assert report["duration_ms"] >= 0.0

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

    def test_vectorized_exemplar_memory_numpy_dot_product_speed_and_accuracy(self, tmp_path: Path):
        """The REAL ExemplarMemory vectorized match (np.dot over the
        pre-normalized (N, 438) exemplar matrix) must yield exact numerical
        equivalence to a per-exemplar cosine loop and stay sub-millisecond
        for the dot product over 100 exemplars.

        100 distinct real crops are loaded through the production JSON path
        (ExemplarMemory.load + _rebuild_matrices), then a query crop identical
        to exemplar #42 must match itself with sim ~1.0 and class 0."""
        N = 100
        D = 438  # 432 HSV bins + aspect ratio + 5 texture bins

        # 100 distinct real crops (cycling hue/sat/val) -> real production features.
        crops = [
            _hsv_patch((i * 37) % 180, 90 + (i * 53) % 160, 110 + (i * 71) % 140, seed=1000 + i)
            for i in range(N)
        ]
        positives = []
        for crop in crops:
            feat = extract_crop_features(crop)
            assert feat is not None
            positives.append({
                "class_id": 0,
                "class_name": "herb_aria",
                "hsv_hist": feat["hsv_hist"],
                "aspect_ratio": feat["aspect_ratio"],
                "texture": feat["texture"],
                "mean_hsv": feat.get("mean_hsv", []),
                "source": "ui_approval",
            })
        ex_file = tmp_path / "learned_exemplars.json"
        ex_file.write_text(json.dumps({"version": 1, "positive": positives, "negative": []}), encoding="utf-8")
        memory = ExemplarMemory(json_path=ex_file)
        assert len(memory._positive) == N
        assert memory._pos_matrix is not None and memory._pos_matrix.shape == (N, D)

        # 1. Production vectorized match on a query identical to exemplar #42.
        query = crops[42]
        t0 = time.perf_counter()
        sim, cid, cname, best_ex = memory.match_positive(query)
        match_duration = time.perf_counter() - t0
        assert cid == 0 and cname == "herb_aria"
        assert best_ex is not None and int(best_ex["class_id"]) == 0
        assert sim >= 0.99, f"Identical query crop must match with sim ~1.0, got {sim}"

        # 2. The same dot product, computed directly over the production matrix,
        #    must equal the production result and the per-exemplar cosine loop.
        query_feat = extract_crop_features(query)
        assert query_feat is not None
        query_vec = feature_dict_to_normalized_vector(query_feat)
        t0 = time.perf_counter()
        vectorized_sims = np.dot(memory._pos_matrix, query_vec)
        vec_duration = time.perf_counter() - t0
        loop_sims = np.array(
            [float(np.dot(feature_dict_to_normalized_vector(ex), query_vec)) for ex in memory._positive],
            dtype=np.float32,
        )
        assert np.allclose(vectorized_sims, loop_sims, atol=1e-5), "Vectorized sims must match loop sims"
        assert np.allclose(float(vectorized_sims.max()), float(sim), atol=1e-3)
        assert int(np.argmax(vectorized_sims)) == 42

        # 3. Vectorized dot product stays sub-millisecond for 100 exemplars.
        assert vec_duration < 0.005, f"Vectorized dot product must take < 5ms, took {vec_duration * 1000:.3f}ms"
        assert match_duration < 1.0, f"Production match_positive must take < 1s, took {match_duration:.3f}s"

        # 4. No learned negatives -> match_negative returns the zero sentinel.
        neg_sim, neg_ex = memory.match_negative(query, class_id=0)
        assert neg_sim == 0.0 and neg_ex is None

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
