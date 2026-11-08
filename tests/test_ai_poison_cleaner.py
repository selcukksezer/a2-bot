# -*- coding: utf-8 -*-
"""
Tests for Work Package I (4-Layer AI Poison Cleaner) and Work Package J (Box-Level Bulk Pruning / Görseli Koru).
- Layer 0: Immunity shield for hand-labeled, human-approved, gather seeds, or positive exemplar matches.
- Layer 1: AI exemplar memory class repair and negative suppression.
- Layer 2: Visual flat-texture / HUD poison detection and negative teaching.
- Layer 3: Omni synthetic cull and NMS deduplication.
- Work Package J: 4-box scenario: 3 high confidence (%92, %95, %91), 1 low confidence (%35).
  Verifies that ONLY the 1 low box is pruned, image is preserved, 3 good boxes are kept as Emin (trusted=True),
  and disk .txt and .json are accurately updated.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import tempfile
import shutil
import pytest
import numpy as np
import cv2

from dataset_factory.label_trust import (
    UNVERIFIED_HEURISTIC_SOURCES,
    calibrate_box_confidence_and_trust,
    clean_poison_labels_ai,
    prune_low_confidence_boxes_in_dataset,
    requeue_labeled_dataset_for_review,
)
from dataset_factory.exemplar_memory import ExemplarMemory


@pytest.fixture
def temp_dataset_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="poison_test_"))
    for split in ("train", "val"):
        (temp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "review" / split).mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _create_synthetic_test_image(path: Path, width: int = 400, height: int = 400, flat: bool = False):
    if flat:
        img = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        # Rich textured image (Laplacian variance >> 10)
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_layer0_immunity_shield(temp_dataset_dir):
    """Layer 0: Hand-labeled, approved, or seed boxes MUST NEVER be deleted or demoted."""
    img_path = temp_dataset_dir / "images" / "train" / "img_immune.jpg"
    _create_synthetic_test_image(img_path, flat=True)  # flat image would normally fail layer 2

    rev_path = temp_dataset_dir / "review" / "train" / "img_immune.json"
    payload = {
        "stem": "img_immune",
        "split": "train",
        "trusted": [
            {
                "class_id": 0,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "confidence": 0.50,  # low confidence but hand_label
                "hand_label": True,
            },
            {
                "class_id": 1,
                "cx": 0.3,
                "cy": 0.3,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.60,
                "human_approved": True,
            },
        ],
        "weak": [],
    }
    rev_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = clean_poison_labels_ai(dataset_dir=temp_dataset_dir)
    assert stats["ok"] is True
    assert stats["protected_boxes"] >= 2
    assert stats["poison_deleted_boxes"] == 0

    saved = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 2
    assert all(b.get("trusted") is True for b in saved["trusted"])


def test_layer1_exemplar_memory_repair_and_suppression(temp_dataset_dir):
    """Layer 1: Auto-repair class if positive exemplar matches >= 0.82; suppress if negative matches."""
    img_path = temp_dataset_dir / "images" / "train" / "img_repair.jpg"
    _create_synthetic_test_image(img_path, flat=False)

    rev_path = temp_dataset_dir / "review" / "train" / "img_repair.json"
    payload = {
        "stem": "img_repair",
        "split": "train",
        "trusted": [],
        "weak": [
            {
                "class_id": 0,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "confidence": 0.88,
                "exemplar_repair_class_id": 2,  # Simulated positive match to class 2
                "exemplar_repair_sim": 0.90,
            },
            {
                "class_id": 1,
                "cx": 0.8,
                "cy": 0.8,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.70,
                "exemplar_neg_sim": 0.92,       # High negative similarity -> poison
                "exemplar_pos_sim": 0.60,       # Margin 0.32 >= 0.15
            },
        ],
    }
    rev_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = clean_poison_labels_ai(dataset_dir=temp_dataset_dir)
    assert stats["ok"] is True
    assert stats["auto_repaired_boxes"] >= 1
    assert stats["poison_deleted_boxes"] >= 1

    saved = json.loads(rev_path.read_text(encoding="utf-8"))
    # One box repaired and kept, the other deleted
    remaining = saved["trusted"] + saved["weak"]
    assert len(remaining) == 1
    assert remaining[0]["class_id"] == 2  # Repaired from 0 to 2


def test_layer2_flat_surface_and_hud_poison(temp_dataset_dir):
    """Layer 2: Boxes on flat texture (Laplacian var < 10) or illegal HUD zones pruned."""
    img_path = temp_dataset_dir / "images" / "train" / "img_flat.jpg"
    _create_synthetic_test_image(img_path, flat=True)  # Completely uniform/flat image

    rev_path = temp_dataset_dir / "review" / "train" / "img_flat.json"
    payload = {
        "stem": "img_flat",
        "split": "train",
        "trusted": [],
        "weak": [
            {
                "class_id": 3,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "confidence": 0.70,  # weak box on flat surface without immunity
            }
        ],
    }
    rev_path.write_text(json.dumps(payload), encoding="utf-8")

    stats = clean_poison_labels_ai(dataset_dir=temp_dataset_dir)
    assert stats["ok"] is True
    assert stats["poison_deleted_boxes"] >= 1

    saved = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 0
    assert len(saved["weak"]) == 0


def test_work_package_j_bulk_prune_preserves_image_and_good_boxes(temp_dataset_dir):
    """Work Package J: Image with 4 boxes (3 high: %92, %95, %91; 1 low: %35).
    Verifies that ONLY the 1 low-confidence box is pruned, the image is preserved,
    the 3 good boxes are kept as Emin (trusted=True), and both .json and .txt labels are updated.
    """
    img_path = temp_dataset_dir / "images" / "train" / "frame_wpj.jpg"
    _create_synthetic_test_image(img_path, flat=False)

    lbl_path = temp_dataset_dir / "labels" / "train" / "frame_wpj.txt"
    rev_path = temp_dataset_dir / "review" / "train" / "frame_wpj.json"

    # 4 boxes: 3 high conf (0.92, 0.95, 0.91), 1 low conf (0.35)
    boxes = [
        {"class_id": 0, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1, "confidence": 0.92},
        {"class_id": 1, "cx": 0.4, "cy": 0.4, "w": 0.1, "h": 0.1, "confidence": 0.95},
        {"class_id": 2, "cx": 0.6, "cy": 0.6, "w": 0.1, "h": 0.1, "confidence": 0.91},
        {"class_id": 3, "cx": 0.8, "cy": 0.8, "w": 0.1, "h": 0.1, "confidence": 0.35},
    ]

    # Write initial .txt
    txt_lines = [
        f"{b['class_id']} {b['cx']} {b['cy']} {b['w']} {b['h']} {b['confidence']}"
        for b in boxes
    ]
    lbl_path.write_text("\n".join(txt_lines) + "\n", encoding="utf-8")

    # Write initial review .json
    rev_payload = {
        "stem": "frame_wpj",
        "split": "train",
        "trusted": [b for b in boxes if b["confidence"] >= 0.85],
        "weak": [b for b in boxes if b["confidence"] < 0.85],
    }
    rev_path.write_text(json.dumps(rev_payload, indent=2), encoding="utf-8")

    # Execute box-level bulk pruning with threshold 0.85
    stats = prune_low_confidence_boxes_in_dataset(
        dataset_dir=temp_dataset_dir,
        min_confidence=0.85,
        teach_negative=True,
    )

    assert stats["ok"] is True
    assert stats["boxes_pruned"] == 1
    assert stats["images_checked"] >= 1

    # 1. Image MUST still exist on disk!
    assert img_path.is_file(), "Görsel silinmemeli, korunmalıdır!"

    # 2. Review .json MUST have exactly 3 trusted (Emin) boxes, 0 weak boxes
    assert rev_path.is_file()
    saved_rev = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(saved_rev["trusted"]) == 3, f"Expected 3 trusted boxes, got {len(saved_rev['trusted'])}"
    assert len(saved_rev["weak"]) == 0, f"Expected 0 weak boxes, got {len(saved_rev['weak'])}"

    saved_classes = [b["class_id"] for b in saved_rev["trusted"]]
    assert set(saved_classes) == {0, 1, 2}, "Classes 0, 1, and 2 must be preserved!"
    assert all(b.get("trusted") is True for b in saved_rev["trusted"])
    assert all(b.get("confidence") >= 0.85 for b in saved_rev["trusted"])

    # 3. Label .txt MUST be updated with exactly 3 high-confidence boxes
    assert lbl_path.is_file()
    txt_content = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(txt_content) == 3, f"Expected 3 lines in .txt, got {len(txt_content)}"


def test_unverified_heuristic_sources_constant():
    """Verify UNVERIFIED_HEURISTIC_SOURCES contains all 10 unverified sources."""
    expected = {
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
    assert expected.issubset(UNVERIFIED_HEURISTIC_SOURCES)


def test_calibrate_box_confidence_and_trust_human_and_seeds():
    """Human approvals and gather seeds MUST be trusted=True, weak=False, conf >= 0.90."""
    # Hand label
    b1 = calibrate_box_confidence_and_trust({"class_id": 0, "confidence": 0.40, "hand_label": True})
    assert b1["trusted"] is True
    assert b1["weak"] is False
    assert b1["confidence"] >= 0.90

    # Human approved
    b2 = calibrate_box_confidence_and_trust({"class_id": 1, "confidence": 0.60, "human_approved": True})
    assert b2["trusted"] is True
    assert b2["weak"] is False
    assert b2["confidence"] >= 0.90

    # Approved flag
    b3 = calibrate_box_confidence_and_trust({"class_id": 2, "confidence": 0.50, "approved": True})
    assert b3["trusted"] is True
    assert b3["weak"] is False
    assert b3["confidence"] >= 0.90

    # F gather prompt template seed
    b4 = calibrate_box_confidence_and_trust({
        "class_id": 7,
        "gather_key": "f",
        "source": "vision_f_gather_prompt",
        "confidence": 0.70,
    })
    assert b4["trusted"] is True
    assert b4["weak"] is False
    assert b4["confidence"] >= 0.90


def test_calibrate_box_confidence_and_trust_unverified_heuristics():
    """Unverified heuristic sources must be capped <= 0.68 and trusted=False, weak=True."""
    for src in UNVERIFIED_HEURISTIC_SOURCES:
        box = {
            "class_id": 0,
            "confidence": 0.95,  # Artificially inflated score
            "source": src,
        }
        cal = calibrate_box_confidence_and_trust(box)
        assert cal["trusted"] is False, f"Source {src} must NEVER be trusted!"
        assert cal["weak"] is True, f"Source {src} must be marked weak!"
        assert cal["confidence"] <= 0.68, f"Source {src} confidence must be capped at 0.68, got {cal['confidence']}!"


def test_calibrate_box_confidence_and_trust_material_conflicts():
    """Material conflicts between stem and box class, or essence on 3D frame, must be capped <= 0.45."""
    # 1. ore_orichalcum on an aria frame
    box_conflict1 = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "confidence": 0.92,
        "source": "auto_scan",
    }
    cal1 = calibrate_box_confidence_and_trust(box_conflict1, stem="boot_478467_000_aria_3d_crest_far")
    assert cal1["trusted"] is False
    assert cal1["weak"] is True
    assert cal1["confidence"] <= 0.45

    # 2. essence_yggdrasil_log on 3D frame
    box_conflict2 = {
        "class_id": 98,
        "class_name": "essence_yggdrasil_log",
        "confidence": 0.98,
        "source": "exemplar_match",
    }
    cal2 = calibrate_box_confidence_and_trust(box_conflict2, stem="boot_478467_000_aria_3d_crest_far")
    assert cal2["trusted"] is False
    assert cal2["weak"] is True
    assert cal2["confidence"] <= 0.45

    # 3. Matching material: herb_aria on aria frame -> NOT conflicting
    box_match = {
        "class_id": 0,
        "class_name": "herb_aria",
        "confidence": 0.92,
        "source": "vision_3d_crest",
    }
    cal3 = calibrate_box_confidence_and_trust(box_match, stem="boot_478467_000_aria_3d_crest_far")
    assert cal3["trusted"] is True
    assert cal3["weak"] is False
    assert cal3["confidence"] == 0.92


def test_requeue_labeled_dataset_default_conf_and_unverified_weak(temp_dataset_dir):
    """Requeue with default_conf=0.58 correctly sends unverified .txt boxes to weak (Şüpheli)."""
    img_path = temp_dataset_dir / "images" / "train" / "boot_001_aria_3d_crest_far.jpg"
    _create_synthetic_test_image(img_path, flat=False)

    lbl_path = temp_dataset_dir / "labels" / "train" / "boot_001_aria_3d_crest_far.txt"
    lines = [
        "0 0.5 0.5 0.1 0.1 0.92",  # Verified explicit >= 0.85 -> trusted (Emin)
        "1 0.3 0.3 0.1 0.1 0.60",  # Explicit < 0.85 -> weak (Şüpheli)
        "0 0.7 0.7 0.1 0.1",       # No explicit conf -> default 0.58 -> weak (Şüpheli)
        "3 0.8 0.8 0.1 0.1 0.90",  # orichalcum on aria stem -> material conflict -> weak (Şüpheli)
    ]
    lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    stats = requeue_labeled_dataset_for_review(
        dataset_dir=temp_dataset_dir,
        emin_threshold=0.85,
    )
    assert stats["ok"] is True
    assert stats["default_conf"] == 0.58

    rev_file = temp_dataset_dir / "review" / "train" / "boot_001_aria_3d_crest_far.json"
    assert rev_file.is_file()
    payload = json.loads(rev_file.read_text(encoding="utf-8"))

    trusted = payload["trusted"]
    weak = payload["weak"]

    # Only line 0 (class 0, conf 0.92, matching material) is trusted!
    assert len(trusted) == 1
    assert trusted[0]["class_id"] == 0
    assert trusted[0]["confidence"] == 0.92
    assert trusted[0]["trusted"] is True

    # Lines 1 (conf 0.60), 2 (unverified default 0.58), and 3 (conflict <= 0.45) are weak!
    assert len(weak) == 3
    assert all(b["trusted"] is False and b["weak"] is True for b in weak)

