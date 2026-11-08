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

import json
from pathlib import Path
import tempfile
import shutil
import pytest
import numpy as np
import cv2

from dataset_factory.label_trust import (
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
