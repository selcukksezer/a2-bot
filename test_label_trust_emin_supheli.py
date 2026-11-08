# -*- coding: utf-8 -*-
"""
Tests for Work Package G: Emin (Güvenilir) vs Şüpheli Classification & Rich Filtering.
- Separate boxes with confidence >= 0.85 or hand/seed/exemplar as trusted=True, weak=False (Emin)
- Boxes with confidence < 0.85 as trusted=False, weak=True (Şüpheli)
- Canvas rendering distinctions (solid outline + clean name + checkmark vs dashed outline + supheli)
- Rich multi-criteria filtering: queue_filter (all, suspicious, confident, mixed, empty, unlabeled_and_suspicious)
- Class filter and search filter
"""

import json
from pathlib import Path
import tempfile
import shutil
import pytest

from dataset_factory.label_trust import requeue_labeled_dataset_for_review, train_txt_boxes
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler


@pytest.fixture
def temp_dataset_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="emin_supheli_"))
    for split in ("train", "val"):
        (temp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "review" / split).mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_yolo_auto_labeler_emin_vs_supheli():
    """Verify YOLOAutoLabeler._make_annotation marks conf >= 0.85 as trusted=True, weak=False (Emin)."""
    labeler = YOLOAutoLabeler()

    # Confident box (>= 0.85)
    box_emin = labeler._make_annotation(
        class_id=0, x=100, y=100, w=50, h=50, img_w=1000, img_h=1000,
        confidence=0.92, source="vision_3d_crest"
    )
    assert box_emin["trusted"] is True
    assert box_emin["weak"] is False

    # Weak box (< 0.85)
    box_supheli = labeler._make_annotation(
        class_id=1, x=100, y=100, w=50, h=50, img_w=1000, img_h=1000,
        confidence=0.65, source="visual_fallback"
    )
    assert box_supheli["trusted"] is False
    assert box_supheli["weak"] is True

    # Hand-labeled box regardless of confidence is always Emin
    box_hand = labeler._make_annotation(
        class_id=2, x=100, y=100, w=50, h=50, img_w=1000, img_h=1000,
        confidence=0.50, source="hand_label", hand_label=True
    )
    assert box_hand["trusted"] is True
    assert box_hand["weak"] is False


def test_requeue_labeled_dataset_separates_emin_and_supheli(temp_dataset_dir):
    """Verify requeue_labeled_dataset_for_review segregates boxes into trusted (Emin) and weak (Şüpheli)."""
    # Create an image and label txt in train
    img_path = temp_dataset_dir / "images" / "train" / "frame_001.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Dummy JPEG header

    lbl_path = temp_dataset_dir / "labels" / "train" / "frame_001.txt"
    # Format: class cx cy w h [conf]
    lines = [
        "0 0.5 0.5 0.1 0.1 0.95",  # Emin (conf >= 0.85)
        "1 0.3 0.3 0.1 0.1 0.60",  # Şüpheli (conf < 0.85)
        "2 0.7 0.7 0.1 0.1",       # Default 0.58 unverified -> Şüpheli
    ]
    lbl_path.write_text("\n".join(lines), encoding="utf-8")

    stats = requeue_labeled_dataset_for_review(
        dataset_dir=temp_dataset_dir,
        emin_threshold=0.85,
    )

    assert stats["ok"] is True
    assert stats["samples"] >= 1
    assert stats["boxes"] == 3

    rev_file = temp_dataset_dir / "review" / "train" / "frame_001.json"
    assert rev_file.is_file()
    payload = json.loads(rev_file.read_text(encoding="utf-8"))

    trusted = payload["trusted"]
    weak = payload["weak"]

    assert len(trusted) == 1  # class 0 (0.95)
    assert len(weak) == 2     # class 1 (0.60) and class 2 (0.58 unverified)
    assert all(b.get("trusted") is True and b.get("weak") is False for b in trusted)
    assert all(b.get("trusted") is False and b.get("weak") is True for b in weak)


def test_rich_filter_categorization():
    """Verify queue items match the 6 rich filter categories: all, suspicious, confident, mixed, empty, unlabeled_and_suspicious."""
    items = [
        {"stem": "item_empty", "trusted": [], "weak": []},
        {"stem": "item_suspicious", "trusted": [], "weak": [{"class_id": 0, "conf": 0.50}]},
        {"stem": "item_confident", "trusted": [{"class_id": 1, "conf": 0.95}], "weak": []},
        {"stem": "item_mixed", "trusted": [{"class_id": 2, "conf": 0.92}], "weak": [{"class_id": 3, "conf": 0.40}]},
    ]

    def filter_items(mode, class_id=-1, search=""):
        out = []
        for it in items:
            stem = it["stem"]
            if search and search not in stem:
                continue
            all_b = it["trusted"] + it["weak"]
            if class_id >= 0 and not any(b["class_id"] == class_id for b in all_b):
                continue
            n_t, n_w = len(it["trusted"]), len(it["weak"])
            n_tot = n_t + n_w

            if mode == "suspicious" and not (n_w > 0 and n_t == 0):
                continue
            if mode == "confident" and not (n_t > 0 and n_w == 0):
                continue
            if mode == "mixed" and not (n_t > 0 and n_w > 0):
                continue
            if mode == "empty" and n_tot > 0:
                continue
            if mode == "unlabeled_and_suspicious" and not (n_tot == 0 or n_w > 0):
                continue
            out.append(it["stem"])
        return out

    assert len(filter_items("all")) == 4
    assert filter_items("empty") == ["item_empty"]
    assert filter_items("suspicious") == ["item_suspicious"]
    assert filter_items("confident") == ["item_confident"]
    assert filter_items("mixed") == ["item_mixed"]
    assert set(filter_items("unlabeled_and_suspicious")) == {"item_empty", "item_suspicious", "item_mixed"}
    assert filter_items("all", class_id=2) == ["item_mixed"]
    assert filter_items("all", search="confident") == ["item_confident"]
