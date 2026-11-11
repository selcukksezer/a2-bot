# -*- coding: utf-8 -*-
"""tests/test_approved_dataset_pool.py
=======================================
Unit and integration tests for ApprovedDatasetPool (Division 1: A04, A07).
Validates quarantine persistence, atomic image/label persistence, YOLO YAML generation,
and 85/15 train/val split balancing.
"""

import json
import shutil
import tempfile
from pathlib import Path
import pytest
import yaml

import cv2
import numpy as np

from dataset_factory.approved_pool import (
    ApprovedDatasetPool,
    normalize_box,
)
from dataset_factory.class_taxonomy import CANONICAL_NAMES


def _valid_jpeg_bytes(width: int = 8, height: int = 8) -> bytes:
    """Encode a tiny REAL JPEG (decodable bytes, SOI..EOI) for pool tests.

    Approved samples are stored as ``.jpg`` files, so fixtures must carry
    genuine JPEG bytes — arbitrary byte strings (``b"fake_jpeg_content"``)
    must never stand in for an image here.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok, "cv2 failed to encode test JPEG fixture"
    data = bytes(buf)
    assert data[:2] == b"\xff\xd8" and data[-2:] == b"\xff\xd9", "fixture is not a JPEG"
    return data


@pytest.fixture
def temp_pool_dir():
    """Create a temporary pool root directory."""
    temp_dir = tempfile.mkdtemp(prefix="test_approved_pool_")
    yield Path(temp_dir)
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass


def test_pool_initialization_and_structure(temp_pool_dir):
    """Test that ApprovedDatasetPool creates all required directories and manifest."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)

    assert pool.images_train_dir.is_dir()
    assert pool.images_val_dir.is_dir()
    assert pool.labels_train_dir.is_dir()
    assert pool.labels_val_dir.is_dir()
    assert pool.manifest_path.is_file()

    stats = pool.get_approved_stats()
    assert stats["total_approved_images"] == 0
    assert stats["train_count"] == 0
    assert stats["val_count"] == 0
    assert stats["total_boxes"] == 0


def test_normalize_box_variants():
    """Test box normalization across dicts, sequences, strings, and invalid items."""
    # Dict with class_id, cx, cy, w, h
    b1 = {"class_id": 9, "cx": 0.5, "cy": 0.4, "w": 0.1, "h": 0.2}
    n1 = normalize_box(b1)
    assert n1 == (9, 0.5, 0.4, 0.1, 0.2)

    # Sequence [cid, cx, cy, w, h]
    b2 = [0, 0.25, 0.35, 0.05, 0.08]
    n2 = normalize_box(b2)
    assert n2 == (0, 0.25, 0.35, 0.05, 0.08)

    # String format
    b3 = "3 0.123456 0.654321 0.100000 0.200000"
    n3 = normalize_box(b3)
    assert n3 == (3, 0.123456, 0.654321, 0.1, 0.2)

    # Invalid box with zero dimension
    b4 = {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1}
    assert normalize_box(b4) is None

    # Invalid format
    assert normalize_box("invalid line") is None
    assert normalize_box(None) is None


def test_add_approved_sample_with_bytes(temp_pool_dir):
    """Test adding an approved sample directly using in-memory image bytes."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)

    dummy_bytes = _valid_jpeg_bytes()
    boxes = [
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        {"class_id": 9, "cx": 0.9, "cy": 0.15, "w": 0.05, "h": 0.05},
    ]

    ok = pool.add_approved_sample(
        stem="frame_test_001",
        split="train",
        trusted_boxes=boxes,
        image_bytes=dummy_bytes,
    )
    assert ok is True

    # Check quarantine
    assert pool.is_approved("frame_test_001") is True
    assert pool.is_approved("frame_test_001.jpg") is True
    assert pool.is_approved("non_existent_frame") is False
    assert "frame_test_001" in pool.get_approved_stems()

    # Check disk files
    img_path = pool.images_train_dir / "frame_test_001.jpg"
    lbl_path = pool.labels_train_dir / "frame_test_001.txt"
    assert img_path.is_file()
    assert lbl_path.is_file()

    # Verify YOLO label content
    lbl_lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lbl_lines) == 2
    assert lbl_lines[0].startswith("0 0.500000 0.500000 0.100000 0.100000")
    assert lbl_lines[1].startswith("9 0.900000 0.150000 0.050000 0.050000")

    # Check stats
    stats = pool.get_approved_stats()
    assert stats["total_approved_images"] == 1
    assert stats["train_count"] == 1
    assert stats["val_count"] == 0
    assert stats["total_boxes"] == 2


def test_quarantine_persists_across_pool_reloads(temp_pool_dir):
    """Test that restarting/re-opening the pool preserves approved status (open/close dialog protection)."""
    pool1 = ApprovedDatasetPool(pool_root=temp_pool_dir)
    dummy_bytes = _valid_jpeg_bytes()

    pool1.add_approved_sample(
        stem="quarantined_stem_42",
        split="val",
        trusted_boxes=[{"class_id": 3, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}],
        image_bytes=dummy_bytes,
    )
    assert pool1.is_approved("quarantined_stem_42") is True

    # Simulate closing dialog and reopening: new pool instance reading from same root
    pool2 = ApprovedDatasetPool(pool_root=temp_pool_dir)
    assert pool2.is_approved("quarantined_stem_42") is True
    assert "quarantined_stem_42" in pool2.get_approved_stems()

    stats = pool2.get_approved_stats()
    assert stats["total_approved_images"] == 1
    assert stats["val_count"] == 1
    assert stats["train_count"] == 0
    assert stats["total_boxes"] == 1


def test_generate_approved_data_yaml(temp_pool_dir):
    """Test generation of valid YOLO data.yaml pointing to approved train/val images with 27 canonical classes."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)

    yaml_str = pool.generate_approved_data_yaml()
    assert Path(yaml_str).is_file()

    content = Path(yaml_str).read_text(encoding="utf-8")
    # Parse the YAML for real instead of substring-matching its text:
    # a malformed file could contain "nc: 27" as a comment and still pass.
    data = yaml.safe_load(content)
    assert isinstance(data, dict)
    # B-09 DÜZELTME (2026-11-10, canlı kanıt): Ultralytics path'i
    # yaml'ın dizinine göre DEĞİL, CWD'ye göre çözer; bu yüzden `path`
    # yaml'ın klasörünün MUTLAK yoludur (forward-slash).
    assert Path(data["path"]).is_absolute()
    assert "\\" not in data["path"]
    assert data["path"] == temp_pool_dir.resolve().as_posix()
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["nc"] == len(CANONICAL_NAMES) == 29
    assert data["names"] == CANONICAL_NAMES


def test_bulk_add_and_remove_approved_sample(temp_pool_dir):
    """Test bulk adding multiple samples and removing one."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)

    dummy_bytes = _valid_jpeg_bytes()
    samples = [
        {
            "stem": f"sample_{i}",
            "split": "train" if i % 2 == 0 else "val",
            "image_bytes": dummy_bytes,
            "trusted_boxes": [{"class_id": i % 5, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        }
        for i in range(10)
    ]

    added = pool.bulk_add_approved_samples(samples)
    assert added == 10

    stats = pool.get_approved_stats()
    assert stats["total_approved_images"] == 10
    assert stats["train_count"] == 5
    assert stats["val_count"] == 5
    assert stats["total_boxes"] == 10

    # Remove one sample
    removed = pool.remove_approved_sample("sample_0")
    assert removed is True
    assert pool.is_approved("sample_0") is False

    stats2 = pool.get_approved_stats()
    assert stats2["total_approved_images"] == 9
    assert stats2["train_count"] == 4
    assert stats2["total_boxes"] == 9


def test_prepare_approved_training_dataset_pipeline(temp_pool_dir):
    """Test A07 prepare_approved_training_dataset scanning review files and balancing 85/15 split."""
    # Create mock source dataset structure
    src_dir = temp_pool_dir / "mock_src_dataset"
    review_train = src_dir / "review" / "train"
    review_val = src_dir / "review" / "val"
    review_train.mkdir(parents=True, exist_ok=True)
    review_val.mkdir(parents=True, exist_ok=True)

    img_train = src_dir / "images" / "train"
    img_train.mkdir(parents=True, exist_ok=True)

    # Create 20 mock items, 15 human approved, 5 unapproved
    for i in range(20):
        stem = f"review_stem_{i:03d}"
        # create source image (a real JPEG, not a fake byte string)
        (img_train / f"{stem}.jpg").write_bytes(_valid_jpeg_bytes())

        is_human = (i < 15)  # first 15 approved
        review_data = {
            "stem": stem,
            "image_path": str(img_train / f"{stem}.jpg"),
            "split": "train",
            "human_approved": is_human,
            "done": "ok" if is_human else "pending",
            # P1-10 / K-21: admitted frames need per-box provenance.
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "human_approved": True}] if is_human else [],
        }
        (review_train / f"{stem}.json").write_text(json.dumps(review_data), encoding="utf-8")

    # Run pool on target directory
    target_pool_dir = temp_pool_dir / "approved_pool"
    pool = ApprovedDatasetPool(pool_root=target_pool_dir, project_root=temp_pool_dir)

    result_stats = pool.prepare_approved_training_dataset(
        source_dataset_dir=src_dir,
        val_ratio=0.15,
        force_rebalance=True,
    )

    # Verify only the 15 approved items were processed
    assert result_stats["total_approved_images"] == 15
    assert pool.is_approved("review_stem_000") is True
    assert pool.is_approved("review_stem_014") is True
    assert pool.is_approved("review_stem_015") is False  # unapproved was not taken

    # Verify the 85/15 balancing claim, not just the total: with
    # val_ratio=0.15 the rebalance is stem-hash based (P1-09 / K-24:
    # `_resolve_split` — MD5(stem) % 1000 < 150), so on 15 candidates
    # val stems are [003, 004] -> 13 train / 2 val. A frame's split
    # depends only on its own stem and the ratio — adding/removing one
    # frame never moves another frame's split (old Bresenham idx-based
    # allocation did: val idx were [6, 13]).
    assert result_stats["train_count"] == 13
    assert result_stats["val_count"] == 2
    assert result_stats["train_count"] + result_stats["val_count"] == 15
    assert sorted(p.name for p in pool.images_train_dir.glob("*.jpg")) == sorted(
        f"review_stem_{i:03d}.jpg" for i in range(15) if i not in (3, 4)
    )
    assert sorted(p.name for p in pool.images_val_dir.glob("*.jpg")) == [
        "review_stem_003.jpg",
        "review_stem_004.jpg",
    ]

    # P1-09 / K-24: the manifest's real split is persisted back into the
    # review JSONs — the two val frames' JSONs now say "val", the train
    # frames' JSONs were already "train" (untouched).
    assert result_stats["split_persisted_count"] == 2
    assert result_stats["pruned_count"] == 0
    for i in (3, 4):
        payload = json.loads((review_train / f"review_stem_{i:03d}.json").read_text(encoding="utf-8"))
        assert payload["split"] == "val"
    payload0 = json.loads((review_train / "review_stem_000.json").read_text(encoding="utf-8"))
    assert payload0["split"] == "train"

    # Check that data_approved.yaml was created
    assert Path(result_stats["data_yaml_path"]).is_file()
