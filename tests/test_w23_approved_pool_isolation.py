# -*- coding: utf-8 -*-
"""Unit tests for ApprovedDatasetPool isolation and manifest updates (W23)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.approved_pool import ApprovedDatasetPool


def test_approved_pool_directory_isolation_structure(tmp_path: Path) -> None:
    """Verify ApprovedDatasetPool initializes isolated dataset/approved hierarchy and manifest."""
    pool = ApprovedDatasetPool(project_root=tmp_path)

    expected_pool_root = tmp_path / "dataset" / "approved"
    assert pool.pool_root == expected_pool_root
    assert (expected_pool_root / "images" / "train").is_dir()
    assert (expected_pool_root / "images" / "val").is_dir()
    assert (expected_pool_root / "labels" / "train").is_dir()
    assert (expected_pool_root / "labels" / "val").is_dir()
    assert pool.manifest_path == expected_pool_root / "approved_manifest.json"
    assert pool.manifest_path.is_file()

    manifest = json.loads(pool.manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_approved_images"] == 0
    assert manifest["train_count"] == 0
    assert manifest["val_count"] == 0
    assert manifest["total_boxes"] == 0
    assert manifest["stems"] == {}


def test_add_approved_sample_copies_image_and_writes_yolo_txt_isolated(tmp_path: Path) -> None:
    """Verify add_approved_sample copies image and writes .txt labels to dataset/approved/ in isolation."""
    src_images_dir = tmp_path / "dataset" / "images" / "train"
    src_labels_dir = tmp_path / "dataset" / "labels" / "train"
    src_images_dir.mkdir(parents=True, exist_ok=True)
    src_labels_dir.mkdir(parents=True, exist_ok=True)

    raw_img_bytes = b"\xff\xd8\xff\xe0JFIF_TEST_IMAGE_DATA"
    src_img = src_images_dir / "sample_001.jpg"
    src_img.write_bytes(raw_img_bytes)

    src_lbl = src_labels_dir / "sample_001.txt"
    src_lbl.write_text("0 0.100000 0.100000 0.100000 0.100000\n", encoding="utf-8")

    pool = ApprovedDatasetPool(project_root=tmp_path)
    ok = pool.add_approved_sample(
        stem="sample_001",
        split="train",
        image_path=src_img,
        trusted_boxes=[
            {"class_id": 3, "cx": 0.5, "cy": 0.4, "w": 0.2, "h": 0.3},
            (10, 0.25, 0.75, 0.15, 0.25),
        ],
    )
    assert ok is True
    assert pool.is_approved("sample_001") is True

    isolated_img = tmp_path / "dataset" / "approved" / "images" / "train" / "sample_001.jpg"
    isolated_lbl = tmp_path / "dataset" / "approved" / "labels" / "train" / "sample_001.txt"

    assert isolated_img.is_file()
    assert isolated_img.read_bytes() == raw_img_bytes
    assert isolated_lbl.is_file()

    lbl_lines = isolated_lbl.read_text(encoding="utf-8").strip().splitlines()
    assert lbl_lines == [
        "3 0.500000 0.400000 0.200000 0.300000",
        "10 0.250000 0.750000 0.150000 0.250000",
    ]
    # Source label outside dataset/approved/ must remain untouched (isolated)
    assert src_lbl.read_text(encoding="utf-8") == "0 0.100000 0.100000 0.100000 0.100000\n"


def test_manifest_json_updated_with_sample_metadata_and_stats(tmp_path: Path) -> None:
    """Verify approved_manifest.json is accurately updated with sample metadata, class counts, and box totals."""
    pool = ApprovedDatasetPool(project_root=tmp_path)

    ok = pool.add_approved_sample(
        stem="frame_alpha",
        split="train",
        image_bytes=b"FAKE_JPEG_ALPHA",
        trusted_boxes=[
            "0 0.5 0.5 0.2 0.2",
            "5 0.3 0.6 0.1 0.1",
        ],
        approved_at=1700000000.0,
    )
    assert ok is True

    manifest = json.loads(pool.manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_approved_images"] == 1
    assert manifest["train_count"] == 1
    assert manifest["val_count"] == 0
    assert manifest["total_boxes"] == 2
    assert manifest["classes"] == {"0": 1, "5": 1}

    stem_entry = manifest["stems"]["frame_alpha"]
    assert stem_entry["stem"] == "frame_alpha"
    assert stem_entry["split"] == "train"
    assert stem_entry["image_file"] == "frame_alpha.jpg"
    assert stem_entry["label_file"] == "frame_alpha.txt"
    assert stem_entry["box_count"] == 2
    assert stem_entry["classes"] == [0, 5]
    assert stem_entry["approved_at"] == 1700000000.0


def test_bulk_add_and_val_split_isolation_with_manifest_persistence(tmp_path: Path) -> None:
    """Verify bulk_add_approved_samples isolates train/val files and persists manifest across instances."""
    src_dir = tmp_path / "raw_source"
    src_dir.mkdir(parents=True, exist_ok=True)
    img_train = src_dir / "f_train.png"
    img_val = src_dir / "f_val.jpg"
    img_train.write_bytes(b"PNG_TRAIN_BYTES")
    img_val.write_bytes(b"JPG_VAL_BYTES")

    pool = ApprovedDatasetPool(project_root=tmp_path)
    added = pool.bulk_add_approved_samples(
        [
            {
                "stem": "f_train",
                "split": "train",
                "image_path": img_train,
                "trusted_boxes": [(1, 0.4, 0.4, 0.2, 0.2)],
            },
            {
                "stem": "f_val",
                "split": "val",
                "image_path": img_val,
                "trusted_boxes": [(1, 0.6, 0.6, 0.1, 0.1), (2, 0.2, 0.2, 0.1, 0.1)],
            },
        ]
    )
    assert added == 2

    assert (pool.pool_root / "images" / "train" / "f_train.png").is_file()
    assert (pool.pool_root / "labels" / "train" / "f_train.txt").is_file()
    assert (pool.pool_root / "images" / "val" / "f_val.jpg").is_file()
    assert (pool.pool_root / "labels" / "val" / "f_val.txt").is_file()

    # Re-instantiate pool to confirm manifest reload and quarantine persistence
    reloaded_pool = ApprovedDatasetPool(project_root=tmp_path)
    assert reloaded_pool.is_approved("f_train") is True
    assert reloaded_pool.is_approved("f_val") is True
    stats = reloaded_pool.get_approved_stats()
    assert stats["total_approved_images"] == 2
    assert stats["train_count"] == 1
    assert stats["val_count"] == 1
    assert stats["total_boxes"] == 3


def test_remove_and_prepare_approved_training_dataset_updates_manifest(tmp_path: Path) -> None:
    """Verify prepare_approved_training_dataset filters only approved items and remove_approved_sample cleans up."""
    dataset_dir = tmp_path / "dataset"
    img_dir = dataset_dir / "images" / "train"
    review_dir = dataset_dir / "review"
    img_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    approved_img = img_dir / "approved_01.jpg"
    unapproved_img = img_dir / "pending_02.jpg"
    approved_img.write_bytes(b"APPROVED_IMG")
    unapproved_img.write_bytes(b"PENDING_IMG")

    (review_dir / "approved_01.json").write_text(
        json.dumps(
            {
                "stem": "approved_01",
                "split": "train",
                "human_approved": True,
                "done": "ok",
                "image_path": str(approved_img),
                "trusted": [{"class_id": 7, "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3}],
            }
        ),
        encoding="utf-8",
    )
    (review_dir / "pending_02.json").write_text(
        json.dumps(
            {
                "stem": "pending_02",
                "split": "train",
                "human_approved": False,
                "done": "pending",
                "image_path": str(unapproved_img),
                "trusted": [{"class_id": 7, "cx": 0.5, "cy": 0.5, "w": 0.3, "h": 0.3}],
            }
        ),
        encoding="utf-8",
    )

    pool = ApprovedDatasetPool(project_root=tmp_path)
    summary = pool.prepare_approved_training_dataset(source_dataset_dir=dataset_dir)

    assert summary["total_approved_images"] == 1
    assert pool.is_approved("approved_01") is True
    assert pool.is_approved("pending_02") is False
    assert (pool.pool_root / "images" / "train" / "approved_01.jpg").is_file()
    assert (pool.pool_root / "labels" / "train" / "approved_01.txt").is_file()
    assert not (pool.pool_root / "images" / "train" / "pending_02.jpg").exists()
    assert pool.data_yaml_path.is_file()

    # Remove the approved sample and verify disk + manifest cleanup
    removed = pool.remove_approved_sample("approved_01")
    assert removed is True
    assert pool.is_approved("approved_01") is False
    assert not (pool.pool_root / "images" / "train" / "approved_01.jpg").exists()
    assert not (pool.pool_root / "labels" / "train" / "approved_01.txt").exists()

    manifest = json.loads(pool.manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_approved_images"] == 0
    assert "approved_01" not in manifest["stems"]
