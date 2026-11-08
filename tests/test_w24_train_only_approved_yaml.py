# -*- coding: utf-8 -*-
"""tests/test_w24_train_only_approved_yaml.py
=============================================
Worker W24 Unit Tests: Verify that `prepare_approved_training_dataset`
exclusively includes approved frames in training, excludes unapproved or
suspicious frames/boxes, and generates a valid `data_approved.yaml`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dataset_factory.approved_pool as approved_pool_mod
from dataset_factory.approved_pool import (
    ApprovedDatasetPool,
    CORE_22_CANONICAL_NAMES,
    prepare_approved_training_dataset,
)


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xD9")


def test_only_approved_frames_included_in_training_pool(tmp_path: Path) -> None:
    """1. Verify that only frames marked as approved (human_approved=True, done='ok', or approved=True) are included."""
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    records = [
        # Approved via human_approved=True
        {
            "stem": "frame_approved_human",
            "split": "train",
            "human_approved": True,
            "done": "pending",
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        },
        # Approved via done="ok"
        {
            "stem": "frame_approved_done_ok",
            "split": "train",
            "human_approved": False,
            "done": "ok",
            "trusted": [{"class_id": 3, "cx": 0.4, "cy": 0.6, "w": 0.1, "h": 0.15}],
        },
        # Approved via approved=True
        {
            "stem": "frame_approved_flag",
            "split": "val",
            "approved": True,
            "trusted": [{"class_id": 10, "cx": 0.3, "cy": 0.3, "w": 0.25, "h": 0.25}],
        },
        # Unapproved pending frame
        {
            "stem": "frame_unapproved_pending",
            "split": "train",
            "human_approved": False,
            "done": "pending",
            "trusted": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        },
    ]

    for rec in records:
        stem = rec["stem"]
        img_file = img_train / f"{stem}.jpg"
        _write_dummy_image(img_file)
        rec["image_path"] = str(img_file)
        (review_train / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")

    pool_root = tmp_path / "approved_pool"
    pool = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)

    stats = pool.prepare_approved_training_dataset(
        source_dataset_dir=src_dataset,
        force_rebalance=False,
    )

    assert stats["total_approved_images"] == 3
    assert stats["train_count"] == 2
    assert stats["val_count"] == 1
    assert stats["total_boxes"] == 3

    assert pool.is_approved("frame_approved_human") is True
    assert pool.is_approved("frame_approved_done_ok") is True
    assert pool.is_approved("frame_approved_flag") is True
    assert pool.is_approved("frame_unapproved_pending") is False


def test_unapproved_and_suspicious_frames_strictly_excluded(tmp_path: Path) -> None:
    """2. Verify that suspicious/unapproved frames, weak boxes, and metadata JSON files are never included in training."""
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    # Suspicious / weak-only / rejected frames
    unapproved_records = [
        {
            "stem": "frame_suspicious_weak_only",
            "split": "train",
            "human_approved": False,
            "done": "weak",
            "trusted": [],
            "weak": [{"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "conf": 0.42}],
        },
        {
            "stem": "frame_rejected",
            "split": "train",
            "human_approved": False,
            "approved": False,
            "done": "rejected",
            "trusted": [{"class_id": 5, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1}],
        },
        {
            "stem": "frame_suspect_auto",
            "split": "val",
            "human_approved": False,
            "done": "suspect",
            "weak": [{"class_id": 9, "cx": 0.8, "cy": 0.2, "w": 0.05, "h": 0.05}],
        },
    ]

    for rec in unapproved_records:
        stem = rec["stem"]
        img_file = img_train / f"{stem}.jpg"
        _write_dummy_image(img_file)
        rec["image_path"] = str(img_file)
        (review_train / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")

    # One approved frame that also contains suspicious 'weak' boxes alongside 'trusted' boxes
    approved_stem = "frame_clean_approved"
    approved_img = img_train / f"{approved_stem}.jpg"
    _write_dummy_image(approved_img)
    approved_rec = {
        "stem": approved_stem,
        "split": "train",
        "image_path": str(approved_img),
        "human_approved": True,
        "done": "ok",
        "trusted": [{"class_id": 4, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        "weak": [{"class_id": 14, "cx": 0.1, "cy": 0.1, "w": 0.3, "h": 0.3}],
    }
    (review_train / f"{approved_stem}.json").write_text(json.dumps(approved_rec), encoding="utf-8")

    # Reserved system JSONs that must be ignored even if they have human_approved=True
    (review_train / "trusted_seed.json").write_text(
        json.dumps({"human_approved": True, "done": "ok", "stem": "trusted_seed"}),
        encoding="utf-8",
    )
    (review_train / "manifest_stats.json").write_text(
        json.dumps({"human_approved": True, "done": "ok", "stem": "manifest_stats"}),
        encoding="utf-8",
    )

    pool_root = tmp_path / "approved_pool"
    pool = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dataset)

    assert stats["total_approved_images"] == 1
    assert pool.get_approved_stems() == {approved_stem}
    assert not pool.is_approved("frame_suspicious_weak_only")
    assert not pool.is_approved("frame_rejected")
    assert not pool.is_approved("frame_suspect_auto")
    assert not pool.is_approved("trusted_seed")
    assert not pool.is_approved("manifest_stats")

    # Verify only the 1 trusted box (class 4) was written to the label file, excluding the weak box (class 14)
    lbl_file = pool.labels_train_dir / f"{approved_stem}.txt"
    assert lbl_file.is_file()
    lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("4 0.500000 0.500000 0.100000 0.100000")


def test_generates_valid_data_approved_yaml_schema_and_paths(tmp_path: Path) -> None:
    """3. Verify that `prepare_approved_training_dataset` produces a valid `data_approved.yaml` file."""
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    img_file = img_train / "sample_01.jpg"
    _write_dummy_image(img_file)
    (review_train / "sample_01.json").write_text(
        json.dumps({
            "stem": "sample_01",
            "split": "train",
            "image_path": str(img_file),
            "human_approved": True,
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        }),
        encoding="utf-8",
    )

    pool_root = tmp_path / "approved_pool"
    pool = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)
    stats = pool.prepare_approved_training_dataset(source_dataset_dir=src_dataset)

    yaml_path = Path(stats["data_yaml_path"])
    assert yaml_path.is_file()
    assert yaml_path.name == "data_approved.yaml"

    parsed = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, dict)
    assert parsed["path"] == pool_root.as_posix()
    assert "\\" not in parsed["path"]
    assert parsed["train"] == "images/train"
    assert parsed["val"] == "images/val"
    assert parsed["nc"] == len(CORE_22_CANONICAL_NAMES) == 22

    names = parsed["names"]
    assert isinstance(names, dict)
    assert len(names) == 22
    for cid, cname in CORE_22_CANONICAL_NAMES.items():
        assert names[cid] == cname


def test_train_val_rebalance_and_split_preservation(tmp_path: Path) -> None:
    """4. Verify deterministic train/val rebalancing when force_rebalance=True vs preserving original splits."""
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    # Create 14 approved samples all originally marked as 'train', plus 4 unapproved samples
    for i in range(18):
        stem = f"item_{i:02d}"
        img_file = img_train / f"{stem}.jpg"
        _write_dummy_image(img_file)
        is_approved_item = i < 14
        (review_train / f"{stem}.json").write_text(
            json.dumps({
                "stem": stem,
                "split": "train",
                "image_path": str(img_file),
                "human_approved": is_approved_item,
                "done": "ok" if is_approved_item else "pending",
                "trusted": [{"class_id": i % 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
            }),
            encoding="utf-8",
        )

    pool_root = tmp_path / "approved_pool"
    pool = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)

    stats = pool.prepare_approved_training_dataset(
        source_dataset_dir=src_dataset,
        val_ratio=0.15,
        force_rebalance=True,
    )

    assert stats["total_approved_images"] == 14
    assert stats["train_count"] + stats["val_count"] == 14
    assert stats["val_count"] >= 2  # idx 0, 6, 12 -> 3 val items for 14 items with step int(1/0.15)=6
    assert stats["train_count"] == 11
    assert stats["val_count"] == 3

    train_imgs = list(pool.images_train_dir.glob("*.jpg"))
    val_imgs = list(pool.images_val_dir.glob("*.jpg"))
    assert len(train_imgs) == 11
    assert len(val_imgs) == 3


def test_top_level_prepare_approved_training_dataset_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5. Verify top-level `prepare_approved_training_dataset` function delegates to the pool and writes valid outputs."""
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    # 2 approved, 2 unapproved/suspicious
    items = [
        ("top_ok_1", True, "ok", [{"class_id": 0, "cx": 0.25, "cy": 0.25, "w": 0.1, "h": 0.1}]),
        ("top_ok_2", False, "ok", [{"class_id": 7, "cx": 0.75, "cy": 0.75, "w": 0.2, "h": 0.2}]),
        ("top_suspect_1", False, "pending", [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]),
        ("top_suspect_2", False, "suspect", []),
    ]

    for stem, human_ok, done_state, boxes in items:
        img_file = img_train / f"{stem}.jpg"
        _write_dummy_image(img_file)
        (review_train / f"{stem}.json").write_text(
            json.dumps({
                "stem": stem,
                "split": "train",
                "image_path": str(img_file),
                "human_approved": human_ok,
                "done": done_state,
                "trusted": boxes,
            }),
            encoding="utf-8",
        )

    isolated_pool = ApprovedDatasetPool(
        pool_root=tmp_path / "isolated_approved",
        project_root=tmp_path,
    )
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", isolated_pool)

    result = prepare_approved_training_dataset(
        source_dataset_dir=src_dataset,
        val_ratio=0.15,
        force_rebalance=False,
    )

    assert result["total_approved_images"] == 2
    assert result["total_boxes"] == 2
    assert isolated_pool.is_approved("top_ok_1") is True
    assert isolated_pool.is_approved("top_ok_2") is True
    assert isolated_pool.is_approved("top_suspect_1") is False
    assert isolated_pool.is_approved("top_suspect_2") is False

    yaml_file = Path(result["data_yaml_path"])
    assert yaml_file.is_file()
    yaml_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    assert yaml_data["nc"] == 22
    assert yaml_data["train"] == "images/train"
    assert yaml_data["val"] == "images/val"
