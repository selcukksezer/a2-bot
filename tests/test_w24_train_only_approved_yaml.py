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
from dataset_factory.class_taxonomy import CANONICAL_NAMES


def _write_dummy_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xD9")


def test_only_approved_frames_included_in_training_pool(tmp_path: Path) -> None:
    """1. Verify that only frames marked as approved (human_approved=True) are included.

    Only an explicit human approval counts (D38/F-0828): done="ok" and
    approved=True are pipeline flags, not human approval, and must not
    admit a sample into the approved pool. P1-10 / K-21: the frame-level
    flag alone is not enough either — at least one trusted box must
    carry per-box human provenance (`human_approved` / `hand_label`),
    so the admitted frame's box carries `"human_approved": True`.
    """
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    records = [
        # Approved via human_approved=True (+ per-box provenance, K-21)
        {
            "stem": "frame_approved_human",
            "split": "train",
            "human_approved": True,
            "done": "pending",
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "human_approved": True}],
        },
        # NOT approved: done="ok" alone (D38/F-0828) — human_approved is False.
        {
            "stem": "frame_approved_done_ok",
            "split": "train",
            "human_approved": False,
            "done": "ok",
            "trusted": [{"class_id": 3, "cx": 0.4, "cy": 0.6, "w": 0.1, "h": 0.15}],
        },
        # NOT approved: approved=True alone (D38/F-0828) — a pipeline flag,
        # not a human approval.
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

    assert stats["total_approved_images"] == 1
    assert stats["train_count"] == 1
    assert stats["val_count"] == 0
    assert stats["total_boxes"] == 1

    assert pool.is_approved("frame_approved_human") is True
    assert pool.is_approved("frame_approved_done_ok") is False
    assert pool.is_approved("frame_approved_flag") is False
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
        "trusted": [{"class_id": 4, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "human_approved": True}],
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
            # P1-10 / K-21: per-box provenance — the frame flag alone admits nothing.
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "human_approved": True}],
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
    # B-09 DÜZELTME (2026-11-10, canlı kanıt): Ultralytics path'i
    # yaml'ın dizinine göre DEĞİL, CWD'ye göre çözer; bu yüzden `path`
    # yaml'ın klasörünün MUTLAK yoludur (forward-slash).
    assert Path(parsed["path"]).is_absolute()
    assert parsed["path"] == yaml_path.parent.resolve().as_posix()
    assert "\\" not in parsed["path"]
    assert parsed["train"] == "images/train"
    assert parsed["val"] == "images/val"
    assert parsed["nc"] == len(CANONICAL_NAMES) == 29

    names = parsed["names"]
    assert isinstance(names, dict)
    assert len(names) == 29
    for cid, cname in CANONICAL_NAMES.items():
        assert names[cid] == cname

    # Semantic checks (beyond field names): the yaml must describe a dataset
    # that actually exists on disk and is consumable by a YOLO trainer.
    # B-09: `path` is '.' relative to the yaml file itself, not to the CWD.
    yaml_root = (yaml_path.parent / parsed["path"]).resolve()
    train_dir = yaml_root / parsed["train"]
    val_dir = yaml_root / parsed["val"]
    assert train_dir.is_dir() and val_dir.is_dir()
    # The approved sample must be present as an image with a matching label file.
    train_img = train_dir / "sample_01.jpg"
    assert train_img.is_file()
    label_file = yaml_root / "labels" / "train" / "sample_01.txt"
    assert label_file.is_file()
    label_lines = [ln.split() for ln in label_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(label_lines) == 1
    class_id = int(label_lines[0][0])
    assert 0 <= class_id < parsed["nc"]
    assert names[class_id] == CANONICAL_NAMES[class_id]
    # Every box coordinate must be a normalized YOLO value in [0, 1].
    for coord in label_lines[0][1:5]:
        assert 0.0 <= float(coord) <= 1.0
    # Every image in the yaml's train/val dirs must have a label file and vice versa.
    for split, img_dir in (("train", train_dir), ("val", val_dir)):
        for img in img_dir.glob("*.jpg"):
            assert (yaml_root / "labels" / split / f"{img.stem}.txt").is_file()


def test_train_val_rebalance_and_split_preservation(tmp_path: Path) -> None:
    """4. Verify deterministic train/val rebalancing when force_rebalance=True vs preserving original splits.

    The split is asserted against the requested *ratio* (val_ratio=0.15), not
    against exact counts: the hash-based rebalance (P1-09 / K-24) gives no
    exact val-count guarantee — a frame's split depends only on its own
    stem hash (MD5(stem) % 1000 < 150) and the ratio, so for these 14
    stems the val set is deterministically EMPTY (0/14, within +/-0.15
    of the ratio). The same input must deterministically produce the
    same val set, and adding one frame must not move any existing
    frame's split.
    """
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
                # P1-10 / K-21: admitted frames need per-box provenance;
                # unapproved frames keep provenance-free boxes.
                "trusted": [{"class_id": i % 6, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, **({"human_approved": True} if is_approved_item else {})}],
            }),
            encoding="utf-8",
        )

    val_ratio = 0.15
    pool_root = tmp_path / "approved_pool"
    pool = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)

    stats = pool.prepare_approved_training_dataset(
        source_dataset_dir=src_dataset,
        val_ratio=val_ratio,
        force_rebalance=True,
    )

    total = stats["total_approved_images"]
    assert total == 14
    assert stats["train_count"] + stats["val_count"] == total
    # Hash-based rebalance (P1-09 / K-24): no exact val-count guarantee.
    # For these 14 stems MD5(stem) % 1000 >= 150 for all of them, so the
    # val set is deterministically empty (0/14, within +/-0.15 of 0.15).
    # This is the locked new behaviour — NOT "val must be non-empty".
    assert stats["val_count"] == 0
    assert stats["train_count"] == 14
    assert abs(stats["val_count"] / total - val_ratio) <= 0.15

    train_imgs = list(pool.images_train_dir.glob("*.jpg"))
    val_imgs = list(pool.images_val_dir.glob("*.jpg"))
    assert len(train_imgs) == stats["train_count"]
    assert len(val_imgs) == stats["val_count"]
    # Every approved image lands in exactly one split (no loss, no duplication).
    assert {p.stem for p in train_imgs} | {p.stem for p in val_imgs} == {
        f"item_{i:02d}" for i in range(14)
    }
    assert {p.stem for p in train_imgs} & {p.stem for p in val_imgs} == set()

    # Determinism: a second run on a fresh pool with the same input must pick
    # exactly the same val set.
    pool2 = ApprovedDatasetPool(pool_root=tmp_path / "approved_pool_2", project_root=tmp_path)
    stats2 = pool2.prepare_approved_training_dataset(
        source_dataset_dir=src_dataset,
        val_ratio=val_ratio,
        force_rebalance=True,
    )
    assert stats2["val_count"] == stats["val_count"]
    assert {p.stem for p in pool2.images_val_dir.glob("*.jpg")} == {
        p.stem for p in val_imgs
    }

    # K-24 core property: adding one approved frame must not move any
    # existing frame's split (the old Bresenham idx-based allocation
    # moved ~2x val_count frames on a single add/remove).
    stem14 = "item_14"
    _write_dummy_image(img_train / f"{stem14}.jpg")
    (review_train / f"{stem14}.json").write_text(
        json.dumps({
            "stem": stem14,
            "split": "train",
            "image_path": str(img_train / f"{stem14}.jpg"),
            "human_approved": True,
            "done": "ok",
            # P1-10 / K-21: per-box provenance — P1-10'un taraması bu
            # bloğu atlamıştı (P1-09'dan gelen ekleme bloğu); köken
            # kanıtı olmayan kutu havuza alınmaz, 15. kare sayılmazdı.
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "human_approved": True}],
        }),
        encoding="utf-8",
    )
    pool3 = ApprovedDatasetPool(pool_root=tmp_path / "approved_pool_3", project_root=tmp_path)
    stats3 = pool3.prepare_approved_training_dataset(
        source_dataset_dir=src_dataset,
        val_ratio=val_ratio,
        force_rebalance=True,
    )
    assert stats3["total_approved_images"] == 15
    assert {p.stem for p in pool3.images_train_dir.glob("*.jpg")} >= {
        p.stem for p in train_imgs
    }
    assert {p.stem for p in pool3.images_val_dir.glob("*.jpg")} == {
        p.stem for p in val_imgs
    }


def test_top_level_prepare_approved_training_dataset_function(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5. Verify top-level `prepare_approved_training_dataset` function delegates to the pool and writes valid outputs."""
    src_dataset = tmp_path / "dataset"
    img_train = src_dataset / "images" / "train"
    review_train = src_dataset / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    # 1 approved (human), 3 not: done="ok" alone is not approval (D38/F-0828),
    # so top_ok_2 (human False, done "ok") is excluded. P1-10 / K-21: the
    # admitted frame's box also needs per-box provenance.
    items = [
        ("top_ok_1", True, "ok", [{"class_id": 0, "cx": 0.25, "cy": 0.25, "w": 0.1, "h": 0.1, "human_approved": True}]),
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

    assert result["total_approved_images"] == 1
    assert result["total_boxes"] == 1
    assert isolated_pool.is_approved("top_ok_1") is True
    assert isolated_pool.is_approved("top_ok_2") is False
    assert isolated_pool.is_approved("top_suspect_1") is False
    assert isolated_pool.is_approved("top_suspect_2") is False

    yaml_file = Path(result["data_yaml_path"])
    assert yaml_file.is_file()
    yaml_data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
    assert yaml_data["nc"] == len(CANONICAL_NAMES) == 29
    assert yaml_data["train"] == "images/train"
    assert yaml_data["val"] == "images/val"
