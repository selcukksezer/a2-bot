# -*- coding: utf-8 -*-
"""W41: Unit tests for atomic writing and synchronization between review/{split}/{stem}.json and labels/{split}/{stem}.txt."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import List
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory.dataset_validator import (
    atomic_write_json,
    atomic_write_text,
    sync_all_review_txt_pairs,
    verify_and_sync_review_txt_pair,
)
from dataset_factory.label_trust import (
    _atomic_write_json,
    _atomic_write_text,
    _rewrite_label_txt,
    commit_split_labels,
    prune_low_confidence_boxes_in_dataset,
)


def _list_tmp_artifacts(directory: Path) -> List[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if ".tmp" in p.name]


def test_w41_01_commit_split_labels_atomic_json_and_txt_sync(tmp_path: Path) -> None:
    """commit_split_labels atomically writes review/{split}/{stem}.json and labels/{split}/{stem}.txt in sync."""
    split = "train"
    stem = "gather_aria_001"
    label_path = tmp_path / "labels" / split / f"{stem}.txt"
    review_path = tmp_path / "review" / split / f"{stem}.json"

    annotations = [
        {
            "class_id": 0,
            "cx": 0.450000,
            "cy": 0.550000,
            "w": 0.120000,
            "h": 0.180000,
            "confidence": 0.94,
            "hand_label": True,
            "trusted": True,
            "source": "human_hand",
        },
        {
            "class_id": 3,
            "cx": 0.250000,
            "cy": 0.350000,
            "w": 0.100000,
            "h": 0.140000,
            "confidence": 0.58,
            "trusted": False,
            "weak": True,
            "source": "dataset_requeue",
        },
    ]

    res = commit_split_labels(
        dataset_dir=tmp_path,
        split=split,
        stem=stem,
        label_path=label_path,
        annotations=annotations,
        human_approved=True,
        done="ok",
    )

    assert review_path.is_file()
    assert label_path.is_file()
    assert _list_tmp_artifacts(review_path.parent) == []
    assert _list_tmp_artifacts(label_path.parent) == []

    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert payload["stem"] == stem
    assert payload["split"] == split
    assert payload["human_approved"] is True
    assert payload["done"] == "ok"

    # Both boxes were marked human_approved=True by commit_split_labels, so both are trusted and synced to .txt
    assert len(payload["trusted"]) == 2
    assert len(payload["weak"]) == 0
    assert res["train_box_count"] == 2

    txt_lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert txt_lines == [
        "0 0.450000 0.550000 0.120000 0.180000",
        "3 0.250000 0.350000 0.100000 0.140000",
    ]
    assert verify_and_sync_review_txt_pair(review_path, label_path, auto_repair=False) is True


def test_w41_02_atomic_write_temp_file_replacement_and_cleanup_on_failure(tmp_path: Path) -> None:
    """Atomic writers use os.replace without residual .tmp files and preserve existing data if os.replace fails."""
    split = "val"
    stem = "atomic_guard_02"
    review_path = tmp_path / "review" / split / f"{stem}.json"
    label_path = tmp_path / "labels" / split / f"{stem}.txt"

    initial_json = {
        "stem": stem,
        "split": split,
        "trusted": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        "weak": [],
    }
    atomic_write_json(review_path, initial_json)
    _rewrite_label_txt(label_path, initial_json["trusted"])

    assert review_path.is_file()
    assert label_path.is_file()
    assert _list_tmp_artifacts(review_path.parent) == []
    assert _list_tmp_artifacts(label_path.parent) == []

    # Simulate an interrupted/failed os.replace in dataset_validator.atomic_write_text:
    # original target must remain untouched and temporary file must be cleaned up.
    with patch("dataset_factory.dataset_validator.os.replace", side_effect=OSError("disk full simulated")):
        with pytest.raises(OSError):
            atomic_write_text(label_path, "9 0.900000 0.900000 0.100000 0.100000\n")

    assert label_path.read_text(encoding="utf-8").strip() == "1 0.500000 0.500000 0.200000 0.200000"
    assert _list_tmp_artifacts(label_path.parent) == []

    # Verify label_trust._atomic_write_json and _atomic_write_text also leave zero .tmp files
    updated_json = {
        "stem": stem,
        "split": split,
        "trusted": [{"class_id": 2, "cx": 0.4, "cy": 0.6, "w": 0.15, "h": 0.15}],
        "weak": [],
    }
    _atomic_write_json(review_path, updated_json)
    _atomic_write_text(label_path, "2 0.400000 0.600000 0.150000 0.150000\n")

    assert _list_tmp_artifacts(review_path.parent) == []
    assert _list_tmp_artifacts(label_path.parent) == []
    assert verify_and_sync_review_txt_pair(review_path, label_path, auto_repair=False) is True


def test_w41_03_verify_and_sync_orphan_json_and_orphan_txt_recovery(tmp_path: Path) -> None:
    """verify_and_sync_review_txt_pair detects and atomically repairs orphan JSON and orphan TXT files."""
    rev_train = tmp_path / "review" / "train"
    lbl_train = tmp_path / "labels" / "train"
    rev_val = tmp_path / "review" / "val"
    lbl_val = tmp_path / "labels" / "val"
    rev_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)
    rev_val.mkdir(parents=True, exist_ok=True)
    lbl_val.mkdir(parents=True, exist_ok=True)

    # 1. Orphan JSON in train split (missing labels/train/orphan_json_stem.txt)
    orphan_json_path = rev_train / "orphan_json_stem.json"
    missing_txt_path = lbl_train / "orphan_json_stem.txt"
    atomic_write_json(
        orphan_json_path,
        {
            "stem": "orphan_json_stem",
            "split": "train",
            "trusted": [
                {"class_id": 4, "cx": 0.35, "cy": 0.45, "w": 0.12, "h": 0.16},
                {"class_id": 7, "cx": 0.52, "cy": 0.48, "w": 0.08, "h": 0.08},
            ],
            "weak": [],
        },
    )

    assert verify_and_sync_review_txt_pair(orphan_json_path, missing_txt_path, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(orphan_json_path, missing_txt_path, auto_repair=True) is True
    assert missing_txt_path.is_file()
    assert _list_tmp_artifacts(lbl_train) == []
    repaired_lines = [ln.strip() for ln in missing_txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert repaired_lines == [
        "4 0.350000 0.450000 0.120000 0.160000",
        "7 0.520000 0.480000 0.080000 0.080000",
    ]

    # 2. Orphan TXT in val split (missing review/val/orphan_txt_stem.json)
    missing_json_path = rev_val / "orphan_txt_stem.json"
    orphan_txt_path = lbl_val / "orphan_txt_stem.txt"
    atomic_write_text(orphan_txt_path, "5 0.600000 0.400000 0.200000 0.220000\n")

    assert verify_and_sync_review_txt_pair(missing_json_path, orphan_txt_path, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(missing_json_path, orphan_txt_path, auto_repair=True) is True
    assert missing_json_path.is_file()
    assert _list_tmp_artifacts(rev_val) == []

    reconstructed = json.loads(missing_json_path.read_text(encoding="utf-8"))
    assert reconstructed["stem"] == "orphan_txt_stem"
    assert reconstructed["split"] == "val"
    assert reconstructed["auto_repaired"] is True
    assert len(reconstructed["trusted"]) == 1
    assert reconstructed["trusted"][0]["class_id"] == 5
    assert reconstructed["trusted"][0]["cx"] == pytest.approx(0.60)


def test_w41_04_verify_and_sync_corrupt_or_mismatched_pair_repair(tmp_path: Path) -> None:
    """verify_and_sync_review_txt_pair repairs corrupt JSON from TXT, corrupt/desynced TXT from JSON, and rejected frames."""
    rev_dir = tmp_path / "review" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    rev_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Case A: Corrupt JSON salvaged atomically from valid TXT
    corrupt_json = rev_dir / "corrupt_json_frame.json"
    valid_txt = lbl_dir / "corrupt_json_frame.txt"
    corrupt_json.write_text("{corrupt_json: [unclosed", encoding="utf-8")
    valid_txt.write_text("2 0.500000 0.500000 0.140000 0.140000\n", encoding="utf-8")

    assert verify_and_sync_review_txt_pair(corrupt_json, valid_txt, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(corrupt_json, valid_txt, auto_repair=True) is True
    salvaged = json.loads(corrupt_json.read_text(encoding="utf-8"))
    assert len(salvaged["trusted"]) == 1
    assert salvaged["trusted"][0]["class_id"] == 2
    assert verify_and_sync_review_txt_pair(corrupt_json, valid_txt, auto_repair=False) is True

    # Case B: Mismatched / malformed TXT overwritten atomically from valid review JSON ground truth
    gt_json = rev_dir / "desynced_frame.json"
    stale_txt = lbl_dir / "desynced_frame.txt"
    atomic_write_json(
        gt_json,
        {
            "stem": "desynced_frame",
            "split": "train",
            "trusted": [
                {"class_id": 10, "cx": 0.30, "cy": 0.50, "w": 0.10, "h": 0.15},
                {"class_id": 12, "cx": 0.70, "cy": 0.50, "w": 0.12, "h": 0.12},
            ],
            "weak": [],
        },
    )
    stale_txt.write_text("0 0.500000 0.500000 0.100000 0.100000\nmalformed_line\n", encoding="utf-8")

    assert verify_and_sync_review_txt_pair(gt_json, stale_txt, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(gt_json, stale_txt, auto_repair=True) is True
    synced_lines = [ln.strip() for ln in stale_txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert synced_lines == [
        "10 0.300000 0.500000 0.100000 0.150000",
        "12 0.700000 0.500000 0.120000 0.120000",
    ]

    # Case C: Rejected review JSON (done="no") empties stale boxes in TXT
    rejected_json = rev_dir / "rejected_frame.json"
    non_empty_txt = lbl_dir / "rejected_frame.txt"
    atomic_write_json(
        rejected_json,
        {
            "stem": "rejected_frame",
            "split": "train",
            "done": "no",
            "trusted": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
            "weak": [],
        },
    )
    non_empty_txt.write_text("1 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")

    assert verify_and_sync_review_txt_pair(rejected_json, non_empty_txt, auto_repair=False) is False
    assert verify_and_sync_review_txt_pair(rejected_json, non_empty_txt, auto_repair=True) is True
    assert non_empty_txt.read_text(encoding="utf-8") == ""


def test_w41_05_sync_all_review_txt_pairs_and_box_pruning_across_splits(tmp_path: Path) -> None:
    """sync_all_review_txt_pairs and prune_low_confidence_boxes_in_dataset keep review JSON and labels TXT synchronized across train/val."""
    for split in ("train", "val"):
        (tmp_path / "review" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # 1. Already synced pair in train
    j_synced = tmp_path / "review" / "train" / "s_ok.json"
    t_synced = tmp_path / "labels" / "train" / "s_ok.txt"
    atomic_write_json(
        j_synced,
        {
            "stem": "s_ok",
            "split": "train",
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.92}],
            "weak": [
                {"class_id": 1, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1, "confidence": 0.52, "weak": True}
            ],
        },
    )
    atomic_write_text(t_synced, "0 0.500000 0.500000 0.100000 0.100000\n")

    # 2. Orphan JSON in train
    j_orphan = tmp_path / "review" / "train" / "s_orphan_j.json"
    atomic_write_json(
        j_orphan,
        {
            "stem": "s_orphan_j",
            "split": "train",
            "trusted": [{"class_id": 3, "cx": 0.4, "cy": 0.4, "w": 0.15, "h": 0.15, "confidence": 0.90}],
            "weak": [],
        },
    )

    # 3. Orphan TXT in val
    t_orphan = tmp_path / "labels" / "val" / "s_orphan_t.txt"
    atomic_write_text(t_orphan, "4 0.600000 0.600000 0.120000 0.120000\n")

    stats = sync_all_review_txt_pairs(tmp_path, splits=("train", "val"), auto_repair=True)
    assert stats["total_pairs"] == 3
    assert stats["already_synced"] == 1
    assert stats["auto_repaired"] == 2
    assert stats["orphan_json_fixed"] == 1
    assert stats["orphan_txt_fixed"] == 1
    assert stats["failed"] == 0

    # Verify box-level pruning atomically updates both review JSON and labels TXT in sync
    # (prunes the weak box in s_ok.json with conf=0.52 and the unscored box in s_orphan_t.json with conf=0.0)
    prune_stats = prune_low_confidence_boxes_in_dataset(
        tmp_path,
        threshold=0.85,
        record_negative_exemplars=False,
    )
    assert prune_stats["ok"] is True
    assert prune_stats["pruned_boxes"] == 2
    assert (tmp_path / "labels" / "val" / "s_orphan_t.txt").read_text(encoding="utf-8") == ""

    post_stats = sync_all_review_txt_pairs(tmp_path, splits=("train", "val"), auto_repair=False)
    assert post_stats["total_pairs"] == 3
    assert post_stats["already_synced"] == 3
    assert post_stats["failed"] == 0
