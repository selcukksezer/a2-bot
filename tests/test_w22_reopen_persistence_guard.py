# -*- coding: utf-8 -*-
"""tests/test_w22_reopen_persistence_guard.py
=============================================
Worker W22 Unit Tests: Reopen Persistence Guard.
Verifies that approved frames (`is_approved(stem)` or JSON `"done": "ok"`,
`"human_approved": True`) never re-enter the pending review queue when the
dialog is closed and reopened.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dataset_factory.approved_pool as approved_pool_mod
from dataset_factory.approved_pool import ApprovedDatasetPool, is_approved
from dataset_factory.dataset_validator import count_pending_review_items
from dataset_factory.review_queue_sorter import ReviewQueueSorter


def _write_dummy_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xD9")


def _load_pending_queue_on_reopen(
    dataset_dir: Path,
    pool: ApprovedDatasetPool,
    sorter: ReviewQueueSorter,
) -> List[Dict[str, Any]]:
    """Simulate dialog reopen scanning review JSONs and filtering out approved stems/items."""
    review_root = dataset_dir / "review"
    loaded_items: List[Dict[str, Any]] = []
    for split in ("train", "val"):
        split_dir = review_root / split
        if not split_dir.is_dir():
            continue
        for jpath in sorted(split_dir.glob("*.json")):
            if jpath.name in ("trusted_seed.json", "manifest_stats.json"):
                continue
            payload = json.loads(jpath.read_text(encoding="utf-8"))
            stem = str(payload.get("stem") or jpath.stem)
            if pool.is_approved(stem):
                continue
            loaded_items.append(payload)
    return sorter.filter_pending(loaded_items)


def test_reopen_persistence_via_approved_pool_is_approved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """1. Verify stems approved in ApprovedDatasetPool remain approved across close/reopen and are excluded via is_approved(stem)."""
    dataset_dir = tmp_path / "dataset"
    pool_root = dataset_dir / "approved"

    # Session 1 (Dialog Open): Approve 2 frames, leave 2 frames unapproved
    pool_session_1 = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)
    for stem in ("frame_001", "frame_002", "frame_003", "frame_004"):
        img_path = dataset_dir / "images" / "train" / f"{stem}.jpg"
        _write_dummy_jpeg(img_path)
        rev_path = dataset_dir / "review" / "train" / f"{stem}.json"
        rev_path.parent.mkdir(parents=True, exist_ok=True)
        rev_path.write_text(
            json.dumps(
                {
                    "stem": stem,
                    "split": "train",
                    "weak": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "conf": 0.55}],
                    "trusted": [],
                    "done": None,
                }
            ),
            encoding="utf-8",
        )

    assert pool_session_1.add_approved_sample(
        stem="frame_001",
        split="train",
        trusted_boxes=[{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
    )
    assert pool_session_1.add_approved_sample(
        stem="frame_002.jpg",
        split="val",
        trusted_boxes=[{"class_id": 3, "cx": 0.4, "cy": 0.4, "w": 0.2, "h": 0.2}],
    )

    # Simulate closing dialog and reopening a brand-new pool instance
    del pool_session_1
    pool_session_2 = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool_session_2)

    assert pool_session_2.is_approved("frame_001") is True
    assert pool_session_2.is_approved("frame_002") is True
    assert is_approved("frame_001") is True
    assert is_approved("frame_002.jpg") is True
    assert pool_session_2.is_approved("frame_003") is False
    assert pool_session_2.is_approved("frame_004") is False

    sorter = ReviewQueueSorter()
    pending_on_reopen = _load_pending_queue_on_reopen(dataset_dir, pool_session_2, sorter)
    pending_stems = [it["stem"] for it in pending_on_reopen]
    assert pending_stems == ["frame_003", "frame_004"]
    assert "frame_001" not in pending_stems
    assert "frame_002" not in pending_stems


def test_reopen_persistence_via_review_json_done_ok_and_human_approved(tmp_path: Path) -> None:
    """2. Verify review JSONs persisted with 'done': 'ok' or 'human_approved': True never re-enter pending queue on reopen."""
    dataset_dir = tmp_path / "dataset"
    review_train = dataset_dir / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "stem": "approved_by_done_ok",
            "split": "train",
            "done": "ok",
            "human_approved": False,
            "trusted": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
            "weak": [],
        },
        {
            "stem": "approved_by_human_flag",
            "split": "train",
            "done": None,
            "human_approved": True,
            "trusted": [{"class_id": 2, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}],
            "weak": [],
        },
        {
            "stem": "approved_by_both_flags",
            "split": "train",
            "done": "ok",
            "human_approved": True,
            "trusted": [{"class_id": 4, "cx": 0.6, "cy": 0.6, "w": 0.15, "h": 0.15}],
            "weak": [],
        },
        {
            "stem": "still_pending_weak",
            "split": "train",
            "done": None,
            "human_approved": False,
            "trusted": [],
            "weak": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "conf": 0.48}],
        },
    ]

    for s in samples:
        (review_train / f"{s['stem']}.json").write_text(json.dumps(s), encoding="utf-8")

    # Simulate reopen: read all JSON files from disk and filter pending
    reloaded = [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(review_train.glob("*.json"))
    ]
    sorter = ReviewQueueSorter()

    assert sorter.is_approved(next(x for x in reloaded if x["stem"] == "approved_by_done_ok")) is True
    assert sorter.is_approved(next(x for x in reloaded if x["stem"] == "approved_by_human_flag")) is True
    assert sorter.is_approved(next(x for x in reloaded if x["stem"] == "approved_by_both_flags")) is True
    assert sorter.is_approved(next(x for x in reloaded if x["stem"] == "still_pending_weak")) is False

    pending_queue = sorter.sort_queue(reloaded, only_pending=True)
    assert [it["stem"] for it in pending_queue] == ["still_pending_weak"]
    assert count_pending_review_items(dataset_dir) == 1


def test_reopen_combined_guard_pool_and_json_flags(tmp_path: Path) -> None:
    """3. Verify dual-layer guard (pool.is_approved + JSON done/human_approved flags) blocks all approved variants on reopen."""
    dataset_dir = tmp_path / "dataset"
    review_train = dataset_dir / "review" / "train"
    img_train = dataset_dir / "images" / "train"
    review_train.mkdir(parents=True, exist_ok=True)
    img_train.mkdir(parents=True, exist_ok=True)

    records = [
        # Variant A: In ApprovedDatasetPool, even if review JSON was not updated
        {"stem": "var_pool_only", "split": "train", "done": None, "human_approved": False, "weak": [{"class_id": 0, "conf": 0.5}]},
        # Variant B: JSON has done="ok"
        {"stem": "var_done_ok", "split": "train", "done": "ok", "human_approved": False, "trusted": [{"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]},
        # Variant C: JSON has human_approved=True
        {"stem": "var_human_true", "split": "train", "done": "pending", "human_approved": True, "trusted": [{"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]},
        # Variant D: Genuinely unapproved pending item
        {"stem": "var_unapproved", "split": "train", "done": None, "human_approved": False, "weak": [{"class_id": 3, "conf": 0.44}]},
    ]

    for rec in records:
        stem = rec["stem"]
        img_file = img_train / f"{stem}.jpg"
        _write_dummy_jpeg(img_file)
        rec["image_path"] = str(img_file)
        (review_train / f"{stem}.json").write_text(json.dumps(rec), encoding="utf-8")

    pool_1 = ApprovedDatasetPool(project_root=tmp_path)
    pool_1.add_approved_sample(
        stem="var_pool_only",
        split="train",
        trusted_boxes=[{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
    )
    # Sync JSON-approved items into the pool as well
    pool_1.prepare_approved_training_dataset(source_dataset_dir=dataset_dir)

    # Reopen dialog with new pool & sorter instances
    pool_2 = ApprovedDatasetPool(project_root=tmp_path)
    sorter = ReviewQueueSorter()

    assert pool_2.is_approved("var_pool_only") is True
    assert pool_2.is_approved("var_done_ok") is True
    assert pool_2.is_approved("var_human_true") is True
    assert pool_2.is_approved("var_unapproved") is False

    pending = _load_pending_queue_on_reopen(dataset_dir, pool_2, sorter)
    assert [it["stem"] for it in pending] == ["var_unapproved"]


def test_reopen_manifest_recovery_from_disk_prevents_requeue(tmp_path: Path) -> None:
    """4. Verify that even if approved_manifest.json is deleted before reopen, disk sync rebuilds approved stems and prevents requeue."""
    dataset_dir = tmp_path / "dataset"
    review_train = dataset_dir / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    pool_1 = ApprovedDatasetPool(project_root=tmp_path)
    for stem in ("disk_guard_1", "disk_guard_2"):
        pool_1.add_approved_sample(
            stem=stem,
            split="train",
            image_bytes=b"FAKE_JPEG_BYTES",
            trusted_boxes=[{"class_id": 5, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        )
        (review_train / f"{stem}.json").write_text(
            json.dumps({"stem": stem, "split": "train", "done": None, "weak": [{"class_id": 5, "conf": 0.4}]}),
            encoding="utf-8",
        )

    (review_train / "pending_guard_3.json").write_text(
        json.dumps({"stem": "pending_guard_3", "split": "train", "done": None, "weak": [{"class_id": 5, "conf": 0.4}]}),
        encoding="utf-8",
    )

    # Delete manifest to simulate corruption/loss prior to dialog reopen
    assert pool_1.manifest_path.is_file()
    pool_1.manifest_path.unlink()

    # Reopen pool: _load_or_rebuild_manifest triggers sync_from_disk automatically
    pool_reopened = ApprovedDatasetPool(project_root=tmp_path)
    assert pool_reopened.is_approved("disk_guard_1") is True
    assert pool_reopened.is_approved("disk_guard_2") is True
    assert pool_reopened.is_approved("pending_guard_3") is False

    sorter = ReviewQueueSorter()
    pending = _load_pending_queue_on_reopen(dataset_dir, pool_reopened, sorter)
    assert [it["stem"] for it in pending] == ["pending_guard_3"]


def test_multiple_close_reopen_cycles_incremental_approvals_never_regress(tmp_path: Path) -> None:
    """5. Verify across 3 consecutive approve -> close -> reopen cycles that approved frames monotonically leave the pending queue."""
    dataset_dir = tmp_path / "dataset"
    review_train = dataset_dir / "review" / "train"
    img_train = dataset_dir / "images" / "train"
    review_train.mkdir(parents=True, exist_ok=True)
    img_train.mkdir(parents=True, exist_ok=True)

    stems = [f"cycle_frame_{i:02d}" for i in range(1, 7)]
    for stem in stems:
        _write_dummy_jpeg(img_train / f"{stem}.jpg")
        (review_train / f"{stem}.json").write_text(
            json.dumps(
                {
                    "stem": stem,
                    "split": "train",
                    "done": None,
                    "human_approved": False,
                    "weak": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "conf": 0.5}],
                    "trusted": [],
                }
            ),
            encoding="utf-8",
        )

    sorter = ReviewQueueSorter()

    # Cycle 1: Approve first 2 frames via done="ok" + pool, then close & reopen
    pool_c1 = ApprovedDatasetPool(project_root=tmp_path)
    for stem in stems[:2]:
        jfile = review_train / f"{stem}.json"
        data = json.loads(jfile.read_text(encoding="utf-8"))
        data["done"] = "ok"
        data["trusted"] = data.pop("weak", [])
        data["weak"] = []
        jfile.write_text(json.dumps(data), encoding="utf-8")
        pool_c1.add_approved_sample(stem=stem, split="train", trusted_boxes=data["trusted"])

    pool_reopen_1 = ApprovedDatasetPool(project_root=tmp_path)
    pending_1 = _load_pending_queue_on_reopen(dataset_dir, pool_reopen_1, sorter)
    assert [it["stem"] for it in pending_1] == stems[2:]
    assert count_pending_review_items(dataset_dir) == 4

    # Cycle 2: Approve next 2 frames via human_approved=True + done="ok", then close & reopen
    for stem in stems[2:4]:
        jfile = review_train / f"{stem}.json"
        data = json.loads(jfile.read_text(encoding="utf-8"))
        data["human_approved"] = True
        data["done"] = "ok"
        data["trusted"] = data.pop("weak", [])
        data["weak"] = []
        jfile.write_text(json.dumps(data), encoding="utf-8")
        pool_reopen_1.add_approved_sample(stem=stem, split="train", trusted_boxes=data["trusted"])

    pool_reopen_2 = ApprovedDatasetPool(project_root=tmp_path)
    pending_2 = _load_pending_queue_on_reopen(dataset_dir, pool_reopen_2, sorter)
    assert [it["stem"] for it in pending_2] == stems[4:]
    assert count_pending_review_items(dataset_dir) == 2

    # Cycle 3: Approve final 2 frames, then close & reopen
    for stem in stems[4:]:
        jfile = review_train / f"{stem}.json"
        data = json.loads(jfile.read_text(encoding="utf-8"))
        data["human_approved"] = True
        data["done"] = "ok"
        data["trusted"] = data.pop("weak", [])
        data["weak"] = []
        jfile.write_text(json.dumps(data), encoding="utf-8")
        pool_reopen_2.add_approved_sample(stem=stem, split="val", trusted_boxes=data["trusted"])

    pool_reopen_3 = ApprovedDatasetPool(project_root=tmp_path)
    pending_3 = _load_pending_queue_on_reopen(dataset_dir, pool_reopen_3, sorter)
    assert pending_3 == []
    assert count_pending_review_items(dataset_dir) == 0
    assert pool_reopen_3.get_approved_stats()["total_approved_images"] == 6

    all_disk_jsons = [
        json.loads((review_train / f"{s}.json").read_text(encoding="utf-8"))
        for s in stems
    ]
    summary = sorter.summarize_queue(all_disk_jsons)
    assert summary["total_frames"] == 6
    assert summary["approved_frames"] == 6
    assert summary["pending_frames"] == 0
