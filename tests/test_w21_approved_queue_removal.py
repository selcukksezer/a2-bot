# -*- coding: utf-8 -*-
"""tests/test_w21_approved_queue_removal.py
===========================================
Worker W21 Unit Tests:
Verifies that an approved frame (`human_approved=True`, `done="ok"`) is saved into
`ApprovedDatasetPool` and removed/decremented from the pending review queue.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_factory.approved_pool import ApprovedDatasetPool
from dataset_factory.label_trust import (
    commit_split_labels,
    requeue_labeled_dataset_for_review,
)
from dataset_factory.review_queue_sorter import ReviewQueueSorter


def _write_dummy_jpeg(path: Path) -> bytes:
    """Create a minimal valid JPEG header on disk and return its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xD9"
    path.write_bytes(data)
    return data


def test_single_approved_frame_saved_to_pool_and_removed_from_pending_queue(
    tmp_path: Path,
) -> None:
    """1. Verify a single approved frame (human_approved=True, done='ok') is saved into ApprovedDatasetPool and removed from the pending queue."""
    dataset_dir = tmp_path / "dataset"
    img_path = dataset_dir / "images" / "train" / "frame_001.jpg"
    lbl_path = dataset_dir / "labels" / "train" / "frame_001.txt"
    img_bytes = _write_dummy_jpeg(img_path)

    # Initial queue with 3 pending items
    queue: List[Dict[str, Any]] = [
        {
            "stem": "frame_001",
            "split": "train",
            "human_approved": False,
            "done": None,
            "trusted": [
                {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.92}
            ],
            "weak": [],
        },
        {
            "stem": "frame_002",
            "split": "train",
            "human_approved": False,
            "done": None,
            "trusted": [],
            "weak": [
                {"class_id": 3, "cx": 0.4, "cy": 0.4, "w": 0.15, "h": 0.15, "confidence": 0.62}
            ],
        },
        {
            "stem": "frame_003",
            "split": "train",
            "human_approved": False,
            "done": None,
            "trusted": [],
            "weak": [],
        },
    ]

    sorter = ReviewQueueSorter()
    assert len(sorter.filter_pending(queue)) == 3

    # Approve frame_001: set human_approved=True and done="ok"
    queue[0]["human_approved"] = True
    queue[0]["done"] = "ok"

    commit_res = commit_split_labels(
        dataset_dir=dataset_dir,
        split="train",
        stem="frame_001",
        label_path=lbl_path,
        annotations=queue[0]["trusted"],
        human_approved=True,
        done="ok",
    )
    assert commit_res["human_approved"] is True
    assert commit_res["done"] == "ok"

    # Save to ApprovedDatasetPool
    pool = ApprovedDatasetPool(project_root=tmp_path)
    added = pool.add_approved_sample(
        stem="frame_001",
        split="train",
        image_path=img_path,
        trusted_boxes=queue[0]["trusted"],
        image_bytes=img_bytes,
    )
    assert added is True
    assert pool.is_approved("frame_001") is True
    assert (pool.images_train_dir / "frame_001.jpg").is_file()
    assert (pool.labels_train_dir / "frame_001.txt").is_file()

    # Verify pending queue decreased from 3 to 2 and frame_001 is no longer pending.
    # F-1380: the removal must come from the PRODUCTION filter
    # (ReviewQueueSorter.filter_pending), not from the test's own list
    # comprehension over pool.is_approved — the old code proved nothing
    # about the production queue path.
    pending_after = sorter.filter_pending(queue)
    assert len(pending_after) == 2
    assert [it["stem"] for it in pending_after] == ["frame_002", "frame_003"]


def test_prepare_approved_training_dataset_migrates_approved_and_shrinks_pending_queue(
    tmp_path: Path,
) -> None:
    """2. Verify prepare_approved_training_dataset saves human_approved=True / done='ok' frames to ApprovedDatasetPool and shrinks pending queue."""
    dataset_dir = tmp_path / "dataset"
    img_train = dataset_dir / "images" / "train"
    review_train = dataset_dir / "review" / "train"
    review_train.mkdir(parents=True, exist_ok=True)

    raw_items: List[Dict[str, Any]] = [
        {
            "stem": "sample_a",
            "split": "train",
            "human_approved": True,
            "done": "ok",
            # P1-10 / K-21: admitted frames need per-box provenance.
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "human_approved": True}],
            "weak": [],
        },
        {
            "stem": "sample_b",
            "split": "train",
            "human_approved": True,
            "done": "ok",
            "trusted": [{"class_id": 3, "cx": 0.3, "cy": 0.6, "w": 0.1, "h": 0.1, "human_approved": True}],
            "weak": [],
        },
        {
            "stem": "sample_c",
            "split": "train",
            "human_approved": False,
            "done": None,
            "trusted": [],
            "weak": [{"class_id": 1, "cx": 0.4, "cy": 0.4, "w": 0.15, "h": 0.15}],
        },
        {
            "stem": "sample_d",
            "split": "train",
            "human_approved": False,
            "done": None,
            "trusted": [],
            "weak": [{"class_id": 2, "cx": 0.7, "cy": 0.3, "w": 0.12, "h": 0.12}],
        },
    ]

    for item in raw_items:
        stem = item["stem"]
        img_file = img_train / f"{stem}.jpg"
        _write_dummy_jpeg(img_file)
        item["image_path"] = str(img_file)
        (review_train / f"{stem}.json").write_text(
            json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    pool = ApprovedDatasetPool(project_root=tmp_path)
    stats = pool.prepare_approved_training_dataset(
        source_dataset_dir=dataset_dir,
        force_rebalance=False,
    )

    assert stats["total_approved_images"] == 2
    assert stats["total_boxes"] == 2
    assert pool.is_approved("sample_a") is True
    assert pool.is_approved("sample_b") is True
    assert pool.is_approved("sample_c") is False
    assert pool.is_approved("sample_d") is False

    sorter = ReviewQueueSorter()
    summary = sorter.summarize_queue(raw_items)
    assert summary["total_frames"] == 4
    assert summary["approved_frames"] == 2
    assert summary["pending_frames"] == 2

    remaining_pending = sorter.filter_pending(raw_items)
    assert [it["stem"] for it in remaining_pending] == ["sample_c", "sample_d"]


def test_approved_frame_pool_quarantine_holds_across_reload_and_requeue(
    tmp_path: Path,
) -> None:
    """3. Verify that once a frame is approved (human_approved=True, done='ok') and saved in ApprovedDatasetPool, the POOL quarantine holds across pool reload and requeue.

    §7 P0-8 requeue kararı: el etiketi / tek tek onaylı kutu içeren kare
    asla kuyruğa geri dönmez — requeue korumalı kareyi atlar, eski onayı
    yerinde bırakır (review JSON'un üzerine yazılmaz). Bu yüzden E-04
    yeniden-onay kuralı bu kareye uğramaz: `human_approved` False'a
    dönmez, `was_previously_approved` yazılmaz, kare pending kuyruğuna
    geri girmez. Havuz karantinası sürer (pool reload sonrası da onaylı);
    TXT-only bekleyen kare de K-14 damgasıyla korumalı atlanır, review
    JSON'u hiç oluşmaz."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir = tmp_path / "dataset"
    img_train = dataset_dir / "images" / "train"
    lbl_train = dataset_dir / "labels" / "train"
    rev_train = dataset_dir / "review" / "train"
    img_train.mkdir(parents=True, exist_ok=True)
    lbl_train.mkdir(parents=True, exist_ok=True)
    rev_train.mkdir(parents=True, exist_ok=True)

    # Create 2 frames: one approved, one unapproved
    img_approved = img_train / "quarantine_ok.jpg"
    img_pending = img_train / "still_pending.jpg"
    _write_dummy_jpeg(img_approved)
    _write_dummy_jpeg(img_pending)

    commit_split_labels(
        dataset_dir=dataset_dir,
        split="train",
        stem="quarantine_ok",
        label_path=lbl_train / "quarantine_ok.txt",
        annotations=[
            {
                "class_id": 0,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.2,
                "h": 0.2,
                "confidence": 0.95,
                "trusted": True,
                "hand_label": True,
                "human_approved": True,
            }
        ],
        human_approved=True,
        done="ok",
    )
    (lbl_train / "still_pending.txt").write_text("1 0.4 0.4 0.1 0.1 0.60\n", encoding="utf-8")

    pool1 = ApprovedDatasetPool(project_root=tmp_path)
    pool1.prepare_approved_training_dataset(source_dataset_dir=dataset_dir)
    assert pool1.is_approved("quarantine_ok") is True
    assert pool1.is_approved("still_pending") is False

    # Run requeue on the dataset and reload ApprovedDatasetPool from disk
    requeue_stats = requeue_labeled_dataset_for_review(dataset_dir=dataset_dir)
    assert requeue_stats["ok"] is True
    # §7 P0-8: iki kare de korumalı atlanır (quarantine_ok: el etiketi +
    # tekli onaylı kutu; still_pending: TXT-only K-14 damgası) — hiçbiri
    # kuyruğa girmez.
    assert requeue_stats["samples"] == 0
    assert requeue_stats["boxes"] == 0
    assert requeue_stats["skipped_protected"] == 2

    pool2 = ApprovedDatasetPool(project_root=tmp_path)
    assert pool2.is_approved("quarantine_ok") is True
    assert pool2.is_approved("still_pending") is False

    # Load review JSONs back and filter pending queue
    loaded_review_items: List[Dict[str, Any]] = []
    for rfile in sorted(rev_train.glob("*.json")):
        if rfile.name == "trusted_seed.json":
            continue
        payload = json.loads(rfile.read_text(encoding="utf-8"))
        loaded_review_items.append(payload)

    by_stem = {it["stem"]: it for it in loaded_review_items}
    # §7 P0-8: eski onay yerinde kalır — requeue review JSON'un üzerine
    # yazmaz, onay False'a dönmez, was_previously_approved yazılmaz.
    assert by_stem["quarantine_ok"]["human_approved"] is True
    assert by_stem["quarantine_ok"]["done"] == "ok"
    assert by_stem["quarantine_ok"].get("was_previously_approved") is not True
    assert by_stem["quarantine_ok"]["trusted"][0]["hand_label"] is True
    # TXT-only bekleyen kare kuyruğa hiç girmedi — review JSON'u yok.
    assert "still_pending" not in by_stem
    assert not (rev_train / "still_pending.json").is_file()

    sorter = ReviewQueueSorter()
    pending_queue = sorter.sort_queue(loaded_review_items, only_pending=True)
    assert len(pending_queue) == 0
    assert [it["stem"] for it in pending_queue] == []


def test_bulk_approval_saves_all_approved_to_pool_and_decrements_pending_count(
    tmp_path: Path,
) -> None:
    """4. Verify bulk approval of multiple frames persists them to ApprovedDatasetPool and decrements the pending queue count accordingly."""
    pool = ApprovedDatasetPool(project_root=tmp_path)
    sorter = ReviewQueueSorter()

    queue: List[Dict[str, Any]] = [
        {
            "stem": f"batch_frame_{i:02d}",
            "split": "train" if i % 2 == 0 else "val",
            "human_approved": False,
            "done": None,
            "image_bytes": b"\xFF\xD8\xFF\xE0BULK_JPEG_DATA",
            "trusted_boxes": [{"class_id": i % 4, "cx": 0.5, "cy": 0.5, "w": 0.15, "h": 0.15}],
        }
        for i in range(6)
    ]

    assert sorter.summarize_queue(queue)["pending_frames"] == 6
    assert pool.get_approved_stats()["total_approved_images"] == 0

    # Approve the first 4 frames in bulk
    to_approve = queue[:4]
    for item in to_approve:
        item["human_approved"] = True
        item["done"] = "ok"

    added_count = pool.bulk_add_approved_samples(to_approve)
    assert added_count == 4

    stats = pool.get_approved_stats()
    assert stats["total_approved_images"] == 4
    assert stats["total_boxes"] == 4

    for i in range(4):
        assert pool.is_approved(f"batch_frame_{i:02d}") is True
    for i in range(4, 6):
        assert pool.is_approved(f"batch_frame_{i:02d}") is False

    # Pending queue must now have exactly 2 frames left
    summary_after = sorter.summarize_queue(queue)
    assert summary_after["approved_frames"] == 4
    assert summary_after["pending_frames"] == 2

    pending_stems = [it["stem"] for it in sorter.filter_pending(queue)]
    assert pending_stems == ["batch_frame_04", "batch_frame_05"]


def test_yolo_factory_dialog_step2_approve_marks_done_ok_and_decrements_pending_queue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """5. Verify YOLOFactoryDialog Step 2 approval marks frame done='ok' & human_approved=True, saves to ApprovedDatasetPool, and removes it from the pending queue.

    F-1492: the old test built the dialog with ``__new__`` and stubbed ~12
    methods — including the core flow (_learn_box_exemplar,
    _update_pending_labels, _mark_model_stale) — so it skipped the real
    Step-2 flow. Only pure Qt-rendering methods are stubbed now
    (_append_log, _render_*, _update_blockers/_update_next_card,
    _mark_step_done, _refresh_stepper_visuals); the approve path itself
    (_review_done -> _learn_box_exemplar, _estimate_box_kinah,
    _persist_review_item, _remove_review_item_at, _sync_pending_from_queue,
    _pending_count, _reload_review_queue, _mark_model_stale) runs for real.
    Images are real decodable JPEGs so exemplar learning really executes.
    """
    import cv2
    import numpy as np

    import dataset_factory.approved_pool as approved_pool_mod
    import dataset_factory.exemplar_memory as exemplar_memory_mod
    from dataset_factory.exemplar_memory import ExemplarMemory
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "review" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)

    pool = ApprovedDatasetPool(project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool)

    # Exemplar learning must land in the tmp dataset, not the repo's.
    tmp_memory = ExemplarMemory(json_path=tmp_path / "dataset" / "learned_exemplars.json")
    monkeypatch.setattr(
        exemplar_memory_mod, "get_exemplar_memory", lambda json_path=None: tmp_memory
    )

    dlg = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: dataset_dir  # type: ignore[assignment]
    dlg._pipeline = {"step": 2, "pending": 0, "reviewed": 0}
    dlg._review_decode_cache = {}
    # P1-12: _update_pending_labels reads the bulk-approve slider
    # (bulk_range) via _bulk_threshold(); a __new__ double has no Qt
    # widgets, so provide the slider the real method reads (90 reproduces
    # the pre-P1-12 0.90 hi-count). The method itself still runs for real
    # (F-1492) — only the widget is faked.
    from types import SimpleNamespace

    dlg.bulk_range = SimpleNamespace(value=lambda: 90)  # type: ignore[attr-defined]
    # ONLY pure UI-rendering methods are stubbed: they paint Qt widgets that
    # do not exist on a __new__ instance. Everything in the approve flow
    # itself is the real implementation.
    dlg._append_log = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._render_queue_list = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._render_review_frame = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._update_blockers = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._update_next_card = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._mark_step_done = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._refresh_stepper_visuals = lambda *a, **kw: None  # type: ignore[assignment]

    items: List[Dict[str, Any]] = []
    for idx, stem in enumerate(("ui_frame_01", "ui_frame_02")):
        img_p = dataset_dir / "images" / "train" / f"{stem}.jpg"
        # Real decodable JPEG (solid BGR crop) so _learn_box_exemplar can
        # actually decode, crop and record a positive exemplar.
        assert cv2.imwrite(
            str(img_p), np.full((120, 90, 3), (40 + 60 * idx, 160, 210), dtype=np.uint8)
        )
        rev_p = dataset_dir / "review" / "train" / f"{stem}.json"
        rev_p.write_text("{}", encoding="utf-8")
        lbl_p = dataset_dir / "labels" / "train" / f"{stem}.txt"
        lbl_p.write_text(f"{idx} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        items.append(
            {
                "stem": stem,
                "split": "train",
                "weak": [
                    {
                        "class_id": idx,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.2,
                        "h": 0.2,
                        "confidence": 0.72,
                    }
                ],
                "trusted": [],
                "done": None,
                "human_approved": False,
                "path": str(rev_p),
            }
        )

    approved_target = items[0]
    dlg._review_items = list(items)
    dlg._review_index = 0
    # B-03: per-box promotion / exemplar learning on approve happens only
    # for boxes the human inspected on the canvas. Without a canvas the
    # weak boxes stay weak and a weak-only frame cannot be approved
    # (B-02: zero trusted boxes). Attach the canvas the new contract
    # requires, holding the same box dicts as the item's weak list.
    from types import SimpleNamespace

    dlg.review_preview = SimpleNamespace(  # type: ignore[attr-defined]
        _annotations=list(items[0]["weak"]), update=lambda *a, **kw: None
    )
    dlg._sync_pending_from_queue()

    assert dlg._pending_count() == 2
    assert dlg._pipeline["pending"] == 2

    # Trigger Step 2 approve on ui_frame_01
    dlg._on_step2_approve()

    # Verify ui_frame_01 was marked done="ok" and human_approved=True, saved to ApprovedDatasetPool, and removed from queue
    assert approved_target["done"] == "ok"
    assert approved_target["human_approved"] is True
    assert len(approved_target["weak"]) == 0
    assert len(approved_target["trusted"]) == 1
    assert [it["stem"] for it in dlg._review_items] == ["ui_frame_02"]
    assert dlg._pending_count() == 1
    assert dlg._pipeline["pending"] == 1

    assert pool.is_approved("ui_frame_01") is True
    assert pool.is_approved("ui_frame_02") is False
    assert pool.get_approved_stats()["total_approved_images"] == 1

    # The real _learn_box_exemplar must have recorded a positive exemplar
    # for the approved box in the tmp exemplar memory (verified through the
    # public match path, not by reading private lists).
    reloaded_mem = ExemplarMemory(json_path=tmp_path / "dataset" / "learned_exemplars.json")
    probe_crop = np.full((120, 90, 3), (40, 160, 210), dtype=np.uint8)
    sim, cid, cname, _ex = reloaded_mem.match_positive(probe_crop)
    assert sim >= 0.99
    assert cid == 0

    # Reload review queue from disk: ui_frame_01 must not come back into the pending queue
    dlg._reload_review_queue()
    assert [it["stem"] for it in dlg._review_items] == ["ui_frame_02"]
    assert dlg._pending_count() == 1


def test_step2_approve_without_canvas_leaves_weak_only_frame_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """6. B-02/B-03 lock: approving a weak-only frame WITHOUT canvas
    inspection does not approve it — no bulk promotion, no exemplar
    teaching, and persist refuses a zero-trusted approval, so the frame
    stays pending (done=None) instead of entering the pool."""
    import cv2
    import numpy as np

    import dataset_factory.approved_pool as approved_pool_mod
    import dataset_factory.exemplar_memory as exemplar_memory_mod
    from dataset_factory.exemplar_memory import ExemplarMemory
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "review" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)

    pool = ApprovedDatasetPool(project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool)
    tmp_memory = ExemplarMemory(json_path=tmp_path / "dataset" / "learned_exemplars.json")
    monkeypatch.setattr(
        exemplar_memory_mod, "get_exemplar_memory", lambda json_path=None: tmp_memory
    )

    dlg = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: dataset_dir  # type: ignore[assignment]
    dlg._pipeline = {"step": 2, "pending": 0, "reviewed": 0}
    dlg._review_decode_cache = {}
    # P1-12: _update_pending_labels reads the bulk-approve slider
    # (bulk_range) via _bulk_threshold(); a __new__ double has no Qt
    # widgets, so provide the slider the real method reads (90 reproduces
    # the pre-P1-12 0.90 hi-count). The method itself still runs for real
    # (F-1492) — only the widget is faked.
    from types import SimpleNamespace

    dlg.bulk_range = SimpleNamespace(value=lambda: 90)  # type: ignore[attr-defined]
    dlg._append_log = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._render_queue_list = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._render_review_frame = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._update_blockers = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._update_next_card = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._mark_step_done = lambda *a, **kw: None  # type: ignore[assignment]
    dlg._refresh_stepper_visuals = lambda *a, **kw: None  # type: ignore[assignment]

    stem = "weak_only_frame"
    img_p = dataset_dir / "images" / "train" / f"{stem}.jpg"
    assert cv2.imwrite(str(img_p), np.full((120, 90, 3), (40, 160, 210), dtype=np.uint8))
    rev_p = dataset_dir / "review" / "train" / f"{stem}.json"
    rev_p.write_text("{}", encoding="utf-8")
    (dataset_dir / "labels" / "train" / f"{stem}.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    item: Dict[str, Any] = {
        "stem": stem,
        "split": "train",
        "weak": [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.72}
        ],
        "trusted": [],
        "done": None,
        "human_approved": False,
        "path": str(rev_p),
    }
    dlg._review_items = [item]
    dlg._review_index = 0
    # NOTE: no review_preview attached — the canvas-less branch.
    dlg._sync_pending_from_queue()
    assert dlg._pending_count() == 1

    dlg._on_step2_approve()

    assert item["done"] is None
    assert item["human_approved"] is False
    assert len(item["weak"]) == 1
    assert len(item["trusted"]) == 0
    assert [it["stem"] for it in dlg._review_items] == [stem]
    assert dlg._pending_count() == 1
    assert pool.is_approved(stem) is False

