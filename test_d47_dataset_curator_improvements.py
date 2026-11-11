# -*- coding: utf-8 -*-
"""
D47 tests for dataset_factory/dataset_curator.py improvements.

1. Vectorized class-aware NMS: the new NumPy greedy NMS must return exactly
   the same kept boxes (same order) as the previous O(n^2) Python loop,
   verified against a brute-force reference copy of the old algorithm.
2. Incremental manifest refresh: save_labeled_sample(update_manifest=True)
   must produce a manifest_stats.json identical (modulo timestamps) to a
   full rebuild_manifest_stats(), including the fallback paths (overwrite
   of an existing stem, external in-place label edit, external file add).

Temp dirs live under ~/workspace (never /tmp: the 512M tmpfs is full).
"""

from __future__ import annotations

import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from dataset_factory.dataset_curator import DatasetCurator


# ---------------------------------------------------------------------------
# Reference: the pre-D47 O(n^2) NMS loop, copied verbatim in logic.
# ---------------------------------------------------------------------------
def _reference_nms(annotations, iou_threshold=0.65):
    if len(annotations) <= 1:
        return list(annotations)
    by_class = {}
    for ann in annotations:
        by_class.setdefault(int(ann["class_id"]), []).append(ann)
    kept = []
    for cid in sorted(by_class.keys()):
        candidates = sorted(
            by_class[cid],
            key=lambda a: (float(a.get("confidence", 0.9)), float(a["w"] * a["h"])),
            reverse=True,
        )
        kept_cls = []
        for cand in candidates:
            ax1 = cand["cx"] - 0.5 * cand["w"]
            ay1 = cand["cy"] - 0.5 * cand["h"]
            ax2 = cand["cx"] + 0.5 * cand["w"]
            ay2 = cand["cy"] + 0.5 * cand["h"]
            area_a = max(1e-9, (ax2 - ax1) * (ay2 - ay1))
            is_dup = False
            for ex in kept_cls:
                bx1 = ex["cx"] - 0.5 * ex["w"]
                by1 = ex["cy"] - 0.5 * ex["h"]
                bx2 = ex["cx"] + 0.5 * ex["w"]
                by2 = ex["cy"] + 0.5 * ex["h"]
                area_b = max(1e-9, (bx2 - bx1) * (by2 - by1))
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
                iou = inter / (area_a + area_b - inter + 1e-9)
                if iou > iou_threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept_cls.append(cand)
        kept.extend(kept_cls)
    return kept


def _sig(anns):
    return [
        (int(a["class_id"]), round(float(a["cx"]), 6), round(float(a["cy"]), 6),
         round(float(a["w"]), 6), round(float(a["h"]), 6),
         round(float(a.get("confidence", 0.9)), 4))
        for a in anns
    ]


def _rand_boxes(rng, n, nclass):
    boxes = []
    for _ in range(n):
        boxes.append({
            "class_id": rng.randrange(nclass),
            "class_name": "bench",
            "cx": round(rng.random(), 6),
            "cy": round(rng.random(), 6),
            "w": round(rng.uniform(0.02, 0.30), 6),
            "h": round(rng.uniform(0.02, 0.30), 6),
            "confidence": round(rng.random(), 4),
            "source": "bench",
        })
    return boxes


@pytest.mark.parametrize("seed,n,nclass,thr", [
    (1, 0, 1, 0.65),
    (2, 1, 1, 0.65),
    (3, 2, 1, 0.65),
    (4, 50, 1, 0.65),
    (5, 300, 1, 0.65),
    (6, 300, 3, 0.65),
    (7, 200, 2, 0.50),
    (8, 200, 2, 0.90),
    (9, 120, 5, 0.65),
    (10, 900, 1, 0.65),  # exercises the >800 row-wise fallback path
])
def test_vectorized_nms_matches_reference(seed, n, nclass, thr):
    rng = random.Random(seed)
    boxes = _rand_boxes(rng, n, nclass)
    got = DatasetCurator._apply_class_aware_nms([dict(b) for b in boxes], iou_threshold=thr)
    want = _reference_nms([dict(b) for b in boxes], iou_threshold=thr)
    assert _sig(got) == _sig(want)


def test_vectorized_nms_exact_duplicates_and_ties():
    base = {"class_id": 0, "class_name": "herb_aria", "cx": 0.5, "cy": 0.5,
            "w": 0.2, "h": 0.2, "confidence": 0.9, "source": "bench"}
    # Five identical boxes -> exactly one survives; equal conf+area ties
    # must keep the first in input order.
    boxes = [dict(base) for _ in range(5)]
    boxes[0]["source"] = "first"
    got = DatasetCurator._apply_class_aware_nms([dict(b) for b in boxes])
    assert len(got) == 1
    assert got[0]["source"] == "first"
    assert _sig(got) == _sig(_reference_nms([dict(b) for b in boxes]))

    # Same geometry, different confidence -> highest confidence wins.
    boxes = [dict(base, confidence=c) for c in (0.5, 0.99, 0.7)]
    got = DatasetCurator._apply_class_aware_nms([dict(b) for b in boxes])
    assert len(got) == 1
    assert got[0]["confidence"] == 0.99


def test_vectorized_nms_does_not_suppress_other_classes():
    boxes = [
        {"class_id": 0, "class_name": "a", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.9, "source": "bench"},
        {"class_id": 1, "class_name": "b", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2,
         "confidence": 0.9, "source": "bench"},
    ]
    got = DatasetCurator._apply_class_aware_nms([dict(b) for b in boxes])
    assert len(got) == 2


# ---------------------------------------------------------------------------
# Incremental manifest refresh
# ---------------------------------------------------------------------------
@pytest.fixture
def curator_tmp():
    base = Path(tempfile.mkdtemp(prefix="d47_test_", dir=str(Path.home() / "workspace")))
    try:
        yield base
    finally:
        shutil.rmtree(base, ignore_errors=True)


def _manifest_without_timestamps(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    data.pop("updated_at_epoch", None)
    data.pop("updated_at_iso", None)
    return data


def _frame():
    return np.zeros((64, 64, 3), dtype=np.uint8)


def _anns(cid=0, conf=0.95, hand=True):
    ann = {"class_id": cid, "class_name": "herb_aria", "cx": 0.5, "cy": 0.5,
           "w": 0.1, "h": 0.1, "confidence": conf, "source": "hand_label"}
    if hand:
        ann["hand_label"] = True
    return [ann]


def test_incremental_manifest_matches_full_rebuild(curator_tmp):
    cur = DatasetCurator(dataset_dir=curator_tmp / "dataset")
    # Seed + full rebuild populates the incremental cache.
    for i in range(5):
        cur.save_labeled_sample(_frame(), _anns(), split="train",
                                sample_name=f"seed_{i}", save_preview=False,
                                update_manifest=False)
    cur.rebuild_manifest_stats()

    # Live-save path (update_manifest=True): train/val mix, previews on/off,
    # multi-class, multi-box and weak boxes.
    saves = [
        ("live_00", "train", _anns(0), True),
        ("live_01", "val", _anns(1), False),
        ("live_02", "train", _anns(0) + _anns(3, 0.8), True),
        ("live_03", "train", _anns(2), False),
        ("live_04", "val", _anns(0), True),
        ("live_05", "train", _anns(5, 0.5, hand=False), False),
        ("live_06", "train", _anns(0), True),
        ("live_07", "val", _anns(4), True),
        ("live_08", "train", _anns(1) + _anns(1, 0.9), False),
        ("live_09", "train", _anns(26), True),
    ]
    for stem, split, anns, preview in saves:
        cur.save_labeled_sample(_frame(), anns, split=split, sample_name=stem,
                                save_preview=preview, update_manifest=True)
        incremental = _manifest_without_timestamps(cur.manifest_stats_path)
        rebuilt = _manifest_without_timestamps(
            Path(cur.rebuild_manifest_stats()["dataset_dir"]) / "manifest_stats.json"
        )
        # rebuild_manifest_stats() rewrites the file; compare the incremental
        # snapshot taken before it against the rebuilt ground truth.
        assert incremental == rebuilt, f"manifest drift after saving {stem}"


def test_incremental_manifest_fallback_on_overwrite(curator_tmp):
    cur = DatasetCurator(dataset_dir=curator_tmp / "dataset")
    cur.save_labeled_sample(_frame(), _anns(0), split="train", sample_name="dup",
                            save_preview=False, update_manifest=True)
    # Same stem again: file overwrite -> must fall back to full rebuild.
    # (commit_split_labels merges with the previous label, so the old
    # class-0 box is kept and the new class-1 box is added: 1 image, 2 boxes.)
    cur.save_labeled_sample(_frame(), _anns(1), split="train", sample_name="dup",
                            save_preview=False, update_manifest=True)
    incremental = _manifest_without_timestamps(cur.manifest_stats_path)
    rebuilt = _manifest_without_timestamps(
        Path(cur.rebuild_manifest_stats()["dataset_dir"]) / "manifest_stats.json"
    )
    assert incremental == rebuilt
    assert rebuilt["total_images"] == 1
    assert rebuilt["total_boxes"] == 2


def test_incremental_manifest_fallback_on_external_change(curator_tmp):
    cur = DatasetCurator(dataset_dir=curator_tmp / "dataset")
    for i in range(3):
        cur.save_labeled_sample(_frame(), _anns(0), split="train",
                                sample_name=f"s_{i}", save_preview=False,
                                update_manifest=True)
    # External in-place label edit (same file count, changed content).
    lbl = cur.labels_train_dir / "s_0.txt"
    lbl.write_text(lbl.read_text(encoding="utf-8") + "3 0.5 0.5 0.1 0.1\n",
                   encoding="utf-8")
    cur.save_labeled_sample(_frame(), _anns(0), split="train", sample_name="s_3",
                            save_preview=False, update_manifest=True)
    incremental = _manifest_without_timestamps(cur.manifest_stats_path)
    rebuilt = _manifest_without_timestamps(
        Path(cur.rebuild_manifest_stats()["dataset_dir"]) / "manifest_stats.json"
    )
    assert incremental == rebuilt
    assert rebuilt["total_boxes"] == 5  # 3 original + 1 appended + 1 new

    # External add of an image+label pair directly on disk.
    from dataset_factory.dataset_curator import imwrite_unicode
    imwrite_unicode(cur.images_train_dir / "external.jpg", _frame())
    (cur.labels_train_dir / "external.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    cur.save_labeled_sample(_frame(), _anns(0), split="train", sample_name="s_4",
                            save_preview=False, update_manifest=True)
    incremental = _manifest_without_timestamps(cur.manifest_stats_path)
    rebuilt = _manifest_without_timestamps(
        Path(cur.rebuild_manifest_stats()["dataset_dir"]) / "manifest_stats.json"
    )
    assert incremental == rebuilt
    assert rebuilt["total_images"] == 6
