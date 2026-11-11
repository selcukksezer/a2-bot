# -*- coding: utf-8 -*-
"""
W43 Unit Tests: Incremental Label Cache Engine (`dataset_factory/label_cache.py`)
Verifies that unchanged files are skipped and only modified files or new
`ExemplarMemory` revisions are processed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pytest

from dataset_factory.exemplar_memory import ExemplarMemory, extract_crop_features
from dataset_factory.label_cache import (
    LabelCache,
    get_current_exemplar_revision,
    plan_batch_relabel,
)


def _create_dummy_image(path: Path, color: tuple[int, int, int] = (40, 180, 90), size: int = 64) -> np.ndarray:
    """Create and write a synthetic BGR image to disk."""
    img = np.full((size, size, 3), color, dtype=np.uint8)
    # Add simple pattern so edge/texture descriptors are non-zero
    cv2.rectangle(img, (8, 8), (size - 8, size - 8), (255, 200, 50), 2)
    cv2.line(img, (0, 0), (size - 1, size - 1), (10, 20, 240), 2)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())
    return img


def test_skip_unchanged_files_when_stat_and_revision_match(tmp_path: Path) -> None:
    """
    Test 1: Unchanged files with positive box count, matching disk stat (mtime_ns, size),
    and matching exemplar_revision are skipped by both should_relabel and plan_batch_relabel.
    """
    cache_file = tmp_path / ".label_cache.json"
    cache = LabelCache(cache_path=cache_file, dataset_dir=tmp_path)

    img_a = tmp_path / "frame_a.png"
    img_b = tmp_path / "frame_b.png"
    _create_dummy_image(img_a, color=(30, 120, 200))
    _create_dummy_image(img_b, color=(80, 60, 150))

    rev = 1001
    cache.record_label(img_a, exemplar_revision=rev, box_count=2)
    cache.record_label(img_b, exemplar_revision=rev, box_count=1)

    assert cache.should_relabel(img_a, exemplar_revision=rev) is False
    assert cache.should_relabel(img_b, exemplar_revision=rev) is False

    plan = plan_batch_relabel([img_a, img_b], exemplar_revision=rev, label_cache=cache)
    assert plan["skip"] == [img_a, img_b]
    assert plan["delta"] == []
    assert plan["full"] == []


def test_relabel_only_modified_or_new_files_in_batch(tmp_path: Path) -> None:
    """
    Test 2a: A file whose mtime changes -- with identical bytes and size --
    is scheduled for relabeling while unchanged files are skipped.
    (mtime signal isolated: content and size are untouched.)
    """
    cache_file = tmp_path / ".label_cache.json"
    cache = LabelCache(cache_path=cache_file, dataset_dir=tmp_path)

    img_unchanged = tmp_path / "unchanged.png"
    img_modified = tmp_path / "modified.png"
    img_new = tmp_path / "new_frame.png"

    _create_dummy_image(img_unchanged, color=(50, 100, 150), size=64)
    _create_dummy_image(img_modified, color=(50, 100, 150), size=64)
    _create_dummy_image(img_new, color=(120, 200, 40), size=64)

    rev = 2002
    cache.record_label(img_unchanged, exemplar_revision=rev, box_count=3)
    cache.record_label(img_modified, exemplar_revision=rev, box_count=2)

    # Touch mtime only: same bytes, same size, only the timestamp moves.
    size_before = img_modified.stat().st_size
    st = img_modified.stat()
    os.utime(img_modified, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000))
    assert img_modified.stat().st_size == size_before

    assert cache.should_relabel(img_unchanged, exemplar_revision=rev) is False
    assert cache.should_relabel(img_modified, exemplar_revision=rev) is True
    assert cache.should_relabel(img_new, exemplar_revision=rev) is True

    plan = plan_batch_relabel(
        [img_unchanged, img_modified, img_new],
        exemplar_revision=rev,
        label_cache=cache,
    )
    assert plan["skip"] == [img_unchanged]
    assert plan["full"] == [img_modified, img_new]
    assert plan["delta"] == []


def test_relabel_when_only_file_size_changes(tmp_path: Path) -> None:
    """
    Test 2b: A file whose size changes -- with the mtime restored to the
    cached value -- is scheduled for relabeling while unchanged files are skipped.
    (size signal isolated: the timestamp is pinned back to the recorded one.)
    """
    cache_file = tmp_path / ".label_cache.json"
    cache = LabelCache(cache_path=cache_file, dataset_dir=tmp_path)

    img_unchanged = tmp_path / "unchanged.png"
    img_modified = tmp_path / "modified.png"

    _create_dummy_image(img_unchanged, color=(50, 100, 150), size=64)
    _create_dummy_image(img_modified, color=(50, 100, 150), size=64)

    rev = 2003
    cache.record_label(img_unchanged, exemplar_revision=rev, box_count=3)
    cache.record_label(img_modified, exemplar_revision=rev, box_count=2)

    # Rewrite with different bytes/size, then restore the original mtime so
    # only the size signal differs from the cache entry.
    st_before = img_modified.stat()
    _create_dummy_image(img_modified, color=(200, 40, 90), size=96)
    assert img_modified.stat().st_size != st_before.st_size
    os.utime(img_modified, ns=(st_before.st_atime_ns, st_before.st_mtime_ns))
    assert img_modified.stat().st_mtime_ns == st_before.st_mtime_ns

    assert cache.should_relabel(img_unchanged, exemplar_revision=rev) is False
    assert cache.should_relabel(img_modified, exemplar_revision=rev) is True

    plan = plan_batch_relabel(
        [img_unchanged, img_modified],
        exemplar_revision=rev,
        label_cache=cache,
    )
    assert plan["skip"] == [img_unchanged]
    assert plan["full"] == [img_modified]
    assert plan["delta"] == []


def test_new_exemplar_memory_revision_triggers_delta_and_full_relabel(tmp_path: Path) -> None:
    """
    Test 3: When ExemplarMemory is updated with a new exemplar (advancing the revision
    timestamp on disk), previously cached unchanged files are triggered for re-evaluation:
    files with cached candidate crops route to 'delta', and files without candidate crops
    route to 'full'.
    """
    exemplar_json = tmp_path / "learned_exemplars.json"
    cache_file = tmp_path / ".label_cache.json"
    memory = ExemplarMemory(json_path=exemplar_json)
    cache = LabelCache(cache_path=cache_file, dataset_dir=tmp_path)

    # Initial exemplar save to establish revision 1
    crop_init = _create_dummy_image(tmp_path / "crop_init.png", color=(20, 200, 20), size=32)
    assert memory.add_positive_exemplar(crop_init, class_id=0, class_name="wood", source="ui_approval")
    rev_1 = get_current_exemplar_revision(exemplar_path=exemplar_json)
    assert rev_1 > 0

    img_with_crops = tmp_path / "with_crops.png"
    img_no_crops = tmp_path / "no_crops.png"
    frame_crops = _create_dummy_image(img_with_crops, color=(40, 180, 40), size=64)
    _create_dummy_image(img_no_crops, color=(90, 90, 200), size=64)

    cache.cache_candidate_crops(
        img_with_crops,
        crops=[{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.5, "h": 0.5, "confidence": 0.75}],
        frame=frame_crops,
        exemplar_revision=rev_1,
    )
    cache.record_label(img_no_crops, exemplar_revision=rev_1, box_count=1)

    # At rev_1, only the really labelled file is skipped. P2-9 (K-06):
    # cache_candidate_crops starts a fresh entry at box_count=0
    # (candidate crops are proposals, not labelled boxes), so the
    # 0-box guarantee keeps img_with_crops out of "skip" — it routes
    # to "full" (same revision, so no delta applies).
    assert cache.should_relabel(img_with_crops, exemplar_revision=rev_1) is True
    assert cache.should_relabel(img_no_crops, exemplar_revision=rev_1) is False
    plan_initial = plan_batch_relabel([img_with_crops, img_no_crops], exemplar_revision=rev_1, label_cache=cache)
    assert plan_initial["skip"] == [img_no_crops]
    assert plan_initial["delta"] == []
    assert plan_initial["full"] == [img_with_crops]

    # Advance ExemplarMemory revision by adding a new exemplar. The revision is
    # the exemplar file's mtime_ns, so a real save must advance it on its own --
    # no hand-forged timestamp is applied.
    crop_new = _create_dummy_image(tmp_path / "crop_new.png", color=(200, 50, 200), size=32)
    assert memory.add_positive_exemplar(crop_new, class_id=1, class_name="ore", source="ui_approval")
    rev_2 = get_current_exemplar_revision(exemplar_path=exemplar_json)
    assert rev_2 > rev_1

    assert cache.should_relabel(img_with_crops, exemplar_revision=rev_2) is True
    assert cache.can_delta_relabel(img_with_crops, exemplar_revision=rev_2) is True
    assert cache.can_delta_relabel(img_no_crops, exemplar_revision=rev_2) is False

    plan_updated = plan_batch_relabel([img_with_crops, img_no_crops], exemplar_revision=rev_2, label_cache=cache)
    assert plan_updated["skip"] == []
    assert plan_updated["delta"] == [img_with_crops]
    assert plan_updated["full"] == [img_no_crops]


def test_zero_box_unlabeled_images_never_skipped(tmp_path: Path) -> None:
    """
    Test 4: Zero-Box Safety Guarantee - images recorded with box_count == 0 are NEVER
    skipped by should_relabel or plan_batch_relabel even when file stat and exemplar_revision
    match the cache.
    """
    cache_file = tmp_path / ".label_cache.json"
    cache = LabelCache(cache_path=cache_file, dataset_dir=tmp_path)

    img_zero = tmp_path / "unlabeled_zero.png"
    img_labeled = tmp_path / "labeled_positive.png"
    _create_dummy_image(img_zero, color=(100, 100, 100))
    _create_dummy_image(img_labeled, color=(100, 200, 100))

    rev = 3003
    cache.record_label(img_zero, exemplar_revision=rev, box_count=0)
    cache.record_label(img_labeled, exemplar_revision=rev, box_count=2)

    assert cache.should_relabel(img_zero, exemplar_revision=rev) is True
    assert cache.should_relabel(img_labeled, exemplar_revision=rev) is False

    plan = plan_batch_relabel([img_zero, img_labeled], exemplar_revision=rev, label_cache=cache)
    assert plan["skip"] == [img_labeled]
    assert plan["full"] == [img_zero]


def test_incremental_delta_re_evaluation_with_updated_exemplar_memory(tmp_path: Path) -> None:
    """
    Test 5: re_evaluate_candidate_crops incrementally re-evaluates cached candidate crops
    against an updated ExemplarMemory (suppressing negative matches, reinforcing positive
    matches, preserving hand-labeled immunity, and updating the entry's exemplar_revision
    so subsequent runs at the new revision skip the file).
    """
    exemplar_json = tmp_path / "learned_exemplars.json"
    cache_file = tmp_path / ".label_cache.json"
    memory = ExemplarMemory(json_path=exemplar_json)
    cache = LabelCache(cache_path=cache_file, dataset_dir=tmp_path)

    img_path = tmp_path / "scene.png"
    frame = _create_dummy_image(img_path, color=(60, 140, 220), size=64)

    # Create two distinct synthetic patches for feature extraction
    pos_patch = _create_dummy_image(tmp_path / "pos_patch.png", color=(20, 220, 40), size=32)
    neg_patch = _create_dummy_image(tmp_path / "neg_patch.png", color=(230, 20, 20), size=32)

    pos_feat = extract_crop_features(pos_patch)
    neg_feat = extract_crop_features(neg_patch)
    assert pos_feat is not None and neg_feat is not None

    rev_initial = 5000
    candidates = [
        {
            "class_id": 0,
            "class_name": "wood",
            "cx": 0.25,
            "cy": 0.25,
            "w": 0.2,
            "h": 0.2,
            "confidence": 0.72,
            "feature": pos_feat,
        },
        {
            "class_id": 2,
            "class_name": "stone",
            "cx": 0.75,
            "cy": 0.75,
            "w": 0.2,
            "h": 0.2,
            "confidence": 0.65,
            "feature": neg_feat,
        },
        {
            "class_id": 5,
            "class_name": "hand_item",
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.2,
            "h": 0.2,
            "confidence": 0.50,
            "hand_label": True,
            "feature": neg_feat,  # Even with negative feature, hand_label immunity must protect it
        },
    ]
    cache.cache_candidate_crops(img_path, crops=candidates, exemplar_revision=rev_initial)

    # Teach ExemplarMemory: pos_patch is positive for class 0, neg_patch is negative for class 2
    assert memory.add_positive_exemplar(pos_patch, class_id=0, class_name="wood", source="ui_approval")
    assert memory.add_negative_exemplar(neg_patch, class_id=2, class_name="stone", source="ui_rejection")

    rev_updated = 6000
    assert cache.can_delta_relabel(img_path, exemplar_revision=rev_updated) is True

    surviving = cache.re_evaluate_candidate_crops(
        img_path,
        exemplar_memory=memory,
        exemplar_revision=rev_updated,
    )

    # Negative candidate (class_id=2) must be suppressed; positive (0) and hand_label (5) survive.
    # The positive candidate is stored as (0, "wood") but the exemplar was
    # canonicalized to (5, "tree_log") when added, so re_evaluate corrects
    # the candidate cid 0 -> 5 (exemplar_corrected=True, sim 1.0 >= 0.78
    # reassign threshold) instead of reinforcing it in place.
    surviving_cids = [c["class_id"] for c in surviving]
    assert surviving_cids == [5, 5]

    pos_out = next(c for c in surviving if c.get("exemplar_corrected") is True)
    hand_out = next(c for c in surviving if c["class_id"] == 5 and c.get("hand_label") is True)
    assert pos_out["class_name"] == "tree_log"
    assert pos_out.get("exemplar_corrected") is True
    assert pos_out["confidence"] > 0.72
    assert hand_out["trusted"] is True
    assert hand_out["confidence"] >= 0.95

    # Cache entry should now reflect rev_updated and skip on subsequent checks for rev_updated
    entry = cache.get_entry(img_path)
    assert entry is not None
    assert entry.exemplar_revision == rev_updated
    assert entry.box_count == 2
    assert cache.should_relabel(img_path, exemplar_revision=rev_updated) is False
