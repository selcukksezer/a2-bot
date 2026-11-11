# -*- coding: utf-8 -*-
"""
Tests for Work Package F: Turbo Auto-Labeling Engine for 50k Scale.

Validates:
1. Buffer reuse & fast_mode execution in YOLOAutoLabeler.
2. Multi-core ThreadPool batch acceleration in relabel_dataset_batch.
3. Incremental caching in .label_cache.json (mtime_ns, size, exemplar_rev).
4. Zero-box unlabeled images are NEVER skipped.
5. Live progress callback emissions and throughput metrics.
6. Cache invalidation on file modification or force_all=True.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

from dataset_factory.dataset_curator import DatasetCurator, relabel_dataset_batch
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler


@pytest.fixture
def temp_dataset():
    """Create a temporary dataset workspace with train and val images."""
    temp_dir = Path(tempfile.mkdtemp(prefix="gatherion_turbo_test_"))
    images_train = temp_dir / "images" / "train"
    images_val = temp_dir / "images" / "val"
    labels_train = temp_dir / "labels" / "train"
    labels_val = temp_dir / "labels" / "val"
    review_train = temp_dir / "review" / "train"
    review_val = temp_dir / "review" / "val"

    for d in (images_train, images_val, labels_train, labels_val, review_train, review_val):
        d.mkdir(parents=True, exist_ok=True)

    # Create dummy images with distinct contours. Frames stay 480x640:
    # the labeler's salient scan finds no boxes at 240x320/360x480 on
    # these images (verified by probe: total_boxes 0 below 480x640), which
    # would break the incremental-cache premise below (0-box images are
    # never skipped), so shrinking the fixture would change behaviour.
    for i in range(5):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add visual colored circles
        cv2.circle(img, (200 + i * 30, 240), 40, (0, 255, 200), -1)
        cv2.imwrite(str(images_train / f"sample_{i:03d}.jpg"), img)

    # One empty image
    empty_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.imwrite(str(images_val / "sample_empty.jpg"), empty_img)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def _intervals_overlap(calls: List[Dict[str, Any]]) -> bool:
    """True if any two recorded [entered, exited) intervals overlap.

    Calls are sorted by entry time; an entry before the latest exit seen
    so far means two labelings ran concurrently.
    """
    latest_exit = None
    for call in sorted(calls, key=lambda c: c["entered"]):
        if latest_exit is not None and call["entered"] < latest_exit:
            return True
        latest_exit = call["exited"] if latest_exit is None else max(latest_exit, call["exited"])
    return False


def test_fast_mode_buffer_reuse():
    """
    Work Package F: fast_mode=True Buffer Reuse.
    Proves reuse instead of timing it (F-1439: a duration threshold cannot
    tell reuse from a fast machine): the first call must populate the
    shared_frame_context cache (bgr/gray/hsv/edges), a second call with the
    same context must reuse those exact objects (identity, not recompute),
    and the reused-context result must equal a fresh-context result.
    """
    labeler = YOLOAutoLabeler()
    # 360x640 frame: same reuse proof as 720x1280 at ~half the per-call
    # cost (detection cost scales with pixels); the scaled rectangle still
    # yields >= 1 annotation, so the structure asserts below are not vacuous.
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 50), (125, 125), (200, 150, 50), -1)

    ctx: Dict[str, Any] = {}
    anns_first = labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=ctx)

    assert isinstance(anns_first, list)
    assert len(anns_first) >= 1, "fast_mode frame must yield >= 1 annotation for the reuse proof"
    for key in ("bgr", "gray", "hsv", "edges"):
        assert key in ctx, f"fast_mode should cache '{key}' in shared_frame_context"
        assert ctx[key] is not None
    gray_first, hsv_first, edges_first = ctx["gray"], ctx["hsv"], ctx["edges"]

    anns_second = labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context=ctx)
    assert ctx["gray"] is gray_first, "second call must reuse cached gray, not recompute it"
    assert ctx["hsv"] is hsv_first, "second call must reuse cached hsv, not recompute it"
    assert ctx["edges"] is edges_first, "second call must reuse cached edges, not recompute them"
    assert anns_second == anns_first

    # Reuse must not change the result: a fresh context labels identically.
    anns_fresh = labeler.auto_label_frame(frame, fast_mode=True, shared_frame_context={})
    assert anns_fresh == anns_first

    # Verify annotation structure if any proposals were detected
    for ann in anns_first:
        assert "class_id" in ann
        assert "cx" in ann and "cy" in ann
        assert "w" in ann and "h" in ann
        assert "confidence" in ann


def test_batch_relabel_and_incremental_cache(temp_dataset: Path):
    """
    Work Package F: ThreadPool Batch Acceleration & Incremental Caching.
    Verifies that:
    1. First batch run processes all images and populates .label_cache.json.
    2. Second batch run uses incremental caching and skips unchanged labeled frames.

    Cache-speedup proof (S-4002): wall-clock alone cannot prove caching
    (F-1439: a duration threshold cannot tell caching from a fast machine,
    and on 6 tiny images both runs are dominated by labeler start-up).
    The proof is therefore in work units, not time: the second run must
    skip exactly the cache entries that carry box_count > 0 and label
    only the rest, so the cache saves `skippable` relabel operations.
    The measured wall times are still used, but only to check that the
    reported elapsed_seconds is consistent with the real call duration
    (and strictly positive) -- the old `fps >= 0.0` assert was true for
    any value, including 0.
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

    # First pass: label all images
    t0 = time.time()
    res1 = curator.relabel_dataset_batch(incremental=True, max_workers=4)
    t_elapsed1 = time.time() - t0

    assert res1["total"] == 6
    assert res1["processed"] == 6
    assert res1["labeled"] == 6, "first run on a fresh cache must label every image"
    assert res1["skipped"] == 0
    assert res1["errors"] == 0
    assert curator.label_cache_path.is_file(), "label cache file should be created"

    cache_data = curator.load_label_cache()
    assert len(cache_data) == 6
    # Every cache entry must match the file on disk (mtime_ns/size).
    for key, entry in cache_data.items():
        cached_file = Path(key)
        assert cached_file.is_file(), f"cache key {key} must be a real image path"
        st = cached_file.stat()
        assert entry["mtime_ns"] == st.st_mtime_ns
        assert entry["size"] == st.st_size
    # Only entries with boxes can ever be skipped (0-box images are
    # never treated as done -- see test_never_skip_zero_box_unlabeled_images).
    skippable = sum(1 for e in cache_data.values() if e.get("box_count", 0) > 0)
    assert skippable >= 1, "expected at least one cached image with boxes"

    # Reported throughput must be real: strictly positive fps and an
    # elapsed_seconds consistent with the measured wall time.
    assert res1["fps"] > 0.0
    assert 0.0 < res1["elapsed_seconds"] <= t_elapsed1 + 0.05
    assert t_elapsed1 > 0.0

    # Second pass: incremental run should skip previously labeled images
    t1 = time.time()
    res2 = curator.relabel_dataset_batch(incremental=True, max_workers=4)
    t_elapsed2 = time.time() - t1

    assert res2["total"] == 6
    assert res2["processed"] == 6
    assert res2["errors"] == 0
    assert res2["skipped"] == skippable, (
        f"incremental run must skip exactly the {skippable} cached images with boxes"
    )
    assert res2["labeled"] == 6 - skippable
    assert res2["skipped"] + res2["labeled"] == res2["processed"]
    # The cache saved exactly `skippable` relabel operations.
    assert res1["labeled"] - res2["labeled"] == skippable
    assert res2["fps"] > 0.0
    assert 0.0 < res2["elapsed_seconds"] <= t_elapsed2 + 0.05
    assert t_elapsed2 > 0.0


def test_never_skip_zero_box_unlabeled_images(temp_dataset: Path):
    """
    Work Package F: Critical User Requirement.
    Verifies that cache and batch relabeler NEVER skip 0-box images thinking they are done.

    S-6801: the old test wrote the fake cache entry under the bare file name,
    but production looks it up by the full resolved path (cache_key =
    str(img_p.resolve()) in dataset_curator.py), and the skip branch also
    requires the review/label files to exist — so the old test passed
    vacuously without ever exercising the box_cnt > 0 guard. This version
    first proves the skip path is reachable (a box_count=1 entry under the
    resolved-path key, with review/label files present, IS skipped), then
    flips the entry to box_count=0 and requires the image to be labeled
    instead. If the box_cnt > 0 guard were removed, the second run would
    skip and this test would fail.
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

    empty_file = temp_dataset / "images" / "val" / "sample_empty.jpg"
    st = empty_file.stat()
    # Production cache key is the full resolved path, not the bare filename.
    cache_key = str(empty_file.resolve())

    # The skip branch requires the review + label files to exist.
    review_file = temp_dataset / "review" / "val" / "sample_empty.json"
    review_file.write_text(
        json.dumps({"stem": "sample_empty", "trusted": [], "weak": []}),
        encoding="utf-8",
    )
    label_file = temp_dataset / "labels" / "val" / "sample_empty.txt"
    label_file.write_text("", encoding="utf-8")

    def _fake_cache(box_count: int):
        return {
            cache_key: {
                "mtime_ns": st.st_mtime_ns,
                "size": st.st_size,
                "box_count": box_count,  # 0 boxes in the second run!
                "exemplar_rev": 0,
            }
        }

    # Control: a 1-box entry under the resolved-path key IS skipped —
    # proves the skip path (key + review/label files) is actually wired.
    curator.save_label_cache(_fake_cache(1))
    res_cached = curator.relabel_dataset_batch(
        image_paths=[empty_file],
        incremental=True,
    )
    assert res_cached["skipped"] == 1, "1-box cached image must be skipped as done"
    assert res_cached["labeled"] == 0

    # The image with box_count == 0 must NOT be skipped as cached:
    # the box_cnt > 0 guard forces a relabel.
    curator.save_label_cache(_fake_cache(0))
    res = curator.relabel_dataset_batch(
        image_paths=[empty_file],
        incremental=True,
    )
    assert res["skipped"] == 0, "0-box image must never be skipped!"
    assert res["labeled"] == 1


def test_multicore_threadpool_and_progress_callback(temp_dataset: Path, monkeypatch):
    """
    Work Package F: Multi-core ThreadPool & Live Progress Callback.
    Verifies that relabel_dataset_batch operates with multiple worker threads
    and regularly invokes progress_callback with throughput metrics.

    Parallelism proof (S-4001; same category as F-1439): the old asserts
    (total/processed/progress count) also pass with max_workers=1, so they
    prove nothing about threading. This test records, via a wrapper around
    YOLOAutoLabeler.auto_label_frame, the executing thread id, the
    thread-local labeler identity and the entry/exit timestamps of every
    labeling call, then runs the batch twice:
      * max_workers=4: at least 2 distinct worker threads, at least 2
        distinct thread-local labelers, and overlapping execution
        intervals (real concurrency, not just queued work).
      * max_workers=1 (negative control): exactly 1 worker thread, 1
        labeler, and no overlapping intervals -- the same asserts that
        pass for the parallel run must fail for serial execution.
    A short sleep inside the wrapper makes the overlap window
    deterministic instead of relying on scheduling luck. No wall-clock
    comparison is asserted: labeler start-up dominates both runs, so a
    duration threshold cannot tell parallel from serial here.
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

    calls: List[Dict[str, Any]] = []
    calls_lock = threading.Lock()
    orig_label = YOLOAutoLabeler.auto_label_frame

    def recording_label(self, frame, *args, **kwargs):
        ident = threading.get_ident()
        entered = time.monotonic()
        time.sleep(0.05)
        try:
            return orig_label(self, frame, *args, **kwargs)
        finally:
            exited = time.monotonic()
            with calls_lock:
                calls.append(
                    {
                        "thread_ident": ident,
                        "labeler_id": id(self),
                        "entered": entered,
                        "exited": exited,
                    }
                )

    monkeypatch.setattr(YOLOAutoLabeler, "auto_label_frame", recording_label)

    progress_events: List[Dict[str, Any]] = []

    def on_progress(p_info: Dict[str, Any]):
        progress_events.append(dict(p_info))

    res = curator.relabel_dataset_batch(
        max_workers=4,
        incremental=False,  # Force re-label to trigger full worker execution
        progress_callback=on_progress,
    )

    assert res["total"] == 6
    assert res["processed"] == 6
    assert res["errors"] == 0
    assert len(calls) == 6, "one labeling call per image expected"
    thread_idents = {c["thread_ident"] for c in calls}
    labeler_ids = {c["labeler_id"] for c in calls}
    assert len(thread_idents) >= 2, (
        f"max_workers=4 must use at least 2 worker threads, saw {len(thread_idents)}"
    )
    assert len(labeler_ids) >= 2, (
        "each worker thread must get its own thread-local labeler, "
        f"saw {len(labeler_ids)} distinct labeler(s)"
    )
    assert _intervals_overlap(calls), (
        "parallel run must have overlapping labeling intervals (real concurrency)"
    )

    # Exactly one progress event per completed image, processed 1..6 in order.
    assert [e["processed"] for e in progress_events] == [1, 2, 3, 4, 5, 6]
    assert all(e["total"] == 6 for e in progress_events)
    assert all(e.get("fps", 0) > 0 for e in progress_events)
    assert all("eta_s" in e and "stem" in e and "status" in e for e in progress_events)

    last_event = progress_events[-1]
    assert "processed" in last_event
    assert "total" in last_event
    assert last_event["processed"] == 6

    # Negative control: max_workers=1 must be serial -- the same
    # parallelism asserts above must NOT hold for this run.
    calls.clear()
    progress_events.clear()
    res_serial = curator.relabel_dataset_batch(
        max_workers=1,
        incremental=False,
        progress_callback=on_progress,
    )

    assert res_serial["total"] == 6
    assert res_serial["processed"] == 6
    assert res_serial["errors"] == 0
    assert len(calls) == 6, "one labeling call per image expected"
    assert len({c["thread_ident"] for c in calls}) == 1, (
        "serial run must use a single worker thread"
    )
    assert len({c["labeler_id"] for c in calls}) == 1, (
        "serial run must reuse one thread-local labeler"
    )
    assert not _intervals_overlap(calls), (
        "serial run must have no overlapping labeling intervals"
    )
    assert [e["processed"] for e in progress_events] == [1, 2, 3, 4, 5, 6]


def test_cache_invalidation_on_force_all(temp_dataset: Path):
    """
    Work Package F: Cache Invalidation via force_all=True.
    Verifies that passing force_all=True re-labels all target images,
    ignoring cached entries.
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

    # Initial run to build cache
    res1 = curator.relabel_dataset_batch(incremental=True, max_workers=2)
    assert res1["processed"] == 6

    # Force all run
    res2 = curator.relabel_dataset_batch(force_all=True, max_workers=2)
    assert res2["skipped"] == 0, "force_all=True should not skip any images!"
    assert res2["labeled"] >= 1
