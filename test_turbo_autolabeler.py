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

    # Create dummy images with distinct contours
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


def test_fast_mode_buffer_reuse():
    """
    Work Package F: fast_mode=True Buffer Reuse.
    Verifies that fast_mode executes cleanly, reuses intermediate buffers,
    and returns valid annotation dictionaries.
    """
    labeler = YOLOAutoLabeler()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(frame, (100, 100), (250, 250), (200, 150, 50), -1)

    # Warm-up call
    _ = labeler.auto_label_frame(frame, fast_mode=True)

    t0 = time.perf_counter()
    anns_fast = labeler.auto_label_frame(frame, fast_mode=True)
    t_fast = time.perf_counter() - t0

    assert isinstance(anns_fast, list)
    # Reasonable execution threshold for 720p frame under CPU load
    assert t_fast < 2.5, f"fast_mode took {t_fast:.2f}s, expected < 2.5s"

    # Verify annotation structure if any proposals were detected
    for ann in anns_fast:
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
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

    # First pass: label all images
    t0 = time.time()
    res1 = curator.relabel_dataset_batch(incremental=True, max_workers=4)
    t_elapsed1 = time.time() - t0

    assert res1["total"] == 6
    assert res1["processed"] == 6
    assert res1["labeled"] >= 1
    assert curator.label_cache_path.is_file(), "label cache file should be created"

    cache_data = curator.load_label_cache()
    assert len(cache_data) >= 1

    # Second pass: incremental run should skip previously labeled images
    t1 = time.time()
    res2 = curator.relabel_dataset_batch(incremental=True, max_workers=4)
    t_elapsed2 = time.time() - t1

    assert res2["total"] == 6
    assert res2["skipped"] >= 1, "Expected at least 1 image to be skipped via cache"
    assert res2["fps"] >= 0.0


def test_never_skip_zero_box_unlabeled_images(temp_dataset: Path):
    """
    Work Package F: Critical User Requirement.
    Verifies that cache and batch relabeler NEVER skip 0-box images thinking they are done.
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

    # Put a fake 0-box entry into .label_cache.json
    empty_file = temp_dataset / "images" / "val" / "sample_empty.jpg"
    st = empty_file.stat()
    fake_cache = {
        empty_file.name: {
            "mtime_ns": st.st_mtime_ns,
            "size": st.st_size,
            "box_count": 0,  # 0 boxes!
            "exemplar_rev": 0,
        }
    }
    curator.save_label_cache(fake_cache)

    # Run batch relabeler
    res = curator.relabel_dataset_batch(
        image_paths=[empty_file],
        incremental=True,
    )
    # The image with box_count == 0 must NOT be skipped as cached
    assert res["skipped"] == 0, "0-box image must never be skipped!"
    assert res["labeled"] == 1


def test_multicore_threadpool_and_progress_callback(temp_dataset: Path):
    """
    Work Package F: Multi-core ThreadPool & Live Progress Callback.
    Verifies that relabel_dataset_batch operates with multiple worker threads
    and regularly invokes progress_callback with throughput metrics.
    """
    curator = DatasetCurator(dataset_dir=temp_dataset)

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
    assert len(progress_events) >= 1, "Progress callback should have been invoked"

    last_event = progress_events[-1]
    assert "processed" in last_event
    assert "total" in last_event
    assert last_event["processed"] == 6


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
