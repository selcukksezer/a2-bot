# -*- coding: utf-8 -*-
"""
tests/test_label_cache_delta.py
===============================
Unit tests for LabelCache and Delta Relabeling in dataset_factory/label_cache.py.

Verifies:
1. Tracking (file_path, file_mtime_ns, file_size, exemplar_revision).
2. JSON persistence and atomic write integrity.
3. 0-box images are NEVER skipped during relabeling.
4. Unchanged images with valid boxes and matching exemplar revisions ARE skipped.
5. Candidate crops caching and fast delta re-evaluation when exemplar revisions update.
6. Backward compatibility with existing .label_cache.json schema.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any, Dict

import numpy as np
import pytest

from dataset_factory.exemplar_memory import ExemplarMemory, extract_crop_features
from dataset_factory.label_cache import (
    LabelCache,
    LabelCacheEntry,
    get_current_exemplar_revision,
    plan_batch_relabel,
)


def _distinct_crop(seed: int, base_bgr: tuple[int, int, int]) -> np.ndarray:
    """A real image crop: flat base colour plus seeded per-pixel jitter, so
    extract_crop_features() yields a genuine (non-identical) feature dict."""
    rng = np.random.default_rng(seed)
    crop = np.full((64, 64, 3), base_bgr, dtype=np.int16)
    crop = crop + rng.integers(-12, 13, (64, 64, 3))
    return np.clip(crop, 0, 255).astype(np.uint8)


def _real_exemplar_memory(tmp_path: Path, pos_crop=None, neg_crop=None) -> ExemplarMemory:
    """A REAL ExemplarMemory (tmp JSON) populated through the production
    add_positive_exemplar / add_negative_exemplar API — no stub match_*."""
    memory = ExemplarMemory(json_path=tmp_path / "learned_exemplars.json")
    if pos_crop is not None:
        assert memory.add_positive_exemplar(pos_crop, class_id=0, class_name="herb_aria", source="ui_approval")
    if neg_crop is not None:
        assert memory.add_negative_exemplar(neg_crop, class_id=14, class_name="obstacle", source="ui_rejection")
    return memory


@pytest.fixture
def temp_cache_dir(tmp_path: Path):
    """Provides a temporary directory with mock image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(parents=True)

    # Create dummy images
    img1 = img_dir / "frame_aria_01.png"
    img1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)

    img2 = img_dir / "frame_empty_02.png"
    img2.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 300)

    img3 = img_dir / "frame_monster_03.png"
    img3.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 400)

    cache_file = tmp_path / ".label_cache.json"

    return {
        "dir": tmp_path,
        "cache_file": cache_file,
        "img1": img1,
        "img2": img2,
        "img3": img3,
    }


def test_label_cache_record_and_track_stat(temp_cache_dir):
    """Verifies tracking of (file_path, file_mtime_ns, file_size, exemplar_revision)."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img1 = temp_cache_dir["img1"]
    ex_rev = 123456789

    entry = cache.record_label(
        file_path=img1,
        exemplar_revision=ex_rev,
        box_count=2,
    )

    assert entry.file_path == str(img1)
    assert entry.exemplar_revision == ex_rev
    assert entry.box_count == 2
    assert entry.file_size == img1.stat().st_size
    assert entry.file_mtime_ns == img1.stat().st_mtime_ns

    # Verify JSON persistence
    assert temp_cache_dir["cache_file"].is_file()
    saved = json.loads(temp_cache_dir["cache_file"].read_text(encoding="utf-8"))
    assert img1.name in saved
    assert saved[img1.name]["file_mtime_ns"] == entry.file_mtime_ns
    assert saved[img1.name]["mtime_ns"] == entry.file_mtime_ns
    assert saved[img1.name]["file_size"] == entry.file_size
    assert saved[img1.name]["exemplar_revision"] == ex_rev


def test_zero_box_images_never_skipped(temp_cache_dir):
    """
    CRITICAL REQUIREMENT: Never skip 0-box (unlabeled) images when user requests relabeling.
    Even if mtime, size, and exemplar_revision match, 0-box images must return should_relabel=True.
    """
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img_empty = temp_cache_dir["img2"]
    ex_rev = 999000111

    # Record with 0 boxes
    cache.record_label(
        file_path=img_empty,
        exemplar_revision=ex_rev,
        box_count=0,
    )

    # Verify entry is recorded
    entry = cache.get_entry(img_empty)
    assert entry is not None
    assert entry.box_count == 0

    # User requests relabeling with the SAME revision:
    # Must still return True because box_count is 0!
    assert cache.should_relabel(img_empty, exemplar_revision=ex_rev) is True


def test_unchanged_images_with_boxes_are_skipped(temp_cache_dir):
    """Verifies that images with > 0 boxes and matching stat + revision are skipped."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img1 = temp_cache_dir["img1"]
    ex_rev = 555444333

    # Record with 3 boxes
    cache.record_label(
        file_path=img1,
        exemplar_revision=ex_rev,
        box_count=3,
    )

    # Same revision & unchanged image -> should NOT relabel
    assert cache.should_relabel(img1, exemplar_revision=ex_rev) is False

    # Force relabel -> should relabel
    assert cache.should_relabel(img1, exemplar_revision=ex_rev, force=True) is True

    # Exemplar revision updated -> should relabel
    assert cache.should_relabel(img1, exemplar_revision=ex_rev + 1) is True


def test_file_modification_triggers_relabel(temp_cache_dir):
    """Verifies that if the image file on disk changes (mtime or size), should_relabel returns True."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img1 = temp_cache_dir["img1"]
    ex_rev = 555444333

    cache.record_label(img1, exemplar_revision=ex_rev, box_count=2)
    assert cache.should_relabel(img1, exemplar_revision=ex_rev) is False

    # Modify the image on disk
    time.sleep(0.01)
    img1.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 250)

    assert cache.should_relabel(img1, exemplar_revision=ex_rev) is True


def test_candidate_crops_delta_relabel(temp_cache_dir):
    """
    Verifies that when exemplar updates, relevant candidate crops can be re-evaluated
    without re-running the heavy visual pipeline.
    """
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img3 = temp_cache_dir["img3"]
    rev1 = 1000

    # Candidate crops carry features extracted from REAL crops with the
    # production extractor (extract_crop_features), not hand-made histograms.
    neg_crop = _distinct_crop(seed=77, base_bgr=(35, 45, 40))
    pos_crop = _distinct_crop(seed=78, base_bgr=(200, 140, 60))

    candidate_crops = [
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "confidence": 0.86,
            "feature": extract_crop_features(pos_crop),
            "bbox": [0.5, 0.5, 0.1, 0.1],
        },
        {
            "class_id": 14,
            "class_name": "obstacle",
            "confidence": 0.70,
            "feature": extract_crop_features(neg_crop),
            "bbox": [0.2, 0.2, 0.05, 0.05],
        },
        {
            "class_id": 2,
            "class_name": "crystal",
            "confidence": 0.95,
            "hand_label": True,  # Protected immunity
            "bbox": [0.8, 0.8, 0.1, 0.1],
        },
    ]

    cache.record_label(
        img3,
        exemplar_revision=rev1,
        candidate_crops=candidate_crops,
        box_count=len(candidate_crops),
    )

    rev2 = 2000
    assert cache.can_delta_relabel(img3, exemplar_revision=rev2) is True

    # Real production memory: the obstacle crop is registered as a learned
    # negative exemplar (class 14) via the real add API, so re-evaluation must
    # suppress it through the real compute_similarity path.
    memory = _real_exemplar_memory(temp_cache_dir["dir"], pos_crop=pos_crop, neg_crop=neg_crop)

    # Re-evaluate candidate crops
    refined = cache.re_evaluate_candidate_crops(
        file_path=img3,
        exemplar_memory=memory,
        exemplar_revision=rev2,
    )

    # The negative match (obstacle) should have been suppressed!
    # The hand-label crystal must be preserved under immunity shield!
    assert any(c.get("hand_label") is True for c in refined)
    assert not any(c.get("class_id") == 14 for c in refined)

    # Cache should be updated to rev2
    entry = cache.get_entry(img3)
    assert entry.exemplar_revision == rev2
    assert entry.box_count == len(refined)


def test_plan_batch_relabel_partitioning(temp_cache_dir):
    """Verifies that plan_batch_relabel correctly partitions skip, delta, and full paths."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img1 = temp_cache_dir["img1"]
    img2 = temp_cache_dir["img2"]
    img3 = temp_cache_dir["img3"]

    rev1 = 1000

    # img1: fully up to date with boxes -> SKIP
    cache.record_label(img1, exemplar_revision=rev1, box_count=2)

    # img2: 0 boxes -> FULL (never skipped!)
    cache.record_label(img2, exemplar_revision=rev1, box_count=0)

    # img3: has candidate crops, but revision updated -> DELTA
    crops = [{"class_id": 1, "confidence": 0.9, "bbox": [0.1, 0.1, 0.1, 0.1]}]
    cache.record_label(img3, exemplar_revision=rev1 - 50, candidate_crops=crops, box_count=1)

    plan = plan_batch_relabel(
        [img1, img2, img3],
        exemplar_revision=rev1,
        label_cache=cache,
    )

    assert img1 in plan["skip"]
    assert img2 in plan["full"]  # 0-box is NEVER skipped!
    assert img3 in plan["delta"]  # Candidate crops can be delta evaluated!


def test_dict_interoperability_and_legacy_keys(temp_cache_dir):
    """Verifies that LabelCache supports dict-like operations and legacy alias keys."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])

    # Set item with legacy dict format
    cache["legacy_frame.png"] = {
        "mtime_ns": 987654321,
        "size": 123456,
        "box_count": 5,
        "exemplar_rev": 11223344,
    }

    assert "legacy_frame.png" in cache
    assert len(cache) == 1

    entry = cache["legacy_frame.png"]
    # Check new attribute names
    assert entry.file_mtime_ns == 987654321
    assert entry.file_size == 123456
    assert entry.exemplar_revision == 11223344
    assert entry.box_count == 5

    # Check dict-like access on entry
    assert entry["mtime_ns"] == 987654321
    assert entry["size"] == 123456
    assert entry["exemplar_rev"] == 11223344
    assert entry.get("box_count") == 5

    # Export to dict
    d = cache.to_dict()
    assert "legacy_frame.png" in d
    assert d["legacy_frame.png"]["mtime_ns"] == 987654321
    assert d["legacy_frame.png"]["file_mtime_ns"] == 987654321


def test_virtual_mock_paths_without_disk_files(temp_cache_dir):
    """Verifies behavior for virtual/mock paths not present on disk."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])

    # Untracked path -> returns True
    assert cache.should_relabel("untracked_virtual.png", exemplar_revision=1) is True

    # Record with 0 boxes -> returns True (0-box never skipped)
    cache.record_label("virtual_zero.png", exemplar_revision=100, box_count=0)
    assert cache.should_relabel("virtual_zero.png", exemplar_revision=100) is True

    # Record with 2 boxes -> same revision returns False
    cache.record_label("virtual_valid.png", exemplar_revision=100, box_count=2)
    assert cache.should_relabel("virtual_valid.png", exemplar_revision=100) is False

    # Revision changed -> returns True
    assert cache.should_relabel("virtual_valid.png", exemplar_revision=101) is True


def test_positive_reinforcement_in_delta_crops(temp_cache_dir):
    """Verifies that matching positive exemplars boost confidence and can correct class."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img = temp_cache_dir["img1"]

    # Feature extracted from a REAL crop with the production extractor.
    pos_crop = _distinct_crop(seed=55, base_bgr=(200, 140, 60))
    pos_feature = extract_crop_features(pos_crop)
    crops = [
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "confidence": 0.75,
            "feature": pos_feature,
            "bbox": [0.4, 0.4, 0.2, 0.2],
        }
    ]

    cache.record_label(img, exemplar_revision=10, candidate_crops=crops, box_count=1)

    # REAL ExemplarMemory holding the same crop as a learned positive exemplar.
    memory = _real_exemplar_memory(temp_cache_dir["dir"], pos_crop=pos_crop)

    refined = cache.re_evaluate_candidate_crops(
        file_path=img,
        exemplar_memory=memory,
        exemplar_revision=20,
    )

    assert len(refined) == 1
    # Confidence should have been reinforced / boosted
    assert refined[0]["confidence"] > 0.75
    assert refined[0].get("exemplar_reinforced") is True


def test_non_numeric_confidence_does_not_abort_delta_relabel(temp_cache_dir):
    """S-2103: a single non-numeric confidence ('high', None) loaded from a
    disk cache must not raise ValueError and abort the whole file's delta
    relabel; the bad crop falls back to its default and the remaining crops
    are still processed."""
    cache = LabelCache(cache_path=temp_cache_dir["cache_file"])
    img = temp_cache_dir["img1"]

    good_crop = _distinct_crop(seed=91, base_bgr=(200, 140, 60))
    crops = [
        {
            "class_id": 2,
            "class_name": "crystal",
            "confidence": "high",  # non-numeric; hand-label branch (was line 675)
            "hand_label": True,
            "bbox": [0.8, 0.8, 0.1, 0.1],
        },
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "confidence": "high",  # non-numeric; confidence-floor check (was line 767)
            "feature": extract_crop_features(good_crop),
            "bbox": [0.4, 0.4, 0.2, 0.2],
        },
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "confidence": None,  # non-numeric; same floor check
            "feature": extract_crop_features(good_crop),
            "bbox": [0.5, 0.5, 0.2, 0.2],
        },
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "confidence": 0.90,
            "feature": extract_crop_features(good_crop),
            "bbox": [0.1, 0.1, 0.2, 0.2],
        },
    ]

    cache.record_label(img, exemplar_revision=10, candidate_crops=crops, box_count=4)

    memory = _real_exemplar_memory(temp_cache_dir["dir"])

    # Must not raise ValueError.
    refined = cache.re_evaluate_candidate_crops(
        file_path=img,
        exemplar_memory=memory,
        exemplar_revision=20,
    )

    # Hand-labeled crop survives with the immunity floor confidence.
    hand = [c for c in refined if c.get("hand_label") is True]
    assert len(hand) == 1
    assert hand[0]["confidence"] == 0.95
    # The good numeric crop survives; the two non-numeric non-hand crops
    # fall back to 0.0 and are dropped by the confidence floor.
    assert any(c.get("bbox") == [0.1, 0.1, 0.2, 0.2] for c in refined)
    assert len(refined) == 2
    entry = cache.get_entry(img)
    assert entry.exemplar_revision == 20
    assert entry.box_count == 2

