# -*- coding: utf-8 -*-
"""tests/test_d39_approved_pool_dev.py
========================================
D39 DEV improvements for dataset_factory/approved_pool.py (no bug fixes here):

I1  YAML-safe class names in generate_approved_data_yaml (raw interpolation
    broke yaml.safe_load for names with ':', '#', bool words, ...).
I2  Manifest stats self-repair on load (stale counters were trusted verbatim).
I3  Disk I/O moved out of the pool lock in add/bulk (is_approved() used to
    block for the whole bulk); manifest commit still under the lock, with a
    files-still-exist re-check so the manifest never points at missing files.
I4  Dedup: _resolve_split (single+bulk), _delete_sample_files (both remove
    paths), _atomic_write_bytes (bytes+numpy image branches).

Temp dirs live under ~/workspace (never /tmp: the VM tmpfs is full).
"""

import json
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from dataset_factory.approved_pool import ApprovedDatasetPool, _yaml_scalar

# NOTE: tests must not write to /tmp (512M tmpfs is full) — keep fixtures here.
_D39_TMP_BASE = Path.home() / "workspace" / "d39-test-tmp"


def _valid_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = (10, 20, 30)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return bytes(buf)


@pytest.fixture
def temp_pool_dir():
    _D39_TMP_BASE.mkdir(parents=True, exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="test_d39_pool_", dir=str(_D39_TMP_BASE))
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def _box(cid: int = 0):
    return {"class_id": cid, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}


# --- I1: YAML-safe class names -------------------------------------------------

def test_yaml_special_class_names_roundtrip(temp_pool_dir):
    """Names with ':', '#', quotes or bool words must survive yaml.safe_load."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    names = {0: "plain_name", 1: "weird: name", 2: "hash # tag", 3: "yes",
             4: 'with "quote"', 5: "trailing ", 6: "1.5", 7: "çiçek_ö"}
    out = pool.generate_approved_data_yaml(class_names=names)
    data = yaml.safe_load(Path(out).read_text(encoding="utf-8"))
    assert data["nc"] == 8
    assert data["names"] == names


def test_yaml_canonical_names_stay_plain(temp_pool_dir):
    """Safe names are emitted unquoted — canonical output is unchanged."""
    assert _yaml_scalar("herb_aria") == "herb_aria"
    assert _yaml_scalar("skill_slot_ready") == "skill_slot_ready"
    assert _yaml_scalar("yes") == '"yes"'
    assert _yaml_scalar("a: b") == '"a: b"'
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    out = pool.generate_approved_data_yaml(class_names={0: "herb_aria", 1: "kinah"})
    content = Path(out).read_text(encoding="utf-8")
    assert "  0: herb_aria\n" in content
    assert "  1: kinah\n" in content


# --- I2: manifest stats self-repair --------------------------------------------

def test_manifest_stale_stats_repaired_on_load(temp_pool_dir):
    """A manifest whose counters disagree with its stems is repaired on load."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    jb = _valid_jpeg_bytes()
    assert pool.add_approved_sample("s1", "train", trusted_boxes=[_box(0)], image_bytes=jb)
    assert pool.add_approved_sample("s2", "val", trusted_boxes=[_box(1)], image_bytes=jb)

    man = json.loads(pool.manifest_path.read_text(encoding="utf-8"))
    man["total_approved_images"] = 99
    man["train_count"] = 0
    man["val_count"] = 0
    man["total_boxes"] = 0
    pool.manifest_path.write_text(json.dumps(man), encoding="utf-8")

    pool2 = ApprovedDatasetPool(pool_root=temp_pool_dir)
    stats = pool2.get_approved_stats()
    assert stats["total_approved_images"] == 2
    assert stats["train_count"] == 1
    assert stats["val_count"] == 1
    assert stats["total_boxes"] == 2
    # Repair is persisted, not just in-memory.
    repaired = json.loads(pool2.manifest_path.read_text(encoding="utf-8"))
    assert repaired["total_approved_images"] == 2
    assert repaired["train_count"] == 1


def test_manifest_not_rewritten_when_consistent(temp_pool_dir):
    """A consistent manifest is left byte-identical on reload (no write churn)."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    pool.add_approved_sample("s1", "train", trusted_boxes=[_box(0)], image_bytes=_valid_jpeg_bytes())
    before = pool.manifest_path.read_bytes()
    ApprovedDatasetPool(pool_root=temp_pool_dir)
    assert pool.manifest_path.read_bytes() == before


# --- I4: shared split resolution -----------------------------------------------

def test_resolve_split_shared_by_single_and_bulk(temp_pool_dir):
    """Same stem + bogus split resolves identically via single add and bulk add."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    val_stem = next(s for s in (f"stem_{i}" for i in range(200))
                    if pool._resolve_split(s, "bogus") == "val")
    train_stem = next(s for s in (f"stem_{i}" for i in range(200))
                      if pool._resolve_split(s, "bogus") == "train")
    jb = _valid_jpeg_bytes()
    assert pool.add_approved_sample(val_stem, "bogus", trusted_boxes=[_box()], image_bytes=jb)
    assert pool.add_approved_sample(train_stem, "bogus", trusted_boxes=[_box()], image_bytes=jb)
    assert (pool.images_val_dir / f"{val_stem}.jpg").is_file()
    assert (pool.images_train_dir / f"{train_stem}.jpg").is_file()
    n = pool.bulk_add_approved_samples([
        {"stem": f"bulk_{val_stem}", "split": "bogus", "image_bytes": jb, "trusted_boxes": [_box()]},
        {"stem": f"bulk_{train_stem}", "split": "bogus", "image_bytes": jb, "trusted_boxes": [_box()]},
    ])
    assert n == 2
    assert (pool.images_val_dir / f"bulk_{val_stem}.jpg").is_file() == (
        pool._resolve_split(f"bulk_{val_stem}", "bogus") == "val")
    assert (pool.images_train_dir / f"bulk_{train_stem}.jpg").is_file() == (
        pool._resolve_split(f"bulk_{train_stem}", "bogus") == "train")


# --- I3: lock hold time ----------------------------------------------------------

def test_bulk_does_not_block_is_approved(temp_pool_dir):
    """During a 150-sample bulk add, is_approved() must stay responsive.

    Before: the prober's worst call blocked for the whole bulk (~bulk
    duration, ratio ~1.0). After: worst call is a fraction of the bulk and
    p99 stays at sub-ms (only the short manifest commit holds the lock).
    """
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    jb = _valid_jpeg_bytes()
    samples = [{"stem": f"b{i:04d}", "split": "train", "image_bytes": jb,
                "trusted_boxes": [_box()]} for i in range(150)]
    lat, stop = [], threading.Event()

    def prober():
        while not stop.is_set():
            t0 = time.perf_counter()
            pool.is_approved("probe_stem")
            lat.append(time.perf_counter() - t0)
            time.sleep(0.001)

    th = threading.Thread(target=prober)
    th.start()
    t0 = time.perf_counter()
    n = pool.bulk_add_approved_samples(samples)
    bulk_dur = time.perf_counter() - t0
    stop.set()
    th.join()
    assert n == 150
    assert pool.get_approved_stats()["total_approved_images"] == 150
    assert len(lat) >= 10, "prober barely ran; measurement is meaningless"
    lat.sort()
    p99 = lat[int(len(lat) * 0.99)]
    assert lat[-1] < 0.5 * bulk_dur, f"max is_approved latency {lat[-1]*1000:.1f}ms vs bulk {bulk_dur*1000:.0f}ms"
    assert p99 < 0.010, f"p99 is_approved latency {p99*1000:.2f}ms"


def test_concurrent_single_adds_all_land(temp_pool_dir):
    """Concurrent add_approved_sample calls from 8 threads all commit."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    jb = _valid_jpeg_bytes()
    errors = []

    def worker(tid: int):
        try:
            for i in range(10):
                ok = pool.add_approved_sample(
                    f"t{tid}_s{i:02d}", "train", trusted_boxes=[_box()], image_bytes=jb)
                assert ok, f"add failed t{tid} s{i}"
        except Exception as ex:  # noqa: BLE001 - collected and re-raised below
            errors.append(ex)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    stats = pool.get_approved_stats()
    assert stats["total_approved_images"] == 80
    assert stats["total_boxes"] == 80
    # Manifest on disk agrees after a fresh reload.
    pool2 = ApprovedDatasetPool(pool_root=temp_pool_dir)
    assert pool2.get_approved_stats()["total_approved_images"] == 80


# --- I4: shared file deletion + atomic bytes write -------------------------------

def test_both_remove_paths_delete_files(temp_pool_dir):
    """remove_sample and remove_approved_sample both delete image+label."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    jb = _valid_jpeg_bytes()
    pool.add_approved_sample("a1", "train", trusted_boxes=[_box()], image_bytes=jb)
    pool.add_approved_sample("a2", "val", trusted_boxes=[_box()], image_bytes=jb)
    assert pool.remove_sample("a1") is True
    assert not (pool.images_train_dir / "a1.jpg").exists()
    assert not (pool.labels_train_dir / "a1.txt").exists()
    assert pool.is_approved("a1") is False
    assert pool.remove_approved_sample("a2") is True
    assert not (pool.images_val_dir / "a2.jpg").exists()
    assert not (pool.labels_val_dir / "a2.txt").exists()
    assert pool.remove_sample("missing_stem") is False
    assert pool.remove_approved_sample("missing_stem") is False


def test_numpy_array_add_still_works(temp_pool_dir):
    """Numpy image input still saves (covers the _atomic_write_bytes branch)."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    arr[:, :] = (40, 50, 60)
    ok = pool.add_approved_sample("np1", "train", image_path=arr, trusted_boxes=[_box(3)])
    assert ok is True
    img_path = pool.images_train_dir / "np1.jpg"
    assert img_path.is_file()
    assert img_path.read_bytes()[:2] == b"\xff\xd8"
