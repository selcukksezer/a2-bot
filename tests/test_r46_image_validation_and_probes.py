# -*- coding: utf-8 -*-
"""tests/test_r46_image_validation_and_probes.py
================================================
R46 (F-1451 / S-1903 / S-1902 / S-2402) regression tests.

- Approved pool ingest rejects image payloads without the magic bytes of
  their claimed format (bytes path, file-copy path, sync_from_disk).
- The quality-gate and dataset-factory-package readiness/contract stubs are
  bound to real probes instead of returning constant True.
- `AtomicReviewTxtSynchronizer` is exported from `dataset_factory`
  (Karar 6 / F-1980 cross-reference).
"""

import json
import shutil
import tempfile
from pathlib import Path

import pytest

import dataset_factory
from dataset_factory import (
    AtomicReviewTxtSynchronizer,
    get_dataset_factory_package_summary,
    validate_dataset_factory_package_contract,
)
from dataset_factory.approved_pool import ApprovedDatasetPool
from dataset_factory.quality_gate import (
    QualityGate,
    get_quality_gate_readiness_status,
    validate_quality_gate_contract,
)

_MINIMAL_JPEG = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xFF\xD9"
_MINIMAL_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8

_BOXES = [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}]


@pytest.fixture
def temp_pool_dir():
    temp_dir = tempfile.mkdtemp(prefix="test_r46_pool_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_rejects_image_bytes_without_jpeg_magic(temp_pool_dir):
    """F-1451: arbitrary bytes must not be persisted as .jpg."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    ok = pool.add_approved_sample(
        stem="corrupt_bytes", split="train", trusted_boxes=_BOXES,
        image_bytes=b"FAKE_JPEG_ALPHA",
    )
    assert ok is False
    assert pool.is_approved("corrupt_bytes") is False
    assert not (pool.images_train_dir / "corrupt_bytes.jpg").exists()


def test_accepts_image_bytes_with_jpeg_magic(temp_pool_dir):
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    ok = pool.add_approved_sample(
        stem="good_bytes", split="train", trusted_boxes=_BOXES,
        image_bytes=_MINIMAL_JPEG,
    )
    assert ok is True
    assert (pool.images_train_dir / "good_bytes.jpg").is_file()


def test_rejects_file_copy_without_matching_header(tmp_path):
    """S-1903: the file-copy path validates the source header, not just the suffix."""
    src = tmp_path / "src.jpg"
    src.write_bytes(b"NOT_AN_IMAGE_AT_ALL")
    pool = ApprovedDatasetPool(pool_root=tmp_path / "pool")
    ok = pool.add_approved_sample(
        stem="bad_src", split="train", trusted_boxes=_BOXES, image_path=src,
    )
    assert ok is False
    assert pool.is_approved("bad_src") is False


def test_accepts_png_file_copy_with_png_magic(tmp_path):
    src = tmp_path / "src.png"
    src.write_bytes(_MINIMAL_PNG)
    pool = ApprovedDatasetPool(pool_root=tmp_path / "pool")
    ok = pool.add_approved_sample(
        stem="good_png", split="train", trusted_boxes=_BOXES, image_path=src,
    )
    assert ok is True
    assert (pool.images_train_dir / "good_png.png").is_file()


def test_sync_from_disk_skips_non_image_file(temp_pool_dir):
    """S-1903: sync_from_disk registers by header, not by suffix alone."""
    pool = ApprovedDatasetPool(pool_root=temp_pool_dir)
    assert pool.add_approved_sample(
        stem="good_one", split="train", trusted_boxes=_BOXES,
        image_bytes=_MINIMAL_JPEG,
    ) is True
    # Plant a corrupt file + label directly in the pool, then rebuild.
    bad = pool.images_train_dir / "corrupt_on_disk.jpg"
    bad.write_bytes(b"GARBAGE_NO_MAGIC")
    (pool.labels_train_dir / "corrupt_on_disk.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    pool.manifest_path.unlink()
    pool2 = ApprovedDatasetPool(pool_root=temp_pool_dir)
    assert pool2.is_approved("good_one") is True
    assert pool2.is_approved("corrupt_on_disk") is False


def test_quality_gate_contract_is_a_real_probe():
    """S-1902: the contract is True only because the live gate probe passes."""
    assert validate_quality_gate_contract() is True
    gate = QualityGate()
    assert gate.get_quality_gate_ready_flag() == {"module": "quality_gate", "ready": True}
    assert gate.get_quality_gate_diagnostic_snapshot()["status"] == "ok"
    assert get_quality_gate_readiness_status()["ready"] is True


def test_package_contract_is_a_real_probe():
    """S-2402: the package summary reports per-component probe results."""
    summary = get_dataset_factory_package_summary()
    assert summary["dataset_factory_ready"] is True
    # F-2000 (84eecb9): the probe covers the LRU frame-decode cache too —
    # the package exports it (dataset_factory/__init__.py) and the probe
    # imports it, so the contract has five components, not four.
    assert summary["components"] == {
        "frame_time_machine": True,
        "quality_gate": True,
        "approved_pool": True,
        "atomic_review_sync": True,
        "lru_frame_cache": True,
    }
    assert validate_dataset_factory_package_contract() is True


def test_atomic_review_sync_exported_from_package():
    """Karar 6 / F-1980: AtomicReviewTxtSynchronizer is public package API."""
    assert dataset_factory.AtomicReviewTxtSynchronizer is AtomicReviewTxtSynchronizer
    assert "AtomicReviewTxtSynchronizer" in dataset_factory.__all__
    assert callable(AtomicReviewTxtSynchronizer)
