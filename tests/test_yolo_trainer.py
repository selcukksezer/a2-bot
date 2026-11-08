# -*- coding: utf-8 -*-
"""
tests/test_yolo_trainer.py
==========================
Unit & Integration Tests for YOLOTrainer and YOLOTrainingWorker (Worker A09):
- Support for `only_approved: bool = True` selective training pipeline.
- Points strictly to `dataset/approved/data_approved.yaml` when `only_approved=True`.
- Correct status logging: `[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: {count} onaylı kare`.
- Proper error handling when approved dataset is empty.
- Verification of QThread worker and standalone trigger functions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from dataset_factory.approved_pool import ApprovedDatasetPool
from dataset_factory.yolo_trainer import (
    RTXTrainingWorker,
    YOLOTrainer,
    YOLOTrainingWorker,
    resolve_yolo_training_yaml,
    start_training_worker,
    start_yolo_training,
)


@pytest.fixture
def mock_dataset_env(tmp_path: Path):
    """Sets up an isolated project and dataset directory structure."""
    proj = tmp_path / "project"
    dataset = proj / "dataset"
    approved = dataset / "approved"

    (dataset / "images" / "train").mkdir(parents=True, exist_ok=True)
    (dataset / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset / "labels" / "val").mkdir(parents=True, exist_ok=True)

    (approved / "images" / "train").mkdir(parents=True, exist_ok=True)
    (approved / "images" / "val").mkdir(parents=True, exist_ok=True)
    (approved / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (approved / "labels" / "val").mkdir(parents=True, exist_ok=True)

    # Base data.yaml
    base_yaml = dataset / "data.yaml"
    base_yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnc: 22\nnames: {0: herb_aria}\n", encoding="utf-8")

    # Add 2 dummy images to unvetted dataset
    (dataset / "images" / "train" / "raw_1.jpg").write_bytes(b"img1")
    (dataset / "images" / "val" / "raw_2.jpg").write_bytes(b"img2")

    return proj, dataset, approved


def test_yolo_trainer_resolves_approved_yaml_strictly(mock_dataset_env, caplog):
    """Verifies that only_approved=True strictly targets data_approved.yaml and logs exact status."""
    proj, dataset, approved = mock_dataset_env

    # Populate approved pool with 3 samples
    (approved / "images" / "train" / "appr_1.jpg").write_bytes(b"appr1")
    (approved / "images" / "train" / "appr_2.jpg").write_bytes(b"appr2")
    (approved / "images" / "val" / "appr_3.jpg").write_bytes(b"appr3")

    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset)

    with caplog.at_level("INFO"):
        yaml_path, count = trainer.resolve_training_data_yaml(only_approved=True)

    assert count == 3
    assert yaml_path == approved / "data_approved.yaml"
    assert yaml_path.is_file()

    # Verify exact required log message
    expected_log = "[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: 3 onaylı kare"
    assert expected_log in caplog.text


def test_yolo_trainer_resolves_full_dataset_when_only_approved_false(mock_dataset_env, caplog):
    """Verifies that only_approved=False targets data.yaml and logs full dataset count."""
    proj, dataset, approved = mock_dataset_env

    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset)

    with caplog.at_level("INFO"):
        yaml_path, count = trainer.resolve_training_data_yaml(only_approved=False)

    assert count == 2
    assert yaml_path == dataset / "data.yaml"
    expected_log = "[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: 2 kare"
    assert expected_log in caplog.text


def test_run_training_sync_with_approved_samples(mock_dataset_env):
    """Verifies run_training_sync passes data_approved.yaml directly to engine."""
    proj, dataset, approved = mock_dataset_env

    # 1 approved sample
    (approved / "images" / "train" / "appr_1.jpg").write_bytes(b"appr1")

    mock_engine = MagicMock()
    mock_engine.run_training_sync.return_value = {
        "status": "success",
        "started": True,
        "mode": "ultralytics_cuda_fp16",
    }

    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=mock_engine)

    status_logs = []
    res = trainer.run_training_sync(
        epochs=10,
        only_approved=True,
        status_callback=status_logs.append,
    )

    assert res["status"] == "success"
    assert res["only_approved"] is True
    assert res["sample_count"] == 1

    # Verify mock engine called with data_approved.yaml
    mock_engine.run_training_sync.assert_called_once()
    call_kwargs = mock_engine.run_training_sync.call_args[1]
    assert call_kwargs["data_yaml_path"] == approved / "data_approved.yaml"
    assert call_kwargs["epochs"] == 10

    assert any("Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: 1 onaylı kare" in s for s in status_logs)


def test_run_training_sync_fails_cleanly_when_approved_empty(mock_dataset_env):
    """Verifies that running with 0 approved samples fails cleanly with clear error message."""
    proj, dataset, approved = mock_dataset_env

    mock_engine = MagicMock()
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=mock_engine)

    status_logs = []
    res = trainer.run_training_sync(
        epochs=10,
        only_approved=True,
        status_callback=status_logs.append,
    )

    assert res["started"] is False
    assert res["status"] == "failed"
    assert "Onaylanmış eğitim karesi bulunamadı" in res["error"]
    assert res["approved_count"] == 0
    # Engine must NOT be called when count is 0
    mock_engine.run_training_sync.assert_not_called()


def test_yolo_training_worker_approved_execution(mock_dataset_env):
    """Verifies YOLOTrainingWorker supports only_approved=True and emits correct signals."""
    proj, dataset, approved = mock_dataset_env

    (approved / "images" / "train" / "appr_1.jpg").write_bytes(b"appr1")

    mock_engine = MagicMock()
    mock_engine.run_training_sync.return_value = {"status": "success", "mode": "ultralytics_cuda_fp16"}
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=mock_engine)

    worker = YOLOTrainingWorker(
        trainer=trainer,
        epochs=5,
        only_approved=True,
    )

    status_received = []
    finished_received = []
    worker.status_text.connect(status_received.append)
    worker.training_finished.connect(finished_received.append)

    # Run synchronously on current thread for deterministic test
    worker.run()

    assert any("Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: 1 onaylı kare" in s for s in status_received)
    assert len(finished_received) == 1
    assert finished_received[0]["status"] == "success"


def test_yolo_training_worker_empty_approved_emits_failed(mock_dataset_env):
    """Verifies YOLOTrainingWorker emits training_failed when 0 approved frames exist."""
    proj, dataset, approved = mock_dataset_env

    mock_engine = MagicMock()
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=mock_engine)

    worker = YOLOTrainingWorker(
        trainer=trainer,
        epochs=5,
        only_approved=True,
    )

    failed_received = []
    worker.training_failed.connect(failed_received.append)

    worker.run()

    assert len(failed_received) == 1
    assert "Onaylanmış eğitim karesi bulunamadı" in failed_received[0]
    mock_engine.run_training_sync.assert_not_called()


def test_standalone_helpers(mock_dataset_env):
    """Verifies start_yolo_training, start_training_worker, and resolve_yolo_training_yaml."""
    proj, dataset, approved = mock_dataset_env
    (approved / "images" / "train" / "appr_1.jpg").write_bytes(b"appr1")

    yaml_p, count = resolve_yolo_training_yaml(
        only_approved=True, project_root=proj, dataset_dir=dataset
    )
    assert count == 1
    assert yaml_p == approved / "data_approved.yaml"

    mock_engine = MagicMock()
    mock_engine.run_training_sync.return_value = {"status": "success"}
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=mock_engine)

    res = start_yolo_training(
        only_approved=True,
        epochs=15,
        trainer=trainer,
    )
    assert res["status"] == "success"
    assert res["only_approved"] is True
