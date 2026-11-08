# -*- coding: utf-8 -*-
"""tests/test_step5_validate_to_step6_publish.py
=============================================
Unit tests verifying Step 5 (Doğrula) -> Step 6 (Yayınla) pipeline transition:
1. Valid fresh weights allow advancing from Step 5 to Step 6 without bouncing back to Step 4.
2. 'Doğrulama tamam — Yayınla adımına geç' clears stale flags and sets validated=True, trained=True.
3. In Step 6, checklist items are checked and btn_publish is enabled.
4. Publishing succeeds and updates published=True.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import YOLOFactoryDialog
from dataset_factory.dataset_validator import mark_model_stale


def _setup_mock_environment(tmp_path: Path) -> Dict[str, Path]:
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    rtx_run = models_dir / "rtx_runs" / "gatherion_rtx_fp16"
    weights_dir = rtx_run / "weights"

    dataset_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    # 1. Freeze manifest at t = 1000.0
    freeze_file = dataset_dir / "frozen_manifest.json"
    freeze_file.write_text(
        json.dumps({
            "version": "dataset-v4",
            "frozen_at": "1970-01-01T00:16:40",
            "total_frames": 10,
        }),
        encoding="utf-8",
    )
    os.utime(freeze_file, (1000.0, 1000.0))

    # 2. Weights and results.csv at t = 2000.0 (fresh after freeze)
    best_pt = weights_dir / "best.pt"
    best_pt.write_bytes(b"MOCK_BEST_PT_WEIGHTS")
    os.utime(best_pt, (2000.0, 2000.0))

    best_onnx = weights_dir / "best.onnx"
    best_onnx.write_bytes(b"MOCK_BEST_ONNX")
    os.utime(best_onnx, (2000.0, 2000.0))

    results_csv = rtx_run / "results.csv"
    results_csv.write_text(
        "epoch,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "1,0.05,0.03,0.02,0.85,0.80,0.75,0.50\n",
        encoding="utf-8",
    )
    os.utime(results_csv, (2000.0, 2000.0))

    return {
        "dataset_dir": dataset_dir,
        "models_dir": models_dir,
        "rtx_run": rtx_run,
        "freeze_file": freeze_file,
        "best_pt": best_pt,
        "best_onnx": best_onnx,
        "results_csv": results_csv,
    }


def test_artifacts_fresh_vs_freeze_ignores_model_stale(tmp_path: Path) -> None:
    """_artifacts_are_fresh_vs_freeze returns True when weights are newer than freeze, even if model is marked stale."""
    env = _setup_mock_environment(tmp_path)

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: env["dataset_dir"]
    dlg._models_dir = lambda: env["models_dir"]
    dlg._find_train_artifact_run_dir = lambda: env["rtx_run"]

    # Mark model stale
    mark_model_stale(
        dataset_dir=env["dataset_dir"],
        models_dir=env["models_dir"],
        reason="Kare silindi",
    )
    dlg._pipeline["model_stale"] = True

    # Must still recognize artifacts are fresh
    assert dlg._artifacts_are_fresh_vs_freeze() is True


def test_finish_validate_advances_to_step6_and_enables_publish(tmp_path: Path) -> None:
    """Clicking 'Doğrulama tamam — Yayınla adımına geç' must advance to Step 6, clear stale, and enable btn_publish."""
    env = _setup_mock_environment(tmp_path)

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: env["dataset_dir"]
    dlg._models_dir = lambda: env["models_dir"]
    dlg._find_train_artifact_run_dir = lambda: env["rtx_run"]

    # Put dialog in Step 5 with stale flag set
    dlg._pipeline["frozen"] = True
    dlg._pipeline["model_stale"] = True
    dlg._pipeline["stale_reason"] = "Kare silindi"
    dlg._go_step(5)

    assert dlg.btn_val_done.isEnabled() is True

    # User clicks validation complete
    dlg._finish_validate()

    # Must be on Step 6
    assert dlg._pipeline["step"] == 6
    assert dlg._pipeline["validated"] is True
    assert dlg._pipeline["trained"] is True
    assert dlg._pipeline["model_stale"] is False

    # Stale manifest must be removed
    stale_file = env["models_dir"] / "stale_manifest.json"
    assert not stale_file.is_file()

    # Step 6 publish button must be enabled
    assert dlg.btn_publish.isEnabled() is True


def test_publish_model_in_step6_succeeds(tmp_path: Path) -> None:
    """Publishing model in Step 6 successfully deploys best.pt and ONNX, setting published=True."""
    env = _setup_mock_environment(tmp_path)

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: env["dataset_dir"]
    dlg._models_dir = lambda: env["models_dir"]
    dlg._find_train_artifact_run_dir = lambda: env["rtx_run"]

    # Validate and go to step 6
    dlg._pipeline["frozen"] = True
    dlg._go_step(5)
    dlg._finish_validate()
    assert dlg._pipeline["step"] == 6

    # Execute publish
    dlg._publish_model()

    assert dlg._pipeline["published"] is True
    assert (env["models_dir"] / "best.pt").is_file()
    assert (env["models_dir"] / "yolov8n_gatherion.onnx").is_file()
    assert dlg._pipeline["step"] == 6  # Does not redirect to step 4!


def test_finish_validate_blocks_when_no_weights(tmp_path: Path) -> None:
    """When no weights exist, finish_validate redirects to Step 4."""
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: dataset_dir
    dlg._models_dir = lambda: models_dir
    dlg._find_train_artifact_run_dir = lambda: None

    dlg._pipeline["frozen"] = True
    dlg._go_step(5)

    dlg._finish_validate()
    # Must redirect to Step 4 because there are no weights to validate
    assert dlg._pipeline["step"] == 4
