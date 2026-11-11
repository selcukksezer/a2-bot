# -*- coding: utf-8 -*-
"""tests/test_step5_validate_to_step6_publish.py
=============================================
Unit tests verifying Step 5 (Doğrula) -> Step 6 (Yayınla) pipeline transition:
1. Valid fresh weights allow advancing from Step 5 to Step 6 without bouncing back to Step 4.
2. 'Doğrulama tamam — Yayınla adımına geç' clears stale flags and sets validated=True, trained=True.
3. In Step 6, checklist items are checked and btn_publish is enabled.
4. Publishing succeeds and updates published=True.

The run artifacts are REAL, not mock bytes: best.onnx is a byte copy of the
repo's genuine training export (runs/detect/.../weights/best.onnx) and the
seeded live model is a byte copy of the repo's real published model
(models/yolov8n_gatherion.onnx). Publish assertions compare published bytes
against those real sources, and assert no best.pt was conjured out of thin
air (no .pt checkpoint exists anywhere in the repo, and this venv has no
torch/ultralytics to produce or validate one, so the test does not pretend
otherwise — it exercises the ONNX publish path).

Dialogs are REAL YOLOFactoryDialog instances. The tmp dataset is wired in
through the dialog's real dependency object (dlg._builtin_helper.dataset_dir),
never by replacing dlg._dataset_dir/_models_dir with lambdas. The only
module-level patch is rtx_runs_dir_for (the drive-letter path helper the
production run finder calls), pointed at the tmp rtx_runs tree so the REAL
_find_train_artifact_run_dir/_candidate_rtx_run_dirs code runs end to end.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

import ui.yolo_factory_dialog as yolo_factory_dialog_mod
from ui.yolo_factory_dialog import YOLOFactoryDialog
from dataset_factory.dataset_validator import mark_model_stale

# Genuine training artifacts shipped in the repo — the only real weights
# available in this environment (no torch/ultralytics to train or export).
REPO_RUN_ONNX = (
    PROJECT_ROOT
    / "runs"
    / "detect"
    / "models"
    / "rtx_runs"
    / "gatherion_rtx_fp16"
    / "weights"
    / "best.onnx"
)
REPO_LIVE_ONNX = PROJECT_ROOT / "models" / "yolov8n_gatherion.onnx"


@pytest.fixture(scope="module")
def shared_dialog() -> Tuple[YOLOFactoryDialog, Dict[str, Any]]:
    """One real dialog for the whole module, with a pipeline snapshot.

    YOLOFactoryDialog construction costs ~3-7 s (it loads the repo's
    on-disk pipeline state); the four tests here each built a fresh one
    for a few seconds of assertions. The dialog is re-wired per test by
    `_make_dialog` (tmp dataset/model dirs + per-test monkeypatch of
    `rtx_runs_dir_for`), and the pipeline dict is restored from the
    snapshot taken right after construction, so no test can leak
    `published`/`validated`/step state into the next one. The dialog is
    closed once, at module teardown.
    """
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    snapshot = copy.deepcopy(dlg._pipeline)
    yield dlg, snapshot
    dlg.close()


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

    # 2. REAL weights + results.csv at t = 2000.0 (fresh after freeze).
    # best.onnx is a byte copy of the repo's genuine Ultralytics export;
    # the live model is a byte copy of the repo's real published ONNX, so
    # publish can back it up instead of publishing from fake bytes.
    assert REPO_RUN_ONNX.is_file(), f"real run artifact missing: {REPO_RUN_ONNX}"
    assert REPO_LIVE_ONNX.is_file(), f"real live model missing: {REPO_LIVE_ONNX}"
    best_onnx = weights_dir / "best.onnx"
    shutil.copy2(REPO_RUN_ONNX, best_onnx)
    os.utime(best_onnx, (2000.0, 2000.0))

    live_onnx = models_dir / "yolov8n_gatherion.onnx"
    shutil.copy2(REPO_LIVE_ONNX, live_onnx)

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
        "best_onnx": best_onnx,
        "live_onnx": live_onnx,
        "results_csv": results_csv,
    }


def _make_dialog(
    env: Dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    shared: Tuple[YOLOFactoryDialog, Dict[str, Any]] | None = None,
) -> YOLOFactoryDialog:
    """Build (or re-wire the shared) real dialog at the tmp dataset through real code paths.

    - dlg._builtin_helper.dataset_dir is the attribute _dataset_dir() reads,
      so _models_dir() derives the tmp models dir by its own tmp/pytest rule.
    - rtx_runs_dir_for is the module-level helper _candidate_rtx_run_dirs()
      calls, so the real _find_train_artifact_run_dir scoring runs.
    - Construction loads the repo's on-disk pipeline state, whose
      _last_train_result still describes the repo's old F:\\a2 run
      (artifact_fresh=False). The tmp fixture replaces that run, so the
      stale cached result is dropped and _run_validate re-hydrates from the
      tmp artifacts instead of judging freshness from the old run.
    - When `shared` is given, the module-scoped dialog is reused: its
      pipeline is restored from the construction-time snapshot first, so
      each test starts from the same pipeline state a fresh dialog would
      have, without paying the ~3-7 s construction cost again.
    """
    if shared is None:
        dlg = YOLOFactoryDialog()
        dlg._auto_confirm_dialogs = True
    else:
        dlg, snapshot = shared
        dlg._pipeline.clear()
        dlg._pipeline.update(copy.deepcopy(snapshot))
    dlg._builtin_helper.dataset_dir = env["dataset_dir"]
    # [O3]: _dataset_dir() prefers the bound factory_controller (D29 b55c0ef);
    # the tmp wiring must redirect it too, or the dialog keeps the repo dataset.
    _ctrl = getattr(dlg, "factory_controller", None)
    if _ctrl is not None and hasattr(_ctrl, "dataset_dir"):
        _ctrl.dataset_dir = env["dataset_dir"]
    assert dlg._dataset_dir() == env["dataset_dir"]
    assert dlg._models_dir() == env["models_dir"]
    monkeypatch.setattr(
        yolo_factory_dialog_mod,
        "rtx_runs_dir_for",
        lambda: env["models_dir"] / "rtx_runs",
    )
    assert dlg._find_train_artifact_run_dir() == env["rtx_run"] or env.get("rtx_run") is None
    dlg._last_train_result = {}
    return dlg


def test_artifacts_fresh_vs_freeze_ignores_model_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_dialog: Tuple[YOLOFactoryDialog, Dict[str, Any]]
) -> None:
    """_artifacts_are_fresh_vs_freeze returns True when weights are newer than freeze, even if model is marked stale."""
    env = _setup_mock_environment(tmp_path)

    dlg = _make_dialog(env, monkeypatch, shared_dialog)
    # Mark model stale
    mark_model_stale(
        dataset_dir=env["dataset_dir"],
        models_dir=env["models_dir"],
        reason="Kare silindi",
    )
    dlg._pipeline["model_stale"] = True

    # Must still recognize artifacts are fresh
    assert dlg._artifacts_are_fresh_vs_freeze() is True


def test_finish_validate_advances_to_step6_and_enables_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_dialog: Tuple[YOLOFactoryDialog, Dict[str, Any]]
) -> None:
    """Clicking 'Doğrulama tamam — Yayınla adımına geç' must advance to Step 6, clear stale, and enable btn_publish."""
    env = _setup_mock_environment(tmp_path)

    dlg = _make_dialog(env, monkeypatch, shared_dialog)
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


def test_publish_model_in_step6_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_dialog: Tuple[YOLOFactoryDialog, Dict[str, Any]]
) -> None:
    """Publishing model in Step 6 deploys the REAL run ONNX, backing up the previous live model, setting published=True."""
    env = _setup_mock_environment(tmp_path)

    dlg = _make_dialog(env, monkeypatch, shared_dialog)
    # Validate and go to step 6
    dlg._pipeline["frozen"] = True
    dlg._go_step(5)
    dlg._finish_validate()
    assert dlg._pipeline["step"] == 6

    # Execute publish
    dlg._publish_model()

    assert dlg._pipeline["published"] is True
    # Published ONNX must be byte-identical to the real run artifact,
    # not a copy of fake weights.
    published_onnx = env["models_dir"] / "yolov8n_gatherion.onnx"
    assert published_onnx.is_file()
    assert published_onnx.read_bytes() == env["best_onnx"].read_bytes()
    # Previous live model must be backed up byte-identically. (Compare
    # against the repo source: env["live_onnx"] is the destination path
    # itself, which publish just overwrote.)
    backup_onnx = env["models_dir"] / "yolov8n_gatherion_prev.onnx"
    assert backup_onnx.is_file()
    assert backup_onnx.read_bytes() == REPO_LIVE_ONNX.read_bytes()
    # No best.pt was conjured: there is no real .pt checkpoint here.
    assert not (env["models_dir"] / "best.pt").is_file()
    assert dlg._pipeline["step"] == 6  # Does not redirect to step 4!


def test_finish_validate_blocks_when_no_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shared_dialog: Tuple[YOLOFactoryDialog, Dict[str, Any]]
) -> None:
    """When no weights exist, finish_validate redirects to Step 4."""
    dataset_dir = tmp_path / "dataset"
    models_dir = tmp_path / "models"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    env = {"dataset_dir": dataset_dir, "models_dir": models_dir, "rtx_run": None}
    dlg = _make_dialog(env, monkeypatch, shared_dialog)
    assert dlg._find_train_artifact_run_dir() is None

    dlg._pipeline["frozen"] = True
    dlg._go_step(5)

    assert dlg.btn_val_done.isEnabled() is False

    dlg._finish_validate()
    # Must redirect to Step 4 because there are no weights to validate
    assert dlg._pipeline["step"] == 4
