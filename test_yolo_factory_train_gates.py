# -*- coding: utf-8 -*-
"""Gates: hydrate must not mark trained; fake/fast/OpenCV results must be refused."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_accept_train_result_rejects_opencv_and_1s() -> None:
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    dlg = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dlg.hp_epoch = SimpleNamespace(value=lambda: 300)
    dlg._train_click_mono = time.monotonic()
    dlg._pipeline = {"trained": False}

    ok, reason = YOLOFactoryDialog._accept_train_result(
        dlg,
        {"status": "completed", "mode": "opencv_centroid_prototype", "mAP50": 0.99, "epochs": 300, "elapsed_sec": 0.4},
    )
    assert ok is False, reason
    assert "OpenCV" in reason or "prototip" in reason.lower()

    ok2, reason2 = YOLOFactoryDialog._accept_train_result(
        dlg,
        {
            "status": "completed",
            "mode": "ultralytics_cuda_fp16",
            "mAP50": 0.2,
            "epochs": 300,
            "epochs_completed": 300,
            "elapsed_sec": 0.8,
            "artifacts": {"ultralytics_best_pt": str(ROOT / "models" / "rtx_runs" / "gatherion_rtx_fp16" / "weights" / "best.pt")},
        },
    )
    assert ok2 is False, reason2
    assert "Sahte" in reason2 or "Şüpheli" in reason2 or "tıklamasından" in reason2


def test_hydrate_does_not_set_trained(tmp_path: Path | None = None) -> None:
    from ui import yolo_factory_dialog as mod

    class Stub(mod.YOLOFactoryDialog):
        def __init__(self) -> None:  # noqa: D107
            self._initialized = True

    dlg = Stub.__new__(Stub)
    dlg._pipeline = {"trained": False, "frozen": True, "train_session_ok": False}
    dlg._last_train_result = {}
    dlg._dataset_dir = lambda: ROOT / "dataset"  # type: ignore[method-assign]
    dlg._models_dir = lambda: ROOT / "models"  # type: ignore[method-assign]
    dlg._freeze_path = lambda: ROOT / "dataset" / "frozen_manifest.json"  # type: ignore[method-assign]
    dlg._candidate_rtx_run_dirs = mod.YOLOFactoryDialog._candidate_rtx_run_dirs.__get__(dlg, Stub)  # type: ignore
    dlg._find_train_artifact_run_dir = mod.YOLOFactoryDialog._find_train_artifact_run_dir.__get__(dlg, Stub)  # type: ignore
    dlg._load_train_metrics_from_artifacts = mod.YOLOFactoryDialog._load_train_metrics_from_artifacts.__get__(dlg, Stub)  # type: ignore
    dlg._read_ultralytics_metrics = mod.YOLOFactoryDialog._read_ultralytics_metrics.__get__(dlg, Stub)  # type: ignore
    dlg.factory_controller = SimpleNamespace(rtx_trainer=None)
    dlg._freeze_mtime = mod.YOLOFactoryDialog._freeze_mtime.__get__(dlg, Stub)  # type: ignore
    dlg._artifact_mtime = mod.YOLOFactoryDialog._artifact_mtime.__get__(dlg, Stub)  # type: ignore
    dlg._artifacts_are_fresh_vs_freeze = mod.YOLOFactoryDialog._artifacts_are_fresh_vs_freeze.__get__(dlg, Stub)  # type: ignore

    mod.YOLOFactoryDialog._hydrate_last_train_result_from_artifacts(dlg)
    assert dlg._pipeline["trained"] is False
    # Existing morning results.csv should load as stale vs evening freeze.
    if dlg._last_train_result:
        assert dlg._last_train_result.get("metrics_kind") in (
            "stale_pre_freeze",
            "fresh_after_freeze",
        )
        if (ROOT / "dataset" / "frozen_manifest.json").is_file():
            freeze_m = dlg._freeze_mtime()
            art_m = dlg._artifact_mtime()
            if freeze_m > art_m:
                assert dlg._last_train_result.get("artifact_fresh") is False


def test_reconcile_clears_stale_trained_flag() -> None:
    from ui import yolo_factory_dialog as mod

    class Stub(mod.YOLOFactoryDialog):
        def __init__(self) -> None:  # noqa: D107
            self._initialized = True

    dlg = Stub.__new__(Stub)
    dlg._pipeline = {
        "trained": True,
        "validated": True,
        "published": True,
        "train_session_ok": False,
    }
    dlg._startup_warnings = []
    dlg._freeze_path = lambda: ROOT / "dataset" / "frozen_manifest.json"  # type: ignore[method-assign]
    dlg._find_train_artifact_run_dir = lambda: ROOT / "models" / "rtx_runs" / "gatherion_rtx_fp16"  # type: ignore
    dlg._freeze_mtime = mod.YOLOFactoryDialog._freeze_mtime.__get__(dlg, Stub)  # type: ignore
    dlg._artifact_mtime = mod.YOLOFactoryDialog._artifact_mtime.__get__(dlg, Stub)  # type: ignore
    dlg._artifacts_are_fresh_vs_freeze = mod.YOLOFactoryDialog._artifacts_are_fresh_vs_freeze.__get__(dlg, Stub)  # type: ignore

    if not dlg._artifacts_are_fresh_vs_freeze():
        mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg)
        assert dlg._pipeline["trained"] is False
        assert dlg._pipeline["validated"] is False
        assert dlg._pipeline["published"] is False
        assert any("ESKİ" in w for w in dlg._startup_warnings)


def test_opencv_fallback_disabled_returns_failed(monkeypatch=None) -> None:
    from dataset_factory.rtx_trainer_engine import RTXTrainerEngine

    eng = RTXTrainerEngine()
    # Bypass game/slot gates; exercise the body path with Ultralytics missing.
    eng.probe_gpu_environment = lambda: {  # type: ignore[method-assign]
        "ultralytics_installed": False,
        "cuda_available": False,
        "gpu_name": "CPU",
    }
    yaml_path = ROOT / "dataset" / "data.yaml"
    if not yaml_path.is_file():
        return
    result = eng._run_training_sync_body(
        data_yaml_path=yaml_path,
        epochs=2,
        force_fallback=False,
        allow_opencv_fallback=False,
        model="yolo11n.pt",
    )
    assert result.get("started") is False
    assert result.get("status") == "failed"
    reason = str(result.get("reason") or result.get("error") or "").lower()
    assert "ultralytics" in reason or "opencv" in reason


def test_stop_training_workflow() -> None:
    from ui.yolo_factory_dialog import YOLOFactoryDialog, RTXTrainingWorker
    from dataset_factory.rtx_trainer_engine import RTXTrainerEngine
    from dataset_factory.factory_controller import AutoLabelingDatasetFactory

    # 1. Test engine stop_training
    engine = RTXTrainerEngine()
    assert not engine._stop_event.is_set()
    engine.stop_training()
    assert engine._stop_event.is_set()

    # 2. Test controller stop_training
    ctrl = AutoLabelingDatasetFactory.__new__(AutoLabelingDatasetFactory)
    ctrl.rtx_trainer = engine
    ctrl.trainer = engine
    engine._stop_event.clear()
    ctrl.stop_training()
    assert engine._stop_event.is_set()

    # 3. Test RTXTrainingWorker stop
    worker = RTXTrainingWorker(controller=ctrl, epochs=10)
    engine._stop_event.clear()
    worker.stop()
    assert worker.is_stop_requested()
    assert engine._stop_event.is_set()

    # 4. Test _accept_train_result rejects stopped result
    dlg = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dlg.hp_epoch = SimpleNamespace(value=lambda: 10)
    dlg._train_click_mono = time.monotonic()
    dlg._pipeline = {"trained": False}
    ok, reason = YOLOFactoryDialog._accept_train_result(
        dlg,
        {
            "started": True,
            "status": "stopped",
            "state": "stopped",
            "stopped": True,
            "reason": "Eğitim kullanıcı tarafından durduruldu.",
        },
    )
    assert ok is False
    assert "durduruldu" in reason.lower()


if __name__ == "__main__":
    test_accept_train_result_rejects_opencv_and_1s()
    test_hydrate_does_not_set_trained()
    test_reconcile_clears_stale_trained_flag()
    test_opencv_fallback_disabled_returns_failed()
    test_stop_training_workflow()
    print("OK")
