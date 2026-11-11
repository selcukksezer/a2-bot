# -*- coding: utf-8 -*-
"""Gates: hydrate must not mark trained; fake/fast/OpenCV results must be refused."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

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
    # B-02: the mAP50 publish gate (MIN_PUBLISH_MAP50 = 0.50) runs BEFORE
    # the fake-duration gate, so a fast low-mAP result is rejected for
    # mAP — not for its suspicious duration.
    assert ok2 is False, reason2
    assert "mAP50" in reason2 and "kabul eşiğinin" in reason2

    # The fake-duration ("1s") gate still fires once mAP is above the
    # gate: 300 epochs in 0.8s cannot be a real epoch loop.
    ok3, reason3 = YOLOFactoryDialog._accept_train_result(
        dlg,
        {
            "status": "completed",
            "mode": "ultralytics_cuda_fp16",
            "mAP50": 0.9,
            "epochs": 300,
            "epochs_completed": 300,
            "elapsed_sec": 0.8,
            "artifacts": {"ultralytics_best_pt": str(ROOT / "models" / "rtx_runs" / "gatherion_rtx_fp16" / "weights" / "best.pt")},
        },
    )
    assert ok3 is False, reason3
    assert "Şüpheli" in reason3 or "Sahte" in reason3


def _make_hydrate_stub(mod):
    """Build the minimal YOLOFactoryDialog stub used by the hydrate tests."""

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
    return dlg


def test_hydrate_does_not_set_trained(tmp_path: Path | None = None) -> None:
    from ui import yolo_factory_dialog as mod

    # Case A: a seeded in-memory result must be annotated with metrics_kind
    # and artifact_fresh, and must NEVER flip trained to True. All asserts
    # are unconditional — the old `if dlg._last_train_result:` guard let
    # this test pass vacuously when hydrate recovered nothing.
    dlg = _make_hydrate_stub(mod)
    dlg._last_train_result = {"status": "completed", "mode": "ultralytics_artifact", "mAP50": 0.5}
    mod.YOLOFactoryDialog._hydrate_last_train_result_from_artifacts(dlg)
    assert dlg._pipeline["trained"] is False
    assert dlg._last_train_result.get("mAP50") == 0.5
    assert dlg._last_train_result.get("metrics_kind") in ("stale_pre_freeze", "fresh_after_freeze")
    # No rtx run artifacts exist in this repo, so the result is stale vs
    # the freeze manifest — assert the exact value, not just membership.
    assert dlg._last_train_result.get("artifact_fresh") is False

    # Case B: the loader branch must merge recovered metrics and stamp the
    # same kind/freshness fields (loader stubbed to a fixed recovered dict so
    # the merge logic itself — not repo artifact state — is under test).
    dlg2 = _make_hydrate_stub(mod)
    dlg2._load_train_metrics_from_artifacts = lambda: {  # type: ignore[method-assign]
        "status": "completed",
        "mode": "ultralytics_artifact",
        "mAP50": 0.42,
        "epochs": 300,
    }
    mod.YOLOFactoryDialog._hydrate_last_train_result_from_artifacts(dlg2)
    assert dlg2._pipeline["trained"] is False
    assert dlg2._last_train_result.get("mAP50") == 0.42
    assert dlg2._last_train_result.get("metrics_kind") == "stale_pre_freeze"
    assert dlg2._last_train_result.get("artifact_fresh") is False

    # Case C: with no in-memory result and nothing recoverable, hydrate
    # must leave the result empty (and still not mark trained).
    dlg3 = _make_hydrate_stub(mod)
    dlg3._load_train_metrics_from_artifacts = lambda: {}  # type: ignore[method-assign]
    mod.YOLOFactoryDialog._hydrate_last_train_result_from_artifacts(dlg3)
    assert dlg3._pipeline["trained"] is False
    assert dlg3._last_train_result == {}


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

    # The staleness premise is asserted explicitly (unconditional) instead
    # of guarding the whole test with `if not fresh:` — the old form passed
    # vacuously whenever artifacts happened to be fresh.
    assert dlg._artifacts_are_fresh_vs_freeze() is False, (
        "Test premise broken: artifacts are fresh vs freeze, so reconcile "
        "would legitimately keep the trained flag"
    )
    mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg)
    assert dlg._pipeline["trained"] is False
    assert dlg._pipeline["validated"] is False
    assert dlg._pipeline["published"] is False
    assert any("ESKİ" in w for w in dlg._startup_warnings)


def test_reconcile_restores_trained_flag_when_artifacts_fresh() -> None:
    """Fresh artifacts + run weights must restore (not clear) the trained flag.

    B-03: fresh weights alone no longer restore `trained` — mAP evidence
    from the artifacts is required. The run dir is seeded hermetically
    (results.csv + weights/best.pt) so the real metrics loader supplies
    that evidence: the repo has no rtx run dir, and its detector JSON is
    Mode B (`custom_detector_json`), which never counts as Ultralytics
    evidence.
    """
    from ui import yolo_factory_dialog as mod

    class Stub(mod.YOLOFactoryDialog):
        def __init__(self) -> None:  # noqa: D107
            self._initialized = True

    def _reconcile_stub(base: Path, map50: float):
        run_dir = base / "models" / "rtx_runs" / "gatherion_rtx_fp16"
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights" / "best.pt").write_bytes(b"fake-weights")
        header = (
            "epoch,train/box_loss,train/cls_loss,metrics/precision(B),"
            "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        )
        rows = "".join(
            f"{i},0.1,0.1,0.9,0.9,{map50},0.5\n" for i in range(1, 301)
        )
        (run_dir / "results.csv").write_text(header + rows, encoding="utf-8")

        dlg = Stub.__new__(Stub)
        dlg._pipeline = {
            "trained": False,
            "validated": False,
            "published": False,
            "train_session_ok": False,
        }
        dlg._startup_warnings = []
        dlg._last_train_result = {}
        dlg._freeze_path = lambda: ROOT / "dataset" / "frozen_manifest.json"  # type: ignore[method-assign]
        dlg._find_train_artifact_run_dir = lambda: run_dir  # type: ignore[method-assign]
        dlg._freeze_mtime = mod.YOLOFactoryDialog._freeze_mtime.__get__(dlg, Stub)  # type: ignore
        dlg._artifact_mtime = mod.YOLOFactoryDialog._artifact_mtime.__get__(dlg, Stub)  # type: ignore
        # Force the fresh branch.
        dlg._artifacts_are_fresh_vs_freeze = lambda: True  # type: ignore[method-assign]
        dlg._models_dir = lambda: base / "models"  # type: ignore[method-assign]
        dlg.factory_controller = SimpleNamespace(rtx_trainer=None)
        return dlg

    with tempfile.TemporaryDirectory() as _td:
        base = Path(_td)
        # mAP50 above the publish gate: the flag is restored.
        dlg = _reconcile_stub(base, 0.62)
        mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg)
        assert dlg._pipeline["trained"] is True
        assert dlg._startup_warnings == []

        # B-03 negative: fresh weights but mAP50 below the gate — the
        # flag must NOT be restored, and a warning must explain why.
        dlg2 = _reconcile_stub(base, 0.32)
        mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg2)
        assert dlg2._pipeline["trained"] is False
        assert any("mAP50" in w for w in dlg2._startup_warnings)


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
        # A missing fixture must SKIP, not silently pass: the old bare
        # `return` made this test green without exercising anything.
        pytest.skip(f"dataset/data.yaml yok: {yaml_path}")
    # B-06 (dilim 27): the empty train/val gate runs before the
    # Ultralytics-missing fallback — the repo dataset has no labeled
    # samples, so the body raises loudly instead of returning a
    # failed dict about the missing Ultralytics install.
    with pytest.raises(RuntimeError, match="Eğitim verisi boş"):
        eng._run_training_sync_body(
            data_yaml_path=yaml_path,
            epochs=2,
            force_fallback=False,
            allow_opencv_fallback=False,
            model="yolo11n.pt",
        )


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


def _seed_p111_run(base: Path, rows: int, map50: float, epochs_req: int):
    """Seed a hermetic Ultralytics run dir: results.csv + best.pt + args.yaml.

    Freeze is faked at t=1000, run weights at t=2000 (fresh after freeze).
    """
    dataset_dir = base / "dataset"
    models_dir = base / "models"
    run_dir = models_dir / "rtx_runs" / "gatherion_rtx_fp16"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    freeze_file = dataset_dir / "frozen_manifest.json"
    freeze_file.write_text(
        json.dumps(
            {
                "version": "dataset-v4",
                "frozen_at": "1970-01-01T00:16:40",
                "total_frames": 10,
            }
        ),
        encoding="utf-8",
    )
    os.utime(freeze_file, (1000.0, 1000.0))
    best_pt = run_dir / "weights" / "best.pt"
    best_pt.write_bytes(b"fake-weights")
    os.utime(best_pt, (2000.0, 2000.0))
    header = (
        "epoch,train/box_loss,train/cls_loss,metrics/precision(B),"
        "metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
    )
    body = "".join(
        f"{i},0.1,0.1,0.9,0.9,{map50},0.5\n" for i in range(1, rows + 1)
    )
    csv_path = run_dir / "results.csv"
    csv_path.write_text(header + body, encoding="utf-8")
    os.utime(csv_path, (2000.0, 2000.0))
    (run_dir / "args.yaml").write_text(
        f"task: detect\nmode: train\nepochs: {epochs_req}\npatience: 15\n",
        encoding="utf-8",
    )
    return {"dataset_dir": dataset_dir, "models_dir": models_dir, "run_dir": run_dir}


def _make_validate_stub(mod, env, last_result):
    """Minimal stub with the REAL _finish_validate/_reconcile helpers bound."""

    class Stub(mod.YOLOFactoryDialog):
        def __init__(self) -> None:  # noqa: D107
            self._initialized = True

    dlg = Stub.__new__(Stub)
    dlg._pipeline = {
        "step": 5,
        "trained": False,
        "validated": False,
        "published": False,
        "frozen": True,
        "model_stale": False,
        "stale_reason": "",
        "stale_at": "",
        "train_session_ok": False,
        "train_completed_at": "",
    }
    dlg._startup_warnings = []
    dlg._last_train_result = dict(last_result or {})
    dlg._logs: list = []
    dlg._append_log = lambda msg: dlg._logs.append(msg)  # type: ignore[method-assign]
    dlg._go_step = lambda s: dlg._pipeline.__setitem__("step", int(s))  # type: ignore[method-assign]
    dlg._mark_step_done = lambda s: None  # type: ignore[method-assign]
    dlg._save_pipeline_disk_state = lambda: None  # type: ignore[method-assign]
    dlg._dataset_dir = lambda: env["dataset_dir"]  # type: ignore[method-assign]
    dlg._models_dir = lambda: env["models_dir"]  # type: ignore[method-assign]
    dlg._freeze_path = lambda: env["dataset_dir"] / "frozen_manifest.json"  # type: ignore[method-assign]
    dlg._find_train_artifact_run_dir = lambda: env["run_dir"]  # type: ignore[method-assign]
    dlg._freeze_mtime = mod.YOLOFactoryDialog._freeze_mtime.__get__(dlg, Stub)  # type: ignore
    dlg._artifact_mtime = mod.YOLOFactoryDialog._artifact_mtime.__get__(dlg, Stub)  # type: ignore
    dlg._artifacts_are_fresh_vs_freeze = mod.YOLOFactoryDialog._artifacts_are_fresh_vs_freeze.__get__(dlg, Stub)  # type: ignore
    dlg._read_ultralytics_metrics = mod.YOLOFactoryDialog._read_ultralytics_metrics.__get__(dlg, Stub)  # type: ignore
    dlg._load_train_metrics_from_artifacts = mod.YOLOFactoryDialog._load_train_metrics_from_artifacts.__get__(dlg, Stub)  # type: ignore
    dlg.factory_controller = SimpleNamespace(rtx_trainer=None)
    return dlg


def test_rejected_training_clears_last_train_result(tmp_path: Path | None = None) -> None:
    """P1-11 (K-26): a rejected run must clear _last_train_result (memory + disk).

    Run A was accepted (mAP50 0.80, 100/100) and sits in memory; run B is
    rejected (mAP50 0.32). After _on_training_finished(B), no trace of A
    may survive — otherwise _finish_validate would pass B's weights with
    A's metrics. The real _save_pipeline_disk_state runs against a tmp
    state file, so the disk snapshot is asserted too.
    """
    from ui import yolo_factory_dialog as mod

    with tempfile.TemporaryDirectory() as _td:
        state_path = Path(_td) / "pipeline_ui_state.json"

        class Stub(mod.YOLOFactoryDialog):
            def __init__(self) -> None:  # noqa: D107
                self._initialized = True

        dlg = Stub.__new__(Stub)
        dlg._pipeline = {
            "step": 4,
            "trained": True,
            "validated": True,
            "published": False,
            "scanned": True,
            "imbalance_fixed": False,
            "frozen": True,
            "model_stale": False,
            "stale_reason": "",
            "stale_at": "",
            "train_completed_at": "2026-11-10T10:00:00",
            "train_session_ok": True,
        }
        dlg._last_train_result = {
            "status": "completed",
            "mode": "ultralytics_cuda_fp16",
            "mAP50": 0.80,
            "epochs": 100,
            "epochs_completed": 100,
            "early_stopped": False,
            "artifacts": {"run_dir": str(Path(_td) / "run_A")},
        }
        dlg.hp_epoch = SimpleNamespace(value=lambda: 100)
        dlg._train_click_mono = 0.0
        # P2-18: _on_training_finished/_on_training_failed kapanış bayrağını
        # (getattr(self, "_yolo_closing", False)) okur; gerçek dialog bunu
        # __init__'te kurar. __new__ stub'unda eksik olunca PyQt, eksik
        # nitelik okumasını AttributeError yerine RuntimeError ile
        # patlatır ("super-class __init__() ... never called") — stub
        # bayrağı açıkça kurmalıdır (F3: v17 birleşim regresyonu).
        dlg._yolo_closing = False
        dlg._logs: list = []
        dlg._append_log = lambda msg: dlg._logs.append(msg)  # type: ignore[method-assign]
        dlg._update_train_lock = lambda: None  # type: ignore[method-assign]
        dlg.btn_stop_train = SimpleNamespace(setEnabled=lambda v: None)
        dlg.lbl_train_status = SimpleNamespace(setText=lambda t: None)
        dlg._pipeline_state_path = lambda: state_path  # type: ignore[method-assign]

        mod.YOLOFactoryDialog._on_training_finished(
            dlg,
            {
                "status": "completed",
                "mode": "ultralytics_cuda_fp16",
                "mAP50": 0.32,
                "epochs": 100,
                "epochs_completed": 33,
                "elapsed_sec": 600.0,
                "artifacts": {"ultralytics_best_pt": ""},
            },
        )
        assert dlg._last_train_result == {}
        assert dlg._pipeline["trained"] is False
        assert dlg._pipeline["validated"] is False
        assert dlg._pipeline["train_session_ok"] is False
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        assert saved["last_train_metrics"] == {}
        assert saved["trained"] is False


def test_failed_training_clears_last_train_result() -> None:
    """P1-11 (K-26), same hole on the failure signal path."""
    from ui import yolo_factory_dialog as mod

    class Stub(mod.YOLOFactoryDialog):
        def __init__(self) -> None:  # noqa: D107
            self._initialized = True

    dlg = Stub.__new__(Stub)
    dlg._pipeline = {"trained": True, "validated": True, "train_session_ok": True}
    dlg._last_train_result = {"mAP50": 0.80, "epochs_completed": 100}
    # P2-18: kapanış bayrağı — bkz. test_rejected_training_clears_last_train_result
    # (eksik nitelik, __new__ stub'unda RuntimeError verir).
    dlg._yolo_closing = False
    dlg._update_train_lock = lambda: None  # type: ignore[method-assign]
    dlg.btn_stop_train = SimpleNamespace(setEnabled=lambda v: None)
    dlg.lbl_train_status = SimpleNamespace(setText=lambda t: None)
    dlg._append_log = lambda msg: None  # type: ignore[method-assign]
    dlg._save_pipeline_disk_state = lambda: None  # type: ignore[method-assign]
    mod.YOLOFactoryDialog._on_training_failed(dlg, "CUDA out of memory")
    assert dlg._last_train_result == {}
    assert dlg._pipeline["trained"] is False


def test_finish_validate_reads_metrics_from_artifacts() -> None:
    """P1-11 (K-26): _finish_validate must judge the ARTIFACT run, not memory.

    Memory holds run A (mAP50 0.80, 100/100 — would pass every gate);
    the newest artifacts are run B (33/100 rows, mAP50 0.80, args.yaml
    says 100 requested). The epoch gate must RED on B's artifact values
    and send the dialog back to Step 4.
    """
    from ui import yolo_factory_dialog as mod

    with tempfile.TemporaryDirectory() as _td:
        base = Path(_td)
        env = _seed_p111_run(base, rows=33, map50=0.80, epochs_req=100)
        dlg = _make_validate_stub(
            mod,
            env,
            {
                "status": "completed",
                "mode": "ultralytics_cuda_fp16",
                "mAP50": 0.80,
                "epochs": 100,
                "epochs_completed": 100,
                "early_stopped": False,
                "artifacts": {"run_dir": str(base / "run_A")},
            },
        )
        assert dlg._artifacts_are_fresh_vs_freeze() is True
        mod.YOLOFactoryDialog._finish_validate(dlg)
        assert dlg._pipeline["step"] == 4
        assert dlg._pipeline["validated"] is False
        assert dlg._pipeline["trained"] is False
        assert any("yarım eğitim" in log for log in dlg._logs)

        # Positive control: a full 100/100 artifact run validates to Step 6.
        env2 = _seed_p111_run(base / "full", rows=100, map50=0.80, epochs_req=100)
        dlg2 = _make_validate_stub(mod, env2, {})
        mod.YOLOFactoryDialog._finish_validate(dlg2)
        assert dlg2._pipeline["step"] == 6
        assert dlg2._pipeline["validated"] is True
        assert dlg2._pipeline["trained"] is True


def test_artifacts_fresh_requires_weight_files() -> None:
    """P1-11 (K-26 / D020-K1): a fresh results.csv alone must not be fresh.

    Run dir with only a fresh results.csv (no best.pt / best.onnx) must
    yield fresh=False; adding a fresh best.pt flips it to True.
    """
    from ui import yolo_factory_dialog as mod

    with tempfile.TemporaryDirectory() as _td:
        base = Path(_td)
        env = _seed_p111_run(base, rows=10, map50=0.80, epochs_req=100)
        (env["run_dir"] / "weights" / "best.pt").unlink()
        dlg = _make_validate_stub(mod, env, {})
        assert dlg._artifacts_are_fresh_vs_freeze() is False
        best_pt = env["run_dir"] / "weights" / "best.pt"
        best_pt.write_bytes(b"fake-weights")
        os.utime(best_pt, (2000.0, 2000.0))
        assert dlg._artifacts_are_fresh_vs_freeze() is True


def test_reconcile_reads_model_stale_flag() -> None:
    """P1-11 (K-26): _reconcile must honour the model_stale flag (memory + disk).

    Fresh artifacts + mAP evidence would otherwise restore trained=True
    even though the model is marked stale.
    """
    from ui import yolo_factory_dialog as mod

    with tempfile.TemporaryDirectory() as _td:
        base = Path(_td)
        env = _seed_p111_run(base, rows=100, map50=0.80, epochs_req=100)

        # Case A: in-memory pipeline flag alone blocks the restore.
        dlg = _make_validate_stub(mod, env, {})
        dlg._pipeline.update(
            {"trained": True, "validated": True, "published": True, "model_stale": True,
             "stale_reason": "Kutu silindi"}
        )
        dlg._artifacts_are_fresh_vs_freeze = lambda: True  # type: ignore[method-assign]
        mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg)
        assert dlg._pipeline["trained"] is False
        assert dlg._pipeline["validated"] is False
        assert dlg._pipeline["published"] is False
        assert any("stale" in w.lower() for w in dlg._startup_warnings)

        # Case B: disk stale manifest alone blocks the restore.
        (env["models_dir"] / "stale_manifest.json").write_text(
            json.dumps({"model_stale": True, "reason": "Etiket güncellendi"}),
            encoding="utf-8",
        )
        dlg2 = _make_validate_stub(mod, env, {})
        dlg2._pipeline.update({"trained": True, "validated": True, "published": True})
        dlg2._artifacts_are_fresh_vs_freeze = lambda: True  # type: ignore[method-assign]
        mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg2)
        assert dlg2._pipeline["trained"] is False
        assert any("stale" in w.lower() for w in dlg2._startup_warnings)


def test_reconcile_blocks_short_high_map_run() -> None:
    """P1-11 (K-26): a short run with high mAP must not resurrect trained.

    33/100 epochs at mAP50 0.80, no in-memory result at all (the rejected
    run cleared it): the artifact-anchored epoch gate (args.yaml says 100)
    must keep trained=False.
    """
    from ui import yolo_factory_dialog as mod

    with tempfile.TemporaryDirectory() as _td:
        base = Path(_td)
        env = _seed_p111_run(base, rows=33, map50=0.80, epochs_req=100)
        dlg = _make_validate_stub(mod, env, {})
        dlg._pipeline.update({"trained": False, "validated": False, "published": False})
        dlg._artifacts_are_fresh_vs_freeze = lambda: True  # type: ignore[method-assign]
        mod.YOLOFactoryDialog._reconcile_trained_flag_with_artifacts(dlg)
        assert dlg._pipeline["trained"] is False


if __name__ == "__main__":
    test_accept_train_result_rejects_opencv_and_1s()
    test_hydrate_does_not_set_trained()
    test_reconcile_clears_stale_trained_flag()
    test_reconcile_restores_trained_flag_when_artifacts_fresh()
    test_opencv_fallback_disabled_returns_failed()
    test_stop_training_workflow()
    test_rejected_training_clears_last_train_result()
    test_failed_training_clears_last_train_result()
    test_finish_validate_reads_metrics_from_artifacts()
    test_artifacts_fresh_requires_weight_files()
    test_reconcile_reads_model_stale_flag()
    test_reconcile_blocks_short_high_map_run()
    print("OK")
