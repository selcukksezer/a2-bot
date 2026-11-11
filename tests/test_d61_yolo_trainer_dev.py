# -*- coding: utf-8 -*-
"""
tests/test_d61_yolo_trainer_dev.py
===================================
DEV slice D61 tests for `dataset_factory/yolo_trainer.py` improvements:

1. `count_images_in_dir` via `os.scandir` — same counts as the old
   `iterdir()` + `Path.is_file()` implementation (incl. case, subdirs, non-dirs).
2. `classify_training_result` — every worker result branch, pure-function
   level (previously only reachable through a full threaded worker run).
3. `normalize_training_params` — single clamping point shared by the sync /
   async entry points and the worker (worker used to clamp, sync/async did not).
4. Sync/async prologue dedup (`_prepare_training_run`) — identical behaviour,
   including the preserved asymmetry: sync reports the empty-approved error
   via status_callback ("[HATA] ..."), async does not.
5. `stop_training` now logs a swallowed failure instead of dropping it silently.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pytest

from dataset_factory.yolo_trainer import (
    YOLOTrainer,
    YOLOTrainingWorker,
    _engine_supports_kwarg,
    _training_status_message,
    classify_training_result,
    count_images_in_dir,
    normalize_training_params,
)


def _add_label_for(image_path: Path) -> None:
    """Write the non-empty same-stem `.txt` label for a fixture image.

    `count_labeled_images` only counts images with a non-empty label
    (dilim 05 F9 / dilim 27 B-05); a bare .jpg counts as 0, so training
    never reaches the engine. The label lives in the sibling
    `labels/<split>/` tree.
    """
    labels_dir = image_path.parents[2] / "labels" / image_path.parent.name
    labels_dir.mkdir(parents=True, exist_ok=True)
    (labels_dir / f"{image_path.stem}.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n", encoding="utf-8"
    )


@pytest.fixture
def mock_dataset_env(tmp_path: Path):
    """Minimal isolated project/dataset structure (mirrors test_yolo_trainer)."""
    proj = tmp_path / "project"
    dataset = proj / "dataset"
    approved = dataset / "approved"
    for base in (dataset, approved):
        (base / "images" / "train").mkdir(parents=True, exist_ok=True)
        (base / "images" / "val").mkdir(parents=True, exist_ok=True)
    (dataset / "data.yaml").write_text(
        "path: .\ntrain: images/train\nval: images/val\nnc: 22\nnames: {0: herb_aria}\n",
        encoding="utf-8",
    )
    (dataset / "images" / "train" / "raw_1.jpg").write_bytes(b"img1")
    (dataset / "images" / "val" / "raw_2.jpg").write_bytes(b"img2")
    return proj, dataset, approved


# ---------------------------------------------------------------------------
# 1. count_images_in_dir (os.scandir)
# ---------------------------------------------------------------------------

def test_count_images_in_dir_counts_only_images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for name in ("a.jpg", "b.JPG", "c.png", "d.jpeg", "e.webp", "f.bmp", "g.tif", "h.TIFF"):
        (d / name).write_bytes(b"x")
    for name in ("i.txt", "j", "k.jpg.bak", "l.mp4"):
        (d / name).write_bytes(b"x")
    (d / "sub").mkdir()
    (d / "sub" / "nested.jpg").write_bytes(b"x")  # not recursive, like before
    assert count_images_in_dir(d) == 8


def test_count_images_in_dir_non_dir_returns_zero(tmp_path):
    assert count_images_in_dir(tmp_path / "missing") == 0
    f = tmp_path / "a_file.txt"
    f.write_text("x")
    assert count_images_in_dir(f) == 0


def test_count_images_in_dir_empty_dir(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert count_images_in_dir(d) == 0


# ---------------------------------------------------------------------------
# 2. classify_training_result — every branch
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "result, expected",
    [
        ({"waiting_for_game_exit": True},
         ("deferred", "Eğitim, oyun kapanana kadar bekliyor.",
          "Eğitim Durumu: Eğitim, oyun kapanana kadar bekliyor.")),
        ({"reason": "aion2_running"},
         ("deferred", "Eğitim, oyun kapanana kadar bekliyor.",
          "Eğitim Durumu: Eğitim, oyun kapanana kadar bekliyor.")),
        ({"stopped": True},
         ("failed", "Eğitim kullanıcı tarafından durduruldu.",
          "Eğitim Durumu: Durduruldu (Eğitim kullanıcı tarafından durduruldu.)")),
        ({"interrupted": True, "reason": "kullanıcı iptali"},
         ("failed", "kullanıcı iptali",
          "Eğitim Durumu: Durduruldu (kullanıcı iptali)")),
        ({"status": "stopped"},
         ("failed", "Eğitim kullanıcı tarafından durduruldu.",
          "Eğitim Durumu: Durduruldu (Eğitim kullanıcı tarafından durduruldu.)")),
        ({"state": "stopped", "reason": "r"},
         ("failed", "r", "Eğitim Durumu: Durduruldu (r)")),
        ({"started": False, "reason": "busy gpu"},
         ("failed", "busy gpu", None)),
        ({"status": "failed", "error": "CUDA out of memory"},
         ("failed", "CUDA out of memory", None)),
        ({"state": "refused"}, ("failed", "Eğitim başarısız.", None)),
        ({"state": "busy"}, ("failed", "Eğitim başarısız.", None)),
        (None, ("failed", "Eğitim sonucu boş/belirsiz — eğitim başarısız sayıldı.", None)),
        ({}, ("failed", "Eğitim sonucu boş/belirsiz — eğitim başarısız sayıldı.", None)),
        ("garbage", ("failed", "Eğitim sonucu boş/belirsiz — eğitim başarısız sayıldı.", None)),
        ({"mode": "ultralytics_cuda_fp16"},
         ("failed", "Eğitim sonucu başarısız sayıldı.", None)),
        ({"status": "success"}, ("finished", None, None)),
        ({"state": "completed"}, ("finished", None, None)),
        ({"started": True}, ("finished", None, None)),
    ],
)
def test_classify_training_result_branches(result, expected):
    assert classify_training_result(result) == expected


def test_classify_deferred_beats_stopped_when_both_present():
    # Branch order preserved from the old worker.run(): waiting check first.
    outcome, _, _ = classify_training_result(
        {"waiting_for_game_exit": True, "stopped": True}
    )
    assert outcome == "deferred"


# ---------------------------------------------------------------------------
# 3. normalize_training_params
# ---------------------------------------------------------------------------

def test_normalize_training_params_clamps_invalid():
    assert normalize_training_params(0, 100, 0, "  ") == (1, 320, 1, "yolo11n.pt")
    assert normalize_training_params(-5, -1, -3, "") == (1, 320, 1, "yolo11n.pt")
    assert normalize_training_params(10, 640, 16, "yolo11n.pt") == (10, 640, 16, "yolo11n.pt")
    assert normalize_training_params(10, 640, 16, None) == (10, 640, 16, "yolo11n.pt")
    assert normalize_training_params(3, 1280, 8, "  yolo11s.pt ") == (3, 1280, 8, "yolo11s.pt")


def test_worker_init_uses_shared_normalization():
    worker = YOLOTrainingWorker(epochs=0, imgsz=100, batch=0, model="  ")
    assert (worker.epochs, worker.imgsz, worker.batch, worker.model) == (1, 320, 1, "yolo11n.pt")


def test_training_status_message_exact_strings():
    assert _training_status_message(True, 3) == (
        "[EĞİTİM] Sadece onaylanan temiz veri seti ile eğitim başlatılıyor: 3 onaylı kare"
    )
    assert _training_status_message(False, 2) == (
        "[EĞİTİM] Tüm veri seti ile eğitim başlatılıyor: 2 kare"
    )


# ---------------------------------------------------------------------------
# 4. Sync/async entry points: clamping + preserved empty-approved asymmetry
# ---------------------------------------------------------------------------

def _trainer_with_approved(mock_dataset_env, engine):
    proj, dataset, approved = mock_dataset_env
    (approved / "images" / "train" / "appr_1.jpg").write_bytes(b"appr1")
    _add_label_for(approved / "images" / "train" / "appr_1.jpg")
    return YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=engine)


def test_run_training_sync_clamps_params_before_engine(mock_dataset_env):
    engine = MagicMock()
    engine.run_training_sync.return_value = {"status": "success", "started": True}
    trainer = _trainer_with_approved(mock_dataset_env, engine)
    trainer.run_training_sync(epochs=0, imgsz=100, batch=0, model="  ")
    call_kwargs = engine.run_training_sync.call_args[1]
    assert call_kwargs["epochs"] == 1
    assert call_kwargs["imgsz"] == 320
    assert call_kwargs["batch"] == 1
    assert call_kwargs["model"] == "yolo11n.pt"


def test_start_training_async_clamps_params_before_engine(mock_dataset_env):
    engine = MagicMock()
    engine.start_training_async.return_value = {"status": "success", "started": True}
    trainer = _trainer_with_approved(mock_dataset_env, engine)
    trainer.start_training_async(epochs=0, imgsz=100, batch=0, model="")
    call_kwargs = engine.start_training_async.call_args[1]
    assert call_kwargs["epochs"] == 1
    assert call_kwargs["imgsz"] == 320
    assert call_kwargs["batch"] == 1
    assert call_kwargs["model"] == "yolo11n.pt"


def test_empty_approved_sync_reports_error_status_async_does_not(mock_dataset_env):
    """Locks the preserved historical asymmetry (NOT a bug fix): sync emits
    '[HATA] ...' via status_callback, async does not; both return the same
    failure dict and neither calls the engine."""
    proj, dataset, approved = mock_dataset_env
    engine = MagicMock()
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=engine)

    sync_logs, async_logs = [], []
    sync_res = trainer.run_training_sync(epochs=5, only_approved=True,
                                         status_callback=sync_logs.append)
    async_res = trainer.start_training_async(epochs=5, only_approved=True,
                                             status_callback=async_logs.append)

    assert sync_res["started"] is False and sync_res["approved_count"] == 0
    assert async_res["started"] is False and async_res["approved_count"] == 0
    assert any("[HATA]" in s for s in sync_logs)
    assert not any("[HATA]" in s for s in async_logs)
    engine.run_training_sync.assert_not_called()
    engine.start_training_async.assert_not_called()


def test_f1987_warning_preserved_in_shared_prologue(mock_dataset_env, caplog):
    """F-1987 uyarısı korunur; P0-5 (K-30) yeni sözleşmesi: var olmayan
    custom yaml artık motora iletilmez — paylaşılan prologdaki yaml
    kapısı `failed` döndürür, mock motora hiç ulaşılmaz. Fixture'da
    `data_approved.yaml` yoktur ve dataset etiketsizdir, yani kapının
    engellemesi doğru davranıştır; bu testin konusu uyarı metnidir,
    eski `success` beklentisi kapı öncesi dönemin artığıdır."""
    proj, dataset, approved = mock_dataset_env
    engine = MagicMock()
    engine.run_training_sync.return_value = {"status": "success", "started": True}
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=engine)
    fake_yaml = dataset / "data_approved.yaml"  # does not exist
    with caplog.at_level("WARNING"):
        res = trainer.run_training_sync(
            epochs=5, only_approved=False, data_yaml_path=fake_yaml
        )
    assert "çağrıcının seçimi" in caplog.text
    assert res["status"] == "failed"
    assert res["started"] is False
    assert "eğitim yaml'ı yok" in res["error"]
    engine.run_training_sync.assert_not_called()


def test_engine_supports_kwarg():
    def with_flag(use_approved_only=True):
        pass

    def without_flag():
        pass

    class Engine:
        def run_training_sync(self, use_approved_only=False):
            pass

    assert _engine_supports_kwarg(Engine(), "run_training_sync", "use_approved_only") is True
    assert _engine_supports_kwarg(Engine(), "missing_method", "use_approved_only") is False
    assert _engine_supports_kwarg(with_flag, "__call__", "use_approved_only") is False
    assert _engine_supports_kwarg(without_flag, "__call__", "use_approved_only") is False
    assert _engine_supports_kwarg(MagicMock(), "run_training_sync", "use_approved_only") is False


# ---------------------------------------------------------------------------
# 5. Worker dispatch through the classifier + stop_training logging
# ---------------------------------------------------------------------------

def test_worker_run_deferred_and_ambiguous_via_classifier(mock_dataset_env):
    proj, dataset, approved = mock_dataset_env
    (approved / "images" / "train" / "appr_1.jpg").write_bytes(b"appr1")
    _add_label_for(approved / "images" / "train" / "appr_1.jpg")

    engine = MagicMock()
    engine.run_training_sync.return_value = {"waiting_for_game_exit": True}
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=engine)
    worker = YOLOTrainingWorker(trainer=trainer, epochs=5, only_approved=True)
    deferred, failed, status = [], [], []
    worker.training_deferred.connect(deferred.append)
    worker.training_failed.connect(failed.append)
    worker.status_text.connect(status.append)
    worker.run()
    assert deferred == ["Eğitim, oyun kapanana kadar bekliyor."]
    assert failed == []
    assert any(s.startswith("Eğitim Durumu:") for s in status)

    engine.run_training_sync.return_value = {"mode": "ultralytics_cuda_fp16"}
    worker2 = YOLOTrainingWorker(trainer=trainer, epochs=5, only_approved=True)
    failed2, finished2 = [], []
    worker2.training_failed.connect(failed2.append)
    worker2.training_finished.connect(finished2.append)
    worker2.run()
    assert failed2 == ["Eğitim sonucu başarısız sayıldı."]
    assert finished2 == []


def test_stop_training_logs_swallowed_failure(mock_dataset_env, caplog):
    proj, dataset, approved = mock_dataset_env
    engine = MagicMock()
    engine.stop_training.side_effect = RuntimeError("engine gone")
    trainer = YOLOTrainer(project_root=proj, dataset_dir=dataset, engine=engine)
    with caplog.at_level(logging.DEBUG, logger="DatasetFactory.YOLOTrainer"):
        trainer.stop_training()  # must not raise
    assert "stop_training failed" in caplog.text
