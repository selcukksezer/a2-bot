# -*- coding: utf-8 -*-
"""
P2-18 (bölüm 4.1) hedefli testler — eğitim + bootstrap worker kapanış ağı.

Kilitlenen davranış (taban: feat/all-combined-v16 @ 3e739f7):
- `RTXTrainerEngine.wait_for_training` süresiz join yapmaz:
  `timeout=None` varsayılan `RTX_TRAIN_WAIT_TIMEOUT_S` (3600 sn) sayılır;
  süre dolarsa dönen sözlük `timed_out=True` taşır.
- TEK kapanış deseni `_stop_qthread_gracefully`: işbirlikçi `stop()` →
  `requestInterruption()` → `quit()` → sonucu kontrol edilen
  `wait(timeout)` → durmadıysa `terminate()` + tekrar `wait()` + log.
  Düz `QThread`'te `stop()` yoktur — adım yalnızca `stop()` tanımlı
  hedefte çalışır.
- `_BackgroundCallWorker` / `_YoloDataLoadWorker` / `RTXTrainingWorker`
  işbirlikçi durdurma bayrağı taşır: kapanış isteği sonrası `run()`
  işi çalıştırmaz, `failed` sinyaliyle döner.
- `YOLOFactoryDialog._stop_yolo_threads`, closeEvent/`__del__`/atexit
  için aynı deseni eğitim + bootstrap + küçük resim + yükleme +
  toplu onay + zehir temizliği thread'lerinin hepsine uygular.
- atexit artık kendi `stop()` + yok sayılan `wait(5000)` kopyasını
  kullanmaz — `_stop_all_yolo_background_threads` aynı desene gider.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dataset_factory.rtx_trainer_engine as rtx_mod
from dataset_factory.rtx_trainer_engine import (
    RTX_TRAIN_WAIT_TIMEOUT_S,
    RTXTrainerEngine,
)

# Headless QApplication (ui.yolo_factory_dialog importu + QThread testleri).
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import (
    RTXTrainingWorker,
    YOLOFactoryDialog,
    _BackgroundCallWorker,
    _YoloDataLoadWorker,
    _YOLO_BACKGROUND_THREADS,
    _stop_all_yolo_background_threads,
    _stop_qthread_gracefully,
)


# ---------------------------------------------------------------------------
# 1. RTXTrainerEngine.wait_for_training
# ---------------------------------------------------------------------------

def _engine(tmp_path: Path) -> RTXTrainerEngine:
    return RTXTrainerEngine(
        project_root=tmp_path,
        dataset_dir=tmp_path / "dataset",
        models_dir=tmp_path / "models",
    )


def test_train_wait_timeout_constant_is_3600_seconds() -> None:
    assert RTX_TRAIN_WAIT_TIMEOUT_S == 3600.0


def test_wait_for_training_without_thread_is_not_timed_out(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    assert engine._training_thread is None
    res = engine.wait_for_training(timeout=1.0)
    assert res["timed_out"] is False
    assert res["state"] == "idle"


def test_wait_for_training_live_thread_times_out_quickly(tmp_path: Path) -> None:
    """Süresiz join yok: canlı thread + kısa timeout hızlı döner."""
    engine = _engine(tmp_path)
    t = threading.Thread(target=lambda: time.sleep(30.0), daemon=True)
    t.start()
    engine._training_thread = t
    t0 = time.monotonic()
    res = engine.wait_for_training(timeout=0.05)
    dt = time.monotonic() - t0
    assert res["timed_out"] is True
    assert dt < 5.0, f"wait_for_training {dt:.2f} sn sürdü — sonlu timeout yok?"
    assert t.is_alive(), "test thread'i hâlâ uyuyor olmalı (join beklememeli)"


def test_wait_for_training_none_means_default_timeout_not_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`timeout=None` artık süresiz demek değil: varsayılan sabite gider."""
    engine = _engine(tmp_path)
    monkeypatch.setattr(rtx_mod, "RTX_TRAIN_WAIT_TIMEOUT_S", 0.05)
    t = threading.Thread(target=lambda: time.sleep(30.0), daemon=True)
    t.start()
    engine._training_thread = t
    t0 = time.monotonic()
    res = engine.wait_for_training()
    dt = time.monotonic() - t0
    assert res["timed_out"] is True
    assert dt < 5.0, f"timeout=None {dt:.2f} sn sürdü — süresiz join geri gelmiş?"


def test_wait_for_training_finished_thread_is_not_timed_out(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    t = threading.Thread(target=lambda: time.sleep(0.05))
    t.start()
    engine._training_thread = t
    res = engine.wait_for_training(timeout=5.0)
    assert res["timed_out"] is False
    assert not t.is_alive()
    assert res["state"] == "idle"


def test_wait_for_training_returns_status_snapshot(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine._latest_status["state"] = "running"
    engine._latest_status["epoch"] = 3
    res = engine.wait_for_training(timeout=1.0)
    assert res["state"] == "running"
    assert res["epoch"] == 3
    assert res["timed_out"] is False


# ---------------------------------------------------------------------------
# 2. _stop_qthread_gracefully — tek kapanış deseni
# ---------------------------------------------------------------------------

def test_stop_qthread_none_and_never_started() -> None:
    assert _stop_qthread_gracefully(None, "yok") is True
    never_started = QThread()
    assert _stop_qthread_gracefully(never_started, "hiç başlamamış") is True


def test_stop_qthread_event_loop_thread_stops_cooperatively() -> None:
    """quit() ile duran thread: terminate'e düşmeden True döner."""
    thread = QThread()
    thread.start()
    assert thread.isRunning()
    assert _stop_qthread_gracefully(thread, "olay döngüsü", timeout_ms=2000) is True
    assert not thread.isRunning()
    assert thread.isFinished()


class _StubbornThread(QThread):
    """quit()'u yok sayan thread: run() olay döngüsüne girmez."""

    def run(self) -> None:
        while True:
            time.sleep(0.05)


def test_stop_qthread_stubborn_thread_falls_back_to_terminate() -> None:
    """Durmayan thread: kontrollü wait False → terminate() + tekrar wait."""
    thread = _StubbornThread()
    thread.start()
    assert thread.isRunning()
    assert _stop_qthread_gracefully(thread, "inatçı", timeout_ms=200) is True
    assert not thread.isRunning()
    assert thread.isFinished()


def test_stop_qthread_calls_worker_stop() -> None:
    """stop() tanımlı worker verilirse işbirlikçi stop() çağrılır."""
    calls: list[str] = []

    class _Worker:
        def stop(self) -> None:
            calls.append("stop")

    thread = QThread()
    thread.start()
    try:
        assert _stop_qthread_gracefully(
            thread, "worker'lı", timeout_ms=2000, worker=_Worker()
        ) is True
    finally:
        if thread.isRunning():
            thread.quit()
            thread.wait(2000)
    assert calls == ["stop"]


def test_stop_qthread_plain_thread_has_no_stop_to_call() -> None:
    """Düz QThread'te stop() yoktur — desen atlar, quit+wait ile durur."""
    thread = QThread()
    thread.start()
    assert not hasattr(thread, "stop")
    assert _stop_qthread_gracefully(thread, "düz", timeout_ms=2000) is True
    assert not thread.isRunning()


def test_stop_all_yolo_background_threads() -> None:
    """atexit yolu: kayıtlı thread'ler aynı desenle durdurulur."""
    thread = QThread()
    thread.start()
    _YOLO_BACKGROUND_THREADS.add(thread)
    assert thread.isRunning()
    _stop_all_yolo_background_threads()
    assert not thread.isRunning()
    assert thread.isFinished()


# ---------------------------------------------------------------------------
# 3. İşbirlikçi durdurma bayrakları (worker'lar)
# ---------------------------------------------------------------------------

def test_background_call_worker_stop_before_run_skips_fn() -> None:
    calls: list[str] = []
    failed: list[str] = []
    worker = _BackgroundCallWorker(lambda: calls.append("ran") or "ok")
    worker.failed.connect(failed.append)
    worker.stop()
    assert worker.is_stop_requested() is True
    worker.run()
    assert calls == [], "durdurma istenmiş worker fn'i çalıştırmamalı"
    assert failed == ["Veri seti taraması durduruldu."]


def test_background_call_worker_normal_run_emits_result() -> None:
    succeeded: list[Any] = []
    failed: list[str] = []
    worker = _BackgroundCallWorker(lambda: {"n": 7})
    worker.succeeded.connect(succeeded.append)
    worker.failed.connect(failed.append)
    assert worker.is_stop_requested() is False
    worker.run()
    assert succeeded == [{"n": 7}]
    assert failed == []


def test_background_call_worker_stop_after_run_result_is_dropped() -> None:
    """fn bitti ama kapanış istendiyse sonuç yayınlanmaz, failed döner."""
    succeeded: list[Any] = []
    failed: list[str] = []
    holder: Dict[str, Any] = {}

    def _slow() -> str:
        holder["w"].stop()
        return "done"

    worker2 = _BackgroundCallWorker(_slow)
    holder["w"] = worker2
    worker2.succeeded.connect(succeeded.append)
    worker2.failed.connect(failed.append)
    worker2.run()
    assert succeeded == []
    assert failed == ["Veri seti taraması durduruldu."]


def test_yolo_data_load_worker_stop_before_run(tmp_path: Path) -> None:
    failed: list[str] = []
    finished: list[Any] = []
    worker = _YoloDataLoadWorker(
        dataset_dir=tmp_path,
        rejected_stems=set(),
        review_audit_mode=False,
        queue_filter_mode="all",
        helper=None,
    )
    worker.failed.connect(failed.append)
    worker.finished.connect(lambda *a: finished.append(a))
    assert worker.is_stop_requested() is False
    worker.stop()
    assert worker.is_stop_requested() is True
    worker.run()
    assert failed == ["Yükleme durduruldu."]
    assert finished == []


def test_rtx_training_worker_stop_requests_and_propagates(tmp_path: Path) -> None:
    controller = MagicMock()
    worker = RTXTrainingWorker(controller=controller, epochs=5)
    assert worker.is_stop_requested() is False
    worker.stop()
    assert worker.is_stop_requested() is True
    controller.stop_training.assert_called_once()


def test_rtx_training_worker_run_after_stop_emits_failed() -> None:
    controller = MagicMock()
    worker = RTXTrainingWorker(controller=controller, epochs=5)
    failed: list[str] = []
    worker.training_failed.connect(failed.append)
    worker.stop()
    worker.run()
    assert failed == ["Eğitim kullanıcı tarafından durduruldu."]
    controller.stop_training.assert_called_once()


# ---------------------------------------------------------------------------
# 4. _stop_yolo_threads — kapanış ağı (dialog)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def factory_dialog(tmp_path_factory):
    dlg = YOLOFactoryDialog()
    base = tmp_path_factory.mktemp("p218_close")
    dlg._dataset_dir = lambda: base  # type: ignore[assignment]
    yield dlg
    try:
        # Yapım sırasında başlayan yükleme thread'i dahil her şeyi
        # TEK desenle durdur, sonra kapat (yetim thread bırakma).
        dlg._stop_yolo_threads()
        dlg.close()
    except Exception:
        pass


def test_stop_yolo_threads_stops_all_registered_threads(factory_dialog) -> None:
    """Eğitim + bootstrap + thumb + yükleme + bulk + purge tek düzende durur."""
    dlg = factory_dialog

    # ÖNCE yapım sırasında başlayan gerçek yükleme thread'ini durdur:
    # __dict__ anahtarları sahte thread'lerle ezilmeden önce, yoksa
    # gerçek thread yetim kalır (QThread destroyed-while-running).
    dlg._stop_yolo_threads()
    assert not dlg.__dict__.get("_yolo_load_thread").isRunning()

    def _running() -> QThread:
        t = QThread()
        t.start()
        assert t.isRunning()
        return t

    threads: Dict[str, QThread] = {
        "_training_worker": _running(),
        "_bootstrap_worker": _running(),
        "_thumb_thread": _running(),
        "_yolo_load_thread": _running(),
        "_bulk_approve_thread": _running(),
        "_purge_thread": _running(),
    }
    load_worker = SimpleNamespace(stop=MagicMock())
    bulk_worker = SimpleNamespace(stop=MagicMock())
    purge_worker = SimpleNamespace(stop=MagicMock())
    for key, thread in threads.items():
        dlg.__dict__[key] = thread
    dlg.__dict__["_yolo_load_worker"] = load_worker
    dlg.__dict__["_bulk_approve_worker"] = bulk_worker
    dlg.__dict__["_purge_worker"] = purge_worker

    dlg._stop_yolo_threads()

    for key, thread in threads.items():
        assert not thread.isRunning(), f"{key} kapanışta durmadı"
    load_worker.stop.assert_called_once()
    bulk_worker.stop.assert_called_once()
    purge_worker.stop.assert_called_once()

    # İkinci çağrı (closeEvent sonrası __del__/atexit tekrarı) güvenli no-op.
    dlg._stop_yolo_threads()
    for thread in threads.values():
        assert not thread.isRunning()
