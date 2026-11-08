# -*- coding: utf-8 -*-
"""Start-training must return before dataset bootstrap, and bootstrap runs on another thread.

No real image scan. No ultralytics. No CUDA train.
"""

from __future__ import annotations

import os
import sys
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _empty_stats() -> dict:
    return {
        "total_frames": 0,
        "train_frames": 0,
        "val_frames": 0,
        "total_boxes": 0,
        "class_counts": {},
        "latest_preview_path": "",
        "gpu_environment": {"gpu_name": "TEST"},
    }


def _pump(app, predicate, timeout: float = 3.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())


def _bind_one_click(host, factory_mod):
    host.prepare_one_click_dataset = (
        factory_mod.AutoLabelingDatasetFactory.prepare_one_click_dataset.__get__(host, type(host))
    )
    host._schedule_one_click_rtx_training = (
        factory_mod.AutoLabelingDatasetFactory._schedule_one_click_rtx_training.__get__(
            host, type(host)
        )
    )


class _Host:
    def __init__(self, bootstrap):
        self.curator = self
        self.data_yaml_path = "dataset/data.yaml"
        self.rtx_trainer = self
        self._one_click_thread = None
        self._last_one_click_result = None
        self._bootstrap = bootstrap
        self.trained = False

    def ensure_dataset_structure(self):
        return {}

    def get_manifest_stats(self):
        return {"total_images": 0}

    def bootstrap_from_existing_frames(self, **kwargs):
        return self._bootstrap(**kwargs)

    def start_training_async(self, **kwargs):
        self.trained = True
        return {"started": True, "mode": "fake"}


def test_factory_async_returns_before_bootstrap() -> None:
    import dataset_factory.factory_controller as factory_mod

    finished = threading.Event()
    ids = {}

    def _bootstrap(**kwargs):
        ids["boot"] = threading.get_ident()
        ids["name"] = threading.current_thread().name
        time.sleep(0.45)
        finished.set()
        return {
            "bootstrapped": True,
            "total_images": 4,
            "train_count": 3,
            "val_count": 1,
            "total_boxes": 5,
            "class_counts": {"herb_aria": 5},
        }

    host = _Host(_bootstrap)
    _bind_one_click(host, factory_mod)
    original = factory_mod.cuda_training_blocked_by_game
    factory_mod.cuda_training_blocked_by_game = lambda: ""
    try:
        ids["caller"] = threading.get_ident()
        started = time.perf_counter()
        result = factory_mod.AutoLabelingDatasetFactory.run_one_click_rtx_training(
            host, async_mode=True, epochs=1
        )
        elapsed = time.perf_counter() - started
        assert elapsed < 0.3, elapsed
        assert result.get("bootstrap_pending") is True
        assert finished.is_set() is False
        assert finished.wait(2.0)
        assert ids["boot"] != ids["caller"]
        assert ids["name"] == "GatherionDatasetBootstrap"
        assert host.trained is True
        assert ids["boot"] != ids["caller"]
    finally:
        factory_mod.cuda_training_blocked_by_game = original
        thread = host._one_click_thread
        if thread is not None:
            thread.join(2.0)


def test_factory_async_does_not_start_cuda_while_game_runs() -> None:
    import dataset_factory.factory_controller as factory_mod

    host = _Host(lambda **kwargs: {"bootstrapped": True, "total_images": 1, "class_counts": {"a": 1}})
    _bind_one_click(host, factory_mod)
    original = factory_mod.cuda_training_blocked_by_game
    factory_mod.cuda_training_blocked_by_game = lambda: factory_mod.GAME_BLOCKS_TRAINING_REASON
    try:
        result = factory_mod.AutoLabelingDatasetFactory.run_one_click_rtx_training(
            host, async_mode=True, epochs=1
        )
        assert result.get("bootstrap_pending") is True
        thread = host._one_click_thread
        assert thread is not None
        thread.join(2.0)
        assert host.trained is False
        payload = host._last_one_click_result
        assert isinstance(payload, dict)
        assert payload.get("waiting_for_game_exit") is True
        assert "oyun kapanana kadar" in str(payload.get("reason"))
    finally:
        factory_mod.cuda_training_blocked_by_game = original


class _LightBuiltin:
    def __init__(self, project_root=None):
        self.auto_label_on_gather = True
        self.passive_shadow_mode = False
        self.synthetic_augment_10x = True
        self.time_machine = None
        self.dataset_dir = os.path.join(ROOT, "dataset")

    def get_dataset_stats(self):
        return _empty_stats()

    def set_passive_shadow_mode(self, enabled):
        self.passive_shadow_mode = bool(enabled)

    def bootstrap_from_existing_frames(self, **kwargs):
        raise AssertionError("builtin bootstrap must not scan images in this test")


class _ClickController:
    def __init__(self, bootstrap):
        self.auto_label_on_gather = True
        self.passive_shadow_mode = False
        self.synthetic_augment_10x = False
        self.time_machine = None
        self._bootstrap = bootstrap
        self.train_threads = []

    def prepare_one_click_dataset(self):
        return self.bootstrap_from_existing_frames(augment_multiplier=3)

    def bootstrap_from_existing_frames(self, **kwargs):
        return self._bootstrap(**kwargs)

    def get_dataset_stats(self):
        return _empty_stats()

    def run_one_click_rtx_training(self, **kwargs):
        self.train_threads.append(threading.get_ident())
        return {
            "started": True,
            "mAP50": 0.2,
            "box_loss": 0.1,
            "cls_loss": 0.05,
            "epochs": 1,
        }


def _dialog(controller):
    from PyQt6.QtWidgets import QApplication
    import ui.yolo_factory_dialog as dialog_mod

    dialog_mod._BuiltinAutoLabelingDatasetFactory = _LightBuiltin
    app = QApplication.instance() or QApplication([])
    dialog = dialog_mod.YOLOFactoryDialog(factory_controller=controller)
    dialog._refresh_timer.stop()
    return app, dialog


def _close_dialog(dialog) -> None:
    worker = getattr(dialog, "_training_worker", None)
    if worker is not None:
        worker.wait(3000)
    dialog._refresh_timer.stop()
    dialog.close()


def test_click_returns_before_bootstrap_on_another_thread() -> None:
    import dataset_factory.factory_controller as factory_mod

    finished = threading.Event()
    ids = {}

    def _bootstrap(**kwargs):
        ids["boot"] = threading.get_ident()
        ids["name"] = threading.current_thread().name
        time.sleep(0.5)
        finished.set()
        return {
            "bootstrapped": True,
            "total_images": 4,
            "train_count": 3,
            "val_count": 1,
            "total_boxes": 2,
            "class_counts": {"herb_aria": 2},
        }

    controller = _ClickController(_bootstrap)
    original = factory_mod.cuda_training_blocked_by_game
    factory_mod.cuda_training_blocked_by_game = lambda: ""
    app, dialog = _dialog(controller)
    try:
        dialog._pipeline["frozen"] = True
        dialog._update_train_lock()
        ids["click"] = threading.get_ident()
        started = time.perf_counter()
        dialog.on_train_rtx_clicked()
        elapsed = time.perf_counter() - started
        assert elapsed < 0.3, elapsed
        assert finished.is_set() is False
        assert "Veri seti taraması sürüyor" in dialog.lbl_train_status.text()
        assert finished.wait(2.0)
        assert ids["boot"] != ids["click"]
        assert ids["name"] == "GatherionDatasetBootstrap"
        assert ids["name"] != "BotWorker"
        assert _pump(app, lambda: bool(controller.train_threads) and "Dataset bootstrap complete" in dialog.training_log_box.toPlainText())
        assert controller.train_threads[0] == ids["boot"]
        assert controller.train_threads[0] != ids["click"]
    finally:
        factory_mod.cuda_training_blocked_by_game = original
        _close_dialog(dialog)


def test_click_waits_when_game_is_running() -> None:
    import dataset_factory.factory_controller as factory_mod

    controller = _ClickController(
        lambda **kwargs: {
            "bootstrapped": True,
            "total_images": 2,
            "train_count": 1,
            "val_count": 1,
            "total_boxes": 1,
            "class_counts": {"herb_aria": 1},
        }
    )
    original = factory_mod.cuda_training_blocked_by_game
    factory_mod.cuda_training_blocked_by_game = lambda: factory_mod.GAME_BLOCKS_TRAINING_REASON
    app, dialog = _dialog(controller)
    try:
        dialog._pipeline["frozen"] = True
        dialog._update_train_lock()
        dialog.on_train_rtx_clicked()
        assert _pump(
            app,
            lambda: "oyun kapanana kadar" in dialog.lbl_train_status.text() and dialog.btn_train_rtx.isEnabled(),
        )
        assert controller.train_threads == []
        assert "Dataset bootstrap complete" in dialog.training_log_box.toPlainText()
    finally:
        factory_mod.cuda_training_blocked_by_game = original
        _close_dialog(dialog)


def test_click_shows_bootstrap_error_without_freezing() -> None:
    import dataset_factory.factory_controller as factory_mod

    def _boom(**kwargs):
        raise RuntimeError("veri seti taraması başarısız")

    controller = _ClickController(_boom)
    original = factory_mod.cuda_training_blocked_by_game
    factory_mod.cuda_training_blocked_by_game = lambda: ""
    app, dialog = _dialog(controller)
    try:
        dialog._pipeline["frozen"] = True
        dialog._update_train_lock()
        started = time.perf_counter()
        dialog.on_train_rtx_clicked()
        assert time.perf_counter() - started < 0.3
        assert _pump(app, lambda: "veri seti taraması başarısız" in dialog.lbl_train_status.text())
        assert dialog.btn_train_rtx.isEnabled()
        assert controller.train_threads == []
    finally:
        factory_mod.cuda_training_blocked_by_game = original
        _close_dialog(dialog)


def main() -> None:
    test_factory_async_returns_before_bootstrap()
    test_factory_async_does_not_start_cuda_while_game_runs()
    test_click_returns_before_bootstrap_on_another_thread()
    test_click_waits_when_game_is_running()
    test_click_shows_bootstrap_error_without_freezing()
    print("train click bootstrap checks passed")


if __name__ == "__main__":
    main()
