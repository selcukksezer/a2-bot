# -*- coding: utf-8 -*-
"""Start-training must return before dataset bootstrap, and bootstrap runs on another thread.

No real image scan. No ultralytics. No CUDA train.

The click test builds its own minimal labeled dataset fixture
(`_write_labeled_dataset`, temp dir) because the worker's post-bootstrap
yaml gate (P0-5/K-30) validates the controller's dataset yaml before
training starts — the repo's tracked `dataset/data.yaml` is relative-path
with empty image splits in a fresh worktree, so the gate would fail and
training would never start.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import threading
import time
import unittest
import zlib
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@contextmanager
def _patched_game_finder(pids):
    """Patch the low-level game-process finder (F-1438): the production
    gate itself must detect the game, never a stubbed gate function.
    Restores the original finder on exit, even on failure."""
    import core.game_gpu_throttle as throttle_mod

    original_find = throttle_mod.find_aion_game_processes
    throttle_mod.find_aion_game_processes = lambda: list(pids)
    try:
        yield
    finally:
        throttle_mod.find_aion_game_processes = original_find


def _one_px_png() -> bytes:
    """Minimal valid 1x1 RGB PNG (stdlib only) — dialog preview decode
    succeeds, so no 'Decode başarısız' empty state is exercised."""
    def _chunk(typ: bytes, data: bytes) -> bytes:
        out = struct.pack(">I", len(data)) + typ + data
        return out + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00\x00\x00\x00")
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


def _write_labeled_dataset(root: Path) -> Path:
    """Minimal REAL labeled dataset: one labeled frame per split + a
    data.yaml with an ABSOLUTE `path:` — the same shape as
    tests/test_yaml_absolute_path.py::_labeled_frame + _write_yaml, but
    with a decodable PNG (dialog preview path) instead of fake bytes.

    This is what the worker's post-bootstrap yaml gate (P0-5/K-30,
    `validate_training_yaml`) needs: absolute `path:`, existing
    train/val dirs, >= 1 labeled image across the splits. The fake
    bootstrap in this file never writes a yaml, so without this fixture
    the gate validates the repo's tracked `dataset/data.yaml`
    (`path: .` — relative — and empty/missing image splits in a fresh
    worktree) and training never starts.
    """
    for split, stem in (("train", "a"), ("val", "b")):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / f"{stem}.png").write_bytes(_one_px_png())
        (lbl_dir / f"{stem}.txt").write_text(
            "6 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8"
        )
    (root / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "nc: 27",
                "names:",
                "  6: herb_edana",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return root


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
    # Exercise the real gate: patch the low-level game-process finder
    # (no AION process) instead of stubbing cuda_training_blocked_by_game
    # itself (F-1438). The gate must then report "not blocked".
    with _patched_game_finder([]):
        ids["caller"] = threading.get_ident()
        started = time.perf_counter()
        result = factory_mod.AutoLabelingDatasetFactory.run_one_click_rtx_training(
            host, async_mode=True, epochs=1
        )
        elapsed = time.perf_counter() - started
        # Generous bound only: the real async proof is the finished-event
        # and thread-identity assertions below. A tight bound (< 0.3 s)
        # flakes under load (F-1437); a synchronous call would leave the
        # bootstrap finished and on the caller thread instead.
        assert elapsed < 2.0, elapsed
        assert result.get("bootstrap_pending") is True
        assert finished.is_set() is False
        assert finished.wait(2.0)
        assert ids["boot"] != ids["caller"]
        assert ids["name"] == "GatherionDatasetBootstrap"
        assert host.trained is True
        assert ids["boot"] != ids["caller"]
    thread = host._one_click_thread
    if thread is not None:
        thread.join(2.0)


def test_factory_async_does_not_start_cuda_while_game_runs() -> None:
    import dataset_factory.factory_controller as factory_mod

    host = _Host(lambda **kwargs: {"bootstrapped": True, "total_images": 1, "class_counts": {"a": 1}})
    _bind_one_click(host, factory_mod)
    # Real gate, fake game process: patch the low-level finder so the
    # production cuda_training_blocked_by_game() itself detects the game
    # (F-1438) instead of stubbing the gate function.
    with _patched_game_finder([424242]):
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
    def __init__(self, bootstrap, dataset_dir=None):
        self.auto_label_on_gather = True
        self.passive_shadow_mode = False
        self.synthetic_augment_10x = False
        self.time_machine = None
        self._bootstrap = bootstrap
        self.train_threads = []
        # Only set when the caller supplies a fixture dir: the worker's
        # yaml gate reads `getattr(controller, "dataset_dir",
        # Path("dataset"))`, and the dialog's `_dataset_dir()` prefers
        # this attribute too. Absent -> repo `dataset/` fallback, as
        # before (the other two click tests never reach the yaml gate).
        if dataset_dir is not None:
            self.dataset_dir = dataset_dir

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


def _yellow_mode(dialog) -> None:
    """Tık testleri için sarı (onaylı-olmayan) yolu seç.

    Dialog varsayılan olarak onaylı modda açılır
    (`chk_only_approved.setChecked(True)`, ui/yolo_factory_dialog.py:4926),
    o yüzden argümansız `on_train_rtx_clicked()` — tıpkı sarı butonun
    gerçek bağlantısı gibi (`btn_train_rtx.clicked`, :5577) — modu
    toggle'dan okuyup onaylı yola girer. Onaylı yol B-10 gereği
    `dataset/approved/data_approved.yaml` ister; o dosya gitignore'lı
    (.gitignore:16) bir üretim artığıdır — taze worktree'de yoktur, o
    yüzden tık bootstrap'a hiç ulaşmadan `[EĞİTİM] RED: onaylı eğitim
    verisi hazır değil` ile döner. Bu testlerin konusu bootstrap
    iş parçacığı davranışıdır (tık erken döner / oyun bekler / hata
    donmadan gösterilir), onaylı kapısı değil — onaylı kapı
    tests/test_train_gate_delta.py'de zaten kapsanıyor. O yüzden
    fixture modu açıkça sarıya çeker; tıklama argümansız kalır ki
    üretimdeki buton yolu aynen çalışsın. P0-4'ün pending kapısı bu
    testlerde hiç devreye girmez (kuyruk 0 kare → engel 0).
    """
    for attr in (
        "chk_only_approved",
        "chk_train_only_approved",
        "toggle_approved_only_prepare",
        "toggle_approved_only_train",
        "toggle_approved_only",
    ):
        widget = getattr(dialog, attr, None)
        if widget is not None:
            widget.setChecked(False)
    assert dialog._only_approved_enabled() is False


def test_click_returns_before_bootstrap_on_another_thread() -> None:
    """v19 (bootstrap): click test now owns its data.

    Root cause of the fresh-worktree failure: the click drives the real
    `RTXTrainingWorker.run()`, whose P0-5/K-30 yaml gate runs AFTER the
    (fake) bootstrap and BEFORE training starts. The fake bootstrap
    writes no yaml, so the gate validated the repo's tracked
    `dataset/data.yaml` — `path: .` (relative, rejected) with
    empty/missing `images/{train,val}` (0 labeled images, rejected).
    The gate emitted `training_failed`, `run_one_click_rtx_training`
    never ran, `controller.train_threads` stayed empty and the final
    `_pump` timed out. On a PC checkout with real labeled images and
    a bootstrap-rewritten absolute-path yaml the gate passed, so the
    test only broke where there was no data. Fix: build a minimal
    real labeled dataset in a temp dir and point the controller at it
    (`dataset_dir` — the same attribute the worker gate and the
    dialog's `_dataset_dir()` read). No production behavior changes;
    the fixture frames are TXT-only hand-labeled frames, so the P0-4
    blockers gate stays 0 (protected frames never block training).
    """
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

    with tempfile.TemporaryDirectory(prefix="gatherion-bootstrap-test-") as tmp:
        ds = _write_labeled_dataset(Path(tmp) / "dataset")
        # Fast, self-diagnosing pre-check: if the fixture ever stops
        # satisfying the real gate, fail here naming the gate errors
        # instead of a cryptic `_pump` timeout below.
        from dataset_factory.yolo_trainer import validate_training_yaml

        _report = validate_training_yaml(ds / "data.yaml")
        assert _report["ok"] is True, _report["errors"]

        controller = _ClickController(_bootstrap, dataset_dir=ds)
        # Real gate, no game: patch the low-level finder, not the gate (F-1438).
        app, dialog = _dialog(controller)
        try:
            with _patched_game_finder([]):
                _yellow_mode(dialog)  # sarı yol: tık bootstrap'a ulaşsın
                dialog._pipeline["frozen"] = True
                dialog._update_train_lock()
                ids["click"] = threading.get_ident()
                started = time.perf_counter()
                dialog.on_train_rtx_clicked()
                elapsed = time.perf_counter() - started
                # Generous bound only (F-1437); async proof is finished-event +
                # thread identity below, not wall-clock timing.
                assert elapsed < 2.0, elapsed
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
            _close_dialog(dialog)


def test_click_waits_when_game_is_running() -> None:
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
    # Real gate, fake game process (F-1438): the production gate itself
    # must detect the game via the finder.
    app, dialog = _dialog(controller)
    try:
        with _patched_game_finder([424242]):
            _yellow_mode(dialog)  # sarı yol: tık bootstrap'a ulaşsın
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
        _close_dialog(dialog)


def test_click_shows_bootstrap_error_without_freezing() -> None:
    def _boom(**kwargs):
        raise RuntimeError("veri seti taraması başarısız")

    controller = _ClickController(_boom)
    # Real gate, no game (F-1438).
    app, dialog = _dialog(controller)
    try:
        with _patched_game_finder([]):
            _yellow_mode(dialog)  # sarı yol: tık bootstrap'a ulaşsın
            dialog._pipeline["frozen"] = True
            dialog._update_train_lock()
            started = time.perf_counter()
            dialog.on_train_rtx_clicked()
            # Generous bound only (F-1437); the _pump assertion below is the
            # real check that the error surfaces without freezing the UI.
            assert time.perf_counter() - started < 2.0
            assert _pump(app, lambda: "veri seti taraması başarısız" in dialog.lbl_train_status.text())
            assert dialog.btn_train_rtx.isEnabled()
            assert controller.train_threads == []
    finally:
        _close_dialog(dialog)


def load_tests(loader, tests, pattern):
    """Expose the plain test functions to unittest discovery (F-1333).

    pytest already collects the module-level test_* functions; unittest
    discovery only finds TestCase classes, so without this it reports
    0 tests for this file. load_tests is only consulted by unittest,
    so pytest collection is unchanged (no double run).
    """
    suite = unittest.TestSuite()
    for _name in (
        "test_factory_async_returns_before_bootstrap",
        "test_factory_async_does_not_start_cuda_while_game_runs",
        "test_click_returns_before_bootstrap_on_another_thread",
        "test_click_waits_when_game_is_running",
        "test_click_shows_bootstrap_error_without_freezing",
    ):
        suite.addTest(unittest.FunctionTestCase(globals()[_name]))
    return suite


def main() -> None:
    test_factory_async_returns_before_bootstrap()
    test_factory_async_does_not_start_cuda_while_game_runs()
    test_click_returns_before_bootstrap_on_another_thread()
    test_click_waits_when_game_is_running()
    test_click_shows_bootstrap_error_without_freezing()
    print("train click bootstrap checks passed")


if __name__ == "__main__":
    main()
