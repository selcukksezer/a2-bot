# -*- coding: utf-8 -*-
"""Dry-run: train slot, image keep, game throttle. No GPU train. No live game throttle."""

from __future__ import annotations

import ast
import os
import sys
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.game_gpu_throttle import (  # noqa: E402
    BELOW_NORMAL_PRIORITY_CLASS,
    GameGpuThrottle,
    current_process_priority,
    exe_is_aion_game,
    find_aion_game_processes,
    title_is_aion_game,
    training_image_retention_decision,
)
from dataset_factory.rtx_trainer_engine import (  # noqa: E402
    RTXTrainerEngine,
    _release_train_slot,
    _train_slot_busy,
)
import core.game_gpu_throttle as throttle_mod  # noqa: E402


def _force_own_priority(priority: int) -> None:
    if os.name != "nt" or not priority:
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.windll.kernel32
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), int(priority))


def test_names_and_missing_process() -> None:
    assert title_is_aion_game("AION 2")
    assert title_is_aion_game("Aion")
    assert not title_is_aion_game("Gatherion-Local v2.0")
    assert exe_is_aion_game("Aion2.exe")
    assert exe_is_aion_game(r"C:\Program Files\NCSOFT\AION2\Aion2Client-Win64-Shipping.exe")
    assert not exe_is_aion_game("python.exe")
    assert not exe_is_aion_game("Purple.exe")

    found = find_aion_game_processes()
    assert isinstance(found, list)

    with GameGpuThrottle(finder=lambda: []) as empty:
        assert empty.game_found is False
        assert empty.applied is False
    with GameGpuThrottle(finder=lambda: [999999999]) as missing:
        assert missing.game_found is True
        assert missing.applied is False


def test_own_process_restore() -> None:
    before = current_process_priority()
    try:
        with GameGpuThrottle(finder=lambda: [os.getpid()]) as session:
            mid = current_process_priority()
            assert session.game_found is True
            if before and before != BELOW_NORMAL_PRIORITY_CLASS:
                assert mid in (before, BELOW_NORMAL_PRIORITY_CLASS)
            raise RuntimeError("restore-on-fail")
    except RuntimeError as exc:
        assert str(exc) == "restore-on-fail"
    finally:
        _force_own_priority(before)
    assert current_process_priority() == before


def test_image_keep_and_refuses_while_aion_runs() -> None:
    decision = training_image_retention_decision(ROOT)
    assert decision["action"] == "keep"
    assert decision["delete"] is False
    assert any(path.endswith(os.path.join("dataset", "images")) for path in decision["keep_paths"])

    engine = RTXTrainerEngine()
    seen = {}

    def _body(self, data_yaml_path, epochs=50, progress_callback=None, imgsz=640, batch=16, force_fallback=False,
            model="yolo11n.pt", allow_opencv_fallback=True, use_approved_only=False, **kwargs):
        seen["batch"] = batch
        seen["called"] = True
        return {"started": True, "batch": batch}

    engine._run_training_sync_body = _body.__get__(engine, RTXTrainerEngine)
    original_find = throttle_mod.find_aion_game_processes
    try:
        throttle_mod.find_aion_game_processes = lambda: [424242]
        refused = engine.run_training_sync(
            data_yaml_path=os.path.join(ROOT, "dataset", "data.yaml"),
            epochs=1,
            batch=16,
            force_fallback=True,
        )
        assert refused.get("started") is False
        assert refused.get("reason") == "aion2_running"
        assert refused.get("game_pids") == [424242]
        assert seen.get("called") is not True
        assert _train_slot_busy() is False

        throttle_mod.find_aion_game_processes = lambda: []
        clear = engine.run_training_sync(
            data_yaml_path=os.path.join(ROOT, "dataset", "data.yaml"),
            epochs=1,
            batch=16,
            force_fallback=True,
        )
        assert clear.get("started") is True
        assert seen["batch"] == 16
        assert _train_slot_busy() is False

        throttle_mod.find_aion_game_processes = lambda: []
        busy_hold = engine._begin_training_session(slot_held=False, batch=16)
        assert busy_hold["refused"] is False
        second = engine.run_training_sync(
            data_yaml_path=os.path.join(ROOT, "dataset", "data.yaml"),
            epochs=1,
            batch=16,
            force_fallback=True,
        )
        assert second.get("started") is False
        engine._end_training_session(busy_hold)
        assert _train_slot_busy() is False
    finally:
        throttle_mod.find_aion_game_processes = original_find
        _release_train_slot()


def test_disk_dolu_does_not_train() -> None:
    import dataset_factory.rtx_trainer_engine as engine_mod

    engine = RTXTrainerEngine()
    called = {"body": False}

    def _body(self, data_yaml_path, epochs=50, progress_callback=None, imgsz=640, batch=16, force_fallback=False,
            model="yolo11n.pt", allow_opencv_fallback=True, use_approved_only=False, **kwargs):
        called["body"] = True
        return {"started": True}

    engine._run_training_sync_body = _body.__get__(engine, RTXTrainerEngine)
    original = engine_mod.target_volume_free_bytes
    engine_mod.target_volume_free_bytes = lambda path: 1024
    try:
        result = engine.run_training_sync(
            data_yaml_path=os.path.join(ROOT, "dataset", "data.yaml"),
            epochs=1,
            batch=16,
            force_fallback=True,
        )
        assert result.get("started") is False
        assert result.get("error") == "disk dolu"
        assert called["body"] is False
        assert _train_slot_busy() is False
        # Platform/behavior-driven runs-dir check (F-1334): never hard-code
        # the Windows "F:\a2" path. The contract is that the runs dir is
        # <offload_root>/models/rtx_runs when an F: volume exists, and the
        # repo-local models/rtx_runs when it does not (S085: F: yokken
        # sabit F:/ yoluna yazım denemesi yok).
        runs_dir = engine_mod.rtx_runs_dir_for()
        assert runs_dir.parts[-2:] == ("models", "rtx_runs")
        root = engine_mod.offload_root()
        if root is None:
            assert runs_dir == engine_mod._project_root() / "models" / "rtx_runs"
        else:
            assert runs_dir == root / "models" / "rtx_runs"
        # fix/v20-egitim-devam: planlanan koşu klasörü artık benzersiz
        # zaman damgalı isim alır (sabit isim üstüne yazma yok); sözleşme
        # klasörün runs_dir altında + damgalı isimde olmasıdır.
        save_dir = Path(str(result.get("save_dir")))
        assert save_dir.parent == runs_dir
        assert save_dir.name.startswith("gatherion_rtx_fp16_")
    finally:
        engine_mod.target_volume_free_bytes = original
        _release_train_slot()


def test_sounds_attribute() -> None:
    import types

    source = open(os.path.join(ROOT, "main.py"), encoding="utf-8").read()
    tree = ast.parse(source)
    click_ok = False
    assigned = "self.sounds = UiClickSounds()" in source
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "UiClickSounds":
            click_ok = any(
                isinstance(item, ast.FunctionDef) and item.name == "play_click"
                for item in node.body
            )
    assert click_ok and assigned
    # winsound is Windows-only. main.py keeps the Linux-safe guarded
    # import (R22), and other test modules may already have installed
    # their own stub (e.g. d81's no-op stub at collection time), so
    # setdefault would be a no-op: assign directly and restore after.
    calls = []
    winsound_stub = types.ModuleType("winsound")
    winsound_stub.Beep = lambda freq, duration: calls.append((freq, duration))
    _previous_winsound = sys.modules.get("winsound")
    sys.modules["winsound"] = winsound_stub
    from main import GatherionOrchestrator, UiClickSounds

    sounds = UiClickSounds()
    assert hasattr(sounds, "play_click")
    assert hasattr(GatherionOrchestrator, "trigger_ai_training_and_open_dialog")
    # F-1553: play_click() must be asserted, not just called — it returns
    # None and must actually attempt the beep on the success path.
    assert sounds.play_click() is None
    assert calls == [(880, 40)]
    # ... and must swallow a failing sound backend instead of raising.
    calls.clear()
    real_beep = winsound_stub.Beep
    winsound_stub.Beep = lambda freq, duration: (_ for _ in ()).throw(RuntimeError("no audio"))
    try:
        assert sounds.play_click() is None
    finally:
        winsound_stub.Beep = real_beep
    assert calls == []
    # Restore whatever stub (if any) occupied sys.modules before this test,
    # so later tests see the suite state as if this test never ran.
    if _previous_winsound is None:
        sys.modules.pop("winsound", None)
    else:
        sys.modules["winsound"] = _previous_winsound


def load_tests(loader, tests, pattern):
    """Expose the plain test functions to unittest discovery (S-4101).

    pytest already collects the module-level test_* functions; unittest
    discovery only finds TestCase classes, so without this it reports
    0 tests for this file. load_tests is only consulted by unittest,
    so pytest collection is unchanged (no double run).
    Same fix as F-1333 in test_train_click_bootstrap.py.
    """
    suite = unittest.TestSuite()
    for _name in (
        "test_names_and_missing_process",
        "test_own_process_restore",
        "test_image_keep_and_refuses_while_aion_runs",
        "test_disk_dolu_does_not_train",
        "test_sounds_attribute",
    ):
        suite.addTest(unittest.FunctionTestCase(globals()[_name]))
    return suite


def main() -> None:
    test_names_and_missing_process()
    test_own_process_restore()
    test_image_keep_and_refuses_while_aion_runs()
    test_disk_dolu_does_not_train()
    test_sounds_attribute()
    print("train session checks passed")


if __name__ == "__main__":
    main()
