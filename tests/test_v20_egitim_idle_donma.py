# -*- coding: utf-8 -*-
"""
fix/v20-egitim-idle — regresyon testleri.

Canlı kanıt (Selçuk, PC, 2026-11-12 21:49 oturumu): IDLE_TIMEOUT_15M
22:09:42'de tetiklendi — bot RTX eğitimi için duraklatılmıştı, oyun
görünürdü. Simge-durumu düzeltmesi çalıştı (oyun 21:54:43'te görünür
oldu, panik tam 15,0 dk sonra geldi; öncesi sayaca işlemedi) ama idle
sayacı duraklatmada donmuyordu: savaş ve istem sayacı donarken idle
donmuyordu (core/opsec_crypto.py: idle_paused yalnızca oyun-yokluğu /
simge-durumuna bakıyordu).

Selçuk kararı (2026-11-12 22:32, seçenekten): "Eğitim sırasında
dondur" — DAR kapsam: idle sayacı YALNIZCA eğitim sürerken donar;
her duraklatmada donmaz. Savaş/istem/ölüm/GM yollarına dokunulmaz.
"""

from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.opsec_crypto import PanicAlertSystem


def _aged_idle_panic():
    panic = PanicAlertSystem(
        idle_limit_sec=900.0, combat_limit_sec=120.0, enable_sound=False
    )
    panic._last_gather_ts = time.monotonic() - 1000.0  # 16+ dk toplama yok
    return panic


def test_training_active_freezes_idle_timer():
    panic = _aged_idle_panic()
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=False,
        paused=True,  # eğitim için duraklatılmış bot (canlı durum)
        training_active=True,
    )
    assert event is None
    assert panic.panic_triggered is False
    # Sayaç "şimdi"ye çekildi — eğitim bitince sıfırdan başlar.
    assert time.monotonic() - panic._last_gather_ts < 5.0


def test_training_active_false_still_panics_on_idle():
    panic = _aged_idle_panic()
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=False,
        training_active=False,
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_training_active_none_preserves_old_behavior():
    panic = _aged_idle_panic()
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=False,
        training_active=None,
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_training_freeze_restarts_counter_from_zero():
    panic = _aged_idle_panic()
    # Eğitim sürerken donar, panik atmaz.
    assert (
        panic.check_watchdog_conditions(
            play_siren=False, game_window_present=True, training_active=True
        )
        is None
    )
    # Eğitim bitince sayaç sıfırdan başlar — anında panik atmaz.
    assert (
        panic.check_watchdog_conditions(
            play_siren=False, game_window_present=True, training_active=False
        )
        is None
    )
    # Süre yeniden dolarsa panik atar (donma kalıcı değil).
    panic._last_gather_ts = time.monotonic() - 1000.0
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True, training_active=False
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_training_does_not_freeze_combat_timer():
    # Dar kapsam: eğitim YALNIZCA idle'ı dondurur; savaş sayacı
    # eğitim sinyalinden etkilenmez (duraklatma kuralı ayrı).
    panic = PanicAlertSystem(
        idle_limit_sec=99999.0, combat_limit_sec=120.0, enable_sound=False
    )
    panic.set_combat_state(True)
    panic._combat_start_ts = time.monotonic() - 200.0
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=False,
        paused=False,
        training_active=True,
    )
    assert event is not None
    assert event["reason"] == "COMBAT_TIMEOUT_2M"


def test_training_does_not_suppress_prompt_stuck():
    panic = PanicAlertSystem(
        idle_limit_sec=99999.0,
        combat_limit_sec=99999.0,
        prompt_stuck_sec=10.0,
        enable_sound=False,
    )
    with panic._lock:
        panic._prompt_appear_ts = time.monotonic() - 20.0
        panic._prompt_stuck_grace_until = 0.0
        panic._prompt_stuck_confirm_streak = 0
    # Confirm streak: iki poll (eğitim idle'ı dondurur, istemi değil).
    assert (
        panic.check_watchdog_conditions(
            play_siren=False,
            game_window_present=True,
            paused=False,
            training_active=True,
        )
        is None
    )
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        paused=False,
        training_active=True,
    )
    assert event is not None
    assert event["reason"] == "PROMPT_STUCK_10S"
    assert event.get("soft_pause") is True
    assert panic.panic_triggered is False


def test_training_does_not_suppress_death():
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        training_active=True,
        character_dead=True,
    )
    assert event is not None
    assert event["reason"] == "CHARACTER_DEAD"


def test_training_does_not_suppress_gm_whisper():
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        training_active=True,
        gm_whisper_detected=True,
    )
    assert event is not None
    assert event["reason"] == "GM_INSPECTION_ALERT"


def test_training_detector_wiring_status():
    panic = PanicAlertSystem(enable_sound=False)
    panic.register_watchdog_detectors(training_active_detector=lambda: True)
    status = panic.get_watchdog_wiring_status()
    assert status["detectors"]["training_active"] is True
    # Zorunlu sayılmaz — yokluğu "degraded" yapmaz.
    assert "training_active" not in status["missing_required_detectors"]


def test_training_detector_bound_from_slot(monkeypatch):
    # bind, açık detektör verilmezse eğitim slotunu tembel okur.
    fake_engine = types.ModuleType("dataset_factory.rtx_trainer_engine")
    state = {"active": True}
    fake_engine.is_training_active = lambda: state["active"]
    monkeypatch.setitem(
        sys.modules, "dataset_factory.rtx_trainer_engine", fake_engine
    )
    panic = PanicAlertSystem(enable_sound=False)
    result = panic.bind_production_watchdog_sources(worker=None)
    assert result["wired"]["training_active"] is True
    assert panic._training_active_detector is not None
    assert panic._training_active_detector() is True
    state["active"] = False
    assert panic._training_active_detector() is False


def test_training_detector_returns_none_when_slot_unreadable(monkeypatch):
    fake_engine = types.ModuleType("dataset_factory.rtx_trainer_engine")

    def _boom():
        raise RuntimeError("boom")

    fake_engine.is_training_active = _boom
    monkeypatch.setitem(
        sys.modules, "dataset_factory.rtx_trainer_engine", fake_engine
    )
    panic = PanicAlertSystem(enable_sound=False)
    panic.bind_production_watchdog_sources(worker=None)
    # Hata = bilinmiyor (None) -> eski davranış, poll ölmez.
    assert panic._training_active_detector() is None


def test_training_detector_explicit_overrides_slot(monkeypatch):
    fake_engine = types.ModuleType("dataset_factory.rtx_trainer_engine")
    fake_engine.is_training_active = lambda: False
    monkeypatch.setitem(
        sys.modules, "dataset_factory.rtx_trainer_engine", fake_engine
    )
    panic = PanicAlertSystem(enable_sound=False)
    panic.bind_production_watchdog_sources(
        worker=None, training_active_detector=lambda: True
    )
    assert panic._training_active_detector() is True


def test_is_training_active_reflects_real_slot():
    from dataset_factory import rtx_trainer_engine as eng

    if eng.is_training_active():
        pytest.skip("eğitim slotu başka bir testte tutuluyor")
    assert eng.is_training_active() is False
    assert eng._acquire_train_slot() is True
    try:
        assert eng.is_training_active() is True
    finally:
        eng._release_train_slot()
    assert eng.is_training_active() is False


def test_poll_freezes_idle_while_training_active():
    panic = PanicAlertSystem(
        idle_limit_sec=1.0, combat_limit_sec=99999.0, enable_sound=False
    )
    panic._last_gather_ts = time.monotonic() - 100.0
    try:
        assert panic.start_watchdog_polling(
            input_ctrl=None,
            interval_sec=0.5,
            play_siren=False,
            training_active_detector=lambda: True,
        )
        time.sleep(1.6)
        assert panic.panic_triggered is False
    finally:
        panic.stop_watchdog_polling()


def test_poll_panics_on_idle_when_training_inactive():
    panic = PanicAlertSystem(
        idle_limit_sec=1.0, combat_limit_sec=99999.0, enable_sound=False
    )
    panic._last_gather_ts = time.monotonic() - 100.0
    try:
        assert panic.start_watchdog_polling(
            input_ctrl=None,
            interval_sec=0.5,
            play_siren=False,
            training_active_detector=lambda: False,
        )
        deadline = time.monotonic() + 4.0
        while not panic.panic_triggered and time.monotonic() < deadline:
            time.sleep(0.2)
        assert panic.panic_triggered is True
    finally:
        panic.stop_watchdog_polling()
