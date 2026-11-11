# -*- coding: utf-8 -*-
"""
fix/v20-panic-combat — Madde 4 regresyon testleri (KOŞTURULMAZ; yalnızca yazılır).

Canlı kanıt (Selçuk, PC v19, 2026-11-11):
- [PANIC] COMBAT_TIMEOUT_2M 14:51:28 (123.2s), 14:54:05 (120.8s),
  ~15:14 (127.7s) — sonuncusu bot DURAKLATILMIŞKEN (HUD "DURAKLATILDI").
- HUD "panik şeridi açıldı" logu her saniye tekrarlıyor (1 sn, 2 sn, ...).

Kapsanan düzeltmeler (core/opsec_crypto.py, core/state_machine.py,
core/combat_recovery.py, ui/overlay_hud.py):
(a) Duraklatma/durdurma savaş sayacını sıfırlar/dondurur.
(b) Oyun penceresi yokken savaş sayacı da donar (idle gibi).
(c) Rota FIGHTING kapısı: bar tek başına yetmez (HP düşüşü / ToT aggro
    gerekir) — kapı testleri tests/test_route_fight_confirmation.py'de.
    Panik katmanı: tek poll'lük True sayacı kuramaz (2 ardışık poll).
(d) HUD panik şeridi epizot başına bir kez açılır/loglanır; sayaç
    güncellemesi sessizdir.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.opsec_crypto import PanicAlertSystem
from core.state_machine import BotState, BotWorker


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _age_combat_clock(panic, seconds=200.0):
    """Savaş sayacını geriye al — limiti aşmış gibi yap."""
    panic.set_combat_state(True)
    panic._combat_start_ts = time.monotonic() - float(seconds)


# --- (a) Duraklatma savaş sayacını dondurur ---------------------------------

def test_pause_freezes_combat_timer():
    """200 sn'lik savaş + duraklatma -> COMBAT_TIMEOUT_2M tetiklenmez."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    _age_combat_clock(panic)
    panic.set_paused(True)
    # set_paused sayacı anında sıfırlar.
    assert panic._combat_start_ts is None
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True
    )
    assert event is None
    assert panic.panic_triggered is False
    # Duraklatılıyken poll bayat True okusa bile sayaç kurulamaz.
    panic.set_combat_state(True)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True
    )
    assert event is None
    assert panic._combat_start_ts is None


def test_pause_kwarg_freezes_combat_timer():
    """paused=True argümanı da sayacı dondurur (set_paused'sız yol)."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    _age_combat_clock(panic)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True, paused=True
    )
    assert event is None
    assert panic._combat_start_ts is None


def test_pause_does_not_suppress_death():
    """Duraklatma yalnızca savaş sayacını dondurur — ölüm aynen tetikler."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    panic.set_paused(True)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True, character_dead=True
    )
    assert event is not None
    assert event["reason"] == "CHARACTER_DEAD"


def test_resume_rearms_combat_from_zero():
    """Devam edince savaş sayacı sıfırdan başlar — eski birikimle anında tetiklenmez."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    _age_combat_clock(panic)
    panic.set_paused(True)
    panic.set_paused(False)
    assert panic._combat_start_ts is None
    # Devam + oyun açık + savaş yok -> tetik yok.
    assert (
        panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
        is None
    )
    # Yeni savaş 10 sn'dir (limit 120) -> tetik yok; 200 sn olunca tetikler.
    panic.set_combat_state(True)
    panic._combat_start_ts = time.monotonic() - 10.0
    assert (
        panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
        is None
    )
    panic._combat_start_ts = time.monotonic() - 200.0
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True
    )
    assert event is not None
    assert event["reason"] == "COMBAT_TIMEOUT_2M"


def test_worker_pause_notifies_panic_system():
    """BotWorker.pause() panic_system.set_paused(True) çağırır."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    _age_combat_clock(panic)
    worker = BotWorker.__new__(BotWorker)
    worker.__dict__.update(
        {
            "_lock": threading.RLock(),
            "_paused": False,
            "_state": BotState.WALKING.value,
            "input_driver": None,
            "panic_system": panic,
        }
    )
    worker._emit_log = lambda msg: None
    worker._emit_hud_snapshot = lambda **kwargs: {}
    worker._set_state_internal = lambda s: worker.__dict__.__setitem__("_state", s)
    worker.pause("test duraklatma")
    assert worker._state == BotState.PAUSED.value
    assert panic._paused is True
    assert panic._combat_start_ts is None


def test_notify_panic_paused_without_panic_system_is_silent():
    """panic_system yoksa/eskiyse pause bildirimi sessizce geçilir."""
    worker = BotWorker.__new__(BotWorker)
    worker.__dict__.update({"_lock": threading.RLock(), "_paused": False})
    worker._notify_panic_paused(True)  # raise etmemeli
    worker._notify_panic_paused(False)


# --- (b) Oyun yokken savaş sayacı donar --------------------------------------

def test_game_window_absent_freezes_combat_timer():
    """200 sn'lik savaş + oyun kapalı -> COMBAT_TIMEOUT_2M tetiklenmez."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    _age_combat_clock(panic)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=False
    )
    assert event is None
    assert panic.panic_triggered is False
    assert panic._combat_start_ts is None


def test_game_window_returns_combat_restarts_from_zero():
    """Oyun geri gelince savaş sayacı sıfırdan başlar — anında tetiklenmez."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    _age_combat_clock(panic)
    assert (
        panic.check_watchdog_conditions(play_siren=False, game_window_present=False)
        is None
    )
    assert (
        panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
        is None
    )
    assert panic.panic_triggered is False
    # Limit yeniden dolunca (oyun açıkken) savaş aynen tetikler.
    _age_combat_clock(panic)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True
    )
    assert event is not None
    assert event["reason"] == "COMBAT_TIMEOUT_2M"


# --- (c) Panik katmanı: tek poll'lük True sayacı kuramaz ----------------------

def test_combat_poll_confirmation_needs_two_consecutive_true():
    """İlk True yalnızca hazırlar; ikinci ardışık True kurar."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    assert panic._combat_poll_confirmed(True) is False
    assert panic._combat_start_ts is None
    assert panic._combat_poll_confirmed(True) is True
    # Üçüncü True da doğrulamayı korur.
    assert panic._combat_poll_confirmed(True) is True


def test_combat_poll_confirmation_false_clears_and_none_untouched():
    """False streak'i sıfırlar + sayacı temizler; None dokunmaz."""
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    assert panic._combat_poll_confirmed(True) is False
    assert panic._combat_poll_confirmed(None) is False
    # None streak'i bozmadı: sıradaki True ikinci sayılır.
    assert panic._combat_poll_confirmed(True) is True
    panic.set_combat_state(True)
    assert panic._combat_start_ts is not None
    assert panic._combat_poll_confirmed(False) is False
    assert panic._combat_start_ts is None
    assert panic._combat_poll_streak == 0


def test_poll_combat_single_true_never_arms():
    """Poll: detektör bir tur True sonra False -> sayaç kurulmaz, panik yok."""
    panic = PanicAlertSystem(combat_limit_sec=1.0, enable_sound=False)
    state = {"calls": 0}

    def _detector():
        state["calls"] += 1
        return state["calls"] == 1

    try:
        assert panic.start_watchdog_polling(
            interval_sec=0.3,
            play_siren=False,
            combat_state_detector=_detector,
        )
        time.sleep(2.0)
        assert panic.panic_triggered is False
        assert panic.get_last_panic_event() is None
        assert panic._combat_start_ts is None
    finally:
        panic.stop_watchdog_polling()


def test_poll_combat_two_consecutive_true_arms_and_fires():
    """Poll: detektör sürekli True -> 2. turda sayaç kurulur, limit dolunca tetikler."""
    panic = PanicAlertSystem(combat_limit_sec=1.0, enable_sound=False)
    try:
        assert panic.start_watchdog_polling(
            interval_sec=0.3,
            play_siren=False,
            combat_state_detector=lambda: True,
        )
        assert _wait_until(
            lambda: panic.get_last_panic_event() is not None, timeout=5.0
        )
        assert panic.get_last_panic_event()["reason"] == "COMBAT_TIMEOUT_2M"
    finally:
        panic.stop_watchdog_polling()


# --- (d) HUD: epizot başına bir kez açılır, sayaç sessiz tazelenir -----------

def _make_hud(monkeypatch):
    """Disk panik kaydından arındırılmış OverlayHUD."""
    from PyQt6.QtWidgets import QApplication

    from core.opsec_crypto import PanicAlertSystem

    monkeypatch.setattr(
        PanicAlertSystem,
        "load_panic_history_from_disk",
        lambda *args, **kwargs: [],
    )
    app = QApplication.instance() or QApplication(sys.argv)
    from ui.overlay_hud import OverlayHUD

    hud = OverlayHUD()
    hud.show()
    app.processEvents()
    return hud, app


def _combat_status(elapsed_sec, details="Savaş süresi 2 dakika sınırını aştı (123.2s)."):
    return {
        "panic_triggered": True,
        "inputs_halted": True,
        "reason": "COMBAT_TIMEOUT_2M",
        "details": details,
        "timestamp": 1750000000.0,
        "elapsed_sec": elapsed_sec,
        "history_count": 1,
    }


def _strip_open_count(caplog):
    return len(
        [r for r in caplog.records if "HUD panik şeridi açıldı" in r.message]
    )


def test_panic_strip_silent_elapsed_refresh_same_episode(monkeypatch, caplog):
    """Aynı epizot 30 kez (elapsed 1..30 sn) -> şerit 1 kez açılır/loglanır,
    ama şeritteki sayaç son değeri (30 sn) gösterir."""
    hud, app = _make_hud(monkeypatch)
    try:
        with caplog.at_level(logging.DEBUG, logger="ui.overlay_hud"):
            for elapsed in range(1, 31):
                hud.update_panic_status(_combat_status(float(elapsed)))
                app.processEvents()
        assert _strip_open_count(caplog) == 1
        label = getattr(hud, "panic_label", None) or getattr(hud, "_panic_label", None)
        assert label is not None and label.isVisible()
        assert "COMBAT_TIMEOUT_2M" in label.text()
        assert "30 sn" in label.text()
    finally:
        hud.close()
        app.processEvents()


def test_panic_strip_reopens_for_new_episode_same_reason(monkeypatch, caplog):
    """Aynı sebeple YENİ epizot (farklı details) -> şerit ikinci kez açılır/loglanır."""
    hud, app = _make_hud(monkeypatch)
    try:
        with caplog.at_level(logging.DEBUG, logger="ui.overlay_hud"):
            hud.update_panic_status(_combat_status(1.0))
            app.processEvents()
            assert _strip_open_count(caplog) == 1
            # Kilit açıldı.
            hud.update_panic_status(
                {
                    "panic_triggered": False,
                    "reason": None,
                    "details": None,
                    "timestamp": None,
                    "elapsed_sec": None,
                    "history_count": 1,
                }
            )
            app.processEvents()
            # Yeni epizot, aynı sebep ama farklı tetiklenme süresi.
            hud.update_panic_status(
                _combat_status(2.0, "Savaş süresi 2 dakika sınırını aştı (127.7s).")
            )
            app.processEvents()
        assert _strip_open_count(caplog) == 2
    finally:
        hud.close()
        app.processEvents()
