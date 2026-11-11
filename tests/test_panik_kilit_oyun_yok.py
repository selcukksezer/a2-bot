# -*- coding: utf-8 -*-
"""
fix/panik-kilit-oyun-yok — Faz 3 testleri.

Sorun (canlı log 2026-11-10 21:05:44-21:10:18):
1) Oyun kapalıyken ("Game window not found" varken) 15,1 dk toplama yok
   diye IDLE_TIMEOUT_15M panik kilidi tetiklendi. Oyun kapalı kalırsa
   ~21:25'te tekrar tetiklenirdi.
2) HUD panik şeridi kilit süresince her saniye yeniden açılıp loglandı
   (~5 dk, ~270 satır).

Düzeltme:
1) Oyun penceresi yokken (game_window_present=False) idle sayacı
   sıfırlanır/durur — yalnızca idle yolu; ölüm/GM/istem aynen
   çalışır. SAVAŞ yolu fix/v20-panic-combat ile ayrıca dondurulur
   (bu dosyadaki eski "savaş oyun yokken de tetikler" vakası
   kaldırıldı; bkz. tests/test_v20_panic_combat.py).
   None (bilinmiyor/verilmedi) = eski davranış.
2) update_panic_status imzası artık elapsed_sec içermez — geçen süre
   her saniye büyüdüğü için imzadayken kilit-başı dedup hiç tutmuyor,
   şerit her yenilemede yeniden açılıp loglanıyordu.

Bu dosya yalnızca YENİ testler içerir; mevcut test dosyaları
değiştirilmedi.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.opsec_crypto import PanicAlertSystem
from core.watchdog import BotWatchdog


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _age_idle_clock(panic, seconds=3600.0):
    """Idle sayacını geriye al — idle limiti aşılmış gibi yap."""
    panic._last_gather_ts -= float(seconds)


# --- 1) Oyun yokken idle tetiklenmez ----------------------------------------

def test_game_window_absent_idle_never_triggers():
    """Oyun kapalıyken 60 dk toplama yoksa bile IDLE tetiklenmez."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    _age_idle_clock(panic)
    for _ in range(3):
        event = panic.check_watchdog_conditions(
            play_siren=False, game_window_present=False
        )
        assert event is None
    assert panic.panic_triggered is False
    assert panic.get_last_panic_event() is None
    # Sayaç sıfırlandı/durdu: son kontrolün ardından geçen süre küçüktür.
    assert time.monotonic() - panic._last_gather_ts < 5.0


def test_game_window_absent_then_present_triggers_after_limit():
    """Oyun yokken duran sayaç, oyun geri gelince sıfırdan başlar."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    _age_idle_clock(panic)
    assert (
        panic.check_watchdog_conditions(play_siren=False, game_window_present=False)
        is None
    )
    assert panic.panic_triggered is False
    # Oyun geri geldi ama sayaç az önce sıfırlandı — hemen tetiklenmez.
    assert (
        panic.check_watchdog_conditions(play_siren=False, game_window_present=True)
        is None
    )
    assert panic.panic_triggered is False
    # Limit yeniden dolunca (oyun açıkken) idle aynen tetikler.
    _age_idle_clock(panic)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_game_window_present_idle_triggers():
    """Oyun açıkken idle tetikleyici aynen çalışır."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    _age_idle_clock(panic)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_game_window_unknown_idle_triggers():
    """game_window_present verilmezse (None) eski davranış korunur."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    _age_idle_clock(panic)
    event = panic.check_watchdog_conditions(play_siren=False)
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


# --- 2) Diğer 4 tetikleyici oyun yokken de çalışır ---------------------------

@pytest.mark.parametrize(
    "setup,kwargs,expected_reason",
    [
        (
            lambda p: None,
            {"character_dead": True},
            "CHARACTER_DEAD",
        ),
        (
            lambda p: None,
            {"gm_whisper_detected": True},
            "GM_INSPECTION_ALERT",
        ),
        (
            lambda p: None,
            {"prompt_stuck_msg": "Prompt stuck for 10s (12.3s boyunca kapanmadı)."},
            "PROMPT_STUCK_10S",
        ),
        # NOT fix/v20-panic-combat: savaş bu listeden ÇIKTI — oyun
        # yokken savaş sayacı donar, COMBAT_TIMEOUT_2M tetiklenmez
        # (bkz. tests/test_v20_panic_combat.py).
    ],
)
def test_game_window_absent_other_triggers_still_fire(setup, kwargs, expected_reason):
    """Ölüm/GM/istem, oyun penceresi yokken bile tetikler."""
    panic = PanicAlertSystem(
        idle_limit_sec=60.0, combat_limit_sec=120.0, enable_sound=False
    )
    _age_idle_clock(panic)  # idle de tutardı — ama oyun yokken bastırılır.
    setup(panic)
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=False, **kwargs
    )
    assert event is not None, "tetikleyici oyun yokken ateşlenmedi"
    assert event["reason"] == expected_reason


# --- 3) Poll döngüsü ---------------------------------------------------------

def test_poll_game_window_absent_no_idle_panic_then_present_fires():
    """Poll: detektör False iken kilit yok; True'ya dönünce idle tetikler."""
    panic = PanicAlertSystem(idle_limit_sec=0.6, enable_sound=False)
    state = {"present": False}
    try:
        started = panic.start_watchdog_polling(
            interval_sec=0.3,
            play_siren=False,
            game_window_present_detector=lambda: state["present"],
        )
        assert started is True
        time.sleep(1.5)  # idle limiti kat kat aşar ama oyun "kapalı"
        assert panic.panic_triggered is False
        assert panic.get_last_panic_event() is None
        state["present"] = True
        assert _wait_until(
            lambda: panic.get_last_panic_event() is not None, timeout=5.0
        )
        assert panic.get_last_panic_event()["reason"] == "IDLE_TIMEOUT_15M"
    finally:
        panic.stop_watchdog_polling()


def test_poll_game_window_present_idle_fires():
    """Poll: detektör True iken idle eskisi gibi tetikler."""
    panic = PanicAlertSystem(idle_limit_sec=0.6, enable_sound=False)
    try:
        panic.start_watchdog_polling(
            interval_sec=0.3,
            play_siren=False,
            game_window_present_detector=lambda: True,
        )
        assert _wait_until(
            lambda: panic.get_last_panic_event() is not None, timeout=5.0
        )
        assert panic.get_last_panic_event()["reason"] == "IDLE_TIMEOUT_15M"
    finally:
        panic.stop_watchdog_polling()


def test_poll_game_window_detector_unknown_keeps_old_behavior():
    """Poll: detektör None (belirsiz) ise idle eskisi gibi tetikler."""
    panic = PanicAlertSystem(idle_limit_sec=0.6, enable_sound=False)
    try:
        panic.start_watchdog_polling(
            interval_sec=0.3,
            play_siren=False,
            game_window_present_detector=lambda: None,
        )
        assert _wait_until(
            lambda: panic.get_last_panic_event() is not None, timeout=5.0
        )
        assert panic.get_last_panic_event()["reason"] == "IDLE_TIMEOUT_15M"
    finally:
        panic.stop_watchdog_polling()


# --- 4) Üretim bağlantısı (bind) ----------------------------------------------

def test_bind_wires_game_window_detector_from_worker_watchdog():
    """bind, worker.watchdog.is_game_window_present'i detektör yapar."""
    panic = PanicAlertSystem(enable_sound=False)
    worker = SimpleNamespace(watchdog=BotWatchdog())
    result = panic.bind_production_watchdog_sources(worker=worker)
    assert result["wired"]["game_window_present"] is True
    assert panic._game_window_present_detector is not None
    # Linux/VM'de pencere belirlenemez (None) — ama "zorunlu" sayılmaz.
    assert panic._game_window_present_detector() is None
    status = panic.get_watchdog_wiring_status()
    assert status["detectors"]["game_window_present"] is True
    assert "game_window_present" not in status["missing_required_detectors"]


def test_bind_game_window_detector_false_path():
    """bind edilen detektör False döndürürse poll idle'ı bastırır."""
    panic = PanicAlertSystem(idle_limit_sec=0.6, enable_sound=False)
    worker = SimpleNamespace(
        watchdog=SimpleNamespace(is_game_window_present=lambda: False)
    )
    result = panic.bind_production_watchdog_sources(worker=worker)
    assert result["wired"]["game_window_present"] is True
    assert panic._game_window_present_detector() is False
    try:
        panic.start_watchdog_polling(interval_sec=0.3, play_siren=False)
        time.sleep(1.5)
        assert panic.panic_triggered is False
        assert panic.get_last_panic_event() is None
    finally:
        panic.stop_watchdog_polling()


def test_bind_without_worker_watchdog_leaves_detector_unwired():
    """watchdog'suz worker'da detektör kayıtsız kalır, eski davranış sürer."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    result = panic.bind_production_watchdog_sources(worker=SimpleNamespace())
    assert result["wired"]["game_window_present"] is False
    assert "game_window_present" in result["missing"]
    assert "game_window_present" not in panic.get_watchdog_wiring_status()[
        "missing_required_detectors"
    ]
    _age_idle_clock(panic)
    event = panic.check_watchdog_conditions(play_siren=False)
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_bot_watchdog_is_game_window_present_unknown_off_windows():
    """BotWatchdog: Windows dışı / başlıksız durumda None (belirsiz)."""
    watchdog = BotWatchdog()
    # Bu VM Linux'tur: Win32 penceresi aranamaz → None dönmeli.
    assert watchdog.is_game_window_present() is None


# --- 5) HUD şeridi kilit başına bir kez açılır/loglanır ------------------------

def _make_hud(monkeypatch):
    """Disk panik kaydından arındırılmış OverlayHUD (adversarial harness kalıbı)."""
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


def _panic_status(reason, elapsed_sec):
    return {
        "panic_triggered": True,
        "inputs_halted": True,
        "reason": reason,
        "details": "Karakter 60.0 dakikadır hiçbir kaynak toplayamadı.",
        "timestamp": time.time(),
        "elapsed_sec": elapsed_sec,
        "history_count": 1,
    }


def _strip_open_count(caplog):
    return len(
        [r for r in caplog.records if "HUD panik şeridi açıldı" in r.message]
    )


def test_panic_strip_opens_once_per_lock(monkeypatch, caplog):
    """Büyüyen elapsed_sec ile 6 yenileme → şerit 1 kez açılır, 1 kez loglanır."""
    hud, app = _make_hud(monkeypatch)
    try:
        with caplog.at_level(logging.DEBUG, logger="ui.overlay_hud"):
            for elapsed in (1.0, 6.0, 11.0, 61.0, 121.0, 301.0):
                hud.update_panic_status(
                    _panic_status("IDLE_TIMEOUT_15M", elapsed)
                )
                app.processEvents()
        assert _strip_open_count(caplog) == 1
        label = getattr(hud, "panic_label", None) or getattr(hud, "_panic_label", None)
        assert label is not None and label.isVisible()
        assert "IDLE_TIMEOUT_15M" in label.text()
    finally:
        hud.close()
        app.processEvents()


def test_panic_strip_reopens_for_new_lock(monkeypatch, caplog):
    """Kilit açılıp yeni kilit gelince şerit yeniden açılır (toplam 2 kez)."""
    hud, app = _make_hud(monkeypatch)
    try:
        with caplog.at_level(logging.DEBUG, logger="ui.overlay_hud"):
            for elapsed in (1.0, 6.0, 301.0):
                hud.update_panic_status(
                    _panic_status("IDLE_TIMEOUT_15M", elapsed)
                )
                app.processEvents()
            assert _strip_open_count(caplog) == 1
            # Kilit açıldı → şerit gizlenir.
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
            label = getattr(hud, "panic_label", None) or getattr(
                hud, "_panic_label", None
            )
            assert label is not None and not label.isVisible()
            # Yeni kilit (farklı sebep) → şerit ikinci kez açılır.
            hud.update_panic_status(_panic_status("CHARACTER_DEAD", 2.0))
            app.processEvents()
        assert _strip_open_count(caplog) == 2
        assert label.isVisible()
        assert "CHARACTER_DEAD" in label.text()
    finally:
        hud.close()
        app.processEvents()
