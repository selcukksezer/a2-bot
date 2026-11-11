# -*- coding: utf-8 -*-
"""
fix/v19-panic-e2e — panik kilidi uçtan uca regresyon testleri.

Canlı olay (Selçuk'un PC logu, 2026-11-11 13:52:01):
Oyun kapalıyken (Watchdog 13:46'dan beri her 2 sn'de "Game window not
found" logluyor) IDLE_TIMEOUT_15M panik kilidi tetiklendi ("Karakter
15.1 dakikadır hiçbir kaynak toplayamadı") ve HUD panik şeridi her
saniye yeniden açıldı/loglandı (422 sn). KÖK NEDEN: PC'de hâlâ v14
çalışıyordu (10 Kasım 19:24 kurulumu); panik düzeltmeleri v15 faz-3'te
geldi, PC'de yok. Yani bu v18/v19 kodunun hatası DEĞİL — ama zincirin
tamamı burada, main.py bağlantısıyla birlikte, uçtan uca doğrulanır.

Doğrulanan zincir (dosya:satır, taban feat/all-combined-v19 @ 75d957c):
1) main.py:716  BotWatchdog(recovery_callback=...) — hedef başlık
   verilmezse varsayılan ["AION 2", "Aion 2", "AION", "Aion"]
   (core/watchdog.py:293-297).
2) main.py:808  self.worker.watchdog = self.watchdog
   main.py:831  self.worker.panic_system = self.panic_system
3) main.py:840  _panic.bind_production_watchdog_sources(worker=self.worker)
   → game_window_present detektörünü worker.watchdog.is_game_window_present
   olarak kaydeder (core/opsec_crypto.py:2394-2417).
4) main.py:844  _panic.start_watchdog_polling(input_ctrl=self.input_ctrl)
   → poll thread'i her turda detektörü çağırıp
   check_watchdog_conditions(game_window_present=...) geçirir
   (core/opsec_crypto.py:2617-2645); False ise idle sayacı
   _last_gather_ts "şimdi"ye çekilerek durur
   (core/opsec_crypto.py:2091-2094).
5) HUD: main.py:1317-1322 f9_menu.bind_targets(overlay, worker, ...)
   → ui/f9_menu.py:1510-1521 overlay.set_panic_lock(worker.panic_system);
   HUD 5 sn'de bir kilidi yoklayıp get_panic_status() → update_panic_status
   çağırır (ui/overlay_hud.py:3125-3130). get_panic_status()'un
   elapsed_sec'i her okumada büyür (core/opsec_crypto.py:1363-1393)
   ama update_panic_status dedup imzası elapsed_sec İÇERMEZ
   (ui/overlay_hud.py:2656-2664) — aynı kilit bir kez açılır/loglanır.

Simülasyon sınırları (dürüstçe):
- Worker, gerçek BotWorker (QThread; config/vision/input_driver ister)
  yerine SimpleNamespace'tir; zincirin okuduğu öznitelikleri aynen
  taşır (watchdog, panic_system, gathered_count). main.py'nin bu
  zincir için yaptığı atamaların tamamı simüle edilir.
- OS pencere probu stub'lanır: bu VM Linux'tur ve
  BotWatchdog.is_game_window_present() Windows dışında None döner
  (core/watchdog.py:736-751). Windows'ta probun bulma/bulmama
  davranışını simüle etmek için metot, bind ÖNCESİ state-driven
  False/True dönen bir fonksiyonla monkeypatch'lenir. Bind, bağlı
  metodu bind anında yakaladığı için yama bind'den önce yapılır.
- start_watchdog_polling'e input_ctrl=None verilir (main.py
  self.input_ctrl geçirir); input_ctrl yalnızca tetik anında
  emergency_halt_inputs'ta kullanılır, bu zincirin konusu değildir.
- idle limiti GERÇEK varsayılandır (900 sn = 15 dk); saat
  _last_gather_ts geriye alınarak 15,1 dk idle simüle edilir —
  test 15 dk beklemez.
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

# Canlı olaydaki idle süresi: 15,1 dk.
LIVE_IDLE_SEC = 901.0
# PanicAlertSystem varsayılan idle limiti (core/opsec_crypto.py:1184).
DEFAULT_IDLE_LIMIT_SEC = 900.0


def _wait_until(predicate, timeout=5.0, interval=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _age_idle_clock(panic, seconds=LIVE_IDLE_SEC):
    """Idle sayacını geriye al — 15,1 dk toplama yokmuş gibi yap."""
    panic._last_gather_ts -= float(seconds)


def _main_wiring(monkeypatch, panic, present_state):
    """main.py:716/808/831/840/844 bağlantısını aynen kur.

    present_state: {"present": False|True} — Windows pencere probunun
    simüle edilen cevabı. Döner: (watchdog, worker, wiring).
    """
    # main.py:716 — BotWatchdog(recovery_callback=...) (geri çağırma bu
    # zincirde kullanılmaz; varsayılan hedef başlıklar test edilir).
    watchdog = BotWatchdog()
    assert watchdog.target_window_titles == ["AION 2", "Aion 2", "AION", "Aion"]

    # Windows pencere probunu simüle et — BIND'DEN ÖNCE yama yapılır.
    monkeypatch.setattr(
        BotWatchdog,
        "is_game_window_present",
        lambda self: present_state["present"],
    )

    # main.py:808 + main.py:831 — worker'a watchdog ve panic_system bağlanır.
    worker = SimpleNamespace(watchdog=watchdog, gathered_count=0)
    worker.panic_system = panic

    # main.py:840 — detektörler poll başlamadan ÖNCE bağlanır.
    wiring = panic.bind_production_watchdog_sources(worker=worker)
    assert wiring["wired"]["game_window_present"] is True
    assert panic._game_window_present_detector is not None

    # main.py:844 — poll başlar (input_ctrl bu zincirde None).
    started = panic.start_watchdog_polling(
        interval_sec=0.2, play_siren=False
    )
    assert started is True
    assert panic.is_watchdog_polling() is True
    return watchdog, worker, wiring


# --- 1) Canlı olayın birebir tekrarı: oyun yokken 15+ dk idle ---------------

def test_main_wiring_game_absent_15min_idle_no_panic(monkeypatch):
    """Oyun kapalıyken 15,1 dk toplama yoksa bile IDLE tetiklenmez.

    2026-11-11 13:52:01 canlı olayının v19'daki karşılığı: Watchdog
    "Game window not found" derken (detektör False) poll her turda
    idle sayacını sıfırlar; kilit asla tetiklenmez.
    """
    panic = PanicAlertSystem(enable_sound=False)
    assert panic.idle_limit_sec == DEFAULT_IDLE_LIMIT_SEC
    present_state = {"present": False}
    try:
        _main_wiring(monkeypatch, panic, present_state)
        _age_idle_clock(panic)  # 15,1 dk idle — limit aşıldı.
        time.sleep(2.0)  # ~4 poll turu: sayaç her turda sıfırlanmalı.
        assert panic.panic_triggered is False
        assert panic.get_last_panic_event() is None
        # Sayaç durdu/sıfırlandı: son kontrolden beri geçen süre küçüktür.
        assert time.monotonic() - panic._last_gather_ts < 5.0
        # Detektör gerçekten danışılıyor ve False dönüyor.
        assert panic._game_window_present_detector() is False
    finally:
        panic.stop_watchdog_polling()


def test_main_wiring_binds_game_window_detector_from_worker_watchdog(monkeypatch):
    """Bind, worker.watchdog.is_game_window_present'i detektör yapar ve
    wiring status bunu doğrular (main.py:840 bağlantısının kanıtı)."""
    panic = PanicAlertSystem(enable_sound=False)
    present_state = {"present": False}
    try:
        _watchdog, _worker, wiring = _main_wiring(
            monkeypatch, panic, present_state
        )
        assert wiring["wired"]["game_window_present"] is True
        status = panic.get_watchdog_wiring_status()
        assert status["detectors"]["game_window_present"] is True
        assert "game_window_present" not in status["missing_required_detectors"]
    finally:
        panic.stop_watchdog_polling()


# --- 2) Kontrol: aynı kurulumda oyun varken 15+ dk idle tetikler -------------

def test_main_wiring_game_present_15min_idle_triggers(monkeypatch):
    """Oyun açıkken (detektör True) 15,1 dk idle IDLE_TIMEOUT_15M tetikler.

    1. testin kontrol grubudur: tetiklenmeme, detektörün False olmasından
    kaynaklanır — poll/check zincirinin körleşmesinden değil.
    """
    panic = PanicAlertSystem(enable_sound=False)
    present_state = {"present": True}
    try:
        _main_wiring(monkeypatch, panic, present_state)
        _age_idle_clock(panic)
        assert _wait_until(
            lambda: panic.get_last_panic_event() is not None, timeout=5.0
        )
        assert panic.panic_triggered is True
        assert panic.get_last_panic_event()["reason"] == "IDLE_TIMEOUT_15M"
    finally:
        panic.stop_watchdog_polling()


# --- 3) Geçiş: oyun kapalıyken duran sayaç, oyun gelince sıfırdan başlar -----

def test_main_wiring_absent_then_present_triggers_after_limit(monkeypatch):
    """Oyun kapalıyken sayaç durur; oyun geri gelince sayaç sıfırdan
    başlar — hemen tetiklenmez, limit yeniden dolunca tetikler."""
    panic = PanicAlertSystem(enable_sound=False)
    present_state = {"present": False}
    try:
        _main_wiring(monkeypatch, panic, present_state)
        _age_idle_clock(panic)
        time.sleep(1.5)
        assert panic.panic_triggered is False
        assert panic.get_last_panic_event() is None
        # Oyun geri geldi ama sayaç az önce sıfırlandı — hemen tetiklenmez.
        present_state["present"] = True
        time.sleep(1.5)
        assert panic.panic_triggered is False
        assert panic.get_last_panic_event() is None
        # Limit yeniden dolunca (oyun açıkken) idle aynen tetikler.
        _age_idle_clock(panic)
        assert _wait_until(
            lambda: panic.get_last_panic_event() is not None, timeout=5.0
        )
        assert panic.get_last_panic_event()["reason"] == "IDLE_TIMEOUT_15M"
    finally:
        panic.stop_watchdog_polling()


# --- 4) HUD: üretim yoklama yoluyla şerit kilit başına bir kez açılır --------

def _make_hud(monkeypatch):
    """Disk panik kaydından arındırılmış OverlayHUD (adversarial harness kalıbı)."""
    from PyQt6.QtWidgets import QApplication

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


def _strip_open_count(caplog):
    return len(
        [r for r in caplog.records if "HUD panik şeridi açıldı" in r.message]
    )


def test_hud_strip_opens_once_via_production_poll_path(monkeypatch, caplog):
    """Kilitliyken HUD 5 sn'lik üretim yoklaması şeridi BİR KEZ açar.

    Üretim yolu: f9_menu.bind_targets → overlay.set_panic_lock
    (ui/f9_menu.py:1510-1521); HUD _update_elapsed_time içinde 5 sn'de
    bir lock.get_panic_status() → update_panic_status çağırır
    (ui/overlay_hud.py:3125-3130). get_panic_status()'un elapsed_sec'i
    her yoklamada büyür ama dedup imzası onu içermez — canlı olaydaki
    422 sn'lik her-saniye-tekrarı bu testte yakalanır.
    """
    hud, app = _make_hud(monkeypatch)
    panic = PanicAlertSystem(enable_sound=False)
    try:
        # Kilidi üretimdeki gibi tetikle (oyun açık, 15,1 dk idle).
        _age_idle_clock(panic)
        event = panic.check_watchdog_conditions(
            play_siren=False, game_window_present=True
        )
        assert event is not None
        assert event["reason"] == "IDLE_TIMEOUT_15M"

        # Üretim bağlantısı: F9Menu.bind_targets'ın yaptığı set_panic_lock.
        hud.set_panic_lock(panic)
        assert hud._panic_lock is panic

        # Üretim yoklama yolunu sür: _update_elapsed_time her çağrıda
        # _slow_tick'i artırır, 5'e bölününce panik yoklaması yapar.
        # _slow_tick'i 4'e kurarak her çağrı yoklama yapar.
        with caplog.at_level(logging.DEBUG, logger="ui.overlay_hud"):
            for _ in range(6):
                hud._slow_tick = 4
                hud._update_elapsed_time()
                app.processEvents()
                time.sleep(0.6)  # elapsed_sec her yoklamada büyüsün.
        assert _strip_open_count(caplog) == 1
        label = getattr(hud, "panic_label", None) or getattr(hud, "_panic_label", None)
        assert label is not None and label.isVisible()
        assert "IDLE_TIMEOUT_15M" in label.text()
    finally:
        hud.close()
        app.processEvents()
