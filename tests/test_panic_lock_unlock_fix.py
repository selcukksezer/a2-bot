# -*- coding: utf-8 -*-
"""
fix/panic-lock — PANIC-LOCK kalıcı girdi kilidi düzeltme testleri.

Regresyon: resume() yalnızca panic bayraklarını (_inputs_halted /
_panic_triggered) temizliyor, emergency_halt_inputs'un maymun-yamaladığı
girdi metotlarını (press_key, key_down, key_up, click, ...) geri
yüklemiyordu — bayraklar "açık" görünürken bot girdi-ölü kalıyordu
(canlı log 20:15:13'te 3x "[PANIC-LOCK] Girdi engellendi"). Tek gerçek
açma yolu reset_panic()/unlock_inputs'tur; resume()/stop() artık onu
çağırır (_unlock_panic_lock) ve açmayı doğrular.

Ayrıca: tetikleyici sebebi yalnızca PC belleğindeki _last_panic_event /
_panic_history'de kalıyor, dosyaya yazılmıyordu. Artık trigger_panic
sebep + detay içeren CRITICAL satırı yazar (logs/bot.log), engel
uyarısı sebebi taşır, get_panic_status()/diagnostic snapshot sebebi
+ süreyi döndürür, resume() HUD altyazısına sebebi yazar.

Kapsam notu: kilit yalnızca botun InputController metotlarını kapsar —
fiziksel klavye o yoldan geçmez. _hw_key_up düşük seviye bırakma yolu
patch'li key_up'tan geçmez; bu dosya onu değiştirmez, yalnızca
bırakma yolunun kilitliyken de çalıştığını doğrular.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.opsec_crypto import PanicAlertSystem
from core.state_machine import BotMode, BotState, BotWorker


class _FakeInputCtrl:
    """emergency_halt_inputs'un yamladığı metotlara sahip minimal denetleyici."""

    def __init__(self) -> None:
        self.halted = False
        self.pressed = []
        self._active_bot_keys = set()
        self._held_movement = set()

    def press_key(self, key, **kwargs):
        self.pressed.append(str(key).lower())
        return True

    def key_down(self, key, **kwargs):
        self._active_bot_keys.add(str(key).lower())
        return True

    def key_up(self, key, **kwargs):
        self._active_bot_keys.discard(str(key).lower())
        return True

    def release_key(self, key):
        return self.key_up(key)

    def click(self, x=0, y=0):
        return True

    def right_click(self, x=0, y=0):
        return True

    def left_click(self, x=0, y=0):
        return True

    def move_mouse(self, x=0, y=0):
        return True

    def human_move_and_click(self, x=0, y=0):
        return True

    def send_input(self, *args, **kwargs):
        return True

    def mount_up(self):
        return True

    def release_all_inputs(self):
        return True

    def _hw_key_up(self, key, post_to_window=False):
        # Düşük seviye bırakma yolu — patch'li key_up'tan geçmez.
        return True

    def abort_motion(self):
        return None

    def release_all_movement_keys(self):
        return None

    def movement_active(self):
        return False

    def resume_motion(self):
        return None


def _same_method(a, b):
    """Bound-method kimlik karşılaştırması.

    Python'da `obj.method` her erişimde YENİ bir bound-method nesnesi
    üretir, bu yüzden iki ayrı erişimde `a is b` her zaman False döner.
    Aynı metot = aynı alttaki işlev (__func__) + aynı bağlı nesne
    (__self__). Düz fonksiyonlar (kilitleme sırasında instance
    özniteliğine yazılan _blocked_input gibi) __func__/__self__ taşımaz;
    onlar için `is` karşılaştırması doğrudur.
    """
    if a is b:
        return True
    func_a = getattr(a, "__func__", None)
    func_b = getattr(b, "__func__", None)
    if func_a is None or func_b is None:
        return False
    return func_a is func_b and a.__self__ is b.__self__


def _trigger_idle_panic(panic, ctrl):
    """Watchdog yolundan IDLE_TIMEOUT_15M tetikler (60 dk toplama yok)."""
    panic._last_gather_ts -= 3600.0
    event = panic.check_watchdog_conditions(input_ctrl=ctrl, play_siren=False)
    assert event is not None, "watchdog IDLE panik tetiklemedi"
    assert event["reason"] == "IDLE_TIMEOUT_15M"
    return event


# --- opsec_crypto: sebep görünürlüğü ----------------------------------------

def test_trigger_panic_logs_reason_at_critical(caplog):
    """Tetikleyici CRITICAL satırında sebep + süreyle yazar (logs/bot.log)."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    ctrl = _FakeInputCtrl()
    with caplog.at_level(logging.DEBUG, logger="OpsecCrypto"):
        event = _trigger_idle_panic(panic, ctrl)
    assert "60.0 dakika" in event["details"]
    assert ctrl.press_key("f") is False
    criticals = [r for r in caplog.records if r.levelname == "CRITICAL"]
    assert any(
        "IDLE_TIMEOUT_15M" in r.message and "60.0 dakika" in r.message
        for r in criticals
    ), "CRITICAL satırında sebep + süre yok"


def test_blocked_input_warning_names_trigger_reason(caplog):
    """Engel uyarısı artık yalnızca 'kilit aktif' demez, sebebi taşır."""
    panic = PanicAlertSystem(enable_sound=False)
    ctrl = _FakeInputCtrl()
    with caplog.at_level(logging.DEBUG, logger="OpsecCrypto"):
        panic.trigger_panic(
            reason="COMBAT_TIMEOUT_2M",
            details="Savaş süresi 2 dakika sınırını aştı (135.2s).",
            input_ctrl=ctrl,
            play_siren=False,
        )
        assert ctrl.press_key("2") is False
    assert any(
        r.message.startswith("[PANIC-LOCK]")
        and "COMBAT_TIMEOUT_2M" in r.message
        and "135.2s" in r.message
        for r in caplog.records
        if r.levelname == "WARNING"
    )


def test_reset_panic_restores_original_methods_by_identity():
    """reset_panic() yamalı metotları geri yükler (aynı __func__ + __self__)."""
    panic = PanicAlertSystem(enable_sound=False)
    ctrl = _FakeInputCtrl()
    orig_press = ctrl.press_key
    orig_key_up = ctrl.key_up
    panic.trigger_panic(
        reason="CHARACTER_DEAD",
        details="Karakter ölüm ekranı algılandı!",
        input_ctrl=ctrl,
        play_siren=False,
    )
    assert not _same_method(ctrl.press_key, orig_press)
    assert ctrl.press_key("f") is False
    assert ctrl.halted is True  # öznitelik de kilitlendi

    assert panic.reset_panic() is True

    assert _same_method(ctrl.press_key, orig_press)
    assert _same_method(ctrl.key_up, orig_key_up)
    assert ctrl.press_key("f") is True
    assert ctrl.halted is False
    assert panic.inputs_halted is False
    assert panic.panic_triggered is False
    assert panic.get_panic_status()["reason"] is None


def test_get_panic_status_and_diagnostic_snapshot_carry_reason():
    """Sebep + detay + süre, status ve diagnostic snapshot'ta görünür."""
    panic = PanicAlertSystem(enable_sound=False)
    ctrl = _FakeInputCtrl()
    event = panic.trigger_panic(
        reason="GM_INSPECTION_ALERT",
        details="Olası GM mesajı algılandı!",
        input_ctrl=ctrl,
        play_siren=False,
    )
    status = panic.get_panic_status()
    assert status["reason"] == "GM_INSPECTION_ALERT"
    assert status["details"] == "Olası GM mesajı algılandı!"
    assert status["timestamp"] == event["timestamp"]
    assert status["elapsed_sec"] is not None and status["elapsed_sec"] >= 0.0
    assert status["panic_triggered"] is True
    assert status["inputs_halted"] is True

    snap = panic.get_opsec_crypto_diagnostic_snapshot()
    assert snap["status"] == "panic"
    assert snap["panic_reason"] == "GM_INSPECTION_ALERT"
    assert snap["panic_details"] == "Olası GM mesajı algılandı!"
    assert snap["panic_elapsed_sec"] is not None


def test_no_retrigger_while_panicked_keeps_single_event():
    """F-0041: kilitliyken periyodik denetim yeniden tetiklemez."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    ctrl = _FakeInputCtrl()
    _trigger_idle_panic(panic, ctrl)
    blocked_fn = ctrl.press_key
    history_len = len(panic.get_panic_history())

    # Koşul sürüyor (hâlâ 60 dk toplama yok) — yine de None dönmeli.
    assert panic.check_watchdog_conditions(input_ctrl=ctrl, play_siren=False) is None
    # Ölüm sinyali bile kilitliyken yeniden tetiklemez.
    assert (
        panic.check_watchdog_conditions(
            input_ctrl=ctrl, character_dead=True, play_siren=False
        )
        is None
    )
    assert len(panic.get_panic_history()) == history_len
    assert ctrl.press_key is blocked_fn  # yeniden yamalanmadı
    assert panic.get_panic_status()["reason"] == "IDLE_TIMEOUT_15M"


def test_reset_panic_resets_timers_no_immediate_retrigger():
    """reset_panic() zamanlayıcıları sıfırlar; koşul sürse de hemen dönmez."""
    panic = PanicAlertSystem(idle_limit_sec=60.0, combat_limit_sec=30.0, enable_sound=False)
    ctrl = _FakeInputCtrl()
    _trigger_idle_panic(panic, ctrl)
    assert panic.panic_triggered is True
    assert panic.reset_panic() is True
    assert panic.check_watchdog_conditions(input_ctrl=ctrl, play_siren=False) is None


# --- BotWorker: resume()/stop() kilidi GERÇEKTEN açar ----------------------

def _make_worker_with_fake_driver():
    """Gerçek BotWorker.__init__ (C7 kalıbı) + sahte sürücü + gerçek PanicAlertSystem."""
    driver = _FakeInputCtrl()
    combat_recovery = MagicMock()
    combat_recovery.is_dead = False
    combat_recovery.in_combat = False
    with patch.object(BotWorker, "_init_optional_subsystem", return_value=None):
        worker = BotWorker(
            config={"test_mode": True},
            vision=MagicMock(),
            input_driver=driver,
            route_manager=MagicMock(),
            combat_recovery=combat_recovery,
            navigator=None,
            recorder=None,
        )
    worker.logs = []
    worker.hud_calls = []
    worker._emit_log = lambda msg: worker.logs.append(str(msg))
    worker._emit_hud_snapshot = lambda **kwargs: worker.hud_calls.append(dict(kwargs)) or {}
    worker._mode = BotMode.ROUTE_MODE.value
    worker._state = BotState.PAUSED.value
    worker._paused = True
    worker.panic_system = PanicAlertSystem(enable_sound=False)
    return worker, driver


def test_resume_really_unlocks_panic_blocked_inputs():
    """Ana regresyon: resume() sonrası press_key yeniden GERÇEK metottur."""
    worker, driver = _make_worker_with_fake_driver()
    orig_press = driver.press_key
    _trigger_idle_panic(worker.panic_system, driver)
    assert driver.press_key("f") is False  # kilitli

    worker.resume()

    assert _same_method(driver.press_key, orig_press)  # metot GERİ YÜKLENDİ
    assert driver.press_key("f") is True
    assert worker.panic_system.panic_triggered is False
    assert worker.panic_system.inputs_halted is False
    assert worker.is_paused is False
    assert any("IDLE_TIMEOUT_15M" in m for m in worker.logs), "sebep logda yok"
    assert any(
        "IDLE_TIMEOUT_15M" in (c.get("subtitle") or "") for c in worker.hud_calls
    ), "sebep HUD altyazısında yok"


def test_stop_resets_panic_lock():
    """stop() da kilidi gerçekten açar — duran bot kilitli metotla kalmaz."""
    worker, driver = _make_worker_with_fake_driver()
    orig_press = driver.press_key
    _trigger_idle_panic(worker.panic_system, driver)
    assert driver.press_key("f") is False

    worker.stop()

    assert _same_method(driver.press_key, orig_press)
    assert driver.press_key("f") is True
    assert worker.panic_system.panic_triggered is False
    assert worker.panic_system.inputs_halted is False
    assert any("IDLE_TIMEOUT_15M" in m for m in worker.logs)


def test_resume_without_panic_does_not_touch_inputs():
    """Panik yokken resume() eskisi gibi davranır, kilit mesajı üretmez."""
    worker, driver = _make_worker_with_fake_driver()
    orig_press = driver.press_key
    worker.resume()
    assert _same_method(driver.press_key, orig_press)
    assert worker.panic_system.panic_triggered is False
    assert not any("Panik kilidi" in m for m in worker.logs)


def test_resume_with_mock_panic_system_does_not_crash():
    """MagicMock panic_system (test_g1 kalıbı) resume()u bozmaz."""
    worker, driver = _make_worker_with_fake_driver()
    worker.panic_system = MagicMock()
    worker.resume()  # çökmemeli
    assert worker.is_paused is False


def test_hw_key_up_release_path_bypasses_blocked_key_up():
    """Bırakma yolu (_hw_key_up) kilitliyken de çalışır — Görev 3'ün alanı,
    burada yalnızca bozulmadığı doğrulanır."""
    panic = PanicAlertSystem(enable_sound=False)
    ctrl = _FakeInputCtrl()
    panic.trigger_panic(
        reason="PROMPT_STUCK_10S",
        details="Prompt stuck for 10s (12.0s boyunca etkileşim istemi kapanmadı).",
        input_ctrl=ctrl,
        play_siren=False,
    )
    assert ctrl.key_up("w") is False  # patch'li key_up engelli
    assert ctrl._hw_key_up("w") is True  # düşük seviye bırakma açık
