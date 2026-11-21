# -*- coding: utf-8 -*-
"""
R22 — core/opsec_crypto.py düzeltme testleri (F-0163, S-1002).

F-0163: _poll_loop artık sabit False/None ile çağırmaz; kayıtlı detektörleri
her turda danışır ve gather/combat/prompt sinyallerini ilgili setter'larla
tüketir.
S-1002: get_opsec_crypto_health_summary artık gerçek şifrele/çöz probudur.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.opsec_crypto import LocalVaultCrypto, PanicAlertSystem


def _make_vault(tmp_path):
    return LocalVaultCrypto(
        vault_path=str(tmp_path / "settings.enc"),
        salt_path=str(tmp_path / ".vault_salt"),
    )


# --- S-1002: sağlık özeti gerçek prob ---------------------------------------

def test_r22_health_summary_roundtrip_ok(tmp_path):
    vault = _make_vault(tmp_path)
    summary = vault.get_opsec_crypto_health_summary()
    assert summary["opsec_crypto_ready"] is True
    assert summary["roundtrip_ok"] is True
    assert summary["cipher"]
    assert summary["vault_path"] == str(tmp_path / "settings.enc")


def test_r22_health_summary_broken_crypto_reports_false(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setattr(
        vault, "encrypt_json_payload", lambda data: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    summary = vault.get_opsec_crypto_health_summary()
    assert summary["opsec_crypto_ready"] is False
    assert summary["roundtrip_ok"] is False
    assert "error" in summary


def test_r22_health_summary_missing_key_material_reports_false(tmp_path, monkeypatch):
    vault = _make_vault(tmp_path)
    monkeypatch.setattr(vault, "_master_key", b"")
    summary = vault.get_opsec_crypto_health_summary()
    assert summary["opsec_crypto_ready"] is False
    assert summary["roundtrip_ok"] is False


# --- F-0163: setter'lar iç durumu günceller ---------------------------------

def test_r22_record_gather_resets_idle_timer():
    panic = PanicAlertSystem(idle_limit_sec=60.0, enable_sound=False)
    panic._last_gather_ts -= 30.0
    before = panic._last_gather_ts
    panic.record_successful_gather()
    assert panic._last_gather_ts > before


def test_r22_set_combat_state_tracks_start_and_clears():
    panic = PanicAlertSystem(enable_sound=False)
    assert panic._combat_start_ts is None
    panic.set_combat_state(True)
    assert panic._combat_start_ts is not None
    panic.set_combat_state(True)  # tekrar True başlangıcı sıfırlamaz
    first = panic._combat_start_ts
    panic.set_combat_state(True)
    assert panic._combat_start_ts == first
    panic.set_combat_state(False)
    assert panic._combat_start_ts is None


def test_r22_register_detectors_only_overwrites_given():
    panic = PanicAlertSystem(enable_sound=False)
    panic.register_watchdog_detectors(character_dead_detector=lambda: True)
    assert panic._character_dead_detector is not None
    assert panic._gm_whisper_detector is None
    panic.register_watchdog_detectors(gm_whisper_detector=lambda: False)
    assert panic._character_dead_detector is not None  # korunur
    assert panic._gm_whisper_detector is not None


def test_r22_broken_detector_does_not_kill_poll():
    panic = PanicAlertSystem(enable_sound=False)
    assert panic._call_detector(None, "none") is None
    assert panic._call_detector(lambda: 1 / 0, "boom") is None
    assert panic._call_detector(lambda: "x", "ok") == "x"


# --- F-0163: poll döngüsü detektörleri tüketir -------------------------------

def _wait_until(predicate, timeout=5.0, step=0.05):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def test_r22_poll_character_dead_detector_triggers_panic():
    panic = PanicAlertSystem(enable_sound=False)
    try:
        assert panic.start_watchdog_polling(
            interval_sec=0.5,
            play_siren=False,
            character_dead_detector=lambda: True,
        )
        assert _wait_until(lambda: panic.get_last_panic_event() is not None)
        event = panic.get_last_panic_event()
        assert event is not None and event["reason"] == "CHARACTER_DEAD"
    finally:
        panic.stop_watchdog_polling()


def test_r22_poll_gm_whisper_detector_triggers_panic():
    panic = PanicAlertSystem(enable_sound=False)
    try:
        panic.start_watchdog_polling(
            interval_sec=0.5,
            play_siren=False,
            gm_whisper_detector=lambda: True,
        )
        assert _wait_until(lambda: panic.get_last_panic_event() is not None)
        assert panic.get_last_panic_event()["reason"] == "GM_INSPECTION_ALERT"
    finally:
        panic.stop_watchdog_polling()


def test_r22_poll_combat_detector_triggers_timeout():
    panic = PanicAlertSystem(combat_limit_sec=0.3, enable_sound=False)
    try:
        panic.start_watchdog_polling(
            interval_sec=0.5,
            play_siren=False,
            combat_state_detector=lambda: True,
        )
        assert _wait_until(lambda: panic.get_last_panic_event() is not None, timeout=5.0)
        assert panic.get_last_panic_event()["reason"] == "COMBAT_TIMEOUT_2M"
    finally:
        panic.stop_watchdog_polling()


def test_r22_poll_prompt_visible_detector_triggers_stuck():
    panic = PanicAlertSystem(prompt_stuck_sec=0.3, enable_sound=False)
    try:
        with panic._lock:
            panic._prompt_stuck_grace_until = 0.0
        panic.start_watchdog_polling(
            interval_sec=0.5,
            play_siren=False,
            prompt_visible_detector=lambda: True,
        )
        assert _wait_until(lambda: panic.get_last_panic_event() is not None, timeout=5.0)
        event = panic.get_last_panic_event()
        assert event["reason"] == "PROMPT_STUCK_10S"
        # F-0049: soft-pause — hard panik kilidi yok.
        assert event.get("soft_pause") is True
        assert panic.panic_triggered is False
    finally:
        panic.stop_watchdog_polling()


def test_r22_poll_gather_heartbeat_prevents_idle_panic():
    panic = PanicAlertSystem(idle_limit_sec=1.0, enable_sound=False)
    try:
        panic.start_watchdog_polling(
            interval_sec=0.5,
            play_siren=False,
            gather_heartbeat_detector=lambda: True,
        )
        time.sleep(1.8)  # idle limiti aşar ama her tur heartbeat sıfırlar
        assert not panic.panic_triggered
        assert panic.get_last_panic_event() is None
    finally:
        panic.stop_watchdog_polling()


def test_r22_poll_without_heartbeat_idle_panic_fires():
    panic = PanicAlertSystem(idle_limit_sec=0.6, enable_sound=False)
    try:
        panic.start_watchdog_polling(interval_sec=0.5, play_siren=False)
        assert _wait_until(lambda: panic.get_last_panic_event() is not None, timeout=5.0)
        assert panic.get_last_panic_event()["reason"] == "IDLE_TIMEOUT_15M"
    finally:
        panic.stop_watchdog_polling()
