# -*- coding: utf-8 -*-
"""
fix/v20-rss-panik — regresyon testleri.

Canlı kanıt (Selçuk, PC, 2026-11-12 20:57-21:29 oturumu):
(a) RTX eğitimi (yolo11s, 50 epoch) RSS 10315 MB'ta Cursor'ın sabit
    10240 MB abort eşiğiyle durduruldu — oysa 11 Kasım gecesi yolo11m
    koşusu 19-21 GB bandında plato yapıp mAP50 0.685'e ulaşmıştı
    (32 GB RAM'de güvenli). Eşik RAM-orantılı olmalı.
(b) Eğitim durduktan SONRA da RSS ~300 MB/dk artmaya devam etti
    (10.6 -> 11.5 GB); tek seferlik bellek dökümü hangi turun
    büyüdüğünü göstermez — tekrarlı döküm + tracemalloc farkı lazım.
(c) IDLE_TIMEOUT_15M 21:12:40'ta tetiklendi — oyun simge durumundayken
    ve eğitim sürerken. Watchdog pencereyi "var" sayıyor, idle sayacı
    işliyordu. Simge durumu oyun-yokluğu gibi sayacı dondurmalı.
"""

from __future__ import annotations

import logging
import os
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core import crash_evidence
from core.crash_evidence import _maybe_memory_dump, _top_object_types
from core.opsec_crypto import PanicAlertSystem
from core.watchdog import BotWatchdog
from dataset_factory.rtx_trainer_engine import (
    RTXTrainerEngine,
    _RssAbortDecider,
    default_rss_abort_mb,
)


# --- (a) RAM-orantılı RSS abort ------------------------------------------------

def test_default_rss_abort_mb_scales_with_ram():
    # 32 GB -> %75 = 24576 MB (eski sabit 10240 değil).
    assert default_rss_abort_mb(32768.0) == pytest.approx(24576.0)
    # 16 GB -> %75 = 12288 MB.
    assert default_rss_abort_mb(16384.0) == pytest.approx(12288.0)


def test_default_rss_abort_mb_floor():
    # Küçük RAM'de taban 10240 MB korunur.
    assert default_rss_abort_mb(8192.0) == pytest.approx(10240.0)
    assert default_rss_abort_mb(0.0) == pytest.approx(10240.0)
    assert default_rss_abort_mb(-100.0) == pytest.approx(10240.0)


def test_default_rss_abort_mb_falls_back_when_ram_unknown(monkeypatch):
    monkeypatch.setattr(
        "dataset_factory.rtx_trainer_engine.total_system_ram_mb",
        lambda: None,
    )
    assert default_rss_abort_mb() == pytest.approx(10240.0)


def test_build_config_exposes_rss_abort_knob():
    cfg = RTXTrainerEngine.build_rtx_training_config_static()
    assert "rss_abort_mb" in cfg
    # None = RAM-orantılı varsayılan (açık ayar değil).
    assert cfg["rss_abort_mb"] is None


def test_rss_decider_requires_consecutive_confirmations():
    decider = _RssAbortDecider(1000.0, confirmations=2)
    assert decider.observe(999.0) is False
    assert decider.observe(1001.0) is False  # 1. aşım — tek başına yetmez
    assert decider.observe(1002.0) is True  # 2. ardışık aşım — abort


def test_rss_decider_resets_on_below_threshold_reading():
    decider = _RssAbortDecider(1000.0, confirmations=2)
    assert decider.observe(1001.0) is False
    assert decider.observe(500.0) is False  # eşik altı — seri sıfırlanır
    assert decider.observe(1001.0) is False
    assert decider.observe(1002.0) is True


def test_rss_decider_disabled_when_zero_or_negative():
    assert _RssAbortDecider(0.0).observe(999999.0) is False
    assert _RssAbortDecider(-5.0).observe(999999.0) is False


def test_rss_decider_emergency_fires_immediately():
    decider = _RssAbortDecider(
        24000.0, confirmations=2, emergency_fraction=0.90, total_ram_mb=32000.0
    )
    # %90 = 28800 MB — tek okumada abort (teyit beklenmez).
    assert decider.observe(29000.0) is True


def test_rss_decider_ignores_unreadable_rss_without_resetting():
    decider = _RssAbortDecider(1000.0, confirmations=2)
    assert decider.observe(1001.0) is False
    assert decider.observe(None) is False  # okunamadı — seri bozulmaz
    assert decider.observe(1002.0) is True


# --- (c) Simge-durumu panik donması --------------------------------------------

def _aged_idle_panic():
    panic = PanicAlertSystem(
        idle_limit_sec=900.0, combat_limit_sec=120.0, enable_sound=False
    )
    panic._last_gather_ts = time.monotonic() - 1000.0  # 16+ dk toplama yok
    return panic


def test_minimized_freezes_idle_timer():
    panic = _aged_idle_panic()
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=True,
    )
    assert event is None
    assert panic.panic_triggered is False
    # Sayaç "şimdi"ye çekildi — oyun geri gelince sıfırdan başlar.
    assert time.monotonic() - panic._last_gather_ts < 5.0


def test_minimized_false_still_panics_on_idle():
    panic = _aged_idle_panic()
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=False,
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_minimized_none_preserves_old_behavior():
    panic = _aged_idle_panic()
    event = panic.check_watchdog_conditions(
        play_siren=False, game_window_present=True, game_window_minimized=None
    )
    assert event is not None
    assert event["reason"] == "IDLE_TIMEOUT_15M"


def test_minimized_freezes_combat_timer():
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    panic.set_combat_state(True)
    panic._combat_start_ts = time.monotonic() - 200.0
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=True,
    )
    assert event is None
    assert panic._combat_start_ts is None


def test_minimized_does_not_suppress_death():
    panic = PanicAlertSystem(combat_limit_sec=120.0, enable_sound=False)
    event = panic.check_watchdog_conditions(
        play_siren=False,
        game_window_present=True,
        game_window_minimized=True,
        character_dead=True,
    )
    assert event is not None
    assert event["reason"] == "CHARACTER_DEAD"


def test_minimized_detector_wiring_status():
    panic = PanicAlertSystem(enable_sound=False)
    panic.register_watchdog_detectors(
        game_window_minimized_detector=lambda: True
    )
    status = panic.get_watchdog_wiring_status()
    assert status["detectors"]["game_window_minimized"] is True
    # Zorunlu sayılmaz — yokluğu "degraded" yapmaz.
    assert "game_window_minimized" not in status["missing_required_detectors"]


def test_minimized_detector_bound_from_watchdog():
    fake_watchdog = SimpleNamespace(
        is_game_window_present=lambda: True,
        is_game_window_minimized=lambda: True,
    )
    worker = SimpleNamespace(watchdog=fake_watchdog)
    panic = PanicAlertSystem(enable_sound=False)
    result = panic.bind_production_watchdog_sources(worker=worker)
    assert result["wired"]["game_window_minimized"] is True
    assert panic._game_window_minimized_detector is not None
    assert panic._game_window_minimized_detector() is True


def test_watchdog_minimized_accessor_simulated_and_safe():
    watchdog = object.__new__(BotWatchdog)
    watchdog._simulated_minimized = True
    watchdog.target_window_titles = []
    assert watchdog.is_game_window_minimized() is True
    watchdog._simulated_minimized = False
    # Linux'ta pencere yok -> False (eski davranış).
    assert watchdog.is_game_window_minimized() is False

    def _boom():
        raise RuntimeError("boom")

    watchdog._is_game_window_minimized = _boom  # type: ignore[method-assign]
    assert watchdog.is_game_window_minimized() is False


# --- (b) Tekrarlı bellek dökümü -------------------------------------------------

class _CapturingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args):
        self.warnings.append(msg % args if args else msg)

    def info(self, msg, *args):
        pass


@pytest.fixture()
def _clean_crash_state():
    crash_evidence._reset_state_for_tests()
    if tracemalloc.is_tracing():
        tracemalloc.stop()
    yield
    crash_evidence._reset_state_for_tests()
    if tracemalloc.is_tracing():
        tracemalloc.stop()


def test_memory_dump_below_threshold_does_nothing(_clean_crash_state):
    log = _CapturingLogger()
    dumped = _maybe_memory_dump(
        log, rss=1000.0, growth=100.0, now_mono=1000.0,
        warn_growth=512.0, warn_total=6144.0,
    )
    assert dumped is False
    assert log.warnings == []


def test_memory_dump_repeats_after_interval(_clean_crash_state):
    log = _CapturingLogger()
    kwargs = dict(rss=7000.0, growth=600.0, warn_growth=512.0, warn_total=6144.0)
    assert _maybe_memory_dump(log, now_mono=1000.0, **kwargs) is True
    assert len(log.warnings) == 1
    # Aralık dolmadan tekrar yok.
    assert _maybe_memory_dump(log, now_mono=1100.0, **kwargs) is False
    assert len(log.warnings) == 1
    # 300 sn sonra tekrar dökülür.
    assert _maybe_memory_dump(log, now_mono=1301.0, **kwargs) is True
    assert len(log.warnings) == 2
    assert "BELLEK UYARISI" in log.warnings[0]
    assert "tracemalloc" in log.warnings[0]


def test_memory_dump_tracemalloc_baseline_then_diff(_clean_crash_state):
    log = _CapturingLogger()
    kwargs = dict(rss=7000.0, growth=600.0, warn_growth=512.0, warn_total=6144.0)
    assert _maybe_memory_dump(log, now_mono=1000.0, **kwargs) is True
    assert "taban anlik goruntusu" in log.warnings[0]
    _hold = [bytearray(1024) for _ in range(200)]
    assert _maybe_memory_dump(log, now_mono=1400.0, **kwargs) is True
    assert len(log.warnings) == 2
    # İkinci dökümde fark satırı var (içerik ortama göre değişir, çökmez).
    assert "tracemalloc farki" in log.warnings[1]
    del _hold


def test_top_object_types_returns_counts():
    text = _top_object_types(limit=5)
    assert isinstance(text, str) and text
    assert "=" in text
