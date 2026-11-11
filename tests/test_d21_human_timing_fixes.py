# -*- coding: utf-8 -*-
"""D21 düzeltmeleri için testler (core/human_timing.py).

Kapsanan bulgular: F-0102, F-0103, F-0104, F-0276, F-0277, F-0278, F-0279, F-0444.
"""
import threading
import time

import pytest

from core.human_timing import (
    DynamicBreakScheduler,
    HumanTimingCoordinator,
    HumanTimingEngine,
    OrganicAntiAFK,
    align_to_gcd as module_align_to_gcd,
)
from core.human_timing import get_human_timing_engine, get_key_release_guard_sec


# --- F-0102: align_to_gcd hata dalı kendisi exception fırlatmamalı ---

@pytest.mark.parametrize("bad_gcd,bad_dur", [(None, 0.05), ("x", 0.05), (0.12, None), (0.12, "x"), (None, None)])
def test_align_to_gcd_error_branch_never_raises(bad_gcd, bad_dur):
    engine = HumanTimingEngine(seed=1)
    result = engine.align_to_gcd(bad_dur, gcd_sec=bad_gcd)
    assert isinstance(result, float)
    assert result >= 0.020
    # Modül düzeyi wrapper da aynı güvenceyi verir.
    wrapped = module_align_to_gcd(bad_dur, gcd_sec=bad_gcd)
    assert isinstance(wrapped, float) and wrapped >= 0.020


def test_align_to_gcd_valid_inputs_unchanged():
    engine = HumanTimingEngine(seed=1)
    # 0.05s burst, 0.12s GCD -> kalan ~0.07s (+jitter), en az 0.02s
    assert 0.020 <= engine.align_to_gcd(0.05, gcd_sec=0.12) <= 0.135
    # Burst GCD'yi aştıysa minimum gap döner.
    assert engine.align_to_gcd(0.20, gcd_sec=0.12, min_gap_sec=0.025) >= 0.025


# --- F-0277: zamanlayıcılar wall-clock yerine monotonic kullanır ---

def test_timing_state_uses_monotonic_clock():
    before = time.monotonic()
    engine = HumanTimingEngine(seed=1)
    sched = DynamicBreakScheduler(timing_engine=HumanTimingEngine(seed=1))
    afk = OrganicAntiAFK(timing_engine=HumanTimingEngine(seed=1))
    after = time.monotonic()
    assert before - 1.0 <= engine._session_start_ts <= after + 1.0
    assert before - 1.0 <= sched._last_macro_ts <= after + 1.0
    assert before - 1.0 <= afk._last_action_ts <= after + 1.0
    # Monotonic saat wall-clock'tan (epoch ~1.7e9) çok küçüktür.
    assert engine._session_start_ts < 1e8


# --- F-0276: mola sonrası kalan süre mola süresini çift saymaz ---

def test_macro_break_remaining_does_not_double_count_duration():
    sched = DynamicBreakScheduler(timing_engine=HumanTimingEngine(seed=7))
    t0 = time.monotonic()
    due_at = t0 + sched._next_macro_interval_sec + 1.0
    triggered, duration = sched.should_take_macro_break(now=due_at)
    assert triggered and 180.0 <= duration <= 540.0
    next_interval = sched._next_macro_interval_sec
    # Mola bittiği anda kalan süre = aralık - mola süresi (mola bir kez sayılır).
    status = sched.get_scheduler_status(now=due_at + duration)
    assert status["remaining_macro_sec"] == pytest.approx(next_interval - duration, abs=1e-6)
    # Mikro zamanlayıcı da aynı kurala uyar.
    assert status["remaining_micro_sec"] == pytest.approx(
        sched._next_micro_interval_sec - duration, abs=1e-6
    )


def test_micro_pause_remaining_does_not_double_count_duration():
    sched = DynamicBreakScheduler(timing_engine=HumanTimingEngine(seed=7))
    t0 = time.monotonic()
    due_at = t0 + sched._next_micro_interval_sec + 1.0
    triggered, duration = sched.should_take_micro_pause(now=due_at)
    assert triggered and 3.5 <= duration <= 9.0
    status = sched.get_scheduler_status(now=due_at + duration)
    assert status["remaining_micro_sec"] == pytest.approx(
        sched._next_micro_interval_sec - duration, abs=1e-6
    )


# --- F-0278: get_break_scheduler_status gerçek alanı okur ---

def test_break_scheduler_status_tracks_real_break_state():
    sched = DynamicBreakScheduler(timing_engine=HumanTimingEngine(seed=7))
    assert hasattr(sched, "_is_on_break")
    assert sched.get_break_scheduler_status()["is_on_break"] is False
    t0 = time.monotonic()
    due_at = t0 + sched._next_macro_interval_sec + 1.0
    triggered, duration = sched.should_take_macro_break(now=due_at)
    assert triggered
    assert sched.get_break_scheduler_status(now=due_at + 1.0)["is_on_break"] is True
    assert sched.get_break_scheduler_status(now=due_at + duration + 1.0)["is_on_break"] is False


# --- F-0104: girdi metodu yokken Anti-AFK sessizce "başarı" döndürmez ---

def test_anti_afk_without_input_method_is_not_success():
    afk = OrganicAntiAFK(timing_engine=HumanTimingEngine(seed=3))
    result = afk.execute_organic_anti_afk(object())
    assert result.get("success") is False
    assert result.get("executed") is False
    assert result.get("error")
    # Uygulanamayan eylem "uygulanmış" sayılmaz ve tam aralıkla ertelenmez.
    assert afk.total_actions_executed == 0
    assert result["next_interval_sec"] == 30.0


def test_anti_afk_jump_without_press_key_is_not_success():
    afk = OrganicAntiAFK(timing_engine=HumanTimingEngine(seed=3))
    result = afk._execute_organic_jump(object())
    assert result.get("success") is False and result.get("error")


# --- F-0279: reentrancy koruması + hata dalında zamanlayıcı güncellenir ---

class _BoomCtrl:
    """test_mode okunurken patlayan girdi denetleyicisi."""

    @property
    def test_mode(self):
        raise RuntimeError("boom")


def test_anti_afk_error_branch_updates_timer():
    afk = OrganicAntiAFK(timing_engine=HumanTimingEngine(seed=3))
    before = time.monotonic()
    result = afk.execute_organic_anti_afk(_BoomCtrl())
    assert result["action"] == "error"
    assert result["next_interval_sec"] == 180.0
    # Hata sonrası zamanlayıcı güncellendi: 10 sn sonra tekrar tetiklenmez.
    assert before - 1.0 <= afk._last_action_ts <= time.monotonic() + 1.0
    assert afk._next_interval_sec == 180.0
    assert afk.should_trigger_anti_afk(now=afk._last_action_ts + 10.0)[0] is False


class _BlockingCtrl:
    """press_key / fare basma çağrısında kilitlenen girdi denetleyicisi."""

    test_mode = False

    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def _block(self):
        self.entered.set()
        assert self.release.wait(timeout=10.0)

    def press_key(self, *args, **kwargs):
        self._block()
        return 0.08

    def _hw_mouse_down(self, *args, **kwargs):
        self._block()

    def _hw_mouse_up(self, *args, **kwargs):
        pass

    def _hw_move_to(self, *args, **kwargs):
        pass


def test_anti_afk_reentrancy_returns_busy():
    afk = OrganicAntiAFK(timing_engine=HumanTimingEngine(seed=3))
    ctrl = _BlockingCtrl()
    results = []

    def _run():
        results.append(afk.execute_organic_anti_afk(ctrl))

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    assert ctrl.entered.wait(timeout=10.0), "ilk eylem girdi metoduna ulaşmadı"
    busy = afk.execute_organic_anti_afk(ctrl)
    assert busy["action"] == "busy"
    assert busy.get("success") is False
    ctrl.release.set()
    worker.join(timeout=10.0)
    assert not worker.is_alive()
    assert len(results) == 1 and results[0].get("success") is True
    assert afk.total_actions_executed == 1


# --- F-0444: modül wrapper'ları tekil motorla aynı RNG akışını paylaşır ---

def test_module_wrappers_share_singleton_rng_stream():
    engine = get_human_timing_engine()
    before = engine._total_samples_generated
    value = get_key_release_guard_sec()
    assert 0.016 <= value <= 0.028
    assert engine._total_samples_generated > before


# --- F-0103: koordinatör molaları + Anti-AFK'yi tek tick'te çalıştırır ---

def test_coordinator_tick_drives_breaks_and_anti_afk():
    engine = HumanTimingEngine(seed=11)
    sched = DynamicBreakScheduler(timing_engine=engine)
    afk = OrganicAntiAFK(timing_engine=engine)
    coord = HumanTimingCoordinator(timing_engine=engine, break_scheduler=sched, anti_afk=afk)

    # Anti-AFK zamanı gelmiş, girdi yok -> pending, sessiz başarı yok.
    afk._last_action_ts = time.monotonic() - 1000.0
    out = coord.tick(input_ctrl=None)
    assert out["macro_break"] is False
    assert out["anti_afk"]["action"] == "pending"
    assert out["anti_afk"]["success"] is False

    # Makro mola zamanı gelmiş -> mola döndürülür (tick kendisi uyumaz).
    sched._next_macro_interval_sec = 1.0
    sched._last_macro_ts = time.monotonic() - 10.0
    out = coord.tick(input_ctrl=object())
    assert out["macro_break"] is True
    assert 180.0 <= out["macro_break_duration_sec"] <= 540.0

    # Henüz zamanı gelmemişse Anti-AFK "waiting" döner.
    afk._last_action_ts = time.monotonic()
    afk._next_interval_sec = 200.0
    sched._next_macro_interval_sec = 3600.0
    sched._last_macro_ts = time.monotonic()
    sched._next_micro_interval_sec = 3600.0
    sched._last_micro_ts = time.monotonic()
    out = coord.tick(input_ctrl=object())
    assert out["anti_afk"]["action"] == "waiting"
    assert out["anti_afk"]["next_interval_sec"] > 0
