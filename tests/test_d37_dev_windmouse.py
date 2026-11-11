# -*- coding: utf-8 -*-
"""
D37 DEV dilimi testleri — core/windmouse_humanizer.py iyileştirmeleri.

Kilitlenen davranışlar:
- Dwell örnekleme tek yardımcıda toplandı (aynı dağılım/pencere/hata yolu).
- smooth_turn_heading / smooth_camera_pitch_yaw / smooth_camera_nudge ortak
  adım döngüsü (_execute_trajectory_steps) — abort/setter/guard yolları aynı.
- Ekran boyutu 5 sn TTL önbellekli; guard adımında ham Win32 probu yok.
- Sözleşme probu zinciri 60 sn TTL önbellekli (8 prob -> 1).
- Guard beyaz liste yoklaması VK başına 28-31 çağrı yerine tek anlık görüntü
  (fail-open yok; eşdeğerlik fuzz ile kilitli).

K9 (360°/24px cap) ve K5 (göreli fare taşıması) bu dilimin dışındadır;
test_no_single_step_exceeds_max_cap_even_on_360_turn bazda da başarısızdır.
"""

import pytest

import core.windmouse_humanizer as wm
from core.windmouse_humanizer import (
    BiometricClickGenerator,
    CameraMovementSmoother,
    CameraSmoothingConfig,
    PhysicalInputGuard,
)


@pytest.fixture(autouse=True)
def _reset_d37_caches():
    wm._SCREEN_SIZE_CACHE["value"] = None
    wm._SCREEN_SIZE_CACHE["ts"] = 0.0
    wm._CONTRACT_PROBE_CACHE["value"] = None
    wm._CONTRACT_PROBE_CACHE["ts"] = 0.0
    yield
    wm._SCREEN_SIZE_CACHE["value"] = None
    wm._SCREEN_SIZE_CACHE["ts"] = 0.0
    wm._CONTRACT_PROBE_CACHE["value"] = None
    wm._CONTRACT_PROBE_CACHE["ts"] = 0.0


class MockInputController:
    """Göreli hareket + RMB kaydı yapan sahte sürücü."""

    def __init__(self, move_result=True):
        self.moves = []
        self.mouse_events = []
        self._move_result = move_result

    def nudge_camera(self, dx, dy=0):
        self.moves.append((int(dx), int(dy)))
        return self._move_result

    def _hw_mouse_down(self, button="left"):
        self.mouse_events.append(("down", str(button)))
        return True

    def _hw_mouse_up(self, button="left"):
        self.mouse_events.append(("up", str(button)))
        return True


# ============================================================================
# A. Gaussian dwell örnekleme tek yardımcıda
# ============================================================================


def test_dwell_sampling_stays_in_window_and_near_mean():
    gen = BiometricClickGenerator()
    keys = [gen.get_key_dwell_sec() for _ in range(2000)]
    assert all(0.045 <= v <= 0.115 for v in keys)
    assert abs(sum(keys) / len(keys) - 0.076) < 0.005

    mouse = [gen.get_mouse_dwell_sec() for _ in range(2000)]
    assert all(0.055 <= v <= 0.130 for v in mouse)
    assert abs(sum(mouse) / len(mouse) - 0.085) < 0.005


def test_dwell_helper_direct_bounds_and_clamp_fallback(monkeypatch):
    # Yardımcı doğrudan: aralık dışı örnekler kırpılır, pencere korunur.
    vals = [
        BiometricClickGenerator._sample_gaussian_dwell(0.076, 0.014, 0.045, 0.115, "x")
        for _ in range(500)
    ]
    assert all(0.045 <= v <= 0.115 for v in vals)


def test_dwell_error_path_returns_mean(monkeypatch):
    monkeypatch.setattr(
        "random.gauss", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    gen = BiometricClickGenerator()
    assert gen.get_key_dwell_sec() == 0.076
    assert gen.get_mouse_dwell_sec() == 0.085


def test_dwell_methods_delegate_with_own_params(monkeypatch):
    seen = {}

    def fake(mean, sigma, lo, hi, label):
        seen[label] = (mean, sigma, lo, hi)
        return 0.077

    monkeypatch.setattr(
        BiometricClickGenerator, "_sample_gaussian_dwell", staticmethod(fake)
    )
    gen = BiometricClickGenerator()
    assert gen.get_key_dwell_sec() == 0.077
    assert gen.get_mouse_dwell_sec() == 0.077
    assert seen["get_key_dwell_sec"] == (0.076, 0.014, 0.045, 0.115)
    assert seen["get_mouse_dwell_sec"] == (0.085, 0.016, 0.055, 0.130)


# ============================================================================
# B. Ortak adım döngüsü — pitch/yaw ve nudge iptal yolları
# ============================================================================


def test_pitch_yaw_abort_callback_stops_early():
    driver = MockInputController()
    smoother = CameraMovementSmoother()
    calls = [0]

    def abort_cb():
        calls[0] += 1
        return calls[0] >= 3

    ok = smoother.smooth_camera_pitch_yaw(
        input_ctrl=driver, yaw_deg=45.0, pitch_deg=10.0,
        abort_callback=abort_cb, sleep_fn=lambda dt: None,
    )
    assert ok is False
    assert len(driver.moves) == 2  # 3. denetimde iptal, 2 adım koştu
    assert smoother.get_metrics()["aborts_count"] == 1


def test_nudge_abort_callback_stops_early():
    driver = MockInputController()
    smoother = CameraMovementSmoother()
    calls = [0]

    def abort_cb():
        calls[0] += 1
        return calls[0] >= 3

    ok = smoother.smooth_camera_nudge(
        input_ctrl=driver, dx=80, dy=-40,
        abort_callback=abort_cb, sleep_fn=lambda dt: None,
    )
    assert ok is False
    assert len(driver.moves) == 2
    assert smoother.get_metrics()["aborts_count"] == 1


def test_setter_false_fails_all_three_smooth_methods():
    for method, kwargs in (
        ("smooth_turn_heading", {"angle_diff_deg": 30.0}),
        ("smooth_camera_pitch_yaw", {"yaw_deg": 30.0, "pitch_deg": 5.0}),
        ("smooth_camera_nudge", {"dx": 60, "dy": 10}),
    ):
        driver = MockInputController(move_result=False)
        smoother = CameraMovementSmoother()
        ok = getattr(smoother, method)(
            input_ctrl=driver, sleep_fn=lambda dt: None, **kwargs
        )
        assert ok is False, method
        # Reddedilen İLK adım denendi (mock kayda alır) ama sonrası koşmadı.
        assert len(driver.moves) == 1, method
        assert smoother.get_metrics()["aborts_count"] == 1, method


def test_mid_trajectory_guard_abort_stops_pitch_yaw():
    # İmleç önce (100,100)'de sabit, ilk adımda (2000,1500)'e sıçrar:
    # pre-check geçer, yörünge-ortası denetim devasa sürüklemeyi yakalar.
    positions = iter([(100, 100), (100, 100), (2000, 1500)])

    def cursor_getter():
        try:
            return next(positions)
        except StopIteration:
            return (2000, 1500)

    guard = PhysicalInputGuard(
        cursor_getter=cursor_getter, key_state_getter=lambda vk: 0
    )
    driver = MockInputController()
    smoother = CameraMovementSmoother(guard=guard)
    ok = smoother.smooth_camera_pitch_yaw(
        input_ctrl=driver, yaw_deg=45.0, pitch_deg=0.0,
        sleep_fn=lambda dt: None,
    )
    assert ok is False
    assert driver.moves == []
    assert smoother.get_metrics()["aborts_count"] == 1


# ============================================================================
# C. Ekran boyutu TTL önbelleği
# ============================================================================


def test_screen_size_cache_probes_once_per_ttl(monkeypatch):
    calls = {"n": 0}
    orig = wm._get_screen_size

    def counting():
        calls["n"] += 1
        return orig()

    monkeypatch.setattr(wm, "_get_screen_size", counting)
    for _ in range(50):
        assert wm._clamp_to_screen(2500, 1400) == (1919, 1079)
    assert calls["n"] == 1

    wm._SCREEN_SIZE_CACHE["ts"] = 0.0  # TTL doldu
    assert wm._clamp_to_screen(10, 10) == (10, 10)
    assert calls["n"] == 2

    # Merkez listesi ham prob ile aynı: statik + dinamik (960,540 zaten listede)
    centers = PhysicalInputGuard._get_screen_centers()
    assert (960, 540) in centers
    assert (1280, 720) in centers
    assert calls["n"] == 2  # merkez sorgusu da önbellekten geldi


# ============================================================================
# D. Sözleşme probu zinciri 60 sn TTL önbellekli
# ============================================================================


def test_validate_chain_runs_probe_once_per_ttl(monkeypatch):
    calls = {"n": 0}
    orig = wm._validate_windmouse_humanizer_contract_uncached

    def counting():
        calls["n"] += 1
        return orig()

    monkeypatch.setattr(
        wm, "_validate_windmouse_humanizer_contract_uncached", counting
    )
    # audit_summary tek başına 8 prob koşuyordu.
    assert wm.get_windmouse_humanizer_audit_summary()["audit_passed"] is True
    assert calls["n"] == 1
    assert wm.get_windmouse_humanizer_readiness_probe()["ready"] is True
    assert calls["n"] == 1

    wm._CONTRACT_PROBE_CACHE["ts"] = 0.0  # TTL doldu
    assert wm.validate_windmouse_humanizer_contract() is True
    assert calls["n"] == 2


# ============================================================================
# E. Beyaz liste anlık görüntüsü — eşdeğerlik + fail-closed
# ============================================================================


def _slow_ignored_set(guard, now):
    wl = guard.self_input_whitelist
    vks = set(PhysicalInputGuard.MONITORED_VK_CODES) | set(
        PhysicalInputGuard.PHYSICAL_MOUSE_BUTTON_VKS
    )
    return {vk for vk in vks if wl.is_within_ignore_window(vk, now=now)}


def test_ignored_vks_snapshot_matches_slow_oracle():
    guard = PhysicalInputGuard(
        cursor_getter=lambda: (100, 100), key_state_getter=lambda vk: 0
    )
    wl = guard.self_input_whitelist
    # t=100.0'da kaydedilenler, now=100.1'de (100 ms < 250 ms) yoksayılır.
    wl.record_self_key("r", timestamp=100.0)
    wl.record_self_key(0x1B, timestamp=100.0)
    wl.record_self_key("t", timestamp=100.0)
    # t=99.0 kaydı now=100.1'de süresi dolmuş (1.1 sn > 250 ms).
    wl.record_self_key("f", timestamp=99.0)
    # Gelecek zaman damgası (elapsed 0 sayılır) yoksayılır.
    wl.record_self_key("g", timestamp=100.5)

    fast = guard._ignored_vks_snapshot(100.1)
    slow = _slow_ignored_set(guard, 100.1)
    assert fast is not None
    assert fast <= slow
    assert 0x52 in fast and 0x1B in fast and 0x54 in fast  # r, esc, t
    assert 0x46 not in fast and 0x46 not in slow  # f süresi doldu
    assert 0x47 in fast and 0x47 in slow  # g (gelecek damga)


def test_ignored_vks_snapshot_numeric_string_edge_is_fail_closed():
    # "82" string kaydı yavas yolun 0x52'yi yoksaymasına yol açar ama
    # _vk_timestamps'a yazılmaz; hızlı yol onu yoksaymaz (fail-CLOSED yön).
    guard = PhysicalInputGuard(
        cursor_getter=lambda: (100, 100), key_state_getter=lambda vk: 0
    )
    guard.self_input_whitelist.record_self_key("82", timestamp=100.0)
    fast = guard._ignored_vks_snapshot(100.1)
    slow = _slow_ignored_set(guard, 100.1)
    assert fast is not None
    assert fast <= slow
    assert 0x52 in slow
    assert 0x52 not in fast


def test_guard_checks_use_snapshot_not_per_vk_probe(monkeypatch):
    calls = {"n": 0}
    guard = PhysicalInputGuard(
        cursor_getter=lambda: (100, 100), key_state_getter=lambda vk: 0
    )
    orig = guard.self_input_whitelist.is_within_ignore_window

    def counting(key, window_ms=None, now=None):
        calls["n"] += 1
        return orig(key, window_ms=window_ms, now=now)

    monkeypatch.setattr(
        guard.self_input_whitelist, "is_within_ignore_window", counting
    )
    assert guard.should_abort_input() == (False, "ok")
    assert guard.check_mid_trajectory_physical_abort() == (False, "ok")
    assert calls["n"] == 0  # hızlı yol: VK başına prob yok

    # Hızlı yol kapalıysa yavaş yola düşülür (28 VK probu).
    guard._ignored_vks_snapshot = lambda now: None
    assert guard.should_abort_input() == (False, "ok")
    assert calls["n"] == len(PhysicalInputGuard.MONITORED_VK_CODES)


def test_guard_still_detects_physical_key_with_fast_path():
    pressed = {0x57}  # W basılı

    def key_getter(vk):
        return 0x8000 if vk in pressed else 0

    guard = PhysicalInputGuard(
        cursor_getter=lambda: (100, 100), key_state_getter=key_getter
    )
    ok, reason = guard.should_abort_input()
    assert ok is True
    assert "physical_keyboard_override_detected: vk=0x57" in reason

    # Aynı tuş botun kendi tuşuysa (beyaz liste + aktif) iptal yok.
    guard2 = PhysicalInputGuard(
        cursor_getter=lambda: (100, 100), key_state_getter=key_getter
    )
    guard2.register_bot_key_press(0x57, pressed=True)
    assert guard2.should_abort_input() == (False, "ok")

    guard3 = PhysicalInputGuard(
        cursor_getter=lambda: (100, 100), key_state_getter=key_getter
    )
    guard3.record_self_key(0x57)
    assert guard3.should_abort_input() == (False, "ok")
