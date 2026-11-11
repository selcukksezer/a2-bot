# -*- coding: utf-8 -*-
"""D11 iyileştirme kilitleme testleri (guards dilimi).

Kapsar:
- D11-1: harita başlık şablonu ölçek erken çıkışı (boolean-eşdeğer, daha az
  matchTemplate çağrısı).
- D11-2: karantina kaydı budama + üst sınır (sınırlı bellek, bayat hata
  sayacının "ardışık" sayılmaması).
- D11-3: davranış korumalarında duvar saati yerine monotonik saat
  (NTP sıçramasına duyarsızlık), davranış eşdeğerliğiyle kilitlenir.
"""
from unittest import mock

import cv2
import numpy as np
import pytest

from core.behavioral_guard import (
    BehavioralStealthGuard,
    NaturalCombatRotation,
    NodeQuarantineManager,
)
from core.stability_guards import MultiCriteriaMapVerifier


# ---------------------------------------------------------------------------
# D11-2: karantina budama
# ---------------------------------------------------------------------------

def test_prune_stale_entries_removes_idle_failures_keeps_active_quarantine():
    mgr = NodeQuarantineManager(default_max_failures=2, default_quarantine_minutes=30)
    for nid in (1, 2, 3):
        mgr.record_node_attempt(nid, success=False, now=1000.0)
    # node 4: 2 hata -> aktif karantina (1000 + 30*60 = 2800'e kadar)
    mgr.record_node_attempt(4, success=False, now=1000.0)
    mgr.record_node_attempt(4, success=False, now=1000.0)
    assert mgr.is_quarantined(4, now=1500.0)

    removed = mgr.prune_stale_entries(now=2000.0, max_idle_sec=600.0)
    assert removed == 3
    assert mgr.get_failure_count(1) == 0
    assert mgr.get_failure_count(4) == 2
    assert mgr.is_quarantined(4, now=2000.0)


def test_stale_single_failure_does_not_count_toward_later_quarantine():
    mgr = NodeQuarantineManager(default_max_failures=2, default_quarantine_minutes=30)
    mgr.record_node_attempt(7, success=False, now=1000.0)
    assert mgr.get_failure_count(7) == 1
    # 30 dk sessizlikten sonra bayat kayıt budanır...
    assert mgr.prune_stale_entries(now=1000.0 + 1801.0) == 1
    # ...yeni hata sıfırdan sayılır: tek hata karantina yapmaz.
    assert mgr.record_node_attempt(7, success=False, now=1000.0 + 1801.0) is False
    assert mgr.get_failure_count(7) == 1
    assert not mgr.is_quarantined(7, now=1000.0 + 1801.0)


def test_tracked_cap_enforced_and_active_quarantine_never_evicted():
    mgr = NodeQuarantineManager(
        default_max_failures=2, default_quarantine_minutes=30, max_tracked_nodes=10
    )
    mgr.record_node_attempt(99, success=False, now=1000.0)
    mgr.record_node_attempt(99, success=False, now=1000.0)
    assert mgr.is_quarantined(99, now=1001.0)
    for i in range(20):
        mgr.record_node_attempt(1000 + i, success=False, now=1000.0 + i)
    assert mgr.tracked_node_count <= 10
    # Aktif karantina cap baskısında bile korunur.
    assert mgr.is_quarantined(99, now=1025.0)
    assert 99 in mgr.get_quarantined_node_ids(now=1025.0)


def test_get_quarantined_node_ids_and_safety_status_use_public_api():
    guard = BehavioralStealthGuard()
    guard.record_node_attempt(5, success=False, now=1000.0)
    guard.record_node_attempt(5, success=False, now=1000.0)
    assert guard.is_quarantined(5, now=1500.0)
    assert guard.quarantine_manager.get_quarantined_node_ids(now=1500.0) == [5]
    status = guard.get_behavioral_safety_status()
    assert status["guard_active"] is True
    assert status["quarantined_nodes_total_tracked"] >= 1


# ---------------------------------------------------------------------------
# D11-3: monotonik saat
# ---------------------------------------------------------------------------

def test_quarantine_uses_monotonic_clock_not_wall_clock():
    mgr = NodeQuarantineManager(default_max_failures=2, default_quarantine_minutes=30)
    with mock.patch("core.behavioral_guard.time.monotonic", return_value=500.0):
        # Duvar saati uçuk bir değerde olsa bile sonuç değişmez.
        with mock.patch("core.behavioral_guard.time.time", return_value=99_999_999.0):
            mgr.record_node_attempt(11, success=False)
            mgr.record_node_attempt(11, success=False)
            assert mgr.is_quarantined(11)
            assert mgr.get_remaining_quarantine_sec(11) == pytest.approx(1800.0)
    with mock.patch("core.behavioral_guard.time.monotonic", return_value=500.0 + 1800.0):
        assert not mgr.is_quarantined(11)


def test_prompt_stuck_threshold_unchanged_under_explicit_now():
    guard = BehavioralStealthGuard()
    assert guard.check_prompt_stuck(True, current_node_id=3, now=100.0) == (False, "")
    assert guard.check_prompt_stuck(True, current_node_id=3, now=109.9) == (False, "")
    stuck, msg = guard.check_prompt_stuck(True, current_node_id=3, now=110.0)
    assert stuck is True and "10s" in msg
    # Farklı düğüm sayacı sıfırlar; görünmez prompt sıfırlar.
    assert guard.check_prompt_stuck(True, current_node_id=4, now=111.0) == (False, "")
    assert guard.check_prompt_stuck(False, current_node_id=4, now=112.0) == (False, "")


def test_combat_rotation_cooldown_unchanged_under_explicit_now():
    # now=0.0 kullanılmaz: `_last_used_at` "hiç kullanılmadı" nöbetçisi 0.0'dır,
    # tam 0.0 anında kullanılan yetenek hazır sayılır (önceden var olan uç durum).
    rot = NaturalCombatRotation(seed=7)
    assert rot.is_skill_ready("1", now=100.0)
    first = rot.get_next_combat_action(now=100.0)
    assert first["action_type"] == "skill"
    assert not rot.is_skill_ready(first["key"], now=101.0)
    assert rot.get_remaining_cooldown(first["key"], now=101.0) > 0.0
    cd = rot.cooldowns[first["key"]]
    assert rot.is_skill_ready(first["key"], now=100.0 + cd)
    # Hız sınırı: gecikme dolmadan yeni aksiyon "wait" döner.
    wait = rot.get_next_combat_action(now=100.05)
    assert wait["action_type"] == "wait"
    assert wait["delay_sec"] > 0.0


# ---------------------------------------------------------------------------
# D11-1: şablon ölçek erken çıkışı
# ---------------------------------------------------------------------------

def _frame_with_template(scale=1.0):
    v = MultiCriteriaMapVerifier()
    tmpl = v.map_header_template
    assert tmpl is not None
    frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    h, w = 720, 1280
    y0, x0 = int(h * 0.02) + 5, int(w * 0.02) + 5
    if abs(scale - 1.0) < 1e-9:
        pasted = tmpl
    else:
        pasted = cv2.resize(
            tmpl, (max(10, int(round(tmpl.shape[1] * scale))),
                   max(6, int(round(tmpl.shape[0] * scale))))
        )
    th, tw = pasted.shape[:2]
    frame[y0:y0 + th, x0:x0 + tw] = np.stack([pasted] * 3, axis=-1)
    return frame


def _old_banner_boolean(v, gray):
    """Eski davranış referansı: 9 ölçeğin tamamı, max skor, eşik 0.68."""
    h, w = gray.shape[:2]
    top_strip = gray[int(h * 0.02):int(h * 0.14), int(w * 0.02):int(w * 0.78)]
    th, tw = v.map_header_template.shape[:2]
    best = 0.0
    for s in (0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.40, 1.5625, 1.80):
        sw = max(10, int(round(tw * s)))
        sh = max(6, int(round(th * s)))
        if sh < top_strip.shape[0] and sw < top_strip.shape[1]:
            rt = v.map_header_template if abs(s - 1.0) < 1e-4 else cv2.resize(
                v.map_header_template, (sw, sh))
            res = cv2.matchTemplate(top_strip, rt, cv2.TM_CCOEFF_NORMED)
            _, mv, _, _ = cv2.minMaxLoc(res)
            best = max(best, float(mv))
    return best >= 0.68


@pytest.mark.parametrize("frame", [
    np.full((720, 1280, 3), 30, dtype=np.uint8),
    np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8),
    _frame_with_template(scale=1.0),
    _frame_with_template(scale=1.4),
    _frame_with_template(scale=0.75),
])
def test_banner_early_exit_boolean_matches_full_scan(frame):
    v = MultiCriteriaMapVerifier()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    matched, score = v.check_region_header_banner(gray, hsv)
    assert matched == _old_banner_boolean(v, gray)
    assert (score >= 0.68) == matched


def test_banner_early_exit_skips_scales_on_native_match(monkeypatch):
    v = MultiCriteriaMapVerifier()
    frame = _frame_with_template(scale=1.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    calls = {"n": 0}
    real_match = cv2.matchTemplate

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_match(*args, **kwargs)

    monkeypatch.setattr(cv2, "matchTemplate", counting)
    matched, _ = v.check_region_header_banner(gray, hsv)
    assert matched is True
    # Yerel ölçek ilk denemede eşiği geçer: 9 yerine 1 çağrı.
    assert calls["n"] == 1

    calls["n"] = 0
    rand = np.random.default_rng(1).integers(0, 255, (720, 1280, 3), dtype=np.uint8)
    gray_r = cv2.cvtColor(rand, cv2.COLOR_BGR2GRAY)
    hsv_r = cv2.cvtColor(rand, cv2.COLOR_BGR2HSV)
    matched_r, _ = v.check_region_header_banner(gray_r, hsv_r)
    n_calls_no_match = calls["n"]
    assert matched_r == _old_banner_boolean(v, gray_r)
    # Eşleşme yoksa tüm ölçekler denenir (eski davranışla aynı maliyet).
    assert n_calls_no_match == 9
