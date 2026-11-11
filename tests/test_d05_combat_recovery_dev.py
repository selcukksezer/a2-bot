# -*- coding: utf-8 -*-
"""D05 dev-op tests: cv2.inRange mask refactor + HP double-read removal.

Locks the behaviour of core/combat_recovery.py after the D05 change:
- every HSV threshold mask built with cv2.inRange returns exactly the same
  pixels as the old numpy boolean-chain construction (same inclusive bounds);
- get_olive_mob_mask still returns a 0/1 mask (test_combat_recovery_c4
  asserts mask == 1);
- check_under_attack reads the HP bar only once per frame (the old code
  re-read it inside _is_bottom_hud_visible when the MP bar did not match);
- detector outputs on synthetic frames are unchanged.

No timing assertions: wall/CPU time on this VM is dominated by parallel
load, so speed is evidenced by the bench scripts in dev-bench-d05, not here.
"""

import numpy as np
import cv2
import pytest

from core.combat_recovery import CombatRecovery


# ---------------------------------------------------------------------------
# Synthetic frames (same HUD geometry as test_combat_recovery_c4.py)
# ---------------------------------------------------------------------------

def _paint_hsv(frame, x1, x2, y1, y2, h, s, v):
    w = frame.shape[1]
    hh = frame.shape[0]
    px1, px2 = int(w * x1), int(w * x2)
    py1, py2 = int(hh * y1), int(hh * y2)
    patch = np.zeros((py2 - py1, px2 - px1, 3), dtype=np.uint8)
    patch[:, :, 0] = h
    patch[:, :, 1] = s
    patch[:, :, 2] = v
    frame[py1:py2, px1:px2] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)


def make_frame(kind="combat"):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # bottom HP bar (full red) + MP bar (blue) so HUD checks pass
    # HP painted to 0.477 so the effective bar (0.3635..0.4775) reads 1.0
    _paint_hsv(frame, 0.365, 0.477, 0.885, 0.902, 0, 200, 200)
    _paint_hsv(frame, 0.520, 0.630, 0.885, 0.902, 100, 180, 180)
    # overhead HP bar (full red) — same geometry as test_combat_recovery_c4
    _paint_hsv(frame, 0.475, 0.524, 0.472, 0.480, 0, 200, 200)
    if kind == "combat":
        _paint_hsv(frame, 0.38, 0.52, 0.04, 0.055, 45, 160, 180)  # olive top bar
        _paint_hsv(frame, 0.28, 0.42, 0.25, 0.27, 42, 170, 190)  # olive combat-zone bar
    elif kind == "lowhp":
        frame[int(720 * 0.885):int(720 * 0.902), int(1280 * 0.365):int(1280 * 0.477)] = 0
        _paint_hsv(frame, 0.365, 0.395, 0.885, 0.902, 0, 200, 200)
        frame[int(720 * 0.472):int(720 * 0.480), int(1280 * 0.475):int(1280 * 0.524)] = 0
        _paint_hsv(frame, 0.475, 0.490, 0.472, 0.480, 0, 200, 200)
    elif kind == "self_overhead":
        _paint_hsv(frame, 0.47, 0.53, 0.47, 0.56, 40, 150, 160)  # olive, self zone only
    elif kind == "blank":
        frame[:] = 40
        return frame
    return frame


@pytest.fixture()
def cr():
    return CombatRecovery()


# ---------------------------------------------------------------------------
# 1. inRange masks == old boolean-chain masks (exact pixel equality)
# ---------------------------------------------------------------------------

def _bool_olive(hsv):
    return (
        (hsv[:, :, 0] >= 26) & (hsv[:, :, 0] <= 68)
        & (hsv[:, :, 1] >= 80) & (hsv[:, :, 2] >= 80)
    )


def test_olive_mask_matches_boolean_reference(cr):
    rng = np.random.default_rng(11)
    for _ in range(5):
        roi = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        got = cr.get_olive_mob_mask(roi)
        assert (got == _bool_olive(hsv).astype(np.uint8)).all()
        assert set(np.unique(got).tolist()) <= {0, 1}  # 0/1 contract


def test_red_mask_matches_boolean_reference(cr):
    rng = np.random.default_rng(12)
    for s_min, v_min, h_lo, h_hi in ((95, 85, 12, 168), (135, 105, 10, 170), (145, 110, 10, 170)):
        for _ in range(3):
            hsv = rng.integers(0, 255, (40, 120, 3), dtype=np.uint8)
            hsv[:, :, 0] = rng.integers(0, 180, (40, 120))
            ref = (
                ((hsv[:, :, 0] <= h_lo) | (hsv[:, :, 0] >= h_hi))
                & (hsv[:, :, 1] >= s_min) & (hsv[:, :, 2] >= v_min)
            )
            got = CombatRecovery._red_mask_from_hsv(hsv, s_min, v_min, h_lo=h_lo, h_hi=h_hi) > 0
            assert (ref == got).all()


def test_hostile_bar_mask_matches_boolean_reference(cr):
    rng = np.random.default_rng(13)
    for red_s, red_v, pink_s, pink_v in ((90, 80, 60, 100), (100, 90, 70, 110)):
        for _ in range(3):
            hsv = rng.integers(0, 255, (80, 300, 3), dtype=np.uint8)
            hsv[:, :, 0] = rng.integers(0, 180, (80, 300))
            ref = (
                (((hsv[:, :, 0] <= 12) | (hsv[:, :, 0] >= 168))
                 & (hsv[:, :, 1] >= red_s) & (hsv[:, :, 2] >= red_v))
                | ((hsv[:, :, 0] >= 140) & (hsv[:, :, 0] <= 175)
                   & (hsv[:, :, 1] >= pink_s) & (hsv[:, :, 2] >= pink_v))
                | _bool_olive(hsv)
            )
            got = CombatRecovery._hostile_bar_mask_from_hsv(
                hsv, red_s=red_s, red_v=red_v, pink_s=pink_s, pink_v=pink_v
            ) > 0
            assert (ref == got).all()


# ---------------------------------------------------------------------------
# 2. Detector outputs unchanged on synthetic frames
# ---------------------------------------------------------------------------

def test_detector_outputs_on_synthetic_frames(cr):
    combat = make_frame("combat")
    # Bar girişi 2 farklı karede doğrulanır (yanlış "saldırı altında"
    # düzeltmesi): ilk karede bar görülse bile henüz saldırı yok.
    # No 14 (A): boyalı bar tam 1.0 okunmuyor — ölçülen oran 0.9865
    # (ham 0.986486...; üretim check_under_attack'ta 4 haneye yuvarlar).
    assert cr.check_under_attack(combat.copy()) == (False, 0.9865)
    # 2. kare içerik olarak FARKLI olmalı: giriş doğrulaması kareleri
    # içerik parmak iziyle ayırır (145fbfb) — bayt-aynı kopya yeni kare
    # sayılmaz. Tek piksel (parmak izi ızgarasında [0,0], tüm tespit
    # bölgelerinin dışında) ikinci kareyi farklı kılar.
    combat2 = make_frame("combat")
    combat2[0, 0] = (7, 7, 7)
    assert cr.check_under_attack(combat2.copy()) == (True, 0.9865)
    assert round(cr.read_hp_ratio_or_none(combat.copy()), 4) == 0.9865
    assert cr.detect_enemy_target_bar(combat.copy(), use_flicker_grace=False) is True
    assert cr.detect_olive_mob_bar(combat.copy()) is True
    assert cr._is_bottom_hud_visible(combat.copy()) is True

    lowhp = make_frame("lowhp")
    ratio = cr.read_hp_ratio_or_none(lowhp.copy())
    assert ratio is not None and 0.2 < ratio < 0.35  # clearly damaged, exact value locked by mask tests
    # Sabit düşük HP tek başına saldırı sayılmaz — ama burada 1.0'dan
    # 0.2-0.35'e GERÇEK düşüş var (üstüne bar 1 kare grace ile sürüyor),
    # o yüzden True.
    assert cr.check_under_attack(lowhp.copy())[0] is True

    blank = make_frame("blank")
    assert cr.check_under_attack(blank.copy()) == (False, 0.0)
    assert cr.read_hp_ratio_or_none(blank.copy()) is None
    assert cr.detect_enemy_target_bar(blank.copy(), use_flicker_grace=False) is False
    assert cr._is_bottom_hud_visible(blank.copy()) is False


def test_self_overhead_olive_does_not_trigger_enemy_lock(cr):
    frame = make_frame("self_overhead")
    assert cr.detect_olive_mob_bar(frame.copy()) is False
    assert cr._raw_detect_enemy_target_bar(frame.copy()) is False


# ---------------------------------------------------------------------------
# 3. HP bar is read exactly once per check_under_attack call
# ---------------------------------------------------------------------------

def test_check_under_attack_reads_hp_once(cr, monkeypatch):
    calls = {"n": 0}
    orig = CombatRecovery._read_hp_ratio_or_none

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(CombatRecovery, "_read_hp_ratio_or_none", counting)
    # blank frame: MP bar absent, so the old code re-read HP inside
    # _is_bottom_hud_visible (2 calls); the new code must do 1.
    cr.check_under_attack(make_frame("blank").copy())
    assert calls["n"] == 1
    calls["n"] = 0
    cr.check_under_attack(make_frame("combat").copy())
    assert calls["n"] == 1


def test_hud_visible_uses_passed_hp_match_without_reread(cr, monkeypatch):
    calls = {"n": 0}
    orig = CombatRecovery._read_hp_ratio_or_none

    def counting(self, *args, **kwargs):
        calls["n"] += 1
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(CombatRecovery, "_read_hp_ratio_or_none", counting)
    blank = make_frame("blank")
    assert cr._is_bottom_hud_visible(blank.copy(), hp_matched=True) is True
    assert cr._is_bottom_hud_visible(blank.copy(), hp_matched=False) is False
    assert calls["n"] == 0
