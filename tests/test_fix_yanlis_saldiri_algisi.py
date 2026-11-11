# -*- coding: utf-8 -*-
"""Yanlış "saldırı altında" algısı düzeltmesi (fix/yanlis-saldiri-algisi).

Kapsar:
1. Sabit düşük HP SEVİYESİ saldırı sayılmaz (eski 0.94 eşiği her karede
   "saldırı altında" üretiyordu) — yalnızca GERÇEK düşüş sayılır.
2. Tek karelik 0.0 HP okuma hatası düşüş sayılmaz (F-0376 niyeti korunur:
   gerçek sıfıra çöküş ardışık 2 sıfıra-yakın örnekle, sağlıklı tabandan
   sayılır ve olay olarak bir kez sayılır).
3. Tek karelik düşman barı savaş girişi sayılmaz — barın ardışık
   2 FARKLI karede ham tespit edilmesi gerekir; çıkış grace'i aynen durur.
"""

import threading
import time
from unittest.mock import MagicMock

import numpy as np

from core.combat_recovery import CombatRecovery
from core.state_machine import BotWorker


def _make_frame(hp_fill: float = 1.0, enemy_bar: bool = False) -> np.ndarray:
    """720p sentetik kare: alt/üst HP barı + MP barı + opsiyonel üst hedef barı."""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Alt-orta HP barı (efektif pencere x: 0.3635..0.4775, y: 0.885..0.902)
    eff_x1, eff_x2 = 0.3635, 0.4775
    fill_x = eff_x1 + max(0.0, min(1.0, hp_fill)) * (eff_x2 - eff_x1)
    frame[int(h * 0.885):int(h * 0.902), int(w * 0.365):int(w * fill_x)] = (0, 0, 230)

    # Baş üstü HP barı (efektif pencere x: ~0.478..0.526, y: 0.470..0.482)
    ov_fill_x = 0.478 + max(0.0, min(1.0, hp_fill)) * (0.526 - 0.478)
    frame[int(h * 0.470):int(h * 0.482), int(w * 0.475):int(w * ov_fill_x)] = (0, 0, 230)

    # MP barı (x: 0.519..0.636, y: 0.885..0.902)
    frame[int(h * 0.885):int(h * 0.902), int(w * 0.519):int(w * 0.636)] = (230, 120, 0)

    # Üst-orta düşman hedef barı (x: 0.42..0.58, y: 0.045..0.060)
    if enemy_bar:
        frame[int(h * 0.045):int(h * 0.060), int(w * 0.42):int(w * 0.58)] = (0, 0, 240)
    return frame


_distinct_tick = [0]


def _make_distinct_frame(hp_fill: float = 1.0, enemy_bar: bool = False) -> np.ndarray:
    """_make_frame + 1 piksel farkı: gerçekten FARKLI içerikli ikinci kare.

    İçerik parmak izi (_frame_fingerprint: 16'şar piksel strided örnek +
    CRC32) aynı pikselli `.copy()`'yi "aynı kare" sayar; ardışık 2 karede
    doğrulama bekleyen testlerde ikinci kare bu yardımcı ile üretilir.
    (0, 0) pikseli parmak izi örneklemesindedir (16'lık ızgara) ama hiçbir
    tespit ROI'sinde değildir — yalnızca parmak izini değiştirir.
    """
    frame = _make_frame(hp_fill=hp_fill, enemy_bar=enemy_bar)
    _distinct_tick[0] += 1
    frame[0, 0] = (_distinct_tick[0] % 250 + 1, 0, 0)
    return frame


def _bare_worker() -> BotWorker:
    """_note_player_hp_drop yalnızca _route_prev_hp_ratio + tembel sayaçları okur."""
    worker = BotWorker.__new__(BotWorker)
    worker._route_prev_hp_ratio = None
    return worker


# ---------------------------------------------------------------------------
# 1. Sabit HP seviyesi saldırı değildir; gerçek düşüş saldırıdır.
# ---------------------------------------------------------------------------

def test_steady_90hp_is_not_under_attack():
    cr = CombatRecovery()
    for _ in range(3):
        under, ratio = cr.check_under_attack(_make_frame(hp_fill=0.90))
        assert under is False
        assert 0.85 < ratio < 0.95


def test_real_hp_drop_is_under_attack():
    cr = CombatRecovery()
    assert cr.check_under_attack(_make_frame(hp_fill=1.0))[0] is False
    assert cr.check_under_attack(_make_frame(hp_fill=1.0))[0] is False
    under, ratio = cr.check_under_attack(_make_frame(hp_fill=0.80))
    assert under is True
    assert 0.75 < ratio < 0.86


# ---------------------------------------------------------------------------
# 2. _note_player_hp_drop: tek karelik 0.0 okuma hatası düşüş değildir.
# ---------------------------------------------------------------------------

def test_note_hp_drop_single_zero_read_is_not_a_drop():
    worker = _bare_worker()
    worker._route_prev_hp_ratio = 0.9
    # Tek karelik 0.0 (okuma hatası) -> düşüş yok
    assert worker._note_player_hp_drop(0.0) is False
    # Ardışık 2. sıfıra-yakın örnek -> gerçek çöküş, bir kez sayılır
    assert worker._note_player_hp_drop(0.0) is True
    # 3. ardışık 0.0 -> olay zaten sayıldı, her karede True dönmez
    # (yoksa okunamayan HP, FIGHTING'i kalıcı kilitler)
    assert worker._note_player_hp_drop(0.0) is False


def test_note_hp_drop_collapse_from_low_hp_does_not_count():
    worker = _bare_worker()
    worker._route_prev_hp_ratio = 0.4
    # F-0376 eşiği korunur: çöküş ancak sağlıklı (>= 0.5) tabandan sayılır.
    assert worker._note_player_hp_drop(0.0) is False
    assert worker._note_player_hp_drop(0.0) is False


def test_note_hp_drop_single_glitch_then_recovery_is_not_a_drop():
    worker = _bare_worker()
    worker._route_prev_hp_ratio = 0.9
    assert worker._note_player_hp_drop(0.0) is False
    assert worker._note_player_hp_drop(0.85) is False


def test_note_hp_drop_real_drop_counts():
    worker = _bare_worker()
    worker._route_prev_hp_ratio = 0.9
    assert worker._note_player_hp_drop(0.85) is True


def test_note_hp_drop_below_threshold_is_not_a_drop():
    worker = _bare_worker()
    worker._route_prev_hp_ratio = 0.9
    assert worker._note_player_hp_drop(0.89) is False


def test_note_hp_drop_none_is_not_a_drop_and_preserves_baseline():
    worker = _bare_worker()
    worker._route_prev_hp_ratio = 0.9
    assert worker._note_player_hp_drop(None) is False
    # None örnek tabanı silmez: sonraki geçerli örnek hâlâ 0.9 ile kıyaslanır.
    assert worker._note_player_hp_drop(0.85) is True


def test_note_hp_drop_without_baseline_never_counts_zero():
    worker = _bare_worker()
    assert worker._note_player_hp_drop(0.0) is False
    assert worker._note_player_hp_drop(0.0) is False


# ---------------------------------------------------------------------------
# 3. Düşman barı: tek kare giriş değil; ardışık 2 farklı kare giriştir.
#    Çıkış grace'i aynen durur.
# ---------------------------------------------------------------------------

def test_single_bar_frame_is_not_combat_entry():
    cr = CombatRecovery()
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=True)) is False
    cr2 = CombatRecovery()
    under, ratio = cr2.check_under_attack(_make_frame(enemy_bar=True))
    assert under is False
    assert ratio == 1.0


def test_two_consecutive_bar_frames_enter_combat():
    cr = CombatRecovery()
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=True)) is False
    # İkinci kare gerçekten farklı içerikli olmalı — aynı karenin .copy()'si
    # içerik parmak iziyle "aynı kare" sayılır ve doğrulama ilerlemez.
    assert cr._detect_enemy_target_bar(_make_distinct_frame(enemy_bar=True)) is True
    cr2 = CombatRecovery()
    assert cr2.check_under_attack(_make_frame(enemy_bar=True)) == (False, 1.0)
    assert cr2.check_under_attack(_make_distinct_frame(enemy_bar=True)) == (True, 1.0)


def test_same_frame_twice_does_not_confirm_entry():
    # Rota işleyici aynı tick içinde check_under_attack + has_bar ile aynı
    # frame_bgr'yi iki kez okur — ikinci okuma yeni kare sayılmaz.
    cr = CombatRecovery()
    frame = _make_frame(enemy_bar=True)
    assert cr._detect_enemy_target_bar(frame) is False
    assert cr._detect_enemy_target_bar(frame) is False
    assert cr._detect_enemy_target_bar(frame.copy()) is False
    # Gerçekten farklı içerikli ikinci kare doğrulamayı tamamlar.
    assert cr._detect_enemy_target_bar(_make_distinct_frame(enemy_bar=True)) is True


def test_unconfirmed_single_bar_gets_no_grace():
    # Doğrulanmamış tek karelik tespit, sonraki boş karede grace ile uzatılmaz.
    cr = CombatRecovery()
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=True)) is False
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=False)) is False
    assert cr.is_targeting_flicker_grace() is False


def test_confirmed_lock_survives_one_flicker_frame_and_dies_on_second():
    cr = CombatRecovery()
    cr.max_flicker_grace_ticks = 1
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=True)) is False
    # İkinci kare gerçekten farklı içerikli olmalı (aynı karenin .copy()'si
    # içerik parmak iziyle "aynı kare" sayılır).
    assert cr._detect_enemy_target_bar(_make_distinct_frame(enemy_bar=True)) is True
    # 1 karelik kayboluş: grace ile sürer
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=False)) is True
    assert cr.is_targeting_flicker_grace() is True
    # Bar geri gelir: grace kapanır, kilit sürer (yeniden doğrulama yok)
    assert cr._detect_enemy_target_bar(_make_distinct_frame(enemy_bar=True)) is True
    assert cr.is_targeting_flicker_grace() is False
    # 2 ardışık kayboluş: 1.'de grace, 2.'de kayıp
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=False)) is True
    assert cr.is_targeting_flicker_grace() is True
    assert cr._detect_enemy_target_bar(_make_frame(enemy_bar=False)) is False
    assert cr.is_targeting_flicker_grace() is False


# ---------------------------------------------------------------------------
# 4. death_ready: sabit %90 HP + 2,5 sn hedef yokluğu -> canavar ölümü sayılır.
#    (state_machine.py `_step_combat_mode` içindeki death_ready dalı:
#    was_fighting + hasar yok + hedef 2,5 sn yok -> record_monster_kill.)
#    GERÇEK CombatRecovery kullanılır: sabit %90 HP, yeni sözleşmede
#    "saldırı altında" sayılmaz; eski 0.94 seviye eşiğiyle bu dal hiç
#    çalışmazdı (still_taking_damage hep True kalırdı).
# ---------------------------------------------------------------------------

def test_death_ready_steady_90hp_no_target_2_5s_records_combat_death(monkeypatch):
    from PyQt6.QtCore import QThread

    calls = []

    class _FakeStats:
        def record_monster_kill(self, mob_key=None, **kwargs):
            calls.append(mob_key)

    monkeypatch.setattr(
        "core.lifetime_stats.get_lifetime_stats", lambda: _FakeStats()
    )

    worker = BotWorker.__new__(BotWorker)
    QThread.__init__(worker)
    worker._lock = threading.RLock()
    worker._paused = False
    worker._mode = "combat"
    worker._in_step_combat_mode = False
    worker._combat_was_targeting = True
    worker._combat_no_target_streak = 0
    worker._combat_last_target_ts = time.time() - 3.0  # hedef 3 sn yok (>= 2.5)
    worker._route_prev_hp_ratio = 0.90
    worker.config = {"test_mode": True}
    worker._get_cfg = lambda k, d=None: worker.config.get(k, d)
    # GERÇEK okuyucu: sabit %90 HP saldırı değildir.
    worker.combat_recovery = CombatRecovery()
    frame = _make_frame(hp_fill=0.90)  # hedef barı yok, HP sabit %90
    worker._grab_frame = lambda grayscale=False: frame
    worker._combat_has_hostile_lock = lambda f: False
    worker._sync_combat_right_button = MagicMock()
    worker._release_combat_left_button = MagicMock()
    worker._emit_log = MagicMock()
    worker._emit_hud_snapshot = MagicMock()

    # Önkoşul: yeni sözleşmede sabit %90 HP saldırı sayılmaz.
    under, ratio = worker.combat_recovery.check_under_attack(frame.copy())
    assert under is False
    assert 0.85 < ratio < 0.95

    worker._step_combat_mode()

    assert calls == ["combat_death"]
    assert worker._combat_was_targeting is False
    assert worker._combat_no_target_streak == 0
