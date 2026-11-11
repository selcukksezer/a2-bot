# -*- coding: utf-8 -*-
"""
v20 / Madde 6 (HP yanlış okuma) regresyon testleri — KOŞTURULMADAN yazıldı,
v20 koordinatörünün doğrulama turunda çalıştırılacak.

Kapsar:
  1. GameHUDReader.read_vital_bars: görünmeyen bar -> None (0.0 değil);
     yuvası görünen ama dolgusu olmayan bar -> 0.0 (gerçekten boş).
  2. read_hp_mp / get_hud_combat_and_health_summary: eski 0.0 sözleşmesi
     korunur (test_d34 pinler).
  3. ManaPhaseCombat: tek karelik uçurum (sağlıklı -> 0.00) iksir/kaçış
     tetiklemez; ardışık 2. örnek tetikler; kademeli düşüş hemen tetikler.
  4. PerceptionHub: okunamayan HP "bilinmiyor" olarak yayılır; recovery
     0.0'ı geçerli HUD ölçümünü ezemez.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
import pytest

from core.mana_phase_combat import KitingAction, ManaPhaseCombat
from core.ocr_distance import GameHUDReader
from core.perception_hub import PerceptionHub


def _red_bgr():
    hsv = np.uint8([[[0, 200, 200]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def _full_hp_frame(height: int = 1000, width: int = 1000) -> np.ndarray:
    """OCR okuyucunun bandında tam dolu kırmızı HP barı (d34 çerçevesi)."""
    f = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(
        f,
        (int(width * 0.365), int(height * 0.885)),
        (int(width * 0.478), int(height * 0.900)),
        _red_bgr(),
        -1,
    )
    return f


def _empty_but_present_hp_frame(height: int = 1000, width: int = 1000) -> np.ndarray:
    """Yuvası görünen ama dolgusu olmayan HP barı: koyu yuva + beyaz yazı."""
    f = np.zeros((height, width, 3), dtype=np.uint8)
    # Koyu yuva (V=30): tüm HP ROI'si.
    f[int(height * 0.885) : int(height * 0.900), int(width * 0.345) : int(width * 0.478)] = (30, 30, 30)
    # Beyaz can yazısı kalıntısı (parlak piksel kanıtı).
    f[int(height * 0.888) : int(height * 0.897), int(width * 0.400) : int(width * 0.440)] = (255, 255, 255)
    return f


# ------------------------------------------------------------------ 1. okuyucu
def test_v20_black_frame_hp_is_unknown_not_zero():
    reader = GameHUDReader()
    black = np.zeros((1000, 1000, 3), dtype=np.uint8)
    vitals = reader.read_vital_bars(black)
    assert vitals["hp_pct"] is None
    assert vitals["hp_valid"] is False
    assert vitals["mp_pct"] is None
    assert vitals["mp_valid"] is False


def test_v20_none_frame_hp_is_unknown_not_zero():
    reader = GameHUDReader()
    vitals = reader.read_vital_bars(None)
    assert vitals["hp_pct"] is None
    assert vitals["hp_valid"] is False


def test_v20_full_bar_still_reads_100():
    reader = GameHUDReader()
    vitals = reader.read_vital_bars(_full_hp_frame())
    assert vitals["hp_pct"] == pytest.approx(100.0, abs=0.5)
    assert vitals["hp_valid"] is True


def test_v20_present_but_empty_bar_reads_zero_valid():
    reader = GameHUDReader()
    vitals = reader.read_vital_bars(_empty_but_present_hp_frame())
    assert vitals["hp_pct"] == 0.0
    assert vitals["hp_valid"] is True


def test_v20_read_player_hp_mp_bars_propagates_unknown():
    reader = GameHUDReader()
    black = np.zeros((1000, 1000, 3), dtype=np.uint8)
    res = reader.read_player_hp_mp_bars(black)
    assert res["hp_pct"] is None
    assert res["hp_valid"] is False


# ------------------------------------------------- 2. eski 0.0 sözleşmesi korunur
def test_v20_legacy_aliases_still_return_zero_for_blank():
    reader = GameHUDReader()
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert reader.read_hp_mp(black) == {"hp_pct": 0.0, "mp_pct": 0.0}
    assert reader.read_hp_mp(None) == {"hp_pct": 0.0, "mp_pct": 0.0}
    assert reader.get_hud_combat_and_health_summary(black) == {
        "hp_pct": 0.0,
        "mp_pct": 0.0,
        "is_under_attack": False,
    }


# ------------------------------------------------------- 3. tek kare doğrulaması
def test_v20_single_frame_cliff_to_zero_does_not_trigger_potion():
    combat = ManaPhaseCombat(hp_potion_threshold=0.65, mp_potion_threshold=0.45)
    combat.check_auto_potions(player_hp=0.90, player_mp=1.0, now=1000.0)
    res = combat.check_auto_potions(player_hp=0.00, player_mp=1.0, now=1001.0)
    assert res.hp_potion_triggered is False
    assert res.keys_to_press == []


def test_v20_second_consecutive_zero_triggers_potion():
    combat = ManaPhaseCombat(hp_potion_threshold=0.65, mp_potion_threshold=0.45)
    combat.check_auto_potions(player_hp=0.90, player_mp=1.0, now=1000.0)
    combat.check_auto_potions(player_hp=0.00, player_mp=1.0, now=1001.0)
    res = combat.check_auto_potions(player_hp=0.00, player_mp=1.0, now=1002.0)
    assert res.hp_potion_triggered is True
    assert "F1" in res.keys_to_press


def test_v20_gradual_decline_triggers_immediately():
    combat = ManaPhaseCombat(hp_potion_threshold=0.65, mp_potion_threshold=0.45)
    combat.check_auto_potions(player_hp=0.40, player_mp=1.0, now=1000.0)
    res = combat.check_auto_potions(player_hp=0.00, player_mp=1.0, now=1001.0)
    assert res.hp_potion_triggered is True


def test_v20_first_sample_without_history_keeps_old_behaviour():
    combat = ManaPhaseCombat(hp_potion_threshold=0.65, mp_potion_threshold=0.45)
    res = combat.check_auto_potions(player_hp=0.00, player_mp=1.0, now=1000.0)
    assert res.hp_potion_triggered is True


def test_v20_single_frame_cliff_to_zero_does_not_kite_panic():
    combat = ManaPhaseCombat(hp_potion_threshold=0.65, mp_potion_threshold=0.45)
    combat.evaluate_kiting(player_hp=0.90, player_mp=1.0, now=1000.0)
    kite = combat.evaluate_kiting(player_hp=0.00, player_mp=1.0, now=1001.0)
    assert kite.should_kite is False
    kite2 = combat.evaluate_kiting(player_hp=0.00, player_mp=1.0, now=1002.0)
    assert kite2.should_kite is True
    assert kite2.action == KitingAction.RETREAT_JUMP.value


def test_v20_decide_end_to_end_cliff_then_confirm():
    combat = ManaPhaseCombat(hp_potion_threshold=0.65, mp_potion_threshold=0.45)
    combat.decide(player_mp=1.0, player_hp=0.90, now=1000.0)
    glitch = combat.decide(player_mp=1.0, player_hp=0.00, now=1001.0)
    assert glitch.potion_keys == []
    assert glitch.kiting is None or glitch.kiting.should_kite is False
    confirmed = combat.decide(player_mp=1.0, player_hp=0.00, now=1002.0)
    assert "F1" in confirmed.potion_keys
    assert confirmed.kiting is not None and confirmed.kiting.should_kite is True


# ------------------------------------------------------------------ 4. hub
class _FakeHUD:
    def __init__(self, hp_pct, mp_pct=100.0):
        self._hp = hp_pct
        self._mp = mp_pct

    def read_player_hp_mp_bars(self, frame):
        return {
            "hp_pct": self._hp,
            "mp_pct": self._mp,
            "has_target": False,
            "target_pct": None,
            "hp_valid": self._hp is not None,
            "mp_valid": self._mp is not None,
        }


class _RecoveryNone:
    def read_hp_ratio_or_none(self, frame):
        return None


class _RecoveryZeroLegacy:
    def get_player_hp_ratio(self, frame):
        return 0.0


def _small_frame():
    rng = np.random.default_rng(3)
    return (rng.random((120, 160, 3)) * 255).astype(np.uint8)


def test_v20_hub_unknown_hp_is_not_zero():
    hub = PerceptionHub(hud_reader=_FakeHUD(None, None), combat_recovery=None)
    cog = hub.observe(_small_frame(), force=True)
    assert cog.player_hp == 0.0
    assert cog.hp_valid is False
    assert cog.as_dict()["player_hp_valid"] is False


def test_v20_hub_recovery_none_does_not_clobber_valid_hud():
    hub = PerceptionHub(hud_reader=_FakeHUD(90.0), combat_recovery=_RecoveryNone())
    cog = hub.observe(_small_frame(), force=True)
    assert cog.player_hp == pytest.approx(0.9, abs=0.01)
    assert cog.hp_valid is True


def test_v20_hub_legacy_recovery_zero_does_not_clobber_valid_hud():
    hub = PerceptionHub(hud_reader=_FakeHUD(90.0), combat_recovery=_RecoveryZeroLegacy())
    cog = hub.observe(_small_frame(), force=True)
    assert cog.player_hp == pytest.approx(0.9, abs=0.01)
    assert cog.hp_valid is True
