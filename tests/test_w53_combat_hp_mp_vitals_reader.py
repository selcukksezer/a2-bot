# -*- coding: utf-8 -*-
"""
Unit tests (W53) for ManaPhaseCombat.read_vitals_from_frame:
Verifies accurate reading of red HP and blue MP bar fill ratios from frames
and triggering of panic / auto-potion thresholds.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest

from core.mana_phase_combat import KitingAction, ManaPhase, ManaPhaseCombat


def _build_synthetic_vitals_frame(
    combat: ManaPhaseCombat,
    hp_fill: float,
    mp_fill: float,
    height: int = 1000,
    width: int = 1000,
) -> np.ndarray:
    """Creates a synthetic BGR frame with exact red HP and blue MP fill ratios inside calibrated ROIs."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    rois = combat.get_calibrated_roi_bounds()

    hp_rect = rois.get("player_hp_bar") or (0.345, 0.880, 0.485, 0.915)
    hy1, hy2 = int(height * hp_rect[1]), int(height * hp_rect[3])
    hx1, hx2 = int(width * hp_rect[0]), int(width * hp_rect[2])
    hp_w = max(1, hx2 - hx1)
    hp_fill_px = int(round(hp_w * max(0.0, min(1.0, hp_fill))))
    if hp_fill_px > 0:
        # Pure red in BGR -> (0, 0, 255) -> HSV (0, 255, 255)
        frame[hy1:hy2, hx1 : hx1 + hp_fill_px] = (0, 0, 255)

    mp_rect = rois.get("player_mp_bar") or (0.515, 0.880, 0.655, 0.915)
    my1, my2 = int(height * mp_rect[1]), int(height * mp_rect[3])
    mx1, mx2 = int(width * mp_rect[0]), int(width * mp_rect[2])
    mp_w = max(1, mx2 - mx1)
    mp_fill_px = int(round(mp_w * max(0.0, min(1.0, mp_fill))))
    if mp_fill_px > 0:
        # Pure blue in BGR -> (255, 0, 0) -> HSV (120, 255, 255)
        frame[my1:my2, mx1 : mx1 + mp_fill_px] = (255, 0, 0)

    return frame


def test_read_vitals_full_hp_and_mp_bars():
    """1. Full red HP (100%) and full blue MP (100%) bars read ~1.0 without triggering panic or potions."""
    combat = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30, hp_panic_threshold=0.50)
    frame = _build_synthetic_vitals_frame(combat, hp_fill=1.0, mp_fill=1.0)

    vitals = combat.read_vitals_from_frame(frame)

    assert vitals["player_hp"] == pytest.approx(1.0, abs=0.02)
    assert vitals["player_mp"] == pytest.approx(1.0, abs=0.02)
    assert vitals["hp_potion_needed"] is False
    assert vitals["mp_potion_needed"] is False
    assert vitals["is_vitality_critical"] is False


def test_read_vitals_partial_hp_and_mp_fill_ratios():
    """2. Partial red HP (70%) and blue MP (60%) bars are measured accurately from HSV masks."""
    combat = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
    frame = _build_synthetic_vitals_frame(combat, hp_fill=0.70, mp_fill=0.60)

    vitals = combat.read_vitals_from_frame(frame)

    assert vitals["player_hp"] == pytest.approx(0.70, abs=0.02)
    assert vitals["player_mp"] == pytest.approx(0.60, abs=0.02)
    assert vitals["hp_potion_needed"] is False
    assert vitals["mp_potion_needed"] is False
    assert vitals["is_vitality_critical"] is False


def test_read_vitals_low_hp_and_mp_triggers_auto_potion_thresholds():
    """3. Low red HP (40% < 50%) and low blue MP (20% < 30%) flag auto-potions and dispatch potion keys."""
    combat = ManaPhaseCombat(
        hp_potion_threshold=0.50,
        mp_potion_threshold=0.30,
        hp_potion_key="7",
        mp_potion_key="8",
    )
    frame = _build_synthetic_vitals_frame(combat, hp_fill=0.40, mp_fill=0.20)

    vitals = combat.read_vitals_from_frame(frame)
    assert vitals["player_hp"] == pytest.approx(0.40, abs=0.02)
    assert vitals["player_mp"] == pytest.approx(0.20, abs=0.02)
    assert vitals["hp_potion_needed"] is True
    assert vitals["mp_potion_needed"] is True
    assert vitals["is_vitality_critical"] is False

    decision = combat.decide_from_frame(frame, now=1000.0)
    assert "7" in decision.potion_keys
    assert "8" in decision.potion_keys
    assert decision.phase == ManaPhase.REGEN


def test_read_vitals_critical_hp_triggers_panic_defense_and_retreat():
    """4. Critical red HP (15% < 25%) sets is_vitality_critical, triggers panic defense skill & retreat jump."""
    combat = ManaPhaseCombat(
        hp_panic_threshold=0.50,
        hp_potion_threshold=0.50,
        mp_full_threshold=0.90,
        kiting_hp_threshold=0.50,
    )
    combat.set_lexicon({"1": "attack", "2": "attack", "r": "defense_heal"})
    frame = _build_synthetic_vitals_frame(combat, hp_fill=0.15, mp_fill=1.0)

    vitals = combat.read_vitals_from_frame(frame)
    assert vitals["player_hp"] == pytest.approx(0.15, abs=0.02)
    assert vitals["player_mp"] == pytest.approx(1.0, abs=0.02)
    assert vitals["hp_potion_needed"] is True
    assert vitals["is_vitality_critical"] is True

    decision = combat.decide_from_frame(frame, ready_skills=["1", "2", "r"], now=2000.0)
    assert "panic" in decision.reason
    assert decision.primary_skill == "r"
    assert decision.kiting is not None
    assert decision.kiting.should_kite is True
    assert decision.kiting.action == KitingAction.RETREAT_JUMP.value


def test_read_vitals_empty_or_invalid_frame_safe_fallback():
    """5. None or empty frames safely return full healthy vitals without false panic/potion triggers."""
    combat = ManaPhaseCombat()

    vitals_none = combat.read_vitals_from_frame(None)
    vitals_empty = combat.read_vitals_from_frame(np.array([], dtype=np.uint8))

    for vitals in (vitals_none, vitals_empty):
        assert vitals["player_hp"] == 1.0
        assert vitals["player_mp"] == 1.0
        assert vitals["hp_potion_needed"] is False
        assert vitals["mp_potion_needed"] is False
        assert vitals["is_vitality_critical"] is False
