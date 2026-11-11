# -*- coding: utf-8 -*-
"""
Unit tests for W33: Smart Consumable Auto-Use (hp_potion, mp_potion, buff_scroll).
Verifies that low HP triggers hp_potion, low MP triggers mp_potion, and an expired
buff_scroll triggers the corresponding scroll key.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.mana_phase_combat import ManaPhaseCombat
from core.skill_bar_reader import SmartConsumableManager


@pytest.fixture
def smart_setup(tmp_path: Path):
    """Provides an isolated SmartConsumableManager and ManaPhaseCombat instance."""
    manager = SmartConsumableManager(
        calibrated_rois_path=tmp_path / "calibrated_rois.json",
        seed_defaults=False,
    )
    manager.learn_slot_assignment("F1", "hp_potion")
    manager.learn_slot_assignment("F2", "mp_potion")
    manager.learn_slot_assignment("F5", "buff_scroll", buff_duration_sec=60.0)

    combat = ManaPhaseCombat(consumable_manager=manager)
    return manager, combat


def _fresh_dispatch_setup(t0: float, buff_duration_sec: float = 60.0):
    """Tuş gönderimini doğrulamak için cooldown'u kirlenmemiş yeni bir kurulum döner.

    `check_emergency_recovery` yalnızca karar verir (tuşa basmaz); gerçek basım
    `trigger_emergency_recovery(..., send_key_fn=...)` yolundan olur. Bu yardımcı,
    karar testlerinin cooldown durumunu bozmadan dispatch'i ayrı bir örnekte
    doğrulamayı sağlar. Dönen TemporaryDirectory test bitene dek tutulmalıdır.
    """
    tmpdir = tempfile.TemporaryDirectory()
    manager = SmartConsumableManager(
        calibrated_rois_path=Path(tmpdir.name) / "calibrated_rois.json",
        seed_defaults=False,
    )
    manager.learn_slot_assignment("F1", "hp_potion")
    manager.learn_slot_assignment("F2", "mp_potion")
    manager.learn_slot_assignment("F5", "buff_scroll", buff_duration_sec=buff_duration_sec)
    manager.mark_buff_applied("F5", duration_sec=buff_duration_sec, now=t0)
    combat = ManaPhaseCombat(consumable_manager=manager)
    return manager, combat, tmpdir


def test_hp_drop_triggers_hp_potion_key(smart_setup) -> None:
    """1. Verify that when HP drops below threshold, the hp_potion key is triggered."""
    manager, combat = smart_setup
    t0 = 1000.0
    manager.mark_buff_applied("F5", duration_sec=60.0, now=t0)

    # Healthy HP (90%) -> no hp_potion triggered
    healthy_res = combat.check_emergency_recovery(player_hp=0.90, player_mp=0.90, now=t0 + 5.0)
    assert healthy_res["hp_potion_needed"] is False
    assert healthy_res["hp_potion_triggered"] is False
    assert healthy_res["hp_potion_key"] is None

    # Low HP (55% < 65% smart threshold) -> needs F1 (hp_potion), but
    # the first low sample after a healthy one is an unconfirmed cliff
    # (v20 madde 6, `_hp_low_reading_confirmed`): the diagnosis already
    # reflects the measurement (hp_potion_needed True) while the key
    # waits for a second consecutive low sample to confirm the drop.
    low_hp_res = combat.check_emergency_recovery(player_hp=0.55, player_mp=0.90, now=t0 + 10.0)
    assert low_hp_res["hp_potion_needed"] is True
    assert low_hp_res["hp_potion_triggered"] is False
    assert low_hp_res["hp_potion_key"] is None
    assert low_hp_res["keys_to_press"] == []

    # Second consecutive low sample confirms the drop -> triggers F1.
    low_hp_confirmed = combat.check_emergency_recovery(player_hp=0.55, player_mp=0.90, now=t0 + 11.0)
    assert low_hp_confirmed["hp_potion_needed"] is True
    assert low_hp_confirmed["hp_potion_triggered"] is True
    assert low_hp_confirmed["hp_potion_key"] == "F1"
    assert "F1" in low_hp_confirmed["keys_to_press"]

    # Gerçek tuş basımı: aynı senaryo trigger yoluyla F1'i gerçekten göndermeli.
    _, combat2, _tmpdir = _fresh_dispatch_setup(t0)
    dispatched: list[str] = []
    healthy_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.90, player_mp=0.90, send_key_fn=dispatched.append, now=t0 + 5.0
    )
    assert healthy_dispatch["hp_potion_triggered"] is False
    assert dispatched == []
    # İlk düşük örnek doğrulanmamış uçurumdur: tuş gönderilmez.
    low_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.55, player_mp=0.90, send_key_fn=dispatched.append, now=t0 + 10.0
    )
    assert low_dispatch["hp_potion_triggered"] is False
    assert dispatched == []
    # İkinci düşük örnek düşüşü doğrular: F1 gerçekten gönderilir.
    low_dispatch_confirmed = combat2.trigger_emergency_recovery(
        player_hp=0.55, player_mp=0.90, send_key_fn=dispatched.append, now=t0 + 11.0
    )
    assert low_dispatch_confirmed["hp_potion_triggered"] is True
    assert dispatched == ["F1"]


def test_mp_drop_triggers_mp_potion_key(smart_setup) -> None:
    """2. Verify that when MP drops below threshold, the mp_potion key is triggered."""
    manager, combat = smart_setup
    t0 = 2000.0
    manager.mark_buff_applied("F5", duration_sec=60.0, now=t0)

    # Healthy MP (80%) -> no mp_potion triggered
    healthy_res = combat.check_emergency_recovery(player_hp=0.90, player_mp=0.80, now=t0 + 5.0)
    assert healthy_res["mp_potion_needed"] is False
    assert healthy_res["mp_potion_triggered"] is False
    assert healthy_res["mp_potion_key"] is None

    # Low MP (35% < 45% smart threshold) -> triggers F2 (mp_potion)
    low_mp_res = combat.check_emergency_recovery(player_hp=0.90, player_mp=0.35, now=t0 + 10.0)
    assert low_mp_res["mp_potion_needed"] is True
    assert low_mp_res["mp_potion_triggered"] is True
    assert low_mp_res["mp_potion_key"] == "F2"
    assert "F2" in low_mp_res["keys_to_press"]

    # Gerçek tuş basımı: aynı senaryo trigger yoluyla F2'yi gerçekten göndermeli.
    _, combat2, _tmpdir = _fresh_dispatch_setup(t0)
    dispatched: list[str] = []
    healthy_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.90, player_mp=0.80, send_key_fn=dispatched.append, now=t0 + 5.0
    )
    assert healthy_dispatch["mp_potion_triggered"] is False
    assert dispatched == []
    low_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.90, player_mp=0.35, send_key_fn=dispatched.append, now=t0 + 10.0
    )
    assert low_dispatch["mp_potion_triggered"] is True
    assert dispatched == ["F2"]


def test_buff_scroll_expired_triggers_scroll_key(smart_setup) -> None:
    """3. Verify that when a buff_scroll expires, its assigned scroll key is triggered."""
    manager, combat = smart_setup
    t0 = 3000.0

    # Apply buff scroll on F5 with 45s duration
    manager.mark_buff_applied("F5", duration_sec=45.0, now=t0)
    assert manager.is_buff_active("F5", now=t0 + 20.0) is True
    assert manager.is_buff_expired("F5", now=t0 + 20.0) is False

    # While buff is still active at t0 + 20s -> no scroll key triggered
    active_res = combat.check_emergency_recovery(player_hp=0.95, player_mp=0.95, now=t0 + 20.0)
    assert active_res["buff_expired"] is False
    assert active_res["buff_keys"] == []

    # After buff duration expires at t0 + 50s -> F5 scroll key is triggered
    assert manager.is_buff_expired("F5", now=t0 + 50.0) is True
    expired_res = combat.check_emergency_recovery(player_hp=0.95, player_mp=0.95, now=t0 + 50.0)
    assert expired_res["buff_expired"] is True
    assert "F5" in expired_res["buff_keys"]
    assert "F5" in expired_res["keys_to_press"]

    # Gerçek tuş basımı: buff aktifken hiçbir tuş, süresi dolunca F5 gönderilmeli.
    _, combat2, _tmpdir = _fresh_dispatch_setup(t0, buff_duration_sec=45.0)
    dispatched: list[str] = []
    active_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.95, player_mp=0.95, send_key_fn=dispatched.append, now=t0 + 20.0
    )
    assert active_dispatch["buff_expired"] is False
    assert dispatched == []
    expired_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.95, player_mp=0.95, send_key_fn=dispatched.append, now=t0 + 50.0
    )
    assert expired_dispatch["buff_expired"] is True
    assert dispatched == ["F5"]


def test_consumable_cooldown_prevents_duplicate_trigger_until_ready(smart_setup) -> None:
    """4. Verify potion cooldown lockouts prevent spamming until cooldown expires."""
    manager, combat = smart_setup
    t0 = 4000.0
    manager.mark_buff_applied("F5", duration_sec=120.0, now=t0)

    # First low HP & MP check triggers F1 and F2 at t0
    first = combat.check_emergency_recovery(player_hp=0.50, player_mp=0.30, now=t0)
    assert first["hp_potion_triggered"] is True
    assert first["mp_potion_triggered"] is True
    assert first["keys_to_press"] == ["F1", "F2"]

    # P0-6: karar tek başına damgalamaz — gönderim onayı olmadan
    # cooldown başlamaz, yeniden karar da tetikler.
    unconfirmed = combat.check_emergency_recovery(player_hp=0.48, player_mp=0.28, now=t0 + 4.0)
    assert unconfirmed["hp_potion_triggered"] is True
    assert unconfirmed["mp_potion_triggered"] is True
    assert unconfirmed["keys_to_press"] == ["F1", "F2"]

    # t0'daki basım onaylanınca cooldown başlar.
    combat.mark_potions_sent(["F1", "F2"], hp_key="F1", mp_key="F2", now=t0)

    # 4 seconds later (< 10s cooldown), neither potion should re-trigger
    during_cd = combat.check_emergency_recovery(player_hp=0.48, player_mp=0.28, now=t0 + 4.0)
    assert during_cd["hp_potion_needed"] is True
    assert during_cd["hp_potion_triggered"] is False
    assert during_cd["mp_potion_needed"] is True
    assert during_cd["mp_potion_triggered"] is False
    assert during_cd["keys_to_press"] == []

    # 11 seconds later (>= 10s cooldown), both potions are ready and trigger again
    after_cd = combat.check_emergency_recovery(player_hp=0.48, player_mp=0.28, now=t0 + 11.0)
    assert after_cd["hp_potion_triggered"] is True
    assert after_cd["mp_potion_triggered"] is True
    assert after_cd["keys_to_press"] == ["F1", "F2"]

    # Gerçek tuş basımı: cooldown boyunca hiçbir tuş, cooldown bitince F1+F2 gönderilmeli.
    _, combat2, _tmpdir = _fresh_dispatch_setup(t0, buff_duration_sec=120.0)
    dispatched: list[str] = []
    first_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.50, player_mp=0.30, send_key_fn=dispatched.append, now=t0
    )
    assert first_dispatch["hp_potion_triggered"] is True
    assert first_dispatch["mp_potion_triggered"] is True
    assert dispatched == ["F1", "F2"]
    combat2.trigger_emergency_recovery(
        player_hp=0.48, player_mp=0.28, send_key_fn=dispatched.append, now=t0 + 4.0
    )
    assert dispatched == ["F1", "F2"]
    after_cd_dispatch = combat2.trigger_emergency_recovery(
        player_hp=0.48, player_mp=0.28, send_key_fn=dispatched.append, now=t0 + 11.0
    )
    assert after_cd_dispatch["hp_potion_triggered"] is True
    assert after_cd_dispatch["mp_potion_triggered"] is True
    assert dispatched == ["F1", "F2", "F1", "F2"]


def test_combined_emergency_hp_mp_and_expired_buff_scroll_dispatch(smart_setup) -> None:
    """5. Verify simultaneous low HP, low MP, and expired buff_scroll dispatch all keys in priority order."""
    manager, combat = smart_setup
    t0 = 5000.0
    manager.mark_buff_applied("F5", duration_sec=30.0, now=t0)

    dispatched: list[str] = []
    # At t0 + 35s: buff is expired, HP is emergency (30% < 40%), MP is low (35% < 45%)
    rec = combat.trigger_emergency_recovery(
        player_hp=0.30,
        player_mp=0.35,
        send_key_fn=dispatched.append,
        now=t0 + 35.0,
    )

    assert rec["is_emergency"] is True
    assert rec["hp_potion_triggered"] is True
    assert rec["mp_potion_triggered"] is True
    assert rec["buff_expired"] is True
    assert rec["hp_potion_key"] == "F1"
    assert rec["mp_potion_key"] == "F2"
    assert "F5" in rec["buff_keys"]
    # Priority order: HP potion first, then MP potion, then expired buff scroll
    assert dispatched == ["F1", "F2", "F5"]


def test_visual_slot_classification_and_decide_consumable_action(tmp_path: Path) -> None:
    """6. Verify visual frame inspection classifies hp_potion, mp_potion, and buff_scroll and triggers keys."""
    manager = SmartConsumableManager(
        calibrated_rois_path=tmp_path / "calibrated_rois_visual.json",
        seed_defaults=False,
    )

    # Half-resolution frame (540x960): the consumable ROI is relative, so
    # slot geometry and every assertion below are identical to 1080p input.
    frame = np.zeros((540, 960, 3), dtype=np.uint8)
    roi = manager.current_consumable_bar_roi
    y1, y2 = int(round(roi[1] * 540)), int(round(roi[3] * 540))
    x1, x2 = int(round(roi[0] * 960)), int(round(roi[2] * 960))
    slot_w = float(x2 - x1) / 10.0

    # Slot F3 (index 2): Red HP potion
    frame[y1:y2, x1 + int(round(2 * slot_w)) : x1 + int(round(3 * slot_w))] = (20, 25, 210)
    # Slot F4 (index 3): Blue MP potion
    frame[y1:y2, x1 + int(round(3 * slot_w)) : x1 + int(round(4 * slot_w))] = (210, 110, 20)
    # Slot F6 (index 5): Beige/Gold Buff Scroll
    frame[y1:y2, x1 + int(round(5 * slot_w)) : x1 + int(round(6 * slot_w))] = (130, 190, 220)

    slots = manager.identify_consumable_slots(frame)
    assert slots["F3"]["item_type"] == "hp_potion"
    assert slots["F4"]["item_type"] == "mp_potion"
    assert slots["F6"]["item_type"] == "buff_scroll"

    t0 = 6000.0
    manager.mark_buff_applied("F6", duration_sec=20.0, now=t0)

    # At t0 + 25s (buff expired), with low HP (50%) and low MP (40%)
    decision = manager.decide_consumable_action(
        player_hp_pct=0.50,
        player_mp_pct=0.40,
        now=t0 + 25.0,
    )
    assert decision["hp_potion"] == "F3"
    assert decision["mp_potion"] == "F4"
    assert "F6" in decision["scroll_buffs"]
    assert decision["keys_to_press"] == ["F3", "F4", "F6"]
