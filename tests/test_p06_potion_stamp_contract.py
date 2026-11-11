# -*- coding: utf-8 -*-
"""P0-6 (v19): iksir damgası sözleşmesi — damga yalnızca gönderim başarısından sonra.

Sözleşme (v18 P0-6 / K-13'ün `mana_phase_combat`'a taşınması; §7-A ile aynı
kural — yeni oyun davranışı kararı gerektirmez):
- `check_auto_potions` / `check_emergency_recovery` SAF KARARDIR: cooldown
  durumunu okur, asla damga yazmaz. Karar tek başına 10 sn cooldown'u
  tüketmez.
- Damga yalnızca gönderim onayı sonrası `mark_potions_sent` ile yazılır
  (`trigger_potions` / `trigger_emergency_recovery` / `apply_to_bus`).
- Onay (`_send_confirmed`): istisna / `False` / `0` / `{"ok": False}` =
  gönderilmedi; `None` dahil rapor vermeyen gönderici = gönderildi
  (v18 burst kuralı). Başarısız gönderim damgalanmaz, sonraki tick
  yeniden dener — canlı `state_machine._check_and_trigger_auto_potions`
  B-02 kuralıyla aynı.
- Başarı yolunda davranış değişmez: eşikler (HP 65 / MP 45 / acil 40),
  tuşlar, 10 sn cooldown aynen kalır; damga aynı tick'te, gönderimden
  hemen sonra yazılır.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.mana_phase_combat import ManaPhaseCombat, _send_confirmed
from core.skill_bar_reader import SmartConsumableManager


def _manager_with_potions(tmp_path: Path, buff_duration_sec: float = 60.0):
    manager = SmartConsumableManager(
        calibrated_rois_path=tmp_path / "calibrated_rois.json",
        seed_defaults=False,
    )
    manager.learn_slot_assignment("F1", "hp_potion")
    manager.learn_slot_assignment("F2", "mp_potion")
    manager.learn_slot_assignment("F5", "buff_scroll", buff_duration_sec=buff_duration_sec)
    return manager


def test_send_confirmed_mapping() -> None:
    """Onay kuralı: açık başarısızlık raporu dışında gönderildi sayılır."""
    assert _send_confirmed(None) is True
    assert _send_confirmed(True) is True
    assert _send_confirmed(2.5) is True
    assert _send_confirmed({"ok": True}) is True
    assert _send_confirmed({"ok": 1}) is True
    assert _send_confirmed(False) is False
    assert _send_confirmed(0) is False
    assert _send_confirmed(0.0) is False
    assert _send_confirmed({"ok": False}) is False
    assert _send_confirmed({}) is False
    assert _send_confirmed({"ok": None}) is False


def test_decision_alone_never_stamps_cooldown(tmp_path: Path) -> None:
    """Saf karar: aynı düşük vitallerle art arda karar hep tetikler."""
    combat = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
    for t in (100.0, 101.0, 105.0):
        res = combat.check_auto_potions(player_hp=0.30, player_mp=1.0, now=t)
        assert res.hp_potion_triggered is True
        assert res.keys_to_press == ["F1"]
    assert combat._last_hp_potion_ts == 0.0

    manager = _manager_with_potions(tmp_path)
    manager.mark_buff_applied("F5", duration_sec=60.0, now=1000.0)
    combat2 = ManaPhaseCombat(consumable_manager=manager)
    for t in (1000.0, 1001.0, 1004.0):
        rec = combat2.check_emergency_recovery(player_hp=0.50, player_mp=0.30, now=t)
        assert rec["hp_potion_triggered"] is True
        assert rec["mp_potion_triggered"] is True
        assert rec["keys_to_press"] == ["F1", "F2"]
    assert combat2._last_hp_potion_ts == 0.0
    assert combat2._last_mp_potion_ts == 0.0
    assert manager.is_slot_ready("F1", now=1004.0) is True
    assert manager.is_slot_ready("F2", now=1004.0) is True


def test_failed_send_does_not_stamp_trigger_potions() -> None:
    """İstisna / {"ok": False} / 0.0 dönüşü damgalanmaz — sonraki çağrı yeniden dener."""
    cases = []

    def _raising(_key):
        raise RuntimeError("driver down")

    cases.append(("raising", _raising))

    class _OkFalseBus:
        def cast_skill(self, key):
            return {"ok": False, "slot": key}

    cases.append(("ok_false_bus", None))

    class _ZeroHoldBus:
        def press_key(self, key):
            return 0.0

    cases.append(("zero_hold_bus", None))

    for name, send_fn in cases:
        brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
        if name == "ok_false_bus":
            res = brain.trigger_potions(
                player_hp=0.35, player_mp=0.20, action_bus=_OkFalseBus(), now=100.0
            )
        elif name == "zero_hold_bus":
            res = brain.trigger_potions(
                player_hp=0.35, player_mp=0.20, action_bus=_ZeroHoldBus(), now=100.0
            )
        else:
            res = brain.trigger_potions(
                player_hp=0.35, player_mp=0.20, send_key_fn=send_fn, now=100.0
            )
        assert res.keys_to_press == ["F1", "F2"], name
        assert brain._last_hp_potion_ts == 0.0, name
        assert brain._last_mp_potion_ts == 0.0, name
        # Aynı tick'te yeniden deneme: damga yok, karar yine tetikler.
        res2 = brain.trigger_potions(
            player_hp=0.35, player_mp=0.20, send_key_fn=(lambda k: None), now=101.0
        )
        assert res2.hp_potion_triggered is True, name
        assert res2.mp_potion_triggered is True, name


def test_bare_bus_does_not_stamp() -> None:
    """Gönderim metodu olmayan bus: karar döner ama damga yazılmaz."""

    class _BareBus:
        pass

    brain = ManaPhaseCombat(hp_potion_threshold=0.50, mp_potion_threshold=0.30)
    res = brain.trigger_potions(player_hp=0.35, player_mp=0.20, action_bus=_BareBus(), now=100.0)
    assert res.keys_to_press == ["F1", "F2"]
    assert brain._last_hp_potion_ts == 0.0
    res2 = brain.trigger_potions(player_hp=0.35, player_mp=0.20, action_bus=_BareBus(), now=101.0)
    assert res2.hp_potion_triggered is True


def test_confirmed_send_stamps_and_cooldown_applies(tmp_path: Path) -> None:
    """Onaylı gönderim (None dönen callback) damgalar; cooldown ondan sonra işler."""
    manager = _manager_with_potions(tmp_path)
    combat = ManaPhaseCombat(consumable_manager=manager)
    t0 = 1000.0
    sent: list[str] = []
    res = combat.trigger_potions(
        player_hp=0.35, player_mp=0.20, send_key_fn=sent.append, now=t0
    )
    assert res.keys_to_press == ["F1", "F2"]
    assert sent == ["F1", "F2"]
    assert combat._last_hp_potion_ts == t0
    assert combat._last_mp_potion_ts == t0
    assert manager.is_slot_ready("F1", now=t0 + 4.0) is False
    assert manager.is_slot_ready("F2", now=t0 + 4.0) is False

    res2 = combat.check_auto_potions(player_hp=0.35, player_mp=0.20, now=t0 + 4.0)
    assert res2.hp_potion_triggered is False
    assert res2.mp_potion_triggered is False
    res3 = combat.check_auto_potions(player_hp=0.35, player_mp=0.20, now=t0 + 11.0)
    assert res3.hp_potion_triggered is True
    assert res3.mp_potion_triggered is True


def test_trigger_emergency_recovery_failed_send_retries(tmp_path: Path) -> None:
    """trigger_emergency_recovery: başarısız gönderim damgalanmaz, tekrar dener."""
    manager = _manager_with_potions(tmp_path)
    combat = ManaPhaseCombat(consumable_manager=manager)
    t0 = 2000.0
    manager.mark_buff_applied("F5", duration_sec=60.0, now=t0)

    def _raising(_key):
        raise RuntimeError("driver down")

    dispatched: list[str] = []
    rec = combat.trigger_emergency_recovery(
        player_hp=0.50, player_mp=0.30, send_key_fn=_raising, now=t0
    )
    assert rec["hp_potion_triggered"] is True
    assert rec["mp_potion_triggered"] is True
    assert dispatched == []
    assert combat._last_hp_potion_ts == 0.0
    rec2 = combat.trigger_emergency_recovery(
        player_hp=0.50, player_mp=0.30, send_key_fn=dispatched.append, now=t0 + 1.0
    )
    assert rec2["hp_potion_triggered"] is True
    assert dispatched == ["F1", "F2"]
    # Onaylı gönderim sonrası cooldown başlar.
    rec3 = combat.trigger_emergency_recovery(
        player_hp=0.48, player_mp=0.28, send_key_fn=dispatched.append, now=t0 + 5.0
    )
    assert rec3["hp_potion_triggered"] is False
    assert dispatched == ["F1", "F2"]


def test_apply_to_bus_ok_false_does_not_stamp() -> None:
    """apply_to_bus: {"ok": False} bus — özet de boş, damga da yok, tekrar denenir."""
    combat = ManaPhaseCombat()
    t0 = time.time()

    class _OkFalseBus:
        def __init__(self):
            self.casts: list[str] = []

        def cast_skill(self, key):
            self.casts.append(key)
            return {"ok": False, "slot": key}

        def hold_basic_attack(self, state):
            pass

    decision = combat.decide_combat_action(
        player_hp=0.30, player_mp=0.40,
        ready_skills=["1", "2"],
        fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
        now=t0,
    )
    assert decision.potion_keys == ["F1", "F2"]
    bus = _OkFalseBus()
    summary = combat.apply_to_bus(decision, action_bus=bus)
    # F-0122: apply_to_bus kararın TAMAMINI basar — iksir (F1/F2) + buff
    # (F5/F7: varsayılan tohumda süresi dolmuş parşömen) + skill ("1"/"2").
    # ok=False yalnızca damgayı ve özeti engeller, gönderim kapsamını
    # değil; casts == ["F1", "F2"] beklemek F-0122 sözleşmesiyle çelişir
    # (bkz. tests/test_mana_phase_combat_apply_to_bus_f0122.py).
    assert bus.casts == (
        list(decision.potion_keys) + list(decision.buff_keys) + list(decision.skill_keys)
    )
    assert bus.casts[:2] == ["F1", "F2"]
    assert summary["potion_keys"] == []
    assert combat._last_hp_potion_ts == 0.0
    assert combat._last_mp_potion_ts == 0.0
    decision2 = combat.decide_combat_action(
        player_hp=0.30, player_mp=0.40,
        ready_skills=["1", "2"],
        fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
        now=t0 + 4.0,
    )
    assert decision2.potion_keys == ["F1", "F2"]


def test_apply_to_bus_ok_true_stamps() -> None:
    """apply_to_bus: {"ok": True} bus — özet dolu, damga yazılı, cooldown işler.

    Cooldown bitince yeniden tetikler; bir sonraki hazır slot LRU
    sırasıyla F3/F4 olur (varsayılan tohumda F3=hp, F4=mp).
    """
    combat = ManaPhaseCombat()
    t0 = time.time()

    class _OkTrueBus:
        def __init__(self):
            self.casts: list[str] = []

        def cast_skill(self, key):
            self.casts.append(key)
            return {"ok": True, "slot": key}

        def hold_basic_attack(self, state):
            pass

    decision = combat.decide_combat_action(
        player_hp=0.30, player_mp=0.40,
        ready_skills=["1", "2"],
        fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
        now=t0,
    )
    assert decision.potion_keys == ["F1", "F2"]
    bus = _OkTrueBus()
    summary = combat.apply_to_bus(decision, action_bus=bus)
    assert summary["potion_keys"] == ["F1", "F2"]
    assert combat._last_hp_potion_ts > 0.0
    decision2 = combat.decide_combat_action(
        player_hp=0.30, player_mp=0.40,
        ready_skills=["1", "2"],
        fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
        now=t0 + 4.0,
    )
    assert decision2.potion_keys == []
    decision3 = combat.decide_combat_action(
        player_hp=0.30, player_mp=0.40,
        ready_skills=["1", "2"],
        fallback_keys=["1", "2", "3", "4", "q", "e", "r"],
        now=t0 + 11.0,
    )
    # Cooldown bitti (11 sn > 10 sn) → yeniden tetikler; ama tuş F1/F2
    # değildir: varsayılan tohumda F3=hp_potion, F4=mp_potion da vardır ve
    # get_best_* (skill_bar_reader.py) hazır slotlar arasından EN AZ
    # kullanılanı seçer — hiç basılmamış F3/F4, t0'da damgalanmış F1/F2'yi
    # geçer. Bu rotasyon canlı B-02 yolunun da davranışıdır
    # (state_machine.py get_best_hp/mp_potion çağrıları); P0-6 sözleşmesi
    # yalnızca damga zamanını kapsar, slot seçimini değiştirmez.
    assert decision3.potion_keys == ["F3", "F4"]


def test_apply_to_bus_ok_false_does_not_mark_buff(tmp_path: Path) -> None:
    """Buff tarafı da aynı kural: ok=False bus'ta buff uygulanmış sayılmaz."""
    tmpdir = tempfile.TemporaryDirectory()
    try:
        manager = _manager_with_potions(Path(tmpdir.name), buff_duration_sec=30.0)
        t0 = time.time()
        manager.mark_buff_applied("F5", duration_sec=30.0, now=t0 - 60.0)
        assert manager.is_buff_expired("F5", now=t0) is True
        combat = ManaPhaseCombat(consumable_manager=manager)

        class _OkFalseBus:
            def cast_skill(self, key):
                return {"ok": False, "slot": key}

            def hold_basic_attack(self, state):
                pass

        decision = combat.decide_combat_action(
            player_hp=0.95, player_mp=0.95, now=t0
        )
        assert "F5" in decision.buff_keys
        summary = combat.apply_to_bus(decision, action_bus=_OkFalseBus())
        assert summary["buff_keys"] == []
        assert manager.is_buff_expired("F5", now=t0) is True
    finally:
        tmpdir.cleanup()
