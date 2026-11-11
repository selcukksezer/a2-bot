"""F-0232 (kismi dogrulama): get_ready_combat_skills uretimde baglidir.

fight_back_step (state_machine tarafindan her savas adiminda cagrilir)
skill dalinda get_ready_combat_skills() sonucunu kullanir: hazir skill
yoksa COMBAT_SKILL_KEYS sirasi, hazir skill varsa yalnizca hazir olanlar
doner. Bu test o baglantiyi kilitler.

wait_for_revive / check_daily_limit_and_anti_afk icin ayni baglanti
R03 icinde kurulamaz: wait_for_revive bloklayan bir API'dir, state_machine
olum beklemesini tick tabanli recover_from_death() yoklamasiyla kendisi
yurutur (core/state_machine.py); check_daily_limit'in cagirani ise
toplama ana dongusudur — her ikisi de R03 disi dosyalardadir.
"""

import numpy as np
import pytest

from core.combat_recovery import CombatRecovery


class _Ctrl:
    def __init__(self):
        self.pressed = []

    def press_key(self, key, min_hold=0.045, max_hold=0.095):
        self.pressed.append(str(key).lower())
        return True


def test_fight_back_step_uses_ready_combat_skills(monkeypatch):
    cr = CombatRecovery()
    ctrl = _Ctrl()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Hedef var say, skill dalina gir (step_idx tek -> skill, cift -> R/T).
    monkeypatch.setattr(cr, "_detect_enemy_target_bar", lambda *a, **k: True)
    monkeypatch.setattr(cr, "get_ready_combat_skills", lambda *a, **k: ["3"])
    cr._fight_step_counter = 1
    cr._fight_skill_idx = 0
    assert cr.fight_back_step(input_ctrl=ctrl, frame=frame) is True
    # Hazir skill ["3"] oldugu icin ilk basilan skill "3" olmalidir;
    # hazir liste yok sayilsaydi sira "1" ile baslardi.
    assert ctrl.pressed and ctrl.pressed[0] == "3"


def test_fight_back_step_falls_back_to_full_rotation_when_no_ready_skill(monkeypatch):
    cr = CombatRecovery()
    ctrl = _Ctrl()
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    monkeypatch.setattr(cr, "_detect_enemy_target_bar", lambda *a, **k: True)
    monkeypatch.setattr(cr, "get_ready_combat_skills", lambda *a, **k: [])
    cr._fight_step_counter = 1
    cr._fight_skill_idx = 0
    assert cr.fight_back_step(input_ctrl=ctrl, frame=frame) is True
    assert ctrl.pressed and ctrl.pressed[0] == "1"
