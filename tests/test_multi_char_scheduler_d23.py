# -*- coding: utf-8 -*-
"""D23 düzeltmeleri için regresyon testleri (core/multi_char_scheduler.py).

Kapsar: F-0144, F-0145, F-1709, F-1713, F-1714, F-1717, F-1718, F-1719,
F-1720, F-1721, F-1722, F-1723, F-1727, F-1731, F-1732, F-1734, F-1736, F-0473.
"""
import json
import os
import threading

import pytest

from core.multi_char_scheduler import (
    AltCharacterManager,
    AutomationTaskScheduler,
    BuffSlot,
    CharacterProfile,
    PetAndBuffManager,
    ScheduleEntry,
    _click_normalized_safe,
    _send_key_safe,
)


class _MockCtrl:
    """press_key dönen, güven bilgisi sunmayan sahte kontrolcü."""

    def __init__(self, press_result=0.05):
        self.press_result = press_result
        self.pressed = []
        self.clicked = []

    def set_test_mode(self, enabled):
        return True

    def press_key(self, key):
        self.pressed.append(key)
        return self.press_result

    def click_normalized(self, nx, ny, button="left", double=False):
        self.clicked.append((nx, ny))
        return (int(nx * 1920), int(ny * 1080), 0.0)


class _UntrustedCtrl(_MockCtrl):
    def is_keyboard_trusted(self):
        return False


class _NoMethodCtrl:
    def set_test_mode(self, enabled):
        return True


def _mgr(tmp_path, **kw):
    return AltCharacterManager(config_path=str(tmp_path / "characters.json"), **kw)


# F-0144: kontrolcü yoksa üretimde True dönmez; simülasyonda döner.
def test_send_key_safe_no_controller_production_false():
    assert _send_key_safe(None, "esc", test_mode=False) is False
    assert _send_key_safe(None, "esc", test_mode=True) is True
    assert _send_key_safe(None, "") is False


def test_send_key_safe_no_send_method_false():
    assert _send_key_safe(_NoMethodCtrl(), "esc", test_mode=False) is False


def test_send_key_safe_press_key_zero_means_failure():
    assert _send_key_safe(_MockCtrl(press_result=0.0), "esc", test_mode=False) is False
    assert _send_key_safe(_MockCtrl(press_result=0.05), "esc", test_mode=False) is True


# F-1717: güvenilir donanım yolu kapalıysa üretimde tuş gönderilmez.
def test_send_key_safe_untrusted_hardware_path():
    ctrl = _UntrustedCtrl()
    assert _send_key_safe(ctrl, "esc", test_mode=False) is False
    assert ctrl.pressed == []


# F-0144 analog: tıklama üretimde kontrolcüsüz patlar, simülasyonda döner.
def test_click_safe_no_controller():
    with pytest.raises(RuntimeError):
        _click_normalized_safe(None, 0.5, 0.5, test_mode=False)
    assert _click_normalized_safe(None, 0.5, 0.5, test_mode=True) == (960, 540)
    with pytest.raises(RuntimeError):
        _click_normalized_safe(_NoMethodCtrl(), 0.5, 0.5, test_mode=False)


# F-1723: 9. slot sessizce 8. slotun koordinatını almaz.
def test_slot_coords_out_of_range_raises(tmp_path):
    mgr = _mgr(tmp_path)
    assert mgr.get_slot_normalized_coords(8) != mgr.get_slot_normalized_coords(7)
    with pytest.raises(ValueError):
        mgr.get_slot_normalized_coords(9)
    with pytest.raises(ValueError):
        mgr.get_slot_normalized_coords(0)


# F-1732: yinelenen slot profilleri tekilleştirilir.
def test_duplicate_slot_profiles_deduped(tmp_path):
    mgr = _mgr(
        tmp_path,
        profiles=[
            CharacterProfile(slot_index=1, name="Bir"),
            CharacterProfile(slot_index=1, name="BirYeni"),
            CharacterProfile(slot_index=2, name="Iki"),
        ],
    )
    roster = mgr.get_roster()
    assert [c.slot_index for c in roster] == [1, 2]
    assert roster[0].name == "BirYeni"


# F-1736: profille kurulum dosyayı yazmaz.
def test_profiles_init_does_not_write_file(tmp_path):
    path = tmp_path / "characters.json"
    _mgr(tmp_path, profiles=[CharacterProfile(slot_index=1, name="Bir")])
    assert not path.exists()


# F-1719: bozuk config kadroyu silmez, yedeğe taşınır, üzerine yazılmaz.
def test_corrupt_config_backed_up_not_overwritten(tmp_path):
    path = tmp_path / "characters.json"
    path.write_text("{ bozuk json", encoding="utf-8")
    mgr = _mgr(tmp_path)
    assert len(mgr.get_roster()) == 3  # varsayılan kadro bellekte
    backups = [p for p in tmp_path.iterdir() if p.name.startswith("characters.json.corrupt-")]
    assert len(backups) == 1
    assert backups[0].read_text(encoding="utf-8") == "{ bozuk json"
    assert not path.exists()  # bozuk dosya yerinde durmuyor, yedekte


def test_corrupt_schedule_backed_up_not_overwritten(tmp_path):
    path = tmp_path / "schedule.json"
    path.write_text("[1, 2", encoding="utf-8")
    sched = AutomationTaskScheduler(config_path=str(path))
    assert len(sched.get_entries()) == 3
    assert any(p.name.startswith("schedule.json.corrupt-") for p in tmp_path.iterdir())


# F-1714: kota takibi için beklenen metot artık var ve sayacı artırır.
def test_record_gather_increments_active_character(tmp_path):
    mgr = _mgr(tmp_path)
    assert mgr.record_gather(count=3) == 3
    assert mgr.get_active_character().gathered_today == 3
    assert mgr.record_gather() == 4


# F-1713: sağlık özeti gerçek sayıyı döner (sahte sıfır değil).
def test_health_summary_real_counts(tmp_path):
    mgr = _mgr(tmp_path)
    summary = mgr.get_scheduler_health_summary()
    assert summary["total_characters"] == 3
    assert summary["available_characters"] == 3
    assert summary["active_character_name"] == "AnaToplayici"


# F-1709 + F-1731: doğrulanamayan değişim kalıcı kayda geçmez, kilit altında uyunmaz.
def test_rotation_not_committed_when_steps_fail(tmp_path):
    mgr = _mgr(tmp_path)
    res = mgr.check_and_rotate_character(
        current_gathered=400, input_ctrl=None, test_mode=False
    )
    assert res["rotated"] is False
    assert mgr.active_slot_index == 1
    assert mgr.get_active_character().name == "AnaToplayici"
    assert all(not st.get("ok") for st in res["sequence_steps"] if st["step"] == 1)


def test_rotation_commits_when_steps_verified(tmp_path):
    mgr = _mgr(tmp_path)
    res = mgr.check_and_rotate_character(
        current_gathered=400, input_ctrl=_MockCtrl(), route_mgr=None, test_mode=True
    )
    assert res["rotated"] is True
    assert mgr.active_slot_index == 2
    assert all(st.get("ok") for st in res["sequence_steps"])


# F-1734: Oyuna Gir adımında Enter basılmaz (yalnızca tıklama).
def test_switch_sequence_step4_has_no_enter_key(tmp_path):
    mgr = _mgr(tmp_path)
    ctrl = _MockCtrl()
    steps = mgr.execute_character_switch_sequence(2, input_ctrl=ctrl, test_mode=True)
    assert [s["step"] for s in steps] == [1, 2, 3, 4]
    assert "enter" not in ctrl.pressed
    assert steps[3]["action"] == "enter_game_world"
    assert "key" not in steps[3]


def test_switch_sequence_unknown_slot_raises_before_input(tmp_path):
    mgr = _mgr(tmp_path)
    ctrl = _MockCtrl()
    with pytest.raises(ValueError):
        mgr.execute_character_switch_sequence(9, input_ctrl=ctrl, test_mode=True)
    assert ctrl.pressed == [] and ctrl.clicked == []


# F-1722: gece yarısını aşan pencere hafta sınırında çalışır.
def test_overnight_window_week_boundary():
    entry = ScheduleEntry(
        name="Gece", start_hhmm="22:15", end_hhmm="02:00",
        action="run_route", days_of_week=[0],  # yalnızca Pazartesi başlar
    )
    assert entry.matches_minutes(23 * 60, 0) is True      # Pazartesi 23:00
    assert entry.matches_minutes(60, 1) is True           # Salı 01:00 (Pazartesi penceresi)
    assert entry.matches_minutes(60, 0) is False          # Pazartesi 01:00 (henüz başlamadı)
    assert entry.matches_minutes(3 * 60, 1) is False      # Salı 03:00 (bitti)
    # F-1722: gece yarısı ÖNCESİ kısım yalnızca başlayan güne aittir;
    # +1 gün yalnızca 00:00 sonrası kısım içindir.
    assert entry.matches_minutes(23 * 60, 1) is False     # Salı 23:00 (Pazartesi kuralı değil)
    assert entry.matches_minutes(60, 2) is False          # Çarşamba 01:00 (Salı başlamadı)
    assert entry.matches_minutes(22 * 60 + 15, 1) is False  # Salı 22:15 (başlangıç günü değil)


# S-1202: tuş gönderimi kilit altında yapılmaz — gönderim sürerken başka
# thread kilitli okuma yapabilmelidir (düzeltme öncesi get_entries bloklanırdı).
def test_evaluate_does_not_hold_lock_during_key_send(tmp_path):
    sched = AutomationTaskScheduler(
        config_path=str(tmp_path / "schedule.json"),
        entries=[ScheduleEntry(
            name="Boss", start_hhmm="21:30", end_hhmm="22:15",
            action="town_wait_event", days_of_week=[0, 1, 2, 3, 4, 5, 6],
        )],
    )
    entered = threading.Event()
    release = threading.Event()

    class _BlockingCtrl(_MockCtrl):
        def press_key(self, key):
            entered.set()
            assert release.wait(timeout=5)
            return 0.05

    ctrl = _BlockingCtrl()
    result: dict = {}

    def _run():
        result["res"] = sched.evaluate_current_schedule("21:45", input_ctrl=ctrl, test_mode=True)

    t = threading.Thread(target=_run)
    t.start()
    assert entered.wait(timeout=5)
    entries = sched.get_entries()  # kilit serbest olmalı, beklemeden dönmeli
    assert len(entries) == 1
    release.set()
    t.join(timeout=5)
    assert not t.is_alive()
    assert result["res"]["town_recall_sent"] is True


# S-1202: buff/pet tuş gönderimi sürerken ikinci eşzamanlı çağrı kilitte
# takılmadan kendi kararını verip gönderime geçebilmelidir.
def test_buff_refresh_does_not_hold_lock_during_key_send():
    mgr = PetAndBuffManager()
    # P0-6: pet ilk tick'te kanıtsız beslenmez — taze yöneticide ilk
    # çağrı besleme yapmaz (last_pet_fed_ts == 0 "aç" sayılmaz, yalnızca
    # zamanlayıcı çapalanır). S-1202 kilit testinin tuş gönderimine
    # ulaşabilmesi için açlık kanıtı gerekir: önceki beslemeyi bir
    # aralık öncesine koy (core/multi_char_scheduler.py:2121-2132).
    import time as _time

    mgr.last_pet_fed_ts = _time.time() - mgr.pet_feed_interval_sec - 10.0
    calls: list = []
    release = threading.Event()

    class _BlockingCtrl(_MockCtrl):
        def press_key(self, key):
            calls.append(key)
            assert release.wait(timeout=5)
            return 0.05

    ctrl = _BlockingCtrl()
    results: dict = {}

    def _run(name):
        results[name] = mgr.check_and_refresh_buffs(frame=None, input_ctrl=ctrl, test_mode=True)

    t1 = threading.Thread(target=_run, args=("first",))
    t1.start()
    # İkinci çağrının da gönderime ulaştığını bekle (kilitte takılsaydı ulaşamazdı).
    end = _time.time() + 5
    t2_started = False
    while _time.time() < end:
        if len(calls) >= 1 and not t2_started:
            t2 = threading.Thread(target=_run, args=("second",))
            t2.start()
            t2_started = True
        if len(calls) >= 2:
            break
        _time.sleep(0.02)
    assert len(calls) >= 2, "ikinci çağrı tuş gönderimine ulaşamadı (kilitte takılıyor olabilir)"
    release.set()
    t1.join(timeout=5)
    t2.join(timeout=5)
    assert not t1.is_alive() and not t2.is_alive()
    assert results["first"]["pet_fed"] is True
    assert results["second"]["pet_fed"] is True
    # İki tuş basılmasına rağmen sayaç tek beslemeyi kaydeder (ikinci kayıt
    # karar anındaki zaman damgası değiştiği için atlanır).
    assert mgr.pet_feed_count == 1


# F-1721: takvim dışı saatte bot durur, çalışmaya devam etmez.
def test_off_schedule_pauses(tmp_path):
    sched = AutomationTaskScheduler(config_path=str(tmp_path / "schedule.json"))
    res = sched.evaluate_current_schedule("03:00")
    assert res["matched"] is False
    assert res["recommended_state"] == "PAUSED"
    assert res["action"] == "pause"


# F-1720: "sadece bakayım" çağrısı şehir dönüşünü kalıcı bastırmaz.
def test_peek_does_not_suppress_town_recall(tmp_path):
    sched = AutomationTaskScheduler(config_path=str(tmp_path / "schedule.json"))
    sched.evaluate_current_schedule("21:45")  # peek: input_ctrl yok
    assert sched._last_action is None
    ctrl = _MockCtrl()
    res = sched.evaluate_current_schedule("21:45", input_ctrl=ctrl, test_mode=True)
    assert res["town_recall_sent"] is True
    assert ctrl.pressed == ["h"]


# F-1718: hiç kullanılmamış buff zamanlayıcıyla "süresi doldu" sayılmaz.
def test_never_used_buff_not_expired_by_timer():
    slot = BuffSlot(name="Parşömen", hotkey="7", duration_sec=1800.0, last_used_ts=0.0)
    assert slot.is_expired_by_timer(now_ts=1_000_000.0) is False
    slot.last_used_ts = 1_000_000.0 - 1801.0
    assert slot.is_expired_by_timer(now_ts=1_000_000.0) is True


def test_first_buff_check_without_frame_presses_no_buff_keys(tmp_path):
    mgr = PetAndBuffManager()
    res = mgr.check_and_refresh_buffs(frame=None, input_ctrl=None, test_mode=True)
    assert res["refreshed_buffs"] == []
    assert "7" not in res["pressed_keys"] and "8" not in res["pressed_keys"]


# F-0473 + F-1727: hazır bayrakları gerçek durumu yansıtır.
def test_ready_flags_reflect_real_state(tmp_path):
    pet = PetAndBuffManager()
    assert pet.get_multi_char_ready_flag() is True
    pet.buff_slots = []
    assert pet.get_multi_char_ready_flag() is False
    assert pet.get_multi_char_scheduler_ready_flag()["ready"] is False
    assert pet.get_multi_char_scheduler_diagnostic_snapshot()["status"] == "degraded"

    mgr = _mgr(tmp_path)
    assert mgr.get_multi_char_scheduler_ready_flag()["ready"] is True
    assert mgr.get_multi_char_scheduler_ready_flag()["total_characters"] == 3
    assert mgr.get_multi_char_scheduler_diagnostic_snapshot()["status"] == "ok"

    sched = AutomationTaskScheduler(config_path=str(tmp_path / "schedule.json"))
    assert sched.get_multi_char_scheduler_ready_flag()["ready"] is True
    sched.clear_entries()
    assert sched.get_multi_char_scheduler_ready_flag()["ready"] is False
