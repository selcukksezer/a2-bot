# -*- coding: utf-8 -*-
"""
tests/test_w52_combat_target_lock_and_loot.py
=============================================
Worker W52 Verification Suite:
Validates immediate loot collection ('F') and next-target acquisition ('TAB')
upon monster death across CombatSystem and CombatRecovery, as well as
ExemplarMemory 3D monster target validation and false-target rejection.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, List, Tuple

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dataset_factory.exemplar_memory as exemplar_mod
from core.combat import CombatSystem
from core.combat_recovery import CombatRecovery
from core.mana_phase_combat import ManaPhaseCombat
from dataset_factory.exemplar_memory import ExemplarMemory


class _DummyInputDriver:
    """Lightweight deterministic input spy for recording key and mouse dispatches.

    NOTE (D89/F-2068): test_mode stays False so the code under test runs its
    production timing paths. With test_mode=True, CombatRecovery shortens the
    fast-TAB interval to 0.05s and skips the hardware input path, so the test
    would exercise test-only timing instead of production timing. Sleeps are
    stubbed by the tests themselves where speed is needed.
    """

    def __init__(self) -> None:
        self.test_mode: bool = False
        self.events: List[Tuple[str, str]] = []
        self.keys_pressed: List[str] = []

    def press_key(self, key: str, **kwargs: Any) -> bool:
        k = str(key).lower()
        self.keys_pressed.append(k)
        self.events.append(("key", k))
        return True

    def send_key(self, key: str, **kwargs: Any) -> bool:
        return self.press_key(key, **kwargs)

    def human_click(self, x: Any = None, y: Any = None, button: str = "left", **kwargs: Any) -> bool:
        self.events.append(("click", str(button).lower()))
        return True


def _make_combat_frame(has_enemy_bar: bool, player_hp_fill: float = 1.0) -> np.ndarray:
    """Create a synthetic 720p BGR game frame with player HP/MP HUD and optional red enemy target bar."""
    h, w = 720, 1280
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Bottom player HP bar (x: 0.345..0.478, y: 0.885..0.902) - Red in BGR: (0, 0, 230)
    hp_x1, hp_x2 = int(w * 0.345), int(w * 0.478)
    hp_y1, hp_y2 = int(h * 0.885), int(h * 0.902)
    fill_x2 = hp_x1 + int((hp_x2 - hp_x1) * max(0.0, min(1.0, player_hp_fill)))
    if fill_x2 > hp_x1:
        frame[hp_y1:hp_y2, hp_x1:fill_x2] = (0, 0, 230)

    # Bottom player MP bar (x: 0.519..0.636, y: 0.885..0.902) - Blue in BGR: (230, 120, 0)
    mp_x1, mp_x2 = int(w * 0.519), int(w * 0.636)
    mp_y1, mp_y2 = int(h * 0.885), int(h * 0.902)
    frame[mp_y1:mp_y2, mp_x1:mp_x2] = (230, 120, 0)

    # Top-center hostile target bar (x: 0.42..0.58, y: 0.04..0.06) - Red in BGR: (0, 0, 240)
    if has_enemy_bar:
        tx1, tx2 = int(w * 0.42), int(w * 0.58)
        ty1, ty2 = int(h * 0.045), int(h * 0.060)
        frame[ty1:ty2, tx1:tx2] = (0, 0, 240)

    return frame


def _make_distinct_crop(hue: int, sat: int, val: int, stripe_bgr: Tuple[int, int, int], size: Tuple[int, int] = (48, 48)) -> np.ndarray:
    """Create a textured BGR crop with distinct HSV color and edge features for ExemplarMemory."""
    h, w = size
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[:, :, 0] = hue
    hsv[:, :, 1] = sat
    hsv[:, :, 2] = val
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.rectangle(bgr, (6, 6), (w - 7, h - 7), stripe_bgr, 2)
    cv2.line(bgr, (0, h // 2), (w - 1, h // 2), stripe_bgr, 2)
    return bgr


def test_1_combat_system_monster_death_triggers_immediate_loot_f_and_tab(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test 1: Verify CombatSystem.fight_back_step locks onto an active monster, and as soon as
    the monster dies, immediately dispatches 'f' (loot) followed by 'tab' (next target) without delay.

    §7-A no 27 (loot tek F): ölüm dalında tek F+Tab v17 davranışıdır;
    P1-11/testupd-A çevirmesi geri alındı (F13).
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
    driver = _DummyInputDriver()
    combat_sys = CombatSystem(driver=driver, profile="melee")

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    dead_frame = _make_combat_frame(has_enemy_bar=False)
    # Tek formül (P1-12): dist = 0.96 / norm_h; melee IN_RANGE için
    # dist <= 3.8 m -> norm_h >= 0.253 -> 720p'de bh >= 182. bh=200
    # (norm_h 0.278 -> 3.5 m) menzildedir; v17'deki bh=160 (4.3 m) P1-12
    # sonrası TOO_FAR'dır — bbox v17'den farklı, davranış beklentisi v17'dir.
    in_range_bbox = (580, 260, 120, 200)

    # Step 1: Engage live monster -> locks target and attacks
    engaged = combat_sys.fight_back_step(input_ctrl=driver, frame=alive_frame, target_bbox=in_range_bbox, target_hp_pct=80.0)
    assert engaged is True
    assert combat_sys.in_combat is True
    assert combat_sys.was_fighting is True
    assert combat_sys.no_target_streak == 0

    # The attack action itself must be asserted, not just the lock state
    # (D89/F-2053): exactly one skill-rotation key is dispatched on engage.
    # Which key is drawn is random by design (random.choice over the 8-key
    # rotation pool), so the assertion pins the pool membership and the count,
    # not a specific key.
    assert len(driver.keys_pressed) == 1
    assert driver.keys_pressed[0] in {"t", "1", "2", "3", "4", "q", "e", "r"}
    assert combat_sys.total_attacks == 1
    assert sum(combat_sys.skill_rotation_counts.values()) == 1

    driver.keys_pressed.clear()
    driver.events.clear()

    # Step 2: Monster dies (2 consecutive frames without enemy bar confirms death)
    step_dead_1 = combat_sys.fight_back_step(input_ctrl=driver, frame=dead_frame)
    assert step_dead_1 is False
    assert combat_sys.no_target_streak == 1

    driver.keys_pressed.clear()
    driver.events.clear()

    step_dead_2 = combat_sys.fight_back_step(input_ctrl=driver, frame=dead_frame)
    assert step_dead_2 is True
    assert combat_sys.in_combat is False
    assert combat_sys.was_fighting is False
    assert combat_sys.no_target_streak == 0

    # Must immediately press 'f' (loot) then 'tab' (next monster) in exact order
    assert driver.keys_pressed == ["f", "tab"]


def test_2_combat_recovery_execute_combat_loop_loot_f_and_fast_tab_to_next_monster(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test 2: Verify CombatRecovery.execute_combat_loop loots ('f') and presses 'tab'
    when a monster dies, seamlessly chains into the newly targeted monster, and returns
    'CLEARED_UNVERIFIED' when the loot is not verified.

    P1-11: "CLEARED" + "loot collected" bildirimi YALNIZCA doğrulanmış
    loot'ta verilir. Bu testte okuyucu enjekte edilmediğinden doğrulama
    "no_reader" ile döner — öldürme kanıtlıdır ama loot kanıtsızdır, sonuç
    "CLEARED_UNVERIFIED" olur (core/combat_recovery.py:1905-1929).

    no27 (F-0419) — doğrulanmış loot akışı: döngü sonrası kör final F
    KALDIRILDI. Eski beklenti (12 tuş / 3×F: 2 öldürme F'si + final F),
    no27'nin yasakladığı kör ikinci F'yi kilitliyordu: döngü yalnızca
    clear dalındaki F'ten sonra break eder, yani final F her zaman
    ikinci (kör) F olurdu ve loot'u bozardı. Yeni beklenti 11 tuş /
    2×F'dir (öldürme başına TEK F + TAB); okuyucu enjekte edilmediğinden
    doğrulama "no_reader" ile anında döner, kare tüketilmez.

    Determinism (D89/F-2063, F-2068): the periodic TAB cadence in the loop is
    gated on the wall clock (time.monotonic), so with a free-running clock the
    number of extra TABs varies run to run (3..7 observed). The test therefore
    freezes the clock (monotonic -> 0.0) alongside the stubbed sleep: the only
    TABs left are the forced ones (initial lock + one per kill), and the full
    key sequence is exactly reproducible. The driver runs with test_mode=False
    (production timing paths).
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)
    driver = _DummyInputDriver()
    recovery = CombatRecovery(input_ctrl=driver)

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    dead_frame = _make_combat_frame(has_enemy_bar=False)

    # Sequence of frames:
    # 1. Monster #1 alive (1 frame)
    # 2. Monster #1 dies (1 flicker grace frame + 2 clear frames)
    # 3. After F + TAB, Monster #2 is detected alive on the post-TAB check frame + 1 loop frame
    # 4. Monster #2 dies (1 flicker grace frame + 2 clear frames + 1 post-TAB empty frame)
    frames = [
        alive_frame,  # Monster 1 locked
        dead_frame,   # Flicker grace (1/1)
        dead_frame,   # Clear streak 1
        dead_frame,   # Clear streak 2 -> triggers F (loot) + TAB (next target)
        alive_frame,  # Post-TAB check: Monster 2 detected! Continues combat loop
        alive_frame,  # Monster 2 locked in loop
        dead_frame,   # Flicker grace (1/1)
        dead_frame,   # Clear streak 1
        dead_frame,   # Clear streak 2 -> triggers F (loot) + TAB
        dead_frame,   # Post-TAB check: no more monsters -> exits loop
    ]
    frame_iter = iter(frames)

    def _capture_next() -> np.ndarray:
        return next(frame_iter, dead_frame)

    statuses: List[str] = []
    result = recovery.execute_combat_loop(
        input_ctrl=driver,
        capture=_capture_next,
        status_callback=lambda state, *_args: statuses.append(state),
        max_fight_sec=2.0,
        stop_gather_sec=0.0,
    )

    assert result == "CLEARED_UNVERIFIED"
    assert "FIGHTING" in statuses
    assert statuses[-1] == "CLEARED_UNVERIFIED"
    assert recovery.in_combat is False

    # Exact key sequence (D89/F-2063): no loose >= counts. With the frozen clock
    # the sequence is fully determined by the scripted frames:
    #   initial forced TAB, rotation t/2/3, kill #1 -> F + TAB, rotation r/q/e,
    #   kill #2 -> F + TAB. NO post-loop final F (no27/F-0419: a blind second
    #   F after a kill F corrupts the loot; the old 12-key / 3xF expectation
    #   locked in that forbidden blind F).
    assert driver.keys_pressed == [
        "tab", "t", "2", "3", "f", "tab", "r", "q", "e", "f", "tab",
    ]
    assert driver.keys_pressed.count("f") == 2  # one verified-attempt F per kill, no final F
    assert driver.keys_pressed.count("tab") == 3  # initial lock + 2 post-kill
    # No cast-bar reader is injected in this test, so verification degrades
    # to "no_reader" (single F, no blind retry) for each loot cycle.
    assert recovery.loot_verification_outcomes == ["no_reader", "no_reader"]
    assert recovery.last_loot_outcome == "no_reader"

    # Verify that after each loot 'f', the very next key action is 'tab'
    # (new target acquisition) — never a second 'f' (no27: a blind second F
    # while looting corrupts the loot).
    for idx, key in enumerate(driver.keys_pressed[:-1]):
        if key == "f":
            assert driver.keys_pressed[idx + 1] == "tab"


class _FakeCastBarReader:
    """no27: verify_gather_cast_bar vekilini senaryoya göre oynatan sahte okuyucu.

    script: her çağrıda döndürülecek (is_present, progress_pct) listesi;
    liste tükenince son değer tekrarlanır. Gerçek GameHUDReader yerine
    geçer; loot başlangıç/bitiş sinyallerini sahte vision ile verir.
    """

    def __init__(self, script: List[Tuple[bool, float]]) -> None:
        self.script = list(script)
        self.calls = 0

    def verify_gather_cast_bar(self, frame: Any) -> Tuple[bool, float]:
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]


def test_6_no27_loot_not_started_presses_single_f_without_blind_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test 6 (no27-a): F basıldı ama loot BAŞLAMADI (kanal barı 1,5 sn / 15
    yoklamada hiç görülmedi) -> kör ikinci F basılmaz, TAB'a geçilir.

    Sahte okuyucu hep (False, 0.0) döner. Donmuş saatte (monotonic -> 0.0)
    yoklama sayısı belirleyicidir: başlangıç fazı 15 yoklamada biter.
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)
    driver = _DummyInputDriver()
    reader = _FakeCastBarReader([(False, 0.0)] * 40)
    recovery = CombatRecovery(input_ctrl=driver, cast_bar_reader=reader)

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    dead_frame = _make_combat_frame(has_enemy_bar=False)

    # Kare senaryosu:
    #   [0] alive (kilit), [1..3] dead (grace + clear 1 + clear 2 -> F),
    #   [4..18] dead x15 (başlangıç yoklaması: 15 karede bar yok),
    #   [19] dead (TAB sonrası kontrol: hedef yok -> döngü biter)
    frames = [alive_frame] + [dead_frame] * 19
    frame_iter = iter(frames)

    def _capture_next() -> np.ndarray:
        return next(frame_iter, dead_frame)

    statuses: List[str] = []
    result = recovery.execute_combat_loop(
        input_ctrl=driver,
        capture=_capture_next,
        status_callback=lambda state, *_args: statuses.append(state),
        max_fight_sec=2.0,
        stop_gather_sec=0.0,
    )

    assert result == "CLEARED_UNVERIFIED"
    assert "FIGHTING" in statuses
    assert statuses[-1] == "CLEARED_UNVERIFIED"
    assert recovery.in_combat is False

    assert driver.keys_pressed == ["tab", "t", "2", "3", "f", "tab"]
    assert driver.keys_pressed.count("f") == 1  # tek F; kör tekrar yok
    assert reader.calls == 15  # başlangıç fazı: 15 yoklama, bar hiç görülmedi
    assert recovery.loot_verification_outcomes == ["not_started"]
    assert recovery.last_loot_outcome == "not_started"


def test_7_no27_loot_not_finished_next_f_waits_for_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Test 7 (no27-b): loot BAŞLADI (bar %30 görüldü) ama BİTMEDİ (3 sn /
    30 yoklamada ne %90+ oldu ne kayboldu) -> bitiş doğrulanmadan bir
    sonraki F basılmaz; TAB'dan sonra yeni canavarın öldürülmesiyle gelen
    İKİNCİ loot döngüsünde bar %95 görülürse o F doğrulanır.

    Sahte okuyucu: ilk 31 çağrı (True, 30.0) [F #1: 1 başlangıç + 30
    bitiş yoklaması], sonrası (True, 95.0) [F #2: ilk yoklamada doğrulandı].
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
    monkeypatch.setattr("time.monotonic", lambda: 0.0)
    driver = _DummyInputDriver()
    reader = _FakeCastBarReader([(True, 30.0)] * 31 + [(True, 95.0)] * 10)
    recovery = CombatRecovery(input_ctrl=driver, cast_bar_reader=reader)

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    dead_frame = _make_combat_frame(has_enemy_bar=False)

    # Kare senaryosu (donmuş saatte yoklama sayısı belirleyicidir):
    #   [0] alive (kilit), [1..3] dead (grace + clear 1 + clear 2 -> F #1),
    #   [4..34] dead x31 (F #1 yoklaması: 1 başlangıç + 30 bitiş, bar hep %30),
    #   [35] alive (TAB sonrası: canavar #2 hedeflendi),
    #   [36] alive (döngü: canavar #2 kilitli),
    #   [37..39] dead (grace + clear 1 + clear 2 -> F #2),
    #   [40] dead (F #2 yoklaması: bar %95 -> verified),
    #   [41] dead (TAB sonrası kontrol: hedef yok -> döngü biter)
    frames = (
        [alive_frame]
        + [dead_frame] * 3
        + [dead_frame] * 31
        + [alive_frame, alive_frame]
        + [dead_frame] * 3
        + [dead_frame, dead_frame]
    )
    frame_iter = iter(frames)

    def _capture_next() -> np.ndarray:
        return next(frame_iter, dead_frame)

    statuses: List[str] = []
    result = recovery.execute_combat_loop(
        input_ctrl=driver,
        capture=_capture_next,
        status_callback=lambda state, *_args: statuses.append(state),
        max_fight_sec=2.0,
        stop_gather_sec=0.0,
    )

    assert result == "CLEARED"
    assert "FIGHTING" in statuses
    assert statuses[-1] == "CLEARED"
    assert recovery.in_combat is False

    assert driver.keys_pressed == [
        "tab", "t", "2", "3", "f", "tab", "r", "q", "e", "f", "tab",
    ]
    assert driver.keys_pressed.count("f") == 2  # öldürme başına TEK F
    # F #1'den hemen sonra TAB gelir — bitmemiş loot'a kör ikinci F yok;
    # ikinci F ancak yeni canavar öldükten sonra (yeni loot döngüsü) basılır.
    assert driver.keys_pressed[5] == "tab"
    assert reader.calls == 32  # 31 (F #1) + 1 (F #2, ilk yoklamada %95)
    assert recovery.loot_verification_outcomes == ["not_finished", "verified"]
    assert recovery.last_loot_outcome == "verified"


def test_3_target_lock_flicker_grace_prevents_false_loot_before_actual_death() -> None:
    """
    Test 3: Verify CombatRecovery 1-frame targeting flicker grace sustains target lock during
    transient UI flicker without false target loss, and confirms loss on the subsequent frame.
    Entry needs 2 consecutive bar frames first (2-frame entry confirmation):
    a single-frame bar must not enter combat. Each tick passes a fresh frame
    copy, as production grabs a new frame per tick.
    """
    driver = _DummyInputDriver()
    recovery = CombatRecovery(input_ctrl=driver)
    recovery.max_flicker_grace_ticks = 1

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    flicker_frame = _make_combat_frame(has_enemy_bar=False)
    # 2. ve 4. kareler gerçekten farklı içerikli olmalı — alive_frame.copy()
    # içerik parmak iziyle "aynı kare" sayılır ve doğrulama ilerlemez.
    # (0, 0) pikseli parmak izi örneklemesindedir, hiçbir tespit ROI'sinde değildir.
    alive_frame_2 = alive_frame.copy()
    alive_frame_2[0, 0] = (1, 0, 0)
    alive_frame_4 = alive_frame.copy()
    alive_frame_4[0, 0] = (2, 0, 0)

    # Frame 1: Hostile target bar visible once -> not yet a confirmed lock
    assert recovery.detect_enemy_target_bar(alive_frame.copy(), use_flicker_grace=True) is False
    assert recovery.is_targeting_flicker_grace() is False

    # Frame 2: Bar on the 2nd consecutive frame -> lock acquired, grace inactive
    assert recovery.detect_enemy_target_bar(alive_frame_2, use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is False

    # Frame 3: 1-frame UI flicker (bar missing for 1 tick) -> lock sustained via grace
    assert recovery.detect_enemy_target_bar(flicker_frame.copy(), use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is True

    # Frame 4: Bar reappears -> grace resets cleanly without interrupting combat
    assert recovery.detect_enemy_target_bar(alive_frame_4, use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is False

    # Frame 5 & 6: Monster actually dies (2 consecutive frames without bar) -> grace expires on 2nd frame
    assert recovery.detect_enemy_target_bar(flicker_frame.copy(), use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is True
    assert recovery.detect_enemy_target_bar(flicker_frame.copy(), use_flicker_grace=True) is False
    assert recovery.is_targeting_flicker_grace() is False

    # Immediate fast_tab_cycle dispatches 'tab' and increments tab_cycle_count
    assert recovery.fast_tab_cycle(driver, force=True) is True
    assert driver.keys_pressed == ["tab"]
    assert recovery.tab_cycle_count == 1


def test_4_exemplar_memory_validates_monster_and_rejects_negative_or_resource_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test 4: Verify ManaPhaseCombat.is_valid_monster_target with ExemplarMemory positively validates
    learned 3D monsters (class_id=10) while rejecting negative exemplars and gatherable resource nodes.
    """
    ex_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=ex_path)
    monkeypatch.setattr(exemplar_mod, "_GLOBAL_EXEMPLAR_MEMORY", mem)
    monkeypatch.setattr(exemplar_mod, "get_exemplar_memory", lambda json_path=None: mem)

    # Distinct visual crops for monster (class 10), gatherable herb (class 0), and negative rock (class 14)
    monster_crop = _make_distinct_crop(hue=15, sat=210, val=190, stripe_bgr=(20, 40, 240))
    herb_crop = _make_distinct_crop(hue=95, sat=220, val=200, stripe_bgr=(240, 190, 20))
    neg_rock_crop = _make_distinct_crop(hue=0, sat=10, val=90, stripe_bgr=(60, 60, 60))

    assert mem.add_positive_exemplar(monster_crop, class_id=10, class_name="monster_3d") is True
    assert mem.add_positive_exemplar(herb_crop, class_id=0, class_name="herb_aria") is True
    assert mem.add_negative_exemplar(neg_rock_crop, class_id=14, class_name="false_rock_npc") is True

    combat = ManaPhaseCombat(exemplar_memory=mem)

    # Held-out variants (D89/F-2060): same visual class rendered at a different
    # size, NEVER added to memory. Validating with the exact stored crops is a
    # self-match (similarity 1.0 by construction) and proves nothing about
    # recognition of a newly seen monster / resource / false target.
    monster_held_out = _make_distinct_crop(hue=15, sat=210, val=190, stripe_bgr=(20, 40, 240), size=(56, 56))
    herb_held_out = _make_distinct_crop(hue=95, sat=220, val=200, stripe_bgr=(240, 190, 20), size=(56, 56))
    neg_rock_held_out = _make_distinct_crop(hue=0, sat=10, val=90, stripe_bgr=(60, 60, 60), size=(56, 56))
    assert not np.array_equal(monster_held_out, monster_crop)
    assert not np.array_equal(herb_held_out, herb_crop)
    assert not np.array_equal(neg_rock_held_out, neg_rock_crop)

    # 1. Held-out 3D monster crop -> Validated with high confidence (>= 0.85)
    valid_mob, conf_mob = combat.is_valid_monster_target(frame=None, target_crop=monster_held_out)
    assert valid_mob is True
    assert conf_mob >= 0.85

    # 2. Held-out negative exemplar (rock / friendly NPC) -> Rejected
    valid_neg, conf_neg = combat.is_valid_monster_target(frame=None, target_crop=neg_rock_held_out)
    assert valid_neg is False
    assert conf_neg <= 0.25

    # 3. Held-out gatherable resource (class_id=0 herb_aria) -> Rejected as non-monster
    valid_herb, conf_herb = combat.is_valid_monster_target(frame=None, target_crop=herb_held_out)
    assert valid_herb is False
    assert conf_herb <= 0.25


def test_5_filter_monster_targets_ranks_valid_monsters_and_filters_false_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Test 5: Verify ManaPhaseCombat.filter_monster_targets extracts candidate bounding boxes from
    the frame, filters out negative exemplars and gatherable resources, and ranks valid monsters first.
    """
    ex_path = tmp_path / "learned_exemplars_filter.json"
    mem = ExemplarMemory(json_path=ex_path)
    monkeypatch.setattr(exemplar_mod, "_GLOBAL_EXEMPLAR_MEMORY", mem)
    monkeypatch.setattr(exemplar_mod, "get_exemplar_memory", lambda json_path=None: mem)

    monster_crop = _make_distinct_crop(hue=12, sat=220, val=195, stripe_bgr=(15, 35, 245), size=(60, 60))
    ore_crop = _make_distinct_crop(hue=55, sat=215, val=185, stripe_bgr=(30, 220, 40), size=(60, 60))
    neg_crop = _make_distinct_crop(hue=120, sat=25, val=75, stripe_bgr=(50, 50, 50), size=(60, 60))
    unknown_mob_crop = _make_distinct_crop(hue=145, sat=180, val=170, stripe_bgr=(200, 50, 200), size=(60, 60))

    assert mem.add_positive_exemplar(monster_crop, class_id=10, class_name="monster_3d") is True
    assert mem.add_positive_exemplar(ore_crop, class_id=3, class_name="ore_orichalcum") is True
    assert mem.add_negative_exemplar(neg_crop, class_id=10, class_name="rejected_statue") is True

    # Place all 4 crops onto a 720x1280 scene frame
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:160, 100:160] = neg_crop
    frame[200:260, 300:360] = ore_crop
    frame[300:360, 500:560] = unknown_mob_crop
    frame[400:460, 700:760] = monster_crop

    candidates = [
        {"id": "neg_statue", "bbox_xywh": [100, 100, 60, 60], "confidence": 0.80},
        {"id": "ore_node", "bbox_xywh": [300, 200, 60, 60], "confidence": 0.82},
        {"id": "unseen_mob", "bbox_xywh": [500, 300, 60, 60], "confidence": 0.72},
        {"id": "known_monster", "bbox_xywh": [700, 400, 60, 60], "confidence": 0.78},
    ]

    combat = ManaPhaseCombat(exemplar_memory=mem)
    filtered = combat.filter_monster_targets(frame, candidates)

    kept_ids = [c["id"] for c in filtered]
    assert "neg_statue" not in kept_ids
    assert "ore_node" not in kept_ids
    assert kept_ids == ["known_monster", "unseen_mob"]
    assert filtered[0]["is_valid_monster"] is True
    assert filtered[0]["monster_confidence"] >= 0.85
    assert filtered[0]["monster_confidence"] > filtered[1]["monster_confidence"]
