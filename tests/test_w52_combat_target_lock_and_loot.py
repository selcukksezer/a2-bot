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
    """Lightweight deterministic input spy for recording key and mouse dispatches."""

    def __init__(self) -> None:
        self.test_mode: bool = True
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
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
    driver = _DummyInputDriver()
    combat_sys = CombatSystem(driver=driver, profile="melee")

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    dead_frame = _make_combat_frame(has_enemy_bar=False)
    in_range_bbox = (580, 260, 120, 160)  # norm_h = 160/720 = 0.222 >= melee_min_norm_h (0.11)

    # Step 1: Engage live monster -> locks target and attacks
    engaged = combat_sys.fight_back_step(input_ctrl=driver, frame=alive_frame, target_bbox=in_range_bbox, target_hp_pct=80.0)
    assert engaged is True
    assert combat_sys.in_combat is True
    assert combat_sys.was_fighting is True
    assert combat_sys.no_target_streak == 0

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
    Test 2: Verify CombatRecovery.execute_combat_loop immediately loots ('f') and presses 'tab'
    when a monster dies, seamlessly chains into the newly targeted monster, and returns 'CLEARED'.
    """
    monkeypatch.setattr("time.sleep", lambda _s: None)
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

    assert result == "CLEARED"
    assert "FIGHTING" in statuses
    assert statuses[-1] == "CLEARED"
    assert recovery.in_combat is False

    # Verify 'f' (loot) was pressed at least twice (for Monster 1 and Monster 2) and 'tab' was cycled
    assert driver.keys_pressed.count("f") >= 2
    assert driver.keys_pressed.count("tab") >= 3

    # Verify that after the first 'f' (loot), the very next key action is 'tab' (new target acquisition)
    first_f_idx = driver.keys_pressed.index("f")
    assert driver.keys_pressed[first_f_idx + 1] == "tab"


def test_3_target_lock_flicker_grace_prevents_false_loot_before_actual_death() -> None:
    """
    Test 3: Verify CombatRecovery 1-frame targeting flicker grace sustains target lock during
    transient UI flicker without false target loss, and confirms loss on the subsequent frame.
    """
    driver = _DummyInputDriver()
    recovery = CombatRecovery(input_ctrl=driver)
    recovery.max_flicker_grace_ticks = 1

    alive_frame = _make_combat_frame(has_enemy_bar=True)
    flicker_frame = _make_combat_frame(has_enemy_bar=False)

    # Frame 1: Hostile target bar visible -> lock acquired, grace inactive
    assert recovery.detect_enemy_target_bar(alive_frame, use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is False

    # Frame 2: 1-frame UI flicker (bar missing for 1 tick) -> lock sustained via grace
    assert recovery.detect_enemy_target_bar(flicker_frame, use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is True

    # Frame 3: Bar reappears -> grace resets cleanly without interrupting combat
    assert recovery.detect_enemy_target_bar(alive_frame, use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is False

    # Frame 4 & 5: Monster actually dies (2 consecutive frames without bar) -> grace expires on 2nd frame
    assert recovery.detect_enemy_target_bar(flicker_frame, use_flicker_grace=True) is True
    assert recovery.is_targeting_flicker_grace() is True
    assert recovery.detect_enemy_target_bar(flicker_frame, use_flicker_grace=True) is False
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

    # 1. Learned 3D monster crop -> Validated with high confidence (>= 0.85)
    valid_mob, conf_mob = combat.is_valid_monster_target(frame=None, target_crop=monster_crop)
    assert valid_mob is True
    assert conf_mob >= 0.85

    # 2. Learned negative exemplar (rock / friendly NPC) -> Rejected
    valid_neg, conf_neg = combat.is_valid_monster_target(frame=None, target_crop=neg_rock_crop)
    assert valid_neg is False
    assert conf_neg <= 0.25

    # 3. Learned gatherable resource (class_id=0 herb_aria) -> Rejected as non-monster
    valid_herb, conf_herb = combat.is_valid_monster_target(frame=None, target_crop=herb_crop)
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
