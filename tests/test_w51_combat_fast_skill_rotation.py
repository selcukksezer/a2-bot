# -*- coding: utf-8 -*-
"""
Unit tests for W51: Fast 7-Key ('1,2,3,4,q,e,r') Round-Robin Combat Skill Rotation
and SkillBarReader Ready Skill Selection without Sticky '2' Key Bias.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.mana_phase_combat import (
    DEFAULT_BURST_INTERVAL_SEC,
    DEFAULT_COMBAT_SKILL_KEYS,
    ManaPhase,
    ManaPhaseCombat,
)
from core.skill_bar_reader import SkillBarReader
from dataset_factory.exemplar_memory import ExemplarMemory


def _draw_skill_slot_icon(
    frame: np.ndarray,
    reader: SkillBarReader,
    slot_key: str,
    *,
    ready: bool = True,
    mana_ok: bool = True,
    seed: int = 42,
) -> None:
    """Draws a synthetic skill icon in the specified slot of the skill bar ROI."""
    fh, fw = frame.shape[:2]
    roi = reader.current_skill_bar_roi
    x1 = int(round(roi[0] * fw))
    y1 = int(round(roi[1] * fh))
    x2 = int(round(roi[2] * fw))
    y2 = int(round(roi[3] * fh))

    slots = list(reader.SKILL_SLOTS)
    idx = slots.index(slot_key.upper() if slot_key.upper() in ("Q", "E", "R") else slot_key)
    slot_w = float(x2 - x1) / float(len(slots))
    sx1 = x1 + int(round(idx * slot_w))
    sx2 = x1 + int(round((idx + 1) * slot_w))
    sh = max(2, y2 - y1)
    sw = max(2, sx2 - sx1)

    rng = np.random.default_rng(seed + idx * 17)
    if ready and mana_ok:
        # High brightness, high saturation, high contrast + blue mana indicator
        hsv = np.zeros((sh, sw, 3), dtype=np.uint8)
        hsv[:, :, 0] = 95  # Cyan/blue hue
        hsv[:, :, 1] = 200
        grad = np.linspace(80, 245, sw, dtype=np.float32)
        noise = rng.normal(0.0, 25.0, (sh, sw))
        hsv[:, :, 2] = np.clip(np.tile(grad, (sh, 1)) + noise, 55, 255).astype(np.uint8)
        patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.rectangle(patch, (3, 3), (sw - 4, sh - 4), (255, 210, 40), 2)
    elif not mana_ok:
        # Red "out of mana" dominant overlay (R >> B and R >> G)
        patch = np.zeros((sh, sw, 3), dtype=np.uint8)
        patch[:, :] = (15, 20, 210)
    else:
        # Dark, low-saturation, low-contrast cooldown state
        patch = np.full((sh, sw, 3), 22, dtype=np.uint8)

    frame[y1:y2, sx1:sx2] = patch


def test_seven_key_fast_round_robin_burst_rotation() -> None:
    """1. Verify 7-key ('1,2,3,4,q,e,r') sub-100ms round-robin rotation cycles evenly without sticking to '2'."""
    combat = ManaPhaseCombat(burst_skill_interval_sec=0.08, gcd_delay_sec=0.12)
    combat.reset_rotation()

    assert combat.combat_skill_keys == ["1", "2", "3", "4", "q", "e", "r"]
    assert combat.burst_skill_interval_sec == DEFAULT_BURST_INTERVAL_SEC == 0.08

    cast_counts: Dict[str, int] = {k: 0 for k in DEFAULT_COMBAT_SKILL_KEYS}
    cast_sequence: List[str] = []
    t = 1000.0

    # Run 21 fast combat burst cycles (3 full rotations of 7 keys)
    for _ in range(21):
        assert combat.should_cast_skill(now=t) is True
        decision = combat.decide(player_mp=1.0, player_hp=1.0, now=t)
        assert decision.phase == ManaPhase.BURST
        primary = decision.skill_keys[0]
        cast_sequence.append(primary)
        cast_counts[primary] += 1
        combat.record_skill_cast(primary, now=t)
        t += 0.085  # Sub-100ms fast burst interval

    # Every one of the 7 keys must be cast exactly 3 times, with no consecutive duplicates
    for key in DEFAULT_COMBAT_SKILL_KEYS:
        assert cast_counts[key] == 3, f"Expected key '{key}' to be cast 3 times, got {cast_counts[key]}"
    for i in range(1, len(cast_sequence)):
        assert cast_sequence[i] != cast_sequence[i - 1], f"Stuck on key '{cast_sequence[i]}' at step {i}"


def test_anti_sticky_key_2_guard_when_only_key_2_reported_ready() -> None:
    """2. Verify that when vision repeatedly reports only ['2'] ready, rotation never locks onto '2'."""
    combat = ManaPhaseCombat()
    combat.reset_rotation()

    primaries: List[str] = []
    t = 2000.0

    for _ in range(14):
        decision = combat.decide(player_mp=1.0, player_hp=1.0, ready_skills=["2"], now=t)
        primary = decision.skill_keys[0]
        primaries.append(primary)
        combat.record_skill_cast(primary, now=t)
        t += 0.08

    # Key '2' must never be cast twice in a row
    for idx in range(1, len(primaries)):
        if primaries[idx] == "2":
            assert primaries[idx - 1] != "2", f"Sticky '2' repetition at index {idx}: {primaries}"

    # All 7 combat keys ('1','2','3','4','q','e','r') participate across the 14 steps
    unique_keys = set(primaries)
    assert unique_keys == set(DEFAULT_COMBAT_SKILL_KEYS)


def test_skill_bar_reader_selects_only_ready_and_mana_ok_skills(tmp_path: Path) -> None:
    """3. Verify SkillBarReader inspects brightness/saturation/mana and selects only ready skills."""
    mem = ExemplarMemory(json_path=tmp_path / "empty_exemplars.json")
    reader = SkillBarReader(
        calibrated_rois_path=tmp_path / "missing_rois.json",
        exemplar_memory=mem,
        cooldown_lockout_sec=0.15,
    )

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    # Mark '2' on cooldown (dark) and '1' out of mana (red), while '3', 'Q', 'R' are ready
    for slot in reader.SKILL_SLOTS:
        if slot in ("3", "Q", "R"):
            _draw_skill_slot_icon(frame, reader, slot, ready=True, mana_ok=True)
        elif slot == "1":
            _draw_skill_slot_icon(frame, reader, slot, ready=True, mana_ok=False)
        else:
            _draw_skill_slot_icon(frame, reader, slot, ready=False, mana_ok=True)

    states = reader.inspect_skill_bar(frame)
    assert states["2"]["ready"] is False
    assert states["2"]["off_cooldown"] is False
    assert states["1"]["mana_ok"] is False
    assert states["1"]["ready"] is False
    assert states["3"]["ready"] is True
    assert states["Q"]["ready"] is True
    assert states["R"]["ready"] is True

    # Selecting 3 times in a row must pick '3', 'Q', 'R' in round-robin order without picking '2' or '1'
    picked = [
        reader.select_best_ready_skill(frame, record_use=True),
        reader.select_best_ready_skill(frame, record_use=True),
        reader.select_best_ready_skill(frame, record_use=True),
    ]
    assert picked == ["3", "Q", "R"]
    assert "2" not in picked
    assert "1" not in picked


def test_skill_bar_reader_7_key_round_robin_without_slot_2_bias(tmp_path: Path) -> None:
    """4. Verify SkillBarReader rotates across all 7 combat slots ('1,2,3,4,Q,E,R') even if slot '2' is brightest."""
    mem = ExemplarMemory(json_path=tmp_path / "empty_exemplars.json")
    reader = SkillBarReader(
        calibrated_rois_path=tmp_path / "missing_rois.json",
        exemplar_memory=mem,
        cooldown_lockout_sec=0.50,
    )

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    seven_slots = ["1", "2", "3", "4", "Q", "E", "R"]
    for slot in seven_slots:
        _draw_skill_slot_icon(frame, reader, slot, ready=True, mana_ok=True)
    # Keep slot '5' on cooldown so only the 7 combat keys are ready
    _draw_skill_slot_icon(frame, reader, "5", ready=False, mana_ok=False)

    # Equal baseline priority weights ensure no artificial bias toward slot '2'
    assert reader.DEFAULT_PRIORITY_WEIGHTS["2"] == reader.DEFAULT_PRIORITY_WEIGHTS["1"] == 1.0

    selected_order: List[str] = []
    for _ in range(14):
        chosen = reader.select_best_ready_skill(frame, preferred_order=seven_slots, record_use=True)
        selected_order.append(chosen)

    # First 7 selections and second 7 selections must both be the exact 7-key round-robin sequence
    assert selected_order[:7] == seven_slots
    assert selected_order[7:14] == seven_slots


def test_integrated_skill_bar_reader_and_mana_phase_combat_rotation(tmp_path: Path) -> None:
    """5. Verify end-to-end integration of SkillBarReader ready skills feeding ManaPhaseCombat 7-key rotation."""
    mem = ExemplarMemory(json_path=tmp_path / "empty_exemplars.json")
    reader = SkillBarReader(
        calibrated_rois_path=tmp_path / "missing_rois.json",
        exemplar_memory=mem,
        cooldown_lockout_sec=0.12,
    )
    combat = ManaPhaseCombat(burst_skill_interval_sec=0.08, gcd_delay_sec=0.12)
    combat.reset_rotation()

    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    for slot in ("1", "2", "3", "4", "Q", "E", "R"):
        _draw_skill_slot_icon(frame, reader, slot, ready=True, mana_ok=True)
    _draw_skill_slot_icon(frame, reader, "5", ready=False, mana_ok=False)

    ready_from_bar = [s.lower() for s in reader.get_ready_skills(frame) if s != "5"]
    assert ready_from_bar == ["1", "2", "3", "4", "q", "e", "r"]

    executed_skills: List[str] = []
    t = 5000.0
    for _ in range(7):
        assert combat.should_cast_skill(now=t) is True
        decision = combat.decide(
            player_mp=1.0,
            player_hp=1.0,
            ready_skills=ready_from_bar,
            now=t,
        )
        primary = decision.skill_keys[0]
        executed_skills.append(primary)
        combat.record_skill_cast(primary, now=t)
        reader.mark_slot_pressed(primary)
        t += 0.08

    # All 7 combat skills were executed once in 7 decisions without getting stuck on '2'
    assert executed_skills == ["1", "4", "r", "3", "e", "2", "q"]
    assert set(executed_skills) == set(DEFAULT_COMBAT_SKILL_KEYS)
    assert executed_skills.count("2") == 1
