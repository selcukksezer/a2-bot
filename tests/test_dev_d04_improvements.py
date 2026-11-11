# -*- coding: utf-8 -*-
"""DEV D04 regression tests: table-driven range evaluation, corridor crop
fidelity, and reflex state_from_boxes fast paths.

Expectations for range/corridor cases were recorded from the pre-change
code (probe run on the base commit) so these tests lock identical
(or, for the corridor crop on dense white noise, bounded-deviation)
behaviour, not just the new code agreeing with itself.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import pytest

from core.combat import (
    CombatPositioner,
    CombatPositioningConfig,
    EngagementProfile,
    RangeStatus,
    analyze_corridor_occlusion,
)
from core import reflex_layer
from core.reflex_layer import _gather_ids, state_from_boxes


# =============================================================================
# I1: table-driven evaluate_target_range — distance branch unchanged; bbox
# branch now uses the SAME single formula (P1-12): dist = 0.96 / norm_h
# (PINHOLE_K_M, core/combat.py:429) against the same meter limits
# (melee 3.8 m; ranged 6-24 m; hybrid 4-25 m). The old norm_h bounds
# (melee 0.11/0.70, ranged 0.035/0.160, hybrid 0.07/0.45) are legacy and
# no longer consulted — they implied incompatible pinhole constants.
# =============================================================================

# (profile, distance_m) -> (status, action, reason)
DIST_CASES = [
    ("melee", 0.5, RangeStatus.IN_RANGE, "none", "Melee target distance 0.5m in range"),
    ("melee", 3.8, RangeStatus.IN_RANGE, "none", "Melee target distance 3.8m in range"),
    ("melee", 3.9, RangeStatus.TOO_FAR, "step_forward", "Melee target distance 3.9m > 3.8m limit"),
    ("ranged", 2.5, RangeStatus.TOO_CLOSE, "backpedal", "Ranged target distance 2.5m < 6.0m danger zone"),
    ("ranged", 6.0, RangeStatus.IN_RANGE, "none", "Ranged target distance 6.0m in optimal standoff zone"),
    ("ranged", 24.0, RangeStatus.IN_RANGE, "none", "Ranged target distance 24.0m in optimal standoff zone"),
    ("ranged", 25.0, RangeStatus.TOO_FAR, "step_forward", "Ranged target distance 25.0m > 24.0m limit"),
    ("hybrid", 3.9, RangeStatus.TOO_CLOSE, "backpedal", "Hybrid target distance 3.9m < 4.0m danger zone"),
    ("hybrid", 25.0, RangeStatus.IN_RANGE, "none", "Hybrid target distance 25.0m in optimal zone"),
    ("hybrid", 26.0, RangeStatus.TOO_FAR, "step_forward", "Hybrid target distance 26.0m > 25.0m limit"),
]


@pytest.mark.parametrize("profile,dist,status,action,reason", DIST_CASES)
def test_range_distance_matches_pre_refactor(profile, dist, status, action, reason):
    pos = CombatPositioner(CombatPositioningConfig(profile=EngagementProfile(profile)))
    ev = pos.evaluate_target_range(estimated_distance_m=dist)
    assert ev.status == status
    assert ev.recommended_action == action
    assert ev.reason == reason
    assert ev.estimated_distance_m == dist


# (profile, bbox_h_px) -> (status, action, reason); screen 1080x1920.
# dist = 0.96 / (bh / 1080), clamped to [1.0, 50.0] (P1-12, core/combat.py).
# Melee has NO too-close check: point-blank is still in melee range.
BBOX_CASES = [
    ("melee", 86, RangeStatus.TOO_FAR, "step_forward", "Melee target distance 12.1m > 3.8m limit"),
    ("melee", 360, RangeStatus.IN_RANGE, "none", "Melee target distance 2.9m in range"),
    ("melee", 864, RangeStatus.IN_RANGE, "none", "Melee target distance 1.2m in range"),
    ("ranged", 25, RangeStatus.TOO_FAR, "step_forward", "Ranged target distance 41.5m > 24.0m limit"),
    ("ranged", 90, RangeStatus.IN_RANGE, "none", "Ranged target distance 11.5m in optimal standoff zone"),
    ("ranged", 270, RangeStatus.TOO_CLOSE, "backpedal", "Ranged target distance 3.8m < 6.0m danger zone"),
    ("hybrid", 75, RangeStatus.IN_RANGE, "none", "Hybrid target distance 13.8m in optimal zone"),
    ("hybrid", 118, RangeStatus.IN_RANGE, "none", "Hybrid target distance 8.8m in optimal zone"),
    ("hybrid", 756, RangeStatus.TOO_CLOSE, "backpedal", "Hybrid target distance 1.4m < 4.0m danger zone"),
]


@pytest.mark.parametrize("profile,bh,status,action,reason", BBOX_CASES)
def test_range_bbox_single_pinhole_formula(profile, bh, status, action, reason):
    pos = CombatPositioner(CombatPositioningConfig(profile=EngagementProfile(profile)))
    ev = pos.evaluate_target_range(target_bbox=(800, 300, 80, bh), screen_shape=(1080, 1920))
    assert ev.status == status
    assert ev.recommended_action == action
    assert ev.reason == reason


def test_range_unknown_without_inputs():
    pos = CombatPositioner()
    ev = pos.evaluate_target_range()
    assert ev.status == RangeStatus.UNKNOWN
    assert ev.recommended_action == "none"


# =============================================================================
# I3: corridor crop — same verdict as the full-frame computation
# =============================================================================

def _rock_canvas():
    canvas = np.full((1080, 1920, 3), (80, 120, 90), dtype=np.uint8)
    cv2.rectangle(canvas, (900, 500), (1020, 600), (20, 20, 25), -1)
    for y in range(510, 590, 8):
        cv2.line(canvas, (910, y), (1010, y), (240, 240, 240), 2)
    return canvas


def test_corridor_clear_and_rock_match_pre_change_values():
    clear = np.full((1080, 1920, 3), (80, 120, 90), dtype=np.uint8)
    is_obs, score, center, flank = analyze_corridor_occlusion(clear, (960, 800), (960, 350))
    assert is_obs is False and score == 0.0 and center is None

    # Pre-change values: (True, 0.41141868512110724, (960, 546), 'right')
    is_obs, score, center, flank = analyze_corridor_occlusion(
        _rock_canvas(), (960, 800), (960, 350), threshold=0.25
    )
    assert is_obs is True
    assert score == pytest.approx(0.41141868512110724, abs=1e-9)
    assert center == (960, 546)
    assert flank == "right"


def test_corridor_dense_noise_deviation_is_bounded():
    # Pre-change: (True, 0.8340703856865558, (960, 577), 'right').
    # Cropping changes Canny hysteresis chains slightly on dense white
    # noise; the verdict must stay identical and the score within 0.01.
    rng = np.random.default_rng(0)
    noisy = rng.integers(60, 140, size=(1080, 1920, 3), dtype=np.uint8)
    is_obs, score, center, flank = analyze_corridor_occlusion(noisy, (960, 800), (960, 350))
    assert is_obs is True
    assert score == pytest.approx(0.8340703856865558, abs=0.01)
    assert center is not None
    assert abs(center[0] - 960) <= 2 and abs(center[1] - 577) <= 2
    assert flank == "right"


# =============================================================================
# I2: state_from_boxes — same verdicts, cached gather ids, early monster exit
# =============================================================================

def test_state_from_boxes_verdicts():
    assert state_from_boxes([]) == "Arıyor"
    assert state_from_boxes([], movement_active=True) == "Yürüyor"
    assert state_from_boxes([{"class_id": 0, "class_name": "herb"}]) == "Topluyor"
    assert state_from_boxes([{"class_id": 10, "class_name": "monster"}]) == "Savaşta"
    # Monster wins even after gather boxes, and via name alone.
    assert state_from_boxes([
        {"class_id": 0, "class_name": "herb"},
        {"class_id": 10, "class_name": "monster"},
    ]) == "Savaşta"
    assert state_from_boxes([{"class_name": "monster"}]) == "Savaşta"
    assert state_from_boxes([{"class_name": "essence_fire"}]) == "Topluyor"
    assert state_from_boxes([{"class_id": 3, "class_name": "monster"}]) == "Savaşta"
    assert state_from_boxes([None, "junk", {"class_id": 1, "class_name": "ore"}]) == "Topluyor"


def test_gather_ids_cached_and_copy_isolated():
    first = _gather_ids()
    assert 0 in first and 12 in first
    first.add(999)  # mutating the returned copy must not poison the cache
    second = _gather_ids()
    assert 999 not in second
    assert reflex_layer._CACHED_GATHER_IDS is not None
    assert _gather_ids() == second


def test_state_monster_first_returns_at_once():
    boxes = [{"class_id": 10, "class_name": "monster"}] + [
        {"class_id": 0, "class_name": "herb"} for _ in range(5000)
    ]
    state_from_boxes(boxes)  # warm the gather-id cache
    start = time.perf_counter()
    assert state_from_boxes(boxes) == "Savaşta"
    assert time.perf_counter() - start < 0.010  # old code scanned all 5001 boxes (~140 ms)
