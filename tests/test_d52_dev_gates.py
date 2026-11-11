# -*- coding: utf-8 -*-
"""D52 DEV tests: vectorized dedup/NMS/gate-3 + precompiled Gate-1 patterns
+ hygiene fast path. Every test locks BEHAVIOUR EQUIVALENCE with the
pre-D52 implementation (reference re-implementations live in this file),
not just "it runs"."""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_factory.game_frame_gate import (
    BLOCKED_WINDOW_TITLES,
    GameFrameGate,
    _BLOCKED_TITLE_PATTERNS,
    _OWN_PID,
)
from dataset_factory.pre_training_hygiene_gate import PreTrainingHygieneGate
from dataset_factory.quality_dedup_gate import DatasetQualityGate

MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------------------
# I1: is_duplicate_hash — vectorized Hamming scan
# ---------------------------------------------------------------------------
def _ref_is_duplicate(gate, probe):
    if not gate._hash_history:
        return False, 64
    min_dist = min(gate.hamming_distance(probe, prev) for prev in gate._hash_history)
    return (min_dist < gate.min_hamming_distance), int(min_dist)


def test_is_duplicate_hash_matches_python_reference():
    rng = np.random.default_rng(11)
    gate = DatasetQualityGate(min_hamming_distance=5, max_hash_history=512)
    for _ in range(512):
        gate._hash_history.append(int(rng.integers(0, 2**63)))
    probes = [int(rng.integers(0, 2**63)) for _ in range(50)]
    probes.append(int(gate._hash_history[0]))  # exact duplicate -> dist 0
    probes.append(int(gate._hash_history[0]) ^ 0b111)  # dist 3 < 5 -> dup
    probes.append(int(gate._hash_history[0]) ^ 0b11111)  # dist 5 == threshold -> NOT dup
    probes.append((1 << 64) + 5)  # >64-bit probe is masked, like hamming_distance
    for probe in probes:
        assert gate.is_duplicate_hash(probe) == _ref_is_duplicate(gate, probe)


def test_is_duplicate_hash_empty_history():
    gate = DatasetQualityGate()
    assert gate.is_duplicate_hash(12345) == (False, 64)


# ---------------------------------------------------------------------------
# I2: vectorized class-aware NMS
# ---------------------------------------------------------------------------
def _ref_nms(gate, annotations):
    """Pure-Python copy of the pre-D52 NMS loop (compute_iou_xywh per pair),
    run on already-canonical boxes so only the NMS stage is compared."""
    by_class = {}
    for item in annotations:
        by_class.setdefault(int(item["class_id"]), []).append(item)
    kept_all = []
    for cid in sorted(by_class):
        cands = sorted(
            by_class[cid],
            key=lambda x: (float(x.get("confidence", 1.0)), float(x.get("area", 0.0))),
            reverse=True,
        )
        kept = []
        for cand in cands:
            box_c = (float(cand["cx"]), float(cand["cy"]), float(cand["w"]), float(cand["h"]))
            if any(
                gate.compute_iou_xywh(
                    box_c,
                    (float(e["cx"]), float(e["cy"]), float(e["w"]), float(e["h"])),
                )
                > gate.nms_iou_threshold
                for e in kept
            ):
                continue
            kept.append(cand)
        kept_all.extend(kept)
    return kept_all


def _canon_key(ann):
    return (
        int(ann["class_id"]),
        float(ann["cx"]),
        float(ann["cy"]),
        float(ann["w"]),
        float(ann["h"]),
        float(ann["confidence"]),
    )


def test_nms_matches_python_reference_on_random_boxes():
    rng = np.random.default_rng(23)
    gate = DatasetQualityGate()
    anns = []
    for k in range(300):
        anns.append(
            {
                "class_id": int(rng.integers(0, 3)),
                # Pre-rounded to 6 decimals so sanitize is the identity and
                # the reference (raw values) compares exactly.
                "cx": round(float(0.2 + (k % 10) * 0.03), 6),
                "cy": round(float(0.2 + (k % 7) * 0.05), 6),
                "w": 0.2,
                "h": 0.15,
                "confidence": round(float(0.95 - (k % 50) * 0.001), 6),
            }
        )
    got = gate.validate_yolo_annotations(anns)
    want = _ref_nms(gate, anns)
    assert [_canon_key(a) for a in got] == [_canon_key(a) for a in want]


def test_nms_keeps_highest_confidence_and_is_class_aware():
    gate = DatasetQualityGate()
    anns = [
        {"class_id": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.9},
        {"class_id": 3, "cx": 0.51, "cy": 0.51, "w": 0.2, "h": 0.2, "confidence": 0.5},
        {"class_id": 4, "cx": 0.51, "cy": 0.51, "w": 0.2, "h": 0.2, "confidence": 0.5},
        {"class_id": 3, "cx": 0.1, "cy": 0.1, "w": 0.1, "h": 0.1, "confidence": 0.8},
    ]
    kept = gate.validate_yolo_annotations(anns)
    assert sorted(_canon_key(a) for a in kept) == sorted(
        _canon_key(a)
        for a in [anns[0], anns[2], anns[3]]
    )


def test_iou_vectorized_matches_scalar():
    rng = np.random.default_rng(5)
    gate = DatasetQualityGate()
    others = np.array(
        [[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1], [0.9, 0.9, 0.3, 0.3]],
        dtype=np.float64,
    )
    box = (0.52, 0.48, 0.22, 0.18)
    vec = gate._iou_xywh_against_many(box, others)
    for i, other in enumerate(others):
        assert float(vec[i]) == pytest.approx(
            gate.compute_iou_xywh(box, tuple(float(v) for v in other))
        )
    assert vec.shape == (3,)


# ---------------------------------------------------------------------------
# I3: GameFrameGate Gate-3 — vectorized novelty scan + LRU touch
# ---------------------------------------------------------------------------
def test_gate3_matches_reference_and_moves_closest_to_end():
    rng = np.random.default_rng(31)
    gate = GameFrameGate(templates_dir=PROJECT_ROOT / "__nonexistent__" / "templates")
    hashes = [int(rng.integers(0, 2**63)) for _ in range(200)]
    for h in hashes:
        gate._hash_cache[h] = None
    base = hashes[50]
    probe_hash = base ^ 0b11  # hamming 2 < 6 -> duplicate; closest must be base
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    frame[:] = (10, 200, 30)

    # Monkeypatch compute_dhash to return the probe hash deterministically.
    gate.compute_dhash = staticmethod(lambda _f: probe_hash)  # type: ignore[method-assign]
    ok, reason = gate.check_gate3_visual_novelty(frame, update_cache=True)
    assert ok is False
    assert "duplicate_frame" in reason
    assert list(gate._hash_cache.keys())[-1] == base  # LRU touch preserved
    assert len(gate._hash_cache) == 200  # duplicate is NOT inserted

    far_hash = base ^ ((1 << 63) | (1 << 40) | (1 << 20) | (1 << 10) | 1 | 2 | 4)
    gate.compute_dhash = staticmethod(lambda _f: far_hash)  # type: ignore[method-assign]
    ok2, _ = gate.check_gate3_visual_novelty(frame, update_cache=True)
    assert ok2 is True
    assert list(gate._hash_cache.keys())[-1] == far_hash
    assert len(gate._hash_cache) == 201

    gate2 = GameFrameGate(templates_dir=PROJECT_ROOT / "__nonexistent__" / "templates")
    gate2.compute_dhash = staticmethod(lambda _f: far_hash)  # type: ignore[method-assign]
    ok3, _ = gate2.check_gate3_visual_novelty(frame, update_cache=False)
    assert ok3 is True
    assert len(gate2._hash_cache) == 0  # update_cache=False inserts nothing


# ---------------------------------------------------------------------------
# I4: Gate-1 — precompiled blocked-title patterns + cached own PID
# ---------------------------------------------------------------------------
class _StubFocus:
    def __init__(self, title, pid=424242):
        self._t = title
        self._pid = pid

    def query_foreground_window_info(self):
        return (1234, self._pid, self._t)

    def is_game_window_focused(self, allow_gatherion_ui=True):
        return True, "ok"

    def set_simulated_foreground_window(self, *a):
        pass

    def clear_simulated_foreground_window(self):
        pass


def _gate1(title):
    gate = GameFrameGate(
        templates_dir=PROJECT_ROOT / "__nonexistent__" / "templates",
        focus_guard=_StubFocus(title),
        is_game_running_fn=lambda: True,
    )
    return gate.check_gate1_process_and_focus()


def test_blocked_title_patterns_are_precompiled():
    assert len(_BLOCKED_TITLE_PATTERNS) == len(BLOCKED_WINDOW_TITLES)
    assert all(isinstance(p, re.Pattern) for p in _BLOCKED_TITLE_PATTERNS)
    assert _OWN_PID > 0


@pytest.mark.parametrize(
    "title,blocked",
    [
        ("AION 2", False),
        ("Visual Studio Code - foo.py", True),
        ("Google Chrome", True),
        ("Task Manager", True),
        ("File Explorer", True),
        ("decode notes", False),  # F-1054: substring must NOT match
        ("my decoder window", False),
        ("cursor rules", True),
    ],
)
def test_gate1_blocked_titles(title, blocked):
    ok, reason = _gate1(title)
    assert ok is (not blocked), reason
    if blocked:
        assert "blocked_foreground_window" in reason


# ---------------------------------------------------------------------------
# I5: hygiene validate_normalized_box fast path == slow path
# ---------------------------------------------------------------------------
def _ref_validate_normalized_box(gate, cx, cy, w, h):
    """Verbatim copy of the pre-D52 slow path (no fast path)."""
    for val in (cx, cy, w, h):
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            return False, "corrupted", None
    if w <= 0.0 or h <= 0.0:
        return False, "corrupted", None
    tol = gate.clamp_tolerance
    if cx < -tol or cx > 1.0 + tol or cy < -tol or cy > 1.0 + tol or w > 1.0 + tol or h > 1.0 + tol:
        return False, "corrupted", None
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    x1, y1, x2, y2 = cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0
    if x1 < -tol or y1 < -tol or x2 > 1.0 + tol or y2 > 1.0 + tol:
        return False, "corrupted", None
    x1, y1, x2, y2 = (
        min(max(x1, 0.0), 1.0),
        min(max(y1, 0.0), 1.0),
        min(max(x2, 0.0), 1.0),
        min(max(y2, 0.0), 1.0),
    )
    w, h = x2 - x1, y2 - y1
    if w <= 0.0 or h <= 0.0:
        return False, "corrupted", None
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    area = w * h
    if w < gate.min_box_side or h < gate.min_box_side:
        return False, "poisoned", None
    if area < gate.min_box_area_ratio or area > gate.max_box_area_ratio:
        return False, "poisoned", None
    if max(w / h, h / w) > gate.max_aspect_ratio:
        return False, "poisoned", None
    return True, "ok", (cx, cy, w, h)


def test_hygiene_fast_path_matches_slow_path_on_random_boxes():
    rng = np.random.default_rng(41)
    gate = PreTrainingHygieneGate()
    cases = []
    for _ in range(400):
        cases.append(tuple(float(v) for v in rng.uniform(-0.05, 1.05, size=4)))
    cases += [
        (0.5, 0.5, 0.2, 0.3),  # plain valid
        (0.0, 0.5, 0.2, 0.2),  # edge-touching, x1 == 0
        (1.0, 1.0, 0.2, 0.2),  # corner, x2 == 1.1 -> corrupted (slow path)
        (0.5, 0.5, 0.0, 0.2),  # zero width -> corrupted, NOT poisoned
        (0.5, 0.5, -0.1, 0.2),
        (float("nan"), 0.5, 0.2, 0.2),
        (float("inf"), 0.5, 0.2, 0.2),
        (0.5, 0.5, float("inf"), 0.2),
        (1.5, 0.5, 0.2, 0.2),
        (0.5, 0.5, 0.00005, 0.2),  # micro side -> poisoned
        (0.5, 0.5, 0.999, 0.001),  # extreme aspect -> poisoned
        (1.0005, 0.5, 0.0001, 0.2),  # in-tolerance drift, clipped -> poisoned
        (0.9999, 0.5, 0.0002, 0.2),  # in-tolerance, edge inside -> poisoned (aspect 1000)
        (0.5, 0.5, 1.0, 1.0),  # full frame -> poisoned (area)
    ]
    for box in cases:
        got = gate.validate_normalized_box(*box)
        want = _ref_validate_normalized_box(gate, *box)
        assert got[0] == want[0] and got[1] == want[1], box
        assert got[2] == want[2], box  # bit-identical box, not just approx


def test_hygiene_valid_box_exact_values():
    gate = PreTrainingHygieneGate()
    ok, reason, box = gate.validate_normalized_box(0.5, 0.5, 0.2, 0.3)
    assert (ok, reason) == (True, "ok")
    assert box == pytest.approx((0.5, 0.5, 0.2, 0.3))
