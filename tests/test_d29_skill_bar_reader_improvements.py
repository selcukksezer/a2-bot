# -*- coding: utf-8 -*-
"""D29 regression tests for the skill_bar_reader improvements.

Locks in the behaviour of the shared helpers introduced by the D29
refactor (single ROI resolver, single slot-centre geometry, single
candidate sort, single-validation readiness chain, row-level colour
conversion + exemplar-skip in the consumable path).

Run:  pytest tests/test_d29_skill_bar_reader_improvements.py -v
"""
import json

import cv2
import numpy as np
import pytest

from core import skill_bar_reader as sbr


# ---------------------------------------------------------------- helpers
def _frame(seed=42):
    rng = np.random.default_rng(seed)
    frame = rng.integers(0, 90, size=(1080, 1920, 3), dtype=np.uint8)
    frame[1015:1063, 662:1240] = rng.integers(120, 255, size=(48, 578, 3), dtype=np.uint8)
    frame[1020:1058, 700:760, 0] = 200
    frame[1020:1058, 700:760, 1] = 120
    frame[1020:1058, 700:760, 2] = 60
    return frame


def _write_rois(tmp_path, payload, name="rois.json"):
    p = tmp_path / name
    p.write_text(json.dumps(payload))
    return str(p)


# ---------------------------------------------------------------- _parse_roi_entry
def test_parse_roi_entry_norm_xywh():
    assert sbr._parse_roi_entry(
        {"norm_x": 0.30, "norm_y": 0.93, "norm_w": 0.35, "norm_h": 0.05}
    ) == (0.3, 0.93, 0.65, 0.98)


def test_parse_roi_entry_normalized_roi_wh():
    # [x, y, w, h] fits inside the frame.
    assert sbr._parse_roi_entry({"normalized_roi": [0.30, 0.93, 0.35, 0.05]}) == (
        0.3, 0.93, 0.65, 0.98,
    )


def test_parse_roi_entry_normalized_roi_xywh_overflow_clamps():
    # S099/S105 (v19/P3-8): normalized_roi is ALWAYS [x, y, w, h] — the
    # old corner-guess branch is gone. [0.30, 0.93, 0.65, 0.98] is an
    # [x, y, w, h] ROI whose w/h overflow the frame edge, so x2/y2 clamp:
    # x+w = 0.95, y+h = 1.91 -> 1.0. It is NOT [x1, y1, x2, y2] corners.
    # (Real stored data proves the format: every normalized_roi in
    # config/calibrated_rois.json equals its norm_x/norm_y/norm_w/norm_h
    # fields and is invalid as corners, e.g. minimap [0.855, 0.065,
    # 0.137, 0.19] would be x2=0.137 < x1=0.855.)
    assert sbr._parse_roi_entry({"normalized_roi": [0.30, 0.93, 0.65, 0.98]}) == (
        0.3, 0.93, 0.95, 1.0,
    )


def test_parse_roi_entry_unusable():
    assert sbr._parse_roi_entry({"foo": 1}) is None
    assert sbr._parse_roi_entry([1, 2, 3, 4]) is None
    assert sbr._parse_roi_entry(None) is None


def test_skill_score_bucket():
    assert sbr._skill_score_bucket({"score": 0.25}) == 2
    assert sbr._skill_score_bucket({"score": 0.29}) == 2
    assert sbr._skill_score_bucket({"score": 0.31}) == 3
    assert sbr._skill_score_bucket({"score": "nope"}) == 0
    assert sbr._skill_score_bucket({}) == 0


# ---------------------------------------------------------------- shared ROI resolver
FALLBACK_BAR = (0.345, 0.940, 0.645, 0.985)
FALLBACK_CONS = (0.345, 0.870, 0.655, 0.938)


@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"skill_bar": {"norm_x": 0.30, "norm_y": 0.93, "norm_w": 0.35, "norm_h": 0.05}},
         (0.3, 0.93, 0.65, 0.98)),
        ({"skill_bar": {"normalized_roi": [0.30, 0.93, 0.35, 0.05]}},
         (0.3, 0.93, 0.65, 0.98)),
        ({"skill_bar": {"normalized_roi": [0.30, 0.93, 0.65, 0.98]}},
         # S099/S105: always [x, y, w, h] — overflow clamps, not corners.
         (0.3, 0.93, 0.95, 1.0)),
        # Too narrow (< 0.10 wide) -> fallback.
        ({"skill_bar": {"norm_x": 0.30, "norm_y": 0.93, "norm_w": 0.05, "norm_h": 0.05}},
         FALLBACK_BAR),
        ({}, FALLBACK_BAR),
    ],
)
def test_resolve_skill_bar_roi(tmp_path, payload, expected):
    path = _write_rois(tmp_path, payload)
    assert sbr.SkillBarReader._resolve_skill_bar_roi_static(
        json_path=path, fallback=FALLBACK_BAR
    ) == expected


def test_resolve_skill_bar_roi_missing_and_malformed(tmp_path):
    assert sbr.SkillBarReader._resolve_skill_bar_roi_static(
        json_path=str(tmp_path / "nope.json"), fallback=FALLBACK_BAR
    ) == FALLBACK_BAR
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert sbr.SkillBarReader._resolve_skill_bar_roi_static(
        json_path=str(bad), fallback=FALLBACK_BAR
    ) == FALLBACK_BAR


def test_resolve_consumable_roi_key_scan(tmp_path):
    # First matching key wins; later keys are only consulted on miss.
    payload = {
        "consumables": {"norm_x": 0.31, "norm_y": 0.86, "norm_w": 0.34, "norm_h": 0.07},
        "consumable_bar": {"norm_x": 0.20, "norm_y": 0.80, "norm_w": 0.40, "norm_h": 0.07},
    }
    path = _write_rois(tmp_path, payload)
    assert sbr.SmartConsumableManager._resolve_consumable_roi_static(
        json_path=path, fallback=FALLBACK_CONS
    ) == (0.2, 0.8, 0.6, 0.87)


def test_resolve_consumable_roi_fallback(tmp_path):
    path = _write_rois(tmp_path, {})
    assert sbr.SmartConsumableManager._resolve_consumable_roi_static(
        json_path=path, fallback=FALLBACK_CONS
    ) == FALLBACK_CONS


# ---------------------------------------------------------------- shared slot-centre geometry
def test_slot_center_px_exact_geometry():
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    roi = (0.25, 0.90, 0.75, 0.98)  # x 200..600, y 90..98 -> 8 slots of 50px
    # Slot "1" centre x = 225, slot "R" (idx 7) centre x = 575, cy = 94.
    assert sbr._slot_center_px(frame, "1", ["1", "2", "3", "4", "5", "Q", "E", "R"], roi, jitter_px=0) == (225, 94)
    assert sbr._slot_center_px(frame, "R", ["1", "2", "3", "4", "5", "Q", "E", "R"], roi, jitter_px=0) == (575, 94)
    # Case-insensitive match ("Q" is idx 5 -> centre 475).
    assert sbr._slot_center_px(frame, "q", ["1", "2", "3", "4", "5", "Q", "E", "R"], roi, jitter_px=0) == (475, 94)


def test_slot_center_px_invalid():
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    roi = (0.25, 0.90, 0.75, 0.98)
    slots = ["1", "2", "3", "4", "5", "Q", "E", "R"]
    assert sbr._slot_center_px(None, "1", slots, roi) is None
    assert sbr._slot_center_px(frame, "X", slots, roi) is None
    assert sbr._slot_center_px(frame, "", slots, roi) is None
    assert sbr._slot_center_px(np.zeros((0, 0, 3), dtype=np.uint8), "1", slots, roi) is None


def test_reader_slot_center_matches_shared_helper(monkeypatch):
    reader = sbr.SkillBarReader()
    monkeypatch.setattr(sbr.SkillBarReader, "current_skill_bar_roi",
                        property(lambda self: (0.25, 0.90, 0.75, 0.98)))
    frame = np.zeros((100, 800, 3), dtype=np.uint8)
    # Kanonik 12'li bar: OLCULMUS merkezler (SLOT_CENTERS_NORM, G2), esit
    # bolunme degil. ROI x 200..600 (bw 400); slot "2" kesri
    # (0.387-0.345)/0.315 = 0.1333 -> cx 200 + round(0.1333*400) = 253.
    # Slot "1" (0.361) -> 220, slot "T" (0.640) -> 575; cy = 94.
    assert reader.slot_center_px(frame, "1", jitter_px=0) == (220, 94)
    assert reader.slot_center_px(frame, "2", jitter_px=0) == (253, 94)
    assert reader.slot_center_px(frame, "T", jitter_px=0) == (575, 94)


def test_slot_centers_norm_measured_12_slots():
    # G2 olcumu (core/skill_bar_reader.py::SLOT_CENTERS_NORM): 12 slot,
    # 4 kare / 3 oturumdan olculmus merkezler; brifingdeki eski REF
    # degerleri degil, kodun kendisi tek dogruluk kaynagi.
    assert list(sbr.SkillBarReader.SKILL_SLOTS) == [
        "1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T",
    ]
    assert sbr.SLOT_CENTERS_NORM == {
        "1": 0.361, "2": 0.387, "3": 0.412, "4": 0.438, "5": 0.463,
        "6": 0.488, "7": 0.513, "8": 0.539,
        "Q": 0.566, "E": 0.589, "R": 0.614, "T": 0.640,
    }
    assert len(sbr.SLOT_CENTER_FRACTIONS) == 12


# ---------------------------------------------------------------- candidate sort
def _candidates():
    return [
        {"slot": "1", "score": 0.25},
        {"slot": "2", "score": 0.24},
        {"slot": "Q", "score": 0.85},
        {"slot": "R", "score": 0.82},
    ]


def test_sort_ready_candidates_score_bucket_then_recency():
    reader = sbr.SkillBarReader()
    cands = _candidates()
    reader._last_pressed_ts["Q"] = 100.0  # Q pressed recently, R never
    reader._sort_ready_candidates(cands)
    # 0.8 bucket first (Q, R — R older press wins), then 0.2 bucket (1, 2).
    assert [c["slot"] for c in cands] == ["R", "Q", "1", "2"]


def test_sort_ready_candidates_preferred_order():
    reader = sbr.SkillBarReader()
    cands = _candidates()
    reader._sort_ready_candidates(cands, {"R": 0, "Q": 1, "1": 2, "2": 3})
    assert [c["slot"] for c in cands] == ["R", "Q", "1", "2"]
    # Preferred order does NOT outrank the score bucket: 0.8 bucket still first.
    cands = [{"slot": "1", "score": 0.85}, {"slot": "Q", "score": 0.25}]
    reader._sort_ready_candidates(cands, {"Q": 0, "1": 1})
    assert [c["slot"] for c in cands] == ["1", "Q"]


def test_select_best_ready_skill_recency_rotation():
    reader = sbr.SkillBarReader()
    frame = _frame()
    first = reader.select_best_ready_skill(frame)
    assert first in sbr.SkillBarReader.SKILL_SLOTS
    # Deterministic for a fresh reader on the same frame.
    assert sbr.SkillBarReader().select_best_ready_skill(frame) == first
    # Pressing the chosen slot (recency) must not change the *set* of
    # candidates — the sort itself is covered by test_sort_ready_candidates_*.
    reader._last_pressed_ts[first] = 1000.0
    second = reader.select_best_ready_skill(frame)
    assert second in sbr.SkillBarReader.SKILL_SLOTS or second == ""


# ---------------------------------------------------------------- readiness chain: validate exactly once
def test_readiness_chain_validates_once(monkeypatch):
    calls = {"n": 0}
    real = sbr.validate_skill_bar_reader_contract

    def counting():
        calls["n"] += 1
        return real()

    monkeypatch.setattr(sbr, "validate_skill_bar_reader_contract", counting)
    snap = sbr.get_skill_bar_reader_heartbeat_snapshot()
    assert calls["n"] == 1, f"heartbeat validated {calls['n']} times, want 1"
    assert snap["alive"] is True
    assert snap["audit"]["probe"]["certificate"]["seal"]["telemetry"]["digest"][
        "audit"
    ]["readiness"]["metrics"]["contract_valid"] is True


def test_readiness_chain_functions_agree():
    valid = sbr.validate_skill_bar_reader_contract()
    chain = sbr._skill_bar_reader_chain(valid)
    assert sbr.get_skill_bar_reader_baseline_metrics() == chain["baseline"]
    assert sbr.get_skill_bar_reader_readiness_status() == chain["readiness"]
    assert sbr.get_skill_bar_reader_audit_record() == chain["audit_record"]
    assert sbr.get_skill_bar_reader_operational_digest() == chain["digest"]
    assert sbr.get_skill_bar_reader_telemetry_payload() == chain["telemetry"]
    assert sbr.get_skill_bar_reader_integrity_seal() == chain["seal"]
    assert sbr.get_skill_bar_reader_compliance_certificate() == chain["certificate"]
    assert sbr.get_skill_bar_reader_readiness_probe() == chain["probe"]
    assert sbr.get_skill_bar_reader_audit_summary() == chain["summary"]
    snap = sbr.get_skill_bar_reader_heartbeat_snapshot()
    assert snap == {"module": "skill_bar_reader", "alive": valid, "audit": chain["summary"]}


# ---------------------------------------------------------------- consumable path: precomputed conversions
def test_classify_precomputed_matches_plain_and_falls_back():
    mgr = sbr.SmartConsumableManager()
    rng = np.random.default_rng(7)
    crop = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
    crop[:, :, 0] = 220
    plain = mgr.classify_slot_crop(crop, slot_key="1")
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    pre = mgr.classify_slot_crop(crop, slot_key="1", precomputed_hsv=hsv, precomputed_gray=gray)
    assert pre == plain
    wrong = mgr.classify_slot_crop(
        crop, slot_key="1",
        precomputed_hsv=np.zeros((5, 5, 3), dtype=np.uint8),
        precomputed_gray=np.zeros((5, 5), dtype=np.uint8),
    )
    assert wrong == plain, "shape-mismatched precomputed arrays must fall back to converting the crop"


def test_classify_learned_exemplar_still_matches():
    mgr = sbr.SmartConsumableManager()
    rng = np.random.default_rng(7)
    crop = rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8)
    crop[:, :, 0] = 220
    assert mgr.learn_slot_assignment("1", "hp_potion", crop) is True
    assert len(mgr._exemplar_store.get("hp_potion", [])) > 0
    first = mgr.classify_slot_crop(crop, slot_key="1")
    second = mgr.classify_slot_crop(crop, slot_key="1")
    assert first == second
    assert "item_type" in first and "confidence" in first


def test_inspect_skill_bar_deterministic():
    reader = sbr.SkillBarReader()
    frame = _frame()
    assert reader.inspect_skill_bar(frame) == reader.inspect_skill_bar(frame)
    assert set(reader.inspect_skill_bar(frame).keys()) == {
        "1", "2", "3", "4", "5", "6", "7", "8", "Q", "E", "R", "T",
    }


def test_inspect_consumables_deterministic():
    mgr = sbr.SmartConsumableManager()
    frame = _frame()
    assert mgr.inspect_consumables(frame) == mgr.inspect_consumables(frame)
