# -*- coding: utf-8 -*-
"""D38 tests for dataset_factory/active_learning_loop.py.

Covers the D38 improvements (locked with before/after measurements):
  I1. Rejection-scan fast path + bounded memo in evaluate_frame_for_active_learning
      (gatherable 50x: 4867 -> 632 us/call CPU; mixed 50x: 3253 -> 526 us/call CPU).
  I2. ShadowGameplayRecorder._recorded_samples keeps a <=320px thumbnail, not the
      full frame (5x 1080p samples: 31.1 MB -> 0.9 MB retained).
  I3. compute_sample_priority shares _extract_annotation_class_keys (was dead
      code); "cls"/"category" keys are now honoured consistently.
Also locks the pre-existing behaviour these touch (triggers, passive guarantee).
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

from dataset_factory.active_learning_loop import (
    KNOWN_GATHERABLE_LABELS,
    ActiveLearningLoop,
    ShadowGameplayRecorder,
    _is_rejected_non_material_cached,
)


def _loop() -> ActiveLearningLoop:
    loop = ActiveLearningLoop(
        initial_class_counts={"herb_aria": 500, "gem_crystal": 30, "ore_orichalcum": 120}
    )
    return loop


# ---------------------------------------------------------------- I1: fast path
# ----------------------------------------------------------------


def test_fast_path_labels_are_never_rejected():
    """Every label in KNOWN_GATHERABLE_LABELS must pass the rejection scan.

    This is what makes skipping the scan for them behaviour-preserving.
    """
    for label in sorted(KNOWN_GATHERABLE_LABELS):
        assert _is_rejected_non_material_cached(label) is False, label
    # Case/whitespace variants take the same code path (scan lowercases).
    for label in ("Herb_Aria", " HERB_ARIA ", "Fresh Aria", "SAPPHIRE", "Gem_Crystal"):
        assert _is_rejected_non_material_cached(label) is False, label


def test_gatherable_detections_not_rejected_but_monsters_are():
    loop = _loop()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dets = [
        {"class_name": name, "confidence": 0.95}
        for name in ("herb_aria", "gem_crystal", "ore_orichalcum", "tree_log", "Sapphire")
    ]
    out = loop.evaluate_frame_for_active_learning(frame, dets)
    assert out["rejected_detections"] == []
    assert len(out["high_confidence_detections"]) == 5

    bad = [
        {"class_name": name, "confidence": 0.95}
        for name in (
            "monster_3d",
            "minimap_monster_dot",
            "monster",
            "white feather",
            "functional npc",
            "quest",
            "canavar",
        )
    ]
    out = loop.evaluate_frame_for_active_learning(frame, bad)
    assert len(out["rejected_detections"]) == len(bad)
    assert out["hard_negative_mining"] is True


def test_reject_scan_cache_is_bounded_and_hit():
    _is_rejected_non_material_cached.cache_clear()
    _is_rejected_non_material_cached("monster_3d")
    _is_rejected_non_material_cached("monster_3d")
    info = _is_rejected_non_material_cached.cache_info()
    assert info.maxsize == 512
    assert info.hits == 1 and info.misses == 1


def test_evaluate_triggers_unchanged():
    loop = _loop()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Uncertainty band edge: 0.58 inside [0.42, 0.74] triggers.
    out = loop.evaluate_frame_for_active_learning(
        frame, [{"class_name": "gem_crystal", "confidence": 0.58}]
    )
    assert out["should_capture"] is True and out["uncertainty_mining"] is True
    # High confidence alone does not trigger.
    out = loop.evaluate_frame_for_active_learning(
        frame, [{"class_name": "herb_aria", "confidence": 0.95}]
    )
    assert out["should_capture"] is False
    # False negative: manual F8 with no confident detection.
    out = loop.evaluate_frame_for_active_learning(frame, [], manual_f8_pressed=True)
    assert out["false_negative_mining"] is True and out["should_capture"] is True
    # Hard negative: extraction failure text.
    out = loop.evaluate_frame_for_active_learning(
        frame, [], status_text="Essence Extraction failed"
    )
    assert out["hard_negative_mining"] is True and out["is_background_negative"] is True


def test_health_summary_fresh_loop_is_ready():
    assert _loop().get_active_learning_health_summary() == {"active_learning_ready": True}


# ------------------------------------------------------- I3: shared extraction
# -------------------------------------------------------


def test_priority_standard_fixtures_unchanged():
    """Values measured on the pre-D38 code; the dedup must not move them."""
    loop = _loop()
    assert loop.compute_sample_priority([{"class_name": "herb_aria", "confidence": 0.9}]) == 1.0
    assert loop.compute_sample_priority([{"class_name": "gem_crystal", "confidence": 0.9}]) == 2.9212
    assert loop.compute_sample_priority([{"class_name": "ore_orichalcum", "confidence": 0.5}]) == 2.8232
    assert loop.compute_sample_priority([{"class_id": 4, "confidence": 0.9}]) == 2.9212
    assert loop.compute_sample_priority([{"confidence": 0.9}]) == 1.0
    assert loop.compute_sample_priority([]) == 1.4


def test_priority_honours_cls_and_category_keys():
    """Previously {"cls": 4} / {"category": ...} fell through to herb_aria (1.0)."""
    loop = _loop()
    assert loop.compute_sample_priority([{"cls": 4, "confidence": 0.9}]) == 2.9212
    assert loop.compute_sample_priority([{"category": "gem_crystal", "confidence": 0.9}]) == 2.9212
    assert ActiveLearningLoop._extract_annotation_class_keys([{"cls": 4}]) == ["gem_crystal"]
    assert ActiveLearningLoop._extract_annotation_class_keys(
        [{"category": "ore_orichalcum"}, {"class_name": "herb_aria"}, "junk", None]
    ) == ["ore_orichalcum", "herb_aria"]


def test_priority_extraction_matches_legacy_key_shapes():
    """The shared helper must agree with the old inline extraction for
    every key shape the old code honoured (class_name / material_name /
    name / label / class_id / no key)."""
    loop = _loop()
    cases = [
        ({"class_name": "gem_crystal"}, 2.9212),
        ({"material_name": "gem_crystal"}, 2.9212),
        ({"name": "gem_crystal"}, 2.9212),
        ({"label": "gem_crystal"}, 2.9212),
        ({"class_id": 4}, 2.9212),
        ({"class_id": 0}, 1.0),
        # STALE-EXPECTATION FIX (R13b; G2-03 commit 9ce6a97, S-5501): D38
        # wrote this case when "sapphire" still mapped to the
        # non-canonical "gem_sapphire" (count 0 -> max priority 3.0).
        # G2-03 canonicalized the alias table ("sapphire" -> "gem_crystal"),
        # so a Sapphire sample now scores gem_crystal's rarity (30 samples
        # vs max 500 -> 2.9212, same as {"class_name": "gem_crystal"}).
        # Production behaviour is correct; the pinned 3.0 was the old bug.
        ({"material_name": "Sapphire"}, 2.9212),
        ({}, 1.0),
    ]
    for ann, expected in cases:
        got = loop.compute_sample_priority([{**ann, "confidence": 0.9}])
        assert got == expected, (ann, got, expected)


# ---------------------------------------------------------- I2: deque thumbnail
# ----------------------------------------------------------


def _harvest_flow(recorder: ShadowGameplayRecorder, big_frame: np.ndarray) -> None:
    t = 1000.0
    for _ in range(5):
        for _ in range(30):
            recorder.observe_frame(big_frame, timestamp=t, user_harvested=False, sync_label=True)
            t += 0.1
        recorder.observe_frame(
            big_frame, timestamp=t, user_harvested=True,
            material_hint="herb_aria", sync_label=True,
        )
        t += 1.5


def test_recorded_samples_keep_thumbnail_not_full_frame():
    recorder = ShadowGameplayRecorder(background_worker=False)
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    big[400:700, 800:1100] = (60, 180, 90)  # coloured patch so auto-label finds evidence
    _harvest_flow(recorder, big)
    samples = recorder.get_recorded_samples()
    assert len(samples) == 5
    retained = sum(s["frame"].nbytes for s in samples if s["frame"] is not None)
    assert retained < 5 * 1024 * 1024, f"deque still holds full frames: {retained} bytes"
    for sample in samples:
        assert sample["frame"] is not None
        assert max(sample["frame"].shape[0], sample["frame"].shape[1]) <= 320
        assert sample["frame_shape"] == (1080, 1920, 3)  # original dims preserved
    recorder.stop_background_worker()


def test_callback_and_returned_record_still_get_full_frame():
    seen: list = []
    recorder = ShadowGameplayRecorder(
        background_worker=False, on_sample_recorded=lambda rec: seen.append(rec)
    )
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    big[400:700, 800:1100] = (60, 180, 90)
    _harvest_flow(recorder, big)
    assert seen, "callback never fired"
    assert all(s["frame"] is not None and s["frame"].shape == (1080, 1920, 3) for s in seen)
    recorder.stop_background_worker()


def test_shadow_mode_stays_passive():
    recorder = ShadowGameplayRecorder(background_worker=False)
    assert recorder.can_send_input() is False
    assert recorder.assert_passive_only() is True
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = recorder.observe_frame(frame, timestamp=1.0, user_harvested=True, sync_label=True)
    assert out["bot_key_presses_sent"] == 0
    assert recorder.get_stats()["bot_key_presses_sent"] == 0
    recorder.stop_background_worker()


# ---------------------------------------------------------------- P1-14: no
# blind legacy translation on the live path (K-36) + no false F4 classes
# ----------------------------------------------------------------


def test_p1_14_bare_ids_are_canonical_no_legacy_translation():
    """P1-14 (K-36): a bare id is canonical by definition.

    The live path must NOT run translate_legacy_yolo_class_id on it:
    canonical 6 is herb_edana (not a gather crest -> 13), 18 is
    target_hp_bar (not hp_potion -> 22), 22 is hp_potion (not
    buff_scroll -> 24). Out-of-range ids are unknown (-1).
    """
    from dataset_factory.active_learning_loop import (
        normalize_material_class_key,
        resolve_class_id,
    )
    from dataset_factory.class_taxonomy import translate_legacy_yolo_class_id

    # The migration helper itself is unchanged — only its live use is gone.
    assert translate_legacy_yolo_class_id(6) == 13
    assert translate_legacy_yolo_class_id(18) == 22

    assert resolve_class_id(6) == 6
    assert resolve_class_id(7) == 7
    assert resolve_class_id(18) == 18
    assert resolve_class_id(22) == 22
    assert resolve_class_id(0) == 0
    assert resolve_class_id("6") == 6  # digit strings are ids too
    assert resolve_class_id(27) == -1
    assert resolve_class_id(-1) == -1

    # Name-first resolution: 6 names herb_edana in the 12-class map
    # (legacy translation would have looked up 13 -> no name at all).
    assert normalize_material_class_key(6) == "herb_edana"
    assert normalize_material_class_key(7) == "herb_lumina"
    assert normalize_material_class_key("6") == "herb_edana"


def test_p1_14_no_false_specific_class_for_generic_ores_and_names():
    """P1-14 (F4): generic ores / unknown names resolve to NO specific class.

    21 ores are not orichalcum, bare "berry" is not mela, odyle is not a
    gem — they must fall through to unknown (-1), never to a guessed
    ore/herb/gem class.
    """
    from dataset_factory.active_learning_loop import resolve_class_id

    for name in (
        "Iron Ore",
        "Copper Ore",
        "Silver Ore",
        "Gold Ore",
        "Titanium Ore",
        "ore_iron_silver",
        "berry",
        "Odyle",
        "Lura",
    ):
        assert resolve_class_id(name) == -1, name
