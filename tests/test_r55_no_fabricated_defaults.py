# -*- coding: utf-8 -*-
"""R55 regression tests (S-2001, S-2002): no fabricated defaults.

- S-2001: a class-less / coordinate-less annotation must never become a
  confident herb_aria box at the frame centre; missing class -> "" and
  missing/unresolvable class id -> -1; missing confidence -> 0.0.
- S-2002: a detection without confidence is UNKNOWN (0.0), never 1.0, so it
  must not land in high_conf_detections or suppress false-negative mining.
"""
import numpy as np

from dataset_factory.active_learning_loop import (
    ActiveLearningLoop,
    ShadowGameplayRecorder,
    normalize_material_class_key,
    resolve_class_id,
)


def test_s2001_normalize_returns_empty_for_missing_class():
    assert normalize_material_class_key(None) == ""
    assert normalize_material_class_key("") == ""
    assert normalize_material_class_key("   ") == ""
    # Explicit evidence still resolves normally.
    assert normalize_material_class_key(0) == "herb_aria"
    assert normalize_material_class_key("herb_aria") == "herb_aria"
    assert normalize_material_class_key("ore_orichalcum") == "ore_orichalcum"


def test_s2001_resolve_returns_minus_one_for_missing_or_unknown():
    assert resolve_class_id(None) == -1
    assert resolve_class_id("") == -1
    assert resolve_class_id(-1) == -1
    assert resolve_class_id("zzz_unknown_qq") == -1
    # Explicit evidence still resolves normally.
    assert resolve_class_id("herb_aria") == 0
    assert resolve_class_id("ore_orichalcum") == 3
    assert resolve_class_id(3) == 3
    assert resolve_class_id(0) == 0


def test_s2001_normalize_annotations_skips_classless_and_coordless():
    norm = ShadowGameplayRecorder._normalize_annotations
    # Empty dict: no class, no coords -> no fabricated herb_aria centre box.
    assert norm([{}], 640, 480) == []
    # Class only, no coordinates -> no invented centre box.
    assert norm([{"class_name": "herb_aria"}], 640, 480) == []
    # Coordinates but unresolvable class -> dropped, not labelled herb_aria.
    assert norm(
        [{"class_name": "zzz_unknown_qq", "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        640,
        480,
    ) == []
    # Coordinates but no class at all -> dropped.
    assert norm(
        [{"cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        640,
        480,
    ) == []


def test_s2001_normalize_annotations_missing_confidence_is_zero():
    norm = ShadowGameplayRecorder._normalize_annotations
    anns = norm(
        [{"class_name": "herb_aria", "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        640,
        480,
    )
    assert len(anns) == 1
    assert anns[0]["confidence"] == 0.0
    assert anns[0]["class_id"] == 0
    # Explicit confidence is preserved.
    anns = norm(
        [
            {
                "class_name": "herb_aria",
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.85,
            }
        ],
        640,
        480,
    )
    assert anns[0]["confidence"] == 0.85


def test_s2002_confidenceless_detection_is_not_high_conf():
    al = ActiveLearningLoop(initial_class_counts={"herb_aria": 500, "gem_crystal": 30})
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # No confidence + manual F: scanner missed the node -> false-negative mining.
    res = al.evaluate_frame_for_active_learning(
        frame=frame,
        detections=[{"class_name": "herb_aria", "cx": 0.5, "cy": 0.5}],
        manual_f_pressed=True,
    )
    assert len(res["high_confidence_detections"]) == 0
    assert len(res["low_confidence_detections"]) == 1
    assert res["low_confidence_detections"][0]["confidence"] == 0.0
    assert "false_negative_mining" in res["mining_types"]
    # A genuinely confident detection still suppresses the F trigger.
    res2 = al.evaluate_frame_for_active_learning(
        frame=frame,
        detections=[{"class_name": "herb_aria", "confidence": 0.9}],
        manual_f_pressed=True,
    )
    assert "false_negative_mining" not in res2["mining_types"]


def test_s2002_garbage_confidence_is_zero_not_one():
    al = ActiveLearningLoop()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = al.evaluate_frame_for_active_learning(
        frame=frame,
        detections=[{"class_name": "herb_aria", "confidence": "garbage"}],
    )
    assert res["low_confidence_detections"][0]["confidence"] == 0.0
    assert len(res["high_confidence_detections"]) == 0


def test_s2001_classless_annotation_does_not_inflate_priority():
    al = ActiveLearningLoop(initial_class_counts={"herb_aria": 500, "gem_crystal": 30})
    counts = {"herb_aria": 500, "gem_crystal": 30}
    # Class-less annotation: no priority (min), not "rare unseen class" ~3.0.
    assert al.compute_sample_priority(
        annotations=[{"confidence": 0.85}], class_counts=counts
    ) == 1.0
    # Rare class still gets high priority.
    assert (
        al.compute_sample_priority(
            annotations=[{"class_name": "gem_crystal", "confidence": 0.85}],
            class_counts=counts,
        )
        >= 2.5
    )
    # Unknown class is not counted as herb_aria.
    assert al.increment_class_count("") == 0
    assert "" not in al.class_counts
    assert "" not in al.update_class_counts({"": 5})
