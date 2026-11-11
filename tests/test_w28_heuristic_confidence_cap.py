# -*- coding: utf-8 -*-
"""W28 unit tests: UNVERIFIED_HEURISTIC_SOURCES confidence cap (<= 0.68) and trusted=False."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.label_trust import (
    UNVERIFIED_HEURISTIC_SOURCES,
    _normalize_requeue_box,
    calibrate_box_confidence_and_trust,
    is_trusted_annotation,
    merge_label_observation,
)


@pytest.mark.parametrize(
    "source,class_id,class_name,cx,cy,w,h,confidence,stem",
    [
        ("chromatic_salient_3d_resource", 0, "herb_aria", 0.5, 0.5, 0.2, 0.2, 0.96, "sample_frame"),
        ("visual_ground_path_segmenter", 15, "path_ground", 0.5, 0.75, 0.4, 0.3, 0.92, "learn_paths_01"),
        ("hud_skill_slot", 13, "skill_slot", 0.45, 0.92, 0.05, 0.06, 0.89, "hud_frame_01"),
        ("legacy_auto", 3, "ore_orichalcum", 0.4, 0.6, 0.15, 0.15, 0.95, "mine_frame_02"),
        ("dataset_requeue", 1, "herb_mela", 0.35, 0.55, 0.12, 0.14, 0.91, "requeue_frame"),
    ],
)
def test_heuristic_source_capped_and_untrusted(
    source: str,
    class_id: int,
    class_name: str,
    cx: float,
    cy: float,
    w: float,
    h: float,
    confidence: float,
    stem: str,
) -> None:
    """Each UNVERIFIED_HEURISTIC_SOURCES entry: raw confidence above the cap
    is cut to 0.68 and the box stays trusted=False / weak=True.

    Parametrized over the five sources (previously five near-identical test
    functions with the same assertions); the dataset_requeue case also covers
    the _normalize_requeue_box path.
    """
    assert source in UNVERIFIED_HEURISTIC_SOURCES
    raw_box = {
        "class_id": class_id,
        "class_name": class_name,
        "cx": cx,
        "cy": cy,
        "w": w,
        "h": h,
        "confidence": confidence,
        "trusted": True,
        "weak": False,
        "source": source,
    }
    assert is_trusted_annotation(raw_box) is False
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem=stem)
    assert calibrated["confidence"] <= 0.68
    assert calibrated["confidence"] == 0.68
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True

    if source == "dataset_requeue":
        requeued = _normalize_requeue_box(
            {
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
                "w": w,
                "h": h,
                "confidence": 0.95,
                "has_explicit_conf": True,
                "source": source,
            },
            stem=stem,
        )
        assert requeued is not None
        assert requeued["confidence"] <= 0.68
        assert requeued["trusted"] is False
        assert requeued["weak"] is True


def test_all_five_heuristic_sources_never_trusted_without_human_approval() -> None:
    target_sources = (
        "chromatic_salient_3d_resource",
        "visual_ground_path_segmenter",
        "hud_skill_slot",
        "legacy_auto",
        "dataset_requeue",
    )
    for src in target_sources:
        assert src in UNVERIFIED_HEURISTIC_SOURCES        # Sub-0.68 confidence is preserved while >= 0.68 is capped at 0.68
        low_box = calibrate_box_confidence_and_trust(
            {
                "class_id": 2,
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.52,
                "source": src,
                "trusted": True,
            }
        )
        assert low_box["confidence"] == 0.52
        assert low_box["confidence"] <= 0.68
        assert low_box["trusted"] is False
        assert low_box["weak"] is True

        high_box = {
            "class_id": 2,
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.1,
            "h": 0.1,
            "confidence": 0.99,
            "source": src,
            "trusted": True,
        }
        cal_high = calibrate_box_confidence_and_trust(dict(high_box))
        assert cal_high["confidence"] <= 0.68
        assert cal_high["trusted"] is False
        assert cal_high["weak"] is True

        trusted_list, weak_list, _ = merge_label_observation(
            [],
            [],
            [cal_high],
            has_seed=True,
            require_human_confirmation=True,
        )
        assert len(trusted_list) == 0
        assert len(weak_list) == 1
        assert weak_list[0]["trusted"] is False
        assert weak_list[0]["confidence"] <= 0.68


def test_human_approved_box_is_exempt_from_heuristic_cap() -> None:
    """Onaylı kutu cap'ten muaftır: human_approved işareti taşıyan kutu,
    kaynağı UNVERIFIED_HEURISTIC_SOURCES'ta olsa bile 0.68'e çekilmez, güveni
    korunur (0.96) ve trusted=True / weak=False kalır.
    Merge-fix (O8): D39 calibrate'ı daralttı — çıplak "approved" bayrağı
    (pipeline işareti, insan kararı değil) muafiyet sağlamaz; yalnızca
    hand_label/human_approved cap'i aşar (D38/F-0828 ile aynı ilke).
    P1-12 (K-34): çıplak "approved" is_trusted_annotation'da da insan
    sayılmaz — o dalda trusted=False beklenir."""
    target_sources = (
        "chromatic_salient_3d_resource",
        "visual_ground_path_segmenter",
        "hud_skill_slot",
        "legacy_auto",
        "dataset_requeue",
    )
    for src in target_sources:
        assert src in UNVERIFIED_HEURISTIC_SOURCES
        for approval_flag in ("human_approved", "approved"):
            raw_box = {
                "class_id": 2,
                "class_name": "herb_targena",
                "cx": 0.5,
                "cy": 0.5,
                "w": 0.1,
                "h": 0.1,
                "confidence": 0.96,
                "trusted": True,
                "weak": False,
                "source": src,
                approval_flag: True,
            }
            # P1-12 (K-34): only an explicit human decision trusts; a bare
            # "approved" pipeline flag does not.
            if approval_flag == "human_approved":
                assert is_trusted_annotation(raw_box) is True
            else:
                assert is_trusted_annotation(raw_box) is False
            calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem="approved_frame")
            if approval_flag == "human_approved":
                assert calibrated["confidence"] == 0.96
                assert calibrated["confidence"] > 0.68
                assert calibrated["trusted"] is True
                assert calibrated["weak"] is False
            else:
                # Bare "approved" is a pipeline flag, not a human decision:
                # the heuristic cap still applies.
                assert calibrated["confidence"] == 0.68
                assert calibrated["trusted"] is False
                assert calibrated["weak"] is True

            # Aynı kutu onaysızken cap'e takılır — muafiyetin onaydan geldiği kanıtlanır.
            unapproved = dict(raw_box)
            unapproved[approval_flag] = False
            assert is_trusted_annotation(unapproved) is False
            capped = calibrate_box_confidence_and_trust(dict(unapproved), stem="approved_frame")
            assert capped["confidence"] == 0.68
            assert capped["trusted"] is False
            assert capped["weak"] is True
