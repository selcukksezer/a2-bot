# -*- coding: utf-8 -*-
"""W28 unit tests: UNVERIFIED_HEURISTIC_SOURCES confidence cap (<= 0.68) and trusted=False."""

from __future__ import annotations

import sys
from pathlib import Path

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


def test_chromatic_salient_3d_resource_capped_and_untrusted() -> None:
    assert "chromatic_salient_3d_resource" in UNVERIFIED_HEURISTIC_SOURCES
    raw_box = {
        "class_id": 0,
        "class_name": "herb_aria",
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.96,
        "trusted": True,
        "weak": False,
        "source": "chromatic_salient_3d_resource",
    }
    assert is_trusted_annotation(raw_box) is False
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem="sample_frame")
    assert calibrated["confidence"] <= 0.68
    assert calibrated["confidence"] == 0.68
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_visual_ground_path_segmenter_capped_and_untrusted() -> None:
    assert "visual_ground_path_segmenter" in UNVERIFIED_HEURISTIC_SOURCES
    raw_box = {
        "class_id": 15,
        "class_name": "path_ground",
        "cx": 0.5,
        "cy": 0.75,
        "w": 0.4,
        "h": 0.3,
        "confidence": 0.92,
        "trusted": True,
        "weak": False,
        "source": "visual_ground_path_segmenter",
    }
    assert is_trusted_annotation(raw_box) is False
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem="learn_paths_01")
    assert calibrated["confidence"] <= 0.68
    assert calibrated["confidence"] == 0.68
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_hud_skill_slot_capped_and_untrusted() -> None:
    assert "hud_skill_slot" in UNVERIFIED_HEURISTIC_SOURCES
    raw_box = {
        "class_id": 13,
        "class_name": "skill_slot",
        "cx": 0.45,
        "cy": 0.92,
        "w": 0.05,
        "h": 0.06,
        "confidence": 0.89,
        "trusted": True,
        "weak": False,
        "source": "hud_skill_slot",
    }
    assert is_trusted_annotation(raw_box) is False
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem="hud_frame_01")
    assert calibrated["confidence"] <= 0.68
    assert calibrated["confidence"] == 0.68
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_legacy_auto_capped_and_untrusted() -> None:
    assert "legacy_auto" in UNVERIFIED_HEURISTIC_SOURCES
    raw_box = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.4,
        "cy": 0.6,
        "w": 0.15,
        "h": 0.15,
        "confidence": 0.95,
        "trusted": True,
        "weak": False,
        "source": "legacy_auto",
    }
    assert is_trusted_annotation(raw_box) is False
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem="mine_frame_02")
    assert calibrated["confidence"] <= 0.68
    assert calibrated["confidence"] == 0.68
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_dataset_requeue_capped_and_untrusted() -> None:
    assert "dataset_requeue" in UNVERIFIED_HEURISTIC_SOURCES
    raw_box = {
        "class_id": 1,
        "class_name": "herb_mela",
        "cx": 0.35,
        "cy": 0.55,
        "w": 0.12,
        "h": 0.14,
        "confidence": 0.91,
        "trusted": True,
        "weak": False,
        "source": "dataset_requeue",
    }
    assert is_trusted_annotation(raw_box) is False
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem="requeue_frame")
    assert calibrated["confidence"] <= 0.68
    assert calibrated["confidence"] == 0.68
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True

    requeued = _normalize_requeue_box(
        {
            "class_id": 1,
            "cx": 0.35,
            "cy": 0.55,
            "w": 0.12,
            "h": 0.14,
            "confidence": 0.95,
            "has_explicit_conf": True,
            "source": "dataset_requeue",
        },
        stem="requeue_frame",
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
        assert src in UNVERIFIED_HEURISTIC_SOURCES
        # Sub-0.68 confidence is preserved while >= 0.68 is capped at 0.68
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
