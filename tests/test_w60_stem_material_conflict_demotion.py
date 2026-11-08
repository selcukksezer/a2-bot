# -*- coding: utf-8 -*-
"""W60 unit tests: Demotion of conflicting ore_orichalcum on boot_*_aria_* frames to <= 0.45 (weak/suspicious)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.label_trust import (
    calibrate_box_confidence_and_trust,
    is_material_conflict,
)
from dataset_factory.stem_material_validator import (
    DEMOTED_REASON_STEM_MATERIAL_CONFLICT,
    MAX_CONFLICT_CONFIDENCE,
    StemMaterialValidator,
    validate_and_demote_box,
)


def test_stem_material_validator_demotes_ore_orichalcum_on_boot_aria_stem() -> None:
    """1. StemMaterialValidator detects conflict on boot_*_aria_* vs ore_orichalcum and caps confidence <= 0.45."""
    validator = StemMaterialValidator()
    stem = "boot_001_aria_node"
    box = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.50,
        "cy": 0.55,
        "w": 0.18,
        "h": 0.22,
        "confidence": 0.94,
        "trusted": True,
        "weak": False,
    }

    assert validator.is_conflict(stem, box) is True
    result = validator.evaluate_box(stem, box, in_place=False)

    assert result.conflict is True
    assert "aria" in result.stem_materials
    assert "orichalcum" in result.box_materials
    assert result.calibrated_confidence <= MAX_CONFLICT_CONFIDENCE
    assert result.calibrated_confidence == 0.45
    assert result.trusted is False
    assert result.weak is True
    assert result.demoted_reason == DEMOTED_REASON_STEM_MATERIAL_CONFLICT
    assert result.box["confidence"] <= 0.45
    assert result.box["trusted"] is False
    assert result.box["weak"] is True


def test_calibrate_box_confidence_and_trust_demotes_ore_orichalcum_on_boot_aria_stem() -> None:
    """2. calibrate_box_confidence_and_trust demotes ore_orichalcum on boot_*_aria_* to <= 0.45 and weak=True."""
    stem = "boot_02_aria_3d_crest"
    raw_box = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.48,
        "cy": 0.52,
        "w": 0.16,
        "h": 0.20,
        "confidence": 0.91,
        "trusted": True,
        "weak": False,
        "source": "yolo_auto",
    }

    assert is_material_conflict(stem, "ore_orichalcum", 3) is True
    calibrated = calibrate_box_confidence_and_trust(dict(raw_box), stem=stem)

    assert calibrated["confidence"] <= 0.45
    assert calibrated["confidence"] == 0.45
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_sub_045_confidence_preserved_while_marked_suspicious_on_boot_aria_conflict() -> None:
    """3. When initial confidence is already <= 0.45, both validators keep it <= 0.45 and enforce weak=True, trusted=False."""
    stem = "boot_03_aria_gather"
    low_conf_box = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.42,
        "cy": 0.60,
        "w": 0.14,
        "h": 0.14,
        "confidence": 0.37,
        "trusted": True,
        "weak": False,
    }

    validated_by_stem = validate_and_demote_box(stem, dict(low_conf_box), in_place=False)
    assert validated_by_stem["confidence"] == 0.37
    assert validated_by_stem["confidence"] <= 0.45
    assert validated_by_stem["trusted"] is False
    assert validated_by_stem["weak"] is True
    assert validated_by_stem["demoted_reason"] == DEMOTED_REASON_STEM_MATERIAL_CONFLICT

    calibrated = calibrate_box_confidence_and_trust(dict(low_conf_box), stem=stem)
    assert calibrated["confidence"] == 0.37
    assert calibrated["confidence"] <= 0.45
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_review_payload_moves_conflicting_orichalcum_from_trusted_to_weak_on_boot_aria() -> None:
    """4. StemMaterialValidator.validate_review_payload moves ore_orichalcum from trusted to weak on boot_*_aria_*."""
    validator = StemMaterialValidator()
    stem = "boot_04_aria_field"
    aria_box = calibrate_box_confidence_and_trust(
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "cx": 0.50,
            "cy": 0.50,
            "w": 0.20,
            "h": 0.20,
            "confidence": 0.92,
            "source": "yolo_auto",
        },
        stem=stem,
    )
    orichalcum_box = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.30,
        "cy": 0.40,
        "w": 0.15,
        "h": 0.15,
        "confidence": 0.89,
        "trusted": True,
        "weak": False,
    }

    payload = {
        "stem": stem,
        "trusted": [aria_box, orichalcum_box],
        "weak": [],
    }

    updated = validator.validate_review_payload(payload, stem=stem, in_place=False)
    assert len(updated["trusted"]) == 1
    assert updated["trusted"][0]["class_name"] == "herb_aria"
    assert updated["trusted"][0]["trusted"] is True
    assert updated["trusted"][0]["confidence"] == 0.92

    assert len(updated["weak"]) == 1
    demoted = updated["weak"][0]
    assert demoted["class_name"] == "ore_orichalcum"
    assert demoted["confidence"] <= 0.45
    assert demoted["confidence"] == 0.45
    assert demoted["trusted"] is False
    assert demoted["weak"] is True
    assert demoted["demoted_reason"] == DEMOTED_REASON_STEM_MATERIAL_CONFLICT


def test_class_id_lookup_and_embedded_boot_aria_stem_demoted_in_both_validators() -> None:
    """5. Box with class_id=3 (ore_orichalcum) and embedded boot_*_aria_* stem is demoted by both validators."""
    validator = StemMaterialValidator()
    embedded_box = {
        "stem": "boot_2026_aria_hill.jpg",
        "class_id": 3,
        "cx": 0.55,
        "cy": 0.48,
        "w": 0.19,
        "h": 0.21,
        "confidence": 0.97,
        "conf": 0.97,
        "trusted": True,
        "weak": False,
    }

    cal_box = calibrate_box_confidence_and_trust(dict(embedded_box))
    assert cal_box["confidence"] <= 0.45
    assert cal_box["confidence"] == 0.45
    assert cal_box["trusted"] is False
    assert cal_box["weak"] is True

    val_box = validator.validate_box(dict(embedded_box), in_place=False)
    assert val_box["confidence"] <= 0.45
    assert val_box["confidence"] == 0.45
    assert val_box["conf"] <= 0.45
    assert val_box["trusted"] is False
    assert val_box["weak"] is True
    assert val_box["demoted_reason"] == DEMOTED_REASON_STEM_MATERIAL_CONFLICT
