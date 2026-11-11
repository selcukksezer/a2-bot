# -*- coding: utf-8 -*-
"""Unit tests for tools/recalibrate_dataset_trust.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from tools.recalibrate_dataset_trust import (
    UNVERIFIED_HEURISTIC_SOURCES,
    calibrate_single_box,
    is_conflicting_material,
    is_essence_learning_stem,
    is_path_learning_stem,
    is_skill_learning_stem,
    recalibrate_entire_dataset_trust,
)


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Redirect CWD-relative production writes into a throwaway dir.

    ``recalibrate_entire_dataset_trust`` calls ``mark_model_stale``, which
    writes ``models/stale_manifest.json`` relative to the current working
    directory, so a run from the repo root rewrote a repo file. All dataset
    paths in this file are absolute, so changing CWD is safe here.
    """
    monkeypatch.chdir(tmp_path)


def test_protected_box_immunity():
    """Hand labels and human approved boxes must remain trusted.

    F-1212: protection keeps the box trusted, but the recorded confidence
    is preserved as-is — a 0.70 protected box stays 0.70 and is NOT
    inflated to 0.90.
    """
    box = {
        "class_id": 0,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.70,
        "source": "hand_label",
        "hand_label": True,
        "trusted": True,
        "weak": False,
    }
    calibrated, action = calibrate_single_box(box, stem="any_stem", emin_threshold=0.85)
    assert calibrated is not None
    assert action == "preserved_trusted"
    assert calibrated["trusted"] is True
    assert calibrated["weak"] is False
    assert calibrated["confidence"] == 0.70


def test_protected_box_without_confidence_not_inflated():
    """F-1212: a protected box with no recorded confidence stays 0.0."""
    box = {
        "class_id": 0,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "source": "hand_label",
        "hand_label": True,
        "trusted": True,
        "weak": False,
    }
    calibrated, action = calibrate_single_box(box, stem="any_stem", emin_threshold=0.85)
    assert calibrated is not None
    assert action == "preserved_trusted"
    assert calibrated["trusted"] is True
    assert calibrated["weak"] is False
    assert calibrated["confidence"] == 0.0


def test_negative_class_id_purged():
    """S-4801: cid=-1 (missing class_id default) must be purged, never trusted.

    Even with high confidence and a verified source it must not reach
    Rule 8 as preserved_trusted, and a negative class_id from a TXT line
    must not be salvaged into the review JSON / label TXT.
    """
    box = {
        "class_id": -1,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "confidence": 0.95,
        "source": "material_scanner_minimap_sprout",
        "trusted": True,
    }
    calibrated, action = calibrate_single_box(box, stem="any_stem", emin_threshold=0.85)
    assert calibrated is None
    assert action == "purged_unknown_class"

    # Missing class_id key defaults to -1 -> same purge.
    box_no_cid = dict(box)
    del box_no_cid["class_id"]
    calibrated2, action2 = calibrate_single_box(box_no_cid, stem="any_stem", emin_threshold=0.85)
    assert calibrated2 is None
    assert action2 == "purged_unknown_class"

    # TXT salvage: a "-1 ..." line must not be written back.
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "review" / "train").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "images" / "train").mkdir(parents=True)
        stem = "boot_100_aria_gather"
        (root / "review" / "train" / f"{stem}.json").write_text(
            json.dumps({"stem": stem, "split": "train", "trusted": [], "weak": []}),
            encoding="utf-8",
        )
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            "-1 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n",
            encoding="utf-8",
        )
        report = recalibrate_entire_dataset_trust(dataset_dir=root, dry_run=False)
        assert report["metrics"]["purged_false_boxes"] == 0  # skipped at TXT parse, not counted as purge
        updated = json.loads((root / "review" / "train" / f"{stem}.json").read_text(encoding="utf-8"))
        all_cids = [b["class_id"] for b in updated["trusted"] + updated["weak"]]
        assert -1 not in all_cids
        assert all_cids == [0]
        # The salvaged -1 line must not be written back to the label TXT
        # (train_txt_boxes only emits trusted boxes, so the weak cid=0 box
        # is not emitted either — the file must simply contain no "-1" line).
        assert "-1" not in (root / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8")


def test_stem_gating_purges_path_on_gathering():
    """Ground path (15) and obstacle (14) must be purged on non-path frames."""
    box = {
        "class_id": 15,
        "cx": 0.5,
        "cy": 0.7,
        "w": 0.4,
        "h": 0.3,
        "confidence": 0.92,
        "source": "visual_ground_path_segmenter",
        "trusted": True,
    }
    # On gathering frame -> PURGED
    calibrated, action = calibrate_single_box(box, stem="boot_478467_000_aria_3d_crest", purge_synthetic=True)
    assert calibrated is None
    assert action == "purged_out_of_context_path_obstacle"

    # On path learning frame -> PRESERVED AS WEAK (because source is heuristic)
    calibrated_path, action_path = calibrate_single_box(box, stem="learn_paths_1793975849145_424", purge_synthetic=True)
    assert calibrated_path is not None
    assert calibrated_path["class_id"] == 15
    assert calibrated_path["weak"] is True
    assert calibrated_path["trusted"] is False
    assert calibrated_path["confidence"] <= 0.68


def test_stem_gating_purges_hud_on_gathering():
    """HUD skills (13) and bars (16, 17) must be purged on gathering frames."""
    box = {
        "class_id": 13,
        "cx": 0.7,
        "cy": 0.9,
        "w": 0.02,
        "h": 0.03,
        "confidence": 0.88,
        "source": "hud_skill_slot",
        "trusted": True,
    }
    # On gathering frame -> PURGED
    calibrated, action = calibrate_single_box(box, stem="boot_478467_000_aria_3d_crest", purge_synthetic=True)
    assert calibrated is None
    assert action == "purged_out_of_context_hud_skill"

    # On skill learning frame -> PRESERVED AS WEAK (because source is heuristic)
    calibrated_skill, action_skill = calibrate_single_box(box, stem="learn_skills_1794032002457_474", purge_synthetic=True)
    assert calibrated_skill is not None
    assert calibrated_skill["class_id"] == 13
    assert calibrated_skill["weak"] is True
    assert calibrated_skill["trusted"] is False
    assert calibrated_skill["confidence"] <= 0.68


def test_stem_gating_purges_essence_on_gathering():
    """Essence classes (>= 22, e.g. 98) must be purged on non-essence frames."""
    box = {
        "class_id": 98,
        "cx": 0.4,
        "cy": 0.5,
        "w": 0.1,
        "h": 0.1,
        "confidence": 0.95,
        "source": "template_match",
        "trusted": True,
    }
    calibrated, action = calibrate_single_box(box, stem="boot_478467_000_aria_3d_crest", purge_synthetic=True)
    assert calibrated is None
    assert action == "purged_out_of_context_essence_on_gather_frame"


def test_unverified_heuristics_demoted():
    """Sources in UNVERIFIED_HEURISTIC_SOURCES must be capped at <= 0.68 and marked weak."""
    box = {
        "class_id": 2,
        "cx": 0.4,
        "cy": 0.5,
        "w": 0.1,
        "h": 0.1,
        "confidence": 0.98,
        "source": "chromatic_salient_3d_resource",
        "trusted": True,
    }
    calibrated, action = calibrate_single_box(box, stem="unknown_stem")
    assert calibrated is not None
    assert action == "demoted_to_weak"
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True
    assert calibrated["confidence"] <= 0.68


def test_material_conflict_detection():
    """Conflicting materials on known stems must be capped at <= 0.45 and marked weak."""
    # Aria stem (class 0) with Ore box (class 3)
    assert is_conflicting_material("boot_123_aria_gather", 3) is True
    assert is_conflicting_material("boot_123_aria_gather", 0) is False

    box = {
        "class_id": 3,
        "cx": 0.4,
        "cy": 0.5,
        "w": 0.1,
        "h": 0.1,
        "confidence": 0.95,
        "source": "some_detector",
        "trusted": True,
    }
    calibrated, action = calibrate_single_box(box, stem="boot_123_aria_gather")
    assert calibrated is not None
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True
    assert calibrated["confidence"] <= 0.45


def test_legacy_txt_box_calibration():
    """Boxes from legacy TXT without explicit confidence get 0.58 and weak."""
    box = {
        "class_id": 0,
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.1,
        "h": 0.1,
        "confidence": 0.0,
        "legacy_auto": True,
        "has_explicit_conf": False,
        "source": "legacy_txt",
        "trusted": False,
        "weak": True,
    }
    calibrated, action = calibrate_single_box(box, stem="boot_123_aria_gather")
    assert calibrated is not None
    assert calibrated["confidence"] == 0.58
    assert calibrated["trusted"] is False
    assert calibrated["weak"] is True


def test_verified_seed_confidence_threshold():
    """Genuine template seeds with >= 0.85 stay trusted; < 0.85 become weak."""
    high_seed = {
        "class_id": 9,
        "cx": 0.95,
        "cy": 0.15,
        "w": 0.01,
        "h": 0.01,
        "confidence": 0.92,
        "source": "material_scanner_minimap_sprout",
        "trusted": True,
    }
    cal_high, act_high = calibrate_single_box(high_seed, emin_threshold=0.85)
    assert cal_high is not None
    assert act_high == "preserved_trusted"
    assert cal_high["trusted"] is True
    assert cal_high["weak"] is False

    low_seed = {
        "class_id": 9,
        "cx": 0.95,
        "cy": 0.15,
        "w": 0.01,
        "h": 0.01,
        "confidence": 0.72,
        "source": "material_scanner_minimap_sprout",
        "trusted": True,
    }
    cal_low, act_low = calibrate_single_box(low_seed, emin_threshold=0.85)
    assert cal_low is not None
    assert act_low == "demoted_to_weak"
    assert cal_low["trusted"] is False
    assert cal_low["weak"] is True


def test_recalibrate_entire_dataset_trust_end_to_end():
    """End-to-end execution on mock dataset directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "review" / "train").mkdir(parents=True)
        (root / "review" / "val").mkdir(parents=True)
        (root / "labels" / "train").mkdir(parents=True)
        (root / "labels" / "val").mkdir(parents=True)
        (root / "images" / "train").mkdir(parents=True)
        (root / "images" / "val").mkdir(parents=True)

        stem = "boot_100_aria_gather"
        # Create a REAL decodable JPEG fixture (not 1-byte garbage): the E2E
        # dataset image must be a genuine image file that actually decodes.
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:] = (60, 120, 200)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok, "fixture JPEG must encode"
        image_path = root / "images" / "train" / f"{stem}.jpg"
        image_path.write_bytes(buf.tobytes())
        decoded = cv2.imread(str(image_path))
        assert decoded is not None, "fixture image must decode as a real JPEG"
        assert decoded.shape == (64, 64, 3)

        # Create review json with 1 genuine sprout, 1 out-of-context path_ground, 1 heuristic targena
        review_payload = {
            "stem": stem,
            "split": "train",
            "trusted": [
                {
                    "class_id": 9,
                    "cx": 0.95,
                    "cy": 0.15,
                    "w": 0.01,
                    "h": 0.01,
                    "confidence": 0.95,
                    "source": "material_scanner_minimap_sprout",
                    "trusted": True,
                },
                {
                    "class_id": 15,
                    "cx": 0.5,
                    "cy": 0.7,
                    "w": 0.4,
                    "h": 0.3,
                    "confidence": 0.92,
                    "source": "visual_ground_path_segmenter",
                    "trusted": True,
                },
                {
                    "class_id": 2,
                    "cx": 0.3,
                    "cy": 0.4,
                    "w": 0.1,
                    "h": 0.1,
                    "confidence": 0.98,
                    "source": "chromatic_salient_3d_resource",
                    "trusted": True,
                },
            ],
            "weak": [],
        }
        (root / "review" / "train" / f"{stem}.json").write_text(
            json.dumps(review_payload), encoding="utf-8"
        )
        (root / "labels" / "train" / f"{stem}.txt").write_text(
            "9 0.95 0.15 0.01 0.01\n15 0.5 0.7 0.4 0.3\n2 0.3 0.4 0.1 0.1\n",
            encoding="utf-8",
        )

        # Run recalibrate_entire_dataset_trust
        report = recalibrate_entire_dataset_trust(dataset_dir=root, dry_run=False)

        assert report["ok"] is True
        assert report["total_images_processed"] == 1
        m = report["metrics"]
        assert m["previous_trusted_boxes"] == 3
        assert m["new_trusted_boxes"] == 1       # only sprout
        assert m["new_weak_boxes"] == 1          # demoted targena
        assert m["purged_false_boxes"] == 1      # purged path_ground
        assert m["demoted_boxes"] == 1           # demoted targena
        assert m["preserved_verified_boxes"] == 1 # preserved sprout

        # Inspect rewritten review json
        updated_rev = json.loads((root / "review" / "train" / f"{stem}.json").read_text(encoding="utf-8"))
        assert len(updated_rev["trusted"]) == 1
        assert updated_rev["trusted"][0]["class_id"] == 9
        assert len(updated_rev["weak"]) == 1
        assert updated_rev["weak"][0]["class_id"] == 2
        assert updated_rev["weak"][0]["confidence"] <= 0.45  # Conflicting with aria stem!

        # Inspect rewritten label txt
        lbl_lines = (root / "labels" / "train" / f"{stem}.txt").read_text(encoding="utf-8").strip().splitlines()
        assert len(lbl_lines) == 1
        assert lbl_lines[0].startswith("9 ")
