# -*- coding: utf-8 -*-
"""Comprehensive verification test suite for Dataset Validator,
Review JSON <-> Labels TXT Atomic Synchronization (L16),
and Box Size Bounds & Out-of-Bounds Rejection (L17).
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from dataset_factory.dataset_validator import (
    MIN_BOX_DIM_BOUND,
    MAX_BOX_DIM_BOUND,
    check_box_size_bounds,
    validate_label_box,
    sanitize_boxes,
    sanitize_dataset_boxes,
    atomic_write_text,
    atomic_write_json,
    verify_and_sync_review_txt_pair,
    sync_all_review_txt_pairs,
    validate_yolo_file,
    validate_dataset,
)


class TestBoxValidationResultAndLabelValidation:
    """Test validate_label_box, check_box_size_bounds, and BoxValidationResult behavior."""

    def test_micro_noise_rejection(self) -> None:
        """Boxes with w < 0.004 or h < 0.004 must be rejected as micro noise."""
        res_w = validate_label_box(0.5, 0.5, 0.0039, 0.1)
        assert not res_w
        assert res_w.is_valid is False
        assert "micro noise" in res_w.reason.lower() or "mikro" in res_w.reason.lower()

        res_h = validate_label_box(0.5, 0.5, 0.1, 0.001)
        assert not res_h
        assert res_h.is_valid is False

    def test_massive_box_rejection(self) -> None:
        """Boxes with w > 0.65 or h > 0.65 must be rejected as screen-filling errors."""
        res_w = validate_label_box(0.5, 0.5, 0.651, 0.2)
        assert not res_w
        assert res_w.is_valid is False
        assert "massive" in res_w.reason.lower() or "aşırı büyük" in res_w.reason.lower()

        res_h = validate_label_box(0.5, 0.5, 0.3, 0.70)
        assert not res_h
        assert res_h.is_valid is False

    def test_exact_boundary_thresholds(self) -> None:
        """Exact min (0.004) and max (0.65) boundaries should be accepted."""
        res_min = validate_label_box(0.5, 0.5, 0.004, 0.004)
        assert res_min
        assert res_min.is_valid is True

        res_max = validate_label_box(0.5, 0.5, 0.65, 0.65)
        assert res_max
        assert res_max.is_valid is True

    def test_zero_and_negative_dimensions(self) -> None:
        """Zero area or negative dimensions must be rejected."""
        assert not validate_label_box(0.5, 0.5, 0.0, 0.1)
        assert not validate_label_box(0.5, 0.5, 0.1, 0.0)
        assert not validate_label_box(0.5, 0.5, -0.05, 0.1)
        assert not validate_label_box(0.5, 0.5, 0.1, -0.02)

    def test_out_of_bounds_center_coords(self) -> None:
        """Center coordinates outside [0.0, 1.0] must be rejected."""
        assert not validate_label_box(-0.01, 0.5, 0.1, 0.1)
        assert not validate_label_box(1.02, 0.5, 0.1, 0.1)
        assert not validate_label_box(0.5, -0.05, 0.1, 0.1)
        assert not validate_label_box(0.5, 1.05, 0.1, 0.1)

    def test_boundary_extents_strict(self) -> None:
        """Box corners extending beyond [0.0, 1.0] must be rejected when strict_bounds=True."""
        # cx=0.98, w=0.1 -> x2 = 1.03 > 1.0
        res = validate_label_box(0.98, 0.5, 0.1, 0.1, strict_bounds=True)
        assert not res
        assert res.is_valid is False
        assert "boundaries extend" in res.reason.lower() or "sınır dışı" in res.reason.lower()

    def test_tuple_unpacking_and_boolean_truthiness(self) -> None:
        """BoxValidationResult behaves as both a 2-tuple (bool, str) and a boolean."""
        # Valid
        res_valid = validate_label_box(0.5, 0.5, 0.1, 0.1)
        is_ok, msg = res_valid
        assert is_ok is True
        assert msg == ""
        assert bool(res_valid) is True
        if not res_valid:
            pytest.fail("Valid result should evaluate to True in boolean context")

        # Invalid
        res_invalid = validate_label_box(0.5, 0.5, 0.001, 0.001)
        is_bad, bad_msg = res_invalid
        assert is_bad is False
        assert len(bad_msg) > 0
        assert bool(res_invalid) is False
        if res_invalid:
            pytest.fail("Invalid result should evaluate to False in boolean context")

    def test_dict_and_sequence_inputs(self) -> None:
        """Accepts dictionaries (both cx/cy and x_center/y_center) and sequences."""
        d1 = {"cx": 0.4, "cy": 0.4, "w": 0.15, "h": 0.15, "class_id": 2}
        assert validate_label_box(d1)

        d2 = {"x_center": 0.4, "y_center": 0.4, "width": 0.15, "height": 0.15, "cls": 2}
        assert validate_label_box(d2)

        # 4-tuple and 5-tuple
        assert validate_label_box([0.5, 0.5, 0.2, 0.2])
        assert validate_label_box([3, 0.5, 0.5, 0.2, 0.2])


class TestSanitizeDatasetBoxes:
    """Test sanitize_dataset_boxes and sanitize_boxes with L17 guardrails."""

    def test_filters_micro_and_massive_boxes(self) -> None:
        """Filters micro noise and massive boxes while preserving clean ones."""
        boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.95},
            {"class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.002, "h": 0.05},  # micro
            {"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.70, "h": 0.50},  # massive
            {"class_id": 3, "cx": 0.1, "cy": 0.1, "w": 0.0, "h": 0.1},    # zero
        ]
        clean, issues = sanitize_dataset_boxes(boxes)
        assert len(clean) == 1
        assert clean[0]["class_id"] == 0

        # Exact issue types: micro box -> micro_noise, massive box ->
        # massive_box, zero-area box -> zero_area. No disjunctive fallback;
        # a misclassified issue must fail this test.
        types = {iss.issue_type for iss in issues}
        assert types == {"micro_noise", "massive_box", "zero_area"}

    def test_auto_fixes_corner_oob_within_bounds(self) -> None:
        """Clips box corners to [0.0, 1.0] if clamped dimensions remain within [0.004, 0.65]."""
        # cx=0.96, w=0.1 -> x1=0.91, x2=1.01 -> clips to x1=0.91, x2=1.0 -> w=0.09
        boxes = [{"class_id": 4, "cx": 0.96, "cy": 0.5, "w": 0.1, "h": 0.1}]
        clean, issues = sanitize_dataset_boxes(boxes, auto_fix_coords=True)
        assert len(clean) == 1
        assert clean[0]["cx"] + clean[0]["w"] * 0.5 <= 1.000001
        assert any(iss.issue_type == "out_of_bounds" and iss.severity == "warning" for iss in issues)

    def test_deduplicates_overlapping_boxes_iou_08(self) -> None:
        """Deduplicates same-class boxes with IoU > 0.8 keeping hand_label / high confidence."""
        boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "confidence": 0.98, "hand_label": True},
            {"class_id": 0, "cx": 0.504, "cy": 0.504, "w": 0.2, "h": 0.2, "confidence": 0.70},
        ]
        clean, issues = sanitize_dataset_boxes(boxes, iou_threshold=0.80)
        assert len(clean) == 1
        assert clean[0]["hand_label"] is True
        assert any(iss.issue_type == "duplicate_overlap" for iss in issues)


class TestCheckBoxSizeBounds:
    """Direct tests for check_box_size_bounds and the L17 bound constants."""

    def test_bounds_constants(self) -> None:
        assert MIN_BOX_DIM_BOUND == 0.004
        assert MAX_BOX_DIM_BOUND == 0.65

    def test_valid_dimensions_accepted(self) -> None:
        is_bad, msg = check_box_size_bounds(0.1, 0.1)
        assert is_bad is False
        assert msg == ""
        # Exact boundaries are accepted.
        assert check_box_size_bounds(0.004, 0.004) == (False, "")
        assert check_box_size_bounds(0.65, 0.65) == (False, "")

    def test_micro_noise_rejected(self) -> None:
        is_bad, msg = check_box_size_bounds(0.002, 0.1)
        assert is_bad is True
        assert "mikro" in msg.lower() or "micro" in msg.lower()
        is_bad, _ = check_box_size_bounds(0.1, 0.001)
        assert is_bad is True

    def test_massive_box_rejected(self) -> None:
        is_bad, msg = check_box_size_bounds(0.7, 0.1)
        assert is_bad is True
        assert "büyük" in msg.lower() or "massive" in msg.lower()
        is_bad, _ = check_box_size_bounds(0.1, 0.66)
        assert is_bad is True

    def test_zero_negative_and_nonfinite_rejected(self) -> None:
        assert check_box_size_bounds(0.0, 0.1)[0] is True
        assert check_box_size_bounds(0.1, -0.05)[0] is True
        assert check_box_size_bounds(float("nan"), 0.1)[0] is True
        assert check_box_size_bounds(float("inf"), 0.1)[0] is True


class TestSanitizeBoxesDirect:
    """Direct tests for sanitize_boxes (the non-dataset entry point)."""

    def test_filters_micro_box_keeping_clean(self) -> None:
        boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
            {"class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.002, "h": 0.05},
        ]
        clean, issues = sanitize_boxes(boxes)
        assert len(clean) == 1
        assert clean[0]["class_id"] == 0
        assert [iss.issue_type for iss in issues] == ["micro_noise"]


class TestValidateYoloFile:
    """Tests for validate_yolo_file (per-file YOLO validation)."""

    def test_clean_file_has_no_issues(self, tmp_path: Path) -> None:
        f = tmp_path / "clean.txt"
        f.write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")
        boxes, issues = validate_yolo_file(f)
        assert len(boxes) == 1
        assert boxes[0]["class_id"] == 0
        assert issues == []

    def test_micro_box_reported_as_micro_noise(self, tmp_path: Path) -> None:
        f = tmp_path / "dirty.txt"
        f.write_text(
            "0 0.500000 0.500000 0.100000 0.100000\n"
            "1 0.300000 0.300000 0.002000 0.050000\n",
            encoding="utf-8",
        )
        boxes, issues = validate_yolo_file(f)
        # F-0852: a line that fails an error-severity check is reported as an
        # issue but NOT added to the returned boxes. The stale assertion
        # len==2 counted the invalid micro box as a returned box.
        assert len(boxes) == 1
        assert boxes[0]["class_id"] == 0
        assert [iss.issue_type for iss in issues] == ["micro_noise"]
        assert issues[0].severity == "error"

    def test_missing_file_reported(self, tmp_path: Path) -> None:
        boxes, issues = validate_yolo_file(tmp_path / "missing.txt")
        assert boxes == []
        assert [iss.issue_type for iss in issues] == ["file_not_found"]
        assert issues[0].severity == "error"


class TestValidateDataset:
    """Tests for validate_dataset (whole-dataset QA audit)."""

    def test_clean_dataset_is_valid(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels" / "train"
        labels.mkdir(parents=True)
        (labels / "good.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        report = validate_dataset(tmp_path, check_pending=False)
        assert report.valid is True
        assert report.total_files_scanned == 1
        assert report.total_boxes_scanned == 1
        assert report.issues == []

    def test_dirty_dataset_reports_exact_issues(self, tmp_path: Path) -> None:
        labels = tmp_path / "labels" / "train"
        labels.mkdir(parents=True)
        (labels / "good.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        (labels / "bad.txt").write_text(
            "1 0.3 0.3 0.002 0.05\n2 0.5 0.5 0.0 0.1\n", encoding="utf-8"
        )
        report = validate_dataset(tmp_path, check_pending=False)
        assert report.valid is False
        assert report.total_files_scanned == 2
        # F-0852: total_boxes_scanned counts returned (valid) boxes only; the
        # micro-noise and zero-area lines are reported as issues, not counted
        # as scanned boxes. The stale assertion ==3 counted all 3 lines.
        assert report.total_boxes_scanned == 1
        assert report.error_count == 2
        assert {iss.issue_type for iss in report.issues} == {"micro_noise", "zero_area"}


class TestAtomicWrites:
    """Tests for atomic_write_text / atomic_write_json."""

    def test_atomic_write_text_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        atomic_write_text(target, "hello\nworld\n")
        assert target.read_text(encoding="utf-8") == "hello\nworld\n"
        # No temp leftovers beside the target.
        assert [p.name for p in tmp_path.iterdir()] == ["out.txt"]

    def test_atomic_write_text_overwrites(self, tmp_path: Path) -> None:
        target = tmp_path / "out.txt"
        target.write_text("stale", encoding="utf-8")
        atomic_write_text(target, "fresh")
        assert target.read_text(encoding="utf-8") == "fresh"

    def test_atomic_write_json_round_trip(self, tmp_path: Path) -> None:
        target = tmp_path / "out.json"
        atomic_write_json(target, {"stem": "x", "trusted": [{"class_id": 0}]})
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data == {"stem": "x", "trusted": [{"class_id": 0}]}
        assert [p.name for p in tmp_path.iterdir()] == ["out.json"]


class TestAtomicSyncReviewTxtPair:
    """Test verify_and_sync_review_txt_pair and sync_all_review_txt_pairs (L16 Guardrail)."""

    def test_clean_synchronized_pair(self, tmp_path: Path) -> None:
        """Fully matched review JSON and label TXT returns True without writing."""
        rev_path = tmp_path / "review.json"
        txt_path = tmp_path / "label.txt"

        payload = {
            "stem": "sample",
            "split": "train",
            "trusted": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
            "weak": [],
            "done": "ok",
        }
        rev_path.write_text(json.dumps(payload), encoding="utf-8")
        txt_path.write_text("0 0.500000 0.500000 0.100000 0.100000\n", encoding="utf-8")

        mtime_txt = txt_path.stat().st_mtime_ns
        mtime_rev = rev_path.stat().st_mtime_ns

        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=False) is True
        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=True) is True

        # Files should not be touched if already synced
        assert txt_path.stat().st_mtime_ns == mtime_txt
        assert rev_path.stat().st_mtime_ns == mtime_rev

    def test_orphan_txt_auto_repaired(self, tmp_path: Path) -> None:
        """Missing JSON is reconstructed atomically from valid TXT."""
        rev_path = tmp_path / "orphan_txt.json"
        txt_path = tmp_path / "orphan_txt.txt"
        txt_path.write_text("1 0.400000 0.400000 0.150000 0.150000\n", encoding="utf-8")

        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=True) is True

        assert rev_path.is_file()
        data = json.loads(rev_path.read_text(encoding="utf-8"))
        assert len(data.get("trusted", [])) == 1
        assert data["trusted"][0]["class_id"] == 1

    def test_orphan_json_auto_repaired(self, tmp_path: Path) -> None:
        """Missing TXT is reconstructed atomically from valid JSON."""
        rev_path = tmp_path / "orphan_json.json"
        txt_path = tmp_path / "orphan_json.txt"
        payload = {
            "stem": "orphan_json",
            "split": "train",
            "trusted": [{"class_id": 2, "cx": 0.6, "cy": 0.6, "w": 0.2, "h": 0.2}],
        }
        rev_path.write_text(json.dumps(payload), encoding="utf-8")

        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=True) is True

        assert txt_path.is_file()
        lines = txt_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert lines[0].startswith("2 ")

    def test_corrupt_json_repaired_from_txt(self, tmp_path: Path) -> None:
        """Corrupted JSON (syntax error) is repaired from valid TXT."""
        rev_path = tmp_path / "corrupt_j.json"
        txt_path = tmp_path / "corrupt_j.txt"
        rev_path.write_text("{broken json syntax!!", encoding="utf-8")
        txt_path.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=True) is True

        recovered = json.loads(rev_path.read_text(encoding="utf-8"))
        assert len(recovered.get("trusted", [])) == 1
        assert recovered["trusted"][0]["class_id"] == 0

    def test_corrupt_txt_repaired_from_json(self, tmp_path: Path) -> None:
        """Corrupted TXT (garbage line) is replaced atomically with clean boxes from JSON."""
        rev_path = tmp_path / "bad_txt.json"
        txt_path = tmp_path / "bad_txt.txt"
        payload = {
            "stem": "bad_txt",
            "split": "train",
            "trusted": [
                {"class_id": 5, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
                {"class_id": 7, "cx": 0.2, "cy": 0.2, "w": 0.05, "h": 0.05},
            ],
        }
        rev_path.write_text(json.dumps(payload), encoding="utf-8")
        txt_path.write_text("invalid line here\n", encoding="utf-8")

        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=False) is False
        assert verify_and_sync_review_txt_pair(rev_path, txt_path, auto_repair=True) is True

        lines = [ln.strip() for ln in txt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 2
        assert lines[0].startswith("5 ")
        assert lines[1].startswith("7 ")

    def test_sync_all_review_txt_pairs_sweep(self, tmp_path: Path) -> None:
        """Sweep across train and val splits repairing all out-of-sync pairs."""
        rev_train = tmp_path / "review" / "train"
        rev_val = tmp_path / "review" / "val"
        lbl_train = tmp_path / "labels" / "train"
        lbl_val = tmp_path / "labels" / "val"

        for d in (rev_train, rev_val, lbl_train, lbl_val):
            d.mkdir(parents=True, exist_ok=True)

        # 1. train: orphan txt
        (lbl_train / "t1.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
        # 2. train: orphan json
        (rev_train / "j1.json").write_text(json.dumps({"trusted": [{"class_id": 1, "cx": 0.4, "cy": 0.4, "w": 0.1, "h": 0.1}]}), encoding="utf-8")
        # 3. val: synced pair
        (rev_val / "v1.json").write_text(json.dumps({"trusted": [{"class_id": 2, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}]}), encoding="utf-8")
        (lbl_val / "v1.txt").write_text("2 0.300000 0.300000 0.100000 0.100000\n", encoding="utf-8")

        stats = sync_all_review_txt_pairs(tmp_path, auto_repair=True)
        assert stats["total_pairs"] == 3
        assert stats["already_synced"] == 1
        assert stats["auto_repaired"] == 2
        assert stats["orphan_txt_fixed"] == 1
        assert stats["orphan_json_fixed"] == 1
        assert stats["failed"] == 0
