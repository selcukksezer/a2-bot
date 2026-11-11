# -*- coding: utf-8 -*-
"""W40 Unit Tests: Box Size Bounds (<0.004 micro noise, >0.65 massive fake box)
and Out-of-Bounds Coordinate Hygiene Gate verification for dataset_validator.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_factory.dataset_validator import (
    MIN_BOX_DIM_BOUND,
    MAX_BOX_DIM_BOUND,
    check_box_size_bounds,
    check_out_of_bounds,
    validate_label_box,
    sanitize_dataset_boxes,
    validate_yolo_file,
    validate_dataset,
    is_dataset_ready_for_training,
)


def test_micro_noise_box_blocked() -> None:
    """1. Verify micro noise boxes (<0.004 in width or height) are blocked."""
    assert MIN_BOX_DIM_BOUND == pytest.approx(0.004)

    # Width < 0.004
    is_bad_w, msg_w = check_box_size_bounds(0.0039, 0.10)
    assert is_bad_w is True
    assert "0.004" in msg_w

    res_w = validate_label_box(0.5, 0.5, 0.0039, 0.10)
    assert res_w.is_valid is False
    assert not res_w

    # Height < 0.004
    is_bad_h, msg_h = check_box_size_bounds(0.10, 0.001)
    assert is_bad_h is True
    assert "0.004" in msg_h

    res_h = validate_label_box({"cx": 0.5, "cy": 0.5, "w": 0.10, "h": 0.002, "class_id": 0})
    assert res_h.is_valid is False

    # Boundary check: 0.004 is valid
    res_boundary = validate_label_box(0.5, 0.5, 0.004, 0.004)
    assert res_boundary.is_valid is True

    # Sanitizer drops micro noise boxes and records micro_noise error
    cleaned, issues = sanitize_dataset_boxes([
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.003, "h": 0.05},
        {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.05, "h": 0.0035},
        {"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.05, "h": 0.05},
    ])
    assert len(cleaned) == 1
    assert cleaned[0]["class_id"] == 2
    micro_issues = [iss for iss in issues if iss.issue_type == "micro_noise" and iss.severity == "error"]
    assert len(micro_issues) == 2


def test_massive_fake_box_blocked() -> None:
    """2. Verify massive screen-filling fake boxes (>0.65 in width or height) are blocked."""
    assert MAX_BOX_DIM_BOUND == pytest.approx(0.65)

    # Width > 0.65
    is_bad_w, msg_w = check_box_size_bounds(0.651, 0.20)
    assert is_bad_w is True
    assert "0.65" in msg_w

    res_w = validate_label_box(0.5, 0.5, 0.66, 0.20)
    assert res_w.is_valid is False
    assert not res_w

    # Height > 0.65
    is_bad_h, msg_h = check_box_size_bounds(0.20, 0.85)
    assert is_bad_h is True
    assert "0.65" in msg_h

    res_h = validate_label_box([0, 0.5, 0.5, 0.20, 0.72])
    assert res_h.is_valid is False

    # Boundary check: 0.65 is valid
    res_boundary = validate_label_box(0.5, 0.5, 0.65, 0.65)
    assert res_boundary.is_valid is True

    # Sanitizer drops massive boxes and records massive_box error
    cleaned, issues = sanitize_dataset_boxes([
        {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.68, "h": 0.30},
        {"class_id": 1, "cx": 0.5, "cy": 0.5, "w": 0.30, "h": 0.90},
        {"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.25, "h": 0.25},
    ])
    assert len(cleaned) == 1
    assert cleaned[0]["class_id"] == 2
    massive_issues = [iss for iss in issues if iss.issue_type == "massive_box" and iss.severity == "error"]
    assert len(massive_issues) == 2


def test_out_of_bounds_coordinates_blocked() -> None:
    """3. Verify off-screen / out-of-bounds center and corner coordinates are blocked."""
    # Center out of [0.0, 1.0]
    assert not validate_label_box(-0.05, 0.5, 0.1, 0.1)
    assert not validate_label_box(1.05, 0.5, 0.1, 0.1)
    assert not validate_label_box(0.5, -0.01, 0.1, 0.1)
    assert not validate_label_box(0.5, 1.02, 0.1, 0.1)

    # Corners extending outside [0.0, 1.0] when center is inside [0.0, 1.0]
    # x2 = 0.95 + 0.10 = 1.05 > 1.0
    is_oob, errors = check_out_of_bounds(0.95, 0.5, 0.20, 0.20)
    assert is_oob is True
    assert len(errors) >= 1

    res_corner = validate_label_box(0.95, 0.5, 0.20, 0.20, strict_bounds=True)
    assert res_corner.is_valid is False

    # Completely off-screen or unfixable boxes rejected by sanitizer
    cleaned, issues = sanitize_dataset_boxes(
        [
            {"class_id": 0, "cx": -0.2, "cy": 0.5, "w": 0.1, "h": 0.1},
            {"class_id": 1, "cx": 0.98, "cy": 0.5, "w": 0.1, "h": 0.1},
            {"class_id": 2, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
        ],
        auto_fix_coords=False,
    )
    assert len(cleaned) == 1
    assert cleaned[0]["class_id"] == 2
    oob_issues = [iss for iss in issues if iss.issue_type == "out_of_bounds" and iss.severity == "error"]
    assert len(oob_issues) == 2


def test_validate_yolo_file_blocks_all_hygiene_violations(tmp_path: Path) -> None:
    """4. Verify validate_yolo_file flags micro noise, massive boxes, and out-of-bounds coords."""
    yolo_txt = tmp_path / "sample_hygiene.txt"
    yolo_txt.write_text(
        "\n".join([
            "0 0.500000 0.500000 0.002000 0.050000",  # line 1: micro_noise (w=0.002 < 0.004)
            "1 0.500000 0.500000 0.750000 0.200000",  # line 2: massive_box (w=0.75 > 0.65)
            "2 0.960000 0.500000 0.200000 0.200000",  # line 3: out_of_bounds (x2=1.06 > 1.0)
            "3 0.500000 0.500000 0.100000 0.100000",  # line 4: valid box
        ]) + "\n",
        encoding="utf-8",
    )

    boxes, issues = validate_yolo_file(yolo_txt, strict_bounds=True)
    # S-3102: üretim tarafı (dataset_validator.py, F-0852) hatalı satırı
    # boxes'a eklemez — 3 hatalı satır atlanır, yalnızca geçerli satır (4.
    # satır, class_id 3) döner. Issue'lar yine de 3 hata tipini taşır.
    assert len(boxes) == 1
    assert boxes[0]["class_id"] == 3

    issue_types = [iss.issue_type for iss in issues if iss.severity == "error"]
    assert "micro_noise" in issue_types
    assert "massive_box" in issue_types
    assert "out_of_bounds" in issue_types


def test_dataset_hygiene_gate_blocks_training_on_invalid_boxes(tmp_path: Path) -> None:
    """5. Verify validate_dataset and is_dataset_ready_for_training block contaminated datasets."""
    train_labels = tmp_path / "labels" / "train"
    train_labels.mkdir(parents=True, exist_ok=True)

    bad_file = train_labels / "frame_001.txt"
    bad_file.write_text(
        "\n".join([
            "0 0.500000 0.500000 0.001500 0.020000",  # micro noise
            "1 0.500000 0.500000 0.200000 0.800000",  # massive fake box
            "2 -0.100000 0.500000 0.100000 0.100000", # off-screen coordinate
        ]) + "\n",
        encoding="utf-8",
    )

    report = validate_dataset(tmp_path, split="train", check_pending=False)
    assert report.valid is False
    assert report.size_bounds_count == 2
    assert report.out_of_bounds_count == 1
    assert report.error_count >= 3

    ready, reason = is_dataset_ready_for_training(tmp_path)
    assert ready is False
    # F-1586: pin the exact gate reason instead of a loose `"QA" in reason
    # or "başarısız" in reason` — the message must carry the QA summary.
    assert reason == f"Eğitim engellendi: QA doğrulaması başarısız ({report.summary})."

    # Replace with clean box and verify gate passes
    bad_file.write_text("0 0.500000 0.500000 0.120000 0.120000\n", encoding="utf-8")
    clean_report = validate_dataset(tmp_path, split="train", check_pending=False)
    assert clean_report.valid is True
    assert clean_report.error_count == 0

    ready_clean, _ = is_dataset_ready_for_training(tmp_path)
    assert ready_clean is True


@pytest.mark.parametrize(
    "w,h,is_bad,msg_part",
    [
        # Micro-noise dimension violations (each dimension independently).
        (0.0039, 0.10, True, "0.004"),
        (0.10, 0.001, True, "0.004"),
        # Exact dimension boundaries are valid (0.004 and 0.65 inclusive).
        (0.004, 0.004, False, ""),
        (0.004, 0.10, False, ""),
        (0.10, 0.004, False, ""),
        (0.65, 0.65, False, ""),
        (0.65, 0.20, False, ""),
        # Massive-box dimension violations.
        (0.651, 0.20, True, "0.65"),
        (0.20, 0.85, True, "0.65"),
        # Clean interior box.
        (0.10, 0.10, False, ""),
        # Zero / negative dimensions — previously untested branch.
        (0.0, 0.10, True, "Sıfır"),
        (-0.05, 0.10, True, "Sıfır"),
        (0.10, -0.01, True, "Sıfır"),
        # Non-finite dimensions — previously untested branch.
        (float("nan"), 0.10, True, "geçersiz sayı"),
        (float("inf"), 0.10, True, "geçersiz sayı"),
        (0.10, float("nan"), True, "geçersiz sayı"),
    ],
)
def test_check_box_size_bounds_parametrized(w: float, h: float, is_bad: bool, msg_part: str) -> None:
    """check_box_size_bounds directly, one case per test.

    The existing tests exercise this function only through
    validate_label_box / sanitize_dataset_boxes and only for the
    micro-noise and massive-box dimensions. The zero/negative and
    non-finite branches had no coverage; parametrizing also gives each
    (w, h) pair its own test id instead of masking later cases behind
    an earlier assertion failure.
    """
    bad, msg = check_box_size_bounds(w, h)
    assert bad is is_bad
    if is_bad:
        assert msg_part in msg
    else:
        assert msg == ""
