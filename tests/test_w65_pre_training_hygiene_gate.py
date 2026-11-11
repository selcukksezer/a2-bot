from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.pre_training_hygiene_gate import PreTrainingHygieneGate  # noqa: E402
# F-2052 (Yüksek, cross-slice blocked): bu gate'i rtx_trainer_engine /
# factory_controller dahil hiçbir üretim modülü çağırmıyor — kablolama bu
# dilimin dosyaları (yalnızca testler) dışında; orchestrator kararı bekler.


def test_is_valid_class_id_rejects_98_and_invalid_values() -> None:
    gate = PreTrainingHygieneGate(min_class_id=0, max_class_id=24)

    assert gate.is_valid_class_id(0) is True
    assert gate.is_valid_class_id(24) is True
    assert gate.is_valid_class_id("12") is True

    # Invalid class ID 98 and out-of-range / malformed inputs
    assert gate.is_valid_class_id(98) is False
    assert gate.is_valid_class_id("98") is False
    assert gate.is_valid_class_id(-1) is False
    assert gate.is_valid_class_id(True) is False
    assert gate.is_valid_class_id("invalid") is False


def test_validate_normalized_box_rejects_corrupted_boxes() -> None:
    gate = PreTrainingHygieneGate()

    ok, reason, box = gate.validate_normalized_box(0.5, 0.5, 0.2, 0.3)
    assert ok is True
    assert reason == "ok"
    assert box == pytest.approx((0.5, 0.5, 0.2, 0.3))

    # Non-positive width / height
    ok_zero_w, reason_zero_w, _ = gate.validate_normalized_box(0.5, 0.5, 0.0, 0.2)
    assert ok_zero_w is False
    assert reason_zero_w == "corrupted"

    ok_neg_h, reason_neg_h, _ = gate.validate_normalized_box(0.5, 0.5, 0.2, -0.1)
    assert ok_neg_h is False
    assert reason_neg_h == "corrupted"

    # Out-of-bounds coordinates and NaN
    ok_oob, reason_oob, _ = gate.validate_normalized_box(1.5, 0.5, 0.2, 0.2)
    assert ok_oob is False
    assert reason_oob == "corrupted"

    ok_nan, reason_nan, _ = gate.validate_normalized_box(float("nan"), 0.5, 0.2, 0.2)
    assert ok_nan is False
    assert reason_nan == "corrupted"


def test_sanitize_txt_file_removes_class_98_and_corrupted_boxes(tmp_path: Path) -> None:
    gate = PreTrainingHygieneGate()
    label_file = tmp_path / "sample.txt"
    label_file.write_text(
        "\n".join(
            [
                "3 0.500000 0.500000 0.200000 0.200000",
                "98 0.400000 0.400000 0.100000 0.100000",  # invalid class ID 98
                "5 0.500000 0.500000 -0.200000 0.200000",  # corrupted negative width
                "7 1.400000 0.500000 0.200000 0.200000",  # corrupted out-of-range cx
                "2 0.300000 0.300000",  # corrupted incomplete box tokens
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = gate.sanitize_txt_file(label_file)

    assert report.scanned_txt_files == 1
    assert report.modified_txt_files == 1
    assert report.total_annotations_scanned == 5
    assert report.valid_annotations_kept == 1
    assert report.removed_invalid_class_id == 1
    assert report.removed_corrupted_boxes == 3

    cleaned_lines = label_file.read_text(encoding="utf-8").strip().splitlines()
    assert cleaned_lines == ["3 0.500000 0.500000 0.200000 0.200000"]


def test_sanitize_json_file_removes_class_98_and_corrupted_boxes(tmp_path: Path) -> None:
    gate = PreTrainingHygieneGate()
    json_file = tmp_path / "annotations.json"
    coco_payload = {
        "images": [{"id": 1, "width": 1000, "height": 1000}],
        "categories": [
            {"id": 1, "name": "valid_obj"},
            {"id": 98, "name": "invalid_obj_98"},
        ],
        "annotations": [
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 200]},
            {"id": 11, "image_id": 1, "category_id": 98, "bbox": [150, 150, 100, 100]},
            {"id": 12, "image_id": 1, "category_id": 1, "bbox": [100, 100, -50, 200]},
            {"id": 13, "image_id": 1, "category_id": 1, "bbox": [900, 900, 300, 300]},
        ],
    }
    json_file.write_text(json.dumps(coco_payload), encoding="utf-8")

    report = gate.sanitize_json_file(json_file)

    assert report.scanned_json_files == 1
    assert report.modified_json_files == 1
    assert report.total_annotations_scanned == 4
    assert report.valid_annotations_kept == 1
    assert report.removed_invalid_class_id == 1
    assert report.removed_corrupted_boxes == 2

    cleaned_data = json.loads(json_file.read_text(encoding="utf-8"))
    assert [c["id"] for c in cleaned_data["categories"]] == [1]
    assert len(cleaned_data["annotations"]) == 1
    assert cleaned_data["annotations"][0]["id"] == 10


def test_scan_and_clean_directory_hygiene_end_to_end(tmp_path: Path) -> None:
    gate = PreTrainingHygieneGate()
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    # Ignored file should not be scanned
    (labels_dir / "classes.txt").write_text("98_ignored\n", encoding="utf-8")

    txt_path = labels_dir / "frame_01.txt"
    txt_path.write_text(
        "98 0.5 0.5 0.2 0.2\n"
        "4 0.5 0.5 0.0 0.2\n"
        "4 0.5 0.5 0.2 0.2\n",
        encoding="utf-8",
    )

    json_path = labels_dir / "frame_01.json"
    json_path.write_text(
        json.dumps(
            [
                {"class_id": 98, "bbox": [0.5, 0.5, 0.2, 0.2]},
                {"class_id": 2, "bbox": [0.5, 0.5, -0.2, 0.2]},
                {"class_id": 2, "bbox": [0.5, 0.5, 0.2, 0.2]},
            ]
        ),
        encoding="utf-8",
    )

    report = gate.scan_and_clean(labels_dir)
    summary = report.to_dict()

    assert summary["scanned_txt_files"] == 1
    assert summary["scanned_json_files"] == 1
    assert summary["removed_invalid_class_id"] == 2
    assert summary["removed_corrupted_boxes"] == 2
    assert summary["valid_annotations_kept"] == 2
    assert summary["total_removed"] == 4
    assert summary["errors"] == []
