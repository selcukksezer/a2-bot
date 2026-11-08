"""W30: Unit tests for requeue default confidence (0.58 instead of 0.90) and trusted=False, weak=True."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from dataset_factory.label_trust import (
    _legacy_weak_from_txt,
    _normalize_requeue_box,
    requeue_labeled_dataset_for_review,
)


def test_w30_01_signature_default_conf_is_058_not_090() -> None:
    """Both _normalize_requeue_box and requeue_labeled_dataset_for_review must default to 0.58 (not 0.90)."""
    sig_norm = inspect.signature(_normalize_requeue_box)
    norm_default = sig_norm.parameters["default_conf"].default
    assert norm_default == pytest.approx(0.58)
    assert norm_default != pytest.approx(0.90)

    sig_requeue = inspect.signature(requeue_labeled_dataset_for_review)
    requeue_default = sig_requeue.parameters["default_conf"].default
    assert requeue_default == pytest.approx(0.58)
    assert requeue_default != pytest.approx(0.90)


def test_w30_02_normalize_requeue_box_unscored_txt_gets_058_and_weak() -> None:
    """_normalize_requeue_box assigns 0.58 confidence, trusted=False, weak=True to unscored .txt boxes."""
    raw_box = {
        "class_id": 0,
        "cx": 0.50,
        "cy": 0.50,
        "w": 0.12,
        "h": 0.12,
        "has_explicit_conf": False,
        "confidence": 0.0,
    }
    normalized = _normalize_requeue_box(raw_box, stem="sample_frame_01")
    assert normalized is not None
    assert normalized["confidence"] == pytest.approx(0.58)
    assert normalized["confidence"] != pytest.approx(0.90)
    assert normalized["trusted"] is False
    assert normalized["weak"] is True
    assert normalized["requeued"] is True


def test_w30_03_legacy_txt_5_column_parsed_and_normalized_to_058_weak(tmp_path: Path) -> None:
    """5-column YOLO .txt line (no confidence column) parses and normalizes to conf=0.58, trusted=False, weak=True."""
    lbl_path = tmp_path / "unscored_frame.txt"
    lbl_path.write_text("0 0.450000 0.550000 0.100000 0.140000\n", encoding="utf-8")

    parsed = _legacy_weak_from_txt(lbl_path)
    assert len(parsed) == 1
    assert bool(parsed[0].get("has_explicit_conf")) is False

    normalized = _normalize_requeue_box(parsed[0], stem="unscored_frame")
    assert normalized is not None
    assert normalized["confidence"] == pytest.approx(0.58)
    assert normalized["confidence"] != pytest.approx(0.90)
    assert normalized["trusted"] is False
    assert normalized["weak"] is True


def test_w30_04_requeue_labeled_dataset_train_txt_without_conf_creates_weak_058(tmp_path: Path) -> None:
    """requeue_labeled_dataset_for_review puts unscored train .txt boxes into weak list with 0.58 confidence."""
    img_dir = tmp_path / "images" / "train"
    lbl_dir = tmp_path / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    (img_dir / "frame_train_01.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 32)
    (lbl_dir / "frame_train_01.txt").write_text(
        "0 0.50 0.50 0.10 0.10\n1 0.30 0.40 0.08 0.08\n",
        encoding="utf-8",
    )

    stats = requeue_labeled_dataset_for_review(tmp_path)
    assert stats["ok"] is True
    assert stats["samples"] == 1
    assert stats["boxes"] == 2
    assert stats["trusted_boxes"] == 0
    assert stats["weak_boxes"] == 2

    review_path = tmp_path / "review" / "train" / "frame_train_01.json"
    assert review_path.is_file()
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert payload["trusted"] == []
    assert len(payload["weak"]) == 2
    for box in payload["weak"]:
        assert box["confidence"] == pytest.approx(0.58)
        assert box["confidence"] != pytest.approx(0.90)
        assert box["trusted"] is False
        assert box["weak"] is True


def test_w30_05_requeue_labeled_dataset_val_split_even_with_high_default_conf_stays_weak(tmp_path: Path) -> None:
    """Unscored .txt boxes in val split get 0.58 by default and stay trusted=False, weak=True even if default_conf=0.90."""
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    (img_dir / "frame_val_01.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    (lbl_dir / "frame_val_01.txt").write_text("2 0.60 0.60 0.15 0.15\n", encoding="utf-8")

    # Default call: must assign 0.58 and weak=True, trusted=False
    stats = requeue_labeled_dataset_for_review(tmp_path)
    assert stats["trusted_boxes"] == 0
    assert stats["weak_boxes"] == 1

    review_path = tmp_path / "review" / "val" / "frame_val_01.json"
    payload = json.loads(review_path.read_text(encoding="utf-8"))
    assert len(payload["trusted"]) == 0
    assert len(payload["weak"]) == 1
    box = payload["weak"][0]
    assert box["confidence"] == pytest.approx(0.58)
    assert box["trusted"] is False
    assert box["weak"] is True

    # Even if an unverified raw .txt box is passed with default_conf=0.90, it still cannot become trusted
    raw_unverified = {"class_id": 2, "cx": 0.6, "cy": 0.6, "w": 0.15, "h": 0.15, "has_explicit_conf": False}
    norm_forced = _normalize_requeue_box(raw_unverified, default_conf=0.90, stem="frame_val_01")
    assert norm_forced is not None
    assert norm_forced["trusted"] is False
    assert norm_forced["weak"] is True
