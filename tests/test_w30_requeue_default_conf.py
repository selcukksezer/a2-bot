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


def _assert_weak_box(box, expected_conf: float = 0.58) -> None:
    """Shared contract for a requeued weak box: the expected confidence
    (0.58 by default, 0.68 when forced against the heuristic cap), never the
    old 0.90 default, never above the 0.68 cap, trusted=False, weak=True."""
    assert box["confidence"] == pytest.approx(expected_conf)
    assert box["confidence"] != pytest.approx(0.90)
    assert box["confidence"] <= 0.68
    assert box["trusted"] is False
    assert box["weak"] is True


def _write_labeled_frame(root: Path, split: str, img_name: str, txt_text: str) -> None:
    """Write one image + its YOLO .txt label under <root>/images|labels/<split>/."""
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32 if img_name.endswith(".png") else b"\xff\xd8\xff\xe0" + b"\x00" * 32
    (img_dir / img_name).write_bytes(img_bytes)
    (lbl_dir / f"{Path(img_name).stem}.txt").write_text(txt_text, encoding="utf-8")


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
    _assert_weak_box(normalized)
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
    _assert_weak_box(normalized)


def test_w30_04_requeue_labeled_dataset_train_txt_without_conf_creates_weak_058(tmp_path: Path) -> None:
    """Unscored train .txt boxes keep the 0.58 default as weak — but only
    on the direct normalize path now.

    §7 P0-8 requeue kararı: TXT-only kare requeue'da hand_label damgası
    alır (diskteki 5-alanlı TXT satırı el etiketiyle ayırt edilemez) ve
    damgalı kare korumalı sayılır — kare düzeyinde atlanır, review JSON
    yazılmaz. Uçtan uca requeue artık bu kareyi kuyruğa almaz; 0.58 weak
    sözleşmesi damgasız ham kutunun doğrudan _normalize_requeue_box
    geçişiyle denetlenir.
    """
    _write_labeled_frame(
        tmp_path,
        "train",
        "frame_train_01.jpg",
        "0 0.50 0.50 0.10 0.10\n1 0.30 0.40 0.08 0.08\n",
    )

    stats = requeue_labeled_dataset_for_review(tmp_path)
    assert stats["ok"] is True
    assert stats["samples"] == 0
    assert stats["boxes"] == 0
    assert stats["trusted_boxes"] == 0
    assert stats["weak_boxes"] == 0
    assert stats["skipped_protected"] == 1
    assert stats["default_conf"] == 0.58

    review_path = tmp_path / "review" / "train" / "frame_train_01.json"
    assert not review_path.is_file()

    # Damgasız ham kutu doğrudan normalize'da 0.58 weak kalır (§7: kanıtsız
    # kutu requeue'da bile trusted olmaz).
    raw_box = {
        "class_id": 0, "cx": 0.50, "cy": 0.50, "w": 0.10, "h": 0.10,
        "has_explicit_conf": False, "confidence": 0.0,
    }
    normalized = _normalize_requeue_box(raw_box, stem="frame_train_01")
    assert normalized is not None
    _assert_weak_box(normalized)


def test_w30_05_requeue_labeled_dataset_val_split_even_with_high_default_conf_stays_weak(tmp_path: Path) -> None:
    """Unscored .txt boxes in val split get 0.58 by default; the direct _normalize_requeue_box path stays trusted=False, weak=True even if default_conf=0.90.

    §7 P0-8 requeue kararı: TXT-only kare requeue'da hand_label damgası
    alır (diskteki 5-alanlı TXT satırı el etiketiyle ayırt edilemez) ve
    damgalı kare korumalı sayılır — kare düzeyinde atlanır, review JSON
    yazılmaz. Uçtan uca requeue artık bu kareyi kuyruğa almaz.
    Damgasız ham kutunun doğrudan _normalize_requeue_box'tan geçişi
    ise aşağıda aynen weak kalır.
    """
    _write_labeled_frame(
        tmp_path,
        "val",
        "frame_val_01.png",
        "2 0.60 0.60 0.15 0.15\n",
    )

    # §7 P0-8: TXT-only kare korumalı atlanır — kuyruğa girmez.
    stats = requeue_labeled_dataset_for_review(tmp_path)
    assert stats["ok"] is True
    assert stats["samples"] == 0
    assert stats["trusted_boxes"] == 0
    assert stats["weak_boxes"] == 0
    assert stats["skipped_protected"] == 1

    review_path = tmp_path / "review" / "val" / "frame_val_01.json"
    assert not review_path.is_file()

    # Even if an unverified raw .txt box is passed with default_conf=0.90, it still cannot become trusted
    raw_unverified = {"class_id": 2, "cx": 0.6, "cy": 0.6, "w": 0.15, "h": 0.15, "has_explicit_conf": False}
    norm_forced = _normalize_requeue_box(raw_unverified, default_conf=0.90, stem="frame_val_01")
    assert norm_forced is not None
    # Güven üst sınırı: doğrulanmamış kutu 0.90 ile zorlansa bile sezgisel-kaynak
    # cap'i (0.68) aşamaz — 0.90 olduğu gibi geçmemeli.
    _assert_weak_box(norm_forced, expected_conf=0.68)
