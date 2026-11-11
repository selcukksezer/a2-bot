# -*- coding: utf-8 -*-
"""
Tests for Work Package G: Emin (Güvenilir) vs Şüpheli Classification & Rich Filtering.
- Separate boxes with confidence >= 0.85 or hand/seed/exemplar as trusted=True, weak=False (Emin)
- Boxes with confidence < 0.85 as trusted=False, weak=True (Şüpheli)
- Canvas rendering distinctions (solid outline + clean name + checkmark vs dashed outline + supheli)
- Rich multi-criteria filtering: queue_filter (all, suspicious, confident, mixed, empty, unlabeled_and_suspicious)
- Class filter and search filter
"""

import json
from pathlib import Path
import sys
import tempfile
import shutil
import pytest

from PyQt6.QtWidgets import QApplication

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.label_trust import (
    _legacy_weak_from_txt,
    _normalize_requeue_box,
    requeue_labeled_dataset_for_review,
    train_txt_boxes,
)
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler
from ui.yolo_factory_dialog import YOLOFactoryDialog


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """YOLOFactoryDialog with its dataset dir redirected to tmp_path."""
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    return dlg


def _filtered_stems(dlg: YOLOFactoryDialog, mode: str, class_id: int = -1, search: str = ""):
    """Drive the REAL production filter: set mode/class/search on the dialog and
    return the stems selected by YOLOFactoryDialog._filtered_review_indices()."""
    if class_id >= 0:
        idx = dlg.review_class_filter.findData(class_id)
        assert idx >= 0, f"class filter has no entry for class_id={class_id}"
        dlg.review_class_filter.setCurrentIndex(idx)
    else:
        idx_all = dlg.review_class_filter.findData(-1)
        if idx_all >= 0:
            dlg.review_class_filter.setCurrentIndex(idx_all)
    dlg.review_search_edit.setText(search)
    dlg.set_queue_filter_mode(mode)
    return [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()]


@pytest.fixture
def temp_dataset_dir():
    temp_dir = Path(tempfile.mkdtemp(prefix="emin_supheli_"))
    for split in ("train", "val"):
        (temp_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (temp_dir / "review" / split).mkdir(parents=True, exist_ok=True)

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_yolo_auto_labeler_emin_vs_supheli():
    """Verify YOLOAutoLabeler._make_annotation marks conf >= 0.85 as trusted=True, weak=False (Emin)."""
    labeler = YOLOAutoLabeler()

    # Confident box (>= 0.85)
    box_emin = labeler._make_annotation(
        class_id=0, x=100, y=100, w=50, h=50, img_w=1000, img_h=1000,
        confidence=0.92, source="vision_3d_crest"
    )
    assert box_emin["trusted"] is True
    assert box_emin["weak"] is False

    # Weak box (< 0.85)
    box_supheli = labeler._make_annotation(
        class_id=1, x=100, y=100, w=50, h=50, img_w=1000, img_h=1000,
        confidence=0.65, source="visual_fallback"
    )
    assert box_supheli["trusted"] is False
    assert box_supheli["weak"] is True

    # Hand-labeled box regardless of confidence is always Emin
    box_hand = labeler._make_annotation(
        class_id=2, x=100, y=100, w=50, h=50, img_w=1000, img_h=1000,
        confidence=0.50, source="hand_label", hand_label=True
    )
    assert box_hand["trusted"] is True
    assert box_hand["weak"] is False


def test_requeue_labeled_dataset_separates_emin_and_supheli(temp_dataset_dir):
    """Verify requeue separates boxes into trusted (Emin) and weak (Şüpheli).

    §7 P0-8 requeue kararı: TXT-only kare requeue'da hand_label damgası
    alır (diskteki 5-alanlı TXT satırı el etiketiyle ayırt edilemez) ve
    damgalı kare korumalı sayılır — kare düzeyinde atlanır, review JSON
    yazılmaz. Bu yüzden uçtan uca requeue artık bu kareyi kuyruğa almaz;
    Emin/Şüpheli ayrımı damgasız kutuların doğrudan
    _normalize_requeue_box geçişiyle denetlenir: ölçülmüş güven >= 0.85
    trusted olur, kanıtsız kutu (0.60 / varsayılan 0.58) weak kalır.
    """
    # Create an image and label txt in train
    img_path = temp_dataset_dir / "images" / "train" / "frame_001.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)  # Dummy JPEG header

    lbl_path = temp_dataset_dir / "labels" / "train" / "frame_001.txt"
    # Format: class cx cy w h [conf]
    lines = [
        "0 0.5 0.5 0.1 0.1 0.95",  # Emin (conf >= 0.85)
        "1 0.3 0.3 0.1 0.1 0.60",  # §7: kanıtsız -> weak
        "2 0.7 0.7 0.1 0.1",       # §7: default 0.58 -> weak
    ]
    lbl_path.write_text("\n".join(lines), encoding="utf-8")

    stats = requeue_labeled_dataset_for_review(
        dataset_dir=temp_dataset_dir,
        emin_threshold=0.85,
    )

    assert stats["ok"] is True
    assert stats["samples"] == 0
    assert stats["boxes"] == 0
    assert stats["skipped_protected"] == 1

    rev_file = temp_dataset_dir / "review" / "train" / "frame_001.json"
    assert not rev_file.is_file()

    # Emin/Şüpheli ayrımı damgasız kutularda korunur (§7: güven yalnızca
    # ölçülmüş güven >= emin ile kazanılır, kanıtsız kutu weak kalır).
    parsed = _legacy_weak_from_txt(lbl_path)
    assert len(parsed) == 3
    normalized = [
        _normalize_requeue_box(dict(b), stem="frame_001", emin_threshold=0.85)
        for b in parsed
    ]
    assert all(n is not None for n in normalized)
    trusted = [n for n in normalized if n["trusted"] is True]
    weak = [n for n in normalized if n["trusted"] is False]
    assert len(trusted) == 1
    assert len(weak) == 2
    assert trusted[0]["confidence"] == 0.95
    assert trusted[0]["weak"] is False
    assert weak[0]["confidence"] == 0.60
    assert weak[1]["confidence"] == 0.58
    assert all(b.get("trusted") is False and b.get("weak") is True for b in weak)


def test_rich_filter_categorization(factory_dialog: YOLOFactoryDialog):
    """Verify queue items match the 6 rich filter categories via the REAL
    production filter (YOLOFactoryDialog._filtered_review_indices):
    all, suspicious, confident, mixed, empty, unlabeled_and_suspicious."""
    dlg = factory_dialog
    dlg._review_items = [
        {"stem": "item_empty", "trusted": [], "weak": []},
        {"stem": "item_suspicious", "trusted": [], "weak": [{"class_id": 0, "conf": 0.50, "confidence": 0.50}]},
        {"stem": "item_confident", "trusted": [{"class_id": 1, "conf": 0.95, "confidence": 0.95}], "weak": []},
        {"stem": "item_mixed", "trusted": [{"class_id": 2, "conf": 0.92, "confidence": 0.92}], "weak": [{"class_id": 3, "conf": 0.40, "confidence": 0.40}]},
    ]

    assert len(_filtered_stems(dlg, "all")) == 4
    assert _filtered_stems(dlg, "empty") == ["item_empty"]
    assert _filtered_stems(dlg, "suspicious") == ["item_suspicious"]
    assert _filtered_stems(dlg, "confident") == ["item_confident"]
    assert _filtered_stems(dlg, "mixed") == ["item_mixed"]
    assert set(_filtered_stems(dlg, "unlabeled_and_suspicious")) == {"item_empty", "item_suspicious", "item_mixed"}
    assert _filtered_stems(dlg, "all", class_id=2) == ["item_mixed"]
    assert _filtered_stems(dlg, "all", search="confident") == ["item_confident"]
