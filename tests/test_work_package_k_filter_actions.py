# -*- coding: utf-8 -*-
"""
tests/test_work_package_k_filter_actions.py

Comprehensive Verification Suite for Work Package K:
Dynamic filter-aware bulk action buttons and strict filter scope isolation.

Requirements:
1. Strict Filter Scope Isolation:
   - _bulk_approve, _bulk_reject_filtered, _bulk_prune_low_confidence_boxes,
     _start_live_relabel_all, and _turbo_relabel_filtered operate STRICTLY AND EXCLUSIVELY
     on self._filtered_review_indices().
   - Photos outside the filter are NEVER touched, approved, pruned, or deleted.
2. Dynamic Button Labels with ({n} kare) Counts:
   - Updates across all filter modes:
     - 'suspicious'
     - 'empty'
     - 'confident'
     - 'unlabeled_and_suspicious' / 'unlabeled_or_suspicious'
     - 'below'
     - 'above'
     - 'mixed'
     - 'all' (with and without class / search filter tags)
   - Proper enable/disable states (e.g. prune disabled in 'empty' mode).
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtWidgets import QApplication

# Initialize QApplication for headless execution
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import CLASS_ID_TO_SPEC, YOLOFactoryDialog


@pytest.fixture(scope="module")
def _shared_dialog() -> YOLOFactoryDialog:
    """One YOLOFactoryDialog for the whole module.

    Constructing the dialog costs ~3-7s per test (full Qt UI build), which
    dominated this file's runtime (setup ~35s of ~52s total). The dialog is
    stateless between tests once the reset fixture below restores the five
    inputs _filtered_review_indices() reads (_review_items, search text,
    class filter, bulk_range threshold, queue filter mode) and rebinds
    _dataset_dir to the current test's tmp_path. Tests never rely on a
    freshly constructed dialog: each one (re)assigns _review_items and its
    filter mode before asserting.
    """
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    return dlg


@pytest.fixture
def factory_dialog(_shared_dialog: YOLOFactoryDialog, tmp_path: Path) -> YOLOFactoryDialog:
    """Per-test dialog: the shared instance, reset to a known state.

    Reset covers every piece of state a test in this file can leave behind:
    review items, search text, class filter, threshold slider (default 90),
    queue filter mode, and the tmp_path-backed dataset dir (bulk actions
    persist/delete real files under it, so it must point at *this* test's
    tmp_path, not a previous test's).
    """
    dlg = _shared_dialog
    # [YOLO-ASYNC] Paylaşılan dialogun __init__'teki başlangıç yüklemesi
    # (üretim dizini) bitmeden durum sıfırlanırsa, bitince kuyruğu üretim
    # taramasıyla ezer — önce bitmesini bekle.
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    # Redirect dataset dir to tmp_path to prevent touching production files
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    dlg._review_items = []
    dlg.review_search_edit.clear()
    all_idx = dlg.review_class_filter.findData(-1)
    if all_idx >= 0:
        dlg.review_class_filter.setCurrentIndex(all_idx)
    dlg.bulk_range.setValue(90)
    dlg.set_queue_filter_mode("all")
    return dlg


def _box(confidence: float, class_id: int, bbox: List[float]) -> Dict[str, Any]:
    """One fixture box in the canonical cx/cy schema (B-04).

    ``bbox`` is [x1, y1, x2, y2] on a 100px canvas (all fixture corners
    are <= 95); the normalizer only accepts normalized cx/cy/w/h, so a
    bbox-only box is rejected as "cx eksik" and can never persist.
    The ``bbox`` key is kept for traceability; persist reads cx/cy/w/h.
    """
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return {
        "confidence": confidence,
        "class_id": class_id,
        "bbox": [x1, y1, x2, y2],
        "cx": round((x1 + x2) / 200.0, 6),
        "cy": round((y1 + y2) / 200.0, 6),
        "w": round((x2 - x1) / 100.0, 6),
        "h": round((y2 - y1) / 100.0, 6),
    }


def _create_sample_review_items(tmp_path: Path) -> List[Dict[str, Any]]:
    """Helper to create a diverse set of review items with actual dummy files."""
    items = [
        # Item 0: Pure Suspicious (1 weak box, 0 trusted)
        {
            "stem": "frame_susp_1",
            "split": "train",
            "weak": [_box(0.42, 0, [10, 10, 50, 50])],
            "trusted": [],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "frame_susp_1.json"),
        },
        # Item 1: Pure Suspicious (1 weak box, 0 trusted)
        {
            "stem": "frame_susp_2",
            "split": "train",
            "weak": [_box(0.55, 3, [20, 20, 60, 60])],
            "trusted": [],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "frame_susp_2.json"),
        },
        # Item 2: Pure Confident (0 weak, 1 trusted box)
        {
            "stem": "frame_conf_1",
            "split": "train",
            "weak": [],
            "trusted": [_box(0.96, 3, [30, 30, 70, 70])],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "frame_conf_1.json"),
        },
        # Item 3: Empty (0 weak, 0 trusted)
        {
            "stem": "frame_empty_1",
            "split": "train",
            "weak": [],
            "trusted": [],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "frame_empty_1.json"),
        },
        # Item 4: Empty (0 weak, 0 trusted)
        {
            "stem": "frame_empty_2",
            "split": "train",
            "weak": [],
            "trusted": [],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "frame_empty_2.json"),
        },
        # Item 5: Mixed (1 trusted box, 1 weak box)
        {
            "stem": "frame_mixed_1",
            "split": "train",
            "weak": [_box(0.35, 1, [15, 15, 45, 45])],
            "trusted": [_box(0.94, 0, [55, 55, 95, 95])],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "frame_mixed_1.json"),
        },
    ]

    # Create dummy files on disk so _persist_review_item and _delete_review_sample work cleanly
    for it in items:
        p = Path(it["path"])
        p.write_text("{}", encoding="utf-8")
        img_p = tmp_path / "images" / it["split"] / f"{it['stem']}.png"
        img_p.write_bytes(b"\x89PNG\r\n\x1a\n")
        lbl_p = tmp_path / "labels" / it["split"] / f"{it['stem']}.txt"
        lbl_p.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    return items


def test_button_aliases_and_initial_setup(factory_dialog: YOLOFactoryDialog):
    """Verify button aliases and initial configuration."""
    assert factory_dialog.btn_bulk_approve is factory_dialog.btn_bulk
    assert factory_dialog.btn_bulk_prune_boxes is factory_dialog.btn_prune_low_confidence
    assert factory_dialog._auto_confirm_dialogs is True
    assert hasattr(factory_dialog, "btn_bulk_reject")
    assert hasattr(factory_dialog, "btn_live_relabel_filtered")
    assert hasattr(factory_dialog, "btn_live_relabel_all")
    assert callable(factory_dialog._bulk_reject_filtered)
    assert callable(factory_dialog._bulk_approve)
    assert callable(factory_dialog._bulk_prune_low_confidence_boxes)
    assert callable(factory_dialog._turbo_relabel_filtered)


def test_dynamic_button_labels_across_all_filter_modes(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that switching filter modes updates button text dynamically with ({n} kare) counts."""
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)

    # 1. Mode: suspicious (2 items: frame_susp_1, frame_susp_2)
    dlg.set_queue_filter_mode("suspicious")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 2
    assert dlg.btn_bulk_reject.text() == "✕ Tüm Şüphelileri Otomatik Reddet (2 kare)"
    assert dlg.btn_bulk.text() == "✓ Tüm Şüphelileri Onayla (2 kare)"
    assert dlg.btn_bulk_approve.text() == "✓ Tüm Şüphelileri Onayla (2 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Şüpheli Etiketleri Sil · Görseli Koru (2 kare)"
    assert dlg.btn_bulk_prune_boxes.text() == "✂️ Şüpheli Etiketleri Sil · Görseli Koru (2 kare)"
    assert dlg.btn_prune_low_confidence.isEnabled() is True
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Şüphelileri Canlı Yeniden Etiketle (2 kare)"

    # 2. Mode: empty (2 items: frame_empty_1, frame_empty_2)
    dlg.set_queue_filter_mode("empty")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 2
    assert dlg.btn_bulk_reject.text() == "✕ Tüm Etiketsizleri Otomatik Reddet (2 kare)"
    assert dlg.btn_bulk.text() == "✓ Etiketsizleri Onayla (2 kare)"
    assert dlg.btn_bulk_approve.text() == "✓ Etiketsizleri Onayla (2 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Etiketsiz Kutu Temizle (2 kare)"
    assert dlg.btn_prune_low_confidence.isEnabled() is False  # Pruning disabled when empty!
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Tüm Etiketsizleri Canlı Etiketle (2 kare)"

    # 3. Mode: confident (1 item: frame_conf_1)
    dlg.set_queue_filter_mode("confident")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 1
    assert dlg.btn_bulk_reject.text() == "✕ Filtredeki Eminleri Reddet (1 kare)"
    assert dlg.btn_bulk.text() == "✓ Tüm Eminleri Otomatik Onayla (1 kare)"
    assert dlg.btn_bulk_approve.text() == "✓ Tüm Eminleri Otomatik Onayla (1 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Düşük Güvenli Etiketleri Sil (1 kare)"
    assert dlg.btn_prune_low_confidence.isEnabled() is True
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Filtredeki Eminleri Canlı Etiketle (1 kare)"

    # 4. Mode: unlabeled_and_suspicious / unlabeled_or_suspicious (4 items: 2 susp + 2 empty)
    # 4. Mode: unlabeled_and_suspicious / unlabeled_or_suspicious (5 items: 2 susp + 2 empty + 1 mixed with weak)
    dlg.set_queue_filter_mode("unlabeled_and_suspicious")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 5
    assert dlg.btn_bulk_reject.text() == "✕ Etiketsiz & Şüphelileri Reddet (5 kare)"
    assert dlg.btn_bulk.text() == "✓ Etiketsiz & Şüphelileri Onayla (5 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Şüpheli Etiketleri Sil · Görseli Koru (5 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Etiketsiz & Şüphelileri Canlı Etiketle (5 kare)"

    dlg.set_queue_filter_mode("unlabeled_or_suspicious")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 5
    assert dlg.btn_bulk_reject.text() == "✕ Etiketsiz & Şüphelileri Reddet (5 kare)"

    # Mode: below (< 0.90: 5 items)
    dlg.bulk_range.setValue(90)
    dlg.set_queue_filter_mode("below")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 5
    assert dlg.btn_bulk_reject.text() == "✕ Eşik Altı Filtredekileri Reddet (5 kare)"
    assert dlg.btn_bulk.text() == "✓ Eşik Altı Filtredekileri Onayla (5 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Eşik Altı Etiketleri Sil · Görseli Koru (5 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Eşik Altı Filtredekileri Canlı Etiketle (5 kare)"

    # Mode: above (>= 0.90: 1 item frame_conf_1)
    dlg.set_queue_filter_mode("above")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 1
    assert dlg.btn_bulk_reject.text() == "✕ Eşik Üstü Filtredekileri Reddet (1 kare)"
    assert dlg.btn_bulk.text() == "✓ Eşik Üstü Filtredekileri Onayla (1 kare)"
    # NOTE (D89/F-2064): in 'above' mode the prune button must say "Eşik Üstü"
    # like its siblings. Production already says "Eşik Üstü" here
    # (ui/yolo_factory_dialog.py _update_filter_action_buttons, 'above' branch),
    # so this assertion pins the correct label against a copy-paste regression.
    assert dlg.btn_prune_low_confidence.text() == "✂️ Eşik Üstü Etiketleri Sil · Görseli Koru (1 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Eşik Üstü Filtredekileri Canlı Etiketle (1 kare)"

    # 5. Mode: mixed (1 item: frame_mixed_1)
    dlg.set_queue_filter_mode("mixed")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 1
    assert dlg.btn_bulk_reject.text() == "✕ Karışık Kareleri Reddet (1 kare)"
    assert dlg.btn_bulk.text() == "✓ Karışık Kareleri Onayla (1 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Sadece Şüpheli Etiketleri Sil · Görseli Koru (1 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Karışık Kareleri Canlı Etiketle (1 kare)"

    # 6. Mode: all (6 items total)
    dlg.set_queue_filter_mode("all")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 6
    assert dlg.btn_bulk_reject.text() == "✕ Filtredekileri Reddet (6 kare)"
    assert dlg.btn_bulk.text() == "✓ Filtredekileri Onayla (6 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ Düşük Güvenli Etiketleri Sil · Görseli Koru (6 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Filtredekileri Canlı Yeniden Etiketle (6 kare)"


def test_dynamic_button_labels_with_class_filter(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that activating a class filter updates scope tag in button labels."""
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)

    # Class 3: ore_orichalcum (appears in frame_susp_2 and frame_conf_1)
    idx = dlg.review_class_filter.findData(3)
    assert idx >= 0
    dlg.review_class_filter.setCurrentIndex(idx)
    dlg.set_queue_filter_mode("all")

    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 2
    assert dlg.btn_bulk_reject.text() == "✕ Filtredeki [ore_orichalcum] Karelerini Reddet (2 kare)"
    assert dlg.btn_bulk.text() == "✓ Filtredeki [ore_orichalcum] Karelerini Onayla (2 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ [ore_orichalcum] Düşük Etiketleri Sil · Görseli Koru (2 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Filtredeki [ore_orichalcum] Karelerini Canlı Etiketle (2 kare)"

    # Reset class filter
    all_idx = dlg.review_class_filter.findData(-1)
    dlg.review_class_filter.setCurrentIndex(all_idx)


def test_dynamic_button_labels_with_search_filter(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that typing a search query updates the scope tag and counts."""
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)

    dlg.review_search_edit.setText("conf")
    dlg.set_queue_filter_mode("all")

    filtered = dlg._filtered_review_indices()
    assert len(filtered) == 1
    assert dlg.btn_bulk_reject.text() == "✕ Filtredeki [conf] Karelerini Reddet (1 kare)"
    assert dlg.btn_bulk.text() == "✓ Filtredeki [conf] Karelerini Onayla (1 kare)"
    assert dlg.btn_prune_low_confidence.text() == "✂️ [conf] Düşük Etiketleri Sil · Görseli Koru (1 kare)"
    assert dlg.btn_live_relabel_all.text() == "🤖 Tümünü Canlı Yeniden Etiketle (6 kare)"  # F-0606 [O3]: "Tümünü" butonu filtre kapsamlı metin almaz, tüm kuyruğu sayar
    assert dlg.btn_live_relabel_filtered.text() == "🎯 Filtredeki [conf] Karelerini Canlı Etiketle (1 kare)"

    dlg.review_search_edit.clear()


def test_strict_filter_scope_isolation_bulk_reject(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that _bulk_reject_filtered strictly deletes ONLY filtered photos and leaves all others untouched."""
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)
    initial_total = len(dlg._review_items)
    assert initial_total == 6

    # Filter strictly to suspicious (frame_susp_1, frame_susp_2)
    dlg.set_queue_filter_mode("suspicious")
    assert len(dlg._filtered_review_indices()) == 2

    # Execute bulk reject
    dlg._bulk_reject_filtered()

    # The 2 suspicious items must be deleted
    remaining_stems = [it["stem"] for it in dlg._review_items]
    assert "frame_susp_1" not in remaining_stems
    assert "frame_susp_2" not in remaining_stems

    # The other 4 items (confident, empty_1, empty_2, mixed_1) MUST BE 100% PRESERVED
    assert "frame_conf_1" in remaining_stems
    assert "frame_empty_1" in remaining_stems
    assert "frame_empty_2" in remaining_stems
    assert "frame_mixed_1" in remaining_stems
    assert len(remaining_stems) == 4

    # Button count for suspicious should now be 0
    assert dlg.btn_bulk_reject.text() == "✕ Tüm Şüphelileri Otomatik Reddet (0 kare)"


def test_strict_filter_scope_isolation_bulk_approve(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that _bulk_approve strictly approves ONLY filtered items."""
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)
    by_stem = {it["stem"]: it for it in dlg._review_items}

    # 1. Filter to confident: only frame_conf_1
    dlg.set_queue_filter_mode("confident")
    assert len(dlg._filtered_review_indices()) == 1

    dlg._bulk_approve()

    # Verify that ONLY frame_conf_1 was approved (done == "ok") and removed from pending queue
    assert by_stem["frame_conf_1"]["done"] == "ok"
    assert by_stem["frame_conf_1"] not in dlg._review_items
    assert by_stem["frame_susp_1"]["done"] is None
    assert by_stem["frame_susp_2"]["done"] is None
    assert by_stem["frame_empty_1"]["done"] is None
    assert by_stem["frame_empty_2"]["done"] is None
    assert by_stem["frame_mixed_1"]["done"] is None

    # 2. Filter to suspicious: frame_susp_1 and frame_susp_2
    dlg.set_queue_filter_mode("suspicious")
    assert len(dlg._filtered_review_indices()) == 2

    dlg._bulk_approve()

    # F-0609 [O3]: suspicious-mode bulk approve no longer promotes
    # sub-threshold boxes — frame_susp_1 (0.42) and frame_susp_2 (0.55)
    # stay pending at the default 0.90 threshold instead of done="ok".
    assert by_stem["frame_susp_1"]["done"] is None
    assert len(by_stem["frame_susp_1"]["weak"]) == 1
    assert len(by_stem["frame_susp_1"]["trusted"]) == 0
    assert by_stem["frame_susp_2"]["done"] is None
    assert len(by_stem["frame_susp_2"]["weak"]) == 1
    assert len(by_stem["frame_susp_2"]["trusted"]) == 0

    # Empty and mixed items must remain untouched (done is None)
    assert by_stem["frame_empty_1"]["done"] is None
    assert by_stem["frame_empty_2"]["done"] is None
    assert by_stem["frame_mixed_1"]["done"] is None


def test_strict_filter_scope_isolation_bulk_prune(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that _bulk_prune_low_confidence_boxes strictly prunes within filtered indices."""
    dlg = factory_dialog

    # Item A: inside filter with 1 weak box (conf 0.30) and 1 high box (conf 0.95)
    # Item B: outside filter with 1 weak box (conf 0.25)
    dlg._review_items = [
        {
            "stem": "target_item",
            "split": "train",
            "weak": [
                _box(0.30, 0, [10, 10, 30, 30]),
                _box(0.95, 3, [40, 40, 80, 80]),
            ],
            "trusted": [],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "target_item.json"),
        },
        {
            "stem": "other_item",
            "split": "train",
            "weak": [_box(0.25, 1, [10, 10, 20, 20])],
            "trusted": [],
            "done": None,
            "path": str(tmp_path / "review" / "train" / "other_item.json"),
        },
    ]

    # Filter to only target_item via search edit
    dlg.review_search_edit.setText("target_item")
    assert len(dlg._filtered_review_indices()) == 1

    dlg._bulk_prune_low_confidence_boxes()

    # Target item was pruned: 0.30 box removed, 0.95 box promoted to trusted
    target = dlg._review_items[0]
    assert len(target["weak"]) == 0
    assert len(target["trusted"]) == 1
    assert target["trusted"][0]["class_id"] == 3

    # Other item was OUTSIDE filter and must remain completely untouched!
    other = dlg._review_items[1]
    assert len(other["weak"]) == 1
    assert other["weak"][0]["confidence"] == 0.25
    assert other["done"] is None


def test_bulk_reject_below_compatibility_alias(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that _bulk_reject_below delegates directly to _bulk_reject_filtered."""
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)

    dlg.set_queue_filter_mode("empty")
    assert len(dlg._filtered_review_indices()) == 2

    # Call legacy alias
    dlg._bulk_reject_below()

    remaining_stems = [it["stem"] for it in dlg._review_items]
    assert "frame_empty_1" not in remaining_stems
    assert "frame_empty_2" not in remaining_stems
    assert "frame_conf_1" in remaining_stems
    assert "frame_susp_1" in remaining_stems


def test_filter_change_events_trigger_button_updates(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verify that GUI events (combobox change, slider change) automatically refresh button labels.

    W4 (57120c9): the queue_filter combo follows prototype panel-2 status
    modes (all/pending/ok/no/high/low) — the old suspicious/empty modes are
    still supported programmatically via set_queue_filter_mode (covered by
    test_dynamic_button_labels_across_all_filter_modes) but are NOT combo
    items, so findData("suspicious") is -1 and setCurrentIndex(-1) would
    silently fall back to "all". Drive the combo through its real items
    instead: pending (all 6 fixture items, done=None) then ok (none
    approved) — the (6 kare) -> (0 kare) label change proves the
    currentIndexChanged event refreshed the buttons.
    """
    dlg = factory_dialog
    dlg._review_items = _create_sample_review_items(tmp_path)

    # 1. ComboBox change: pending -> ok
    idx_pending = dlg.queue_filter.findData("pending")
    assert idx_pending >= 0
    dlg.queue_filter.setCurrentIndex(idx_pending)
    assert dlg._queue_filter_mode == "pending"
    assert len(dlg._filtered_review_indices()) == 6
    assert dlg.btn_bulk_reject.text() == "✕ Filtredekileri Reddet (6 kare)"

    idx_ok = dlg.queue_filter.findData("ok")
    assert idx_ok >= 0
    dlg.queue_filter.setCurrentIndex(idx_ok)
    assert dlg._queue_filter_mode == "ok"
    assert len(dlg._filtered_review_indices()) == 0
    assert dlg.btn_bulk_reject.text() == "✕ Filtredekileri Reddet (0 kare)"

    # 2. Threshold slider change
    dlg.bulk_range.setValue(95)
    assert dlg.bulk_val.text() == "95"
