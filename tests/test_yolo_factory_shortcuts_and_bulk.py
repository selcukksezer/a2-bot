# -*- coding: utf-8 -*-
"""
tests/test_yolo_factory_shortcuts_and_bulk.py

Comprehensive Test Suite for Worker Y1:
1. 'A' (Onayla) and 'R' (Reddet) Review Shortcuts:
   - Bulletproof QShortcut and keyPressEvent / eventFilter fallback.
   - Input focus guard: text entry widgets (QLineEdit, QTextEdit, QSpinBox) suppress shortcuts.
   - Advancing review queue on approve/reject.
2. Non-Freezing Bulk Operations & Batching:
   - Batch deletion of 50+ items with periodic QApplication.processEvents yielding.
   - Strict filter isolation: only targeted files deleted, all others 100% untouched.
   - Single UI refresh at batch completion.
3. 3-Tier Canvas & Detail Row Color Coding:
   - 🟢 Emin (>= 85% or trusted): Green #22c55e, badge "✓ cname %pct"
   - 🟠 Şüpheli (70% - 84%): Orange #fb923c, badge "🟠 şüpheli cname %pct"
   - 🔴 Düşük Güven (< 70%): Red #ef4444, white text #ffffff badge, row status "Düşük Güven (Şüpheli)"
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QKeyEvent, QKeySequence
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLineEdit, QSpinBox

# Headless QApplication instance
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import YOLOFactoryDialog, YOLOPreviewCanvas


@contextmanager
def _counting_process_events() -> Iterator[Dict[str, int]]:
    """Count QApplication.processEvents calls inside the block.

    Both bulk tests spied on processEvents with an identical closure
    (~11 lines each). One shared helper keeps the yield-count assertions
    while removing the duplication; the counter dict is yielded so the
    tests read ``counter["calls"]`` after the block.
    """
    counter = {"calls": 0}
    orig_process_events = QApplication.processEvents

    def counted_process_events(*args: object, **kwargs: object) -> object:
        counter["calls"] += 1
        return orig_process_events(*args, **kwargs)

    with patch.object(QApplication, "processEvents", side_effect=counted_process_events):
        yield counter


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """Fixture providing an initialized YOLOFactoryDialog in Step 2 with test dataset directory."""
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dlg._models_dir = lambda: models_dir  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    dlg._go_step(2)
    # [YOLO-ASYNC] Adım 2 kuyruğu worker'da dolar; test gövdesi kuyruğu
    # senkron okumadan önce yüklemenin bitmesini bekle, sonra tmp_path
    # taramasını senkron tazele (başlangıç yüklemesi üretim dizinini
    # taramış olabilir — _go_step isteği düşmüşse kuyruk üretim verisi kalır).
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    dlg._reload_review_queue()
    return dlg


def _box(confidence: float, class_id: int, bbox: List[float] | None = None) -> Dict[str, Any]:
    """One fixture box in the canonical cx/cy schema (B-04).

    ``bbox`` is [x1, y1, x2, y2] on a 100px canvas (all fixture corners
    are <= 95); the normalizer only accepts normalized cx/cy/w/h, so a
    bbox-only (or geometry-less) box is rejected as "cx eksik" and can
    never persist. Boxes without a bbox get a centered default geometry.
    The ``bbox`` key is kept for traceability; persist reads cx/cy/w/h.
    """
    box: Dict[str, Any] = {"confidence": confidence, "class_id": class_id}
    if bbox is not None:
        x1, y1, x2, y2 = (float(v) for v in bbox)
        box["bbox"] = [x1, y1, x2, y2]
        box["cx"] = round((x1 + x2) / 200.0, 6)
        box["cy"] = round((y1 + y2) / 200.0, 6)
        box["w"] = round((x2 - x1) / 100.0, 6)
        box["h"] = round((y2 - y1) / 100.0, 6)
    else:
        box["cx"] = 0.5
        box["cy"] = 0.5
        box["w"] = 0.2
        box["h"] = 0.2
    return box


def _make_dummy_sample(
    tmp_path: Path,
    stem: str,
    split: str = "train",
    weak: List[Dict[str, Any]] | None = None,
    trusted: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Helper creating real files on disk and returning a review item dict."""
    weak_boxes = weak if weak is not None else []
    trusted_boxes = trusted if trusted is not None else []
    json_path = tmp_path / "review" / split / f"{stem}.json"
    json_path.write_text("{}", encoding="utf-8")
    # A REAL decodable PNG (not dummy bytes no decoder ever reads):
    # the preview canvas only forms its annotations from a decodable
    # image, and single approve promotes weak->trusted through those
    # canvas annotations — with dummy bytes the canvas stays empty and
    # approve can never yield done="ok" with a promoted box.
    import cv2
    import numpy as np

    img_path = tmp_path / "images" / split / f"{stem}.png"
    assert cv2.imwrite(str(img_path), np.full((200, 200, 3), (60, 120, 180), dtype=np.uint8))
    decoded = cv2.imread(str(img_path))
    assert decoded is not None and decoded.shape == (200, 200, 3)
    lbl_path = tmp_path / "labels" / split / f"{stem}.txt"
    lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    return {
        "stem": stem,
        "split": split,
        "weak": weak_boxes,
        "trusted": trusted_boxes,
        "done": None,
        "path": str(json_path),
    }


# ============================================================================
# 1. Review Shortcuts ('A' Approve and 'R' Reject) Tests
# ============================================================================

def test_shortcuts_exist_and_alias_coverage(factory_dialog: YOLOFactoryDialog):
    """Verify that shortcuts exist with WindowShortcut context and aliases are intact."""
    dlg = factory_dialog
    assert hasattr(dlg, "_shortcut_approve")
    assert hasattr(dlg, "_shortcut_reject")
    assert hasattr(dlg, "_shortcut_approve_a")
    assert hasattr(dlg, "_shortcut_approve_lower")
    assert hasattr(dlg, "_shortcut_reject_r")
    assert hasattr(dlg, "_shortcut_reject_lower")
    assert dlg.edit_search is dlg.review_search_edit
    assert callable(dlg._on_step2_approve)
    assert callable(dlg._on_step2_reject)
    assert callable(dlg._is_step_2_active)
    assert callable(dlg._is_text_input_focused)


def test_approve_shortcut_a_advances_queue(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Pressing 'A' approves the current item, promotes weak boxes, and advances review index."""
    dlg = factory_dialog
    item0 = _make_dummy_sample(tmp_path, "frame_001", weak=[_box(0.65, 0, [10, 10, 50, 50])])
    item1 = _make_dummy_sample(tmp_path, "frame_002", weak=[_box(0.55, 1, [20, 20, 60, 60])])
    dlg._review_items = [item0, item1]
    dlg._review_index = 0
    dlg._render_queue_list()

    # Verify initial state
    assert dlg._review_index == 0
    assert dlg._review_items[0]["done"] is None
    assert len(dlg._review_items[0]["weak"]) == 1
    assert len(dlg._review_items[0]["trusted"]) == 0

    # Simulate pressing 'A' via direct method and keyPressEvent
    key_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "A")
    dlg.keyPressEvent(key_event)

    # Item 0 should be approved (done="ok") and weak box promoted to trusted
    assert item0["done"] == "ok"
    assert len(item0["weak"]) == 0
    assert len(item0["trusted"]) == 1
    # Review queue should have popped item0 and shifted item1 to index 0
    assert len(dlg._review_items) == 1
    assert dlg._review_items[0]["stem"] == "frame_002"
    assert dlg._review_index == 0

    # Second press 'a' (lowercase) approves item 1
    key_event_lower = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "a")
    dlg.keyPressEvent(key_event_lower)
    assert item1["done"] == "ok"
    assert len(dlg._review_items) == 0


def test_reject_shortcut_r_deletes_and_advances(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Pressing 'R' deletes the current item from disk and advances the review queue."""
    dlg = factory_dialog
    item0 = _make_dummy_sample(tmp_path, "del_001", weak=[_box(0.40, 0)])
    item1 = _make_dummy_sample(tmp_path, "del_002", weak=[_box(0.45, 1)])
    dlg._review_items = [item0, item1]
    dlg._review_index = 0
    dlg._render_queue_list()

    p_img0 = tmp_path / "images" / "train" / "del_001.png"
    p_json0 = tmp_path / "review" / "train" / "del_001.json"
    assert p_img0.exists()
    assert p_json0.exists()

    # Trigger rejection via keyPressEvent ('R')
    key_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier, "R")
    dlg.keyPressEvent(key_event)

    # Item 0 files should be deleted from disk
    assert not p_img0.exists()
    assert not p_json0.exists()

    # Item 0 should be removed from memory
    remaining = [it["stem"] for it in dlg._review_items]
    assert "del_001" not in remaining
    assert "del_002" in remaining
    assert len(dlg._review_items) == 1
    assert dlg._review_index == 0


def test_text_input_focus_guard_prevents_shortcuts(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """When a text entry widget (such as edit_search) has keyboard focus, pressing 'A' or 'R' types text and does NOT trigger review actions."""
    dlg = factory_dialog
    item0 = _make_dummy_sample(tmp_path, "guard_001", weak=[_box(0.50, 0)])
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()

    # Give focus to the search edit
    dlg.show()
    _app.setActiveWindow(dlg)
    dlg.edit_search.setFocus()
    QApplication.processEvents()
    assert dlg._is_text_input_focused() is True

    # Sending 'A' to the dialog while edit_search has focus should be ignored by the review logic
    key_event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "A")
    dlg.keyPressEvent(key_event)

    # Item 0 must remain unapproved and NOT deleted
    assert dlg._review_items[0]["done"] is None
    assert len(dlg._review_items) == 1

    # Sending 'R' to the dialog while edit_search has focus should be ignored
    key_event_r = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_R, Qt.KeyboardModifier.NoModifier, "R")
    dlg.keyPressEvent(key_event_r)
    assert dlg._review_items[0]["done"] is None
    assert len(dlg._review_items) == 1
    assert (tmp_path / "images" / "train" / "guard_001.png").exists()

    # Clear focus
    dlg.edit_search.clearFocus()
    QApplication.processEvents()
    assert dlg._is_text_input_focused() is False


def test_live_window_shortcuts_trigger_actions(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verify that pressing 'A', 'R', 'S' in a live window triggers the
    REAL review actions (no mocked handlers) and is never ambiguous.

    'A' approves the current item for real (done="ok", item leaves the
    queue), 'R' rejects it for real (files deleted from disk, item leaves
    the queue), 'S' skips to the next item for real (index advances, the
    skipped item stays unapproved).
    """
    dlg = factory_dialog
    dlg.show()
    QApplication.processEvents()
    assert dlg._is_step_2_active()

    item_a = _make_dummy_sample(tmp_path, "live_001", weak=[_box(0.65, 0)])
    item_b = _make_dummy_sample(tmp_path, "live_002", weak=[_box(0.55, 1)])
    item_c = _make_dummy_sample(tmp_path, "live_003", weak=[_box(0.60, 0)])
    item_d = _make_dummy_sample(tmp_path, "live_004", weak=[_box(0.60, 1)])
    dlg._review_items = [item_a, item_b, item_c, item_d]
    dlg._review_index = 0
    dlg._render_queue_list()

    # 'A' when btn_approve has focus: real approve of live_001
    dlg.btn_approve.setFocus()
    QApplication.processEvents()
    QTest.keyPress(dlg.btn_approve, Qt.Key.Key_A)
    QApplication.processEvents()
    assert item_a["done"] == "ok"
    assert [it["stem"] for it in dlg._review_items] == ["live_002", "live_003", "live_004"]
    assert dlg._review_index == 0

    # 'R' when queue_list has focus: real reject of live_002 (files deleted)
    dlg.queue_list.setFocus()
    QApplication.processEvents()
    QTest.keyPress(dlg.queue_list, Qt.Key.Key_R)
    QApplication.processEvents()
    assert not (tmp_path / "images" / "train" / "live_002.png").exists()
    assert not (tmp_path / "review" / "train" / "live_002.json").exists()
    assert [it["stem"] for it in dlg._review_items] == ["live_003", "live_004"]
    assert dlg._review_index == 0

    # 'S' when review_preview has focus: real skip to live_004
    dlg.review_preview.setFocus()
    QApplication.processEvents()
    QTest.keyPress(dlg.review_preview, Qt.Key.Key_S)
    QApplication.processEvents()
    assert dlg._review_index == 1
    assert dlg._review_items[dlg._review_index]["stem"] == "live_004"
    assert item_c["done"] is None, "Skipped item must stay unapproved"


# ============================================================================
# 2. Non-Freezing Bulk Operations & Batching Tests
# ============================================================================

def test_bulk_reject_batching_responsiveness_and_isolation(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Test batch bulk rejection on a simulated queue of 50+ items:
    1. Verifies periodic processEvents yielding (non-freezing).
    2. Deletes strictly the target files on disk.
    3. Leaves non-targeted items 100% untouched.
    4. Refreshes UI in a single batch.
    """
    dlg = factory_dialog
    total_empty = 50
    total_confident = 10

    items: List[Dict[str, Any]] = []
    # 50 empty items
    for i in range(total_empty):
        items.append(_make_dummy_sample(tmp_path, f"bulk_empty_{i:03d}", weak=[], trusted=[]))
    # 10 confident items
    for i in range(total_confident):
        items.append(_make_dummy_sample(tmp_path, f"bulk_conf_{i:03d}", weak=[], trusted=[_box(0.95, 3)]))

    dlg._review_items = items
    dlg.set_queue_filter_mode("empty")
    filtered = dlg._filtered_review_indices()
    assert len(filtered) == total_empty

    # Spy on QApplication.processEvents to verify non-blocking yielding
    with _counting_process_events() as counter:
        dlg._bulk_reject_filtered()
    process_events_calls = counter["calls"]

    # processEvents must yield at least once per 15-item chunk (plus the
    # final flush): ceil(50 / 15) == 4. The old `>= total_empty // 15` (= 3)
    # also passed if the loop yielded only every ~17 items.
    assert process_events_calls == (total_empty + 14) // 15, (
        f"Expected one processEvents yield per 15-item chunk (4 for 50 items), got {process_events_calls}"
    )

    # All 50 empty items must be deleted from memory and disk
    remaining_stems = [it["stem"] for it in dlg._review_items]
    assert len(remaining_stems) == total_confident
    for i in range(total_empty):
        stem = f"bulk_empty_{i:03d}"
        assert stem not in remaining_stems
        assert not (tmp_path / "images" / "train" / f"{stem}.png").exists()
        assert not (tmp_path / "labels" / "train" / f"{stem}.txt").exists()
        assert not (tmp_path / "review" / "train" / f"{stem}.json").exists()

    # All 10 confident items must be 100% PRESERVED
    for i in range(total_confident):
        stem = f"bulk_conf_{i:03d}"
        assert stem in remaining_stems
        assert (tmp_path / "images" / "train" / f"{stem}.png").exists()
        assert (tmp_path / "labels" / "train" / f"{stem}.txt").exists()
        assert (tmp_path / "review" / "train" / f"{stem}.json").exists()

    # Filter count should update to 0 in button label
    assert dlg.btn_bulk_reject.text() == "✕ Tüm Etiketsizleri Otomatik Reddet (0 kare)"


def test_bulk_approve_batching_and_responsiveness(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Test batch bulk approval on 35+ suspicious items with periodic processEvents yielding."""
    dlg = factory_dialog
    n_items = 35
    items = []
    for i in range(n_items):
        items.append(_make_dummy_sample(tmp_path, f"susp_{i:03d}", weak=[_box(0.50, 1)]))

    dlg._review_items = items
    dlg.set_queue_filter_mode("suspicious")
    assert len(dlg._filtered_review_indices()) == n_items

    with _counting_process_events() as counter:
        dlg._bulk_approve()
    process_events_calls = counter["calls"]

    assert process_events_calls >= (n_items // 15)
    # F-0609 [O3]: suspicious-mode bulk approve no longer approves
    # sub-threshold boxes. All 35 items stay pending: the conf-0.50 weak
    # boxes are kept (not promoted), nothing is marked done="ok".
    assert all(it.get("done") is None for it in dlg._review_items)
    assert all(len(it["trusted"]) == 0 for it in dlg._review_items)
    assert all(len(it["weak"]) == 1 for it in dlg._review_items)


# ============================================================================
# 3. 3-Tier Visual Color Coding Tests
# ============================================================================

def test_canvas_and_detail_rows_3_tier_color_coding(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verify 3-tier color coding:
    Tier 1 (Emin): Green (#22c55e), solid line, '✓ cname %pct'
    Tier 2 (Şüpheli 70-84%): Orange (#fb923c), dashed line, '🟠 şüpheli cname %pct'
    Tier 3 (Düşük Güven < 70%): Red (#ef4444), white text badge, status 'Düşük Güven (Şüpheli)'
    """
    dlg = factory_dialog

    # 1. Tier 1: Confident box (norm_conf = 0.95, trusted=True)
    det_emin = {"class_name": "ore_orichalcum", "class_id": 3, "confidence": 0.95, "trusted": True}
    row_emin = dlg._create_review_det_row(det_emin, is_trusted=True)
    conf_lbl_emin = row_emin.findChild(object, "PipeDetConf")
    assert conf_lbl_emin is not None
    assert "#4ade80" in conf_lbl_emin.styleSheet()

    # 2. Tier 2: Suspicious box (norm_conf = 0.78, trusted=False)
    det_susp = {"class_name": "herb_aria", "class_id": 0, "confidence": 0.78, "trusted": False}
    row_susp = dlg._create_review_det_row(det_susp, is_trusted=False)
    conf_lbl_susp = row_susp.findChild(object, "PipeDetConf")
    assert conf_lbl_susp is not None
    assert "#fb923c" in conf_lbl_susp.styleSheet()

    # 3. Tier 3: Low confidence box (norm_conf = 0.42, trusted=False)
    det_low = {"class_name": "herb_mela", "class_id": 1, "confidence": 0.42, "trusted": False}
    row_low = dlg._create_review_det_row(det_low, is_trusted=False)
    conf_lbl_low = row_low.findChild(object, "PipeDetConf")
    assert conf_lbl_low is not None
    assert "#ef4444" in conf_lbl_low.styleSheet()

    # Check status labels
    labels_emin = [w.text() for w in row_emin.findChildren(object) if hasattr(w, "text")]
    assert any("Emin · %95 güven" in t for t in labels_emin)

    labels_susp = [w.text() for w in row_susp.findChildren(object) if hasattr(w, "text")]
    assert any("Şüpheli · %78 güven" in t for t in labels_susp)

    labels_low = [w.text() for w in row_low.findChildren(object) if hasattr(w, "text")]
    assert any("Düşük Güven (Şüpheli) · %42 güven" in t for t in labels_low)


# ============================================================================
# 4. Header Refresh & Persistent Deletion Across Splits Tests
# ============================================================================

def test_btn_refresh_triggers_forced_dashboard_and_queue_reload(factory_dialog: YOLOFactoryDialog):
    """Verify that clicking '↻ Yenile' triggers on_refresh_clicked and reloads Step 2 queue.

    [YOLO-ASYNC] Yenile artık senkron `_reload_review_queue` yerine async
    `_request_queue_reload` yolunu kullanır; mock da o yolu izler.
    """
    dlg = factory_dialog
    reload_called = False
    refresh_force = None

    orig_refresh = dlg.refresh_dashboard

    def mock_reload(*args, **kwargs):
        nonlocal reload_called
        reload_called = True
        return True

    def mock_refresh(force=False, *args):
        nonlocal refresh_force
        refresh_force = force
        return orig_refresh(force=force, *args)

    with patch.object(dlg, "_request_queue_reload", side_effect=mock_reload):
        with patch.object(dlg, "refresh_dashboard", side_effect=mock_refresh):
            dlg.btn_refresh.click()

    assert reload_called is True
    assert refresh_force is True


def test_rejected_sample_purged_across_splits_and_persisted(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verify that rejecting a sample:
    1. Purges files across BOTH train and val splits.
    2. Writes stem to rejected_stems.json.
    3. Prevents sample from ever returning on _reload_review_queue().
    4. Auto-cleans orphan image if recreated on disk.
    """
    dlg = factory_dialog
    stem = "frame_reject_test_001"

    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "val").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "review" / "val").mkdir(parents=True, exist_ok=True)

    # Place files in train AND an orphan file in val
    (tmp_path / "images" / "train" / f"{stem}.png").write_bytes(b"img_train")
    (tmp_path / "images" / "val" / f"{stem}.png").write_bytes(b"img_val")
    (tmp_path / "labels" / "train" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (tmp_path / "labels" / "val" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (tmp_path / "review" / "train" / f"{stem}.json").write_text("{}", encoding="utf-8")
    (tmp_path / "review" / "val" / f"{stem}.json").write_text("{}", encoding="utf-8")

    sample = {
        "stem": stem,
        "split": "train",
        "weak": [_box(0.50, 0)],
        "trusted": [],
        "done": None,
        "path": str(tmp_path / "review" / "train" / f"{stem}.json"),
    }
    dlg._review_items = [sample]
    dlg._review_index = 0

    # Reject sample
    dlg._on_step2_reject()

    # 1. Assert files deleted in BOTH train and val
    assert not (tmp_path / "images" / "train" / f"{stem}.png").exists()
    assert not (tmp_path / "images" / "val" / f"{stem}.png").exists()
    assert not (tmp_path / "labels" / "train" / f"{stem}.txt").exists()
    assert not (tmp_path / "labels" / "val" / f"{stem}.txt").exists()
    assert not (tmp_path / "review" / "train" / f"{stem}.json").exists()
    assert not (tmp_path / "review" / "val" / f"{stem}.json").exists()

    # 2. Assert stem recorded in persistent rejected_stems.json
    rejected_path = tmp_path / "rejected_stems.json"
    assert rejected_path.is_file()
    assert stem in dlg._load_rejected_stems()

    # 3. Reload queue - verify empty
    dlg._reload_review_queue()
    assert all(it["stem"] != stem for it in dlg._review_items)

    # 4. If an orphan file appears on disk with rejected stem, reload should delete it and NOT resurrect it
    (tmp_path / "images" / "train" / f"{stem}.png").write_bytes(b"orphan")
    dlg._reload_review_queue()
    assert not (tmp_path / "images" / "train" / f"{stem}.png").exists()
    assert all(it["stem"] != stem for it in dlg._review_items)


def test_stat_model_displays_date_and_time(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verify that the 'SON MODEL' KPI card subtext displays the date and time of the last model."""
    dlg = factory_dialog
    from datetime import datetime
    import time

    # Mock model file
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_pt = models_dir / "best.pt"
    best_pt.write_bytes(b"fake_model_weights")

    # Set known timestamp (e.g. 2026-11-08 10:23)
    target_dt = datetime(2026, 11, 8, 10, 23, 0)
    target_ts = target_dt.timestamp()
    import os
    os.utime(best_pt, (target_ts, target_ts))
    dlg._pipeline["train_completed_at"] = ""

    with patch.object(dlg, "_models_dir", return_value=models_dir):
        dlg.refresh_dashboard(force=True)

    # Subtext should contain formatted date and time
    assert "08.11.2026 10:23" in dlg.stat_model_sub.text()
    assert dlg.stat_model_value.text() == "best.pt"


