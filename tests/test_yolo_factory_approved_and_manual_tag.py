# -*- coding: utf-8 -*-
"""
tests/test_yolo_factory_approved_and_manual_tag.py

Comprehensive Test Suite for Worker A_UI:
Division 1: Approved Pool UI Wiring (ApprovedDatasetPool, lbl_approved_count, reload skipping, only_approved)
Division 2: Manual Box Drawing (Rubber-band, crosshair, boxDrawn/boxAdded signals) & Searchable Tag List (25 classes, filtering, exemplar learning)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Headless QApplication instance
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from dataset_factory.approved_pool import ApprovedDatasetPool, get_approved_pool
from dataset_factory.class_taxonomy import CANONICAL_NAMES
from ui.yolo_factory_dialog import (
    CLASS_COLORS_MAP,
    CLASS_ID_TO_SPEC,
    ClassSelectDialog,
    RTXTrainingWorker,
    SearchableTagListPanel,
    YOLOFactoryDialog,
    YOLOPreviewCanvas,
)

_CANONICAL_CLASS_COUNT = len(CANONICAL_NAMES)


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """Fixture providing an initialized YOLOFactoryDialog in Step 2 with temporary dataset layout."""
    import dataset_factory.approved_pool as ap_mod

    prev_pool = ap_mod._GLOBAL_POOL
    test_pool = ApprovedDatasetPool(pool_root=tmp_path / "approved", project_root=tmp_path)
    ap_mod._GLOBAL_POOL = test_pool

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    (tmp_path / "approved" / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "approved" / "train" / "labels").mkdir(parents=True, exist_ok=True)
    dlg._go_step(2)
    # [YOLO-ASYNC] Adım 2 kuyruğu worker'da dolar; test gövdesi kuyruğu
    # senkron okumadan önce yüklemenin bitmesini bekle, sonra tmp_path
    # taramasını senkron tazele (başlangıç yüklemesi üretim dizinini
    # taramış olabilir — _go_step isteği düşmüşse kuyruk üretim verisi kalır).
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    dlg._reload_review_queue()
    yield dlg
    ap_mod._GLOBAL_POOL = prev_pool


def _create_dummy_image_and_review(
    tmp_path: Path,
    stem: str,
    split: str = "train",
    weak: List[Dict[str, Any]] | None = None,
    trusted: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Helper creating real files on disk for a sample."""
    weak_boxes = weak if weak is not None else []
    trusted_boxes = trusted if trusted is not None else []

    img_path = tmp_path / "images" / split / f"{stem}.png"
    import cv2
    import numpy as np
    dummy_bgr = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_path), dummy_bgr)

    lbl_path = tmp_path / "labels" / split / f"{stem}.txt"
    lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    json_path = tmp_path / "review" / split / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "stem": stem,
                "split": split,
                "trusted": trusted_boxes,
                "weak": weak_boxes,
                "done": None,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "stem": stem,
        "split": split,
        "weak": weak_boxes,
        "trusted": trusted_boxes,
        "done": None,
        "path": str(json_path),
    }


# ============================================================================
# Division 1: Approved Pool UI Wiring Tests
# ============================================================================

def test_approved_pool_wiring_and_counter_badge(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies ApprovedDatasetPool integration, badge update on sample approval, and persist to approved pool."""
    dlg = factory_dialog
    pool = get_approved_pool(tmp_path)
    initial_count = pool.get_approved_count()
    assert initial_count == 0

    item0 = _create_dummy_image_and_review(
        tmp_path,
        "sample_app_01",
        weak=[{"confidence": 0.65, "class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
    )
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()

    # Approve item via _on_step2_approve
    dlg._on_step2_approve()

    # Item must be approved in memory and persisted
    assert item0["done"] == "ok"
    assert pool.is_approved("sample_app_01")
    assert pool.get_approved_count() == 1
    assert "✓ 1 Onaylandı" in dlg.lbl_approved_count.text()

    # File must exist in approved pool directory
    approved_manifest = tmp_path / "approved" / "approved_manifest.json"
    assert approved_manifest.is_file()
    manifest_data = json.loads(approved_manifest.read_text(encoding="utf-8"))
    assert "sample_app_01" in manifest_data.get("stems", {})


def test_reload_review_queue_skips_approved_samples(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Samples already in ApprovedDatasetPool or marked done must be skipped across dialog reload/restarts."""
    dlg = factory_dialog
    pool = get_approved_pool(tmp_path)

    # Create 3 real samples on disk
    _create_dummy_image_and_review(tmp_path, "sample_pending_01")
    _create_dummy_image_and_review(tmp_path, "sample_pending_02")
    _create_dummy_image_and_review(tmp_path, "sample_already_approved")

    # Manually add sample_already_approved to pool
    pool.add_approved_sample(
        stem="sample_already_approved",
        split="train",
        image_path=tmp_path / "images" / "train" / "sample_already_approved.png",
        trusted_boxes=[{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
    )
    assert pool.is_approved("sample_already_approved")

    # Reload review queue from disk
    dlg._reload_review_queue()

    stems = [it["stem"] for it in dlg._review_items]
    assert "sample_pending_01" in stems
    assert "sample_pending_02" in stems
    assert "sample_already_approved" not in stems
    assert len(dlg._review_items) == 2


def test_remove_review_item_at_and_review_items_property(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that _remove_review_item_at and dlg.review_items property correctly manipulate review items."""
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "item_a")
    item1 = _create_dummy_image_and_review(tmp_path, "item_b")
    dlg.review_items = [item0, item1]
    assert len(dlg.review_items) == 2

    # Remove item at index 0
    dlg._remove_review_item_at(0)
    assert len(dlg.review_items) == 1
    assert dlg.review_items[0]["stem"] == "item_b"

    # Pop item
    popped = dlg.review_items.pop(0)
    assert popped["stem"] == "item_b"
    assert len(dlg.review_items) == 0


def test_approved_pool_filter_mode(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that queue_filter 'approved_pool' correctly filters approved samples."""
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "frame_normal")
    item1 = _create_dummy_image_and_review(tmp_path, "frame_approved")
    item1["done"] = "ok"
    item1["human_approved"] = True

    dlg._review_items = [item0, item1]
    dlg.set_queue_filter_mode("approved_pool")

    indices = dlg._filtered_review_indices()
    assert len(indices) == 1
    assert dlg._review_items[indices[0]]["stem"] == "frame_approved"


def test_only_approved_checkbox_and_training_wiring(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies chk_only_approved toggle and propagation of only_approved=True to training worker and freeze."""
    dlg = factory_dialog
    assert hasattr(dlg, "chk_only_approved")
    assert dlg.chk_only_approved.isChecked() is True
    assert hasattr(dlg, "chk_train_only_approved")
    assert hasattr(dlg, "btn_train_only_approved")
    assert "Sadece Onaylananlarla Eğit" in dlg.btn_train_only_approved.text()

    # Freeze dataset with only_approved — REAL ApprovedDatasetPool.prepare
    # (no mock): 5 genuinely approved samples are placed under
    # tmp/dataset/review + images, freeze must prepare the real approved
    # training set and write data_approved.yaml for exactly those 5.
    import cv2
    import numpy as np

    src_ds = tmp_path / "dataset"
    (src_ds / "review" / "train").mkdir(parents=True, exist_ok=True)
    (src_ds / "images" / "train").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for i in range(5):
        stem = f"freeze_appr_{i}"
        cv2.imwrite(
            str(src_ds / "images" / "train" / f"{stem}.png"),
            rng.integers(0, 255, (80, 80, 3), dtype=np.uint8),
        )
        (src_ds / "review" / "train" / f"{stem}.json").write_text(
            json.dumps(
                {
                    "stem": stem,
                    "split": "train",
                    "done": "ok",
                    "human_approved": True,
                    # P1-10 / K-21: admitted frames need per-box provenance.
                    "trusted": [{"class_id": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "human_approved": True}],
                    "image_path": str(src_ds / "images" / "train" / f"{stem}.png"),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    assert dlg.toggle_approved_only.isChecked() is True
    dlg._freeze_dataset()
    pool_stats = get_approved_pool(tmp_path).get_approved_stats()
    assert pool_stats["total_approved_images"] == 5
    assert pool_stats["train_count"] == 5
    yaml_path = tmp_path / "approved" / "data_approved.yaml"
    assert yaml_path.is_file()
    assert "train:" in yaml_path.read_text(encoding="utf-8")
    assert (tmp_path / "frozen_manifest.json").is_file()

    # Verify worker creation passes only_approved=True
    with patch.object(RTXTrainingWorker, "start"):
        worker = dlg._start_training_worker(epochs=1, only_approved=True)
        assert isinstance(worker, RTXTrainingWorker)
        assert worker.only_approved is True


# ============================================================================
# Division 2: Manual Box Drawing & Searchable Tag List Tests
# ============================================================================

def test_searchable_tag_list_panel_coverage_and_filtering(factory_dialog: YOLOFactoryDialog):
    """Verifies SearchableTagListPanel has all canonical classes (incl. kinah, player, F4 generics), QCompleter, and search filtering."""
    panel = factory_dialog.tag_panel
    assert isinstance(panel, SearchableTagListPanel)
    assert panel.list_widget.count() == _CANONICAL_CLASS_COUNT
    assert panel.combo.count() == _CANONICAL_CLASS_COUNT

    # Check that canonical classes exist, including pots, scroll, kinah, player, F4
    cnames = [panel.combo.itemText(i) for i in range(panel.combo.count())]
    assert any("herb_aria" in s for s in cnames)
    assert any("ore_orichalcum" in s for s in cnames)
    assert any("hp_potion" in s for s in cnames)
    assert any("mp_potion" in s for s in cnames)
    assert any("buff_scroll" in s for s in cnames)
    assert any("kinah" in s for s in cnames)
    assert any("player_character" in s for s in cnames)
    assert any("other_ore" in s for s in cnames)
    assert any("other_odyle" in s for s in cnames)

    # Check completer attached
    assert panel.combo.completer() is not None

    # Test search edit filtering
    panel.search_edit.setText("scroll")
    visible_count = sum(1 for i in range(panel.list_widget.count()) if not panel.list_widget.item(i).isHidden())
    assert visible_count == 1
    assert "buff_scroll" in panel.list_widget.item(24).text()

    panel.search_edit.setText("kinah")
    visible_count_kinah = sum(1 for i in range(panel.list_widget.count()) if not panel.list_widget.item(i).isHidden())
    assert visible_count_kinah == 1
    assert "kinah" in panel.list_widget.item(25).text()

    panel.search_edit.setText("")
    visible_all = sum(1 for i in range(panel.list_widget.count()) if not panel.list_widget.item(i).isHidden())
    assert visible_all == _CANONICAL_CLASS_COUNT


def test_drawing_class_selection_and_signals(factory_dialog: YOLOFactoryDialog):
    """Verifies that selecting a class in tag_panel emits classSelected and updates active drawing class."""
    dlg = factory_dialog
    panel = dlg.tag_panel

    received_signals = []
    panel.classSelected.connect(lambda cid, cname: received_signals.append((cid, cname)))

    # Select class 3: ore_orichalcum
    item = panel.list_widget.item(3)
    panel._on_item_clicked(item)

    assert len(received_signals) == 1
    assert received_signals[0][0] == 3
    assert received_signals[0][1] == "ore_orichalcum"
    assert dlg.active_drawing_class_id == 3
    assert dlg.active_drawing_class_name == "ore_orichalcum"
    assert dlg.review_preview.active_drawing_class_id == 3
    assert dlg.review_preview.active_drawing_class_name == "ore_orichalcum"
    assert "3: ore_orichalcum" in panel.lbl_active_pill.text()


def test_canvas_rubber_band_drag_and_signals(factory_dialog: YOLOFactoryDialog):
    """Verifies YOLOPreviewCanvas mouse drag emits boxDrawn and boxAdded with crosshair cursor."""
    canvas = factory_dialog.review_preview
    canvas._last_target_rect = canvas.rect()

    box_drawn_coords = []
    canvas.boxDrawn.connect(lambda cx, cy, w, h: box_drawn_coords.append((cx, cy, w, h)))

    box_added_list = []
    canvas.boxAdded.connect(lambda b: box_added_list.append(b))

    canvas.active_drawing_class_id = 3
    canvas.active_drawing_class_name = "ore_orichalcum"

    # Simulate LeftButton press at (50, 50)
    ev_press = QMouseEvent(QMouseEvent.Type.MouseButtonPress, QPointF(50, 50), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.mousePressEvent(ev_press)
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor
    assert canvas._drag_start_pos == QPointF(50, 50)

    # Simulate mouse move drag to (150, 150)
    ev_move = QMouseEvent(QMouseEvent.Type.MouseMove, QPointF(150, 150), Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    canvas.mouseMoveEvent(ev_move)
    assert canvas._is_dragging_box is True
    assert canvas._last_drag_distance >= 8.0

    # Simulate mouse release at (150, 150)
    ev_release = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(150, 150), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    canvas.mouseReleaseEvent(ev_release)

    assert len(box_drawn_coords) == 1
    assert len(box_added_list) == 1
    new_box = box_added_list[0]
    assert new_box["class_id"] == 3
    assert new_box["class_name"] == "ore_orichalcum"
    assert new_box["confidence"] == 1.0
    assert new_box["trusted"] is True
    assert new_box["hand_label"] is True


def test_canvas_drag_suppresses_lightbox(factory_dialog: YOLOFactoryDialog):
    """Verifies eventFilter returns True and suppresses lightbox when mouse drag distance >= 8px."""
    dlg = factory_dialog
    canvas = dlg.review_preview
    canvas._last_drag_distance = 15.0

    with patch.object(dlg, "_open_review_lightbox") as mock_lightbox:
        ev_release = QMouseEvent(QMouseEvent.Type.MouseButtonRelease, QPointF(150, 150), Qt.MouseButton.LeftButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        res = dlg.eventFilter(canvas, ev_release)
        # Event must be consumed and lightbox must NOT open
        assert res is True
        mock_lightbox.assert_not_called()


def test_manual_box_addition_and_exemplar_learning(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that newly drawn box is assigned confidence=1.0, learned into ExemplarMemory, and persisted.

    Exemplar learning is exercised for REAL (no mock of _learn_box_exemplar):
    the box uses canonical class 3 (ore_orichalcum) — class 24 buff_scroll
    is rejected by ExemplarMemory as non-canonical, so asserting learning
    for it would be false — on a textured image crop, and the exemplar
    store is redirected to tmp_path so the repo's learned_exemplars.json
    is untouched. The test then asserts a positive exemplar for class 3
    with source "manual_draw" is actually persisted.
    """
    import cv2
    import numpy as np

    import dataset_factory.exemplar_memory as ex_mod

    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "draw_test_frame")
    # Textured image: a uniform black crop yields no extractable features,
    # so the real learner would (correctly) record nothing.
    rng = np.random.default_rng(7)
    cv2.imwrite(
        str(tmp_path / "images" / "train" / "draw_test_frame.png"),
        rng.integers(0, 255, (100, 100, 3), dtype=np.uint8),
    )
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()

    dlg.active_drawing_class_id = 3
    dlg.active_drawing_class_name = "ore_orichalcum"

    box_data = {
        "cx": 0.45,
        "cy": 0.55,
        "w": 0.12,
        "h": 0.18,
        "class_id": 3,
        "class_name": "ore_orichalcum",
    }

    prev_exemplar_memory = ex_mod._GLOBAL_EXEMPLAR_MEMORY
    ex_path = tmp_path / "learned_exemplars.json"
    ex_mod.get_exemplar_memory(ex_path)
    try:
        dlg._on_manual_box_added(box_data)
    finally:
        ex_mod._GLOBAL_EXEMPLAR_MEMORY = prev_exemplar_memory

    # Box must be added to trusted list of current item
    assert len(item0["trusted"]) == 1
    added_box = item0["trusted"][0]
    assert added_box["class_id"] == 3
    assert added_box["class_name"] == "ore_orichalcum"
    assert added_box["confidence"] == 1.0
    assert added_box["hand_label"] is True
    assert added_box["human_approved"] is True
    assert added_box["trusted"] is True

    # Real exemplar learning: a positive exemplar for class 3 from the
    # manual draw must be persisted to the tmp exemplar store.
    assert ex_path.is_file(), "Exemplar store was never written — learning did not run"
    stored = json.loads(ex_path.read_text(encoding="utf-8"))
    positives = [e for e in stored.get("positive", []) if int(e.get("class_id", -1)) == 3]
    assert len(positives) >= 1, "No positive exemplar for class 3 was learned"
    assert any(e.get("source") == "manual_draw" for e in positives)


def test_detail_row_cls_combo_has_completer(factory_dialog: YOLOFactoryDialog):
    """Verifies that _create_review_det_row adds QCompleter and makes cls_combo editable."""
    dlg = factory_dialog
    det = {"class_id": 0, "class_name": "herb_aria", "confidence": 0.95}
    row = dlg._create_review_det_row(det, is_trusted=True)

    combo = row.findChild(type(dlg.manual_tag_combo), "PipeClassCombo")
    assert combo is not None
    assert combo.isEditable() is True
    assert combo.completer() is not None
    assert combo.count() == _CANONICAL_CLASS_COUNT


def test_manual_box_drawn_only_on_target_frame_and_persisted(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """
    Verifies that when a manual box is drawn on frame A:
    1. The box is persisted to frame A's review JSON and YOLO txt label file.
    2. The drag rubberband state is cleared immediately.
    3. Switching to frame B shows NO annotations from frame A, and no drag ghost box.
    4. Switching back to frame A properly shows frame A's drawn annotation.
    """
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "frame_alpha")
    item1 = _create_dummy_image_and_review(tmp_path, "frame_beta")

    dlg._review_items = [item0, item1]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)

    # Frame 0 is active
    assert dlg._review_index == 0
    canvas = dlg.review_preview
    assert canvas._drag_start_pos is None
    assert canvas._is_dragging_box is False

    # Draw a box on frame_alpha
    dlg.active_drawing_class_id = 3
    dlg.active_drawing_class_name = "ore_orichalcum"

    box_data = {
        "cx": 0.5,
        "cy": 0.5,
        "w": 0.2,
        "h": 0.2,
        "class_id": 3,
        "class_name": "ore_orichalcum",
    }

    with patch.object(dlg, "_learn_box_exemplar"):
        dlg._on_manual_box_added(box_data)

    # 1. Box added to item0 trusted
    assert len(item0["trusted"]) == 1
    assert item0["trusted"][0]["class_id"] == 3
    assert item0["trusted"][0]["hand_label"] is True

    # 2. Rubberband drag state is completely cleared
    assert canvas._drag_start_pos is None
    assert canvas._is_dragging_box is False

    # 3. Label file for frame_alpha contains class 3
    lbl_file_alpha = tmp_path / "labels" / "train" / "frame_alpha.txt"
    assert lbl_file_alpha.exists()
    content_alpha = lbl_file_alpha.read_text(encoding="utf-8")
    assert "3 0.500000 0.500000 0.200000 0.200000" in content_alpha

    # 4. Label file for frame_beta has NOT been touched with frame_alpha's box
    lbl_file_beta = tmp_path / "labels" / "train" / "frame_beta.txt"
    content_beta = lbl_file_beta.read_text(encoding="utf-8")
    assert "3 0.500000 0.500000 0.200000 0.200000" not in content_beta

    # 5. Switch to frame_beta (index 1)
    dlg._on_queue_row_changed(1)
    assert dlg._review_index == 1

    # Verify canvas annotations do NOT contain frame_alpha's drawn box
    ann_beta = canvas._annotations
    assert not any(a.get("class_id") == 3 and a.get("hand_label") for a in ann_beta)
    assert canvas._drag_start_pos is None
    assert canvas._is_dragging_box is False

    # 6. Switch back to frame_alpha (index 0)
    dlg._on_queue_row_changed(0)
    assert dlg._review_index == 0

    ann_alpha = canvas._annotations
    assert any(a.get("class_id") == 3 and a.get("hand_label") for a in ann_alpha)
    assert canvas._drag_start_pos is None
    assert canvas._is_dragging_box is False


def test_class_select_dialog_coverage_and_filtering():
    """Verifies ClassSelectDialog lists all canonical classes, filters on text input, and returns selected class."""
    dlg = ClassSelectDialog(parent=None, initial_class_id=3)
    assert dlg.list_widget.count() == _CANONICAL_CLASS_COUNT
    assert dlg.selected_class_id() == 3

    # Test filtering
    dlg.search_edit.setText("potion")
    visible = [dlg.list_widget.item(i) for i in range(dlg.list_widget.count()) if not dlg.list_widget.item(i).isHidden()]
    assert len(visible) == 2
    assert any("hp_potion" in it.text() for it in visible)
    assert any("mp_potion" in it.text() for it in visible)

    # Search return selects first match
    dlg._on_search_return()
    # "potion" matches exactly hp_potion (22) and mp_potion (23), in that
    # order; the return key must select the FIRST match (22) — accepting
    # either (22, 23) would hide a wrong-order / wrong-selection bug.
    visible_ids = [int(dlg.list_widget.item(i).data(Qt.ItemDataRole.UserRole)) for i in range(dlg.list_widget.count()) if not dlg.list_widget.item(i).isHidden()]
    assert visible_ids == [22, 23]
    assert dlg.selected_class_id() == 22


def test_delete_box_auto_persists_to_disk_and_clears_yolo_txt(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that clicking delete removes the box from trusted, and auto-saves clearing it from disk."""
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "del_test_frame")
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)

    # Add a box first
    dlg.active_drawing_class_id = 3
    box = {
        "cx": 0.4,
        "cy": 0.4,
        "w": 0.15,
        "h": 0.15,
        "class_id": 3,
        "class_name": "ore_orichalcum",
    }
    with patch.object(dlg, "_learn_box_exemplar"):
        dlg._on_manual_box_added(box)

    lbl_file = tmp_path / "labels" / "train" / "del_test_frame.txt"
    assert "3 0.400000 0.400000 0.150000 0.150000" in lbl_file.read_text(encoding="utf-8")
    assert len(item0["trusted"]) == 1

    # Delete the box using _reject_box
    with patch.object(dlg, "_learn_box_exemplar"):
        dlg._reject_box(item0["trusted"][0], from_trusted=True)

    # 1. Box is removed from trusted
    assert len(item0["trusted"]) == 0

    # 2. Disk label file is auto-persisted and no longer contains class 3
    content_after = lbl_file.read_text(encoding="utf-8")
    assert "3 0.400000 0.400000" not in content_after

    # 3. Review JSON on disk has empty trusted list
    rev_json = tmp_path / "review" / "train" / "del_test_frame.json"
    rev_data = json.loads(rev_json.read_text(encoding="utf-8"))
    assert len(rev_data.get("trusted", [])) == 0


def test_manual_box_class_selection_and_auto_save(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that when interactive class selection is triggered, user choice is saved automatically."""
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "prompt_test_frame")
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)

    # Mock user selecting class 24 (buff_scroll) in the dialog
    dlg._auto_confirm_dialogs = False
    with patch.object(dlg, "_prompt_class_selection", return_value=24) as mock_prompt, \
         patch.object(dlg, "_learn_box_exemplar"):
        box_coords = {"cx": 0.6, "cy": 0.6, "w": 0.1, "h": 0.1}
        dlg._on_manual_box_added(box_coords)
        mock_prompt.assert_called_once()

    assert len(item0["trusted"]) == 1
    added = item0["trusted"][0]
    assert added["class_id"] == 24
    assert added["class_name"] == "buff_scroll"
    assert dlg.active_drawing_class_id == 24

    # Verify auto-saved to disk immediately
    lbl_file = tmp_path / "labels" / "train" / "prompt_test_frame.txt"
    assert "24 0.600000 0.600000 0.100000 0.100000" in lbl_file.read_text(encoding="utf-8")


def test_click_close_x_deletes_annotation_without_opening_lightbox(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that clicking '✕' deletes the box and NEVER opens the review lightbox."""
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(
        tmp_path,
        "lightbox_test_frame",
        trusted=[{"class_id": 0, "class_name": "herb_aria", "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2, "trusted": True}],
    )
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)
    dlg.review_preview.resize(800, 600)
    # Render offscreen to populate paint hit rects
    pix = QPixmap(dlg.review_preview.size())
    dlg.review_preview.render(pix)

    assert len(dlg.review_preview._close_hit_rects) >= 1
    close_hit_rect, target_ann, _idx = dlg.review_preview._close_hit_rects[0]
    click_pos = close_hit_rect.center()

    with patch.object(dlg, "_open_review_lightbox") as mock_lightbox, \
         patch.object(dlg, "_learn_box_exemplar"):
        # Mouse press on '✕'
        ev_press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            click_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        assert dlg.eventFilter(dlg.review_preview, ev_press) is True

        # Mouse release on '✕'
        ev_release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            click_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        assert dlg.eventFilter(dlg.review_preview, ev_release) is True

        # Lightbox must NOT be opened!
        mock_lightbox.assert_not_called()

    # The box must be deleted from trusted list
    assert len(item0["trusted"]) == 0


def test_double_click_canvas_opens_lightbox_single_click_does_not(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that double-clicking opens lightbox while single click only selects/deselects."""
    dlg = factory_dialog
    item0 = _create_dummy_image_and_review(tmp_path, "click_test_frame")
    dlg._review_items = [item0]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)
    dlg.review_preview.resize(800, 600)

    center_pos = QPointF(400.0, 300.0)

    with patch.object(dlg, "_open_review_lightbox") as mock_lightbox:
        # Single click (press + release)
        ev_press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            center_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.eventFilter(dlg.review_preview, ev_press)

        ev_release = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            center_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.eventFilter(dlg.review_preview, ev_release)

        mock_lightbox.assert_not_called()

        # Double click
        ev_dbl = QMouseEvent(
            QMouseEvent.Type.MouseButtonDblClick,
            center_pos,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        dlg.eventFilter(dlg.review_preview, ev_dbl)
        mock_lightbox.assert_called_once()


def test_yolo_preview_dialog_canvas_click_does_not_close_dialog():
    """Verifies that clicking on the canvas inside YOLOPreviewDialog does not close the dialog."""
    from PyQt6.QtGui import QPixmap
    from ui.yolo_panel import YOLOPreviewDialog

    pix = QPixmap(200, 200)
    pix.fill(Qt.GlobalColor.black)
    dlg = YOLOPreviewDialog(source_pixmap=pix, annotations=[])
    dlg.resize(960, 600)
    dlg.show()

    # Canvas click event
    canvas_center = dlg.canvas.rect().center()
    ev = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(float(canvas_center.x()), float(canvas_center.y())),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    dlg.canvas.mousePressEvent(ev)
    # The dialog must still be open with result 0 (Rejected, i.e. not
    # accepted): `result() != 1` alone also passes for a dialog that was
    # never shown or was already closed.
    assert dlg.result() == 0
    assert dlg.isVisible(), "Preview dialog closed after canvas click"
    dlg.close()


def test_requeue_existing_dataset_loads_all_samples_including_previously_approved(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
):
    """Verifies that 'Mevcut seti onaya geri al' reloads all samples into the queue even if previously approved."""
    dlg = factory_dialog
    pool = get_approved_pool(tmp_path)

    _create_dummy_image_and_review(tmp_path, "requeue_frame_1")
    _create_dummy_image_and_review(tmp_path, "requeue_frame_2")

    # Manually approve frame 1
    pool.add_approved_sample(
        stem="requeue_frame_1",
        split="train",
        image_path=tmp_path / "images" / "train" / "requeue_frame_1.png",
        trusted_boxes=[{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
    )

    # Standard reload skips requeue_frame_1
    dlg._reload_review_queue()
    stems = [it["stem"] for it in dlg._review_items]
    assert "requeue_frame_1" not in stems
    assert "requeue_frame_2" in stems

    # Now run _requeue_existing_dataset
    dlg._auto_confirm_dialogs = True
    with patch("PyQt6.QtWidgets.QMessageBox.question", return_value=pytest.importorskip("PyQt6.QtWidgets").QMessageBox.StandardButton.Yes):
        dlg._requeue_existing_dataset()

    # Both samples must now be in the review queue
    requeued_stems = [it["stem"] for it in dlg._review_items]
    assert "requeue_frame_1" in requeued_stems
    assert "requeue_frame_2" in requeued_stems
    assert len(dlg._review_items) == 2


def test_on_refresh_clicked_refreshes_queue_and_dashboard(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Verifies that on_refresh_clicked reloads queue, updates dashboard, and gives feedback."""
    dlg = factory_dialog
    _create_dummy_image_and_review(tmp_path, "refresh_test_frame")

    dlg.on_refresh_clicked()
    # [YOLO-ASYNC] Yenile kuyruğu worker'da doldurur; okumadan önce bekle.
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    assert len(dlg._review_items) >= 1
    assert any(it["stem"] == "refresh_test_frame" for it in dlg._review_items)


def test_class_8_is_ore_obsidian():
    """Verifies Class 8 is canonical ore_obsidian and aliases work.

    F-1946 remap (Decision 1): the dialog spec table follows CANONICAL_NAMES,
    so 8 = ore_obsidian. The legacy combo/skill-prompt names remain only as
    aliases resolving to canonical 13 (skill_ui_prompt), not 8.
    """
    from ui.yolo_factory_dialog import CLASS_ID_TO_SPEC, CLASS_KEY_ALIASES

    spec = CLASS_ID_TO_SPEC[8]
    assert spec["key"] == "ore_obsidian"
    assert "Obsidyen" in spec["label_tr"]
    assert CLASS_KEY_ALIASES["ore_obsidian"] == 8
    assert CLASS_KEY_ALIASES["e_gather_icon"] == 13
    assert CLASS_KEY_ALIASES["combo_skill_prompt"] == 13
    assert CLASS_KEY_ALIASES["combo_icon"] == 13
    assert CLASS_KEY_ALIASES["q_combo_icon"] == 13


def test_class_10_is_monster_3d_and_aliases_work():
    """Verifies Class 10 is configured as monster_3d without forcing negative, and aliases work."""
    from ui.yolo_factory_dialog import CLASS_ID_TO_SPEC, CLASS_KEY_ALIASES

    spec10 = CLASS_ID_TO_SPEC[10]
    assert spec10["key"] == "monster_3d"
    assert "monster_3d" in spec10["label_tr"]
    assert CLASS_KEY_ALIASES["monster_3d"] == 10
    assert CLASS_KEY_ALIASES["monster_3d_negative"] == 10

    spec11 = CLASS_ID_TO_SPEC[11]
    assert spec11["key"] == "minimap_monster_dot"
    assert CLASS_KEY_ALIASES["minimap_monster_dot"] == 11
    assert CLASS_KEY_ALIASES["minimap_monster_negative"] == 11


def test_class_25_is_kinah_and_aliases_work():
    """Verifies Class 25 is configured as kinah and aliases work across taxonomy and UI."""
    from dataset_factory.class_taxonomy import (
        ALL_CANONICAL_CLASSES,
        CANONICAL_NAMES,
        is_currency,
        is_hud_element,
        sanitize_class_id_and_name,
    )
    from dataset_factory.manual_tag_search import search_label_classes
    from ui.yolo_factory_dialog import CLASS_ID_TO_SPEC, CLASS_KEY_ALIASES

    assert 25 in ALL_CANONICAL_CLASSES
    assert CANONICAL_NAMES[25] == "kinah"
    assert is_currency(25) is True
    assert is_currency("kinah") is True
    assert is_currency("kina") is True
    assert is_hud_element(25) is True

    # Sanitize tests
    assert sanitize_class_id_and_name(25) == (25, "kinah")
    assert sanitize_class_id_and_name(-1, "kinah") == (25, "kinah")
    assert sanitize_class_id_and_name(-1, "kina") == (25, "kinah")
    assert sanitize_class_id_and_name(-1, "altin") == (25, "kinah")
    assert sanitize_class_id_and_name(-1, "gold") == (25, "kinah")

    # UI Specs
    assert 25 in CLASS_ID_TO_SPEC
    spec = CLASS_ID_TO_SPEC[25]
    assert spec["key"] == "kinah"
    assert "Kinah" in spec["label_tr"]
    assert CLASS_KEY_ALIASES["kinah"] == 25
    assert CLASS_KEY_ALIASES["kina"] == 25

    # Manual Tag Search
    results = search_label_classes("kinah")
    assert any(int(r["class_id"]) == 25 for r in results)
    results_kina = search_label_classes("kina")
    assert any(int(r["class_id"]) == 25 for r in results_kina)


def test_kinah_earnings_display_and_tracking(factory_dialog: YOLOFactoryDialog, tmp_path: Path):
    """Kinah bu ekranda GÖSTERİLMEZ (görev kuralı) ama backend'de izlenir.

    Eski davranış (rozette "Bu Kare: ~43.5K Kinah") kaldırıldı:
    lbl_review_kinah yok, review_sub'da para değeri yok. Buna karşılık
    _estimate_box_kinah backend tahmini ve _session_approved_kinah
    sayacı aynen durur; onaylanan karenin değeri sayaca yazılır.
    """
    dlg = factory_dialog

    # Frame with 1x ore_orichalcum (cid 3 -> 3x hasat ~ 43.500 Kinah) and 1x kinah tag (cid 25)
    sample = _create_dummy_image_and_review(
        tmp_path,
        "kinah_frame_test",
        trusted=[
            {"class_id": 3, "class_name": "ore_orichalcum", "confidence": 0.95},
            {"class_id": 25, "class_name": "kinah", "confidence": 0.99},
        ],
    )
    dlg._review_items = [sample]
    dlg._review_index = 0
    dlg._render_queue_list()
    dlg._on_queue_row_changed(0)

    # Ekranda Kinah rozeti YOK — para değeri gösterilmez.
    assert not hasattr(dlg, "lbl_review_kinah")
    assert "Kinah" not in dlg.review_sub.text()
    assert "43.5K" not in dlg.review_sub.text()

    # Backend tahmini hâlâ doğru: ore_orichalcum 43.500, kinah etiketi 0.
    assert dlg._estimate_box_kinah({"class_id": 3}) == 43500
    assert dlg._estimate_box_kinah({"class_id": 25}) == 0

    # Detail rows still exist.
    row_count = dlg.det_layout.count()
    assert row_count > 0

    # Test approving frame updates session earnings
    initial_earned = getattr(dlg, "_session_approved_kinah", 0)
    dlg._on_step2_approve()
    # Exact earnings for this frame: 43.500 Kinah (backend),
    # not just "at least 40.000".
    assert getattr(dlg, "_session_approved_kinah", 0) == initial_earned + 43500




