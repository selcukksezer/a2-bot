# -*- coding: utf-8 -*-
"""
tests/test_step2_move_resize_and_large_canvas.py
================================================
Test suite verifying:
1. YOLOPreviewCanvas 8 resize handles ('tl', 'tr', 'bl', 'br', 't', 'b', 'l', 'r') and interior 'move' hit testing.
2. Mouse drag moves the bounding box (updates cx, cy; emits boxEdited).
3. Mouse drag resizes the bounding box (updates w, h, cx, cy; emits boxEdited).
4. Step 2 "Onayla" (_on_step2_approve):
   - Guarantees modified boxes are persisted to labels/{split}/{stem}.txt in YOLO format.
   - Guarantees sample is saved into ApprovedDatasetPool with human_approved=True.
5. Large canvas layout: review_preview has minimumHeight >= 600 and sidebar has width 340.

Hit/move/resize tests drive the REAL paintEvent: annotations are set on the
canvas, the canvas is rendered to a QPixmap, and _box_draw_info is read back
from what the painter actually computed. Nothing about the drawn boxes is
hand-filled. The approve test builds a REAL YOLOFactoryDialog (no __new__,
no stubbed methods) and points it at a tmp dataset through its real
dependency object (dlg._builtin_helper.dataset_dir); the real
_learn_box_exemplar path runs against a real decoded image, with the
exemplar memory singleton redirected to tmp so nothing leaks into the repo.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import QApplication

from dataset_factory.approved_pool import ApprovedDatasetPool
import dataset_factory.approved_pool as approved_pool_mod
import dataset_factory.exemplar_memory as exemplar_memory_mod
from dataset_factory.exemplar_memory import ExemplarMemory
from ui.yolo_factory_dialog import YOLOFactoryDialog, YOLOPreviewCanvas


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_mouse_event(
    event_type: QEvent.Type,
    pos: QPointF,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    buttons: Qt.MouseButton = Qt.MouseButton.LeftButton,
) -> QMouseEvent:
    return QMouseEvent(
        event_type,
        pos,
        pos,
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def _paint_canvas(canvas: YOLOPreviewCanvas) -> Dict[str, Any]:
    """Render the canvas for real and return the box geometry paintEvent computed.

    paintEvent rebuilds _box_draw_info (and _last_target_rect) from
    canvas._annotations every time it runs, so reading it back here proves
    the hit/move/resize tests exercise the real drawing code, not a
    hand-filled cache.
    """
    pixmap = QPixmap(canvas.size())
    pixmap.fill(Qt.GlobalColor.transparent)
    canvas.render(pixmap)
    assert len(canvas._box_draw_info) == 1, "paintEvent must draw exactly one box"
    info = canvas._box_draw_info[0]
    assert info["idx"] == 0
    assert info["ann"] is canvas._annotations[0]
    return info


def test_canvas_handles_and_interior_hit_testing(qapp: QApplication) -> None:
    """1. Test that hit_test_handle correctly detects all 8 handles and the interior move zone."""
    canvas = YOLOPreviewCanvas()
    canvas.resize(500, 500)

    # Box centered at (0.5, 0.5) with width=0.4, height=0.4
    canvas._annotations = [
        {
            "class_id": 1,
            "class_name": "monster_3d",
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.4,
            "h": 0.4,
            "trusted": True,
        }
    ]
    canvas._selected_box_idx = 0

    # Real paint: _box_draw_info/_last_target_rect come from paintEvent.
    info = _paint_canvas(canvas)
    box_rect: QRectF = info["box_rect"]
    trect = canvas._last_target_rect
    assert not trect.isEmpty()
    # Painted geometry must match the annotation math (no pixmap: target == view rect).
    assert box_rect.left() == pytest.approx(trect.left() + (0.5 - 0.2) * trect.width(), abs=1e-6)
    assert box_rect.top() == pytest.approx(trect.top() + (0.5 - 0.2) * trect.height(), abs=1e-6)
    assert box_rect.width() == pytest.approx(0.4 * trect.width(), abs=1e-6)
    assert box_rect.height() == pytest.approx(0.4 * trect.height(), abs=1e-6)

    # Corners
    h, idx, _ = canvas.hit_test_handle(box_rect.topLeft())
    assert h == "tl" and idx == 0

    h, idx, _ = canvas.hit_test_handle(box_rect.topRight())
    assert h == "tr" and idx == 0

    h, idx, _ = canvas.hit_test_handle(box_rect.bottomLeft())
    assert h == "bl" and idx == 0

    h, idx, _ = canvas.hit_test_handle(box_rect.bottomRight())
    assert h == "br" and idx == 0

    # Edge midpoints
    h, idx, _ = canvas.hit_test_handle(QPointF(box_rect.center().x(), box_rect.top()))
    assert h == "t" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(box_rect.center().x(), box_rect.bottom()))
    assert h == "b" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(box_rect.left(), box_rect.center().y()))
    assert h == "l" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(box_rect.right(), box_rect.center().y()))
    assert h == "r" and idx == 0

    # Interior -> "move"
    h, idx, _ = canvas.hit_test_handle(box_rect.center())
    assert h == "move" and idx == 0

    # Outside -> None
    h, idx, _ = canvas.hit_test_handle(QPointF(50.0, 50.0))
    assert h is None and idx == -1


def test_canvas_box_moving_workflow(qapp: QApplication) -> None:
    """2. Test that dragging the interior of a box moves it and emits boxEdited."""
    canvas = YOLOPreviewCanvas()
    canvas.resize(500, 500)

    canvas._annotations = [
        {
            "class_id": 0,
            "class_name": "herb_aria",
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.2,
            "h": 0.2,
            "trusted": True,
        }
    ]
    canvas._selected_box_idx = 0

    info = _paint_canvas(canvas)
    box_rect: QRectF = info["box_rect"]
    trect = canvas._last_target_rect

    edited_boxes: List[Dict[str, Any]] = []
    canvas.boxEdited.connect(lambda b: edited_boxes.append(dict(b)))

    # Press inside box at its painted center
    press_pos = box_rect.center()
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, press_pos))
    assert canvas._interaction_mode == "moving"
    assert canvas._selected_box_idx == 0

    # Drag by +40px horizontally and +40px vertically.
    drag = 40.0
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, QPointF(press_pos.x() + drag, press_pos.y() + drag)))
    assert canvas._annotations[0]["cx"] == pytest.approx(0.5 + drag / trect.width(), abs=1e-3)
    assert canvas._annotations[0]["cy"] == pytest.approx(0.5 + drag / trect.height(), abs=1e-3)
    assert canvas._annotations[0]["w"] == pytest.approx(0.20, abs=1e-3)  # Width unchanged
    assert canvas._annotations[0]["h"] == pytest.approx(0.20, abs=1e-3)  # Height unchanged

    # Release mouse
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            QPointF(press_pos.x() + drag, press_pos.y() + drag),
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    assert len(edited_boxes) == 1
    assert edited_boxes[0]["cx"] == pytest.approx(0.5 + drag / trect.width(), abs=1e-3)
    assert edited_boxes[0]["cy"] == pytest.approx(0.5 + drag / trect.height(), abs=1e-3)
    assert edited_boxes[0]["trusted"] is True


def test_canvas_box_resizing_workflow(qapp: QApplication) -> None:
    """3. Test that dragging a corner/edge handle resizes the box and emits boxEdited."""
    canvas = YOLOPreviewCanvas()
    canvas.resize(500, 500)

    canvas._annotations = [
        {
            "class_id": 2,
            "class_name": "herb_targena",
            "cx": 0.5,
            "cy": 0.5,
            "w": 0.2,
            "h": 0.2,
            "trusted": True,
        }
    ]
    canvas._selected_box_idx = 0

    info = _paint_canvas(canvas)
    box_rect: QRectF = info["box_rect"]
    trect = canvas._last_target_rect

    edited_boxes: List[Dict[str, Any]] = []
    canvas.boxEdited.connect(lambda b: edited_boxes.append(dict(b)))

    # Press on the painted Bottom-Right handle
    press_pos = box_rect.bottomRight()
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, press_pos))
    assert canvas._interaction_mode == "resizing"
    assert canvas._active_handle == "br"

    # Drag BR outward by +40px horizontally and +40px vertically.
    # New width/height grow by 40px; the center moves by half that drag.
    drag = 40.0
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, QPointF(press_pos.x() + drag, press_pos.y() + drag)))
    assert canvas._annotations[0]["w"] == pytest.approx(0.2 + drag / trect.width(), abs=1e-3)
    assert canvas._annotations[0]["h"] == pytest.approx(0.2 + drag / trect.height(), abs=1e-3)
    assert canvas._annotations[0]["cx"] == pytest.approx(0.5 + drag / (2.0 * trect.width()), abs=1e-3)
    assert canvas._annotations[0]["cy"] == pytest.approx(0.5 + drag / (2.0 * trect.height()), abs=1e-3)

    # Release
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            QPointF(press_pos.x() + drag, press_pos.y() + drag),
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    assert len(edited_boxes) == 1
    assert edited_boxes[0]["w"] == pytest.approx(0.2 + drag / trect.width(), abs=1e-3)
    assert edited_boxes[0]["h"] == pytest.approx(0.2 + drag / trect.height(), abs=1e-3)


def test_step2_approve_guarantees_disk_and_pool_persistence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qapp: QApplication
) -> None:
    """4. Test that clicking 'Onayla' writes YOLO format labels to disk and registers into ApprovedDatasetPool.

    Runs through a REAL YOLOFactoryDialog: no __new__, no stubbed dialog
    methods. The dialog is pointed at the tmp dataset through its real
    dependency object (dlg._builtin_helper.dataset_dir); the real
    _learn_box_exemplar runs against a real decoded frame, with the
    exemplar-memory and approved-pool singletons redirected to tmp so the
    repo stays untouched.
    """
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "review" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)

    pool = ApprovedDatasetPool(project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool)
    exemplar_path = tmp_path / "learned_exemplars.json"
    monkeypatch.setattr(
        exemplar_memory_mod, "_GLOBAL_EXEMPLAR_MEMORY", ExemplarMemory(json_path=exemplar_path)
    )

    stem = "test_frame_guarantee"
    rev_p = dataset_dir / "review" / "train" / f"{stem}.json"
    rev_p.write_text("{}", encoding="utf-8")
    img_p = dataset_dir / "images" / "train" / f"{stem}.jpg"
    # Real frame (not a 1x1 stub): textured so the real exemplar crop path
    # extracts features and records a positive exemplar.
    rng = np.random.default_rng(7)
    assert cv2.imwrite(str(img_p), rng.integers(0, 256, (240, 320, 3), dtype=np.uint8))

    dlg = YOLOFactoryDialog()
    try:
        dlg._auto_confirm_dialogs = True
        # Real path resolution: _dataset_dir() reads _builtin_helper.dataset_dir.
        dlg._builtin_helper.dataset_dir = dataset_dir
        # [O3]: _dataset_dir() prefers the bound factory_controller (D29 b55c0ef);
        # the tmp wiring must redirect it too, or the dialog keeps the repo dataset.
        _ctrl = getattr(dlg, "factory_controller", None)
        if _ctrl is not None and hasattr(_ctrl, "dataset_dir"):
            _ctrl.dataset_dir = dataset_dir
        assert dlg._dataset_dir() == dataset_dir
        dlg._pipeline.update({"step": 2, "pending": 1, "reviewed": 0})

        # Canvas has updated boxes (user moved/resized to cx=0.62, cy=0.48, w=0.25, h=0.35)
        # on the dialog's OWN preview canvas.
        dlg.review_preview._annotations = [
            {
                "class_id": 3,
                "class_name": "ore_orichalcum",
                "cx": 0.62,
                "cy": 0.48,
                "w": 0.25,
                "h": 0.35,
                "confidence": 1.0,
                "trusted": True,
                "human_approved": True,
            }
        ]

        dlg._review_items = [
            {
                "stem": stem,
                "split": "train",
                "weak": [],
                "trusted": [
                    # Old coordinates in review item
                    {"class_id": 3, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
                ],
                "done": None,
                "human_approved": False,
                "path": str(rev_p),
            }
        ]
        dlg._review_index = 0
        dlg._sync_pending_from_queue()

        # Trigger Step 2 Approve (real _on_step2_approve -> _review_done).
        dlg._on_step2_approve()

        # 1. Verify file on disk exists in labels/train/test_frame_guarantee.txt
        lbl_p = dataset_dir / "labels" / "train" / f"{stem}.txt"
        assert lbl_p.is_file(), "YOLO label file must be written to disk"
        lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        parts = lines[0].split()
        assert int(parts[0]) == 3
        assert float(parts[1]) == pytest.approx(0.62, abs=1e-4)
        assert float(parts[2]) == pytest.approx(0.48, abs=1e-4)
        assert float(parts[3]) == pytest.approx(0.25, abs=1e-4)
        assert float(parts[4]) == pytest.approx(0.35, abs=1e-4)

        # 2. Verify ApprovedDatasetPool
        assert pool.is_approved(stem) is True
        assert pool.get_approved_stats()["total_approved_images"] == 1

        # 3. Verify the REAL _learn_box_exemplar ran: a positive exemplar for
        # class 3 was recorded in the tmp exemplar memory (not stubbed away).
        assert exemplar_path.is_file(), "real exemplar learning must persist a record"
        exemplar_data = json.loads(exemplar_path.read_text(encoding="utf-8"))
        positives = [
            e
            for e in exemplar_data.get("positive", [])
            if int(e.get("class_id", -1)) == 3
        ]
        assert positives, "expected a positive exemplar for class 3 (ore_orichalcum)"
    finally:
        dlg.close()


def test_large_canvas_dimensions_and_sidebar_structure(qapp: QApplication) -> None:
    """5. Test that the review canvas is large (minHeight >= 600) and sidebar exists with width 420 (v14 layout fix: 340 -> 420)."""
    dlg = YOLOFactoryDialog()
    try:
        # Check minimum dialog size
        assert dlg.minimumHeight() >= 800
        assert dlg.minimumWidth() >= 1200

        # Check review_preview minimum height
        assert dlg.review_preview.minimumHeight() >= 600

        # Check sidebar scroll exists
        sidebar = dlg.findChild(object, "PipeSidebarScroll")
        assert sidebar is not None
        assert sidebar.width() == 420  # v14: YOLO layout fix sol panel 340 -> 420 px
    finally:
        dlg.close()
