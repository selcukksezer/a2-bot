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

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, QRectF, Qt
from PyQt6.QtGui import QMouseEvent, QPixmap
from PyQt6.QtWidgets import QApplication

from dataset_factory.approved_pool import ApprovedDatasetPool
import dataset_factory.approved_pool as approved_pool_mod
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


def test_canvas_handles_and_interior_hit_testing(qapp: QApplication) -> None:
    """1. Test that hit_test_handle correctly detects all 8 handles and the interior move zone."""
    canvas = YOLOPreviewCanvas()
    canvas.resize(500, 500)
    canvas._last_target_rect = QRectF(50.0, 50.0, 400.0, 400.0)

    # Box centered at (0.5, 0.5) with width=0.4, height=0.4
    # On 400x400 target (offset 50, 50):
    # left = 50 + (0.5 - 0.2)*400 = 170
    # right = 50 + (0.5 + 0.2)*400 = 330
    # top = 50 + (0.5 - 0.2)*400 = 170
    # bottom = 50 + (0.5 + 0.2)*400 = 330
    # center = (250, 250)
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
    canvas._box_draw_info = [
        {
            "idx": 0,
            "ann": canvas._annotations[0],
            "box_rect": QRectF(170.0, 170.0, 160.0, 160.0),
            "lbl_rect": QRectF(170.0, 150.0, 100.0, 18.0),
            "close_rect": QRectF(255.0, 151.0, 14.0, 14.0),
            "close_hit": QRectF(250.0, 146.0, 24.0, 24.0),
        }
    ]
    canvas._selected_box_idx = 0

    # Corners
    h, idx, _ = canvas.hit_test_handle(QPointF(170.0, 170.0))
    assert h == "tl" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(330.0, 170.0))
    assert h == "tr" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(170.0, 330.0))
    assert h == "bl" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(330.0, 330.0))
    assert h == "br" and idx == 0

    # Edge midpoints
    h, idx, _ = canvas.hit_test_handle(QPointF(250.0, 170.0))
    assert h == "t" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(250.0, 330.0))
    assert h == "b" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(170.0, 250.0))
    assert h == "l" and idx == 0

    h, idx, _ = canvas.hit_test_handle(QPointF(330.0, 250.0))
    assert h == "r" and idx == 0

    # Interior -> "move"
    h, idx, _ = canvas.hit_test_handle(QPointF(250.0, 250.0))
    assert h == "move" and idx == 0

    # Outside -> None
    h, idx, _ = canvas.hit_test_handle(QPointF(50.0, 50.0))
    assert h is None and idx == -1


def test_canvas_box_moving_workflow(qapp: QApplication) -> None:
    """2. Test that dragging the interior of a box moves it and emits boxEdited."""
    canvas = YOLOPreviewCanvas()
    canvas.resize(500, 500)
    canvas._last_target_rect = QRectF(0.0, 0.0, 400.0, 400.0)

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
    canvas._box_draw_info = [
        {
            "idx": 0,
            "ann": canvas._annotations[0],
            "box_rect": QRectF(160.0, 160.0, 80.0, 80.0),
            "lbl_rect": QRectF(160.0, 140.0, 80.0, 18.0),
            "close_rect": QRectF(225.0, 141.0, 14.0, 14.0),
            "close_hit": QRectF(220.0, 136.0, 24.0, 24.0),
        }
    ]

    edited_boxes: List[Dict[str, Any]] = []
    canvas.boxEdited.connect(lambda b: edited_boxes.append(dict(b)))

    # Press inside box at (200, 200)
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, QPointF(200.0, 200.0)))
    assert canvas._interaction_mode == "moving"
    assert canvas._selected_box_idx == 0

    # Drag by +40px horizontally and +40px vertically -> delta in norm coords: 40/400 = 0.10
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, QPointF(240.0, 240.0)))
    assert canvas._annotations[0]["cx"] == pytest.approx(0.60, abs=1e-3)
    assert canvas._annotations[0]["cy"] == pytest.approx(0.60, abs=1e-3)
    assert canvas._annotations[0]["w"] == pytest.approx(0.20, abs=1e-3)  # Width unchanged
    assert canvas._annotations[0]["h"] == pytest.approx(0.20, abs=1e-3)  # Height unchanged

    # Release mouse
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            QPointF(240.0, 240.0),
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    assert len(edited_boxes) == 1
    assert edited_boxes[0]["cx"] == pytest.approx(0.60, abs=1e-3)
    assert edited_boxes[0]["cy"] == pytest.approx(0.60, abs=1e-3)
    assert edited_boxes[0]["trusted"] is True


def test_canvas_box_resizing_workflow(qapp: QApplication) -> None:
    """3. Test that dragging a corner/edge handle resizes the box and emits boxEdited."""
    canvas = YOLOPreviewCanvas()
    canvas.resize(500, 500)
    canvas._last_target_rect = QRectF(0.0, 0.0, 400.0, 400.0)

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
    # Box rect: left=160, right=240, top=160, bottom=240
    canvas._box_draw_info = [
        {
            "idx": 0,
            "ann": canvas._annotations[0],
            "box_rect": QRectF(160.0, 160.0, 80.0, 80.0),
            "lbl_rect": QRectF(160.0, 140.0, 80.0, 18.0),
            "close_rect": QRectF(225.0, 141.0, 14.0, 14.0),
            "close_hit": QRectF(220.0, 136.0, 24.0, 24.0),
        }
    ]
    canvas._selected_box_idx = 0

    edited_boxes: List[Dict[str, Any]] = []
    canvas.boxEdited.connect(lambda b: edited_boxes.append(dict(b)))

    # Press on Bottom-Right handle (240, 240)
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, QPointF(240.0, 240.0)))
    assert canvas._interaction_mode == "resizing"
    assert canvas._active_handle == "br"

    # Drag BR outward by +40px horizontally and +40px vertically
    # New right=280, bottom=280, left=160, top=160
    # width = 120 -> norm w = 120/400 = 0.30
    # height = 120 -> norm h = 120/400 = 0.30
    # center = (220, 220) -> norm cx = 220/400 = 0.55
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, QPointF(280.0, 280.0)))
    assert canvas._annotations[0]["w"] == pytest.approx(0.30, abs=1e-3)
    assert canvas._annotations[0]["h"] == pytest.approx(0.30, abs=1e-3)
    assert canvas._annotations[0]["cx"] == pytest.approx(0.55, abs=1e-3)
    assert canvas._annotations[0]["cy"] == pytest.approx(0.55, abs=1e-3)

    # Release
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            QPointF(280.0, 280.0),
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    assert len(edited_boxes) == 1
    assert edited_boxes[0]["w"] == pytest.approx(0.30, abs=1e-3)
    assert edited_boxes[0]["h"] == pytest.approx(0.30, abs=1e-3)


def test_step2_approve_guarantees_disk_and_pool_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """4. Test that clicking 'Onayla' writes YOLO format labels to disk and registers into ApprovedDatasetPool."""
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "review" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "images" / "train").mkdir(parents=True, exist_ok=True)

    pool = ApprovedDatasetPool(project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool)

    dlg = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: dataset_dir
    dlg._pipeline = {"step": 2, "pending": 1, "reviewed": 0}
    dlg._review_decode_cache = {}
    dlg._learn_box_exemplar = lambda *a, **kw: None
    dlg._mark_model_stale = lambda *a, **kw: None
    dlg._append_log = lambda *a, **kw: None
    dlg._render_queue_list = lambda *a, **kw: None
    dlg._render_review_frame = lambda *a, **kw: None
    dlg._update_blockers = lambda *a, **kw: None
    dlg._update_next_card = lambda *a, **kw: None
    dlg._mark_step_done = lambda *a, **kw: None
    dlg._refresh_stepper_visuals = lambda *a, **kw: None

    stem = "test_frame_guarantee"
    rev_p = dataset_dir / "review" / "train" / f"{stem}.json"
    rev_p.write_text("{}", encoding="utf-8")
    img_p = dataset_dir / "images" / "train" / f"{stem}.jpg"
    img_p.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xbf\x00\xff\xd9")

    # Canvas has updated boxes (user moved/resized to cx=0.62, cy=0.48, w=0.25, h=0.35)
    canvas = YOLOPreviewCanvas()
    canvas._annotations = [
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
    dlg.review_preview = canvas

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

    # Trigger Step 2 Approve
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


def test_large_canvas_dimensions_and_sidebar_structure(qapp: QApplication) -> None:
    """5. Test that the review canvas is large (minHeight >= 600) and sidebar exists with width 340."""
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
        assert sidebar.width() == 340
    finally:
        dlg.close()
