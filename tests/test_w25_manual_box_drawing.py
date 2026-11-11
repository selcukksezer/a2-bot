# -*- coding: utf-8 -*-
"""
tests/test_w25_manual_box_drawing.py
====================================
Unit tests for Worker W25 verifying:
1. Conversion of mouse-drawn rubber-band boxes into normalized `(cx, cy, w, h)` coordinates.
2. Minimum drag distance gate (`>= 8px`) and viewport boundary clamping (`[0.0, 1.0]`).
3. Persistence of manually drawn boxes with `trusted=True, hand_label=True, confidence=1.0`.
4. Immediate positive exemplar teaching into `ExemplarMemory` from the drawn box crop.
5. End-to-end UI canvas mouse drag -> `boxAdded` signal -> dialog persistence & exemplar learning.
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
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication, QComboBox

from dataset_factory.exemplar_memory import ExemplarMemory, extract_box_crop
from dataset_factory.manual_tag_search import create_manual_annotation_box
from dataset_factory.rubberband_box_converter import RubberBandBoxConverter
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


def test_rubberband_drag_converts_pixels_to_normalized_cx_cy_w_h(qapp: QApplication) -> None:
    """1. Verify mouse-drawn pixel box converts accurately to normalized (cx, cy, w, h)."""
    converter = RubberBandBoxConverter(min_drag_px=8.0)
    target_rect = QRectF(100.0, 50.0, 400.0, 200.0)

    # Drag from (180, 90) to (340, 190) inside target_rect(100, 50, 400, 200)
    # Relative x: 80..240 -> center 160 / 400 = 0.40, width 160 / 400 = 0.40
    # Relative y: 40..140 -> center 90 / 200 = 0.45, height 100 / 200 = 0.50
    res = converter.convert_result(180.0, 90.0, 340.0, 190.0, target_rect=target_rect)
    assert res.valid is True
    assert res.cx == pytest.approx(0.40, abs=1e-4)
    assert res.cy == pytest.approx(0.45, abs=1e-4)
    assert res.w == pytest.approx(0.40, abs=1e-4)
    assert res.h == pytest.approx(0.50, abs=1e-4)

    # Also verify YOLOPreviewCanvas mouse press -> move -> release emits normalized (cx, cy, w, h)
    canvas = YOLOPreviewCanvas()
    canvas.resize(600, 400)
    frame = np.full((200, 400, 3), (40, 80, 120), dtype=np.uint8)
    canvas.set_live_frame(frame, "test_frame", [])
    canvas._last_target_rect = QRectF(100.0, 50.0, 400.0, 200.0)

    emitted_boxes: List[Dict[str, Any]] = []
    canvas.boxAdded.connect(lambda b: emitted_boxes.append(dict(b)))

    p_start = QPointF(340.0, 190.0)  # Reverse drag direction (bottom-right to top-left)
    p_end = QPointF(180.0, 90.0)
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, p_start))
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, p_end))
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            p_end,
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    assert len(emitted_boxes) == 1
    box = emitted_boxes[0]
    assert box["cx"] == pytest.approx(0.40, abs=1e-4)
    assert box["cy"] == pytest.approx(0.45, abs=1e-4)
    assert box["w"] == pytest.approx(0.40, abs=1e-4)
    assert box["h"] == pytest.approx(0.50, abs=1e-4)


def test_short_drag_below_threshold_rejected_and_clamping_to_bounds(qapp: QApplication) -> None:
    """2. Verify micro drags (< 8px) are ignored and out-of-bounds drags clamp to [0.0, 1.0]."""
    converter = RubberBandBoxConverter(min_drag_px=8.0)
    target_rect = QRectF(50.0, 50.0, 500.0, 300.0)

    # Short accidental click/drag (4px x 3px < 8px threshold)
    short_res = converter.convert_result(120.0, 120.0, 124.0, 123.0, target_rect=target_rect)
    assert short_res.valid is False
    assert short_res.reason == "drag_too_short"

    # Drag starting before top-left and ending past bottom-right of target_rect
    clamped_res = converter.convert_result(10.0, 20.0, 700.0, 450.0, target_rect=target_rect)
    assert clamped_res.valid is True
    assert 0.0 <= clamped_res.cx <= 1.0
    assert 0.0 <= clamped_res.cy <= 1.0
    assert 0.0 < clamped_res.w <= 1.0
    assert 0.0 < clamped_res.h <= 1.0
    assert clamped_res.cx == pytest.approx(0.5, abs=1e-4)
    assert clamped_res.cy == pytest.approx(0.5, abs=1e-4)
    assert clamped_res.w == pytest.approx(1.0, abs=1e-4)
    assert clamped_res.h == pytest.approx(1.0, abs=1e-4)

    # Verify canvas does not emit boxAdded for a 3px micro-drag
    canvas = YOLOPreviewCanvas()
    canvas._last_target_rect = target_rect
    emitted: List[Dict[str, Any]] = []
    canvas.boxAdded.connect(lambda b: emitted.append(dict(b)))

    p0 = QPointF(100.0, 100.0)
    p1 = QPointF(103.0, 102.0)
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, p0))
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, p1))
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            p1,
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )
    assert len(emitted) == 0


def test_manual_box_saved_with_trusted_hand_label_and_full_confidence() -> None:
    """3. Verify manual annotation builders set trusted=True, hand_label=True, confidence=1.0."""
    converter = RubberBandBoxConverter(min_drag_px=8.0)
    target_rect = (0.0, 0.0, 1000.0, 500.0)

    ann_from_converter = converter.convert_to_annotation(
        200.0,
        100.0,
        400.0,
        250.0,
        target_rect=target_rect,
        class_id=0,
        class_name="herb_aria",
    )
    assert ann_from_converter is not None
    assert ann_from_converter["cx"] == pytest.approx(0.30, abs=1e-4)
    assert ann_from_converter["cy"] == pytest.approx(0.35, abs=1e-4)
    assert ann_from_converter["w"] == pytest.approx(0.20, abs=1e-4)
    assert ann_from_converter["h"] == pytest.approx(0.30, abs=1e-4)
    assert ann_from_converter["trusted"] is True
    assert ann_from_converter["hand_label"] is True
    assert ann_from_converter["human_approved"] is True
    assert ann_from_converter["confidence"] == pytest.approx(1.0)
    assert ann_from_converter["conf"] == pytest.approx(1.0)

    ann_from_tag_search = create_manual_annotation_box(
        3,
        ann_from_converter["cx"],
        ann_from_converter["cy"],
        ann_from_converter["w"],
        ann_from_converter["h"],
    )
    assert ann_from_tag_search["class_id"] == 3
    assert ann_from_tag_search["class_name"] == "ore_orichalcum"
    assert ann_from_tag_search["trusted"] is True
    assert ann_from_tag_search["weak"] is False
    assert ann_from_tag_search["hand_label"] is True
    assert ann_from_tag_search["human_approved"] is True
    assert ann_from_tag_search["confidence"] == pytest.approx(1.0)
    assert ann_from_tag_search["conf"] == pytest.approx(1.0)


def test_manual_box_teaches_exemplar_memory_positive_prototype(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """4. Verify a manually drawn box crop is learned as a positive exemplar in ExemplarMemory.

    The teaching must go through the DIALOG wiring — a real canvas drag emits
    ``boxAdded`` and ``YOLOFactoryDialog._on_manual_box_added`` resolves the
    class, persists the box and teaches the exemplar via
    ``_learn_box_exemplar`` — not through a direct ``add_positive_exemplar``
    API call that bypasses the dialog entirely.
    """
    ds_dir = tmp_path / "dataset"
    (ds_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "review" / "train").mkdir(parents=True, exist_ok=True)

    stem = "manual_draw_exemplar_01"
    img_path = ds_dir / "images" / "train" / f"{stem}.png"
    rev_path = ds_dir / "review" / "train" / f"{stem}.json"

    # Synthetic image with a distinct copper-orange Orichalcum patch at center
    img = np.full((400, 400, 3), 35, dtype=np.uint8)
    patch = np.zeros((100, 100, 3), dtype=np.uint8)
    patch[:, :] = (25, 110, 230)  # Distinct BGR color
    patch[::4, ::4] = (60, 160, 255)  # Texture edges for descriptor
    img[150:250, 150:250] = patch
    ok_enc, buf = cv2.imencode(".png", img)
    assert ok_enc is True
    buf.tofile(str(img_path))
    assert img_path.is_file()

    ex_json = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=ex_json)
    monkeypatch.setattr("dataset_factory.exemplar_memory.get_exemplar_memory", lambda: mem)

    # Lightweight harness binding YOLOFactoryDialog's manual box methods
    dialog = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dialog._dataset_dir = lambda: ds_dir
    dialog._review_decode_cache = {}
    dialog._pipeline = {"pending": 1, "reviewed": 0, "step": 2}
    dialog.manual_tag_combo = QComboBox()
    dialog.manual_tag_combo.addItem("0: herb_aria — 0: Aria", 0)
    dialog.manual_tag_combo.addItem("3: ore_orichalcum — 3: Orichalcum", 3)
    dialog.manual_tag_combo.setCurrentIndex(1)  # Select class_id = 3 (ore_orichalcum)
    dialog._review_items = [
        {
            "path": rev_path,
            "split": "train",
            "stem": stem,
            "weak": [],
            "trusted": [],
            "done": None,
            "conf": 0.0,
        }
    ]
    dialog._review_index = 0
    dialog._resolve_review_image = lambda split, s: img_path
    dialog._imread_cached = lambda p: cv2.imread(str(p))
    dialog._mark_model_stale = lambda reason="": None
    dialog._append_log = lambda msg="": None
    dialog._render_queue_list = lambda: None
    dialog._render_review_frame = lambda: None

    canvas = YOLOPreviewCanvas()
    canvas._last_target_rect = QRectF(0.0, 0.0, 400.0, 400.0)
    canvas.boxAdded.connect(lambda b: dialog._on_manual_box_added(b))

    # Real mouse drag from (150, 150) to (250, 250) on the 400x400 target rect:
    # the drawn box covers exactly the Orichalcum patch (cx=0.5, cy=0.5, w=0.25, h=0.25).
    p_start = QPointF(150.0, 150.0)
    p_end = QPointF(250.0, 250.0)
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, p_start))
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, p_end))
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            p_end,
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    # The dialog wiring must have taught the exemplar from the drawn box crop
    assert mem.has_positive_exemplars(3) is True
    pos_entries = [p for p in mem._positive if int(p.get("class_id", -1)) == 3]
    assert len(pos_entries) == 1
    assert pos_entries[0]["source"] == "manual_draw"
    assert pos_entries[0]["class_name"] == "ore_orichalcum"

    # Verify persistence to disk JSON
    assert ex_json.is_file()
    saved_payload = json.loads(ex_json.read_text(encoding="utf-8"))
    pos_list = saved_payload.get("positive", [])
    assert len(pos_list) == 1
    assert pos_list[0]["class_id"] == 3
    assert pos_list[0]["class_name"] == "ore_orichalcum"
    assert pos_list[0]["source"] == "manual_draw"

    # Verify matching against the learned positive exemplar, using the crop
    # of the box the dialog itself persisted (not a hand-built crop).
    saved_box = dialog._review_items[0]["trusted"][0]
    assert saved_box["class_id"] == 3
    assert saved_box["cx"] == pytest.approx(0.50, abs=1e-4)
    assert saved_box["cy"] == pytest.approx(0.50, abs=1e-4)
    assert saved_box["w"] == pytest.approx(0.25, abs=1e-4)
    assert saved_box["h"] == pytest.approx(0.25, abs=1e-4)
    crop = extract_box_crop(img, saved_box)
    assert crop is not None and crop.size > 0
    sim, matched_cid, matched_cname, matched_ex = mem.match_positive(crop, allowed_classes={3})
    assert matched_cid == 3
    assert matched_cname == "ore_orichalcum"
    assert sim >= 0.95
    assert matched_ex is not None


def test_end_to_end_canvas_mouse_drag_to_dialog_persistence_and_exemplar_learning(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """5. End-to-end: dragging a box on review_preview saves trusted=True, hand_label=True, confidence=1.0 and teaches ExemplarMemory."""
    ds_dir = tmp_path / "dataset"
    (ds_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (ds_dir / "review" / "train").mkdir(parents=True, exist_ok=True)

    stem = "manual_draw_sample_01"
    img_path = ds_dir / "images" / "train" / f"{stem}.png"
    rev_path = ds_dir / "review" / "train" / f"{stem}.json"

    # Create a 400x400 image with a rich green/purple herb_aria visual region in [100:300, 100:300]
    img = np.full((400, 400, 3), 30, dtype=np.uint8)
    img[100:300, 100:300] = (180, 60, 210)
    img[110:290:6, 110:290:6] = (90, 230, 120)
    ok_enc, buf = cv2.imencode(".png", img)
    assert ok_enc is True
    buf.tofile(str(img_path))
    assert img_path.is_file()

    ex_json = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=ex_json)
    monkeypatch.setattr("dataset_factory.exemplar_memory.get_exemplar_memory", lambda: mem)

    # Create lightweight harness binding YOLOFactoryDialog's manual box methods
    dialog = YOLOFactoryDialog.__new__(YOLOFactoryDialog)
    dialog._dataset_dir = lambda: ds_dir
    dialog._review_decode_cache = {}
    dialog._pipeline = {"pending": 1, "reviewed": 0, "step": 2}
    dialog.manual_tag_combo = QComboBox()
    dialog.manual_tag_combo.addItem("0: herb_aria — 0: Aria", 0)
    dialog.manual_tag_combo.addItem("3: ore_orichalcum — 3: Orichalcum", 3)
    dialog.manual_tag_combo.setCurrentIndex(1)  # Select class_id = 3 (ore_orichalcum)

    dialog._review_items = [
        {
            "path": rev_path,
            "split": "train",
            "stem": stem,
            "weak": [],
            "trusted": [],
            "done": None,
            "conf": 0.0,
        }
    ]
    dialog._review_index = 0
    dialog._resolve_review_image = lambda split, s: img_path
    dialog._mark_model_stale = lambda reason="": None
    dialog._append_log = lambda msg="": None
    dialog._render_queue_list = lambda: None
    dialog._render_review_frame = lambda: None

    canvas = YOLOPreviewCanvas()
    canvas._last_target_rect = QRectF(0.0, 0.0, 400.0, 400.0)
    canvas.boxAdded.connect(lambda b: dialog._on_manual_box_added(b))

    # Simulate mouse drag from (100, 100) to (300, 300) on 400x400 target_rect
    # Expected normalized: cx=0.5, cy=0.5, w=0.5, h=0.5
    p_start = QPointF(100.0, 100.0)
    p_end = QPointF(300.0, 300.0)
    canvas.mousePressEvent(_make_mouse_event(QEvent.Type.MouseButtonPress, p_start))
    canvas.mouseMoveEvent(_make_mouse_event(QEvent.Type.MouseMove, p_end))
    canvas.mouseReleaseEvent(
        _make_mouse_event(
            QEvent.Type.MouseButtonRelease,
            p_end,
            button=Qt.MouseButton.LeftButton,
            buttons=Qt.MouseButton.NoButton,
        )
    )

    trusted_boxes = dialog._review_items[0]["trusted"]
    assert len(trusted_boxes) == 1
    saved_box = trusted_boxes[0]

    assert saved_box["class_id"] == 3
    assert saved_box["class_name"] == "ore_orichalcum"
    assert saved_box["cx"] == pytest.approx(0.50, abs=1e-4)
    assert saved_box["cy"] == pytest.approx(0.50, abs=1e-4)
    assert saved_box["w"] == pytest.approx(0.50, abs=1e-4)
    assert saved_box["h"] == pytest.approx(0.50, abs=1e-4)
    assert saved_box["trusted"] is True
    assert saved_box["weak"] is False
    assert saved_box["hand_label"] is True
    assert saved_box["human_approved"] is True
    assert saved_box["confidence"] == pytest.approx(1.0)
    assert saved_box["conf"] == pytest.approx(1.0)

    # Verify review JSON persisted on disk
    assert rev_path.is_file()
    disk_data = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(disk_data["trusted"]) == 1
    disk_box = disk_data["trusted"][0]
    assert disk_box["trusted"] is True
    assert disk_box["hand_label"] is True
    assert disk_box["confidence"] == pytest.approx(1.0)
    assert disk_box["cx"] == pytest.approx(0.50, abs=1e-4)
    assert disk_box["cy"] == pytest.approx(0.50, abs=1e-4)
    assert disk_box["w"] == pytest.approx(0.50, abs=1e-4)
    assert disk_box["h"] == pytest.approx(0.50, abs=1e-4)

    # Verify ExemplarMemory learned the positive example from the drawn box
    assert mem.has_positive_exemplars(3) is True
    pos_entries = [p for p in mem._positive if int(p.get("class_id", -1)) == 3]
    assert len(pos_entries) == 1
    assert pos_entries[0]["source"] == "manual_draw"
