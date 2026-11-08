# -*- coding: utf-8 -*-
"""
tests/test_w70_end_to_end_labeling_pipeline.py
==============================================
Worker W70 — End-to-End Labeling Pipeline Integration Tests (5 Tests):

Validates the complete workflow:
1. Auto-labeling (`YOLOAutoLabeler` & `YOLOFactoryDialog._relabel_single_item`) -> Review Queue
2. 3-Tier Color Coding:
   - Tier 1: Green (Yeşil, >= 85% trusted or hand_label) -> #22c55e / #4ade80 ("Emin")
   - Tier 2: Orange (Turuncu, 70% - 84% weak) -> #fb923c ("Şüpheli")
   - Tier 3: Red (Kırmızı, < 70% low confidence) -> #ef4444 / #f87171 ("Düşük Güven (Şüpheli)")
3. Manual Box Drawing & Bilingual Tag Search (`manual_tag_search`, `manual_tag_combo`, `YOLOPreviewCanvas`)
4. Approving & Queue Deduction (`_on_step2_approve`, `'A'` shortcut, reopen persistence guard)
5. `ApprovedDatasetPool` (`data_approved.yaml`) Training Preparation exclusively with approved samples
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import pytest
import yaml
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QImage, QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

import dataset_factory.approved_pool as approved_pool_mod
from dataset_factory.approved_pool import ApprovedDatasetPool, CORE_22_CANONICAL_NAMES
from dataset_factory.manual_tag_search import (
    create_manual_annotation_box,
    get_tag_registry,
    search_label_classes,
)
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler
from ui.yolo_factory_dialog import YOLOFactoryDialog, _safe_imwrite_unicode


def _create_synthetic_game_frame(
    path: Path,
    draw_salient_resource: bool = True,
    draw_monster: bool = False,
) -> np.ndarray:
    """Create and save a valid BGR game frame on disk for testing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Subtle terrain background in viewport
    frame[:, :] = (28, 34, 24)

    if draw_salient_resource:
        # High-saturation chromatic resource node in central viewport (x=520, y=360)
        cv2.circle(frame, (520, 360), 32, (0, 195, 245), -1)
        cv2.circle(frame, (520, 360), 18, (20, 230, 255), -1)
        cv2.rectangle(frame, (495, 335), (545, 385), (15, 160, 220), 2)

    if draw_monster:
        # Yellow nameplate + cyan brackets + mob body
        hsv_yellow = np.uint8([[[25, 170, 200]]])
        bgr_yellow = cv2.cvtColor(hsv_yellow, cv2.COLOR_HSV2BGR)[0][0]
        yellow_tuple = (int(bgr_yellow[0]), int(bgr_yellow[1]), int(bgr_yellow[2]))
        cv2.putText(
            frame,
            "< Stone Gargoyle >",
            (700, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            yellow_tuple,
            2,
        )
        hsv_cyan = np.uint8([[[90, 190, 210]]])
        bgr_cyan = cv2.cvtColor(hsv_cyan, cv2.COLOR_HSV2BGR)[0][0]
        cyan_tuple = (int(bgr_cyan[0]), int(bgr_cyan[1]), int(bgr_cyan[2]))
        cv2.rectangle(frame, (680, 236), (696, 254), cyan_tuple, -1)
        cv2.rectangle(frame, (875, 236), (891, 254), cyan_tuple, -1)
        cv2.rectangle(frame, (690, 260), (880, 370), (50, 50, 95), -1)

    _safe_imwrite_unicode(path, frame)
    return frame


@pytest.fixture
def isolated_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Provide an isolated dataset directory, ApprovedDatasetPool, and YOLOFactoryDialog."""
    dataset_dir = tmp_path / "dataset"
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "review" / split).mkdir(parents=True, exist_ok=True)
    (dataset_dir / "previews").mkdir(parents=True, exist_ok=True)

    pool_root = dataset_dir / "approved"
    pool = ApprovedDatasetPool(pool_root=pool_root, project_root=tmp_path)
    monkeypatch.setattr(approved_pool_mod, "_GLOBAL_POOL", pool)
    monkeypatch.setattr(approved_pool_mod, "get_approved_pool", lambda *_a, **_kw: pool)

    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: dataset_dir  # type: ignore[assignment]
    dlg._learn_box_exemplar = lambda *args, **kwargs: None  # type: ignore[assignment]
    dlg._go_step(2)
    return dlg, dataset_dir, pool


# ============================================================================
# 1. Auto-Labeling -> Review Queue Integration Test
# ============================================================================
def test_1_auto_labeling_to_review_queue_integration(isolated_env) -> None:
    """
    Validate that YOLOAutoLabeler scans a frame, produces calibrated weak/trusted
    detections without artificial 0.85 floors on heuristic fallbacks, and that
    YOLOFactoryDialog._relabel_single_item populates the review queue and JSON on disk.
    """
    dlg, dataset_dir, _pool = isolated_env

    stem = "gather_orichalcum_001"
    img_path = dataset_dir / "images" / "train" / f"{stem}.jpg"
    frame = _create_synthetic_game_frame(img_path, draw_salient_resource=True)

    # 1. Direct YOLOAutoLabeler check on salient 3D resource fallback
    labeler = YOLOAutoLabeler()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    salient_anns = labeler._scan_salient_3d_resources(
        frame_bgr=frame,
        hsv=hsv,
        existing_boxes=[],
        hint_material_name="Orichalcum Ore",
        has_crest_or_prompt=False,
    )
    assert len(salient_anns) >= 1
    for ann in salient_anns:
        assert 0.38 <= float(ann["confidence"]) <= 0.68
        assert ann["trusted"] is False
        assert ann["weak"] is True
        assert int(ann["class_id"]) <= 12

    # 2. Load into dialog review queue and run _relabel_single_item
    dlg._reload_review_queue()
    assert len(dlg._review_items) == 1
    item = dlg._review_items[0]
    assert item["stem"] == stem

    # Inject a deterministic auto-label return to verify dialog categorization (Emin vs Şüpheli)
    simulated_auto_anns = [
        {
            "class_id": 6,
            "class_name": "world_gather_crest",
            "cx": 0.45,
            "cy": 0.45,
            "w": 0.08,
            "h": 0.10,
            "confidence": 0.94,
            "source": "gather_crest_template",
        },
        {
            "class_id": 3,
            "class_name": "ore_orichalcum",
            "cx": 0.41,
            "cy": 0.50,
            "w": 0.10,
            "h": 0.12,
            "confidence": 0.62,
            "source": "salient_3d_resource",
            "trusted": False,
            "weak": True,
        },
    ]
    orig_auto_label = YOLOAutoLabeler.auto_label_frame
    try:
        YOLOAutoLabeler.auto_label_frame = lambda self, *a, **kw: simulated_auto_anns  # type: ignore[assignment]
        n_boxes = dlg._relabel_single_item(item)
    finally:
        YOLOAutoLabeler.auto_label_frame = orig_auto_label  # type: ignore[assignment]

    assert n_boxes == 2
    assert len(item["trusted"]) == 1
    assert len(item["weak"]) == 1
    assert item["trusted"][0]["class_id"] == 6
    assert item["weak"][0]["class_id"] == 3
    assert item["done"] is None

    # Verify review JSON was persisted to disk
    rev_json = dataset_dir / "review" / "train" / f"{stem}.json"
    assert rev_json.is_file()
    saved = json.loads(rev_json.read_text(encoding="utf-8"))
    assert len(saved["trusted"]) == 1
    assert len(saved["weak"]) == 1


# ============================================================================
# 2. 3-Tier Color Coding (Green / Orange / Red) Integration Test
# ============================================================================
def test_2_three_tier_color_coding_green_orange_red_integration(isolated_env) -> None:
    """
    Validate 3-tier color coding in both detail rows and YOLOPreviewCanvas:
    - Tier 1 (Yeşil / Emin): conf >= 0.85 & trusted (or hand_label) -> #4ade80 / #22c55e
    - Tier 2 (Turuncu / Şüpheli): 0.70 <= conf < 0.85 -> #fb923c
    - Tier 3 (Kırmızı / Düşük Güven): conf < 0.70 -> #ef4444 / #f87171
    """
    dlg, dataset_dir, _pool = isolated_env

    stem = "color_tier_sample"
    img_path = dataset_dir / "images" / "train" / f"{stem}.jpg"
    frame = _create_synthetic_game_frame(img_path)

    tier1_green_box = {
        "class_id": 3,
        "class_name": "ore_orichalcum",
        "cx": 0.25,
        "cy": 0.40,
        "w": 0.12,
        "h": 0.16,
        "confidence": 0.93,
        "trusted": True,
        "weak": False,
    }
    tier2_orange_box = {
        "class_id": 0,
        "class_name": "herb_aria",
        "cx": 0.55,
        "cy": 0.45,
        "w": 0.12,
        "h": 0.16,
        "confidence": 0.76,
        "trusted": False,
        "weak": True,
    }
    tier3_red_box = {
        "class_id": 1,
        "class_name": "herb_mela",
        "cx": 0.80,
        "cy": 0.50,
        "w": 0.12,
        "h": 0.16,
        "confidence": 0.48,
        "trusted": False,
        "weak": True,
    }

    rev_path = dataset_dir / "review" / "train" / f"{stem}.json"
    rev_path.write_text(
        json.dumps(
            {
                "stem": stem,
                "split": "train",
                "trusted": [tier1_green_box],
                "weak": [tier2_orange_box, tier3_red_box],
            }
        ),
        encoding="utf-8",
    )

    dlg._reload_review_queue()
    dlg._render_review_frame()

    # Check detail rows for each tier
    row_green = dlg._create_review_det_row(tier1_green_box, is_trusted=True)
    conf_green = row_green.findChild(object, "PipeDetConf")
    assert conf_green is not None
    assert "#4ade80" in conf_green.styleSheet()
    texts_green = [w.text() for w in row_green.findChildren(object) if hasattr(w, "text")]
    assert any("Emin · %93 güven" in t for t in texts_green)

    row_orange = dlg._create_review_det_row(tier2_orange_box, is_trusted=False)
    conf_orange = row_orange.findChild(object, "PipeDetConf")
    assert conf_orange is not None
    assert "#fb923c" in conf_orange.styleSheet()
    texts_orange = [w.text() for w in row_orange.findChildren(object) if hasattr(w, "text")]
    assert any("Şüpheli · %76 güven" in t for t in texts_orange)

    row_red = dlg._create_review_det_row(tier3_red_box, is_trusted=False)
    conf_red = row_red.findChild(object, "PipeDetConf")
    assert conf_red is not None
    assert "#ef4444" in conf_red.styleSheet()
    texts_red = [w.text() for w in row_red.findChildren(object) if hasattr(w, "text")]
    assert any("Düşük Güven (Şüpheli) · %48 güven" in t for t in texts_red)

    # Render YOLOPreviewCanvas onto an offscreen QImage to verify paintEvent & hit rects
    canvas = dlg.review_preview
    canvas.resize(800, 500)
    ok = canvas.set_live_frame(
        frame,
        badge=f"review · {stem}.jpg",
        annotations=[tier1_green_box, tier2_orange_box, tier3_red_box],
    )
    assert ok is True
    rendered = QImage(canvas.size(), QImage.Format.Format_ARGB32_Premultiplied)
    canvas.render(rendered)
    # All 3 annotations must have interactive close hit rects registered during paintEvent
    assert len(canvas._close_hit_rects) == 3


# ============================================================================
# 3. Manual Box Drawing & Tag Search Integration Test
# ============================================================================
def test_3_manual_tag_search_and_box_drawing_integration(isolated_env) -> None:
    """
    Validate bilingual tag search ('orichalcum', 'pot', 'parşömen', 'canavar', 'filiz')
    and interactive manual box drawing on YOLOPreviewCanvas updating the review item
    with a 100% trusted, human-approved box.
    """
    dlg, dataset_dir, _pool = isolated_env

    stem = "manual_draw_frame_01"
    img_path = dataset_dir / "images" / "train" / f"{stem}.jpg"
    frame = _create_synthetic_game_frame(img_path, draw_salient_resource=False)
    rev_path = dataset_dir / "review" / "train" / f"{stem}.json"
    rev_path.write_text(
        json.dumps({"stem": stem, "split": "train", "trusted": [], "weak": []}),
        encoding="utf-8",
    )

    dlg._reload_review_queue()
    assert len(dlg._review_items) == 1
    assert len(dlg._review_items[0]["trusted"]) == 0

    # 1. Verify SearchableTagRegistry & search_label_classes queries
    assert search_label_classes("orichalcum")[0]["class_id"] == 3
    assert search_label_classes("aria")[0]["class_id"] == 0
    assert search_label_classes("pot")[0]["class_id"] == 22
    assert search_label_classes("iksir")[0]["class_id"] == 22
    assert search_label_classes("parşömen")[0]["class_id"] == 24
    assert search_label_classes("canavar")[0]["class_id"] == 10
    assert search_label_classes("filiz")[0]["class_id"] == 9

    manual_helper_box = create_manual_annotation_box("orichalcum", 0.5, 0.5, 0.15, 0.20)
    assert manual_helper_box["class_id"] == 3
    assert manual_helper_box["class_name"] == "ore_orichalcum"
    assert manual_helper_box["trusted"] is True
    assert manual_helper_box["weak"] is False
    assert manual_helper_box["hand_label"] is True
    assert manual_helper_box["human_approved"] is True
    assert manual_helper_box["confidence"] == 1.0

    # 2. Filter manual_tag_combo via manual_tag_search in UI and select class 3 (ore_orichalcum)
    dlg.manual_tag_search.setText("orichalcum")
    QApplication.processEvents()
    assert dlg.manual_tag_combo.count() >= 1
    assert int(dlg.manual_tag_combo.currentData()) == 3

    # Clear focus from search input so canvas / shortcuts behave normally
    dlg.manual_tag_search.clearFocus()

    # 3. Simulate mouse drag on review_preview canvas to draw a manual bounding box
    canvas = dlg.review_preview
    canvas.resize(800, 500)
    canvas.set_live_frame(frame, badge="manual_test", annotations=[])
    canvas.active_drawing_class_id = 3
    canvas.active_drawing_class_name = "ore_orichalcum"

    trect = canvas._compute_target_rect()
    canvas._last_target_rect = trect
    start_pt = QPointF(trect.left() + trect.width() * 0.30, trect.top() + trect.height() * 0.30)
    end_pt = QPointF(trect.left() + trect.width() * 0.50, trect.top() + trect.height() * 0.55)

    press_evt = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        start_pt,
        start_pt,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    move_evt = QMouseEvent(
        QEvent.Type.MouseMove,
        end_pt,
        end_pt,
        Qt.MouseButton.NoButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    release_evt = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        end_pt,
        end_pt,
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    canvas.mousePressEvent(press_evt)
    canvas.mouseMoveEvent(move_evt)
    canvas.mouseReleaseEvent(release_evt)
    QApplication.processEvents()

    # Verify manual box was added to current review item's trusted list and saved to disk
    current_item = dlg._review_items[dlg._review_index]
    assert len(current_item["trusted"]) == 1
    added_box = current_item["trusted"][0]
    assert added_box["class_id"] == 3
    assert added_box["class_name"] == "ore_orichalcum"
    assert added_box["trusted"] is True
    assert added_box["weak"] is False
    assert added_box["hand_label"] is True
    assert added_box["human_approved"] is True
    assert float(added_box["confidence"]) == 1.0
    assert abs(float(added_box["cx"]) - 0.40) < 0.03
    assert abs(float(added_box["cy"]) - 0.425) < 0.03
    assert abs(float(added_box["w"]) - 0.20) < 0.03
    assert abs(float(added_box["h"]) - 0.25) < 0.03

    saved_payload = json.loads(rev_path.read_text(encoding="utf-8"))
    assert len(saved_payload["trusted"]) == 1
    assert saved_payload["trusted"][0]["class_id"] == 3
    assert saved_payload["trusted"][0]["hand_label"] is True


# ============================================================================
# 4. Approval, Queue Deduction & Reopen Persistence Guard Integration Test
# ============================================================================
def test_4_approve_and_queue_deduction_persistence_integration(isolated_env) -> None:
    """
    Validate that approving an item ('A' shortcut or button):
    - Promotes weak boxes to trusted with human_approved=True, done='ok'
    - Deducts the approved item from the active review queue (_review_items & pending count)
    - Adds the sample to ApprovedDatasetPool
    - Never reloads the approved item back into the queue when _reload_review_queue() is called
    """
    dlg, dataset_dir, pool = isolated_env

    stems = ["queue_item_01", "queue_item_02", "queue_item_03"]
    for idx, stem in enumerate(stems):
        img_path = dataset_dir / "images" / "train" / f"{stem}.jpg"
        _create_synthetic_game_frame(img_path)
        rev_path = dataset_dir / "review" / "train" / f"{stem}.json"
        rev_path.write_text(
            json.dumps(
                {
                    "stem": stem,
                    "split": "train",
                    "trusted": [],
                    "weak": [
                        {
                            "class_id": idx,
                            "class_name": get_tag_registry().get_by_id(idx)["class_name"],
                            "cx": 0.5,
                            "cy": 0.5,
                            "w": 0.15,
                            "h": 0.15,
                            "confidence": 0.74,
                            "trusted": False,
                            "weak": True,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

    dlg._reload_review_queue()
    assert len(dlg._review_items) == 3
    assert dlg._pending_count() == 3
    assert dlg.queue_list.count() == 3

    first_stem = dlg._review_items[0]["stem"]

    # Approve first item via 'A' keyPressEvent
    key_a = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier, "A")
    dlg.keyPressEvent(key_a)
    QApplication.processEvents()

    # Queue count must decrease from 3 -> 2
    assert len(dlg._review_items) == 2
    assert dlg._pending_count() == 2
    assert dlg.queue_list.count() == 2
    assert first_stem not in [it["stem"] for it in dlg._review_items]
    assert pool.is_approved(first_stem) is True

    # Approve second item via _on_step2_approve()
    second_stem = dlg._review_items[0]["stem"]
    dlg._on_step2_approve()
    QApplication.processEvents()

    assert len(dlg._review_items) == 1
    assert dlg._pending_count() == 1
    assert dlg.queue_list.count() == 1
    assert second_stem not in [it["stem"] for it in dlg._review_items]
    assert pool.is_approved(second_stem) is True

    remaining_stem = dlg._review_items[0]["stem"]
    assert pool.is_approved(remaining_stem) is False

    # Simulate closing and reopening the dialog by reloading review queue from disk
    dlg._reload_review_queue()
    assert len(dlg._review_items) == 1
    assert dlg._pending_count() == 1
    assert [it["stem"] for it in dlg._review_items] == [remaining_stem]


# ============================================================================
# 5. Full End-to-End Pipeline -> ApprovedDatasetPool & data_approved.yaml Test
# ============================================================================
def test_5_end_to_end_pipeline_to_approved_dataset_pool_and_yaml(isolated_env) -> None:
    """
    Validate the full end-to-end flow:
    Auto-labeling -> 3-tier color verification -> Manual box drawing & tag search ->
    Approving & queue deduction -> ApprovedDatasetPool.prepare_approved_training_dataset ->
    Verifying `data_approved.yaml` and isolated `dataset/approved/` contain ONLY approved
    frames and trusted/manual boxes (strictly excluding unapproved frames and weak boxes).
    """
    dlg, dataset_dir, pool = isolated_env

    # Create 3 frames:
    # - e2e_frame_auto_approved: auto-labeled weak box -> user approves in Step 2
    # - e2e_frame_manual_approved: empty frame -> user searches tag & draws manual box -> approves
    # - e2e_frame_unapproved_suspect: suspicious weak box -> left UNAPPROVED in queue
    stems = [
        "e2e_frame_auto_approved",
        "e2e_frame_manual_approved",
        "e2e_frame_unapproved_suspect",
    ]
    for stem in stems:
        img_path = dataset_dir / "images" / "train" / f"{stem}.jpg"
        _create_synthetic_game_frame(img_path)

    # Seed review JSONs representing auto-labeler output
    (dataset_dir / "review" / "train" / "e2e_frame_auto_approved.json").write_text(
        json.dumps(
            {
                "stem": "e2e_frame_auto_approved",
                "split": "train",
                "trusted": [
                    {
                        "class_id": 0,
                        "class_name": "herb_aria",
                        "cx": 0.40,
                        "cy": 0.45,
                        "w": 0.14,
                        "h": 0.18,
                        "confidence": 0.92,
                        "trusted": True,
                        "weak": False,
                    }
                ],
                "weak": [
                    {
                        "class_id": 9,
                        "class_name": "minimap_sprout",
                        "cx": 0.88,
                        "cy": 0.15,
                        "w": 0.04,
                        "h": 0.04,
                        "confidence": 0.78,
                        "trusted": False,
                        "weak": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "review" / "train" / "e2e_frame_manual_approved.json").write_text(
        json.dumps(
            {
                "stem": "e2e_frame_manual_approved",
                "split": "train",
                "trusted": [],
                "weak": [],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "review" / "train" / "e2e_frame_unapproved_suspect.json").write_text(
        json.dumps(
            {
                "stem": "e2e_frame_unapproved_suspect",
                "split": "train",
                "trusted": [],
                "weak": [
                    {
                        "class_id": 14,
                        "class_name": "obstacle",
                        "cx": 0.50,
                        "cy": 0.50,
                        "w": 0.30,
                        "h": 0.30,
                        "confidence": 0.44,
                        "trusted": False,
                        "weak": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dlg._reload_review_queue()
    assert len(dlg._review_items) == 3

    # Step A: Locate and approve 'e2e_frame_auto_approved'
    idx_auto = next(i for i, it in enumerate(dlg._review_items) if it["stem"] == "e2e_frame_auto_approved")
    dlg._review_index = idx_auto
    dlg._render_review_frame()
    dlg._on_step2_approve()
    assert len(dlg._review_items) == 2
    assert pool.is_approved("e2e_frame_auto_approved") is True

    # Step B: Select 'e2e_frame_manual_approved', search tag 'canavar' (class 10: monster_3d),
    # draw manual box, and approve
    idx_manual = next(i for i, it in enumerate(dlg._review_items) if it["stem"] == "e2e_frame_manual_approved")
    dlg._review_index = idx_manual
    dlg._render_review_frame()

    dlg.manual_tag_search.setText("canavar")
    QApplication.processEvents()
    assert int(dlg.manual_tag_combo.currentData()) == 10
    dlg.manual_tag_search.clearFocus()

    dlg._on_manual_box_added(
        {
            "class_id": 10,
            "class_name": "monster_3d",
            "cx": 0.60,
            "cy": 0.55,
            "w": 0.18,
            "h": 0.22,
        }
    )
    assert len(dlg._review_items[dlg._review_index]["trusted"]) == 1
    dlg._on_step2_approve()

    # Queue must now only have the 1 unapproved suspect frame left
    assert len(dlg._review_items) == 1
    assert dlg._review_items[0]["stem"] == "e2e_frame_unapproved_suspect"
    assert pool.is_approved("e2e_frame_manual_approved") is True
    assert pool.is_approved("e2e_frame_unapproved_suspect") is False

    # Step C: Prepare Approved Training Dataset & generate data_approved.yaml
    summary = pool.prepare_approved_training_dataset(
        source_dataset_dir=dataset_dir,
        val_ratio=0.15,
        force_rebalance=False,
    )

    assert summary["total_approved_images"] == 2
    assert summary["total_boxes"] == 3  # 2 from auto_approved (1 trusted + 1 promoted) + 1 manual box
    assert pool.get_approved_stems() == {"e2e_frame_auto_approved", "e2e_frame_manual_approved"}
    assert not (pool.images_train_dir / "e2e_frame_unapproved_suspect.jpg").exists()
    assert not (pool.labels_train_dir / "e2e_frame_unapproved_suspect.txt").exists()

    # Verify YOLO label files in approved pool
    lbl_auto = (pool.labels_train_dir / "e2e_frame_auto_approved.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lbl_auto) == 2
    assert lbl_auto[0].startswith("0 0.400000 0.450000 0.140000 0.180000")
    assert lbl_auto[1].startswith("9 0.880000 0.150000 0.040000 0.040000")

    lbl_manual = (pool.labels_train_dir / "e2e_frame_manual_approved.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lbl_manual) == 1
    assert lbl_manual[0].startswith("10 0.600000 0.550000 0.180000 0.220000")

    # Verify data_approved.yaml structure and canonical 22 training classes
    yaml_path = Path(summary["data_yaml_path"])
    assert yaml_path.is_file()
    assert yaml_path.name == "data_approved.yaml"
    parsed_yaml = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert parsed_yaml["path"] == pool.pool_root.as_posix()
    assert parsed_yaml["train"] == "images/train"
    assert parsed_yaml["val"] == "images/val"
    assert parsed_yaml["nc"] == 22
    for cid, cname in CORE_22_CANONICAL_NAMES.items():
        assert parsed_yaml["names"][cid] == cname
