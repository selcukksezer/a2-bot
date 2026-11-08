# -*- coding: utf-8 -*-
"""
tests/test_w46_filter_scoped_bulk_actions.py

Worker W46 Unit Tests:
Verifies that when a queue filter is selected ('suspicious', 'empty', 'confident',
'mixed', 'below', 'above'), bulk reject (_bulk_reject_filtered), bulk approve
(_bulk_approve), and bulk prune (_bulk_prune_low_confidence_boxes) strictly affect
ONLY the frames matching the active filter and leave all other filters untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest
from PyQt6.QtWidgets import QApplication

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.yolo_factory_dialog import YOLOFactoryDialog


@pytest.fixture
def factory_dialog(tmp_path: Path) -> YOLOFactoryDialog:
    """Provides an isolated YOLOFactoryDialog pointed at tmp_path."""
    for _missing_cb in (
        "_on_manual_tag_search_changed",
        "_populate_manual_tag_combo",
        "_on_manual_box_added",
    ):
        if not hasattr(YOLOFactoryDialog, _missing_cb):
            setattr(YOLOFactoryDialog, _missing_cb, lambda self, *a, **kw: None)
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    return dlg


def _build_diverse_queue(tmp_path: Path) -> List[Dict[str, Any]]:
    """Creates a review queue covering suspicious, empty, confident, mixed, below, and above."""
    specs = [
        # Pure suspicious (weak only, conf < 0.85 -> also 'below' when threshold=85)
        {
            "stem": "item_susp_low",
            "weak": [{"confidence": 0.45, "class_id": 0, "bbox": [10, 10, 40, 40]}],
            "trusted": [],
        },
        # Pure suspicious with 1 low weak box and 1 high weak box (for pruning in suspicious mode)
        {
            "stem": "item_susp_pair",
            "weak": [
                {"confidence": 0.35, "class_id": 1, "bbox": [12, 12, 32, 32]},
                {"confidence": 0.92, "class_id": 2, "bbox": [50, 50, 80, 80]},
            ],
            "trusted": [],
        },
        # Empty (0 weak, 0 trusted -> conf=0.0, also 'below')
        {
            "stem": "item_empty_1",
            "weak": [],
            "trusted": [],
        },
        # Pure confident (0 weak, 1 trusted with conf=0.96 -> also 'above' when threshold=85)
        {
            "stem": "item_conf_high",
            "weak": [],
            "trusted": [{"confidence": 0.96, "class_id": 3, "bbox": [25, 25, 75, 75]}],
        },
        # Mixed (1 weak low conf=0.40, 1 trusted high conf=0.95)
        {
            "stem": "item_mixed_1",
            "weak": [{"confidence": 0.40, "class_id": 1, "bbox": [15, 15, 45, 45]}],
            "trusted": [{"confidence": 0.95, "class_id": 0, "bbox": [55, 55, 95, 95]}],
        },
    ]

    items: List[Dict[str, Any]] = []
    for sp in specs:
        stem = sp["stem"]
        rev_path = tmp_path / "review" / "train" / f"{stem}.json"
        rev_path.write_text("{}", encoding="utf-8")
        img_path = tmp_path / "images" / "train" / f"{stem}.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
        lbl_path = tmp_path / "labels" / "train" / f"{stem}.txt"
        lbl_path.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

        items.append(
            {
                "stem": stem,
                "split": "train",
                "weak": [dict(b) for b in sp["weak"]],
                "trusted": [dict(b) for b in sp["trusted"]],
                "done": None,
                "path": str(rev_path),
            }
        )
    return items


def test_bulk_reject_scoped_per_filter_suspicious_empty_confident_mixed(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """1. Verifies _bulk_reject_filtered isolates deletions strictly to the selected
    categorical filter ('suspicious', 'empty', 'confident', 'mixed') without touching others."""
    dlg = factory_dialog

    # --- A) 'suspicious' filter reject ---
    dlg._review_items = _build_diverse_queue(tmp_path)
    dlg.set_queue_filter_mode("suspicious")
    assert [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()] == [
        "item_susp_low",
        "item_susp_pair",
    ]
    dlg._bulk_reject_filtered()
    remaining = {it["stem"] for it in dlg._review_items}
    assert remaining == {"item_empty_1", "item_conf_high", "item_mixed_1"}
    assert not (tmp_path / "images" / "train" / "item_susp_low.png").exists()
    assert not (tmp_path / "images" / "train" / "item_susp_pair.png").exists()
    assert (tmp_path / "images" / "train" / "item_empty_1.png").exists()
    assert (tmp_path / "images" / "train" / "item_conf_high.png").exists()
    assert (tmp_path / "images" / "train" / "item_mixed_1.png").exists()

    # --- B) 'empty' filter reject (from remaining queue) ---
    dlg.set_queue_filter_mode("empty")
    assert [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()] == [
        "item_empty_1"
    ]
    dlg._bulk_reject_filtered()
    remaining = {it["stem"] for it in dlg._review_items}
    assert remaining == {"item_conf_high", "item_mixed_1"}
    assert not (tmp_path / "images" / "train" / "item_empty_1.png").exists()
    assert (tmp_path / "images" / "train" / "item_conf_high.png").exists()
    assert (tmp_path / "images" / "train" / "item_mixed_1.png").exists()

    # --- C) 'mixed' filter reject (from remaining queue) ---
    dlg.set_queue_filter_mode("mixed")
    assert [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()] == [
        "item_mixed_1"
    ]
    dlg._bulk_reject_filtered()
    remaining = {it["stem"] for it in dlg._review_items}
    assert remaining == {"item_conf_high"}
    assert not (tmp_path / "images" / "train" / "item_mixed_1.png").exists()
    assert (tmp_path / "images" / "train" / "item_conf_high.png").exists()

    # --- D) 'confident' filter reject on fresh queue ---
    dlg._review_items = _build_diverse_queue(tmp_path)
    dlg.set_queue_filter_mode("confident")
    assert [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()] == [
        "item_conf_high"
    ]
    dlg._bulk_reject_filtered()
    remaining = {it["stem"] for it in dlg._review_items}
    assert "item_conf_high" not in remaining
    assert remaining == {"item_susp_low", "item_susp_pair", "item_empty_1", "item_mixed_1"}


def test_bulk_reject_scoped_per_filter_below_and_above_threshold(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """2. Verifies _bulk_reject_filtered strictly respects 'below' and 'above' threshold filters."""
    dlg = factory_dialog
    dlg.bulk_range.setValue(85)  # threshold = 0.85

    # --- A) 'above' filter reject (conf >= 0.85: only item_conf_high has min/avg conf 0.96) ---
    dlg._review_items = _build_diverse_queue(tmp_path)
    dlg.set_queue_filter_mode("above")
    above_stems = [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()]
    assert above_stems == ["item_conf_high"]

    dlg._bulk_reject_filtered()
    remaining_after_above = {it["stem"] for it in dlg._review_items}
    assert "item_conf_high" not in remaining_after_above
    assert remaining_after_above == {
        "item_susp_low",
        "item_susp_pair",
        "item_empty_1",
        "item_mixed_1",
    }
    for kept_stem in remaining_after_above:
        assert (tmp_path / "images" / "train" / f"{kept_stem}.png").exists()

    # --- B) 'below' filter reject on fresh queue (conf < 0.85) ---
    dlg._review_items = _build_diverse_queue(tmp_path)
    dlg.set_queue_filter_mode("below")
    below_stems = {dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()}
    assert "item_conf_high" not in below_stems
    assert below_stems == {
        "item_susp_low",
        "item_susp_pair",
        "item_empty_1",
        "item_mixed_1",
    }

    dlg._bulk_reject_filtered()
    remaining_after_below = [it["stem"] for it in dlg._review_items]
    assert remaining_after_below == ["item_conf_high"]
    assert (tmp_path / "images" / "train" / "item_conf_high.png").exists()


def test_bulk_approve_scoped_across_suspicious_empty_confident_mixed(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """3. Verifies _bulk_approve only approves/promotes frames in the active filter
    ('confident', 'empty', 'suspicious', 'mixed') and leaves all other frames unapproved."""
    dlg = factory_dialog
    dlg.bulk_range.setValue(85)  # threshold = 0.85
    dlg._review_items = _build_diverse_queue(tmp_path)
    by_stem = {it["stem"]: it for it in dlg._review_items}

    # 1) 'confident' filter -> only item_conf_high becomes done="ok"
    dlg.set_queue_filter_mode("confident")
    dlg._bulk_approve()
    assert by_stem["item_conf_high"]["done"] == "ok"
    assert by_stem["item_susp_low"]["done"] is None
    assert by_stem["item_susp_pair"]["done"] is None
    assert by_stem["item_empty_1"]["done"] is None
    assert by_stem["item_mixed_1"]["done"] is None

    # 2) 'empty' filter -> only item_empty_1 becomes done="ok"
    dlg.set_queue_filter_mode("empty")
    dlg._bulk_approve()
    assert by_stem["item_empty_1"]["done"] == "ok"
    assert by_stem["item_susp_low"]["done"] is None
    assert by_stem["item_susp_pair"]["done"] is None
    assert by_stem["item_mixed_1"]["done"] is None

    # 3) 'suspicious' filter -> promotes weak boxes in item_susp_low & item_susp_pair, leaves item_mixed_1 untouched
    dlg.set_queue_filter_mode("suspicious")
    dlg._bulk_approve()
    assert by_stem["item_susp_low"]["done"] == "ok"
    assert len(by_stem["item_susp_low"]["weak"]) == 0
    assert len(by_stem["item_susp_low"]["trusted"]) == 1
    assert by_stem["item_susp_pair"]["done"] == "ok"
    assert len(by_stem["item_susp_pair"]["weak"]) == 0
    assert len(by_stem["item_susp_pair"]["trusted"]) == 2
    # Mixed frame must still be untouched!
    assert by_stem["item_mixed_1"]["done"] is None
    assert len(by_stem["item_mixed_1"]["weak"]) == 1
    assert len(by_stem["item_mixed_1"]["trusted"]) == 1

    # 4) 'mixed' filter with lowered threshold (30%) -> approves item_mixed_1
    dlg.bulk_range.setValue(30)
    dlg.set_queue_filter_mode("mixed")
    dlg._bulk_approve()
    assert by_stem["item_mixed_1"]["done"] == "ok"
    assert len(by_stem["item_mixed_1"]["weak"]) == 0
    assert len(by_stem["item_mixed_1"]["trusted"]) == 2


def test_bulk_approve_scoped_across_below_and_above_filters(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """4. Verifies _bulk_approve under 'above' and 'below' filters affects only matching frames."""
    dlg = factory_dialog
    dlg.bulk_range.setValue(85)  # threshold = 0.85
    dlg._review_items = _build_diverse_queue(tmp_path)
    by_stem = {it["stem"]: it for it in dlg._review_items}

    # --- A) 'above' filter approve ---
    dlg.set_queue_filter_mode("above")
    assert [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()] == [
        "item_conf_high"
    ]
    dlg._bulk_approve()
    assert by_stem["item_conf_high"]["done"] == "ok"
    for other in ("item_susp_low", "item_susp_pair", "item_empty_1", "item_mixed_1"):
        assert by_stem[other]["done"] is None

    # --- B) 'below' filter approve on fresh queue ---
    dlg._review_items = _build_diverse_queue(tmp_path)
    by_stem = {it["stem"]: it for it in dlg._review_items}
    dlg.bulk_range.setValue(85)
    dlg.set_queue_filter_mode("below")
    dlg._bulk_approve()

    # item_conf_high is 'above' (>= 0.85), so it MUST NOT be touched when 'below' is active
    assert by_stem["item_conf_high"]["done"] is None
    # item_susp_pair had a 0.92 weak box (>= 0.85) so that box was promoted while 0.35 stayed weak
    assert len(by_stem["item_susp_pair"]["trusted"]) == 1
    assert len(by_stem["item_susp_pair"]["weak"]) == 1


def test_bulk_prune_scoped_across_suspicious_confident_mixed_below_above_and_empty_guard(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """5. Verifies _bulk_prune_low_confidence_boxes strictly prunes only the active filter
    across 'mixed', 'suspicious', 'below', 'above', 'confident', and disables prune in 'empty'."""
    dlg = factory_dialog
    dlg.bulk_range.setValue(85)  # threshold = 0.85
    dlg._review_items = _build_diverse_queue(tmp_path)
    by_stem = {it["stem"]: it for it in dlg._review_items}

    # 1) 'empty' filter: prune button is disabled and no boxes exist to prune
    dlg.set_queue_filter_mode("empty")
    assert dlg.btn_prune_low_confidence.isEnabled() is False

    # 2) 'mixed' filter: only item_mixed_1 is pruned; suspicious items keep their weak boxes
    dlg.set_queue_filter_mode("mixed")
    assert [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()] == [
        "item_mixed_1"
    ]
    dlg._bulk_prune_low_confidence_boxes()

    assert by_stem["item_mixed_1"]["done"] == "ok"
    assert len(by_stem["item_mixed_1"]["weak"]) == 0
    assert len(by_stem["item_mixed_1"]["trusted"]) == 1
    # Suspicious items were outside 'mixed' filter and must still have their weak boxes intact!
    assert by_stem["item_susp_low"]["done"] is None
    assert len(by_stem["item_susp_low"]["weak"]) == 1
    assert by_stem["item_susp_pair"]["done"] is None
    assert len(by_stem["item_susp_pair"]["weak"]) == 2
    assert by_stem["item_conf_high"]["done"] is None

    # 3) 'suspicious' filter: prunes item_susp_low (0.45 removed) and item_susp_pair (0.35 removed, 0.92 promoted)
    dlg.set_queue_filter_mode("suspicious")
    dlg._bulk_prune_low_confidence_boxes()
    assert by_stem["item_susp_low"]["done"] == "ok"
    assert len(by_stem["item_susp_low"]["weak"]) == 0
    assert len(by_stem["item_susp_low"]["trusted"]) == 0
    assert by_stem["item_susp_pair"]["done"] == "ok"
    assert len(by_stem["item_susp_pair"]["weak"]) == 0
    assert len(by_stem["item_susp_pair"]["trusted"]) == 1
    # Confident item was outside 'suspicious' filter and remains unapproved
    assert by_stem["item_conf_high"]["done"] is None

    # 4) 'above' / 'confident' vs 'below' isolation on a fresh queue
    dlg._review_items = _build_diverse_queue(tmp_path)
    by_stem = {it["stem"]: it for it in dlg._review_items}
    dlg.set_queue_filter_mode("above")
    dlg._bulk_prune_low_confidence_boxes()
    assert by_stem["item_conf_high"]["done"] == "ok"
    assert len(by_stem["item_susp_low"]["weak"]) == 1
    assert len(by_stem["item_mixed_1"]["weak"]) == 1

    dlg.set_queue_filter_mode("below")
    dlg._bulk_prune_low_confidence_boxes()
    assert len(by_stem["item_susp_low"]["weak"]) == 0
    assert len(by_stem["item_mixed_1"]["weak"]) == 0
    # All image files on disk must still exist (pruning never deletes images)
    for stem in ("item_susp_low", "item_susp_pair", "item_empty_1", "item_conf_high", "item_mixed_1"):
        assert (tmp_path / "images" / "train" / f"{stem}.png").exists()
