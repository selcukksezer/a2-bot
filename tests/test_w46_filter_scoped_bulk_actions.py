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
    """Provides an isolated YOLOFactoryDialog pointed at tmp_path.

    NOTE (D89/F-2056): no production method is monkeypatched here. An earlier
    draft silently stapled no-op lambdas onto the class for any "missing"
    callback -- including `_on_manual_tag_search_changed`, a method that
    neither exists in production nor is ever called by it. That hid a real
    interface mismatch behind a fake. The callbacks this suite relies on must
    exist in production; if one is ever removed, this fixture fails loudly.
    """
    for _required_cb in ("_populate_manual_tag_combo", "_on_manual_box_added"):
        assert hasattr(YOLOFactoryDialog, _required_cb), (
            f"YOLOFactoryDialog.{_required_cb} is required by this suite but "
            "is missing from production -- do not paper over it with a no-op"
        )
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    # [YOLO-ASYNC] __init__'teki başlangıç yüklemesi (üretim dizini) test
    # gövdesi kuyruğu okumadan önce bitmeli; yoksa bitince kuyruğu sıfırlar.
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
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


def _build_diverse_queue(tmp_path: Path) -> List[Dict[str, Any]]:
    """Creates a review queue covering suspicious, empty, confident, mixed, below, and above."""
    specs = [
        # Pure suspicious (weak only, conf < 0.85 -> also 'below' when threshold=85)
        {
            "stem": "item_susp_low",
            "weak": [_box(0.45, 0, [10, 10, 40, 40])],
            "trusted": [],
        },
        # Pure suspicious with 1 low weak box and 1 high weak box (for pruning in suspicious mode)
        {
            "stem": "item_susp_pair",
            "weak": [
                _box(0.35, 1, [12, 12, 32, 32]),
                _box(0.92, 2, [50, 50, 80, 80]),
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
            "trusted": [_box(0.96, 3, [25, 25, 75, 75])],
        },
        # Mixed (1 weak low conf=0.40, 1 trusted high conf=0.95)
        {
            "stem": "item_mixed_1",
            "weak": [_box(0.40, 1, [15, 15, 45, 45])],
            "trusted": [_box(0.95, 0, [55, 55, 95, 95])],
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


def _filtered_stems(dlg: YOLOFactoryDialog) -> List[str]:
    """Stems of the items matching the dialog's active queue filter.

    The 8 inline ``[dlg._review_items[i]["stem"] for i in ...]``
    comprehensions this replaces were identical; one helper keeps the
    filter-assertion intent in one place (same pattern already used in
    test_label_trust_emin_supheli.py).
    """
    return [dlg._review_items[i]["stem"] for i in dlg._filtered_review_indices()]


def test_bulk_reject_scoped_per_filter_suspicious_empty_confident_mixed(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """1. Verifies _bulk_reject_filtered isolates deletions strictly to the selected
    categorical filter ('suspicious', 'empty', 'confident', 'mixed') without touching others."""
    dlg = factory_dialog

    # --- A) 'suspicious' filter reject ---
    dlg._review_items = _build_diverse_queue(tmp_path)
    dlg.set_queue_filter_mode("suspicious")
    assert _filtered_stems(dlg) == [
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
    assert _filtered_stems(dlg) == [
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
    assert _filtered_stems(dlg) == [
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
    assert _filtered_stems(dlg) == [
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
    above_stems = _filtered_stems(dlg)
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
    below_stems = set(_filtered_stems(dlg))
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

    # 2) 'empty' filter -> item_empty_1 is NOT approved (KARAR B, v12):
    # B-02 ön-kontrolü sıfır geçerli kutuyla toplu onayı reddeder
    # (_persist_review_item False döner -> done geri alınır). Red artık
    # DOĞRU sonuçtur: boş kare onaylanmaz, kuyrukta kalır, diske de
    # onaylı yazılmaz; diğer karelere dokunulmaz.
    dlg.set_queue_filter_mode("empty")
    dlg._bulk_approve()
    assert by_stem["item_empty_1"]["done"] is None
    assert any(it["stem"] == "item_empty_1" for it in dlg._review_items)
    assert "ok" not in (tmp_path / "review" / "train" / "item_empty_1.json").read_text(encoding="utf-8")
    assert by_stem["item_susp_low"]["done"] is None
    assert by_stem["item_susp_pair"]["done"] is None
    assert by_stem["item_mixed_1"]["done"] is None

    # 3) 'suspicious' filter -> only above-threshold weak boxes are bulk-marked.
    # F-0609 [O3]: the old threshold bypass is gone, so at th=0.85 neither
    # suspicious item is approved: item_susp_low keeps its 0.45 box.
    # B-03: item_susp_pair's 0.92 box is NOT promoted to trusted — it gets
    # a `bulk_approved` mark and stays weak (2 weak, 0 trusted); the 0.35
    # box stays weak as well, so the frame stays pending (done is None).
    dlg.set_queue_filter_mode("suspicious")
    dlg._bulk_approve()
    assert by_stem["item_susp_low"]["done"] is None
    assert len(by_stem["item_susp_low"]["weak"]) == 1
    assert len(by_stem["item_susp_low"]["trusted"]) == 0
    assert by_stem["item_susp_pair"]["done"] is None
    assert len(by_stem["item_susp_pair"]["weak"]) == 2
    assert len(by_stem["item_susp_pair"]["trusted"]) == 0
    # Mixed frame must still be untouched!
    assert by_stem["item_mixed_1"]["done"] is None
    assert len(by_stem["item_mixed_1"]["weak"]) == 1
    assert len(by_stem["item_mixed_1"]["trusted"]) == 1

    # 4) 'mixed' filter with lowered threshold (30%) -> approves item_mixed_1.
    # B-03: the 0.40 weak box is bulk-marked, NOT promoted — it stays weak
    # (1 weak) while the 0.95 box stays trusted (1 trusted); the frame
    # itself is approved (done="ok") because no unmarked weak box remains.
    dlg.bulk_range.setValue(30)
    dlg.set_queue_filter_mode("mixed")
    dlg._bulk_approve()
    assert by_stem["item_mixed_1"]["done"] == "ok"
    assert len(by_stem["item_mixed_1"]["weak"]) == 1
    assert len(by_stem["item_mixed_1"]["trusted"]) == 1


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
    assert _filtered_stems(dlg) == [
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
    # B-03: item_susp_pair's 0.92 weak box (>= 0.85) is bulk-marked and stays
    # weak while 0.35 stays weak — no promotion to trusted.
    assert len(by_stem["item_susp_pair"]["trusted"]) == 0
    assert len(by_stem["item_susp_pair"]["weak"]) == 2


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
    assert _filtered_stems(dlg) == [
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
    # item_susp_low ends up with ZERO boxes: its only box was pruned and none
    # was promoted. It must NOT be marked done="ok" (approved) -- an emptied
    # frame is not an approved frame; it stays pending for re-label/empty
    # confirmation (D89/F-2055). NOTE: this assertion currently FAILS against
    # production: _bulk_prune_low_confidence_boxes marks every pruned frame
    # done="ok" unconditionally (ui/yolo_factory_dialog.py, D29). It is pinned
    # here as the correct behavior; the production fix belongs to D29.
    assert by_stem["item_susp_low"]["done"] is None
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
