# -*- coding: utf-8 -*-
"""R35 regression tests (fix2): F-0610, S-1501, S-1503.

- F-0610: external prune rewrites review JSONs on disk; the dialog must
  reload the review queue so in-memory weak boxes / done state are not stale.
- S-1501: ready/version/diagnostic helpers must measure real state via the
  health-summary probe instead of unconditionally returning True/"ok".
- S-1503: the fallback prune loop must teach every pruned box as a negative
  exemplar (the `total <= 20` cap from bulk approve / F-0775 is gone here too).
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
    dlg = YOLOFactoryDialog()
    dlg._auto_confirm_dialogs = True
    dlg._dataset_dir = lambda: tmp_path  # type: ignore[assignment]
    (tmp_path / "review" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "images" / "train").mkdir(parents=True, exist_ok=True)
    (tmp_path / "previews").mkdir(parents=True, exist_ok=True)
    dlg.bulk_range.setValue(85)
    # [YOLO-ASYNC] __init__'teki başlangıç yüklemesi (üretim dizini) test
    # gövdesi kuyruğu okumadan önce bitmeli; yoksa bitince kuyruğu sıfırlar.
    assert dlg.wait_for_yolo_load(), "async kuyruk yüklemesi 10 sn'de bitmedi"
    return dlg


def _dummy_png(path: Path) -> None:
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)


def test_f0610_external_prune_reloads_review_queue(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """External prune succeeds -> dialog memory must match the disk JSONs.

    Fixture kutuları kanonik cx/cy/w/h biçimindedir (P0-1 fixup: eski
    ``bbox`` [piksel] biçimindeki kutuyu uzlaştırma slot eşleşmesi
    göremez — TXT satırıyla aynı sınıf/geometri olsa bile — ve TXT
    kutusunu ikinci trusted olarak eklerdi; hiçbir üretim yazıcısı
    review JSON'a ``bbox`` biçiminde kutu yazmaz).
    """
    dlg = factory_dialog
    stem = "frame_prune_reload"
    rev_path = tmp_path / "review" / "train" / f"{stem}.json"
    rev_path.write_text(
        json.dumps(
            {
                "stem": stem,
                "split": "train",
                "trusted": [
                    {"confidence": 0.95, "class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}
                ],
                "weak": [
                    {"confidence": 0.35, "class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}
                ],
            }
        ),
        encoding="utf-8",
    )
    _dummy_png(tmp_path / "images" / "train" / f"{stem}.png")
    (tmp_path / "labels" / "train" / f"{stem}.txt").write_text(
        "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
    )
    old_item: Dict[str, Any] = {
        "stem": stem,
        "split": "train",
        "weak": [{"confidence": 0.35, "class_id": 1, "cx": 0.3, "cy": 0.3, "w": 0.1, "h": 0.1}],
        "trusted": [{"confidence": 0.95, "class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2}],
        "done": None,
        "path": str(rev_path),
    }
    dlg._review_items = [old_item]
    dlg.set_queue_filter_mode("all")

    dlg._bulk_prune_low_confidence_boxes()

    # Disk is the truth: the 0.35 box is gone.
    saved = json.loads(rev_path.read_text(encoding="utf-8"))
    assert all(
        float(b.get("confidence", 0.0)) != 0.35
        for b in saved.get("trusted", []) + saved.get("weak", [])
    )
    # Memory must now agree with disk (F-0610): the stale in-memory item
    # still carries the 0.35 weak box, so the queue must have been reloaded.
    assert len(dlg._review_items) == 1
    current = dlg._review_items[0]
    assert current is not old_item
    assert all(
        float(b.get("confidence", 0.0)) != 0.35
        for b in current.get("trusted", []) + current.get("weak", [])
    )
    assert len(current.get("trusted", [])) + len(current.get("weak", [])) == 1


def test_s1501_ready_flags_measure_real_state(
    factory_dialog: YOLOFactoryDialog,
) -> None:
    """ready/version/diagnostic follow the health-summary probe."""
    dlg = factory_dialog
    dlg.factory_controller = None
    assert dlg.get_yolo_factory_dialog_ready_flag()["ready"] is False
    assert dlg.get_yolo_factory_dialog_version_info()["ready"] is False
    snap = dlg.get_yolo_factory_dialog_diagnostic_snapshot()
    assert snap["ready"] is False
    assert snap["status"] == "degraded"

    dlg.factory_controller = object()
    assert dlg.get_yolo_factory_dialog_ready_flag()["ready"] is True
    assert dlg.get_yolo_factory_dialog_version_info()["ready"] is True
    assert dlg.get_yolo_factory_dialog_version_info()["version"] == "2.0.0"
    snap = dlg.get_yolo_factory_dialog_diagnostic_snapshot()
    assert snap["ready"] is True
    assert snap["status"] == "ok"


def test_s1503_fallback_prune_teaches_all_pruned_boxes(
    factory_dialog: YOLOFactoryDialog, tmp_path: Path
) -> None:
    """Fallback path with >20 targets: every pruned box is taught (no cap)."""
    dlg = factory_dialog
    taught: List[Dict[str, Any]] = []

    def _record(it: Dict[str, Any], box: Dict[str, Any], **kwargs: Any) -> None:
        taught.append({"stem": it.get("stem"), "box": dict(box), **kwargs})

    dlg._learn_box_exemplar = _record  # type: ignore[method-assign]
    items: List[Dict[str, Any]] = []
    for n in range(25):
        stem = f"frame_fallback_{n:02d}"
        rev_path = tmp_path / "review" / "train" / f"{stem}.json"
        rev_path.write_text("{}", encoding="utf-8")
        _dummy_png(tmp_path / "images" / "train" / f"{stem}.png")
        (tmp_path / "labels" / "train" / f"{stem}.txt").write_text(
            "0 0.5 0.5 0.2 0.2\n", encoding="utf-8"
        )
        items.append(
            {
                "stem": stem,
                "split": "train",
                "weak": [
                    {"confidence": 0.30, "class_id": 0, "bbox": [10, 10, 30, 30]}
                ],
                "trusted": [],
                "done": None,
                "path": str(rev_path),
            }
        )
    dlg._review_items = items
    dlg.set_queue_filter_mode("all")

    dlg._bulk_prune_low_confidence_boxes()

    # External prune sees only "{}" review JSONs -> 0 pruned -> fallback runs.
    # With the old `total <= 20` cap, 0 of the 25 pruned boxes were taught.
    assert len(taught) == 25
    assert all(t["box"]["confidence"] == 0.30 for t in taught)
    assert all(t.get("positive") is False for t in taught)
    assert all(t.get("source") == "bulk_box_prune" for t in taught)
