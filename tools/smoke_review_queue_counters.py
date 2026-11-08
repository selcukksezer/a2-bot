# -*- coding: utf-8 -*-
"""Smoke: review queue pending / frame_conf / reject-below stay in sync.

  python tools/smoke_review_queue_counters.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _item(conf: float, stem: str) -> Dict[str, Any]:
    return {
        "stem": stem,
        "weak": [{"class_id": 3, "confidence": conf, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}],
        "trusted": [],
        "done": None,
        "n_weak": 1,
    }


def main() -> int:
    from ui.yolo_factory_dialog import YOLOFactoryDialog

    frame_conf = YOLOFactoryDialog._frame_conf
    items: List[Dict[str, Any]] = [
        _item(0.50, "a"),
        _item(0.80, "b"),
        _item(0.95, "c"),
    ]
    th = 0.77
    confs = [frame_conf(it) for it in items]
    assert confs == [0.50, 0.80, 0.95], confs

    targets = [
        i
        for i, it in enumerate(items)
        if not it.get("done") and frame_conf(it) < th
    ]
    assert targets == [0], targets  # yalnız 0.50

    # Reject-below simülasyonu
    for i in sorted(targets, reverse=True):
        items.pop(i)
    pending = sum(1 for it in items if not it.get("done"))
    assert pending == 2
    assert len(items) == 2
    assert pending == len(items)

    # Eksik conf → 0.0 → eşik altı
    assert frame_conf({"stem": "x", "weak": [{"class_id": 1}], "done": None}) == 0.0
    assert frame_conf({"stem": "y", "weak": [], "conf": 0.9}) == 0.9

    print("OK review_queue_counters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
