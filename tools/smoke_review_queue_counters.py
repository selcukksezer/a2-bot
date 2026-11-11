# -*- coding: utf-8 -*-
"""Smoke: review queue pending / frame_conf / reject-below stay in sync.

  python tools/smoke_review_queue_counters.py
"""

from __future__ import annotations

import os
import sys
import time
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

    t0 = time.perf_counter()
    # NOTE: plain asserts are skipped under `python -O`, so every check here
    # is explicit and collected — a failed check returns exit 1.
    failures: list[str] = []
    n_checks = 0

    def check(condition: bool, label: str, detail: Any = None) -> None:
        nonlocal n_checks
        n_checks += 1
        if not condition:
            failures.append(f"{label}: {detail!r}" if detail is not None else label)

    frame_conf = YOLOFactoryDialog._frame_conf
    items: List[Dict[str, Any]] = [
        _item(0.50, "a"),
        _item(0.80, "b"),
        _item(0.95, "c"),
    ]
    th = 0.77
    confs = [frame_conf(it) for it in items]
    check(confs == [0.50, 0.80, 0.95], "frame_conf values", confs)

    targets = [
        i
        for i, it in enumerate(items)
        if not it.get("done") and frame_conf(it) < th
    ]
    check(targets == [0], "reject-below targets (only 0.50)", targets)

    # Reject-below simülasyonu
    for i in sorted(targets, reverse=True):
        items.pop(i)
    pending = sum(1 for it in items if not it.get("done"))
    check(pending == 2, "pending after reject-below", pending)
    check(len(items) == 2, "remaining items", len(items))
    check(pending == len(items), "pending == len(items)", (pending, len(items)))

    # Eksik conf → 0.0 → eşik altı
    missing_conf = frame_conf({"stem": "x", "weak": [{"class_id": 1}], "done": None})
    check(missing_conf == 0.0, "missing conf -> 0.0", missing_conf)
    # NOTE: the original smoke expected 0.9 here, but _frame_conf's own
    # contract is "0 kutu -> 0.0" (docstring) and it never reads a top-level
    # "conf" key — item["conf"] is its *output* (callers assign it). An item
    # with no boxes is 0.0 even if a stale "conf" key is present. The old
    # assert failed at base b183551 for this reason; the expectation, not
    # _frame_conf (D29's file), was wrong.
    no_box_conf = frame_conf({"stem": "y", "weak": [], "conf": 0.9})
    check(no_box_conf == 0.0, "no boxes -> 0.0 despite top-level conf", no_box_conf)

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(f"OK review_queue_counters ({n_checks} checks, {elapsed_ms:.1f} ms)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
