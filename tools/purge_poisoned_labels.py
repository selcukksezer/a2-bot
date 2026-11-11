# -*- coding: utf-8 -*-
"""Bulk-clean omni synthetic + HUD/weak poison from dataset/review + labels.

Does not touch hand / F/E trusted seeds. Images are kept.

Usage:
  python tools/purge_poisoned_labels.py
  python tools/purge_poisoned_labels.py --keep-weak
  python tools/purge_poisoned_labels.py --dataset F:/a2/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Purge poisoned YOLO auto labels")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset",
        help="Path to dataset/ root",
    )
    parser.add_argument(
        "--keep-weak",
        action="store_true",
        help="Demote omni trusted → weak (below floor) instead of dropping",
    )
    args = parser.parse_args(argv)
    if not args.dataset.is_dir():
        print(f"ERROR: dataset not found: {args.dataset}", file=sys.stderr)
        return 2
    from dataset_factory.label_trust import demote_omni_synthetic_labels

    stats = demote_omni_synthetic_labels(
        args.dataset,
        keep_as_weak=bool(args.keep_weak),
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    scanned = int(stats.get("scanned", 0) or 0)
    if scanned == 0:
        print(
            f"WARNING: no review files scanned in {args.dataset} "
            "— dataset empty or missing review/ dirs; nothing purged",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
