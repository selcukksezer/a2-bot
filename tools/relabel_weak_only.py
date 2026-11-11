# -*- coding: utf-8 -*-
"""Re-run YOLOAutoLabeler on frames that only have weak/auto boxes.

Trusted / hand / F/E boxes are preserved. Existing weak is replaced by a fresh
auto-label pass (+ HUD filter). Images are never deleted.

Usage:
  python tools/relabel_weak_only.py
  python tools/relabel_weak_only.py --limit 50
  python tools/relabel_weak_only.py --dataset F:/a2/dataset
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from typing import Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _imread(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _is_seed(ann: dict) -> bool:
    from dataset_factory.label_trust import _is_hand_or_gather_seed

    return _is_hand_or_gather_seed(ann) or ann.get("hand_label") is True


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-label weak-only review frames")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset")
    parser.add_argument("--limit", type=int, default=0, help="Max frames (0=all)")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    args = parser.parse_args()

    from dataset_factory.label_trust import (
        TRAIN_CONF_FLOOR,
        _rewrite_label_txt,
        commit_split_labels,
        is_omni_synthetic_source,
    )

    # YOLOAutoLabeler is built lazily on first use, not at startup: its
    # constructor builds VisionEngine + scanners + exemplar memory (~1 s
    # plus ~3.6 s module import), which is wasted work — and a hard failure
    # — on runs that have nothing to relabel (no review dirs, or only
    # trusted_seed.json). Import + construction happen on first frame.
    labeler = None

    def _get_labeler():
        nonlocal labeler
        if labeler is None:
            from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

            labeler = YOLOAutoLabeler()
        return labeler

    splits = ("train", "val") if args.split == "both" else (args.split,)
    done = 0
    skipped = 0
    zero_box = 0
    for split in splits:
        review_dir = args.dataset / "review" / split
        images_dir = args.dataset / "images" / split
        labels_dir = args.dataset / "labels" / split
        if not review_dir.is_dir():
            continue
        for path in sorted(review_dir.glob("*.json")):
            if path.name == "trusted_seed.json":
                continue
            if args.limit and done >= int(args.limit):
                break
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                # Corrupt review JSON must not vanish silently: count it
                # as skipped so the summary reflects the lost frame.
                skipped += 1
                print(f"skip {path.name}: unreadable review JSON: {exc}")
                continue
            if not isinstance(payload, dict):
                skipped += 1
                print(f"skip {path.name}: review JSON is not an object")
                continue
            stem = str(payload.get("stem") or path.stem)
            trusted = [b for b in (payload.get("trusted") or []) if isinstance(b, dict)]
            # Keep only real seeds in trusted; drop residual omni if any
            keep_trusted = [
                b
                for b in trusted
                if _is_seed(b)
                and not is_omni_synthetic_source(str(b.get("source", "")))
            ]
            img = None
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                cand = images_dir / f"{stem}{ext}"
                if cand.is_file():
                    img = _imread(cand)
                    if img is not None:
                        break
            if img is None:
                skipped += 1
                continue
            try:
                fresh = _get_labeler().auto_label_frame(img)  # type: ignore[attr-defined]
            except Exception:
                try:
                    fresh = _get_labeler().label_frame(img)  # type: ignore[attr-defined]
                except Exception as exc:
                    print(f"skip {stem}: {exc}")
                    skipped += 1
                    continue
            if not isinstance(fresh, list):
                fresh = []
            # Merge: seed trusted first, then fresh as observations (weak unless seed promotes)
            observations = list(keep_trusted) + [
                {**dict(b), "trusted": False, "weak": True, "source": str(b.get("source", "relabel_weak"))}
                for b in fresh
                if isinstance(b, dict)
            ]
            # S-4702: clear the legacy txt BEFORE deleting the review JSON.
            # The old order unlinked the review first; if the txt clear then
            # failed, the skip path left the stale txt on disk while the
            # review JSON (trusted seeds included) was already gone, so
            # _load_state fell back to _legacy_weak_from_txt and resurrected
            # the stale weak/poison boxes. Clearing first keeps the review
            # JSON intact on the failure path, so nothing is lost.
            lbl = labels_dir / f"{stem}.txt"
            # commit_split_labels falls back to _legacy_weak_from_txt(label_path)
            # when the review JSON is gone, so the txt must already be empty
            # before the unlink below; commit rewrites the txt, so starting
            # empty is safe even when trusted seeds are kept.
            # P3-26: the clear goes through `_rewrite_label_txt`
            # (atomic, .bak before overwrite) instead of a hand-rolled
            # `write_text("")` — same empty result, single writer.
            try:
                if lbl.is_file():
                    _rewrite_label_txt(lbl, [], allow_empty=True)
            except Exception as exc:
                # S-4905: the old silent `except: pass` let commit_split_labels
                # run with the stale txt still on disk, and _load_state falls
                # back to _legacy_weak_from_txt when the review JSON is gone —
                # resurrecting the old weak/poison boxes. Skip this frame
                # instead so the stale txt is never committed over.
                print(f"skip {stem}: could not clear stale label {lbl}: {exc}")
                skipped += 1
                continue
            # Clear prior weak by rewriting from empty weak state: delete review then commit
            try:
                path.unlink(missing_ok=True)
            except TypeError:
                if path.is_file():
                    path.unlink()
            info = commit_split_labels(
                dataset_dir=args.dataset,
                split=split,
                stem=stem,
                label_path=lbl,
                annotations=observations,
                floor=TRAIN_CONF_FLOOR,
            )
            done += 1
            # P2-21 (K-07): commit'in yazdığı kutu sayısı okunur —
            # 0 kutu yazan kare ayrı sayılır.
            if int(info.get("train_box_count", 0) or 0) <= 0:
                zero_box += 1
            if done % 25 == 0:
                print(f"... {done} relabeled")
    print(json.dumps({"relabeled": done, "skipped": skipped, "zero_box": zero_box}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
