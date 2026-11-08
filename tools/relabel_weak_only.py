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
        commit_split_labels,
        is_omni_synthetic_source,
    )
    from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

    labeler = YOLOAutoLabeler()
    splits = ("train", "val") if args.split == "both" else (args.split,)
    done = 0
    skipped = 0
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
            except Exception:
                continue
            if not isinstance(payload, dict):
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
                fresh = labeler.auto_label_frame(img)  # type: ignore[attr-defined]
            except Exception:
                try:
                    fresh = labeler.label_frame(img)  # type: ignore[attr-defined]
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
            # Clear prior weak by rewriting from empty weak state: delete review then commit
            try:
                path.unlink(missing_ok=True)
            except TypeError:
                if path.is_file():
                    path.unlink()
            # Also clear legacy txt so _load_state does not resurrect poison
            lbl = labels_dir / f"{stem}.txt"
            if keep_trusted:
                # commit will rewrite; start empty so only new merge applies
                pass
            else:
                try:
                    if lbl.is_file():
                        lbl.write_text("", encoding="utf-8")
                except Exception:
                    pass
            commit_split_labels(
                dataset_dir=args.dataset,
                split=split,
                stem=stem,
                label_path=lbl,
                annotations=observations,
                floor=TRAIN_CONF_FLOOR,
            )
            done += 1
            if done % 25 == 0:
                print(f"... {done} relabeled")
    print(json.dumps({"relabeled": done, "skipped": skipped}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
