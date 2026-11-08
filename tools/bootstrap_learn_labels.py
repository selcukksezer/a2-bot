# -*- coding: utf-8 -*-
"""Fill empty learn_* YOLO labels from omni layout / game_learning JSON.

Omni captures used to land as empty train .txt (conf floor + obstacles skip).
This pass writes weak learn_auto boxes so Adım 2 can cull instead of hand-labeling.

Usage:
  python tools/bootstrap_learn_labels.py
  python tools/bootstrap_learn_labels.py --limit 100
  python tools/bootstrap_learn_labels.py --force
  python tools/bootstrap_learn_labels.py --dataset F:/a2/dataset
  python tools/bootstrap_learn_labels.py --categories monsters,skills,vitality
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_KNOWN_CATS = ("monsters", "skills", "paths", "obstacles", "vitality", "general", "mix")


def _parse_category(stem: str) -> Optional[str]:
    s = str(stem or "").strip().lower()
    if not s.startswith("learn_"):
        return None
    rest = s[len("learn_") :]
    for cat in _KNOWN_CATS:
        if rest.startswith(cat + "_") or rest == cat:
            return cat
    # learn_<category>_<ts>_<rnd>
    part = rest.split("_", 1)[0]
    return part or None


def _boxes_from_learning_json(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    out: List[Dict[str, Any]] = []
    raw_boxes = payload.get("boxes")
    if isinstance(raw_boxes, list) and raw_boxes:
        for item in raw_boxes:
            if not isinstance(item, dict):
                continue
            box = item.get("box")
            if not (isinstance(box, (list, tuple)) and len(box) >= 4):
                continue
            try:
                out.append(
                    {
                        "class_id": int(item.get("class_id", payload.get("class_id", 10))),
                        "box": (
                            float(box[0]),
                            float(box[1]),
                            float(box[2]),
                            float(box[3]),
                        ),
                    }
                )
            except (TypeError, ValueError):
                continue
        if out:
            return out
    box = payload.get("box_norm")
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        try:
            return [
                {
                    "class_id": int(payload.get("class_id", 10)),
                    "box": (
                        float(box[0]),
                        float(box[1]),
                        float(box[2]),
                        float(box[3]),
                    ),
                }
            ]
        except (TypeError, ValueError):
            return []
    return []


def _find_learning_json(learning_root: Path, stem: str, category: str) -> Optional[Path]:
    cat_dir = learning_root / category
    if cat_dir.is_dir():
        candidate = cat_dir / f"{stem}.json"
        if candidate.is_file():
            return candidate
    # Fallback: search one level under game_learning
    if learning_root.is_dir():
        for sub in learning_root.iterdir():
            if not sub.is_dir():
                continue
            candidate = sub / f"{stem}.json"
            if candidate.is_file():
                return candidate
    return None


def _layout_annotations(
    category: str,
    layout_boxes: Sequence[Dict[str, Any]],
    *,
    conf: float,
) -> List[Dict[str, Any]]:
    anns: List[Dict[str, Any]] = []
    for item in layout_boxes:
        box = item.get("box", (0.5, 0.5, 0.2, 0.2))
        cx, cy, bw, bh = box
        anns.append(
            {
                "class_id": int(item.get("class_id", 10)),
                "cx": float(cx),
                "cy": float(cy),
                "w": float(bw),
                "h": float(bh),
                "confidence": float(conf),
                "source": f"omni_{category}",
                "trusted": False,
                "weak": True,
                "omni_synthetic": True,
                "learn_auto": True,
                "bootstrap": True,
            }
        )
    return anns


def _txt_nonempty(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return False
    return bool(text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bootstrap empty learn_* labels with weak layout boxes"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset",
        help="Path to dataset/ root",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max frames to rewrite (0 = all)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite even when labels/*.txt already has lines",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="",
        help="Comma list filter: monsters,skills,paths,obstacles,vitality",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count only; do not write",
    )
    args = parser.parse_args()

    from core.omni_game_learner import layout_boxes_for_learn_category
    from dataset_factory.label_trust import (
        LEARN_AUTO_CONF,
        commit_split_labels,
        filter_hud_false_positives,
    )

    root = Path(args.dataset)
    learning_root = root / "game_learning"
    cat_filter = {
        c.strip().lower()
        for c in str(args.categories or "").split(",")
        if c.strip()
    }

    done = 0
    skipped = 0
    empty_skipped = 0
    by_cat: Dict[str, int] = {}

    for split in ("train", "val"):
        images_dir = root / "images" / split
        labels_dir = root / "labels" / split
        if not images_dir.is_dir():
            continue
        labels_dir.mkdir(parents=True, exist_ok=True)
        try:
            images = sorted(
                p
                for p in images_dir.iterdir()
                if p.is_file()
                and p.suffix.lower() in _IMAGE_EXTS
                and p.stem.lower().startswith("learn_")
            )
        except OSError:
            continue

        for img in images:
            if args.limit and done >= int(args.limit):
                break
            stem = img.stem
            category = _parse_category(stem)
            if not category:
                skipped += 1
                continue
            if cat_filter and category not in cat_filter:
                skipped += 1
                continue

            label_path = labels_dir / f"{stem}.txt"
            if not args.force and _txt_nonempty(label_path):
                skipped += 1
                continue

            layout: List[Dict[str, Any]] = []
            meta_src = "layout_default"
            learning_json = _find_learning_json(learning_root, stem, category)
            if learning_json is not None:
                from_json = _boxes_from_learning_json(learning_json)
                # Prefer multi-box / cognition-aware JSON; skip ancient poison
                # center blobs (class 10 obstacle, minimap path, etc.) by
                # falling back to current layout when class looks wrong.
                if from_json:
                    # Old captures put target HP bar as class 10 at top — remap.
                    if category == "monsters":
                        fixed: List[Dict[str, Any]] = []
                        for b in from_json:
                            nb = dict(b)
                            try:
                                cy = float(nb["box"][1])
                                cid = int(nb["class_id"])
                            except (TypeError, ValueError, KeyError, IndexError):
                                fixed.append(nb)
                                continue
                            if cy < 0.22 and cid == 10:
                                nb["class_id"] = 20
                            fixed.append(nb)
                        from_json = fixed
                    cids = {int(b["class_id"]) for b in from_json}
                    poison = False
                    if category == "obstacles" and cids & {10}:
                        poison = True
                    if category == "paths" and cids & {9}:
                        poison = True
                    if category == "skills" and cids <= {7}:
                        poison = True
                    if category == "vitality" and cids <= {13}:
                        poison = True
                    if not poison:
                        layout = from_json
                        meta_src = "game_learning_json"

            if not layout:
                layout = layout_boxes_for_learn_category(category)
                meta_src = "layout_default"

            if not layout:
                empty_skipped += 1
                skipped += 1
                continue

            annotations = _layout_annotations(
                category, layout, conf=float(LEARN_AUTO_CONF)
            )
            annotations = filter_hud_false_positives(
                annotations, stem=stem, remap_sprout=False
            )
            if category == "paths":
                annotations = [
                    a
                    for a in annotations
                    if float(a.get("cy", 0.0)) >= 0.55
                    and float(a.get("cx", 0.5)) < 0.82
                ]
            if not annotations:
                empty_skipped += 1
                skipped += 1
                continue

            if args.dry_run:
                done += 1
                by_cat[category] = by_cat.get(category, 0) + 1
                continue

            # Drop prior empty/poisoned review so merge starts clean for this stem.
            review = root / "review" / split / f"{stem}.json"
            if review.is_file():
                try:
                    review.unlink()
                except OSError:
                    pass

            info = commit_split_labels(
                dataset_dir=root,
                split=split,
                stem=stem,
                label_path=label_path,
                annotations=annotations,
            )
            done += 1
            by_cat[category] = by_cat.get(category, 0) + 1
            if done <= 5 or done % 200 == 0:
                print(
                    f"... {done} {stem} cat={category} src={meta_src} "
                    f"train={info.get('train_box_count')} weak={info.get('weak_box_count')}"
                )

        if args.limit and done >= int(args.limit):
            break

    print(
        json.dumps(
            {
                "rewritten": done,
                "skipped": skipped,
                "empty_layout_skipped": empty_skipped,
                "by_category": by_cat,
                "force": bool(args.force),
                "dry_run": bool(args.dry_run),
                "conf": float(LEARN_AUTO_CONF),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
