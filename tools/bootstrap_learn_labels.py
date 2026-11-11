# -*- coding: utf-8 -*-
"""Fill empty learn_* YOLO labels from game_learning JSON (real detections only).

Omni captures used to land as empty train .txt (conf floor + obstacles skip).
This pass writes weak learn_auto boxes so Adım 2 can cull instead of hand-labeling.

K1 (tarama): layout/template boxes (layout_boxes_for_learn_category)
never enter commit_split_labels — a template box is not evidence, even
when the game_learning JSON has no real detection. A frame without a real
detection is skipped (empty label stays empty) instead of being filled
with fixed template coordinates. Human-approved template boxes still pass
through the normal review path (label_trust K1 gate).

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
    # F-1262: an unknown category is NOT a category — return None instead
    # of silently accepting the unknown part (callers skip + warn).
    return None


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


def _find_learning_json(
    learning_root: Path,
    stem: str,
    category: str,
    index: Optional[Dict[str, Path]] = None,
) -> Optional[Path]:
    if index is not None:
        # F-1259: the caller-built index already covers every subdir in one
        # pass — a per-miss folder rescan here would defeat it (one full
        # iterdir scan per image whose JSON is absent). The index is
        # authoritative: a miss stays a miss. The direct category-dir
        # check below must not run either: it would return a file the
        # index deliberately does not contain.
        return index.get(stem)
    cat_dir = learning_root / category
    if cat_dir.is_dir():
        candidate = cat_dir / f"{stem}.json"
        if candidate.is_file():
            return candidate
    # Fallback (index-less callers only): search one level under game_learning
    if learning_root.is_dir():
        for sub in learning_root.iterdir():
            if not sub.is_dir():
                continue
            candidate = sub / f"{stem}.json"
            if candidate.is_file():
                return candidate
    return None


def _build_learning_index(learning_root: Path) -> Dict[str, Path]:
    """One-pass index of game_learning/**/*.json by stem (F-1259)."""
    index: Dict[str, Path] = {}
    if not learning_root.is_dir():
        return index
    try:
        subdirs = [p for p in learning_root.iterdir() if p.is_dir()]
    except OSError:
        return index
    for sub in subdirs:
        try:
            for jf in sub.glob("*.json"):
                index.setdefault(jf.stem, jf)
        except OSError:
            continue
    return index


def _layout_annotations(
    category: str,
    layout_boxes: Sequence[Dict[str, Any]],
    *,
    conf: float,
) -> List[Dict[str, Any]]:
    """Layout -> annotation transform (kept for tests / manual inspection).

    K1: this transform no longer feeds commit_split_labels — main() skips
    frames without a real detection instead of filling them with template
    boxes. Kept because test_learn_auto_labels.py pins the transform.
    """
    anns: List[Dict[str, Any]] = []
    for item in layout_boxes:
        box = item.get("box", (0.5, 0.5, 0.2, 0.2))
        # F-1202: a malformed box (e.g. 5 elements) must not kill the run.
        try:
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"expected 4-element box, got {box!r}")
            cx, cy, bw, bh = (float(v) for v in box)
            class_id = int(item.get("class_id", 10))
        except (TypeError, ValueError) as exc:
            print(f"WARN: skipping malformed layout box {item!r}: {exc}", file=sys.stderr)
            continue
        anns.append(
            {
                "class_id": class_id,
                "cx": cx,
                "cy": cy,
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


def _review_has_boxes(path: Path) -> bool:
    """Review JSON'da kutu var mı? (P1-17 / K-09 idempotency.)

    Dosya yoksa veya kutusuz (trusted+weak boş) ise False — bootstrap
    doldurabilir. Kutu varsa True — yeniden etiketleme + commit
    çalıştırılmaz, kare atlanır. Bozuk/okunamaz JSON da True sayılır
    (fail-closed: bilinmeyen içerik üzerine yazılmaz).
    """
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"WARN: unreadable review {path.name}; skipping (fail-closed): {exc}",
            file=sys.stderr,
        )
        return True
    if not isinstance(payload, dict):
        return True
    for key in ("trusted", "weak"):
        boxes = payload.get(key)
        if isinstance(boxes, list) and boxes:
            return True
    return False


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
    dry_run_count = 0  # F-1261: dry-run counts separately from "rewritten".
    by_cat: Dict[str, int] = {}
    learning_index = _build_learning_index(learning_root)

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
            if args.limit and done + dry_run_count >= int(args.limit):
                break
            stem = img.stem
            category = _parse_category(stem)
            if not category:
                # F-1262: unknown learn_* category — warn instead of
                # silently skipping with an invented category.
                print(f"WARN: unknown learn category in {stem!r}; skipped", file=sys.stderr)
                skipped += 1
                continue
            if cat_filter and category not in cat_filter:
                skipped += 1
                continue

            label_path = labels_dir / f"{stem}.txt"
            if not args.force and _txt_nonempty(label_path):
                skipped += 1
                continue

            # P1-17 / K-09: idempotent — mevcut review'da kutu varsa
            # atla (el etiketli veya daha önce bootstrap'lanmış kareye
            # yeniden commit çalıştırılmaz; el etiketli TXT, review
            # JSON yokken commit'te sıfırlanıyordu — C011). Boş review
            # (kutusuz) doldurulabilir; --force iki kontrolü de aşar.
            review_path = root / "review" / split / f"{stem}.json"
            if not args.force and _review_has_boxes(review_path):
                skipped += 1
                continue

            layout: List[Dict[str, Any]] = []
            meta_src = "game_learning_json"
            learning_json = _find_learning_json(learning_root, stem, category, learning_index)
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
                    # F-1260: poison suspicion drops only the poisoned
                    # boxes, not the whole JSON — keep the good boxes.
                    poison_cids = {
                        "obstacles": {10},
                        "paths": {9},
                        "skills": {7},
                        "vitality": {13},
                    }.get(category, set())
                    if poison_cids:
                        kept = [
                            b
                            for b in from_json
                            if int(b["class_id"]) not in poison_cids
                        ]
                        dropped = len(from_json) - len(kept)
                        if dropped:
                            print(
                                f"WARN: dropped {dropped} poison-suspect box(es) "
                                f"(class {sorted(poison_cids)}) from {learning_json.name}",
                                file=sys.stderr,
                            )
                        from_json = kept
                    if from_json:
                        layout = from_json
                        meta_src = "game_learning_json"

            # K1 (tarama): no real detection -> no commit. Template boxes
            # (layout_boxes_for_learn_category) are not evidence and must
            # not fill an empty label: the old fallback wrote fixed
            # coordinates into train via commit_split_labels.
            if not layout:
                print(
                    f"SKIP: no real detection for {stem!r} — template boxes "
                    "no longer fill empty labels (K1)",
                    file=sys.stderr,
                )
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
                dry_run_count += 1
                by_cat[category] = by_cat.get(category, 0) + 1
                continue

            try:
                info = commit_split_labels(
                    dataset_dir=root,
                    split=split,
                    stem=stem,
                    label_path=label_path,
                    annotations=annotations,
                )
            except Exception as exc:
                # F-1203: the review JSON is deleted only AFTER a
                # successful commit — a failed commit must not lose data.
                print(f"ERROR: commit failed for {stem!r}: {exc}; review kept", file=sys.stderr)
                skipped += 1
                continue
            # P1-17 / K-09: commit sonrası review.unlink() KALDIRILDI.
            # Commit'in az önce yazdığı review JSON'u silmek taze
            # review'u yok ediyordu (O002): kare bir sonraki taramada
            # TXT-only sayılıyor ama TXT zaten commit'te boşaltılmış
            # olabiliyordu. Review diske kalır — ikinci çalışmada
            # yukarıdaki _review_has_boxes atlaması idempotency'yi
            # sağlar.
            done += 1
            by_cat[category] = by_cat.get(category, 0) + 1
            if done <= 5 or done % 200 == 0:
                print(
                    f"... {done} {stem} cat={category} src={meta_src} "
                    f"train={info.get('train_box_count')} weak={info.get('weak_box_count')}"
                )

        if args.limit and done + dry_run_count >= int(args.limit):
            break

    print(
        json.dumps(
            {
                # F-1261: "rewritten" counts only work actually written;
                # dry-run work is reported separately as "would_rewrite".
                "rewritten": done,
                "would_rewrite": dry_run_count,
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
