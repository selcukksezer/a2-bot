# -*- coding: utf-8 -*-
"""Master Dataset Trust Recalibration Script.

Scans all review JSONs and label TXT files across `dataset/review/{train,val}`
and `dataset/labels/{train,val}`.

Solves the critical user problem:
"sadece emin olunanlara tıkladığımda 2400den 2200 e düşüyor 2200 fotoğrafı emin
gösteriyor ama çoğu yanlış güven sınırı yanlış yüzde 85 üzeri"

Strict Recalibration & Purge Rules:
1. Protected Verification Immunity:
   Boxes with `hand_label = True`, `human_approved = True`, `approved = True`,
   or `confirmed = True` remain `trusted = True, weak = False, confidence >= 0.90`.
2. Context-Based STEM Gating (Zero Out-of-Context Clutter):
   - On non-`learn_paths_*` frames: strictly purge `15: path_ground` and `14: obstacle`.
   - On non-`learn_skills_*` / non-`learn_vitality_*` frames: strictly purge
     `13: skill_ui_prompt`, `16: player_hp_bar`, `17: player_mp_bar`, `19: skill_bar`.
   - On non-essence frames (gathering/combat): strictly purge `22..134: essence_*`
     (including `98: essence_yggdrasil_log`).
3. Heuristic Source Demotion & Confidence Capping:
   - Boxes from `UNVERIFIED_HEURISTIC_SOURCES` (such as `chromatic_salient_3d_resource`,
     `visual_ground_path_segmenter`, `hud_skill_slot`, etc.) are capped at `<= 0.68`
     and demoted to `weak = True, trusted = False`. Never allow unverified heuristics
     to be trusted or >= 0.85.
4. Material Conflict Detection:
   - When a known material stem (e.g. `boot_*_aria_*`) contains conflicting resource
     boxes (e.g. `ore_orichalcum` or `herb_targena`), confidence is capped at `<= 0.45`
     and marked `weak = True, trusted = False`.
5. Non-explicit Legacy Boxes from TXT:
   - Boxes originating from legacy `.txt` without explicit confidence get `0.58`
     confidence and `weak = True, trusted = False`.
6. Verified Template Seed & High Confidence Proof:
   - Genuine template seeds (F/E prompt, 3D crest, minimap sprout, monster scanner)
     with confidence >= emin_threshold (0.85) remain `trusted = True, weak = False`.
   - Sub-threshold template boxes (< 0.85) are kept as `weak = True, trusted = False`.
7. Atomic Synchronization & Manifest Rebuild:
   - Cleanly rewrites both `dataset/review/{train,val}/*.json` and `dataset/labels/{train,val}/*.txt`.
   - Rebuilds `dataset/manifest_stats.json` via `DatasetCurator.rebuild_manifest_stats()`.
   - Logs exact before/after metrics.

Usage:
  python tools/recalibrate_dataset_trust.py
  python tools/recalibrate_dataset_trust.py --dry-run
  python tools/recalibrate_dataset_trust.py --emin-threshold 0.85
  python tools/recalibrate_dataset_trust.py --dataset F:/a2/dataset
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset_factory.label_trust import (
    MINIMAP_HUD_XYXY,
    TRAIN_CONF_FLOOR,
    _is_hand_or_gather_seed,
    _rewrite_label_txt,
    _yolo_line,
    annotation_confidence,
    box_center_in_minimap_hud,
    filter_hud_false_positives,
    is_path_learning_stem,
    mark_model_stale,
    slot_iou,
    train_txt_boxes,
)

logger = logging.getLogger("recalibrate_dataset_trust")

UNVERIFIED_HEURISTIC_SOURCES: frozenset = frozenset({
    "chromatic_salient_3d_resource",
    "visual_ground_path_segmenter",
    "visual_obstacle_segmenter",
    "hud_skill_slot",
    "hud_player_hp_bar",
    "hud_player_mp_bar",
    "dataset_requeue",
    "legacy_auto",
    "center_anchor_fallback",
    "gather_material_hint",
    "salient_fallback",
    "chromatic_contour",
    "heuristic",
})

HEURISTIC_TOKENS: Tuple[str, ...] = (
    "salient",
    "segmenter",
    "fallback",
    "hint",
    "legacy",
    "requeue",
    "omni_",
)

# Known material tokens in filename stems -> target material class_id
KNOWN_MATERIAL_STEM_MAP: Dict[str, int] = {
    "aria": 0,       # herb_aria
    "mela": 1,       # herb_mela
    "targena": 2,    # herb_targena
    "orichalcum": 3, # ore_orichalcum
    "ori": 3,
    "crystal": 4,    # gem_crystal
    "tree": 5,       # wood_tree
}

WORLD_RESOURCE_CLASSES: frozenset = frozenset({0, 1, 2, 3, 4, 5})
PATH_OBSTACLE_CLASSES: frozenset = frozenset({14, 15})
HUD_SKILL_CLASSES: frozenset = frozenset({13, 16, 17, 19})


def is_skill_learning_stem(stem: str) -> bool:
    """True if stem represents a frame dedicated to skill / vitality / HUD learning."""
    s = str(stem or "").strip().lower()
    return (
        s.startswith("learn_skills_")
        or s.startswith("learn_vitality_")
        or "_skills_" in s
        or "_vitality_" in s
        or "exemplar_cls13_" in s
        or "exemplar_cls16_" in s
        or "exemplar_cls17_" in s
        or "exemplar_cls19_" in s
        or "exemplar_cls20_" in s
        or "exemplar_cls21_" in s
    )


def is_essence_learning_stem(stem: str) -> bool:
    """True if stem represents a dedicated essence learning frame."""
    s = str(stem or "").strip().lower()
    return (
        s.startswith("learn_essence_")
        or s.startswith("essence_")
        or "_essence_" in s
        or "learn_aethertapping_" in s
    )


def get_stem_expected_material(stem: str) -> Optional[int]:
    """Extract expected resource class_id from stem if it specifies a known material."""
    s = str(stem or "").lower()
    for token, cid in KNOWN_MATERIAL_STEM_MAP.items():
        # Match as whole word or surrounded by delimiters
        if f"_{token}_" in s or s.startswith(f"{token}_") or s.endswith(f"_{token}") or f"_{token}." in s:
            return cid
    return None


def is_conflicting_material(stem: str, class_id: int) -> bool:
    """True if stem specifies a material but box is an incompatible world resource."""
    expected = get_stem_expected_material(stem)
    if expected is None:
        return False
    if class_id in WORLD_RESOURCE_CLASSES and class_id != expected:
        return True
    return False


def is_unverified_heuristic_source(source: str) -> bool:
    """True if source is an unverified heuristic."""
    src = str(source or "").strip().lower()
    if not src:
        return True
    if src in UNVERIFIED_HEURISTIC_SOURCES:
        return True
    return any(token in src for token in HEURISTIC_TOKENS)


def is_genuine_verified_seed(box: Dict[str, Any]) -> bool:
    """True if box originates from a genuine verified template seed / detector."""
    if _is_hand_or_gather_seed(box):
        return True
    src = str(box.get("source", "")).strip().lower()
    verified_sources = (
        "vision_3d_crest",
        "material_scanner_f_gather_prompt",
        "material_scanner_floating_icon",
        "material_scanner_minimap_sprout",
        "hostile_red_hp_bar_monster",
        "3d_monster_scanner",
        "target_hp_bar_scanner",
        "3d_mesh_below_crest",
        "3d_mesh_under_floating_icon",
    )
    return any(v in src for v in verified_sources)


def calibrate_single_box(
    raw_box: Dict[str, Any],
    stem: str = "",
    emin_threshold: float = 0.85,
    purge_synthetic: bool = True,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Calibrate a single bounding box according to strict trust and purge rules.

    Returns:
        (calibrated_box, action_str)
        If action_str starts with "purged_", calibrated_box will be None.
        Otherwise action_str is one of "preserved_trusted", "demoted_to_weak", "kept_weak".
    """
    if not isinstance(raw_box, dict) or "cx" not in raw_box or "cy" not in raw_box:
        return None, "purged_invalid_format"

    try:
        cid = int(raw_box.get("class_id", -1))
        cx = float(raw_box["cx"])
        cy = float(raw_box["cy"])
        w = float(raw_box.get("w", 0.0))
        h = float(raw_box.get("h", 0.0))
    except (TypeError, ValueError):
        return None, "purged_invalid_geometry"

    # Geometric sanity check
    if w <= 0.0 or h <= 0.0 or not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None, "purged_out_of_bounds"

    box = dict(raw_box)
    box["class_id"] = cid
    box["cx"] = round(cx, 6)
    box["cy"] = round(cy, 6)
    box["w"] = round(w, 6)
    box["h"] = round(h, 6)

    was_trusted = bool(box.get("trusted") is True)
    conf = annotation_confidence(box)
    src = str(box.get("source", "")).lower()

    # Rule 1: Protected Immunity (Hand labels / human approved)
    is_protected = bool(
        box.get("hand_label") is True
        or box.get("human_approved") is True
        or box.get("approved") is True
        or box.get("confirmed") is True
    )

    if is_protected:
        box["trusted"] = True
        box["weak"] = False
        box["confidence"] = max(0.90, round(float(conf), 4) if conf > 0 else 0.90)
        return box, "preserved_trusted"

    # Rule 2: Illegal Minimap HUD False Positives
    if box_center_in_minimap_hud(cx, cy, MINIMAP_HUD_XYXY):
        # Only class 7 (F prompt), 8 (E icon), 9 (sprout), 11 (monster dot) allowed in minimap
        if cid not in {7, 8, 9, 11}:
            return None, "purged_illegal_minimap_hud_box"

    # Rule 3: Context-Based Stem Gating (Purge out-of-context clutter)
    if purge_synthetic:
        # Ground Path (15) and Obstacle (14) ONLY on path learning frames
        if cid in PATH_OBSTACLE_CLASSES:
            if not is_path_learning_stem(stem):
                return None, "purged_out_of_context_path_obstacle"

        # HUD Skills (13) & Bars (16, 17, 19) ONLY on skill/vitality learning frames
        if cid in HUD_SKILL_CLASSES:
            if not is_skill_learning_stem(stem):
                return None, "purged_out_of_context_hud_skill"

        # Essence classes (22..134) ONLY on dedicated essence frames
        if cid >= 22:
            if not is_essence_learning_stem(stem):
                return None, "purged_out_of_context_essence_on_gather_frame"

    # Rule 4: Material Conflict Detection on Known Stems
    if is_conflicting_material(stem, cid):
        box["confidence"] = min(round(float(conf), 4), 0.45)
        box["trusted"] = False
        box["weak"] = True
        box["demoted_reason"] = "stem_material_conflict"
        return box, "demoted_to_weak" if was_trusted else "kept_weak"

    # Rule 5: Non-explicit Legacy Boxes from TXT (Task 1: Ensure non-explicit legacy boxes from .txt get 0.58 confidence and weak = True)
    if box.get("legacy_auto") is True or not box.get("has_explicit_conf", True) or conf <= 1e-6 or "legacy" in src:
        box["confidence"] = 0.58
        box["trusted"] = False
        box["weak"] = True
        box["demoted_reason"] = "legacy_txt_unverified"
        return box, "demoted_to_weak" if was_trusted else "kept_weak"

    # Rule 6: Unverified Heuristic Source Demotion
    if is_unverified_heuristic_source(src):
        box["confidence"] = min(round(float(conf), 4), 0.68)
        box["trusted"] = False
        box["weak"] = True
        box["demoted_reason"] = "unverified_heuristic_source"
        return box, "demoted_to_weak" if was_trusted else "kept_weak"

    # Rule 7: Genuine Template Seed & High Confidence Proof
    if is_genuine_verified_seed(box):
        if conf >= float(emin_threshold):
            box["trusted"] = True
            box["weak"] = False
            box["confidence"] = round(float(conf), 4)
            return box, "preserved_trusted"
        else:
            box["trusted"] = False
            box["weak"] = True
            box["demoted_reason"] = "seed_below_emin_threshold"
            return box, "demoted_to_weak" if was_trusted else "kept_weak"

    # Rule 8: General Confidence Gate for Other Sources
    if conf >= float(emin_threshold):
        # Boxes >= emin_threshold with verified source remain trusted
        box["trusted"] = True
        box["weak"] = False
        box["confidence"] = round(float(conf), 4)
        return box, "preserved_trusted"
    else:
        box["trusted"] = False
        box["weak"] = True
        box["demoted_reason"] = "confidence_below_emin_threshold"
        return box, "demoted_to_weak" if was_trusted else "kept_weak"


def _dedupe_and_nms(
    boxes: Sequence[Dict[str, Any]],
    iou_thresh: float = 0.65,
) -> Tuple[List[Dict[str, Any]], int]:
    """Perform Non-Maximum Suppression (NMS) deduplication across boxes."""
    def _rank(b: Dict[str, Any]) -> Tuple[float, float]:
        is_t = 1.0 if (b.get("trusted") is True or b.get("hand_label") is True) else 0.0
        return (is_t, annotation_confidence(b))

    sorted_boxes = sorted(boxes, key=_rank, reverse=True)
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for b in sorted_boxes:
        suppress = False
        for k in kept:
            if int(b.get("class_id", -1)) == int(k.get("class_id", -1)):
                if slot_iou(b, k) > iou_thresh:
                    suppress = True
                    dropped += 1
                    break
        if not suppress:
            kept.append(b)
    return kept, dropped


def recalibrate_entire_dataset_trust(
    dataset_dir: Union[str, Path] = None,
    *,
    emin_threshold: float = 0.85,
    dry_run: bool = False,
    purge_synthetic: bool = True,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """Master recalibration runner across `dataset/review/{train,val}` and `dataset/labels/{train,val}`.

    Scans every review JSON and label TXT, strictly purges out-of-context synthetic boxes,
    demotes unverified heuristics to weak, enforces 0.58 confidence on legacy txt boxes,
    rewrites JSONs and TXTs cleanly, and rebuilds `dataset/manifest_stats.json`.
    """
    start_time = time.time()
    root = Path(dataset_dir) if dataset_dir is not None else ROOT / "dataset"
    review_dir = root / "review"
    labels_dir = root / "labels"

    emin_th = float(emin_threshold)
    total_images_processed = 0
    previous_trusted_images = 0
    new_trusted_images = 0
    previous_weak_images = 0
    new_weak_images = 0
    previous_trusted_boxes = 0
    new_trusted_boxes = 0
    previous_weak_boxes = 0
    new_weak_boxes = 0

    demoted_boxes = 0
    purged_false_boxes = 0
    preserved_verified_boxes = 0
    files_rewritten = 0
    labels_rewritten = 0

    purged_breakdown: Dict[str, int] = {}

    # Discover all review files across train and val splits
    all_targets: List[Tuple[str, Path, Path]] = []
    for split in ("train", "val"):
        s_review = review_dir / split
        s_labels = labels_dir / split
        if not s_review.is_dir():
            continue
        for json_path in sorted(s_review.glob("*.json")):
            if json_path.name == "trusted_seed.json":
                continue
            stem = json_path.stem
            txt_path = s_labels / f"{stem}.txt"
            all_targets.append((split, json_path, txt_path))

    total_files = len(all_targets)
    logger.info("Found %d review JSON targets across %s", total_files, root)

    for idx, (split, json_path, txt_path) in enumerate(all_targets, 1):
        if progress_callback:
            try:
                progress_callback(idx, total_files, json_path.stem)
            except Exception:
                pass

        total_images_processed += 1
        stem = json_path.stem

        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse JSON %s: %s (salvaging from label txt)", json_path, e)
            payload = {"stem": stem, "split": split, "floor": TRAIN_CONF_FLOOR, "trusted": [], "weak": []}

        if not isinstance(payload, dict):
            payload = {"stem": stem, "split": split, "floor": TRAIN_CONF_FLOOR, "trusted": [], "weak": []}

        trusted_in = [b for b in (payload.get("trusted") or []) if isinstance(b, dict)]
        weak_in = [b for b in (payload.get("weak") or []) if isinstance(b, dict)]

        # If review file has no boxes or was salvaged, load legacy boxes from txt
        if not trusted_in and not weak_in and txt_path.is_file():
            try:
                txt_content = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                if txt_content:
                    for line in txt_content.splitlines():
                        parts = line.split()
                        if len(parts) >= 5:
                            cid = int(float(parts[0]))
                            cx = float(parts[1])
                            cy = float(parts[2])
                            w = float(parts[3])
                            h = float(parts[4])
                            weak_in.append({
                                "class_id": cid,
                                "cx": cx,
                                "cy": cy,
                                "w": w,
                                "h": h,
                                "confidence": 0.58,
                                "trusted": False,
                                "weak": True,
                                "legacy_auto": True,
                                "has_explicit_conf": False,
                                "source": "legacy_txt",
                            })
            except Exception:
                pass

        # Record BEFORE metrics
        trusted_before_count = len(trusted_in)
        weak_before_count = len(weak_in)
        previous_trusted_boxes += trusted_before_count
        previous_weak_boxes += weak_before_count
        if trusted_before_count > 0:
            previous_trusted_images += 1
        if weak_before_count > 0:
            previous_weak_images += 1

        # Process all boxes through strict calibration and purge rules
        candidate_boxes: List[Dict[str, Any]] = []
        all_boxes_in = list(trusted_in) + list(weak_in)

        for b in all_boxes_in:
            calibrated, action = calibrate_single_box(
                b,
                stem=stem,
                emin_threshold=emin_th,
                purge_synthetic=purge_synthetic,
            )

            if calibrated is None:
                purged_false_boxes += 1
                purged_breakdown[action] = purged_breakdown.get(action, 0) + 1
            else:
                candidate_boxes.append(calibrated)
                if action == "preserved_trusted":
                    preserved_verified_boxes += 1
                elif action == "demoted_to_weak":
                    demoted_boxes += 1

        # NMS Deduplication
        deduped_boxes, nms_dropped = _dedupe_and_nms(candidate_boxes, iou_thresh=0.65)
        if nms_dropped > 0:
            purged_false_boxes += nms_dropped
            purged_breakdown["purged_nms_duplicate"] = (
                purged_breakdown.get("purged_nms_duplicate", 0) + nms_dropped
            )

        # Final segregation into trusted (Emin) and weak (Şüpheli)
        final_trusted = [b for b in deduped_boxes if b.get("trusted") is True]
        final_weak = [b for b in deduped_boxes if b.get("trusted") is not True]

        # Record AFTER metrics
        trusted_after_count = len(final_trusted)
        weak_after_count = len(final_weak)
        new_trusted_boxes += trusted_after_count
        new_weak_boxes += weak_after_count
        if trusted_after_count > 0:
            new_trusted_images += 1
        if weak_after_count > 0:
            new_weak_images += 1

        # Write modifications to disk unless dry-run
        if not dry_run:
            payload["trusted"] = final_trusted
            payload["weak"] = final_weak
            payload["stem"] = stem
            payload["split"] = split
            payload["floor"] = TRAIN_CONF_FLOOR
            payload["recalibrated"] = True
            payload["recalibrated_at"] = datetime.now(timezone.utc).isoformat()

            try:
                json_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                files_rewritten += 1
            except Exception as e:
                logger.error("Failed to write JSON %s: %s", json_path, e)

            # Rewrite corresponding label .txt cleanly
            emit = train_txt_boxes(
                final_trusted,
                final_weak,
                stem=stem,
                floor=TRAIN_CONF_FLOOR,
            )
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [_yolo_line(box) for box in emit if "cx" in box]
            txt_content = ("\n".join(lines) + "\n") if lines else ""
            try:
                txt_path.write_text(txt_content, encoding="utf-8")
                labels_rewritten += 1
            except Exception as e:
                logger.error("Failed to write label txt %s: %s", txt_path, e)

    # Rebuild manifest_stats.json at completion
    manifest_stats: Dict[str, Any] = {}
    if not dry_run:
        try:
            from dataset_factory.dataset_curator import DatasetCurator
            curator = DatasetCurator(dataset_dir=root)
            manifest_stats = curator.rebuild_manifest_stats()
            mark_model_stale(
                reason=(
                    f"Dataset güven skorları yeniden kalibre edildi "
                    f"({purged_false_boxes} kutu temizlendi, {demoted_boxes} kutu şüpheliye taşındı)"
                ),
                dataset_dir=root,
            )
        except Exception as e:
            logger.error("Failed to rebuild manifest_stats.json: %s", e)

    duration = round(time.time() - start_time, 2)

    report: Dict[str, Any] = {
        "ok": True,
        "dry_run": bool(dry_run),
        "dataset_dir": str(root),
        "emin_threshold": emin_th,
        "duration_sec": duration,
        "total_images_processed": total_images_processed,
        "metrics": {
            "previous_trusted_images": previous_trusted_images,
            "new_trusted_images": new_trusted_images,
            "previous_weak_images": previous_weak_images,
            "new_weak_images": new_weak_images,
            "previous_trusted_boxes": previous_trusted_boxes,
            "new_trusted_boxes": new_trusted_boxes,
            "previous_weak_boxes": previous_weak_boxes,
            "new_weak_boxes": new_weak_boxes,
            "demoted_boxes": demoted_boxes,
            "purged_false_boxes": purged_false_boxes,
            "preserved_verified_boxes": preserved_verified_boxes,
        },
        "purged_breakdown": purged_breakdown,
        "files_rewritten": files_rewritten,
        "labels_rewritten": labels_rewritten,
        "manifest_stats": manifest_stats,
    }

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalibrate dataset trust, purge out-of-context clutter, and demote unverified heuristics."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset",
        help="Path to dataset root folder (defaults to ROOT/dataset)",
    )
    parser.add_argument(
        "--emin-threshold",
        type=float,
        default=0.85,
        help="Confidence threshold for trusted (Emin) boxes (default 0.85)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate recalibration without writing changes to disk",
    )
    parser.add_argument(
        "--keep-synthetic",
        action="store_true",
        help="Do not purge out-of-context synthetic boxes (not recommended)",
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=None,
        help="Optional path to write JSON recalibration report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 72)
    print("      DATASET TRUST RECALIBRATION & MASTER PURGE RUNNER        ")
    print("=" * 72)
    print(f"Dataset Root:    {args.dataset}")
    print(f"Emin Threshold:  {args.emin_threshold}")
    print(f"Dry Run:         {args.dry_run}")
    print(f"Purge Synthetic: {not args.keep_synthetic}")
    print("-" * 72)

    def _progress(cur: int, tot: int, stem: str) -> None:
        if cur % 250 == 0 or cur == tot:
            pct = (cur / float(tot)) * 100.0 if tot > 0 else 100.0
            print(f"  [{cur}/{tot}] ({pct:.1f}%) Processing {stem}...")

    report = recalibrate_entire_dataset_trust(
        dataset_dir=args.dataset,
        emin_threshold=args.emin_threshold,
        dry_run=args.dry_run,
        purge_synthetic=not args.keep_synthetic,
        progress_callback=_progress,
    )

    m = report["metrics"]
    print("=" * 72)
    print("RECALIBRATION COMPLETE")
    print(f"Duration: {report['duration_sec']} seconds")
    print(f"Total Images Processed:        {report['total_images_processed']}")
    print("-" * 72)
    print(f"Previous 'Emin' (Trusted) Images: {m['previous_trusted_images']}")
    print(f"New 'Emin' (Trusted) Images:      {m['new_trusted_images']}")
    print(f"Previous 'Şüpheli' (Weak) Images: {m['previous_weak_images']}")
    print(f"New 'Şüpheli' (Weak) Images:      {m['new_weak_images']}")
    print("-" * 72)
    print(f"Previous Trusted Boxes:           {m['previous_trusted_boxes']}")
    print(f"New Trusted Boxes:                {m['new_trusted_boxes']}")
    print(f"Previous Weak Boxes:              {m['previous_weak_boxes']}")
    print(f"New Weak Boxes:                   {m['new_weak_boxes']}")
    print("-" * 72)
    print(f"Demoted Boxes (Trusted -> Weak):  {m['demoted_boxes']}")
    print(f"Purged False / Out-of-Context:    {m['purged_false_boxes']}")
    print(f"Preserved Verified Boxes:         {m['preserved_verified_boxes']}")
    print("-" * 72)
    print("Purge Breakdown:")
    for reason, count in sorted(report["purged_breakdown"].items()):
        print(f"  - {reason}: {count}")
    print("-" * 72)
    print(f"Files Rewritten (Review JSONs):   {report['files_rewritten']}")
    print(f"Labels Rewritten (YOLO TXTs):     {report['labels_rewritten']}")
    print("=" * 72)

    if args.export_report:
        args.export_report.parent.mkdir(parents=True, exist_ok=True)
        args.export_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Report exported to: {args.export_report}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
