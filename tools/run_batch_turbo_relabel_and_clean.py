# -*- coding: utf-8 -*-
"""
Disk Batch Turbo Relabel Runner & High-Precision AI Cleaner (~3,000 Images).

Executes end-to-end:
1. Discovers all images in dataset/images/train/ and dataset/images/val/.
2. Runs high-throughput multi-core batch relabeling via DatasetCurator.relabel_dataset_batch
   (using YOLOAutoLabeler with fast_mode=True).
3. Applies Katman 0-3 Smart AI Poison Purge (smart_ai_purge_poisoned_labels / clean_poison_labels_ai):
   - Katman 0: Immunity Shield for hand/seed/exemplar boxes.
   - Katman 1: Auto class repair via positive exemplars & negative exemplar suppression.
   - Katman 2: Visual flat terrain, sky, and HUD false-positive elimination.
   - Katman 3: Synthetic omni_* drop & NMS deduplication.
4. Applies box-level low-confidence pruning (prune_low_confidence_boxes_in_dataset)
   without deleting images.
5. Rebuilds dataset/manifest_stats.json via curator.rebuild_manifest_stats().
6. Prints detailed benchmark performance: images processed, boxes labeled, boxes auto-repaired,
   boxes pruned, elapsed time, and average fps.

Usage:
  python tools/run_batch_turbo_relabel_and_clean.py
  python tools/run_batch_turbo_relabel_and_clean.py --max-workers 12
  python tools/run_batch_turbo_relabel_and_clean.py --force-all
  python tools/run_batch_turbo_relabel_and_clean.py --prune-threshold 0.50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

# Ensure repository root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

# Core Engine Imports
import dataset_factory.label_trust as label_trust
from dataset_factory.dataset_curator import DatasetCurator
from dataset_factory.exemplar_memory import ExemplarMemory, extract_box_crop
from dataset_factory.yolo_auto_labeler import YOLOAutoLabeler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TurboRelabelRunner")


# ==============================================================================
# Box-Level Low-Confidence Pruner ("✂️ Düşük Güvenli Etiketleri Sil (Görseli Koru)")
# ==============================================================================
def prune_low_confidence_boxes_in_dataset(
    dataset_dir: Union[str, Path],
    threshold: float = 0.50,
    indices_or_stems: Optional[Iterable[str]] = None,
    record_negative_exemplars: bool = True,
) -> Dict[str, Any]:
    """
    Remove ONLY low-confidence boxes (< threshold) from review JSON and label TXT.
    The image and all remaining confident boxes (>= threshold) are 100% PRESERVED.
    Pruned low-confidence crops are optionally recorded into ExemplarMemory as
    negative exemplars so the model learns from the cleanup.
    """
    # If label_trust already provides this function, delegate directly
    if hasattr(label_trust, "prune_low_confidence_boxes_in_dataset") and (
        getattr(label_trust, "prune_low_confidence_boxes_in_dataset")
        != prune_low_confidence_boxes_in_dataset
    ):
        return getattr(label_trust, "prune_low_confidence_boxes_in_dataset")(
            dataset_dir=dataset_dir,
            threshold=threshold,
            indices_or_stems=indices_or_stems,
            record_negative_exemplars=record_negative_exemplars,
        )

    from dataset_factory.label_trust import (
        TRAIN_CONF_FLOOR,
        _is_hand_or_gather_seed,
        _rewrite_label_txt,
        annotation_confidence,
        train_txt_boxes,
    )

    root = Path(dataset_dir)
    stem_filter = set(indices_or_stems) if indices_or_stems is not None else None

    ex_memory: Optional[ExemplarMemory] = None
    if record_negative_exemplars:
        try:
            ex_memory = ExemplarMemory()
        except Exception:
            ex_memory = None

    scanned_frames = 0
    modified_frames = 0
    total_boxes_before = 0
    pruned_boxes = 0
    preserved_boxes = 0
    negatives_taught = 0

    for split in ("train", "val"):
        review_dir = root / "review" / split
        labels_dir = root / "labels" / split
        images_dir = root / "images" / split

        if not review_dir.is_dir():
            continue

        for rev_file in sorted(review_dir.glob("*.json")):
            stem = rev_file.stem
            if stem == "trusted_seed":
                continue
            if stem_filter is not None and stem not in stem_filter:
                continue

            scanned_frames += 1
            try:
                payload = json.loads(rev_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            if not isinstance(payload, dict):
                continue

            trusted_in = [b for b in payload.get("trusted", []) if isinstance(b, dict)]
            weak_in = [b for b in payload.get("weak", []) if isinstance(b, dict)]
            all_boxes = list(trusted_in) + list(weak_in)
            if not all_boxes:
                continue

            total_boxes_before += len(all_boxes)

            # Fast check: are there any prunable boxes below threshold that are not protected?
            prunable_exists = any(
                annotation_confidence(b) < float(threshold)
                and not (
                    b.get("hand_label") is True
                    or b.get("human_approved") is True
                    or b.get("approved") is True
                    or _is_hand_or_gather_seed(b)
                )
                for b in all_boxes
            )

            frame: Optional[np.ndarray] = None
            if prunable_exists and record_negative_exemplars and ex_memory is not None:
                for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
                    img_cand = images_dir / f"{stem}{ext}"
                    if img_cand.is_file():
                        try:
                            data = np.fromfile(str(img_cand), dtype=np.uint8)
                            if data.size > 0:
                                frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                        except Exception:
                            frame = None
                        break

            new_trusted: List[Dict[str, Any]] = []
            new_weak: List[Dict[str, Any]] = []
            frame_pruned_count = 0

            for b in all_boxes:
                b_copy = dict(b)
                conf = annotation_confidence(b_copy)
                cid = int(b_copy.get("class_id", 0))
                cname = str(b_copy.get("class_name", ""))

                is_protected = (
                    b_copy.get("hand_label") is True
                    or b_copy.get("human_approved") is True
                    or b_copy.get("approved") is True
                    or _is_hand_or_gather_seed(b_copy)
                )

                # Prune if strictly below threshold and not immune
                if conf < float(threshold) and not is_protected:
                    frame_pruned_count += 1
                    pruned_boxes += 1
                    if (
                        record_negative_exemplars
                        and ex_memory is not None
                        and frame is not None
                    ):
                        try:
                            crop = extract_box_crop(frame, b_copy)
                            if crop is not None and crop.size >= 16:
                                ex_memory.add_negative_exemplar(
                                    crop, cid, cname, source="low_confidence_prune"
                                )
                                negatives_taught += 1
                        except Exception:
                            pass
                    continue

                # Preserve box: if >= threshold, promote to trusted (🟢 Emin)
                if conf >= float(threshold) or is_protected:
                    b_copy["trusted"] = True
                    b_copy["weak"] = False
                    new_trusted.append(b_copy)
                else:
                    b_copy["trusted"] = False
                    b_copy["weak"] = True
                    new_weak.append(b_copy)
                preserved_boxes += 1

            if frame_pruned_count > 0:
                payload["trusted"] = new_trusted
                payload["weak"] = new_weak
                rev_file.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

                # Sync labels/*.txt
                lbl_file = labels_dir / f"{stem}.txt"
                labels_dir.mkdir(parents=True, exist_ok=True)
                emit = train_txt_boxes(
                    new_trusted, new_weak, stem=stem, floor=float(threshold)
                )
                _rewrite_label_txt(lbl_file, emit)
                modified_frames += 1

    return {
        "scanned_frames": scanned_frames,
        "modified_frames": modified_frames,
        "total_boxes_before": total_boxes_before,
        "pruned_boxes": pruned_boxes,
        "preserved_boxes": preserved_boxes,
        "negatives_taught": negatives_taught,
        "threshold": float(threshold),
    }


# Ensure label_trust has aliases if missing
if not hasattr(label_trust, "prune_low_confidence_boxes_in_dataset"):
    setattr(
        label_trust,
        "prune_low_confidence_boxes_in_dataset",
        prune_low_confidence_boxes_in_dataset,
    )

if not hasattr(label_trust, "smart_ai_purge_poisoned_labels"):
    setattr(
        label_trust,
        "smart_ai_purge_poisoned_labels",
        getattr(label_trust, "clean_poison_labels_ai", None),
    )


# ==============================================================================
# Main Runner Pipeline
# ==============================================================================
def run_batch_turbo_relabel_and_clean(
    dataset_dir: Union[str, Path] = ROOT / "dataset",
    *,
    max_workers: Optional[int] = None,
    conf_floor: float = 0.45,
    emin_threshold: float = 0.85,
    prune_threshold: float = 0.50,
    force_all: bool = False,
    incremental: bool = True,
    skip_relabel: bool = False,
    limit: int = 0,
) -> Dict[str, Any]:
    """
    Executes the full pipeline across dataset/images/{train,val}.
    """
    dataset_path = Path(dataset_dir).resolve()
    images_train = dataset_path / "images" / "train"
    images_val = dataset_path / "images" / "val"

    if max_workers is None:
        max_workers = min(16, max(2, (os.cpu_count() or 4) * 2))

    logger.info("Initializing DatasetCurator on dataset: %s", dataset_path)
    curator = DatasetCurator(dataset_dir=dataset_path)

    # 1. Discover all images
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    target_images: List[Path] = []
    for split_dir in (images_train, images_val):
        if split_dir.is_dir():
            for f in sorted(split_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in valid_exts:
                    target_images.append(f)

    if limit > 0:
        target_images = target_images[:limit]

    total_images = len(target_images)
    logger.info("Discovered %d total images across train/val splits.", total_images)

    t_start_total = time.time()
    relabel_stats: Dict[str, Any] = {
        "processed": 0,
        "labeled": 0,
        "skipped": 0,
        "total_boxes": 0,
        "fps": 0.0,
        "elapsed_seconds": 0.0,
    }

    # 2. Batch Turbo Relabeling Pass
    if not skip_relabel and total_images > 0:
        logger.info(
            "Starting multi-core Turbo Relabel pass with %d workers (force_all=%s, incremental=%s)...",
            max_workers,
            force_all,
            incremental,
        )

        last_log_time = time.time()
        last_log_count = 0

        def _on_progress(info: Dict[str, Any]) -> None:
            nonlocal last_log_time, last_log_count
            now = time.time()
            proc = info.get("processed", 0)
            tot = info.get("total", total_images)
            if now - last_log_time >= 5.0 or proc == tot or proc - last_log_count >= 100:
                pct = (proc / max(1, tot)) * 100.0
                fps = info.get("fps", 0.0)
                eta = info.get("eta_s", 0.0)
                stem = info.get("stem", "")
                boxes = info.get("boxes", 0)
                logger.info(
                    "[%d/%d (%.1f%%)] %.1f fps | ETA: %.1fs | Last: %s (%d boxes)",
                    proc,
                    tot,
                    pct,
                    fps,
                    eta,
                    stem,
                    boxes,
                )
                last_log_time = now
                last_log_count = proc

        t_relabel_start = time.time()
        relabel_stats = curator.relabel_dataset_batch(
            image_paths=target_images,
            max_workers=max_workers,
            incremental=incremental,
            force_all=force_all,
            conf_floor=conf_floor,
            emin_threshold=emin_threshold,
            progress_callback=_on_progress,
        )
        t_relabel_end = time.time()
        relabel_stats["elapsed_seconds"] = round(t_relabel_end - t_relabel_start, 2)
        logger.info(
            "Batch relabel complete: %d images processed, %d labeled, %d skipped, %d boxes in %.2fs (%.1f fps).",
            relabel_stats.get("processed", 0),
            relabel_stats.get("labeled", 0),
            relabel_stats.get("skipped", 0),
            relabel_stats.get("total_boxes", 0),
            relabel_stats.get("elapsed_seconds", 0.0),
            relabel_stats.get("fps", 0.0),
        )
    else:
        logger.info("Skipping batch relabel pass as requested or 0 images found.")

    # 3. Smart AI Poison Purge (Katman 0-3)
    logger.info("Executing 4-Layer Smart AI Poison Purge...")
    t_purge_start = time.time()
    purge_fn = getattr(
        label_trust,
        "smart_ai_purge_poisoned_labels",
        getattr(label_trust, "clean_poison_labels_ai", None),
    )
    if purge_fn is not None:
        purge_stats = purge_fn(
            dataset_dir=dataset_path,
            floor=conf_floor,
            emin_threshold=emin_threshold,
            teach_negatives=True,
        )
    else:
        purge_stats = {"poison_deleted_boxes": 0, "auto_repaired_boxes": 0}
    t_purge_end = time.time()
    logger.info(
        "Smart AI Poison Purge complete: %d protected (Katman 0), %d auto-repaired (Katman 1), "
        "%d poison dropped (Katman 1-3), %d taught to negative memory in %.2fs.",
        purge_stats.get("protected_boxes", 0),
        purge_stats.get("auto_repaired_boxes", 0),
        purge_stats.get("poison_deleted_boxes", 0),
        purge_stats.get("negative_taught_boxes", 0),
        t_purge_end - t_purge_start,
    )

    # 4. Box-Level Bulk Low-Confidence Pruning ("Görseli Koru")
    logger.info(
        "Executing Box-Level Low-Confidence Pruning (< %.2f conf)...", prune_threshold
    )
    t_prune_start = time.time()
    prune_stats = prune_low_confidence_boxes_in_dataset(
        dataset_dir=dataset_path,
        threshold=prune_threshold,
        record_negative_exemplars=True,
    )
    t_prune_end = time.time()
    logger.info(
        "Low-confidence pruning complete: %d boxes pruned, %d boxes preserved across %d modified frames in %.2fs.",
        prune_stats.get("pruned_boxes", 0),
        prune_stats.get("preserved_boxes", 0),
        prune_stats.get("modified_frames", 0),
        t_prune_end - t_prune_start,
    )

    # 5. Rebuild Manifest Stats
    logger.info("Rebuilding manifest_stats.json...")
    manifest = curator.rebuild_manifest_stats()
    t_total_end = time.time()
    total_elapsed = max(0.001, t_total_end - t_start_total)
    overall_fps = total_images / total_elapsed if total_images > 0 else 0.0

    # Benchmark aggregation
    benchmark = {
        "dataset_dir": str(dataset_path),
        "total_images_discovered": total_images,
        "images_processed": relabel_stats.get("processed", total_images),
        "images_labeled": relabel_stats.get("labeled", 0),
        "images_skipped": relabel_stats.get("skipped", 0),
        "boxes_labeled": relabel_stats.get("total_boxes", 0),
        "boxes_auto_repaired": purge_stats.get("auto_repaired_boxes", 0),
        "boxes_poison_deleted": purge_stats.get("poison_deleted_boxes", 0),
        "boxes_low_conf_pruned": prune_stats.get("pruned_boxes", 0),
        "total_boxes_purged_or_pruned": (
            purge_stats.get("poison_deleted_boxes", 0)
            + prune_stats.get("pruned_boxes", 0)
        ),
        "negative_exemplars_taught": (
            purge_stats.get("negative_taught_boxes", 0)
            + prune_stats.get("negatives_taught", 0)
        ),
        "relabel_elapsed_seconds": relabel_stats.get("elapsed_seconds", 0.0),
        "relabel_fps": relabel_stats.get("fps", 0.0),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "overall_fps": round(overall_fps, 2),
        "manifest_stats": manifest,
    }

    # Print Clean Formatted Report
    print("\n" + "=" * 80)
    print("       TURBO BATCH RELABEL & HIGH-PRECISION AI CLEAN BENCHMARK REPORT")
    print("=" * 80)
    print(f" Dataset Path:                  {dataset_path}")
    print(f" Total Images Discovered:       {total_images}")
    print(f" Images Processed:              {benchmark['images_processed']}")
    print(f" Images Newly Labeled:          {benchmark['images_labeled']}")
    print(f" Images Cache-Skipped:          {benchmark['images_skipped']}")
    print(f" Total Bounding Boxes Labeled:  {benchmark['boxes_labeled']}")
    print(f" Boxes Auto-Repaired (Katman 1):{benchmark['boxes_auto_repaired']}")
    print(f" Poison Boxes Purged (Layer 1-3):{benchmark['boxes_poison_deleted']}")
    print(f" Low-Conf Boxes Pruned (<{prune_threshold:.2f}):{benchmark['boxes_low_conf_pruned']}")
    print(f" Total Boxes Removed:           {benchmark['total_boxes_purged_or_pruned']}")
    print(f" Negatives Taught to Memory:    {benchmark['negative_exemplars_taught']}")
    print(f" Relabel Time (s):              {benchmark['relabel_elapsed_seconds']}s")
    print(f" Relabel Throughput (fps):      {benchmark['relabel_fps']} fps")
    print(f" Total End-to-End Elapsed:      {benchmark['total_elapsed_seconds']}s")
    print(f" Overall Average Speed:         {benchmark['overall_fps']} fps")
    print("-" * 80)
    print(" Manifest Stats Summary (manifest_stats.json):")
    print(f"   Train Images:                {manifest.get('train_count', 0)}")
    print(f"   Val Images:                  {manifest.get('val_count', 0)}")
    print(f"   Total Images:                {manifest.get('total_images', 0)}")
    print(f"   Total Valid Bounding Boxes:  {manifest.get('total_boxes', 0)}")
    print(f"   Classes Represented:         {len(manifest.get('class_counts', {}))}")
    print(" Class Distribution:")
    for cid_str, cnt in sorted(
        manifest.get("class_counts", {}).items(), key=lambda x: int(x[0]) if x[0].isdigit() else 999
    ):
        cname = manifest.get("class_names", {}).get(cid_str, f"class_{cid_str}")
        print(f"   - Class {cid_str:2s} ({cname:20s}): {cnt:5d} boxes")
    print("=" * 80 + "\n")

    return benchmark


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Turbo Batch Relabel & AI Poison Purge Runner"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dataset",
        help="Path to dataset directory (default: ROOT/dataset)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Number of threads for parallel relabeling",
    )
    parser.add_argument(
        "--conf-floor",
        type=float,
        default=0.45,
        help="Confidence floor for label export (default: 0.45)",
    )
    parser.add_argument(
        "--emin-threshold",
        type=float,
        default=0.85,
        help="Threshold for Emin (trusted) annotations (default: 0.85)",
    )
    parser.add_argument(
        "--prune-threshold",
        type=float,
        default=0.50,
        help="Threshold for low-confidence box pruning (default: 0.50)",
    )
    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Force relabel all frames even if cached",
    )
    parser.add_argument(
        "--no-incremental",
        action="store_true",
        help="Disable incremental caching",
    )
    parser.add_argument(
        "--skip-relabel",
        action="store_true",
        help="Skip relabeling pass and execute only AI poison purge & pruning",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of images processed (default: 0 = all)",
    )
    args = parser.parse_args()

    results = run_batch_turbo_relabel_and_clean(
        dataset_dir=args.dataset,
        max_workers=args.max_workers,
        conf_floor=args.conf_floor,
        emin_threshold=args.emin_threshold,
        prune_threshold=args.prune_threshold,
        force_all=bool(args.force_all),
        incremental=not bool(args.no_incremental),
        skip_relabel=bool(args.skip_relabel),
        limit=int(args.limit),
    )

    return 0 if results.get("manifest_stats", {}).get("total_images", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
