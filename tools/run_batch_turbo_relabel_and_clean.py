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

import dataset_factory.label_trust as label_trust
from dataset_factory.dataset_curator import DatasetCurator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TurboRelabelRunner")


# ==============================================================================
# Box-Level Low-Confidence Pruner ("✂️ Düşük Güvenli Etiketleri Sil (Görseli Koru)")
# ==============================================================================
# F-1216: the canonical implementation lives in dataset_factory.label_trust
# (which HAS prune_low_confidence_boxes_in_dataset). The ~150-line local
# copy below it used to be dead code — the old delegation branch at the top
# of the function always won — and it had drifted from the real
# implementation. It is replaced by a thin delegating wrapper so there is
# exactly one implementation; the wrapper only normalises the result keys
# this runner's reporting expects.
def prune_low_confidence_boxes_in_dataset(
    dataset_dir: Union[str, Path],
    threshold: float = 0.50,
    indices_or_stems: Optional[Iterable[str]] = None,
    record_negative_exemplars: bool = True,
) -> Dict[str, Any]:
    """
    Remove ONLY low-confidence boxes (< threshold) from review JSON and label TXT.
    The image and all remaining confident boxes (>= threshold) are 100% PRESERVED.
    Delegates to dataset_factory.label_trust.prune_low_confidence_boxes_in_dataset.
    """
    result = dict(
        label_trust.prune_low_confidence_boxes_in_dataset(
            dataset_dir=dataset_dir,
            threshold=threshold,
            indices_or_stems=indices_or_stems,
            record_negative_exemplars=record_negative_exemplars,
        )
    )
    # Normalise keys for this runner's reporting (canonical impl reports
    # "scanned"/"files_rewritten"/"promoted_boxes"; it does not report a
    # negative-exemplar count, so that stays 0 rather than invented).
    result.setdefault("scanned_frames", result.get("scanned", 0))
    result.setdefault("modified_frames", result.get("files_rewritten", 0))
    result.setdefault("preserved_boxes", result.get("promoted_boxes", 0))
    result.setdefault(
        "total_boxes_before",
        result.get("pruned_boxes", 0) + result.get("promoted_boxes", 0),
    )
    result.setdefault("negatives_taught", 0)
    return result


# Ensure label_trust has aliases if missing
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
    purge_available = purge_fn is not None
    if purge_fn is not None:
        purge_stats = purge_fn(
            dataset_dir=dataset_path,
            floor=conf_floor,
            emin_threshold=emin_threshold,
            teach_negatives=True,
        )
    else:
        # F-1272: a missing purge function must NOT silently report zeros
        # while the run still exits 0 — flag it and fail at the end.
        logger.error(
            "Smart AI Poison Purge UNAVAILABLE: neither "
            "label_trust.smart_ai_purge_poisoned_labels nor "
            "label_trust.clean_poison_labels_ai exists — purge step skipped."
        )
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
        "images_processed": relabel_stats.get("processed", 0),
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
        "purge_available": purge_available,
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
    # F-1215: rebuild_manifest_stats keys "class_counts" by CLASS NAME
    # (e.g. {"monster_3d": 225}) and "class_id_counts" by numeric id
    # (e.g. {"10": 225}); the old code sorted the NAME keys as if they
    # were numeric ids and looked names up in a "class_names" key that
    # the manifest does not have. Prefer the id-keyed map (numeric order,
    # names from the curator taxonomy) and fall back to the name map.
    class_counts = manifest.get("class_counts", {}) or {}
    class_id_counts = manifest.get("class_id_counts", {}) or {}
    print(f"   Classes Represented:         {len(class_counts)}")
    print(" Class Distribution:")
    if class_id_counts:
        for cid_str, cnt in sorted(
            class_id_counts.items(),
            key=lambda x: int(x[0]) if str(x[0]).isdigit() else 999,
        ):
            try:
                cname = curator.class_names.get(int(cid_str), f"class_{cid_str}")
            except (TypeError, ValueError):
                cname = f"class_{cid_str}"
            print(f"   - Class {cid_str:>2s} ({cname:20s}): {cnt:5d} boxes")
    else:
        for cname, cnt in sorted(class_counts.items()):
            print(f"   - {cname:20s}: {cnt:5d} boxes")
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

    # F-1272: exit code reflects the purge step too, not just the manifest.
    if not results.get("purge_available", True):
        return 1
    return 0 if results.get("manifest_stats", {}).get("total_images", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
