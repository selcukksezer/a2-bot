# -*- coding: utf-8 -*-
"""
Desktop & Non-Game Screenshot Purging Utility (`dataset_factory/clean_dataset.py`).
==================================================================================
Scans existing YOLO training datasets (`dataset/images/train/`, `dataset/images/val/`,
`dataset/review/`, and `dataset/game_learning/`) for accidental non-game screenshots
(Windows desktop, browser canvases, IDE windows, solid color, or blank captures).

Routes each image through `is_valid_game_frame(frame, require_foreground=False)`
(Kapı 2 Visual HUD Evidence check). Invalid captures and their corresponding `.txt`
label files (and `.json` metadata files) are safely purged or quarantined.

Exclusive Owner: Worker 15 (`dataset_factory/clean_dataset.py`)
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from dataset_factory.game_frame_gate import is_valid_game_frame

logger = logging.getLogger("DatasetFactory.CleanDataset")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] [CleanDataset] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _read_image_unicode(path: Union[str, Path]) -> Optional[np.ndarray]:
    """Reads an image using numpy to safely handle Windows Unicode paths."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        raw = np.fromfile(str(p), dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception as exc:
        logger.debug("Failed to read image %s: %s", path, exc)
        return None


def purge_non_game_screenshots(
    dataset_dir: Union[str, Path] = "dataset",
    action: str = "remove",
    quarantine_dir: Optional[Union[str, Path]] = None,
    require_foreground: bool = False,
    require_novelty: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Scans dataset image directories and purges or quarantines non-game captures.

    Args:
        dataset_dir: Root dataset folder (default `dataset`).
        action: Either `"remove"` to delete files or `"move"` to relocate to quarantine.
        quarantine_dir: Target folder if `action="move"`. Defaults to `<dataset_dir>/quarantine`.
        require_foreground: If False (default), Kapı 1 is bypassed for offline analysis.
        require_novelty: If False (default), dHash novelty check is bypassed during purging
                         to avoid duplicate purging of similar game frames.
        dry_run: If True, only reports invalid files without modifying the disk.
        verbose: Log detailed progress.

    Returns:
        Summary dictionary of scanned, valid, and purged files.
    """
    t0 = time.time()
    root = Path(dataset_dir).resolve()
    action = str(action).strip().lower()
    if action not in ("remove", "move"):
        action = "remove"

    if quarantine_dir is not None:
        q_dir = Path(quarantine_dir).resolve()
    else:
        q_dir = root / "quarantine"

    if action == "move" and not dry_run:
        q_dir.mkdir(parents=True, exist_ok=True)

    target_scan_dirs: List[Tuple[str, Path]] = [
        ("train", root / "images" / "train"),
        ("val", root / "images" / "val"),
        ("review", root / "review"),
        ("game_learning", root / "game_learning"),
    ]

    scanned_dirs: List[str] = []
    purged_images: List[str] = []
    purged_labels: List[str] = []
    purged_jsons: List[str] = []
    reasons_count: Dict[str, int] = {}
    total_scanned = 0
    valid_count = 0

    if verbose:
        logger.info(
            "Starting screenshot scan across %s (action=%s, dry_run=%s)",
            root,
            action,
            dry_run,
        )

    for dir_key, dir_path in target_scan_dirs:
        if not dir_path.is_dir():
            continue
        scanned_dirs.append(str(dir_path))

        image_files = [
            p for p in dir_path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        ]
        image_files.sort()

        if verbose:
            logger.info("Scanning %s (%d candidate images found)", dir_path, len(image_files))

        for img_path in image_files:
            total_scanned += 1
            frame = _read_image_unicode(img_path)

            if frame is None:
                is_valid = False
                reason = "unreadable_corrupt_image"
            else:
                is_valid, reason = is_valid_game_frame(
                    frame=frame,
                    require_foreground=require_foreground,
                    require_novelty=require_novelty,
                )

            if is_valid:
                valid_count += 1
                continue

            # Invalid non-game screenshot detected!
            reason_category = reason.split(":")[0].strip()
            reasons_count[reason_category] = reasons_count.get(reason_category, 0) + 1
            purged_images.append(str(img_path))

            # Locate associated .txt label
            associated_label: Optional[Path] = None
            if dir_key == "train":
                lbl_cand = root / "labels" / "train" / f"{img_path.stem}.txt"
                if lbl_cand.is_file():
                    associated_label = lbl_cand
            elif dir_key == "val":
                lbl_cand = root / "labels" / "val" / f"{img_path.stem}.txt"
                if lbl_cand.is_file():
                    associated_label = lbl_cand
            else:
                lbl_cand = img_path.parent / f"{img_path.stem}.txt"
                if lbl_cand.is_file():
                    associated_label = lbl_cand

            if associated_label is not None:
                purged_labels.append(str(associated_label))

            # Locate associated .json metadata (common in game_learning / review)
            associated_json: Optional[Path] = None
            json_cand = img_path.parent / f"{img_path.stem}.json"
            if json_cand.is_file():
                associated_json = json_cand
                purged_jsons.append(str(associated_json))

            # Locate associated preview
            preview_cand = root / "previews" / f"{img_path.stem}.jpg"
            if not preview_cand.is_file():
                preview_cand = root / "previews" / f"{img_path.stem}.png"

            # Execute purging or quarantine
            if not dry_run:
                try:
                    if action == "remove":
                        if img_path.exists():
                            img_path.unlink()
                        if associated_label and associated_label.exists():
                            associated_label.unlink()
                        if associated_json and associated_json.exists():
                            associated_json.unlink()
                        if preview_cand.exists():
                            preview_cand.unlink()
                    elif action == "move":
                        rel_path = img_path.relative_to(root)
                        dest_img = q_dir / rel_path
                        dest_img.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(img_path), str(dest_img))

                        if associated_label and associated_label.exists():
                            rel_lbl = associated_label.relative_to(root)
                            dest_lbl = q_dir / rel_lbl
                            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(associated_label), str(dest_lbl))

                        if associated_json and associated_json.exists():
                            rel_json = associated_json.relative_to(root)
                            dest_json = q_dir / rel_json
                            dest_json.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(associated_json), str(dest_json))

                        if preview_cand.exists():
                            rel_prev = preview_cand.relative_to(root)
                            dest_prev = q_dir / rel_prev
                            dest_prev.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(preview_cand), str(dest_prev))
                except Exception as exc:
                    logger.warning("Error processing %s (%s): %s", img_path.name, action, exc)

    elapsed = round(time.time() - t0, 3)
    summary: Dict[str, Any] = {
        "total_scanned": total_scanned,
        "valid_game_frames": valid_count,
        "invalid_purged": len(purged_images),
        "purged_images": purged_images,
        "purged_labels": purged_labels,
        "purged_jsons": purged_jsons,
        "reasons": reasons_count,
        "scanned_directories": scanned_dirs,
        "dry_run": dry_run,
        "action": action,
        "elapsed_sec": elapsed,
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("DESKTOP & NON-GAME SCREENSHOT PURGING SUMMARY:")
        logger.info("  Total scanned:       %d", total_scanned)
        logger.info("  Valid game frames:   %d", valid_count)
        logger.info("  Invalid non-game:    %d", len(purged_images))
        logger.info("  Label files purged:  %d", len(purged_labels))
        logger.info("  JSON files purged:   %d", len(purged_jsons))
        logger.info("  Reasons distribution: %s", reasons_count)
        logger.info("  Action executed:     %s (dry_run=%s)", action, dry_run)
        logger.info("  Elapsed time:        %.2fs", elapsed)
        logger.info("=" * 60)

    return summary


# Backward-compatible alias
clean_dataset = purge_non_game_screenshots


# ==============================================================================
# CLI Entrypoint
# ==============================================================================
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan dataset directories and purge desktop / non-game screenshots."
    )
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        help="Root path to the dataset folder (default: dataset)",
    )
    parser.add_argument(
        "--action",
        choices=["remove", "move"],
        default="remove",
        help="Action on invalid images: 'remove' (delete) or 'move' (quarantine)",
    )
    parser.add_argument(
        "--quarantine-dir",
        default=None,
        help="Destination directory if action is 'move' (default: dataset/quarantine)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report invalid images without deleting or moving files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging output",
    )

    args = parser.parse_args()
    res = purge_non_game_screenshots(
        dataset_dir=args.dataset_dir,
        action=args.action,
        quarantine_dir=args.quarantine_dir,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )
    return 0 if res["invalid_purged"] == 0 or args.dry_run else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
