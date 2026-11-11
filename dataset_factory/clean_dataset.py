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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from dataset_factory.game_frame_gate import GameFrameGate, is_valid_game_frame

logger = logging.getLogger("DatasetFactory.CleanDataset")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s] [CleanDataset] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# F-0919: public production surface of this module. The module is an operator
# CLI tool (python -m dataset_factory.clean_dataset); its production caller is
# the operator, not another module. Pipeline wiring (auto-purge step inside
# dataset_factory/dataset_hygiene_gate.py) is cross-slice (R47).
__all__ = [
    "purge_non_game_screenshots",
    "clean_dataset",
    "main",
    "SUPPORTED_IMAGE_EXTS",
]

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def _resolve_workers(max_workers: Optional[int]) -> int:
    """None / <=0 means auto: up to 8 threads, bounded by CPU count."""
    if max_workers is None or max_workers <= 0:
        return max(1, min(8, os.cpu_count() or 1))
    return max(1, int(max_workers))


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


_thread_gate_state = threading.local()


def _get_thread_gate() -> GameFrameGate:
    """Per-thread gate instance, so parallel validation never shares mutable
    gate state (dHash cache / stats) across threads. Built lazily, once per
    worker thread; template loading cost is paid once, not per image."""
    gate = getattr(_thread_gate_state, "gate", None)
    if gate is None:
        gate = GameFrameGate()
        _thread_gate_state.gate = gate
    return gate


def _validate_one_image(
    img_path: Path,
    require_foreground: bool,
    require_novelty: bool,
    gate: Optional[GameFrameGate] = None,
) -> Tuple[bool, str]:
    """Decode + gate one image. ``gate=None`` uses the shared module-level
    gate (exact pre-existing sequential behaviour)."""
    frame = _read_image_unicode(img_path)
    if frame is None:
        return False, "unreadable_corrupt_image"
    return is_valid_game_frame(
        frame=frame,
        require_foreground=require_foreground,
        require_novelty=require_novelty,
        gate=gate,
    )


def _validate_images(
    image_files: Sequence[Path],
    require_foreground: bool,
    require_novelty: bool,
    max_workers: Optional[int],
) -> List[Tuple[bool, str]]:
    """Validate a batch of images, preserving input order in the result.

    The expensive part (decode + Kapı 2 visual HUD evidence, measured
    ~3.5-5.2 s per textured 1080p noise frame sequentially on this VM,
    ~2.3-3.5 s/frame with 4 workers, ~1.5x — min of 2-3 runs of 6 frames;
    timings are noisy on this shared VM) runs in a thread pool — cv2/numpy
    release the GIL, so threads give a real speedup. Filesystem actions
    (move/remove) stay sequential in the caller, so on-disk behaviour and
    result ordering are identical to the sequential run.

    Sequential fallback is forced when ``require_novelty`` is True: the
    novelty gate shares one dHash cache, and per-thread caches would make
    duplicate decisions depend on thread scheduling.
    """
    if require_novelty:
        return [
            _validate_one_image(p, require_foreground, require_novelty)
            for p in image_files
        ]
    workers = _resolve_workers(max_workers)
    if workers <= 1 or len(image_files) <= 1:
        return [
            _validate_one_image(p, require_foreground, require_novelty)
            for p in image_files
        ]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            pool.map(
                lambda p: _validate_one_image(
                    p, require_foreground, require_novelty, gate=_get_thread_gate()
                ),
                image_files,
            )
        )


def purge_non_game_screenshots(
    dataset_dir: Union[str, Path] = "dataset",
    action: str = "move",
    quarantine_dir: Optional[Union[str, Path]] = None,
    require_foreground: bool = False,
    require_novelty: bool = False,
    dry_run: bool = False,
    verbose: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Scans dataset image directories and purges or quarantines non-game captures.

    Args:
        dataset_dir: Root dataset folder (default `dataset`).
        action: Either `"remove"` to delete files or `"move"` to relocate to quarantine.
            F-0832: varsayılan artık `"move"` (karantina) — yıkıcı
            "remove" varsayılanı yanlışlıkla silmeye yol açıyordu.
        quarantine_dir: Target folder if `action="move"`. Defaults to `<dataset_dir>/quarantine`.
        require_foreground: If False (default), Kapı 1 is bypassed for offline analysis.
        require_novelty: If False (default), dHash novelty check is bypassed during purging
                         to avoid duplicate purging of similar game frames.
        dry_run: If True, only reports invalid files without modifying the disk.
        verbose: Log detailed progress.
        max_workers: Threads used for the decode+validate phase (None = auto,
            up to 8 bounded by CPU count; 1 = sequential). Ignored when
            ``require_novelty`` is True — novelty then forces the sequential
            path so duplicate decisions stay scheduling-independent.
            Mutation stays sequential and in sorted order, so the summary
            is deterministic.

    Returns:
        Summary dictionary of scanned, valid, and purged files.
    """
    t0 = time.time()
    root = Path(dataset_dir).resolve()
    action = str(action).strip().lower()
    if action not in ("remove", "move"):
        action = "move"

    if not root.is_dir():
        logger.error("Dataset root does not exist: %s", root)
        return {
            "total_scanned": 0,
            "valid_game_frames": 0,
            "invalid_purged": 0,
            "purged_images": [],
            "purged_labels": [],
            "purged_jsons": [],
            "failed_images": [],
            "errors": 1,
            "reasons": {},
            "scanned_directories": [],
            "dry_run": dry_run,
            "action": action,
            "elapsed_sec": round(time.time() - t0, 3),
        }

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
    failed_images: List[str] = []
    errors = 0
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

    # S-5403: previews live in a flat previews/ dir and are resolved by bare
    # stem. If the same stem exists in more than one scanned directory (or
    # twice in one directory with different extensions), a flat preview file
    # cannot be attributed to one image -- purging it could delete or
    # quarantine another image's preview. Collect the scan plan once and
    # count stems up front; ambiguous stems skip preview handling.
    dir_images: Dict[str, List[Path]] = {}
    stem_counts: Dict[str, int] = {}
    for _dir_key, _dir_path in target_scan_dirs:
        if not _dir_path.is_dir():
            continue
        _files = sorted(
            p for p in _dir_path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        dir_images[_dir_key] = _files
        for _p in _files:
            stem_counts[_p.stem] = stem_counts.get(_p.stem, 0) + 1

    for dir_key, dir_path in target_scan_dirs:
        if not dir_path.is_dir():
            continue
        scanned_dirs.append(str(dir_path))

        image_files = dir_images.get(dir_key, [])

        if verbose:
            logger.info("Scanning %s (%d candidate images found)", dir_path, len(image_files))

        # Phase 1: decode + gate (parallel unless novelty forces sequential).
        # Phase 2 below (filesystem actions) stays sequential and in sorted
        # order, so results are identical to the old single-pass loop.
        validations = _validate_images(
            image_files,
            require_foreground=require_foreground,
            require_novelty=require_novelty,
            max_workers=max_workers,
        )

        for img_path, (is_valid, reason) in zip(image_files, validations):
            total_scanned += 1

            if is_valid:
                valid_count += 1
                continue

            # Invalid non-game screenshot detected!
            reason_category = reason.split(":")[0].strip()
            reasons_count[reason_category] = reasons_count.get(reason_category, 0) + 1

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

            # Locate associated .json metadata (common in game_learning / review)
            associated_json: Optional[Path] = None
            json_cand = img_path.parent / f"{img_path.stem}.json"
            if json_cand.is_file():
                associated_json = json_cand

            # Locate associated preview
            # S-5403: only when this stem is unique across all scanned
            # images; otherwise the flat previews/<stem>.jpg file may belong
            # to another directory's image with the same stem -- leave it
            # untouched instead of deleting/quarantining the wrong preview.
            preview_cand: Optional[Path] = None
            if stem_counts.get(img_path.stem, 0) == 1:
                cand = root / "previews" / f"{img_path.stem}.jpg"
                if not cand.is_file():
                    cand = root / "previews" / f"{img_path.stem}.png"
                if cand.is_file():
                    preview_cand = cand
            elif verbose:
                logger.warning(
                    "Ambiguous preview stem %r (%d images share it); "
                    "leaving preview untouched",
                    img_path.stem,
                    stem_counts.get(img_path.stem, 0),
                )

            # F-0918: özet yalnızca GERÇEKTEN silinen/taşınan dosyayı
            # "purged" sayar. Silinemeyen dosya failed_images'e düşer.
            if dry_run:
                purged_images.append(str(img_path))
                if associated_label is not None:
                    purged_labels.append(str(associated_label))
                if associated_json is not None:
                    purged_jsons.append(str(associated_json))
                continue

            # Execute purging or quarantine
            try:
                if action == "remove":
                    # ai/A15: image deleted LAST — if an associated file
                    # fails, the image survives and the run is retryable
                    # instead of leaving an orphaned label for a lost image.
                    if associated_label and associated_label.exists():
                        associated_label.unlink()
                    if associated_json and associated_json.exists():
                        associated_json.unlink()
                    if preview_cand is not None and preview_cand.exists():
                        preview_cand.unlink()
                    if img_path.exists():
                        img_path.unlink()
                elif action == "move":
                    # ai/A15: track every move; on failure roll back the
                    # ones already done so no half-moved sample is left
                    # (image in quarantine, label orphaned in dataset).
                    moved: List[Tuple[Path, Path]] = []
                    try:
                        rel_path = img_path.relative_to(root)
                        dest_img = q_dir / rel_path
                        dest_img.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(img_path), str(dest_img))
                        moved.append((dest_img, img_path))

                        if associated_label and associated_label.exists():
                            rel_lbl = associated_label.relative_to(root)
                            dest_lbl = q_dir / rel_lbl
                            dest_lbl.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(associated_label), str(dest_lbl))
                            moved.append((dest_lbl, associated_label))

                        if associated_json and associated_json.exists():
                            rel_json = associated_json.relative_to(root)
                            dest_json = q_dir / rel_json
                            dest_json.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(associated_json), str(dest_json))
                            moved.append((dest_json, associated_json))

                        if preview_cand is not None and preview_cand.exists():
                            rel_prev = preview_cand.relative_to(root)
                            dest_prev = q_dir / rel_prev
                            dest_prev.parent.mkdir(parents=True, exist_ok=True)
                            shutil.move(str(preview_cand), str(dest_prev))
                            moved.append((dest_prev, preview_cand))
                    except Exception:
                        for done_dst, done_src in reversed(moved):
                            try:
                                done_src.parent.mkdir(parents=True, exist_ok=True)
                                shutil.move(str(done_dst), str(done_src))
                            except Exception:
                                logger.warning(
                                    "Rollback failed for %s (stays in quarantine)",
                                    done_dst,
                                )
                        raise
            except Exception as exc:
                errors += 1
                failed_images.append(str(img_path))
                logger.warning("Error processing %s (%s): %s", img_path.name, action, exc)
                continue

            purged_images.append(str(img_path))
            if associated_label is not None:
                purged_labels.append(str(associated_label))
            if associated_json is not None:
                purged_jsons.append(str(associated_json))

    elapsed = round(time.time() - t0, 3)
    summary: Dict[str, Any] = {
        "total_scanned": total_scanned,
        "valid_game_frames": valid_count,
        "invalid_purged": len(purged_images),
        "purged_images": purged_images,
        "purged_labels": purged_labels,
        "purged_jsons": purged_jsons,
        "failed_images": failed_images,
        "errors": errors,
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
        logger.info("  Failed:              %d", errors)
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
        default="move",
        help="Action on invalid images: 'remove' (delete) or 'move' (quarantine, default)",
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
    # F-0832: main() artık her durumda 0 dönmez; hata olduysa 1 döner.
    return 1 if res.get("errors", 0) else 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
