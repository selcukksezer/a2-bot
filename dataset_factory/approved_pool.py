# -*- coding: utf-8 -*-
"""dataset_factory/approved_pool.py
===================================
Aion 2 Autonomous Dataset Factory - Approved Dataset Pool & Isolated Training Pipeline
Division 1 (A04, A07): Exclusive Isolated Pool for Human-Approved YOLO Training Samples.

User Requirement:
"onaylananlar kuyruktan eksilsin birde tekrar tekrar aç kapa yapınca geri gelmesin
onaylananlar ayrı bir yere gitsin eğitim için bekletilsin / onaylananlarla eğit seçeneği olsun hepsini eğitime almasın"

Key Capabilities:
1. Isolated Root Hierarchy (`dataset/approved/`):
   - `dataset/approved/images/train/`, `dataset/approved/images/val/`
   - `dataset/approved/labels/train/`, `dataset/approved/labels/val/`
   - `dataset/approved/approved_manifest.json`
   - `dataset/approved/data_approved.yaml`
2. Sample Quarantine & Dedup:
   - Once a sample is approved (`is_approved(stem) == True`), it never re-enters
     the Step 2 review queue on dialog open/close or disk rescan.
   - Atomically copies/saves image and verified YOLO `.txt` labels to `dataset/approved/`.
   - Records metadata, box counts, classes, and timestamps in `approved_manifest.json`.
3. Canonical 22-Class YOLO YAML Generation:
   - Generates valid Ultralytics `data_approved.yaml` pointing to approved images.
4. Selective Training Pipeline:
   - Enables "Sadece Onaylananlarla Eğit" (Train Only with Approved Samples).
   - Automatically supports 85%/15% train/val balanced distribution.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

logger = logging.getLogger("DatasetFactory.ApprovedPool")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

# =============================================================================
# Canonical Class Definitions Fallback (0..21 Core Classes)
# =============================================================================

DEFAULT_CORE_22_CANONICAL_NAMES: Dict[int, str] = {
    0: "herb_aria",
    1: "herb_mela",
    2: "herb_targena",
    3: "ore_orichalcum",
    4: "gem_crystal",
    5: "tree_log",
    6: "herb_edana",
    7: "herb_lumina",
    8: "ore_obsidian",
    9: "minimap_sprout",
    10: "monster_3d",
    11: "minimap_monster_dot",
    12: "crystal_blood",
    13: "skill_ui_prompt",
    14: "obstacle",
    15: "path_ground",
    16: "player_hp_bar",
    17: "player_mp_bar",
    18: "target_hp_bar",
    19: "skill_bar",
    20: "skill_slot_ready",
    21: "skill_slot_cooldown",
}

try:
    from dataset_factory.class_taxonomy import CORE_22_CANONICAL_NAMES
except Exception:
    CORE_22_CANONICAL_NAMES = DEFAULT_CORE_22_CANONICAL_NAMES


# =============================================================================
# Helper Utilities: Atomic Disk I/O & Bounding Box Normalization
# =============================================================================

def _safe_replace(src: Path, dst: Path, max_retries: int = 5, delay: float = 0.05) -> None:
    """Replace destination file atomically with retries for Windows file lock tolerance."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    for i in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if i == max_retries - 1:
                raise
            time.sleep(delay)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Atomically write text content using a temporary file and atomic rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        tmp_path.write_text(text, encoding=encoding)
        _safe_replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def atomic_copy_file(src_path: Path, dest_path: Path) -> None:
    """Atomically copy a file to dest_path using a temp file and atomic rename."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{time.time_ns()}")
    try:
        shutil.copy2(src_path, tmp_path)
        _safe_replace(tmp_path, dest_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        raise


def normalize_box(b: Any) -> Optional[Tuple[int, float, float, float, float]]:
    """
    Normalizes a single box representation into (class_id, cx, cy, w, h).
    Accepts dict, list, tuple, or YOLO-format string.
    Ensures coordinates are clamped to [0.0, 1.0] and dimensions are positive.
    """
    try:
        if isinstance(b, str):
            parts = b.strip().split()
            if len(parts) < 5:
                return None
            cid = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        elif isinstance(b, dict):
            cid = int(b.get("class_id", b.get("cid", b.get("cls", b.get("label", 0)))))
            cx = float(b.get("cx", b.get("x_center", b.get("x", 0.0))))
            cy = float(b.get("cy", b.get("y_center", b.get("y", 0.0))))
            w = float(b.get("w", b.get("width", 0.0)))
            h = float(b.get("h", b.get("height", 0.0)))
        elif isinstance(b, (list, tuple)) and len(b) >= 5:
            cid = int(float(b[0]))
            cx = float(b[1])
            cy = float(b[2])
            w = float(b[3])
            h = float(b[4])
        else:
            return None

        if w <= 0.0 or h <= 0.0:
            return None

        # Clamp normalized coordinates safely
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0001, min(1.0, w))
        h = max(0.0001, min(1.0, h))

        return cid, cx, cy, w, h
    except (ValueError, TypeError, KeyError):
        return None


def parse_and_format_yolo_boxes(raw_boxes: Optional[Sequence[Any]]) -> Tuple[List[str], List[Tuple[int, float, float, float, float]], Set[int]]:
    """
    Parses arbitrary box collections into:
      1. List of YOLO lines: "cid cx cy w h"
      2. List of validated box tuples: (cid, cx, cy, w, h)
      3. Set of unique class IDs present.
    """
    lines: List[str] = []
    valid_boxes: List[Tuple[int, float, float, float, float]] = []
    classes_present: Set[int] = set()

    if not raw_boxes:
        return lines, valid_boxes, classes_present

    for item in raw_boxes:
        norm = normalize_box(item)
        if norm is not None:
            cid, cx, cy, w, h = norm
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            valid_boxes.append(norm)
            classes_present.add(cid)

    return lines, valid_boxes, classes_present


# =============================================================================
# ApprovedDatasetPool Class
# =============================================================================

class ApprovedDatasetPool:
    """
    Manages the isolated `dataset/approved/` pool for human-approved frames.
    
    Guarantees:
    - Atomicity: Images and YOLO label files are written with temp-rename.
    - Dedup / Quarantine: Once a stem is marked approved, `is_approved(stem)`
      returns True so it never reappears in pending Step 2 queues.
    - YAML Export: Produces valid Ultralytics `data_approved.yaml` with
      canonical 22 classes.
    - Statistics: Real-time telemetry tracking total images, split counts,
      box counts, and class distributions.
    """

    def __init__(
        self,
        pool_root: Optional[Union[str, Path]] = None,
        project_root: Optional[Union[str, Path]] = None,
    ) -> None:
        if project_root is not None:
            p = Path(project_root).resolve()
            if p.name == "dataset":
                self.project_root = p.parent
                if pool_root is None:
                    pool_root = p / "approved"
            else:
                self.project_root = p
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        if pool_root is not None:
            self.pool_root = Path(pool_root).resolve()
        else:
            self.pool_root = self.project_root / "dataset" / "approved"

        # Directory structure
        self.images_train_dir: Path = self.pool_root / "images" / "train"
        self.images_val_dir: Path = self.pool_root / "images" / "val"
        self.labels_train_dir: Path = self.pool_root / "labels" / "train"
        self.labels_val_dir: Path = self.pool_root / "labels" / "val"
        self.manifest_path: Path = self.pool_root / "approved_manifest.json"
        self.data_yaml_path: Path = self.pool_root / "data_approved.yaml"

        self._lock = threading.RLock()
        self._approved_stems: Set[str] = set()
        self._manifest: Dict[str, Any] = {
            "version": "1.0",
            "updated_at": 0.0,
            "total_approved_images": 0,
            "train_count": 0,
            "val_count": 0,
            "total_boxes": 0,
            "classes": {},
            "stems": {},
        }

        # Initialize directory layout and reload manifest
        self.ensure_pool_structure()
        self._load_or_rebuild_manifest()

    # -------------------------------------------------------------------------
    # Structure & Initialization
    # -------------------------------------------------------------------------

    def ensure_pool_structure(self) -> None:
        """Create the complete `dataset/approved/` directory hierarchy."""
        with self._lock:
            self.images_train_dir.mkdir(parents=True, exist_ok=True)
            self.images_val_dir.mkdir(parents=True, exist_ok=True)
            self.labels_train_dir.mkdir(parents=True, exist_ok=True)
            self.labels_val_dir.mkdir(parents=True, exist_ok=True)

    def _load_or_rebuild_manifest(self) -> None:
        """Load `approved_manifest.json` from disk, or rebuild from disk files if missing/corrupt."""
        with self._lock:
            loaded_ok = False
            if self.manifest_path.is_file():
                try:
                    data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                    if isinstance(data, dict) and "stems" in data:
                        self._manifest = data
                        self._approved_stems = set(data.get("stems", {}).keys())
                        loaded_ok = True
                except Exception as ex:
                    logger.warning("Manifest parse failed, backing up and rebuilding: %s", ex)
                    backup = self.manifest_path.with_suffix(".corrupt.json")
                    try:
                        shutil.copy2(self.manifest_path, backup)
                    except Exception:
                        pass

            if not loaded_ok:
                self.sync_from_disk()

    # -------------------------------------------------------------------------
    # Split Directories Helper
    # -------------------------------------------------------------------------

    def images_dir_for_split(self, split: str) -> Path:
        """Return image directory for a split ('train' or 'val')."""
        return self.images_val_dir if str(split).lower().strip() == "val" else self.images_train_dir

    def labels_dir_for_split(self, split: str) -> Path:
        """Return label directory for a split ('train' or 'val')."""
        return self.labels_val_dir if str(split).lower().strip() == "val" else self.labels_train_dir

    # -------------------------------------------------------------------------
    # Core Operation: Add Approved Sample
    # -------------------------------------------------------------------------

    def add_approved_sample(
        self,
        stem: str,
        split: str = "train",
        image_path: Optional[Union[str, Path, Any]] = None,
        trusted_boxes: Optional[Sequence[Any]] = None,
        image_bytes: Optional[bytes] = None,
        approved_at: Optional[float] = None,
        boxes: Optional[Sequence[Any]] = None,
    ) -> bool:
        """
        Atomically copies/saves image and verified YOLO `.txt` label into the approved pool,
        and registers the stem in `approved_manifest.json`.

        Args:
            stem: Clean image stem identifier (e.g. 'frame_000123').
            split: 'train' or 'val'. If other/empty, deterministically assigns 85% train / 15% val.
            image_path: Path to source image file, or numpy array. If None, resolves from dataset.
            trusted_boxes: List of trusted bounding boxes (dicts, tuples, or YOLO lines).
            image_bytes: Optional raw image bytes if already loaded in memory.
            approved_at: Optional approval epoch timestamp. Defaults to time.time().
            boxes: Optional alias for trusted_boxes to support robust callers.

        Returns:
            bool: True if sample was successfully persisted and recorded; False otherwise.
        """
        if trusted_boxes is None and boxes is not None:
            trusted_boxes = boxes

        if not stem:
            return False

        clean_stem = Path(stem).stem.strip()
        if not clean_stem:
            return False

        norm_split = str(split).lower().strip()
        if norm_split not in ("train", "val"):
            # Deterministic split based on MD5 hash (85% train, 15% val)
            h = int(hashlib.md5(clean_stem.encode("utf-8")).hexdigest(), 16)
            norm_split = "train" if (h % 100 < 85) else "val"

        timestamp = float(approved_at) if approved_at is not None else time.time()

        with self._lock:
            try:
                dest_images_dir = self.images_dir_for_split(norm_split)
                dest_labels_dir = self.labels_dir_for_split(norm_split)

                # 1. Resolve and copy image
                resolved_img_path, dest_img_name = self._save_or_copy_image(
                    clean_stem=clean_stem,
                    norm_split=norm_split,
                    image_path=image_path,
                    image_bytes=image_bytes,
                    dest_images_dir=dest_images_dir,
                )

                if resolved_img_path is None and dest_img_name is None:
                    logger.error("Could not find or save image for approved stem '%s'", clean_stem)
                    return False

                # 2. Parse and format YOLO label
                lines, valid_boxes, classes_present = parse_and_format_yolo_boxes(trusted_boxes)
                txt_content = "\n".join(lines) + ("\n" if lines else "")
                dest_label_path = dest_labels_dir / f"{clean_stem}.txt"
                atomic_write_text(dest_label_path, txt_content, encoding="utf-8")

                # 3. Update manifest and in-memory index
                stems_map = self._manifest.setdefault("stems", {})
                stems_map[clean_stem] = {
                    "stem": clean_stem,
                    "split": norm_split,
                    "image_file": dest_img_name,
                    "label_file": f"{clean_stem}.txt",
                    "box_count": len(valid_boxes),
                    "classes": sorted(list(classes_present)),
                    "approved_at": timestamp,
                    "source_image": str(resolved_img_path) if resolved_img_path else "",
                }

                self._approved_stems.add(clean_stem)
                self._recalculate_manifest_stats_and_save()
                return True

            except Exception as ex:
                logger.exception("Failed to add approved sample '%s': %s", clean_stem, ex)
                return False

    def remove_sample(self, stem: str) -> bool:
        """Remove an approved sample from the pool and manifest."""
        clean_stem = str(stem or "").strip()
        if not clean_stem:
            return False
        with self._lock:
            stems_map = self._manifest.setdefault("stems", {})
            entry = stems_map.pop(clean_stem, None)
            self._approved_stems.discard(clean_stem)
            for split_dir in (self.images_train_dir, self.images_val_dir):
                for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                    p = split_dir / f"{clean_stem}{ext}"
                    try:
                        if p.is_file():
                            p.unlink()
                    except OSError:
                        pass
            for split_dir in (self.labels_train_dir, self.labels_val_dir):
                p = split_dir / f"{clean_stem}.txt"
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass
            self._recalculate_manifest_stats_and_save()
            return True

    def _save_or_copy_image(
        self,
        clean_stem: str,
        norm_split: str,
        image_path: Optional[Union[str, Path, Any]],
        image_bytes: Optional[bytes],
        dest_images_dir: Path,
    ) -> Tuple[Optional[Path], Optional[str]]:
        """Internal helper to copy or write image to destination directory atomically."""
        # 1. Direct bytes provided
        if image_bytes is not None and len(image_bytes) > 0:
            dest_name = f"{clean_stem}.jpg"
            dest_path = dest_images_dir / dest_name
            tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{time.time_ns()}")
            tmp_path.write_bytes(image_bytes)
            _safe_replace(tmp_path, dest_path)
            return dest_path, dest_name

        # 2. Check if destination image already exists
        for ext in (".jpg", ".png", ".jpeg"):
            cand = dest_images_dir / f"{clean_stem}{ext}"
            if cand.is_file():
                return cand, cand.name

        # 3. Handle numpy image array
        if image_path is not None and hasattr(image_path, "shape") and hasattr(image_path, "dtype"):
            try:
                import cv2
                dest_name = f"{clean_stem}.jpg"
                dest_path = dest_images_dir / dest_name
                success, buf = cv2.imencode(".jpg", image_path)
                if success:
                    tmp_path = dest_path.with_suffix(f"{dest_path.suffix}.tmp_{os.getpid()}_{time.time_ns()}")
                    tmp_path.write_bytes(buf.tobytes())
                    _safe_replace(tmp_path, dest_path)
                    return dest_path, dest_name
            except Exception as ex:
                logger.warning("Failed to save numpy array for stem '%s': %s", clean_stem, ex)

        # 4. Resolve from filepath
        src_candidate: Optional[Path] = None
        if image_path is not None and isinstance(image_path, (str, Path)):
            p = Path(image_path)
            if p.is_file():
                src_candidate = p
            elif (self.project_root / p).is_file():
                src_candidate = (self.project_root / p).resolve()

        if src_candidate is None:
            # Fallback search in standard dataset image directories
            search_dirs = [
                self.project_root / "dataset" / "images" / norm_split,
                self.project_root / "dataset" / "images" / "train",
                self.project_root / "dataset" / "images" / "val",
                self.project_root / "dataset" / "images",
            ]
            for sdir in search_dirs:
                if not sdir.is_dir():
                    continue
                for ext in (".jpg", ".png", ".jpeg"):
                    cand = sdir / f"{clean_stem}{ext}"
                    if cand.is_file():
                        src_candidate = cand
                        break
                if src_candidate is not None:
                    break

        if src_candidate is not None and src_candidate.is_file():
            ext = src_candidate.suffix.lower()
            if ext not in (".png", ".jpg", ".jpeg"):
                ext = ".jpg"
            dest_name = f"{clean_stem}{ext}"
            dest_path = dest_images_dir / dest_name
            atomic_copy_file(src_candidate, dest_path)
            return src_candidate, dest_name

        return None, None

    # -------------------------------------------------------------------------
    # Bulk Operations
    # -------------------------------------------------------------------------

    def bulk_add_approved_samples(self, samples: Sequence[Dict[str, Any]]) -> int:
        """
        Adds multiple approved samples efficiently, updating the manifest once at the end.
        
        Each item in `samples` should be a dict with:
          - stem (required)
          - split (optional, default 'train')
          - image_path (optional)
          - trusted_boxes (optional)
          - approved_at (optional)
        """
        if not samples:
            return 0

        success_count = 0
        with self._lock:
            for s in samples:
                stem = s.get("stem")
                if not stem:
                    continue
                clean_stem = Path(stem).stem.strip()
                split = s.get("split", "train")
                norm_split = str(split).lower().strip()
                if norm_split not in ("train", "val"):
                    h = int(hashlib.md5(clean_stem.encode("utf-8")).hexdigest(), 16)
                    norm_split = "train" if (h % 100 < 85) else "val"

                dest_images_dir = self.images_dir_for_split(norm_split)
                dest_labels_dir = self.labels_dir_for_split(norm_split)

                resolved_img_path, dest_img_name = self._save_or_copy_image(
                    clean_stem=clean_stem,
                    norm_split=norm_split,
                    image_path=s.get("image_path"),
                    image_bytes=s.get("image_bytes"),
                    dest_images_dir=dest_images_dir,
                )

                if resolved_img_path is None and dest_img_name is None:
                    continue

                lines, valid_boxes, classes_present = parse_and_format_yolo_boxes(
                    s.get("trusted_boxes") or s.get("trusted")
                )
                txt_content = "\n".join(lines) + ("\n" if lines else "")
                dest_label_path = dest_labels_dir / f"{clean_stem}.txt"
                atomic_write_text(dest_label_path, txt_content, encoding="utf-8")

                timestamp = float(s.get("approved_at", time.time()))
                stems_map = self._manifest.setdefault("stems", {})
                stems_map[clean_stem] = {
                    "stem": clean_stem,
                    "split": norm_split,
                    "image_file": dest_img_name,
                    "label_file": f"{clean_stem}.txt",
                    "box_count": len(valid_boxes),
                    "classes": sorted(list(classes_present)),
                    "approved_at": timestamp,
                    "source_image": str(resolved_img_path) if resolved_img_path else "",
                }
                self._approved_stems.add(clean_stem)
                success_count += 1

            if success_count > 0:
                self._recalculate_manifest_stats_and_save()

        return success_count

    # -------------------------------------------------------------------------
    # Fast In-Memory Quarantine & Query API
    # -------------------------------------------------------------------------

    def is_approved(self, stem: str) -> bool:
        """
        Checks if stem is recorded as approved so it NEVER re-enters the pending
        review queue on dialog open/close or disk scan.
        """
        if not stem:
            return False
        clean_stem = Path(stem).stem.strip()
        with self._lock:
            return clean_stem in self._approved_stems

    def get_approved_stems(self) -> Set[str]:
        """Returns a copy of the set of all approved stems."""
        with self._lock:
            return set(self._approved_stems)

    def get_sample_manifest(self, stem: str) -> Optional[Dict[str, Any]]:
        """Return the manifest entry for a specific stem, or None if not found."""
        clean_stem = Path(stem).stem.strip()
        with self._lock:
            entry = self._manifest.get("stems", {}).get(clean_stem)
            return dict(entry) if entry else None

    def remove_approved_sample(self, stem: str) -> bool:
        """
        Removes an approved sample from the pool (deletes images and labels from disk
        and clears from manifest).
        """
        if not stem:
            return False
        clean_stem = Path(stem).stem.strip()

        with self._lock:
            if clean_stem not in self._approved_stems:
                return False

            stems_map = self._manifest.get("stems", {})
            entry = stems_map.pop(clean_stem, {})
            self._approved_stems.discard(clean_stem)

            split = entry.get("split", "train")
            dest_images_dir = self.images_dir_for_split(split)
            dest_labels_dir = self.labels_dir_for_split(split)

            # Clean disk files
            for ext in (".jpg", ".png", ".jpeg"):
                img_f = dest_images_dir / f"{clean_stem}{ext}"
                if img_f.is_file():
                    try:
                        img_f.unlink()
                    except Exception:
                        pass
            lbl_f = dest_labels_dir / f"{clean_stem}.txt"
            if lbl_f.is_file():
                try:
                    lbl_f.unlink()
                except Exception:
                    pass

            self._recalculate_manifest_stats_and_save()
            return True

    # -------------------------------------------------------------------------
    # Telemetry & Statistics
    # -------------------------------------------------------------------------

    def get_approved_stats(self) -> Dict[str, Any]:
        """
        Returns telemetry dictionary:
          {"total_approved_images": N, "train_count": M, "val_count": K, "total_boxes": B, ...}
        """
        with self._lock:
            return {
                "total_approved_images": int(self._manifest.get("total_approved_images", 0)),
                "train_count": int(self._manifest.get("train_count", 0)),
                "val_count": int(self._manifest.get("val_count", 0)),
                "total_boxes": int(self._manifest.get("total_boxes", 0)),
                "classes": dict(self._manifest.get("classes", {})),
                "last_updated": float(self._manifest.get("updated_at", 0.0)),
                "pool_root": str(self.pool_root),
                "data_yaml_path": str(self.data_yaml_path),
            }

    def _recalculate_manifest_stats_and_save(self) -> None:
        """Internal helper to calculate summary stats and write manifest atomically."""
        stems_map = self._manifest.get("stems", {})
        total_images = len(stems_map)
        train_count = 0
        val_count = 0
        total_boxes = 0
        class_counts: Dict[str, int] = {}

        for entry in stems_map.values():
            split = entry.get("split", "train")
            if split == "val":
                val_count += 1
            else:
                train_count += 1

            boxes = entry.get("box_count", 0)
            total_boxes += boxes

            for cid in entry.get("classes", []):
                s_cid = str(cid)
                class_counts[s_cid] = class_counts.get(s_cid, 0) + 1

        self._manifest["version"] = "1.0"
        self._manifest["updated_at"] = time.time()
        self._manifest["total_approved_images"] = total_images
        self._manifest["train_count"] = train_count
        self._manifest["val_count"] = val_count
        self._manifest["total_boxes"] = total_boxes
        self._manifest["classes"] = class_counts

        manifest_json = json.dumps(self._manifest, indent=2, ensure_ascii=False)
        atomic_write_text(self.manifest_path, manifest_json, encoding="utf-8")

    # -------------------------------------------------------------------------
    # Disk Re-Synchronization
    # -------------------------------------------------------------------------

    def sync_from_disk(self) -> Dict[str, Any]:
        """
        Reconciles in-memory index and `approved_manifest.json` directly from the
        image and label files existing in `dataset/approved/{images,labels}`.
        """
        with self._lock:
            self.ensure_pool_structure()
            stems_map: Dict[str, Any] = {}
            approved_set: Set[str] = set()

            for split, img_dir, lbl_dir in [
                ("train", self.images_train_dir, self.labels_train_dir),
                ("val", self.images_val_dir, self.labels_val_dir),
            ]:
                if not img_dir.is_dir():
                    continue

                for img_file in img_dir.iterdir():
                    if img_file.suffix.lower() not in (".jpg", ".png", ".jpeg"):
                        continue
                    stem = img_file.stem
                    approved_set.add(stem)

                    lbl_file = lbl_dir / f"{stem}.txt"
                    boxes_count = 0
                    classes_in_file: Set[int] = set()
                    if lbl_file.is_file():
                        try:
                            lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                            for line in lines:
                                norm = normalize_box(line)
                                if norm is not None:
                                    boxes_count += 1
                                    classes_in_file.add(norm[0])
                        except Exception:
                            pass

                    stems_map[stem] = {
                        "stem": stem,
                        "split": split,
                        "image_file": img_file.name,
                        "label_file": f"{stem}.txt",
                        "box_count": boxes_count,
                        "classes": sorted(list(classes_in_file)),
                        "approved_at": img_file.stat().st_mtime,
                        "source_image": str(img_file),
                    }

            self._manifest["stems"] = stems_map
            self._approved_stems = approved_set
            self._recalculate_manifest_stats_and_save()
            return self.get_approved_stats()

    # -------------------------------------------------------------------------
    # YOLO data.yaml Generation (A07)
    # -------------------------------------------------------------------------

    def generate_approved_data_yaml(
        self,
        target_path: Optional[Union[str, Path]] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> str:
        """
        Generates valid Ultralytics YOLO `data.yaml` pointing to
        `dataset/approved/images/train` and `dataset/approved/images/val`
        with canonical 22 classes.

        Args:
            target_path: Destination path for data.yaml. Defaults to `dataset/approved/data_approved.yaml`.
            class_names: Optional class names dictionary. Defaults to CORE_22_CANONICAL_NAMES (22 classes).

        Returns:
            str: Posix string path to the generated data.yaml.
        """
        with self._lock:
            if target_path is not None:
                out_path = Path(target_path)
                if not out_path.is_absolute():
                    out_path = (self.project_root / out_path).resolve()
            else:
                out_path = self.data_yaml_path

            classes = class_names if class_names is not None else CORE_22_CANONICAL_NAMES

            # Ultralytics requires forward slashes on Windows
            pool_posix = self.pool_root.as_posix()

            lines: List[str] = [
                "# Ultralytics YOLOv8 / YOLOv11 Dataset Configuration - Approved Dataset Pool",
                "# Division 1 (A04, A07) — Isolated Human-Approved Ground Truth Training Set",
                f"path: {pool_posix}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(classes)}",
                "names:",
            ]

            for cid in sorted(classes.keys()):
                lines.append(f"  {int(cid)}: {classes[cid]}")

            content = "\n".join(lines) + "\n"
            atomic_write_text(out_path, content, encoding="utf-8")
            logger.info("Generated approved data.yaml at: %s (%d classes)", out_path, len(classes))
            return out_path.as_posix()

    # -------------------------------------------------------------------------
    # High-Level Preparation Pipeline (A07)
    # -------------------------------------------------------------------------

    def prepare_approved_training_dataset(
        self,
        source_dataset_dir: Optional[Union[str, Path]] = None,
        val_ratio: float = 0.15,
        force_rebalance: bool = False,
    ) -> Dict[str, Any]:
        """
        Fulfills A07: Scans source dataset review files for all `human_approved=True`
        or `done="ok"` samples, migrates/syncs them to `dataset/approved/`, automatically
        balances train/val splits (85%/15%), and produces `data_approved.yaml`.

        Args:
            source_dataset_dir: Root dataset folder (defaults to `project_root / dataset`).
            val_ratio: Proportion of samples allocated to validation (default 0.15 = 15%).
            force_rebalance: If True, re-assigns train/val according to val_ratio for all items.

        Returns:
            Dict[str, Any]: Telemetry summary of the approved training pool.
        """
        with self._lock:
            src_dir = Path(source_dataset_dir).resolve() if source_dataset_dir else self.project_root / "dataset"
            review_dir = src_dir / "review"

            candidates: List[Dict[str, Any]] = []

            if review_dir.is_dir():
                for json_file in review_dir.rglob("*.json"):
                    if json_file.name in ("trusted_seed.json", "manifest_stats.json"):
                        continue
                    try:
                        payload = json.loads(json_file.read_text(encoding="utf-8"))
                        if not isinstance(payload, dict):
                            continue

                        # Check human approval flags
                        is_ok = (
                            payload.get("human_approved") is True
                            or payload.get("done") == "ok"
                            or payload.get("approved") is True
                        )
                        if not is_ok:
                            continue

                        stem = payload.get("stem") or json_file.stem
                        trusted = payload.get("trusted", [])
                        img_path = payload.get("image_path")
                        split = payload.get("split", "train")

                        candidates.append({
                            "stem": stem,
                            "split": split,
                            "image_path": img_path,
                            "trusted_boxes": trusted,
                            "approved_at": payload.get("approved_at", json_file.stat().st_mtime),
                        })
                    except Exception as ex:
                        logger.debug("Error parsing review json %s: %s", json_file, ex)

            # Sort deterministically by stem
            candidates.sort(key=lambda x: str(x["stem"]))

            # Rebalance train/val split if requested or needed
            if force_rebalance and candidates:
                for idx, c in enumerate(candidates):
                    # Deterministic balanced distribution
                    c["split"] = "val" if (idx % int(1.0 / max(0.01, val_ratio)) == 0) else "train"

            # Add candidates to pool
            self.bulk_add_approved_samples(candidates)

            # Generate YOLO configuration
            self.generate_approved_data_yaml()

            return self.get_approved_stats()


# =============================================================================
# Global Singleton & Top-Level Module API
# =============================================================================

_GLOBAL_POOL: Optional[ApprovedDatasetPool] = None
_GLOBAL_LOCK = threading.Lock()


def get_approved_pool(
    project_root: Optional[Union[str, Path]] = None,
    pool_root: Optional[Union[str, Path]] = None,
) -> ApprovedDatasetPool:
    """Returns or initializes the thread-safe global `ApprovedDatasetPool` singleton."""
    global _GLOBAL_POOL
    with _GLOBAL_LOCK:
        if _GLOBAL_POOL is None:
            _GLOBAL_POOL = ApprovedDatasetPool(pool_root=pool_root, project_root=project_root)
        return _GLOBAL_POOL


def is_approved(stem: str) -> bool:
    """Check if stem is already in the approved dataset pool."""
    return get_approved_pool().is_approved(stem)


def get_approved_stems() -> Set[str]:
    """Return all stems currently in the approved dataset pool."""
    return get_approved_pool().get_approved_stems()


def add_approved_sample(
    stem: str,
    split: str = "train",
    image_path: Optional[Union[str, Path, Any]] = None,
    trusted_boxes: Optional[Sequence[Any]] = None,
    image_bytes: Optional[bytes] = None,
) -> bool:
    """Add a verified sample to the approved dataset pool."""
    return get_approved_pool().add_approved_sample(
        stem=stem,
        split=split,
        image_path=image_path,
        trusted_boxes=trusted_boxes,
        image_bytes=image_bytes,
    )


def remove_approved_sample(stem: str) -> bool:
    """Remove a sample from the approved dataset pool."""
    return get_approved_pool().remove_sample(stem)


def generate_approved_data_yaml(target_path: str = "dataset/approved/data_approved.yaml") -> str:
    """Generate canonical 22-class data_approved.yaml."""
    return get_approved_pool().generate_approved_data_yaml(target_path=target_path)


def get_approved_stats() -> Dict[str, Any]:
    """Return summary telemetry of the approved pool."""
    return get_approved_pool().get_approved_stats()


def prepare_approved_training_dataset(
    source_dataset_dir: Optional[Union[str, Path]] = None,
    val_ratio: float = 0.15,
    force_rebalance: bool = False,
) -> Dict[str, Any]:
    """Execute the preparation pipeline for approved training dataset (A07)."""
    return get_approved_pool().prepare_approved_training_dataset(
        source_dataset_dir=source_dataset_dir,
        val_ratio=val_ratio,
        force_rebalance=force_rebalance,
    )
