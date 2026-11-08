# -*- coding: utf-8 -*-
"""
dataset_factory/label_cache.py
==============================
Delta Relabeling Cache for YOLO dataset auto-labeling and review workflows.

Tracks:
  (file_path, file_mtime_ns, file_size, exemplar_revision)
Persists in:
  dataset/.label_cache.json

Key Capabilities:
1. Fast Delta Skipping:
   Skips unchanged images in sub-millisecond time when disk stat (mtime_ns, size)
   and exemplar revision match the cache.
2. Zero-Box Safety Guarantee:
   NEVER skips 0-box (unlabeled) images when user requests relabeling. Images with
   0 boxes are always candidates for re-evaluation against newly learned exemplars
   or adjusted detection thresholds.
3. Candidate Crop Delta Relabeling:
   When an exemplar is updated, only re-evaluates relevant candidate crops rather
   than re-running the entire heavy visual pipeline (edge detection, multi-scale
   contours, full-frame template scanning) on unchanged images.
4. Concurrency & Persistence Integrity:
   Thread-safe with atomic file writes (JSON temporary file + atomic rename)
   and backwards-compatible aliases for legacy format consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import cv2
import numpy as np

logger = logging.getLogger("Gatherion.LabelCache")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = PROJECT_ROOT / "dataset" / ".label_cache.json"
DEFAULT_EXEMPLAR_PATH = PROJECT_ROOT / "dataset" / "learned_exemplars.json"


def imread_unicode(path: Union[str, Path], flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Safely decode an image file from disk supporting Windows Unicode paths."""
    try:
        p = Path(path)
        if not p.is_file():
            return None
        data = np.fromfile(str(p), dtype=np.uint8)
        if data is None or data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception as exc:
        logger.debug("imread_unicode failed for %s: %s", path, exc)
        return None


def get_current_exemplar_revision(
    exemplar_path: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
) -> int:
    """
    Returns current revision integer for learned exemplars (nanosecond mtime).
    Returns 0 if the exemplar file does not exist or cannot be read.
    """
    try:
        if exemplar_path is not None:
            ex_p = Path(exemplar_path)
        elif dataset_dir is not None:
            ex_p = Path(dataset_dir) / "learned_exemplars.json"
        else:
            ex_p = DEFAULT_EXEMPLAR_PATH

        if ex_p.is_file():
            return int(ex_p.stat().st_mtime_ns)
    except Exception as exc:
        logger.debug("Failed to read exemplar revision from %s: %s", exemplar_path, exc)
    return 0


@dataclass
class LabelCacheEntry:
    """
    Cache metadata entry for an image file.

    Tracks:
      (file_path, file_mtime_ns, file_size, exemplar_revision)
    plus box count, cached candidate crops, and timestamp.
    """

    file_path: str
    file_mtime_ns: int
    file_size: int
    exemplar_revision: int
    box_count: int = 0
    candidate_crops: List[Dict[str, Any]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary with backwards-compatible alias keys."""
        return {
            "file_path": str(self.file_path),
            "file_mtime_ns": int(self.file_mtime_ns),
            "file_size": int(self.file_size),
            "exemplar_revision": int(self.exemplar_revision),
            # Legacy compatibility aliases (used by dataset_curator / earlier code):
            "mtime_ns": int(self.file_mtime_ns),
            "size": int(self.file_size),
            "exemplar_rev": int(self.exemplar_revision),
            "box_count": int(self.box_count),
            "candidate_crops": list(self.candidate_crops),
            "updated_at": float(self.updated_at),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], default_path: str = "") -> "LabelCacheEntry":
        """Instantiate entry from a dictionary, supporting both new and legacy keys."""
        path = str(data.get("file_path") or default_path or "")
        mtime_ns = int(data.get("file_mtime_ns", data.get("mtime_ns", 0)))
        size = int(data.get("file_size", data.get("size", 0)))
        ex_rev = int(data.get("exemplar_revision", data.get("exemplar_rev", 0)))
        box_count = int(data.get("box_count", 0))
        candidate_crops = list(data.get("candidate_crops", []))
        updated_at = float(data.get("updated_at", time.time()))
        metadata = dict(data.get("metadata", {}))

        return cls(
            file_path=path,
            file_mtime_ns=mtime_ns,
            file_size=size,
            exemplar_revision=ex_rev,
            box_count=box_count,
            candidate_crops=candidate_crops,
            updated_at=updated_at,
            metadata=metadata,
        )

    # Dictionary-like interoperability
    def __getitem__(self, key: str) -> Any:
        aliases = {
            "mtime_ns": "file_mtime_ns",
            "size": "file_size",
            "exemplar_rev": "exemplar_revision",
        }
        resolved = aliases.get(key, key)
        if hasattr(self, resolved):
            return getattr(self, resolved)
        if key in self.metadata:
            return self.metadata[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        aliases = {
            "mtime_ns": "file_mtime_ns",
            "size": "file_size",
            "exemplar_rev": "exemplar_revision",
        }
        resolved = aliases.get(key, key)
        return hasattr(self, resolved) or key in self.metadata


class LabelCache:
    """
    Delta Relabeling Cache (`LabelCache`).

    Tracks:
      (file_path, file_mtime_ns, file_size, exemplar_revision)
    Persistence in:
      dataset/.label_cache.json
    """

    def __init__(
        self,
        cache_path: Optional[Union[str, Path]] = None,
        dataset_dir: Optional[Union[str, Path]] = None,
        auto_save: bool = True,
    ) -> None:
        if cache_path is not None:
            self.cache_path = Path(cache_path).resolve()
        elif dataset_dir is not None:
            self.cache_path = (Path(dataset_dir) / ".label_cache.json").resolve()
        else:
            self.cache_path = DEFAULT_CACHE_PATH.resolve()

        self.dataset_dir = Path(dataset_dir).resolve() if dataset_dir else self.cache_path.parent
        self.auto_save = auto_save

        self._lock = threading.RLock()
        self._entries: Dict[str, LabelCacheEntry] = {}
        self._dirty: bool = False

        self.load()

    def load(self) -> bool:
        """Load cache entries from JSON file."""
        with self._lock:
            if not self.cache_path.is_file():
                return False
            try:
                raw_text = self.cache_path.read_text(encoding="utf-8")
                if not raw_text.strip():
                    return False
                data = json.loads(raw_text)
                if not isinstance(data, dict):
                    return False

                entries: Dict[str, LabelCacheEntry] = {}
                for key, val in data.items():
                    if isinstance(val, dict):
                        entry = LabelCacheEntry.from_dict(val, default_path=key)
                        entries[key] = entry

                self._entries = entries
                self._dirty = False
                logger.debug(
                    "Loaded %d label cache entries from %s", len(self._entries), self.cache_path
                )
                return True
            except Exception as exc:
                logger.warning("Failed to load label cache from %s: %s", self.cache_path, exc)
                return False

    def save(self, force: bool = False) -> bool:
        """
        Atomically persist cache entries to JSON on disk.
        Uses a temporary file and replace to guarantee no corrupted writes.
        """
        with self._lock:
            if not self._dirty and not force:
                return True
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                payload: Dict[str, Any] = {
                    key: entry.to_dict() for key, entry in self._entries.items()
                }

                tmp_path = self.cache_path.with_suffix(".json.tmp")
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )

                if tmp_path.exists():
                    tmp_path.replace(self.cache_path)

                self._dirty = False
                logger.debug("Saved %d label cache entries to %s", len(self._entries), self.cache_path)
                return True
            except Exception as exc:
                logger.error("Failed to save label cache to %s: %s", self.cache_path, exc)
                return False

    def _normalize_key(self, file_path: Union[str, Path]) -> str:
        """Return the normalized key for indexing (filename)."""
        return Path(file_path).name

    def _resolve_existing_path(self, file_path: Union[str, Path]) -> Optional[Path]:
        """Resolve a file path to an existing Path object on disk if possible."""
        p = Path(file_path)
        if p.is_file():
            return p

        # Check relative to dataset_dir or common subdirectories
        candidates = [
            self.dataset_dir / p,
            self.dataset_dir / "images" / "train" / p.name,
            self.dataset_dir / "images" / "val" / p.name,
            PROJECT_ROOT / p,
        ]
        for c in candidates:
            if c.is_file():
                return c

        return None

    def get_entry(self, file_path: Union[str, Path]) -> Optional[LabelCacheEntry]:
        """Lookup cache entry by file path, key, or filename."""
        key = self._normalize_key(file_path)
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            # Try raw str key
            raw_key = str(file_path)
            if raw_key in self._entries:
                return self._entries[raw_key]
            return None

    def has_entry(self, file_path: Union[str, Path]) -> bool:
        """Check if file is present in cache."""
        return self.get_entry(file_path) is not None

    def should_relabel(
        self,
        file_path: Union[str, Path],
        exemplar_revision: int,
        force: bool = False,
        allow_zero_box_skip: bool = False,
    ) -> bool:
        """
        Determine whether an image should be relabeled.

        Returns True if:
          - `force` is True
          - file is not in cache
          - file has 0 boxes in cache (never skip unlabeled/0-box images when relabeling is requested!)
          - exemplar_revision does not match the cached revision
          - file mtime or size on disk has changed

        Returns False ONLY if:
          - file stat (mtime_ns, size) matches cache,
          - exemplar_revision matches cache exactly,
          - AND file has at least one box recorded (box_count > 0).
        """
        if force:
            return True

        entry = self.get_entry(file_path)
        if entry is None:
            return True

        # CRITICAL MANDATE: Never skip 0-box (unlabeled) images when user requests relabeling!
        if not allow_zero_box_skip and entry.box_count <= 0:
            return True

        # Check exemplar revision
        if entry.exemplar_revision != int(exemplar_revision):
            return True

        # Check disk stat if file exists
        path_obj = self._resolve_existing_path(file_path)
        if path_obj is not None and path_obj.is_file():
            try:
                st = path_obj.stat()
                disk_mtime = int(st.st_mtime_ns)
                disk_size = int(st.st_size)
                if disk_mtime != entry.file_mtime_ns or disk_size != entry.file_size:
                    return True
            except Exception:
                return True
        elif not Path(file_path).is_file() and entry.file_mtime_ns > 0:
            # File previously had disk mtime recorded, but is now gone from disk
            return False

        # Unchanged on disk, exemplar revision matches, and has > 0 boxes -> skip
        return False

    def record_label(
        self,
        file_path: Union[str, Path],
        exemplar_revision: int,
        box_count: Optional[int] = None,
        boxes: Optional[Sequence[Any]] = None,
        candidate_crops: Optional[List[Dict[str, Any]]] = None,
        file_mtime_ns: Optional[int] = None,
        file_size: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        auto_save: Optional[bool] = None,
    ) -> LabelCacheEntry:
        """
        Record labeling results for a file in the cache.

        Tracks:
          - file_path
          - file_mtime_ns
          - file_size
          - exemplar_revision
          - box_count
          - candidate_crops
        """
        path_obj = self._resolve_existing_path(file_path)
        actual_path = path_obj if path_obj is not None else Path(file_path)
        key = self._normalize_key(actual_path)

        # Resolve mtime and size from disk if not explicitly given
        if file_mtime_ns is None or file_size is None:
            try:
                st = actual_path.stat()
                if file_mtime_ns is None:
                    file_mtime_ns = int(st.st_mtime_ns)
                if file_size is None:
                    file_size = int(st.st_size)
            except Exception:
                file_mtime_ns = int(file_mtime_ns or 0)
                file_size = int(file_size or 0)

        # Resolve box count
        if box_count is None:
            if boxes is not None:
                box_count = len(boxes)
            else:
                box_count = 0

        with self._lock:
            existing = self._entries.get(key)
            preserved_crops = (
                list(candidate_crops)
                if candidate_crops is not None
                else (list(existing.candidate_crops) if existing else [])
            )
            merged_meta = dict(existing.metadata if existing else {})
            if metadata:
                merged_meta.update(metadata)

            entry = LabelCacheEntry(
                file_path=str(actual_path),
                file_mtime_ns=int(file_mtime_ns),
                file_size=int(file_size),
                exemplar_revision=int(exemplar_revision),
                box_count=int(box_count),
                candidate_crops=preserved_crops,
                updated_at=time.time(),
                metadata=merged_meta,
            )

            self._entries[key] = entry
            self._dirty = True

            should_save = self.auto_save if auto_save is None else auto_save
            if should_save:
                self.save()

            return entry

    def cache_candidate_crops(
        self,
        file_path: Union[str, Path],
        crops: Sequence[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
        exemplar_revision: Optional[int] = None,
        auto_save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Cache candidate proposals / crops with visual descriptors for fast delta re-evaluation.

        If `frame` is provided, automatically extracts visual features (color histogram,
        aspect ratio, edge texture) for candidates that do not already have them.
        """
        from dataset_factory.exemplar_memory import extract_box_crop, extract_crop_features

        key = self._normalize_key(file_path)
        stored_crops: List[Dict[str, Any]] = []

        for c in crops:
            c_dict = dict(c)
            # If feature descriptor missing and frame available, extract visual features
            if "feature" not in c_dict and frame is not None and frame.size > 0:
                crop_patch = extract_box_crop(frame, c_dict)
                if crop_patch is not None and crop_patch.size > 0:
                    feat = extract_crop_features(crop_patch)
                    if feat:
                        c_dict["feature"] = feat
            stored_crops.append(c_dict)

        with self._lock:
            entry = self.get_entry(file_path)
            if entry is not None:
                entry.candidate_crops = stored_crops
                entry.updated_at = time.time()
                if exemplar_revision is not None:
                    entry.exemplar_revision = int(exemplar_revision)
                self._dirty = True
                if auto_save and self.auto_save:
                    self.save()
            else:
                # Record initial entry
                rev = int(exemplar_revision or 0)
                self.record_label(
                    file_path=file_path,
                    exemplar_revision=rev,
                    box_count=len(stored_crops),
                    candidate_crops=stored_crops,
                    auto_save=auto_save,
                )

        return stored_crops

    def can_delta_relabel(
        self,
        file_path: Union[str, Path],
        exemplar_revision: int,
    ) -> bool:
        """
        Check if an image can use fast candidate-crop delta relabeling instead
        of running the entire heavy visual pipeline.

        Returns True if:
          1. Cache entry exists and has candidate crops stored (len > 0).
          2. Exemplar revision has changed (needs re-evaluation against new exemplars).
          3. If image exists on disk, disk mtime/size match cache (pixels unchanged).
        """
        entry = self.get_entry(file_path)
        if entry is None or not entry.candidate_crops:
            return False

        # Only delta-relabel if exemplar revision has updated
        if entry.exemplar_revision == int(exemplar_revision):
            return False

        path_obj = self._resolve_existing_path(file_path)
        if path_obj is not None and path_obj.is_file():
            try:
                st = path_obj.stat()
                if int(st.st_mtime_ns) != entry.file_mtime_ns or int(st.st_size) != entry.file_size:
                    return False
            except Exception:
                return False

        return True

    def re_evaluate_candidate_crops(
        self,
        file_path: Union[str, Path],
        exemplar_memory: Any,
        frame: Optional[np.ndarray] = None,
        exemplar_revision: Optional[int] = None,
        conf_floor: float = 0.45,
        emin_threshold: float = 0.85,
        auto_save: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Fast Delta Relabeling:
        Only re-evaluates cached candidate crops against updated ExemplarMemory,
        skipping full-frame visual segmentation and heavy filters!

        - Immunity Shield: Hand-labeled / human-approved boxes are preserved 100%.
        - Negative Exemplar Suppression: Dropped if matching learned negative exemplars.
        - Positive Exemplar Reinforcement & Class Auto-Repair: Confidence boosted or
          class corrected if matching learned positive exemplars.
        - Updates cache entry with surviving boxes and new exemplar revision.
        """
        from dataset_factory.exemplar_memory import (
            compute_similarity,
            extract_box_crop,
            extract_crop_features,
        )

        entry = self.get_entry(file_path)
        if entry is None or not entry.candidate_crops:
            return []

        # If frame not provided, attempt to load only if candidate lacks precomputed feature
        loaded_frame: Optional[np.ndarray] = frame
        needs_frame = any("feature" not in c for c in entry.candidate_crops)
        if needs_frame and loaded_frame is None:
            resolved_p = self._resolve_existing_path(file_path)
            if resolved_p is not None:
                loaded_frame = imread_unicode(resolved_p)

        surviving: List[Dict[str, Any]] = []

        for cand in entry.candidate_crops:
            c = dict(cand)

            # Layer 0: Immunity Shield for human approved / hand labeled boxes
            is_hand = bool(c.get("hand_label") or c.get("human_approved") or c.get("approved"))
            if is_hand:
                c["trusted"] = True
                c["weak"] = False
                c["confidence"] = max(0.95, float(c.get("confidence", 0.95)))
                surviving.append(c)
                continue

            # Obtain visual feature
            feat = c.get("feature")
            if feat is None and loaded_frame is not None and loaded_frame.size > 0:
                crop = extract_box_crop(loaded_frame, c)
                if crop is not None and crop.size > 0:
                    feat = extract_crop_features(crop)
                    if feat:
                        c["feature"] = feat

            cid = int(c.get("class_id", 0))

            # 1. Negative Exemplar Check (Suppression)
            suppressed = False
            if hasattr(exemplar_memory, "match_negative"):
                if feat is not None and hasattr(exemplar_memory, "_negative"):
                    # Fast vector / dict similarity
                    best_neg_sim = 0.0
                    for neg_ex in getattr(exemplar_memory, "_negative", []):
                        if cid is not None and int(neg_ex.get("class_id", -1)) != cid:
                            sim = compute_similarity(feat, neg_ex) * 0.90
                        else:
                            sim = compute_similarity(feat, neg_ex)
                        if sim > best_neg_sim:
                            best_neg_sim = sim
                    suppress_thresh = getattr(exemplar_memory, "suppress_neg_threshold", 0.82)
                    if best_neg_sim >= suppress_thresh:
                        suppressed = True
                elif loaded_frame is not None:
                    crop = extract_box_crop(loaded_frame, c)
                    if crop is not None and crop.size > 0:
                        neg_sim, _ = exemplar_memory.match_negative(crop, class_id=cid)
                        if neg_sim >= getattr(exemplar_memory, "suppress_neg_threshold", 0.82):
                            suppressed = True

            if suppressed:
                logger.debug("Delta relabel suppressed candidate crop in %s", file_path)
                continue

            # 2. Positive Exemplar Check (Reinforcement & Auto-Repair)
            if hasattr(exemplar_memory, "match_positive"):
                best_pos_sim = 0.0
                best_cid: Optional[int] = None
                best_cname: Optional[str] = None
                if feat is not None and hasattr(exemplar_memory, "_positive"):
                    for pos_ex in getattr(exemplar_memory, "_positive", []):
                        sim = compute_similarity(feat, pos_ex)
                        if sim > best_pos_sim:
                            best_pos_sim = sim
                            best_cid = int(pos_ex.get("class_id", 0))
                            best_cname = str(pos_ex.get("class_name", ""))
                elif loaded_frame is not None:
                    crop = extract_box_crop(loaded_frame, c)
                    if crop is not None and crop.size > 0:
                        best_pos_sim, best_cid, best_cname, _ = exemplar_memory.match_positive(crop)

                reinforce_thresh = getattr(exemplar_memory, "reinforce_pos_threshold", 0.70)
                reassign_thresh = getattr(exemplar_memory, "reassign_pos_threshold", 0.78)

                if best_pos_sim >= reinforce_thresh and best_cid is not None:
                    if best_cid == cid:
                        old_conf = float(c.get("confidence", 0.80))
                        boost = (1.0 - old_conf) * (best_pos_sim - 0.50) * 0.70
                        c["confidence"] = round(float(np.clip(old_conf + boost, old_conf, 0.99)), 4)
                        c["exemplar_reinforced"] = True
                        c["exemplar_sim"] = best_pos_sim
                    elif best_pos_sim >= reassign_thresh:
                        c["class_id"] = best_cid
                        c["class_name"] = best_cname or f"class_{best_cid}"
                        c["exemplar_corrected"] = True
                        c["exemplar_sim"] = best_pos_sim
                        old_c = float(c.get("confidence", 0.80))
                        c["confidence"] = round(
                            float(np.clip(max(old_c, best_pos_sim * 0.95), 0.85, 0.99)), 4
                        )

            # Check confidence floor
            conf = float(c.get("confidence", 0.0))
            if conf < conf_floor:
                continue

            # Assign trust / weak tags
            is_emin = (
                c.get("hand_label") is True
                or c.get("human_approved") is True
                or c.get("exemplar_match") is True
                or conf >= emin_threshold
            )
            c["trusted"] = bool(is_emin)
            c["weak"] = not bool(is_emin)
            surviving.append(c)

        # Update cache entry
        with self._lock:
            entry.candidate_crops = surviving
            entry.box_count = len(surviving)
            if exemplar_revision is not None:
                entry.exemplar_revision = int(exemplar_revision)
            entry.updated_at = time.time()
            self._dirty = True
            if auto_save and self.auto_save:
                self.save()

        return surviving

    def remove_entry(self, file_path: Union[str, Path], auto_save: bool = True) -> bool:
        """Remove a file from the cache."""
        key = self._normalize_key(file_path)
        with self._lock:
            removed = False
            if key in self._entries:
                del self._entries[key]
                removed = True
            raw_key = str(file_path)
            if raw_key in self._entries:
                del self._entries[raw_key]
                removed = True
            if removed:
                self._dirty = True
                if auto_save and self.auto_save:
                    self.save()
            return removed

    def clear(self, auto_save: bool = True) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._entries.clear()
            self._dirty = True
            if auto_save and self.auto_save:
                self.save()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache health and coverage metrics."""
        with self._lock:
            total = len(self._entries)
            zero_box = sum(1 for e in self._entries.values() if e.box_count <= 0)
            with_crops = sum(1 for e in self._entries.values() if len(e.candidate_crops) > 0)
            total_boxes = sum(e.box_count for e in self._entries.values())
            return {
                "total_entries": total,
                "zero_box_entries": zero_box,
                "labeled_entries": total - zero_box,
                "entries_with_cached_crops": with_crops,
                "total_boxes": total_boxes,
                "cache_path": str(self.cache_path),
                "is_dirty": self._dirty,
            }

    def to_dict(self) -> Dict[str, Any]:
        """Export cache contents as a serializable dictionary."""
        with self._lock:
            return {k: entry.to_dict() for k, entry in self._entries.items()}

    # Dict-like magic methods for complete backward compatibility
    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def __contains__(self, key: Any) -> bool:
        return self.has_entry(key)

    def __getitem__(self, key: str) -> LabelCacheEntry:
        entry = self.get_entry(key)
        if entry is None:
            raise KeyError(key)
        return entry

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            if isinstance(value, LabelCacheEntry):
                self._entries[key] = value
            elif isinstance(value, dict):
                self._entries[key] = LabelCacheEntry.from_dict(value, default_path=key)
            else:
                raise TypeError(f"Value must be LabelCacheEntry or dict, got {type(value)}")
            self._dirty = True
            if self.auto_save:
                self.save()

    def __delitem__(self, key: str) -> None:
        if not self.remove_entry(key):
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        entry = self.get_entry(key)
        return entry if entry is not None else default

    def keys(self) -> Iterator[str]:
        with self._lock:
            return iter(list(self._entries.keys()))

    def values(self) -> Iterator[LabelCacheEntry]:
        with self._lock:
            return iter(list(self._entries.values()))

    def items(self) -> Iterator[Tuple[str, LabelCacheEntry]]:
        with self._lock:
            return iter(list(self._entries.items()))

    # Context manager
    def __enter__(self) -> "LabelCache":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._dirty:
            self.save()


_GLOBAL_LABEL_CACHE: Optional[LabelCache] = None
_GLOBAL_LOCK = threading.RLock()


def get_label_cache(
    cache_path: Optional[Union[str, Path]] = None,
    dataset_dir: Optional[Union[str, Path]] = None,
) -> LabelCache:
    """Get or create singleton LabelCache instance."""
    global _GLOBAL_LABEL_CACHE
    with _GLOBAL_LOCK:
        if _GLOBAL_LABEL_CACHE is None or (
            cache_path is not None and Path(cache_path).resolve() != _GLOBAL_LABEL_CACHE.cache_path
        ):
            _GLOBAL_LABEL_CACHE = LabelCache(cache_path=cache_path, dataset_dir=dataset_dir)
        return _GLOBAL_LABEL_CACHE


def plan_batch_relabel(
    image_paths: Sequence[Union[str, Path]],
    exemplar_revision: int,
    label_cache: Optional[LabelCache] = None,
    force: bool = False,
) -> Dict[str, List[Path]]:
    """
    Partition target image paths into execution plans:
      - 'skip': Unchanged images that can be skipped completely (< 0.01ms check).
      - 'delta': Images that have cached candidate crops and can be delta-re-evaluated.
      - 'full': Images that require full visual pipeline auto-labeling.

    GUARANTEE: 0-box (unlabeled) images are NEVER assigned to 'skip'.
    """
    cache = label_cache or get_label_cache()
    skipped: List[Path] = []
    delta: List[Path] = []
    full: List[Path] = []

    for p in image_paths:
        p_obj = Path(p)
        if not p_obj.is_file():
            continue

        if not force and not cache.should_relabel(p_obj, exemplar_revision):
            skipped.append(p_obj)
        elif cache.can_delta_relabel(p_obj, exemplar_revision):
            delta.append(p_obj)
        else:
            full.append(p_obj)

    return {
        "skip": skipped,
        "delta": delta,
        "full": full,
    }
