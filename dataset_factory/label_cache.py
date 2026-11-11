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
import tempfile
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


# F-2000: use the canonical unicode-safe decoder from lru_frame_cache
# instead of a local duplicate (same behaviour for valid images: ASCII
# fast-path via cv2.imread, np.fromfile + cv2.imdecode fallback for
# non-ASCII paths; None for missing/empty/undecodable files).
from dataset_factory.lru_frame_cache import imread_unicode


def _safe_class_id(value: Any, default: int = -1) -> int:
    """Best-effort int() for class ids; returns `default` for non-numeric values."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_confidence(value: Any, default: float = 0.0) -> float:
    """Best-effort float() for confidences; returns `default` for non-numeric
    (e.g. 'high' loaded from a disk cache) or non-finite values.

    S-2103: a single non-numeric confidence in a cached candidate crop used
    to raise ValueError and abort the whole file's delta relabel, even
    though only the class_id parse was guarded. Callers pass the same
    default they previously used with float(c.get("confidence", <default>)).
    """
    try:
        conf = float(value)
    except (TypeError, ValueError):
        logger.warning("Non-numeric confidence %r; using default %s", value, default)
        return default
    if conf != conf or conf in (float("inf"), float("-inf")):
        logger.warning("Non-finite confidence %r; using default %s", value, default)
        return default
    return conf


def _safe_int(value: Any, default: int = 0) -> int:
    """Best-effort int() for cache metadata; returns `default` for non-numeric
    values instead of raising.

    S-5601: LabelCacheEntry.from_dict used bare int()/float() conversions, so
    one corrupt value (e.g. file_mtime_ns='abc') raised and aborted the
    whole cache load. Same guarded pattern as _safe_confidence (S-2103).
    """
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        logger.warning("Non-numeric int value %r; using default %s", value, default)
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float() for cache metadata; returns `default` for
    non-numeric or non-finite values instead of raising (S-5601)."""
    try:
        val = float(value)
    except (TypeError, ValueError, OverflowError):
        logger.warning("Non-numeric float value %r; using default %s", value, default)
        return default
    if val != val or val in (float("inf"), float("-inf")):
        logger.warning("Non-finite float value %r; using default %s", value, default)
        return default
    return val


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
        """Instantiate entry from a dictionary, supporting both new and legacy keys.

        S-5601: every conversion is guarded (_safe_int/_safe_float, dict/list
        type checks) so a single corrupt value degrades to its default
        instead of raising and aborting the whole cache load in load().
        This method never raises for malformed field values.
        """
        if not isinstance(data, dict):
            logger.warning(
                "LabelCacheEntry.from_dict got non-dict data (%r); using defaults",
                type(data).__name__,
            )
            data = {}
        path = str(data.get("file_path") or default_path or "")
        mtime_ns = _safe_int(data.get("file_mtime_ns", data.get("mtime_ns", 0)), 0)
        size = _safe_int(data.get("file_size", data.get("size", 0)), 0)
        ex_rev = _safe_int(data.get("exemplar_revision", data.get("exemplar_rev", 0)), 0)
        box_count = _safe_int(data.get("box_count", 0), 0)
        raw_crops = data.get("candidate_crops", [])
        if isinstance(raw_crops, (list, tuple)):
            candidate_crops = [c for c in raw_crops if isinstance(c, dict)]
            if len(candidate_crops) != len(raw_crops):
                logger.warning(
                    "Dropped %d non-dict candidate crop(s) for %r",
                    len(raw_crops) - len(candidate_crops),
                    path,
                )
        else:
            logger.warning(
                "Non-list candidate_crops %r for %r; using empty list",
                raw_crops,
                path,
            )
            candidate_crops = []
        default_ts = time.time()
        updated_at = _safe_float(data.get("updated_at", default_ts), default_ts)
        raw_metadata = data.get("metadata", {})
        if isinstance(raw_metadata, dict):
            metadata = dict(raw_metadata)
        else:
            logger.warning(
                "Non-dict metadata %r for %r; using empty dict", raw_metadata, path
            )
            metadata = {}

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
        # Basename -> set of internal keys. The legacy bare-filename lookup
        # in get_entry() used to scan every entry (O(n) per cache miss);
        # the index keeps it O(1). It is keyed by BOTH the entry file
        # basename and the internal-key basename, mirroring the old
        # predicate (see _index_names_for), and is maintained at every
        # mutation site (load / record_label / remove_entry / clear /
        # __setitem__).
        self._by_basename: Dict[str, Set[str]] = {}
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
                    logger.warning("Label cache %s is not a JSON object; backing it up", self.cache_path)
                    self._backup_corrupt_cache()
                    return False

                entries: Dict[str, LabelCacheEntry] = {}
                skipped_malformed = 0
                for key, val in data.items():
                    if not isinstance(val, dict):
                        continue
                    # S-5601: one corrupt entry must not abort the whole
                    # load (the outer except would move the entire file
                    # aside as .corrupt-* and drop every delta entry).
                    # from_dict is guarded, but never let an unexpected
                    # error in a single entry lose the rest of the cache.
                    try:
                        entry = LabelCacheEntry.from_dict(val, default_path=key)
                    except Exception as exc:
                        # One malformed entry must not wipe the whole
                        # cache (previously the outer except backed up
                        # and dropped every entry). Skip it, keep the
                        # healthy ones, and mark dirty so the next save
                        # drops the bad entry (self-heal).
                        logger.warning(
                            "Skipping malformed label cache entry %r: %s", key, exc
                        )
                        skipped_malformed += 1
                        continue
                    # Re-key by resolved absolute path (old files are
                    # keyed by bare filename — see _normalize_key).
                    new_key = (
                        self._normalize_key(entry.file_path)
                        if entry.file_path
                        else self._normalize_key(key)
                    )
                    prev = entries.get(new_key)
                    if prev is None or entry.updated_at >= prev.updated_at:
                        entries[new_key] = entry

                self._entries = entries
                self._rebuild_basename_index()
                # _dirty=True only when entries were dropped, so the next
                # save persists the healed cache; a clean load stays clean.
                self._dirty = skipped_malformed > 0
                logger.debug(
                    "Loaded %d label cache entries from %s", len(self._entries), self.cache_path
                )
                return True
            except Exception as exc:
                logger.warning("Failed to load label cache from %s: %s", self.cache_path, exc)
                self._backup_corrupt_cache()
                return False

    # -- basename index (see __init__) -------------------------------------
    def _index_names_for(self, key: str, entry: Optional[LabelCacheEntry]) -> Set[str]:
        """Basenames under which (key, entry) is findable by get_entry().

        Mirrors the old linear-scan predicate exactly: an entry matched
        when the queried name equalled the entry file's basename OR the
        internal key's basename.
        """
        names: Set[str] = set()
        if entry is not None:
            base = Path(entry.file_path).name
            if base:
                names.add(base)
        key_base = Path(key).name
        if key_base:
            names.add(key_base)
        return names

    def _index_add(self, key: str, entry: LabelCacheEntry) -> None:
        for name in self._index_names_for(key, entry):
            self._by_basename.setdefault(name, set()).add(key)

    def _index_remove(self, key: str, entry: Optional[LabelCacheEntry]) -> None:
        for name in self._index_names_for(key, entry):
            bucket = self._by_basename.get(name)
            if bucket is not None:
                bucket.discard(key)
                if not bucket:
                    del self._by_basename[name]

    def _rebuild_basename_index(self) -> None:
        self._by_basename = {}
        for key, entry in self._entries.items():
            self._index_add(key, entry)

    def _backup_corrupt_cache(self) -> None:
        """
        Preserve a corrupt cache file instead of silently wiping history:
        move it aside as `<name>.corrupt-<timestamp>` so the entries can
        still be recovered. In-memory entries are left untouched; they are
        marked dirty (when non-empty) so the next save persists them.
        """
        try:
            if self.cache_path.is_file() and self.cache_path.stat().st_size > 0:
                backup = self.cache_path.with_name(
                    f"{self.cache_path.name}.corrupt-{int(time.time())}"
                )
                self.cache_path.replace(backup)
                logger.warning("Backed up corrupt label cache to %s", backup)
                if self._entries:
                    self._dirty = True
        except Exception as exc:  # never let the backup break loading
            logger.debug("Failed to back up corrupt label cache: %s", exc)

    def _persist_keys(self) -> Dict[str, str]:
        """
        Map internal keys to JSON keys: the bare filename when it is unique
        among entries (backward compatible with existing .label_cache.json
        files and their tests), the full resolved path when two entries
        share a filename (the F-0803 collision case).
        """
        counts: Dict[str, int] = {}
        for entry in self._entries.values():
            base = Path(entry.file_path).name
            if base:
                counts[base] = counts.get(base, 0) + 1
        out: Dict[str, str] = {}
        for key, entry in self._entries.items():
            base = Path(entry.file_path).name
            out[key] = base if base and counts.get(base, 0) == 1 else key
        return out

    def save(self, force: bool = False) -> bool:
        """
        Atomically persist cache entries to JSON on disk.

        Uses a uniquely-named temporary file in the same directory +
        os.replace, so two processes (or threads) saving the same cache
        can no longer interleave into one shared `<name>.json.tmp` file
        and publish a torn JSON. The temp file is fsynced before the
        rename, and the directory is fsynced best-effort afterwards, so
        a completed save() survives a crash; a failed save never leaves
        a partial cache behind (the temp file is removed).
        """
        with self._lock:
            if not self._dirty and not force:
                return True
            try:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                persist_keys = self._persist_keys()
                payload: Dict[str, Any] = {
                    persist_keys[key]: entry.to_dict()
                    for key, entry in self._entries.items()
                }
                text = json.dumps(payload, ensure_ascii=False, indent=2)

                # Preserve the existing cache file's mode (default 0644).
                try:
                    mode = self.cache_path.stat().st_mode & 0o777
                except OSError:
                    mode = 0o644

                fd, tmp_name = tempfile.mkstemp(
                    prefix=f".{self.cache_path.name}.",
                    suffix=".tmp",
                    dir=str(self.cache_path.parent),
                )
                tmp_path = Path(tmp_name)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        fh.write(text)
                        fh.flush()
                        os.fsync(fh.fileno())
                    os.chmod(tmp_path, mode)
                    os.replace(tmp_path, self.cache_path)
                except Exception:
                    try:
                        tmp_path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise
                try:  # best-effort directory fsync so the rename is durable
                    dir_fd = os.open(str(self.cache_path.parent), os.O_RDONLY)
                    try:
                        os.fsync(dir_fd)
                    finally:
                        os.close(dir_fd)
                except OSError:
                    pass

                self._dirty = False
                logger.debug("Saved %d label cache entries to %s", len(self._entries), self.cache_path)
                return True
            except Exception as exc:
                logger.error("Failed to save label cache to %s: %s", self.cache_path, exc)
                return False

    def _normalize_key(self, file_path: Union[str, Path]) -> str:
        """
        Return the normalized key for indexing: the resolved absolute path.

        The old key was just the filename, so images with the same name in
        different folders overwrote each other in the cache. Relative paths
        resolve against dataset_dir. (Persisted JSON keys stay
        basename-based when unique — see _persist_keys.)
        """
        p = Path(file_path)
        if not p.is_absolute():
            p = self.dataset_dir / p
        try:
            return str(p.resolve())
        except Exception:
            return str(p.absolute())

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

    def _lookup_entry_and_stat(
        self, file_path: Union[str, Path]
    ) -> Tuple[Optional[LabelCacheEntry], str, int, int]:
        """
        Single-pass lookup for the relabel decisions: one get_entry() and
        one disk stat. Returns (entry, stat_kind, mtime_ns, size) where
        stat_kind is "ok" (file stat'ed), "error" (file found but stat
        failed) or "missing" (not found anywhere on disk). Previously
        should_relabel() and can_delta_relabel() each did their own
        lookup + resolve + stat, so plan_batch_relabel() paid for both.
        """
        entry = self.get_entry(file_path)
        path_obj = self._resolve_existing_path(file_path)
        if path_obj is not None and path_obj.is_file():
            try:
                st = path_obj.stat()
                return entry, "ok", int(st.st_mtime_ns), int(st.st_size)
            except Exception:
                return entry, "error", 0, 0
        return entry, "missing", 0, 0

    @staticmethod
    def _is_fresh(
        entry: Optional[LabelCacheEntry],
        stat_kind: str,
        mtime_ns: int,
        size: int,
        exemplar_revision: int,
        allow_zero_box_skip: bool,
    ) -> bool:
        """
        True iff the cached entry is up to date and may be skipped — the
        exact skip condition of should_relabel() (minus `force`).
        """
        if entry is None:
            return False
        # 0-box (unlabeled) images are never skipped unless explicitly allowed.
        if not allow_zero_box_skip and entry.box_count <= 0:
            return False
        # Unparseable revision fails open to relabel instead of raising
        # ValueError into the batch loop.
        try:
            exemplar_rev = int(exemplar_revision)
        except (TypeError, ValueError):
            return False
        if entry.exemplar_revision != exemplar_rev:
            return False
        if stat_kind == "ok":
            return mtime_ns == entry.file_mtime_ns and size == entry.file_size
        if stat_kind == "error":
            return False
        # Missing from disk: only entries that never existed on disk
        # (file_mtime_ns == 0) count as fresh; a recorded file that is
        # now gone must be (re)considered, never silently skipped.
        return entry.file_mtime_ns <= 0

    @staticmethod
    def _is_delta_eligible(
        entry: Optional[LabelCacheEntry],
        stat_kind: str,
        mtime_ns: int,
        size: int,
        exemplar_revision: int,
    ) -> bool:
        """True iff fast candidate-crop delta relabeling applies."""
        if entry is None or not entry.candidate_crops:
            return False
        # Only delta-relabel if exemplar revision has updated
        # (unparseable revision: no delta, fall back to full relabel).
        try:
            exemplar_rev = int(exemplar_revision)
        except (TypeError, ValueError):
            return False
        if entry.exemplar_revision == exemplar_rev:
            return False
        if stat_kind == "ok":
            return mtime_ns == entry.file_mtime_ns and size == entry.file_size
        if stat_kind == "error":
            return False
        return True

    def classify(
        self,
        file_path: Union[str, Path],
        exemplar_revision: int,
        force: bool = False,
    ) -> str:
        """
        Single-pass bucket for one image: "skip" | "delta" | "full" |
        "missing". One get_entry() + one disk stat, where the old
        should_relabel() + can_delta_relabel() pair paid for two of each.
        Semantics are identical to plan_batch_relabel()'s old two-call
        decision (skip iff not force and not should_relabel; delta iff
        can_delta_relabel; missing iff the path is not a file on disk).
        """
        p_obj = Path(file_path)
        if not p_obj.is_file():
            return "missing"
        entry, kind, mtime_ns, size = self._lookup_entry_and_stat(p_obj)
        if not force and self._is_fresh(
            entry, kind, mtime_ns, size, exemplar_revision, False
        ):
            return "skip"
        if self._is_delta_eligible(entry, kind, mtime_ns, size, exemplar_revision):
            return "delta"
        return "full"

    def get_entry(self, file_path: Union[str, Path]) -> Optional[LabelCacheEntry]:
        """Lookup cache entry by resolved absolute path, raw key, or unique filename."""
        key = self._normalize_key(file_path)
        with self._lock:
            if key in self._entries:
                return self._entries[key]
            # Try raw str key
            raw_key = str(file_path)
            if raw_key in self._entries:
                return self._entries[raw_key]
            # Legacy basename lookup — only when exactly one entry matches.
            # An ambiguous bare name (the F-0803 collision) returns None
            # instead of silently returning the wrong folder's entry.
            # Served by the _by_basename index (O(1)) instead of a full
            # scan; each candidate is re-checked against the original
            # predicate so a stale bucket can never return a wrong entry.
            name = Path(file_path).name
            bucket = self._by_basename.get(name)
            if not bucket:
                return None
            matches = [
                self._entries[k]
                for k in bucket
                if k in self._entries
                and (
                    Path(self._entries[k].file_path).name == name
                    or Path(k).name == name
                )
            ]
            if len(matches) == 1:
                return matches[0]
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

        entry, kind, mtime_ns, size = self._lookup_entry_and_stat(file_path)
        return not self._is_fresh(
            entry, kind, mtime_ns, size, exemplar_revision, allow_zero_box_skip
        )

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
            self._index_remove(key, existing)
            self._index_add(key, entry)
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
                # Do NOT overwrite box_count with len(crops): candidate
                # crops are proposals, not labelled boxes. Overwriting let
                # a 0-box image look labelled (box_count>0) and be skipped,
                # defeating the zero-box safety in should_relabel. The
                # real box_count is owned by record_label / re-evaluate.
                entry.updated_at = time.time()
                if exemplar_revision is not None:
                    entry.exemplar_revision = int(exemplar_revision)
                self._dirty = True
                if auto_save and self.auto_save:
                    self.save()
            else:
                # Record initial entry. P2-9 (K-06): a fresh entry also
                # starts at box_count=0 — candidate crops are proposals,
                # not labelled boxes. Starting at len(stored_crops) let a
                # 0-box image look labelled (box_count>0) and be skipped
                # by should_relabel / plan_batch_relabel, piercing the
                # zero-box guarantee. The real box_count is set later by
                # record_label / re_evaluate_candidate_crops.
                rev = int(exemplar_revision or 0)
                self.record_label(
                    file_path=file_path,
                    exemplar_revision=rev,
                    box_count=0,
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
        entry, kind, mtime_ns, size = self._lookup_entry_and_stat(file_path)
        return self._is_delta_eligible(entry, kind, mtime_ns, size, exemplar_revision)

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
        any_unverified = False

        for cand in entry.candidate_crops:
            c = dict(cand)

            # Layer 0: Immunity Shield for human approved / hand labeled boxes.
            # P1-12 (K-34): çıplak "approved" boru hattı bayrağıdır,
            # insan sayılmaz — is_hand yalnızca hand_label /
            # human_approved'dir. Bu delta yolunun bağışıklık TABANI
            # 0.95'tir (v16 davranışı; test_label_cache_delta + w43
            # kilitler): el kutusu hayatta kalır ve güveni 0.95
            # tabanına çekilir — bu, model ölçümünü şişirmek değil,
            # insan-doğrulamalı kutunun güven TANIMIDIR (el kutusu
            # ölçülmüş model skoru taşımaz; taban olmazsa insan kutusu
            # Tier-3 "Düşük Güven"e düşer). calibrate yolu
            # (label_trust) etkilenmez — oradaki F-1212 never-inflate
            # (0.5 aynen 0.5 kalır) aynen korunur; şişirme yasağı
            # ölçülmüş MODEL güvenleri içindir.
            is_hand = bool(c.get("hand_label") or c.get("human_approved"))
            if is_hand:
                c["trusted"] = True
                c["weak"] = False
                c["confidence"] = max(0.95, _safe_confidence(c.get("confidence"), 0.95))
                c["delta_reevaluated"] = True
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

            try:
                cid = int(c.get("class_id", 0))
            except (TypeError, ValueError):
                logger.warning(
                    "Skipping candidate crop with non-numeric class_id %r in %s",
                    c.get("class_id"),
                    file_path,
                )
                continue

            # 1. Negative Exemplar Check (Suppression)
            suppressed = False
            suppress_thresh = getattr(exemplar_memory, "suppress_neg_threshold", 0.82)
            if hasattr(exemplar_memory, "match_negative"):
                if feat is not None and hasattr(exemplar_memory, "_negative"):
                    # Fast vector / dict similarity
                    best_neg_sim = 0.0
                    for neg_ex in getattr(exemplar_memory, "_negative", []):
                        if cid is not None and _safe_class_id(neg_ex.get("class_id", -1)) != cid:
                            sim = compute_similarity(feat, neg_ex) * 0.90
                        else:
                            sim = compute_similarity(feat, neg_ex)
                        if sim > best_neg_sim:
                            best_neg_sim = sim
                        # Early exit: best_neg_sim is only ever compared
                        # against suppress_thresh below, so once the
                        # threshold is reached the crop is suppressed no
                        # matter what the remaining exemplars score.
                        if best_neg_sim >= suppress_thresh:
                            break
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
                            pos_cid = _safe_class_id(pos_ex.get("class_id", 0), default=-1)
                            if pos_cid < 0:
                                continue
                            best_pos_sim = sim
                            best_cid = pos_cid
                            best_cname = str(pos_ex.get("class_name", ""))
                            # Early exit: compute_similarity() is clipped
                            # to [0, 1], so no later exemplar can beat a
                            # perfect 1.0 match (strict `>` keeps the
                            # first maximum, which we already hold).
                            if best_pos_sim >= 1.0:
                                break
                elif loaded_frame is not None:
                    crop = extract_box_crop(loaded_frame, c)
                    if crop is not None and crop.size > 0:
                        best_pos_sim, best_cid, best_cname, _ = exemplar_memory.match_positive(crop)

                reinforce_thresh = getattr(exemplar_memory, "reinforce_pos_threshold", 0.70)
                reassign_thresh = getattr(exemplar_memory, "reassign_pos_threshold", 0.78)

                if best_pos_sim >= reinforce_thresh and best_cid is not None:
                    if best_cid == cid:
                        old_conf = _safe_confidence(c.get("confidence"), 0.80)
                        boost = (1.0 - old_conf) * (best_pos_sim - 0.50) * 0.70
                        c["confidence"] = round(float(np.clip(old_conf + boost, old_conf, 0.99)), 4)
                        c["exemplar_reinforced"] = True
                        c["exemplar_sim"] = best_pos_sim
                    elif best_pos_sim >= reassign_thresh:
                        c["class_id"] = best_cid
                        c["class_name"] = best_cname or f"class_{best_cid}"
                        c["exemplar_corrected"] = True
                        c["exemplar_sim"] = best_pos_sim
                        old_c = _safe_confidence(c.get("confidence"), 0.80)
                        c["confidence"] = round(
                            float(np.clip(max(old_c, best_pos_sim * 0.95), 0.85, 0.99)), 4
                        )

            # Check confidence floor (S-2103: _safe_confidence so one bad
            # cached value cannot abort the whole file's delta relabel).
            conf = _safe_confidence(c.get("confidence"), 0.0)
            if conf < conf_floor:
                continue

            # If no visual feature could be obtained (neither cached nor
            # extractable from a frame), the exemplar checks above were
            # silently skipped: mark the crop unverified (kept, but NOT
            # counted as re-evaluated) instead of letting it pass as a
            # freshly re-evaluated box. A mere available frame is not
            # enough — without a feature no exemplar comparison ran.
            if feat is None:
                c["delta_reevaluated"] = False
                any_unverified = True
            else:
                c["delta_reevaluated"] = True

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
                if any_unverified:
                    # Do NOT advance the revision: some crops could not be
                    # re-evaluated, so the entry must be retried next run.
                    logger.warning(
                        "Delta relabel of %s left %d crop(s) unverified (no feature/frame); "
                        "exemplar_revision NOT advanced",
                        file_path,
                        sum(1 for c in surviving if not c.get("delta_reevaluated", True)),
                    )
                else:
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
            for cand in (key, str(file_path)):
                if cand in self._entries:
                    self._index_remove(cand, self._entries[cand])
                    del self._entries[cand]
                    removed = True
            if not removed:
                entry = self.get_entry(file_path)
                if entry is not None:
                    for k, v in list(self._entries.items()):
                        if v is entry:
                            self._index_remove(k, v)
                            del self._entries[k]
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
            self._by_basename.clear()
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
            persist_keys = self._persist_keys()
            return {persist_keys[k]: entry.to_dict() for k, entry in self._entries.items()}

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
            norm_key = self._normalize_key(key)
            old = self._entries.get(norm_key)
            if isinstance(value, LabelCacheEntry):
                self._entries[norm_key] = value
            elif isinstance(value, dict):
                self._entries[norm_key] = LabelCacheEntry.from_dict(value, default_path=key)
            else:
                raise TypeError(f"Value must be LabelCacheEntry or dict, got {type(value)}")
            self._index_remove(norm_key, old)
            self._index_add(norm_key, self._entries[norm_key])
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
    """
    Get or create singleton LabelCache instance.

    A second call with a different cache_path OR a different dataset_dir
    rebuilds the singleton: previously only cache_path was compared, so a
    second call with another dataset_dir silently kept serving the first
    dataset's cache.
    """
    global _GLOBAL_LABEL_CACHE
    with _GLOBAL_LOCK:
        if cache_path is not None:
            effective_path = Path(cache_path).resolve()
        elif dataset_dir is not None:
            effective_path = (Path(dataset_dir) / ".label_cache.json").resolve()
        else:
            effective_path = DEFAULT_CACHE_PATH.resolve()
        if _GLOBAL_LABEL_CACHE is None or (
            effective_path != _GLOBAL_LABEL_CACHE.cache_path
            or (
                dataset_dir is not None
                and Path(dataset_dir).resolve() != _GLOBAL_LABEL_CACHE.dataset_dir
            )
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
      - 'missing': Paths that do not exist on disk (reported, never silently dropped).

    GUARANTEE: 0-box (unlabeled) images are NEVER assigned to 'skip'.
    """
    cache = label_cache or get_label_cache()
    buckets: Dict[str, List[Path]] = {"skip": [], "delta": [], "full": [], "missing": []}

    for p in image_paths:
        # One classify() call per image (one lookup + one stat) instead of
        # the old should_relabel() + can_delta_relabel() pair (two of each).
        buckets[cache.classify(p, exemplar_revision, force=force)].append(Path(p))

    return buckets
