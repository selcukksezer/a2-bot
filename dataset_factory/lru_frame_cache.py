# -*- coding: utf-8 -*-
"""
dataset_factory/lru_frame_cache.py
==================================
Thread-safe, `OrderedDict`-based, `mtime`-validated LRU Frame Decode Cache
(`LRUFrameDecodeCache`) for Step 2 review and rapid scrubbing across 2,400+ frames.

Key Capabilities:
1. `OrderedDict` LRU Policy (Default Capacity = 32 Frames):
   Eliminates repeated disk reads and JPEG/PNG decoding during rapid forward/backward
   frame navigation while bounding RAM usage strictly to at most `capacity` frames.
2. Disk `mtime` & Size Validation:
   Automatically validates cached entries against on-disk modification timestamps
   (`st_mtime` / `st_mtime_ns`) and file size so externally overwritten or relabeled
   images are never served stale.
3. Thread Safety:
   Protected by a re-entrant lock (`threading.RLock`) for safe concurrent access
   between UI rendering and background relabeling / prefetch threads.
4. Windows Unicode Path Support:
   Decodes paths containing Turkish / non-ASCII characters (e.g. `selçuk`) via
   `np.fromfile` + `cv2.imdecode` with an ASCII fast-path.

Public API & Wiring (F-2000 / R65 — DO NOT delete as "dead code"):
    This module is the canonical frame-decode cache for Step 2 review
    scrubbing (`ui/yolo_factory_dialog.py`) and part of the public
    `dataset_factory` API: `LRUFrameDecodeCache`, `get_lru_frame_cache`
    (process-wide shared instance), `imread_unicode`, `FrameCacheEntry` and
    `DEFAULT_CACHE_CAPACITY`. It is wired through two points outside this
    file, not by direct imports from every consumer:
      * package-level export in `dataset_factory/__init__.py` (slice R46);
      * Step 2 consumer wiring in `ui/yolo_factory_dialog.py` (slice R44).
    Candidate in-package call sites that decode the same frame paths
    repeatedly and could use the shared cache:
    `dataset_factory/factory_controller.py` and
    `dataset_factory/label_cache.py` (their `imread_unicode` call sites).
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
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
    Tuple,
    Union,
)

import cv2
import numpy as np

logger = logging.getLogger("DatasetFactory.LRUFrameDecodeCache")

DEFAULT_CACHE_CAPACITY: int = 32

# Sentinel for _decode_and_store(disk_stat=...): distinguishes "no stat was
# passed, compute one" from an explicitly passed None (file missing).
_STAT_NOT_PROVIDED: Any = object()


def imread_unicode(
    path: Union[str, Path],
    flags: int = cv2.IMREAD_COLOR,
) -> Optional[np.ndarray]:
    """
    Safely decode an image file from disk supporting Windows Unicode / Turkish paths.
    Uses direct `cv2.imread` for pure-ASCII paths and `np.fromfile` + `cv2.imdecode`
    for non-ASCII paths.
    """
    try:
        file_path = Path(path)
        if not file_path.is_file():
            return None
        path_str = str(file_path)
        try:
            path_str.encode("ascii")
            img = cv2.imread(path_str, flags)
            if img is not None and getattr(img, "size", 0) > 0:
                return img
        except UnicodeEncodeError:
            pass

        raw_bytes = np.fromfile(path_str, dtype=np.uint8)
        if raw_bytes is None or raw_bytes.size == 0:
            return None
        img = cv2.imdecode(raw_bytes, flags)
        if img is None or getattr(img, "size", 0) == 0:
            return None
        return img
    except Exception as exc:
        logger.debug("imread_unicode failed for %s: %s", path, exc)
        return None


@dataclass
class FrameCacheEntry:
    """
    Single decoded frame record stored inside `LRUFrameDecodeCache`.
    Supports both attribute access (`.mtime`, `.image`) and 2-tuple unpacking
    (`mtime, img = entry`) for backwards compatibility.
    """

    key: str
    path: str
    mtime: float
    mtime_ns: int
    file_size: int
    image: np.ndarray
    existed_on_disk: bool = True
    cached_at: float = 0.0

    @property
    def frame(self) -> np.ndarray:
        return self.image

    @property
    def nbytes(self) -> int:
        return int(getattr(self.image, "nbytes", 0))

    def __iter__(self) -> Iterator[Any]:
        yield self.mtime
        yield self.image

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> Any:
        if index == 0:
            return self.mtime
        if index == 1:
            return self.image
        raise IndexError(index)


class LRUFrameDecodeCache:
    """
    Thread-safe, `OrderedDict`-based, `mtime`-validated LRU decoded image cache.

    Prevents repeated disk reads when scrubbing back and forth across 2,400+ frames
    in Step 2 while capping memory consumption to `capacity` frames (default: 32).
    """

    DEFAULT_CAPACITY: int = DEFAULT_CACHE_CAPACITY

    def __init__(
        self,
        capacity: int = DEFAULT_CACHE_CAPACITY,
        *,
        max_capacity: Optional[int] = None,
        maxsize: Optional[int] = None,
        max_items: Optional[int] = None,
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        flags: int = cv2.IMREAD_COLOR,
        copy_on_read: bool = True,
    ) -> None:
        resolved_cap = (
            max_capacity
            if max_capacity is not None
            else (maxsize if maxsize is not None else (max_items if max_items is not None else capacity))
        )
        self._capacity: int = max(1, int(resolved_cap))
        self._decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = decoder
        self._flags: int = int(flags)
        self._copy_on_read: bool = bool(copy_on_read)

        self._lock = threading.RLock()
        self._cache: "OrderedDict[str, FrameCacheEntry]" = OrderedDict()
        # Keys currently being decoded by another thread (in-flight dedup:
        # concurrent misses for the same path share one decode instead of
        # decoding twice). Guarded by `_inflight_cond` (same RLock).
        self._inflight: set = set()
        self._inflight_cond = threading.Condition(self._lock)

        # Telemetry counters
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._stale_invalidations: int = 0
        self._decode_errors: int = 0

    # ==========================================================================
    # Properties
    # ==========================================================================
    @property
    def capacity(self) -> int:
        """Maximum number of decoded frames kept in memory."""
        with self._lock:
            return self._capacity

    @capacity.setter
    def capacity(self, new_capacity: int) -> None:
        self.resize(new_capacity)

    @property
    def maxsize(self) -> int:
        """Alias for `capacity`."""
        return self.capacity

    @maxsize.setter
    def maxsize(self, new_capacity: int) -> None:
        self.resize(new_capacity)

    @property
    def hits(self) -> int:
        with self._lock:
            return self._hits

    @property
    def misses(self) -> int:
        with self._lock:
            return self._misses

    @property
    def evictions(self) -> int:
        with self._lock:
            return self._evictions

    @property
    def stale_invalidations(self) -> int:
        with self._lock:
            return self._stale_invalidations

    # ==========================================================================
    # Internal Path & Stat Helpers
    # ==========================================================================
    @staticmethod
    def _normalize_key(path: Union[str, Path]) -> str:
        """Normalize a file path into a canonical cache key.

        Uses the memoized lexical abspath (see _abspath_cached): resolve()
        is ~400-740us per call on this filesystem, and every cache operation
        normalizes its key.
        """
        if isinstance(path, Path):
            text = str(path)
        else:
            text = str(path).strip()
            if not text:
                return ""
        return _abspath_cached(text)

    @staticmethod
    def _stat_file(path: Union[str, Path]) -> Optional[Tuple[float, int, int]]:
        """
        Return `(st_mtime, st_mtime_ns, st_size)` for `path`, or `None` if the
        file does not exist or cannot be stat'd.
        """
        try:
            st = Path(path).stat()
            mtime = float(st.st_mtime)
            mtime_ns = int(getattr(st, "st_mtime_ns", int(mtime * 1e9)))
            size = int(st.st_size)
            return mtime, mtime_ns, size
        except OSError:
            return None

    @staticmethod
    def _is_mtime_valid(
        entry: FrameCacheEntry,
        disk_stat: Optional[Tuple[float, int, int]],
        explicit_mtime: Optional[float] = None,
    ) -> bool:
        """
        Check whether a cached entry matches the current file `mtime` on disk
        (or an explicitly supplied `mtime`).
        """
        if explicit_mtime is not None:
            return abs(entry.mtime - float(explicit_mtime)) <= 1e-6

        if disk_stat is None:
            # If the file existed on disk when cached and is now gone, it is invalid.
            # If it was a synthetic in-memory entry that never existed on disk, keep it valid.
            return not entry.existed_on_disk

        disk_mtime, disk_mtime_ns, disk_size = disk_stat
        if entry.file_size > 0 and disk_size > 0 and entry.file_size != disk_size:
            return False
        if entry.mtime_ns > 0 and disk_mtime_ns > 0:
            # Nanosecond stamps are exact: a differing ns value means the file
            # changed, even when the float-second mtimes look equal within 1e-6.
            return entry.mtime_ns == disk_mtime_ns
        return abs(entry.mtime - disk_mtime) <= 1e-6

    def _evict_overflow_locked(self) -> int:
        """Evict oldest (LRU) entries until size <= capacity. Assumes `_lock` is held."""
        evicted = 0
        while len(self._cache) > self._capacity:
            self._cache.popitem(last=False)
            self._evictions += 1
            evicted += 1
        return evicted

    def _prepare_output_image(
        self,
        image: np.ndarray,
        copy: Optional[bool] = None,
    ) -> np.ndarray:
        should_copy = self._copy_on_read if copy is None else bool(copy)
        return image.copy() if should_copy else image

    # ==========================================================================
    # Core Cache Operations
    # ==========================================================================
    def get_entry(
        self,
        path: Union[str, Path],
        *,
        mtime: Optional[float] = None,
        touch: bool = True,
    ) -> Optional[FrameCacheEntry]:
        """
        Look up and validate the `FrameCacheEntry` for `path`.
        Moves the entry to the MRU end of the `OrderedDict` on hit when `touch=True`.
        Automatically invalidates and removes the entry if its `mtime` changed on disk.
        """
        key = self._normalize_key(path)
        if not key:
            return None

        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None

            disk_stat = None if mtime is not None else self._stat_file(path)
            if not self._is_mtime_valid(entry, disk_stat, explicit_mtime=mtime):
                del self._cache[key]
                self._stale_invalidations += 1
                self._misses += 1
                return None

            if touch:
                self._cache.move_to_end(key, last=True)
            self._hits += 1
            return entry

    def get(
        self,
        path: Union[str, Path],
        default: Any = None,
        *,
        mtime: Optional[float] = None,
        decode_on_miss: bool = False,
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """
        Retrieve a cached frame for `path` if present and `mtime`-valid.
        If `decode_on_miss=True`, decodes from disk and caches the frame on miss.
        """
        entry = self.get_entry(path, mtime=mtime, touch=True)
        if entry is not None:
            return self._prepare_output_image(entry.image, copy=copy)

        if decode_on_miss:
            decoded = self._decode_and_store(path, decoder=decoder, mtime=mtime, copy=copy)
            if decoded is not None:
                return decoded

        return default

    def put(
        self,
        path: Union[str, Path],
        image: Union[np.ndarray, Tuple[float, np.ndarray]],
        *,
        mtime: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Insert or update a decoded frame in the LRU cache with its current `mtime`.
        Evicts the least-recently-used frame(s) when `len(cache) > capacity`.
        """
        # Support `(mtime, image)` tuple if passed by legacy callers
        if (
            isinstance(image, tuple)
            and len(image) == 2
            and isinstance(image[1], np.ndarray)
        ):
            try:
                mtime = float(image[0])
            except (TypeError, ValueError):
                pass
            image = image[1]

        if not isinstance(image, np.ndarray) or getattr(image, "size", 0) == 0:
            return None

        return self._put_with_stat(path, image, self._stat_file(path), mtime=mtime)

    def _put_with_stat(
        self,
        path: Union[str, Path],
        image: np.ndarray,
        disk_stat: Optional[Tuple[float, int, int]],
        *,
        mtime: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Insert/update a frame using an already-computed stat result (D42).

        `put()` stats the file itself; `_decode_and_store()` passes the stat
        it already took, so a decode miss costs ONE stat instead of three
        (`get_or_decode` + `_decode_and_store` + `put` each stat'd the same
        file — 3.00 stats/miss measured before this change).
        """
        # Defensive copy: the caller keeps owning its array, so later mutation
        # of the caller's buffer (e.g. a reused VideoCapture/scratch array)
        # cannot silently corrupt the cached frame. The stored entry holds
        # the copy; the caller's original array is returned unchanged.
        stored_image = image.copy()
        key = self._normalize_key(path)
        if not key:
            return None

        existed = disk_stat is not None

        if mtime is not None:
            resolved_mtime = float(mtime)
            resolved_mtime_ns = (
                disk_stat[1]
                if (disk_stat is not None and abs(disk_stat[0] - resolved_mtime) <= 1e-6)
                else int(resolved_mtime * 1e9)
            )
            resolved_size = disk_stat[2] if disk_stat is not None else 0
        elif disk_stat is not None:
            resolved_mtime, resolved_mtime_ns, resolved_size = disk_stat
        else:
            resolved_mtime = 0.0
            resolved_mtime_ns = 0
            resolved_size = 0

        entry = FrameCacheEntry(
            key=key,
            path=str(path),
            mtime=resolved_mtime,
            mtime_ns=resolved_mtime_ns,
            file_size=resolved_size,
            image=stored_image,
            existed_on_disk=existed,
            cached_at=time.time(),
        )

        with self._lock:
            if key in self._cache:
                self._cache[key] = entry
                self._cache.move_to_end(key, last=True)
            else:
                self._cache[key] = entry
                self._evict_overflow_locked()

        return image

    def set(
        self,
        path: Union[str, Path],
        image: Union[np.ndarray, Tuple[float, np.ndarray]],
        *,
        mtime: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """Alias for `put`."""
        return self.put(path, image, mtime=mtime)

    def _decode_and_store(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        flags: Optional[int] = None,
        mtime: Optional[float] = None,
        copy: Optional[bool] = None,
        disk_stat: Any = _STAT_NOT_PROVIDED,
        cache_key: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Decode image from disk outside the lock and store it in the cache.

        Concurrent misses for the same path share a single decode: a thread
        that finds the key already in flight waits for the decoding thread
        and then reuses the stored entry instead of decoding a second copy
        (avoids double decode + TOCTOU window between check and store).

        `disk_stat` (D42): pass the stat result the caller already took
        (e.g. `get_or_decode`) so the miss path stats the file once instead
        of three times; omitted it is computed here. The sentinel default
        distinguishes "no stat was passed" from an explicit None (stat
        failed / file missing). `cache_key` lets `get_or_decode` (which
        already stat'd the file and resolved the key for its hit check)
        reuse them instead of paying a second `stat` + `Path.resolve()`
        on the miss path.
        """
        file_path = Path(path)
        if disk_stat is _STAT_NOT_PROVIDED:
            disk_stat = self._stat_file(file_path)
        if disk_stat is None and mtime is None:
            return None

        key = cache_key if cache_key is not None else self._normalize_key(file_path)
        with self._inflight_cond:
            while key in self._inflight:
                self._inflight_cond.wait()
            entry = self._cache.get(key)
            if entry is not None and self._is_mtime_valid(
                entry, disk_stat, explicit_mtime=mtime
            ):
                # Another thread stored it while we waited (or between the
                # caller's check and ours). Reuse it; the caller already
                # counted this lookup as a miss, so do NOT count a hit here.
                self._cache.move_to_end(key, last=True)
                return self._prepare_output_image(entry.image, copy=copy)
            self._inflight.add(key)

        try:
            decode_fn = decoder or self._decoder
            try:
                if decode_fn is not None:
                    img = decode_fn(file_path)
                else:
                    read_flags = self._flags if flags is None else int(flags)
                    img = imread_unicode(file_path, flags=read_flags)
            except Exception as exc:
                logger.debug("Frame decode failed for %s: %s", file_path, exc)
                with self._lock:
                    self._decode_errors += 1
                return None

            if img is None or not isinstance(img, np.ndarray) or getattr(img, "size", 0) == 0:
                with self._lock:
                    self._decode_errors += 1
                return None

            effective_mtime = float(mtime) if mtime is not None else (disk_stat[0] if disk_stat else 0.0)
            self._put_with_stat(file_path, img, disk_stat, mtime=effective_mtime)
            return self._prepare_output_image(img, copy=copy)
        finally:
            with self._inflight_cond:
                self._inflight.discard(key)
                self._inflight_cond.notify_all()

    def get_or_decode(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        *,
        flags: Optional[int] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """
        Return the cached decoded frame for `path` if present and `mtime`-valid;
        otherwise decode from disk, store in the LRU cache, and return it.
        """
        if path is None:
            return None
        file_path = Path(path)
        disk_stat = self._stat_file(file_path)
        if disk_stat is None:
            key = self._normalize_key(file_path)
            with self._lock:
                entry = self._cache.get(key)
                if entry is not None and not entry.existed_on_disk:
                    # Synthetic in-memory entry that never existed on disk:
                    # keep it, exactly like get()/__contains__ do. Only an
                    # entry that used to exist on disk is stale when the
                    # file disappears.
                    self._cache.move_to_end(key, last=True)
                    self._hits += 1
                    return self._prepare_output_image(entry.image, copy=copy)
                if entry is not None:
                    del self._cache[key]
                    self._stale_invalidations += 1
                self._misses += 1
            return None

        key = self._normalize_key(file_path)
        with self._lock:
            entry = self._cache.get(key)
            if entry is not None:
                if self._is_mtime_valid(entry, disk_stat):
                    self._cache.move_to_end(key, last=True)
                    self._hits += 1
                    return self._prepare_output_image(entry.image, copy=copy)
                else:
                    del self._cache[key]
                    self._stale_invalidations += 1
            self._misses += 1

        return self._decode_and_store(
            file_path,
            decoder=decoder,
            flags=flags,
            mtime=disk_stat[0],
            copy=copy,
            disk_stat=disk_stat,
            cache_key=key,
        )

    def decode(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        *,
        flags: Optional[int] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Alias for `get_or_decode`."""
        return self.get_or_decode(path, decoder=decoder, flags=flags, copy=copy)

    def imread(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        *,
        flags: Optional[int] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Alias for `get_or_decode`."""
        return self.get_or_decode(path, decoder=decoder, flags=flags, copy=copy)

    def get_frame(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        *,
        flags: Optional[int] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Alias for `get_or_decode`."""
        return self.get_or_decode(path, decoder=decoder, flags=flags, copy=copy)

    def load(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        *,
        flags: Optional[int] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Alias for `get_or_decode`."""
        return self.get_or_decode(path, decoder=decoder, flags=flags, copy=copy)

    def __call__(
        self,
        path: Union[str, Path],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
        *,
        flags: Optional[int] = None,
        copy: Optional[bool] = None,
    ) -> Optional[np.ndarray]:
        """Allow calling the cache instance directly as a function."""
        return self.get_or_decode(path, decoder=decoder, flags=flags, copy=copy)

    # ==========================================================================
    # Prefetching for Smooth Step 2 Scrubbing
    # ==========================================================================
    def prefetch(
        self,
        paths: Sequence[Union[str, Path]],
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
    ) -> int:
        """
        Warm the cache for a sequence of paths (up to `capacity`).
        Returns the number of frames available in cache from `paths`.
        """
        loaded = 0
        for p in list(paths)[: self.capacity]:
            img = self.get_or_decode(p, decoder=decoder)
            if img is not None:
                loaded += 1
        return loaded

    def prefetch_neighbors(
        self,
        paths: Sequence[Union[str, Path]],
        current_index: int,
        radius: int = 2,
        decoder: Optional[Callable[[Union[str, Path]], Optional[np.ndarray]]] = None,
    ) -> int:
        """
        Prefetch frames in `[current_index - radius, current_index + radius]`
        for lag-free forward/backward scrubbing in Step 2, keeping `current_index`
        at the most-recently-used position.
        """
        total = len(paths)
        if total == 0 or current_index < 0 or current_index >= total:
            return 0

        max_radius = max(0, min(int(radius), (self.capacity - 1) // 2))
        neighbor_indices: List[int] = []
        for offset in range(max_radius, 0, -1):
            prev_i = current_index - offset
            next_i = current_index + offset
            if 0 <= prev_i < total:
                neighbor_indices.append(prev_i)
            if 0 <= next_i < total:
                neighbor_indices.append(next_i)

        count = 0
        for idx in neighbor_indices:
            if self.get_or_decode(paths[idx], decoder=decoder) is not None:
                count += 1

        # Touch current_index last so it remains the most-recently-used entry
        if self.get_or_decode(paths[current_index], decoder=decoder) is not None:
            count += 1
        return count

    # ==========================================================================
    # Invalidation, Resize & Maintenance
    # ==========================================================================
    def invalidate(self, path: Union[str, Path]) -> bool:
        """Remove a specific file path from the cache. Returns True if removed."""
        key = self._normalize_key(path)
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def remove(self, path: Union[str, Path]) -> bool:
        """Alias for `invalidate`."""
        return self.invalidate(path)

    def evict(self, path: Union[str, Path]) -> bool:
        """Alias for `invalidate`."""
        return self.invalidate(path)

    def pop(self, path: Union[str, Path], default: Any = None) -> Any:
        """Remove and return the cached image for `path`, or `default` if absent."""
        key = self._normalize_key(path)
        with self._lock:
            entry = self._cache.pop(key, None)
            if entry is None:
                return default
            return entry.image

    def clear(self) -> None:
        """Remove all cached frames from memory and reset telemetry counters."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._stale_invalidations = 0
            self._decode_errors = 0

    def resize(self, new_capacity: int) -> None:
        """Update cache capacity and immediately evict oldest entries if shrunk."""
        with self._lock:
            self._capacity = max(1, int(new_capacity))
            self._evict_overflow_locked()

    def get_stats(self) -> Dict[str, Any]:
        """Return thread-safe cache telemetry and memory statistics."""
        with self._lock:
            total_lookups = self._hits + self._misses
            hit_rate = (self._hits / total_lookups) if total_lookups > 0 else 0.0
            mem_bytes = sum(entry.nbytes for entry in self._cache.values())
            return {
                "capacity": self._capacity,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "stale_invalidations": self._stale_invalidations,
                "decode_errors": self._decode_errors,
                "hit_rate": round(hit_rate, 4),
                "memory_bytes": mem_bytes,
                "memory_mb": round(mem_bytes / (1024.0 * 1024.0), 3),
                "keys": list(self._cache.keys()),
            }

    @property
    def stats(self) -> Dict[str, Any]:
        return self.get_stats()

    # ==========================================================================
    # Dictionary / Mapping Protocol Compatibility
    # ==========================================================================
    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    def __contains__(self, path: Union[str, Path]) -> bool:
        key = self._normalize_key(path)
        if not key:
            return False
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False
            disk_stat = self._stat_file(path)
            if not self._is_mtime_valid(entry, disk_stat):
                del self._cache[key]
                self._stale_invalidations += 1
                return False
            return True

    def __getitem__(self, path: Union[str, Path]) -> np.ndarray:
        # Decode from disk on miss (like get(decode_on_miss=True)); only raise
        # KeyError when the frame cannot be obtained at all.
        img = self.get(path, decode_on_miss=True)
        if img is None:
            raise KeyError(path)
        return img

    def __setitem__(
        self,
        path: Union[str, Path],
        value: Union[np.ndarray, Tuple[float, np.ndarray]],
    ) -> None:
        if self.put(path, value) is None:
            raise ValueError(f"Invalid frame value for cache key: {path!r}")

    def __delitem__(self, path: Union[str, Path]) -> None:
        if not self.invalidate(path):
            raise KeyError(path)

    def _validated_entries_locked(self) -> List[Tuple[str, FrameCacheEntry]]:
        """Return [(key, entry)] for mtime-valid entries only. Assumes `_lock`.

        Stale entries (file changed or deleted on disk) are purged first, so
        keys()/values()/items() never serve entries that get()/__contains__
        would reject. Synthetic in-memory entries (never on disk) stay valid.
        """
        valid: List[Tuple[str, FrameCacheEntry]] = []
        for key, entry in list(self._cache.items()):
            if self._is_mtime_valid(entry, self._stat_file(entry.path)):
                valid.append((key, entry))
            else:
                del self._cache[key]
                self._stale_invalidations += 1
        return valid

    def keys(self) -> List[str]:
        with self._lock:
            return [key for key, _ in self._validated_entries_locked()]

    def values(self) -> List[np.ndarray]:
        with self._lock:
            return [
                self._prepare_output_image(entry.image)
                for _, entry in self._validated_entries_locked()
            ]

    def items(self) -> List[Tuple[str, np.ndarray]]:
        with self._lock:
            return [
                (key, self._prepare_output_image(entry.image))
                for key, entry in self._validated_entries_locked()
            ]

    def __repr__(self) -> str:
        with self._lock:
            return (
                f"LRUFrameDecodeCache(capacity={self._capacity}, "
                f"size={len(self._cache)}, hits={self._hits}, "
                f"misses={self._misses}, evictions={self._evictions})"
            )


_GLOBAL_FRAME_CACHE: Optional[LRUFrameDecodeCache] = None
_GLOBAL_FRAME_CACHE_LOCK = threading.RLock()


# --- D42: absolute-path memo ---------------------------------------------
# _normalize_key() used to call Path.resolve() on EVERY cache lookup, and
# resolve() costs ~400-740us per call on this filesystem (symlink walk +
# stat syscalls) — more than the mtime stat itself (~150us). The key only
# needs a canonical spelling of the path, so normalize with the purely
# lexical os.path.abspath() (~4us) and memoize that per input string
# (~0.5us on a hit). Only ABSOLUTE paths are memoized: abspath() of a
# relative path depends on the process cwd, so caching it would go stale
# after a chdir; relative paths are recomputed fresh (still ~100x cheaper
# than resolve()). Symlink spellings are NOT canonicalized, unlike
# resolve(): two spellings of the same file (symlink vs target) now get
# separate entries (one extra decode at most); they can never serve each
# other's content, because every read still validates against the
# caller-supplied path's live mtime/size (see _is_mtime_valid).
_ABSPATH_CACHE_MAX_ENTRIES: int = 8192
_abspath_cache: Dict[str, str] = {}
_abspath_cache_lock = threading.Lock()


def _abspath_cached(path_str: str) -> str:
    """Return os.path.abspath(path_str), memoized for absolute inputs only.

    Relative inputs are resolved fresh on every call: their abspath depends
    on the process working directory, so a memoized value could silently
    point at the wrong directory after a chdir.
    """
    if not os.path.isabs(path_str):
        return os.path.abspath(path_str)
    with _abspath_cache_lock:
        cached = _abspath_cache.get(path_str)
        if cached is not None:
            return cached
    resolved = os.path.abspath(path_str)
    with _abspath_cache_lock:
        if len(_abspath_cache) >= _ABSPATH_CACHE_MAX_ENTRIES:
            _abspath_cache.clear()
        _abspath_cache[path_str] = resolved
    return resolved


def get_lru_frame_cache(capacity: int = DEFAULT_CACHE_CAPACITY) -> LRUFrameDecodeCache:
    """Return or initialize the shared process-wide `LRUFrameDecodeCache` instance.

    `capacity` only applies when the shared instance is first created. A later
    call with a different capacity does NOT resize the shared cache (silently
    resizing it would evict frames another caller is relying on); the mismatch
    is logged and the existing capacity is kept. Use ``.resize()`` explicitly
    on the returned instance if a resize is really intended.
    """
    global _GLOBAL_FRAME_CACHE
    with _GLOBAL_FRAME_CACHE_LOCK:
        if _GLOBAL_FRAME_CACHE is None:
            _GLOBAL_FRAME_CACHE = LRUFrameDecodeCache(capacity=capacity)
        elif int(capacity) != _GLOBAL_FRAME_CACHE.capacity:
            logger.warning(
                "get_lru_frame_cache(capacity=%d) ignored: shared cache already "
                "exists with capacity=%d; call .resize() explicitly to change it.",
                int(capacity),
                _GLOBAL_FRAME_CACHE.capacity,
            )
        return _GLOBAL_FRAME_CACHE


__all__ = [
    "DEFAULT_CACHE_CAPACITY",
    "FrameCacheEntry",
    "LRUFrameDecodeCache",
    "get_lru_frame_cache",
    "imread_unicode",
]
