# -*- coding: utf-8 -*-
"""
High-Speed Thread-Safe Screen Capture Engine (core/capture.py)

Provides ultra-fast screen grabbing using `mss` with per-thread `threading.local()`
isolation (preventing Windows GDI thread-local errors across PyQt6 QThreads and
worker threads), Fullscreen Exclusive black-frame validation, and dynamic
resolution scaling relative to the 1920x1080 reference resolution.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import mss
import numpy as np
from version import __version__

logger = logging.getLogger(__name__)

REF_WIDTH: int = 1920
REF_HEIGHT: int = 1080
SHARED_BUFFER_WIDTH: int = 2560
SHARED_BUFFER_HEIGHT: int = 1440
BLACK_FRAME_WARNING: str = (
    "Screen capture is black (Fullscreen Exclusive mode detected - switch to Borderless Windowed)"
)

# D03: validate_frame istatistik örnekleme üst sınırı. Tam kare üzerinde
# np.mean + np.max iki tam geçiş yapar (2560x1440x3 karede ~54 ms); örnekleme
# aynı kararı ~2 ms'de verir. Örnek, karenin sistematik (stride) alt kümesidir:
# örnek max'ı tam kare max'ını aşamaz, yani siyah kare hiçbir zaman geçerli
# sayılmaz (yanlış-geçerli imkânsız); örnek ortalaması yansız bir kestirimdir,
# karar yalnızca mean≈3.0 bıçak sırtı karelerde oynayabilir.
_MAX_VALIDATION_SAMPLE_ELEMENTS: int = 262144


def _frame_sample_stride(height: int, width: int, channels: int = 1) -> int:
    """Returns the spatial stride so a frame sample holds at most
    `_MAX_VALIDATION_SAMPLE_ELEMENTS` elements (1 = whole frame, exact)."""
    total = max(1, int(height) * int(width) * max(1, int(channels)))
    if total <= _MAX_VALIDATION_SAMPLE_ELEMENTS:
        return 1
    return max(1, int(math.ceil(math.sqrt(total / _MAX_VALIDATION_SAMPLE_ELEMENTS))))


class ZeroCopyFrameRingBuffer:
    """
    Item 41 — Paylaşımlı Bellek (ZeroCopyFrameRingBuffer) ile Sıfır Kopyalı Görüntü Akışı.

    Pre-allocates a ring buffer of contiguous BGR `uint8` frames (defaulting to 3 slots of
    up to `2560x1440x3` ~ 11MB/14MB) or backed by `multiprocessing.shared_memory` when
    requested. `write_frame()` writes/copies incoming pixels directly into the next ring
    slot in-place without allocating new heap arrays on every frame, while
    `get_shared_frame_view()` and `grab_roi_view(x1, y1, x2, y2)` return zero-copy
    read-only NumPy memory views (`flags.writeable = False`, `np.shares_memory(...) == True`)
    so `VisionEngine`, `AutoCartographer`, and `MobileRemoteServer` all read the exact
    same memory reference concurrently.
    """

    def __init__(
        self,
        width: int = SHARED_BUFFER_WIDTH,
        height: int = SHARED_BUFFER_HEIGHT,
        channels: int = 3,
        ring_size: int = 8,
    ) -> None:
        self.max_width: int = max(64, int(width))
        self.max_height: int = max(64, int(height))
        self.channels: int = max(1, int(channels))
        # F-0062/F-1855: 3 slotluk halka, tutulan bir görünüm 3 yazma sonra
        # alttan eziliyordu (torn frame). 8 slot ezilme penceresini büyütür;
        # uzun ömürlü tutulması gereken kareler için get_frame_copy() kullanılır.
        self.ring_size: int = max(2, int(ring_size))
        self._lock: threading.RLock = threading.RLock()

        # Pre-allocate contiguous ring buffer backing array once on startup
        self._ring_storage: np.ndarray = np.zeros(
            (self.ring_size, self.max_height, self.max_width, self.channels),
            dtype=np.uint8,
        )
        self._write_idx: int = 0
        self._latest_idx: int = 0
        # F-0415: ilk yazmadan önce geçerli bir kare yoktur; aktif boyutlar
        # 0 başlar (önceden 2560x1440 siyah kare "gerçek kare" gibi görünürdü).
        self._active_width: int = 0
        self._active_height: int = 0
        self._frame_seq: int = 0
        self._last_write_ts: float = 0.0

    @property
    def frame_sequence(self) -> int:
        """Returns the monotonically increasing sequence number of the latest written frame."""
        with self._lock:
            return self._frame_seq

    @property
    def active_shape(self) -> Tuple[int, int, int]:
        """Returns `(height, width, channels)` of the latest active frame in the ring buffer."""
        with self._lock:
            return (self._active_height, self._active_width, self.channels)

    def _ensure_capacity(self, height: int, width: int, channels: int = 3) -> None:
        """Expands the pre-allocated ring buffer only if a monitor resolution exceeds current capacity."""
        if height <= self.max_height and width <= self.max_width and channels == self.channels:
            return
        with self._lock:
            new_h = max(self.max_height, int(height))
            new_w = max(self.max_width, int(width))
            new_c = max(self.channels, int(channels))
            self._ring_storage = np.zeros(
                (self.ring_size, new_h, new_w, new_c),
                dtype=np.uint8,
            )
            self.max_height = new_h
            self.max_width = new_w
            self.channels = new_c
            self._write_idx = 0
            self._latest_idx = 0

    def write_frame(self, src_frame: np.ndarray) -> np.ndarray:
        """
        Writes `src_frame` into the next pre-allocated slot of the ring buffer in-place
        (handling BGRA -> BGR slicing or grayscale expansion without extra intermediate heap
        allocations) and returns a read-only zero-copy NumPy view of that slot.
        """
        # F-0222: 1 boyutlu (veya boyutsuz) dizi shape[1]/shape[2] erişiminde
        # IndexError fırlatırdı; geçersiz girdi mevcut görünümü döndürür.
        if src_frame is None or not isinstance(src_frame, np.ndarray) or src_frame.size == 0:
            return self.get_shared_frame_view()
        if src_frame.ndim < 2:
            logger.warning(
                "ZeroCopyFrameRingBuffer.write_frame: %d boyutlu dizi yoksayıldı "
                "(en az 2 boyut gerekir).",
                src_frame.ndim,
            )
            return self.get_shared_frame_view()

        h = int(src_frame.shape[0])
        w = int(src_frame.shape[1])
        c = 1 if src_frame.ndim == 2 else int(src_frame.shape[2])

        self._ensure_capacity(h, w, 3)

        with self._lock:
            slot_idx = self._write_idx
            dest_view = self._ring_storage[slot_idx, :h, :w, :3]

            if src_frame.ndim == 3 and c == 4:
                # Copy BGR channels directly into pre-allocated destination slot without allocating a new array
                np.copyto(dest_view, src_frame[:, :, :3])
            elif src_frame.ndim == 3 and c >= 3:
                np.copyto(dest_view, src_frame[:, :, :3])
            elif src_frame.ndim == 2:
                dest_view[:, :, 0] = src_frame
                dest_view[:, :, 1] = src_frame
                dest_view[:, :, 2] = src_frame
            else:
                np.copyto(dest_view, src_frame.reshape(h, w, 3))

            self._active_height = h
            self._active_width = w
            self._latest_idx = slot_idx
            self._write_idx = (slot_idx + 1) % self.ring_size
            self._frame_seq += 1
            self._last_write_ts = time.perf_counter()

            ro_view = dest_view.view()
            ro_view.flags.writeable = False
            return ro_view

    def get_shared_frame_view(self) -> np.ndarray:
        """
        Returns a zero-copy read-only NumPy view (`flags.writeable == False`) of the
        most recently captured full frame in the ring buffer.
        Multiple callers (`VisionEngine`, `AutoCartographer`, `MobileRemoteServer`)
        receive views backed by the exact same physical memory (`np.shares_memory` is True).
        """
        with self._lock:
            h = self._active_height
            w = self._active_width
            slot_idx = self._latest_idx
            view = self._ring_storage[slot_idx, :h, :w, :3].view()
            view.flags.writeable = False
            return view

    def get_frame_copy(self) -> np.ndarray:
        """
        Returns an owned (heap-allocated) copy of the latest frame.

        F-0062/F-1855: zero-copy görünümler halka sarınca alttan ezilir;
        kareyi yazma döngüsünden uzun tutacak tüketiciler bu kopyayı kullanır.
        Henüz kare yazılmadıysa boş (0, 0, 3) dizi döner.
        """
        with self._lock:
            h = self._active_height
            w = self._active_width
            slot_idx = self._latest_idx
            if self._frame_seq == 0 or h <= 0 or w <= 0:
                return np.zeros((0, 0, 3), dtype=np.uint8)
            return np.array(self._ring_storage[slot_idx, :h, :w, :3], dtype=np.uint8, copy=True)

    def grab_roi_view(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
    ) -> np.ndarray:
        """
        Returns a zero-copy read-only slice view `[y1:y2, x1:x2]` directly from the
        latest shared frame buffer without copying pixels.
        Supports both normalized coordinates (`0.0 .. 1.0` when values <= 1.0) and
        absolute pixel coordinates.
        """
        frame_view = self.get_shared_frame_view()
        fh, fw = frame_view.shape[:2]
        if fh <= 0 or fw <= 0:
            empty = frame_view[:1, :1, :3].view()
            empty.flags.writeable = False
            return empty

        # Detect normalized coordinates if all coordinates are floats within [0.0, 1.0]
        # F-0223: int (0, 0, 1, 1) de tam kare anlamındaki normalize girdidir;
        # piksel olarak yorumlansaydı 1 piksellik ROI dönerdi. Tam-kare int
        # dörtlüsü de normalize sayılır; diğer int girdiler pikseldir.
        all_in_unit_range = (
            0.0 <= float(x1) <= 1.0
            and 0.0 <= float(y1) <= 1.0
            and 0.0 <= float(x2) <= 1.0
            and 0.0 <= float(y2) <= 1.0
        )
        is_normalized = (
            (
                isinstance(x1, float)
                or isinstance(y1, float)
                or isinstance(x2, float)
                or isinstance(y2, float)
            )
            or (all_in_unit_range and float(x1) == 0.0 and float(y1) == 0.0 and float(x2) == 1.0 and float(y2) == 1.0)
        ) and all_in_unit_range

        if is_normalized and (float(x2) <= 1.0 and float(y2) <= 1.0):
            px1 = int(round(float(x1) * fw))
            py1 = int(round(float(y1) * fh))
            px2 = int(round(float(x2) * fw))
            py2 = int(round(float(y2) * fh))
        else:
            px1 = int(round(float(x1)))
            py1 = int(round(float(y1)))
            px2 = int(round(float(x2)))
            py2 = int(round(float(y2)))

        px1 = max(0, min(px1, fw - 1))
        py1 = max(0, min(py1, fh - 1))
        px2 = max(px1 + 1, min(px2, fw))
        py2 = max(py1 + 1, min(py2, fh))

        roi_view = frame_view[py1:py2, px1:px2, :3].view()
        roi_view.flags.writeable = False
        return roi_view


class ScreenCapture:
    """
    Thread-safe, high-FPS screen capture engine backed by `mss`.

    Each calling thread lazily initializes its own `mss.mss()` instance inside
    `threading.local()`, avoiding Windows GDI device-context (`srcdc`) errors when
    invoked concurrently from PyQt6 UI threads and background worker threads.
    """

    def __init__(
        self,
        monitor_idx: int = 1,
        ref_width: int = REF_WIDTH,
        ref_height: int = REF_HEIGHT,
    ) -> None:
        self.monitor_idx: int = max(1, int(monitor_idx))
        self.ref_width: int = max(1, int(ref_width))
        self.ref_height: int = max(1, int(ref_height))
        self._thread_local: threading.local = threading.local()
        self._lock: threading.RLock = threading.RLock()

        # Default fallback geometry until monitor metadata is queried
        self.monitor: Dict[str, int] = {
            "left": 0,
            "top": 0,
            "width": self.ref_width,
            "height": self.ref_height,
        }
        self.screen_w: int = self.ref_width
        self.screen_h: int = self.ref_height
        self.last_fps: float = 0.0
        self._last_grab_ts: float = 0.0
        self._last_valid: bool = True
        self._last_validation_msg: str = "OK"
        # F-0004/F-0005/F-0063: yakalama arızası siyah kare olarak gizlenmesin
        # diye arıza durumu ayrıca izlenir (bkz. grab_frame/grab_region).
        self._last_grab_failed: bool = False
        self._last_grab_error: str = ""
        self._grab_error_count: int = 0
        self._last_grab_ok_ts: float = 0.0
        self._fallback_log_ts: float = 0.0
        # F-0416: close() tüm thread'lerin mss örneğini kapatabilsin diye
        # oluşturulan örnekler burada izlenir.
        self._sct_instances: set = set()
        self._sct_lock: threading.Lock = threading.Lock()
        self._bitblt_warned: bool = False
        self.ring_buffer: ZeroCopyFrameRingBuffer = ZeroCopyFrameRingBuffer(
            width=max(SHARED_BUFFER_WIDTH, self.ref_width),
            height=max(SHARED_BUFFER_HEIGHT, self.ref_height),
            channels=3,
            ring_size=8,
        )

        self.refresh_monitor_info()

    # ------------------------------------------------------------------
    # Thread-Local MSS Instance Management
    # ------------------------------------------------------------------
    def _get_sct(self) -> Any:
        """
        Returns the calling thread's dedicated `mss.mss()` instance,
        creating it on demand if not yet initialized for this thread.
        """
        sct = getattr(self._thread_local, "sct", None)
        if sct is not None:
            # P2-4 (K-17): close() başka thread'in thread-local'ini
            # temizleyemez (threading.local yalnızca sahibi thread'den
            # yazılır) — ama close() örneği _sct_instances'tan çıkarır.
            # Kayıtta olmayan örnek kapalıdır/atılmıştır: bu thread'in
            # elindeki bayat örneği düşürüp taze örnek açılır; yoksa
            # close() sonrası ilk grab her thread'de kapalı handle'a
            # çarpardı (stealth tarafında kurtarma da yoktu).
            with self._sct_lock:
                if sct in self._sct_instances:
                    return sct
            self._thread_local.sct = None
            sct = None
        if sct is None:
            try:
                mss_cls = getattr(mss, "MSS", mss.mss)
                sct = mss_cls()
                self._thread_local.sct = sct
                # F-0416: close() bu thread'in dışındaki örnekleri de kapatabilsin.
                with self._sct_lock:
                    # Prune stale/closed instances to prevent unbounded GDI leaks
                    if len(self._sct_instances) > 8:
                        self._sct_instances = {inst for inst in self._sct_instances if getattr(inst, "monitors", None)}
                    self._sct_instances.add(sct)
            except Exception as exc:
                logger.error("Failed to initialize thread-local mss instance: %s", exc)
                raise
        return sct

    def refresh_monitor_info(self) -> Dict[str, int]:
        """Queries and caches the active monitor's geometry (`left`, `top`, `width`, `height`)."""
        try:
            sct = self._get_sct()
            monitors = getattr(sct, "monitors", None) or []
            if len(monitors) > self.monitor_idx:
                mon = dict(monitors[self.monitor_idx])
            elif len(monitors) > 1:
                # F-0224: istenen monitör yoksa uyarısız monitors[1]'e düşülürdü.
                logger.warning(
                    "İstenen monitör idx=%d bulunamadı (%d monitör var); "
                    "monitors[1] kullanılıyor.",
                    self.monitor_idx,
                    len(monitors),
                )
                mon = dict(monitors[1])
            elif len(monitors) == 1:
                mon = dict(monitors[0])
            else:
                mon = {
                    "left": 0,
                    "top": 0,
                    "width": self.ref_width,
                    "height": self.ref_height,
                }

            with self._lock:
                self.monitor = {
                    "left": int(mon.get("left", 0)),
                    "top": int(mon.get("top", 0)),
                    "width": max(1, int(mon.get("width", self.ref_width))),
                    "height": max(1, int(mon.get("height", self.ref_height))),
                }
                self.screen_w = self.monitor["width"]
                self.screen_h = self.monitor["height"]
        except Exception as exc:
            logger.warning("Could not query monitor geometry via mss, using fallback: %s", exc)
        return dict(self.monitor)

    # ------------------------------------------------------------------
    # Core Capture Methods & Zero-Copy Shared Frame Buffer (Item 41)
    # ------------------------------------------------------------------
    def grab_frame(self, grayscale: bool = False) -> np.ndarray:
        """
        Captures the full monitor directly into the pre-allocated `ZeroCopyFrameRingBuffer`
        and returns a read-only zero-copy NumPy view (`get_shared_frame_view()`) so
        `VisionEngine`, `AutoCartographer`, and `MobileRemoteServer` share the exact same
        memory buffer without copying 14MB arrays per frame.
        """
        t0 = time.perf_counter()
        target = dict(self.monitor)
        try:
            sct = self._get_sct()
            raw = sct.grab(target)
            img = np.asarray(raw, dtype=np.uint8)
            dt = time.perf_counter() - t0
            if dt > 1e-6:
                self.last_fps = round(1.0 / dt, 1)
            self._last_grab_ts = time.time()
            self._last_grab_failed = False
            self._last_grab_error = ""
            self._last_grab_ok_ts = time.time()
            shared_view = self.ring_buffer.write_frame(img)
            if grayscale:
                return cv2.cvtColor(shared_view, cv2.COLOR_BGR2GRAY)
            return shared_view
        except Exception as exc:
            # F-0004/F-0005: arıza siyah kareye çevrilip gerçek kareymiş gibi
            # ring buffer'a yazılmaz ve sessiz geçilmez. Yedek grab denendikten
            # sonra bile kare alınamazsa son geçerli kare (yazmadan) döner;
            # hiç geçerli kare yoksa arıza yükseltilir.
            logger.error("ScreenCapture.grab_frame yakalama arızası: %s", exc)
            self._last_grab_failed = True
            self._last_grab_error = str(exc)
            self._grab_error_count += 1
            try:
                frame = self.grab_region(target, grayscale=False)
            except Exception:
                frame = None
            if (
                frame is not None
                and isinstance(frame, np.ndarray)
                and frame.size > 0
                and not self._last_grab_failed
            ):
                shared_view = self.ring_buffer.write_frame(frame)
                if grayscale:
                    return cv2.cvtColor(shared_view, cv2.COLOR_BGR2GRAY)
                return shared_view
            if self.ring_buffer.frame_sequence > 0:
                logger.warning(
                    "grab_frame: geçerli kare alınamadı; son geçerli kare "
                    "döndürülüyor (ring buffer'a yazılmadı)."
                )
                return self.ring_buffer.get_shared_frame_view()
            raise RuntimeError(f"Screen capture failed: {exc}") from exc

    def get_shared_frame_view(self) -> np.ndarray:
        """
        Returns a read-only zero-copy NumPy view (`flags.writeable == False`) of the
        latest frame in the shared ring buffer (`ZeroCopyFrameRingBuffer`).
        If no frame has been written yet, triggers an initial `grab_frame()`.
        """
        if self.ring_buffer.frame_sequence == 0:
            self.grab_frame(grayscale=False)
        return self.ring_buffer.get_shared_frame_view()

    def grab_roi_view(
        self,
        x1: Union[int, float],
        y1: Union[int, float],
        x2: Union[int, float],
        y2: Union[int, float],
        refresh: bool = False,
    ) -> np.ndarray:
        """
        Returns a zero-copy read-only slice view `[y1:y2, x1:x2]` from the shared
        ring buffer (`ZeroCopyFrameRingBuffer`). Supports both pixel coordinates and
        normalized `[0.0 .. 1.0]` coordinates.
        """
        if refresh or self.ring_buffer.frame_sequence == 0:
            self.grab_frame(grayscale=False)
        return self.ring_buffer.grab_roi_view(x1, y1, x2, y2)

    def grab_full(self, grayscale: bool = False) -> np.ndarray:
        """
        Grabs the entire configured monitor frame via the shared zero-copy ring buffer.

        Args:
            grayscale: If True, returns a 2D `uint8` grayscale image `(H, W)`.
                       If False, returns a 3D `uint8` BGR image `(H, W, 3)`.
        """
        return self.grab_frame(grayscale=grayscale)

    def grab_region(
        self,
        rect: Optional[Union[Dict[str, int], Tuple[int, int, int, int], list]] = None,
        grayscale: bool = False,
    ) -> np.ndarray:
        """
        Grabs a rectangular screen region at high speed.

        Args:
            rect: Region specified either as `{"left": x, "top": y, "width": w, "height": h}`,
                  `{"x": x, "y": y, "w": w, "h": h}`, or a 4-tuple `(x, y, w, h)`.
                  If `None`, grabs the full monitor.
            grayscale: Whether to convert the captured BGRA buffer to grayscale.

        Returns:
            `np.ndarray` in BGR `(H, W, 3)` or Grayscale `(H, W)` format.
            Yakalama arızasında (yeniden deneme de başarısızsa) gerçek
            kareyle karışmaması için boş `(0, 0)` / `(0, 0, 3)` kare döner
            (F-0063); çağıran `frame.size == 0` veya
            `last_capture_failed` ile arızayı tanır.
        """
        target = self._normalize_rect_dict(rect)
        t0 = time.perf_counter()
        try:
            sct = self._get_sct()
            raw = sct.grab(target)
            img = np.asarray(raw, dtype=np.uint8)
            dt = time.perf_counter() - t0
            if dt > 1e-6:
                self.last_fps = round(1.0 / dt, 1)
            self._last_grab_ts = time.time()
            self._last_grab_failed = False
            self._last_grab_error = ""
            self._last_grab_ok_ts = time.time()

            if img.ndim == 3 and img.shape[2] == 4:
                if grayscale:
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            if img.ndim == 3 and img.shape[2] == 3:
                if grayscale:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                return img
            if img.ndim == 2:
                if grayscale:
                    return img
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            self._last_grab_failed = False
            self._last_grab_error = ""
            self._last_grab_ok_ts = time.time()
            return img
        except Exception as exc:
            # F-0225: yeniden denemeden önce eski mss handle'ı kapatılır.
            old_sct = getattr(self._thread_local, "sct", None)
            if old_sct is not None:
                try:
                    old_sct.close()
                except Exception:
                    pass
                with self._sct_lock:
                    self._sct_instances.discard(old_sct)
                self._thread_local.sct = None
            # Re-initialize thread-local mss handle once in case GDI context was reset
            try:
                new_sct = getattr(mss, 'MSS', mss.mss)()
                self._thread_local.sct = new_sct
                with self._sct_lock:
                    self._sct_instances.add(new_sct)
                raw = self._thread_local.sct.grab(target)
                img = np.asarray(raw, dtype=np.uint8)
                self._last_grab_failed = False
                self._last_grab_error = ""
                self._last_grab_ok_ts = time.time()
                if grayscale:
                    return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            except Exception as retry_exc:
                # F-0063: arıza dönüşü artık gerçek kareden ayırt edilemeyen
                # siyah (h, w) kare DEĞİL, boş (0, 0) karedir; arıza durumu
                # _last_grab_failed bayrağıyla da işaretlenir. Boş kare
                # grab_normalized'in (0, 0, 0, 0) kuralıyla tutarlıdır ve
                # doğrudan grab_region çağıranlar (vision, incident_recorder,
                # mobile_server) `frame.size == 0` ile arızayı tanır; siyah
                # kare gerçek kare sanılıp işlenmez. Kalıcı arıza warning
                # olarak (30 sn'de bir) loglanır, debug'a gömülmez.
                self._last_grab_failed = True
                self._last_grab_error = str(retry_exc)
                self._grab_error_count += 1
                now = time.perf_counter()
                if now - self._fallback_log_ts >= 30.0:
                    self._fallback_log_ts = now
                    logger.warning(
                        "ScreenCapture.grab_region yakalama arızası "
                        "(boş yedek kare dönülüyor, #%d): %s",
                        self._grab_error_count,
                        retry_exc,
                    )
                else:
                    logger.debug("ScreenCapture.grab_region fallback: %s", retry_exc)
                if grayscale:
                    return np.zeros((0, 0), dtype=np.uint8)
                return np.zeros((0, 0, 3), dtype=np.uint8)

    def grab_normalized(
        self,
        nx: float,
        ny: float,
        nw: float,
        nh: float,
        grayscale: bool = False,
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Grabs a region defined by normalized `[0.0, 1.0]` screen coordinates.

        Args:
            nx: Normalized left coordinate `(0.0 .. 1.0)`.
            ny: Normalized top coordinate `(0.0 .. 1.0)`.
            nw: Normalized width `(0.0 .. 1.0)`.
            nh: Normalized height `(0.0 .. 1.0)`.
            grayscale: If True, returns a 2D grayscale array; otherwise 3D BGR array.

        Returns:
            Tuple of `(captured_frame, (px_x, px_y, px_w, px_h))` in absolute screen pixels.
        """
        try:
            clamped_nx = float(np.clip(nx, 0.0, 1.0))
            clamped_ny = float(np.clip(ny, 0.0, 1.0))
            clamped_nw = float(np.clip(nw, 0.001, 1.0 - clamped_nx))
            clamped_nh = float(np.clip(nh, 0.001, 1.0 - clamped_ny))

            px_x = int(round(self.monitor["left"] + clamped_nx * self.screen_w))
            px_y = int(round(self.monitor["top"] + clamped_ny * self.screen_h))
            px_w = max(1, int(round(clamped_nw * self.screen_w)))
            px_h = max(1, int(round(clamped_nh * self.screen_h)))

            rect_dict = {
                "left": px_x,
                "top": px_y,
                "width": px_w,
                "height": px_h,
            }
            frame = self.grab_region(rect_dict, grayscale=grayscale)
            # F-0227: grab_region arızada boş (0, 0) yedek kare döner; bu
            # kare gerçek bölge karesiymiş gibi (0,0,10,10) ile sunulmaz.
            if self._last_grab_failed:
                logger.error(
                    "ScreenCapture.grab_normalized: bölge yakalanamadı "
                    "(%s); boş kare ve (0, 0, 0, 0) dönülüyor.",
                    self._last_grab_error,
                )
                empty = (
                    np.zeros((0, 0), dtype=np.uint8)
                    if grayscale
                    else np.zeros((0, 0, 3), dtype=np.uint8)
                )
                return empty, (0, 0, 0, 0)
            return frame, (px_x, px_y, px_w, px_h)
        except Exception as exc:
            logger.error("ScreenCapture.grab_normalized failed: %s", exc)
            self._last_grab_failed = True
            self._last_grab_error = str(exc)
            empty = (
                np.zeros((0, 0), dtype=np.uint8)
                if grayscale
                else np.zeros((0, 0, 3), dtype=np.uint8)
            )
            return empty, (0, 0, 0, 0)

    # ------------------------------------------------------------------
    # Frame Validation (Fullscreen Exclusive / Out-of-Bounds Detection)
    # ------------------------------------------------------------------
    def validate_frame(self, frame: Optional[np.ndarray]) -> Tuple[bool, str]:
        """
        Validates that a captured frame is non-empty, within valid dimensions,
        and not completely black (`mean < 3.0`, which happens when the game is
        running in Fullscreen Exclusive mode).

        Returns:
            `(is_valid: bool, message: str)`
        """
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                self._last_valid = False
                self._last_validation_msg = "empty frame: no pixels captured"
                logger.warning("Frame validation failed: empty frame")
                return False, self._last_validation_msg

            if frame.ndim < 2:
                self._last_valid = False
                self._last_validation_msg = "invalid frame: expected at least 2 dimensions"
                logger.warning("Frame validation failed: %s", self._last_validation_msg)
                return False, self._last_validation_msg

            h, w = int(frame.shape[0]), int(frame.shape[1])
            if h < 8 or w < 8 or h > self.screen_h * 4 or w > self.screen_w * 4:
                self._last_valid = False
                self._last_validation_msg = f"invalid frame dimensions: {w}x{h}"
                logger.warning("Frame validation failed: %s", self._last_validation_msg)
                return False, self._last_validation_msg

            # D03: tam karede mean+max iki tam geçiş yapar (1440p'de ~54 ms).
            # Örneklenmiş istatistik aynı kararı verir: örnek max'ı tam max'ı
            # aşamaz (siyah kare geçerli sayılmaz), örnek ortalaması yansızdır.
            channels = int(frame.shape[2]) if frame.ndim == 3 else 1
            stride = _frame_sample_stride(h, w, channels)
            sample = frame[::stride, ::stride] if stride > 1 else frame
            mean_brightness = float(np.mean(sample))
            max_brightness = float(np.max(sample))
            # F-0228: mean<3.0 koyu sahneyi de siyah sayardı. Tamamen siyah
            # (max<16) ile koyu ama içerikli sahne ayrılır; koyu sahne
            # geçerli sayılır, yalnızca uyarı loglanır.
            if mean_brightness < 3.0 and max_brightness < 16.0:
                self._last_valid = False
                self._last_validation_msg = BLACK_FRAME_WARNING
                logger.warning("%s (mean=%.2f, max=%.1f)", BLACK_FRAME_WARNING, mean_brightness, max_brightness)
                return False, BLACK_FRAME_WARNING
            if mean_brightness < 3.0:
                logger.warning(
                    "Dark (but non-black) scene detected (mean=%.2f, max=%.1f); "
                    "frame accepted as valid.",
                    mean_brightness,
                    max_brightness,
                )

            self._last_valid = True
            self._last_validation_msg = "OK"
            return True, "OK"
        except Exception as exc:
            logger.error("Frame validation error: %s", exc)
            return False, f"frame validation error: {exc}"

    # ------------------------------------------------------------------
    # Resolution Scaling Helpers (Relative to 1920x1080 Reference)
    # ------------------------------------------------------------------
    def get_scale_factor(
        self,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Returns `(scale_x, scale_y)` relative to the 1920x1080 reference resolution.
        For example:
        - 1920x1080 -> `(1.0, 1.0)`
        - 2560x1440 -> `(1.3333, 1.3333)`
        - 1280x720  -> `(0.6667, 0.6667)`
        """
        # F-0229: arızada (1.0, 1.0) varsayımı yapılmaz; bilinen ekran
        # boyutundan hesaplanan gerçek ölçek döner.
        try:
            w = int(width) if width is not None and int(width) > 0 else self.screen_w
            h = int(height) if height is not None and int(height) > 0 else self.screen_h
        except Exception:
            w, h = self.screen_w, self.screen_h
        scale_x = float(w) / float(self.ref_width)
        scale_y = float(h) / float(self.ref_height)
        return scale_x, scale_y

    def norm_to_px(
        self,
        nx: float,
        ny: float,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Converts normalized coordinates `(nx, ny)` in `[0.0, 1.0]` to pixel coordinates `(x, y)`.
        """
        # F-0229: arızada (0, 0) geçerli bir koordinatmış gibi dönülmez;
        # sonlu olmayan / dönüştürülemeyen girdi ValueError yükseltir.
        try:
            w = int(width) if width is not None and int(width) > 0 else self.screen_w
            h = int(height) if height is not None and int(height) > 0 else self.screen_h
            clamped_nx = float(np.clip(nx, 0.0, 1.0))
            clamped_ny = float(np.clip(ny, 0.0, 1.0))
        except Exception as exc:
            logger.error("norm_to_px error: %s", exc)
            raise ValueError(f"norm_to_px: invalid coordinates ({nx!r}, {ny!r}): {exc}") from exc
        if not (clamped_nx == clamped_nx and clamped_ny == clamped_ny):
            raise ValueError(f"norm_to_px: non-finite coordinates ({nx!r}, {ny!r})")
        x = int(round(clamped_nx * w))
        y = int(round(clamped_ny * h))
        return x, y

    def px_to_norm(
        self,
        x: Union[int, float],
        y: Union[int, float],
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> Tuple[float, float]:
        """
        Converts pixel coordinates `(x, y)` to normalized coordinates `(nx, ny)` in `[0.0, 1.0]`.
        """
        # S-0204: norm_to_px ile aynı kural (F-0229) — arızada (0.0, 0.0)
        # geçerli bir koordinatmış gibi dönülmez; geçersiz / sonlu olmayan
        # girdi ValueError yükseltir, çağıran arızayı ayırt edebilir.
        try:
            fx = float(x)
            fy = float(y)
            if not bool(np.isfinite(fx)) or not bool(np.isfinite(fy)):
                raise ValueError(f"px_to_norm: non-finite coordinates ({x!r}, {y!r})")
            w = max(1, int(width) if width is not None and int(width) > 0 else self.screen_w)
            h = max(1, int(height) if height is not None and int(height) > 0 else self.screen_h)
            nx = float(np.clip(fx / float(w), 0.0, 1.0))
            ny = float(np.clip(fy / float(h), 0.0, 1.0))
        except Exception as exc:
            logger.error("px_to_norm error: %s", exc)
            raise ValueError(f"px_to_norm: invalid coordinates ({x!r}, {y!r}): {exc}") from exc
        if not (nx == nx and ny == ny):
            raise ValueError(f"px_to_norm: non-finite coordinates ({x!r}, {y!r})")
        return round(nx, 6), round(ny, 6)

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------
    def _normalize_rect_dict(
        self,
        rect: Optional[Union[Dict[str, int], Tuple[int, int, int, int], list]],
    ) -> Dict[str, int]:
        """Normalizes a region specification into an `mss`-compatible dict."""
        if rect is None:
            return dict(self.monitor)

        try:
            if isinstance(rect, dict):
                left = int(rect.get("left", rect.get("x", self.monitor["left"])))
                top = int(rect.get("top", rect.get("y", self.monitor["top"])))
                width = int(rect.get("width", rect.get("w", self.screen_w)))
                height = int(rect.get("height", rect.get("h", self.screen_h)))
            elif isinstance(rect, (tuple, list)) and len(rect) >= 4:
                left = int(rect[0])
                top = int(rect[1])
                width = int(rect[2])
                height = int(rect[3])
            else:
                return dict(self.monitor)

            # Clamp dimensions to valid positive integers inside monitor bounds
            left = max(self.monitor["left"], min(left, self.monitor["left"] + self.screen_w - 1))
            top = max(self.monitor["top"], min(top, self.monitor["top"] + self.screen_h - 1))
            max_w = max(1, (self.monitor["left"] + self.screen_w) - left)
            max_h = max(1, (self.monitor["top"] + self.screen_h) - top)
            width = max(1, min(width, max_w))
            height = max(1, min(height, max_h))

            return {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
        except Exception as exc:
            logger.warning("Invalid region %s passed to _normalize_rect_dict, using full monitor: %s", rect, exc)
            return dict(self.monitor)

    @property
    def last_validation(self) -> Tuple[bool, str]:
        """F-0414: son doğrulama sonucu (validate_frame tarafından yazılır)."""
        return self._last_valid, self._last_validation_msg

    @property
    def last_capture_failed(self) -> bool:
        """Son yakalama denemesi arızayla mı bitti (siyah yedek kare dönüldü mü)."""
        return self._last_grab_failed

    @property
    def last_frame_age_s(self) -> Optional[float]:
        """Son başarılı yakalamadan beri geçen süre (saniye); hiç kare
        yoksa None. Bayat kareyi çağıran ayırt edebilsin diye (A17)."""
        if self._last_grab_ok_ts <= 0.0:
            return None
        return max(0.0, time.time() - self._last_grab_ok_ts)

    def close(self) -> None:
        """F-0416: bu thread'in değil, oluşturulmuş TÜM mss örneklerini kapatır.

        P2-4 (K-17): set'ten çıkarılan örnekler `_get_sct()` tarafından
        artık canlı sayılmaz — diğer thread'lerin thread-local'inde
        kalan bayat referans, üyelik denetiminde elenip taze örnekle
        değiştirilir (threading.local başka thread'den temizlenemez).
        close() bırakma (release) anlamındadır: sonraki grab taze
        örnekle yeniden açılır.
        """
        with self._sct_lock:
            instances = list(self._sct_instances)
            self._sct_instances.clear()
        for sct in instances:
            try:
                sct.close()
            except Exception as exc:
                logger.debug("Error closing mss instance: %s", exc)
        try:
            self._thread_local.sct = None
        except Exception:
            pass

    def get_capture_backend_summary(self) -> dict:
        """Returns a safe summary dictionary of the capture backend status."""
        try:
            sct = self._get_sct()
            monitors = getattr(sct, "monitors", None) or []
            return {
                "backend_ready": len(monitors) > 0,
                "backend": "mss",
                "monitor_count": len(monitors),
                "last_capture_failed": self._last_grab_failed,
                "last_capture_error": self._last_grab_error,
            }
        except Exception as exc:
            return {"backend_ready": False, "backend": "mss", "error": str(exc)}

    def _compute_ready(self) -> Tuple[bool, str]:
        """Gerçek hazırlık kontrolü: mss monitörleri + son yakalama durumu."""
        try:
            sct = self._get_sct()
            monitors = getattr(sct, "monitors", None) or []
            if not monitors:
                return False, "no monitors reported by mss"
        except Exception as exc:
            return False, f"mss unavailable: {exc}"
        if self._last_grab_failed:
            return False, f"last capture failed: {self._last_grab_error or 'unknown error'}"
        if self.ring_buffer.frame_sequence == 0 and self._last_grab_ok_ts == 0.0:
            return False, "no frame captured yet"
        if not self._last_valid:
            return False, f"last frame invalid: {self._last_validation_msg}"
        return True, "OK"

    def get_capture_health_summary(self) -> dict:
        # F-0064: koşulsuz "ready: True" yerine gerçek kontrol.
        ready, reason = self._compute_ready()
        return {
            "module": "capture",
            "ready": ready,
            "reason": reason,
            "last_capture_failed": self._last_grab_failed,
            "frames_captured": self.ring_buffer.frame_sequence,
            # A17: bayat kare teşhisi — yaş + hata sayacı.
            "last_frame_age_s": self.last_frame_age_s,
            "grab_error_count": self._grab_error_count,
        }

    def get_capture_ready_flag(self) -> dict:
        ready, reason = self._compute_ready()
        return {"module": "capture", "ready": ready, "reason": reason}

    def get_capture_version_info(self) -> dict:
        ready, reason = self._compute_ready()
        return {"module": "capture", "version": __version__, "ready": ready, "reason": reason}

    def get_capture_diagnostic_snapshot(self) -> dict:
        ready, reason = self._compute_ready()
        return {
            "module": "capture",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "reason": reason,
            "last_validation": self._last_validation_msg,
            "grab_error_count": self._grab_error_count,
        }


def get_capture_capabilities() -> list:
    return ["capture", "screen_grabber", "diagnostic_snapshot"]


def _probe_capture_backend() -> bool:
    """Gerçek backend probu (S-0201): yetenek listesi + mss monitör sorgusu.

    Liste sabiti hiç boşalmadığı için `bool(liste)` totolojiydi; prob
    `mss` örneği açıp gerçek monitör listesini okur. Ekransız / mss
    açılamayan ortamda False döner — yakalama gerçekten yoktur.
    """
    try:
        if not get_capture_capabilities():
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with mss.mss() as sct:
                monitors = getattr(sct, "monitors", None) or []
                return len(monitors) > 0
    except Exception:
        return False


def validate_capture_contract() -> bool:
    # S-0201 (Karar 3): totoloji değil — gerçek mss backend probuna delege.
    return _probe_capture_backend()


def get_capture_baseline_metrics() -> dict:
    return {"module": "capture", "contract_valid": validate_capture_contract(), "capability_count": len(get_capture_capabilities())}


def get_capture_readiness_status() -> dict:
    return {"module": "capture", "ready": validate_capture_contract(), "metrics": get_capture_baseline_metrics()}


def get_capture_audit_record() -> dict:
    return {"module": "capture", "audit_passed": validate_capture_contract(), "readiness": get_capture_readiness_status()}


def get_capture_operational_digest() -> dict:
    return {"module": "capture", "operational": validate_capture_contract(), "audit": get_capture_audit_record()}


def get_capture_telemetry_payload() -> dict:
    return {"module": "capture", "payload": get_capture_operational_digest()}


def get_capture_integrity_seal() -> dict:
    return {"module": "capture", "sealed": validate_capture_contract(), "telemetry": get_capture_telemetry_payload()}


def get_capture_compliance_certificate() -> dict:
    return {"module": "capture", "certified": validate_capture_contract(), "seal": get_capture_integrity_seal()}


def get_capture_readiness_probe() -> dict:
    return {"module": "capture", "ready": validate_capture_contract(), "certificate": get_capture_compliance_certificate()}


def get_capture_audit_summary() -> dict:
    return {"module": "capture", "audit_passed": validate_capture_contract(), "probe": get_capture_readiness_probe()}



