# -*- coding: utf-8 -*-
"""
Stealth Capture Isolation, Adaptive Frame Pacing, and Click-Through Window Privacy
(core/stealth_capture.py)

Layer II implementation ensuring zero access to game process memory (`OpenProcess`,
`ReadProcessMemory`) or game files, capturing exclusively from the desktop output
buffer via DXGI Desktop Duplication (`dxcam`) or thread-safe `mss` fallback,
pacing capture intervals with state-aware Gaussian jitter, and cloaking the PyQt6
overlay window with safe system identities and `WS_EX_TRANSPARENT` click-through
styles.
"""

from __future__ import annotations

import ctypes
import logging
import math
import random
import statistics
import sys
import threading
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mss
import numpy as np

from core.capture import SHARED_BUFFER_HEIGHT, SHARED_BUFFER_WIDTH, ZeroCopyFrameRingBuffer
from version import __version__

try:
    import dxcam  # type: ignore

    _HAS_DXCAM = True
except Exception:
    dxcam = None  # type: ignore
    _HAS_DXCAM = False

logger = logging.getLogger(__name__)

# D03: normalize_illumination her çağrıda cv2.createCLAHE yapıyordu (çağrı
# başına ~2 ms + tekrarlı native nesne kurulumu). CLAHE parametreleri sabittir,
# tek paylaşılan örnek yeter; apply çağrıları kilit altında serileştirilir.
_CLAHE_LOCK: threading.RLock = threading.RLock()
_CLAHE: Optional[Any] = None


def _get_clahe() -> Any:
    """Returns the shared CLAHE instance (clipLimit=2.2, tile 8x8)."""
    global _CLAHE
    if _CLAHE is None:
        with _CLAHE_LOCK:
            if _CLAHE is None:
                _CLAHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    return _CLAHE


# D03: varyans kontrolleri için örnekleme üst sınırı. match_template_tolerant
# kareyi float64'e çevirip tam std alıyordu (1080p BGR'de ~625 ms ve ~50 MB
# geçici dizi); uint8 örnek std aynı "sabit mi?" kararını ~5 ms'de verir.
# Şablonlar genelde küçük kalır (stride 1, tam std, davranış değişmez).
_MAX_STD_SAMPLE_ELEMENTS: int = 262144


def _sampled_std(arr: np.ndarray) -> float:
    """Standard deviation over a systematic spatial sample of `arr`.

    Exact (stride 1) below the sample cap; above it, a strided view with at
    most `_MAX_STD_SAMPLE_ELEMENTS` elements. A constant array still yields
    exactly 0.0 at any stride, so the zero-variance guard keeps its verdict.
    """
    if arr is None or arr.size == 0:
        return 0.0
    if arr.size <= _MAX_STD_SAMPLE_ELEMENTS:
        return float(np.std(arr))
    stride = max(1, int(math.ceil(math.sqrt(arr.size / _MAX_STD_SAMPLE_ELEMENTS))))
    return float(np.std(arr[::stride, ::stride]))

# Windows Extended Window Style Constants
GWL_EXSTYLE: int = -20
WS_EX_TRANSPARENT: int = 0x00000020
WS_EX_TOOLWINDOW: int = 0x00000080
WS_EX_LAYERED: int = 0x00080000
WS_EX_NOACTIVATE: int = 0x08000000

SWP_NOSIZE: int = 0x0001
SWP_NOMOVE: int = 0x0002
SWP_NOZORDER: int = 0x0004
SWP_NOACTIVATE: int = 0x0010
SWP_FRAMECHANGED: int = 0x0020

SAFE_SYSTEM_IDENTITIES: Tuple[str, ...] = (
    "NVIDIA GeForce Overlay",
    "Microsoft Game Bar Overlay",
    "Realtek HD Audio Monitor",
    "Radeon Software Host",
    "Discord Overlay Host",
)

FORBIDDEN_TITLE_KEYWORDS: Tuple[str, ...] = (
    "bot",
    "gather",
    "aion",
    "python",
    "cheat",
    "hack",
    "macro",
    "auto",
)


class DXGIDesktopCapture:
    """
    Isolated desktop buffer capture engine.

    Reads frames exclusively from the GPU desktop duplication output (`dxcam` when
    available, falling back cleanly to thread-local `mss`) without hooking any game
    window (`HWND`) via `BitBlt` or touching game process memory. Also provides
    illumination/opacity normalization and tolerant template matching with dynamic
    thresholds between 75% and 85%.
    """

    def __init__(
        self,
        monitor_idx: int = 1,
        prefer_dxcam: bool = True,
        min_threshold: float = 0.75,
        max_threshold: float = 0.85,
    ) -> None:
        self.monitor_idx: int = max(1, int(monitor_idx))
        self.prefer_dxcam: bool = bool(prefer_dxcam)
        self.min_threshold: float = float(np.clip(min_threshold, 0.50, 0.95))
        self.max_threshold: float = float(np.clip(max_threshold, self.min_threshold, 0.99))

        self._lock: threading.RLock = threading.RLock()
        self._thread_local: threading.local = threading.local()
        # F-1842: close() tüm thread'lerin mss örneklerini kapatabilsin.
        self._sct_instances: set = set()
        self._dxcam_camera: Optional[Any] = None
        self._backend_name: str = "mss"
        self._last_frame: Optional[np.ndarray] = None
        self._last_capture_ts: float = 0.0
        # F-1837: yakalama arızası siyah kare olarak gizlenmesin.
        self._last_capture_ok: bool = False
        self._last_capture_error: str = ""
        self._capture_error_count: int = 0
        # F-1848: son grab bölge grab'i ise halka bayat tam kare tutar.
        self._last_grab_was_region: bool = False
        self.monitor_geometry: Dict[str, int] = {
            "left": 0,
            "top": 0,
            "width": 1920,
            "height": 1080,
        }
        self.ring_buffer: ZeroCopyFrameRingBuffer = ZeroCopyFrameRingBuffer(
            width=SHARED_BUFFER_WIDTH,
            height=SHARED_BUFFER_HEIGHT,
            channels=3,
            ring_size=8,
        )

        # Optional lazy-loaded DPI / Focus helpers
        self._dpi_synchronizer: Optional[Any] = None
        self._focus_guard: Optional[Any] = None

        self._initialize_backend()

    def get_dpi_topology_info(self) -> Dict[str, Any]:
        """
        Lazy-loads `MonitorTopologyScaler` (or `DPIScalingSynchronizer`) and
        `WindowFocusLossSuspender` (or `ForegroundFocusGuard`) from `core.dpi_focus_guard`
        (wrapped in try/except) and returns monitor scale and focus status dict.
        """
        # F-1853: yardımcılar yüklenemezse varsayılan "odaklı / 1.0x" raporu
        # verilmez; bilinmeyen durum bilinmeyen olarak raporlanır.
        result: Dict[str, Any] = {
            "scale_factor": None,
            "scale_percent": None,
            "dpi": None,
            "awareness_level": "unknown",
            "is_focused": None,
            "focus_info": "unknown — FocusGuard unavailable",
            "latency_ms": 0.0,
        }

        # Lazy-load DPI synchronizer
        if self._dpi_synchronizer is None:
            try:
                from core.dpi_focus_guard import DPIScalingSynchronizer
                self._dpi_synchronizer = DPIScalingSynchronizer()
            except Exception as exc:
                try:
                    from core.dpi_focus_guard import MonitorTopologyScaler
                    self._dpi_synchronizer = MonitorTopologyScaler()
                except Exception as exc2:
                    logger.debug("Could not lazy-load DPIScalingSynchronizer / MonitorTopologyScaler: %s, %s", exc, exc2)

        if self._dpi_synchronizer is not None:
            try:
                diag_fn = getattr(self._dpi_synchronizer, "get_diagnostics", None)
                if callable(diag_fn):
                    diag = diag_fn()
                    if isinstance(diag, dict):
                        result.update(diag)
                else:
                    scale_fn = getattr(self._dpi_synchronizer, "get_scale_factor", None)
                    if callable(scale_fn):
                        scale = float(scale_fn())
                        result["scale_factor"] = scale
                        result["scale_percent"] = int(round(scale * 100))
                        result["dpi"] = int(round(scale * 96))
            except Exception as exc:
                logger.debug("Error getting DPI diagnostics: %s", exc)

        # Lazy-load Focus Guard / Suspender
        if self._focus_guard is None:
            try:
                from core.dpi_focus_guard import ForegroundFocusGuard
                self._focus_guard = ForegroundFocusGuard()
            except Exception as exc:
                try:
                    from core.dpi_focus_guard import WindowFocusLossSuspender
                    self._focus_guard = WindowFocusLossSuspender()
                except Exception as exc2:
                    logger.debug("Could not lazy-load ForegroundFocusGuard / WindowFocusLossSuspender: %s, %s", exc, exc2)

        if self._focus_guard is not None:
            try:
                focus_fn = getattr(self._focus_guard, "is_game_window_focused", None)
                if not callable(focus_fn):
                    focus_fn = getattr(self._focus_guard, "check_focus", None)

                if callable(focus_fn):
                    res = focus_fn()
                    if isinstance(res, tuple) and len(res) >= 2:
                        result["is_focused"] = bool(res[0])
                        result["focus_info"] = str(res[1])
                    elif isinstance(res, bool):
                        result["is_focused"] = res
                
                lat_prop = getattr(self._focus_guard, "last_check_latency_ms", None)
                if lat_prop is not None:
                    result["latency_ms"] = float(lat_prop)
            except Exception as exc:
                logger.debug("Error getting focus diagnostics: %s", exc)

        return result

    def _initialize_backend(self) -> None:
        """Initializes DXGI Desktop Duplication if available, else thread-safe MSS."""
        with self._lock:
            # S-0203: mss örneği açılamazsa bunu ayırt et — geometri sorgusu
            # hatası mss'in kendisinin kurulamadığı anlamına gelmez, yalnızca
            # mss.mss() çağrısının patlaması backend'i kullanılamaz yapar.
            mss_available = False
            mss_error = ""
            try:
                sct = self._get_mss()
                mss_available = True
                monitors = getattr(sct, "monitors", None) or []
                if len(monitors) > self.monitor_idx:
                    mon = dict(monitors[self.monitor_idx])
                elif len(monitors) > 1:
                    mon = dict(monitors[1])
                elif len(monitors) == 1:
                    mon = dict(monitors[0])
                else:
                    mon = {"left": 0, "top": 0, "width": 1920, "height": 1080}

                self.monitor_geometry = {
                    "left": int(mon.get("left", 0)),
                    "top": int(mon.get("top", 0)),
                    "width": max(1, int(mon.get("width", 1920))),
                    "height": max(1, int(mon.get("height", 1080))),
                }
            except Exception as exc:
                if not mss_available:
                    mss_error = str(exc)
                logger.warning("Failed to query initial monitor geometry via mss: %s", exc)

            if self.prefer_dxcam and _HAS_DXCAM and dxcam is not None:
                try:
                    output_idx = max(0, self.monitor_idx - 1)
                    camera = dxcam.create(output_idx=output_idx, output_color="BGR")
                    if camera is not None:
                        self._dxcam_camera = camera
                        self._backend_name = "dxcam"
                        logger.info(
                            "DXGIDesktopCapture initialized with DXGI Desktop Duplication (output_idx=%d).",
                            output_idx,
                        )
                        return
                except Exception as exc:
                    logger.info(
                        "DXGI dxcam initialization unavailable (%s); falling back to isolated MSS desktop capture.",
                        exc,
                    )

            # S-0203: ne dxcam ne mss kurulabildiyse backend "mss" diye
            # raporlanmaz — hiç kurulmamışken hazır görünürdü. "unavailable"
            # _compute_ready'de False üretir; grab_desktop zaten bu durumda
            # arızayı yükseltir (son geçerli kare yoksa RuntimeError).
            if not mss_available:
                self._dxcam_camera = None
                self._backend_name = "unavailable"
                logger.error(
                    "DXGIDesktopCapture backend unavailable: mss could not be "
                    "created (%s) and dxcam is not available.",
                    mss_error or "unknown error",
                )
                return
            self._dxcam_camera = None
            self._backend_name = "mss"
            logger.info(
                "DXGIDesktopCapture initialized with MSS desktop buffer backend (monitor_idx=%d).",
                self.monitor_idx,
            )

    def _get_mss(self) -> Any:
        """Returns the current thread's isolated `mss.mss()` instance."""
        sct = getattr(self._thread_local, "sct", None)
        if sct is not None:
            # P2-4 (K-17): close() başka thread'in thread-local'ini
            # temizleyemez (threading.local yalnızca sahibi thread'den
            # yazılır) — ama close() örneği _sct_instances'tan çıkarır.
            # Kayıtta olmayan örnek kapalıdır/atılmıştır: bayat örneği
            # düşürüp taze örnek açılır; yoksa close() sonrası ilk grab
            # her thread'de kapalı handle'a çarpıp RuntimeError verirdi.
            with self._lock:
                if sct in self._sct_instances:
                    return sct
            self._thread_local.sct = None
            sct = None
        if sct is None:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    sct = mss.mss()
                self._thread_local.sct = sct
                with self._lock:
                    self._sct_instances.add(sct)
                    # close() bırakma (release) anlamındadır: kapalı
                    # backend, taze örnekle yeniden canlanır.
                    if self._backend_name == "closed":
                        self._backend_name = "mss"
            except Exception as exc:
                logger.error("Unable to create thread-local MSS instance: %s", exc)
                raise
        return sct

    @property
    def last_capture_ok(self) -> bool:
        """Son yakalama denemesi gerçek bir kare üretti mi (F-1837)."""
        return self._last_capture_ok

    @property
    def last_frame_age_s(self) -> Optional[float]:
        """Son başarılı yakalamadan beri geçen süre (saniye); hiç kare
        yoksa None. Bayat kareyi çağıran ayırt edebilsin diye (A17)."""
        if self._last_capture_ts <= 0.0:
            return None
        return max(0.0, time.perf_counter() - self._last_capture_ts)

    @property
    def backend_name(self) -> str:
        """Returns the active desktop capture backend identifier (`'dxcam'`, `'mss'`, `'unavailable'` or `'closed'`)."""
        return self._backend_name

    def grab_desktop(
        self,
        region: Optional[Union[Dict[str, int], Tuple[int, int, int, int]]] = None,
    ) -> np.ndarray:
        """
        Captures a BGR frame directly from the desktop display buffer without
        attaching to any game window handle (`HWND`).

        Parameters
        ----------
        region : dict or tuple, optional
            Either `{"left": x, "top": y, "width": w, "height": h}` in absolute screen
            coordinates or `(left, top, right, bottom)` tuple relative to the desktop.
        """
        bbox = self._normalize_region(region)
        frame: Optional[np.ndarray] = None

        if self._dxcam_camera is not None:
            try:
                dxcam_region = (
                    max(0, bbox["left"] - self.monitor_geometry["left"]),
                    max(0, bbox["top"] - self.monitor_geometry["top"]),
                    max(1, bbox["left"] - self.monitor_geometry["left"] + bbox["width"]),
                    max(1, bbox["top"] - self.monitor_geometry["top"] + bbox["height"]),
                )
                grabbed = self._dxcam_camera.grab(region=dxcam_region)
                if grabbed is not None and isinstance(grabbed, np.ndarray) and grabbed.size > 0:
                    frame = np.ascontiguousarray(grabbed[:, :, :3], dtype=np.uint8)
            except Exception as exc:
                logger.debug("DXGI grab failed on frame, falling back to MSS buffer: %s", exc)

        if frame is None:
            try:
                sct = self._get_mss()
                raw = sct.grab(bbox)
                bgra = np.asarray(raw, dtype=np.uint8)
                frame = np.ascontiguousarray(bgra[:, :, :3], dtype=np.uint8)
            except Exception as exc:
                # F-1837: yakalama tamamen başarısız olunca siyah kare gerçek
                # kareymiş gibi dönülüp ring buffer'a yazılmaz. Son geçerli
                # kare varsa o döner; yoksa arıza yükseltilir.
                self._last_capture_ok = False
                self._last_capture_error = str(exc)
                self._capture_error_count += 1
                logger.error(
                    "Desktop buffer capture failed (#%d): %s",
                    self._capture_error_count,
                    exc,
                )
                if self._last_frame is not None:
                    logger.warning(
                        "grab_desktop: geçerli kare alınamadı; son geçerli "
                        "kare döndürülüyor (ring buffer'a yazılmadı)."
                    )
                    return self._last_frame
                raise RuntimeError(f"Desktop capture failed: {exc}") from exc

        self._last_capture_ok = True
        self._last_capture_error = ""
        with self._lock:
            if region is None:
                shared_view = self.ring_buffer.write_frame(frame)
                self._last_frame = shared_view
                self._last_grab_was_region = False
            else:
                # F-1848: bölge grab'i tam kare halkasını güncellemez; bu
                # bayrak grab_roi_view'un bayat kareden okumasını engeller.
                self._last_frame = frame
                self._last_grab_was_region = True
            self._last_capture_ts = time.perf_counter()

        return self._last_frame if self._last_frame is not None else frame

    def grab_frame(self, grayscale: bool = False) -> np.ndarray:
        """
        Item 41 — Captures a full desktop frame directly into the pre-allocated
        `ZeroCopyFrameRingBuffer` and returns a read-only zero-copy NumPy view
        (`get_shared_frame_view()`) so `VisionEngine`, `AutoCartographer`, and
        `MobileRemoteServer` share the same memory reference without copying 14MB arrays.
        """
        shared_view = self.grab_desktop(region=None)
        if grayscale:
            return cv2.cvtColor(shared_view, cv2.COLOR_BGR2GRAY)
        return shared_view

    def get_shared_frame_view(self) -> np.ndarray:
        """
        Returns a read-only zero-copy NumPy view (`flags.writeable == False`) of the
        latest frame in the shared ring buffer (`ZeroCopyFrameRingBuffer`).
        """
        if self.ring_buffer.frame_sequence == 0:
            self.grab_desktop(region=None)
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
        Returns a zero-copy read-only slice view `[y1:y2, x1:x2]` directly from the
        shared ring buffer (`ZeroCopyFrameRingBuffer`). Supports both pixel coordinates
        and normalized `[0.0 .. 1.0]` coordinates.
        """
        # F-1848: son grab bölge grab'i ise halka bayat tam kare tutar;
        # ROI okumadan önce tam kare tazelenir.
        if refresh or self.ring_buffer.frame_sequence == 0 or self._last_grab_was_region:
            self.grab_desktop(region=None)
        return self.ring_buffer.grab_roi_view(x1, y1, x2, y2)

    def _normalize_region(
        self,
        region: Optional[Union[Dict[str, int], Tuple[int, int, int, int]]],
    ) -> Dict[str, int]:
        """Normalizes a region specification into an MSS-compatible bounding dictionary.

        Tuple anlamı — TEK tanım (P3-7, v18): tuple HER ZAMAN
        ``(left, top, right, bottom)`` — mutlak ekran koordinatlarında
        sol/üst ve sağ/alt köşe; ``(x, y, w, h)`` olarak okunmaz.
        Eskiden heuristik vardı: üçüncü/dördüncü değer ilk ikisinden
        büyükse sağ/alt, değilse genişlik/yükseklik sayılırdı — örn.
        ``(100, 100, 200, 150)`` girdisi (x, y, w, h) ise 200x150'lik
        bölge demekken sessizce 100x50'lik bölgeye dönüşüyordu.
        grab_desktop'un docstring'i zaten (left, top, right, bottom)
        diyordu; kod şimdi docstring ile aynı tanımı kullanıyor.
        Bozuk tuple (right <= left veya bottom <= top) tüm monitöre düşer
        (diğer biçimsiz girdilerle aynı kural).
        """
        if region is None:
            return dict(self.monitor_geometry)

        geo = self.monitor_geometry
        gx, gy = int(geo.get("left", 0)), int(geo.get("top", 0))
        gw, gh = int(geo.get("width", 1920)), int(geo.get("height", 1080))

        def _clamp(left: int, top: int, width: int, height: int) -> Dict[str, int]:
            # F-1847: dict dalı hiç clamp yapmıyordu; bölge monitör
            # sınırları içine çekilir.
            left = max(gx, min(left, gx + gw - 1))
            top = max(gy, min(top, gy + gh - 1))
            width = max(1, min(width, gx + gw - left))
            height = max(1, min(height, gy + gh - top))
            return {"left": left, "top": top, "width": width, "height": height}

        if isinstance(region, dict):
            left = int(region.get("left", region.get("x", gx)))
            top = int(region.get("top", region.get("y", gy)))
            width = max(1, int(region.get("width", region.get("w", gw))))
            height = max(1, int(region.get("height", region.get("h", gh))))
            return _clamp(left, top, width, height)

        if isinstance(region, (tuple, list)) and len(region) == 4:
            left, top, right, bottom = (int(v) for v in region)
            # Tek tanım: (left, top, right, bottom). Ters/bozuk kutu
            # (right <= left veya bottom <= top) tüm monitöre düşer.
            if right <= left or bottom <= top:
                logger.debug(
                    "Degenerate region tuple %r; falling back to full monitor geometry",
                    region,
                )
                return dict(self.monitor_geometry)
            return _clamp(left, top, max(1, right - left), max(1, bottom - top))

        return dict(self.monitor_geometry)

    @staticmethod
    def normalize_illumination(image: np.ndarray) -> np.ndarray:
        """
        Normalizes lighting, gamma, and semi-transparent UI opacity variations
        using CLAHE on the grayscale luminance channel so template
        matching remains resilient to day/night cycles and UI alpha blending.
        (The 3D input path used to round-trip through LAB; the output was
        grayscale either way, so CLAHE now runs directly on gray.)
        """
        try:
            if image is None or image.size == 0:
                return np.zeros((1, 1), dtype=np.uint8)

            if image.ndim == 3 and image.shape[2] >= 3:
                # D03: eski yol BGR->LAB->split->CLAHE(L)->merge->LAB->BGR->
                # GRAY yapıyordu (~450 ms @1080p) ama sonuç yine gri tondu;
                # a/b kanalları CLAHE sonrası atılıyordu. Çıktı zaten gri
                # olduğundan CLAHE doğrudan gri ton üzerinde uygulanır
                # (~115 ms @1080p); 2D dalı zaten hep böyle çalışıyordu.
                gray_in = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
                clahe = _get_clahe()
                with _CLAHE_LOCK:
                    gray = clahe.apply(gray_in)
            elif image.ndim == 2:
                clahe = _get_clahe()
                with _CLAHE_LOCK:
                    gray = clahe.apply(image)
            else:
                gray = np.asarray(image, dtype=np.uint8)

            normalized = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            return np.ascontiguousarray(normalized, dtype=np.uint8)
        except Exception as exc:
            logger.error("Illumination normalization failed: %s", exc)
            if image is not None and image.ndim == 3:
                return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)
            return np.asarray(image if image is not None else np.zeros((1, 1), dtype=np.uint8), dtype=np.uint8)

    def compute_dynamic_threshold(
        self,
        frame_gray: np.ndarray,
        base_threshold: Optional[float] = None,
    ) -> float:
        """
        Calculates a dynamic template-matching threshold clamped strictly within
        [0.75, 0.85] (or `[self.min_threshold, self.max_threshold]`) based on local
        contrast/illumination variance. Lower contrast or high alpha transparency
        relaxes the threshold toward 0.75; crisp contrast tightens it toward 0.85.
        """
        try:
            target_base = (
                float(base_threshold)
                if base_threshold is not None
                else 0.5 * (self.min_threshold + self.max_threshold)
            )
            if frame_gray is None or frame_gray.size == 0:
                return float(np.clip(target_base, self.min_threshold, self.max_threshold))

            # D03: tam kare std yerine örneklenmiş std (1080p gri: ~15 ms -> ~4 ms).
            std_dev = _sampled_std(frame_gray)
            # Typical UI grayscale std_dev spans ~20 (low contrast/translucent) to ~75 (sharp contrast)
            contrast_factor = float(np.clip((std_dev - 20.0) / 55.0, 0.0, 1.0))
            dynamic_thr = self.min_threshold + contrast_factor * (self.max_threshold - self.min_threshold)
            # F-1849: eşik neredeyse hiç dinamik değildi (0.5 dinamik +
            # 0.5 çağıran eşiği). Dinamik bileşen baskındır; çağıran eşiği
            # yalnızca hafifçe çeker.
            blended = 0.75 * dynamic_thr + 0.25 * target_base
            return float(np.clip(blended, self.min_threshold, self.max_threshold))
        except Exception as exc:
            logger.debug("compute_dynamic_threshold error: %s", exc)
            return float(0.5 * (self.min_threshold + self.max_threshold))

    def match_template_tolerant(
        self,
        frame: np.ndarray,
        template: np.ndarray,
        threshold: Optional[float] = None,
        normalize_lighting: bool = True,
    ) -> Tuple[bool, float, Tuple[int, int], float]:
        """
        Performs illumination-normalized template matching using dynamic thresholds
        in the 75% - 85% range (`0.75 - 0.85`).

        Returns
        -------
        tuple
            `(matched: bool, max_val: float, max_loc: Tuple[int, int], effective_threshold: float)`
        """
        try:
            if frame is None or template is None or frame.size == 0 or template.size == 0:
                return False, 0.0, (-1, -1), self.min_threshold

            # F-1840: siyah/boş (varyanssız) kare boş şablonla %100 eşleşirdi.
            # Varyanssız kare veya şablon eşleşme sayılmaz.
            # D03 + A-üst: önce örneklenmiş uint8 std ile hızlı yoklama
            # (1080p'de ~625 ms -> ~5 ms); örnek 0.0 verirse tam std ile
            # doğrulanır — seyrek varyanslı kare örneklemede 0.0 görünebilir.
            # Tam std uint8 üzerinde hesaplanır: float64 kopyasıyla bit-bit
            # aynı sonuç verir (sıfır karede 0.0), kesin kararı o verir.
            if _sampled_std(np.asarray(frame)) == 0.0 and float(np.std(frame)) == 0.0:
                return False, 0.0, (-1, -1), self.min_threshold
            if _sampled_std(np.asarray(template)) == 0.0 and float(np.std(template)) == 0.0:
                return False, 0.0, (-1, -1), self.min_threshold

            if normalize_lighting:
                proc_frame = self.normalize_illumination(frame)
                proc_tmpl = self.normalize_illumination(template)
            else:
                proc_frame = (
                    cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
                    if frame.ndim == 3
                    else np.asarray(frame, dtype=np.uint8)
                )
                proc_tmpl = (
                    cv2.cvtColor(template[:, :, :3], cv2.COLOR_BGR2GRAY)
                    if template.ndim == 3
                    else np.asarray(template, dtype=np.uint8)
                )

            fh, fw = proc_frame.shape[:2]
            th, tw = proc_tmpl.shape[:2]
            if th > fh or tw > fw or th < 2 or tw < 2:
                return False, 0.0, (-1, -1), self.min_threshold

            effective_threshold = self.compute_dynamic_threshold(proc_frame, base_threshold=threshold)
            res = cv2.matchTemplate(proc_frame, proc_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            score = float(max_val)
            if math.isnan(score) or math.isinf(score):
                score = 0.0

            matched = score >= effective_threshold
            # F-1850: başarısızlıkta (0, 0) geçerli bir konum gibi dönülmez;
            # eşleşme yoksa konum (-1, -1) sentinel'idir.
            loc = (int(max_loc[0]), int(max_loc[1])) if matched else (-1, -1)
            return matched, score, loc, effective_threshold
        except Exception as exc:
            logger.error("Tolerant template match failed: %s", exc)
            return False, 0.0, (-1, -1), self.min_threshold

    def close(self) -> None:
        """Releases any held desktop duplication or MSS resources.

        P2-4 (K-17): set'ten çıkarılan örnekler `_get_mss()` tarafından
        artık canlı sayılmaz — diğer thread'lerin thread-local'inde
        kalan bayat referans, üyelik denetiminde elenip taze örnekle
        değiştirilir (threading.local başka thread'den temizlenemez).
        close() bırakma (release) anlamındadır: sonraki grab taze
        örnekle yeniden açılır ve backend adı "mss"e döner.
        """
        with self._lock:
            if self._dxcam_camera is not None:
                try:
                    stop_fn = getattr(self._dxcam_camera, "stop", None)
                    if callable(stop_fn):
                        stop_fn()
                except Exception as exc:
                    logger.debug("Error stopping dxcam camera: %s", exc)
                self._dxcam_camera = None

            # F-1842: yalnızca çağıran thread'in örneği değil, oluşturulmuş
            # tüm mss örnekleri kapatılır; backend adı da "closed" olur.
            instances = list(self._sct_instances)
            self._sct_instances.clear()
            for sct in instances:
                try:
                    sct.close()
                except Exception as exc:
                    logger.debug("Error closing MSS instance: %s", exc)
            try:
                self._thread_local.sct = None
            except Exception:
                pass
            self._backend_name = "closed"

    def _compute_ready(self) -> Tuple[bool, str]:
        """Gerçek hazırlık kontrolü: backend + son yakalama durumu."""
        if self._backend_name == "closed":
            return False, "capture backend closed"
        if self._backend_name not in ("mss", "dxcam"):
            return False, f"capture backend unavailable: {self._backend_name}"
        # S-0203: hiç kare alınmamışken hazır sayılmaz (kardeş sınıf
        # ScreenCapture._compute_ready ile aynı kural: "no frame captured yet").
        # Önceden bu dal yoktu: _last_capture_ts yalnızca başarı yolunda
        # atandığı için başlangıçta ts=0.0, ok=False iken ready=True dönüyordu.
        if self._last_capture_ts <= 0.0:
            return False, "no frame captured yet"
        if not self._last_capture_ok:
            return False, f"last capture failed: {self._last_capture_error or 'unknown error'}"
        return True, "OK"

    def get_stealth_capture_health_summary(self) -> dict:
        """Returns stealth capture readiness health summary."""
        # F-1839: koşulsuz True yerine gerçek kontrol.
        ready, reason = self._compute_ready()
        return {
            "stealth_capture_ready": ready,
            "reason": reason,
            "backend": self._backend_name,
            "last_capture_ok": self._last_capture_ok,
            # A17: bayat kare teşhisi — yaş + sıra + hata sayacı.
            "frame_sequence": self.ring_buffer.frame_sequence,
            "last_frame_age_s": self.last_frame_age_s,
            "capture_error_count": self._capture_error_count,
        }

    def get_stealth_capture_ready_flag(self) -> dict:
        ready, reason = self._compute_ready()
        return {"module": "stealth_capture", "ready": ready, "reason": reason}


class AdaptiveFramePacer:
    """
    State-dependent variable FPS frame pacer with Gaussian jitter.

    Eliminates fixed 60.0 FPS (`16.67ms`) robotic capture loops by dynamically
    selecting a Gaussian-jittered target FPS based on the current bot activity state:
      - `WALKING`: 8 - 12 FPS (`83.3ms - 125.0ms`)
      - `APPROACHING` / `GATHERING`: 20 - 30 FPS (`33.3ms - 50.0ms`)
      - `COMBAT`: 12 - 18 FPS (`55.6ms - 83.3ms`)
      - `PAUSED` / `BREAK`: 2 - 4 FPS (`250.0ms - 500.0ms`)

    Bilinmeyen durumlar sessizce WALKING sayılmaz; uyarı loglanır ve
    WALKING aralığı açıkça yedek olarak kullanılır (F-1854).
    """

    DEFAULT_STATE_FPS_RANGES: Dict[str, Tuple[float, float]] = {
        "WALKING": (8.0, 12.0),
        "PATROLLING": (8.0, 12.0),
        "SEARCHING": (8.0, 12.0),
        "APPROACHING": (20.0, 30.0),
        "GATHERING": (20.0, 30.0),
        "LOOTING": (20.0, 30.0),
        # Combat ticks must stay fast enough for ms-scale skill bursts (was 6–10 FPS /
        # ~100–166ms idle between bursts — felt sticky with Tab+soft-aim overhead).
        "COMBAT": (12.0, 18.0),
        "ATTACKING": (12.0, 18.0),
        "FIGHTING": (12.0, 18.0),
        "PAUSED": (2.0, 4.0),
        "BREAK": (2.0, 4.0),
        "AFK": (2.0, 4.0),
        "IDLE": (4.0, 8.0),
    }

    def __init__(
        self,
        custom_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._rng: random.Random = random.Random(seed)
        self.state_fps_ranges: Dict[str, Tuple[float, float]] = dict(self.DEFAULT_STATE_FPS_RANGES)
        if custom_ranges:
            for k, v in custom_ranges.items():
                if isinstance(v, (tuple, list)) and len(v) == 2:
                    lo = max(0.5, float(min(v[0], v[1])))
                    hi = max(lo + 0.1, float(max(v[0], v[1])))
                    self.state_fps_ranges[str(k).upper().strip()] = (lo, hi)

        self._last_frame_ts: float = time.perf_counter()
        self._last_sleep_interval: float = 0.1
        self._last_target_fps: float = 10.0
        self._recent_intervals: List[float] = []
        self._unknown_states_warned: set = set()

    def _resolve_fps_bounds(self, state: str) -> Tuple[float, float]:
        """Maps any state string to its configured `(min_fps, max_fps)` bounds."""
        normalized = str(state or "WALKING").upper().strip()
        if normalized in self.state_fps_ranges:
            return self.state_fps_ranges[normalized]

        # Prefix / substring heuristics for composite states
        if any(tok in normalized for tok in ("APPROACH", "GATHER", "MINE", "HARVEST", "INTERACT")):
            return self.state_fps_ranges["GATHERING"]
        if any(tok in normalized for tok in ("COMBAT", "FIGHT", "ATTACK", "TARGET", "EVADE")):
            return self.state_fps_ranges["COMBAT"]
        if any(tok in normalized for tok in ("PAUSE", "BREAK", "SLEEP", "STOP", "WAIT")):
            return self.state_fps_ranges["PAUSED"]
        if any(tok in normalized for tok in ("WALK", "MOVE", "NAV", "PATROL", "ROAM")):
            return self.state_fps_ranges["WALKING"]

        # F-1854: bilinmeyen durum sessizce WALKING olmaz; bir kez uyarı
        # loglanır, WALKING aralığı açık yedek olarak döner.
        if normalized not in self._unknown_states_warned:
            self._unknown_states_warned.add(normalized)
            logger.warning(
                "AdaptiveFramePacer: bilinmeyen durum '%s'; WALKING aralığı "
                "yedek olarak kullanılıyor.",
                normalized,
            )
        return self.state_fps_ranges["WALKING"]

    def get_next_sleep_interval(self, state: str = "WALKING") -> float:
        """
        Computes the next sleep interval in seconds using a Gaussian distribution
        centered between the state's `[min_fps, max_fps]` range, clamped strictly
        to the state's interval bounds and guaranteed never to lock onto a constant
        60.0 FPS (`16.67ms`) cadence.
        """
        with self._lock:
            try:
                min_fps, max_fps = self._resolve_fps_bounds(state)
                mean_fps = 0.5 * (min_fps + max_fps)
                # Standard deviation spans 1/4 of the range so ~95% of draws fall naturally inside [min_fps, max_fps]
                sigma_fps = max(0.15, (max_fps - min_fps) / 4.0)

                sampled_fps = self._rng.gauss(mean_fps, sigma_fps)
                # Micro-perturbation so two consecutive calls never produce identical floats
                sampled_fps += self._rng.uniform(-0.08, 0.08)
                clamped_fps = max(min_fps, min(max_fps, sampled_fps))

                interval_sec = 1.0 / clamped_fps
                min_interval = 1.0 / max_fps
                max_interval = 1.0 / min_fps
                interval_sec = max(min_interval, min(max_interval, interval_sec))
                # A17: mikro-pertürbasyon clamp ÖNCESİ uygulanıyordu; sınıra
                # takılan çekişlerin hepsi özdeş aralığa düşüyordu (1000
                # örnekte ~39 tekrar). Pertürbasyon clamp SONRASI, sınırda
                # içe dönük uygulanır ki ardışık çağrılar özdeş aralık
                # üretmesin (simetrik pertürbasyon sınırı aşıp yeniden
                # clamp'lenerek yine özdeş değer üretirdi).
                if interval_sec <= min_interval:
                    interval_sec = min_interval + self._rng.uniform(0.0, 0.001)
                elif interval_sec >= max_interval:
                    interval_sec = max_interval - self._rng.uniform(0.0, 0.001)
                else:
                    interval_sec += self._rng.uniform(-0.0005, 0.0005)
                    interval_sec = max(min_interval, min(max_interval, interval_sec))

                # F-1860: eski 60 FPS koruması ölü koddu — tüm durum
                # aralıkları en fazla 30 FPS'tir (en kısa aralık ~33 ms),
                # 16.67 ms harmoniği zaten ulaşılamaz; kaldırıldı.

                self._last_target_fps = 1.0 / interval_sec
                self._last_sleep_interval = interval_sec
                self._recent_intervals.append(interval_sec)
                if len(self._recent_intervals) > 250:
                    self._recent_intervals.pop(0)

                return float(interval_sec)
            except Exception as exc:
                logger.error("Error computing adaptive frame sleep interval: %s", exc)
                return 0.100

    def wait_next_frame(self, state: str = "WALKING") -> float:
        """
        Sleeps for the remaining duration needed to satisfy the state-dependent
        jittered frame interval (accounting for frame processing time already elapsed)
        and returns the actual sleep duration applied in seconds.
        """
        try:
            target_interval = self.get_next_sleep_interval(state)
            now = time.perf_counter()
            with self._lock:
                elapsed = max(0.0, now - self._last_frame_ts)

            remaining = max(0.001, target_interval - elapsed)
            time.sleep(remaining)

            with self._lock:
                self._last_frame_ts = time.perf_counter()

            return float(remaining)
        except Exception as exc:
            logger.error("AdaptiveFramePacer.wait_next_frame error: %s", exc)
            time.sleep(0.05)
            return 0.05

    def get_diagnostics(self) -> Dict[str, float]:
        """Returns summary statistics of recent frame intervals and effective FPS."""
        with self._lock:
            if not self._recent_intervals:
                return {
                    "mean_interval_ms": 0.0,
                    "stdev_interval_ms": 0.0,
                    "mean_fps": 0.0,
                    "last_target_fps": self._last_target_fps,
                }
            intervals_ms = [v * 1000.0 for v in self._recent_intervals]
            mean_ms = statistics.mean(intervals_ms)
            stdev_ms = statistics.pstdev(intervals_ms) if len(intervals_ms) > 1 else 0.0
            mean_fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
            return {
                "mean_interval_ms": round(mean_ms, 3),
                "stdev_interval_ms": round(stdev_ms, 3),
                "mean_fps": round(mean_fps, 2),
                "last_target_fps": round(self._last_target_fps, 2),
            }


class StealthOverlayManager:
    """
    Cloaks the PyQt6 overlay window by sanitizing its window title to a benign
    system utility identity and applying Win32 extended window styles
    (`WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW`) so mouse clicks pass
    straight through the overlay into the underlying game window.
    """

    SAFE_TITLES: Tuple[str, ...] = SAFE_SYSTEM_IDENTITIES

    def __init__(self, deterministic_index: Optional[int] = None) -> None:
        self.deterministic_index: Optional[int] = deterministic_index
        self._rng: random.Random = random.Random()
        self._active_identity: str = SAFE_SYSTEM_IDENTITIES[0]
        self._click_through_enabled: bool = False

    @property
    def click_through_enabled(self) -> bool:
        """F-1858: _click_through_enabled artık bu dosyada okunur."""
        return self._click_through_enabled

    def select_safe_identity(self, deterministic_index: Optional[int] = None) -> str:
        """
        Returns a benign system overlay title from `SAFE_SYSTEM_IDENTITIES`, either
        deterministically by index or randomly on launch.
        """
        idx = deterministic_index if deterministic_index is not None else self.deterministic_index
        if idx is not None:
            chosen = self.SAFE_TITLES[int(idx) % len(self.SAFE_TITLES)]
        else:
            chosen = self._rng.choice(self.SAFE_TITLES)
        self._active_identity = chosen
        return chosen

    @staticmethod
    def is_title_safe(title: str) -> bool:
        """Checks that the window title contains none of the forbidden bot/game keywords."""
        if not title or not title.strip():
            return False
        lowered = title.lower()
        return not any(keyword in lowered for keyword in FORBIDDEN_TITLE_KEYWORDS)

    def apply_stealth_window_identity(
        self,
        widget: Any,
        deterministic_index: Optional[int] = None,
    ) -> str:
        """
        Sanitizes the PyQt6 widget's `WindowTitle` and `ObjectName`, replacing any
        suspicious title (`"Bot"`, `"Gather"`, `"Aion"`, `"Python"`, etc.) with one
        of the safe system identities (`"NVIDIA GeForce Overlay"`,
        `"Microsoft Game Bar Overlay"`, `"Realtek HD Audio Monitor"`,
        `"Radeon Software Host"`, `"Discord Overlay Host"`). Also updates the native
        Win32 window text if a native `HWND` exists.
        """
        chosen_title = self.select_safe_identity(deterministic_index=deterministic_index)
        # F-1861: widget yokken / setter başarısızken bile "uygulandı"
        # dönülmezdi; başarı en az bir setter'ın çalışmasına bağlıdır.
        applied = False
        try:
            if widget is None:
                logger.warning(
                    "apply_stealth_window_identity: widget yok; kimlik uygulanamadı."
                )
                return ""
            set_title_fn = getattr(widget, "setWindowTitle", None)
            if callable(set_title_fn):
                set_title_fn(chosen_title)
                applied = True

            set_obj_name_fn = getattr(widget, "setObjectName", None)
            if callable(set_obj_name_fn):
                safe_obj_name = chosen_title.replace(" ", "") + "Window"
                set_obj_name_fn(safe_obj_name)
                applied = True

            hwnd = self._extract_hwnd(widget)
            if hwnd and sys.platform == "win32":
                try:
                    ctypes.windll.user32.SetWindowTextW(ctypes.wintypes.HWND(hwnd), chosen_title)
                    applied = True
                except Exception as win_exc:
                    logger.debug("SetWindowTextW skipped or failed for HWND=%s: %s", hwnd, win_exc)

            if not applied:
                logger.error(
                    "Stealth window identity uygulanamadı: widget üzerinde "
                    "kullanılabilir setter yok."
                )
                return ""
            logger.info("Applied stealth window identity: '%s'", chosen_title)
        except Exception as exc:
            logger.error("Failed to apply stealth window identity: %s", exc)
            return ""

        return chosen_title

    @staticmethod
    def _extract_hwnd(widget: Any) -> int:
        """Extracts the integer native window handle (`HWND`) from a PyQt6 widget or integer."""
        if widget is None:
            return 0
        if isinstance(widget, int):
            return int(widget)
        try:
            win_id_fn = getattr(widget, "winId", None)
            if callable(win_id_fn):
                wid = win_id_fn()
                if wid is not None:
                    return int(wid)
        except Exception as exc:
            logger.debug("Could not extract winId from widget: %s", exc)
        return 0

    def set_click_through(self, widget: Any, enabled: bool = True) -> bool:
        """
        Applies or removes Win32 click-through extended window flags
        (`WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW`) using
        `ctypes.windll.user32.GetWindowLongW` and `SetWindowLongW`.

        Also synchronizes PyQt6 `WA_TransparentForMouseEvents` attribute on the widget
        so both Qt event dispatch and OS hit-testing pass mouse clicks directly to the
        underlying game window.
        """
        try:
            if widget is not None and not isinstance(widget, int):
                try:
                    from PyQt6.QtCore import Qt

                    set_attr_fn = getattr(widget, "setAttribute", None)
                    if callable(set_attr_fn):
                        set_attr_fn(Qt.WidgetAttribute.WA_TransparentForMouseEvents, bool(enabled))
                except Exception as qt_exc:
                    logger.debug("Qt WA_TransparentForMouseEvents update skipped: %s", qt_exc)

            hwnd = self._extract_hwnd(widget)
            if not hwnd or sys.platform != "win32":
                logger.warning("Cannot apply Win32 SetWindowLongW: invalid HWND (%s) or non-Win32 platform.", hwnd)
                return False

            user32 = ctypes.windll.user32
            current_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))

            if enabled:
                new_style = current_style | WS_EX_TRANSPARENT | WS_EX_LAYERED | WS_EX_TOOLWINDOW
            else:
                # Preserve WS_EX_LAYERED and WS_EX_TOOLWINDOW so the overlay stays out of Alt-Tab,
                # but clear WS_EX_TRANSPARENT so the user can interact with the HUD controls.
                new_style = (current_style | WS_EX_LAYERED | WS_EX_TOOLWINDOW) & ~WS_EX_TRANSPARENT

            user32.SetWindowLongW(hwnd, GWL_EXSTYLE, new_style)
            user32.SetWindowPos(
                hwnd,
                0,
                0,
                0,
                0,
                0,
                SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED,
            )

            verified_style = int(user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
            if verified_style == 0 and new_style != 0:
                # F-1841: doğrulama okuması 0 ise GetWindowLongW başarısız
                # olmuştur; sanal stil yedeğiyle sahte başarı üretilmez.
                logger.error(
                    "set_click_through doğrulaması başarısız: "
                    "GetWindowLongW 0 döndü (HWND=0x%X).",
                    hwnd,
                )
                return False

            is_transparent_set = bool(verified_style & WS_EX_TRANSPARENT)
            is_layered_set = bool(verified_style & WS_EX_LAYERED)
            is_toolwindow_set = bool(verified_style & WS_EX_TOOLWINDOW)

            if enabled:
                success = is_transparent_set and is_layered_set and is_toolwindow_set
            else:
                success = (not is_transparent_set) and is_layered_set and is_toolwindow_set

            self._click_through_enabled = bool(enabled) if success else self._click_through_enabled
            logger.info(
                "StealthOverlayManager.set_click_through(enabled=%s) -> HWND=0x%X style=0x%08X (success=%s)",
                enabled,
                hwnd,
                verified_style,
                success,
            )
            return success
        except Exception as exc:
            logger.error("Failed to configure click-through window style: %s", exc)
            return False

    def get_window_ex_style(self, widget: Any) -> int:
        """Returns the current Win32 `GWL_EXSTYLE` bitmask for the given widget or `HWND`."""
        try:
            hwnd = self._extract_hwnd(widget)
            if hwnd and sys.platform == "win32":
                style = int(ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE))
                if style != 0:
                    return style
            if widget is not None and not isinstance(widget, int):
                return int(getattr(widget, "_stealth_ex_style", 0))
            return 0
        except Exception as exc:
            logger.error("Failed to read GWL_EXSTYLE: %s", exc)
            return 0

    def get_stealth_capture_ready_flag(self) -> dict:
        # S-0202: koşulsuz True yerine yöneticinin gerçek durumu: aktif
        # kimlik güvenli sistem kimliklerinden biri ve yasak sözcük
        # içermiyor olmalı. Click-through bir moddur (HUD etkileşimliyken
        # kapalı olması meşrudur), hazırlık kapısı değildir; yine de
        # raporlanır. Bu sınıfın yakalama backend'i yoktur; yakalama
        # hazırlığı DXGIDesktopCapture / StealthCapture _compute_ready
        # yolundadır.
        identity_ok = (
            bool(self._active_identity)
            and self._active_identity in self.SAFE_TITLES
            and self.is_title_safe(self._active_identity)
        )
        return {
            "module": "stealth_capture",
            "ready": identity_ok,
            "reason": "OK" if identity_ok else f"unsafe overlay identity: {self._active_identity!r}",
            "click_through_enabled": self._click_through_enabled,
            "active_identity": self._active_identity,
        }


class StealthCapture(DXGIDesktopCapture):
    """Stealth capture facade inheriting isolated desktop duplication capture."""

    def get_stealth_capture_ready_flag(self) -> dict:
        ready, reason = self._compute_ready()
        return {"module": "stealth_capture", "ready": ready, "reason": reason}

    def get_stealth_capture_version_info(self) -> dict:
        ready, reason = self._compute_ready()
        return {"module": "stealth_capture", "version": __version__, "ready": ready, "reason": reason}

    def get_stealth_capture_diagnostic_snapshot(self) -> dict:
        ready, reason = self._compute_ready()
        return {
            "module": "stealth_capture",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "reason": reason,
            "backend": self._backend_name,
            "frame_sequence": self.ring_buffer.frame_sequence,
            "last_frame_age_s": self.last_frame_age_s,
            "capture_error_count": self._capture_error_count,
        }


def get_stealth_capture_capabilities() -> list:
    return ["stealth_capture", "frame_acquisition", "diagnostic_snapshot"]


def _probe_stealth_capture_backend() -> bool:
    """Gerçek backend probu (S-0202): yetenek listesi + mss monitör sorgusu.

    Liste sabiti hiç boşalmadığı için `bool(liste)` totolojiydi; prob
    `mss` örneği açıp gerçek monitör listesini okur (dxcam Windows'a
    özeldir; yedek backend mss'tir). Ekransız / mss açılamayan ortamda
    False döner — yakalama gerçekten yoktur.
    """
    try:
        if not get_stealth_capture_capabilities():
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with mss.mss() as sct:
                monitors = getattr(sct, "monitors", None) or []
                return len(monitors) > 0
    except Exception:
        return False


def validate_stealth_capture_contract() -> bool:
    # S-0202 (Karar 3): totoloji değil — gerçek mss backend probuna delege.
    return _probe_stealth_capture_backend()


def _stealth_capture_contract_ok(_contract_ok: Optional[bool] = None) -> bool:
    """P2-1: zincir ici tek dogrulama (bkz. telemetry.py'deki aciklama).

    ``validate_stealth_capture_contract()`` her cagrida
    ``_probe_stealth_capture_backend()``'i kosar; prob zaten
    ``with mss.mss()`` ile ornegi kapatir (kaynak sizintisi yok),
    zincir bunu ust duzey okuma basina bir kez yapar. Cagrilar arasi
    onbellek yoktur.
    """
    if _contract_ok is not None:
        return bool(_contract_ok)
    return bool(validate_stealth_capture_contract())


def get_stealth_capture_baseline_metrics(_contract_ok: Optional[bool] = None) -> dict:
    return {"module": "stealth_capture", "contract_valid": _stealth_capture_contract_ok(_contract_ok), "capability_count": len(get_stealth_capture_capabilities())}


def get_stealth_capture_readiness_status(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "ready": ok, "metrics": get_stealth_capture_baseline_metrics(_contract_ok=ok)}


def get_stealth_capture_audit_record(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "audit_passed": ok, "readiness": get_stealth_capture_readiness_status(_contract_ok=ok)}


def get_stealth_capture_operational_digest(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "operational": ok, "audit": get_stealth_capture_audit_record(_contract_ok=ok)}


def get_stealth_capture_telemetry_payload(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "telemetry_ready": ok, "digest": get_stealth_capture_operational_digest(_contract_ok=ok)}


def get_stealth_capture_integrity_seal(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "sealed": ok, "telemetry": get_stealth_capture_telemetry_payload(_contract_ok=ok)}


def get_stealth_capture_compliance_certificate(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "certified": ok, "seal": get_stealth_capture_integrity_seal(_contract_ok=ok)}


def get_stealth_capture_readiness_probe(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "ready": ok, "certificate": get_stealth_capture_compliance_certificate(_contract_ok=ok)}


def get_stealth_capture_audit_summary(_contract_ok: Optional[bool] = None) -> dict:
    ok = _stealth_capture_contract_ok(_contract_ok)
    return {"module": "stealth_capture", "audit_passed": ok, "probe": get_stealth_capture_readiness_probe(_contract_ok=ok)}


