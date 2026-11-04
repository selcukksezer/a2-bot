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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import mss
import numpy as np

from core.capture import SHARED_BUFFER_HEIGHT, SHARED_BUFFER_WIDTH, ZeroCopyFrameRingBuffer

try:
    import dxcam  # type: ignore

    _HAS_DXCAM = True
except Exception:
    dxcam = None  # type: ignore
    _HAS_DXCAM = False

logger = logging.getLogger(__name__)

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
        self._dxcam_camera: Optional[Any] = None
        self._backend_name: str = "mss"
        self._last_frame: Optional[np.ndarray] = None
        self._last_capture_ts: float = 0.0
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
            ring_size=3,
        )

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """Initializes DXGI Desktop Duplication if available, else thread-safe MSS."""
        with self._lock:
            try:
                sct = self._get_mss()
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

            self._dxcam_camera = None
            self._backend_name = "mss"
            logger.info(
                "DXGIDesktopCapture initialized with MSS desktop buffer backend (monitor_idx=%d).",
                self.monitor_idx,
            )

    def _get_mss(self) -> Any:
        """Returns the current thread's isolated `mss.mss()` instance."""
        sct = getattr(self._thread_local, "sct", None)
        if sct is None:
            try:
                sct = mss.mss()
                self._thread_local.sct = sct
            except Exception as exc:
                logger.error("Unable to create thread-local MSS instance: %s", exc)
                raise
        return sct

    @property
    def backend_name(self) -> str:
        """Returns the active desktop capture backend identifier (`'dxcam'` or `'mss'`)."""
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
                logger.error("Desktop buffer capture failed: %s", exc)
                h = max(1, bbox.get("height", 100))
                w = max(1, bbox.get("width", 100))
                frame = np.zeros((h, w, 3), dtype=np.uint8)

        with self._lock:
            if region is None:
                shared_view = self.ring_buffer.write_frame(frame)
                self._last_frame = shared_view
            else:
                self._last_frame = frame
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
        if refresh or self.ring_buffer.frame_sequence == 0:
            self.grab_desktop(region=None)
        return self.ring_buffer.grab_roi_view(x1, y1, x2, y2)

    def _normalize_region(
        self,
        region: Optional[Union[Dict[str, int], Tuple[int, int, int, int]]],
    ) -> Dict[str, int]:
        """Normalizes a region specification into an MSS-compatible bounding dictionary."""
        if region is None:
            return dict(self.monitor_geometry)

        if isinstance(region, dict):
            left = int(region.get("left", self.monitor_geometry["left"]))
            top = int(region.get("top", self.monitor_geometry["top"]))
            width = max(1, int(region.get("width", self.monitor_geometry["width"])))
            height = max(1, int(region.get("height", self.monitor_geometry["height"])))
            return {"left": left, "top": top, "width": width, "height": height}

        if isinstance(region, (tuple, list)) and len(region) == 4:
            x0, y0, x1_or_w, y1_or_h = (int(v) for v in region)
            if x1_or_w > x0 and y1_or_h > y0:
                return {
                    "left": x0,
                    "top": y0,
                    "width": max(1, x1_or_w - x0),
                    "height": max(1, y1_or_h - y0),
                }
            return {
                "left": x0,
                "top": y0,
                "width": max(1, x1_or_w),
                "height": max(1, y1_or_h),
            }

        return dict(self.monitor_geometry)

    @staticmethod
    def normalize_illumination(image: np.ndarray) -> np.ndarray:
        """
        Normalizes lighting, gamma, and semi-transparent UI opacity variations
        using CLAHE on the luminance channel (or grayscale directly) so template
        matching remains resilient to day/night cycles and UI alpha blending.
        """
        try:
            if image is None or image.size == 0:
                return np.zeros((1, 1), dtype=np.uint8)

            if image.ndim == 3 and image.shape[2] >= 3:
                lab = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
                l_norm = clahe.apply(l_channel)
                merged = cv2.merge((l_norm, a_channel, b_channel))
                bgr_norm = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
                gray = cv2.cvtColor(bgr_norm, cv2.COLOR_BGR2GRAY)
            elif image.ndim == 2:
                clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
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

            std_dev = float(np.std(frame_gray))
            # Typical UI grayscale std_dev spans ~20 (low contrast/translucent) to ~75 (sharp contrast)
            contrast_factor = float(np.clip((std_dev - 20.0) / 55.0, 0.0, 1.0))
            dynamic_thr = self.min_threshold + contrast_factor * (self.max_threshold - self.min_threshold)
            blended = 0.5 * dynamic_thr + 0.5 * target_base
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
                return False, 0.0, (0, 0), self.min_threshold

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
                return False, 0.0, (0, 0), self.min_threshold

            effective_threshold = self.compute_dynamic_threshold(proc_frame, base_threshold=threshold)
            res = cv2.matchTemplate(proc_frame, proc_tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            score = float(max_val)
            if math.isnan(score) or math.isinf(score):
                score = 0.0

            matched = score >= effective_threshold
            return matched, score, (int(max_loc[0]), int(max_loc[1])), effective_threshold
        except Exception as exc:
            logger.error("Tolerant template match failed: %s", exc)
            return False, 0.0, (0, 0), self.min_threshold

    def close(self) -> None:
        """Releases any held desktop duplication or MSS resources."""
        with self._lock:
            if self._dxcam_camera is not None:
                try:
                    stop_fn = getattr(self._dxcam_camera, "stop", None)
                    if callable(stop_fn):
                        stop_fn()
                except Exception as exc:
                    logger.debug("Error stopping dxcam camera: %s", exc)
                self._dxcam_camera = None

            sct = getattr(self._thread_local, "sct", None)
            if sct is not None:
                try:
                    sct.close()
                except Exception as exc:
                    logger.debug("Error closing thread-local MSS: %s", exc)
                self._thread_local.sct = None


class AdaptiveFramePacer:
    """
    State-dependent variable FPS frame pacer with Gaussian jitter.

    Eliminates fixed 60.0 FPS (`16.67ms`) robotic capture loops by dynamically
    selecting a Gaussian-jittered target FPS based on the current bot activity state:
      - `WALKING`: 8 - 12 FPS (`83.3ms - 125.0ms`)
      - `APPROACHING` / `GATHERING`: 20 - 30 FPS (`33.3ms - 50.0ms`)
      - `COMBAT`: 15 - 22 FPS (`45.5ms - 66.7ms`)
      - `PAUSED` / `BREAK`: 2 - 4 FPS (`250.0ms - 500.0ms`)
    """

    DEFAULT_STATE_FPS_RANGES: Dict[str, Tuple[float, float]] = {
        "WALKING": (8.0, 12.0),
        "PATROLLING": (8.0, 12.0),
        "SEARCHING": (8.0, 12.0),
        "APPROACHING": (20.0, 30.0),
        "GATHERING": (20.0, 30.0),
        "LOOTING": (20.0, 30.0),
        "COMBAT": (15.0, 22.0),
        "ATTACKING": (15.0, 22.0),
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

                # Explicitly guard against accidental 60 FPS (16.666ms) harmonic lock
                if abs(interval_sec - (1.0 / 60.0)) < 0.0015:
                    interval_sec += self._rng.uniform(0.004, 0.009)

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
        try:
            if widget is not None:
                set_title_fn = getattr(widget, "setWindowTitle", None)
                if callable(set_title_fn):
                    set_title_fn(chosen_title)

                set_obj_name_fn = getattr(widget, "setObjectName", None)
                if callable(set_obj_name_fn):
                    safe_obj_name = chosen_title.replace(" ", "") + "Window"
                    set_obj_name_fn(safe_obj_name)

                hwnd = self._extract_hwnd(widget)
                if hwnd and sys.platform == "win32":
                    try:
                        ctypes.windll.user32.SetWindowTextW(ctypes.wintypes.HWND(hwnd), chosen_title)
                    except Exception as win_exc:
                        logger.debug("SetWindowTextW skipped or failed for HWND=%s: %s", hwnd, win_exc)

            logger.info("Applied stealth window identity: '%s'", chosen_title)
        except Exception as exc:
            logger.error("Failed to apply stealth window identity: %s", exc)

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
                # Offscreen QPA platform does not maintain Win32 HWND tables; store virtual style on widget
                if widget is not None and not isinstance(widget, int):
                    setattr(widget, "_stealth_ex_style", int(new_style))
                verified_style = int(new_style)

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
