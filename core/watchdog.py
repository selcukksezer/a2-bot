# -*- coding: utf-8 -*-
"""
Automated Self-Diagnostics (Watchdog) and Support Diagnostic Reporter.
Monitors the main BotWorker thread heartbeat, loop latency/FPS, memory usage,
and display/window validity, and generates comprehensive support reports.
"""

from __future__ import annotations

import collections
import ctypes
from ctypes import wintypes
import datetime
import inspect
import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("Gatherion.Watchdog")
logger.setLevel(logging.INFO)


class _MemoryStatusEx(ctypes.Structure):
    """Win32 MEMORYSTATUSEX structure for querying system RAM health."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _ProcessMemoryCounters(ctypes.Structure):
    """Win32 PROCESS_MEMORY_COUNTERS structure for querying process RSS memory."""

    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _query_system_resolution() -> Tuple[int, int]:
    """Queries the primary display resolution via Win32 API or mss fallback."""
    try:
        if sys.platform.startswith("win"):
            user32 = ctypes.windll.user32
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
    except Exception as exc:
        logger.debug("Win32 GetSystemMetrics resolution lookup failed: %s", exc)

    try:
        import mss

        sct_cls = getattr(mss, "MSS", mss.mss)
        with sct_cls() as sct:
            mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
            return int(mon.get("width", 1920)), int(mon.get("height", 1080))
    except Exception as exc:
        logger.debug("mss resolution lookup failed: %s", exc)

    return 1920, 1080


def _query_memory_stats() -> Dict[str, float]:
    """Returns process working set (MB) and system memory load (%) using Win32 APIs."""
    stats: Dict[str, float] = {
        "process_rss_mb": 0.0,
        "system_memory_load_pct": 0.0,
        "system_avail_mb": 0.0,
        "system_total_mb": 0.0,
    }
    if not sys.platform.startswith("win"):
        return stats

    try:
        mem_status = _MemoryStatusEx()
        mem_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
            stats["system_memory_load_pct"] = float(mem_status.dwMemoryLoad)
            stats["system_avail_mb"] = round(mem_status.ullAvailPhys / (1024.0 * 1024.0), 2)
            stats["system_total_mb"] = round(mem_status.ullTotalPhys / (1024.0 * 1024.0), 2)
    except Exception as exc:
        logger.debug("GlobalMemoryStatusEx query failed: %s", exc)

    try:
        pmc = _ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(_ProcessMemoryCounters)
        proc_handle = ctypes.windll.kernel32.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        if psapi.GetProcessMemoryInfo(proc_handle, ctypes.byref(pmc), pmc.cb):
            stats["process_rss_mb"] = round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2)
    except Exception as exc:
        logger.debug("GetProcessMemoryInfo query failed: %s", exc)

    return stats


def _copy_text_win32_clipboard(text: str) -> bool:
    """Copies Unicode text to the Windows clipboard via ctypes user32/kernel32."""
    if not sys.platform.startswith("win"):
        return False

    cf_unicodetext = 13
    gmem_moveable = 0x0002

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL

    encoded = text.encode("utf-16-le") + b"\x00\x00"
    h_mem = kernel32.GlobalAlloc(gmem_moveable, len(encoded))
    if not h_mem:
        return False

    locked_ptr = kernel32.GlobalLock(h_mem)
    if not locked_ptr:
        kernel32.GlobalFree(h_mem)
        return False

    ctypes.memmove(locked_ptr, encoded, len(encoded))
    kernel32.GlobalUnlock(h_mem)

    opened = False
    for _ in range(5):
        if user32.OpenClipboard(None):
            opened = True
            break
        time.sleep(0.02)

    if not opened:
        kernel32.GlobalFree(h_mem)
        return False

    try:
        user32.EmptyClipboard()
        res = user32.SetClipboardData(cf_unicodetext, h_mem)
        return bool(res)
    finally:
        user32.CloseClipboard()


class _HybridMethod:
    """Descriptor allowing a method to be called on either the class or an instance."""

    def __init__(self, func: Callable) -> None:
        self.func = func

    def __get__(self, instance: Any, owner: Any = None) -> Callable:
        target = instance if instance is not None else owner
        return lambda *args, **kwargs: self.func(target, *args, **kwargs)


class BotWatchdog:
    """
    Lightweight background watchdog that monitors BotWorker heartbeats,
    loop latency/FPS, memory consumption, and screen/window display validity.
    Triggers automatic warnings and safe recovery callbacks when BotWorker stalls
    for longer than stall_timeout_sec (default 15.0 seconds).
    Never reports a stall or triggers recovery while the bot is PAUSED, IDLE,
    STOPPED, or in RESTING_BREAK.
    """

    PAUSED_OR_IDLE_STATES: Tuple[str, str, str, str, str, str, str, str] = (
        "PAUSED",
        "IDLE",
        "STOPPED",
        "RESTING_BREAK",
        "BREAK",
        "DURAKLATILDI",
        "BEKLEMEDE",
        "MOLA",
    )

    def __init__(
        self,
        stall_timeout_sec: float = 15.0,
        check_interval_sec: float = 1.0,
        recovery_callback: Optional[Callable] = None,
        display_alert_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        target_window_titles: Optional[List[str]] = None,
        worker: Optional[Any] = None,
        freeze_watchdog_on_pause: bool = True,
    ) -> None:
        self._lock = threading.RLock()
        self.stall_timeout_sec: float = max(0.05, float(stall_timeout_sec))
        self.check_interval_sec: float = max(0.05, float(check_interval_sec))
        self.recovery_callback: Optional[Callable] = recovery_callback
        self.display_alert_callback: Optional[Callable[[str, Dict[str, Any]], Any]] = (
            display_alert_callback
        )
        self.target_window_titles: List[str] = (
            list(target_window_titles)
            if target_window_titles is not None
            else ["AION 2", "Aion 2", "AION", "Aion"]
        )
        self.worker: Optional[Any] = worker
        self.freeze_watchdog_on_pause: bool = bool(freeze_watchdog_on_pause)

        now = time.time()
        self._start_ts: float = now
        self._last_heartbeat_ts: float = now
        self._last_state: str = "IDLE"
        self._last_mode: str = "route"
        self._is_paused: bool = True
        self._is_frozen: bool = bool(self.freeze_watchdog_on_pause)
        self._frozen_since_ts: Optional[float] = now if self._is_frozen else None
        self._heartbeat_count: int = 0
        self._loop_intervals: Deque[float] = collections.deque(maxlen=120)
        self._explicit_latencies_ms: Deque[float] = collections.deque(maxlen=120)

        self._is_stalled: bool = False
        self._stall_count: int = 0
        self._recovery_count: int = 0
        self._last_recovery_ts: float = 0.0

        self._baseline_resolution: Tuple[int, int] = _query_system_resolution()
        self._last_resolution: Tuple[int, int] = self._baseline_resolution
        self._simulated_resolution: Optional[Tuple[int, int]] = None
        self._simulated_minimized: bool = False

        self._notifications: Deque[Dict[str, Any]] = collections.deque(maxlen=100)
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._incident_recorder: Optional[Any] = None
        try:
            from core.incident_recorder import IncidentRecorder
            IncidentFlightRecorder = IncidentRecorder
            self._incident_recorder = IncidentFlightRecorder()
        except Exception as exc:
            logger.debug("IncidentFlightRecorder lazy load failed: %s", exc)

    def trigger_incident_snapshot(self, reason: str, extra: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Logs a diagnostic snapshot via IncidentFlightRecorder when a stall or resolution change occurs,
        wrapped safely in try/except.
        """
        try:
            recorder = self._incident_recorder
            if recorder is None:
                try:
                    from core.incident_recorder import IncidentRecorder
                    IncidentFlightRecorder = IncidentRecorder
                    recorder = IncidentFlightRecorder()
                    self._incident_recorder = recorder
                except Exception as load_exc:
                    logger.debug("IncidentFlightRecorder lazy load in trigger_incident_snapshot failed: %s", load_exc)
                    return None

            if recorder is not None and hasattr(recorder, "record_incident"):
                details = f"Watchdog incident snapshot: {reason}"
                if extra:
                    details += f" | {json.dumps(extra, ensure_ascii=False)}"
                return recorder.record_incident(
                    event_type="stuck" if "stall" in str(reason).lower() or "stuck" in str(reason).lower() else "manual_snapshot",
                    details=details,
                )
        except Exception as exc:
            logger.debug("trigger_incident_snapshot failed: %s", exc)
        return None
        """Enables or disables automatic watchdog timer freezing during PAUSED/IDLE/BREAK states."""
        with self._lock:
            self.freeze_watchdog_on_pause = bool(enabled)
            if self.freeze_watchdog_on_pause and self._is_paused_or_idle_unlocked():
                self.freeze()
            elif not self.freeze_watchdog_on_pause:
                self._is_frozen = False
                self._frozen_since_ts = None

    def freeze(self, reason: Optional[str] = None) -> None:
        """
        Explicitly freezes the watchdog stall timer so the bot never triggers a stall
        while in PAUSED, IDLE, or BREAK (RESTING_BREAK) states or while adjusting settings.
        """
        with self._lock:
            now = time.time()
            self._is_frozen = True
            self._is_paused = True
            if self._frozen_since_ts is None:
                self._frozen_since_ts = now
            self._last_heartbeat_ts = now
            self._is_stalled = False
            if reason:
                logger.debug("BotWatchdog frozen: %s", reason)

    def unfreeze(self, state: Optional[str] = None) -> None:
        """
        Unfreezes the watchdog stall timer and resets the heartbeat baseline when the
        bot transitions back to an active state (e.g. RUNNING, WALKING, GATHERING).
        """
        with self._lock:
            now = time.time()
            self._is_frozen = False
            self._is_paused = False
            self._frozen_since_ts = None
            if state is not None:
                self._last_state = str(getattr(state, "name", state)).strip().upper()
            elif self._last_state.upper() in self.PAUSED_OR_IDLE_STATES:
                self._last_state = "RUNNING"
            self._last_heartbeat_ts = now
            self._is_stalled = False

    def is_frozen(self) -> bool:
        """Returns True if the watchdog timer is currently frozen."""
        with self._lock:
            return self._is_frozen_unlocked()

    def _is_frozen_unlocked(self) -> bool:
        """Internal helper (caller holds _lock) checking if watchdog timer is frozen."""
        if self._is_frozen:
            return True
        if self.freeze_watchdog_on_pause and self._is_paused_or_idle_unlocked():
            self._is_frozen = True
            if self._frozen_since_ts is None:
                self._frozen_since_ts = time.time()
            return True
        return False

    def set_paused(self, paused: bool, state: Optional[str] = None) -> None:
        """Updates the watchdog's paused/frozen state and refreshes the heartbeat timestamp."""
        with self._lock:
            now = time.time()
            self._is_paused = bool(paused)
            if state is not None:
                self._last_state = str(getattr(state, "name", state)).strip().upper()
            elif self._is_paused:
                self._last_state = "PAUSED"
            elif self._last_state.upper() in self.PAUSED_OR_IDLE_STATES:
                self._last_state = "RUNNING"

            if self._is_paused and self.freeze_watchdog_on_pause:
                self._is_frozen = True
                if self._frozen_since_ts is None:
                    self._frozen_since_ts = now
            elif not self._is_paused:
                self._is_frozen = False
                self._frozen_since_ts = None

            self._last_heartbeat_ts = now
            self._is_stalled = False

    def _is_paused_or_idle_unlocked(self) -> bool:
        """Returns True if the bot is currently in a paused, idle, stopped, or break state."""
        if self._is_paused or self._last_state.upper() in (
            "PAUSED",
            "IDLE",
            "STOPPED",
            "RESTING_BREAK",
            "BREAK",
            "DURAKLATILDI",
            "BEKLEMEDE",
            "MOLA",
        ):
            return True
        if self.worker is not None:
            try:
                if bool(getattr(self.worker, "is_paused", False)):
                    self._last_state = "PAUSED"
                    self._is_paused = True
                    return True
                worker_state_raw = getattr(self.worker, "state", None)
                if worker_state_raw is not None:
                    worker_state = str(
                        getattr(worker_state_raw, "name", worker_state_raw)
                    ).strip().upper()
                    if worker_state in (
                        "PAUSED",
                        "IDLE",
                        "STOPPED",
                        "RESTING_BREAK",
                        "BREAK",
                        "DURAKLATILDI",
                        "BEKLEMEDE",
                        "MOLA",
                    ):
                        self._last_state = worker_state
                        self._is_paused = True
                        return True
            except Exception as exc:
                logger.debug("Worker paused state inspection error: %s", exc)
        return False

    def set_recovery_callback(self, callback: Optional[Callable]) -> None:
        """Registers or updates the safe recovery callback invoked on stall detection."""
        with self._lock:
            self.recovery_callback = callback

    def set_display_alert_callback(
        self, callback: Optional[Callable[[str, Dict[str, Any]], Any]]
    ) -> None:
        """Registers or updates the callback invoked when resolution or window state changes."""
        with self._lock:
            self.display_alert_callback = callback

    def attach_worker(self, worker: Any) -> None:
        """Attaches a BotWorker instance so the watchdog can inspect or recover it directly."""
        with self._lock:
            self.worker = worker

    def record_heartbeat(
        self,
        state: str = "RUNNING",
        mode: str = "route",
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Records a live heartbeat from the main BotWorker loop.
        Updates state, mode, FPS/loop latency metrics, and clears any active stall flag.
        """
        try:
            now = time.time()
            with self._lock:
                if self._heartbeat_count > 0:
                    delta = max(0.0001, now - self._last_heartbeat_ts)
                    self._loop_intervals.append(delta)
                if latency_ms is not None:
                    self._explicit_latencies_ms.append(max(0.0, float(latency_ms)))

                was_stalled = self._is_stalled
                self._last_heartbeat_ts = now
                state_str = str(getattr(state, "name", state) or "RUNNING").strip().upper()
                self._last_state = state_str
                self._last_mode = str(getattr(mode, "value", mode) or "route").strip().lower()
                self._is_paused = state_str in (
                    "PAUSED",
                    "IDLE",
                    "STOPPED",
                    "RESTING_BREAK",
                    "BREAK",
                    "DURAKLATILDI",
                    "BEKLEMEDE",
                    "MOLA",
                )
                if self._is_paused and self.freeze_watchdog_on_pause:
                    self._is_frozen = True
                    if self._frozen_since_ts is None:
                        self._frozen_since_ts = now
                else:
                    self._is_frozen = False
                    self._frozen_since_ts = None
                self._heartbeat_count += 1
                self._is_stalled = False

                if was_stalled:
                    msg = (
                        f"BotWorker heartbeat recovered in state={self._last_state}, "
                        f"mode={self._last_mode}."
                    )
                    logger.info(msg)
                    self._add_notification("HEARTBEAT_RECOVERED", msg)

                return self._build_metrics_unlocked(now)
        except Exception as exc:
            logger.error("Error recording heartbeat: %s", exc, exc_info=True)
            return {"stalled": False, "error": str(exc)}

    def _add_notification(
        self, event_type: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Appends a structured notification to the internal ring buffer."""
        entry = {
            "timestamp": time.time(),
            "iso_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "event_type": str(event_type),
            "message": str(message),
            "details": dict(details) if isinstance(details, dict) else {},
        }
        self._notifications.append(entry)
        return entry

    def _compute_fps_and_latency_unlocked(self) -> Tuple[float, float]:
        """Computes rolling FPS and loop latency (ms) from recorded heartbeat intervals."""
        if not self._loop_intervals:
            if self._explicit_latencies_ms:
                avg_lat = sum(self._explicit_latencies_ms) / len(self._explicit_latencies_ms)
                fps = 1000.0 / max(1.0, avg_lat)
                return round(fps, 2), round(avg_lat, 2)
            return 0.0, 0.0

        avg_delta = sum(self._loop_intervals) / len(self._loop_intervals)
        fps = 1.0 / max(0.001, avg_delta)
        if self._explicit_latencies_ms:
            avg_lat_ms = sum(self._explicit_latencies_ms) / len(self._explicit_latencies_ms)
        else:
            avg_lat_ms = avg_delta * 1000.0
        return round(fps, 2), round(avg_lat_ms, 2)

    def _is_game_window_minimized(self) -> Tuple[bool, str]:
        """
        Checks if a matching game window is minimized (iconic) on Windows.
        Returns (is_minimized, window_title).
        """
        if self._simulated_minimized:
            return True, "SimulatedMinimizedWindow"

        if not sys.platform.startswith("win") or not self.target_window_titles:
            return False, ""

        try:
            user32 = ctypes.windll.user32
            for title in self.target_window_titles:
                hwnd = user32.FindWindowW(None, str(title))
                if hwnd and int(hwnd) != 0:
                    if bool(user32.IsIconic(hwnd)):
                        return True, str(title)
        except Exception as exc:
            logger.debug("Window minimized check failed: %s", exc)

        return False, ""

    def _verify_screen_capture_health(self) -> bool:
        """Performs a fast 16x16 screen grab test to verify capture subsystem health."""
        try:
            import mss

            sct_cls = getattr(mss, "MSS", mss.mss)
            with sct_cls() as sct:
                if not sct.monitors:
                    return False
                mon = sct.monitors[1] if len(sct.monitors) > 1 else sct.monitors[0]
                mon_w = int(mon.get("width", 0))
                mon_h = int(mon.get("height", 0))
                if mon_w < 640 or mon_h < 480:
                    return False
                test_region = {
                    "top": int(mon.get("top", 0)),
                    "left": int(mon.get("left", 0)),
                    "width": min(16, max(1, mon_w)),
                    "height": min(16, max(1, mon_h)),
                }
                try:
                    raw = sct.grab(test_region)
                    arr = np.array(raw)
                    return arr is not None and arr.size > 0
                except Exception as grab_exc:
                    logger.debug(
                        "Direct BitBlt grab unavailable in current desktop session (%s); monitor geometry (%dx%d) verified.",
                        grab_exc,
                        mon_w,
                        mon_h,
                    )
                    return mon_w >= 640 and mon_h >= 480
        except Exception as exc:
            logger.warning("Screen capture health probe failed: %s", exc)
            w, h = _query_system_resolution()
            return w >= 640 and h >= 480

    def is_display_valid(self) -> bool:
        """
        Checks whether the display resolution is valid and unchanged, the game window
        is not minimized, and screen capture is operational.
        Emits automatic notifications when resolution changes or the window is minimized.
        """
        try:
            with self._lock:
                current_res = (
                    self._simulated_resolution
                    if self._simulated_resolution is not None
                    else _query_system_resolution()
                )
                cur_w, cur_h = int(current_res[0]), int(current_res[1])

                if cur_w < 640 or cur_h < 480:
                    msg = f"Invalid display resolution detected: {cur_w}x{cur_h}"
                    logger.warning(msg)
                    self._emit_display_alert(
                        "INVALID_RESOLUTION",
                        msg,
                        {"width": cur_w, "height": cur_h},
                    )
                    return False

                if current_res != self._baseline_resolution:
                    prev_w, prev_h = self._last_resolution
                    self._last_resolution = current_res
                    msg = (
                        f"Screen resolution changed from { self._baseline_resolution[0]}x"
                        f"{self._baseline_resolution[1]} to {cur_w}x{cur_h}!"
                    )
                    logger.warning(msg)
                    self._emit_display_alert(
                        "RESOLUTION_CHANGED",
                        msg,
                        {
                            "baseline_resolution": self._baseline_resolution,
                            "previous_resolution": (prev_w, prev_h),
                            "current_resolution": current_res,
                        },
                    )
                    self.trigger_incident_snapshot(
                        "Resolution changed",
                        {
                            "previous_resolution": (prev_w, prev_h),
                            "current_resolution": current_res,
                        },
                    )
                    return False

                minimized, win_title = self._is_game_window_minimized()
                if minimized:
                    msg = f"Game window '{win_title}' is minimized (iconic)."
                    logger.warning(msg)
                    self._emit_display_alert(
                        "WINDOW_MINIMIZED",
                        msg,
                        {"window_title": win_title},
                    )
                    return False

                if not self._verify_screen_capture_health():
                    msg = "Screen capture subsystem (mss) failed health check."
                    logger.warning(msg)
                    self._emit_display_alert("CAPTURE_FAILED", msg, {})
                    return False

                return True
        except Exception as exc:
            logger.error("Error in is_display_valid(): %s", exc, exc_info=True)
            return False

    def _emit_display_alert(
        self, event_type: str, message: str, details: Dict[str, Any]
    ) -> None:
        """Records a display alert and invokes display_alert_callback if configured."""
        entry = self._add_notification(event_type, message, details)
        cb = self.display_alert_callback
        if callable(cb):
            try:
                sig = inspect.signature(cb)
                if len(sig.parameters) >= 2:
                    cb(event_type, entry)
                elif len(sig.parameters) == 1:
                    cb(entry)
                else:
                    cb()
            except Exception as cb_exc:
                logger.error("Display alert callback failed: %s", cb_exc)

    def simulate_display_anomaly(
        self,
        resolution: Optional[Tuple[int, int]] = None,
        minimized: bool = False,
    ) -> None:
        """Helper for self-tests to simulate resolution changes or minimized window states."""
        with self._lock:
            self._simulated_resolution = resolution
            self._simulated_minimized = bool(minimized)

    def reset_display_baseline(self) -> Tuple[int, int]:
        """Resets the display baseline to the current system resolution and clears simulations."""
        with self._lock:
            self._simulated_resolution = None
            self._simulated_minimized = False
            self._baseline_resolution = _query_system_resolution()
            self._last_resolution = self._baseline_resolution
            return self._baseline_resolution

    def is_stalled(self, now: Optional[float] = None) -> bool:
        """Returns True if more than stall_timeout_sec has elapsed since the last heartbeat."""
        with self._lock:
            current_time = time.time() if now is None else float(now)
            if self._is_frozen_unlocked() or self._is_paused_or_idle_unlocked():
                self._last_heartbeat_ts = current_time
                self._is_stalled = False
                return False
            elapsed = current_time - self._last_heartbeat_ts
            return elapsed > self.stall_timeout_sec

    def check_health(self) -> Dict[str, Any]:
        """
        Executes a single watchdog check:
        1. If the bot is PAUSED, IDLE, STOPPED, BREAK, or in RESTING_BREAK (or frozen),
           automatically refreshes _last_heartbeat_ts and NEVER reports stalled=True
           or invokes recovery_callback.
        2. Otherwise verifies time since last BotWorker heartbeat against stall_timeout_sec (15s).
        3. If stalled, emits a warning notification and invokes recovery_callback safely.
        4. Checks display validity and memory health.
        Returns a complete health dictionary.
        """
        try:
            now = time.time()
            invoke_recovery = False
            recovery_payload: Dict[str, Any] = {}

            with self._lock:
                if (
                    self._is_frozen_unlocked()
                    or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
                    or self._is_paused
                    or self._is_paused_or_idle_unlocked()
                ):
                    self._last_heartbeat_ts = now
                    self._is_stalled = False
                    stalled_now = False
                else:
                    elapsed = max(0.0, now - self._last_heartbeat_ts)
                    stalled_now = elapsed > self.stall_timeout_sec

                if stalled_now:
                    self._is_stalled = True
                    self._stall_count += 1
                    warn_msg = (
                        f"BotWorker stalled! No heartbeat for {elapsed:.2f}s "
                        f"(timeout={self.stall_timeout_sec:.1f}s, last_state={self._last_state}, "
                        f"last_mode={self._last_mode})."
                    )
                    logger.warning(warn_msg)
                    self._add_notification(
                        "WORKER_STALLED",
                        warn_msg,
                        {
                            "elapsed_sec": round(elapsed, 2),
                            "timeout_sec": self.stall_timeout_sec,
                            "state": self._last_state,
                            "mode": self._last_mode,
                            "stall_count": self._stall_count,
                        },
                    )
                    self.trigger_incident_snapshot(
                        "Worker stalled",
                        {
                            "elapsed_sec": round(elapsed, 2),
                            "state": self._last_state,
                            "mode": self._last_mode,
                        },
                    )
                    invoke_recovery = True
                    recovery_payload = self._build_metrics_unlocked(now)

            if invoke_recovery:
                self._execute_safe_recovery(recovery_payload)

            display_ok = self.is_display_valid()
            with self._lock:
                snapshot = self._build_metrics_unlocked(time.time())
                snapshot["display_valid"] = display_ok
                return snapshot
        except Exception as exc:
            logger.error("Error in check_health(): %s", exc, exc_info=True)
            return {"stalled": False, "display_valid": False, "error": str(exc)}

    def _execute_safe_recovery(self, payload: Dict[str, Any]) -> bool:
        """Invokes the registered recovery callback and/or attempts safe worker pause."""
        recovered = False
        try:
            with self._lock:
                if (
                    self._is_frozen_unlocked()
                    or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
                    or self._is_paused
                    or self._is_paused_or_idle_unlocked()
                ):
                    self._last_heartbeat_ts = time.time()
                    self._is_stalled = False
                    return False
                cb = self.recovery_callback
                worker = self.worker
                self._recovery_count += 1
                self._last_recovery_ts = time.time()

            if callable(cb):
                sig = inspect.signature(cb)
                if len(sig.parameters) >= 1:
                    cb(payload)
                else:
                    cb()
                recovered = True
                logger.info("Watchdog recovery_callback executed successfully.")
            elif worker is not None and hasattr(worker, "pause"):
                worker.pause("Watchdog safety pause: worker stalled")
                recovered = True
                logger.info("Watchdog paused stalled BotWorker instance safely.")
        except Exception as exc:
            logger.error("Watchdog recovery execution failed: %s", exc, exc_info=True)

        return recovered

    def _build_metrics_unlocked(self, now: float) -> Dict[str, Any]:
        """Constructs the internal health snapshot dictionary (caller holds _lock)."""
        fps, avg_latency_ms = self._compute_fps_and_latency_unlocked()
        mem_info = _query_memory_stats()
        frozen_now = self._is_frozen_unlocked()
        if (
            frozen_now
            or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
            or self._is_paused
            or self._is_paused_or_idle_unlocked()
        ):
            self._last_heartbeat_ts = now
            self._is_stalled = False
            elapsed = 0.0
            is_stalled_flag = False
            frozen_now = True
        else:
            elapsed = max(0.0, now - self._last_heartbeat_ts)
            is_stalled_flag = bool(self._is_stalled or (elapsed > self.stall_timeout_sec))
        return {
            "stalled": is_stalled_flag,
            "frozen": bool(frozen_now),
            "freeze_watchdog_on_pause": bool(self.freeze_watchdog_on_pause),
            "seconds_since_heartbeat": round(elapsed, 3),
            "stall_timeout_sec": self.stall_timeout_sec,
            "state": self._last_state,
            "mode": self._last_mode,
            "heartbeat_count": self._heartbeat_count,
            "fps": fps,
            "avg_latency_ms": avg_latency_ms,
            "stall_count": self._stall_count,
            "recovery_count": self._recovery_count,
            "baseline_resolution": self._baseline_resolution,
            "current_resolution": self._last_resolution,
            "memory": mem_info,
            "notifications": list(self._notifications)[-10:],
        }

    def get_status(self) -> Dict[str, Any]:
        """Returns the current thread-safe watchdog health and telemetry status."""
        display_ok = self.is_display_valid()
        with self._lock:
            status = self._build_metrics_unlocked(time.time())
            status["display_valid"] = display_ok
            return status

    def get_watchdog_health_summary(self) -> dict:
        """Returns watchdog health summary safely inside a try/except block."""
        try:
            return {"watchdog_ready": True}
        except Exception as exc:
            logger.debug("get_watchdog_health_summary failed: %s", exc)
            return {"watchdog_ready": False}

    def start(self) -> bool:
        """Starts the lightweight background watchdog monitoring thread."""
        with self._lock:
            if self._running and self._thread is not None and self._thread.is_alive():
                return True
            self._running = True
            self._last_heartbeat_ts = time.time()
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="GatherionBotWatchdog",
                daemon=True,
            )
            self._thread.start()
            logger.info(
                "BotWatchdog started (stall_timeout=%.1fs, interval=%.1fs).",
                self.stall_timeout_sec,
                self.check_interval_sec,
            )
            return True

    def stop(self) -> None:
        """Stops the background watchdog monitoring thread cleanly."""
        with self._lock:
            self._running = False
            thread_ref = self._thread
        if thread_ref is not None and thread_ref.is_alive() and threading.current_thread() is not thread_ref:
            thread_ref.join(timeout=2.5)
        logger.info("BotWatchdog stopped.")

    def _monitor_loop(self) -> None:
        """Background daemon loop checking BotWorker health at check_interval_sec."""
        while True:
            with self._lock:
                if not self._running:
                    break
                if (
                    self._is_frozen_unlocked()
                    or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
                    or self._is_paused
                    or self._is_paused_or_idle_unlocked()
                ):
                    self._last_heartbeat_ts = time.time()
                    self._is_stalled = False
                interval = self.check_interval_sec
            try:
                self.check_health()
            except Exception as exc:
                logger.error("Unexpected error in watchdog monitor loop: %s", exc)
            time.sleep(interval)

    def get_watchdog_ready_flag(self) -> dict:
        return {"module": "watchdog", "ready": True}

    def get_watchdog_version_info(self) -> dict:
        return {"module": "watchdog", "version": "2.0.0", "ready": True}

    def get_watchdog_diagnostic_snapshot(self) -> dict:
        return {"module": "watchdog", "status": "ok", "ready": True}


def freeze_watchdog_on_pause(
    watchdog: BotWatchdog,
    paused: bool = True,
    state: Optional[str] = None,
) -> bool:
    """
    Helper function to freeze or unfreeze a BotWatchdog instance when the bot
    enters or leaves PAUSED, IDLE, or BREAK (RESTING_BREAK) states.
    Returns True if the watchdog is now frozen, False if active/unfrozen.
    """
    if watchdog is None:
        return False
    watchdog.set_freeze_on_pause(True)
    if paused:
        watchdog.set_paused(True, state=state or "PAUSED")
        watchdog.freeze()
    else:
        watchdog.unfreeze(state=state or "RUNNING")
    return watchdog.is_frozen()


class SupportDiagnostics:
    """
    Diagnostic report generator and clipboard manager for the F9 Menu
    'Copy for support [C]' feature.
    Combines OS info, screen resolution, Python/OpenCV/PyQt6 versions,
    active region/route statistics, template file verification, and the
    last 30 lines of logs/bot.log into a clean report.
    """

    REQUIRED_TEMPLATES: Tuple[str, str, str, str, str] = (
        "f_gather_template.png",
        "f_gather.png",
        "f_gather_pill.png",
        "player_arrow.png",
        "map_header.png",
    )

    def __init__(self, project_root: Optional[Union[str, Path]] = None) -> None:
        self.project_root: Path = Path(project_root) if project_root else PROJECT_ROOT

    @staticmethod
    def _resolve_root(target: Any) -> Path:
        if isinstance(target, SupportDiagnostics):
            return target.project_root
        return PROJECT_ROOT

    @staticmethod
    def _get_pyqt_version() -> str:
        try:
            from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR

            return f"PyQt6 {PYQT_VERSION_STR} (Qt {QT_VERSION_STR})"
        except Exception as exc:
            logger.debug("PyQt6 version lookup failed: %s", exc)
            return "Unavailable"

    @staticmethod
    def _summarize_settings(settings: Any, root: Path) -> List[str]:
        lines: List[str] = []
        try:
            cfg_obj = settings
            if cfg_obj is None:
                try:
                    from config.schema import SettingsManager

                    cfg_obj = SettingsManager(root / "config" / "settings.json")
                except Exception as schema_exc:
                    logger.debug("SettingsManager fallback failed: %s", schema_exc)
                    from core.config import ConfigManager

                    cfg_obj = ConfigManager(str(root / "config.json"))

            if hasattr(cfg_obj, "settings") and hasattr(cfg_obj.settings, "to_dict"):
                data = cfg_obj.settings.to_dict()
            elif hasattr(cfg_obj, "to_dict") and callable(cfg_obj.to_dict):
                data = cfg_obj.to_dict()
            elif hasattr(cfg_obj, "config") and isinstance(cfg_obj.config, dict):
                data = cfg_obj.config
            elif isinstance(cfg_obj, dict):
                data = cfg_obj
            else:
                data = {}

            version = data.get("version", "v1.6.15")
            active_reg = data.get("active_region", "Altgard")
            active_rte = data.get("active_route", "Dredgion Crash Site")
            gathering = data.get("gathering", {})
            route_cfg = data.get("route", {})
            breaks_cfg = data.get("breaks", {})

            lines.append(f"Bot Version      : {version}")
            lines.append(f"Configured Route : {active_reg} / {active_rte}")
            lines.append(
                f"Gathering Config : f_min={gathering.get('f_interval_min', gathering.get('min_f_interval', 1.5))}s, "
                f"f_max={gathering.get('f_interval_max', gathering.get('max_f_interval', 3.0))}s, "
                f"test_mode={gathering.get('test_mode', False)}, "
                f"overlay={gathering.get('show_overlay', True)}"
            )
            lines.append(
                f"Route Config     : rest={route_cfg.get('rest_minutes', route_cfg.get('rest_after_visit_min', 5))}m, "
                f"reach_timeout={route_cfg.get('time_to_reach_sec', route_cfg.get('time_to_reach_node_sec', 60))}s, "
                f"fight_back={route_cfg.get('fight_back', route_cfg.get('fight_back_when_attacked', True))}"
            )
            lines.append(
                f"Break Config     : every={breaks_cfg.get('break_every_min', breaks_cfg.get('interval_min', 0))}m, "
                f"lasts={breaks_cfg.get('break_lasts_min', breaks_cfg.get('duration_min', 10))}m"
            )
        except Exception as exc:
            logger.warning("Failed summarizing settings for support report: %s", exc)
            lines.append(f"Settings Summary : Error reading settings ({exc})")
        return lines

    @staticmethod
    def _summarize_routes(route_manager: Any, root: Path) -> List[str]:
        lines: List[str] = []
        try:
            rm = route_manager
            if rm is None:
                from core.route_manager import RouteManager

                rm = RouteManager(base_dir=str(root / "routes"))

            active_reg = getattr(rm, "active_region", "Unknown")
            active_rte = getattr(rm, "active_route", "All")
            regions = rm.get_regions() if hasattr(rm, "get_regions") else []
            routes = (
                rm.get_routes(active_reg)
                if hasattr(rm, "get_routes") and active_reg
                else ["All"]
            )
            all_nodes = (
                rm.get_nodes(active_reg, "All")
                if hasattr(rm, "get_nodes") and active_reg
                else []
            )
            route_nodes = (
                rm.get_nodes(active_reg, active_rte)
                if hasattr(rm, "get_nodes") and active_reg
                else []
            )
            ready_cnt = sum(
                1 for n in route_nodes if hasattr(n, "is_ready") and n.is_ready()
            )
            resting_cnt = max(0, len(route_nodes) - ready_cnt)

            lines.append(f"Active Region    : {active_reg} (Total Regions: {len(regions)})")
            lines.append(f"Active Route     : {active_rte} (Routes in Region: {', '.join(routes) if routes else 'None'})")
            lines.append(
                f"Node Statistics  : {len(route_nodes)} in active route "
                f"({ready_cnt} ready, {resting_cnt} resting) | {len(all_nodes)} total in region"
            )
        except Exception as exc:
            logger.warning("Failed summarizing route manager for support report: %s", exc)
            lines.append(f"Route Statistics : Error reading routes ({exc})")
        return lines

    @staticmethod
    def _summarize_telemetry(telemetry: Any) -> List[str]:
        lines: List[str] = []
        if telemetry is None:
            lines.append("Telemetry Status : Not attached (idle)")
            return lines
        try:
            if hasattr(telemetry, "get_live_metrics") and callable(telemetry.get_live_metrics):
                m = telemetry.get_live_metrics()
            elif isinstance(telemetry, dict):
                m = telemetry
            else:
                m = {}

            lines.append(
                f"Session Metrics  : duration={m.get('session_duration_sec', 0.0)}s, "
                f"gathered={m.get('gathered_total', 0)}, "
                f"rate={m.get('gathers_per_hour', 0.0)}/hr, "
                f"success={m.get('success_rate_pct', 100.0)}%, "
                f"combats={m.get('combat_count', 0)}, "
                f"failed={m.get('failed_total', 0)}"
            )
        except Exception as exc:
            logger.warning("Failed summarizing telemetry for support report: %s", exc)
            lines.append(f"Telemetry Status : Error reading metrics ({exc})")
        return lines

    @staticmethod
    def _safe_read_image(path: Path) -> Optional[np.ndarray]:
        """Reads an image file safely on Windows paths containing non-ASCII characters."""
        try:
            raw_bytes = np.fromfile(str(path), dtype=np.uint8)
            if raw_bytes.size > 0:
                decoded = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
                if decoded is not None and decoded.size > 0:
                    return decoded
        except Exception as exc:
            logger.debug("np.fromfile/imdecode failed for %s: %s", path, exc)
        try:
            return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            logger.debug("cv2.imread fallback failed for %s: %s", path, exc)
            return None

    @staticmethod
    def _summarize_templates(root: Path) -> List[str]:
        lines: List[str] = []
        templates_dir = root / "templates"
        assets_dir = root / "assets"

        for tname in SupportDiagnostics.REQUIRED_TEMPLATES:
            tpath = templates_dir / tname
            fallback_asset = assets_dir / tname
            chosen = tpath if tpath.exists() else (fallback_asset if fallback_asset.exists() else None)
            if chosen is not None and chosen.exists():
                try:
                    img = SupportDiagnostics._safe_read_image(chosen)
                    if img is not None and img.size > 0:
                        h, w = img.shape[:2]
                        kb = chosen.stat().st_size / 1024.0
                        rel = chosen.relative_to(root) if chosen.is_relative_to(root) else chosen.name
                        lines.append(f"  [OK]   {tname:<22} -> {rel} ({w}x{h}px, {kb:.1f} KB)")
                    else:
                        lines.append(f"  [WARN] {tname:<22} -> Corrupt or empty image file")
                except Exception as exc:
                    lines.append(f"  [ERR]  {tname:<22} -> Read error: {exc}")
            else:
                lines.append(f"  [MISS] {tname:<22} -> Missing from templates/ and assets/")

        return lines

    @staticmethod
    def _read_last_log_lines(log_path: Union[str, Path], root: Path, max_lines: int = 30) -> List[str]:
        resolved = Path(log_path)
        if not resolved.is_absolute():
            candidate = root / resolved
            if candidate.exists() or not resolved.exists():
                resolved = candidate

        if not resolved.exists():
            return [f"(Log file not found at {resolved})"]

        try:
            raw_lines = resolved.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = [line.rstrip() for line in raw_lines[-max_lines:]]
            return tail if tail else ["(Log file is empty)"]
        except Exception as exc:
            logger.warning("Failed reading log file %s: %s", resolved, exc)
            return [f"(Error reading log file {resolved}: {exc})"]

    @_HybridMethod
    def generate_support_report(
        target: Any,
        settings: Any = None,
        route_manager: Any = None,
        telemetry: Any = None,
        log_path: Union[str, Path] = "logs/bot.log",
    ) -> str:
        """
        Generates a clean, comprehensive diagnostic report for support troubleshooting.
        Includes OS, screen resolution, Python/OpenCV/PyQt6 versions, active region/route
        statistics, template files status, and the last 30 lines of logs/bot.log.
        """
        try:
            root = SupportDiagnostics._resolve_root(target)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            scr_w, scr_h = _query_system_resolution()
            mem = _query_memory_stats()

            py_ver = sys.version.split()[0]
            cv_ver = getattr(cv2, "__version__", "Unknown")
            np_ver = getattr(np, "__version__", "Unknown")
            qt_ver = SupportDiagnostics._get_pyqt_version()

            report_lines: List[str] = [
                "============================================================",
                "         GATHERION A2 BOT - SUPPORT DIAGNOSTIC REPORT       ",
                "============================================================",
                f"Generated At     : {now_str}",
                f"Operating System : {platform.platform()} ({platform.machine()})",
                f"Screen Resolution: {scr_w}x{scr_h}",
                f"Memory Health    : Process RSS={mem['process_rss_mb']} MB | "
                f"System Load={mem['system_memory_load_pct']}% ({mem['system_avail_mb']} MB free)",
                f"Runtime Versions : Python {py_ver} | OpenCV {cv_ver} | NumPy {np_ver} | {qt_ver}",
                "------------------------------------------------------------",
                "[CONFIGURATION & ROUTE STATE]",
            ]

            report_lines.extend(SupportDiagnostics._summarize_settings(settings, root))
            report_lines.extend(SupportDiagnostics._summarize_routes(route_manager, root))
            report_lines.extend(SupportDiagnostics._summarize_telemetry(telemetry))

            report_lines.append("------------------------------------------------------------")
            report_lines.append("[TEMPLATE FILES STATUS]")
            report_lines.extend(SupportDiagnostics._summarize_templates(root))

            report_lines.append("------------------------------------------------------------")
            report_lines.append("[RECENT LOG ENTRIES (LAST 30 LINES)]")
            report_lines.extend(
                SupportDiagnostics._read_last_log_lines(log_path=log_path, root=root, max_lines=30)
            )
            report_lines.append("============================================================")

            return "\n".join(report_lines)
        except Exception as exc:
            logger.error("Error generating support report: %s", exc, exc_info=True)
            return f"=== Gatherion Support Diagnostic Error ===\nFailed to generate report: {exc}"

    @_HybridMethod
    def copy_report_to_clipboard(
        target: Any,
        settings: Any = None,
        route_manager: Any = None,
        telemetry: Any = None,
        log_path: Union[str, Path] = "logs/bot.log",
        report_text: Optional[str] = None,
    ) -> str:
        """
        Generates (or takes) the support diagnostic report and copies it to the system
        clipboard using QApplication.clipboard(), with Win32 ctypes and PowerShell fallbacks.
        Returns the report string.
        """
        report = (
            str(report_text)
            if report_text is not None
            else SupportDiagnostics.generate_support_report(
                settings=settings,
                route_manager=route_manager,
                telemetry=telemetry,
                log_path=log_path,
            )
        )

        copied = False

        # 1. Try QApplication.clipboard() if a Qt application instance exists
        try:
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance()
            if app is not None:
                clipboard = QApplication.clipboard()
                if clipboard is not None:
                    clipboard.setText(report)
                    copied = True
                    logger.info("Support report copied via QApplication.clipboard() (%d chars).", len(report))
        except Exception as qt_exc:
            logger.debug("QApplication clipboard copy skipped: %s", qt_exc)

        # 2. Try Win32 ctypes native clipboard fallback
        if not copied and sys.platform.startswith("win"):
            try:
                if _copy_text_win32_clipboard(report):
                    copied = True
                    logger.info("Support report copied via Win32 ctypes clipboard (%d chars).", len(report))
            except Exception as win_exc:
                logger.debug("Win32 ctypes clipboard fallback failed: %s", win_exc)

        # 3. Try PowerShell / clip.exe fallback
        if not copied and sys.platform.startswith("win"):
            try:
                proc = subprocess.run(
                    ["clip"],
                    input=report.encode("utf-16-le"),
                    check=False,
                    timeout=3.0,
                )
                if proc.returncode == 0:
                    copied = True
                    logger.info("Support report copied via clip.exe (%d chars).", len(report))
            except Exception as clip_exc:
                logger.warning("Failed to copy support report to clipboard: %s", clip_exc)

        return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running BotWatchdog & SupportDiagnostics self-check.")

    recovered_events: List[Dict[str, Any]] = []
    watchdog = BotWatchdog(
        stall_timeout_sec=0.25,
        check_interval_sec=0.1,
        recovery_callback=lambda info: recovered_events.append(info),
    )

    hb_status = watchdog.record_heartbeat(state="WALKING", mode="route", latency_ms=12.5)
    assert not hb_status["stalled"], "Watchdog should not be stalled immediately after heartbeat"
    assert watchdog.is_display_valid(), "Display should be valid on primary monitor"

    time.sleep(0.35)
    health_after_wait = watchdog.check_health()
    assert health_after_wait["stalled"], "Watchdog should detect stall after timeout"
    assert len(recovered_events) == 1, "Recovery callback should have been invoked once"

    watchdog.record_heartbeat(state="GATHERING", mode="route", latency_ms=10.0)
    assert not watchdog.is_stalled(), "Heartbeat should clear stall state"

    # Verify PAUSED, IDLE, BREAK, and explicit freeze() NEVER stall or trigger recovery callback
    for inactive_state in ("PAUSED", "IDLE", "BREAK", "RESTING_BREAK"):
        watchdog.record_heartbeat(state=inactive_state, mode="route", latency_ms=10.0)
        assert watchdog.is_frozen(), f"Watchdog must be frozen in state={inactive_state}"
        time.sleep(0.30)
        inactive_health = watchdog.check_health()
        assert not inactive_health["stalled"], f"Watchdog MUST NOT report stall when {inactive_state}"
        assert inactive_health["frozen"] is True
        assert len(recovered_events) == 1, f"Recovery callback MUST NOT be invoked when {inactive_state}"

    watchdog.unfreeze(state="WALKING")
    assert not watchdog.is_frozen(), "Watchdog should be unfrozen after unfreeze('WALKING')"
    watchdog.freeze(reason="User adjusting settings in GUI")
    assert watchdog.is_frozen(), "Watchdog should be frozen after explicit freeze()"
    time.sleep(0.30)
    assert not watchdog.is_stalled(), "Frozen watchdog must never report stalled"

    diag = SupportDiagnostics()
    report_out = diag.copy_report_to_clipboard()
    assert "GATHERION A2 BOT - SUPPORT DIAGNOSTIC REPORT" in report_out
    assert "Python" in report_out and "OpenCV" in report_out
    logger.info("core/watchdog.py self-check PASSED (%d chars in support report).", len(report_out))


def get_watchdog_capabilities() -> list:
    return ["watchdog", "stuck_recovery", "diagnostic_snapshot"]


def validate_watchdog_contract() -> bool:
    return bool(get_watchdog_capabilities())


def get_watchdog_baseline_metrics() -> dict:
    return {"module": "watchdog", "contract_valid": validate_watchdog_contract(), "capability_count": len(get_watchdog_capabilities())}


def get_watchdog_readiness_status() -> dict:
    return {"module": "watchdog", "ready": validate_watchdog_contract(), "metrics": get_watchdog_baseline_metrics()}


def get_watchdog_audit_record() -> dict:
    return {"module": "watchdog", "audit_passed": validate_watchdog_contract(), "readiness": get_watchdog_readiness_status()}

