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
from version import __version__

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("Gatherion.Watchdog")
logger.setLevel(logging.INFO)


# v20/game-window: oyun penceresi tanıma anahtarları.
# Canlı AION 2 penceresinin gerçek başlığı, hedef listedeki dört tam
# başlıkla ne tam ne de önek (S147) eşleşmesi yapmıyor — detektör oyun
# açıkken bile sürekli False dönüyordu. Bu yüzden eşleşme
# katmanlıdır: tam → önek → süreç exe adı → alt-dize.
_GAME_PROCESS_KEY = "aion"
# Tanı logu seyretme aralığı (sn): False'a ilk geçişte + sonra en fazla
# bu sıklıkta; her 2 sn'de bir log spam'i olmasın.
_GAME_WINDOW_DIAG_INTERVAL_SEC = 30.0
_GAME_WINDOW_DIAG_MAX_TITLES = 10
_GAME_WINDOW_DIAG_TITLE_LEN = 60

# EnumWindows geri-çağrı tipi.
#
# KÖK NEDEN (v20-hotfix, canlı log 2026-11-11 16:54:19 "(0): []"):
# ctypes.wintypes'te WNDENUMPROC YOKTUR — wintypes yalnızca HANDLE
# türlerini (HWND vb.) tanımlar, geri-çağrı prosedür türlerini değil.
# v19/S147'den beri (fb0cd14) `_find_game_window` her iki katmanda da
# `wintypes.WNDENUMPROC` okuyordu; bu AttributeError verip EnumWindows
# hiç çağrılmadan DEBUG'a gömülüyordu ("Window title fallback failed"),
# visible_titles hep boş kalıyordu. Doğru tip ctypes.WINFUNCTYPE ile
# kurulur (repo içindeki çalışan örnekler: input_driver._find_aion_hwnd,
# game_gpu_throttle). Linux/macOS'ta WINFUNCTYPE yoktur; Windows dalı o
# platformlarda hiç çalışmadığı için None bırakılır.
try:
    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
except AttributeError:  # pragma: no cover - Windows-only
    _WNDENUMPROC = None


def match_game_window_title(
    text: str, target_titles: List[str]
) -> Optional[Tuple[str, int]]:
    """Pencere başlığını oyun hedef başlıklarıyla eşleştirir (saf fonksiyon).

    Öncelik (düşük = daha iyi): 0 tam eşleşme, 1 önek (S147, v19/P3-8),
    2 büyük/küçük harf duyarsız alt-dize (v20). Önek alt-dizeden
    önceliklidir: örn. "Aion 2 Rehberi" başlıklı tarayıcı sekmesi,
    gerçek oyun penceresinden önce gelmemeli.
    Döner: (eşleşen_hedef, öncelik) ya da eşleşme yoksa None.
    """
    if not text:
        return None
    lowered = text.lower()
    best: Optional[Tuple[str, int]] = None
    for target in target_titles:
        target = str(target)
        if not target:
            continue
        if text == target:
            return target, 0
        if text.startswith(target):
            if best is None or best[1] > 1:
                best = (target, 1)
        elif target.lower() in lowered:
            if best is None:
                best = (target, 2)
    return best


def game_process_exe_matches(exe_name: str, key: str = _GAME_PROCESS_KEY) -> bool:
    """Oyun sürecinin exe adı anahtarı içeriyor mu? (saf fonksiyon).

    Örn. "AION2.exe" / "aion.exe" → True, "chrome.exe" → False.
    Başlık hiçbir hedefle eşleşmediğinde süreç yedeği olarak kullanılır.
    """
    if not exe_name:
        return False
    return key.lower() in exe_name.lower()


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
            user32.GetSystemMetrics.argtypes = [ctypes.c_int]
            user32.GetSystemMetrics.restype = ctypes.c_int
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

    # Last resort: a hardcoded guess, logged once so it is never mistaken
    # for a measured resolution.
    global _RESOLUTION_FALLBACK_WARNED
    if not _RESOLUTION_FALLBACK_WARNED:
        _RESOLUTION_FALLBACK_WARNED = True
        logger.warning("Çözünürlük ölçülemedi, varsayılan 1920x1080 kullanılıyor.")
    return 1920, 1080


_RESOLUTION_FALLBACK_WARNED = False

# Same display alert (e.g. CAPTURE_FAILED on a headless machine) is not
# re-notified more than once per window: the notification ring holds only
# 100 entries, and a 1 Hz poll re-alerting every second would evict every
# other notification (including WORKER_STALLED) within ~100 s. The status
# dict still reports display_valid=False on every poll; only the
# notification/callback spam is suppressed.
_DISPLAY_ALERT_REPEAT_SUPPRESS_SEC = 60.0


def _query_memory_stats() -> Dict[str, float]:
    """Returns process working set (MB) and system memory load (%) using Win32 APIs."""
    stats: Dict[str, float] = {
        "process_rss_mb": 0.0,
        "system_memory_load_pct": 0.0,
        "system_avail_mb": 0.0,
        "system_total_mb": 0.0,
        "measured": 0.0,
    }
    if not sys.platform.startswith("win"):
        return stats

    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemoryStatusEx)]
        kernel32.GlobalMemoryStatusEx.restype = wintypes.BOOL
        mem_status = _MemoryStatusEx()
        mem_status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status)):
            stats["system_memory_load_pct"] = float(mem_status.dwMemoryLoad)
            stats["system_avail_mb"] = round(mem_status.ullAvailPhys / (1024.0 * 1024.0), 2)
            stats["system_total_mb"] = round(mem_status.ullTotalPhys / (1024.0 * 1024.0), 2)
            stats["measured"] = 1.0
    except Exception as exc:
        logger.debug("GlobalMemoryStatusEx query failed: %s", exc)

    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        pmc = _ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(_ProcessMemoryCounters)
        proc_handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(proc_handle, ctypes.byref(pmc), pmc.cb):
            stats["process_rss_mb"] = round(pmc.WorkingSetSize / (1024.0 * 1024.0), 2)
            stats["measured"] = 1.0
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
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL
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
        # EmptyClipboard failing means the old content is still owned by
        # another app: abort without touching it and free our buffer.
        if not user32.EmptyClipboard():
            kernel32.GlobalFree(h_mem)
            return False
        res = user32.SetClipboardData(cf_unicodetext, h_mem)
        if not res:
            # Ownership did NOT transfer to the system: free the buffer
            # instead of leaking it.
            kernel32.GlobalFree(h_mem)
            return False
        return True
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
    A stall pause is marked with STALL_PAUSE_REASON and is resumed when the
    worker heartbeat returns. A pause whose reason is the Ctrl+Alt+F10 user
    chord is never resumed.
    Never reports a stall or triggers recovery while the bot is PAUSED, IDLE,
    STOPPED, or in RESTING_BREAK.
    """

    # Exact pause reason this watchdog sets for a stall. Heartbeat recovery
    # resumes the route only when the worker is paused for this reason.
    STALL_PAUSE_REASON = "Watchdog safety pause: worker stalled"
    # Ctrl+Alt+F10 chord. Never auto-resume a pause that carries this label.
    USER_PAUSE_CHORD = "Ctrl+Alt+F10"

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
        recovery_cooldown_sec: float = 30.0,
        probe_cache_ttl_sec: float = 2.0,
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
        # v20/game-window: pencere-varlık durumu + tanı seyretme.
        # _last_game_window_present: son _find_game_window sonucu
        # (None = henüz bakılmadı); tanı yalnızca True/None → False
        # geçişinde ve sonra en fazla 30 sn'de bir basılır.
        self._last_game_window_present: Optional[bool] = None
        self._last_game_window_diag_ts: float = 0.0
        # EnumWindows geri-çağrı nesneleri için kalıcı referanslar
        # (v20-hotfix): ctypes geri-çağrısı GC edilirse Windows
        # geri-çağrısı çöker; _find_game_window her çağrıda yeniler.
        self._enum_cb_ref: Optional[Any] = None
        self._enum_proc_cb_ref: Optional[Any] = None
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
        # A stall that persists must not fire the recovery callback every
        # second: at most one recovery per cooldown window.
        self.recovery_cooldown_sec: float = max(0.0, float(recovery_cooldown_sec))
        # P2-21 (F031-02): stall warning'i de aynı pencerede throttle'la —
        # recovery throttle'lıydı ama warning her saniye basılıyordu
        # (5 dk stall ≈ 300 satır). Sayaç (_stall_count) her denetimde
        # artmaya devam eder; yalnızca warning + bildirim + anlık
        # görüntü bu pencerede bir kez üretilir.
        self._last_stall_warning_ts: float = 0.0
        self._last_worker_hp_ratio: Optional[float] = None
        self._worker_was_dead: bool = False
        self._last_death_alert_ts: float = 0.0
        self._last_low_hp_alert_ts: float = 0.0

        # Probe-result cache (see is_display_valid / _cached_memory_stats):
        # display resolution, window state and screen-grab probes are slow
        # OS calls, and get_status()/check_health() poll them about once
        # per second. Display/memory state changes rarely, so results are
        # cached for probe_cache_ttl_sec; a change is therefore reported
        # at most one TTL late, which is negligible next to the 15 s stall
        # timeout. TTL 0 disables caching (every call probes, as before).
        # simulate_display_anomaly()/reset_display_baseline() invalidate
        # the display cache immediately.
        self._probe_cache_ttl_sec: float = max(0.0, float(probe_cache_ttl_sec))
        self._mem_stats_cache: Optional[Dict[str, float]] = None
        self._mem_stats_cache_ts: float = 0.0
        self._display_valid_cache: Optional[bool] = None
        self._display_valid_cache_ts: float = 0.0
        self._last_display_alert_ts: Dict[str, float] = {}

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

    def set_freeze_on_pause(self, enabled: bool) -> None:
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
        if self._is_paused or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES:
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
                    if worker_state in self.PAUSED_OR_IDLE_STATES:
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

    @classmethod
    def _is_user_chord_reason(cls, reason: str) -> bool:
        """True when a pause reason is the Ctrl+Alt+F10 user chord."""
        text = str(reason or "").strip().lower()
        return bool(text) and cls.USER_PAUSE_CHORD.lower() in text

    @classmethod
    def _is_stall_pause_reason(cls, reason: str) -> bool:
        """True when a pause reason is the stall reason this watchdog sets."""
        return str(reason or "").strip() == cls.STALL_PAUSE_REASON

    @staticmethod
    def _read_worker_pause_reason(worker: Any) -> str:
        try:
            return str(getattr(worker, "_pause_reason", "") or "").strip()
        except Exception as exc:
            logger.debug("Worker pause reason read failed: %s", exc)
            return ""

    @staticmethod
    def _worker_is_paused(worker: Any) -> bool:
        try:
            flag = getattr(worker, "is_paused", False)
            if callable(flag):
                flag = flag()
            return bool(flag)
        except Exception as exc:
            logger.debug("Worker paused flag read failed: %s", exc)
            return False

    def _worker_paused_by_user_chord(self, worker: Any) -> bool:
        """True only while the worker is paused by the Ctrl+Alt+F10 chord."""
        if worker is None or not self._worker_is_paused(worker):
            return False
        return self._is_user_chord_reason(self._read_worker_pause_reason(worker))

    def _ensure_stall_pause(self, worker: Any) -> bool:
        """
        Pause the worker with STALL_PAUSE_REASON after a stall.
        Leaves a Ctrl+Alt+F10 pause untouched.
        """
        if worker is None or not callable(getattr(worker, "pause", None)):
            return False
        if self._worker_paused_by_user_chord(worker):
            return False
        reason = self._read_worker_pause_reason(worker)
        if not (self._worker_is_paused(worker) and self._is_stall_pause_reason(reason)):
            worker.pause(self.STALL_PAUSE_REASON)
        reason_after = self._read_worker_pause_reason(worker)
        return self._worker_is_paused(worker) and self._is_stall_pause_reason(reason_after)

    def _stall_resume_candidate_unlocked(self) -> Optional[Any]:
        """Worker to resume, if it is paused for this watchdog's stall reason."""
        worker = self.worker
        if worker is None or not callable(getattr(worker, "resume", None)):
            return None
        if self._worker_paused_by_user_chord(worker):
            return None
        if not self._worker_is_paused(worker):
            return None
        if not self._is_stall_pause_reason(self._read_worker_pause_reason(worker)):
            return None
        return worker

    def _resume_after_stall(self, worker: Any) -> None:
        """Resume a stall pause once the worker heartbeat is alive again."""
        try:
            if self._worker_paused_by_user_chord(worker):
                return
            reason = self._read_worker_pause_reason(worker)
            if not self._is_stall_pause_reason(reason) or not self._worker_is_paused(worker):
                return
            worker.resume()
            logger.info(
                "Watchdog resumed route after stall heartbeat (%s).",
                self.STALL_PAUSE_REASON,
            )
        except Exception as exc:
            logger.error("Watchdog stall resume failed: %s", exc, exc_info=True)
            return

        with self._lock:
            now = time.time()
            self._is_paused = False
            self._is_frozen = False
            self._frozen_since_ts = None
            self._is_stalled = False
            self._last_heartbeat_ts = now
            state = getattr(worker, "state", "WALKING")
            state_name = str(getattr(state, "name", state) or "WALKING").strip().upper()
            if state_name in self.PAUSED_OR_IDLE_STATES or not state_name:
                state_name = "WALKING"
            self._last_state = state_name

    def record_heartbeat(
        self,
        state: str = "RUNNING",
        mode: str = "route",
        latency_ms: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Records a live heartbeat from the main BotWorker loop.
        Updates state, mode, FPS/loop latency metrics, and clears any active stall flag.
        If the worker is paused for STALL_PAUSE_REASON, the returning heartbeat
        resumes the route. A Ctrl+Alt+F10 pause is left paused.
        """
        try:
            now = time.time()
            resume_target: Optional[Any] = None
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
                self._is_paused = state_str in self.PAUSED_OR_IDLE_STATES
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

                metrics = self._build_metrics_unlocked(now)
                resume_target = self._stall_resume_candidate_unlocked()

            if resume_target is not None:
                self._resume_after_stall(resume_target)
                with self._lock:
                    metrics = self._build_metrics_unlocked(time.time())
            return metrics
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

    def _cached_memory_stats(self, now: float) -> Dict[str, float]:
        """Memory stats, cached for _probe_cache_ttl_sec (caller holds _lock).

        On Windows each query is two Win32 API round-trips, and
        _build_metrics_unlocked() runs on every heartbeat, so re-querying
        every time is wasted work for a value that barely moves within
        2 s. TTL 0 disables caching.
        """
        if (
            self._probe_cache_ttl_sec > 0.0
            and self._mem_stats_cache is not None
            and (now - self._mem_stats_cache_ts) < self._probe_cache_ttl_sec
        ):
            return self._mem_stats_cache
        stats = _query_memory_stats()
        self._mem_stats_cache = stats
        self._mem_stats_cache_ts = now
        return stats

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

    def _find_game_window(self) -> Tuple[Optional[int], str]:
        """Returns (hwnd, title) of the first matching game window, or (None, "").

        Eşleşme katmanları (öncelik sırası):
        1. FindWindowW tam başlık eşleşmesi (hızlı yol).
        2. Görünür pencerelerde tam → önek (S147, v19/P3-8) →
           büyük/küçük harf duyarsız alt-dize (v20) — en iyi öncelik
           kazanır (match_game_window_title).
        3. Süreç yedeği (v20): başlık hiç eşleşmezse, exe adı "aion"
           içeren sürecin görünür penceresi kabul edilir.
        Hiçbiri tutmazsa seyreltilmiş tanı logu basılır (gerçek pencere
        başlığı tahmin edilmez — canlı logda görülür).
        """
        if self._simulated_minimized:
            return 1, "SimulatedMinimizedWindow"
        if not sys.platform.startswith("win") or not self.target_window_titles:
            return None, ""
        visible_titles: List[str] = []
        # Numaralandırma hatası + gezilen pencere sayısı (tanı için):
        # "0 pencere" ile "EnumWindows attı" ancak böyle ayırt edilir.
        enum_error: Optional[str] = None
        visited: List[int] = [0]
        try:
            user32 = ctypes.windll.user32
            user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            user32.FindWindowW.restype = wintypes.HWND
            user32.IsIconic.argtypes = [wintypes.HWND]
            user32.IsIconic.restype = wintypes.BOOL
            for title in self.target_window_titles:
                hwnd = user32.FindWindowW(None, str(title))
                if hwnd and int(hwnd) != 0:
                    self._last_game_window_present = True
                    return int(hwnd), str(title)
            # Katman 2: görünür pencerelerde katmanlı başlık eşleşmesi.
            # Aynı geçişte tanı için görünen başlıklar da toplanır.
            # _check_window hem EnumWindows geri-çağrısı hem de katman 2b
            # üst-düzey zincir yürüyüşü tarafından kullanılır.
            best: list = []  # [(öncelik, hwnd, eşleşen_hedef)]
            try:
                user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
                user32.EnumWindows.restype = wintypes.BOOL
                user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
                user32.GetWindowTextW.restype = ctypes.c_int
                user32.IsWindowVisible.argtypes = [wintypes.HWND]
                user32.IsWindowVisible.restype = wintypes.BOOL

                def _check_window(hwnd):
                    """Tek pencereyi değerlendir; durmak gerekiyorsa False döner."""
                    visited[0] += 1
                    try:
                        if not user32.IsWindowVisible(hwnd):
                            return True
                        buf = ctypes.create_unicode_buffer(512)
                        if user32.GetWindowTextW(hwnd, buf, 512) <= 0:
                            return True
                        text = buf.value or ""
                        if len(visible_titles) < _GAME_WINDOW_DIAG_MAX_TITLES and text:
                            visible_titles.append(text[:_GAME_WINDOW_DIAG_TITLE_LEN])
                        matched = match_game_window_title(text, self.target_window_titles)
                        if matched is None:
                            return True
                        matched_title, priority = matched
                        if not best or priority < best[0][0]:
                            best.clear()
                            best.append((priority, int(hwnd), matched_title))
                            if priority == 0:
                                return False
                    except Exception:
                        return True
                    return True

                def _enum_cb(hwnd, _lparam):
                    return _check_window(hwnd)

                # Kalıcı referans: ctypes geri-çağrı nesnesi çağrı
                # boyunca canlı tutulmalı; geçici nesne GC edilirse
                # Windows geri-çağrısı çöker.
                self._enum_cb_ref = _WNDENUMPROC(_enum_cb)
                enum_ok = user32.EnumWindows(self._enum_cb_ref, 0)
                if not enum_ok and not best:
                    enum_error = "EnumWindows returned False (title layer)"
                if best:
                    self._last_game_window_present = True
                    return best[0][1], best[0][2]
            except Exception as exc:
                enum_error = f"title fallback: {exc}"
            # Katman 2b: EnumWindows hiç pencere gezmediyse üst-düzey
            # zincir (GetTopWindow → GetNextWindow/GW_HWNDNEXT) yedeği.
            if not best and visited[0] == 0:
                try:
                    user32.GetTopWindow.argtypes = [wintypes.HWND]
                    user32.GetTopWindow.restype = wintypes.HWND
                    user32.GetNextWindow.argtypes = [wintypes.HWND, wintypes.UINT]
                    user32.GetNextWindow.restype = wintypes.HWND
                    _GW_HWNDNEXT = 2
                    _hwnd = user32.GetTopWindow(None)
                    _steps = 0
                    while _hwnd and int(_hwnd) != 0 and _steps < 512:
                        _steps += 1
                        if not _check_window(int(_hwnd)):
                            break
                        _hwnd = user32.GetNextWindow(_hwnd, _GW_HWNDNEXT)
                    if best:
                        self._last_game_window_present = True
                        return best[0][1], best[0][2]
                except Exception as exc:
                    enum_error = f"top-window chain fallback: {exc}"
            # Katman 3: süreç yedeği — başlık eşleşmesi yoksa exe adı
            # "aion" içeren sürecin görünür penceresini kabul et.
            try:
                kernel32 = ctypes.windll.kernel32
                kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                kernel32.OpenProcess.restype = wintypes.HANDLE
                kernel32.QueryFullProcessImageNameW.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.LPWSTR,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
                kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
                kernel32.CloseHandle.restype = wintypes.BOOL
                user32.GetWindowThreadProcessId.argtypes = [
                    wintypes.HWND,
                    ctypes.POINTER(wintypes.DWORD),
                ]
                user32.GetWindowThreadProcessId.restype = wintypes.DWORD
                found_proc: list = []

                def _enum_proc_cb(hwnd, _lparam):
                    if found_proc:
                        return False
                    try:
                        if not user32.IsWindowVisible(hwnd):
                            return True
                        pid = wintypes.DWORD(0)
                        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                        if not pid.value:
                            return True
                        # PROCESS_QUERY_LIMITED_INFORMATION
                        hproc = kernel32.OpenProcess(0x1000, False, pid.value)
                        if not hproc:
                            return True
                        try:
                            size = wintypes.DWORD(1024)
                            exe_buf = ctypes.create_unicode_buffer(1024)
                            if not kernel32.QueryFullProcessImageNameW(
                                hproc, 0, exe_buf, ctypes.byref(size)
                            ):
                                return True
                            exe_name = (exe_buf.value or "").rsplit("\\", 1)[-1]
                            if not game_process_exe_matches(exe_name):
                                return True
                            title_buf = ctypes.create_unicode_buffer(512)
                            title = ""
                            if user32.GetWindowTextW(hwnd, title_buf, 512) > 0:
                                title = title_buf.value or ""
                            found_proc.append((int(hwnd), title or exe_name))
                            return False
                        finally:
                            kernel32.CloseHandle(hproc)
                    except Exception:
                        return True
                    return True

                # Kalıcı referans (katman 2'deki gerekçeyle aynı).
                self._enum_proc_cb_ref = _WNDENUMPROC(_enum_proc_cb)
                enum_ok = user32.EnumWindows(self._enum_proc_cb_ref, 0)
                if not enum_ok and not found_proc:
                    enum_error = "EnumWindows returned False (process layer)"
                if found_proc:
                    self._last_game_window_present = True
                    return found_proc[0]
            except Exception as exc:
                enum_error = f"process fallback: {exc}"
        except Exception as exc:
            enum_error = f"window lookup: {exc}"
        self._maybe_log_game_window_diagnostic(
            visible_titles, visited=visited[0], enum_error=enum_error
        )
        self._last_game_window_present = False
        return None, ""

    def _maybe_log_game_window_diagnostic(
        self,
        visible_titles: List[str],
        visited: int = 0,
        enum_error: Optional[str] = None,
    ) -> None:
        """Oyun penceresi bulunamadığında görünen başlıkları seyreltilmiş basar.

        v20/game-window tanısı: gerçek pencere başlığı TAHMİN EDİLMEZ;
        False'a ilk geçişte ve sonra en fazla 30 sn'de bir, o anki ilk
        ~10 görünür üst-düzey pencere başlığı (kısaltılmış) loga yazılır
        ki canlı logda gerçek başlık görülsün. Windows dışında çalışmaz.
        v20-hotfix: gezilen pencere sayısı (visited) ve numaralandırma
        hatası (enum_error) da basılır — "0 pencere" ile "EnumWindows
        attı" ancak böyle ayırt edilir.
        """
        if not sys.platform.startswith("win"):
            return
        now = time.time()
        transitioned = self._last_game_window_present is not False
        due = (
            now - self._last_game_window_diag_ts
        ) >= _GAME_WINDOW_DIAG_INTERVAL_SEC
        if not transitioned and not due:
            return
        self._last_game_window_diag_ts = now
        if enum_error:
            logger.warning("Game window enumeration failed: %s", enum_error)
        logger.warning(
            "Game window not found; enumerated %d windows; "
            "visible top-level windows (%d): %s",
            visited,
            len(visible_titles),
            visible_titles,
        )

    def is_game_window_present(self) -> Optional[bool]:
        """Oyun penceresi şu an açık mı?

        fix/panik-kilit-oyun-yok: PanicAlertSystem'in oyun-yokken idle
        sayacını duraklatması için üretim kaynağı (bind üzerinden
        game_window_present detektörü olarak bağlanır).
        - True: hedef başlıklardan biriyle eşleşen pencere bulundu.
          Eşleşme katmanlıdır (v20): tam başlık → önek → süreç exe adı
          ("aion" içeren) → büyük/küçük harf duyarsız alt-dize.
        - False: Windows'ta hedef başlıklar tanımlı ama pencere yok
          (oyun kapalı / çökmüş).
        - None: belirlenemiyor — Windows dışı platform ya da hedef
          başlık yok; çağrıcı eski davranışa döner (idle sayacı çalışır).
        Simüle-minimize durumda pencere "var" sayılır (yalnızca simge
        durumuna indirgenmiş, kapanmamış).
        """
        if not sys.platform.startswith("win") or not self.target_window_titles:
            return None
        hwnd, _title = self._find_game_window()
        return hwnd is not None

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
            hwnd, title = self._find_game_window()
            if hwnd is not None:
                user32 = ctypes.windll.user32
                if bool(user32.IsIconic(hwnd)):
                    return True, title
        except Exception as exc:
            logger.debug("Window minimized check failed: %s", exc)

        return False, ""

    def _verify_screen_capture_health(self) -> bool:
        """Performs a fast 16x16 screen grab test to verify capture subsystem health.

        A failed grab means capture is NOT healthy: monitor geometry alone
        can never prove that frames can actually be grabbed.
        """
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
                    logger.warning(
                        "Screen capture grab failed (monitor %dx%d present): %s",
                        mon_w,
                        mon_h,
                        grab_exc,
                    )
                    return False
        except Exception as exc:
            logger.warning("Screen capture health probe failed: %s", exc)
            return False

    def is_display_valid(self) -> bool:
        """Cached display-validity probe (real work: _probe_display_valid).

        The underlying probe does a resolution query, Win32 window lookups
        and a real screen grab — slow OS calls — while get_status() and
        check_health() poll this about once per second. Results are cached
        for probe_cache_ttl_sec (default 2 s), so a display change is
        reported at most one TTL late; that is negligible next to the 15 s
        stall timeout. simulate_display_anomaly() and
        reset_display_baseline() invalidate the cache immediately, so
        self-tests and the F9 flow always see the fresh state.
        """
        now = time.time()
        with self._lock:
            if (
                self._probe_cache_ttl_sec > 0.0
                and self._display_valid_cache is not None
                and (now - self._display_valid_cache_ts) < self._probe_cache_ttl_sec
            ):
                return self._display_valid_cache
        result = self._probe_display_valid()
        with self._lock:
            self._display_valid_cache = result
            self._display_valid_cache_ts = time.time()
        return result

    def _probe_display_valid(self) -> bool:
        """
        Uncached display probe: checks whether the display resolution is
        valid and unchanged, the game window is not minimized, and screen
        capture is operational.
        Emits automatic notifications when resolution changes or the window is minimized.

        Slow I/O (resolution query, window lookup, screen grab) runs OUTSIDE
        the lock; only the baseline comparison/update is locked, so a
        per-second get_status() poll can no longer stall other threads.
        A resolution change is reported once and then adopted as the new
        baseline, instead of staying invalid and alerting every second.
        """
        try:
            with self._lock:
                baseline = self._baseline_resolution
                last_res = self._last_resolution
                simulated_res = self._simulated_resolution

            current_res = (
                simulated_res
                if simulated_res is not None
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

            if current_res != baseline:
                # Report once, then adopt the new resolution as baseline.
                msg = (
                    f"Screen resolution changed from {baseline[0]}x"
                    f"{baseline[1]} to {cur_w}x{cur_h}!"
                )
                logger.warning(msg)
                resolution_alert: Optional[Tuple[str, str, Dict[str, Any]]] = None
                resolution_snapshot: Optional[Dict[str, Any]] = None
                with self._lock:
                    # Only alert when the resolution actually moved again;
                    # a repeated identical reading must not re-alert.
                    if current_res != self._last_resolution:
                        resolution_alert = (
                            "RESOLUTION_CHANGED",
                            msg,
                            {
                                "baseline_resolution": baseline,
                                "previous_resolution": last_res,
                                "current_resolution": current_res,
                            },
                        )
                        resolution_snapshot = {
                            "previous_resolution": last_res,
                            "current_resolution": current_res,
                        }
                    self._last_resolution = current_res
                    self._baseline_resolution = current_res
                # Alert callback + incident snapshot do I/O and run user
                # code: never under the lock, or a slow recorder/callback
                # stalls heartbeats (same rule as check_health).
                if resolution_alert is not None:
                    self._emit_display_alert(*resolution_alert)
                if resolution_snapshot is not None:
                    self.trigger_incident_snapshot(
                        "Resolution changed", resolution_snapshot
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

            # A missing game window is not "healthy": the game may have
            # crashed. (Windows only; on other platforms there is no
            # Win32 window to look for.)
            if sys.platform.startswith("win") and self.target_window_titles:
                hwnd, _title = self._find_game_window()
                if hwnd is None:
                    msg = "Game window not found (game may have crashed)."
                    logger.warning(msg)
                    self._emit_display_alert("GAME_WINDOW_MISSING", msg, {})
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
        """Records a display alert and invokes display_alert_callback if configured.

        The same alert type is emitted at most once per
        _DISPLAY_ALERT_REPEAT_SUPPRESS_SEC: a persistent failure (e.g.
        capture failing on a headless machine) re-alerting on every 1 Hz
        poll would flood the 100-entry notification ring and evict
        unrelated entries such as WORKER_STALLED. The live status dict
        still reports display_valid=False on every poll, so no state is
        hidden — only the repeated notification/callback is suppressed.
        (The timestamp dict is written without the lock; the worst case
        of a race is one duplicate alert, never a lost one.)
        """
        now = time.time()
        last_ts = self._last_display_alert_ts.get(event_type, 0.0)
        if now - last_ts < _DISPLAY_ALERT_REPEAT_SUPPRESS_SEC:
            return
        self._last_display_alert_ts[event_type] = now
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
            # Simulated state changes what the probe returns: drop the cache.
            self._display_valid_cache = None

    def reset_display_baseline(self) -> Tuple[int, int]:
        """Resets the display baseline to the current system resolution and clears simulations."""
        with self._lock:
            self._simulated_resolution = None
            self._simulated_minimized = False
            self._baseline_resolution = _query_system_resolution()
            self._last_resolution = self._baseline_resolution
            # Baseline changed: the cached verdict no longer applies.
            self._display_valid_cache = None
            return self._baseline_resolution

    def is_stalled(self, now: Optional[float] = None) -> bool:
        """Returns True if more than stall_timeout_sec has elapsed since the last heartbeat."""
        with self._lock:
            current_time = time.time() if now is None else float(now)
            if self._is_frozen_unlocked() or self._is_paused_or_idle_unlocked():
                # Read-only query: never refresh the heartbeat here, or a
                # polling UI would keep resetting the stall timer.
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
            pending_snapshot: Optional[Tuple[str, Dict[str, Any]]] = None

            with self._lock:
                if (
                    self._is_frozen_unlocked()
                    or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
                    or self._is_paused
                    or self._is_paused_or_idle_unlocked()
                ):
                    self._is_stalled = False
                    stalled_now = False
                else:
                    elapsed = max(0.0, now - self._last_heartbeat_ts)
                    stalled_now = elapsed > self.stall_timeout_sec

                if stalled_now:
                    self._is_stalled = True
                    self._stall_count += 1
                    # P2-21 (F031-02): warning throttle — kalıcı stall'da
                    # her saniye warning + bildirim + anlık görüntü yok;
                    # pencerede (recovery_cooldown_sec, 30 sn) bir kez.
                    warn_now = (
                        now - self._last_stall_warning_ts
                    ) >= self.recovery_cooldown_sec
                    if warn_now:
                        self._last_stall_warning_ts = now
                    warn_msg = (
                        f"BotWorker stalled! No heartbeat for {elapsed:.2f}s "
                        f"(timeout={self.stall_timeout_sec:.1f}s, last_state={self._last_state}, "
                        f"last_mode={self._last_mode})."
                    )
                    if warn_now:
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
                    # Snapshot is collected here but recorded AFTER the
                    # lock is released (below): record_incident does file
                    # and cv2 I/O, and holding the lock through it stalls
                    # record_heartbeat() on other threads for its whole
                    # duration. The notification above stays under the lock.
                    # Throttle'lıdır (warning ile aynı pencere): kalıcı
                    # stall'da her saniye dosya+cv2 anlık görüntüsü yok.
                    pending_snapshot = (
                        (
                            "Worker stalled",
                            {
                                "elapsed_sec": round(elapsed, 2),
                                "state": self._last_state,
                                "mode": self._last_mode,
                            },
                        )
                        if warn_now
                        else None
                    )
                    invoke_recovery = True
                    recovery_payload = self._build_metrics_unlocked(now)

            if pending_snapshot is not None:
                self.trigger_incident_snapshot(*pending_snapshot)

            if invoke_recovery:
                # Cooldown: a persisting stall fires recovery at most once
                # per recovery_cooldown_sec instead of every second.
                if (now - self._last_recovery_ts) >= self.recovery_cooldown_sec:
                    self._execute_safe_recovery(recovery_payload)
                else:
                    logger.debug(
                        "Watchdog recovery throttled (%.1fs since last recovery).",
                        now - self._last_recovery_ts,
                    )

            self._check_worker_vitals()

            display_ok = self.is_display_valid()
            with self._lock:
                snapshot = self._build_metrics_unlocked(time.time())
                snapshot["display_valid"] = display_ok
                return snapshot
        except Exception as exc:
            logger.error("Error in check_health(): %s", exc, exc_info=True)
            return {"stalled": False, "display_valid": False, "error": str(exc)}

    def _check_worker_vitals(self) -> None:
        """Monitor worker death / HP / revival when the worker exposes vitals.

        Reads (never requires) ``hp_ratio`` / ``hp_pct`` / ``health_ratio``
        or ``current_hp`` / ``max_hp`` and a ``DEAD``-like state from the
        attached worker. Death and low-HP alerts are throttled to one per
        60s; a revival (HP back above zero after death) is reported once.
        Workers without vitals are a silent no-op.
        """
        worker = self.worker
        if worker is None:
            return
        try:
            hp_ratio: Optional[float] = None
            for attr in ("hp_ratio", "hp_pct", "health_ratio"):
                raw = getattr(worker, attr, None)
                if raw is not None:
                    try:
                        hp_ratio = max(0.0, min(1.0, float(raw)))
                    except (TypeError, ValueError):
                        hp_ratio = None
                    break
            if hp_ratio is None:
                cur = getattr(worker, "current_hp", None)
                max_hp = getattr(worker, "max_hp", None)
                if cur is not None and max_hp:
                    try:
                        hp_ratio = max(0.0, min(1.0, float(cur) / float(max_hp)))
                    except (TypeError, ValueError, ZeroDivisionError):
                        hp_ratio = None
            state_raw = getattr(worker, "state", None)
            state_name = str(getattr(state_raw, "name", state_raw) or "").strip().upper()
            is_dead = hp_ratio is not None and hp_ratio <= 0.0
            is_dead = is_dead or state_name in ("DEAD", "DIED", "CHARACTER_DEAD", "ÖLÜ")

            now = time.time()
            pending_snapshot: Optional[Tuple[str, Dict[str, Any]]] = None
            with self._lock:
                self._last_worker_hp_ratio = hp_ratio
                if is_dead and not self._worker_was_dead:
                    self._worker_was_dead = True
                    if now - self._last_death_alert_ts >= 60.0:
                        self._last_death_alert_ts = now
                        msg = "Worker death detected (HP 0 / DEAD state)."
                        logger.warning(msg)
                        self._add_notification("WORKER_DEAD", msg, {"hp_ratio": hp_ratio})
                        # Recorded after the lock is released (below):
                        # record_incident does file/cv2 I/O.
                        pending_snapshot = (
                            "Character dead",
                            {"hp_ratio": hp_ratio},
                        )
                elif not is_dead and self._worker_was_dead:
                    self._worker_was_dead = False
                    msg = "Worker revived (HP above zero after death)."
                    logger.info(msg)
                    self._add_notification("WORKER_REVIVED", msg, {"hp_ratio": hp_ratio})
                elif (
                    hp_ratio is not None
                    and 0.0 < hp_ratio < 0.25
                    and now - self._last_low_hp_alert_ts >= 60.0
                ):
                    self._last_low_hp_alert_ts = now
                    msg = f"Worker low HP: {hp_ratio * 100:.0f}%."
                    logger.warning(msg)
                    self._add_notification("WORKER_LOW_HP", msg, {"hp_ratio": hp_ratio})
            if pending_snapshot is not None:
                self.trigger_incident_snapshot(*pending_snapshot)
        except Exception as exc:
            logger.debug("Worker vitals check failed: %s", exc)

    def _execute_safe_recovery(self, payload: Dict[str, Any]) -> bool:
        """Invokes the registered recovery callback and pauses a stalled worker.

        The pause reason is STALL_PAUSE_REASON so a later heartbeat can resume
        the route. A worker already paused by Ctrl+Alt+F10 is not paused again.
        """
        recovered = False
        worker: Optional[Any] = None
        try:
            with self._lock:
                if (
                    self._is_frozen_unlocked()
                    or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
                    or self._is_paused
                    or self._is_paused_or_idle_unlocked()
                ):
                    self._is_stalled = False
                    return False
                if (time.time() - self._last_recovery_ts) < self.recovery_cooldown_sec:
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

            if worker is not None and self._worker_paused_by_user_chord(worker):
                return recovered

            if self._ensure_stall_pause(worker):
                recovered = True
                logger.info(
                    "Watchdog paused stalled BotWorker (%s).",
                    self.STALL_PAUSE_REASON,
                )
        except Exception as exc:
            logger.error("Watchdog recovery execution failed: %s", exc, exc_info=True)

        return recovered

    def _build_metrics_unlocked(self, now: float) -> Dict[str, Any]:
        """Constructs the internal health snapshot dictionary (caller holds _lock)."""
        fps, avg_latency_ms = self._compute_fps_and_latency_unlocked()
        mem_info = self._cached_memory_stats(now)
        frozen_now = self._is_frozen_unlocked()
        if (
            frozen_now
            or self._last_state.upper() in self.PAUSED_OR_IDLE_STATES
            or self._is_paused
            or self._is_paused_or_idle_unlocked()
        ):
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
            "recovery_cooldown_sec": self.recovery_cooldown_sec,
            "worker_hp_ratio": self._last_worker_hp_ratio,
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
        """Returns watchdog health summary based on a real internal check:
        sane timeouts and a live (or never started) monitor thread."""
        try:
            with self._lock:
                thread_ok = self._thread is None or not self._running or self._thread.is_alive()
                ready = (
                    self.stall_timeout_sec > 0
                    and self.check_interval_sec > 0
                    and thread_ok
                )
                return {
                    "watchdog_ready": bool(ready),
                    "running": bool(self._running),
                    "stalled": bool(self._is_stalled),
                    "frozen": bool(self._is_frozen),
                }
        except Exception as exc:
            logger.debug("get_watchdog_health_summary failed: %s", exc)
            return {"watchdog_ready": False}

    def start(self) -> bool:
        """Starts the lightweight background watchdog monitoring thread."""
        with self._lock:
            if self._running and self._thread is not None and self._thread.is_alive():
                return True
            old_thread = self._thread
        # Never run two monitor threads: if a previous thread is still
        # alive after stop(), wait for it before replacing the handle.
        if old_thread is not None and old_thread.is_alive():
            old_thread.join(timeout=2.5)
            if old_thread.is_alive():
                logger.error("Önceki watchdog thread'i hâlâ yaşıyor, yeni thread başlatılmadı.")
                return False
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
            if thread_ref.is_alive():
                # Keep the handle so a later start() can see the live
                # thread instead of silently orphaning it.
                logger.warning("Watchdog thread 2.5s içinde durmadı, handle saklanıyor.")
                return
        with self._lock:
            self._thread = None
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
                    self._is_stalled = False
                interval = self.check_interval_sec
            try:
                self.check_health()
            except Exception as exc:
                logger.error("Unexpected error in watchdog monitor loop: %s", exc)
            time.sleep(interval)

    def get_watchdog_ready_flag(self) -> dict:
        # F-0518: was a constant True; now delegates to the real health check.
        return {"module": "watchdog", "ready": bool(self.get_watchdog_health_summary().get("watchdog_ready", False))}

    def get_watchdog_version_info(self) -> dict:
        # F-0518: ready now reflects the real health check, not a constant.
        return {
            "module": "watchdog",
            "version": __version__,
            "ready": bool(self.get_watchdog_health_summary().get("watchdog_ready", False)),
        }

    def get_watchdog_diagnostic_snapshot(self) -> dict:
        # F-0518: status/ready now reflect the real health check.
        summary = self.get_watchdog_health_summary()
        ready = bool(summary.get("watchdog_ready", False))
        return {
            "module": "watchdog",
            "status": "ok" if ready else "error",
            "ready": ready,
            "running": bool(summary.get("running", False)),
            "stalled": bool(summary.get("stalled", False)),
            "frozen": bool(summary.get("frozen", False)),
        }


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
    last 30 lines of the current session log into a clean report.
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

            version = data.get("version") or "Unknown"
            active_reg = data.get("active_region") or "Unknown"
            active_rte = data.get("active_route") or "Unknown"
            gathering = data.get("gathering", {}) or {}
            route_cfg = data.get("route", {}) or {}
            breaks_cfg = data.get("breaks", {}) or {}

            def _fmt(value: Any, suffix: str = "") -> str:
                return f"{value}{suffix}" if value is not None else "Unknown"

            lines.append(f"Bot Version      : {version}")
            lines.append(f"Configured Route : {active_reg} / {active_rte}")
            lines.append(
                f"Gathering Config : f_min={_fmt(gathering.get('f_interval_min', gathering.get('min_f_interval')), 's')}, "
                f"f_max={_fmt(gathering.get('f_interval_max', gathering.get('max_f_interval')), 's')}, "
                f"test_mode={_fmt(gathering.get('test_mode'))}, "
                f"overlay={_fmt(gathering.get('show_overlay'))}"
            )
            lines.append(
                f"Route Config     : rest={_fmt(route_cfg.get('rest_minutes', route_cfg.get('rest_after_visit_min')), 'm')}, "
                f"reach_timeout={_fmt(route_cfg.get('time_to_reach_sec', route_cfg.get('time_to_reach_node_sec')), 's')}, "
                f"fight_back={_fmt(route_cfg.get('fight_back', route_cfg.get('fight_back_when_attacked')))}"
            )
            lines.append(
                f"Break Config     : every={_fmt(breaks_cfg.get('break_every_min', breaks_cfg.get('interval_min')), 'm')}, "
                f"lasts={_fmt(breaks_cfg.get('break_lasts_min', breaks_cfg.get('duration_min')), 'm')}"
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
                f"success={m.get('success_rate_pct', 0.0)}%, "  # S-0101: unknown success is 0.0 (neutral, like the other missing metrics), never an invented 100.0%
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
            # v20/session-logs: bot.log artik yazilmiyor; guncel oturum
            # dosyasina dus (cagirici eski varsayilan yolu gecse bile).
            try:
                from core.session_log import current_session_log_path

                current = current_session_log_path(resolved.parent)
                if current is not None:
                    resolved = current
            except Exception:
                pass

        if not resolved.exists():
            return [f"(Log file not found at {resolved})"]

        # Tail read: bot.log grows without bound, and the old implementation
        # read the whole file into RAM (a 50 MB log cost ~1 s wall time and
        # ~123 MB peak) just to keep the last 30 lines. Seek from the end
        # and read only enough 64 KB chunks to cover max_lines newlines
        # (bounded at 256 KB, far more than 30 log lines). Behaviour for
        # missing/empty/short files, and the returned line format, is
        # unchanged; only the amount of file touched changes.
        try:
            size = resolved.stat().st_size
            chunk_size = 65536
            max_tail_bytes = 256 * 1024
            data = b""
            pos = size
            while pos > 0 and len(data) < max_tail_bytes:
                if max_lines > 0 and data.count(b"\n") > max_lines:
                    break
                read_from = max(0, pos - chunk_size)
                with open(resolved, "rb") as f:
                    f.seek(read_from)
                    data = f.read(pos - read_from) + data
                pos = read_from
            text = data.decode("utf-8", errors="replace")
            lines = text.splitlines()
            if pos > 0 and lines:
                # First line is a partial line cut by the tail window.
                lines = lines[1:]
            tail = [line.rstrip() for line in lines[-max_lines:]]
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
        statistics, template files status, and the last 30 lines of the current session log.
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
    # Display validity is environment-dependent: on a headless machine the
    # screen grab fails and the watchdog must REPORT it (CAPTURE_FAILED),
    # not silently claim a healthy display.
    display_ok = watchdog.is_display_valid()
    if not display_ok:
        assert any(
            n.get("event_type") == "CAPTURE_FAILED" for n in watchdog._notifications
        ), "Failed capture must raise CAPTURE_FAILED, not pass silently"

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


def _probe_watchdog_contract() -> bool:
    """Real probe: required watchdog methods exist and the resolution query
    returns positive dimensions (no import-time side effects)."""
    try:
        for name in (
            "record_heartbeat",
            "check_health",
            "is_stalled",
            "is_display_valid",
            "start",
            "stop",
            "set_freeze_on_pause",
            "freeze",
            "unfreeze",
        ):
            if not callable(getattr(BotWatchdog, name, None)):
                return False
        width, height = _query_system_resolution()
        return int(width) > 0 and int(height) > 0
    except Exception as exc:
        logger.debug("Watchdog contract probe failed: %s", exc)
        return False


def get_watchdog_capabilities() -> list:
    return ["watchdog", "stuck_recovery", "diagnostic_snapshot"]


def validate_watchdog_contract() -> bool:
    return _probe_watchdog_contract()


def get_watchdog_baseline_metrics() -> dict:
    return {"module": "watchdog", "contract_valid": validate_watchdog_contract(), "capability_count": len(get_watchdog_capabilities())}


def get_watchdog_readiness_status() -> dict:
    return {"module": "watchdog", "ready": validate_watchdog_contract(), "metrics": get_watchdog_baseline_metrics()}


def get_watchdog_audit_record() -> dict:
    return {"module": "watchdog", "audit_passed": validate_watchdog_contract(), "readiness": get_watchdog_readiness_status()}


def get_watchdog_operational_digest() -> dict:
    return {"module": "watchdog", "operational": validate_watchdog_contract(), "audit": get_watchdog_audit_record()}


def get_watchdog_telemetry_payload() -> dict:
    return {"module": "watchdog", "telemetry_ready": validate_watchdog_contract(), "digest": get_watchdog_operational_digest()}


def get_watchdog_integrity_seal() -> dict:
    return {"module": "watchdog", "sealed": validate_watchdog_contract(), "telemetry": get_watchdog_telemetry_payload()}


def get_watchdog_compliance_certificate() -> dict:
    return {"module": "watchdog", "certified": validate_watchdog_contract(), "seal": get_watchdog_integrity_seal()}


def get_watchdog_readiness_probe() -> dict:
    return {"module": "watchdog", "ready": validate_watchdog_contract(), "certificate": get_watchdog_compliance_certificate()}


def get_watchdog_audit_summary() -> dict:
    return {"module": "watchdog", "audit_passed": validate_watchdog_contract(), "probe": get_watchdog_readiness_probe()}



