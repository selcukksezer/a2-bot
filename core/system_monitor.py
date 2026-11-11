# -*- coding: utf-8 -*-
"""
core/system_monitor.py

Lightweight, safe, and cached hardware and screen capture metrics engine.
Provides non-blocking snapshots for GPU (pynvml), CPU (psutil), and MSS capture.
Gracefully handles environments without NVIDIA GPUs or pynvml.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("Gatherion.SystemMonitor")

# Module-level state for NVML initialization
_NVML_INITIALIZED: Optional[bool] = None
_NVML_HANDLE = None
_MONITOR_LOCK = threading.Lock()
_CACHED_SNAPSHOT: Optional[Dict[str, Any]] = None
_LAST_SAMPLE_TS: float = 0.0
_CACHE_TTL_SEC: float = 0.5  # 500ms caching to prevent excessive polling overhead
_CPU_PRIMED: bool = False


def _prime_cpu_percent() -> None:
    """Prime psutil's cpu_percent baseline once (F-0394).

    psutil.cpu_percent(interval=None) returns a meaningless 0.0 on its very
    first call. Priming at import makes the first real snapshot measure CPU
    use over the time since import instead of reporting 0.0.
    """
    global _CPU_PRIMED
    if _CPU_PRIMED:
        return
    try:
        import psutil

        psutil.cpu_percent(interval=None)
        _CPU_PRIMED = True
    except Exception as exc:  # pragma: no cover - psutil absent
        logger.debug("Could not prime psutil cpu_percent: %s", exc)


_prime_cpu_percent()


def _init_nvml_if_needed() -> bool:
    """Attempts to initialize NVML safely once. Returns True if available."""
    global _NVML_INITIALIZED, _NVML_HANDLE
    if _NVML_INITIALIZED is not None:
        return _NVML_INITIALIZED

    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
            _NVML_INITIALIZED = True
            logger.info("NVML initialized successfully with %d GPU(s).", device_count)
            return True
        else:
            _NVML_INITIALIZED = False
            return False
    except Exception as exc:
        logger.debug("NVML not available or initialization failed: %s", exc)
        _NVML_INITIALIZED = False
        _NVML_HANDLE = None
        return False


def _gpu_unavailable() -> Dict[str, Any]:
    """Fresh 'no usable GPU' fallback dict (D25: was an identical 11-line
    literal duplicated in _query_gpu_metrics' NVML-missing and query-error
    paths). A new dict per call, so no caller can mutate a shared one."""
    return {
        "name": "GPU · Entegre/Bilinmiyor",
        "raw_name": "Integrated/Unknown",
        "pct": 0,
        "temp": 0,
        "temp_str": "—°C",
        "mem_used_gb": 0.0,
        "mem_total_gb": 0.0,
        "label_str": "0% · —°C · 0.0/0.0 GB",
        "available": False,
    }


def _copy_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Caller-owned copy of a cached snapshot.

    Every section (gpu/cpu/capture) is a flat dict of scalars plus a float
    timestamp, so copying each section dict isolates the caller from the
    cache exactly as deepcopy did here, at a fraction of the cost (D25,
    measured on this VM: cached-path get_snapshot min 98.3us/call with
    deepcopy -> min 14.7us/call with this copy). If a section
    ever gains a nested value, this must go back to deepcopy — the
    flat-structure assumption is locked by test_d25 (section values must
    all be scalars).
    """
    return {key: (dict(value) if isinstance(value, dict) else value) for key, value in snapshot.items()}


def _query_gpu_metrics() -> Dict[str, Any]:
    """Queries NVIDIA GPU load, temperature, and VRAM via pynvml or returns clean idle fallback."""
    if not _init_nvml_if_needed():
        return _gpu_unavailable()

    try:
        import pynvml
        h = _NVML_HANDLE
        if h is None:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)

        raw_name = pynvml.nvmlDeviceGetName(h)
        if isinstance(raw_name, bytes):
            raw_name = raw_name.decode("utf-8", errors="ignore")
        short_name = (
            str(raw_name)
            .replace("NVIDIA GeForce ", "")
            .replace("NVIDIA ", "")
            .strip()
        )
        gpu_name = f"GPU · {short_name}" if short_name else "GPU · NVIDIA"

        util_obj = pynvml.nvmlDeviceGetUtilizationRates(h)
        gpu_pct = int(getattr(util_obj, "gpu", 0))

        try:
            temp = int(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
            temp_str = f"{temp}°C"
        except Exception:
            temp = 0
            temp_str = "—°C"

        try:
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used_gb = round(mem.used / (1024**3), 1)
            mem_total_gb = round(mem.total / (1024**3), 1)
        except Exception:
            mem_used_gb = 0.0
            mem_total_gb = 0.0

        label_str = f"{gpu_pct}% · {temp_str} · {mem_used_gb:.1f}/{mem_total_gb:.1f} GB"

        return {
            "name": gpu_name,
            "raw_name": str(raw_name),
            "pct": max(0, min(100, gpu_pct)),
            "temp": temp,
            "temp_str": temp_str,
            "mem_used_gb": mem_used_gb,
            "mem_total_gb": mem_total_gb,
            "label_str": label_str,
            "available": True,
        }
    except Exception as exc:
        logger.debug("NVML query error: %s", exc)
        return _gpu_unavailable()


def _query_cpu_metrics() -> Dict[str, Any]:
    """Queries CPU utilization, core count, and clock frequency via psutil.

    Unknown values are reported as ``None`` with an ``available: False`` flag
    and a "—" placeholder in the label (F-0200): no invented 2.4 GHz / 4-core
    fallback is ever presented as a real measurement.
    """
    _prime_cpu_percent()
    try:
        import psutil
        cores_raw = psutil.cpu_count(logical=True)
        cores = int(cores_raw) if cores_raw else None
        pct = int(psutil.cpu_percent(interval=None))
        freq_info = psutil.cpu_freq()
        if freq_info and getattr(freq_info, "current", None):
            freq_ghz: Optional[float] = round(freq_info.current / 1000.0, 1)
        else:
            freq_ghz = None
        freq_str = f"{freq_ghz:.1f} GHz" if freq_ghz is not None else "— GHz"
        cores_str = f"{cores} Çekirdek" if cores is not None else "Bilinmiyor"
        label_str = f"{pct}% · {freq_str}"
        return {
            "name": f"CPU · {cores_str}",
            "cores": cores,
            "pct": max(0, min(100, pct)),
            "freq_ghz": freq_ghz,
            "label_str": label_str,
            "available": freq_ghz is not None and cores is not None,
        }
    except Exception as exc:
        logger.debug("psutil query error: %s", exc)
        return {
            "name": "CPU · Bilinmiyor",
            "cores": None,
            "pct": 0,
            "freq_ghz": None,
            "label_str": "0% · — GHz",
            "available": False,
        }


def _query_capture_metrics(capture_instance: Optional[Any] = None) -> Dict[str, Any]:
    """Extracts screen capture FPS and latency from the provided capture instance.

    Latency is reported only when the capture instance actually measures it
    (``last_latency_ms`` / ``latency_ms`` / ``frame_latency_ms`` /
    ``last_frame_time_ms`` / ``frame_time_ms``) — it is never derived from FPS
    as ``1000/fps`` (F-0395). ``latency_ms`` stays numeric (0.0) for callers
    that do arithmetic on it; ``latency_measured`` tells whether it is real.
    """
    fps = 0.0
    latency_ms = 0.0
    latency_measured = False

    try:
        if capture_instance is not None:
            raw_fps = getattr(capture_instance, "last_fps", 0.0)
            if raw_fps is not None:
                try:
                    f = float(raw_fps)
                    if math.isfinite(f):
                        fps = f
                    else:
                        fps = 0.0
                except (ValueError, TypeError):
                    fps = 0.0
            for attr in (
                "last_latency_ms",
                "latency_ms",
                "frame_latency_ms",
                "last_frame_time_ms",
                "frame_time_ms",
            ):
                raw_lat = getattr(capture_instance, attr, None)
                if raw_lat is None:
                    continue
                try:
                    lat = float(raw_lat)
                except (ValueError, TypeError):
                    continue
                if math.isfinite(lat) and lat >= 0.0:
                    latency_ms = round(lat, 1)
                    latency_measured = True
                    break

        if math.isfinite(fps) and fps > 0.0:
            lat_str = f"{latency_ms:.1f} ms" if latency_measured else "— ms"
            label_str = f"{fps:.0f} FPS · {lat_str}"
            pct = min(100, max(0, int(round(fps * 100.0 / 60.0))))
        else:
            # S-0902: fps<=0 iken gecikme olculmemistir (latency_measured=False),
            # o yuzden "0.0 ms" iddia edilmez; fps>0 dali gibi "— ms" yazilir.
            label_str = "0 FPS · — ms"
            pct = 0
            latency_ms = 0.0
            latency_measured = False

        return {
            "fps": fps,
            "latency_ms": latency_ms,
            "latency_measured": latency_measured,
            "pct": pct,
            "label_str": label_str,
        }
    except Exception as exc:
        logger.debug("Safely handled capture metrics error: %s", exc)
        return {
            "fps": 0.0,
            "latency_ms": 0.0,
            "latency_measured": False,
            "pct": 0,
            "label_str": "0 FPS · — ms",
        }


class SystemMonitor:
    """Safe, cached system monitor for GUI performance chips and system detail cards."""

    @classmethod
    def get_snapshot(cls, capture_instance: Optional[Any] = None) -> Dict[str, Any]:
        """
        Returns a cached hardware and capture performance snapshot:
        {
            "gpu": {"name": str, "pct": int, "temp": int, "temp_str": str, "mem_used_gb": float, "mem_total_gb": float, "label_str": str, "available": bool},
            "cpu": {"name": str, "cores": int, "pct": int, "freq_ghz": float, "label_str": str},
            "capture": {"fps": float, "latency_ms": float, "pct": int, "label_str": str}
        }
        """
        global _CACHED_SNAPSHOT, _LAST_SAMPLE_TS
        now = time.monotonic()
        with _MONITOR_LOCK:
            if _CACHED_SNAPSHOT is not None and (now - _LAST_SAMPLE_TS) < _CACHE_TTL_SEC:
                # If a specific capture instance is supplied, update capture metrics dynamically
                if capture_instance is not None:
                    _CACHED_SNAPSHOT["capture"] = _query_capture_metrics(capture_instance)
                # Return a copy: callers must never mutate the cache
                # (F-0396) or observe later refreshes through an old handle.
                return _copy_snapshot(_CACHED_SNAPSHOT)

            gpu_data = _query_gpu_metrics()
            cpu_data = _query_cpu_metrics()
            cap_data = _query_capture_metrics(capture_instance)

            _CACHED_SNAPSHOT = {
                "gpu": gpu_data,
                "cpu": cpu_data,
                "capture": cap_data,
                "timestamp": now,
            }
            _LAST_SAMPLE_TS = now
            return _copy_snapshot(_CACHED_SNAPSHOT)

    @classmethod
    def reset_cache(cls) -> None:
        """Forces the next get_snapshot() call to re-query hardware metrics."""
        global _CACHED_SNAPSHOT, _LAST_SAMPLE_TS
        with _MONITOR_LOCK:
            _CACHED_SNAPSHOT = None
            _LAST_SAMPLE_TS = 0.0
