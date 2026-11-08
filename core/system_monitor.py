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


def _query_gpu_metrics() -> Dict[str, Any]:
    """Queries NVIDIA GPU load, temperature, and VRAM via pynvml or returns clean idle fallback."""
    if not _init_nvml_if_needed():
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


def _query_cpu_metrics() -> Dict[str, Any]:
    """Queries CPU utilization, core count, and clock frequency via psutil."""
    try:
        import psutil
        cores = psutil.cpu_count(logical=True) or 4
        # interval=None provides instantaneous load without blocking the calling thread
        pct = int(psutil.cpu_percent(interval=None))
        freq_info = psutil.cpu_freq()
        if freq_info and getattr(freq_info, "current", None):
            freq_ghz = round(freq_info.current / 1000.0, 1)
        else:
            freq_ghz = 2.4
        label_str = f"{pct}% · {freq_ghz:.1f} GHz"
        return {
            "name": f"CPU · {cores} Çekirdek",
            "cores": cores,
            "pct": max(0, min(100, pct)),
            "freq_ghz": freq_ghz,
            "label_str": label_str,
        }
    except Exception as exc:
        logger.debug("psutil query error: %s", exc)
        return {
            "name": "CPU · 4 Çekirdek",
            "cores": 4,
            "pct": 0,
            "freq_ghz": 2.4,
            "label_str": "0% · 2.4 GHz",
        }


def _query_capture_metrics(capture_instance: Optional[Any] = None) -> Dict[str, Any]:
    """Extracts screen capture FPS and latency from the provided or cached capture instance."""
    fps = 0.0
    latency_ms = 0.0

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

        if math.isfinite(fps) and fps > 0.0:
            latency_ms = round(1000.0 / fps, 1)
            label_str = f"{fps:.0f} FPS · {latency_ms:.1f} ms"
            pct = min(100, max(0, int(round(fps * 100.0 / 60.0))))
        else:
            label_str = "0 FPS · 0.0 ms"
            pct = 0
            latency_ms = 0.0

        return {
            "fps": fps,
            "latency_ms": latency_ms,
            "pct": pct,
            "label_str": label_str,
        }
    except Exception as exc:
        logger.debug("Safely handled capture metrics error: %s", exc)
        return {
            "fps": 0.0,
            "latency_ms": 0.0,
            "pct": 0,
            "label_str": "0 FPS · 0.0 ms",
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
                return _CACHED_SNAPSHOT

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
            return _CACHED_SNAPSHOT

    @classmethod
    def reset_cache(cls) -> None:
        """Forces the next get_snapshot() call to re-query hardware metrics."""
        global _CACHED_SNAPSHOT, _LAST_SAMPLE_TS
        with _MONITOR_LOCK:
            _CACHED_SNAPSHOT = None
            _LAST_SAMPLE_TS = 0.0
