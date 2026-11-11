# -*- coding: utf-8 -*-
"""
dataset_factory/calibrated_rois.py
===================================
Calibrated ROI manager for the Aion 2 overlay / skill-bar / HP-MP regions.

Moved here from dataset_factory/exemplar_memory.py (wrong home: ROI
calibration has nothing to do with exemplar memory). The old location
re-exports CalibratedRoisManager / get_calibrated_rois for compatibility.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("Gatherion.ExemplarMemory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent


DEFAULT_CALIBRATED_ROIS_PATH = PROJECT_ROOT / "config" / "calibrated_rois.json"


class CalibratedRoisManager:
    """
    Dynamically loads and hot-reloads calibrated ROIs from config/calibrated_rois.json.
    Uses mtime check to reload within <50ms (<0.1ms typical) whenever calibration changes.
    """

    def __init__(self, json_path: Optional[Union[str, Path]] = None) -> None:
        self.json_path = Path(json_path) if json_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        self._lock = threading.RLock()
        self._rois: Dict[str, Dict[str, Any]] = {}
        self._last_mtime: float = -1.0
        self._last_error_mtime: float = -1.0
        self.check_reload()

    def check_reload(self, force: bool = False) -> bool:
        """Inspects mtime and reloads if modified on disk."""
        with self._lock:
            mtime = -1.0
            try:
                if not self.json_path.is_file():
                    if self._last_mtime != -1.0:
                        self._rois.clear()
                        self._last_mtime = -1.0
                    return False
                mtime = self.json_path.stat().st_mtime
                if not force and mtime == self._last_mtime:
                    return True
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    data = {}
                rois_dict: Dict[str, Dict[str, Any]] = {}
                # Support both data["rois"] and top-level keys
                if "rois" in data and isinstance(data["rois"], dict):
                    rois_dict.update(data["rois"])
                for k, v in data.items():
                    if k not in ("rois", "updated_at", "version") and isinstance(v, dict):
                        rois_dict[k] = v
                self._rois = rois_dict
                self._last_mtime = mtime
                return True
            except Exception as exc:
                # A14: a malformed config used to fail silently (debug
                # only), leaving the manager empty / stale with no
                # visible trace. Warn once per mtime so a persistently
                # broken file does not spam the log on every lookup.
                if mtime != self._last_error_mtime:
                    logger.warning(
                        "CalibratedRoisManager: cannot load %s (%s); keeping previous ROIs",
                        self.json_path,
                        exc,
                    )
                    self._last_error_mtime = mtime
                return False

    ROI_KEY_ALIASES: Dict[str, Tuple[str, ...]] = {
        # A14: "skill_bar" no longer falls back to "skill_slot": the
        # config's skill_slot is ONE 3.75%-wide slot, and callers
        # asking for the full bar got that slot as the bar
        # (mana_phase_combat used it verbatim; skill_bar_reader
        # rejects it with its >= 0.10 width gate). Missing key now
        # returns None so callers use their full-bar fallback.
        "skill_bar": ("skill_bar",),
        "skill_slot": ("skill_slot", "skill_bar", "skill_slots"),
        "player_hp_bar": ("player_hp_bar", "player_hp", "hp_bar"),
        "player_mp_bar": ("player_mp_bar", "player_mp", "mp_bar"),
        # A14: "target_hp_bar" no longer falls back to "mob_nameplate":
        # the production config's mob_nameplate is the 3D world-space
        # nameplate (y ~0.39), while every target_hp_bar consumer
        # expects the top-center target frame (y < 0.25) — mana_phase_combat's
        # fallback, the gate's target ROI, and vision.py's
        # explicit y1 < 0.25 guard all agree. The reverse alias
        # (mob_nameplate -> target_hp_bar) is kept: test_w48 locks it.
        "target_hp_bar": ("target_hp_bar", "target_hp", "target_bar"),
        "mob_nameplate": ("mob_nameplate", "target_hp_bar", "target_hp"),
        "minimap": ("minimap", "minimap_roi", "minimap_inner"),
    }

    def get_roi_entry(self, key: str) -> Optional[Dict[str, Any]]:
        """S317 (v19/P3-8): derin kopya döner — eskiden canlı önbellek
        dict'i dönüyordu, çağıran değişimi önbelleği bozuyordu."""
        self.check_reload()
        raw_key = str(key or "").strip()
        with self._lock:
            if raw_key in self._rois:
                return copy.deepcopy(self._rois[raw_key])
            lower_key = raw_key.lower()
            if lower_key in self._rois:
                return copy.deepcopy(self._rois[lower_key])
            aliases = self.ROI_KEY_ALIASES.get(lower_key, ())
            for alias in aliases:
                if alias in self._rois:
                    return copy.deepcopy(self._rois[alias])
            return None

    def has_roi(self, key: str) -> bool:
        """Check if ROI is present in calibrated ROIs (with alias fallback)."""
        return self.get_roi_entry(key) is not None

    def get_all_rois(self) -> Dict[str, Dict[str, Any]]:
        """Return deep copy of all currently loaded ROIs (S317: eskiden
        yalnızca dış dict kopyalanıyordu, iç dict'ler canlıydı)."""
        self.check_reload()
        with self._lock:
            return copy.deepcopy(self._rois)

    def reload_now(self) -> bool:
        """Forces an immediate reload from disk."""
        return self.check_reload(force=True)

    @staticmethod
    def _validated_rect(x: Any, y: Any, w: Any, h: Any) -> Optional[Tuple[float, float, float, float]]:
        """Clamps a (x, y, w, h) rect to [0, 1] and rejects degenerate
        ones. A14: out-of-range / zero / negative rects used to be
        returned verbatim, and get_pixel_crop then masked the error
        by clamping to a bogus 2x2 crop instead of None."""
        try:
            x1, y1 = float(x), float(y)
            x2, y2 = x1 + float(w), y1 + float(h)
        except (TypeError, ValueError):
            return None
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        x1 = min(1.0, max(0.0, x1))
        y1 = min(1.0, max(0.0, y1))
        x2 = min(1.0, max(0.0, x2))
        y2 = min(1.0, max(0.0, y2))
        if x2 - x1 <= 0 or y2 - y1 <= 0:
            return None
        return (round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))

    def get_normalized_rect(self, key: str) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (x1, y1, x2, y2) normalized ratios [0.0..1.0] for the requested ROI key.

        Malformed entries (non-numeric / non-finite values, zero or negative
        width/height, rect fully outside [0, 1]) return None instead of
        raising or returning an inverted/degenerate rect: every caller uses
        ``... or <fallback>``, so None lets the caller's fallback apply
        rather than crashing a vision loop or cropping a 2px sliver.
        """
        entry = self.get_roi_entry(key)
        if not entry:
            return None
        # Format 1: normalized_roi: [x, y, w, h]
        # A None / non-sized value (e.g. {"normalized_roi": None}) is a
        # malformed entry: len(None) used to raise TypeError here instead
        # of returning None like every other malformed shape. Guard the
        # len() call and fall through to the format-2 check (which returns
        # None when no norm_x/norm_y fields exist).
        roi = entry.get("normalized_roi")
        if roi is not None:
            try:
                is_four = len(roi) == 4
            except TypeError:
                is_four = False
            if is_four:
                x, y, w, h = roi
                rect = self._validated_rect(x, y, w, h)
                if rect is not None:
                    return rect
                # S318 (v19/P3-8): geçersiz 4 elemanlı normalized_roi,
                # geçerli Format-2 alanlarını gölgelemesin — aşağı düş.
        # Format 2: norm_x, norm_y, norm_w, norm_h
        if "norm_x" in entry and "norm_y" in entry:
            return self._validated_rect(
                entry["norm_x"],
                entry["norm_y"],
                entry.get("norm_w", 0.0),
                entry.get("norm_h", 0.0),
            )
        return None

    def get_pixel_crop(self, frame: np.ndarray, key: str) -> Optional[np.ndarray]:
        if frame is None or frame.size == 0:
            return None
        rect = self.get_normalized_rect(key)
        if not rect:
            return None
        fh, fw = frame.shape[:2]
        x1 = max(0, min(fw - 2, int(round(rect[0] * fw))))
        y1 = max(0, min(fh - 2, int(round(rect[1] * fh))))
        x2 = max(x1 + 2, min(fw, int(round(rect[2] * fw))))
        y2 = max(y1 + 2, min(fh, int(round(rect[3] * fh))))
        crop = frame[y1:y2, x1:x2]
        return crop if crop.size > 0 else None


