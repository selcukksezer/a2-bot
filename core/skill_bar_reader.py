# -*- coding: utf-8 -*-
"""
Dynamic Skill Bar Reader & Micro Camera Look-Around (core/skill_bar_reader.py)

Implements:
- Item 44: `SkillBarReader` — Inspects the bottom skill bar slots (`1, 2, 3, 4, 5, Q, E, R`
  at `x: 0.345..0.645, y: 0.935..0.985` in `pilgrims_ridge_1.jpg`), measuring each skill
  icon's brightness/saturation and blue mana-availability indicator so during combat recovery
  the bot only presses skills that are off cooldown (`ready=True`) and have enough mana
  (`select_best_ready_skill(frame) -> str`) instead of mashing `1-2-3-4-5` blindly.
- Item 45: `MicroCameraLookAround` — During long straight walks, every `15–25 seconds`
  (Gaussian interval), gently sways the camera by `±8°` left/right and back using smooth
  WindMouse micro-steps (`perform_micro_camera_sway(input_ctrl)`) so the character never
  looks like a rigid straight-line robot.
"""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from core.windmouse_humanizer import WindMouseEngine

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CALIBRATED_ROIS_PATH = PROJECT_ROOT / "config" / "calibrated_rois.json"

try:
    from dataset_factory.exemplar_memory import compute_similarity, extract_crop_features
except Exception:
    extract_crop_features = None  # type: ignore
    compute_similarity = None  # type: ignore



class _SkillBarRoiDescriptor:
    """
    Descriptor ensuring both SkillBarReader.SKILL_BAR_ROI and reader.SKILL_BAR_ROI
    return the dynamically reloaded tuple from config/calibrated_rois.json.
    """

    def __init__(self, fallback_roi: Tuple[float, float, float, float]) -> None:
        self.fallback_roi = fallback_roi

    def __get__(
        self, instance: Optional[Any], owner: Optional[type] = None
    ) -> Tuple[float, float, float, float]:
        if instance is not None and hasattr(instance, "current_skill_bar_roi"):
            return instance.current_skill_bar_roi
        # Accessed on class itself: SkillBarReader.SKILL_BAR_ROI
        try:
            return SkillBarReader._resolve_skill_bar_roi_static(fallback=self.fallback_roi)
        except Exception:
            return self.fallback_roi

    def __set__(self, instance: Any, value: Sequence[float]) -> None:
        if instance is not None and hasattr(instance, "set_custom_roi"):
            instance.set_custom_roi(value)


class _ConsumableBarRoiDescriptor:
    """
    Descriptor ensuring both SmartConsumableManager.CONSUMABLE_ROW_ROI and
    manager.CONSUMABLE_ROW_ROI return the dynamically reloaded tuple from config/calibrated_rois.json.
    """

    def __init__(self, fallback_roi: Tuple[float, float, float, float]) -> None:
        self.fallback_roi = fallback_roi

    def __get__(
        self, instance: Optional[Any], owner: Optional[type] = None
    ) -> Tuple[float, float, float, float]:
        if instance is not None and hasattr(instance, "current_consumable_roi"):
            return instance.current_consumable_roi
        try:
            return SmartConsumableManager._resolve_consumable_roi_static(fallback=self.fallback_roi)
        except Exception:
            return self.fallback_roi

    def __set__(self, instance: Any, value: Sequence[float]) -> None:
        if instance is not None and hasattr(instance, "set_custom_roi"):
            instance.set_custom_roi(value)


class SkillBarReader:
    """
    Item 44 — Dinamik Yetenek (Skill) Barı Okuyucu (`SkillBarReader`).

    Inspects the bottom skill bar region (`x: 0.345..0.645, y: 0.935..0.985` in
    `pilgrims_ridge_1.jpg`) across the 8 primary combat slots (`1, 2, 3, 4, 5, Q, E, R`).
    Evaluates:
      1. Brightness (`V` channel mean in HSV) & local contrast (`std`) to detect whether
         a skill slot is off cooldown vs darkened by a cooldown sweep.
      2. Color Saturation (`S` channel mean in HSV) to distinguish active icons from
         greyed-out/disabled slots.
      3. Blue Mana-Availability indicator (`B - R` channel dominance and cyan/blue HSV mask
         in the slot / badge corner) to verify sufficient mana.
    Provides `inspect_skill_bar(frame)` and `select_best_ready_skill(frame) -> str`.
    """

    # AION2 bottom hotbar (~slots 1-5,Q,E,R). Slightly higher than potion row.
    DEFAULT_SKILL_BAR_ROI: Tuple[float, float, float, float] = (0.355, 0.905, 0.655, 0.955)
    SKILL_BAR_ROI: Any = _SkillBarRoiDescriptor(DEFAULT_SKILL_BAR_ROI)
    PLAYER_HP_BAR_ROI: Tuple[float, float, float, float] = (0.345, 0.880, 0.485, 0.915)
    PLAYER_MP_BAR_ROI: Tuple[float, float, float, float] = (0.515, 0.880, 0.655, 0.915)
    TARGET_HP_BAR_ROI: Tuple[float, float, float, float] = (0.350, 0.030, 0.650, 0.075)
    SKILL_SLOTS: Tuple[str, ...] = ("1", "2", "3", "4", "5", "Q", "E", "R")

    @classmethod
    def _resolve_skill_bar_roi_static(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        fallback: Tuple[float, float, float, float] = (0.355, 0.905, 0.655, 0.955),
    ) -> Tuple[float, float, float, float]:
        path = Path(json_path) if json_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        if not path.is_file():
            return fallback
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rois_dict: Dict[str, Any] = {}
            if isinstance(data, dict):
                if "rois" in data and isinstance(data["rois"], dict):
                    rois_dict.update(data["rois"])
                for k, v in data.items():
                    if k not in ("rois", "updated_at", "version") and isinstance(v, dict):
                        rois_dict[k] = v

            entry = rois_dict.get("skill_bar")
            if entry and isinstance(entry, dict):
                rect: Optional[Tuple[float, float, float, float]] = None
                if "normalized_roi" in entry and len(entry["normalized_roi"]) == 4:
                    n = [float(v) for v in entry["normalized_roi"]]
                    if n[1] + n[3] > 1.05 and n[3] > n[1]:
                        rect = (round(n[0], 6), round(n[1], 6), round(n[2], 6), round(n[3], 6))
                    else:
                        rect = (
                            round(n[0], 6),
                            round(n[1], 6),
                            round(min(1.0, n[0] + n[2]), 6),
                            round(min(1.0, n[1] + n[3]), 6),
                        )
                elif "norm_x" in entry and "norm_y" in entry:
                    x = float(entry["norm_x"])
                    y = float(entry["norm_y"])
                    w = float(entry.get("norm_w", 0.0))
                    h = float(entry.get("norm_h", 0.0))
                    rect = (round(x, 6), round(y, 6), round(min(1.0, x + w), 6), round(min(1.0, y + h), 6))
                if rect and (rect[2] - rect[0]) >= 0.10:
                    return rect

            try:
                from dataset_factory.exemplar_memory import get_calibrated_rois
                rect = get_calibrated_rois(path).get_normalized_rect("skill_bar")
                if rect and len(rect) == 4 and (rect[2] - rect[0]) >= 0.10:
                    return rect
            except Exception:
                pass
        except Exception:
            pass
        return fallback

    def get_skill_bar_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        """
        Dynamically reloads and returns SKILL_BAR_ROI from config/calibrated_rois.json.
        Performs an mtime check (<50ms, typically <0.05ms) so changes saved by
        Calibration Wizard or manual edits take effect immediately without restart.
        """
        if self._custom_roi is not None:
            return self._custom_roi

        json_path = self.calibrated_rois_path
        if not json_path or not json_path.is_file():
            return self.DEFAULT_SKILL_BAR_ROI

        try:
            mtime = json_path.stat().st_mtime
            if not force and mtime == self._last_roi_mtime and self._cached_roi is not None:
                return self._cached_roi

            with self._lock:
                rect = self._resolve_skill_bar_roi_static(
                    json_path=json_path,
                    fallback=self.DEFAULT_SKILL_BAR_ROI,
                )
                self._cached_roi = rect
                self._last_roi_mtime = mtime
                return rect
        except Exception as exc:
            logger.debug("SkillBarReader get_skill_bar_roi error: %s", exc)
            return self._cached_roi or self.DEFAULT_SKILL_BAR_ROI

    def check_reload_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        """Checks mtime of config/calibrated_rois.json and reloads SKILL_BAR_ROI if modified (<50ms)."""
        return self.get_skill_bar_roi(force=force)

    def reload_calibrated_rois(self, force: bool = True) -> Tuple[float, float, float, float]:
        """Force-reload calibrated skill bar ROI from disk."""
        return self.get_skill_bar_roi(force=force)

    def set_custom_roi(self, roi: Sequence[float]) -> None:
        """Manually override skill bar ROI with custom normalized coordinates (x1, y1, x2, y2)."""
        if len(roi) == 4:
            self._custom_roi = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))

    @property
    def current_skill_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads SKILL_BAR_ROI from config/calibrated_rois.json on file modification (<50ms)."""
        return self.get_skill_bar_roi()

    @property
    def current_player_hp_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads player_hp_bar ROI from config/calibrated_rois.json (<50ms)."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_hp_bar")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_hp")
            if rect and len(rect) == 4:
                return rect
        except Exception:
            pass
        return self.PLAYER_HP_BAR_ROI

    @property
    def current_player_mp_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads player_mp_bar ROI from config/calibrated_rois.json (<50ms)."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_mp_bar")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("player_mp")
            if rect and len(rect) == 4:
                return rect
        except Exception:
            pass
        return self.PLAYER_MP_BAR_ROI

    @property
    def current_target_hp_bar_roi(self) -> Tuple[float, float, float, float]:
        """Dynamically reloads target_hp_bar ROI from config/calibrated_rois.json (<50ms)."""
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("target_hp_bar")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("target_hp")
            if not rect:
                rect = get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect("mob_nameplate")
            if rect and len(rect) == 4:
                return rect
        except Exception:
            pass
        return self.TARGET_HP_BAR_ROI

    # Convenience properties
    @property
    def player_hp_bar(self) -> Tuple[float, float, float, float]:
        return self.current_player_hp_bar_roi

    @property
    def player_mp_bar(self) -> Tuple[float, float, float, float]:
        return self.current_player_mp_bar_roi

    @property
    def target_hp_bar(self) -> Tuple[float, float, float, float]:
        return self.current_target_hp_bar_roi

    @property
    def skill_bar_roi(self) -> Tuple[float, float, float, float]:
        return self.current_skill_bar_roi

    def get_calibrated_roi(self, name: str) -> Optional[Tuple[float, float, float, float]]:
        """Get calibrated normalized rect for named ROI."""
        name_lower = str(name).lower()
        if "skill" in name_lower:
            return self.current_skill_bar_roi
        if "player_hp" in name_lower or name_lower == "hp":
            return self.current_player_hp_bar_roi
        if "player_mp" in name_lower or name_lower == "mp":
            return self.current_player_mp_bar_roi
        if "target" in name_lower or "mob" in name_lower:
            return self.current_target_hp_bar_roi
        try:
            from dataset_factory.exemplar_memory import get_calibrated_rois
            return get_calibrated_rois(self.calibrated_rois_path).get_normalized_rect(name)
        except Exception:
            return None


    # Priority weights for combat rotation when multiple skills are ready & have mana
    # Equal baseline weights to eliminate artificial bias towards slot '2'.
    DEFAULT_PRIORITY_WEIGHTS: Dict[str, float] = {
        "1": 1.00,
        "2": 1.00,
        "3": 1.00,
        "4": 1.00,
        "5": 1.00,
        "Q": 1.00,
        "E": 1.00,
        "R": 1.00,
    }

    def __init__(
        self,
        min_brightness_ready: float = 50.0,
        min_saturation_ready: float = 45.0,
        min_contrast_std: float = 22.0,
        min_mana_blue_ratio: float = 0.03,
        cooldown_lockout_sec: float = 0.15,
        calibrated_rois_path: Optional[Union[str, Path]] = None,
        exemplar_memory: Optional[Any] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.min_brightness_ready: float = float(min_brightness_ready)
        self.min_saturation_ready: float = float(min_saturation_ready)
        self.min_contrast_std: float = float(min_contrast_std)
        self.min_mana_blue_ratio: float = float(min_mana_blue_ratio)
        self.cooldown_lockout_sec: float = max(0.0, float(cooldown_lockout_sec))

        self.calibrated_rois_path: Path = (
            Path(calibrated_rois_path) if calibrated_rois_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        )
        self._last_roi_mtime: float = -1.0
        self._cached_roi: Optional[Tuple[float, float, float, float]] = None
        self._custom_roi: Optional[Tuple[float, float, float, float]] = None
        self.exemplar_memory: Optional[Any] = exemplar_memory
        self._consumable_manager: Optional[SmartConsumableManager] = None

        self._last_pressed_ts: Dict[str, float] = {}
        self._last_slot_states: Dict[str, Dict[str, Any]] = {}

    def get_exemplar_memory(self) -> Optional[Any]:
        """Resolves active ExemplarMemory instance (injected or global singleton)."""
        if self.exemplar_memory is not None:
            return self.exemplar_memory
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            return get_exemplar_memory()
        except Exception as exc:
            logger.debug("Could not resolve ExemplarMemory: %s", exc)
            return None

    def reset_history(self) -> None:
        """Clears recent skill press lockout timestamps."""
        with self._lock:
            self._last_pressed_ts.clear()
            self._last_slot_states.clear()

    def mark_slot_pressed(self, slot_key: str) -> None:
        """Record a UI/key cast so inspect_skill_bar applies cooldown lockout."""
        key = str(slot_key or "").strip()
        if not key:
            return
        # Normalize to reader slot casing (1..5, Q, E, R).
        upper = key.upper()
        if upper in ("Q", "E", "R"):
            key = upper
        else:
            key = key.lower()
        with self._lock:
            self._call_counter = getattr(self, "_call_counter", 0) + 1
            now_ts = time.monotonic() + (self._call_counter * 1e-6)
            self._last_pressed_ts[key] = now_ts
            # Also store letter variants used by combat configs.
            self._last_pressed_ts[key.lower()] = now_ts
            self._last_pressed_ts[key.upper()] = now_ts

    def slot_center_px(
        self,
        frame: Optional[np.ndarray],
        slot_key: str,
        jitter_px: int = 3,
    ) -> Optional[Tuple[int, int]]:
        """
        Pixel center of a skill-bar slot inside SKILL_BAR_ROI.
        ROI is split into len(SKILL_SLOTS) equal columns (1,2,3,4,5,Q,E,R).
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        key = str(slot_key or "").strip()
        if not key:
            return None
        upper = key.upper()
        if upper in ("Q", "E", "R"):
            lookup = upper
        else:
            lookup = key.lower()
        slots = list(self.SKILL_SLOTS)
        # Match case-insensitively against configured slots.
        idx = None
        for i, name in enumerate(slots):
            if str(name).lower() == lookup.lower():
                idx = i
                break
        if idx is None:
            return None
        try:
            fh, fw = frame.shape[:2]
            roi = self.current_skill_bar_roi
            x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            num_slots = len(slots)
            slot_w = float(bw) / float(num_slots)
            sx1 = x1 + int(round(idx * slot_w))
            sx2 = x1 + int(round((idx + 1) * slot_w))
            cx = (sx1 + sx2) // 2
            cy = y1 + bh // 2
            j = max(0, int(jitter_px))
            if j > 0:
                cx += random.randint(-j, j)
                cy += random.randint(-j, j)
            cx = max(0, min(fw - 1, cx))
            cy = max(0, min(fh - 1, cy))
            return int(cx), int(cy)
        except Exception as exc:
            logger.debug("slot_center_px failed (%s): %s", slot_key, exc)
            return None

    def inspect_skill_bar(self, frame: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """
        Inspects all 8 skill slots (`1, 2, 3, 4, 5, Q, E, R`) in the bottom HUD skill bar
        (`x: 0.345..0.645, y: 0.935..0.985`) and returns a per-slot diagnostics dictionary:
        `{slot_key: {"slot": str, "ready": bool, "mana_ok": bool, "off_cooldown": bool,
                     "brightness": float, "saturation": float, "blue_ratio": float,
                     "contrast_std": float, "score": float}}`.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {}

        try:
            fh, fw = frame.shape[:2]
            roi = self.current_skill_bar_roi
            x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))

            bar_crop = frame[y1:y2, x1:x2]
            if bar_crop.size == 0:
                return {}

            if bar_crop.ndim == 2:
                bar_bgr = cv2.cvtColor(bar_crop, cv2.COLOR_GRAY2BGR)
            else:
                bar_bgr = bar_crop[:, :, :3]

            bh, bw = bar_bgr.shape[:2]
            num_slots = len(self.SKILL_SLOTS)
            slot_w = float(bw) / float(num_slots)

            now = time.monotonic()
            results: Dict[str, Dict[str, Any]] = {}

            for idx, slot_key in enumerate(self.SKILL_SLOTS):
                sx1 = max(0, int(round(idx * slot_w)))
                sx2 = max(sx1 + 2, min(bw, int(round((idx + 1) * slot_w))))
                slot_img = bar_bgr[:, sx1:sx2]

                # Trim outer 12% border so slot frame lines don't skew icon interior metrics
                sh, sw_local = slot_img.shape[:2]
                pad_y = max(1, int(round(sh * 0.12)))
                pad_x = max(1, int(round(sw_local * 0.12)))
                inner = slot_img[pad_y : max(pad_y + 2, sh - pad_y), pad_x : max(pad_x + 2, sw_local - pad_x)]
                if inner.size == 0:
                    inner = slot_img

                hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
                brightness = float(np.mean(hsv[:, :, 2]))
                saturation = float(np.mean(hsv[:, :, 1]))
                contrast_std = float(np.std(inner))

                # Measure blue mana-availability indicator across the full slot (including mana cost badge)
                b_chan = slot_img[:, :, 0].astype(np.int16)
                g_chan = slot_img[:, :, 1].astype(np.int16)
                r_chan = slot_img[:, :, 2].astype(np.int16)
                blue_dominant_mask = (b_chan - r_chan > 14) & (b_chan > 55) & (g_chan >= r_chan - 10)
                blue_ratio = float(np.mean(blue_dominant_mask))

                # Red "no mana / unavailable" overlay check
                red_dominant_mask = (r_chan - b_chan > 45) & (r_chan - g_chan > 35) & (r_chan > 110)
                red_penalty_ratio = float(np.mean(red_dominant_mask))

                baseline_off_cooldown = bool(
                    brightness >= self.min_brightness_ready
                    and saturation >= self.min_saturation_ready
                    and contrast_std >= self.min_contrast_std
                )

                # A skill has enough mana unless covered by the red "out of mana" overlay
                # or completely darkened/depleted.
                mana_ok = bool(
                    (blue_ratio >= self.min_mana_blue_ratio or red_penalty_ratio < 0.25)
                    and red_penalty_ratio < 0.38
                )

                with self._lock:
                    last_used = self._last_pressed_ts.get(slot_key, 0.0)
                    recently_pressed = (now - last_used) < self.cooldown_lockout_sec if last_used > 0.0 else False
                    lockout_remaining_sec = max(0.0, self.cooldown_lockout_sec - (now - last_used)) if (last_used > 0.0 and (now - last_used) < self.cooldown_lockout_sec) else 0.0

                off_cooldown = baseline_off_cooldown
                ready = bool(off_cooldown and mana_ok and not recently_pressed)

                # Consult ExemplarMemory for active/ready (class 20) vs cooldown/empty (class 21)
                exemplar_reinforced = False
                exemplar_state: str = "none"
                ready_sim: float = 0.0
                cooldown_sim: float = 0.0
                exemplar_sim: float = 0.0
                exemplar_confidence: float = 0.0

                mem = self.get_exemplar_memory()
                if mem is not None and inner.size > 0:
                    try:
                        # 1. Negative exemplar check (known false/empty/corrupted slots)
                        is_neg, neg_sim, _ = mem.is_negative_patch(inner, threshold=0.72)
                        if not is_neg and slot_img.size > 0:
                            is_neg, neg_sim, _ = mem.is_negative_patch(slot_img, threshold=0.75)

                        # 2. Specific class queries: 20=skill_slot_ready, 21=skill_slot_cooldown
                        if hasattr(mem, "match_class"):
                            r_sim, _ = mem.match_class(inner, 20)
                            c_sim, _ = mem.match_class(inner, 21)
                            ready_sim = max(ready_sim, float(r_sim))
                            cooldown_sim = max(cooldown_sim, float(c_sim))

                        # 3. Classify patch across all learned positive exemplars
                        pos_cid, pos_cname, p_sim = mem.classify_patch(inner, min_sim=0.65)
                        if pos_cid is None and slot_img.size > 0:
                            pos_cid, pos_cname, p_sim = mem.classify_patch(slot_img, min_sim=0.68)

                        if pos_cid is not None:
                            cname_lower = str(pos_cname or "").lower()
                            p_float = float(p_sim)
                            if pos_cid == 20 or "ready" in cname_lower:
                                ready_sim = max(ready_sim, p_float)
                            elif pos_cid == 21 or "cooldown" in cname_lower or "empty" in cname_lower:
                                cooldown_sim = max(cooldown_sim, p_float)
                            elif pos_cid == 13 or "skill" in cname_lower:
                                ready_sim = max(ready_sim, p_float * 0.90)

                        # Adjudicate exemplar verdict
                        if is_neg:
                            off_cooldown = False
                            ready = False
                            exemplar_state = "negative"
                            exemplar_sim = neg_sim
                            exemplar_confidence = neg_sim
                        elif cooldown_sim >= 0.70 and cooldown_sim > (ready_sim + 0.05):
                            off_cooldown = False
                            ready = False
                            exemplar_state = "cooldown"
                            exemplar_sim = cooldown_sim
                            exemplar_confidence = cooldown_sim
                        elif ready_sim >= 0.68 and ready_sim > (cooldown_sim + 0.05):
                            off_cooldown = True
                            ready = bool(mana_ok and not recently_pressed)
                            exemplar_reinforced = True
                            exemplar_state = "ready"
                            exemplar_sim = ready_sim
                            exemplar_confidence = ready_sim
                        else:
                            exemplar_state = "neutral"
                            exemplar_sim = max(ready_sim, cooldown_sim)
                            exemplar_confidence = exemplar_sim

                    except Exception as exc:
                        logger.debug("Exemplar memory slot evaluation error: %s", exc)

                # Unbiased composite readiness score:
                # Primary readiness indicator (off-cooldown, mana available, not locked out)
                # without brightness dominating or biasing towards slot '2'.
                base_readiness = 1.0 if ready else (0.45 if (off_cooldown and mana_ok) else 0.10)
                norm_v = min(1.0, brightness / 100.0)
                norm_s = min(1.0, saturation / 140.0)
                norm_contrast = min(1.0, contrast_std / 50.0)
                priority_mult = self.DEFAULT_PRIORITY_WEIGHTS.get(slot_key, 1.0)
                visual_quality = 0.50 * norm_s + 0.30 * norm_contrast + 0.20 * norm_v

                if exemplar_state == "ready" and ready:
                    score = round(float(np.clip(0.65 * ready_sim + 0.35 * visual_quality, 0.75, 0.99)) * priority_mult, 4)
                elif exemplar_state in ("cooldown", "negative") or not off_cooldown:
                    score = round(float(max(0.01, min(0.15, 0.15 * visual_quality))), 4)
                else:
                    score = round(float(base_readiness * visual_quality * priority_mult), 4)
                    if cooldown_sim > 0.40:
                        score = round(float(max(0.05, score * (1.0 - 0.5 * cooldown_sim))), 4)

                results[slot_key] = {
                    "slot": slot_key,
                    "ready": ready,
                    "off_cooldown": off_cooldown,
                    "mana_ok": mana_ok,
                    "recently_pressed": recently_pressed,
                    "lockout_remaining_sec": round(lockout_remaining_sec, 3),
                    "brightness": round(brightness, 2),
                    "saturation": round(saturation, 2),
                    "blue_ratio": round(blue_ratio, 4),
                    "contrast_std": round(contrast_std, 2),
                    "score": score,
                    "exemplar_state": exemplar_state,
                    "exemplar_sim": round(exemplar_sim, 3),
                    "exemplar_ready_sim": round(ready_sim, 3),
                    "exemplar_cooldown_sim": round(cooldown_sim, 3),
                    "exemplar_confidence": round(exemplar_confidence, 3),
                    "exemplar_reinforced": exemplar_reinforced,
                }

            with self._lock:
                self._last_slot_states = dict(results)

            return results

        except Exception as exc:
            logger.error("SkillBarReader.inspect_skill_bar failed: %s", exc)
            return {}

    def select_best_ready_skill(
        self,
        frame: Optional[np.ndarray],
        preferred_order: Optional[Sequence[str]] = None,
        record_use: bool = True,
    ) -> str:
        """
        Evaluates all 8 skill slots (`1, 2, 3, 4, 5, Q, E, R`) on `frame` and returns
        the key name (`"1"`, `"2"`, `"3"`, `"4"`, `"5"`, `"Q"`, `"E"`, or `"R"`) of the
        highest-scoring skill that is BOTH off cooldown (`ready=True`) and has sufficient
        mana (`mana_ok=True`).

        If all skills are currently on cooldown or recently pressed within the lockout
        window, falls back to the brightest off-cooldown slot (or `"1"` as safe basic skill).
        """
        slot_states = self.inspect_skill_bar(frame)
        if not slot_states:
            return "1"

        ready_candidates: List[Dict[str, Any]] = [
            info for info in slot_states.values() if info.get("ready", False)
        ]

        if not ready_candidates:
            # Relax recent-press lockout if every ready skill was just cycled
            ready_candidates = [
                info
                for info in slot_states.values()
                if info.get("off_cooldown", False) and info.get("mana_ok", False)
            ]

        if ready_candidates:
            slot_indices = {s: i for i, s in enumerate(self.SKILL_SLOTS)}
            if preferred_order:
                order_map = {str(k).upper(): i for i, k in enumerate(preferred_order)}
                ready_candidates.sort(
                    key=lambda item: (
                        self._last_pressed_ts.get(item["slot"], 0.0),
                        order_map.get(item["slot"], 999),
                        slot_indices.get(item["slot"], 999),
                    )
                )
            else:
                # Unbiased round-robin rotation across all ready skills:
                # Order primarily by least recently pressed timestamp (oldest first)
                # so every ready skill gets used, breaking ties by natural slot order (1, 2, 3, 4, 5, Q, E, R).
                ready_candidates.sort(
                    key=lambda item: (
                        self._last_pressed_ts.get(item["slot"], 0.0),
                        slot_indices.get(item["slot"], 999),
                    )
                )
            chosen_key = str(ready_candidates[0]["slot"])
        else:
            # Fallback without brightness bias: cycle across available slots or use basic skill '1'
            fallback_candidates = [
                info for info in slot_states.values() if info.get("off_cooldown", False)
            ] or list(slot_states.values())
            if fallback_candidates:
                slot_indices = {s: i for i, s in enumerate(self.SKILL_SLOTS)}
                fallback_candidates.sort(
                    key=lambda item: (
                        self._last_pressed_ts.get(item["slot"], 0.0),
                        slot_indices.get(item["slot"], 999),
                    )
                )
                chosen_key = str(fallback_candidates[0].get("slot", "1"))
            else:
                chosen_key = "1"

        if record_use:
            with self._lock:
                self._call_counter = getattr(self, "_call_counter", 0) + 1
                now_ts = time.monotonic() + (self._call_counter * 1e-6)
                self._last_pressed_ts[chosen_key] = now_ts
                self._last_pressed_ts[chosen_key.lower()] = now_ts
                self._last_pressed_ts[chosen_key.upper()] = now_ts

        return chosen_key

    def get_ready_skills(self, frame: Optional[np.ndarray]) -> List[str]:
        """Returns a list of slot keys that are currently ready and off cooldown."""
        slot_states = self.inspect_skill_bar(frame)
        if not slot_states:
            return list(self.SKILL_SLOTS)
        ready = [
            info["slot"]
            for info in slot_states.values()
            if info.get("ready", False) or (info.get("off_cooldown", False) and info.get("mana_ok", False))
        ]
        return ready if ready else list(self.SKILL_SLOTS)

    def read_vitals_from_frame(self, frame: Optional[np.ndarray]) -> Dict[str, float]:
        """Convenience method reading player HP/MP and target HP using calibrated ROIs (<50ms)."""
        try:
            from core.mana_phase_combat import ManaPhaseCombat
            return ManaPhaseCombat().read_vitals_from_frame(frame)
        except Exception:
            return {"player_hp": 1.0, "player_mp": 1.0, "target_hp": 0.0, "has_target": 0.0}

    def get_skill_bar_reader_version_info(self) -> dict:
        return {"module": "skill_bar_reader", "version": "2.1.0", "ready": True}

    def get_skill_bar_reader_diagnostic_snapshot(self) -> dict:
        mem = self.get_exemplar_memory()
        mem_loaded = mem is not None
        pos_cnt = len(getattr(mem, "_positive", [])) if mem else 0
        neg_cnt = len(getattr(mem, "_negative", [])) if mem else 0
        return {
            "module": "skill_bar_reader",
            "status": "ok",
            "ready": True,
            "current_roi": self.current_skill_bar_roi,
            "calibrated_rois_path": str(self.calibrated_rois_path),
            "last_roi_mtime": self._last_roi_mtime,
            "exemplar_memory_active": mem_loaded,
            "exemplar_positive_count": pos_cnt,
            "exemplar_negative_count": neg_cnt,
            "recent_slots_count": len(self._last_slot_states),
        }

    @property
    def consumable_manager(self) -> SmartConsumableManager:
        """Dynamically provides attached SmartConsumableManager."""
        if self._consumable_manager is None:
            self._consumable_manager = SmartConsumableManager(
                calibrated_rois_path=self.calibrated_rois_path,
                exemplar_memory=self.exemplar_memory,
            )
        return self._consumable_manager

    def inspect_consumables(self, frame: Optional[np.ndarray]) -> Dict[str, Dict[str, Any]]:
        """Inspects upper consumable slots (F1..F8, X, B) via SmartConsumableManager."""
        return self.consumable_manager.inspect_consumables(frame)

    def get_consumable_manager(self) -> SmartConsumableManager:
        """Returns active SmartConsumableManager instance."""
        return self.consumable_manager


class MicroCameraLookAround:
    """
    Item 45 — İnsani Kamera Salınımı (`MicroCameraLookAround`).

    During long straight walks, every `15–25 seconds` (sampled from a Gaussian
    distribution centered at `20.0s`, clamped to `[15.0, 25.0]`), gently sways the
    camera by `±8°` left/right and smoothly returns to the original heading using
    `WindMouseEngine` micro-steps (`perform_micro_camera_sway(input_ctrl)`) so the
    character never looks like a rigid straight-line robot.
    """

    MIN_INTERVAL_SEC: float = 15.0
    MAX_INTERVAL_SEC: float = 25.0
    DEFAULT_SWAY_DEG: float = 8.0
    PIXELS_PER_DEGREE: float = 11.5

    def __init__(
        self,
        min_interval_sec: float = MIN_INTERVAL_SEC,
        max_interval_sec: float = MAX_INTERVAL_SEC,
        sway_degrees: float = DEFAULT_SWAY_DEG,
        pixels_per_degree: float = PIXELS_PER_DEGREE,
        windmouse: Optional[WindMouseEngine] = None,
        seed: Optional[int] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.min_interval_sec: float = max(5.0, float(min_interval_sec))
        self.max_interval_sec: float = max(self.min_interval_sec + 1.0, float(max_interval_sec))
        self.sway_degrees: float = max(1.0, min(25.0, float(sway_degrees)))
        self.pixels_per_degree: float = max(2.0, float(pixels_per_degree))
        self._rng: random.Random = random.Random(seed)
        self.windmouse: WindMouseEngine = windmouse or WindMouseEngine(auto_reveal_cursor=False)

        self._last_sway_ts: float = time.monotonic()
        self._next_interval_sec: float = self.sample_next_interval()
        self.sway_count: int = 0
        self.last_sway_summary: Dict[str, Any] = {}

    def sample_next_interval(self) -> float:
        """
        Samples the next look-around interval in `[15.0, 25.0]` seconds using a
        Gaussian distribution (`mean=20.0s, sigma=2.5s`).
        """
        with self._lock:
            mean_sec = 0.5 * (self.min_interval_sec + self.max_interval_sec)
            sigma_sec = (self.max_interval_sec - self.min_interval_sec) / 4.0
            sampled = self._rng.gauss(mean_sec, sigma_sec)
            clamped = float(np.clip(sampled, self.min_interval_sec, self.max_interval_sec))
            self._next_interval_sec = round(clamped, 3)
            return self._next_interval_sec

    def should_trigger_sway(self, now: Optional[float] = None) -> bool:
        """Returns True if at least `_next_interval_sec` (`15–25s`) has elapsed since the last sway."""
        current = float(now) if now is not None else time.monotonic()
        with self._lock:
            return (current - self._last_sway_ts) >= self._next_interval_sec

    @staticmethod
    def _resolve_cursor_pos(input_ctrl: Any) -> Tuple[int, int]:
        if input_ctrl is not None:
            get_pos = getattr(input_ctrl, "get_cursor_position", None)
            if callable(get_pos):
                try:
                    pos = get_pos()
                    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                        return int(pos[0]), int(pos[1])
                except Exception as exc:
                    logger.debug("Could not query cursor position from input_ctrl: %s", exc)
        return 960, 540

    @staticmethod
    def _resolve_move_fn(input_ctrl: Any) -> Optional[Callable[[int, int], Any]]:
        if input_ctrl is None:
            return None
        for attr in ("_hw_move_to", "_sendinput_mouse_move", "move_to", "move_mouse_abs", "set_cursor_pos"):
            fn = getattr(input_ctrl, attr, None)
            if callable(fn):
                return fn
        return None

    def perform_micro_camera_sway(
        self,
        input_ctrl: Any,
        direction: Optional[int] = None,
        angle_deg: Optional[float] = None,
        force: bool = True,
    ) -> Dict[str, Any]:
        """
        Gently sways the camera by `±8°` (`angle_deg`) left or right and smoothly returns
        to the original forward heading using WindMouse micro-steps so the character
        looks around naturally during long straight walks without altering its net path.

        Parameters
        ----------
        input_ctrl : Any
            The active `InputController` or mock input driver.
        direction : int, optional
            `-1` for left, `+1` for right. Randomly chosen if `None`.
        angle_deg : float, optional
            Target sway angle in degrees (defaults to `~8.0°` with slight Gaussian variance).
        force : bool
            If True (default when called directly), executes immediately and resets the
            15–25s timer. If False, only executes if `should_trigger_sway()` is True.
        """
        now = time.monotonic()
        if not force and not self.should_trigger_sway(now=now):
            return {
                "triggered": False,
                "reason": "interval_not_elapsed",
                "next_interval_sec": self._next_interval_sec,
            }

        with self._lock:
            chosen_dir = int(direction) if direction in (-1, 1) else self._rng.choice((-1, 1))
            base_deg = float(angle_deg) if angle_deg is not None else self.sway_degrees
            # Natural micro-variance around ±8° (e.g., 7.2° .. 8.8°)
            effective_deg = float(np.clip(self._rng.gauss(base_deg, 0.45), 5.5, 10.5))
            delta_px = int(round(chosen_dir * effective_deg * self.pixels_per_degree))
            if delta_px == 0:
                delta_px = chosen_dir * int(round(self.sway_degrees * self.pixels_per_degree))

            start_x, start_y = self._resolve_cursor_pos(input_ctrl)
            # Subtle vertical pitch micro-drift (±3..8 px) for organic human hand feel
            pitch_jitter_px = int(round(self._rng.gauss(0.0, 3.0)))
            peak_x = start_x + delta_px
            peak_y = start_y + pitch_jitter_px

            # Generate smooth WindMouse outward trajectory (center -> ±8°) and return trajectory (±8° -> center)
            outward_path = self.windmouse.generate_windmouse_path(
                start_x=start_x,
                start_y=start_y,
                dest_x=peak_x,
                dest_y=peak_y,
                gravity=9.5,
                wind=2.4,
                max_step=14.0,
            )
            return_path = self.windmouse.generate_windmouse_path(
                start_x=peak_x,
                start_y=peak_y,
                dest_x=start_x,
                dest_y=start_y,
                gravity=10.0,
                wind=2.0,
                max_step=14.0,
            )

            move_fn = self._resolve_move_fn(input_ctrl)
            is_test_mode = bool(getattr(input_ctrl, "test_mode", False)) if input_ctrl is not None else True

            executed_steps = 0
            for step in outward_path:
                px, py = int(step[0]), int(step[1])
                dt = float(step[2]) if len(step) >= 3 else 0.0025
                if move_fn is not None:
                    try:
                        move_fn(px, py)
                    except Exception as exc:
                        logger.debug("MicroCameraLookAround outward step error: %s", exc)
                executed_steps += 1
                if not is_test_mode:
                    time.sleep(max(0.001, min(0.012, dt)))

            if not is_test_mode:
                time.sleep(self._rng.uniform(0.035, 0.075))

            for step in return_path:
                px, py = int(step[0]), int(step[1])
                dt = float(step[2]) if len(step) >= 3 else 0.0025
                if move_fn is not None:
                    try:
                        move_fn(px, py)
                    except Exception as exc:
                        logger.debug("MicroCameraLookAround return step error: %s", exc)
                executed_steps += 1
                if not is_test_mode:
                    time.sleep(max(0.001, min(0.012, dt)))

            self._last_sway_ts = time.monotonic()
            next_interval = self.sample_next_interval()
            self.sway_count += 1

            summary = {
                "triggered": True,
                "direction": "RIGHT" if chosen_dir > 0 else "LEFT",
                "angle_deg": round(effective_deg * chosen_dir, 2),
                "delta_px": delta_px,
                "start_pos": (start_x, start_y),
                "peak_pos": (peak_x, peak_y),
                "end_pos": (start_x, start_y),
                "outward_steps": len(outward_path),
                "return_steps": len(return_path),
                "total_micro_steps": executed_steps,
                "next_interval_sec": next_interval,
                "sway_count": self.sway_count,
            }
            self.last_sway_summary = summary
            return summary

    def get_skill_bar_reader_ready_flag(self) -> dict:
        return {"module": "skill_bar_reader", "ready": True}

    def get_skill_reader_health_summary(self) -> dict:
        try:
            return {"skill_reader_ready": True}
        except Exception:
            return {"skill_reader_ready": False}

    def get_skill_bar_reader_capabilities(self) -> list:
        return ["skill_bar_reader", "smart_consumable_manager", "diagnostic_snapshot", "version_info"]


def get_skill_bar_reader_capabilities() -> list:
    return [
        "skill_bar_reader",
        "smart_consumable_manager",
        "dynamic_roi_reload",
        "exemplar_memory_confidence",
        "cooldown_lockout_tracking",
        "buff_scroll_expiration_tracking",
        "diagnostic_snapshot",
        "version_info",
    ]


def validate_skill_bar_reader_contract() -> bool:
    return bool(get_skill_bar_reader_capabilities())


def get_skill_bar_reader_baseline_metrics() -> dict:
    return {"module": "skill_bar_reader", "contract_valid": validate_skill_bar_reader_contract(), "capability_count": len(get_skill_bar_reader_capabilities())}


def get_skill_bar_reader_readiness_status() -> dict:
    return {"module": "skill_bar_reader", "ready": validate_skill_bar_reader_contract(), "metrics": get_skill_bar_reader_baseline_metrics()}


def get_skill_bar_reader_audit_record() -> dict:
    return {"module": "skill_bar_reader", "audit_passed": validate_skill_bar_reader_contract(), "readiness": get_skill_bar_reader_readiness_status()}


def get_skill_bar_reader_operational_digest() -> dict:
    return {"module": "skill_bar_reader", "operational": validate_skill_bar_reader_contract(), "audit": get_skill_bar_reader_audit_record()}


def get_skill_bar_reader_telemetry_payload() -> dict:
    return {"module": "skill_bar_reader", "telemetry_ready": validate_skill_bar_reader_contract(), "digest": get_skill_bar_reader_operational_digest()}


def get_skill_bar_reader_integrity_seal() -> dict:
    return {"module": "skill_bar_reader", "sealed": validate_skill_bar_reader_contract(), "telemetry": get_skill_bar_reader_telemetry_payload()}


def get_skill_bar_reader_compliance_certificate() -> dict:
    return {"module": "skill_bar_reader", "certified": validate_skill_bar_reader_contract(), "seal": get_skill_bar_reader_integrity_seal()}


def get_skill_bar_reader_readiness_probe() -> dict:
    return {"module": "skill_bar_reader", "ready": validate_skill_bar_reader_contract(), "certificate": get_skill_bar_reader_compliance_certificate()}


def get_skill_bar_reader_audit_summary() -> dict:
    return {"module": "skill_bar_reader", "audit_passed": validate_skill_bar_reader_contract(), "probe": get_skill_bar_reader_readiness_probe()}


def get_skill_bar_reader_heartbeat_snapshot() -> dict:
    return {"module": "skill_bar_reader", "alive": validate_skill_bar_reader_contract(), "audit": get_skill_bar_reader_audit_summary()}


# Alias for backwards compatibility with combat_recovery and perception subsystems
AdaptiveSkillBarReader = SkillBarReader

CONSUMABLE_SLOTS: Tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "X", "B")


class ConsumableActionResult(dict):
    """
    Hybrid dict/key result returned by `SmartConsumableManager.decide_consumable_action`.
    Works both as a full diagnostic dictionary (`res["hp_potion"]`, `res["keys_to_press"]`, etc.)
    and compares equal to the primary chosen key string (`res == "F1"`, `res in ("F1", "F3")`).
    """

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            primary = self.get("key")
            if primary is not None and primary == other:
                return True
            if other in self.get("keys_to_press", ()):
                return True
            if self.get("action") == other:
                return True
            return False
        if other is None:
            return self.get("key") is None and not self.get("keys_to_press")
        return super().__eq__(other)

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __contains__(self, item: object) -> bool:
        if super().__contains__(item):
            return True
        if item in self.get("keys_to_press", ()):
            return True
        if item is not None and item == self.get("key"):
            return True
        return False

    def __str__(self) -> str:
        return str(self.get("key") or "")

    def upper(self) -> str:
        return str(self.get("key") or "").upper()

    def lower(self) -> str:
        return str(self.get("key") or "").lower()

    def strip(self, chars: Optional[str] = None) -> str:
        return str(self.get("key") or "").strip(chars)

    def startswith(self, prefix: Any, *args: Any) -> bool:
        return str(self.get("key") or "").startswith(prefix, *args)


class SmartConsumableManager:
    """
    Item 46 — Akıllı Tüketilebilir (Pot ve Parşömen) Yöneticisi (`SmartConsumableManager`).

    Inspects upper consumable hotbar row (`y ≈ 0.870 .. 0.938*H`, keybinds `F1..F8, X, B`):
    - Differentiates consumables from lower combat skills (`y >= 0.940*H`).
    - Learns slot assignments dynamically using pixel color hue density & ExemplarMemory:
        * hp_potion: Red liquid hue density (H < 10 or H > 168, S >= 50, V >= 35).
        * mp_potion: Blue liquid hue density (H 90..140) or Yellow liquid hue density (H 18..42, high S).
        * buff_scroll: Beige parchment paper icon (`H:10..28, S:30..175, V:85..235`).
        * empty: Low saturation / low contrast (`saturation < 30` or `brightness < 35`).
    - Tracks cooldown / readiness state of each slot (`ready` vs `depleted`/`on_cooldown`).
    - Tracks buff duration (`scroll_duration_sec=300.0`) and expiration timers for scroll slots (`F5, F6, F7`).
    - Connects with `ExemplarMemory` so user approvals/rejections directly teach the manager.
    """

    DEFAULT_CONSUMABLE_ROW_ROI: Tuple[float, float, float, float] = (0.345, 0.870, 0.655, 0.938)
    CONSUMABLE_ROW_ROI: Any = _ConsumableBarRoiDescriptor(DEFAULT_CONSUMABLE_ROW_ROI)
    CONSUMABLE_SLOTS: Tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "X", "B")
    SCROLL_SLOTS: Tuple[str, ...] = ("F5", "F7")
    FORBIDDEN_MANUAL_SLOTS: Tuple[str, ...] = ("X", "F6", "B")

    CLASS_ID_HP_POTION: int = 22
    CLASS_NAME_HP_POTION: str = "hp_potion"
    CLASS_ID_MP_POTION: int = 23
    CLASS_NAME_MP_POTION: str = "mp_potion"
    CLASS_ID_BUFF_SCROLL: int = 24
    CLASS_NAME_BUFF_SCROLL: str = "buff_scroll"
    CLASS_ID_EMPTY: int = 25
    CLASS_NAME_EMPTY: str = "empty"
    CLASS_ID_AUTOMATIC: int = 26
    CLASS_NAME_AUTOMATIC: str = "automatic"

    DEFAULT_SLOT_ASSIGNMENTS: Dict[str, str] = {
        "F1": "hp_potion",
        "F2": "mp_potion",
        "F3": "hp_potion",
        "F4": "mp_potion",
        "F5": "buff_scroll",
        "F6": "automatic",
        "F7": "buff_scroll",
        "F8": "empty",
        "X": "automatic",
        "B": "automatic",
    }

    @classmethod
    def _resolve_consumable_roi_static(
        cls,
        json_path: Optional[Union[str, Path]] = None,
        fallback: Tuple[float, float, float, float] = (0.345, 0.870, 0.655, 0.938),
    ) -> Tuple[float, float, float, float]:
        path = Path(json_path) if json_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        if not path.is_file():
            return fallback
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            rois_dict: Dict[str, Any] = {}
            if isinstance(data, dict):
                if "rois" in data and isinstance(data["rois"], dict):
                    rois_dict.update(data["rois"])
                for k, v in data.items():
                    if k not in ("rois", "updated_at", "version") and isinstance(v, dict):
                        rois_dict[k] = v

            for key in ("consumable_bar", "consumables", "upper_skill_bar", "consumable_row", "upper_consumable_row"):
                entry = rois_dict.get(key)
                if entry and isinstance(entry, dict):
                    rect: Optional[Tuple[float, float, float, float]] = None
                    if "normalized_roi" in entry and len(entry["normalized_roi"]) == 4:
                        n = [float(v) for v in entry["normalized_roi"]]
                        if n[1] + n[3] > 1.05 and n[3] > n[1]:
                            rect = (round(n[0], 6), round(n[1], 6), round(n[2], 6), round(n[3], 6))
                        else:
                            rect = (
                                round(n[0], 6),
                                round(n[1], 6),
                                round(min(1.0, n[0] + n[2]), 6),
                                round(min(1.0, n[1] + n[3]), 6),
                            )
                    elif "norm_x" in entry and "norm_y" in entry:
                        x = float(entry["norm_x"])
                        y = float(entry["norm_y"])
                        w = float(entry.get("norm_w", 0.0))
                        h = float(entry.get("norm_h", 0.0))
                        rect = (round(x, 6), round(y, 6), round(min(1.0, x + w), 6), round(min(1.0, y + h), 6))
                    if rect and (rect[2] - rect[0]) >= 0.10:
                        return rect

            try:
                from dataset_factory.exemplar_memory import get_calibrated_rois
                for key in ("consumable_bar", "consumables", "upper_skill_bar"):
                    rect = get_calibrated_rois(path).get_normalized_rect(key)
                    if rect and len(rect) == 4 and (rect[2] - rect[0]) >= 0.10:
                        return rect
            except Exception:
                pass
        except Exception:
            pass
        return fallback

    def __init__(
        self,
        calibrated_rois_path: Optional[Union[str, Path]] = None,
        exemplar_memory: Optional[Any] = None,
        consumable_roi: Optional[Sequence[float]] = None,
        default_hp_cooldown_sec: float = 10.0,
        default_mp_cooldown_sec: float = 10.0,
        default_scroll_cooldown_sec: float = 2.0,
        default_buff_duration_sec: float = 300.0,
        scroll_duration_sec: Optional[float] = None,
        hp_threshold: float = 0.65,
        mp_threshold: float = 0.45,
        seed_defaults: bool = True,
        forbidden_slots: Optional[Sequence[str]] = None,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.calibrated_rois_path: Path = (
            Path(calibrated_rois_path) if calibrated_rois_path is not None else DEFAULT_CALIBRATED_ROIS_PATH
        )
        self.exemplar_memory: Optional[Any] = exemplar_memory
        if forbidden_slots is not None:
            self.forbidden_slots: Set[str] = {str(s).strip().upper() for s in forbidden_slots}
        elif seed_defaults:
            self.forbidden_slots = {"X", "F6", "B"}
        else:
            self.forbidden_slots = set()
        self.default_hp_cooldown_sec: float = max(0.5, float(default_hp_cooldown_sec))
        self.default_mp_cooldown_sec: float = max(0.5, float(default_mp_cooldown_sec))
        self.default_scroll_cooldown_sec: float = max(0.2, float(default_scroll_cooldown_sec))

        eff_scroll_dur = float(scroll_duration_sec) if scroll_duration_sec is not None else float(default_buff_duration_sec)
        self.scroll_duration_sec: float = max(1.0, eff_scroll_dur)
        self.default_buff_duration_sec: float = self.scroll_duration_sec
        self.hp_threshold: float = float(hp_threshold)
        self.mp_threshold: float = float(mp_threshold)

        self._custom_roi: Optional[Tuple[float, float, float, float]] = None
        if consumable_roi is not None and len(consumable_roi) == 4:
            self._custom_roi = (
                float(consumable_roi[0]),
                float(consumable_roi[1]),
                float(consumable_roi[2]),
                float(consumable_roi[3]),
            )
        self._cached_roi: Optional[Tuple[float, float, float, float]] = None
        self._last_roi_mtime: float = -1.0

        self._slot_assignments: Dict[str, str] = {}
        self._slot_confidences: Dict[str, float] = {}
        self._slot_last_used: Dict[str, float] = {}
        self._slot_cooldowns: Dict[str, float] = {}
        self._buff_durations: Dict[str, float] = {}
        self._buff_applied_at: Dict[str, Optional[float]] = {}
        self._buff_expires_at: Dict[str, Optional[float]] = {}
        self._last_diagnostics: Dict[str, Dict[str, Any]] = {}

        self._exemplar_store: Dict[str, List[Dict[str, Any]]] = {
            "hp_potion": [],
            "mp_potion": [],
            "buff_scroll": [],
            "empty": [],
        }
        self._negative_exemplar_store: List[Dict[str, Any]] = []

        if seed_defaults:
            for s, itype in self.DEFAULT_SLOT_ASSIGNMENTS.items():
                self._slot_assignments[s] = itype
                self._slot_confidences[s] = 0.85
                if itype == "buff_scroll":
                    self._buff_durations[s] = self.scroll_duration_sec
                    self._slot_cooldowns[s] = self.default_scroll_cooldown_sec
                elif itype == "hp_potion":
                    self._slot_cooldowns[s] = self.default_hp_cooldown_sec
                elif itype == "mp_potion":
                    self._slot_cooldowns[s] = self.default_mp_cooldown_sec

    @staticmethod
    def normalize_key(slot_key: str) -> str:
        s = str(slot_key or "").strip().upper()
        return s

    def get_exemplar_memory(self) -> Optional[Any]:
        if self.exemplar_memory is not None:
            return self.exemplar_memory
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            return get_exemplar_memory()
        except Exception as exc:
            logger.debug("SmartConsumableManager could not resolve ExemplarMemory: %s", exc)
            return None

    def get_consumable_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        if self._custom_roi is not None:
            return self._custom_roi

        json_path = self.calibrated_rois_path
        if not json_path or not json_path.is_file():
            return self.DEFAULT_CONSUMABLE_ROW_ROI

        try:
            mtime = json_path.stat().st_mtime
            if not force and mtime == self._last_roi_mtime and self._cached_roi is not None:
                return self._cached_roi

            with self._lock:
                rect = self._resolve_consumable_roi_static(
                    json_path=json_path,
                    fallback=self.DEFAULT_CONSUMABLE_ROW_ROI,
                )
                self._cached_roi = rect
                self._last_roi_mtime = mtime
                return rect
        except Exception as exc:
            logger.debug("SmartConsumableManager get_consumable_roi error: %s", exc)
            return self._cached_roi or self.DEFAULT_CONSUMABLE_ROW_ROI

    def check_reload_roi(self, force: bool = False) -> Tuple[float, float, float, float]:
        return self.get_consumable_roi(force=force)

    def reload_calibrated_rois(self, force: bool = True) -> Tuple[float, float, float, float]:
        return self.get_consumable_roi(force=force)

    def set_custom_roi(self, roi: Sequence[float]) -> None:
        if len(roi) == 4:
            self._custom_roi = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))

    @property
    def current_consumable_roi(self) -> Tuple[float, float, float, float]:
        return self.get_consumable_roi()

    @property
    def current_consumable_bar_roi(self) -> Tuple[float, float, float, float]:
        """Convenience alias for current_consumable_roi."""
        return self.get_consumable_roi()

    def slot_center_px(
        self,
        frame: Optional[np.ndarray],
        slot_key: str,
        jitter_px: int = 3,
    ) -> Optional[Tuple[int, int]]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        norm_key = self.normalize_key(slot_key)
        slots = list(self.CONSUMABLE_SLOTS)
        idx = None
        for i, s in enumerate(slots):
            if s.upper() == norm_key:
                idx = i
                break
        if idx is None:
            return None
        try:
            fh, fw = frame.shape[:2]
            roi = self.current_consumable_roi
            x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            num_slots = len(slots)
            slot_w = float(bw) / float(num_slots)
            sx1 = x1 + int(round(idx * slot_w))
            sx2 = x1 + int(round((idx + 1) * slot_w))
            cx = (sx1 + sx2) // 2
            cy = y1 + bh // 2
            j = max(0, int(jitter_px))
            if j > 0:
                cx += random.randint(-j, j)
                cy += random.randint(-j, j)
            cx = max(0, min(fw - 1, cx))
            cy = max(0, min(fh - 1, cy))
            return int(cx), int(cy)
        except Exception as exc:
            logger.debug("SmartConsumableManager slot_center_px failed (%s): %s", slot_key, exc)
            return None

    def classify_slot_crop(
        self,
        slot_img: Optional[np.ndarray],
        slot_key: str = "",
    ) -> Dict[str, Any]:
        if slot_img is None or not isinstance(slot_img, np.ndarray) or slot_img.size == 0:
            return {
                "item_type": "empty",
                "state": "empty",
                "ready": False,
                "confidence": 0.0,
                "red_density": 0.0,
                "blue_density": 0.0,
                "yellow_density": 0.0,
                "parchment_density": 0.0,
                "brightness": 0.0,
                "saturation": 0.0,
                "contrast_std": 0.0,
                "exemplar_type": None,
                "exemplar_sim": 0.0,
                "is_empty": True,
                "is_negative": False,
            }

        if slot_img.ndim == 2:
            img_bgr = cv2.cvtColor(slot_img, cv2.COLOR_GRAY2BGR)
        else:
            img_bgr = slot_img[:, :, :3]

        sh, sw = img_bgr.shape[:2]
        if sh >= 8 and sw >= 8:
            pad_y = max(1, int(round(sh * 0.12)))
            pad_x = max(1, int(round(sw * 0.12)))
            inner = img_bgr[pad_y : max(pad_y + 2, sh - pad_y), pad_x : max(pad_x + 2, sw - pad_x)]
            if inner.size == 0:
                inner = img_bgr
        else:
            inner = img_bgr

        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        h_chan = hsv[:, :, 0]
        s_chan = hsv[:, :, 1]
        v_chan = hsv[:, :, 2]

        brightness = float(np.mean(v_chan))
        saturation = float(np.mean(s_chan))
        contrast_std = float(np.std(inner))

        b_chan = inner[:, :, 0].astype(np.int16)
        g_chan = inner[:, :, 1].astype(np.int16)
        r_chan = inner[:, :, 2].astype(np.int16)

        # 1. Red liquid mask (HP Potion: kırmızı sıvı)
        red_mask = (
            ((h_chan < 10) | (h_chan > 168))
            & (s_chan >= 50)
            & (v_chan >= 25)
            & (r_chan > b_chan + 12)
            & (r_chan > g_chan + 10)
        )
        red_density = float(np.mean(red_mask))

        # 2. Blue liquid mask (MP Potion: mavi sıvı)
        blue_mask = (
            (h_chan >= 90)
            & (h_chan <= 140)
            & (s_chan >= 45)
            & (v_chan >= 25)
            & (b_chan > r_chan + 12)
        )
        blue_density = float(np.mean(blue_mask))

        # 3. Beige parchment mask (Buff Scroll: bej parşömen kağıdı H:10..28, S:30..175, V:85..235)
        parchment_mask = (
            (h_chan >= 10)
            & (h_chan <= 28)
            & (s_chan >= 30)
            & (s_chan <= 175)
            & (v_chan >= 85)
            & (v_chan <= 235)
        )
        parchment_density = float(np.mean(parchment_mask))

        # 4. Yellow liquid mask (MP Potion: sarı sıvı — high saturation S > 175 or H:29..42)
        yellow_mask = (
            (((h_chan >= 18) & (h_chan <= 42) & (s_chan > 175) & (v_chan >= 45))
             | ((h_chan >= 29) & (h_chan <= 42) & (s_chan >= 75) & (v_chan >= 45)))
            & (r_chan > b_chan + 15)
            & (g_chan > b_chan + 15)
        )
        yellow_density = float(np.mean(yellow_mask))

        max_color_density = max(red_density, blue_density, yellow_density, parchment_density)
        is_empty = bool(
            max_color_density < 0.12
            and (
                saturation < 30.0
                or brightness < 38.0
                or (brightness < 55.0 and contrast_std < 18.0)
            )
        )

        exemplar_type: Optional[str] = None
        exemplar_sim: float = 0.0
        is_negative: bool = False

        if extract_crop_features is not None and compute_similarity is not None:
            try:
                feats = extract_crop_features(inner)
                if feats is not None:
                    for neg_ex in self._negative_exemplar_store:
                        sim = compute_similarity(feats, neg_ex)
                        if sim >= 0.78:
                            is_negative = True
                            exemplar_sim = max(exemplar_sim, sim)
                            break
                    if not is_negative:
                        for itype, ex_list in self._exemplar_store.items():
                            for ex in ex_list:
                                sim = compute_similarity(feats, ex)
                                if sim > exemplar_sim:
                                    exemplar_sim = sim
                                    exemplar_type = itype
            except Exception as exc:
                logger.debug("Internal exemplar store matching error: %s", exc)

        mem = self.get_exemplar_memory()
        if mem is not None:
            try:
                if hasattr(mem, "is_negative_patch"):
                    is_neg, neg_sim, neg_meta = mem.is_negative_patch(inner, threshold=0.80)
                    if is_neg:
                        neg_cid = neg_meta.get("class_id") if isinstance(neg_meta, dict) else getattr(neg_meta, "class_id", None)
                        neg_src = str(neg_meta.get("source", "") if isinstance(neg_meta, dict) else getattr(neg_meta, "source", "")).lower()
                        neg_cname = str(neg_meta.get("class_name", "") if isinstance(neg_meta, dict) else getattr(neg_meta, "class_name", "")).lower()
                        if (
                            neg_cid in (
                                self.CLASS_ID_HP_POTION,
                                self.CLASS_ID_MP_POTION,
                                self.CLASS_ID_BUFF_SCROLL,
                                self.CLASS_ID_EMPTY,
                            )
                            or "consumable" in neg_src
                            or "consumable" in neg_cname
                            or "rejected" in neg_cname
                            or "empty" in neg_cname
                        ):
                            is_negative = True
                            exemplar_sim = max(exemplar_sim, float(neg_sim))
                if hasattr(mem, "classify_patch"):
                    pos_cid, pos_cname, pos_sim = mem.classify_patch(inner, min_sim=0.65)
                    if pos_cid is not None and float(pos_sim) > exemplar_sim:
                        cname = str(pos_cname or "").lower()
                        if pos_cid == self.CLASS_ID_HP_POTION or "hp" in cname:
                            exemplar_type = "hp_potion"
                            exemplar_sim = float(pos_sim)
                        elif pos_cid == self.CLASS_ID_MP_POTION or "mp" in cname or "mana" in cname:
                            exemplar_type = "mp_potion"
                            exemplar_sim = float(pos_sim)
                        elif pos_cid == self.CLASS_ID_BUFF_SCROLL or "scroll" in cname or "buff" in cname:
                            exemplar_type = "buff_scroll"
                            exemplar_sim = float(pos_sim)
                        elif pos_cid == self.CLASS_ID_EMPTY or "empty" in cname:
                            exemplar_type = "empty"
                            exemplar_sim = float(pos_sim)
                if hasattr(mem, "match_class"):
                    for cid, cname in (
                        (self.CLASS_ID_HP_POTION, "hp_potion"),
                        (self.CLASS_ID_MP_POTION, "mp_potion"),
                        (self.CLASS_ID_BUFF_SCROLL, "buff_scroll"),
                        (self.CLASS_ID_EMPTY, "empty"),
                    ):
                        m_sim, _ = mem.match_class(inner, cid)
                        if float(m_sim) > exemplar_sim and float(m_sim) >= 0.65:
                            exemplar_sim = float(m_sim)
                            exemplar_type = cname
            except Exception as exc:
                logger.debug("External ExemplarMemory query error: %s", exc)

        norm_key = self.normalize_key(slot_key)
        stored_type = self._slot_assignments.get(norm_key)

        if is_negative:
            predicted_type = "empty"
            confidence = round(exemplar_sim, 3)
        elif exemplar_type is not None and exemplar_sim >= 0.68:
            predicted_type = exemplar_type
            confidence = round(exemplar_sim, 3)
        elif max_color_density >= 0.12:
            mp_density = max(blue_density, yellow_density)
            if red_density >= max(mp_density, parchment_density):
                predicted_type = "hp_potion"
                confidence = round(float(np.clip(red_density * 1.5 + 0.35, 0.65, 0.98)), 3)
            elif parchment_density >= max(red_density, mp_density):
                predicted_type = "buff_scroll"
                confidence = round(float(np.clip(parchment_density * 1.4 + 0.35, 0.65, 0.98)), 3)
            else:
                predicted_type = "mp_potion"
                confidence = round(float(np.clip(mp_density * 1.5 + 0.35, 0.65, 0.98)), 3)
        elif is_empty or not norm_key:
            predicted_type = "empty"
            confidence = 0.85
        elif stored_type is not None:
            predicted_type = stored_type
            confidence = round(self._slot_confidences.get(norm_key, 0.60), 3)
        else:
            predicted_type = "empty"
            confidence = 0.50

        return {
            "item_type": predicted_type,
            "confidence": confidence,
            "is_empty": is_empty,
            "is_negative": is_negative,
            "exemplar_type": exemplar_type,
            "exemplar_sim": round(exemplar_sim, 3),
            "red_density": round(red_density, 4),
            "blue_density": round(blue_density, 4),
            "yellow_density": round(yellow_density, 4),
            "parchment_density": round(parchment_density, 4),
            "brightness": round(brightness, 2),
            "saturation": round(saturation, 2),
            "contrast_std": round(contrast_std, 2),
        }

    def classify_consumable_slot_crop(
        self,
        crop_bgr: Optional[np.ndarray],
        slot_key: str = "",
    ) -> str:
        """
        Classifies a single consumable slot crop into one of:
        - `"hp_potion"` (red liquid)
        - `"mp_potion"` (blue or yellow liquid)
        - `"buff_scroll"` (beige parchment paper `H:10..28, S:30..175, V:85..235`)
        - `"empty"` (low saturation / low contrast)
        Also matches against `ExemplarMemory`.
        """
        diag = self.classify_slot_crop(crop_bgr, slot_key=slot_key)
        itype = str(diag.get("item_type", "empty"))
        if itype not in ("hp_potion", "mp_potion", "buff_scroll", "empty", "automatic"):
            return "empty"
        return itype

    def inspect_consumables(
        self,
        frame: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {}
        try:
            fh, fw = frame.shape[:2]
            roi = self.current_consumable_roi
            x1 = max(0, min(int(round(roi[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(roi[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(roi[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(roi[3] * fh)), fh))

            row_crop = frame[y1:y2, x1:x2]
            if row_crop.size == 0:
                return {}

            if row_crop.ndim == 2:
                row_bgr = cv2.cvtColor(row_crop, cv2.COLOR_GRAY2BGR)
            else:
                row_bgr = row_crop[:, :, :3]

            bh, bw = row_bgr.shape[:2]
            num_slots = len(self.CONSUMABLE_SLOTS)
            slot_w = float(bw) / float(num_slots)

            t = time.monotonic() if now is None else float(now)
            results: Dict[str, Dict[str, Any]] = {}
            is_blank_frame = bool(np.max(row_bgr) == 0)

            with self._lock:
                for idx, slot_key in enumerate(self.CONSUMABLE_SLOTS):
                    norm_key = self.normalize_key(slot_key)
                    sx1 = max(0, int(round(idx * slot_w)))
                    sx2 = max(sx1 + 2, min(bw, int(round((idx + 1) * slot_w))))
                    slot_img = row_bgr[:, sx1:sx2]

                    diag = self.classify_slot_crop(slot_img, slot_key="")
                    classified_type = diag["item_type"]
                    conf = diag["confidence"]

                    if not is_blank_frame and classified_type in ("hp_potion", "mp_potion", "buff_scroll"):
                        self._slot_assignments[norm_key] = classified_type
                        self._slot_confidences[norm_key] = conf
                        effective_type = classified_type
                    elif not is_blank_frame and classified_type == "empty":
                        effective_type = self._slot_assignments.get(norm_key, "empty")
                    else:
                        effective_type = self._slot_assignments.get(norm_key, classified_type)

                    last_used = self._slot_last_used.get(norm_key, 0.0)
                    cd_dur = self._slot_cooldowns.get(norm_key, self._get_default_cd(effective_type))
                    cooldown_remaining = max(0.0, cd_dur - (t - last_used)) if last_used > 0.0 else 0.0
                    is_locked_out = cooldown_remaining > 0.0

                    visual_darkened = bool(
                        not is_blank_frame
                        and (
                            classified_type == "empty"
                            or diag["brightness"] < 55.0
                            or diag["saturation"] < 28.0
                        )
                    )

                    if classified_type == "empty" and effective_type == "empty":
                        state = "empty"
                        status = "empty"
                        is_ready = False
                        is_depleted = True
                        is_on_cd = False
                    elif is_locked_out:
                        state = "on_cooldown"
                        status = "on_cooldown"
                        is_ready = False
                        is_depleted = False
                        is_on_cd = True
                    elif visual_darkened:
                        state = "depleted"
                        status = "depleted"
                        is_ready = False
                        is_depleted = True
                        is_on_cd = True
                    elif is_blank_frame and effective_type == "empty":
                        state = "empty"
                        status = "empty"
                        is_ready = False
                        is_depleted = True
                        is_on_cd = False
                    else:
                        state = "ready"
                        status = "ready"
                        is_ready = True
                        is_depleted = False
                        is_on_cd = False

                    buff_active = False
                    buff_expired = False
                    buff_remaining_sec = 0.0

                    if effective_type == "buff_scroll" or classified_type == "buff_scroll":
                        applied_at = self._buff_applied_at.get(norm_key)
                        expires_at = self._buff_expires_at.get(norm_key)
                        if applied_at is None or expires_at is None or t >= expires_at:
                            buff_expired = True
                            buff_active = False
                            buff_remaining_sec = 0.0
                        else:
                            buff_expired = False
                            buff_active = True
                            buff_remaining_sec = round(max(0.0, expires_at - t), 2)

                    reported_type = classified_type if (not is_blank_frame and classified_type != "empty") else effective_type

                    slot_info = {
                        "slot": norm_key,
                        "type": reported_type,
                        "item_type": reported_type,
                        "consumable_type": reported_type,
                        "detected_type": classified_type,
                        "state": state,
                        "status": status,
                        "ready": is_ready,
                        "depleted": is_depleted,
                        "on_cooldown": is_on_cd,
                        "confidence": conf,
                        "cooldown_remaining_sec": round(cooldown_remaining, 3),
                        "buff_active": buff_active,
                        "buff_expired": buff_expired,
                        "expired": buff_expired,
                        "buff_remaining_sec": buff_remaining_sec,
                        "red_density": diag["red_density"],
                        "blue_density": diag["blue_density"],
                        "yellow_density": diag["yellow_density"],
                        "parchment_density": diag["parchment_density"],
                        "brightness": diag["brightness"],
                        "saturation": diag["saturation"],
                        "contrast_std": diag["contrast_std"],
                        "exemplar_sim": diag["exemplar_sim"],
                    }
                    results[norm_key] = slot_info
                    self._last_diagnostics[norm_key] = slot_info

            return results
        except Exception as exc:
            logger.error("SmartConsumableManager.inspect_consumables failed: %s", exc)
            return {}

    def _get_default_cd(self, item_type: str) -> float:
        if item_type == "hp_potion":
            return self.default_hp_cooldown_sec
        elif item_type == "mp_potion":
            return self.default_mp_cooldown_sec
        elif item_type == "buff_scroll":
            return self.default_scroll_cooldown_sec
        return 1.0

    def get_slot_assignment(self, slot_key: str) -> Optional[str]:
        """Returns the current learned item type ('hp_potion', 'mp_potion', 'buff_scroll', 'empty') for a slot."""
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            return self._slot_assignments.get(norm_key)

    def get_slot_confidence(self, slot_key: str) -> float:
        """Returns the confidence score [0.0..1.0] for the slot assignment."""
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            return float(self._slot_confidences.get(norm_key, 0.0))

    def learn_slot_assignment(
        self,
        slot_key: str,
        item_type: str,
        crop: Optional[np.ndarray] = None,
        buff_duration_sec: Optional[float] = None,
    ) -> bool:
        norm_key = self.normalize_key(slot_key)
        clean_type = str(item_type or "").strip().lower()
        if clean_type not in ("hp_potion", "mp_potion", "buff_scroll", "empty"):
            clean_type = "unknown"

        with self._lock:
            self._slot_assignments[norm_key] = clean_type
            self._slot_confidences[norm_key] = 0.99
            if buff_duration_sec is not None:
                self._buff_durations[norm_key] = max(1.0, float(buff_duration_sec))

            if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
                if extract_crop_features is not None:
                    try:
                        feats = extract_crop_features(crop)
                        if feats is not None:
                            self._exemplar_store.setdefault(clean_type, []).append(feats)
                            if len(self._exemplar_store[clean_type]) > 40:
                                self._exemplar_store[clean_type].pop(0)
                    except Exception as exc:
                        logger.debug("Failed saving internal exemplar: %s", exc)

                mem = self.get_exemplar_memory()
                if mem is not None:
                    cid = (
                        self.CLASS_ID_HP_POTION
                        if clean_type == "hp_potion"
                        else (
                            self.CLASS_ID_MP_POTION
                            if clean_type == "mp_potion"
                            else (self.CLASS_ID_BUFF_SCROLL if clean_type == "buff_scroll" else self.CLASS_ID_EMPTY)
                        )
                    )
                    try:
                        mem.add_positive_exemplar(
                            crop,
                            class_id=cid,
                            class_name=clean_type,
                            source="consumable_manager_approval",
                        )
                    except Exception as exc:
                        logger.debug("External ExemplarMemory add_positive_exemplar: %s", exc)
        return True

    def reject_slot_assignment(
        self,
        slot_key: str,
        crop: Optional[np.ndarray] = None,
    ) -> bool:
        norm_key = self.normalize_key(slot_key)
        with self._lock:
            self._slot_assignments[norm_key] = "empty"
            self._slot_confidences[norm_key] = 0.85

            if crop is not None and isinstance(crop, np.ndarray) and crop.size > 0:
                if extract_crop_features is not None:
                    try:
                        feats = extract_crop_features(crop)
                        if feats is not None:
                            self._negative_exemplar_store.append(feats)
                            if len(self._negative_exemplar_store) > 100:
                                self._negative_exemplar_store.pop(0)
                    except Exception as exc:
                        logger.debug("Failed saving negative internal exemplar: %s", exc)

                mem = self.get_exemplar_memory()
                if mem is not None:
                    try:
                        mem.add_negative_exemplar(
                            crop,
                            class_id=self.CLASS_ID_EMPTY,
                            class_name="rejected_consumable",
                            source="consumable_manager_rejection",
                        )
                    except Exception as exc:
                        logger.debug("External ExemplarMemory add_negative_exemplar: %s", exc)
        return True

    def mark_used(
        self,
        slot_key: str,
        now: Optional[float] = None,
        cooldown_sec: Optional[float] = None,
    ) -> None:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            self._slot_last_used[norm_key] = t
            if cooldown_sec is not None:
                self._slot_cooldowns[norm_key] = max(0.1, float(cooldown_sec))
            itype = self._slot_assignments.get(norm_key)
            if itype == "buff_scroll":
                dur = self._buff_durations.get(norm_key, self.scroll_duration_sec)
                self._buff_applied_at[norm_key] = t
                self._buff_expires_at[norm_key] = t + dur

    def mark_buff_applied(
        self,
        slot_key: str,
        duration_sec: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            dur = (
                max(1.0, float(duration_sec))
                if duration_sec is not None
                else self._buff_durations.get(norm_key, self.scroll_duration_sec)
            )
            self._buff_durations[norm_key] = dur
            self._buff_applied_at[norm_key] = t
            self._buff_expires_at[norm_key] = t + dur
            self._slot_last_used[norm_key] = t
            self._slot_cooldowns[norm_key] = self.default_scroll_cooldown_sec

    def is_slot_ready(self, slot_key: str, now: Optional[float] = None) -> bool:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            itype = self._slot_assignments.get(norm_key, "unknown")
            if itype == "empty":
                return False
            last_used = self._slot_last_used.get(norm_key, 0.0)
            cd_dur = self._slot_cooldowns.get(norm_key, self._get_default_cd(itype))
            if last_used > 0.0 and (t - last_used) < cd_dur:
                return False
            diag = self._last_diagnostics.get(norm_key)
            if diag is not None and not diag.get("ready", True):
                return False
            return True

    def is_buff_expired(self, slot_key: str, now: Optional[float] = None) -> bool:
        norm_key = self.normalize_key(slot_key)
        t = time.monotonic() if now is None else float(now)
        with self._lock:
            itype = self._slot_assignments.get(norm_key)
            if itype != "buff_scroll" and norm_key not in self.SCROLL_SLOTS:
                return False
            applied = self._buff_applied_at.get(norm_key)
            expires = self._buff_expires_at.get(norm_key)
            dur = self._buff_durations.get(norm_key, self.scroll_duration_sec)
            if applied is None:
                return True
            if expires is not None:
                return t >= expires
            return (t - applied) >= dur

    def is_buff_active(self, slot_key: str, now: Optional[float] = None) -> bool:
        return not self.is_buff_expired(slot_key, now=now)

    def get_best_hp_potion(self, now: Optional[float] = None) -> Optional[str]:
        t = time.monotonic() if now is None else float(now)
        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            ready_slots = []
            for slot_key, itype in self._slot_assignments.items():
                if slot_key.upper() in forbidden:
                    continue
                if itype == "hp_potion" and self.is_slot_ready(slot_key, now=t):
                    ready_slots.append(slot_key)
            if not ready_slots:
                return None
            ready_slots.sort(
                key=lambda s: (
                    self._slot_last_used.get(s, 0.0),
                    self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99,
                )
            )
            return ready_slots[0]

    def get_best_mp_potion(self, now: Optional[float] = None) -> Optional[str]:
        t = time.monotonic() if now is None else float(now)
        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            ready_slots = []
            for slot_key, itype in self._slot_assignments.items():
                if slot_key.upper() in forbidden:
                    continue
                if itype == "mp_potion" and self.is_slot_ready(slot_key, now=t):
                    ready_slots.append(slot_key)
            if not ready_slots:
                return None
            ready_slots.sort(
                key=lambda s: (
                    self._slot_last_used.get(s, 0.0),
                    self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99,
                )
            )
            return ready_slots[0]

    def get_expired_buff_scrolls(self, now: Optional[float] = None) -> List[str]:
        t = time.monotonic() if now is None else float(now)
        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            expired = []
            for slot_key, itype in self._slot_assignments.items():
                if slot_key.upper() in forbidden:
                    continue
                if itype == "buff_scroll":
                    if self.is_buff_expired(slot_key, now=t) and self.is_slot_ready(slot_key, now=t):
                        expired.append(slot_key)
            expired.sort(key=lambda s: (self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99))
            return expired

    def identify_consumable_slots(
        self,
        frame: Optional[np.ndarray],
        now: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Inspects and classifies upper consumable slots (`F1..F8, X, B`) from `frame`.
        Returns per-slot dictionary indicating which potion/scroll is in each slot and
        whether it is `ready` (`True`) or `depleted`/`on_cooldown` (`False`).
        """
        return self.inspect_consumables(frame, now=now)

    def check_and_refresh_scroll_buffs(
        self,
        frame: Optional[Any] = None,
        now: Optional[float] = None,
        *,
        active_buffs: Optional[Any] = None,
    ) -> List[str]:
        """
        Checks `buff_scroll` slots and returns the list of keys to press
        automatically when their duration (`scroll_duration_sec=300.0`) has expired.
        Supports both `check_and_refresh_scroll_buffs(frame, now=None)` and
        `check_and_refresh_scroll_buffs(active_buffs=..., now=None)`.
        Excludes automatic client slots (`F6, X, B` when enabled).
        """
        actual_frame: Optional[np.ndarray] = None
        if frame is not None:
            if isinstance(frame, np.ndarray):
                actual_frame = frame
            elif isinstance(frame, (int, float)) and now is None:
                now = float(frame)
            elif active_buffs is None and isinstance(frame, (dict, list, set, tuple)):
                active_buffs = frame

        t = time.monotonic() if now is None else float(now)

        if actual_frame is not None and actual_frame.size > 0 and int(np.max(actual_frame)) > 0:
            self.identify_consumable_slots(actual_frame, now=t)

        keys_to_refresh: List[str] = []
        active_keys_set = set()
        if active_buffs is not None:
            if isinstance(active_buffs, dict):
                active_keys_set = {str(k).upper() for k, v in active_buffs.items() if v}
            else:
                active_keys_set = {str(k).upper() for k in active_buffs}

        forbidden = getattr(self, "forbidden_slots", set())
        with self._lock:
            scroll_candidates = [
                slot_key
                for slot_key, itype in self._slot_assignments.items()
                if itype == "buff_scroll"
                and slot_key.upper() not in forbidden
            ]
            if not scroll_candidates:
                scroll_candidates = [
                    s for s in self.SCROLL_SLOTS
                    if s.upper() not in forbidden
                ]

            for slot_key in scroll_candidates:
                if not self.is_slot_ready(slot_key, now=t):
                    continue
                if active_buffs is not None:
                    is_active = (slot_key in active_keys_set) or (slot_key.lower() in active_keys_set)
                    if not is_active:
                        keys_to_refresh.append(slot_key)
                        self.mark_buff_applied(slot_key, duration_sec=self.scroll_duration_sec, now=t)
                else:
                    if self.is_buff_expired(slot_key, now=t):
                        keys_to_refresh.append(slot_key)
                        self.mark_buff_applied(slot_key, duration_sec=self.scroll_duration_sec, now=t)

            keys_to_refresh.sort(key=lambda s: (self.CONSUMABLE_SLOTS.index(s) if s in self.CONSUMABLE_SLOTS else 99))
            return keys_to_refresh

    def decide_consumable_action(
        self,
        frame_or_hp: Any = None,
        hp_pct: Optional[float] = None,
        mp_pct: Optional[float] = None,
        now: Optional[float] = None,
        *,
        frame: Optional[np.ndarray] = None,
        player_hp_pct: Optional[float] = None,
        player_mp_pct: Optional[float] = None,
        active_buffs: Optional[Any] = None,
    ) -> ConsumableActionResult:
        """
        Decides consumable action from `frame`, `hp_pct`, `mp_pct`, and `now`:
        - If `hp_pct < 0.65`: selects a ready `hp_potion` slot key.
        - If `mp_pct < 0.45`: selects a ready `mp_potion` slot key.
        - If a `buff_scroll` (`F5, F6, F7`) has expired (`scroll_duration_sec=300.0`): selects the ready `buff_scroll` key.
        Supports both `(frame, hp_pct, mp_pct, now=None)` and `(player_hp_pct, player_mp_pct, active_buffs=None, now=None)`.
        """
        actual_frame: Optional[np.ndarray] = frame
        if isinstance(frame_or_hp, np.ndarray):
            actual_frame = frame_or_hp
            eff_hp = hp_pct if hp_pct is not None else (player_hp_pct if player_hp_pct is not None else 1.0)
            eff_mp = mp_pct if mp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0)
        elif frame_or_hp is None:
            eff_hp = hp_pct if hp_pct is not None else (player_hp_pct if player_hp_pct is not None else 1.0)
            eff_mp = mp_pct if mp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0)
        elif isinstance(frame_or_hp, (int, float)):
            if mp_pct is None and player_mp_pct is None and isinstance(hp_pct, (int, float)):
                eff_hp = float(frame_or_hp)
                eff_mp = float(hp_pct)
            elif isinstance(hp_pct, (int, float)) and isinstance(mp_pct, (dict, list, set, tuple)):
                eff_hp = float(frame_or_hp)
                eff_mp = float(hp_pct)
                active_buffs = mp_pct
            elif isinstance(hp_pct, (int, float)) and isinstance(mp_pct, (int, float)) and now is None:
                eff_hp = float(frame_or_hp)
                eff_mp = float(hp_pct)
                now = float(mp_pct)
            else:
                eff_hp = float(frame_or_hp)
                eff_mp = float(hp_pct if hp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0))
        else:
            eff_hp = hp_pct if hp_pct is not None else (player_hp_pct if player_hp_pct is not None else 1.0)
            eff_mp = mp_pct if mp_pct is not None else (player_mp_pct if player_mp_pct is not None else 1.0)

        t = time.monotonic() if now is None else float(now)

        if actual_frame is not None and isinstance(actual_frame, np.ndarray) and actual_frame.size > 0 and int(np.max(actual_frame)) > 0:
            self.identify_consumable_slots(actual_frame, now=t)

        hp = float(eff_hp) if float(eff_hp) <= 1.0 else float(eff_hp) / 100.0
        mp = float(eff_mp) if float(eff_mp) <= 1.0 else float(eff_mp) / 100.0
        hp = max(0.0, min(1.0, hp))
        mp = max(0.0, min(1.0, mp))

        is_emergency = hp < 0.40
        hp_needed = hp < self.hp_threshold
        mp_needed = mp < self.mp_threshold

        chosen_hp: Optional[str] = None
        chosen_mp: Optional[str] = None
        reasons: List[str] = []

        if hp_needed:
            hp_cand = self.get_best_hp_potion(now=t)
            if hp_cand and self.is_slot_ready(hp_cand, now=t):
                chosen_hp = hp_cand
                self.mark_used(hp_cand, now=t)
                severity = "EMERGENCY_HP" if is_emergency else "HP_POTION"
                reasons.append(f"{severity}: {hp*100:.1f}% < {self.hp_threshold*100:.0f}% -> [{chosen_hp}]")

        if mp_needed:
            mp_cand = self.get_best_mp_potion(now=t)
            if mp_cand and self.is_slot_ready(mp_cand, now=t):
                chosen_mp = mp_cand
                self.mark_used(mp_cand, now=t)
                reasons.append(f"MP_POTION: {mp*100:.1f}% < {self.mp_threshold*100:.0f}% -> [{chosen_mp}]")

        scroll_keys = self.check_and_refresh_scroll_buffs(
            frame=None,
            now=t,
            active_buffs=active_buffs,
        )
        if scroll_keys:
            reasons.append(f"BUFF_REFRESH -> {scroll_keys}")

        chosen_scroll: Optional[str] = scroll_keys[0] if scroll_keys else None

        forbidden = getattr(self, "forbidden_slots", set())
        keys_to_press: List[str] = []
        if chosen_hp and str(chosen_hp).upper() not in forbidden:
            keys_to_press.append(chosen_hp)
        if chosen_mp and str(chosen_mp).upper() not in forbidden:
            keys_to_press.append(chosen_mp)
        if scroll_keys:
            keys_to_press.extend([
                s for s in scroll_keys
                if str(s).upper() not in forbidden
            ])

        primary_key: Optional[str] = keys_to_press[0] if keys_to_press else None
        action_name = "emergency" if is_emergency else ("consumable_buff" if keys_to_press else "none")

        return ConsumableActionResult(
            {
                "action": action_name,
                "key": primary_key,
                "recommended_key": primary_key,
                "emergency": is_emergency,
                "hp_potion": chosen_hp,
                "mp_potion": chosen_mp,
                "buff_scroll": chosen_scroll,
                "hp_potion_key": chosen_hp,
                "mp_potion_key": chosen_mp,
                "scroll_key": chosen_scroll,
                "scroll_buffs": scroll_keys,
                "buff_keys": scroll_keys,
                "keys_to_press": keys_to_press,
                "player_hp": hp,
                "player_mp": mp,
                "hp_pct": hp,
                "mp_pct": mp,
                "reason": "; ".join(reasons) if reasons else "vitals_healthy",
            }
        )

    def get_slot_diagnostics(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._last_diagnostics)

    def get_diagnostic_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            mem = self.get_exemplar_memory()
            return {
                "module": "smart_consumable_manager",
                "current_roi": self.current_consumable_roi,
                "slots_count": len(self.CONSUMABLE_SLOTS),
                "learned_assignments": dict(self._slot_assignments),
                "hp_potions": [s for s, t in self._slot_assignments.items() if t == "hp_potion"],
                "mp_potions": [s for s, t in self._slot_assignments.items() if t == "mp_potion"],
                "buff_scrolls": [s for s, t in self._slot_assignments.items() if t == "buff_scroll"],
                "empty_slots": [s for s, t in self._slot_assignments.items() if t == "empty"],
                "exemplar_store_counts": {k: len(v) for k, v in self._exemplar_store.items()},
                "negative_exemplar_count": len(self._negative_exemplar_store),
                "external_exemplar_memory": mem is not None,
            }


def classify_consumable_slot_crop(
    crop_bgr: Optional[np.ndarray],
    exemplar_memory: Optional[Any] = None,
) -> str:
    """Module-level convenience wrapper for `SmartConsumableManager.classify_consumable_slot_crop`."""
    mgr = SmartConsumableManager(exemplar_memory=exemplar_memory, seed_defaults=False)
    return mgr.classify_consumable_slot_crop(crop_bgr)


__all__ = [
    "CONSUMABLE_SLOTS",
    "SkillBarReader",
    "SmartConsumableManager",
    "ConsumableActionResult",
    "MicroCameraLookAround",
    "AdaptiveSkillBarReader",
    "classify_consumable_slot_crop",
    "get_skill_bar_reader_capabilities",
    "validate_skill_bar_reader_contract",
]


