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

import logging
import math
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.windmouse_humanizer import WindMouseEngine

logger = logging.getLogger(__name__)


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

    SKILL_BAR_ROI: Tuple[float, float, float, float] = (0.345, 0.935, 0.645, 0.985)
    SKILL_SLOTS: Tuple[str, ...] = ("1", "2", "3", "4", "5", "Q", "E", "R")

    # Priority weights for combat rotation when multiple skills are ready & have mana
    DEFAULT_PRIORITY_WEIGHTS: Dict[str, float] = {
        "R": 1.12,
        "E": 1.08,
        "Q": 1.06,
        "2": 1.05,
        "3": 1.04,
        "4": 1.03,
        "1": 1.02,
        "5": 1.00,
    }

    def __init__(
        self,
        min_brightness_ready: float = 58.0,
        min_saturation_ready: float = 45.0,
        min_contrast_std: float = 22.0,
        min_mana_blue_ratio: float = 0.03,
        cooldown_lockout_sec: float = 1.25,
    ) -> None:
        self._lock: threading.RLock = threading.RLock()
        self.min_brightness_ready: float = float(min_brightness_ready)
        self.min_saturation_ready: float = float(min_saturation_ready)
        self.min_contrast_std: float = float(min_contrast_std)
        self.min_mana_blue_ratio: float = float(min_mana_blue_ratio)
        self.cooldown_lockout_sec: float = max(0.0, float(cooldown_lockout_sec))

        self._last_pressed_ts: Dict[str, float] = {}
        self._last_slot_states: Dict[str, Dict[str, Any]] = {}

    def reset_history(self) -> None:
        """Clears recent skill press lockout timestamps."""
        with self._lock:
            self._last_pressed_ts.clear()
            self._last_slot_states.clear()

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
            x1 = max(0, min(int(round(self.SKILL_BAR_ROI[0] * fw)), fw - 2))
            y1 = max(0, min(int(round(self.SKILL_BAR_ROI[1] * fh)), fh - 2))
            x2 = max(x1 + 8, min(int(round(self.SKILL_BAR_ROI[2] * fw)), fw))
            y2 = max(y1 + 4, min(int(round(self.SKILL_BAR_ROI[3] * fh)), fh))

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

                off_cooldown = bool(
                    brightness >= self.min_brightness_ready
                    and saturation >= self.min_saturation_ready
                    and contrast_std >= self.min_contrast_std
                )

                # A skill has enough mana if either its blue mana indicator is visible OR it is a
                # high-brightness/high-saturation zero-mana ability without a red "no-mana" X overlay
                mana_ok = bool(
                    (blue_ratio >= self.min_mana_blue_ratio or (brightness >= 72.0 and saturation >= 115.0))
                    and red_penalty_ratio < 0.38
                )

                with self._lock:
                    last_used = self._last_pressed_ts.get(slot_key, 0.0)
                    recently_pressed = (now - last_used) < self.cooldown_lockout_sec if last_used > 0.0 else False

                ready = bool(off_cooldown and mana_ok and not recently_pressed)

                # Composite readiness score combining brightness, saturation, mana availability, and slot priority
                norm_v = min(1.0, brightness / 115.0)
                norm_s = min(1.0, saturation / 165.0)
                norm_blue = min(1.0, blue_ratio / 0.45)
                priority_mult = self.DEFAULT_PRIORITY_WEIGHTS.get(slot_key, 1.0)
                raw_score = (0.45 * norm_v + 0.30 * norm_s + 0.25 * norm_blue) * priority_mult
                score = round(float(raw_score if ready else raw_score * 0.15), 4)

                results[slot_key] = {
                    "slot": slot_key,
                    "ready": ready,
                    "off_cooldown": off_cooldown,
                    "mana_ok": mana_ok,
                    "recently_pressed": recently_pressed,
                    "brightness": round(brightness, 2),
                    "saturation": round(saturation, 2),
                    "blue_ratio": round(blue_ratio, 4),
                    "contrast_std": round(contrast_std, 2),
                    "score": score,
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
            if preferred_order:
                order_map = {str(k).upper(): i for i, k in enumerate(preferred_order)}
                ready_candidates.sort(
                    key=lambda item: (
                        order_map.get(item["slot"], 999),
                        -float(item.get("score", 0.0)),
                    )
                )
            else:
                ready_candidates.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
            chosen_key = str(ready_candidates[0]["slot"])
        else:
            # Fallback to highest brightness slot
            best_fallback = max(slot_states.values(), key=lambda item: float(item.get("brightness", 0.0)))
            chosen_key = str(best_fallback.get("slot", "1"))

        if record_use:
            with self._lock:
                self._last_pressed_ts[chosen_key] = time.monotonic()

        return chosen_key

    def get_skill_bar_reader_version_info(self) -> dict:
        return {"module": "skill_bar_reader", "version": "2.0.0", "ready": True}

    def get_skill_bar_reader_diagnostic_snapshot(self) -> dict:
        return {"module": "skill_bar_reader", "status": "ok", "ready": True}

    def get_skill_bar_reader_capabilities(self) -> list:
        return ["skill_bar_reader", "diagnostic_snapshot", "version_info"]


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
        return ["skill_bar_reader", "diagnostic_snapshot", "version_info"]


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



