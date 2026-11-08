# -*- coding: utf-8 -*-
"""
GameActionBus — AION2 girdi yolu.

Mouse-look (varsayılan): ekranda imleç yok; fare hareketi = kamera.
  - Skill / Tab: yalnız klavye (scan-code).
  - Skill özellik / tooltip okumak için: Alt basılı tut → imleç açılır →
    skill yuvasının üzerine gel → oku → Alt bırak (kamera kontrolüne dönüş).
Cursor mode (mouse_look_mode=False): eski UI tık yolu.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Gatherion.GameActionBus")

CLOSE_INTERACT_ROI = (0.42, 0.38, 0.69, 0.66)


class GameActionBus:
    """Exclusive lock around UI clicks so combat right-hold is not starved."""

    # AION2 mouse-look: no cursor; mouse motion = camera. Skills/Tab must be keys only.
    mouse_look_mode: bool = True

    def __init__(
        self,
        input_driver: Any = None,
        skill_bar_reader: Any = None,
        vision: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
        release_basic_attack: Optional[Callable[[], None]] = None,
        restore_basic_attack: Optional[Callable[[], None]] = None,
        mouse_look_mode: Optional[bool] = None,
    ) -> None:
        self.input_driver = input_driver
        self.skill_bar_reader = skill_bar_reader
        self.vision = vision
        self._on_log = on_log
        self._release_basic_attack = release_basic_attack
        self._restore_basic_attack = restore_basic_attack
        self._lock = threading.RLock()
        self._exclusive_depth = 0
        self._exclusive_reason = ""
        self.last_actions: List[Dict[str, Any]] = []
        self._max_audit = 40
        self._left_held = False
        self.input_port: Any = None
        if mouse_look_mode is not None:
            self.mouse_look_mode = bool(mouse_look_mode)

    def bind(
        self,
        input_driver: Any = None,
        skill_bar_reader: Any = None,
        vision: Any = None,
        on_log: Optional[Callable[[str], None]] = None,
        release_basic_attack: Optional[Callable[[], None]] = None,
        restore_basic_attack: Optional[Callable[[], None]] = None,
    ) -> None:
        if input_driver is not None:
            self.input_driver = input_driver
        if skill_bar_reader is not None:
            self.skill_bar_reader = skill_bar_reader
        if vision is not None:
            self.vision = vision
        if on_log is not None:
            self._on_log = on_log
        if release_basic_attack is not None:
            self._release_basic_attack = release_basic_attack
        if restore_basic_attack is not None:
            self._restore_basic_attack = restore_basic_attack

    def _emit(self, message: str) -> None:
        logger.info(message)
        if callable(self._on_log):
            try:
                self._on_log(message)
            except Exception:
                pass

    def _audit(self, payload: Dict[str, Any]) -> None:
        payload = dict(payload)
        payload["ts"] = time.time()
        with self._lock:
            self.last_actions.append(payload)
            if len(self.last_actions) > self._max_audit:
                self.last_actions = self.last_actions[-self._max_audit :]

    def begin_exclusive(self, reason: str = "") -> None:
        with self._lock:
            self._exclusive_depth += 1
            self._exclusive_reason = str(reason or "")
            if self._exclusive_depth == 1:
                # Mouse-look: keep LMB basic-attack; only clear RMB (camera) / middle.
                # Full mouse-up was killing AA while skill keys often deal no damage.
                if self.mouse_look_mode:
                    driver = self.input_driver
                    try:
                        if driver is not None:
                            up = getattr(driver, "_hw_mouse_up", None)
                            if callable(up):
                                up("right")
                                up("middle")
                    except Exception as exc:
                        logger.debug("begin_exclusive mouse-look release failed: %s", exc)
                else:
                    if callable(self._release_basic_attack):
                        try:
                            self._release_basic_attack()
                        except Exception as exc:
                            logger.debug("begin_exclusive release failed: %s", exc)
                    self._force_all_mouse_up()

    def end_exclusive(self, restore_attack: bool = True) -> None:
        with self._lock:
            if self._exclusive_depth <= 0:
                return
            self._exclusive_depth -= 1
            if self._exclusive_depth == 0:
                self._exclusive_reason = ""
                # Mouse-look: never move cursor (camera). Cursor mode parks before RMB restore.
                if restore_attack and not self.mouse_look_mode:
                    self._park_cursor_center()
                if restore_attack and callable(self._restore_basic_attack) and not self.mouse_look_mode:
                    try:
                        self._restore_basic_attack()
                    except Exception as exc:
                        logger.debug("end_exclusive restore failed: %s", exc)

    def _force_all_mouse_up(self) -> None:
        driver = self.input_driver
        if driver is None:
            return
        try:
            release = getattr(driver, "_release_mouse_buttons", None)
            if callable(release):
                release()
                return
        except Exception:
            pass
        for btn in ("left", "right", "middle"):
            try:
                up = getattr(driver, "_hw_mouse_up", None)
                if callable(up):
                    up(btn)
            except Exception:
                pass

    def _park_cursor_center(self) -> None:
        """Move cursor to mid-screen so RMB restore does not orbit from the skill bar."""
        driver = self.input_driver
        if driver is None:
            return
        try:
            self._force_all_mouse_up()
            cx, cy = None, None
            if hasattr(driver, "get_game_window_center"):
                try:
                    center = driver.get_game_window_center()
                    if center and len(center) >= 2:
                        cx, cy = int(center[0]), int(center[1])
                except Exception:
                    cx, cy = None, None
            if cx is None or cy is None:
                try:
                    import ctypes

                    class _POINT(ctypes.Structure):
                        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

                    user32 = ctypes.windll.user32
                    w = int(user32.GetSystemMetrics(0))
                    h = int(user32.GetSystemMetrics(1))
                    cx, cy = w // 2, h // 2
                except Exception:
                    return
            if hasattr(driver, "_hw_move_to"):
                driver._hw_move_to(int(cx), int(cy))
            elif hasattr(driver, "smooth_move_to"):
                driver.smooth_move_to(int(cx), int(cy))
        except Exception as exc:
            logger.debug("park cursor failed: %s", exc)

    def hold_basic_attack(self, on: bool) -> None:
        """Historical API: right-click basic attack hold via combat callbacks."""
        if on and callable(self._restore_basic_attack):
            self._restore_basic_attack()
        elif not on and callable(self._release_basic_attack):
            self._release_basic_attack()

    def hold_left_attack(self, on: bool) -> None:
        """Mana regen phase: hold/release left mouse as basic attack."""
        want = bool(on)
        if want == self._left_held:
            return
        port = self.input_port
        driver = self.input_driver
        try:
            if port is not None and hasattr(port, "mouse_button"):
                if port.mouse_button("left", want):
                    self._left_held = want
                    self._audit({"ok": True, "method": "left_hold", "on": want})
                    return
            if driver is None:
                return
            if want:
                down = getattr(driver, "_hw_mouse_down", None)
                if callable(down):
                    down("left")
                    self._left_held = True
                    self._audit({"ok": True, "method": "left_hold", "on": True})
                    return
                send = getattr(driver, "_sendinput_mouse_button", None)
                if callable(send):
                    send("left", True)
                    self._left_held = True
                    self._audit({"ok": True, "method": "left_hold", "on": True})
                    return
            else:
                up = getattr(driver, "_hw_mouse_up", None)
                if callable(up):
                    up("left")
                    self._left_held = False
                    self._audit({"ok": True, "method": "left_hold", "on": False})
                    return
                send = getattr(driver, "_sendinput_mouse_button", None)
                if callable(send):
                    send("left", False)
                    self._left_held = False
                    self._audit({"ok": True, "method": "left_hold", "on": False})
                    return
        except Exception as exc:
            logger.debug("hold_left_attack failed: %s", exc)

    def release_left_attack(self) -> None:
        self.hold_left_attack(False)

    def bind_input_port(self, port: Any) -> None:
        self.input_port = port

    def _click_px(self, x: int, y: int) -> bool:
        # Never drag with a stuck RMB — release first, then move.
        self._force_all_mouse_up()
        port = self.input_port
        if port is not None and hasattr(port, "click"):
            try:
                if port.click(int(x), int(y), button="left"):
                    return True
            except Exception as exc:
                logger.debug("input_port click failed: %s", exc)
        driver = self.input_driver
        if driver is None:
            return False
        try:
            if hasattr(driver, "focus_game_window"):
                driver.focus_game_window()
            self._force_all_mouse_up()
            if hasattr(driver, "human_click"):
                driver.human_click(int(x), int(y), button="left")
                self._force_all_mouse_up()
                return True
            if hasattr(driver, "click_map"):
                driver.click_map(int(x), int(y), button="left")
                self._force_all_mouse_up()
                return True
            if hasattr(driver, "_hw_move_to") and hasattr(driver, "_hw_mouse_down"):
                driver._hw_move_to(int(x), int(y))
                driver._hw_mouse_down("left")
                time.sleep(0.05)
                driver._hw_mouse_up("left")
                return True
        except Exception as exc:
            logger.error("UI click failed at (%s,%s): %s", x, y, exc)
        return False

    def _keyboard_trusted(self) -> bool:
        """Soft SendInput 'success' is not proof AION2 accepted the key."""
        driver = self.input_driver
        if driver is None:
            return False
        # Prefer late-bind NetworkHID before deciding soft-skip → Alt+bar.
        ensure = getattr(driver, "ensure_serial_hid", None)
        if callable(ensure):
            try:
                ensure(force=False)
            except Exception:
                pass
        trusted = getattr(driver, "keyboard_inject_trusted", None)
        if callable(trusted):
            try:
                return bool(trusted())
            except Exception:
                return False
        if isinstance(trusted, bool):
            return bool(trusted)
        return False

    def _trusted_double_pulse_enabled(self) -> bool:
        """Opt-in only. Interception/NetworkHID rarely need a second pulse; it halves cast rate."""
        raw = os.environ.get("GATHERION_TRUSTED_DOUBLE_PULSE", "").strip().lower()
        if raw in ("1", "true", "yes", "on"):
            return True
        if raw in ("0", "false", "no", "off"):
            return False
        driver = self.input_driver
        if driver is not None:
            cfg = getattr(driver, "_last_applied_settings_snapshot", None)
            if isinstance(cfg, dict) and "trusted_double_pulse" in cfg:
                return bool(cfg.get("trusted_double_pulse"))
        return False

    def _key_fallback(self, key: str, min_hold: float = 0.08, max_hold: float = 0.12) -> bool:
        driver = self.input_driver
        if driver is None or not hasattr(driver, "press_key"):
            return False
        key_l = str(key).lower().strip()
        trusted = self._keyboard_trusted()
        try:
            # AION2: stuck Alt = cursor mode; Tab/skills ignored. Clear mods first.
            # Do NOT Esc before Tab/skills — Esc clears the current target.
            if hasattr(driver, "release_modifier_keys"):
                try:
                    driver.release_modifier_keys()
                except Exception:
                    pass
            if hasattr(driver, "focus_game_window"):
                try:
                    driver.focus_game_window()
                except Exception:
                    pass
            # Trusted Interception/HID: skip prepare_game_keyboard — it sleeps 50ms and
            # releases LMB (breaks mouse-look AA). Soft path still needs the full prep.
            if not trusted and hasattr(driver, "prepare_game_keyboard"):
                try:
                    driver.prepare_game_keyboard()
                except Exception:
                    pass
            # Network/Serial/Interception HID: hardware-only single pulse (no PostMessage soft cascade).
            if trusted:
                down = getattr(driver, "_hw_key_down", None)
                up = getattr(driver, "_hw_key_up", None)
                hold_base = max(0.025, min(float(min_hold), float(max_hold)))
                timing = getattr(driver, "_human_timing", None) or getattr(driver, "human_timing", None)
                if timing is not None and hasattr(timing, "sample_gaussian_jitter"):
                    hold = timing.sample_gaussian_jitter(
                        base_sec=hold_base,
                        std_sec=0.003,
                        min_sec=max(0.020, hold_base - 0.006),
                        max_sec=hold_base + 0.010,
                    )
                else:
                    hold = hold_base + random.uniform(-0.003, 0.005)
                ok = False
                if callable(down) and callable(up):
                    try:
                        ok = bool(down(key_l, False)) or ok
                        time.sleep(max(0.018, hold))
                        up(key_l, False)
                    except Exception as exc:
                        logger.debug("trusted HID pulse failed (%s): %s", key_l, exc)
                    # Optional second pulse (env/config). Default OFF for fast skill spam.
                    if self._trusted_double_pulse_enabled():
                        try:
                            time.sleep(0.03)
                            ok = bool(down(key_l, False)) or ok
                            time.sleep(max(0.04, hold))
                            up(key_l, False)
                        except Exception:
                            pass
                    # Key release guard with Gaussian jitter
                    if timing is not None and hasattr(timing, "get_key_release_guard_sec"):
                        guard_sec = timing.get_key_release_guard_sec()
                    else:
                        guard_sec = random.uniform(0.016, 0.026)
                    time.sleep(guard_sec)
                    return bool(ok)
                # Fallback to press_key if hw helpers missing.

            logger.warning(
                "[GameActionBus] [OPSEC FALLBACK] Hardware-trusted keyboard (Interception/HID) unavailable — using soft key dispatch fallback for '%s'",
                key_l,
            )
            self._emit(f"[OPSEC FALLBACK] Soft key dispatch fallback for '{key_l}'")
            driver.press_key(
                key_l,
                min_hold=max(0.12, float(min_hold)),
                max_hold=max(0.20, float(max_hold)),
                release_mouse=True,
                post_to_window=False,
            )
            if trusted:
                return True
            # Soft path only: second/third pulses (untrusted; may still be filtered).
            time.sleep(0.04)
            down = getattr(driver, "_hw_key_down", None)
            up = getattr(driver, "_hw_key_up", None)
            if callable(down) and callable(up):
                try:
                    down(key_l, False)
                    time.sleep(0.08)
                    up(key_l, False)
                except Exception:
                    pass
            # Third pulse with PostMessage for clients that ignore injected SendInput.
            if key_l in ("tab", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0") and callable(down) and callable(up):
                try:
                    time.sleep(0.03)
                    down(key_l, True)
                    time.sleep(0.08)
                    up(key_l, True)
                except Exception:
                    pass
            return True
        except TypeError:
            try:
                driver.press_key(key_l, min_hold=min_hold, max_hold=max_hold, release_mouse=True)
                return True
            except Exception as exc:
                logger.debug("key fallback failed (%s): %s", key, exc)
                return False
        except Exception as exc:
            logger.debug("key fallback failed (%s): %s", key, exc)
            return False

    def cast_skill(self, slot_key: str, frame: Optional[Any] = None) -> Dict[str, Any]:
        """Cast skill. AION2 mouse-look: keyboard only (mouse move = camera)."""
        key = str(slot_key or "1").strip()
        cast_start_ts = time.perf_counter()
        result: Dict[str, Any] = {
            "ok": False,
            "method": "",
            "slot": key,
            "x": None,
            "y": None,
            "burst_duration_sec": 0.0,
            "gcd_align_sec": 0.0,
        }
        self.begin_exclusive(reason=f"skill:{key}")
        restore_attack = False
        trusted = self._keyboard_trusted()
        driver = self.input_driver
        timing = getattr(driver, "_human_timing", None) or getattr(driver, "human_timing", None)
        try:
            # Trusted Interception/HID: minimal settle with Gaussian jitter so skill spam stays fast.
            if timing is not None and hasattr(timing, "sample_gaussian_jitter"):
                settle_delay = timing.sample_gaussian_jitter(
                    base_sec=0.008 if trusted else 0.038,
                    std_sec=0.0015 if trusted else 0.005,
                    min_sec=0.004 if trusted else 0.025,
                    max_sec=0.014 if trusted else 0.055,
                )
            else:
                settle_delay = random.uniform(0.006, 0.011) if trusted else random.uniform(0.030, 0.048)
            time.sleep(settle_delay)
            # Mouse-look (AION2): trusted HID keyboard first; else Alt+skill-bar click.
            # Soft SendInput must NOT short-circuit — it logs success while Raw Input drops keys.
            if self.mouse_look_mode:
                if trusted and self._key_fallback(key, min_hold=0.03, max_hold=0.05):
                    burst_dur = time.perf_counter() - cast_start_ts
                    gcd_rem = self.align_to_gcd(burst_dur, gcd_sec=0.12)
                    result["ok"] = True
                    result["method"] = "key"
                    result["burst_duration_sec"] = burst_dur
                    result["gcd_align_sec"] = gcd_rem
                    result["trusted"] = True
                    msg = f"SKILL KEY '{key}'"
                    self._emit(msg)
                    self._audit(result)
                    reader = self.skill_bar_reader
                    if reader is not None and hasattr(reader, "mark_slot_pressed"):
                        try:
                            reader.mark_slot_pressed(key)
                        except Exception:
                            pass
                    return result
                if not self._keyboard_trusted():
                    logger.warning(
                        "[GameActionBus] [OPSEC FALLBACK] Hardware-trusted keyboard not available — soft fallback triggered for skill '%s'",
                        key,
                    )
                    self._emit(f"[OPSEC FALLBACK] SKILL KEY SKIP '{key}' (soft SendInput untrusted → Alt+bar)")
                # Keyboard filtered / untrusted → Alt+skill-bar click if coords known.
                point = None
                reader = self.skill_bar_reader
                if reader is not None and hasattr(reader, "slot_center_px"):
                    try:
                        point = reader.slot_center_px(frame, key)
                    except Exception as exc:
                        logger.debug("mouse-look slot_center_px failed: %s", exc)
                driver = self.input_driver
                if point is not None and driver is not None:
                    x, y = int(point[0]), int(point[1])
                    result["x"], result["y"] = x, y
                    alt_click = getattr(driver, "cast_skill_alt_click", None)
                    if callable(alt_click) and alt_click(x, y):
                        result["ok"] = True
                        result["method"] = "alt_ui_click"
                        self._emit(f"SKILL ALT_CLICK '{key}' @ ({x},{y})")
                        # Alt+click releases LMB; restore AA hold in mouse-look combat.
                        if self._left_held:
                            try:
                                down = getattr(driver, "_hw_mouse_down", None)
                                if callable(down):
                                    down("left")
                            except Exception:
                                pass
                        # Sticky RMB must stay clear after leaving Alt cursor mode.
                        try:
                            up = getattr(driver, "_hw_mouse_up", None)
                            if callable(up):
                                up("right")
                        except Exception:
                            pass
                        self._audit(result)
                        if reader is not None and hasattr(reader, "mark_slot_pressed"):
                            try:
                                reader.mark_slot_pressed(key)
                            except Exception:
                                pass
                        return result
                # Last resort: reticle LMB (basic attack) — does not cast hotbar skills.
                if driver is not None and hasattr(driver, "_hw_mouse_down"):
                    try:
                        driver._hw_mouse_up("right")
                        driver._hw_mouse_down("left")
                        time.sleep(0.35)
                        # Keep LMB held if combat already requested AA; else pulse only.
                        if not self._left_held:
                            driver._hw_mouse_up("left")
                        result["ok"] = True
                        result["method"] = "lmb_basic"
                        self._emit(f"SKILL LMB_BASIC '{key}' (keyboard inject blocked)")
                        self._audit(result)
                        return result
                    except Exception as exc:
                        logger.debug("lmb basic attack failed: %s", exc)
                result["method"] = "failed"
                self._emit(f"SKILL FAILED '{key}' (mouse-look key)")
                self._audit(result)
                return result

            # Legacy cursor mode: UI click primary, key fallback.
            restore_attack = True
            point = None
            reader = self.skill_bar_reader
            if reader is not None and hasattr(reader, "slot_center_px"):
                try:
                    point = reader.slot_center_px(frame, key)
                except Exception as exc:
                    logger.debug("slot_center_px failed: %s", exc)
            if point is not None:
                x, y = int(point[0]), int(point[1])
                result["x"], result["y"] = x, y
                if self._click_px(x, y):
                    result["ok"] = True
                    result["method"] = "ui_click"
                    msg = f"SKILL UI_CLICK '{key}' @ ({x},{y})"
                    self._emit(msg)
                    self._audit(result)
                    if reader is not None and hasattr(reader, "mark_slot_pressed"):
                        try:
                            reader.mark_slot_pressed(key)
                        except Exception:
                            pass
                    return result
            if self._key_fallback(key):
                result["ok"] = True
                result["method"] = "key_fallback"
                msg = f"SKILL KEY_FALLBACK '{key}'"
                self._emit(msg)
                self._audit(result)
                return result
            result["method"] = "failed"
            self._emit(f"SKILL FAILED '{key}'")
            self._audit(result)
            return result
        finally:
            time.sleep(0.008 if trusted else 0.05)
            self.end_exclusive(restore_attack=restore_attack)

    def gather_interact(
        self,
        frame: Optional[Any] = None,
        preferred_key: str = "f",
    ) -> Dict[str, Any]:
        """Primary: click gather prompt / interaction ROI. Fallback: F/E key."""
        key = str(preferred_key or "f").strip().lower()
        if key not in ("f", "e"):
            key = "f"
        result: Dict[str, Any] = {
            "ok": False,
            "method": "",
            "key": key,
            "x": None,
            "y": None,
        }
        self.begin_exclusive(reason=f"gather:{key}")
        try:
            time.sleep(0.03)
            click_target = self._resolve_gather_click(frame, key)
            if click_target is not None:
                x, y = click_target
                result["x"], result["y"] = x, y
                if self._click_px(x, y):
                    result["ok"] = True
                    result["method"] = "ui_click"
                    msg = f"GATHER UI_CLICK '{key}' @ ({x},{y})"
                    self._emit(msg)
                    self._audit(result)
                    return result
            if self._key_fallback(key):
                result["ok"] = True
                result["method"] = "key_fallback"
                msg = f"GATHER KEY_FALLBACK '{key}'"
                self._emit(msg)
                self._audit(result)
                return result
            result["method"] = "failed"
            self._emit(f"GATHER FAILED '{key}'")
            self._audit(result)
            return result
        finally:
            time.sleep(0.04)
            self.end_exclusive(restore_attack=False)

    def _resolve_gather_click(
        self,
        frame: Optional[Any],
        preferred_key: str,
    ) -> Optional[Tuple[int, int]]:
        vision = self.vision
        if frame is not None and vision is not None and hasattr(vision, "detect_gather_prompt"):
            try:
                found, _conf, bbox = vision.detect_gather_prompt(frame=frame)
                if found and bbox is not None and len(bbox) >= 4:
                    bx, by, bw, bh = [int(v) for v in bbox[:4]]
                    return (
                        bx + max(1, bw // 2) + random.randint(-2, 2),
                        by + max(1, bh // 2) + random.randint(-2, 2),
                    )
                detected = str(getattr(vision, "last_detected_gather_key", "") or "").lower()
                if detected in ("f", "e"):
                    preferred_key = detected
            except Exception as exc:
                logger.debug("detect_gather_prompt for click skipped: %s", exc)

        # Close interaction ROI center (floating [F]/[E] zone).
        if frame is not None:
            try:
                import numpy as np

                if isinstance(frame, np.ndarray) and frame.size > 0:
                    h, w = frame.shape[:2]
                    x1, y1, x2, y2 = CLOSE_INTERACT_ROI
                    cx = int(round(((x1 + x2) * 0.5) * w)) + random.randint(-4, 4)
                    cy = int(round(((y1 + y2) * 0.5) * h)) + random.randint(-4, 4)
                    return (
                        max(0, min(w - 1, cx)),
                        max(0, min(h - 1, cy)),
                    )
            except Exception:
                pass
        _ = preferred_key
        return None

    def press_tab_target(self, frame: Optional[Any] = None, soft_aim: bool = False) -> Dict[str, Any]:
        """Tab for next target; optional mouse-only soft-aim when keys are filtered."""
        result = {"ok": False, "method": "key", "slot": "tab"}
        self.begin_exclusive(reason="tab")
        try:
            time.sleep(0.04)
            # Soft inject often "succeeds" OS-side while AION2 Raw Input drops Tab —
            # only trust kernel/HID keyboard; otherwise skip to soft-aim when requested.
            keyed = False
            if self._keyboard_trusted():
                result["ok"] = self._key_fallback("tab", min_hold=0.10, max_hold=0.16)
                keyed = bool(result["ok"])
                result["method"] = "key" if keyed else "key_failed"
                self._emit("TAB KEY" if keyed else "TAB FAILED")
            else:
                result["ok"] = False
                result["method"] = "key_untrusted"
                logger.warning(
                    "[GameActionBus] [OPSEC FALLBACK] Hardware-trusted keyboard unavailable — TAB key soft fallback triggered",
                )
                self._emit("[OPSEC FALLBACK] TAB KEY SKIPPED (soft SendInput untrusted)")
            # soft_aim=False is a deliberate key-only probe — do NOT fall through to mouse lock.
            if soft_aim:
                driver = self.input_driver
                soft = getattr(driver, "soft_aim_click", None) if driver is not None else None
                if callable(soft):
                    try:
                        # restore_lmb keeps AA after nameplate Alt+click / LMB pulses.
                        soft(frame=frame, restore_lmb=True)
                        result["ok"] = True
                        result["method"] = "tab+soft_aim" if keyed else "soft_aim"
                        self._emit("TAB+SOFT_AIM" if keyed else "SOFT_AIM ONLY")
                        # soft_aim(restore_lmb=True) re-holds LMB after Alt/nameplate work.
                        self._left_held = True
                    except TypeError:
                        try:
                            soft(frame=frame)
                            result["ok"] = True
                            result["method"] = "soft_aim"
                            self._emit("SOFT_AIM ONLY")
                        except Exception as exc:
                            logger.debug("press_tab_target soft_aim failed: %s", exc)
                    except Exception as exc:
                        logger.debug("press_tab_target soft_aim failed: %s", exc)
            self._audit(result)
            return result
        finally:
            time.sleep(0.05)
            # Mouse-look: do not restore RMB (sticky camera). LMB kept by soft_aim/hold.
            self.end_exclusive(restore_attack=False)

    def hover_skill_with_alt(
        self,
        slot_key: str,
        frame: Optional[Any] = None,
        hold_sec: float = 0.55,
        on_hover: Optional[Callable[[int, int], Any]] = None,
    ) -> Dict[str, Any]:
        """
        AION2: hold Alt to show cursor, move onto skill slot (tooltip), release Alt.

        Does NOT cast the skill — combat cast stays keyboard-only.
        `on_hover(x, y)` runs while Alt is still held (OCR tooltip here).
        """
        key = str(slot_key or "1").strip()
        result: Dict[str, Any] = {
            "ok": False,
            "method": "alt_hover",
            "slot": key,
            "x": None,
            "y": None,
            "hover_payload": None,
        }
        driver = self.input_driver
        reader = self.skill_bar_reader
        if driver is None or reader is None or not hasattr(reader, "slot_center_px"):
            self._emit(f"ALT_HOVER FAILED '{key}' (no driver/reader)")
            self._audit(result)
            return result

        point = None
        try:
            point = reader.slot_center_px(frame, key)
        except Exception as exc:
            logger.debug("alt hover slot_center_px: %s", exc)
        if point is None:
            self._emit(f"ALT_HOVER FAILED '{key}' (no slot px)")
            self._audit(result)
            return result

        x, y = int(point[0]), int(point[1])
        result["x"], result["y"] = x, y
        self.begin_exclusive(reason=f"alt_hover:{key}")
        alt_down = False
        up = getattr(driver, "_hw_key_up", None)
        try:
            self._force_all_mouse_up()
            # Show OS/game cursor (AION2 free-look → cursor mode)
            down = getattr(driver, "_hw_key_down", None)
            if not callable(down) or not callable(up):
                self._emit(f"ALT_HOVER FAILED '{key}' (no key hold API)")
                self._audit(result)
                return result
            down("alt", False)
            alt_down = True
            time.sleep(0.08)
            if hasattr(driver, "_hw_move_to"):
                driver._hw_move_to(x, y)
            elif hasattr(driver, "smooth_move_to"):
                driver.smooth_move_to(x, y)
            time.sleep(max(0.25, float(hold_sec)))
            if callable(on_hover):
                try:
                    result["hover_payload"] = on_hover(x, y)
                except Exception as exc:
                    logger.debug("on_hover callback failed: %s", exc)
            result["ok"] = True
            self._emit(f"ALT_HOVER '{key}' @ ({x},{y})")
            self._audit(result)
            return result
        except Exception as exc:
            logger.error("alt hover failed: %s", exc)
            self._emit(f"ALT_HOVER FAILED '{key}'")
            self._audit(result)
            return result
        finally:
            try:
                if alt_down and callable(up):
                    up("alt", False)
            except Exception:
                pass
            self._force_all_mouse_up()
            time.sleep(0.05)
            self.end_exclusive(restore_attack=False)

    def align_to_gcd(
        self,
        burst_duration_sec: float,
        gcd_sec: float = 0.12,
        min_gap_sec: float = 0.020,
    ) -> float:
        """Align next action with GCD boundary using Gaussian jitter."""
        driver = self.input_driver
        if driver is not None and hasattr(driver, "align_to_gcd"):
            return driver.align_to_gcd(
                burst_duration_sec, gcd_sec=gcd_sec, min_gap_sec=min_gap_sec
            )
        clean_gcd = max(0.05, float(gcd_sec))
        clean_dur = max(0.0, float(burst_duration_sec))
        clean_min = max(0.005, float(min_gap_sec))
        rem = clean_gcd - (clean_dur % clean_gcd) if clean_gcd > 0 else clean_min
        if rem < clean_min:
            rem += clean_gcd
        jitter = random.gauss(0.0, 0.004)
        return max(clean_min, rem + jitter)

    def sleep_gcd_align(
        self,
        burst_duration_sec: float,
        gcd_sec: float = 0.12,
        min_gap_sec: float = 0.020,
    ) -> float:
        """Sleep remaining duration to align with GCD with Gaussian jitter."""
        driver = self.input_driver
        if driver is not None and hasattr(driver, "sleep_gcd_align"):
            return driver.sleep_gcd_align(
                burst_duration_sec, gcd_sec=gcd_sec, min_gap_sec=min_gap_sec
            )
        delay = self.align_to_gcd(
            burst_duration_sec, gcd_sec=gcd_sec, min_gap_sec=min_gap_sec
        )
        time.sleep(delay)
        return delay

    def anti_stuck_guard(self, reason: str = "state_transition") -> None:
        """
        Anti-stuck key release guard on state transitions:
        Releases any held attack, movement keys, skill keys, modifier keys, and mouse buttons.
        """
        try:
            self.release_left_attack()
            self.hold_basic_attack(False)
            with self._lock:
                self._exclusive_depth = 0
                self._exclusive_reason = ""
            driver = self.input_driver
            if driver is not None:
                if hasattr(driver, "release_all_held_keys"):
                    driver.release_all_held_keys(reason=reason)
                else:
                    if hasattr(driver, "release_all_movement_keys"):
                        driver.release_all_movement_keys()
                    if hasattr(driver, "release_modifier_keys"):
                        driver.release_modifier_keys()
                    if hasattr(driver, "_release_mouse_buttons"):
                        driver._release_mouse_buttons()
            logger.info(
                "[GameActionBus] [ANTI-STUCK GUARD] State transition anti-stuck release completed (reason='%s')",
                reason,
            )
        except Exception as exc:
            logger.debug("anti_stuck_guard error: %s", exc)

