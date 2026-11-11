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
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("Gatherion.GameActionBus")

CLOSE_INTERACT_ROI = (0.42, 0.38, 0.69, 0.66)


class GameActionBus:
    """Exclusive lock around UI clicks so combat right-hold is not starved."""

    # AION2 mouse-look: no cursor; mouse motion = camera. Skills/Tab must be keys only.
    # Instance attribute (set in __init__): a class attribute here would be shared
    # across all buses and a single assignment would flip every instance.
    mouse_look_mode: bool

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
        self._exclusive_lock = threading.RLock()
        self._exclusive_owner: Optional[int] = None
        self._exclusive_acquires = 0
        self.mouse_look_mode = True if mouse_look_mode is None else bool(mouse_look_mode)

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
            except Exception as exc:
                logger.debug("on_log callback failed: %s", exc)

    @staticmethod
    def _sample_jitter(
        timing: Any,
        base_sec: float,
        std_sec: float,
        min_sec: float,
        max_sec: float,
        fallback_min_sec: float,
        fallback_max_sec: float,
    ) -> float:
        """Gaussian jitter from the driver's human-timing sampler, else uniform.

        Single definition of the sample-or-fallback branch that cast_skill
        (settle delay) and _key_fallback (key hold) each inlined with their
        own fallback range. Fallback bounds are parameters, so both callers
        keep their exact old fallback windows — no timing behaviour changes.
        """
        if timing is not None and hasattr(timing, "sample_gaussian_jitter"):
            return timing.sample_gaussian_jitter(
                base_sec=base_sec,
                std_sec=std_sec,
                min_sec=min_sec,
                max_sec=max_sec,
            )
        return random.uniform(fallback_min_sec, fallback_max_sec)

    def _mark_slot_pressed(self, key: str) -> None:
        """Best-effort skill-bar bookkeeping (was copy-pasted 4x in cast_skill)."""
        reader = self.skill_bar_reader
        if reader is not None and hasattr(reader, "mark_slot_pressed"):
            try:
                reader.mark_slot_pressed(key)
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
        # Real mutual exclusion (not just a counter): block other threads until
        # the matching end_exclusive. RLock so the same thread may nest pairs;
        # one acquire per begin, one release per end.
        self._exclusive_lock.acquire()
        try:
            with self._lock:
                self._exclusive_depth += 1
                self._exclusive_acquires += 1
                if self._exclusive_acquires == 1:
                    self._exclusive_owner = threading.get_ident()
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
        except Exception:
            # Keep acquire/release balanced if the side effects raise.
            with self._lock:
                self._exclusive_depth = max(0, self._exclusive_depth - 1)
                self._exclusive_acquires = max(0, self._exclusive_acquires - 1)
                if self._exclusive_acquires == 0:
                    self._exclusive_owner = None
            try:
                self._exclusive_lock.release()
            except RuntimeError:
                pass
            raise

    def end_exclusive(self, restore_attack: bool = True) -> None:
        release = False
        with self._lock:
            if self._exclusive_depth <= 0:
                return
            self._exclusive_depth -= 1
            self._exclusive_acquires = max(0, self._exclusive_acquires - 1)
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
                if self._exclusive_acquires == 0:
                    self._exclusive_owner = None
            release = True
        if release:
            try:
                self._exclusive_lock.release()
            except RuntimeError:
                logger.debug("end_exclusive lock release without matching acquire")

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
        """Historical API: T-key basic attack hold (right-click attack = T) via combat callbacks."""
        if on and callable(self._restore_basic_attack):
            self._restore_basic_attack()
        elif not on and callable(self._release_basic_attack):
            self._release_basic_attack()

    def hold_left_attack(self, on: bool) -> None:
        """Mana regen phase: hold/release R key as basic attack (left-click attack = R)."""
        want = bool(on)
        if want == self._left_held:
            return
        driver = self.input_driver
        try:
            if driver is None:
                return
            if want:
                down = getattr(driver, "key_down", None)
                if callable(down) and down("r"):
                    self._left_held = True
                    self._audit({"ok": True, "method": "left_hold_key_r", "on": True})
                    return
                # Fallback: press_key pulse (routes through the hardware-key gate).
                press = getattr(driver, "press_key", None)
                if callable(press):
                    sent = press("r", min_hold=0.08, max_hold=0.12)
                    # press_key returns the hold seconds (0.0 = aborted, nothing
                    # sent); only a None return (legacy mocks) carries no signal.
                    if sent is not False and sent != 0.0:
                        self._left_held = True
                        self._audit({"ok": True, "method": "left_hold_key_r_pulse", "on": True})
                    else:
                        logger.debug("hold_left_attack press_key pulse failed")
                    return
            else:
                up = getattr(driver, "key_up", None)
                if callable(up):
                    released = up("r")
                    if released is False:
                        # Key may still be held — keep _left_held so the next
                        # release attempt retries instead of assuming it is up.
                        logger.debug("hold_left_attack key_up('r') reported failure")
                        return
                    self._left_held = False
                    self._audit({"ok": True, "method": "left_hold_key_r", "on": False})
                    return
        except Exception as exc:
            logger.debug("hold_left_attack failed: %s", exc)

    def release_left_attack(self) -> None:
        self.hold_left_attack(False)

    def bind_input_port(self, port: Any) -> None:
        self.input_port = port

    def _interpret_click_result(self, clicked: Any, x: int, y: int) -> bool:
        """Honest success check for a driver click return (F-1868).

        The real driver ``human_click``/``click_map`` return
        ``(applied_x, applied_y, hold_sec)`` on EVERY path — including
        motion-abort and internal-exception paths — so the old
        ``clicked is False`` check was dead code and a bare tuple/None was
        counted as success. Bool-returning drivers/ports keep their
        meaning; a tuple is accepted only when it is well-formed, carries
        a positive hold, and reports the cursor at (jittered) target —
        the exception path reports the *actual* cursor position instead,
        which fails the tolerance check. A click aborted mid-way
        (``motion_aborted()``) is a failure even with a plausible tuple.
        Anything else (None, malformed value) is NOT success: no signal
        means no success claim.
        """
        if clicked is True:
            return True
        if clicked is False or clicked is None:
            return False
        if isinstance(clicked, (tuple, list)) and len(clicked) == 3:
            ax, ay, hold = clicked
            if (
                isinstance(ax, (int, float))
                and not isinstance(ax, bool)
                and isinstance(ay, (int, float))
                and not isinstance(ay, bool)
                and isinstance(hold, (int, float))
                and not isinstance(hold, bool)
                and float(hold) > 0.0
                # Driver jitter is +/-2px (DEFAULT_JITTER_PX); 8px covers
                # jitter + rounding while still catching the exception
                # path, which reports the real cursor position.
                and abs(float(ax) - float(x)) <= 8
                and abs(float(ay) - float(y)) <= 8
            ):
                driver = self.input_driver
                aborted = getattr(driver, "motion_aborted", None)
                if callable(aborted):
                    try:
                        if bool(aborted()):
                            logger.debug(
                                "click at (%s,%s) aborted mid-way; not success", x, y
                            )
                            return False
                    except Exception:
                        pass
                return True
            logger.debug(
                "click at (%s,%s) returned unusable tuple %r; not success", x, y, clicked
            )
            return False
        logger.debug(
            "click at (%s,%s) returned %r (no success signal); not success",
            x,
            y,
            clicked,
        )
        return False

    def _click_px(self, x: int, y: int) -> bool:
        # Never drag with a stuck RMB — release first, then move.
        self._force_all_mouse_up()
        port = self.input_port
        if port is not None and hasattr(port, "click"):
            try:
                # S-1501: port.click sonucu da doğrulanır — eskiden port
                # True dönerse koşulsuz başarı sayılıyordu; SoftInputAdapter
                # human_click tuple'unu atıp her zaman True döndüğü için
                # abort edilmiş / başarısız tık ok=True sayılıyordu. F-1868
                # doğrulaması (_interpret_click_result) port yoluna da
                # uygulanır; doğrulanamayan tık driver yoluna düşer.
                clicked = port.click(int(x), int(y), button="left")
                if self._interpret_click_result(clicked, int(x), int(y)):
                    return True
                logger.debug(
                    "input_port click at (%s,%s) not verified; falling through to driver",
                    x,
                    y,
                )
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
                clicked = driver.human_click(int(x), int(y), button="left")
                self._force_all_mouse_up()
                # human_click reports (applied_x, applied_y, hold_sec) even
                # on abort/exception paths — validate it, never assume True.
                return self._interpret_click_result(clicked, int(x), int(y))
            if hasattr(driver, "click_map"):
                clicked = driver.click_map(int(x), int(y), button="left")
                self._force_all_mouse_up()
                return self._interpret_click_result(clicked, int(x), int(y))
            if hasattr(driver, "_hw_move_to") and hasattr(driver, "_hw_mouse_down"):
                # P0-5: taşıma reddedildiyse kör tık atılmaz — down eski
                # konumda isabet edip "tıklandı" sayılırdı.
                if driver._hw_move_to(int(x), int(y)) is False:
                    return False
                down_ok = driver._hw_mouse_down("left")
                time.sleep(0.05)
                up_ok = driver._hw_mouse_up("left")
                if down_ok is False or up_ok is False:
                    return False
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

    def _key_fallback(
        self,
        key: str,
        min_hold: float = 0.08,
        max_hold: float = 0.12,
        trusted: Optional[bool] = None,
    ) -> bool:
        driver = self.input_driver
        if driver is None or not hasattr(driver, "press_key"):
            return False
        key_l = str(key).lower().strip()
        # F-1880: callers that already probed trust (cast_skill,
        # press_tab_target) pass it in so this does NOT re-run
        # ensure_serial_hid + keyboard_inject_trusted on every cast.
        # _keyboard_trusted() re-runs ensure_serial_hid() on every call,
        # so one cast used to probe twice. None = probe here (gather/cursor
        # paths with no prior probe). The R/T mapping and the hard HID gate
        # are untouched — this only avoids a redundant re-probe.
        if trusted is None:
            trusted = self._keyboard_trusted()
        else:
            trusted = bool(trusted)
        try:
            # AION2: stuck Alt = cursor mode; Tab/skills ignored. Clear mods first.
            # Do NOT Esc before Tab/skills — Esc clears the current target.
            if hasattr(driver, "release_modifier_keys"):
                try:
                    driver.release_modifier_keys()
                except Exception:
                    pass
            if hasattr(driver, "focus_game_window"):
                # v20/madde 2 (madde 3'e not): odak sonucu yutulmayacak —
                # oyun penceresi odakta değilken HID tuşu öndeki pencereye
                # gider, oyun kayda geçirmez; logda "SKILL KEY" yine de
                # ok=True görünür. Yalnızca loglanır, odak çalınmaz.
                focus_ok: Optional[bool] = None
                try:
                    focus_ok = bool(driver.focus_game_window())
                except Exception:
                    pass
                self._last_key_focus_ok = focus_ok
                if (
                    focus_ok is False
                    and sys.platform.startswith("win")
                    and not bool(getattr(driver, "test_mode", False))
                ):
                    now_w = time.monotonic()
                    if (now_w - float(getattr(self, "_last_focus_warn_ts", 0.0) or 0.0)) >= 5.0:
                        self._last_focus_warn_ts = now_w
                        logger.warning(
                            "[GameActionBus] key '%s' sent while game window NOT focused "
                            "(focus_game_window False) — game may ignore it",
                            key_l,
                        )
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
                hold = self._sample_jitter(
                    timing,
                    base_sec=hold_base,
                    std_sec=0.003,
                    min_sec=max(0.020, hold_base - 0.006),
                    max_sec=hold_base + 0.010,
                    fallback_min_sec=hold_base - 0.003,
                    fallback_max_sec=hold_base + 0.005,
                )
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

            # NOTE (W3, fix/v5-W3-misc): the old "using soft key dispatch
            # fallback" wording was misleading. There is no soft dispatch
            # here: press_key() below fail-closes — it returns 0.0 and
            # sends nothing when no trusted path exists (see
            # core/input_driver.py: press_key "güvenilir yol yok" /
            # "tuş gönderilmedi"). Log text only; no behaviour change.
            logger.warning(
                "[GameActionBus] [OPSEC FALLBACK] Hardware-trusted keyboard (Interception/HID) unavailable — soft key dispatch is DISABLED, no soft fallback is sent for '%s' (press_key below fail-closes with 0.0 when no trusted path exists)",
                key_l,
            )
            self._emit(f"[OPSEC FALLBACK] Soft key dispatch DISABLED for '{key_l}' (no soft fallback sent)")
            # Pass the caller's hold through: the old max(0.12, ...)/max(0.20, ...)
            # inflated every soft press to >=120ms and halved the cast rate.
            sent = driver.press_key(
                key_l,
                min_hold=float(min_hold),
                max_hold=float(max_hold),
                release_mouse=True,
                post_to_window=False,
            )
            # press_key returns the hold seconds; 0.0 means the press was
            # aborted and nothing was sent. Never report that as success.
            sent_ok = sent is not False and sent != 0.0
            if trusted:
                return bool(sent_ok)
            if not sent_ok:
                return False
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
                sent = driver.press_key(key_l, min_hold=min_hold, max_hold=max_hold, release_mouse=True)
                return sent is not False and sent != 0.0
            except Exception as exc:
                logger.debug("key fallback failed (%s): %s", key, exc)
                return False
        except Exception as exc:
            logger.debug("key fallback failed (%s): %s", key, exc)
            return False

    def cast_skill(self, slot_key: str, frame: Optional[Any] = None) -> Dict[str, Any]:
        """Cast skill. AION2 mouse-look: keyboard only (mouse move = camera)."""
        key = str(slot_key or "1").strip()
        if key.lower() in ("x", "f6", "b"):
            logger.debug("[AutoGate] GameActionBus suppressed automatic slot '%s'", key)
            return {
                "ok": False,
                "method": "suppressed_automatic",
                "slot": key,
                "x": None,
                "y": None,
                "burst_duration_sec": 0.0,
                "gcd_align_sec": 0.0,
            }
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
            settle_delay = self._sample_jitter(
                timing,
                base_sec=0.008 if trusted else 0.038,
                std_sec=0.0015 if trusted else 0.005,
                min_sec=0.004 if trusted else 0.025,
                max_sec=0.014 if trusted else 0.055,
                fallback_min_sec=0.006 if trusted else 0.030,
                fallback_max_sec=0.011 if trusted else 0.048,
            )
            time.sleep(settle_delay)
            # Mouse-look (AION2): trusted HID keyboard first; else Alt+skill-bar click.
            # Soft SendInput must NOT short-circuit — it logs success while Raw Input drops keys.
            # v20/madde 2: 0.03-0.05 sn basım logda "SKILL KEY" olarak görünüyor
            # ama oyunun giriş yoklamasına takılmayıp kayda geçmeyebiliyor
            # ("2 basıyor 2 basmıyor"). Güvenli alt sınır: insan basımı
            # ölçeğinde 0.08-0.12 sn (_key_fallback varsayılanıyla aynı).
            if self.mouse_look_mode:
                if trusted and self._key_fallback(key, min_hold=0.08, max_hold=0.12, trusted=trusted):
                    burst_dur = time.perf_counter() - cast_start_ts
                    try:
                        # O-4 (dilim 20): GCD hizalaması eskiden yalnızca
                        # hesaplanıp kayda yazılıyor, uyunmuyordu — skill
                        # GCD'den hızlı basılabiliyordu. Artık kalan süre
                        # gerçekten beklenir (abort-aware uyku).
                        gcd_rem = self.sleep_gcd_align(burst_dur, gcd_sec=0.12)
                    except Exception as exc:
                        # The key was already sent — a GCD helper failure must
                        # not lose the cast or its audit record.
                        logger.debug("cast_skill sleep_gcd_align failed: %s", exc)
                        gcd_rem = 0.0
                    result["ok"] = True
                    result["method"] = "key"
                    result["burst_duration_sec"] = burst_dur
                    result["gcd_align_sec"] = gcd_rem
                    result["trusted"] = True
                    msg = f"SKILL KEY '{key}'"
                    self._emit(msg)
                    self._audit(result)
                    self._mark_slot_pressed(key)
                    return result
                # Trust was already checked once above and is passed through
                # to _key_fallback (trusted=...) — no re-probe of the driver
                # (ensure_serial_hid + trusted check) mid-cast (F-1880).
                if not trusted:
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
                        # Alt+click releases the held attack key; restore R hold in mouse-look combat.
                        if self._left_held:
                            try:
                                down = getattr(driver, "key_down", None)
                                if callable(down):
                                    down("r")
                            except Exception:
                                pass
                        # Sticky T must stay clear after leaving Alt cursor mode.
                        try:
                            up = getattr(driver, "key_up", None)
                            if callable(up):
                                up("t")
                        except Exception:
                            pass
                        self._audit(result)
                        self._mark_slot_pressed(key)
                        return result
                # Last resort: R-key basic attack pulse (left-click attack = R) — does not cast hotbar skills.
                if driver is not None and hasattr(driver, "key_down"):
                    try:
                        key_up = getattr(driver, "key_up", None)
                        if callable(key_up):
                            key_up("t")
                        ok_down = bool(driver.key_down("r"))
                        time.sleep(0.35)
                        # Keep R held if combat already requested AA; else pulse only.
                        if not self._left_held and callable(key_up):
                            key_up("r")
                        # The requested skill was NOT cast — report failure even
                        # though a basic-attack pulse went out (old code returned
                        # ok=True here, a fake skill-cast success).
                        result["ok"] = False
                        result["method"] = "key_basic" if ok_down else "failed"
                        self._emit(f"SKILL KEY_BASIC '{key}' (skill NOT cast; basic-attack pulse only)")
                        self._audit(result)
                        return result
                    except Exception as exc:
                        logger.debug("key basic attack failed: %s", exc)
                result["method"] = "failed"
                self._emit(f"SKILL FAILED '{key}' (mouse-look key)")
                self._audit(result)
                return result

            # Legacy cursor mode: trusted hardware key first; UI click only when
            # keys are untrusted (cursor mode exists for exactly that case).
            restore_attack = True
            point = None
            reader = self.skill_bar_reader
            if trusted and self._key_fallback(key, trusted=trusted):
                # O-4 (dilim 20): mouse-look dalındaki gibi GCD hizalaması
                # burada da gerçekten beklenir.
                burst_dur = time.perf_counter() - cast_start_ts
                try:
                    gcd_rem = self.sleep_gcd_align(burst_dur, gcd_sec=0.12)
                except Exception as exc:
                    logger.debug("cast_skill sleep_gcd_align failed: %s", exc)
                    gcd_rem = 0.0
                result["ok"] = True
                result["method"] = "key"
                result["burst_duration_sec"] = burst_dur
                result["gcd_align_sec"] = gcd_rem
                msg = f"SKILL KEY '{key}' (cursor mode, trusted)"
                self._emit(msg)
                self._audit(result)
                self._mark_slot_pressed(key)
                return result
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
                    self._mark_slot_pressed(key)
                    return result
            # F-1880: trust was probed once at cast start — pass it through
            # instead of re-probing inside _key_fallback.
            if self._key_fallback(key, trusted=trusted):
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
        prompt_checked = False
        prompt_found = False
        if frame is not None and vision is not None and hasattr(vision, "detect_gather_prompt"):
            try:
                found, _conf, bbox = vision.detect_gather_prompt(frame=frame)
                prompt_checked = True
                prompt_found = bool(found)
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

        if prompt_checked and not prompt_found:
            # Vision explicitly reports no gather prompt — do NOT blind-click
            # the ROI centre; the F/E key fallback handles it.
            return None

        # O8 (dilim 15): vision nesnesi yok / detect_gather_prompt
        # çağrılamadıysa prompt doğrulanmış sayılmaz — ROI merkezine kör
        # tıklama YOK. gather_interact F/E tuş fallback'ine düşer (Aion 2'de
        # toplama zaten F tuşudur; prompt'suz tıklama yanlış UI etkileşimi
        # / hedef değiştirme riskiydi). CLOSE_INTERACT_ROI sabiti referans
        # için tutulur; kör tıklama dalı kaldırıldı.
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
            # F-1880: probe trust once and pass it through — _key_fallback
            # must not re-run ensure_serial_hid + keyboard_inject_trusted.
            tab_trusted = self._keyboard_trusted()
            if tab_trusted:
                result["ok"] = self._key_fallback(
                    "tab", min_hold=0.10, max_hold=0.16, trusted=tab_trusted
                )
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
                        aimed = soft(frame=frame, restore_lmb=True)
                        # soft_aim_click returns False when it could not aim; a
                        # None return (legacy mocks) carries no signal. Never
                        # assume the R-key AA hold was restored on failure.
                        if aimed is False:
                            logger.debug("press_tab_target soft_aim reported failure")
                        else:
                            result["ok"] = True
                            result["method"] = "tab+soft_aim" if keyed else "soft_aim"
                            self._emit("TAB+SOFT_AIM" if keyed else "SOFT_AIM ONLY")
                            # soft_aim(restore_lmb=True) re-holds the R-key AA
                            # only when it actually ran — never assume it.
                            self._left_held = True
                    except TypeError:
                        try:
                            aimed = soft(frame=frame)
                            if aimed is False:
                                logger.debug("press_tab_target soft_aim reported failure")
                            else:
                                result["ok"] = True
                                result["method"] = "soft_aim"
                                self._emit("SOFT_AIM ONLY")
                                self._left_held = True
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
            if down("alt", False) is False:
                # Alt never engaged — no cursor, no hover; fail honestly
                # instead of moving the camera and reporting success.
                self._emit(f"ALT_HOVER FAILED '{key}' (Alt press rejected)")
                self._audit(result)
                return result
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
                    if up("alt", False) is False:
                        logger.warning("ALT_HOVER Alt release rejected for '%s'", key)
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
            try:
                return driver.align_to_gcd(
                    burst_duration_sec, gcd_sec=gcd_sec, min_gap_sec=min_gap_sec
                )
            except Exception as exc:
                # A GCD helper failure must never crash the caller (cast_skill
                # already sent the key) — fall back to the local computation.
                logger.debug("align_to_gcd driver delegation failed, using local fallback: %s", exc)
        try:
            clean_gcd = max(0.05, float(gcd_sec))
            clean_dur = max(0.0, float(burst_duration_sec))
            clean_min = max(0.005, float(min_gap_sec))
        except (TypeError, ValueError):
            return 0.020
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
            pending_releases = 0
            with self._lock:
                if self._exclusive_owner is not None and self._exclusive_owner != threading.get_ident():
                    # Another thread owns the in-flight exclusive section; do
                    # not corrupt its depth/lock balance — it will end it.
                    logger.debug("anti_stuck_guard: exclusive section owned by another thread; leaving lock alone")
                else:
                    pending_releases = self._exclusive_acquires
                    self._exclusive_depth = 0
                    self._exclusive_acquires = 0
                    self._exclusive_owner = None
                    self._exclusive_reason = ""
            for _ in range(pending_releases):
                try:
                    self._exclusive_lock.release()
                except RuntimeError:
                    break
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

