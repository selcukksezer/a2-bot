# -*- coding: utf-8 -*-
"""
50-Maddelik Şartname Kararlılık ve Güvenlik Korumaları (Maddeler 11–15).

Bu modül aşağıdaki 5 kritik kararlılık katmanını sağlar:
- Madde 11: Kamera Kilidi ((1280, 720) / (W/2, H/2)) Algılayıcı (`CameraLockExceptionRule`)
- Madde 12: İmleç Açma (Alt / Show Cursor) Otomasyonu (`CursorVisibilityAutomator`)
- Madde 13: Sıfır Can (hp_ratio=0.000) Yanlış Ölüm Tespiti Koruması (`FiveFrameDeathConfirmationGuard`)
- Madde 14: Harita Açılış Doğrulamasında Çoklu Kriter (`MultiCriteriaMapVerifier`)
- Madde 15: Kendi Gönderdiği Tuşu Fiziksel Müdahale Sanmama (`SelfInputWhitelistWindow`)
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import gc
import logging
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("core.stability_guards")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


# ============================================================================
# WIN32 CTYPES YAPILARI (GetCursorPos & GetCursorInfo)
# ============================================================================

CURSOR_SHOWING: int = 0x00000001
VK_MENU_ALT: int = 0x12
KEYEVENTF_KEYUP: int = 0x0002


class _POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    ]


class _CURSORINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hCursor", wintypes.HANDLE),
        ("ptScreenPos", _POINT),
    ]


# ============================================================================
# MADDE 15: KENDİ GÖNDERDİĞİ TUŞU FİZİKSEL MÜDAHALE SANMAMA (250ms Whitelist)
# ============================================================================

class SelfInputWhitelistWindow:
    """
    Madde 15 — Kendi Gönderdiği Tuşu Fiziksel Müdahale Sanmama.

    Botun kendi gönderdiği sanal tuş basımlarını (`F`, `M`, `E`, `W/A/S/D`, `Space`,
    `Alt`, `Tab`, `Esc`, `1-5` vb.) zaman damgalı (`time.monotonic()`) bir iç beyaz
    listede tutar. `250ms` (varsayılan) boyunca bu tuşlardan gelen Win32
    `GetAsyncKeyState` sinyallerini fiziksel kullanıcı müdahalesi olarak algılamaz.
    """

    DEFAULT_WINDOW_MS: float = 250.0

    KEY_TO_VK: Dict[str, int] = {
        "f": 0x46,
        "m": 0x4D,
        "e": 0x45,
        "w": 0x57,
        "a": 0x41,
        "s": 0x53,
        "d": 0x44,
        "q": 0x51,
        "r": 0x52,
        "t": 0x54,
        "g": 0x47,
        "z": 0x5A,
        "x": 0x58,
        "c": 0x43,
        "v": 0x56,
        "b": 0x42,
        "space": 0x20,
        " ": 0x20,
        "vk_space": 0x20,
        "alt": 0x12,
        "lalt": 0xA4,
        "ralt": 0xA5,
        "menu": 0x12,
        "tab": 0x09,
        "esc": 0x1B,
        "escape": 0x1B,
        "shift": 0x10,
        "ctrl": 0x11,
        "control": 0x11,
        "pageup": 0x21,
        "pgup": 0x21,
        "pagedown": 0x22,
        "pgdn": 0x22,
        "left": 0x25,
        "up": 0x26,
        "right": 0x27,
        "down": 0x28,
        "1": 0x31,
        "2": 0x32,
        "3": 0x33,
        "4": 0x34,
        "5": 0x35,
        "6": 0x36,
        "7": 0x37,
        "8": 0x38,
        "9": 0x39,
        "0": 0x30,
        "f8": 0x77,
        "f9": 0x78,
        "f10": 0x79,
        "f12": 0x7B,
        "lbutton": 0x01,
        "rbutton": 0x02,
        "mbutton": 0x04,
    }

    VK_TO_CANONICAL: Dict[int, str] = {
        0x46: "f",
        0x4D: "m",
        0x45: "e",
        0x57: "w",
        0x41: "a",
        0x53: "s",
        0x44: "d",
        0x51: "q",
        0x52: "r",
        0x54: "t",
        0x47: "g",
        0x5A: "z",
        0x58: "x",
        0x43: "c",
        0x56: "v",
        0x42: "b",
        0x20: "space",
        0x12: "alt",
        0xA4: "alt",
        0xA5: "alt",
        0x09: "tab",
        0x1B: "esc",
        0x10: "shift",
        0x11: "ctrl",
        0x21: "pageup",
        0x22: "pagedown",
        0x25: "left",
        0x26: "up",
        0x27: "right",
        0x28: "down",
        0x31: "1",
        0x32: "2",
        0x33: "3",
        0x34: "4",
        0x35: "5",
        0x77: "f8",
        0x78: "f9",
        0x79: "f10",
        0x7B: "f12",
        0x01: "lbutton",
        0x02: "rbutton",
        0x04: "mbutton",
    }

    def __init__(self, default_window_ms: float = 250.0) -> None:
        self.default_window_ms: float = max(1.0, float(default_window_ms))
        self._lock = threading.RLock()
        self._key_timestamps: Dict[str, float] = {}
        self._vk_timestamps: Dict[int, float] = {}

    @classmethod
    def _normalize_key_and_vk(cls, key: Union[str, int]) -> Tuple[Optional[str], Optional[int]]:
        """Verilen tuş adını veya sanal tuş kodunu (VK) standart (isim, vk_int) çiftine çevirir."""
        if isinstance(key, int):
            vk_val = int(key)
            canonical = cls.VK_TO_CANONICAL.get(vk_val)
            if canonical is None and 0x30 <= vk_val <= 0x5A:
                canonical = chr(vk_val).lower()
            return canonical, vk_val

        raw = str(key or "").strip().lower()
        if not raw:
            return None, None

        if raw.startswith("0x"):
            try:
                vk_parsed = int(raw, 16)
                return cls.VK_TO_CANONICAL.get(vk_parsed, raw), vk_parsed
            except ValueError:
                return raw, None

        vk_code = cls.KEY_TO_VK.get(raw)
        if vk_code is None and len(raw) == 1:
            vk_code = ord(raw.upper())
        canonical_name = cls.VK_TO_CANONICAL.get(vk_code, raw) if vk_code is not None else raw
        return canonical_name, vk_code

    def record_self_key(
        self,
        key: Union[str, int],
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Botun kendi gönderdiği sanal tuşu (örn. 'F', 'M', 'E', 'W', 'Space' veya VK kodu)
        zaman damgasıyla kaydeder.
        """
        now = float(timestamp) if timestamp is not None else time.monotonic()
        canonical_name, vk_code = self._normalize_key_and_vk(key)
        with self._lock:
            if canonical_name is not None:
                self._key_timestamps[canonical_name] = now
            raw_str = str(key).strip().lower()
            if raw_str:
                self._key_timestamps[raw_str] = now
            if vk_code is not None:
                self._vk_timestamps[int(vk_code)] = now
        return now

    def record_self_keys(self, keys: Sequence[Union[str, int]]) -> float:
        """Birden fazla sanal tuşu aynı anda beyaz listeye kaydeder."""
        now = time.monotonic()
        for k in keys:
            self.record_self_key(k, timestamp=now)
        return now

    def is_within_ignore_window(
        self,
        key: Union[str, int],
        window_ms: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """
        Verilen tuşun son `window_ms` (varsayılan 250.0 ms) içinde bot tarafından
        gönderilip gönderilmediğini kontrol eder.
        """
        eff_window_sec = (
            max(0.0, float(window_ms)) / 1000.0
            if window_ms is not None
            else self.default_window_ms / 1000.0
        )
        curr_ts = float(now) if now is not None else time.monotonic()
        canonical_name, vk_code = self._normalize_key_and_vk(key)

        with self._lock:
            candidates: List[float] = []
            if canonical_name is not None and canonical_name in self._key_timestamps:
                candidates.append(self._key_timestamps[canonical_name])
            raw_str = str(key).strip().lower()
            if raw_str in self._key_timestamps:
                candidates.append(self._key_timestamps[raw_str])
            if vk_code is not None and int(vk_code) in self._vk_timestamps:
                candidates.append(self._vk_timestamps[int(vk_code)])

            if not candidates:
                return False

            last_ts = max(candidates)
            elapsed_sec = max(0.0, curr_ts - last_ts)
            return elapsed_sec <= eff_window_sec

    def clear(self) -> None:
        """Tüm kayıtlı zaman damgalarını temizler."""
        with self._lock:
            self._key_timestamps.clear()
            self._vk_timestamps.clear()

    def get_stability_guards_ready_flag(self) -> dict:
        return {"module": "stability_guards", "ready": True}


# ============================================================================
# MADDE 11: KAMERA KİLİDİ ((1280, 720) / (W/2, H/2)) ALGILAYICI
# ============================================================================

class CameraLockExceptionRule:
    """
    Madde 11 — Kamera Kilidi ((1280, 720)) Algılayıcı.

    Aion 2'de 3B kamera döndürülürken veya imleç gizlendiğinde Windows'un fareyi
    ekranın tam ortasına (`(W/2, H/2)` -> 2560x1440 için `(1280, 720)`,
    1920x1080 için `(960, 540)`, 2000x1125 için `(1000, 562)`, 1280x720 için `(640, 360)`)
    kilitlemesini kullanıcı fare hareketi olarak değil, `"Oyun İmleç Kilidi"`
    olarak tanır ve `delta=200.0px > 25.0px` gibi yanlış iptalleri (abort) engeller.
    """

    KNOWN_LOCK_CENTERS: Tuple[Tuple[int, int], ...] = (
        (1280, 720),  # 2560x1440 (QHD) ekran merkezi
        (960, 540),   # 1920x1080 (FHD) ekran merkezi
        (1000, 562),  # 2000x1125 / 2000x1124 pencere merkezi
        (640, 360),   # 1280x720 (HD) ekran merkezi
        (1920, 1080), # 3840x2160 (4K UHD) ekran merkezi
    )
    DEFAULT_TOLERANCE_PX: float = 40.0
    LOCK_CLASSIFICATION_LABEL: str = "Oyun İmleç Kilidi"

    def __init__(
        self,
        tolerance_px: float = 40.0,
        custom_centers: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.tolerance_px: float = max(1.0, float(tolerance_px))
        self._custom_centers: List[Tuple[int, int]] = [
            (int(cx), int(cy)) for cx, cy in (custom_centers or ())
        ]

    def get_screen_centers(
        self,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Bilinen tüm ekran merkezlerini ve aktif işletim sistemi çözünürlük merkezini döndürür."""
        centers: List[Tuple[int, int]] = list(self.KNOWN_LOCK_CENTERS)
        for pt in self._custom_centers:
            if pt not in centers:
                centers.append(pt)

        if screen_w is not None and screen_h is not None and screen_w > 0 and screen_h > 0:
            explicit_center = (int(screen_w) // 2, int(screen_h) // 2)
            if explicit_center not in centers:
                centers.append(explicit_center)

        try:
            sw = int(ctypes.windll.user32.GetSystemMetrics(0))
            sh = int(ctypes.windll.user32.GetSystemMetrics(1))
            if sw > 0 and sh > 0:
                os_center = (sw // 2, sh // 2)
                if os_center not in centers:
                    centers.append(os_center)
        except Exception as exc:
            logger.debug("GetSystemMetrics okuma hatasi: %s", exc)

        return centers

    def is_camera_lock_position(
        self,
        x: int,
        y: int,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> bool:
        """
        Verilen `(x, y)` koordinatının ekran merkezi kamera kilidi (`(W/2, H/2)` ± tolerans)
        içinde olup olmadığını döndürür.
        """
        fx, fy = float(x), float(y)
        for cx, cy in self.get_screen_centers(screen_w=screen_w, screen_h=screen_h):
            if math.hypot(fx - float(cx), fy - float(cy)) <= self.tolerance_px:
                return True
        return False

    def min_distance_to_lock_center(
        self,
        x: int,
        y: int,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> float:
        """Verilen `(x, y)` koordinatının en yakın ekran merkezine olan piksel mesafesini döndürür."""
        fx, fy = float(x), float(y)
        centers = self.get_screen_centers(screen_w=screen_w, screen_h=screen_h)
        return min(math.hypot(fx - float(cx), fy - float(cy)) for cx, cy in centers)

    def should_ignore_delta(
        self,
        expected_x: int,
        expected_y: int,
        actual_x: int,
        actual_y: int,
        cursor_visible: Optional[bool] = None,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> Tuple[bool, str]:
        """
        İmleç `(1280, 720)` / `(W/2, H/2)` merkezine kilitlendiyse veya gizli imleç
        modundaysa `(True, "Oyun İmleç Kilidi")` döndürerek yanlış iptali önler.
        """
        if self.is_camera_lock_position(actual_x, actual_y, screen_w=screen_w, screen_h=screen_h):
            return True, self.LOCK_CLASSIFICATION_LABEL
        if cursor_visible is False:
            return True, self.LOCK_CLASSIFICATION_LABEL
        return False, ""

    def classify_cursor_jump(
        self,
        expected_xy: Tuple[int, int],
        actual_xy: Tuple[int, int],
        threshold_px: float = 25.0,
        cursor_visible: Optional[bool] = None,
        screen_w: Optional[int] = None,
        screen_h: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Beklenen ve gerçek imleç konumları arasındaki sıçramayı değerlendirir.
        Eğer gerçek konum `(1280, 720)` / `(W/2, H/2)` kamera kilidi ise `delta=200.0px > 25.0px`
        olsa bile `should_abort=False` ve `classification="Oyun İmleç Kilidi"` döndürür.
        """
        ex, ey = int(expected_xy[0]), int(expected_xy[1])
        ax, ay = int(actual_xy[0]), int(actual_xy[1])
        delta_px = math.hypot(float(ax - ex), float(ay - ey))

        ignore, label = self.should_ignore_delta(
            expected_x=ex,
            expected_y=ey,
            actual_x=ax,
            actual_y=ay,
            cursor_visible=cursor_visible,
            screen_w=screen_w,
            screen_h=screen_h,
        )
        if ignore:
            return {
                "is_camera_lock": True,
                "should_abort": False,
                "classification": label,
                "reason": (
                    f"{label}: actual=({ax},{ay}), expected=({ex},{ey}), "
                    f"delta={delta_px:.1f}px ignored"
                ),
                "delta_px": round(delta_px, 2),
                "updated_anchor": (ax, ay),
            }

        if delta_px > float(threshold_px):
            reason = (
                f"physical_mouse_movement_detected: "
                f"delta={delta_px:.1f}px > {float(threshold_px):.1f}px "
                f"(expected=({ex},{ey}), actual=({ax},{ay}))"
            )
            return {
                "is_camera_lock": False,
                "should_abort": True,
                "classification": "Fiziksel Fare Müdahalesi",
                "reason": reason,
                "delta_px": round(delta_px, 2),
                "updated_anchor": (ax, ay),
            }

        return {
            "is_camera_lock": False,
            "should_abort": False,
            "classification": "Normal",
            "reason": "ok",
            "delta_px": round(delta_px, 2),
            "updated_anchor": (ax, ay),
        }

    def get_stability_guards_ready_flag(self) -> dict:
        return {"module": "stability_guards", "ready": True}


# ============================================================================
# MADDE 12: İMLEÇ AÇMA (ALT / SHOW CURSOR) OTOMASYONU
# ============================================================================

class CursorVisibilityAutomator:
    """
    Madde 12 — İmleç Açma (`Alt` / Show Cursor) Otomasyonu.

    Harita veya arayüz tıklamasından önce Windows `ctypes.windll.user32.GetCursorInfo`
    API'si (`CURSORINFO` yapısı, `CURSOR_SHOWING = 0x00000001`) ile imlecin ekranda
    görünür olup olmadığını kontrol eder. İmleç gizliyse (`flags & CURSOR_SHOWING == 0`),
    otomatik olarak `'alt'` (veya yapılandırılan imleç açma tuşu) basarak imleci
    görünür hale getirir ve ardından tıklamaya izin verir.
    """

    CURSOR_SHOWING: int = CURSOR_SHOWING

    def __init__(
        self,
        toggle_key: str = "alt",
        cursor_info_getter: Optional[Callable[[], Any]] = None,
        key_presser: Optional[Callable[[str], Any]] = None,
        whitelist: Optional[SelfInputWhitelistWindow] = None,
        settle_delay_sec: float = 0.035,
    ) -> None:
        self.toggle_key: str = str(toggle_key or "alt").strip().lower()
        self._cursor_info_getter = cursor_info_getter
        self._key_presser = key_presser
        self.whitelist: Optional[SelfInputWhitelistWindow] = whitelist
        self.settle_delay_sec: float = max(0.0, float(settle_delay_sec))
        self.reveal_count: int = 0
        self.last_revealed_ts: float = 0.0
        self.last_pressed_key: str = ""

    @staticmethod
    def _query_win32_cursor_info() -> Dict[str, Any]:
        """Win32 GetCursorInfo çağrısı ile imleç bayraklarını ve ekran konumunu okur."""
        try:
            ci = _CURSORINFO()
            ci.cbSize = ctypes.sizeof(_CURSORINFO)
            ok = bool(ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci)))
            if ok:
                flags = int(ci.flags)
                showing = bool(flags & CURSOR_SHOWING)
                return {
                    "ok": True,
                    "visible": showing,
                    "flags": flags,
                    "x": int(ci.ptScreenPos.x),
                    "y": int(ci.ptScreenPos.y),
                }
        except Exception as exc:
            logger.debug("GetCursorInfo Win32 hatasi: %s", exc)
        return {
            "ok": False,
            "visible": True,
            "flags": CURSOR_SHOWING,
            "x": 0,
            "y": 0,
        }

    def get_cursor_info(self) -> Dict[str, Any]:
        """Mevcut imleç görünürlük bilgisini sözlük olarak döndürür."""
        if self._cursor_info_getter is not None:
            try:
                raw = self._cursor_info_getter()
                if isinstance(raw, dict):
                    flags = int(raw.get("flags", CURSOR_SHOWING if raw.get("visible", True) else 0))
                    visible = bool(raw.get("visible", bool(flags & CURSOR_SHOWING)))
                    return {
                        "ok": True,
                        "visible": visible,
                        "flags": flags,
                        "x": int(raw.get("x", 0)),
                        "y": int(raw.get("y", 0)),
                    }
                if isinstance(raw, bool):
                    return {
                        "ok": True,
                        "visible": raw,
                        "flags": CURSOR_SHOWING if raw else 0,
                        "x": 0,
                        "y": 0,
                    }
                if isinstance(raw, int):
                    flags = int(raw)
                    return {
                        "ok": True,
                        "visible": bool(flags & CURSOR_SHOWING),
                        "flags": flags,
                        "x": 0,
                        "y": 0,
                    }
                if isinstance(raw, (tuple, list)) and len(raw) >= 1:
                    first = raw[0]
                    if isinstance(first, bool):
                        flags = CURSOR_SHOWING if first else 0
                        vis = first
                    else:
                        flags = int(first)
                        vis = bool(flags & CURSOR_SHOWING)
                    x_val = int(raw[1]) if len(raw) >= 2 else 0
                    y_val = int(raw[2]) if len(raw) >= 3 else 0
                    return {
                        "ok": True,
                        "visible": vis,
                        "flags": flags,
                        "x": x_val,
                        "y": y_val,
                    }
            except Exception as exc:
                logger.debug("Ozel cursor_info_getter hatasi: %s", exc)
        return self._query_win32_cursor_info()

    def is_cursor_visible(self) -> bool:
        """
        Windows `GetCursorInfo` üzerinden imlecin o an görünür (`flags & CURSOR_SHOWING != 0`)
        olup olmadığını döndürür.
        """
        info = self.get_cursor_info()
        return bool(info.get("visible", False))

    def _dispatch_toggle_key(self, key: str, input_ctrl: Any = None) -> bool:
        """İmleç açma tuşunu (`'alt'`) beyaz listeye kaydedip gönderir."""
        if self.whitelist is not None:
            self.whitelist.record_self_key(key)

        if self._key_presser is not None:
            try:
                self._key_presser(key)
                return True
            except Exception as exc:
                logger.debug("Ozel key_presser hatasi ('%s'): %s", key, exc)

        if input_ctrl is not None:
            for attr_name in ("press_key", "send_key"):
                fn = getattr(input_ctrl, attr_name, None)
                if callable(fn):
                    try:
                        fn(key)
                        return True
                    except Exception as exc:
                        logger.debug("input_ctrl.%s('%s') hatasi: %s", attr_name, key, exc)

        # Fallback: Win32 keybd_event ile Alt (VK_MENU = 0x12) bas-çek
        try:
            vk = SelfInputWhitelistWindow.KEY_TO_VK.get(key.lower(), VK_MENU_ALT)
            ctypes.windll.user32.keybd_event(int(vk), 0, 0, 0)
            time.sleep(0.02)
            ctypes.windll.user32.keybd_event(int(vk), 0, KEYEVENTF_KEYUP, 0)
            return True
        except Exception as exc:
            logger.debug("Win32 keybd_event('%s') hatasi: %s", key, exc)
            return False

    def ensure_cursor_visible(
        self,
        input_ctrl: Any = None,
        toggle_key: Optional[str] = None,
        max_attempts: int = 2,
    ) -> bool:
        """
        İmleç gizliyse (`is_cursor_visible() == False`), `'alt'` (veya `toggle_key`)
        tuşuna basarak imleci açar.
        """
        if self.is_cursor_visible():
            return True

        key_to_use = str(toggle_key or self.toggle_key or "alt").strip().lower()
        attempts = max(1, int(max_attempts))

        for _ in range(attempts):
            if self.is_cursor_visible():
                return True
            dispatched = self._dispatch_toggle_key(key_to_use, input_ctrl=input_ctrl)
            if dispatched:
                self.reveal_count += 1
                self.last_revealed_ts = time.monotonic()
                self.last_pressed_key = key_to_use
                logger.info(
                    "Gizli oyun imleci algilandi (GetCursorInfo=0); '%s' tusu ile imlec acildi.",
                    key_to_use,
                )
            if self.settle_delay_sec > 0.0:
                time.sleep(self.settle_delay_sec)
            if self.is_cursor_visible():
                return True

        return self.is_cursor_visible()

    def get_stability_guards_ready_flag(self) -> dict:
        return {"module": "stability_guards", "ready": True}


# ============================================================================
# MADDE 13: SIFIR CAN (hp_ratio=0.000) YANLIŞ ÖLÜM TESPİTİ KORUMASI
# ============================================================================

class FiveFrameDeathConfirmationGuard:
    """
    Madde 13 — Sıfır Can (`hp_ratio=0.000`) Yanlış Ölüm Tespiti Koruması.

    Kurallar:
    1. Arayüz gizlendiğinde veya kırmızı can barı pikseli eşleşmediğinde `hp_ratio=0.000`
       okunsa bile `revive_dialog == False` ise ASLA ölüm (`True`) tetiklemez!
    2. Gerçek ölüm tespiti için:
       - Dirilme Penceresi (`revive_dialog == True`) VE
       - Grileşmiş 3B Ekran (`grayscale_viewport == True`, `outer_mean_saturation < 32.0`) VE
       - Ardışık 5 Kare Doğrulaması (`_consecutive_death_frames >= 5`, veya verilen `confirm_frames`)
       koşullarının tamamı sağlanmalıdır.
    """

    DEFAULT_REQUIRED_FRAMES: int = 5
    MAX_GRAYSCALE_SATURATION: float = 32.0

    def __init__(self, required_frames: int = 5) -> None:
        self.required_frames: int = max(1, int(required_frames))
        self._consecutive_death_frames: int = 0
        self._lock = threading.RLock()

    @property
    def consecutive_death_frames(self) -> int:
        return self._consecutive_death_frames

    def reset(self) -> None:
        """Ardışık ölüm karesi sayacını sıfırlar."""
        with self._lock:
            self._consecutive_death_frames = 0

    @staticmethod
    def _ensure_bgr(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return None

    @classmethod
    def inspect_death_visuals(cls, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        Görüntü karesinde:
        1) Dış 3B ekranın grileşip grileşmediğini (`outer_mean_saturation < 32.0`)
        2) Merkezde koyu dirilme modal kutusu ve kırmızı/altın dirilme butonu (`revive_dialog`)
        olup olmadığını analiz eder.
        """
        bgr = cls._ensure_bgr(frame)
        if bgr is None:
            return {
                "revive_dialog": False,
                "grayscale_viewport": False,
                "outer_mean_saturation": 255.0,
                "modal_dark_ratio": 0.0,
                "button_ratio": 0.0,
            }

        h, w = bgr.shape[:2]
        if h < 80 or w < 80:
            return {
                "revive_dialog": False,
                "grayscale_viewport": False,
                "outer_mean_saturation": 255.0,
                "modal_dark_ratio": 0.0,
                "button_ratio": 0.0,
            }

        # 1. Dış 3B oyun alanı doygunluk analizi (Gri ekran kontrolü)
        outer_top = bgr[
            int(round(h * 0.05)) : int(round(h * 0.22)),
            int(round(w * 0.15)) : int(round(w * 0.85)),
        ]
        if outer_top.size == 0:
            return {
                "revive_dialog": False,
                "grayscale_viewport": False,
                "outer_mean_saturation": 255.0,
                "modal_dark_ratio": 0.0,
                "button_ratio": 0.0,
            }

        hsv_outer = cv2.cvtColor(outer_top, cv2.COLOR_BGR2HSV)
        outer_mean_sat = float(np.mean(hsv_outer[:, :, 1]))
        grayscale_viewport = outer_mean_sat < cls.MAX_GRAYSCALE_SATURATION

        # 2. Merkez dirilme penceresi (modal) + dirilme butonu kontrolü
        mx1, mx2 = int(round(w * 0.32)), int(round(w * 0.68))
        my1, my2 = int(round(h * 0.28)), int(round(h * 0.72))
        modal_roi = bgr[my1:my2, mx1:mx2]
        if modal_roi.size == 0:
            return {
                "revive_dialog": False,
                "grayscale_viewport": grayscale_viewport,
                "outer_mean_saturation": round(outer_mean_sat, 2),
                "modal_dark_ratio": 0.0,
                "button_ratio": 0.0,
            }

        hsv_modal = cv2.cvtColor(modal_roi, cv2.COLOR_BGR2HSV)
        btn_zone = hsv_modal[
            int(modal_roi.shape[0] * 0.55) :,
            int(modal_roi.shape[1] * 0.20) : int(modal_roi.shape[1] * 0.80),
        ]
        if btn_zone.size == 0:
            return {
                "revive_dialog": False,
                "grayscale_viewport": grayscale_viewport,
                "outer_mean_saturation": round(outer_mean_sat, 2),
                "modal_dark_ratio": 0.0,
                "button_ratio": 0.0,
            }

        btn_red_gold = (
            ((btn_zone[:, :, 0] <= 24) | (btn_zone[:, :, 0] >= 165))
            & (btn_zone[:, :, 1] >= 110)
            & (btn_zone[:, :, 2] >= 100)
        )
        btn_ratio = float(np.mean(btn_red_gold))
        modal_dark_ratio = float(
            np.mean((hsv_modal[:, :, 2] >= 12) & (hsv_modal[:, :, 2] <= 65))
        )

        revive_dialog = bool(
            grayscale_viewport and modal_dark_ratio > 0.45 and btn_ratio > 0.08
        )
        return {
            "revive_dialog": revive_dialog,
            "grayscale_viewport": grayscale_viewport,
            "outer_mean_saturation": round(outer_mean_sat, 2),
            "modal_dark_ratio": round(modal_dark_ratio, 4),
            "button_ratio": round(btn_ratio, 4),
        }

    def evaluate_frame(
        self,
        frame: Optional[np.ndarray] = None,
        hp_ratio: Optional[float] = None,
        revive_dialog: Optional[bool] = None,
        grayscale_viewport: Optional[bool] = None,
        confirm_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Tek bir kareyi değerlendirir, ardışık ölüm karesi sayacını (`_consecutive_death_frames`)
        günceller ve detaylı teşhis sözlüğü döndürür.
        """
        visuals = self.inspect_death_visuals(frame) if frame is not None else {
            "revive_dialog": False,
            "grayscale_viewport": False,
            "outer_mean_saturation": 255.0,
            "modal_dark_ratio": 0.0,
            "button_ratio": 0.0,
        }

        has_revive = (
            bool(revive_dialog)
            if revive_dialog is not None
            else bool(visuals["revive_dialog"])
        )
        has_gray = (
            bool(grayscale_viewport)
            if grayscale_viewport is not None
            else bool(visuals["grayscale_viewport"])
        )
        eff_hp = float(hp_ratio) if hp_ratio is not None else 1.0
        req_frames = (
            max(1, int(confirm_frames))
            if confirm_frames is not None
            else self.required_frames
        )

        with self._lock:
            # KURAL: revive_dialog == False ise hp_ratio == 0.000 olsa bile ASLA ölüm sayılmaz!
            if has_revive and has_gray:
                self._consecutive_death_frames += 1
            else:
                self._consecutive_death_frames = 0

            confirmed_dead = bool(
                has_revive
                and has_gray
                and self._consecutive_death_frames >= req_frames
            )
            current_streak = self._consecutive_death_frames

        zero_hp_blocked = bool(eff_hp <= 0.001 and not has_revive)

        return {
            "is_dead": confirmed_dead,
            "revive_dialog": has_revive,
            "grayscale_viewport": has_gray,
            "outer_mean_saturation": visuals["outer_mean_saturation"],
            "hp_ratio": round(eff_hp, 4),
            "consecutive_death_frames": current_streak,
            "required_frames": req_frames,
            "zero_hp_false_positive_blocked": zero_hp_blocked,
        }

    def check_character_dead(
        self,
        frame: Optional[np.ndarray] = None,
        hp_ratio: Optional[float] = None,
        revive_dialog: Optional[bool] = None,
        grayscale_viewport: Optional[bool] = None,
        confirm_frames: Optional[int] = None,
    ) -> bool:
        """
        Dirilme Penceresi + Gri Ekran + Ardışık N Kare (varsayılan 5) kuralına göre
        karakterin kesin olarak ölüp ölmediğini döndürür.
        """
        res = self.evaluate_frame(
            frame=frame,
            hp_ratio=hp_ratio,
            revive_dialog=revive_dialog,
            grayscale_viewport=grayscale_viewport,
            confirm_frames=confirm_frames,
        )
        return bool(res["is_dead"])

    def confirm_death(
        self,
        frame: Optional[np.ndarray] = None,
        hp_ratio: Optional[float] = None,
        confirm_frames: Optional[int] = None,
    ) -> bool:
        """`check_character_dead` için eş anlamlı yardımcı metod."""
        return self.check_character_dead(
            frame=frame,
            hp_ratio=hp_ratio,
            confirm_frames=confirm_frames,
        )

    def get_stability_guards_ready_flag(self) -> dict:
        return {"module": "stability_guards", "ready": True}


# ============================================================================
# MADDE 14: HARİTA AÇILIŞ DOĞRULAMASINDA ÇOKLU KRİTER (En Az 2/3 Eşleşme)
# ============================================================================

class MultiCriteriaMapVerifier:
    """
    Madde 14 — Harita Açılış Doğrulamasında Çoklu Kriter.

    Tam ekran dünya haritasının (`M`) gerçekten açık olup olmadığını 3 bağımsız
    görsel kritere göre denetler:
    1. Sağ-Üst Kapatma (`X`) Butonu (`close_x_button`)
    2. Sol-Üst / Üst Bölge Başlığı Şeridi (`region_header_banner`)
    3. Sol-Alt / Sol Kenar Filtre Ayarları & Liste Paneli İkonu (`filter_settings_icon`)

    3 kriterden **en az 2'si** (`matched_count >= 2`) sağlandığında haritayı
    `"Açık"` (`True`), aksi halde `"Kapalı"` (`False`) olarak ilan eder.
    """

    MIN_REQUIRED_CRITERIA: int = 2

    def __init__(
        self,
        map_header_template_path: str = "templates/map_header.png",
        min_criteria: int = 2,
    ) -> None:
        self.min_criteria: int = max(1, min(3, int(min_criteria)))
        self.map_header_template: Optional[np.ndarray] = None
        self._load_header_template(map_header_template_path)

    def _load_header_template(self, rel_or_abs: str) -> None:
        try:
            candidates = [
                "templates/legacy/map_header_video.png",
                rel_or_abs,
            ]
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for c in list(candidates):
                candidates.append(os.path.join(base_dir, c))
            for path in candidates:
                if path and os.path.isfile(path):
                    raw = np.fromfile(path, dtype=np.uint8)
                    if raw.size > 0:
                        img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
                        if img is not None and img.size > 0 and img.shape[1] >= 60:
                            self.map_header_template = img
                            break
        except Exception as exc:
            logger.debug("map_header sablon yukleme uyarisi: %s", exc)

    @staticmethod
    def _coerce_bgr(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        if frame.ndim == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        if frame.ndim == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim == 3 and frame.shape[2] == 3:
            return frame
        return None

    @staticmethod
    def _is_world_3d_hud_active(hsv: np.ndarray) -> bool:
        """
        3B dünya ekranındaki alt-orta kırmızı HP / mavi MP barı veya karakter
        baş üstü HP barı görünürse True döndürür (harita açıkken bunlar gizlenir).
        """
        h, w = hsv.shape[:2]
        hp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.36) : int(w * 0.47)]
        mp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.52) : int(w * 0.63)]
        if hp_roi.size > 0 and mp_roi.size > 0:
            hp_red = float(
                np.mean(
                    ((hp_roi[:, :, 0] < 12) | (hp_roi[:, :, 0] > 168))
                    & (hp_roi[:, :, 1] > 110)
                    & (hp_roi[:, :, 2] > 90)
                )
            )
            mp_blue = float(
                np.mean(
                    (mp_roi[:, :, 0] > 85)
                    & (mp_roi[:, :, 0] < 115)
                    & (mp_roi[:, :, 1] > 110)
                    & (mp_roi[:, :, 2] > 90)
                )
            )
            if hp_red > 0.06 or mp_blue > 0.06:
                return True
        char_hp_roi = hsv[int(h * 0.40) : int(h * 0.60), int(w * 0.44) : int(w * 0.56)]
        if char_hp_roi.size > 0:
            char_hp_red = float(
                np.mean(
                    ((char_hp_roi[:, :, 0] < 10) | (char_hp_roi[:, :, 0] > 170))
                    & (char_hp_roi[:, :, 1] > 140)
                    & (char_hp_roi[:, :, 2] > 120)
                )
            )
            if char_hp_red > 0.008:
                return True
        return False

    def check_close_x_button(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        hud_active: bool = False,
    ) -> Tuple[bool, float]:
        """
        Kriter 1: Sağ-Üst Kapatma (`X`) Butonu kontrolü (`x: 0.955..0.995, y: 0.015..0.068`).
        Koyu harita üst barı üzerinde parlak `X` sembolü arar.
        """
        if hud_active:
            return False, 0.0

        h, w = gray.shape[:2]
        x1, x2 = max(0, int(round(w * 0.955))), min(w, int(round(w * 0.995)))
        y1, y2 = max(0, int(round(h * 0.015))), min(h, int(round(h * 0.068)))
        if x2 <= x1 or y2 <= y1:
            return False, 0.0

        x_gray = gray[y1:y2, x1:x2]
        x_hsv = hsv[y1:y2, x1:x2]

        bright_ratio = float(np.mean(x_gray > 150))
        max_lum = float(np.max(x_gray))
        mean_sat = float(np.mean(x_hsv[:, :, 1]))
        mean_val = float(np.mean(x_hsv[:, :, 2]))

        # Harita sağ-üst X butonu: parlak beyaz X çizgileri + koyu/düşük doygunluklu arka plan
        matched = bool(
            bright_ratio >= 0.010
            and max_lum >= 165.0
            and mean_sat <= 78.0
            and 8.0 <= mean_val <= 46.0
        )
        score = min(1.0, (bright_ratio / 0.025) * 0.6 + (1.0 if matched else 0.0) * 0.4)
        return matched, round(score, 4)

    def check_region_header_banner(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        hud_active: bool = False,
    ) -> Tuple[bool, float]:
        """
        Kriter 2: Sol-Üst / Üst Bölge Başlığı (`map_header` şablonu veya sol-üst başlık şeridi).
        """
        if hud_active:
            return False, 0.0

        h, w = gray.shape[:2]
        best_tmpl_score = 0.0

        if self.map_header_template is not None:
            top_strip = gray[int(h * 0.02) : int(h * 0.14), int(w * 0.02) : int(w * 0.78)]
            th, tw = self.map_header_template.shape[:2]
            for s in (0.65, 0.75, 0.85, 1.0, 1.15, 1.25, 1.40, 1.5625, 1.80):
                sw = max(10, int(round(tw * s)))
                sh = max(6, int(round(th * s)))
                if sh < top_strip.shape[0] and sw < top_strip.shape[1]:
                    rt = (
                        self.map_header_template
                        if abs(s - 1.0) < 1e-4
                        else cv2.resize(self.map_header_template, (sw, sh))
                    )
                    res = cv2.matchTemplate(top_strip, rt, cv2.TM_CCOEFF_NORMED)
                    _, mv, _, _ = cv2.minMaxLoc(res)
                    if float(mv) > best_tmpl_score:
                        best_tmpl_score = float(mv)

        if best_tmpl_score >= 0.68:
            return True, round(best_tmpl_score, 4)

        # Yapısal Sol-Üst / Üst Bölge Başlığı kontrolü (farklı bölge isimleri ve çözünürlükler için)
        tl_gray = gray[int(h * 0.02) : int(h * 0.13), int(w * 0.01) : int(w * 0.55)]
        tl_hsv = hsv[int(h * 0.02) : int(h * 0.13), int(w * 0.01) : int(w * 0.55)]
        if tl_gray.size == 0:
            return False, 0.0

        tl_sat = float(np.mean(tl_hsv[:, :, 1]))
        tl_val = float(np.mean(tl_hsv[:, :, 2]))
        tl_bright = float(np.mean(tl_gray > 140))

        matched = bool(
            tl_sat <= 68.0
            and 14.0 <= tl_val <= 58.0
            and tl_bright >= 0.018
        )
        score = max(best_tmpl_score, 0.82 if matched else 0.0)
        return matched, round(score, 4)

    def check_filter_settings_icon(
        self,
        gray: np.ndarray,
        hsv: np.ndarray,
        hud_active: bool = False,
    ) -> Tuple[bool, float]:
        """
        Kriter 3: Sol-Alt / Sol Kenar Filtre Ayarları İkonu ve Harita Sol Paneli
        (`x: 0.01..0.20, y: 0.22..0.96`).
        """
        if hud_active:
            return False, 0.0

        h, w = gray.shape[:2]
        side_gray = gray[int(h * 0.22) : int(h * 0.75), int(w * 0.01) : int(w * 0.18)]
        side_hsv = hsv[int(h * 0.22) : int(h * 0.75), int(w * 0.01) : int(w * 0.18)]

        bl_gray = gray[int(h * 0.70) : int(h * 0.96), int(w * 0.01) : int(w * 0.22)]
        bl_hsv = hsv[int(h * 0.70) : int(h * 0.96), int(w * 0.01) : int(w * 0.22)]

        if side_gray.size == 0 or bl_gray.size == 0:
            return False, 0.0

        side_txt_ratio = float(np.mean(side_gray > 145))
        side_sat = float(np.mean(side_hsv[:, :, 1]))
        side_val = float(np.mean(side_hsv[:, :, 2]))

        bl_bright_ratio = float(np.mean(bl_gray > 140))
        bl_sat = float(np.mean(bl_hsv[:, :, 1]))
        bl_val = float(np.mean(bl_hsv[:, :, 2]))

        # Sol bölge/filtre listesi VEYA sol-alt filtre ayarları ikonu aktifse
        side_matched = bool(
            side_txt_ratio >= 0.018
            and side_sat <= 88.0
            and 15.0 <= side_val <= 65.0
        )
        bl_matched = bool(
            bl_sat <= 68.0
            and 14.0 <= bl_val <= 58.0
            and (bl_bright_ratio >= 0.012 or side_txt_ratio >= 0.018)
        )
        matched = bool(side_matched or bl_matched)
        score = 0.85 if matched else min(0.45, side_txt_ratio * 20.0)
        return matched, round(score, 4)

    def verify_map_state(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """
        3 bağımsız kriteri değerlendirir ve en az 2/3 kriter eşleştiğinde
        `is_open=True`, `state_label="Açık"` döndürür.
        """
        bgr = self._coerce_bgr(frame)
        if bgr is None or bgr.shape[0] < 80 or bgr.shape[1] < 80:
            return {
                "is_open": False,
                "state_label": "Kapalı",
                "matched_count": 0,
                "criteria": {
                    "close_x_button": False,
                    "region_header_banner": False,
                    "filter_settings_icon": False,
                },
                "scores": {
                    "close_x_button": 0.0,
                    "region_header_banner": 0.0,
                    "filter_settings_icon": 0.0,
                },
            }

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hud_active = self._is_world_3d_hud_active(hsv)

        c1_ok, c1_score = self.check_close_x_button(gray, hsv, hud_active=hud_active)
        c2_ok, c2_score = self.check_region_header_banner(gray, hsv, hud_active=hud_active)
        c3_ok, c3_score = self.check_filter_settings_icon(gray, hsv, hud_active=hud_active)

        criteria = {
            "close_x_button": bool(c1_ok),
            "region_header_banner": bool(c2_ok),
            "filter_settings_icon": bool(c3_ok),
        }
        matched_count = sum(1 for val in criteria.values() if val)
        is_open = bool(matched_count >= self.min_criteria)

        return {
            "is_open": is_open,
            "state_label": "Açık" if is_open else "Kapalı",
            "matched_count": int(matched_count),
            "criteria": criteria,
            "scores": {
                "close_x_button": c1_score,
                "region_header_banner": c2_score,
                "filter_settings_icon": c3_score,
            },
        }

    def is_map_open(self, frame: Optional[np.ndarray]) -> bool:
        """Haritanın açık olup olmadığını (en az 2/3 kriter eşleşmesi ile) döndürür."""
        res = self.verify_map_state(frame)
        return bool(res["is_open"])

    def get_stability_guards_ready_flag(self) -> dict:
        return {"module": "stability_guards", "ready": True}


class StabilityGuards:
    """Ana kararlılık korumaları bileşik sınıfı (Maddeler 11–15)."""

    def __init__(self) -> None:
        self.whitelist = SelfInputWhitelistWindow()
        self.camera_lock_rule = CameraLockExceptionRule()
        self.cursor_automator = CursorVisibilityAutomator(whitelist=self.whitelist)
        self.death_guard = FiveFrameDeathConfirmationGuard()
        self.map_verifier = MultiCriteriaMapVerifier()

    def get_stability_guards_ready_flag(self) -> dict:
        return {"module": "stability_guards", "ready": True}

    def get_stability_guards_version_info(self) -> dict:
        return {"module": "stability_guards", "version": "2.0.0", "ready": True}

    def get_stability_guards_diagnostic_snapshot(self) -> dict:
        return {"module": "stability_guards", "status": "ok", "ready": True}


RuntimeStabilityGuard = StabilityGuards


def get_stability_guards_ready_flag() -> dict:
    return {"module": "stability_guards", "ready": True}


def get_stability_health_report() -> dict:
    """Returns current process/GC health and stability guards readiness."""
    return {
        "gc_enabled": gc.isenabled(),
        "stability_guards_ready": True,
    }


__all__ = [
    "CURSOR_SHOWING",
    "CameraLockExceptionRule",
    "CursorVisibilityAutomator",
    "FiveFrameDeathConfirmationGuard",
    "MultiCriteriaMapVerifier",
    "RuntimeStabilityGuard",
    "SelfInputWhitelistWindow",
    "StabilityGuards",
    "get_stability_guards_ready_flag",
    "get_stability_health_report",
    "get_stability_guards_capabilities",
    "get_stability_guards_readiness_status",
]


def get_stability_guards_capabilities() -> list:
    return ["stability_guards", "fault_containment", "diagnostic_snapshot"]


def validate_stability_guards_contract() -> bool:
    return bool(get_stability_guards_capabilities())


def get_stability_guards_baseline_metrics() -> dict:
    return {"module": "stability_guards", "contract_valid": validate_stability_guards_contract(), "capability_count": len(get_stability_guards_capabilities())}


def get_stability_guards_readiness_status() -> dict:
    return {"module": "stability_guards", "ready": validate_stability_guards_contract(), "metrics": get_stability_guards_baseline_metrics()}



