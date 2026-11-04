"""
WindMouse / Cubic Bezier Fare İvmesi, 2D Gaussian Sapma ve Fiziksel Çakışma Korumalı Girdi Motoru.

Harici Arduino donanımı gerektirmeden kullanıcının kendi bilgisayarında yerel olarak çalışan,
insan biyometrisini (Fitts's Law ease-in-out ivmelenme, WindMouse stokastik el titremesi,
Gaussian tuş/tıklama basılı tutma süreleri ve 2D uzamsal koordinat sapması) birebir simüle eden
ve fiziksel kullanıcı müdahalesini anında algılayarak çakışmayı önleyen modül.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import logging
import math
import os
import random
import threading
import time
from typing import Any, Callable, Optional, Union

from core.stability_guards import (
    CURSOR_SHOWING,
    CameraLockExceptionRule,
    CursorVisibilityAutomator,
    SelfInputWhitelistWindow,
)

logger = logging.getLogger("core.windmouse_humanizer")
if logger.handlers:
    logger.propagate = False


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.UINT),
        ("dwTime", wintypes.DWORD),
    ]


def _get_cursor_pos() -> tuple[int, int]:
    """Win32 GetCursorPos çağrısı ile mevcut ekran imleç koordinatlarını döndürür."""
    try:
        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception as exc:
        logger.debug("GetCursorPos hatasi: %s", exc)
    return 0, 0


def _set_cursor_pos_win32(x: int, y: int) -> bool:
    """Win32 SetCursorPos çağrısı ile imleci verilen koordinata taşır."""
    try:
        return bool(ctypes.windll.user32.SetCursorPos(int(x), int(y)))
    except Exception as exc:
        logger.debug("SetCursorPos hatasi (%d, %d): %s", x, y, exc)
        return False


class BiometricClickGenerator:
    """
    İnsan biyometrisine uygun Gaussian basılı tutma (dwell time) ve
    2D uzamsal tıklama koordinatı sapması (spatial jitter) üreteci.
    """

    def __init__(
        self,
        key_mean_sec: float = 0.076,
        key_sigma_sec: float = 0.014,
        key_min_sec: float = 0.045,
        key_max_sec: float = 0.115,
        mouse_mean_sec: float = 0.085,
        mouse_sigma_sec: float = 0.016,
        mouse_min_sec: float = 0.055,
        mouse_max_sec: float = 0.130,
    ) -> None:
        self.key_mean_sec = float(key_mean_sec)
        self.key_sigma_sec = float(key_sigma_sec)
        self.key_min_sec = float(key_min_sec)
        self.key_max_sec = float(key_max_sec)

        self.mouse_mean_sec = float(mouse_mean_sec)
        self.mouse_sigma_sec = float(mouse_sigma_sec)
        self.mouse_min_sec = float(mouse_min_sec)
        self.mouse_max_sec = float(mouse_max_sec)

    def get_key_dwell_sec(self) -> float:
        """
        F veya M gibi tuş basımlarında KeyDown ile KeyUp arasında
        45 ms ile 115 ms (0.045s - 0.115s, ortalama 76ms, sigma=14ms)
        arasında Gaussian dağılımlı süre döndürür.
        """
        try:
            for _ in range(8):
                val = random.gauss(self.key_mean_sec, self.key_sigma_sec)
                if self.key_min_sec <= val <= self.key_max_sec:
                    return round(val, 6)
            clamped = max(
                self.key_min_sec,
                min(self.key_max_sec, random.gauss(self.key_mean_sec, self.key_sigma_sec)),
            )
            return round(clamped, 6)
        except Exception as exc:
            logger.error("get_key_dwell_sec hesaplama hatasi: %s", exc)
            return self.key_mean_sec

    def get_mouse_dwell_sec(self) -> float:
        """
        Sol/sağ fare tıklamalarında MouseDown ile MouseUp arasında
        55 ms ile 130 ms (0.055s - 0.130s, ortalama 85ms, sigma=16ms)
        arasında Gaussian dağılımlı süre döndürür.
        """
        try:
            for _ in range(8):
                val = random.gauss(self.mouse_mean_sec, self.mouse_sigma_sec)
                if self.mouse_min_sec <= val <= self.mouse_max_sec:
                    return round(val, 6)
            clamped = max(
                self.mouse_min_sec,
                min(self.mouse_max_sec, random.gauss(self.mouse_mean_sec, self.mouse_sigma_sec)),
            )
            return round(clamped, 6)
        except Exception as exc:
            logger.error("get_mouse_dwell_sec hesaplama hatasi: %s", exc)
            return self.mouse_mean_sec

    def apply_2d_gaussian_jitter(
        self,
        x: int,
        y: int,
        min_radius: float = 3.0,
        max_radius: float = 6.0,
    ) -> tuple[int, int]:
        """
        Haritadaki veya ekrandaki aynı kaynağa her gidildiğinde milimetrik aynı
        piksele tıklamamak için merkezin etrafında ±3 ile ±6 piksel yarıçapında
        2 boyutlu normal (Gaussian) dağılımlı sapma uygular.
        """
        try:
            lo = max(0.5, float(min(min_radius, max_radius)))
            hi = max(lo + 0.5, float(max(min_radius, max_radius)))
            mean_r = (lo + hi) * 0.5
            sigma_r = max(0.35, (hi - lo) / 3.0)

            radius = random.gauss(mean_r, sigma_r)
            radius = max(lo, min(hi, abs(radius)))
            angle = random.uniform(0.0, 2.0 * math.pi)

            dx = int(round(radius * math.cos(angle)))
            dy = int(round(radius * math.sin(angle)))

            if dx == 0 and dy == 0:
                dx = int(math.copysign(max(1, int(round(lo))), random.choice([-1, 1])))

            dist = math.hypot(dx, dy)
            if dist < lo:
                scale = lo / max(dist, 1e-6)
                dx = int(round(dx * scale))
                dy = int(round(dy * scale))
            elif dist > hi:
                scale = hi / dist
                dx = int(round(dx * scale))
                dy = int(round(dy * scale))

            return int(x) + dx, int(y) + dy
        except Exception as exc:
            logger.error("apply_2d_gaussian_jitter hatasi (%d, %d): %s", x, y, exc)
            return int(x), int(y)


class PhysicalInputGuard:
    """
    Fiziksel Fare ve Klavye Çakışma Koruması.

    - Win32 GetCursorPos ile kullanıcının fiziksel faresini oynatıp oynatmadığını
      (botun kendi WindMouse hareketi ve oyunun 3D kamera merkez kilidi dışındaki
      beklenmedik > 350px imleç değişimini) izler.
    - Oyun motorunun 3D aksiyon kamerası modunda imleci ekran merkezine
      ((1280, 720) veya (960, 540), ±40px) kilitlemesini (center-snap) tanır ve
      fiziksel kullanıcı müdahalesi olarak işaretlemez.
    - Fiziksel klavye tuşlarını (GetAsyncKeyState ile botun o an basmadığı tuşları) izler.
    """

    GAME_CENTER_SNAP_POINTS: tuple[tuple[int, int], ...] = (
        (1280, 720),  # 2560x1440 (QHD) ekran merkezi
        (960, 540),   # 1920x1080 (FHD) ekran merkezi
        (1000, 562),  # 2000x1125 pencere merkezi
        (640, 360),   # 1280x720 (HD) ekran merkezi
    )
    CENTER_SNAP_TOLERANCE_PX: float = 40.0

    # İzlenecek yaygın fiziksel klavye sanal tuş kodları (WASD, Ok tuşları, ESC, Space, Shift, Ctrl, Alt, Tab, Q, E, R, T, F, G, M, Z, X, C, V, B, 1-5)
    MONITORED_VK_CODES: tuple[int, ...] = (
        0x1B,  # VK_ESCAPE
        0x20,  # VK_SPACE
        0x09,  # VK_TAB
        0x25,  # VK_LEFT
        0x26,  # VK_UP
        0x27,  # VK_RIGHT
        0x28,  # VK_DOWN
        0x57,  # W
        0x41,  # A
        0x53,  # S
        0x44,  # D
        0x51,  # Q
        0x45,  # E
        0x46,  # F
        0x4D,  # M
        0x52,  # R
        0x54,  # T
        0x47,  # G
        0x5A,  # Z
        0x58,  # X
        0x43,  # C
        0x56,  # V
        0x42,  # B
        0x31,  # 1
        0x32,  # 2
        0x33,  # 3
        0x34,  # 4
        0x35,  # 5
    )

    PHYSICAL_MOUSE_BUTTON_VKS: tuple[int, ...] = (
        0x01,  # VK_LBUTTON
        0x02,  # VK_RBUTTON
        0x04,  # VK_MBUTTON
    )

    def __init__(
        self,
        cursor_deviation_threshold_px: float = 350.0,
        cooldown_sec: float = 1.25,
        cursor_getter: Optional[Callable[[], tuple[int, int]]] = None,
        key_state_getter: Optional[Callable[[int], int]] = None,
        mouse_delta_threshold_px: Optional[float] = None,
        center_snap_radius_px: float = 40.0,
        camera_lock_rule: Optional[CameraLockExceptionRule] = None,
        self_input_whitelist: Optional[SelfInputWhitelistWindow] = None,
        whitelist_window_ms: float = 250.0,
    ) -> None:
        resolved_threshold = (
            float(mouse_delta_threshold_px)
            if mouse_delta_threshold_px is not None
            else float(cursor_deviation_threshold_px)
        )
        self._mouse_delta_threshold_px: float = resolved_threshold
        self.center_snap_radius_px: float = float(center_snap_radius_px)
        self.cooldown_sec = float(cooldown_sec)
        self._cursor_getter = cursor_getter or _get_cursor_pos
        self._key_state_getter = key_state_getter or self._default_get_async_key_state

        self.camera_lock_rule: CameraLockExceptionRule = (
            camera_lock_rule
            or CameraLockExceptionRule(tolerance_px=self.center_snap_radius_px)
        )
        self.self_input_whitelist: SelfInputWhitelistWindow = (
            self_input_whitelist
            or SelfInputWhitelistWindow(default_window_ms=float(whitelist_window_ms))
        )

        self._lock = threading.RLock()
        cur_x, cur_y = self._safe_get_cursor()
        self._expected_x: int = cur_x
        self._expected_y: int = cur_y
        self._has_expected_anchor: bool = True
        self._bot_active_vks: set[int] = set()
        self._last_conflict_ts: float = 0.0
        self._last_conflict_reason: str = ""
        self.last_camera_lock_reason: str = ""

    @property
    def mouse_delta_threshold_px(self) -> float:
        return self._mouse_delta_threshold_px

    @mouse_delta_threshold_px.setter
    def mouse_delta_threshold_px(self, value: float) -> None:
        self._mouse_delta_threshold_px = float(value)

    @property
    def cursor_deviation_threshold_px(self) -> float:
        return self._mouse_delta_threshold_px

    @cursor_deviation_threshold_px.setter
    def cursor_deviation_threshold_px(self, value: float) -> None:
        self._mouse_delta_threshold_px = float(value)

    @staticmethod
    def _default_get_async_key_state(vk_code: int) -> int:
        try:
            return int(ctypes.windll.user32.GetAsyncKeyState(int(vk_code)))
        except Exception:
            return 0

    def _safe_get_cursor(self) -> tuple[int, int]:
        try:
            pos = self._cursor_getter()
            return int(pos[0]), int(pos[1])
        except Exception as exc:
            logger.debug("PhysicalInputGuard imlec okuma hatasi: %s", exc)
            return 0, 0

    @classmethod
    def _get_screen_centers(cls) -> list[tuple[int, int]]:
        centers: list[tuple[int, int]] = list(cls.GAME_CENTER_SNAP_POINTS)
        try:
            sw = int(ctypes.windll.user32.GetSystemMetrics(0))
            sh = int(ctypes.windll.user32.GetSystemMetrics(1))
            if sw > 0 and sh > 0:
                dyn_center = (sw // 2, sh // 2)
                if dyn_center not in centers:
                    centers.append(dyn_center)
        except Exception as exc:
            logger.debug("GetSystemMetrics hatasi: %s", exc)
        return centers

    def is_game_center_snap(self, x: int, y: int) -> bool:
        """
        Verilen (x, y) koordinatının oyun motorunun 3D kamera merkez kilidi
        ((1280, 720) veya (960, 540) ±40px) içinde olup olmadığını döndürür.
        """
        if self.camera_lock_rule is not None:
            self.camera_lock_rule.tolerance_px = max(0.0, float(self.center_snap_radius_px))
            if self.camera_lock_rule.is_camera_lock_position(x, y):
                return True
        tol = max(0.0, float(self.center_snap_radius_px))
        for cx, cy in self._get_screen_centers():
            if math.hypot(float(x) - float(cx), float(y) - float(cy)) <= tol:
                return True
        return False

    def min_distance_to_screen_center(self, x: int, y: int) -> float:
        """Verilen (x, y) noktasının bilinen ekran merkezlerine olan en kısa mesafesini döndürür."""
        centers = self._get_screen_centers()
        return min(
            math.hypot(float(x) - float(cx), float(y) - float(cy))
            for cx, cy in centers
        )

    def record_virtual_mouse(self, x: int, y: int) -> None:
        """Botun sanal fare konumunu kaydeder ve beklenen imleç referansını günceller."""
        with self._lock:
            self._expected_x = int(x)
            self._expected_y = int(y)
            self._has_expected_anchor = True

    def update_expected_cursor(self, x: int, y: int) -> None:
        """Botun kendi WindMouse hareketi sırasında beklenen imleç konumunu günceller."""
        self.record_virtual_mouse(x, y)

    def sync_cursor_anchor(self) -> tuple[int, int]:
        """Mevcut gerçek imleç konumunu beklenen referans noktası olarak senkronize eder."""
        with self._lock:
            cx, cy = self._safe_get_cursor()
            self.record_virtual_mouse(cx, cy)
            return cx, cy

    def record_self_key(self, key: Union[str, int]) -> float:
        """
        Madde 15: Botun kendi gönderdiği sanal tuşu 250ms beyaz listesine kaydeder.
        """
        return self.self_input_whitelist.record_self_key(key)

    def is_within_ignore_window(
        self,
        key: Union[str, int],
        window_ms: Optional[float] = None,
    ) -> bool:
        """
        Madde 15: Verilen tuşun 250ms sanal girdi beyaz listesi penceresinde olup olmadığını döndürür.
        """
        return self.self_input_whitelist.is_within_ignore_window(key, window_ms=window_ms)

    def register_bot_key_press(self, vk_code: Union[int, str], pressed: bool = True) -> None:
        """Botun kendi bastığı sanal tuşların fiziksel müdahale sanılmaması için kaydeder."""
        with self._lock:
            self.self_input_whitelist.record_self_key(vk_code)
            _, resolved_vk = SelfInputWhitelistWindow._normalize_key_and_vk(vk_code)
            if resolved_vk is not None:
                if pressed:
                    self._bot_active_vks.add(int(resolved_vk))
                else:
                    self._bot_active_vks.discard(int(resolved_vk))

    def clear_conflict_state(self) -> None:
        """Çakışma kilidini sıfırlar ve mevcut imleç konumunu referans alır."""
        with self._lock:
            self._last_conflict_ts = 0.0
            self._last_conflict_reason = ""
            cx, cy = self._safe_get_cursor()
            self.record_virtual_mouse(cx, cy)

    def should_abort_input(self) -> tuple[bool, str]:
        """
        Kullanıcının fiziksel fareyi (> 350px beklenmedik sapma ve ekran merkezi dışında)
        veya fiziksel klavyeyi kullanıp kullanmadığını denetler.
        Dönüş: (iptal_edilmeli_mi: bool, sebep_mesajı: str)
        """
        try:
            with self._lock:
                now = time.monotonic()
                actual_x, actual_y = self._safe_get_cursor()

                if self._has_expected_anchor:
                    if self.is_game_center_snap(actual_x, actual_y):
                        # Madde 11: Oyun motorunun 3D kamera merkez kilidi ("Oyun İmleç Kilidi")
                        self.last_camera_lock_reason = CameraLockExceptionRule.LOCK_CLASSIFICATION_LABEL
                        self._expected_x = actual_x
                        self._expected_y = actual_y
                    else:
                        delta_px = math.hypot(
                            actual_x - self._expected_x,
                            actual_y - self._expected_y,
                        )
                        if delta_px > self._mouse_delta_threshold_px:
                            reason = (
                                f"physical_mouse_movement_detected: "
                                f"delta={delta_px:.1f}px > {self._mouse_delta_threshold_px:.1f}px "
                                f"(expected=({self._expected_x},{self._expected_y}), actual=({actual_x},{actual_y}))"
                            )
                            self._expected_x = actual_x
                            self._expected_y = actual_y
                            self._last_conflict_ts = now
                            self._last_conflict_reason = reason
                            logger.warning("Fiziksel fare mudahalesi algilandi: %s", reason)
                            return True, reason

                for vk in self.MONITORED_VK_CODES:
                    if vk in self._bot_active_vks or self.self_input_whitelist.is_within_ignore_window(vk, now=now):
                        continue
                    state = self._key_state_getter(vk)
                    if state & 0x8000:
                        reason = f"physical_keyboard_override_detected: vk=0x{vk:02X}"
                        self._last_conflict_ts = now
                        self._last_conflict_reason = reason
                        logger.warning("Fiziksel klavye mudahalesi algilandi: %s", reason)
                        return True, reason

                if (
                    self._last_conflict_ts > 0.0
                    and (now - self._last_conflict_ts) < self.cooldown_sec
                ):
                    remaining = self.cooldown_sec - (now - self._last_conflict_ts)
                    return (
                        True,
                        f"physical_override_cooldown_active ({remaining:.2f}s left): {self._last_conflict_reason}",
                    )

                return False, "ok"
        except Exception as exc:
            logger.error("should_abort_input denetim hatasi: %s", exc)
            return False, "guard_exception_recovered"

    def check_mid_trajectory_physical_abort(
        self,
        massive_drag_threshold_px: float = 450.0,
    ) -> tuple[bool, str]:
        """
        Yörünge hareketi sırasında yalnızca açık fiziksel tuş/buton basımı veya
        hem beklenen konumdan hem de ekran merkezinden > 450px uzakta devasa bir
        kullanıcı sürüklemesi (massive drag) varsa iptal döndürür.
        """
        try:
            with self._lock:
                now = time.monotonic()
                actual_x, actual_y = self._safe_get_cursor()

                # Madde 11: Oyun merkezi yakalaması (center-snap / "Oyun İmleç Kilidi") ise asla kesme
                if self.is_game_center_snap(actual_x, actual_y):
                    self.last_camera_lock_reason = CameraLockExceptionRule.LOCK_CLASSIFICATION_LABEL
                    self._expected_x = actual_x
                    self._expected_y = actual_y
                elif self._has_expected_anchor:
                    delta_expected = math.hypot(
                        actual_x - self._expected_x,
                        actual_y - self._expected_y,
                    )
                    dist_center = self.min_distance_to_screen_center(actual_x, actual_y)
                    if (
                        delta_expected > massive_drag_threshold_px
                        and dist_center > massive_drag_threshold_px
                    ):
                        reason = (
                            f"massive_physical_mouse_drag_detected: "
                            f"delta={delta_expected:.1f}px, center_dist={dist_center:.1f}px > {massive_drag_threshold_px:.1f}px "
                            f"(expected=({self._expected_x},{self._expected_y}), actual=({actual_x},{actual_y}))"
                        )
                        self._expected_x = actual_x
                        self._expected_y = actual_y
                        self._last_conflict_ts = now
                        self._last_conflict_reason = reason
                        return True, reason

                for vk in self.PHYSICAL_MOUSE_BUTTON_VKS + self.MONITORED_VK_CODES:
                    if vk in self._bot_active_vks or self.self_input_whitelist.is_within_ignore_window(vk, now=now):
                        continue
                    state = self._key_state_getter(vk)
                    if state & 0x8000:
                        reason = f"physical_button_override_detected: vk=0x{vk:02X}"
                        self._last_conflict_ts = now
                        self._last_conflict_reason = reason
                        return True, reason

                return False, "ok"
        except Exception as exc:
            logger.debug("check_mid_trajectory_physical_abort hatasi: %s", exc)
            return False, "ok"


class WindMouseEngine:
    """
    WindMouse + Cubic Bezier İnsan Eli İvmesi Motoru.

    - Farenin hedef noktaya doğrudan ışınlanmasını (SetCursorPos sıçramasını) tamamen engeller.
    - Yerçekimi (gravity), rüzgar/el titremesi (wind), Cubic Bezier mikro kavis ve
      başlangıçta hızlanıp hedefe yaklaşırken yavaşlayan (Fitts's Law / ease-in-out)
      insan eli yörüngesi `(x, y, sleep_dt)` noktaları üretir.
    - Madde 12 (`CursorVisibilityAutomator`): Harita veya UI tıklamasından önce
      `GetCursorInfo` ile imlecin açık olup olmadığını kontrol eder; gizliyse `'alt'`
      tuşuna basarak imleci otomatik açar.
    """

    def __init__(
        self,
        guard: Optional[PhysicalInputGuard] = None,
        biometrics: Optional[BiometricClickGenerator] = None,
        input_guard: Optional[PhysicalInputGuard] = None,
        abort_on_physical_mouse: bool = False,
        cursor_automator: Optional[CursorVisibilityAutomator] = None,
        auto_reveal_cursor: bool = True,
    ) -> None:
        resolved_guard = input_guard or guard or PhysicalInputGuard()
        self.guard: PhysicalInputGuard = resolved_guard
        self.input_guard: PhysicalInputGuard = resolved_guard
        self.biometrics: BiometricClickGenerator = biometrics or BiometricClickGenerator()
        self.abort_on_physical_mouse: bool = bool(abort_on_physical_mouse)
        self.auto_reveal_cursor: bool = bool(auto_reveal_cursor)
        self.cursor_automator: CursorVisibilityAutomator = (
            cursor_automator
            or CursorVisibilityAutomator(
                whitelist=self.input_guard.self_input_whitelist,
            )
        )
        self._sqrt3 = math.sqrt(3.0)
        self._sqrt5 = math.sqrt(5.0)

    def ensure_cursor_visible(self, input_ctrl: Any = None) -> bool:
        """
        Madde 12: Tıklama öncesinde imlecin görünür (`CURSOR_SHOWING`) olup olmadığını
        denetler ve gizliyse `'alt'` tuşu ile açar.
        """
        if self.cursor_automator is None:
            return True
        return self.cursor_automator.ensure_cursor_visible(input_ctrl=input_ctrl)

    @staticmethod
    def _fitts_ease_in_out(progress: float) -> float:
        """
        Fitts's Law insan kolu ivmelenme profili:
        Hareketin başında ivmelenir, orta bölgede akıcı ilerler, hedefe yaklaşırken
        hassas nişan alma (fine-motor homing) nedeniyle yavaşlar.
        Dönüş değeri: bekleme süresi çarpanı (yüksek = daha yavaş hareket).
        """
        p = max(0.0, min(1.0, float(progress)))
        bell_velocity = math.sin(math.pi * p) ** 0.85
        # Hedefe son %20 yaklaşımda ekstra hassas nişan yavaşlaması
        homing_brake = 1.35 if p > 0.80 else 1.0
        speed_factor = max(0.18, min(1.0, bell_velocity))
        return (1.0 - 0.72 * speed_factor) * homing_brake

    @staticmethod
    def _cubic_bezier_point(
        t: float,
        p0: tuple[float, float],
        p1: tuple[float, float],
        p2: tuple[float, float],
        p3: tuple[float, float],
    ) -> tuple[float, float]:
        """Verilen t in [0, 1] için Cubic Bezier eğrisi üzerindeki (x, y) noktasını hesaplar."""
        u = 1.0 - t
        tt = t * t
        uu = u * u
        uuu = uu * u
        ttt = tt * t
        bx = uuu * p0[0] + 3.0 * uu * t * p1[0] + 3.0 * u * tt * p2[0] + ttt * p3[0]
        by = uuu * p0[1] + 3.0 * uu * t * p1[1] + 3.0 * u * tt * p2[1] + ttt * p3[1]
        return bx, by

    def generate_windmouse_path(
        self,
        start_x: int,
        start_y: int,
        dest_x: int,
        dest_y: int,
        gravity: float = 9.0,
        wind: float = 3.0,
        min_wait: float = 0.002,
        max_wait: float = 0.008,
        max_step: float = 12.0,
        target_area: float = 8.0,
    ) -> list[tuple[int, int, float]]:
        """
        Başlangıç `(start_x, start_y)` noktasından hedef `(dest_x, dest_y)` noktasına
        WindMouse + Cubic Bezier kavisli ve Fitts's Law ivmeli `(x, y, sleep_dt)` yörüngesi üretir.
        """
        try:
            sx, sy = float(start_x), float(start_y)
            dx, dy = float(dest_x), float(dest_y)
            total_dist = math.hypot(dx - sx, dy - sy)

            lo_wait = max(0.0005, float(min(min_wait, max_wait)))
            hi_wait = max(lo_wait + 0.0005, float(max(min_wait, max_wait)))

            if total_dist < 1.0:
                return [(int(round(dx)), int(round(dy)), round(lo_wait, 6))]

            # Cubic Bezier kontrol noktaları: insan bilek rotasyonunun doğal kavisini temsil eder
            ux = (dx - sx) / total_dist
            uy = (dy - sy) / total_dist
            px, py = -uy, ux  # Dik vektör

            arc_sign = random.choice([-1.0, 1.0])
            arc_mag_1 = min(45.0, total_dist * random.uniform(0.06, 0.16)) * arc_sign
            arc_mag_2 = min(35.0, total_dist * random.uniform(0.03, 0.11)) * arc_sign

            p0 = (sx, sy)
            p1 = (
                sx + (dx - sx) * 0.30 + px * arc_mag_1,
                sy + (dy - sy) * 0.30 + py * arc_mag_1,
            )
            p2 = (
                sx + (dx - sx) * 0.72 + px * arc_mag_2,
                sy + (dy - sy) * 0.72 + py * arc_mag_2,
            )
            p3 = (dx, dy)

            cur_x, cur_y = sx, sy
            velo_x, velo_y = 0.0, 0.0
            wind_x, wind_y = 0.0, 0.0

            cur_max_step = max(2.0, float(max_step))
            eff_gravity = max(1.0, float(gravity))
            eff_wind = max(0.0, float(wind))
            eff_target_area = max(2.0, float(target_area))

            path: list[tuple[int, int, float]] = []
            last_int_pos = (int(round(sx)), int(round(sy)))
            max_iterations = 600

            for _ in range(max_iterations):
                rem_dist = math.hypot(dx - cur_x, dy - cur_y)
                if rem_dist < 1.0:
                    break

                progress = 1.0 - min(1.0, rem_dist / max(total_dist, 1.0))

                # Hedefe yaklaştıkça Bezier kılavuz noktasını nihai hedefe yaklaştır
                bezier_t = min(1.0, progress + 0.18)
                bx, by = self._cubic_bezier_point(bezier_t, p0, p1, p2, p3)
                blend_to_final = min(1.0, progress * progress)
                guide_x = bx * (1.0 - blend_to_final) + dx * blend_to_final
                guide_y = by * (1.0 - blend_to_final) + dy * blend_to_final
                guide_dist = math.hypot(guide_x - cur_x, guide_y - cur_y) or rem_dist

                w_mag = min(eff_wind, rem_dist)
                if rem_dist >= eff_target_area:
                    wind_x = wind_x / self._sqrt3 + (
                        random.uniform(-w_mag, w_mag) / self._sqrt5
                    )
                    wind_y = wind_y / self._sqrt3 + (
                        random.uniform(-w_mag, w_mag) / self._sqrt5
                    )
                else:
                    wind_x /= self._sqrt3
                    wind_y /= self._sqrt3
                    if cur_max_step < 3.0:
                        cur_max_step = random.uniform(2.5, 5.0)
                    else:
                        cur_max_step /= self._sqrt5

                velo_x += wind_x + eff_gravity * (guide_x - cur_x) / guide_dist
                velo_y += wind_y + eff_gravity * (guide_y - cur_y) / guide_dist

                velo_mag = math.hypot(velo_x, velo_y)
                # Fitts's Law: başlangıçta ve sonda adım boyunu yumuşat
                envelope = 0.35 + 0.65 * math.sin(math.pi * progress)
                dynamic_step_limit = max(1.5, min(cur_max_step * envelope, rem_dist))

                if velo_mag > dynamic_step_limit:
                    random_dist = dynamic_step_limit * (0.55 + 0.45 * random.random())
                    velo_x = (velo_x / velo_mag) * random_dist
                    velo_y = (velo_y / velo_mag) * random_dist

                cur_x += velo_x
                cur_y += velo_y

                ix = int(round(cur_x))
                iy = int(round(cur_y))

                if (ix, iy) != last_int_pos:
                    ease_mult = self._fitts_ease_in_out(progress)
                    raw_wait = lo_wait + (hi_wait - lo_wait) * min(1.0, ease_mult)
                    jittered_wait = random.gauss(raw_wait, (hi_wait - lo_wait) * 0.08)
                    sleep_dt = round(max(lo_wait, min(hi_wait, jittered_wait)), 6)
                    path.append((ix, iy, sleep_dt))
                    last_int_pos = (ix, iy)

            final_pos = (int(round(dx)), int(round(dy)))
            if not path or (path[-1][0], path[-1][1]) != final_pos:
                final_wait = round(
                    max(lo_wait, min(hi_wait, hi_wait * 0.92)),
                    6,
                )
                path.append((final_pos[0], final_pos[1], final_wait))

            return path
        except Exception as exc:
            logger.error(
                "generate_windmouse_path hatasi ((%d,%d)->(%d,%d)): %s",
                start_x,
                start_y,
                dest_x,
                dest_y,
                exc,
            )
            return [(int(dest_x), int(dest_y), round(float(min_wait), 6))]

    def move_cursor_humanized(
        self,
        input_ctrl: Any = None,
        dest_x: int = 0,
        dest_y: int = 0,
        gravity: float = 9.0,
        wind: float = 3.0,
        min_wait: float = 0.002,
        max_wait: float = 0.008,
        max_step: float = 12.0,
        target_area: float = 8.0,
        abort_on_physical_mouse: Optional[bool] = None,
    ) -> bool:
        """
        Üretilen WindMouse eğrisi boyunca imleci pürüzsüzce kaydırır.
        - Başlangıçta mevcut OS imleç konumunu `(cur_x, cur_y)` okur ve
          `self.input_guard.record_virtual_mouse(cur_x, cur_y)` ile senkronize eder
          (böylece önceki tıklamadan kalan bayat konuma veya oyunun 1280x720 merkez
          kilidine karşı karşılaştırma yapıp başlamadan iptal etmez).
        - Yörünge sırasında yalnızca `abort_on_physical_mouse` açıkça etkinleştirilmişse
          (varsayılan `False`) fiziksel tuş basımı veya > 450px devasa sürükleme durumunda keser.
        """
        try:
            # 2-arg çağrı desteği: move_cursor_humanized(dest_x, dest_y)
            if isinstance(input_ctrl, (int, float)) and not hasattr(input_ctrl, "set_cursor_pos"):
                actual_dest_x = int(input_ctrl)
                actual_dest_y = int(dest_x)
                actual_ctrl = None
            else:
                actual_dest_x = int(dest_x)
                actual_dest_y = int(dest_y)
                actual_ctrl = input_ctrl

            # 1. Başlangıçta mevcut OS imleç konumunu oku ve hemen record_virtual_mouse ile senkronize et
            cur_x, cur_y = self.input_guard._safe_get_cursor()
            self.input_guard.record_virtual_mouse(cur_x, cur_y)

            path = self.generate_windmouse_path(
                start_x=cur_x,
                start_y=cur_y,
                dest_x=actual_dest_x,
                dest_y=actual_dest_y,
                gravity=gravity,
                wind=wind,
                min_wait=min_wait,
                max_wait=max_wait,
                max_step=max_step,
                target_area=target_area,
            )

            setter_fn = self._resolve_cursor_setter(actual_ctrl)
            use_physical_abort = (
                self.abort_on_physical_mouse
                if abort_on_physical_mouse is None
                else bool(abort_on_physical_mouse)
            )

            for step_x, step_y, sleep_dt in path:
                if use_physical_abort:
                    abort, reason = self.input_guard.check_mid_trajectory_physical_abort(
                        massive_drag_threshold_px=450.0,
                    )
                    if abort:
                        logger.warning(
                            "move_cursor_humanized hareket sirasinda kullanici mudahalesi ile kesildi: %s",
                            reason,
                        )
                        return False

                setter_fn(step_x, step_y)
                self.input_guard.record_virtual_mouse(step_x, step_y)

                if sleep_dt > 0.0:
                    time.sleep(sleep_dt)

            return True
        except Exception as exc:
            logger.error(
                "move_cursor_humanized hatasi -> (%s, %s): %s",
                dest_x,
                dest_y,
                exc,
            )
            return False

    def click_humanized(
        self,
        input_ctrl: Any = None,
        x: int = 0,
        y: int = 0,
        button: str = "left",
        apply_jitter: bool = True,
        abort_on_physical_mouse: Optional[bool] = None,
    ) -> bool:
        """
        İmleci hedef koordinata WindMouse ile insansı şekilde taşır ve biyometrik
        basılı tutma süresiyle tıklama gerçekleştirir.
        Başlangıçta mevcut OS imleç konumunu `(cur_x, cur_y)` okuyup
        `self.input_guard.record_virtual_mouse(cur_x, cur_y)` çağırır.
        """
        try:
            if isinstance(input_ctrl, (int, float)) and not hasattr(input_ctrl, "set_cursor_pos"):
                target_x = int(input_ctrl)
                target_y = int(x)
                actual_ctrl = None
            else:
                target_x = int(x)
                target_y = int(y)
                actual_ctrl = input_ctrl

            # Madde 12: Tıklamadan önce imleç gizliyse 'alt' tuşu ile görünür yap
            if self.auto_reveal_cursor and self.cursor_automator is not None:
                self.cursor_automator.ensure_cursor_visible(input_ctrl=actual_ctrl)

            # Başlangıçta mevcut OS imleç konumunu senkronize et
            cur_x, cur_y = self.input_guard._safe_get_cursor()
            self.input_guard.record_virtual_mouse(cur_x, cur_y)

            if apply_jitter and self.biometrics is not None:
                target_x, target_y = self.biometrics.apply_2d_gaussian_jitter(
                    target_x, target_y
                )

            moved = self.move_cursor_humanized(
                actual_ctrl,
                target_x,
                target_y,
                abort_on_physical_mouse=abort_on_physical_mouse,
            )
            if not moved:
                return False

            self.input_guard.record_virtual_mouse(target_x, target_y)
            btn_vk = 0x02 if str(button).lower() == "right" else 0x01
            self.input_guard.record_self_key(btn_vk)

            if actual_ctrl is not None:
                for click_attr in ("click_map", "human_click", "click"):
                    click_fn = getattr(actual_ctrl, click_attr, None)
                    if callable(click_fn):
                        click_fn(target_x, target_y, button=button)
                        self.input_guard.record_virtual_mouse(target_x, target_y)
                        self.input_guard.record_self_key(btn_vk)
                        return True

            # Fallback Win32 mouse_event tıklaması
            dwell_sec = (
                self.biometrics.get_mouse_dwell_sec()
                if self.biometrics is not None
                else 0.085
            )
            btn = str(button).lower()
            down_flag, up_flag = (0x0008, 0x0010) if btn == "right" else (0x0002, 0x0004)
            _set_cursor_pos_win32(target_x, target_y)
            self.input_guard.record_virtual_mouse(target_x, target_y)
            ctypes.windll.user32.mouse_event(down_flag, 0, 0, 0, 0)
            time.sleep(max(0.01, dwell_sec))
            ctypes.windll.user32.mouse_event(up_flag, 0, 0, 0, 0)
            self.input_guard.record_virtual_mouse(target_x, target_y)
            return True
        except Exception as exc:
            logger.error("click_humanized hatasi (%s, %s): %s", x, y, exc)
            return False

    @staticmethod
    def _resolve_cursor_setter(input_ctrl: Any) -> Callable[[int, int], Any]:
        """input_ctrl nesnesindeki uygun imleç taşıma metodunu veya Win32 SetCursorPos'u çözer."""
        if input_ctrl is not None:
            for attr_name in (
                "_set_cursor_pos_raw",
                "_set_cursor_pos",
                "set_cursor_pos",
                "move_cursor_raw",
            ):
                fn = getattr(input_ctrl, attr_name, None)
                if callable(fn):
                    return fn
        return _set_cursor_pos_win32


# Geriye dönük uyumluluk ve alternatif sınıf adı takma adı
WindMouseController = WindMouseEngine


def check_driver_backend_status() -> dict[str, Any]:
    """
    Sanal Sürücü Durum Denetleyicisi.
    Bilgisayarda `FakerInput`, `Interception` (`interception.dll`) veya yerel
    `SendInput ScanCode` katmanlarından hangisinin aktif olduğunu raporlar.
    """
    status: dict[str, Any] = {
        "active_backend": "sendinput_scancode",
        "fakerinput_available": False,
        "fakerinput_device_path": None,
        "interception_available": False,
        "interception_dll_path": None,
        "sendinput_scancode_available": False,
        "hardware_free_mode": True,
        "details": [],
    }

    # 1. FakerInput sanal sürücü / sembolik bağ kontrolü
    try:
        fakerinput_candidates = (
            r"\\.\FakerInput",
            r"C:\Windows\System32\drivers\FakerInput.sys",
            r"C:\Program Files\FakerInput\FakerInput.dll",
        )
        for candidate in fakerinput_candidates:
            if candidate.startswith("\\\\.\\"):
                # CreateFileW ile sembolik cihaz varlığını sorgula (GENERIC_READ=0, OPEN_EXISTING=3)
                handle = ctypes.windll.kernel32.CreateFileW(
                    candidate,
                    0,
                    0x00000001 | 0x00000002,
                    None,
                    3,
                    0,
                    None,
                )
                if handle not in (-1, 0, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFF):
                    ctypes.windll.kernel32.CloseHandle(handle)
                    status["fakerinput_available"] = True
                    status["fakerinput_device_path"] = candidate
                    status["details"].append(f"FakerInput device active: {candidate}")
                    break
            elif os.path.exists(candidate):
                status["fakerinput_available"] = True
                status["fakerinput_device_path"] = candidate
                status["details"].append(f"FakerInput driver file detected: {candidate}")
                break
    except Exception as exc:
        logger.debug("FakerInput denetim hatasi: %s", exc)

    # 2. Interception (interception.dll / keyboard.sys) kontrolü
    try:
        interception_candidates = (
            "interception.dll",
            r"C:\Windows\System32\interception.dll",
            r"C:\Windows\System32\drivers\keyboard.sys",
        )
        for dll_candidate in interception_candidates:
            if dll_candidate.endswith(".sys"):
                if os.path.exists(dll_candidate):
                    status["interception_available"] = True
                    status["interception_dll_path"] = dll_candidate
                    status["details"].append(
                        f"Interception kernel driver detected: {dll_candidate}"
                    )
                    break
            else:
                try:
                    lib = ctypes.WinDLL(dll_candidate)
                    if hasattr(lib, "interception_create_context"):
                        status["interception_available"] = True
                        status["interception_dll_path"] = dll_candidate
                        status["details"].append(
                            f"Interception DLL loaded: {dll_candidate}"
                        )
                        break
                except OSError:
                    continue
    except Exception as exc:
        logger.debug("Interception denetim hatasi: %s", exc)

    # 3. Yerel Win32 SendInput ScanCode katmanı kontrolü
    try:
        user32 = ctypes.windll.user32
        has_send_input = hasattr(user32, "SendInput") and hasattr(
            user32, "MapVirtualKeyW"
        )
        status["sendinput_scancode_available"] = bool(has_send_input)
        if has_send_input:
            status["details"].append(
                "Win32 SendInput + MapVirtualKeyW ScanCode layer ready"
            )
    except Exception as exc:
        logger.debug("SendInput denetim hatasi: %s", exc)

    if status["fakerinput_available"]:
        status["active_backend"] = "fakerinput"
    elif status["interception_available"]:
        status["active_backend"] = "interception"
    elif status["sendinput_scancode_available"]:
        status["active_backend"] = "sendinput_scancode"
    else:
        status["active_backend"] = "unavailable"

    return status
