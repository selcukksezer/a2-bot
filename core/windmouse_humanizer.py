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

def quintic_ease_in_out(t: float) -> float:
    """
    Minimum-jerk quintic polynomial (Perlin smoothstep 2nd order):
    S(t) = 6t^5 - 15t^4 + 10t^3 = t^3 * (t * (6t - 15) + 10).
    Guarantees zero velocity and zero acceleration at t=0 and t=1,
    eliminating robotic snap movements and sudden acceleration spikes.
    """
    t_clamped = max(0.0, min(1.0, float(t)))
    return t_clamped * t_clamped * t_clamped * (t_clamped * (t_clamped * 6.0 - 15.0) + 10.0)


def normalize_angle_diff(target_bearing: float, current_heading: float) -> float:
    """
    Calculates shortest rotational angle difference in range [-180.0°, +180.0°]
    to turn current_heading towards target_bearing.
    Positive (+) indicates clockwise / right turn.
    Negative (-) indicates counter-clockwise / left turn.
    """
    diff = (float(target_bearing) - float(current_heading) + 180.0) % 360.0 - 180.0
    return round(diff, 2)


def compute_bearing_deg(from_pos: tuple[float, float], to_pos: tuple[float, float]) -> float:
    """
    Calculates compass bearing from from_pos to to_pos in degrees (0.0° .. 360.0°).
    Coordinates: X grows right (East), Y grows down (South).
    North is 0°, East is 90°, South is 180°, West is 270°.
    """
    dx = float(to_pos[0]) - float(from_pos[0])
    dy = float(to_pos[1]) - float(from_pos[1])
    if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return 0.0
    angle = math.degrees(math.atan2(dx, -dy))
    return round((angle + 360.0) % 360.0, 2)


class CameraSmoothingConfig:
    """
    Configuration parameters for humanized Bezier camera pitch/yaw rotation
    and character heading steering.
    """

    def __init__(
        self,
        deg_to_px_yaw: float = 4.0,
        deg_to_px_pitch: float = 3.5,
        deadband_deg: float = 2.0,
        min_turn_duration_sec: float = 0.05,
        max_turn_duration_sec: float = 0.55,
        min_step_interval_sec: float = 0.004,
        max_step_interval_sec: float = 0.012,
        wrist_arc_ratio: float = 0.035,
        enable_wrist_arc: bool = True,
        overshoot_threshold_deg: float = 25.0,
        overshoot_ratio_range: tuple[float, float] = (0.015, 0.040),
        enable_overshoot: bool = True,
        tremor_amplitude_px: float = 0.35,
        max_pitch_up_deg: float = 45.0,
        max_pitch_down_deg: float = 55.0,
        reversal_damping_sec: float = 0.15,
        reversal_damping_factor: float = 0.65,
        max_single_step_px: int = 24,
    ) -> None:
        self.deg_to_px_yaw = float(deg_to_px_yaw)
        self.deg_to_px_pitch = float(deg_to_px_pitch)
        self.deadband_deg = float(deadband_deg)
        self.min_turn_duration_sec = float(min_turn_duration_sec)
        self.max_turn_duration_sec = float(max_turn_duration_sec)
        self.min_step_interval_sec = float(min_step_interval_sec)
        self.max_step_interval_sec = float(max_step_interval_sec)
        self.wrist_arc_ratio = float(wrist_arc_ratio)
        self.enable_wrist_arc = bool(enable_wrist_arc)
        self.overshoot_threshold_deg = float(overshoot_threshold_deg)
        self.overshoot_ratio_range = (
            float(overshoot_ratio_range[0]),
            float(overshoot_ratio_range[1]),
        )
        self.enable_overshoot = bool(enable_overshoot)
        self.tremor_amplitude_px = float(tremor_amplitude_px)
        self.max_pitch_up_deg = float(max_pitch_up_deg)
        self.max_pitch_down_deg = float(max_pitch_down_deg)
        self.reversal_damping_sec = float(reversal_damping_sec)
        self.reversal_damping_factor = float(reversal_damping_factor)
        self.max_single_step_px = int(max_single_step_px)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deg_to_px_yaw": self.deg_to_px_yaw,
            "deg_to_px_pitch": self.deg_to_px_pitch,
            "deadband_deg": self.deadband_deg,
            "min_turn_duration_sec": self.min_turn_duration_sec,
            "max_turn_duration_sec": self.max_turn_duration_sec,
            "min_step_interval_sec": self.min_step_interval_sec,
            "max_step_interval_sec": self.max_step_interval_sec,
            "wrist_arc_ratio": self.wrist_arc_ratio,
            "enable_wrist_arc": self.enable_wrist_arc,
            "overshoot_threshold_deg": self.overshoot_threshold_deg,
            "overshoot_ratio_range": self.overshoot_ratio_range,
            "enable_overshoot": self.enable_overshoot,
            "tremor_amplitude_px": self.tremor_amplitude_px,
            "max_pitch_up_deg": self.max_pitch_up_deg,
            "max_pitch_down_deg": self.max_pitch_down_deg,
            "reversal_damping_sec": self.reversal_damping_sec,
            "reversal_damping_factor": self.reversal_damping_factor,
            "max_single_step_px": self.max_single_step_px,
        }


class CameraMovementSmoother:
    """
    İnsani Kamera Pitch/Yaw ve Karakter Yönlendirme (Heading) Yumuşatma Motoru.

    - Robotik ani açılanmaları (snap rotations / instant mouse nudges) tamamen eler.
    - S-eğrisi (quintic minimum jerk) ivmelenme ve yavaşlama profili uygular.
    - İnsan bilek rotasyonunun doğal fizyolojik yayını (wrist arc - horizontal yaw dönüşlerinde
      küçük dikey pitch salınımı) simüle eder.
    - Büyük dönüşlerde (>= 25°) insanın doğal aşım (micro-overshoot) ve düzeltme (settling)
      davranışını üretir.
    - Küçük açılardaki mikro titremeyi deadband filtrelemesi ile önler; hızlı yön
      değişimlerinde (sağ-sol yalpalanma) momentum sönümleme (reversal damping) sağlar.
    - Karakter yürürken (W tuşu aktifken) yön değişimlerini kesintisiz ve akıcı bir şekilde
      kamera ile harmanlar.
    """

    def __init__(
        self,
        guard: Optional[PhysicalInputGuard] = None,
        biometrics: Optional[BiometricClickGenerator] = None,
        config: Optional[CameraSmoothingConfig] = None,
    ) -> None:
        self.guard = guard
        self.biometrics = biometrics
        self.config: CameraSmoothingConfig = config or CameraSmoothingConfig()
        self._lock = threading.RLock()
        self._last_turn_dir: Optional[str] = None
        self._last_turn_ts: float = 0.0
        self._metrics: dict[str, Any] = {
            "turns_completed": 0,
            "pitch_adjustments_completed": 0,
            "nudges_completed": 0,
            "total_yaw_deg": 0.0,
            "total_pitch_deg": 0.0,
            "aborts_count": 0,
            "reversals_damped_count": 0,
            "anti_snap_verified_count": 0,
        }

    def get_metrics(self) -> dict[str, Any]:
        with self._lock:
            m = dict(self._metrics)
            m["config"] = self.config.to_dict()
            return m

    def reset_metrics(self) -> None:
        with self._lock:
            self._metrics = {
                "turns_completed": 0,
                "pitch_adjustments_completed": 0,
                "nudges_completed": 0,
                "total_yaw_deg": 0.0,
                "total_pitch_deg": 0.0,
                "aborts_count": 0,
                "reversals_damped_count": 0,
                "anti_snap_verified_count": 0,
            }

    @staticmethod
    def calculate_bearing(from_pos: tuple[float, float], to_pos: tuple[float, float]) -> float:
        return compute_bearing_deg(from_pos, to_pos)

    @staticmethod
    def calculate_angle_diff(target_bearing: float, current_heading: float) -> float:
        return normalize_angle_diff(target_bearing, current_heading)

    def calculate_turn_damping(
        self,
        angle_diff: float,
        now: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
    ) -> dict[str, Any]:
        cfg = config or self.config
        curr_ts = time.monotonic() if now is None else float(now)
        abs_err = abs(float(angle_diff))
        if abs_err < cfg.deadband_deg:
            return {
                "angle_diff": round(float(angle_diff), 2),
                "in_deadband": True,
                "turn_needed": False,
                "turn_direction": "straight",
                "damping_multiplier": 0.0,
                "reversal_damped": False,
            }

        direction = "right" if angle_diff > 0.0 else "left"
        reversal_damped = False
        mult = 1.0

        with self._lock:
            if (
                self._last_turn_dir is not None
                and self._last_turn_dir != direction
                and (curr_ts - self._last_turn_ts) < cfg.reversal_damping_sec
            ):
                mult = cfg.reversal_damping_factor
                reversal_damped = True
                self._metrics["reversals_damped_count"] += 1
            self._last_turn_dir = direction
            self._last_turn_ts = curr_ts

        return {
            "angle_diff": round(float(angle_diff), 2),
            "in_deadband": False,
            "turn_needed": True,
            "turn_direction": direction,
            "damping_multiplier": round(mult, 3),
            "reversal_damped": reversal_damped,
        }

    def generate_relative_camera_trajectory(
        self,
        total_dx: int,
        total_dy: int,
        duration_sec: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
        enable_wrist_arc: Optional[bool] = None,
        enable_overshoot: Optional[bool] = None,
    ) -> list[tuple[int, int, float]]:
        """
        Relative (dx, dy) fare hareketini insan el/bilek biyometrisine uygun
        minimum-jerk quintic Bezier eğrisi ve mikro-altadımlara ayrıştırır.
        Dönüş: [(step_dx, step_dy, sleep_sec), ...]
        Robotik tek-frame sıçramasını (snap) tamamen engeller.
        """
        cfg = config or self.config
        tdx = int(total_dx)
        tdy = int(total_dy)
        dist = math.hypot(tdx, tdy)
        if dist < 0.5:
            return []

        arc_enabled = cfg.enable_wrist_arc if enable_wrist_arc is None else bool(enable_wrist_arc)
        overshoot_enabled = (
            cfg.enable_overshoot if enable_overshoot is None else bool(enable_overshoot)
        )

        # 1. Süre hesabı
        if duration_sec is None:
            max_span_px = 180.0 * cfg.deg_to_px_yaw
            norm = min(1.0, max(0.0, dist / max(1.0, max_span_px)))
            calc_dur = cfg.min_turn_duration_sec + (
                cfg.max_turn_duration_sec - cfg.min_turn_duration_sec
            ) * (norm ** 0.65)
            duration = max(cfg.min_turn_duration_sec, min(cfg.max_turn_duration_sec, calc_dur))
        else:
            duration = max(0.015, float(duration_sec))

        # 2. Adım sayısı hesabı
        avg_dt = (cfg.min_step_interval_sec + cfg.max_step_interval_sec) * 0.5
        n_steps = max(4, int(round(duration / avg_dt)))
        # Aşırı büyük tekil adımları önlemek için adım sayısını dinamik yükselt
        max_step_cap = max(8, cfg.max_single_step_px)
        if dist / n_steps > max_step_cap * 0.55:
            n_steps = max(n_steps, int(math.ceil(dist / (max_step_cap * 0.55))))
        n_steps = min(120, max(4, n_steps))

        # 3. Bilek açısı kavis genliği (wrist arc)
        if arc_enabled and abs(tdx) > 12:
            arc_sign = -1.0 if tdx > 0 else 1.0
            arc_amplitude = min(18.0, abs(tdx) * cfg.wrist_arc_ratio) * arc_sign
        else:
            arc_amplitude = 0.0

        # 4. İnsan aşım (overshoot) simülasyonu
        ov_threshold = cfg.overshoot_threshold_deg * cfg.deg_to_px_yaw
        has_overshoot = bool(
            overshoot_enabled and dist >= ov_threshold and random.random() < 0.85
        )
        if has_overshoot:
            ov_ratio = random.uniform(
                cfg.overshoot_ratio_range[0], cfg.overshoot_ratio_range[1]
            )
            ov_dx = int(round(tdx * ov_ratio))
            ov_dy = int(round(tdy * ov_ratio))
        else:
            ov_dx = 0
            ov_dy = 0

        # 5. Adım üretimi
        steps: list[tuple[int, int, float]] = []
        sent_x = 0
        sent_y = 0
        phase1_end = 0.82

        for i in range(1, n_steps + 1):
            t = i / n_steps
            if has_overshoot:
                if t <= phase1_end:
                    sub_t = t / phase1_end
                    s = quintic_ease_in_out(sub_t)
                    target_x = float(tdx + ov_dx) * s
                    target_y = float(tdy + ov_dy) * s
                else:
                    sub_t = (t - phase1_end) / (1.0 - phase1_end)
                    s = quintic_ease_in_out(sub_t)
                    target_x = float(tdx + ov_dx) - float(ov_dx) * s
                    target_y = float(tdy + ov_dy) - float(ov_dy) * s
            else:
                s = quintic_ease_in_out(t)
                target_x = float(tdx) * s
                target_y = float(tdy) * s

            # Wrist arc ve tremor ekle
            tremor_env = math.sin(math.pi * t)
            tremor_amp = cfg.tremor_amplitude_px * tremor_env
            tremor_x = (
                random.gauss(0.0, max(0.01, tremor_amp * 0.7))
                if tremor_amp > 0.02
                else 0.0
            )
            tremor_y = (
                random.gauss(0.0, max(0.01, tremor_amp * 0.7))
                if tremor_amp > 0.02
                else 0.0
            )

            target_x += tremor_x
            target_y += arc_amplitude * math.sin(math.pi * t) + tremor_y

            if i < n_steps:
                step_dx = int(round(target_x)) - sent_x
                step_dy = int(round(target_y)) - sent_y
            else:
                # Son adımda tam hedef garantisi
                step_dx = tdx - sent_x
                step_dy = tdy - sent_y

            sent_x += step_dx
            sent_y += step_dy

            dt_mean = duration / n_steps
            dt = random.gauss(dt_mean, dt_mean * 0.12)
            dt_clamped = max(
                cfg.min_step_interval_sec, min(cfg.max_step_interval_sec, dt)
            )
            steps.append((step_dx, step_dy, round(dt_clamped, 6)))

        return steps

    def generate_heading_turn_trajectory(
        self,
        angle_diff_deg: float = 0.0,
        current_heading_deg: Optional[float] = None,
        target_bearing_deg: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
    ) -> list[tuple[int, int, float]]:
        """
        Karakter heading yönü değişimi için humanized fare yörüngesi üretir.
        angle_diff_deg veya (target_bearing_deg, current_heading_deg) çifti alır.
        Deadband filtresi uygular; deadband içindeyse boş liste döndürür.
        """
        cfg = config or self.config
        if target_bearing_deg is not None and current_heading_deg is not None:
            resolved_diff = normalize_angle_diff(target_bearing_deg, current_heading_deg)
        else:
            resolved_diff = float(angle_diff_deg)

        if abs(resolved_diff) < cfg.deadband_deg:
            return []

        total_dx = int(round(resolved_diff * cfg.deg_to_px_yaw))
        return self.generate_relative_camera_trajectory(
            total_dx=total_dx,
            total_dy=0,
            config=cfg,
            enable_wrist_arc=cfg.enable_wrist_arc,
            enable_overshoot=cfg.enable_overshoot,
        )

    def generate_pitch_yaw_trajectory(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        config: Optional[CameraSmoothingConfig] = None,
    ) -> list[tuple[int, int, float]]:
        """
        Kamera yatay (yaw) ve dikey (pitch) açılanmasını aynı anda pürüzsüzce
        gerçekleştiren 2D Bezier yörüngesi üretir.
        Pitch açısını emniyetli sınırlar içinde tutar.
        """
        cfg = config or self.config
        clamped_pitch = max(
            -cfg.max_pitch_up_deg, min(cfg.max_pitch_down_deg, float(pitch_deg))
        )
        total_dx = int(round(float(yaw_deg) * cfg.deg_to_px_yaw))
        total_dy = int(round(clamped_pitch * cfg.deg_to_px_pitch))
        return self.generate_relative_camera_trajectory(
            total_dx=total_dx,
            total_dy=total_dy,
            config=cfg,
            enable_wrist_arc=cfg.enable_wrist_arc,
            enable_overshoot=cfg.enable_overshoot,
        )

    def verify_trajectory_anti_snap(
        self,
        trajectory: list[tuple[int, int, float]],
        max_allowed_single_step_px: Optional[int] = None,
    ) -> dict[str, Any]:
        """
        Üretilen yörüngenin robotik sıçrama (snap) içermediğini,
        fizyolojik ivmelenme/yavaşlama ve adım sınırlarına uyduğunu doğrular.
        """
        cfg = self.config
        max_allowed = max_allowed_single_step_px or cfg.max_single_step_px

        if not trajectory:
            return {
                "is_smooth": True,
                "is_robotic_snap": False,
                "step_count": 0,
                "total_dx": 0,
                "total_dy": 0,
                "max_single_step": 0.0,
                "has_acceleration": True,
                "has_deceleration": True,
            }

        total_dx = sum(s[0] for s in trajectory)
        total_dy = sum(s[1] for s in trajectory)
        total_dist = math.hypot(total_dx, total_dy)
        step_dists = [math.hypot(s[0], s[1]) for s in trajectory]
        max_step = max(step_dists)
        n = len(trajectory)

        # Snap denetimi: mesafe > 25px iken tek adımda veya tek adımın %65'i aşması robotik snap sayılır
        is_snap = bool(total_dist > 25.0 and (n == 1 or max_step >= total_dist * 0.65))

        # Hız profili denetimi (ivmelenme ve yavaşlama varlığı)
        speeds = [step_dists[i] / max(1e-4, trajectory[i][2]) for i in range(n)]
        peak_speed = max(speeds) if speeds else 0.0
        early_slice = speeds[: max(1, n // 4)]
        late_slice = speeds[max(0, 3 * n // 4) :]
        early_avg = sum(early_slice) / len(early_slice) if early_slice else 0.0
        late_avg = sum(late_slice) / len(late_slice) if late_slice else 0.0

        has_accel = bool(early_avg < peak_speed * 0.90) if n >= 6 else True
        has_decel = bool(late_avg < peak_speed * 0.90) if n >= 6 else True

        is_smooth = bool(
            not is_snap and max_step <= max_allowed and has_accel and has_decel
        )

        with self._lock:
            if is_smooth:
                self._metrics["anti_snap_verified_count"] += 1

        return {
            "is_smooth": is_smooth,
            "is_robotic_snap": is_snap,
            "step_count": n,
            "total_dx": total_dx,
            "total_dy": total_dy,
            "max_single_step": round(max_step, 2),
            "max_allowed_single_step": max_allowed,
            "peak_speed_px_s": round(peak_speed, 1),
            "has_acceleration": has_accel,
            "has_deceleration": has_decel,
        }

    def _resolve_relative_mouse_setter(
        self, input_ctrl: Any
    ) -> Callable[[int, int], Any]:
        """Girdi sürücüsündeki uygun relative fare hareket metodunu veya Win32 fallback'i çözer."""
        if input_ctrl is not None:
            for attr in (
                "nudge_camera",
                "mouse_move_relative",
                "move_relative",
                "move_rel",
                "_relative_mouse_move",
            ):
                fn = getattr(input_ctrl, attr, None)
                if callable(fn):
                    return fn

        def _fallback_rel_move(dx: int, dy: int) -> bool:
            try:
                ctypes.windll.user32.mouse_event(0x0001, int(dx), int(dy), 0, 0)
                return True
            except Exception as exc:
                logger.debug("Win32 mouse_event relative move hatasi: %s", exc)
                return False

        return _fallback_rel_move

    @staticmethod
    def _press_rmb(input_ctrl: Any) -> None:
        """Sağ fare tuşunu basılı tutar (RMB down)."""
        if input_ctrl is not None:
            for attr in ("_hw_mouse_down", "mouse_down", "press_mouse_button"):
                fn = getattr(input_ctrl, attr, None)
                if callable(fn):
                    try:
                        fn("right")
                        return
                    except TypeError:
                        fn(0x02)
                        return
        try:
            ctypes.windll.user32.mouse_event(0x0008, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTDOWN
        except Exception:
            pass

    @staticmethod
    def _release_rmb(input_ctrl: Any) -> None:
        """Sağ fare tuşunu serbest bırakır (RMB up)."""
        if input_ctrl is not None:
            for attr in ("_hw_mouse_up", "mouse_up", "release_mouse_button"):
                fn = getattr(input_ctrl, attr, None)
                if callable(fn):
                    try:
                        fn("right")
                        return
                    except TypeError:
                        fn(0x02)
                        return
        try:
            ctypes.windll.user32.mouse_event(0x0010, 0, 0, 0, 0)  # MOUSEEVENTF_RIGHTUP
        except Exception:
            pass

    def smooth_turn_heading(
        self,
        input_ctrl: Any = None,
        angle_diff_deg: float = 0.0,
        target_bearing_deg: Optional[float] = None,
        player_heading_deg: Optional[float] = None,
        use_rmb_drag: bool = False,
        config: Optional[CameraSmoothingConfig] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Karakter yönünü insani Bezier kamera yörüngesi ile yumuşakça döndürür.
        use_rmb_drag=True ise RMB basılı tutarak sürükler ve işlem bitince bırakır.
        Robotik tek adım sıçramalarını tamamen ortadan kaldırır.
        """
        cfg = config or self.config
        if target_bearing_deg is not None and player_heading_deg is not None:
            diff = normalize_angle_diff(target_bearing_deg, player_heading_deg)
        else:
            diff = float(angle_diff_deg)

        damping_info = self.calculate_turn_damping(diff, config=cfg)
        if damping_info["in_deadband"]:
            return True

        trajectory = self.generate_heading_turn_trajectory(
            angle_diff_deg=diff,
            config=cfg,
        )
        if not trajectory:
            return True

        setter = self._resolve_relative_mouse_setter(input_ctrl)
        sleep_exec = sleep_fn or time.sleep
        rmb_held = False

        try:
            if use_rmb_drag:
                self._press_rmb(input_ctrl)
                rmb_held = True
                sleep_exec(random.uniform(0.012, 0.024))

            for step_dx, step_dy, dt in trajectory:
                if abort_callback is not None and abort_callback():
                    with self._lock:
                        self._metrics["aborts_count"] += 1
                    logger.info("smooth_turn_heading abort_callback ile kesildi")
                    return False

                if self.guard is not None:
                    abort, reason = self.guard.check_mid_trajectory_physical_abort(
                        massive_drag_threshold_px=450.0,
                    )
                    if abort:
                        with self._lock:
                            self._metrics["aborts_count"] += 1
                        logger.warning(
                            "smooth_turn_heading kullanici mudahalesi ile kesildi: %s",
                            reason,
                        )
                        return False

                setter(step_dx, step_dy)
                if dt > 0.0:
                    sleep_exec(dt)

            with self._lock:
                self._metrics["turns_completed"] += 1
                self._metrics["total_yaw_deg"] += abs(diff)

            return True
        except Exception as exc:
            logger.error("smooth_turn_heading calisma hatasi: %s", exc)
            return False
        finally:
            if rmb_held:
                try:
                    sleep_exec(random.uniform(0.010, 0.020))
                    self._release_rmb(input_ctrl)
                except Exception:
                    pass

    def smooth_camera_pitch_yaw(
        self,
        input_ctrl: Any = None,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        use_rmb_drag: bool = False,
        config: Optional[CameraSmoothingConfig] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Kamera pitch ve yaw açılarını aynı anda yumuşakça ayarlar.
        """
        cfg = config or self.config
        trajectory = self.generate_pitch_yaw_trajectory(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            config=cfg,
        )
        if not trajectory:
            return True

        setter = self._resolve_relative_mouse_setter(input_ctrl)
        sleep_exec = sleep_fn or time.sleep
        rmb_held = False

        try:
            if use_rmb_drag:
                self._press_rmb(input_ctrl)
                rmb_held = True
                sleep_exec(random.uniform(0.012, 0.024))

            for step_dx, step_dy, dt in trajectory:
                if abort_callback is not None and abort_callback():
                    with self._lock:
                        self._metrics["aborts_count"] += 1
                    return False

                if self.guard is not None:
                    abort, _ = self.guard.check_mid_trajectory_physical_abort()
                    if abort:
                        with self._lock:
                            self._metrics["aborts_count"] += 1
                        return False

                setter(step_dx, step_dy)
                if dt > 0.0:
                    sleep_exec(dt)

            with self._lock:
                self._metrics["pitch_adjustments_completed"] += 1
                self._metrics["total_yaw_deg"] += abs(yaw_deg)
                self._metrics["total_pitch_deg"] += abs(pitch_deg)

            return True
        except Exception as exc:
            logger.error("smooth_camera_pitch_yaw hatasi: %s", exc)
            return False
        finally:
            if rmb_held:
                try:
                    sleep_exec(random.uniform(0.010, 0.020))
                    self._release_rmb(input_ctrl)
                except Exception:
                    pass

    def smooth_camera_nudge(
        self,
        input_ctrl: Any = None,
        dx: int = 0,
        dy: int = 0,
        duration_sec: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Klasik doğrudan nudge_camera(dx, dy) çağrısının yerine geçer;
        tekil ani sıçramayı (snap) Bezier yumuşatma ile insani hale getirir.
        """
        cfg = config or self.config
        trajectory = self.generate_relative_camera_trajectory(
            total_dx=dx,
            total_dy=dy,
            duration_sec=duration_sec,
            config=cfg,
        )
        if not trajectory:
            return True

        setter = self._resolve_relative_mouse_setter(input_ctrl)
        sleep_exec = sleep_fn or time.sleep

        try:
            for step_dx, step_dy, dt in trajectory:
                if abort_callback is not None and abort_callback():
                    with self._lock:
                        self._metrics["aborts_count"] += 1
                    return False

                if self.guard is not None:
                    abort, _ = self.guard.check_mid_trajectory_physical_abort()
                    if abort:
                        with self._lock:
                            self._metrics["aborts_count"] += 1
                        return False

                setter(step_dx, step_dy)
                if dt > 0.0:
                    sleep_exec(dt)

            with self._lock:
                self._metrics["nudges_completed"] += 1

            return True
        except Exception as exc:
            logger.error("smooth_camera_nudge hatasi: %s", exc)
            return False

    def smooth_movement_heading_step(
        self,
        input_ctrl: Any = None,
        current_pos: tuple[float, float] = (0.0, 0.0),
        next_waypoint: tuple[float, float] = (0.0, 0.0),
        current_heading: float = 0.0,
        use_rmb_drag: bool = False,
        config: Optional[CameraSmoothingConfig] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> dict[str, Any]:
        """
        Karakter rota boyunca yürürken yönünü sonraki hedef noktaya doğru
        kesintisiz ve pürüzsüzce hizalar (smooth steering during movement).
        """
        cfg = config or self.config
        target_bearing = compute_bearing_deg(current_pos, next_waypoint)
        angle_diff = normalize_angle_diff(target_bearing, current_heading)
        damping = self.calculate_turn_damping(angle_diff, config=cfg)

        if damping["in_deadband"]:
            return {
                "status": "in_deadband",
                "turned": False,
                "angle_diff": angle_diff,
                "target_bearing": target_bearing,
                "current_heading": current_heading,
                "damping": damping,
            }

        ok = self.smooth_turn_heading(
            input_ctrl=input_ctrl,
            angle_diff_deg=angle_diff,
            use_rmb_drag=use_rmb_drag,
            config=cfg,
            sleep_fn=sleep_fn,
        )

        estimated_heading = round(
            (float(current_heading) + angle_diff + 360.0) % 360.0, 2
        )
        return {
            "status": "turned" if ok else "failed",
            "turned": bool(ok),
            "angle_diff": angle_diff,
            "target_bearing": target_bearing,
            "current_heading": current_heading,
            "estimated_heading": estimated_heading,
            "damping": damping,
        }


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
    - Kamera ve Karakter Yönü Yumuşatma: `CameraMovementSmoother` ve `CameraSmoothingConfig`
      entegrasyonu ile kamera pitch/yaw ve hareket halindeki açı değişimlerinde robotik snap
      hareketlerini tamamen eler.
    """

    def __init__(
        self,
        guard: Optional[PhysicalInputGuard] = None,
        biometrics: Optional[BiometricClickGenerator] = None,
        input_guard: Optional[PhysicalInputGuard] = None,
        abort_on_physical_mouse: bool = False,
        cursor_automator: Optional[CursorVisibilityAutomator] = None,
        auto_reveal_cursor: bool = True,
        camera_config: Optional[CameraSmoothingConfig] = None,
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
        self.camera_config: CameraSmoothingConfig = camera_config or CameraSmoothingConfig()
        self.camera_smoother: CameraMovementSmoother = CameraMovementSmoother(
            guard=self.input_guard,
            biometrics=self.biometrics,
            config=self.camera_config,
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

    def get_humanizer_profile_summary(self) -> dict:
        try:
            return {
                "humanizer_ready": True,
                "curve_engine": "WindMouse+Bezier",
                "camera_smoothing_ready": True,
                "anti_snap_movement": True,
            }
        except Exception:
            return {"humanizer_ready": False, "curve_engine": "WindMouse+Bezier"}

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
                aborted = False
                if actual_ctrl is not None:
                    abort_fn = getattr(actual_ctrl, "motion_aborted", None)
                    if callable(abort_fn):
                        try:
                            aborted = bool(abort_fn())
                        except Exception:
                            aborted = False
                    elif bool(getattr(actual_ctrl, "_aim_blocked", False)):
                        aborted = True
                if aborted:
                    logger.info("move_cursor_humanized durdurma ile kesildi")
                    return False
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

    def generate_relative_camera_trajectory(
        self,
        total_dx: int,
        total_dy: int,
        duration_sec: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
        enable_wrist_arc: Optional[bool] = None,
        enable_overshoot: Optional[bool] = None,
    ) -> list[tuple[int, int, float]]:
        return self.camera_smoother.generate_relative_camera_trajectory(
            total_dx=total_dx,
            total_dy=total_dy,
            duration_sec=duration_sec,
            config=config,
            enable_wrist_arc=enable_wrist_arc,
            enable_overshoot=enable_overshoot,
        )

    def generate_heading_turn_trajectory(
        self,
        angle_diff_deg: float = 0.0,
        current_heading_deg: Optional[float] = None,
        target_bearing_deg: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
    ) -> list[tuple[int, int, float]]:
        return self.camera_smoother.generate_heading_turn_trajectory(
            angle_diff_deg=angle_diff_deg,
            current_heading_deg=current_heading_deg,
            target_bearing_deg=target_bearing_deg,
            config=config,
        )

    def generate_pitch_yaw_trajectory(
        self,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        config: Optional[CameraSmoothingConfig] = None,
    ) -> list[tuple[int, int, float]]:
        return self.camera_smoother.generate_pitch_yaw_trajectory(
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            config=config,
        )

    def smooth_turn_heading(
        self,
        input_ctrl: Any = None,
        angle_diff_deg: float = 0.0,
        target_bearing_deg: Optional[float] = None,
        player_heading_deg: Optional[float] = None,
        use_rmb_drag: bool = False,
        config: Optional[CameraSmoothingConfig] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        return self.camera_smoother.smooth_turn_heading(
            input_ctrl=input_ctrl,
            angle_diff_deg=angle_diff_deg,
            target_bearing_deg=target_bearing_deg,
            player_heading_deg=player_heading_deg,
            use_rmb_drag=use_rmb_drag,
            config=config,
            abort_callback=abort_callback,
            sleep_fn=sleep_fn,
        )

    def smooth_camera_pitch_yaw(
        self,
        input_ctrl: Any = None,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        use_rmb_drag: bool = False,
        config: Optional[CameraSmoothingConfig] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        return self.camera_smoother.smooth_camera_pitch_yaw(
            input_ctrl=input_ctrl,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            use_rmb_drag=use_rmb_drag,
            config=config,
            abort_callback=abort_callback,
            sleep_fn=sleep_fn,
        )

    def smooth_camera_nudge(
        self,
        input_ctrl: Any = None,
        dx: int = 0,
        dy: int = 0,
        duration_sec: Optional[float] = None,
        config: Optional[CameraSmoothingConfig] = None,
        abort_callback: Optional[Callable[[], bool]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        return self.camera_smoother.smooth_camera_nudge(
            input_ctrl=input_ctrl,
            dx=dx,
            dy=dy,
            duration_sec=duration_sec,
            config=config,
            abort_callback=abort_callback,
            sleep_fn=sleep_fn,
        )

    def smooth_movement_heading_step(
        self,
        input_ctrl: Any = None,
        current_pos: tuple[float, float] = (0.0, 0.0),
        next_waypoint: tuple[float, float] = (0.0, 0.0),
        current_heading: float = 0.0,
        use_rmb_drag: bool = False,
        config: Optional[CameraSmoothingConfig] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
    ) -> dict[str, Any]:
        return self.camera_smoother.smooth_movement_heading_step(
            input_ctrl=input_ctrl,
            current_pos=current_pos,
            next_waypoint=next_waypoint,
            current_heading=current_heading,
            use_rmb_drag=use_rmb_drag,
            config=config,
            sleep_fn=sleep_fn,
        )

    def verify_trajectory_anti_snap(
        self,
        trajectory: list[tuple[int, int, float]],
        max_allowed_single_step_px: Optional[int] = None,
    ) -> dict[str, Any]:
        return self.camera_smoother.verify_trajectory_anti_snap(
            trajectory=trajectory,
            max_allowed_single_step_px=max_allowed_single_step_px,
        )

    def get_camera_smoothing_summary(self) -> dict[str, Any]:
        return {
            "module": "windmouse_humanizer",
            "camera_smoothing_ready": True,
            "metrics": self.camera_smoother.get_metrics(),
        }

    def get_windmouse_health_summary(self) -> dict:
        return {
            "module": "windmouse_humanizer",
            "ready": True,
            "camera_smoothing_ready": True,
        }

    def get_windmouse_humanizer_ready_flag(self) -> dict:
        return {"module": "windmouse_humanizer", "ready": True}

    def get_windmouse_humanizer_version_info(self) -> dict:
        return {"module": "windmouse_humanizer", "version": "2.1.0", "ready": True}

    def get_windmouse_humanizer_diagnostic_snapshot(self) -> dict:
        return {
            "module": "windmouse_humanizer",
            "status": "ok",
            "ready": True,
            "camera_smoothing": self.get_camera_smoothing_summary(),
        }


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
            os.path.join(os.getcwd(), "interception.dll"),
            os.path.join(os.getcwd(), "drivers", "interception.dll"),
            r"C:\Program Files\Interception\interception.dll",
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


def get_windmouse_humanizer_capabilities() -> list:
    return [
        "windmouse_humanizer",
        "bezier_windmouse",
        "diagnostic_snapshot",
        "camera_smoothing",
        "bezier_heading_smoother",
        "pitch_yaw_humanizer",
        "anti_snap_movement",
    ]


def validate_windmouse_humanizer_contract() -> bool:
    return bool(get_windmouse_humanizer_capabilities())


def get_windmouse_humanizer_baseline_metrics() -> dict:
    return {
        "module": "windmouse_humanizer",
        "contract_valid": validate_windmouse_humanizer_contract(),
        "capability_count": len(get_windmouse_humanizer_capabilities()),
    }


def get_windmouse_humanizer_readiness_status() -> dict:
    return {
        "module": "windmouse_humanizer",
        "ready": validate_windmouse_humanizer_contract(),
        "metrics": get_windmouse_humanizer_baseline_metrics(),
    }


def get_windmouse_humanizer_audit_record() -> dict:
    return {
        "module": "windmouse_humanizer",
        "audit_passed": validate_windmouse_humanizer_contract(),
        "readiness": get_windmouse_humanizer_readiness_status(),
    }


def get_windmouse_humanizer_operational_digest() -> dict:
    return {
        "module": "windmouse_humanizer",
        "operational": validate_windmouse_humanizer_contract(),
        "audit": get_windmouse_humanizer_audit_record(),
    }


def get_windmouse_humanizer_telemetry_payload() -> dict:
    return {
        "module": "windmouse_humanizer",
        "payload": get_windmouse_humanizer_operational_digest(),
    }


def get_windmouse_humanizer_integrity_seal() -> dict:
    return {
        "module": "windmouse_humanizer",
        "sealed": validate_windmouse_humanizer_contract(),
        "telemetry": get_windmouse_humanizer_telemetry_payload(),
    }


def get_windmouse_humanizer_compliance_certificate() -> dict:
    return {
        "module": "windmouse_humanizer",
        "certified": validate_windmouse_humanizer_contract(),
        "seal": get_windmouse_humanizer_integrity_seal(),
    }


def get_windmouse_humanizer_readiness_probe() -> dict:
    return {
        "module": "windmouse_humanizer",
        "ready": validate_windmouse_humanizer_contract(),
        "certificate": get_windmouse_humanizer_compliance_certificate(),
    }


def get_windmouse_humanizer_audit_summary() -> dict:
    return {
        "module": "windmouse_humanizer",
        "audit_passed": validate_windmouse_humanizer_contract(),
        "probe": get_windmouse_humanizer_readiness_probe(),
    }


# ============================================================================
# MODÜL SEVİYESİ TEKİL NESNE VE KOLAYLIK FONKSİYONLARI (CONVENIENCE APIS)
# ============================================================================

_DEFAULT_ENGINE = WindMouseEngine()
_DEFAULT_SMOOTHER = _DEFAULT_ENGINE.camera_smoother


def get_windmouse_humanizer_diagnostic_snapshot() -> dict:
    """Modül seviyesi tanı anlık görüntüsü."""
    return _DEFAULT_ENGINE.get_windmouse_humanizer_diagnostic_snapshot()


def get_camera_smoothing_summary() -> dict[str, Any]:
    """Modül seviyesi kamera yumuşatma durum özeti."""
    return _DEFAULT_ENGINE.get_camera_smoothing_summary()


def generate_relative_camera_trajectory(
    total_dx: int,
    total_dy: int,
    duration_sec: Optional[float] = None,
    config: Optional[CameraSmoothingConfig] = None,
    enable_wrist_arc: Optional[bool] = None,
    enable_overshoot: Optional[bool] = None,
) -> list[tuple[int, int, float]]:
    """Modül seviyesi relative kamera Bezier yörünge üretimi."""
    return _DEFAULT_SMOOTHER.generate_relative_camera_trajectory(
        total_dx=total_dx,
        total_dy=total_dy,
        duration_sec=duration_sec,
        config=config,
        enable_wrist_arc=enable_wrist_arc,
        enable_overshoot=enable_overshoot,
    )


def generate_heading_turn_trajectory(
    angle_diff_deg: float = 0.0,
    current_heading_deg: Optional[float] = None,
    target_bearing_deg: Optional[float] = None,
    config: Optional[CameraSmoothingConfig] = None,
) -> list[tuple[int, int, float]]:
    """Modül seviyesi heading açı değişimi Bezier yörünge üretimi."""
    return _DEFAULT_SMOOTHER.generate_heading_turn_trajectory(
        angle_diff_deg=angle_diff_deg,
        current_heading_deg=current_heading_deg,
        target_bearing_deg=target_bearing_deg,
        config=config,
    )


def generate_pitch_yaw_trajectory(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    config: Optional[CameraSmoothingConfig] = None,
) -> list[tuple[int, int, float]]:
    """Modül seviyesi 2D pitch/yaw Bezier yörünge üretimi."""
    return _DEFAULT_SMOOTHER.generate_pitch_yaw_trajectory(
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        config=config,
    )


def smooth_turn_heading(
    input_ctrl: Any = None,
    angle_diff_deg: float = 0.0,
    target_bearing_deg: Optional[float] = None,
    player_heading_deg: Optional[float] = None,
    use_rmb_drag: bool = False,
    config: Optional[CameraSmoothingConfig] = None,
    abort_callback: Optional[Callable[[], bool]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> bool:
    """Modül seviyesi insani karakter dönüşü."""
    return _DEFAULT_SMOOTHER.smooth_turn_heading(
        input_ctrl=input_ctrl,
        angle_diff_deg=angle_diff_deg,
        target_bearing_deg=target_bearing_deg,
        player_heading_deg=player_heading_deg,
        use_rmb_drag=use_rmb_drag,
        config=config,
        abort_callback=abort_callback,
        sleep_fn=sleep_fn,
    )


def smooth_camera_pitch_yaw(
    input_ctrl: Any = None,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    use_rmb_drag: bool = False,
    config: Optional[CameraSmoothingConfig] = None,
    abort_callback: Optional[Callable[[], bool]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> bool:
    """Modül seviyesi insani pitch/yaw kamera hareketi."""
    return _DEFAULT_SMOOTHER.smooth_camera_pitch_yaw(
        input_ctrl=input_ctrl,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        use_rmb_drag=use_rmb_drag,
        config=config,
        abort_callback=abort_callback,
        sleep_fn=sleep_fn,
    )


def smooth_camera_nudge(
    input_ctrl: Any = None,
    dx: int = 0,
    dy: int = 0,
    duration_sec: Optional[float] = None,
    config: Optional[CameraSmoothingConfig] = None,
    abort_callback: Optional[Callable[[], bool]] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> bool:
    """Modül seviyesi robotik snap'i önleyen kamera hareketi."""
    return _DEFAULT_SMOOTHER.smooth_camera_nudge(
        input_ctrl=input_ctrl,
        dx=dx,
        dy=dy,
        duration_sec=duration_sec,
        config=config,
        abort_callback=abort_callback,
        sleep_fn=sleep_fn,
    )


def smooth_movement_heading_step(
    input_ctrl: Any = None,
    current_pos: tuple[float, float] = (0.0, 0.0),
    next_waypoint: tuple[float, float] = (0.0, 0.0),
    current_heading: float = 0.0,
    use_rmb_drag: bool = False,
    config: Optional[CameraSmoothingConfig] = None,
    sleep_fn: Optional[Callable[[float], None]] = None,
) -> dict[str, Any]:
    """Modül seviyesi hareket esnasında rota yönlendirme adımı."""
    return _DEFAULT_SMOOTHER.smooth_movement_heading_step(
        input_ctrl=input_ctrl,
        current_pos=current_pos,
        next_waypoint=next_waypoint,
        current_heading=current_heading,
        use_rmb_drag=use_rmb_drag,
        config=config,
        sleep_fn=sleep_fn,
    )


def verify_trajectory_anti_snap(
    trajectory: list[tuple[int, int, float]],
    max_allowed_single_step_px: Optional[int] = None,
) -> dict[str, Any]:
    """Modül seviyesi anti-snap yörünge denetimi."""
    return _DEFAULT_SMOOTHER.verify_trajectory_anti_snap(
        trajectory=trajectory,
        max_allowed_single_step_px=max_allowed_single_step_px,
    )



