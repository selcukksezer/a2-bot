"""
WindMouse / Cubic Bezier Fare İvmesi, 2D Gaussian Sapma ve Fiziksel Çakışma Korumalı Girdi Motoru.

Harici Arduino donanımı gerektirmeden kullanıcının kendi bilgisayarında yerel olarak çalışan,
insan biyometrisini (Fitts's Law ease-in-out ivmelenme, WindMouse stokastik el titremesi,
Gaussian tuş/tıklama basılı tutma süreleri ve 2D uzamsal koordinat sapması) birebir simüle eden
ve fiziksel kullanıcı müdahalesini anında algılayarak çakışmayı önleyen modül.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import random
import sys
import threading
import time
from typing import Any, Callable, Optional, Union

from core.stability_guards import (
    CameraLockExceptionRule,
    CursorVisibilityAutomator,
    SelfInputWhitelistWindow,
)
from version import __version__

logger = logging.getLogger("core.windmouse_humanizer")
if logger.handlers:
    logger.propagate = False


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def _get_cursor_pos() -> tuple[int, int]:
    """Win32 GetCursorPos çağrısı ile mevcut ekran imleç koordinatlarını döndürür.

    S-1405: Windows'ta okuma başarısız olursa (0, 0) DÖNDÜRMEZ, hata fırlatır —
    (0, 0) gerçek bir imleç konumu da olabileceğinden sessiz (0, 0) dönüşü
    guard'ın beklenen/gerçek karşılaştırmasını 0 delta ile geçiştirip
    fail-open bırakıyordu. Çağıran PhysicalInputGuard._safe_get_cursor hatayı
    yakalar, okuma-basarısız bayrağını kurar ve denetimler fail-CLOSED
    davranır. Windows-dışı platformda (Linux/macOS) Win32 API hiç yoktur;
    orada eski (0, 0) yedeği korunur (geliştirme/test ortamı; üretim
    Windows'ta çalışır).
    """
    if not sys.platform.startswith("win"):
        return 0, 0
    try:
        pt = _POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return int(pt.x), int(pt.y)
    except Exception as exc:
        logger.debug("GetCursorPos hatasi: %s", exc)
        raise RuntimeError(f"GetCursorPos okunamadi: {exc}") from exc
    raise RuntimeError("GetCursorPos basarisiz dondu")


def _get_screen_size() -> tuple[int, int]:
    """Ekran boyutunu döndürür; okunamazsa 1920x1080 varsayar (clamp için)."""
    try:
        sw = int(ctypes.windll.user32.GetSystemMetrics(0))
        sh = int(ctypes.windll.user32.GetSystemMetrics(1))
        if sw > 0 and sh > 0:
            return sw, sh
    except Exception as exc:
        logger.debug("GetSystemMetrics hatasi: %s", exc)
    return 1920, 1080


# Ekran boyutu her yörünge adımında (guard denetimi + clamp) yeniden
# Win32'ye soruluyordu; çözünürlük saniyeler içinde değişmez, o yüzden
# kısa TTL'li önbellek yeterli. TTL dolunca yeniden prob edilir.
_SCREEN_SIZE_CACHE_TTL_SEC: float = 5.0
_SCREEN_SIZE_CACHE: dict[str, Any] = {"value": None, "ts": 0.0}
_SCREEN_SIZE_CACHE_LOCK = threading.Lock()


def _get_screen_size_cached() -> tuple[int, int]:
    """TTL'li ekran boyutu: TTL içinde ham probu tekrarlamaz."""
    now = time.monotonic()
    with _SCREEN_SIZE_CACHE_LOCK:
        cached = _SCREEN_SIZE_CACHE["value"]
        if cached is not None and (now - _SCREEN_SIZE_CACHE["ts"]) < _SCREEN_SIZE_CACHE_TTL_SEC:
            return cached
    fresh = _get_screen_size()
    with _SCREEN_SIZE_CACHE_LOCK:
        _SCREEN_SIZE_CACHE["value"] = fresh
        _SCREEN_SIZE_CACHE["ts"] = time.monotonic()
    return fresh


def _clamp_to_screen(x: int, y: int) -> tuple[int, int]:
    """Koordinatı ekran sınırları içine çeker (jitter sonrası taşmayı önler)."""
    sw, sh = _get_screen_size_cached()
    return max(0, min(sw - 1, int(x))), max(0, min(sh - 1, int(y)))


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

    @staticmethod
    def _sample_gaussian_dwell(
        mean_sec: float,
        sigma_sec: float,
        min_sec: float,
        max_sec: float,
        label: str,
    ) -> float:
        """
        Ortalama `mean_sec`, sapma `sigma_sec` ile [min_sec, max_sec] araliginda
        Gaussian basili tutma suresi ornekler: 8 denemede aralik tutmazsa son
        ornegi araliga kirpar; hata durumunda ortalamayi dondurur.
        get_key_dwell_sec / get_mouse_dwell_sec ortak yardimcisidir.
        """
        try:
            for _ in range(8):
                val = random.gauss(mean_sec, sigma_sec)
                if min_sec <= val <= max_sec:
                    return round(val, 6)
            clamped = max(
                min_sec,
                min(max_sec, random.gauss(mean_sec, sigma_sec)),
            )
            return round(clamped, 6)
        except Exception as exc:
            logger.error("%s hesaplama hatasi: %s", label, exc)
            return float(mean_sec)

    def get_key_dwell_sec(self) -> float:
        """
        F veya M gibi tuş basımlarında KeyDown ile KeyUp arasında
        45 ms ile 115 ms (0.045s - 0.115s, ortalama 76ms, sigma=14ms)
        arasında Gaussian dağılımlı süre döndürür.
        """
        return self._sample_gaussian_dwell(
            self.key_mean_sec,
            self.key_sigma_sec,
            self.key_min_sec,
            self.key_max_sec,
            "get_key_dwell_sec",
        )

    def get_mouse_dwell_sec(self) -> float:
        """
        Sol/sağ fare tıklamalarında MouseDown ile MouseUp arasında
        55 ms ile 130 ms (0.055s - 0.130s, ortalama 85ms, sigma=16ms)
        arasında Gaussian dağılımlı süre döndürür.
        """
        return self._sample_gaussian_dwell(
            self.mouse_mean_sec,
            self.mouse_sigma_sec,
            self.mouse_min_sec,
            self.mouse_max_sec,
            "get_mouse_dwell_sec",
        )

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
        # S-1405: son imleç okumasının başarı durumu — _safe_get_cursor kurar.
        # Okuma başarısızsa beklenen konum (0, 0) ile çapa ATILMAZ.
        self._last_cursor_read_ok: bool = True
        self._last_cursor_read_error: str = ""
        cur_x, cur_y = self._safe_get_cursor()
        self._expected_x: int = cur_x
        self._expected_y: int = cur_y
        self._has_expected_anchor: bool = bool(self._last_cursor_read_ok)
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
        """İmleç konumunu okur; okuma başarısızsa (0, 0) yedeği döner AMA
        S-1405 gereği başarısızlığı _last_cursor_read_ok=False ile işaretler —
        denetim yapanlar (should_abort_input / check_mid_trajectory_...) bu
        bayrağı görüp fail-CLOSED (iptal) döner, (0, 0) çapası kurulmaz.
        Dönüş imzası korunur: yalnızca koordinata ihtiyacı olan çağıranlar
        değişmeden çalışmaya devam eder.
        """
        try:
            pos = self._cursor_getter()
            self._last_cursor_read_ok = True
            self._last_cursor_read_error = ""
            return int(pos[0]), int(pos[1])
        except Exception as exc:
            self._last_cursor_read_ok = False
            self._last_cursor_read_error = str(exc)
            logger.debug("PhysicalInputGuard imlec okuma hatasi: %s", exc)
            return 0, 0

    def _ignored_vks_snapshot(self, now: float) -> Optional[set[int]]:
        """
        O an beyaz liste penceresinde (varsayilan 250 ms) olan bot VK'larini
        tek seferde dondurur; hizli yol mumkun degilse None doner ve cagiran
        VK basina is_within_ignore_window cagrisina duser.

        Semantik: yalnizca `_vk_timestamps` okunur. Botun kendi kaydettigi
        her tus (int VK ya da VK'si cozulen isim) oraya da yazildigindan,
        hizli yolun "yoksay" dedigi her VK yavas yolun da yoksayacagi
        VK'dir (fail-open yok). Ters yonde sapma yalnizca sayisal-string
        ("82" gibi) kayitlarda olur — o durumda yavas yola dusulmez,
        fiziksel girdi sanilip fail-CLOSED iptal olur (guvenli yon).
        Herhangi bir suphede None doner.
        """
        try:
            whitelist = self.self_input_whitelist
            vk_ts = getattr(whitelist, "_vk_timestamps", None)
            if not isinstance(vk_ts, dict):
                return None
            window_sec = max(
                0.0, float(getattr(whitelist, "default_window_ms", 250.0)) / 1000.0
            )
            lock = getattr(whitelist, "_lock", None)
            if lock is not None:
                with lock:
                    items = list(vk_ts.items())
            else:
                items = list(vk_ts.items())
        except Exception:
            return None
        ignored: set[int] = set()
        for vk, ts in items:
            try:
                if isinstance(vk, int) and (float(now) - float(ts)) <= window_sec:
                    ignored.add(int(vk))
            except Exception:
                continue
        return ignored

    def _is_self_key_ignored(
        self, vk: int, now: float, ignored_vks: Optional[set[int]]
    ) -> bool:
        """VK botun kendi beyaz listedeki tusuysa True; hizli yol yoksa VK basina sorar."""
        if vk in self._bot_active_vks:
            return True
        if ignored_vks is not None:
            return vk in ignored_vks
        return bool(self.self_input_whitelist.is_within_ignore_window(vk, now=now))

    @classmethod
    def _get_screen_centers(cls) -> list[tuple[int, int]]:
        centers: list[tuple[int, int]] = list(cls.GAME_CENTER_SNAP_POINTS)
        # Dinamik merkez TTL'li ekran boyutundan gelir — her guard adımında
        # ayrı Win32 probu yapılmaz.
        sw, sh = _get_screen_size_cached()
        if sw > 0 and sh > 0:
            dyn_center = (sw // 2, sh // 2)
            if dyn_center not in centers:
                centers.append(dyn_center)
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
            if not self._last_cursor_read_ok:
                # S-1405: okuma başarısızsa (0, 0) yedeğini çapa yapma;
                # mevcut beklenen konumu koru, çağıran eski çapayı alır.
                return self._expected_x, self._expected_y
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
            if not self._last_cursor_read_ok:
                # S-1405: okuma başarısızsa (0, 0) yedeğini çapa yapma;
                # çapa bayrağını düşür ki sonraki denetim fail-CLOSED kalsın.
                self._has_expected_anchor = False
                return
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

                # S-1405: imleç okunamıyorsa fail-CLOSED — denetim yapılamıyorsa
                # girdi durdurulur; (0, 0) yedeğiyle delta=0 sayıp devam etmek
                # (fail-open) fiziksel müdahaleyi kaçırır (F-1697 ile tutarlı).
                if not self._last_cursor_read_ok:
                    reason = (
                        "cursor_read_failed_fail_closed: "
                        f"{self._last_cursor_read_error or 'GetCursorPos okunamadi'}"
                    )
                    self._last_conflict_ts = now
                    self._last_conflict_reason = reason
                    logger.warning("Imlec okunamadigi icin girdi iptal: %s", reason)
                    return True, reason

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

                # Beyaz liste tek anlik goruntuyle sorulur (VK basina 28 ayri
                # prob yerine); hizli yol yoksa _is_self_key_ignored yavas yola duser.
                ignored_vks = self._ignored_vks_snapshot(now)
                for vk in self.MONITORED_VK_CODES:
                    if self._is_self_key_ignored(vk, now, ignored_vks):
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
            # F-1697: guard çökerse fail-CLOSED — denetim yapılamıyorsa girdi
            # durdurulur; "devam et" (fail-open) fiziksel müdahaleyi kaçırır.
            logger.error("should_abort_input denetim hatasi: %s", exc)
            return True, f"guard_exception_fail_closed: {exc}"

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

                # S-1405: imleç okunamıyorsa fail-CLOSED — (0, 0) yedeğiyle
                # merkez/sürükleme karşılaştırması yapmak fail-open olur.
                if not self._last_cursor_read_ok:
                    reason = (
                        "cursor_read_failed_fail_closed: "
                        f"{self._last_cursor_read_error or 'GetCursorPos okunamadi'}"
                    )
                    self._last_conflict_ts = now
                    self._last_conflict_reason = reason
                    return True, reason

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

                ignored_vks = self._ignored_vks_snapshot(now)
                for vk in self.PHYSICAL_MOUSE_BUTTON_VKS + self.MONITORED_VK_CODES:
                    if self._is_self_key_ignored(vk, now, ignored_vks):
                        continue
                    state = self._key_state_getter(vk)
                    if state & 0x8000:
                        reason = f"physical_button_override_detected: vk=0x{vk:02X}"
                        self._last_conflict_ts = now
                        self._last_conflict_reason = reason
                        return True, reason

                return False, "ok"
        except Exception as exc:
            # F-1697 ile tutarlı: guard çökerse fail-CLOSED.
            logger.debug("check_mid_trajectory_physical_abort hatasi: %s", exc)
            return True, f"guard_exception_fail_closed: {exc}"

def quintic_ease_in_out(t: float) -> float:
    """
    Minimum-jerk quintic polynomial (Perlin smoothstep 2nd order):
    S(t) = 6t^5 - 15t^4 + 10t^3 = t^3 * (t * (6t - 15) + 10).
    Guarantees zero velocity and zero acceleration at t=0 and t=1,
    eliminating robotic snap movements and sudden acceleration spikes.
    """
    t_clamped = max(0.0, min(1.0, float(t)))
    return t_clamped * t_clamped * t_clamped * (t_clamped * (t_clamped * 6.0 - 15.0) + 10.0)


def _split_step(
    dx: int, dy: int, dt: float, parts: int
) -> list[tuple[int, int, float]]:
    """F-1301: tek bir (dx, dy, dt) adımını tamsayı toplamını koruyarak
    `parts` eşit alt adıma böler. divmod kalanı ilk parçalara dağıtılır
    (negatif değerlerde de toplam korunur); süre parçalara eşit bölünür."""
    parts = max(1, int(parts))

    def _split_int(value: int) -> list[int]:
        base, rem = divmod(int(value), parts)
        return [base + 1 if i < rem else base for i in range(parts)]

    xs = _split_int(dx)
    ys = _split_int(dy)
    sub_dt = round(float(dt) / parts, 6)
    return [(xs[i], ys[i], sub_dt) for i in range(parts)]


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
        # F-1301: eski formül (ortalama adım <= cap * 0.55) tepe adımı hesaba
        # katmıyordu — quintic ease-in-out'un tepe türevi ortalama adımın
        # ~1.875 katıdır; overshoot varsa ana yolculuk 0.82'lik faza sıkışır
        # (x1/0.82) ve hedef %4'e kadar aşılır (x1.04). 360° (1440px) dönüşte
        # bu, ~32px'lik cap-aşan (cap=24px) adımlar üretiyordu. Tepe faktörü
        # 1.875 / 0.82 * (1 + max_overshoot_orani) ~= 2.38; bilek kavis /
        # tremor payıyla 2.5 alınır ve cap ASLA kalkmaz — dönüş, cap'e uygun
        # küçük adımlara bölünür (Karar 8).
        max_step_cap = max(8, cfg.max_single_step_px)
        ov_max_ratio = float(cfg.overshoot_ratio_range[1])
        peak_factor = 1.875 / 0.82 * (1.0 + ov_max_ratio) * 1.05
        n_steps = max(
            n_steps, int(math.ceil(dist * peak_factor / max_step_cap))
        )
        n_steps = min(400, max(4, n_steps))

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

        # F-1301 güvenlik ağı: rastgele tremor / bilek kavis payı da dahil
        # hiçbir adım cap'i aşamaz. Cap'i aşan adım, tamsayı toplamı ve
        # toplam süreyi koruyarak eşit alt adımlara bölünür (360° dönüş
        # dahil). Bölme, yörünge toplamını ve ivme profilini değiştirmez.
        capped_steps: list[tuple[int, int, float]] = []
        for sdx, sdy, sdt in steps:
            if math.hypot(sdx, sdy) <= max_step_cap:
                capped_steps.append((sdx, sdy, sdt))
                continue
            parts = int(math.ceil(math.hypot(sdx, sdy) / max_step_cap))
            for _ in range(4):
                split = _split_step(sdx, sdy, sdt, parts)
                if all(math.hypot(a, b) <= max_step_cap for a, b, _ in split):
                    break
                parts += 1
            else:
                split = _split_step(sdx, sdy, sdt, parts)
            capped_steps.extend(split)
        return capped_steps

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
            # F-1706: boş yörünge denetimden geçemez — "pürüzsüz" sayılmaz.
            return {
                "is_smooth": False,
                "is_robotic_snap": False,
                "step_count": 0,
                "total_dx": 0,
                "total_dy": 0,
                "max_single_step": 0.0,
                "has_acceleration": False,
                "has_deceleration": False,
                "reason": "empty_trajectory",
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
        if n >= 6:
            early_slice = speeds[: max(1, n // 4)]
            late_slice = speeds[max(0, 3 * n // 4) :]
            early_avg = sum(early_slice) / len(early_slice) if early_slice else 0.0
            late_avg = sum(late_slice) / len(late_slice) if late_slice else 0.0
            has_accel = bool(early_avg < peak_speed * 0.90)
            has_decel = bool(late_avg < peak_speed * 0.90)
        elif n >= 3:
            # F-1706: kısa yörüngede de profil gerçekten ölçülür — varsayılan True yok.
            has_accel = bool(speeds[0] < peak_speed * 0.90)
            has_decel = bool(speeds[-1] < peak_speed * 0.90)
        else:
            # F-1706: 1-2 adımlık yörüngede ivme profili kurulamaz — doğrulanamaz.
            has_accel = False
            has_decel = False

        is_smooth = bool(
            n >= 3 and not is_snap and max_step <= max_allowed and has_accel and has_decel
        )
        reason = ""
        if n < 3:
            reason = "too_short_to_verify"
        elif is_snap:
            reason = "robotic_snap"
        elif max_step > max_allowed:
            reason = "single_step_exceeds_cap"
        elif not (has_accel and has_decel):
            reason = "missing_acceleration_profile"

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
            "reason": reason,
        }

    def _resolve_relative_mouse_setter(
        self, input_ctrl: Any
    ) -> Callable[[int, int], Any]:
        """Girdi sürücüsündeki uygun relative fare hareket metodunu çözer.

        F-1693 / Karar 4: YALNIZCA donanım zincirli göreli-hareket metotları
        çözülür — önce `mouse_move_relative` (Interception tabanlı,
        core/input_driver.py'ye R29 ekler; donanım yoksa False döner, soft
        SendInput kullanmaz), sonra diğer donanım adlandırmaları, en sonda
        `nudge_camera`. S-2901 (F2-22) ile gerçek sürücünün `nudge_camera`
        gövdesi artık doğrudan `mouse_move_relative`'e (Interception
        donanım HID) delege ediyor — yani `nudge_camera`'yı çözmek yumuşak
        SendInput yoluna düşmek DEĞİLDİR; yalnızca `mouse_move_relative`
        sunmayan (ama aynı donanım yolunu `nudge_camera` adıyla sunan)
        sürücüleri de kabul eder. `mouse_move_relative` her zaman önce
        çözülür; donanım metodu yoksa fallback her zaman False döndürür
        ve çağıranlar hareketi başarısız sayar — sessiz soft düşüş YOKTUR.
        """
        if input_ctrl is not None:
            for attr in (
                "mouse_move_relative",
                "move_relative",
                "move_rel",
                "_relative_mouse_move",
                "nudge_camera",
            ):
                fn = getattr(input_ctrl, attr, None)
                if callable(fn):
                    return fn

        def _no_hardware_rel_move(dx: int, dy: int) -> bool:
            logger.warning(
                "Relative fare hareketi icin donanim zinciri yok "
                "(input_ctrl=%s; mouse_move_relative/nudge_camera "
                "bulunamadi); soft yola dusulmedi, "
                "hareket yapilmadi (dx=%d, dy=%d)",
                type(input_ctrl).__name__ if input_ctrl is not None else None,
                int(dx),
                int(dy),
            )
            return False

        return _no_hardware_rel_move

    def _guard_should_abort(self) -> tuple[bool, str]:
        """F-1696: guard varsa should_abort_input çağrılır; guard yoksa engel yok."""
        if self.guard is None:
            return False, "no_guard"
        return self.guard.should_abort_input()

    @staticmethod
    def _call_mouse_button_method(
        input_ctrl: Any, attrs: tuple[str, ...], button: str, vk: int
    ) -> bool:
        """input_ctrl'deki fare buton metodunu dener; dönüş değerini denetler.

        F-1694: yalnızca açık False başarısızlıktır (sonraki metoda geçilir);
        True başarıdır; None "donanım zincirine istisnasız iletildi" sayılır
        (örn. InputDriver._hw_mouse_down None döndürür) — ama hiçbir durumda
        yumuşak Win32 mouse_event fallback'ine düşülmez.
        """
        if input_ctrl is None:
            return False
        for attr in attrs:
            fn = getattr(input_ctrl, attr, None)
            if not callable(fn):
                continue
            try:
                result = fn(button)
            except TypeError:
                try:
                    result = fn(vk)
                except Exception as exc:
                    logger.debug("Fare buton metodu %s hatasi: %s", attr, exc)
                    continue
            except Exception as exc:
                logger.debug("Fare buton metodu %s hatasi: %s", attr, exc)
                continue
            if result is False:
                continue
            return True
        return False

    def _press_rmb(self, input_ctrl: Any) -> bool:
        """Sağ fare tuşunu donanım zinciri üzerinden basılı tutar (RMB down).

        F-1694: donanım yolu yoksa False döndürür — sessiz yumuşak fallback yok.
        Başarılı basış guard'a bildirilir ki botun kendi RMB'si fiziksel
        müdahale sanılıp hareket yarıda kesilmesin (F-1696).
        """
        ok = self._call_mouse_button_method(
            input_ctrl, ("_hw_mouse_down", "mouse_down", "press_mouse_button"), "right", 0x02
        )
        if not ok:
            logger.error("RMB basilamadi: donanim fare yolu yok (input_ctrl=%s)",
                         type(input_ctrl).__name__ if input_ctrl is not None else None)
            return False
        if self.guard is not None:
            self.guard.register_bot_key_press(0x02, pressed=True)
        return True

    def _release_rmb(self, input_ctrl: Any) -> bool:
        """Sağ fare tuşunu donanım zinciri üzerinden serbest bırakır (RMB up).

        F-1694: donanım yolu yoksa False döndürür. Bırakma güvenlik yoludur:
        fare bırakılırken basılı olabilecek R/T skill tuşları da (varsa
        key_up üzerinden) bırakılır — bırakma yolunda tuş asılı kalmaz.
        """
        ok = self._call_mouse_button_method(
            input_ctrl, ("_hw_mouse_up", "mouse_up", "release_mouse_button"), "right", 0x02
        )
        if self.guard is not None:
            self.guard.register_bot_key_press(0x02, pressed=False)
        # Güvenlik: basılı kalmış olabilecek R/T skill tuşlarını da bırak.
        if input_ctrl is not None:
            key_up = getattr(input_ctrl, "key_up", None)
            if callable(key_up):
                for skill_key in ("r", "t"):
                    try:
                        key_up(skill_key)
                    except Exception as exc:
                        logger.debug("Skill tusu birakma hatasi (%s): %s", skill_key, exc)
        if not ok:
            logger.error("RMB birakilamadi: donanim fare yolu yok (input_ctrl=%s)",
                         type(input_ctrl).__name__ if input_ctrl is not None else None)
            return False
        return True

    def _execute_trajectory_steps(
        self,
        setter: Callable[[int, int], Any],
        trajectory: list[tuple[int, int, float]],
        abort_callback: Optional[Callable[[], bool]],
        sleep_exec: Callable[[float], None],
        op_name: str,
        log_aborts: bool,
    ) -> bool:
        """
        smooth_turn_heading / smooth_camera_pitch_yaw / smooth_camera_nudge
        ortak adım döngüsü: abort_callback denetimi, yörünge-ortası fiziksel
        müdahale denetimi, göreli setter çağrısı (açık False = başarısız)
        ve adım beklemesi. Her iptal yolunda aborts_count bir artar.
        log_aborts=True yalnızca smooth_turn_heading'in mevcut log satırlarını
        üretir (diğer iki metodun sessiz iptal davranışı korunur).
        """
        for step_dx, step_dy, dt in trajectory:
            if abort_callback is not None and abort_callback():
                with self._lock:
                    self._metrics["aborts_count"] += 1
                if log_aborts:
                    logger.info("%s abort_callback ile kesildi", op_name)
                return False

            if self.guard is not None:
                abort, reason = self.guard.check_mid_trajectory_physical_abort(
                    massive_drag_threshold_px=450.0,
                )
                if abort:
                    with self._lock:
                        self._metrics["aborts_count"] += 1
                    if log_aborts:
                        logger.warning(
                            "%s kullanici mudahalesi ile kesildi: %s",
                            op_name,
                            reason,
                        )
                    return False

            # F-1693: setter'ın açık False dönüşü başarısız harekettir —
            # sessizce "dönüldü" sayılmaz, hareket durdurulur.
            if setter(step_dx, step_dy) is False:
                with self._lock:
                    self._metrics["aborts_count"] += 1
                logger.error(
                    "%s relative hareket reddedildi "
                    "(adim dx=%d, dy=%d); donanim zinciri yok",
                    op_name,
                    step_dx,
                    step_dy,
                )
                return False
            if dt > 0.0:
                sleep_exec(dt)

        return True

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

        # F-1696: başlamadan önce fiziksel müdahale denetimi (should_abort_input).
        abort, reason = self._guard_should_abort()
        if abort:
            with self._lock:
                self._metrics["aborts_count"] += 1
            logger.warning("smooth_turn_heading baslamadan iptal: %s", reason)
            return False

        setter = self._resolve_relative_mouse_setter(input_ctrl)
        sleep_exec = sleep_fn or time.sleep
        rmb_held = False
        # S-1406: RMB bırakma başarısızlığı dönüş değerine yansır — bırakma
        # başarısızken "dönüş tamamlandı" (True) dönülmez.
        rmb_release_failed = False
        outcome = False

        try:
            if use_rmb_drag:
                # F-1694: RMB basılamazsa sürükleme hiç başlamaz.
                if not self._press_rmb(input_ctrl):
                    return False
                rmb_held = True
                sleep_exec(random.uniform(0.012, 0.024))

            if not self._execute_trajectory_steps(
                setter,
                trajectory,
                abort_callback,
                sleep_exec,
                "smooth_turn_heading",
                log_aborts=True,
            ):
                return False

            with self._lock:
                self._metrics["turns_completed"] += 1
                self._metrics["total_yaw_deg"] += abs(diff)

            outcome = True
        except Exception as exc:
            logger.error("smooth_turn_heading calisma hatasi: %s", exc)
            outcome = False
        finally:
            if rmb_held:
                try:
                    sleep_exec(random.uniform(0.010, 0.020))
                except Exception:
                    pass
                try:
                    if not self._release_rmb(input_ctrl):
                        logger.error("smooth_turn_heading: RMB birakilamadi (asili kalabilir)")
                        rmb_release_failed = True
                except Exception as exc:
                    logger.error("smooth_turn_heading: RMB birakma hatasi: %s", exc)
                    rmb_release_failed = True
        if rmb_release_failed:
            logger.error(
                "smooth_turn_heading: RMB birakilamadigi icin donus "
                "basarisiz sayildi (tuslar/fare asili kalabilir)"
            )
            return False
        return outcome

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

        # F-1696: başlamadan önce fiziksel müdahale denetimi (should_abort_input).
        abort, reason = self._guard_should_abort()
        if abort:
            with self._lock:
                self._metrics["aborts_count"] += 1
            logger.warning("smooth_camera_pitch_yaw baslamadan iptal: %s", reason)
            return False

        setter = self._resolve_relative_mouse_setter(input_ctrl)
        sleep_exec = sleep_fn or time.sleep
        rmb_held = False
        # S-1406: RMB bırakma başarısızlığı dönüş değerine yansır — bırakma
        # başarısızken "ayar tamamlandı" (True) dönülmez.
        rmb_release_failed = False
        outcome = False

        try:
            if use_rmb_drag:
                # F-1694: RMB basılamazsa sürükleme hiç başlamaz.
                if not self._press_rmb(input_ctrl):
                    return False
                rmb_held = True
                sleep_exec(random.uniform(0.012, 0.024))

            if not self._execute_trajectory_steps(
                setter,
                trajectory,
                abort_callback,
                sleep_exec,
                "smooth_camera_pitch_yaw",
                log_aborts=False,
            ):
                return False

            with self._lock:
                self._metrics["pitch_adjustments_completed"] += 1
                self._metrics["total_yaw_deg"] += abs(yaw_deg)
                self._metrics["total_pitch_deg"] += abs(pitch_deg)

            outcome = True
        except Exception as exc:
            logger.error("smooth_camera_pitch_yaw hatasi: %s", exc)
            outcome = False
        finally:
            if rmb_held:
                try:
                    sleep_exec(random.uniform(0.010, 0.020))
                except Exception:
                    pass
                try:
                    if not self._release_rmb(input_ctrl):
                        logger.error("smooth_camera_pitch_yaw: RMB birakilamadi (asili kalabilir)")
                        rmb_release_failed = True
                except Exception as exc:
                    logger.error("smooth_camera_pitch_yaw: RMB birakma hatasi: %s", exc)
                    rmb_release_failed = True
        if rmb_release_failed:
            logger.error(
                "smooth_camera_pitch_yaw: RMB birakilamadigi icin ayar "
                "basarisiz sayildi (tuslar/fare asili kalabilir)"
            )
            return False
        return outcome

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

        # F-1696: başlamadan önce fiziksel müdahale denetimi (should_abort_input).
        abort, reason = self._guard_should_abort()
        if abort:
            with self._lock:
                self._metrics["aborts_count"] += 1
            logger.warning("smooth_camera_nudge baslamadan iptal: %s", reason)
            return False

        setter = self._resolve_relative_mouse_setter(input_ctrl)
        sleep_exec = sleep_fn or time.sleep

        try:
            if not self._execute_trajectory_steps(
                setter,
                trajectory,
                abort_callback,
                sleep_exec,
                "smooth_camera_nudge",
                log_aborts=False,
            ):
                return False

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

        # F-1705: dönüş başarısızsa tahmini heading değişmemiş heading'dir —
        # başarısız dönüş "dönülmüş" gibi raporlanmaz.
        estimated_heading = round(
            (float(current_heading) + (angle_diff if ok else 0.0) + 360.0) % 360.0, 2
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
        abort_on_physical_mouse: bool = True,
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
        # S-1404: sabit True yok — gerçek prob: sözleşme denetimi +
        # kamera yumuşatıcı yörünge üretimi + anti-snap doğrulaması.
        try:
            ready = validate_windmouse_humanizer_contract()
            smoothing_ready = self._probe_camera_smoothing_ready()
            anti_snap = False
            if smoothing_ready:
                traj = self.camera_smoother.generate_heading_turn_trajectory(
                    angle_diff_deg=30.0
                )
                res = self.verify_trajectory_anti_snap(traj)
                anti_snap = bool(isinstance(res, dict) and res.get("is_smooth"))
            return {
                "humanizer_ready": bool(ready),
                "curve_engine": "WindMouse+Bezier",
                "camera_smoothing_ready": bool(smoothing_ready),
                "anti_snap_movement": bool(anti_snap),
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
                # S-1001: 600 iterasyon dolduysa hedef tek adımda eklenemez —
                # son nokta yüzlerce px uzakta olabilir (ışınlanma). Kalan
                # mesafe, adım boyu üst sınırını aşmayan düz adımlarla
                # yürünür; son ekleme ancak bitişik noktaya yapılır.
                step_cap = max(2.0, cur_max_step)
                tail_wait = round(
                    max(lo_wait, min(hi_wait, hi_wait * 0.92)),
                    6,
                )
                guard = 0
                while (
                    (int(round(cur_x)), int(round(cur_y))) != final_pos
                    and guard < 10000
                ):
                    guard += 1
                    rem_x, rem_y = dx - cur_x, dy - cur_y
                    rem = math.hypot(rem_x, rem_y)
                    if rem < 1.0:
                        break
                    step = min(step_cap, rem)
                    cur_x += rem_x / rem * step
                    cur_y += rem_y / rem * step
                    ix = int(round(cur_x))
                    iy = int(round(cur_y))
                    if (ix, iy) != last_int_pos:
                        path.append((ix, iy, tail_wait))
                        last_int_pos = (ix, iy)
                if not path or (path[-1][0], path[-1][1]) != final_pos:
                    path.append((final_pos[0], final_pos[1], tail_wait))

            return path
        except Exception as exc:
            # F-1702: hata dalı tek noktalı ışınlanma döndüremez — boş yörünge
            # döner, çağıranlar (move_cursor_humanized) bunu başarısız sayar.
            logger.error(
                "generate_windmouse_path hatasi ((%d,%d)->(%d,%d)): %s",
                start_x,
                start_y,
                dest_x,
                dest_y,
                exc,
            )
            return []

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
        - Yörünge sırasında `abort_on_physical_mouse` etkinse (varsayılan True,
          F-1704) fiziksel tuş basımı veya > 450px devasa sürükleme durumunda keser.
        - Mutlak imleç setter'ı yoksa SetCursorPos ışınlanması yapılmaz (F-1703);
          bunun yerine yörünge noktaları donanım zincirli göreli hareketle adım
          adım uygulanır.
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

            use_physical_abort = (
                self.abort_on_physical_mouse
                if abort_on_physical_mouse is None
                else bool(abort_on_physical_mouse)
            )

            # F-1696: başlamadan önce fiziksel müdahale denetimi.
            if use_physical_abort:
                abort, reason = self.input_guard.should_abort_input()
                if abort:
                    logger.warning(
                        "move_cursor_humanized baslamadan iptal: %s", reason
                    )
                    return False

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
            # F-1702: boş yörünge (üretim hatası) başarısız harekettir — True sayılmaz.
            if not path:
                logger.error(
                    "move_cursor_humanized yörünge üretilemedi -> (%s, %s)",
                    actual_dest_x,
                    actual_dest_y,
                )
                return False

            setter_fn = self._resolve_cursor_setter(actual_ctrl)
            # F-1703: mutlak setter yoksa göreli zincirle adım adım ilerle.
            rel_setter = None
            if setter_fn is None:
                rel_setter = self.camera_smoother._resolve_relative_mouse_setter(
                    actual_ctrl
                )

            prev_x, prev_y = cur_x, cur_y
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

                if setter_fn is not None:
                    # Açık False = setter hareketi reddetti.
                    if setter_fn(step_x, step_y) is False:
                        logger.error(
                            "move_cursor_humanized imlec tasiyici reddetti "
                            "-> (%d, %d)",
                            step_x,
                            step_y,
                        )
                        return False
                else:
                    # F-1703: ışınlanma yok — önceki noktadan göreli adımlarla ilerle.
                    if rel_setter is None or rel_setter(
                        step_x - prev_x, step_y - prev_y
                    ) is False:
                        logger.error(
                            "move_cursor_humanized göreli hareket reddedildi "
                            "-> (%d, %d); donanim zinciri yok",
                            step_x,
                            step_y,
                        )
                        return False
                prev_x, prev_y = step_x, step_y
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

    # F-1700: fare butonuyla yapılan skill aksiyonu fare butonu olarak değil
    # R/T tuşuyla gönderilir (sol tık -> R, sağ tık -> T); tuşlar Interception
    # zinciri üzerinden input_ctrl.press_key ile gider.
    SKILL_KEY_FOR_MOUSE_BUTTON: dict[str, str] = {"left": "r", "right": "t"}
    SKILL_VK_FOR_MOUSE_BUTTON: dict[str, int] = {"left": 0x52, "right": 0x54}

    def send_skill_key_for_button(
        self,
        input_ctrl: Any = None,
        button: str = "left",
    ) -> bool:
        """
        Fare butonu aksiyonunu skill tuşu olarak gönderir: sol -> R, sağ -> T.

        Tuş input_ctrl.press_key üzerinden (Interception donanım zinciri)
        gönderilir; yumuşak fallback yoktur. Basış guard'a bildirilir
        (F-1696) ve press_key'in dönüşü denetlenir: 0.0 / False = gönderilmedi.
        """
        btn = str(button).strip().lower()
        if btn not in self.SKILL_KEY_FOR_MOUSE_BUTTON:
            logger.error(
                "send_skill_key_for_button: bilinmeyen buton '%s' (left/right beklenir)",
                button,
            )
            return False
        if input_ctrl is None:
            logger.error(
                "send_skill_key_for_button: input_ctrl yok — skill tusu (%s) "
                "gonderilemedi, yumusak fallback yok",
                self.SKILL_KEY_FOR_MOUSE_BUTTON[btn],
            )
            return False
        press_fn = getattr(input_ctrl, "press_key", None)
        if not callable(press_fn):
            logger.error(
                "send_skill_key_for_button: input_ctrl'de press_key yok — "
                "skill tusu (%s) gonderilemedi",
                self.SKILL_KEY_FOR_MOUSE_BUTTON[btn],
            )
            return False
        key = self.SKILL_KEY_FOR_MOUSE_BUTTON[btn]
        vk = self.SKILL_VK_FOR_MOUSE_BUTTON[btn]
        self.input_guard.register_bot_key_press(vk, pressed=True)
        try:
            result = press_fn(key)
        except Exception as exc:
            logger.error("send_skill_key_for_button('%s') hatasi: %s", key, exc)
            self.input_guard.register_bot_key_press(vk, pressed=False)
            return False
        self.input_guard.register_bot_key_press(vk, pressed=False)
        if isinstance(result, (int, float)):
            ok = float(result) > 0.0
        else:
            ok = bool(result)
        if not ok:
            logger.warning(
                "send_skill_key_for_button: press_key('%s') gonderimi "
                "dogrulanamadi (donus=%r)",
                key,
                result,
            )
            return False
        return True

    def click_humanized(
        self,
        input_ctrl: Any = None,
        x: int = 0,
        y: int = 0,
        button: str = "left",
        apply_jitter: bool = True,
        abort_on_physical_mouse: Optional[bool] = None,
        use_skill_key: bool = False,
    ) -> bool:
        """
        İmleci hedef koordinata WindMouse ile insansı şekilde taşır ve biyometrik
        basılı tutma süresiyle tıklama gerçekleştirir.
        Başlangıçta mevcut OS imleç konumunu `(cur_x, cur_y)` okuyup
        `self.input_guard.record_virtual_mouse(cur_x, cur_y)` çağırır.
        `use_skill_key=True` ise tıklama fare butonu olarak değil R/T tuşuyla
        yapılır (sol -> R, sağ -> T; F-1700).
        """
        try:
            btn = str(button).strip().lower()
            # F-1701: orta tuş artık sol tuşa düşmez — desteklenmeyen buton reddedilir.
            if btn not in ("left", "right"):
                logger.error(
                    "click_humanized: desteklenmeyen buton '%s' (yalnizca left/right)",
                    button,
                )
                return False
            if isinstance(input_ctrl, (int, float)) and not hasattr(input_ctrl, "set_cursor_pos"):
                target_x = int(input_ctrl)
                target_y = int(x)
                actual_ctrl = None
            else:
                target_x = int(x)
                target_y = int(y)
                actual_ctrl = input_ctrl

            use_physical_abort = (
                self.abort_on_physical_mouse
                if abort_on_physical_mouse is None
                else bool(abort_on_physical_mouse)
            )
            # F-1696: başlamadan önce fiziksel müdahale denetimi.
            if use_physical_abort:
                abort, reason = self.input_guard.should_abort_input()
                if abort:
                    logger.warning("click_humanized baslamadan iptal: %s", reason)
                    return False

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
            # F-1701: jitter sonrası hedef ekran sınırları içine çekilir.
            target_x, target_y = _clamp_to_screen(target_x, target_y)

            moved = self.move_cursor_humanized(
                actual_ctrl,
                target_x,
                target_y,
                abort_on_physical_mouse=abort_on_physical_mouse,
            )
            if not moved:
                return False

            self.input_guard.record_virtual_mouse(target_x, target_y)
            btn_vk = 0x02 if btn == "right" else 0x01
            self.input_guard.record_self_key(btn_vk)

            # F-1700: skill aksiyonu fare butonu yerine R/T tuşuyla yapılır.
            if use_skill_key:
                return self.send_skill_key_for_button(actual_ctrl, button=btn)

            if actual_ctrl is not None:
                for click_attr in ("click_map", "human_click", "click"):
                    click_fn = getattr(actual_ctrl, click_attr, None)
                    if not callable(click_fn):
                        continue
                    try:
                        result = click_fn(target_x, target_y, button=btn)
                    except Exception as exc:
                        logger.debug(
                            "click_humanized %s hatasi: %s", click_attr, exc
                        )
                        continue
                    # F-1694 ile tutarlı: açık False = tıklama yapılmadı.
                    if result is False:
                        continue
                    self.input_guard.record_virtual_mouse(target_x, target_y)
                    self.input_guard.record_self_key(btn_vk)
                    return True

            # F-1694/F-1701: donanım zincirli tıklama yolu yoksa sessiz yumuşak
            # Win32 mouse_event fallback'i YOKTUR — tıklama yapılmadı sayılır.
            logger.error(
                "click_humanized: input_ctrl'de tiklama metodu yok "
                "(%d, %d, button=%s); yumusak fallback kaldirildi",
                target_x,
                target_y,
                btn,
            )
            return False
        except Exception as exc:
            logger.error("click_humanized hatasi (%s, %s): %s", x, y, exc)
            return False

    @staticmethod
    def _resolve_cursor_setter(input_ctrl: Any) -> Optional[Callable[[int, int], Any]]:
        """input_ctrl nesnesindeki uygun imleç taşıma metodunu çözer.

        F-1703: uygun metod yoksa None döner — artık Win32 SetCursorPos
        ışınlanması fallback'i yoktur; çağıran göreli zincire düşer.
        """
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
        return None

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

    def _probe_camera_smoothing_ready(self) -> bool:
        """S-1404: kamera yumuşatıcının gerçek probu — 30°'lik heading
        yörüngesi üretilebiliyorsa hazır sayılır (sabit True yok)."""
        try:
            traj = self.camera_smoother.generate_heading_turn_trajectory(
                angle_diff_deg=30.0
            )
            return bool(traj)
        except Exception:
            return False

    def get_camera_smoothing_summary(self) -> dict[str, Any]:
        return {
            "module": "windmouse_humanizer",
            "camera_smoothing_ready": self._probe_camera_smoothing_ready(),
            "metrics": self.camera_smoother.get_metrics(),
        }

    def get_windmouse_health_summary(self) -> dict:
        # S-1404: hazır bilgisi gerçek sözleşme + yumuşatıcı probundan gelir.
        return {
            "module": "windmouse_humanizer",
            "ready": bool(validate_windmouse_humanizer_contract()),
            "camera_smoothing_ready": self._probe_camera_smoothing_ready(),
        }

    def get_windmouse_humanizer_ready_flag(self) -> dict:
        # S-1404: hazır bilgisi gerçek sözleşme denetiminden gelir.
        return {
            "module": "windmouse_humanizer",
            "ready": bool(validate_windmouse_humanizer_contract()),
        }

    def get_windmouse_humanizer_version_info(self) -> dict:
        # S-1404: sürüm dizesi sabittir ama ready gerçek sözleşme denetimidir.
        return {
            "module": "windmouse_humanizer",
            "version": __version__,
            "ready": bool(validate_windmouse_humanizer_contract()),
        }

    def get_windmouse_humanizer_diagnostic_snapshot(self) -> dict:
        # S-1404: status/ready gerçek sözleşme denetiminden türetilir.
        ready = bool(validate_windmouse_humanizer_contract())
        return {
            "module": "windmouse_humanizer",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "camera_smoothing": self.get_camera_smoothing_summary(),
        }


def check_driver_backend_status() -> dict[str, Any]:
    """
    Sanal Sürücü Durum Denetleyicisi.
    Bilgisayarda `FakerInput`, `Interception` (`interception.dll`) veya yerel
    `SendInput ScanCode` katmanlarından hangisinin aktif olduğunu raporlar.
    """
    status: dict[str, Any] = {
        "active_backend": "unavailable",
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

    # 2. Interception (interception.dll + kernel driver) kontrolü.
    # F-1699: salt keyboard.sys dosyasının varlığı Interception kanıtı DEĞİLDİR
    # (Windows sistem dosyasıdır) — yalnızca interception_create_context
    # sunan ve context açabilen DLL sayılır.
    try:
        interception_candidates = (
            "interception.dll",
            os.path.join(os.getcwd(), "interception.dll"),
            os.path.join(os.getcwd(), "drivers", "interception.dll"),
            r"C:\Program Files\Interception\interception.dll",
            r"C:\Windows\System32\interception.dll",
        )
        for dll_candidate in interception_candidates:
            try:
                lib = ctypes.WinDLL(dll_candidate)
            except (OSError, AttributeError):
                continue
            try:
                if not hasattr(lib, "interception_create_context"):
                    continue
                ctx = lib.interception_create_context()
                if not ctx:
                    # DLL var ama kernel driver yok — context açılamadı.
                    status["details"].append(
                        f"Interception DLL bulundu ama driver yok (context NULL): {dll_candidate}"
                    )
                    continue
                try:
                    if hasattr(lib, "interception_destroy_context"):
                        lib.interception_destroy_context(ctx)
                except Exception:
                    pass
                status["interception_available"] = True
                status["interception_dll_path"] = dll_candidate
                status["details"].append(
                    f"Interception DLL loaded + context OK: {dll_candidate}"
                )
                break
            except (OSError, AttributeError) as exc:
                logger.debug("Interception prob hatasi (%s): %s", dll_candidate, exc)
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

    # F-1699: hardware_free_mode artık sabit True değil — donanım backend'i
    # (FakerInput/Interception) varsa False'dur.
    status["hardware_free_mode"] = not (
        status["fakerinput_available"] or status["interception_available"]
    )

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


def _validate_windmouse_humanizer_contract_uncached() -> bool:
    """Sözleşmeyi gerçekten denetler — F-1698: koşulsuz True yok.

    Denetlenenler: capability listesi dolu; WindMouseEngine kurulabiliyor;
    üretilen WindMouse yörüngesi hedefte bitiyor; anti-snap denetimi
    çalışıp beklenen alanları döndürüyor; backend durum raporu beklenen
    şemada. Herhangi biri sağlanamazsa False.
    """
    try:
        caps = get_windmouse_humanizer_capabilities()
        if not caps or "windmouse_humanizer" not in caps:
            logger.error("validate_windmouse_humanizer_contract: capability listesi bos")
            return False
        engine = WindMouseEngine()
        path = engine.generate_windmouse_path(0, 0, 120, 60)
        if not path or (path[-1][0], path[-1][1]) != (120, 60):
            logger.error(
                "validate_windmouse_humanizer_contract: yorunge hedefte bitmiyor: %r",
                path[-3:] if path else path,
            )
            return False
        res = engine.verify_trajectory_anti_snap(path)
        if not isinstance(res, dict) or "is_smooth" not in res:
            logger.error(
                "validate_windmouse_humanizer_contract: anti-snap denetimi calismadi"
            )
            return False
        status = check_driver_backend_status()
        if not isinstance(status, dict) or status.get("active_backend") not in (
            "fakerinput",
            "interception",
            "sendinput_scancode",
            "unavailable",
        ):
            logger.error(
                "validate_windmouse_humanizer_contract: backend durum raporu gecersiz: %r",
                status,
            )
            return False
        return True
    except Exception as exc:
        logger.error("validate_windmouse_humanizer_contract hatasi: %s", exc)
        return False


# Prob zinciri (readiness -> baseline, audit -> readiness, digest -> audit,
# telemetry -> digest, seal -> telemetry, certificate -> seal, probe ->
# certificate, summary -> probe) tek dis cagrida probu 8 kez kosuyordu
# (her biri engine kurup yörünge üretip backend probu yapıyor). Prob sonucu
# yalnizca surec ortamına bağlı; 60 sn TTL memoizasyon yeterli. Hicbir stub
# silinmedi/kablolanmadi, yeni ready_flag/validate kalibi eklenmedi (K4).
_CONTRACT_PROBE_CACHE_TTL_SEC: float = 60.0
_CONTRACT_PROBE_CACHE: dict[str, Any] = {"value": None, "ts": 0.0}
_CONTRACT_PROBE_CACHE_LOCK = threading.Lock()


def validate_windmouse_humanizer_contract() -> bool:
    """Sözleşme probunu 60 sn TTL ile önbellekli dondurur (zincir 8x kosmasin)."""
    now = time.monotonic()
    with _CONTRACT_PROBE_CACHE_LOCK:
        cached = _CONTRACT_PROBE_CACHE["value"]
        if cached is not None and (now - _CONTRACT_PROBE_CACHE["ts"]) < _CONTRACT_PROBE_CACHE_TTL_SEC:
            return bool(cached)
    fresh = _validate_windmouse_humanizer_contract_uncached()
    with _CONTRACT_PROBE_CACHE_LOCK:
        _CONTRACT_PROBE_CACHE["value"] = bool(fresh)
        _CONTRACT_PROBE_CACHE["ts"] = time.monotonic()
    return bool(fresh)


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

# F-1708: import anında tekil nesne kurulmaz — ilk kullanımda tembel kurulur.
_DEFAULT_ENGINE: Optional[WindMouseEngine] = None
_DEFAULT_SMOOTHER: Optional[CameraMovementSmoother] = None
_DEFAULT_LOCK = threading.Lock()


def _get_default_engine() -> WindMouseEngine:
    global _DEFAULT_ENGINE, _DEFAULT_SMOOTHER
    if _DEFAULT_ENGINE is None:
        with _DEFAULT_LOCK:
            if _DEFAULT_ENGINE is None:
                _DEFAULT_ENGINE = WindMouseEngine()
                _DEFAULT_SMOOTHER = _DEFAULT_ENGINE.camera_smoother
    return _DEFAULT_ENGINE


def _get_default_smoother() -> CameraMovementSmoother:
    _get_default_engine()
    assert _DEFAULT_SMOOTHER is not None
    return _DEFAULT_SMOOTHER


def get_windmouse_humanizer_diagnostic_snapshot() -> dict:
    """Modül seviyesi tanı anlık görüntüsü."""
    return _get_default_engine().get_windmouse_humanizer_diagnostic_snapshot()


def get_camera_smoothing_summary() -> dict[str, Any]:
    """Modül seviyesi kamera yumuşatma durum özeti."""
    return _get_default_engine().get_camera_smoothing_summary()


def generate_relative_camera_trajectory(
    total_dx: int,
    total_dy: int,
    duration_sec: Optional[float] = None,
    config: Optional[CameraSmoothingConfig] = None,
    enable_wrist_arc: Optional[bool] = None,
    enable_overshoot: Optional[bool] = None,
) -> list[tuple[int, int, float]]:
    """Modül seviyesi relative kamera Bezier yörünge üretimi."""
    return _get_default_smoother().generate_relative_camera_trajectory(
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
    return _get_default_smoother().generate_heading_turn_trajectory(
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
    return _get_default_smoother().generate_pitch_yaw_trajectory(
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
    return _get_default_smoother().smooth_turn_heading(
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
    return _get_default_smoother().smooth_camera_pitch_yaw(
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
    return _get_default_smoother().smooth_camera_nudge(
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
    return _get_default_smoother().smooth_movement_heading_step(
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
    return _get_default_smoother().verify_trajectory_anti_snap(
        trajectory=trajectory,
        max_allowed_single_step_px=max_allowed_single_step_px,
    )



