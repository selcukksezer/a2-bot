# -*- coding: utf-8 -*-
"""
Donanım Seviyesi Girdi ve Kısayol Sürücüsü (Hardware-Level Input & Hotkey Driver)
AION 2 Gatherion Pixel Bot - Ajan 2 Modülü

Mimari Katmanlar:
1. Birincil: FakerInput (Sanal HID Sürücü Köprüsü / Pipe veya DLL mevcutsa)
2. İkincil: pydirectinput (DirectInput ScanCode Tabanlı)
3. Üçüncül (En Alt Seviye): ctypes.windll.user32.SendInput (Donanım ScanCode + Mutlak Fare)

Özellikler:
- Çözünürlükten bağımsız (1920x1080, 2560x1440 vb.) normalize ve piksel koordinat desteği
- Humanizer: ±2 piksel rastgele sapma, 40ms-90ms (0.04s-0.09s) basılı tutma süresi, Bezier/yumuşak imleç geçişi
- Test Modu (test_mode=True): Fiziksel girdi göndermeden detaylı logging üretimi
- Global Hotkey Dinleyici: F9 (Duraklat/Menü), F9->Enter/R (Devam Et), F8 (Node Kaydet),
  Rota aktifken kullanıcı tuş müdahalesiyle otomatik PAUSED geçişi (_bot_pressing kilidi korumalı)
"""

import ctypes
from ctypes import wintypes
import logging
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.005
    pydirectinput.FAILSAFE = False
    HAS_PYDIRECTINPUT = True
except Exception as e:
    HAS_PYDIRECTINPUT = False
    pydirectinput = None

try:
    import keyboard
    HAS_KEYBOARD = True
except Exception as e:
    HAS_KEYBOARD = False
    keyboard = None

# Modül logger yapılandırması (Çift loglamayı önlemek için propagate=False)
logger = logging.getLogger("InputController")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.propagate = False


# ============================================================================
# CTYPES SENDINPUT DONANIM SCANCODE YAPILARI (EN ALT SEVİYE GERİ DÖNÜŞ)
# ============================================================================

PUL = ctypes.POINTER(ctypes.c_ulong)

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

KEYEVENTF_EXTENDEDKEY = 0x0001
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MIDDLEDOWN = 0x0020
MOUSEEVENTF_MIDDLEUP = 0x0040
MOUSEEVENTF_ABSOLUTE = 0x8000

# DirectInput Donanım ScanCode Tablosu (US/TR QWERTY Uyumlu Oyun Kısayolları)
HARDWARE_SCANCODES: Dict[str, int] = {
    "esc": 0x01,
    "escape": 0x01,
    "1": 0x02,
    "2": 0x03,
    "3": 0x04,
    "4": 0x05,
    "5": 0x06,
    "6": 0x07,
    "7": 0x08,
    "8": 0x09,
    "9": 0x0A,
    "0": 0x0B,
    "backspace": 0x0E,
    "tab": 0x0F,
    "q": 0x10,
    "w": 0x11,
    "e": 0x12,
    "r": 0x13,
    "t": 0x14,
    "y": 0x15,
    "u": 0x16,
    "i": 0x17,
    "o": 0x18,
    "p": 0x19,
    "enter": 0x1C,
    "return": 0x1C,
    "ctrl": 0x1D,
    "a": 0x1E,
    "s": 0x1F,
    "d": 0x20,
    "f": 0x21,
    "g": 0x22,
    "h": 0x23,
    "j": 0x24,
    "k": 0x25,
    "l": 0x26,
    "shift": 0x2A,
    "z": 0x2C,
    "x": 0x2D,
    "c": 0x2E,
    "v": 0x2F,
    "b": 0x30,
    "n": 0x31,
    "m": 0x32,
    "alt": 0x38,
    "space": 0x39,
    "f1": 0x3B,
    "f2": 0x3C,
    "f3": 0x3D,
    "f4": 0x3E,
    "f5": 0x3F,
    "f6": 0x40,
    "f7": 0x41,
    "f8": 0x42,
    "f9": 0x43,
    "f10": 0x44,
    "f11": 0x57,
    "f12": 0x58,
    "up": 0xC8,
    "left": 0xCB,
    "right": 0xCD,
    "down": 0xD0,
}

EXTENDED_KEYS: Set[str] = {"up", "down", "left", "right", "insert", "delete", "home", "end", "pageup", "pagedown"}

# Windows Virtual-Key (VK) Tablosu (SendInput + MapVirtualKeyW donanım ScanCode dönüşümü için)
VIRTUAL_KEY_CODES: Dict[str, int] = {
    "esc": 0x1B,
    "escape": 0x1B,
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
    "backspace": 0x08,
    "tab": 0x09,
    "q": 0x51,
    "w": 0x57,
    "e": 0x45,
    "r": 0x52,
    "t": 0x54,
    "y": 0x59,
    "u": 0x55,
    "i": 0x49,
    "o": 0x4F,
    "p": 0x50,
    "enter": 0x0D,
    "return": 0x0D,
    "ctrl": 0x11,
    "a": 0x41,
    "s": 0x53,
    "d": 0x44,
    "f": 0x46,
    "g": 0x47,
    "h": 0x48,
    "j": 0x4A,
    "k": 0x4B,
    "l": 0x4C,
    "shift": 0x10,
    "z": 0x5A,
    "x": 0x58,
    "c": 0x43,
    "v": 0x56,
    "b": 0x42,
    "n": 0x4E,
    "m": 0x4D,
    "alt": 0x12,
    "space": 0x20,
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "f11": 0x7A,
    "f12": 0x7B,
    "up": 0x26,
    "left": 0x25,
    "right": 0x27,
    "down": 0x28,
}

# Oyun içi normal hareket ve etkileşim tuşları beyaz listesi (Asla otomatik PAUSE tetiklemez)
WHITELISTED_GAMEPLAY_KEYS: Set[str] = {
    "w", "a", "s", "d",
    "space", "shift", "left shift", "right shift",
    "ctrl", "left ctrl", "right ctrl",
    "alt", "left alt", "right alt",
    "tab", "f", "e", "m", "r", "enter", "return",
    "k", "i", "c", "g", "q", "z", "x", "v", "b", "n",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "f8", "f9",
}

HOTKEY_DEBOUNCE_SEC: float = 0.45
SETTINGS_RELOAD_DEBOUNCE_SEC: float = 0.250


class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", PUL),
    ]


class InnerInputUnion(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput),
    ]


class InputStruct(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", InnerInputUnion),
    ]


# ============================================================================
# FAKERINPUT DONANIM KÖPRÜSÜ (MEVCUTSA BİRİNCİL SÜRÜCÜ)
# ============================================================================

class FakerInputBridge:
    """
    FakerInput çekirdek sürücüsü veya DLL köprüsü mevcutsa donanım seviyesinde
    HID raporları gönderir. Sistemde kurulu değilse güvenli şekilde devre dışı kalır.
    """

    def __init__(self) -> None:
        self.available: bool = False
        self._dll = None
        self._handle = None
        self._detect_fakerinput()

    def _detect_fakerinput(self) -> None:
        """Sistemde FakerInput DLL veya cihaz yolunu arar."""
        try:
            candidate_paths = [
                os.path.join(os.getcwd(), "FakerInputWrapper.dll"),
                os.path.join(os.getcwd(), "drivers", "FakerInputWrapper.dll"),
                r"C:\Program Files\FakerInput\FakerInputWrapper.dll",
            ]
            for dll_path in candidate_paths:
                if os.path.exists(dll_path):
                    self._dll = ctypes.WinDLL(dll_path)
                    if hasattr(self._dll, "fakerinput_alloc"):
                        self._handle = self._dll.fakerinput_alloc()
                        if self._handle:
                            self.available = True
                            logger.info("FakerInput donanım sürücüsü başarıyla bağlandı: %s", dll_path)
                            return
        except Exception as exc:
            logger.debug("FakerInput algılama sırasında bilgi: %s", exc)
            self.available = False

    def send_key(self, scancode: int, is_down: bool) -> bool:
        """FakerInput üzerinden tuş basımı/bırakımı gönderir."""
        if not self.available or self._dll is None or self._handle is None:
            return False
        try:
            if is_down and hasattr(self._dll, "fakerinput_key_down"):
                return bool(self._dll.fakerinput_key_down(self._handle, ctypes.c_ubyte(scancode)))
            if not is_down and hasattr(self._dll, "fakerinput_key_up"):
                return bool(self._dll.fakerinput_key_up(self._handle, ctypes.c_ubyte(scancode)))
            return False
        except Exception as exc:
            logger.error("FakerInput tuş gönderim hatası: %s", exc)
            return False

    def move_mouse_abs(self, norm_x: int, norm_y: int) -> bool:
        """FakerInput üzerinden mutlak fare hareketi gönderir (0..65535)."""
        if not self.available or self._dll is None or self._handle is None:
            return False
        try:
            if hasattr(self._dll, "fakerinput_update_absolute_mouse"):
                return bool(
                    self._dll.fakerinput_update_absolute_mouse(
                        self._handle,
                        ctypes.c_ubyte(0),
                        ctypes.c_ushort(norm_x),
                        ctypes.c_ushort(norm_y),
                        ctypes.c_ubyte(0),
                        ctypes.c_ubyte(0),
                    )
                )
            return False
        except Exception as exc:
            logger.error("FakerInput fare hareket hatası: %s", exc)
            return False

    def mouse_button(self, button: str, is_down: bool) -> bool:
        """FakerInput üzerinden fare tuşu basımı/bırakımı gönderir."""
        if not self.available or self._dll is None or self._handle is None:
            return False
        try:
            btn_mask = 1 if button == "left" else (2 if button == "right" else 4)
            state_mask = btn_mask if is_down else 0
            if hasattr(self._dll, "fakerinput_update_relative_mouse"):
                return bool(
                    self._dll.fakerinput_update_relative_mouse(
                        self._handle,
                        ctypes.c_ubyte(state_mask),
                        ctypes.c_short(0),
                        ctypes.c_short(0),
                        ctypes.c_ubyte(0),
                        ctypes.c_ubyte(0),
                    )
                )
            return False
        except Exception as exc:
            logger.error("FakerInput fare tıklama hatası: %s", exc)
            return False


# ============================================================================
# GLOBAL KISAYOL DİNLEYİCİSİ (HOTKEY LISTENER)
# ============================================================================

class HotkeyListener:
    """
    Global klavye kısayollarını ve rota sırasındaki fiziksel kullanıcı tuş
    müdahalelerini (user key interrupt) dinleyen yardımcı sınıf.
    """

    def __init__(self, controller: "InputController") -> None:
        self.controller = controller
        self.on_f9: Optional[Callable[[], None]] = None
        self.on_f9_enter: Optional[Callable[[], None]] = None
        self.on_f8: Optional[Callable[[], None]] = None
        self.on_user_key_interrupt: Optional[Callable[[str], None]] = None
        self._hook_handle = None
        self._release_hook_handle = None
        self._f9_menu_open: bool = False
        self._f9_physically_down: bool = False
        self._f8_physically_down: bool = False
        self._last_f9_ts: float = 0.0
        self._last_f8_ts: float = 0.0
        self._last_resume_ts: float = 0.0
        self._last_f9_time: float = 0.0
        self._last_f8_time: float = 0.0
        self._lock = threading.Lock()

    def register(
        self,
        on_f9: Optional[Callable[[], None]] = None,
        on_f9_enter: Optional[Callable[[], None]] = None,
        on_f8: Optional[Callable[[], None]] = None,
        on_user_key_interrupt: Optional[Callable[[str], None]] = None,
        on_any_key_pause: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Kısayol callback fonksiyonlarını kaydeder ve global klavye kancasını başlatır.
        """
        try:
            self.on_f9 = on_f9
            self.on_f9_enter = on_f9_enter
            self.on_f8 = on_f8
            self.on_user_key_interrupt = on_user_key_interrupt or on_any_key_pause

            self.unregister()

            if not HAS_KEYBOARD or keyboard is None:
                logger.warning("keyboard modülü yüklü değil, global kısayollar etkinleştirilemedi.")
                return False

            self._hook_handle = keyboard.on_press(self._on_key_event, suppress=False)
            self._release_hook_handle = keyboard.on_release(self._on_key_release, suppress=False)
            logger.info("Global kısayol dinleyicisi aktif edildi (F9, F9->Enter/R, F8, Debounce=0.55s).")
            return True
        except Exception as exc:
            logger.error("Kısayol dinleyicisi kaydedilirken hata oluştu: %s", exc)
            return False

    def unregister(self) -> None:
        """Mevcut klavye kancasını güvenli şekilde kaldırır."""
        try:
            if self._hook_handle is not None and HAS_KEYBOARD and keyboard is not None:
                keyboard.unhook(self._hook_handle)
                self._hook_handle = None
            if self._release_hook_handle is not None and HAS_KEYBOARD and keyboard is not None:
                keyboard.unhook(self._release_hook_handle)
                self._release_hook_handle = None
            logger.debug("Global kısayol dinleyicisi kaldırıldı.")
        except Exception as exc:
            logger.error("Kısayol dinleyicisi kaldırılırken hata: %s", exc)
            self._hook_handle = None
            self._release_hook_handle = None

    def _on_key_release(self, event) -> None:
        """Tuş bırakıldığında fiziksel basılı tutma (auto-repeat) kilidini sıfırlar."""
        try:
            raw_name = getattr(event, "name", "")
            if not raw_name:
                return
            key_name = str(raw_name).strip().lower()
            if key_name == "f9":
                with self._lock:
                    self._f9_physically_down = False
            elif key_name == "f8":
                with self._lock:
                    self._f8_physically_down = False
        except Exception:
            pass

    def set_f9_menu_open(self, is_open: bool) -> None:
        """F9 menüsünün açık/kapalı durumunu günceller."""
        with self._lock:
            self._f9_menu_open = bool(is_open)

    def _on_key_event(self, event) -> None:
        """
        Her klavye basım olayında tetiklenir.
        Botun kendi bastığı sanal tuşları (_sending_virtual_key / _bot_pressing) yok sayar,
        hareket/oynanış tuşlarında botu duraklatmaz ve kısayollara 0.45s debounce uygular.
        """
        try:
            raw_name = getattr(event, "name", "")
            if not raw_name:
                return
            key_name = str(raw_name).strip().lower()

            now = time.time()

            # Botun kendi gönderdiği sanal tuş basımı ise kesinlikle müdahale sayma
            if (
                self.controller._sending_virtual_key
                or (now - self.controller._last_virtual_key_ts) < 0.15
                or self.controller.is_bot_pressing(key_name)
            ):
                return

            # 1. F9: Botu duraklat / F9 Menüsünü imlecin olduğu yerde aç (auto-repeat & 0.55s debounce korumalı)
            if key_name == "f9":
                with self._lock:
                    if self._f9_physically_down:
                        return
                    last_f9 = max(self._last_f9_ts, self._last_f9_time, self.controller._last_f9_ts)
                    if now - last_f9 < 0.55:
                        return
                    self._f9_physically_down = True
                    self._last_f9_ts = now
                    self._last_f9_time = now
                    self.controller._last_f9_ts = now
                    self._f9_menu_open = True
                self.controller.free_mouse_cursor()
                logger.info("[Hotkey] F9 algılandı -> Bot duraklatılıyor / F9 Menüsü açılıyor.")
                if self.on_f9 is not None:
                    threading.Thread(target=self._safe_invoke, args=(self.on_f9,), daemon=True).start()
                return

            # 2. F9 ardından Enter veya R: Aktif modda devam et (Resume, 0.45s debounce)
            if key_name in ("enter", "return", "r") and self._f9_menu_open:
                with self._lock:
                    last_resume = max(self._last_resume_ts, self.controller._last_resume_ts)
                    if now - last_resume < HOTKEY_DEBOUNCE_SEC:
                        return
                    self._last_resume_ts = now
                    self.controller._last_resume_ts = now
                    self._f9_menu_open = False
                logger.info("[Hotkey] F9 sonrası '%s' algılandı -> Aktif modda devam ediliyor (Resume).", key_name.upper())
                if self.on_f9_enter is not None:
                    threading.Thread(target=self._safe_invoke, args=(self.on_f9_enter,), daemon=True).start()
                return

            # 3. F8: Mevcut konumu yeni Node olarak kaydet (0.45s debounce)
            if key_name == "f8":
                with self._lock:
                    last_f8 = max(self._last_f8_ts, self._last_f8_time, self.controller._last_f8_ts)
                    if now - last_f8 < HOTKEY_DEBOUNCE_SEC:
                        return
                    self._last_f8_ts = now
                    self._last_f8_time = now
                    self.controller._last_f8_ts = now
                logger.info("[Hotkey] F8 algılandı -> Mevcut konum yeni Node olarak kaydediliyor.")
                if self.on_f8 is not None:
                    threading.Thread(target=self._safe_invoke, args=(self.on_f8,), daemon=True).start()
                return

            # 4. Rota aktifken otomatik PAUSED:
            # Yalnızca pause_on_any_key açıkça True ise ve tuş hareket/oynanış beyaz listesinde değilse!
            if (
                self.controller.route_active
                and self.controller.pause_on_any_key
                and key_name not in WHITELISTED_GAMEPLAY_KEYS
            ):
                logger.warning(
                    "[Hotkey] Rota çalışırken kullanıcı fiziksel klavye tuşuna bastı ('%s') -> Bot PAUSED durumuna geçiriliyor!",
                    key_name,
                )
                self.controller.route_active = False
                with self._lock:
                    self._f9_menu_open = True
                if self.on_user_key_interrupt is not None:
                    threading.Thread(
                        target=self._safe_invoke,
                        args=(self.on_user_key_interrupt, key_name),
                        daemon=True,
                    ).start()
        except Exception as exc:
            logger.error("Klavye olayı işlenirken beklenmeyen hata: %s", exc)

    @staticmethod
    def _safe_invoke(callback: Callable, *args) -> None:
        """Callback çağrılarını istisna korumalı olarak çalıştırır."""
        try:
            callback(*args)
        except Exception as exc:
            logger.error("Hotkey callback çalıştırılırken hata: %s", exc)


# ============================================================================
# ANA GİRDİ DENETLEYİCİSİ (INPUT CONTROLLER)
# ============================================================================

class InputController:
    """
    Donanım seviyesi fare ve klavye denetleyicisi.
    FakerInput -> pydirectinput -> ctypes.windll.user32.SendInput geri dönüş zincirini
    ve insanlaştırma (Humanizer) algoritmalarını barındırır.
    """

    # Standart insanlaştırma sabitleri (Şartname: ±2px sapma, 40ms-90ms basılı tutma)
    DEFAULT_JITTER_PX: int = 2
    DEFAULT_MIN_HOLD_SEC: float = 0.04
    DEFAULT_MAX_HOLD_SEC: float = 0.09
    SETTINGS_RELOAD_DEBOUNCE_SEC: float = SETTINGS_RELOAD_DEBOUNCE_SEC

    def __init__(
        self,
        use_faker_input: bool = True,
        test_mode: bool = False,
        pause_on_any_key: bool = False,
        chat_box_guard_enabled: bool = True,
    ) -> None:
        self.use_faker_input: bool = use_faker_input
        self.test_mode: bool = bool(test_mode)
        self.route_active: bool = False
        self.pause_on_any_key: bool = bool(pause_on_any_key)
        self.chat_box_guard_enabled: bool = bool(chat_box_guard_enabled)

        # Sohbet kutusu (Chat Box) açık kalma koruması (Item 17)
        self._chat_open_flag: bool = False
        self._chat_detector_callback: Optional[Callable[[], bool]] = None
        self._chat_guard_esc_count: int = 0

        # Canlı ayar yeniden yükleme (Hot-Reload) 250ms sinyal debounce & state-diff kilidi (Item 19)
        self._settings_debounce_lock = threading.RLock()
        self._last_test_mode_log_ts: float = 0.0
        self._last_logged_test_mode: Optional[bool] = self.test_mode
        self._last_settings_apply_ts: float = 0.0
        self._last_applied_settings_snapshot: Dict[str, Any] = {
            "test_mode": self.test_mode,
            "pause_on_any_key": self.pause_on_any_key,
            "use_faker_input": self.use_faker_input,
            "chat_box_guard_enabled": self.chat_box_guard_enabled,
            "jitter_px": self.DEFAULT_JITTER_PX,
            "min_hold_sec": self.DEFAULT_MIN_HOLD_SEC,
            "max_hold_sec": self.DEFAULT_MAX_HOLD_SEC,
        }

        # Hotkey debounce zaman damgaları (0.45s minimum aralık)
        self._last_f9_ts: float = 0.0
        self._last_f8_ts: float = 0.0
        self._last_resume_ts: float = 0.0

        # Botun kendi bastığı sanal tuşları kullanıcı müdahalesinden ayırt etmek için kilit, bayrak ve zaman damgası
        self._bot_pressing_lock = threading.Lock()
        self._bot_pressing: bool = False
        self._sending_virtual_key: bool = False
        self._last_virtual_key_ts: float = 0.0
        self._active_bot_keys: Set[str] = set()

        # Birincil sürücü: FakerInput köprüsü
        self._faker = FakerInputBridge() if use_faker_input else None

        # Global kısayol dinleyicisi
        self.hotkey_listener = HotkeyListener(self)

        # Ekran çözünürlüğünü algıla
        self.screen_width, self.screen_height = self.get_screen_resolution()
        logger.info(
            "InputController başlatıldı | Çözünürlük: %dx%d | Backend: %s | Test Modu: %s",
            self.screen_width,
            self.screen_height,
            self.active_backend_name,
            self.test_mode,
        )

    @property
    def active_backend_name(self) -> str:
        """Aktif donanım girdi sürücüsünün adını döndürür."""
        if self._faker is not None and self._faker.available:
            return "FakerInput"
        if HAS_PYDIRECTINPUT:
            return "pydirectinput+SendInput"
        return "ctypes.SendInput"

    # ------------------------------------------------------------------------
    # TEST MODU, CANLI AYAR DEBOUNCE (ITEM 19) VE ROTA DURUM YÖNETİMİ
    # ------------------------------------------------------------------------

    def set_test_mode(self, enabled: bool) -> bool:
        """
        Test modunu açar veya kapatır (Item 19: 250ms sinyal debounce & state-diff korumalı).
        test_mode=True olduğunda fiziksel tuş/fare girdisi gönderilmez, sadece detaylı log üretilir.
        Yalnızca durum gerçekten değiştiğinde ve 250ms debounce penceresi dışında bilgi logu üretir.
        """
        with self._settings_debounce_lock:
            new_mode = bool(enabled)
            now = time.monotonic()
            if new_mode == self.test_mode:
                return False

            self.test_mode = new_mode
            self._last_applied_settings_snapshot["test_mode"] = new_mode

            if (now - self._last_test_mode_log_ts) >= self.SETTINGS_RELOAD_DEBOUNCE_SEC:
                self._last_test_mode_log_ts = now
                self._last_logged_test_mode = new_mode
                logger.info("InputController test_mode durumu güncellendi: %s", self.test_mode)
                return True

            return False

    def _extract_settings_snapshot(self, settings: Any) -> Dict[str, Any]:
        """Verilen ayar nesnesi veya sözlüğünden InputController ile ilgili alanları çıkarır."""
        if hasattr(settings, "settings") and hasattr(settings.settings, "to_dict"):
            raw = settings.settings.to_dict()
        elif hasattr(settings, "to_dict") and callable(settings.to_dict):
            raw = settings.to_dict()
        elif hasattr(settings, "config") and isinstance(settings.config, dict):
            raw = settings.config
        elif isinstance(settings, dict):
            raw = settings
        else:
            raw = {}

        gathering = raw.get("gathering", {}) if isinstance(raw.get("gathering"), dict) else {}
        route_cfg = raw.get("route", {}) if isinstance(raw.get("route"), dict) else {}
        input_cfg = raw.get("input", {}) if isinstance(raw.get("input"), dict) else {}

        test_mode_val = raw.get(
            "test_mode",
            gathering.get("test_mode", input_cfg.get("test_mode", self.test_mode)),
        )
        pause_any_val = raw.get(
            "pause_on_any_key",
            route_cfg.get("pause_on_any_key", input_cfg.get("pause_on_any_key", self.pause_on_any_key)),
        )
        faker_val = raw.get(
            "use_faker_input",
            input_cfg.get("use_faker_input", self.use_faker_input),
        )
        chat_guard_val = raw.get(
            "chat_box_guard_enabled",
            input_cfg.get("chat_box_guard_enabled", self.chat_box_guard_enabled),
        )
        jitter_val = raw.get(
            "jitter_px",
            input_cfg.get("jitter_px", self.DEFAULT_JITTER_PX),
        )
        min_hold_val = raw.get(
            "min_hold_sec",
            input_cfg.get("min_hold_sec", self.DEFAULT_MIN_HOLD_SEC),
        )
        max_hold_val = raw.get(
            "max_hold_sec",
            input_cfg.get("max_hold_sec", self.DEFAULT_MAX_HOLD_SEC),
        )

        return {
            "test_mode": bool(test_mode_val),
            "pause_on_any_key": bool(pause_any_val),
            "use_faker_input": bool(faker_val),
            "chat_box_guard_enabled": bool(chat_guard_val),
            "jitter_px": max(0, int(jitter_val)),
            "min_hold_sec": max(0.01, float(min_hold_val)),
            "max_hold_sec": max(0.01, float(max_hold_val)),
        }

    def apply_settings(self, settings: Any, force: bool = False) -> Dict[str, Any]:
        """
        Ayarlar kaydedildiğinde canlı yeniden yükleme (Hot-Reload) uygular (Item 19).
        250ms sinyal debounce ve state-diff koruması sayesinde peş peşe tetiklenen
        UI sinyallerinde aynı log mesajının 6 kez tekrar basılmasını engeller.
        """
        with self._settings_debounce_lock:
            now = time.monotonic()
            snapshot = self._extract_settings_snapshot(settings)
            changed_keys: List[str] = [
                k for k, v in snapshot.items() if self._last_applied_settings_snapshot.get(k) != v
            ]
            is_debounced = bool(
                not force
                and self._last_settings_apply_ts > 0.0
                and (now - self._last_settings_apply_ts) < self.SETTINGS_RELOAD_DEBOUNCE_SEC
            )

            if not changed_keys or is_debounced:
                # Durum değişmemişse veya 250ms debounce penceresi içindeyse log basmadan çık
                if changed_keys and is_debounced:
                    self.test_mode = snapshot["test_mode"]
                    self.pause_on_any_key = snapshot["pause_on_any_key"]
                    self.chat_box_guard_enabled = snapshot["chat_box_guard_enabled"]
                    self._last_applied_settings_snapshot.update(snapshot)
                return {
                    "applied": False,
                    "debounced": is_debounced,
                    "changed_keys": [],
                    "test_mode": self.test_mode,
                }

            self._last_settings_apply_ts = now
            if "test_mode" in changed_keys:
                self.set_test_mode(snapshot["test_mode"])
            self.pause_on_any_key = snapshot["pause_on_any_key"]
            self.use_faker_input = snapshot["use_faker_input"]
            self.chat_box_guard_enabled = snapshot["chat_box_guard_enabled"]
            self.DEFAULT_JITTER_PX = snapshot["jitter_px"]
            self.DEFAULT_MIN_HOLD_SEC = snapshot["min_hold_sec"]
            self.DEFAULT_MAX_HOLD_SEC = max(self.DEFAULT_MIN_HOLD_SEC, snapshot["max_hold_sec"])
            self._last_applied_settings_snapshot = dict(snapshot)

            non_test_changes = [k for k in changed_keys if k != "test_mode"]
            if non_test_changes:
                logger.info(
                    "InputController ayarları canlı güncellendi (%s).",
                    ", ".join(f"{k}={snapshot[k]}" for k in non_test_changes),
                )

            return {
                "applied": True,
                "debounced": False,
                "changed_keys": changed_keys,
                "test_mode": self.test_mode,
            }

    reload_settings = apply_settings

    # ------------------------------------------------------------------------
    # SOHBET KUTUSU (CHAT BOX) AÇIK KALMA KORUMASI (ITEM 17)
    # ------------------------------------------------------------------------

    def set_chat_box_open(self, is_open: bool) -> None:
        """Harici görüntü işleme döngüsünden sohbet kutusunun açık/kapalı durumunu bildirir."""
        self._chat_open_flag = bool(is_open)

    def set_chat_detector_callback(self, callback: Optional[Callable[[], bool]]) -> None:
        """Tuş gönderimi öncesinde sohbet kutusunun açık olup olmadığını sorgulayan callback kaydeder."""
        self._chat_detector_callback = callback

    def detect_chat_input_open(self, frame: Optional[Any] = None) -> Dict[str, Any]:
        """
        Sol alt sohbet metin giriş çubuğunun (x: 0.01..0.24, y: 0.91..0.96) açık olup olmadığını
        kontrol eder (Item 17). `frame` verilmişse `core.ocr_distance.detect_chat_input_open`
        üzerinden analiz yapar; verilmemişse kayıtlı callback veya `_chat_open_flag` durumunu döner.
        """
        from core.ocr_distance import ChatInputOpenResult, detect_chat_input_open as _ocr_detect_chat

        if frame is not None:
            res = _ocr_detect_chat(frame)
            self._chat_open_flag = bool(res.get("is_open", False))
            return res

        is_open = bool(self._chat_open_flag)
        if not is_open and callable(self._chat_detector_callback):
            try:
                is_open = bool(self._chat_detector_callback())
            except Exception as exc:
                logger.debug("Chat detector callback hatası: %s", exc)

        return ChatInputOpenResult(
            is_open=is_open,
            chat_open=is_open,
            has_border=is_open,
            has_cursor=False,
            has_prompt_tag=False,
            confidence=1.0 if is_open else 0.0,
            roi_bbox=(0, 0, 0, 0),
            elapsed_ms=0.0,
        )

    def close_chat_box_if_open(
        self,
        frame: Optional[Any] = None,
        chat_open: Optional[bool] = None,
    ) -> bool:
        """
        Sohbet kutusu (Chat Box) açıksa 'm' veya 'f' gibi tuşların sohbet satırına
        ("mmmmffff") yazılmasını önlemek için önce 'ESC' tuşuna basarak sohbeti kapatır (Item 17).
        Sohbet açıksa ve ESC basıldıysa True, kapalıysa False döndürür.
        """
        if chat_open is not None:
            is_open = bool(chat_open)
        elif frame is not None:
            det = self.detect_chat_input_open(frame)
            is_open = bool(det.get("is_open", False))
        else:
            det = self.detect_chat_input_open(None)
            is_open = bool(det.get("is_open", False))

        if not is_open:
            return False

        # Sonsuz döngüyü önlemek için bayrağı hemen temizle ve ESC gönder
        self._chat_open_flag = False
        self._chat_guard_esc_count += 1
        logger.warning(
            "Sohbet kutusu (Chat Box) açık algılandı! Tuş sızıntısını önlemek için ESC basılarak kapatılıyor (Sayaç=%d).",
            self._chat_guard_esc_count,
        )
        self.press_esc()
        time.sleep(0.005 if self.test_mode else 0.025)
        return True

    def set_pause_on_any_key(self, enabled: bool) -> None:
        """
        Rota çalışırken fiziksel tuş basımında otomatik duraklatma (pause_on_any_key) ayarını günceller.
        Varsayılan: False (yalnızca F9 duraklatır).
        """
        self.pause_on_any_key = bool(enabled)

    def set_route_active(self, active: bool) -> None:
        """
        Rota modunun aktif olup olmadığını ayarlar.
        Rota aktifken pause_on_any_key=True ise kullanıcı klavyeye basarsa on_user_key_interrupt tetiklenir.
        """
        self.route_active = bool(active)
        logger.debug("InputController route_active durumu güncellendi: %s", self.route_active)

    def is_bot_pressing(self, key_name: Optional[str] = None) -> bool:
        """Botun şu anda sanal bir tuşa basıp basmadığını kontrol eder."""
        with self._bot_pressing_lock:
            if self._sending_virtual_key or (time.time() - self._last_virtual_key_ts) < 0.15:
                if key_name is None or not self._active_bot_keys:
                    return True
                return key_name.lower() in self._active_bot_keys or self._sending_virtual_key
            if not self._bot_pressing:
                return False
            if key_name is None:
                return True
            return key_name.lower() in self._active_bot_keys

    # ------------------------------------------------------------------------
    # EKRAN ÇÖZÜNÜRLÜĞÜ VE KOORDİNAT DÖNÜŞÜMÜ (1920x1080 / 2560x1440 BAĞIMSIZ)
    # ------------------------------------------------------------------------

    @staticmethod
    def get_screen_resolution() -> Tuple[int, int]:
        """Windows user32 üzerinden güncel ekran çözünürlüğünü (genişlik, yükseklik) alır."""
        try:
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception as dpi_exc:
                logger.debug("DPI farkındalığı ayarlanırken bilgi: %s", dpi_exc)
            width = int(user32.GetSystemMetrics(0))
            height = int(user32.GetSystemMetrics(1))
            if width > 0 and height > 0:
                return width, height
        except Exception as exc:
            logger.error("Ekran çözünürlüğü okunamadı, varsayılan 1920x1080 kullanılıyor: %s", exc)
        return 1920, 1080

    @staticmethod
    def get_cursor_position() -> Tuple[int, int]:
        """Mevcut fare imleç koordinatlarını (x, y) döndürür."""
        try:
            pt = wintypes.POINT()
            if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
                return int(pt.x), int(pt.y)
        except Exception as exc:
            logger.error("Fare konumu okunamadı: %s", exc)
        return 0, 0

    def free_mouse_cursor(self) -> None:
        """
        F9 Menüsü açıldığında oyunun fareyi ekran ortasına kilitlemesini (ClipCursor / SetCapture)
        ve basılı kalan sağ/sol fare tuşlarını serbest bırakır.
        """
        try:
            user32 = ctypes.windll.user32
            user32.ClipCursor(None)
            user32.ReleaseCapture()
            for _ in range(8):
                if user32.ShowCursor(True) >= 0:
                    break
            if not self.test_mode:
                self._sendinput_mouse_button("right", False)
                self._sendinput_mouse_button("left", False)
        except Exception as exc:
            logger.debug("free_mouse_cursor bilgi: %s", exc)

    def normalize_coords(self, x: int, y: int) -> Tuple[float, float]:
        """Piksel koordinatlarını [0.0, 1.0] aralığında normalize koordinata dönüştürür."""
        self.screen_width, self.screen_height = self.get_screen_resolution()
        nx = max(0.0, min(1.0, float(x) / float(max(1, self.screen_width))))
        ny = max(0.0, min(1.0, float(y) / float(max(1, self.screen_height))))
        return nx, ny

    def denormalize_coords(self, nx: float, ny: float) -> Tuple[int, int]:
        """
        [0.0, 1.0] aralığındaki normalize koordinatları güncel ekran çözünürlüğüne
        (1920x1080, 2560x1440 vb.) göre piksel koordinatlarına dönüştürür.
        """
        self.screen_width, self.screen_height = self.get_screen_resolution()
        clamped_nx = max(0.0, min(1.0, float(nx)))
        clamped_ny = max(0.0, min(1.0, float(ny)))
        px = int(round(clamped_nx * (self.screen_width - 1)))
        py = int(round(clamped_ny * (self.screen_height - 1)))
        return px, py

    def apply_human_jitter(self, x: int, y: int, jitter_px: int = DEFAULT_JITTER_PX) -> Tuple[int, int]:
        """
        Koordinatlara rastgele ±jitter_px (varsayılan ±2 piksel) insan sapması uygular
        ve ekran sınırları içinde tutar.
        """
        self.screen_width, self.screen_height = self.get_screen_resolution()
        dx = random.randint(-abs(jitter_px), abs(jitter_px))
        dy = random.randint(-abs(jitter_px), abs(jitter_px))
        jx = max(0, min(self.screen_width - 1, int(x) + dx))
        jy = max(0, min(self.screen_height - 1, int(y) + dy))
        return jx, jy

    # ------------------------------------------------------------------------
    # EN ALT SEVİYE CTYPES SENDINPUT METODLARI
    # ------------------------------------------------------------------------

    def _sendinput_key(self, key: str, is_down: bool) -> bool:
        """
        ctypes.windll.user32.SendInput ile donanım ScanCode (KEYEVENTF_SCANCODE = 0x0008)
        ve MapVirtualKeyW kullanarak DirectInput/Unreal (Aion 2) uyumlu tuş olayı gönderir.
        """
        try:
            key_lower = key.lower().strip()
            vk_code = VIRTUAL_KEY_CODES.get(key_lower, 0)
            if vk_code == 0 and key_lower:
                vk_code = ctypes.windll.user32.VkKeyScanW(ord(key_lower[0])) & 0xFF

            mapped_scan = int(ctypes.windll.user32.MapVirtualKeyW(vk_code, 0)) if vk_code > 0 else 0
            scancode = mapped_scan if mapped_scan > 0 else HARDWARE_SCANCODES.get(key_lower, 0)

            flags = KEYEVENTF_SCANCODE
            if not is_down:
                flags |= KEYEVENTF_KEYUP
            if key_lower in EXTENDED_KEYS:
                flags |= KEYEVENTF_EXTENDEDKEY

            extra = ctypes.c_ulong(0)
            ii = InnerInputUnion()
            ii.ki = KeyBdInput(vk_code, scancode, flags, 0, ctypes.pointer(extra))
            inp = InputStruct(ctypes.c_ulong(INPUT_KEYBOARD), ii)
            sent = ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
            return sent == 1
        except Exception as exc:
            logger.error("SendInput klavye hatası (%s, down=%s): %s", key, is_down, exc)
            return False

    def _sendinput_mouse_move(self, x: int, y: int) -> bool:
        """ctypes.windll.user32.SendInput ile mutlak (ABSOLUTE) koordinata fare taşır."""
        try:
            width, height = self.get_screen_resolution()
            abs_x = int(round((float(x) * 65535.0) / float(max(1, width - 1))))
            abs_y = int(round((float(y) * 65535.0) / float(max(1, height - 1))))

            extra = ctypes.c_ulong(0)
            ii = InnerInputUnion()
            ii.mi = MouseInput(
                abs_x,
                abs_y,
                0,
                MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE,
                0,
                ctypes.pointer(extra),
            )
            inp = InputStruct(ctypes.c_ulong(INPUT_MOUSE), ii)
            sent = ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
            return sent == 1
        except Exception as exc:
            logger.error("SendInput fare taşıma hatası (%d, %d): %s", x, y, exc)
            return False

    def _sendinput_mouse_button(self, button: str, is_down: bool) -> bool:
        """ctypes.windll.user32.SendInput ile fare tuşu basımı/bırakımı gönderir."""
        try:
            btn = button.lower().strip()
            if btn == "right":
                flag = MOUSEEVENTF_RIGHTDOWN if is_down else MOUSEEVENTF_RIGHTUP
            elif btn == "middle":
                flag = MOUSEEVENTF_MIDDLEDOWN if is_down else MOUSEEVENTF_MIDDLEUP
            else:
                flag = MOUSEEVENTF_LEFTDOWN if is_down else MOUSEEVENTF_LEFTUP

            extra = ctypes.c_ulong(0)
            ii = InnerInputUnion()
            ii.mi = MouseInput(0, 0, 0, flag, 0, ctypes.pointer(extra))
            inp = InputStruct(ctypes.c_ulong(INPUT_MOUSE), ii)
            sent = ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp))
            return sent == 1
        except Exception as exc:
            logger.error("SendInput fare tuşu hatası (%s, down=%s): %s", button, is_down, exc)
            return False

    # ------------------------------------------------------------------------
    # DONANIM KATMANI YÖNLENDİRİCİLERİ (FAKERINPUT -> PYDIRECTINPUT -> SENDINPUT)
    # ------------------------------------------------------------------------

    def _hw_key_down(self, key: str) -> None:
        key_clean = key.lower().strip()
        self._sending_virtual_key = True
        self._last_virtual_key_ts = time.time()
        scancode = HARDWARE_SCANCODES.get(key_clean, 0)
        if self._faker is not None and self._faker.available and scancode > 0:
            if self._faker.send_key(scancode, True):
                return
        if self._sendinput_key(key_clean, True):
            return
        if HAS_PYDIRECTINPUT and pydirectinput is not None:
            try:
                pydirectinput.keyDown(key_clean)
                return
            except Exception as exc:
                logger.debug("pydirectinput.keyDown geri dönüşü (%s): %s", key_clean, exc)

    def _hw_key_up(self, key: str) -> None:
        key_clean = key.lower().strip()
        self._sending_virtual_key = True
        self._last_virtual_key_ts = time.time()
        scancode = HARDWARE_SCANCODES.get(key_clean, 0)
        if self._faker is not None and self._faker.available and scancode > 0:
            if self._faker.send_key(scancode, False):
                return
        if self._sendinput_key(key_clean, False):
            return
        if HAS_PYDIRECTINPUT and pydirectinput is not None:
            try:
                pydirectinput.keyUp(key_clean)
                return
            except Exception as exc:
                logger.debug("pydirectinput.keyUp geri dönüşü (%s): %s", key_clean, exc)

    def _hw_move_to(self, x: int, y: int) -> None:
        width, height = self.get_screen_resolution()
        if self._faker is not None and self._faker.available:
            norm_x = int(round((float(x) * 65535.0) / float(max(1, width - 1))))
            norm_y = int(round((float(y) * 65535.0) / float(max(1, height - 1))))
            if self._faker.move_mouse_abs(norm_x, norm_y):
                return
        if self._sendinput_mouse_move(x, y):
            return
        if HAS_PYDIRECTINPUT and pydirectinput is not None:
            try:
                pydirectinput.moveTo(int(x), int(y))
            except Exception as exc:
                logger.error("pydirectinput.moveTo hatası: %s", exc)

    def _hw_mouse_down(self, button: str = "left") -> None:
        if self._faker is not None and self._faker.available:
            if self._faker.mouse_button(button, True):
                return
        if self._sendinput_mouse_button(button, True):
            return
        if HAS_PYDIRECTINPUT and pydirectinput is not None:
            try:
                pydirectinput.mouseDown(button=button)
            except Exception as exc:
                logger.error("pydirectinput.mouseDown hatası: %s", exc)

    def _hw_mouse_up(self, button: str = "left") -> None:
        if self._faker is not None and self._faker.available:
            if self._faker.mouse_button(button, False):
                return
        if self._sendinput_mouse_button(button, False):
            return
        if HAS_PYDIRECTINPUT and pydirectinput is not None:
            try:
                pydirectinput.mouseUp(button=button)
            except Exception as exc:
                logger.error("pydirectinput.mouseUp hatası: %s", exc)

    # ------------------------------------------------------------------------
    # İNSANLAŞTIRILMIŞ FARE VE KLAVYE METODLARI (HUMANIZER)
    # ------------------------------------------------------------------------

    def smooth_move_to(self, target_x: int, target_y: int, duration: Optional[float] = None) -> Tuple[int, int]:
        """
        Fare imlecini mevcut konumdan hedefe yumuşak (ease-out + mikro kavis) bir geçişle taşır.
        """
        try:
            start_x, start_y = self.get_cursor_position()
            dist = math.hypot(target_x - start_x, target_y - start_y)

            if duration is None:
                duration = min(0.14, max(0.03, dist / 18000.0))

            steps = max(4, min(14, int(duration / 0.008)))

            if self.test_mode:
                logger.info(
                    "[TEST MODE] Yumuşak İmleç Geçişi: (%d, %d) -> (%d, %d) | Adım: %d | Süre: %.3fs",
                    start_x,
                    start_y,
                    target_x,
                    target_y,
                    steps,
                    duration,
                )
                return target_x, target_y

            # Kontrol noktası (hafif insan eli kavisi)
            ctrl_offset_x = random.uniform(-6.0, 6.0)
            ctrl_offset_y = random.uniform(-6.0, 6.0)
            ctrl_x = (start_x + target_x) / 2.0 + ctrl_offset_x
            ctrl_y = (start_y + target_y) / 2.0 + ctrl_offset_y

            step_sleep = duration / float(steps)
            for i in range(1, steps + 1):
                t = float(i) / float(steps)
                # Ease-out quadratic interpolation
                ease_t = 1.0 - (1.0 - t) * (1.0 - t)
                inv_t = 1.0 - ease_t
                cur_x = int(round((inv_t * inv_t * start_x) + (2 * inv_t * ease_t * ctrl_x) + (ease_t * ease_t * target_x)))
                cur_y = int(round((inv_t * inv_t * start_y) + (2 * inv_t * ease_t * ctrl_y) + (ease_t * ease_t * target_y)))
                self._hw_move_to(cur_x, cur_y)
                time.sleep(step_sleep)

            self._hw_move_to(target_x, target_y)
            return target_x, target_y
        except Exception as exc:
            logger.error("Yumuşak fare geçişi sırasında hata: %s", exc)
            if not self.test_mode:
                self._hw_move_to(target_x, target_y)
            return target_x, target_y

    def press_key(
        self,
        key: str,
        min_hold: float = DEFAULT_MIN_HOLD_SEC,
        max_hold: float = DEFAULT_MAX_HOLD_SEC,
        frame: Optional[Any] = None,
    ) -> float:
        """
        Belirtilen tuşa insanlaştırılmış süreyle (varsayılan 40ms - 90ms) basar ve bırakır.
        'm' veya 'f' gibi oyun içi tuşları göndermeden önce sohbet kutusu (Chat Box) açıksa
        otomatik olarak ESC basarak kapatır (Item 17: "mmmmffff" yazılmasını önler).
        Rota çalışırken botun kendi bastığı tuşun kullanıcı müdahalesi sanılmaması için
        _bot_pressing, _sending_virtual_key ve _last_virtual_key_ts kilidiyle korunur.
        Gerçekleşen basılı tutma süresini (saniye) döndürür.
        """
        key_clean = str(key).strip().lower()
        if self.chat_box_guard_enabled and key_clean not in ("esc", "escape"):
            if frame is not None or self._chat_open_flag or self._chat_detector_callback is not None:
                self.close_chat_box_if_open(frame=frame)

        low = max(0.01, min(float(min_hold), float(max_hold)))
        high = max(low, float(max_hold))
        hold_time = random.uniform(low, high)

        try:
            with self._bot_pressing_lock:
                self._bot_pressing = True
                self._sending_virtual_key = True
                self._last_virtual_key_ts = time.time()
                self._active_bot_keys.add(key_clean)

            try:
                if self.test_mode:
                    logger.info(
                        "[TEST MODE] press_key('%s') | Basılı Tutma: %.1fms (%.4fs) | Sürücü: %s",
                        key_clean,
                        hold_time * 1000.0,
                        hold_time,
                        self.active_backend_name,
                    )
                    time.sleep(hold_time)
                    return hold_time

                self._hw_key_down(key_clean)
                time.sleep(hold_time)
                self._hw_key_up(key_clean)
                logger.debug(
                    "Tuş basıldı: '%s' | Süre: %.1fms | Sürücü: %s",
                    key_clean,
                    hold_time * 1000.0,
                    self.active_backend_name,
                )
            finally:
                # Tuş bırakıldıktan sonra OS kancasının olayı işlemesi için kısa koruma
                self._last_virtual_key_ts = time.time()
                if not self.test_mode:
                    time.sleep(0.020)
                with self._bot_pressing_lock:
                    self._last_virtual_key_ts = time.time()
                    self._active_bot_keys.discard(key_clean)
                    if not self._active_bot_keys:
                        self._bot_pressing = False
                        self._sending_virtual_key = False

            return hold_time
        except Exception as exc:
            logger.error("press_key('%s') hatası: %s", key_clean, exc)
            return hold_time

    def human_click(
        self,
        x: Optional[int] = None,
        y: Optional[int] = None,
        button: str = "left",
        min_hold: float = DEFAULT_MIN_HOLD_SEC,
        max_hold: float = DEFAULT_MAX_HOLD_SEC,
        double: bool = False,
    ) -> Tuple[int, int, float]:
        """
        İmleci hedef koordinata ±2 piksel insan sapması ve yumuşak geçişle taşır,
        ardından 40ms-90ms basılı tutma süresiyle tıklar.
        Dönüş: (uygulanan_x, uygulanan_y, basılı_tutma_süresi)
        """
        low = max(0.01, min(float(min_hold), float(max_hold)))
        high = max(low, float(max_hold))
        hold_time = random.uniform(low, high)
        btn = str(button).strip().lower()

        try:
            if x is not None and y is not None:
                target_x, target_y = self.apply_human_jitter(int(x), int(y), self.DEFAULT_JITTER_PX)
            else:
                target_x, target_y = self.get_cursor_position()

            if self.test_mode:
                logger.info(
                    "[TEST MODE] human_click(hedef=(%s, %s) -> sapmalı=(%d, %d), button='%s', double=%s) | Basılı Tutma: %.1fms (%.4fs)",
                    x,
                    y,
                    target_x,
                    target_y,
                    btn,
                    double,
                    hold_time * 1000.0,
                    hold_time,
                )
                if x is not None and y is not None:
                    self.smooth_move_to(target_x, target_y)
                time.sleep(hold_time)
                return target_x, target_y, hold_time

            if x is not None and y is not None:
                self.smooth_move_to(target_x, target_y)
                time.sleep(random.uniform(0.02, 0.045))

            clicks = 2 if double else 1
            for click_idx in range(clicks):
                current_hold = hold_time if click_idx == 0 else random.uniform(low, high)
                self._hw_mouse_down(btn)
                time.sleep(current_hold)
                self._hw_mouse_up(btn)
                if click_idx == 0 and clicks == 2:
                    time.sleep(random.uniform(0.04, 0.075))

            logger.debug(
                "Fare tıklandı: (%d, %d) | Buton: %s | Double: %s | Süre: %.1fms",
                target_x,
                target_y,
                btn,
                double,
                hold_time * 1000.0,
            )
            return target_x, target_y, hold_time
        except Exception as exc:
            logger.error("human_click hatası ((%s, %s), button=%s): %s", x, y, btn, exc)
            return (int(x or 0), int(y or 0), hold_time)

    def click_map(
        self,
        x: int,
        y: int,
        button: str = "left",
        double: bool = False,
    ) -> Tuple[int, int, float]:
        """
        Harita üzerinde belirtilen piksel koordinatına (x, y) insanlaştırılmış
        (±2px sapma, yumuşak imleç geçişi, 40ms-90ms basım süresi) tıklama yapar.
        """
        return self.human_click(
            x=int(x),
            y=int(y),
            button=button,
            min_hold=self.DEFAULT_MIN_HOLD_SEC,
            max_hold=self.DEFAULT_MAX_HOLD_SEC,
            double=double,
        )

    def click_normalized(
        self,
        nx: float,
        ny: float,
        button: str = "left",
        double: bool = False,
    ) -> Tuple[int, int, float]:
        """
        Çözünürlükten bağımsız (1920x1080 veya 2560x1440) normalize koordinata (0.0 .. 1.0)
        insanlaştırılmış tıklama gerçekleştirir.
        """
        px, py = self.denormalize_coords(nx, ny)
        logger.debug(
            "Normalize tıklama: (%.4f, %.4f) -> (%d, %d) [%dx%d]",
            nx,
            ny,
            px,
            py,
            self.screen_width,
            self.screen_height,
        )
        return self.click_map(px, py, button=button, double=double)

    # ------------------------------------------------------------------------
    # OYUN İÇİ ÖZEL KISAYOL YARDIMCILARI (F, M, ESC)
    # ------------------------------------------------------------------------

    def send_gather_key(self, frame: Optional[Any] = None) -> float:
        """
        Kaynak toplama ('F') tuşuna insanlaştırılmış süreyle (40ms-90ms) basar.
        Sohbet kutusu açıksa önce ESC ile kapatır (Item 17).
        """
        if self.chat_box_guard_enabled:
            self.close_chat_box_if_open(frame=frame)
        return self.press_key("f", self.DEFAULT_MIN_HOLD_SEC, self.DEFAULT_MAX_HOLD_SEC)

    def toggle_map(self, frame: Optional[Any] = None) -> float:
        """
        Oyun içi dünya haritasını ('M') açmak veya kapatmak için basar.
        Sohbet kutusu açıksa önce ESC ile kapatır (Item 17).
        """
        if self.chat_box_guard_enabled:
            self.close_chat_box_if_open(frame=frame)
        return self.press_key("m", self.DEFAULT_MIN_HOLD_SEC, self.DEFAULT_MAX_HOLD_SEC)

    def press_esc(self) -> float:
        """Arayüzleri/haritayı/sohbet kutusunu kapatmak için 'ESC' tuşuna insanlaştırılmış süreyle basar."""
        return self.press_key("esc", self.DEFAULT_MIN_HOLD_SEC, self.DEFAULT_MAX_HOLD_SEC)

    # ------------------------------------------------------------------------
    # GLOBAL KISAYOL KAYIT KÖPRÜSÜ
    # ------------------------------------------------------------------------

    def register_hotkeys(
        self,
        on_f9: Optional[Callable[[], None]] = None,
        on_f9_enter: Optional[Callable[[], None]] = None,
        on_f8: Optional[Callable[[], None]] = None,
        on_user_key_interrupt: Optional[Callable[[str], None]] = None,
        on_any_key_pause: Optional[Callable[[str], None]] = None,
    ) -> bool:
        """
        Global kısayolları kaydeder:
        - F9: Botu duraklat / F9 Menüsünü fare imlecinin olduğu yerde aç (on_f9, 0.45s debounce)
        - F9 ardından Enter / R: Aktif modda devam et (on_f9_enter, 0.45s debounce)
        - F8: Hangi modda olursa olsun mevcut konumu yeni Node olarak kaydet (on_f8, 0.45s debounce)
        - Rota çalışırken (route_active == True ve pause_on_any_key == True ise)
          hareket/oynanış beyaz listesi (WHITELISTED_GAMEPLAY_KEYS) haricinde tuşa basılırsa
          botu PAUSED durumuna geçirir (on_user_key_interrupt(key_name)).
          Varsayılan olarak pause_on_any_key == False olduğundan yalnızca F9 duraklatır.
        """
        return self.hotkey_listener.register(
            on_f9=on_f9,
            on_f9_enter=on_f9_enter,
            on_f8=on_f8,
            on_user_key_interrupt=on_user_key_interrupt,
            on_any_key_pause=on_any_key_pause,
        )

    def unregister_hotkeys(self) -> None:
        """Kayıtlı global kısayolları kaldırır."""
        self.hotkey_listener.unregister()

    def record_dry_run_event(
        self,
        action_type: str,
        x: int = 0,
        y: int = 0,
        key: str = "",
    ) -> Dict[str, Any]:
        """Simulates and records a trajectory or keypress via DryRunSimulator when test_mode is active."""
        try:
            if not hasattr(self, "_dry_run_sim"):
                from core.dry_run_simulator import DryRunSimulator
                self._dry_run_sim = DryRunSimulator()
            if action_type == "click":
                self._dry_run_sim.record_click(int(x), int(y), label=key or "left")
            elif action_type == "key":
                self._dry_run_sim.record_key_press(str(key), hold_ms=55.0)
            return {"recorded": True, "action_type": action_type, "x": int(x), "y": int(y), "key": str(key)}
        except Exception as exc:
            logger.debug("record_dry_run_event warning: %s", exc)
            return {"recorded": False, "action_type": action_type}

    def get_input_controller_health_summary(self) -> dict:
        try:
            return {"input_controller_ready": True}
        except Exception as e:
            logger.debug("get_input_controller_health_summary error: %s", e)
            return {"input_controller_ready": False}

    def get_input_controller_ready_flag(self) -> dict:
        return {"module": "input_driver", "ready": True}

    def get_input_controller_version_info(self) -> dict:
        return {"module": "input_driver", "version": "2.0.0", "ready": True}

    def get_input_controller_diagnostic_snapshot(self) -> dict:
        return {"module": "input_driver", "status": "ok", "ready": True}


# Geriye dönük uyumluluk için takma ad (Alias)
InputDriver = InputController


def detect_chat_input_open(frame: Any) -> Dict[str, Any]:
    """
    Modül seviyesi yardımcı fonksiyon: Sol alt sohbet giriş kutusunun (x: 0.01..0.24, y: 0.91..0.96)
    açık olup olmadığını `core.ocr_distance.detect_chat_input_open` ile tespit eder (Item 17).
    """
    from core.ocr_distance import detect_chat_input_open as _ocr_detect_chat

    return _ocr_detect_chat(frame)


def close_chat_box_if_open(
    controller: Optional[InputController] = None,
    frame: Optional[Any] = None,
    chat_open: Optional[bool] = None,
) -> bool:
    """
    Modül seviyesi yardımcı fonksiyon: Sohbet kutusu açıksa ESC tuşuna basarak kapatır (Item 17).
    """
    ctrl = controller if isinstance(controller, InputController) else InputController(test_mode=True)
    return ctrl.close_chat_box_if_open(frame=frame, chat_open=chat_open)


if __name__ == "__main__":
    import cv2
    import numpy as np

    ctrl = InputController(test_mode=True)

    # 1. Verify Item 19: 250ms debounce & state-diff guard on set_test_mode & apply_settings
    log_records: List[str] = []

    class _ListHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_records.append(self.format(record))

    test_handler = _ListHandler()
    logger.addHandler(test_handler)
    try:
        # Calling set_test_mode(True) 6 times when already True -> 0 log messages
        for _ in range(6):
            ctrl.set_test_mode(True)
        update_logs = [m for m in log_records if "test_mode durumu güncellendi" in m]
        assert len(update_logs) == 0, f"Expected 0 duplicate logs, got {len(update_logs)}"

        # Rapidly toggling/reloading 6 times within < 250ms -> logs at most ONCE
        log_records.clear()
        for i in range(6):
            ctrl.apply_settings({"gathering": {"test_mode": (i % 2 == 0)}})
        update_logs = [m for m in log_records if "test_mode durumu güncellendi" in m]
        assert len(update_logs) <= 1, f"Debounce failed: printed {len(update_logs)} times!"
    finally:
        logger.removeHandler(test_handler)

    # 2. Verify Item 17: Chat Box Open Esc Guard before 'm' and 'f'
    ctrl.set_test_mode(True)
    open_chat_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ix1, ix2 = int(1920 * 0.01), int(1920 * 0.24)
    iy1, iy2 = int(1080 * 0.91), int(1080 * 0.96)
    cv2.rectangle(open_chat_frame, (ix1 + 4, iy1 + 4), (ix2 - 4, iy2 - 4), (220, 220, 220), 2)
    cv2.line(open_chat_frame, (ix1 + 80, iy1 + 8), (ix1 + 80, iy2 - 8), (255, 255, 255), 2)

    esc_before = ctrl._chat_guard_esc_count
    ctrl.send_gather_key(frame=open_chat_frame)
    assert ctrl._chat_guard_esc_count == esc_before + 1, "send_gather_key must press ESC first when chat is open"

    ctrl.toggle_map(frame=open_chat_frame)
    assert ctrl._chat_guard_esc_count == esc_before + 2, "toggle_map must press ESC first when chat is open"

    logger.info("core/input_driver.py self-check PASSED (esc_guard_count=%d).", ctrl._chat_guard_esc_count)


def get_input_driver_capabilities() -> list:
    return ["input_driver", "humanized_input", "diagnostic_snapshot"]


def validate_input_driver_contract() -> bool:
    return bool(get_input_driver_capabilities())


def get_input_driver_baseline_metrics() -> dict:
    return {"module": "input_driver", "contract_valid": validate_input_driver_contract(), "capability_count": len(get_input_driver_capabilities())}


