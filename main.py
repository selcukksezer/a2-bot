# -*- coding: utf-8 -*-
"""
Gatherion-Local v2.0 (Türkçe) — Master Orchestrator Entry Point (main.py)
Integrates all specialized agent modules:
- Core Configuration & Encrypted Vault (config/schema.py, core/opsec_crypto.py)
- Hardware Input, WindMouse & Native Hotkeys (core/input_driver.py, core/windmouse_humanizer.py)
- High-Speed DXGI/MSS Screen Capture & Vision Engine (core/capture.py, core/vision.py, core/stealth_capture.py)
- Route, Region & Node Manager + Questlog Auto-Importer (core/route_manager.py, core/questlog_importer.py)
- Passive 3D/Minimap Auto-Cartography, 3x Harvest & Anti-Ping-Pong Sequential Router (core/material_scanner.py,
  core/auto_cartographer.py, core/sequential_router.py, core/explorer_mode.py)
- Combat, Death, Stuck Recovery & Behavioral Guard (core/combat_recovery.py, core/behavioral_guard.py)
- Ex-Gaussian Human Timing, Telemetry & Watchdog (core/human_timing.py, core/telemetry.py, core/watchdog.py)
- Mobile Remote Control Server & Cloud Tunnel (core/mobile_server.py, core/mobile_tunnel.py, ui/mobile_connect_dialog.py)
- PyQt6 Dark Turkish UI Suite (ui/overlay_hud.py, ui/f9_menu.py, ui/settings_window.py, ui/map_viewer.py,
  ui/cartography_radar.py, ui/analytics_dialog.py, ui/calibration_wizard.py, ui/guide_dialog.py)
"""

import os
import sys
from version import __version__


def _resolve_bot_home() -> str:
    """Bot home dizini — platform-bağımsız çözümleme (P2-25, bölüm 8).

    Sıra: ``GATHERION_HOME`` ortam değişkeni (varsa ve dizinse) ->
    Windows'ta eski ``F:\\a2`` / ``F:\\Gatherion`` (sürücü varsa) ->
    "" (repo-yerel yedek; log/temp/ayar repo altına çözülür).
    Üç ayrı yerdeki dağınık ``F:\\`` kontrolünün tek adresidir;
    F: olmayan PC'de (veya Windows-dışı sistemde) yollar sessizce
    başka bir diske kaymaz, repo altına düşer.
    """
    env_home = os.environ.get("GATHERION_HOME", "").strip()
    if env_home and os.path.isdir(env_home):
        return env_home
    if os.name == "nt":
        if os.path.isdir(r"F:\a2"):
            return r"F:\a2"
        if os.path.isdir(r"F:\\"):
            return r"F:\Gatherion"
    return ""


def _attach_windowless_stdio() -> None:
    """pythonw.exe has no console; append prints and tracebacks to logs/gatherion.log."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    _home = _resolve_bot_home()
    log_dir = os.path.join(_home, "logs") if _home else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "logs"
    )
    # F-1254: import anında dosya açılışı çökmesin; hata olursa stderr'de kalınır.
    try:
        os.makedirs(log_dir, exist_ok=True)
        stream = open(
            os.path.join(log_dir, "gatherion.log"),
            "a",
            encoding="utf-8",
            errors="replace",
            buffering=1,
        )
    except Exception:
        # sweep-26 F9: gatherion.log açılamazsa pythonw altında hata tamamen
        # görünmez kalırdı. Son çare: %TEMP% altına tek satırlık iz bırak;
        # o da açılamazsa gerçekten yazılacak yer yoktur.
        try:
            import tempfile

            with open(
                os.path.join(tempfile.gettempdir(), "gatherion-stdio-error.txt"),
                "a",
                encoding="utf-8",
                errors="replace",
            ) as fh:
                fh.write("gatherion.log acilamadi; pythonw stdio baglanamadi.\n")
        except Exception:
            pass
        return
    if sys.stdout is None:
        sys.stdout = stream
    if sys.stderr is None:
        sys.stderr = stream


_attach_windowless_stdio()

# sweep-26 F8: _bot_home() logging kurulmadan önce çağrılır; F: yoksa
# sessizce sistem varsayılanlarına düşülürdü. Uyarı metni burada saklanır,
# main() içinde logger kurulduktan sonra warning olarak basılır.
_BOOT_HOME_WARNING: str = ""


def _show_startup_alert(message: str, title: str = "Gatherion") -> None:
    """Windows mesaj kutusu — pythonw altında stderr görünmez (F9), bu
    yüzden kritik açılış hataları ayrıca MessageBox ile gösterilir.
    Windows-dışı platformda ve hata halinde sessizce atlanır."""
    if os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, message, title, 0x40)  # MB_ICONINFORMATION
    except Exception as exc:
        try:
            logger.debug("Başlangıç uyarı penceresi gösterilemedi: %s", exc)
        except Exception:
            pass


# sweep-26 F1: tek instance kilidi. CreateMutex hiç yoktu; çift bot aynı
# anda çalışabiliyordu. Oturum-içi (Local) mutex: yönetici yetkisi
# gerektirmez, süreç bitince işletim sistemi serbest bırakır.
_SINGLE_INSTANCE_MUTEX_NAME = "Gatherion-Local-v2.0-SingleInstance"
_SINGLE_INSTANCE_MUTEX_HANDLE = None


def _acquire_single_instance_lock() -> bool:
    """True: bu örnek tek örnek (kilit alındı). False: başka bir örnek
    zaten çalışıyor. Windows-dışı platformda ve mutex oluşturulamazsa
    kilitsiz devam edilir (eski davranış — fail-open, sessiz değil:
    oluşturamama debug loglanır)."""
    global _SINGLE_INSTANCE_MUTEX_HANDLE
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        kernel32.CreateMutexW.restype = wintypes.HANDLE
        kernel32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        kernel32.GetLastError.restype = wintypes.DWORD
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel32.CreateMutexW(None, True, _SINGLE_INSTANCE_MUTEX_NAME)
        if not handle:
            logger.debug("Tek instance mutex'i oluşturulamadı; kilitsiz devam ediliyor.")
            return True
        if kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
            kernel32.CloseHandle(handle)
            return False
        _SINGLE_INSTANCE_MUTEX_HANDLE = handle
        return True
    except Exception as exc:
        logger.debug("Tek instance kilidi kurulamadı: %s", exc)
        return True


def _release_single_instance_lock() -> None:
    """Kapatışta mutex'i bırak (süreç bitince OS zaten bırakır; bu açık bırakma)."""
    global _SINGLE_INSTANCE_MUTEX_HANDLE
    handle = _SINGLE_INSTANCE_MUTEX_HANDLE
    _SINGLE_INSTANCE_MUTEX_HANDLE = None
    if handle is None or os.name != "nt":
        return
    try:
        import ctypes

        ctypes.windll.kernel32.CloseHandle(handle)
    except Exception as exc:
        logger.debug("Tek instance mutex'i bırakılamadı: %s", exc)


def _bot_home() -> str:
    """Bot home dizini (``_resolve_bot_home``); yoksa "" (repo-yerel yedek)."""
    home = _resolve_bot_home()
    if home:
        os.makedirs(home, exist_ok=True)
        return home
    global _BOOT_HOME_WARNING
    _BOOT_HOME_WARNING = (
        "F: surucusu bulunamadi; TEMP/TORCH/YOLO yollari pinlenmedi, "
        "sistem varsayilanlariyla devam ediliyor (ayarlar/rota F: uzerindeyse bot baska dizindeki bos ayarlarla acilmis olabilir)."
    )
    return ""


def _pin_bot_env() -> None:
    """Point temp and model caches at the F: home before Ultralytics or Torch import."""
    home = _bot_home()
    if not home:
        return
    temp_dir = os.path.join(home, "temp")
    appdata_dir = os.path.join(home, "appdata")
    torch_home = os.path.join(home, "torch")
    pip_cache = os.path.join(home, "pip-cache")
    for path in (temp_dir, appdata_dir, torch_home, pip_cache):
        os.makedirs(path, exist_ok=True)
    os.environ["TEMP"] = temp_dir
    os.environ["TMP"] = temp_dir
    os.environ["YOLO_CONFIG_DIR"] = appdata_dir
    os.environ["ULTRALYTICS_CONFIG_DIR"] = appdata_dir
    os.environ["TORCH_HOME"] = torch_home
    os.environ["PIP_CACHE_DIR"] = pip_cache


_pin_bot_env()

import logging

# P0-9 (K-15): logger LOG_DIR çözümünden ÖNCE bağlanır. setup_logging()
# LOG_FILE'a (dolayısıyla LOG_DIR'a) bağımlı olduğu için logger ancak
# aşağıda kurulabiliyordu; oysa LOG_DIR makedirs koruması (F-1254)
# logger.debug kullanıyordu — makedirs hatasında import NameError ile
# çöküyordu. Erken logger handler'sızdır (root'a propagate eder);
# setup_logging() aşağıda aynı logger nesnesine (getLogger aynı adı
# döndürür) dosya/stdout handler'larını ekler.
logger = logging.getLogger("Gatherion.Main")

import socket
import threading
import time

# [STARTUP] Python import süresi ölçümü (main() içinde loglanır).
_IMPORT_T0 = time.monotonic()
import weakref
from typing import Any, Dict, Optional

try:
    import winsound
except ImportError:  # Windows-only modul; Linux/macOS'ta yok (R22)
    winsound = None  # type: ignore[assignment]

from PyQt6.QtCore import QObject, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QAction, QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QApplication, QMenu, QMessageBox, QSystemTrayIcon

# Ensure workspace root is on sys.path and stdout uses UTF-8 for Turkish output
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# v20/session-logs: oturum dosyasi yardimcilari (core/session_log.py;
# yalnizca stdlib — core/__init__ yalnizca `version` import eder).
from core.session_log import prune_old_sessions, resolve_session_log_path

# v20/bot-ani-kapanma: sessiz process olumu kanit altyapisi (faulthandler +
# excepthook + atexit kapanis kaydi + RSS izleme). Yalnizca stdlib
# (+istege bagli psutil); Qt yok, main.py import aninda guvenle alinir.
from core.crash_evidence import install_crash_evidence, note_graceful_shutdown

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _stdout_exc:
        os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _stderr_exc:
        os.environ["PYTHONIOENCODING"] = "utf-8"

# Configure root & file logger. New lines go to the bot home logs dir
# (P2-25: _resolve_bot_home) so C: is not the log disk; without a bot
# home they stay repo-local.
def _resolve_log_dir() -> str:
    home = _resolve_bot_home()
    if home:
        path = os.path.join(home, "logs")
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            # P0-9 (K-15): F-1254 sözü — log dizini oluşturulamazsa
            # import çökmesin. Buradaki makedirs korumasızdı; hata
            # LOG_DIR atamasında import'u öldürüyordu. Repo-yerel
            # yedeğe düşülür; aşağıdaki makedirs koruması durumu
            # (artık bağlı olan) logger ile kaydeder.
            return os.path.join(BASE_DIR, "logs")
        return path
    return os.path.join(BASE_DIR, "logs")


LOG_DIR = _resolve_log_dir()
# F-1254: log dizini oluşturulamazsa import çökmesin; setup_logging
# dosya handler'ı açamazsa yalnızca stdout'a loglar.
try:
    os.makedirs(LOG_DIR, exist_ok=True)
except Exception as exc:
    logger.debug("Log dizini oluşturulamadı: %s", exc)
# v20/session-logs: tek upuzun bot.log yerine her başlatmada yeni oturum
# dosyası (logs/session_YYYY-MM-DD_HHMMSS.log); en son 20 tutulur.
# LOG_FILE adı korunur — copy_report_to_clipboard / open_log_file /
# hata mesajları aynı adı kullanır, artık oturum dosyasını gösterir.
SESSION_LOG_FILE = os.fspath(resolve_session_log_path(LOG_DIR))
LOG_FILE = SESSION_LOG_FILE
LOG_FORMAT = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"
# sweep-29 O-5: saat biçiminde tarih yoktu — çok günlük tek dosyada aynı
# saat iki güne ait olabiliyordu; panik satırının günü belirsizdi.
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

# Base anchored paths for configuration, dataset, models, templates, and routes
CONFIG_DIR = os.path.join(BASE_DIR, "config")
SETTINGS_JSON_PATH = os.path.join(CONFIG_DIR, "settings.json")
CALIBRATED_ROIS_PATH = os.path.join(CONFIG_DIR, "calibrated_rois.json")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
LEARNED_EXEMPLARS_PATH = os.path.join(DATASET_DIR, "learned_exemplars.json")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
ROUTES_DIR = os.path.join(BASE_DIR, "routes")


def setup_logging() -> logging.Logger:
    """Configures unified root logging to the current session log file
    (logs/session_YYYY-MM-DD_HHMMSS.log) and stdout without duplicate handlers."""
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATEFMT)

    # Clear any pre-existing handlers on root to prevent duplicate lines
    for existing in list(root_logger.handlers):
        root_logger.removeHandler(existing)

    file_handler = None
    try:
        # v20/session-logs: oturum dosyası başlatmada bir kez açılır;
        # rotasyon yok — her oturum kendi dosyasında, eskiler budanır
        # (eski tek dosya bot.log 5 MB'de dönüyor, 5 yedek tutuyordu).
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        # En son 20 oturum tutulur; eskiler sessizce silinir (F-1254:
        # budama hatası import'u öldürmez).
        prune_old_sessions(LOG_DIR)
    except Exception as exc:
        # F-1254: log dosyası açılamazsa (salt okunur disk vb.) import
        # çökmez; yalnızca stdout handler ile devam edilir.
        print(f"WARNING: log dosyası açılamadı ({LOG_FILE}): {exc}", file=sys.stderr)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    if file_handler is not None:
        root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    return logging.getLogger("Gatherion.Main")


def deduplicate_loggers() -> None:
    """
    Deduplicates handlers across all loggers in logging.root.manager.loggerDict.
    Removes redundant StreamHandlers from child loggers and ensures messages propagate
    cleanly to the single root StreamHandler + FileHandler so every log line is printed
    EXACTLY ONCE.
    """
    root_logger = logging.getLogger()
    seen_stream = False
    seen_file = False
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            if not seen_file:
                seen_file = True
            else:
                root_logger.removeHandler(handler)
        elif isinstance(handler, logging.StreamHandler):
            if not seen_stream:
                seen_stream = True
            else:
                root_logger.removeHandler(handler)

    for _logger_name, logger_obj in list(logging.root.manager.loggerDict.items()):
        if not isinstance(logger_obj, logging.Logger):
            continue
        for child_handler in list(logger_obj.handlers):
            if isinstance(child_handler, logging.StreamHandler) and not isinstance(
                child_handler, logging.FileHandler
            ):
                logger_obj.removeHandler(child_handler)
        if logger_obj.handlers:
            logger_obj.propagate = False
        else:
            logger_obj.propagate = True


logger = setup_logging()

# Core imports (Agents 1-7 & 11-25 + Enterprise Suite & 50-Point Vision/Navigation Guards)
from config.schema import SettingsManager
from core.behavioral_guard import BehavioralStealthGuard
from core.capture import ScreenCapture
from core.combat_recovery import CombatRecovery
from core.depth_glide_navigator import (
    GlideWaypointController,
    MonocularDepthObstacleAvoidance,
)
from core.dpi_focus_guard import (
    DPIScalingSynchronizer,
    ForegroundFocusGuard,
    RouteBackupManager,
    SelfHealingTemplateTuner,
)
from core.dry_run_simulator import DryRunReplaySimulator
from core.economy_broker import (
    AutoCraftingMorphQueue,
    SessionProfitSimulator,
    TradeBrokerPriceScanner,
)
from core.explorer_mode import ExplorerModeController
from core.human_timing import HumanTimingEngine, OrganicAntiAFK
from core.incident_recorder import IncidentRecorder, MobileTokenAuthenticator
from core.input_driver import InputController
from core.material_scanner import MaterialScanner
from core.minimap_scanner import MinimapScanner
from core.multi_char_scheduler import (
    AltCharacterManager,
    AutomationTaskScheduler,
    PetAndBuffManager,
)
from core.neural_audio_perception import AudioLoopbackFFTAnalyzer, ONNXYolo3DDetector
from core.ocr_distance import AdaptivePreprocessor, GameHUDReader
from core.opsec_crypto import LocalVaultCrypto, NativeWin32HotkeyManager, PanicAlertSystem
from core.patch_normalizer import (
    HighResROICropper,
    MinimapOrientationGuard,
    MotionBlurGate,
    SobelClahePatchNormalizer,
    WatershedClusterSeparator,
)
from core.path_optimizer import RouteOptimizer
from core.questlog_importer import QuestlogImporter
from core.route_manager import RouteManager
from core.sequential_router import SequentialRoutePlanner
from core.skill_bar_reader import MicroCameraLookAround, SkillBarReader
from core.state_machine import BotMode, BotWorker
from core.stealth_capture import AdaptiveFramePacer, DXGIDesktopCapture, StealthOverlayManager
from core.telemetry import TelemetryEngine
from core.uac_elevator import ensure_admin_elevation, is_admin
from core.vision import VisionEngine
from core.vision_filters import (
    DynamicInteractionKeyReader,
    FeatherNegativeFilter,
    MonsterCircularityFilter,
    NameplateHSVIsolator,
    Pyramid3DCrestDetector,
)
from core.watchdog import BotWatchdog, SupportDiagnostics
from core.windmouse_humanizer import BiometricClickGenerator, PhysicalInputGuard, WindMouseEngine

# UI imports (Agents 8-10, 12, 13, 20)
from ui.app_shell import GatherionShell
from ui.analytics_dialog import AnalyticsDialog
from ui.calibration_wizard import CalibrationWizard
from ui.cartography_radar import CartographyRadarDialog
from ui.f9_menu import F9Menu
from ui.guide_dialog import GuideDialog
from ui.map_viewer import MapViewerWindow
from ui.overlay_hud import OverlayHUD, OverlaySignals
from ui.settings_window import SettingsWindow

# Deduplicate any StreamHandlers added at module import time
deduplicate_loggers()


_LAN_IP_CACHE_TTL_S = 60.0
_LAN_IP_CACHE: Dict[str, Any] = {"ip": "", "ts": 0.0}


def get_local_lan_ip() -> str:
    """Resolves the primary LAN IPv4 address for the Mobile Remote Dashboard URL.

    The result is cached for _LAN_IP_CACHE_TTL_S seconds: the UDP probe (plus
    a possible DNS lookup) is pure overhead when the dashboard URL is asked
    for repeatedly, and a LAN address does not change second to second.
    """
    now = time.monotonic()
    cached_ip = _LAN_IP_CACHE.get("ip", "")
    if cached_ip and now - _LAN_IP_CACHE.get("ts", 0.0) < _LAN_IP_CACHE_TTL_S:
        return str(cached_ip)
    resolved = "127.0.0.1"
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.settimeout(0.5)
            sock.connect(("8.8.8.8", 80))
            ip_addr = sock.getsockname()[0]
            if ip_addr and not ip_addr.startswith("127."):
                resolved = str(ip_addr)
    except Exception:
        logger.debug("UDP LAN IP probe fallback")
    if resolved == "127.0.0.1":
        try:
            hostname_ip = socket.gethostbyname(socket.gethostname())
            if hostname_ip:
                resolved = str(hostname_ip)
        except Exception:
            logger.debug("Hostname IP lookup fallback")
    _LAN_IP_CACHE["ip"] = resolved
    _LAN_IP_CACHE["ts"] = now
    return resolved


def _safe_reconnect(signal: Any, slot: Any) -> bool:
    """Disconnect-then-connect a Qt signal, tolerating stale/missing links.

    Disconnect failures (slot was never connected) are swallowed, exactly
    like the inline try/except blocks this replaces; connect failures are
    debug-logged. Returns True only when connect() succeeded, never raises.
    """
    disconnect = getattr(signal, "disconnect", None)
    if callable(disconnect):
        try:
            disconnect(slot)
        except Exception:
            pass
    connect = getattr(signal, "connect", None)
    if not callable(connect):
        return False
    try:
        connect(slot)
        return True
    except Exception as exc:
        logger.debug("Sinyal bağlama hatası: %s", exc)
        return False


def _reload_detector(detector: Any, target_path: str, label: str) -> bool:
    """Point one ONNX detector at target_path and reload it.

    Returns True only when the detector reports a successful load.
    A missing detector/loader, a False load result, or an exception all
    return False (and are logged) instead of raising — F-1137: a False
    load result must never be reported as a live update.
    """
    if detector is None:
        return False
    try:
        detector.model_path = target_path
    except Exception as exc:
        logger.warning("%s model yolu ayarlanamadı: %s", label, exc)
        return False
    loader = getattr(detector, "_load_model_if_available", None)
    if not callable(loader):
        return False
    try:
        loaded = bool(loader())
    except Exception as exc:
        logger.warning("%s yeniden yükleme hatası: %s", label, exc)
        return False
    if not loaded:
        logger.warning("%s modeli yükleyemedi: %s", label, target_path)
    return loaded


def _stop_quietly(target: Any, method_name: str, label: str) -> bool:
    """Call target.method_name() if present. Errors are NOT swallowed
    silently (sweep Y-4): they are logged as warnings with the label so a
    failing shutdown step is visible in bot.log. Returns True only when
    the call ran without error. A missing target/method is a no-op
    returning False."""
    if target is None:
        return False
    method = getattr(target, method_name, None)
    if not callable(method):
        return False
    try:
        method()
        return True
    except Exception as exc:
        logger.warning("%s kapanış hatası (yoksayıldı): %s", label, exc, exc_info=True)
        return False


# Menu action -> (BotMode value or None for plain resume, watchdog state).
# Single source of truth for the six mode branches of handle_menu_action.
_MENU_MODE_ACTIONS: Dict[str, tuple] = {
    "resume": (None, "WALKING"),
    "auto_record": (BotMode.AUTO_RECORD.value, "WALKING"),
    "manual_record": (BotMode.MANUAL_RECORD.value, "WAITING"),
    "explorer": (BotMode.EXPLORER_MODE.value, "WALKING"),
    "explorer_mode": (BotMode.EXPLORER_MODE.value, "WALKING"),
    "combat": (BotMode.COMBAT_MODE.value, "FIGHTING"),
    "combat_mode": (BotMode.COMBAT_MODE.value, "FIGHTING"),
    "savas": (BotMode.COMBAT_MODE.value, "FIGHTING"),
    "savaş": (BotMode.COMBAT_MODE.value, "FIGHTING"),
    "fight": (BotMode.COMBAT_MODE.value, "FIGHTING"),
    "route": (BotMode.ROUTE_MODE.value, "WALKING"),
    "route_mode": (BotMode.ROUTE_MODE.value, "WALKING"),
    "gather": (BotMode.ROUTE_MODE.value, "WALKING"),
    "topla": (BotMode.ROUTE_MODE.value, "WALKING"),
}


class HotkeyBridge(QObject):
    """Thread-safe Qt signal bridge for global hotkeys (F9, F9+Enter, F8, F10, user interrupt)."""

    f9_pressed = pyqtSignal()
    f9_resume_pressed = pyqtSignal()
    f8_pressed = pyqtSignal()
    f10_pressed = pyqtSignal()
    user_key_interrupted = pyqtSignal(str)


def create_tray_icon_pixmap() -> QIcon:
    """Creates a clean cyan/emerald Gatherion emblem icon in memory for the System Tray."""
    pixmap = QPixmap(64, 64)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

    # Outer dark circle — 3px cyan ring (not 1px): the 64px pixmap is
    # downscaled to ~16px in the tray, where a 1px ring aliases away and
    # the dark disc blends into a dark taskbar. Ring colour unchanged.
    painter.setBrush(QColor("#12161c"))
    painter.setPen(QPen(QColor("#38bdf8"), 3))
    painter.drawEllipse(4, 4, 56, 56)

    # Inner glowing diamond / node emblem
    painter.setBrush(QColor("#22c55e"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(20, 20, 24, 24)
    painter.end()
    return QIcon(pixmap)


class UiClickSounds:
    """Kısa tıklama sesi. Eksik olursa tek tık eğitim 'sounds' ile çökmez."""

    def play_click(self) -> None:
        # Backend'i çağrı anında çöz: modül import anında bağlanan
        # winsound, test stub'ları kurulmadan önce bağlanmış olabilir
        # (örn. d81'in no-op stub'u) ya da Linux'ta None olabilir (R22
        # koruması). sys.modules'taki güncel kayıt her zaman önceliklidir.
        try:
            backend = sys.modules.get("winsound", winsound)
            if backend is None:
                logger.debug("Tıklama sesi atlandı")
                return
            backend.Beep(880, 40)
        except Exception:
            logger.debug("Tıklama sesi atlandı", exc_info=True)


# S-0204: modül düzeyi readiness probunun başvurduğu canlı orchestrator
# (zayıf referans — GC'yi engellemez; kayıt yoksa yalnızca sözleşme probu
# çalışır). GatherionOrchestrator.__init__'de kaydedilir, shutdown()'da
# temizlenir.
_LIVE_ORCHESTRATOR_REF = None


class GatherionOrchestrator(QObject):
    """
    Master Application Orchestrator binding all specialized agent subsystems,
    encrypted local config, stealth overlay HUD, F9 popup menu, settings window,
    live auto-cartography radar, analytics, calibration wizard, mobile remote control,
    and system tray.
    """

    APP_TITLE = f"Gatherion-Local v{__version__} (Türkçe)"

    def __init__(self, app: QApplication) -> None:
        super().__init__()
        # [STARTUP] Açılış faz kronometresi (faz + toplam süre logları için).
        self._startup_t0 = time.monotonic()
        self._startup_last_t = self._startup_t0
        # S-0204: canlı örnek modül düzeyi readiness probuna bağlanır.
        global _LIVE_ORCHESTRATOR_REF
        _LIVE_ORCHESTRATOR_REF = weakref.ref(self)
        self.app = app
        self.app.setApplicationName(self.APP_TITLE)
        if hasattr(self.app, "setApplicationDisplayName"):
            self.app.setApplicationDisplayName(self.APP_TITLE)
        self.hotkey_bridge = HotkeyBridge()

        # 1. Initialize Configuration & OPSEC Encrypted Vault (Agents 1 & 25)
        # sweep-26 F3: bozuk settings.json sessizce varsayılan olurdu —
        # load_error açılışta hiç okunmuyordu. Dosya varlığı SettingsManager
        # kurulmadan ÖNCE ölçülür (reset_to_defaults dosyayı yazar).
        settings_existed = os.path.exists(SETTINGS_JSON_PATH)
        self.settings_mgr = SettingsManager(SETTINGS_JSON_PATH)
        self.settings = getattr(self.settings_mgr, "settings", None) or self.settings_mgr.load()
        self._settings_load_error = getattr(self.settings_mgr, "load_error", None)
        self._settings_was_missing = not settings_existed
        self.vault_crypto = LocalVaultCrypto()
        self._sync_encrypted_vault()
        self._startup_phase("1. Ayarlar + kasa")

        # 2. Initialize Capture, Vision & Stealth Managers (Agents 3, 14, 22)
        # F-23-07 (v11): monitor_idx ayarı bağlandı — çoklu monitör
        # ayarı önceden ölüydü (ScreenCapture hep varsayılan 1).
        self.capture = ScreenCapture(monitor_idx=int(self.settings.display.monitor_idx))
        self.dxg_capture = DXGIDesktopCapture()
        self.vision = VisionEngine(monitor_idx=int(self.settings.display.monitor_idx))
        self.stealth_overlay_mgr = StealthOverlayManager()
        self.frame_pacer = AdaptiveFramePacer()
        self.hud_reader = GameHUDReader()
        self.preprocessor = AdaptivePreprocessor()
        self._startup_phase("2. Yakalama + görüş")

        # 3. Initialize Input Driver, WindMouse & Biometrics (Agents 2, 21, 23)
        self.input_ctrl = InputController()
        self.input_ctrl.set_test_mode(bool(self.settings.gathering.test_mode))
        self.windmouse = WindMouseEngine()
        self.biometric_clicker = BiometricClickGenerator()
        self.physical_guard = PhysicalInputGuard()
        self.human_timing = HumanTimingEngine()
        self.organic_anti_afk = OrganicAntiAFK()
        self._startup_phase("3. Girdi sürücüsü")

        # 4. Initialize Route Manager, Questlog Importer & Sequential Cartography (Agents 4, 5, 11, 16-19)
        self.route_mgr = RouteManager(base_dir=ROUTES_DIR)
        self.questlog_importer = QuestlogImporter()
        self.route_optimizer = RouteOptimizer()
        self.minimap_scanner = MinimapScanner()
        self.sequential_planner = SequentialRoutePlanner()
        self.material_scanner = MaterialScanner()
        self.explorer_ctrl = ExplorerModeController(
            route_manager=self.route_mgr,
            vision_engine=self.vision,
            input_controller=self.input_ctrl,
            test_mode=False,
        )

        # Ensure default region/route is populated or synced with settings
        # F-1199: rota hazırlanamazsa bot boş rotayla "başlatıldı" sayılmaz.
        self._initial_route_ready: bool = self._ensure_initial_route_ready()
        if not self._initial_route_ready:
            logger.warning(
                "Başlangıç rotası hazır değil — bot rota boşken açılıyor; "
                "rota noktaları eklenmeden rota moduna geçmeyin."
            )
        self._startup_phase("4. Rota + keşif")

        # 5. Initialize Combat, Behavioral Guard, Telemetry, Watchdog & Panic (Agents 6, 12, 15, 24, 25)
        # F-23-07 (v11): fly_unreachable ayarı bağlandı — önceden
        # CombatRecovery hep True alıyordu (legacy shim); artık
        # ayardaki değer (varsayılan True) geçiliyor.
        self.combat_recovery = CombatRecovery(
            fly_unreachable=bool(self.settings.route.fly_unreachable),
            vision=self.vision,
            input_ctrl=self.input_ctrl,
        )
        self.behavioral_guard = BehavioralStealthGuard()
        # v20-istatistik (R2): üretim motoru geçmiş oturum olaylarını yükler —
        # "Bugün" restart sonrası da bugünün daha önceki oturumlarını gösterir.
        self.telemetry = TelemetryEngine(history_path="logs/telemetry_history.json")
        self.panic_system = PanicAlertSystem()
        self.watchdog = BotWatchdog(recovery_callback=self._on_watchdog_recovery)
        self._startup_phase("5. Savaş + watchdog")

        # 5.5 Initialize Enterprise Vision Filters, Patch Normalizer, Economy, Audio/YOLO, Multi-Char & 3D Depth Guards
        self.admin_elevated: bool = bool(is_admin())
        self.dpi_guard = DPIScalingSynchronizer()
        self.focus_monitor = ForegroundFocusGuard()
        self.route_backup_mgr = RouteBackupManager()
        self.template_tuner = SelfHealingTemplateTuner()
        self.feather_filter = FeatherNegativeFilter()
        self.pyramid_crest_detector = Pyramid3DCrestDetector()
        self.dynamic_key_reader = DynamicInteractionKeyReader()
        self.monster_circ_filter = MonsterCircularityFilter()
        self.nameplate_isolator = NameplateHSVIsolator()
        self.minimap_orientation_guard = MinimapOrientationGuard()
        self.patch_lighting_normalizer = SobelClahePatchNormalizer()
        self.roi_cropper = HighResROICropper()
        self.watershed_separator = WatershedClusterSeparator()
        self.motion_blur_guard = MotionBlurGate()
        self.skill_bar_monitor = SkillBarReader()
        self.micro_camera_look = MicroCameraLookAround()
        self.incident_recorder = IncidentRecorder()
        self.mobile_token_auth = MobileTokenAuthenticator()
        self.dry_run_simulator = DryRunReplaySimulator()
        self.broker_scanner = TradeBrokerPriceScanner()
        self.crafting_controller = AutoCraftingMorphQueue()
        self.economy_calculator = SessionProfitSimulator()
        self.yolo_detector = ONNXYolo3DDetector()
        self.audio_analyzer = AudioLoopbackFFTAnalyzer()
        self.alt_char_mgr = AltCharacterManager()
        self.task_scheduler = AutomationTaskScheduler()
        self.pet_buff_mgr = PetAndBuffManager()
        self.depth_avoider = MonocularDepthObstacleAvoidance()
        self.glide_pilot = GlideWaypointController()
        self._startup_phase("5.5. YOLO/3D/audio bileşenleri")

        # 6. Initialize UI Windows (Agents 8, 9, 10, 12, 13, 20)
        self.overlay = OverlayHUD()
        self.overlay.setWindowTitle(f"{self.APP_TITLE} — Oyun İçi Durum Paneli")
        if hasattr(self.overlay, "version_label"):
            self.overlay.version_label.setText(f"v{__version__}")
        self.stealth_overlay_mgr.apply_stealth_window_identity(self.overlay)
        if self.settings.gathering.show_overlay:
            self.overlay.show()
        self.sounds = UiClickSounds()
        self._startup_phase("6. Arayüz (HUD + F9)")

        # sweep-26 F3/F8/F6: açılış koşulları artık yalnızca logda kalmıyor —
        # bozuk/eksik ayar dosyası ve yönetici-yoksun çalışma overlay'de görünür.
        if getattr(self, "_settings_load_error", None):
            logger.warning(
                "Ayar dosyası bozuk; varsayılanlarla açıldı: %s",
                self._settings_load_error,
            )
            self.overlay.add_log(
                "⚠️ Ayar dosyası bozuk — varsayılanlarla açıldı "
                "(yedek: settings.json.corrupt.bak)."
            )
        elif getattr(self, "_settings_was_missing", False):
            logger.warning("settings.json yoktu; varsayılan ayarlarla açıldı.")
            self.overlay.add_log("⚠️ settings.json bulunamadı — varsayılan ayarlarla açıldı.")
        if not getattr(self, "admin_elevated", True):
            logger.warning(
                "Bot yönetici yetkisi olmadan çalışıyor; Interception "
                "(donanım HID) yönetici isteyebilir, tuşlar oyuna gitmeyebilir."
            )
            self.overlay.add_log(
                "⚠️ Yönetici yetkisi yok — tuşlar oyuna gitmezse yöneticiyle başlatın."
            )

        self.f9_menu = F9Menu(route_manager=self.route_mgr)
        if hasattr(self.overlay, "bind_f9_menu"):
            self.overlay.bind_f9_menu(self.f9_menu)
        self.settings_win: Optional[SettingsWindow] = None
        self.map_viewer_win: Optional[MapViewerWindow] = None
        self.radar_win: Optional[CartographyRadarDialog] = None
        self.analytics_win: Optional[AnalyticsDialog] = None
        self.calibration_win: Optional[CalibrationWizard] = None
        self.guide_win: Optional[GuideDialog] = None
        self.mobile_connect_win: Optional[Any] = None
        self.yolo_factory_win: Optional[Any] = None

        # 7. Initialize BotWorker State Machine (Agent 7)
        self.worker = BotWorker(
            config=self.settings_mgr,
            vision=self.vision,
            input_driver=self.input_ctrl,
            route_manager=self.route_mgr,
            combat_recovery=self.combat_recovery,
            overlay_rect_provider=self.overlay.get_screen_rect,
            material_scanner=self.material_scanner,
            minimap_scanner=self.minimap_scanner,
            yolo_detector=self.yolo_detector,
        )
        self.worker.telemetry = self.telemetry
        self.worker.watchdog = self.watchdog
        self.watchdog.attach_worker(self.worker)
        if hasattr(self.overlay, "bind_live_sources"):
            self.overlay.bind_live_sources(route_manager=self.route_mgr, worker=self.worker)
        self.worker.sequential_planner = self.sequential_planner
        self.worker.explorer_controller = self.explorer_ctrl
        # Wire YZ (reflex/YOLO) into Oto-Keşif / Auto Mapping from boot.
        if hasattr(self.explorer_ctrl, "bind_ai_sources"):
            try:
                self.explorer_ctrl.bind_ai_sources(
                    yolo_detector=getattr(self, "yolo_detector", None),
                )
            except Exception as exc:
                # F-1196: bağlama hatası çıplak pass ile yutulurdu.
                logger.warning("Explorer AI kaynak bağlama hatası: %s", exc)
        self.worker.human_timing = self.human_timing
        self.worker.organic_anti_afk = self.organic_anti_afk
        self.worker.frame_pacer = self.frame_pacer
        self.worker.windmouse = self.windmouse
        self.worker.biometric_clicker = self.biometric_clicker
        self.worker.behavioral_guard = self.behavioral_guard
        self.worker.panic_system = self.panic_system
        # F-0163 [O1]: watchdog daemon'ını üretimde başlat — PanicAlertSystem
        # tek başına poll etmez; zaten çalışıyorsa False döner.
        try:
            _panic = getattr(self, "panic_system", None)
            if _panic is not None and hasattr(_panic, "start_watchdog_polling"):
                # sweep-21 B-01: detektörler poll başlamadan ÖNCE bağlanır —
                # yoksa poll 6/6 detektörsüz (kör) çalışır. Bind docstring'in
                # istediği tek satırdır.
                if hasattr(_panic, "bind_production_watchdog_sources"):
                    try:
                        _wiring = _panic.bind_production_watchdog_sources(worker=self.worker)
                        logger.info("Watchdog üretim kaynakları bağlandı: %s", _wiring)
                    except Exception as _bind_exc:
                        logger.warning("Watchdog üretim kaynağı bağlanamadı: %s", _bind_exc)
                _poll_started = _panic.start_watchdog_polling(input_ctrl=self.input_ctrl)
                if not _poll_started:
                    # sweep-21 B-12: başlamama artık yalnızca debug'da değil.
                    logger.warning("Watchdog polling başlatılamadı (zaten çalışıyor olabilir).")
        except Exception as _wd_exc:
            logger.warning("Watchdog polling başlatılamadı: %s", _wd_exc)
        self.worker.dpi_guard = self.dpi_guard
        self.worker.focus_monitor = self.focus_monitor
        if hasattr(self.input_ctrl, "focus_monitor"):
            self.input_ctrl.focus_monitor = self.focus_monitor
        # F-0083: focus guard'ı input sürücüsüne bağla — oyun odakta
        # değilken tuş gönderilmez (press_key 0.0 döner). Guard
        # çalışamazsa (Linux) sarmalayıcı tuşu geçirir, kilitlemez.
        try:
            self.focus_monitor.attach_to_input_driver(self.input_ctrl)
        except Exception as _fg_exc:
            logger.warning("Focus guard input sürücüsüne bağlanamadı: %s", _fg_exc)
        self.worker.route_backup_mgr = self.route_backup_mgr
        self.worker.template_tuner = self.template_tuner
        self.worker.feather_filter = self.feather_filter
        self.worker.pyramid_crest_detector = self.pyramid_crest_detector
        self.worker.dynamic_key_reader = self.dynamic_key_reader
        self.worker.monster_circ_filter = self.monster_circ_filter
        self.worker.nameplate_isolator = self.nameplate_isolator
        self.worker.minimap_orientation_guard = self.minimap_orientation_guard
        self.worker.patch_lighting_normalizer = self.patch_lighting_normalizer
        self.worker.roi_cropper = self.roi_cropper
        self.worker.watershed_separator = self.watershed_separator
        self.worker.motion_blur_guard = self.motion_blur_guard
        self.worker.skill_bar_monitor = self.skill_bar_monitor
        self.worker.micro_camera_look = self.micro_camera_look
        self.worker.incident_recorder = self.incident_recorder
        self.worker.mobile_token_auth = self.mobile_token_auth
        self.worker.dry_run_simulator = self.dry_run_simulator
        self.worker.broker_scanner = self.broker_scanner
        self.worker.crafting_controller = self.crafting_controller
        self.worker.economy_calculator = self.economy_calculator
        # v20-onnx-reload: tek paylaşımlı detektör — hub/reflex/scanner
        # BotWorker kurulumunda bağlandı; adopt hem yeniden bağlar hem de
        # (eski yol) üzerine yazılmış orphan instance varsa close eder.
        if hasattr(self.worker, "adopt_shared_yolo_detector"):
            self.worker.adopt_shared_yolo_detector(self.yolo_detector)
        else:
            self.worker.yolo_detector = self.yolo_detector
        for _ms in (
            getattr(self.worker, "material_scanner", None),
            getattr(self, "material_scanner", None),
        ):
            try:
                if _ms is not None and hasattr(_ms, "set_yolo_detector"):
                    _ms.set_yolo_detector(self.yolo_detector)
            except Exception as _ms_exc:
                logger.debug("material scanner detector wiring skipped: %s", _ms_exc)
        self.worker.audio_analyzer = self.audio_analyzer
        self.worker.alt_char_mgr = self.alt_char_mgr
        self.worker.task_scheduler = self.task_scheduler
        self.worker.pet_buff_mgr = self.pet_buff_mgr
        # F-0143: alt-karakter/rotasyon/buff yöneticilerinin üretim
        # bağlamını bağla — sonrasında her toplamada zaten çalışan
        # record_gather zinciri takvim/rotasyon/buff denetimini tetikler.
        try:
            self.alt_char_mgr.bind_runtime_context(
                input_ctrl=self.input_ctrl,
                route_mgr=self.route_mgr,
                task_scheduler=self.task_scheduler,
                pet_buff_mgr=self.pet_buff_mgr,
                frame_provider=lambda: self.vision.capture_screen(),
            )
        except Exception as _mc_exc:
            logger.warning("Alt-karakter üretim bağlamı bağlanamadı: %s", _mc_exc)
        self.worker.depth_avoider = self.depth_avoider
        self.worker.glide_pilot = self.glide_pilot

        # Unified subsystem aliases
        self.bot_worker = self.worker
        self.routes = self.route_mgr
        self.overlay_hud = self.overlay
        self._startup_phase("7. BotWorker")

        # 8. Connect Signals & Hotkeys
        self._connect_signals()
        self._setup_hotkeys()
        self._setup_system_tray()
        self._startup_phase("8. Sinyaller + kısayol tuşları")
        self.shell: Optional[GatherionShell] = None

        # 9. Initialize Mobile Remote Control Server & Cloud Tunnel (v2.0 Mobile Suite)
        self.tunnel_mgr: Optional[Any] = None
        self.mobile_server: Optional[Any] = None
        try:
            from core.mobile_tunnel import FreeCloudTunnelManager

            self.tunnel_mgr = FreeCloudTunnelManager()
        except Exception as tun_exc:
            # sweep-26 F5: tünel yöneticisi kurulamazsa mobil tünel hiç
            # olmayacaktı ama bu yalnızca debug'da kalıyordu.
            logger.warning(
                "FreeCloudTunnelManager başlatılamadı; mobil tünel kurulmayacak: %s",
                tun_exc,
            )

        try:
            from core.mobile_server import MobileRemoteServer

            self.mobile_server = MobileRemoteServer(orchestrator=self, port=8765)
            # F-0105/F-1889 [O1]: 256-bit kimlik doğrulayıcıyı canlı sunucuya
            # bağla — sunucunun kendi kısa token'ı yerine
            # MobileTokenAuthenticator token'ı zorunlu olur.
            try:
                _token_auth = getattr(self, "mobile_token_auth", None)
                if _token_auth is not None:
                    _token_auth.bind_to_mobile_server(self.mobile_server)
                    # Y-3 (dilim 22): bind sonrası diskteki
                    # config/mobile_token.txt artık geçersizdir (sunucunun
                    # kendi kısa token'ını tutar). Canlı 256-bit token
                    # dosyaya yazılır ki o dosyadaki tokenla giriş 401
                    # vermesin.
                    try:
                        if hasattr(self.mobile_server, "sync_token_file"):
                            self.mobile_server.sync_token_file()
                    except Exception as _sync_exc:
                        logger.debug("Mobil token dosya senkronu atlandı: %s", _sync_exc)
            except Exception as _bind_exc:
                logger.debug("Mobil token bağlama atlandı: %s", _bind_exc)
            self.mobile_server.start()
        except Exception as mob_exc:
            # F-1140: başlatma hatası yalnızca debug loga gömülürdü.
            self.mobile_server = None
            logger.warning("MobileRemoteServer başlatılamadı: %s", mob_exc)
            try:
                self.overlay.add_log(f"⚠️ Mobil sunucu başlatılamadı: {mob_exc}")
            except Exception as exc:
                logger.debug("Mobil sunucu hatası overlay'e yazılamadı: %s", exc)

        # F-0140 [O1]: imzalı ntfy komutlarını canlı sunucuya bağla.
        if self.tunnel_mgr is not None:
            try:
                self.tunnel_mgr.set_remote_command_handler(self._on_remote_command)
            except Exception as _rc_exc:
                logger.debug("Uzaktan komut bağlama atlandı: %s", _rc_exc)

        # Otomatik Ücretsiz Cloudflare Tünelini arka planda başlat
        if self.tunnel_mgr is not None:
            def _on_mobile_tunnel_ready(public_url: str, method: str) -> None:
                # Y-2 (dilim 22): 8765 doluysa sunucu 8766-8775'e geçer;
                # tünel + LAN URL'si sunucunun GERÇEK portunu kullanır.
                srv_port = 8765
                try:
                    if self.mobile_server is not None:
                        srv_port = int(getattr(self.mobile_server, "port", 8765) or 8765)
                except Exception:
                    srv_port = 8765
                # P2-25 (bölüm 8, F079-K1): token'lı URL hiçbir şekilde
                # print edilmez — pythonw altında stdout logs/gatherion.log'a
                # bağlanır, yani print == loga yazmak demektir. v16'da URL
                # zaten "<redacted>" ile basılıyordu; baskı yine de tamamen
                # kaldırıldı (token bu kapsama hiç okunmuyor): hazır bilgisi
                # token'suz logger + overlay üzerinden verilir; gerçek
                # bağlantı Mobil Bağlan penceresindeki QR kodla kurulur.
                lan_ip = self._get_lan_ip()
                logger.info(
                    "Mobil Tünel Aktif: %s (Wi-Fi: http://%s:%d)",
                    public_url,
                    lan_ip,
                    srv_port,
                )
                if hasattr(self, "overlay") and self.overlay is not None:
                    self.overlay.add_log(f"📱 Mobil Link: {public_url}")
                try:
                    url_file = os.path.join(BASE_DIR, "config", "mobile_url.txt")
                    with open(url_file, "w", encoding="utf-8") as f:
                        f.write(f"PUBLIC_URL={public_url}\nLAN_URL=http://{lan_ip}:{srv_port}\n")
                except Exception as exc:
                    logger.debug("mobile_url.txt yazılamadı: %s", exc)

            self.tunnel_mgr.on_url_ready = _on_mobile_tunnel_ready
            # F-1198: cloudflared ikilisi zaten varsa indirme istenmez;
            # yalnızca eksikse otomatik indirilir.
            _cloudflared_missing = True
            try:
                _find_binary = getattr(self.tunnel_mgr, "find_cloudflared_binary", None)
                if callable(_find_binary):
                    _cloudflared_missing = _find_binary() is None
            except Exception:
                _cloudflared_missing = True
            if _cloudflared_missing:
                logger.info("cloudflared bulunamadı; otomatik indirme yapılacak.")
            # Y-2 (dilim 22): tünel sunucunun gerçek portuna açılır.
            _tunnel_port = 8765
            try:
                if self.mobile_server is not None:
                    _tunnel_port = int(getattr(self.mobile_server, "port", 8765) or 8765)
            except Exception:
                _tunnel_port = 8765
            threading.Thread(
                target=lambda: self._start_tunnel_in_background(_tunnel_port, _cloudflared_missing),
                name="AutoCloudflareStarter",
                daemon=True,
            ).start()

            # Y-6 (dilim 22): tünel koptuğunda bot kör devam etmesin.
            # İzleyici 30 sn'de bir kontrol eder; süreç ölmüşse bayat
            # URL temizlenir (check_tunnel_alive) ve tünel yeniden
            # başlatılır (yeniden deneme en fazla 120 sn'de bir).
            def _monitor_mobile_tunnel() -> None:
                last_restart_ts = 0.0
                while True:
                    try:
                        time.sleep(30.0)
                    except Exception:
                        return
                    try:
                        mgr = getattr(self, "tunnel_mgr", None)
                        if mgr is None:
                            return
                        check = getattr(mgr, "check_tunnel_alive", None)
                        if not callable(check):
                            return
                        if check():
                            continue
                        logger.warning(
                            "Mobil tünel koptu (canlı değil); bayat URL temizlendi, yeniden başlatma denenecek."
                        )
                        try:
                            lan_ip = self._get_lan_ip()
                        except Exception:
                            lan_ip = ""
                        try:
                            url_file = os.path.join(BASE_DIR, "config", "mobile_url.txt")
                            with open(url_file, "w", encoding="utf-8") as f:
                                f.write(f"LAN_URL=http://{lan_ip}:{_tunnel_port}\nTUNNEL_DOWN=1\n")
                        except Exception as exc:
                            logger.debug("mobile_url.txt bayat URL temizliği yazılamadı: %s", exc)
                        now = time.time()
                        if now - last_restart_ts < 120.0:
                            continue
                        last_restart_ts = now
                        try:
                            mgr.start_tunnel(port=_tunnel_port, auto_download_cloudflared=False)
                        except Exception as exc:
                            logger.warning("Mobil tünel yeniden başlatılamadı: %s", exc)
                    except Exception as exc:
                        logger.debug("Mobil tünel izleyici uyarısı: %s", exc)

            threading.Thread(
                target=_monitor_mobile_tunnel,
                name="MobileTunnelMonitor",
                daemon=True,
            ).start()
        self._startup_phase("9. Mobil sunucu + tünel")

        self._install_shell()

        # Deduplicate any loggers created by subsystems during initialization
        deduplicate_loggers()

        # 10. Start Watchdog & Worker Thread
        self.watchdog.start()
        self.worker.start()

        logger.info(
            f"Gatherion-Local v{__version__} (Türkçe) başlatıldı (Bölge: %s | Rota: %s)",
            self.route_mgr.active_region,
            self.route_mgr.active_route,
        )
        self.overlay.add_log(
            f"Gatherion v{__version__} hazır "
            f"({getattr(self, '_pause_chord_label', 'Ctrl+Alt+F10')}: Durdur/Başlat, "
            f"{getattr(self, '_menu_chord_label', 'Ctrl+Alt+F9')}: Menü, "
            f"{getattr(self, '_node_chord_label', 'Ctrl+Alt+F8')}: Nokta Kaydet)"
        )
        # sweep-26 F7: boş rotayla "hazır" deniyordu; bayrak yalnızca
        # logda kalıyordu. Rota yoksa overlay'de de görünür.
        if not self._initial_route_ready:
            self.overlay.add_log(
                "⚠️ Başlangıç rotası hazır değil — rota noktası eklenmeden "
                "rota moduna geçmeyin."
            )

        # [STARTUP] Açılış tamam + arayüz hazır logu.
        _startup_total = time.monotonic() - self._startup_t0
        logger.info("[STARTUP] Açılış tamam: %.1f sn", _startup_total)
        QTimer.singleShot(
            0,
            lambda: logger.info(
                "[STARTUP] Arayüz hazır (UI responsive) — toplam %.1f sn",
                time.monotonic() - self._startup_t0,
            ),
        )

        # [UI-WATCHDOG] UI thread donma dedektörü: timer UI thread'de
        # çalışır; donma sırasında tetiklenmez, donma bitince bir kez
        # büyük farkla loglar — bu istenen davranış.
        self._ui_watchdog_last_tick = time.monotonic()
        self._ui_watchdog_timer = QTimer(self)
        self._ui_watchdog_timer.setInterval(500)
        self._ui_watchdog_timer.timeout.connect(self._on_ui_watchdog_tick)
        self._ui_watchdog_timer.start()

    def _startup_phase(self, label: str) -> None:
        """[STARTUP] faz süresi + toplam açılış süresi logu."""
        now = time.monotonic()
        phase = now - self._startup_last_t
        total = now - self._startup_t0
        self._startup_last_t = now
        logger.info("[STARTUP] %s: %.1f sn (toplam %.1f sn)", label, phase, total)

    def _on_ui_watchdog_tick(self) -> None:
        """500 ms'lik UI timer tick'i; 2 sn'yi aşan gecikme donma şüphesidir."""
        now = time.monotonic()
        gap = now - self._ui_watchdog_last_tick
        self._ui_watchdog_last_tick = now
        if gap > 2.0:
            logger.warning(
                "[UI-WATCHDOG] UI thread %.1f sn yanıt vermedi (donma şüphesi)",
                gap,
            )

    # =========================================================================
    # Initialization & Sync Helpers
    # =========================================================================

    def _sync_encrypted_vault(self) -> None:
        """Source of truth: config/settings.json. settings.enc is an encrypted mirror and is never loaded over that file."""
        try:
            if not getattr(self.settings_mgr, "_allow_overwrite", True):
                return
            enc_path = os.path.join(BASE_DIR, "config", "settings.enc")
            if hasattr(self.settings_mgr, "export_payload"):
                payload = self.settings_mgr.export_payload()
            else:
                payload = self.settings.to_dict()
            self.vault_crypto.save_encrypted_config(payload, enc_path)
        except Exception as exc:
            logger.debug("Şifreli kasa senkronizasyonu atlandı: %s", exc)

    def _persist_settings_on_exit(self) -> None:
        shell = getattr(self, "shell", None)
        if shell is not None and hasattr(shell, "persist_settings"):
            shell.persist_settings()
            return
        win = getattr(self, "settings_win", None)
        if win is not None and hasattr(win, "save_settings"):
            win.save_settings()
            return
        # sweep-26 F10: shell de ayar penceresi de yoksa kayıt sessizce
        # atlanırdı — oturumdaki ayar değişikliği kaybolurdu.
        logger.warning(
            "Çıkışta ayar kaydı atlandı: ana pencere ve ayar penceresi yok."
        )

    def _overlay_log_safe(self, message: str) -> None:
        """Overlay'e log satırı; overlay yoksa/hatalıysa debug'da kalır."""
        try:
            overlay = getattr(self, "overlay", None)
            if overlay is not None:
                overlay.add_log(message)
        except Exception as exc:
            logger.debug("Overlay log yazılamadı: %s", exc)

    def _start_tunnel_in_background(self, port: int, auto_download_cloudflared: bool) -> None:
        """Tünel kurulumu arka plan thread'inde. sweep-26 F5: start_tunnel
        dönüş değeri atılıyordu — cloudflared/tünel hatası hiç görünmüyordu.
        Artık başarısızlık warning + overlay'de görünür.
        port: mobil sunucunun gerçek portu (I Y-2 — 8765 sabitlenmez)."""
        mgr = getattr(self, "tunnel_mgr", None)
        if mgr is None:
            return
        try:
            status = mgr.start_tunnel(
                port=port, auto_download_cloudflared=auto_download_cloudflared
            )
        except Exception as exc:
            logger.warning("Mobil tünel başlatılamadı: %s", exc)
            self._overlay_log_safe(f"⚠️ Mobil tünel kurulamadı: {exc}")
            return
        started = False
        error = ""
        if isinstance(status, dict):
            started = bool(status.get("tunnel_started"))
            error = str(status.get("tunnel_error") or "")
        else:
            started = bool(status)
        if not started:
            logger.warning("Mobil tünel kurulamadı: %s", error or "bilinmeyen hata")
            self._overlay_log_safe(
                f"⚠️ Mobil tünel kurulamadı: {error or 'bilinmeyen hata'}"
            )

    def _get_lan_ip(self) -> str:
        """F-1122: tünel-hazır geri çağrısının çağırdığı metot (önceden tanımsızdı)."""
        return get_local_lan_ip()

    def _ensure_initial_route_ready(self) -> bool:
        """
        Ensures the active region/route from settings exists in RouteManager.
        If Altgard · Dredgion Crash Site has no nodes yet, pre-populates or syncs cleanly.

        Returns True only when the active route has at least one node
        afterwards; False when route preparation failed or the route is
        still empty (F-1199: bot boş rotayla "başlatıldı" sayılmaz).
        """
        try:
            target_reg = self.settings.active_region or "Altgard"
            target_rt = self.settings.active_route or "Dredgion Crash Site"
            self.route_mgr.active_region = target_reg
            self.route_mgr.active_route = target_rt

            nodes = self.route_mgr.get_route_nodes(target_reg, target_rt)
            if not nodes and target_reg == "Altgard" and target_rt == "Dredgion Crash Site":
                logger.info("Altgard · Dredgion Crash Site rota noktaları Questlog'dan hazırlanıyor")
                self.questlog_importer.import_sub_zones(
                    zone="Altgard",
                    selected_resources=["Orichalcum Ore", "Odyle"],
                    selected_sub_zones=["Dredgion Crash Site"],
                    route_manager=self.route_mgr,
                    vision_engine=self.vision,
                    fast_mode=True,
                )
                self.sequential_planner.apply_sequential_order_to_route(
                    self.route_mgr, region="Altgard", route="Dredgion Crash Site", mode="smooth_loop"
                )
                nodes = self.route_mgr.get_route_nodes(target_reg, target_rt)
            if not nodes:
                logger.warning(
                    "Başlangıç rotası hazır değil: %s · %s için rota noktası yok; "
                    "bot rota boşken başlatılıyor.",
                    target_reg,
                    target_rt,
                )
                return False
            return True
        except Exception as exc:
            logger.error("Başlangıç rotası hazırlanamadı: %s", exc)
            return False

    def _connect_signals(self) -> None:
        """Wires signals between BotWorker, OverlayHUD, F9Menu, and HotkeyBridge."""
        self.worker.hud_update.connect(self._on_worker_hud_update)
        self.worker.log_added.connect(self.overlay.add_log)
        self.worker.node_recorded.connect(self._on_node_recorded)

        # Wire BotWorker 3x harvest stage updates if present
        if hasattr(self.overlay, "update_harvest_stage"):
            for sig_name in (
                "harvest_stage_changed",
                "harvest_stage_updated",
                "harvest_progress",
            ):
                if hasattr(self.worker, sig_name):
                    sig_obj = getattr(self.worker, sig_name)
                    if hasattr(sig_obj, "connect"):
                        sig_obj.connect(self.overlay.update_harvest_stage)

        # Wire BotWorker minimap mapped count signal if present
        if hasattr(self.overlay, "update_minimap_mapped"):
            for sig_name in ("minimap_mapped_changed", "minimap_mapped_updated"):
                if hasattr(self.worker, sig_name):
                    sig_obj = getattr(self.worker, sig_name)
                    if hasattr(sig_obj, "connect"):
                        sig_obj.connect(self.overlay.update_minimap_mapped)

        self.overlay.f9_clicked.connect(self.on_f9_hotkey)
        self.overlay.f10_clicked.connect(self.on_f10_hotkey)
        self.overlay.f8_clicked.connect(self._on_f8_record_node)
        self.overlay.quit_requested.connect(self.shutdown)

        self.f9_menu.action_triggered.connect(self._on_menu_action)
        self.f9_menu.panel_open_changed.connect(self._on_f9_panel_open_changed)
        self.f9_menu.route_selected.connect(self.handle_route_selected)
        if hasattr(self.f9_menu, "minimap_auto_record_toggled"):
            self.f9_menu.minimap_auto_record_toggled.connect(
                self._on_minimap_auto_record_toggled
            )
        if hasattr(self.f9_menu, "selective_gather_changed"):
            self.f9_menu.selective_gather_changed.connect(
                self.handle_selective_gather_changed
            )

        self.hotkey_bridge.f9_pressed.connect(self.on_f9_hotkey)
        self.hotkey_bridge.f9_resume_pressed.connect(self.on_f9_enter_hotkey)
        self.hotkey_bridge.f8_pressed.connect(self._on_f8_record_node)
        self.hotkey_bridge.f10_pressed.connect(self.on_f10_hotkey)
        self.hotkey_bridge.user_key_interrupted.connect(self.worker.on_user_key_pressed)

        # F-0569 [O1]: F9 menüsünü worker/overlay/route_manager ile bağla
        # (shell bu noktada henüz None — _install_shell'da yeniden bağlanır).
        try:
            self.f9_menu.bind_targets(
                shell=getattr(self, "shell", None),
                overlay=self.overlay,
                worker=self.worker,
                route_manager=self.route_mgr,
            )
        except Exception as _bind_exc:
            logger.debug("F9Menu bind_targets atlandı: %s", _bind_exc)

    def _on_remote_command(self, cmd: str, record: Dict[str, Any]) -> None:
        """F-0140 [O1]: imzalı ntfy uzaktan komutunu canlı HTTP sunucusunun
        execute_action dispatcher'ına yönlendir. ntfy komut adları sunucunun
        eylem adlarına çevrilir (f8->record_f8, skip->skip_node,
        panic->panic_stop); çevrilemeyenler execute_action'da dürüstçe
        reddedilir (ok:False)."""
        server = getattr(self, "mobile_server", None)
        if server is None or not hasattr(server, "execute_action"):
            logger.debug("Uzaktan komut yoksayıldı (mobil sunucu yok): %s", cmd)
            return
        action_map = {"f8": "record_f8", "skip": "skip_node", "panic": "panic_stop"}
        payload = dict((record or {}).get("meta") or {})
        payload["action"] = action_map.get(str(cmd).strip().lower(), str(cmd).strip().lower())
        try:
            server.execute_action(payload)
        except Exception as exc:
            logger.warning("Uzaktan komut çalıştırılamadı (%s): %s", cmd, exc)

    @pyqtSlot(dict)
    def _on_worker_hud_update(self, data: dict) -> None:
        """Forwards HUD snapshots to OverlayHUD and syncs 3x harvest stage if included."""
        try:
            self.overlay.update_state(data)
            if hasattr(self.overlay, "update_harvest_stage") and isinstance(data, dict):
                stage = data.get("harvest_stage", data.get("current_harvest_stage"))
                max_stages = data.get("max_harvest_stages", data.get("harvests_per_node", 3))
                if stage is not None:
                    self.overlay.update_harvest_stage(int(stage), int(max_stages or 3))
        except Exception as exc:
            logger.debug("HUD güncelleme yönlendirme hatası: %s", exc)

    @pyqtSlot(bool)
    def _on_minimap_auto_record_toggled(self, enabled: bool) -> None:
        """Handles toggling top-right Minimap Auto-Record from F9Menu."""
        try:
            is_on = bool(enabled)
            self.worker.auto_cartography_enabled = is_on
            if hasattr(self.worker, "set_minimap_auto_record"):
                self.worker.set_minimap_auto_record(is_on)
            elif hasattr(self.worker, "minimap_auto_record_enabled"):
                setattr(self.worker, "minimap_auto_record_enabled", is_on)

            if hasattr(self.settings, "gathering") and hasattr(
                self.settings.gathering, "minimap_auto_record"
            ):
                self.settings.gathering.minimap_auto_record = is_on
                self.settings_mgr.save(self.settings)
                self._sync_encrypted_vault()

            state_tr = "AÇIK" if is_on else "KAPALI"
            logger.info("Sağ Üst Minimap Oto-Kayıt durumu güncellendi: %s", state_tr)
            self.overlay.add_log(f"Minimap Oto-Kayıt: {state_tr}")
            self.worker._emit_hud_snapshot()
        except Exception as exc:
            logger.error("Minimap Oto-Kayıt değiştirilirken hata: %s", exc)

    def _hotkey_specs(self):
        hk = getattr(self.settings, "hotkeys", None)
        node = getattr(hk, "record_node_f8", "ctrl+alt+f8")
        menu = getattr(hk, "toggle_menu_f9", "ctrl+alt+f9")
        pause = getattr(hk, "pause_resume_f10", "ctrl+alt+f10")
        return str(node), str(menu), str(pause)

    def _apply_hotkey_labels(self) -> None:
        from config.schema import format_hotkey_label

        node, menu, pause = self._hotkey_specs()
        self._node_chord_label = format_hotkey_label(node)
        self._menu_chord_label = format_hotkey_label(menu)
        self._pause_chord_label = format_hotkey_label(pause)
        listener = getattr(self.input_ctrl, "hotkey_listener", None)
        if listener is not None and hasattr(listener, "set_chords"):
            listener.set_chords(node, menu, pause)
        if hasattr(self, "overlay") and hasattr(self.overlay, "set_hotkey_labels"):
            self.overlay.set_hotkey_labels(
                self._node_chord_label, self._menu_chord_label, self._pause_chord_label
            )
        if hasattr(self, "f9_menu") and hasattr(self.f9_menu, "apply_hotkey_labels"):
            self.f9_menu.apply_hotkey_labels(
                self._menu_chord_label, self._node_chord_label, self._pause_chord_label
            )
        shell = getattr(self, "shell", None)
        if shell is not None and hasattr(shell, "set_hotkey_line"):
            shell.set_hotkey_line(
                self._node_chord_label,
                self._menu_chord_label,
                self._pause_chord_label,
            )

    def _register_native_chords(self) -> None:
        node, menu, pause = self._hotkey_specs()
        if getattr(self, "native_hotkeys", None) is None:
            self.native_hotkeys = NativeWin32HotkeyManager()
        results = self.native_hotkeys.register_default_f8_f9(
            on_f8=lambda: self.hotkey_bridge.f8_pressed.emit(),
            on_f9=lambda: self.hotkey_bridge.f9_pressed.emit(),
            on_f10=lambda: self.hotkey_bridge.f10_pressed.emit(),
            node_chord=node,
            menu_chord=menu,
            pause_chord=pause,
        )
        claimed = set()
        if results.get("F8"):
            claimed.add("f8")
        if results.get("F9"):
            claimed.add("f9")
        if results.get("F10"):
            claimed.add("f10")
        # F-1200: üç akor da kaydedilemezse hiçbir uyarı yoktu.
        if not claimed:
            logger.warning(
                "Yerel kısayol akorlarının hiçbiri kaydedilemedi "
                "(F8/F9/F10: %s); kısayollar yalnızca klavye kancasıyla çalışır.",
                results,
            )
            try:
                self.overlay.add_log("⚠️ Yerel kısayollar kaydedilemedi (F8/F9/F10).")
            except Exception as exc:
                logger.debug("Kısayol hatası overlay'e yazılamadı: %s", exc)
        elif len(claimed) < 3:
            missing = {"f8", "f9", "f10"} - claimed
            logger.warning("Yerel kısayol akorları kısmen kaydedildi; eksik: %s", sorted(missing))
        listener = getattr(self.input_ctrl, "hotkey_listener", None)
        if listener is not None and hasattr(listener, "set_hook_actions"):
            listener.set_hook_actions({"f8", "f9", "f10"} - claimed)

    def _setup_hotkeys(self) -> None:
        """Register Ctrl+Alt chords. Bare F8/F9/F10 stay with the game and Windows."""
        self._apply_hotkey_labels()
        try:
            registered = self.input_ctrl.register_hotkeys(
                on_f9=lambda: self.hotkey_bridge.f9_pressed.emit(),
                on_f9_enter=lambda: self.hotkey_bridge.f9_resume_pressed.emit(),
                on_f8=lambda: self.hotkey_bridge.f8_pressed.emit(),
                on_f10=lambda: self.hotkey_bridge.f10_pressed.emit(),
                on_user_key_interrupt=lambda k: self.hotkey_bridge.user_key_interrupted.emit(str(k)),
            )
            if not registered:
                raise RuntimeError("keyboard hook unavailable")
        except Exception as exc:
            logger.warning("Klavye kancası kurulamadı, yalnız RegisterHotKey akorları denenecek: %s", exc)
        try:
            self._register_native_chords()
        except Exception as nat_exc:
            logger.error("Yerel kısayol akorları kaydedilemedi: %s", nat_exc)

    def _setup_system_tray(self) -> None:
        """Creates the System Tray icon and Turkish quick-access context menu."""
        try:
            self.tray = QSystemTrayIcon(create_tray_icon_pixmap(), self.app)
            self.tray.setToolTip(self.APP_TITLE)

            tray_menu = QMenu()
            tray_menu.setStyleSheet("""
                QMenu {
                    background-color: #181b20;
                    color: #e5e7eb;
                    border: 1px solid #2d323b;
                    border-radius: 6px;
                    padding: 4px;
                    font-family: 'Segoe UI', sans-serif;
                    font-size: 13px;
                }
                QMenu::item {
                    padding: 8px 22px 8px 14px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #2563eb;
                    color: #ffffff;
                }
                QMenu::separator {
                    height: 1px;
                    background: #2d323b;
                    margin: 6px 8px;
                }
            """)
            act_resume = QAction(
                f"Devam Et / Duraklat ({getattr(self, '_pause_chord_label', 'Ctrl+Alt+F10')})",
                self.app,
            )
            act_resume.triggered.connect(self.worker.toggle_pause)
            tray_menu.addAction(act_resume)

            act_explorer = QAction("Oto-Keşif & Haritalama Modu (Gezdikçe Haritalandır)", self.app)
            act_explorer.triggered.connect(lambda: self.handle_menu_action("explorer"))
            tray_menu.addAction(act_explorer)

            act_route = QAction("Rota Modu (Sıralı Rota Gezgini)", self.app)
            act_route.triggered.connect(lambda: self.handle_menu_action("route"))
            tray_menu.addAction(act_route)

            tray_menu.addSeparator()

            act_mobile = QAction("Mobil Uzaktan Kontrol & QR (Mobile Dashboard)", self.app)
            act_mobile.triggered.connect(self.open_mobile_connect_dialog)
            tray_menu.addAction(act_mobile)

            act_radar = QAction("Canlı Haritalama Radarı & Sıralı Rota", self.app)
            act_radar.triggered.connect(self.open_cartography_radar)
            tray_menu.addAction(act_radar)

            act_settings = QAction("Ayarlar & Noktalar", self.app)
            act_settings.triggered.connect(self.open_settings_window)
            tray_menu.addAction(act_settings)

            act_map = QAction("Rota ve Harita Görüntüleyici", self.app)
            act_map.triggered.connect(self.open_map_viewer)
            tray_menu.addAction(act_map)

            act_analytics = QAction("Canlı Telemetri & Analiz", self.app)
            act_analytics.triggered.connect(self.open_analytics_dialog)
            tray_menu.addAction(act_analytics)

            act_calibrate = QAction("Görüntü Kalibrasyon Sihirbazı", self.app)
            act_calibrate.triggered.connect(self.open_calibration_wizard)
            tray_menu.addAction(act_calibrate)

            act_guide = QAction("Kullanım Kılavuzu & Tanılama", self.app)
            act_guide.triggered.connect(self.open_guide_dialog)
            tray_menu.addAction(act_guide)

            tray_menu.addSeparator()

            act_exit = QAction("Gatherion'dan Çık (Exit)", self.app)
            act_exit.triggered.connect(self.shutdown)
            tray_menu.addAction(act_exit)

            self.tray.setContextMenu(tray_menu)
            self.tray.show()
        except Exception as exc:
            logger.warning("Sistem tepsisi başlatma uyarısı: %s", exc)

    # =========================================================================
    # Hotkey & Menu Action Handlers
    # =========================================================================

    @pyqtSlot(bool)
    def _on_f9_panel_open_changed(self, is_open: bool) -> None:
        listener = getattr(self.input_ctrl, "hotkey_listener", None)
        if listener is not None and hasattr(listener, "set_f9_menu_open"):
            listener.set_f9_menu_open(bool(is_open))

    def _install_shell(self) -> None:
        """One window for settings and tools. Overlay HUD stays its own window."""
        try:
            self.shell = GatherionShell(
                settings_mgr=self.settings_mgr,
                route_mgr=self.route_mgr,
                f9_menu=self.f9_menu,
                telemetry=self.telemetry,
                explorer=self.explorer_ctrl,
                mobile_server=self.mobile_server,
                tunnel_mgr=self.tunnel_mgr,
                factory=getattr(self.worker, "dataset_factory", None),
                log_path=LOG_FILE,
                window_title=self.APP_TITLE,
                worker=self.worker,
            )
            if hasattr(self.shell, "set_shutdown"):
                self.shell.set_shutdown(self.shutdown)
            if hasattr(self.shell, "set_chord_handlers"):
                self.shell.set_chord_handlers(
                    record_node=self._on_f8_record_node,
                    toggle_menu=self.on_f9_hotkey,
                    pause_resume=self.on_f10_hotkey,
                )
            if hasattr(self.shell, "set_hotkey_line"):
                self.shell.set_hotkey_line(
                    getattr(self, "_node_chord_label", "Ctrl+Alt+F8"),
                    getattr(self, "_menu_chord_label", "Ctrl+Alt+F9"),
                    getattr(self, "_pause_chord_label", "Ctrl+Alt+F10"),
                )
            # Safely hook page creation (supports page_created signal, lazy mounting, or pre-mounted pages)
            if hasattr(self.shell, "page_created"):
                try:
                    self.shell.page_created.connect(self._on_shell_page_created)
                    logger.info("GatherionShell.page_created sinyali bağlandı.")
                except Exception as exc:
                    logger.debug("shell.page_created bağlantı uyarısı: %s", exc)

            self._hook_shell_page_creation(self.shell)

            # Eagerly bind any pages already instantiated at boot
            pages_dict = getattr(self.shell, "pages", {})
            if isinstance(pages_dict, dict):
                for p_key, p_widget in list(pages_dict.items()):
                    if p_widget is not None:
                        self._on_shell_page_created(p_key, p_widget)

            # [STARTUP] Eager sayfa sorgulama döngüsü kaldırıldı: shell.page()
            # çağrısı lazy sayfaları zorla kurdurup açılışı yavaşlatıyordu.
            # Sayfa bağlaması page_created sinyali + _mount/open_tool
            # hook'larıyla (_hook_shell_page_creation) çalışmaya devam eder.

            self.shell.show()
            # F-0569 [O1]: shell artık var — F9 menüsüne shell'i de bağla.
            try:
                self.f9_menu.bind_targets(shell=self.shell)
            except Exception as _bind_exc:
                logger.debug("F9Menu shell bind atlandı: %s", _bind_exc)
        except Exception as exc:
            logger.error("Ana pencere kurulamadı, F9 menüsü ayrı açılacak: %s", exc, exc_info=True)
            self.shell = None
            try:
                self.f9_menu.show_docked_left()
                self._nudge_overlay_beside_menu()
            except Exception:
                logger.debug("F9 yedek penceresi açılamadı", exc_info=True)

    def _hook_shell_page_creation(self, shell: Any) -> None:
        """
        Safely hooks page creation on GatherionShell to ensure lazy-loaded or dynamically
        instantiated pages immediately trigger _on_shell_page_created.
        """
        if shell is None:
            return

        orig_mount = getattr(shell, "_mount", None)
        if callable(orig_mount) and not getattr(orig_mount, "_gatherion_hooked", False):
            def _wrapped_mount(key: str, widget: Any) -> Any:
                res = orig_mount(key, widget)
                try:
                    self._on_shell_page_created(key, res)
                except Exception as exc:
                    logger.debug("Shell _mount hook hatası (%s): %s", key, exc)
                return res

            _wrapped_mount._gatherion_hooked = True  # type: ignore[attr-defined]
            shell._mount = _wrapped_mount

        orig_open_tool = getattr(shell, "open_tool", None)
        if callable(orig_open_tool) and not getattr(orig_open_tool, "_gatherion_hooked", False):
            def _wrapped_open_tool(key: str) -> None:
                orig_open_tool(key)
                try:
                    p = None
                    if hasattr(shell, "pages") and isinstance(shell.pages, dict):
                        p = shell.pages.get(key)
                    if p is None and hasattr(shell, "page"):
                        p = shell.page(key)
                    if p is not None:
                        self._on_shell_page_created(key, p)
                except Exception as exc:
                    logger.debug("Shell open_tool hook hatası (%s): %s", key, exc)

            _wrapped_open_tool._gatherion_hooked = True  # type: ignore[attr-defined]
            shell.open_tool = _wrapped_open_tool

    @pyqtSlot(str, object)
    def _on_shell_page_created(self, key: str, widget: Any) -> None:
        """
        Dynamic wiring handler invoked whenever a shell page is instantiated.
        Wires calibration, map, yolo, stats, settings, radar, etc.
        """
        if widget is None:
            return

        norm_key = str(key or "").strip().lower()
        alias_map = {
            "yolo_factory": "yolo",
            "yolov8_factory": "yolo",
            "dataset_factory": "yolo",
            "train_ai": "yolo",
            "train_yolo": "yolo",
            "calibrate": "calibration",
            "analytics": "stats",
            "map_viewer": "map",
            "settings_window": "settings",
        }
        norm_key = alias_map.get(norm_key, norm_key)

        if norm_key == "calibration":
            self.calibration_win = widget
            if hasattr(widget, "template_updated"):
                _safe_reconnect(widget.template_updated, self._on_template_updated)

        elif norm_key == "map":
            self.map_viewer_win = widget
            self._connect_map_viewer_signals(widget)

        elif norm_key == "yolo":
            self.yolo_factory_win = widget
            self._connect_yolo_factory_signals(widget)

        elif norm_key == "stats":
            self.analytics_win = widget

        elif norm_key == "settings":
            self.settings_win = widget
            if hasattr(widget, "settings_saved"):
                _safe_reconnect(widget.settings_saved, self._on_settings_saved)
            if hasattr(widget, "quit_requested"):
                _safe_reconnect(widget.quit_requested, self.shutdown)
            if hasattr(widget, "route_changed"):
                _safe_reconnect(widget.route_changed, self._on_settings_route_changed)
            if hasattr(widget, "selective_gather_changed"):
                _safe_reconnect(widget.selective_gather_changed, self.handle_selective_gather_changed)

        elif norm_key == "radar":
            self.radar_win = widget

        elif norm_key == "guide":
            self.guide_win = widget

        elif norm_key == "mobile":
            self.mobile_connect_win = widget

    @pyqtSlot(str)
    @pyqtSlot()
    def _on_template_updated(self, template_path: Optional[str] = None) -> None:
        """
        Reloads vision templates (self.vision.load_templates()), minimap templates
        (self.minimap_scanner._load_optional_templates()), material templates, and
        updates exemplar consumers (self.worker.dataset_factory.auto_labeler / ExemplarMemory).
        """
        try:
            # 1. Reload Vision Engine templates
            if hasattr(self, "vision") and hasattr(self.vision, "load_templates"):
                try:
                    self.vision.load_templates()
                    logger.info("VisionEngine şablonları başarıyla yeniden yüklendi.")
                except Exception as exc:
                    logger.warning("VisionEngine load_templates hatası: %s", exc)

            # 2. Reload Minimap Scanner optional templates
            if hasattr(self, "minimap_scanner") and hasattr(self.minimap_scanner, "_load_optional_templates"):
                try:
                    self.minimap_scanner._load_optional_templates()
                    logger.info("MinimapScanner şablonları başarıyla yeniden yüklendi.")
                except Exception as exc:
                    logger.warning("MinimapScanner _load_optional_templates hatası: %s", exc)

            worker_scanner = getattr(self.worker, "minimap_scanner", None)
            if worker_scanner is not None and worker_scanner is not self.minimap_scanner:
                if hasattr(worker_scanner, "_load_optional_templates"):
                    try:
                        worker_scanner._load_optional_templates()
                    except Exception as exc:
                        logger.debug("İsteğe bağlı şablonlar yüklenemedi: %s", exc)

            # 3. Reload Material Scanner optional templates if available
            mat_scanner = getattr(self, "material_scanner", None)
            if mat_scanner is not None:
                try:
                    if hasattr(mat_scanner, "reload_templates"):
                        mat_scanner.reload_templates()
                    elif hasattr(mat_scanner, "_load_optional_gray_template"):
                        mat_scanner._disk_crest_tpl = (
                            mat_scanner._load_optional_gray_template("world_gather_crest_template.png")
                            or mat_scanner._load_optional_gray_template("world_gather_crest.png")
                        )
                        mat_scanner._disk_e_box_tpl = (
                            mat_scanner._load_optional_gray_template("e_gather_icon.png")
                            or mat_scanner._load_optional_gray_template("e_gather_icon_template.png")
                        )
                        mat_scanner._disk_sprout_tpl = (
                            mat_scanner._load_optional_gray_template("minimap_sprout_template.png")
                            or mat_scanner._load_optional_gray_template("minimap_sprout_node.png")
                        )
                except Exception as exc:
                    logger.debug("MaterialScanner şablon yenileme hatası: %s", exc)

            # 4. Update ExemplarMemory and YOLO AutoLabeler consumers
            try:
                from dataset_factory.exemplar_memory import get_exemplar_memory
                mem = get_exemplar_memory(json_path=LEARNED_EXEMPLARS_PATH)
                if mem is not None:
                    mem._last_loaded_mtime = -1.0
                    mem.load()
            except Exception as exc:
                logger.debug("ExemplarMemory reload hatası: %s", exc)

            factory = getattr(self.worker, "dataset_factory", None)
            if factory is not None:
                auto_labeler = getattr(factory, "auto_labeler", None)
                if auto_labeler is not None:
                    if hasattr(auto_labeler, "exemplar_memory") and auto_labeler.exemplar_memory is not None:
                        try:
                            auto_labeler.exemplar_memory._last_loaded_mtime = -1.0
                            auto_labeler.exemplar_memory.load()
                        except Exception as exc:
                            logger.debug("Exemplar memory yüklenemedi: %s", exc)
                    if hasattr(auto_labeler, "reload_exemplars"):
                        try:
                            auto_labeler.reload_exemplars()
                        except Exception as exc:
                            logger.debug("Exemplar'lar yeniden yüklenemedi: %s", exc)

            t_name = os.path.basename(str(template_path)) if template_path else "şablon"
            if hasattr(self, "overlay") and hasattr(self.overlay, "add_log"):
                self.overlay.add_log(f"Kalibrasyon şablonları güncellendi: {t_name}")
        except Exception as exc:
            logger.error("Şablon güncelleme zincirinde hata: %s", exc)

    @pyqtSlot(str, str)
    @pyqtSlot(str)
    def _on_settings_route_changed(self, region_name: str, route_name: Optional[str] = None) -> None:
        """
        Synchronizes f9_menu, overlay, map_viewer_win (if created), and radar_win when route changes.
        """
        try:
            reg = str(region_name or "Altgard").strip()
            rte = str(route_name or getattr(self.route_mgr, "active_route", "") or "All").strip()

            self.route_mgr.active_region = reg
            if rte and rte != "All":
                self.route_mgr.active_route = rte
            self.settings.active_region = reg
            if rte and rte != "All":
                self.settings.active_route = rte
            try:
                self.settings_mgr.save(self.settings)
            except Exception as exc:
                logger.debug("Ayarlar kaydedilemedi: %s", exc)

            # 1. Synchronize F9 Menu
            if self.f9_menu is not None:
                if hasattr(self.f9_menu, "set_active_state"):
                    try:
                        self.f9_menu.set_active_state(
                            active_mode=self.worker.mode,
                            active_region=reg,
                            active_route=rte,
                            minimap_auto_record=bool(
                                getattr(self.worker, "auto_cartography_enabled", True)
                            ),
                        )
                    except Exception as exc:
                        logger.debug("f9_menu set_active_state hatası: %s", exc)
                elif hasattr(self.f9_menu, "set_active_route"):
                    try:
                        self.f9_menu.set_active_route(reg, rte)
                    except Exception as exc:
                        logger.debug("f9_menu set_active_route hatası: %s", exc)

            # 2. Synchronize Overlay HUD
            if self.overlay is not None:
                if hasattr(self.overlay, "add_log"):
                    self.overlay.add_log(f"Rota senkronize edildi: {reg} · {rte}")

            self.worker._emit_hud_snapshot()

            # 3. Synchronize MapViewerWindow (if created)
            if self.map_viewer_win is not None:
                try:
                    if hasattr(self.map_viewer_win, "_region_name"):
                        self.map_viewer_win._region_name = reg
                    if hasattr(self.map_viewer_win, "_route_name"):
                        self.map_viewer_win._route_name = rte
                    if hasattr(self.map_viewer_win, "reload_map_and_nodes"):
                        self.map_viewer_win.reload_map_and_nodes()
                    if hasattr(self.map_viewer_win, "update"):
                        self.map_viewer_win.update()
                except Exception as exc:
                    logger.debug("map_viewer_win senkronizasyon hatası: %s", exc)

            # 4. Synchronize RadarWindow
            if self.radar_win is not None:
                try:
                    if hasattr(self.radar_win, "reload_map_and_nodes"):
                        self.radar_win.reload_map_and_nodes()
                    if hasattr(self.radar_win, "update"):
                        self.radar_win.update()
                except Exception as exc:
                    logger.debug("radar_win senkronizasyon hatası: %s", exc)

            # 5. Synchronize Shell
            if self.shell is not None:
                try:
                    if hasattr(self.shell, "_sync_map_from_settings"):
                        self.shell._sync_map_from_settings()
                    if hasattr(self.shell, "_publish_home"):
                        self.shell._publish_home()
                except Exception as exc:
                    logger.debug("shell senkronizasyon hatası: %s", exc)

        except Exception as exc:
            logger.error("_on_settings_route_changed hatası: %s", exc)

    def _connect_map_viewer_signals(self, widget: Any) -> None:
        """Connects region and node change handlers to MapViewerWindow."""
        if widget is None:
            return

        _safe_reconnect(getattr(widget, "node_toggled", None), self._on_map_node_toggled)
        _safe_reconnect(getattr(widget, "node_deleted", None), self._on_map_node_deleted)
        _safe_reconnect(getattr(widget, "route_changed", None), self._on_settings_route_changed)
        _safe_reconnect(getattr(widget, "region_changed", None), self._on_settings_route_changed)

        canvas = getattr(widget, "canvas", None)
        if canvas is not None:
            for sig_name in ("node_added", "node_created", "on_canvas_add_node", "on_canvas_add_node_signal"):
                sig = getattr(canvas, sig_name, None)
                if sig is not None and hasattr(sig, "connect"):
                    try:
                        sig.connect(self._on_canvas_node_added)
                    except Exception as exc:
                        logger.debug("MapViewer canvas sinyali (%s) bağlanamadı: %s", sig_name, exc)
            for sig_name in ("node_moved", "node_dragged", "node_position_changed"):
                sig = getattr(canvas, sig_name, None)
                if sig is not None and hasattr(sig, "connect"):
                    try:
                        sig.connect(self._on_canvas_node_moved)
                    except Exception as exc:
                        logger.debug("MapViewer canvas sinyali (%s) bağlanamadı: %s", sig_name, exc)

    def _connect_yolo_factory_signals(self, widget: Any) -> None:
        """Connects model publish signals/buttons to live ONNX reload in self.worker."""
        if widget is None:
            return
        # [P2-19/F044-02] Odak/modalite kararı: YOLO Fabrikası açıkken
        # bot durur. Interception HID enjeksiyonu OS seviyesinde gerçek
        # tuş gibi görünür — dialog, botun R basışını insanın R basışından
        # ayırt edemez; Alt-Tab auto-pause da kurtarmaz (own-PID pencereler
        # "Gatherion UI" sayılıp odaklı sayılır). Dialog görünürken
        # `BotWorker.pause`, gizlenince/kapanınca yalnızca dialogun
        # duraklattığı bot `resume` edilir (kullanıcının önceden
        # duraklattığı bota dokunulmaz — bayrak dialog tarafında tutulur).
        if hasattr(widget, "set_bot_pause_hooks"):
            try:
                worker = getattr(self, "worker", None)
                if worker is not None:
                    widget.set_bot_pause_hooks(
                        pause_fn=lambda: worker.pause(
                            "YOLO Fabrikası açık — bot duraklatıldı"
                        ),
                        resume_fn=worker.resume,
                        is_paused_fn=lambda: bool(getattr(worker, "is_paused", False)),
                    )
            except Exception as exc:
                logger.debug("YOLO factory bot-pause kancası bağlanamadı: %s", exc)
        if hasattr(widget, "model_published"):
            _safe_reconnect(widget.model_published, self.reload_ai_model)
        btn_pub = getattr(widget, "btn_publish", None)
        if btn_pub is not None and hasattr(btn_pub, "clicked"):
            try:
                btn_pub.clicked.connect(lambda: QTimer.singleShot(600, self.reload_ai_model))
            except Exception as exc:
                logger.debug("YOLO factory btn_publish connect atlandı: %s", exc)

    def _refresh_map_views(self) -> None:
        """HUD snapshot + radar + settings-table refresh shared by the four
        map/node sync handlers. Raises so callers keep their own error
        logging (identical to the pre-refactor behaviour)."""
        self.worker._emit_hud_snapshot()
        for window, method in (
            (getattr(self, "radar_win", None), "reload_map_and_nodes"),
            (getattr(self, "settings_win", None), "load_nodes_table"),
        ):
            if window is not None and hasattr(window, method):
                getattr(window, method)()

    @pyqtSlot(int)
    def _on_map_node_toggled(self, node_id: int) -> None:
        try:
            self.overlay.add_log(f"Harita: Nokta #{node_id} durumu güncellendi")
            self._refresh_map_views()
        except Exception as exc:
            logger.debug("Map node toggled hatası: %s", exc)

    @pyqtSlot(int)
    def _on_map_node_deleted(self, node_id: int) -> None:
        try:
            self.overlay.add_log(f"Harita: Nokta #{node_id} silindi")
            self._refresh_map_views()
        except Exception as exc:
            logger.debug("Map node deleted hatası: %s", exc)

    def _on_canvas_node_added(self, *args: Any) -> None:
        try:
            self._refresh_map_views()
        except Exception as exc:
            logger.debug("Canvas node added sync hatası: %s", exc)

    def _on_canvas_node_moved(self, *args: Any) -> None:
        try:
            self._refresh_map_views()
        except Exception as exc:
            logger.debug("Canvas node moved sync hatası: %s", exc)

    def reload_ai_model(self, model_path: Optional[str] = None) -> bool:
        """
        Hot-reloads published ONNX model weights in self.worker live without requiring bot restart.
        Candidate paths are anchored to BASE_DIR/models (best.onnx, yolov8n_gatherion.onnx, yolo11n_gatherion.onnx).
        """
        target_path: Optional[str] = None
        if model_path:
            p = model_path if os.path.isabs(model_path) else os.path.join(BASE_DIR, model_path)
            if os.path.isfile(p):
                target_path = os.path.abspath(p)

        if target_path is None:
            models_dir = MODELS_DIR
            for candidate in ("best.onnx", "yolov8n_gatherion.onnx", "yolo11n_gatherion.onnx"):
                c_path = os.path.join(models_dir, candidate)
                if os.path.isfile(c_path):
                    target_path = os.path.abspath(c_path)
                    break

        if target_path is None:
            logger.info("Yeniden yüklenecek ONNX modeli bulunamadı (models/best.onnx vb.).")
            return False

        logger.info("Canlı ONNX modeli yeniden yükleniyor: %s", target_path)
        reloaded = False

        # 1-3. Reload each detector via the shared helper (F-1137: a False
        # load result must never be reported as a live update).
        # NOTE: the state_machine detector's load exception used to log at
        # debug level; the helper logs it at warning like the other two.
        if hasattr(self, "yolo_detector") and self.yolo_detector is not None:
            if _reload_detector(self.yolo_detector, target_path, "self.yolo_detector"):
                reloaded = True

        # 2. Update self.worker.yolo_detector
        worker_detector = getattr(self.worker, "yolo_detector", None)
        if worker_detector is not None and worker_detector is not getattr(self, "yolo_detector", None):
            if _reload_detector(worker_detector, target_path, "worker.yolo_detector"):
                reloaded = True

        # 3. If worker has state_machine with yolo_detector
        sm = getattr(self.worker, "state_machine", None)
        if sm is not None:
            sm_detector = getattr(sm, "yolo_detector", None)
            if sm_detector is not None and sm_detector is not getattr(self, "yolo_detector", None):
                if _reload_detector(sm_detector, target_path, "state_machine yolo_detector"):
                    reloaded = True

        # 4. If worker has dedicated reload methods
        if hasattr(self.worker, "reload_model"):
            try:
                _reload_result = self.worker.reload_model(target_path)
                if _reload_result is not False:
                    reloaded = True
            except Exception as exc:
                logger.debug("worker.reload_model hatası: %s", exc)
        elif hasattr(self.worker, "reload_yolo_weights"):
            try:
                _reload_result = self.worker.reload_yolo_weights(target_path)
                if _reload_result is not False:
                    reloaded = True
            except Exception as exc:
                logger.debug("worker.reload_yolo_weights hatası: %s", exc)

        if reloaded:
            if hasattr(self, "overlay") and hasattr(self.overlay, "add_log"):
                self.overlay.add_log(f"YZ Modeli canlı güncellendi: {os.path.basename(target_path)}")
        else:
            logger.warning(
                "ONNX modeli canlı güncellenemedi: %s (hiçbir dedektör modeli yükleyemedi)",
                target_path,
            )
            if hasattr(self, "overlay") and hasattr(self.overlay, "add_log"):
                self.overlay.add_log("⚠️ YZ Modeli güncellenemedi (model yüklenemedi).")

        return reloaded

    def _nudge_overlay_beside_menu(self) -> None:
        """Keep the full-height HUD on the left edge instead of sliding it off the column."""
        try:
            dock = getattr(self.overlay, "dock_to_primary_left", None)
            if callable(dock):
                dock()
        except Exception as exc:
            logger.debug("Overlay sola yaslanamadı: %s", exc)

    @pyqtSlot()
    def on_f9_hotkey(self) -> None:
        """Toggle the left-docked F9 menu. Visibility never pauses or resumes the bot."""
        try:
            now = time.monotonic()
            if now - getattr(self, "_last_f9_ui_ts", 0.0) < 0.35:
                return
            self._last_f9_ui_ts = now

            shell = getattr(self, "shell", None)
            if shell is not None and hasattr(shell, "toggle_f9"):
                active_reg = self.route_mgr.active_region or "Altgard"
                active_rt = self.route_mgr.active_route or "Dredgion Crash Site"
                if hasattr(self.f9_menu, "set_active_state"):
                    self.f9_menu.set_active_state(
                        active_mode=self.worker.mode,
                        active_region=active_reg,
                        active_route=active_rt,
                        minimap_auto_record=bool(
                            getattr(self.worker, "auto_cartography_enabled", True)
                        ),
                    )
                opening = not shell.is_f9_open() if hasattr(shell, "is_f9_open") else True
                if opening and hasattr(self.input_ctrl, "free_mouse_cursor"):
                    self.input_ctrl.free_mouse_cursor()
                shell.toggle_f9()
                if opening:
                    try:
                        winsound.Beep(900, 75)
                    except Exception:
                        logger.debug("Beep skipped")
                return

            if self.f9_menu.is_panel_open():
                self.f9_menu.dismiss()
                return

            if hasattr(self.input_ctrl, "free_mouse_cursor"):
                self.input_ctrl.free_mouse_cursor()

            try:
                winsound.Beep(900, 75)
            except Exception:
                logger.debug("Beep skipped")

            active_reg = self.route_mgr.active_region or "Altgard"
            active_rt = self.route_mgr.active_route or "Dredgion Crash Site"
            if hasattr(self.f9_menu, "set_active_state"):
                self.f9_menu.set_active_state(
                    active_mode=self.worker.mode,
                    active_region=active_reg,
                    active_route=active_rt,
                    minimap_auto_record=bool(
                        getattr(self.worker, "auto_cartography_enabled", True)
                    ),
                )
            self.f9_menu.show_docked_left()
        except Exception as exc:
            logger.error("F9 kısayolu işlenirken hata: %s", exc)

    @pyqtSlot()
    def on_f9_enter_hotkey(self) -> None:
        """Resumes the bot in the active mode when F9 -> Enter / R is pressed."""
        try:
            if hasattr(self.watchdog, "set_paused"):
                self.watchdog.set_paused(False, state="WALKING")
            self.worker.resume()
            try:
                winsound.Beep(1350, 110)
            except Exception:
                logger.debug("Beep skipped")
        except Exception as exc:
            logger.error("F9+Enter devam komutu işlenirken hata: %s", exc)

    @pyqtSlot()
    def on_f10_hotkey(self) -> None:
        """Toggles bot pause/resume or stops the bot cleanly (F10 hotkey)."""
        try:
            if not self.worker.is_paused:
                pause_name = getattr(self, "_pause_chord_label", "Ctrl+Alt+F10")
                self.worker.pause(f"Durduruldu ({pause_name})")
                self.overlay.add_log(f"🛑 [{pause_name}] Bot Durduruldu!")
                logger.info("[Hotkey] %s -> bot kullanıcı tarafından durduruldu.", pause_name)
                try:
                    winsound.Beep(700, 180)
                except Exception as exc:
                    logger.debug("Bip sesi çalınamadı: %s", exc)
            else:
                pause_name = getattr(self, "_pause_chord_label", "Ctrl+Alt+F10")
                self.worker.resume()
                self.overlay.add_log(f"▶️ [{pause_name}] Bot Başlatıldı / Devam Ediyor.")
                logger.info("[Hotkey] %s -> bot tekrar başlatıldı/devam ettirildi.", pause_name)
                try:
                    winsound.Beep(1200, 150)
                except Exception as exc:
                    logger.debug("Bip sesi çalınamadı: %s", exc)
        except Exception as exc:
            logger.error("F10 kısayolu işlenirken hata: %s", exc)

    @pyqtSlot()
    def _on_f8_record_node(self) -> None:
        """
        Records the current player location as a new node in any mode (F8),
        logs a clear Turkish confirmation, and refreshes OverlayHUD & MapViewerWindow immediately.
        """
        try:
            recorded = self.worker.trigger_f8_record()
            region = self.route_mgr.active_region or "Altgard"
            route = self.route_mgr.active_route or "Dredgion Crash Site"
            # F-1138: trigger_f8_record() None dönerken (kuyruğa alındı veya
            # başarısız) koşulsuz "kaydedildi" yazılırdı.
            if isinstance(recorded, dict):
                logger.info(
                    "F8 ile mevcut konum rota noktası olarak kaydedildi (%s · %s).",
                    region,
                    route,
                )
                self.overlay.add_log(f"F8: Yeni nokta kaydedildi ({region} · {route})")
            elif getattr(self.worker, "isRunning", lambda: False)():
                # Worker çalışırken kayıt kuyruğa alınır; sonuç
                # node_recorded sinyaliyle gelir.
                logger.info("F8 kaydı kuyruğa alındı (%s · %s).", region, route)
                self.overlay.add_log(f"F8: Nokta kaydı kuyruğa alındı ({region} · {route})")
            else:
                logger.warning("F8 nokta kaydı başarısız oldu (%s · %s).", region, route)
                self.overlay.add_log(f"⚠️ F8: Nokta kaydedilemedi ({region} · {route})")
                return
            self.worker._emit_hud_snapshot()

            if self.map_viewer_win is not None and self.map_viewer_win.isVisible():
                if hasattr(self.map_viewer_win, "reload_map_and_nodes"):
                    self.map_viewer_win.reload_map_and_nodes()
                self.map_viewer_win.update()

            if recorded and isinstance(recorded, dict):
                self._on_node_recorded(recorded)
        except Exception as exc:
            logger.error("F8 nokta kayıt kısayolu işlenirken hata: %s", exc)

    @pyqtSlot()
    def on_f8_hotkey(self) -> None:
        """Backward-compatible slot alias for _on_f8_record_node."""
        self._on_f8_record_node()

    @pyqtSlot(str)
    def _on_menu_action(self, action: str) -> None:
        """Handles selections emitted from F9Menu or System Tray."""
        self.handle_menu_action(action)

    def _activate_bot_mode(self, mode_value: Optional[str], watchdog_state: str) -> None:
        """Shared resume/set_mode sequence for the menu mode actions listed
        in _MENU_MODE_ACTIONS (mode_value None = plain resume, no set_mode)."""
        if hasattr(self.watchdog, "set_paused"):
            self.watchdog.set_paused(False, state=watchdog_state)
        if mode_value is not None:
            self.worker.set_mode(mode_value)
        self.worker.resume()

    @pyqtSlot(str)
    def handle_menu_action(self, action: str) -> None:
        """Handles selections emitted from F9Menu or System Tray."""
        act = (action or "").strip().lower()
        logger.info("Menü eylemi tetiklendi: %s", act)

        mode_action = _MENU_MODE_ACTIONS.get(act)
        if mode_action is not None:
            mode_value, watchdog_state = mode_action
            self._activate_bot_mode(mode_value, watchdog_state)
        elif act in ("toggle_minimap_auto_record", "minimap_auto_record"):
            if hasattr(self.f9_menu, "is_minimap_auto_record_enabled"):
                self._on_minimap_auto_record_toggled(
                    self.f9_menu.is_minimap_auto_record_enabled()
                )
        elif act in ("mobile", "mobile_connect", "mobile_dashboard"):
            self.open_mobile_connect_dialog()
        elif act == "settings":
            self.open_settings_window()
        elif act in ("map", "map_viewer"):
            self.open_map_viewer()
        elif act == "radar":
            self.open_cartography_radar()
        elif act == "analytics":
            self.open_analytics_dialog()
        elif act == "calibrate":
            self.open_calibration_wizard()
        elif act in ("yolo_factory", "yolov8_factory", "dataset_factory", "train_ai", "train_yolo"):
            self.trigger_ai_training_and_open_dialog()
        elif act in ("open_log", "log"):
            self.open_log_file()
        elif act in ("guide", "stuck_guide"):
            self.open_guide_dialog()
        elif act in ("copy_support", "support"):
            # F-1192: copy_report_to_clipboard rapor metnini döner, kopya
            # başarısını bildirmez; pano içeriği doğrulanarak yazılır.
            report_text = SupportDiagnostics.copy_report_to_clipboard(
                settings=self.settings,
                route_manager=self.route_mgr,
                telemetry=self.telemetry,
                log_path=LOG_FILE,
            )
            copied = False
            try:
                from PyQt6.QtWidgets import QApplication

                app = QApplication.instance()
                clipboard = QApplication.clipboard() if app is not None else None
                if clipboard is not None and isinstance(report_text, str) and report_text:
                    copied = clipboard.text() == report_text
            except Exception as exc:
                logger.debug("Pano doğrulama okuması atlandı: %s", exc)
            if copied:
                self.overlay.add_log("Destek tanılama raporu panoya kopyalandı")
            else:
                logger.warning("Destek raporu panoya kopyalanamadı (doğrulama başarısız).")
                self.overlay.add_log("⚠️ Destek raporu panoya kopyalanamadı.")
        elif act in ("exit", "quit"):
            self.shutdown()
        else:
            # F-1252: bilinmeyen menü eylemi sessizce yutulurdu.
            logger.warning("Bilinmeyen menü eylemi yoksayıldı: %r", action)
            try:
                self.overlay.add_log(f"⚠️ Bilinmeyen menü eylemi: {act}")
            except Exception as exc:
                logger.debug("Bilinmeyen menü eylemi overlay'e yazılamadı: %s", exc)

    @pyqtSlot(str, str)
    def handle_route_selected(self, region: str, route: str) -> None:
        """Updates active region and route when chosen from F9Menu submenu."""
        try:
            self.route_mgr.active_region = region
            self.route_mgr.active_route = route
            self.settings.active_region = region
            self.settings.active_route = route
            self.settings_mgr.save(self.settings)
            self._sync_encrypted_vault()
            self.worker._auto_sort_sequential_route(mode="smooth_loop")
            self.overlay.add_log(f"Aktif rota: {region} · {route}")
            self.worker._emit_hud_snapshot()
            if self.map_viewer_win is not None and self.map_viewer_win.isVisible():
                if hasattr(self.map_viewer_win, "reload_map_and_nodes"):
                    self.map_viewer_win.reload_map_and_nodes()
        except Exception as exc:
            logger.error("Aktif rota değiştirilirken hata: %s", exc)

    @pyqtSlot(list, bool)
    @pyqtSlot(list)
    def handle_selective_gather_changed(
        self, materials: list, auto_proximity: bool = True
    ) -> None:
        """
        Orchestrates live selective material gathering across subsystems:
        1. Updates self.bot_worker.set_selected_gather_materials(materials).
        2. Calls self.routes.build_proximity_material_route(...) if auto_proximity is True.
        3. Notifies self.map_viewer_win.set_material_filter(materials) if opened/available.
        4. Notifies self.radar_win (CartographyRadarDialog) if opened/available.
        5. Updates self.overlay_hud (OverlayHUD) of active selective gathering status.
        6. Persists settings and keeps F9Menu & SettingsWindow in sync.
        """
        try:
            mats = [str(m).strip() for m in materials if str(m).strip()] if materials else []
            is_enabled = bool(mats)
            auto_prox = bool(auto_proximity)

            # 1. Update Config & Persist
            if hasattr(self.settings, "gathering"):
                self.settings.gathering.selective_gather_enabled = is_enabled
                self.settings.gathering.selected_materials = list(mats)
                self.settings.gathering.proximity_route_auto = auto_prox
                try:
                    self.settings_mgr.save(self.settings)
                    self._sync_encrypted_vault()
                except Exception as save_exc:
                    logger.debug("Seç-topla ayarları kaydedilirken hata: %s", save_exc)

            # 2. Update BotWorker
            worker = getattr(self, "bot_worker", None) or getattr(self, "worker", None)
            if worker is not None:
                if hasattr(worker, "set_selected_gather_materials"):
                    try:
                        worker.set_selected_gather_materials(mats)
                    except Exception as w_exc:
                        logger.debug("worker.set_selected_gather_materials hatası: %s", w_exc)
                setattr(worker, "selective_gather_enabled", is_enabled)
                setattr(worker, "selected_materials", list(mats))
                setattr(worker, "proximity_route_auto", auto_prox)

            # 3. Call build_proximity_material_route on route manager if auto_proximity is True
            routes = getattr(self, "routes", None) or getattr(self, "route_mgr", None)
            if auto_prox and routes is not None:
                region = (
                    getattr(routes, "active_region", "")
                    or getattr(self.settings, "active_region", "Altgard")
                )
                start_xy = (
                    getattr(worker, "current_xy", None)
                    or getattr(worker, "current_player_xy", None)
                    or getattr(worker, "_player_xy", None)
                )
                if hasattr(routes, "build_proximity_material_route"):
                    try:
                        # S-6901: dönüş değeri yok sayılmayacak — [] dönerse
                        # (bölge yüklenemedi / materyale uyan düğüm yok,
                        # core/route_manager.py:2155,2166) 'oluşturuldu' logu
                        # yazılmaz, uyarı loglanır.
                        route_nodes = routes.build_proximity_material_route(
                            region_name=region,
                            target_materials=mats,
                            start_xy=start_xy,
                            persist_route=True,
                        )
                        if route_nodes:
                            logger.info(
                                "Yakınlığa göre materyal rotası oluşturuldu: %s · %s (%d düğüm)",
                                region,
                                mats,
                                len(route_nodes),
                            )
                        else:
                            logger.warning(
                                "Yakınlığa göre materyal rotası oluşturulamadı: %s · %s "
                                "— bölge yüklenemedi veya materyale uyan düğüm yok",
                                region,
                                mats,
                            )
                    except Exception as prox_exc:
                        logger.debug(
                            "build_proximity_material_route çağrı hatası: %s", prox_exc
                        )
                if worker is not None and hasattr(worker, "_auto_sort_sequential_route"):
                    try:
                        worker._auto_sort_sequential_route(mode="smooth_loop")
                    except Exception as exc:
                        logger.debug("Seçili rota otomatik sıralanamadı: %s", exc)

            # 4. Notify MapViewerWindow
            map_win = getattr(self, "map_viewer_win", None)
            if map_win is not None:
                try:
                    if hasattr(map_win, "set_material_filter"):
                        map_win.set_material_filter(mats)
                    if hasattr(map_win, "reload_map_and_nodes"):
                        map_win.reload_map_and_nodes()
                    if hasattr(map_win, "update"):
                        map_win.update()
                except Exception as map_exc:
                    logger.debug(
                        "MapViewerWindow materyal filtresi senkronizasyon hatası: %s",
                        map_exc,
                    )

            # 5. Notify CartographyRadarDialog
            radar_win = getattr(self, "radar_win", None)
            if radar_win is not None:
                try:
                    if hasattr(radar_win, "set_material_filter"):
                        radar_win.set_material_filter(mats)
                    if hasattr(radar_win, "reload_map_and_nodes"):
                        radar_win.reload_map_and_nodes()
                    if hasattr(radar_win, "update"):
                        radar_win.update()
                except Exception as rad_exc:
                    logger.debug(
                        "CartographyRadarDialog materyal filtresi senkronizasyon hatası: %s",
                        rad_exc,
                    )

            # 6. Notify OverlayHUD
            overlay = getattr(self, "overlay_hud", None) or getattr(self, "overlay", None)
            if overlay is not None:
                mat_desc = ", ".join(mats) if mats else "Tümü (Kapalı)"
                status_msg = (
                    f"🎯 Seç-Topla: {mat_desc}"
                    if is_enabled
                    else "🎯 Seç-Topla: Kapalı (Tüm Materyaller)"
                )
                if hasattr(overlay, "add_log"):
                    overlay.add_log(status_msg)
                if hasattr(overlay, "update_state"):
                    overlay.update_state({
                        "selective_gather_enabled": is_enabled,
                        "selected_materials": mats,
                        "proximity_route_auto": auto_prox,
                    })
                if hasattr(overlay, "set_selective_gather_status"):
                    overlay.set_selective_gather_status(is_enabled, mats)

            # 7. Synchronize F9Menu and SettingsWindow
            if hasattr(self, "f9_menu") and self.f9_menu is not None:
                if hasattr(self.f9_menu, "set_selected_materials"):
                    try:
                        self.f9_menu.set_selected_materials(
                            mats, auto_proximity=auto_prox
                        )
                    except Exception as exc:
                        logger.debug("F9 menü malzeme seçimi uygulanamadı: %s", exc)

            if hasattr(self, "settings_win") and self.settings_win is not None:
                if hasattr(self.settings_win, "set_selected_materials"):
                    try:
                        self.settings_win.set_selected_materials(
                            mats, auto_proximity=auto_prox
                        )
                    except Exception as exc:
                        logger.debug("Ayar penceresi malzeme seçimi uygulanamadı: %s", exc)

            if worker is not None and hasattr(worker, "_emit_hud_snapshot"):
                worker._emit_hud_snapshot()

            logger.info(
                "handle_selective_gather_changed tamamlandı: materyaller=%s, auto_proximity=%s",
                mats,
                auto_prox,
            )
        except Exception as exc:
            logger.error("handle_selective_gather_changed işlenirken hata: %s", exc)

    @pyqtSlot(dict)
    def _on_node_recorded(self, node_dict: dict) -> None:
        """Refreshes OverlayHUD, MapViewerWindow, SettingsWindow, and Radar whenever a node is recorded."""
        try:
            self.worker._emit_hud_snapshot()
            if hasattr(self.overlay, "update_minimap_mapped") and isinstance(node_dict, dict):
                if str(node_dict.get("source_type", "")).lower() == "minimap_auto":
                    cur_cnt = int(getattr(self.overlay, "minimap_mapped_count", 0)) + 1
                    self.overlay.update_minimap_mapped(cur_cnt)

            if self.map_viewer_win is not None and self.map_viewer_win.isVisible():
                if hasattr(self.map_viewer_win, "reload_map_and_nodes"):
                    self.map_viewer_win.reload_map_and_nodes()
                self.map_viewer_win.update()

            if self.settings_win is not None and self.settings_win.isVisible():
                if hasattr(self.settings_win, "load_nodes_table"):
                    self.settings_win.load_nodes_table()

            if self.radar_win is not None and self.radar_win.isVisible():
                if hasattr(self.radar_win, "add_discovered_node"):
                    self.radar_win.add_discovered_node(node_dict)
        except Exception as exc:
            logger.debug("UI nokta yenileme hatası: %s", exc)

    def _on_watchdog_recovery(self, reason: Any) -> None:
        """
        Callback invoked by BotWatchdog if a stall or display change is detected.
        NEVER resumes or unpauses a PAUSED or IDLE worker!
        """
        try:
            state_obj = getattr(self.worker, "state", "PAUSED")
            state_name = getattr(state_obj, "name", str(state_obj)).upper()
            if (
                state_name in ("PAUSED", "IDLE", "STOPPED", "RESTING_BREAK", "BREAK")
                or getattr(self.worker, "is_paused", False)
            ):
                if hasattr(self.watchdog, "set_paused"):
                    self.watchdog.set_paused(True, state=state_name)
                return

            if isinstance(reason, dict):
                payload_state = str(reason.get("state", "")).upper()
                if payload_state in ("PAUSED", "IDLE", "STOPPED", "RESTING_BREAK", "BREAK"):
                    if hasattr(self.watchdog, "set_paused"):
                        self.watchdog.set_paused(True, state=payload_state)
                    return

            logger.warning("Watchdog kurtarma tetiklendi (güvenli duraklatma): %s", reason)
            self.worker.pause(f"Watchdog güvenli duraklatma: {reason}")
        except Exception as exc:
            logger.error("Watchdog kurtarma işleyicisi hatası: %s", exc)

    # =========================================================================
    # Dialog Launchers
    # =========================================================================

    def _show_shell_tab(self, key: str) -> bool:
        shell = getattr(self, "shell", None)
        if shell is None:
            return False
        shell.open_tool(key)
        return True

    def open_mobile_connect_dialog(self) -> None:
        """Opens the Mobile Remote Control, QR Code & Cloud Tunnel dialog."""
        if self._show_shell_tab("mobile"):
            return
        try:
            from ui.mobile_connect_dialog import MobileConnectDialog

            if self.mobile_connect_win is None:
                self.mobile_connect_win = MobileConnectDialog(
                    mobile_server=self.mobile_server,
                    tunnel_mgr=self.tunnel_mgr,
                    parent=self.settings_win,
                )
            self.mobile_connect_win.show()
            self.mobile_connect_win.raise_()
            self.mobile_connect_win.activateWindow()
        except Exception as exc:
            logger.error("MobileConnectDialog açılamadı: %s", exc)

    def open_settings_window(self) -> None:
        if self._show_shell_tab("settings"):
            return
        try:
            if self.settings_win is None:
                # S-4301: live server + shared token authenticator are
                # wired in so the "Mobil Token Döndür" button in the
                # live SettingsWindow actually rotates the token the
                # server enforces (previously the rotation code lived
                # only in the never-instantiated SettingsDialog).
                self.settings_win = SettingsWindow(
                    self.settings_mgr,
                    self.route_mgr,
                    mobile_server=self.mobile_server,
                    mobile_token_auth=self.mobile_token_auth,
                    telemetry=self.telemetry,
                )
                self.settings_win.setWindowTitle(f"{self.APP_TITLE} — Ayarlar & Noktalar")
                if hasattr(self.settings_win, "settings_saved"):
                    self.settings_win.settings_saved.connect(self._on_settings_saved)
                if hasattr(self.settings_win, "quit_requested"):
                    self.settings_win.quit_requested.connect(self.shutdown)
                if hasattr(self.settings_win, "route_changed"):
                    self.settings_win.route_changed.connect(self._on_settings_route_changed)
                if hasattr(self.settings_win, "selective_gather_changed"):
                    self.settings_win.selective_gather_changed.connect(
                        self.handle_selective_gather_changed
                    )
            self.settings_win.show()
            self.settings_win.raise_()
            self.settings_win.activateWindow()
        except Exception as exc:
            logger.error("SettingsWindow açılamadı: %s", exc, exc_info=True)

    @staticmethod
    def _resolve_pause_on_alt_tab(settings: Any) -> bool:
        """F-1193: `or` mantığı açık False ayarını yok sayardı.

        route.pause_on_alt_tab açıkça False ise False'dur; yalnızca
        ayarlanmamışsa (None/eksik) gathering değerine, o da yoksa True'ya
        düşülür.
        """
        route_val = getattr(getattr(settings, "route", None), "pause_on_alt_tab", None)
        if route_val is not None:
            return bool(route_val)
        gathering_val = getattr(getattr(settings, "gathering", None), "pause_on_alt_tab", None)
        if gathering_val is not None:
            return bool(gathering_val)
        return True

    def _on_settings_saved(self) -> None:
        try:
            self.settings = self.settings_mgr.load()
            self._sync_encrypted_vault()
            self.input_ctrl.set_test_mode(bool(self.settings.gathering.test_mode))
            if self.settings.gathering.show_overlay:
                self.overlay.show()
            else:
                self.overlay.hide()
            self.overlay.add_log("Ayarlar kaydedildi ve uygulandı")
            if hasattr(self.worker, "set_pause_on_alt_tab"):
                self.worker.set_pause_on_alt_tab(
                    self._resolve_pause_on_alt_tab(self.settings)
                )
            # F-23-01/F-23-11: iksir eşikleri + keşif oranını canlı nesnelere
            # yaz (ManaPhaseCombat.set_thresholds, SlotEconomy.set_explore_ratio).
            # Eskiden bu değerler yalnızca bot başlangıcında okunuyor, ayar
            # değişse bile yeniden başlatmaya kadar eski değer kullanılıyordu.
            if hasattr(self.worker, "refresh_consumable_settings"):
                try:
                    self.worker.refresh_consumable_settings()
                except Exception as exc:
                    logger.debug("Canlı tüketim/keşif ayarları uygulanamadı: %s", exc)
            self._apply_hotkey_labels()
            self._register_native_chords()
            self.worker._emit_hud_snapshot()
        except Exception as exc:
            logger.error("Kaydedilen ayarlar uygulanırken hata: %s", exc)

    def open_map_viewer(self) -> None:
        if self._show_shell_tab("map"):
            return
        try:
            region = self.route_mgr.active_region or "Altgard"
            route = self.route_mgr.active_route or "Dredgion Crash Site"
            self.map_viewer_win = MapViewerWindow(
                route_manager=self.route_mgr, region_name=region, route_name=route
            )
            self._connect_map_viewer_signals(self.map_viewer_win)
            # v20/map-nodes: bağımsız pencere de keşif anında yenilensin.
            # register_map_viewer üretimde hiç çağrılmıyordu ve bu pencereye
            # set_auto_cartographer da yapılmıyordu — poll'u kartografsız
            # no-op'tu, düğüm ancak pencere yeniden açılınca görünürdü.
            try:
                explorer = getattr(self, "explorer_ctrl", None)
                if explorer is not None and hasattr(explorer, "register_map_viewer"):
                    explorer.register_map_viewer(self.map_viewer_win)
                cartographer = getattr(explorer, "auto_cartographer", None) if explorer is not None else None
                if cartographer is not None and hasattr(self.map_viewer_win, "set_auto_cartographer"):
                    self.map_viewer_win.set_auto_cartographer(cartographer)
            except Exception as exc:
                logger.debug("Harita görüntüleyici canlı senkronizasyona bağlanamadı: %s", exc)
            self.map_viewer_win.show()
            self.map_viewer_win.raise_()
        except Exception as exc:
            logger.error("MapViewerWindow açılamadı: %s", exc)

    def open_cartography_radar(self) -> None:
        if self._show_shell_tab("radar"):
            return
        try:
            if self.radar_win is None:
                self.radar_win = CartographyRadarDialog(
                    route_manager=self.route_mgr,
                    cartography_engine=self.explorer_ctrl,
                )
            self.radar_win.show()
            self.radar_win.raise_()
            self.radar_win.activateWindow()
        except Exception as exc:
            logger.error("CartographyRadarDialog açılamadı: %s", exc)

    def open_analytics_dialog(self) -> None:
        if self._show_shell_tab("stats"):
            return
        try:
            if self.analytics_win is None:
                self.analytics_win = AnalyticsDialog(telemetry=self.telemetry)
            self.analytics_win.show()
            self.analytics_win.raise_()
            self.analytics_win.activateWindow()
        except Exception as exc:
            logger.error("AnalyticsDialog açılamadı: %s", exc)

    def open_calibration_wizard(self) -> None:
        if self._show_shell_tab("calibration"):
            return
        try:
            if self.calibration_win is None:
                self.calibration_win = CalibrationWizard()
                if hasattr(self.calibration_win, "template_updated"):
                    self.calibration_win.template_updated.connect(self._on_template_updated)
            self.calibration_win.show()
            self.calibration_win.raise_()
            self.calibration_win.activateWindow()
        except Exception as exc:
            logger.error("CalibrationWizard açılamadı: %s", exc)

    def open_guide_dialog(self) -> None:
        if self._show_shell_tab("guide"):
            return
        try:
            if self.guide_win is None:
                self.guide_win = GuideDialog()
            self.guide_win.show()
            self.guide_win.raise_()
            self.guide_win.activateWindow()
        except Exception as exc:
            logger.error("GuideDialog açılamadı: %s", exc)

    def open_yolo_factory_dialog(self) -> None:
        """Opens the Self-Training YOLO Dataset Factory & RTX Trainer dialog."""
        if self._show_shell_tab("yolo"):
            return
        try:
            from ui.yolo_factory_dialog import YOLOFactoryDialog

            if self.yolo_factory_win is None:
                factory_ctrl = getattr(self.worker, "dataset_factory", None)
                self.yolo_factory_win = YOLOFactoryDialog(
                    factory_controller=factory_ctrl,
                    parent=self.settings_win,
                )
                if hasattr(self.yolo_factory_win, "set_shared_yolo_detector"):
                    self.yolo_factory_win.set_shared_yolo_detector(self.yolo_detector)
                self._connect_yolo_factory_signals(self.yolo_factory_win)
            self.yolo_factory_win.show()
            self.yolo_factory_win.raise_()
            self.yolo_factory_win.activateWindow()
        except Exception as exc:
            logger.error("YOLOFactoryDialog açılamadı: %s", exc)

    def trigger_ai_training_and_open_dialog(self) -> None:
        """Launches 1-click RTX AI model training in background and opens the factory dialog."""
        try:
            if not self._show_shell_tab("yolo"):
                self.open_yolo_factory_dialog()
            dialog = getattr(self, "yolo_factory_win", None)
            if dialog is None and self.shell is not None:
                for getter in ("get_page_if_created", "get_page", "page"):
                    fn = getattr(self.shell, getter, None)
                    if callable(fn):
                        try:
                            dialog = fn("yolo")
                            if dialog is not None:
                                break
                        except Exception as exc:
                            logger.debug("YOLO dialog getter hatası yoksayıldı: %s", exc)
                if dialog is None and hasattr(self.shell, "pages"):
                    dialog = self.shell.pages.get("yolo")
                if dialog is not None:
                    self.yolo_factory_win = dialog
                    self._on_shell_page_created("yolo", dialog)

            factory = getattr(self.worker, "dataset_factory", None)
            if dialog is not None and hasattr(dialog, "on_train_rtx_clicked"):
                already = (
                    getattr(dialog, "_training_worker", None) is not None
                    and dialog._training_worker.isRunning()
                )
                dialog.on_train_rtx_clicked()
                if not already:
                    self.overlay.add_log(
                        "Veri seti taraması sürüyor. Eğitim, oyun kapalıysa arka planda başlar."
                    )
            elif factory is not None and hasattr(factory, "run_one_click_rtx_training"):
                rtx_trainer = getattr(factory, "rtx_trainer", None)
                is_busy = False
                if rtx_trainer is not None:
                    is_busy = getattr(rtx_trainer, "is_training", False) or getattr(rtx_trainer, "_training_active", False)

                if not is_busy:
                    result = factory.run_one_click_rtx_training(async_mode=True)
                    if isinstance(result, dict) and result.get("started") is False:
                        reason = str(result.get("reason") or "")
                        if result.get("waiting_for_game_exit"):
                            self.overlay.add_log(
                                "⏳ [YZ Eğitimi] Eğitim, oyun kapanana kadar bekliyor."
                            )
                        elif reason:
                            self.overlay.add_log(f"⏳ [YZ Eğitimi] {reason}")
                        else:
                            self.overlay.add_log("⏳ [YZ Eğitimi] Model eğitimi şu anda arka planda devam ediyor.")
                    else:
                        self.overlay.add_log(
                            "Veri seti taraması sürüyor. Eğitim, oyun kapalıysa arka planda başlar."
                        )
                else:
                    self.overlay.add_log("⏳ [YZ Eğitimi] Model eğitimi şu anda arka planda devam ediyor.")
            else:
                self.overlay.add_log("YOLO Fabrikası sekmesi açıldı.")
        except Exception as exc:
            logger.error("Tek tıkla YZ eğitimi başlatılamadı: %s", exc)
        sounds = getattr(self, "sounds", None)
        if sounds is not None and hasattr(sounds, "play_click"):
            try:
                sounds.play_click()
            except Exception:
                logger.debug("Tıklama sesi atlandı", exc_info=True)

    def open_log_file(self) -> None:
        if self._show_shell_tab("logs"):
            return
        try:
            if os.path.exists(LOG_FILE):
                os.startfile(LOG_FILE)
        except Exception as exc:
            logger.warning("Günlük dosyası harici olarak açılamadı: %s", exc)

    def get_mobile_dashboard_url(self) -> Optional[str]:
        """Canlı mobil panel URL'si. sweep-26 F5: sunucu çalışmıyorken bile
        ölü `http://<LAN>:8765` döndürüyordu; artık None döner — banner
        ölü link basmaz."""
        try:
            if self.mobile_server is not None:
                if hasattr(self.mobile_server, "get_lan_url") and callable(
                    self.mobile_server.get_lan_url
                ):
                    return str(self.mobile_server.get_lan_url())
                if hasattr(self.mobile_server, "lan_url") and self.mobile_server.lan_url:
                    return str(self.mobile_server.lan_url)
        except Exception as exc:
            logger.warning("Mobil sunucu URL'si okunamadı: %s", exc)
            return None
        return None

    def get_main_controller_ready_flag(self) -> dict:
        """Returns the readiness flag for GatherionController."""
        health = self.get_main_controller_health_summary()
        return {
            "module": "main",
            "ready": bool(health.get("controller_ready")),
            "checks": health.get("checks", {}),
        }

    def get_main_controller_health_summary(self) -> dict:
        """Returns health summary for GatherionMainController safely."""
        # F-1253: her zaman "hazır/sertifikalı" dönülmezdi; alt sistemlerin
        # gerçek hazırlık bayrakları toplanır.
        try:
            checks: dict = {}
            capture = getattr(self, "capture", None)
            if capture is not None and hasattr(capture, "get_capture_health_summary"):
                checks["capture"] = bool(capture.get_capture_health_summary().get("ready"))
            optimizer = getattr(self, "route_optimizer", None)
            if optimizer is not None and hasattr(optimizer, "get_path_optimizer_health_summary"):
                checks["path_optimizer"] = bool(
                    optimizer.get_path_optimizer_health_summary().get("ready")
                )
            dxg = getattr(self, "dxg_capture", None)
            if dxg is not None and hasattr(dxg, "get_stealth_capture_health_summary"):
                checks["stealth_capture"] = bool(
                    dxg.get_stealth_capture_health_summary().get("stealth_capture_ready")
                )
            worker = getattr(self, "worker", None)
            checks["worker_present"] = worker is not None
            ready = bool(checks) and all(checks.values())
            wired = sum(1 for v in checks.values() if v)
            return {
                "controller_ready": ready,
                "enterprise_modules_wired": wired,
                "checks": checks,
            }
        except Exception as exc:
            return {"controller_ready": False, "enterprise_modules_wired": 0, "error": str(exc)}

    def get_main_version_info(self) -> dict:
        """Returns version info for main module."""
        return {"module": "main", "version": __version__, "ready": True}

    def shutdown(self) -> None:
        """Gracefully stops all background threads, mobile server, tunnel, saves telemetry, and exits."""
        if getattr(self, "_shutdown_done", False):
            return
        self._shutdown_done = True
        # S-0204: kapanan örnek artık canlı prob sayılmaz.
        global _LIVE_ORCHESTRATOR_REF
        _LIVE_ORCHESTRATOR_REF = None
        logger.info(f"Gatherion-Local v{__version__} (Türkçe) kapatılıyor")
        # v20/bot-ani-kapanma: temiz kapanis isareti -- atexit kaydi
        # graceful=True yazarsa olum normal kapanisti; bu satir olmadan
        # atexit'e ulasilmasi anormal olumun kanitidir.
        try:
            note_graceful_shutdown()
        except Exception:
            pass
        try:
            self._persist_settings_on_exit()
        except Exception as exc:
            # sweep Y-4: kapanışta ayar kaydı hatası debug'da yutuluyordu.
            logger.warning("Çıkışta ayar kaydı atlandı: %s", exc, exc_info=True)
        _stop_quietly(getattr(self, "watchdog", None), "stop", "Watchdog stop")
        _stop_quietly(getattr(self, "worker", None), "stop", "Worker stop")
        worker = getattr(self, "worker", None)
        if worker is not None and getattr(worker, "isRunning", lambda: False)():
            try:
                worker.requestInterruption()
            except Exception as exc:
                # sweep Y-4: kapanış hataları debug'da yutuluyordu.
                logger.warning("Worker interruption hatası: %s", exc, exc_info=True)
            # F-1194: güvensiz terminate() + yok sayılan bekleme yerine önce
            # kibar bekleme. P2-8 (K-17): `terminate()` ani öldürmedir,
            # kilit/kaynak bırakır — son çare olarak bile kullanılmaz;
            # durmayan worker ikinci grace penceresi + hata loguyla
            # bırakılır, thread referansı korunur.
            try:
                stopped = worker.wait(4000)
            except Exception as exc:
                stopped = False
                logger.warning("Worker bekleme hatası: %s", exc, exc_info=True)
            if not stopped and getattr(worker, "isRunning", lambda: False)():
                logger.warning(
                    "Worker 4 sn'de durmadı; 4 sn daha bekleniyor (terminate yok, P2-8)."
                )
                try:
                    stopped = worker.wait(4000)
                except Exception as exc:
                    stopped = False
                    logger.warning("Worker bekleme hatası: %s", exc, exc_info=True)
            if not stopped and getattr(worker, "isRunning", lambda: False)():
                logger.error(
                    "Worker kapanışta durmadı; terminate() uygulanmadı (P2-8)."
                )
        # F-1194: yakalama/girdi kapatılmadan çıkılırdı.
        _stop_quietly(getattr(self, "capture", None), "close", "Capture close")
        _stop_quietly(getattr(self, "dxg_capture", None), "close", "DXGI capture close")
        _stop_quietly(getattr(self, "input_ctrl", None), "close", "Input controller close")
        _stop_quietly(getattr(self, "mobile_server", None), "stop", "Mobile server stop")
        _stop_quietly(getattr(self, "tunnel_mgr", None), "stop", "Tunnel manager stop")
        _stop_quietly(getattr(self, "telemetry", None), "save_session_report", "Telemetry save")
        _stop_quietly(getattr(self, "tray", None), "hide", "Tray hide")
        self.app.quit()

    def get_main_diagnostic_snapshot(self) -> dict:
        """Returns diagnostic snapshot for main module.

        S-0204: "ready" artık sabit True değil; modül sözleşmesinin gerçek
        probuna (validate_main_contract) bağlıdır.
        """
        return {"module": "main", "status": "ok", "ready": validate_main_contract()}

    def get_main_capabilities(self) -> list:
        """Returns supported capabilities for main module."""
        return list(_MAIN_CAPABILITIES)


def main() -> int:
    # [STARTUP] Python import süresi (modül düzeyi _IMPORT_T0'dan beri).
    logger.info("[STARTUP] Python importları: %.1f sn", time.monotonic() - _IMPORT_T0)
    # v20/bot-ani-kapanma (11 Kasim 17:33:54 sessiz olum): faulthandler +
    # excepthook + atexit + RSS izleme app.exec() oncesi kurulur; bir daha
    # ani kapanmada sebep logs/crash_*.log / faulthandler.log /
    # exit_trace.log uclusunden okunur. Kurulum hatasi botu durdurmaz.
    try:
        install_crash_evidence(LOG_DIR, logger)
    except Exception as exc:
        logger.warning("Crash kanit altyapisi kurulamadi: %s", exc)
    # sweep-26 F8: F: yoksa _pin_bot_env sessizce çıkıyordu; uyarı burada
    # logger kurulduktan sonra görünür olur.
    if _BOOT_HOME_WARNING:
        logger.warning("%s", _BOOT_HOME_WARNING)

    if "--elevate" in sys.argv:
        # F-1139: dönüş değeri yok sayılırdı; yükseltme hem yeni hem eski
        # süreci çalışır bırakırdı. exit_on_elevate ile yükselen süreç
        # başlatılınca bu süreç çıkar; yükseltme reddedilirse bot
        # yükseltmesiz başlatılmaz.
        try:
            elevated = ensure_admin_elevation(dry_run=False, exit_on_elevate=True)
        except SystemExit:
            raise
        except Exception as exc:
            logger.error("Yönetici yükseltmesi sırasında hata: %s", exc)
            return 1
        if not elevated:
            logger.error(
                "Yönetici yükseltmesi başarısız/reddedildi; --elevate ile "
                "bot yükseltmesiz başlatılmıyor."
            )
            print("Yönetici yükseltmesi başarısız oldu; bot başlatılmadı.", file=sys.stderr)
            return 1

    # sweep-26 F1: tek instance kilidi. İkinci örnek burada durur — çift
    # bot, çift tuş ve ayar dosyası çakışması olmaz. pythonw altında stderr
    # görünmez; kullanıcıya mesaj kutusu + oturum log dosyası ile bildirilir.
    if not _acquire_single_instance_lock():
        _dup_msg = (
            "Gatherion zaten çalışıyor — ikinci örnek başlatılmadı. "
            "Çalışan botu kapatıp tekrar deneyin."
        )
        logger.error(_dup_msg)
        print(_dup_msg, file=sys.stderr)
        _show_startup_alert(_dup_msg, "Gatherion zaten çalışıyor")
        return 2

    # QWebEngine / YOLO factory: must be set before QApplication is created.
    try:
        from PyQt6.QtCore import Qt

        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    except Exception as exc:
        logger.debug("AA_ShareOpenGLContexts ayarlanamadı: %s", exc)

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    # F-1195: orchestrator kurulumu korumasızdı; init hatası çıplak
    # traceback ile çökerdi.
    try:
        orchestrator = GatherionOrchestrator(app)
    except Exception as exc:
        logger.error("Gatherion başlatılamadı (orchestrator kurulum hatası): %s", exc)
        print(f"Gatherion başlatılamadı: {exc}", file=sys.stderr)
        print(f"Detay için {os.path.basename(LOG_FILE)} dosyasına bakın.", file=sys.stderr)
        # sweep-26 F9: pythonw altında stderr görünmez — hata mesaj
        # kutusuyla da gösterilir.
        _show_startup_alert(
            f"Gatherion başlatılamadı: {exc}\nDetay için {os.path.basename(LOG_FILE)} dosyasına bakın.",
            "Gatherion başlatılamadı",
        )
        return 1
    mobile_url = orchestrator.get_mobile_dashboard_url()

    if "--self-test" in sys.argv:
        QTimer.singleShot(800, orchestrator.shutdown)

    print("\n" + "=" * 72)
    print(f"  GATHERION-LOCAL v{__version__} (TÜRKÇE) BAŞLATILDI!")
    print(f"  [{orchestrator._menu_chord_label}]  -> Soldaki menüyü aç/kapa (yürüyüşü duraklatmaz)")
    print(f"  [{orchestrator._menu_chord_label} + Enter] -> Aktif Modda Devam Et (Resume)")
    print(f"  [{orchestrator._node_chord_label}]  -> Mevcut Konumu Rota Noktası Olarak Kaydet")
    print(f"  [{orchestrator._pause_chord_label}] -> Duraklat / Devam")
    print("  [3x Toplama Modu] -> Her Noktada 3 Kademeli Hasat & Minimap Oto-Kayıt")
    print(f"  [Mobil Kontrol]   -> {mobile_url}" if mobile_url else f"  [Mobil Kontrol]   -> ÇALIŞMIYOR (mobil sunucu başlatılamadı; ayrıntı: {os.path.basename(LOG_FILE)})")
    print("=" * 72 + "\n")

    try:
        return app.exec()
    finally:
        _release_single_instance_lock()


if __name__ == "__main__":
    sys.exit(main())


# S-0204: modül düzeyi capability listesi — daha önce yalnızca sınıf
# metodu vardı; modül düzeyi validate_main_contract() / baseline
# fonksiyonları tanımsız ada başvurup NameError veriyordu.
_MAIN_CAPABILITIES = ["main", "diagnostic_snapshot", "version_info"]


def get_main_capabilities() -> list:
    """Returns supported capabilities for main module (module level)."""
    return list(_MAIN_CAPABILITIES)


def _get_live_orchestrator():
    """Returns the live GatherionOrchestrator instance, or None.

    S-0204: modül düzeyi prob fonksiyonlarının gerçek bileşen durumuna
    delege edebilmesi için zayıf referans üzerinden çözülür.
    """
    ref = _LIVE_ORCHESTRATOR_REF
    if ref is None:
        return None
    try:
        return ref()
    except Exception:
        return None


def validate_main_contract() -> bool:
    """Validates module contract for main.py.

    S-0204 (Karar 3): totoloji değil gerçek prob — capability listesinin
    boş olmaması yetmez; GatherionOrchestrator sınıfının sözleşmeye dahil
    arayüzü (sinyal bağlama, kapatma, sağlık özeti, capability) yerinde mi
    kontrol edilir. Arayüz eksikse False döner.
    """
    try:
        caps = get_main_capabilities()
        if not caps:
            return False
        required = (
            "_connect_signals",
            "_connect_map_viewer_signals",
            "_connect_yolo_factory_signals",
            "shutdown",
            "get_main_controller_health_summary",
            "get_main_controller_ready_flag",
            "get_main_diagnostic_snapshot",
            "get_main_capabilities",
        )
        return all(callable(getattr(GatherionOrchestrator, name, None)) for name in required)
    except Exception as exc:
        logger.debug("main contract probu başarısız: %s", exc)
        return False


def get_main_baseline_metrics() -> dict:
    """Returns baseline metrics for main.py."""
    return {"module": "main", "contract_valid": validate_main_contract(), "capability_count": len(get_main_capabilities())}


def get_main_readiness_status() -> dict:
    """Returns readiness status for main.py.

    S-0204 (Karar 3): canlı orchestrator kaydı varsa gerçek alt sistem
    sağlığı (get_main_controller_health_summary) probalanır; kayıt yoksa
    yalnızca modül sözleşmesi probalanır ve "live": False olarak
    işaretlenir — çalışmayan bir örneğin sağlığı "hazır" sayılmaz.
    """
    contract_valid = validate_main_contract()
    live = _get_live_orchestrator()
    if live is None:
        return {
            "module": "main",
            "ready": contract_valid,
            "live": False,
            "metrics": get_main_baseline_metrics(),
        }
    try:
        health = live.get_main_controller_health_summary()
        ready = bool(contract_valid and health.get("controller_ready"))
        return {
            "module": "main",
            "ready": ready,
            "live": True,
            "checks": health.get("checks", {}),
            "metrics": get_main_baseline_metrics(),
        }
    except Exception as exc:
        logger.debug("Canlı readiness probu başarısız: %s", exc)
        return {
            "module": "main",
            "ready": False,
            "live": True,
            "error": str(exc),
            "metrics": get_main_baseline_metrics(),
        }


def get_main_audit_record() -> dict:
    """Returns audit record for main.py."""
    return {"module": "main", "audit_passed": validate_main_contract(), "readiness": get_main_readiness_status()}


def get_main_operational_digest() -> dict:
    """Returns operational digest for main.py."""
    return {"module": "main", "operational": validate_main_contract(), "audit": get_main_audit_record()}


def get_main_telemetry_payload() -> dict:
    """Returns telemetry payload for main.py."""
    return {"module": "main", "telemetry_ready": validate_main_contract(), "digest": get_main_operational_digest()}


def get_main_integrity_seal() -> dict:
    """Returns integrity seal for main.py."""
    return {"module": "main", "sealed": validate_main_contract(), "telemetry": get_main_telemetry_payload()}


def get_main_compliance_certificate() -> dict:
    """Returns compliance certificate for main.py."""
    return {"module": "main", "certified": validate_main_contract(), "seal": get_main_integrity_seal()}


def get_main_readiness_probe() -> dict:
    """Returns readiness probe for main.py."""
    return {"module": "main", "ready": validate_main_contract(), "certificate": get_main_compliance_certificate()}


def get_main_audit_summary() -> dict:
    """Returns audit summary for main.py."""
    return {"module": "main", "audit_passed": validate_main_contract(), "probe": get_main_readiness_probe()}


def get_main_heartbeat_snapshot() -> dict:
    """Returns heartbeat snapshot for main.py."""
    return {"module": "main", "alive": validate_main_contract(), "audit": get_main_audit_summary()}


