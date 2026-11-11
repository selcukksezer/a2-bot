# -*- coding: utf-8 -*-
"""AION 2 süreç araması. Eğitim bu süreç varken başlamaz.

RTX 3070 8 GB kartta eğitim ile 2560x1440 oyun aynı GPU'da
DXGI_ERROR_DEVICE_REMOVED üretti. Batch 4, BELOW_NORMAL ve EcoQoS
oyunu kurtarmadı. RTXTrainerEngine oyun açıkken eğitimi reddeder.
Oyun kapandıysa eğitim normal batch ile F: üzerinde çalışabilir.

Politika (F-0100): Bu dosyadaki öncelik ve EcoQoS yardımcıları artık
eğitimle GPU paylaşmak için kullanılmaz. GameGpuThrottle oyunu ASLA
kısmaz; yalnızca oyun sürecini tespit eder. Oyun açıkken eğitim
reddedilir (kısılıp batch 4 ile sürdürülmez); FPS sınırı ve nvidia-smi
saat kilidi yazılmaz.
"""

from __future__ import annotations

import ctypes
import logging
import os
from ctypes import wintypes
from typing import Callable, List, Optional, Sequence

logger = logging.getLogger("Gatherion.GameGpuThrottle")

BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
PROCESS_SET_INFORMATION = 0x0200
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
PROCESS_POWER_THROTTLING = 4
PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
TRAIN_BATCH_WHEN_GAME_PRESENT = 4  # Miras sabit (F-0100): oyun açıkken
# eğitim batch düşürüp sürdürülmez, reddedilir; capped_train_batch artık
# bu sabiti kullanmaz.


class _PROCESSENTRY32W(ctypes.Structure):
    """Shared Toolhelp32 process-entry struct (D25: was defined identically
    in _snapshot_pid_to_exe_map and _pids_from_process_names). Windows-only
    use; defining the layout at import is pure ctypes and safe on Linux."""

    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.c_size_t),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", wintypes.WCHAR * 260),
    ]

# core/watchdog.py pencere başlıkları + core/dpi_focus_guard.py "aion" / "ncsoft".
# Exe adı repoda yok. Aion2 ve NCSOFT altındaki Aion süreçleri eşleşir.
_WINDOW_TITLE_EXACT = frozenset({"aion 2", "aion2", "aion"})
_SKIP_EXE = frozenset(
    {
        "python.exe",
        "pythonw.exe",
        "python3.exe",
        "py.exe",
        "cursor.exe",
        "code.exe",
        "powershell.exe",
        "pwsh.exe",
        "cmd.exe",
        "wt.exe",
        "conhost.exe",
        "windowsterminal.exe",
        "explorer.exe",
        "chrome.exe",
        "msedge.exe",
        "firefox.exe",
        "brave.exe",
        "opera.exe",
        "opera_gx.exe",
        "vivaldi.exe",
        "arc.exe",
        "iexplore.exe",
        "pycharm.exe",
        "pycharm64.exe",
        "idea.exe",
        "idea64.exe",
        "clion.exe",
        "clion64.exe",
        "rider.exe",
        "rider64.exe",
        "webstorm.exe",
        "webstorm64.exe",
        "devenv.exe",
        "sublime_text.exe",
        "notepad.exe",
        "notepad++.exe",
        "taskmgr.exe",
    }
)

_EXCLUDED_STEMS = frozenset(
    {
        "chrome",
        "msedge",
        "edge",
        "firefox",
        "brave",
        "opera",
        "vivaldi",
        "arc",
        "iexplore",
        "code",
        "cursor",
        "pycharm",
        "idea",
        "clion",
        "rider",
        "webstorm",
        "devenv",
        "sublime_text",
        "notepad",
        "explorer",
        "powershell",
        "pwsh",
        "cmd",
        "wt",
        "conhost",
        "windowsterminal",
        "taskmgr",
        "python",
        "pythonw",
        "python3",
        "py",
    }
)

_SKIP_TITLE_SUBSTRINGS = frozenset(
    {
        "google chrome",
        "chrome",
        "microsoft edge",
        "edge",
        "firefox",
        "brave",
        "opera",
        "vivaldi",
        "visual studio code",
        "visual studio",
        "cursor",
        "pycharm",
        "sublime text",
        "windows powershell",
        "powershell",
        "command prompt",
        "cmd.exe",
        "terminal",
        "dosya gezgini",
        "file explorer",
        "explorer",
    }
)


def _basename_windows(path: str) -> str:
    """Dosya adını Windows ayraçlarıyla da böler (F-0273: Linux'ta
    os.path.basename ters bölü içeren Windows yolunu bölemez; exe adı
    yanlış çıkar ve hariç tutulan exe'ler oyun sanılır)."""
    return (path or "").replace("\\", "/").rsplit("/", 1)[-1]


def is_excluded_exe(image_name: str) -> bool:
    """True if image_name belongs to an excluded browser, IDE, shell, explorer, or python."""
    base = _basename_windows(image_name).casefold()
    if not base:
        return True
    if base in _SKIP_EXE:
        return True
    stem = os.path.splitext(base)[0]
    if stem in _EXCLUDED_STEMS:
        return True
    for stem_prefix in _EXCLUDED_STEMS:
        if base.startswith(stem_prefix + ".") or base.startswith(stem_prefix + "-") or base.startswith(stem_prefix + "_"):
            return True
    return False


def training_image_retention_decision(project_root: os.PathLike) -> dict:
    """Sonraki eğitim yolov8n.pt ile data.yaml'ı baştan okur. Kareler silinmez.

    Gerekçe, dosya sistemine bakılarak kurulur (F-0441): data.yaml'ın
    varlığı ile images/labels/game_learning dizinlerindeki dosya
    sayıları gerçekten ölçülür; doğrulanmamış iddialar yazılmaz.
    """
    root = os.fspath(project_root)
    images = os.path.join(root, "dataset", "images")
    labels = os.path.join(root, "dataset", "labels")
    learning = os.path.join(root, "dataset", "game_learning")
    data_yaml = os.path.join(root, "dataset", "data.yaml")

    def _file_count(path: str) -> int:
        try:
            if not os.path.isdir(path):
                return 0
            return sum(
                1
                for name in os.listdir(path)
                if os.path.isfile(os.path.join(path, name))
            )
        except OSError:
            return 0

    yaml_exists = os.path.isfile(data_yaml)
    images_n = _file_count(images)
    labels_n = _file_count(labels)
    learning_n = _file_count(learning)
    observed = (
        f"data.yaml {'bulundu' if yaml_exists else 'bulunamadı'}; "
        f"dataset/images: {images_n} dosya; dataset/labels: {labels_n} dosya; "
        f"dataset/game_learning: {learning_n} dosya"
    )
    return {
        "action": "keep",
        "delete": False,
        "data_yaml": data_yaml,
        "keep_paths": [images, labels, learning],
        "data_yaml_exists": yaml_exists,
        "observed": observed,
        "reason": (
            "Kareler silinmez. "
            + observed
            + ". Sonraki eğitim resume=False ve pretrained yolov8n.pt ile "
            "dataset/images ve dataset/labels üzerinden tam eğitimdir; "
            "game_learning data.yaml içinde yoktur, learner oraya yazmaya "
            "devam eder, o yüzden o da durur."
        ),
    }


def _compact_name(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def title_is_aion_game(title: str) -> bool:
    folded = " ".join((title or "").casefold().split())
    if not folded or "gatherion" in folded:
        return False
    for marker in _SKIP_TITLE_SUBSTRINGS:
        if marker in folded:
            return False
    if folded in _WINDOW_TITLE_EXACT:
        return True
    compact = _compact_name(folded)
    return "aion2" in compact


def exe_is_aion_game(image_name: str) -> bool:
    base = _basename_windows(image_name).casefold()
    if not base or is_excluded_exe(base):
        return False
    compact = _compact_name(os.path.splitext(base)[0])
    if not compact:
        return False
    if "aion2" in compact:
        return True
    if base in ("aion.bin", "aion2.bin", "aion.exe", "aion2.exe"):
        return True
    if compact in ("aion", "aionbin", "aion2bin", "aionclient", "aionclassic"):
        return True
    # F-0273: yol-alt-dize eşleşmesi ("ncsoft" ve "aion" yolun herhangi
    # bir yerinde) yanlış pozitif üretir (örn. yedek/rehber klasörü).
    # Artık dizin bileşeni bazında bakılır: bir dizin tam olarak
    # "ncsoft" VE bir dizin tam olarak "aion"/"aion2" (veya exe adı
    # "aion" içeriyor) olmalı.
    full = (image_name or "").replace("\\", "/").casefold()
    parts = [p for p in full.split("/") if p]
    dirs = parts[:-1]
    has_ncsoft_dir = any(p == "ncsoft" for p in dirs)
    has_aion_dir = any(p in ("aion", "aion2") for p in dirs)
    return bool(has_ncsoft_dir and (has_aion_dir or "aion" in compact))


def is_game_running() -> bool:
    """True only when a verified AION 2 game process is currently active.

    Tarama patlarsa fail-closed: oyun açık varsayılır (F-0101), eğitim
    başlamaz.
    """
    try:
        return bool(find_aion_game_processes())
    except Exception as exc:
        logger.warning("AION süreç araması başarısız; oyun açık varsayılıyor: %s", exc)
        return True


def find_aion_game_processes() -> List[int]:
    """AION penceresi veya Aion2 exe pid listesi. Oyun yoksa boş liste.

    Tarama adımlarından biri patlarsa hata yutulmaz (F-0101): çağıran
    oyunu "yok" sanıp eğitimi başlatmasın diye RuntimeError yükselir;
    is_game_running() bunu fail-closed (True) yorumlar.
    """
    if os.name != "nt":
        return []
    found: List[int] = []
    own = os.getpid()
    try:
        found.extend(_pids_from_window_titles())
    except Exception as exc:
        logger.error("AION pencere taraması başarısız: %s", exc)
        raise RuntimeError(f"AION window scan failed: {exc}") from exc
    try:
        found.extend(_pids_from_process_names())
    except Exception as exc:
        logger.error("AION süreç taraması başarısız: %s", exc)
        raise RuntimeError(f"AION process scan failed: {exc}") from exc
    unique: List[int] = []
    seen = set()
    for pid in found:
        pid_i = int(pid)
        if pid_i <= 0 or pid_i == own or pid_i in seen:
            continue
        exe_name = _get_exe_for_pid(pid_i)
        base = _basename_windows(exe_name).casefold()
        if base and (is_excluded_exe(base) or not exe_is_aion_game(exe_name)):
            continue
        seen.add(pid_i)
        unique.append(pid_i)
    return unique


class _ProcSnap:
    def __init__(self, pid: int) -> None:
        self.pid = int(pid)
        self.handle: Optional[int] = None
        self.close_handle = False
        self.old_priority: Optional[int] = None
        self.priority_changed = False
        self.old_throttle_state: Optional[int] = None
        self.throttle_known = False
        self.throttle_changed = False


class GameGpuThrottle:
    """Oyun sürecini tespit eder; oyunu ASLA kısmaz (F-0100).

    Politika: oyun açıkken eğitim reddedilir, oyun BELOW_NORMAL /
    EcoQoS ile kısılıp batch 4 ile sürdürülmez. apply() yalnızca
    tespit yapar; restore() durumu sıfırlar. _apply_to_pid /
    _restore_snap yalnızca geriye dönük uyumluluk için durur ve
    apply() tarafından çağrılmaz.
    """

    def __init__(
        self,
        finder: Optional[Callable[[], Sequence[int]]] = None,
    ) -> None:
        self._finder = finder or find_aion_game_processes
        self._snaps: List[_ProcSnap] = []
        self.game_found = False
        self.applied = False
        self.pids: List[int] = []

    def __enter__(self) -> "GameGpuThrottle":
        self.apply()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.restore()
        return False

    def apply(self) -> None:
        # F-0442: önceki çalışmadan kalan snap/pid durumu temizlenir.
        self._snaps = []
        self.game_found = False
        self.applied = False
        self.pids = []
        try:
            pids = [int(p) for p in self._finder()]
        except Exception as exc:
            # F-0101: arama patlarsa oyun "yok" sayılıp eğitim
            # başlamasın — oyun açık varsayılır.
            logger.warning("AION süreç araması başarısız; oyun açık varsayılıyor, eğitim başlamamalı: %s", exc)
            self.game_found = True
            self.pids = []
            return
        self.pids = pids
        self.game_found = bool(pids)
        if not pids:
            logger.info("AION süreci yok. Eğitim normal batch ile sürer.")
            return
        # F-0100/F-0274: kısma uygulanmaz; log yalnızca gerçeği söyler.
        # "öncelik BELOW_NORMAL, EcoQoS açık / eğitim GPU'da kalır,
        # batch 4" iddiası artık yazılmaz.
        logger.warning(
            "AION pid=%s açık: öncelik/EcoQoS kısması uygulanmaz; "
            "eğitim reddedilmelidir.",
            ",".join(str(p) for p in pids),
        )

    def restore(self) -> None:
        while self._snaps:
            snap = self._snaps.pop()
            _restore_snap(snap)
        self.applied = False
        # F-0442: restore sonrası bayat game_found/pids kalmaz.
        self.game_found = False
        self.pids = []


def capped_train_batch(requested_batch: int, game_found: bool) -> int:
    """Oyun kapalıyken istenen batch döner.

    Oyun açıkken batch 4'e düşürüp eğitimi sürdürmek politika dışıdır
    (F-0100): eğitim reddedilir, bu yüzden RuntimeError yükselir.
    """
    requested = max(1, int(requested_batch))
    if game_found:
        raise RuntimeError(
            "AION 2 açıkken eğitim reddedilir; batch düşürüp GPU paylaşılmaz"
        )
    return requested


def _win32():
    import ctypes
    from ctypes import wintypes

    return ctypes, wintypes


def _snapshot_pid_to_exe_map() -> dict[int, str]:
    if os.name != "nt":
        return {}
    ctypes, wintypes = _win32()
    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPPROCESS = 0x00000002

    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot is None or int(ctypes.cast(snapshot, ctypes.c_void_p).value or 0) in (0, invalid):
        return {}
    pid_map: dict[int, str] = {}
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            pid_map[int(entry.th32ProcessID)] = entry.szExeFile
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return pid_map


def _get_exe_for_pid(pid: int, snapshot_map: Optional[dict[int, str]] = None) -> str:
    """Returns the full image path or executable name for a given process ID."""
    if os.name != "nt" or pid <= 0:
        return ""
    try:
        ctypes, wintypes = _win32()
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if handle:
            try:
                buf = ctypes.create_unicode_buffer(1024)
                sz = wintypes.DWORD(1024)
                if hasattr(kernel32, "QueryFullProcessImageNameW"):
                    kernel32.QueryFullProcessImageNameW.argtypes = [
                        wintypes.HANDLE,
                        wintypes.DWORD,
                        wintypes.LPWSTR,
                        ctypes.POINTER(wintypes.DWORD),
                    ]
                    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
                    if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(sz)):
                        if buf.value:
                            return buf.value
            finally:
                kernel32.CloseHandle(handle)
    except Exception:
        pass
    if snapshot_map and pid in snapshot_map:
        return snapshot_map[pid]
    try:
        s_map = _snapshot_pid_to_exe_map()
        return s_map.get(pid, "")
    except Exception:
        return ""


def _pids_from_window_titles() -> List[int]:
    if os.name != "nt":
        return []
    ctypes, wintypes = _win32()
    user32 = ctypes.windll.user32
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    if hasattr(user32, "IsIconic"):
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.IsIconic.restype = wintypes.BOOL

    snapshot_map = _snapshot_pid_to_exe_map()
    found: List[int] = []
    own_pid = os.getpid()

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _callback(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        # F-0275: simge durumuna düşürülmüş (minimized) oyun penceresi
        # atlanmaz; başlığı okunabilir ve süreç taramasıyla doğrulanır.

        length = int(user32.GetWindowTextLengthW(hwnd) or 0)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value

        if title_is_aion_game(title):
            pid = wintypes.DWORD(0)
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pid_val = int(pid.value)
            if pid_val and pid_val != own_pid:
                exe_name = _get_exe_for_pid(pid_val, snapshot_map)
                base = _basename_windows(exe_name).casefold()
                if base and not is_excluded_exe(base) and exe_is_aion_game(exe_name):
                    found.append(pid_val)
        return True

    user32.EnumWindows(_callback, 0)
    if not found:
        try:
            if hasattr(user32, "OpenInputDesktop") and hasattr(user32, "EnumDesktopWindows"):
                user32.OpenInputDesktop.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
                user32.OpenInputDesktop.restype = wintypes.HANDLE
                user32.CloseDesktop.argtypes = [wintypes.HANDLE]
                user32.CloseDesktop.restype = wintypes.BOOL
                DESKENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
                user32.EnumDesktopWindows.argtypes = [wintypes.HANDLE, DESKENUMPROC, wintypes.LPARAM]
                user32.EnumDesktopWindows.restype = wintypes.BOOL

                desk = user32.OpenInputDesktop(0, False, 0x0100)
                if desk:
                    try:
                        user32.EnumDesktopWindows(desk, DESKENUMPROC(_callback), 0)
                    finally:
                        user32.CloseDesktop(desk)
        except Exception:
            pass

    return found


def _pids_from_process_names() -> List[int]:
    # F-0443: kardeş fonksiyonlardaki os.name != "nt" koruması burada
    # da var; Windows dışı platformda ctypes.windll'e dokunulmaz.
    if os.name != "nt":
        return []
    ctypes, wintypes = _win32()
    kernel32 = ctypes.windll.kernel32
    TH32CS_SNAPPROCESS = 0x00000002

    kernel32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.Process32FirstW.restype = wintypes.BOOL
    kernel32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PROCESSENTRY32W)]
    kernel32.Process32NextW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    invalid = ctypes.c_void_p(-1).value
    if snapshot is None or int(ctypes.cast(snapshot, ctypes.c_void_p).value or 0) in (0, invalid):
        return []
    found: List[int] = []
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(_PROCESSENTRY32W)
        ok = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while ok:
            exe_name = entry.szExeFile
            if not is_excluded_exe(exe_name) and exe_is_aion_game(exe_name):
                found.append(int(entry.th32ProcessID))
            ok = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel32.CloseHandle(snapshot)
    return found



def _bind_priority_apis():
    ctypes, wintypes = _win32()
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    kernel32.GetPriorityClass.restype = wintypes.DWORD
    kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel32.SetPriorityClass.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.SetProcessInformation.restype = wintypes.BOOL
    kernel32.GetProcessInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.GetProcessInformation.restype = wintypes.BOOL
    return ctypes, wintypes, kernel32


def _power_state(state_mask: int):
    ctypes, wintypes = _win32()

    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    state = PROCESS_POWER_THROTTLING_STATE()
    state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION
    state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
    state.StateMask = int(state_mask)
    return state


def _read_throttle_state(kernel32, handle) -> Optional[int]:
    state = _power_state(0)
    ok = kernel32.GetProcessInformation(
        handle,
        PROCESS_POWER_THROTTLING,
        ctypes_address(state),
        ctypes_sizeof(state),
    )
    if not ok:
        return None
    return int(state.StateMask)


def ctypes_address(state):
    import ctypes

    return ctypes.byref(state)


def ctypes_sizeof(state):
    import ctypes

    return ctypes.sizeof(state)


def _set_throttle_state(kernel32, handle, state_mask: int) -> bool:
    state = _power_state(state_mask)
    return bool(
        kernel32.SetProcessInformation(
            handle,
            PROCESS_POWER_THROTTLING,
            ctypes_address(state),
            ctypes_sizeof(state),
        )
    )


def _apply_to_pid(pid: int) -> _ProcSnap:
    snap = _ProcSnap(pid)
    if os.name != "nt":
        return snap
    ctypes, wintypes, kernel32 = _bind_priority_apis()
    access = PROCESS_SET_INFORMATION | PROCESS_QUERY_INFORMATION | PROCESS_QUERY_LIMITED_INFORMATION
    if int(pid) == os.getpid():
        handle = kernel32.GetCurrentProcess()
        snap.close_handle = False
    else:
        handle = kernel32.OpenProcess(access, False, int(pid))
        snap.close_handle = True
    if not handle:
        logger.warning("AION pid=%s açılamadı. Öncelik değişmedi.", pid)
        snap.close_handle = False
        return snap
    snap.handle = handle
    try:
        old_priority = int(kernel32.GetPriorityClass(handle) or 0)
        snap.old_priority = old_priority or None
        if old_priority and old_priority != BELOW_NORMAL_PRIORITY_CLASS:
            if kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS):
                snap.priority_changed = True
        previous = _read_throttle_state(kernel32, handle)
        if previous is None:
            snap.throttle_known = False
        else:
            snap.throttle_known = True
            snap.old_throttle_state = int(previous)
        desired = PROCESS_POWER_THROTTLING_EXECUTION_SPEED
        if snap.old_throttle_state != desired and _set_throttle_state(kernel32, handle, desired):
            snap.throttle_changed = True
    except Exception as exc:
        logger.warning("AION pid=%s kısılırken hata: %s", pid, exc)
    return snap


def _restore_snap(snap: _ProcSnap) -> None:
    if os.name != "nt" or not snap.handle:
        return
    _ctypes, _wintypes, kernel32 = _bind_priority_apis()
    try:
        if snap.priority_changed and snap.old_priority:
            kernel32.SetPriorityClass(snap.handle, int(snap.old_priority))
        if snap.throttle_changed:
            if snap.throttle_known and snap.old_throttle_state is not None:
                _set_throttle_state(kernel32, snap.handle, int(snap.old_throttle_state))
            else:
                _set_throttle_state(kernel32, snap.handle, 0)
    except Exception as exc:
        logger.warning("AION pid=%s geri alınamadı: %s", snap.pid, exc)
    finally:
        if snap.close_handle:
            try:
                kernel32.CloseHandle(snap.handle)
            except Exception:
                pass
        snap.handle = None
        snap.priority_changed = False
        snap.throttle_changed = False


def current_process_priority() -> int:
    if os.name != "nt":
        return 0
    _ctypes, _wintypes, kernel32 = _bind_priority_apis()
    return int(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()) or 0)
