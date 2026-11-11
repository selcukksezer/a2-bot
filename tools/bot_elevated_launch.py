# -*- coding: utf-8 -*-
"""Launch Gatherion at High integrity without per-start UAC.

Why this exists
---------------
UIPI blocks Medium→High SendInput. When AION2 is High, Gatherion must also be
High or Alt/skills/MP input is dropped.

baslat.bat uses Verb=runas every cold start → interactive UAC each time.
Fix: register a Task Scheduler action with RunLevel=Highest once (one UAC at
register time). After that, `schtasks /Run` starts High Gatherion with no prompt.

Mirrors tools/steam_unelevated_launch.py (Gatherion_SteamMedium / Limited),
but for the bot: Gatherion_BotHigh / Highest.

Usage
-----
  python tools/bot_elevated_launch.py --status
  python tools/bot_elevated_launch.py --register   # one UAC if not already admin
  python tools/bot_elevated_launch.py --start      # schtasks /Run, no UAC
  python tools/bot_elevated_launch.py --restart    # kill stale + /Run
  python tools/bot_elevated_launch.py              # status→start; register if missing
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_NAME = "Gatherion_BotHigh"
MAIN_PY = PROJECT_ROOT / "main.py"

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
TOKEN_QUERY = 0x0008
TokenElevation = 20
TokenIntegrityLevel = 25
RID_NAMES = {
    0x1000: "Low",
    0x2000: "Medium",
    0x2100: "MediumPlus",
    0x3000: "High",
    0x4000: "System",
}

kernel32 = None
advapi32 = None
shell32 = None
OpenProcess = None
CloseHandle = None
OpenProcessToken = None
GetTokenInformation = None
GetSidSubAuthorityCount = None
GetSidSubAuthority = None


def _load_win32() -> bool:
    """Load kernel32/advapi32/shell32 and bind the token-probe functions.

    Windows only. Mirrors tools/steam_unelevated_launch.py (F-1285): the
    bindings used to live at import time, so merely importing this module
    crashed on non-Windows (AttributeError: ctypes has no WinDLL) and no
    --status/--json diagnostic could run there. Now the module imports
    anywhere; token probing reports "unavailable" instead of crashing.
    """
    global kernel32, advapi32, shell32
    global OpenProcess, CloseHandle, OpenProcessToken
    global GetTokenInformation, GetSidSubAuthorityCount, GetSidSubAuthority
    if kernel32 is not None and advapi32 is not None and shell32 is not None:
        return True
    if sys.platform != "win32":
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    except Exception:
        return False
    OpenProcess = kernel32.OpenProcess
    OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    OpenProcess.restype = wintypes.HANDLE
    CloseHandle = kernel32.CloseHandle
    OpenProcessToken = advapi32.OpenProcessToken
    OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    OpenProcessToken.restype = wintypes.BOOL
    GetTokenInformation = advapi32.GetTokenInformation
    GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    GetTokenInformation.restype = wintypes.BOOL
    GetSidSubAuthorityCount = advapi32.GetSidSubAuthorityCount
    GetSidSubAuthorityCount.argtypes = [ctypes.c_void_p]
    GetSidSubAuthorityCount.restype = ctypes.POINTER(ctypes.c_ubyte)
    GetSidSubAuthority = advapi32.GetSidSubAuthority
    GetSidSubAuthority.argtypes = [ctypes.c_void_p, wintypes.DWORD]
    GetSidSubAuthority.restype = ctypes.POINTER(wintypes.DWORD)
    return True


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


def is_admin() -> bool:
    if not _load_win32():
        return False
    try:
        return bool(shell32.IsUserAnAdmin())
    except Exception:
        return False


def probe_pid(pid: int) -> dict:
    if not _load_win32():
        return {"pid": pid, "ok": False, "error": "Windows-only token probe unavailable on this platform"}
    h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return {"pid": pid, "ok": False, "error": f"OpenProcess={ctypes.get_last_error()}"}
    try:
        tok = wintypes.HANDLE()
        if not OpenProcessToken(h, TOKEN_QUERY, ctypes.byref(tok)):
            return {"pid": pid, "ok": False, "error": f"OpenProcessToken={ctypes.get_last_error()}"}
        try:
            elev = wintypes.DWORD(0)
            ret = wintypes.DWORD(0)
            ok = GetTokenInformation(
                tok, TokenElevation, ctypes.byref(elev), ctypes.sizeof(elev), ctypes.byref(ret)
            )
            elevated = bool(elev.value) if ok else None
            needed = wintypes.DWORD(0)
            GetTokenInformation(tok, TokenIntegrityLevel, None, 0, ctypes.byref(needed))
            buf = (ctypes.c_char * needed.value)()
            if not GetTokenInformation(tok, TokenIntegrityLevel, buf, needed.value, ctypes.byref(needed)):
                return {
                    "pid": pid,
                    "ok": False,
                    "elevated": elevated,
                    "error": f"IL={ctypes.get_last_error()}",
                }
            tml = ctypes.cast(buf, ctypes.POINTER(TOKEN_MANDATORY_LABEL)).contents
            sid = tml.Label.Sid
            count = GetSidSubAuthorityCount(sid).contents.value
            # S295 (v19/P3-8): count == 0 ise count-1 == -1 DWORD'a
            # sarılır, NULL .contents ValueError fırlatır. Gerçek
            # SID'lerde count >= 1'dir; 0 gelirse hata dön.
            if count <= 0:
                return {
                    "pid": pid,
                    "ok": False,
                    "elevated": elevated,
                    "error": f"IL: SID sub-authority count={count}",
                }
            rid = GetSidSubAuthority(sid, count - 1).contents.value
            return {
                "pid": pid,
                "ok": True,
                "elevated": elevated,
                "integrity": RID_NAMES.get(rid, f"RID=0x{rid:X}"),
                "rid": rid,
            }
        finally:
            CloseHandle(tok)
    finally:
        CloseHandle(h)


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _python_install_candidates() -> list[Path]:
    """Discover installed CPython folders instead of one hard-coded user path."""
    bases: list[Path] = []
    for env_name in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        root = os.environ.get(env_name)
        if root:
            bases.append(Path(root) / "Programs" / "Python")
    bases.append(Path.home() / "AppData" / "Local" / "Programs" / "Python")
    found: list[Path] = []
    for base in bases:
        try:
            found.extend(sorted(base.glob("Python3*"), reverse=True))
        except OSError:
            continue
    return found


def resolve_python_pair() -> tuple[Path, Path | None]:
    """Return (python.exe, pythonw.exe | None).

    The second element is None when no pythonw.exe twin exists next to the
    chosen python.exe — callers must fall back explicitly instead of being
    handed python.exe mislabelled as pythonw.exe.
    """
    candidates: list[Path] = []
    for folder in (".venv", "venv"):
        candidates.append(PROJECT_ROOT / folder / "Scripts")
    candidates.extend(_python_install_candidates())
    # Whatever interpreter is running us (e.g. a venv python.exe).
    candidates.append(Path(sys.executable).resolve().parent)
    for base in candidates:
        py = base / "python.exe"
        pyw = base / "pythonw.exe"
        if py.exists():
            return py, (pyw if pyw.exists() else None)
    # Last resort: the running interpreter itself; no pythonw twin.
    cur = Path(sys.executable).resolve()
    if cur.name.lower() == "pythonw.exe":
        twin = cur.with_name("python.exe")
        return (twin if twin.exists() else cur), cur
    return cur, None


def task_exists() -> bool:
    try:
        cp = _run(["schtasks", "/Query", "/TN", TASK_NAME])
    except OSError:
        # schtasks.exe itself missing (non-Windows / broken PATH): the task
        # is not visible, which is exactly what False means. Without this,
        # every caller — status_report(), register_task()'s first check,
        # _register_task_as_admin()'s verification — raised FileNotFoundError
        # instead of reporting "not registered".
        return False
    return cp.returncode == 0


def list_gatherion_pythonw() -> list[dict]:
    """Find pythonw.exe processes whose command line looks like Gatherion main.py."""
    out: list[dict] = []
    root_s = str(PROJECT_ROOT).replace("/", "\\").lower()
    # Only the project tree itself or an explicit Gatherion command line.
    # (A bare '*\a2\*' match used to kill any process whose path merely
    # contained "\a2\" — far too broad for a destructive taskkill.)
    ps = f"""
$ErrorActionPreference='SilentlyContinue'
$root = {_ps_quote(root_s)}
Get-CimInstance Win32_Process -Filter "Name='pythonw.exe'" | ForEach-Object {{
  $cl = $_.CommandLine
  if (-not $cl) {{ return }}
  $cll = $cl.ToLower()
  if ($cll -notlike '*main.py*') {{ return }}
  if ($cll -like ('*' + $root + '*') -or $cll -like '*gatherion*') {{
    [PSCustomObject]@{{ pid=$_.ProcessId; commandLine=$cl }} | ConvertTo-Json -Compress
  }}
}}
"""
    try:
        cp = _run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]
        )
    except OSError:
        # powershell.exe missing (non-Windows): no pythonw processes can be
        # enumerated this way — report none instead of taking down
        # status_report() / list_related_processes() with FileNotFoundError.
        return out
    for line in (cp.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        pid = int(row.get("pid") or 0)
        if pid <= 0:
            continue
        info = probe_pid(pid)
        info["name"] = "pythonw.exe"
        info["commandLine"] = row.get("commandLine")
        out.append(info)
    return out


def list_related_processes() -> dict:
    want = {"steam.exe", "aion2.exe", "explorer.exe", "pythonw.exe", "python.exe"}
    out: list[dict] = []
    try:
        raw = subprocess.check_output(
            ["tasklist", "/FO", "CSV", "/NH"], text=True, errors="replace", timeout=15
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        # A failed tasklist must not take down the whole status query:
        # report the related-process part as unavailable, keep gatherion.
        return {
            "all": out,
            "gatherion": list_gatherion_pythonw(),
            "error": f"tasklist failed: {exc}",
        }
    seen: set[tuple[str, str]] = set()
    for line in raw.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) < 2:
            continue
        name, pid_s = parts[0], parts[1]
        if name.lower() not in want:
            continue
        key = (name.lower(), pid_s)
        if key in seen:
            continue
        seen.add(key)
        try:
            pid = int(pid_s)
        except ValueError:
            continue
        info = probe_pid(pid)
        info["name"] = name
        out.append(info)
    return {"all": out, "gatherion": list_gatherion_pythonw()}


def status_report() -> dict:
    related = list_related_processes()
    bots = related["gatherion"]
    high_bots = [
        p
        for p in bots
        if p.get("ok") and (p.get("integrity") == "High" or p.get("elevated"))
    ]
    aion = [
        p
        for p in related["all"]
        if p.get("name", "").lower() == "aion2.exe"
    ]
    steam = [
        p
        for p in related["all"]
        if p.get("name", "").lower() == "steam.exe"
    ]
    return {
        "task_name": TASK_NAME,
        "task_registered": task_exists(),
        "bot_high_elevated": bool(high_bots),
        "gatherion_bots": bots,
        "aion2": aion,
        "steam": steam,
        "self_admin": is_admin(),
        "hint": (
            "Cift tik BOT_HIGH_NO_UAC.bat — UAC yok (gorev kayitliyken). "
            "Ilk --register bir kez UAC isteyebilir."
        ),
    }


def kill_stale_gatherion_report(*, exclude_pid: int | None = None) -> dict:
    """Kill stale Gatherion pythonw processes; report failures instead of swallowing them."""
    killed: list[int] = []
    failed: list[dict] = []
    me = exclude_pid if exclude_pid is not None else os.getpid()
    for p in list_gatherion_pythonw():
        pid = int(p.get("pid") or 0)
        if pid <= 0 or pid == me:
            continue
        cp = _run(["taskkill", "/PID", str(pid), "/F"])
        if cp.returncode == 0:
            killed.append(pid)
        else:
            detail = (cp.stderr or cp.stdout or f"exit={cp.returncode}").strip()
            failed.append({"pid": pid, "error": detail})
            print(f"WARNING: taskkill /PID {pid} failed: {detail}", file=sys.stderr)
    if killed:
        time.sleep(1.5)
    return {"killed": killed, "failed": failed}


def kill_stale_gatherion(*, exclude_pid: int | None = None) -> list[int]:
    return kill_stale_gatherion_report(exclude_pid=exclude_pid)["killed"]


def ensure_env_for_bot() -> None:
    root = str(PROJECT_ROOT)
    os.environ["GATHERION_HOME"] = root
    os.environ["PYTHONPATH"] = root + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONUTF8"] = "1"
    for name, sub in (
        ("TEMP", "temp"),
        ("TMP", "temp"),
        ("YOLO_CONFIG_DIR", "appdata"),
        ("ULTRALYTICS_CONFIG_DIR", "appdata"),
        ("TORCH_HOME", "torch"),
        ("PIP_CACHE_DIR", "pip-cache"),
    ):
        path = PROJECT_ROOT / sub
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)


def spawn_bot_pythonw() -> dict:
    """Start main.py via pythonw (same entry as baslat.bat :RUN_BOT)."""
    if not MAIN_PY.exists():
        raise FileNotFoundError(f"main.py not found: {MAIN_PY}")
    ensure_env_for_bot()
    _py, pyw = resolve_python_pair()
    if pyw is None:
        # No pythonw.exe twin: run under python.exe explicitly (console
        # window) instead of pretending python.exe *is* pythonw.exe.
        print(
            "WARNING: pythonw.exe not found next to "
            f"{_py}; starting bot with python.exe (console window visible).",
            file=sys.stderr,
        )
        pyw = _py
    creationflags = 0
    if sys.platform.startswith("win"):
        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
        )
    proc = subprocess.Popen(
        [str(pyw), str(MAIN_PY)],
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
        close_fds=True,
        creationflags=creationflags,
    )
    return {"pythonw": str(pyw), "main": str(MAIN_PY), "pid": proc.pid}


def task_entry() -> dict:
    """Scheduled-task body: runs High, kills stale Medium/High twins, starts bot.

    Never reports ok=True on spawn alone — the bot must actually be observed
    alive at High integrity afterwards, otherwise the task scheduler (and the
    operator) would believe a dead bot is running.
    """
    kill_report = kill_stale_gatherion_report()
    spawned = spawn_bot_pythonw()
    after = wait_for_high_bot()
    spawned_pid = int(spawned.get("pid") or 0)
    seen_pids = {int(p.get("pid") or 0) for p in after.get("gatherion_bots", [])}
    alive = bool(after.get("bot_high_elevated")) and spawned_pid in seen_pids
    result: dict = {
        "ok": alive,
        "mode": "task-entry",
        "killed": kill_report["killed"],
        "kill_failures": kill_report["failed"],
        "spawned": spawned,
        "after": after,
    }
    if not alive:
        result["error"] = (
            "Bot spawned (pid="
            f"{spawned_pid}) but was not observed alive at High integrity "
            f"within the wait window (seen_pids={sorted(seen_pids)})."
        )
        if kill_report["failed"]:
            result["error"] += f" Stale-kill failures: {kill_report['failed']}."
    return result


def _register_task_as_admin() -> None:
    """Create/replace Highest interactive task. Caller must already be elevated."""
    if not is_admin():
        raise PermissionError("Registering RunLevel=Highest requires an elevated process.")
    py, _pyw = resolve_python_pair()
    script = Path(__file__).resolve()
    # python.exe (not pythonw) so --task-entry can log briefly; it exits after spawn.
    execute = str(py)
    arguments = f'"{script}" --task-entry'
    _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    ps = f"""
$tn = {_ps_quote(TASK_NAME)}
$action = New-ScheduledTaskAction -Execute {_ps_quote(execute)} -Argument {_ps_quote(arguments)} -WorkingDirectory {_ps_quote(str(PROJECT_ROOT))}
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable
Register-ScheduledTask -TaskName $tn -Action $action -Principal $principal -Settings $settings -Force | Out-Null
"""
    cp = _run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            ps,
        ]
    )
    if cp.returncode != 0 or not task_exists():
        raise RuntimeError(
            "Failed to register Highest scheduled task:\n"
            + (cp.stderr or cp.stdout or f"exit={cp.returncode}")
        )


def register_task(*, force: bool = False, allow_uac: bool = True) -> dict:
    """Ensure Gatherion_BotHigh exists. May show one UAC when elevating to register."""
    existed = task_exists()
    if existed and not force:
        return {
            "ok": True,
            "registered": False,
            "already_present": True,
            "task": TASK_NAME,
            "uac_used": False,
            "message": f"{TASK_NAME} zaten kayitli — UAC yok.",
        }

    if is_admin():
        _register_task_as_admin()
        return {
            "ok": True,
            "registered": True,
            "already_present": existed,
            "task": TASK_NAME,
            "uac_used": False,
            "message": f"{TASK_NAME} kaydedildi (bu surec zaten yonetici).",
        }

    if not allow_uac:
        return {
            "ok": False,
            "registered": False,
            "task": TASK_NAME,
            "uac_used": False,
            "error": (
                f"{TASK_NAME} yok ve kayit icin bir kez yonetici (UAC) gerekli. "
                "BOT_HIGH_NO_UAC.bat --register calistirin ve Evet deyin."
            ),
        }

    if not _load_win32():
        return {
            "ok": False,
            "registered": False,
            "task": TASK_NAME,
            "uac_used": False,
            "error": (
                f"{TASK_NAME} yok ve UAC ile kayit yalnizca Windows'ta "
                "mumkun (bu platformda ShellExecuteW yok)."
            ),
        }

    # One-shot elevated register via ShellExecute runas (single UAC).
    this_py, _ = resolve_python_pair()
    script = str(Path(__file__).resolve())
    params = subprocess.list2cmdline([script, "--register-elevated"])
    ret = int(
        shell32.ShellExecuteW(
            None,
            "runas",
            str(this_py),
            params,
            str(PROJECT_ROOT),
            1,
        )
    )
    if ret <= 32:
        return {
            "ok": False,
            "registered": False,
            "task": TASK_NAME,
            "uac_used": True,
            "error": f"UAC register reddedildi veya basarisiz (ShellExecuteW={ret}).",
        }
    # Wait for elevated sibling to create the task.
    for _ in range(40):
        if task_exists():
            return {
                "ok": True,
                "registered": True,
                "already_present": existed,
                "task": TASK_NAME,
                "uac_used": True,
                "message": (
                    f"{TASK_NAME} bir kez UAC ile kaydedildi. "
                    "Bundan sonra cift tik / schtasks /Run → UAC yok."
                ),
            }
        time.sleep(0.25)
    return {
        "ok": False,
        "registered": False,
        "task": TASK_NAME,
        "uac_used": True,
        "error": "UAC onaylandi ama gorev henuz gorunmedi. --status ile tekrar bakin.",
    }


def start_high_task() -> None:
    cp = _run(["schtasks", "/Run", "/TN", TASK_NAME])
    if cp.returncode != 0:
        raise RuntimeError(
            "Failed to run Highest scheduled task:\n"
            + (cp.stderr or cp.stdout or f"exit={cp.returncode}")
        )


def wait_for_high_bot(timeout_s: float = 20.0) -> dict:
    deadline = time.time() + timeout_s
    last: dict = {}
    while time.time() < deadline:
        last = status_report()
        if last.get("bot_high_elevated"):
            return last
        time.sleep(0.5)
    return last or status_report()


def launch_bot_high(
    *,
    restart: bool = False,
    register_if_missing: bool = True,
    allow_uac_register: bool = True,
) -> dict:
    before = status_report()

    # Already High and not asking to restart → leave it (no UAC spam).
    if before.get("bot_high_elevated") and not restart:
        return {
            "ok": True,
            "skipped": True,
            "reason": "already_high",
            "task": TASK_NAME,
            "method": "noop",
            "before": before,
            "after": before,
            "hint": "Bot zaten High — birakildi. Yeniden baslatmak icin --restart.",
        }

    if not before.get("task_registered"):
        if not register_if_missing:
            return {
                "ok": False,
                "error": f"{TASK_NAME} kayitli degil. Once --register calistirin.",
                "before": before,
            }
        reg = register_task(force=False, allow_uac=allow_uac_register)
        if not reg.get("ok"):
            return {"ok": False, "error": reg.get("error"), "register": reg, "before": before}
        register_info = reg
    else:
        register_info = {
            "ok": True,
            "registered": False,
            "already_present": True,
            "uac_used": False,
            "task": TASK_NAME,
        }

    if restart:
        # Medium cannot kill High; Highest task-entry will kill stale twins.
        # If we are already High, pre-kill for a cleaner handoff.
        pre_killed: list[int] = []
        if is_admin():
            pre_killed = kill_stale_gatherion()
    else:
        pre_killed = []

    start_high_task()
    after = wait_for_high_bot()
    ok = bool(after.get("bot_high_elevated"))
    result = {
        "ok": ok,
        "method": f"schtasks /Run RunLevel=Highest -> {TASK_NAME}",
        "task": TASK_NAME,
        "register": register_info,
        "pre_killed": pre_killed,
        "before": before,
        "after": after,
        "hint": (
            "Bot High OK. Cift tik BOT_HIGH_NO_UAC.bat — UAC yok."
            if ok
            else "Bot High gelmedi. --status ve Gorev Zamanlayici'ya bakin."
        ),
    }
    if not ok:
        result["error"] = (
            f"schtasks /Run {TASK_NAME} ran but the bot did not reach High "
            "integrity within the wait window."
        )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Launch Gatherion High via Task Scheduler (no per-start UAC)."
    )
    ap.add_argument("--status", action="store_true", help="Probe integrity / task only")
    ap.add_argument("--register", action="store_true", help="Create/update Highest task (UAC once if needed)")
    ap.add_argument(
        "--register-elevated",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: already elevated register then exit
    )
    ap.add_argument("--force-register", action="store_true", help="Replace task even if present")
    ap.add_argument("--start", action="store_true", help="schtasks /Run only (no register UAC)")
    ap.add_argument("--restart", action="store_true", help="Kill stale via task-entry + /Run")
    ap.add_argument("--task-entry", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    ap.add_argument(
        "--no-uac",
        action="store_true",
        help="Never prompt UAC; fail if register needed",
    )
    args = ap.parse_args(argv)

    try:
        if args.task_entry:
            result = task_entry()
        elif args.register_elevated:
            _register_task_as_admin()
            result = {
                "ok": True,
                "registered": True,
                "task": TASK_NAME,
                "message": "Elevated register complete.",
            }
        elif args.status:
            result = status_report()
        elif args.register or args.force_register:
            result = register_task(
                force=bool(args.force_register),
                allow_uac=not args.no_uac,
            )
        elif args.start:
            if not task_exists():
                result = {
                    "ok": False,
                    "error": f"{TASK_NAME} yok. Once --register (bir kez UAC).",
                }
            else:
                before = status_report()
                if before.get("bot_high_elevated") and not args.restart:
                    result = {
                        "ok": True,
                        "skipped": True,
                        "reason": "already_high",
                        "before": before,
                        "after": before,
                    }
                else:
                    start_high_task()
                    after = wait_for_high_bot()
                    start_ok = bool(after.get("bot_high_elevated"))
                    result = {
                        "ok": start_ok,
                        "method": f"schtasks /Run -> {TASK_NAME}",
                        "task": TASK_NAME,
                        "before": before,
                        "after": after,
                    }
                    if not start_ok:
                        result["error"] = (
                            f"schtasks /Run {TASK_NAME} ran but the bot did not "
                            "reach High integrity within the wait window."
                        )
        else:
            # Default / --restart: smart launch
            result = launch_bot_high(
                restart=bool(args.restart),
                register_if_missing=True,
                allow_uac_register=not args.no_uac,
            )
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if isinstance(result, dict) and result.get("ok") is False and result.get("error"):
        print(f"\nERROR: {result['error']}", file=sys.stderr)
        return 1
    if args.json:
        # --json means machine-readable stdout: JSON only, no human-readable
        # trailer lines (they used to be printed unconditionally, so any
        # caller doing json.loads(stdout) failed). Exit codes are the same
        # as the human-readable path below: 0/2 for status, 0/1 otherwise.
        if isinstance(result, dict) and "bot_high_elevated" in result:
            return 0 if result.get("bot_high_elevated") else 2
        return 0 if isinstance(result, dict) and result.get("ok") else 1
    if isinstance(result, dict) and "bot_high_elevated" in result:
        flag = "YES" if result["bot_high_elevated"] else "NO"
        print(f"\nBot High elevated: {flag}")
        print(f"Task {TASK_NAME} registered: {'YES' if result.get('task_registered') else 'NO'}")
        # A registered task alone is not success: exit 0 only when the bot
        # is actually observed at High integrity.
        return 0 if result.get("bot_high_elevated") else 2
    if isinstance(result, dict) and result.get("ok"):
        if result.get("bot_high_elevated") or (result.get("after") or {}).get("bot_high_elevated"):
            print("\nBot High elevated: YES")
        elif result.get("registered") or result.get("already_present"):
            print(f"\nTask {TASK_NAME}: OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
