# -*- coding: utf-8 -*-
"""Launch Steam (and optionally AION2) at Medium integrity without UAC.

Why this exists
---------------
UIPI blocks Medium→High SendInput. If Steam is elevated/High, AION2 inherits High
and Gatherion Tab/skills fail unless the bot is also High.

On this machine the usual traps are NOT Steam's manifest (asInvoker) and NOT
shortcut RUNASADMIN. The real failure mode is launching Steam from an elevated
parent (elevated Explorer/CMD/Gatherion). Children inherit that High token.

Fix: start Steam through a Task Scheduler action with RunLevel=Limited so the
new process is always Medium / unelevated, even if the caller is elevated.

Usage
-----
  python tools/steam_unelevated_launch.py              # Steam only
  python tools/steam_unelevated_launch.py --aion2      # Steam + AION2
  python tools/steam_unelevated_launch.py --status     # probe only
  python tools/steam_unelevated_launch.py --restart    # shutdown Steam first
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

STEAM_EXE = Path(r"C:\Program Files (x86)\Steam\steam.exe")
AION2_APPID = "3393110"
TASK_NAME = "Gatherion_SteamMedium"
TASK_XML_NAME = "Gatherion_SteamMedium"

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

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

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


class SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class TOKEN_MANDATORY_LABEL(ctypes.Structure):
    _fields_ = [("Label", SID_AND_ATTRIBUTES)]


def probe_pid(pid: int) -> dict:
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


def list_processes(*names: str) -> list[dict]:
    want = {n.lower() for n in names}
    out: list[dict] = []
    raw = subprocess.check_output(["tasklist", "/FO", "CSV", "/NH"], text=True, errors="replace")
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
    return out


def status_report() -> dict:
    targets = list_processes("steam.exe", "AION2.exe", "aion2.exe", "explorer.exe", "pythonw.exe")
    steam = [p for p in targets if p.get("name", "").lower() == "steam.exe"]
    aion = [p for p in targets if p.get("name", "").lower() == "aion2.exe"]
    ok_steam = bool(steam) and all(
        p.get("ok") and p.get("integrity") == "Medium" and not p.get("elevated") for p in steam
    )
    ok_aion = (not aion) or all(
        p.get("ok") and p.get("integrity") == "Medium" and not p.get("elevated") for p in aion
    )
    return {
        "steam_medium_unelevated": ok_steam,
        "aion2_medium_unelevated_or_absent": ok_aion,
        "processes": targets,
    }


def _run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, text=True, capture_output=True, check=check)


def steam_shutdown(timeout_s: float = 25.0) -> bool:
    if not list_processes("steam.exe"):
        return True
    if STEAM_EXE.exists():
        _run([str(STEAM_EXE), "-shutdown"])
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not list_processes("steam.exe"):
            return True
        time.sleep(0.5)
    # Last resort: taskkill (fails against High from Medium)
    _run(["taskkill", "/IM", "steam.exe", "/F"])
    _run(["taskkill", "/IM", "steamwebhelper.exe", "/F"])
    time.sleep(1.0)
    return not list_processes("steam.exe")


def _ps_quote(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def ensure_medium_task(command: str, arguments: str) -> None:
    """Register/replace a Limited (Medium) interactive scheduled task."""
    # Remove stale task quietly.
    _run(["schtasks", "/Delete", "/TN", TASK_NAME, "/F"])
    if arguments.strip():
        action_ps = (
            f"New-ScheduledTaskAction -Execute {_ps_quote(command)} "
            f"-Argument {_ps_quote(arguments)}"
        )
    else:
        action_ps = f"New-ScheduledTaskAction -Execute {_ps_quote(command)}"
    ps = f"""
$tn = {_ps_quote(TASK_NAME)}
$action = {action_ps}
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
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
    if cp.returncode != 0:
        raise RuntimeError(
            "Failed to register Medium scheduled task:\n"
            + (cp.stderr or cp.stdout or f"exit={cp.returncode}")
        )


def start_medium_task() -> None:
    cp = _run(["schtasks", "/Run", "/TN", TASK_NAME])
    if cp.returncode != 0:
        raise RuntimeError(
            "Failed to run Medium scheduled task:\n"
            + (cp.stderr or cp.stdout or f"exit={cp.returncode}")
        )


def launch_steam_medium(*, with_aion2: bool, restart: bool) -> dict:
    if not STEAM_EXE.exists():
        raise FileNotFoundError(f"Steam not found: {STEAM_EXE}")

    before = status_report()
    if restart or any(
        p.get("elevated") or p.get("integrity") == "High"
        for p in before["processes"]
        if p.get("name", "").lower() == "steam.exe"
    ):
        if not steam_shutdown():
            return {
                "ok": False,
                "error": (
                    "Steam is still running and could not be stopped from this Medium shell. "
                    "If Steam is High, kill it from an elevated admin prompt "
                    "(taskkill /IM steam.exe /F) or Task Manager (Run as admin), then re-run."
                ),
                "before": before,
                "after": status_report(),
            }

    if with_aion2:
        # steam:// through steam.exe keeps the same integrity as the Steam process we start.
        args = f'-applaunch {AION2_APPID}'
    else:
        args = ""

    ensure_medium_task(str(STEAM_EXE), args)
    start_medium_task()

    # Wait for steam.exe to appear / settle.
    steam_ready = False
    for _ in range(40):
        steam = list_processes("steam.exe")
        if steam:
            steam_ready = True
            break
        time.sleep(0.5)

    after = status_report()
    ok = bool(after.get("steam_medium_unelevated")) and steam_ready
    return {
        "ok": ok,
        "method": f"schtasks RunLevel=Limited -> {STEAM_EXE.name} {args}".strip(),
        "task": TASK_NAME,
        "with_aion2": with_aion2,
        "before": before,
        "after": after,
        "hint": (
            "Steam Medium OK. Launch AION2 from this Steam (Play) or re-run with --aion2. "
            "Child AION2 inherits Steam integrity — do NOT 'Run as administrator'."
            if ok
            else "Steam did not come up Medium. See 'after' probe."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Launch Steam/AION2 at Medium integrity (no UAC).")
    ap.add_argument("--status", action="store_true", help="Probe integrity only")
    ap.add_argument("--aion2", action="store_true", help="Also -applaunch AION2 (appid 3393110)")
    ap.add_argument("--restart", action="store_true", help="Shutdown existing Steam first")
    ap.add_argument("--json", action="store_true", help="Print JSON only")
    args = ap.parse_args(argv)

    try:
        if args.status:
            result = status_report()
        else:
            result = launch_steam_medium(with_aion2=args.aion2, restart=args.restart)
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.json:
        print(text)
    else:
        print(text)
        if isinstance(result, dict) and result.get("ok") is False and result.get("error"):
            print(f"\nERROR: {result['error']}", file=sys.stderr)
            return 1
        if isinstance(result, dict) and "steam_medium_unelevated" in result:
            flag = "YES" if result["steam_medium_unelevated"] else "NO"
            print(f"\nSteam Medium unelevated: {flag}")
            return 0 if result["steam_medium_unelevated"] else 2
        if isinstance(result, dict) and result.get("ok"):
            print("\nSteam Medium unelevated: YES")
            return 0
        return 1
    return 0 if result.get("ok") or result.get("steam_medium_unelevated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
