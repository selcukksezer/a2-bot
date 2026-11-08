# -*- coding: utf-8 -*-
"""Push rooted-Android USB HID server + optional adb forward for Gatherion.

Does NOT root the phone or flash TWRP/Magisk — operator must do that first.
See docs/ANDROID_USB_NO_BT_TR.md (S8 Android 9 checklist).

Usage:
  python tools/setup_android_usb_hid.py --check
  python tools/setup_android_usb_hid.py --push
  python tools/setup_android_usb_hid.py --push --forward --write-settings
  python tools/setup_android_usb_hid.py --enable-gadget
  python tools/setup_android_usb_hid.py --start-server
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware_input_port import normalize_hid_endpoint

USB_GADGET = ROOT / "vendor" / "android_hid" / "usb_gadget"
SERVER_PY = USB_GADGET / "android_usb_hid_server.py"
ENABLE_SH = USB_GADGET / "enable_hid_gadget.sh"
PHONE_DIR = "/data/local/tmp"
DEFAULT_FORWARD_LOCAL = 18765
DEFAULT_PHONE_PORT = 8765


def _adb_bin() -> str:
    """Resolve adb.exe via ADB env, PATH, then absolute SDK/vendor fallbacks."""
    env = str(os.environ.get("ADB", "") or "").strip()
    if env and Path(env).is_file():
        return str(Path(env).resolve())

    which = shutil.which("adb")
    if which and Path(which).is_file():
        return which

    localapp = os.environ.get("LOCALAPPDATA", "") or ""
    android_home = os.environ.get("ANDROID_HOME", "") or os.environ.get("ANDROID_SDK_ROOT", "") or ""
    candidates = [
        ROOT / "vendor" / "android_sdk" / "platform-tools" / "adb.exe",
        Path(r"F:\a2\vendor\android_sdk\platform-tools\adb.exe"),
        Path(localapp) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path.home() / "AppData" / "Local" / "Android" / "Sdk" / "platform-tools" / "adb.exe",
        Path(android_home) / "platform-tools" / "adb.exe" if android_home else None,
        Path(r"C:\platform-tools\adb.exe"),
        Path(r"C:\Android\Sdk\platform-tools\adb.exe"),
        Path(r"C:\Android\android-sdk\platform-tools\adb.exe"),
        ROOT / "tools" / "platform-tools" / "adb.exe",
    ]
    for cand in candidates:
        if cand is None:
            continue
        try:
            if cand.is_file():
                return str(cand.resolve())
        except OSError:
            continue
    return "adb"


def run_adb(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    cmd = [_adb_bin(), *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
    )


def adb_devices() -> list[str]:
    proc = run_adb(["devices"])
    lines = (proc.stdout or "").splitlines()
    serials: list[str] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serials.append(parts[0])
    return serials


def write_settings_hid_tcp(endpoint: str) -> Path:
    path = ROOT / "config" / "settings.json"
    data: dict = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        data = {}
    input_cfg = data.get("input")
    if not isinstance(input_cfg, dict):
        input_cfg = {}
        data["input"] = input_cfg
    input_cfg["hid_tcp"] = endpoint
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def cmd_check() -> int:
    print("=== Gatherion Android USB HID setup check ===")
    print(f"server: {SERVER_PY}  exists={SERVER_PY.is_file()}")
    print(f"enable: {ENABLE_SH}  exists={ENABLE_SH.is_file()}")
    adb = _adb_bin()
    print(f"adb: {adb}")
    serials = adb_devices()
    if not serials:
        print("adb devices: (none) — enable USB debugging / authorize PC")
        print("Root/HID cannot be verified until a device is listed.")
        return 1
    print("adb devices:", ", ".join(serials))
    # Root + hidg0 probes (best-effort; may fail pre-root)
    who = run_adb(["shell", "su", "-c", "id"])
    print("su -c id:", (who.stdout or who.stderr or "").strip() or f"exit={who.returncode}")
    hid = run_adb(["shell", "su", "-c", "ls -l /dev/hidg0"])
    out = (hid.stdout or hid.stderr or "").strip()
    print("hidg0:", out or f"missing (exit={hid.returncode})")
    if who.returncode != 0:
        print("NEXT: root S8 (OEM unlock → TWRP → Magisk). See docs/ANDROID_USB_NO_BT_TR.md")
        return 2
    if hid.returncode != 0 or "hidg0" not in out:
        print("NEXT: enable HID gadget (NetHunter / USB Gadget Tool / enable_hid_gadget.sh)")
        return 3
    print("OK: root + /dev/hidg0 look present")
    return 0


def cmd_push() -> int:
    if not SERVER_PY.is_file() or not ENABLE_SH.is_file():
        print("ERROR: usb_gadget files missing under vendor/android_hid/usb_gadget", file=sys.stderr)
        return 1
    if not adb_devices():
        print("ERROR: no adb device", file=sys.stderr)
        return 1
    for src, name in ((SERVER_PY, "android_usb_hid_server.py"), (ENABLE_SH, "enable_hid_gadget.sh")):
        dest = f"{PHONE_DIR}/{name}"
        proc = run_adb(["push", str(src), dest])
        print(proc.stdout or proc.stderr or f"push {name}")
        if proc.returncode != 0:
            return proc.returncode
        run_adb(["shell", "su", "-c", f"chmod 755 {dest}"])
    print(f"OK pushed to {PHONE_DIR}/")
    return 0


def cmd_forward(local: int, remote: int) -> int:
    if not adb_devices():
        print("ERROR: no adb device", file=sys.stderr)
        return 1
    proc = run_adb(["forward", f"tcp:{local}", f"tcp:{remote}"])
    if proc.returncode != 0:
        print(proc.stderr or proc.stdout or "adb forward failed", file=sys.stderr)
        return proc.returncode
    print(f"OK adb forward tcp:{local} → tcp:{remote}")
    print(f"set GATHERION_HID_TCP=127.0.0.1:{local}")
    return 0


def cmd_enable_gadget() -> int:
    if not adb_devices():
        print("ERROR: no adb device", file=sys.stderr)
        return 1
    script = f"{PHONE_DIR}/enable_hid_gadget.sh"
    # Ensure script is on device
    if ENABLE_SH.is_file():
        run_adb(["push", str(ENABLE_SH), script])
    proc = run_adb(["shell", "su", "-c", f"sh {script}"])
    print((proc.stdout or "").strip())
    if proc.stderr:
        print(proc.stderr.strip(), file=sys.stderr)
    return proc.returncode


def cmd_start_server() -> int:
    if not adb_devices():
        print("ERROR: no adb device", file=sys.stderr)
        return 1
    server = f"{PHONE_DIR}/android_usb_hid_server.py"
    # Non-blocking: start in background via nohup-ish adb shell
    proc = run_adb(
        [
            "shell",
            "su",
            "-c",
            f"nohup python {server} >/data/local/tmp/gatherion_hid.log 2>&1 &",
        ]
    )
    print((proc.stdout or proc.stderr or "").strip() or "start requested")
    print("Log on phone: /data/local/tmp/gatherion_hid.log")
    print("Verify: adb shell su -c \"tail -n 20 /data/local/tmp/gatherion_hid.log\"")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rooted Android USB HID → Gatherion (push/forward/check)"
    )
    ap.add_argument("--check", action="store_true", help="Probe adb/root/hidg0")
    ap.add_argument("--push", action="store_true", help="adb push server + enable script")
    ap.add_argument(
        "--forward",
        action="store_true",
        help=f"adb forward tcp:{DEFAULT_FORWARD_LOCAL} tcp:{DEFAULT_PHONE_PORT}",
    )
    ap.add_argument("--local-port", type=int, default=DEFAULT_FORWARD_LOCAL)
    ap.add_argument("--phone-port", type=int, default=DEFAULT_PHONE_PORT)
    ap.add_argument(
        "--write-settings",
        action="store_true",
        help="Write config/settings.json input.hid_tcp for adb-forward endpoint",
    )
    ap.add_argument(
        "--endpoint",
        default="",
        help="Override hid_tcp endpoint written with --write-settings",
    )
    ap.add_argument(
        "--enable-gadget",
        action="store_true",
        help="Run enable_hid_gadget.sh on phone as root (best-effort)",
    )
    ap.add_argument(
        "--start-server",
        action="store_true",
        help="Start android_usb_hid_server.py on phone as root (background)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Offline: print planned paths/endpoints only",
    )
    args = ap.parse_args()

    # Default to --check when nothing selected
    if not any(
        (
            args.check,
            args.push,
            args.forward,
            args.write_settings,
            args.enable_gadget,
            args.start_server,
            args.dry_run,
        )
    ):
        args.check = True

    if args.dry_run:
        ep = args.endpoint or f"127.0.0.1:{args.local_port}"
        print("dry-run OK")
        print(f"  SERVER_PY={SERVER_PY} exists={SERVER_PY.is_file()}")
        print(f"  ENABLE_SH={ENABLE_SH} exists={ENABLE_SH.is_file()}")
        print(f"  PHONE_DIR={PHONE_DIR}")
        print(f"  forward tcp:{args.local_port} → tcp:{args.phone_port}")
        print(f"  endpoint={normalize_hid_endpoint(ep)}")
        print("  docs=docs/ANDROID_USB_NO_BT_TR.md")
        return 0 if SERVER_PY.is_file() and ENABLE_SH.is_file() else 1

    rc = 0
    if args.check:
        rc = cmd_check() or rc
    if args.push:
        rc = cmd_push() or rc
    if args.enable_gadget:
        rc = cmd_enable_gadget() or rc
    if args.start_server:
        rc = cmd_start_server() or rc
    if args.forward:
        rc = cmd_forward(args.local_port, args.phone_port) or rc
    if args.write_settings:
        ep = (args.endpoint or f"127.0.0.1:{args.local_port}").strip()
        # Prefer host:port form for settings (normalize strips scheme for TCP)
        canon = normalize_hid_endpoint(ep)
        stored = canon[6:] if canon.lower().startswith("tcp://") else canon
        path = write_settings_hid_tcp(stored)
        print(f"OK wrote {path} → input.hid_tcp={stored}")
        print(f"set GATHERION_HID_TCP={stored}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
