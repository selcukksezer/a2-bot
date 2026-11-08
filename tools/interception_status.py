# -*- coding: utf-8 -*-
"""Interception readiness probe — never installs the driver.

Reports whether vendor/drivers DLLs exist and whether create_context works
(driver installed). Detects Admin elevation and prints exact manual install
command. Optional --smoke sends one key only if --confirm-smoke.

Usage:
  python tools/interception_status.py
  python tools/interception_status.py --smoke --confirm-smoke --key tab
"""
from __future__ import annotations

import argparse
import ctypes
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INSTALLER_CANDIDATES = (
    ROOT / "vendor" / "interception" / "install-interception.exe",
    ROOT
    / "vendor"
    / "interception"
    / "_extract"
    / "Interception"
    / "command line installer"
    / "install-interception.exe",
)


def _is_elevated() -> bool:
    """True when this process token has admin privileges (UAC elevated)."""
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def _find_installer() -> Path | None:
    for p in INSTALLER_CANDIDATES:
        if p.is_file():
            return p
    return None


def _print_operator_steps(installer: Path | None, elevated: bool) -> None:
    print("")
    print("--- EN / Operator (manual only; Gatherion never auto-installs) ---")
    print("  POLICY: do NOT run install-interception.exe from bot launchers.")
    if installer is None:
        print("  Installer missing — unzip Interception-v1.0.1.zip under vendor/interception/")
        print("  Expected: vendor/interception/install-interception.exe")
    else:
        print(f"  Installer staged: {installer}")
        print(f"  This shell elevated (Admin): {elevated}")
        print("  Exact Admin CMD (copy-paste):")
        print(f'    "{installer}" /install')
        print("  (uninstall: /uninstall — /command is INVALID on this build)")
        uac_bat = ROOT / "tools" / "install_interception_admin.bat"
        print(f"  One-click UAC (you still click Yes): {uac_bat}")
        print("    double-click OR: tools\\install_interception_admin.bat")
        if not elevated:
            print("  NEXT: run the bat (UAC) OR Admin CMD with a line above.")
            print("  Then reboot if the installer asks; re-run this status tool.")
        else:
            print("  You ARE elevated here — still run the installer YOURSELF (not via bot).")
            print("  After install + reboot: re-run this tool; expect INTERCEPTION_OK.")
    print("  Rootless 1B interim (Interception preferred inject):")
    print("    python tools\\rootless_hid_bridge.py")
    print("    set GATHERION_HID_TCP=udp://127.0.0.1:18766")
    print("    python tools\\set_hid_tcp.py --rootless")
    print("")
    print("--- TR / Operatör (elle; bot UAC açmaz) ---")
    print("  POLİTİKA: install-interception.exe bot/launcher ile ÇALIŞTIRILMAZ.")
    if installer is None:
        print("  Installer yok — zip'i vendor/interception/ altına açın.")
    else:
        print(f"  Hazır installer: {installer}")
        print(f"  Bu kabuk Yönetici (elevated): {elevated}")
        print("  Yönetici CMD'de birebir:")
        print(f'    "{installer}" /install')
        print("  (kaldır: /uninstall — bu build'de /command GEÇERSİZ)")
        print("  Tek tık UAC (Yine de Evet'e basarsınız): tools\\install_interception_admin.bat")
        if not elevated:
            print("  SONRAKİ ADIM: bat veya Yönetici CMD → kurulum → gerekirse reboot → bu araç.")
            print("  İsterse yeniden başlat → bu aracı tekrar çalıştır.")
        else:
            print("  Elevated'siniz; yine de installer'ı SİZ çalıştırın (sessiz auto-install yok).")
    print("  Rootless 1B:")
    print("    python tools\\rootless_hid_bridge.py")
    print("    set GATHERION_HID_TCP=udp://127.0.0.1:18766")
    print("  Ban/risk: vendor/interception/README.md")


def main() -> int:
    ap = argparse.ArgumentParser(description="Interception DLL/driver status (no install)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Attempt one key inject (requires --confirm-smoke)",
    )
    ap.add_argument(
        "--confirm-smoke",
        action="store_true",
        help="Required safety gate for --smoke",
    )
    ap.add_argument("--key", default="tab", help="Key name for smoke (default tab)")
    args = ap.parse_args()

    from core.input_driver import (
        EXTENDED_KEYS,
        HARDWARE_SCANCODES,
        InterceptionBridge,
    )

    staged = [
        ROOT / "vendor" / "interception" / "interception.dll",
        ROOT / "drivers" / "interception.dll",
        ROOT / "interception.dll",
    ]
    installer = _find_installer()
    elevated = _is_elevated()

    print("=== Interception status ===")
    for p in staged:
        print(f"  staged DLL: {p}  exists={p.is_file()}")
    print(
        f"  installer (manual only): {installer or '(missing)'}  "
        f"exists={installer is not None and installer.is_file()}"
    )
    print(f"  process elevated (Admin): {elevated}")
    print("  POLICY: do NOT auto-run install-interception.exe from Gatherion.")

    bridge = object.__new__(InterceptionBridge)
    paths = InterceptionBridge._candidate_dll_paths(bridge)
    print("  candidate search paths (first 8):")
    for p in paths[:8]:
        print(f"    {p}  exists={os.path.isfile(p)}")

    live = InterceptionBridge()
    print(f"  live.available={live.available}")
    print(f"  live.dll_path={live.dll_path or '(none)'}")
    if not live.available:
        print(
            "RESULT: DLL may be present but driver NOT installed "
            "(create_context NULL) OR DLL missing."
        )
        _print_operator_steps(installer, elevated)
        live.close()
        return 2

    print("RESULT: INTERCEPTION_OK — kernel bridge ready")
    print("  Cascade: Network/Serial HID → Interception → FakerInput → soft")
    print("  Optional 1B same protocol: GATHERION_HID_TCP=udp://127.0.0.1:18766")
    if args.smoke:
        if not args.confirm_smoke:
            print("REFUSED smoke: pass --confirm-smoke to actually send a key")
            live.close()
            return 3
        key = str(args.key).lower().strip()
        scan = int(HARDWARE_SCANCODES.get(key, 0) or 0)
        if scan <= 0:
            print(f"FAIL unknown key {key!r}")
            live.close()
            return 4
        extended = key in EXTENDED_KEYS or scan > 0xFF
        ok_down = live.send_key(scan & 0xFF, True, extended=extended)
        ok_up = live.send_key(scan & 0xFF, False, extended=extended)
        print(f"SMOKE key={key} down={ok_down} up={ok_up}")
        if ok_down and ok_up:
            print("EVIDENCE: INTERCEPTION_OK live key smoke succeeded")
        live.close()
        return 0 if (ok_down and ok_up) else 5

    live.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
