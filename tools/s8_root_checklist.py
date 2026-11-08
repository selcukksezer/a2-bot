# -*- coding: utf-8 -*-
"""SM-G950F (dreamlte) Android 9 → Magisk root operator checklist.

Does NOT flash, unlock, or silently install drivers. Probes adb when available,
detects already-unlocked bootloader (flash.locked=0 / orange), stages local
artifact paths under vendor/android_hid/s8_root/, and prints copy-paste
TWRP/Magisk/Odin steps (TR+EN).

Usage:
  python tools/s8_root_checklist.py
  python tools/s8_root_checklist.py --serial ce0617164cd200600d
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
S8_ROOT_DIR = ROOT / "vendor" / "android_hid" / "s8_root"

EXPECTED_MODEL = "SM-G950F"
EXPECTED_DEVICE = "dreamlte"
EXPECTED_RELEASE = "9"

# Official mirrors (operator verifies; script may stage copies under s8_root/)
TWRP_DEVICE_PAGE = "https://twrp.me/samsung/samsunggalaxys8.html"
TWRP_DL_INDEX = "https://dl.twrp.me/dreamlte/"
TWRP_TAR_NAME = "twrp-3.7.0_9-0-dreamlte.img.tar"
TWRP_TAR_URL = f"https://dl.twrp.me/dreamlte/{TWRP_TAR_NAME}"
TWRP_TAR_URL_EU = f"https://eu.dl.twrp.me/dreamlte/{TWRP_TAR_NAME}"
# Local staged digest (operator re-verify against dl.twrp.me sha256 page if desired)
TWRP_TAR_SHA256 = "0b3c44b73297657ced4d4cfa5c2f080f5ca8c909bd3956d7b72efdd52d00175b"
MAGISK_RELEASES = "https://github.com/topjohnwu/Magisk/releases"
MAGISK_APK_NAME = "Magisk-v31.1.apk"
MAGISK_APK_URL = (
    "https://github.com/topjohnwu/Magisk/releases/download/v31.1/Magisk-v31.1.apk"
)
# Published digest for Magisk-v31.1.apk (GitHub asset); empty = skip compare
MAGISK_APK_SHA256 = "89cefac0a318afa39fb3880c45c87b036c05bc1ebf02a7a715b7f8d7c26ac6e7"


def _find_adb() -> Optional[str]:
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
        Path(android_home) / "platform-tools" / "adb.exe" if android_home else None,
        Path(r"C:\platform-tools\adb.exe"),
        Path(r"C:\Android\Sdk\platform-tools\adb.exe"),
        Path(r"C:\Android\android-sdk\platform-tools\adb.exe"),
        ROOT / "tools" / "platform-tools" / "adb.exe",
    ]
    for p in candidates:
        if p is None:
            continue
        try:
            if p.is_file():
                return str(p.resolve())
        except OSError:
            continue
    return None


def _run(adb: str, args: List[str], serial: str = "") -> Tuple[int, str]:
    cmd = [adb]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(args)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=20,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except Exception as exc:
        return 99, str(exc)


def _prop(adb: str, serial: str, name: str) -> str:
    code, out = _run(adb, ["shell", "getprop", name], serial=serial)
    if code != 0:
        return ""
    return (out or "").splitlines()[0].strip() if out else ""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _print_staged_artifacts() -> None:
    print("")
    print(f"--- Staged folder: {S8_ROOT_DIR} ---")
    S8_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    twrp_local = S8_ROOT_DIR / TWRP_TAR_NAME
    magisk_local = S8_ROOT_DIR / MAGISK_APK_NAME
    magisk_zip = S8_ROOT_DIR / "Magisk-v31.1.zip"

    if twrp_local.is_file() and twrp_local.stat().st_size > 1_000_000:
        digest = _sha256_file(twrp_local)
        print(f"  TWRP tar: {twrp_local}")
        print(f"    size={twrp_local.stat().st_size}  sha256={digest}")
        if TWRP_TAR_SHA256:
            ok = digest.lower() == TWRP_TAR_SHA256.lower()
            print(f"    staged_expected={TWRP_TAR_SHA256}  match={ok}")
        print("    Odin: load this file in AP slot (Auto Reboot OFF)")
    else:
        print(f"  TWRP tar: MISSING under {S8_ROOT_DIR}")
        print(f"    Download (browser may require clicking 'Download' on dl.twrp.me):")
        print(f"      {TWRP_TAR_URL}")
        print(f"      {TWRP_TAR_URL_EU}")
        print(f"    Index: {TWRP_DL_INDEX}  Device page: {TWRP_DEVICE_PAGE}")
        print(f"    Save as: {twrp_local}")
        if TWRP_TAR_SHA256:
            print(f"    Staged SHA256 reference: {TWRP_TAR_SHA256}")
        print("    Verify: Get-FileHash <path> -Algorithm SHA256")

    if magisk_local.is_file() and magisk_local.stat().st_size > 100_000:
        digest = _sha256_file(magisk_local)
        print(f"  Magisk APK: {magisk_local}")
        print(f"    size={magisk_local.stat().st_size}  sha256={digest}")
        if MAGISK_APK_SHA256:
            ok = digest.lower() == MAGISK_APK_SHA256.lower()
            print(f"    expected={MAGISK_APK_SHA256}  match={ok}")
        print("    TWRP flash: copy/rename APK → Magisk-v31.1.zip (same bytes) then Install")
        if not magisk_zip.is_file():
            try:
                shutil.copy2(magisk_local, magisk_zip)
                print(f"  Magisk ZIP (staged copy): {magisk_zip}")
            except OSError as exc:
                print(f"  Magisk ZIP copy failed: {exc}")
                print(f"    Manual: copy {MAGISK_APK_NAME} → Magisk-v31.1.zip")
        else:
            print(f"  Magisk ZIP: {magisk_zip}  size={magisk_zip.stat().st_size}")
    else:
        print(f"  Magisk APK: MISSING under {S8_ROOT_DIR}")
        print(f"    Download: {MAGISK_APK_URL}")
        print(f"    Releases: {MAGISK_RELEASES}")
        print(f"    Save as: {magisk_local}")
        if MAGISK_APK_SHA256:
            print(f"    Expected SHA256: {MAGISK_APK_SHA256}")
        print("    Verify: Get-FileHash <path> -Algorithm SHA256")

    odin_dir = S8_ROOT_DIR / "odin"
    odin_exe_candidates = list(odin_dir.glob("**/Odin3*.exe")) if odin_dir.is_dir() else []
    if odin_exe_candidates:
        print(f"  Odin staged: {odin_exe_candidates[0]}")
    elif (odin_dir / "Odin3-v3.14.1.zip").is_file():
        print(f"  Odin zip staged: {odin_dir / 'Odin3-v3.14.1.zip'} (unzip if needed)")
    else:
        print("  Odin: obtain Samsung Odin 3.x from a trusted XDA thread (or stage under s8_root/odin/).")
    print("  Samsung USB drivers: required for Download mode COM port.")
    md = S8_ROOT_DIR / "multidisabler-samsung-3.1-flashable.zip"
    if md.is_file() and md.stat().st_size > 1000:
        print(f"  Multidisabler: {md}  size={md.stat().st_size}  sha256={_sha256_file(md)}")
        print("    TWRP Install FIRST (before Magisk). See docs/S8_MAGISK_SEQUENCE_SAFE.md")
    else:
        print("  Multidisabler: MISSING — fetch Ian Macdonald v3.1 for dreamlte/Pie")
        print("    https://github.com/ianmacd/multidisabler-samsung/releases/tag/v3.1")


def _print_odin_steps(*, skip_oem_unlock: bool, unlocked_hint: str) -> None:
    print("")
    print("--- EN · Exact dreamlte flash path (YOU flash; we do not) ---")
    print("WARN: Knox trip likely; wrong package can soft-brick. Backup first.")
    print("1) Backup photos/WhatsApp. Charge >60%. Data USB cable.")
    print("2) Confirm model SM-G950F / dreamlte / Android 9 (this script's live props).")
    if skip_oem_unlock:
        print(f"3) OEM unlock: SKIP — already unlocked ({unlocked_hint}).")
        print("   Re-confirm only: Developer options still shows OEM unlocking allowed,")
        print("   and Download mode does NOT say 'FRP LOCK' / locked in a way that blocks custom AP.")
    elif unlocked_hint == "probe-failed-recheck":
        print("3) OEM unlock: RECHECK live props first (adb was offline this run).")
        print("   Prior evidence this goal: flash.locked=0 + verifiedbootstate=orange → SKIP OEM.")
        print("   If re-probe shows locked≠0: Settings → OEM unlocking → Download unlock (XDA).")
    else:
        print("3) OEM unlock (only if flash.locked≠0 and verifiedbootstate≠orange):")
        print("   Settings → About → tap Build 7× → Developer options → OEM unlocking + USB debugging.")
        print("   Power off → Download mode (Vol− + Bixby + Power) → unlock per current XDA G950F guide.")
        print("   Factory reset usually follows unlock.")
    print("4) Download mode again: Vol− + Bixby + Power. Odin must show a blue COM port.")
    print("   If ID:COM empty: Windows must show Present VID_04E8&PID_685D + Modem COMx.")
    print("   Then: close Odin → Run as administrator (exact next click for Added!!).")
    print("   Still empty: USB2 port, data cable, then samsung_usb_driver/ README (TR).")
    print("   adb empty in Download mode is OK — Odin uses Samsung modem driver, not adb.")
    print("5) Odin options: Auto Reboot OFF, Re-Partition OFF. Load TWRP .img.tar into AP only.")
    print(f"   File: {S8_ROOT_DIR / TWRP_TAR_NAME}")
    print("   Start → wait PASS. Do NOT boot system yet.")
    print("6) Boot recovery manually NOW: hold Vol+ + Bixby + Power (release Power at logo).")
    print("   If stock recovery returns, TWRP was overwritten — reflash AP with Auto Reboot OFF.")
    print("7) HOLD if Download recently said 'Only official binaries' / RMM Prenormal:")
    print("   Wait ~7 days on stock with OEM unlocking ON — docs/S8_RMM_WAIT_CHECKLIST.md")
    print("   Full safe order: docs/S8_MAGISK_SEQUENCE_SAFE.md (do not flash without user confirm).")
    print("8) TWRP first boot:")
    print("   • Allow modifications if prompted.")
    print("   • If data is encrypted / cannot mount: Wipe → Format Data (yes) — wipes userdata.")
    print("9) Multidisabler FIRST, then Magisk (Samsung Pie):")
    print(f"   Push: {S8_ROOT_DIR / 'multidisabler-samsung-3.1-flashable.zip'}")
    print(f"   Push: {S8_ROOT_DIR / 'Magisk-v31.1.zip'}")
    print("   TWRP Install → multidisabler zip → then Magisk-v31.1.zip → Reboot System.")
    print("   (APK and ZIP are the same bytes since Magisk v22+.)")
    print("10) vbmeta / dm-verity (A9 dreamlte caveats):")
    print("   • Classic Exynos S8 Pie usually has NO separate vbmeta.img slot like A10+ devices.")
    print("   • Do NOT flash random 'vbmeta disable' for newer phones onto G950F.")
    print("   • Magisk typically patches verity/force-encrypt in boot; if stock kernel refuses")
    print("     system mods, prefer Magisk over manual dm-verity hacks (see twrp.me dreamlte note).")
    print("11) Verify root:")
    print("    adb shell su -c id     → must show uid=0(root)")
    print("    Re-run: python tools\\s8_root_checklist.py")
    print("12) AFTER Magisk (next for 1A live keys): open /dev/hidg0")
    print("    USB Gadget Tool / NetHunter HID / Magisk HID module, then:")
    print("    python tools\\setup_android_usb_hid.py --push --forward --write-settings --enable-gadget")
    print("    set GATHERION_HID_TCP=127.0.0.1:18765")
    print("    python tools\\hid_tcp_smoke.py --host 127.0.0.1:18765")
    print("")
    print("--- TR · dreamlte Odin/TWRP/Magisk (sen flash edersin) ---")
    print("1) Yedek. Knox 0x1 kalıcı olabilir.")
    print("2) Model = SM-G950F (Exynos). G950U ise DUR.")
    if skip_oem_unlock:
        print(f"3) OEM unlock: ATLA — zaten unlock ({unlocked_hint}).")
    elif unlocked_hint == "probe-failed-recheck":
        print("3) OEM: önce adb props tekrar ölç; önceki kanıt locked=0/orange → ATLA.")
    else:
        print("3) OEM kilit açma + USB hata ayıklama → Download → unlock (XDA).")
    print("4) Odin AP ← TWRP dreamlte .img.tar; Auto Reboot KAPALI; Re-Partition KAPALI.")
    print("   ID:COM boşsa: PnP'te PID_685D+COMx doğrula → Odin'i Yönetici olarak yeniden aç.")
    print("   Kablo/USB2/port; sürücü: vendor\\android_hid\\s8_root\\samsung_usb_driver\\")
    print("5) Hemen recovery: Vol+ + Bixby + Power.")
    print("6) Prenormal ise flash YOK — docs/S8_RMM_WAIT_CHECKLIST.md (~7 gün, OEM ON).")
    print("7) Format Data gerekirse → önce multidisabler zip, sonra Magisk ZIP → sistem.")
    print("8) adb shell su -c id → uid=0")
    print("9) Sonra hidg0 + setup_android_usb_hid.py")
    print("10) vbmeta: A9 S8'de genelde ayrı vbmeta slot yok; rastgele vbmeta flash etme.")


def main() -> int:
    ap = argparse.ArgumentParser(description="SM-G950F root/HID operator checklist (no flash)")
    ap.add_argument("--serial", default="", help="adb serial (optional)")
    args = ap.parse_args()
    serial = str(args.serial or "").strip()

    print("=== SM-G950F root kit checklist (operator; no remote flash) ===")
    print("Master doc: docs/ANDROID_USB_NO_BT_TR.md")
    print("S8 card:    docs/S8_USB_HID_GADGET_TR.md")
    print(f"Artifacts:  {S8_ROOT_DIR}")
    print("")
    print("WARN / UYARI:")
    print("  • Custom recovery / Magisk → Knox 0x1 (warranty / Samsung Pay / some banking).")
    print("  • Wrong TWRP/Odin package can soft-brick. Backup photos first.")
    print("  • This script NEVER runs Odin/fastboot unlock for you.")
    print("")

    skip_oem = False
    unlocked_hint = "unknown"
    rooted = False
    has_hid = False
    adb = _find_adb()

    if not adb:
        print("ADB: not found. Install platform-tools or set ADB=...\\adb.exe")
        print("Live probe skipped — print static steps only.")
    else:
        print(f"ADB: {adb}")
        code, devices = _run(adb, ["devices", "-l"])
        print("--- adb devices ---")
        print(devices or "(empty)")
        if not serial:
            for line in (devices or "").splitlines()[1:]:
                parts = line.split()
                if len(parts) >= 2 and parts[1] == "device":
                    serial = parts[0]
                    break
        if not serial:
            print("No authorized device — connect SM-G950F, accept RSA, re-run.")
        else:
            print(f"Using serial: {serial}")
            model = _prop(adb, serial, "ro.product.model")
            device = _prop(adb, serial, "ro.product.device")
            release = _prop(adb, serial, "ro.build.version.release")
            bootloader = _prop(adb, serial, "ro.boot.bootloader")
            vbs = _prop(adb, serial, "ro.boot.verifiedbootstate")
            locked = _prop(adb, serial, "ro.boot.flash.locked")
            warranty = _prop(adb, serial, "ro.boot.warranty_bit")
            oem_allowed = _prop(adb, serial, "sys.oem_unlock_allowed")
            usb = _prop(adb, serial, "sys.usb.config")
            code_id, id_out = _run(adb, ["shell", "id"], serial=serial)
            code_hid, hid_out = _run(adb, ["shell", "ls", "-la", "/dev/hidg0"], serial=serial)
            code_su, su_out = _run(adb, ["shell", "su", "-c", "id"], serial=serial)

            print("--- live props ---")
            print(f"  model={model!r}  (want {EXPECTED_MODEL})")
            print(f"  device={device!r}  (want {EXPECTED_DEVICE})")
            print(f"  release={release!r}  (want {EXPECTED_RELEASE})")
            print(f"  bootloader={bootloader!r}")
            print(
                f"  verifiedbootstate={vbs!r}  flash.locked={locked!r}  "
                f"warranty_bit={warranty!r}  oem_unlock_allowed={oem_allowed!r}"
            )
            print(f"  sys.usb.config={usb!r}")
            print(f"  id: {id_out}")
            print(f"  hidg0: {hid_out or f'exit={code_hid}'}")
            print(f"  su -c id: {su_out or f'exit={code_su}'}")

            device_reachable = bool(model or device or bootloader or locked or vbs)
            if not device_reachable:
                print(
                    "PROBE FAIL: serial set but getprop empty / device not found. "
                    "Reconnect USB, accept RSA, re-run (do not treat as locked)."
                )
                skip_oem = False
                unlocked_hint = "probe-failed-recheck"
            else:
                ok_model = model == EXPECTED_MODEL and device == EXPECTED_DEVICE
                if "G950U" in model or "G950W" in model:
                    print("STOP: Snapdragon / unlock-hostile variant — do NOT use dreamlte TWRP.")
                    print("Fall back: Pico HID or Interception (PC).")
                    return 3
                if not ok_model:
                    print(
                        f"WARN: model/device mismatch — expected "
                        f"{EXPECTED_MODEL}/{EXPECTED_DEVICE}"
                    )

                # Unlock heuristic: flash.locked=0 and/or orange verified boot
                skip_oem = locked == "0" or vbs == "orange"
                if skip_oem:
                    unlocked_hint = f"flash.locked={locked} verifiedbootstate={vbs}"
                    print(
                        "UNLOCK: bootloader likely ALREADY unlocked — "
                        "SKIP OEM unlock step."
                    )
                    print(
                        "  Still need TWRP + Magisk for su; "
                        "still no remote flash from this script."
                    )
                else:
                    print(
                        "UNLOCK: not confirmed unlocked — "
                        "OEM unlock step required before TWRP."
                    )

                rooted = "uid=0" in (su_out or "")
                has_hid = code_hid == 0 and "hidg0" in (hid_out or "")
                print(
                    f"SUMMARY: rooted={rooted}  hidg0={has_hid}  "
                    f"skip_oem_unlock={skip_oem}"
                )
                if warranty == "1":
                    print("NOTE: warranty_bit=1 → Knox already tripped.")
                elif warranty == "0" and skip_oem:
                    print(
                        "NOTE: warranty_bit=0 with unlocked props — Knox may still trip "
                        "on first custom AP (TWRP/Magisk)."
                    )
                if rooted and has_hid:
                    print(
                        "NEXT: python tools\\setup_android_usb_hid.py "
                        "--push --forward --write-settings"
                    )
                    print("      set GATHERION_HID_TCP=127.0.0.1:18765")
                    _print_staged_artifacts()
                    return 0
                if rooted and not has_hid:
                    print(
                        "NEXT: open /dev/hidg0 "
                        "(USB Gadget Tool / NetHunter / enable_hid_gadget.sh)"
                    )
                    print(
                        "      then: python tools\\setup_android_usb_hid.py "
                        "--push --enable-gadget"
                    )
                    _print_staged_artifacts()
                    return 0

    _print_staged_artifacts()
    _print_odin_steps(skip_oem_unlock=skip_oem, unlocked_hint=unlocked_hint)

    print("")
    print("Links (verify hashes yourself):")
    print(f"  • Magisk:     {MAGISK_RELEASES}")
    print(f"  • Magisk APK: {MAGISK_APK_URL}")
    print(f"  • TWRP page:  {TWRP_DEVICE_PAGE}")
    print(f"  • TWRP tar:   {TWRP_TAR_URL}")
    print("  • Interception interim: python tools\\interception_status.py")
    print("  • Rootless 1B: python tools\\rootless_hid_bridge.py + set_hid_tcp --rootless")
    print("")
    print("Local verify (copy-paste):")
    adb_hint = adb or r"%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe"
    print(f'  "{adb_hint}" devices -l')
    print(f'  "{adb_hint}" shell getprop ro.product.model')
    print(f'  "{adb_hint}" shell getprop ro.boot.flash.locked')
    print(f'  "{adb_hint}" shell getprop ro.boot.verifiedbootstate')
    print(f'  "{adb_hint}" shell su -c id')
    print(f'  "{adb_hint}" shell su -c "ls -l /dev/hidg0"')
    print("")
    print("GOAL: live key proof via /dev/hidg0 OR Interception — docs alone ≠ done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
