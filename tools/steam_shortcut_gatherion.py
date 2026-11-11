# -*- coding: utf-8 -*-
"""Register Gatherion as a Steam non-Steam shortcut so elevated Steam can launch it without a second UAC.

Paths
-----
The defaults below are Selcuk's machine (hard-coded user id / drive / Python
3.10 path) and are kept as the default, so behaviour on that machine is
unchanged. When the install lives elsewhere, override without editing code:

  python tools/steam_shortcut_gatherion.py --steam-cfg <dir> --exe <pythonw.exe> \\
      --start-dir <project dir> --main-py <main.py> [--appname Gatherion] [--dry-run]

or with env vars: STEAM_SHORTCUT_CFG_DIR, GATHERION_EXE, GATHERION_START_DIR,
GATHERION_MAIN_PY, GATHERION_APPNAME. The Steam config dir is otherwise
auto-discovered (historical dir first, then a scan of Steam install roots
for userdata/*/config).
"""
from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import zlib
from pathlib import Path

STEAM_CFG = Path(r"C:\Program Files (x86)\Steam\userdata\220165889\config")
EXE = Path(r"C:\Users\selçuk\AppData\Local\Programs\Python\Python310\pythonw.exe")
START_DIR = Path(r"F:\a2")
MAIN_PY = Path(r"F:\a2\main.py")
APPNAME = "Gatherion"

ENV_STEAM_CFG = "STEAM_SHORTCUT_CFG_DIR"
ENV_EXE = "GATHERION_EXE"
ENV_START_DIR = "GATHERION_START_DIR"
ENV_MAIN_PY = "GATHERION_MAIN_PY"
ENV_APPNAME = "GATHERION_APPNAME"


def _steam_install_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("PROGRAMFILES(X86)", "PROGRAMFILES"):
        root = os.environ.get(env_name)
        if root:
            roots.append(Path(root) / "Steam")
    roots.append(Path(r"C:\Program Files (x86)\Steam"))
    roots.append(Path(r"C:\Program Files\Steam"))
    seen: set[Path] = set()
    out: list[Path] = []
    for root in roots:
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def discover_steam_cfg(
    explicit: str | Path | None = None, *, steam_roots: list[Path] | None = None
) -> Path:
    """Resolve the Steam userdata config dir holding shortcuts.vdf.

    Order: explicit --steam-cfg / STEAM_SHORTCUT_CFG_DIR env var, then the
    historical hard-coded dir when it exists, then a scan of Steam install
    roots for userdata/*/config (preferring one that already holds a
    shortcuts.vdf), then the hard-coded dir as a last resort so the old
    error paths are preserved when nothing is found.
    """
    if explicit:
        return Path(explicit)
    env = os.environ.get(ENV_STEAM_CFG)
    if env:
        return Path(env)
    if STEAM_CFG.exists():
        return STEAM_CFG
    roots = [Path(r) for r in steam_roots] if steam_roots is not None else _steam_install_roots()
    found: list[Path] = []
    for root in roots:
        try:
            found.extend(sorted((root / "userdata").glob("*/config")))
        except OSError:
            continue
    with_vdf = [c for c in found if (c / "shortcuts.vdf").exists()]
    if with_vdf:
        return with_vdf[0]
    if found:
        return found[0]
    return STEAM_CFG


def resolve_launch_paths(
    *,
    exe: str | Path | None = None,
    start_dir: str | Path | None = None,
    main_py: str | Path | None = None,
    appname: str | None = None,
) -> dict:
    """Effective launch paths: CLI arg > env var > hard-coded default."""
    return {
        "exe": Path(exe or os.environ.get(ENV_EXE) or EXE),
        "start_dir": Path(start_dir or os.environ.get(ENV_START_DIR) or START_DIR),
        "main_py": Path(main_py or os.environ.get(ENV_MAIN_PY) or MAIN_PY),
        "appname": str(appname or os.environ.get(ENV_APPNAME) or APPNAME),
    }


def _sobj() -> bytes:
    return b"\x00"


def _eobj() -> bytes:
    return b"\x08"


def _sstr(key: str, value: str) -> bytes:
    return b"\x01" + key.encode("utf-8") + b"\x00" + str(value).encode("utf-8") + b"\x00"


def _sint(key: str, value: int) -> bytes:
    return b"\x02" + key.encode("utf-8") + b"\x00" + struct.pack("<i", int(value))


def build_shortcuts_vdf(exe_field: str, start_dir: str, appname: str) -> bytes:
    body = bytearray()
    body += _sobj() + b"shortcuts" + b"\x00"
    body += _sobj() + b"0" + b"\x00"
    body += _sint("appid", 0)
    body += _sstr("AppName", appname)
    body += _sstr("Exe", exe_field)
    body += _sstr("StartDir", start_dir)
    body += _sstr("icon", "")
    body += _sstr("ShortcutPath", "")
    body += _sstr("LaunchOptions", "")
    body += _sint("IsHidden", 0)
    body += _sint("AllowDesktopConfig", 1)
    body += _sint("AllowOverlay", 1)
    body += _sint("OpenVR", 0)
    body += _sint("Devkit", 0)
    body += _sstr("DevkitGameID", "")
    body += _sint("DevkitOverrideAppID", 0)
    body += _sint("LastPlayTime", int(time.time()))
    body += _sstr("FlatpakAppID", "")
    body += _sobj() + b"tags" + b"\x00" + _eobj()
    body += _eobj()
    body += _eobj()
    return bytes(body)


def shortcut_app_ids(
    exe_field: str, appname: str, exe: str | Path | None = None
) -> dict:
    exe_s = str(exe if exe is not None else EXE)
    out = {}
    for label, blob in (
        ("exe_field+name", exe_field + appname),
        ("exe_only+name", exe_s + appname),
        ("quoted_exe+name", f'"{exe_s}"' + appname),
    ):
        crc = zlib.crc32(blob.encode("utf-8")) & 0xFFFFFFFF
        out[label] = crc | 0x80000000
    return out


def _existing_app_names(data: bytes) -> list:
    """Extract AppName values from a binary shortcuts.vdf blob."""
    names: list = []
    marker = b"\x01AppName\x00"
    pos = 0
    while True:
        idx = data.find(marker, pos)
        if idx < 0:
            break
        start = idx + len(marker)
        end = data.find(b"\x00", start)
        if end < 0:
            break
        try:
            names.append(data[start:end].decode("utf-8", errors="replace"))
        except Exception:
            pass
        pos = end + 1
    return names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Register Gatherion as a Steam non-Steam shortcut."
    )
    ap.add_argument("--steam-cfg", default=None,
                    help="Steam userdata config dir holding shortcuts.vdf "
                         "(default: auto-discovered; env STEAM_SHORTCUT_CFG_DIR)")
    ap.add_argument("--exe", default=None,
                    help="pythonw.exe to launch (default: hard-coded Python 3.10; "
                         "env GATHERION_EXE)")
    ap.add_argument("--start-dir", default=None,
                    help="Gatherion project dir (default: F:\\a2; env GATHERION_START_DIR)")
    ap.add_argument("--main-py", default=None,
                    help="Gatherion main.py (default: F:\\a2\\main.py; env GATHERION_MAIN_PY)")
    ap.add_argument("--appname", default=None,
                    help="Shortcut name in Steam (default: Gatherion; env GATHERION_APPNAME)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print what would be written without touching shortcuts.vdf")
    args = ap.parse_args(argv)

    cfg = discover_steam_cfg(args.steam_cfg)
    paths = resolve_launch_paths(
        exe=args.exe, start_dir=args.start_dir, main_py=args.main_py,
        appname=args.appname,
    )
    exe = paths["exe"]
    appname = paths["appname"]
    path = cfg / "shortcuts.vdf"
    exe_field = f'"{exe}" "{paths["main_py"]}"'
    start_dir = f'"{paths["start_dir"]}"'
    data = build_shortcuts_vdf(exe_field, start_dir, appname)
    if args.dry_run:
        print("dry-run: would write", path, "bytes", len(data))
        print("dry-run: exe_field", exe_field)
        print("dry-run: start_dir", start_dir)
        ids = shortcut_app_ids(exe_field, appname, exe)
        for k, v in ids.items():
            print(f"dry-run: appid[{k}]={v} steam://rungameid/{v}")
        return 0
    cfg.mkdir(parents=True, exist_ok=True)
    if path.exists():
        # F-1130: shortcuts.vdf tamamen uzerine yaziliyor, mevcut Steam
        # kisayollari siliniyordu. Artik: dosyada Gatherion disinda kisayol
        # varsa (veya icerik cozumlenemiyorsa) YAZMA, hata ver ve cik; yalnizca
        # dosya yoksa ya da yalnizca Gatherion iceriyorsa guncelle.
        existing = path.read_bytes()
        names = _existing_app_names(existing)
        if not names:
            print(
                "ERROR: refusing to overwrite shortcuts.vdf: existing content "
                "could not be parsed (unknown shortcuts may be lost). "
                f"Back it up manually: {path}",
                file=sys.stderr,
            )
            return 1
        foreign = [n for n in names if n != appname]
        if foreign:
            print(
                "ERROR: refusing to overwrite shortcuts.vdf: it contains "
                f"other Steam shortcuts ({', '.join(foreign)}). Overwriting "
                "would delete them. Remove/merge them manually first.",
                file=sys.stderr,
            )
            return 1
        bak = path.with_suffix(".vdf.bak_gatherion")
        bak.write_bytes(existing)
        print("backup", bak)
    path.write_bytes(data)
    print("wrote", path, "bytes", path.stat().st_size)
    ids = shortcut_app_ids(exe_field, appname, exe)
    for k, v in ids.items():
        print(f"appid[{k}]={v} steam://rungameid/{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
