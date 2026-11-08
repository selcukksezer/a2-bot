# -*- coding: utf-8 -*-
"""Register Gatherion as a Steam non-Steam shortcut so elevated Steam can launch it without a second UAC."""
from __future__ import annotations

import struct
import time
import zlib
from pathlib import Path

STEAM_CFG = Path(r"C:\Program Files (x86)\Steam\userdata\220165889\config")
EXE = Path(r"C:\Users\selçuk\AppData\Local\Programs\Python\Python310\pythonw.exe")
START_DIR = Path(r"F:\a2")
MAIN_PY = Path(r"F:\a2\main.py")
APPNAME = "Gatherion"


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


def shortcut_app_ids(exe_field: str, appname: str) -> dict:
    out = {}
    for label, blob in (
        ("exe_field+name", exe_field + appname),
        ("exe_only+name", str(EXE) + appname),
        ("quoted_exe+name", f'"{EXE}"' + appname),
    ):
        crc = zlib.crc32(blob.encode("utf-8")) & 0xFFFFFFFF
        out[label] = crc | 0x80000000
    return out


def main() -> int:
    STEAM_CFG.mkdir(parents=True, exist_ok=True)
    path = STEAM_CFG / "shortcuts.vdf"
    exe_field = f'"{EXE}" "{MAIN_PY}"'
    start_dir = f'"{START_DIR}"'
    data = build_shortcuts_vdf(exe_field, start_dir, APPNAME)
    if path.exists():
        bak = path.with_suffix(".vdf.bak_gatherion")
        bak.write_bytes(path.read_bytes())
        print("backup", bak)
    path.write_bytes(data)
    print("wrote", path, "bytes", path.stat().st_size)
    ids = shortcut_app_ids(exe_field, APPNAME)
    for k, v in ids.items():
        print(f"appid[{k}]={v} steam://rungameid/{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
