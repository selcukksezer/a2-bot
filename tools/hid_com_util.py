# -*- coding: utf-8 -*-
"""COM/HID plug helpers for Pico / Leonardo Gatherion bridges (no hardware required to import)."""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = ROOT / "config" / "settings.json"

# Common USB IDs seen as CDC serial after Arduino-Pico / Leonardo flash.
PICO_VIDS = {0x2E8A}  # Raspberry Pi
LEONARDO_VIDS = {0x2341, 0x1B4F, 0x2A03}  # Arduino / SparkFun / clones
HINT_VIDS = PICO_VIDS | LEONARDO_VIDS


@dataclass(frozen=True)
class ComPortInfo:
    device: str
    description: str = ""
    hwid: str = ""
    vid: Optional[int] = None
    pid: Optional[int] = None
    manufacturer: str = ""

    @property
    def is_hid_candidate(self) -> bool:
        if self.vid is not None and int(self.vid) in HINT_VIDS:
            return True
        blob = f"{self.description} {self.hwid} {self.manufacturer}".lower()
        # F-1208: aday tanimi daraltildi. Jenerik "cdc" / "usb serial" /
        # "serial (com" eslesmeleri hemen her USB-seri cihazi aday sayiyordu;
        # yalnizca bilinen Pico/Leonardo kimlikleri adaydir (VID eslesmesi
        # zaten yukarida denetlenir).
        needles = (
            "pico",
            "rp2040",
            "raspberry",
            "leonardo",
            "pro micro",
            "arduino",
        )
        return any(n in blob for n in needles)


def list_com_ports() -> List[ComPortInfo]:
    """Return current COMx devices (empty list if pyserial missing)."""
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception:
        return []
    out: List[ComPortInfo] = []
    for p in list_ports.comports():
        out.append(
            ComPortInfo(
                device=str(getattr(p, "device", "") or ""),
                description=str(getattr(p, "description", "") or ""),
                hwid=str(getattr(p, "hwid", "") or ""),
                vid=getattr(p, "vid", None),
                pid=getattr(p, "pid", None),
                manufacturer=str(getattr(p, "manufacturer", "") or ""),
            )
        )
    return [x for x in out if x.device]


def port_devices(ports: Optional[Sequence[ComPortInfo]] = None) -> Set[str]:
    items = list_com_ports() if ports is None else ports
    return {p.device.upper() for p in items if p.device}


def format_port_table(ports: Optional[Sequence[ComPortInfo]] = None) -> str:
    items = list(ports) if ports is not None else list_com_ports()
    if not items:
        return "(no COM ports)"
    lines = []
    for p in items:
        flag = " *" if p.is_hid_candidate else ""
        vidpid = ""
        if p.vid is not None:
            vidpid = f" VID={p.vid:04X}"
            if p.pid is not None:
                vidpid += f":{p.pid:04X}"
        lines.append(f"  {p.device:<8}{flag}  {p.description}{vidpid}")
    lines.append("  (* = Pico/Leonardo/CDC candidate)")
    return "\n".join(lines)


def hid_candidates(
    ports: Optional[Sequence[ComPortInfo]] = None,
) -> List[ComPortInfo]:
    """Return the HID-candidate ports (Pico/Leonardo/CDC), in listed order."""
    items = list(ports) if ports is not None else list_com_ports()
    return [p for p in items if p.is_hid_candidate]


def _single_pico(cands: Sequence[ComPortInfo]) -> str:
    """Return the device of the single Pico candidate among `cands`, else "".

    Shared by both branches of pick_best_port() (fresh-plug and plain
    selection previously duplicated this filter).
    """
    pico = [p for p in cands if p.vid is not None and int(p.vid) in PICO_VIDS]
    return pico[0].device if len(pico) == 1 else ""


def pick_best_port(
    ports: Optional[Sequence[ComPortInfo]] = None,
    prefer_new: Optional[Iterable[str]] = None,
) -> str:
    """Prefer newly appeared candidates, then a unique HID-ish port.

    F-1208: birden cok aday varken keyfi `cands[0]` secilmiyor. Secim yalnizca
    tek bir aday (veya tek bir Pico adayi) varken yapilir; belirsizlikte bos
    string doner, cagiran `hid_candidates()` ile adaylari listeleyip
    kullanicidan secmesini ister.
    """
    items = list(ports) if ports is not None else list_com_ports()
    prefer = {str(x).upper() for x in (prefer_new or []) if str(x).strip()}
    if prefer:
        fresh_cands = [p for p in items if p.device.upper() in prefer]
        fresh_hid = [p for p in fresh_cands if p.is_hid_candidate]
        if len(fresh_hid) == 1:
            return fresh_hid[0].device
        if len(fresh_cands) == 1:
            return fresh_cands[0].device
        return _single_pico(fresh_hid)
    cands = [p for p in items if p.is_hid_candidate]
    if len(cands) == 1:
        return cands[0].device
    # Birden cok esit oncelikli aday: keyfi secim yok (belirsiz);
    # tek Pico adayi varsa o secilir, yoksa bos doner.
    return _single_pico(cands)


def tap_keys(adapter, keys, hold_ms: int = 80, gap_sec: float = 0.15, on_key=None) -> List[str]:
    """Tap each key (down, hold, up) on `adapter`; return failed key names.

    Shared by hid_com_smoke.py and hid_tcp_smoke.py (previously two copies
    of the same loop; the TCP copy wrapped each call in try/except while the
    COM copy crashed on a raising adapter). A key whose down/up raises or
    returns False is recorded as failed and the loop continues with the next
    key — one bad key never aborts the smoke run. Hold is clamped to a
    20 ms minimum so a tap is never a zero-length press. `on_key`, when
    given, is called after each key as on_key(key, down_ok, up_ok) so the
    caller can print its own per-key / failure lines.
    """
    hold = max(20, int(hold_ms)) / 1000.0
    failed: List[str] = []
    for raw in keys or []:
        key = str(raw or "").strip().lower()
        if not key:
            continue
        try:
            down_ok = bool(adapter.key_down(key))
        except Exception:
            down_ok = False
        time.sleep(hold)
        try:
            up_ok = bool(adapter.key_up(key))
        except Exception:
            up_ok = False
        if not down_ok or not up_ok:
            failed.append(key)
        if on_key is not None:
            on_key(key, down_ok, up_ok)
        if gap_sec:
            time.sleep(max(0.0, float(gap_sec)))
    return failed


def wait_for_new_com(
    baseline: Optional[Iterable[str]] = None,
    timeout_sec: float = 60.0,
    poll_sec: float = 0.5,
) -> List[str]:
    """Block until a new COMx appears vs baseline (or current snapshot)."""
    base = {str(x).upper() for x in (baseline if baseline is not None else port_devices())}
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    while time.monotonic() < deadline:
        now = port_devices()
        fresh = sorted(now - base)
        if fresh:
            return fresh
        time.sleep(max(0.1, float(poll_sec)))
    return []


def write_settings_serial_port(port: str, settings_path: Optional[Path] = None) -> Path:
    """Persist input.serial_port in config/settings.json (preserve other keys).

    S-4902 (same pattern as F-1228 in set_hid_tcp.py / setup_android_usb_hid.py):
    a corrupt settings.json used to crash the caller (bare json.loads) and the
    write was non-atomic (direct write_text). Now a corrupt file is backed up
    and we start fresh, and the write goes through tmp + os.replace.
    """
    path = Path(settings_path) if settings_path else SETTINGS_PATH
    port_s = str(port or "").strip()
    if not port_s:
        raise ValueError("empty COM port")
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            # Corrupt settings.json must not crash the tool or be silently
            # wiped: back it up, then start from an empty dict.
            backup = path.with_name(f"{path.name}.corrupt-{int(time.time())}")
            try:
                os.replace(path, backup)
            except OSError:
                backup = path.with_name(f"{path.name}.corrupt")
            print(
                f"WARNING: corrupt settings.json ({exc}); "
                f"backed up to {backup}, starting fresh",
                file=sys.stderr,
            )
            data = {}
        if not isinstance(data, dict):
            data = {}
    input_cfg = data.get("input")
    if not isinstance(input_cfg, dict):
        input_cfg = {}
    input_cfg["serial_port"] = port_s
    comment = str(input_cfg.get("_comment") or "").strip()
    if not comment:
        input_cfg["_comment"] = (
            "PRIMARY no-BT keyboard: Pico/Leonardo COM via GATHERION_HID_COM or serial_port. "
            "COM wins over TCP."
        )
    data["input"] = input_cfg
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file in the same dir, then replace.
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def env_export_line(port: str) -> str:
    return f"set GATHERION_HID_COM={str(port).strip()}"
