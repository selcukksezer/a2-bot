# -*- coding: utf-8 -*-
"""COM/HID plug helpers for Pico / Leonardo Gatherion bridges (no hardware required to import)."""
from __future__ import annotations

import json
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
        needles = (
            "pico",
            "rp2040",
            "raspberry",
            "leonardo",
            "pro micro",
            "arduino",
            "cdc",
            "usb serial",
            "serial (com",
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


def pick_best_port(
    ports: Optional[Sequence[ComPortInfo]] = None,
    prefer_new: Optional[Iterable[str]] = None,
) -> str:
    """Prefer newly appeared candidates, then any HID-ish port, else newest single COM."""
    items = list(ports) if ports is not None else list_com_ports()
    prefer = {str(x).upper() for x in (prefer_new or []) if str(x).strip()}
    if prefer:
        for p in items:
            if p.device.upper() in prefer and p.is_hid_candidate:
                return p.device
        for p in items:
            if p.device.upper() in prefer:
                return p.device
    cands = [p for p in items if p.is_hid_candidate]
    if len(cands) == 1:
        return cands[0].device
    if cands:
        # Prefer Pico VID when several candidates exist.
        for p in cands:
            if p.vid is not None and int(p.vid) in PICO_VIDS:
                return p.device
        return cands[0].device
    return ""


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
    """Persist input.serial_port in config/settings.json (preserve other keys)."""
    path = Path(settings_path) if settings_path else SETTINGS_PATH
    port_s = str(port or "").strip()
    if not port_s:
        raise ValueError("empty COM port")
    data: dict = {}
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
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
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def env_export_line(port: str) -> str:
    return f"set GATHERION_HID_COM={str(port).strip()}"
