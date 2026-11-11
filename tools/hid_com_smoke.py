# -*- coding: utf-8 -*-
"""Smoke-test Gatherion HID line protocol against Pico/Leonardo SerialHidAdapter.

Usage:
  python tools/hid_com_smoke.py --list
  python tools/hid_com_smoke.py --wait-plug --keys tab,1,2,3
  python tools/hid_com_smoke.py --port COM5 --keys tab,1,2,3,f
  set GATHERION_HID_COM=COM5 && python tools/hid_com_smoke.py
  python tools/hid_com_smoke.py --port COM5 --write-settings
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware_input_port import SerialHidAdapter
from tools.hid_com_util import (
    env_export_line,
    format_port_table,
    hid_candidates,
    list_com_ports,
    pick_best_port,
    port_devices,
    tap_keys,
    wait_for_new_com,
    write_settings_serial_port,
)


def _resolve_port(args: argparse.Namespace) -> str:
    port = (args.port or os.environ.get("GATHERION_HID_COM", "") or "").strip()
    if args.wait_plug:
        baseline = port_devices()
        print("Waiting for new COM (plug / reboot Pico)…")
        print(format_port_table())
        fresh = wait_for_new_com(baseline=baseline, timeout_sec=float(args.timeout))
        if not fresh:
            print("ERROR: no new COM within timeout", file=sys.stderr)
            return ""
        print("New COM:", ", ".join(fresh))
        # S-4901 (same pattern as F-1227 in set_hid_com.py): `pick_best_port(...)
        # or fresh[0]` used the first fresh COM without checking whether it is
        # a HID candidate. When pick_best_port returns '' (ambiguous / no
        # candidate), fail loudly instead of smoking the wrong port.
        picked = pick_best_port(prefer_new=fresh)
        if not picked:
            cands = hid_candidates()
            print(
                "ERROR: new COM(s) could not be resolved to a single HID "
                f"candidate: {', '.join(fresh)}; candidates: "
                + (", ".join(c.device for c in cands) or "(none)"),
                file=sys.stderr,
            )
            print(format_port_table(), file=sys.stderr)
            return ""
        port = picked
    if not port and args.auto:
        port = pick_best_port()
    if not port:
        # Fall back to settings.json via env-less resolve
        try:
            from core.hardware_input_port import (
                is_tcp_hid_endpoint,
                resolve_serial_hid_port,
            )

            cand = str(resolve_serial_hid_port() or "").strip()
            # resolve_serial_hid_port may return a network endpoint
            # (tcp://, udp://, or bare host:port) from GATHERION_HID_TCP /
            # input.hid_tcp — that is NOT a COM port and must never be
            # handed to SerialHidAdapter as port_name.
            if cand and not is_tcp_hid_endpoint(cand):
                port = cand
        except Exception:
            pass
    return port


def _try_read_pong(hid: SerialHidAdapter, wait_sec: float = 0.8) -> str:
    ser = getattr(hid, "_serial", None)
    if ser is None:
        return ""
    deadline = time.monotonic() + max(0.2, float(wait_sec))
    buf = b""
    while time.monotonic() < deadline:
        try:
            n = getattr(ser, "in_waiting", 0) or 0
            if n:
                buf += ser.read(n)
                if b"PONG" in buf.upper() or b"GATHERION_HID_READY" in buf.upper():
                    break
            else:
                time.sleep(0.05)
        except Exception:
            break
    try:
        return buf.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description="COM serial HID bridge smoke test")
    ap.add_argument("--port", default="", help="COMx (or set GATHERION_HID_COM)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument(
        "--keys",
        default="tab,1,2,3,f",
        help="Comma-separated key names to tap",
    )
    ap.add_argument("--hold-ms", type=int, default=80)
    ap.add_argument(
        "--ping-only",
        action="store_true",
        help="Only send PING (expect PONG on device / readback)",
    )
    ap.add_argument("--list", action="store_true", help="List COM ports and exit")
    ap.add_argument("--auto", action="store_true", help="Pick Pico/Leonardo candidate COM")
    ap.add_argument(
        "--wait-plug",
        action="store_true",
        help="Wait for a new COM after plug, then smoke it",
    )
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument(
        "--write-settings",
        action="store_true",
        help="Persist chosen port to config/settings.json input.serial_port",
    )
    args = ap.parse_args()

    if args.list:
        print(format_port_table(list_com_ports()))
        return 0

    port = _resolve_port(args)
    if not port:
        print(
            "ERROR: pass --port COMx or set GATHERION_HID_COM "
            "(or --auto / --wait-plug)",
            file=sys.stderr,
        )
        print("  Flash: tools\\flash_pico.bat   or vendor/usb_hid/pico/", file=sys.stderr)
        print("  Set:   python tools\\set_hid_com.py --wait-plug", file=sys.stderr)
        print("  Docs:  docs/PICO_OPERATOR_TR.md", file=sys.stderr)
        print(format_port_table(), file=sys.stderr)
        return 2

    print(env_export_line(port))
    if args.write_settings:
        path = write_settings_serial_port(port)
        print(f"OK wrote {path} → input.serial_port={port}")

    hid = SerialHidAdapter(port_name=port, baud=int(args.baud))
    if not hid.available:
        print(f"FAIL: cannot open {port} @ {args.baud}")
        print("  1) Pico/Leonardo plugged and firmware uploaded")
        print("  2) Correct COMx (python tools\\hid_com_smoke.py --list)")
        print("  3) Close Arduino Serial Monitor (port exclusive)")
        print("  4) python tools\\set_hid_com.py --wait-plug")
        return 1

    print(f"OK open {port} @ {args.baud} (SerialHID)")
    # Drain banner / PONG if firmware speaks.
    banner = _try_read_pong(hid, wait_sec=0.35)
    if banner:
        print(f"  device said: {banner[:120]}")

    ok_ping = hid._send("PING")
    pong = _try_read_pong(hid, wait_sec=0.8)
    if pong:
        print(f"  PING→ {pong[:80]}")
    else:
        print(f"  PING sent={ok_ping} (no readback — OK if Serial Monitor closed)")

    if args.ping_only:
        hid.close()
        return 0 if ok_ping else 1

    keys = [k.strip().lower() for k in str(args.keys).split(",") if k.strip()]

    def _report(key: str, down_ok: bool, up_ok: bool) -> None:
        print(f"  KEYDOWN/UP {key}")
        if not down_ok or not up_ok:
            print(
                f"  FAIL key {key}: key_down={down_ok} key_up={up_ok}",
                file=sys.stderr,
            )

    failed_keys = tap_keys(hid, keys, hold_ms=int(args.hold_ms), gap_sec=0.15, on_key=_report)
    hid.close()
    if failed_keys:
        print(
            f"FAIL: key send failed for {failed_keys} on {port}",
            file=sys.stderr,
        )
        return 1
    print("Done. Focus Notepad/AION2 — keys must come from the USB HID device.")
    print("Skills are NOT proven until MP/CD drops with this COM path live.")
    print("Combat path: SerialHID connected → Tab/1/2/3 hardware-only (no soft SendInput).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
