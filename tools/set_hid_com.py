# -*- coding: utf-8 -*-
"""Set Gatherion Serial HID COM port (settings.json + print env line).

Usage:
  python tools/set_hid_com.py COM5
  python tools/set_hid_com.py --auto
  python tools/set_hid_com.py --wait-plug --timeout 90
  python tools/set_hid_com.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.hid_com_util import (
    env_export_line,
    format_port_table,
    list_com_ports,
    pick_best_port,
    port_devices,
    wait_for_new_com,
    write_settings_serial_port,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist GATHERION_HID_COM / input.serial_port")
    ap.add_argument("port", nargs="?", default="", help="COMx (optional with --auto/--wait-plug)")
    ap.add_argument("--list", action="store_true", help="List COM ports and exit")
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Pick best Pico/Leonardo/CDC candidate from current ports",
    )
    ap.add_argument(
        "--wait-plug",
        action="store_true",
        help="Wait for a new COM after you plug the board, then set it",
    )
    ap.add_argument("--timeout", type=float, default=90.0, help="Seconds for --wait-plug")
    ap.add_argument(
        "--settings",
        default="",
        help="Override settings.json path (default config/settings.json)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print choice only; do not write settings.json",
    )
    args = ap.parse_args()

    if args.list:
        print(format_port_table())
        return 0

    port = str(args.port or "").strip()
    prefer_new: list[str] = []

    if args.wait_plug:
        baseline = port_devices()
        print("Waiting for new COM (plug Pico now / leave BOOTSEL)…")
        print("Baseline:")
        print(format_port_table())
        fresh = wait_for_new_com(baseline=baseline, timeout_sec=float(args.timeout))
        if not fresh:
            print("ERROR: no new COM within timeout", file=sys.stderr)
            print(format_port_table(), file=sys.stderr)
            return 1
        prefer_new = fresh
        print("New COM:", ", ".join(fresh))
        port = pick_best_port(prefer_new=fresh) or fresh[0]

    if args.auto and not port:
        port = pick_best_port()
        if not port:
            print("ERROR: no Pico/Leonardo/CDC candidate COM found", file=sys.stderr)
            print(format_port_table(), file=sys.stderr)
            return 1

    if not port:
        print("ERROR: pass COMx, or use --auto / --wait-plug", file=sys.stderr)
        print(format_port_table(), file=sys.stderr)
        return 2

    print(env_export_line(port))
    if args.dry_run:
        print(f"dry-run: would set input.serial_port={port}")
        return 0

    settings = Path(args.settings) if args.settings else None
    path = write_settings_serial_port(port, settings_path=settings)
    print(f"OK wrote {path} → input.serial_port={port}")
    print("Bot: same session →", env_export_line(port))
    print("Smoke: python tools\\hid_com_smoke.py --port", port, "--keys tab,1,2,3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
