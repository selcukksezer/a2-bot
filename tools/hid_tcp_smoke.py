# -*- coding: utf-8 -*-
"""Smoke-test Gatherion HID line protocol against Android GatherionHidBridge.

Usage:
  python tools/hid_tcp_smoke.py --host 192.168.1.20
  python tools/hid_tcp_smoke.py --host 192.168.1.20 --keys tab,1,2,3,f
  set GATHERION_HID_TCP=192.168.1.20:8765 && python tools/hid_tcp_smoke.py
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

from core.hardware_input_port import NetworkHidAdapter, normalize_hid_endpoint


def main() -> int:
    ap = argparse.ArgumentParser(description="TCP HID bridge smoke test")
    ap.add_argument("--host", default="", help="Phone IP (or host:port)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument(
        "--keys",
        default="tab,1,2,3,f",
        help="Comma-separated key names to tap",
    )
    ap.add_argument("--hold-ms", type=int, default=80)
    args = ap.parse_args()

    endpoint = (args.host or os.environ.get("GATHERION_HID_TCP", "") or "").strip()
    if not endpoint:
        print("ERROR: pass --host PHONE_IP or set GATHERION_HID_TCP", file=sys.stderr)
        return 2
    if ":" not in endpoint:
        endpoint = f"{endpoint}:{int(args.port)}"
    endpoint = normalize_hid_endpoint(endpoint)

    hid = NetworkHidAdapter(endpoint=endpoint)
    if not hid.available:
        print(f"FAIL: cannot connect {endpoint}")
        print("  1) Phone: GatherionHidBridge running, HID paired to this PC")
        print("  2) Same Wi-Fi; app listening on 8765")
        return 1

    print(f"OK connected {endpoint}")
    keys = [k.strip().lower() for k in str(args.keys).split(",") if k.strip()]
    hold = max(20, int(args.hold_ms)) / 1000.0
    for key in keys:
        print(f"  KEYDOWN/UP {key}")
        hid.key_down(key)
        time.sleep(hold)
        hid.key_up(key)
        time.sleep(0.15)
    hid.close()
    print("Done. Focus Notepad/AION2 — keys should come from the phone HID.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
