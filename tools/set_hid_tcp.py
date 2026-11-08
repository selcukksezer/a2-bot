# -*- coding: utf-8 -*-
"""Persist Gatherion Network HID endpoint (settings.json + print env line).

Usage:
  python tools/set_hid_tcp.py 192.168.1.55:8765
  python tools/set_hid_tcp.py udp://192.168.1.55:8765
  python tools/set_hid_tcp.py 127.0.0.1:18765
  python tools/set_hid_tcp.py --adb-forward
  python tools/set_hid_tcp.py --rootless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.hardware_input_port import normalize_hid_endpoint


def _settings_path(override: str = "") -> Path:
    if override:
        return Path(override)
    return ROOT / "config" / "settings.json"


def write_settings_hid_tcp(endpoint: str, settings_path: Path | None = None) -> Path:
    path = settings_path or _settings_path()
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Persist GATHERION_HID_TCP / input.hid_tcp")
    ap.add_argument(
        "endpoint",
        nargs="?",
        default="",
        help="host:port | tcp://host:port | udp://host:port",
    )
    ap.add_argument(
        "--adb-forward",
        action="store_true",
        help="Use 127.0.0.1:18765 (pair with: adb forward tcp:18765 tcp:8765)",
    )
    ap.add_argument(
        "--rootless",
        action="store_true",
        help="1B bridge: udp://127.0.0.1:18766 (tools/rootless_hid_bridge.py)",
    )
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

    endpoint = str(args.endpoint or "").strip()
    if args.adb_forward and args.rootless:
        print("ERROR: use only one of --adb-forward / --rootless", file=sys.stderr)
        return 2
    if args.adb_forward:
        endpoint = "127.0.0.1:18765"
    if args.rootless:
        endpoint = "udp://127.0.0.1:18766"
    if not endpoint:
        print(
            "ERROR: pass host:port or --adb-forward or --rootless",
            file=sys.stderr,
        )
        return 2

    canonical = normalize_hid_endpoint(endpoint)
    # settings / env historically store host:port or udp://… without requiring tcp://
    stored = endpoint
    if canonical.lower().startswith("tcp://"):
        stored = canonical[6:]
    elif canonical.lower().startswith("udp://"):
        stored = canonical

    print(f"set GATHERION_HID_TCP={stored}")
    if args.dry_run:
        print(f"dry-run: would set input.hid_tcp={stored}")
        return 0

    path = write_settings_hid_tcp(stored, settings_path=_settings_path(args.settings) if args.settings else None)
    print(f"OK wrote {path} → input.hid_tcp={stored}")
    print("Bot same session → set GATHERION_HID_TCP=" + stored)
    print("Smoke: python tools\\hid_tcp_smoke.py --host", stored)
    if args.adb_forward:
        print("Remember: adb forward tcp:18765 tcp:8765")
    if args.rootless:
        print("Remember: python tools\\rootless_hid_bridge.py  (Interception preferred)")
        print("Cascade also uses Interception directly when driver present (no bridge needed).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
