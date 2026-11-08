# -*- coding: utf-8 -*-
"""Rootless (1B) KEY-protocol listener → Interception / soft inject.

Stock Android cannot expose USB HID without root. This PC-side bridge speaks the
same KEYDOWN/KEYUP/KEY/PING lines as NetworkHidAdapter / android_usb_hid_server,
and injects locally via Interception (preferred) or Soft SendInput (last resort).

Usage:
  python tools/rootless_hid_bridge.py --self-smoke --no-soft
  python tools/rootless_hid_bridge.py
  python tools/rootless_hid_bridge.py --bind 127.0.0.1 --port 18766 --tcp
  set GATHERION_HID_TCP=udp://127.0.0.1:18766
  python tools/set_hid_tcp.py --rootless
  python tools/hid_tcp_smoke.py --host udp://127.0.0.1:18766 --keys tab,1,2,3

Does NOT install Interception. Does NOT claim real USB HID from the phone.
When Interception driver is present, InputController cascade also uses it directly
(GATHERION_HID_TCP bridge optional for same KEY protocol / one-config path).
"""
from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 18766


def _load_injectors():
    import ctypes

    from core.input_driver import (
        EXTENDED_KEYS,
        HARDWARE_SCANCODES,
        INPUT_KEYBOARD,
        KEYEVENTF_EXTENDEDKEY,
        KEYEVENTF_KEYUP,
        KEYEVENTF_SCANCODE,
        InnerInputUnion,
        InputStruct,
        InterceptionBridge,
        KeyBdInput,
    )

    return {
        "ctypes": ctypes,
        "scancodes": HARDWARE_SCANCODES,
        "extended": EXTENDED_KEYS,
        "InterceptionBridge": InterceptionBridge,
        "KEYEVENTF_EXTENDEDKEY": KEYEVENTF_EXTENDEDKEY,
        "KEYEVENTF_KEYUP": KEYEVENTF_KEYUP,
        "KEYEVENTF_SCANCODE": KEYEVENTF_SCANCODE,
        "INPUT_KEYBOARD": INPUT_KEYBOARD,
        "InnerInputUnion": InnerInputUnion,
        "InputStruct": InputStruct,
        "KeyBdInput": KeyBdInput,
    }


class LocalKeyInjector:
    """Inject named keys via Interception if ready, else soft SendInput."""

    def __init__(self, *, allow_soft: bool = True) -> None:
        deps = _load_injectors()
        self._deps = deps
        self._scancodes = deps["scancodes"]
        self._extended = deps["extended"]
        self._bridge = deps["InterceptionBridge"]()
        self.allow_soft = bool(allow_soft)
        self.backend = "none"
        self.inject_ok = 0
        self.inject_fail = 0
        if self._bridge.available:
            self.backend = "interception"
        elif self.allow_soft:
            self.backend = "soft"
        print(
            f"injector backend={self.backend} "
            f"interception_available={self._bridge.available} "
            f"dll={self._bridge.dll_path or '-'}",
            flush=True,
        )
        if self.backend == "none":
            print(
                "WARN: Interception driver not ready and --no-soft set; "
                "KEY lines will be acknowledged but not injected.",
                flush=True,
            )

    def _record(self, ok: bool) -> bool:
        if ok:
            self.inject_ok += 1
        else:
            self.inject_fail += 1
        return ok

    def _soft_send(self, key: str, is_down: bool) -> bool:
        scan = int(self._scancodes.get(key, 0) or 0)
        if scan <= 0:
            print(f"unknown key: {key!r}", flush=True)
            return False
        d = self._deps
        ctypes = d["ctypes"]
        flags = d["KEYEVENTF_SCANCODE"]
        if not is_down:
            flags |= d["KEYEVENTF_KEYUP"]
        if key in self._extended or scan > 0xFF:
            flags |= d["KEYEVENTF_EXTENDEDKEY"]
        extra = ctypes.c_ulong(0)
        ii = d["InnerInputUnion"]()
        ii.ki = d["KeyBdInput"](0, scan & 0xFF, flags, 0, ctypes.pointer(extra))
        inp = d["InputStruct"](ctypes.c_ulong(d["INPUT_KEYBOARD"]), ii)
        return ctypes.windll.user32.SendInput(1, ctypes.pointer(inp), ctypes.sizeof(inp)) == 1

    def key_down(self, name: str) -> bool:
        key = str(name or "").lower().strip()
        if not key:
            return self._record(False)
        if self.backend == "interception":
            scan = int(self._scancodes.get(key, 0) or 0)
            if scan <= 0:
                print(f"unknown key: {name!r}", flush=True)
                return self._record(False)
            extended = key in self._extended or scan > 0xFF
            return self._record(
                bool(self._bridge.send_key(scan & 0xFF, True, extended=extended))
            )
        if self.backend == "soft":
            return self._record(self._soft_send(key, True))
        return self._record(False)

    def key_up(self, name: str) -> bool:
        key = str(name or "").lower().strip()
        if not key:
            return self._record(False)
        if self.backend == "interception":
            scan = int(self._scancodes.get(key, 0) or 0)
            if scan <= 0:
                return self._record(False)
            extended = key in self._extended or scan > 0xFF
            return self._record(
                bool(self._bridge.send_key(scan & 0xFF, False, extended=extended))
            )
        if self.backend == "soft":
            return self._record(self._soft_send(key, False))
        return self._record(False)

    def key_pulse(self, name: str, hold_ms: int = 80) -> bool:
        if not self.key_down(name):
            return False
        time.sleep(max(0.02, hold_ms / 1000.0))
        return self.key_up(name)

    def close(self) -> None:
        try:
            self._bridge.close()
        except Exception:
            pass


def parse_line(line: str) -> Tuple[str, tuple]:
    """Return (verb, args). Unknown → ('', ())."""
    text = (line or "").strip()
    if not text:
        return "", ()
    parts = text.split()
    verb = parts[0].upper()
    if verb == "PING":
        return "PING", ()
    if verb == "KEYDOWN" and len(parts) >= 2:
        return "KEYDOWN", (parts[1],)
    if verb == "KEYUP" and len(parts) >= 2:
        return "KEYUP", (parts[1],)
    if verb == "KEY" and len(parts) >= 2:
        hold = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else 80
        return "KEY", (parts[1], hold)
    return "", ()


def handle_line(injector: LocalKeyInjector, line: str) -> Optional[str]:
    verb, args = parse_line(line)
    if verb == "PING":
        return "PONG"
    if verb == "KEYDOWN":
        ok = injector.key_down(args[0])
        print(f"inject KEYDOWN {args[0]} ok={ok} backend={injector.backend}", flush=True)
        return None
    if verb == "KEYUP":
        ok = injector.key_up(args[0])
        print(f"inject KEYUP {args[0]} ok={ok} backend={injector.backend}", flush=True)
        return None
    if verb == "KEY":
        ok = injector.key_pulse(args[0], int(args[1]))
        print(f"inject KEY {args[0]} ok={ok} backend={injector.backend}", flush=True)
        return None
    return None


def run_self_smoke(
    host: str,
    port: int,
    *,
    allow_soft: bool,
    key: str = "tab",
) -> int:
    """One-shot: bind UDP → NetworkHidAdapter KEYDOWN/UP → Interception inject → exit."""
    from core.hardware_input_port import NetworkHidAdapter

    injector = LocalKeyInjector(allow_soft=allow_soft)
    if injector.backend != "interception" and not allow_soft:
        print("FAIL: Interception required (--no-soft) but backend != interception", flush=True)
        injector.close()
        return 2
    if injector.backend == "none":
        print("FAIL: no inject backend", flush=True)
        injector.close()
        return 2

    stop = threading.Event()
    ready = threading.Event()
    error_box: list = []
    threading.Thread(
        target=_serve_udp,
        args=(host, port, injector, stop, ready, error_box),
        daemon=True,
    ).start()
    if not ready.wait(3.0):
        stop.set()
        detail = error_box[0] if error_box else "timeout waiting for bind"
        print(f"FAIL: UDP listen not ready ({detail})", flush=True)
        injector.close()
        return 3

    ep = f"udp://{host}:{port}"
    print(f"self-smoke endpoint={ep} backend={injector.backend}", flush=True)
    hid = NetworkHidAdapter(endpoint=ep)
    if not hid.available:
        print(f"FAIL: NetworkHidAdapter unavailable {ep}", flush=True)
        stop.set()
        injector.close()
        return 4

    backend = injector.backend
    before = injector.inject_ok
    ok_down = hid.key_down(key)
    time.sleep(0.08)
    ok_up = hid.key_up(key)
    hid.close()
    # UDP recv/inject is on another thread — wait for counters, not just sendto.
    deadline = time.time() + 2.0
    while time.time() < deadline and injector.inject_ok < before + 2:
        time.sleep(0.05)
    injected = injector.inject_ok - before
    stop.set()
    time.sleep(0.15)
    injector.close()

    print(
        f"EVIDENCE: 1B NetworkHidAdapter KEYDOWN/UP {key} "
        f"send_down={ok_down} send_up={ok_up} injected={injected} "
        f"backend={backend}",
        flush=True,
    )
    # Require real inject via this process (not UDP sendto-only false positive).
    if ok_down and ok_up and injected >= 2 and backend == "interception":
        print("RESULT: 1B_SMOKE_OK (Interception)", flush=True)
        return 0
    if ok_down and ok_up and injected >= 2:
        print(f"RESULT: 1B_SMOKE_OK (backend={backend})", flush=True)
        return 0
    print("RESULT: 1B_SMOKE_FAIL", flush=True)
    return 5


def _serve_tcp(host: str, port: int, injector: LocalKeyInjector, stop: threading.Event) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)
    srv.settimeout(0.5)
    print(f"TCP listen {host}:{port}", flush=True)
    try:
        while not stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print(f"TCP client {addr}", flush=True)
            threading.Thread(
                target=_tcp_client,
                args=(conn, injector, stop),
                daemon=True,
            ).start()
    finally:
        srv.close()


def _tcp_client(conn: socket.socket, injector: LocalKeyInjector, stop: threading.Event) -> None:
    buf = b""
    conn.settimeout(0.5)
    try:
        while not stop.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                reply = handle_line(injector, line)
                if reply:
                    try:
                        conn.sendall((reply + "\n").encode("utf-8"))
                    except OSError:
                        return
    finally:
        try:
            conn.close()
        except OSError:
            pass


def _serve_udp(
    host: str,
    port: int,
    injector: LocalKeyInjector,
    stop: threading.Event,
    ready: Optional[threading.Event] = None,
    error_box: Optional[list] = None,
) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        print(f"UDP bind FAIL {host}:{port}: {exc}", flush=True)
        if error_box is not None:
            error_box.append(exc)
        sock.close()
        stop.set()
        return
    sock.settimeout(0.5)
    print(f"UDP listen {host}:{port}", flush=True)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            for raw in data.split(b"\n"):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                reply = handle_line(injector, line)
                if reply:
                    try:
                        sock.sendto((reply + "\n").encode("utf-8"), addr)
                    except OSError:
                        pass
    finally:
        sock.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rootless KEY bridge → Interception/soft (not real USB HID)"
    )
    ap.add_argument("--bind", default=DEFAULT_HOST)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument(
        "--tcp",
        action="store_true",
        help="Also listen TCP on same port (UDP always on by default)",
    )
    ap.add_argument(
        "--no-soft",
        action="store_true",
        help="Refuse soft SendInput fallback if Interception unavailable",
    )
    ap.add_argument(
        "--status-only",
        action="store_true",
        help="Print injector status and exit (no listen)",
    )
    ap.add_argument(
        "--self-smoke",
        action="store_true",
        help="One-shot: listen + NetworkHidAdapter KEYDOWN/UP tab + exit (proves 1B)",
    )
    ap.add_argument(
        "--smoke-key",
        default="tab",
        help="Key name for --self-smoke (default tab)",
    )
    args = ap.parse_args()

    if args.self_smoke:
        return run_self_smoke(
            args.bind,
            args.port,
            allow_soft=not args.no_soft,
            key=str(args.smoke_key).lower().strip() or "tab",
        )

    injector = LocalKeyInjector(allow_soft=not args.no_soft)
    if args.status_only:
        print(f"STATUS backend={injector.backend}")
        return 0 if injector.backend != "none" else 1

    stop = threading.Event()
    threading.Thread(
        target=_serve_udp,
        args=(args.bind, args.port, injector, stop),
        daemon=True,
    ).start()
    if args.tcp:
        threading.Thread(
            target=_serve_tcp,
            args=(args.bind, args.port, injector, stop),
            daemon=True,
        ).start()

    scheme = "udp+tcp" if args.tcp else "udp"
    ep = f"udp://{args.bind}:{args.port}"
    print(f"Ready ({scheme}). Bot: set GATHERION_HID_TCP={ep}", flush=True)
    print(
        "Honest: stock phone is NOT USB HID here — inject is Interception/soft on PC.",
        flush=True,
    )
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Stopping…", flush=True)
        stop.set()
        time.sleep(0.6)
        injector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
