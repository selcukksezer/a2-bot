# -*- coding: utf-8 -*-
"""Rootless (1B) KEY-protocol listener → Interception / soft inject.

Stock Android cannot expose USB HID without root. This PC-side bridge speaks the
same KEYDOWN/KEYUP/KEY/PING lines as NetworkHidAdapter / android_usb_hid_server,
and injects locally via Interception (required). Soft SendInput is ignored by
AION 2, so it is OFF by default and only available with an explicit
--allow-soft opt-in — and even then the bot is told (ERR reply) that the key
was not delivered through hardware.

Usage:
  python tools/rootless_hid_bridge.py --self-smoke
  python tools/rootless_hid_bridge.py
  python tools/rootless_hid_bridge.py --bind 127.0.0.1 --port 18766 --tcp
  python tools/rootless_hid_bridge.py --auth-token <sır>  # veya GATHERION_HID_TOKEN
  set GATHERION_HID_TCP=udp://127.0.0.1:18766
  python tools/set_hid_tcp.py --rootless
  python tools/hid_tcp_smoke.py --host udp://127.0.0.1:18766 --keys tab,1,2,3

Kimlik doğrulama (K-1, dilim 22): --auth-token / GATHERION_HID_TOKEN
ayarlıysa istemci önce `AUTH <token>` göndermelidir; doğrulanmamış
bağlantıdan/göndericiden gelen KEY/MOUSE satırları ERR ile reddedilir.
Token ayarlı değilse eski davranış korunur (uyarıyla) — Android
GatherionHidBridge APK'sının AUTH'u zorunlu kılıp kılmadığı bu repoda
doğrulanamadı (APK kodu yok), o yüzden tokensız mod kaldırılmadı.
"""
from __future__ import annotations

import argparse
import os
import secrets
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

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
    """Inject named keys via Interception.

    S-4703: soft SendInput is OFF by default (allow_soft=False). AION 2
    ignores soft input, so silently serving KEY lines through the soft
    backend lied to the bot (key_down returned True, handle_line replied
    None). Soft injection now needs an explicit opt-in, and handle_line
    reports it as an ERR reply even when the soft call "succeeds".
    """

    def __init__(self, *, allow_soft: bool = False) -> None:
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
                "WARN: Interception driver not ready and soft fallback is off "
                "(default; pass --allow-soft only to opt in explicitly); "
                "KEY lines will be acknowledged with an ERR reply, not injected.",
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

    def _inject(self, name: str, is_down: bool) -> bool:
        """Shared key_down/key_up path: normalize, resolve scancode, send.

        Previously key_down/key_up were two near-identical 13-line copies;
        a fix in one (e.g. the extended-key flag) could silently miss the
        other. This single path serves both directions.
        """
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
                bool(self._bridge.send_key(scan & 0xFF, is_down, extended=extended))
            )
        if self.backend == "soft":
            return self._record(self._soft_send(key, is_down))
        return self._record(False)

    def key_down(self, name: str) -> bool:
        return self._inject(name, True)

    def key_up(self, name: str) -> bool:
        return self._inject(name, False)

    def key_pulse(self, name: str, hold_ms: int = 80) -> bool:
        if not self.key_down(name):
            return False
        time.sleep(max(0.02, hold_ms / 1000.0))
        # A failed key_up must never leave the key stuck down: retry, then
        # try a soft SendInput release as a last-resort unstick attempt.
        # S-4701: that soft release is best-effort only and must NEVER be
        # reported as a completed pulse — AION 2 ignores soft input, so a
        # soft "success" here used to return True while backend stayed
        # "interception", making handle_line reply None (success) and the
        # bot believe the key was released while it stayed down in-game.
        for attempt in range(3):
            if self.key_up(name):
                return True
            print(
                f"key_up retry {attempt + 1}/3 for {name!r} (stuck-key guard)",
                flush=True,
            )
            time.sleep(0.05)
        if self.backend == "interception" and self.allow_soft:
            soft_ok = self._record(self._soft_send(str(name).lower().strip(), False))
            print(
                f"soft release for {name!r} ok={soft_ok} (best-effort unstick "
                "only; pulse still FAILED — Interception key_up required)",
                flush=True,
            )
        print(f"STUCK KEY RISK: key_up failed for {name!r} after retries", flush=True)
        return False

    def mouse_button(self, button: str, is_down: bool) -> bool:
        """Fare tuşu — yalnızca Interception (K-1, dilim 22: MOUSE satırı
        eskiden parse_line'da tanınmıyor, sessizce düşüyordu)."""
        if self.backend != "interception":
            return self._record(False)
        fn = getattr(self._bridge, "send_mouse_button", None)
        if not callable(fn):
            return self._record(False)
        return self._record(bool(fn(str(button or "left").strip().lower(), bool(is_down))))

    def move_absolute(self, x: int, y: int) -> bool:
        """Mutlak fare taşıma — yalnızca Interception (K-1, dilim 22)."""
        if self.backend != "interception":
            return self._record(False)
        fn = getattr(self._bridge, "send_mouse_absolute", None)
        if not callable(fn):
            return self._record(False)
        return self._record(bool(fn(int(x), int(y))))

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Mutlak taşıma + tuş bas/bırak (K-1, dilim 22: CLICK satırı)."""
        if not self.move_absolute(x, y):
            return False
        if not self.mouse_button(button, True):
            return False
        time.sleep(0.05)
        return self.mouse_button(button, False)

    def close(self) -> None:
        try:
            self._bridge.close()
        except Exception:
            pass


def parse_line(line: str) -> Tuple[str, tuple]:
    """Return (verb, args). Unknown → ('', ()).

    A KEY line with a present-but-invalid hold is ('INVALID', (line,)) so the
    caller can report it instead of silently substituting the 80 ms default.
    K-1 (dilim 22): AUTH / MOUSE / MOVE / CLICK da tanınır — eskiden bu
    fiiller sessizce düşüyordu (istemci _LineHidMixin onları gönderebiliyor).
    """
    text = (line or "").strip()
    if not text:
        return "", ()
    parts = text.split()
    verb = parts[0].upper()
    if verb == "PING":
        return "PING", ()
    if verb == "AUTH" and len(parts) >= 2:
        return "AUTH", (parts[1],)
    if verb == "AUTH":
        return "INVALID", (text,)
    if verb == "KEYDOWN" and len(parts) >= 2:
        return "KEYDOWN", (parts[1],)
    if verb == "KEYUP" and len(parts) >= 2:
        return "KEYUP", (parts[1],)
    if verb == "KEY" and len(parts) >= 2:
        if len(parts) >= 3:
            if not parts[2].isdigit():
                return "INVALID", (text,)
            hold = int(parts[2])
            if hold <= 0 or hold > 60000:
                return "INVALID", (text,)
            return "KEY", (parts[1], hold)
        return "KEY", (parts[1], 80)
    if verb == "MOUSE" and len(parts) >= 3:
        btn = parts[1].lower()
        state = parts[2].upper()
        if btn in ("left", "right", "middle", "x1", "x2") and state in ("DOWN", "UP"):
            return "MOUSE", (btn, state == "DOWN")
        return "INVALID", (text,)
    if verb == "MOVE" and len(parts) >= 3:
        try:
            return "MOVE", (int(parts[1]), int(parts[2]))
        except ValueError:
            return "INVALID", (text,)
    if verb == "CLICK" and len(parts) >= 3:
        try:
            x, y = int(parts[1]), int(parts[2])
        except ValueError:
            return "INVALID", (text,)
        btn = parts[3].lower() if len(parts) >= 4 else "left"
        if btn not in ("left", "right", "middle", "x1", "x2"):
            return "INVALID", (text,)
        return "CLICK", (x, y, btn)
    return "", ()


class BridgeAuth:
    """Per-peer auth state for the KEY protocol (K-1, dilim 22).

    Token yapılandırıldıysa (expected dolu) ilk satır `AUTH <token>`
    olmalıdır; doğrulanmamış peer'dan gelen KEY/MOUSE/MOVE/CLICK
    reddedilir. PING her zaman serbesttir (UDP canlılık yoklaması).
    Token yapılandırılmadıysa (expected boş) eski davranış korunur:
    kimlik doğrulama uygulanmaz (yalnızca başlangıç uyarısı).
    """

    def __init__(self, expected_token: str = "") -> None:
        self.expected = str(expected_token or "")
        self.authed = False

    @property
    def required(self) -> bool:
        return bool(self.expected)

    def check(self, verb: str) -> Optional[str]:
        """None = devam edilebilir; str = reddetme yanıtı."""
        if not self.required or self.authed:
            return None
        if verb in ("PING", "AUTH"):
            return None
        return "ERR auth required: send AUTH <token> first"

    def try_auth(self, token: str) -> str:
        if not self.required:
            return "ERR auth not configured on bridge (token ignored)"
        if secrets.compare_digest(str(token or ""), self.expected):
            self.authed = True
            return "OK"
        return "ERR bad auth"


def _soft_reply(verb: str, key: str) -> str:
    """S-4703: a soft "success" is not a success for AION 2.

    Soft SendInput is ignored by the game, so the bot must be told the key
    was NOT delivered through hardware — returning None here would let the
    bot believe the key landed.
    """
    return (
        f"ERR {verb} {key} backend=soft: soft SendInput is ignored by AION 2; "
        "Interception required (restart without --allow-soft, install the "
        "Interception driver)"
    )


def _dispatch(injector, line: str, auth: Optional[BridgeAuth] = None) -> Optional[str]:
    """handle_line() with the listener-thread guard: one bad line (or a
    raising injector) never kills the UDP/TCP listener. Shared by
    _serve_udp and _tcp_client, which previously each had their own
    identical try/except around handle_line."""
    try:
        return handle_line(injector, line, auth=auth)
    except Exception as exc:  # noqa: BLE001 - a bad line must not kill the listener
        # P3-8 (S-251): hata logu AUTH satırını tam yazıp tokenı sızdırırdı.
        shown = "<AUTH ...>" if isinstance(line, str) and line.strip().upper().startswith("AUTH") else line
        print(f"handle_line error for {shown!r}: {exc}", flush=True)
        return None


def handle_line(injector: LocalKeyInjector, line: str, auth: Optional[BridgeAuth] = None) -> Optional[str]:
    """Handle one KEY-protocol line.

    Returns an "ERR ..." reply when the injection failed so the remote bot
    can see the failure; returns None on success (no reply) and for PING
    returns "PONG". Failures were previously only printed locally.
    S-4703: a soft-backend "success" also returns an ERR reply (see
    _soft_reply) — the bot must never be told a soft key landed.
    K-1 (dilim 22): AUTH zorunluluğu (token yapılandırıldıysa) ve
    MOUSE/MOVE/CLICK fiilleri.
    """
    verb, args = parse_line(line)
    if verb == "PING":
        return "PONG"
    if verb == "INVALID":
        print(f"invalid KEY line rejected: {args[0]!r}", flush=True)
        return f"ERR invalid line: {args[0]}"
    if verb == "AUTH":
        if auth is None:
            # Tokensız köprü: eski sessiz yoksayma korunur (testler +
            # legacy istemciler); tokenlı köprüde try_auth cevap verir.
            return None
        return auth.try_auth(args[0])
    if auth is not None:
        denial = auth.check(verb)
        if denial is not None:
            print(f"unauthenticated {verb} rejected", flush=True)
            return denial
    if verb == "KEYDOWN":
        ok = injector.key_down(args[0])
        print(f"inject KEYDOWN {args[0]} ok={ok} backend={injector.backend}", flush=True)
        if ok and injector.backend == "soft":
            return _soft_reply("KEYDOWN", args[0])
        return None if ok else f"ERR KEYDOWN {args[0]} backend={injector.backend}"
    if verb == "KEYUP":
        ok = injector.key_up(args[0])
        print(f"inject KEYUP {args[0]} ok={ok} backend={injector.backend}", flush=True)
        if ok and injector.backend == "soft":
            return _soft_reply("KEYUP", args[0])
        return None if ok else f"ERR KEYUP {args[0]} backend={injector.backend}"
    if verb == "KEY":
        ok = injector.key_pulse(args[0], int(args[1]))
        print(f"inject KEY {args[0]} ok={ok} backend={injector.backend}", flush=True)
        if ok and injector.backend == "soft":
            return _soft_reply("KEY", args[0])
        return None if ok else f"ERR KEY {args[0]} backend={injector.backend}"
    if verb == "MOUSE":
        fn = getattr(injector, "mouse_button", None)
        ok = bool(fn(args[0], args[1])) if callable(fn) else False
        print(f"inject MOUSE {args[0]} {'DOWN' if args[1] else 'UP'} ok={ok} backend={injector.backend}", flush=True)
        return None if ok else f"ERR MOUSE {args[0]} backend={injector.backend}"
    if verb == "MOVE":
        fn = getattr(injector, "move_absolute", None)
        ok = bool(fn(args[0], args[1])) if callable(fn) else False
        print(f"inject MOVE {args[0]},{args[1]} ok={ok} backend={injector.backend}", flush=True)
        return None if ok else f"ERR MOVE {args[0]},{args[1]} backend={injector.backend}"
    if verb == "CLICK":
        fn = getattr(injector, "click", None)
        ok = bool(fn(args[0], args[1], args[2])) if callable(fn) else False
        print(f"inject CLICK {args[0]},{args[1]} {args[2]} ok={ok} backend={injector.backend}", flush=True)
        return None if ok else f"ERR CLICK {args[0]},{args[1]} backend={injector.backend}"
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
    # A soft SendInput pass is NOT a 1B pass: soft input is ignored by
    # AION 2, so reporting SMOKE_OK on the soft backend would be a lie.
    if ok_down and ok_up and injected >= 2:
        print(
            f"RESULT: 1B_SMOKE_FAIL (soft fallback backend={backend}; "
            "Interception required)",
            flush=True,
        )
        return 5
    print("RESULT: 1B_SMOKE_FAIL", flush=True)
    return 5


def _serve_tcp(
    host: str,
    port: int,
    injector: LocalKeyInjector,
    stop: threading.Event,
    ready: Optional[threading.Event] = None,
    error_box: Optional[list] = None,
    auth_token: str = "",
) -> None:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind((host, port))
        srv.listen(8)
    except OSError as exc:
        print(f"TCP bind FAIL {host}:{port}: {exc}", flush=True)
        if error_box is not None:
            error_box.append(exc)
        srv.close()
        stop.set()
        return
    srv.settimeout(0.5)
    print(f"TCP listen {host}:{port}", flush=True)
    if ready is not None:
        ready.set()
    try:
        while not stop.is_set():
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            print(f"TCP client {addr}", flush=True)
            threading.Thread(
                target=_tcp_client,
                args=(conn, injector, stop, auth_token),
                daemon=True,
            ).start()
    finally:
        srv.close()


def _tcp_client(
    conn: socket.socket,
    injector: LocalKeyInjector,
    stop: threading.Event,
    auth_token: str = "",
) -> None:
    buf = b""
    conn.settimeout(0.5)
    auth = BridgeAuth(auth_token)
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
                reply = _dispatch(injector, line, auth=auth)
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
    auth_token: str = "",
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
    # K-1: UDP'de kimlik gönderici adresine bağlı tutulur (best-effort;
    # UDP spoof edilebilir — kimlik doğrulama gerekiyorsa TCP önerilir).
    authed_peers: Dict[tuple, BridgeAuth] = {}
    try:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            auth = authed_peers.get(addr)
            if auth is None:
                auth = BridgeAuth(auth_token)
                authed_peers[addr] = auth
            for raw in data.split(b"\n"):
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                reply = _dispatch(injector, line, auth=auth)
                if reply:
                    try:
                        sock.sendto((reply + "\n").encode("utf-8"), addr)
                    except OSError:
                        pass
    finally:
        sock.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Rootless KEY bridge → Interception (not real USB HID)"
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
        help="Refuse soft SendInput fallback if Interception unavailable "
        "(this is now the default; kept for compatibility)",
    )
    ap.add_argument(
        "--allow-soft",
        action="store_true",
        help="S-4703 opt-in ONLY: allow soft SendInput when Interception is "
        "unavailable. Soft input is ignored by AION 2, so the bot is told "
        "via an ERR reply that the key was not delivered through hardware.",
    )
    ap.add_argument(
        "--status-only",
        action="store_true",
        help="Print injector status and exit (no listen)",
    )
    ap.add_argument(
        "--auth-token",
        default=os.environ.get("GATHERION_HID_TOKEN", ""),
        help="K-1 (dilim 22): AUTH token'ı — ayarlıysa istemci önce "
        "`AUTH <token>` göndermelidir, yoksa KEY/MOUSE satırları ERR ile "
        "reddedilir. Verilmezse GATHERION_HID_TOKEN ortam değişkeni okunur; "
        "hiçbiri yoksa eski tokensız davranış korunur (uyarıyla).",
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
            allow_soft=args.allow_soft and not args.no_soft,
            key=str(args.smoke_key).lower().strip() or "tab",
        )

    injector = LocalKeyInjector(allow_soft=args.allow_soft and not args.no_soft)
    if args.status_only:
        print(f"STATUS backend={injector.backend}")
        return 0 if injector.backend != "none" else 1

    stop = threading.Event()
    udp_ready = threading.Event()
    udp_error: list = []
    auth_token = str(args.auth_token or "")
    if auth_token:
        print("AUTH enforced: unauthenticated KEY/MOUSE lines are rejected", flush=True)
    else:
        print(
            "WARNING: no --auth-token / GATHERION_HID_TOKEN — bridge accepts "
            "KEY lines from anyone (legacy mode, K-1 dilim 22)",
            flush=True,
        )
    udp_thread = threading.Thread(
        target=_serve_udp,
        args=(args.bind, args.port, injector, stop, udp_ready, udp_error, auth_token),
        daemon=True,
    )
    udp_thread.start()
    if not udp_ready.wait(3.0):
        detail = udp_error[0] if udp_error else "timeout waiting for bind"
        print(f"FAIL: UDP listen not ready ({detail})", flush=True)
        injector.close()
        return 2
    tcp_ready = threading.Event()
    tcp_error: list = []
    tcp_thread = None
    if args.tcp:
        tcp_thread = threading.Thread(
            target=_serve_tcp,
            args=(args.bind, args.port, injector, stop, tcp_ready, tcp_error, auth_token),
            daemon=True,
        )
        tcp_thread.start()
        if not tcp_ready.wait(3.0):
            detail = tcp_error[0] if tcp_error else "timeout waiting for bind"
            print(f"FAIL: TCP listen not ready ({detail})", flush=True)
            stop.set()
            injector.close()
            return 2

    scheme = "udp+tcp" if args.tcp else "udp"
    ep = f"udp://{args.bind}:{args.port}"
    print(f"Ready ({scheme}). Bot: set GATHERION_HID_TCP={ep}", flush=True)
    print(
        "Honest: stock phone is NOT USB HID here — inject is Interception on PC "
        "(soft SendInput is off by default; AION 2 ignores it).",
        flush=True,
    )
    try:
        while True:
            time.sleep(0.5)
            # A dead listener must not leave main printing Ready forever.
            if not udp_thread.is_alive() and not stop.is_set():
                print("FAIL: UDP listener thread died; stopping.", flush=True)
                stop.set()
                injector.close()
                return 2
            if tcp_thread is not None and not tcp_thread.is_alive() and not stop.is_set():
                print("FAIL: TCP listener thread died; stopping.", flush=True)
                stop.set()
                injector.close()
                return 2
    except KeyboardInterrupt:
        print("Stopping…", flush=True)
        stop.set()
        time.sleep(0.6)
        injector.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
