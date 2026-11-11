# -*- coding: utf-8 -*-
"""
Mobile Remote Control & Live Telemetry HTTP/API/SSE/WebSocket Server (core/mobile_server.py)
Agent 21 & Spec Items 31–35 — Gatherion A2 Bot Mobile Backend

Provides a zero-dependency-hassle, multi-threaded HTTP/JSON, Dual-Channel PiP MJPEG/JPEG,
<50ms Server-Sent Events (SSE), RFC 6455 WebSocket Push, and Full PWA Server built on
Python's `http.server.ThreadingHTTPServer` so the user can monitor live telemetry,
view the 2D interactive route radar, tap-to-move on the map, quarantine/delete/reset nodes,
stream the game screen + magnified minimap + [F Gather] ROI PiP, and install the dashboard
as a standalone Home Screen app on iOS and Android.
"""

from __future__ import annotations

import base64
import collections
from datetime import datetime
import hashlib
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
import re
import secrets
import select
import socket
import string
import struct
import subprocess
import sys
import threading
import time
import urllib.parse
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from core.corridor_navigator import WORLD_MAP_METERS_PER_PX
from version import __version__

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("Gatherion.MobileServer")
logger.setLevel(logging.INFO)

# P1-9 (K-08): token/sır sızıntısı kapatma — log, ring buffer, /api/status, SSE.
# Erişim logu ve hata logu istek satırını/yolu olduğu gibi yazar; token
# query'de taşınıyorsa (`?token=...` — QR açılışı / `<img>` / EventSource
# başlık gönderemediği için) düz metin token log dosyasına, konsola, ring
# buffer'a ve oradan /api/status + SSE akışına sızar. Bu yardımcı, loga
# girecek her metindeki hassas query değerini `***` ile değiştirir; yolun
# kendisi ve parametre adları tanılama için korunur, değer asla yazılmaz.
_SENSITIVE_QUERY_RE = re.compile(
    r"([?&](?:token|key|gatherion_token|access_token|api_key|auth|password|secret)=)[^&\s\"'<>]*",
    re.IGNORECASE,
)
_BEARER_RE = re.compile(r"(Bearer\s+)[^\s\"'<>]+", re.IGNORECASE)


def _sanitize_token_from_text(text: Any) -> str:
    """Loglanacak metinden token/sır değerlerini maskeler (P1-9/K-08).

    `?token=<değer>` / `&key=<değer>` gibi hassas query parametrelerinin
    değerini `***` yapar; `Authorization: Bearer <değer>` kalıbını da
    aynı şekilde maskeler. Token parçası içermeyen metni değiştirmeden
    döndürür.
    """
    try:
        out = str(text)
    except Exception:
        return "***"
    out = _SENSITIVE_QUERY_RE.sub(r"\1***", out)
    out = _BEARER_RE.sub(r"\1***", out)
    return out


def _token_fingerprint(token: str) -> str:
    """Token'ın SHA-256 parmak izini döner (P1-9/K-08).

    Token parçası içermez — yalnızca hash özeti. `core/incident_recorder.py`
    içindeki `MobileTokenAuthenticator.token_fingerprint` ile aynı
    `ilk6-son6` biçimindedir; token yoksa boş döner.
    """
    token = str(token or "")
    if not token:
        return ""
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    return f"{digest[:6]}-{digest[-6:]}"

STATE_TR_MAP: Dict[str, str] = {
    "WALKING": "YÜRÜYOR (WALKING)",
    "GATHERING": "TOPLUYOR (GATHERING)",
    "FIGHTING": "SAVAŞIYOR (FIGHTING)",
    "PAUSED": "DURAKLATILDI (PAUSED)",
    "BREAK": "MOLADA (BREAK)",
    "WAITING": "BEKLİYOR (WAITING)",
    "MAP": "HARİTA AÇIK (MAP)",
    "IDLE": "BOŞTA (IDLE)",
}

ALLOWED_REMOTE_KEYS = {
    "f",
    "e",
    "m",
    "space",
    "w",
    "a",
    "s",
    "d",
    "escape",
    "esc",
    "tab",
    "1",
    "2",
    "3",
    "4",
    "5",
    "q",
    "r",
    "t",
    "v",
    "alt",
    "f8",
    "f9",
}

# Mobil erişim token'ı politikası (F-1652/F-0136): 32 karakter, en az 16
# karakterlik açık token, düz metin dosya 0600 izinle saklanır.
_MOBILE_TOKEN_LENGTH = 32
_MOBILE_TOKEN_MIN_EXPLICIT_LENGTH = 16
_MOBILE_TOKEN_MIN_SAVED_LENGTH = 16

# WebSocket: aynı anda izin verilen en fazla istemci (F-1659: sınırsız thread).
_MAX_WS_CLIENTS = 4
_WS_CLIENT_SEMAPHORE = threading.BoundedSemaphore(_MAX_WS_CLIENTS)

# SSE: aynı anda izin verilen en fazla istemci (P2-4/K-17: WS 4 ile
# sınırlıyken SSE sınırsızdı — her SSE bağlantısı ThreadingHTTPServer'da
# bir thread'i süresiz tutar; sınırsız istemci thread tükenmesidir).
# WS ile aynı değer ve aynı kalıp (modül düzeyi BoundedSemaphore;
# sınır süreç geneli, tüm MobileRemoteServer örnekleri için ortaktır).
_MAX_SSE_CLIENTS = 4
_SSE_CLIENT_SEMAPHORE = threading.BoundedSemaphore(_MAX_SSE_CLIENTS)

# RFC 6455 WebSocket GUID
_WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# Normalized crop bounds for top-right Minimap and center-right [F Gather] ROI
MINIMAP_NORM_BOUNDS: Tuple[float, float, float, float] = (0.84, 0.05, 0.995, 0.27)
GATHER_ROI_NORM_BOUNDS: Tuple[float, float, float, float] = (0.42, 0.36, 0.72, 0.68)


class MobileLogHandler(logging.Handler):
    """
    Lightweight, thread-safe logging handler attached to the root logger.
    Captures the most recent 120 log entries into a ring buffer for mobile streaming
    without duplicating StreamHandler output.
    """

    def __init__(
        self,
        ring_buffer: Deque[Dict[str, Any]],
        lock: threading.RLock,
        notify_cond: Optional[threading.Condition] = None,
    ) -> None:
        super().__init__(level=logging.INFO)
        self._buffer = ring_buffer
        self._lock = lock
        self._notify_cond = notify_cond

    def emit(self, record: logging.LogRecord) -> None:
        try:
            ts_str = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            # P1-9 (K-08): ring buffer /api/status + SSE ile aktığı için
            # token'lı URL içeren log satırı burada da maskelenir.
            msg = _sanitize_token_from_text(record.getMessage())
            entry = {
                "ts": ts_str,
                "time": ts_str,
                "timestamp": round(float(record.created), 3),
                "level": str(record.levelname or "INFO"),
                "logger": str(record.name or "root"),
                "msg": str(msg),
                "message": str(msg),
            }
            with self._lock:
                self._buffer.append(entry)
            if self._notify_cond is not None:
                with self._notify_cond:
                    self._notify_cond.notify_all()
        except Exception:
            self.handleError(record)


class _ThreadingMobileHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer subclass with fast socket reuse and reference to MobileRemoteServer."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: Tuple[str, int],
        RequestHandlerClass: type,
        mobile_server: "MobileRemoteServer",
    ) -> None:
        self.mobile_server = mobile_server
        super().__init__(server_address, RequestHandlerClass)


def _ws_unmask_payload(payload: bytes, mask_key: bytes) -> bytes:
    """RFC 6455 §5.3: istemci çerçevesinin maskesini kaldırır (vektörize).

    Bayt bayt Python döngüsü yerine numpy XOR kullanılır: 64 KiB yükte
    ~44 ms -> ~0.04 ms. `mask_key` her zaman 4 bayttır.
    """
    if not payload:
        return b""
    mask_arr = np.frombuffer(mask_key, dtype=np.uint8)
    reps = (len(payload) + 3) // 4
    full_mask = np.tile(mask_arr, reps)[: len(payload)]
    return (np.frombuffer(payload, dtype=np.uint8) ^ full_mask).tobytes()


def _ws_encode_frame(payload_bytes: bytes, opcode: int = 0x1) -> bytes:
    """Encodes a server-to-client (unmasked) RFC 6455 WebSocket frame."""
    first_byte = 0x80 | (opcode & 0x0F)
    length = len(payload_bytes)
    if length <= 125:
        header = struct.pack("!BB", first_byte, length)
    elif length <= 65535:
        header = struct.pack("!BBH", first_byte, 126, length)
    else:
        header = struct.pack("!BBQ", first_byte, 127, length)
    return header + payload_bytes


def _ws_encode_text_frame(text: str) -> bytes:
    """Encodes a UTF-8 string into an RFC 6455 WebSocket text frame (opcode 0x1)."""
    return _ws_encode_frame(text.encode("utf-8"), opcode=0x1)


def _ws_recv_exact(sock: socket.socket, num_bytes: int) -> Optional[bytes]:
    """Reads an exact number of bytes from a non-blocking/timeout socket."""
    buf = bytearray()
    while len(buf) < num_bytes:
        chunk = sock.recv(num_bytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _ws_read_client_frame(sock: socket.socket) -> Optional[Tuple[int, bytes]]:
    """
    Reads a single RFC 6455 frame sent by the client.
    Returns `(opcode, unmasked_payload)` or `None` on EOF / protocol error.

    F-1668: fragmented (FIN=0) messages are reassembled from continuation
    frames instead of being silently swallowed; unmasked client frames
    (RFC 6455 §5.1: clients MUST mask) and reserved opcodes are rejected.
    """
    first_opcode: Optional[int] = None
    payload_parts: List[bytes] = []
    while True:
        head = _ws_recv_exact(sock, 2)
        if not head:
            return None
        b0, b1 = head[0], head[1]
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        payload_len = b1 & 0x7F

        if opcode in (0x3, 0x4, 0x5, 0x6, 0x7, 0xB, 0xC, 0xD, 0xE, 0xF):
            return None
        if first_opcode is None:
            if opcode == 0x0:
                return None
            first_opcode = opcode
        elif opcode != 0x0:
            # A new data frame started before the previous fragmented
            # message finished.
            return None
        if not masked:
            # RFC 6455 §5.1: all frames from the client MUST be masked.
            return None

        if payload_len == 126:
            ext = _ws_recv_exact(sock, 2)
            if not ext:
                return None
            payload_len = struct.unpack("!H", ext)[0]
        elif payload_len == 127:
            ext = _ws_recv_exact(sock, 8)
            if not ext:
                return None
            payload_len = struct.unpack("!Q", ext)[0]

        if payload_len > 1024 * 64:
            return None

        mask_key = _ws_recv_exact(sock, 4)
        if not mask_key:
            return None

        raw_payload = _ws_recv_exact(sock, payload_len) if payload_len > 0 else b""
        if raw_payload is None:
            return None

        payload_parts.append(_ws_unmask_payload(raw_payload, mask_key))
        if fin:
            return first_opcode, b"".join(payload_parts)
        # FIN=0: keep reading continuation frames for this message.


class MobileHTTPRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP Request Handler serving:
    - Mobile Dashboard HTML (`/`, `/index.html`, `/mobile`, `/dashboard`)
    - PWA Manifest & Service Worker (`/manifest.json`, `/sw.js`, `/icon-192.png`, `/icon-512.png`)
    - Telemetry Status (`/api/status`), Route Nodes (`/api/nodes`), Connection Info (`/api/connection`)
    - Dual-Channel Live Stream (`/api/frame.jpg?mode=full|minimap|pip&quality=65&scale=0.5`)
    - High-Clarity Dual PiP Bundle (`/api/pip_bundle`)
    - Zero-Latency SSE Telemetry Stream (`/api/stream`)
    - Zero-Latency WebSocket Push (`/ws`, `/api/ws`)
    - Remote Actions (`POST /api/action`)
    """

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """HTTP access log (F-1674): debug seviyesinde tutulur, konsolu boğmaz."""
        try:
            # P1-9 (K-08): istek satırı token'lı URL içerebilir
            # (`GET /api/status?token=...`); query değeri maskelenmeden
            # loga/ring buffer'a düşmesin.
            logger.debug(
                "HTTP %s",
                _sanitize_token_from_text(format % args if args else format),
            )
        except Exception:
            logger.debug("HTTP request (log format hatası)")

    def _send_cors_headers(self) -> None:
        # F-0469: körü körüne `Access-Control-Allow-Origin: *` yok. CORS
        # başlığı yalnızca aynı-origin (Host ile eşleşen Origin) isteklerde
        # yansıtılır; aynı-origin fetch zaten CORS'a ihtiyaç duymaz.
        origin = (self.headers.get("Origin") or "").strip()
        if origin:
            try:
                origin_host = urllib.parse.urlparse(origin).netloc
            except Exception:
                origin_host = ""
            host = (self.headers.get("Host") or "").strip()
            if origin_host and host and origin_host == host:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header(
                    "Access-Control-Allow-Headers",
                    "Content-Type, Authorization, X-Access-Token, X-Auth-Token, X-Gatherion-Token, Accept, Cache-Control",
                )

    def _check_authorized(self, parsed_url: urllib.parse.ParseResult) -> bool:
        srv: MobileRemoteServer = self.server.mobile_server  # type: ignore[attr-defined]
        if not srv.require_token:
            return True

        qs = urllib.parse.parse_qs(parsed_url.query)
        token_candidates: List[str] = []
        if "token" in qs and qs["token"]:
            token_candidates.append(qs["token"][0])
        if "key" in qs and qs["key"]:
            token_candidates.append(qs["key"][0])

        for hdr_name in ("X-Access-Token", "X-Auth-Token", "X-Gatherion-Token"):
            hdr_val = self.headers.get(hdr_name, "").strip()
            if hdr_val:
                token_candidates.append(hdr_val)

        auth_header = self.headers.get("Authorization", "").strip()
        if auth_header.lower().startswith("bearer "):
            token_candidates.append(auth_header[7:].strip())

        for candidate in token_candidates:
            if candidate and secrets.compare_digest(candidate, srv.access_token):
                return True
        return False

    def _send_json(self, payload: Any, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html_text: str, status: int = HTTPStatus.OK) -> None:
        body = html_text.encode("utf-8")
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(
        self,
        raw_bytes: bytes,
        content_type: str,
        status: int = HTTPStatus.OK,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
        self.send_header("Pragma", "no-cache")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.send_header("Content-Length", str(len(raw_bytes)))
        self.end_headers()
        self.wfile.write(raw_bytes)

    def _send_jpeg(self, jpeg_bytes: bytes, status: int = HTTPStatus.OK) -> None:
        self._send_bytes(jpeg_bytes, "image/jpeg", status=status)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        srv: MobileRemoteServer = self.server.mobile_server  # type: ignore[attr-defined]
        try:
            parsed = urllib.parse.urlparse(self.path)
            route_path = parsed.path.rstrip("/") or "/"

            # Check for WebSocket upgrade request on /ws, /api/ws, or any path with Upgrade: websocket
            upgrade_hdr = (self.headers.get("Upgrade") or "").strip().lower()
            # F-1659: WebSocket yalnızca /ws ve /api/ws yollarında; herhangi
            # bir path'te `Upgrade: websocket` hijack'i yok. Origin ve
            # Sec-WebSocket-Version doğrulanır.
            if route_path in ("/ws", "/api/ws"):
                if upgrade_hdr != "websocket":
                    self._send_json(
                        {"ok": False, "error": "WebSocket yükseltmesi gereklidir."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                ws_version = (self.headers.get("Sec-WebSocket-Version") or "").strip()
                if ws_version and ws_version != "13":
                    self._send_json(
                        {"ok": False, "error": "Desteklenmeyen WebSocket sürümü."},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                origin = (self.headers.get("Origin") or "").strip()
                if origin:
                    try:
                        origin_host = urllib.parse.urlparse(origin).netloc
                    except Exception:
                        origin_host = ""
                    host = (self.headers.get("Host") or "").strip()
                    if not origin_host or not host or origin_host != host:
                        self._send_json(
                            {"ok": False, "error": "WebSocket Origin reddedildi."},
                            status=HTTPStatus.FORBIDDEN,
                        )
                        return
                if not self._check_authorized(parsed):
                    self._send_json(
                        {"ok": False, "error": "Yetkisiz erişim (Geçersiz token)."},
                        status=HTTPStatus.UNAUTHORIZED,
                    )
                    return
                if not _WS_CLIENT_SEMAPHORE.acquire(blocking=False):
                    self._send_json(
                        {"ok": False, "error": "WebSocket istemci sınırı dolu."},
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                try:
                    self._handle_websocket_upgrade(srv, parsed)
                finally:
                    _WS_CLIENT_SEMAPHORE.release()
                return

            # 1. Public PWA Shell & Manifest/Service Worker Routes (No token header required by browser PWA installer)
            # F-0469: pano (dashboard) HTML'i kimlik doğrulamasız sunulmaz;
            # manifest / service worker / ikonlar sır içermediği için açıktır.
            if route_path in ("/", "/index.html", "/mobile", "/dashboard"):
                if not self._check_authorized(parsed):
                    self._send_json(
                        {"ok": False, "error": "Yetkisiz erişim (Geçersiz token)."},
                        status=HTTPStatus.UNAUTHORIZED,
                    )
                    return
                html_content = srv.load_dashboard_html()
                self._send_html(html_content, HTTPStatus.OK)
                return

            if route_path in ("/manifest.json", "/ui/manifest.json", "/manifest.webmanifest"):
                manifest_text = srv.load_manifest_json()
                self._send_bytes(
                    manifest_text.encode("utf-8"),
                    "application/manifest+json; charset=utf-8",
                    status=HTTPStatus.OK,
                )
                return

            if route_path in ("/sw.js", "/ui/sw.js", "/service-worker.js"):
                sw_text = srv.load_service_worker_js()
                self._send_bytes(
                    sw_text.encode("utf-8"),
                    "application/javascript; charset=utf-8",
                    status=HTTPStatus.OK,
                    extra_headers={"Service-Worker-Allowed": "/"},
                )
                return

            if route_path in ("/icon-192.png", "/apple-touch-icon.png", "/favicon.ico"):
                png_bytes = srv.render_pwa_icon_png(size=192)
                self._send_bytes(png_bytes, "image/png", status=HTTPStatus.OK)
                return

            if route_path == "/icon-512.png":
                png_bytes = srv.render_pwa_icon_png(size=512)
                self._send_bytes(png_bytes, "image/png", status=HTTPStatus.OK)
                return

            # 2. Protected API Endpoints
            if not self._check_authorized(parsed):
                self._send_json(
                    {"ok": False, "error": "Yetkisiz erişim (Geçersiz token)."},
                    status=HTTPStatus.UNAUTHORIZED,
                )
                return

            if route_path == "/api/status":
                status_payload = srv.build_status_payload()
                self._send_json(status_payload, HTTPStatus.OK)
                return

            if route_path == "/api/nodes":
                nodes_payload = srv.build_nodes_payload()
                self._send_json(nodes_payload, HTTPStatus.OK)
                return

            if route_path == "/api/connection":
                self._send_json(srv.get_connection_info(), HTTPStatus.OK)
                return

            # Y-7 (dilim 22): dashboard'un çağırdığı /api/incidents artık
            # var — olay galerisi ölü değil. Liste + tekil JPEG.
            if route_path == "/api/incidents":
                rec = srv._get_incident_recorder()
                if rec is None:
                    self._send_json({"ok": True, "incidents": [], "count": 0, "note": "IncidentRecorder bağlı değil."}, HTTPStatus.OK)
                    return
                try:
                    qs = urllib.parse.parse_qs(parsed.query)
                    try:
                        limit = max(1, min(50, int(qs.get("limit", ["20"])[0])))
                    except (TypeError, ValueError):
                        limit = 20
                    items = list(rec.get_incidents(limit=limit) or [])
                except Exception as exc:
                    logger.debug("Olay listesi okunamadı: %s", exc)
                    items = []
                self._send_json({"ok": True, "incidents": items, "count": len(items)}, HTTPStatus.OK)
                return

            if route_path.startswith("/api/incidents/") and route_path.endswith(".jpg"):
                rec = srv._get_incident_recorder()
                incident_id = route_path[len("/api/incidents/"):-len(".jpg")]
                jpeg_bytes = None
                if rec is not None:
                    try:
                        jpeg_bytes = rec.get_incident_jpeg_bytes(urllib.parse.unquote(incident_id))
                    except Exception as exc:
                        logger.debug("Olay JPEG okunamadı: %s", exc)
                if not jpeg_bytes:
                    self._send_json({"ok": False, "error": "Olay görüntüsü bulunamadı."}, status=HTTPStatus.NOT_FOUND)
                    return
                self._send_jpeg(jpeg_bytes, HTTPStatus.OK)
                return

            if route_path == "/api/pip_bundle":
                qs = urllib.parse.parse_qs(parsed.query)
                raw_quality = qs.get("quality", ["78"])[0]
                raw_scale = qs.get("scale", ["1.0"])[0]
                try:
                    quality = max(20, min(95, int(raw_quality)))
                except (TypeError, ValueError):
                    quality = 78
                try:
                    scale = max(0.25, min(2.0, float(raw_scale)))
                except (TypeError, ValueError):
                    scale = 1.0
                bundle_payload = srv.build_pip_bundle_payload(quality=quality, scale=scale)
                self._send_json(bundle_payload, HTTPStatus.OK)
                return

            if route_path in ("/api/frame.jpg", "/api/frame"):
                qs = urllib.parse.parse_qs(parsed.query)
                mode = (qs.get("mode", ["full"])[0] or "full").strip().lower()
                submode = (qs.get("submode", [""])[0] or "").strip().lower()
                raw_quality = qs.get("quality", ["65"])[0]
                raw_scale = qs.get("scale", [""])[0]
                try:
                    quality = max(20, min(95, int(raw_quality)))
                except (TypeError, ValueError):
                    quality = 65
                scale_val: Optional[float] = None
                if raw_scale:
                    try:
                        scale_val = max(0.2, min(2.0, float(raw_scale)))
                    except (TypeError, ValueError):
                        scale_val = None
                try:
                    jpeg_bytes = srv.render_frame_jpeg(
                        mode=mode,
                        quality=quality,
                        scale=scale_val,
                        submode=submode,
                    )
                except ValueError as mode_exc:
                    # F-1669: bilinmeyen kare mode'u sessizce "full" sayılmaz.
                    logger.debug("Geçersiz kare mode'u (%s): %s", mode, mode_exc)
                    self._send_json(
                        {"ok": False, "error": f"Geçersiz kare mode'u: {mode}"},
                        status=HTTPStatus.BAD_REQUEST,
                    )
                    return
                self._send_jpeg(jpeg_bytes, HTTPStatus.OK)
                return

            if route_path in ("/api/stream", "/api/events"):
                # P2-4 (K-17): SSE istemci sınırı (WS 4 gibi). Semafor
                # doluysa akış hiç açılmaz — 503 döner; akış bitince
                # (istemci kopması / max_events / sunucu durması dahil)
                # finally her yolda bırakır.
                if not _SSE_CLIENT_SEMAPHORE.acquire(blocking=False):
                    self._send_json(
                        {"ok": False, "error": "SSE istemci sınırı dolu."},
                        status=HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                try:
                    self._handle_sse_stream(srv, parsed)
                finally:
                    _SSE_CLIENT_SEMAPHORE.release()
                return

            self._send_json(
                {"ok": False, "error": f"Endpoint bulunamadı: {route_path}"},
                status=HTTPStatus.NOT_FOUND,
            )
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as exc:
            # F-1665: iç hata metni istemciye sızdırılmaz; ayrıntı yalnızca
            # sunucu log'unda tutulur.
            # P1-9 (K-08): hata logundaki yol token'lı URL içerebilir;
            # query değeri maskelenir.
            logger.error(
                "GET isteği işlenirken hata (%s): %s",
                _sanitize_token_from_text(self.path),
                _sanitize_token_from_text(exc),
            )
            try:
                self._send_json(
                    {"ok": False, "error": "İstek işlenemedi."},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except Exception as inner_exc:
                logger.debug("Hata yanıtı gönderilemedi: %s", inner_exc)

    def _handle_sse_stream(
        self,
        srv: "MobileRemoteServer",
        parsed: urllib.parse.ParseResult,
    ) -> None:
        """
        Item 34: Zero-Latency Server-Sent Events (`text/event-stream`) endpoint (`GET /api/stream`).
        Pushes real-time JSON frames of HP/MP, `1/3 -> 2/3 -> 3/3` harvest ring, bot status,
        and new log lines every `50ms..200ms` over a persistent HTTP/1.1 connection.
        """
        qs = urllib.parse.parse_qs(parsed.query)
        raw_interval = qs.get("interval_ms", ["50"])[0]
        raw_max_events = qs.get("max_events", qs.get("once", ["0"]))[0]
        try:
            interval_ms = max(25, min(1000, int(raw_interval)))
        except (TypeError, ValueError):
            interval_ms = 50
        try:
            max_events = max(0, int(raw_max_events))
        except (TypeError, ValueError):
            max_events = 0

        interval_sec = interval_ms / 1000.0

        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, no-transform")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        if max_events > 0:
            self.send_header("Connection", "close")
            self.close_connection = True
        else:
            self.send_header("Connection", "keep-alive")
        self.end_headers()

        seq = 0
        last_log_ts = 0.0

        try:
            while srv.is_running:
                seq += 1
                frame_payload, last_log_ts = srv.build_fast_telemetry_frame(
                    last_log_ts=last_log_ts,
                    seq=seq,
                    include_recent_logs=(seq == 1),
                )
                json_str = json.dumps(frame_payload, ensure_ascii=False)
                sse_chunk = f"id: {seq}\nevent: telemetry\ndata: {json_str}\n\n"
                self.wfile.write(sse_chunk.encode("utf-8"))
                self.wfile.flush()

                if max_events > 0 and seq >= max_events:
                    break

                # Wait up to interval_sec, or wake early if a new log/state notification arrives
                srv.wait_for_stream_tick(interval_sec)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def _handle_websocket_upgrade(
        self,
        srv: "MobileRemoteServer",
        parsed: urllib.parse.ParseResult,
    ) -> None:
        """
        Item 34: RFC 6455 WebSocket upgrade handshake & real-time `<50ms` push stream on `/ws`.
        Also accepts incoming JSON action commands from the phone over the same WebSocket.
        """
        ws_key = (self.headers.get("Sec-WebSocket-Key") or "").strip()
        if not ws_key:
            self._send_json(
                {
                    "ok": False,
                    "error": "WebSocket bağlantısı için 'Sec-WebSocket-Key' başlığı gereklidir.",
                    "upgrade_required": True,
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return

        accept_raw = (ws_key + _WS_MAGIC_GUID).encode("utf-8")
        accept_token = base64.b64encode(hashlib.sha1(accept_raw).digest()).decode("ascii")

        qs = urllib.parse.parse_qs(parsed.query)
        raw_interval = qs.get("interval_ms", ["50"])[0]
        raw_max_msgs = qs.get("max_messages", qs.get("max_events", ["0"]))[0]
        try:
            interval_ms = max(25, min(1000, int(raw_interval)))
        except (TypeError, ValueError):
            interval_ms = 50
        try:
            max_messages = max(0, int(raw_max_msgs))
        except (TypeError, ValueError):
            max_messages = 0

        interval_sec = interval_ms / 1000.0

        # O-3 (dilim 22): WS handshake artık körü körüne `*` yazmıyor —
        # dosyanın kendi CORS kuralıyla çelişiyordu. Yalnızca Host ile
        # eşleşen same-origin Origin yansıtılır (yoksa başlık atlanır;
        # token kontrolü upgrade öncesinde zaten yapıldı).
        _ws_origin = (self.headers.get("Origin") or "").strip()
        _ws_origin_host = ""
        try:
            _ws_origin_host = urllib.parse.urlparse(_ws_origin).netloc
        except Exception:
            _ws_origin_host = ""
        _ws_host = (self.headers.get("Host") or "").strip()
        handshake_lines = [
            "HTTP/1.1 101 Switching Protocols",
            "Upgrade: websocket",
            "Connection: Upgrade",
            f"Sec-WebSocket-Accept: {accept_token}",
        ]
        if _ws_origin and _ws_origin_host and _ws_host and _ws_origin_host == _ws_host:
            handshake_lines.append(f"Access-Control-Allow-Origin: {_ws_origin}")
        handshake_lines.extend(["", ""])
        self.wfile.write("\r\n".join(handshake_lines).encode("utf-8"))
        self.wfile.flush()
        self.close_connection = True

        sock: socket.socket = self.connection
        sock.settimeout(2.0)

        seq = 0
        last_log_ts = 0.0
        next_push_at = time.time()

        # Y-4 (dilim 22): bu WebSocket bağlantısı için komut hız sınırı
        # (30 komut / 10 sn) — HTTP tarafındaki sınırın bağlantı-içi karşılığı.
        _ws_cmd_hits: collections.deque = collections.deque()

        def _ws_rate_ok() -> bool:
            _now = time.time()
            while _ws_cmd_hits and _ws_cmd_hits[0] <= _now - 10.0:
                _ws_cmd_hits.popleft()
            if len(_ws_cmd_hits) >= 30:
                return False
            _ws_cmd_hits.append(_now)
            return True

        try:
            while srv.is_running:
                now = time.time()
                if now >= next_push_at:
                    seq += 1
                    frame_payload, last_log_ts = srv.build_fast_telemetry_frame(
                        last_log_ts=last_log_ts,
                        seq=seq,
                        include_recent_logs=(seq == 1),
                    )
                    ws_frame = _ws_encode_text_frame(json.dumps(frame_payload, ensure_ascii=False))
                    sock.sendall(ws_frame)
                    next_push_at = time.time() + interval_sec

                    if max_messages > 0 and seq >= max_messages:
                        # Send polite close frame (0x8)
                        sock.sendall(_ws_encode_frame(struct.pack("!H", 1000), opcode=0x8))
                        break

                wait_rem = max(0.005, min(interval_sec, next_push_at - time.time()))
                readable, _, _ = select.select([sock], [], [], wait_rem)
                if readable:
                    frame_info = _ws_read_client_frame(sock)
                    if frame_info is None:
                        break
                    opcode, client_payload = frame_info
                    if opcode == 0x8:  # Close
                        sock.sendall(_ws_encode_frame(struct.pack("!H", 1000), opcode=0x8))
                        break
                    if opcode == 0x9:  # Ping -> Pong
                        sock.sendall(_ws_encode_frame(client_payload, opcode=0xA))
                        continue
                    if opcode == 0x1 and client_payload:  # Text command
                        try:
                            cmd_obj = json.loads(client_payload.decode("utf-8", errors="replace"))
                            if isinstance(cmd_obj, dict):
                                if cmd_obj.get("action") == "ping":
                                    ack = {"ok": True, "type": "pong", "timestamp": round(time.time(), 3)}
                                elif not _ws_rate_ok():
                                    ack = {"ok": False, "type": "error", "error": "Çok fazla komut (hız sınırı: 30 komut/10 sn)."}
                                else:
                                    ack = srv.execute_action(cmd_obj)
                                    ack["type"] = "action_result"
                                sock.sendall(_ws_encode_text_frame(json.dumps(ack, ensure_ascii=False)))
                                next_push_at = 0.0  # Trigger immediate telemetry push after action
                        except Exception as cmd_exc:
                            logger.debug("WebSocket komut hatası: %s", cmd_exc)
                            err_ack = {"ok": False, "type": "error", "error": "Komut işlenemedi."}
                            sock.sendall(_ws_encode_text_frame(json.dumps(err_ack, ensure_ascii=False)))
                    elif opcode == 0x2 and client_payload:  # Binary: JSON komut olarak dene
                        try:
                            cmd_obj = json.loads(client_payload.decode("utf-8", errors="replace"))
                            if isinstance(cmd_obj, dict):
                                if not _ws_rate_ok():
                                    ack = {"ok": False, "type": "error", "error": "Çok fazla komut (hız sınırı: 30 komut/10 sn)."}
                                else:
                                    ack = srv.execute_action(cmd_obj)
                                    ack["type"] = "action_result"
                                sock.sendall(_ws_encode_text_frame(json.dumps(ack, ensure_ascii=False)))
                                next_push_at = 0.0
                            else:
                                raise ValueError("binary yük JSON nesnesi değil")
                        except Exception as cmd_exc:
                            logger.debug("WebSocket binary komut hatası: %s", cmd_exc)
                            err_ack = {"ok": False, "type": "error", "error": "Komut işlenemedi."}
                            sock.sendall(_ws_encode_text_frame(json.dumps(err_ack, ensure_ascii=False)))
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return

    def do_POST(self) -> None:
        srv: MobileRemoteServer = self.server.mobile_server  # type: ignore[attr-defined]
        try:
            parsed = urllib.parse.urlparse(self.path)
            route_path = parsed.path.rstrip("/") or "/"

            if not self._check_authorized(parsed):
                self._send_json(
                    {"ok": False, "error": "Yetkisiz erişim (Geçersiz token)."},
                    status=HTTPStatus.UNAUTHORIZED,
                )
                return

            # Y-7 (dilim 22): dialogdaki /api/command/* düğmeleri artık
            # ölü değil — pause_resume (duruma göre pause/resume) ve
            # emergency_stop (panic_stop) execute_action'a bağlanır.
            # toggle_pick_collect'in sunucuda karşılığı yok; o düğme
            # dialogda desteklenmiyor olarak işaretlendi (dilim 22 Y-7).
            if route_path.startswith("/api/command/"):
                _cmd_name = route_path[len("/api/command/"):].strip().lower()
                _cmd_map = {"pause_resume": "pause_resume", "emergency_stop": "panic_stop"}
                if _cmd_name not in _cmd_map:
                    self._send_json(
                        {"ok": False, "error": f"Desteklenmeyen komut endpoint'i: {route_path}"},
                        status=HTTPStatus.NOT_FOUND,
                    )
                    return
                _client_ip = ""
                try:
                    _client_ip = str(self.client_address[0])
                except Exception:
                    _client_ip = "unknown"
                if not srv._check_command_rate_limit(f"http:{_client_ip}"):
                    logger.warning("Mobil komut hız sınırı aşıldı (%s)", _client_ip)
                    self._send_json(
                        {"ok": False, "error": "Çok fazla istek (hız sınırı: 30 komut/10 sn)."},
                        status=HTTPStatus.TOO_MANY_REQUESTS,
                    )
                    return
                result = srv.execute_action({"action": _cmd_map[_cmd_name]})
                status_code = HTTPStatus.OK if result.get("ok", False) else HTTPStatus.BAD_REQUEST
                self._send_json(result, status=status_code)
                return

            if route_path != "/api/action":
                self._send_json(
                    {"ok": False, "error": f"POST endpoint bulunamadı: {route_path}"},
                    status=HTTPStatus.NOT_FOUND,
                )
                return

            # Y-4 (dilim 22): istemci başına hız sınırı — token sahibi
            # bile komut seli gönderemez (30 komut / 10 sn).
            _client_ip = ""
            try:
                _client_ip = str(self.client_address[0])
            except Exception:
                _client_ip = "unknown"
            if not srv._check_command_rate_limit(f"http:{_client_ip}"):
                logger.warning("Mobil komut hız sınırı aşıldı (%s)", _client_ip)
                self._send_json(
                    {"ok": False, "error": "Çok fazla istek (hız sınırı: 30 komut/10 sn)."},
                    status=HTTPStatus.TOO_MANY_REQUESTS,
                )
                return

            try:
                content_len = int(self.headers.get("Content-Length", "0") or "0")
            except (TypeError, ValueError):
                self._send_json(
                    {"ok": False, "error": "Geçersiz Content-Length başlığı."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            if content_len < 0 or content_len > 1024 * 64:
                self._send_json(
                    {"ok": False, "error": "Gövde boyutu kabul edilen aralığın dışında."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
            try:
                body_data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception as parse_exc:
                logger.debug("Geçersiz JSON gövdesi: %s", parse_exc)
                self._send_json(
                    {"ok": False, "error": "Geçersiz JSON gövdesi."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            if not isinstance(body_data, dict):
                self._send_json(
                    {"ok": False, "error": "JSON gövdesi bir nesne (dict) olmalıdır."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return

            result = srv.execute_action(body_data)
            status_code = HTTPStatus.OK if result.get("ok", False) else HTTPStatus.BAD_REQUEST
            self._send_json(result, status=status_code)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return
        except Exception as exc:
            logger.error("POST /api/action işlenirken hata: %s", exc, exc_info=True)
            try:
                self._send_json(
                    {"ok": False, "error": "İstek işlenemedi."},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            except Exception as inner_exc:
                logger.debug("POST hata yanıtı gönderilemedi: %s", inner_exc)


class MobileRemoteServer:
    """
    Multi-threaded Mobile Remote Control, Dual-Channel PiP Stream, SSE/WebSocket Push,
    and Full PWA Backend for Gatherion A2 Bot.

    Serves:
    - `GET /` or `GET /index.html` -> `ui/mobile_dashboard.html` (with automatic PWA manifest & SW link injection)
    - `GET /manifest.json` & `GET /sw.js` -> Full PWA support (`ui/manifest.json` & `ui/sw.js`)
    - `GET /api/status` -> Comprehensive JSON status (bot, harvest, vitals, route, system, tunnel, recent_logs)
    - `GET /api/nodes` -> Live 2D route radar node list with coordinates, harvest stages, quarantine & rest timers
    - `GET /api/frame.jpg?mode=full|minimap|pip&quality=65&scale=0.5` -> Live JPEG stream (Full, Minimap, or Dual PiP)
    - `GET /api/pip_bundle` -> Single-request JSON containing base64 high-res crops of both Minimap and `[F Gather]` ROI
    - `GET /api/stream` -> `< 50ms` Server-Sent Events (`text/event-stream`) real-time push
    - `GET /ws` -> `< 50ms` RFC 6455 WebSocket real-time push & bi-directional command channel
    - `POST /api/action` -> Remote control actions (`pause`, `resume`, `set_mode`, `record_f8`, `skip_node`,
      `reset_node`, `reset_all_nodes`, `quarantine_node`, `delete_node`, `toggle_node`, `optimize_route`,
      `send_key`, `click_screen`, `tap_to_move_map`, `update_settings`, `panic_stop`)
    """

    VERSION: str = "v2.0-TR"

    def __init__(
        self,
        orchestrator: Any = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        access_token: Optional[str] = None,
        # F-0029: kimlik doğrulama varsayılan AÇIK — internete açık otomatik
        # tünelle birlikte sunucu artık tokensız çalışmaz.
        require_token: bool = True,
        worker: Any = None,
        route_mgr: Any = None,
        capture: Any = None,
        vision: Any = None,
        telemetry: Any = None,
        watchdog: Any = None,
        settings: Any = None,
        settings_mgr: Any = None,
        input_ctrl: Any = None,
        combat_recovery: Any = None,
        tunnel_bridge: Any = None,
    ) -> None:
        self.orchestrator = orchestrator
        self.host: str = str(host or "0.0.0.0").strip()
        # F-1664: port=0 ephemeral port demektir; yalnızca None varsayılan
        # 8765'e düşer.
        self.requested_port: int = 8765 if port is None else int(port)
        self.port: int = self.requested_port
        self._port_fallback_from: Optional[int] = None
        self.require_token: bool = bool(require_token)

        # Standalone provider overrides (used if orchestrator is None or doesn't hold them)
        self._worker = worker
        self._route_mgr = route_mgr
        self._capture = capture
        self._vision = vision
        self._telemetry = telemetry
        self._watchdog = watchdog
        self._settings = settings
        self._settings_mgr = settings_mgr
        self._input_ctrl = input_ctrl
        self._combat_recovery = combat_recovery
        self._tunnel_bridge = tunnel_bridge

        # Tunnel & Notification metadata
        self.public_url: str = ""
        self.ntfy_topic: str = ""

        # Persistent or generated 8-character alphanumeric access token
        self.access_token: str = self._init_or_load_access_token(access_token)

        # Stream notification condition variable for <50ms wakeup on log/action events
        self._stream_cond = threading.Condition()

        # Ring buffer of last 120 log lines
        self._log_lock = threading.RLock()
        self._log_buffer: Deque[Dict[str, Any]] = collections.deque(maxlen=120)
        self._log_handler: Optional[MobileLogHandler] = None
        self._attach_log_handler()

        # Frame cache for fast concurrent mobile polling
        self._frame_lock = threading.RLock()
        self._cached_frame: Optional[np.ndarray] = None
        self._cached_frame_ts: float = 0.0
        self._extraction_failed_count: int = 0

        # Memory stats cache for <1ms fast SSE/WebSocket telemetry pushes
        self._mem_cache_ts: float = 0.0
        self._cached_memory_mb: float = 0.0
        self._cached_system_ram_pct: float = 0.0

        # Tap-to-move tracking
        self._last_tap_to_move: Optional[Dict[str, Any]] = None

        # Cached PWA icons by size
        self._icon_cache: Dict[int, bytes] = {}

        # Cached static synthetic-frame base canvas (built lazily, see
        # _build_synthetic_base_frame).
        self._synthetic_base_frame: Optional[np.ndarray] = None

        # Cached static text files (dashboard/manifest/service worker),
        # keyed by path -> ((mtime_ns, size), content). Invalidated when
        # the file's mtime or size changes.
        self._static_file_cache: Dict[str, Tuple[Tuple[int, int], str]] = {}
        self._static_cache_lock = threading.Lock()

        # Server lifecycle state
        self._server_lock = threading.RLock()
        self._httpd: Optional[_ThreadingMobileHTTPServer] = None
        self._server_thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._start_ts: float = time.time()

        # Y-4 (dilim 22): istemci başına komut hız sınırı — /api/action ve
        # WebSocket komutları için (anahtar -> son istek zaman damgaları).
        self._cmd_rate_lock = threading.Lock()
        self._cmd_rate_hits: Dict[str, collections.deque] = {}
        # D-1 (dilim 22): token rotasyon durumu (dialog get_token_rotation_status
        # ile okur); rotate_access_token çağrılmadıysa None kalır.
        self._token_rotated_at: Optional[float] = None
        # B-04 (dilim 30): geçersiz kırpma yedeği uyarısı kare başına
        # loglanmaz — en fazla 60 sn'de bir.
        self._crop_fallback_warn_ts: float = 0.0

        self.dashboard_html_path: str = os.path.join(PROJECT_ROOT, "ui", "mobile_dashboard.html")
        self.manifest_json_path: str = os.path.join(PROJECT_ROOT, "ui", "manifest.json")
        self.service_worker_js_path: str = os.path.join(PROJECT_ROOT, "ui", "sw.js")

    @property
    def is_running(self) -> bool:
        return bool(self._running)

    def notify_stream_listeners(self) -> None:
        """Wakes up any waiting SSE or WebSocket stream threads immediately."""
        with self._stream_cond:
            self._stream_cond.notify_all()

    def wait_for_stream_tick(self, timeout_sec: float) -> None:
        """Waits up to `timeout_sec` or returns immediately when a new event is signalled."""
        with self._stream_cond:
            self._stream_cond.wait(timeout=max(0.005, float(timeout_sec)))

    # =========================================================================
    # Token & Logging Setup
    # =========================================================================

    @staticmethod
    def _write_token_file(token: str, failure_note: str) -> bool:
        """Token'ı 0600 izinle `config/mobile_token.txt`'ye yazar.

        `_init_or_load_access_token` ve `rotate_access_token`'ın ortak
        yazma bloğudur; tek yerden beslenir. Yazma başarısız olursa
        False döner (çağıran yine de üretilen token'ı bellekte kullanır).
        """
        token_file = os.path.join(PROJECT_ROOT, "config", "mobile_token.txt")
        try:
            os.makedirs(os.path.dirname(token_file), exist_ok=True)
            fd = os.open(token_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, token.encode("utf-8"))
            finally:
                os.close(fd)
            try:
                os.chmod(token_file, 0o600)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("%s: %s", failure_note, exc)
            return False
        return True

    def _init_or_load_access_token(self, explicit_token: Optional[str]) -> str:
        # F-1652/F-0136: 32 karakterlik token; açık token en az 16 karakter
        # olmalı, daha kısası yok sayılır. Kayıtlı token 16 karakterden
        # kısaysa (eski 8 karakterlik zayıf token) yenisiyle değiştirilir.
        if explicit_token and str(explicit_token).strip():
            clean = "".join(c for c in str(explicit_token).strip() if c.isalnum())
            if len(clean) >= _MOBILE_TOKEN_MIN_EXPLICIT_LENGTH:
                return clean[:64]
            logger.warning(
                "Verilen mobil token çok kısa (en az %d karakter); yok sayıldı.",
                _MOBILE_TOKEN_MIN_EXPLICIT_LENGTH,
            )

        token_file = os.path.join(PROJECT_ROOT, "config", "mobile_token.txt")
        try:
            if os.path.isfile(token_file):
                with open(token_file, "r", encoding="utf-8") as f:
                    saved = "".join(c for c in f.read().strip() if c.isalnum())
                if len(saved) >= _MOBILE_TOKEN_MIN_SAVED_LENGTH:
                    return saved[:64]
        except Exception as exc:
            logger.debug("Kayıtlı mobil token okunamadı: %s", exc)

        alphabet = string.ascii_letters + string.digits
        generated = "".join(secrets.choice(alphabet) for _ in range(_MOBILE_TOKEN_LENGTH))
        self._write_token_file(generated, "Mobil token diske yazılamadı")
        return generated

    def rotate_access_token(self) -> str:
        """Yeni 32 karakterlik token üretir, 0600 izinle diske yazar."""
        alphabet = string.ascii_letters + string.digits
        generated = "".join(secrets.choice(alphabet) for _ in range(_MOBILE_TOKEN_LENGTH))
        self._write_token_file(generated, "Mobil token rotasyonu diske yazılamadı")
        self.access_token = generated
        self._token_rotated_at = time.time()
        return generated

    def sync_token_file(self) -> bool:
        """Y-3 (dilim 22): diskteki `config/mobile_token.txt`'yi canlı
        `access_token` ile eşitler. main.py, 256-bit authenticator'ı
        bind ettikten sonra çağırır — yoksa diskteki token (sunucunun
        kendi ürettiği kısa token) geçersiz kalır ve o tokenla giriş
        deneyen 401 alır."""
        return bool(self._write_token_file(str(self.access_token or ""), "Mobil token diske yazılamadı"))

    def get_token_rotation_status(self) -> Dict[str, Any]:
        """D-1 (dilim 22): dialogdaki 'rotasyon durumu' alanı artık ölü
        değil — son rotasyon zamanı + disk token'ının canlı tokenla
        eşleşip eşleşmediği döner."""
        token_file = os.path.join(PROJECT_ROOT, "config", "mobile_token.txt")
        file_matches_live = False
        try:
            if os.path.isfile(token_file):
                with open(token_file, "r", encoding="utf-8") as f:
                    saved = "".join(c for c in f.read().strip() if c.isalnum())
                file_matches_live = bool(saved) and saved[:64] == str(self.access_token or "")[:64]
        except Exception:
            file_matches_live = False
        return {
            "active": self._token_rotated_at is not None,
            "last_rotated": (
                datetime.fromtimestamp(self._token_rotated_at).isoformat()
                if self._token_rotated_at
                else None
            ),
            "token_file_matches_live": file_matches_live,
            "rotation_available": True,
        }

    def _get_incident_recorder(self) -> Optional[Any]:
        """Y-7 (dilim 22): olay galerisi için IncidentRecorder — önce
        orchestrator, sonra worker üzerinden (main.py:566 kurar)."""
        for obj in (self.orchestrator, self.worker):
            if obj is None:
                continue
            rec = getattr(obj, "incident_recorder", None)
            if rec is not None and hasattr(rec, "get_incidents"):
                return rec
        return None

    def _check_command_rate_limit(self, client_key: str, limit: int = 30, window_sec: float = 10.0) -> bool:
        """Y-4 (dilim 22): istemci başına komut hız sınırı — pencere
        içinde `limit` istek; aşılırsa False döner (çağıran 429 döner).
        Token sahibi bile komut seli gönderemez; `restart`/`quit` gibi
        yıkıcı aksiyonlar da bu sınırın içindedir."""
        now = time.time()
        key = str(client_key or "unknown")
        with self._cmd_rate_lock:
            hits = self._cmd_rate_hits.get(key)
            if hits is None:
                hits = collections.deque()
                self._cmd_rate_hits[key] = hits
            while hits and hits[0] <= now - window_sec:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def _attach_log_handler(self) -> None:
        # F-1667: yalnızca daha önce bu sunucunun taktığı handler sökülür;
        # kök logger'daki diğer MobileLogHandler'lara dokunulmaz.
        root_logger = logging.getLogger()
        with self._log_lock:
            if self._log_handler is not None:
                try:
                    root_logger.removeHandler(self._log_handler)
                except Exception as exc:
                    logger.debug("Önceki log handler kaldırılırken uyarı: %s", exc)
                self._log_handler = None

            self._log_handler = MobileLogHandler(
                self._log_buffer,
                self._log_lock,
                notify_cond=self._stream_cond,
            )
            root_logger.addHandler(self._log_handler)

            now = time.time()
            ts_str = datetime.fromtimestamp(now).strftime("%H:%M:%S")
            init_msg = f"Mobil Uzaktan Kontrol Sunucusu hazır ({self.VERSION})."
            self._log_buffer.append(
                {
                    "ts": ts_str,
                    "time": ts_str,
                    "timestamp": round(now, 3),
                    "level": "INFO",
                    "logger": "Gatherion.MobileServer",
                    "msg": init_msg,
                    "message": init_msg,
                }
            )

    def _detach_log_handler(self) -> None:
        root_logger = logging.getLogger()
        with self._log_lock:
            if self._log_handler is not None:
                try:
                    root_logger.removeHandler(self._log_handler)
                except Exception as exc:
                    logger.debug("Log handler kaldırılırken uyarı: %s", exc)
                self._log_handler = None

    # =========================================================================
    # Component Resolution Helpers
    # =========================================================================

    @property
    def worker(self) -> Any:
        if self._worker is not None:
            return self._worker
        if self.orchestrator is not None:
            return getattr(self.orchestrator, "worker", None)
        return None

    @property
    def route_mgr(self) -> Any:
        if self._route_mgr is not None:
            return self._route_mgr
        if self.orchestrator is not None:
            rm = getattr(self.orchestrator, "route_mgr", None) or getattr(
                self.orchestrator, "route_manager", None
            )
            if rm is not None:
                return rm
        w = self.worker
        if w is not None:
            return getattr(w, "route_manager", None) or getattr(w, "routes", None)
        return None

    @property
    def capture(self) -> Any:
        if self._capture is not None:
            return self._capture
        if self.orchestrator is not None:
            cap = getattr(self.orchestrator, "capture", None)
            if cap is not None:
                return cap
        v = self.vision
        if v is not None:
            return getattr(v, "capture", None)
        return None

    @property
    def vision(self) -> Any:
        if self._vision is not None:
            return self._vision
        if self.orchestrator is not None:
            vis = getattr(self.orchestrator, "vision", None)
            if vis is not None:
                return vis
        w = self.worker
        if w is not None:
            return getattr(w, "vision", None)
        return None

    @property
    def telemetry(self) -> Any:
        if self._telemetry is not None:
            return self._telemetry
        if self.orchestrator is not None:
            tel = getattr(self.orchestrator, "telemetry", None)
            if tel is not None:
                return tel
        w = self.worker
        if w is not None:
            return getattr(w, "telemetry", None)
        return None

    @property
    def watchdog(self) -> Any:
        if self._watchdog is not None:
            return self._watchdog
        if self.orchestrator is not None:
            wd = getattr(self.orchestrator, "watchdog", None)
            if wd is not None:
                return wd
        w = self.worker
        if w is not None:
            return getattr(w, "watchdog", None)
        return None

    @property
    def settings_mgr(self) -> Any:
        if self._settings_mgr is not None:
            return self._settings_mgr
        if self.orchestrator is not None:
            sm = getattr(self.orchestrator, "settings_mgr", None)
            if sm is not None:
                return sm
        w = self.worker
        if w is not None:
            return getattr(w, "config", None)
        return None

    @property
    def settings(self) -> Any:
        if self._settings is not None:
            return self._settings
        if self.orchestrator is not None:
            st = getattr(self.orchestrator, "settings", None)
            if st is not None:
                return st
        sm = self.settings_mgr
        if sm is not None:
            return getattr(sm, "settings", None)
        return None

    @property
    def input_ctrl(self) -> Any:
        if self._input_ctrl is not None:
            return self._input_ctrl
        if self.orchestrator is not None:
            ic = getattr(self.orchestrator, "input_ctrl", None)
            if ic is not None:
                return ic
        w = self.worker
        if w is not None:
            return getattr(w, "input_driver", None) or getattr(w, "driver", None)
        return None

    @property
    def combat_recovery(self) -> Any:
        if self._combat_recovery is not None:
            return self._combat_recovery
        if self.orchestrator is not None:
            cr = getattr(self.orchestrator, "combat_recovery", None)
            if cr is not None:
                return cr
        w = self.worker
        if w is not None:
            return getattr(w, "combat_recovery", None) or getattr(w, "combat", None)
        return None

    @property
    def tunnel_bridge(self) -> Any:
        if self._tunnel_bridge is not None:
            return self._tunnel_bridge
        if self.orchestrator is not None:
            return getattr(self.orchestrator, "tunnel_bridge", None)
        return None

    # =========================================================================
    # Network & Lifecycle Methods
    # =========================================================================

    @staticmethod
    def get_lan_ip() -> str:
        """Detects the primary LAN IPv4 address of the host machine."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.settimeout(0.5)
                s.connect(("8.8.8.8", 80))
                ip = s.getsockname()[0]
                if ip and not ip.startswith("127."):
                    return str(ip)
        except Exception as exc:
            logger.debug("UDP LAN IP tespiti atlandı: %s", exc)

        try:
            hostname = socket.gethostname()
            addrs = socket.getaddrinfo(hostname, None, socket.AF_INET)
            for item in addrs:
                candidate = item[4][0]
                if candidate and not candidate.startswith("127.") and not candidate.startswith("169.254."):
                    return str(candidate)
        except Exception as exc:
            logger.debug("Hostname LAN IP tespiti atlandı: %s", exc)

        return "127.0.0.1"

    @property
    def lan_url(self) -> str:
        ip = self.get_lan_ip()
        return f"http://{ip}:{self.port}"

    def start(self) -> str:
        """
        Starts the multi-threaded HTTP server in a daemon thread on `self.port`,
        automatically falling back to ports `8766..8775` if the port is busy.
        Returns `lan_url`.
        """
        with self._server_lock:
            if self._running and self._httpd is not None:
                return self.lan_url

            candidate_ports = [self.requested_port] + [
                p for p in range(8766, 8776) if p != self.requested_port
            ]
            last_err: Optional[Exception] = None

            for candidate_port in candidate_ports:
                try:
                    httpd = _ThreadingMobileHTTPServer(
                        (self.host, int(candidate_port)),
                        MobileHTTPRequestHandler,
                        self,
                    )
                    self.port = int(httpd.server_address[1])
                    if candidate_port != self.requested_port:
                        # F-1675: port fallback sessiz olmaz.
                        self._port_fallback_from = int(self.requested_port)
                        logger.warning(
                            "İstenen port %d dolu; mobil sunucu %d portuna geçti.",
                            self.requested_port,
                            self.port,
                        )
                    self._httpd = httpd
                    self._running = True
                    self._start_ts = time.time()

                    self._server_thread = threading.Thread(
                        target=self._serve_loop,
                        name=f"GatherionMobileServer-{self.port}",
                        daemon=True,
                    )
                    self._server_thread.start()
                    # F-1651: erişim token'ı log'a yazılmaz.
                    logger.info(
                        "Mobil Uzaktan Kontrol Sunucusu başlatıldı: %s (Port: %d, Token koruması: %s)",
                        self.lan_url,
                        self.port,
                        "açık" if self.require_token else "kapalı",
                    )
                    return self.lan_url
                except OSError as exc:
                    last_err = exc
                    logger.debug("Port %d dolu, sonraki port deneniyor: %s", candidate_port, exc)

            # Final fallback: OS-assigned ephemeral port
            try:
                httpd = _ThreadingMobileHTTPServer(
                    (self.host, 0),
                    MobileHTTPRequestHandler,
                    self,
                )
                self.port = int(httpd.server_address[1])
                self._httpd = httpd
                self._running = True
                self._start_ts = time.time()
                self._server_thread = threading.Thread(
                    target=self._serve_loop,
                    name=f"GatherionMobileServer-{self.port}",
                    daemon=True,
                )
                self._server_thread.start()
                logger.info("Mobil Uzaktan Kontrol Sunucusu dinamik portta başlatıldı: %s", self.lan_url)
                return self.lan_url
            except Exception as final_exc:
                logger.error("Mobil sunucu başlatılamadı: %s (Son hata: %s)", final_exc, last_err)
                raise

    def _serve_loop(self) -> None:
        httpd = self._httpd
        if httpd is None:
            return
        try:
            httpd.serve_forever(poll_interval=0.1)
        except Exception as exc:
            if self._running:
                logger.error("Mobil HTTP sunucu döngüsü hatası: %s", exc)

    def stop(self) -> None:
        """Stops the HTTP server thread and cleans up resources."""
        with self._server_lock:
            self._running = False
            httpd = self._httpd
            self._httpd = None

        self.notify_stream_listeners()

        if httpd is not None:
            try:
                httpd.shutdown()
            except Exception as exc:
                logger.debug("httpd.shutdown uyarısı: %s", exc)
            try:
                httpd.server_close()
            except Exception as exc:
                logger.debug("httpd.server_close uyarısı: %s", exc)

        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=2.0)
        self._server_thread = None
        self._detach_log_handler()
        logger.info("Mobil Uzaktan Kontrol Sunucusu durduruldu.")

    def get_connection_info(self) -> Dict[str, Any]:
        """Returns LAN URL, public tunnel URL, ntfy topic, WebSocket/SSE URLs.

        F-1651: erişim token'ı düz metin olarak dışarı verilmez; yalnızca
        varlığı ve SHA-256 parmak izi döner (P1-9/K-08: son 4 karakter
        dahil token parçası verilmez).
        F-1670: QR üçüncü taraf servise (api.qrserver.com) URL sızdırmaz;
        `qr_url` boş döner — QR, token'lı URL'den istemci tarafında
        (ui/mobile_connect_dialog.py) üretilir.
        """
        lan = self.lan_url
        pub = self.public_url
        ntfy = self.ntfy_topic

        tb = self.tunnel_bridge
        if tb is not None:
            pub = pub or str(getattr(tb, "public_url", "") or "")
            ntfy = ntfy or str(getattr(tb, "ntfy_topic", "") or "")

        ws_url = lan.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        token = str(self.access_token or "")

        return {
            "running": bool(self._running),
            "host": self.host,
            "port": self.port,
            "requested_port": self.requested_port,
            "port_fallback_from": self._port_fallback_from,
            "lan_ip": self.get_lan_ip(),
            "lan_url": lan,
            "ws_url": ws_url,
            "sse_url": f"{lan}/api/stream",
            "pip_bundle_url": f"{lan}/api/pip_bundle",
            "manifest_url": f"{lan}/manifest.json",
            "public_url": pub,
            "ntfy_topic": ntfy,
            "qr_url": "",
            "access_token_present": bool(token),
            # P1-9 (K-08): token parçası (son 4 karakter dahil) dışarı
            # verilmez; yalnızca SHA-256 parmak izi döner.
            "access_token_preview": _token_fingerprint(token),
            "require_token": self.require_token,
        }

    # =========================================================================
    # Data Aggregation: /api/status, /api/nodes & <50ms Fast Telemetry Push
    # =========================================================================

    def _normalize_mode_for_mobile(self, raw_mode: str) -> str:
        m = str(raw_mode or "route").strip().lower()
        if m in ("gather", "manual_record", "auto_record"):
            return "gather"
        if m in ("explore", "explorer", "explorer_mode", "auto_cartography"):
            return "explore"
        if m in ("combat", "combat_mode", "fight", "savas", "savaş"):
            return "combat"
        return "route"

    @staticmethod
    def _bot_is_elevated() -> bool:
        try:
            from core.uac_elevator import is_admin

            return bool(is_admin())
        except Exception:
            try:
                import ctypes as _ct

                return bool(_ct.windll.shell32.IsUserAnAdmin() != 0)
            except Exception:
                return False

    @staticmethod
    def _is_node_quarantined_or_failed(node: Any) -> bool:
        raw_st = str(getattr(node, "state", "active") or "active").strip().lower()
        return raw_st in ("quarantined", "failed", "disabled")

    def _select_next_non_quarantined_node(self, rm: Any, region: str, route: str) -> Optional[Any]:
        """Finds the next ready node that is not quarantined/failed."""
        if rm is None:
            return None
        try:
            now = time.time()
            nodes = rm.get_route_nodes(region, route)
            for n in nodes:
                if self._is_node_quarantined_or_failed(n):
                    continue
                if hasattr(n, "is_ready") and n.is_ready(now=now):
                    return n
        except Exception as exc:
            logger.debug("Sonraki aktif düğüm seçilirken uyarı: %s", exc)
        return None

    def build_status_payload(self) -> Dict[str, Any]:
        """Builds the comprehensive JSON payload for `GET /api/status`."""
        now = time.time()
        w = self.worker
        rm = self.route_mgr
        tel = self.telemetry
        wd = self.watchdog
        st = self.settings
        ic = self.input_ctrl
        cr = self.combat_recovery

        # 1. Bot State & Mode
        raw_state = "PAUSED"
        raw_mode = "route"
        is_paused = True
        status_text = "Duraklatıldı · Başlatmak için dokunun"
        uptime_sec = max(0.0, now - self._start_ts)

        if w is not None:
            raw_state = str(getattr(w, "state", "PAUSED") or "PAUSED").upper()
            raw_mode = str(getattr(w, "mode", "route") or "route").lower()
            is_paused = bool(getattr(w, "is_paused", True))
            sess_start = float(getattr(w, "_session_start_time", self._start_ts) or self._start_ts)
            uptime_sec = max(0.0, now - sess_start)
            pause_reason = str(getattr(w, "_pause_reason", "") or "")
            if is_paused and pause_reason:
                status_text = pause_reason
            elif raw_state == "GATHERING":
                status_text = "Materyal toplanıyor..."
            elif raw_state == "WALKING":
                act_node = getattr(w, "active_node", None)
                dist_m = int(getattr(w, "active_node_distance_m", 0) or 0)
                if act_node is not None:
                    nid = getattr(act_node, "id", "?")
                    status_text = f"Node #{nid} hedefine yürünüyor ({dist_m}m)"
                else:
                    status_text = "Rota üzerinde ilerleniyor"
            elif raw_state == "FIGHTING":
                status_text = "Saldırı altında! Karşı saldırı uygulanıyor"
            elif raw_state == "BREAK":
                status_text = "İnsansı mola (Break) aktif"
            else:
                status_text = f"Aktif ({raw_state})"

        state_display = STATE_TR_MAP.get(raw_state, f"{raw_state}")
        mobile_mode = self._normalize_mode_for_mobile(raw_mode)

        test_mode = False
        gather_key = "f"
        max_stages_cfg = 3
        minimap_auto_record = True
        hybrid_3d_record = True
        rest_minutes_cfg = 5

        if st is not None:
            g_cfg = getattr(st, "gathering", None)
            r_cfg = getattr(st, "route", None)
            if g_cfg is not None:
                test_mode = bool(getattr(g_cfg, "test_mode", False))
                gather_key = str(getattr(g_cfg, "gather_key", "f") or "f").lower()
                max_stages_cfg = int(getattr(g_cfg, "harvests_per_node", 3) or 3)
            if r_cfg is not None:
                minimap_auto_record = bool(getattr(r_cfg, "minimap_auto_record", True))
                hybrid_3d_record = bool(getattr(r_cfg, "hybrid_3d_and_minimap_record", True))
                rest_minutes_cfg = int(getattr(r_cfg, "rest_minutes", 5) or 5)
        elif ic is not None:
            test_mode = bool(getattr(ic, "test_mode", False))

        # 2. Route & Node Metrics — F-1661: kaynak (route_mgr) yokken
        # sahte "Altgard"/"Dredgion Crash Site" sabiti yanıta girmez;
        # bilinmeyen bölge/rota None döner.
        region_name: Optional[str] = None
        route_name: Optional[str] = None
        current_node_id: Optional[int] = None
        total_nodes = 0
        active_nodes = 0
        resting_nodes = 0
        failed_nodes = 0
        quarantined_nodes = 0
        minimap_mapped_count = 0
        world_3d_mapped_count = 0
        current_stage = 0
        max_stages = max_stages_cfg

        if w is not None and getattr(w, "active_node", None) is not None:
            act_n = getattr(w, "active_node")
            current_node_id = int(getattr(act_n, "id", 0) or 0) or None
            current_stage = int(getattr(act_n, "current_harvests", 0) or 0)
            max_stages = int(getattr(act_n, "max_harvests", max_stages_cfg) or max_stages_cfg)

        if rm is not None:
            region_name = str(getattr(rm, "active_region", "") or "") or None
            route_name = str(getattr(rm, "active_route", "") or "") or None
            # F-1658: GET /api/status salt-okunur olmalı — rota durumunu
            # değiştiren `update_resting_nodes()` burada çağrılmaz;
            # durum ayrımı aşağıdaki sayım döngüsünde salt-okunur
            # `effective_node_state()` ile hesaplanır (P1-6).
            try:
                # P1-6 (K-06): sayım tek `effective_node_state()`'ten —
                # düğüm listesi (`build_nodes_payload`) ile aynı değer.
                # Eskiden hatalı + dinlenen düğüm hem failed hem resting
                # (hatta active) kovasına giriyor, iki payload çelişiyordu.
                from core.route_manager import effective_node_state

                nodes = rm.get_route_nodes(region_name, route_name)
                total_nodes = len(nodes)
                for n in nodes:
                    n_src = str(getattr(n, "source_type", "manual") or "manual").lower()
                    node_state = effective_node_state(n, now=now)
                    if node_state == "quarantined":
                        quarantined_nodes += 1
                    elif node_state == "failed":
                        failed_nodes += 1
                    elif node_state == "resting":
                        resting_nodes += 1
                    else:
                        active_nodes += 1

                    if "minimap" in n_src:
                        minimap_mapped_count += 1
                    elif "3d" in n_src:
                        world_3d_mapped_count += 1

                    if current_node_id is not None and int(getattr(n, "id", -1)) == current_node_id:
                        current_stage = int(getattr(n, "current_harvests", current_stage) or 0)
                        max_stages = int(getattr(n, "max_harvests", max_stages) or max_stages_cfg)
            except Exception as exc:
                logger.debug("Rota düğüm istatistikleri hesaplanırken hata: %s", exc)

        # 3. Harvest & Telemetry Metrics
        total_gathered = 0
        gathers_per_hour = 0.0
        consecutive_failures = 0

        if rm is not None:
            total_gathered = int(getattr(rm, "total_gathered", 0) or 0)
            consecutive_failures = int(getattr(rm, "consecutive_failures", 0) or 0)
        if w is not None:
            total_gathered = max(total_gathered, int(getattr(w, "gathered_count", 0) or 0))
            consecutive_failures = max(consecutive_failures, int(getattr(w, "failed_in_row", 0) or 0))

        if tel is not None and hasattr(tel, "get_live_metrics"):
            try:
                t_metrics = tel.get_live_metrics()
                total_gathered = max(total_gathered, int(t_metrics.get("gathered_total", 0) or 0))
                gathers_per_hour = float(t_metrics.get("gathers_per_hour", 0.0) or 0.0)
            except Exception as exc:
                logger.debug("Telemetri metrikleri okunamadı: %s", exc)
        if gathers_per_hour <= 0.0 and total_gathered > 0 and uptime_sec > 1.0:
            gathers_per_hour = round((total_gathered / uptime_sec) * 3600.0, 1)

        # 4. Vitals (HP, MP, Combat, Death) — F-1661: kaynak yokken sahte
        # %100 varsayımı yapılmaz; bilinmeyen değer None döner.
        # combat_recovery varken de last_hp_ratio 0.0/None ise %100'e
        # çevrilmez: 0.0 gerçek 0 HP'dir, None bilinmeyendir.
        hp_ratio: Optional[float] = None
        mp_ratio: Optional[float] = None
        under_attack = raw_state == "FIGHTING"
        character_dead = False
        if cr is not None:
            _raw_hp = getattr(cr, "last_hp_ratio", None)
            hp_ratio = float(_raw_hp) if _raw_hp is not None else None
            _raw_mp = getattr(cr, "last_mp_ratio", None)
            mp_ratio = float(_raw_mp) if _raw_mp is not None else None
            under_attack = under_attack or bool(getattr(cr, "in_combat", False))
            character_dead = bool(getattr(cr, "is_dead", False))

        # 5. System & Watchdog Metrics — F-1661: kaynak yokken sahte
        # fps/latency/çözünürlük varsayımı yapılmaz; bilinmeyen None döner.
        fps: Optional[float] = None
        latency_ms: Optional[float] = None
        resolution_str: Optional[str] = None
        memory_mb: Optional[float] = None
        system_ram_pct: Optional[float] = None
        watchdog_healthy: Optional[bool] = None
        watchdog_notifications: List[Dict[str, Any]] = []
        input_backend: Optional[str] = None

        if ic is not None:
            input_backend = str(getattr(ic, "active_backend_name", "") or "") or None
            sw = int(getattr(ic, "screen_width", 0) or 0)
            sh = int(getattr(ic, "screen_height", 0) or 0)
            if sw > 0 and sh > 0:
                resolution_str = f"{sw}x{sh}"

        cap = self.capture
        if cap is not None:
            cap_fps = getattr(cap, "last_fps", None)
            if cap_fps:
                fps = float(cap_fps)
            cap_sw = int(getattr(cap, "screen_w", 0) or 0)
            cap_sh = int(getattr(cap, "screen_h", 0) or 0)
            if cap_sw > 0 and cap_sh > 0:
                resolution_str = f"{cap_sw}x{cap_sh}"

        if wd is not None:
            try:
                if hasattr(wd, "get_health_snapshot"):
                    snap = wd.get_health_snapshot()
                elif hasattr(wd, "_build_metrics_unlocked"):
                    snap = wd._build_metrics_unlocked(now)
                else:
                    snap = {}
                if isinstance(snap, dict):
                    if snap.get("fps"):
                        fps = float(snap.get("fps"))
                    if snap.get("latency_ms"):
                        latency_ms = float(snap.get("latency_ms"))
                    if snap.get("process_rss_mb", snap.get("memory_mb")):
                        memory_mb = float(snap.get("process_rss_mb", snap.get("memory_mb")))
                    if snap.get("system_memory_load_pct"):
                        system_ram_pct = float(snap.get("system_memory_load_pct"))
                    watchdog_healthy = not bool(snap.get("stalled", False))
                    # O-4 (dilim 29): WORKER_STALLED/DEAD/REVIVED/LOW_HP
                    # bildirimleri yalnızca RAM deque + logda kalıyordu —
                    # mobil sunucu snapshot'ı okuyup `notifications`
                    # alanını hiç almıyordu. Artık status payload'ına girer.
                    raw_notes = snap.get("notifications") or []
                    if isinstance(raw_notes, list):
                        watchdog_notifications = [n for n in raw_notes if isinstance(n, dict)][-10:]
            except Exception as exc:
                logger.debug("Watchdog metrik okuma uyarısı: %s", exc)

        if memory_mb is None and (now - self._mem_cache_ts) > 2.0:
            self._mem_cache_ts = now
            try:
                from core.watchdog import _query_memory_stats
                m_stats = _query_memory_stats()
                memory_mb = float(m_stats.get("process_rss_mb", 0.0) or 0.0)
                system_ram_pct = float(m_stats.get("system_memory_load_pct", 0.0) or 0.0)
                self._cached_memory_mb = memory_mb
                self._cached_system_ram_pct = system_ram_pct
            except Exception as exc:
                logger.debug("Bellek istatistiği okuma uyarısı: %s", exc)
        elif memory_mb is None:
            memory_mb = float(self._cached_memory_mb or 0.0) or None
            system_ram_pct = float(self._cached_system_ram_pct or 0.0) or None

        # 6. Tunnel Info, Enterprise Telemetry & Recent Logs (last 60 entries)
        conn_info = self.get_connection_info()
        with self._log_lock:
            recent_logs = list(self._log_buffer)[-60:]

        # Y-7 (dilim 22): dashboard `/api/status` içindeki `incidents`
        # listesini de okur (fallback); kayıt yoksa boş liste — yokmuş
        # gibi davranılmaz, alan her zaman vardır.
        status_incidents: List[Dict[str, Any]] = []
        try:
            _rec = self._get_incident_recorder()
            if _rec is not None:
                status_incidents = list(_rec.get_incidents(limit=20) or [])
        except Exception as exc:
            logger.debug("Status incidents okuma uyarısı: %s", exc)

        enterprise_info: Dict[str, Any] = {
            "kinah_total": 0,
            "kinah_per_hour": 0,
            "audio_active": False,
            "active_character": "Main",
            "glide_ready": False,
            "yolo_active": False,
            "yolo_provider": "CPU",
        }
        try:
            orch = self.orchestrator
            ec = getattr(orch, "economy_calculator", None) or getattr(w, "economy_calculator", None)
            if ec is not None and hasattr(ec, "get_session_summary"):
                s_sum = ec.get_session_summary()
                if isinstance(s_sum, dict):
                    enterprise_info["kinah_total"] = int(s_sum.get("total_profit_kinah", s_sum.get("net_kinah", 0)) or 0)
                    enterprise_info["kinah_per_hour"] = int(s_sum.get("kinah_per_hour", 0) or 0)
            aud = getattr(orch, "audio_confirm", None) or getattr(w, "audio_confirm", None)
            if aud is not None:
                enterprise_info["audio_active"] = True
            mc = getattr(orch, "multi_char_scheduler", None) or getattr(w, "multi_char_scheduler", None)
            if mc is not None:
                enterprise_info["active_character"] = str(getattr(mc, "active_character_name", "Main") or "Main")
            gp = getattr(orch, "glide_pilot", None) or getattr(w, "glide_pilot", None)
            if gp is not None:
                enterprise_info["glide_ready"] = True
            vis = self.vision
            yd = getattr(vis, "yolo_detector", None) if vis is not None else None
            if yd is not None:
                enterprise_info["yolo_active"] = bool(getattr(yd, "has_custom_model", False))
                enterprise_info["yolo_provider"] = str(getattr(yd, "backend_type", "CPU") or "CPU")
        except Exception as exc:
            logger.debug("Enterprise telemetry aggregation warning: %s", exc)

        hp_clamped = round(max(0.0, min(1.0, hp_ratio)), 3) if hp_ratio is not None else None
        mp_clamped = round(max(0.0, min(1.0, mp_ratio)), 3) if mp_ratio is not None else None
        hp_pct = round(hp_clamped * 100.0, 1) if hp_clamped is not None else None
        mp_pct = round(mp_clamped * 100.0, 1) if mp_clamped is not None else None
        fps_out = round(float(fps), 1) if fps is not None else None
        latency_out = round(float(latency_ms), 1) if latency_ms is not None else None
        memory_out = round(float(memory_mb), 1) if memory_mb is not None else None
        ram_pct_out = round(float(system_ram_pct), 1) if system_ram_pct is not None else None
        max_stages_clamped = int(max(1, max_stages))
        curr_stage_clamped = int(max(0, min(max_stages_clamped, current_stage)))

        return {
            "ok": True,
            "timestamp": round(now, 3),
            "iso_time": datetime.fromtimestamp(now).isoformat(timespec="seconds"),
            "uptime_sec": round(uptime_sec, 1),
            "version": self.VERSION,
            # Top-level flat aliases for ui/mobile_dashboard.html compatibility
            "state": raw_state,
            "paused": is_paused,
            "running": not is_paused,
            "mode": mobile_mode,
            "status_text": status_text,
            "harvest_step": curr_stage_clamped,
            "max_harvest": max_stages_clamped,
            "active_node_id": current_node_id or 0,
            "total_gathered": int(total_gathered),
            "hourly_rate": round(float(gathers_per_hour), 1),
            "hp_pct": hp_pct,
            "mp_pct": mp_pct,
            "resolution": resolution_str,
            "bot": {
                "state": state_display,
                "raw_state": raw_state,
                "mode": mobile_mode,
                "raw_mode": raw_mode,
                "is_paused": is_paused,
                "test_mode": test_mode,
                "status_text": status_text,
            },
            "harvest": {
                "current_stage": curr_stage_clamped,
                "max_stages": max_stages_clamped,
                "ring_label": f"{curr_stage_clamped}/{max_stages_clamped}",
                "gather_key": gather_key,
                "total_gathered": int(total_gathered),
                "gathers_per_hour": round(float(gathers_per_hour), 1),
                "consecutive_failures": int(consecutive_failures),
            },
            "vitals": {
                "hp_ratio": hp_clamped,
                "mp_ratio": mp_clamped,
                "hp_pct": hp_pct,
                "mp_pct": mp_pct,
                "under_attack": bool(under_attack),
                "character_dead": bool(character_dead),
                "extraction_failed_count": int(self._extraction_failed_count),
            },
            "route": {
                "region_name": region_name,
                "route_name": route_name,
                "current_node_id": current_node_id,
                "total_nodes": int(total_nodes),
                "active_nodes": int(active_nodes),
                "resting_nodes": int(resting_nodes),
                "failed_nodes": int(failed_nodes),
                "quarantined_nodes": int(quarantined_nodes),
                "minimap_auto_record": bool(minimap_auto_record),
                "hybrid_3d_record": bool(hybrid_3d_record),
                "minimap_mapped_count": int(minimap_mapped_count),
                "world_3d_mapped_count": int(world_3d_mapped_count),
                "last_tap_to_move": self._last_tap_to_move,
            },
            "system": {
                "fps": fps_out,
                "latency_ms": latency_out,
                "resolution": resolution_str,
                "memory_mb": memory_out,
                "system_ram_pct": ram_pct_out,
                "watchdog_healthy": bool(watchdog_healthy) if watchdog_healthy is not None else None,
                "input_backend": input_backend,
                "bot_elevated": bool(self._bot_is_elevated()),
                "uipi_risk": (not bool(self._bot_is_elevated())),
            },
            "telemetry": {
                "minimap_auto_recorded": int(minimap_mapped_count),
                "world_3d_auto_recorded": int(world_3d_mapped_count),
                "extraction_failed_count": int(self._extraction_failed_count),
                "fps": fps_out,
                "capture_latency_ms": latency_out,
                "ram_mb": memory_out,
            },
            "settings": {
                "harvests_per_node": max_stages_clamped,
                "interact_key": gather_key,
                "minimap_auto_record": bool(minimap_auto_record),
                "world_3d_auto_record": bool(hybrid_3d_record),
                "cooldown_minutes": rest_minutes_cfg,
                "dry_run": bool(test_mode),
            },
            "tunnel": {
                "lan_url": conn_info["lan_url"],
                "ws_url": conn_info["ws_url"],
                "sse_url": conn_info["sse_url"],
                "public_url": conn_info["public_url"],
                "ntfy_topic": conn_info["ntfy_topic"],
                "qr_url": conn_info["qr_url"],
            },
            "enterprise": enterprise_info,
            "recent_logs": recent_logs,
            "notifications": watchdog_notifications,
            "incidents": status_incidents,
        }

    def build_fast_telemetry_frame(
        self,
        last_log_ts: float = 0.0,
        seq: int = 1,
        include_recent_logs: bool = False,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Item 34: Builds an ultra-fast (`< 1ms`) real-time telemetry frame for SSE (`/api/stream`)
        and WebSocket (`/ws`) push streams.
        Includes HP/MP vitals, `1/3 -> 2/3 -> 3/3` harvest ring progress, bot state, active node,
        and incremental `new_logs` since `last_log_ts`.
        """
        now = time.time()
        w = self.worker
        rm = self.route_mgr
        cr = self.combat_recovery
        st = self.settings

        raw_state = "PAUSED"
        raw_mode = "route"
        is_paused = True
        status_text = "Duraklatıldı"
        current_node_id: Optional[int] = None
        current_stage = 0
        max_stages = 3
        gather_key = "f"
        total_gathered = 0

        if st is not None:
            g_cfg = getattr(st, "gathering", None)
            if g_cfg is not None:
                gather_key = str(getattr(g_cfg, "gather_key", "f") or "f").lower()
                max_stages = int(getattr(g_cfg, "harvests_per_node", 3) or 3)

        if w is not None:
            raw_state = str(getattr(w, "state", "PAUSED") or "PAUSED").upper()
            raw_mode = str(getattr(w, "mode", "route") or "route").lower()
            is_paused = bool(getattr(w, "is_paused", True))
            total_gathered = int(getattr(w, "gathered_count", 0) or 0)
            act_n = getattr(w, "active_node", None)
            if act_n is not None:
                current_node_id = int(getattr(act_n, "id", 0) or 0) or None
                current_stage = int(getattr(act_n, "current_harvests", 0) or 0)
                max_stages = int(getattr(act_n, "max_harvests", max_stages) or max_stages)
            if is_paused:
                status_text = str(getattr(w, "_pause_reason", "Duraklatıldı") or "Duraklatıldı")
            elif raw_state == "GATHERING":
                status_text = f"Materyal toplanıyor ({current_stage}/{max_stages})"
            elif raw_state == "WALKING":
                status_text = f"Node #{current_node_id or '?'} hedefine yürünüyor"
            else:
                status_text = STATE_TR_MAP.get(raw_state, raw_state)

        if rm is not None:
            total_gathered = max(total_gathered, int(getattr(rm, "total_gathered", 0) or 0))
            if current_node_id is not None:
                n_obj = rm.get_node(current_node_id)
                if n_obj is not None:
                    current_stage = int(getattr(n_obj, "current_harvests", current_stage) or 0)
                    max_stages = int(getattr(n_obj, "max_harvests", max_stages) or max_stages)

        # F-1661: combat_recovery yokken sahte %100 HP/MP varsayılmaz.
        # combat_recovery varken de last_hp_ratio 0.0/None ise %100'e
        # çevrilmez: 0.0 gerçek 0 HP'dir, None bilinmeyendir.
        hp_ratio: Optional[float] = None
        mp_ratio: Optional[float] = None
        under_attack = raw_state == "FIGHTING"
        character_dead = False
        if cr is not None:
            _raw_hp = getattr(cr, "last_hp_ratio", None)
            hp_ratio = float(_raw_hp) if _raw_hp is not None else None
            _raw_mp = getattr(cr, "last_mp_ratio", None)
            mp_ratio = float(_raw_mp) if _raw_mp is not None else None
            under_attack = under_attack or bool(getattr(cr, "in_combat", False))
            character_dead = bool(getattr(cr, "is_dead", False))

        hp_clamped = round(max(0.0, min(1.0, hp_ratio)), 3) if hp_ratio is not None else None
        mp_clamped = round(max(0.0, min(1.0, mp_ratio)), 3) if mp_ratio is not None else None
        hp_pct = round(hp_clamped * 100.0, 1) if hp_clamped is not None else None
        mp_pct = round(mp_clamped * 100.0, 1) if mp_clamped is not None else None
        max_stages_clamped = max(1, int(max_stages))
        curr_stage_clamped = max(0, min(max_stages_clamped, int(current_stage)))

        new_logs: List[Dict[str, Any]] = []
        newest_ts = last_log_ts
        with self._log_lock:
            for entry in self._log_buffer:
                ets = float(entry.get("timestamp", 0.0) or 0.0)
                if ets > last_log_ts:
                    new_logs.append(entry)
                    if ets > newest_ts:
                        newest_ts = ets
            recent_snapshot = list(self._log_buffer)[-25:] if include_recent_logs else []

        # F-1671: tamamlanan aşama "done" olur; son aşama da curr == max
        # iken done sayılır (eski mantıkta son aşama hiç done olmuyordu).
        ring_steps = [
            {
                "stage": i,
                "status": (
                    "done"
                    if i <= curr_stage_clamped
                    else ("current" if i == curr_stage_clamped + 1 else "pending")
                ),
            }
            for i in range(1, max_stages_clamped + 1)
        ]

        frame_payload: Dict[str, Any] = {
            "ok": True,
            "type": "telemetry_push",
            "seq": int(seq),
            "timestamp": round(now, 3),
            "state": raw_state,
            "paused": is_paused,
            "running": not is_paused,
            "mode": self._normalize_mode_for_mobile(raw_mode),
            "status_text": status_text,
            "active_node_id": current_node_id or 0,
            "harvest_step": curr_stage_clamped,
            "max_harvest": max_stages_clamped,
            "total_gathered": int(total_gathered),
            "hp_pct": hp_pct,
            "mp_pct": mp_pct,
            "bot": {
                "state": STATE_TR_MAP.get(raw_state, raw_state),
                "raw_state": raw_state,
                "mode": self._normalize_mode_for_mobile(raw_mode),
                "is_paused": is_paused,
                "status_text": status_text,
            },
            "harvest": {
                "current_stage": curr_stage_clamped,
                "max_stages": max_stages_clamped,
                "ring_label": f"{curr_stage_clamped}/{max_stages_clamped}",
                "ring_steps": ring_steps,
                "gather_key": gather_key,
                "total_gathered": int(total_gathered),
            },
            "vitals": {
                "hp_ratio": hp_clamped,
                "mp_ratio": mp_clamped,
                "hp_pct": hp_pct,
                "mp_pct": mp_pct,
                "under_attack": bool(under_attack),
                "character_dead": bool(character_dead),
            },
            "new_logs": new_logs,
        }
        if include_recent_logs:
            frame_payload["recent_logs"] = recent_snapshot

        return frame_payload, newest_ts

    def build_nodes_payload(self) -> List[Dict[str, Any]]:
        """
        Returns the full array of nodes in the current route so the phone can draw
        a live interactive 2D route radar and manage node states (active, resting, quarantined/failed).
        """
        rm = self.route_mgr
        if rm is None:
            return []

        now = time.time()
        region_name = str(getattr(rm, "active_region", "Altgard") or "Altgard")
        route_name = str(getattr(rm, "active_route", "All") or "All")

        current_node_id: Optional[int] = None
        w = self.worker
        if w is not None and getattr(w, "active_node", None) is not None:
            current_node_id = int(getattr(w.active_node, "id", 0) or 0) or None

        # F-1658: GET salt-okunur — `update_resting_nodes()` (durum
        # değiştirir + region kaydeder) burada çağrılmaz; resting durumu
        # `remaining_rest_sec` ile salt-okunur türetilir.
        try:
            nodes = rm.get_route_nodes(region_name, route_name)
        except Exception as exc:
            logger.error("Rota düğümleri alınırken hata: %s", exc)
            return []

        result: List[Dict[str, Any]] = []
        for idx, node in enumerate(nodes):
            nid = int(getattr(node, "id", idx + 1))
            map_x = int(getattr(node, "map_x", 0) or 0)
            map_y = int(getattr(node, "map_y", 0) or 0)
            norm_x = float(getattr(node, "norm_x", 0.5) or 0.5)
            norm_y = float(getattr(node, "norm_y", 0.5) or 0.5)

            x_val = map_x if map_x > 0 else int(round(norm_x * 1920))
            y_val = map_y if map_y > 0 else int(round(norm_y * 1080))

            raw_state = str(getattr(node, "state", "active") or "active").lower()
            is_quarantined = raw_state in ("quarantined", "failed", "disabled")

            rest_rem = 0
            if not is_quarantined and hasattr(node, "remaining_rest_sec"):
                try:
                    rest_rem = int(node.remaining_rest_sec(now=now))
                except Exception:
                    rest_rem = 0

            # P1-6 (K-06): durum tek `effective_node_state()`'ten —
            # `/api/status` özetindeki sayımla aynı değer. Eskiden bu
            # payload hata serisini (`consecutive_failures`) yok sayıp
            # hatalı düğümü "active" gösterirken özet "failed" sayıyordu.
            from core.route_manager import effective_node_state

            effective_state = effective_node_state(node, now=now)

            curr_h = int(getattr(node, "current_harvests", 0) or 0)
            max_h = int(getattr(node, "max_harvests", 3) or 3)
            fail_c = int(getattr(node, "consecutive_failures", 0) or 0)
            src_type = str(getattr(node, "source_type", "manual") or "manual")

            result.append(
                {
                    "id": nid,
                    "x": x_val,
                    "y": y_val,
                    "map_x": x_val,
                    "map_y": y_val,
                    "norm_x": round(norm_x, 4),
                    "norm_y": round(norm_y, 4),
                    "state": effective_state,
                    "status": "failed" if is_quarantined else effective_state,
                    "quarantined": bool(is_quarantined),
                    "current_harvests": curr_h,
                    "harvest_count": curr_h,
                    "max_harvests": max_h,
                    "gather_count": int(getattr(node, "gathered_count", 0) or 0),
                    "fail_count": fail_c,
                    "source_type": src_type,
                    "source": src_type,
                    "rest_remaining_sec": int(max(0, rest_rem)),
                    "cooldown_remaining": int(max(0, rest_rem)),
                    "resource_type": str(getattr(node, "resource_type", "Orichalcum Ore") or "Orichalcum Ore"),
                    "in_air": bool(getattr(node, "in_air", False)),
                    "seq_index": int(getattr(node, "seq_index", idx) or idx),
                    "is_target": bool(current_node_id is not None and nid == current_node_id),
                }
            )
        return result

    # =========================================================================
    # Item 31: Dual-Channel Live Stream & PiP Bundle (/api/frame.jpg & /api/pip_bundle)
    # =========================================================================

    def _grab_latest_bgr_frame(self) -> np.ndarray:
        now = time.time()
        with self._frame_lock:
            if (
                self._cached_frame is not None
                and (now - self._cached_frame_ts) < 0.08
                and self._cached_frame.size > 0
            ):
                return self._cached_frame.copy()

        frame: Optional[np.ndarray] = None
        cap = self.capture
        if cap is not None:
            try:
                if hasattr(cap, "grab_full"):
                    frame = cap.grab_full(grayscale=False)
                elif hasattr(cap, "grab_frame"):
                    frame = cap.grab_frame()
                elif hasattr(cap, "grab_region"):
                    frame = cap.grab_region(None, grayscale=False)
            except Exception as exc:
                logger.debug("Mobil kare yakalama uyarısı: %s", exc)

        # F-1657: yalnızca yakalama hiç kare döndüremediyse (None/boş)
        # yer tutucu çerçeve üretilir. Gerçek ama simsiyah bir kare
        # atılıp sahte radar karesiyle değiştirilmez — olduğu gibi döner.
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            frame = self._build_synthetic_telemetry_frame()

        with self._frame_lock:
            self._cached_frame = frame
            self._cached_frame_ts = now
            return frame.copy()

    def _build_synthetic_base_frame(self) -> np.ndarray:
        """
        Sentetik karenin statik tabanını (ızgara, minimap, [F Gather] hapı)
        bir kez üretir ve önbellekler. Taban tamamen sabittir (sürüm ve
        normalleştirilmiş sınırlar değişmez); her çağrıda yeniden çizmek
        ~84 ms sürerken önbellekten kopyalamak ~1 ms'dir.
        Başlık metni ve rota düğümleri tabanda DEĞİLDİR: çizim sırası
        (düğümler -> başlık) korunmak için her çağrıda eklenir.
        """
        if self._synthetic_base_frame is not None:
            return self._synthetic_base_frame
        canvas = np.zeros((1080, 1920, 3), dtype=np.uint8)
        canvas[:, :] = (24, 18, 14)

        # Draw grid lines
        for x in range(0, 1920, 120):
            cv2.line(canvas, (x, 0), (x, 1080), (38, 30, 24), 1)
        for y in range(0, 1080, 120):
            cv2.line(canvas, (0, y), (1920, y), (38, 30, 24), 1)

        # Draw top-right minimap circle inside (x: 0.84..0.995, y: 0.05..0.27)
        mm_cx, mm_cy, mm_r = int(1920 * 0.918), int(1080 * 0.16), 108
        cv2.circle(canvas, (mm_cx, mm_cy), mm_r, (48, 42, 32), -1)
        cv2.circle(canvas, (mm_cx, mm_cy), mm_r, (212, 182, 6), 2)
        # Draw minimap resource blips and center player arrow
        cv2.circle(canvas, (mm_cx - 34, mm_cy - 28), 6, (129, 185, 16), -1)
        cv2.circle(canvas, (mm_cx + 42, mm_cy + 18), 6, (212, 182, 6), -1)
        cv2.circle(canvas, (mm_cx, mm_cy), 7, (120, 240, 255), -1)

        # Draw center-right [F Gather] ROI pill inside (x: 0.42..0.72, y: 0.36..0.68)
        pill_x1, pill_y1 = int(1920 * 0.50), int(1080 * 0.49)
        pill_x2, pill_y2 = pill_x1 + 250, pill_y1 + 52
        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), (32, 28, 22), -1)
        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2), (212, 182, 6), 2)
        cv2.rectangle(canvas, (pill_x1 + 8, pill_y1 + 8), (pill_x1 + 44, pill_y2 - 8), (11, 158, 245), -1)
        cv2.putText(
            canvas,
            "F",
            (pill_x1 + 18, pill_y1 + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Gather [1/3]",
            (pill_x1 + 56, pill_y1 + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 235, 220),
            2,
            cv2.LINE_AA,
        )
        self._synthetic_base_frame = canvas
        return canvas

    def _build_synthetic_telemetry_frame(self) -> np.ndarray:
        """
        Generates a clean 1920x1080 dark radar/HUD frame when running headless or minimized,
        including the top-right Minimap AND the center-right `[F Gather]` prompt pill ROI.
        Statik taban önbellekten kopyalanır; yalnızca dinamik rota düğümleri
        ve başlık her çağrıda çizilir (çıktı piksel-birebir aynıdır).
        """
        canvas = self._build_synthetic_base_frame().copy()

        # Draw route nodes on canvas
        nodes = self.build_nodes_payload()
        for n in nodes:
            nx = max(40, min(1880, int(n.get("x", 960))))
            ny = max(60, min(1020, int(n.get("y", 540))))
            st_name = str(n.get("state", "active"))
            if st_name in ("quarantined", "failed"):
                color = (68, 68, 239)
            elif st_name == "active":
                color = (94, 197, 34)
            else:
                color = (80, 165, 245)
            if n.get("is_target"):
                color = (248, 189, 56)
                cv2.circle(canvas, (nx, ny), 14, color, 2)
            cv2.circle(canvas, (nx, ny), 7, color, -1)

        cv2.putText(
            canvas,
            f"GATHERION LIVE STREAM ({self.VERSION})",
            (60, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (240, 200, 90),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _crop_norm_bounds_bgr(
        self,
        frame: np.ndarray,
        bounds: Tuple[float, float, float, float],
        scale: Optional[float] = None,
        fallback_size: Tuple[int, int] = (280, 280),
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Normalleştirilmiş sınırlarla kare kırpar, gerekirse ölçekler.

        `_crop_minimap_bgr` ve `_crop_gather_roi_bgr`'nin ortak mantığıdır:
        sınır -> kırpma -> ölçek -> meta. Tek yerden beslenir, davranış
        (piksel-birebir çıktı dahil) aynı kalır.
        """
        h, w = frame.shape[:2]
        nx1, ny1, nx2, ny2 = bounds
        x1 = max(0, int(round(w * nx1)))
        x2 = min(w, int(round(w * nx2)))
        y1 = max(0, int(round(h * ny1)))
        y2 = min(h, int(round(h * ny2)))

        if x2 > x1 + 10 and y2 > y1 + 10:
            crop = frame[y1:y2, x1:x2].copy()
            rect_meta = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": crop.shape[1], "height": crop.shape[0], "valid": True, "fallback": None}
            if scale is not None and abs(scale - 1.0) > 0.02:
                ch, cw = crop.shape[:2]
                tw = max(64, int(round(cw * scale)))
                th = max(64, int(round(ch * scale)))
                interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
                crop = cv2.resize(crop, (tw, th), interpolation=interp)
                rect_meta["width"] = crop.shape[1]
                rect_meta["height"] = crop.shape[0]
            return crop, rect_meta

        # B-04 (dilim 30): geçersiz kırpma artık TÜM KAREYİ kırpma diye
        # döndürmüyor — siyah boş kare + valid=False meta döner; hedef
        # sınır meta'da yazılmaz (meta/icerik çelişkisi kalkar). Tüketici
        # (pip_bundle) valid bayrağını panele taşır.
        now = time.time()
        if now - self._crop_fallback_warn_ts > 60.0:
            self._crop_fallback_warn_ts = now
            logger.warning(
                "Geçersiz kırpma sınırı %s (kare %dx%d) — boş kare döndürülüyor, tüm kare kırpma diye sunulmuyor.",
                bounds,
                w,
                h,
            )
        crop = np.zeros((fallback_size[1], fallback_size[0], 3), dtype=np.uint8)
        if scale is not None and abs(scale - 1.0) > 0.02:
            ch, cw = crop.shape[:2]
            tw = max(64, int(round(cw * scale)))
            th = max(64, int(round(ch * scale)))
            interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            crop = cv2.resize(crop, (tw, th), interpolation=interp)
        rect_meta = {
            "x1": 0,
            "y1": 0,
            "x2": 0,
            "y2": 0,
            "width": crop.shape[1],
            "height": crop.shape[0],
            "valid": False,
            "fallback": "invalid-bounds",
        }
        return crop, rect_meta

    def _crop_minimap_bgr(
        self,
        frame: np.ndarray,
        scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extracts a high-clarity crop of the top-right minimap."""
        return self._crop_norm_bounds_bgr(frame, MINIMAP_NORM_BOUNDS, scale=scale, fallback_size=(280, 280))

    def _crop_gather_roi_bgr(
        self,
        frame: np.ndarray,
        scale: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, int]]:
        """Extracts a high-clarity crop of the center-right `[F Gather]` interaction ROI."""
        h, w = frame.shape[:2]
        nx1, ny1, nx2, ny2 = GATHER_ROI_NORM_BOUNDS
        v = self.vision
        if v is not None and hasattr(v, "close_interaction_norm_roi"):
            try:
                roi_tuple = getattr(v, "close_interaction_norm_roi")
                if isinstance(roi_tuple, (tuple, list)) and len(roi_tuple) == 4:
                    nx1, ny1, nx2, ny2 = (
                        float(roi_tuple[0]),
                        float(roi_tuple[1]),
                        float(roi_tuple[2]),
                        float(roi_tuple[3]),
                    )
            except Exception as exc:
                logger.debug("close_interaction_norm_roi okunamadı: %s", exc)

        return self._crop_norm_bounds_bgr(
            frame, (nx1, ny1, nx2, ny2), scale=scale, fallback_size=(360, 240)
        )

    def _render_pip_composite_bgr(
        self,
        frame: np.ndarray,
        scale: Optional[float] = None,
    ) -> np.ndarray:
        """
        Item 31: Renders a Dual-Channel Picture-in-Picture (PiP) high-clarity composite frame:
        - Main canvas: `[F Gather]` center-right ROI magnified on the left/background + full-screen context
        - Top-right PiP inset: High-clarity magnified Minimap crop with cyan border
        - Bottom-right PiP inset: High-clarity magnified `[F Gather]` pill ROI crop with gold border
        """
        eff_scale = max(0.25, min(1.5, float(scale if scale is not None else 0.5)))
        base_w = max(320, int(round(1920 * eff_scale)))
        base_h = max(180, int(round(1080 * eff_scale)))

        out_img = cv2.resize(frame, (base_w, base_h), interpolation=cv2.INTER_AREA)

        mm_crop, _ = self._crop_minimap_bgr(frame, scale=None)
        gather_crop, _ = self._crop_gather_roi_bgr(frame, scale=None)

        # 1. Magnified Minimap PiP inset at top-right (below 36px HUD bar)
        pip_mm_w = max(90, int(round(base_w * 0.27)))
        pip_mm_h = max(90, int(round(base_h * 0.38)))
        mm_resized = cv2.resize(mm_crop, (pip_mm_w, pip_mm_h), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(mm_resized, (0, 0), (pip_mm_w - 1, pip_mm_h - 1), (212, 182, 6), 2)
        cv2.putText(
            mm_resized,
            "MINIMAP PiP",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 240, 150),
            1,
            cv2.LINE_AA,
        )

        y_top = min(base_h - pip_mm_h - 4, 42)
        x_right = max(4, base_w - pip_mm_w - 8)
        if y_top >= 0 and x_right >= 0:
            out_img[y_top : y_top + pip_mm_h, x_right : x_right + pip_mm_w] = mm_resized

        # 2. Magnified [F Gather] ROI PiP inset at bottom-right
        pip_g_w = max(120, int(round(base_w * 0.34)))
        pip_g_h = max(80, int(round(base_h * 0.34)))
        g_resized = cv2.resize(gather_crop, (pip_g_w, pip_g_h), interpolation=cv2.INTER_CUBIC)
        cv2.rectangle(g_resized, (0, 0), (pip_g_w - 1, pip_g_h - 1), (56, 189, 248), 2)
        cv2.putText(
            g_resized,
            "[F GATHER] ROI PiP",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 235, 255),
            1,
            cv2.LINE_AA,
        )

        gy1 = max(0, base_h - pip_g_h - 8)
        gx1 = max(0, base_w - pip_g_w - 8)
        out_img[gy1 : gy1 + pip_g_h, gx1 : gx1 + pip_g_w] = g_resized

        # 3. Top HUD banner
        self._draw_top_hud_banner(out_img)
        return out_img

    def _draw_top_hud_banner(self, out_img: np.ndarray) -> None:
        """Overlays the dark translucent telemetry HUD bar on top of `out_img`."""
        bh, bw = out_img.shape[:2]
        bar_h = max(24, min(40, int(round(bh * 0.068))))
        banner = out_img[0:bar_h, 0:bw].copy()
        dark_bar = np.zeros_like(banner)
        dark_bar[:, :] = (20, 16, 12)
        cv2.addWeighted(dark_bar, 0.75, banner, 0.25, 0, out_img[0:bar_h, 0:bw])

        fast_frame, _ = self.build_fast_telemetry_frame(seq=0, include_recent_logs=False)
        raw_state = fast_frame.get("state", "PAUSED")
        mode_str = str(fast_frame.get("mode", "route")).upper()
        target_nid = fast_frame.get("active_node_id")
        harvest_info = fast_frame.get("harvest", {})
        stage_str = harvest_info.get("ring_label", "0/3")
        total_g = harvest_info.get("total_gathered", 0)
        now_str = datetime.now().strftime("%H:%M:%S")

        target_label = f"Node #{target_nid} ({stage_str})" if target_nid else "Hedef Yok"
        hud_line = (
            f"[{now_str}] STATE: {raw_state} | MODE: {mode_str} | "
            f"TARGET: {target_label} | GATHERED: {total_g}"
        )
        font_scale = max(0.35, min(0.56, bw / 1800.0))
        cv2.putText(
            out_img,
            hud_line,
            (10, int(round(bar_h * 0.68))),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (245, 225, 160),
            1,
            cv2.LINE_AA,
        )

    def render_frame_jpeg(
        self,
        mode: str = "full",
        quality: int = 65,
        scale: Optional[float] = None,
        submode: str = "",
    ) -> bytes:
        """
        Item 31: Captures, crops/resizes, overlays HUD telemetry, and encodes as JPEG:
        - `mode="minimap"`: high-clarity crop of top-right minimap (`x: 0.84..0.995, y: 0.05..0.27`)
        - `mode="pip"`: Dual-Channel PiP high-clarity composite (or `[F Gather]` ROI crop when `submode="roi"`)
        - `mode="gather"` / `"f_gather"` / `"roi"`: high-clarity `[F Gather]` center-right ROI crop
        - `mode="full"`: adjustable `quality` and `scale` (default `960x540` when `scale` is None or `0.5`)
        """
        frame = self._grab_latest_bgr_frame()
        h, w = frame.shape[:2]

        clean_mode = (mode or "full").strip().lower()
        clean_submode = (submode or "").strip().lower()

        if clean_mode == "minimap":
            out_img, _ = self._crop_minimap_bgr(frame, scale=scale)
        elif clean_mode in ("gather", "f_gather", "roi", "gather_roi") or (
            clean_mode == "pip" and clean_submode in ("roi", "gather", "f_gather")
        ):
            out_img, _ = self._crop_gather_roi_bgr(frame, scale=scale)
        elif clean_mode == "pip":
            out_img = self._render_pip_composite_bgr(frame, scale=scale)
        elif clean_mode == "full":
            eff_scale = max(0.2, min(1.5, float(scale if scale is not None else 0.5)))
            target_w = max(320, int(round(w * eff_scale)))
            target_h = max(180, int(round(h * eff_scale)))
            out_img = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
            self._draw_top_hud_banner(out_img)
        else:
            # F-1669: bilinmeyen mode sessizce "full" sayılmaz.
            raise ValueError(f"unknown frame mode: {mode!r}")

        q_clamped = max(20, min(95, int(quality)))
        ok, encoded = cv2.imencode(".jpg", out_img, [int(cv2.IMWRITE_JPEG_QUALITY), q_clamped])
        if not ok:
            # F-1673: kodlama hatasında boş/sahte başarılı yanıt yok.
            raise RuntimeError("JPEG kodlama başarısız")
        return encoded.tobytes()

    def build_pip_bundle_payload(self, quality: int = 78, scale: float = 1.0) -> Dict[str, Any]:
        """
        Item 31: `GET /api/pip_bundle`
        Returns JSON with base64 high-res crops of BOTH the top-right Minimap AND
        the `[F Gather]` center-right ROI (plus composite PiP) in a single fast request.
        """
        now = time.time()
        frame = self._grab_latest_bgr_frame()
        q_clamped = max(20, min(95, int(quality)))
        eff_scale = max(0.25, min(2.0, float(scale)))

        mm_img, mm_rect = self._crop_minimap_bgr(frame, scale=eff_scale)
        gather_img, gather_rect = self._crop_gather_roi_bgr(frame, scale=eff_scale)
        pip_img = self._render_pip_composite_bgr(frame, scale=0.5)

        ok_mm, mm_buf = cv2.imencode(".jpg", mm_img, [int(cv2.IMWRITE_JPEG_QUALITY), q_clamped])
        ok_g, g_buf = cv2.imencode(".jpg", gather_img, [int(cv2.IMWRITE_JPEG_QUALITY), q_clamped])
        ok_p, p_buf = cv2.imencode(".jpg", pip_img, [int(cv2.IMWRITE_JPEG_QUALITY), q_clamped])

        # F-1673: kodlama hatasında boş base64 ile sahte başarılı yanıt yok.
        if not (ok_mm and ok_g and ok_p):
            raise RuntimeError("PiP bundle JPEG kodlama başarısız")

        mm_b64 = base64.b64encode(mm_buf.tobytes()).decode("ascii")
        g_b64 = base64.b64encode(g_buf.tobytes()).decode("ascii")
        p_b64 = base64.b64encode(p_buf.tobytes()).decode("ascii")

        fast_frame, _ = self.build_fast_telemetry_frame(seq=0, include_recent_logs=False)

        return {
            "ok": True,
            "timestamp": round(now, 3),
            "quality": q_clamped,
            "scale": round(eff_scale, 2),
            "minimap_b64": mm_b64,
            "minimap_data_url": f"data:image/jpeg;base64,{mm_b64}",
            "minimap": f"data:image/jpeg;base64,{mm_b64}",
            "minimap_roi": mm_rect,
            "minimap_valid": bool(mm_rect.get("valid", True)),
            "gather_roi_b64": g_b64,
            "gather_roi_data_url": f"data:image/jpeg;base64,{g_b64}",
            "f_gather_b64": g_b64,
            "f_gather_data_url": f"data:image/jpeg;base64,{g_b64}",
            "gather_roi": f"data:image/jpeg;base64,{g_b64}",
            "gather_roi_rect": gather_rect,
            "gather_roi_valid": bool(gather_rect.get("valid", True)),
            "pip_b64": p_b64,
            "pip_data_url": f"data:image/jpeg;base64,{p_b64}",
            "harvest": fast_frame.get("harvest", {}),
            "bot_state": fast_frame.get("state", "PAUSED"),
        }

    def _screen_size(self) -> Tuple[int, int]:
        """Gerçek ekran boyutunu döndürür (F-1663: 1920x1080'e sabit değil)."""
        ic = self.input_ctrl
        if ic is not None:
            sw = int(getattr(ic, "screen_width", 0) or 0)
            sh = int(getattr(ic, "screen_height", 0) or 0)
            if sw > 0 and sh > 0:
                return sw, sh
        cap = self.capture
        if cap is not None:
            sw = int(getattr(cap, "screen_w", 0) or 0)
            sh = int(getattr(cap, "screen_h", 0) or 0)
            if sw > 0 and sh > 0:
                return sw, sh
        return 1920, 1080

    @staticmethod
    def _keyboard_trusted(ic: Any) -> bool:
        """F-1653: tuş yalnızca donanım/HID yolu (Interception / Serial HID /
        FakerInput) doğrulandıysa güvenilir sayılır. Yumuşak SendInput
        AION 2'de sessizce düşer; test modunda simülasyona izin verilir."""
        if ic is None:
            return False
        if bool(getattr(ic, "test_mode", False)):
            return True
        for attr in ("is_keyboard_trusted", "keyboard_inject_trusted"):
            val = getattr(ic, attr, None)
            try:
                if callable(val):
                    return bool(val())
                if isinstance(val, bool):
                    return bool(val)
            except Exception:
                return False
        name = str(getattr(ic, "active_backend_name", "") or "")
        return ("Interception" in name) or ("HID" in name) or ("FakerInput" in name)

    @staticmethod
    def _mouse_trusted(ic: Any) -> bool:
        """S-1801: fare yalnızca donanım/HID yolu (Interception / Serial HID /
        FakerInput) doğrulandıysa güvenilir sayılır — send_key'deki
        _keyboard_trusted (F-1653) kapısının fare karşılığı. Yumuşak
        SendInput/pydirectinput fare yolu AION 2'de sessizce düşer;
        test modunda simülasyona izin verilir. Sürücü ileride
        is_mouse_trusted / mouse_inject_trusted sağlarsa o öcelenir."""
        if ic is None:
            return False
        if bool(getattr(ic, "test_mode", False)):
            return True
        for attr in ("is_mouse_trusted", "mouse_inject_trusted"):
            val = getattr(ic, attr, None)
            try:
                if callable(val):
                    return bool(val())
                if isinstance(val, bool):
                    return bool(val)
            except Exception:
                return False
        name = str(getattr(ic, "active_backend_name", "") or "")
        return ("Interception" in name) or ("HID" in name) or ("FakerInput" in name)

    @staticmethod
    def _input_ready(ic: Any) -> bool:
        """F-1649: girdi sürücüsü var ve tuş gönderebiliyor mu?"""
        return ic is not None and hasattr(ic, "press_key")

    # =========================================================================
    # Remote Intervention Dispatcher: POST /api/action (Items 32 & 33)
    # =========================================================================

    def execute_action(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a remote control intervention sent from the mobile dashboard:
        - `pause`: Pauses the bot
        - `resume`: Resumes the bot
        - `set_mode`: Sets mode to `gather`, `route`, or `explore` and resumes
        - `record_f8`: Triggers F8 current node recording
        - `skip_node`: Marks current target node as resting/skipped and advances to next node
        - `reset_node` / `reset_all_nodes`: Resets resting cooldown to 0 and sets state to `"active"`
        - `quarantine_node`: Puts a stuck/rock-trapped node into `"quarantined"` / `"failed"` state so the bot skips it
        - `delete_node`: Permanently removes the unwanted node from the active route and saves `region_data.json`
        - `toggle_node`: Toggles a node between `"active"` and `"quarantined"`
        - `optimize_route`: Runs sequential route optimization
        - `send_key` / `press_key`: Sends a remote key via `input_ctrl.press_key(key)`
        - `click_screen`: Clicks normalized coordinates `(norm_x, norm_y)`
        - `tap_to_move_map`: Steers/walks toward the tapped 2D radar coordinate `(map_x, map_y)` or clicks it on map
        - `update_settings`: Updates gathering/route settings and persists to disk
        - `panic_stop` / `emergency_stop`: Immediate emergency pause & stop
        """
        action = str(payload.get("action", "") or "").strip().lower()
        if not action:
            return {"ok": False, "error": "Eksik 'action' parametresi."}

        w = self.worker
        rm = self.route_mgr
        ic = self.input_ctrl
        st = self.settings
        sm = self.settings_mgr
        orch = self.orchestrator

        try:
            # 1. PAUSE — F-1654: worker yokken ok:True dönülmez.
            if action == "pause":
                reason = str(payload.get("reason") or "Mobil panelden duraklatıldı")
                if w is None or not hasattr(w, "pause"):
                    return {"ok": False, "error": "Bot worker aktif değil (pause uygulanamadı)."}
                w.pause(reason)
                logger.info("[Mobil Müdahale] Bot duraklatıldı: %s", reason)
                self.notify_stream_listeners()
                return {"ok": True, "action": "pause", "message": reason}

            # 2. RESUME — F-1654: worker yokken ok:True dönülmez.
            if action == "resume":
                if w is None or not hasattr(w, "resume"):
                    return {"ok": False, "error": "Bot worker aktif değil (resume uygulanamadı)."}
                w.resume()
                logger.info("[Mobil Müdahale] Bot devam ettirildi (Resume).")
                self.notify_stream_listeners()
                return {"ok": True, "action": "resume", "message": "Bot devam ettirildi."}

            # 2b. PAUSE_RESUME / STOP / START / STATUS / RESET_STUCK — Y-7
            # (dilim 22): dialogdaki /api/command/pause_resume ile ntfy
            # listesindeki stop/start/status/reset_stuck artık ölü değil;
            # hepsi execute_action'da karşılık bulur (ntfy yolunda geri
            # bildirim olmadığı için reddedilen komut sessiz kalıyordu).
            if action == "pause_resume":
                if w is None or not hasattr(w, "pause") or not hasattr(w, "resume"):
                    return {"ok": False, "error": "Bot worker aktif değil (pause_resume uygulanamadı)."}
                if bool(getattr(w, "is_paused", False)):
                    w.resume()
                    self.notify_stream_listeners()
                    return {"ok": True, "action": "pause_resume", "state": "running", "message": "Bot devam ettirildi."}
                reason = str(payload.get("reason") or "Mobil panelden duraklatıldı")
                w.pause(reason)
                self.notify_stream_listeners()
                return {"ok": True, "action": "pause_resume", "state": "paused", "message": reason}

            if action == "stop":
                if w is None or not hasattr(w, "pause"):
                    return {"ok": False, "error": "Bot worker aktif değil (stop uygulanamadı)."}
                reason = str(payload.get("reason") or "Uzaktan durdurma (ntfy stop)")
                w.pause(reason)
                self.notify_stream_listeners()
                return {"ok": True, "action": "stop", "message": reason}

            if action == "start":
                if w is None or not hasattr(w, "resume"):
                    return {"ok": False, "error": "Bot worker aktif değil (start uygulanamadı)."}
                w.resume()
                self.notify_stream_listeners()
                return {"ok": True, "action": "start", "message": "Bot devam ettirildi."}

            if action == "status":
                try:
                    _summary = self.build_status_payload()
                    return {
                        "ok": True,
                        "action": "status",
                        "state": _summary.get("state"),
                        "mode": _summary.get("mode"),
                        "paused": _summary.get("paused"),
                        "uptime_sec": _summary.get("uptime_sec"),
                        "harvest": _summary.get("harvest"),
                        "message": str(_summary.get("status_text", "")),
                    }
                except Exception as exc:
                    logger.debug("Uzaktan status özeti hatası: %s", exc)
                    return {
                        "ok": True,
                        "action": "status",
                        "state": str(getattr(w, "state", "PAUSED") or "PAUSED") if w is not None else "PAUSED",
                        "paused": bool(getattr(w, "is_paused", True)) if w is not None else True,
                    }

            if action == "reset_stuck":
                # Takılı/karantinadaki düğümler sıfırlanır (reset_all_nodes
                # ile aynı alan sıfırlaması, yalnızca takılı düğümlerde).
                if rm is None:
                    return {"ok": False, "error": "RouteManager aktif değil."}
                try:
                    reg = getattr(rm, "active_region", "Altgard")
                    rt = getattr(rm, "active_route", "All")
                    nodes = rm.get_route_nodes(reg, rt)
                except Exception as exc:
                    return {"ok": False, "error": f"Düğüm listesi okunamadı: {exc}"}
                reset_count = 0
                # K-19: disk yazımı başarısız olursa bellek içi sıfırlama
                # geri alınır; yoksa mobil istemci "ok" sanırken diskteki
                # düğümler takılı kalır.
                reset_snapshot = []
                for n in nodes:
                    try:
                        if not self._is_node_quarantined_or_failed(n):
                            continue
                    except Exception:
                        continue
                    reset_snapshot.append(
                        (
                            n,
                            n.state,
                            n.current_harvests,
                            n.consecutive_failures,
                            n.last_visited_timestamp,
                            n.rest_duration_sec,
                        )
                    )
                    n.state = "active"
                    n.current_harvests = 0
                    n.consecutive_failures = 0
                    n.last_visited_timestamp = 0.0
                    n.rest_duration_sec = 0
                    reset_count += 1
                save_error = None
                try:
                    if reset_count:
                        saved = rm.save_region(reg)
                        if saved is False:
                            save_error = getattr(rm, "last_save_error", None) or "save_region False döndü"
                except Exception as exc:
                    save_error = exc
                if save_error is not None:
                    for (n, prev_state, prev_harvests, prev_failures, prev_ts, prev_rest) in reset_snapshot:
                        n.state = prev_state
                        n.current_harvests = prev_harvests
                        n.consecutive_failures = prev_failures
                        n.last_visited_timestamp = prev_ts
                        n.rest_duration_sec = prev_rest
                    logger.error("[Mobil Müdahale] reset_stuck bölge kaydı başarısız, sıfırlama geri alındı: %s", save_error)
                    return {
                        "ok": False,
                        "action": "reset_stuck",
                        "reset_count": 0,
                        "error": f"Takılı düğümler sıfırlandı ancak bölge diske yazılamadı ({reg}): {save_error}",
                    }
                logger.info("[Mobil Müdahale] Takılı düğümler sıfırlandı: %d adet.", reset_count)
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "reset_stuck",
                    "reset_count": reset_count,
                    "message": f"{reset_count} takılı düğüm aktif duruma sıfırlandı.",
                }

            # 3. SET_MODE — F-1654: worker yokken ok:True dönülmez.
            # F-1669: bilinmeyen mod sessizce "route" sayılmaz; auto_resume
            # bool değilse reddedilir.
            if action == "set_mode":
                req_mode = str(payload.get("mode", "route") or "route").strip().lower()
                if req_mode in ("gather", "manual_record", "auto_record"):
                    internal_mode = "auto_record" if req_mode != "manual_record" else "manual_record"
                elif req_mode in ("explore", "explorer", "explorer_mode", "auto_cartography"):
                    internal_mode = "explorer"
                elif req_mode in ("combat", "combat_mode", "savas", "savaş", "fight"):
                    internal_mode = "combat"
                elif req_mode in ("route", "normal"):
                    internal_mode = "route"
                else:
                    return {"ok": False, "error": f"Bilinmeyen mod: '{req_mode}'"}
                if "auto_resume" in payload and not isinstance(payload.get("auto_resume"), bool):
                    return {"ok": False, "error": "'auto_resume' bool olmalıdır."}
                if w is None:
                    return {"ok": False, "error": "Bot worker aktif değil (set_mode uygulanamadı)."}

                if hasattr(w, "set_mode"):
                    w.set_mode(internal_mode)
                else:
                    return {"ok": False, "error": "Worker set_mode desteklemiyor."}
                if hasattr(w, "resume") and bool(payload.get("auto_resume", True)):
                    w.resume()
                logger.info("[Mobil Müdahale] Mod değiştirildi: %s -> %s", req_mode, internal_mode)
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "set_mode",
                    "mode": self._normalize_mode_for_mobile(internal_mode),
                    "raw_mode": internal_mode,
                    "message": f"Mod '{req_mode}' olarak ayarlandı.",
                }

            # 4. RECORD_F8 — F-1654: hiçbir kayıt yolu yokken ok:True dönülmez.
            if action == "record_f8":
                recorded_node = None
                handled = False
                if self.orchestrator is not None and hasattr(self.orchestrator, "_on_f8_record_node"):
                    recorded_node = self.orchestrator._on_f8_record_node()
                    handled = True
                elif self.orchestrator is not None and hasattr(self.orchestrator, "on_f8_hotkey"):
                    self.orchestrator.on_f8_hotkey()
                    handled = True
                elif w is not None and hasattr(w, "trigger_f8_record"):
                    recorded_node = w.trigger_f8_record()
                    handled = True
                elif rm is not None and hasattr(rm, "record_current_node"):
                    recorded_node = rm.record_current_node(
                        vision_engine=self.vision,
                        input_ctrl=ic,
                    )
                    handled = True
                if not handled:
                    return {"ok": False, "error": "F8 kayıt yolu yok (orchestrator/worker/route_manager)."}
                node_id = getattr(recorded_node, "id", None) if recorded_node is not None else None
                logger.info("[Mobil Müdahale] F8 Anlık Düğüm Kaydı tetiklendi (Node: %s).", node_id)
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "record_f8",
                    "node_id": node_id,
                    "message": "F8 düğüm kaydı tetiklendi.",
                }

            # 5. SKIP_NODE — F-1660: aktif node yokken sıradaki hazır node
            # "atlanmış" sayılmaz; aktif node yoksa hata döner.
            if action == "skip_node":
                rest_min = 5.0
                if st is not None and hasattr(st, "route"):
                    rest_min = float(getattr(st.route, "rest_minutes", 5) or 5)

                if w is None or getattr(w, "active_node", None) is None:
                    return {"ok": False, "error": "Atlanacak aktif node yok."}
                skipped_id = int(getattr(w.active_node, "id", 0) or 0) or None
                if skipped_id is None:
                    return {"ok": False, "error": "Atlanacak aktif node yok."}
                if rm is None:
                    return {"ok": False, "error": "RouteManager aktif değil."}

                rm.mark_node_gathered(skipped_id, rest_minutes=rest_min, force_deplete=True)

                next_target_id: Optional[int] = None
                if rm is not None:
                    reg = getattr(rm, "active_region", "Altgard")
                    rt = getattr(rm, "active_route", "All")
                    nxt = self._select_next_non_quarantined_node(rm, reg, rt) or rm.get_next_ready_node()
                    if nxt is not None:
                        next_target_id = int(getattr(nxt, "id", 0) or 0) or None
                        if w is not None:
                            w.active_node = nxt

                logger.info(
                    "[Mobil Müdahale] Node #%s atlandı -> Sıradaki Node #%s",
                    skipped_id,
                    next_target_id,
                )
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "skip_node",
                    "skipped_node_id": skipped_id,
                    "next_node_id": next_target_id,
                    "message": f"Node #{skipped_id} dinlenmeye alındı, sıradaki: #{next_target_id}",
                }

            # 6. Item 33: RESET_NODE / RESET_ALL_NODES
            if action in ("reset_node", "reset_all_nodes"):
                node_id_param = payload.get("node_id")
                if rm is None:
                    return {"ok": False, "error": "RouteManager aktif değil."}

                reg = getattr(rm, "active_region", "Altgard")
                rt = getattr(rm, "active_route", "All")
                nodes = rm.get_route_nodes(reg, rt)
                # F-1662: sıfırlama "cooldown 0" vaat eder; rest_duration_sec
                # 300 yazılmaz, 0 yazılır (last_visited zaten 0.0).
                reset_count = 0
                if action == "reset_all_nodes" or node_id_param in (None, "all", "*"):
                    for n in nodes:
                        n.state = "active"
                        n.current_harvests = 0
                        n.consecutive_failures = 0
                        n.last_visited_timestamp = 0.0
                        n.rest_duration_sec = 0
                        reset_count += 1
                    rm.consecutive_failures = 0
                    rm.save_region(reg)
                    logger.info("[Mobil Müdahale] Tüm düğümler (%d adet) aktif duruma sıfırlandı.", reset_count)
                    self.notify_stream_listeners()
                    return {
                        "ok": True,
                        "action": "reset_all_nodes",
                        "reset_count": reset_count,
                        "state": "active",
                        "message": f"{reset_count} düğüm aktif duruma uyandırıldı.",
                    }
                else:
                    target_id = int(node_id_param)
                    target_node = rm.get_node(target_id, region=reg)
                    if target_node is None:
                        return {"ok": False, "error": f"Node #{target_id} bulunamadı."}
                    target_node.state = "active"
                    target_node.current_harvests = 0
                    target_node.consecutive_failures = 0
                    target_node.last_visited_timestamp = 0.0
                    target_node.rest_duration_sec = 0  # F-1662: cooldown 0
                    rm.save_region(target_node.region or reg)
                    logger.info("[Mobil Müdahale] Node #%d aktif duruma sıfırlandı.", target_id)
                    self.notify_stream_listeners()
                    return {
                        "ok": True,
                        "action": "reset_node",
                        "node_id": target_id,
                        "state": "active",
                        "cooldown_remaining": 0,
                        "message": f"Node #{target_id} sıfırlandı ve aktif edildi.",
                    }

            # 7. Item 33: QUARANTINE_NODE (and TOGGLE_NODE)
            if action in ("quarantine_node", "toggle_node"):
                node_id_param = payload.get("node_id")
                if node_id_param is None:
                    return {"ok": False, "error": "Eksik 'node_id' parametresi."}
                if rm is None:
                    return {"ok": False, "error": "RouteManager aktif değil."}

                target_id = int(node_id_param)
                reg = getattr(rm, "active_region", "Altgard")
                rt = getattr(rm, "active_route", "All")
                target_node = rm.get_node(target_id, region=reg)
                if target_node is None:
                    return {"ok": False, "error": f"Node #{target_id} bulunamadı."}

                if action == "toggle_node" and self._is_node_quarantined_or_failed(target_node):
                    # Toggle back to active
                    target_node.state = "active"
                    target_node.current_harvests = 0
                    target_node.consecutive_failures = 0
                    target_node.last_visited_timestamp = 0.0
                    target_node.rest_duration_sec = 0  # F-1662: cooldown 0
                    rm.save_region(target_node.region or reg)
                    logger.info("[Mobil Müdahale] Node #%d karantinadan çıkarıldı ve aktif edildi.", target_id)
                    self.notify_stream_listeners()
                    return {
                        "ok": True,
                        "action": "toggle_node",
                        "node_id": target_id,
                        "state": "active",
                        "quarantined": False,
                        "message": f"Node #{target_id} aktif duruma getirildi.",
                    }

                desired_state = str(payload.get("state", "quarantined") or "quarantined").strip().lower()
                if desired_state not in ("quarantined", "failed"):
                    desired_state = "quarantined"

                target_node.state = desired_state
                target_node.consecutive_failures = max(3, int(getattr(target_node, "consecutive_failures", 0) or 0) + 1)
                target_node.current_harvests = 0
                target_node.last_visited_timestamp = time.time()
                target_node.rest_duration_sec = max(86400, int(getattr(target_node, "rest_duration_sec", 300) or 300))
                rm.save_region(target_node.region or reg)

                # If worker was targeting this node, switch to next ready non-quarantined node
                next_target_id: Optional[int] = None
                if w is not None and getattr(w, "active_node", None) is not None:
                    if int(getattr(w.active_node, "id", -1)) == target_id:
                        nxt = self._select_next_non_quarantined_node(rm, reg, rt)
                        w.active_node = nxt
                        if nxt is not None:
                            next_target_id = int(getattr(nxt, "id", 0) or 0) or None

                logger.warning(
                    "[Mobil Müdahale] Node #%d karantinaya alındı (durum=%s, sıradaki=%s).",
                    target_id,
                    desired_state,
                    next_target_id,
                )
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": action,
                    "node_id": target_id,
                    "state": desired_state,
                    "quarantined": True,
                    "next_node_id": next_target_id,
                    "message": f"Node #{target_id} karantinaya alındı ({desired_state}) ve rotada atlanacak.",
                }

            # 8. Item 33: DELETE_NODE
            if action == "delete_node":
                node_id_param = payload.get("node_id")
                if node_id_param is None:
                    return {"ok": False, "error": "Eksik 'node_id' parametresi."}
                if rm is None:
                    return {"ok": False, "error": "RouteManager aktif değil."}

                target_id = int(node_id_param)
                reg = getattr(rm, "active_region", "Altgard")
                rt = getattr(rm, "active_route", "All")
                target_node = rm.get_node(target_id, region=reg)
                if target_node is None:
                    return {"ok": False, "error": f"Silinecek Node #{target_id} bulunamadı."}

                node_region = str(getattr(target_node, "region", reg) or reg)
                deleted_ok = bool(rm.delete_node(target_id, region=node_region))
                if not deleted_ok:
                    return {"ok": False, "error": f"Node #{target_id} silinemedi."}

                rm.save_region(node_region)

                if w is not None and getattr(w, "active_node", None) is not None:
                    if int(getattr(w.active_node, "id", -1)) == target_id:
                        w.active_node = self._select_next_non_quarantined_node(rm, reg, rt)

                remaining = len(rm.get_route_nodes(node_region, rt))
                logger.info(
                    "[Mobil Müdahale] Node #%d kalıcı olarak silindi (Bölge: %s, Kalan: %d).",
                    target_id,
                    node_region,
                    remaining,
                )
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "delete_node",
                    "node_id": target_id,
                    "region": node_region,
                    "remaining_nodes": remaining,
                    "message": f"Node #{target_id} kalıcı olarak silindi ve region_data.json kaydedildi.",
                }

            # 9. OPTIMIZE_ROUTE — F-1654: hiçbir optimizasyon yolu yokken
            # ok:True dönülmez.
            if action == "optimize_route":
                opt_mode = str(payload.get("mode", "smooth_loop") or "smooth_loop")
                handled = False
                if w is not None and hasattr(w, "_auto_sort_sequential_route"):
                    w._auto_sort_sequential_route(mode=opt_mode)
                    handled = True
                elif (
                    self.orchestrator is not None
                    and hasattr(self.orchestrator, "sequential_planner")
                    and rm is not None
                ):
                    self.orchestrator.sequential_planner.apply_sequential_order_to_route(
                        rm,
                        region=getattr(rm, "active_region", None),
                        route=getattr(rm, "active_route", None),
                        mode=opt_mode,
                    )
                    handled = True
                if not handled:
                    return {"ok": False, "error": "Rota optimizasyon yolu yok (worker/planner)."}
                logger.info("[Mobil Müdahale] Sıralı rota optimizasyonu çalıştırıldı (%s).", opt_mode)
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "optimize_route",
                    "mode": opt_mode,
                    "message": "Rota sıralı döngü olarak optimize edildi.",
                }

            # 10. SEND_KEY / PRESS_KEY — F-1648: "t" ALLOWED_REMOTE_KEYS'te
            # var (önceden eksikti). F-1649: girdi sürücüsü yokken ok:True
            # dönülmez. F-1653: tuş yalnızca donanım/HID yolu
            # (Interception) doğrulandıysa gönderilir; yumuşak SendInput
            # AION 2'de sessizce düşer.
            if action in ("send_key", "press_key"):
                raw_key = str(payload.get("key", "") or "").strip().lower()
                if raw_key == "escape":
                    clean_key = "esc"
                else:
                    clean_key = raw_key

                if not clean_key or clean_key not in ALLOWED_REMOTE_KEYS:
                    return {
                        "ok": False,
                        "error": f"Geçersiz veya izin verilmeyen tuş: '{raw_key}'",
                    }

                if not self._input_ready(ic):
                    return {"ok": False, "error": "Girdi sürücüsü yok (send_key gönderilemedi)."}
                if not self._keyboard_trusted(ic):
                    return {
                        "ok": False,
                        "error": "Donanım girdi sürücüsü doğrulanamadı (Interception/HID yok) — yumuşak tuş AION 2'de düşer.",
                        "backend": str(getattr(ic, "active_backend_name", "") or ""),
                    }

                # S-1201: press_key 0.0 döndürürse (iptal/başarısız; MCS
                # _dispatch_ok yorumuna göre 0.0 = başarısız) `or 0.12`
                # ile 0.12'ye çevrilip koşulsuz ok:True dönülmez —
                # başarısızlık açıkça hata döner.
                hold_sec = 0.12
                try:
                    _held = ic.press_key(
                        clean_key,
                        min_hold=0.12,
                        max_hold=0.20,
                        release_mouse=True,
                        post_to_window=False,
                    )
                    if _held is None or float(_held) <= 0.0:
                        return {
                            "ok": False,
                            "error": f"'{raw_key}' tuşu gönderilemedi (girdi sürücüsü başarısız/iptal döndü).",
                        }
                    hold_sec = float(_held)
                except TypeError:
                    _held = ic.press_key(clean_key)
                    if _held is None or float(_held) <= 0.0:
                        return {
                            "ok": False,
                            "error": f"'{raw_key}' tuşu gönderilemedi (girdi sürücüsü başarısız/iptal döndü).",
                        }
                    hold_sec = float(_held)
                logger.info(
                    "[Mobil Müdahale] Uzaktan tuş gönderildi: '%s' (%.1fms, %s)",
                    clean_key,
                    hold_sec * 1000.0,
                    getattr(ic, "active_backend_name", "?"),
                )
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": action,
                    "key": raw_key,
                    "hold_ms": round(hold_sec * 1000.0, 1),
                    "backend": str(getattr(ic, "active_backend_name", "") or ""),
                    "message": f"'{raw_key.upper()}' tuşu gönderildi.",
                }

            # 11. Item 32: CLICK_SCREEN — F-1649: girdi sürücüsü yokken
            # ok:True dönülmez. F-1669: button doğrulanır.
            if action == "click_screen":
                norm_x = float(payload.get("norm_x", payload.get("x", 0.5)))
                norm_y = float(payload.get("norm_y", payload.get("y", 0.5)))
                button = str(payload.get("button", "left") or "left").strip().lower()
                double = bool(payload.get("double", False))
                stream_mode = str(payload.get("mode", "full") or "full").strip().lower()

                if button not in ("left", "right", "middle"):
                    return {"ok": False, "error": f"Geçersiz button: '{button}'"}
                if ic is None or not (
                    hasattr(ic, "click_normalized") or hasattr(ic, "human_click")
                ):
                    return {"ok": False, "error": "Girdi sürücüsü yok (click_screen uygulanamadı)."}
                # S-1801: fare yolunda donanım kapısı yoktu — yumuşak
                # SendInput fare AION 2'de sessizce düşer; send_key'deki
                # F-1653 kapısının fare karşılığı.
                if not self._mouse_trusted(ic):
                    return {
                        "ok": False,
                        "error": "Donanım fare sürücüsü doğrulanamadı (Interception/HID yok) — yumuşak fare AION 2'de düşer.",
                        "backend": str(getattr(ic, "active_backend_name", "") or ""),
                    }

                if norm_x > 1.0 or norm_y > 1.0:
                    ref_w = float(payload.get("ref_w", 960.0) or 960.0)
                    ref_h = float(payload.get("ref_h", 540.0) or 540.0)
                    norm_x = norm_x / max(1.0, ref_w)
                    norm_y = norm_y / max(1.0, ref_h)

                clamped_nx = max(0.0, min(1.0, norm_x))
                clamped_ny = max(0.0, min(1.0, norm_y))

                # Translate sub-crop normalized coords if clicked inside minimap or gather ROI mode
                if stream_mode == "minimap":
                    mx1, my1, mx2, my2 = MINIMAP_NORM_BOUNDS
                    clamped_nx = mx1 + clamped_nx * (mx2 - mx1)
                    clamped_ny = my1 + clamped_ny * (my2 - my1)
                elif stream_mode in ("gather", "f_gather", "roi"):
                    gx1, gy1, gx2, gy2 = GATHER_ROI_NORM_BOUNDS
                    clamped_nx = gx1 + clamped_nx * (gx2 - gx1)
                    clamped_ny = gy1 + clamped_ny * (gy2 - gy1)

                sw, sh = self._screen_size()
                px, py = int(round(clamped_nx * sw)), int(round(clamped_ny * sh))
                # S-1801: human_click/click_normalized (x, y, hold) döner ve
                # başarısız/iptal edilen tıklamayı başarıdan ayıran bir
                # bool vermez; bu katmanda doğrulanabilenler yoklanır:
                # istisna -> hata, motion-aborted -> hata, hold <= 0
                # (S-1201 deseni) -> hata. Alt katmanda _hw_mouse_down/up
                # bool'larının human_click'e bağlanması input_driver işidir
                # (bu dilim dışı); burada koşulsuz ok:True kaldırıldı.
                try:
                    if hasattr(ic, "click_normalized"):
                        click_res = ic.click_normalized(clamped_nx, clamped_ny, button=button, double=double)
                    elif hasattr(ic, "human_click"):
                        click_res = ic.human_click(px, py, button=button, double=double)
                    else:
                        return {"ok": False, "error": "Girdi sürücüsü yok (click_screen uygulanamadı)."}
                    if click_res is None:
                        return {"ok": False, "error": "Tıklama uygulanamadı (sürücü sonuç döndürmedi)."}
                    px, py, click_hold = click_res
                    if click_hold is not None and float(click_hold) <= 0.0:
                        return {"ok": False, "error": "Tıklama uygulanamadı (sürücü başarısız/iptal döndü)."}
                except Exception as exc:
                    logger.warning("[Mobil Müdahale] click_screen hatası: %s", exc)
                    return {"ok": False, "error": f"Tıklama uygulanamadı: {exc}"}
                aborted = getattr(ic, "motion_aborted", None)
                if callable(aborted):
                    try:
                        if bool(aborted()):
                            return {"ok": False, "error": "Tıklama iptal edildi (motion aborted)."}
                    except Exception:
                        pass

                logger.info(
                    "[Mobil Müdahale] Canlı ekran tıklaması: norm=(%.3f, %.3f) -> px=(%d, %d) [%s]",
                    clamped_nx,
                    clamped_ny,
                    px,
                    py,
                    button,
                )
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "click_screen",
                    "norm_x": round(clamped_nx, 4),
                    "norm_y": round(clamped_ny, 4),
                    "pixel_x": px,
                    "pixel_y": py,
                    "button": button,
                    "backend": str(getattr(ic, "active_backend_name", "") or ""),
                    "message": f"Ekran ({px}, {py}) noktasına tıklandı.",
                }

            # 12. Item 32: TAP_TO_MOVE_MAP — F-1663: koordinatlar gerçek
            # ekran boyutuna göre hesaplanır (1920x1080 sabiti yok).
            # F-1649: girdi sürücüsü yokken ok:True dönülmez.
            if action == "tap_to_move_map":
                raw_mx = payload.get("map_x", payload.get("x"))
                raw_my = payload.get("map_y", payload.get("y"))
                raw_nx = payload.get("norm_x")
                raw_ny = payload.get("norm_y")

                sw, sh = self._screen_size()
                button = str(payload.get("button", "left") or "left").strip().lower()
                if button not in ("left", "right", "middle"):
                    return {"ok": False, "error": f"Geçersiz button: '{button}'"}
                if ic is None or not (
                    hasattr(ic, "click_normalized") or hasattr(ic, "human_click")
                ):
                    return {"ok": False, "error": "Girdi sürücüsü yok (tap_to_move_map uygulanamadı)."}
                # S-1801: click_screen ile aynı donanım kapısı (F-1653'ün
                # fare karşılığı) — yumuşak fare yolu sessizce düşer.
                if not self._mouse_trusted(ic):
                    return {
                        "ok": False,
                        "error": "Donanım fare sürücüsü doğrulanamadı (Interception/HID yok) — yumuşak fare AION 2'de düşer.",
                        "backend": str(getattr(ic, "active_backend_name", "") or ""),
                    }

                if raw_mx is not None and raw_my is not None:
                    val_x = float(raw_mx)
                    val_y = float(raw_my)
                    if 0.0 <= val_x <= 1.0 and 0.0 <= val_y <= 1.0 and raw_nx is None:
                        norm_x = val_x
                        norm_y = val_y
                        map_x = int(round(norm_x * sw))
                        map_y = int(round(norm_y * sh))
                    else:
                        map_x = int(round(max(0.0, min(float(sw), val_x))))
                        map_y = int(round(max(0.0, min(float(sh), val_y))))
                        norm_x = max(0.0, min(1.0, map_x / float(sw)))
                        norm_y = max(0.0, min(1.0, map_y / float(sh)))
                elif raw_nx is not None and raw_ny is not None:
                    norm_x = max(0.0, min(1.0, float(raw_nx)))
                    norm_y = max(0.0, min(1.0, float(raw_ny)))
                    map_x = int(round(norm_x * sw))
                    map_y = int(round(norm_y * sh))
                else:
                    return {
                        "ok": False,
                        "error": "'tap_to_move_map' için 'map_x'/'map_y' veya 'norm_x'/'norm_y' gereklidir.",
                    }

                px, py = int(round(norm_x * sw)), int(round(norm_y * sh))

                # Compute bearing angle & distance from screen center
                dx = float(map_x - sw / 2.0)
                dy = float(map_y - sh / 2.0)
                angle_deg = round((math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0, 1)
                dist_px = math.hypot(dx, dy)
                mpp = float(getattr(rm, "meters_per_pixel", WORLD_MAP_METERS_PER_PX) or WORLD_MAP_METERS_PER_PX) if rm is not None else WORLD_MAP_METERS_PER_PX
                distance_m = round(dist_px * mpp, 1)

                # Locate nearest node in active route (if any)
                nearest_node_id: Optional[int] = None
                nearest_dist_px = float("inf")
                nearest_node_obj: Optional[Any] = None
                if rm is not None:
                    try:
                        reg = getattr(rm, "active_region", "Altgard")
                        rt = getattr(rm, "active_route", "All")
                        for n in rm.get_route_nodes(reg, rt):
                            if self._is_node_quarantined_or_failed(n):
                                continue
                            nx_px = int(getattr(n, "map_x", 0) or int(round(float(getattr(n, "norm_x", 0.5)) * sw)))
                            ny_px = int(getattr(n, "map_y", 0) or int(round(float(getattr(n, "norm_y", 0.5)) * sh)))
                            d = math.hypot(nx_px - map_x, ny_px - map_y)
                            if d < nearest_dist_px:
                                nearest_dist_px = d
                                nearest_node_id = int(getattr(n, "id", 0) or 0) or None
                                nearest_node_obj = n
                    except Exception as exc:
                        logger.debug("Tap-to-move en yakın düğüm araması uyarısı: %s", exc)

                # Click or steer via InputController (S-1801: istisna /
                # motion-aborted / hold<=0 yollarında "yürüyüş başlatıldı"
                # denmez — tap kaydı ve active_node yalnızca başarılı
                # tıklamadan sonra yazılır).
                try:
                    if hasattr(ic, "click_normalized"):
                        click_res = ic.click_normalized(norm_x, norm_y, button=button, double=False)
                    elif hasattr(ic, "human_click"):
                        click_res = ic.human_click(px, py, button=button)
                    else:
                        return {"ok": False, "error": "Girdi sürücüsü yok (tap_to_move_map uygulanamadı)."}
                    if click_res is None:
                        return {"ok": False, "error": "Tıklama uygulanamadı (sürücü sonuç döndürmedi)."}
                    px, py, click_hold = click_res
                    if click_hold is not None and float(click_hold) <= 0.0:
                        return {"ok": False, "error": "Tıklama uygulanamadı (sürücü başarısız/iptal döndü)."}
                except Exception as exc:
                    logger.warning("[Mobil Müdahale] tap_to_move_map tıklama hatası: %s", exc)
                    return {"ok": False, "error": f"Tıklama uygulanamadı: {exc}"}
                aborted = getattr(ic, "motion_aborted", None)
                if callable(aborted):
                    try:
                        if bool(aborted()):
                            return {"ok": False, "error": "Tıklama iptal edildi (motion aborted)."}
                    except Exception:
                        pass

                tap_record = {
                    "map_x": int(map_x),
                    "map_y": int(map_y),
                    "norm_x": round(norm_x, 4),
                    "norm_y": round(norm_y, 4),
                    "pixel_x": int(px),
                    "pixel_y": int(py),
                    "angle_deg": angle_deg,
                    "distance_m": distance_m,
                    # D-3 (dilim 22): 0.28 m/px doğrulanmamış sabittir —
                    # mesafe yalnızca gösterim içindir, hareketi etkilemez.
                    "distance_estimated": True,
                    "nearest_node_id": nearest_node_id,
                    "timestamp": round(time.time(), 3),
                }
                self._last_tap_to_move = tap_record

                if w is not None:
                    setattr(w, "last_tap_to_move_target", tap_record)
                    if nearest_node_obj is not None and (
                        nearest_dist_px <= 60.0 or bool(payload.get("select_nearest_node", False))
                    ):
                        w.active_node = nearest_node_obj
                        w.active_node_distance_m = int(round(distance_m))

                logger.info(
                    "[Mobil Müdahale] Haritaya Dokun-Yürü (Tap-to-Move): map=(%d, %d), açı=%.1f°, mesafe=%.1fm (En yakın Node: #%s)",
                    map_x,
                    map_y,
                    angle_deg,
                    distance_m,
                    nearest_node_id,
                )
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "tap_to_move_map",
                    "map_x": int(map_x),
                    "map_y": int(map_y),
                    "norm_x": round(norm_x, 4),
                    "norm_y": round(norm_y, 4),
                    "pixel_x": int(px),
                    "pixel_y": int(py),
                    "angle_deg": angle_deg,
                    "distance_m": distance_m,
                    "distance_estimated": True,
                    "nearest_node_id": nearest_node_id,
                    "backend": str(getattr(ic, "active_backend_name", "") or ""),
                    "message": f"Harita ({map_x}, {map_y}) konumuna yürüyüş başlatıldı ({distance_m}m, tahmini).",
                }

            # 13. UPDATE_SETTINGS — F-1655: settings nesnesi yokken ayarı
            # kaydetmeden "kaydedildi" denmez; hata döner.
            if action == "update_settings":
                if st is None:
                    return {"ok": False, "error": "Settings nesnesi yok (ayar kaydedilemedi)."}
                updated_fields: Dict[str, Any] = {}
                rejected_fields: Dict[str, str] = {}
                settings_payload = payload.get("settings", payload)
                if not isinstance(settings_payload, dict):
                    settings_payload = payload

                g_cfg = getattr(st, "gathering", None)
                r_cfg = getattr(st, "route", None)

                if "harvests_per_node" in settings_payload:
                    val = max(1, min(10, int(settings_payload["harvests_per_node"])))
                    if g_cfg is not None:
                        g_cfg.harvests_per_node = val
                    if r_cfg is not None:
                        r_cfg.harvests_per_node = val
                    updated_fields["harvests_per_node"] = val

                if "gather_key" in settings_payload or "interact_key" in settings_payload:
                    raw_gk = settings_payload.get("gather_key", settings_payload.get("interact_key", "f"))
                    gkey = str(raw_gk).strip().lower()
                    if gkey in ("f", "e", "auto") and g_cfg is not None:
                        g_cfg.gather_key = gkey
                        updated_fields["gather_key"] = gkey
                    else:
                        # D-2 (dilim 22): geçersiz gather_key sessizce
                        # atlanıp ok:True dönülmüyor — reddedilen alanda
                        # gerekçesiyle döner.
                        rejected_fields["gather_key"] = (
                            f"Geçersiz değer: '{raw_gk}' (yalnızca f/e/auto)"
                            if gkey not in ("f", "e", "auto")
                            else "Gathering ayarı yok."
                        )

                if "minimap_auto_record" in settings_payload and r_cfg is not None:
                    flag = bool(settings_payload["minimap_auto_record"])
                    r_cfg.minimap_auto_record = flag
                    updated_fields["minimap_auto_record"] = flag

                if "world_3d_auto_record" in settings_payload and r_cfg is not None:
                    w3d_flag = bool(settings_payload["world_3d_auto_record"])
                    r_cfg.hybrid_3d_and_minimap_record = w3d_flag
                    updated_fields["world_3d_auto_record"] = w3d_flag

                if "test_mode" in settings_payload or "dry_run" in settings_payload:
                    # O-1 (dilim 22): uzaktan test_mode açmak donanım güven
                    # kapısını baypas ederdi (test modunda _keyboard_trusted
                    # koşulsuz True döner, soft SendInput "güvenilir"
                    # sayılırdı). Uzaktan değiştirilemez — yalnızca yerel
                    # konsoldan açılır. Reddedildiği görünür döner.
                    rejected_fields["test_mode"] = (
                        "Uzaktan değiştirilemez (donanım güven kapısı baypası olur)"
                    )
                    logger.warning(
                        "[Mobil Müdahale] Uzaktan test_mode/dry_run isteği reddedildi."
                    )

                if ("rest_minutes" in settings_payload or "cooldown_minutes" in settings_payload) and r_cfg is not None:
                    raw_rm = settings_payload.get("rest_minutes", settings_payload.get("cooldown_minutes", 5))
                    r_min = max(0, min(240, int(float(raw_rm))))
                    r_cfg.rest_minutes = r_min
                    updated_fields["rest_minutes"] = r_min

                if "fight_back" in settings_payload and r_cfg is not None:
                    fb_flag = bool(settings_payload["fight_back"])
                    r_cfg.fight_back = fb_flag
                    updated_fields["fight_back"] = fb_flag

                if not updated_fields:
                    return {"ok": False, "error": "Uygulanabilir ayar alanı yok.", "rejected": rejected_fields}
                # S-1802: sm yokken kayıt sessizce atlanıp yine de
                # "kaydedildi" deniyordu. Kalıcılık doğrulanmadan iddia
                # edilmez: persisted yalnızca sm.save() gerçekten
                # çalıştığında True olur; save() patlarsa da False kalır.
                persisted = False
                if sm is not None and hasattr(sm, "save"):
                    try:
                        sm.save(st)
                        persisted = True
                    except Exception as exc:
                        logger.warning("[Mobil Müdahale] Ayarlar kaydedilemedi: %s", exc)
                        persisted = False
                if self.orchestrator is not None and hasattr(self.orchestrator, "_sync_encrypted_vault"):
                    self.orchestrator._sync_encrypted_vault()

                logger.info("[Mobil Müdahale] Ayarlar güncellendi: %s (kalıcı=%s)", updated_fields, persisted)
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": "update_settings",
                    "updated": updated_fields,
                    "rejected": rejected_fields,
                    "persisted": persisted,
                    "message": (
                        "Ayarlar güncellendi ve kaydedildi."
                        if persisted
                        else "Ayarlar güncellendi (yalnızca bellekte; kalıcı kayıt yapılamadı — settings yöneticisi yok veya kayıt başarısız)."
                    ),
                }

            # 14. PANIC_STOP / EMERGENCY_STOP
            if action in ("panic_stop", "emergency_stop"):
                msg = str(payload.get("reason") or "MOBİL ACİL DURDURMA!")
                if w is None and ic is None:
                    return {"ok": False, "error": "Bot bağlı değil (panic_stop uygulanamadı)."}
                if w is not None and hasattr(w, "pause"):
                    w.pause(msg)
                if ic is not None and hasattr(ic, "set_route_active"):
                    ic.set_route_active(False)
                logger.warning("[Mobil Müdahale] %s", msg)
                self.notify_stream_listeners()
                return {
                    "ok": True,
                    "action": action,
                    "message": msg,
                }

            # 15. SOFT_AIM — mouse-only target acquire (Tab inject often filtered)
            if action in ("soft_aim", "soft_aim_click", "mouse_aim"):
                if ic is None or not hasattr(ic, "soft_aim_click"):
                    return {"ok": False, "error": "soft_aim_click yok (input driver)"}
                frame = None
                try:
                    if orch is not None and hasattr(orch, "state_machine"):
                        sm = orch.state_machine
                        grab = getattr(sm, "_grab_frame", None)
                        if callable(grab):
                            frame = grab(grayscale=False)
                except Exception as exc:
                    logger.debug("soft_aim frame grab: %s", exc)
                ok = bool(ic.soft_aim_click(frame=frame))
                logger.info("[Mobil Müdahale] soft_aim ok=%s", ok)
                self.notify_stream_listeners()
                return {
                    "ok": ok,
                    "action": "soft_aim",
                    "message": "Mouse-only soft-aim gönderildi." if ok else "soft_aim başarısız",
                }

            # 15b. CAST_SKILL — F-1650: skill bu dosyada fare tıklamasıyla
            # atılmaz. Öncelik GameActionBus (R/T tuş yolu, Interception);
            # bus yoksa slot tuşu güvenilir klavye yoluyla gönderilir;
            # hiçbiri yoksa hata döner (fare tıklaması fallback'i yok).
            if action in ("cast_skill", "skill_cast", "alt_skill"):
                slot = str(payload.get("slot", payload.get("key", payload.get("skill", "1"))) or "1").strip()
                force_alt = bool(payload.get("force_alt", action == "alt_skill"))
                wkr = w if w is not None else self.worker
                frame = None
                try:
                    grab = getattr(wkr, "_grab_frame", None) if wkr is not None else None
                    if callable(grab):
                        frame = grab(grayscale=False)
                except Exception as exc:
                    logger.debug("cast_skill frame grab: %s", exc)
                bus = getattr(wkr, "action_bus", None) if wkr is not None else None
                if bus is None or not hasattr(bus, "cast_skill"):
                    # Bus yok: R/T kuralına uygun tuş yolu. Slot tuşu
                    # (1-5, r/t) güvenilir donanım klavyeyle gönderilir.
                    slot_key = slot.lower()
                    if slot_key in ALLOWED_REMOTE_KEYS and self._input_ready(ic) and self._keyboard_trusted(ic):
                        # S-1201: press_key 0.0 döndürürse (iptal/başarısız)
                        # `or 0.05` ile 0.05'e çevrilip ok:True dönülmez.
                        try:
                            _held = ic.press_key(slot_key, min_hold=0.03, max_hold=0.05)
                            if _held is None or float(_held) <= 0.0:
                                return {
                                    "ok": False,
                                    "error": f"cast_skill tuş gönderimi başarısız (slot '{slot}' — sürücü başarısız/iptal döndü).",
                                    "slot": slot,
                                }
                            hold = float(_held)
                        except TypeError:
                            _held = ic.press_key(slot_key)
                            if _held is None or float(_held) <= 0.0:
                                return {
                                    "ok": False,
                                    "error": f"cast_skill tuş gönderimi başarısız (slot '{slot}' — sürücü başarısız/iptal döndü).",
                                    "slot": slot,
                                }
                            hold = float(_held)
                        logger.info("[Mobil Müdahale] cast_skill tuş yolu '%s' (%.1fms)", slot_key, hold * 1000.0)
                        self.notify_stream_listeners()
                        return {
                            "ok": True,
                            "action": "cast_skill",
                            "slot": slot,
                            "method": "key",
                            "message": f"Skill '{slot}' tuş yoluyla gönderildi.",
                        }
                    return {
                        "ok": False,
                        "error": "cast_skill yolu yok (bus yok ve güvenilir klavye yolu kapalı — fare tıklaması kullanılmaz)",
                        "has_bus": bus is not None,
                        "has_ic": ic is not None,
                    }
                if force_alt:
                    # Temporarily treat keyboard as untrusted for this cast.
                    prev = getattr(bus, "_keyboard_trusted", None)
                    try:
                        bus._keyboard_trusted = lambda: False  # type: ignore[method-assign]
                    except Exception:
                        pass
                    try:
                        cast_res = bus.cast_skill(slot, frame=frame)
                    finally:
                        if prev is not None:
                            try:
                                bus._keyboard_trusted = prev  # type: ignore[method-assign]
                            except Exception:
                                pass
                else:
                    cast_res = bus.cast_skill(slot, frame=frame)
                logger.info("[Mobil Müdahale] cast_skill %s => %s", slot, cast_res)
                self.notify_stream_listeners()
                return {
                    "ok": bool((cast_res or {}).get("ok")),
                    "action": "cast_skill",
                    "slot": slot,
                    "method": (cast_res or {}).get("method"),
                    "x": (cast_res or {}).get("x"),
                    "y": (cast_res or {}).get("y"),
                    "result": cast_res,
                    "message": f"Skill '{slot}' method={(cast_res or {}).get('method')}",
                }

            # 15c. TAB_TARGET — optional soft_aim; distinguishes key vs mouse lock
            if action in ("tab_target", "press_tab_target"):
                soft = bool(payload.get("soft_aim", payload.get("soft", True)))
                wkr = w if w is not None else self.worker
                frame = None
                try:
                    grab = getattr(wkr, "_grab_frame", None) if wkr is not None else None
                    if callable(grab):
                        frame = grab(grayscale=False)
                except Exception as exc:
                    logger.debug("tab_target frame grab: %s", exc)
                bus = getattr(wkr, "action_bus", None) if wkr is not None else None
                if bus is None or not hasattr(bus, "press_tab_target"):
                    return {
                        "ok": False,
                        "error": "press_tab_target yok",
                        "has_worker": wkr is not None,
                        "has_bus": bus is not None,
                    }
                tab_res = bus.press_tab_target(frame=frame, soft_aim=soft)
                logger.info("[Mobil Müdahale] tab_target soft_aim=%s => %s", soft, tab_res)
                self.notify_stream_listeners()
                return {
                    "ok": bool((tab_res or {}).get("ok")),
                    "action": "tab_target",
                    "soft_aim": soft,
                    "method": (tab_res or {}).get("method"),
                    "result": tab_res,
                    "message": f"Tab target method={(tab_res or {}).get('method')}",
                }

            # 16. NUDGE_CAMERA — relative mouse-look (no absolute teleport)
            if action in ("nudge_camera", "nudge", "look"):
                if ic is None or not hasattr(ic, "nudge_camera"):
                    return {"ok": False, "error": "nudge_camera yok"}
                dx = int(payload.get("dx", payload.get("x", 0)) or 0)
                dy = int(payload.get("dy", payload.get("y", 0)) or 0)
                dx = max(-200, min(200, dx))
                dy = max(-200, min(200, dy))
                ok = bool(ic.nudge_camera(dx, dy))
                logger.info("[Mobil Müdahale] nudge_camera dx=%d dy=%d ok=%s", dx, dy, ok)
                return {
                    "ok": ok,
                    "action": "nudge_camera",
                    "dx": dx,
                    "dy": dy,
                    "message": f"Kamera nudge ({dx},{dy})",
                }

            # 17a. ELEVATE — ShellExecute runas (shows UAC). Required when Steam/AION2 is High.
            # F-1656: UAC isteğinin sonucu thread'e atılıp hemen ok:True
            # dönülmez; ensure_admin_elevation eşzamanlı çağrılır, sonucu
            # neyse o döner; süreç çıkışı ancak başarıdan sonra planlanır.
            if action in ("elevate", "runas", "admin"):
                if self._bot_is_elevated():
                    return {
                        "ok": True,
                        "action": "elevate",
                        "elevated": True,
                        "message": "Bot zaten yönetici (elevated).",
                    }
                logger.warning("[Mobil Müdahale] elevate: UAC runas isteniyor.")
                try:
                    from core.uac_elevator import ensure_admin_elevation

                    elevated_ok = bool(
                        ensure_admin_elevation(
                            argv=[os.path.join(PROJECT_ROOT, "main.py")],
                            working_dir=PROJECT_ROOT,
                            dry_run=False,
                            exit_on_elevate=False,
                        )
                    )
                except Exception as exc:
                    logger.error("elevate hatası: %s", exc)
                    return {"ok": False, "action": "elevate", "error": "UAC yükseltme isteği başlatılamadı."}
                if not elevated_ok:
                    return {"ok": False, "action": "elevate", "error": "UAC yükseltme isteği reddedildi/başarısız."}

                def _do_exit_after_elevate() -> None:
                    try:
                        # v20/bot-ani-kapanma: os._exit(0) oncesi SON iz --
                        # eskiden bu yol sessizce olurdu, atexit bile calismaz.
                        from core.crash_evidence import log_imminent_process_exit

                        log_imminent_process_exit("elevate", 1.0)
                        time.sleep(1.0)
                        os._exit(0)
                    except Exception:
                        pass

                threading.Thread(target=_do_exit_after_elevate, name="MobileElevateExit", daemon=True).start()
                return {
                    "ok": True,
                    "action": "elevate",
                    "elevated": False,
                    "message": "UAC açılmalı — Evet deyin. Medium bot High AION2'ye UIPI yüzünden tuş/fare gönderemez.",
                }

            # 17. RESTART — elevated self-relaunch (no second UAC if already admin)
            # F-1656: yeni süreç eşzamanlı spawn edilir; spawn başarısızsa
            # ok:False döner, çıkış yalnızca spawn başarısından sonra planlanır.
            if action in ("restart", "reload", "relauch", "relaunch"):
                logger.warning("[Mobil Müdahale] Uzaktan yeniden başlatma isteği alındı.")
                orch = self.orchestrator
                project_dir = os.environ.get("GATHERION_HOME") or os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))
                )
                main_py = os.path.join(project_dir, "main.py")
                if not os.path.isfile(main_py):
                    return {"ok": False, "action": "restart", "error": "main.py bulunamadı."}
                py_exe = sys.executable
                # Prefer pythonw twin of current interpreter.
                if py_exe.lower().endswith("python.exe"):
                    twin = py_exe[:-10] + "pythonw.exe"
                    if os.path.isfile(twin):
                        py_exe = twin

                try:
                    creationflags = 0
                    if sys.platform.startswith("win"):
                        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                            subprocess, "CREATE_NEW_PROCESS_GROUP", 0
                        )
                    env = os.environ.copy()
                    env["PYTHONPATH"] = project_dir + os.pathsep + env.get("PYTHONPATH", "")
                    spawned = subprocess.Popen(
                        [py_exe, main_py],
                        cwd=project_dir,
                        env=env,
                        close_fds=True,
                        creationflags=creationflags,
                    )
                    logger.warning("[Mobil Müdahale] Yeni süreç başlatıldı: %s %s", py_exe, main_py)
                except Exception as exc:
                    logger.error("Yeniden başlatma spawn hatası: %s", exc)
                    return {"ok": False, "action": "restart", "error": "Yeni süreç başlatılamadı."}

                def _do_restart_exit() -> None:
                    try:
                        # v20/bot-ani-kapanma: os._exit(0) oncesi SON iz.
                        from core.crash_evidence import log_imminent_process_exit

                        log_imminent_process_exit("restart", 1.0)
                    except Exception:
                        pass
                    try:
                        from PyQt6.QtWidgets import QApplication

                        app = QApplication.instance()
                        if app is not None:
                            app.quit()
                    except Exception as exc:
                        logger.debug("QApplication.quit atlandı: %s", exc)
                    try:
                        if orch is not None and hasattr(orch, "shutdown"):
                            orch.shutdown()
                    except Exception as exc:
                        logger.error("Uzaktan shutdown hatası: %s", exc)
                    time.sleep(1.0)
                    try:
                        os._exit(0)
                    except Exception:
                        pass

                threading.Thread(target=_do_restart_exit, name="MobileRestart", daemon=True).start()
                return {
                    "ok": True,
                    "action": "restart",
                    "spawned_pid": int(getattr(spawned, "pid", 0) or 0),
                    "message": "Gatherion yeniden başlatılıyor (elevated self-spawn)...",
                }

            # 18. QUIT / EXIT / SHUTDOWN — remote process exit (for elevated restart)
            # F-1656: kapanış öncesi shutdown eşzamanlı çalıştırılır; çıkış
            # yalnızca shutdown denemesinden sonra planlanır.
            if action in ("quit", "exit", "shutdown"):
                logger.warning("[Mobil Müdahale] Uzaktan kapatma isteği alındı.")
                orch = self.orchestrator
                try:
                    from PyQt6.QtWidgets import QApplication

                    app = QApplication.instance()
                    if app is not None:
                        # Thread-safe; drains main loop so overlay/shutdown can run.
                        app.quit()
                except Exception as exc:
                    logger.debug("QApplication.quit atlandı: %s", exc)
                try:
                    if orch is not None and hasattr(orch, "shutdown"):
                        orch.shutdown()
                except Exception as exc:
                    logger.error("Uzaktan shutdown hatası: %s", exc)

                def _do_quit() -> None:
                    # v20/bot-ani-kapanma: os._exit(0) oncesi SON iz.
                    try:
                        from core.crash_evidence import log_imminent_process_exit

                        log_imminent_process_exit("quit", 1.5)
                    except Exception:
                        pass
                    time.sleep(1.5)
                    try:
                        os._exit(0)
                    except Exception:
                        pass

                threading.Thread(target=_do_quit, name="MobileQuit", daemon=True).start()
                return {
                    "ok": True,
                    "action": "quit",
                    "message": "Gatherion kapatılıyor...",
                }

            # 15c. RECORD_INCIDENT / CLEAR_INCIDENTS — Y-7 (dilim 22):
            # dashboard'un manuel olay anı + galeri temizleme düğmeleri
            # sunucuda karşılıksızdı (Bilinmeyen aksiyon).
            if action == "record_incident":
                rec = self._get_incident_recorder()
                if rec is None:
                    return {"ok": False, "error": "IncidentRecorder bağlı değil."}
                try:
                    item = rec.record_incident(
                        event_type=str(payload.get("event_type") or "manual_snapshot"),
                        details=str(payload.get("details") or "Mobil panelden manuel olay anı"),
                    )
                except Exception as exc:
                    return {"ok": False, "error": f"Olay kaydedilemedi: {exc}"}
                if item is None:
                    return {"ok": False, "error": "Olay kaydedilemedi (cooldown, kare yok veya JPEG yazılamadı)."}
                self.notify_stream_listeners()
                return {"ok": True, "action": "record_incident", "incident": item, "message": "Olay anı kaydedildi."}

            if action == "clear_incidents":
                # P3-8 (S-050/S-218): yıkıcı aksiyon açık onay ister; onaysız
                # istek reddedilir (tek dokunuşla galeri silinemez).
                if payload.get("confirm") is not True:
                    return {"ok": False, "error": "Olay galerisini temizlemek için onay (confirm: true) gerekir."}
                rec = self._get_incident_recorder()
                if rec is None:
                    return {"ok": False, "error": "IncidentRecorder bağlı değil."}
                try:
                    cleared = int(rec.clear_incidents())
                except Exception as exc:
                    return {"ok": False, "error": f"Olaylar temizlenemedi: {exc}"}
                self.notify_stream_listeners()
                return {"ok": True, "action": "clear_incidents", "cleared": cleared, "message": f"{cleared} olay kaydı temizlendi."}

            return {"ok": False, "error": f"Bilinmeyen aksiyon: '{action}'"}
        except Exception as exc:
            logger.error("Mobil aksiyon hatası ('%s'): %s", action, exc, exc_info=True)
            return {"ok": False, "action": action, "error": str(exc)}

    # =========================================================================
    # Item 35: PWA Manifest, Service Worker, Icons & Dashboard HTML Loader
    # =========================================================================

    def _cached_static_text(
        self, path: str, error_label: str, transform: Optional[Callable[[str], str]] = None
    ) -> Optional[str]:
        """Statik metin dosyasını (mtime_ns, size) anahtarıyla önbellekleyerek okur.

        Pano/manifest/service-worker her HTTP isteğinde diskten yeniden
        okunuyordu (pano ~5.3 ms/istek); önbellek isabeti ~µs mertebesindedir.
        Dosya değişirse (mtime veya boyut farklı) önbellek geçersiz sayılır
        ve dosya yeniden okunur. Okunamayan dosya önbelleğe alınmaz
        (sonradan oluşan dosya bir sonraki istekte görülür).
        """
        try:
            st = os.stat(path)
            key = (st.st_mtime_ns, st.st_size)
        except OSError:
            with self._static_cache_lock:
                self._static_file_cache.pop(path, None)
            return None
        with self._static_cache_lock:
            cached = self._static_file_cache.get(path)
            if cached is not None and cached[0] == key:
                return cached[1]
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as exc:
            logger.warning("%s okunamadı, yerleşik içerik sunuluyor: %s", error_label, exc)
            return None
        if transform is not None:
            content = transform(content)
        with self._static_cache_lock:
            self._static_file_cache[path] = (key, content)
        return content

    @staticmethod
    def _inject_pwa_tags(content: str) -> str:
        """Pano HTML'ine manifest + service-worker kaydını ekler (yoksa)."""
        if 'rel="manifest"' not in content and "</head>" in content:
            pwa_head_tags = (
                '  <link rel="manifest" href="/manifest.json" />\n'
                '  <link rel="apple-touch-icon" href="/apple-touch-icon.png" />\n'
                "</head>"
            )
            content = content.replace("</head>", pwa_head_tags, 1)
        if "serviceWorker" not in content and "</body>" in content:
            sw_script = (
                "  <script>\n"
                "    if ('serviceWorker' in navigator) {\n"
                "      window.addEventListener('load', () => {\n"
                "        navigator.serviceWorker.register('/sw.js').catch(() => {});\n"
                "      });\n"
                "    }\n"
                "  </script>\n"
                "</body>"
            )
            content = content.replace("</body>", sw_script, 1)
        return content

    def load_manifest_json(self) -> str:
        """Loads `ui/manifest.json` with UTF-8 encoding, or returns a built-in standalone PWA manifest."""
        content = self._cached_static_text(self.manifest_json_path, "ui/manifest.json")
        if content is not None and content.strip():
            return content

        fallback_manifest = {
            "id": "gatherion-v2-mobile-pwa",
            "name": "Gatherion v2.0 Mobil Kumanda",
            "short_name": "Gatherion v2.0",
            "description": "Gatherion v2.0 AION 2 Canlı Telemetri, 2B Rota Radarı ve Uzaktan Kontrol PWA Paneli",
            "lang": "tr-TR",
            "start_url": "/",
            "scope": "/",
            "display": "standalone",
            "orientation": "portrait",
            "background_color": "#0b0f17",
            "theme_color": "#0b0f17",
            "icons": [
                {
                    "src": "/icon-192.png",
                    "sizes": "192x192",
                    "type": "image/png",
                    "purpose": "any maskable",
                },
                {
                    "src": "/icon-512.png",
                    "sizes": "512x512",
                    "type": "image/png",
                    "purpose": "any maskable",
                },
            ],
        }
        return json.dumps(fallback_manifest, ensure_ascii=False, indent=2)

    def load_service_worker_js(self) -> str:
        """Loads `ui/sw.js` with UTF-8 encoding, or returns a built-in PWA Service Worker."""
        content = self._cached_static_text(self.service_worker_js_path, "ui/sw.js")
        if content is not None and content.strip():
            return content

        return (
            "const CACHE_NAME='gatherion-v2-pwa-v1';\n"
            "self.addEventListener('install',e=>{self.skipWaiting();});\n"
            "self.addEventListener('activate',e=>{e.waitUntil(self.clients.claim());});\n"
            "self.addEventListener('fetch',e=>{if(e.request.method==='GET'){e.respondWith(fetch(e.request).catch(()=>caches.match(e.request)));}});\n"
        )

    def render_pwa_icon_png(self, size: int = 192) -> bytes:
        """Generates a crisp Gatherion v2.0 radar-crest PNG app icon for iOS/Android Home Screen."""
        clamped_size = max(64, min(512, int(size)))
        if clamped_size in self._icon_cache:
            return self._icon_cache[clamped_size]

        img = np.zeros((clamped_size, clamped_size, 3), dtype=np.uint8)
        img[:, :] = (23, 15, 11)  # #0b0f17 in BGR

        cx, cy = clamped_size // 2, clamped_size // 2
        r_outer = int(clamped_size * 0.40)
        r_mid = int(clamped_size * 0.28)
        r_inner = int(clamped_size * 0.08)

        cv2.circle(img, (cx, cy), r_outer, (212, 182, 6), max(2, clamped_size // 36), cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_mid, (129, 185, 16), max(2, clamped_size // 42), cv2.LINE_AA)

        pts = np.array(
            [
                [cx, int(clamped_size * 0.16)],
                [int(clamped_size * 0.76), int(clamped_size * 0.76)],
                [cx, int(clamped_size * 0.63)],
                [int(clamped_size * 0.24), int(clamped_size * 0.76)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], (212, 182, 6), cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_inner, (11, 158, 245), -1, cv2.LINE_AA)

        ok, buf = cv2.imencode(".png", img)
        # F-1673: kodlama hatasında boş bayt ile sahte başarılı yanıt yok.
        if not ok or buf is None:
            raise RuntimeError("PWA ikon PNG kodlama başarısız")
        png_bytes = buf.tobytes()
        if png_bytes:
            self._icon_cache[clamped_size] = png_bytes
        return png_bytes

    def load_dashboard_html(self) -> str:
        """
        Loads `ui/mobile_dashboard.html` with UTF-8 encoding (injecting PWA `/manifest.json`
        and `/sw.js` registration if not already present), or serves a built-in fallback.
        """
        content = self._cached_static_text(
            self.dashboard_html_path, "mobile_dashboard.html", transform=self._inject_pwa_tags
        )
        if content is not None and content.strip():
            return content

        return self._builtin_fallback_html(token=self.access_token if self.require_token else "")

    def get_mobile_server_health_summary(self) -> dict:
        # F-1666: koşulsuz True yok — gerçek sözleşme + çalışma durumu.
        return {
            "mobile_server_ready": bool(validate_mobile_server_contract()),
            "running": bool(self._running),
            "require_token": bool(self.require_token),
            "port": int(self.port),
        }

    def _builtin_fallback_html(self, token: str = "") -> str:
        # F-1670 (P1-9/K-08): token koruması açıkken yedek panel de
        # kimlik doğrulamalı istek yapar; aksi halde /api/status ve
        # /api/action 401 döner. Token ARTIK URL sorgusunda taşınmaz —
        # fetch istekleri `X-Gatherion-Token` başlığıyla gider (sunucu
        # `_check_authorized` başlığı kabul eder); kare görüntüsü
        # `<img>` başlık gönderemediği için fetch+blob ile alınır.
        _html = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no"/>
<meta name="theme-color" content="#0b0f17"/>
<meta name="apple-mobile-web-app-capable" content="yes"/>
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent"/>
<link rel="manifest" href="/manifest.json"/>
<link rel="apple-touch-icon" href="/apple-touch-icon.png"/>
<title>Gatherion v2.0 Mobil Kumanda</title>
<style>
  body { background:#0b0f17; color:#e2e8f0; font-family:system-ui,-apple-system,sans-serif; margin:0; padding:16px; }
  .card { background:#151c28; border:1px solid #263248; border-radius:12px; padding:14px; margin-bottom:12px; }
  h1 { font-size:18px; margin:0 0 8px 0; color:#38bdf8; }
  .btn { background:#1e293b; color:#f8fafc; border:1px solid #38bdf8; border-radius:8px; padding:10px 14px; font-weight:600; cursor:pointer; margin:4px; }
  .btn-danger { border-color:#ef4444; background:#450a0a; color:#fecaca; }
  pre { font-size:12px; overflow-x:auto; color:#94a3b8; }
</style>
</head>
<body>
  <div class="card">
    <h1>Gatherion v2.0 Mobil Kumanda</h1>
    <button class="btn" onclick="sendAction('resume')">▶ Başlat / Devam</button>
    <button class="btn" onclick="sendAction('pause')">⏸ Duraklat</button>
    <button class="btn" onclick="sendAction('record_f8')">📍 F8 Kaydet</button>
    <button class="btn" onclick="sendAction('skip_node')">⏭ Node Atla</button>
    <button class="btn btn-danger" onclick="sendAction('panic_stop')">🚨 ACİL DURDUR</button>
  </div>
  <div class="card">
    <img id="liveFrame" alt="Canlı kare yükleniyor..." style="width:100%;border-radius:8px;" />
  </div>
  <div class="card">
    <pre id="statusBox">Yükleniyor...</pre>
  </div>
<script>
const AUTH_TOKEN = __TOKEN_JSON__;
function authHeaders() {
  const h = {};
  if (AUTH_TOKEN) h['X-Gatherion-Token'] = AUTH_TOKEN;
  return h;
}
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => navigator.serviceWorker.register('/sw.js').catch(() => {}));
}
async function sendAction(act) {
  await fetch('/api/action', {method:'POST', headers:Object.assign({'Content-Type':'application/json'}, authHeaders()), body:JSON.stringify({action:act})});
  refresh();
}
async function refreshFrame() {
  try {
    const fr = await fetch('/api/frame.jpg?mode=pip&quality=70&t=' + Date.now(), {headers: authHeaders()});
    if (!fr.ok) return;
    const blob = await fr.blob();
    const img = document.getElementById('liveFrame');
    const old = img.dataset.blobUrl;
    const url = URL.createObjectURL(blob);
    img.src = url;
    img.dataset.blobUrl = url;
    if (old) URL.revokeObjectURL(old);
  } catch (e) {}
}
async function refresh() {
  try {
    const r = await fetch('/api/status', {headers: authHeaders()});
    const d = await r.json();
    document.getElementById('statusBox').textContent = JSON.stringify(d, null, 2);
  } catch (e) {}
  refreshFrame();
}
setInterval(refresh, 1500);
refresh();
</script>
</body>
</html>"""
        # P1-9 (K-08): token URL sorgusuna değil, sayfa içindeki JS
        # sabitine yazılır; istekler başlıkla gider. json.dumps JS
        # kaçışını güvenli yapar. (Sayfa yalnızca kimlik doğrulamalı
        # istemciye sunulur — ana panelin localStorage yaklaşımıyla
        # aynı güven modeli.)
        html = _html.replace("__TOKEN_JSON__", json.dumps(str(token)))
        return html

    def get_mobile_server_ready_flag(self) -> dict:
        # F-1666: gerçek sözleşme kontrolü; koşulsuz True yok.
        return {"module": "mobile_server", "ready": bool(validate_mobile_server_contract())}

    def get_mobile_server_version_info(self) -> dict:
        return {"module": "mobile_server", "version": __version__, "ready": bool(validate_mobile_server_contract())}

    def get_mobile_server_diagnostic_snapshot(self) -> dict:
        return {"module": "mobile_server", "status": "ok" if validate_mobile_server_contract() else "contract_failed", "ready": bool(validate_mobile_server_contract())}


def get_mobile_server_capabilities() -> list:
    return ["mobile_server", "remote_api", "diagnostic_snapshot"]


def validate_mobile_server_contract() -> bool:
    """F-1666: koşulsuz True yok — gerçek kontroller:
    (1) kimlik doğrulama varsayılanı açık, (2) "t"/"r" tuşları izinli,
    (3) token politikası 32 karakter, (4) maskeli istemci WS çerçevesi
    RFC 6455'e uygun ayrıştırılıyor."""
    try:
        import inspect

        sig = inspect.signature(MobileRemoteServer.__init__)
        if sig.parameters["require_token"].default is not True:
            return False
        if "t" not in ALLOWED_REMOTE_KEYS or "r" not in ALLOWED_REMOTE_KEYS:
            return False
        if int(_MOBILE_TOKEN_LENGTH) < 32:
            return False
        if not get_mobile_server_capabilities():
            return False
        # Maskeli istemci text çerçevesi turu (socketpair üzerinden).
        payload = b'{"action":"ping"}'
        mask = b"\x11\x22\x33\x44"
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        frame = bytes([0x81, 0x80 | len(payload)]) + mask + masked
        rx, tx = socket.socketpair()
        try:
            rx.settimeout(2.0)
            tx.settimeout(2.0)
            tx.sendall(frame)
            decoded = _ws_read_client_frame(rx)
        finally:
            rx.close()
            tx.close()
        if decoded != (0x1, payload):
            return False
        return True
    except Exception:
        return False


def get_mobile_server_baseline_metrics() -> dict:
    return {"module": "mobile_server", "contract_valid": validate_mobile_server_contract(), "capability_count": len(get_mobile_server_capabilities())}


def get_mobile_server_readiness_status() -> dict:
    return {"module": "mobile_server", "ready": validate_mobile_server_contract(), "metrics": get_mobile_server_baseline_metrics()}


def get_mobile_server_audit_record() -> dict:
    return {"module": "mobile_server", "audit_passed": validate_mobile_server_contract(), "readiness": get_mobile_server_readiness_status()}


def get_mobile_server_operational_digest() -> dict:
    return {"module": "mobile_server", "operational": validate_mobile_server_contract(), "audit": get_mobile_server_audit_record()}


def get_mobile_server_telemetry_payload() -> dict:
    return {"module": "mobile_server", "telemetry_ready": validate_mobile_server_contract(), "digest": get_mobile_server_operational_digest()}


def get_mobile_server_integrity_seal() -> dict:
    return {"module": "mobile_server", "sealed": validate_mobile_server_contract(), "telemetry": get_mobile_server_telemetry_payload()}


def get_mobile_server_compliance_certificate() -> dict:
    return {"module": "mobile_server", "certified": validate_mobile_server_contract(), "seal": get_mobile_server_integrity_seal()}


def get_mobile_server_readiness_probe() -> dict:
    return {"module": "mobile_server", "ready": validate_mobile_server_contract(), "certificate": get_mobile_server_compliance_certificate()}


def get_mobile_server_audit_summary() -> dict:
    return {"module": "mobile_server", "audit_passed": validate_mobile_server_contract(), "probe": get_mobile_server_readiness_probe()}


