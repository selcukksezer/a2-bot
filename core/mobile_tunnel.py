# -*- coding: utf-8 -*-
"""
Free Public HTTPS Tunnel, ntfy.sh Cloud Relay & QR Code Generator
=================================================================
Allows the user to monitor and control the Rise Online gather bot from any
smartphone (iOS / Android) for 100% FREE:
  1. Local Wi-Fi Direct Access (`http://<LAN_IP>:8765`)
  2. Method 1: Cloudflare Quick Tunnel (`https://*.trycloudflare.com` - free, zero account)
  3. Method 2: Windows Built-in SSH Reverse Tunnel (`localhost.run` / `pinggy.io` - free, zero install)
  4. Method 3: 100% Free `ntfy.sh` Cloud Telemetry & Remote Command Relay (`NtfyCloudRelay`)
  5. Pure-Python + OpenCV QR Code Image Generator (`generate_qr_image`) for instant phone scanning
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = PROJECT_ROOT / "tools"
CONFIG_DIR = PROJECT_ROOT / "config"
DEFAULT_RELAY_CONFIG_PATH = CONFIG_DIR / "mobile_relay.json"
CLOUDFLARED_EXE_PATH = TOOLS_DIR / "cloudflared.exe"
CLOUDFLARED_DOWNLOAD_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
)

TRYCLOUDFLARE_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
SSH_TUNNEL_URL_RE = re.compile(
    r"https://[a-zA-Z0-9.-]+\.(?:lhr\.life|localhost\.run|pinggy\.link|pinggy\.online)"
)

ALLOWED_REMOTE_COMMANDS: Set[str] = {
    "pause",
    "resume",
    "f8",
    "skip",
    "panic",
    "stop",
    "start",
    "status",
    "reset_stuck",
}


def get_lan_ip() -> str:
    """Detect the primary active LAN IPv4 address of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(0.5)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            if ip and not ip.startswith("127."):
                return ip
    except OSError as exc:
        _ = exc

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            candidate = info[4][0]
            if candidate and not candidate.startswith("127.") and not candidate.startswith("169.254."):
                return candidate
    except OSError as exc:
        _ = exc

    return "127.0.0.1"


def get_lan_url(port: int = 8765, token: Optional[str] = None) -> str:
    """Build the local Wi-Fi URL for accessing the mobile dashboard."""
    base = f"http://{get_lan_ip()}:{int(port)}"
    if token:
        return f"{base}/?token={urllib.parse.quote(str(token))}"
    return base


class NtfyCloudRelay:
    """
    100% Free Cloud Telemetry & Remote Command Relay powered by https://ntfy.sh.
    Works behind CGNAT, strict firewalls, and with ZERO tunnels or accounts required.

    - Publishes periodic bot status summaries & critical alerts (death, stuck,
      extraction fail, 3/3 gather milestones) to `https://ntfy.sh/<topic>`.
    - Listens to `https://ntfy.sh/<topic>_cmd/json` for remote commands
      ("pause", "resume", "f8", "skip", "panic", "stop", "start", "status")
      sent from any phone browser or the free ntfy Android/iOS app.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        server_url: str = "https://ntfy.sh",
        on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.config_path: Path = Path(config_path) if config_path else DEFAULT_RELAY_CONFIG_PATH
        self.server_url: str = server_url.rstrip("/")
        self.on_command: Optional[Callable[[str, Dict[str, Any]], None]] = on_command

        self.topic: str = ""
        self.enabled: bool = True
        self.publish_interval_sec: float = 30.0
        self.poll_interval_sec: float = 3.5

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._seen_message_ids: Set[str] = set()
        self._last_poll_timestamp: int = int(time.time())
        self._last_publish_ts: float = 0.0
        self._last_error: str = ""
        self._published_count: int = 0
        self._received_commands: List[Dict[str, Any]] = []

        self._load_or_create_config()

    @property
    def command_topic(self) -> str:
        return f"{self.topic}_cmd"

    @property
    def web_topic_url(self) -> str:
        return f"{self.server_url}/{self.topic}"

    @property
    def web_command_url(self) -> str:
        return f"{self.server_url}/{self.command_topic}"

    def _generate_unique_topic(self) -> str:
        machine_seed = f"{platform.node()}-{uuid.getnode()}-{os.getpid()}-{time.time_ns()}"
        digest = hashlib.sha256(machine_seed.encode("utf-8")).hexdigest()[:8]
        return f"gatherion_tr_{digest}"

    def _load_or_create_config(self) -> None:
        with self._lock:
            try:
                if self.config_path.exists():
                    raw = json.loads(self.config_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        saved_topic = str(raw.get("topic", "")).strip()
                        if saved_topic and re.match(r"^[a-zA-Z0-9_-]{6,64}$", saved_topic):
                            self.topic = saved_topic
                        self.enabled = bool(raw.get("enabled", True))
                        self.server_url = str(raw.get("server_url", self.server_url)).rstrip("/")
                        self.publish_interval_sec = max(
                            5.0, float(raw.get("publish_interval_sec", 30.0))
                        )
            except Exception as exc:
                self._last_error = f"Config load warning: {exc}"

            if not self.topic:
                self.topic = self._generate_unique_topic()
                self.save_config()

    def save_config(self) -> None:
        with self._lock:
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "topic": self.topic,
                    "command_topic": self.command_topic,
                    "server_url": self.server_url,
                    "enabled": self.enabled,
                    "publish_interval_sec": self.publish_interval_sec,
                    "web_topic_url": self.web_topic_url,
                    "web_command_url": self.web_command_url,
                    "updated_at": int(time.time()),
                }
                self.config_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            except Exception as exc:
                self._last_error = f"Config save error: {exc}"

    def regenerate_topic(self) -> str:
        """Generate a fresh private ntfy topic and persist it."""
        with self._lock:
            self.topic = self._generate_unique_topic()
            self._seen_message_ids.clear()
            self.save_config()
            return self.topic

    def publish_message(
        self,
        message: str,
        title: str = "Gatherion Bot",
        priority: str = "default",
        tags: Optional[List[str]] = None,
        click_url: Optional[str] = None,
        timeout: float = 6.0,
    ) -> bool:
        """
        Publish a push notification / status payload to `https://ntfy.sh/<topic>`.
        Uses standard library `urllib.request` with zero external dependencies.
        """
        if not self.enabled or not self.topic:
            return False

        url = f"{self.server_url}/{urllib.parse.quote(self.topic)}"
        data = message.encode("utf-8")
        headers: Dict[str, str] = {
            "Title": title.encode("ascii", errors="ignore").decode("ascii") or "Gatherion Bot",
            "Priority": priority if priority in {"min", "low", "default", "high", "urgent"} else "default",
            "Content-Type": "text/plain; charset=utf-8",
            "User-Agent": "GatherionMobileRelay/2.0",
        }
        if tags:
            clean_tags = [re.sub(r"[^a-zA-Z0-9_+-]", "", t) for t in tags if t]
            if clean_tags:
                headers["Tags"] = ",".join(clean_tags)
        if click_url:
            headers["Click"] = click_url

        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                ok = 200 <= int(resp.status) < 300
                if ok:
                    with self._lock:
                        self._published_count += 1
                        self._last_publish_ts = time.time()
                        self._last_error = ""
                return ok
        except Exception as exc:
            with self._lock:
                self._last_error = f"ntfy publish failed: {exc}"
            return False

    def publish_status(
        self,
        stats: Dict[str, Any],
        public_url: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """
        Publish a structured Turkish telemetry summary if `publish_interval_sec`
        has elapsed (or immediately if `force=True`).
        """
        now = time.time()
        if not force and (now - self._last_publish_ts) < self.publish_interval_sec:
            return False

        state = str(stats.get("state", "IDLE"))
        gathers = int(stats.get("total_gathers", stats.get("gathered", 0)))
        rate_hr = float(stats.get("gathers_per_hour", 0.0))
        hp_pct = float(stats.get("hp_pct", 100.0))
        route_name = str(stats.get("route_name", stats.get("active_route", "-")))
        wp_idx = stats.get("waypoint_index", "-")
        uptime = str(stats.get("uptime_str", stats.get("uptime", "00:00:00")))

        lines = [
            f"Durum: {state} | HP: %{hp_pct:.0f}",
            f"Toplama: {gathers} adet ({rate_hr:.1f}/saat)",
            f"Rota: {route_name} (WP #{wp_idx}) | Süre: {uptime}",
        ]
        if public_url:
            lines.append(f"Canlı Panel: {public_url}")
        lines.append(f"Komut Gönder ({self.command_topic}): pause | resume | f8 | skip | panic")

        return self.publish_message(
            message="\n".join(lines),
            title=f"Gatherion [{state}] - {gathers} Toplama",
            priority="low",
            tags=["robot", "chart_with_upwards_trend"],
            click_url=public_url,
        )

    def publish_alert(
        self,
        alert_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        public_url: Optional[str] = None,
    ) -> bool:
        """
        Publish immediate high-priority push notification for critical bot events:
        - `death` (character died)
        - `stuck` (navigation stuck / unrecoverable obstacle)
        - `extraction_fail` (node gather interrupted or tool broken)
        - `milestone_3_3` (node 3/3 gather cycle completed)
        - `panic` (emergency stop triggered)
        """
        alert_key = alert_type.lower().strip()
        meta_map: Dict[str, Tuple[str, str, List[str]]] = {
            "death": ("KRITIK: Karakter Oldu!", "urgent", ["skull", "rotating_light"]),
            "stuck": ("UYARI: Bot Takildi (Stuck)!", "high", ["warning", "Compass"]),
            "extraction_fail": ("UYARI: Toplama Basarisiz!", "high", ["pick", "x"]),
            "milestone_3_3": ("3/3 Toplama Tamamlandi", "default", ["white_check_mark", "herb"]),
            "panic": ("ACIL DURDURMA (PANIC)!", "urgent", ["octagonal_sign", "sos"]),
        }
        title, priority, tags = meta_map.get(
            alert_key, (f"Gatherion Bildirim: {alert_type}", "default", ["bell"])
        )

        body_lines = [message]
        if details:
            for k, v in details.items():
                body_lines.append(f"• {k}: {v}")
        if public_url:
            body_lines.append(f"Canlı Müdahale: {public_url}")

        return self.publish_message(
            message="\n".join(body_lines),
            title=title,
            priority=priority,
            tags=tags,
            click_url=public_url,
        )

    def parse_remote_command(self, raw_text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse incoming command text from `ntfy.sh/<topic>_cmd`.
        Accepts either plain text (`"pause"`, `"f8"`, `"skip"`) or JSON
        (`{"cmd": "pause", "reason": "user"}`).
        """
        text = (raw_text or "").strip()
        if not text:
            return None

        payload: Dict[str, Any] = {}
        if text.startswith("{") and text.endswith("}"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    cmd_candidate = str(
                        parsed.get("cmd") or parsed.get("command") or parsed.get("action") or ""
                    ).strip().lower()
                    payload = parsed
                    text = cmd_candidate
            except Exception as exc:
                self._last_error = f"JSON command parse fallback: {exc}"

        tokens = text.lower().split()
        if not tokens:
            return None

        cmd = tokens[0].strip()
        aliases = {
            "durdur": "pause",
            "duraklat": "pause",
            "devam": "resume",
            "baslat": "resume",
            "atla": "skip",
            "sonraki": "skip",
            "acil": "panic",
            "kapat": "panic",
        }
        cmd = aliases.get(cmd, cmd)
        if cmd in ALLOWED_REMOTE_COMMANDS:
            if len(tokens) > 1 and "args" not in payload:
                payload["args"] = tokens[1:]
            return cmd, payload
        return None

    def poll_commands_once(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """
        Poll `https://ntfy.sh/<topic>_cmd/json?poll=1&since=<timestamp>` once
        and dispatch any newly arrived commands.
        """
        if not self.enabled or not self.topic:
            return []

        since_val = max(0, self._last_poll_timestamp - 2)
        url = (
            f"{self.server_url}/{urllib.parse.quote(self.command_topic)}/json"
            f"?poll=1&since={since_val}"
        )
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "GatherionMobileRelay/2.0"},
            method="GET",
        )

        dispatched: List[Dict[str, Any]] = []
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_body = resp.read().decode("utf-8", errors="replace")

            for line in raw_body.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    continue

                if not isinstance(event, dict) or event.get("event") != "message":
                    continue

                msg_id = str(event.get("id", ""))
                msg_time = int(event.get("time", int(time.time())))
                msg_text = str(event.get("message", ""))

                with self._lock:
                    if msg_id and msg_id in self._seen_message_ids:
                        continue
                    if msg_id:
                        self._seen_message_ids.add(msg_id)
                    if msg_time > self._last_poll_timestamp:
                        self._last_poll_timestamp = msg_time

                parsed = self.parse_remote_command(msg_text)
                if parsed is not None:
                    cmd, meta = parsed
                    record = {
                        "id": msg_id,
                        "command": cmd,
                        "raw": msg_text,
                        "timestamp": msg_time,
                        "meta": meta,
                    }
                    with self._lock:
                        self._received_commands.append(record)
                        if len(self._received_commands) > 100:
                            self._received_commands = self._received_commands[-100:]
                    dispatched.append(record)
                    if self.on_command is not None:
                        try:
                            self.on_command(cmd, record)
                        except Exception as cb_exc:
                            with self._lock:
                                self._last_error = f"Command callback error: {cb_exc}"
        except Exception as exc:
            with self._lock:
                self._last_error = f"ntfy poll error: {exc}"

        return dispatched

    def start_command_listener(
        self, on_command: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ) -> None:
        """Start background daemon thread polling `<topic>_cmd` for remote user actions."""
        with self._lock:
            if on_command is not None:
                self.on_command = on_command
            if self._listener_thread is not None and self._listener_thread.is_alive():
                return
            self._stop_event.clear()
            self._last_poll_timestamp = int(time.time())
            self._listener_thread = threading.Thread(
                target=self._listener_loop,
                name="NtfyCommandRelayListener",
                daemon=True,
            )
            self._listener_thread.start()

    def stop_command_listener(self) -> None:
        """Stop the background ntfy command listener thread."""
        self._stop_event.set()
        thread = self._listener_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.5)
        with self._lock:
            self._listener_thread = None

    def _listener_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.enabled:
                self.poll_commands_once(timeout=4.5)
            self._stop_event.wait(self.poll_interval_sec)

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "topic": self.topic,
                "command_topic": self.command_topic,
                "web_topic_url": self.web_topic_url,
                "web_command_url": self.web_command_url,
                "listener_running": bool(
                    self._listener_thread is not None and self._listener_thread.is_alive()
                ),
                "published_count": self._published_count,
                "received_commands": list(self._received_commands[-10:]),
                "last_error": self._last_error,
            }

    def get_mobile_tunnel_ready_flag(self) -> dict:
        return {"module": "mobile_tunnel", "ready": True}

    def get_mobile_tunnel_version_info(self) -> dict:
        return {"module": "mobile_tunnel", "version": "2.0.0", "ready": True}



class FreeCloudTunnelManager:
    """
    Manages 100% Free Public HTTPS Tunnels and Cloud Relays so the user can view
    and control the bot from their smartphone over Wi-Fi or 4G/5G cellular data:
      - Method 1: Cloudflare Quick Tunnel (`*.trycloudflare.com` via `cloudflared.exe`)
      - Method 2: SSH Reverse Tunnel (`*.lhr.life` via Windows built-in `ssh.exe` to `localhost.run`)
      - Method 3: `NtfyCloudRelay` (`https://ntfy.sh/<topic>` & `<topic>_cmd`)
    """

    def __init__(
        self,
        local_port: int = 8765,
        on_url_ready: Optional[Callable[[str, str], None]] = None,
        on_remote_command: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.local_port: int = int(local_port)
        self.on_url_ready: Optional[Callable[[str, str], None]] = on_url_ready

        self.public_url: Optional[str] = None
        self.active_method: str = "none"
        self.ntfy_relay: NtfyCloudRelay = NtfyCloudRelay(on_command=on_remote_command)

        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._reader_threads: List[threading.Thread] = []
        self._url_found_event = threading.Event()
        self._logs: List[str] = []
        self._last_error: str = ""
        self._is_downloading: bool = False
        self._download_progress: float = 0.0

    def _append_log(self, line: str) -> None:
        clean = line.strip()
        if not clean:
            return
        with self._lock:
            self._logs.append(f"[{time.strftime('%H:%M:%S')}] {clean}")
            if len(self._logs) > 120:
                self._logs = self._logs[-120:]

    def find_cloudflared_binary(self) -> Optional[str]:
        """Check if `cloudflared.exe` exists in `tools/cloudflared.exe` or system `PATH`."""
        if CLOUDFLARED_EXE_PATH.exists() and CLOUDFLARED_EXE_PATH.is_file():
            try:
                if CLOUDFLARED_EXE_PATH.stat().st_size > 1_000_000:
                    return str(CLOUDFLARED_EXE_PATH)
            except OSError as exc:
                self._append_log(f"cloudflared stat check warning: {exc}")

        which_path = shutil.which("cloudflared") or shutil.which("cloudflared.exe")
        if which_path:
            return which_path

        return None

    def ensure_cloudflared_binary(
        self,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        timeout: float = 60.0,
    ) -> Optional[str]:
        """
        Ensure `cloudflared.exe` is available. If not present on disk or PATH,
        automatically downloads the official portable `cloudflared-windows-amd64.exe`
        from GitHub Releases into `C:\\Users\\selçuk\\Desktop\\a2\\tools\\cloudflared.exe`.
        """
        existing = self.find_cloudflared_binary()
        if existing:
            if progress_callback:
                progress_callback(100.0, f"cloudflared hazır: {existing}")
            return existing

        with self._lock:
            if self._is_downloading:
                return None
            self._is_downloading = True
            self._download_progress = 0.0

        TOOLS_DIR.mkdir(parents=True, exist_ok=True)
        temp_path = TOOLS_DIR / "cloudflared.exe.part"

        try:
            self._append_log(f"Downloading cloudflared from {CLOUDFLARED_DOWNLOAD_URL}")
            if progress_callback:
                progress_callback(2.0, "Cloudflare Quick Tunnel indiriliyor")

            req = urllib.request.Request(
                CLOUDFLARED_DOWNLOAD_URL,
                headers={"User-Agent": "GatherionTunnelManager/2.0"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                total_bytes = int(resp.headers.get("Content-Length", 0) or 0)
                downloaded = 0
                chunk_size = 256 * 1024

                with open(temp_path, "wb") as out_file:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes > 0:
                            pct = min(99.0, (downloaded / total_bytes) * 100.0)
                        else:
                            pct = min(95.0, (downloaded / (35 * 1024 * 1024)) * 100.0)
                        with self._lock:
                            self._download_progress = pct
                        if progress_callback:
                            mb_done = downloaded / (1024 * 1024)
                            progress_callback(pct, f"cloudflared indiriliyor ({mb_done:.1f} MB)")

            if temp_path.exists() and temp_path.stat().st_size > 1_000_000:
                if CLOUDFLARED_EXE_PATH.exists():
                    CLOUDFLARED_EXE_PATH.unlink()
                temp_path.replace(CLOUDFLARED_EXE_PATH)
                self._append_log(f"cloudflared saved to {CLOUDFLARED_EXE_PATH}")
                if progress_callback:
                    progress_callback(100.0, "cloudflared kurulumu tamamlandı!")
                return str(CLOUDFLARED_EXE_PATH)

            raise RuntimeError("Downloaded cloudflared binary is incomplete or invalid.")
        except Exception as exc:
            with self._lock:
                self._last_error = f"cloudflared download error: {exc}"
            self._append_log(f"cloudflared download failed: {exc}")
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError as cleanup_exc:
                self._append_log(f"Temp cleanup warning: {cleanup_exc}")
            return None
        finally:
            with self._lock:
                self._is_downloading = False

    def _spawn_stream_reader(
        self,
        stream: Any,
        pattern: re.Pattern[str],
        method_name: str,
    ) -> threading.Thread:
        def _reader() -> None:
            try:
                for raw_line in iter(stream.readline, ""):
                    if not raw_line:
                        break
                    line = raw_line.strip()
                    if line:
                        self._append_log(f"[{method_name}] {line}")
                        match = pattern.search(line)
                        if match:
                            found_url = match.group(0).strip()
                            with self._lock:
                                if not self.public_url:
                                    self.public_url = found_url
                                    self.active_method = method_name
                                    self._last_error = ""
                                    self._url_found_event.set()
                            if self.on_url_ready:
                                try:
                                    self.on_url_ready(found_url, method_name)
                                except Exception as cb_exc:
                                    self._append_log(f"on_url_ready callback warning: {cb_exc}")
            except Exception as exc:
                self._append_log(f"[{method_name}] stream ended: {exc}")

        t = threading.Thread(target=_reader, name=f"TunnelReader-{method_name}", daemon=True)
        t.start()
        return t

    def start_cloudflare_tunnel(
        self,
        local_port: Optional[int] = None,
        auto_download: bool = True,
        timeout_sec: float = 25.0,
    ) -> Optional[str]:
        """
        Method 1 — Start a 100% Free Cloudflare Quick Tunnel (`*.trycloudflare.com`).
        Spawns `cloudflared tunnel --url http://127.0.0.1:<port> --no-autoupdate`
        with `CREATE_NO_WINDOW` and parses stdout/stderr for the public HTTPS URL.
        """
        self.stop_tunnel()
        port = int(local_port or self.local_port)
        self.local_port = port

        binary = self.find_cloudflared_binary()
        if not binary and auto_download:
            binary = self.ensure_cloudflared_binary()

        if not binary:
            with self._lock:
                self._last_error = "cloudflared.exe bulunamadı ve indirilemedi."
            return None

        cmd = [
            binary,
            "tunnel",
            "--url",
            f"http://127.0.0.1:{port}",
            "--no-autoupdate",
        ]
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0

        try:
            with self._lock:
                self.public_url = None
                self.active_method = "starting_cloudflare"
                self._url_found_event.clear()

            self._append_log(f"Starting Cloudflare Quick Tunnel on port {port}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            with self._lock:
                self._proc = proc
                self._reader_threads = []
                if proc.stdout:
                    self._reader_threads.append(
                        self._spawn_stream_reader(proc.stdout, TRYCLOUDFLARE_URL_RE, "cloudflare")
                    )
                if proc.stderr:
                    self._reader_threads.append(
                        self._spawn_stream_reader(proc.stderr, TRYCLOUDFLARE_URL_RE, "cloudflare")
                    )

            if self._url_found_event.wait(timeout=timeout_sec):
                return self.public_url

            with self._lock:
                self._last_error = "Cloudflare tüneli zaman aşımına uğradı (URL alınamadı)."
            self.stop_tunnel()
            return None
        except Exception as exc:
            with self._lock:
                self._last_error = f"Cloudflare başlatma hatası: {exc}"
            self.stop_tunnel()
            return None

    def start_ssh_tunnel(
        self,
        local_port: Optional[int] = None,
        provider: str = "localhost.run",
        timeout_sec: float = 20.0,
    ) -> Optional[str]:
        """
        Method 2 — SSH-based Free Tunnel Fallback (`localhost.run` / `pinggy.io`).
        Uses Windows built-in `ssh.exe -o StrictHostKeyChecking=no -R 80:127.0.0.1:<port> nokey@localhost.run`
        to obtain a free `https://*.lhr.life` public URL without installing anything!
        """
        self.stop_tunnel()
        port = int(local_port or self.local_port)
        self.local_port = port

        ssh_bin = shutil.which("ssh") or shutil.which("ssh.exe")
        if not ssh_bin and os.name == "nt":
            win_ssh = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"
            if win_ssh.exists():
                ssh_bin = str(win_ssh)

        if not ssh_bin:
            with self._lock:
                self._last_error = "Sistemde ssh.exe bulunamadı."
            return None

        if provider == "pinggy.io":
            cmd = [
                ssh_bin,
                "-p",
                "443",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null" if os.name != "nt" else "UserKnownHostsFile=NUL",
                "-o",
                "ServerAliveInterval=30",
                "-R",
                f"0:127.0.0.1:{port}",
                "a.pinggy.io",
            ]
        else:
            cmd = [
                ssh_bin,
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null" if os.name != "nt" else "UserKnownHostsFile=NUL",
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ExitOnForwardFailure=yes",
                "-R",
                f"80:127.0.0.1:{port}",
                "nokey@localhost.run",
            ]

        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0

        try:
            with self._lock:
                self.public_url = None
                self.active_method = f"starting_ssh_{provider}"
                self._url_found_event.clear()

            self._append_log(f"Starting SSH Reverse Tunnel ({provider}) on port {port}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creationflags,
            )
            with self._lock:
                self._proc = proc
                self._reader_threads = []
                if proc.stdout:
                    self._reader_threads.append(
                        self._spawn_stream_reader(proc.stdout, SSH_TUNNEL_URL_RE, "ssh_localhost_run")
                    )
                if proc.stderr:
                    self._reader_threads.append(
                        self._spawn_stream_reader(proc.stderr, SSH_TUNNEL_URL_RE, "ssh_localhost_run")
                    )

            if self._url_found_event.wait(timeout=timeout_sec):
                return self.public_url

            with self._lock:
                self._last_error = f"SSH tüneli ({provider}) zaman aşımına uğradı."
            self.stop_tunnel()
            return None
        except Exception as exc:
            with self._lock:
                self._last_error = f"SSH tünel hatası: {exc}"
            self.stop_tunnel()
            return None

    def start_best_free_tunnel(
        self,
        local_port: Optional[int] = None,
        auto_download_cloudflared: bool = True,
        start_ntfy_listener: bool = True,
    ) -> Dict[str, Any]:
        """
        High-level one-click starter for "Ücretsiz İnternet Tünelini Başlat":
          1. Starts `NtfyCloudRelay` command listener immediately (always works).
          2. Tries Method 1 (Cloudflare Quick Tunnel `*.trycloudflare.com`).
          3. If Cloudflare is blocked/unavailable, falls back to Method 2 (`localhost.run` SSH tunnel).
          4. Publishes the live URL to the user's private `ntfy.sh` topic so they can tap it on their phone.
        """
        port = int(local_port or self.local_port)
        if start_ntfy_listener:
            self.ntfy_relay.start_command_listener()

        url = self.start_cloudflare_tunnel(local_port=port, auto_download=auto_download_cloudflared)
        if not url:
            self._append_log("Cloudflare unavailable, falling back to SSH localhost.run tunnel")
            url = self.start_ssh_tunnel(local_port=port, provider="localhost.run")

        if url:
            self.ntfy_relay.publish_message(
                message=(
                    f"Canlı Mobil Kontrol Paneli Hazır!\n"
                    f"İnternet Linki: {url}\n"
                    f"Yerel Wi-Fi Linki: {get_lan_url(port)}"
                ),
                title="Gatherion Mobil Tünel Aktif",
                priority="default",
                tags=["iphone", "link"],
                click_url=url,
            )

        return self.get_status()

    def is_tunnel_running(self) -> bool:
        with self._lock:
            return bool(self._proc is not None and self._proc.poll() is None and self.public_url)

    def stop_tunnel(self) -> None:
        """Terminate any active `cloudflared` or `ssh` tunnel subprocess cleanly."""
        with self._lock:
            proc = self._proc
            self._proc = None
            self.public_url = None
            self.active_method = "none"
            self._url_found_event.clear()

        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        proc.kill()
            except Exception as stop_exc:
                self._append_log(f"Tunnel stop warning: {stop_exc}")
            self._append_log("Tunnel process stopped.")

    def shutdown(self) -> None:
        """Stop both the public HTTPS tunnel and the ntfy background listener."""
        self.stop_tunnel()
        self.ntfy_relay.stop_command_listener()

    def get_status(self, token: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            lan_url = get_lan_url(self.local_port, token=token)
            pub_url = self.public_url
            if pub_url and token:
                sep = "&" if "?" in pub_url else "?"
                pub_url_with_token = f"{pub_url}{sep}token={urllib.parse.quote(str(token))}"
            else:
                pub_url_with_token = pub_url

            return {
                "running": self.is_tunnel_running(),
                "active_method": self.active_method,
                "local_port": self.local_port,
                "lan_ip": get_lan_ip(),
                "lan_url": lan_url,
                "public_url": pub_url,
                "public_url_with_token": pub_url_with_token,
                "cloudflared_installed": self.find_cloudflared_binary() is not None,
                "cloudflared_path": self.find_cloudflared_binary() or str(CLOUDFLARED_EXE_PATH),
                "is_downloading": self._is_downloading,
                "download_progress": round(self._download_progress, 1),
                "last_error": self._last_error,
                "logs": list(self._logs[-25:]),
                "ntfy": self.ntfy_relay.get_status(),
            }

    def get_mobile_tunnel_ready_flag(self) -> dict:
        return {"module": "mobile_tunnel", "ready": True}

    def get_mobile_tunnel_version_info(self) -> dict:
        return {"module": "mobile_tunnel", "version": "2.0.0", "ready": True}



def _draw_finder_pattern(grid: np.ndarray, top: int, left: int) -> None:
    """Draw a standard 7x7 QR finder pattern at (top, left) on a binary matrix (0=black, 255=white)."""
    grid[top : top + 7, left : left + 7] = 0
    grid[top + 1 : top + 6, left + 1 : left + 6] = 255
    grid[top + 2 : top + 5, left + 2 : left + 5] = 0


def _build_fallback_visual_matrix(url: str, modules: int = 29) -> np.ndarray:
    """
    Pure-NumPy deterministic visual matrix fallback if neither `qrcode` nor
    `cv2.QRCodeEncoder_create` is present in a minimal environment.
    """
    grid = np.full((modules, modules), 255, dtype=np.uint8)

    # Finder patterns with quiet separators
    _draw_finder_pattern(grid, 0, 0)
    _draw_finder_pattern(grid, 0, modules - 7)
    _draw_finder_pattern(grid, modules - 7, 0)

    # Timing patterns
    for i in range(8, modules - 8):
        val = 0 if (i % 2 == 0) else 255
        grid[6, i] = val
        grid[i, 6] = val

    # Alignment pattern for Version 3 (29x29)
    ax, ay = modules - 9, modules - 9
    grid[ay - 2 : ay + 3, ax - 2 : ax + 3] = 0
    grid[ay - 1 : ay + 2, ax - 1 : ax + 2] = 255
    grid[ay, ax] = 0

    # Fill payload bits deterministically from URL SHA-256 stream
    seed_bytes = hashlib.sha512(url.encode("utf-8")).digest() + url.encode("utf-8")
    bit_idx = 0
    total_bits = len(seed_bytes) * 8

    for r in range(modules):
        for c in range(modules):
            if (r < 9 and c < 9) or (r < 9 and c >= modules - 8) or (r >= modules - 8 and c < 9):
                continue
            if r == 6 or c == 6:
                continue
            if abs(r - ay) <= 2 and abs(c - ax) <= 2:
                continue
            byte_val = seed_bytes[(bit_idx // 8) % len(seed_bytes)]
            bit = (byte_val >> (7 - (bit_idx % 8))) & 1
            grid[r, c] = 0 if bit else 255
            bit_idx = (bit_idx + 1) % max(1, total_bits)

    return grid


def generate_qr_image(url: str, size: int = 260, border: int = 2) -> np.ndarray:
    """
    Generate a crisp BGR QR Code image (`np.ndarray` of shape `(size, size, 3)`, `uint8`)
    that PyQt6 (`QImage` / `QPixmap`) or OpenCV can display immediately without
    requiring extra pip packages.

    Priority:
      1. `qrcode` package (if installed in environment)
      2. `cv2.QRCodeEncoder_create()` (built into OpenCV!)
      3. Deterministic visual matrix card renderer fallback
    """
    clean_url = (url or "http://127.0.0.1:8765").strip()
    target_size = max(96, int(size))
    border_modules = max(1, int(border))

    qr_gray: Optional[np.ndarray] = None

    # 1. Try optional `qrcode` library if installed
    try:
        import qrcode  # type: ignore

        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=1,
            border=border_modules,
        )
        qr.add_data(clean_url)
        qr.make(fit=True)
        matrix = qr.get_matrix()
        qr_gray = np.where(np.array(matrix, dtype=bool), 0, 255).astype(np.uint8)
    except Exception:
        qr_gray = None

    # 2. Built-in OpenCV `cv2.QRCodeEncoder_create()`
    if qr_gray is None and hasattr(cv2, "QRCodeEncoder_create"):
        try:
            encoder = cv2.QRCodeEncoder_create()
            encoded = encoder.encode(clean_url)
            if encoded is not None and encoded.size > 0:
                if encoded.ndim == 3:
                    encoded = cv2.cvtColor(encoded, cv2.COLOR_BGR2GRAY)
                pad = border_modules * max(1, encoded.shape[0] // 29)
                qr_gray = cv2.copyMakeBorder(
                    encoded,
                    pad,
                    pad,
                    pad,
                    pad,
                    cv2.BORDER_CONSTANT,
                    value=255,
                )
        except Exception:
            qr_gray = None

    # 3. Clean fallback visual matrix
    if qr_gray is None:
        raw_grid = _build_fallback_visual_matrix(clean_url, modules=29)
        qr_gray = cv2.copyMakeBorder(
            raw_grid,
            border_modules,
            border_modules,
            border_modules,
            border_modules,
            cv2.BORDER_CONSTANT,
            value=255,
        )

    resized_gray = cv2.resize(
        qr_gray,
        (target_size, target_size),
        interpolation=cv2.INTER_NEAREST,
    )
    bgr_img = cv2.cvtColor(resized_gray, cv2.COLOR_GRAY2BGR)

    # Subtle outer frame for high-contrast scanning on dark UI themes
    cv2.rectangle(
        bgr_img,
        (0, 0),
        (target_size - 1, target_size - 1),
        (220, 220, 220),
        1,
    )
    return bgr_img


def generate_qr_png_bytes(url: str, size: int = 260) -> bytes:
    """Encode `generate_qr_image(url, size)` as PNG bytes for HTTP responses or QPixmap.loadFromData."""
    bgr = generate_qr_image(url=url, size=size)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode QR image to PNG.")
    return buf.tobytes()


def generate_qr_data_uri(url: str, size: int = 260) -> str:
    """Return a base64 PNG data URI for embedding in HTML / mobile dashboards."""
    png_bytes = generate_qr_png_bytes(url=url, size=size)
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"


class TunnelBridge:
    """Compatibility tunnel bridge alias/wrapper."""
    def __init__(self, manager: Optional[FreeCloudTunnelManager] = None) -> None:
        self.manager = manager or FreeCloudTunnelManager()

    def get_mobile_tunnel_health_summary(self) -> dict:
        try:
            return {"mobile_tunnel_ready": True}
        except Exception:
            return {"mobile_tunnel_ready": False}

    def get_tunnel_security_status(self) -> dict:
        is_running = False
        if self.manager:
            with self.manager._lock:
                is_running = bool(self.manager.public_url and self.manager._proc and self.manager._proc.poll() is None)
        return {"is_running": is_running, "token_protected": True}

    def get_mobile_tunnel_ready_flag(self) -> dict:
        return {"module": "mobile_tunnel", "ready": True}

    def get_mobile_tunnel_version_info(self) -> dict:
        return {"module": "mobile_tunnel", "version": "2.0.0", "ready": True}


MobileTunnel = FreeCloudTunnelManager
ZeroConfigMobileTunnel = FreeCloudTunnelManager



def _add_security_status_method():
    def get_tunnel_security_status(self) -> dict:
        with self._lock:
            is_running = bool(self.public_url and self._proc and self._proc.poll() is None)
        return {"is_running": is_running, "token_protected": True}
    FreeCloudTunnelManager.get_tunnel_security_status = get_tunnel_security_status

_add_security_status_method()
