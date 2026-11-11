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
import hmac
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from version import __version__


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

# ntfy topic formatlari: yeni konular 128-bit rastgele (32 hex), eski
# deterministik konular 32-bit (8 hex) idi ve tahmin edilebiliyordu.
_NTFY_TOPIC_RE = re.compile(r"^gatherion_tr_[0-9a-f]{32}$")
_WEAK_TOPIC_RE = re.compile(r"^gatherion_tr_[0-9a-f]{8}$")
_MAX_SEEN_MESSAGE_IDS = 5000
# Imzali uzaktan komut gecerlilik penceresi (P0-4 / K-09): imza yukundeki
# zaman damgasi, dogrulama anina gore bu pencerenin disindaysa komut
# reddedilir. Ornek sinif duzeyinde degistirilebilir; tekil relay
# ornegi icin `signed_command_max_age_sec` niteligi ayni degeri tasir.
SIGNED_COMMAND_MAX_AGE_SEC = 300
# Tek kullanimlik komut id'leri (cid) icin gorulmus-kume tavani. Asimda
# en eski kayit atilir (tum kumenin silinmesi eski id'lerin replay'ine
# yol acardi). Kayitlar yalnizca surec omru boyunca tutulur.
_MAX_SEEN_COMMAND_IDS = 5000
_COMMAND_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,128}$")
# ntfy yoklamasi cevrimdisi/hatali iken araligi katlayarak artirir:
# 3.5 sn -> 7 -> 14 -> 28 -> 30 sn (tavan); basarili yoklamada tabana doner.
_POLL_BACKOFF_CAP_SEC = 30.0


def _validate_server_url(url: str, default: str = "https://ntfy.sh") -> str:
    """Yalnizca https:// semali, host'lu bir relay URL'si kabul et.

    S325 (v19/P3-8): köşeli parantezsiz çıplak IPv6
    (https://2001:db8::1/...) eskiden geçiyordu — urlparse netloc'u
    doldurur ama hostname '2001' olur, istek yanlış hosta gider.
    Birden fazla ':' içeren netloc köşeli parantezli IPv6 olmak
    zorunda; ayrıca geçersiz port da reddedilir.
    """
    try:
        parsed = urllib.parse.urlparse(str(url or "").strip())
        if parsed.scheme == "https" and parsed.netloc and " " not in parsed.netloc:
            host = parsed.netloc.split("@")[-1]
            if host.count(":") > 1 and not host.startswith("["):
                return default
            _ = parsed.hostname  # geçersiz IPv6/port ValueError fırlatır
            _ = parsed.port
            return str(url).strip().rstrip("/")
    except Exception:
        pass
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


_LAN_IP_CACHE: Dict[str, Any] = {"ip": None, "at": 0.0}
_LAN_IP_TTL_SEC = 30.0


def _detect_lan_ip() -> str:
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


def get_lan_ip() -> str:
    """Detect the primary active LAN IPv4 address of this machine.

    Sonuc 30 sn TTL ile onbellege alinir: LAN IP'si nadiren degisir ama
    get_status()/get_lan_url() her cagrida (panel yoklamasi) soket aciyordu.
    TTL dolunca yeniden tespit edilir; cevrimdisi iken donen 127.0.0.1 de
    ayni TTL ile onbellege alinir (kalici degil, 30 sn sonra tekrar denenir).
    """
    now = time.monotonic()
    cached_ip = _LAN_IP_CACHE.get("ip")
    if cached_ip and (now - float(_LAN_IP_CACHE.get("at", 0.0))) < _LAN_IP_TTL_SEC:
        return str(cached_ip)
    ip = _detect_lan_ip()
    _LAN_IP_CACHE["ip"] = ip
    _LAN_IP_CACHE["at"] = now
    return ip


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
        # Uzaktan komut kimlik dogrulama: her kuruluma ozel, config dosyasinda
        # saklanan gizli anahtarla HMAC-SHA256 imzasi. Imzasiz komutlar
        # poll_commands_once tarafindan dagitilmaz (yalnizca sayilir).
        self.command_secret: str = ""
        self.require_signed_commands: bool = True
        # Imzali komut zaman penceresi (sn); SIGNED_COMMAND_MAX_AGE_SEC
        # varsayilanidir, ornek basina degistirilebilir.
        self.signed_command_max_age_sec: float = float(SIGNED_COMMAND_MAX_AGE_SEC)
        self._rejected_commands: int = 0

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._listener_thread: Optional[threading.Thread] = None
        self._seen_message_ids: Set[str] = set()
        # Gorulmus tek-kullanimlik komut id'leri (cid -> goruldugu zaman):
        # imzali komutlarda replay korumasi bu kumeye dayanir, ntfy
        # mesaj id'sine degil. Yalnizca surec omru boyunca tutulur.
        self._seen_command_ids: Dict[str, float] = {}
        self._last_poll_timestamp: int = int(time.time())
        self._last_publish_ts: float = 0.0
        self._last_error: str = ""
        self._published_count: int = 0
        self._received_commands: List[Dict[str, Any]] = []
        # Uyarlanabilir yoklama araligi: art arda hata sayisi ve o anki aralik.
        # _listener_loop bu araligi bekler; basarili yoklama tabana dondurur.
        self._consecutive_poll_errors: int = 0
        self._current_poll_interval_sec: float = self.poll_interval_sec

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
        # 128-bit kriptografik rastgele konu: hostname/PID/zaman gibi
        # tahmin edilebilir girdilerden turetilmez.
        return f"gatherion_tr_{secrets.token_hex(16)}"

    def _load_or_create_config(self) -> None:
        with self._lock:
            needs_save = False
            try:
                if self.config_path.exists():
                    raw = json.loads(self.config_path.read_text(encoding="utf-8"))
                    if isinstance(raw, dict):
                        saved_topic = str(raw.get("topic", "")).strip()
                        if saved_topic and re.match(r"^[a-zA-Z0-9_-]{6,64}$", saved_topic):
                            self.topic = saved_topic
                        self.enabled = bool(raw.get("enabled", True))
                        self.server_url = _validate_server_url(
                            str(raw.get("server_url", self.server_url)), self.server_url
                        )
                        self.publish_interval_sec = max(
                            5.0, float(raw.get("publish_interval_sec", 30.0))
                        )
                        saved_secret = str(raw.get("command_secret", "")).strip()
                        if saved_secret:
                            self.command_secret = saved_secret
                        if isinstance(raw.get("require_signed_commands"), bool):
                            self.require_signed_commands = bool(
                                raw.get("require_signed_commands")
                            )
            except Exception as exc:
                self._last_error = f"Config load warning: {exc}"

            # Eski 32-bit deterministik konu (8 hex) tahmin edilebilir oldugu
            # icin ilk yuklemede yenisiyle degistirilir. Bu goc yolu
            # regenerate_topic() uzerinden gecer: rotasyon mantigi tek
            # yerde kalir ve metot canli bir cagirana sahip olur (F-0138).
            if self.topic and _WEAK_TOPIC_RE.match(self.topic):
                self.regenerate_topic()

            if not self.command_secret:
                self.command_secret = secrets.token_hex(32)
                needs_save = True

            if not self.topic:
                self.topic = self._generate_unique_topic()
                needs_save = True

            if needs_save:
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
                    "command_secret": self.command_secret,
                    "require_signed_commands": self.require_signed_commands,
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
        """Generate a fresh private ntfy topic and persist it.

        Canli cagiranlar: zayif (8-hex) konu goc yolu
        (_load_or_create_config) ve FreeCloudTunnelManager.regenerate_ntfy_topic
        (diyalogdaki rotasyon butonunun baglanacagi herkese acik API).
        """
        with self._lock:
            self.topic = self._generate_unique_topic()
            self._seen_message_ids.clear()
            self._seen_command_ids.clear()
            self.save_config()
            return self.topic

    @staticmethod
    def _signed_payload(cmd: str, ts: int, command_id: str) -> str:
        """HMAC'a giren kanonik yuk: komut + zaman damgasi + tek-kullanimlik id."""
        return f"{cmd}\n{ts}\n{command_id}"

    def _record_seen_command_id(self, command_id: str, now: float) -> bool:
        """cid'yi gorulmus kumesine ekle; daha once gorulduyse False don.

        Suresi dolmus kayitlar budanir, kume tasarsa en eski kayit atilir
        (tum kumenin bosaltilmasi eski id'lere replay araligi acardi).
        """
        with self._lock:
            if command_id in self._seen_command_ids:
                return False
            cutoff = now - max(0.0, float(self.signed_command_max_age_sec))
            expired = [
                cid for cid, seen_at in self._seen_command_ids.items()
                if seen_at < cutoff
            ]
            for cid in expired:
                del self._seen_command_ids[cid]
            while len(self._seen_command_ids) >= _MAX_SEEN_COMMAND_IDS:
                self._seen_command_ids.pop(next(iter(self._seen_command_ids)))
            self._seen_command_ids[command_id] = now
            return True

    def sign_command(
        self,
        cmd: str,
        *,
        ts: Optional[int] = None,
        command_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Imzali uzaktan komut yuku uret (P0-4 / K-09).

        Yuk: {"cmd", "ts", "cid", "sig"}; "sig", kanonik
        "cmd\\n<ts>\\n<cid>" dizesinin HMAC-SHA256'sidir. "ts" unix
        saniye, "cid" tek-kullanimlik komut id'sidir (varsayilan:
        128-bit rastgele, 32 hex). Telefon tarafi komutlari bu imzayla
        gondermelidir; imzasiz / eski formatli komutlar
        poll_commands_once tarafindan dagitilmaz.
        """
        clean = str(cmd or "").strip().lower()
        stamp = int(time.time()) if ts is None else int(ts)
        cid = command_id if command_id else secrets.token_hex(16)
        sig = hmac.new(
            self.command_secret.encode("utf-8"),
            self._signed_payload(clean, stamp, cid).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {"cmd": clean, "ts": stamp, "cid": cid, "sig": sig}

    def verify_command_signature(
        self,
        cmd: str,
        signature: Any,
        *,
        ts: Any = None,
        command_id: Any = None,
        now: Optional[float] = None,
    ) -> bool:
        """Imzali komutu dogrula; fail-closed (P0-4 / K-09).

        Gecerli sayilmasi icin: (1) "ts" ve "cid" verilmis olmali —
        eski {"cmd", "sig"} yukleri reddedilir; (2) |simdi - ts| <=
        signed_command_max_age_sec (varsayilan 300 sn) olmali;
        (3) HMAC kanonik "cmd\\n<ts>\\n<cid>" uzerinden sabit sureli
        karsilastirmayla tutmali; (4) "cid" daha once gorulmemis
        olmali (replay reddi). Basarili dogrulama cid'yi gorulmus
        kumesine yazar, boylece ayni yukun ikinci sunumu (yeni ntfy
        mesaj id'siyle bile) reddedilir. "now" yalnizca test edilebilirlik
        icindir; verilmezse time.time() kullanilir.
        """
        if not cmd or not isinstance(signature, str) or not self.command_secret:
            return False
        clean = str(cmd).strip().lower()
        if not isinstance(command_id, str) or not _COMMAND_ID_RE.match(command_id):
            return False
        if isinstance(ts, bool):
            return False
        if isinstance(ts, int):
            stamp = ts
        elif isinstance(ts, str) and re.fullmatch(r"[0-9]+", ts.strip()):
            stamp = int(ts.strip())
        else:
            # Zaman damgasi yok / sayi degil: eski imzasiz form dahil reddet.
            return False
        moment = float(time.time()) if now is None else float(now)
        if abs(moment - stamp) > max(0.0, float(self.signed_command_max_age_sec)):
            return False
        expected = hmac.new(
            self.command_secret.encode("utf-8"),
            self._signed_payload(clean, stamp, command_id).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        try:
            if not hmac.compare_digest(expected, signature.strip()):
                return False
        except Exception:
            return False
        # HMAC + pencere gecti: cid'yi isaretle; tekrar sunum replay'dir.
        return self._record_seen_command_id(command_id, moment)

    def _note_message_id(self, dedup_key: str) -> bool:
        """ntfy mesaj id'sini gorulmus kumesine ekle; tekrar ise False don.

        Yalnizca cid tasimayan (imzasiz/legacy ya da komut olmayan)
        mesajlar icin kullanilir; imzali komutlarda dedup cid ile
        yapilir (verify_command_signature), mesaj id'siyle degil.
        """
        with self._lock:
            if dedup_key in self._seen_message_ids:
                return False
            if len(self._seen_message_ids) > _MAX_SEEN_MESSAGE_IDS:
                self._seen_message_ids.clear()
            self._seen_message_ids.add(dedup_key)
            return True

    def publish_command(self, cmd: str, timeout: float = 6.0) -> bool:
        """Y-5 (dilim 22): imzalı uzaktan komut gönder — `sign_command`
        ile üretilen `{"cmd", "ts", "cid", "sig"}` yükünü
        `<topic>_cmd` kanalına yazar; `poll_commands_once` bunu HMAC
        doğrulamasıyla dağıtır.
        Bu, imza üretecek repo-içi yoldur (CLI: `python -m
        core.mobile_tunnel send <komut>`). Komut ALLOWED listesinde
        değilse gönderilmez."""
        clean = str(cmd or "").strip().lower()
        if clean not in ALLOWED_REMOTE_COMMANDS:
            with self._lock:
                self._last_error = f"publish_command reddedildi (izinli komut değil): {clean}"
            return False
        if not self.enabled or not self.topic:
            return False
        signed = self.sign_command(clean)
        url = f"{self.server_url}/{urllib.parse.quote(self.command_topic)}"
        data = json.dumps(signed).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "User-Agent": "GatherionMobileRelay/2.0",
            },
            method="POST",
        )
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
                self._last_error = f"ntfy signed command publish failed: {exc}"
            return False

    def set_remote_command_handler(
        self, on_command: Optional[Callable[[str, Dict[str, Any]], None]]
    ) -> None:
        """Uzaktan komut callback'ini bagla (uretimde main.py tarafindan cagrilir)."""
        with self._lock:
            self.on_command = on_command

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
        gathers = _safe_int(stats.get("total_gathers", stats.get("gathered", 0)))
        rate_hr = _safe_float(stats.get("gathers_per_hour", 0.0))
        # HP okunamiyorsa uydurma varsayilan YOK (S-1901/S-2002): bilinmeyen
        # HP "%100" diye raporlanamaz; okunamayan deger "bilinmiyor" yazilir.
        hp_pct: Optional[float] = None
        hp_raw = stats.get("hp_pct")
        if hp_raw is not None:
            try:
                hp_pct = float(hp_raw)
            except (TypeError, ValueError):
                hp_pct = None
        hp_text = f"%{hp_pct:.0f}" if hp_pct is not None else "bilinmiyor"
        route_name = str(stats.get("route_name", stats.get("active_route", "-")))
        wp_idx = stats.get("waypoint_index", "-")
        uptime = str(stats.get("uptime_str", stats.get("uptime", "00:00:00")))

        lines = [
            f"Durum: {state} | HP: {hp_text}",
            f"Toplama: {gathers} adet ({rate_hr:.1f}/saat)",
            f"Rota: {route_name} (WP #{wp_idx}) | Süre: {uptime}",
        ]
        if public_url:
            lines.append(f"Canlı Panel: {public_url}")
        # Komut kanali (command_topic) telemetri mesajinda duyurulmaz:
        # konu adini bilen herkes imzasiz komut deneyebilirdi.

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
            # Basarili yoklama (yeni komut olmasa bile): geri cekilmeyi sifirla.
            with self._lock:
                self._consecutive_poll_errors = 0
                self._current_poll_interval_sec = self.poll_interval_sec

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
                    if msg_time > self._last_poll_timestamp:
                        self._last_poll_timestamp = msg_time

                parsed = self.parse_remote_command(msg_text)
                if parsed is not None:
                    cmd, meta = parsed
                    # Dedup (P0-4 / K-09): cid tasiyan imzali komutlarda
                    # dedup komut id'siyledir — yeni ntfy mesaj id'siyle
                    # yeniden gonderilen ayni komut, verify icindeki
                    # gorulmus-cid kontrolunde reddedilir. cid tasimayan
                    # (imzasiz/legacy) yuklerde dedup eskisi gibi mesaj
                    # id'si / (zaman + icerik) karmasiyla yapilir.
                    if not (isinstance(meta.get("cid"), str) and meta.get("cid")):
                        dedup_key = msg_id or hashlib.sha256(
                            f"{msg_time}:{msg_text}".encode("utf-8", errors="ignore")
                        ).hexdigest()[:16]
                        if not self._note_message_id(dedup_key):
                            continue
                    # Imzasiz/token'suz uzaktan komut kabul edilmez: yalnizca
                    # gecerli HMAC imzali komutlar dagitilir. verify; ts/cid
                    # eksik ya da suresi dolmus ya da cid tekrar ise reddeder.
                    if self.require_signed_commands and not self.verify_command_signature(
                        cmd, meta.get("sig"), ts=meta.get("ts"),
                        command_id=meta.get("cid"),
                    ):
                        with self._lock:
                            self._rejected_commands += 1
                            self._last_error = (
                                f"Rejected unsigned remote command: {cmd}"
                            )
                        # Reddedilen yukun `since` ortusmesinde (2 sn)
                        # tekrar sayilmamasi icin mesaj id'sini isaretle;
                        # bu, imzali komutlarda dedup degildir — dedup
                        # cid ile verify icinde yapilir.
                        self._note_message_id(
                            msg_id or hashlib.sha256(
                                f"{msg_time}:{msg_text}".encode("utf-8", errors="ignore")
                            ).hexdigest()[:16]
                        )
                        continue
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
                # Cevrimdisi/hatali iken yoklama araligini katla (tavan 30 sn):
                # ntfy.sh'a sabit 3.5 sn aralikla vurmak yerine hata suresince
                # istek sayisi azalir; basarili yoklamada aralik tabana doner.
                self._consecutive_poll_errors += 1
                self._current_poll_interval_sec = min(
                    _POLL_BACKOFF_CAP_SEC,
                    self.poll_interval_sec * (2 ** self._consecutive_poll_errors),
                )

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
            self._consecutive_poll_errors = 0
            self._current_poll_interval_sec = self.poll_interval_sec
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
            # Hata suresince aralik katlanarak artar (tavan 30 sn); basarida
            # poll_commands_once araligi poll_interval_sec tabanina dondurur.
            self._stop_event.wait(self._current_poll_interval_sec)

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
                "require_signed_commands": self.require_signed_commands,
                "rejected_commands": self._rejected_commands,
                "signed_command_max_age_sec": self.signed_command_max_age_sec,
                "seen_command_ids": len(self._seen_command_ids),
                "poll_interval_sec": self._current_poll_interval_sec,
                "consecutive_poll_errors": self._consecutive_poll_errors,
                "last_error": self._last_error,
            }

    def get_mobile_tunnel_ready_flag(self) -> dict:
        # Kosulsuz True YOK (S-2001): hazirlik, relay'in gercek durumuna
        # baglidir — etkin + konu yapilandirilmis olmali.
        ready = bool(self.enabled and self.topic)
        return {
            "module": "mobile_tunnel",
            "ready": ready,
            "enabled": self.enabled,
            "topic_configured": bool(self.topic),
        }

    def get_mobile_tunnel_version_info(self) -> dict:
        return {
            "module": "mobile_tunnel",
            "version": __version__,
            "ready": bool(self.enabled and self.topic),
        }

    def get_mobile_tunnel_diagnostic_snapshot(self) -> dict:
        # Kosulsuz "ok"/True YOK (S-2001): durum canli yoklamadan cikar.
        with self._lock:
            listener_running = bool(
                self._listener_thread is not None and self._listener_thread.is_alive()
            )
            last_error = self._last_error
            published_count = self._published_count
        ready = bool(self.enabled and self.topic)
        if not ready:
            status = "disabled" if not self.enabled else "unconfigured"
        elif last_error:
            status = "degraded"
        else:
            status = "ok"
        return {
            "module": "mobile_tunnel",
            "status": status,
            "ready": ready,
            "listener_running": listener_running,
            "published_count": published_count,
            "last_error": last_error,
        }



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
        # Guvenlik token'i yalnizca cagiran (mobile_server) tarafindan
        # belirlenirse token korumasi var sayilir; varsayilan korumasizdir.
        self.security_token: Optional[str] = None

        self._lock = threading.RLock()
        self._proc: Optional[subprocess.Popen] = None
        self._reader_threads: List[threading.Thread] = []
        self._url_found_event = threading.Event()
        self._url_ready_fired: bool = False
        # P2-8 (K-17): her tünel denemesi nesli artırır; eski denemenin
        # okuyucu thread'i bayat nesille paylaşımlı duruma yazamaz —
        # yoksa eski tünelin satırı yeni denemeyi "başarılı" sanıp
        # çalışan tüneli öldürtür.
        self._tunnel_generation: int = 0
        self._logs: List[str] = []
        self._last_error: str = ""
        self._is_downloading: bool = False
        self._download_progress: float = 0.0

    def set_security_token(self, token: Optional[str]) -> None:
        """Mobil panel token'ini kaydet (token_protected raporu buna dayanir)."""
        with self._lock:
            self.security_token = str(token).strip() if token else None

    def set_remote_command_handler(
        self, on_command: Optional[Callable[[str, Dict[str, Any]], None]]
    ) -> None:
        """ntfy uzaktan komut callback'ini bagla (uretimde main.py cagrisi gerekir)."""
        self.ntfy_relay.set_remote_command_handler(on_command)

    def regenerate_ntfy_topic(self) -> str:
        """ntfy konusunu yenile (eski zayif konularin goc yolu).

        Herkese acik rotasyon API'si: mobil baglanti diyalogundaki
        rotasyon butonu bu metoda baglanir.
        """
        return self.ntfy_relay.regenerate_topic()

    def _append_log(self, line: str) -> None:
        clean = line.strip()
        if not clean:
            return
        with self._lock:
            self._logs.append(f"[{time.strftime('%H:%M:%S')}] {clean}")
            if len(self._logs) > 120:
                self._logs = self._logs[-120:]

    def find_cloudflared_binary(self) -> Optional[str]:
        """Check if `cloudflared.exe` exists in `tools/cloudflared.exe` or system `PATH`.

        Her aday _verify_cloudflared_binary ile dogrulanir (S-0504):
        dogrulama basarisiz olan ikili (degistirilmis/eski/bozuk) dondurulmez;
        siradaki adaya gecilir, hicbiri gecemezse None doner ve
        ensure_cloudflared_binary taze indirmeye duser.
        """
        if CLOUDFLARED_EXE_PATH.exists() and CLOUDFLARED_EXE_PATH.is_file():
            try:
                if CLOUDFLARED_EXE_PATH.stat().st_size > 1_000_000:
                    if self._verify_cloudflared_binary(CLOUDFLARED_EXE_PATH):
                        return str(CLOUDFLARED_EXE_PATH)
                    self._append_log(
                        f"cloudflared dogrulama basarisiz, yoksayiliyor: {CLOUDFLARED_EXE_PATH}"
                    )
            except OSError as exc:
                self._append_log(f"cloudflared stat check warning: {exc}")

        which_path = shutil.which("cloudflared") or shutil.which("cloudflared.exe")
        if which_path:
            if self._verify_cloudflared_binary(Path(which_path)):
                return which_path
            self._append_log(
                f"cloudflared dogrulama basarisiz, yoksayiliyor: {which_path}"
            )

        return None

    def _verify_cloudflared_binary(self, path: Path, expected_sha256: Optional[str] = None) -> bool:
        """cloudflared ikilisini dogrula: boyut + calistirilabilir imza + `--version` calisir.

        O-4 (dilim 22): GitHub 'latest' indirmesinde sabit bir hash
        yayinlanmadigi icin eskiden yalnızca calistirilabilirlik kaniti
        araniyordu. Artık indirilen ikilinin SHA-256'sı her zaman
        hesaplanıp loglanır (operatör `config/cloudflared.sha256`
        dosyasına yazarak pin'leyebilir); pin dosyası varsa uyumsuz
        hash reddedilir (fail-closed). Pin yoksa doğrulama eski
        seviyede kalır — bu durum logda açıkça uyarı olarak yazılır.
        Windows PE (MZ) ve Linux ELF ikilileri kabul edilir.
        """
        try:
            if not path.exists() or path.stat().st_size <= 1_000_000:
                return False
            with open(path, "rb") as fh:
                header = fh.read(4)
                if header[:2] != b"MZ" and header != b"\x7fELF":
                    return False
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            self._append_log(f"cloudflared SHA-256: {actual_sha256}")
            if expected_sha256:
                expected = str(expected_sha256).strip().lower().split()[0]
                if not expected or not hmac.compare_digest(actual_sha256, expected):
                    self._append_log(
                        "cloudflared SHA-256 pin uyumsuzluğu — ikili reddedildi "
                        f"(beklenen {expected[:16]}..., alınan {actual_sha256[:16]}...)."
                    )
                    return False
            completed = subprocess.run(
                [str(path), "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return completed.returncode == 0
        except Exception as exc:
            self._append_log(f"cloudflared dogrulama hatasi: {exc}")
            return False

    @staticmethod
    def _read_pinned_cloudflared_sha256() -> Optional[str]:
        """`config/cloudflared.sha256` varsa ilk token'ı pin olarak döner."""
        try:
            pin_path = CONFIG_DIR / "cloudflared.sha256"
            if pin_path.is_file():
                text = pin_path.read_text(encoding="utf-8").strip()
                if text:
                    return text.split()[0].lower()
        except Exception:
            pass
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
                # O-4 (dilim 22): pin dosyası varsa hash zorunlu tutulur;
                # yoksa indirilen hash loglanır ki operatör pin'leyebilsin.
                pinned_sha256 = self._read_pinned_cloudflared_sha256()
                if not pinned_sha256:
                    self._append_log(
                        "UYARI: config/cloudflared.sha256 yok — cloudflared "
                        "hash pin'i olmadan indiriliyor (yalnızca boyut + "
                        "imza + --version doğrulaması)."
                    )
                if not self._verify_cloudflared_binary(temp_path, expected_sha256=pinned_sha256):
                    raise RuntimeError(
                        "Downloaded cloudflared binary failed verification "
                        "(invalid executable, --version check failed, or SHA-256 pin mismatch)."
                    )
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
        # P2-8: okuyucu doğduğu andaki nesli yakalar; deneme yenilenirse
        # (yeni nesil) bu okuyucu artık paylaşımlı duruma yazmaz — log
        # satırı yazmaya devam eder, URL/event yazmaz.
        with self._lock:
            generation = self._tunnel_generation

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
                            fire_callback = False
                            with self._lock:
                                if generation != self._tunnel_generation:
                                    continue
                                if not self.public_url:
                                    self.public_url = found_url
                                    self.active_method = method_name
                                    self._last_error = ""
                                    self._url_found_event.set()
                                    # on_url_ready yalnizca ilk URL'de bir kez
                                    # cagrilir; sonraki eslesen satirlar tekrarlamaz.
                                    if not self._url_ready_fired:
                                        self._url_ready_fired = True
                                        fire_callback = True
                            if fire_callback and self.on_url_ready:
                                try:
                                    self.on_url_ready(found_url, method_name)
                                except Exception as cb_exc:
                                    self._append_log(f"on_url_ready callback warning: {cb_exc}")
            except Exception as exc:
                self._append_log(f"[{method_name}] stream ended: {exc}")

        t = threading.Thread(target=_reader, name=f"TunnelReader-{method_name}", daemon=True)
        t.start()
        return t

    def _snapshot_tunnel(self) -> Dict[str, Any]:
        """Çalışan tünelin geri yüklenebilir anlık görüntüsü (P2-8).

        Yenileme denemesi öncesi alınır; deneme başarısız olursa eski
        tünel bu görüntüden geri yüklenir. Canlı tünel yoksa proc/URL
        None döner — davranış o durumda eskisiyle aynıdır.
        """
        with self._lock:
            return {
                "proc": self._proc,
                "public_url": self.public_url,
                "active_method": self.active_method,
                "url_ready_fired": self._url_ready_fired,
            }

    def _snapshot_has_live_tunnel(self, snapshot: Optional[Dict[str, Any]]) -> bool:
        """Anlık görüntüdeki tünel hâlâ canlı mı? (P2-8)

        Deneme saniyeler sürer; eski süreç o arada ölmüş olabilir —
        ölü sürecin bayat URL'si geri yüklenmez, durum temizlenir.
        """
        if not snapshot or not snapshot.get("public_url"):
            return False
        proc = snapshot.get("proc")
        if proc is None:
            return False
        try:
            return proc.poll() is None
        except Exception:
            return False

    def _restore_tunnel_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """`_snapshot_tunnel` görüntüsünü paylaşımlı duruma geri yazar."""
        with self._lock:
            self._proc = snapshot.get("proc")
            self.public_url = snapshot.get("public_url")
            self.active_method = snapshot.get("active_method") or "none"
            self._url_ready_fired = bool(snapshot.get("url_ready_fired"))
            if self.public_url:
                self._url_found_event.set()
            else:
                self._url_found_event.clear()

    def _terminate_proc(self, proc: Any) -> None:
        """Tek bir alt süreci kibarca durdur (paylaşımlı duruma dokunmaz).

        Kilit tutulmazken çağrılır (deadlock yok): terminate → 3 sn bekle
        → gerekirse kill. `poll()` ölü ise no-op. Fake/eksik `kill`
        olan süreçlerde (test double) kill adımı atlanır.
        """
        if proc is None:
            return
        try:
            if proc.poll() is not None:
                return
        except Exception:
            return
        try:
            proc.terminate()
        except Exception as exc:
            self._append_log(f"Tunnel process terminate warning: {exc}")
            return
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            kill = getattr(proc, "kill", None)
            if callable(kill):
                try:
                    kill()
                except Exception as exc:
                    self._append_log(f"Tunnel process kill warning: {exc}")
                    return
                try:
                    proc.wait(timeout=3.0)
                except Exception:
                    pass
        except Exception as exc:
            self._append_log(f"Tunnel process wait warning: {exc}")

    def _run_tunnel_process(
        self,
        cmd: List[str],
        pattern: re.Pattern[str],
        *,
        starting_label: str,
        reader_name: str,
        start_log: str,
        timeout_sec: float,
        timeout_message: str,
        error_prefix: str,
        previous: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Tünel alt sürecini başlatıp stdout/stderr'de public URL bekleyen
        ortak gövde (start_cloudflare_tunnel / start_ssh_tunnel).

        P2-8 (K-17): eski tünel deneme ÖNCESİ öldürülmez. `previous`
        (`_snapshot_tunnel` çıktısı) verilen canlı tünel, deneme
        başarısız olursa (zaman aşımı / Popen hatası / ikili yok)
        aynen geri yüklenir ve hata loglanır; yalnızca deneme
        BAŞARILI olursa eski süreç durdurulur. `previous` yoksa
        davranış eskisiyle aynıdır (başarısızlıkta durum temizlenir).
        """
        creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000) if os.name == "nt" else 0
        new_proc: Any = None

        try:
            with self._lock:
                self._tunnel_generation += 1
                self.public_url = None
                self.active_method = starting_label
                self._url_found_event.clear()
                self._url_ready_fired = False

            self._append_log(start_log)
            new_proc = subprocess.Popen(
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
                self._proc = new_proc
                self._reader_threads = []
                if new_proc.stdout:
                    self._reader_threads.append(
                        self._spawn_stream_reader(new_proc.stdout, pattern, reader_name)
                    )
                if new_proc.stderr:
                    self._reader_threads.append(
                        self._spawn_stream_reader(new_proc.stderr, pattern, reader_name)
                    )

            if self._url_found_event.wait(timeout=timeout_sec):
                # Başarı: artık eski tünel süreci durdurulur (yenisi
                # URL'yi aldı; iki tünel aynı yerel porta baksa da tek
                # aktif tünel kalır).
                prev_proc = previous.get("proc") if previous else None
                if prev_proc is not None and prev_proc is not new_proc:
                    self._terminate_proc(prev_proc)
                return self.public_url

            with self._lock:
                self._last_error = timeout_message
            self._terminate_proc(new_proc)
            if self._snapshot_has_live_tunnel(previous):
                self._restore_tunnel_snapshot(previous)
                self._append_log(
                    "Tünel yenileme zaman aşımına uğradı; çalışan eski tünel korunuyor."
                )
            else:
                self._stop_tunnel_process()
            return None
        except Exception as exc:
            with self._lock:
                self._last_error = f"{error_prefix}: {exc}"
            if new_proc is not None:
                self._terminate_proc(new_proc)
            if self._snapshot_has_live_tunnel(previous):
                self._restore_tunnel_snapshot(previous)
                self._append_log(
                    "Tünel yenileme hatası; çalışan eski tünel korunuyor: "
                    f"{self._last_error}"
                )
            else:
                self._stop_tunnel_process()
            return None

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

        P2-8: çalışan tünel deneme öncesi öldürülmez — önce anlık
        görüntü alınır, yeni tünel kurulamazsa eski tünel geri yüklenir.
        """
        # P2-8: anlık görüntü İLK alınır; ikili bulunamazsa bile eski
        # tünel yaşar (eski kod burada `_stop_tunnel_process` çağırıp
        # canlı tüneli yok ediyordu).
        previous = self._snapshot_tunnel()
        port = int(local_port or self.local_port)
        self.local_port = port

        binary = self.find_cloudflared_binary()
        if not binary and auto_download:
            binary = self.ensure_cloudflared_binary()

        if not binary:
            with self._lock:
                self._last_error = "cloudflared.exe bulunamadı ve indirilemedi."
            if self._snapshot_has_live_tunnel(previous):
                self._append_log(
                    "Cloudflare yenileme: ikili bulunamadı; çalışan eski tünel korunuyor."
                )
            return None

        cmd = [
            binary,
            "tunnel",
            "--url",
            f"http://127.0.0.1:{port}",
            "--no-autoupdate",
            "--protocol",
            "http2",
        ]
        return self._run_tunnel_process(
            cmd,
            TRYCLOUDFLARE_URL_RE,
            starting_label="starting_cloudflare",
            reader_name="cloudflare",
            start_log=f"Starting Cloudflare Quick Tunnel on port {port}",
            timeout_sec=timeout_sec,
            timeout_message="Cloudflare tüneli zaman aşımına uğradı (URL alınamadı).",
            error_prefix="Cloudflare başlatma hatası",
            previous=previous,
        )

    def start_ssh_tunnel(
        self,
        local_port: Optional[int] = None,
        provider: str = "localhost.run",
        timeout_sec: float = 20.0,
    ) -> Optional[str]:
        """
        Method 2 — SSH-based Free Tunnel Fallback (`localhost.run` / `pinggy.io`).
        Uses Windows built-in `ssh.exe -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=<config>/ssh_known_hosts -R 80:127.0.0.1:<port> nokey@localhost.run`
        to obtain a free `https://*.lhr.life` public URL without installing anything!
        Host keys are stored persistently (TOFU): the first connection records
        the server key in `config/ssh_known_hosts`; later connections verify
        against it. `/dev/null` (or `NUL` on Windows) is never used here —
        with it the presented key would be accepted unconditionally on every
        connection and a changed key (MITM) would go unnoticed.

        P2-8: çalışan tünel deneme öncesi öldürülmez (bkz.
        `start_cloudflare_tunnel`); başarısız denemede eski tünel korunur.
        """
        previous = self._snapshot_tunnel()
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
            if self._snapshot_has_live_tunnel(previous):
                self._append_log(
                    "SSH yenileme: ssh bulunamadı; çalışan eski tünel korunuyor."
                )
            return None

        # Kalıcı TOFU: sunucu anahtarı config/ssh_known_hosts dosyasına
        # yazılır; accept-new ilk bağlantıda kaydeder, sonrakilerde doğrular.
        known_hosts = CONFIG_DIR / "ssh_known_hosts"
        try:
            known_hosts.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        known_hosts_opt = f"UserKnownHostsFile={known_hosts.as_posix()}"

        if provider == "pinggy.io":
            cmd = [
                ssh_bin,
                "-p",
                "443",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                known_hosts_opt,
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
                "StrictHostKeyChecking=accept-new",
                "-o",
                known_hosts_opt,
                "-o",
                "ServerAliveInterval=30",
                "-o",
                "ExitOnForwardFailure=yes",
                "-R",
                f"80:127.0.0.1:{port}",
                "nokey@localhost.run",
            ]

        return self._run_tunnel_process(
            cmd,
            SSH_TUNNEL_URL_RE,
            starting_label=f"starting_ssh_{provider}",
            reader_name="ssh_localhost_run",
            start_log=f"Starting SSH Reverse Tunnel ({provider}) on port {port}",
            timeout_sec=timeout_sec,
            timeout_message=f"SSH tüneli ({provider}) zaman aşımına uğradı.",
            error_prefix="SSH tünel hatası",
            previous=previous,
        )

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

        # Hata donusu yoksayilmaz: cagiran basari/hata durumunu acikca gorur.
        status = self.get_status()
        status["tunnel_started"] = bool(url)
        if not url:
            status["tunnel_error"] = self._last_error or "Tünel kurulamadı."
        return status

    def start_tunnel(
        self,
        port: Optional[int] = None,
        auto_download_cloudflared: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Convenience alias for start_best_free_tunnel."""
        return self.start_best_free_tunnel(
            local_port=port,
            auto_download_cloudflared=auto_download_cloudflared,
            **kwargs,
        )

    def is_tunnel_running(self) -> bool:
        with self._lock:
            return bool(self._proc is not None and self._proc.poll() is None and self.public_url)

    def check_tunnel_alive(self) -> bool:
        """Y-6 (dilim 22): tünel süreci ölmüşse bayat `public_url`
        temizlenir — `is_tunnel_running()` False der ama eski URL
        `public_url` / dialog / `mobile_url.txt`'de kalırdı. Ölü süreç
        + bayat URL durumunda False döner ve URL sıfırlanır; canlı
        tünelde True döner. Çağıran (main.py izleyicisi) False'da
        yeniden başlatır."""
        with self._lock:
            proc = self._proc
            if proc is None:
                return False
            if proc.poll() is None:
                return bool(self.public_url)
            if self.public_url:
                self._append_log("Tunnel process died; clearing stale public URL.")
                self.public_url = None
                self.active_method = "none"
                self._url_found_event.clear()
                self._url_ready_fired = False
            return False

    def _stop_tunnel_process(self) -> None:
        """Yalnizca tünel alt sürecini durdur (ntfy dinleyicisine dokunmaz).

        start_*_tunnel metotlari P2-8'den beri baslangicta bunu
        cagirmaz (canlı tünel deneme öncesi öldürülmez); yalnızca
        açık kapatma (stop_tunnel/shutdown) ve tünelsiz başarısız
        deneme sonrası durum temizliği için kullanılır.
        """
        with self._lock:
            proc = self._proc
            self._proc = None
            self.public_url = None
            self.active_method = "none"
            self._url_found_event.clear()
            self._url_ready_fired = False

        if proc is not None:
            self._terminate_proc(proc)
            self._append_log("Tunnel process stopped.")

    def stop_tunnel(self) -> None:
        """Aktif tüneli ve ntfy komut dinleyicisini durdur."""
        self._stop_tunnel_process()
        self.ntfy_relay.stop_command_listener()

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

            # Binary aramasi (PATH tarama + stat, ~1 ms) cagri basina bir kez:
            # once installed ve path icin ayri ayri cagriliyordu.
            cloudflared_binary = self.find_cloudflared_binary()
            return {
                "running": self.is_tunnel_running(),
                "active_method": self.active_method,
                "local_port": self.local_port,
                "lan_ip": get_lan_ip(),
                "lan_url": lan_url,
                "public_url": pub_url,
                "public_url_with_token": pub_url_with_token,
                "cloudflared_installed": cloudflared_binary is not None,
                "cloudflared_path": cloudflared_binary or str(CLOUDFLARED_EXE_PATH),
                "is_downloading": self._is_downloading,
                "download_progress": round(self._download_progress, 1),
                "last_error": self._last_error,
                "logs": list(self._logs[-25:]),
                "ntfy": self.ntfy_relay.get_status(),
            }

    def get_mobile_tunnel_ready_flag(self) -> dict:
        # Kosulsuz True YOK (S-2001): hazirlik canli duruma baglidir —
        # tunel calisiyor olmali ya da ntfy relay'i yapilandirilmis olmali.
        running = self.is_tunnel_running()
        relay_ready = bool(self.ntfy_relay.enabled and self.ntfy_relay.topic)
        return {
            "module": "mobile_tunnel",
            "ready": bool(running or relay_ready),
            "tunnel_running": running,
            "active_method": self.active_method,
        }

    def get_mobile_tunnel_version_info(self) -> dict:
        return {
            "module": "mobile_tunnel",
            "version": __version__,
            "ready": bool(self.get_mobile_tunnel_ready_flag().get("ready")),
        }

    def get_mobile_tunnel_diagnostic_snapshot(self) -> dict:
        # Kosulsuz "ok"/True YOK (S-2001): durum canli yoklamadan cikar.
        running = self.is_tunnel_running()
        relay_ready = bool(self.ntfy_relay.enabled and self.ntfy_relay.topic)
        if running:
            status = "ok"
        elif relay_ready:
            status = "relay_only"
        else:
            status = "down"
        return {
            "module": "mobile_tunnel",
            "status": status,
            "ready": bool(running or relay_ready),
            "tunnel_running": running,
            "active_method": self.active_method,
            "ntfy": self.ntfy_relay.get_status(),
        }



def generate_qr_image(url: str, size: int = 260, border: int = 2) -> np.ndarray:
    """
    Generate a crisp BGR QR Code image (`np.ndarray` of shape `(size, size, 3)`, `uint8`)
    that PyQt6 (`QImage` / `QPixmap`) or OpenCV can display immediately without
    requiring extra pip packages.

    Priority:
      1. `qrcode` package (if installed in environment)
      2. `cv2.QRCodeEncoder_create()` (built into OpenCV!)
    Raises RuntimeError if no real QR encoder is available instead of
    returning a fake, unscannable matrix.
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

    # 3. Sahte matris YOK: hash ile doldurulmus bir matris QR degildir,
    # taranamaz ve "basari" diye gosterilemez. Gercek uretici yoksa hata ver.
    if qr_gray is None:
        raise RuntimeError(
            "QR code cannot be generated: neither the 'qrcode' package nor "
            "cv2.QRCodeEncoder_create is available in this environment."
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
        # Kosulsuz True YOK (S-2001): saglik, manager'in canli durumundan
        # cikar; tunel kapali ve relay yapilandirilmamissa hazir degildir.
        try:
            if self.manager is None:
                return {"mobile_tunnel_ready": False}
            running = self.manager.is_tunnel_running()
            relay = getattr(self.manager, "ntfy_relay", None)
            relay_ready = bool(
                relay is not None and relay.enabled and relay.topic
            )
            return {
                "mobile_tunnel_ready": bool(running or relay_ready),
                "tunnel_running": running,
                "active_method": self.manager.active_method,
            }
        except Exception:
            return {"mobile_tunnel_ready": False}

    def get_tunnel_security_status(self) -> dict:
        is_running = False
        if self.manager:
            with self.manager._lock:
                is_running = bool(self.manager.public_url and self.manager._proc and self.manager._proc.poll() is None)
        token_protected = bool(
            self.manager is not None and getattr(self.manager, "security_token", None)
        )
        return {"is_running": is_running, "token_protected": token_protected}

    def get_mobile_tunnel_ready_flag(self) -> dict:
        # Kosulsuz True YOK (S-2001): manager'in canli hazirlik bayragi.
        if self.manager is None:
            return {"module": "mobile_tunnel", "ready": False}
        return self.manager.get_mobile_tunnel_ready_flag()

    def get_mobile_tunnel_version_info(self) -> dict:
        if self.manager is None:
            return {"module": "mobile_tunnel", "version": __version__, "ready": False}
        return self.manager.get_mobile_tunnel_version_info()

    def get_mobile_tunnel_diagnostic_snapshot(self) -> dict:
        # Kosulsuz "ok"/True YOK (S-2001): manager'in canli tani anlik goruntusu.
        if self.manager is None:
            return {"module": "mobile_tunnel", "status": "down", "ready": False}
        return self.manager.get_mobile_tunnel_diagnostic_snapshot()


MobileTunnel = FreeCloudTunnelManager
ZeroConfigMobileTunnel = FreeCloudTunnelManager



def _add_security_status_method():
    def get_tunnel_security_status(self) -> dict:
        with self._lock:
            is_running = bool(self.public_url and self._proc and self._proc.poll() is None)
            token_protected = bool(self.security_token)
        return {"is_running": is_running, "token_protected": token_protected}
    FreeCloudTunnelManager.get_tunnel_security_status = get_tunnel_security_status

_add_security_status_method()


def get_mobile_tunnel_capabilities() -> list:
    return ["mobile_tunnel", "ntfy_relay", "diagnostic_snapshot"]


def _probe_live_tunnel_state(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> Optional[dict]:
    """Canli tunel/relay durumunu yokla (S-2001).

    `manager` verilmediyse None doner: canli durum yoklanmadi, bilinmiyor.
    Bu, statik sozlesme kontrolunun (validate_mobile_tunnel_contract)
    canli basari gibi sunulmasini engeller — yoklanmamis durum True degildir.
    """
    if manager is None:
        return None
    try:
        snap = manager.get_mobile_tunnel_diagnostic_snapshot()
        return {
            "ready": bool(snap.get("ready")),
            "status": snap.get("status"),
            "tunnel_running": bool(snap.get("tunnel_running")),
        }
    except Exception:
        return {"ready": False, "status": "probe_failed", "tunnel_running": False}


def _live_audit_verdict(manager: Optional["FreeCloudTunnelManager"] = None) -> dict:
    """Sozlesme gecerliligi + canli durum birlesik karari (S-2001).

    audit_passed/certified/ready yalnizca sozlesme gecerli VE canli tunel
    ya da relay hazirsa True olur. Canli ornek verilmediyse (manager=None)
    durum yoklanmamistir: fail-closed — False + "not_probed" gerekcesi.
    """
    contract_valid = validate_mobile_tunnel_contract()
    live = _probe_live_tunnel_state(manager)
    if live is None:
        return {
            "passed": False,
            "contract_valid": contract_valid,
            "live": "not_probed",
            "reason": "no live tunnel instance probed; contract check alone does not prove a running tunnel",
        }
    passed = bool(contract_valid and live.get("ready"))
    return {
        "passed": passed,
        "contract_valid": contract_valid,
        "live": live,
        "reason": "" if passed else "live tunnel/relay not ready",
    }


def validate_mobile_tunnel_contract() -> bool:
    """Real probe (not a tautology): every declared capability must map to
    a real, present component — no instantiation and no I/O, so this is safe
    to call from audit/compliance helpers.

    - "mobile_tunnel": FreeCloudTunnelManager exposes the real tunnel
      entry points (Cloudflare / SSH / best-free starter).
    - "ntfy_relay": NtfyCloudRelay exposes publish + poll entry points.
    - "diagnostic_snapshot": TunnelBridge exposes the diagnostic
      snapshot method (checked on the class, not an instance — building
      one would create a manager/relay and touch config files).
    Returns False if any capability is missing or any entry point was
    removed/renamed, instead of always returning True for a non-empty
    static list.

    NOT: bu yalnizca statik bir yetenek/sozlesme kontroludur; canli bir
    tunelin/relay'in calistigini KANITLAMAZ (S-2001). audit_passed /
    certified / ready kararlari icin `_live_audit_verdict(manager)` kullanilir;
    yalnizca bu fonksiyonun sonucu canli basari gibi raporlanamaz.
    """
    try:
        caps = get_mobile_tunnel_capabilities()
        if not caps:
            return False
        if not {"mobile_tunnel", "ntfy_relay", "diagnostic_snapshot"}.issubset(set(caps)):
            return False
        if not all(
            callable(getattr(FreeCloudTunnelManager, name, None))
            for name in (
                "start_cloudflare_tunnel",
                "start_ssh_tunnel",
                "start_best_free_tunnel",
            )
        ):
            return False
        if not all(
            callable(getattr(NtfyCloudRelay, name, None))
            for name in ("publish_message", "poll_commands_once")
        ):
            return False
        if not callable(getattr(TunnelBridge, "get_mobile_tunnel_diagnostic_snapshot", None)):
            return False
        return True
    except Exception:
        return False


def get_mobile_tunnel_baseline_metrics(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    live = _probe_live_tunnel_state(manager)
    return {
        "module": "mobile_tunnel",
        "contract_valid": validate_mobile_tunnel_contract(),
        "capability_count": len(get_mobile_tunnel_capabilities()),
        "live": live if live is not None else "not_probed",
    }


def get_mobile_tunnel_readiness_status(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    # Kosulsuz True YOK (S-2001): hazirlik = sozlesme gecerli VE canli hazir.
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "ready": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "metrics": get_mobile_tunnel_baseline_metrics(manager),
    }


def get_mobile_tunnel_audit_record(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "audit_passed": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "readiness": get_mobile_tunnel_readiness_status(manager),
    }


def get_mobile_tunnel_operational_digest(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "operational": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "audit": get_mobile_tunnel_audit_record(manager),
    }


def get_mobile_tunnel_telemetry_payload(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    return {
        "module": "mobile_tunnel",
        "payload": get_mobile_tunnel_operational_digest(manager),
    }


def get_mobile_tunnel_integrity_seal(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "sealed": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "telemetry": get_mobile_tunnel_telemetry_payload(manager),
    }


def get_mobile_tunnel_compliance_certificate(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    # Kosulsuz certified YOK (S-2001): sertifika, canli tunel/relay
    # yoklanmadan verilmez; yoklanmamis durum fail-closed False'dur.
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "certified": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "seal": get_mobile_tunnel_integrity_seal(manager),
    }


def get_mobile_tunnel_readiness_probe(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "ready": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "certificate": get_mobile_tunnel_compliance_certificate(manager),
    }


def get_mobile_tunnel_audit_summary(
    manager: Optional["FreeCloudTunnelManager"] = None,
) -> dict:
    # Kosulsuz audit_passed YOK (S-2001): tunel kapaliyken ya da canli
    # durum yoklanmamissa audit gecmis sayilmaz.
    verdict = _live_audit_verdict(manager)
    return {
        "module": "mobile_tunnel",
        "audit_passed": verdict["passed"],
        "contract_valid": verdict["contract_valid"],
        "live": verdict["live"],
        "reason": verdict["reason"],
        "probe": get_mobile_tunnel_readiness_probe(manager),
    }





def _tunnel_cli() -> int:
    """Y-5 (dilim 22): imzalı ntfy komut kanalı için repo-içi istemci.

    `python -m core.mobile_tunnel sign pause`  -> {"cmd", "ts", "cid",
    "sig"} yazar (P0-4: ts unix saniye, cid tek-kullanimlik komut id'si;
    sig, "cmd\n<ts>\n<cid>" uzerinden HMAC-SHA256'dir)
    `python -m core.mobile_tunnel send pause`  -> imzalı komutu
    `<topic>_cmd` kanalına gönderir (bot `poll_commands_once` ile
    HMAC doğrulamasıyla alır; ts penceresi ±300 sn, cid tekrarina
    replay reddi uygulanir). İzinli komutlar: pause, resume, f8,
    skip, panic, stop, start, status, reset_stuck.
    """
    import argparse

    ap = argparse.ArgumentParser(
        description="Gatherion imzalı ntfy uzaktan komut istemcisi (dilim 22 Y-5)"
    )
    sub = ap.add_subparsers(dest="command", required=True)
    p_sign = sub.add_parser("sign", help="İmzalı komut yükünü yazdır (göndermez)")
    p_sign.add_argument("cmd", help="Komut (pause/resume/f8/skip/panic/stop/start/status/reset_stuck)")
    p_send = sub.add_parser("send", help="İmzalı komutu ntfy _cmd kanalına gönder")
    p_send.add_argument("cmd", help="Komut (pause/resume/f8/skip/panic/stop/start/status/reset_stuck)")
    args = ap.parse_args()

    mgr = FreeCloudTunnelManager()
    relay = mgr.ntfy_relay
    clean = str(args.cmd or "").strip().lower()
    if clean not in ALLOWED_REMOTE_COMMANDS:
        print(f"Reddedildi: '{clean}' izinli komut değil: {sorted(ALLOWED_REMOTE_COMMANDS)}")
        return 2
    if args.command == "sign":
        print(json.dumps(relay.sign_command(clean), ensure_ascii=False))
        return 0
    ok = relay.publish_command(clean)
    print(("Gönderildi: " if ok else "GÖNDERİLEMEDİ: ") + clean)
    if not ok:
        print(f"Son hata: {relay._last_error}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_tunnel_cli())
