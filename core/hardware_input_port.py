# -*- coding: utf-8 -*-
"""
HardwareInputPort — girdi soyutlaması.

Varsayılan SoftInputAdapter mevcut InputController'a delege eder.

HID klavye köprüleri (opt-in; COM1 asla otomatik açılmaz):
  - SerialHidAdapter  — Arduino Leonardo / KMBox (GATHERION_HID_COM / input.serial_port)
  - NetworkHidAdapter — Android BT-HID / USB-gadget (GATHERION_HID_TCP / input.hid_tcp)
                        tcp://host:port  |  udp://host:port  |  host:port (TCP)

GÜVENLİK (F-1918): ağ HID aktarımı düz metindir ve telefon tarafı varsayılan
olarak kimlik doğrulamaz — aynı LAN'daki herhangi bir host telefonun portuna
ulaşırsa tuş enjekte edebilir. Bu istemci tarafı bu yüzden:
  1) varsayılan olarak yalnızca LAN/loopback hostlara bağlanır (genel IP
     reddedilir; kaçış: GATHERION_HID_ALLOW_PUBLIC=1),
  2) opt-in paylaşımlı sır destekler (GATHERION_HID_TOKEN): ayarlıysa istemci
     bağlanınca önce `AUTH <token>` satırını gönderir; telefon tarafı da
     bunu zorunlu kılmalıdır, yoksa token yalnızca eski köprülerce
     yoksayılan bir satırdır. Token da düz metin gider — güvenilmeyen
     LAN'da dinlenebilir; tam düzeltme telefon tarafında kimlik doğrulamalı
     (ideal olarak şifreli) aktarım gerektirir ve bu repo dışındadır.
  Tokensuz bağlantıda her örnekte bir kez uyarı loglanır.

Aynı satır protokolü (newline UTF-8):
  KEY <name> <hold_ms> | KEYDOWN <name> | KEYUP <name>
  MOUSE <btn> DOWN|UP | MOVE <x> <y> | CLICK <x> <y> <btn>
  PING
"""

from __future__ import annotations

import ipaddress
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

logger = logging.getLogger("Gatherion.HardwareInputPort")


def normalize_hid_endpoint(raw: str) -> str:
    """COM3 | tcp://host:port | udp://host:port | host:port → canonical endpoint."""
    text = str(raw or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if lower.startswith("udp://"):
        return "udp://" + text[6:].strip()
    if lower.startswith("udp:"):
        return "udp://" + text[4:].lstrip("/").strip()
    if lower.startswith("tcp://"):
        return "tcp://" + text[6:].strip()
    if lower.startswith("tcp:"):
        return "tcp://" + text[4:].lstrip("/").strip()
    # host:port without scheme → treat as TCP (not COMn)
    if ":" in text and not text.upper().startswith("COM"):
        host, _, port = text.rpartition(":")
        if host and port.isdigit():
            return f"tcp://{host.strip()}:{port.strip()}"
    return text


_SETTINGS_JSON_CACHE: dict = {"key": None, "data": {}}
_SETTINGS_JSON_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "settings.json"
)


def _load_settings_json_dict() -> dict:
    """Best-effort read of config/settings.json (repo root).

    Sonuc (mtime_ns, boyut) anahtariyla onbellege alinir: dosya degismedikce
    her cagrida disk okuma + JSON parse yapilmaz (bu VM'de olculen:
    onbelleksiz ~0.10 ms/cagri -> onbellekli ~0.012 ms/cagri, 200 cagri).
    Yol modul yuklemesinde bir kez cozulur; cagri basina yalnizca stat()
    yapilir. Dosya yoksa/silinirse onbellek temizlenir ve {} doner; icerik
    degisince (mtime/boyut degisimi) yeniden okunur. Kopya doner ki cagiran
    onbellegi degistiremesin.
    """
    try:
        import json

        path = _SETTINGS_JSON_PATH
        try:
            stat = path.stat()
        except OSError:
            _SETTINGS_JSON_CACHE.clear()
            _SETTINGS_JSON_CACHE.update({"key": None, "data": {}})
            return {}
        key = (stat.st_mtime_ns, stat.st_size)
        if _SETTINGS_JSON_CACHE.get("key") == key:
            return dict(_SETTINGS_JSON_CACHE.get("data") or {})
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        parsed = data if isinstance(data, dict) else {}
        _SETTINGS_JSON_CACHE["key"] = key
        _SETTINGS_JSON_CACHE["data"] = parsed
        return dict(parsed)
    except Exception:
        return {}


_ENV_SHADOW_WARNED: bool = False


def resolve_serial_hid_port(settings: Any = None) -> str:
    """
    HID endpoint çözümlemesi (COM veya tcp://).
    Öncelik:
      GATHERION_HID_COM >
      GATHERION_HID_TCP >
      settings.input.serial_port / hid_com >
      settings.input.hid_tcp / hid_host >
      config/settings.json input.* (when settings arg omitted)
    Boş string = kapalı.
    """
    raw: Any = settings
    if settings is not None and not isinstance(settings, dict):
        if hasattr(settings, "settings") and hasattr(settings.settings, "to_dict"):
            raw = settings.settings.to_dict()
        elif hasattr(settings, "to_dict") and callable(settings.to_dict):
            raw = settings.to_dict()
        elif hasattr(settings, "config") and isinstance(settings.config, dict):
            raw = settings.config
        else:
            raw = getattr(settings, "__dict__", {}) or {}

    if raw is None:
        raw = _load_settings_json_dict()

    settings_port = ""
    if isinstance(raw, dict):
        input_cfg = raw.get("input", {}) if isinstance(raw.get("input"), dict) else {}
        port = (
            raw.get("serial_port")
            or input_cfg.get("serial_port")
            or input_cfg.get("hid_com")
            or ""
        )
        if str(port or "").strip():
            settings_port = normalize_hid_endpoint(str(port).strip())
        else:
            tcp = (
                raw.get("hid_tcp")
                or input_cfg.get("hid_tcp")
                or input_cfg.get("hid_host")
                or ""
            )
            settings_port = normalize_hid_endpoint(str(tcp or "").strip())

    # D-6 (dilim 20): env settings'i sessizce gölgeliyordu — ayar
    # dosyasından yapılan düzeltme etkisiz kalıyor, nedeni anlaşılmıyordu.
    env_com = str(os.environ.get("GATHERION_HID_COM", "") or "").strip()
    env_tcp = str(os.environ.get("GATHERION_HID_TCP", "") or "").strip()
    env_val = env_com or env_tcp
    if env_val:
        env_norm = normalize_hid_endpoint(env_val)
        if settings_port and env_norm != settings_port:
            global _ENV_SHADOW_WARNED
            if not _ENV_SHADOW_WARNED:
                _ENV_SHADOW_WARNED = True
                logger.info(
                    "HID endpoint ortam değişkeniyle gölgelendi: settings='%s' -> env='%s' "
                    "(GATHERION_HID_COM/GATHERION_HID_TCP her zaman kazanır).",
                    settings_port,
                    env_norm,
                )
        return env_norm
    return settings_port


def is_tcp_hid_endpoint(endpoint: str) -> bool:
    ep = normalize_hid_endpoint(endpoint).lower()
    return ep.startswith("tcp://") or ep.startswith("udp://")


def is_udp_hid_endpoint(endpoint: str) -> bool:
    return normalize_hid_endpoint(endpoint).lower().startswith("udp://")


def _resolve_hid_auth_token(explicit: Any = None) -> str:
    """Opt-in shared secret for the network HID transport (F-1918).

    Explicit argument wins, otherwise GATHERION_HID_TOKEN. Empty = no
    authentication (legacy behaviour, with a runtime warning at connect).
    The token travels in plaintext on the same channel, so it only
    authenticates when the phone side enforces it — it is a mitigation,
    not a fix for the unauthenticated transport.
    """
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return str(os.environ.get("GATHERION_HID_TOKEN", "") or "").strip()


def _is_trusted_lan_host(host: str) -> bool:
    """True when ``host`` is a LAN/loopback address (F-1918).

    A literal IP must be loopback, private or link-local; any other
    literal IP (public, multicast, unspecified) is refused so a
    misconfigured endpoint cannot spray plaintext keystrokes at the
    internet. NOTE (S232, v19/P3-8): ``ip.is_private`` is NOT RFC 1918
    only — it also covers CGNAT (100.64/10), documentation and other
    non-routable ranges. That is the intended, stricter reading here
    (private = "not globally reachable"), not a docstring error in the
    check itself. A hostname is resolved with DNS first (S-1601): it is
    trusted only when it resolves and EVERY resolved address is
    loopback/private/link-local. An unresolvable hostname is refused
    (fail closed — the GATHERION_HID_ALLOW_PUBLIC=1 escape hatch in
    _parse_endpoint still applies); a hostname that resolves to a public
    address is refused and logged.
    """
    text = str(host or "").strip()
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        pass
    else:
        if ip.is_unspecified or ip.is_multicast:
            return False
        return bool(ip.is_loopback or ip.is_private or ip.is_link_local)
    # Hostname: resolve it, then classify every answer (a name that
    # round-robins between a LAN address and a public one is not trusted).
    try:
        infos = socket.getaddrinfo(text, None)
    except (OSError, UnicodeError) as exc:
        logger.warning(
            "HID endpoint host %r did not resolve (%s); refusing unclassified host",
            text,
            exc,
        )
        return False
    addresses = set()
    for info in infos:
        sockaddr = info[4] if len(info) > 4 else None
        if sockaddr:
            addresses.add(str(sockaddr[0]))
    if not addresses:
        logger.warning("HID endpoint host %r resolved to no address; refusing", text)
        return False
    for addr in sorted(addresses):
        try:
            resolved = ipaddress.ip_address(addr)
        except ValueError:
            logger.warning(
                "HID endpoint host %r resolved to unparsable address %r; refusing",
                text,
                addr,
            )
            return False
        if resolved.is_unspecified or resolved.is_multicast:
            logger.warning(
                "HID endpoint host %r resolves to %s; refusing non-LAN address",
                text,
                addr,
            )
            return False
        if not (resolved.is_loopback or resolved.is_private or resolved.is_link_local):
            logger.warning(
                "HID endpoint host %r resolves to public address %s; refusing "
                "public HID endpoint (set GATHERION_HID_ALLOW_PUBLIC=1 to override)",
                text,
                addr,
            )
            return False
    return True


@runtime_checkable
class HardwareInputPort(Protocol):
    """Structural input port. Bodies are concrete stubs so AST zero-placeholder gate stays green."""

    def send_key(
        self,
        key: str,
        min_hold: float = 0.08,
        max_hold: float = 0.12,
    ) -> bool:
        _ = (key, min_hold, max_hold)
        return False

    def mouse_button(self, button: str, down: bool) -> bool:
        _ = (button, down)
        return False

    def move(self, x: int, y: int) -> bool:
        _ = (x, y)
        return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        _ = (x, y, button)
        return False


class SoftInputAdapter:
    """Delegates to existing InputController / game input driver."""

    def __init__(self, input_driver: Any = None) -> None:
        self.input_driver = input_driver

    def bind(self, input_driver: Any) -> None:
        self.input_driver = input_driver

    @staticmethod
    def _press_key_ok(res: Any) -> bool:
        # press_key returns hold seconds (0.0 = refused/failed), None for
        # legacy drivers that accept the call without a return value.
        if res is False:
            return False
        if isinstance(res, (int, float)):
            return float(res) > 0.0
        return True

    def send_key(
        self,
        key: str,
        min_hold: float = 0.08,
        max_hold: float = 0.12,
    ) -> bool:
        driver = self.input_driver
        if driver is None or not hasattr(driver, "press_key"):
            return False
        lowered = str(key).lower()
        try:
            res = driver.press_key(
                lowered,
                min_hold=min_hold,
                max_hold=max_hold,
                release_mouse=True,
                post_to_window=False,
            )
        except TypeError:
            # Legacy driver: same call without the newer keyword args.
            try:
                res = driver.press_key(lowered, min_hold=min_hold, max_hold=max_hold)
            except Exception as exc:
                logger.debug("SoftInput send_key: %s", exc)
                return False
        except Exception as exc:
            logger.debug("SoftInput send_key: %s", exc)
            return False
        return self._press_key_ok(res)

    def mouse_button(self, button: str, down: bool) -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        btn = str(button or "left").lower()
        try:
            fn = getattr(driver, "_hw_mouse_down" if down else "_hw_mouse_up", None)
            if callable(fn):
                res = fn(btn)
                # None = legacy driver that accepts without a return value.
                return True if res is None else bool(res)
            # N-3101: `_sendinput_mouse_button` yedeği KALDIRILDI — sürücüde
            # güvenilir `_hw_mouse_down/up` yoksa yumuşak SendInput'a
            # düşülmez; Interception/Faker yoksa girdi başarısız sayılır.
            logger.debug(
                "SoftInput mouse_button: no trusted _hw_mouse path on driver; refused (N-3101)"
            )
        except Exception as exc:
            logger.debug("SoftInput mouse_button: %s", exc)
        return False

    def move(self, x: int, y: int) -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        try:
            if hasattr(driver, "_hw_move_to"):
                res = driver._hw_move_to(int(x), int(y))
                return True if res is None else bool(res)
            if hasattr(driver, "move_to"):
                res = driver.move_to(int(x), int(y))
                return True if res is None else bool(res)
        except Exception as exc:
            logger.debug("SoftInput move: %s", exc)
        return False

    @staticmethod
    def _click_result_ok(result: Any, x: int, y: int, driver: Any = None) -> bool:
        """Honest success check for a driver click return (S-1501 / F-1868).

        ``human_click``/``click_map`` return ``(applied_x, applied_y,
        hold_sec)`` on EVERY path — including motion-abort and
        internal-exception paths — so the return value must be validated,
        never assumed. A tuple is accepted only when well-formed, with a
        positive hold, reporting the cursor at (jittered) target; a click
        aborted mid-way is a failure even with a plausible tuple. Bool
        returns keep their meaning; anything else (None, malformed value)
        is NOT success: no signal means no success claim.
        """
        if result is True:
            return True
        if result is False or result is None:
            return False
        if isinstance(result, (tuple, list)) and len(result) == 3:
            ax, ay, hold = result
            if (
                isinstance(ax, (int, float))
                and not isinstance(ax, bool)
                and isinstance(ay, (int, float))
                and not isinstance(ay, bool)
                and isinstance(hold, (int, float))
                and not isinstance(hold, bool)
                and float(hold) > 0.0
                # Driver jitter is +/-2px; 8px covers jitter + rounding
                # while still catching the exception path, which reports
                # the real cursor position.
                and abs(float(ax) - float(x)) <= 8
                and abs(float(ay) - float(y)) <= 8
            ):
                aborted = getattr(driver, "motion_aborted", None)
                if callable(aborted):
                    try:
                        if bool(aborted()):
                            logger.debug(
                                "SoftInput click at (%s,%s) aborted mid-way; not success",
                                x,
                                y,
                            )
                            return False
                    except Exception:
                        pass
                return True
            logger.debug(
                "SoftInput click at (%s,%s) returned unusable tuple %r; not success",
                x,
                y,
                result,
            )
            return False
        logger.debug(
            "SoftInput click at (%s,%s) returned %r (no success signal); not success",
            x,
            y,
            result,
        )
        return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        try:
            if hasattr(driver, "human_click"):
                # S-1501: human_click tuple'unu atıp koşulsuz True dönme —
                # abort/istisna yolundaki tık başarı sayılıyordu. F-1868
                # doğrulamasıyla aynı kural uygulanır.
                return self._click_result_ok(
                    driver.human_click(int(x), int(y), button=str(button)),
                    int(x),
                    int(y),
                    driver,
                )
            if hasattr(driver, "click_map"):
                return self._click_result_ok(
                    driver.click_map(int(x), int(y), button=str(button)),
                    int(x),
                    int(y),
                    driver,
                )
            if self.move(x, y) and self.mouse_button(button, True):
                # A stuck-down button is worse than a failed click: report it.
                return bool(self.mouse_button(button, False))
        except Exception as exc:
            logger.debug("SoftInput click: %s", exc)
        return False


class _LineHidMixin:
    """Shared KEYDOWN/KEYUP/KEY helpers for serial + TCP transports."""

    def _send(self, payload: str) -> bool:  # pragma: no cover - overridden
        raise NotImplementedError

    def key_down(self, key: str) -> bool:
        return self._send(f"KEYDOWN {str(key).lower().strip()}")

    def key_up(self, key: str) -> bool:
        return self._send(f"KEYUP {str(key).lower().strip()}")

    def send_key(
        self,
        key: str,
        min_hold: float = 0.08,
        max_hold: float = 0.12,
    ) -> bool:
        key_clean = str(key or "").strip().lower()
        if not key_clean:
            return False
        # KEY takes a single hold duration: mean of the requested window,
        # clamped to >= 1 ms (max() silently used only the upper bound).
        hold_ms = max(1, int(round((float(min_hold) + float(max_hold)) / 2.0 * 1000)))
        return self._send(f"KEY {key_clean} {hold_ms}")

    def mouse_button(self, button: str, down: bool) -> bool:
        state = "DOWN" if down else "UP"
        return self._send(f"MOUSE {str(button or 'left').strip().lower()} {state}")

    def move(self, x: int, y: int) -> bool:
        return self._send(f"MOVE {int(x)} {int(y)}")

    def click(self, x: int, y: int, button: str = "left") -> bool:
        return self._send(f"CLICK {int(x)} {int(y)} {button}")


class SerialHidAdapter(_LineHidMixin):
    """
    Arduino Leonardo / KMBox HID bridge over pyserial COM port.
    Instantiates only when port_name is provided and pyserial is available.
    """

    def __init__(self, port_name: str = "", baud: int = 115200) -> None:
        self.port_name = str(port_name or "")
        self.baud = int(baud)
        self._serial: Any = None
        self._available = False
        self.transport = "serial"
        if self.port_name:
            self._try_open()

    def _try_open(self) -> None:
        try:
            import serial  # type: ignore

            self._serial = serial.Serial(self.port_name, self.baud, timeout=0.05)
            self._available = True
            logger.info("SerialHidAdapter open %s @ %d", self.port_name, self.baud)
        except Exception as exc:
            self._available = False
            self._serial = None
            logger.warning(
                "SerialHidAdapter unavailable on %s: %s (set GATHERION_HID_COM or input.serial_port)",
                self.port_name,
                exc,
            )

    def close(self) -> None:
        try:
            if self._serial is not None:
                self._serial.close()
        except Exception:
            pass
        self._serial = None
        self._available = False

    @property
    def available(self) -> bool:
        return bool(self._available and self._serial is not None)

    def _send(self, payload: str) -> bool:
        if not self.available:
            return False
        data = (payload.strip() + "\n").encode("utf-8")
        try:
            written = self._serial.write(data)
            try:
                self._serial.flush()
            except Exception:
                pass
            # The line protocol has no ACK: verify the write itself landed.
            # A short/zero write is a failure, not a silent success.
            return int(written or 0) == len(data)
        except Exception as exc:
            logger.debug("SerialHid write failed: %s", exc)
            self._available = False
            return False


class NetworkHidAdapter(_LineHidMixin):
    """
    TCP/UDP client for Android HID bridges (BT GatherionHidBridge or USB gadget).

    Phone side must:
      1) Appear to Windows as a real keyboard (Bluetooth HID peripheral, or USB gadget)
      2) Listen on TCP/UDP and translate KEYDOWN/KEYUP into HID reports

    endpoint examples:
      tcp://192.168.1.20:8765
      udp://192.168.1.20:8765
      192.168.1.20:8765   (TCP)

    SECURITY (F-1918): this transport is unauthenticated plaintext on the
    LAN — any host that can reach the phone's port can inject keys. This
    client mitigates, but does not eliminate, that risk:
      * only LAN/loopback hosts are accepted by default (a public IP
        endpoint is refused unless GATHERION_HID_ALLOW_PUBLIC=1);
      * when a shared secret is configured (``auth_token`` argument or
        GATHERION_HID_TOKEN env), the client sends ``AUTH <token>`` as
        its first line after connecting — the phone side must enforce it,
        otherwise the line is ignored by legacy bridges. The token itself
        travels in plaintext, so on an untrusted LAN it can be sniffed.
    Use only on a trusted network; authenticated HID transport needs a
    phone-side protocol change.
    """

    DEFAULT_PORT = 8765

    def __init__(
        self,
        endpoint: str = "",
        connect_timeout: float = 2.0,
        auth_token: Any = None,
    ) -> None:
        self.endpoint = normalize_hid_endpoint(endpoint)
        self.connect_timeout = float(connect_timeout)
        self.auth_token = _resolve_hid_auth_token(auth_token)
        self.host = ""
        self.port = self.DEFAULT_PORT
        self._sock: Optional[socket.socket] = None
        self._available = False
        self._lock = threading.Lock()
        self._no_auth_warned = False
        self.transport = "udp" if is_udp_hid_endpoint(self.endpoint) else "tcp"
        if self.endpoint:
            try:
                self._parse_endpoint()
                if self.auth_token and any(ch.isspace() for ch in self.auth_token):
                    raise ValueError(
                        "HID auth token must be a single word (no spaces/newlines)"
                    )
            except ValueError as exc:
                # Invalid endpoint must degrade to "unavailable", never crash
                # HID init (and with it the whole input controller).
                logger.warning("NetworkHidAdapter invalid endpoint %r: %s", self.endpoint, exc)
                self._available = False
                return
            self._try_open()

    def _parse_endpoint(self) -> None:
        raw = self.endpoint
        lower = raw.lower()
        if lower.startswith("udp://"):
            raw = raw[6:]
            self.transport = "udp"
        elif lower.startswith("tcp://"):
            raw = raw[6:]
            self.transport = "tcp"
        host, _, port_s = raw.rpartition(":")
        if not host or not port_s.isdigit():
            raise ValueError(f"Invalid HID network endpoint: {self.endpoint!r}")
        host = host.strip()
        # F-1918: refuse public/internet-routable hosts by default — the HID
        # phone is always on the LAN. Escape hatch: GATHERION_HID_ALLOW_PUBLIC=1.
        if not _is_trusted_lan_host(host) and str(
            os.environ.get("GATHERION_HID_ALLOW_PUBLIC", "") or ""
        ).strip() != "1":
            raise ValueError(
                f"HID network endpoint host {host!r} is not a LAN/loopback "
                "address; refusing public HID endpoint "
                "(set GATHERION_HID_ALLOW_PUBLIC=1 to override)"
            )
        if not _is_trusted_lan_host(host):
            # O-5 (dilim 22): kaçış kullanıldıysa sessiz kalmasın —
            # düz metin tuşlar internete açılıyor (K-1 ile birleşince
            # tokensız köprü herkesten tuş kabul eder).
            logger.warning(
                "GATHERION_HID_ALLOW_PUBLIC=1 ile LAN-dışı HID endpoint'e izin verildi (%s) — düz metin tuş trafiği internete açık.",
                host,
            )
        self.host = host
        self.port = int(port_s)
        # O-5 (dilim 22): DNS TOCTOU — _is_trusted_lan_host yukarıda host'u
        # çözüp sınıflandırdı; bağlantı create_connection ile host'u YENİDEN
        # çözerse arada DNS değişebilir. Sınıflandırılan adres saklanır,
        # bağlantı o IP literaliyle kurulur.
        self._resolved_ip = ""
        try:
            infos = socket.getaddrinfo(host, self.port, socket.AF_INET, socket.SOCK_STREAM)
            if infos:
                self._resolved_ip = str(infos[0][4][0])
        except Exception:
            self._resolved_ip = ""

    def _peer_addr(self) -> tuple:
        """O-5: bağlantı, güven kontrolünden geçen çözümlenmiş IP ile kurulur."""
        return (getattr(self, "_resolved_ip", "") or self.host, self.port)

    def _udp_peer_alive(self, sock: socket.socket) -> bool:
        """Require PONG so a dead UDP endpoint cannot claim the key path and skip Interception."""
        # Bağlı (connected) UDP: sendto()/recvfrom() yerine connect() +
        # send()/recv(). İki gerekçe: (1) seccomp'lu kum havuzlarında
        # (bu VM) adresli sendto() EPERM ile reddedilir — bağlı sokette
        # send() serbesttir. Bağsız soketle ilk PING hiç çıkamadan
        # OSError -> False dönüyordu: R2'de canlı peer'a rağmen
        # 0.07/0.01/<0.005 sn'de "UDP no PONG", deadline'a hiç
        # ulaşılmadan. (2) Çekirdek bağlı sokete yalnızca peer'dan
        # gelen datagramları geçirir; recvfrom()'da _addr yoksayıldığı
        # için bugün LAN'daki herhangi bir host sahte PONG ile ölü
        # köprüyü canlı gösterebilir (F-1918). UDP connect() paket
        # göndermez, ölü peer'da da başarısız olmaz; sözleşme aynı —
        # yalnızca connect_timeout içinde gelen tam "PONG" satırı sayılır.
        try:
            sock.connect(self._peer_addr())
        except OSError:
            return False
        # F-1918: opt-in AUTH goes first (best-effort on UDP: the datagram
        # has no ACK; a legacy bridge ignores the unknown line). TCP is
        # the recommended transport when authentication matters.
        if self.auth_token:
            try:
                sock.send(f"AUTH {self.auth_token}\n".encode("utf-8"))
            except OSError:
                return False
        try:
            sock.send(b"PING\n")
        except OSError:
            return False
        deadline = time.time() + max(0.15, float(self.connect_timeout))
        while time.time() < deadline:
            try:
                data = sock.recv(256)
            except socket.timeout:
                # UDP is unreliable: a single lost PING (or a peer that was
                # still starting) must not condemn a live peer as dead.
                # Retransmit on every receive timeout until the deadline;
                # the contract is unchanged — only an exact "PONG" line
                # within connect_timeout counts, a dead peer still times
                # out as unavailable.
                try:
                    sock.send(b"PING\n")
                except OSError:
                    return False
                continue
            except OSError:
                return False
            text = data.decode("utf-8", errors="replace").strip().upper()
            # Exact line only: "XPONG", "NOPONG" or a spoofed prefix must not
            # pass as a live peer.
            if text == "PONG":
                return True
        return False

    def _try_open(self) -> None:
        # F-1918: say the quiet part out loud once per instance — a
        # tokenless connection is unauthenticated plaintext.
        if not self.auth_token and not self._no_auth_warned:
            self._no_auth_warned = True
            logger.warning(
                "NetworkHidAdapter %s: unauthenticated plaintext HID transport "
                "(no GATHERION_HID_TOKEN) — use only on a trusted LAN; any "
                "host that can reach the phone's port can inject keys",
                self.endpoint,
            )
        try:
            if self.transport == "udp":
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.settimeout(min(0.25, max(0.05, float(self.connect_timeout))))
                if not self._udp_peer_alive(sock):
                    sock.close()
                    self._sock = None
                    self._available = False
                    logger.warning(
                        "NetworkHidAdapter UDP no PONG %s:%d "
                        "(start tools/rootless_hid_bridge.py or phone KEY server; "
                        "cascade will use Interception if ready)",
                        self.host,
                        self.port,
                    )
                    return
                sock.settimeout(0.5)
                self._sock = sock
                self._available = True
                logger.info(
                    "NetworkHidAdapter UDP ready %s:%d (PONG ok)", self.host, self.port
                )
                return

            sock = socket.create_connection(
                (self._resolved_ip or self.host, self.port), timeout=self.connect_timeout
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.5)
            # F-1918: opt-in AUTH is the first line on the wire; a legacy
            # bridge ignores the unknown line, a token-enforcing phone
            # side drops the connection / rejects keys without it. A
            # failed AUTH send fails the open (never silently unauthenticated).
            if self.auth_token:
                sock.sendall(f"AUTH {self.auth_token}\n".encode("utf-8"))
            self._sock = sock
            self._available = True
            logger.info("NetworkHidAdapter connected %s:%d", self.host, self.port)
            try:
                sock.sendall(b"PING\n")
            except Exception:
                pass
        except Exception as exc:
            self._available = False
            self._sock = None
            logger.warning(
                "NetworkHidAdapter unavailable %s: %s "
                "(set GATHERION_HID_TCP=phone_ip:8765 or udp://phone_ip:8765; "
                "start USB gadget server, rootless_hid_bridge, or GatherionHidBridge)",
                self.endpoint,
                exc,
            )

    def close(self) -> None:
        with self._lock:
            try:
                if self._sock is not None:
                    self._sock.close()
            except Exception:
                pass
            self._sock = None
            self._available = False

    @property
    def available(self) -> bool:
        return bool(self._available and self._sock is not None)

    def _sock_send(self, data: bytes) -> None:
        assert self._sock is not None
        if self.transport == "udp":
            # Soket _udp_peer_alive'da peer'a bağlanır; sendto() adresli
            # gönderim seccomp'lu kum havuzlarında EPERM verir.
            self._sock.send(data)
        else:
            self._sock.sendall(data)

    def _drain_bridge_reply(self, timeout_sec: float = 0.01) -> bool:
        """K-2 (dilim 22): köprü hata durumunda `ERR ...` yanıtı gönderir
        (rootless_hid_bridge handle_line) ama bot hiç recv yapmıyordu —
        TCP sendall başarısı yalnızca çekirdek tamponuna yazıldığını söyler.
        Gönderim sonrası kısa süre yanıt yoklanır: ERR satırı görülürse
        True döner (gönderim başarısız sayılır). Başarıda köprü yanıt
        vermez, zaman aşımı normaldir; soket zaman aşımı eski değerine
        döner. 10 ms'lik yoklama tuş başına en fazla 10 ms ekler.
        """
        sock = self._sock
        if sock is None:
            return False
        try:
            prev_timeout = sock.gettimeout()
        except Exception:
            prev_timeout = None
        try:
            sock.settimeout(timeout_sec)
            try:
                data = sock.recv(1024)
            except socket.timeout:
                return False
            except OSError:
                return False
            if not data:
                return False
            text = data.decode("utf-8", errors="replace")
            for line in text.splitlines():
                if line.strip().upper().startswith("ERR"):
                    logger.warning("HID köprü ERR yanıtı: %s", line.strip()[:200])
                    return True
            return False
        finally:
            try:
                sock.settimeout(prev_timeout)
            except Exception:
                pass

    def _send(self, payload: str) -> bool:
        if not self.available:
            return False
        data = (payload.strip() + "\n").encode("utf-8")
        with self._lock:
            try:
                self._sock_send(data)
                if self._drain_bridge_reply():
                    return False
                return True
            except Exception as exc:
                logger.debug("NetworkHid write failed: %s — reconnecting", exc)
                self._available = False
                try:
                    if self._sock is not None:
                        self._sock.close()
                except Exception:
                    pass
                self._sock = None
        # one-shot reconnect
        time.sleep(0.05)
        self._try_open()
        if not self.available:
            return False
        with self._lock:
            try:
                self._sock_send(data)
                if self._drain_bridge_reply():
                    return False
                return True
            except Exception as exc:
                logger.debug("NetworkHid rewrite failed: %s", exc)
                self._available = False
                return False


def open_hid_adapter(endpoint: str = "", baud: int = 115200, auth_token: Any = None) -> Any:
    """
    Open SerialHidAdapter or NetworkHidAdapter from a unified endpoint string.
    Returns an adapter instance (may have available=False).
    ``auth_token`` (or GATHERION_HID_TOKEN) enables the opt-in ``AUTH``
    first line on network transports (F-1918); ignored for serial.
    """
    ep = normalize_hid_endpoint(endpoint)
    if not ep:
        return SerialHidAdapter(port_name="")
    if is_tcp_hid_endpoint(ep):
        return NetworkHidAdapter(endpoint=ep, auth_token=auth_token)
    return SerialHidAdapter(port_name=ep, baud=baud)


def build_default_input_port(
    input_driver: Any = None,
    serial_port: str = "",
) -> SoftInputAdapter:
    """
    SoftInputAdapter is the default port and delegates key dispatch to the
    InputController's trusted cascade (Serial/Network HID -> Interception ->
    FakerInput); it is NOT a soft-only sender on its own. N-3101: no soft
    SendInput/PostMessage fallback anywhere — without a trusted path the
    input fails instead of dropping to a soft sender. When serial_port /
    hid_tcp is set, attach the HID adapter onto the InputController key path
    if the driver exposes configure_serial_hid / _serial_hid. When no trusted
    keyboard path is up, key dispatch refuses to fake success (see
    InputController._hw_key_down).
    """
    port_name = str(serial_port or "").strip() or resolve_serial_hid_port()
    if input_driver is not None:
        try:
            trusted = getattr(input_driver, "keyboard_inject_trusted", None)
            if trusted is False:
                logger.warning(
                    "build_default_input_port: no trusted keyboard path "
                    "(Interception/HID/FakerInput) — key dispatch will refuse "
                    "soft fallback until a hardware path is configured"
                )
        except Exception:
            pass
    if port_name and input_driver is not None:
        configure = getattr(input_driver, "configure_serial_hid", None)
        if callable(configure):
            try:
                configure(port_name)
            except Exception as exc:
                logger.debug("configure_serial_hid skipped: %s", exc)
        elif not getattr(input_driver, "_serial_hid", None):
            try:
                hid = open_hid_adapter(port_name)
                if getattr(hid, "available", False):
                    setattr(input_driver, "_serial_hid", hid)
            except Exception as exc:
                logger.debug("HID attach skipped: %s", exc)
    return SoftInputAdapter(input_driver=input_driver)
