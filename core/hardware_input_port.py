# -*- coding: utf-8 -*-
"""
HardwareInputPort — girdi soyutlaması.

Varsayılan SoftInputAdapter mevcut InputController'a delege eder.

HID klavye köprüleri (opt-in; COM1 asla otomatik açılmaz):
  - SerialHidAdapter  — Arduino Leonardo / KMBox (GATHERION_HID_COM / input.serial_port)
  - NetworkHidAdapter — Android BT-HID / USB-gadget (GATHERION_HID_TCP / input.hid_tcp)
                        tcp://host:port  |  udp://host:port  |  host:port (TCP)

Aynı satır protokolü (newline UTF-8):
  KEY <name> <hold_ms> | KEYDOWN <name> | KEYUP <name>
  MOUSE <btn> DOWN|UP | MOVE <x> <y> | CLICK <x> <y> <btn>
  PING
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
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


def _load_settings_json_dict() -> dict:
    """Best-effort read of config/settings.json (repo root)."""
    try:
        from pathlib import Path
        import json

        path = Path(__file__).resolve().parent.parent / "config" / "settings.json"
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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
    env_com = str(os.environ.get("GATHERION_HID_COM", "") or "").strip()
    if env_com:
        return normalize_hid_endpoint(env_com)

    env_tcp = str(os.environ.get("GATHERION_HID_TCP", "") or "").strip()
    if env_tcp:
        return normalize_hid_endpoint(env_tcp)

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

    if not isinstance(raw, dict):
        return ""

    input_cfg = raw.get("input", {}) if isinstance(raw.get("input"), dict) else {}
    port = (
        raw.get("serial_port")
        or input_cfg.get("serial_port")
        or input_cfg.get("hid_com")
        or ""
    )
    if str(port or "").strip():
        return normalize_hid_endpoint(str(port).strip())

    tcp = (
        raw.get("hid_tcp")
        or input_cfg.get("hid_tcp")
        or input_cfg.get("hid_host")
        or ""
    )
    return normalize_hid_endpoint(str(tcp or "").strip())


def is_tcp_hid_endpoint(endpoint: str) -> bool:
    ep = normalize_hid_endpoint(endpoint).lower()
    return ep.startswith("tcp://") or ep.startswith("udp://")


def is_udp_hid_endpoint(endpoint: str) -> bool:
    return normalize_hid_endpoint(endpoint).lower().startswith("udp://")


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

    def send_key(
        self,
        key: str,
        min_hold: float = 0.08,
        max_hold: float = 0.12,
    ) -> bool:
        driver = self.input_driver
        if driver is None or not hasattr(driver, "press_key"):
            return False
        try:
            driver.press_key(
                str(key).lower(),
                min_hold=min_hold,
                max_hold=max_hold,
                release_mouse=True,
                post_to_window=False,
            )
            return True
        except TypeError:
            try:
                driver.press_key(str(key).lower(), min_hold=min_hold, max_hold=max_hold)
                return True
            except Exception as exc:
                logger.debug("SoftInput send_key: %s", exc)
                return False
        except Exception as exc:
            logger.debug("SoftInput send_key: %s", exc)
            return False

    def mouse_button(self, button: str, down: bool) -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        btn = str(button or "left").lower()
        try:
            if down:
                fn = getattr(driver, "_hw_mouse_down", None)
                if callable(fn):
                    fn(btn)
                    return True
                send = getattr(driver, "_sendinput_mouse_button", None)
                if callable(send):
                    send(btn, True)
                    return True
            else:
                fn = getattr(driver, "_hw_mouse_up", None)
                if callable(fn):
                    fn(btn)
                    return True
                send = getattr(driver, "_sendinput_mouse_button", None)
                if callable(send):
                    send(btn, False)
                    return True
        except Exception as exc:
            logger.debug("SoftInput mouse_button: %s", exc)
        return False

    def move(self, x: int, y: int) -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        try:
            if hasattr(driver, "_hw_move_to"):
                driver._hw_move_to(int(x), int(y))
                return True
            if hasattr(driver, "move_to"):
                driver.move_to(int(x), int(y))
                return True
        except Exception as exc:
            logger.debug("SoftInput move: %s", exc)
        return False

    def click(self, x: int, y: int, button: str = "left") -> bool:
        driver = self.input_driver
        if driver is None:
            return False
        try:
            if hasattr(driver, "human_click"):
                driver.human_click(int(x), int(y), button=str(button))
                return True
            if hasattr(driver, "click_map"):
                driver.click_map(int(x), int(y), button=str(button))
                return True
            if self.move(x, y) and self.mouse_button(button, True):
                self.mouse_button(button, False)
                return True
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
        hold_ms = int(max(min_hold, max_hold) * 1000)
        return self._send(f"KEY {key} {hold_ms}")

    def mouse_button(self, button: str, down: bool) -> bool:
        state = "DOWN" if down else "UP"
        return self._send(f"MOUSE {button} {state}")

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
        try:
            self._serial.write((payload.strip() + "\n").encode("utf-8"))
            return True
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
    """

    DEFAULT_PORT = 8765

    def __init__(self, endpoint: str = "", connect_timeout: float = 2.0) -> None:
        self.endpoint = normalize_hid_endpoint(endpoint)
        self.connect_timeout = float(connect_timeout)
        self.host = ""
        self.port = self.DEFAULT_PORT
        self._sock: Optional[socket.socket] = None
        self._available = False
        self._lock = threading.Lock()
        self.transport = "udp" if is_udp_hid_endpoint(self.endpoint) else "tcp"
        if self.endpoint:
            self._parse_endpoint()
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
        self.host = host.strip()
        self.port = int(port_s)

    def _udp_peer_alive(self, sock: socket.socket) -> bool:
        """Require PONG so a dead UDP endpoint cannot claim the key path and skip Interception."""
        try:
            sock.sendto(b"PING\n", (self.host, self.port))
        except OSError:
            return False
        deadline = time.time() + max(0.15, float(self.connect_timeout))
        while time.time() < deadline:
            try:
                data, _addr = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                return False
            text = data.decode("utf-8", errors="replace").strip().upper()
            if "PONG" in text:
                return True
        return False

    def _try_open(self) -> None:
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
                (self.host, self.port), timeout=self.connect_timeout
            )
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(0.5)
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
            self._sock.sendto(data, (self.host, self.port))
        else:
            self._sock.sendall(data)

    def _send(self, payload: str) -> bool:
        if not self.available:
            return False
        data = (payload.strip() + "\n").encode("utf-8")
        with self._lock:
            try:
                self._sock_send(data)
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
                return True
            except Exception as exc:
                logger.debug("NetworkHid rewrite failed: %s", exc)
                self._available = False
                return False


def open_hid_adapter(endpoint: str = "", baud: int = 115200) -> Any:
    """
    Open SerialHidAdapter or NetworkHidAdapter from a unified endpoint string.
    Returns an adapter instance (may have available=False).
    """
    ep = normalize_hid_endpoint(endpoint)
    if not ep:
        return SerialHidAdapter(port_name="")
    if is_tcp_hid_endpoint(ep):
        return NetworkHidAdapter(endpoint=ep)
    return SerialHidAdapter(port_name=ep, baud=baud)


def build_default_input_port(
    input_driver: Any = None,
    serial_port: str = "",
) -> SoftInputAdapter:
    """
    SoftInputAdapter is the default port (mouse soft-aim stays intact).
    When serial_port / hid_tcp is set, attach HID adapter onto the InputController
    key path if the driver exposes configure_serial_hid / _serial_hid.
    """
    port_name = str(serial_port or "").strip() or resolve_serial_hid_port()
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
