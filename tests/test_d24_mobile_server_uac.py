# -*- coding: utf-8 -*-
"""D24 düzeltme testleri: core/mobile_server.py + core/uac_elevator.py.

Kapsar: F-1648 ("t" tuşu), F-0029 (token varsayılan açık), F-1649
(sürücüsüz ok:True yok), F-1650 (cast_skill fare yolu yok), F-1659 (WS
kısıtları), F-1652/F-0136 (32 karakter token), F-1651 (token sızıntısı
yok), F-1658 (GET salt-okunur), F-1657 (simsiyah kare atılmaz), F-1654
(worker yokken ok:True yok), F-1660 (skip_node), F-1653 (Interception
doğrulaması), F-1655 (update_settings), F-1656 (elevate/restart/quit),
F-1668 (WS çerçeve), F-1665 (POST koruması), F-1664 (port=0), F-1667
(log handler), F-1661 (sahte metrik yok), F-1669 (mode/button
doğrulama), F-1662 (reset cooldown 0), F-1663 (1920x1080 sabiti yok),
F-1670 (QR sızıntısı + yedek panel token), F-1666 (hazır stub'lar),
F-1931/F-1930 (UAC), F-1674 (HTTP log), F-0469 (CORS/dashboard auth),
F-1675 (port fallback), F-1672 (çift sayım), F-1671 (halka son aşama),
F-1673 (kodlama hatası).
"""

import contextlib
import inspect
import logging
import os
import socket
import struct
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.mobile_server as ms
from core.mobile_server import MobileRemoteServer
import core.uac_elevator as ue

TOKEN32 = "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4Z"


class FakeInputCtrl:
    """test_mode=True => güvenilir sayılır (simülasyon)."""

    def __init__(self, width=1920, height=1080, trusted=True):
        self.test_mode = bool(trusted)
        self.screen_width = width
        self.screen_height = height
        self.pressed_keys = []
        self.clicked = []

    def press_key(self, key, *args, **kwargs):
        self.pressed_keys.append(str(key))
        return 0.12

    def click_normalized(self, norm_x, norm_y, button="left", double=False):
        px = int(round(float(norm_x) * self.screen_width))
        py = int(round(float(norm_y) * self.screen_height))
        self.clicked.append((px, py, button))
        return px, py, 0.04


class UntrustedInputCtrl(FakeInputCtrl):
    """Yumuşak SendInput backend'i; test_mode kapalı, trusted API yok."""

    def __init__(self):
        super().__init__(trusted=False)
        self.test_mode = False
        self.active_backend_name = "ctypes.SendInput"


class FakeNode:
    def __init__(self, node_id, state="active", rest_until=0.0):
        self.id = node_id
        self.map_x = 960
        self.map_y = 540
        self.norm_x = 0.5
        self.norm_y = 0.5
        self.state = state
        self.source_type = "manual"
        self.resource_type = "Orichalcum Ore"
        self.region = "Altgard"
        self.current_harvests = 0
        self.max_harvests = 3
        self.consecutive_failures = 0
        self.last_visited_timestamp = 0.0
        self.rest_duration_sec = 300
        self.rest_until = rest_until
        self.gathered_count = 0
        self.in_air = False
        self.seq_index = 0

    def is_ready(self, now=None):
        current = float(now) if now is not None else time.time()
        return self.state == "active" and current >= self.rest_until

    def remaining_rest_sec(self, now=None):
        current = float(now) if now is not None else time.time()
        return max(0, int(self.rest_until - current))


class FakeRouteManager:
    def __init__(self, nodes):
        self.active_region = "Altgard"
        self.active_route = "All"
        self.total_gathered = 0
        self.consecutive_failures = 0
        self.meters_per_pixel = 0.28
        self.default_rest_sec = 300
        self._nodes = list(nodes)
        self.update_resting_calls = 0
        self.saved_regions = []

    def get_route_nodes(self, region=None, route=None):
        return list(self._nodes)

    def get_node(self, node_id, region=None):
        for n in self._nodes:
            if n.id == int(node_id):
                return n
        return None

    def get_next_ready_node(self, *args, **kwargs):
        for n in self._nodes:
            if n.state == "active":
                return n
        return None

    def update_resting_nodes(self, region=None, **kwargs):
        self.update_resting_calls += 1
        return []

    def mark_node_gathered(self, node_id, rest_minutes=5, force_deplete=False):
        n = self.get_node(node_id)
        if n is not None:
            n.state = "resting"

    def save_region(self, region=None, **kwargs):
        self.saved_regions.append(region)
        return True


class FakeWorker:
    def __init__(self, rm=None, active_node=None):
        self.state = "WALKING"
        self.mode = "route"
        self.is_paused = False
        self._pause_reason = ""
        self._session_start_time = time.time() - 60.0
        self.active_node = active_node
        self.active_node_distance_m = 0
        self.gathered_count = 0
        self.failed_in_row = 0
        self.route_manager = rm
        self.set_modes = []

    def pause(self, reason=""):
        self.is_paused = True
        self.state = "PAUSED"
        self._pause_reason = reason

    def resume(self):
        self.is_paused = False
        self.state = "WALKING"

    def set_mode(self, mode):
        self.set_modes.append(mode)
        self.mode = mode


def make_server(**kwargs):
    kwargs.setdefault("access_token", TOKEN32)
    srv = MobileRemoteServer(**kwargs)
    return srv


@contextlib.contextmanager
def _server(**kwargs):
    """make_server() + guaranteed log-handler detach.

    The suite used to repeat the same 3-line pattern ~20 times:
    ``srv = make_server(...); try: ... finally: srv._detach_log_handler()``.
    The context manager is exactly that pattern — same construction, same
    detach in ``finally`` — with the boilerplate in one place. It detaches
    even when the body raises (see test_server_context_manager_detaches_*).
    """
    srv = make_server(**kwargs)
    try:
        yield srv
    finally:
        srv._detach_log_handler()


@pytest.fixture()
def server():
    srv = make_server()
    yield srv
    srv._detach_log_handler()


# --- F-1648 / F-1653 / F-1649: send_key -------------------------------------

def test_server_context_manager_detaches_on_exit():
    root = logging.getLogger()
    with _server() as srv:
        handler = srv._log_handler
        assert handler is not None
        assert handler in root.handlers
    assert srv._log_handler is None
    assert handler not in root.handlers


def test_server_context_manager_detaches_on_error():
    root = logging.getLogger()
    with pytest.raises(RuntimeError, match="boom"):
        with _server() as srv:
            handler = srv._log_handler
            assert handler in root.handlers
            raise RuntimeError("boom")
    assert srv._log_handler is None
    assert handler not in root.handlers


def test_send_key_t_allowed_and_sent():
    assert "t" in ms.ALLOWED_REMOTE_KEYS
    assert "r" in ms.ALLOWED_REMOTE_KEYS
    ic = FakeInputCtrl()
    with _server(input_ctrl=ic) as srv:
        res = srv.execute_action({"action": "send_key", "key": "t"})
        assert res["ok"] is True
        assert ic.pressed_keys == ["t"]


def test_send_key_no_driver_returns_not_ok():
    with _server() as srv:
        res = srv.execute_action({"action": "send_key", "key": "t"})
        assert res["ok"] is False


def test_send_key_untrusted_backend_refused():
    with _server(input_ctrl=UntrustedInputCtrl()) as srv:
        res = srv.execute_action({"action": "send_key", "key": "f"})
        assert res["ok"] is False
        assert "Interception" in res["error"] or "HID" in res["error"]


def test_click_screen_no_driver_returns_not_ok():
    with _server() as srv:
        assert srv.execute_action({"action": "click_screen", "norm_x": 0.5, "norm_y": 0.5})["ok"] is False
        assert srv.execute_action({"action": "tap_to_move_map", "map_x": 100, "map_y": 100})["ok"] is False


def test_click_screen_invalid_button():
    with _server(input_ctrl=FakeInputCtrl()) as srv:
        res = srv.execute_action({"action": "click_screen", "norm_x": 0.5, "norm_y": 0.5, "button": "bogus"})
        assert res["ok"] is False


# --- F-0029 / F-1652 / F-0136 / F-1651: token -------------------------------

def test_require_token_default_true():
    sig = inspect.signature(MobileRemoteServer.__init__)
    assert sig.parameters["require_token"].default is True


def test_generated_token_is_32_chars(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "PROJECT_ROOT", str(tmp_path))
    srv = MobileRemoteServer(access_token=None)
    try:
        assert len(srv.access_token) == 32
        assert srv.access_token.isalnum()
        tok_file = tmp_path / "config" / "mobile_token.txt"
        assert tok_file.exists()
        assert oct(os.stat(tok_file).st_mode & 0o777) == "0o600"
    finally:
        srv._detach_log_handler()


def test_short_explicit_token_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "PROJECT_ROOT", str(tmp_path))
    srv = MobileRemoteServer(access_token="abc")
    try:
        assert srv.access_token != "abc"
        assert len(srv.access_token) == 32
    finally:
        srv._detach_log_handler()


def test_connection_info_does_not_leak_token(server):
    info = server.get_connection_info()
    assert "access_token" not in info
    assert info["access_token_present"] is True
    assert TOKEN32 not in str(info)
    assert info["qr_url"] == ""  # F-1670: üçüncü taraf QR yok
    assert "qrserver" not in str(info)


# --- F-1654 / F-1660 / F-1655: worker yokken ok:True yok --------------------

def test_actions_without_worker_return_not_ok():
    with _server() as srv:
        for payload in (
            {"action": "pause"},
            {"action": "resume"},
            {"action": "set_mode", "mode": "gather"},
            {"action": "record_f8"},
            {"action": "optimize_route"},
            {"action": "update_settings", "harvests_per_node": 4},
            {"action": "skip_node"},
        ):
            res = srv.execute_action(payload)
            assert res["ok"] is False, payload


def test_set_mode_unknown_mode_rejected():
    w = FakeWorker()
    with _server(worker=w) as srv:
        assert srv.execute_action({"action": "set_mode", "mode": "warp"})["ok"] is False
        assert srv.execute_action({"action": "set_mode", "mode": "gather", "auto_resume": "yes"})["ok"] is False
        res = srv.execute_action({"action": "set_mode", "mode": "gather", "auto_resume": False})
        assert res["ok"] is True and w.mode == "auto_record"


def test_skip_node_without_active_node_does_not_mark_next():
    node = FakeNode(7, state="active")
    rm = FakeRouteManager([node])
    w = FakeWorker(rm=rm, active_node=None)
    with _server(worker=w, route_mgr=rm) as srv:
        res = srv.execute_action({"action": "skip_node"})
        assert res["ok"] is False
        assert node.state == "active"  # sıradaki hazır node toplanmış sayılmaz


# --- F-1662: reset cooldown 0 ----------------------------------------------

def test_reset_node_writes_rest_duration_zero():
    node = FakeNode(2, state="resting")
    node.rest_duration_sec = 300
    rm = FakeRouteManager([node])
    with _server(route_mgr=rm) as srv:
        res = srv.execute_action({"action": "reset_node", "node_id": 2})
        assert res["ok"] is True
        assert node.rest_duration_sec == 0
        assert res["cooldown_remaining"] == 0


# --- F-1658 / F-1672 / F-1661: status payload -------------------------------

def test_status_payload_readonly_and_counts():
    q_node = FakeNode(1, state="quarantined")
    a_node = FakeNode(2, state="active")
    rm = FakeRouteManager([q_node, a_node])
    with _server(route_mgr=rm) as srv:
        payload = srv.build_status_payload()
        assert rm.update_resting_calls == 0  # GET durumu değiştirmez
        assert payload["route"]["quarantined_nodes"] == 1
        assert payload["route"]["failed_nodes"] == 0  # çift sayım yok
        assert payload["system"]["fps"] is None  # sahte varsayılan yok
        assert payload["system"]["input_backend"] is None
        assert payload["vitals"]["hp_ratio"] is None
        assert payload["route"]["total_nodes"] == 2


def test_nodes_payload_readonly():
    rm = FakeRouteManager([FakeNode(1)])
    with _server(route_mgr=rm) as srv:
        nodes = srv.build_nodes_payload()
        assert len(nodes) == 1
        assert rm.update_resting_calls == 0


# --- F-1671: halka son aşama -------------------------------------------------

def test_ring_last_stage_done():
    node = FakeNode(1)
    node.current_harvests = 3
    node.max_harvests = 3
    w = FakeWorker(active_node=node)
    w.state = "GATHERING"
    with _server(worker=w) as srv:
        frame, _ = srv.build_fast_telemetry_frame()
        steps = frame["harvest"]["ring_steps"]
        assert [s["status"] for s in steps] == ["done", "done", "done"]


# --- F-1657 / F-1669 / F-1673: kare ------------------------------------------

class BlackCapture:
    def grab_full(self, grayscale=False):
        return np.zeros((1080, 1920, 3), dtype=np.uint8)


def test_black_frame_not_replaced_by_synthetic():
    with _server(capture=BlackCapture()) as srv:
        frame = srv._grab_latest_bgr_frame()
        assert int(np.max(frame)) == 0  # gerçek simsiyah kare aynen döner


def test_render_frame_unknown_mode_raises(server):
    with pytest.raises(ValueError):
        server.render_frame_jpeg(mode="sideways")


def test_render_pwa_icon_png_ok(server):
    png = server.render_pwa_icon_png(size=192)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 100


# --- F-1663: tap_to_move gerçek ekran boyutu ---------------------------------

def test_tap_to_move_uses_real_screen_size():
    ic = FakeInputCtrl(width=2560, height=1440)
    with _server(input_ctrl=ic) as srv:
        res = srv.execute_action({"action": "tap_to_move_map", "norm_x": 0.5, "norm_y": 0.5})
        assert res["ok"] is True
        assert res["map_x"] == 1280
        assert res["map_y"] == 720
        assert res["pixel_x"] == 1280


# --- F-1650: cast_skill fare yolu yok -----------------------------------------

def test_cast_skill_without_bus_or_driver_fails():
    with _server() as srv:
        res = srv.execute_action({"action": "cast_skill", "slot": "1"})
        assert res["ok"] is False


def test_cast_skill_key_path_with_trusted_driver():
    ic = FakeInputCtrl()
    with _server(input_ctrl=ic) as srv:
        res = srv.execute_action({"action": "cast_skill", "slot": "1"})
        assert res["ok"] is True
        assert res["method"] == "key"
        assert ic.pressed_keys == ["1"]


# --- F-1668: WS çerçeve -------------------------------------------------------

def _masked_frame(opcode, payload, fin=True):
    first = (0x80 if fin else 0x00) | opcode
    mask = b"\x11\x22\x33\x44"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    return bytes([first, 0x80 | len(payload)]) + mask + masked


def test_ws_masked_frame_roundtrip():
    rx, tx = socket.socketpair()
    try:
        rx.settimeout(2.0)
        tx.settimeout(2.0)
        tx.sendall(_masked_frame(0x1, b'{"action":"ping"}'))
        assert ms._ws_read_client_frame(rx) == (0x1, b'{"action":"ping"}')
    finally:
        rx.close()
        tx.close()


def test_ws_unmasked_frame_rejected():
    rx, tx = socket.socketpair()
    try:
        rx.settimeout(2.0)
        tx.settimeout(2.0)
        tx.sendall(bytes([0x81, 0x05]) + b"hello")
        assert ms._ws_read_client_frame(rx) is None
    finally:
        rx.close()
        tx.close()


def test_ws_fragmented_frame_reassembled():
    rx, tx = socket.socketpair()
    try:
        rx.settimeout(2.0)
        tx.settimeout(2.0)
        tx.sendall(_masked_frame(0x1, b"he", fin=False) + _masked_frame(0x0, b"llo", fin=True))
        assert ms._ws_read_client_frame(rx) == (0x1, b"hello")
    finally:
        rx.close()
        tx.close()


# --- F-1664 / F-1675: port -----------------------------------------------------

def test_start_port_zero_gets_ephemeral_port():
    srv = make_server(host="127.0.0.1", port=0, require_token=False)
    try:
        srv.start()
        assert srv.is_running
        assert srv.port not in (0,)
        assert srv._port_fallback_from is None
    finally:
        srv.stop()
        srv._detach_log_handler()


# --- F-1667: log handler -------------------------------------------------------

def test_attach_log_handler_keeps_foreign_handlers():
    root = logging.getLogger()
    foreign = ms.MobileLogHandler(__import__("collections").deque(maxlen=10), threading.RLock())
    root.addHandler(foreign)
    try:
        with _server() as srv:
            assert foreign in root.handlers
        assert foreign in root.handlers
    finally:
        root.removeHandler(foreign)


# --- F-1666: hazır bayrakları gerçek -------------------------------------------

def test_validate_mobile_server_contract():
    assert ms.validate_mobile_server_contract() is True


def test_mobile_ready_flag_real(server):
    assert server.get_mobile_server_ready_flag()["ready"] is True
    assert server.get_mobile_server_health_summary()["mobile_server_ready"] is True


# --- F-1670 (P1-9/K-08): yedek panel token'lı — başlıkla, sorguda değil --------

def test_fallback_html_uses_header_token_not_query_token(server):
    html = server._builtin_fallback_html(token=TOKEN32)
    # Token URL sorgusunda taşınmaz (log/QR sızıntısı): istekler
    # `X-Gatherion-Token` başlığıyla gider; token sayfa içindeki JS
    # sabitine yazılır (core/mobile_server.py:4293-4380).
    assert "?token=" not in html
    assert "X-Gatherion-Token" in html
    assert 'const AUTH_TOKEN = "' + TOKEN32 + '"' in html
    assert "h['X-Gatherion-Token'] = AUTH_TOKEN" in html


def test_load_dashboard_fallback_uses_header_token():
    with _server() as srv:
        srv.dashboard_html_path = "/nonexistent/mobile_dashboard.html"
        html = srv.load_dashboard_html()
        assert "?token=" not in html
        assert "X-Gatherion-Token" in html


# --- F-1931 / F-1930: UAC -------------------------------------------------------

def test_uac_dry_run_false_off_windows():
    if not sys.platform.startswith("win"):
        assert ue.ensure_admin_elevation(dry_run=True) is False


def test_uac_already_admin_does_not_call_mock():
    calls = []

    def _mock(hwnd, verb, exe, prms, wdir, show):
        calls.append((verb, exe))
        return 42

    if ue.is_admin():
        assert ue.ensure_admin_elevation(shell_execute_fn=_mock) is True
        assert calls == []
    else:
        pytest.skip("not admin; mock path covered by __main__ self-check")


def test_uac_contract_is_real():
    assert ue.validate_uac_elevator_contract() is True
    flag = ue.get_uac_elevator_ready_flag()
    assert flag["platform_supported"] == sys.platform.startswith("win")
    if not sys.platform.startswith("win"):
        assert flag["ready"] is False
