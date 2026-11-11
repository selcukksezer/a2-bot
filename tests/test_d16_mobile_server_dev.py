# -*- coding: utf-8 -*-
"""D16 DEV testleri: core/mobile_server.py iyileştirmeleri.

Kapsar:
- (B) `_ws_unmask_payload` vektörize maske kaldırma: sınır boyutlarda
  tur (0/125/126/1000/65535/65536), parçalı çerçeve, maskesiz ret.
- (A) statik dosya önbelleği (`_cached_static_text`): pano/manifest/SW
  ikinci çağrıda diskten okunmaz; dosya değişince önbellek geçersiz.
- (C) sentetik kare statik taban önbelleği: çıktı piksel-birebir aynı
  (değişiklik öncesi bench hash'iyle kilitli), dönen kare tabanın
  kopyasıdır (çağıran kareyi bozsa bile sonraki kare etkilenmez).
- (D) `_crop_norm_bounds_bgr` tekilleştirmesi: minimap/gather kırpmaları
  piksel-birebir aynı (değişiklik öncesi bench hash'iyle kilitli).
- (E) `_write_token_file`: rotate 32 karakterlik token'ı 0600 izinle yazar.
"""

import hashlib
import os
import socket
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core.mobile_server as ms
from core.mobile_server import MobileRemoteServer

TOKEN32 = "Ab3dEf6hIj9kLm2nOp5qRs8tUv1wXy4Z"

# Değişiklik ÖNCESİ bench (~/workspace/d16_bench_before.json) hash'leri:
# aynı fikstürle üretilen kare/kırpma çıktıları piksel-birebir aynı kalmalı.
EXPECTED_SYNTHETIC_SHA = "f95d5dde832e58fb"
EXPECTED_MINIMAP_CROP_SHA = "2b1ae693c0480344"
EXPECTED_GATHER_CROP_SHA = "83237d6cc8772e89"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


class FakeNode:
    def __init__(self, node_id, state="active"):
        self.id = node_id
        self.map_x = 960
        self.map_y = 540
        self.norm_x = 0.5
        self.norm_y = 0.5
        self.state = state
        self.source_type = "manual"
        self.resource_type = "Orichalcum Ore"
        self.region = "Altgard"
        self.current_harvests = 1
        self.max_harvests = 3
        self.consecutive_failures = 0
        self.last_visited_timestamp = 0.0
        self.rest_duration_sec = 300
        self.gathered_count = 0
        self.in_air = False
        self.seq_index = 0

    def is_ready(self, now=None):
        return self.state == "active"

    def remaining_rest_sec(self, now=None):
        return 0


class FakeRouteManager:
    def __init__(self, nodes):
        self.active_region = "Altgard"
        self.active_route = "All"
        self.total_gathered = 0
        self.consecutive_failures = 0
        self._nodes = list(nodes)

    def get_route_nodes(self, region=None, route=None):
        return list(self._nodes)

    def get_node(self, node_id, region=None):
        for n in self._nodes:
            if n.id == int(node_id):
                return n
        return None


def make_server(**kwargs):
    kwargs.setdefault("access_token", TOKEN32)
    return MobileRemoteServer(**kwargs)


@pytest.fixture()
def server():
    srv = make_server()
    yield srv
    srv._detach_log_handler()


# --- (B) WS unmask -----------------------------------------------------------

def _masked_frame(opcode, payload, fin=True):
    first = (0x80 if fin else 0x00) | opcode
    mask = b"\x11\x22\x33\x44"
    masked = ms._ws_unmask_payload(payload, mask)  # XOR kendi tersidir
    n = len(payload)
    if n <= 125:
        header = bytes([first, 0x80 | n])
    elif n <= 65535:
        header = bytes([first, 0x80 | 126]) + n.to_bytes(2, "big")
    else:
        header = bytes([first, 0x80 | 127]) + n.to_bytes(8, "big")
    return header + mask + masked


@pytest.mark.parametrize("size", [0, 1, 125, 126, 127, 1000, 65535, 65536])
def test_ws_unmask_roundtrip_boundary_sizes(size):
    payload = bytes((i * 7 + 3) % 256 for i in range(size))
    rx, tx = socket.socketpair()
    try:
        rx.settimeout(5.0)
        tx.settimeout(5.0)
        tx.sendall(_masked_frame(0x1, payload))
        assert ms._ws_read_client_frame(rx) == (0x1, payload)
    finally:
        rx.close()
        tx.close()


def test_ws_unmask_payload_direct():
    rng = np.random.default_rng(16)
    payload = rng.integers(0, 256, size=4097, dtype=np.uint8).tobytes()
    mask = b"\xDE\xAD\xBE\xEF"
    masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
    assert ms._ws_unmask_payload(masked, mask) == payload
    assert ms._ws_unmask_payload(b"", mask) == b""
    # Maske kaldırma kendi tersidir (XOR involüsyonu).
    assert ms._ws_unmask_payload(ms._ws_unmask_payload(payload, mask), mask) == payload


def test_ws_unmasked_frame_still_rejected():
    rx, tx = socket.socketpair()
    try:
        rx.settimeout(2.0)
        tx.settimeout(2.0)
        tx.sendall(bytes([0x81, 0x05]) + b"hello")
        assert ms._ws_read_client_frame(rx) is None
    finally:
        rx.close()
        tx.close()


def test_ws_fragmented_frame_still_reassembled():
    rx, tx = socket.socketpair()
    try:
        rx.settimeout(2.0)
        tx.settimeout(2.0)
        tx.sendall(_masked_frame(0x1, b"he", fin=False) + _masked_frame(0x0, b"llo", fin=True))
        assert ms._ws_read_client_frame(rx) == (0x1, b"hello")
    finally:
        rx.close()
        tx.close()


# --- (A) statik dosya önbelleği ----------------------------------------------

def test_dashboard_html_cached_and_identical(server):
    first = server.load_dashboard_html()
    assert len(first) > 1000
    assert server.dashboard_html_path in server._static_file_cache
    second = server.load_dashboard_html()
    assert second == first
    # Önbellek isabeti: dosya okunmadan aynı içerik döner.
    assert 'rel="manifest"' in second


def test_dashboard_html_cache_invalidated_on_file_change(server, tmp_path):
    html_path = tmp_path / "mobile_dashboard.html"
    html_path.write_text("<html><head></head><body>ILK</body></html>", encoding="utf-8")
    server.dashboard_html_path = str(html_path)
    assert "ILK" in server.load_dashboard_html()
    # Farklı boyut + yeni mtime ile dosyayı değiştir.
    time.sleep(0.02)
    html_path.write_text("<html><head></head><body>YENI-ICERIK-DAHA-UZUN</body></html>", encoding="utf-8")
    os.utime(html_path, (time.time() + 2, time.time() + 2))
    assert "YENI-ICERIK-DAHA-UZUN" in server.load_dashboard_html()


def test_manifest_and_sw_cached(server, tmp_path):
    man_path = tmp_path / "manifest.json"
    man_path.write_text('{"name": "Test"}', encoding="utf-8")
    server.manifest_json_path = str(man_path)
    assert server.load_manifest_json() == '{"name": "Test"}'
    assert server.manifest_json_path in server._static_file_cache
    assert server.load_manifest_json() == '{"name": "Test"}'

    sw_path = tmp_path / "sw.js"
    sw_path.write_text("console.log(1);", encoding="utf-8")
    server.service_worker_js_path = str(sw_path)
    assert server.load_service_worker_js() == "console.log(1);"
    assert server.load_service_worker_js() == "console.log(1);"


def test_static_cache_missing_file_falls_back(server):
    server.dashboard_html_path = "/nonexistent/mobile_dashboard.html"
    assert server.dashboard_html_path not in server._static_file_cache
    html = server.load_dashboard_html()
    assert "Gatherion" in html  # yerleşik yedek panel
    assert server.dashboard_html_path not in server._static_file_cache


# --- (C) sentetik kare: piksel-birebir + kopya semantiği ----------------------

def _synthetic_server():
    rm = FakeRouteManager([FakeNode(1), FakeNode(2, "quarantined"), FakeNode(3)])
    return make_server(route_mgr=rm)


def test_synthetic_frame_pixel_identical_to_before():
    srv = _synthetic_server()
    try:
        frame = srv._build_synthetic_telemetry_frame()
        assert frame.shape == (1080, 1920, 3)
        assert _sha(frame.tobytes()) == EXPECTED_SYNTHETIC_SHA
        # İkinci çağrı da aynı pikselleri verir.
        assert _sha(srv._build_synthetic_telemetry_frame().tobytes()) == EXPECTED_SYNTHETIC_SHA
    finally:
        srv._detach_log_handler()


def test_synthetic_frame_returns_copy_not_cached_base():
    srv = _synthetic_server()
    try:
        frame = srv._build_synthetic_telemetry_frame()
        frame[:, :] = 0  # çağıran kareyi bozarsa...
        frame2 = srv._build_synthetic_telemetry_frame()
        assert _sha(frame2.tobytes()) == EXPECTED_SYNTHETIC_SHA
        assert srv._synthetic_base_frame is not None
        assert int(np.max(srv._synthetic_base_frame)) > 0
    finally:
        srv._detach_log_handler()


def test_synthetic_frame_reflects_node_state_change():
    nodes = [FakeNode(1), FakeNode(2), FakeNode(3)]
    # Düğümler farklı koordinatlarda olmalı (aynı piksele çizilirlerse
    # son çizilen diğerlerinin rengini örter).
    for n, (mx, my) in zip(nodes, [(400, 300), (960, 540), (1500, 800)]):
        n.map_x, n.map_y = mx, my
    rm = FakeRouteManager(nodes)
    srv = make_server(route_mgr=rm)
    try:
        before = _sha(srv._build_synthetic_telemetry_frame().tobytes())
        rm._nodes[1].state = "quarantined"  # dinamik kısım değişir
        after = _sha(srv._build_synthetic_telemetry_frame().tobytes())
        assert before != after
    finally:
        srv._detach_log_handler()


# --- (D) kırpma tekilleştirmesi: piksel-birebir ------------------------------

def test_crops_pixel_identical_to_before():
    srv = _synthetic_server()
    try:
        frame = srv._build_synthetic_telemetry_frame()
        mm, mm_rect = srv._crop_minimap_bgr(frame)
        gr, gr_rect = srv._crop_gather_roi_bgr(frame)
        assert _sha(mm.tobytes()) == EXPECTED_MINIMAP_CROP_SHA
        assert _sha(gr.tobytes()) == EXPECTED_GATHER_CROP_SHA
        # _crop_norm_bounds_bgr artık rect'e "valid" + "fallback" da yazar
        # (core/mobile_server.py:2428) — piksel içerik aynı, meta genişledi.
        assert mm_rect == {"x1": 1613, "y1": 54, "x2": 1910, "y2": 292, "width": 297, "height": 238, "valid": True, "fallback": None}
        assert gr_rect == {"x1": 806, "y1": 389, "x2": 1382, "y2": 734, "width": 576, "height": 345, "valid": True, "fallback": None}
    finally:
        srv._detach_log_handler()


def test_crop_helper_matches_named_crops():
    srv = _synthetic_server()
    try:
        frame = srv._build_synthetic_telemetry_frame()
        mm1, r1 = srv._crop_minimap_bgr(frame, scale=0.5)
        mm2, r2 = srv._crop_norm_bounds_bgr(frame, ms.MINIMAP_NORM_BOUNDS, scale=0.5, fallback_size=(280, 280))
        assert r1 == r2 and np.array_equal(mm1, mm2)
        gr1, r3 = srv._crop_gather_roi_bgr(frame, scale=2.0)
        gr2, r4 = srv._crop_norm_bounds_bgr(frame, ms.GATHER_ROI_NORM_BOUNDS, scale=2.0, fallback_size=(360, 240))
        assert r3 == r4 and np.array_equal(gr1, gr2)
    finally:
        srv._detach_log_handler()


# --- (E) token yazma tekilleştirmesi ------------------------------------------

def test_rotate_access_token_writes_0600_file(tmp_path, monkeypatch):
    monkeypatch.setattr(ms, "PROJECT_ROOT", str(tmp_path))
    srv = MobileRemoteServer(access_token=TOKEN32)
    try:
        new_token = srv.rotate_access_token()
        assert len(new_token) == 32 and new_token.isalnum()
        assert new_token != TOKEN32
        tok_file = tmp_path / "config" / "mobile_token.txt"
        assert tok_file.exists()
        assert tok_file.read_text(encoding="utf-8") == new_token
        assert oct(os.stat(tok_file).st_mode & 0o777) == "0o600"
        assert srv._write_token_file("X" * 32, "test") is True
    finally:
        srv._detach_log_handler()
