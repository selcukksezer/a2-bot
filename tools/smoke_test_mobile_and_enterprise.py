# -*- coding: utf-8 -*-
"""
Gatherion v2.0 — Mobil Kumanda (PWA/SSE/PiP), OPSEC/Performans (Madde 31–50)
ve Kurumsal Yeni Nesil Modüller (Bölüm 2–5) Otomatik Duman Testi (Smoke Test) Paketi.

Kapsam:
1. Madde 31–40:
   - MobileRemoteServer (/api/status, /api/nodes, /api/frame.jpg?mode=full|minimap|pip,
     /api/pip_bundle, /api/stream SSE, /manifest.json, /sw.js, POST /api/action:
     pause, resume, record_f8, reset_node, quarantine_node, delete_node,
     tap_to_move_map, update_settings)
   - ui/mobile_dashboard.html (PWA, Web Audio / Vibration Alarm, Sanal Joystick, Bölge Seçici)
   - core/mobile_tunnel.py (generate_qr_image, NtfyCloudRelay, FreeCloudTunnelManager)
   - ui/mobile_connect_dialog.py (MobileConnectDialog)
   - core/incident_recorder.py (MobileTokenAuthenticator 256-Bit QR Token, IncidentRecorder)
2. Madde 41–50:
   - core/capture.py (ZeroCopyFrameRingBuffer, ScreenCapture.get_shared_frame_view, grab_roi_view)
   - core/stealth_capture.py (DXGIDesktopCapture, AdaptiveFramePacer, StealthOverlayManager)
   - core/telemetry.py (YieldHeatmapTracker, InventoryFullnessEstimator 61/75, TelemetryEngine)
   - core/skill_bar_reader.py (SkillBarReader, MicroCameraLookAround)
   - core/dpi_focus_guard.py (RouteBackupManager, DPIScalingSynchronizer, ForegroundFocusGuard, SelfHealingTemplateTuner)
   - core/dry_run_simulator.py (DryRunReplaySimulator 8x Rota Denetimi)
3. Kurumsal Yeni Nesil Modüller (Bölüm 2, 3, 4, 5):
   - core/economy_broker.py (TradeBrokerPriceScanner, AutoCraftingMorphQueue, SessionProfitSimulator)
   - core/neural_audio_perception.py (ONNXYolo3DDetector, AudioLoopbackFFTAnalyzer: Kristal Çınlaması, Kırılma Sesi, Canavar Ayak Sesi)
   - core/multi_char_scheduler.py (AltCharacterManager, AutomationTaskScheduler, PetAndBuffManager)
   - core/depth_glide_navigator.py (MonocularDepthObstacleAvoidance, GlideWaypointController)
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from version import __version__

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class _SmokeMockNode:
    """Mobil sunucu ve rota testleri için hafif sahte düğüm (Node) nesnesi."""

    def __init__(
        self,
        node_id: int,
        x: int = 960,
        y: int = 540,
        state: str = "active",
        source_type: str = "manual",
        resource_type: str = "Orichalcum Ore",
    ) -> None:
        self.id: int = int(node_id)
        self.x: int = int(x)
        self.y: int = int(y)
        self.map_x: int = int(x)
        self.map_y: int = int(y)
        self.norm_x: float = round(float(x) / 1920.0, 4)
        self.norm_y: float = round(float(y) / 1080.0, 4)
        self.state: str = str(state)
        self.source_type: str = str(source_type)
        self.resource_type: str = str(resource_type)
        self.region: str = "Altgard"
        self.current_harvests: int = 1
        self.max_harvests: int = 3
        self.consecutive_failures: int = 0
        self.rest_until: float = 0.0
        self.cooldown_until: float = 0.0
        self.last_harvest_ts: float = 0.0
        self.last_visited_timestamp: float = 0.0
        self.rest_duration_sec: int = 300

    def is_ready(self, now: Optional[float] = None) -> bool:
        current = float(now) if now is not None else time.time()
        return self.state == "active" and current >= self.rest_until

    def reset_cooldown(self) -> None:
        self.state = "active"
        self.rest_until = 0.0
        self.cooldown_until = 0.0
        self.current_harvests = 0
        self.consecutive_failures = 0


class _SmokeMockRouteManager:
    """Mobil sunucu ve ekonomi testleri için izole sahte RouteManager."""

    def __init__(self) -> None:
        self.active_region: str = "Altgard"
        self.active_route: str = "Dredgion Crash Site"
        self.total_gathered: int = 42
        self.meters_per_pixel: float = 0.28
        self.saved_count: int = 0
        self._nodes: List[_SmokeMockNode] = [
            _SmokeMockNode(1, 520, 360, state="active", source_type="manual", resource_type="Orichalcum Ore"),
            _SmokeMockNode(2, 860, 480, state="resting", source_type="minimap", resource_type="Aria"),
            _SmokeMockNode(3, 1240, 620, state="active", source_type="3d", resource_type="Pure Aether"),
            _SmokeMockNode(4, 1480, 720, state="active", source_type="manual", resource_type="Orichalcum Ore"),
        ]

    def get_route_nodes(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
    ) -> List[_SmokeMockNode]:
        _ = (region, route)
        return list(self._nodes)

    def get_node(
        self,
        node_id: int,
        region: Optional[str] = None,
    ) -> Optional[_SmokeMockNode]:
        _ = region
        target = int(node_id)
        for n in self._nodes:
            if n.id == target:
                return n
        return None

    def save_region(self, region: Optional[str] = None) -> bool:
        _ = region
        self.saved_count += 1
        return True

    def update_resting_nodes(self, region: Optional[str] = None) -> int:
        _ = region
        return 0

    def record_current_node(
        self,
        vision_engine: Any = None,
        input_ctrl: Any = None,
    ) -> _SmokeMockNode:
        _ = (vision_engine, input_ctrl)
        new_id = max((n.id for n in self._nodes), default=0) + 1
        node = _SmokeMockNode(new_id, 990, 510, state="active", source_type="manual")
        self._nodes.append(node)
        return node

    def delete_node(
        self,
        node_id: int,
        region: Optional[str] = None,
        route: Optional[str] = None,
    ) -> bool:
        _ = (region, route)
        before = len(self._nodes)
        self._nodes = [n for n in self._nodes if n.id != int(node_id)]
        return len(self._nodes) < before

    def remove_node(self, node_id: int) -> bool:
        return self.delete_node(node_id)

    def reset_all_nodes(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
    ) -> int:
        _ = (region, route)
        for n in self._nodes:
            n.reset_cooldown()
        return len(self._nodes)

    def save(self) -> bool:
        self.saved_count += 1
        return True

    def get_regions(self) -> List[str]:
        return ["Altgard", "Morheim", "Beluslan"]

    def get_routes(self, region: Optional[str] = None) -> List[str]:
        reg = str(region or self.active_region)
        if reg == "Morheim":
            return ["Desert Oasis", "Ice Claw Village"]
        return ["Dredgion Crash Site", "Pilgrims Ridge"]

    def set_active_region_route(self, region: str, route: str) -> bool:
        self.active_region = str(region)
        self.active_route = str(route)
        return True


class _SmokeMockGatheringSettings:
    """update_settings'in yazdığı gathering ayarları (F-1655 sahte nesnesi)."""

    def __init__(self) -> None:
        self.harvests_per_node: int = 3
        self.gather_key: str = "f"
        self.test_mode: bool = False


class _SmokeMockRouteSettings:
    """update_settings'in yazdığı route ayarları (F-1655 sahte nesnesi)."""

    def __init__(self) -> None:
        self.harvests_per_node: int = 3
        self.minimap_auto_record: bool = False
        self.hybrid_3d_and_minimap_record: bool = False
        self.rest_minutes: int = 5
        self.fight_back: bool = False


class _SmokeMockSettings:
    """Mobil sunucu update_settings aksiyonu için sahte settings nesnesi.

    F-1655 (D24, merged): settings nesnesi yokken update_settings ok:True
    dönmez — sunucuya gerçek (sahte) bir settings nesnesi verilmeli.
    """

    def __init__(self) -> None:
        self.gathering = _SmokeMockGatheringSettings()
        self.route = _SmokeMockRouteSettings()


class _SmokeMockWorker:
    """Mobil sunucu aksiyonlarını doğrulamak için hafif sahte BotWorker."""

    def __init__(self, route_mgr: _SmokeMockRouteManager) -> None:
        self.route_manager = route_mgr
        self.state: str = "WALKING"
        self.mode: str = "route"
        self.is_paused: bool = False
        self._pause_reason: str = ""
        self._session_start_time: float = time.time() - 180.0
        self.active_node: Optional[_SmokeMockNode] = route_mgr.get_route_nodes()[0]
        self.active_node_distance_m: int = 14
        self.last_tap_to_move_target: Optional[Dict[str, Any]] = None

    def pause(self, reason: str = "") -> None:
        self.is_paused = True
        self.state = "PAUSED"
        self._pause_reason = str(reason or "Duraklatıldı")

    def resume(self) -> None:
        self.is_paused = False
        self.state = "WALKING"
        self._pause_reason = ""

    def set_mode(self, mode: str) -> None:
        self.mode = str(mode)


class _SmokeMockInputCtrl:
    """Donanım/fare yan etkisi oluşturmayan izole sahte InputController."""

    def __init__(self) -> None:
        self.test_mode: bool = True
        self.screen_width: int = 1920
        self.screen_height: int = 1080
        self.pressed_keys: List[str] = []
        self.clicked_coords: List[Tuple[int, int, str]] = []
        self.cursor_x: int = 960
        self.cursor_y: int = 540
        self.moved_points: List[Tuple[int, int]] = []

    def set_test_mode(self, enabled: bool) -> None:
        self.test_mode = bool(enabled)

    def press_key(self, key: str, hold_sec: float = 0.05) -> float:
        self.pressed_keys.append(str(key))
        return float(hold_sec)

    def click_normalized(
        self,
        norm_x: float,
        norm_y: float,
        button: str = "left",
        double: bool = False,
    ) -> Tuple[int, int, float]:
        _ = double
        px = int(round(float(norm_x) * self.screen_width))
        py = int(round(float(norm_y) * self.screen_height))
        self.clicked_coords.append((px, py, str(button)))
        return px, py, 0.04

    def get_cursor_position(self) -> Tuple[int, int]:
        return self.cursor_x, self.cursor_y

    def move_to(self, x: int, y: int) -> None:
        self.cursor_x = int(x)
        self.cursor_y = int(y)
        self.moved_points.append((self.cursor_x, self.cursor_y))


def _load_reference_screenshot() -> np.ndarray:
    """
    `reference_screenshots/pilgrims_ridge_1.jpg` dosyasını Windows Türkçe/Unicode
    dosya yolu uyumlu (`np.fromfile` + `cv2.imdecode`) şekilde yükler.

    Referans kare yoksa/okunamazsa asla sessizce düz renkli sahte kareye
    düşmez — eksik referansla geçen bir duman testi, gerçek görüntüyü hiç
    görmeden geçmiş sayılır. Çağıran, hatayı başarısız test olarak kaydeder.
    """
    ref_path = os.path.join(PROJECT_ROOT, "reference_screenshots", "pilgrims_ridge_1.jpg")
    if not os.path.isfile(ref_path):
        raise FileNotFoundError(
            f"Referans ekran görüntüsü bulunamadı: {ref_path} — "
            "duman testi sahte kareyle devam etmeyecek."
        )
    try:
        raw_buf = np.fromfile(ref_path, dtype=np.uint8)
        img = cv2.imdecode(raw_buf, cv2.IMREAD_COLOR)
    except Exception as exc:
        raise RuntimeError(f"Referans ekran görüntüsü okunamadı ({ref_path}): {exc}") from exc
    if img is None or img.size == 0:
        raise RuntimeError(f"Referans ekran görüntüsü çözümlenemedi (boş/gecersiz): {ref_path}")
    return img


def _decode_jpeg_bytes(raw: bytes, what: str) -> np.ndarray:
    """Decode JPEG bytes and prove they are a real, non-blank image."""
    assert len(raw) > 512 and raw[:2] == b"\xff\xd8", f"Geçersiz JPEG verisi ({what})"
    buf = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    assert img is not None and img.size > 0 and img.ndim == 3, f"JPEG çözümlenemedi ({what})"
    assert float(img.std()) > 1.0, f"JPEG tek renk/boş kare ({what})"
    return img


def _http_get_status(url: str, headers: Optional[Dict[str, str]] = None, timeout: float = 5.0) -> int:
    """GET and return only the HTTP status (401/403 etc. do not raise)."""
    req = urllib.request.Request(url, method="GET", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return int(resp.status)
    except urllib.error.HTTPError as exc:
        return int(exc.code)


def _http_get_json(url: str, timeout: float = 5.0) -> Tuple[int, Dict[str, Any]]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(resp.status)
        raw = resp.read().decode("utf-8")
        return status, json.loads(raw)


def _http_get_bytes(url: str, timeout: float = 5.0) -> Tuple[int, bytes, str]:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(resp.status)
        ctype = str(resp.headers.get("Content-Type", ""))
        return status, resp.read(), ctype


def _http_post_json(url: str, payload: Dict[str, Any], timeout: float = 5.0) -> Tuple[int, Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = int(resp.status)
        raw = resp.read().decode("utf-8")
        return status, json.loads(raw)


def run_mobile_and_enterprise_smoke_tests(verbose: bool = True) -> Tuple[int, int, List[dict]]:
    """
    Madde 31–40, Madde 41–50 ve Kurumsal Yeni Nesil Modülleri (Bölüm 2–5)
    uçtan uca test eder ve `(passed_count, failed_count, results)` döndürür.
    """
    results: List[dict] = []

    def _run_case(test_id: str, category: str, title: str, fn: Callable[[], str]) -> None:
        t0 = time.perf_counter()
        try:
            detail = fn()
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            results.append(
                {
                    "id": test_id,
                    "category": category,
                    "name": title,
                    "passed": True,
                    "status": "GEÇTİ",
                    "elapsed_ms": elapsed_ms,
                    "detail": str(detail or "Başarılı"),
                }
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
            # Hata detayı tek kaynaktan: traceback'in son satırı zaten
            # "Tür: mesaj" içerir — eski `f"{exc} ({tb_short})"` biçimi
            # aynı mesajı iki kez yazıyordu (örn. "X (AssertionError: X)").
            tb_lines = traceback.format_exc().splitlines()
            tb_short = tb_lines[-1] if tb_lines else f"{type(exc).__name__}: {exc}"
            results.append(
                {
                    "id": test_id,
                    "category": category,
                    "name": title,
                    "passed": False,
                    "status": "KALDI",
                    "elapsed_ms": elapsed_ms,
                    "detail": tb_short,
                }
            )

    try:
        ref_frame = _load_reference_screenshot()
    except (FileNotFoundError, RuntimeError) as exc:
        results.append(
            {
                "id": "REF",
                "category": "Referans Görüntü",
                "name": "reference_screenshots/pilgrims_ridge_1.jpg yüklenmeli",
                "passed": False,
                "status": "KALDI",
                "elapsed_ms": 0.0,
                "detail": f"{exc} (sahte kareye düşülmedi)",
            }
        )
        if verbose:
            _print_turkish_report_table(0, 1, results)
        return 0, 1, results

    # =========================================================================
    # BÖLÜM 1: MADDE 31–40 (MOBİL UZAKTAN KUMANDA, PiP, SSE, PWA, QR TOKEN, INCIDENT)
    # =========================================================================

    def _test_mobile_server_endpoints_and_actions() -> str:
        from core.mobile_server import MobileRemoteServer

        mock_rm = _SmokeMockRouteManager()
        mock_worker = _SmokeMockWorker(mock_rm)
        mock_ic = _SmokeMockInputCtrl()
        mock_settings = _SmokeMockSettings()

        server = MobileRemoteServer(
            host="127.0.0.1",
            port=0,
            require_token=False,
            worker=mock_worker,
            route_mgr=mock_rm,
            input_ctrl=mock_ic,
            settings=mock_settings,
        )
        server.start()
        try:
            base_url = f"http://127.0.0.1:{server.port}"

            # 1-6. Bağımsız GET'ler (status, nodes, 3 frame modu, pip_bundle,
            # manifest, sw.js) birbirinin durumunu değiştirmez ve sunucu
            # ThreadingHTTPServer'dır: ThreadPool (4 işçi) ile paralel
            # çekilir, sonuçlar sabit sırada toplanır, assertion'lar aynen
            # korunur. SSE akışı ve durum değiştiren POST aksiyonları
            # sıralı kalır (POST'lar birbirine bağımlıdır).
            _get_jobs: List[Tuple[str, Callable[[], Any]]] = [
                ("status", lambda: _http_get_json(f"{base_url}/api/status")),
                ("nodes", lambda: _http_get_json(f"{base_url}/api/nodes")),
                ("frame_full", lambda: _http_get_bytes(f"{base_url}/api/frame.jpg?mode=full&quality=70")),
                ("frame_minimap", lambda: _http_get_bytes(f"{base_url}/api/frame.jpg?mode=minimap&quality=70")),
                ("frame_pip", lambda: _http_get_bytes(f"{base_url}/api/frame.jpg?mode=pip&quality=70")),
                ("pip_bundle", lambda: _http_get_json(f"{base_url}/api/pip_bundle?quality=70")),
                ("manifest", lambda: _http_get_bytes(f"{base_url}/manifest.json")),
                ("sw.js", lambda: _http_get_bytes(f"{base_url}/sw.js")),
            ]
            with ThreadPoolExecutor(max_workers=4) as _pool:
                _futures = [_pool.submit(_fn) for _, _fn in _get_jobs]
                _fetched = [_f.result() for _f in _futures]
            (
                (code_st, st_data),
                (code_nd, nd_data),
                (f_full_code, f_full_bytes, f_full_ctype),
                (f_mini_code, f_mini_bytes, f_mini_ctype),
                (f_pip_code, f_pip_bytes, f_pip_ctype),
                (code_pip, pip_data),
                (m_code, m_bytes, _m_ctype),
                (sw_code, sw_bytes, _sw_ctype),
            ) = _fetched

            # 1. GET /api/status
            assert code_st == 200 and st_data.get("ok") is True, f"/api/status hatası: {st_data}"
            route_info = st_data.get("route", {})
            assert route_info.get("region_name") == "Altgard", f"Bölge adı eşleşmedi: {route_info}"

            # 2. GET /api/nodes
            nodes_list = nd_data if isinstance(nd_data, list) else nd_data.get("nodes", [])
            assert code_nd == 200 and len(nodes_list) == 4, f"/api/nodes hatası: {nd_data}"

            # 3. GET /api/frame.jpg?mode=full, mode=minimap, mode=pip
            # Sihirli bayt + uzunluk yetmez: JPEG gerçekten çözümlenmeli,
            # boyutlu ve tek renk/boş olmayan bir görüntü olmalı.
            expected_shapes = {"full": (540, 960, 3)}
            for fmode, (f_code, f_bytes, f_ctype) in (
                ("full", (f_full_code, f_full_bytes, f_full_ctype)),
                ("minimap", (f_mini_code, f_mini_bytes, f_mini_ctype)),
                ("pip", (f_pip_code, f_pip_bytes, f_pip_ctype)),
            ):
                assert f_code == 200 and "image/jpeg" in f_ctype, f"JPEG içerik tipi hatalı ({fmode}): {f_ctype}"
                decoded = _decode_jpeg_bytes(f_bytes, fmode)
                if fmode in expected_shapes:
                    assert decoded.shape == expected_shapes[fmode], (
                        f"JPEG boyutu hatalı ({fmode}): {decoded.shape} != {expected_shapes[fmode]}"
                    )

            # 4. GET /api/pip_bundle — base64 alanlar da çözümlenip doğrulanmalı.
            assert code_pip == 200 and pip_data.get("ok") is True, "/api/pip_bundle başarısız"
            assert len(pip_data.get("minimap_b64", "")) > 100, "minimap_b64 boş"
            assert len(pip_data.get("gather_roi_b64", "")) > 100, "gather_roi_b64 boş"
            for _b64_key in ("minimap_b64", "gather_roi_b64", "pip_b64"):
                _b64 = pip_data.get(_b64_key, "")
                if not _b64:
                    continue
                _decoded = _decode_jpeg_bytes(base64.b64decode(_b64), f"pip_bundle.{_b64_key}")

            # 6. GET /manifest.json & /sw.js (PWA)
            manifest_obj = json.loads(m_bytes.decode("utf-8"))
            assert m_code == 200 and manifest_obj.get("display") == "standalone", "PWA manifest hatalı"

            sw_text = sw_bytes.decode("utf-8")
            assert sw_code == 200 and "addEventListener" in sw_text, "PWA Service Worker hatalı"

            # 5. GET /api/stream (SSE <50ms Push) — data satırları geçerli JSON olmalı.
            sse_code, sse_bytes, sse_ctype = _http_get_bytes(
                f"{base_url}/api/stream?max_events=2&interval_ms=25",
                timeout=5.0,
            )
            sse_text = sse_bytes.decode("utf-8", errors="replace")
            assert sse_code == 200 and "text/event-stream" in sse_ctype, f"SSE başlığı hatalı: {sse_ctype}"
            assert "data:" in sse_text, "SSE akışında 'data:' satırı bulunamadı"
            sse_data_lines = [ln[5:].strip() for ln in sse_text.splitlines() if ln.startswith("data:")]
            assert len(sse_data_lines) >= 2, f"SSE'de en az 2 data satırı bekleniyordu: {len(sse_data_lines)}"
            for _payload in sse_data_lines:
                _obj = json.loads(_payload)
                assert isinstance(_obj, dict) and _obj, f"SSE data yükü JSON obje değil: {_payload[:120]}"

            # 7. POST /api/action -> pause, resume, record_f8, reset_node, quarantine_node, delete_node, tap_to_move_map, update_settings
            _, act_pause = _http_post_json(f"{base_url}/api/action", {"action": "pause", "reason": "SmokeTest"})
            assert act_pause.get("ok") is True and mock_worker.is_paused is True, "pause aksiyonu başarısız"

            _, act_resume = _http_post_json(f"{base_url}/api/action", {"action": "resume"})
            assert act_resume.get("ok") is True and mock_worker.is_paused is False, "resume aksiyonu başarısız"

            _, act_f8 = _http_post_json(f"{base_url}/api/action", {"action": "record_f8"})
            assert act_f8.get("ok") is True and len(mock_rm.get_route_nodes()) == 5, "record_f8 başarısız"

            _, act_reset = _http_post_json(f"{base_url}/api/action", {"action": "reset_node", "node_id": 2})
            assert act_reset.get("ok") is True, "reset_node başarısız"

            _, act_quar = _http_post_json(f"{base_url}/api/action", {"action": "quarantine_node", "node_id": 3})
            assert act_quar.get("ok") is True, "quarantine_node başarısız"

            _, act_del = _http_post_json(f"{base_url}/api/action", {"action": "delete_node", "node_id": 4})
            assert act_del.get("ok") is True, "delete_node başarısız"

            _, act_tap = _http_post_json(
                f"{base_url}/api/action",
                {"action": "tap_to_move_map", "map_x": 1120, "map_y": 640},
            )
            assert act_tap.get("ok") is True and act_tap.get("distance_m", 0) > 0, "tap_to_move_map başarısız"

            _, act_cfg = _http_post_json(
                f"{base_url}/api/action",
                {"action": "update_settings", "harvests_per_node": 4, "test_mode": True},
            )
            assert act_cfg.get("ok") is True, "update_settings başarısız"
            # [O10] F-1655 (D24, merged): ayar yalnızca settings nesnesi
            # varsa uygulanır — değerin gerçekten işlendiği kanıtlanır.
            assert mock_settings.gathering.harvests_per_node == 4, (
                "harvests_per_node ayara işlenmedi"
            )
            # v19-smoke: uzaktan test_mode BİLEREK reddedilir (O-1, dilim 22:
            # uzaktan test_mode donanım güven kapısını baypas ederdi —
            # test modunda soft SendInput "güvenilir" sayılırdı). Üretim
            # davranışı değişmez; test reddi doğrular.
            assert act_cfg.get("rejected", {}).get("test_mode"), (
                "uzaktan test_mode reddedilen alanlarda bildirilmeli"
            )
            assert mock_settings.gathering.test_mode is False, (
                "uzaktan test_mode ayara işlenmemeli (O-1: donanım güven kapısı)"
            )

            return f"Port={server.port}, Status/Nodes/PiP/SSE/PWA + 8 POST Aksiyon doğrulandı"
        finally:
            server.stop()

    def _test_mobile_server_token_auth() -> str:
        # Kimlik doğrulama kapalı mock sunucu tek başına yeterli değil:
        # gerçek token zorunluluğu ayrı bir sunucuda uçtan uca kanıtlanır.
        from core.mobile_server import MobileRemoteServer

        token = "SmokeTestTok1234"  # 16 karakter: D24 token politikası (F-1652) en az 16 ister; sunucu açık tokenı aynen alır
        auth_server = MobileRemoteServer(
            host="127.0.0.1",
            port=0,
            require_token=True,
            access_token=token,
            worker=_SmokeMockWorker(_SmokeMockRouteManager()),
            route_mgr=_SmokeMockRouteManager(),
            input_ctrl=_SmokeMockInputCtrl(),
        )
        assert auth_server.access_token == token, "Sunucu açık tokenı aynen almalı"
        auth_server.start()
        try:
            auth_base = f"http://127.0.0.1:{auth_server.port}"
            assert _http_get_status(f"{auth_base}/api/status") == 401, "Tokensuz /api/status 401 vermeli"
            assert _http_get_status(f"{auth_base}/api/status?token=wrongtoken") == 401, "Yanlış token 401 vermeli"
            assert _http_get_status(f"{auth_base}/api/status?token={token}") == 200, "Doğru token (query) 200 vermeli"
            assert _http_get_status(
                f"{auth_base}/api/status", headers={"Authorization": f"Bearer {token}"}
            ) == 200, "Doğru token (Bearer) 200 vermeli"
            code_ok, ok_data = _http_get_json(f"{auth_base}/api/status?token={token}")
            assert code_ok == 200 and ok_data.get("ok") is True, f"Tokenlı /api/status hatası: {ok_data}"
            return "Token zorunluluğu: tokensuz/yanlış=401, doğru token (query+Bearer)=200"
        finally:
            auth_server.stop()

    _run_case(
        "M31-35",
        "Mobil Kumanda & PWA/SSE/PiP",
        "MobileRemoteServer (/api/status, /api/nodes, /api/frame.jpg, /api/pip_bundle, /api/stream, POST /api/action)",
        _test_mobile_server_endpoints_and_actions,
    )

    _run_case(
        "M31-AUTH",
        "Mobil Kumanda & PWA/SSE/PiP",
        "MobileRemoteServer token zorunluluğu (require_token=True: 401/200)",
        _test_mobile_server_token_auth,
    )

    def _test_mobile_dashboard_html_features() -> str:
        from core.mobile_server import MobileRemoteServer

        html_path = os.path.join(PROJECT_ROOT, "ui", "mobile_dashboard.html")
        assert os.path.isfile(html_path), "ui/mobile_dashboard.html bulunamadı"
        with open(html_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        server = MobileRemoteServer(host="127.0.0.1", port=0, require_token=False)
        served_html = server.load_dashboard_html()
        server._detach_log_handler()

        assert len(raw_content) > 5000, "mobile_dashboard.html beklenenden kısa"
        assert "/api/action" in served_html, "mobile_dashboard.html içinde /api/action eksik"
        assert "manifest.json" in served_html and "serviceWorker" in served_html, "PWA manifest/SW enjeksiyonu eksik"
        assert (
            "AudioContext" in served_html
            or "webkitAudioContext" in served_html
            or "vibrate" in served_html
        ), "Web Audio / Titreşim alarm desteği eksik"
        # Alt dize araması yetmez: HTML gerçekten ayrıştırılabilmeli, PWA
        # manifest alanları dolu olmalı, SW fetch/cache mantığı içermeli.
        # (Not: JS burada çalıştırılmıyor — statik doğrulama + sunulan
        # manifest/sw.js içerik denetimi; tarayıcıda çalıştırma bu testin
        # dışında kalır ve raporda böyle belirtilir.)
        class _TagCollector(HTMLParser):
            def __init__(self) -> None:
                super().__init__()
                self.tags: List[str] = []
                self.errors: List[str] = []

            def handle_starttag(self, tag: str, attrs: list) -> None:
                self.tags.append(tag)

            def error(self, message: str) -> None:  # noqa: D102
                self.errors.append(message)

        collector = _TagCollector()
        collector.feed(served_html)
        assert not collector.errors, f"mobile_dashboard.html ayrıştırma hatası: {collector.errors}"
        for _needed in ("html", "head", "body", "script"):
            assert _needed in collector.tags, f"mobile_dashboard.html içinde <{_needed}> yok"
        assert "serviceWorker" in served_html and "register" in served_html, (
            "Service Worker kaydı (navigator.serviceWorker.register) eksik"
        )
        assert "fetch('/api/status')" in served_html or 'fetch("/api/status")' in served_html or "withTokenUrl('/api/status')" in served_html, (
            "Panel /api/status çekmiyor — canlı veri akışı yok"
        )
        # manifest/sw.js, M31-35'te doğrulanan uçlarla aynı sunucudan
        # alınır: ikinci bir MobileRemoteServer kurmak (önceki server2)
        # fazladan bir port bind + start/stop döngüsüydü.
        server.start()
        try:
            m_url = f"http://127.0.0.1:{server.port}"
            m_code, m_bytes, _ = _http_get_bytes(f"{m_url}/manifest.json")
            manifest = json.loads(m_bytes.decode("utf-8"))
            assert m_code == 200 and manifest.get("display") == "standalone", "PWA manifest hatalı"
            assert manifest.get("name") and manifest.get("short_name"), "Manifest name/short_name eksik"
            assert manifest.get("start_url"), "Manifest start_url eksik"
            assert manifest.get("icons"), "Manifest icons eksik"
            sw_code, sw_bytes, _ = _http_get_bytes(f"{m_url}/sw.js")
            sw_text = sw_bytes.decode("utf-8")
            assert sw_code == 200 and "addEventListener" in sw_text, "PWA Service Worker hatalı"
            assert "fetch" in sw_text and ("caches" in sw_text or "Cache" in sw_text), (
                "Service Worker fetch/cache mantığı eksik"
            )
        finally:
            server.stop()
        return f"ui/mobile_dashboard.html ({len(served_html)} bayt) PWA, Audio/Vibration ve Radar kontrolleri doğrulandı (JS çalıştırılmadan statik + sunulan manifest/SW denetimi)"

    _run_case(
        "M36-37",
        "Mobil PWA Arayüzü",
        "ui/mobile_dashboard.html (PWA, Web Audio/Titreşim Alarmı, Harita & Müdahale Kontrolleri)",
        _test_mobile_dashboard_html_features,
    )

    def _test_mobile_tunnel_and_connect_dialog() -> str:
        from core.mobile_tunnel import (
            FreeCloudTunnelManager,
            NtfyCloudRelay,
            generate_qr_data_uri,
            generate_qr_image,
            get_lan_ip,
            get_lan_url,
        )
        from ui.mobile_connect_dialog import MobileConnectDialog

        with tempfile.TemporaryDirectory() as tmp_dir:
            lan_ip = get_lan_ip()
            lan_url = get_lan_url(8765)
            assert lan_ip and lan_url.startswith("http://"), f"Geçersiz LAN URL: {lan_url}"

            qr_bgr = generate_qr_image(f"{lan_url}/?token=ABCDEF12", size=220)
            assert isinstance(qr_bgr, np.ndarray) and qr_bgr.shape == (220, 220, 3), f"QR boyutu hatalı: {qr_bgr.shape}"

            qr_uri = generate_qr_data_uri(lan_url, size=180)
            assert qr_uri.startswith("data:image/png;base64,"), "QR data URI üretilemedi"

            cfg_path = Path(tmp_dir) / "ntfy_test.json"
            relay = NtfyCloudRelay(config_path=cfg_path)
            relay.enabled = False
            parsed_cmd = relay.parse_remote_command("durdur")
            assert parsed_cmd is not None and parsed_cmd[0] == "pause", f"Ntfy komut ayrıştırma hatası: {parsed_cmd}"

            tunnel_mgr = FreeCloudTunnelManager(local_port=8765)
            assert tunnel_mgr.local_port == 8765, "FreeCloudTunnelManager port bilgisi hatalı"

            # Tünel hiç başlatılmadan geçiliyordu: cloudflared indirmeden
            # (ağ yok) başlatmayı gerçekten dene. Binary yoksa başlatma
            # None dönmeli, gerekçeli hata kaydetmeli ve durum "çalışmıyor"
            # demeli — sessiz geçme yok. Binary varsa URL beklenir.
            assert tunnel_mgr.find_cloudflared_binary() is None or isinstance(
                tunnel_mgr.find_cloudflared_binary(), str
            )
            tunnel_url = tunnel_mgr.start_cloudflare_tunnel(
                local_port=8765, auto_download=False, timeout_sec=5.0
            )
            tunnel_status = tunnel_mgr.get_status()
            if tunnel_mgr.find_cloudflared_binary() is None:
                assert tunnel_url is None, f"Binary yokken tünel URL dönmemeli: {tunnel_url}"
                assert tunnel_status.get("last_error"), "Binary yokken gerekçeli hata kaydedilmeli"
                assert tunnel_mgr.is_tunnel_running() is False, "Binary yokken tünel çalışıyor görünmemeli"
                tunnel_note = f"tünel başlatılamadı (beklenen): {tunnel_status.get('last_error')}"
            else:
                assert tunnel_url, "cloudflared varken tünel URL üretmeli"
                tunnel_note = f"tünel URL={tunnel_url}"
                tunnel_mgr.stop_tunnel()

            # MobileConnectDialog da gerçekten kurulmalı (offscreen Qt):
            # QR URL'i tünel/LAN hedefinden üretildiğini kanıtla.
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PyQt6.QtWidgets import QApplication

            app = QApplication.instance() or QApplication([])
            dialog = MobileConnectDialog(tunnel_mgr=tunnel_mgr)
            assert hasattr(MobileConnectDialog, "get_active_qr_url"), "MobileConnectDialog.get_active_qr_url eksik"
            dialog_qr = dialog.get_active_qr_url()
            assert dialog_qr.startswith("http://") and "8765" in dialog_qr, (
                f"Dialog QR URL LAN tabanlı olmalı: {dialog_qr}"
            )
            return f"LAN={lan_url}, QR=220x220x3 BGR, DataURI={len(qr_uri)} krk, NtfyTopic={relay.topic}, {tunnel_note}, DialogQR={dialog_qr}"

    _run_case(
        "M38",
        "Mobil Tünel & QR",
        "core/mobile_tunnel.py (generate_qr_image, NtfyCloudRelay, FreeCloudTunnelManager) & MobileConnectDialog",
        _test_mobile_tunnel_and_connect_dialog,
    )

    def _test_qr_token_auth_and_incident_recorder() -> str:
        from core.incident_recorder import IncidentRecorder, MobileTokenAuthenticator

        with tempfile.TemporaryDirectory() as tmp_dir:
            tok_path = os.path.join(tmp_dir, "mobile_token.txt")
            auth = MobileTokenAuthenticator(token_path=tok_path, require_token=True)
            assert len(auth.token) == 64, f"256-bit token 64 hex karakter olmalı: {len(auth.token)}"

            qr_url = auth.build_qr_authenticated_url("http://192.168.1.25:8765/")
            assert f"token={auth.token}" in qr_url, "QR URL içinde token parametresi eksik"
            assert auth.verify_request(query_string=f"token={auth.token}") is True, "Geçerli token reddedildi"
            assert auth.verify_request(query_string="token=invalid_token") is False, "Geçersiz token kabul edildi"

            inc_dir = os.path.join(tmp_dir, "incidents")
            recorder = IncidentRecorder(incidents_dir=inc_dir, max_incidents=10, cooldown_sec=0.0)
            ev1 = recorder.record_death(frame=ref_frame, region="Altgard", route="Pilgrims Ridge", node_id=3)
            ev2 = recorder.record_stuck(frame=ref_frame, region="Altgard", route="Pilgrims Ridge", node_id=4)
            ev3 = recorder.record_prolonged_combat(frame=ref_frame, duration_sec=135.0)
            ev4 = recorder.record_inventory_full(frame=ref_frame, fill_ratio=0.98)

            assert ev1 is not None and ev2 is not None and ev3 is not None and ev4 is not None, "Olay kaydı eksik"
            incidents = recorder.get_incidents()
            assert len(incidents) == 4, f"4 olay bekleniyordu, {len(incidents)} bulundu"

            jpg_bytes = recorder.get_incident_jpeg_bytes(ev1["id"])
            assert jpg_bytes is not None and jpg_bytes[:2] == b"\xff\xd8", "Olay JPEG görüntüsü okunamadı"

            cleared = recorder.clear_incidents()
            assert cleared == 4 and len(recorder.get_incidents()) == 0, "Olay geçmişi temizlenemedi"

            # P1-9 (K-08): rapor/konsol çıktısına token parçası (ilk 10
            # hex karakter dahil) yazılmaz; yalnızca SHA-256 parmak izi
            # verilir — varlık/yokluk + hash özeti yeterlidir.
            return f"256-Bit Token (parmak izi: {auth.token_fingerprint}), 4 Olay (Death/Stuck/Combat/Inventory) JPEG doğrulandı"

    _run_case(
        "M39-40",
        "256-Bit Token & Olay Kaydedici",
        "core/incident_recorder.py (MobileTokenAuthenticator & IncidentRecorder 4 Olay Anlık Görüntüsü)",
        _test_qr_token_auth_and_incident_recorder,
    )

    # =========================================================================
    # BÖLÜM 2: MADDE 41–50 (ZERO-COPY, HEATMAP, 61/75 ÇANTA, SKILL BAR, DPI, 8X DRY-RUN)
    # =========================================================================

    def _test_zero_copy_capture_and_stealth() -> str:
        from core.capture import ScreenCapture, ZeroCopyFrameRingBuffer
        from core.stealth_capture import AdaptiveFramePacer, DXGIDesktopCapture, StealthOverlayManager

        ring = ZeroCopyFrameRingBuffer(height=1080, width=1920, channels=3)
        view0 = ring.write_frame(ref_frame)
        view1 = ring.get_shared_frame_view()
        roi_view = ring.grab_roi_view(200, 150, 600, 450)

        assert view0.flags.writeable is False and view1.flags.writeable is False, "Paylaşımlı görünüm salt-okunur olmalı"
        assert np.shares_memory(view0, view1), "get_shared_frame_view sıfır-kopya (zero-copy) belleği paylaşmıyor"
        assert np.shares_memory(view1, roi_view), "grab_roi_view ana halka tamponu ile bellek paylaşmıyor"
        assert roi_view.shape == (300, 400, 3), f"ROI boyutu hatalı: {roi_view.shape}"

        cap = ScreenCapture()
        cap.ring_buffer.write_frame(ref_frame)
        cap_view = cap.get_shared_frame_view()
        cap_roi = cap.grab_roi_view(100, 100, 300, 300)
        assert np.shares_memory(cap_view, cap_roi), "ScreenCapture zero-copy ROI bellek paylaşımı başarısız"

        dxgi = DXGIDesktopCapture(monitor_idx=1, prefer_dxcam=True)
        pacer = AdaptiveFramePacer(seed=42)
        dt_walk = pacer.get_next_sleep_interval("WALKING")
        dt_gather = pacer.get_next_sleep_interval("GATHERING")
        assert 0.02 <= dt_gather < dt_walk <= 0.20, (
            f"AdaptiveFramePacer uyku aralığı hatalı: walk={dt_walk}, gather={dt_gather}"
        )
        stealth_mgr = StealthOverlayManager(deterministic_index=0)
        safe_title = stealth_mgr.select_safe_identity()
        assert StealthOverlayManager.is_title_safe(safe_title), f"Güvensiz pencere başlığı: {safe_title}"
        _ = dxgi.monitor_geometry

        return f"Zero-Copy RingBuffer shares_memory=True, ROI={roi_view.shape}, AdaptiveDt={dt_walk:.3f}s/{dt_gather:.3f}s"

    _run_case(
        "M41",
        "Zero-Copy & Stealth Yakalama",
        "core/capture.py (ZeroCopyFrameRingBuffer) & core/stealth_capture.py (DXGI / AdaptiveFramePacer)",
        _test_zero_copy_capture_and_stealth,
    )

    def _test_telemetry_heatmap_and_inventory_61_75() -> str:
        from core.telemetry import InventoryFullnessEstimator, TelemetryEngine, YieldHeatmapTracker

        hm = YieldHeatmapTracker(cluster_radius=0.025)
        hm.record_node_yield_event(node_id=1, x=520, y=360, success=True)
        hm.record_node_yield_event(node_id=2, x=860, y=480, success=False)
        hm.record_combat_ambush_event(x=870, y=490)

        pts = hm.get_heatmap_points()
        assert len(pts) >= 2, f"Isı haritası noktaları eksik: {len(pts)}"

        inv = InventoryFullnessEstimator(default_max_slots=75, near_full_threshold_pct=95.0)
        parsed_slots = inv.read_bag_counter_from_frame(ref_frame)
        assert parsed_slots == (61, 75), f"pilgrims_ridge_1.jpg üzerinde (61, 75) bekleniyordu, {parsed_slots} okundu"

        st_initial = inv.get_status()
        assert st_initial["used_slots"] == 61 and st_initial["max_slots"] == 75, "Çanta durumu 61/75 değil"
        assert st_initial["inventory_near_full"] is False, "61/75 dolulukta near_full False olmalı"

        inv.set_bag_slots(72, 75)
        st_full = inv.get_status()
        assert st_full["inventory_near_full"] is True, "72/75 (%96) dolulukta inventory_near_full True olmalı"

        tel = TelemetryEngine()
        tel.update_inventory_from_frame(ref_frame)
        assert tel.inventory_estimator.used_slots == 61, "TelemetryEngine çanta senkronizasyonu başarısız"

        return f"Isı Haritası={len(pts)} küme, HUD Çanta Sayacı={parsed_slots[0]}/{parsed_slots[1]}, 72/75 NearFull=True"

    _run_case(
        "M42-43",
        "Isı Haritası & Çanta Sayacı",
        "core/telemetry.py (YieldHeatmapTracker & InventoryFullnessEstimator 61/75 HUD Okuma)",
        _test_telemetry_heatmap_and_inventory_61_75,
    )

    def _test_skill_bar_reader_and_micro_camera_sway() -> str:
        from core.skill_bar_reader import MicroCameraLookAround, SkillBarReader

        reader = SkillBarReader()
        slots = reader.inspect_skill_bar(ref_frame)
        # v19-smoke: bar 12 slotludur (Sweep G2, bb91a71: 1..8, Q, E, R, T);
        # eski 8'li beklenti bayattı.
        assert len(slots) == 12, f"12 yetenek slotu (1..8, Q, E, R, T) bekleniyordu, {len(slots)} bulundu"

        best_skill = reader.select_best_ready_skill(ref_frame)
        assert best_skill in SkillBarReader.SKILL_SLOTS, f"Geçersiz yetenek seçimi: {best_skill}"

        sway_ctrl = MicroCameraLookAround(seed=42)
        interval = sway_ctrl.sample_next_interval()
        assert 15.0 <= interval <= 25.0, f"Kamera salınım aralığı [15, 25] sn dışında: {interval}"

        mock_ic = _SmokeMockInputCtrl()
        sway_res = sway_ctrl.perform_micro_camera_sway(mock_ic, direction=1, angle_deg=8.0, force=True)
        assert sway_res.get("triggered") is True, "MicroCameraLookAround tetiklenmedi"
        assert sway_res.get("start_pos") == sway_res.get("end_pos"), "Kamera salınımı başlangıç açısına geri dönmedi"
        assert sway_res.get("total_micro_steps", 0) >= 4, "WindMouse mikro adım sayısı yetersiz"

        return (
            f"12 Slot Okundu (Seçilen Hazır Yetenek='{best_skill}'), "
            f"Kamera Salınımı={sway_res['angle_deg']}° ({sway_res['total_micro_steps']} mikro adım)"
        )

    _run_case(
        "M44-45",
        "Yetenek Barı & Kamera Salınımı",
        "core/skill_bar_reader.py (SkillBarReader 12 Slot Analizi & MicroCameraLookAround ±8° Salınım)",
        _test_skill_bar_reader_and_micro_camera_sway,
    )

    def _test_dpi_focus_guard_backups_and_self_healing() -> str:
        from core.dpi_focus_guard import (
            DPIScalingSynchronizer,
            ForegroundFocusGuard,
            RouteBackupManager,
            SelfHealingTemplateTuner,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Madde 46: 10-Min Rolling Route Backups
            # v19-smoke: daemon kapalı kurulur — varsayılan açık daemon
            # başlangıçta hemen backup_all_regions() çalıştırır
            # (dpi_focus_guard.py _daemon_loop) ve testin yazdığı
            # region_data.json'u testin force yedeğinden ÖNCE yedekleyip
            # listeyi 2'ye çıkarabilir (tur-2'de 1/3 koşuda böyle kaldı).
            backup_mgr = RouteBackupManager(
                base_dir=tmp_dir,
                backup_interval_sec=600.0,
                max_snapshots=5,
                auto_start_daemon=False,
            )
            reg_json_path = backup_mgr.get_region_json_path("altgard")
            reg_json_path.parent.mkdir(parents=True, exist_ok=True)
            reg_json_path.write_text(
                json.dumps({"region": "altgard", "nodes": [{"id": 1, "x": 500, "y": 400}]}),
                encoding="utf-8",
            )

            b_path = backup_mgr.backup_region("altgard", force=True)
            assert b_path is not None and b_path.exists(), "Rota yedeği oluşturulamadı"
            assert len(backup_mgr.list_backups_for_region("altgard")) == 1, "Yedek listesi 1 eleman içermeli"

            # Bozuk JSON otomatik geri yükleme testi
            reg_json_path.write_text("{corrupted_json", encoding="utf-8")
            ok_restore, restored_data, _ = backup_mgr.ensure_region_integrity_or_restore("altgard")
            assert ok_restore is True and len(restored_data.get("nodes", [])) == 1, "Bozuk rota JSON geri yüklenemedi"

            # Madde 47: DPI Scaling Synchronizer (125% -> 1.25)
            dpi_sync = DPIScalingSynchronizer(override_scale_factor=1.25)
            phys_x, phys_y = dpi_sync.qt_to_physical_px(800, 400)
            qt_x, qt_y = dpi_sync.physical_to_qt_px(phys_x, phys_y)
            assert (phys_x, phys_y) == (1000, 500) and (qt_x, qt_y) == (800, 400), (
                f"DPI dönüşüm hatası: phys=({phys_x},{phys_y}), qt=({qt_x},{qt_y})"
            )

            # Madde 48: Foreground Focus Guard
            focus_guard = ForegroundFocusGuard()
            focus_guard.set_simulated_foreground_window("Discord - Genel Sohbet")
            focused_discord, _ = focus_guard.is_game_window_focused()
            allowed_discord, _ = focus_guard.guard_keystroke("f")
            assert focused_discord is False and allowed_discord is False, "Discord aktifken tuş vuruşu engellenmeli"

            focus_guard.set_simulated_foreground_window("AION 2 - Pilgrims Ridge")
            focused_aion, _ = focus_guard.is_game_window_focused()
            allowed_aion, _ = focus_guard.guard_keystroke("f")
            assert focused_aion is True and allowed_aion is True, "AION 2 aktifken tuş vuruşuna izin verilmeli"

            # Simülasyon temizlenip GERÇEK ön-plan sorgusu da çalıştırılmalı:
            # simüle pencereyle geçen test, gerçek Win32 yolunu hiç denemez.
            focus_guard.set_simulated_foreground_window(None)
            real_focused, real_reason = focus_guard.is_game_window_focused()
            assert isinstance(real_focused, bool) and isinstance(real_reason, str), (
                f"Gerçek odak sorgusu (bool, str) dönmeli: {(real_focused, real_reason)}"
            )
            if sys.platform != "win32":
                assert real_focused is False, "Win32 yokken gerçek odak False olmalı"

            # Madde 49: Self-Healing Template Tuner
            tuner = SelfHealingTemplateTuner()
            patch_crop = ref_frame[500:580, 920:1000].copy()
            patch_file = Path(tmp_dir) / "node_7_template.png"
            tuner.record_visit_score(7, 0.68, current_patch_bgr=patch_crop, patch_save_path=patch_file)
            tuner.record_visit_score(7, 0.70, current_patch_bgr=patch_crop, patch_save_path=patch_file)
            heal_res = tuner.record_visit_score(7, 0.69, current_patch_bgr=patch_crop, patch_save_path=patch_file)
            assert heal_res.get("refreshed") is True, f"3 sınırda skor sonrası kendi kendini iyileştirme çalışmadı: {heal_res}"

            return (
                f"Backup+AutoRestore=OK, DPI(125%)=800->1000px, FocusGuard(Discord=Blok/AION=İzin, "
                f"GerçekSorgu=({real_focused}, '{real_reason}')), "
                f"SelfHeal Eşik={heal_res.get('local_threshold')}"
            )

    _run_case(
        "M46-49",
        "Yedekleme, DPI, Odak & Self-Healing",
        "core/dpi_focus_guard.py (RouteBackupManager, DPIScalingSynchronizer, ForegroundFocusGuard, SelfHealingTemplateTuner)",
        _test_dpi_focus_guard_backups_and_self_healing,
    )

    def _test_dry_run_8x_replay_simulator() -> str:
        from core.dry_run_simulator import DryRunReplaySimulator

        sim = DryRunReplaySimulator(speed_multiplier=8.0, canvas_size=(640, 360))
        nodes = [
            {"id": 1, "x": 420, "y": 320, "state": "active"},
            {"id": 2, "x": 680, "y": 420, "state": "active"},
            {"id": 3, "x": 700, "y": 430, "state": "active"},
            {"id": 4, "x": 1150, "y": 640, "state": "active"},
        ]
        report = sim.run_8x_route_audit(nodes, execute_time_scaled_sleep=True, max_total_sleep_sec=0.25)
        assert report.get("node_count") == 4 and report.get("speed_multiplier") == 8.0, f"8x denetim hatası: {report}"
        assert report.get("keyboard_mouse_touched") is False, "Simülatör donanım girdisine dokunmamalı"
        assert report.get("simulated_8x_duration_sec", 999) < report.get("estimated_real_total_sec", 0), (
            "8x simülasyon süresi 1x süresinden kısa olmalı"
        )

        preview_bgr = sim.render_simulation_frame(nodes, background_bgr=ref_frame)
        assert isinstance(preview_bgr, np.ndarray) and preview_bgr.shape == (360, 640, 3), "Simülasyon karesi çizilemedi"

        return (
            f"8x Süre={report['simulated_8x_duration_sec']}sn (1x={report['estimated_real_total_sec']}sn), "
            f"Mesafe={report['total_distance_px']}px, Kesişim={report['self_intersection_count']}, "
            f"ÖlçekliUyku=Çalıştırıldı(max 0.25sn)"
        )

    _run_case(
        "M50",
        "8x Dry-Run Rota Simülatörü",
        "core/dry_run_simulator.py (DryRunReplaySimulator.run_8x_route_audit & render_simulation_frame)",
        _test_dry_run_8x_replay_simulator,
    )

    # =========================================================================
    # BÖLÜM 3: KURUMSAL YENİ NESİL MODÜLLER (BÖLÜM 2, 3, 4, 5)
    # =========================================================================

    def _test_economy_broker_and_profit_simulator() -> str:
        from core.economy_broker import (
            AutoCraftingMorphQueue,
            SessionProfitSimulator,
            TradeBrokerPriceScanner,
            format_kinah_tr,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = os.path.join(tmp_dir, "market_prices.json")
            scanner = TradeBrokerPriceScanner(price_db_path=db_path)
            # Sıfır satır okunması varsayılan fiyat sözlüğüyle geçmemeli:
            # tarama gerçekten satır okuduysa scan_count artar ve
            # price_history'ye yeni kayıt düşer. İkisi de kanıtlanmalı.
            scan_count_before = int(getattr(scanner, "scan_count", 0))
            history_before = sum(len(v) for v in getattr(scanner, "price_history", {}).values())
            prices = scanner.scan_broker_window(ref_frame)
            scan_count_after = int(getattr(scanner, "scan_count", 0))
            history_after = sum(len(v) for v in getattr(scanner, "price_history", {}).values())
            assert scan_count_after > scan_count_before, "scan_broker_window scan_count artırmadı"
            assert history_after > history_before, (
                f"Pazar taraması sıfır satır okudu (price_history artmadı: {history_before}->{history_after}) — "
                "varsayılan fiyatlarla geçilemez"
            )
            assert len(prices) >= 3 and all(v > 0 for v in prices.values()), f"Pazar fiyatları okunamadı: {prices}"

            mock_rm = _SmokeMockRouteManager()
            best_eval = scanner.select_most_profitable_route(mock_rm, region="Altgard", activate=True)
            assert best_eval.get("best_route"), f"En kârlı rota seçilemedi: {best_eval}"
            assert best_eval.get("expected_kinah_per_hour", 0) > 0, "Saatlik Kinah projeksiyonu 0'dan büyük olmalı"

            morph_q = AutoCraftingMorphQueue(default_threshold=20, price_scanner=scanner)
            morph_q.record_gathered_material("Orichalcum Ore", count=25)
            craft_jobs = morph_q.process_crafting_queue(test_mode=True)
            assert len(craft_jobs) >= 1 and craft_jobs[0].get("produced_quantity", 0) > 0, (
                f"AutoCraftingMorphQueue 25 cevherde tetiklenmedi: {craft_jobs}"
            )

            profit_sim = SessionProfitSimulator(price_scanner=scanner, crafting_queue=morph_q)
            profit_sim.record_gather("Pure Aether", count=12)
            profit_sim.record_gather("Aria", count=18)
            summary = profit_sim.get_session_economy_summary(elapsed_hours=0.5)
            assert summary.get("net_kinah_earned", 0) > 0, f"Net kâr pozitif olmalı: {summary}"
            formatted = format_kinah_tr(1420000)
            assert "M" in formatted or "1" in formatted, f"Kinah formatı hatalı: {formatted}"

            return (
                f"En Kârlı Rota='{best_eval['best_route']}' ({best_eval['formatted_kinah_per_hour']}/saat), "
                f"Morph={craft_jobs[0]['produced_quantity']}x {craft_jobs[0]['output_item']}, "
                f"Saatlik Net={summary['formatted_kinah_per_hour']}/saat"
            )

    _run_case(
        "ENT-2",
        "Otonom Ekonomi & Pazar (Bölüm 2)",
        "core/economy_broker.py (TradeBrokerPriceScanner, AutoCraftingMorphQueue, SessionProfitSimulator)",
        _test_economy_broker_and_profit_simulator,
    )

    def _test_neural_yolo_3d_and_audio_fft() -> str:
        from core.neural_audio_perception import AudioLoopbackFFTAnalyzer, ONNXYolo3DDetector

        detector = ONNXYolo3DDetector(conf_threshold=0.35, iou_threshold=0.45)
        # ONNX'suz fallback ile geçen 3B tespiti sayılmaz: repo'daki
        # models/yolov8n_gatherion.onnx ile gerçek ONNX/DNN backend kanıtlanmalı.
        # v19-smoke: yükleme LAZY'dir (kurucu model yüklemez; ilk
        # detect_3d_objects check_for_model_update ile yükler) — backend
        # iddiası ilk tespitten SONRA okunur, yoksa kurucudaki
        # "fast_3d_mesh_tensor_fallback" varsayılanı görülür.
        detections = detector.detect_3d_objects(ref_frame)
        backend_name = str(getattr(detector, "backend_name", ""))
        assert "fallback" not in backend_name.lower(), (
            f"3B dedektör fallback backend kullanıyor ({backend_name}) — ONNX modeli yüklenmedi"
        )
        assert backend_name.startswith(("onnxruntime", "opencv_dnn")), (
            f"Beklenen ONNX/DNN backend, gelen: {backend_name}"
        )
        assert isinstance(detections, list) and len(detections) >= 1, "ONNXYolo3DDetector hiçbir 3B nesne tespit etmedi"
        first_det = detections[0]
        assert "class_name" in first_det and "confidence" in first_det and "bbox" in first_det, (
            f"Eksik 3B tespit alanı: {first_det}"
        )
        assert 0.0 < float(first_det["confidence"]) <= 1.0, f"Geçersiz güven: {first_det['confidence']}"
        assert len(first_det["bbox"]) == 4, f"bbox 4 eleman olmalı: {first_det['bbox']}"

        analyzer = AudioLoopbackFFTAnalyzer(sample_rate=44100)
        chime_pcm = analyzer.synthesize_crystal_chime(duration_sec=0.25)
        res_chime = analyzer.analyze_pcm_buffer(chime_pcm)
        assert res_chime.get("gather_success_chime") is True, f"Kristal çınlaması (3200Hz) algılanmadı: {res_chime}"

        fail_pcm = analyzer.synthesize_failure_shatter(duration_sec=0.25)
        res_fail = analyzer.analyze_pcm_buffer(fail_pcm)
        assert res_fail.get("gather_fail_sound") is True, f"Toplama kırılma sesi (290Hz) algılanmadı: {res_fail}"

        step_pcm = analyzer.synthesize_monster_footstep(duration_sec=0.25)
        res_step = analyzer.analyze_pcm_buffer(step_pcm)
        assert res_step.get("danger_audio_alert") is True, f"Canavar ayak sesi (95Hz) algılanmadı: {res_step}"

        # Negatif kontroller: kendi ürettiği sesi kendi analiz eden test,
        # sessizliği ve ilgisiz tonu da reddetmeli — yoksa her girdiye
        # "olumlu" diyen bir analizör de geçer.
        silence = np.zeros(int(0.25 * 44100), dtype=np.float32)
        res_silence = analyzer.analyze_pcm_buffer(silence)
        assert res_silence.get("gather_success_chime") is False, "Sessizlik çınlama sayılmamalı"
        assert res_silence.get("gather_fail_sound") is False, "Sessizlik kırılma sayılmamalı"
        assert res_silence.get("danger_audio_alert") is False, "Sessizlik tehlike sayılmamalı"
        t_axis = np.linspace(0.0, 0.25, int(0.25 * 44100), endpoint=False, dtype=np.float32)
        off_tone = (0.5 * np.sin(2.0 * np.pi * 1000.0 * t_axis)).astype(np.float32)
        res_off = analyzer.analyze_pcm_buffer(off_tone)
        assert res_off.get("gather_success_chime") is False, "1000Hz ton çınlama sayılmamalı"
        assert res_off.get("danger_audio_alert") is False, "1000Hz ton tehlike sayılmamalı"

        return (
            f"3B YOLO Tespit={len(detections)} nesne ({first_det['class_name']} @{first_det['confidence']:.2f}, "
            f"backend={backend_name}), "
            f"FFT: Chime=True, Fail=True, DangerFootstep=True, NegatifKontroller(Sessizlik/1000Hz)=Reddedildi"
        )

    _run_case(
        "ENT-3",
        "3B Derin Öğrenme & Ses FFT (Bölüm 3)",
        "core/neural_audio_perception.py (ONNXYolo3DDetector & AudioLoopbackFFTAnalyzer 3 Frekans Bandı)",
        _test_neural_yolo_3d_and_audio_fft,
    )

    def _test_multi_char_scheduler_and_pet_buffs() -> str:
        from core.multi_char_scheduler import (
            AltCharacterManager,
            AutomationTaskScheduler,
            PetAndBuffManager,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            roster_path = os.path.join(tmp_dir, "char_roster.json")
            alt_mgr = AltCharacterManager(config_path=roster_path)
            mock_rm = _SmokeMockRouteManager()
            rot_res = alt_mgr.check_and_rotate_character(
                current_gathered=400,
                route_mgr=mock_rm,
                test_mode=True,
            )
            assert rot_res.get("rotated") is True, f"Kota (400) dolduğunda yan karaktere geçilmedi: {rot_res}"

            sched_path = os.path.join(tmp_dir, "schedule.json")
            scheduler = AutomationTaskScheduler(config_path=sched_path)
            ev_1930 = scheduler.evaluate_current_schedule("19:30")
            ev_2145 = scheduler.evaluate_current_schedule("21:45")
            ev_2230 = scheduler.evaluate_current_schedule("22:30")

            assert ev_1930.get("action") == "run_route" and ev_1930.get("recommended_state") == "WALKING", (
                f"19:30 Akşam Rotası planı hatalı: {ev_1930}"
            )
            assert ev_2145.get("town_wait_active") is True and ev_2145.get("recommended_state") == "WAITING", (
                f"21:45 Şehirde Bekleme planı hatalı: {ev_2145}"
            )
            assert ev_2230.get("action") == "run_route" and ev_2230.get("recommended_state") == "WALKING", (
                f"22:30 Gece Toplama planı hatalı: {ev_2230}"
            )

            pet_mgr = PetAndBuffManager(pet_feed_interval_sec=1800.0)
            for slot in pet_mgr.buff_slots:
                slot.last_used_ts = 0.0
            # test_mode=True donanıma dokunmaz ama sahte girdiye tuşları
            # gerçekten göndermeli: dönen pressed_keys listesi, mock'un
            # kaydettiği tuşlarla birebir eşleşmeli (yoksa test sadece
            # sözlük döndürmüş olur).
            mock_ic = _SmokeMockInputCtrl()
            buff_res = pet_mgr.check_and_refresh_buffs(frame=ref_frame, input_ctrl=mock_ic, test_mode=True)
            assert buff_res.get("pet_fed") is True and len(buff_res.get("pressed_keys", [])) >= 1, (
                f"Pet/Parşömen yenilemesi tetiklenmedi: {buff_res}"
            )
            assert mock_ic.pressed_keys == list(buff_res.get("pressed_keys", [])), (
                f"Mock girdiye giden tuşlar dönen listeyle eşleşmiyor: {mock_ic.pressed_keys} != {buff_res.get('pressed_keys')}"
            )

            prev_name = rot_res.get("previous_character", {}).get("name", "Slot1")
            next_name = rot_res.get("active_character", {}).get("name", "Slot2")
            return (
                f"AltRotasyon={prev_name}->{next_name}, "
                f"Takvim(19:30=Rota, 21:45=ŞehirBekleme, 22:30=Morheim), "
                f"Pet+Buff Tuşları={buff_res['pressed_keys']} (mock girdide doğrulandı, test_mode=True — donanım yok)"
            )

    _run_case(
        "ENT-4",
        "Çoklu Karakter & Zamanlayıcı (Bölüm 4)",
        "core/multi_char_scheduler.py (AltCharacterManager, AutomationTaskScheduler, PetAndBuffManager)",
        _test_multi_char_scheduler_and_pet_buffs,
    )

    def _test_depth_obstacle_avoidance_and_glide_controller() -> str:
        from core.depth_glide_navigator import (
            GlideWaypointController,
            MonocularDepthObstacleAvoidance,
        )

        depth_nav = MonocularDepthObstacleAvoidance(obstacle_clearance_threshold=0.35)
        depth_map = depth_nav.estimate_depth_map(ref_frame)
        assert isinstance(depth_map, np.ndarray) and depth_map.ndim == 2 and depth_map.size > 0, (
            f"Derinlik haritası boyutu hatalı: {depth_map.shape}"
        )

        # Merkez koridora yapay yakın bir kaya/duvar engeli yerleştirip kaçınma manevrasını doğrula
        blocked_frame = ref_frame.copy()
        fh, fw = blocked_frame.shape[:2]
        cx1, cy1 = int(fw * 0.42), int(fh * 0.36)
        cx2, cy2 = int(fw * 0.58), int(fh * 0.72)
        cv2.rectangle(blocked_frame, (cx1, cy1), (cx2, cy2), (25, 25, 25), -1)
        for gx in range(cx1 + 8, cx2 - 8, 12):
            cv2.line(blocked_frame, (gx, cy1), (gx, cy2), (220, 220, 220), 2)

        corridor = depth_nav.analyze_forward_corridor(blocked_frame)
        assert "recommended_steer_deg" in corridor and "obstacle_detected" in corridor, (
            f"Koridor analizi eksik: {corridor}"
        )
        mock_ic = _SmokeMockInputCtrl()
        steer_res = depth_nav.apply_avoidance_steering(mock_ic, corridor, test_mode=True)
        assert "steer_deg" in steer_res, f"Direksiyon kaçınma sonucu hatalı: {steer_res}"
        # test_mode=True donanıma dokunmamalı ama direksiyon hesabı
        # tutarlı olmalı: mouse_dx = steer_deg x 4.2 px/deg.
        assert steer_res.get("test_mode") is True, "Kaçınma test_mode kaydı eksik"
        if corridor.get("obstacle_detected") and abs(float(corridor.get("recommended_steer_deg", 0.0))) >= 1.5:
            assert steer_res.get("applied") is True, f"Engel varken kaçınma uygulanmadı: {steer_res}"
            assert steer_res.get("mouse_dx") == int(round(float(steer_res["steer_deg"]) * 4.2)), (
                f"mouse_dx hesabı tutarsız: {steer_res}"
            )
        assert mock_ic.moved_points == [], "test_mode=True iken fare hareket etmemeli"

        glide_ctrl = GlideWaypointController(min_elevation_drop=10.0, min_horizontal_dist=20.0)
        current_wp = {"id": 10, "x": 500, "y": 350, "z": 145.0, "is_glide_waypoint": True}
        target_wp = {"id": 11, "x": 620, "y": 460, "z": 112.0}

        can_glide = glide_ctrl.should_use_glide(current_wp, target_wp)
        assert can_glide is True, "33m irtifa düşüşü olan hedef için should_use_glide True olmalı"

        params = glide_ctrl.estimate_glide_parameters(current_wp, target_wp)
        assert params.get("elevation_drop", 0) >= 30.0 and params.get("glide_duration_sec", 0) > 1.0, (
            f"Süzülme parametreleri hatalı: {params}"
        )

        glide_exec = glide_ctrl.execute_glide_maneuver(
            mock_ic,
            target_heading_deg=params["target_heading_deg"],
            glide_duration_sec=params["glide_duration_sec"],
            test_mode=True,
        )
        assert glide_exec.get("success") is True and glide_exec.get("wings_deployed") is True, (
            f"Kanat süzülme manevrası başarısız: {glide_exec}"
        )
        assert glide_exec.get("test_mode") is True, "Glide test_mode kaydı eksik"

        return (
            f"Derinlik={depth_map.shape}, Steer={corridor['recommended_steer_deg']} deg (Engel={corridor['obstacle_type']}), "
            f"Glide(FarkZ=-{params['elevation_drop']}m, Sure={params['glide_duration_sec']}sn, test_mode=True — donanım yok)"
        )

    _run_case(
        "ENT-5",
        "Derinlik Engel Kaçınma & Glide (Bölüm 5)",
        "core/depth_glide_navigator.py (MonocularDepthObstacleAvoidance & GlideWaypointController)",
        _test_depth_obstacle_avoidance_and_glide_controller,
    )

    def _test_report_table_bounds_and_failure_format() -> str:
        # Rapor tablosu kilit testi: (1) hiçbir satır 118 sütunu aşmamalı
        # (uzun ENT-3 detayı eskiden ~1100 karakterlik tek satırdı),
        # (2) alt bilgi toplam süre + en yavaş vakayı vermeli,
        # (3) gerçek kategoriler tabloda kesilmeden görünmeli,
        # (4) kalan vakaların hata detayı mesajı iki kez yazmamalı.
        fake = [
            {"id": "X1", "category": "Kısa Kategori", "name": "sahte", "passed": True,
             "status": "GEÇTİ", "elapsed_ms": 12.3, "detail": "kısa"},
            {"id": "X2", "category": "K" * 60, "name": "sahte", "passed": False,
             "status": "KALDI", "elapsed_ms": 9876.5, "detail": "D" * 500},
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_turkish_report_table(1, 1, fake)
        fake_lines = buf.getvalue().splitlines()
        assert fake_lines, "rapor tablosu boş çıktı"
        assert all(len(ln) <= 118 for ln in fake_lines), (
            f"118 sütunu aşan satır var (en uzun {max(len(ln) for ln in fake_lines)})"
        )
        assert any("TOPLAM SÜRE" in ln for ln in fake_lines), "TOPLAM SÜRE satırı eksik"
        assert any("EN YAVAŞ" in ln and "X2" in ln for ln in fake_lines), "EN YAVAŞ satırı eksik/yanlış"

        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            _print_turkish_report_table(0, 0, list(results))
        real_text = buf2.getvalue()
        assert all(len(ln) <= 118 for ln in real_text.splitlines()), "gerçek sonuç tablosu 118 sütunu aşıyor"
        for _row in results:
            assert str(_row.get("category", "")) in real_text, (
                f"Kategori tabloda kesilmiş: {_row.get('category', '')!r}"
            )
        for _row in results:
            if not _row.get("passed"):
                _det = str(_row.get("detail", ""))
                assert " (AssertionError:" not in _det, f"Hata detayı mesajı iki kez yazılmış: {_det[:120]}"
                assert _det.count("AssertionError") <= 1, f"Hata detayı tekrarlı: {_det[:120]}"
        return "Rapor tablosu ≤118 sütun, TOPLAM SÜRE + EN YAVAŞ var, kategoriler tam, hata detayı tekrarsız"

    _run_case(
        "RPT",
        "Raporlama",
        "Rapor tablosu 118 sütun sınırı + toplam süre/en yavaş + tekrarsız hata detayı",
        _test_report_table_bounds_and_failure_format,
    )

    passed_count = sum(1 for r in results if r["passed"])
    failed_count = len(results) - passed_count

    if verbose:
        _print_turkish_report_table(passed_count, failed_count, results)

    return passed_count, failed_count, results


def _print_turkish_report_table(passed_count: int, failed_count: int, results: List[dict]) -> None:
    """Türkçe formatlı detaylı duman testi sonuç tablosunu konsola yazdırır.

    Tablo 118 sütuna sığar: kategori sütunu en uzun gerçek kategoriye
    (40 karakter) göre ayarlıdır, detay sütunu sığmayanı "…" ile kısaltır
    (tam detay `results` sözlüğünde kalır). Alt bilgi toplam süreyi ve
    en yavaş vakayı verir — en yavaş vaka, koşum süresi optimizasyonunda
    ilk bakılacak yerdir.
    """
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception as exc:
            _ = exc
    total = len(results)
    width = 118
    cat_w = 40
    # Satır: " {KOD:<8} | {DURUM:<6} | {SÜRE:>8} | {KATEGORİ:<40} | {DETAY}"
    detail_w = width - (1 + 8 + 3 + 6 + 3 + 8 + 3 + cat_w + 3)
    sep = "=" * width
    thin = "-" * width

    def _trunc(text: str, limit: int) -> str:
        text = str(text)
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)] + "…"

    def _out(line: str) -> None:
        print(_trunc(line, width))

    _out("")
    _out(sep)
    _out(f" GATHERION v{__version__} — MOBİL PWA/SSE/PiP, OPSEC/PERFORMANS (31-50) & KURUMSAL MODÜLLER DUMAN TESTİ RAPORU")
    _out(sep)
    _out(f" {'KOD':<8} | {'DURUM':<6} | {'SÜRE':>8} | {'KATEGORİ':<{cat_w}} | {'DETAY / DOĞRULAMA SONUCU'}")
    _out(thin)
    for item in results:
        code = _trunc(str(item.get("id", "")), 8)
        status = _trunc(str(item.get("status", "")), 6)
        ms_str = f"{item.get('elapsed_ms', 0.0):.1f}ms"
        cat = _trunc(str(item.get("category", "")), cat_w)
        det = _trunc(str(item.get("detail", "")), detail_w)
        _out(f" {code:<8} | {status:<6} | {ms_str:>8} | {cat:<{cat_w}} | {det}")
    _out(sep)
    rate = (passed_count / total * 100.0) if total > 0 else 0.0
    _out(
        f" TOPLAM TEST: {total}  |  GEÇEN: {passed_count}  |  KALAN: {failed_count}  |  BAŞARI ORANI: %{rate:.1f}"
    )
    total_ms = sum(float(r.get("elapsed_ms", 0.0) or 0.0) for r in results)
    slowest = max(results, key=lambda r: float(r.get("elapsed_ms", 0.0) or 0.0), default=None)
    if slowest is not None:
        _out(
            f" TOPLAM SÜRE: {total_ms:.1f}ms  |  EN YAVAŞ: {slowest.get('id', '')} "
            f"({float(slowest.get('elapsed_ms', 0.0) or 0.0):.1f}ms)"
        )
    _out(sep)
    print()


# S-4701 / karar 3: hazır/audit bayrakları hiçbir test çalıştırmadan
# "ready: True" dönemez. Tek kaynak, mobil+kurumsal duman testlerinin
# gerçek koşusudur (önbellekli — aynı süreçte bir kez koşulur, sonrası
# aynı sonucu verir; desen: tools/smoke_test_vision_and_3x.py F-1129/F-1241).
_MOBILE_ENTERPRISE_SMOKE_CACHE: Optional[Dict[str, Any]] = None


def _get_mobile_enterprise_smoke_ready_result() -> Dict[str, Any]:
    """Mobil+kurumsal duman testlerini gerçekten koşup hazır sonucunu döndürür."""
    global _MOBILE_ENTERPRISE_SMOKE_CACHE
    if _MOBILE_ENTERPRISE_SMOKE_CACHE is None:
        try:
            passed, failed, results = run_mobile_and_enterprise_smoke_tests(verbose=False)
            failed_ids = [str(r["id"]) for r in results if not r.get("passed")]
            total = int(passed) + int(failed)
            _MOBILE_ENTERPRISE_SMOKE_CACHE = {
                "module": "smoke_test_mobile_and_enterprise",
                "ready": bool(passed == total and total > 0),
                "passed": int(passed),
                "failed": int(failed),
                "total": total,
                "failed_checks": failed_ids,
            }
        except Exception as exc:  # noqa: BLE001 — koşu çökerse hazır değildir
            _MOBILE_ENTERPRISE_SMOKE_CACHE = {
                "module": "smoke_test_mobile_and_enterprise",
                "ready": False,
                "passed": 0,
                "failed": 0,
                "total": 0,
                "failed_checks": [],
                "error": str(exc),
            }
    return dict(_MOBILE_ENTERPRISE_SMOKE_CACHE)


def get_mobile_enterprise_smoke_summary() -> dict:
    """Mobil+kurumsal duman testlerini gerçekten çalıştırır ve gerçek özeti döndürür."""
    res = _get_mobile_enterprise_smoke_ready_result()
    return {
        "mobile_enterprise_smoke_ready": bool(res["ready"]),
        "passed_checks": res["passed"],
        "failed_checks": res["failed_checks"],
        "total_checks": res["total"],
    }


def get_mobile_enterprise_smoke_ready_flag() -> dict:
    res = _get_mobile_enterprise_smoke_ready_result()
    return {
        "module": "smoke_test_mobile_and_enterprise",
        "ready": bool(res["ready"]),
        "passed": res["passed"],
        "total": res["total"],
        "failed_checks": res["failed_checks"],
    }


if __name__ == "__main__":
    passed, failed, _ = run_mobile_and_enterprise_smoke_tests(verbose=True)
    sys.exit(0 if failed == 0 else 1)


def get_smoke_test_mobile_and_enterprise_version_info() -> dict:
    res = _get_mobile_enterprise_smoke_ready_result()
    return {"module": "smoke_test_mobile_and_enterprise", "version": __version__, "ready": bool(res["ready"])}


def get_smoke_test_mobile_and_enterprise_diagnostic_snapshot() -> dict:
    res = _get_mobile_enterprise_smoke_ready_result()
    ready = bool(res["ready"])
    return {
        "module": "smoke_test_mobile_and_enterprise",
        "status": "ok" if ready else "failing",
        "ready": ready,
        "passed": res["passed"],
        "total": res["total"],
        "failed_checks": res["failed_checks"],
    }


def get_smoke_test_mobile_and_enterprise_capabilities() -> list:
    return ["smoke_test_mobile_and_enterprise", "diagnostic_snapshot", "version_info"]


def validate_smoke_test_mobile_and_enterprise_contract() -> bool:
    # S-4701: sözleşme, hardcoded capability listesinin bool'u değil,
    # duman testlerinin gerçek koşu sonucudur (önbellekli prob).
    res = _get_mobile_enterprise_smoke_ready_result()
    return bool(get_smoke_test_mobile_and_enterprise_capabilities()) and bool(res["ready"])


def get_smoke_test_mobile_and_enterprise_baseline_metrics() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "contract_valid": validate_smoke_test_mobile_and_enterprise_contract(), "capability_count": len(get_smoke_test_mobile_and_enterprise_capabilities())}


def get_smoke_test_mobile_readiness_status() -> dict:
    # S-4701 zinciri: "ready" hardcoded True değil, gerçek koşu sonucudur.
    res = _get_mobile_enterprise_smoke_ready_result()
    return {"module": "smoke_test_mobile_and_enterprise", "ready": bool(res["ready"]), "metrics": get_smoke_test_mobile_baseline_metrics() if 'get_smoke_test_mobile_baseline_metrics' in globals() else {}}


def get_smoke_test_mobile_audit_record() -> dict:
    res = _get_mobile_enterprise_smoke_ready_result()
    return {"module": "smoke_test_mobile_and_enterprise", "audit_passed": bool(res["ready"]), "readiness": get_smoke_test_mobile_readiness_status()}


def get_smoke_test_mobile_and_enterprise_audit_record() -> dict:
    return get_smoke_test_mobile_audit_record()


def get_smoke_test_mobile_and_enterprise_operational_digest() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "operational": validate_smoke_test_mobile_and_enterprise_contract(), "audit": get_smoke_test_mobile_and_enterprise_audit_record()}


def get_smoke_test_mobile_and_enterprise_telemetry_payload() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "telemetry_ready": validate_smoke_test_mobile_and_enterprise_contract(), "digest": get_smoke_test_mobile_and_enterprise_operational_digest()}


def get_smoke_test_mobile_and_enterprise_integrity_seal() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "sealed": validate_smoke_test_mobile_and_enterprise_contract(), "telemetry": get_smoke_test_mobile_and_enterprise_telemetry_payload()}


def get_smoke_test_mobile_and_enterprise_compliance_certificate() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "certified": validate_smoke_test_mobile_and_enterprise_contract(), "seal": get_smoke_test_mobile_and_enterprise_integrity_seal()}


def get_smoke_test_mobile_and_enterprise_readiness_probe() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "ready": validate_smoke_test_mobile_and_enterprise_contract(), "certificate": get_smoke_test_mobile_and_enterprise_compliance_certificate()}


def get_smoke_test_mobile_and_enterprise_audit_summary() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "audit_passed": validate_smoke_test_mobile_and_enterprise_contract(), "probe": get_smoke_test_mobile_and_enterprise_readiness_probe()}


def get_smoke_test_mobile_and_enterprise_heartbeat_snapshot() -> dict:
    return {"module": "smoke_test_mobile_and_enterprise", "alive": validate_smoke_test_mobile_and_enterprise_contract(), "audit": get_smoke_test_mobile_and_enterprise_audit_summary()}

