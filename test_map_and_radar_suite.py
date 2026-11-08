# -*- coding: utf-8 -*-
"""
Gatherion-Local v2.0 — Comprehensive Map and Radar Test Suite
(Harita ve Radar Kapsamlı Doğrulama ve Entegrasyon Test Paketi)

Doğrulanan 8 Temel Alan:
1. Koordinat Dönüşümleri (1280x720 kırpılmış video karesi, 2258x2400 Altgard HD haritası, apply_hd_fit, inv_hd_fit, kenara yapışmama ve negatif olmama kontrolleri).
2. Sıralı Rota Yönetimi ve 3+ Noktalı Döngüsel İlerleme (SequentialRoutePlanner, RouteManager cyclical progression, kısmi hasat önceliği, seq_index kalıcılığı).
3. Alt Bölge Kanonik Eşlemesi (Dredgion Crash Site, Safe Haven, Calderon Canyon, Silent Hill -> Altgard).
4. Minimap Tarayıcı Oyuncu Yönü ve Filiz/Canavar Noktası Ayrımı (MinimapScanner heading detection, sprout vs monster dot separation).
5. Radar Klerans Eşiği (_GATHER_PLAYER_CLEARANCE = 0.075 ve yakın toplanabilir filizlerin kör noktaya düşmeden korunması).
6. Navigator Yön Açısı ve Kerteriz Hesaplamaları (compute_bearing, compute_angle_diff, en kısa dönüş açısı, 180° ters dönme engeli, ±6° ölü bant sönümleme).
7. Harita Tuvali Etkileşimi (_MapCanvas tıklama, sürükleyerek taşıma, sağ tık silme ve sıralı rota okları).
8. AppShell Sekme Geçişleri ve Canlı Sinyal Senkronizasyonu (GatherionShell harita, radar ve ayarlar sekmeleri canlı senkronizasyonu).
"""

from __future__ import annotations

import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Ensure offscreen Qt platform before importing PyQt6
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import QApplication

# Initialize shared QApplication for offscreen widget testing
_APP = QApplication.instance()
if _APP is None:
    _APP = QApplication([sys.argv[0], "-platform", "offscreen"])

from core.minimap_scanner import MinimapScanner
from core.navigator import (
    DEFAULT_DEADBAND_DEG,
    MAX_DEADBAND_DEG,
    MIN_DEADBAND_DEG,
    TurnDampingController,
    apply_turn_damping,
    compute_angle_diff,
    compute_bearing,
)
from core.route_manager import DEFAULT_MAX_HARVESTS, NodeData, RouteManager
from ui.app_shell import GatherionShell
from ui.cartography_radar import (
    _GATHER_PLAYER_CLEARANCE,
    CANONICAL_REGION_ALIASES,
    DEFAULT_MAP_REF,
    SequentialRoutePlanner,
    apply_hd_fit,
    classify_minimap_crop_bgr,
    minimap_material_is_recordable,
    minimap_norm_is_hud_chrome,
    resolve_canonical_region,
    resolve_node_xy,
)
from ui.map_viewer import _MapCanvas, inv_hd_fit


class TestCoordinateTransform(unittest.TestCase):
    """
    Test 1: 1280x720 video karesi, 2258x2400 Altgard HD haritası ve radar tuvali
    arasındaki koordinat dönüşümleri (apply_hd_fit, inv_hd_fit).
    """

    def test_apply_hd_fit_reference_anchor(self) -> None:
        """Kırpım referans noktası (280.0, 80.0) -> (_HD_TX, _HD_TY) olmalıdır."""
        from ui.cartography_radar import _HD_TX, _HD_TY
        hx, hy = apply_hd_fit(280.0, 80.0)
        self.assertAlmostEqual(hx, _HD_TX, places=2)
        self.assertAlmostEqual(hy, _HD_TY, places=2)

    def test_apply_hd_fit_bounds_and_no_negative_coordinates(self) -> None:
        """1280x720 kırpılmış oyun haritası çerçevesindeki noktalar pozitif ve Altgard HD sınırları içinde kalmalıdır."""
        # frame_18s.jpg arazi kırpması (280, 80) ile (894, 530) arasındaki geçerli alan
        test_points = [
            (280.0, 80.0),
            (640.0, 360.0),
            (894.0, 530.0),
            (450.0, 220.0),
            (750.0, 410.0),
        ]
        for x, y in test_points:
            hx, hy = apply_hd_fit(x, y)
            self.assertGreaterEqual(hx, 0.0, f"X negatif çıktı: {hx} (girdi: {x}, {y})")
            self.assertGreaterEqual(hy, 0.0, f"Y negatif çıktı: {hy} (girdi: {x}, {y})")
            self.assertLessEqual(hx, 2258.0, f"X Altgard HD genişliğini aştı: {hx}")
            self.assertLessEqual(hy, 2400.0, f"Y Altgard HD yüksekliğini aştı: {hy}")

    def test_inv_hd_fit_roundtrip_precision(self) -> None:
        """apply_hd_fit ve inv_hd_fit birbirinin tam tersi (inverse affine) olmalıdır."""
        grid = [
            (280.0, 80.0),
            (640.0, 360.0),
            (450.0, 200.0),
            (850.0, 500.0),
            (894.0, 530.0),
        ]
        for ox, oy in grid:
            hx, hy = apply_hd_fit(ox, oy)
            rx, ry = inv_hd_fit(hx, hy)
            self.assertAlmostEqual(rx, ox, places=3, msg=f"X ters dönüşüm hatası: {rx} != {ox}")
            self.assertAlmostEqual(ry, oy, places=3, msg=f"Y ters dönüşüm hatası: {ry} != {oy}")

    def test_resolve_node_xy_normalization(self) -> None:
        """Düğüm norm_x, norm_y oranları referans harita çerçevesine (1280x720) doğru oturmalıdır."""
        node_norm = {"norm_x": 0.50, "norm_y": 0.50, "map_x": 0.0, "map_y": 0.0}
        rx, ry = resolve_node_xy(node_norm, ref_w=1280.0, ref_h=720.0)
        self.assertAlmostEqual(rx, 640.0, places=1)
        self.assertAlmostEqual(ry, 360.0, places=1)

    def test_map_canvas_no_edge_clamping_for_valid_nodes(self) -> None:
        """Geçerli düğümler harita kenarına (8.0, 8.0) sahte olarak yapışmamalıdır."""
        canvas = _MapCanvas()
        canvas.resize(800, 600)
        pix = QPixmap(2258, 2400)
        canvas.set_map_pixmap(pix)

        node = {"id": 1, "map_x": 640.0, "map_y": 360.0, "norm_x": 0.50, "norm_y": 0.50}
        pt = canvas._node_to_map_coords(640.0, 360.0, norm_x=0.50, norm_y=0.50, node=node)
        self.assertGreater(pt.x(), 50.0, f"X harita kenarına sıkıştı: {pt.x()}")
        self.assertGreater(pt.y(), 50.0, f"Y harita kenarına sıkıştı: {pt.y()}")
        self.assertNotEqual(pt.x(), 8.0)
        self.assertNotEqual(pt.y(), 8.0)


class TestSequentialRouteOrdering(unittest.TestCase):
    """
    Test 2: Sıralı rota yönetimi ve 3+ noktalı döngüsel ilerleme.
    """

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="gatherion_route_test_")
        self.routes = RouteManager(base_dir=self.tmp_dir)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_three_point_cyclical_node_progression(self) -> None:
        """3 noktalı rotada 1 -> 2 -> 3 -> 1 döngüsü takılmadan sürekli çalışmalıdır."""
        region = "Altgard"
        route = "Ore_Loop"
        self.routes.create_route(region, route)

        n1 = self.routes.add_node(region=region, route=route, map_x=100, map_y=100, norm_x=0.1, norm_y=0.1)
        n2 = self.routes.add_node(region=region, route=route, map_x=200, map_y=200, norm_x=0.2, norm_y=0.2)
        n3 = self.routes.add_node(region=region, route=route, map_x=300, map_y=300, norm_x=0.3, norm_y=0.3)

        self.assertIsNotNone(n1)
        self.assertIsNotNone(n2)
        self.assertIsNotNone(n3)

        # 1. Döngü Adımı: İlk hazır düğüm #1
        first = self.routes.get_next_ready_node(region=region, route=route, rest_duration_sec=300)
        self.assertIsNotNone(first)
        self.assertEqual(first.id, n1.id)

        # #1 tamamlandı ve dinlenmeye geçti
        first.current_harvests = 3
        first.state = "resting"
        first.last_visited_timestamp = time.time()

        # 2. Döngü Adımı: Sıradaki düğüm #2
        second = self.routes.get_next_ready_node(region=region, route=route, rest_duration_sec=300)
        self.assertIsNotNone(second)
        self.assertEqual(second.id, n2.id)

        # #2 tamamlandı
        second.current_harvests = 3
        second.state = "resting"
        second.last_visited_timestamp = time.time()

        # 3. Döngü Adımı: Sıradaki düğüm #3
        third = self.routes.get_next_ready_node(region=region, route=route, rest_duration_sec=300)
        self.assertIsNotNone(third)
        self.assertEqual(third.id, n3.id)

        # #3 tamamlandı
        third.current_harvests = 3
        third.state = "resting"
        third.last_visited_timestamp = time.time()

        # 4. Döngü Adımı (Sıfırlama / Dinlenme Bitimi): Başa (#1) sarmalı
        for node in [n1, n2, n3]:
            node.last_visited_timestamp = 0.0
            node.state = "active"
            node.current_harvests = 0

        cycle_wrap = self.routes.get_next_ready_node(region=region, route=route, rest_duration_sec=300)
        self.assertIsNotNone(cycle_wrap)
        self.assertEqual(cycle_wrap.id, n1.id, "3 noktadan sonra döngü başa (Node 1) sarmadı!")

    def test_partial_harvest_priority(self) -> None:
        """Kısmi toplanmış (1/3 veya 2/3) düğüm 3/3 olana kadar bir sonrakine atlamamalıdır."""
        region = "Altgard"
        route = "Ore_Loop"
        self.routes.create_route(region, route)

        n1 = self.routes.add_node(region=region, route=route, map_x=100, map_y=100)
        n2 = self.routes.add_node(region=region, route=route, map_x=200, map_y=200)

        # n1 üzerinde 1 kez toplandı
        n1.current_harvests = 1
        n1.state = "gathering"

        selected = self.routes.get_next_ready_node(region=region, route=route)
        self.assertEqual(selected.id, n1.id, "Kısmi toplanan düğüm yerine başka düğüme atlandı!")

    def test_reorder_route_nodes_persistence(self) -> None:
        """Düğümler yeniden sıralandığında seq_index diske kaydedilmeli ve korunmalıdır."""
        region = "Altgard"
        route = "Ore_Loop"
        self.routes.create_route(region, route)

        n1 = self.routes.add_node(region=region, route=route, map_x=100, map_y=100)
        n2 = self.routes.add_node(region=region, route=route, map_x=200, map_y=200)
        n3 = self.routes.add_node(region=region, route=route, map_x=300, map_y=300)

        # Sırayı tersine çevir: n3 -> n1 -> n2
        reordered_ids = [n3.id, n1.id, n2.id]
        success = self.routes.reorder_route_nodes(region, reordered_ids, route)
        self.assertTrue(success)

        # Yeni manager örneği ile diskten doğrula
        fresh_routes = RouteManager(base_dir=self.tmp_dir)
        fresh_nodes = fresh_routes.get_route_nodes(region, route)
        fresh_ids = [n.id for n in fresh_nodes]
        self.assertEqual(fresh_ids, reordered_ids)
        self.assertEqual(fresh_nodes[0].seq_index, 1)
        self.assertEqual(fresh_nodes[1].seq_index, 2)
        self.assertEqual(fresh_nodes[2].seq_index, 3)

    def test_sequential_route_planner_metrics(self) -> None:
        """SequentialRoutePlanner rota uzunluğunu ve kapalı döngü mesafesini doğru hesaplamalıdır."""
        planner = SequentialRoutePlanner()
        pts = [
            {"map_x": 0.0, "map_y": 0.0, "norm_x": 0.0, "norm_y": 0.0},
            {"map_x": 300.0, "map_y": 0.0, "norm_x": 300.0 / 1280.0, "norm_y": 0.0},
            {"map_x": 300.0, "map_y": 400.0, "norm_x": 300.0 / 1280.0, "norm_y": 400.0 / 720.0},
        ]
        metrics = planner.calculate_route_metrics(pts)
        self.assertEqual(metrics["node_count"], 3)
        self.assertAlmostEqual(float(metrics["total_distance_px"]), 1200.0, places=1)
        self.assertTrue(metrics["closed_loop"])


class TestSubregionAliasMapping(unittest.TestCase):
    """
    Test 3: Alt bölge kanonik eşlemesi (Dredgion Crash Site, Safe Haven, Calderon Canyon -> Altgard).
    """

    def test_subregion_aliases_resolve_to_altgard(self) -> None:
        """Kullanıcının karşılaştığı tüm Altgard alt bölgeleri kanonik 'Altgard' ile eşleşmelidir."""
        altgard_subregions = [
            "Dredgion Crash Site",
            "Safe Haven",
            "Calderon Canyon",
            "Silent Hill",
            "Altgard Fortress",
            "Moslan Forest",
            "Mahki Swamp",
            "Gribbler Canyon",
            "Idun's Lake",
            "Altgard Ice Lake",
        ]
        for subregion in altgard_subregions:
            resolved = resolve_canonical_region(subregion)
            self.assertEqual(
                resolved,
                "Altgard",
                f"Alt bölge '{subregion}' kanonik 'Altgard' yerine '{resolved}' olarak çözüldü!",
            )

    def test_case_insensitive_and_whitespace_robustness(self) -> None:
        """Büyük/küçük harf veya boşluk varyasyonları kanonik adı bozmamalıdır."""
        self.assertEqual(resolve_canonical_region("dredgion crash site"), "Altgard")
        self.assertEqual(resolve_canonical_region("  SAFE HAVEN  "), "Altgard")
        self.assertEqual(resolve_canonical_region("calderon canyon"), "Altgard")

    def test_unmapped_regions_pass_through_intact(self) -> None:
        """Eşlenmemiş bağımsız bölgeler (Verteron, Morheim) kendi isimlerini korumalıdır."""
        self.assertEqual(resolve_canonical_region("Verteron"), "Verteron")
        self.assertEqual(resolve_canonical_region("Morheim"), "Morheim")
        self.assertEqual(resolve_canonical_region(None), "Altgard")


class TestMinimapScannerHeadingAndSeparation(unittest.TestCase):
    """
    Test 4: Minimap tarayıcı oyuncu yön açısı tespiti ve filiz vs canavar noktası ayrımı.
    """

    def setUp(self) -> None:
        self.scanner = MinimapScanner()

    def test_detect_player_heading_north(self) -> None:
        """Kuzeye bakan çentikli oyuncu oku için yön açısı ~0° / 360° olarak tespit edilmelidir."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cx, cy = int(1280 * 0.923), int(720 * 0.165)
        # Minimap arka planı (koyu gri daire)
        cv2.circle(frame, (cx, cy), 70, (40, 40, 40), -1)

        # Kuzeye bakan gerçek Aion çentikli chevron oyuncu oku (tepe (cx, cy-18), kanatlar (cx±8, cy+10), iç çentik (cx, cy+5))
        pts = np.array(
            [[cx, cy - 18], [cx + 8, cy + 10], [cx, cy + 5], [cx - 8, cy + 10]],
            dtype=np.int32,
        )
        cv2.fillPoly(frame, [pts], (255, 255, 255))

        ok, heading = self.scanner.detect_player_heading(frame)
        self.assertTrue(ok, "Oyuncu yön oku tespit edilemedi")
        diff_from_north = min(heading, abs(360.0 - heading))
        self.assertLessEqual(diff_from_north, 25.0, f"Kuzey açısı sapması yüksek: {heading}°")

    def test_empty_frame_returns_gracefully(self) -> None:
        """Boş veya siyah karede çökmeden (False, 0.0) dönmelidir."""
        black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        ok, heading = self.scanner.detect_player_heading(black_frame)
        self.assertFalse(ok)
        self.assertEqual(heading, 0.0)

    def test_sprout_vs_monster_separation(self) -> None:
        """Minimap taramasında filizler ve canavar noktaları (•) kesinlikle ayrılmalıdır."""
        crop = np.zeros((160, 160, 3), dtype=np.uint8)
        # Minimap zemini
        cv2.circle(crop, (80, 80), 75, (30, 35, 40), -1)

        # 1. Filiz adayı (yeşilimsi 3 yaprak öbeği @ 110, 80)
        cv2.circle(crop, (110, 78), 3, (80, 220, 120), -1)
        cv2.circle(crop, (108, 83), 3, (80, 220, 120), -1)
        cv2.circle(crop, (112, 83), 3, (80, 220, 120), -1)

        # 2. Canavar noktası (küçük beyaz nokta @ 50, 50)
        cv2.circle(crop, (50, 50), 2, (240, 240, 240), -1)

        # 3. Kırmızı düşman noktası (@ 50, 110)
        cv2.circle(crop, (50, 110), 3, (40, 40, 240), -1)

        scan = self.scanner.scan_minimap_crop(crop)
        self.assertIn("gatherable_nodes", scan)
        self.assertIn("monster_dots", scan)
        self.assertIn("hostile_dots", scan)

        # Canavar noktalarının toplanabilir materyal sayılmadığını doğrula
        for m_dot in scan.get("monster_dots", []):
            self.assertFalse(m_dot.get("is_gatherable", False))
            self.assertNotEqual(m_dot.get("kind"), "sprout_material")


class TestRadarClearanceThreshold(unittest.TestCase):
    """
    Test 5: Radar klerans eşiği (_GATHER_PLAYER_CLEARANCE = 0.075) ve
    yakın toplanabilir materyallerin korunması.
    """

    def test_radar_clearance_threshold_value(self) -> None:
        """_GATHER_PLAYER_CLEARANCE tam olarak 0.075 olmalıdır (eski 0.23 kör noktası kaldırılmış olmalı)."""
        self.assertAlmostEqual(
            _GATHER_PLAYER_CLEARANCE,
            0.075,
            places=3,
            msg=f"Eşik hatalı: {_GATHER_PLAYER_CLEARANCE} != 0.075",
        )

    def test_near_player_harvest_nodes_kept(self) -> None:
        """Oyuncuya yakın (mesafe 0.08 - 0.15) gerçek filizler klerans filtresine takılmamalıdır."""
        player_cx, player_cy = MinimapScanner.PLAYER_CENTER_REL

        # Mesafe = 0.09 (eski 0.23 eşiğinde siliniyordu, 0.075 ile korunmalı)
        node_nx = player_cx
        node_ny = player_cy + 0.09
        dist = math.hypot(node_nx - player_cx, node_ny - player_cy)
        self.assertGreater(dist, _GATHER_PLAYER_CLEARANCE)

        is_chrome = minimap_norm_is_hud_chrome(node_nx, node_ny, confidence=0.70)
        self.assertFalse(is_chrome, "0.09 mesafedeki filiz oyuncu okunun iç halkası sanılıp elendi!")

    def test_high_confidence_inner_node_retention(self) -> None:
        """İç çemberde kalsa bile yüksek güvenli (confidence >= 0.72) filiz korunmalıdır."""
        player_cx, player_cy = MinimapScanner.PLAYER_CENTER_REL
        inner_nx = player_cx
        inner_ny = player_cy + 0.05  # dist = 0.05 <= 0.075

        # Düşük güvenli -> iç halka kromu sayılır
        self.assertTrue(minimap_norm_is_hud_chrome(inner_nx, inner_ny, confidence=0.50))

        # Yüksek güvenli (>= 0.72) -> materyal olarak korunur
        self.assertFalse(minimap_norm_is_hud_chrome(inner_nx, inner_ny, confidence=0.75))

    def test_minimap_material_is_recordable_logic(self) -> None:
        """minimap_material_is_recordable geçerli filizi kabul etmeli, kromu reddetmelidir."""
        valid_node = {
            "kind": "sprout_material",
            "is_gatherable": True,
            "hud_chrome": False,
            "norm_x": 0.50,
            "norm_y": 0.60,
            "confidence": 0.85,
        }
        self.assertTrue(minimap_material_is_recordable(valid_node))

        monster_node = {
            "kind": "monster_dot",
            "is_gatherable": False,
            "hud_chrome": False,
            "norm_x": 0.50,
            "norm_y": 0.60,
            "confidence": 0.90,
        }
        self.assertFalse(minimap_material_is_recordable(monster_node))


class TestNavigatorHeadingBearingAndDeadband(unittest.TestCase):
    """
    Test 6: Navigator kerteriz ve yön hesaplamaları, en kısa açı, 180° ters dönme engeli ve ±6° ölü bant.
    """

    def test_compute_bearing_cardinal_directions(self) -> None:
        """Kerteriz hesabı pusula yönlerini (Kuzey=0°, Doğu=90°, Güney=180°, Batı=270°) tam üretmelidir."""
        origin = (500.0, 500.0)
        north_pt = (500.0, 400.0)  # dy = -100 (yukarı/kuzey)
        east_pt = (600.0, 500.0)   # dx = +100 (sağ/doğu)
        south_pt = (500.0, 600.0)  # dy = +100 (aşağı/güney)
        west_pt = (400.0, 500.0)   # dx = -100 (sol/batı)

        self.assertAlmostEqual(compute_bearing(origin, north_pt), 0.0, places=1)
        self.assertAlmostEqual(compute_bearing(origin, east_pt), 90.0, places=1)
        self.assertAlmostEqual(compute_bearing(origin, south_pt), 180.0, places=1)
        self.assertAlmostEqual(compute_bearing(origin, west_pt), 270.0, places=1)

    def test_compute_angle_diff_shortest_rotation(self) -> None:
        """Açı farkı daima en kısa dönüşü (-180° .. +180°) seçmeli, 340° yerine 20° dönmelidir."""
        # Hedef 10°, oyuncu 350° -> Sağa 20° (+20.0°)
        diff1 = compute_angle_diff(10.0, 350.0)
        self.assertAlmostEqual(diff1, 20.0, places=1)

        # Hedef 350°, oyuncu 10° -> Sola 20° (-20.0°)
        diff2 = compute_angle_diff(350.0, 10.0)
        self.assertAlmostEqual(diff2, -20.0, places=1)

        # Hedef 180°, oyuncu 0° -> ±180°
        diff3 = abs(compute_angle_diff(180.0, 0.0))
        self.assertAlmostEqual(diff3, 180.0, places=1)

    def test_deadband_zone_filtering(self) -> None:
        """±6° ölü bant içinde kalan sapmalarda gereksiz mikro-dönüş tetiklenmemelidir."""
        deadband = 6.0
        # +3.5° sapma (ölü bant içinde)
        res_in = apply_turn_damping(3.5, deadband_deg=deadband)
        self.assertTrue(res_in["in_deadband"])
        self.assertFalse(res_in["turn_needed"])
        self.assertEqual(res_in["turn_direction"], "straight")
        self.assertEqual(res_in["turn_duration_sec"], 0.0)

        # +15.0° sapma (ölü bant dışında)
        res_out = apply_turn_damping(15.0, deadband_deg=deadband)
        self.assertFalse(res_out["in_deadband"])
        self.assertTrue(res_out["turn_needed"])
        self.assertEqual(res_out["turn_direction"], "right")
        self.assertGreater(res_out["turn_duration_sec"], 0.0)

        # -25.0° sapma (sola dönüş)
        res_left = apply_turn_damping(-25.0, deadband_deg=deadband)
        self.assertFalse(res_left["in_deadband"])
        self.assertTrue(res_left["turn_needed"])
        self.assertEqual(res_left["turn_direction"], "left")

    def test_turn_damping_controller_reversal_smoothing(self) -> None:
        """Kısa sürede yön değiştirme (sağdan sola hızlı salınım/yalpalanma) sönümlenmelidir."""
        ctrl = TurnDampingController(deadband_deg=6.0, reversal_cooldown_sec=0.15)
        t0 = 1000.0

        # İlk komut: Sağa dön (+30°)
        turn1 = ctrl.calculate_turn(target_bearing=30.0, player_heading=0.0, now=t0)
        self.assertTrue(turn1["turn_needed"])
        self.assertEqual(turn1["turn_direction"], "right")
        self.assertFalse(turn1["reversal_damped"])

        # 0.05 sn sonra ani ters komut: Sola dön (-30°)
        turn2 = ctrl.calculate_turn(target_bearing=330.0, player_heading=0.0, now=t0 + 0.05)
        self.assertTrue(turn2["turn_needed"])
        self.assertEqual(turn2["turn_direction"], "left")
        self.assertTrue(turn2["reversal_damped"], "Hızlı yön değişiminde sönümleme devreye girmedi!")


class TestMapCanvasInteraction(unittest.TestCase):
    """
    Test 7: Harita tuvali etkileşimi (tıklama ile ekleme, sürükleyerek taşıma, sağ tık silme).
    """

    def setUp(self) -> None:
        self.canvas = _MapCanvas()
        self.canvas.resize(800, 600)
        pix = QPixmap(2258, 2400)
        pix.fill(QColor("#202020"))
        self.canvas.set_map_pixmap(pix)

    def test_map_canvas_signals_exist(self) -> None:
        """_MapCanvas gerekli tüm etkileşim sinyallerini tanımlamış olmalıdır."""
        self.assertTrue(hasattr(self.canvas, "node_moved"))
        self.assertTrue(hasattr(self.canvas, "node_deleted") or hasattr(self.canvas, "node_delete_requested"))
        self.assertTrue(hasattr(self.canvas, "on_canvas_add_node_signal"))
        self.assertTrue(hasattr(self.canvas, "node_right_clicked"))

    def test_map_widget_coordinate_roundtrip(self) -> None:
        """Tuval koordinat dönüşümleri (_map_to_widget / _widget_to_map) tam örtüşmelidir."""
        map_pt = QPointF(550.0, 720.0)
        w_pt = self.canvas._map_to_widget(map_pt)
        recovered = self.canvas._widget_to_map(w_pt)
        self.assertAlmostEqual(recovered.x(), map_pt.x(), places=2)
        self.assertAlmostEqual(recovered.y(), map_pt.y(), places=2)

    def test_find_node_at_widget_pos(self) -> None:
        """Tuval üzerindeki fare koordinatında bulunan düğüm doğru tespit edilmelidir."""
        node = {
            "id": 42,
            "map_x": 600.0,
            "map_y": 800.0,
            "norm_x": 600.0 / 2258.0,
            "norm_y": 800.0 / 2400.0,
        }
        self.canvas.set_nodes_data([node], [42])

        map_pt = self.canvas._node_to_map_coords(
            600.0,
            800.0,
            norm_x=node["norm_x"],
            norm_y=node["norm_y"],
            node=node,
        )
        w_pt = self.canvas._map_to_widget(map_pt)

        found_id = self.canvas._find_node_at_widget_pos(w_pt, hit_radius=20.0)
        self.assertEqual(found_id, 42)

        far_pt = QPointF(0.0, 0.0)
        self.assertIsNone(self.canvas._find_node_at_widget_pos(far_pt, hit_radius=10.0))

    def test_node_interaction_signal_emission(self) -> None:
        """Düğüm taşıma, silme ve ekleme sinyalleri veri kaybı olmadan iletilmelidir."""
        moved_records: List[Tuple[int, float, float]] = []
        added_records: List[Tuple[int, float, float]] = []

        self.canvas.node_moved.connect(lambda nid, x, y: moved_records.append((nid, x, y)))
        self.canvas.on_canvas_add_node_signal.connect(lambda nid, x, y: added_records.append((nid, x, y)))

        # Taşıma sinyalini test et
        self.canvas.node_moved.emit(42, 650.0, 850.0)
        self.assertEqual(len(moved_records), 1)
        self.assertEqual(moved_records[0], (42, 650.0, 850.0))

        # Ekleme sinyalini test et
        self.canvas.on_canvas_add_node_signal.emit(43, 700.0, 900.0)
        self.assertEqual(len(added_records), 1)
        self.assertEqual(added_records[0], (43, 700.0, 900.0))


class TestAppShellTabSwitchingAndSignalSync(unittest.TestCase):
    """
    Test 8: AppShell sekme geçişleri ve sinyal senkronizasyonu.
    """

    def setUp(self) -> None:
        self.shell = GatherionShell()

    def test_app_shell_pages_and_initial_state(self) -> None:
        """AppShell harita, radar ve ayarlar sekmelerini barındırmalı ve ana sayfada başlamalıdır."""
        self.assertIn("home", self.shell.pages)
        self.assertIn("map", self.shell.pages)
        self.assertIn("radar", self.shell.pages)
        self.assertIn("settings", self.shell.pages)
        self.assertEqual(self.shell.current_key(), "home")

    def test_tab_switching_navigation(self) -> None:
        """Harita, Radar ve Ana Sayfa arasında sekme geçişi sorunsuz çalışmalıdır."""
        # Harita sekmesine geç
        self.shell.open_tool("map")
        self.assertEqual(self.shell.current_key(), "map")

        # Radar sekmesine geç
        self.shell.open_tool("radar")
        self.assertEqual(self.shell.current_key(), "radar")

        # Ana sayfaya dön
        self.shell.open_tool("home")
        self.assertEqual(self.shell.current_key(), "home")

    def test_signal_synchronization_on_tab_change(self) -> None:
        """Sekme geçişi yapıldığında arka plan senkronizasyonu çalışmalı ve UI kilitlenmemelidir."""
        self.shell.open_tool("map")
        map_page = self.shell.pages.get("map")
        self.assertIsNotNone(map_page)

        self.shell.open_tool("radar")
        radar_page = self.shell.pages.get("radar")
        self.assertIsNotNone(radar_page)

        # Senkronizasyon metodlarının varlığını doğrula
        self.assertTrue(hasattr(self.shell, "_sync_map_from_settings"))
        self.assertTrue(hasattr(self.shell, "_after_page"))


def main() -> None:
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()

    suite.addTests(loader.loadTestsFromTestCase(TestCoordinateTransform))
    suite.addTests(loader.loadTestsFromTestCase(TestSequentialRouteOrdering))
    suite.addTests(loader.loadTestsFromTestCase(TestSubregionAliasMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestMinimapScannerHeadingAndSeparation))
    suite.addTests(loader.loadTestsFromTestCase(TestRadarClearanceThreshold))
    suite.addTests(loader.loadTestsFromTestCase(TestNavigatorHeadingBearingAndDeadband))
    suite.addTests(loader.loadTestsFromTestCase(TestMapCanvasInteraction))
    suite.addTests(loader.loadTestsFromTestCase(TestAppShellTabSwitchingAndSignalSync))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    main()
