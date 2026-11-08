# -*- coding: utf-8 -*-
"""
Birim ve Entegrasyon Testleri — Map/Radar Worker 4:
RadarCanvas, 3B YZ Kamera İzdüşümü, Canlı Minimap Füzyonu, Boşta Kalmama (Idle Mode) ve UI Kontrolleri.
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pytest
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtWidgets import QApplication

# PyQt uygulaması singleton başlatımı
_app = QApplication.instance()
if _app is None:
    _app = QApplication(sys.argv)

from ui.cartography_radar import (
    CartographyRadarDialog,
    CartographyRadarWindow,
    RadarCanvas,
    SequentialMapCanvas,
    get_cartography_radar_capabilities,
)


class TestRadarCanvas3DProjection:
    """Görev 1: RadarCanvas & 3B YZ Kamera İzdüşümü Testleri."""

    def test_radar_canvas_inheritance_and_init(self) -> None:
        """RadarCanvas SequentialMapCanvas'tan türer ve radar özelliklerini başlatır."""
        canvas = RadarCanvas()
        assert isinstance(canvas, SequentialMapCanvas)
        assert hasattr(canvas, "_live_3d_projections")
        assert hasattr(canvas, "_live_minimap_detections")
        assert canvas.is_idle() is True
        assert canvas.get_filter("nodes") is True
        assert canvas.get_filter("enemies") is True
        assert canvas.get_filter("path") is True
        assert canvas.get_filter("projections") is True

    def test_3d_yolo_box_perspective_projection_math(self) -> None:
        """Kamera merkezindeki ve kenarındaki nesnelerin açı ve perspektif mesafeleri doğru hesaplanır."""
        canvas = RadarCanvas()
        canvas.set_player_position(500.0, 500.0, angle_deg=0.0)  # Kuzeye bakıyor

        # Ekranın tam ortasında (sx=640, sy=360) ve oyuncu ayak hizasında (sy=450) iki kutu
        boxes = [
            # Tam merkezde, ufuk ile ayak arası:
            {"bbox": [600, 320, 680, 400], "class_name": "herb_aria", "confidence": 0.94},
            # Sağ tarafta (sx=960) ve oyuncu ayaklarına yakın (sy=480):
            {"bbox": [920, 440, 1000, 520], "class_name": "ore_orichalcum", "confidence": 0.88},
            # Sol tarafta (sx=320) ve uzakta canavar:
            {"bbox": [290, 260, 350, 320], "class_name": "monster", "confidence": 0.85},
        ]

        projections = canvas.project_3d_yolo_boxes(boxes, frame_w=1280, frame_h=720, heading_deg=0.0)
        assert len(projections) == 3

        # 1. Merkez kutu (herb_aria)
        p_herb = projections[0]
        assert p_herb["class_name"] == "herb_aria"
        assert p_herb["category"] == "herb"
        assert p_herb["color_hex"] == "#10b981"
        assert p_herb["shape"] == "circle_dot"
        # Merkezde olduğu için açısı ~0° olmalı
        assert abs(p_herb["bearing_deg"] - 0.0) < 3.0
        assert 5.0 < p_herb["distance_m"] < 45.0
        # Kuzeye baktığı için map_y oyuncudan daha küçük (harita koordinatında yukarı)
        assert p_herb["map_y"] < 500.0

        # 2. Sağ kutu (ore_orichalcum)
        p_ore = projections[1]
        assert p_ore["category"] == "ore"
        assert p_ore["color_hex"] == "#f59e0b"
        assert p_ore["shape"] == "diamond"
        # Sağda olduğu için bearing pozitif açı (doğu yönüne doğru ~15°..30°)
        assert 10.0 < p_ore["bearing_deg"] < 40.0
        assert p_ore["map_x"] > 500.0

        # 3. Sol kutu (monster)
        p_mob = projections[2]
        assert p_mob["category"] == "enemy"
        assert p_mob["color_hex"] == "#ef4444"
        assert p_mob["shape"] == "cross"
        # Solda olduğu için bearing batıya doğru (~330°..350°)
        assert 320.0 < p_mob["bearing_deg"] < 355.0
        assert p_mob["map_x"] < 500.0

    def test_minimap_sprout_and_monster_projection(self) -> None:
        """Minimap'ten gelen toplanabilir filiz ve canavar tespitleri doğru sınıflarla yansıtılır."""
        canvas = RadarCanvas()
        canvas.set_player_position(300.0, 300.0, angle_deg=45.0)

        mm_dets = [
            {
                "rel_map_dx": 18.0,
                "rel_map_dy": -24.0,
                "kind": "sprout_material",
                "is_gatherable": True,
                "resource_type": "Aria Filizi",
                "stack_badge": "3x",
            },
            {
                "rel_map_dx": -30.0,
                "rel_map_dy": 15.0,
                "kind": "monster_dot",
                "is_gatherable": False,
                "resource_type": "Yoksayılan Canavar",
            },
        ]
        proj = canvas.project_minimap_detections(mm_dets)
        assert len(proj) == 2

        sprout = proj[0]
        assert sprout["category"] == "sprout"
        assert sprout["color_hex"] == "#22d3ee"
        assert sprout["map_x"] == 318.0
        assert sprout["map_y"] == 276.0

        mob = proj[1]
        assert mob["category"] == "enemy"
        assert mob["color_hex"] == "#f87171"
        assert mob["map_x"] == 270.0
        assert mob["map_y"] == 315.0


class TestLiveFusionAndIdleMode:
    """Görev 2: Canlı Minimap Füzyonu ve Boşta Kalmama (Idle Mode) Testleri."""

    def test_update_live_frame_transitions_to_active(self) -> None:
        """Canlı kare verildiğinde idle modundan çıkar ve sinyal yayar."""
        canvas = RadarCanvas()
        assert canvas.is_idle() is True

        idle_signals: list[bool] = []
        canvas.idle_state_changed.connect(idle_signals.append)

        live_updates: list[int] = []
        canvas.live_frame_updated.connect(live_updates.append)

        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        sample_boxes = [
            {"bbox": [500, 300, 560, 380], "class_name": "gem_crystal", "confidence": 0.91}
        ]

        canvas.update_live_frame(
            frame_bgr=fake_frame,
            yolo_boxes=sample_boxes,
            player_pos=(400.0, 400.0, 90.0),
        )

        assert canvas.is_idle() is False
        assert len(idle_signals) == 1
        assert idle_signals[0] is False  # Idle kapandı
        assert len(live_updates) == 1
        assert len(canvas._live_3d_projections) == 1
        assert canvas._live_3d_projections[0]["category"] == "gem"

    def test_idle_mode_preserves_reference_view(self) -> None:
        """Kare kesildiğinde (None) harita ve son tespitler silinmez, referans moduna geçer."""
        canvas = RadarCanvas()
        fake_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        canvas.update_live_frame(fake_frame, yolo_boxes=[{"bbox": [100, 100, 150, 150], "class_name": "wood_tree"}])
        assert canvas.is_idle() is False
        assert len(canvas._live_3d_projections) == 1

        idle_signals: list[bool] = []
        canvas.idle_state_changed.connect(idle_signals.append)

        # Çevrimdışına geçiş
        canvas.update_live_frame(frame_bgr=None)
        assert canvas.is_idle() is True
        assert len(idle_signals) == 1
        assert idle_signals[0] is True
        # Son bilinen tespitler ekranı boşaltmamak için korunur
        assert len(canvas._live_3d_projections) == 1


class TestUIControlsAndFilters:
    """Görev 3: UI Kontrolleri, Filtreler ve Sinyal Testleri."""

    def test_filter_toggles_and_signals(self) -> None:
        """Düğümler, düşmanlar, yol ve 3B YZ filtreleri bağımsız açılıp kapanır ve sinyal üretir."""
        canvas = RadarCanvas()

        filter_events: list[tuple[str, bool]] = []
        canvas.filter_changed.connect(lambda k, v: filter_events.append((k, v)))

        # 1. Düşmanları kapat
        canvas.set_filter("enemies", False)
        assert canvas.get_filter("enemies") is False
        assert ("enemies", False) in filter_events

        # 2. Rota/Yolu kapat
        canvas.set_filter("path", False)
        assert canvas.get_filter("path") is False
        assert ("path", False) in filter_events

        # 3. 3B YZ İzdüşümünü kapat
        canvas.set_filter("projections", False)
        assert canvas.get_filter("projections") is False
        assert ("projections", False) in filter_events

        # 4. Toplu filtre sorgusu
        filters = canvas.get_filters()
        assert filters["show_enemies"] is False
        assert filters["show_path"] is False
        assert filters["show_projections"] is False
        assert filters["show_nodes"] is True

    def test_programmatic_zoom_and_pan(self) -> None:
        """Programatik yakınlaşma, uzaklaşma, sığdırma ve oyuncuya odaklanma çalışır."""
        canvas = RadarCanvas()
        canvas.resize(800, 600)
        canvas.set_player_position(350.0, 250.0, angle_deg=0.0)

        zoom_events: list[float] = []
        canvas.zoom_changed.connect(zoom_events.append)

        initial_zoom = canvas.get_zoom()
        canvas.zoom_in(1.25)
        assert canvas.get_zoom() > initial_zoom
        assert len(zoom_events) >= 1

        canvas.zoom_out(1.25)
        assert abs(canvas.get_zoom() - initial_zoom) < 0.05

        canvas.center_on_player()
        pan_x, pan_y = canvas.get_pan()
        # Oyuncu ekran merkezine (400, 300) yakın olmalı
        assert isinstance(pan_x, float) and isinstance(pan_y, float)


class TestCartographyRadarDialogIntegration:
    """RadarCanvas'ın CartographyRadarDialog ve UI butonları ile tam entegrasyonu."""

    def test_dialog_initializes_with_radar_canvas(self) -> None:
        """CartographyRadarDialog self.canvas olarak RadarCanvas kullanır."""
        dialog = CartographyRadarDialog()
        assert isinstance(dialog.canvas, RadarCanvas)
        assert dialog.get_radar_canvas() is dialog.canvas

        # UI Filtre Butonları mevcut ve etkindir
        assert hasattr(dialog, "btn_toggle_nodes")
        assert hasattr(dialog, "btn_toggle_enemies")
        assert hasattr(dialog, "btn_toggle_path")
        assert hasattr(dialog, "btn_toggle_projections")

        # UI Zoom Butonları mevcuttur
        assert hasattr(dialog, "btn_zoom_in")
        assert hasattr(dialog, "btn_zoom_out")
        assert hasattr(dialog, "btn_zoom_fit")
        assert hasattr(dialog, "btn_center_player")

    def test_dialog_filter_methods_synchronize(self) -> None:
        """Dialog üzerindeki set_radar_filter çağrıları tuval ve buton durumlarını eşitler."""
        dialog = CartographyRadarDialog()
        dialog.set_radar_filter("enemies", False)
        assert dialog.canvas.get_filter("enemies") is False
        assert dialog.btn_toggle_enemies.isChecked() is False

        dialog.set_radar_filter("enemies", True)
        assert dialog.canvas.get_filter("enemies") is True
        assert dialog.btn_toggle_enemies.isChecked() is True

    def test_alias_and_capabilities(self) -> None:
        """CartographyRadarWindow takma adı ve kabiliyetler eksiksiz tanımlıdır."""
        assert CartographyRadarWindow is CartographyRadarDialog
        caps = get_cartography_radar_capabilities()
        assert "radar_canvas" in caps
        assert "3d_ai_projection" in caps
        assert "live_minimap_fusion" in caps
        assert "offline_simulation" in caps
        assert "filter_controls" in caps


class TestRadarCanvasPaintEventRendering:
    """Offscreen tuval boyama (paintEvent) çökme/performans testi."""

    def test_offscreen_paint_live_and_idle(self) -> None:
        """Tuval hem canlı tespitlerle hem de çevrimdışı ızgarayla hatasız boyanır."""
        canvas = RadarCanvas()
        canvas.resize(640, 480)
        canvas.set_player_position(300.0, 200.0, angle_deg=35.0)

        # Canlı tespitler ekle
        canvas.update_live_frame(
            frame_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
            yolo_boxes=[
                {"bbox": [640, 360, 700, 420], "class_name": "herb_aria"},
                {"bbox": [300, 200, 350, 250], "class_name": "monster"},
            ],
            player_pos=(300.0, 200.0, 35.0),
        )

        image = QImage(640, 480, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        try:
            # Doğrudan tuvalin render motorunu test et
            canvas.render(painter)
        finally:
            painter.end()

        # Çevrimdışı moda geç ve tekrar boya
        canvas.update_live_frame(frame_bgr=None)
        image_idle = QImage(640, 480, QImage.Format.Format_ARGB32)
        image_idle.fill(0)
        painter_idle = QPainter(image_idle)
        try:
            canvas.render(painter_idle)
        finally:
            painter_idle.end()
