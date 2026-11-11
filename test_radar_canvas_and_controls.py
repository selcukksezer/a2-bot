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
    get_cartography_radar_baseline_metrics,
    get_cartography_radar_capabilities,
    validate_cartography_radar_contract,
)


def _count_exact_color_pixels(image: QImage, hex_color: str) -> int:
    """QImage içindeki `hex_color` rengine tam uyan piksel sayısı.

    QImage.Format_ARGB32 bellekte little-endian BGRA düzenindedir, bu yüzden
    hedef renk BGR sırasıyla karşılaştırılır. Tam eşleşme sayımı antialiasing
    kenarlarını saymaz; işaretçi renkleri düz dolguyla çizildiği için
    işaretçi varlığını kanıtlamaya yeter.
    """
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.width(), 4)[:, :, :3]
    target = np.array([b, g, r], dtype=np.uint8)
    return int(np.count_nonzero((arr == target).all(axis=2)))


def _render_to_image(canvas: RadarCanvas, width: int = 640, height: int = 480) -> QImage:
    image = QImage(width, height, QImage.Format.Format_ARGB32)
    image.fill(0)
    painter = QPainter(image)
    try:
        canvas.render(painter)
    finally:
        painter.end()
    return image


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
        # Perspektif hesabı deterministiktir: merkez kutu ~7.9 m'dedir.
        # ±1.0 m bandı, 5..45 m'lik eski gevşek aralığın yerini alır.
        assert p_herb["distance_m"] == pytest.approx(7.9, abs=1.0)
        # Kuzeye baktığı için map_y oyuncudan daha küçük (harita koordinatında yukarı)
        assert p_herb["map_y"] < 500.0
        assert p_herb["map_x"] == pytest.approx(500.0, abs=1.0)
        assert p_herb["map_y"] == pytest.approx(471.8, abs=2.0)

        # 2. Sağ kutu (ore_orichalcum)
        p_ore = projections[1]
        assert p_ore["category"] == "ore"
        assert p_ore["color_hex"] == "#f59e0b"
        assert p_ore["shape"] == "diamond"
        # Sağda olduğu için bearing pozitif açı (doğu yönüne doğru ~15°..30°)
        assert 10.0 < p_ore["bearing_deg"] < 40.0
        assert p_ore["map_x"] > 500.0
        # Ayak hizasına yakın sağ kutu ~4.8 m'dedir.
        assert p_ore["distance_m"] == pytest.approx(4.8, abs=1.0)

        # 3. Sol kutu (monster)
        p_mob = projections[2]
        assert p_mob["category"] == "enemy"
        assert p_mob["color_hex"] == "#ef4444"
        assert p_mob["shape"] == "cross"
        # Solda olduğu için bearing batıya doğru (~330°..350°)
        assert 320.0 < p_mob["bearing_deg"] < 355.0
        assert p_mob["map_x"] < 500.0
        # Uzakta, ufka yakın sol kutu ~16.8 m'dedir.
        assert p_mob["distance_m"] == pytest.approx(16.8, abs=1.5)

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
        # Oyuncu (350, 250) ekran merkezine (400, 300) gelmeli:
        # pan = merkez - oyuncu * zoom, zoom burada 1.0'dır.
        zoom = canvas.get_zoom()
        assert zoom == pytest.approx(1.0, abs=0.01)
        assert pan_x == pytest.approx(400.0 - 350.0 * zoom, abs=0.01)
        assert pan_y == pytest.approx(300.0 - 250.0 * zoom, abs=0.01)
        # Geometrik karşılığı: oyuncunun widget koordinatı tam merkezdir.
        widget_x = pan_x + 350.0 * zoom
        widget_y = pan_y + 250.0 * zoom
        assert widget_x == pytest.approx(400.0, abs=0.5)
        assert widget_y == pytest.approx(300.0, abs=0.5)
        # Yanlış pan değeri merkezi kaydırır: pan +10px olsaydı oyuncu
        # merkezden 10px sapardı; bu eşitlik onu yakalar.


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
        """CartographyRadarWindow takma adı ve kabiliyetler eksiksiz tanımlıdır.

        Kabiliyet listesi yalnızca sabit bir listeyle karşılaştırılmaz: listedeki
        her kabiliyet, üretim kodundaki sözleşme yardımcılarıyla ve gerçek
        davranışla çapraz doğrulanır.
        """
        assert CartographyRadarWindow is CartographyRadarDialog
        caps = get_cartography_radar_capabilities()
        # Tam liste, sırasız karşılaştırma + tekrar yok + hepsi dolu metin.
        expected_caps = {
            "cartography_radar",
            "live_radar_overlay",
            "diagnostic_snapshot",
            "radar_canvas",
            "3d_ai_projection",
            "live_minimap_fusion",
            "offline_simulation",
            "filter_controls",
        }
        assert set(caps) == expected_caps
        assert len(caps) == len(set(caps))
        assert all(isinstance(c, str) and c for c in caps)
        # Üretim sözleşmesi yardımcıları aynı listeyle tutarlı olmalı.
        assert validate_cartography_radar_contract() is True
        metrics = get_cartography_radar_baseline_metrics()
        assert metrics["capability_count"] == len(caps)
        assert metrics["contract_valid"] is True

        # Kabiliyet -> gerçek davranış eşleşmesi (sabit liste tekrarı değil):
        dialog = CartographyRadarDialog()
        canvas = dialog.canvas
        assert isinstance(canvas, RadarCanvas)  # radar_canvas
        projections = canvas.project_3d_yolo_boxes(
            [{"bbox": [600, 320, 680, 400], "class_name": "herb_aria"}],
            frame_w=1280,
            frame_h=720,
            heading_deg=0.0,
        )
        assert len(projections) == 1  # 3d_ai_projection
        assert canvas.is_idle() is True
        canvas.update_live_frame(
            frame_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
            yolo_boxes=[{"bbox": [600, 320, 680, 400], "class_name": "herb_aria"}],
            player_pos=(500.0, 500.0, 0.0),
        )
        assert canvas.is_idle() is False  # live_minimap_fusion
        canvas.update_live_frame(frame_bgr=None)
        assert canvas.is_idle() is True  # offline_simulation
        canvas.set_filter("enemies", False)
        assert canvas.get_filter("enemies") is False  # filter_controls


class TestRadarCanvasPaintEventRendering:
    """Offscreen tuval boyama (paintEvent) çökme/performans testi."""

    def test_offscreen_paint_live_and_idle(self) -> None:
        """Tuval hem canlı tespitlerle hem de çevrimdışı ızgarayla hatasız boyanır.

        Eski test hiç assert içermiyordu: boş çizim de geçerdi. Şimdi boyanan
        görüntüdeki işaretçi renkleri sayılır — işaretçiler çizilmezse sayı 0
        olur ve test düşer.
        """
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

        live_image = _render_to_image(canvas)
        # Herb (#10b981) ve canavar (#ef4444) işaretçileri gerçekten çizilmeli.
        assert _count_exact_color_pixels(live_image, "#10b981") >= 10
        assert _count_exact_color_pixels(live_image, "#ef4444") >= 10

        # Çevrimdışı moda geç ve tekrar boya: son tespitler korunduğu için
        # işaretçiler hâlâ görünür olmalı.
        canvas.update_live_frame(frame_bgr=None)
        idle_image = _render_to_image(canvas)
        assert _count_exact_color_pixels(idle_image, "#10b981") >= 10
        assert _count_exact_color_pixels(idle_image, "#ef4444") >= 10

    def test_offscreen_paint_projections_filter_hides_markers(self) -> None:
        """3B izdüşüm filtresi kapalıyken işaretçi renkleri çizilmez (kontrol)."""
        canvas = RadarCanvas()
        canvas.resize(640, 480)
        canvas.set_player_position(300.0, 200.0, angle_deg=35.0)
        canvas.set_filter("projections", False)
        canvas.update_live_frame(
            frame_bgr=np.zeros((720, 1280, 3), dtype=np.uint8),
            yolo_boxes=[
                {"bbox": [640, 360, 700, 420], "class_name": "herb_aria"},
                {"bbox": [300, 200, 350, 250], "class_name": "monster"},
            ],
            player_pos=(300.0, 200.0, 35.0),
        )
        image = _render_to_image(canvas)
        assert _count_exact_color_pixels(image, "#10b981") == 0
        assert _count_exact_color_pixels(image, "#ef4444") == 0

    def test_offscreen_paint_empty_canvas_has_no_markers(self) -> None:
        """Tespitsiz boş tuvalde işaretçi rengi olmaz (sayacın sıfır noktası)."""
        canvas = RadarCanvas()
        canvas.resize(640, 480)
        image = _render_to_image(canvas)
        assert _count_exact_color_pixels(image, "#10b981") == 0
        assert _count_exact_color_pixels(image, "#ef4444") == 0
