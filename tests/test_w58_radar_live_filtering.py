# -*- coding: utf-8 -*-
"""
W58: Canlı Radar Filtrelemesi ve classify_minimap_crop_bgr Birim Testleri.
Oyuncu oku, HUD yazıları (bölge başlığı, mesafe etiketi, saat, büyüteç) ve
canavar noktalarının dışlanıp yalnızca gerçek filizlerin (sprout_material)
kabul edildiğini doğrular.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import cv2
import numpy as np

_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ui.cartography_radar import (
    classify_minimap_crop_bgr,
    find_zone_distance_bbox,
    minimap_material_is_recordable,
    minimap_norm_is_hud_chrome,
)


def _blank_minimap(width: int = 200, height: int = 200) -> np.ndarray:
    """Koyu zeminli sentetik minimap kırpımı üretir."""
    return np.full((height, width, 3), 24, dtype=np.uint8)


def test_classify_minimap_crop_accepts_only_genuine_sprouts_and_separates_monsters():
    """
    Gerçek filiz (sprout_material, is_gatherable=True) materyal listesine kabul edilmeli;
    filiz olmayan adaylar veya is_gatherable=False olanlar materyal listesine girmemeli,
    canavar noktaları ise yalnızca monsters listesinde yer almalıdır.
    """
    crop = _blank_minimap(200, 200)
    fake_scan = {
        "player_center": (100.0, 108.0),
        "gatherable_nodes": [
            {
                "pos": (60.0, 90.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.88,
                "harvests_available": 6,
            },
            {
                "pos": (140.0, 100.0),
                "kind": "player_arrow",
                "is_gatherable": True,
                "confidence": 0.95,
            },
            {
                "pos": (70.0, 130.0),
                "kind": "sprout_material",
                "is_gatherable": False,
                "confidence": 0.91,
            },
        ],
        "monster_dots": [
            {"pos": (55.0, 120.0)},
        ],
        "hostile_dots": [
            {"pos": (145.0, 125.0)},
        ],
    }

    with patch("ui.cartography_radar._panel_minimap_scanner") as mock_scanner_fn:
        mock_scanner_fn.return_value.scan_minimap_crop.return_value = fake_scan
        materials, monsters = classify_minimap_crop_bgr(crop)

    assert len(materials) == 1
    mat = materials[0]
    assert mat["kind"] == "sprout_material"
    assert mat["is_gatherable"] is True
    assert mat["hud_chrome"] is False
    assert mat["stack_count"] == 6
    assert mat["stack_badge"] == "6x"
    assert mat["norm_x"] == 0.3
    assert mat["norm_y"] == 0.45

    assert len(monsters) == 2
    assert all(m["is_gatherable"] is False for m in monsters)
    assert {m["kind"] for m in monsters} == {"monster_dot", "hostile_dot"}


def test_classify_minimap_crop_excludes_player_arrow_inner_clearance_low_confidence():
    """
    Oyuncu okunun merkezindeki iç halkada (_GATHER_PLAYER_CLEARANCE) düşük güvenli
    adaylar (oyuncu oku parlaması vb.) elenmeli, ancak yüksek güvenli (>= 0.72) gerçek
    filizler kör noktaya düşmemek için korunmalıdır.
    """
    crop = _blank_minimap(200, 200)
    # Oyuncu merkezi (100, 108); clearance = 0.075 * 200 = 15 px
    fake_scan = {
        "player_center": (100.0, 108.0),
        "gatherable_nodes": [
            # Oyuncu okunun tam üstünde düşük güvenli sahte tespit -> elenmeli
            {
                "pos": (102.0, 109.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.55,
                "harvests_available": 3,
            },
            # Oyuncu yakını yüksek güvenli gerçek filiz -> korunmalı
            {
                "pos": (105.0, 110.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.85,
                "harvests_available": 9,
            },
        ],
        "monster_dots": [
            # Oyuncu oku iç halkasındaki canavar noktası -> elenmeli
            {"pos": (101.0, 107.0)},
        ],
        "hostile_dots": [],
    }

    with patch("ui.cartography_radar._panel_minimap_scanner") as mock_scanner_fn:
        mock_scanner_fn.return_value.scan_minimap_crop.return_value = fake_scan
        materials, monsters = classify_minimap_crop_bgr(crop)

    assert len(materials) == 1
    assert materials[0]["confidence"] == 0.85
    assert materials[0]["stack_count"] == 9
    assert len(monsters) == 0


def test_classify_minimap_crop_excludes_hud_title_magnifier_and_clock_regions():
    """
    Üst bölge başlığı şeridi, sağ üst büyüteç ve sağ alt saat HUD bölgelerine
    denk gelen parlak noktalar filiz olarak kabul edilmemelidir.
    """
    crop = _blank_minimap(200, 200)
    fake_scan = {
        "player_center": (100.0, 108.0),
        "gatherable_nodes": [
            # Üst başlık şeridi (y < 0.18 * 200 = 36)
            {
                "pos": (100.0, 20.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.90,
            },
            # Sağ üst büyüteç (x >= 160, y <= 72)
            {
                "pos": (175.0, 50.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.90,
            },
            # Sağ alt saat (x >= 124, y >= 148)
            {
                "pos": (150.0, 170.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.90,
            },
            # Geçerli orta-sol oyun alanı
            {
                "pos": (50.0, 100.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.82,
            },
        ],
        "monster_dots": [],
        "hostile_dots": [],
    }

    with patch("ui.cartography_radar._panel_minimap_scanner") as mock_scanner_fn:
        mock_scanner_fn.return_value.scan_minimap_crop.return_value = fake_scan
        materials, _ = classify_minimap_crop_bgr(crop)

    assert len(materials) == 1
    assert materials[0]["norm_x"] == 0.25
    assert materials[0]["norm_y"] == 0.5


def test_classify_minimap_crop_excludes_zone_distance_hud_text_bbox():
    """
    Bölge adının altındaki parlak mesafe yazısı ('5225m') kutusu tespit edilmeli ve
    bu yazı kutusunun içine düşen sahte filiz adayları dışlanmalıdır.
    """
    crop = _blank_minimap(200, 200)
    # y=42..52, x=80..120 aralığına parlak HUD mesafe yazısı ('5225m') simülasyonu çiz
    cv2.putText(
        crop,
        "5225m",
        (78, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    dist_box = find_zone_distance_bbox(crop)
    assert dist_box is not None
    bx, by, bw, bh = dist_box
    text_center_x = bx + bw * 0.5
    text_center_y = by + bh * 0.5

    fake_scan = {
        "player_center": (text_center_x, text_center_y),  # Parlak yazı merkezi yukarı çekmeye çalışsa bile
        "gatherable_nodes": [
            # Mesafe yazısının üstündeki sahte filiz adayı -> elenmeli
            {
                "pos": (text_center_x, text_center_y),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.92,
            },
            # Yazı dışındaki gerçek filiz -> kabul edilmeli
            {
                "pos": (55.0, 115.0),
                "kind": "sprout_material",
                "is_gatherable": True,
                "confidence": 0.84,
            },
        ],
        "monster_dots": [],
        "hostile_dots": [],
    }

    with patch("ui.cartography_radar._panel_minimap_scanner") as mock_scanner_fn:
        mock_scanner_fn.return_value.scan_minimap_crop.return_value = fake_scan
        materials, _ = classify_minimap_crop_bgr(crop)

    assert len(materials) == 1
    assert materials[0]["norm_x"] == round(55.0 / 200.0, 4)
    assert materials[0]["norm_y"] == round(115.0 / 200.0, 4)


def test_minimap_material_is_recordable_and_norm_hud_chrome_filtering():
    """
    minimap_norm_is_hud_chrome ve minimap_material_is_recordable fonksiyonlarının
    HUD kromunu, oyuncu oku merkezini ve filiz olmayan kayıtları reddedip yalnızca
    geçerli filizleri onayladığını doğrular.
    """
    # Başlık, büyüteç, saat ve düşük güvenli oyuncu merkezi HUD kromu sayılmalı
    assert minimap_norm_is_hud_chrome(0.50, 0.10, confidence=0.95) is True
    assert minimap_norm_is_hud_chrome(0.85, 0.25, confidence=0.95) is True
    assert minimap_norm_is_hud_chrome(0.75, 0.85, confidence=0.95) is True
    assert minimap_norm_is_hud_chrome(0.50, 0.54, confidence=0.50) is True
    # Yüksek güvenli filiz oyuncu kleransında olsa bile HUD kromu sayılmaz
    assert minimap_norm_is_hud_chrome(0.50, 0.54, confidence=0.80) is False

    valid_sprout = {
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
        "norm_x": 0.35,
        "norm_y": 0.48,
        "confidence": 0.86,
    }
    assert minimap_material_is_recordable(valid_sprout) is True

    # HUD kromu olarak işaretlenmiş, canavar olan veya başlık şeridine düşen düğümler reddedilmeli
    assert minimap_material_is_recordable({**valid_sprout, "hud_chrome": True}) is False
    assert minimap_material_is_recordable({**valid_sprout, "kind": "monster_dot"}) is False
    assert minimap_material_is_recordable({**valid_sprout, "is_gatherable": False}) is False
    assert minimap_material_is_recordable({**valid_sprout, "norm_y": 0.08}) is False
    assert minimap_material_is_recordable({**valid_sprout, "norm_x": 0.50, "norm_y": 0.54, "confidence": 0.45}) is False
