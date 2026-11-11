# -*- coding: utf-8 -*-
"""
Birim Testleri — Map/Radar Worker 3:
Minimap Filtreleme, HD Fit & Coordinate Resolution, SequentialRoutePlanner.
"""

from __future__ import annotations

import math
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui.cartography_radar import (
    _HD_A,
    _HD_B,
    _HD_C,
    _HD_D,
    _HD_TX,
    _HD_TY,
    HD_FIT_OFFSET,
    CANONICAL_REGION_ALIASES,
    SequentialRoutePlanner,
    apply_hd_fit,
    classify_minimap_crop_bgr,
    find_zone_distance_bbox,
    inv_hd_fit,
    minimap_material_is_recordable,
    minimap_norm_is_hud_chrome,
    resolve_canonical_region,
    resolve_node_xy,
)


# =====================================================================
# Görev 1: Player Clearance Azaltma & Kör Nokta Düğüm Koruma
# =====================================================================

def test_gather_player_clearance_boundary_is_behavioral() -> None:
    """Player clearance is pinned by BEHAVIOUR at its boundary, not by a literal.

    F-1557: the old test only asserted ``_GATHER_PLAYER_CLEARANCE == 0.075``,
    which passes for any constant that is never used. This test probes the
    blind zone around the real player centre (MinimapScanner.PLAYER_CENTER_REL)
    with low-confidence nodes: just inside the clearance they are rejected as
    HUD chrome, just outside they are recordable — and the zone is radial,
    not axis-aligned. Any change to the clearance value (or to the code
    ignoring it) flips at least one of these probes.
    """
    from core.minimap_scanner import MinimapScanner

    cx, cy = MinimapScanner.PLAYER_CENTER_REL
    assert 0.0 < cx < 1.0 and 0.0 < cy < 1.0

    def _node(nx: float, ny: float, confidence: float = 0.50) -> dict:
        return {
            "norm_x": nx,
            "norm_y": ny,
            "confidence": confidence,
            "kind": "sprout_material",
            "is_gatherable": True,
            "hud_chrome": False,
        }

    # Tight straddle of the 0.075 boundary on the +x axis (low confidence).
    assert minimap_norm_is_hud_chrome(cx + 0.074, cy, confidence=0.50) is True
    assert minimap_material_is_recordable(_node(cx + 0.074, cy)) is False
    assert minimap_norm_is_hud_chrome(cx + 0.076, cy, confidence=0.50) is False
    assert minimap_material_is_recordable(_node(cx + 0.076, cy)) is True

    # The zone is radial: the same distances on the diagonal behave identically.
    d = 0.074 / math.sqrt(2.0)
    assert minimap_norm_is_hud_chrome(cx + d, cy + d, confidence=0.50) is True
    d = 0.076 / math.sqrt(2.0)
    assert minimap_norm_is_hud_chrome(cx + d, cy + d, confidence=0.50) is False

    # Well outside the old 0.23 clearance a low-confidence node is recordable,
    # proving the clearance really shrank (not just the constant's spelling).
    assert minimap_norm_is_hud_chrome(cx + 0.12, cy, confidence=0.50) is False
    assert minimap_material_is_recordable(_node(cx + 0.12, cy)) is True


def test_player_clearance_blind_zone_low_confidence_rejected() -> None:
    """
    Oyuncu merkezine yakın (dist <= 0.075) ve düşük güvenli (confidence < 0.72)
    noktalar oyuncu oku sanılarak elenmelidir.
    """
    # Merkez (0.502, 0.545). Nokta (0.52, 0.55) -> dist ~ 0.018 <= 0.075
    low_conf_node = {
        "norm_x": 0.52,
        "norm_y": 0.55,
        "confidence": 0.50,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.52, 0.55, confidence=0.50) is True
    assert minimap_material_is_recordable(low_conf_node) is False


def test_player_adjacent_harvest_node_high_confidence_retained() -> None:
    """
    Oyuncu merkezine çok yakın (dist <= 0.075) olsa dahi güveni >= 0.72 olan
    toplanabilir filizler (sprout_material) kör noktaya düşürülmeyip KORUNMALIDIR.
    """
    # Merkez (0.502, 0.545). Mesafe ~ 0.02 <= 0.075
    high_conf_node = {
        "norm_x": 0.52,
        "norm_y": 0.55,
        "confidence": 0.75,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.52, 0.55, confidence=0.75) is False
    assert minimap_material_is_recordable(high_conf_node) is True

    # Eşik tam 0.72 durumunda da korunmalı
    exact_conf_node = {
        "norm_x": 0.51,
        "norm_y": 0.54,
        "confidence": 0.72,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.51, 0.54, confidence=0.72) is False
    assert minimap_material_is_recordable(exact_conf_node) is True


def test_reduced_clearance_rescues_player_adjacent_harvest_nodes() -> None:
    """
    Mesafe 0.12 (eski 0.23 kleransında elenen ama 0.075 kleransında serbest kalan)
    düğüm, güveni normal olsa bile artık elenmemelidir.
    """
    # Merkez (0.502, 0.545). Nokta (0.60, 0.55) -> dist ~ 0.098 (> 0.075 ve < 0.23)
    rescued_node = {
        "norm_x": 0.60,
        "norm_y": 0.55,
        "confidence": 0.65,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.60, 0.55, confidence=0.65) is False
    assert minimap_material_is_recordable(rescued_node) is True


def test_hud_chrome_still_rejects_sprouts_even_if_high_confidence() -> None:
    """
    Başlık (title), saat (clock) ve büyüteç (magnifier) gerçek HUD kromudur;
    buralardaki noktalar güveni yüksek olsa bile materyal sayılamaz.
    """
    # Başlık şeridi (ny < 0.18)
    title_node = {
        "norm_x": 0.50,
        "norm_y": 0.10,
        "confidence": 0.99,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.50, 0.10, confidence=0.99) is True
    assert minimap_material_is_recordable(title_node) is False

    # Saat bölgesi (nx >= 0.62, ny >= 0.74)
    clock_node = {
        "norm_x": 0.70,
        "norm_y": 0.85,
        "confidence": 0.95,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.70, 0.85, confidence=0.95) is True
    assert minimap_material_is_recordable(clock_node) is False

    # Büyüteç bölgesi (nx >= 0.80, ny <= 0.36)
    mag_node = {
        "norm_x": 0.85,
        "norm_y": 0.20,
        "confidence": 0.90,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_norm_is_hud_chrome(0.85, 0.20, confidence=0.90) is True
    assert minimap_material_is_recordable(mag_node) is False


# =====================================================================
# Görev 2: HD Fit & Coordinate Resolution (Altgard HD 2258x2400)
# =====================================================================

def test_hd_fit_constants() -> None:
    """_HD_A, _HD_D ve HD_FIT_OFFSET sabitleri doğru atanmalıdır.

    F-1338: the old test only checked apply_hd_fit(280, 80) against
    (_HD_TX, _HD_TY) — but HD_FIT_OFFSET is DEFINED from that exact anchor
    point, so the assertion could never fail. This test verifies the affine
    formula at several NON-anchor points and that the fit is not a constant
    function returning the anchor; the anchor itself is kept only as a
    documented-target check among the others.
    """
    assert abs(_HD_A - 1.46687753) < 1e-6
    assert abs(_HD_D - 1.42560995) < 1e-6
    ox, oy = HD_FIT_OFFSET
    non_anchor_points = [
        (0.0, 0.0),
        (550.0, 320.0),
        (640.0, 360.0),
        (100.0, 50.0),
        (1200.0, 700.0),
    ]
    for x, y in non_anchor_points:
        fit_x, fit_y = apply_hd_fit(x, y)
        assert abs(fit_x - (_HD_A * x + _HD_B * y + ox)) < 1e-6
        assert abs(fit_y - (_HD_C * x + _HD_D * y + oy)) < 1e-6
    # The fit must actually vary with its input (not hardcode the anchor).
    assert apply_hd_fit(550.0, 320.0) != apply_hd_fit(280.0, 80.0)
    assert apply_hd_fit(100.0, 50.0) != apply_hd_fit(1200.0, 700.0)
    # Anchor point maps to the documented HD target.
    fit_x, fit_y = apply_hd_fit(280.0, 80.0)
    assert abs(fit_x - _HD_TX) < 1e-4
    assert abs(fit_y - _HD_TY) < 1e-4


def test_inv_hd_fit_reversibility() -> None:
    """inv_hd_fit, apply_hd_fit dönüşümünü milimetrik hassasiyetle tersine çevirmelidir."""
    test_points = [
        (280.0, 80.0),
        (550.0, 320.0),
        (640.0, 360.0),
        (100.0, 50.0),
        (1200.0, 700.0),
    ]
    for orig_x, orig_y in test_points:
        hd_x, hd_y = apply_hd_fit(orig_x, orig_y)
        rev_x, rev_y = inv_hd_fit(hd_x, hd_y)
        assert abs(rev_x - orig_x) < 1e-4
        assert abs(rev_y - orig_y) < 1e-4


def test_resolve_node_xy_regular_and_hd() -> None:
    """resolve_node_xy 1280x720 video çerçevesi ve 2258x2400 HD harita hizalamasını desteklemelidir."""
    node_dict = {"map_x": 280.0, "map_y": 80.0, "norm_x": 280.0 / 1280.0, "norm_y": 80.0 / 720.0}

    # Normal 1280x720
    rx, ry = resolve_node_xy(node_dict, 1280.0, 720.0, apply_hd=False)
    assert abs(rx - 280.0) < 1e-2
    assert abs(ry - 80.0) < 1e-2

    # apply_hd=True ile Altgard HD dönüşümü
    hx1, hy1 = resolve_node_xy(node_dict, apply_hd=True)
    assert abs(hx1 - _HD_TX) < 1e-3
    assert abs(hy1 - _HD_TY) < 1e-3

    # ref=(2258, 2400) ile doğrudan Altgard HD hedeflemesi
    hx2, hy2 = resolve_node_xy(node_dict, 2258.0, 2400.0)
    assert abs(hx2 - _HD_TX) < 1e-3
    assert abs(hy2 - _HD_TY) < 1e-3

    # Zaten HD koordinatında olan bir düğüm ikinci kez ölçeklenmemeli
    already_hd_node = {"map_x": 1850.0, "map_y": 1200.0, "norm_x": 0.819, "norm_y": 0.50}
    ahx, ahy = resolve_node_xy(already_hd_node, 2258.0, 2400.0)
    assert abs(ahx - 1850.0) < 1e-2
    assert abs(ahy - 1200.0) < 1e-2


def test_find_zone_distance_bbox_hd_fit_scaling() -> None:
    """find_zone_distance_bbox apply_hd=True ile HD piksel alanına ölçeklenmelidir."""
    crop = np.full((154, 166, 3), 40, np.uint8)
    # distance text '5225m' simülasyonu
    cv2.putText(crop, "5225m", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    base_box = find_zone_distance_bbox(crop, apply_hd=False)
    assert base_box is not None
    bx, by, bw, bh = base_box

    hd_box = find_zone_distance_bbox(crop, apply_hd=True)
    assert hd_box is not None
    h_bx, h_by, h_bw, h_bh = hd_box

    # HD genişlik ve yükseklik _HD_A ve _HD_D ile ölçeklenmiş olmalı
    assert abs(h_bw - round(bw * _HD_A)) <= 2
    assert abs(h_bh - round(bh * _HD_D)) <= 2


# =====================================================================
# Görev 3: SequentialRoutePlanner (Döngüsel Sıralama, Metrikler, Alt Bölge)
# =====================================================================

def test_canonical_region_aliases() -> None:
    """'Dredgion Crash Site', 'Safe Haven' vb. alt bölgeler 'Altgard' kanonik adına eşlenmelidir."""
    aliases_to_test = [
        ("Dredgion Crash Site", "Altgard"),
        ("dredgion crash site", "Altgard"),
        ("Safe Haven", "Altgard"),
        ("Calderon Canyon", "Altgard"),
        ("Silent Hill", "Altgard"),
        ("Altgard Fortress", "Altgard"),
        ("Moslan Forest", "Altgard"),
        ("Altgard", "Altgard"),
    ]
    planner = SequentialRoutePlanner()
    for sub, expected in aliases_to_test:
        assert resolve_canonical_region(sub) == expected
        assert planner.canonical_region(sub) == expected
        assert planner.get_canonical_alias(sub) == expected
        assert planner.is_subregion_of(sub, "Altgard") is True


def test_sequential_route_planner_cyclical_sequencing() -> None:
    """
    SequentialRoutePlanner optimize_smooth_loop döngüsel sıralama üretmeli,
    her düğüme 1..N seq_index atamalı ve cyclical_next / get_next_in_sequence
    rotanın sonundan başına (N -> 1) kesintisiz sarmalıdır.
    """
    planner = SequentialRoutePlanner()
    # 5 noktalı rastgele dağınık rota
    input_nodes = [
        {"id": 101, "map_x": 100.0, "map_y": 100.0},
        {"id": 102, "map_x": 500.0, "map_y": 500.0},
        {"id": 103, "map_x": 100.0, "map_y": 500.0},
        {"id": 104, "map_x": 500.0, "map_y": 100.0},
        {"id": 105, "map_x": 300.0, "map_y": 300.0},
    ]

    ordered = planner.optimize_smooth_loop(input_nodes)
    assert len(ordered) == 5

    # Her düğüme ardışık 1..5 seq_index verilmiş olmalı
    for idx, node in enumerate(ordered, start=1):
        assert node["seq_index"] == idx

    # Döngüsel sarma (cyclical wrap-around): son düğümün ardından ilk düğüme dönülmeli
    for i in range(len(ordered) - 1):
        nxt = planner.get_next_in_sequence(ordered[i], ordered, loop=True)
        assert nxt["id"] == ordered[i + 1]["id"]

    # Son düğümden (index 4) ilk düğüme (index 0) sarma
    wrap_next = planner.get_next_in_sequence(ordered[-1], ordered, loop=True)
    assert wrap_next["id"] == ordered[0]["id"]

    # cyclical_next de aynı şekilde çalışmalı
    assert planner.cyclical_next(ordered[-1], ordered)["id"] == ordered[0]["id"]


def test_sequential_route_planner_plan_cyclical_route_with_subregion() -> None:
    """plan_cyclical_route alt bölge adını ('Dredgion Crash Site') 'Altgard' olarak bağlamalıdır."""
    planner = SequentialRoutePlanner()
    nodes = [
        {"id": 1, "map_x": 50.0, "map_y": 50.0},
        {"id": 2, "map_x": 150.0, "map_y": 50.0},
        {"id": 3, "map_x": 150.0, "map_y": 150.0},
    ]
    res = planner.plan_cyclical_route(nodes, region_name="Dredgion Crash Site")
    assert len(res) == 3
    for n in res:
        assert n["region"] == "Altgard"
        assert 1 <= n["seq_index"] <= 3


def test_sequential_route_planner_insert_node_in_sequence() -> None:
    """Dinamik araya ekleme (insert_node_in_sequence) döngüsel düzeni korumalı ve seq_index'leri güncellemelidir."""
    planner = SequentialRoutePlanner()
    # Kare şeklinde 4 nokta: (0,0), (100,0), (100,100), (0,100)
    nodes = [
        {"id": 1, "map_x": 0.0, "map_y": 0.0, "seq_index": 1},
        {"id": 2, "map_x": 100.0, "map_y": 0.0, "seq_index": 2},
        {"id": 3, "map_x": 100.0, "map_y": 100.0, "seq_index": 3},
        {"id": 4, "map_x": 0.0, "map_y": 100.0, "seq_index": 4},
    ]
    # (50, 0) noktası (0,0) ile (100,0) arasına oturmalı
    new_node = {"id": 99, "map_x": 50.0, "map_y": 0.0}
    updated = planner.insert_node_in_sequence(new_node, nodes)
    assert len(updated) == 5
    for idx, n in enumerate(updated, start=1):
        assert n["seq_index"] == idx


def test_sequential_route_planner_metrics_accuracy() -> None:
    """calculate_route_metrics mesafe, adım ve pürüzsüzlük metriklerini doğru hesaplamalıdır."""
    planner = SequentialRoutePlanner(meters_per_pixel=0.5)

    # 100x100 piksel kare: çevre = 400 px, scale=0.5 -> 200.0 metre
    square_nodes = [
        {"map_x": 0.0, "map_y": 0.0},
        {"map_x": 100.0, "map_y": 0.0},
        {"map_x": 100.0, "map_y": 100.0},
        {"map_x": 0.0, "map_y": 100.0},
    ]

    metrics = planner.calculate_route_metrics(square_nodes, meters_per_pixel=0.5)
    assert metrics["discovered_nodes"] == 4
    assert metrics["node_count"] == 4
    assert metrics["total_distance_px"] == 400.0
    assert metrics["total_loop_distance_m"] == 200.0
    assert metrics["total_distance_m"] == 200.0
    assert metrics["avg_step_px"] == 100.0
    assert metrics["avg_step_m"] == 50.0
    assert metrics["max_jump_px"] == 100.0
    assert metrics["max_jump_m"] == 50.0
    assert metrics["min_step_px"] == 100.0
    assert metrics["min_step_m"] == 50.0
    assert metrics["ping_pong_jumps"] == 0
    assert metrics["closed_loop"] is True
    assert metrics["smoothness_score_pct"] >= 90.0

    # evaluate_sequence_quality metodunun da aynı metrikleri döndürdüğünü doğrula
    quality = planner.evaluate_sequence_quality(square_nodes, closed_loop=True)
    assert quality["total_distance_m"] == 200.0
    assert quality["closed_loop"] is True


def test_route_planner_with_hd_fit_enabled() -> None:
    """set_hd_fit(True) durumunda metrikler HD koordinat ölçeğinde hesaplanmalıdır."""
    planner = SequentialRoutePlanner(meters_per_pixel=0.28)
    planner.set_hd_fit(False)
    p1 = {"map_x": 280.0, "map_y": 80.0}
    p2 = {"map_x": 380.0, "map_y": 80.0}
    d_raw = planner._dist(p1, p2)
    assert abs(d_raw - 100.0) < 1e-4

    planner.set_hd_fit(True)
    d_hd = planner._dist(p1, p2)
    # HD dönüşümünde _HD_A ~ 1.46687753 olduğu için 100 piksel ~ 146.68 piksele genişler
    assert abs(d_hd - (100.0 * _HD_A)) < 1.0


def main() -> None:
    test_gather_player_clearance_boundary_is_behavioral()
    test_player_clearance_blind_zone_low_confidence_rejected()
    test_player_adjacent_harvest_node_high_confidence_retained()
    test_reduced_clearance_rescues_player_adjacent_harvest_nodes()
    test_hud_chrome_still_rejects_sprouts_even_if_high_confidence()
    test_hd_fit_constants()
    test_inv_hd_fit_reversibility()
    test_resolve_node_xy_regular_and_hd()
    test_find_zone_distance_bbox_hd_fit_scaling()
    test_canonical_region_aliases()
    test_sequential_route_planner_cyclical_sequencing()
    test_sequential_route_planner_plan_cyclical_route_with_subregion()
    test_sequential_route_planner_insert_node_in_sequence()
    test_sequential_route_planner_metrics_accuracy()
    test_route_planner_with_hd_fit_enabled()
    print("ALL MAP/RADAR WORKER 3 UNIT TESTS PASSED!")


if __name__ == "__main__":
    main()
