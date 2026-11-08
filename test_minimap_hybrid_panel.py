# -*- coding: utf-8 -*-
"""Offscreen checks for the minimap hybrid scanner panel."""

from __future__ import annotations

import math
import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from core.minimap_scanner import MinimapScanner
from ui.cartography_radar import (
    MinimapLiveCropPreviewWidget,
    classify_minimap_crop_bgr,
    find_zone_distance_bbox,
    minimap_material_is_recordable,
)


def _load(path: str) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise AssertionError(f"unreadable image: {path}")
    return image


def _crop_frame(name: str) -> np.ndarray:
    image = _load(os.path.join(os.path.dirname(__file__), "video_frames", name))
    x, y, width, height = MinimapLiveCropPreviewWidget.crop_box(image.shape[1], image.shape[0])
    return image[y : y + height, x : x + width]


def _paste_sprout(image: np.ndarray, x: int, y: int) -> None:
    patch = cv2.cvtColor(MinimapScanner._CANONICAL_SPROUT_TPL, cv2.COLOR_GRAY2BGR)
    image[y : y + 16, x : x + 16] = patch


def test_crop_box_ignores_distance_text() -> None:
    plain = MinimapLiveCropPreviewWidget.crop_box(1280, 720)
    marked = np.zeros((720, 1280, 3), np.uint8)
    cv2.putText(
        marked,
        "5225m",
        (1100, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    assert MinimapLiveCropPreviewWidget.crop_box(marked.shape[1], marked.shape[0]) == plain
    assert plain == (
        int(1280 * 0.855),
        int(720 * 0.065),
        max(32, int(1280 * 0.130)),
        max(32, int(720 * 0.215)),
    )


def test_dredgion_frame_has_no_placeholder_materials() -> None:
    """frame_00s is the Dredgion crop in the screenshot (clock 08:41)."""
    materials, monsters = classify_minimap_crop_bgr(_crop_frame("frame_00s.jpg"))
    assert materials == [], materials
    assert all(not minimap_material_is_recordable(m) for m in monsters)
    banned = {"Orichalcum Ore", "Aether Gem", "Vitality Herb", "Pure Aether Ore"}
    assert not any(m.get("resource_type") in banned for m in materials)
    placeholder_norms = ((0.28, 0.30), (0.72, 0.26), (0.68, 0.71), (0.24, 0.68))
    for monster in monsters:
        for nx, ny in placeholder_norms:
            assert math.hypot(monster["norm_x"] - nx, monster["norm_y"] - ny) > 0.02


def test_player_chevron_is_not_a_material() -> None:
    materials, _monsters = classify_minimap_crop_bgr(_crop_frame("frame_33s.jpg"))
    for mat in materials:
        assert minimap_material_is_recordable(mat)
        assert not (
            abs(mat["norm_x"] - 0.51) < 0.06 and abs(mat["norm_y"] - 0.34) < 0.06
        ), mat
        assert mat["norm_y"] >= 0.18
        assert not (mat["norm_x"] >= 0.62 and mat["norm_y"] >= 0.74)
    assert materials == [], materials


def test_reference_sprout_counts_and_monsters_are_not_saved() -> None:
    root = os.path.dirname(__file__)
    scanner = MinimapScanner()
    sprout = _load(os.path.join(root, "reference_screenshots", "minimap_sprout_closeup.jpg"))
    sprout_crop, _bounds = scanner.extract_minimap(sprout)
    materials, _monsters = classify_minimap_crop_bgr(sprout_crop)
    assert len(materials) >= 1, materials
    assert all(minimap_material_is_recordable(m) for m in materials)
    assert all(m["kind"] == "sprout_material" for m in materials)

    feather = _load(os.path.join(root, "reference_screenshots", "minimap_filter_closeup.jpg"))
    feather_crop, _bounds = scanner.extract_minimap(feather)
    _feather_mats, monsters = classify_minimap_crop_bgr(feather_crop)
    assert len(monsters) >= 1, monsters
    assert all(m["kind"] == "monster_dot" and m["is_gatherable"] is False for m in monsters)
    assert all(not minimap_material_is_recordable(m) for m in monsters)


def test_synthetic_chrome_is_rejected_and_sprout_counts() -> None:
    height, width = 257, 332
    image = np.full((height, width, 3), (42, 58, 70), np.uint8)
    cv2.rectangle(image, (0, 0), (width - 1, 36), (24, 24, 28), -1)
    cv2.putText(
        image,
        "Dredgion Crash Site",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    text_org = (110, 70)
    cv2.putText(
        image,
        "5225m",
        text_org,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.15,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    (text_w, text_h), _baseline = cv2.getTextSize("5225m", cv2.FONT_HERSHEY_SIMPLEX, 1.15, 2)
    text_rect = (text_org[0], text_org[1] - text_h, text_w, text_h)
    cv2.putText(
        image,
        "08:41",
        (250, 245),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (250, 240, 200),
        2,
        cv2.LINE_AA,
    )
    center = (
        int(round(width * MinimapScanner.PLAYER_CENTER_REL[0])),
        int(round(height * MinimapScanner.PLAYER_CENTER_REL[1])),
    )
    cv2.circle(image, center, 5, (250, 250, 250), -1)
    _paste_sprout(image, center[0] + 18, center[1] - 8)
    _paste_sprout(image, text_org[0] + 8, text_org[1] - text_h)
    _paste_sprout(image, 8, 4)
    _paste_sprout(image, 270, 220)
    far_x, far_y = 36, 188
    _paste_sprout(image, far_x, far_y)

    materials, monsters = classify_minimap_crop_bgr(image)
    assert monsters == [] or all(not minimap_material_is_recordable(m) for m in monsters)
    # Player-adjacent harvest node (confidence 1.0 >= 0.72) is kept, far sprout is kept = 2 materials
    # Title bar, distance text, and clock chrome sprouts are rejected
    assert len(materials) == 2, materials
    far_mats = [m for m in materials if abs(m["norm_x"] * width - (far_x + 8)) < 14]
    adj_mats = [m for m in materials if abs(m["norm_x"] * width - (center[0] + 18 + 8)) < 14]
    assert len(far_mats) == 1, far_mats
    assert len(adj_mats) == 1, adj_mats

    # Low-confidence (< 0.72) node inside player clearance must be dropped
    low_conf_mat = {
        "norm_x": 0.51,
        "norm_y": 0.55,
        "confidence": 0.50,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert not minimap_material_is_recordable(low_conf_mat)

    # High-confidence (>= 0.72) node inside player clearance must be kept
    high_conf_mat = {
        "norm_x": 0.51,
        "norm_y": 0.55,
        "confidence": 0.75,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
    }
    assert minimap_material_is_recordable(high_conf_mat)

    mat = far_mats[0]
    assert minimap_material_is_recordable(mat)
    assert mat["kind"] == "sprout_material"
    px = mat["norm_x"] * width
    py = mat["norm_y"] * height
    assert abs(px - (far_x + 8)) < 14 and abs(py - (far_y + 8)) < 14, mat
    tx, ty, tw, th = text_rect
    assert not (tx <= px <= tx + tw and ty <= py <= ty + th)
    assert mat["norm_y"] >= 0.18
    assert not (mat["norm_x"] >= 0.62 and mat["norm_y"] >= 0.74)
    assert math.hypot(mat["norm_x"] - 0.502, mat["norm_y"] - 0.545) > 0.075
    distance_box = find_zone_distance_bbox(image)
    assert distance_box is not None
    dx, dy, dw, dh = distance_box
    assert not (dx <= px <= dx + dw and dy <= py <= dy + dh)
    stable_x = width * MinimapScanner.PLAYER_CENTER_REL[0]
    stable_y = height * MinimapScanner.PLAYER_CENTER_REL[1]
    assert abs(mat["rel_map_dx"] - (px - stable_x)) < 2.0
    assert abs(mat["rel_map_dy"] - (py - stable_y)) < 2.0


def test_dredgion_crop_counts_a_real_sprout() -> None:
    """frame_00s itself has no sprout pixels; a filiz pasted on that crop must count."""
    crop = _crop_frame("frame_00s.jpg")
    bare_mats, _bare_mobs = classify_minimap_crop_bgr(crop)
    assert bare_mats == [], bare_mats
    distance = find_zone_distance_bbox(crop)
    assert distance is not None
    pasted = crop.copy()
    origin = (24, 100)
    _paste_sprout(pasted, origin[0], origin[1])
    materials, monsters = classify_minimap_crop_bgr(pasted)
    assert len(materials) == 1, materials
    mat = materials[0]
    assert minimap_material_is_recordable(mat)
    assert mat["stack_badge"] == "3x"
    assert mat["kind"] == "sprout_material"
    height, width = pasted.shape[:2]
    px = mat["norm_x"] * width
    py = mat["norm_y"] * height
    assert abs(px - (origin[0] + 8)) < 14 and abs(py - (origin[1] + 8)) < 14, mat
    dx, dy, dw, dh = distance
    assert not (dx <= px <= dx + dw and dy <= py <= dy + dh)
    assert mat["norm_y"] >= 0.18
    assert not (mat["norm_x"] >= 0.62 and mat["norm_y"] >= 0.74)
    assert math.hypot(mat["norm_x"] - 0.502, mat["norm_y"] - 0.545) > 0.23
    assert "box_w" in mat and mat["box_w"] > 0 and mat["box_h"] > 0
    box_cx = (mat["box_x"] + mat["box_w"] * 0.5) * width
    box_cy = (mat["box_y"] + mat["box_h"] * 0.5) * height
    assert abs(box_cx - px) < 4 and abs(box_cy - py) < 4
    assert all(not minimap_material_is_recordable(m) for m in monsters)


def test_save_skips_monsters_chrome_and_placeholders() -> None:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from ui.cartography_radar import CartographyRadarDialog

    dialog = CartographyRadarDialog(route_manager=None)
    app.processEvents()
    live = dialog.minimap_preview_widget.get_gatherable_materials()
    assert live == [], live
    assert "AÇIK" in dialog.btn_toggle_minimap_inset.text()
    assert dialog.canvas._show_minimap_overlay is True
    _assert_no_bot_rings_or_center_dot(dialog.minimap_preview_widget)
    good = {
        "norm_x": 0.75,
        "norm_y": 0.70,
        "rel_map_dx": 40.0,
        "rel_map_dy": -12.0,
        "resource_type": "Toplanabilir Materyal",
        "stack_count": 3,
        "max_harvests": 3,
        "kind": "sprout_material",
        "is_gatherable": True,
        "hud_chrome": False,
        "source_type": "minimap_auto",
    }
    if dialog.canvas._player_pos is not None:
        base_x = float(dialog.canvas._player_pos[0])
    elif dialog._walk_trajectory:
        base_x = float(dialog._walk_trajectory[-1][0])
    else:
        base_x = float(dialog.canvas._base_pixmap.width()) * 0.5
    dialog.minimap_preview_widget._gatherable_materials = [
        good,
        {
            "norm_x": 0.40,
            "norm_y": 0.55,
            "kind": "monster_dot",
            "is_gatherable": False,
            "label": "•✕",
            "rel_map_dx": 10.0,
            "rel_map_dy": 8.0,
        },
        {
            "norm_x": 0.72,
            "norm_y": 0.26,
            "rel_map_dx": 36.0,
            "rel_map_dy": -32.0,
            "resource_type": "Aether Gem",
            "stack_count": 6,
            "stack_badge": "6x",
            "source_type": "minimap_auto",
        },
        {
            "norm_x": 0.50,
            "norm_y": 0.08,
            "rel_map_dx": 0.0,
            "rel_map_dy": -80.0,
            "kind": "sprout_material",
            "is_gatherable": True,
            "resource_type": "Toplanabilir Materyal",
        },
        {
            "norm_x": 0.86,
            "norm_y": 0.90,
            "rel_map_dx": 70.0,
            "rel_map_dy": 60.0,
            "kind": "sprout_material",
            "is_gatherable": True,
            "resource_type": "Toplanabilir Materyal",
        },
        {
            "norm_x": 0.50,
            "norm_y": 0.545,
            "rel_map_dx": 0.0,
            "rel_map_dy": 0.0,
            "kind": "sprout_material",
            "is_gatherable": True,
            "resource_type": "Toplanabilir Materyal",
        },
    ]
    saved = dialog.record_minimap_now()
    assert len(saved) == 1, saved
    assert saved[0]["resource_type"] == "Toplanabilir Materyal"
    assert saved[0].get("kind") == "sprout_material"
    assert abs(float(saved[0]["map_x"]) - (base_x + 40.0)) <= 1.0
    assert all(n.get("resource_type") != "Aether Gem" for n in dialog._ordered_nodes)
    assert all(n.get("kind") != "monster_dot" for n in dialog._ordered_nodes)

    dialog._minimap_inset_user_override = False
    dialog.minimap_preview_widget.set_minimap_detections([good], [])
    app.processEvents()
    assert dialog.canvas._show_minimap_overlay is True
    assert "AÇIK" in dialog.btn_toggle_minimap_inset.text()

    dialog._on_user_toggle_minimap_inset(False)
    dialog.minimap_preview_widget.set_minimap_detections([good], [])
    app.processEvents()
    assert dialog.canvas._show_minimap_overlay is False
    assert "KAPALI" in dialog.btn_toggle_minimap_inset.text()

    dialog._minimap_inset_user_override = False
    dialog.minimap_preview_widget.set_minimap_detections([], [])
    app.processEvents()
    assert dialog.canvas._show_minimap_overlay is True
    assert "AÇIK" in dialog.btn_toggle_minimap_inset.text()
    dialog.close()


def _assert_no_bot_rings_or_center_dot(widget) -> None:
    from PyQt6.QtGui import QColor, QImage, QPainter
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    widget.resize(240, 400)
    widget.show()
    app.processEvents()
    image = QImage(widget.size(), QImage.Format.Format_RGB888)
    image.fill(QColor("#10141a"))
    painter = QPainter(image)
    widget.render(painter)
    painter.end()
    width, height = image.width(), image.height()
    header_h, footer_h, inner = 20.0, 22.0, 6.0
    crop_w = width - inner * 2.0
    crop_h = height - header_h - footer_h - 10.0
    cx = int(inner + crop_w * 0.5)
    cy = int(header_h + 4.0 + crop_h * 0.5)
    yellow = 0
    for dy in range(-8, 9):
        for dx in range(-8, 9):
            color = image.pixelColor(cx + dx, cy + dy)
            if abs(color.red() - 250) < 28 and abs(color.green() - 204) < 28 and color.blue() < 70:
                yellow += 1
    assert yellow < 6, yellow
    ring = 0
    radius = int(min(crop_w, crop_h) * 0.44)
    for deg in range(0, 360, 6):
        px = int(cx + radius * math.cos(math.radians(deg)))
        py = int(cy + radius * math.sin(math.radians(deg)))
        if 0 <= px < width and 0 <= py < height:
            color = image.pixelColor(px, py)
            if abs(color.red() - 56) < 22 and abs(color.green() - 189) < 22 and abs(color.blue() - 248) < 22:
                ring += 1
    assert ring < 4, ring


def main() -> None:
    test_crop_box_ignores_distance_text()
    test_dredgion_frame_has_no_placeholder_materials()
    test_player_chevron_is_not_a_material()
    test_reference_sprout_counts_and_monsters_are_not_saved()
    test_synthetic_chrome_is_rejected_and_sprout_counts()
    test_dredgion_crop_counts_a_real_sprout()
    test_save_skips_monsters_chrome_and_placeholders()
    print("minimap hybrid panel tests passed")


if __name__ == "__main__":
    main()
