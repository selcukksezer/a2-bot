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


def _real_sprout_patch(size: int) -> np.ndarray:
    """A sprout patch cut from real game pixels — never the production template.

    Loads ``reference_screenshots/minimap_sprout_closeup.jpg``, locates a
    sprout with the scanner, and cuts a 16x16 window centred on the detected
    node. The returned pixels are the game's own sprout glyph: the helper
    asserts they differ from ``MinimapScanner._CANONICAL_SPROUT_TPL`` so a
    test using them cannot pass by matching the production template (F-1421).
    """
    root = os.path.dirname(__file__)
    shot = _load(os.path.join(root, "reference_screenshots", "minimap_sprout_closeup.jpg"))
    scanner = MinimapScanner()
    shot_crop, _bounds = scanner.extract_minimap(shot)
    nodes = scanner.scan_minimap_crop(shot_crop)["gatherable_nodes"]
    assert len(nodes) >= 1, nodes
    cx, cy = nodes[0]["pos"]
    patch = shot_crop[cy - 8 : cy + 8, cx - 8 : cx + 8].copy()
    assert patch.shape == (16, 16, 3), patch.shape
    tpl = cv2.cvtColor(MinimapScanner._CANONICAL_SPROUT_TPL, cv2.COLOR_GRAY2BGR)
    assert float(np.mean(np.abs(patch.astype(np.int16) - tpl.astype(np.int16)))) > 1.0, (
        "extracted patch is (near-)identical to the production template; "
        "it would not prove real pixels are counted"
    )
    if size == 16:
        return patch
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_AREA)


def test_crop_box_ignores_distance_text() -> None:
    # crop_box is pure geometry: the same (w, h) must give the same rect no
    # matter what text the frame carries. The old test only called it on one
    # marked image and never showed the text was really there (F-1330).
    plain = MinimapLiveCropPreviewWidget.crop_box(1280, 720)
    variants = []
    for text, org, scale, thick in (
        ("5225m", (1100, 80), 1.4, 3),
        ("12m", (50, 700), 1.0, 2),
        ("99999m", (1100, 80), 2.0, 4),
        ("", (0, 0), 1.0, 1),
    ):
        marked = np.zeros((720, 1280, 3), np.uint8)
        if text:
            cv2.putText(
                marked,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                (255, 255, 255),
                thick,
                cv2.LINE_AA,
            )
            # Prove the text pixels are really in this image: the crop is
            # supposed to ignore them, not to never have seen them.
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
            region = marked[org[1] - th : org[1] + 2, org[0] : org[0] + tw]
            assert int(np.count_nonzero(region)) > 0, text
        variants.append(marked)
        assert MinimapLiveCropPreviewWidget.crop_box(marked.shape[1], marked.shape[0]) == plain, text
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
    # The 166px-wide crop is half the scanner's 332px reference width, so the
    # sprout must be pasted at the scanner's expected kernel size (9px), and
    # it must be real game pixels from the reference screenshot — the old
    # test pasted the production template itself at 16px and counted nothing
    # (F-1421; it failed at baseline with 0 materials).
    patch_size = 9
    patch = _real_sprout_patch(patch_size)
    pasted[origin[1] : origin[1] + patch_size, origin[0] : origin[0] + patch_size] = patch
    materials, monsters = classify_minimap_crop_bgr(pasted)
    assert len(materials) == 1, materials
    mat = materials[0]
    assert minimap_material_is_recordable(mat)
    assert mat["stack_badge"] == "3x"
    assert mat["kind"] == "sprout_material"
    height, width = pasted.shape[:2]
    px = mat["norm_x"] * width
    py = mat["norm_y"] * height
    center_x, center_y = origin[0] + patch_size // 2, origin[1] + patch_size // 2
    assert abs(px - center_x) < 14 and abs(py - center_y) < 14, mat
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
    # Feed the record path through the real classifier instead of assigning
    # a hand-made list to the widget's private state (F-1422): a synthetic
    # minimap with a REAL sprout (reference-screenshot pixels) at rel dx +40
    # and a hostile red dot. The classifier must produce exactly one
    # recordable material and one rejected monster.
    height, width = 257, 332
    scene = np.full((height, width, 3), (42, 58, 70), np.uint8)
    pcx = int(round(width * MinimapScanner.PLAYER_CENTER_REL[0]))
    pcy = int(round(height * MinimapScanner.PLAYER_CENTER_REL[1]))
    cv2.circle(scene, (pcx, pcy), 5, (250, 250, 250), -1)
    scene[pcy - 8 : pcy + 8, pcx + 32 : pcx + 48] = _real_sprout_patch(16)
    cv2.circle(scene, (pcx - 60, pcy + 20), 4, (20, 20, 240), -1)
    materials, monsters = classify_minimap_crop_bgr(scene)
    assert len(materials) == 1, materials
    assert materials[0]["kind"] == "sprout_material"
    assert abs(materials[0]["rel_map_dx"] - 40.0) < 0.5, materials
    assert len(monsters) == 1, monsters
    assert monsters[0]["kind"] == "hostile_dot", monsters
    assert monsters[0]["is_gatherable"] is False

    dialog.minimap_preview_widget.set_minimap_detections(materials, monsters)
    live = dialog.minimap_preview_widget.get_gatherable_materials()
    assert len(live) == 1, live
    assert live[0]["kind"] == "sprout_material"
    rejected = dialog.minimap_preview_widget.get_rejected_monsters()
    assert len(rejected) == 1, rejected
    assert rejected[0]["kind"] == "hostile_dot"
    good = live[0]
    if dialog.canvas._player_pos is not None:
        base_x = float(dialog.canvas._player_pos[0])
    elif dialog._walk_trajectory:
        base_x = float(dialog._walk_trajectory[-1][0])
    else:
        base_x = float(dialog.canvas._base_pixmap.width()) * 0.5
    saved = dialog.record_minimap_now()
    assert len(saved) == 1, saved
    assert saved[0]["resource_type"] == "Toplanabilir Materyal"
    assert saved[0].get("kind") == "sprout_material"
    assert abs(float(saved[0]["map_x"]) - (base_x + 40.0)) <= 1.0
    # Only the real sprout was recorded: the hostile dot never became a node.
    assert all(n.get("kind") != "hostile_dot" for n in dialog._ordered_nodes)
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
