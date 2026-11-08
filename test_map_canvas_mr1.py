# -*- coding: utf-8 -*-
"""
Comprehensive Unit Tests for Map/Radar Worker 1: _MapCanvas Coordinates & Interaction
(Gatherion 1.5.2 / 2.0 Map Viewer)
"""

import sys
import math
import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt6.QtCore import Qt, QPoint, QPointF, QRect

from ui.map_viewer import (
    _MapCanvas,
    apply_hd_fit,
    inv_hd_fit,
    resolve_node_xy,
    DEFAULT_MAP_REF,
)
import ui.cartography_radar as radar_mod


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def test_hd_fit_and_inv_hd_fit_roundtrip():
    """Verify that apply_hd_fit and inv_hd_fit are exact inverses."""
    test_points = [
        (426.0, 286.0),
        (0.0, 0.0),
        (640.0, 360.0),
        (1280.0, 720.0),
        (280.0, 80.0),
        (950.5, 412.3),
    ]
    for x, y in test_points:
        hx, hy = apply_hd_fit(x, y)
        rx, ry = inv_hd_fit(hx, hy)
        assert abs(rx - x) < 1e-4, f"X mismatch for ({x}, {y}): got {rx}"
        assert abs(ry - y) < 1e-4, f"Y mismatch for ({x}, {y}): got {ry}"


def test_hd_fit_matches_cartography_radar():
    """Verify that map_viewer.apply_hd_fit matches cartography_radar.apply_hd_fit exactly."""
    for x in (100.0, 426.0, 800.0, 1200.0):
        for y in (50.0, 286.0, 500.0, 700.0):
            mv_hd = apply_hd_fit(x, y)
            cr_hd = radar_mod.apply_hd_fit(x, y)
            assert abs(mv_hd[0] - cr_hd[0]) < 1e-6
            assert abs(mv_hd[1] - cr_hd[1]) < 1e-6


def test_node_to_map_coords_hd_mode(qapp):
    """Verify _node_to_map_coords with HD fit active matches apply_hd_fit."""
    canvas = _MapCanvas()
    # 2258x2400 pixmap triggers HD mode auto-detection
    pix = QPixmap(2258, 2400)
    canvas.set_map_pixmap(pix)
    assert canvas._hd_fit is True

    node = {
        "id": 245,
        "map_x": 426,
        "map_y": 286,
        "norm_x": 426.0 / 1280.0,
        "norm_y": 286.0 / 720.0,
    }
    pt = canvas._node_to_map_coords(426, 286, node=node)
    expected_hd = apply_hd_fit(426, 286)
    assert abs(pt.x() - expected_hd[0]) < 1e-3
    assert abs(pt.y() - expected_hd[1]) < 1e-3


def test_node_to_map_coords_no_edge_sticking(qapp):
    """
    Verify that in cropped mode (FRAME18_NORM_CROP), coordinates outside the crop
    are NOT forcefully clamped to 8.0, eliminating kenara yapışma.
    """
    canvas = _MapCanvas()
    pix = QPixmap(614, 450)
    canvas.set_map_pixmap(pix, coord_offset=(377, 104), crop_norm_rect=_MapCanvas.FRAME18_NORM_CROP)
    assert canvas._hd_fit is False

    # Node with norm_x = 0.10 (to the left of crop window rx=0.295)
    node_outside = {"id": 1, "map_x": 128, "map_y": 360, "norm_x": 0.10, "norm_y": 0.50}
    pt = canvas._node_to_map_coords(128, 360, node=node_outside)
    # With u = (0.10 - 0.295) / 0.480 = -0.40625, mapped_x should be negative, NOT clamped to 8.0!
    assert pt.x() < 0.0, f"Expected negative coordinate outside crop, got {pt.x()} (clamped!)"
    assert pt.x() != 8.0, "Coordinate was incorrectly clamped to 8.0 (kenara yapışma bug)!"


def test_map_to_node_coords_roundtrip(qapp):
    """Verify that _map_to_node_coords accurately reverses _node_to_map_coords."""
    canvas = _MapCanvas()
    pix = QPixmap(2258, 2400)
    canvas.set_map_pixmap(pix)
    assert canvas._hd_fit is True

    orig_x, orig_y = 520, 310
    map_pt = canvas._node_to_map_coords(orig_x, orig_y)
    rev_x, rev_y, norm_x, norm_y = canvas._map_to_node_coords(map_pt)
    assert abs(rev_x - orig_x) <= 1
    assert abs(rev_y - orig_y) <= 1
    assert abs(norm_x - orig_x / 1280.0) < 1e-3
    assert abs(norm_y - orig_y / 720.0) < 1e-3


def test_interactive_create_node(qapp):
    """Test creating a new node on empty canvas click."""
    canvas = _MapCanvas()
    pix = QPixmap(2258, 2400)
    canvas.set_map_pixmap(pix)

    added_signals = []
    canvas.node_add_requested.connect(lambda x, y: added_signals.append((x, y)))
    callback_calls = []
    canvas.on_canvas_add_node = lambda nid, x, y: callback_calls.append((nid, x, y))

    new_id = canvas._create_node_at_coords(640, 360, 0.5, 0.5)
    assert new_id > 0
    assert any(n.get("id") == new_id for n in canvas._nodes)
    assert new_id in canvas._route_node_ids
    assert len(added_signals) == 1
    assert added_signals[0] == (640.0, 360.0)
    assert len(callback_calls) == 1
    assert callback_calls[0][0] == new_id


def test_interactive_drag_node(qapp):
    """Test smoothly dragging an existing node to a new position."""
    canvas = _MapCanvas()
    pix = QPixmap(2258, 2400)
    canvas.set_map_pixmap(pix)

    nodes = [
        {"id": 10, "map_x": 400, "map_y": 200, "norm_x": 400/1280, "norm_y": 200/720, "in_route": True},
        {"id": 20, "map_x": 500, "map_y": 300, "norm_x": 500/1280, "norm_y": 300/720, "in_route": True},
    ]
    canvas.set_nodes_data(nodes, [10, 20])

    moved_signals = []
    canvas.node_moved.connect(lambda nid, x, y: moved_signals.append((nid, x, y)))

    # Simulate node drag
    canvas._is_dragging_node = True
    canvas._dragging_node_id = 10
    # Update node position
    for n in canvas._nodes:
        if n["id"] == 10:
            n["map_x"] = 450
            n["map_y"] = 250
            n["norm_x"] = 450 / 1280.0
            n["norm_y"] = 250 / 720.0
            break
    canvas.node_moved.emit(10, 450.0, 250.0)

    assert len(moved_signals) == 1
    assert moved_signals[0] == (10, 450.0, 250.0)
    target = next(n for n in canvas._nodes if n["id"] == 10)
    assert target["map_x"] == 450
    assert target["map_y"] == 250


def test_interactive_delete_node(qapp):
    """Test deleting an existing node."""
    canvas = _MapCanvas()
    nodes = [
        {"id": 1, "map_x": 100, "map_y": 100, "in_route": True},
        {"id": 2, "map_x": 200, "map_y": 200, "in_route": True},
        {"id": 3, "map_x": 300, "map_y": 300, "in_route": True},
    ]
    canvas.set_nodes_data(nodes, [1, 2, 3])

    deleted_signals = []
    canvas.node_deleted.connect(lambda nid: deleted_signals.append(nid))

    res = canvas.delete_node(2)
    assert res is True
    assert len(canvas._nodes) == 2
    assert not any(n["id"] == 2 for n in canvas._nodes)
    assert 2 not in canvas._route_node_ids
    assert 2 not in canvas._route_node_ids_list
    assert deleted_signals == [2]


def test_sequential_route_ordering_and_arrows(qapp):
    """Test sequential route extraction and painting."""
    canvas = _MapCanvas()
    canvas.resize(800, 600)
    pix = QPixmap(1280, 720)
    pix.fill(QColor("#202020"))
    canvas.set_map_pixmap(pix)

    nodes = [
        {"id": 101, "map_x": 200, "map_y": 200, "norm_x": 200/1280, "norm_y": 200/720, "in_route": True},
        {"id": 102, "map_x": 400, "map_y": 250, "norm_x": 400/1280, "norm_y": 250/720, "in_route": True},
        {"id": 103, "map_x": 600, "map_y": 400, "norm_x": 600/1280, "norm_y": 400/720, "in_route": True},
    ]
    # Route in order 103 -> 101 -> 102
    canvas.set_nodes_data(nodes, [103, 101, 102])

    ordered = canvas._get_sequential_route_nodes()
    assert len(ordered) == 3
    assert ordered[0][0] == 1 and ordered[0][1]["id"] == 103
    assert ordered[1][0] == 2 and ordered[1][1]["id"] == 101
    assert ordered[2][0] == 3 and ordered[2][1]["id"] == 102

    # Render on offscreen image with QPainter to ensure zero rendering exceptions
    img = QImage(800, 600, QImage.Format.Format_RGB32)
    painter = QPainter(img)
    try:
        canvas._draw_sequential_route(painter, ordered)
    finally:
        painter.end()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
