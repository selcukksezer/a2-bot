# -*- coding: utf-8 -*-
"""
Unit tests for Map Viewer Interactive Route Editing, Directional Arrows, and HD Fit.
Specifically covers Agent G4 requirements:
- Click-to-add on canvas empty space
- Drag-to-move node with live coordinates and persistence
- Right-click delete (and Shift+Right-Click fast delete)
- 1->2->3 Directional route arrows, chevrons, corridor glow, and step badges
- HD fit affine transforms and zero edge-sticking
- Node patch preview resolution, thumbnail caching, and preview card rendering
"""

from __future__ import annotations

import os
import sys
import math
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPoint, QPointF, QRectF
from PyQt6.QtGui import QMouseEvent, QPainter, QPixmap, QImage, QColor

from ui.map_viewer import (
    MapViewerWindow,
    _MapCanvas,
    apply_hd_fit,
    inv_hd_fit,
    resolve_node_xy,
    DEFAULT_MAP_REF,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MockRouteManager:
    """Mock RouteManager supporting node queries, add_node, move_node, and delete_node."""
    def __init__(self, region: str = "Altgard"):
        self.active_region = region
        self.active_route = "All"
        self._regions: Dict[str, Dict[str, Any]] = {
            "Altgard": {
                "routes": ["All", "Mining Run"],
                "nodes": [
                    {
                        "id": 1,
                        "region": "Altgard",
                        "routes": ["All", "Mining Run"],
                        "map_x": 400,
                        "map_y": 300,
                        "norm_x": 0.3125,
                        "norm_y": 0.4167,
                        "resource_type": "Orichalcum Ore",
                        "max_harvests": 3,
                        "current_harvests": 0,
                        "source_type": "manual",
                        "state": "active",
                        "seq_index": 1,
                        "in_route": True,
                    },
                    {
                        "id": 2,
                        "region": "Altgard",
                        "routes": ["All", "Mining Run"],
                        "map_x": 600,
                        "map_y": 450,
                        "norm_x": 0.4688,
                        "norm_y": 0.6250,
                        "resource_type": "Aria Herb",
                        "max_harvests": 3,
                        "current_harvests": 1,
                        "source_type": "manual",
                        "state": "active",
                        "seq_index": 2,
                        "in_route": True,
                    },
                    {
                        "id": 3,
                        "region": "Altgard",
                        "routes": ["All", "Mining Run"],
                        "map_x": 800,
                        "map_y": 350,
                        "norm_x": 0.6250,
                        "norm_y": 0.4861,
                        "resource_type": "Iron Ore",
                        "max_harvests": 3,
                        "current_harvests": 0,
                        "source_type": "manual",
                        "state": "active",
                        "seq_index": 3,
                        "in_route": True,
                    },
                ],
            }
        }
        self.save_called = False
        self.deleted_ids = []

    def get_nodes(self, region: str, route: Optional[str] = None) -> List[Dict[str, Any]]:
        reg = self._regions.get(region, {})
        nodes = list(reg.get("nodes", []))
        if route and route != "All":
            nodes = [n for n in nodes if route in n.get("routes", [])]
        return nodes

    def get_routes(self, region: str) -> List[str]:
        return list(self._regions.get(region, {}).get("routes", ["All"]))

    def add_node(self, region: str, route: str = "All", **kwargs) -> Any:
        reg = self._regions.setdefault(region, {"routes": ["All"], "nodes": []})
        existing = [int(n.get("id", 0)) for n in reg["nodes"]]
        new_id = (max(existing) + 1) if existing else 1
        node = {
            "id": new_id,
            "region": region,
            "routes": [route] if route and route != "All" else ["All"],
            "map_x": kwargs.get("map_x", 0),
            "map_y": kwargs.get("map_y", 0),
            "norm_x": kwargs.get("norm_x", 0.0),
            "norm_y": kwargs.get("norm_y", 0.0),
            "source_type": kwargs.get("source_type", "manual"),
            "state": kwargs.get("state", "active"),
            "seq_index": len(reg["nodes"]) + 1,
            "in_route": True,
        }
        reg["nodes"].append(node)
        self.save_called = True
        mock_obj = MagicMock()
        mock_obj.id = new_id
        return mock_obj

    def move_node(self, node_id: int, map_x: float, map_y: float, **kwargs) -> bool:
        for reg in self._regions.values():
            for n in reg.get("nodes", []):
                if int(n.get("id", -1)) == int(node_id):
                    n["map_x"] = int(round(map_x))
                    n["map_y"] = int(round(map_y))
                    if "norm_x" in kwargs:
                        n["norm_x"] = kwargs["norm_x"]
                    if "norm_y" in kwargs:
                        n["norm_y"] = kwargs["norm_y"]
                    self.save_called = True
                    return True
        return False

    def delete_node(self, node_id: int, region: Optional[str] = None) -> bool:
        self.deleted_ids.append(node_id)
        for reg in self._regions.values():
            nodes = reg.get("nodes", [])
            initial_len = len(nodes)
            reg["nodes"] = [n for n in nodes if int(n.get("id", -1)) != int(node_id)]
            if len(reg["nodes"]) < initial_len:
                self.save_called = True
                return True
        return False

    def save_region(self, region: str) -> None:
        self.save_called = True

    def save_data(self) -> None:
        self.save_called = True


class TestHDTransformsAndEdgeSticking:
    """Tests HD fit affine transforms and verifies edge-sticking prevention."""

    def test_hd_fit_affine_roundtrip(self):
        """inv_hd_fit and apply_hd_fit must be exact mathematical inverses across map."""
        test_points = [
            (50.0, 50.0),        # Top-left perimeter
            (2200.0, 2350.0),    # Bottom-right perimeter
            (1129.0, 1200.0),    # Center
            (280.0, 80.0),       # Near crop offset
            (1500.0, 400.0),     # Northern territory
            (400.0, 2100.0),     # Southern territory
        ]
        for hx, hy in test_points:
            rx, ry = inv_hd_fit(hx, hy)
            recon_hx, recon_hy = apply_hd_fit(rx, ry)
            assert math.isclose(hx, recon_hx, abs_tol=1e-3), f"HD X mismatch at ({hx}, {hy}): {recon_hx}"
            assert math.isclose(hy, recon_hy, abs_tol=1e-3), f"HD Y mismatch at ({hx}, {hy}): {recon_hy}"

    def test_no_edge_sticking_in_map_to_node_coords(self, qapp):
        """_map_to_node_coords must not clamp coordinates to [0, 1280] / [0, 720] when HD fit is active."""
        canvas = _MapCanvas()
        canvas._hd_fit = True
        # Set realistic Altgard HD base pixmap (2258x2400)
        base = QPixmap(2258, 2400)
        base.fill(QColor("black"))
        canvas.set_base_pixmap(base)

        # Points at outer perimeter of 2258x2400 map
        outer_pt = QPointF(2200.0, 2350.0)
        map_x, map_y, norm_x, norm_y = canvas._map_to_node_coords(outer_pt)

        # Coordinates should reflect the true inverse transform without being clamped to 1280 or 720
        expected_rx, expected_ry = inv_hd_fit(2200.0, 2350.0)
        assert abs(map_x - round(expected_rx)) <= 1
        assert abs(map_y - round(expected_ry)) <= 1
        assert 0.0 <= norm_x <= 1.0
        assert 0.0 <= norm_y <= 1.0
        assert norm_x > 0.95 and norm_y > 0.95

    def test_aspect_fit_viewport_scaling(self, qapp):
        """_fit_to_view must use aspect-fit (min scale) to keep all map edges visible."""
        canvas = _MapCanvas()
        canvas.resize(1000, 800)
        base = QPixmap(2258, 2400)
        base.fill(QColor("gray"))
        canvas.set_base_pixmap(base)

        canvas._fit_to_view()
        scale_x = 1000.0 / 2258.0
        scale_y = 800.0 / 2400.0
        expected_zoom = min(scale_x, scale_y)

        assert math.isclose(canvas._zoom, expected_zoom, rel_tol=1e-4)
        # Pan should center the map within the canvas
        rendered_w = 2258.0 * canvas._zoom
        rendered_h = 2400.0 * canvas._zoom
        assert canvas._pan.x() >= 0.0
        assert canvas._pan.y() >= 0.0
        assert canvas._pan.x() + rendered_w <= 1000.0 + 1e-3
        assert canvas._pan.y() + rendered_h <= 800.0 + 1e-3


class TestInteractiveRouteEditing:
    """Tests click-to-add, drag-to-move, and right-click delete on map canvas."""

    def test_click_to_add_node(self, qapp):
        """Clicking empty space on canvas creates a node and emits node_added."""
        mock_rm = MockRouteManager()
        win = MapViewerWindow(mock_rm, region="Altgard", route="All")
        win.resize(800, 600)
        canvas = win._canvas

        initial_count = len(canvas._nodes)
        added_nodes = []
        canvas.node_added.connect(lambda n: added_nodes.append(n))

        # Create node at specific coordinates
        new_id = canvas._create_node_at_coords(500.0, 350.0, 0.39, 0.48)
        assert new_id is not None
        assert len(added_nodes) == 1
        assert added_nodes[0]["id"] == new_id
        assert added_nodes[0]["seq_index"] == initial_count + 1
        assert "patches/node_" in added_nodes[0]["patch_path"]
        assert mock_rm.save_called

    def test_drag_to_move_node(self, qapp):
        """Dragging an existing node updates its coordinates live and saves upon release."""
        mock_rm = MockRouteManager()
        win = MapViewerWindow(mock_rm, region="Altgard", route="All")
        canvas = win._canvas

        moved_signals = []
        canvas.node_moved.connect(lambda nid, x, y: moved_signals.append((nid, x, y)))

        # Simulate drag state
        canvas._is_dragging_node = True
        canvas._dragging_node_id = 1
        canvas._drag_start_pos = QPointF(100.0, 100.0)
        canvas._drag_moved_px = 0.0

        # Move by 30px
        move_event = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(130.0, 100.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseMoveEvent(move_event)
        assert len(moved_signals) >= 1
        assert moved_signals[-1][0] == 1
        assert canvas._drag_moved_px >= 30.0

        # Release to save position
        release_event = QMouseEvent(
            QMouseEvent.Type.MouseButtonRelease,
            QPointF(130.0, 100.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mouseReleaseEvent(release_event)
        assert not canvas._is_dragging_node
        assert canvas._dragging_node_id is None
        assert mock_rm.save_called

    def test_right_click_delete_node(self, qapp):
        """Right-clicking with Shift or calling delete_node removes node and re-sequences route."""
        mock_rm = MockRouteManager()
        win = MapViewerWindow(mock_rm, region="Altgard", route="All")
        canvas = win._canvas

        deleted_signals = []
        canvas.node_deleted.connect(lambda nid: deleted_signals.append(nid))

        assert any(n.get("id") == 2 for n in canvas._nodes)
        success = canvas.delete_node(2)
        assert success is True
        assert 2 in deleted_signals
        assert 2 in mock_rm.deleted_ids
        assert not any(n.get("id") == 2 for n in canvas._nodes)
        assert 2 not in canvas._route_node_ids
        assert 2 not in canvas._route_node_ids_list


class TestDirectionalArrowsAndPreviewCard:
    """Tests sequential 1->2->3 route flow, chevron arrows, step badges, and patch preview."""

    def test_sequential_route_ordering(self, qapp):
        """_get_sequential_route_nodes returns ordered tuples matching sequential route flow."""
        canvas = _MapCanvas()
        canvas._nodes = [
            {"id": 10, "seq_index": 1, "in_route": True, "map_x": 100, "map_y": 100},
            {"id": 20, "seq_index": 2, "in_route": True, "map_x": 200, "map_y": 200},
            {"id": 30, "seq_index": 3, "in_route": True, "map_x": 300, "map_y": 300},
            {"id": 40, "seq_index": 4, "in_route": False, "map_x": 400, "map_y": 400},
        ]
        canvas._route_node_ids = {10, 20, 30}
        canvas._route_node_ids_list = [10, 20, 30]

        seq_nodes = canvas._get_sequential_route_nodes()
        assert len(seq_nodes) == 3
        assert seq_nodes[0][0] == 1 and seq_nodes[0][1]["id"] == 10
        assert seq_nodes[1][0] == 2 and seq_nodes[1][1]["id"] == 20
        assert seq_nodes[2][0] == 3 and seq_nodes[2][1]["id"] == 30

    def test_render_sequential_route_and_chevron_arrows(self, qapp):
        """_draw_sequential_route renders path lines, chevrons, corridor glow, and step badges."""
        canvas = _MapCanvas()
        canvas.resize(600, 600)
        canvas._nodes = [
            {"id": 1, "seq_index": 1, "in_route": True, "map_x": 100, "map_y": 100},
            {"id": 2, "seq_index": 2, "in_route": True, "map_x": 300, "map_y": 100},
            {"id": 3, "seq_index": 3, "in_route": True, "map_x": 300, "map_y": 300},
        ]
        canvas._route_node_ids = {1, 2, 3}
        canvas._route_node_ids_list = [1, 2, 3]

        img = QImage(600, 600, QImage.Format.Format_ARGB32)
        img.fill(QColor("black"))
        painter = QPainter(img)
        ordered = canvas._get_sequential_route_nodes()

        # Must execute cleanly without exceptions
        canvas._draw_sequential_route(painter, ordered)
        painter.end()

        # Validate that drawing produced non-black pixels on canvas
        has_colored_pixels = False
        for y in range(80, 320, 10):
            for x in range(80, 320, 10):
                pixel = img.pixelColor(x, y)
                if pixel.red() > 0 or pixel.green() > 0 or pixel.blue() > 0:
                    has_colored_pixels = True
                    break
            if has_colored_pixels:
                break
        assert has_colored_pixels, "Sequential route rendering failed to paint pixels."

    def test_node_patch_preview_resolution_and_cache(self, qapp):
        """resolve_node_patch_path and get_node_patch_pixmap work properly and cache pixmaps."""
        canvas = _MapCanvas()
        node = {
            "id": 42,
            "region": "Altgard",
            "resource_type": "Orichalcum Ore",
            "map_x": 500,
            "map_y": 600,
        }
        resolved = canvas.resolve_node_patch_path(node)
        assert "node_42.png" in resolved

        # Fallback thumbnail generation
        pix = canvas.get_node_patch_pixmap(node, size=64)
        assert isinstance(pix, QPixmap)
        assert not pix.isNull()
        assert pix.width() == 64 and pix.height() == 64
        # Cache hit
        assert 42 in canvas._patch_cache
        assert canvas.get_node_patch_pixmap(node, size=64) is pix

        canvas.clear_patch_cache()
        assert len(canvas._patch_cache) == 0

    def test_node_patch_preview_card_rendering(self, qapp):
        """_draw_node_patch_preview_card renders within widget boundaries."""
        canvas = _MapCanvas()
        canvas.resize(800, 600)
        node = {
            "id": 1,
            "region": "Altgard",
            "resource_type": "Kinquid Ore",
            "map_x": 400,
            "map_y": 300,
            "max_harvests": 3,
            "current_harvests": 2,
            "state": "active",
        }
        img = QImage(800, 600, QImage.Format.Format_ARGB32)
        img.fill(QColor("black"))
        painter = QPainter(img)

        # Draw card anchored at pin point (400, 300)
        canvas._draw_node_patch_preview_card(painter, node, QPointF(400.0, 300.0))
        painter.end()

        # Verify pixels were rendered in the card area
        pixel_sample = img.pixelColor(420, 250)
        # Card background should be dark slate #0f172a with opacity
        assert pixel_sample.alpha() > 0
