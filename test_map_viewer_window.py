# -*- coding: utf-8 -*-
"""Comprehensive unit tests for MapViewerWindow (Map/Radar Worker 2)."""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from ui.map_viewer import MapViewerWindow, _MapCanvas
from core.route_manager import RouteManager, NodeData


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class MockRouteManager:
    """Mock RouteManager supporting authoritative _regions and atomic save_region."""
    def __init__(self, region: str = "Altgard"):
        self.active_region = region
        self.active_route = "All"
        self._regions: Dict[str, Dict[str, Any]] = {
            "Altgard": {
                "routes": ["All", "Dredgion Crash Site", "Safe Haven"],
                "nodes": [
                    {
                        "id": 1,
                        "region": "Altgard",
                        "routes": ["All", "Dredgion Crash Site"],
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
                    },
                    {
                        "id": 2,
                        "region": "Altgard",
                        "routes": ["All", "Safe Haven"],
                        "map_x": 450,
                        "map_y": 350,
                        "norm_x": 0.3515,
                        "norm_y": 0.4861,
                        "resource_type": "Orichalcum Ore",
                        "max_harvests": 3,
                        "current_harvests": 0,
                        "source_type": "manual",
                        "state": "active",
                        "seq_index": 2,
                    },
                ],
            }
        }
        self.saved_regions: List[str] = []

    def save_region(self, region: Optional[str] = None) -> bool:
        target = region or self.active_region
        self.saved_regions.append(target)
        return True


class MockAutoCartographer:
    """Mock AutoCartographer tracking live auto-mapped nodes and odometry."""
    def __init__(self, region: str = "Altgard"):
        self.region = region
        self.map_x = 425.0
        self.map_y = 320.0
        self._tracked_nodes = {
            "Altgard": [
                {
                    "id": 991,
                    "region": "Altgard",
                    "routes": ["All"],
                    "map_x": 420,
                    "map_y": 310,
                    "norm_x": 0.328,
                    "norm_y": 0.430,
                    "resource_type": "Orichalcum Ore",
                    "max_harvests": 3,
                    "current_harvests": 0,
                    "source_type": "minimap_auto",
                    "state": "active",
                }
            ]
        }

    def get_tracked_nodes(self, region: Optional[str] = None) -> List[Dict[str, Any]]:
        reg = region or self.region
        return list(self._tracked_nodes.get(reg, []))


# ---------------------------------------------------------------------------
# Test 1: Subregion Canonical Mapping
# ---------------------------------------------------------------------------
def test_subregion_canonical_mapping(qapp):
    """Subregions must resolve to parent canonical region 'Altgard'."""
    assert MapViewerWindow.canonical_region_name("Dredgion Crash Site") == "Altgard"
    assert MapViewerWindow.canonical_region_name("Safe Haven") == "Altgard"
    assert MapViewerWindow.canonical_region_name("Calderon Canyon") == "Altgard"
    assert MapViewerWindow.canonical_region_name("dredgion crash site") == "Altgard"
    assert MapViewerWindow.canonical_region_name("calderon") == "Altgard"
    assert MapViewerWindow.canonical_region_name("Safe Haven") == "Altgard"
    assert MapViewerWindow.canonical_region_name("Altgard") == "Altgard"
    assert MapViewerWindow.canonical_region_name(None) == "Altgard"
    assert MapViewerWindow.canonical_region_name("") == "Altgard"
    assert MapViewerWindow.canonical_region_name("Morheim") == "Morheim"


def test_subregion_set_region_integration(qapp):
    """Calling set_region with a subregion sets canonical region Altgard and selects route."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm, region_name="Altgard")
    
    # Subregion: Dredgion Crash Site
    win.set_region("Dredgion Crash Site")
    assert win._region_name == "Altgard"
    assert win._route_name == "Dredgion Crash Site"
    # Dredgion node #1 should be in route_ids, #2 should not
    assert 1 in win.canvas._route_node_ids
    assert 2 not in win.canvas._route_node_ids

    # Subregion: Safe Haven
    win.set_region("Safe Haven")
    assert win._region_name == "Altgard"
    assert win._route_name == "Safe Haven"
    assert 2 in win.canvas._route_node_ids
    assert 1 not in win.canvas._route_node_ids

    # Subregion: Calderon Canyon
    win.set_region("Calderon Canyon")
    assert win._region_name == "Altgard"
    assert win._route_name == "Calderon Canyon"


# ---------------------------------------------------------------------------
# Test 2: Background Map Selection
# ---------------------------------------------------------------------------
def test_background_map_switching(qapp):
    """Seamless switching between HD, reference frame, mosaic, and auto."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm)

    # 1. Mosaic mode
    win.set_background_mode("mosaic")
    assert win.get_background_mode() == "mosaic"
    assert win.combo_bg_mode.currentIndex() == 3
    assert win.canvas._base_pixmap is not None
    assert not win.canvas._base_pixmap.isNull()
    assert win.canvas._base_pixmap.width() == 1180
    assert win.canvas._base_pixmap.height() == 740

    # 2. Reference frame mode
    win.set_background_mode("frame18")
    assert win.get_background_mode() == "frame18"
    assert win.combo_bg_mode.currentIndex() == 2
    assert win.canvas._base_pixmap is not None
    assert not win.canvas._base_pixmap.isNull()

    # 3. HD Map mode
    win.set_background_mode("hd")
    assert win.get_background_mode() == "hd"
    assert win.combo_bg_mode.currentIndex() == 1
    assert win.canvas._base_pixmap is not None
    assert not win.canvas._base_pixmap.isNull()
    # Altgard HD is 2258x2400
    assert win.canvas._base_pixmap.width() == 2258
    assert win.canvas._base_pixmap.height() == 2400

    # 4. Auto mode
    win.set_background_mode("auto")
    assert win.get_background_mode() == "auto"
    assert win.combo_bg_mode.currentIndex() == 0


def test_background_mode_combo_signal(qapp):
    """Changing combo box index triggers set_background_mode."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm)
    win.combo_bg_mode.setCurrentIndex(3)  # Mosaic
    assert win.get_background_mode() == "mosaic"
    win.combo_bg_mode.setCurrentIndex(1)  # HD
    assert win.get_background_mode() == "hd"


# ---------------------------------------------------------------------------
# Test 3: AutoCartographer Live Sync
# ---------------------------------------------------------------------------
def test_autocartographer_live_sync(qapp):
    """Live auto-mapped points from AutoCartographer appear in MapViewerWindow."""
    rm = MockRouteManager()
    carto = MockAutoCartographer()
    win = MapViewerWindow(route_manager=rm)

    # Attach AutoCartographer
    win.set_auto_cartographer(carto)
    assert win.auto_cartographer is carto

    # Node 991 from carto should now be merged into canvas
    canvas_node_ids = [n.get("id") for n in win.canvas._nodes]
    assert 991 in canvas_node_ids

    # Check source_type tag
    carto_node = next(n for n in win.canvas._nodes if n.get("id") == 991)
    assert carto_node.get("source_type") == "minimap_auto"

    # UI badge must display the auto-mapped count
    assert "◎ Minimap/3B Oto-Kayıt (1)" in win.lbl_badge_minimap_auto.text()

    # Check odometry polling updates player position
    carto.map_x = 555.0
    carto.map_y = 666.0
    win.poll_live_updates()
    assert win.canvas._player_pos is not None
    assert win.canvas._player_pos[0] == 555.0
    assert win.canvas._player_pos[1] == 666.0


# ---------------------------------------------------------------------------
# Test 4: Interactive Canvas Callbacks & Atomic Persistence
# ---------------------------------------------------------------------------
def test_interactive_node_add_atomic_persistence(qapp):
    """Adding a node atomically updates self.routes._regions[reg]['nodes'] and calls save_region."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm)
    initial_count = len(rm._regions["Altgard"]["nodes"])

    # Add node via add_node
    new_node = win.add_node(500.0, 320.0, resource_type="Aria Herb")
    assert new_node["id"] == 3
    assert new_node["region"] == "Altgard"
    assert new_node["map_x"] == 500
    assert new_node["map_y"] == 320
    assert len(rm._regions["Altgard"]["nodes"]) == initial_count + 1
    assert "Altgard" in rm.saved_regions

    # Add node via canvas callback slot _on_canvas_node_added
    win._on_canvas_node_added((600.0, 400.0))
    assert len(rm._regions["Altgard"]["nodes"]) == initial_count + 2
    assert rm._regions["Altgard"]["nodes"][-1]["id"] == 4


def test_interactive_node_move_atomic_persistence(qapp):
    """Moving a node updates coordinates and immediately saves."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm)

    # Move node #1
    success = win.move_node(1, 510.0, 315.0)
    assert success is True
    assert rm._regions["Altgard"]["nodes"][0]["map_x"] == 510
    assert rm._regions["Altgard"]["nodes"][0]["map_y"] == 315
    assert "Altgard" in rm.saved_regions

    # Move via slot _on_canvas_node_moved
    win._on_canvas_node_moved(2, 490.0, 370.0)
    assert rm._regions["Altgard"]["nodes"][1]["map_x"] == 490
    assert rm._regions["Altgard"]["nodes"][1]["map_y"] == 370


def test_interactive_node_delete_atomic_persistence(qapp):
    """Deleting a node removes it, renumbers seq_index, and immediately saves."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm)

    # Delete node #1
    win.delete_node(1)
    remaining_ids = [n.get("id") if isinstance(n, dict) else n.id for n in rm._regions["Altgard"]["nodes"]]
    assert 1 not in remaining_ids
    assert 2 in remaining_ids
    # seq_index of node #2 should now be 1
    assert rm._regions["Altgard"]["nodes"][0]["seq_index"] == 1
    assert "Altgard" in rm.saved_regions


def test_route_toggling_atomic_persistence(qapp):
    """Toggling a node in/out of active route updates and saves."""
    rm = MockRouteManager()
    win = MapViewerWindow(route_manager=rm)
    win._route_name = "CustomRoute"

    # Toggle node #1 into CustomRoute
    win._toggle_node_in_active_route(1)
    assert "CustomRoute" in rm._regions["Altgard"]["nodes"][0]["routes"]
    assert "Altgard" in rm.saved_regions

    # Toggle node #1 out of CustomRoute
    win._toggle_node_in_active_route(1)
    assert "CustomRoute" not in rm._regions["Altgard"]["nodes"][0]["routes"]


# ---------------------------------------------------------------------------
# Test 5: Real RouteManager Disk Persistence Integration
# ---------------------------------------------------------------------------
def test_real_routemanager_disk_persistence(qapp):
    """Verifies that add/move/delete operations on MapViewerWindow persist genuine JSON to disk."""
    temp_dir = tempfile.mkdtemp(prefix="gatherion_test_routes_")
    try:
        rm = RouteManager(base_dir=temp_dir)
        # Create Altgard region
        rm.create_region("Altgard")
        assert "Altgard" in rm._regions

        win = MapViewerWindow(route_manager=rm, region_name="Altgard")

        # 1. Add node
        n1 = win.add_node(400, 300, resource_type="Orichalcum Ore")
        json_path = os.path.join(temp_dir, "Altgard", "region_data.json")
        assert os.path.isfile(json_path)

        # Verify disk contents after add
        rm2 = RouteManager(base_dir=temp_dir)
        rm2.load_all_regions()
        assert len(rm2._regions["Altgard"]["nodes"]) == 1
        assert rm2._regions["Altgard"]["nodes"][0].id == n1["id"]
        assert rm2._regions["Altgard"]["nodes"][0].map_x == 400

        # 2. Move node
        win.move_node(n1["id"], 550, 450)
        rm3 = RouteManager(base_dir=temp_dir)
        rm3.load_all_regions()
        assert rm3._regions["Altgard"]["nodes"][0].map_x == 550
        assert rm3._regions["Altgard"]["nodes"][0].map_y == 450

        # 3. Add second node
        n2 = win.add_node(600, 500)
        rm4 = RouteManager(base_dir=temp_dir)
        rm4.load_all_regions()
        assert len(rm4._regions["Altgard"]["nodes"]) == 2

        # 4. Delete first node
        win.delete_node(n1["id"])
        rm5 = RouteManager(base_dir=temp_dir)
        rm5.load_all_regions()
        assert len(rm5._regions["Altgard"]["nodes"]) == 1
        assert rm5._regions["Altgard"]["nodes"][0].id == n2["id"]
        assert rm5._regions["Altgard"]["nodes"][0].seq_index == 1

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
