# -*- coding: utf-8 -*-
"""
Test Worker MR6 Map & Radar Test Suite (test_worker_mr6_map_radar.py)

Covers:
1. Radar Coordinate Ingestion:
   - 3D camera detections and minimap sprout detections generate accurate world coordinates without center-sticking.
   - Dead-reckoning displacement and odometry integration.
   - VisualOdometryTracker coordinate synchronization.
2. Spatial Deduplication & Clustering:
   - Intra-frame clustering of multiple close detections.
   - Inter-frame spatial deduplication avoiding duplicate stacked nodes.
   - ExplorerModeController ordered nodes deduplication within cluster radius.
3. Real-time Radar Event Notification:
   - Node discovered and node updated callback emission.
   - Instantaneous updates to registered radar widgets and map viewers.
"""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock

from core.auto_cartographer import (
    AutoCartographer,
    VisualOdometryTracker,
    TrackedResourceNode,
    DEFAULT_DEDUP_RADIUS_PX,
)
from core.explorer_mode import ExplorerModeController
from core.route_manager import RouteManager, NodeData


class TestRadarCoordinateIngestion:
    """Verifies radar coordinate ingestion without center-sticking and with dead-reckoning."""

    def test_odometry_tracker_synchronization_and_offset_projection(self):
        """VisualOdometryTracker uses updated player position rather than sticking to (960, 540)."""
        tracker = VisualOdometryTracker(map_width=1920, map_height=1080)
        # Default start
        assert tracker.map_x == 960.0
        assert tracker.map_y == 540.0

        # Update position
        tracker.map_x = 450.0
        tracker.map_y = 300.0
        wx, wy, nx, ny = tracker.offset_to_world_coords(dx=25.0, dy=-15.0)

        assert wx == 475.0
        assert wy == 285.0
        assert nx == round(475.0 / 1920, 4)
        assert ny == round(285.0 / 1080, 4)

    def test_dead_reckoning_integration_in_world_coords(self):
        """Dead-reckoning displacement is factored into projected coordinates."""
        tracker = VisualOdometryTracker(map_width=1920, map_height=1080)
        tracker.map_x = 500.0
        tracker.map_y = 500.0
        tracker.dead_reckoning_dx = 12.0
        tracker.dead_reckoning_dy = -8.0

        wx, wy, nx, ny = tracker.offset_to_world_coords(
            dx=30.0,
            dy=20.0,
            dead_reckoning_dx=tracker.dead_reckoning_dx,
            dead_reckoning_dy=tracker.dead_reckoning_dy,
        )

        # 500 + 30 + 12 = 542, 500 + 20 - 8 = 512
        assert wx == 542.0
        assert wy == 512.0

    def test_auto_cartographer_set_position_syncs_tracker(self):
        """AutoCartographer.set_position keeps tracker coordinates in sync."""
        carto = AutoCartographer()
        carto.set_position(320.0, 640.0, map_width=1600, map_height=900)

        assert carto.map_x == 320.0
        assert carto.map_y == 640.0
        assert carto.odometry_tracker.map_x == 320.0
        assert carto.odometry_tracker.map_y == 640.0

        wx, wy, _, _ = carto.offset_to_world_coords(dx=40.0, dy=-20.0)
        assert wx == 360.0
        assert wy == 620.0

    def test_3d_camera_detection_no_center_sticking(self):
        """3D camera detections across different screen coordinates project to distinct world coordinates."""
        carto = AutoCartographer()
        carto.set_position(1000.0, 500.0)

        # Left side of screen vs right side of screen
        left_det = carto._parse_material_detection(
            {"screen_x": 400, "screen_y": 700, "resource_type": "Aria", "source_type": "3d_auto"},
            frame_w=1920,
            frame_h=1080,
        )
        right_det = carto._parse_material_detection(
            {"screen_x": 1500, "screen_y": 700, "resource_type": "Aria", "source_type": "3d_auto"},
            frame_w=1920,
            frame_h=1080,
        )

        assert left_det is not None
        assert right_det is not None
        assert left_det["rel_map_dx"] < 0  # To the left of player
        assert right_det["rel_map_dx"] > 0  # To the right of player

        left_wx, _, _, _ = carto.offset_to_world_coords(left_det["rel_map_dx"], left_det["rel_map_dy"])
        right_wx, _, _, _ = carto.offset_to_world_coords(right_det["rel_map_dx"], right_det["rel_map_dy"])

        assert left_wx < 1000.0
        assert right_wx > 1000.0
        assert abs(left_wx - right_wx) > 50.0

    def test_minimap_sprout_detection_offset_calculation(self):
        """Minimap detections calculate offsets based on minimap center rather than screen center."""
        carto = AutoCartographer()
        carto.set_position(800.0, 400.0)
        # Minimap ROI for 1920x1080
        x1, y1, x2, y2 = carto._get_minimap_pixel_rect(1920, 1080)
        mcx = (x1 + x2) / 2.0
        mcy = y1 + (y2 - y1) * 0.545

        # Sprout located 30px to the east and 20px to the south of minimap center
        sprout_sx = mcx + 30.0
        sprout_sy = mcy + 20.0

        det = carto._parse_material_detection(
            {"screen_x": sprout_sx, "screen_y": sprout_sy, "resource_type": "Gatherable Material"},
            frame_w=1920,
            frame_h=1080,
        )

        assert det is not None
        assert det["source_type"] == "minimap_auto"
        assert det["rel_map_dx"] > 0
        assert det["rel_map_dy"] > 0

        wx, wy, _, _ = carto.offset_to_world_coords(det["rel_map_dx"], det["rel_map_dy"])
        assert wx > 800.0
        assert wy > 400.0


class TestSpatialDeduplicationAndClustering:
    """Verifies spatial clustering and deduplication within cluster radius."""

    def test_intra_frame_cluster_detections(self):
        """Multiple detections within cluster_radius in a single frame are merged."""
        carto = AutoCartographer()
        # 3 detections: 2 are very close (dx=10, dy=12 and dx=14, dy=16, distance ~5.6px <= 22px), 1 is far (dx=100, dy=100)
        detections = [
            {"rel_map_dx": 10.0, "rel_map_dy": 12.0, "confidence": 0.8, "resource_type": "Gatherable Material"},
            {"rel_map_dx": 14.0, "rel_map_dy": 16.0, "confidence": 0.9, "resource_type": "Aria"},
            {"rel_map_dx": 100.0, "rel_map_dy": 100.0, "confidence": 0.85, "resource_type": "Mela"},
        ]

        clustered = carto.cluster_detections(detections, cluster_radius=22.0)
        assert len(clustered) == 2

        # First cluster is the average of detection 1 and 2
        cl1 = clustered[0]
        assert cl1["_cluster_count"] == 2
        assert math.isclose(cl1["rel_map_dx"], 12.0, abs_tol=0.1)
        assert math.isclose(cl1["rel_map_dy"], 14.0, abs_tol=0.1)
        assert cl1["confidence"] == 0.9
        assert cl1["resource_type"] == "Aria"  # Inherited specific material

    def test_inter_frame_deduplication_refines_kalman_ema(self):
        """Observations of the same node across consecutive frames are merged without creating stacked nodes."""
        carto = AutoCartographer()
        carto.set_position(500.0, 500.0)

        # Manually register a node in region
        tracked = TrackedResourceNode(
            node_id=1,
            region="test_reg",
            routes=["Default"],
            resource_type="Gatherable Material",
            map_x=520.0,
            map_y=510.0,
            norm_x=520.0 / 1920,
            norm_y=510.0 / 1080,
            sightings_count=1,
        )
        carto._tracked_nodes["test_reg"] = [tracked]

        # Second observation nearby (node_map_x = 522, node_map_y = 512, distance ~2.8px <= 22px)
        nearest, dist = carto._find_nearest_tracked_node("test_reg", 522.0, 512.0)
        assert nearest is not None
        assert nearest.node_id == 1
        assert dist < carto.dedup_radius_px

        # Refine existing node
        carto._refine_existing_node_kalman_ema(
            tracked=nearest,
            obs_map_x=522.0,
            obs_map_y=512.0,
            obs_confidence=0.9,
            obs_stack_count=2,
            obs_resource_type="Aria",
        )

        assert nearest.sightings_count == 2
        assert nearest.resource_type == "Aria"
        assert nearest.stack_count == 2
        assert 520.0 <= nearest.map_x <= 522.0
        assert 510.0 <= nearest.map_y <= 512.0

    def test_explorer_mode_ordered_nodes_spatial_deduplication(self):
        """ExplorerModeController does not create stacked duplicates in _ordered_nodes."""
        rm = RouteManager()
        rm.create_region("test_explorer_reg")
        rm.active_region = "test_explorer_reg"

        explorer = ExplorerModeController(route_manager=rm, region="test_explorer_reg")
        explorer.auto_cartographer.set_position(600.0, 600.0)

        # Node 1 at (620, 610)
        n1 = NodeData(id=1, map_x=620, map_y=610, region="test_explorer_reg", resource_type="Aria")
        explorer._ordered_nodes.append(n1)

        # Simulating step with newly discovered node spatially adjacent at (623, 612) (dist ~3.6px)
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        # Mock newly discovered node from auto_cartographer that might have received a different ID
        new_node_dict = {
            "id": 2,
            "node_id": 2,
            "map_x": 623,
            "map_y": 612,
            "region": "test_explorer_reg",
            "resource_type": "Aria",
            "max_harvests": 6,
        }

        # Step cartography with mocked process_frame_discoveries
        explorer.auto_cartographer.process_frame_discoveries = MagicMock(return_value=[new_node_dict])
        res = explorer.step_passive_cartography(dummy_frame)

        # _ordered_nodes should NOT have added duplicate stacked node; it merges with n1
        assert len(explorer._ordered_nodes) == 1
        assert explorer._ordered_nodes[0].id == 1
        assert explorer._ordered_nodes[0].max_harvests == 6


class TestRealTimeRadarEventNotification:
    """Verifies callbacks and signal emissions for radar canvas and map viewer updates."""

    def test_auto_cartographer_callback_emission(self):
        """Emits callbacks to discovered, updated, and UI listeners instantaneously."""
        carto = AutoCartographer()

        discovered_mock = MagicMock()
        updated_mock = MagicMock()
        radar_event_mock = MagicMock()
        ui_listener_mock = MagicMock()

        carto.add_node_discovered_callback(discovered_mock)
        carto.add_node_updated_callback(updated_mock)
        carto.add_radar_event_callback(radar_event_mock)
        carto.register_ui_listener(ui_listener_mock)

        test_node = {"node_id": 42, "map_x": 100, "map_y": 200, "resource_type": "Mela"}

        # Discovery emission
        carto._emit_node_discovered(test_node)
        discovered_mock.assert_called_once_with(test_node)
        ui_listener_mock.add_discovered_node.assert_called_once_with(test_node, trigger_pulse=True)
        radar_event_mock.assert_called_with("node_discovered", test_node)

        # Update emission
        carto._emit_node_updated(test_node)
        updated_mock.assert_called_once_with(test_node)
        ui_listener_mock.update_node.assert_called_once_with(test_node)
        radar_event_mock.assert_called_with("node_updated", test_node)

    def test_explorer_mode_ui_listener_registration_and_notification(self):
        """ExplorerModeController registers radar widget and map viewer and notifies on step discoveries."""
        rm = RouteManager()
        rm.create_region("notif_reg")
        rm.active_region = "notif_reg"

        explorer = ExplorerModeController(route_manager=rm, region="notif_reg")

        radar_widget = MagicMock()
        map_viewer = MagicMock()
        on_discovered_cb = MagicMock()

        explorer.register_radar_widget(radar_widget)
        explorer.register_map_viewer(map_viewer)
        explorer.add_node_discovered_callback(on_discovered_cb)

        test_node = {"node_id": 99, "map_x": 350, "map_y": 450, "resource_type": "Orichalcum Ore"}
        explorer._emit_node_discovered(test_node)

        radar_widget.add_discovered_node.assert_called_once_with(test_node, trigger_pulse=True)
        map_viewer.reload_map_and_nodes.assert_called_once()
        map_viewer.update.assert_called_once()
        on_discovered_cb.assert_called_once_with(test_node)
