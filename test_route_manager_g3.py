# -*- coding: utf-8 -*-
"""
Agent G3 Verification Test Suite.
Tests Route Creation, F8 Hotkey Recording, Snap-to-Path, Deduplication, Sequence Indexing, and Atomic Persistence.
"""

import json
import math
import os
import shutil
import tempfile
import unittest
import numpy as np

from core.route_manager import (
    DEFAULT_MAX_HARVESTS,
    DEFAULT_REGION,
    DEFAULT_ROUTE,
    NodeData,
    RouteManager,
    SnapResult,
    get_route_manager_capabilities,
)


class MockOdometryTracker:
    def __init__(self, x: float = 100.0, y: float = 100.0):
        self.map_x = float(x)
        self.map_y = float(y)

    def get_position(self):
        return (self.map_x, self.map_y, self.map_x / 1920.0, self.map_y / 1080.0)

    def set_position(self, x: float, y: float):
        self.map_x = float(x)
        self.map_y = float(y)


class MockVisionEngine:
    def __init__(self, w: int = 1920, h: int = 1080, player_pos=(500, 500)):
        self.screen_w = w
        self.screen_h = h
        self.player_pos = player_pos
        self.odometry_tracker = MockOdometryTracker(player_pos[0], player_pos[1])

    def grab_frame(self):
        frame = np.full((self.screen_h, self.screen_w, 3), 120, dtype=np.uint8)
        # Draw a bright center arrow on minimap zone
        cx, cy = int(round(self.screen_w * 0.923)), int(round(self.screen_h * 0.165))
        frame[cy - 5 : cy + 5, cx - 5 : cx + 5] = [255, 255, 255]
        return frame

    def is_map_open(self, frame=None):
        return False

    def find_player_on_map(self, frame=None):
        return self.player_pos


class TestRouteManagerG3(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rm_g3_test_")
        self.rm = RouteManager(base_dir=self.temp_dir, viewport_width=1920, viewport_height=1080)

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_capabilities_contract(self):
        caps = get_route_manager_capabilities()
        self.assertIn("snap_to_path", caps)
        self.assertIn("odometry_snap", caps)
        self.assertIn("intermediate_waypoints", caps)
        self.assertIn("deduplicate_nodes", caps)
        self.assertIn("sequence_indexing", caps)
        self.assertIn("atomic_persistence", caps)

    def test_f8_waypoint_recording_basic(self):
        """F8 records a node, saves 80x80 patch, sets max_harvests=3 and seq_index=1."""
        vision = MockVisionEngine(player_pos=(400, 300))
        node = self.rm.record_current_node(
            input_ctrl=None,
            vision_engine=vision,
            region="Verteron",
            route="Tolbas Route",
            resource_type="Aria Herb",
        )

        self.assertIsNotNone(node)
        self.assertEqual(node.region, "Verteron")
        self.assertIn("Tolbas Route", node.routes)
        self.assertEqual(node.seq_index, 1)
        self.assertEqual(node.max_harvests, 3)
        self.assertEqual(node.current_harvests, 0)
        self.assertEqual(node.resource_type, "Aria Herb")
        self.assertEqual(node.map_x, 400)
        self.assertEqual(node.map_y, 300)

        # Check patch file existence on disk
        disk_patch = self.rm._resolve_patch_disk_path(node)
        self.assertTrue(os.path.isfile(disk_patch))

        # Check region_data.json existence and content
        json_path = self.rm._get_region_json_path("Verteron")
        self.assertTrue(os.path.isfile(json_path))
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["seq_index"], 1)

    def test_f8_waypoint_recording_rapid_duplicate_prevention(self):
        """Rapid duplicate F8 clicks while stationary return existing node without spamming duplicates."""
        vision = MockVisionEngine(player_pos=(400, 300))
        node1 = self.rm.record_current_node(
            vision_engine=vision,
            region="Verteron",
            route="Main",
            min_record_distance=10.0,
        )
        self.assertEqual(node1.seq_index, 1)

        # Second press at exactly same location (or within 10px)
        node2 = self.rm.record_current_node(
            vision_engine=vision,
            region="Verteron",
            route="Main",
            min_record_distance=10.0,
        )
        self.assertEqual(node1.id, node2.id)

        # Confirm only 1 node exists in route
        nodes = self.rm.get_route_nodes("Verteron", "Main")
        self.assertEqual(len(nodes), 1)

    def test_snap_to_path_orthogonal_projection(self):
        """Orthogonal projection onto a polyline segment."""
        # Create a horizontal route segment: A=(100, 200) to B=(300, 200)
        nA = self.rm.add_node(region="Verteron", route="Loop", map_x=100, map_y=200)
        nB = self.rm.add_node(region="Verteron", route="Loop", map_x=300, map_y=200)

        # Test query point at (200, 215) -> should snap to (200, 200) with dist=15.0
        snap_res = self.rm.snap_to_path(point=(200, 215), region="Verteron", route="Loop", max_snap_distance=25.0)

        self.assertTrue(snap_res.snapped)
        self.assertEqual(snap_res.x, 200)
        self.assertEqual(snap_res.y, 200)
        self.assertAlmostEqual(snap_res.distance, 15.0, places=2)
        self.assertEqual(snap_res.segment_index, 0)
        self.assertAlmostEqual(snap_res.projection_t, 0.5, places=2)

        # Test tuple unpacking
        coords, is_snapped, dist = snap_res
        self.assertEqual(coords, (200, 200))
        self.assertTrue(is_snapped)
        self.assertAlmostEqual(dist, 15.0, places=2)

        # Test dict access
        self.assertEqual(snap_res["coords"], (200, 200))

    def test_snap_to_path_beyond_threshold(self):
        """Point further than max_snap_distance returns original coordinates with snapped=False."""
        self.rm.add_node(region="Verteron", route="Loop", map_x=100, map_y=200)
        self.rm.add_node(region="Verteron", route="Loop", map_x=300, map_y=200)

        # Point at (200, 300) -> distance 100 > threshold 30
        snap_res = self.rm.snap_to_path(point=(200, 300), region="Verteron", route="Loop", max_snap_distance=30.0)

        self.assertFalse(snap_res.snapped)
        self.assertEqual(snap_res.x, 200)
        self.assertEqual(snap_res.y, 300)
        self.assertAlmostEqual(snap_res.distance, 100.0, places=2)

    def test_snap_odometry_position_resets_drift(self):
        """Odometry tracker position with slight drift is snapped to route and tracker coords are corrected."""
        self.rm.add_node(region="Verteron", route="Corridor", map_x=500, map_y=100)
        self.rm.add_node(region="Verteron", route="Corridor", map_x=500, map_y=500)

        # Player walking down the corridor at x=500, but odometry drifted to x=512, y=300
        tracker = MockOdometryTracker(x=512.0, y=300.0)
        res = self.rm.snap_odometry_position(odometry_tracker=tracker, region="Verteron", route="Corridor", max_snap_distance=25.0)

        self.assertIsNotNone(res)
        self.assertTrue(res.snapped)
        self.assertEqual(res.x, 500)
        self.assertEqual(res.y, 300)
        # Verify tracker was updated
        self.assertEqual(tracker.map_x, 500.0)
        self.assertEqual(tracker.map_y, 300.0)

    def test_f8_recording_with_snap_to_path(self):
        """F8 recording with snap_to_path=True automatically snaps to the corridor."""
        self.rm.add_node(region="Verteron", route="Main", map_x=200, map_y=100)
        self.rm.add_node(region="Verteron", route="Main", map_x=200, map_y=500)

        # Player presses F8 at (210, 250) slightly off center
        vision = MockVisionEngine(player_pos=(210, 250))
        node = self.rm.record_current_node(
            vision_engine=vision,
            region="Verteron",
            route="Main",
            snap_to_path=True,
            max_snap_distance=20.0,
        )

        self.assertEqual(node.map_x, 200)
        self.assertEqual(node.map_y, 250)

    def test_interpolate_intermediate_waypoints_on_long_segments(self):
        """Long segment (> max_segment_distance) is divided by intermediate waypoints."""
        # 2 nodes with distance 400px (100, 100) to (500, 100)
        nA = self.rm.add_node(region="Altgard", route="Trail", map_x=100, map_y=100, resource_type="Herb")
        nB = self.rm.add_node(region="Altgard", route="Trail", map_x=500, map_y=100, resource_type="Ore")

        # Interpolate with max_segment_distance=150 (400 / 150 -> 3 segments -> 2 intermediate nodes)
        created = self.rm.interpolate_intermediate_waypoints(
            region="Altgard",
            route="Trail",
            max_segment_distance=150.0,
            resource_type="Waypoint",
        )

        self.assertEqual(len(created), 2)
        all_nodes = self.rm.get_route_nodes("Altgard", "Trail")
        self.assertEqual(len(all_nodes), 4)

        # Check sequential order and coordinates
        self.assertEqual(all_nodes[0].map_x, 100)
        self.assertEqual(all_nodes[0].seq_index, 1)

        self.assertEqual(all_nodes[1].map_x, 233)
        self.assertEqual(all_nodes[1].seq_index, 2)
        self.assertTrue(all_nodes[1].not_gatherable)

        self.assertEqual(all_nodes[2].map_x, 367)
        self.assertEqual(all_nodes[2].seq_index, 3)
        self.assertTrue(all_nodes[2].not_gatherable)

        self.assertEqual(all_nodes[3].map_x, 500)
        self.assertEqual(all_nodes[3].seq_index, 4)

        # Verify disk persistence of seq_index
        json_path = self.rm._get_region_json_path("Altgard")
        with open(json_path, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        indices = [n["seq_index"] for n in disk_data["nodes"]]
        self.assertEqual(indices, [1, 2, 3, 4])

    def test_deduplicate_nodes(self):
        """Duplicate nodes within min_distance are removed and routes/stats merged."""
        n1 = self.rm.add_node(region="Verteron", route="RouteA", map_x=100, map_y=100, resource_type="Aria Herb")
        n2 = self.rm.add_node(region="Verteron", route="RouteB", map_x=104, map_y=103, resource_type="Aria Herb")  # dist=5px < 12px
        n3 = self.rm.add_node(region="Verteron", route="RouteA", map_x=300, map_y=300, resource_type="Iron Ore")

        removed_ids = self.rm.deduplicate_nodes(region="Verteron", min_distance=12.0)

        self.assertEqual(removed_ids, [n2.id])
        remaining = self.rm.get_nodes("Verteron")
        self.assertEqual(len(remaining), 2)
        self.assertEqual(remaining[0].id, n1.id)
        # Route tags should be merged
        self.assertIn("RouteA", remaining[0].routes)
        self.assertIn("RouteB", remaining[0].routes)

        # Seq index contiguous check
        self.assertEqual(remaining[0].seq_index, 1)
        self.assertEqual(remaining[1].seq_index, 2)

    def test_atomic_persistence_and_backup(self):
        """Atomic save guarantees .tmp clean replace and creates region_data.backup.json."""
        self.rm.add_node(region="Verteron", route="Loop", map_x=150, map_y=250)
        json_path = self.rm._get_region_json_path("Verteron")
        backup_path = os.path.join(self.rm._get_region_dir("Verteron"), "region_data.backup.json")

        self.assertTrue(os.path.isfile(json_path))

        # Add second node to trigger save with backup
        self.rm.add_node(region="Verteron", route="Loop", map_x=350, map_y=450)
        self.assertTrue(os.path.isfile(backup_path))

        # Check backup content has 1 node, main has 2 nodes
        with open(backup_path, "r", encoding="utf-8") as f:
            backup_data = json.load(f)
        with open(json_path, "r", encoding="utf-8") as f:
            main_data = json.load(f)

        self.assertEqual(len(backup_data["nodes"]), 1)
        self.assertEqual(len(main_data["nodes"]), 2)

    def test_sequence_index_ordering_and_movement(self):
        """resequence_nodes, move_node_in_sequence and delete_node maintain strict 1..N order."""
        n1 = self.rm.add_node(region="Verteron", route="Loop", map_x=10, map_y=10)
        n2 = self.rm.add_node(region="Verteron", route="Loop", map_x=20, map_y=20)
        n3 = self.rm.add_node(region="Verteron", route="Loop", map_x=30, map_y=30)
        n4 = self.rm.add_node(region="Verteron", route="Loop", map_x=40, map_y=40)

        # Initial sequence: n1=1, n2=2, n3=3, n4=4
        self.assertEqual([n1.seq_index, n2.seq_index, n3.seq_index, n4.seq_index], [1, 2, 3, 4])

        # Move n4 to position 1
        ok = self.rm.move_node_in_sequence(n4.id, 1, region="Verteron")
        self.assertTrue(ok)
        nodes = self.rm.get_route_nodes("Verteron", "Loop")
        self.assertEqual([n.id for n in nodes], [n4.id, n1.id, n2.id, n3.id])
        self.assertEqual([n.seq_index for n in nodes], [1, 2, 3, 4])

        # Delete middle node (n1)
        self.rm.delete_node(n1.id, region="Verteron")
        nodes_after_del = self.rm.get_route_nodes("Verteron", "Loop")
        self.assertEqual([n.id for n in nodes_after_del], [n4.id, n2.id, n3.id])
        self.assertEqual([n.seq_index for n in nodes_after_del], [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
