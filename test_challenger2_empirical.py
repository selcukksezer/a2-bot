# -*- coding: utf-8 -*-
"""
EMPIRICAL STRESS TEST HARNESS — CHALLENGER 2
Empirically stress-tests:
1. 3+ point route cycling with arbitrary ID ordering (no trapping).
2. Odometry / world coordinates without artificial spirals & viewport scaling (720p/1080p/1440p).
3. Geometric QA validator on overlapping, zero-area, out-of-bounds boxes.
4. Human confirmation gate on pending reviews.
5. GPU throttle game detection when closed vs running.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

# Imports under test
import core.game_gpu_throttle as throttle
from core.auto_cartographer import AutoCartographer, VisualOdometryTracker
from core.route_manager import DEFAULT_REGION, DEFAULT_ROUTE, NodeData, RouteManager
from dataset_factory.dataset_validator import (
    DEFAULT_IOU_OVERLAP_THRESHOLD,
    box_xywh_to_xyxy,
    box_xyxy_to_xywh,
    check_out_of_bounds,
    check_zero_area,
    compute_box_iou,
    count_pending_review_items,
    detect_overlapping_boxes,
    fix_out_of_bounds,
    is_dataset_ready_for_training,
    is_model_stale,
    mark_model_stale,
    sanitize_boxes,
    validate_dataset,
    validate_yolo_file,
)


class TestChallenger2Empirical(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="challenger2_stress_")
        print(f"\n[SETUP] Created temp workspace: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        try:
            shutil.rmtree(cls.temp_dir, ignore_errors=True)
            print(f"[TEARDOWN] Cleaned up temp workspace: {cls.temp_dir}")
        except Exception as exc:
            print(f"[TEARDOWN] Cleanup warning: {exc}")

    # =========================================================================
    # SUITE 1: 3+ Point Route Cycling with Arbitrary IDs (No Trapping)
    # =========================================================================

    def test_01_arbitrary_id_cycling_no_trapping(self):
        """Stress-test arbitrary ID ordering [245, 40, 84, 12, 999] across 5 full cycles."""
        rm = RouteManager()
        reg = "ArbitraryReg"
        rm.create_region(reg)

        arbitrary_ids = [245, 40, 84, 12, 999]
        rm._regions[reg]["nodes"] = [
            NodeData(id=nid, region=reg, seq_index=idx + 1)
            for idx, nid in enumerate(arbitrary_ids)
        ]

        # Call get_next_ready_node across 25 iterations (5 full loops)
        selected_ids = []
        for _ in range(25):
            node = rm.get_next_ready_node(region=reg)
            self.assertIsNotNone(node, "get_next_ready_node returned None on ready route!")
            selected_ids.append(node.id)

        expected_sequence = arbitrary_ids * 5
        self.assertEqual(
            selected_ids,
            expected_sequence,
            f"Cycling trapped or misordered! Got {selected_ids[:10]}... expected {expected_sequence[:10]}...",
        )
        print(f"\n  [PASS] 1.1 Arbitrary ID cycling 5 full loops ({len(selected_ids)} steps): {selected_ids[:5]}...")

    def test_02_cycling_with_resting_and_recovery(self):
        """Stress-test route cycling when an arbitrary node enters resting state and then recovers."""
        rm = RouteManager()
        reg = "RestingTestReg"
        rm.create_region(reg)

        arbitrary_ids = [245, 40, 84, 12, 999]
        nodes = [
            NodeData(id=nid, region=reg, seq_index=idx + 1)
            for idx, nid in enumerate(arbitrary_ids)
        ]
        rm._regions[reg]["nodes"] = nodes

        # Advance 2 nodes (245 -> 40)
        n1 = rm.get_next_ready_node(region=reg)
        n2 = rm.get_next_ready_node(region=reg)
        self.assertEqual(n1.id, 245)
        self.assertEqual(n2.id, 40)

        # Mark node 84 as resting (simulating full 3/3 harvest completed)
        node_84 = next(n for n in nodes if n.id == 84)
        node_84.state = "resting"
        node_84.last_visited_timestamp = time.time()
        node_84.rest_duration_sec = 99999

        # Next nodes should skip 84 and proceed directly to 12 -> 999 -> 245 -> 40
        n3 = rm.get_next_ready_node(region=reg)
        n4 = rm.get_next_ready_node(region=reg)
        n5 = rm.get_next_ready_node(region=reg)
        n6 = rm.get_next_ready_node(region=reg)
        self.assertEqual(n3.id, 12, "Did not skip resting node 84!")
        self.assertEqual(n4.id, 999)
        self.assertEqual(n5.id, 245)
        self.assertEqual(n6.id, 40)

        # Now node 84 finishes rest and becomes active again
        node_84.last_visited_timestamp = time.time() - 100000
        n7 = rm.get_next_ready_node(region=reg)
        self.assertEqual(n7.id, 84, "Node 84 was not picked after recovering from rest!")
        n8 = rm.get_next_ready_node(region=reg)
        self.assertEqual(n8.id, 12)
        print("  [PASS] 1.2 Route cycling skips resting node and smoothly re-integrates upon ready")

    def test_03_partial_harvest_stickiness(self):
        """Stress-test partial harvest (1/3, 2/3): bot must NOT jump to next node until 3/3 complete."""
        rm = RouteManager()
        reg = "PartialReg"
        rm.create_region(reg)
        nodes = [
            NodeData(id=10, region=reg, max_harvests=3, current_harvests=0),
            NodeData(id=20, region=reg, max_harvests=3, current_harvests=0),
            NodeData(id=30, region=reg, max_harvests=3, current_harvests=0),
        ]
        rm._regions[reg]["nodes"] = nodes

        # Pick node 10
        n = rm.get_next_ready_node(region=reg)
        self.assertEqual(n.id, 10)

        # Harvest 1/3 on node 10
        n.current_harvests = 1
        n_again = rm.get_next_ready_node(region=reg)
        self.assertEqual(n_again.id, 10, "Jumped away from node during partial harvest 1/3!")

        # Harvest 2/3 on node 10
        n.current_harvests = 2
        n_again2 = rm.get_next_ready_node(region=reg)
        self.assertEqual(n_again2.id, 10, "Jumped away from node during partial harvest 2/3!")

        # Harvest 3/3 complete -> marks resting
        n.current_harvests = 3
        n.state = "resting"
        n.last_visited_timestamp = time.time()
        n.rest_duration_sec = 600

        # Now next node must be 20
        n_next = rm.get_next_ready_node(region=reg)
        self.assertEqual(n_next.id, 20)
        print("  [PASS] 1.3 Partial harvest priority remains sticky until 3/3 harvests completed")

    def test_04_route_persistence_and_reordering(self):
        """Verify reorder_route_nodes preserves sequence and 1-based seq_index on disk."""
        reg_dir = os.path.join(self.temp_dir, "routes", "ReorderReg")
        os.makedirs(reg_dir, exist_ok=True)
        rm = RouteManager(base_dir=os.path.join(self.temp_dir, "routes"))
        rm.create_region("ReorderReg")

        # Add 4 nodes with arbitrary IDs
        for nid in [500, 200, 800, 100]:
            rm.add_node(region="ReorderReg", node_id=nid, map_x=100, map_y=100)

        # Reorder to [800, 100, 500, 200]
        reordered_ids = [800, 100, 500, 200]
        ok = rm.reorder_route_nodes("ReorderReg", reordered_ids)
        self.assertTrue(ok)

        # Read back from clean RouteManager instance from disk
        rm_clean = RouteManager(base_dir=os.path.join(self.temp_dir, "routes"))
        rm_clean.load_all_regions()
        loaded_nodes = rm_clean.get_route_nodes("ReorderReg")
        loaded_ids = [n.id for n in loaded_nodes]
        loaded_seqs = [n.seq_index for n in loaded_nodes]

        self.assertEqual(loaded_ids, reordered_ids, "Reordered sequence did not persist to disk!")
        self.assertEqual(loaded_seqs, [1, 2, 3, 4], "seq_index was not set to 1-based sequential indices!")
        print(f"  [PASS] 1.4 Route reordering persisted to disk with sequence: {loaded_ids}")

    # =========================================================================
    # SUITE 2: Odometry & Coordinate Normalization Without Spirals
    # =========================================================================

    def test_05_minimap_recording_no_artificial_spirals(self):
        """Verify that record_current_node does NOT produce artificial spirals.
        In the old code, successive calls used angle = count * 0.65, radius = min(...) * (1 + count*0.08)
        which caused coordinates to orbit and spiral away from center artificially.
        """
        routes_base = os.path.join(self.temp_dir, "spiral_test_routes")
        os.makedirs(routes_base, exist_ok=True)
        rm = RouteManager(base_dir=routes_base)
        reg = "SpiralTestReg"
        rm.create_region(reg)

        # Provide a synthetic 1080p frame
        frame_1080p = np.zeros((1080, 1920, 3), dtype=np.uint8)
        # Draw simulated minimap background in upper right
        cv2.rectangle(frame_1080p, (1640, 70), (1900, 275), (100, 100, 100), -1)

        # Record 8 nodes sequentially with identical frame (no player movement)
        recorded_nodes = []
        for i in range(8):
            node = rm.record_current_node(
                input_ctrl=frame_1080p,
                region=reg,
                use_minimap_for_f8=True,
            )
            recorded_nodes.append(node)

        # Calculate distances from center (960, 540)
        # In the old code, distances grew by 18-20px each step in a trigonometric spiral:
        # Step 0: r=0, Step 1: r=18, Step 2: r=28, Step 3: r=38 ... angle rotating 0.65 rad.
        # With fix: when frame is stationary, coords should stay tightly anchored without outward spiraling.
        positions = [(n.map_x, n.map_y) for n in recorded_nodes]
        print(f"\n  [INFO] 8 stationary recorded positions: {positions}")

        # Check whether coordinates exhibit artificial outward spiral expansion
        distances_from_first = [
            math.hypot(p[0] - positions[0][0], p[1] - positions[0][1])
            for p in positions
        ]
        max_drift = max(distances_from_first)
        self.assertLess(
            max_drift,
            50.0,
            f"Artificial spiral detected! Consecutive stationary recordings drifted by {max_drift:.1f}px: {distances_from_first}",
        )

        # Check that norm_x and norm_y are strictly normalized [0.0, 1.0]
        for n in recorded_nodes:
            self.assertTrue(0.0 <= n.norm_x <= 1.0, f"norm_x out of bounds: {n.norm_x}")
            self.assertTrue(0.0 <= n.norm_y <= 1.0, f"norm_y out of bounds: {n.norm_y}")

        print(f"  [PASS] 2.1 Minimap recording verified without artificial spirals (max drift: {max_drift:.1f}px)")

    def test_06_viewport_scaling_720p_1080p_1440p(self):
        """Verify coordinate normalization scales accurately across 720p, 1080p, and 1440p viewports."""
        resolutions = [
            (1280, 720, "720p"),
            (1920, 1080, "1080p"),
            (2560, 1440, "1440p"),
        ]

        rm = RouteManager(base_dir=os.path.join(self.temp_dir, "viewport_routes"))
        rm.create_region("ViewportReg")

        for width, height, label in resolutions:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            # Update route manager viewport
            rm.update_viewport(width, height)
            self.assertEqual(rm.viewport_width, width)
            self.assertEqual(rm.viewport_height, height)

            # Test add_node with map coordinates at 25% and 75%
            test_x = int(round(width * 0.25))
            test_y = int(round(height * 0.75))

            node = rm.add_node(
                region="ViewportReg",
                map_x=test_x,
                map_y=test_y,
                viewport_width=width,
                viewport_height=height,
            )

            # Normalized coordinates should be exactly 0.25 and 0.75
            self.assertAlmostEqual(
                node.norm_x,
                0.25,
                places=3,
                msg=f"Failed {label} norm_x: got {node.norm_x}, expected 0.25",
            )
            self.assertAlmostEqual(
                node.norm_y,
                0.75,
                places=3,
                msg=f"Failed {label} norm_y: got {node.norm_y}, expected 0.75",
            )

            # Test locate_node_on_open_map fallback on dynamic viewport (e.g. when patch is not matched)
            node_fallback = NodeData(
                id=9990 + int(width),
                map_x=test_x,
                map_y=test_y,
                norm_x=0.25,
                norm_y=0.75,
                patch_path="nonexistent_test_patch.png",
            )
            res = rm.locate_node_on_open_map(node_fallback, open_map_frame=frame, match_threshold=0.99)
            self.assertTrue(res.used_fallback)
            expected_fallback_x = int(round(node_fallback.norm_x * width))
            expected_fallback_y = int(round(node_fallback.norm_y * height))
            self.assertEqual(res.x, expected_fallback_x)
            self.assertEqual(res.y, expected_fallback_y)
            print(f"  [PASS] 2.2 Viewport {label} ({width}x{height}) scaling verified: norm=({node.norm_x}, {node.norm_y}) -> ({res.x}, {res.y})")

    def test_07_visual_odometry_tracker_rescaling(self):
        """Verify VisualOdometryTracker handles dynamic resolution re-scaling without warping."""
        tracker = VisualOdometryTracker(initial_map_x=960, initial_map_y=540, map_width=1920, map_height=1080)
        pos = tracker.get_position()
        self.assertEqual(pos, (960.0, 540.0, 0.5, 0.5))

        # Test offset to world coords
        wx, wy, nx, ny = tracker.offset_to_world_coords(dx=50.0, dy=-50.0)
        self.assertEqual(wx, 1010.0)
        self.assertEqual(wy, 490.0)
        self.assertAlmostEqual(nx, round(1010.0 / 1920.0, 4), places=4)
        self.assertAlmostEqual(ny, round(490.0 / 1080.0, 4), places=4)

        # Test AutoCartographer resolution adaptation
        carto = AutoCartographer(initial_map_x=960, initial_map_y=540, map_width=1920, map_height=1080)
        frame_1440p = np.zeros((1440, 2560, 3), dtype=np.uint8)
        carto.update_player_position(frame_1440p)
        self.assertEqual(carto.map_width, 2560)
        self.assertEqual(carto.map_height, 1440)
        self.assertEqual(carto.odometry_tracker.map_width, 2560)
        self.assertEqual(carto.odometry_tracker.map_height, 1440)
        print("  [PASS] 2.3 VisualOdometryTracker and AutoCartographer dynamic viewport adaptation verified")

    # =========================================================================
    # SUITE 3: Geometric QA Validator (sanitize_boxes)
    # =========================================================================

    def test_08_qa_overlapping_boxes_dedup(self):
        """Stress-test IoU > 0.8 deduplication on same class vs different class."""
        # Box A: class 0, center (0.5, 0.5), w=0.2, h=0.2, hand_label=True
        box_a = {"class_id": 0, "cx": 0.50, "cy": 0.50, "w": 0.20, "h": 0.20, "hand_label": True, "confidence": 0.95}
        # Box B: class 0, nearly identical (IoU ~ 0.90), automated weak label
        box_b = {"class_id": 0, "cx": 0.51, "cy": 0.51, "w": 0.20, "h": 0.20, "hand_label": False, "confidence": 0.70}
        # Box C: class 1 (different class), same coords as Box A
        box_c = {"class_id": 1, "cx": 0.50, "cy": 0.50, "w": 0.20, "h": 0.20, "hand_label": False, "confidence": 0.88}
        # Box D: class 0, far away (no overlap, IoU = 0)
        box_d = {"class_id": 0, "cx": 0.10, "cy": 0.10, "w": 0.05, "h": 0.05, "confidence": 0.90}

        cleaned, issues = sanitize_boxes([box_a, box_b, box_c, box_d], iou_threshold=0.80)

        # Expected:
        # Box B is dropped because IoU(A, B) > 0.8 and Box A has higher priority (hand_label).
        # Box C is KEPT because class_id == 1 is different.
        # Box D is KEPT because no overlap.
        self.assertEqual(len(cleaned), 3, f"Expected 3 surviving boxes, got {len(cleaned)}")
        surviving_classes = [b["class_id"] for b in cleaned]
        self.assertEqual(surviving_classes.count(0), 2)  # Box A and Box D
        self.assertEqual(surviving_classes.count(1), 1)  # Box C

        # Verify Box A was kept over Box B
        class_0_center = next(b for b in cleaned if b["class_id"] == 0 and b["cx"] == 0.50)
        self.assertTrue(class_0_center.get("hand_label"), "Hand label was not prioritized during deduplication!")

        # Verify issue was recorded
        overlap_issues = [it for it in issues if it.issue_type == "duplicate_overlap"]
        self.assertGreaterEqual(len(overlap_issues), 1)
        print(f"  [PASS] 3.1 Overlapping boxes (IoU > 0.8): duplicate correctly dropped, issue logged: {overlap_issues[0].message}")

    def test_09_qa_zero_area_and_corrupt_boxes(self):
        """Stress-test zero-area, negative dimension, and NaN box rejection."""
        corrupt_boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.0, "h": 0.1},        # zero width
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.0},        # zero height
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": -0.05, "h": 0.1},      # negative width
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 1e-8, "h": 1e-8},      # sub-minimal area
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": float("nan"), "h": 0.1}, # NaN
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": float("inf")}, # Inf
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},        # VALID box
        ]

        cleaned, issues = sanitize_boxes(corrupt_boxes, drop_zero_area=True)
        self.assertEqual(len(cleaned), 1, f"Expected exactly 1 valid survivor, got {len(cleaned)}")
        self.assertEqual(cleaned[0]["w"], 0.1)
        self.assertEqual(cleaned[0]["h"], 0.1)

        zero_issues = [it for it in issues if it.issue_type == "zero_area"]
        self.assertGreaterEqual(len(zero_issues), 4)
        print(f"  [PASS] 3.2 Zero-area and non-finite boxes dropped ({len(zero_issues)} issues recorded)")

    def test_10_qa_out_of_bounds_and_clipping(self):
        """Stress-test out-of-bounds coordinates (negative, > 1.0, unnormalized pixels)."""
        oob_boxes = [
            # 1. Completely out of bounds (negative cx) -> cannot be saved, must be dropped
            {"class_id": 0, "cx": -0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
            # 2. Completely out of bounds (cx > 1.5) -> cannot be saved, must be dropped
            {"class_id": 0, "cx": 1.5, "cy": 0.5, "w": 0.1, "h": 0.1},
            # 3. Slightly shifted box: center 0.95, width 0.20 -> extends to 1.05
            # auto_fix_coords should clip it to [0.85, 1.0] -> cx=0.925, w=0.15
            {"class_id": 0, "cx": 0.95, "cy": 0.5, "w": 0.20, "h": 0.10},
            # 4. Valid box
            {"class_id": 0, "cx": 0.30, "cy": 0.30, "w": 0.10, "h": 0.10},
        ]

        cleaned, issues = sanitize_boxes(oob_boxes, auto_fix_coords=True)
        # Boxes 1 and 2 should be dropped, box 3 clipped, box 4 preserved -> 2 survivors
        self.assertEqual(len(cleaned), 2, f"Expected 2 surviving boxes, got {len(cleaned)}")

        # Verify clipped box is strictly within [0.0, 1.0]
        clipped_box = next(b for b in cleaned if b["cx"] > 0.8)
        x1, y1, x2, y2 = box_xywh_to_xyxy(clipped_box["cx"], clipped_box["cy"], clipped_box["w"], clipped_box["h"])
        self.assertGreaterEqual(x1, 0.0)
        self.assertLessEqual(x2, 1.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(y2, 1.0)

        # Test fix_out_of_bounds with unnormalized pixel coordinates (e.g. 1920x1080)
        pixel_cx, pixel_cy, pixel_w, pixel_h = 960.0, 540.0, 192.0, 108.0
        fixed = fix_out_of_bounds(pixel_cx, pixel_cy, pixel_w, pixel_h, image_width=1920, image_height=1080)
        self.assertIsNotNone(fixed)
        self.assertEqual(fixed, (0.5, 0.5, 0.1, 0.1))
        print("  [PASS] 3.3 Out-of-bounds coordinate clipping and pixel normalizer verified")

    # =========================================================================
    # SUITE 4: Human Confirmation Gate on Pending Reviews
    # =========================================================================

    def test_11_human_gate_blocks_training_when_pending_exist(self):
        """Verify is_dataset_ready_for_training() strictly blocks training when pending reviews > 0."""
        ds_root = Path(self.temp_dir) / "test_human_gate_dataset"
        review_dir = ds_root / "review" / "train"
        labels_dir = ds_root / "labels" / "train"
        review_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Create 1 valid label file
        (labels_dir / "frame_001.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

        # Create a review file with pending items (trusted empty, weak has items, done is None)
        pending_review = {
            "image": "frame_001.jpg",
            "trusted": [],
            "weak": [{"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": 0.75}],
            "done": None,
        }
        (review_dir / "frame_001.json").write_text(json.dumps(pending_review), encoding="utf-8")

        pending_count = count_pending_review_items(ds_root)
        self.assertEqual(pending_count, 1)

        ready, msg = is_dataset_ready_for_training(ds_root)
        self.assertFalse(ready, "Human gate failed to block training when pending reviews exist!")
        self.assertIn("İnsan Onay Kapısı", msg)
        print(f"  [PASS] 4.1 Human confirmation gate blocked training: {msg}")

        # Now simulate human approval (done=True, trusted has items)
        pending_review["trusted"] = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "human_approved": True}
        ]
        pending_review["weak"] = []
        pending_review["done"] = True
        (review_dir / "frame_001.json").write_text(json.dumps(pending_review), encoding="utf-8")

        pending_count_after = count_pending_review_items(ds_root)
        self.assertEqual(pending_count_after, 0)

        ready_after, msg_after = is_dataset_ready_for_training(ds_root)
        self.assertTrue(ready_after, f"Failed to allow training after human approval! Reason: {msg_after}")
        print(f"  [PASS] 4.2 After human approval (pending=0), dataset is ready for training: {msg_after}")

    def test_12_actual_disk_dataset_gate_check(self):
        """Audit the real disk dataset F:\\a2\\dataset with human gate validator."""
        real_ds = Path("dataset")
        if real_ds.is_dir():
            pending = count_pending_review_items(real_ds)
            ready, msg = is_dataset_ready_for_training(real_ds)
            print(f"\n  [AUDIT] Real disk dataset pending count: {pending}")
            print(f"  [AUDIT] is_dataset_ready_for_training: ready={ready}, msg={msg}")
            if pending > 0:
                self.assertFalse(ready, "Real disk dataset has pending reviews but gate returned ready=True!")
                self.assertIn("İnsan Onay Kapısı", msg)
            print("  [PASS] 4.3 Real disk dataset correctly enforced human gate")

    def test_13_stale_model_marking(self):
        """Verify mark_model_stale properly records stale manifest and flags."""
        models_dir = Path(self.temp_dir) / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        ds_dir = Path(self.temp_dir) / "dataset_stale"
        ds_dir.mkdir(parents=True, exist_ok=True)
        (ds_dir / "pipeline_ui_state.json").write_text(
            json.dumps({"trained": True, "validated": True, "published": True}),
            encoding="utf-8",
        )

        mark_model_stale("Unit test label modification", dataset_dir=ds_dir, models_dir=models_dir)

        is_stale, reason = is_model_stale(dataset_dir=ds_dir, models_dir=models_dir)
        self.assertTrue(is_stale)
        self.assertEqual(reason, "Unit test label modification")

        # Verify pipeline state updated
        state = json.loads((ds_dir / "pipeline_ui_state.json").read_text(encoding="utf-8"))
        self.assertTrue(state.get("model_stale"))
        self.assertFalse(state.get("trained"))
        print("  [PASS] 4.4 mark_model_stale and is_model_stale verified")

    # =========================================================================
    # SUITE 5: GPU Throttle Game Detection When Closed vs Running
    # =========================================================================

    def test_14_gpu_throttle_live_game_detection_when_closed(self):
        """Verify that find_aion_game_processes() and is_game_running() return False/empty
        on this live workstation when Aion 2 is closed, even with VS Code, Python, and Chrome open.
        """
        pids = throttle.find_aion_game_processes()
        running = throttle.is_game_running()

        print(f"\n  [LIVE INSPECTION] find_aion_game_processes() -> {pids}")
        print(f"  [LIVE INSPECTION] is_game_running() -> {running}")

        self.assertEqual(
            pids,
            [],
            f"False positive game detection! Returned PIDs {pids} when game is closed!",
        )
        self.assertFalse(
            running,
            "is_game_running() returned True when Aion 2 is not running!",
        )
        print("  [PASS] 5.1 Live system check: no false positive game detection")

    def test_15_gpu_throttle_executable_and_title_hardening(self):
        """Stress-test process and window title filters against hostile edge cases."""
        # Excluded executables
        excluded_samples = [
            "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
            "code.exe", "cursor.exe", "pycharm64.exe", "explorer.exe",
            "powershell.exe", "cmd.exe", "python.exe", "pythonw.exe", "taskmgr.exe",
        ]
        for exe in excluded_samples:
            self.assertTrue(throttle.is_excluded_exe(exe), f"Failed to exclude {exe}")
            self.assertFalse(throttle.exe_is_aion_game(exe), f"Falsely identified {exe} as game!")

        # Legitimate game executables
        game_samples = [
            "Aion2.exe", "aion2.exe", "AION.bin", "aion.bin", "aion2.bin",
            "Aion.exe", "Aion2Client-Win64-Shipping.exe",
            r"C:\Program Files\NCSOFT\AION2\Aion2Client-Win64-Shipping.exe",
            r"C:\Program Files\NCSOFT\AionClassic\aion.bin",
        ]
        for gexe in game_samples:
            self.assertFalse(throttle.is_excluded_exe(gexe), f"Excluded legitimate game exe {gexe}")
            self.assertTrue(throttle.exe_is_aion_game(gexe), f"Failed to identify game exe {gexe}")

        # Window titles with Aion keyword that MUST be rejected (browsers, IDEs, explorer)
        false_title_samples = [
            "Aion 2 build guide - Google Chrome",
            "aion 2 database - Microsoft Edge",
            "dataset_factory - aion2 - Visual Studio Code",
            "aion2 - Cursor",
            "aion2 - Dosya Gezgini",
            "aion2 - File Explorer",
            "Gatherion-Local v2.0 (Türkçe UI)",
            "Windows PowerShell - aion2",
        ]
        for title in false_title_samples:
            self.assertFalse(
                throttle.title_is_aion_game(title),
                f"Falsely matched browser/IDE/explorer title as game: '{title}'",
            )

        # Legitimate game window titles that MUST be accepted
        legit_titles = ["AION 2", "Aion 2", "aion 2", "Aion", "AION"]
        for title in legit_titles:
            self.assertTrue(
                throttle.title_is_aion_game(title),
                f"Failed to match valid game window title: '{title}'",
            )

        print("  [PASS] 5.2 Process name, executable path, and window title hardening verified")

    # =========================================================================
    # SUITE 6: Adversarial Boundary Conditions & Stress Attacks
    # =========================================================================

    def test_16_single_and_empty_route_cycling(self):
        """Stress-test 1-node route, empty route, and all-resting route."""
        rm = RouteManager()
        rm.create_region("SingleNodeReg")
        rm._regions["SingleNodeReg"]["nodes"] = [NodeData(id=42, region="SingleNodeReg")]

        # Single node should repeat without getting trapped or throwing
        picks = [rm.get_next_ready_node("SingleNodeReg").id for _ in range(5)]
        self.assertEqual(picks, [42, 42, 42, 42, 42])

        # Empty route returns None
        rm.create_region("EmptyRouteReg")
        self.assertIsNone(rm.get_next_ready_node("EmptyRouteReg"))

        # All-resting route returns None without crash
        rm.create_region("AllRestingReg")
        rm._regions["AllRestingReg"]["nodes"] = [
            NodeData(id=1, region="AllRestingReg", state="resting", last_visited_timestamp=time.time(), rest_duration_sec=9999),
            NodeData(id=2, region="AllRestingReg", state="resting", last_visited_timestamp=time.time(), rest_duration_sec=9999),
        ]
        self.assertIsNone(rm.get_next_ready_node("AllRestingReg"))
        print("  [PASS] 6.1 Single-node, empty, and all-resting route edge cases verified")

    def test_17_qa_1000_identical_boxes_scalability(self):
        """Stress-test sanitize_boxes against a 1000-box denial-of-service attack."""
        many_boxes = [
            {"class_id": 0, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1, "confidence": float(i) / 1000.0}
            for i in range(1000)
        ]
        t0 = time.time()
        cleaned, issues = sanitize_boxes(many_boxes, iou_threshold=0.80)
        elapsed = time.time() - t0

        self.assertEqual(len(cleaned), 1, f"Expected 1 survivor from 1000 identical boxes, got {len(cleaned)}")
        self.assertEqual(len(issues), 999)
        self.assertLess(elapsed, 1.0, f"Deduplication took too long: {elapsed:.2f}s")
        print(f"  [PASS] 6.2 1000 identical boxes deduplicated to 1 survivor in {elapsed:.3f}s")

    def test_18_corrupt_json_in_review_queue(self):
        """Verify count_pending_review_items handles malformed JSON without crashing."""
        temp_ds = Path(self.temp_dir) / "corrupt_json_dataset"
        rev_dir = temp_ds / "review" / "train"
        rev_dir.mkdir(parents=True, exist_ok=True)

        (rev_dir / "broken1.json").write_text("{malformed: true, missing quotes", encoding="utf-8")
        (rev_dir / "broken2.json").write_text("", encoding="utf-8")
        (rev_dir / "valid_done.json").write_text(json.dumps({"done": True, "trusted": []}), encoding="utf-8")

        # Must not throw, should safely ignore broken files
        count = count_pending_review_items(temp_ds)
        self.assertEqual(count, 0)
        print("  [PASS] 6.3 Corrupted review JSON files handled gracefully without crashing")


if __name__ == "__main__":
    unittest.main(verbosity=2)
