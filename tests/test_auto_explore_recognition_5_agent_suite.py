# -*- coding: utf-8 -*-
"""
Gatherion-Local v2.0 — 5-Agent Comprehensive Auto-Explore & Material Recognition Test Suite
(tests/test_auto_explore_recognition_5_agent_suite.py)

Exclusive Owner: Agent E5
Focus Areas Verified Across 5 Cooperating Agents (E1..E5):
1. Autonomous Exploration Route Generation & Breadcrumb Graph (E1: AutoCartographer & VisualOdometryTracker):
   - Odometry tracking, minimap texture displacement, dead-reckoning calculation.
   - Breadcrumb trajectory recording with timestamps.
   - Discovered resource nodes connected into navigable routes.
   - Persistence to routes/<region>/region_data.json and 80x80 patch generation.
   - 3x harvest capacity tracking and depletion lifecycle (3 * stack_count).
   - Real-time event notifications (node_discovered, node_updated, radar events).
2. 15-25px Spatial Clustering & Deduplication (E1/E4: AutoCartographer & RouteManager):
   - Cluster radius verified within 15.0 - 25.0 px (default 22.0 px).
   - Sighting deduplication: candidates within radius merged without duplicating nodes.
   - Smooth coordinate refinement via 1D/2D Kalman & EMA weighted average.
   - Sightings count incrementation and timestamp updating.
   - Separate node creation when spatial distance > 25.0 px.
   - Batch multi-detection screen clustering (cluster_detections).
3. Material Taxonomy Classification & Exact 3D Identification (E2: MaterialScanner & materials_db):
   - Comprehensive encyclopedia coverage: Herbs, Ores, Gemstones, Logs, Odyle.
   - Exact 3D HSV visual classification (classify_3d_material_patch):
     Aria (purple crystal foliage), Mela (red berries + green bush), Targena (gold-green herb),
     Orichalcum Ore (bronze/teal vein), Odyle (glowing teal/green orb).
   - Non-material strict rejection (feathers, monster dots, NPCs, UI tokens).
   - ExemplarMemory positive refinement: elevates confidence >= 0.85 and refines classes.
4. Minimap-to-Radar Projection, Sprout Recognition & Clearance (E3: MinimapScanner & CartographyRadar):
   - Minimap sprout (class 9) vs monster dot (class 11) geometric/color separation.
   - Subregion canonical mapping (e.g. Dredgion Crash Site, Safe Haven -> Altgard).
   - High-precision coordinate projection (apply_hd_fit, inv_hd_fit).
   - Close-range clearance (_GATHER_PLAYER_CLEARANCE = 0.075) protecting near-player sprouts.
   - Outer HUD chrome filtering (title bar, clock, compass bezel).
   - Cyclical route wrap-around (N -> 1) with SequentialRoutePlanner.
5. Wrong Node Purge with Negative Exemplar Memory (E4: NodeQualityGate & WrongNodeScan):
   - Multi-signal candidate evaluation (evaluate_node_candidate, GateVerdict).
   - Hard-wrong model class rejection (classes 7, 10, 11, etc.).
   - Rejection of barren stone and flat dirt terrain (is_fake_gray_rock_or_ground).
   - Active learning feedback loop (save_wrong_nodes_as_negative_exemplars).
   - Automatic safety backup (region_data.backup.json) when deletions exceed 50%.
   - Full purge execution and verification of clean route state.
6. Full 5-Agent End-to-End Autonomous Exploration Integration Scenario (E1..E5):
   - Multi-step simulated exploration tour covering walking, discovery, classification,
     deduplication, quality gating, wrong node purge, active learning suppression,
     and playable route generation.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure headless Qt operation
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.auto_cartographer import (
    DEFAULT_DEDUP_RADIUS_PX,
    AutoCartographer,
    TrackedResourceNode,
    VisualOdometryTracker,
)
from core.material_scanner import DetectedMaterial, MaterialScanner
from config.materials_db import (
    AION2_MATERIALS_CATALOG,
    MATERIAL_CATEGORIES,
    classify_3d_material_patch,
    get_material_spec,
    get_region_expected_materials,
    is_rejected_non_material,
)
from core.minimap_scanner import MinimapScanner
from ui.cartography_radar import (
    _GATHER_PLAYER_CLEARANCE,
    CANONICAL_REGION_ALIASES,
    SequentialRoutePlanner,
    apply_hd_fit,
    inv_hd_fit,
    minimap_material_is_recordable,
    minimap_norm_is_hud_chrome,
    resolve_canonical_region,
)
from core.node_quality_gate import (
    HARD_WRONG_CLASS_IDS,
    KEEP_CLASS_IDS,
    GateVerdict,
    conf_min_for_tier,
    evaluate_node_candidate,
    is_fake_gray_rock_or_ground,
    is_hard_wrong_class,
    is_keep_class,
    patch_is_feather,
    patch_is_monster_dot,
)
from core.route_manager import DEFAULT_MAX_HARVESTS, NodeData, RouteManager
from core.wrong_node_scan import (
    WrongNode,
    classify_wrong_nodes,
    purge_wrong_nodes,
    save_wrong_nodes_as_negative_exemplars,
    scan_would_exceed_half,
)
from dataset_factory.exemplar_memory import ExemplarMemory, get_exemplar_memory


# =============================================================================
# Helper Utilities & Synthetic Patches
# =============================================================================

def _create_synthetic_aria_patch(size: int = 80) -> np.ndarray:
    """Create a synthetic purple/violet Aria flower patch."""
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :] = (140, 160, 190)  # Purple hue
    # Add crystal texture detail
    rng = np.random.default_rng(42)
    noise = rng.integers(-15, 15, (size, size, 3), dtype=np.int16)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(np.int16)
    return np.clip(bgr + noise, 0, 255).astype(np.uint8)


def _create_synthetic_mela_patch(size: int = 80) -> np.ndarray:
    """Create a synthetic Mela patch (red berries + green foliage)."""
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[: size // 2, :] = (5, 210, 210)    # Red berries
    hsv[size // 2 :, :] = (50, 160, 160)   # Green foliage
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _create_synthetic_ore_patch(size: int = 80) -> np.ndarray:
    """Create a synthetic Orichalcum Ore patch (teal/bronze mineral vein on dark rock)."""
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[: size // 2, :] = (85, 160, 160)   # Teal mineral vein
    hsv[size // 2 :, :] = (0, 0, 65)       # Dark rock base
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _create_synthetic_odyle_patch(size: int = 80) -> np.ndarray:
    """Create a synthetic Odyle energy orb patch (glowing teal/green)."""
    hsv = np.zeros((size, size, 3), dtype=np.uint8)
    hsv[:, :] = (75, 185, 225)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _create_synthetic_flat_ground_patch(size: int = 80) -> np.ndarray:
    """Create a synthetic barren dirt/flat ground patch."""
    rng = np.random.default_rng(101)
    base = 120.0 + rng.normal(0.0, 3.0, (size, size))
    ground = np.stack([base, base, base], axis=-1)
    return np.clip(ground, 0, 255).astype(np.uint8)


# =============================================================================
# Suite 1: Autonomous Exploration Route Generation & Breadcrumb Graph (E1)
# =============================================================================

class TestAutonomousRouteGeneration(unittest.TestCase):
    """Verifies autonomous route builder, odometry tracking, and node graph persistence."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="test_auto_explore_e1_")
        self.route_mgr = RouteManager(base_dir=self.tmp_dir, active_region="Altgard")
        self.route_mgr.create_region("Altgard")
        self.cartographer = AutoCartographer(
            route_manager=self.route_mgr,
            initial_map_x=600.0,
            initial_map_y=400.0,
            map_width=1920,
            map_height=1080,
            dedup_radius_px=22.0,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_visual_odometry_tracker_displacement_and_dead_reckoning(self) -> None:
        """VisualOdometryTracker updates player position and computes dead-reckoning trajectory."""
        tracker = VisualOdometryTracker(
            initial_map_x=500.0,
            initial_map_y=300.0,
            map_width=1920,
            map_height=1080,
        )
        pos0 = tracker.get_position()
        self.assertAlmostEqual(pos0[0], 500.0)
        self.assertAlmostEqual(pos0[1], 300.0)

        # Update displacement with forward movement (+20 map_x, +15 map_y)
        dx, dy = tracker.update_displacement(dx=20.0, dy=15.0, dt=0.5)
        self.assertEqual(dx, 20.0)
        self.assertEqual(dy, 15.0)

        pos1 = tracker.get_position()
        self.assertAlmostEqual(pos1[0], 520.0)
        self.assertAlmostEqual(pos1[1], 315.0)

        # Verify dead-reckoning projection during brief blind intervals
        step_x, step_y = tracker.apply_dead_reckoning(dt=0.2)
        self.assertGreater(step_x, 0.0)
        self.assertGreater(step_y, 0.0)
        self.assertGreater(tracker.map_x, 520.0)
        self.assertGreater(tracker.map_y, 315.0)

    def test_visual_odometry_breadcrumb_trajectory_recording(self) -> None:
        """Player footprints are continuously logged into walk_trajectory with timestamps."""
        tracker = self.cartographer.odometry_tracker
        self.assertEqual(len(tracker.walk_trajectory), 0)

        # Append trajectory at distinct waypoints
        t0 = time.time()
        tracker.append_trajectory(timestamp=t0)
        tracker.update_displacement(10.0, 5.0)
        tracker.append_trajectory(timestamp=t0 + 1.0)
        tracker.update_displacement(15.0, 8.0)
        tracker.append_trajectory(timestamp=t0 + 2.0)

        traj = self.cartographer.get_walk_trajectory()
        self.assertEqual(len(traj), 3)
        self.assertAlmostEqual(traj[0][0], 600.0)
        self.assertAlmostEqual(traj[1][0], 610.0)
        self.assertAlmostEqual(traj[2][0], 625.0)
        self.assertAlmostEqual(traj[2][2], t0 + 2.0)

    def test_autonomous_route_mapping_and_node_graph_creation(self) -> None:
        """AutoCartographer maps discovered nodes into route with coordinates and metadata."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        det_aria = {
            "rel_map_dx": 15.0,
            "rel_map_dy": 25.0,
            "resource_type": "Aria",
            "confidence": 0.92,
            "source_type": "3d_auto",
        }

        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            discovered = self.cartographer.record_from_3d_world(
                frame,
                detections=[det_aria],
                route_manager=self.route_mgr,
                region="Altgard",
            )

        self.assertEqual(len(discovered), 1)
        node_info = discovered[0]
        self.assertEqual(node_info["id"], 1)
        self.assertEqual(node_info["resource_type"], "Aria")
        self.assertAlmostEqual(node_info["map_x"], 615.0)
        self.assertAlmostEqual(node_info["map_y"], 425.0)
        self.assertEqual(node_info["max_harvests"], 3)
        self.assertEqual(node_info["current_harvests"], 0)

        # RouteManager should reflect the newly mapped node
        rm_nodes = self.route_mgr.get_nodes(region="Altgard")
        self.assertEqual(len(rm_nodes), 1)
        self.assertEqual(rm_nodes[0].resource_type, "Aria")

    def test_route_persistence_live_region_data_json(self) -> None:
        """Discovered nodes and routes are immediately persisted to region_data.json."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        det = {
            "rel_map_dx": 50.0,
            "rel_map_dy": 60.0,
            "resource_type": "Mela",
            "confidence": 0.90,
            "source_type": "3d_auto",
        }

        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            self.cartographer.record_from_3d_world(
                frame,
                detections=[det],
                route_manager=self.route_mgr,
                region="Altgard",
            )

        region_file = os.path.join(self.tmp_dir, "Altgard", "region_data.json")
        self.assertTrue(os.path.exists(region_file))

        with open(region_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertIn("nodes", data)
        self.assertEqual(len(data["nodes"]), 1)
        self.assertEqual(data["nodes"][0]["resource_type"], "Mela")
        self.assertAlmostEqual(data["nodes"][0]["map_x"], 650.0)

    def test_auto_cartographer_harvest_capacity_and_depletion(self) -> None:
        """Nodes follow 3x harvest capacity per stack and enter depletion after max harvests."""
        node = TrackedResourceNode(
            node_id=10,
            region="Altgard",
            map_x=700.0,
            map_y=450.0,
            resource_type="Orichalcum Ore",
            stack_count=2,
            max_harvests=6,  # 3x * 2 stacks = 6 harvests
        )
        self.assertEqual(node.max_harvests, 6)
        self.assertEqual(node.remaining_harvests, 6)
        self.assertFalse(node.is_depleted)

        # Simulate 4 harvest events
        node.current_harvests = 4
        self.assertEqual(node.remaining_harvests, 2)
        self.assertFalse(node.is_depleted)

        # Deplete the node at 6 harvests
        node.current_harvests = 6
        self.assertEqual(node.remaining_harvests, 0)
        self.assertTrue(node.is_depleted)

    def test_auto_cartographer_event_callbacks_and_ui_listeners(self) -> None:
        """AutoCartographer notifies external subscribers via discovery and radar callbacks."""
        discovered_events = []
        radar_events = []

        self.cartographer.add_node_discovered_callback(lambda d: discovered_events.append(d))
        self.cartographer.add_radar_event_callback(lambda ev, data: radar_events.append((ev, data)))

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        det = {
            "rel_map_dx": 30.0,
            "rel_map_dy": 30.0,
            "resource_type": "Gem",
            "confidence": 0.88,
            "source_type": "3d_auto",
        }

        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            self.cartographer.record_from_3d_world(
                frame,
                detections=[det],
                route_manager=self.route_mgr,
                region="Altgard",
            )

        self.assertEqual(len(discovered_events), 1)
        self.assertEqual(discovered_events[0]["resource_type"], "Gem")

        # Radar event for discovery should have fired
        self.assertTrue(any(ev[0] == "node_discovered" for ev in radar_events))

    def test_cartographer_metrics_and_health_summary(self) -> None:
        """get_cartography_stats and health diagnostics provide operational metrics."""
        stats = self.cartographer.get_cartography_stats("Altgard")
        self.assertIn("total_nodes", stats)
        self.assertIn("player_map_x", stats)
        self.assertIn("walk_trajectory_points", stats)

        health = self.cartographer.get_cartographer_health_summary()
        self.assertTrue(health.get("cartographer_ready"))

        version_info = self.cartographer.get_auto_cartographer_version_info()
        self.assertEqual(version_info.get("module"), "auto_cartographer")
        self.assertTrue(version_info.get("ready"))


# =============================================================================
# Suite 2: 15-25px Spatial Clustering & Deduplication (E1/E4)
# =============================================================================

class TestSpatialClusteringAndDeduplication(unittest.TestCase):
    """Verifies 15-25px spatial clustering, EMA Kalman refinement, and anti-duplication."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="test_auto_explore_e2_")
        self.route_mgr = RouteManager(base_dir=self.tmp_dir, active_region="Altgard")
        self.route_mgr.create_region("Altgard")
        self.cartographer = AutoCartographer(
            route_manager=self.route_mgr,
            initial_map_x=500.0,
            initial_map_y=300.0,
            dedup_radius_px=22.0,  # Explicitly within 15-25px range
            ema_alpha=0.30,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_cluster_radius_configuration_within_15_to_25px(self) -> None:
        """The spatial cluster radius is strictly verified within the 15-25px operational band."""
        self.assertGreaterEqual(DEFAULT_DEDUP_RADIUS_PX, 15.0)
        self.assertLessEqual(DEFAULT_DEDUP_RADIUS_PX, 25.0)
        self.assertEqual(self.cartographer.cluster_radius, 22.0)

        # Dynamic setter supports values inside range
        self.cartographer.cluster_radius = 18.5
        self.assertEqual(self.cartographer.cluster_radius, 18.5)
        self.assertEqual(self.cartographer.dedup_radius_px, 18.5)

    def test_spatial_deduplication_within_radius_merges_node(self) -> None:
        """Candidate sightings within 15-25px merge into existing node without duplicating."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Initial sighting at (510, 315) -> creates Node #1
        det1 = {
            "rel_map_dx": 10.0,
            "rel_map_dy": 15.0,
            "resource_type": "Aria",
            "confidence": 0.85,
            "source_type": "3d_auto",
        }
        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res1 = self.cartographer.record_from_3d_world(
                frame, detections=[det1], route_manager=self.route_mgr, region="Altgard"
            )

        self.assertEqual(len(res1), 1)
        self.assertEqual(res1[0]["id"], 1)

        # Second sighting at (514, 318): distance = hypot(4, 3) = 5.0 px <= 22.0 px
        det2 = {
            "rel_map_dx": 14.0,
            "rel_map_dy": 18.0,
            "resource_type": "Aria",
            "confidence": 0.89,
            "source_type": "3d_auto",
        }
        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res2 = self.cartographer.record_from_3d_world(
                frame, detections=[det2], route_manager=self.route_mgr, region="Altgard"
            )

        # res2 must be empty (merged, not newly discovered)
        self.assertEqual(len(res2), 0)

        # Total tracked nodes remain 1, sightings_count incremented to 2
        tracked = self.cartographer.get_tracked_nodes("Altgard")
        self.assertEqual(len(tracked), 1)
        self.assertEqual(tracked[0]["sightings_count"], 2)

    def test_kalman_ema_weighted_coordinate_refinement(self) -> None:
        """Subsequent sightings within radius refine coordinates via Kalman/EMA average."""
        tracked_node = TrackedResourceNode(
            node_id=1,
            region="Altgard",
            map_x=510.0,
            map_y=315.0,
            resource_type="Aria",
            confidence=0.80,
            sightings_count=1,
            kalman_variance=16.0,
        )

        # Observe second sighting at (520.0, 325.0)
        self.cartographer._refine_existing_node_kalman_ema(
            tracked=tracked_node,
            obs_map_x=520.0,
            obs_map_y=325.0,
            obs_confidence=0.90,
            obs_stack_count=1,
            obs_resource_type="Aria",
        )

        self.assertEqual(tracked_node.sightings_count, 2)
        # Position should smoothly move between 510 and 520, not jump directly
        self.assertGreater(tracked_node.map_x, 510.0)
        self.assertLess(tracked_node.map_x, 520.0)
        self.assertGreater(tracked_node.map_y, 315.0)
        self.assertLess(tracked_node.map_y, 325.0)

    def test_new_node_created_when_distance_exceeds_radius(self) -> None:
        """Candidate sighting with distance > 25px creates a distinct, separate node."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Node 1 at (510, 315)
        det1 = {"rel_map_dx": 10.0, "rel_map_dy": 15.0, "resource_type": "Aria", "confidence": 0.85, "source_type": "3d_auto"}
        # Node 2 at (550, 360) -> distance = hypot(40, 45) ≈ 60.2 px > 25.0 px
        det2 = {"rel_map_dx": 50.0, "rel_map_dy": 60.0, "resource_type": "Orichalcum", "confidence": 0.88, "source_type": "3d_auto"}

        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res1 = self.cartographer.record_from_3d_world(frame, detections=[det1], route_manager=self.route_mgr, region="Altgard")
            res2 = self.cartographer.record_from_3d_world(frame, detections=[det2], route_manager=self.route_mgr, region="Altgard")

        self.assertEqual(len(res1), 1)
        self.assertEqual(len(res2), 1)
        self.assertEqual(res1[0]["id"], 1)
        self.assertEqual(res2[0]["id"], 2)

        tracked = self.cartographer.get_tracked_nodes("Altgard")
        self.assertEqual(len(tracked), 2)

    def test_batch_cluster_detections_aggregates_screen_centroids(self) -> None:
        """cluster_detections clusters multi-detection screen boxes within radius."""
        raw_items = [
            {"rel_map_dx": 10.0, "rel_map_dy": 10.0, "resource_type": "Aria", "confidence": 0.80},
            {"rel_map_dx": 14.0, "rel_map_dy": 12.0, "resource_type": "Aria", "confidence": 0.90},  # Distance 4.47 <= 22px
            {"rel_map_dx": 70.0, "rel_map_dy": 70.0, "resource_type": "Mela", "confidence": 0.85},  # Far away
        ]

        clustered = self.cartographer.cluster_detections(raw_items, cluster_radius=22.0)
        self.assertEqual(len(clustered), 2)
        # First cluster merges item 0 and 1, taking higher confidence or weighted centroid
        self.assertAlmostEqual(clustered[0]["rel_map_dx"], 12.0, delta=2.5)
        self.assertEqual(clustered[1]["resource_type"], "Mela")

    def test_multi_stack_harvest_count_preservation_on_deduplication(self) -> None:
        """When an updated sighting exhibits a higher stack count (e.g. 2 or 3), capacity scales."""
        tracked_node = TrackedResourceNode(
            node_id=1,
            region="Altgard",
            map_x=500.0,
            map_y=300.0,
            stack_count=1,
            max_harvests=3,
        )

        # Later sighting detects a 2-stack badge (e.g. stack_count=2, max_harvests=6)
        self.cartographer._refine_existing_node_kalman_ema(
            tracked=tracked_node,
            obs_map_x=502.0,
            obs_map_y=301.0,
            obs_confidence=0.92,
            obs_stack_count=2,
            obs_max_harvests=6,
        )

        self.assertEqual(tracked_node.stack_count, 2)
        self.assertEqual(tracked_node.max_harvests, 6)


# =============================================================================
# Suite 3: Material Taxonomy Classification & Exact 3D Identification (E2)
# =============================================================================

class TestMaterialTaxonomyAnd3DIdentification(unittest.TestCase):
    """Verifies encyclopedia taxonomy, HSV 3D mesh classification, and Exemplar refinement."""

    def setUp(self) -> None:
        self.scanner = MaterialScanner()

    def test_material_spec_catalog_completeness_and_lookup(self) -> None:
        """All 6 core material categories are present and searchable in catalog."""
        self.assertIn("Herbs", MATERIAL_CATEGORIES)
        self.assertIn("Ores & Metals", MATERIAL_CATEGORIES)
        self.assertIn("Gemstones", MATERIAL_CATEGORIES)
        self.assertIn("Logs & Trees", MATERIAL_CATEGORIES)
        self.assertIn("Cooking & Fibers", MATERIAL_CATEGORIES)
        self.assertIn("Odyle & Aether", MATERIAL_CATEGORIES)

        # Lookup by English canonical name
        aria_spec = get_material_spec("Aria")
        self.assertIsNotNone(aria_spec)
        self.assertEqual(aria_spec.category, "Herbs")
        self.assertEqual(aria_spec.max_harvests, 3)

        # Lookup by Turkish name
        safir_spec = get_material_spec("Safir Değerli Taş")
        self.assertIsNotNone(safir_spec)
        self.assertEqual(safir_spec.base_material, "Sapphire Gemstone")

    def test_regional_expected_materials_lookup(self) -> None:
        """get_region_expected_materials returns expected gatherables for official regions."""
        altgard_materials = get_region_expected_materials("Altgard")
        self.assertGreater(len(altgard_materials), 0)
        names = {m.base_material for m in altgard_materials}
        self.assertTrue("Aria" in names or "Calendula" in names or "Orichalcum Ore" in names)

    def test_strict_rejection_of_non_material_tokens(self) -> None:
        """Non-materials (feathers, monster dots, NPCs, quests) are strictly rejected."""
        self.assertTrue(is_rejected_non_material("feather"))
        self.assertTrue(is_rejected_non_material("minimap feather"))
        self.assertTrue(is_rejected_non_material("monster"))
        self.assertTrue(is_rejected_non_material("quest marker"))
        self.assertTrue(is_rejected_non_material("blue wing"))

        # Legitimate gatherable materials are NOT rejected
        self.assertFalse(is_rejected_non_material("Aria"))
        self.assertFalse(is_rejected_non_material("Mela"))
        self.assertFalse(is_rejected_non_material("Orichalcum Ore"))

    def test_classify_3d_material_patch_aria_herb(self) -> None:
        """Purple/violet crystal foliage is classified as Aria."""
        aria_patch = _create_synthetic_aria_patch(40)
        classification = classify_3d_material_patch(aria_patch)

        self.assertEqual(classification["name"], "Aria")
        self.assertEqual(classification["category"], "Herbs")
        self.assertGreaterEqual(classification["confidence"], 0.70)
        self.assertEqual(classification["max_harvests"], 3)

    def test_classify_3d_material_patch_mela_berries(self) -> None:
        """Red berries with green bush foliage is classified as Mela."""
        mela_patch = _create_synthetic_mela_patch(40)
        classification = classify_3d_material_patch(mela_patch)

        self.assertEqual(classification["name"], "Mela")
        self.assertEqual(classification["category"], "Cooking & Fibers")
        self.assertGreaterEqual(classification["confidence"], 0.70)

    def test_classify_3d_material_patch_targena_gold_green(self) -> None:
        """Golden-yellow/green herb foliage is classified as Targena."""
        hsv = np.zeros((40, 40, 3), dtype=np.uint8)
        hsv[:, :] = (35, 160, 180)  # Gold-green
        targena_patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        classification = classify_3d_material_patch(targena_patch)
        self.assertEqual(classification["name"], "Targena")
        self.assertEqual(classification["category"], "Herbs")

    def test_classify_3d_material_patch_orichalcum_ore(self) -> None:
        """Teal/bronze ore vein on dark rock is classified as Orichalcum Ore."""
        ore_patch = _create_synthetic_ore_patch(40)
        classification = classify_3d_material_patch(ore_patch)

        self.assertEqual(classification["name"], "Orichalcum Ore")
        self.assertEqual(classification["category"], "Ores & Metals")
        self.assertGreaterEqual(classification["confidence"], 0.65)

    def test_classify_3d_material_patch_odyle_energy_orb(self) -> None:
        """Glowing teal/green orb is classified as Odyle."""
        odyle_patch = _create_synthetic_odyle_patch(40)
        classification = classify_3d_material_patch(odyle_patch)

        self.assertEqual(classification["name"], "Odyle")
        self.assertEqual(classification["category"], "Odyle & Aether")
        self.assertGreaterEqual(classification["confidence"], 0.70)

    def test_material_scanner_refines_material_via_exemplar_memory(self) -> None:
        """ExemplarMemory positive exemplars refine candidate classes and elevate confidence."""
        tmp = tempfile.mkdtemp(prefix="test_scanner_exemplar_")
        try:
            ex_path = os.path.join(tmp, "exemplars.json")
            mem = get_exemplar_memory(json_path=ex_path)

            # Register positive exemplar for Gem (class_id=4)
            gem_crop = np.zeros((40, 40, 3), dtype=np.uint8)
            gem_crop[:, :] = (200, 150, 40)
            mem.add_positive_exemplar(gem_crop, class_id=4, class_name="gem_sapphire")

            frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
            frame[300:340, 500:540] = gem_crop

            candidate = DetectedMaterial(
                material_type="Raw_Node",
                source="3d_world",
                screen_x=520,
                screen_y=320,
                bbox=(500, 300, 40, 40),
                rel_map_dx=0.0,
                rel_map_dy=0.0,
                estimated_distance_m=12.0,
                confidence=0.60,
            )

            refined = self.scanner._refine_candidates_with_exemplars(
                frame, [candidate], exemplar_memory=mem
            )

            self.assertEqual(len(refined), 1)
            self.assertTrue("Gem" in refined[0].material_type or "Gemstone" in refined[0].material_type)
            self.assertGreaterEqual(refined[0].confidence, 0.85)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# =============================================================================
# Suite 4: Minimap-to-Radar Projection, Sprout Recognition & Clearance (E3)
# =============================================================================

class TestMinimapToRadarProjectionAndClearance(unittest.TestCase):
    """Verifies sprout vs monster discrimination, subregion canonical mapping, and 0.075 clearance."""

    def setUp(self) -> None:
        self.planner = SequentialRoutePlanner()

    def test_minimap_sprout_vs_monster_dot_discrimination(self) -> None:
        """Sprouts (class 9) are kept; red monster dots (class 10/11) are rejected."""
        self.assertFalse(is_hard_wrong_class(9))
        self.assertTrue(is_keep_class(0))   # Aria
        self.assertTrue(is_keep_class(3))   # Orichalcum
        self.assertTrue(is_keep_class(4))   # Gem
        self.assertTrue(is_keep_class(12))  # World Resource

        self.assertTrue(is_hard_wrong_class(10))  # Monster 3D
        self.assertTrue(is_hard_wrong_class(11))  # Monster text/dot
        self.assertFalse(is_keep_class(10))
        self.assertFalse(is_keep_class(11))

        # Rejection of non-material tokens via materials encyclopedia
        self.assertTrue(is_rejected_non_material("feather"))
        self.assertTrue(is_rejected_non_material("monster"))
        self.assertFalse(is_rejected_non_material("Aria"))
        self.assertFalse(is_rejected_non_material("Orichalcum Ore"))

    def test_subregion_canonical_resolution_mapping(self) -> None:
        """Subregions are resolved to canonical region (e.g. Dredgion Crash Site -> Altgard)."""
        self.assertEqual(resolve_canonical_region("Dredgion Crash Site"), "Altgard")
        self.assertEqual(resolve_canonical_region("Safe Haven"), "Altgard")
        self.assertEqual(resolve_canonical_region("Calderon Canyon"), "Altgard")
        self.assertEqual(resolve_canonical_region("Altgard Fortress"), "Altgard")

        # Route planner alias lookup
        self.assertEqual(self.planner.canonical_region("Dredgion Crash Site"), "Altgard")
        self.assertTrue(self.planner.is_subregion_of("Safe Haven", "Altgard"))

    def test_coordinate_projection_hd_fit_and_inv_hd_fit(self) -> None:
        """apply_hd_fit and inv_hd_fit maintain sub-pixel invertible transformations."""
        test_points = [(100.0, 150.0), (640.0, 360.0), (1200.0, 680.0)]
        for vx, vy in test_points:
            hx, hy = apply_hd_fit(vx, vy)
            # Invert back to screen video frame space
            ox, oy = inv_hd_fit(hx, hy)
            self.assertAlmostEqual(ox, vx, delta=0.01)
            self.assertAlmostEqual(oy, vy, delta=0.01)

    def test_radar_close_range_clearance_preserves_near_sprouts(self) -> None:
        """_GATHER_PLAYER_CLEARANCE = 0.075 preserves near-player sprouts from blind zone filtering."""
        self.assertEqual(_GATHER_PLAYER_CLEARANCE, 0.075)

        px, py = MinimapScanner.PLAYER_CENTER_REL
        near_x = px + 0.04
        near_y = py + 0.04  # Distance ~ 0.056 <= 0.075 clearance

        # High-confidence sprout inside clearance zone MUST be recordable
        mat_high = {
            "is_gatherable": True,
            "kind": "sprout_material",
            "confidence": 0.85,
            "norm_x": near_x,
            "norm_y": near_y,
            "hud_chrome": False,
        }
        self.assertTrue(minimap_material_is_recordable(mat_high))
        self.assertFalse(minimap_norm_is_hud_chrome(near_x, near_y, confidence=0.85))

        # Low-confidence noise within clearance zone is suppressed
        mat_low = {
            "is_gatherable": True,
            "kind": "sprout_material",
            "confidence": 0.50,
            "norm_x": near_x,
            "norm_y": near_y,
            "hud_chrome": False,
        }
        self.assertFalse(minimap_material_is_recordable(mat_low))
        self.assertTrue(minimap_norm_is_hud_chrome(near_x, near_y, confidence=0.50))

    def test_minimap_hud_chrome_rejection(self) -> None:
        """Minimap outer HUD chrome (title bar, clock, magnifying glass) is rejected."""
        # Top title area (y < 0.18)
        self.assertTrue(minimap_norm_is_hud_chrome(0.50, 0.08, confidence=0.99))
        # Bottom-right clock area (x >= 0.62, y >= 0.74)
        self.assertTrue(minimap_norm_is_hud_chrome(0.70, 0.80, confidence=0.99))

    def test_sequential_route_planner_cyclical_wrap_around(self) -> None:
        """SequentialRoutePlanner executes clean cyclical progression from Node N back to Node 1."""
        nodes = [
            {"id": 1, "map_x": 100.0, "map_y": 100.0},
            {"id": 2, "map_x": 200.0, "map_y": 100.0},
            {"id": 3, "map_x": 200.0, "map_y": 200.0},
        ]

        next_after_1 = self.planner.cyclical_next(nodes[0], nodes)
        next_after_2 = self.planner.cyclical_next(nodes[1], nodes)
        next_after_3 = self.planner.cyclical_next(nodes[2], nodes)

        self.assertEqual(next_after_1["id"], 2)
        self.assertEqual(next_after_2["id"], 3)
        self.assertEqual(next_after_3["id"], 1)  # Clean cyclical wrap-around

    def test_radar_projection_of_3d_and_minimap_discoveries(self) -> None:
        """Both 3D perspective and minimap detections translate accurately to world coordinates."""
        ac = AutoCartographer(initial_map_x=500.0, initial_map_y=300.0)

        # 3D screen detection with dx=25.0, dy=-15.0
        coords = ac.offset_to_world_coords(dx=25.0, dy=-15.0)
        wx, wy = coords[0], coords[1]
        self.assertAlmostEqual(wx, 525.0)
        self.assertAlmostEqual(wy, 285.0)


# =============================================================================
# Suite 5: Wrong Node Purge with Negative Exemplar Memory (E4)
# =============================================================================

class TestWrongNodePurgeAndNegativeExemplarMemory(unittest.TestCase):
    """Verifies quality gate validation, wrong node purge, and active learning memory."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="test_auto_explore_e4_")
        self.route_mgr = RouteManager(base_dir=self.tmp_dir, active_region="Altgard")
        self.route_mgr.create_region("Altgard")
        self.ex_path = os.path.join(self.tmp_dir, "learned_exemplars.json")
        self.mem = get_exemplar_memory(json_path=self.ex_path)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_evaluate_node_candidate_accepts_valid_gatherable(self) -> None:
        """Genuine textured resource patch passes candidate evaluation."""
        aria_patch = _create_synthetic_aria_patch(80)
        verdict = evaluate_node_candidate(
            aria_patch,
            source_type="3d_auto",
            resource_type="Aria",
            check_exemplars=False,
            check_terrain=False,
        )
        self.assertTrue(verdict.accept)
        self.assertFalse(verdict.reject)
        self.assertEqual(len(verdict.reasons), 0)

    def test_evaluate_node_candidate_rejects_hard_wrong_classes(self) -> None:
        """Hard-wrong class (e.g. monster class 10) triggers immediate gate rejection."""
        patch = np.full((80, 80, 3), 90, dtype=np.uint8)

        class MockMonsterDetector:
            num_classes = 16
            def top_class(self, _img):
                return 10, 0.88  # Class 10 = monster

        verdict = evaluate_node_candidate(
            patch,
            source_type="minimap_auto",
            resource_type="Gatherable Material",
            detector=MockMonsterDetector(),
            conf_min=0.13,
            check_feather=False,
            check_monster=False,
            check_exemplars=False,
            check_terrain=False,
        )
        self.assertFalse(verdict.accept)
        self.assertTrue(verdict.reject)
        self.assertIn("model", verdict.reasons)
        self.assertEqual(verdict.class_id, 10)

    def test_evaluate_node_candidate_rejects_feather_and_monster_filters(self) -> None:
        """Feather icons and monster circularities trigger rejection in evaluate_node_candidate."""
        img_patch = np.zeros((80, 80, 3), dtype=np.uint8)

        # 1. When feather filter triggers
        with patch("core.node_quality_gate.patch_is_feather", return_value=True):
            v_feather = evaluate_node_candidate(
                img_patch,
                source_type="minimap_auto",
                resource_type="Gatherable Material",
                check_feather=True,
                check_monster=False,
                check_exemplars=False,
                check_terrain=False,
            )
        self.assertFalse(v_feather.accept)
        self.assertTrue(v_feather.reject)
        self.assertIn("feather", v_feather.reasons)

        # 2. When monster dot filter triggers
        with patch("core.node_quality_gate.patch_is_monster_dot", return_value=True):
            v_monster = evaluate_node_candidate(
                img_patch,
                source_type="minimap_auto",
                resource_type="Gatherable Material",
                check_feather=False,
                check_monster=True,
                check_exemplars=False,
                check_terrain=False,
            )
        self.assertFalse(v_monster.accept)
        self.assertTrue(v_monster.reject)
        self.assertIn("monster", v_monster.reasons)

    def test_is_fake_gray_rock_or_ground_detection(self) -> None:
        """Flat barren ground texture triggers fake terrain detection."""
        # 1. Default exemplar memory check
        img = np.full((80, 80, 3), 90, dtype=np.uint8)
        is_fake_def, conf_def, reason_def = is_fake_gray_rock_or_ground(img)
        self.assertTrue(is_fake_def)
        self.assertGreater(conf_def, 0.50)

        # 2. Custom exemplar memory check with learned negative
        rock_patch = np.zeros((80, 80, 3), dtype=np.uint8)
        rock_patch[:, :] = (100, 105, 110)
        self.mem.add_negative_exemplar(rock_patch, class_id=14, class_name="gray_rock_candidate")

        is_fake_mem, conf_mem, reason_mem = is_fake_gray_rock_or_ground(rock_patch, exemplar_memory=self.mem)
        self.assertTrue(is_fake_mem)
        self.assertIn("learned_negative", reason_mem)

    def test_learned_negative_exemplar_rejection(self) -> None:
        """Purged negative exemplar instantly suppresses matching future candidates."""
        erroneous_patch = np.zeros((80, 80, 3), dtype=np.uint8)
        erroneous_patch[:, :] = (80, 85, 90)

        # Register as negative exemplar
        self.mem.add_negative_exemplar(erroneous_patch, class_id=15, class_name="false_dirt_patch")
        self.assertEqual(len(self.mem.get_negative_exemplars()), 1)

        verdict = evaluate_node_candidate(
            erroneous_patch,
            source_type="3d_auto",
            resource_type="Raw_Node",
            exemplar_memory=self.mem,
            check_exemplars=True,
            check_terrain=True,
        )
        self.assertFalse(verdict.accept)
        self.assertTrue(verdict.reject)
        self.assertIn("negative_exemplar", verdict.reasons)

    def test_classify_wrong_nodes_detects_all_failing_candidates(self) -> None:
        """classify_wrong_nodes flags invalid nodes with accurate failure reasons."""
        fake_patch = np.full((80, 80, 3), 110, dtype=np.uint8)
        aria_patch = _create_synthetic_aria_patch(80)
        nodes = [
            {"id": 1, "map_x": 100.0, "map_y": 100.0, "resource_type": "Aria"},
            {"id": 2, "map_x": 200.0, "map_y": 200.0, "resource_type": "Fake_Rock"},
        ]

        class MockDet:
            num_classes = 16
            def top_class(self, img):
                if np.array_equal(img, fake_patch):
                    return 10, 0.85  # Node 2 is monster class 10 (hard wrong)
                return 0, 0.90       # Node 1 is Aria class 0 (keep class)

        wrong = classify_wrong_nodes(
            nodes=nodes,
            detector=MockDet(),
            images_by_id={1: aria_patch, 2: fake_patch},
            conf_min=0.13,
            exemplar_memory=self.mem,
        )
        self.assertEqual(len(wrong), 1)
        self.assertEqual(wrong[0].node_id, 2)
        self.assertEqual(wrong[0].primary_reason, "model")

    def test_save_wrong_nodes_as_negative_exemplars_active_learning(self) -> None:
        """save_wrong_nodes_as_negative_exemplars records purged thumbnails into ExemplarMemory."""
        fake_patch = np.full((80, 80, 3), 115, dtype=np.uint8)
        wrong = [
            WrongNode(node_id=5, reasons=["model:10"], class_id=10, confidence=0.88),
        ]

        saved_count = save_wrong_nodes_as_negative_exemplars(
            wrong=wrong,
            images_by_id={5: fake_patch},
            exemplar_memory=self.mem,
        )
        self.assertEqual(saved_count, 1)
        self.assertEqual(len(self.mem.get_negative_exemplars()), 1)

    def test_purge_wrong_nodes_with_automatic_safety_backup(self) -> None:
        """When purge exceeds 50% threshold, automatic backup region_data.backup.json is made."""
        self.route_mgr.add_node(region="Altgard", map_x=100, map_y=100, resource_type="Wrong1")
        self.route_mgr.add_node(region="Altgard", map_x=200, map_y=200, resource_type="Wrong2")

        fake_patch = np.full((80, 80, 3), 105, dtype=np.uint8)
        wrong = [
            WrongNode(node_id=1, reasons=["model:10"], class_id=10, confidence=0.9),
            WrongNode(node_id=2, reasons=["model:10"], class_id=10, confidence=0.9),
        ]

        # 2 out of 2 nodes = 100% > 50% threshold -> backup must be created
        ok, msg, count = purge_wrong_nodes(
            region="Altgard",
            wrong_nodes=wrong,
            route_manager=self.route_mgr,
            images_by_id={1: fake_patch, 2: fake_patch},
            exemplar_memory=self.mem,
            base_dir=self.tmp_dir,
            auto_backup=True,
        )

        self.assertTrue(ok)
        self.assertEqual(count, 2)
        backup_file = os.path.join(self.tmp_dir, "Altgard", "region_data.backup.json")
        self.assertTrue(os.path.exists(backup_file))

    def test_purge_wrong_nodes_deletes_from_route_manager(self) -> None:
        """purge_wrong_nodes purges wrong nodes and leaves valid nodes intact."""
        n1 = self.route_mgr.add_node(region="Altgard", map_x=100, map_y=100, resource_type="Aria")
        n2 = self.route_mgr.add_node(region="Altgard", map_x=200, map_y=200, resource_type="Fake_Rock")

        wrong = [WrongNode(node_id=n2.id, reasons=["model:10"], class_id=10, confidence=0.9)]

        ok, msg, count = purge_wrong_nodes(
            region="Altgard",
            wrong_nodes=wrong,
            route_manager=self.route_mgr,
            images_by_id={n2.id: np.full((80, 80, 3), 100, dtype=np.uint8)},
            exemplar_memory=self.mem,
            base_dir=self.tmp_dir,
        )

        self.assertTrue(ok)
        self.assertEqual(count, 1)

        remaining = self.route_mgr.get_nodes("Altgard")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0].id, n1.id)
        self.assertEqual(remaining[0].resource_type, "Aria")


# =============================================================================
# Suite 6: Full 5-Agent End-to-End Autonomous Exploration Integration (E1..E5)
# =============================================================================

class Test5AgentAutoExploreEndToEndIntegration(unittest.TestCase):
    """End-to-end integration test validating the cooperation of all 5 exploration agents."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp(prefix="test_auto_explore_e5_e2e_")
        self.route_mgr = RouteManager(base_dir=self.tmp_dir, active_region="Altgard")
        self.route_mgr.create_region("Altgard")
        self.ex_path = os.path.join(self.tmp_dir, "learned_exemplars.json")
        self.mem = get_exemplar_memory(json_path=self.ex_path)

        self.cartographer = AutoCartographer(
            route_manager=self.route_mgr,
            initial_map_x=500.0,
            initial_map_y=300.0,
            dedup_radius_px=22.0,
        )
        self.scanner = MaterialScanner()
        self.planner = SequentialRoutePlanner()

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_e2e_autonomous_explore_discover_classify_cluster_and_persist_loop(self) -> None:
        """Comprehensive multi-step tour proving autonomous discovery, classification,
        deduplication, quality gating, negative feedback, and playable route construction."""

        # ---------------------------------------------------------------------
        # Step 1: Walking & Visual Odometry Tracking (E1)
        # ---------------------------------------------------------------------
        t0 = time.time()
        self.cartographer.odometry_tracker.append_trajectory(timestamp=t0)
        self.cartographer.odometry_tracker.update_displacement(dx=25.0, dy=15.0)
        self.cartographer.odometry_tracker.append_trajectory(timestamp=t0 + 1.0)
        self.assertEqual(len(self.cartographer.get_walk_trajectory()), 2)

        # ---------------------------------------------------------------------
        # Step 2: 3D Discovery & Classification of Aria Herb (E1 + E2)
        # ---------------------------------------------------------------------
        aria_patch = _create_synthetic_aria_patch(80)
        aria_class = classify_3d_material_patch(aria_patch)
        self.assertEqual(aria_class["name"], "Aria")

        frame1 = np.zeros((720, 1280, 3), dtype=np.uint8)
        det_aria = {
            "rel_map_dx": 10.0,
            "rel_map_dy": 10.0,
            "resource_type": aria_class["name"],
            "confidence": aria_class["confidence"],
            "source_type": "3d_auto",
        }
        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res_aria = self.cartographer.record_from_3d_world(
                frame1, detections=[det_aria], route_manager=self.route_mgr, region="Altgard"
            )
        self.assertEqual(len(res_aria), 1)
        self.assertEqual(res_aria[0]["id"], 1)
        self.assertEqual(res_aria[0]["resource_type"], "Aria")

        # ---------------------------------------------------------------------
        # Step 3: 15-25px Spatial Clustering & Coordinate Refinement (E1/E4)
        # ---------------------------------------------------------------------
        # Next frame sees Aria again at offset (13, 13) -> distance ~ 4.2px <= 22px
        det_aria_repeat = {
            "rel_map_dx": 13.0,
            "rel_map_dy": 13.0,
            "resource_type": "Aria",
            "confidence": 0.95,
            "source_type": "3d_auto",
        }
        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res_repeat = self.cartographer.record_from_3d_world(
                frame1, detections=[det_aria_repeat], route_manager=self.route_mgr, region="Altgard"
            )
        self.assertEqual(len(res_repeat), 0)  # Deduplicated!
        tracked_aria = self.cartographer.get_tracked_nodes("Altgard")[0]
        self.assertEqual(tracked_aria["sightings_count"], 2)

        # ---------------------------------------------------------------------
        # Step 4: Minimap Sprout Discovery & Radar Clearance Projection (E3)
        # ---------------------------------------------------------------------
        sprout_det = {
            "rel_map_dx": 60.0,
            "rel_map_dy": 50.0,
            "resource_type": "Gatherable Material",
            "confidence": 0.88,
            "source_type": "minimap_auto",
            "stack_count": 2,  # Multi-stack sprout!
        }
        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res_sprout = self.cartographer.record_from_minimap(
                frame1, detections=[sprout_det], route_manager=self.route_mgr, region="Altgard", update_odometry=False
            )
        self.assertEqual(len(res_sprout), 1)
        self.assertEqual(res_sprout[0]["id"], 2)
        self.assertEqual(res_sprout[0]["stack_count"], 2)
        self.assertEqual(res_sprout[0]["max_harvests"], 6)

        # ---------------------------------------------------------------------
        # Step 5: Filter Rejection of Non-Materials (Monster & Feathers) (E3/E4)
        # ---------------------------------------------------------------------
        self.assertTrue(is_hard_wrong_class(10))  # Monster 3D
        self.assertTrue(is_hard_wrong_class(11))  # Monster text/dot
        self.assertTrue(is_rejected_non_material("feather"))
        self.assertTrue(is_rejected_non_material("monster"))
        with patch("core.node_quality_gate.patch_is_monster_dot", return_value=True):
            v_monster = evaluate_node_candidate(
                np.zeros((80, 80, 3), dtype=np.uint8),
                source_type="minimap_auto",
                resource_type="Monster_Dot",
                check_monster=True,
                check_feather=False,
                check_exemplars=False,
                check_terrain=False,
            )
        self.assertTrue(v_monster.reject)
        self.assertIn("monster", v_monster.reasons)

        # ---------------------------------------------------------------------
        # Step 6: 3D Discovery & Classification of Orichalcum Ore (E2)
        # ---------------------------------------------------------------------
        ore_patch = _create_synthetic_ore_patch(80)
        ore_class = classify_3d_material_patch(ore_patch)
        self.assertEqual(ore_class["name"], "Orichalcum Ore")

        det_ore = {
            "rel_map_dx": 120.0,
            "rel_map_dy": 90.0,
            "resource_type": ore_class["name"],
            "confidence": ore_class["confidence"],
            "source_type": "3d_auto",
        }
        with patch("core.auto_cartographer.evaluate_node_candidate", return_value=GateVerdict(accept=True)):
            res_ore = self.cartographer.record_from_3d_world(
                frame1, detections=[det_ore], route_manager=self.route_mgr, region="Altgard"
            )
        self.assertEqual(len(res_ore), 1)
        self.assertEqual(res_ore[0]["id"], 3)
        self.assertEqual(res_ore[0]["resource_type"], "Orichalcum Ore")

        # ---------------------------------------------------------------------
        # Step 7: Erroneous Rock Node Purged & Saved into Active Learning (E4)
        # ---------------------------------------------------------------------
        # Insert a faulty fake rock candidate
        fake_rock_node = self.route_mgr.add_node(region="Altgard", map_x=800, map_y=600, resource_type="Fake_Rock")
        fake_patch = _create_synthetic_flat_ground_patch(80)

        wrong = [WrongNode(node_id=fake_rock_node.id, reasons=["terrain:barren_stone"], class_id=15, confidence=0.85)]
        ok_purge, msg_purge, count_purge = purge_wrong_nodes(
            region="Altgard",
            wrong_nodes=wrong,
            route_manager=self.route_mgr,
            images_by_id={fake_rock_node.id: fake_patch},
            exemplar_memory=self.mem,
            base_dir=self.tmp_dir,
        )
        self.assertTrue(ok_purge)
        self.assertEqual(count_purge, 1)

        # Negative exemplar memory now stores the purged thumbnail
        self.assertGreaterEqual(len(self.mem.get_negative_exemplars()), 1)

        # ---------------------------------------------------------------------
        # Step 8: Playable Cyclic Sequential Route Verification (E1/E3)
        # ---------------------------------------------------------------------
        final_nodes = self.route_mgr.get_nodes("Altgard")
        self.assertEqual(len(final_nodes), 3)  # Nodes 1 (Aria), 2 (Sprout), 3 (Orichalcum)
        ids = [n.id for n in final_nodes]
        self.assertEqual(ids, [1, 2, 3])

        # Test route planning cyclicity
        next_after_3 = self.planner.cyclical_next(final_nodes[2].to_dict(), [n.to_dict() for n in final_nodes])
        self.assertEqual(next_after_3["id"], 1)

        # Region data file is fully persisted and readable
        region_file = os.path.join(self.tmp_dir, "Altgard", "region_data.json")
        self.assertTrue(os.path.exists(region_file))
        with open(region_file, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        self.assertEqual(len(persisted["nodes"]), 3)

    def test_e2e_active_learning_negative_memory_suppresses_subsequent_false_nodes(self) -> None:
        """Purging an erroneous patch guarantees subsequent sightings of that patch are dropped."""
        barren_patch = _create_synthetic_flat_ground_patch(80)

        # Step A: Register the wrong node and purge it
        wrong_node = self.route_mgr.add_node(region="Altgard", map_x=450, map_y=350, resource_type="Barren_Dirt")
        wrong = [WrongNode(node_id=wrong_node.id, reasons=["model:15"], class_id=15, confidence=0.90)]
        purge_wrong_nodes(
            region="Altgard",
            wrong_nodes=wrong,
            route_manager=self.route_mgr,
            images_by_id={wrong_node.id: barren_patch},
            exemplar_memory=self.mem,
            base_dir=self.tmp_dir,
        )

        # Step B: Quality Gate evaluates identical barren patch with ExemplarMemory active
        verdict = evaluate_node_candidate(
            barren_patch,
            source_type="3d_auto",
            resource_type="Unknown_3D",
            exemplar_memory=self.mem,
            check_exemplars=True,
            check_terrain=True,
        )

        self.assertFalse(verdict.accept)
        self.assertTrue(verdict.reject)
        self.assertIn("negative_exemplar", verdict.reasons)


if __name__ == "__main__":
    unittest.main()
