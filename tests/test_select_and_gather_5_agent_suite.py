# -*- coding: utf-8 -*-
"""
Gatherion-Local v2.0 — 5-Agent Comprehensive "Seç-Topla" Test Suite
(tests/test_select_and_gather_5_agent_suite.py)

Exclusive Owner: Agent S5
Focus Areas Verified Across 5 Cooperating Agents (S1..S5):
1. Material Node Filtering (Agent S1: core/route_manager.py & core/sequential_router.py):
   - Selecting "Orichalcum Ore" filters a mixed territory of nodes down to strictly Orichalcum nodes.
   - Multi-representation taxonomy matching across naming variants (ore_orichalcum, Orichalcum Ore, orichalcum, ORICHALCUM).
   - Additional Aion 2 material families (Aria/herb_aria, Mela/herb_mela, Targena/herb_targena, Gem/gem_crystal, Tree/tree_log).
   - Multi-material selection (["Orichalcum Ore", "Aria"]).
   - Empty/All target handling and exclusion of not_gatherable=True nodes.
   - Dual compatibility with NodeData objects and dictionary structures.

2. Proximity Optimization (Nearest-Neighbor + 2-Opt TSP) (Agent S1):
   - Starting from player coordinates (px, py), sort 5+ scattered nodes:
     - The first visited node is guaranteed to be the closest to (px, py).
     - 2-Opt local search swaps untangle crossed route edges and minimize total perimeter travel distance.
   - Robust edge cases: empty node list, 1 node, 2 nodes, start_xy=None.
   - Integration with RouteOptimizer in core/path_optimizer.py.

3. Route Construction & Dynamic Re-sequencing (Agent S1):
   - RouteManager.build_proximity_material_route creates and persists a valid route with sequential seq_index values (1..N).
   - Dynamic re-ordering updates the waypoint sequence when player position changes.
   - Full JSON persistence round-trip to routes/<region>/region_data.json.
   - SequentialRouter module-level exports and active route integration.

4. Selective 3D Viewport Harvesting in MaterialScanner & BotWorker (Agent S2):
   - MaterialScanner: set_selected_materials, get_selected_materials, has_selected_materials, is_material_selected.
   - Viewport candidate filtering: drops non-matching candidates and retains selected targets.
   - BotWorker: set_selected_gather_materials, selective_gather_enabled, synchronization with scanner.
   - Opportunistic gathering bypass: _should_bypass_unselected_gather and is_material_selected_for_gather.
   - Waypoint navigation filtering for selective gathering.

5. UI & Schema Verification (Agent S3 & Agent S4):
   - GatheringSettings schema fields: selective_gather_enabled, selected_materials, proximity_route_auto.
   - JSON serialization and from_dict/to_dict round-trip.
   - F9Menu selective_gather_changed signal and controls.
   - SettingsWindow selective_gather_changed signal and controls.
   - MapViewerWindow & _MapCanvas: set_material_filter, get_material_filter, calculate_proximity_route,
     numbered badges (#1, #2...) and directional arrows.
   - RadarCanvas: set_material_filter, get_material_filter.

6. Full 5-Agent End-to-End Integration Scenario (S1..S5):
   - Complete simulated tour connecting schema config, RouteManager 2-Opt proximity route generation,
     Map/Radar display, BotWorker & MaterialScanner live filtering, and dynamic player resume.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Ensure headless Qt operation
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

_qapp = QApplication.instance() or QApplication([])

# Core imports
import config.schema as schema_mod
from config.schema import GatheringSettings
import core.route_manager as rm_mod
from core.route_manager import (
    DEFAULT_MAX_HARVESTS,
    DEFAULT_REGION,
    DEFAULT_RESOURCE_TYPE,
    NodeData,
    RouteManager,
    filter_nodes_by_material,
    sort_nodes_by_proximity,
    build_proximity_material_route,
)
import core.sequential_router as sr_mod
from core.sequential_router import SequentialRoutePlanner
from core.path_optimizer import RouteOptimizer
import core.material_scanner as ms_mod
from core.material_scanner import DetectedMaterial, MaterialScanner
import core.state_machine as sm_mod
from core.state_machine import BotWorker
import ui.map_viewer as mv_mod
from ui.map_viewer import _MapCanvas, MapViewerWindow, node_matches_material
import ui.cartography_radar as cr_mod
from ui.cartography_radar import RadarCanvas, CartographyRadarDialog
import ui.f9_menu as f9_mod
from ui.f9_menu import F9Menu
import ui.settings_window as settings_mod
from ui.settings_window import SettingsWindow


# ==============================================================================
# TEST FIXTURES & HELPER UTILITIES
# ==============================================================================



def _create_mock_detected_material(
    mtype: str,
    sx: int = 200,
    sy: int = 300,
    conf: float = 0.92,
    dist: float = 8.5,
) -> DetectedMaterial:
    return DetectedMaterial(
        material_type=mtype,
        source="3d_world",
        screen_x=sx,
        screen_y=sy,
        bbox=(sx - 20, sy - 20, sx + 20, sy + 20),
        rel_map_dx=0.0,
        rel_map_dy=0.0,
        estimated_distance_m=dist,
        confidence=conf,
        in_air=False,
        stack_count=1,
        harvests_remaining=3,
        priority_score=0.88,
        rarity_tier=1,
    )


def _fast_bot_worker(**kwargs: Any) -> BotWorker:
    """Build a BotWorker without constructing unrelated heavy subsystems.

    The BotWorker tests in this file only exercise the selective-gather
    surface (selected-materials state + MaterialScanner sync). BotWorker
    __init__ also constructs ~40 optional subsystems; the slowest are
    ONNXYolo3DDetector (~4.0s) and AutoLabelingDatasetFactory (~2.1s),
    making one construction ~9.6s on this VM (measured 2026-11-08).
    Constructing them proves nothing these tests assert, so they are
    replaced with MagicMocks during construction. The ONE exception is
    MaterialScanner: it is the unit under test, so it is always built for
    real (unless the caller passes one in explicitly).
    """
    real_init = BotWorker._init_optional_subsystem

    def _patched_init(self: Any, module_name: str, class_name: str, *args: Any, **kwargs: Any) -> Any:
        if (module_name, class_name) != ("core.material_scanner", "MaterialScanner"):
            return MagicMock(name=f"stubbed.{module_name}.{class_name}")
        return real_init(self, module_name, class_name, *args, **kwargs)

    with patch.object(BotWorker, "_init_optional_subsystem", _patched_init):
        return BotWorker(**kwargs)


# ==============================================================================
# SUITE 1: MATERIAL NODE FILTERING (Agent S1)
# ==============================================================================

class TestMaterialNodeFilteringS1(unittest.TestCase):
    """
    Verifies material filtering logic across naming variants, canonical groups,
    mixed node populations, wildcards, and data formats.
    """

    def setUp(self) -> None:
        self.sample_nodes = [
            NodeData(id=1, map_x=100, map_y=100, resource_type="Orichalcum Ore", seq_index=1),
            NodeData(id=2, map_x=200, map_y=150, resource_type="Aria", seq_index=2),
            NodeData(id=3, map_x=300, map_y=200, resource_type="ore_orichalcum", seq_index=3),
            NodeData(id=4, map_x=400, map_y=250, resource_type="Mela", seq_index=4),
            NodeData(id=5, map_x=500, map_y=300, resource_type="Iron Ore", seq_index=5),
            NodeData(id=6, map_x=600, map_y=350, resource_type="Titanium Ore", seq_index=6),
            NodeData(id=7, map_x=700, map_y=400, resource_type="gem_crystal", seq_index=7),
            NodeData(id=8, map_x=800, map_y=450, resource_type="tree_log", seq_index=8),
            NodeData(id=9, map_x=900, map_y=500, resource_type="Orichalcum Ore", not_gatherable=True, seq_index=9),
        ]

    def test_filter_nodes_single_material_orichalcum(self) -> None:
        """Filtering by 'Orichalcum Ore' excludes herbs, iron, titanium, and returns only active Orichalcum."""
        filtered = filter_nodes_by_material(self.sample_nodes, "Orichalcum Ore")
        matched_ids = [n.id for n in filtered]
        self.assertIn(1, matched_ids)
        self.assertIn(3, matched_ids)
        self.assertNotIn(2, matched_ids)  # Aria
        self.assertNotIn(4, matched_ids)  # Mela
        self.assertNotIn(5, matched_ids)  # Iron
        self.assertNotIn(6, matched_ids)  # Titanium
        self.assertNotIn(7, matched_ids)  # Gem
        self.assertNotIn(8, matched_ids)  # Tree
        self.assertNotIn(9, matched_ids)  # not_gatherable
        self.assertEqual(len(filtered), 2)

    def test_filter_nodes_naming_variants_orichalcum(self) -> None:
        """Verifies naming variants: 'ore_orichalcum', 'Orichalcum Ore', 'orichalcum', 'ORICHALCUM ORE'."""
        variants = ["ore_orichalcum", "Orichalcum Ore", "orichalcum", "ORICHALCUM ORE", "  orichalcum  "]
        for var in variants:
            res = filter_nodes_by_material(self.sample_nodes, var)
            ids = [n.id for n in res]
            self.assertEqual(ids, [1, 3], f"Failed variant matching for '{var}'")

    def test_filter_nodes_aria_and_mela_variants(self) -> None:
        """Verifies taxonomy aliases for Aria ('herb_aria', 'Aria') and Mela ('herb_mela', 'Mela')."""
        aria_res = filter_nodes_by_material(self.sample_nodes, "Aria")
        self.assertEqual([n.id for n in aria_res], [2])

        aria_res_yolo = filter_nodes_by_material(self.sample_nodes, "herb_aria")
        self.assertEqual([n.id for n in aria_res_yolo], [2])

        mela_res = filter_nodes_by_material(self.sample_nodes, "Mela")
        self.assertEqual([n.id for n in mela_res], [4])

        mela_res_yolo = filter_nodes_by_material(self.sample_nodes, "herb_mela")
        self.assertEqual([n.id for n in mela_res_yolo], [4])

    def test_filter_nodes_gem_and_tree_variants(self) -> None:
        """Verifies taxonomy aliases for Crystal/Gem ('gem_crystal') and Tree/Log ('tree_log')."""
        gem_res = filter_nodes_by_material(self.sample_nodes, "gem_crystal")
        self.assertEqual([n.id for n in gem_res], [7])

        tree_res = filter_nodes_by_material(self.sample_nodes, "tree_log")
        self.assertEqual([n.id for n in tree_res], [8])

    def test_filter_nodes_multiple_target_materials(self) -> None:
        """Multi-selection ['Orichalcum Ore', 'Aria'] returns both resource classes and excludes others."""
        res = filter_nodes_by_material(self.sample_nodes, ["Orichalcum Ore", "Aria"])
        ids = [n.id for n in res]
        self.assertIn(1, ids)
        self.assertIn(2, ids)
        self.assertIn(3, ids)
        self.assertNotIn(4, ids)
        self.assertNotIn(5, ids)
        self.assertEqual(len(res), 3)

    def test_filter_nodes_all_or_empty_returns_all_gatherable(self) -> None:
        """Targets None, [], 'All', or 'Tümü' return all active gatherable nodes (8 out of 9)."""
        for empty_val in [None, [], "", "All", "Tümü", "tumu"]:
            res = filter_nodes_by_material(self.sample_nodes, empty_val)
            self.assertEqual(len(res), 8)
            self.assertNotIn(9, [n.id for n in res])  # Node 9 is not_gatherable

    def test_filter_nodes_not_gatherable_exclusion(self) -> None:
        """Nodes marked not_gatherable=True are strictly excluded even if their material matches."""
        node_bad = NodeData(id=99, map_x=10, map_y=10, resource_type="Orichalcum Ore", not_gatherable=True)
        res = filter_nodes_by_material([node_bad], "Orichalcum Ore")
        self.assertEqual(len(res), 0)

    def test_filter_nodes_dict_compatibility(self) -> None:
        """filter_nodes_by_material works seamlessly with raw dictionary node representations."""
        dict_nodes = [
            {"id": 101, "map_x": 50, "map_y": 50, "resource_type": "Orichalcum Ore"},
            {"id": 102, "map_x": 60, "map_y": 60, "resource_type": "Aria"},
            {"id": 103, "map_x": 70, "map_y": 70, "resource_type": "ore_orichalcum"},
        ]
        res = filter_nodes_by_material(dict_nodes, "Orichalcum Ore")
        self.assertEqual([d["id"] for d in res], [101, 103])

    def test_filter_nodes_empty_list_returns_empty(self) -> None:
        """Empty input list returns an empty list without errors."""
        self.assertEqual(filter_nodes_by_material([], "Orichalcum Ore"), [])


# ==============================================================================
# SUITE 2: PROXIMITY OPTIMIZATION (NN + 2-OPT TSP) (Agent S1)
# ==============================================================================

class TestProximityOptimizationNNAnd2OptS1(unittest.TestCase):
    """
    Verifies Greedy Nearest-Neighbor starting from player position followed by
    2-Opt local search TSP perimeter distance minimization and edge untangling.
    """

    def test_nearest_neighbor_closest_node_first_5_plus_nodes(self) -> None:
        """
        Given 6 scattered nodes across the map and player start at (110, 100):
        Node 4 at (112, 106) has Euclidean distance ~6.3px and must be visited FIRST.
        """
        nodes = [
            NodeData(id=1, map_x=100, map_y=100),
            NodeData(id=2, map_x=850, map_y=850),
            NodeData(id=3, map_x=450, map_y=450),
            NodeData(id=4, map_x=112, map_y=106),  # Closest to (110, 100)
            NodeData(id=5, map_x=300, map_y=300),
            NodeData(id=6, map_x=700, map_y=700),
        ]

        sorted_nodes = sort_nodes_by_proximity(nodes, start_xy=(110, 100))
        self.assertEqual(len(sorted_nodes), 6)
        self.assertEqual(sorted_nodes[0].id, 4, "Closest node to player must be first in sequence")
        self.assertEqual(sorted_nodes[1].id, 1, "Next closest node must follow")
        self.assertEqual(sorted_nodes[-1].id, 2, "Furthest node must be at end of sequence")

    def test_2opt_untangling_crossing_edges_reduces_distance(self) -> None:
        """
        Given 4 nodes at square vertices in hour-glass crossing order:
        N0(0,0), N1(10,10), N2(0,10), N3(10,0).
        Initial crossed distance = ~48.28px.
        2-Opt swap untangles crossed edges to produce optimal perimeter tour = 40.0px.
        """
        nodes = [
            NodeData(id=1, map_x=0, map_y=0),
            NodeData(id=2, map_x=10, map_y=10),
            NodeData(id=3, map_x=0, map_y=10),
            NodeData(id=4, map_x=10, map_y=0),
        ]

        ordered = sort_nodes_by_proximity(nodes, start_xy=(0, 0))
        self.assertEqual(len(ordered), 4)

        # Calculate path perimeter length
        coords = [(n.map_x, n.map_y) for n in ordered]
        total_dist = 0.0
        for i in range(len(coords)):
            p1 = coords[i]
            p2 = coords[(i + 1) % len(coords)]
            total_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        self.assertAlmostEqual(total_dist, 40.0, places=3, msg="2-Opt must eliminate crossing edges")

    def test_2opt_complex_scrambled_circuit_optimization(self) -> None:
        """
        Takes 8 circular nodes arranged in a scrambled zigzag sequence and verifies
        that 2-Opt local search produces an untangled, minimum-distance tour.
        """
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        radius = 200.0
        # Scrambled zigzag indices
        scrambled_indices = [0, 4, 1, 5, 2, 6, 3, 7]
        nodes = []
        for i, idx in enumerate(scrambled_indices):
            x = int(500 + radius * math.cos(angles[idx]))
            y = int(500 + radius * math.sin(angles[idx]))
            nodes.append(NodeData(id=i + 1, map_x=x, map_y=y))

        # Initial scrambled tour distance
        def _calc_dist(nl: List[NodeData]) -> float:
            d = 0.0
            for k in range(len(nl) - 1):
                d += math.hypot(nl[k + 1].map_x - nl[k].map_x, nl[k + 1].map_y - nl[k].map_y)
            return d

        initial_dist = _calc_dist(nodes)
        optimized = sort_nodes_by_proximity(nodes, start_xy=(nodes[0].map_x, nodes[0].map_y))
        optimized_dist = _calc_dist(optimized)

        self.assertLess(
            optimized_dist,
            initial_dist * 0.70,
            f"2-Opt must significantly shorten scrambled route: {optimized_dist} vs {initial_dist}",
        )

    def test_sort_nodes_edge_cases(self) -> None:
        """Handles empty list, 1 node, 2 nodes with/without start_xy, and start_xy=None."""
        self.assertEqual(sort_nodes_by_proximity([], start_xy=(0, 0)), [])

        single = [NodeData(id=1, map_x=50, map_y=50)]
        self.assertEqual(sort_nodes_by_proximity(single, start_xy=(0, 0)), single)

        two_nodes = [
            NodeData(id=1, map_x=100, map_y=100),
            NodeData(id=2, map_x=20, map_y=20),
        ]
        # Start closer to Node 2
        res2 = sort_nodes_by_proximity(two_nodes, start_xy=(25, 25))
        self.assertEqual([n.id for n in res2], [2, 1])

        # Start closer to Node 1
        res1 = sort_nodes_by_proximity(two_nodes, start_xy=(95, 95))
        self.assertEqual([n.id for n in res1], [1, 2])

        # None start_xy preserves first element (and input order)
        res_none = sort_nodes_by_proximity(two_nodes, start_xy=None)
        self.assertEqual(len(res_none), 2)
        self.assertIs(res_none[0], two_nodes[0])
        self.assertEqual([n.id for n in res_none], [1, 2])

    def test_sort_nodes_dict_compatibility(self) -> None:
        """sort_nodes_by_proximity sorts dictionaries containing map_x and map_y."""
        dict_nodes = [
            {"id": 1, "map_x": 500, "map_y": 500},
            {"id": 2, "map_x": 105, "map_y": 105},
            {"id": 3, "map_x": 300, "map_y": 300},
        ]
        res = sort_nodes_by_proximity(dict_nodes, start_xy=(100, 100))
        self.assertEqual([d["id"] for d in res], [2, 3, 1])


# ==============================================================================
# SUITE 3: ROUTE CONSTRUCTION & DYNAMIC RESEQUENCING (Agent S1)
# ==============================================================================

class TestRouteConstructionAndDynamicResequencingS1(unittest.TestCase):
    """
    Verifies RouteManager.build_proximity_material_route, 1-based sequential
    seq_index assignment, persistence to region_data.json, and dynamic re-ordering.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_prox_route_s1_")
        self.rm = RouteManager(base_dir=self.temp_dir, active_region="Altgard")
        self.rm.create_region("Altgard")

        # Populate region with 10 mixed resource nodes
        # 4 Orichalcum, 3 Aria, 3 Mela
        coords_and_mats = [
            (100, 100, "Orichalcum Ore"),
            (150, 120, "Aria"),
            (500, 500, "Orichalcum Ore"),
            (250, 200, "Mela"),
            (300, 300, "ore_orichalcum"),
            (350, 280, "Aria"),
            (700, 700, "Orichalcum Ore"),
            (400, 380, "Mela"),
            (450, 420, "Aria"),
            (600, 580, "Mela"),
        ]
        for idx, (x, y, mtype) in enumerate(coords_and_mats, start=1):
            self.rm.add_node(
                region="Altgard",
                node_id=idx,
                map_x=x,
                map_y=y,
                resource_type=mtype,
                max_harvests=3,
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_proximity_material_route_creation(self) -> None:
        """
        Filters Altgard for 'Orichalcum Ore', sorts from player (105, 100).
        Merge-fix (O8): D25/F-1647 — matched nodes take the matched slots
        in proximity order and unmatched nodes keep their place, so the
        route's seq_index values are the matched slots ([1, 3, 5, 7]),
        not a renumbered 1..4.
        """
        route = self.rm.build_proximity_material_route(
            region_name="Altgard",
            target_materials="Orichalcum Ore",
            start_xy=(105, 100),
            persist_route=True,
        )

        self.assertEqual(len(route), 4)
        for n in route:
            self.assertIn("orichalcum", n.resource_type.lower())

        # seq_index follows the matched slots in proximity order (D25/F-1647)
        seq_indices = [n.seq_index for n in route]
        self.assertEqual(seq_indices, [1, 3, 5, 7])

        # First node must be Node 1 at (100, 100), closest to (105, 100)
        self.assertEqual(route[0].id, 1)

        # Route must be registered in region's routes
        reg_routes = self.rm.get_routes("Altgard")
        self.assertTrue(
            any("proximity" in r and "orichalcum" in r for r in reg_routes),
            f"Expected proximity route in {reg_routes}",
        )

    def test_dynamic_resequencing_when_player_moves(self) -> None:
        """
        When player moves to (680, 710) near Node 7, re-calling
        build_proximity_material_route makes Node 7 seq_index=1.
        """
        # First call from south-west
        route_sw = self.rm.build_proximity_material_route(
            region_name="Altgard",
            target_materials="Orichalcum Ore",
            start_xy=(105, 100),
            persist_route=False,
        )
        self.assertEqual(route_sw[0].id, 1)

        # Player moves to north-east near Node 7 at (700, 700)
        route_ne = self.rm.build_proximity_material_route(
            region_name="Altgard",
            target_materials="Orichalcum Ore",
            start_xy=(680, 710),
            persist_route=False,
        )
        self.assertEqual(route_ne[0].id, 7, "Closest node to new player pos must become #1")
        self.assertEqual([n.seq_index for n in route_ne], [1, 2, 3, 4])

    def test_route_persistence_roundtrip_save_and_reload(self) -> None:
        """
        Persisted proximity route is written to region_data.json on disk
        and reloaded faithfully by a fresh RouteManager instance.
        """
        route = self.rm.build_proximity_material_route(
            region_name="Altgard",
            target_materials="Orichalcum Ore",
            start_xy=(100, 100),
            persist_route=True,
        )
        self.assertEqual(len(route), 4)

        # Load fresh instance from the same disk folder
        fresh_rm = RouteManager(base_dir=self.temp_dir, active_region="Altgard")
        fresh_rm.load_region("Altgard")

        reg_routes = fresh_rm.get_routes("Altgard")
        prox_route_name = next((r for r in reg_routes if "proximity" in r and "orichalcum" in r), None)
        self.assertIsNotNone(prox_route_name)

        saved_nodes = fresh_rm.get_route_nodes("Altgard", route_name=prox_route_name)
        self.assertEqual(len(saved_nodes), 4)
        for n in saved_nodes:
            self.assertIn("orichalcum", n.resource_type.lower())

    def test_sequential_router_module_exports_proximity_functions(self) -> None:
        """core/sequential_router.py exports proximity helpers that really filter/sort/build routes.

        The module-level names must be the production route_manager functions
        (not stubs), and SequentialRoutePlanner must delegate to them with
        identical results on real nodes.
        """
        # Module-level exports are the production functions themselves.
        self.assertIs(sr_mod.filter_nodes_by_material, filter_nodes_by_material)
        self.assertIs(sr_mod.sort_nodes_by_proximity, sort_nodes_by_proximity)
        self.assertIs(sr_mod.build_proximity_material_route, build_proximity_material_route)

        planner = SequentialRoutePlanner()
        nodes = [
            NodeData(id=1, map_x=100, map_y=100, resource_type="Orichalcum Ore"),
            NodeData(id=2, map_x=200, map_y=200, resource_type="Aria"),
            NodeData(id=3, map_x=300, map_y=300, resource_type="ore_orichalcum"),
        ]
        # Planner.filter_nodes_by_material actually filters (same result as production).
        self.assertEqual(
            [n.id for n in planner.filter_nodes_by_material(nodes, "Orichalcum Ore")],
            [1, 3],
        )
        self.assertEqual(
            [n.id for n in planner.filter_nodes_by_material(nodes, "Orichalcum Ore")],
            [n.id for n in filter_nodes_by_material(nodes, "Orichalcum Ore")],
        )
        # Planner.sort_nodes_by_proximity actually sorts closest-first.
        ordered = planner.sort_nodes_by_proximity(nodes, start_xy=(105, 100))
        self.assertEqual([n.id for n in ordered], [1, 2, 3])
        # Planner.build_proximity_material_route builds a real persisted route
        # through the RouteManager (Altgard nodes from setUp).
        route = planner.build_proximity_material_route(
            region_name="Altgard",
            target_materials="Orichalcum Ore",
            start_xy=(105, 100),
            persist_route=False,
            route_manager=self.rm,
        )
        self.assertEqual(len(route), 4)
        self.assertEqual(route[0].id, 1)
        self.assertEqual([n.seq_index for n in route], [1, 2, 3, 4])


# ==============================================================================
# SUITE 4: SELECTIVE 3D VIEWPORT HARVESTING (Agent S2)
# ==============================================================================

class TestSelective3DViewportHarvestingS2(unittest.TestCase):
    """
    Verifies selective gathering candidate filtering in MaterialScanner,
    synchronization with BotWorker, and opportunistic harvesting bypass.
    """

    def test_material_scanner_set_and_get_selected_materials(self) -> None:
        """MaterialScanner configures and queries selected materials."""
        ms = MaterialScanner()
        self.assertEqual(ms.get_selected_materials(), [])
        self.assertFalse(ms.has_selected_materials())

        ms.set_selected_materials(["Orichalcum Ore", "ore_orichalcum"])
        self.assertTrue(ms.has_selected_materials())
        mats = ms.get_selected_materials()
        self.assertIn("Orichalcum Ore", mats)

        # Matching tests
        self.assertTrue(ms.is_material_selected("Orichalcum Ore"))
        self.assertTrue(ms.is_material_selected("ore_orichalcum"))
        self.assertFalse(ms.is_material_selected("Aria"))
        self.assertFalse(ms.is_material_selected("herb_mela"))

        # Clear selection
        ms.set_selected_materials([])
        self.assertFalse(ms.has_selected_materials())
        self.assertEqual(ms.get_selected_materials(), [])

    def test_material_scanner_candidate_selective_filtering(self) -> None:
        """
        When selective gather is active, the production gather pipeline drops
        non-matching candidates and retains selected targets.

        Filtering is driven through BotWorker.is_material_selected_for_gather —
        the production gate _check_nearby_materials uses to decide whether a
        candidate is harvested or bypassed — not through a hand-rolled
        list comprehension in the test.
        """
        ms = MaterialScanner()
        bot = _fast_bot_worker(
            config={},
            vision=MagicMock(),
            input_driver=MagicMock(),
            route_manager=MagicMock(),
            combat_recovery=MagicMock(),
            navigator=MagicMock(),
            material_scanner=ms,
        )
        bot.set_selected_gather_materials(["Orichalcum Ore"])
        self.assertEqual(ms.get_selected_materials(), ["Orichalcum Ore"])

        d_ori1 = _create_mock_detected_material("Orichalcum Ore", 100, 100)
        d_ori2 = _create_mock_detected_material("ore_orichalcum", 150, 150)
        d_aria = _create_mock_detected_material("Aria", 200, 200)
        d_mela = _create_mock_detected_material("Mela", 250, 250)

        all_candidates = [d_ori1, d_aria, d_ori2, d_mela]
        # Production gate: only selected materials pass.
        filtered = [c for c in all_candidates if bot.is_material_selected_for_gather(c)]
        self.assertEqual(len(filtered), 2)
        for c in filtered:
            self.assertIn("orichalcum", c.material_type.lower())

        # The same gate rejects each non-target individually (incl. alias form).
        self.assertTrue(bot.is_material_selected_for_gather(d_ori2))
        self.assertFalse(bot.is_material_selected_for_gather(d_aria))
        self.assertFalse(bot.is_material_selected_for_gather(d_mela))

        # Prioritization itself does not filter: all candidates are still scored.
        prioritized = ms.prioritize_candidates(list(all_candidates))
        self.assertEqual(len(prioritized), 4)

        # With selective gathering off, the gate lets every candidate through.
        bot.set_selected_gather_materials([], enabled=False)
        self.assertEqual(
            [c for c in all_candidates if bot.is_material_selected_for_gather(c)],
            all_candidates,
        )

    def test_bot_worker_set_selected_gather_materials_sync(self) -> None:
        """BotWorker.set_selected_gather_materials sets state and synchronizes with MaterialScanner."""
        ms = MaterialScanner()
        bot = _fast_bot_worker(
            config={},
            vision=MagicMock(),
            input_driver=MagicMock(),
            route_manager=MagicMock(),
            combat_recovery=MagicMock(),
            navigator=MagicMock(),
            material_scanner=ms,
        )

        self.assertFalse(bot.selective_gather_enabled)
        bot.set_selected_gather_materials(["Orichalcum Ore"])

        self.assertTrue(bot.selective_gather_enabled)
        self.assertEqual(bot.get_selected_gather_materials(), ["Orichalcum Ore"])
        self.assertEqual(ms.get_selected_materials(), ["Orichalcum Ore"])

        # Disabling clears scanner selection
        bot.set_selected_gather_materials([], enabled=False)
        self.assertFalse(bot.selective_gather_enabled)
        self.assertEqual(ms.get_selected_materials(), [])

    def test_bot_worker_opportunistic_harvest_bypass_unselected(self) -> None:
        """
        BotWorker._should_bypass_unselected_gather returns True for non-target
        nodes and False for selected targets.
        """
        bot = _fast_bot_worker(
            config={},
            vision=MagicMock(),
            input_driver=MagicMock(),
            route_manager=MagicMock(),
            combat_recovery=MagicMock(),
            navigator=MagicMock(),
        )
        bot.set_selected_gather_materials(["Orichalcum Ore"])

        # 1. Non-target Aria node -> MUST BYPASS
        bot.active_node = NodeData(id=1, resource_type="Aria")
        self.assertTrue(bot._should_bypass_unselected_gather(), "Must bypass unselected Aria node")
        self.assertFalse(bot.is_material_selected_for_gather(bot.active_node))

        # 2. Target Orichalcum node -> MUST NOT BYPASS
        bot.active_node = NodeData(id=2, resource_type="Orichalcum Ore")
        self.assertFalse(bot._should_bypass_unselected_gather(), "Must not bypass target Orichalcum node")
        self.assertTrue(bot.is_material_selected_for_gather(bot.active_node))

        # 3. Inactive selective gather -> NEVER BYPASS
        bot.selective_gather_enabled = False
        bot.active_node = NodeData(id=3, resource_type="Aria")
        self.assertFalse(bot._should_bypass_unselected_gather())
        self.assertTrue(bot.is_material_selected_for_gather(bot.active_node))


# ==============================================================================
# SUITE 5: UI & SCHEMA INTEGRATION (Agent S3 & Agent S4)
# ==============================================================================

class TestUIAndSchemaIntegrationS3AndS4(unittest.TestCase):
    """
    Verifies GatheringSettings schema fields, F9Menu & SettingsWindow signal controls,
    and MapViewerWindow / CartographyRadar proximity visualization.
    """

    def test_gathering_settings_schema_fields_and_defaults(self) -> None:
        """GatheringSettings defines selective_gather_enabled, selected_materials, and proximity_route_auto."""
        cfg = GatheringSettings()
        self.assertFalse(cfg.selective_gather_enabled)
        self.assertEqual(cfg.selected_materials, [])
        self.assertTrue(cfg.proximity_route_auto)

    def test_gathering_settings_json_serialization_roundtrip(self) -> None:
        """GatheringSettings survives a full JSON round-trip preserving all values.

        GatheringSettings has no to_dict of its own, so serialization goes
        through the stdlib dataclass mapping (dataclasses.asdict) + json —
        the same shape from_dict consumes — in both directions, not only
        from_dict.
        """
        cfg = GatheringSettings(
            selective_gather_enabled=True,
            selected_materials=["Orichalcum Ore", "Aria"],
            proximity_route_auto=False,
            f_interval_min=2.0,
            harvests_per_node=2,
        )
        # to-dict direction: serialize to JSON text and back.
        payload = json.loads(json.dumps(dataclasses.asdict(cfg)))
        self.assertTrue(payload["selective_gather_enabled"])
        self.assertEqual(payload["selected_materials"], ["Orichalcum Ore", "Aria"])
        self.assertFalse(payload["proximity_route_auto"])
        self.assertEqual(payload["harvests_per_node"], 2)

        # from-dict direction: the parsed payload rebuilds an equal config.
        restored = GatheringSettings.from_dict(payload)
        self.assertEqual(restored, cfg)
        self.assertTrue(restored.selective_gather_enabled)
        self.assertEqual(restored.selected_materials, ["Orichalcum Ore", "Aria"])
        self.assertFalse(restored.proximity_route_auto)
        self.assertEqual(restored.harvests_per_node, 2)
        self.assertEqual(restored.f_interval_min, 2.0)

        # A second round-trip is stable (idempotent serialization).
        self.assertEqual(
            GatheringSettings.from_dict(
                json.loads(json.dumps(dataclasses.asdict(restored)))
            ),
            cfg,
        )

        # Partial input still round-trips through defaults.
        partial = GatheringSettings.from_dict({"selected_materials": ["Mela"]})
        self.assertEqual(partial.selected_materials, ["Mela"])
        self.assertFalse(partial.selective_gather_enabled)
        self.assertTrue(partial.proximity_route_auto)

    def test_f9_menu_selective_gather_signals(self) -> None:
        """F9Menu selective-gather controls really emit selective_gather_changed.

        Selecting a material via the combo fires the signal with the chosen
        materials, and the proximity button fires it with auto-proximity on
        plus the build_proximity_route action — not just hasattr checks.
        """
        menu = F9Menu()
        emitted: List[Tuple[List[str], bool]] = []
        menu.selective_gather_changed.connect(
            lambda mats, auto: emitted.append((list(mats), bool(auto)))
        )
        actions: List[str] = []
        menu.action_triggered.connect(actions.append)

        self.assertGreater(menu.combo_selective_materials.count(), 0)
        menu.set_selected_materials(["Orichalcum Ore"])
        self.assertEqual(menu.get_selected_materials(), ["Orichalcum Ore"])
        self.assertTrue(menu.chk_selective_gather.isChecked())

        # Choosing the material in the combo emits the signal with it.
        menu._on_selective_material_changed(menu.combo_selective_materials.currentIndex())
        self.assertTrue(emitted, "combo selection must emit selective_gather_changed")
        self.assertIn("Orichalcum Ore", emitted[-1][0])
        self.assertTrue(emitted[-1][1])
        self.assertTrue(menu._selective_gather_enabled)

        # The proximity button emits the signal and requests a route build.
        menu._on_proximity_route_button_clicked()
        self.assertIn("Orichalcum Ore", emitted[-1][0])
        self.assertTrue(emitted[-1][1])
        self.assertTrue(menu._proximity_route_auto)
        # Merge-fix (O8): D33/F-0731 removed the dead "build_proximity_route"
        # action (no counterpart in main.py); selective_gather_changed is
        # the channel main.py handles, so no such action must be emitted.
        self.assertNotIn("build_proximity_route", actions)

        # Toggling the checkbox off emits an empty selection.
        menu._on_selective_gather_toggled(False)
        self.assertEqual(emitted[-1][0], [])
        self.assertFalse(menu._selective_gather_enabled)

    def test_settings_window_selective_gather_wiring(self) -> None:
        """SettingsWindow selective-gather controls really emit selective_gather_changed.

        set_selected_materials populates the combo + toggles, the proximity
        button emits the signal with auto-proximity on, and switching the
        toggle off emits an empty selection.
        """
        win = SettingsWindow()
        emitted: List[Tuple[List[str], bool]] = []
        win.selective_gather_changed.connect(
            lambda mats, auto: emitted.append((list(mats), bool(auto)))
        )

        win.populate_selective_materials()
        self.assertGreater(win.combo_selective_materials.count(), 0)

        win.set_selected_materials(["Orichalcum Ore"])
        self.assertIn("Orichalcum Ore", win.get_selected_materials())
        self.assertTrue(win.chk_selective_gather.isChecked())

        # The proximity button emits the signal with the chosen materials.
        win._on_proximity_route_clicked()
        self.assertTrue(emitted, "proximity click must emit selective_gather_changed")
        self.assertIn("Orichalcum Ore", emitted[-1][0])
        self.assertTrue(emitted[-1][1])
        self.assertTrue(win.chk_proximity_route_auto.isChecked())

        # Switching the toggle off emits an empty selection.
        win.chk_selective_gather.setChecked(False)
        win._on_selective_gather_ui_changed()
        self.assertEqual(emitted[-1][0], [])

    def test_map_viewer_set_material_filter_and_proximity(self) -> None:
        """
        _MapCanvas in MapViewerWindow highlights matching nodes, dims others,
        and calculates numbered proximity order badges.
        """
        canvas = _MapCanvas()
        nodes = [
            {"id": 1, "map_x": 100, "map_y": 100, "resource_type": "Orichalcum Ore"},
            {"id": 2, "map_x": 200, "map_y": 200, "resource_type": "Aria"},
            {"id": 3, "map_x": 300, "map_y": 300, "resource_type": "ore_orichalcum"},
            {"id": 4, "map_x": 400, "map_y": 400, "resource_type": "Mela"},
        ]
        canvas.set_nodes_data(nodes, [1, 2, 3, 4])

        # Apply selective filter
        canvas.set_material_filter("Orichalcum Ore")
        self.assertEqual(canvas.get_material_filter(), "Orichalcum Ore")

        prox_route = canvas.calculate_proximity_route()
        self.assertEqual(len(prox_route), 2)
        self.assertEqual([n["id"] for n in prox_route], [1, 3])

        # Clear filter: the canvas shows the full unfiltered node set again.
        # calculate_proximity_route() is selective-only by design (it orders
        # only the filtered matches), so an empty result there after "Tümü"
        # proves nothing — the full set is verified through the APIs that
        # actually return unfiltered nodes.
        canvas.set_material_filter("Tümü")
        self.assertIsNone(canvas.get_material_filter())
        self.assertEqual(
            [n["id"] for n in canvas.filter_nodes_by_selected_material("Tümü")],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            canvas.resequence_active_route_by_proximity(start_xy=(100, 100)),
            [1, 2, 3, 4],
        )

    def test_cartography_radar_material_filter_safe_execution(self) -> None:
        """RadarCanvas sets and queries material filter cleanly without errors."""
        canvas = RadarCanvas()
        self.assertTrue(hasattr(canvas, "set_material_filter"))
        canvas.set_material_filter("Orichalcum Ore")
        self.assertEqual(canvas.get_material_filter(), "Orichalcum Ore")

        canvas.set_material_filter(None)
        self.assertIsNone(canvas.get_material_filter())


# ==============================================================================
# SUITE 6: FULL 5-AGENT END-TO-END INTEGRATION SCENARIO (S1..S5)
# ==============================================================================

class Test5AgentEndToEndScenario(unittest.TestCase):
    """
    Simulates complete multi-agent workflow:
    1. User selects '⛏️ Orichalcum Ore' in UI.
    2. SettingsManager and GatheringSettings record selective gather mode.
    3. RouteManager builds 2-Opt proximity route starting from player coordinates.
    4. MapViewer and CartographyRadar filter views and render numbered badges.
    5. BotWorker and MaterialScanner strictly collect target nodes and ignore clutter.
    6. Dynamic re-sequence upon player progression.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="test_e2e_5agent_")
        self.rm = RouteManager(base_dir=self.temp_dir, active_region="Verteron")
        self.rm.create_region("Verteron")

        # 6 scattered nodes
        nodes_spec = [
            (100, 100, "Orichalcum Ore"),
            (120, 110, "Aria"),
            (300, 300, "ore_orichalcum"),
            (320, 310, "Mela"),
            (500, 500, "Orichalcum Ore"),
            (700, 700, "ore_orichalcum"),
        ]
        for idx, (x, y, m) in enumerate(nodes_spec, start=1):
            self.rm.add_node(region="Verteron", node_id=idx, map_x=x, map_y=y, resource_type=m)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_selective_gathering_lifecycle(self) -> None:
        """Full simulated integration lifecycle across all 5 agents."""
        # 1. Config initialization (S3)
        g_settings = GatheringSettings(
            selective_gather_enabled=True,
            selected_materials=["Orichalcum Ore"],
            proximity_route_auto=True,
        )
        self.assertTrue(g_settings.selective_gather_enabled)

        # 2. Build proximity route from player start (95, 95) (S1)
        prox_route = self.rm.build_proximity_material_route(
            region_name="Verteron",
            target_materials=g_settings.selected_materials,
            start_xy=(95, 95),
            persist_route=True,
        )
        self.assertEqual(len(prox_route), 4)
        self.assertEqual([n.id for n in prox_route], [1, 3, 5, 6])
        # Merge-fix (O8): D25/F-1647 — seq_index follows the matched slots
        # in proximity order, not a renumbered 1..4.
        self.assertEqual([n.seq_index for n in prox_route], [1, 3, 5, 6])

        # 3. Synchronize BotWorker and MaterialScanner (S2)
        ms = MaterialScanner()
        bot = _fast_bot_worker(
            config={},
            vision=MagicMock(),
            input_driver=MagicMock(),
            route_manager=self.rm,
            combat_recovery=MagicMock(),
            navigator=MagicMock(),
            material_scanner=ms,
        )
        bot.set_selected_gather_materials(g_settings.selected_materials)
        self.assertTrue(bot.selective_gather_enabled)
        self.assertEqual(ms.get_selected_materials(), ["Orichalcum Ore"])

        # 4. Viewport harvest check (S2)
        # Attempt to harvest Aria (Node 2) -> BYPASSED
        bot.active_node = self.rm.get_node(2, region="Verteron")
        self.assertTrue(bot._should_bypass_unselected_gather())

        # Attempt to harvest Orichalcum (Node 1) -> ACCEPTED
        bot.active_node = self.rm.get_node(1, region="Verteron")
        self.assertFalse(bot._should_bypass_unselected_gather())

        # 5. UI Map Canvas sync (S4)
        canvas = _MapCanvas()
        all_nodes = [
            {"id": n.id, "map_x": n.map_x, "map_y": n.map_y, "resource_type": n.resource_type}
            for n in self.rm.get_route_nodes("Verteron")
        ]
        canvas.set_nodes_data(all_nodes, [1, 2, 3, 4, 5, 6])
        canvas.set_material_filter(g_settings.selected_materials)
        map_route = canvas.calculate_proximity_route()
        self.assertEqual(len(map_route), 4)

        # 6. Dynamic re-sequencing as player moves near Node 6 at (700, 700) (S1)
        resequenced = self.rm.build_proximity_material_route(
            region_name="Verteron",
            target_materials=g_settings.selected_materials,
            start_xy=(710, 690),
            persist_route=False,
        )
        self.assertEqual(resequenced[0].id, 6, "Node 6 must now be #1 after player moves")
        self.assertEqual([n.seq_index for n in resequenced], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
