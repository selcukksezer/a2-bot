# -*- coding: utf-8 -*-
"""
Unit tests for W50: Select-and-Gather (Seç-Topla) Orichalcum Ore Filtering
& Nearest-Neighbor + 2-Opt TSP Proximity Route Optimization.

Verifies that when only 'Orichalcum Ore' is selected in Seç-Topla mode:
1. Other materials (Aria, Mela, Targena, Gem, Tree, Iron, etc.) and ungatherable nodes are skipped.
2. MaterialScanner and BotWorker strictly bypass non-Orichalcum nodes in 3D viewport / active gathering.
3. Nearest-Neighbor initialization starts from the Orichalcum node closest to the player's position.
4. 2-Opt TSP local search untangles crossed edges and minimizes total route travel distance.
5. RouteManager.build_proximity_material_route builds, persists, and dynamically re-sequences
   the Orichalcum-only Nearest-Neighbor + 2-Opt TSP route.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.material_scanner import DetectedMaterial, MaterialScanner
from core.route_manager import (
    NodeData,
    RouteManager,
    filter_nodes_by_material,
    sort_nodes_by_proximity,
)
from core.state_machine import BotWorker


def _make_detected_material(mtype: str, sx: int = 300, sy: int = 300, dist: float = 10.0) -> DetectedMaterial:
    return DetectedMaterial(
        material_type=mtype,
        source="3d_world",
        screen_x=sx,
        screen_y=sy,
        bbox=(sx - 20, sy - 20, sx + 20, sy + 20),
        rel_map_dx=0.0,
        rel_map_dy=0.0,
        estimated_distance_m=dist,
        confidence=0.93,
        in_air=False,
        stack_count=1,
        harvests_remaining=3,
        priority_score=0.90,
        rarity_tier=1,
    )


def _tour_open_distance(nodes: List[NodeData]) -> float:
    total = 0.0
    for i in range(len(nodes) - 1):
        total += math.hypot(
            nodes[i + 1].map_x - nodes[i].map_x,
            nodes[i + 1].map_y - nodes[i].map_y,
        )
    return total


def _tour_closed_distance(nodes: List[NodeData]) -> float:
    total = 0.0
    n = len(nodes)
    for i in range(n):
        p1 = nodes[i]
        p2 = nodes[(i + 1) % n]
        total += math.hypot(p2.map_x - p1.map_x, p2.map_y - p1.map_y)
    return total


def test_filter_only_orichalcum_ore_skips_other_materials() -> None:
    """1. Verify selecting only 'Orichalcum Ore' keeps Orichalcum variants and skips all other materials."""
    mixed_nodes = [
        NodeData(id=1, map_x=100, map_y=100, resource_type="Orichalcum Ore"),
        NodeData(id=2, map_x=150, map_y=120, resource_type="Aria"),
        NodeData(id=3, map_x=220, map_y=180, resource_type="ore_orichalcum"),
        NodeData(id=4, map_x=300, map_y=240, resource_type="herb_mela"),
        NodeData(id=5, map_x=380, map_y=290, resource_type="herb_targena"),
        NodeData(id=6, map_x=450, map_y=350, resource_type="gem_crystal"),
        NodeData(id=7, map_x=520, map_y=410, resource_type="tree_log"),
        NodeData(id=8, map_x=600, map_y=470, resource_type="Pure Orichalcum Ore"),
        NodeData(id=9, map_x=680, map_y=530, resource_type="Orichalcum Ore", not_gatherable=True),
    ]

    filtered = filter_nodes_by_material(mixed_nodes, "Orichalcum Ore")
    filtered_ids = [n.id for n in filtered]

    assert filtered_ids == [1, 3, 8]
    for node in filtered:
        assert "orichalcum" in node.resource_type.lower()
        assert node.not_gatherable is False


def test_scanner_and_bot_worker_bypass_non_orichalcum_targets() -> None:
    """2. Verify MaterialScanner and BotWorker bypass non-Orichalcum materials when only Orichalcum Ore is selected."""
    scanner = MaterialScanner()
    bot = BotWorker(
        config={},
        vision=MagicMock(),
        input_driver=MagicMock(),
        route_manager=MagicMock(),
        combat_recovery=MagicMock(),
        navigator=MagicMock(),
        material_scanner=scanner,
    )

    bot.set_selected_gather_materials(["Orichalcum Ore"])
    assert bot.selective_gather_enabled is True
    assert scanner.has_selected_materials() is True
    assert scanner.get_selected_materials() == ["Orichalcum Ore"]

    # Viewport candidate filtering in MaterialScanner
    viewport_candidates = [
        _make_detected_material("Aria", 200, 200),
        _make_detected_material("Orichalcum Ore", 350, 300),
        _make_detected_material("herb_mela", 400, 250),
        _make_detected_material("ore_orichalcum", 500, 320),
        _make_detected_material("gem_crystal", 600, 400),
        _make_detected_material("tree_log", 700, 450),
    ]
    retained = [c for c in viewport_candidates if scanner.is_material_selected(c.material_type)]
    assert [c.material_type for c in retained] == ["Orichalcum Ore", "ore_orichalcum"]

    # BotWorker opportunistic harvest bypass check
    for non_target in ("Aria", "Mela", "herb_targena", "gem_crystal", "tree_log"):
        bot.active_node = NodeData(id=99, resource_type=non_target)
        assert bot._should_bypass_unselected_gather() is True
        assert bot.is_material_selected_for_gather(bot.active_node) is False

    for target_ori in ("Orichalcum Ore", "ore_orichalcum"):
        bot.active_node = NodeData(id=10, resource_type=target_ori)
        assert bot._should_bypass_unselected_gather() is False
        assert bot.is_material_selected_for_gather(bot.active_node) is True


def test_nearest_neighbor_starts_from_closest_orichalcum_node() -> None:
    """3. Verify Nearest-Neighbor selects the Orichalcum node closest to player start_xy first."""
    mixed_nodes = [
        NodeData(id=1, map_x=800, map_y=800, resource_type="Orichalcum Ore"),
        NodeData(id=2, map_x=115, map_y=115, resource_type="Aria"),  # Closer to player, but NOT Orichalcum!
        NodeData(id=3, map_x=500, map_y=500, resource_type="ore_orichalcum"),
        NodeData(id=4, map_x=130, map_y=125, resource_type="Orichalcum Ore"),  # Closest Orichalcum to player (110, 110)
        NodeData(id=5, map_x=300, map_y=300, resource_type="Orichalcum Ore"),
        NodeData(id=6, map_x=250, map_y=250, resource_type="herb_mela"),
    ]

    orichalcum_only = filter_nodes_by_material(mixed_nodes, "Orichalcum Ore")
    assert [n.id for n in orichalcum_only] == [1, 3, 4, 5]

    route = sort_nodes_by_proximity(orichalcum_only, start_xy=(110.0, 110.0))
    ordered_ids = [n.id for n in route]

    # Node 2 (Aria) was skipped; Node 4 is the closest Orichalcum node, followed by 5 -> 3 -> 1
    assert ordered_ids == [4, 5, 3, 1]


def test_two_opt_tsp_untangles_crossed_orichalcum_route_and_minimizes_distance() -> None:
    """4. Verify 2-Opt TSP local search untangles crossed edges and reduces total route length."""
    # Case A: 4 Orichalcum nodes at square corners given in crossed hourglass sequence
    crossed_square = [
        NodeData(id=1, map_x=0, map_y=0, resource_type="Orichalcum Ore"),
        NodeData(id=2, map_x=100, map_y=100, resource_type="Orichalcum Ore"),
        NodeData(id=3, map_x=0, map_y=100, resource_type="Orichalcum Ore"),
        NodeData(id=4, map_x=100, map_y=0, resource_type="Orichalcum Ore"),
    ]
    crossed_perimeter = _tour_closed_distance(crossed_square)
    assert crossed_perimeter > 480.0  # 141.42 + 100 + 141.42 + 100 = 482.84

    untangled_square = sort_nodes_by_proximity(crossed_square, start_xy=(0.0, 0.0))
    untangled_perimeter = _tour_closed_distance(untangled_square)
    assert math.isclose(untangled_perimeter, 400.0, rel_tol=1e-4)

    # Case B: 8 Orichalcum nodes on a circle in a tangled star/zigzag sequence
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    radius = 250.0
    zigzag_indices = [0, 4, 1, 5, 2, 6, 3, 7]
    scrambled_circle = [
        NodeData(
            id=i + 1,
            map_x=int(round(500 + radius * math.cos(angles[idx]))),
            map_y=int(round(500 + radius * math.sin(angles[idx]))),
            resource_type="Orichalcum Ore",
        )
        for i, idx in enumerate(zigzag_indices)
    ]

    initial_open_dist = _tour_open_distance(scrambled_circle)
    start_pt = (float(scrambled_circle[0].map_x), float(scrambled_circle[0].map_y))
    optimized_circle = sort_nodes_by_proximity(scrambled_circle, start_xy=start_pt)
    optimized_open_dist = _tour_open_distance(optimized_circle)

    assert len(optimized_circle) == 8
    assert optimized_circle[0].id == scrambled_circle[0].id
    assert optimized_open_dist < initial_open_dist * 0.65


def test_build_proximity_material_route_end_to_end_orichalcum_and_dynamic_resequencing(tmp_path: Path) -> None:
    """5. Verify RouteManager builds, persists, and dynamically re-sequences an Orichalcum-only TSP route."""
    rm = RouteManager(base_dir=str(tmp_path), active_region="Altgard")
    rm.create_region("Altgard")

    region_nodes = [
        (1, 100, 100, "Orichalcum Ore"),
        (2, 110, 105, "Aria"),
        (3, 250, 120, "ore_orichalcum"),
        (4, 260, 130, "herb_mela"),
        (5, 420, 220, "Orichalcum Ore"),
        (6, 430, 230, "gem_crystal"),
        (7, 650, 480, "ore_orichalcum"),
        (8, 660, 490, "tree_log"),
        (9, 820, 700, "Orichalcum Ore"),
    ]
    for nid, mx, my, rtype in region_nodes:
        rm.add_node(region="Altgard", node_id=nid, map_x=mx, map_y=my, resource_type=rtype, max_harvests=3)

    # Build route starting near (95, 95) -> closest Orichalcum is Node 1 (Node 2 Aria is ignored)
    route_start = rm.build_proximity_material_route(
        region_name="Altgard",
        target_materials=["Orichalcum Ore"],
        start_xy=(95.0, 95.0),
        persist_route=True,
    )

    assert [n.id for n in route_start] == [1, 3, 5, 7, 9]
    assert [n.seq_index for n in route_start] == [1, 2, 3, 4, 5]
    for node in route_start:
        assert "orichalcum" in node.resource_type.lower()

    # Dynamic re-sequencing when player moves near (830, 710) -> closest Orichalcum is now Node 9
    route_moved = rm.build_proximity_material_route(
        region_name="Altgard",
        target_materials="Orichalcum Ore",
        start_xy=(830.0, 710.0),
        persist_route=True,
    )
    assert route_moved[0].id == 9
    assert [n.id for n in route_moved] == [9, 7, 5, 3, 1]
    assert [n.seq_index for n in route_moved] == [1, 2, 3, 4, 5]

    # Verify persistence round-trip from disk
    reloaded_rm = RouteManager(base_dir=str(tmp_path), active_region="Altgard")
    reloaded_rm.load_region("Altgard")
    prox_routes = [r for r in reloaded_rm.get_routes("Altgard") if "proximity" in r and "orichalcum" in r]
    assert len(prox_routes) == 1
    persisted_nodes = reloaded_rm.get_route_nodes("Altgard", route_name=prox_routes[0])
    assert len(persisted_nodes) == 5
    assert all("orichalcum" in n.resource_type.lower() for n in persisted_nodes)
