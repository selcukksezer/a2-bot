# -*- coding: utf-8 -*-
"""
D27 regression tests — route_manager performance improvements.

Locks the behaviour of the four O(n^2) -> O(n)/O(1) optimisations in
core/route_manager.py (dev/D027):

1. filter_nodes_by_subregion: identity-set dedup instead of dataclass
   __eq__ list scan.
2. sort_nodes_by_proximity: precomputed pairwise distance matrix instead
   of repeated math.hypot in the 2-Opt loops.
3. _match_node_material: module-level YOLO 6-group map + per-call
   normalisation caches.
4. RouteManager.get_next_ready_node: position dict instead of
   list.index() per ready candidate.

Each test compares the optimised code against an independent naive
reference implementation (or an explicit expected selection), so a
future regression in either speed or result is caught.
"""

from __future__ import annotations

import math
import os
import random
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest

from core.route_manager import (
    NodeData,
    RouteManager,
    _match_node_material,
    filter_nodes_by_subregion,
    get_subregion_info,
    sort_nodes_by_proximity,
)


def _naive_filter_by_subregion(nodes, subregion):
    """Pre-D27 logic: dataclass __eq__ list-membership dedup."""
    clean_sub = str(subregion or "").strip()
    clean_sub_lower = clean_sub.lower()
    by_route = [
        n for n in nodes
        if any(str(r).strip().lower() == clean_sub_lower for r in getattr(n, "routes", []))
    ]
    sub_info = get_subregion_info(clean_sub)
    by_geom = []
    if sub_info:
        bbox = sub_info.get("bbox")
        norm_bbox = sub_info.get("norm_bbox")
        center = sub_info.get("center")
        radius = float(sub_info.get("radius", 260.0))
        for n in nodes:
            mx = float(getattr(n, "map_x", 0) or 0)
            my = float(getattr(n, "map_y", 0) or 0)
            nx = float(getattr(n, "norm_x", 0) or 0)
            ny = float(getattr(n, "norm_y", 0) or 0)
            in_box = False
            if bbox and (mx > 0 or my > 0):
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    in_box = True
            elif norm_bbox and (nx > 0 or ny > 0):
                if norm_bbox[0] <= nx <= norm_bbox[2] and norm_bbox[1] <= ny <= norm_bbox[3]:
                    in_box = True
            if not in_box and center and (mx > 0 or my > 0):
                dist = math.hypot(mx - float(center[0]), my - float(center[1]))
                if dist <= (radius * 1.25):
                    in_box = True
            if in_box and n not in by_geom:
                by_geom.append(n)
    if by_route and by_geom:
        seen = {id(n) for n in by_route}
        combined = list(by_route)
        for n in by_geom:
            if id(n) not in seen:
                combined.append(n)
                seen.add(id(n))
        return combined
    elif by_route:
        return by_route
    elif by_geom:
        return by_geom
    return []


def _extract_xy(item):
    if isinstance(item, dict):
        return float(item.get("map_x", 0) or 0), float(item.get("map_y", 0) or 0)
    return float(getattr(item, "map_x", 0) or 0), float(getattr(item, "map_y", 0) or 0)


def _naive_sort_by_proximity(nodes, start_xy=None):
    """Pre-D27 logic: greedy nearest neighbour + 2-Opt with live hypot."""
    if not nodes:
        return []
    node_list = list(nodes)
    n = len(node_list)
    if n <= 1:
        return node_list
    coords = [_extract_xy(item) for item in node_list]
    if n == 2:
        if start_xy is not None:
            sx, sy = float(start_xy[0]), float(start_xy[1])
            d0 = math.hypot(coords[0][0] - sx, coords[0][1] - sy)
            d1 = math.hypot(coords[1][0] - sx, coords[1][1] - sy)
            if d1 < d0:
                return [node_list[1], node_list[0]]
            return [node_list[0], node_list[1]]
        return node_list

    def d(i, j):
        return math.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])

    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        first_idx = min(range(n), key=lambda i: math.hypot(coords[i][0] - sx, coords[i][1] - sy))
    else:
        first_idx = 0
    visited = [False] * n
    tour = [first_idx]
    visited[first_idx] = True
    curr_idx = first_idx
    for _ in range(n - 1):
        best_next, best_dist = -1, float("inf")
        curr_x, curr_y = coords[curr_idx]
        for j in range(n):
            if not visited[j]:
                dd = math.hypot(coords[j][0] - curr_x, coords[j][1] - curr_y)
                if dd < best_dist:
                    best_dist, best_next = dd, j
        tour.append(best_next)
        visited[best_next] = True
        curr_idx = best_next
    for _ in range(60):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[(i + 1) % n]
                c, dd = tour[j], tour[(j + 1) % n]
                if (d(a, c) + d(b, dd)) + 1e-6 < (d(a, b) + d(c, dd)):
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
        if not improved:
            break
    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        best_pos = min(range(n), key=lambda k: math.hypot(coords[tour[k]][0] - sx, coords[tour[k]][1] - sy))
        tour = tour[best_pos:] + tour[:best_pos]
        dist_fwd = math.hypot(coords[tour[1]][0] - sx, coords[tour[1]][1] - sy)
        dist_bwd = math.hypot(coords[tour[-1]][0] - sx, coords[tour[-1]][1] - sy)
        if dist_bwd < dist_fwd:
            tour = [tour[0]] + list(reversed(tour[1:]))
    else:
        if 0 in tour:
            zpos = tour.index(0)
            tour = tour[zpos:] + tour[:zpos]
    for _ in range(30):
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                a, b = tour[i], tour[i + 1]
                c, dd = tour[j], tour[j + 1]
                if (d(a, c) + d(b, dd)) + 1e-6 < (d(a, b) + d(c, dd)):
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
        if not improved:
            break
    return [node_list[i] for i in tour]


def _make_nodes(count, seed=42):
    rng = random.Random(seed)
    return [
        NodeData(
            id=i + 1,
            region="Altgard",
            routes=["dredgion crash site"] if i % 3 == 0 else [],
            map_x=rng.randint(400, 1500),
            map_y=rng.randint(200, 900),
            norm_x=rng.uniform(0.2, 0.8),
            norm_y=rng.uniform(0.2, 0.8),
            resource_type="ore_orichalcum",
        )
        for i in range(count)
    ]


def test_filter_subregion_matches_naive_and_is_fast():
    nodes = _make_nodes(300)
    t0 = time.perf_counter()
    got = filter_nodes_by_subregion(nodes, "dredgion crash site")
    elapsed = time.perf_counter() - t0
    want = _naive_filter_by_subregion(nodes, "dredgion crash site")
    assert [n.id for n in got] == [n.id for n in want]
    # Pre-D27 this took ~1s at n=300 and ~27s at n=1500 (O(n^2) __eq__ scans).
    assert elapsed < 2.0, f"filter too slow: {elapsed:.3f}s"


def test_filter_subregion_large_input_fast():
    nodes = _make_nodes(1500, seed=7)
    t0 = time.perf_counter()
    got = filter_nodes_by_subregion(nodes, "dredgion crash site")
    elapsed = time.perf_counter() - t0
    assert len(got) == 1500
    assert elapsed < 2.0, f"filter too slow at n=1500: {elapsed:.3f}s"


def test_sort_proximity_matches_naive_reference():
    rng = random.Random(123)
    coords = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(40)]
    nodes = [
        NodeData(id=i + 1, map_x=int(x), map_y=int(y), norm_x=x / 1000, norm_y=y / 1000)
        for i, (x, y) in enumerate(coords)
    ]
    got = sort_nodes_by_proximity(nodes, start_xy=(0, 0))
    want = _naive_sort_by_proximity(nodes, start_xy=(0, 0))
    assert [n.id for n in got] == [n.id for n in want]


def test_sort_proximity_is_permutation_and_fast():
    rng = random.Random(99)
    nodes = [
        NodeData(id=i + 1, map_x=rng.randint(0, 1000), map_y=rng.randint(0, 1000))
        for i in range(300)
    ]
    t0 = time.perf_counter()
    got = sort_nodes_by_proximity(nodes, start_xy=(0, 0))
    elapsed = time.perf_counter() - t0
    assert sorted(n.id for n in got) == sorted(n.id for n in nodes)
    assert elapsed < 5.0, f"sort too slow at n=300: {elapsed:.3f}s"


@pytest.mark.parametrize(
    "node,targets,expected",
    [
        ({"resource_type": "ore_orichalcum"}, ["ore_orichalcum"], True),
        ({"resource_type": "Orichalcum Ore"}, ["ore_orichalcum"], True),
        ({"class_id": 3}, ["ore_orichalcum"], True),  # YOLO 6-group map: 3 -> ore_orichalcum
        ({"class_id": 0}, ["herb_aria"], True),  # YOLO 6-group map: 0 -> herb_aria
        ({"class_id": 5}, ["tree_log"], True),  # YOLO 6-group map: 5 -> tree_log
        ({"resource_type": "herb_aria"}, ["ore_orichalcum"], False),
        ({"resource_type": "ore_orichalcum"}, ["herb_aria", "gem_crystal"], False),
    ],
)
def test_match_node_material_cases(node, targets, expected):
    assert _match_node_material(node, targets) is expected


def _make_manager(count):
    mgr = RouteManager(base_dir=os.path.join(PROJECT_ROOT, "routes"))
    nodes = [
        NodeData(id=i + 1, region="Altgard", routes=["All"],
                 map_x=i, map_y=i, resource_type="ore_orichalcum")
        for i in range(count)
    ]
    mgr._regions["Altgard"] = {"routes": ["All"], "nodes": nodes}
    mgr.active_region = "Altgard"
    mgr.active_route = "All"
    return mgr


def test_get_next_ready_node_round_robin():
    mgr = _make_manager(20)
    mgr._last_selected_node_id[("Altgard", "All")] = 5
    nxt = mgr.get_next_ready_node(region="Altgard", route="All")
    assert nxt is not None and nxt.id == 6


def test_get_next_ready_node_wraps_around():
    mgr = _make_manager(20)
    mgr._last_selected_node_id[("Altgard", "All")] = 20
    nxt = mgr.get_next_ready_node(region="Altgard", route="All")
    assert nxt is not None and nxt.id == 1


def test_get_next_ready_node_large_is_fast():
    mgr = _make_manager(1500)
    mgr._last_selected_node_id[("Altgard", "All")] = 1499
    t0 = time.perf_counter()
    nxt = mgr.get_next_ready_node(region="Altgard", route="All")
    elapsed = time.perf_counter() - t0
    assert nxt is not None and nxt.id == 1500
    # Pre-D27: ~0.42s per call at n=1500 (O(n^2) list.index scans).
    assert elapsed < 0.1, f"selection too slow: {elapsed:.3f}s"
