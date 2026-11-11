# -*- coding: utf-8 -*-
"""
D64 regression tests for ui/map_viewer.py performance improvements.

Locks in the behaviour of:
  1. sort_nodes_by_proximity — coordinate precomputation + hoisted adjacent
     segment lengths. The expected id orderings below were produced by the
     ORIGINAL implementation (pre-D64) and must be reproduced exactly.
  2. normalize_material_token / node_matches_material — module-level
     translation table, compiled regex and synonym table (same results).

Run: QT_QPA_PLATFORM=offscreen ~/workspace/ytvenv/bin/python -m pytest tests/test_d64_map_viewer_perf.py -q
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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.map_viewer import (  # noqa: E402
    normalize_material_token,
    node_matches_material,
    resolve_node_xy,
    sort_nodes_by_proximity,
)

# Expected orderings + total path lengths, captured from the ORIGINAL
# (pre-D64) implementation. See D64.json for how they were captured.
EXPECTED = {
    "n120_start": (
        [120, 45, 93, 100, 20, 75, 89, 119, 6, 70, 22, 44, 25, 56, 57, 8, 14, 46, 94, 37, 95, 111, 91, 5, 2, 26, 54, 80, 41, 69, 12, 88, 63, 82, 4, 76, 36, 79, 9, 31, 59, 90, 7, 40, 96, 29, 87, 62, 107, 67, 52, 115, 101, 48, 17, 83, 30, 99, 113, 103, 61, 84, 35, 81, 106, 77, 92, 118, 43, 78, 24, 51, 97, 108, 16, 42, 27, 32, 64, 50, 105, 19, 74, 71, 116, 86, 15, 72, 28, 109, 21, 55, 68, 13, 104, 73, 114, 65, 18, 3, 34, 102, 33, 98, 85, 49, 1, 66, 58, 39, 110, 10, 117, 53, 11, 60, 23, 38, 112, 47],
        7304.637921,
    ),
    "n120_nostart": (
        [1, 66, 58, 49, 85, 33, 98, 102, 34, 3, 18, 65, 114, 104, 73, 13, 68, 55, 21, 117, 10, 53, 11, 60, 110, 39, 47, 112, 38, 23, 109, 28, 81, 35, 106, 77, 84, 92, 118, 43, 78, 24, 15, 72, 116, 71, 86, 74, 19, 105, 50, 64, 32, 27, 42, 51, 97, 16, 108, 31, 90, 59, 61, 103, 113, 99, 30, 48, 17, 83, 96, 40, 7, 29, 87, 101, 115, 82, 52, 107, 62, 67, 4, 76, 36, 9, 79, 63, 88, 25, 44, 22, 70, 6, 119, 89, 75, 56, 57, 20, 45, 120, 93, 100, 8, 14, 46, 12, 69, 95, 37, 94, 111, 91, 5, 2, 26, 54, 41, 80],
        7082.081429,
    ),
    "n30_start": (
        [30, 19, 26, 8, 9, 21, 3, 14, 1, 13, 20, 18, 6, 4, 23, 5, 11, 25, 15, 7, 12, 16, 27, 2, 24, 29, 22, 17, 10, 28],
        4822.362827,
    ),
    "n5_start": ([4, 2, 3, 5, 1], 2963.202482),
    "n40_maponly": (
        [1, 10, 2, 19, 11, 28, 20, 3, 39, 22, 5, 13, 30, 38, 29, 21, 4, 32, 15, 7, 35, 18, 27, 36, 9, 26, 34, 17, 8, 25, 33, 16, 24, 6, 23, 14, 31, 40, 12, 37],
        5657.655791,
    ),
    "n25_obj": (
        [1, 8, 2, 23, 17, 11, 5, 20, 14, 21, 6, 12, 18, 24, 3, 9, 15, 22, 16, 10, 4, 25, 19, 13, 7],
        3936.556281,
    ),
}


def _make_nodes(n, seed=7):
    rng = random.Random(seed)
    types = ["Orichalcum Ore", "Aria", "Mela", "Targena", "Gem Crystal", "Tree Log"]
    nodes = []
    for i in range(n):
        cx, cy = rng.choice([(300, 200), (900, 500), (1500, 800), (600, 900)])
        x = cx + rng.uniform(-120, 120)
        y = cy + rng.uniform(-120, 120)
        nodes.append({
            "id": i + 1,
            "map_x": x * 1.5,
            "map_y": y * 1.5,
            "norm_x": x / 1280.0,
            "norm_y": y / 720.0,
            "resource_type": types[i % len(types)],
            "state": "active",
            "in_route": True,
        })
    return nodes


def _path_len(ordered):
    tot = 0.0
    for a, b in zip(ordered, ordered[1:]):
        xa, ya = resolve_node_xy(a)
        xb, yb = resolve_node_xy(b)
        tot += math.hypot(xb - xa, yb - ya)
    return tot


class _Obj:
    def __init__(self, i, x, y):
        self.id = i
        self.map_x = x
        self.map_y = y
        self.norm_x = x / 1280.0
        self.norm_y = y / 720.0
        self.resource_type = "Mela"
        self.state = "active"
        self.in_route = True


def _check_case(name, ordered):
    ids, length = EXPECTED[name]
    assert [n["id"] for n in ordered] == ids, name
    assert abs(_path_len(ordered) - length) < 1e-3, name


def test_sort_120_with_start_matches_original():
    _check_case("n120_start", sort_nodes_by_proximity(_make_nodes(120), start_xy=(100.0, 100.0)))


def test_sort_120_no_start_matches_original():
    _check_case("n120_nostart", sort_nodes_by_proximity(_make_nodes(120)))


def test_sort_30_with_start_matches_original():
    _check_case("n30_start", sort_nodes_by_proximity(_make_nodes(30, seed=3), start_xy=(640.0, 360.0)))


def test_sort_5_greedy_only_matches_original():
    _check_case("n5_start", sort_nodes_by_proximity(_make_nodes(5, seed=5), start_xy=(0.0, 0.0)))


def test_sort_map_only_coords_matches_original():
    nodes = [
        {"id": i + 1, "map_x": float((i * 137) % 1200), "map_y": float((i * 251) % 700),
         "resource_type": "Aria", "state": "active", "in_route": True}
        for i in range(40)
    ]
    _check_case("n40_maponly", sort_nodes_by_proximity(nodes, start_xy=(10.0, 10.0)))


def test_sort_object_nodes_matches_original():
    objs = [_Obj(i + 1, float((i * 173) % 1200), float((i * 229) % 700)) for i in range(25)]
    _check_case("n25_obj", sort_nodes_by_proximity(objs, start_xy=(50.0, 50.0)))


def test_sort_edge_cases():
    assert sort_nodes_by_proximity([]) == []
    single = [{"id": 1, "map_x": 10.0, "map_y": 20.0}]
    assert [n["id"] for n in sort_nodes_by_proximity(single)] == [1]
    # Input order / dicts are not mutated by sorting (copies are sorted).
    nodes = _make_nodes(10, seed=9)
    before = [n["id"] for n in nodes]
    out = sort_nodes_by_proximity(nodes, start_xy=(0.0, 0.0))
    assert sorted(n["id"] for n in out) == sorted(before)
    assert [n["id"] for n in nodes] == before


def test_sort_120_perf_smoke():
    # Baseline (pre-D64) was ~2000 ms/call for 120 clustered nodes; the
    # optimized version runs in ~150-300 ms on this VM. The 1000 ms bound
    # is deliberately generous to avoid flakes while still locking the
    # order-of-magnitude gain.
    nodes = _make_nodes(120)
    t0 = time.perf_counter()
    out = sort_nodes_by_proximity(nodes, start_xy=(100.0, 100.0))
    dt_ms = (time.perf_counter() - t0) * 1000.0
    assert len(out) == 120
    assert dt_ms < 1000.0, f"sort took {dt_ms:.0f} ms, expected < 1000 ms"


def test_normalize_material_token_cases():
    assert normalize_material_token("Orichalcum Ore") == "orichalcum ore"
    assert normalize_material_token("Çiçekli Targena!") == "cicekli targena"
    assert normalize_material_token("Ağaç / Kereste #3") == "agac kereste 3"
    assert normalize_material_token("ĞÜŞÖÇİ") == "gusoci"
    assert normalize_material_token("") == ""
    assert normalize_material_token(None) == ""


def test_node_matches_material_cases():
    ore = {"id": 1, "resource_type": "Orichalcum Ore"}
    aria = {"id": 2, "resource_type": "Aria"}
    tree = {"id": 3, "resource_type": "Tree Log"}
    assert node_matches_material(ore, "Orichalcum Ore") is True
    assert node_matches_material(aria, "Orichalcum Ore") is False
    assert node_matches_material(ore, None) is True
    assert node_matches_material(ore, "Tümü") is True
    assert node_matches_material(ore, "") is True
    assert node_matches_material(ore, ["Aria", "Orichalcum"]) is True
    assert node_matches_material(aria, ["Orichalcum", "Mela"]) is False
    # Category synonyms (module-level table, same semantics as before):
    # the FILTER word must be a category key ("tree", "gem", ...); it then
    # matches any synonym token in the node text. Turkish words as filter
    # are NOT keys, so they only match literally.
    assert node_matches_material(ore, "orichalcum") is True
    assert node_matches_material(ore, "maden") is False
    assert node_matches_material(aria, "ot") is False
    assert node_matches_material(tree, "ağaç") is False
    assert node_matches_material(tree, "odun") is False
    assert node_matches_material(tree, "tree") is True
    assert node_matches_material(tree, "log") is True
    # Synonym direction that does work: English key vs Turkish node token.
    turkish_tree = {"id": 4, "resource_type": "Ağaç Kütüğü"}
    assert node_matches_material(turkish_tree, "tree") is True
    assert node_matches_material(turkish_tree, "log") is True
    odyle = {"id": 5, "resource_type": "Odyle Kristali"}
    assert node_matches_material(odyle, "gem") is True
    assert node_matches_material(odyle, "crystal") is True
    assert node_matches_material(aria, "maden") is False
    # Turkish normalization in the filter itself. "AĞAÇ" normalizes to
    # "agac", which matches literally inside the normalized node text
    # "agac kutugu" — this is real production behaviour (the material
    # combo is populated with raw Turkish resource_type strings, see
    # _populate_material_filter_combo), so it must stay True. (An earlier
    # draft of this test asserted False here; that was wrong: the word is
    # not a synonym-table key, but the literal-substring path matches it.)
    assert node_matches_material(turkish_tree, "AĞAÇ") is True  # literal match
