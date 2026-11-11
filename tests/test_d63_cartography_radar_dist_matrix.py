# -*- coding: utf-8 -*-
"""
D63 — cartography_radar mesafe-matrisi iyileştirmesi kilit testleri.

Kapsanan değişiklik (ui/cartography_radar.py):
- `_distance_matrix` bir kez kurulur; `optimize_smooth_loop` aynı tabloyu
  tur maliyeti, multi-start NN, 2-opt ve or-opt adımlarına geçirir.
- `_tour_cost_indices` / `_two_opt_indices` / `_or_opt_indices` isteğe
  bağlı `dist` parametresi alır; verilmezse matrisi kendisi kurar.
- 2-opt ve or-opt iç döngülerindeki `math.hypot` çağrıları tablo
  okumasıyla (`dist[a, b]`) değiştirildi.

Davranış değişikliği YOK: matris değerleri `math.hypot` ile aynı float64
Öklid mesafesidir; bu testler matrisli/matrissiz yolların aynı turu ve
aynı maliyeti ürettiğini kilitler. Testleri koşmak test ajanlarına aittir
(2026-11-08 21:12 kuralı: kodlama ajanları pytest çalıştırmaz).
"""

from __future__ import annotations

import math
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ui.cartography_radar import (
    SequentialRoutePlanner,
    _distance_matrix,
)


def _coords(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]


def _nodes(n: int, seed: int) -> list:
    rng = random.Random(seed)
    return [
        {"id": 1000 + i, "map_x": rng.uniform(0, 1000), "map_y": rng.uniform(0, 1000)}
        for i in range(n)
    ]


def _planner_internal() -> SequentialRoutePlanner:
    """Dahili NN/2-opt/or-opt yolunu izole et (harici planlayıcı bypass)."""
    planner = SequentialRoutePlanner()
    planner._external_planner = None
    return planner


def test_distance_matrix_matches_math_hypot() -> None:
    coords = _coords(25, seed=11)
    dist = _distance_matrix(coords)
    assert dist.shape == (25, 25)
    assert dist.dtype == np.float64
    for i in range(25):
        assert dist[i, i] == 0.0
        for j in range(25):
            expected = math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1])
            assert dist[i, j] == pytest.approx(expected, rel=1e-12, abs=1e-12)
            assert dist[i, j] == dist[j, i]


def test_tour_cost_with_and_without_dist_agree() -> None:
    coords = _coords(40, seed=23)
    dist = _distance_matrix(coords)
    rng = random.Random(5)
    for _ in range(10):
        order = list(range(40))
        rng.shuffle(order)
        without = SequentialRoutePlanner._tour_cost_indices(order, coords)
        with_dist = SequentialRoutePlanner._tour_cost_indices(order, coords, dist)
        assert with_dist == pytest.approx(without, rel=1e-12)


def test_tour_cost_edge_cases() -> None:
    assert SequentialRoutePlanner._tour_cost_indices([], []) == 0.0
    assert SequentialRoutePlanner._tour_cost_indices([0], [(3.0, 4.0)]) == 0.0
    cost = SequentialRoutePlanner._tour_cost_indices([0, 1], [(0.0, 0.0), (3.0, 4.0)])
    assert cost == pytest.approx(10.0)  # kapalı tur: 5 gidiş + 5 dönüş


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_two_opt_with_and_without_dist_identical(seed: int) -> None:
    coords = _coords(30, seed=seed)
    dist = _distance_matrix(coords)
    rng = random.Random(seed)
    order = list(range(30))
    rng.shuffle(order)
    a = SequentialRoutePlanner._two_opt_indices(order, coords)
    b = SequentialRoutePlanner._two_opt_indices(order, coords, dist)
    assert a == b


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_or_opt_with_and_without_dist_identical(seed: int) -> None:
    coords = _coords(30, seed=seed)
    dist = _distance_matrix(coords)
    rng = random.Random(seed)
    order = list(range(30))
    rng.shuffle(order)
    a = SequentialRoutePlanner._or_opt_indices(order, coords)
    b = SequentialRoutePlanner._or_opt_indices(order, coords, dist)
    assert a == b


def test_two_opt_never_worsens_tour() -> None:
    coords = _coords(30, seed=41)
    rng = random.Random(41)
    order = list(range(30))
    rng.shuffle(order)
    before = SequentialRoutePlanner._tour_cost_indices(order, coords)
    after_order = SequentialRoutePlanner._two_opt_indices(order, coords)
    after = SequentialRoutePlanner._tour_cost_indices(after_order, coords)
    assert sorted(after_order) == list(range(30))
    assert after <= before + 1e-9


@pytest.mark.parametrize("n,seed", [(5, 7), (30, 8), (60, 9)])
def test_optimize_smooth_loop_internal_path_valid_and_deterministic(n: int, seed: int) -> None:
    planner = _planner_internal()
    nodes = _nodes(n, seed)
    first = planner.optimize_smooth_loop(list(nodes))
    second = planner.optimize_smooth_loop(list(nodes))
    assert [nd["id"] for nd in first] == [nd["id"] for nd in second]
    assert len(first) == n
    assert sorted(nd["id"] for nd in first) == sorted(nd["id"] for nd in nodes)
    for idx, nd in enumerate(first, start=1):
        assert nd["seq_index"] == idx
    coords = [planner._node_xy(nd) for nd in nodes]
    id_to_pos = {nd["id"]: i for i, nd in enumerate(nodes)}
    order = [id_to_pos[nd["id"]] for nd in first]
    cost = SequentialRoutePlanner._tour_cost_indices(order, coords)
    assert cost > 0.0
    input_cost = SequentialRoutePlanner._tour_cost_indices(list(range(n)), coords)
    assert cost <= input_cost + 1e-9
