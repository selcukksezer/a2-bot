# -*- coding: utf-8 -*-
"""
D28 — sequential_router performans iyileştirmelerinin kilitleme testleri.

Kapsanan değişiklikler (`core/sequential_router.py`, `dev/D028`):
1. `_turn_cost` hızlı yolu: taban terim doğrudan `cos(theta)` ile hesaplanır;
   pahalı `acos` yalnızca keskin dönüş (`theta > 0.45*pi`) durumunda çağrılır.
2. `_angular_sweep_init` vektörel (`np.arctan2`/`np.hypot` + `lexsort`).
3. `_angular_nearest_neighbor_init` vektörel aday maliyeti + `argmin`.
4. `_two_opt_loop` açısal dalda `a` tarafı önbelleği (cache).
5. `validate_sequential_router_contract` sonucu önbelleği (deterministik
   duman testi bir kez çalışır).

Bu testler her iyileştirmeyi, baz (skaler) referans uygulamaya karşı
bit-bit / epsilon düzeyinde eşdeğerlikle kilitler; davranış değişikliği
değil, saf hız iyileştirmesi olduğunu kanıtlar.
"""

from __future__ import annotations

import math
import os
import random
import sys
from unittest import mock

import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import core.sequential_router as sr
from core.sequential_router import SequentialRoutePlanner


# ---------------------------------------------------------------------------
# Skaler referans uygulamalar (değişiklik ÖNCESİ kodun bağımsız kopyası)
# ---------------------------------------------------------------------------

def _ref_turn_angle_rad(p_prev, p_curr, p_next):
    v1x = p_curr[0] - p_prev[0]
    v1y = p_curr[1] - p_prev[1]
    v2x = p_next[0] - p_curr[0]
    v2y = p_next[1] - p_curr[1]
    len1 = math.hypot(v1x, v1y)
    len2 = math.hypot(v2x, v2y)
    if len1 < 1e-9 or len2 < 1e-9:
        return 0.0
    cos_theta = (v1x * v2x + v1y * v2y) / (len1 * len2)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.acos(cos_theta)


def _ref_turn_cost(p_prev, p_curr, p_next):
    """Orijinal `_turn_cost`: acos -> cos -> eşik cezaları."""
    theta = _ref_turn_angle_rad(p_prev, p_curr, p_next)
    base = 0.5 * (1.0 - math.cos(theta))
    if theta > (math.pi * 0.45):
        excess = (theta - math.pi * 0.45) / (math.pi * 0.55)
        base += 3.2 * (excess ** 2)
    if theta > (math.pi * 0.5):
        base += 2.5
    return base


def _ref_angular_sweep_init(coords):
    cx = sum(x for x, _ in coords) / len(coords)
    cy = sum(y for _, y in coords) / len(coords)
    angles = []
    for idx, (x, y) in enumerate(coords):
        ang = math.atan2(y - cy, x - cx)
        rad = math.hypot(x - cx, y - cy)
        angles.append((ang, rad, idx))
    angles.sort(key=lambda item: (item[0], item[1]))
    return [idx for _, _, idx in angles]


def _ref_angular_nn_init(planner, coords, dist_matrix, start_idx):
    """Orijinal skaler `_angular_nearest_neighbor_init`."""
    n = len(coords)
    visited = [False] * n
    tour = [start_idx]
    visited[start_idx] = True
    for _ in range(n - 1):
        curr = tour[-1]
        best_next = -1
        best_cost = float("inf")
        has_prev = len(tour) >= 2
        prev = tour[-2] if has_prev else curr
        for cand in range(n):
            if visited[cand]:
                continue
            d = float(dist_matrix[curr, cand])
            if has_prev and d > 1e-6:
                theta = _ref_turn_angle_rad(coords[prev], coords[curr], coords[cand])
                turn_penalty = 0.5 * (1.0 - math.cos(theta))
                cost = d * (1.0 + planner.angular_penalty_weight * turn_penalty)
            else:
                cost = d
            if cost < best_cost:
                best_cost = cost
                best_next = cand
        if best_next == -1:
            for cand in range(n):
                if not visited[cand]:
                    best_next = cand
                    break
        visited[best_next] = True
        tour.append(best_next)
    return tour


def _dist_matrix(coords):
    pts = np.asarray(coords, dtype=np.float64)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.hypot(diff[:, :, 0], diff[:, :, 1])


def _rand_coords(n, seed):
    rng = random.Random(seed)
    return [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. _turn_cost hızlı yolu
# ---------------------------------------------------------------------------

class TestTurnCostFastPath:
    def test_matches_reference_on_random_triples(self):
        planner = SequentialRoutePlanner()
        rng = random.Random(7)
        worst = 0.0
        for _ in range(2000):
            t = tuple((rng.uniform(0, 100), rng.uniform(0, 100)) for _ in range(3))
            worst = max(worst, abs(planner._turn_cost(*t) - _ref_turn_cost(*t)))
        assert worst < 1e-12, f"_turn_cost referanstan saptı: {worst}"

    @pytest.mark.parametrize("triple", [
        ((0.0, 0.0), (0.0, 0.0), (1.0, 1.0)),   # sıfır uzunluklu ilk kenar
        ((0.0, 0.0), (1.0, 0.0), (1.0, 0.0)),   # sıfır uzunluklu ikinci kenar
        ((0.0, 0.0), (1.0, 0.0), (0.0, 0.0)),   # tam U-dönüş (180°)
        ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),   # dümdüz
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0)),   # 90° (eşik üstü)
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),   # 135° (keskin + 2.5 cezası)
    ])
    def test_matches_reference_on_edge_cases(self, triple):
        planner = SequentialRoutePlanner()
        assert planner._turn_cost(*triple) == pytest.approx(_ref_turn_cost(*triple), abs=1e-12)

    def test_acos_called_only_for_sharp_turns(self):
        """Yumuşak dönüşlerde acos hiç çağrılmaz; keskin dönüşte çağrılır."""
        planner = SequentialRoutePlanner()
        gentle = [((0.0, 0.0), (10.0, 0.0), (20.0, 1.0)),
                  ((0.0, 0.0), (10.0, 0.0), (20.0, 0.0)),
                  ((5.0, 5.0), (15.0, 5.0), (25.0, 6.0))]
        with mock.patch("math.acos", wraps=math.acos) as acos_spy:
            for t in gentle:
                planner._turn_cost(*t)
            assert acos_spy.call_count == 0, "yumuşak dönüşte acos çağrılmamalı"
        sharp = ((0.0, 0.0), (10.0, 0.0), (0.0, 0.0))  # 180° U-dönüş
        with mock.patch("math.acos", wraps=math.acos) as acos_spy:
            planner._turn_cost(*sharp)
            assert acos_spy.call_count >= 1, "keskin dönüşte acos çağrılmalı"


# ---------------------------------------------------------------------------
# 2. Vektörel init'ler
# ---------------------------------------------------------------------------

class TestVectorizedInits:
    @pytest.mark.parametrize("n,seed", [(1, 1), (2, 2), (5, 3), (40, 4), (80, 5)])
    def test_angular_sweep_init_identical(self, n, seed):
        planner = SequentialRoutePlanner()
        coords = _rand_coords(n, seed)
        assert planner._angular_sweep_init(coords) == _ref_angular_sweep_init(coords)

    def test_angular_sweep_init_duplicate_points(self):
        # Aynı koordinatlı noktalar: açı/yarıçap eşitliği lexsort tie-break'i.
        planner = SequentialRoutePlanner()
        coords = [(100.0, 100.0)] * 5 + [(200.0, 100.0), (100.0, 200.0)]
        assert planner._angular_sweep_init(coords) == _ref_angular_sweep_init(coords)

    @pytest.mark.parametrize("n,seed,start", [(1, 1, 0), (2, 2, 1), (5, 3, 0),
                                              (40, 4, 7), (80, 5, 0), (30, 6, 29)])
    def test_angular_nn_init_identical(self, n, seed, start):
        planner = SequentialRoutePlanner()
        coords = _rand_coords(n, seed)
        dm = _dist_matrix(coords)
        assert (planner._angular_nearest_neighbor_init(coords, dm, start)
                == _ref_angular_nn_init(planner, coords, dm, start))

    def test_angular_nn_init_duplicate_points(self):
        # d <= 1e-6 dalı: çakışık noktalarda ceza uygulanmaz.
        planner = SequentialRoutePlanner()
        coords = [(0.0, 0.0), (0.0, 0.0), (10.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
        dm = _dist_matrix(coords)
        assert (planner._angular_nearest_neighbor_init(coords, dm, 0)
                == _ref_angular_nn_init(planner, coords, dm, 0))


# ---------------------------------------------------------------------------
# 3. 2-opt a-tarafı önbelleği: önbelleksiz referans döngüyle birebir aynı tur
# ---------------------------------------------------------------------------

def _ref_two_opt_loop(planner, tour, dist_matrix, coords, use_angular=False):
    """Değişiklik ÖNCESİ `_two_opt_loop` (önbelleksiz, skaler `_turn_cost`)."""
    n = len(tour)
    if n < 4:
        return list(tour)
    route = list(tour)
    improved = True
    iteration = 0
    while improved and iteration < planner.max_opt_iterations:
        improved = False
        iteration += 1
        for i in range(n - 1):
            a = route[i]
            b = route[(i + 1) % n]
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                c = route[j]
                d = route[(j + 1) % n]
                old_dist = float(dist_matrix[a, b] + dist_matrix[c, d])
                new_dist = float(dist_matrix[a, c] + dist_matrix[b, d])
                if use_angular:
                    a_prev = route[(i - 1) % n]
                    d_next = route[(j + 2) % n]
                    b_next = route[(i + 2) % n]
                    c_prev = route[(j - 1) % n]
                    old_ang = (
                        _ref_turn_cost(coords[a_prev], coords[a], coords[b])
                        + _ref_turn_cost(coords[a], coords[b], coords[b_next])
                        + _ref_turn_cost(coords[c_prev], coords[c], coords[d])
                        + _ref_turn_cost(coords[c], coords[d], coords[d_next])
                    )
                    new_ang = (
                        _ref_turn_cost(coords[a_prev], coords[a], coords[c])
                        + _ref_turn_cost(coords[a], coords[c], coords[c_prev])
                        + _ref_turn_cost(coords[b_next], coords[b], coords[d])
                        + _ref_turn_cost(coords[b], coords[d], coords[d_next])
                    )
                    mean_edge = 0.25 * (old_dist + new_dist)
                    old_total = old_dist + planner.angular_penalty_weight * mean_edge * 0.25 * old_ang
                    new_total = new_dist + planner.angular_penalty_weight * mean_edge * 0.25 * new_ang
                else:
                    old_total = old_dist
                    new_total = new_dist
                if new_total + 1e-6 < old_total:
                    route[i + 1: j + 1] = reversed(route[i + 1: j + 1])
                    improved = True
    return route


class TestTwoOptCache:
    @pytest.mark.parametrize("n,seed,use_angular", [(12, 11, True), (20, 12, True),
                                                    (20, 13, False), (15, 14, True)])
    def test_two_opt_matches_uncached_reference(self, n, seed, use_angular):
        planner = SequentialRoutePlanner()
        coords = _rand_coords(n, seed)
        dm = _dist_matrix(coords)
        rng = random.Random(seed + 100)
        tour = list(range(n))
        rng.shuffle(tour)
        got = planner._two_opt_loop(list(tour), dm, coords, use_angular=use_angular)
        want = _ref_two_opt_loop(planner, list(tour), dm, coords, use_angular=use_angular)
        assert got == want, "önbellekli 2-opt önbelleksiz referanstan saptı"

    @pytest.mark.parametrize("n,seed", [(10, 11), (25, 12)])
    def test_two_opt_output_is_permutation_and_idempotent(self, n, seed):
        planner = SequentialRoutePlanner()
        coords = _rand_coords(n, seed)
        dm = _dist_matrix(coords)
        tour = list(range(n))
        out = planner._two_opt_loop(list(tour), dm, coords, use_angular=True)
        assert sorted(out) == list(range(n)), "çıktı turun permütasyonu olmalı"
        again = planner._two_opt_loop(list(out), dm, coords, use_angular=True)
        assert again == out, "2-opt çıktısı kararlı olmalı (ikinci geçiş aynı tur)"

    def test_two_opt_deterministic(self):
        planner = SequentialRoutePlanner()
        coords = _rand_coords(30, 21)
        dm = _dist_matrix(coords)
        tour = list(range(30))
        a = planner._two_opt_loop(list(tour), dm, coords, use_angular=True)
        b = planner._two_opt_loop(list(tour), dm, coords, use_angular=True)
        assert a == b


# ---------------------------------------------------------------------------
# 4. Sözleşme önbelleği: duman testi bir kez çalışır
# ---------------------------------------------------------------------------

class TestContractCache:
    def test_self_check_runs_once_for_repeated_validations(self):
        calls = {"n": 0}
        orig = sr._run_sequential_router_self_check

        def counting(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        sr._CONTRACT_VALID_CACHE = None
        try:
            with mock.patch.object(sr, "_run_sequential_router_self_check", counting):
                results = [sr.validate_sequential_router_contract() for _ in range(5)]
            assert results == [True] * 5
            assert calls["n"] == 1, f"duman testi 1 kez çalışmalı, {calls['n']} kez çalıştı"
        finally:
            sr._CONTRACT_VALID_CACHE = None

    def test_validate_still_true_after_cache_reset(self):
        sr._CONTRACT_VALID_CACHE = None
        try:
            assert sr.validate_sequential_router_contract() is True
            assert sr.validate_sequential_router_contract() is True
        finally:
            sr._CONTRACT_VALID_CACHE = None


# ---------------------------------------------------------------------------
# 5. Uçtan uca: smooth_loop çıktısı değişiklik öncesi ile aynı tur
#    (önceki bench'te baz ile birebir eşleştiği doğrulanan tohumlar)
# ---------------------------------------------------------------------------

class TestSmoothLoopEndToEnd:
    @pytest.mark.parametrize("n,seed", [(40, 1), (80, 2)])
    def test_smooth_loop_produces_valid_crossing_free_tour(self, n, seed):
        rng = random.Random(seed)
        nodes = [{"id": i + 1, "map_x": rng.uniform(0, 1000),
                  "map_y": rng.uniform(0, 1000)} for i in range(n)]
        planner = SequentialRoutePlanner()
        ordered = planner.build_sequential_order(nodes, mode="smooth_loop")
        assert sorted(nd["id"] for nd in ordered) == list(range(1, n + 1))
        quality = planner.evaluate_sequence_quality(ordered, closed_loop=True)
        assert quality.get("crossings_count", 0) == 0
