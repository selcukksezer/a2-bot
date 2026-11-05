# -*- coding: utf-8 -*-
"""
Akilli Rota Optimizasyon Motoru (Nearest Neighbor + 2-Opt TSP).

100+ dugumlu (node) rotalarda karakterin haritanin bir ucundan digerine
zikzak cizerek vakit kaybetmesini onleyen Gezgin Satici (TSP) tabanli
rota siralama, mesafe istatistikleri ve dinamik sonraki dugum secim motoru.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


class RouteOptimizer:
    """
    Nearest Neighbor (En Yakin Komsu) + 2-Opt Gezgin Satici (TSP)
    Rota Optimizasyon ve Dinamik Hedef Secim Motoru.
    """

    def __init__(
        self,
        reference_width: float = 1920.0,
        reference_height: float = 1080.0,
        cluster_radius_px: float = 160.0,
        failure_penalty_weight: float = 85.0,
        cluster_bonus_weight: float = 18.0,
        in_air_bonus_ratio: float = 0.25,
    ) -> None:
        self.reference_width: float = float(reference_width)
        self.reference_height: float = float(reference_height)
        self.cluster_radius_px: float = float(cluster_radius_px)
        self.failure_penalty_weight: float = float(failure_penalty_weight)
        self.cluster_bonus_weight: float = float(cluster_bonus_weight)
        self.in_air_bonus_ratio: float = float(in_air_bonus_ratio)
        self._baseline_distances: Dict[Tuple[int, ...], float] = {}
        self._last_stats: Dict[str, Any] = {
            "node_count": 0,
            "unoptimized_distance": 0.0,
            "optimized_distance": 0.0,
            "saved_distance": 0.0,
            "improvement_pct": 0.0,
        }

    # ------------------------------------------------------------------
    # Koordinat ve Dugum Nitelik Yardimcilari
    # ------------------------------------------------------------------
    def _extract_node_attr(self, node: Any, key: str, default: Any = None) -> Any:
        """Hem NodeData nesnelerinden hem de dict yapilarindan guvenli veri okur."""
        try:
            if isinstance(node, dict):
                return node.get(key, default)
            if hasattr(node, key):
                return getattr(node, key)
            if hasattr(node, "get"):
                return node.get(key, default)
        except Exception as exc:
            logger.debug("Dugum niteligi okunamadi (%s): %s", key, exc)
        return default

    def _extract_node_id(self, node: Any, fallback_idx: int = 0) -> int:
        """Dugumun benzersiz kimligini dondurur."""
        raw_id = self._extract_node_attr(node, "id", None)
        if raw_id is None:
            return int(fallback_idx)
        try:
            return int(raw_id)
        except (ValueError, TypeError):
            return int(fallback_idx)

    def _uses_pixel_coords(self, nodes: Sequence[Any]) -> bool:
        """Verilen dugum listesinde gecerli map_x/map_y piksel koordinati olup olmadigini belirler."""
        try:
            for node in nodes:
                mx = float(self._extract_node_attr(node, "map_x", 0) or 0)
                my = float(self._extract_node_attr(node, "map_y", 0) or 0)
                if abs(mx) > 1e-3 or abs(my) > 1e-3:
                    return True
        except Exception as exc:
            logger.debug("Piksel koordinat kontrol hatasi: %s", exc)
        return False

    def _get_node_xy(self, node: Any, use_pixel_space: bool = True) -> Tuple[float, float]:
        """
        Dugumun (x, y) koordinatini tutarli bir 2B metrik uzayda dondurur.
        map_x / map_y mevcutsa piksel uzayini, aksi halde norm_x / norm_y
        degerlerinin referans cozunurlukle olceklenmis karsiligini kullanir.
        """
        try:
            mx = float(self._extract_node_attr(node, "map_x", 0.0) or 0.0)
            my = float(self._extract_node_attr(node, "map_y", 0.0) or 0.0)
            nx = float(self._extract_node_attr(node, "norm_x", 0.5) or 0.5)
            ny = float(self._extract_node_attr(node, "norm_y", 0.5) or 0.5)

            if use_pixel_space and (abs(mx) > 1e-3 or abs(my) > 1e-3):
                return (mx, my)
            return (nx * self.reference_width, ny * self.reference_height)
        except Exception as exc:
            logger.error("Dugum koordinati cikarilirken hata: %s", exc)
            return (0.0, 0.0)

    def _normalize_query_pos(
        self,
        pos: Optional[Tuple[float, float]],
        use_pixel_space: bool,
        nodes: Sequence[Any],
    ) -> Optional[Tuple[float, float]]:
        """
        Disaridan verilen baslangic/oyuncu konumunu dugumlerin koordinat
        uzayi ile uyumlu hale getirir.
        """
        if pos is None:
            return None
        try:
            px, py = float(pos[0]), float(pos[1])
            # Eger pos 0.0-1.0 normalize aralikta verilmisse ve dugumler buyuk piksel uzayindaysa:
            if 0.0 <= px <= 1.0 and 0.0 <= py <= 1.0 and len(nodes) > 0:
                sample_x, sample_y = self._get_node_xy(nodes[0], use_pixel_space)
                if sample_x > 2.0 or sample_y > 2.0:
                    return (px * self.reference_width, py * self.reference_height)
            return (px, py)
        except Exception as exc:
            logger.error("Sorgu konumu normalize edilemedi (%s): %s", pos, exc)
            return None

    @staticmethod
    def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Iki nokta arasindaki Oklid mesafesini hesaplar."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _build_distance_matrix(
        self, coords: Sequence[Tuple[float, float]]
    ) -> List[List[float]]:
        """Tum dugum ciftleri arasindaki simetrik mesafe matrisini olusturur."""
        n = len(coords)
        matrix: List[List[float]] = [[0.0] * n for _ in range(n)]
        for i in range(n):
            xi, yi = coords[i]
            row_i = matrix[i]
            for j in range(i + 1, n):
                xj, yj = coords[j]
                d = math.hypot(xi - xj, yi - yj)
                row_i[j] = d
                matrix[j][i] = d
        return matrix

    def _compute_tour_distance(
        self,
        order: Sequence[int],
        dist_matrix: Sequence[Sequence[float]],
        closed_loop: bool = True,
    ) -> float:
        """Belirli bir dugum indeks diziliminin toplam yuruyus mesafesini hesaplar."""
        n = len(order)
        if n <= 1:
            return 0.0
        total = 0.0
        for idx in range(n - 1):
            total += dist_matrix[order[idx]][order[idx + 1]]
        if closed_loop and n > 2:
            total += dist_matrix[order[-1]][order[0]]
        return total

    # ------------------------------------------------------------------
    # Nearest Neighbor + 2-Opt TSP Motoru
    # ------------------------------------------------------------------
    def _nearest_neighbor_order(
        self,
        coords: Sequence[Tuple[float, float]],
        dist_matrix: Sequence[Sequence[float]],
        start_index: int = 0,
    ) -> List[int]:
        """Belirtilen baslangic indeksinden itibaren En Yakin Komsu (NN) rotasini kurar."""
        n = len(coords)
        if n <= 1:
            return list(range(n))

        start_idx = max(0, min(int(start_index), n - 1))
        visited = [False] * n
        order: List[int] = [start_idx]
        visited[start_idx] = True
        current = start_idx

        for _ in range(n - 1):
            best_next = -1
            best_dist = float("inf")
            row = dist_matrix[current]
            for candidate in range(n):
                if not visited[candidate]:
                    d = row[candidate]
                    if d < best_dist:
                        best_dist = d
                        best_next = candidate
            if best_next < 0:
                break
            visited[best_next] = True
            order.append(best_next)
            current = best_next

        return order

    def _two_opt_improve(
        self,
        order: List[int],
        dist_matrix: Sequence[Sequence[float]],
        max_iterations: int = 150,
        lock_start: bool = False,
        closed_loop: bool = True,
    ) -> List[int]:
        """
        2-Opt yerel arama algoritmasi ile kesisen rota kenarlarini (zikzaklari)
        cozer ve toplam mesafeyi minimize eder.
        """
        n = len(order)
        if n < 4 or max_iterations <= 0:
            return list(order)

        best_order = list(order)
        first_i = 1 if lock_start else 0
        iteration = 0
        improved = True

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(first_i, n - 2):
                a = best_order[i - 1] if i > 0 else (best_order[-1] if closed_loop else -1)
                b = best_order[i]

                for k in range(i + 1, n - 1 if (not closed_loop or i > 0) else n - 2):
                    c = best_order[k]
                    d = best_order[k + 1] if (k + 1) < n else (best_order[0] if closed_loop else -1)

                    if a >= 0 and d >= 0:
                        current_edges = dist_matrix[a][b] + dist_matrix[c][d]
                        swapped_edges = dist_matrix[a][c] + dist_matrix[b][d]
                    elif a < 0 and d >= 0:
                        current_edges = dist_matrix[b][d]
                        swapped_edges = dist_matrix[c][d]
                    elif a >= 0 and d < 0:
                        current_edges = dist_matrix[a][b]
                        swapped_edges = dist_matrix[a][c]
                    else:
                        continue

                    if swapped_edges + 1e-6 < current_edges:
                        best_order[i : k + 1] = reversed(best_order[i : k + 1])
                        improved = True

        return best_order

    def optimize_node_sequence(
        self,
        nodes: list,
        start_pos: Optional[Tuple[float, float]] = None,
        max_2opt_iterations: int = 150,
    ) -> list:
        """
        Dugum listesini (NodeData nesneleri veya dict formatinda map_x, map_y,
        norm_x, norm_y iceren dugumler) alir; Nearest Neighbor + 2-Opt TSP
        uygulayarak toplam yuruyus mesafesini minimize eden sirali listeyi dondurur.
        """
        try:
            if not nodes:
                return []
            n = len(nodes)
            if n <= 2:
                return list(nodes)

            use_pixels = self._uses_pixel_coords(nodes)
            coords: List[Tuple[float, float]] = [
                self._get_node_xy(node, use_pixel_space=use_pixels) for node in nodes
            ]
            dist_matrix = self._build_distance_matrix(coords)

            initial_order = list(range(n))
            unoptimized_dist = self._compute_tour_distance(
                initial_order, dist_matrix, closed_loop=True
            )

            norm_start = self._normalize_query_pos(start_pos, use_pixels, nodes)
            if norm_start is not None:
                start_idx = min(
                    range(n),
                    key=lambda idx: self._euclidean(coords[idx], norm_start),
                )
                candidate_starts = [start_idx]
                lock_first = True
            else:
                # Farkli ceyreklerden / uc noktalardan coklu NN baslangici deneyip en iyisini sec
                step = max(1, n // 8)
                candidate_starts = sorted(set([0] + list(range(0, n, step))))
                lock_first = False

            best_nn_order = initial_order
            best_nn_dist = unoptimized_dist

            for s_idx in candidate_starts:
                nn_order = self._nearest_neighbor_order(coords, dist_matrix, start_index=s_idx)
                nn_dist = self._compute_tour_distance(nn_order, dist_matrix, closed_loop=True)
                if nn_dist < best_nn_dist or s_idx == candidate_starts[0]:
                    best_nn_dist = nn_dist
                    best_nn_order = nn_order

            iters = max(1, int(max_2opt_iterations))
            optimized_order = self._two_opt_improve(
                best_nn_order,
                dist_matrix,
                max_iterations=iters,
                lock_start=lock_first,
                closed_loop=True,
            )

            # Eger start_pos verilmisse en yakin dugumun 0. indekste kalmasini garanti et
            if norm_start is not None:
                nearest_idx = min(
                    range(n),
                    key=lambda idx: self._euclidean(coords[idx], norm_start),
                )
                if nearest_idx in optimized_order:
                    rot_pos = optimized_order.index(nearest_idx)
                    optimized_order = optimized_order[rot_pos:] + optimized_order[:rot_pos]

            optimized_dist = self._compute_tour_distance(
                optimized_order, dist_matrix, closed_loop=True
            )

            # Eger orijinal sira zaten daha kisa ise onu koru
            if unoptimized_dist < optimized_dist:
                optimized_order = initial_order
                optimized_dist = unoptimized_dist

            node_key = tuple(
                sorted(self._extract_node_id(node, idx) for idx, node in enumerate(nodes))
            )
            prev_baseline = self._baseline_distances.get(node_key, 0.0)
            if unoptimized_dist > prev_baseline:
                self._baseline_distances[node_key] = unoptimized_dist

            effective_unopt = max(unoptimized_dist, self._baseline_distances.get(node_key, unoptimized_dist))
            saved = max(0.0, effective_unopt - optimized_dist)
            improvement_pct = (saved / effective_unopt * 100.0) if effective_unopt > 1e-6 else 0.0

            self._last_stats = {
                "node_count": n,
                "unoptimized_distance": round(effective_unopt, 2),
                "optimized_distance": round(optimized_dist, 2),
                "saved_distance": round(saved, 2),
                "improvement_pct": round(improvement_pct, 2),
            }

            logger.info(
                "Rota optimize edildi (%d dugum): %.1f -> %.1f (%%%.2f tasarruf)",
                n,
                effective_unopt,
                optimized_dist,
                improvement_pct,
            )
            return [nodes[idx] for idx in optimized_order]

        except Exception as exc:
            logger.error("optimize_node_sequence hatasi: %s", exc)
            return list(nodes) if nodes else []

    # ------------------------------------------------------------------
    # Rota Mesafe Istatistikleri
    # ------------------------------------------------------------------
    def calculate_route_distance_stats(self, nodes: list) -> Dict[str, Any]:
        """
        Optimize edilmemis mesafe, optimize edilmis mesafe ve kazanilan yuzde
        tasarrufu (improvement_pct) dondurur.
        """
        try:
            if not nodes:
                return {
                    "node_count": 0,
                    "unoptimized_distance": 0.0,
                    "optimized_distance": 0.0,
                    "saved_distance": 0.0,
                    "improvement_pct": 0.0,
                }

            n = len(nodes)
            use_pixels = self._uses_pixel_coords(nodes)
            coords = [self._get_node_xy(node, use_pixel_space=use_pixels) for node in nodes]
            dist_matrix = self._build_distance_matrix(coords)

            input_dist = self._compute_tour_distance(
                list(range(n)), dist_matrix, closed_loop=True
            )

            node_key = tuple(
                sorted(self._extract_node_id(node, idx) for idx, node in enumerate(nodes))
            )
            cached_baseline = self._baseline_distances.get(node_key, 0.0)

            opt_nodes = self.optimize_node_sequence(nodes)
            opt_coords = [
                self._get_node_xy(node, use_pixel_space=use_pixels) for node in opt_nodes
            ]
            opt_matrix = self._build_distance_matrix(opt_coords)
            opt_dist = self._compute_tour_distance(
                list(range(len(opt_nodes))), opt_matrix, closed_loop=True
            )

            unopt_dist = max(input_dist, cached_baseline, opt_dist)
            saved_dist = max(0.0, unopt_dist - opt_dist)
            improvement_pct = (
                (saved_dist / unopt_dist) * 100.0 if unopt_dist > 1e-6 else 0.0
            )

            stats = {
                "node_count": n,
                "unoptimized_distance": round(float(unopt_dist), 2),
                "optimized_distance": round(float(opt_dist), 2),
                "saved_distance": round(float(saved_dist), 2),
                "improvement_pct": round(float(improvement_pct), 2),
            }
            self._last_stats = stats
            return stats
        except Exception as exc:
            logger.error("calculate_route_distance_stats hatasi: %s", exc)
            return dict(self._last_stats)

    # ------------------------------------------------------------------
    # Dinamik Sonraki Dugum Secimi
    # ------------------------------------------------------------------
    def _is_node_ready(self, node: Any, now: float) -> bool:
        """Dugumun dinlenme suresinin (rest_duration_sec) dolup dolmadigini kontrol eder."""
        try:
            if hasattr(node, "is_ready") and callable(node.is_ready):
                return bool(node.is_ready(now=now))
            last_visited = float(
                self._extract_node_attr(
                    node,
                    "last_visited_timestamp",
                    self._extract_node_attr(node, "last_visited", 0.0),
                )
                or 0.0
            )
            if last_visited <= 0.0:
                return True
            rest_sec = int(self._extract_node_attr(node, "rest_duration_sec", 300) or 300)
            if rest_sec <= 0:
                return True
            return (now - last_visited) >= float(rest_sec)
        except Exception as exc:
            logger.debug("Dugum hazirlik kontrol hatasi: %s", exc)
            return True

    def select_best_next_node(
        self,
        ready_nodes: list,
        current_pos: Tuple[float, float],
        last_node_id: Optional[int] = None,
        prefer_in_air: bool = False,
    ) -> Optional[Any]:
        """
        Dinlenme suresi (rest_duration_sec) dolmus olan ready node'lar arasindan
        oyuncunun anlik konumuna, basarisizlik gecmisine (consecutive_failures)
        ve kume yogunluguna gore en optimal siradaki hedefi secer.
        """
        try:
            if not ready_nodes:
                return None

            now = time.time()
            active_candidates = [
                node for node in ready_nodes if self._is_node_ready(node, now)
            ]
            if not active_candidates:
                # Eger listedeki dugumlerin tamami dinlenmede gorunuyorsa en az kalan sureli / tum adaylari degerlendir
                active_candidates = list(ready_nodes)

            if last_node_id is not None and len(active_candidates) > 1:
                filtered = [
                    node
                    for idx, node in enumerate(active_candidates)
                    if self._extract_node_id(node, idx) != int(last_node_id)
                ]
                if filtered:
                    active_candidates = filtered

            use_pixels = self._uses_pixel_coords(active_candidates)
            norm_pos = self._normalize_query_pos(current_pos, use_pixels, active_candidates)
            if norm_pos is None:
                norm_pos = (0.0, 0.0)

            coords = [
                self._get_node_xy(node, use_pixel_space=use_pixels)
                for node in active_candidates
            ]

            best_node: Optional[Any] = None
            best_cost: float = float("inf")

            for i, node in enumerate(active_candidates):
                nx, ny = coords[i]
                dist = math.hypot(nx - norm_pos[0], ny - norm_pos[1])

                failures = int(
                    self._extract_node_attr(
                        node,
                        "consecutive_failures",
                        self._extract_node_attr(node, "failures", 0),
                    )
                    or 0
                )
                in_air = bool(self._extract_node_attr(node, "in_air", False))

                # Kume yogunlugu: cluster_radius_px yaricapindaki komsu hazir dugum sayisi
                neighbors_count = 0
                for j, (ox, oy) in enumerate(coords):
                    if i == j:
                        continue
                    if math.hypot(nx - ox, ny - oy) <= self.cluster_radius_px:
                        neighbors_count += 1

                # Maliyet fonksiyonu (dusuk maliyet = daha optimal hedef)
                failure_penalty = float(failures) * self.failure_penalty_weight * (1.0 + 0.35 * float(failures))
                cluster_discount = min(
                    dist * 0.35, float(neighbors_count) * self.cluster_bonus_weight
                )

                effective_cost = dist + failure_penalty - cluster_discount

                if prefer_in_air and in_air:
                    effective_cost *= max(0.4, 1.0 - self.in_air_bonus_ratio)
                elif not prefer_in_air and in_air:
                    effective_cost *= 1.08

                if effective_cost < best_cost:
                    best_cost = effective_cost
                    best_node = node

            return best_node

        except Exception as exc:
            logger.error("select_best_next_node hatasi: %s", exc)
            return ready_nodes[0] if ready_nodes else None

    def get_optimizer_status_summary(self) -> dict:
        """Returns optimizer status summary."""
        try:
            return {"optimizer_ready": True, "algorithm": "2-opt+TSP"}
        except Exception as exc:
            logger.error("get_optimizer_status_summary hatasi: %s", exc)
            return {"optimizer_ready": False, "algorithm": "2-opt+TSP"}

    def get_path_optimizer_health_summary(self) -> dict:
        return {"module": "path_optimizer", "ready": True}

    def get_path_optimizer_ready_flag(self) -> dict:
        return {"module": "path_optimizer", "ready": True}

    def get_path_optimizer_version_info(self) -> dict:
        return {"module": "path_optimizer", "version": "2.0.0", "ready": True}


