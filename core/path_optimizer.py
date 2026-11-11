# -*- coding: utf-8 -*-
"""
Akilli Rota Optimizasyon Motoru (Nearest Neighbor + 2-Opt TSP).

100+ dugumlu (node) rotalarda karakterin haritanin bir ucundan digerine
zikzak cizerek vakit kaybetmesini onleyen Gezgin Satici (TSP) tabanli
rota siralama, mesafe istatistikleri ve dinamik sonraki dugum secim motoru.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from version import __version__

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

    def _get_node_xy(self, node: Any, use_pixel_space: bool = True) -> Optional[Tuple[float, float]]:
        """
        Dugumun (x, y) koordinatini tutarli bir 2B metrik uzayda dondurur.
        map_x / map_y mevcutsa piksel uzayini, aksi halde norm_x / norm_y
        degerlerinin referans cozunurlukle olceklenmis karsiligini kullanir.

        P1-7 (K-20): koordinat cozulemiyorsa `(0.0, 0.0)` hayaleti
        dondurmez — `None` doner. "Koordinat yok" sayilan haller:
        piksel (0, 0) + norm eksik / norm (0.5, 0.5) yer tutucu.
        Acikca verilmis norm (0, 0) gercek kose koordinatidir, reddedilmez.
        """
        try:
            mx = float(self._extract_node_attr(node, "map_x", 0.0) or 0.0)
            my = float(self._extract_node_attr(node, "map_y", 0.0) or 0.0)
            # F-1898: norm_x/norm_y = 0 geçerli bir köşe koordinatıdır;
            # `or 0.5` onu merkeze kaydırırdı. Yalnızca None/eksik 0.5 sayılır.
            raw_nx = self._extract_node_attr(node, "norm_x", None)
            raw_ny = self._extract_node_attr(node, "norm_y", None)
            nx = 0.5 if raw_nx is None else float(raw_nx)
            ny = 0.5 if raw_ny is None else float(raw_ny)

            if use_pixel_space and (abs(mx) > 1e-3 or abs(my) > 1e-3):
                return (mx, my)
            # Piksel yok + norm yer tutucu/eksik: koordinat bilinmiyor.
            if raw_nx is None or raw_ny is None or (nx == 0.5 and ny == 0.5):
                return None
            return (nx * self.reference_width, ny * self.reference_height)
        except Exception as exc:
            logger.error("Dugum koordinati cikarilirken hata: %s", exc)
            return None

    def _resolved_coords(
        self, nodes: Sequence[Any], use_pixel_space: bool = True
    ) -> Tuple[List[Any], List[Tuple[float, float]]]:
        """
        Koordinati cozulebilen dugumleri ve koordinatlarini dondurur.

        P1-7 (K-20): koordinatsiz dugum TSP/mesafe hesabina (0,0)
        hayaleti olarak girmez — listeden cikarilir ve sayisi loglanir.
        """
        kept_nodes: List[Any] = []
        coords: List[Tuple[float, float]] = []
        dropped = 0
        for node in nodes:
            xy = self._get_node_xy(node, use_pixel_space=use_pixel_space)
            if xy is None:
                dropped += 1
                continue
            kept_nodes.append(node)
            coords.append(xy)
        if dropped:
            logger.warning(
                "Koordinatsiz %d dugum rota hesabindan cikarildi.", dropped
            )
        return kept_nodes, coords

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
                # P1-7 (K-20): ornek dugum koordinatsizsa (None) olcek
                # cikarimi yapilamaz — konum oldugu gibi birakilir.
                sample_xy = self._get_node_xy(nodes[0], use_pixel_space)
                if sample_xy is not None and (sample_xy[0] > 2.0 or sample_xy[1] > 2.0):
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

    @staticmethod
    def _segments_intersect(
        a1: Tuple[float, float],
        a2: Tuple[float, float],
        b1: Tuple[float, float],
        b2: Tuple[float, float],
    ) -> bool:
        """İki 2B doğru parçasının uç noktalar hariç kesişip kesişmediğini döndürür."""
        def _ccw(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
            return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

        eps = 1e-6
        eps_sq = eps * eps

        def _sq_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
            # Karesel mesafe: hypot(p1, p2) < eps  <=>  sq < eps^2
            # (ikisi de negatif değil). Kesişim sayımında çift başına 4
            # sqrt çağrısı kalkar; predikat aynı kalır.
            dx = p1[0] - p2[0]
            dy = p1[1] - p2[1]
            return dx * dx + dy * dy

        if (
            _sq_dist(a1, b1) < eps_sq
            or _sq_dist(a1, b2) < eps_sq
            or _sq_dist(a2, b1) < eps_sq
            or _sq_dist(a2, b2) < eps_sq
        ):
            return False

        d1 = _ccw(a1, a2, b1)
        d2 = _ccw(a1, a2, b2)
        d3 = _ccw(b1, b2, a1)
        d4 = _ccw(b1, b2, a2)

        if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and (
            (d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)
        ):
            return True
        return False

    def count_crossings(
        self,
        nodes: Sequence[Any],
        closed_loop: bool = True,
    ) -> int:
        """Rotadaki kendi kendini kesen kenar çifti sayısını döndürür."""
        if not nodes or len(nodes) < 4:
            return 0
        use_pixels = self._uses_pixel_coords(nodes)
        # P1-7 (K-20): koordinatsiz dugum kesisim sayimina (0,0)
        # hayaleti olarak girmez.
        _, coords = self._resolved_coords(nodes, use_pixel_space=use_pixels)
        if len(coords) < 4:
            return 0
        return self._count_coords_crossings(coords, closed_loop=closed_loop)

    def _count_coords_crossings(
        self,
        coords: Sequence[Tuple[float, float]],
        closed_loop: bool = True,
    ) -> int:
        n = len(coords)
        if n < 4:
            return 0
        edges = []
        num_e = n if closed_loop else (n - 1)
        for i in range(num_e):
            edges.append((coords[i], coords[(i + 1) % n]))

        crossings = 0
        for i in range(num_e):
            a1, a2 = edges[i]
            for j in range(i + 2, num_e):
                if closed_loop and i == 0 and j == num_e - 1:
                    continue
                b1, b2 = edges[j]
                if self._segments_intersect(a1, a2, b1, b2):
                    crossings += 1
        return crossings

    def is_non_intersecting(
        self,
        nodes: Sequence[Any],
        closed_loop: bool = True,
    ) -> bool:
        """Rotanın kesişimsiz (non-intersecting) olduğunu doğrular."""
        return self.count_crossings(nodes, closed_loop=closed_loop) == 0

    def _eliminate_crossings(
        self,
        order: List[int],
        coords: Sequence[Tuple[float, float]],
        closed_loop: bool = True,
        max_passes: int = 60,
    ) -> List[int]:
        """Kesişen kenarları 2-Opt ters çevirme ile gidermeye çalışır.

        F-1904: eski docstring "tamamen ortadan kaldırır" diyordu ama döngü
        max_passes ile sınırlıdır; kesişim kalırsa uyarı loglanır ve kalan
        kesişim sayısı optimize_node_sequence istatistiklerine yazılır.
        Garanti için is_non_intersecting() ile doğrulayın.
        """
        route, _remaining = self._eliminate_crossings_counted(
            order, coords, closed_loop=closed_loop, max_passes=max_passes
        )
        return route

    def _eliminate_crossings_counted(
        self,
        order: List[int],
        coords: Sequence[Tuple[float, float]],
        closed_loop: bool = True,
        max_passes: int = 60,
    ) -> Tuple[List[int], int]:
        """_eliminate_crossings ile aynı işi yapar; ayrıca sonunda kalan
        kesişim sayısını döndürür, böylece çağıran aynı O(n^2) sayımı
        tekrar yapmak zorunda kalmaz (optimize_node_sequence kullanır)."""
        n = len(order)
        if n < 4:
            return list(order), 0

        route = list(order)
        num_edges = n if closed_loop else (n - 1)
        for _ in range(min(max_passes, n * 3)):
            found_crossing = False
            for i in range(num_edges - 2):
                a1 = coords[route[i]]
                a2 = coords[route[(i + 1) % n]]
                for j in range(i + 2, num_edges):
                    if closed_loop and i == 0 and j == num_edges - 1:
                        continue
                    b1 = coords[route[j]]
                    b2 = coords[route[(j + 1) % n]]
                    if self._segments_intersect(a1, a2, b1, b2):
                        route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                        found_crossing = True
                        break
                if found_crossing:
                    break
            if not found_crossing:
                break
        remaining = self._count_coords_crossings(
            [coords[i] for i in route], closed_loop=closed_loop
        )
        if remaining > 0:
            logger.warning(
                "Kesişim giderme %d geçişte tamamlanamadı; %d kesişim kaldı.",
                min(max_passes, n * 3),
                remaining,
            )
        return route, remaining

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
                for k in range(i + 1, n - 1 if (not closed_loop or i > 0) else n - 2):
                    # F-1894: takas sonrası best_order[i] (b) değişir; a/b her
                    # k adımında yeniden okunur, yoksa takas bayat değerle
                    # değerlendirilirdi.
                    a = best_order[i - 1] if i > 0 else (best_order[-1] if closed_loop else -1)
                    b = best_order[i]
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

    @staticmethod
    def _set_seq_index(node_item: Any, seq_idx: int) -> bool:
        """F-1912: seq_index yazılamaması artık sessizce geçilmez."""
        try:
            if isinstance(node_item, dict):
                node_item["seq_index"] = seq_idx
                return True
            if hasattr(node_item, "seq_index"):
                node_item.seq_index = seq_idx
                return True
            if hasattr(node_item, "__setitem__"):
                node_item["seq_index"] = seq_idx
                return True
        except Exception as exc:
            logger.warning("seq_index yazılamadı (node=%r): %s", node_item, exc)
            return False
        logger.warning("seq_index yazılamadı: düğüm ne dict ne de seq_index niteliği taşıyor (%r)", node_item)
        return False

    def optimize_node_sequence(
        self,
        nodes: list,
        start_pos: Optional[Tuple[float, float]] = None,
        max_2opt_iterations: int = 150,
        closed_loop: bool = True,
    ) -> list:
        """
        Dugum listesini (NodeData nesneleri veya dict formatinda map_x, map_y,
        norm_x, norm_y iceren dugumler) alir; Nearest Neighbor + 2-Opt TSP
        uygulayarak kesişimleri azaltan ve toplam mesafeyi minimize eden
        sirali listeyi dondurur. Kesişimsizlik garanti edilmez; doğrulama için
        is_non_intersecting() kullanın. Her dugumun seq_index degerini (1..N) gunceller.
        F-1897: kapalı tur artık parametredir (varsayılan kapalı tur).
        """
        try:
            if not nodes:
                return []
            # P1-7 (K-20): koordinatsiz dugum TSP turuna (0,0) hayaleti
            # olarak girmez — turdan cikarilir (n<=2 kisa yolundan once).
            use_pixels = self._uses_pixel_coords(nodes)
            nodes, coords = self._resolved_coords(
                list(nodes), use_pixel_space=use_pixels
            )
            n = len(nodes)
            if n == 0:
                return []
            if n <= 2:
                result_nodes = list(nodes)
                for idx, node_item in enumerate(result_nodes, start=1):
                    self._set_seq_index(node_item, idx)
                return result_nodes
            dist_matrix = self._build_distance_matrix(coords)

            initial_order = list(range(n))
            unoptimized_dist = self._compute_tour_distance(
                initial_order, dist_matrix, closed_loop=closed_loop
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

            # F-1906: ilk NN adayı koşulsuz kabul edilmezdi; en iyi NN
            # gerçekten en kısa tur mesafesine sahip olandır.
            best_nn_order = initial_order
            best_nn_dist = float("inf")

            for s_idx in candidate_starts:
                nn_order = self._nearest_neighbor_order(coords, dist_matrix, start_index=s_idx)
                nn_dist = self._compute_tour_distance(nn_order, dist_matrix, closed_loop=closed_loop)
                if nn_dist < best_nn_dist:
                    best_nn_dist = nn_dist
                    best_nn_order = nn_order

            iters = max(1, int(max_2opt_iterations))
            optimized_order = self._two_opt_improve(
                best_nn_order,
                dist_matrix,
                max_iterations=iters,
                lock_start=lock_first,
                closed_loop=closed_loop,
            )

            # Kesişen kenarları azalt (garanti vermez; bkz. _eliminate_crossings).
            # Sayılı sürüm kalan kesişim sayısını da döndürür; aşağıdaki
            # opt_crossings hesabı için aynı O(n^2) sayım tekrarlanmaz.
            optimized_order, elim_remaining = self._eliminate_crossings_counted(
                optimized_order,
                coords,
                closed_loop=closed_loop,
            )

            # Eger start_pos verilmisse en yakin dugumun 0. indekste kalmasini garanti et
            rotated = False
            if norm_start is not None:
                nearest_idx = min(
                    range(n),
                    key=lambda idx: self._euclidean(coords[idx], norm_start),
                )
                if nearest_idx in optimized_order:
                    rot_pos = optimized_order.index(nearest_idx)
                    if rot_pos != 0:
                        optimized_order = optimized_order[rot_pos:] + optimized_order[:rot_pos]
                        rotated = True

            optimized_dist = self._compute_tour_distance(
                optimized_order, dist_matrix, closed_loop=closed_loop
            )

            # Eger orijinal sira zaten daha kisa ise ve kesişim içermiyorsa onu koru
            # Döndürme (rotasyon) kapalı turda kenar kümesini değiştirmez,
            # açık turda değiştirir; o yüzden açık tur + döndürme durumunda
            # kalan kesişim yeniden sayılır, diğer durumlarda elim_remaining
            # olduğu gibi kullanılır.
            if rotated and not closed_loop:
                opt_crossings = self._count_coords_crossings([coords[i] for i in optimized_order], closed_loop=closed_loop)
            else:
                opt_crossings = elim_remaining
            init_crossings = self._count_coords_crossings(coords, closed_loop=closed_loop)
            if unoptimized_dist < optimized_dist and (init_crossings <= opt_crossings):
                optimized_order = initial_order
                optimized_dist = unoptimized_dist
                opt_crossings = init_crossings

            # F-1893: tasarruf istatistiği tarihî tabanla şişirilmezdi;
            # baz her zaman BU çalıştırmanın girdi sırası mesafesidir.
            saved = max(0.0, unoptimized_dist - optimized_dist)
            improvement_pct = (saved / unoptimized_dist * 100.0) if unoptimized_dist > 1e-6 else 0.0

            self._last_stats = {
                "node_count": n,
                "unoptimized_distance": round(unoptimized_dist, 2),
                "optimized_distance": round(optimized_dist, 2),
                "saved_distance": round(saved, 2),
                "improvement_pct": round(improvement_pct, 2),
                "closed_loop": bool(closed_loop),
                "crossings_remaining": int(opt_crossings),
            }

            logger.info(
                "Rota optimize edildi (%d dugum): %.1f -> %.1f (%%%.2f tasarruf)",
                n,
                unoptimized_dist,
                optimized_dist,
                improvement_pct,
            )

            ordered_nodes = [nodes[idx] for idx in optimized_order]
            for seq_idx, node_item in enumerate(ordered_nodes, start=1):
                self._set_seq_index(node_item, seq_idx)

            return ordered_nodes

        except Exception as exc:
            logger.error("optimize_node_sequence hatasi: %s", exc)
            fallback_nodes = list(nodes) if nodes else []
            for seq_idx, node_item in enumerate(fallback_nodes, start=1):
                self._set_seq_index(node_item, seq_idx)
            return fallback_nodes

    # ------------------------------------------------------------------
    # Rota Mesafe Istatistikleri
    # ------------------------------------------------------------------
    def calculate_route_distance_stats(
        self, nodes: list, closed_loop: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize edilmemis mesafe, optimize edilmis mesafe ve kazanilan yuzde
        tasarrufu (improvement_pct) dondurur. Baz bu çalıştırmanın girdi
        sırasıdır (F-1893: tarihî taban kullanılmaz).
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
            # P1-7 (K-20): koordinatsiz dugum mesafe istatistigine (0,0)
            # hayaleti olarak girmez; node_count cozulebilen dugum sayisidir.
            nodes, coords = self._resolved_coords(nodes, use_pixel_space=use_pixels)
            n = len(nodes)
            if n == 0:
                return {
                    "node_count": 0,
                    "unoptimized_distance": 0.0,
                    "optimized_distance": 0.0,
                    "saved_distance": 0.0,
                    "improvement_pct": 0.0,
                }
            dist_matrix = self._build_distance_matrix(coords)

            input_dist = self._compute_tour_distance(
                list(range(n)), dist_matrix, closed_loop=closed_loop
            )

            stats_before = self._last_stats
            # P2-9 (K-06): bu yöntem salt-istatistiktir — optimize_node_sequence
            # seq_index'i yerinde yazar; çağıranın düğümlerine dokunulmaması
            # için kopyalar üzerinde çalıştırılır (aynı dosyada
            # optimize_node_sequence([dict(n) for n in nodes]) önceliği;
            # copy.copy dict dışı düğüm nesnelerinde de çalışır). optimize
            # yalnızca mesafe için gerekir.
            opt_nodes = self.optimize_node_sequence(
                [copy.copy(node) for node in nodes], closed_loop=closed_loop
            )
            # optimize_node_sequence zaten optimize mesafeyi _last_stats'a
            # yazar; optimize edilmiş koordinatlar için ikinci bir O(n^2)
            # mesafe matrisi kurmak aynı sayıyı yeniden hesaplamaktır.
            # Başarı yolunda _last_stats yenilenir (yeni dict); başarısızlık /
            # erken dönüş yolunda yenilenmezse eski hesap yoluna düşülür.
            if self._last_stats is not stats_before and "optimized_distance" in self._last_stats:
                opt_dist = float(self._last_stats["optimized_distance"])
            else:
                # optimize_node_sequence koordinatsiz dugumu zaten
                # cikardi; savunma olarak None kalirsa atlanir.
                _opt_paired = [
                    (node, self._get_node_xy(node, use_pixel_space=use_pixels))
                    for node in opt_nodes
                ]
                opt_nodes = [node for node, xy in _opt_paired if xy is not None]
                opt_coords = [xy for _, xy in _opt_paired if xy is not None]
                opt_matrix = self._build_distance_matrix(opt_coords)
                opt_dist = self._compute_tour_distance(
                    list(range(len(opt_nodes))), opt_matrix, closed_loop=closed_loop
                )

            # F-1893: baz yalnızca girdi sırası mesafesidir; tarihî taban ve
            # optimize mesafeyle şişirilmiş max() kullanılmaz.
            unopt_dist = input_dist
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
                "closed_loop": bool(closed_loop),
            }
            self._last_stats = stats
            return stats
        except Exception as exc:
            logger.error("calculate_route_distance_stats hatasi: %s", exc)
            # P2-9 (K-06): hata yolunda bayat _last_stats döndürme — önceki
            # başarılı çalıştırmanın sayısı bu çağrının sonucu gibi sunulurdu.
            # Bilinmiyor açıkça işaretlenir (mesafe alanları None + "error");
            # _last_stats'a dokunulmaz.
            try:
                node_count = len(nodes)
            except Exception:
                node_count = 0
            return {
                "node_count": node_count,
                "unoptimized_distance": None,
                "optimized_distance": None,
                "saved_distance": None,
                "improvement_pct": None,
                "closed_loop": bool(closed_loop),
                "error": str(exc),
            }

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
            # F-1899: rest_duration_sec=0 "dinlenme yok" demektir; `or 300`
            # onu 300 sn'ye çevirirdi. Yalnızca None/eksik 300 varsayılır.
            raw_rest = self._extract_node_attr(node, "rest_duration_sec", None)
            rest_sec = 300 if raw_rest is None else int(raw_rest)
            if rest_sec <= 0:
                return True
            return (now - last_visited) >= float(rest_sec)
        except Exception as exc:
            logger.debug("Dugum hazirlik kontrol hatasi: %s", exc)
            return True

    def _remaining_rest_sec(self, node: Any, now: float) -> float:
        """Düğümün dinlenmesinin bitmesine kalan saniye (hazırsa 0)."""
        try:
            if hasattr(node, "is_ready") and callable(node.is_ready):
                return 0.0 if bool(node.is_ready(now=now)) else float("inf")
            last_visited = float(
                self._extract_node_attr(
                    node,
                    "last_visited_timestamp",
                    self._extract_node_attr(node, "last_visited", 0.0),
                )
                or 0.0
            )
            if last_visited <= 0.0:
                return 0.0
            raw_rest = self._extract_node_attr(node, "rest_duration_sec", None)
            rest_sec = 300 if raw_rest is None else int(raw_rest)
            if rest_sec <= 0:
                return 0.0
            return max(0.0, float(rest_sec) - (now - last_visited))
        except Exception:
            return 0.0

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
                # F-1900: eskiden dinlenmedeki TÜM düğümler sessizce aday
                # sayılırdı (dinlenme mantığı baypas edilirdi). Artık yalnızca
                # dinlenmesi en erken bitecek tek düğüm zorunlu aday olur ve
                # bu durum açıkça loglanır.
                forced = min(
                    ready_nodes,
                    key=lambda nd: self._remaining_rest_sec(nd, now),
                )
                logger.warning(
                    "Tüm düğümler dinlenmede; dinlenmesi en erken bitecek "
                    "düğüm (kalan %.1fs) zorunlu aday seçildi.",
                    self._remaining_rest_sec(forced, now),
                )
                active_candidates = [forced]

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
                # P1-7 (K-20): oyuncu konumu cozulemediyse hedef (0,0)'a
                # gore secilmez — hedef secimi yapilmaz (bilinmiyor doner).
                logger.warning(
                    "select_best_next_node: oyuncu konumu cozulemedi; "
                    "hedef secimi yapilmiyor."
                )
                return None

            # P1-7 (K-20): koordinatsiz aday (0,0)'a gore olculmez.
            active_candidates, coords = self._resolved_coords(
                active_candidates, use_pixel_space=use_pixels
            )
            if not active_candidates:
                logger.warning(
                    "select_best_next_node: koordinatli aday kalmadi; "
                    "hedef secimi yapilmiyor."
                )
                return None

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

                # Kume yogunlugu: cluster_radius_px yaricapindaki komsu hazir dugum sayisi.
                # Karesel mesafe: hypot <= r  <=>  dx^2+dy^2 <= r^2; çift başına
                # sqrt kalkar, predikat aynı kalır.
                radius_sq = self.cluster_radius_px * self.cluster_radius_px
                neighbors_count = 0
                for j, (ox, oy) in enumerate(coords):
                    if i == j:
                        continue
                    dx = nx - ox
                    dy = ny - oy
                    if dx * dx + dy * dy <= radius_sq:
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
            # F-1900 (3. yer): arızada dinlenme kontrolsüz ready_nodes[0]
            # dönülmezdi; yalnızca dinlenmesi dolmuş ilk düğüm yedek olur.
            logger.error("select_best_next_node hatasi: %s", exc)
            try:
                now = time.time()
                for node in ready_nodes or []:
                    if self._is_node_ready(node, now):
                        return node
            except Exception:
                pass
            return None

    def _self_check(self) -> Tuple[bool, str]:
        """Optimizasyon motorunu gercekten calistiran ayirt edici prob.

        S-2301: eski prob yalnizca (0,0),(100,0),(100,100),(0,100) kare
        dugumleriyle optimal sirada optimize_node_sequence calistirip
        len==4 ve dugum kumesi kontrol ediyordu; girdi zaten optimal ve
        kesisimsiz oldugundan _two_opt_improve / _eliminate_crossings
        no-op olsa bile gecerdi. Yeni prob:
        (1) ucu ucuna carpisma (bowtie) sirasi: kare koseleri
            [1,3,2,4] sirasinda verilir — tur 482.84 ve 1 kesisim;
            optimize_node_sequence 400.0'lik cevresel tura ve 0 kesisime
            inmek zorunda (mesafe + kesisim olculur);
        (2) _two_opt_improve dogrudan yoklanir: ayni bowtie sirasi
            482.84'ten 400.0'a inmek zorunda (no-op olsa 482.84 kalir);
        (3) _eliminate_crossings dogrudan yoklanir: bowtie 1 kesisimden
            0'a inmek zorunda (no-op olsa 1 kalir).
        Koseler (0,0) yerine (10,10) tabanlidir: P1-7 (K-20) oncesi (0,0)
        dugumu `_get_node_xy`'de "piksel koordinat yok" sayilip merkeze
        (960,540) kayardi, kare kare olmazdi. Artik koordinatsiz dugum
        `None` doner ve turdan cikarilir; prob yine de (10,10) tabanli
        koselerle calisir (merkez hayaletine dayanmaz). Her probda once
        taban degerin gercekten kotu oldugu dogrulanir (bos prob olmasin).
        """
        try:
            # --- (1) uctan uca bowtie probu (optimal sira, cokme/dugum kaybi) ---
            nodes = [
                {"id": 1, "map_x": 10.0, "map_y": 10.0},
                {"id": 2, "map_x": 110.0, "map_y": 10.0},
                {"id": 3, "map_x": 110.0, "map_y": 110.0},
                {"id": 4, "map_x": 10.0, "map_y": 110.0},
            ]
            ordered = self.optimize_node_sequence([dict(n) for n in nodes])
            if len(ordered) != 4:
                return False, f"self-check: expected 4 nodes, got {len(ordered)}"
            if sorted(self._extract_node_id(n, i) for i, n in enumerate(ordered)) != [1, 2, 3, 4]:
                return False, "self-check: optimized order lost nodes"
            if self.count_crossings(ordered) != 0:
                return False, "self-check: optimized order still has crossings"

            # --- bowtie girdisi: ayirt edici (2)/(3)/(4) problarinin zemini ---
            # Kare koseleri: 0:(10,10) 1:(110,10) 2:(110,110) 3:(10,110);
            # [0,2,1,3] sirasi kosegenleri caprazlar (bowtie).
            coords = [(10.0, 10.0), (110.0, 10.0), (110.0, 110.0), (10.0, 110.0)]
            dist_matrix = self._build_distance_matrix(coords)
            bowtie_order = [0, 2, 1, 3]
            optimal_dist = 400.0  # cevresel tur: 4 x 100
            baseline_dist = self._compute_tour_distance(bowtie_order, dist_matrix, closed_loop=True)
            baseline_crossings = self._count_coords_crossings(
                [coords[i] for i in bowtie_order], closed_loop=True
            )
            if not (baseline_dist > optimal_dist + 1e-6 and baseline_crossings >= 1):
                return False, (
                    "self-check: bowtie probe degenerated "
                    f"(dist={baseline_dist:.2f}, crossings={baseline_crossings})"
                )

            # --- (2) _two_opt_improve dogrudan: mesafeyi optimal tura indirmeli ---
            improved_order = self._two_opt_improve(bowtie_order, dist_matrix)
            improved_dist = self._compute_tour_distance(
                improved_order, dist_matrix, closed_loop=True
            )
            if improved_dist > optimal_dist + 1e-6:
                return False, (
                    "self-check: _two_opt_improve did not improve bowtie "
                    f"({baseline_dist:.2f} -> {improved_dist:.2f}, optimal {optimal_dist:.2f})"
                )

            # --- (3) _eliminate_crossings dogrudan: kesisimi sifirlamali ---
            cleaned_order = self._eliminate_crossings(bowtie_order, coords, closed_loop=True)
            remaining = self._count_coords_crossings(
                [coords[i] for i in cleaned_order], closed_loop=True
            )
            if remaining != 0:
                return False, (
                    f"self-check: _eliminate_crossings left {remaining} crossing(s)"
                )

            # --- (4) uctan uca bowtie: optimize_node_sequence mesafeyi dusurmeli ---
            bowtie_nodes = [
                {"id": 1, "map_x": 10.0, "map_y": 10.0},
                {"id": 3, "map_x": 110.0, "map_y": 110.0},
                {"id": 2, "map_x": 110.0, "map_y": 10.0},
                {"id": 4, "map_x": 10.0, "map_y": 110.0},
            ]
            bowtie_out = self.optimize_node_sequence([dict(n) for n in bowtie_nodes])
            if len(bowtie_out) != 4:
                return False, f"self-check: bowtie run lost nodes ({len(bowtie_out)})"
            if self.count_crossings(bowtie_out) != 0:
                return False, "self-check: bowtie run still has crossings"
            out_coords = [
                xy
                for xy in (
                    self._get_node_xy(n, use_pixel_space=True) for n in bowtie_out
                )
                if xy is not None
            ]
            out_matrix = self._build_distance_matrix(out_coords)
            out_dist = self._compute_tour_distance(
                list(range(len(out_coords))), out_matrix, closed_loop=True
            )
            if not (out_dist < baseline_dist - 1e-6):
                return False, (
                    "self-check: bowtie run did not shorten tour "
                    f"({baseline_dist:.2f} -> {out_dist:.2f})"
                )
            return True, "OK"
        except Exception as exc:
            return False, f"self-check failed: {exc}"

    def get_optimizer_status_summary(self) -> dict:
        """Returns optimizer status summary."""
        ready, reason = self._self_check()
        return {"optimizer_ready": ready, "algorithm": "2-opt+TSP", "reason": reason}

    def get_path_optimizer_health_summary(self) -> dict:
        # F-1903: koşulsuz "ready: True" yerine gerçek self-check.
        ready, reason = self._self_check()
        return {"module": "path_optimizer", "ready": ready, "reason": reason}

    def get_path_optimizer_ready_flag(self) -> dict:
        ready, reason = self._self_check()
        return {"module": "path_optimizer", "ready": ready, "reason": reason}

    def get_path_optimizer_version_info(self) -> dict:
        ready, reason = self._self_check()
        return {"module": "path_optimizer", "version": __version__, "ready": ready, "reason": reason}

    def get_path_optimizer_diagnostic_snapshot(self) -> dict:
        ready, reason = self._self_check()
        return {
            "module": "path_optimizer",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "reason": reason,
            "last_stats": dict(self._last_stats),
        }

    def get_path_optimizer_capabilities(self) -> list:
        return ["path_optimizer", "2opt_tsp_solver", "diagnostic_snapshot"]


def get_path_optimizer_capabilities() -> list:
    return ["path_optimizer", "2opt_tsp_solver", "diagnostic_snapshot"]


def validate_path_optimizer_contract() -> bool:
    # S-0203: eskiden `bool(get_path_optimizer_capabilities())` idi — statik,
    # boş olmayan bir liste her zaman True döndürüyordu; readiness/audit/
    # digest/integrity/compliance zincirinin tamamı sahte başarı üretiyordu.
    # Karar 3 (muhafazakâr seçenek): gerçek proba bağla — RouteOptimizer
    # örneğinin _self_check() probunu (4 düğümlü 2-opt sıralama kontrolü)
    # çalıştır; ayrıca capability listesi de boş olmamalı.
    if not get_path_optimizer_capabilities():
        return False
    try:
        ready, _reason = RouteOptimizer()._self_check()
    except Exception:
        return False
    return bool(ready)


def get_path_optimizer_baseline_metrics() -> dict:
    return {
        "module": "path_optimizer",
        "contract_valid": validate_path_optimizer_contract(),
        "capability_count": len(get_path_optimizer_capabilities()),
    }


def get_path_optimizer_readiness_status() -> dict:
    return {
        "module": "path_optimizer",
        "ready": validate_path_optimizer_contract(),
        "metrics": get_path_optimizer_baseline_metrics(),
    }


def get_path_optimizer_audit_record() -> dict:
    return {
        "module": "path_optimizer",
        "audit_passed": validate_path_optimizer_contract(),
        "readiness": get_path_optimizer_readiness_status(),
    }


def get_path_optimizer_operational_digest() -> dict:
    return {
        "module": "path_optimizer",
        "operational": validate_path_optimizer_contract(),
        "audit": get_path_optimizer_audit_record(),
    }


def get_path_optimizer_telemetry_payload() -> dict:
    return {
        "module": "path_optimizer",
        "payload": get_path_optimizer_operational_digest(),
    }


def get_path_optimizer_integrity_seal() -> dict:
    return {
        "module": "path_optimizer",
        "sealed": validate_path_optimizer_contract(),
        "telemetry": get_path_optimizer_telemetry_payload(),
    }


def get_path_optimizer_compliance_certificate() -> dict:
    return {
        "module": "path_optimizer",
        "certified": validate_path_optimizer_contract(),
        "seal": get_path_optimizer_integrity_seal(),
    }


def get_path_optimizer_readiness_probe() -> dict:
    return {
        "module": "path_optimizer",
        "ready": validate_path_optimizer_contract(),
        "certificate": get_path_optimizer_compliance_certificate(),
    }


def get_path_optimizer_audit_summary() -> dict:
    return {
        "module": "path_optimizer",
        "audit_passed": validate_path_optimizer_contract(),
        "probe": get_path_optimizer_readiness_probe(),
    }



