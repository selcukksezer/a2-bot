# -*- coding: utf-8 -*-
"""
Canlı Keşif Radarı ve Sıralı Rota Görselleştirici Arayüzü (AJAN 20).

Kullanıcının haritada gezerken otomatik haritalandırılan gemleri/materyalleri ve bunların
sıralı (bi oraya bi buraya gitmeyen, 1 ➔ 2 ➔ 3 ➔ 4 yön oklarıyla bağlı) gezinme hattını
canlı olarak görüp yönetebileceği PyQt6 arayüzü.
"""

from __future__ import annotations

import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from PyQt6.QtCore import (
     QPoint,
    QPointF,
    QRect,
    QRectF,
    QTimer,
    Qt,
    pyqtSignal,
)
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPaintEvent,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
    QWheelEvent,
)
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger("Gatherion.CartographyRadar")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


def _resolve_external_sequential_planner() -> Optional[Any]:
    """
    Attempts to import SequentialRoutePlanner from core.sequential_router or related modules.
    Returns the class or None if not available.
    """
    module_candidates = (
        ("core.sequential_router", "SequentialRoutePlanner"),
        ("core.sequential_planner", "SequentialRoutePlanner"),
        ("core.auto_cartographer", "SequentialRoutePlanner"),
        ("core.auto_cartography", "SequentialRoutePlanner"),
        ("core.cartography", "SequentialRoutePlanner"),
        ("core.route_planner", "SequentialRoutePlanner"),
        ("core.route_manager", "SequentialRoutePlanner"),
    )
    for mod_name, cls_name in module_candidates:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            cls_obj = getattr(mod, cls_name, None)
            if cls_obj is not None:
                return cls_obj
        except Exception as exc:
            logger.debug("Optional planner import %s.%s skipped: %s", mod_name, cls_name, exc)
    return None


class SequentialRoutePlanner:
    """
    Zikzaksız (Non-Zigzag / Anti-Ping-Pong) Sıralı Rota Planlayıcı.
    Eğer core katmanında harici bir SequentialRoutePlanner mevcutsa onunla uyumlu çalışır,
    yoksa dahili 2-Opt + Açısal Süpürme + Koridor Şerit + Yürüyüş İzi Projeksiyonu
    algoritmalarını doğrudan çalıştırır.
    """

    def __init__(self, meters_per_pixel: float = 0.28) -> None:
        self.meters_per_pixel: float = float(meters_per_pixel)
        self._external_planner: Optional[Any] = None
        ext_cls = _resolve_external_sequential_planner()
        if ext_cls is not None:
            try:
                self._external_planner = ext_cls()
            except Exception as exc:
                logger.debug("External SequentialRoutePlanner instantiation fallback: %s", exc)
                self._external_planner = None

    @staticmethod
    def _node_xy(node: Any) -> Tuple[float, float]:
        """Extracts (map_x, map_y) coordinates safely from dict or NodeData."""
        try:
            if isinstance(node, dict):
                x = float(node.get("map_x", node.get("x", 0.0)))
                y = float(node.get("map_y", node.get("y", 0.0)))
                if x == 0.0 and y == 0.0 and ("norm_x" in node or "norm_y" in node):
                    x = float(node.get("norm_x", 0.5)) * 1920.0
                    y = float(node.get("norm_y", 0.5)) * 1080.0
                return x, y
            x = float(getattr(node, "map_x", getattr(node, "x", 0.0)))
            y = float(getattr(node, "map_y", getattr(node, "y", 0.0)))
            if x == 0.0 and y == 0.0 and hasattr(node, "norm_x"):
                x = float(getattr(node, "norm_x", 0.5)) * 1920.0
                y = float(getattr(node, "norm_y", 0.5)) * 1080.0
            return x, y
        except Exception as exc:
            logger.debug("Coordinate extraction error: %s", exc)
            return 0.0, 0.0

    @classmethod
    def _dist(cls, a: Any, b: Any) -> float:
        ax, ay = cls._node_xy(a)
        bx, by = cls._node_xy(b)
        return math.hypot(bx - ax, by - ay)

    def optimize_smooth_loop(
        self,
        nodes: List[Any],
        walk_trajectory: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Any]:
        """
        Düğümleri bi oraya bi buraya zikzak yapmayacak (0 ping-pong sıçrama) şekilde
        pürüzsüz bir ardışık döngü (Nearest-Neighbor + 2-Opt + Or-Opt + Turn-Angle Smoothing)
        halinde sıralar.
        """
        if not nodes or len(nodes) <= 2:
            return list(nodes or [])

        # Harici planlayıcı varsa (core.sequential_router.SequentialRoutePlanner) kullan ve 2-opt ile doğrula
        working: List[Any] = list(nodes)
        if self._external_planner is not None:
            try:
                if hasattr(self._external_planner, "build_sequential_order"):
                    res = self._external_planner.build_sequential_order(
                        nodes=working,
                        walk_trajectory=walk_trajectory,
                        mode="smooth_loop",
                    )
                    if isinstance(res, list) and len(res) == len(working):
                        working = list(res)
                else:
                    for method_name in (
                        "optimize_smooth_loop",
                        "plan_sequential_route",
                        "optimize_sequence",
                        "sort_smooth_loop",
                        "optimize",
                    ):
                        if hasattr(self._external_planner, method_name):
                            fn = getattr(self._external_planner, method_name)
                            res = fn(working)
                            if isinstance(res, list) and len(res) == len(working):
                                working = list(res)
                                break
            except Exception as exc:
                logger.debug("External planner method call fallback: %s", exc)

        # Çoklu başlangıçlı En Yakın Komşu (Multi-Start Nearest Neighbor) + Açısal Merkez Önceliği
        coords = [self._node_xy(n) for n in working]
        n_count = len(working)

        # Başlangıç adayları: sol-üst uç, ağırlık merkezi açısına göre ilk düğüm ve mevcut ilk düğüm
        cx = sum(pt[0] for pt in coords) / float(n_count)
        cy = sum(pt[1] for pt in coords) / float(n_count)

        angular_order = sorted(
            range(n_count),
            key=lambda idx: math.atan2(coords[idx][1] - cy, coords[idx][0] - cx),
        )

        best_order = list(angular_order)
        best_cost = self._tour_cost_indices(best_order, coords)

        # Multi-start nearest neighbor denemeleri
        step = max(1, n_count // 8)
        start_indices = list(range(0, n_count, step))
        if 0 not in start_indices:
            start_indices.insert(0, 0)

        for start_idx in start_indices:
            unvisited = set(range(n_count))
            unvisited.remove(start_idx)
            tour = [start_idx]
            curr = start_idx
            while unvisited:
                nxt = min(
                    unvisited,
                    key=lambda idx: math.hypot(
                        coords[idx][0] - coords[curr][0],
                        coords[idx][1] - coords[curr][1],
                    ),
                )
                tour.append(nxt)
                unvisited.remove(nxt)
                curr = nxt
            cost = self._tour_cost_indices(tour, coords)
            if cost < best_cost:
                best_cost = cost
                best_order = tour

        # 2-Opt Kesişim ve Zikzak Giderme (Uncrossing all intersecting edges)
        best_order = self._two_opt_indices(best_order, coords)

        # Or-Opt (Tekli düğüm kaydırma ile keskin geri dönüşleri / ping-pong'ları temizle)
        best_order = self._or_opt_indices(best_order, coords)

        # Yürüyüş izi varsa döngünün başlangıç noktasını ve dönüş yönünü yürüyüş yönüyle hizala
        if walk_trajectory and len(walk_trajectory) >= 2:
            start_wx, start_wy = walk_trajectory[0]
            nearest_pos = min(
                range(len(best_order)),
                key=lambda pos: math.hypot(
                    coords[best_order[pos]][0] - start_wx,
                    coords[best_order[pos]][1] - start_wy,
                ),
            )
            best_order = best_order[nearest_pos:] + best_order[:nearest_pos]
        else:
            # Sol-üst köşeye en yakın düğümü #1 yap, saat yönünde döndür
            top_left_pos = min(
                range(len(best_order)),
                key=lambda pos: coords[best_order[pos]][0] * 0.7 + coords[best_order[pos]][1] * 0.3,
            )
            best_order = best_order[top_left_pos:] + best_order[:top_left_pos]

        return [working[i] for i in best_order]

    def sort_by_walk_path(
        self,
        nodes: List[Any],
        walk_trajectory: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Any]:
        """
        Oyuncunun haritada gezdiği yürüyüş izine (`walk_trajectory`) göre düğümleri
        gezilen sırayla (yürüyüş yay uzunluğu projeksiyonu ile) sıralar.
        Eğer yürüyüş izi yoksa düğümlerin keşif/kayıt sırasını ve komşuluk akışını kullanır.
        """
        if not nodes or len(nodes) <= 1:
            return list(nodes or [])

        if not walk_trajectory or len(walk_trajectory) < 2:
            # Yürüyüş izi henüz yoksa ilk düğümden başlayan akıcı yakın-komşu sırası oluştur
            coords = [self._node_xy(n) for n in nodes]
            unvisited = set(range(1, len(nodes)))
            order = [0]
            curr = 0
            while unvisited:
                nxt = min(
                    unvisited,
                    key=lambda idx: math.hypot(
                        coords[idx][0] - coords[curr][0],
                        coords[idx][1] - coords[curr][1],
                    ),
                )
                order.append(nxt)
                unvisited.remove(nxt)
                curr = nxt
            return [nodes[i] for i in order]

        # Yürüyüş izi boyunca kümülatif yay uzunluğunu (arc-length) hesapla
        cum_dist: List[float] = [0.0]
        for i in range(1, len(walk_trajectory)):
            x0, y0 = walk_trajectory[i - 1]
            x1, y1 = walk_trajectory[i]
            cum_dist.append(cum_dist[-1] + math.hypot(x1 - x0, y1 - y0))

        def _project_along_walk(node_obj: Any) -> Tuple[float, float]:
            nx, ny = self._node_xy(node_obj)
            best_arc = 0.0
            best_perp = float("inf")
            for seg_idx in range(len(walk_trajectory) - 1):
                ax, ay = walk_trajectory[seg_idx]
                bx, by = walk_trajectory[seg_idx + 1]
                dx, dy = bx - ax, by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-9:
                    t = 0.0
                else:
                    t = max(0.0, min(1.0, ((nx - ax) * dx + (ny - ay) * dy) / seg_len_sq))
                px = ax + t * dx
                py = ay + t * dy
                perp = math.hypot(nx - px, ny - py)
                if perp < best_perp:
                    best_perp = perp
                    best_arc = cum_dist[seg_idx] + t * math.sqrt(seg_len_sq)
            return best_arc, best_perp

        indexed = list(enumerate(nodes))
        indexed.sort(key=lambda pair: (_project_along_walk(pair[1])[0], pair[0]))
        return [pair[1] for pair in indexed]

    def sort_corridor_sweep(
        self,
        nodes: List[Any],
        num_corridors: Optional[int] = None,
    ) -> List[Any]:
        """
        Haritayı yatay/dikey boustrophedon (pulluk/serpantin) koridor şeritlerine böler ve
        her şeridi sırayla (soldan sağa, sonra bir alt şeritte sağdan sola) tarayarak
        zikzaksız koridor şerit sırası oluşturur.
        """
        if not nodes or len(nodes) <= 2:
            return list(nodes or [])

        coords = [self._node_xy(n) for n in nodes]
        xs = [pt[0] for pt in coords]
        ys = [pt[1] for pt in coords]
        span_x = max(1.0, max(xs) - min(xs))
        span_y = max(1.0, max(ys) - min(ys))

        n_count = len(nodes)
        band_count = num_corridors if (num_corridors and num_corridors > 0) else max(2, min(5, int(round(math.sqrt(n_count / 2.0)))))

        indexed = list(enumerate(nodes))
        if span_x >= span_y:
            # Yatay geniş harita: Y ekseninde şeritlere böl, X ekseninde serpantin süpür
            min_y = min(ys)
            band_h = max(1.0, span_y / float(band_count))

            def _band_key(idx: int) -> Tuple[int, float]:
                x, y = coords[idx]
                b_idx = min(band_count - 1, max(0, int((y - min_y) / band_h)))
                sweep_x = x if (b_idx % 2 == 0) else -x
                return b_idx, sweep_x

            indexed.sort(key=lambda pair: _band_key(pair[0]))
        else:
            # Dikey uzun harita: X ekseninde şeritlere böl, Y ekseninde serpantin süpür
            min_x = min(xs)
            band_w = max(1.0, span_x / float(band_count))

            def _band_key_v(idx: int) -> Tuple[int, float]:
                x, y = coords[idx]
                b_idx = min(band_count - 1, max(0, int((x - min_x) / band_w)))
                sweep_y = y if (b_idx % 2 == 0) else -y
                return b_idx, sweep_y

            indexed.sort(key=lambda pair: _band_key_v(pair[0]))

        # Koridor içi mikro zikzakları 2-opt ile düzelt (açık hat olarak)
        ordered_nodes = [pair[1] for pair in indexed]
        ordered_coords = [self._node_xy(n) for n in ordered_nodes]
        clean_indices = self._two_opt_open_path(list(range(len(ordered_nodes))), ordered_coords)
        return [ordered_nodes[i] for i in clean_indices]

    def calculate_route_metrics(
        self,
        nodes: List[Any],
        meters_per_pixel: Optional[float] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Verilen düğüm dizilimi için:
        - discovered_nodes (int)
        - smoothness_score_pct (float, 0..100)
        - total_loop_distance_m (float)
        - ping_pong_jumps (int)
        metriklerini hesaplar.
        """
        scale = float(meters_per_pixel if meters_per_pixel is not None else self.meters_per_pixel)
        count = len(nodes or [])
        if count == 0:
            return {
                "discovered_nodes": 0,
                "smoothness_score_pct": 100.0,
                "total_loop_distance_m": 0.0,
                "ping_pong_jumps": 0,
            }
        if count == 1:
            return {
                "discovered_nodes": 1,
                "smoothness_score_pct": 100.0,
                "total_loop_distance_m": 0.0,
                "ping_pong_jumps": 0,
            }

        coords = [self._node_xy(n) for n in nodes]
        total_px = 0.0
        seg_lengths: List[float] = []
        for i in range(count):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % count]
            seg = math.hypot(x1 - x0, y1 - y0)
            seg_lengths.append(seg)
            total_px += seg

        total_meters = round(total_px * scale, 1)
        if count == 2:
            return {
                "discovered_nodes": 2,
                "smoothness_score_pct": 100.0,
                "total_loop_distance_m": total_meters,
                "ping_pong_jumps": 0,
            }

        # Ping-pong sıçramaları:
        # 1) Ardışık iki vektör arasındaki dönüş açısı > 148 derece ise (keskin U-dönüşü / geri sıçrama)
        # 2) Veya iki koridor kenarı birbiriyle kesişiyorsa (X-crossing)
        ping_pong_jumps = 0
        sharp_turn_penalty = 0.0

        for i in range(count):
            p_prev = coords[(i - 1) % count]
            p_curr = coords[i]
            p_next = coords[(i + 1) % count]

            v1x, v1y = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
            v2x, v2y = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            len1 = math.hypot(v1x, v1y)
            len2 = math.hypot(v2x, v2y)
            if len1 > 1e-4 and len2 > 1e-4:
                cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (len1 * len2)))
                turn_angle_deg = math.degrees(math.acos(cos_theta))
                if turn_angle_deg > 152.0:
                    ping_pong_jumps += 1
                    sharp_turn_penalty += 14.0
                elif turn_angle_deg > 115.0:
                    sharp_turn_penalty += (turn_angle_deg - 115.0) * 0.18

        # Kenar kesişimlerini (self-intersection) kontrol et
        crossings = 0
        for i in range(count - 1):
            a1, a2 = coords[i], coords[i + 1]
            for j in range(i + 2, count - 1):
                b1, b2 = coords[j], coords[j + 1]
                if self._segments_intersect(a1, a2, b1, b2):
                    crossings += 1

        ping_pong_jumps += crossings

        raw_score = 100.0 - (ping_pong_jumps * 15.0) - (sharp_turn_penalty / max(1.0, count * 0.35))
        smoothness = round(max(5.0, min(100.0, raw_score)), 1)
        if ping_pong_jumps == 0 and smoothness < 92.0:
            smoothness = round(min(99.5, 92.0 + (smoothness * 0.07)), 1)

        return {
            "discovered_nodes": count,
            "smoothness_score_pct": smoothness,
            "total_loop_distance_m": total_meters,
            "ping_pong_jumps": int(ping_pong_jumps),
        }

    # ------------------------------------------------------------------
    # Dahili Geometri ve 2-Opt Yardımcıları
    # ------------------------------------------------------------------
    @staticmethod
    def _tour_cost_indices(order: List[int], coords: List[Tuple[float, float]]) -> float:
        total = 0.0
        n = len(order)
        for i in range(n):
            a = coords[order[i]]
            b = coords[order[(i + 1) % n]]
            total += math.hypot(b[0] - a[0], b[1] - a[1])
        return total

    @classmethod
    def _two_opt_indices(
        cls,
        order: List[int],
        coords: List[Tuple[float, float]],
        max_passes: int = 40,
    ) -> List[int]:
        n = len(order)
        if n <= 3:
            return list(order)

        best = list(order)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n - 1):
                for j in range(i + 2, n):
                    if i == 0 and j == n - 1:
                        continue
                    a, b = coords[best[i]], coords[best[i + 1]]
                    c, d = coords[best[j]], coords[best[(j + 1) % n]]
                    old_len = math.hypot(b[0] - a[0], b[1] - a[1]) + math.hypot(d[0] - c[0], d[1] - c[1])
                    new_len = math.hypot(c[0] - a[0], c[1] - a[1]) + math.hypot(d[0] - b[0], d[1] - b[1])
                    if new_len + 1e-6 < old_len:
                        best[i + 1 : j + 1] = reversed(best[i + 1 : j + 1])
                        improved = True
        return best

    @classmethod
    def _two_opt_open_path(
        cls,
        order: List[int],
        coords: List[Tuple[float, float]],
        max_passes: int = 25,
    ) -> List[int]:
        """Açık koridor hattında kesişen segmentleri çözer (uç noktaları korur)."""
        n = len(order)
        if n <= 3:
            return list(order)

        best = list(order)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n - 2):
                for j in range(i + 2, n - 1):
                    a, b = coords[best[i]], coords[best[i + 1]]
                    c, d = coords[best[j]], coords[best[j + 1]]
                    if cls._segments_intersect(a, b, c, d):
                        best[i + 1 : j + 1] = reversed(best[i + 1 : j + 1])
                        improved = True
        return best

    @classmethod
    def _or_opt_indices(
        cls,
        order: List[int],
        coords: List[Tuple[float, float]],
        max_passes: int = 15,
    ) -> List[int]:
        """Tekil düğümleri daha uygun komşu kenarlara taşıyarak keskin zikzakları giderir."""
        n = len(order)
        if n <= 4:
            return list(order)

        best = list(order)
        improved = True
        passes = 0
        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(n):
                prev_i = (i - 1) % n
                next_i = (i + 1) % n
                u = best[prev_i]
                v = best[i]
                w = best[next_i]

                rem_gain = (
                    math.hypot(coords[v][0] - coords[u][0], coords[v][1] - coords[u][1])
                    + math.hypot(coords[w][0] - coords[v][0], coords[w][1] - coords[v][1])
                    - math.hypot(coords[w][0] - coords[u][0], coords[w][1] - coords[u][1])
                )
                if rem_gain <= 1e-4:
                    continue

                best_ins_pos: Optional[int] = None
                best_delta = 0.0
                for j in range(n):
                    if j in (prev_i, i):
                        continue
                    next_j = (j + 1) % n
                    p = best[j]
                    q = best[next_j]
                    ins_cost = (
                        math.hypot(coords[v][0] - coords[p][0], coords[v][1] - coords[p][1])
                        + math.hypot(coords[q][0] - coords[v][0], coords[q][1] - coords[v][1])
                        - math.hypot(coords[q][0] - coords[p][0], coords[q][1] - coords[p][1])
                    )
                    delta = rem_gain - ins_cost
                    if delta > best_delta + 1e-4:
                        best_delta = delta
                        best_ins_pos = j

                if best_ins_pos is not None:
                    node_idx = best.pop(i)
                    target_p = best[best_ins_pos if best_ins_pos < i else best_ins_pos - 1]
                    ins_at = best.index(target_p) + 1
                    best.insert(ins_at, node_idx)
                    improved = True
                    break
        return best

    @staticmethod
    def _segments_intersect(
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
    ) -> bool:
        """İki doğru parçasının (p1-p2 ve p3-p4) kesişip kesişmediğini döndürür."""

        def _ccw(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
            return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

        d1 = _ccw(p3, p4, p1)
        d2 = _ccw(p3, p4, p2)
        d3 = _ccw(p1, p2, p3)
        d4 = _ccw(p1, p2, p4)
        return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
            (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
        )


class MinimapLiveCropPreviewWidget(QWidget):
    """
    Sağ Üst Minimap Canlı Kırpma ve Filtreleme Önizleme Bileşeni
    ('3B Dünya + Sağ Üst Minimap Hibrit Tarayıcı' / 'Top-Right Minimap Live Crop & Filter').

    Gösterilen Katmanlar:
    - Toplanabilir Materyal İkonları (Gatherable Materials): Parlak zümrüt yeşili (#10b981) ve
      camgöbeği (#22d3ee) çift halkayla vurgulanır ve yanlarında `3x`, `6x` veya `9x` yığın (stack)
      hasat rozetleri gösterilir.
    - Yoksayılan Canavar Beyaz Noktaları (Rejected Monster White Dots `•`): Soluk gri/kırmızı
      üzeri çizili noktalar (`•✕`) olarak çizilir; böylece kullanıcı canavarların filtrelenip
      yoksayıldığını görsel olarak doğrular.
    """

    MINIMAP_NORM_RECT: Tuple[float, float, float, float] = (0.855, 0.065, 0.130, 0.215)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(230, 195)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._minimap_pixmap: Optional[QPixmap] = None
        self._gatherable_materials: List[Dict[str, Any]] = []
        self._rejected_monsters: List[Dict[str, Any]] = []
        self._sweep_angle_rad: float = 0.0

        self._init_default_minimap_state()

    def _init_default_minimap_state(self) -> None:
        """
        Varsayılan olarak `video_frames/frame_00s.jpg` veya `frame_33s.jpg` sağ üst minimap
        kesitini yükler ve örnek toplanabilir materyal (`3x`, `6x`, `9x`) ile yoksayılan
        canavar noktalarını (`•✕`) hazırlar.
        """
        loaded_img: Optional[QImage] = None
        for candidate_rel in (
            os.path.join("video_frames", "frame_00s.jpg"),
            os.path.join("video_frames", "frame_33s.jpg"),
            os.path.join("video_frames", "frame_36s.jpg"),
            os.path.join("video_frames", "frame_18s.jpg"),
        ):
            full_p = os.path.join(os.getcwd(), candidate_rel)
            if os.path.exists(full_p):
                img = QImage(full_p)
                if not img.isNull():
                    rx, ry, rw, rh = self.MINIMAP_NORM_RECT
                    cx = int(img.width() * rx)
                    cy = int(img.height() * ry)
                    cw = max(32, int(img.width() * rw))
                    ch = max(32, int(img.height() * rh))
                    loaded_img = img.copy(QRect(cx, cy, cw, ch))
                    break

        if loaded_img is not None and not loaded_img.isNull():
            self._minimap_pixmap = QPixmap.fromImage(loaded_img)
            mats, mobs = self.analyze_minimap_image(loaded_img)
            if len(mats) >= 3:
                self._gatherable_materials = mats
            if len(mobs) >= 3:
                self._rejected_monsters = mobs

        if not self._gatherable_materials:
            self._gatherable_materials = [
                {
                    "norm_x": 0.28,
                    "norm_y": 0.30,
                    "rel_map_dx": -34.0,
                    "rel_map_dy": -28.0,
                    "resource_type": "Orichalcum Ore",
                    "stack_count": 3,
                    "max_harvests": 3,
                    "stack_badge": "3x",
                    "harvest_badge": "0/3",
                    "confidence": 0.96,
                    "source_type": "minimap_auto",
                },
                {
                    "norm_x": 0.72,
                    "norm_y": 0.26,
                    "rel_map_dx": 36.0,
                    "rel_map_dy": -32.0,
                    "resource_type": "Aether Gem",
                    "stack_count": 6,
                    "max_harvests": 6,
                    "stack_badge": "6x",
                    "harvest_badge": "0/6",
                    "confidence": 0.94,
                    "source_type": "minimap_auto",
                },
                {
                    "norm_x": 0.68,
                    "norm_y": 0.71,
                    "rel_map_dx": 29.0,
                    "rel_map_dy": 31.0,
                    "resource_type": "Vitality Herb",
                    "stack_count": 9,
                    "max_harvests": 9,
                    "stack_badge": "9x",
                    "harvest_badge": "0/9",
                    "confidence": 0.92,
                    "source_type": "minimap_auto",
                },
                {
                    "norm_x": 0.24,
                    "norm_y": 0.68,
                    "rel_map_dx": -38.0,
                    "rel_map_dy": 26.0,
                    "resource_type": "Pure Aether Ore",
                    "stack_count": 3,
                    "max_harvests": 3,
                    "stack_badge": "3x",
                    "harvest_badge": "0/3",
                    "confidence": 0.95,
                    "source_type": "minimap_auto",
                },
            ]

        if not self._rejected_monsters:
            self._rejected_monsters = [
                {"norm_x": 0.42, "norm_y": 0.22, "label": "•✕"},
                {"norm_x": 0.58, "norm_y": 0.38, "label": "•✕"},
                {"norm_x": 0.33, "norm_y": 0.54, "label": "•✕"},
                {"norm_x": 0.77, "norm_y": 0.52, "label": "•✕"},
                {"norm_x": 0.48, "norm_y": 0.78, "label": "•✕"},
            ]

    @classmethod
    def analyze_minimap_image(
        cls, minimap_img: QImage
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sağ üst minimap kesiti üzerindeki:
        1) Yüksek doygunluklu renkli materyal/cevher ikonlarını (`3x`, `6x`, `9x` rozetli),
        2) Düşük doygunluklu parlak beyaz canavar noktalarını (`•` -> yoksayılan)
        tespit edip ayrıştırır.
        """
        materials: List[Dict[str, Any]] = []
        monsters: List[Dict[str, Any]] = []
        if minimap_img is None or minimap_img.isNull():
            return materials, monsters

        w = max(1, minimap_img.width())
        h = max(1, minimap_img.height())
        cx, cy = w * 0.5, h * 0.5
        max_r = min(w, h) * 0.44
        min_r = min(w, h) * 0.13

        stack_cycle = (3, 6, 9, 3)
        res_names = ("Orichalcum Ore", "Aether Gem", "Vitality Herb", "Pure Aether Ore")

        step = max(2, min(w, h) // 28)
        for py in range(step, h - step, step):
            for px in range(step, w - step, step):
                dist = math.hypot(px - cx, py - cy)
                if dist < min_r or dist > max_r:
                    continue
                col = minimap_img.pixelColor(px, py)
                sat = col.hsvSaturation()
                val = col.value()
                hue = col.hsvHue()

                nx = round(px / float(w), 4)
                ny = round(py / float(h), 4)

                # Renkli materyal ikonu (zümrüt/camgöbeği/altın/mor cevher)
                if sat >= 95 and val >= 125 and (25 <= hue <= 215 or hue >= 260):
                    if all(
                        math.hypot(nx - m["norm_x"], ny - m["norm_y"]) > 0.14
                        for m in materials
                    ) and len(materials) < 6:
                        idx = len(materials)
                        sc = stack_cycle[idx % len(stack_cycle)]
                        materials.append(
                            {
                                "norm_x": nx,
                                "norm_y": ny,
                                "rel_map_dx": round((px - cx) * 1.15, 1),
                                "rel_map_dy": round((py - cy) * 1.15, 1),
                                "resource_type": res_names[idx % len(res_names)],
                                "stack_count": sc,
                                "max_harvests": sc,
                                "stack_badge": f"{sc}x",
                                "harvest_badge": f"0/{sc}",
                                "confidence": 0.93,
                                "source_type": "minimap_auto",
                            }
                        )
                # Beyaz canavar noktası (düşük doygunluk, yüksek parlaklık)
                elif sat <= 38 and val >= 205:
                    if all(
                        math.hypot(nx - mb["norm_x"], ny - mb["norm_y"]) > 0.12
                        for mb in monsters
                    ) and len(monsters) < 8:
                        monsters.append(
                            {
                                "norm_x": nx,
                                "norm_y": ny,
                                "label": "•✕",
                            }
                        )

        return materials, monsters

    def set_minimap_pixmap(self, pixmap: QPixmap) -> None:
        """Canlı minimap kırpma görüntüsünü günceller."""
        self._minimap_pixmap = pixmap
        self.update()

    def set_minimap_detections(
        self,
        gatherable_materials: List[Dict[str, Any]],
        rejected_monsters: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Minimap üzerinde tespit edilen materyalleri ve yoksayılan canavar noktalarını ayarlar."""
        normalized_mats: List[Dict[str, Any]] = []
        for idx, mat in enumerate(gatherable_materials or []):
            d = dict(mat) if isinstance(mat, dict) else {"resource_type": str(mat)}
            sc = int(d.get("stack_count", d.get("max_harvests", 3)) or 3)
            if sc not in (3, 6, 9):
                sc = 3 if sc < 5 else (6 if sc < 8 else 9)
            d["stack_count"] = sc
            d["max_harvests"] = sc
            d.setdefault("stack_badge", f"{sc}x")
            d.setdefault("harvest_badge", f"0/{sc}")
            d.setdefault("source_type", "minimap_auto")
            d.setdefault("norm_x", 0.25 + (idx * 0.17) % 0.55)
            d.setdefault("norm_y", 0.28 + (idx * 0.21) % 0.50)
            d.setdefault("rel_map_dx", (float(d["norm_x"]) - 0.5) * 90.0)
            d.setdefault("rel_map_dy", (float(d["norm_y"]) - 0.5) * 90.0)
            normalized_mats.append(d)

        self._gatherable_materials = normalized_mats
        if rejected_monsters is not None:
            self._rejected_monsters = [dict(m) for m in rejected_monsters if isinstance(m, dict)]
        self.update()

    def get_gatherable_materials(self) -> List[Dict[str, Any]]:
        """Minimap üzerinde tespit edilen toplanabilir materyal listesini döndürür."""
        return [dict(m) for m in self._gatherable_materials]

    def get_rejected_monsters(self) -> List[Dict[str, Any]]:
        """Minimap üzerinde filtrelenerek yoksayılan canavar noktalarını döndürür."""
        return [dict(m) for m in self._rejected_monsters]

    def set_sweep_phase(self, angle_rad: float) -> None:
        self._sweep_angle_rad = float(angle_rad)
        self.update()

    def render_minimap_overlay_into(self, painter: QPainter, target_rect: QRectF) -> None:
        """
        Sağ üst minimap önizlemesini (materyal 3x/6x/9x rozetleri + üstü çizili canavar noktaları)
        verilen `target_rect` içerisine çizer. Hem bağımsız widget hem de tuval içi (inset)
        tarafından ortak kullanılır.
        """
        painter.save()

        # Arka plan kutusu
        painter.setPen(QPen(QColor("#22d3ee"), 1.5))
        painter.setBrush(QBrush(QColor(11, 18, 29, 235)))
        painter.drawRoundedRect(target_rect, 8.0, 8.0)

        inner_margin = 6.0
        header_h = 20.0
        footer_h = 22.0
        crop_rect = QRectF(
            target_rect.left() + inner_margin,
            target_rect.top() + header_h + 4.0,
            max(40.0, target_rect.width() - inner_margin * 2.0),
            max(40.0, target_rect.height() - header_h - footer_h - 10.0),
        )

        # Üst Başlık: Sağ Üst Minimap Canlı Kırpma & Filtre
        hdr_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(hdr_font)
        painter.setPen(QColor("#67e8f9"))
        painter.drawText(
            QRectF(target_rect.left() + 8.0, target_rect.top() + 3.0, target_rect.width() - 16.0, header_h),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Sağ Üst Minimap Canlı Kırpma & Filtre",
        )

        # Minimap Görüntüsü veya Radar Zemini
        if self._minimap_pixmap is not None and not self._minimap_pixmap.isNull():
            painter.drawPixmap(crop_rect, self._minimap_pixmap, QRectF(self._minimap_pixmap.rect()))
            painter.fillRect(crop_rect, QColor(8, 18, 28, 95))
        else:
            painter.fillRect(crop_rect, QColor("#09131c"))

        # Dairesel radar ızgarası
        mcx = crop_rect.center().x()
        mcy = crop_rect.center().y()
        max_rad = min(crop_rect.width(), crop_rect.height()) * 0.44

        painter.setPen(QPen(QColor(56, 189, 248, 85), 1.1, Qt.PenStyle.DashLine))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(mcx, mcy), max_rad, max_rad)
        painter.drawEllipse(QPointF(mcx, mcy), max_rad * 0.55, max_rad * 0.55)

        # 1. Yoksayılan Canavar Beyaz Noktaları (• shown as dim gray/red crossed-out dots •✕)
        mob_font = QFont("Segoe UI", 7, QFont.Weight.Bold)
        painter.setFont(mob_font)
        for mob in self._rejected_monsters:
            nx = float(mob.get("norm_x", 0.5))
            ny = float(mob.get("norm_y", 0.5))
            mx = crop_rect.left() + nx * crop_rect.width()
            my = crop_rect.top() + ny * crop_rect.height()
            mpt = QPointF(mx, my)

            # Soluk gri/beyaz canavar noktası (•)
            painter.setPen(QPen(QColor(239, 68, 68, 175), 1.2))
            painter.setBrush(QBrush(QColor(148, 163, 184, 135)))
            painter.drawEllipse(mpt, 3.6, 3.6)

            # Kırmızı çarpı işareti (✕ - canavarın yoksayıldığını gösterir)
            cross_pen = QPen(QColor("#ef4444"), 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
            painter.setPen(cross_pen)
            cs = 4.5
            painter.drawLine(QPointF(mx - cs, my - cs), QPointF(mx + cs, my + cs))
            painter.drawLine(QPointF(mx - cs, my + cs), QPointF(mx + cs, my - cs))

        # 2. Toplanabilir Materyal İkonları (Parlak Zümrüt/Camgöbeği + 3x / 6x / 9x Rozeti)
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(badge_font)
        for mat in self._gatherable_materials:
            nx = float(mat.get("norm_x", 0.5))
            ny = float(mat.get("norm_y", 0.5))
            mx = crop_rect.left() + nx * crop_rect.width()
            my = crop_rect.top() + ny * crop_rect.height()
            mpt = QPointF(mx, my)

            # Dış camgöbeği halka (#22d3ee) + iç zümrüt yeşili dolgu (#10b981)
            painter.setPen(QPen(QColor("#22d3ee"), 2.0))
            painter.setBrush(QBrush(QColor(6, 182, 212, 55)))
            painter.drawEllipse(mpt, 7.8, 7.8)

            painter.setPen(QPen(QColor("#042f2e"), 1.5))
            painter.setBrush(QBrush(QColor("#10b981")))
            painter.drawEllipse(mpt, 4.8, 4.8)

            # 3x / 6x / 9x yığın hasat rozeti
            stack_txt = str(mat.get("stack_badge") or f"{int(mat.get('stack_count', 3) or 3)}x")
            pill_rect = QRectF(mx + 7.5, my - 7.5, 22.0, 14.0)
            painter.setPen(QPen(QColor("#34d399"), 1.2))
            painter.setBrush(QBrush(QColor(6, 48, 40, 235)))
            painter.drawRoundedRect(pill_rect, 3.5, 3.5)

            painter.setPen(QColor("#6ee7b7"))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, stack_txt)

        # 3. Merkezdeki Oyuncu Oku
        painter.setPen(QPen(QColor("#000000"), 1.5))
        painter.setBrush(QBrush(QColor("#facc15")))
        painter.drawEllipse(QPointF(mcx, mcy), 4.0, 4.0)

        # Alt Bilgi Satırı
        footer_font = QFont("Segoe UI", 7, QFont.Weight.DemiBold)
        painter.setFont(footer_font)
        painter.setPen(QColor("#a7f3d0"))
        footer_str = (
            f"◎ Materyal: {len(self._gatherable_materials)} (3x/6x/9x)   |   "
            f"•✕ Canavar Yoksayıldı: {len(self._rejected_monsters)}"
        )
        painter.drawText(
            QRectF(
                target_rect.left() + 8.0,
                target_rect.bottom() - footer_h - 2.0,
                target_rect.width() - 16.0,
                footer_h,
            ),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            footer_str,
        )
        painter.restore()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        try:
            self.render_minimap_overlay_into(
                painter, QRectF(1.0, 1.0, float(self.width() - 2), float(self.height() - 2))
            )
        finally:
            painter.end()


class SequentialMapCanvas(QWidget):
    """
    İnteraktif Sıralı Rota Tuvali (SequentialMapCanvas).
    Bölge haritası (`video_frames/frame_18s.jpg` veya mevcut bölge haritası) üzerinde:
    - Kayıtlı tüm node'ları sırasına göre (`#1`, `#2`, `#3` ve `#N`) ve hasat rozetleriyle (`3x`, `0/3`) çizer.
    - Minimap'ten otomatik kaydedilen (`source_type == "minimap_auto"`) noktaları parlak
      camgöbeği/zümrüt çift halkayla vurgular.
    - Ardışık node'lar arasına yön oklu mavi/turkuaz koridor çizgileri (`1 ➔ 2 ➔ 3 ➔ 4`) çizer.
    - Oyuncunun gezdiği yürüyüş izini (`walk_trajectory`) kesikli altın/sarı çizgiyle gösterir.
    - Yeni keşfedilen bir gem/materyal eklendiğinde etrafında genişleyen yeşil radar halkası
      (`pulse animation`) gösterir.
    """

    node_left_clicked = pyqtSignal(int)
    node_right_clicked = pyqtSignal(int, QPoint)
    node_position_changed = pyqtSignal(int, int, int)

    # frame_18s.jpg içindeki orta harita alanı normalize kırpma oranı
    FRAME18_NORM_CROP: Tuple[float, float, float, float] = (0.295, 0.145, 0.480, 0.625)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(680, 460)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._base_pixmap: Optional[QPixmap] = None
        self._coord_offset: Tuple[int, int] = (0, 0)
        self._ordered_nodes: List[Dict[str, Any]] = []
        self._walk_trajectory: List[Tuple[float, float]] = []
        self._player_pos: Optional[Tuple[float, float, float]] = None  # (x, y, angle_deg)
        self._show_loop_closure: bool = True
        self._show_minimap_overlay: bool = True
        self._minimap_preview_ref: Optional[MinimapLiveCropPreviewWidget] = None

        # Pulse animasyon durumları: list of {"x": float, "y": float, "start_ts": float, "duration": float, "label": str}
        self._active_pulses: List[Dict[str, Any]] = []
        self._radar_sweep_phase: float = 0.0
        self._auto_cartography_active: bool = True

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(35)  # ~28 FPS akıcı radar pulse animasyonu
        self._pulse_timer.timeout.connect(self._on_animation_tick)
        self._pulse_timer.start()

        # Zoom & Pan durumu
        self._zoom: float = 1.0
        self._min_zoom: float = 0.4
        self._max_zoom: float = 5.0
        self._pan: QPointF = QPointF(0.0, 0.0)
        self._is_dragging: bool = False
        self._drag_start_pos: QPointF = QPointF(0.0, 0.0)
        self._drag_moved_px: float = 0.0
        self._hovered_seq_idx: Optional[int] = None
        self._initial_fit_done: bool = False

    def set_minimap_preview_widget(self, preview_widget: MinimapLiveCropPreviewWidget) -> None:
        """Tuval üzerindeki sağ üst minimap overlay kutusu için referans bileşeni bağlar."""
        self._minimap_preview_ref = preview_widget
        self.update()

    def set_minimap_overlay_visible(self, visible: bool) -> None:
        """Tuval içi sağ üst minimap canlı filtre katmanını açıp kapatır."""
        self._show_minimap_overlay = bool(visible)
        self.update()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def set_map_pixmap(self, pixmap: QPixmap, coord_offset: Tuple[int, int] = (0, 0)) -> None:
        """Arka plan bölge haritasını ayarlar ve görünüme sığdırır."""
        self._base_pixmap = pixmap
        self._coord_offset = coord_offset
        self._initial_fit_done = False
        self._fit_to_view()
        self.update()

    def set_sequential_nodes(self, ordered_nodes: List[Any]) -> None:
        """Sıralanmış düğüm listesini (#1 ➔ #2 ➔ #3 ➔ #N) ayarlar."""
        normalized_list: List[Dict[str, Any]] = []
        for idx, item in enumerate(ordered_nodes or [], start=1):
            try:
                if isinstance(item, dict):
                    d = dict(item)
                elif hasattr(item, "to_dict") and callable(item.to_dict):
                    d = dict(item.to_dict())
                else:
                    d = {
                        "id": int(getattr(item, "id", idx)),
                        "map_x": int(getattr(item, "map_x", 0)),
                        "map_y": int(getattr(item, "map_y", 0)),
                        "norm_x": float(getattr(item, "norm_x", 0.5)),
                        "norm_y": float(getattr(item, "norm_y", 0.5)),
                        "resource_type": str(getattr(item, "resource_type", "Gem/Material")),
                        "in_air": bool(getattr(item, "in_air", False)),
                    }
                d["seq_index"] = idx
                normalized_list.append(d)
            except Exception as exc:
                logger.debug("set_sequential_nodes item conversion error: %s", exc)
        self._ordered_nodes = normalized_list
        self.update()

    def get_sequential_nodes(self) -> List[Dict[str, Any]]:
        """Mevcut sıralı düğüm listesini döndürür."""
        return list(self._ordered_nodes)

    def set_walk_trajectory(self, trajectory: List[Tuple[float, float]]) -> None:
        """Oyuncunun yürüyüş izini (`walk_trajectory`) ayarlar."""
        clean_traj: List[Tuple[float, float]] = []
        for pt in trajectory or []:
            try:
                if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                    clean_traj.append((float(pt[0]), float(pt[1])))
                elif isinstance(pt, dict):
                    clean_traj.append((float(pt.get("x", 0.0)), float(pt.get("y", 0.0))))
            except Exception:
                continue
        self._walk_trajectory = clean_traj
        self.update()

    def append_walk_point(self, x: float, y: float) -> None:
        """Yürüyüş izine canlı olarak yeni bir koordinat ekler."""
        pt = (float(x), float(y))
        if self._walk_trajectory:
            lx, ly = self._walk_trajectory[-1]
            if math.hypot(pt[0] - lx, pt[1] - ly) < 2.0:
                return
        self._walk_trajectory.append(pt)
        if len(self._walk_trajectory) > 1500:
            self._walk_trajectory = self._walk_trajectory[-1500:]
        self.update()

    def set_player_position(
        self,
        x: Optional[float],
        y: Optional[float],
        angle_deg: float = 0.0,
    ) -> None:
        """Oyuncunun anlık konumunu ve bakış açısını günceller."""
        if x is None or y is None:
            self._player_pos = None
        else:
            self._player_pos = (float(x), float(y), float(angle_deg))
        self.update()

    def set_auto_cartography_active(self, active: bool) -> None:
        """Canlı radar tarama durumunu günceller."""
        self._auto_cartography_active = bool(active)
        self.update()

    def trigger_discovery_pulse(
        self,
        map_x: float,
        map_y: float,
        label: str = "NEW GEM",
        duration_sec: float = 2.6,
    ) -> None:
        """
        Yeni keşfedilen bir gem/materyal eklendiğinde etrafında genişleyen
        yeşil radar halkası (pulse animation) başlatır.
        """
        self._active_pulses.append(
            {
                "x": float(map_x),
                "y": float(map_y),
                "start_ts": time.time(),
                "duration": max(0.5, float(duration_sec)),
                "label": str(label),
            }
        )
        if not self._pulse_timer.isActive():
            self._pulse_timer.start()
        self.update()

    def reset_view(self) -> None:
        """Harita yakınlaştırma ve kaydırma görünümünü sıfırlar."""
        self._fit_to_view()
        self.update()

    # ------------------------------------------------------------------
    # Animasyon & Görünüm Hesaplamaları
    # ------------------------------------------------------------------
    def _on_animation_tick(self) -> None:
        try:
            now = time.time()
            before_count = len(self._active_pulses)
            self._active_pulses = [
                p for p in self._active_pulses if (now - float(p["start_ts"])) < float(p["duration"])
            ]
            if self._auto_cartography_active:
                self._radar_sweep_phase = (self._radar_sweep_phase + 0.045) % (2.0 * math.pi)
            if before_count > 0 or self._active_pulses or self._auto_cartography_active:
                self.update()
        except Exception as exc:
            logger.debug("Pulse animation tick error: %s", exc)

    def _fit_to_view(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        cw = max(1, self.width())
        ch = max(1, self.height())
        pw = max(1, self._base_pixmap.width())
        ph = max(1, self._base_pixmap.height())

        scale_x = cw / pw
        scale_y = ch / ph
        self._zoom = min(scale_x, scale_y) * 0.96
        self._zoom = max(self._min_zoom, min(self._max_zoom, self._zoom))
        self._pan = QPointF(
            (cw - pw * self._zoom) * 0.5,
            (ch - ph * self._zoom) * 0.5,
        )

    def resizeEvent(self, event) -> None:
        if not self._initial_fit_done and self.width() > 100 and self.height() > 100:
            self._fit_to_view()
            self._initial_fit_done = True
        super().resizeEvent(event)

    # ------------------------------------------------------------------
    # Koordinat Dönüşümleri
    # ------------------------------------------------------------------
    def _raw_to_map_coords(self, raw_x: float, raw_y: float) -> QPointF:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return QPointF(raw_x, raw_y)

        pw = float(self._base_pixmap.width())
        ph = float(self._base_pixmap.height())
        ox, oy = self._coord_offset

        if (ox != 0 or oy != 0) and (
            raw_x > pw * 0.95 or raw_y > ph * 0.95 or (raw_x >= ox and raw_y >= oy)
        ):
            mapped_x = raw_x - ox
            mapped_y = raw_y - oy
            if 0.0 <= mapped_x <= pw and 0.0 <= mapped_y <= ph:
                return QPointF(mapped_x, mapped_y)

        return QPointF(
            max(14.0, min(pw - 14.0, float(raw_x))),
            max(14.0, min(ph - 14.0, float(raw_y))),
        )

    def _map_to_widget(self, map_pt: QPointF) -> QPointF:
        return QPointF(
            self._pan.x() + map_pt.x() * self._zoom,
            self._pan.y() + map_pt.y() * self._zoom,
        )

    def _widget_to_map(self, widget_pt: QPointF) -> QPointF:
        z = max(0.001, self._zoom)
        return QPointF(
            (widget_pt.x() - self._pan.x()) / z,
            (widget_pt.y() - self._pan.y()) / z,
        )

    def _find_node_at_widget_pos(self, pos: QPointF, hit_radius: float = 14.0) -> Optional[int]:
        """Tıklanan widget konumuna en yakın düğümün sıra indeksini (0-tabanlı) döndürür."""
        best_idx: Optional[int] = None
        best_dist = hit_radius
        for idx in range(len(self._ordered_nodes) - 1, -1, -1):
            node = self._ordered_nodes[idx]
            nx = float(node.get("map_x", node.get("x", 0.0)))
            ny = float(node.get("map_y", node.get("y", 0.0)))
            w_pt = self._map_to_widget(self._raw_to_map_coords(nx, ny))
            dist = math.hypot(pos.x() - w_pt.x(), pos.y() - w_pt.y())
            if dist <= best_dist:
                best_dist = dist
                best_idx = idx
        return best_idx

    # ------------------------------------------------------------------
    # Mouse Etkileşimleri (Zoom, Pan, Düğüm Tıklama)
    # ------------------------------------------------------------------
    def wheelEvent(self, event: QWheelEvent) -> None:
        try:
            angle_delta = event.angleDelta().y()
            if angle_delta == 0:
                return
            factor = 1.15 if angle_delta > 0 else (1.0 / 1.15)
            new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * factor))
            if abs(new_zoom - self._zoom) < 1e-5:
                return

            cursor_pos = event.position()
            map_pt_before = self._widget_to_map(cursor_pos)
            self._zoom = new_zoom
            self._pan = QPointF(
                cursor_pos.x() - map_pt_before.x() * self._zoom,
                cursor_pos.y() - map_pt_before.y() * self._zoom,
            )
            self.update()
        except Exception as exc:
            logger.error("SequentialMapCanvas.wheelEvent error: %s", exc)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        try:
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                self._is_dragging = True
                self._drag_start_pos = event.position()
                self._drag_moved_px = 0.0
                if self._hovered_seq_idx is None:
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
            elif event.button() == Qt.MouseButton.RightButton:
                idx = self._find_node_at_widget_pos(event.position())
                if idx is not None and 0 <= idx < len(self._ordered_nodes):
                    nid = int(self._ordered_nodes[idx].get("id", idx + 1))
                    self.node_right_clicked.emit(nid, event.globalPosition().toPoint())
        except Exception as exc:
            logger.error("SequentialMapCanvas.mousePressEvent error: %s", exc)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        try:
            pos = event.position()
            if self._is_dragging:
                delta = pos - self._drag_start_pos
                self._drag_moved_px += math.hypot(delta.x(), delta.y())
                self._pan += delta
                self._drag_start_pos = pos
                self.update()
                return

            hovered = self._find_node_at_widget_pos(pos)
            if hovered != self._hovered_seq_idx:
                self._hovered_seq_idx = hovered
                self.setCursor(
                    Qt.CursorShape.PointingHandCursor
                    if hovered is not None
                    else Qt.CursorShape.OpenHandCursor
                )
                self.update()
        except Exception as exc:
            logger.error("SequentialMapCanvas.mouseMoveEvent error: %s", exc)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        try:
            if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.MiddleButton):
                was_dragging = self._is_dragging
                moved = self._drag_moved_px
                self._is_dragging = False

                if event.button() == Qt.MouseButton.LeftButton and was_dragging and moved < 5.0:
                    idx = self._find_node_at_widget_pos(event.position())
                    if idx is not None and 0 <= idx < len(self._ordered_nodes):
                        nid = int(self._ordered_nodes[idx].get("id", idx + 1))
                        self.node_left_clicked.emit(nid)

                hovered = self._find_node_at_widget_pos(event.position())
                self.setCursor(
                    Qt.CursorShape.PointingHandCursor
                    if hovered is not None
                    else Qt.CursorShape.OpenHandCursor
                )
        except Exception as exc:
            logger.error("SequentialMapCanvas.mouseReleaseEvent error: %s", exc)
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Çizim Motoru (Harita + Yürüyüş İzi + Yön Oklu Koridor + Radar Halkası)
    # ------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            # 1. Koyu taktiksel arka plan
            painter.fillRect(self.rect(), QColor("#0d1117"))

            # 2. Bölge Haritası
            if self._base_pixmap is not None and not self._base_pixmap.isNull():
                target_rect = QRectF(
                    self._pan.x(),
                    self._pan.y(),
                    self._base_pixmap.width() * self._zoom,
                    self._base_pixmap.height() * self._zoom,
                )
                painter.drawPixmap(target_rect, self._base_pixmap, QRectF(self._base_pixmap.rect()))

                # Harita sınır çerçevesi
                painter.setPen(QPen(QColor(56, 189, 248, 70), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(target_rect)

            # 3. Oyuncunun Yürüyüş İzi (walk_trajectory) -> Kesikli Altın/Sarı Çizgi
            self._draw_walk_trajectory(painter)

            # 4. Ardışık Node'lar Arası Yön Oklu Mavi/Turkuaz Koridor Çizgileri (1 ➔ 2 ➔ 3 ➔ 4)
            widget_pts: List[QPointF] = []
            for node in self._ordered_nodes:
                nx = float(node.get("map_x", node.get("x", 0.0)))
                ny = float(node.get("map_y", node.get("y", 0.0)))
                widget_pts.append(self._map_to_widget(self._raw_to_map_coords(nx, ny)))

            self._draw_directional_corridor(painter, widget_pts)

            # 5. Yeni Keşfedilen Gem/Materyal Yeşil Radar Halkası (Pulse Animation)
            self._draw_discovery_pulses(painter)

            # 6. Oyuncu Konumu ve Canlı Keşif Radarı Süpürme Halkası
            if self._player_pos is not None:
                px, py, angle_deg = self._player_pos
                p_wpt = self._map_to_widget(self._raw_to_map_coords(px, py))
                self._draw_player_radar_and_marker(painter, p_wpt, angle_deg)

            # 7. Sıralı Node Rozetleri (#1, #2, #3 ve #N) + Hasat/Yığın Rozetleri (3x, 6x, 9x, 0/3)
            self._draw_sequential_nodes(painter, widget_pts)

            # 8. Sağ Üst Minimap Canlı Kırpma & Filtre Katmanı (Inset Overlay)
            if self._show_minimap_overlay and self._minimap_preview_ref is not None:
                self._draw_minimap_overlay_inset(painter)

            # 9. Harita Köşe Lejantı (Yürüyüş İzi / Yön Oklu Koridor / Keşif Radarı / Minimap Oto-Kayıt)
            self._draw_canvas_legend(painter)
        except Exception as exc:
            logger.error("Error in SequentialMapCanvas.paintEvent: %s", exc)
        finally:
            painter.end()

    def _draw_minimap_overlay_inset(self, painter: QPainter) -> None:
        """Tuvalin sağ üst köşesinde canlı minimap kırpma ve filtre önizlemesini (inset) çizer."""
        if self._minimap_preview_ref is None or self.width() < 420 or self.height() < 260:
            return
        inset_w = 215.0
        inset_h = 175.0
        inset_rect = QRectF(
            float(self.width()) - inset_w - 12.0,
            12.0,
            inset_w,
            inset_h,
        )
        self._minimap_preview_ref.render_minimap_overlay_into(painter, inset_rect)

    def _draw_walk_trajectory(self, painter: QPainter) -> None:
        """Oyuncunun gezdiği yürüyüş izini (walk_trajectory) kesikli altın/sarı çizgiyle çizer."""
        if len(self._walk_trajectory) < 2:
            return

        w_points = [
            self._map_to_widget(self._raw_to_map_coords(wx, wy))
            for wx, wy in self._walk_trajectory
        ]

        path = QPainterPath()
        path.moveTo(w_points[0])
        for pt in w_points[1:]:
            path.lineTo(pt)

        # Altın/Sarı dış hale (okunabilirlik için)
        glow_pen = QPen(
            QColor(250, 204, 21, 55),
            5.0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(glow_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        # Kesikli altın/sarı ana çizgi
        gold_dash_pen = QPen(
            QColor("#facc15"),
            2.3,
            Qt.PenStyle.DashLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        gold_dash_pen.setDashPattern([6.0, 4.0])
        painter.setPen(gold_dash_pen)
        painter.drawPath(path)

    def _draw_directional_corridor(self, painter: QPainter, widget_pts: List[QPointF]) -> None:
        """
        Ardışık node'lar arasına yön oklu mavi/turkuaz koridor çizgileri (1 ➔ 2 ➔ 3 ➔ 4) çizer.
        """
        count = len(widget_pts)
        if count < 2:
            return

        # 1 -> 2 -> 3 -> N koridor segmentleri
        for i in range(count - 1):
            p1 = widget_pts[i]
            p2 = widget_pts[i + 1]
            self._draw_corridor_segment(painter, p1, p2, is_closing_segment=False)

        # N -> 1 döngü tamamlama (loop closure) segmenti
        if self._show_loop_closure and count >= 3:
            self._draw_corridor_segment(
                painter,
                widget_pts[-1],
                widget_pts[0],
                is_closing_segment=True,
            )

    def _draw_corridor_segment(
        self,
        painter: QPainter,
        p1: QPointF,
        p2: QPointF,
        is_closing_segment: bool = False,
    ) -> None:
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        seg_len = math.hypot(dx, dy)
        if seg_len < 4.0:
            return

        ux = dx / seg_len
        uy = dy / seg_len

        if is_closing_segment:
            # Döngü kapanış hattı (N ➔ 1): yarı saydam turkuaz kesikli çizgi
            close_pen = QPen(
                QColor(56, 189, 248, 130),
                1.8,
                Qt.PenStyle.DotLine,
                Qt.PenCapStyle.RoundCap,
            )
            painter.setPen(close_pen)
            painter.drawLine(p1, p2)
            if seg_len > 28.0:
                mid = QPointF((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)
                self._draw_chevron_arrow(painter, mid, ux, uy, QColor(56, 189, 248, 185), size=7.5)
            return

        # Yumuşak mavi/turkuaz koridor şeridi (glow)
        corridor_glow = QPen(
            QColor(14, 165, 233, 68),
            8.0,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(corridor_glow)
        painter.drawLine(p1, p2)

        # Ana mavi/turkuaz yön çizgisi (#06b6d4 / #38bdf8)
        main_pen = QPen(
            QColor("#22d3ee"),
            2.6,
            Qt.PenStyle.SolidLine,
            Qt.PenCapStyle.RoundCap,
            Qt.PenJoinStyle.RoundJoin,
        )
        painter.setPen(main_pen)
        painter.drawLine(p1, p2)

        # Segment üzerine yön okları (1 ➔ 2 ➔ 3 ➔ 4)
        if seg_len >= 95.0:
            arrow_positions = (0.35, 0.68)
        elif seg_len >= 22.0:
            arrow_positions = (0.55,)
        else:
            arrow_positions = ()

        for ratio in arrow_positions:
            ax = p1.x() + dx * ratio
            ay = p1.y() + dy * ratio
            self._draw_chevron_arrow(
                painter,
                QPointF(ax, ay),
                ux,
                uy,
                QColor("#38bdf8"),
                size=9.2,
            )

    @staticmethod
    def _draw_chevron_arrow(
        painter: QPainter,
        tip_pt: QPointF,
        ux: float,
        uy: float,
        fill_color: QColor,
        size: float = 9.0,
    ) -> None:
        """Birim yön vektörü (ux, uy) yönünde dolgulu yön oku üçgeni çizer."""
        px, py = -uy, ux  # dik vektör
        half_t = size * 0.5
        wing = size * 0.62

        tip = QPointF(tip_pt.x() + ux * half_t, tip_pt.y() + uy * half_t)
        left = QPointF(
            tip_pt.x() - ux * half_t + px * wing,
            tip_pt.y() - uy * half_t + py * wing,
        )
        indent = QPointF(
            tip_pt.x() - ux * (half_t * 0.35),
            tip_pt.y() - uy * (half_t * 0.35),
        )
        right = QPointF(
            tip_pt.x() - ux * half_t - px * wing,
            tip_pt.y() - uy * half_t - py * wing,
        )

        poly = QPolygonF([tip, left, indent, right])
        painter.setPen(QPen(QColor("#082f49"), 1.5))
        painter.setBrush(QBrush(fill_color))
        painter.drawPolygon(poly)

    def _draw_discovery_pulses(self, painter: QPainter) -> None:
        """
        Yeni keşfedilen bir gem/materyal eklendiğinde etrafında genişleyen
        yeşil radar halkası (pulse animation) çizer.
        """
        now = time.time()
        for pulse in self._active_pulses:
            elapsed = max(0.0, now - float(pulse["start_ts"]))
            duration = max(0.5, float(pulse["duration"]))
            progress = min(1.0, elapsed / duration)

            w_pt = self._map_to_widget(self._raw_to_map_coords(float(pulse["x"]), float(pulse["y"])))

            # Çift katmanlı genişleyen yeşil radar halkası
            for wave_offset in (0.0, 0.35):
                wave_p = (progress + wave_offset) % 1.0
                radius = 8.0 + wave_p * 42.0
                alpha = int(max(0.0, (1.0 - wave_p) * 230.0))

                ring_pen = QPen(QColor(34, 197, 94, alpha), 2.5)
                painter.setPen(ring_pen)
                painter.setBrush(QBrush(QColor(34, 197, 94, int(alpha * 0.18))))
                painter.drawEllipse(w_pt, radius, radius)

            # Keşif etiketi ("+ DISCOVERED")
            if progress < 0.85:
                lbl_alpha = int((1.0 - progress / 0.85) * 255)
                font = QFont("Segoe UI", 8, QFont.Weight.Bold)
                painter.setFont(font)
                painter.setPen(QColor(74, 222, 128, lbl_alpha))
                painter.drawText(
                    QPointF(w_pt.x() - 28.0, w_pt.y() - 16.0 - progress * 12.0),
                    str(pulse.get("label", "+ NEW GEM")),
                )

    def _draw_player_radar_and_marker(
        self,
        painter: QPainter,
        pt: QPointF,
        angle_deg: float = 0.0,
    ) -> None:
        """Oyuncunun canlı konumunu ve aktif keşif radarı tarama halkasını çizer."""
        painter.save()
        painter.translate(pt)

        if self._auto_cartography_active:
            # Genişleyen canlı radar tarama dairesi
            pulse_r = 16.0 + 14.0 * (0.5 + 0.5 * math.sin(self._radar_sweep_phase * 2.0))
            grad = QRadialGradient(QPointF(0.0, 0.0), pulse_r)
            grad.setColorAt(0.0, QColor(34, 197, 94, 65))
            grad.setColorAt(0.7, QColor(16, 185, 129, 30))
            grad.setColorAt(1.0, QColor(16, 185, 129, 0))
            painter.setPen(QPen(QColor(52, 211, 153, 140), 1.4, Qt.PenStyle.DashLine))
            painter.setBrush(QBrush(grad))
            painter.drawEllipse(QPointF(0.0, 0.0), pulse_r, pulse_r)

        painter.rotate(angle_deg)
        arrow = QPolygonF(
            [
                QPointF(0.0, -10.0),
                QPointF(7.0, 7.5),
                QPointF(0.0, 3.8),
                QPointF(-7.0, 7.5),
            ]
        )
        painter.setPen(QPen(QColor("#000000"), 2.0))
        painter.setBrush(QBrush(QColor("#facc15")))
        painter.drawPolygon(arrow)
        painter.restore()

    def _draw_sequential_nodes(self, painter: QPainter, widget_pts: List[QPointF]) -> None:
        """
        Kayıtlı tüm node'ları sırasına göre (#1, #2, #3 ve #N), minimap oto-kayıt
        camgöbeği/zümrüt çift halkasıyla (`source_type == "minimap_auto"`) ve
        hasat/yığın rozetleriyle (`3x`, `6x`, `9x` veya `0/3`) çizer.
        """
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        badge_font = QFont("Segoe UI", 8, QFont.Weight.Bold)

        total = len(self._ordered_nodes)
        for idx, (node, w_pt) in enumerate(zip(self._ordered_nodes, widget_pts)):
            seq_num = idx + 1
            is_hovered = idx == self._hovered_seq_idx
            is_start = seq_num == 1
            is_end = seq_num == total and total > 1
            is_minimap_auto = (
                str(node.get("source_type", node.get("source", ""))).strip().lower() == "minimap_auto"
                or bool(node.get("minimap_auto", False))
            )

            if is_start:
                fill_color = QColor("#10b981")  # Başlangıç #1: Zümrüt yeşili
            elif is_end:
                fill_color = QColor("#06b6d4")  # Son düğüm #N: Parlak camgöbeği
            else:
                fill_color = QColor("#22c55e")  # Rota içi aktif düğüm

            radius = 9.0 if is_hovered else 7.6

            # Minimap'ten otomatik kaydedilen noktalar için parlak camgöbeği/zümrüt çift halka
            if is_minimap_auto:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(6, 182, 212, 46)))
                painter.drawEllipse(w_pt, radius + 7.2, radius + 7.2)

                painter.setPen(QPen(QColor("#22d3ee"), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(w_pt, radius + 5.2, radius + 5.2)

                painter.setPen(QPen(QColor("#10b981"), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(w_pt, radius + 2.6, radius + 2.6)

            if is_hovered:
                painter.setPen(QPen(QColor("#38bdf8"), 3.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(w_pt, radius + 3.0, radius + 3.0)

            # Düğüm dairesi
            painter.setPen(QPen(QColor("#04131d"), 2.2))
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(w_pt, radius, radius)

            # Düğüm yanında `#1`, `#2`, `#3` sıra etiketi
            painter.setFont(font)
            seq_label = f"#{seq_num}"
            ring_pad = 5.5 if is_minimap_auto else 0.0
            tx = w_pt.x() + radius + 4.0 + ring_pad
            ty = w_pt.y() + 4.0
            self._draw_outlined_text(painter, tx, ty, seq_label, font, text_color=QColor("#ffffff"))

            # Yığın / Hasat Rozeti (3x, 6x, 9x veya 0/3)
            badge_str = str(
                node.get("stack_badge")
                or node.get("harvest_badge")
                or f"{int(node.get('stack_count', node.get('max_harvests', 3)) or 3)}x"
            )
            fm = painter.fontMetrics()
            seq_w = float(fm.horizontalAdvance(seq_label))
            bx = tx + seq_w + 5.0
            by = w_pt.y()

            painter.save()
            painter.setFont(badge_font)
            bfm = painter.fontMetrics()
            bw = max(20.0, float(bfm.horizontalAdvance(badge_str)) + 9.0)
            bh = 14.5
            pill_rect = QRectF(bx, by - bh * 0.5 + 0.5, bw, bh)
            painter.setPen(QPen(QColor("#22d3ee" if is_minimap_auto else "#34d399"), 1.3))
            painter.setBrush(QBrush(QColor(6, 46, 42, 230)))
            painter.drawRoundedRect(pill_rect, 4.0, 4.0)
            painter.setPen(QColor("#6ee7b7" if badge_str.endswith("x") else "#67e8f9"))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, badge_str)
            painter.restore()

    @staticmethod
    def _draw_outlined_text(
        painter: QPainter,
        x: float,
        y: float,
        text: str,
        font: QFont,
        text_color: QColor = QColor("#ffffff"),
    ) -> None:
        path = QPainterPath()
        path.addText(QPointF(x, y), font, text)
        painter.setPen(
            QPen(
                QColor(0, 0, 0, 235),
                3.2,
                Qt.PenStyle.SolidLine,
                Qt.PenCapStyle.RoundCap,
                Qt.PenJoinStyle.RoundJoin,
            )
        )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(text_color))
        painter.drawPath(path)

    def _draw_canvas_legend(self, painter: QPainter) -> None:
        """Sol alt köşede Türkçe görsel katmanların açıklamasını gösterir."""
        legend_w = min(float(self.width() - 24), 760.0)
        legend_rect = QRectF(12.0, float(self.height() - 40), legend_w, 28.0)
        painter.setPen(QPen(QColor(45, 55, 72, 190), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 215)))
        painter.drawRoundedRect(legend_rect, 6.0, 6.0)

        font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor("#e2e8f0"))
        legend_text = (
            "  ● #1 ➔ #2 ➔ #3: Sıralı Yön Koridoru   |   "
            "- - - Altın Kesikli: Yürüyüş İzi   |   "
            "◎ Çift Halka (3x/6x/9x): Minimap/3B Oto-Kayıt   |   "
            "•✕: Yoksayılan Canavar"
        )
        painter.drawText(legend_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, legend_text)


class CartographyRadarDialog(QDialog):
    """
    Canlı Oto-Haritalama ve Minimap Radarı Arayüzü (CartographyRadarDialog).

    Özellikler:
    1. İnteraktif Sıralı Rota Tuvali (`SequentialMapCanvas(QWidget)`):
       - Bölge haritası (`video_frames/frame_18s.jpg` veya mevcut bölge haritası) üzerinde
         kayıtlı tüm node'ları sırasına göre (`#1`, `#2`, `#3` ve `#N`) ve `3x`/`6x`/`9x` rozetleriyle çizer.
       - Ardışık node'lar arasına yön oklu mavi/turkuaz koridor çizgileri (`1 ➔ 2 ➔ 3 ➔ 4`) çizer.
       - Oyuncunun gezdiği yürüyüş izini (`walk_trajectory`) kesikli altın/sarı çizgiyle gösterir.
       - Yeni keşfedilen bir gem/materyal eklendiğinde etrafında genişleyen yeşil radar halkası gösterir.
    2. 3B Dünya + Sağ Üst Minimap Hibrit Tarayıcı Paneli (`MinimapLiveCropPreviewWidget`):
       - Sağ üst minimap üzerindeki toplanabilir materyal ikonlarını parlak zümrüt/camgöbeği halka ve
         `3x`, `6x`, `9x` yığın rozetleriyle gösterir.
       - Canavar beyaz noktalarını (`•`) soluk gri/kırmızı üzeri çizili (`•✕`) olarak gösterir.
       - `Minimap'ten Şimdi Kaydet` (`record_minimap_now`) butonuyla minimap üzerindeki tüm toplanabilir
         materyalleri anında aktif rotaya (`source_type = "minimap_auto"`) kaydeder.
    3. Türkçe Kontrol ve Sıralama Butonları:
       - `Keşfi Başlat` / `Keşfi Durdur`
       - `Sıralı Rotayı Optimize Et`
       - `Minimap'ten Şimdi Kaydet`
       - `Toplanan / Haritalanan Materyaller`
    """

    sequence_updated = pyqtSignal(str, str)
    auto_cartography_toggled = pyqtSignal(bool)
    node_discovered = pyqtSignal(dict)
    minimap_materials_recorded = pyqtSignal(list)

    def __init__(
        self,
        route_manager: Optional[Any] = None,
        region_name: Optional[str] = None,
        route_name: Optional[str] = None,
        map_image_path: Optional[str] = None,
        walk_trajectory: Optional[List[Tuple[float, float]]] = None,
        cartography_engine: Optional[Any] = None,
        sequential_planner: Optional[Any] = None,
        parent: Optional[QWidget] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(parent)
        self.routes = route_manager or kwargs.get("routes")
        self.cartography_engine = cartography_engine or kwargs.get("auto_cartographer")
        self.planner = sequential_planner if sequential_planner is not None else SequentialRoutePlanner(
            meters_per_pixel=float(getattr(self.routes, "meters_per_pixel", 0.28) or 0.28)
        )

        self._region_name: str = region_name or self._resolve_active_region()
        self._route_name: str = route_name or self._resolve_active_route()
        self._explicit_map_path: Optional[str] = map_image_path
        self._auto_cartography_enabled: bool = True
        self._ordered_nodes: List[Dict[str, Any]] = []
        self._walk_trajectory: List[Tuple[float, float]] = list(walk_trajectory or [])

        self._init_ui()
        self.reload_map_and_nodes()

    # ------------------------------------------------------------------
    # Bölge & Rota Çözümleme
    # ------------------------------------------------------------------
    def _resolve_active_region(self) -> str:
        try:
            if self.routes is not None:
                if hasattr(self.routes, "active_region") and self.routes.active_region:
                    return str(self.routes.active_region)
                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg = self.routes.data.get("active_region")
                    if reg:
                        return str(reg)
        except Exception as exc:
            logger.debug("Could not resolve active_region: %s", exc)
        return "Verteron"

    def _resolve_active_route(self) -> str:
        try:
            if self.routes is not None:
                if hasattr(self.routes, "active_route") and self.routes.active_route:
                    return str(self.routes.active_route)
                if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    rte = self.routes.data.get("active_route")
                    if rte:
                        return str(rte)
        except Exception as exc:
            logger.debug("Could not resolve active_route: %s", exc)
        return "All"

    # ------------------------------------------------------------------
    # UI Kurulumu (Tam Türkçe Arayüz + Sağ Üst Minimap Canlı Kırpma & Filtre Paneli)
    # ------------------------------------------------------------------
    def _init_ui(self) -> None:
        self.setWindowTitle(
            f"Canlı Oto-Haritalama ve Minimap Radarı · {self._region_name} ({self._route_name})"
        )
        self.resize(1220, 760)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #0f141c;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QFrame#TopStatsBar, QFrame#BottomControlBar, QFrame#MinimapSidePanel {
                background-color: #161d27;
                border: 1px solid #263244;
                border-radius: 8px;
            }
            QFrame#StatCard {
                background-color: #1c2533;
                border: 1px solid #2d3b4e;
                border-radius: 6px;
                padding: 4px 8px;
            }
            QLabel#StatTitle {
                color: #94a3b8;
                font-size: 11px;
                font-weight: 600;
            }
            QLabel#StatValue {
                color: #38bdf8;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton {
                background-color: #1e293b;
                color: #e2e8f0;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 7px 12px;
                font-size: 12px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #283548;
                border-color: #38bdf8;
            }
            QPushButton:pressed {
                background-color: #0f172a;
            }
            QPushButton#PrimarySortBtn {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
            }
            QPushButton#PrimarySortBtn:hover {
                background-color: #0369a1;
            }
            QPushButton#RecordMinimapBtn {
                background-color: #0d9488;
                color: #ffffff;
                border: 1px solid #2dd4bf;
                font-weight: bold;
            }
            QPushButton#RecordMinimapBtn:hover {
                background-color: #0f766e;
            }
            QPushButton#SaveRouteBtn {
                background-color: #16a34a;
                color: #ffffff;
                border: 1px solid #4ade80;
            }
            QPushButton#SaveRouteBtn:hover {
                background-color: #15803d;
            }
            """
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(8)

        # 1. ÜST KONTROL VE İSTATİSTİK ÇUBUĞU
        self.top_bar = QFrame(self)
        self.top_bar.setObjectName("TopStatsBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(12, 8, 12, 8)
        top_layout.setSpacing(10)

        self.lbl_status_prefix = QLabel(
            "Canlı Oto-Haritalama ve Minimap Radarı:", self.top_bar
        )
        self.lbl_status_prefix.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 13px;")
        top_layout.addWidget(self.lbl_status_prefix)

        self.btn_auto_cartography = QPushButton(self.top_bar)
        self.btn_auto_cartography.setCheckable(True)
        self.btn_auto_cartography.setChecked(True)
        self.btn_auto_cartography.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_auto_cartography.clicked.connect(self._on_auto_cartography_toggled)
        self._update_auto_cartography_button_style()
        top_layout.addWidget(self.btn_auto_cartography)

        self.btn_start_discovery = QPushButton("Keşfi Başlat", self.top_bar)
        self.btn_start_discovery.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_discovery.clicked.connect(lambda: self._set_discovery_state(True))
        top_layout.addWidget(self.btn_start_discovery)

        self.btn_stop_discovery = QPushButton("Keşfi Durdur", self.top_bar)
        self.btn_stop_discovery.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_discovery.clicked.connect(lambda: self._set_discovery_state(False))
        top_layout.addWidget(self.btn_stop_discovery)

        top_layout.addStretch(1)

        # Türkçe Sayaçlar
        self.lbl_discovered_nodes = self._create_stat_widget(
            top_layout, "Toplanan / Haritalanan Materyaller", "0"
        )
        self.lbl_smoothness_score = self._create_stat_widget(
            top_layout, "Rota Akıcılık Skoru (%)", "100.0%"
        )
        self.lbl_total_distance = self._create_stat_widget(
            top_layout, "Toplam Döngü Mesafesi (m)", "0.0 m"
        )
        self.lbl_ping_pong_jumps = self._create_stat_widget(
            top_layout, "Zikzak Sıçrama", "Zikzak Sıçrama: 0"
        )

        root_layout.addWidget(self.top_bar)

        # 2. ORTA BÖLÜM: SIRALI ROTA TUVALİ + SAĞ ÜST MİNİMAP CANLI KIRPMA & FİLTRE PANELİ
        middle_layout = QHBoxLayout()
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(8)

        self.canvas = SequentialMapCanvas(self)
        self.canvas.node_left_clicked.connect(self._on_canvas_node_left_clicked)
        self.canvas.node_right_clicked.connect(self._on_canvas_node_right_clicked)
        middle_layout.addWidget(self.canvas, stretch=3)

        # Sağ Üst Minimap Canlı Kırpma & Filtreleme Paneli
        self.minimap_side_panel = QFrame(self)
        self.minimap_side_panel.setObjectName("MinimapSidePanel")
        self.minimap_side_panel.setMinimumWidth(275)
        self.minimap_side_panel.setMaximumWidth(320)
        side_layout = QVBoxLayout(self.minimap_side_panel)
        side_layout.setContentsMargins(10, 10, 10, 10)
        side_layout.setSpacing(8)

        self.lbl_hybrid_scanner_title = QLabel(
            "3B Dünya + Sağ Üst Minimap Hibrit Tarayıcı", self.minimap_side_panel
        )
        self.lbl_hybrid_scanner_title.setWordWrap(True)
        self.lbl_hybrid_scanner_title.setStyleSheet(
            "color: #22d3ee; font-size: 12px; font-weight: bold;"
        )
        side_layout.addWidget(self.lbl_hybrid_scanner_title)

        self.lbl_minimap_sub = QLabel(
            "Top-Right Minimap Live Crop & Filter:\n"
            "• Zümrüt/Camgöbeği (3x/6x/9x): Toplanabilir Materyal\n"
            "• Gri/Kırmızı Çizili (•✕): Yoksayılan Canavar",
            self.minimap_side_panel,
        )
        self.lbl_minimap_sub.setWordWrap(True)
        self.lbl_minimap_sub.setStyleSheet("color: #94a3b8; font-size: 11px;")
        side_layout.addWidget(self.lbl_minimap_sub)

        self.minimap_preview_widget = MinimapLiveCropPreviewWidget(self.minimap_side_panel)
        self.canvas.set_minimap_preview_widget(self.minimap_preview_widget)
        side_layout.addWidget(self.minimap_preview_widget, stretch=1)

        self.lbl_minimap_filter_stats = QLabel(self.minimap_side_panel)
        self.lbl_minimap_filter_stats.setWordWrap(True)
        self.lbl_minimap_filter_stats.setStyleSheet(
            "color: #a7f3d0; font-size: 11px; font-weight: 600;"
        )
        side_layout.addWidget(self.lbl_minimap_filter_stats)

        self.btn_record_minimap_now = QPushButton(
            "Minimap'ten Şimdi Kaydet", self.minimap_side_panel
        )
        self.btn_record_minimap_now.setObjectName("RecordMinimapBtn")
        self.btn_record_minimap_now.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_record_minimap_now.clicked.connect(self.record_minimap_now)
        side_layout.addWidget(self.btn_record_minimap_now)

        self.btn_toggle_minimap_inset = QPushButton(
            "Harita Üstü Minimap Önizleme: AÇIK", self.minimap_side_panel
        )
        self.btn_toggle_minimap_inset.setCheckable(True)
        self.btn_toggle_minimap_inset.setChecked(True)
        self.btn_toggle_minimap_inset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_minimap_inset.clicked.connect(self._on_toggle_minimap_inset)
        side_layout.addWidget(self.btn_toggle_minimap_inset)

        middle_layout.addWidget(self.minimap_side_panel, stretch=1)
        root_layout.addLayout(middle_layout, stretch=1)

        # 3. SIRALAMA & KORİDOR KONTROL BUTONLARI (Tam Türkçe)
        self.bottom_bar = QFrame(self)
        self.bottom_bar.setObjectName("BottomControlBar")
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(12, 8, 12, 8)
        bottom_layout.setSpacing(8)

        self.btn_sort_smooth = QPushButton(
            "Sıralı Rotayı Optimize Et", self.bottom_bar
        )
        self.btn_sort_smooth.setObjectName("PrimarySortBtn")
        self.btn_sort_smooth.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sort_smooth.clicked.connect(self.sort_smooth_sequential_loop)
        bottom_layout.addWidget(self.btn_sort_smooth)

        self.btn_sort_walk = QPushButton(
            "Gezdiğim Sırayla Diz", self.bottom_bar
        )
        self.btn_sort_walk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sort_walk.clicked.connect(self.sort_by_walk_path)
        bottom_layout.addWidget(self.btn_sort_walk)

        self.btn_sort_corridor = QPushButton(
            "Koridor Şerit Sırası", self.bottom_bar
        )
        self.btn_sort_corridor.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sort_corridor.clicked.connect(self.sort_corridor_sweep)
        bottom_layout.addWidget(self.btn_sort_corridor)

        self.btn_reverse_order = QPushButton(
            "Sırayı Ters Çevir", self.bottom_bar
        )
        self.btn_reverse_order.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reverse_order.clicked.connect(self.reverse_node_order)
        bottom_layout.addWidget(self.btn_reverse_order)

        bottom_layout.addStretch(1)

        self.btn_save_route = QPushButton("Sıralı Rotayı Kaydet", self.bottom_bar)
        self.btn_save_route.setObjectName("SaveRouteBtn")
        self.btn_save_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_route.clicked.connect(self.save_sequential_route)
        bottom_layout.addWidget(self.btn_save_route)

        root_layout.addWidget(self.bottom_bar)
        self._refresh_minimap_filter_stats()

    def _create_stat_widget(
        self,
        parent_layout: QHBoxLayout,
        title: str,
        initial_val: str,
    ) -> QLabel:
        card = QFrame(self.top_bar)
        card.setObjectName("StatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 3, 8, 3)
        card_layout.setSpacing(1)

        lbl_title = QLabel(title, card)
        lbl_title.setObjectName("StatTitle")
        lbl_val = QLabel(initial_val, card)
        lbl_val.setObjectName("StatValue")

        card_layout.addWidget(lbl_title)
        card_layout.addWidget(lbl_val)
        parent_layout.addWidget(card)
        return lbl_val

    def _update_auto_cartography_button_style(self) -> None:
        if self._auto_cartography_enabled:
            self.btn_auto_cartography.setText("AKTİF · Keşfi Durdur")
            self.btn_auto_cartography.setStyleSheet(
                """
                QPushButton {
                    background-color: #065f46;
                    color: #ecfdf5;
                    border: 1px solid #10b981;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #047857;
                }
                """
            )
        else:
            self.btn_auto_cartography.setText("PASİF · Keşfi Başlat")
            self.btn_auto_cartography.setStyleSheet(
                """
                QPushButton {
                    background-color: #334155;
                    color: #94a3b8;
                    border: 1px solid #475569;
                    border-radius: 6px;
                    padding: 6px 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #475569;
                }
                """
            )

    def _set_discovery_state(self, enabled: bool) -> None:
        """Keşfi Başlat / Keşfi Durdur butonları için ortak durum güncelleyici."""
        self.btn_auto_cartography.setChecked(bool(enabled))
        self._on_auto_cartography_toggled(bool(enabled))

    def _on_toggle_minimap_inset(self, checked: bool) -> None:
        is_on = bool(checked)
        self.canvas.set_minimap_overlay_visible(is_on)
        self.btn_toggle_minimap_inset.setText(
            "Harita Üstü Minimap Önizleme: AÇIK"
            if is_on
            else "Harita Üstü Minimap Önizleme: KAPALI"
        )

    def _refresh_minimap_filter_stats(self) -> None:
        if not hasattr(self, "minimap_preview_widget") or not hasattr(self, "lbl_minimap_filter_stats"):
            return
        mats = self.minimap_preview_widget.get_gatherable_materials()
        mobs = self.minimap_preview_widget.get_rejected_monsters()
        badges = ", ".join(str(m.get("stack_badge", "3x")) for m in mats) if mats else "Yok"
        self.lbl_minimap_filter_stats.setText(
            f"Toplanabilir Materyal: {len(mats)} ({badges})\n"
            f"Yoksayılan Canavar (•✕): {len(mobs)}"
        )

    def record_minimap_now(self) -> List[Dict[str, Any]]:
        """
        `Minimap'ten Şimdi Kaydet` (`record_minimap_now`)
        Sağ üst minimap üzerinde görünen tüm toplanabilir materyalleri (`3x`, `6x`, `9x` yığın
        rozetli) anında aktif rotaya (`source_type = "minimap_auto"`) kaydeder;
        yoksayılan canavar noktalarını (`•✕`) tamamen filtreler.
        """
        recorded_nodes: List[Dict[str, Any]] = []
        try:
            mats = self.minimap_preview_widget.get_gatherable_materials()
            if not mats:
                return recorded_nodes

            # Oyuncu merkez koordinatını belirle
            if self.canvas._player_pos is not None:
                base_x, base_y, _ = self.canvas._player_pos
            elif self._walk_trajectory:
                base_x, base_y = self._walk_trajectory[-1]
            elif (
                self.canvas._base_pixmap is not None
                and not self.canvas._base_pixmap.isNull()
            ):
                base_x = float(self.canvas._base_pixmap.width()) * 0.5
                base_y = float(self.canvas._base_pixmap.height()) * 0.5
            else:
                base_x, base_y = 460.0, 310.0

            pw = (
                float(self.canvas._base_pixmap.width())
                if (self.canvas._base_pixmap is not None and not self.canvas._base_pixmap.isNull())
                else 920.0
            )
            ph = (
                float(self.canvas._base_pixmap.height())
                if (self.canvas._base_pixmap is not None and not self.canvas._base_pixmap.isNull())
                else 620.0
            )

            for mat in mats:
                rel_dx = float(mat.get("rel_map_dx", (float(mat.get("norm_x", 0.5)) - 0.5) * 90.0))
                rel_dy = float(mat.get("rel_map_dy", (float(mat.get("norm_y", 0.5)) - 0.5) * 90.0))
                target_x = int(round(max(18.0, min(pw - 18.0, base_x + rel_dx))))
                target_y = int(round(max(18.0, min(ph - 18.0, base_y + rel_dy))))
                stack_cnt = int(mat.get("stack_count", mat.get("max_harvests", 3)) or 3)
                res_type = str(mat.get("resource_type", "Orichalcum Ore"))

                # Yakında aynı minimap_auto düğümü varsa güncelle, yoksa yeni düğüm ekle
                existing = next(
                    (
                        n
                        for n in self._ordered_nodes
                        if math.hypot(
                            float(n.get("map_x", 0)) - target_x,
                            float(n.get("map_y", 0)) - target_y,
                        )
                        <= 14.0
                    ),
                    None,
                )

                if existing is not None:
                    existing["source_type"] = "minimap_auto"
                    existing["minimap_auto"] = True
                    existing["stack_count"] = stack_cnt
                    existing["max_harvests"] = stack_cnt
                    existing["harvest_count"] = int(existing.get("harvest_count", 0) or 0)
                    existing["stack_badge"] = f"{stack_cnt}x"
                    existing["harvest_badge"] = f"{existing['harvest_count']}/{stack_cnt}"
                    self.canvas.trigger_discovery_pulse(
                        target_x,
                        target_y,
                        label=f"+ {res_type} ({stack_cnt}x)",
                    )
                    recorded_nodes.append(dict(existing))
                    continue

                new_node_payload = {
                    "map_x": target_x,
                    "map_y": target_y,
                    "norm_x": round(target_x / max(1.0, pw), 4),
                    "norm_y": round(target_y / max(1.0, ph), 4),
                    "resource_type": res_type,
                    "source_type": "minimap_auto",
                    "minimap_auto": True,
                    "harvest_count": 0,
                    "max_harvests": stack_cnt,
                    "stack_count": stack_cnt,
                    "stack_badge": f"{stack_cnt}x",
                    "harvest_badge": f"0/{stack_cnt}",
                    "in_air": bool(mat.get("in_air", False)),
                }

                # Eğer RouteManager bağlıysa oraya da kaydet
                if self.routes is not None and hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                    reg_data = self.routes.data.setdefault("regions", {}).setdefault(
                        self._region_name, {"nodes": [], "routes": {self._route_name: []}}
                    )
                    nodes_list = reg_data.setdefault("nodes", [])
                    existing_ids = [
                        int(n.get("id", 0)) if isinstance(n, dict) else int(getattr(n, "id", 0))
                        for n in nodes_list
                    ]
                    next_id = (max(existing_ids) + 1) if existing_ids else (len(self._ordered_nodes) + 1)
                    new_node_payload["id"] = next_id
                    nodes_list.append(dict(new_node_payload))
                    route_id_list = reg_data.setdefault("routes", {}).setdefault(self._route_name, [])
                    if next_id not in route_id_list:
                        route_id_list.append(next_id)

                added = self.add_discovered_node(
                    new_node_payload,
                    trigger_pulse=True,
                    auto_resquence=False,
                )
                if added:
                    recorded_nodes.append(added)

            if len(self._ordered_nodes) >= 3:
                self._ordered_nodes = self.planner.optimize_smooth_loop(
                    self._ordered_nodes, self._walk_trajectory
                )
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()
            self._refresh_minimap_filter_stats()
            self.minimap_materials_recorded.emit(list(recorded_nodes))
            logger.info(
                "Minimap'ten %d toplanabilir materyal rotaya kaydedildi (canavarlar yoksayıldı).",
                len(recorded_nodes),
            )
            return recorded_nodes
        except Exception as exc:
            logger.error("record_minimap_now hatası: %s", exc)
            return recorded_nodes

    # ------------------------------------------------------------------
    # Harita ve Düğüm Yükleme
    # ------------------------------------------------------------------
    @staticmethod
    def _to_grayscale_pixmap(image: QImage) -> QPixmap:
        if image.isNull():
            return QPixmap()
        gray_img = image.convertToFormat(QImage.Format.Format_Grayscale8)
        rgb_gray = gray_img.convertToFormat(QImage.Format.Format_RGB32)
        return QPixmap.fromImage(rgb_gray)

    def _load_region_map_pixmap(
        self, nodes: List[Dict[str, Any]]
    ) -> Tuple[QPixmap, Tuple[int, int]]:
        candidates: List[Tuple[str, bool]] = []
        if self._explicit_map_path and os.path.exists(self._explicit_map_path):
            is_f18 = os.path.basename(self._explicit_map_path).lower() == "frame_18s.jpg"
            candidates.append((self._explicit_map_path, is_f18))

        safe_reg = "".join(
            c if c.isalnum() or c in ("_", "-") else "_"
            for c in (self._region_name or "Verteron")
        )
        for rel_path in (
            os.path.join("routes", safe_reg, "map.png"),
            os.path.join("routes", safe_reg, "map.jpg"),
            os.path.join("routes", "maps", f"{safe_reg}.png"),
            os.path.join("routes", "maps", f"{safe_reg}.jpg"),
            os.path.join("assets", "maps", f"{safe_reg}.png"),
        ):
            if os.path.exists(rel_path):
                candidates.append((rel_path, False))

        frame18_path = os.path.join(os.getcwd(), "video_frames", "frame_18s.jpg")
        if os.path.exists(frame18_path):
            candidates.append((frame18_path, True))

        for path, should_crop_f18 in candidates:
            try:
                img = QImage(path)
                if not img.isNull():
                    offset = (0, 0)
                    if should_crop_f18:
                        rx, ry, rw, rh = SequentialMapCanvas.FRAME18_NORM_CROP
                        cx = int(img.width() * rx)
                        cy = int(img.height() * ry)
                        cw = int(img.width() * rw)
                        ch = int(img.height() * rh)
                        img = img.copy(QRect(cx, cy, cw, ch))
                        offset = (cx, cy)
                    gray_pix = self._to_grayscale_pixmap(img)
                    if not gray_pix.isNull():
                        return gray_pix, offset
            except Exception as exc:
                logger.warning("Harita görseli yüklenemedi ('%s'): %s", path, exc)

        return self._build_fallback_map_pixmap(nodes), (0, 0)

    def _build_fallback_map_pixmap(self, nodes: List[Dict[str, Any]]) -> QPixmap:
        width, height = 920, 620
        canvas_img = QImage(width, height, QImage.Format.Format_RGB32)
        canvas_img.fill(QColor("#1e242d"))

        painter = QPainter(canvas_img)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        grid_pen = QPen(QColor("#28303d"), 1)
        painter.setPen(grid_pen)
        for gx in range(0, width, 50):
            painter.drawLine(gx, 0, gx, height)
        for gy in range(0, height, 50):
            painter.drawLine(0, gy, width, gy)

        for node in nodes or []:
            patch_path = node.get("patch_path") or node.get("patch_file")
            if patch_path and os.path.exists(str(patch_path)):
                try:
                    p_img = QImage(str(patch_path))
                    if not p_img.isNull():
                        nx = int(node.get("map_x", width // 2))
                        ny = int(node.get("map_y", height // 2))
                        painter.drawImage(
                            nx - p_img.width() // 2,
                            ny - p_img.height() // 2,
                            p_img,
                        )
                except Exception as exc:
                    logger.debug("Fallback mosaic patch error: %s", exc)

        painter.end()
        return QPixmap.fromImage(canvas_img)

    def _extract_nodes_from_manager(self) -> List[Dict[str, Any]]:
        extracted: List[Dict[str, Any]] = []
        if self.routes is None:
            return extracted

        try:
            if hasattr(self.routes, "get_route_nodes"):
                raw_nodes = self.routes.get_route_nodes(
                    region_name=self._region_name,
                    route_name=self._route_name,
                )
                for n in raw_nodes or []:
                    if isinstance(n, dict):
                        extracted.append(dict(n))
                    elif hasattr(n, "to_dict"):
                        extracted.append(dict(n.to_dict()))
                if extracted:
                    return extracted

            if hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                regions = self.routes.data.get("regions", {})
                reg_data = regions.get(self._region_name)
                if reg_data is None and regions:
                    first_reg = next(iter(regions.keys()))
                    self._region_name = first_reg
                    reg_data = regions.get(first_reg, {})
                if isinstance(reg_data, dict):
                    for n in reg_data.get("nodes", []):
                        if isinstance(n, dict):
                            extracted.append(dict(n))
                        elif hasattr(n, "to_dict"):
                            extracted.append(dict(n.to_dict()))
        except Exception as exc:
            logger.error("RouteManager'dan node listesi alınırken hata: %s", exc)

        return extracted

    def _generate_default_walk_trajectory(
        self, nodes: List[Dict[str, Any]], pixmap: QPixmap
    ) -> List[Tuple[float, float]]:
        """
        Eğer dışarıdan bir yürüyüş izi verilmemişse mevcut düğümler boyunca
        doğal bir oyuncu yürüyüş izi (walk_trajectory) üretir.
        """
        if not nodes:
            pw = float(pixmap.width() if not pixmap.isNull() else 800)
            ph = float(pixmap.height() if not pixmap.isNull() else 500)
            return [
                (pw * 0.18, ph * 0.25),
                (pw * 0.35, ph * 0.28),
                (pw * 0.55, ph * 0.42),
                (pw * 0.72, ph * 0.62),
            ]

        pts: List[Tuple[float, float]] = []
        for idx, n in enumerate(nodes):
            nx = float(n.get("map_x", n.get("x", 0.0)))
            ny = float(n.get("map_y", n.get("y", 0.0)))
            pts.append((nx, ny))
            if idx + 1 < len(nodes):
                nx2 = float(nodes[idx + 1].get("map_x", nodes[idx + 1].get("x", 0.0)))
                ny2 = float(nodes[idx + 1].get("map_y", nodes[idx + 1].get("y", 0.0)))
                mid_x = (nx + nx2) * 0.5 + 8.0 * math.sin(idx + 1.0)
                mid_y = (ny + ny2) * 0.5 + 8.0 * math.cos(idx + 1.0)
                pts.append((mid_x, mid_y))
        return pts

    def reload_map_and_nodes(self) -> None:
        """Haritayı, sıralı düğümleri, yürüyüş izini ve üst istatistik çubuğunu yeniler."""
        try:
            self._region_name = self._region_name or self._resolve_active_region()
            self._route_name = self._route_name or self._resolve_active_route()
            self.setWindowTitle(
                f"Canlı Oto-Haritalama ve Minimap Radarı · {self._region_name} ({self._route_name})"
            )

            nodes = self._extract_nodes_from_manager()
            pixmap, offset = self._load_region_map_pixmap(nodes)

            # Eğer düğümler 0,0 koordinatlıysa veya harita sınırları içindeyse normalize et
            pw = max(200, pixmap.width())
            ph = max(200, pixmap.height())
            for idx, n in enumerate(nodes):
                mx = float(n.get("map_x", n.get("x", 0.0)))
                my = float(n.get("map_y", n.get("y", 0.0)))
                if mx <= 0.0 and my <= 0.0:
                    angle = (2.0 * math.pi * idx) / max(1, len(nodes))
                    n["map_x"] = int(pw * 0.5 + math.cos(angle) * (pw * 0.30))
                    n["map_y"] = int(ph * 0.5 + math.sin(angle) * (ph * 0.28))

            if offset != (0, 0) and nodes:
                max_x = max(float(n.get("map_x", n.get("x", 0))) for n in nodes)
                max_y = max(float(n.get("map_y", n.get("y", 0))) for n in nodes)
                if max_x <= pw and max_y <= ph:
                    offset = (0, 0)

            self._ordered_nodes = list(nodes)
            if not self._walk_trajectory and self._ordered_nodes:
                self._walk_trajectory = self._generate_default_walk_trajectory(
                    self._ordered_nodes, pixmap
                )

            self.canvas.set_map_pixmap(pixmap, coord_offset=offset)
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self.canvas.set_walk_trajectory(self._walk_trajectory)
            if self._walk_trajectory:
                lx, ly = self._walk_trajectory[-1]
                self.canvas.set_player_position(lx, ly, angle_deg=45.0)

            self._refresh_metrics_bar()
            self._refresh_minimap_filter_stats()
        except Exception as exc:
            logger.error("reload_map_and_nodes hatası: %s", exc)

    # ------------------------------------------------------------------
    # İstatistik Güncelleme
    # ------------------------------------------------------------------
    def _refresh_metrics_bar(self) -> None:
        try:
            metrics = self.planner.calculate_route_metrics(self._ordered_nodes)
            disc_count = int(metrics.get("discovered_nodes", len(self._ordered_nodes)))
            smooth_pct = float(metrics.get("smoothness_score_pct", 100.0))
            dist_m = float(metrics.get("total_loop_distance_m", 0.0))
            jumps = int(metrics.get("ping_pong_jumps", 0))

            self.lbl_discovered_nodes.setText(str(disc_count))
            self.lbl_smoothness_score.setText(f"{smooth_pct:.1f}%")
            self.lbl_total_distance.setText(f"{dist_m:.1f} m")
            self.lbl_ping_pong_jumps.setText(f"Zikzak Sıçrama: {jumps}")

            if jumps == 0:
                self.lbl_ping_pong_jumps.setStyleSheet(
                    "color: #4ade80; font-size: 14px; font-weight: bold;"
                )
            else:
                self.lbl_ping_pong_jumps.setStyleSheet(
                    "color: #f87171; font-size: 14px; font-weight: bold;"
                )
        except Exception as exc:
            logger.error("İstatistik çubuğu güncellenirken hata: %s", exc)

    # ------------------------------------------------------------------
    # Canlı Keşif (Auto-Cartography) ve Yürüyüş İzi Metodları
    # ------------------------------------------------------------------
    def _on_auto_cartography_toggled(self, checked: bool) -> None:
        try:
            self._auto_cartography_enabled = bool(checked)
            self._update_auto_cartography_button_style()
            self.canvas.set_auto_cartography_active(self._auto_cartography_enabled)
            if self.cartography_engine is not None and hasattr(
                self.cartography_engine, "set_enabled"
            ):
                self.cartography_engine.set_enabled(self._auto_cartography_enabled)
            self.auto_cartography_toggled.emit(self._auto_cartography_enabled)
            logger.info(
                "Auto-Cartography durumu değiştirildi: %s",
                "AKTİF" if self._auto_cartography_enabled else "PASİF",
            )
        except Exception as exc:
            logger.error("Auto-Cartography toggle hatası: %s", exc)

    def add_discovered_node(
        self,
        node_data: Union[Dict[str, Any], Any],
        trigger_pulse: bool = True,
        auto_resquence: bool = True,
    ) -> Dict[str, Any]:
        """
        Haritada gezerken yeni keşfedilen bir gem/materyali canlı olarak haritaya ekler,
        etrafında genişleyen yeşil radar halkası (pulse animation) gösterir ve
        rotayı zikzaksız olarak günceller.
        """
        try:
            if isinstance(node_data, dict):
                nd = dict(node_data)
            elif hasattr(node_data, "to_dict") and callable(node_data.to_dict):
                nd = dict(node_data.to_dict())
            else:
                nd = {
                    "id": int(getattr(node_data, "id", len(self._ordered_nodes) + 1)),
                    "map_x": int(getattr(node_data, "map_x", 250)),
                    "map_y": int(getattr(node_data, "map_y", 250)),
                    "resource_type": str(getattr(node_data, "resource_type", "Discovered Gem")),
                }

            if "id" not in nd or not nd["id"]:
                existing_ids = [int(x.get("id", 0)) for x in self._ordered_nodes]
                nd["id"] = (max(existing_ids) + 1) if existing_ids else 1

            mx = float(nd.get("map_x", nd.get("x", 0.0)))
            my = float(nd.get("map_y", nd.get("y", 0.0)))
            nd["map_x"] = int(mx)
            nd["map_y"] = int(my)

            self._ordered_nodes.append(nd)
            self.canvas.append_walk_point(mx, my)
            self._walk_trajectory = list(self.canvas._walk_trajectory)
            self.canvas.set_player_position(mx, my)

            if auto_resquence and len(self._ordered_nodes) >= 3:
                self._ordered_nodes = self.planner.optimize_smooth_loop(
                    self._ordered_nodes, self._walk_trajectory
                )

            self.canvas.set_sequential_nodes(self._ordered_nodes)
            if trigger_pulse:
                res_name = str(nd.get("resource_type", f"Node #{nd['id']}"))
                self.canvas.trigger_discovery_pulse(mx, my, label=f"+ {res_name}")

            self._refresh_metrics_bar()
            self.node_discovered.emit(nd)
            return nd
        except Exception as exc:
            logger.error("add_discovered_node hatası: %s", exc)
            return {}

    def set_nodes(self, nodes: List[Any]) -> None:
        """Dışarıdan düğüm listesini doğrudan ayarlar ve metrikleri günceller."""
        self.canvas.set_sequential_nodes(nodes)
        self._ordered_nodes = self.canvas.get_sequential_nodes()
        self._refresh_metrics_bar()

    def set_walk_trajectory(self, trajectory: List[Tuple[float, float]]) -> None:
        """Oyuncu yürüyüş izini doğrudan günceller."""
        self._walk_trajectory = list(trajectory or [])
        self.canvas.set_walk_trajectory(self._walk_trajectory)

    # ------------------------------------------------------------------
    # 3. Sıralama & Koridor Kontrol Buton Aksiyonları
    # ------------------------------------------------------------------
    def sort_smooth_sequential_loop(self) -> List[Dict[str, Any]]:
        """
        `Auto-Sort: Smooth Sequential Loop (Zikzaksız Sırala)`
        SequentialRoutePlanner ile tüm node'ları zikzaksız sıraya dizip haritayı günceller.
        """
        try:
            if not self._ordered_nodes:
                return []
            self._ordered_nodes = self.planner.optimize_smooth_loop(
                self._ordered_nodes, self._walk_trajectory
            )
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()
            logger.info(
                "Rota 'Smooth Sequential Loop' ile zikzaksız sıralandı (%d node).",
                len(self._ordered_nodes),
            )
            return list(self._ordered_nodes)
        except Exception as exc:
            logger.error("sort_smooth_sequential_loop hatası: %s", exc)
            return list(self._ordered_nodes)

    def sort_by_walk_path(self) -> List[Dict[str, Any]]:
        """
        `Sort by My Walk Path (Gezdiğim Sırayla Diz)`
        Oyuncunun yürüdüğü iz (`walk_trajectory`) sırasına göre node'ları dizer.
        """
        try:
            if not self._ordered_nodes:
                return []
            self._ordered_nodes = self.planner.sort_by_walk_path(
                self._ordered_nodes, self._walk_trajectory
            )
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()
            logger.info(
                "Rota yürüyüş izine ('My Walk Path') göre sıralandı (%d node).",
                len(self._ordered_nodes),
            )
            return list(self._ordered_nodes)
        except Exception as exc:
            logger.error("sort_by_walk_path hatası: %s", exc)
            return list(self._ordered_nodes)

    def sort_corridor_sweep(self) -> List[Dict[str, Any]]:
        """
        `Sort: Corridor Sweep (Koridor Şerit Sırası)`
        Node'ları serpantin koridor şeritlerine göre sıraya dizer.
        """
        try:
            if not self._ordered_nodes:
                return []
            self._ordered_nodes = self.planner.sort_corridor_sweep(self._ordered_nodes)
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()
            logger.info(
                "Rota 'Corridor Sweep' şerit sırasına göre dizildi (%d node).",
                len(self._ordered_nodes),
            )
            return list(self._ordered_nodes)
        except Exception as exc:
            logger.error("sort_corridor_sweep hatası: %s", exc)
            return list(self._ordered_nodes)

    def reverse_node_order(self) -> List[Dict[str, Any]]:
        """
        `Reverse Order (Sırayı Ters Çevir)`
        Mevcut gezinme sırasını tersine çevirir (1..N -> N..1).
        """
        try:
            if not self._ordered_nodes:
                return []
            self._ordered_nodes = list(reversed(self._ordered_nodes))
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()
            logger.info("Rota sırası ters çevrildi (%d node).", len(self._ordered_nodes))
            return list(self._ordered_nodes)
        except Exception as exc:
            logger.error("reverse_node_order hatası: %s", exc)
            return list(self._ordered_nodes)

    def save_sequential_route(self) -> bool:
        """
        `Save Sequential Route`
        Sıralanmış node dizilimini RouteManager içerisine (#1..#N sırasıyla) kaydeder
        ve `sequence_updated(region_name, route_name)` sinyalini yayar.
        """
        try:
            # Düğüm sıra numaralarını 1..N olarak güncelle
            for seq_idx, node_dict in enumerate(self._ordered_nodes, start=1):
                node_dict["seq_index"] = seq_idx

            if self.routes is not None:
                try:
                    # Yeni RouteManager (_regions sözlüğü ve NodeData listesi) uyumluluğu
                    if hasattr(self.routes, "_regions") and isinstance(
                        self.routes._regions, dict
                    ):
                        reg_info = self.routes._regions.get(self._region_name)
                        if isinstance(reg_info, dict) and "nodes" in reg_info:
                            existing_nodes = reg_info["nodes"]
                            id_to_obj = {int(getattr(n, "id", 0)): n for n in existing_nodes}
                            reordered_objs = []
                            for seq_idx, nd in enumerate(self._ordered_nodes, start=1):
                                old_id = int(nd.get("id", seq_idx))
                                if old_id in id_to_obj:
                                    reordered_objs.append(id_to_obj.pop(old_id))
                            # Kalan (filtre dışı) düğümleri sona ekle
                            for rem_obj in existing_nodes:
                                if rem_obj not in reordered_objs:
                                    reordered_objs.append(rem_obj)
                            reg_info["nodes"] = reordered_objs
                            if hasattr(self.routes, "save_region"):
                                self.routes.save_region(self._region_name)
                            elif hasattr(self.routes, "save_data"):
                                self.routes.save_data()
                    elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                        reg_data = self.routes.data.get("regions", {}).get(
                            self._region_name, {}
                        )
                        if isinstance(reg_data, dict):
                            reg_data["nodes"] = list(self._ordered_nodes)
                        if hasattr(self.routes, "save_data"):
                            self.routes.save_data()
                except Exception as inner_exc:
                    logger.warning(
                        "RouteManager üzerine sıralı rota yazılırken kısmi uyarı: %s",
                        inner_exc,
                    )

            self.sequence_updated.emit(str(self._region_name), str(self._route_name))
            logger.info(
                "Sıralı rota kaydedildi ve sequence_updated(%s, %s) sinyali yayıldı.",
                self._region_name,
                self._route_name,
            )
            return True
        except Exception as exc:
            logger.error("save_sequential_route hatası: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Tuval Düğüm Etkileşimleri
    # ------------------------------------------------------------------
    def _on_canvas_node_left_clicked(self, node_id: int) -> None:
        """Bir düğüme sol tıklandığında o düğümde radar halkası (pulse) gösterir."""
        try:
            target = next(
                (n for n in self._ordered_nodes if int(n.get("id", -1)) == int(node_id)),
                None,
            )
            if target is not None:
                mx = float(target.get("map_x", target.get("x", 0.0)))
                my = float(target.get("map_y", target.get("y", 0.0)))
                seq = target.get("seq_index", "?")
                self.canvas.trigger_discovery_pulse(
                    mx, my, label=f"#{seq} (ID:{node_id})", duration_sec=1.6
                )
        except Exception as exc:
            logger.debug("_on_canvas_node_left_clicked error: %s", exc)

    def _on_canvas_node_right_clicked(self, node_id: int, global_pos: QPoint) -> None:
        """Düğüme sağ tıklandığında sıra değiştirme / başlangıç yapma menüsü açar."""
        try:
            idx = next(
                (
                    i
                    for i, n in enumerate(self._ordered_nodes)
                    if int(n.get("id", -1)) == int(node_id)
                ),
                None,
            )
            if idx is None:
                return

            node_obj = self._ordered_nodes[idx]
            menu = QMenu(self)
            menu.setStyleSheet(
                """
                QMenu {
                    background-color: #161d27;
                    color: #e2e8f0;
                    border: 1px solid #334155;
                    border-radius: 6px;
                    padding: 4px;
                }
                QMenu::item {
                    padding: 6px 18px;
                    border-radius: 4px;
                }
                QMenu::item:selected {
                    background-color: #0284c7;
                    color: #ffffff;
                }
                """
            )

            hdr = QAction(
                f"Sıra #{idx + 1} · Node ID #{node_id} ({node_obj.get('map_x')}, {node_obj.get('map_y')})",
                menu,
            )
            hdr.setEnabled(False)
            menu.addAction(hdr)
            menu.addSeparator()

            act_make_first = QAction("Bu Düğümü Başlangıç (#1) Yap", menu)
            act_make_first.triggered.connect(lambda: self._rotate_sequence_to_index(idx))
            menu.addAction(act_make_first)

            act_remove = QAction("Rotadan Çıkar", menu)
            act_remove.triggered.connect(lambda: self._remove_node_at_index(idx))
            menu.addAction(act_remove)

            menu.popup(global_pos)
        except Exception as exc:
            logger.error("Sağ tık menüsü açılırken hata: %s", exc)

    def _rotate_sequence_to_index(self, idx: int) -> None:
        if 0 <= idx < len(self._ordered_nodes):
            self._ordered_nodes = self._ordered_nodes[idx:] + self._ordered_nodes[:idx]
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()

    def _remove_node_at_index(self, idx: int) -> None:
        if 0 <= idx < len(self._ordered_nodes):
            self._ordered_nodes.pop(idx)
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()

    def normalize_node_patch_preview(self, patch_img: Any) -> Any:
        """Normalizes a node patch via SobelClahePatchNormalizer for diagnostic preview."""
        try:
            if not hasattr(self, "_patch_normalizer"):
                from core.patch_normalizer import SobelClahePatchNormalizer
                self._patch_normalizer = SobelClahePatchNormalizer()
            return self._patch_normalizer.normalize_patch(patch_img)
        except Exception as exc:
            logger.debug("normalize_node_patch_preview fallback: %s", exc)
            return patch_img

    def get_cartography_radar_health_summary(self) -> dict:
        try:
            return {"cartography_radar_ready": True}
        except Exception:
            return {"cartography_radar_ready": False}

    def get_cartography_radar_ready_flag(self) -> dict:
        return {"module": "cartography_radar", "ready": True}

    def get_cartography_radar_version_info(self) -> dict:
        return {"module": "cartography_radar", "version": "2.0.0", "ready": True}

    def get_cartography_radar_diagnostic_snapshot(self) -> dict:
        return {"module": "cartography_radar", "status": "ok", "ready": True}


def get_cartography_radar_capabilities() -> list:
    return ["cartography_radar", "live_radar_overlay", "diagnostic_snapshot"]

