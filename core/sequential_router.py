# -*- coding: utf-8 -*-
"""
Sıralı Koridor, Sürekli İlerleme ve Geri Dönüşsüz (Anti-Ping-Pong) Rota Sıralayıcı.

Bu modül, harita üzerindeki kaynak düğümlerinin (NodeData) rastgele veya zikzaklı
("bi oraya bi buraya") gezilmesini tamamen ortadan kaldıran matematiksel ve algoritmik
çözümleri barındırır:

1. Yönlü Koridor & Pürüzsüz Çember Sıralama (`build_sequential_order`):
   - `walk_path`: Oyuncunun yürüyüş izi (`walk_trajectory`) çoklu doğrusuna dik izdüşüm
     (orthogonal polyline projection) ile gezilen sırayla 1..N dizilim.
   - `smooth_loop`: TSP + Açısal Süreklilik Cezası (`cos(theta)`) + 2-Opt + Or-Opt
     optimizasyonu ile kesişimsiz, zikzaksız, kapalı pürüzsüz halka.
   - `corridor_sweep`: Temel bileşen analizi (PCA) ekseninde haritayı paralel koridor
     şeritlerine bölerek serpantin (boustrophedon) + sınır yumuşatma ile tarama.
2. Anti-Ping-Pong Sıradaki Hedef Seçici (`get_next_in_sequence`):
   - Sadece ileri penceredeki (`+1 .. +forward_window`) hazır düğümleri seçer,
     haritanın diğer ucuna geri sıçramayı (ping-pong) kesin olarak engeller.
3. Dinamik Araya Ekleme (`insert_node_in_sequence`):
   - Yeni keşfedilen düğümü mevcut sırayı bozmadan en düşük ekleme maliyetli
     (`d(i, new) + d(new, i+1) - d(i, i+1)`) iki ardışık düğüm arasına yerleştirir.
4. Rota Kalite Metrikleri (`evaluate_sequence_quality`):
   - Toplam mesafe, ortalama adım, maksimum sıçrama, keskin dönüş sayısı,
     kenar kesişim sayısı ve pürüzsüzlük skoru (`0..100`) hesaplar.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from core.corridor_navigator import (
        ContestedNodeDetector,
        PredictivePreWalkScheduler,
        ZAxisCliffFilter,
    )
except ImportError:
    from .corridor_navigator import (  # type: ignore[no-redef]
        ContestedNodeDetector,
        PredictivePreWalkScheduler,
        ZAxisCliffFilter,
    )

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


Point2D = Tuple[float, float]


def _extract_xy(
    node: Any,
    fallback_width: float = 1920.0,
    fallback_height: float = 1080.0,
) -> Point2D:
    """
    Bir düğüm nesnesinden (`NodeData`, `dict` veya `(x, y)` tuple) 2B harita
    koordinatını `(x, y)` piksel cinsinden güvenli biçimde çıkarır.
    """
    try:
        if isinstance(node, (tuple, list, np.ndarray)) and len(node) >= 2:
            return float(node[0]), float(node[1])

        if isinstance(node, dict):
            mx = node.get("map_x", node.get("x", 0))
            my = node.get("map_y", node.get("y", 0))
            if (mx is None or float(mx) == 0.0) and (my is None or float(my) == 0.0):
                nx = node.get("norm_x")
                ny = node.get("norm_y")
                if nx is not None and ny is not None:
                    return float(nx) * fallback_width, float(ny) * fallback_height
            return float(mx or 0.0), float(my or 0.0)

        mx = getattr(node, "map_x", getattr(node, "x", 0))
        my = getattr(node, "map_y", getattr(node, "y", 0))
        if (mx is None or float(mx) == 0.0) and (my is None or float(my) == 0.0):
            nx = getattr(node, "norm_x", None)
            ny = getattr(node, "norm_y", None)
            if nx is not None and ny is not None:
                return float(nx) * fallback_width, float(ny) * fallback_height
        return float(mx or 0.0), float(my or 0.0)
    except Exception as exc:
        logger.debug("Koordinat çıkarılamadı (%r): %s", node, exc)
        return 0.0, 0.0


def _set_seq_index(node: Any, seq_idx: int) -> None:
    """Düğüm üzerine 1-tabanlı `seq_index` değerini yazar."""
    try:
        if isinstance(node, dict):
            node["seq_index"] = int(seq_idx)
        elif not isinstance(node, (tuple, list)):
            setattr(node, "seq_index", int(seq_idx))
    except Exception as exc:
        logger.debug("seq_index atanamadı (%r): %s", node, exc)


def _is_node_ready(
    node: Any,
    now: Optional[float] = None,
    rest_duration_sec: Optional[int] = None,
) -> bool:
    """Düğümün toplanmaya hazır (`is_ready`) olup olmadığını güvenli şekilde kontrol eder."""
    try:
        if hasattr(node, "is_ready") and callable(node.is_ready):
            return bool(node.is_ready(now=now, rest_duration_sec=rest_duration_sec))
        if isinstance(node, dict):
            if "ready" in node:
                return bool(node["ready"])
            if "rest_remaining_sec" in node and node["rest_remaining_sec"] is not None:
                return float(node["rest_remaining_sec"]) <= 0.0
            last_ts = float(node.get("last_visited_timestamp", node.get("last_visited", 0.0)) or 0.0)
            if last_ts <= 0.0:
                return True
            curr = time.time() if now is None else float(now)
            rest = int(
                rest_duration_sec
                if rest_duration_sec is not None
                else node.get("rest_duration_sec", 300)
            )
            return (curr - last_ts) >= float(rest)
        explicit_rem = getattr(node, "rest_remaining_sec", None)
        if explicit_rem is not None:
            return float(explicit_rem) <= 0.0
        last_ts = float(
            getattr(node, "last_visited_timestamp", getattr(node, "last_visited", 0.0)) or 0.0
        )
        if last_ts > 0.0:
            curr = time.time() if now is None else float(now)
            rest = int(
                rest_duration_sec
                if rest_duration_sec is not None
                else (getattr(node, "rest_duration_sec", 300) or 300)
            )
            return (curr - last_ts) >= float(rest)
        return True
    except Exception as exc:
        logger.debug("Düğüm hazırlık kontrolü hatası: %s", exc)
        return True


def _remaining_rest_seconds(
    node: Any,
    now: Optional[float] = None,
    rest_duration_sec: Optional[int] = None,
) -> float:
    """Düğümün hazır olmasına kalan süreyi saniye cinsinden döndürür."""
    try:
        if hasattr(node, "remaining_rest_sec") and callable(node.remaining_rest_sec):
            return float(node.remaining_rest_sec(now=now, rest_duration_sec=rest_duration_sec))
        if isinstance(node, dict):
            if "rest_remaining_sec" in node and node["rest_remaining_sec"] is not None:
                return max(0.0, float(node["rest_remaining_sec"]))
            last_ts = float(node.get("last_visited_timestamp", node.get("last_visited", 0.0)) or 0.0)
            if last_ts <= 0.0:
                return 0.0
            curr = time.time() if now is None else float(now)
            rest = float(
                rest_duration_sec
                if rest_duration_sec is not None
                else node.get("rest_duration_sec", 300)
            )
            return max(0.0, rest - (curr - last_ts))
        explicit_rem = getattr(node, "rest_remaining_sec", None)
        if explicit_rem is not None:
            return max(0.0, float(explicit_rem))
        last_ts = float(
            getattr(node, "last_visited_timestamp", getattr(node, "last_visited", 0.0)) or 0.0
        )
        if last_ts > 0.0:
            curr = time.time() if now is None else float(now)
            rest = float(
                rest_duration_sec
                if rest_duration_sec is not None
                else (getattr(node, "rest_duration_sec", 300) or 300)
            )
            return max(0.0, rest - (curr - last_ts))
        return 0.0
    except Exception as exc:
        logger.debug("Kalan dinlenme süresi hesaplama hatası: %s", exc)
        return 0.0


def _extract_max_harvests(node: Any) -> int:
    """
    Düğümün maksimum hasat kapasitesini (`max_harvests`) güvenli biçimde döndürür.
    Normal düğümler için varsayılan `3`, minimap `2x`/`3x` stack kümeleri için `6` veya `9` olur.
    """
    try:
        if isinstance(node, dict):
            raw_max = node.get("max_harvests")
            stack_cnt = node.get("stack_count", 1)
        else:
            raw_max = getattr(node, "max_harvests", None)
            stack_cnt = getattr(node, "stack_count", 1)

        stack_val = max(1, int(stack_cnt or 1))
        base_max = int(raw_max) if raw_max is not None and int(raw_max) > 0 else (3 * stack_val)
        return max(1, max(base_max, 3 * stack_val))
    except Exception as exc:
        logger.debug("max_harvests çıkarılamadı (%r): %s", node, exc)
        return 3


def _extract_current_harvests(node: Any) -> int:
    """Düğüm üzerindeki aktif kısmi hasat sayacını (`current_harvests`, `0..max_harvests-1`) döndürür."""
    try:
        if isinstance(node, dict):
            val = node.get("current_harvests", 0)
        else:
            val = getattr(node, "current_harvests", 0)
        return max(0, int(val or 0))
    except Exception as exc:
        logger.debug("current_harvests çıkarılamadı (%r): %s", node, exc)
        return 0


def _is_partially_harvested(node: Any) -> bool:
    """
    Düğümün şu anda kısmi hasat aşamasında (`1/3` veya `2/3`) olup olmadığını kontrol eder.
    """
    try:
        state_val = (
            node.get("state", "active")
            if isinstance(node, dict)
            else getattr(node, "state", "active")
        )
        if str(state_val or "active").strip().lower() == "resting":
            return False
        curr_h = _extract_current_harvests(node)
        max_h = _extract_max_harvests(node)
        return 0 < curr_h < max_h
    except Exception as exc:
        logger.debug("Kısmi hasat kontrolü hatası: %s", exc)
        return False


def _stack_cluster_tiebreaker_bonus(node: Any, avg_step_px: float = 120.0) -> float:
    """
    `2x` / `3x` minimap materyal yığınları (`max_harvests >= 6`) için ileri koridor
    boyunca hafif pozitif eşitlik bozucu (tie-breaker) bonus üretir.
    Monotonik sıra düzenini (`0 -> 1 -> 2 -> ... -> N-1`) asla bozmaz; yalnızca
    eşit mesafeli / eşdeğer adaylar arasında zengin yığınları hafifçe destekler.
    """
    max_h = _extract_max_harvests(node)
    if max_h < 6:
        return 0.0
    # 2x stack (max_harvests=6) -> +1 kademe, 3x stack (max_harvests>=9) -> +2 kademe
    stack_tiers = min(3.0, float(max_h - 3) / 3.0)
    scale_px = max(4.0, min(14.0, float(avg_step_px) * 0.06))
    return round(stack_tiers * scale_px, 4)


def _euclidean(p1: Point2D, p2: Point2D) -> float:
    """İki 2B nokta arasındaki Öklid mesafesini döndürür."""
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def _turn_angle_rad(p_prev: Point2D, p_curr: Point2D, p_next: Point2D) -> float:
    """
    `p_prev -> p_curr` yön vektörü ile `p_curr -> p_next` yön vektörü arasındaki
    sapma açısını `[0, pi]` radyan aralığında hesaplar.
    `0` = dümdüz ilerleme, `pi` = 180 derece geri dönüş (U-turn / ping-pong).
    """
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


def _segments_intersect(a1: Point2D, a2: Point2D, b1: Point2D, b2: Point2D) -> bool:
    """İki doğru parçasının (`a1-a2` ve `b1-b2`) uç noktalar hariç kesişip kesişmediğini döndürür."""
    def _ccw(p1: Point2D, p2: Point2D, p3: Point2D) -> float:
        return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])

    # Ortak uç nokta varsa kesişim sayma
    eps = 1e-6
    if (
        _euclidean(a1, b1) < eps
        or _euclidean(a1, b2) < eps
        or _euclidean(a2, b1) < eps
        or _euclidean(a2, b2) < eps
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


class SequentialRoutePlanner:
    """
    Sıralı Koridor, Sürekli İlerleme ve Geri Dönüşsüz (Anti-Ping-Pong) Rota Sıralayıcı.

    Kullanıcının "sıralı olsun yani bi oraya bi buraya gitmesin sırayla gezsin"
    talebini matematiksel olarak çözer:
    - `build_sequential_order`: Düğümleri pürüzsüz çember (`smooth_loop`), yürüyüş izi
      (`walk_path`) veya bölgesel şerit (`corridor_sweep`) modunda 1..N sıraya sokar.
    - `get_next_in_sequence`: Sadece ileri penceredeki hazır düğümleri seçerek
      harita genelinde geri dönüşleri ve rastgele sıçramaları engeller.
    - `insert_node_in_sequence`: Yeni keşfedilen düğümü mevcut sırayı bozmadan
      en uygun iki ardışık düğüm arasına yerleştirir.
    - `evaluate_sequence_quality`: Rotanın mesafe, sıçrama, keskin dönüş ve
      pürüzsüzlük (`0..100`) metriklerini üretir.
    """

    VALID_MODES = ("smooth_loop", "walk_path", "corridor_sweep")

    def __init__(
        self,
        angular_penalty_weight: float = 0.35,
        corridor_count: Optional[int] = None,
        max_opt_iterations: int = 80,
        cliff_filter: Optional[ZAxisCliffFilter] = None,
        prewalk_scheduler: Optional[PredictivePreWalkScheduler] = None,
        contested_detector: Optional[ContestedNodeDetector] = None,
    ) -> None:
        self.angular_penalty_weight: float = float(max(0.0, angular_penalty_weight))
        self.corridor_count: Optional[int] = (
            int(corridor_count) if corridor_count is not None and corridor_count > 0 else None
        )
        self.max_opt_iterations: int = int(max(10, max_opt_iterations))
        self._current_seq_pos: int = -1
        self.cliff_filter: ZAxisCliffFilter = cliff_filter or ZAxisCliffFilter()
        self.prewalk_scheduler: PredictivePreWalkScheduler = (
            prewalk_scheduler or PredictivePreWalkScheduler()
        )
        self.contested_detector: ContestedNodeDetector = (
            contested_detector or ContestedNodeDetector()
        )

    def record_approach_attempt(
        self,
        node: Any,
        minimap_dist_px: float,
        interaction_succeeded: bool,
        now: Optional[float] = None,
    ) -> bool:
        """
        Madde 27 (`ZAxisCliffFilter`):
        2B minimap üzerinde yakın (`< 25 px`) görünen düğüme ardışık `2` başarısız
        yaklaşma denemesinde `unreachable_cliff = True` ve `altitude_band = "cliff_mismatch"`
        etiketini atar.
        """
        return self.cliff_filter.record_approach_attempt(
            node=node,
            minimap_dist_px=minimap_dist_px,
            interaction_succeeded=interaction_succeeded,
            now=now,
        )

    def _is_candidate_selectable(
        self,
        node: Any,
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        player_z: Optional[float] = None,
        now: Optional[float] = None,
    ) -> bool:
        """
        Düğümün uçurum (`ZAxisCliffFilter`, Madde 27) veya rakip oyuncu
        (`ContestedNodeDetector`, Madde 30) nedeniyle dışlanıp dışlanmadığını kontrol eder.
        """
        if self.cliff_filter.is_node_excluded_by_cliff(
            node,
            is_flying=is_flying,
            player_altitude_band=player_altitude_band,
            player_z=player_z,
        ):
            return False
        if self.contested_detector.is_node_currently_contested_flag(node, now=now):
            return False
        return True

    def _is_ready_or_predictive_prewalk(
        self,
        node: Any,
        player_pos: Optional[Point2D] = None,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
        enable_predictive_prewalk: bool = True,
    ) -> bool:
        """
        Düğümün şu anda toplanmaya hazır (`_is_node_ready`) VEYA
        Madde 28 (`PredictivePreWalkScheduler`) kapsamında erken yürüyüşe (`<= 8s` kalan
        ve `rest_remaining_sec <= estimated_walk_sec + 2.0`) uygun olup olmadığını döndürür.
        """
        if _is_node_ready(node, now=now, rest_duration_sec=rest_duration_sec):
            return True
        if enable_predictive_prewalk and self.prewalk_scheduler.should_pre_walk_to_node(
            node=node,
            player_pos=player_pos,
            now=now,
            rest_duration_sec=rest_duration_sec,
        ):
            return True
        return False

    # ------------------------------------------------------------------
    # 1. Yönlü Koridor & Pürüzsüz Çember Sıralama
    # ------------------------------------------------------------------
    def build_sequential_order(
        self,
        nodes: Sequence[Any],
        walk_trajectory: Optional[Sequence[Point2D]] = None,
        mode: str = "smooth_loop",
    ) -> List[Any]:
        """
        Verilen düğüm listesini seçilen moda göre pürüzsüz ve sıralı biçimde
        (`1, 2, 3, ..., N`) dizer ve her düğüme `seq_index` atar.

        Desteklenen modlar:
        - `"walk_path"`: `walk_trajectory` çoklu doğrusuna dik izdüşüm sırası.
        - `"smooth_loop"`: Açısal süreklilik cezalı TSP + 2-Opt + Or-Opt + Açısal düzeltme.
        - `"corridor_sweep"`: PCA ana eksenine göre pürüzsüz koridor şerit taraması.
        """
        try:
            if not nodes:
                return []

            node_list = list(nodes)
            n = len(node_list)
            if n <= 2:
                for idx, item in enumerate(node_list, start=1):
                    _set_seq_index(item, idx)
                return node_list

            clean_mode = str(mode or "smooth_loop").strip().lower()
            if clean_mode not in self.VALID_MODES:
                logger.warning(
                    "Bilinmeyen sıralama modu '%s', 'smooth_loop' kullanılacak.",
                    clean_mode,
                )
                clean_mode = "smooth_loop"

            coords: List[Point2D] = [_extract_xy(item) for item in node_list]

            if clean_mode == "walk_path":
                order_indices = self._order_by_walk_path(coords, walk_trajectory)
            elif clean_mode == "corridor_sweep":
                order_indices = self._order_by_corridor_sweep(coords)
            else:
                order_indices = self._order_by_smooth_loop(coords)

            ordered_nodes = [node_list[i] for i in order_indices]
            for seq_idx, item in enumerate(ordered_nodes, start=1):
                _set_seq_index(item, seq_idx)

            logger.info(
                "Sıralı rota oluşturuldu: mod=%s, düğüm_sayısı=%d",
                clean_mode,
                len(ordered_nodes),
            )
            return ordered_nodes
        except Exception as exc:
            logger.error("build_sequential_order sırasında hata: %s", exc)
            fallback = list(nodes) if nodes else []
            for seq_idx, item in enumerate(fallback, start=1):
                _set_seq_index(item, seq_idx)
            return fallback

    def apply_sequential_order_to_route(
        self,
        route_manager: Any,
        region: Optional[str] = None,
        route: str = "All",
        mode: str = "smooth_loop",
        walk_trajectory: Optional[Sequence[Point2D]] = None,
        reassign_ids: bool = False,
    ) -> List[Any]:
        """
        `RouteManager` içindeki belirtilen bölge ve rotanın düğümlerini sıralar,
        her düğüme `seq_index` (`1..N`) atar ve `RouteManager` veri yapısını
        bu kesin sıraya göre güncelleyip diske kaydeder.
        """
        try:
            if route_manager is None:
                logger.warning("apply_sequential_order_to_route: route_manager None.")
                return []

            target_region = region or getattr(route_manager, "active_region", "Verteron")
            target_route = route or getattr(route_manager, "active_route", "All")

            route_nodes = route_manager.get_route_nodes(
                region_name=target_region,
                route_name=target_route,
            )
            if not route_nodes:
                logger.info(
                    "Sıralanacak düğüm bulunamadı (bölge=%s, rota=%s).",
                    target_region,
                    target_route,
                )
                return []

            ordered_nodes = self.build_sequential_order(
                nodes=route_nodes,
                walk_trajectory=walk_trajectory,
                mode=mode,
            )

            # RouteManager içindeki bölge düğüm listesini sıralı hale getir
            regions_store = getattr(route_manager, "_regions", None)
            if isinstance(regions_store, dict) and target_region in regions_store:
                all_region_nodes: List[Any] = regions_store[target_region].get("nodes", [])
                if not target_route or target_route == "All" or len(ordered_nodes) == len(all_region_nodes):
                    regions_store[target_region]["nodes"] = list(ordered_nodes)
                else:
                    ordered_set = {id(n) for n in ordered_nodes}
                    ordered_iter = iter(ordered_nodes)
                    merged_nodes: List[Any] = []
                    for existing in all_region_nodes:
                        if id(existing) in ordered_set:
                            merged_nodes.append(next(ordered_iter))
                        else:
                            merged_nodes.append(existing)
                    regions_store[target_region]["nodes"] = merged_nodes

                if reassign_ids:
                    for new_id, node_obj in enumerate(regions_store[target_region]["nodes"], start=1):
                        if hasattr(node_obj, "id"):
                            node_obj.id = new_id

            # Son seçilen düğüm işaretçisini sıfırla ki yeni sıranın başından başlasın
            last_sel = getattr(route_manager, "_last_selected_node_id", None)
            if isinstance(last_sel, dict):
                last_sel.pop((target_region, target_route), None)

            if hasattr(route_manager, "save_region") and callable(route_manager.save_region):
                route_manager.save_region(target_region)
            elif hasattr(route_manager, "_sync_legacy_data") and callable(route_manager._sync_legacy_data):
                route_manager._sync_legacy_data()

            self._current_seq_pos = -1
            return ordered_nodes
        except Exception as exc:
            logger.error("apply_sequential_order_to_route hatası: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Mod A: Yürüyüş İzi Sırası ("walk_path")
    # ------------------------------------------------------------------
    def _order_by_walk_path(
        self,
        coords: List[Point2D],
        walk_trajectory: Optional[Sequence[Point2D]] = None,
    ) -> List[int]:
        """
        Oyuncunun haritayı gezerken izlediği `walk_trajectory` çoklu doğrusuna
        (polyline) her düğümü dik izdüşümle (`orthogonal projection`) düşürür ve
        eğri boyunca kümülatif yay uzunluğu (`arc_length_s`) sırasına göre dizer.
        Eğer `walk_trajectory` verilmemişse düğümlerin doğal omurga eğrisini çıkarır.
        """
        n = len(coords)
        if n <= 1:
            return list(range(n))

        # Geçerli yürüyüş izi yoksa veya tek noktaysa doğal açık koridor eğrisi oluştur
        traj: List[Point2D] = []
        if walk_trajectory is not None:
            for pt in walk_trajectory:
                xy = _extract_xy(pt)
                if not traj or _euclidean(traj[-1], xy) > 1e-5:
                    traj.append(xy)

        if len(traj) < 2:
            # PCA ana ekseni boyunca omurga noktaları üret ve yerel akışla düzelt
            return self._order_by_corridor_sweep(coords)

        # Çoklu doğru (polyline) segmentlerinin kümülatif uzunluklarını hesapla
        seg_lengths: List[float] = []
        cum_lengths: List[float] = [0.0]
        total_len = 0.0
        for i in range(len(traj) - 1):
            seg_len = _euclidean(traj[i], traj[i + 1])
            seg_lengths.append(seg_len)
            total_len += seg_len
            cum_lengths.append(total_len)

        if total_len < 1e-6:
            return self._order_by_smooth_loop(coords)

        projections: List[Tuple[float, float, int]] = []
        for idx, pt in enumerate(coords):
            best_dist_sq = float("inf")
            best_s = 0.0
            px, py = pt
            for seg_idx in range(len(traj) - 1):
                ax, ay = traj[seg_idx]
                bx, by = traj[seg_idx + 1]
                dx = bx - ax
                dy = by - ay
                seg_len_sq = dx * dx + dy * dy
                if seg_len_sq < 1e-12:
                    t = 0.0
                    proj_x, proj_y = ax, ay
                else:
                    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
                    t = max(0.0, min(1.0, t))
                    proj_x = ax + t * dx
                    proj_y = ay + t * dy

                dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
                if dist_sq < best_dist_sq:
                    best_dist_sq = dist_sq
                    best_s = cum_lengths[seg_idx] + t * seg_lengths[seg_idx]

            projections.append((best_s, best_dist_sq, idx))

        projections.sort(key=lambda item: (item[0], item[1], item[2]))
        initial_order = [idx for _, _, idx in projections]

        # Aynı yay uzunluğuna düşen yakın düğümler arasındaki mikro-zikzakları
        # yerel pencere (pencere boyutu 6) Or-Opt ile sıra bütünlüğünü bozmadan yumuşat
        initial_order = self._local_window_smooth(initial_order, coords, window_size=6)
        initial_order = self._eliminate_crossings(initial_order, coords, closed_loop=False)
        return initial_order

    def _local_window_smooth(
        self,
        order: List[int],
        coords: List[Point2D],
        window_size: int = 6,
    ) -> List[int]:
        """
        Yürüyüş izi veya koridor sıralamasında genel ilerleme yönünü koruyarak
        yalnızca dar bir komşuluk penceresindeki mikro geri-dönüşleri (zikzakları)
        hem mesafe hem de açısal süreklilik (`cos(theta)`) kriteriyle düzeltir.
        """
        n = len(order)
        if n < 4:
            return order

        result = list(order)
        improved = True
        passes = 0
        while improved and passes < 10:
            improved = False
            passes += 1
            for i in range(1, n - 2):
                upper = min(n - 1, i + window_size)
                for j in range(i + 1, upper):
                    if (j + 1) >= n:
                        continue
                    # [i-1 .. j+1] alt dizisinin mevcut ve ters çevrilmiş maliyetini karşılaştır
                    sub_old = result[max(0, i - 2) : min(n, j + 3)]
                    sub_new = (
                        result[max(0, i - 2) : i]
                        + list(reversed(result[i : j + 1]))
                        + result[j + 1 : min(n, j + 3)]
                    )
                    cost_old = self._subpath_cost(sub_old, coords)
                    cost_new = self._subpath_cost(sub_new, coords)
                    if cost_new + 1e-4 < cost_old:
                        result[i : j + 1] = reversed(result[i : j + 1])
                        improved = True
        return result

    def _subpath_cost(self, sub_indices: List[int], coords: List[Point2D]) -> float:
        """Açık bir alt-yolun mesafe + açısal kırılma maliyetini hesaplar."""
        m = len(sub_indices)
        if m <= 1:
            return 0.0
        dist_sum = 0.0
        for k in range(m - 1):
            dist_sum += _euclidean(coords[sub_indices[k]], coords[sub_indices[k + 1]])
        avg_d = dist_sum / float(m - 1)
        ang_sum = 0.0
        for k in range(m - 2):
            ang_sum += self._turn_cost(
                coords[sub_indices[k]],
                coords[sub_indices[k + 1]],
                coords[sub_indices[k + 2]],
            )
        return dist_sum + (0.65 + self.angular_penalty_weight) * avg_d * ang_sum

    # ------------------------------------------------------------------
    # Mod B: Geri Dönüşsüz Pürüzsüz Çember ("smooth_loop")
    # ------------------------------------------------------------------
    def _order_by_smooth_loop(self, coords: List[Point2D]) -> List[int]:
        """
        TSP + Açısal Süreklilik (`cos(theta)`) + 2-Opt + Or-Opt algoritmalarını
        birleştirerek zikzak yapmayan, kenarları kesişmeyen, sürekli ileri akan
        pürüzsüz bir kapalı çember (`loop`) oluşturur.
        """
        n = len(coords)
        if n <= 2:
            return list(range(n))

        dist_matrix = self._build_distance_matrix(coords)

        # 1. Aday Başlangıç Rotaları Üret:
        #    a) Merkez etrafında açısal süpürme (Angular Sweep) - doğal çember omurgası
        #    b) Çoklu başlangıçlı Açısal Cezalı En Yakın Komşu (Angular-Continuity NN)
        #    c) Koridor serpantin başlangıç rotası
        candidates: List[List[int]] = []
        candidates.append(self._angular_sweep_init(coords))
        candidates.append(self._corridor_initial_order(coords))

        # Farklı bölgelerden seçilen temsilci düğümlerle açısal-süreklilik NN dene
        num_starts = min(n, 6)
        step = max(1, n // num_starts)
        for start_idx in range(0, n, step):
            candidates.append(self._angular_nearest_neighbor_init(coords, dist_matrix, start_idx))

        best_tour: List[int] = candidates[0]
        best_score = float("inf")

        for cand in candidates:
            # Hızlı 2-Opt + Or-Opt ile adayları değerlendir
            refined = self._two_opt_loop(cand, dist_matrix, coords, use_angular=False)
            refined = self._or_opt_loop(refined, dist_matrix)
            score = self._tour_objective(refined, dist_matrix, coords)
            if score < best_score:
                best_score = score
                best_tour = refined

        # En iyi tur üzerinde Açısal Süreklilik (Angular Continuity) + kesişim giderme uygula
        best_tour = self._two_opt_loop(best_tour, dist_matrix, coords, use_angular=True)
        best_tour = self._or_opt_angular_loop(best_tour, dist_matrix, coords)
        best_tour = self._eliminate_crossings(best_tour, coords, closed_loop=True)

        # Saat yönü (clockwise) ve başlangıç noktası normalizasyonu
        best_tour = self._normalize_loop_orientation(best_tour, coords)
        return best_tour

    def _build_distance_matrix(self, coords: List[Point2D]) -> np.ndarray:
        """Noktalar arası simetrik Öklid mesafe matrisini vektörel olarak hesaplar."""
        pts = np.asarray(coords, dtype=np.float64)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        return np.hypot(diff[:, :, 0], diff[:, :, 1])

    def _angular_sweep_init(self, coords: List[Point2D]) -> List[int]:
        """Merkez (centroid) etrafında saat yönünde açısal sıralama ile başlangıç çemberi üretir."""
        pts = np.asarray(coords, dtype=np.float64)
        cx, cy = float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        angles = []
        for idx, (x, y) in enumerate(coords):
            ang = math.atan2(y - cy, x - cx)
            rad = math.hypot(x - cx, y - cy)
            angles.append((ang, rad, idx))
        angles.sort(key=lambda item: (item[0], item[1]))
        return [idx for _, _, idx in angles]

    def _angular_nearest_neighbor_init(
        self,
        coords: List[Point2D],
        dist_matrix: np.ndarray,
        start_idx: int = 0,
    ) -> List[int]:
        """
        Klasik En Yakın Komşu yerine bir önceki hareket vektörüyle yapılan dönüş açısına
        `1 + w * (1 - cos(theta)) / 2` çarpanı uygulayan yön-koruyucu açgözlü tur kurucu.
        """
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
                    theta = _turn_angle_rad(coords[prev], coords[curr], coords[cand])
                    # cos(theta) = 1 ise ceza 0; cos(theta) = -1 (180 derece dönüş) ise maksimum ceza
                    turn_penalty = 0.5 * (1.0 - math.cos(theta))
                    cost = d * (1.0 + self.angular_penalty_weight * turn_penalty)
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

    def _two_opt_loop(
        self,
        tour: List[int],
        dist_matrix: np.ndarray,
        coords: List[Point2D],
        use_angular: bool = False,
    ) -> List[int]:
        """
        Kapalı çember üzerinde 2-Opt kenar değişimi yapar.
        Kesişen tüm kenarları çözer ve `use_angular=True` olduğunda keskin dönüşleri törpüler.
        """
        n = len(tour)
        if n < 4:
            return list(tour)

        route = list(tour)
        improved = True
        iteration = 0

        while improved and iteration < self.max_opt_iterations:
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
                            self._turn_cost(coords[a_prev], coords[a], coords[b])
                            + self._turn_cost(coords[a], coords[b], coords[b_next])
                            + self._turn_cost(coords[c_prev], coords[c], coords[d])
                            + self._turn_cost(coords[c], coords[d], coords[d_next])
                        )
                        new_ang = (
                            self._turn_cost(coords[a_prev], coords[a], coords[c])
                            + self._turn_cost(coords[a], coords[c], coords[c_prev])
                            + self._turn_cost(coords[b_next], coords[b], coords[d])
                            + self._turn_cost(coords[b], coords[d], coords[d_next])
                        )
                        mean_edge = 0.25 * (old_dist + new_dist)
                        old_total = old_dist + self.angular_penalty_weight * mean_edge * 0.25 * old_ang
                        new_total = new_dist + self.angular_penalty_weight * mean_edge * 0.25 * new_ang
                    else:
                        old_total = old_dist
                        new_total = new_dist

                    if new_total + 1e-6 < old_total:
                        route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                        improved = True

        return route

    def _or_opt_loop(
        self,
        tour: List[int],
        dist_matrix: np.ndarray,
    ) -> List[int]:
        """
        Or-Opt optimizasyonu: 1, 2 ve 3 uzunluğundaki ardışık düğüm zincirlerini
        çember üzerinde daha düşük maliyetli kenarların arasına taşır.
        Uzun sıçramaları ve yerel çıkıntıları (detour spikes) ortadan kaldırır.
        """
        n = len(tour)
        if n < 5:
            return list(tour)

        route = list(tour)
        improved = True
        iteration = 0

        while improved and iteration < self.max_opt_iterations:
            improved = False
            iteration += 1
            for chain_len in (1, 2, 3):
                if n <= chain_len + 2:
                    continue
                i = 0
                while i < n:
                    # Zincir: route[i ... i + chain_len - 1] (mod n yerine doğrusal dilim için rotayı kaydır)
                    if i + chain_len >= n:
                        break
                    prev_i = route[i - 1]
                    first_c = route[i]
                    last_c = route[i + chain_len - 1]
                    next_i = route[(i + chain_len) % n]

                    # Zinciri mevcut yerinden çıkarmanın kazancı
                    remove_gain = (
                        float(dist_matrix[prev_i, first_c])
                        + float(dist_matrix[last_c, next_i])
                        - float(dist_matrix[prev_i, next_i])
                    )

                    best_delta = -1e-6
                    best_pos = -1
                    best_reverse = False
                    chain_rev_delta = 0.0
                    if chain_len == 2:
                        # Ters çevirme mesafeyi değiştirmez (simetrik), sadece uçları değiştirir
                        chain_rev_delta = 0.0

                    for j in range(n):
                        # Hedef kenar: (route[j], route[(j+1)%n])
                        if i - 1 <= j < i + chain_len:
                            continue
                        u = route[j]
                        v = route[(j + 1) % n]
                        base_edge = float(dist_matrix[u, v])

                        # Düz ekleme maliyeti
                        insert_cost_fwd = (
                            float(dist_matrix[u, first_c])
                            + float(dist_matrix[last_c, v])
                            - base_edge
                        )
                        delta_fwd = insert_cost_fwd - remove_gain
                        if delta_fwd < best_delta:
                            best_delta = delta_fwd
                            best_pos = j
                            best_reverse = False

                        # Ters çevrilmiş zincir ekleme maliyeti
                        if chain_len > 1:
                            insert_cost_rev = (
                                float(dist_matrix[u, last_c])
                                + float(dist_matrix[first_c, v])
                                - base_edge
                                + chain_rev_delta
                            )
                            delta_rev = insert_cost_rev - remove_gain
                            if delta_rev < best_delta:
                                best_delta = delta_rev
                                best_pos = j
                                best_reverse = True

                    if best_pos != -1:
                        segment = route[i : i + chain_len]
                        if best_reverse:
                            segment.reverse()
                        del route[i : i + chain_len]
                        insert_idx = best_pos + 1 if best_pos < i else best_pos - chain_len + 1
                        route[insert_idx:insert_idx] = segment
                        improved = True
                        break
                    i += 1
                if improved:
                    break

        return route

    def _or_opt_angular_loop(
        self,
        tour: List[int],
        dist_matrix: np.ndarray,
        coords: List[Point2D],
    ) -> List[int]:
        """
        Keskin dönüş (`> 80°`) yapan düğümleri ve yerel dörtlü grupları tespit edip,
        kenar kesişimi oluşturmadan açısal kırılmayı ortadan kaldıran sıraya taşır.
        """
        n = len(tour)
        if n < 5:
            return list(tour)

        route = list(tour)
        for _ in range(min(30, self.max_opt_iterations)):
            changed = False
            for i in range(n):
                prev_node = route[(i - 1) % n]
                curr_node = route[i]
                next_node = route[(i + 1) % n]

                angle = _turn_angle_rad(coords[prev_node], coords[curr_node], coords[next_node])
                if angle <= (math.pi * 0.45):
                    continue

                # Bu düğümde keskin dönüş var; komşu kenarlara taşımayı dene
                current_obj = self._tour_objective(route, dist_matrix, coords)
                best_obj = current_obj
                best_candidate: Optional[List[int]] = None

                without = route[:i] + route[i + 1 :]
                m = len(without)
                for offset in range(-10, 11):
                    if offset in (-1, 0):
                        continue
                    ins_pos = (i + offset) % m
                    cand = without[: ins_pos + 1] + [curr_node] + without[ins_pos + 1 :]
                    cand_obj = self._tour_objective(cand, dist_matrix, coords)
                    if cand_obj + 1e-4 < best_obj:
                        best_obj = cand_obj
                        best_candidate = cand

                if best_candidate is not None:
                    route = best_candidate
                    changed = True
                    break
            if not changed:
                break

        return route

    def _turn_cost(self, p_prev: Point2D, p_curr: Point2D, p_next: Point2D) -> float:
        """
        Açısal kırılma cezası: `0.5 * (1 - cos(theta))` + keskin dönüş (`>90°`) ek cezası.
        """
        theta = _turn_angle_rad(p_prev, p_curr, p_next)
        base = 0.5 * (1.0 - math.cos(theta))
        if theta > (math.pi * 0.45):
            excess = (theta - math.pi * 0.45) / (math.pi * 0.55)
            base += 3.2 * (excess ** 2)
        if theta > (math.pi * 0.5):
            # 90 dereceden keskin dönüşlere (geri adım / zikzak) güçlü ceza uygula
            base += 2.5
        return base

    def _tour_objective(
        self,
        tour: List[int],
        dist_matrix: np.ndarray,
        coords: List[Point2D],
    ) -> float:
        """
        Bir tur için birleşik hedef fonksiyonu:
        `Toplam Mesafe + Açısal Kırılma Cezası + Maksimum Sıçrama Cezası`.
        """
        n = len(tour)
        if n <= 1:
            return 0.0

        total_dist = 0.0
        max_step = 0.0
        total_turn_cost = 0.0

        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            w = tour[(i + 2) % n]
            step = float(dist_matrix[u, v])
            total_dist += step
            if step > max_step:
                max_step = step
            total_turn_cost += self._turn_cost(coords[u], coords[v], coords[w])

        avg_step = total_dist / float(n)
        angular_term = (0.45 + self.angular_penalty_weight) * avg_step * total_turn_cost
        jump_term = 0.45 * max_step
        return total_dist + angular_term + jump_term

    def _normalize_loop_orientation(
        self,
        tour: List[int],
        coords: List[Point2D],
    ) -> List[int]:
        """
        Kapalı çemberin yönünü saat yönüne (clockwise) sabitler ve ilk düğümü
        sol-üst (veya orijinal ilk düğüme en yakın) konumdan başlatır.
        """
        n = len(tour)
        if n <= 2:
            return list(tour)

        # Shoelace formülü ile işaretli alan (ekran koordinatlarında y aşağı doğru büyür)
        signed_area = 0.0
        for i in range(n):
            x1, y1 = coords[tour[i]]
            x2, y2 = coords[tour[(i + 1) % n]]
            signed_area += (x1 * y2 - x2 * y1)

        route = list(tour)
        if signed_area < 0.0:
            route.reverse()

        # Sol-üst köşeye (x + y minimum) en yakın düğümü başlangıç (seq_index=1) yap
        start_pos = min(
            range(n),
            key=lambda k: (coords[route[k]][0] + coords[route[k]][1], route[k]),
        )
        route = route[start_pos:] + route[:start_pos]
        return route

    # ------------------------------------------------------------------
    # Mod C: Bölgesel Şerit / Koridor Sıralaması ("corridor_sweep")
    # ------------------------------------------------------------------
    def _corridor_initial_order(self, coords: List[Point2D]) -> List[int]:
        """
        Haritayı temel bileşen analizi (PCA) eksenlerine göre paralel koridor şeritlerine
        ayırır, şerit içindeki büyük boşlukları (engel/göl/orta boşluk) alt-koridorlara
        böler ve alt-koridorları en yakın uç noktalarından birbirine zincirler.
        """
        n = len(coords)
        if n <= 2:
            return list(range(n))

        pts = np.asarray(coords, dtype=np.float64)
        mean = np.mean(pts, axis=0)
        centered = pts - mean

        # Kovaryans matrisi üzerinden ana ve yan eksenleri bul
        cov = np.cov(centered.T)
        if cov.ndim == 2 and cov.shape == (2, 2):
            eigvals, eigvecs = np.linalg.eigh(cov)
            major_axis = eigvecs[:, 1]
            minor_axis = eigvecs[:, 0]
        else:
            major_axis = np.array([1.0, 0.0])
            minor_axis = np.array([0.0, 1.0])

        if major_axis[0] + major_axis[1] < 0:
            major_axis = -major_axis
        if minor_axis[0] * (-major_axis[1]) + minor_axis[1] * major_axis[0] < 0:
            minor_axis = -minor_axis

        u_proj = centered @ major_axis
        v_proj = centered @ minor_axis

        v_min, v_max = float(np.min(v_proj)), float(np.max(v_proj))
        u_min, u_max = float(np.min(u_proj)), float(np.max(u_proj))
        v_span = max(1e-6, v_max - v_min)
        u_span = max(1e-6, u_max - u_min)

        if self.corridor_count is not None:
            num_bands = max(1, min(n // 3, self.corridor_count))
        else:
            num_bands = max(2, min(7, int(round(math.sqrt(n / 4.0)))))

        band_bins: List[List[Tuple[float, int]]] = [[] for _ in range(num_bands)]
        for idx in range(n):
            norm_v = (float(v_proj[idx]) - v_min) / v_span
            b_idx = min(num_bands - 1, max(0, int(norm_v * num_bands)))
            band_bins[b_idx].append((float(u_proj[idx]), idx))

        # Her şeridi u eksenine göre sırala; şerit ortasında büyük boşluk varsa alt-koridora böl
        gap_threshold = max(120.0, u_span * 0.22)
        sub_corridors: List[List[int]] = []
        for b_idx in range(num_bands):
            bucket = band_bins[b_idx]
            if not bucket:
                continue
            bucket.sort(key=lambda item: item[0])
            current_seg: List[int] = [bucket[0][1]]
            for k in range(1, len(bucket)):
                prev_idx = bucket[k - 1][1]
                curr_idx = bucket[k][1]
                if _euclidean(coords[prev_idx], coords[curr_idx]) > gap_threshold:
                    sub_corridors.append(current_seg)
                    current_seg = [curr_idx]
                else:
                    current_seg.append(curr_idx)
            if current_seg:
                sub_corridors.append(current_seg)

        if not sub_corridors:
            return list(range(n))

        # Alt-koridorları en yakın uç noktalarından (serpantin + en kısa köprü) birbirine bağla
        used = [False] * len(sub_corridors)
        order: List[int] = list(sub_corridors[0])
        used[0] = True

        for _ in range(len(sub_corridors) - 1):
            last_pt = coords[order[-1]]
            best_seg_idx = -1
            best_rev = False
            best_dist = float("inf")

            for s_idx, seg in enumerate(sub_corridors):
                if used[s_idx]:
                    continue
                d_start = _euclidean(last_pt, coords[seg[0]])
                d_end = _euclidean(last_pt, coords[seg[-1]])
                if d_start < best_dist:
                    best_dist = d_start
                    best_seg_idx = s_idx
                    best_rev = False
                if d_end < best_dist:
                    best_dist = d_end
                    best_seg_idx = s_idx
                    best_rev = True

            if best_seg_idx != -1:
                used[best_seg_idx] = True
                chosen_seg = list(sub_corridors[best_seg_idx])
                if best_rev:
                    chosen_seg.reverse()
                order.extend(chosen_seg)

        return order

    def _order_by_corridor_sweep(self, coords: List[Point2D]) -> List[int]:
        """
        Bölgesel Şerit / Koridor Sıralaması (`corridor_sweep`):
        1. Haritayı PCA koridor şeritlerine bölerek alt-koridorları birbirine bağlar.
        2. Şerit içi ve şerit geçişlerindeki yerel çaprazlamaları ve keskin dönüşleri
           koridor yapısını bozmadan yerel 2-Opt ve Or-Opt ile pürüzsüzleştirir.
        """
        n = len(coords)
        if n <= 2:
            return list(range(n))

        order = self._corridor_initial_order(coords)
        dist_matrix = self._build_distance_matrix(coords)

        # Koridor sırasını koruyarak pencere içi (local window) 2-Opt ve Or-Opt uygula
        order = self._local_window_smooth(order, coords, window_size=min(18, max(8, n // 4)))
        order = self._open_path_or_opt(order, dist_matrix, coords, max_shift=min(16, max(6, n // 5)))
        order = self._local_window_smooth(order, coords, window_size=min(14, max(6, n // 5)))
        order = self._eliminate_crossings(order, coords, closed_loop=False)
        return order

    def _eliminate_crossings(
        self,
        order: List[int],
        coords: List[Point2D],
        closed_loop: bool = True,
    ) -> List[int]:
        """
        Geometrik olarak kesişen tüm kenar çiftlerini (`_segments_intersect`)
        2-Opt ters çevirme işlemiyle tamamen ortadan kaldırır (`crossings_count = 0`).
        """
        n = len(order)
        if n < 4:
            return list(order)

        route = list(order)
        num_edges = n if closed_loop else (n - 1)
        for _ in range(min(40, n)):
            found_crossing = False
            for i in range(num_edges - 2):
                a1 = coords[route[i]]
                a2 = coords[route[(i + 1) % n]]
                for j in range(i + 2, num_edges):
                    if closed_loop and i == 0 and j == num_edges - 1:
                        continue
                    b1 = coords[route[j]]
                    b2 = coords[route[(j + 1) % n]]
                    if _segments_intersect(a1, a2, b1, b2):
                        route[i + 1 : j + 1] = reversed(route[i + 1 : j + 1])
                        found_crossing = True
                        break
                if found_crossing:
                    break
            if not found_crossing:
                break
        return route

    def _open_path_or_opt(
        self,
        order: List[int],
        dist_matrix: np.ndarray,
        coords: List[Point2D],
        max_shift: int = 10,
    ) -> List[int]:
        """
        Koridor şerit yapısını bozmadan, sadece `±max_shift` komşuluğunda
        düğüm yeri değiştirerek şerit içi zikzakları ve sıçramaları temizler.
        """
        n = len(order)
        if n < 5:
            return list(order)

        route = list(order)
        for _ in range(15):
            improved = False
            for i in range(1, n - 1):
                u = route[i - 1]
                curr = route[i]
                v = route[i + 1]

                remove_gain = (
                    float(dist_matrix[u, curr])
                    + float(dist_matrix[curr, v])
                    - float(dist_matrix[u, v])
                )
                best_gain = 1e-5
                best_j = -1

                low = max(0, i - max_shift)
                high = min(n - 2, i + max_shift)
                for j in range(low, high + 1):
                    if j in (i - 1, i):
                        continue
                    a = route[j]
                    b = route[j + 1]
                    insert_cost = (
                        float(dist_matrix[a, curr])
                        + float(dist_matrix[curr, b])
                        - float(dist_matrix[a, b])
                    )
                    # Açısal ceza kontrolü
                    turn_new = self._turn_cost(coords[a], coords[curr], coords[b])
                    if turn_new > 1.2:
                        continue
                    gain = remove_gain - insert_cost
                    if gain > best_gain:
                        best_gain = gain
                        best_j = j

                if best_j != -1:
                    node_val = route.pop(i)
                    ins_idx = best_j + 1 if best_j < i else best_j
                    route.insert(ins_idx, node_val)
                    improved = True
                    break
            if not improved:
                break

        return route

    # ------------------------------------------------------------------
    # 2. Anti-Ping-Pong Sıradaki Hedef Seçici & Stack Cluster Tie-Breaker
    # ------------------------------------------------------------------
    def score_forward_corridor_candidate(
        self,
        nodes: Sequence[Any],
        curr_idx: int,
        cand_idx: int,
        forward_step: int,
        player_pos: Optional[Point2D] = None,
        avg_step_px: float = 120.0,
    ) -> float:
        """
        İleri koridor üzerindeki aday düğümün öncelik skorunu hesaplar (yüksek skor = daha iyi).
        - Monotonik sıra düzenini (`0 -> 1 -> 2 -> ... -> N-1`) kesin olarak korur (`forward_step == 1` en yüksek taban skora sahiptir).
        - `node.max_harvests` / minimap stack kümelerini (`max_harvests >= 6`, `2x`/`3x` stack)
          ileri koridor boyunca hafif pozitif bir eşitlik bozucu (`tie-breaker`) olarak hesaba katar.
        """
        n = len(nodes)
        if n == 0 or cand_idx < 0 or cand_idx >= n or forward_step <= 0:
            return -1e9

        cand_node = nodes[cand_idx]
        curr_xy = _extract_xy(nodes[curr_idx]) if 0 <= curr_idx < n else (0.0, 0.0)
        cand_xy = _extract_xy(cand_node)
        ref_xy = (float(player_pos[0]), float(player_pos[1])) if player_pos is not None else curr_xy
        dist_px = _euclidean(ref_xy, cand_xy)

        # Monotonik sıra koruma terimi: step=1 her zaman step=2'den kesin olarak üstündür (sıfır ping-pong)
        monotonic_order_score = 1000.0 - float(forward_step - 1) * 150.0

        # Mesafe terimi (aynı step kademesi içinde daha yakın olan öndedir)
        distance_penalty = min(80.0, dist_px / max(1.0, float(avg_step_px)) * 12.0)

        # Stack cluster pozitif tie-breaker (max_harvests >= 6 olan 2x/3x minimap kümeleri için hafif +bonus)
        max_h = _extract_max_harvests(cand_node)
        stack_bonus = 0.0
        if max_h >= 9:
            stack_bonus = 18.0
        elif max_h >= 6:
            stack_bonus = 12.0
        elif max_h > 3:
            stack_bonus = 4.0 * float(max_h - 3)

        return round(monotonic_order_score - distance_penalty + stack_bonus, 4)

    def select_next_sequential_node(
        self,
        nodes: Sequence[Any],
        current_seq_pos: Optional[int] = None,
        player_pos: Optional[Point2D] = None,
        forward_window: int = 12,
        allow_wrap: bool = True,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
        current_index: Optional[int] = None,
        stay_on_partial_harvest: bool = True,
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        player_z: Optional[float] = None,
        enable_predictive_prewalk: bool = True,
    ) -> Tuple[Optional[Any], int]:
        """
        Anti-Ping-Pong Sıradaki Hedef Seçici (Stack Cluster Tie-Breaker + 3x Hasat Kilitli
        + Madde 27 Uçurum Filtresi + Madde 28 Cooldown Erken Yürüyüş + Madde 30 Rakip Filtresi):

        - Bot çalışırken bir düğümü topladıktan sonra haritanın öbür ucundaki
          rastgele bir düğüme ASLA gitmez (sıfır ping-pong).
        - Madde 27 (`ZAxisCliffFilter`): `unreachable_cliff = True` (`altitude_band = "cliff_mismatch"`)
          etiketli düğümleri yer koridorundan dışlar (uçuş modu veya aynı irtifa hariç).
        - Madde 28 (`PredictivePreWalkScheduler`): 5 dakikalık (`300 sn`) dinlenme süresinin
          bitmesine `<= 8 sn` kalan ve yürüyüş süresi `~10 sn` olan
          (`rest_remaining_sec <= estimated_walk_sec + 2.0`) düğümleri erken seçer.
        - Madde 30 (`ContestedNodeDetector`): Rakip oyuncu tarafından tutulan düğümleri atlar.
        - Eğer mevcut düğüm (`current_seq_pos`) henüz `1/3` veya `2/3` kısmi hasat
          aşamasındaysa (`0 < current_harvests < max_harvests` ve `stay_on_partial_harvest=True`),
          3 hasat tamamlanana kadar aynı düğümde kalır.

        Dönüş:
            `(selected_node, selected_index)` — Liste boşsa `(None, -1)`.
        """
        try:
            if not nodes:
                return None, -1

            node_list = list(nodes)
            n = len(node_list)
            check_time = time.time() if now is None else float(now)

            if n == 1:
                if not self._is_candidate_selectable(
                    node_list[0],
                    is_flying=is_flying,
                    player_altitude_band=player_altitude_band,
                    player_z=player_z,
                    now=check_time,
                ):
                    return None, -1
                self._current_seq_pos = 0
                return node_list[0], 0

            if current_seq_pos is None and current_index is not None:
                current_seq_pos = current_index

            win = max(1, min(n, int(forward_window)))

            # Ortalama adım uzunluğunu tie-breaker ölçeklemesi için hesapla
            sample_steps = [
                _euclidean(_extract_xy(node_list[i]), _extract_xy(node_list[(i + 1) % n]))
                for i in range(min(n, 16))
            ]
            avg_step_px = float(np.mean(sample_steps)) if sample_steps else 120.0

            # Seçilebilir (uçurum olmayan ve rakip tarafından tutulmayan) indeksler kümesi
            selectable_indices = [
                idx
                for idx in range(n)
                if self._is_candidate_selectable(
                    node_list[idx],
                    is_flying=is_flying,
                    player_altitude_band=player_altitude_band,
                    player_z=player_z,
                    now=check_time,
                )
            ]
            if not selectable_indices:
                return None, -1
            selectable_set = set(selectable_indices)

            # Başlangıç indeksi belirlenmemişse (-1 veya None):
            if current_seq_pos is None or int(current_seq_pos) < 0 or int(current_seq_pos) >= n:
                if player_pos is not None:
                    px, py = float(player_pos[0]), float(player_pos[1])
                    ready_candidates: List[Tuple[float, float, int]] = []
                    for idx in selectable_indices:
                        cand = node_list[idx]
                        if self._is_ready_or_predictive_prewalk(
                            cand,
                            player_pos=player_pos,
                            now=check_time,
                            rest_duration_sec=rest_duration_sec,
                            enable_predictive_prewalk=enable_predictive_prewalk,
                        ):
                            cx, cy = _extract_xy(cand)
                            raw_dist = math.hypot(cx - px, cy - py)
                            tie_bonus = _stack_cluster_tiebreaker_bonus(cand, avg_step_px=avg_step_px)
                            effective_dist = max(0.0, raw_dist - tie_bonus)
                            ready_candidates.append((effective_dist, -float(_extract_max_harvests(cand)), idx))

                    if ready_candidates:
                        ready_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                        chosen_idx = ready_candidates[0][2]
                        self._current_seq_pos = chosen_idx
                        return node_list[chosen_idx], chosen_idx

                    all_candidates: List[Tuple[float, float, int]] = []
                    for idx in selectable_indices:
                        cand = node_list[idx]
                        cx, cy = _extract_xy(cand)
                        raw_dist = math.hypot(cx - px, cy - py)
                        tie_bonus = _stack_cluster_tiebreaker_bonus(cand, avg_step_px=avg_step_px)
                        effective_dist = max(0.0, raw_dist - tie_bonus)
                        all_candidates.append((effective_dist, -float(_extract_max_harvests(cand)), idx))

                    all_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                    chosen_idx = all_candidates[0][2]
                    self._current_seq_pos = chosen_idx
                    return node_list[chosen_idx], chosen_idx
                else:
                    for idx in selectable_indices:
                        if self._is_ready_or_predictive_prewalk(
                            node_list[idx],
                            player_pos=player_pos,
                            now=check_time,
                            rest_duration_sec=rest_duration_sec,
                            enable_predictive_prewalk=enable_predictive_prewalk,
                        ):
                            self._current_seq_pos = idx
                            return node_list[idx], idx
                    first_sel = selectable_indices[0]
                    self._current_seq_pos = first_sel
                    return node_list[first_sel], first_sel

            curr_idx = int(current_seq_pos) % n

            # 0. Aşama: Mevcut düğümde 1/3 veya 2/3 kısmi hasat devam ediyorsa, 3/3 bitene kadar aynı düğümde kal
            if (
                stay_on_partial_harvest
                and curr_idx in selectable_set
                and _is_partially_harvested(node_list[curr_idx])
            ):
                if _is_node_ready(node_list[curr_idx], now=check_time, rest_duration_sec=rest_duration_sec):
                    self._current_seq_pos = curr_idx
                    return node_list[curr_idx], curr_idx

            # 1. Aşama: İleri koridor penceresi (+1 .. +forward_window) içindeki hazır VEYA
            #    Madde 28 (Predictive Pre-Walk: <= 8s kalan ve rest_remaining_sec <= estimated_walk_sec + 2.0)
            #    düğümleri monotonik sıra düzenini (0 -> 1 -> 2 -> ... -> N-1) KESİN koruyarak puanla
            best_ready_idx = -1
            best_ready_score = -1e12
            ref_pos = player_pos if player_pos is not None else _extract_xy(node_list[curr_idx])
            for step in range(1, win + 1):
                cand_idx = curr_idx + step
                if cand_idx >= n:
                    if not allow_wrap:
                        break
                    cand_idx %= n
                if cand_idx not in selectable_set:
                    continue
                candidate = node_list[cand_idx]
                if self._is_ready_or_predictive_prewalk(
                    candidate,
                    player_pos=ref_pos,
                    now=check_time,
                    rest_duration_sec=rest_duration_sec,
                    enable_predictive_prewalk=enable_predictive_prewalk,
                ):
                    score = self.score_forward_corridor_candidate(
                        nodes=node_list,
                        curr_idx=curr_idx,
                        cand_idx=cand_idx,
                        forward_step=step,
                        player_pos=player_pos,
                        avg_step_px=avg_step_px,
                    )
                    if score > best_ready_score:
                        best_ready_score = score
                        best_ready_idx = cand_idx

            if best_ready_idx >= 0:
                self._current_seq_pos = best_ready_idx
                return node_list[best_ready_idx], best_ready_idx

            # 2. Aşama: İleri penceredeki tüm düğümler dinleniyorsa,
            #    geri dönmek (ping-pong) yerine ileri yönde (+win+1 .. +n) sıradaki ilk hazır / pre-walk düğüme ilerle
            max_scan = n if allow_wrap else (n - 1 - curr_idx)
            for step in range(win + 1, max_scan + 1):
                cand_idx = (curr_idx + step) % n
                if cand_idx not in selectable_set:
                    continue
                candidate = node_list[cand_idx]
                if self._is_ready_or_predictive_prewalk(
                    candidate,
                    player_pos=ref_pos,
                    now=check_time,
                    rest_duration_sec=rest_duration_sec,
                    enable_predictive_prewalk=enable_predictive_prewalk,
                ):
                    self._current_seq_pos = cand_idx
                    return candidate, cand_idx

            # 3. Aşama: Rotadaki TÜM düğümler uzun dinlenme süresindeyse,
            #    asla geri gitme; ileri yöndeki ilk seçilebilir düğümü veya ileri penceredeki
            #    en erken hazır olacak sıradaki düğümü (stack cluster tie-breaker ile) seç.
            forward_selectables: List[int] = []
            for step in range(1, max_scan + 1):
                c_idx = (curr_idx + step) % n
                if c_idx in selectable_set:
                    forward_selectables.append(c_idx)
            if not forward_selectables:
                forward_selectables = list(selectable_indices)

            best_idx = forward_selectables[0]
            next_max_h = _extract_max_harvests(node_list[best_idx])
            next_stack_discount = 3.5 if next_max_h >= 6 else 0.0
            best_wait = _remaining_rest_seconds(
                node_list[best_idx], now=check_time, rest_duration_sec=rest_duration_sec
            ) - next_stack_discount

            for cand_idx in forward_selectables[: min(len(forward_selectables), min(win, 4))]:
                cand_max_h = _extract_max_harvests(node_list[cand_idx])
                stack_discount = 4.5 if cand_max_h >= 9 else (3.5 if cand_max_h >= 6 else 0.0)
                wait_sec = (
                    _remaining_rest_seconds(
                        node_list[cand_idx], now=check_time, rest_duration_sec=rest_duration_sec
                    )
                    - stack_discount
                )
                if wait_sec + 5.0 < best_wait:
                    best_wait = wait_sec
                    best_idx = cand_idx

            self._current_seq_pos = best_idx
            return node_list[best_idx], best_idx
        except Exception as exc:
            logger.error("select_next_sequential_node hatası: %s", exc)
            if nodes:
                fallback_idx = 0 if current_seq_pos is None else (int(current_seq_pos) + 1) % len(nodes)
                return nodes[fallback_idx], fallback_idx
            return None, -1

    def get_next_in_sequence(
        self,
        nodes: Sequence[Any],
        current_seq_pos: Optional[int] = None,
        player_pos: Optional[Point2D] = None,
        forward_window: int = 12,
        allow_wrap: bool = True,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
        stay_on_partial_harvest: bool = True,
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        player_z: Optional[float] = None,
        enable_predictive_prewalk: bool = True,
    ) -> Tuple[Optional[Any], int]:
        """
        `select_next_sequential_node` ile tam uyumlu Anti-Ping-Pong sıradaki hedef seçici.
        """
        return self.select_next_sequential_node(
            nodes=nodes,
            current_seq_pos=current_seq_pos,
            player_pos=player_pos,
            forward_window=forward_window,
            allow_wrap=allow_wrap,
            now=now,
            rest_duration_sec=rest_duration_sec,
            stay_on_partial_harvest=stay_on_partial_harvest,
            is_flying=is_flying,
            player_altitude_band=player_altitude_band,
            player_z=player_z,
            enable_predictive_prewalk=enable_predictive_prewalk,
        )

    # ------------------------------------------------------------------
    # 3. Dinamik Araya Ekleme (Cheapest Insertion)
    # ------------------------------------------------------------------
    def insert_node_in_sequence(
        self,
        ordered_nodes: Sequence[Any],
        new_node: Any,
        closed_loop: bool = True,
    ) -> Tuple[List[Any], int]:
        """
        Oyuncu gezerken yeni bir kaynak keşfedildiğinde, mevcut sırayı bozmadan
        ekleme maliyeti `d(i, new) + d(new, i+1) - d(i, i+1)` (ve açısal uyum)
        en düşük olan iki ardışık düğümün tam arasına yerleştirir.

        Dönüş:
            `(updated_nodes_list, inserted_index)`
        """
        try:
            current_list = list(ordered_nodes) if ordered_nodes else []
            n = len(current_list)

            if n == 0:
                _set_seq_index(new_node, 1)
                return [new_node], 0

            if n == 1:
                current_list.append(new_node)
                for idx, item in enumerate(current_list, start=1):
                    _set_seq_index(item, idx)
                return current_list, 1

            new_pt = _extract_xy(new_node)
            coords = [_extract_xy(item) for item in current_list]

            best_cost = float("inf")
            best_insert_idx = n

            num_edges = n if closed_loop else (n - 1)
            for i in range(num_edges):
                j = (i + 1) % n
                pi = coords[i]
                pj = coords[j]

                d_i_new = _euclidean(pi, new_pt)
                d_new_j = _euclidean(new_pt, pj)
                d_i_j = _euclidean(pi, pj)

                # Temel Cheapest-Insertion maliyeti: d(i, new) + d(new, i+1) - d(i, i+1)
                insertion_delta = (d_i_new + d_new_j) - d_i_j

                # Açısal süreklilik katkısı (keskin zikzak oluşturmasını önler)
                turn_at_new = self._turn_cost(pi, new_pt, pj)
                angular_adjust = 0.15 * self.angular_penalty_weight * max(1.0, d_i_j) * turn_at_new
                total_cost = insertion_delta + angular_adjust

                if total_cost < best_cost:
                    best_cost = total_cost
                    best_insert_idx = i + 1

            if not closed_loop:
                # Açık rotada başa veya sona ekleme maliyetini de kontrol et
                cost_head = _euclidean(new_pt, coords[0])
                cost_tail = _euclidean(coords[-1], new_pt)
                if cost_head < best_cost:
                    best_cost = cost_head
                    best_insert_idx = 0
                if cost_tail < best_cost:
                    best_insert_idx = n

            current_list.insert(best_insert_idx, new_node)
            for seq_idx, item in enumerate(current_list, start=1):
                _set_seq_index(item, seq_idx)

            logger.info(
                "Yeni düğüm sıraya eklendi: indeks=%d/%d, ek_maliyet=%.2f px",
                best_insert_idx + 1,
                len(current_list),
                best_cost,
            )
            return current_list, best_insert_idx
        except Exception as exc:
            logger.error("insert_node_in_sequence hatası: %s", exc)
            fallback = list(ordered_nodes) if ordered_nodes else []
            fallback.append(new_node)
            for seq_idx, item in enumerate(fallback, start=1):
                _set_seq_index(item, seq_idx)
            return fallback, len(fallback) - 1

    # ------------------------------------------------------------------
    # 4. Rota Kalite Metrikleri
    # ------------------------------------------------------------------
    def evaluate_sequence_quality(
        self,
        nodes: Sequence[Any],
        closed_loop: bool = True,
    ) -> Dict[str, Any]:
        """
        Verilen düğüm diziliminin geometrik ve akış kalitesini ölçer:
        - `total_distance_px`: Toplam rota uzunluğu (piksel)
        - `avg_step_px`: Ardışık iki düğüm arası ortalama adım mesafesi
        - `max_jump_px`: En büyük tekil sıçrama mesafesi
        - `min_step_px`: En küçük adım mesafesi
        - `std_step_px`: Adım mesafelerinin standart sapması
        - `sharp_turns_count`: 90 dereceden keskin dönüş (`cos(theta) < 0`) sayısı
        - `mean_turn_angle_deg`: Ortalama dönüş açısı (derece)
        - `crossings_count`: Kendi kendini kesen kenar çifti sayısı
        - `smoothness_score`: `0.0 .. 100.0` arası pürüzsüzlük skoru
        - `node_count`: Düğüm sayısı
        """
        try:
            node_list = list(nodes) if nodes else []
            n = len(node_list)
            if n <= 1:
                return {
                    "node_count": n,
                    "total_distance_px": 0.0,
                    "avg_step_px": 0.0,
                    "max_jump_px": 0.0,
                    "min_step_px": 0.0,
                    "std_step_px": 0.0,
                    "sharp_turns_count": 0,
                    "mean_turn_angle_deg": 0.0,
                    "crossings_count": 0,
                    "smoothness_score": 100.0,
                }

            coords = [_extract_xy(item) for item in node_list]
            steps: List[float] = []
            edges: List[Tuple[Point2D, Point2D]] = []

            num_steps = n if (closed_loop and n > 2) else (n - 1)
            for i in range(num_steps):
                p1 = coords[i]
                p2 = coords[(i + 1) % n]
                d = _euclidean(p1, p2)
                steps.append(d)
                edges.append((p1, p2))

            total_dist = float(sum(steps))
            avg_step = float(total_dist / len(steps)) if steps else 0.0
            max_jump = float(max(steps)) if steps else 0.0
            min_step = float(min(steps)) if steps else 0.0
            std_step = float(np.std(steps)) if steps else 0.0

            # Dönüş açıları ve keskin dönüş (> 90 derece) sayısı
            turn_angles_deg: List[float] = []
            sharp_turns = 0
            if n >= 3:
                num_turns = n if closed_loop else (n - 2)
                for i in range(num_turns):
                    p_prev = coords[i]
                    p_curr = coords[(i + 1) % n]
                    p_next = coords[(i + 2) % n]
                    ang_rad = _turn_angle_rad(p_prev, p_curr, p_next)
                    ang_deg = math.degrees(ang_rad)
                    turn_angles_deg.append(ang_deg)
                    if ang_deg > 90.0:
                        sharp_turns += 1

            mean_turn_deg = float(np.mean(turn_angles_deg)) if turn_angles_deg else 0.0

            # Kesişen kenar sayısı
            crossings = 0
            num_e = len(edges)
            for i in range(num_e):
                a1, a2 = edges[i]
                for j in range(i + 2, num_e):
                    if closed_loop and i == 0 and j == num_e - 1:
                        continue
                    b1, b2 = edges[j]
                    if _segments_intersect(a1, a2, b1, b2):
                        crossings += 1

            # Pürüzsüzlük Skoru (0..100):
            # 1) Açısal akış puanı (45 puan): ortalama dönüş açısı ve keskin dönüş oranı
            sharp_ratio = (sharp_turns / float(max(1, len(turn_angles_deg)))) if turn_angles_deg else 0.0
            angle_norm = min(1.0, mean_turn_deg / 135.0)
            angular_score = 45.0 * max(0.0, 1.0 - (0.55 * sharp_ratio + 0.45 * angle_norm))

            # 2) Adım tutarlılığı ve sıçrama puanı (30 puan): max_jump / ideal_step ve CV
            pts_arr = np.asarray(coords, dtype=np.float64)
            span_x = float(np.max(pts_arr[:, 0]) - np.min(pts_arr[:, 0]))
            span_y = float(np.max(pts_arr[:, 1]) - np.min(pts_arr[:, 1]))
            bbox_diag = max(1.0, math.hypot(span_x, span_y))
            jump_ratio = max(0.0, (max_jump / max(1.0, avg_step)) - 1.5) / 6.0
            length_efficiency = min(1.0, (bbox_diag * 2.6) / max(1.0, total_dist))
            step_score = 30.0 * max(0.0, 0.55 * length_efficiency + 0.45 * max(0.0, 1.0 - jump_ratio))

            # 3) Topolojik temizlik / kesişimsizlik puanı (25 puan)
            crossing_penalty = min(1.0, crossings / max(1.0, float(n) * 0.25))
            topology_score = 25.0 * (1.0 - crossing_penalty)

            smoothness_score = round(max(0.0, min(100.0, angular_score + step_score + topology_score)), 2)

            return {
                "node_count": n,
                "total_distance_px": round(total_dist, 2),
                "avg_step_px": round(avg_step, 2),
                "max_jump_px": round(max_jump, 2),
                "min_step_px": round(min_step, 2),
                "std_step_px": round(std_step, 2),
                "sharp_turns_count": int(sharp_turns),
                "mean_turn_angle_deg": round(mean_turn_deg, 2),
                "crossings_count": int(crossings),
                "smoothness_score": smoothness_score,
            }
        except Exception as exc:
            logger.error("evaluate_sequence_quality hatası: %s", exc)
            return {
                "node_count": len(nodes) if nodes else 0,
                "total_distance_px": 0.0,
                "avg_step_px": 0.0,
                "max_jump_px": 0.0,
                "min_step_px": 0.0,
                "std_step_px": 0.0,
                "sharp_turns_count": 0,
                "mean_turn_angle_deg": 0.0,
                "crossings_count": 0,
                "smoothness_score": 0.0,
            }

    def compute_route_efficiency_score(self, nodes: list) -> float:
        """
        Computes a normalized efficiency score in [0.0, 100.0] for the given nodes list.
        Returns 100.0 if len(nodes) <= 1, else a bounded smoothness/distance metric.
        """
        try:
            if not nodes or len(nodes) <= 1:
                return 100.0
            quality = self.evaluate_sequence_quality(nodes, closed_loop=False)
            score = quality.get("smoothness_score", 0.0)
            return float(max(0.0, min(100.0, score)))
        except Exception as exc:
            logger.error("compute_route_efficiency_score error: %s", exc)
            return 0.0

    def get_sequential_router_health_summary(self) -> dict:
        try:
            return {"sequential_router_ready": True}
        except Exception as exc:
            logger.error("get_sequential_router_health_summary error: %s", exc)
            return {"sequential_router_ready": False}

    def get_sequential_router_ready_flag(self) -> dict:
        return {"module": "sequential_router", "ready": True}

    def get_sequential_router_version_info(self) -> dict:
        return {"module": "sequential_router", "version": "2.0.0", "ready": True}

    def get_sequential_router_diagnostic_snapshot(self) -> dict:
        return {"module": "sequential_router", "status": "ok", "ready": True}


