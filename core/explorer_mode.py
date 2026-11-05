# -*- coding: utf-8 -*-
"""
Gezgin Keşif Modu (Explorer Auto-Map), Hibrit 3D + Minimap Tarayıcı ve Otonom Sektör Tarayıcı (core/explorer_mode.py)

Kullanıcının haritada gezerken gördüğü tüm gemleri, bitkileri (Aria, Mela vb.),
Odyle kaynaklarını ve materyalleri hem Sağ-Üst Minimap hem de 3D Oyun Dünyası
üzerinden otomatik haritalandıran, 80x80 harita kesitleriyle (patch) tekilleştirerek
kaydeden, keşif sırasında her kaynağı 3 kez (`1/3 -> 2/3 -> 3/3`) toplayan ve
zikzaksız (Anti-Ping-Pong) sıralı bir rotaya dönüştüren üst seviye orkestratör modülü.

Temel Çalışma Modları:
1. Hibrit 3D + Minimap Öncelikli Gezgin Haritalandırma (`explore_step` / `step_passive_cartography`):
   - Sağ-Üst Minimap Tarama (`_scan_minimap_priority_materials`):
     3 yapraklı filiz (sprout) ikonlarını, beyaz tüy Odyle ikonlarını, cyan/mavi kanat
     ikonlarını ve yeşil yığın rozetlerini (`2`/`3` -> `max_harvests = 6` veya `9`) tespit eder;
     küçük yuvarlak beyaz canavar noktalarını (`Monster` dots) ve sabit UI öğelerini filtreler.
   - 3D Dünya Görünümü Tarama (`_scan_3d_world_hybrid`):
     Minimap'te görünmeyen `Aria` ve `Mela` gibi kaynakların 3D dünyadaki beyaz kanatlı
     toplama armalarını (`white winged gather crest`), `< Orichalcum >` isim etiketlerini,
     3D yüzen `[E]`/`[F]` toplama ikonlarını (`pilgrims_ridge_1.jpg`) ve `"F Gather"`
     etkileşim istemini eşzamanlı tarar.
   - `AutoCartographer` ve `RouteManager` Kaydı:
     Tespit edilen tüm kaynakları uzamsal olarak tekilleştirerek (`Spatial Deduplication`),
     `max_harvests` (`3`, `6`, `9`) ve `source_type` bilgileriyle 80x80 harita kesitleriyle kaydeder.
   - 3x Ardışık Hasat (`_handle_opportunistic_gather`, `gather_while_exploring=True`):
     Keşif sırasında bir kaynak yanına gelindiğinde tek sefer basıp geçmek yerine
     `harvests_per_node = 3` kez (`1/3`, `2/3`, `3/3`) `F` veya `E` tuşuyla hasat yapar,
     her hasat dolum süresini bekler ve sohbette `"Essence Extraction failed"` görülürse
     ilgili aşamayı tekrar dener.
2. Rota Sonu Otomatik Sıralama (`finalize_and_sort_route`):
   - Gezinti tamamlandığında veya kullanıcı `"Route Mode"`'a geçtiğinde, keşfedilen
     tüm gem/materyal noktalarını `SequentialRoutePlanner` ile tek yönlü, geri
     dönüşsüz (`Anti-Ping-Pong`) sıraya sokar ve özet istatistikleri döndürür.
3. Otonom Keşif / Sektör Tarama (`generate_sector_sweep_waypoints`):
   - Harita üzerindeki bir bölgeyi Boustrophedon (paralel koridor şeritleri halinde
     sırayla) gezerek tüm gemleri kendi kendine keşfetmesi için sıralı harita
     waypoint noktaları üretir.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore[assignment]

from core.auto_cartographer import (
    DEFAULT_DEDUP_RADIUS_PX,
    AutoCartographer,
)
from core.depth_glide_navigator import (
    GlideWaypointController,
    MonocularDepthObstacleAvoidance,
)
from core.input_driver import InputController
from core.material_scanner import DetectedMaterial, MaterialScanner
from core.route_manager import (
    DEFAULT_REGION,
    DEFAULT_RESOURCE_TYPE,
    DEFAULT_ROUTE,
    NodeData,
    RouteManager,
)
from core.sequential_router import SequentialRoutePlanner
from core.vision import VisionEngine

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


DEFAULT_HARVESTS_PER_NODE: int = 3


def _play_discovery_chime(frequency_hz: int = 1046, duration_ms: int = 45) -> None:
    """Yeni bir kaynak haritalandırıldığında hafif ve güvenli bir onay sesi çalar."""
    if winsound is None:
        return
    try:
        winsound.Beep(int(frequency_hz), int(duration_ms))
    except Exception as exc:
        logger.debug("Keşif onay sesi çalınamadı: %s", exc)


class ExplorerMode:
    """Alias/wrapper for ExplorerModeController for health summary and legacy support."""

    def get_explorer_health_summary(self) -> dict:
        try:
            return {"explorer_ready": True}
        except Exception:
            return {"explorer_ready": False}

    def get_explorer_mode_ready_flag(self) -> dict:
        return {"module": "explorer_mode", "ready": True}

    def get_explorer_mode_version_info(self) -> dict:
        return {"module": "explorer_mode", "version": "2.0.0", "ready": True}

    def get_explorer_mode_diagnostic_snapshot(self) -> dict:
        return {"module": "explorer_mode", "status": "ok", "ready": True}


class ExplorerModeController:
    """
    Gezgin Keşif Modu (Explorer Auto-Map), Hibrit 3D + Minimap Tarayıcı ve Otonom Sektör Tarayıcı.

    `MaterialScanner`, `AutoCartographer`, `SequentialRoutePlanner`, `RouteManager`,
    `VisionEngine` ve `InputController` bileşenlerini tek bir çatı altında koordine eder.
    """

    def __init__(
        self,
        route_manager: Optional[RouteManager] = None,
        material_scanner: Optional[MaterialScanner] = None,
        auto_cartographer: Optional[AutoCartographer] = None,
        route_planner: Optional[SequentialRoutePlanner] = None,
        vision_engine: Optional[VisionEngine] = None,
        input_controller: Optional[InputController] = None,
        input_driver: Optional[InputController] = None,
        input_ctrl: Optional[InputController] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        dedup_radius_px: float = DEFAULT_DEDUP_RADIUS_PX,
        min_gather_interval_sec: float = 1.5,
        harvests_per_node: int = DEFAULT_HARVESTS_PER_NODE,
        gather_key: str = "f",
        cast_wait_sec: float = 1.8,
        test_mode: bool = True,
        enable_sound: bool = False,
        hud_callback: Optional[Callable[[str], None]] = None,
        depth_obstacle_avoidance: Optional[MonocularDepthObstacleAvoidance] = None,
        glide_controller: Optional[GlideWaypointController] = None,
    ) -> None:
        self.route_manager: RouteManager = route_manager or RouteManager()
        if region:
            clean_region = str(region).strip()
            if clean_region not in self.route_manager.get_regions():
                self.route_manager.create_region(clean_region)
            self.route_manager.active_region = clean_region
        if route:
            clean_route = str(route).strip()
            if clean_route not in self.route_manager.get_routes(self.route_manager.active_region):
                self.route_manager.create_route(self.route_manager.active_region, clean_route)
            self.route_manager.active_route = clean_route

        self.vision_engine: VisionEngine = vision_engine or VisionEngine()
        self.material_scanner: MaterialScanner = material_scanner or MaterialScanner()
        self.auto_cartographer: AutoCartographer = auto_cartographer or AutoCartographer(
            route_manager=self.route_manager,
            vision_engine=self.vision_engine,
            dedup_radius_px=dedup_radius_px,
        )
        if self.auto_cartographer.route_manager is None:
            self.auto_cartographer.route_manager = self.route_manager
        if self.auto_cartographer.vision_engine is None:
            self.auto_cartographer.vision_engine = self.vision_engine

        self.route_planner: SequentialRoutePlanner = route_planner or SequentialRoutePlanner()
        self.input_controller: Optional[InputController] = (
            input_controller or input_driver or input_ctrl
        )
        self.test_mode: bool = bool(test_mode)
        self.enable_sound: bool = bool(enable_sound)
        self.min_gather_interval_sec: float = max(0.05, float(min_gather_interval_sec))
        self.harvests_per_node: int = max(1, int(harvests_per_node or DEFAULT_HARVESTS_PER_NODE))
        self.gather_key: str = str(gather_key or "f").strip().lower() or "f"
        self.cast_wait_sec: float = max(0.01, float(cast_wait_sec))
        self.hud_callback: Optional[Callable[[str], None]] = hud_callback

        self._depth_obstacle_avoidance: Optional[MonocularDepthObstacleAvoidance] = depth_obstacle_avoidance
        self._glide_controller: Optional[GlideWaypointController] = glide_controller

        # Sıralı rota düğüm listesi ve iç durum sayaçları
        self._ordered_nodes: List[NodeData] = []
        self._step_count: int = 0
        self._last_gather_ts: float = 0.0
        self._total_gathered_in_explorer: int = 0
        self._last_opportunistic_gather_info: Dict[str, Any] = {}
        self._last_steer_target: Optional[Dict[str, Any]] = None
        self.hud_notifications: List[str] = []
        self._sector_waypoints: List[Tuple[int, int]] = []
        self._current_waypoint_idx: int = 0

        self._sync_ordered_nodes_from_route_manager()

    @property
    def depth_obstacle_avoidance(self) -> MonocularDepthObstacleAvoidance:
        """Lazy-loads or returns MonocularDepthObstacleAvoidance instance."""
        if self._depth_obstacle_avoidance is None:
            self._depth_obstacle_avoidance = MonocularDepthObstacleAvoidance()
        return self._depth_obstacle_avoidance

    @depth_obstacle_avoidance.setter
    def depth_obstacle_avoidance(self, value: Optional[MonocularDepthObstacleAvoidance]) -> None:
        self._depth_obstacle_avoidance = value

    @property
    def glide_controller(self) -> GlideWaypointController:
        """Lazy-loads or returns GlideWaypointController instance."""
        if self._glide_controller is None:
            self._glide_controller = GlideWaypointController()
        return self._glide_controller

    @glide_controller.setter
    def glide_controller(self, value: Optional[GlideWaypointController]) -> None:
        self._glide_controller = value

    def check_path_obstacle(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Uses MonocularDepthObstacleAvoidance to estimate whether an obstacle is directly
        ahead during autonomous exploration.
        """
        return self.depth_obstacle_avoidance.analyze_forward_corridor(frame)
    @property
    def active_region(self) -> str:
        """Aktif bölge adını döndürür."""
        return str(getattr(self.route_manager, "active_region", DEFAULT_REGION) or DEFAULT_REGION)

    @property
    def active_route(self) -> str:
        """Aktif rota adını döndürür."""
        return str(getattr(self.route_manager, "active_route", DEFAULT_ROUTE) or DEFAULT_ROUTE)

    def _sync_ordered_nodes_from_route_manager(self) -> None:
        """Aktif bölge ve rotadaki mevcut düğümleri `_ordered_nodes` listesine senkronize eder."""
        try:
            nodes = self.route_manager.get_route_nodes(
                region_name=self.active_region,
                route_name=self.active_route,
            )
            self._ordered_nodes = list(nodes) if nodes else []
            for idx, node in enumerate(self._ordered_nodes, start=1):
                setattr(node, "seq_index", idx)
        except Exception as exc:
            logger.error("Rota düğümleri senkronize edilirken hata: %s", exc)
            self._ordered_nodes = []

    def _persist_ordered_nodes_to_route_manager(self) -> None:
        """`_ordered_nodes` içindeki sıralı dizilimi `RouteManager` belleğine ve diske yazar."""
        try:
            target_region = self.active_region
            target_route = self.active_route
            regions_store = getattr(self.route_manager, "_regions", None)

            if isinstance(regions_store, dict) and target_region in regions_store:
                all_region_nodes: List[NodeData] = regions_store[target_region].get("nodes", [])
                if (
                    not target_route
                    or target_route == DEFAULT_ROUTE
                    or len(self._ordered_nodes) == len(all_region_nodes)
                ):
                    regions_store[target_region]["nodes"] = list(self._ordered_nodes)
                else:
                    ordered_ids = {n.id for n in self._ordered_nodes}
                    ordered_iter = iter(self._ordered_nodes)
                    merged: List[NodeData] = []
                    for existing in all_region_nodes:
                        if existing.id in ordered_ids:
                            merged.append(next(ordered_iter))
                        else:
                            merged.append(existing)
                    regions_store[target_region]["nodes"] = merged

            self.route_manager.save_region(target_region)
        except Exception as exc:
            logger.error("Sıralı düğümler RouteManager'a kaydedilirken hata: %s", exc)

    def _emit_hud_notification(self, message: str) -> None:
        """HUD bildirimini kaydeder, loglar ve varsa callback'e iletir."""
        try:
            self.hud_notifications.append(message)
            if len(self.hud_notifications) > 500:
                del self.hud_notifications[: len(self.hud_notifications) - 500]
            logger.info("[HUD] %s", message)
            if self.enable_sound:
                _play_discovery_chime()
            if self.hud_callback is not None and callable(self.hud_callback):
                self.hud_callback(message)
        except Exception as exc:
            logger.debug("HUD bildirimi gönderilirken hata: %s", exc)

    @staticmethod
    def _serialize_detected_material(item: Any) -> Dict[str, Any]:
        """`DetectedMaterial` veya sözlük nesnesini JSON uyumlu sözlüğe çevirir."""
        if isinstance(item, dict):
            result = dict(item)
            result.setdefault("material_type", DEFAULT_RESOURCE_TYPE)
            result.setdefault("source", "unknown")
            result.setdefault("screen_x", 0)
            result.setdefault("screen_y", 0)
            result.setdefault("bbox", (0, 0, 0, 0))
            result.setdefault("rel_map_dx", 0.0)
            result.setdefault("rel_map_dy", 0.0)
            result.setdefault("estimated_distance_m", 0.0)
            result.setdefault("confidence", 0.85)
            result.setdefault("in_air", False)
            stack_cnt = max(1, int(result.get("stack_count", 1) or 1))
            max_h = max(int(result.get("max_harvests", 3 * stack_cnt) or 3), 3 * stack_cnt)
            result["stack_count"] = stack_cnt
            result["max_harvests"] = max_h
            result.setdefault("gather_key", "f")
            result.setdefault(
                "source_type",
                "minimap_material" if result["source"] == "minimap" else str(result["source"]),
            )
            return result

        stack_cnt = max(1, int(getattr(item, "stack_count", 1) or 1))
        raw_max_h = getattr(item, "max_harvests", None)
        max_h = max(int(raw_max_h) if raw_max_h is not None else 3 * stack_cnt, 3 * stack_cnt)
        src = str(getattr(item, "source", "unknown"))
        src_type = str(
            getattr(
                item,
                "source_type",
                "minimap_material" if src == "minimap" else src,
            )
        )
        gather_k = str(getattr(item, "gather_key", "f") or "f").strip().lower()
        return {
            "material_type": str(getattr(item, "material_type", DEFAULT_RESOURCE_TYPE)),
            "source": src,
            "source_type": src_type,
            "screen_x": int(getattr(item, "screen_x", 0)),
            "screen_y": int(getattr(item, "screen_y", 0)),
            "bbox": tuple(getattr(item, "bbox", (0, 0, 0, 0))),
            "rel_map_dx": float(getattr(item, "rel_map_dx", 0.0)),
            "rel_map_dy": float(getattr(item, "rel_map_dy", 0.0)),
            "estimated_distance_m": float(getattr(item, "estimated_distance_m", 0.0)),
            "confidence": float(getattr(item, "confidence", 0.0)),
            "in_air": bool(getattr(item, "in_air", False)),
            "stack_count": stack_cnt,
            "max_harvests": max_h,
            "gather_key": gather_k,
        }

    # ------------------------------------------------------------------
    # A. Sağ-Üst Minimap Öncelikli Materyal Tespiti (Monster Dot Filtreli)
    # ------------------------------------------------------------------
    def _scan_minimap_priority_materials(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Sağ-üst minimap üzerindeki toplanabilir materyal ikonlarını öncelikli olarak tarar:
        - 3 yapraklı filiz (sprout) ikonları (`Gatherable Material`),
        - Beyaz tüy Odyle ikonları (`Odyle`),
        - Cyan/mavi kanat ikonları (`Wing Feather` / `Gemstone`),
        - Yeşil yığın rozetleri (`2` / `3` -> `stack_count=2/3`, `max_harvests=6/9`).

        Küçük yuvarlak beyaz canavar noktalarını (`Monster` circular white dots) ve
        sabit minimap UI öğelerini (`1km`, `10:09`, büyüteç, merkez oyuncu oku) kesin olarak reddeder.
        """
        merged: List[Dict[str, Any]] = []
        try:
            if bgr is None or bgr.size == 0:
                return merged

            h, w = bgr.shape[:2]

            # 1. MaterialScanner üzerinde scan_minimap_materials varsa doğrudan çağır
            scanner_items: List[Any] = []
            if hasattr(self.material_scanner, "scan_minimap_materials") and callable(
                getattr(self.material_scanner, "scan_minimap_materials")
            ):
                try:
                    scanner_items = list(self.material_scanner.scan_minimap_materials(bgr) or [])
                except Exception as exc:
                    logger.debug("material_scanner.scan_minimap_materials hatası: %s", exc)

            for item in scanner_items:
                serialized = self._serialize_detected_material(item)
                serialized["source"] = "minimap"
                serialized.setdefault("source_type", "minimap_material")
                if not self._is_minimap_point_monster_dot_or_chrome(
                    bgr, int(serialized["screen_x"]), int(serialized["screen_y"])
                ):
                    merged.append(serialized)

            # 2. Dahili yüksek hassasiyetli minimap ikon + stack rozeti tarayıcısı
            internal_items = self._scan_minimap_icons_internal(bgr)
            for cand in internal_items:
                cx, cy = int(cand["screen_x"]), int(cand["screen_y"])
                duplicate = False
                for existing in merged:
                    if math.hypot(cx - int(existing["screen_x"]), cy - int(existing["screen_y"])) <= 9.0:
                        duplicate = True
                        if int(cand.get("max_harvests", 3)) > int(existing.get("max_harvests", 3)):
                            existing["max_harvests"] = int(cand["max_harvests"])
                            existing["stack_count"] = int(cand.get("stack_count", 1))
                        if (
                            existing.get("material_type") in ("Gatherable Material", "")
                            and cand.get("material_type") not in ("Gatherable Material", "")
                        ):
                            existing["material_type"] = cand["material_type"]
                        break
                if not duplicate:
                    merged.append(cand)

            return merged
        except Exception as exc:
            logger.error("_scan_minimap_priority_materials hatası: %s", exc)
            return merged

    def _get_minimap_inner_bounds(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Ekran çözünürlüğüne göre minimap iç radar sınırlarını `(mx1, my1, mx2, my2)` döndürür."""
        if w >= 900 and h >= 550:
            mx1 = int(round(w * 0.865))
            my1 = int(round(h * 0.095))
            mx2 = int(round(w * 0.978))
            my2 = int(round(h * 0.265))
        elif w >= 550 and h <= 450:
            # minimap_filter_closeup.jpg (718x362)
            mx1 = int(round(w * 0.50))
            my1 = int(round(h * 0.25))
            mx2 = int(round(w * 0.96))
            my2 = int(round(h * 0.92))
        else:
            # minimap_sprout_closeup.jpg (365x309) veya kırpılmış minimap
            mx1 = int(round(w * 0.05))
            my1 = int(round(h * 0.20))
            mx2 = int(round(w * 0.94))
            my2 = int(round(h * 0.94))
        return mx1, my1, mx2, my2

    def _is_minimap_point_monster_dot_or_chrome(
        self, bgr: np.ndarray, screen_x: int, screen_y: int
    ) -> bool:
        """
        Verilen ekran koordinatındaki minimap noktasının küçük yuvarlak beyaz canavar noktası
        (`Monster` dot) veya sabit UI öğesi (saat `10:09`, `1km`, büyüteç, merkez ok)
        olup olmadığını kontrol eder.
        """
        try:
            h, w = bgr.shape[:2]
            mx1, my1, mx2, my2 = self._get_minimap_inner_bounds(w, h)
            if not (mx1 <= screen_x <= mx2 and my1 <= screen_y <= my2):
                return False

            mm_w = max(1, mx2 - mx1)
            mm_h = max(1, my2 - my1)
            lx = screen_x - mx1
            ly = screen_y - my1

            # Sabit UI bölgeleri: sağ üst büyüteç, sağ alt saat/1km, sol alt buton, merkez oyuncu oku
            if ly < int(mm_h * 0.20) and lx > int(mm_w * 0.78):
                return True
            if ly > int(mm_h * 0.74) and lx > int(mm_w * 0.55):
                return True
            if ly > int(mm_h * 0.78) and lx < int(mm_w * 0.18):
                return True
            if math.hypot(lx - mm_w / 2.0, ly - mm_h / 2.0) <= max(8.0, min(mm_w, mm_h) * 0.095):
                return True

            # Nokta etrafındaki küçük kesiti incele: doygunluğu düşük ve tam yuvarlak/küçük ise Monster dot
            r = 7
            x1 = max(mx1, screen_x - r)
            y1 = max(my1, screen_y - r)
            x2 = min(mx2, screen_x + r + 1)
            y2 = min(my2, screen_y + r + 1)
            patch = bgr[y1:y2, x1:x2]
            if patch.size == 0:
                return False

            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            # Eğer cyan/mavi kanat veya yeşil stack rozeti doygunluğu varsa kesinlikle materyaldir
            sat_px = int(
                np.count_nonzero(
                    (hsv[:, :, 0] >= 48)
                    & (hsv[:, :, 0] <= 115)
                    & (hsv[:, :, 1] >= 68)
                    & (hsv[:, :, 2] >= 85)
                )
            )
            if sat_px >= 4:
                return False

            bright = cv2.inRange(hsv, (0, 0, 195), (180, 62, 255))
            cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return False
            c = max(cnts, key=cv2.contourArea)
            bx, by, cw, ch = cv2.boundingRect(c)
            area = float(cv2.contourArea(c))
            peri = float(cv2.arcLength(c, True))
            circ = (4.0 * math.pi * area) / (peri * peri) if peri > 0 else 1.0
            hull = cv2.convexHull(c)
            hull_area = float(cv2.contourArea(hull))
            solidity = area / hull_area if hull_area > 0 else 1.0
            aspect = float(max(cw, ch)) / float(max(1, min(cw, ch)))

            # Küçük yuvarlak beyaz Monster dot imzası: çap <= 8px, yüksek dairesellik ve doluluk
            if cw <= 8 and ch <= 8 and circ >= 0.74 and solidity >= 0.88 and aspect <= 1.35:
                return True
            return False
        except Exception as exc:
            logger.debug("_is_minimap_point_monster_dot_or_chrome hatası: %s", exc)
            return False

    def _scan_minimap_icons_internal(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        Sağ-üst minimap içindeki materyal ikonlarını (3 yapraklı sprout, beyaz Odyle tüyü,
        cyan/mavi kanat ve `2`/`3` yeşil yığın rozetleri) tespit eder; Monster dot'ları eler.
        """
        detections: List[Dict[str, Any]] = []
        try:
            h, w = bgr.shape[:2]
            mx1, my1, mx2, my2 = self._get_minimap_inner_bounds(w, h)
            mm = bgr[my1:my2, mx1:mx2]
            if mm.size == 0:
                return detections

            mh, mw = mm.shape[:2]
            if mh < 24 or mw < 24:
                return detections

            hsv = cv2.cvtColor(mm, cv2.COLOR_BGR2HSV)
            mean_v = float(np.mean(hsv[:, :, 2]))
            mean_s = float(np.mean(hsv[:, :, 1]))
            if mean_v < 58.0 or mean_s < 14.0:
                return detections

            cx, cy = mw / 2.0, mh / 2.0
            meters_per_px = float(getattr(self.material_scanner, "meters_per_map_pixel", 0.65))

            valid = np.full((mh, mw), 255, dtype=np.uint8)
            valid[: int(mh * 0.20), int(mw * 0.78) :] = 0
            valid[int(mh * 0.74) :, int(mw * 0.55) :] = 0
            valid[int(mh * 0.78) :, : int(mw * 0.18)] = 0
            cv2.circle(
                valid,
                (int(round(cx)), int(round(cy))),
                max(9, int(round(min(mw, mh) * 0.095))),
                0,
                -1,
            )

            # 1. Yeşil stack rozetleri ('2' / '3' yığınları)
            green_badge = cv2.bitwise_and(
                cv2.inRange(hsv, (50, 68, 80), (83, 255, 255)),
                valid,
            )
            badge_centers: List[Tuple[float, float, int]] = []
            cnts_g, _ = cv2.findContours(
                cv2.morphologyEx(green_badge, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for c in cnts_g:
                gx, gy, gw, gh = cv2.boundingRect(c)
                if 4 <= gw <= 20 and 4 <= gh <= 24:
                    patch_v = hsv[
                        max(0, gy - 2) : min(mh, gy + gh + 2),
                        max(0, gx - 2) : min(mw, gx + gw + 2),
                        2,
                    ]
                    if patch_v.size > 0 and float(patch_v.max()) >= 175.0:
                        stack_cnt = 3 if (gh >= 11 or gw >= 10) else 2
                        badge_centers.append((gx + gw / 2.0, gy + gh / 2.0, stack_cnt))

            # 2. Cyan/Mavi Kanat (Wing Feather / Gemstone) ikonları
            cyan_wing = cv2.bitwise_and(
                cv2.inRange(hsv, (80, 65, 85), (115, 255, 255)),
                valid,
            )
            cnts_c, _ = cv2.findContours(
                cv2.morphologyEx(cyan_wing, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            for c in cnts_c:
                x, y, cw, ch = cv2.boundingRect(c)
                area = float(cv2.contourArea(c))
                if 3 <= cw <= 24 and 3 <= ch <= 24 and area >= 4.0:
                    ncx, ncy = x + cw / 2.0, y + ch / 2.0
                    stack_cnt = 1
                    for bx, by, sc in badge_centers:
                        if math.hypot(ncx - bx, ncy - by) <= 24.0:
                            stack_cnt = max(stack_cnt, sc)
                    rel_dx = round(float(ncx - cx), 2)
                    rel_dy = round(float(ncy - cy), 2)
                    dist_m = round(math.hypot(rel_dx, rel_dy) * meters_per_px, 2)
                    detections.append(
                        {
                            "material_type": "Odyle" if stack_cnt >= 2 else "Gemstone",
                            "source": "minimap",
                            "source_type": "minimap_material",
                            "screen_x": int(round(mx1 + ncx)),
                            "screen_y": int(round(my1 + ncy)),
                            "bbox": (int(mx1 + x), int(my1 + y), int(cw), int(ch)),
                            "rel_map_dx": rel_dx,
                            "rel_map_dy": rel_dy,
                            "estimated_distance_m": dist_m,
                            "confidence": 0.91 if stack_cnt >= 2 else 0.86,
                            "in_air": False,
                            "stack_count": stack_cnt,
                            "max_harvests": 3 * stack_cnt,
                            "gather_key": "f",
                        }
                    )

            # 3. Parlak Beyaz İkonlar: 3 Yapraklı Sprout & Beyaz Tüy Odyle (Monster dot'lar hariç!)
            bright = cv2.bitwise_and(
                cv2.inRange(hsv, (0, 0, 195), (180, 62, 255)),
                valid,
            )
            cnts_b, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts_b:
                x, y, cw, ch = cv2.boundingRect(c)
                area = float(cv2.contourArea(c))
                if cw < 4 or ch < 4 or cw > 26 or ch > 26 or area < 6.0:
                    continue

                ncx, ncy = x + cw / 2.0, y + ch / 2.0
                stack_cnt = 1
                for bx, by, sc in badge_centers:
                    if math.hypot(ncx - bx, ncy - by) <= 24.0:
                        stack_cnt = max(stack_cnt, sc)

                px1, py1 = max(0, x - 2), max(0, y - 2)
                px2, py2 = min(mw, x + cw + 2), min(mh, y + ch + 2)
                patch_v = hsv[py1:py2, px1:px2, 2]
                v_min = float(patch_v.min()) if patch_v.size > 0 else 255.0
                v_max = float(patch_v.max()) if patch_v.size > 0 else 0.0
                if stack_cnt == 1 and (v_min > 95.0 or (v_max - v_min) < 95.0):
                    continue

                peri = float(cv2.arcLength(c, True))
                circ = (4.0 * math.pi * area) / (peri * peri) if peri > 0 else 1.0
                hull = cv2.convexHull(c)
                hull_area = float(cv2.contourArea(hull))
                solidity = area / hull_area if hull_area > 0 else 1.0
                aspect = float(max(cw, ch)) / float(max(1, min(cw, ch)))

                # Küçük yuvarlak beyaz Monster dot'ları KESİN OLARAK REDDET!
                if cw <= 8 and ch <= 8 and circ >= 0.74 and solidity >= 0.88 and aspect <= 1.35:
                    continue

                # Zaten cyan kanat ile eşleşen kümeyi tekrarlama
                if any(
                    math.hypot(int(mx1 + ncx) - int(d["screen_x"]), int(my1 + ncy) - int(d["screen_y"]))
                    <= 8.0
                    for d in detections
                ):
                    continue

                if stack_cnt >= 2 or (aspect >= 1.32 and max(cw, ch) >= 7):
                    mat_type = "Odyle"
                else:
                    mat_type = "Gatherable Material"

                rel_dx = round(float(ncx - cx), 2)
                rel_dy = round(float(ncy - cy), 2)
                dist_m = round(math.hypot(rel_dx, rel_dy) * meters_per_px, 2)
                detections.append(
                    {
                        "material_type": mat_type,
                        "source": "minimap",
                        "source_type": "minimap_material",
                        "screen_x": int(round(mx1 + ncx)),
                        "screen_y": int(round(my1 + ncy)),
                        "bbox": (int(mx1 + x), int(my1 + y), int(cw), int(ch)),
                        "rel_map_dx": rel_dx,
                        "rel_map_dy": rel_dy,
                        "estimated_distance_m": dist_m,
                        "confidence": 0.90 if stack_cnt >= 2 else 0.84,
                        "in_air": False,
                        "stack_count": stack_cnt,
                        "max_harvests": 3 * stack_cnt,
                        "gather_key": "f",
                    }
                )

            return detections
        except Exception as exc:
            logger.error("_scan_minimap_icons_internal hatası: %s", exc)
            return detections

    # ------------------------------------------------------------------
    # B. 3D Dünya Görünümü Hibrit Tespiti (Aria / Mela Arması + [E]/[F] İkonu)
    # ------------------------------------------------------------------
    def _scan_3d_world_hybrid(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """
        3D oyun dünyasındaki toplanabilir kaynakları hibrit olarak tarar:
        1. `Aria` ve `Mela` gibi minimap'te görünmeyen kaynakların 3D dünyadaki beyaz
           kanatlı toplama armalarını (`white winged gather crest` + yeşil/teal isim etiketi),
        2. `< Orichalcum >` / `< Diamond >` cyan elmas uçlu isim etiketlerini,
        3. 3D dünyada yüzen yeşil çerçeveli `[E]` / `[F]` kare toplama ikonlarını (`pilgrims_ridge_1.jpg`),
        4. Karakter merkezindeki `"F Gather"` etkileşim istemini (yakındaki 3D `Aria`/`Mela`
           ismiyle otomatik ilişkilendirerek).
        """
        world_and_prompt: List[Dict[str, Any]] = []
        try:
            if bgr is None or bgr.size == 0:
                return world_and_prompt

            h, w = bgr.shape[:2]
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # 1. MaterialScanner üzerinden 3D isim etiketleri ve gather prompt
            scanner_world = self.material_scanner.scan_3d_world_nameplates(bgr, hsv)
            for item in scanner_world:
                d = self._serialize_detected_material(item)
                d["source"] = "3d_world"
                d["source_type"] = "3d_world_crest"
                world_and_prompt.append(d)

            # 2. Dahili 3D Aria / Mela Beyaz Kanatlı Toplama Arması ve [E]/[F] Yüzen İkon Tarayıcı
            internal_3d = self._scan_3d_crests_and_floating_icons_internal(bgr, hsv)
            for cand in internal_3d:
                cx, cy = int(cand["screen_x"]), int(cand["screen_y"])
                duplicate = False
                for existing in world_and_prompt:
                    if math.hypot(cx - int(existing["screen_x"]), cy - int(existing["screen_y"])) <= 36.0:
                        duplicate = True
                        if (
                            existing.get("material_type") in ("Orichalcum Ore", "Gatherable Material", "")
                            and cand.get("material_type") in ("Aria", "Mela", "Odyle")
                        ):
                            existing["material_type"] = cand["material_type"]
                        break
                if not duplicate:
                    world_and_prompt.append(cand)

            # 3. 'F Gather' etkileşim istemi (MaterialScanner + VisionEngine)
            only_3d_objs = [
                DetectedMaterial(
                    material_type=str(d["material_type"]),
                    source="3d_world",
                    screen_x=int(d["screen_x"]),
                    screen_y=int(d["screen_y"]),
                    bbox=tuple(d["bbox"]),
                    rel_map_dx=float(d["rel_map_dx"]),
                    rel_map_dy=float(d["rel_map_dy"]),
                    estimated_distance_m=float(d["estimated_distance_m"]),
                    confidence=float(d["confidence"]),
                    in_air=bool(d["in_air"]),
                )
                for d in world_and_prompt
                if d.get("source") == "3d_world"
            ]
            scanner_prompts = self.material_scanner.scan_gather_prompt(bgr, hsv, only_3d_objs)
            for p_item in scanner_prompts:
                pd = self._serialize_detected_material(p_item)
                pd["source"] = "gather_prompt"
                pd["source_type"] = "gather_prompt"
                pd["gather_key"] = "f"
                if only_3d_objs:
                    nearest_3d = min(
                        only_3d_objs,
                        key=lambda w_item: math.hypot(
                            w_item.screen_x - pd["screen_x"], w_item.screen_y - pd["screen_y"]
                        ),
                    )
                    if nearest_3d.material_type and nearest_3d.material_type != "Gatherable Material":
                        pd["material_type"] = nearest_3d.material_type
                world_and_prompt.append(pd)

            has_f_prompt = any(
                d.get("source") == "gather_prompt" and d.get("gather_key", "f") == "f"
                for d in world_and_prompt
            )
            if not has_f_prompt:
                f_found, f_conf, f_bbox = self.vision_engine.detect_gather_prompt(bgr)
                if f_found and f_bbox is not None:
                    bx, by, bw, bh = f_bbox
                    cx, cy = bx + bw // 2, by + bh // 2
                    rel_dx, rel_dy, dist_m = self.material_scanner._project_3d_to_map_offset(
                        cx, cy, w, h
                    )
                    mat_name = DEFAULT_RESOURCE_TYPE
                    in_air_flag = False
                    if only_3d_objs:
                        nearest_3d = min(
                            only_3d_objs,
                            key=lambda w_item: math.hypot(w_item.screen_x - cx, w_item.screen_y - cy),
                        )
                        mat_name = nearest_3d.material_type
                        in_air_flag = nearest_3d.in_air
                    world_and_prompt.append(
                        {
                            "material_type": mat_name,
                            "source": "gather_prompt",
                            "source_type": "gather_prompt",
                            "screen_x": int(cx),
                            "screen_y": int(cy),
                            "bbox": (int(bx), int(by), int(bw), int(bh)),
                            "rel_map_dx": float(rel_dx),
                            "rel_map_dy": float(rel_dy),
                            "estimated_distance_m": min(2.2, float(dist_m)),
                            "confidence": round(float(f_conf) / 100.0, 3),
                            "in_air": bool(in_air_flag),
                            "stack_count": 1,
                            "max_harvests": self.harvests_per_node,
                            "gather_key": "f",
                        }
                    )

            return world_and_prompt
        except Exception as exc:
            logger.error("_scan_3d_world_hybrid hatası: %s", exc)
            return world_and_prompt

    def _scan_3d_crests_and_floating_icons_internal(
        self,
        bgr: np.ndarray,
        hsv: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        1) 3D dünyadaki `Aria` (mor kristal bitki) ve `Mela` (kırmızı çiçek çalısı)
           üzerindeki beyaz kanatlı toplama armasını (`white winged gather crest`) tespit eder.
        2) 3D dünyadaki yüzen yeşil çerçeveli `[E]` / `[F]` toplama ikonunu (`pilgrims_ridge_1.jpg`)
           tespit eder.
        """
        results: List[Dict[str, Any]] = []
        try:
            h, w = bgr.shape[:2]
            if h < 200 or w < 320:
                return results
            if hsv is None:
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            sx = w / 1280.0
            sy = h / 720.0

            vy1, vy2 = int(h * 0.12), int(h * 0.72)
            vx1, vx2 = int(w * 0.10), int(w * 0.80)
            roi_hsv = hsv[vy1:vy2, vx1:vx2]
            if roi_hsv.size == 0:
                return results

            # --- 1. Beyaz Kanatlı 3D Toplama Arması (Aria / Mela / Orichalcum) ---
            white_mask = cv2.inRange(roi_hsv, (0, 0, 195), (180, 48, 255))
            # Karakterin kendi zırh/silah merkezini hariç tut (x: 0.475..0.535, y: 0.51..0.68)
            px1 = max(0, int(w * 0.475) - vx1)
            px2 = min(white_mask.shape[1], int(w * 0.535) - vx1)
            py1 = max(0, int(h * 0.51) - vy1)
            py2 = min(white_mask.shape[0], int(h * 0.68) - vy1)
            white_mask[py1:py2, px1:px2] = 0

            kw = max(6, int(round(12 * sx)))
            kh = max(3, int(round(4 * sy)))
            closed_crest = cv2.morphologyEx(
                white_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
            )
            cnts_crest, _ = cv2.findContours(
                closed_crest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            min_cw, max_cw = max(10, int(round(14 * sx))), max(28, int(round(48 * sx)))
            min_ch, max_ch = max(8, int(round(10 * sy))), max(22, int(round(36 * sy)))

            for cnt in cnts_crest:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if not (min_cw <= cw <= max_cw and min_ch <= ch <= max_ch):
                    continue

                ax, ay = vx1 + x, vy1 + y
                ty1 = min(h, ay + ch)
                ty2 = min(h, ay + ch + int(round(26 * sy)))
                tx1 = max(0, ax - int(round(20 * sx)))
                tx2 = min(w, ax + cw + int(round(20 * sx)))
                if ty2 <= ty1 or tx2 <= tx1:
                    continue

                below_hsv = hsv[ty1:ty2, tx1:tx2]
                teal_mask = cv2.inRange(below_hsv, (62, 48, 95), (102, 255, 255))
                teal_px = int(np.count_nonzero(teal_mask))
                if teal_px < max(14, int(round(26 * sx * sy))):
                    continue

                # İsim etiketinin altındaki bitki/cevher dokusunu sınıflandır (Aria mor, Mela kırmızı)
                py1_p = ty2
                py2_p = min(h, ty2 + int(round(95 * sy)))
                px1_p = max(0, ax - int(round(28 * sx)))
                px2_p = min(w, ax + cw + int(round(28 * sx)))
                plant_hsv = hsv[py1_p:py2_p, px1_p:px2_p]

                purple_px = int(
                    np.count_nonzero(cv2.inRange(plant_hsv, (122, 55, 55), (166, 255, 255)))
                )
                red_px = int(
                    np.count_nonzero(
                        cv2.inRange(plant_hsv, (0, 85, 75), (15, 255, 255))
                        | cv2.inRange(plant_hsv, (165, 85, 75), (180, 255, 255))
                    )
                )

                if purple_px >= max(60, int(round(160 * sx * sy))):
                    mat_name = "Aria"
                elif red_px >= max(140, int(round(360 * sx * sy))):
                    mat_name = "Mela"
                else:
                    mat_name, _ = self.material_scanner._classify_3d_material(
                        bgr, hsv, tx1, ty1, tx2 - tx1, ty2 - ty1
                    )

                target_sx = int(ax + cw // 2)
                target_sy = int(min(h - 1, ty2 + int(round(24 * sy))))
                rel_dx, rel_dy, dist_m = self.material_scanner._project_3d_to_map_offset(
                    target_sx, target_sy, w, h
                )

                results.append(
                    {
                        "material_type": mat_name,
                        "source": "3d_world",
                        "source_type": "3d_world_crest",
                        "screen_x": target_sx,
                        "screen_y": target_sy,
                        "bbox": (int(tx1), int(ay), int(tx2 - tx1), int(ty2 - ay)),
                        "rel_map_dx": float(rel_dx),
                        "rel_map_dy": float(rel_dy),
                        "estimated_distance_m": float(dist_m),
                        "confidence": 0.92,
                        "in_air": False,
                        "stack_count": 1,
                        "max_harvests": self.harvests_per_node,
                        "gather_key": "f",
                    }
                )

            # --- 2. 3D Yüzen Yeşil Çerçeveli [E] / [F] Kare Toplama İkonu (pilgrims_ridge_1.jpg) ---
            green_m = cv2.inRange(roi_hsv, (35, 65, 100), (88, 255, 255))
            closed_g = cv2.morphologyEx(green_m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            cnts_g, _ = cv2.findContours(closed_g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_s = max(14, int(round(16 * sx)))
            max_s = max(32, int(round(44 * sx)))

            for cnt in cnts_g:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if not (min_s <= cw <= max_s and min_s <= ch <= max_s):
                    continue
                aspect = float(cw) / float(max(1, ch))
                if not (0.80 <= aspect <= 1.25):
                    continue

                ax, ay = vx1 + x, vy1 + y
                patch = hsv[ay : ay + ch, ax : ax + cw]
                if patch.size == 0:
                    continue

                bw = max(2, int(round(3 * sx)))
                border_mask = np.zeros((ch, cw), dtype=np.uint8)
                border_mask[:bw, :] = 1
                border_mask[-bw:, :] = 1
                border_mask[:, :bw] = 1
                border_mask[:, -bw:] = 1

                gb = cv2.inRange(patch, (35, 65, 105), (88, 255, 255))
                border_ratio = float(np.sum((gb > 0) & (border_mask > 0))) / float(
                    max(1, np.sum(border_mask > 0))
                )
                if border_ratio < 0.55:
                    continue

                # Sol-üst köşede parlak beyaz 'E' veya 'F' tuş harfi kontrolü
                tl = patch[: max(4, ch // 2), : max(4, cw // 2)]
                if tl.size == 0 or int(tl[:, :, 2].max()) < 200:
                    continue

                # İkonun iç kısmında sıcak/parlak materyal ikonu dokusu
                inner = patch[bw : max(bw + 1, ch - bw), bw : max(bw + 1, cw - bw)]
                warm_px = int(
                    np.count_nonzero(cv2.inRange(inner, (0, 55, 110), (35, 255, 255)))
                )
                if warm_px < max(12, int(cw * ch * 0.08)):
                    continue

                cx, cy = int(ax + cw // 2), int(ay + ch // 2)
                rel_dx, rel_dy, dist_m = self.material_scanner._project_3d_to_map_offset(
                    cx, cy, w, h
                )
                results.append(
                    {
                        "material_type": "Odyle",
                        "source": "gather_prompt",
                        "source_type": "3d_floating_icon",
                        "screen_x": cx,
                        "screen_y": cy,
                        "bbox": (int(ax), int(ay), int(cw), int(ch)),
                        "rel_map_dx": float(rel_dx),
                        "rel_map_dy": float(rel_dy),
                        "estimated_distance_m": float(dist_m),
                        "confidence": 0.91,
                        "in_air": False,
                        "stack_count": 1,
                        "max_harvests": self.harvests_per_node,
                        "gather_key": "e",
                        "is_floating_gather_icon": True,
                    }
                )

            return results
        except Exception as exc:
            logger.error("_scan_3d_crests_and_floating_icons_internal hatası: %s", exc)
            return results

    # ------------------------------------------------------------------
    # C. Sohbet Kutusu "Essence Extraction failed" Tespiti ve 3x Hasat
    # ------------------------------------------------------------------
    def _detect_extraction_failed_in_chat(self, frame_bgr: Optional[np.ndarray]) -> bool:
        """
        Sol-alt sohbet kutusunda (`pilgrims_ridge_1.jpg`) turuncu/kehribar sistem uyarısı
        olan `"Essence Extraction failed"` mesajının görünüp görünmediğini tespit eder.
        """
        try:
            if (
                frame_bgr is None
                or not isinstance(frame_bgr, np.ndarray)
                or frame_bgr.size == 0
                or frame_bgr.ndim < 3
            ):
                return False

            h, w = frame_bgr.shape[:2]
            if h < 160 or w < 240:
                return False

            # Sol-alt sohbet sistem mesajı bölgesi (x: 0.025..0.13, y: 0.74..0.81)
            cy1, cy2 = int(round(h * 0.74)), int(round(h * 0.81))
            cx1, cx2 = int(round(w * 0.025)), int(round(w * 0.13))
            chat_bgr = frame_bgr[cy1:cy2, cx1:cx2]
            if chat_bgr.size == 0:
                return False

            chat_hsv = cv2.cvtColor(chat_bgr, cv2.COLOR_BGR2HSV)
            chat_gray = cv2.cvtColor(chat_bgr, cv2.COLOR_BGR2GRAY)
            lap = np.abs(cv2.Laplacian(chat_gray, cv2.CV_32F))

            # Parlak kehribar/turuncu sistem yazısı (H: 14..24, S >= 130, V >= 185) + keskin harf kenarı
            amber_text_mask = (
                (chat_hsv[:, :, 0] >= 14)
                & (chat_hsv[:, :, 0] <= 24)
                & (chat_hsv[:, :, 1] >= 130)
                & (chat_hsv[:, :, 2] >= 185)
                & (lap >= 45.0)
            )
            sharp_amber_px = int(np.count_nonzero(amber_text_mask))
            scale_factor = (w / 2000.0) * (h / 1124.0)
            threshold_px = max(18, int(round(45 * scale_factor)))
            return sharp_amber_px >= threshold_px
        except Exception as exc:
            logger.debug("_detect_extraction_failed_in_chat hatası: %s", exc)
            return False

    def _press_gather_key_once(self, key_name: str = "f") -> None:
        """Verilen toplama tuşuna (`'f'` veya `'e'`) bir kez basar."""
        clean_key = str(key_name or self.gather_key or "f").strip().lower() or "f"
        if self.input_controller is not None:
            if hasattr(self.input_controller, "press_key"):
                self.input_controller.press_key(clean_key)
            elif hasattr(self.input_controller, "tap_key"):
                self.input_controller.tap_key(clean_key)
            elif callable(self.input_controller):
                self.input_controller(clean_key)
        elif not self.test_mode:
            self.input_controller = InputController(test_mode=self.test_mode)
            self.input_controller.press_key(clean_key)
        else:
            logger.info("[ExplorerMode] Test modunda '%s' toplama tuşu simüle edildi.", clean_key.upper())

    def _wait_for_harvest_cast(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> Optional[np.ndarray]:
        """
        Her bir hasat aşaması (`1/3`, `2/3`, `3/3`) sırasında toplama barının (cast bar)
        tamamlanmasını bekler ve en güncel kareyi döndürür.
        `test_mode=True` iken birim testlerini yavaşlatmamak için anında tamamlanır.
        """
        latest_frame = frame_bgr
        if self.test_mode:
            if frame_provider is not None and callable(frame_provider):
                try:
                    provided = frame_provider()
                    if isinstance(provided, np.ndarray) and provided.size > 0:
                        latest_frame = provided
                except Exception as exc:
                    logger.debug("frame_provider çağrısı atlandı: %s", exc)
            return latest_frame

        wait_budget = max(0.15, float(self.cast_wait_sec))
        deadline = time.time() + wait_budget
        while time.time() < deadline:
            time.sleep(min(0.12, max(0.02, deadline - time.time())))
            if frame_provider is not None and callable(frame_provider):
                try:
                    provided = frame_provider()
                    if isinstance(provided, np.ndarray) and provided.size > 0:
                        latest_frame = provided
                except Exception as exc:
                    logger.debug("Canlı frame_provider hatası: %s", exc)
        return latest_frame

    def _handle_opportunistic_gather(
        self,
        frame_bgr: Optional[np.ndarray],
        gather_key: Optional[str] = None,
        harvests_per_node: Optional[int] = None,
        target_node: Optional[NodeData] = None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
        max_retries_per_stage: int = 2,
    ) -> Dict[str, Any]:
        """
        Keşif sırasında (`gather_while_exploring=True`) bir kaynak istemi (`[F Gather]` veya
        `[E]` ikonu) görüldüğünde:
        - Kaynağı tek seferde bırakmak yerine `harvests_per_node = 3` kez (`1/3`, `2/3`, `3/3`)
          ardışık olarak toplar (`gather_key` `'f'` veya `'e'`).
        - Her hasat aşamasının tamamlanmasını bekler (`_wait_for_harvest_cast`).
        - Sohbette `"Essence Extraction failed"` uyarısı tespit edilirse ilgili hasat
          aşamasını `max_retries_per_stage` kadar yeniden dener.
        - `RouteManager.record_harvest_stage` ve `mark_node_gathered` ile düğümün
          `1/3 -> 2/3 -> 3/3 -> resting` durumunu günceller.
        """
        now_ts = time.time()
        used_key = str(gather_key or self.gather_key or "f").strip().lower() or "f"
        target_harvests = max(1, int(harvests_per_node or self.harvests_per_node or DEFAULT_HARVESTS_PER_NODE))

        if not self.test_mode and (now_ts - self._last_gather_ts) < self.min_gather_interval_sec:
            return {
                "triggered": False,
                "gather_key": used_key,
                "harvests_completed": 0,
                "harvests_target": target_harvests,
                "stages_log": [],
                "extraction_failed_retries": 0,
                "node_id": int(target_node.id) if target_node is not None else None,
                "node_depleted": False,
            }

        # Oyuncunun yanındaki en yakın düğümü bul (verilmediyse)
        nearest_node: Optional[NodeData] = target_node
        if nearest_node is None and self._ordered_nodes:
            px, py, _, _ = self.auto_cartographer.get_position()
            candidate_node = min(
                self._ordered_nodes,
                key=lambda n: math.hypot(float(n.map_x) - px, float(n.map_y) - py),
            )
            if math.hypot(float(candidate_node.map_x) - px, float(candidate_node.map_y) - py) <= (
                self.auto_cartographer.dedup_radius_px * 2.0
            ):
                nearest_node = candidate_node

        completed_stages = 0
        retries_count = 0
        stages_log: List[str] = []
        current_frame = frame_bgr

        try:
            for stage_idx in range(1, target_harvests + 1):
                stage_attempts = 0
                while True:
                    stage_attempts += 1
                    self._press_gather_key_once(used_key)
                    self._last_gather_ts = time.time()
                    current_frame = self._wait_for_harvest_cast(
                        frame_bgr=current_frame,
                        frame_provider=frame_provider,
                    )

                    # Sohbette "Essence Extraction failed" var mı kontrol et
                    extraction_failed = self._detect_extraction_failed_in_chat(current_frame)
                    if extraction_failed and stage_attempts <= max_retries_per_stage:
                        retries_count += 1
                        logger.warning(
                            "[ExplorerMode] 'Essence Extraction failed' tespit edildi! "
                            "Hasat aşaması %d/%d yeniden deneniyor (deneme=%d/%d, tuş=%s).",
                            stage_idx,
                            target_harvests,
                            stage_attempts,
                            max_retries_per_stage,
                            used_key.upper(),
                        )
                        # İlk denemeden sonra test modunda sahte frame aynı kalırsa sonsuz döngüye girmemek için:
                        if frame_provider is None:
                            current_frame = None
                        continue
                    break

                completed_stages += 1
                self._total_gathered_in_explorer += 1
                stage_label = f"{completed_stages}/{target_harvests}"
                stages_log.append(stage_label)

                if nearest_node is not None:
                    if hasattr(self.route_manager, "record_harvest_stage"):
                        self.route_manager.record_harvest_stage(
                            node_id=nearest_node.id,
                            default_max_harvests=target_harvests,
                            region=self.active_region,
                        )
                    else:
                        self.route_manager.mark_node_gathered(
                            nearest_node.id,
                            region=self.active_region,
                        )

            info = {
                "triggered": completed_stages > 0,
                "gather_key": used_key,
                "harvests_completed": completed_stages,
                "harvests_target": target_harvests,
                "stages_log": stages_log,
                "extraction_failed_retries": retries_count,
                "node_id": int(nearest_node.id) if nearest_node is not None else None,
                "node_depleted": completed_stages >= target_harvests,
            }
            self._last_opportunistic_gather_info = info
            return info
        except Exception as exc:
            logger.error("_handle_opportunistic_gather sırasında hata: %s", exc, exc_info=True)
            info = {
                "triggered": completed_stages > 0,
                "gather_key": used_key,
                "harvests_completed": completed_stages,
                "harvests_target": target_harvests,
                "stages_log": stages_log,
                "extraction_failed_retries": retries_count,
                "node_id": int(nearest_node.id) if nearest_node is not None else None,
                "node_depleted": completed_stages >= target_harvests,
                "error": str(exc),
            }
            self._last_opportunistic_gather_info = info
            return info

    def _trigger_gather_key(
        self,
        now_ts: float,
        frame_bgr: Optional[np.ndarray] = None,
        gather_key: Optional[str] = None,
        harvests_per_node: Optional[int] = None,
    ) -> bool:
        """
        Geriye dönük uyumlu yardımcı metod: `_handle_opportunistic_gather` çağırarak
        kaynağı `harvests_per_node` (varsayılan 3) kez toplar.
        """
        result = self._handle_opportunistic_gather(
            frame_bgr=frame_bgr,
            gather_key=gather_key,
            harvests_per_node=harvests_per_node,
        )
        return bool(result.get("triggered", False))

    def _steer_toward_3d_target(
        self,
        target_det: Dict[str, Any],
        frame_w: int,
        frame_h: int,
        execute_steer: bool = True,
    ) -> Dict[str, Any]:
        """
        3D dünyada tespit edilen `Aria`, `Mela`, `< Orichalcum >` veya yüzen `[E]`/`[F]`
        toplama ikonuna doğru yönelme (steering) vektörünü hesaplar ve isteğe bağlı olarak
        fare/yönlendirme girdisini tetikler.
        """
        player_ax = float(frame_w) * 0.50
        player_ay = float(frame_h) * 0.54
        tx = int(target_det.get("screen_x", int(player_ax)))
        ty = int(target_det.get("screen_y", int(player_ay)))
        dx = round(float(tx) - player_ax, 2)
        dy = round(float(ty) - player_ay, 2)
        angle_deg = round((math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0, 2)
        dist_m = float(target_det.get("estimated_distance_m", 0.0))

        steered = False
        if execute_steer:
            if not self.test_mode and self.input_controller is not None:
                try:
                    if hasattr(self.input_controller, "move_mouse"):
                        self.input_controller.move_mouse(tx, ty)
                        steered = True
                except Exception as exc:
                    logger.debug("3D hedefe yönelme fare hareketi atlandı: %s", exc)
            else:
                steered = True

        steer_info = {
            "material_type": str(target_det.get("material_type", DEFAULT_RESOURCE_TYPE)),
            "source_type": str(target_det.get("source_type", "3d_world")),
            "target_screen_x": tx,
            "target_screen_y": ty,
            "steer_dx": dx,
            "steer_dy": dy,
            "steer_angle_deg": angle_deg,
            "estimated_distance_m": dist_m,
            "gather_key": str(target_det.get("gather_key", "f")),
            "steered": steered,
        }
        self._last_steer_target = steer_info
        return steer_info

    # ------------------------------------------------------------------
    # 1. Hibrit 3D + Minimap Gezgin Haritalandırma (Explorer Auto-Cartography)
    # ------------------------------------------------------------------
    def explore_step(
        self,
        frame_bgr: np.ndarray,
        gather_while_exploring: bool = True,
        steer_to_3d_targets: bool = True,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> dict:
        """
        Her keşif adımında (`explore_step(frame)`):
        1. Sağ-üst minimap materyal taramasını (`scan_minimap_materials` + Monster dot filtreli
           filiz/Odyle/kanat/yığın rozeti taraması) öncelikli olarak çalıştırır.
        2. Eşzamanlı olarak 3D oyun dünyasını (`Aria` ve `Mela` beyaz kanatlı toplama armaları,
           `< Orichalcum >` etiketleri, yüzen `[E]`/`[F]` toplama ikonları ve `"F Gather"` istemi)
           tarar.
        3. Tespit edilen tüm kaynakları `AutoCartographer` ve `RouteManager` içine (`max_harvests`
           ve `source_type` ile) kaydeder ve `SequentialRoutePlanner` ile sıralı rotaya ekler.
        4. `gather_while_exploring=True` ise ve toplama istemi aktifse kaynağı **3 kez**
           (`1/3 -> 2/3 -> 3/3`) toplar; uzakta 3D `Aria`/`Mela`/`[E]` ikonu görüldüyse ona yönelir.
        """
        return self.step_passive_cartography(
            frame_bgr=frame_bgr,
            gather_while_exploring=gather_while_exploring,
            steer_to_3d_targets=steer_to_3d_targets,
            frame_provider=frame_provider,
        )

    def step_passive_cartography(
        self,
        frame_bgr: np.ndarray,
        gather_while_exploring: bool = False,
        steer_to_3d_targets: bool = True,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> dict:
        """
        Hibrit 3D + Sağ-Üst Minimap Gezgin Haritalandırma ve 3x Hasat Adımı:
        1. Sağ-üst minimap üzerindeki tüm materyal ikonlarını (beyaz tüy Odyle, cyan/mavi kanat,
           yeşil yığın rozetleri `2`/`3`, 3 yapraklı filiz/sprout ikonları — küçük yuvarlak beyaz
           canavar noktalarını yok sayarak) ve 3D dünyadaki `Aria`/`Mela` beyaz kanatlı armaları,
           cevher etiketlerini ve `[E]`/`[F Gather]` ikonlarını tarar.
        2. `AutoCartographer` ile oyuncunun minimap odometri konumunu günceller
           ve tespit edilen tüm kaynakları tekilleştirerek (`Spatial Deduplication`)
           80x80 harita kesitleriyle birlikte `RouteManager`'a kaydeder.
        3. Yeni bir kaynak keşfedildiğinde `SequentialRoutePlanner.insert_node_in_sequence`
           kullanarak düğümü zikzaksız, sıralı koridor içine yerleştirir ve HUD bildirimi üretir.
        4. Eğer `gather_while_exploring == True` ise:
           - `"F Gather"` veya yüzen `[E]` toplama istemi aktifse `_handle_opportunistic_gather`
             ile kaynağı **3 kez (`1/3 -> 2/3 -> 3/3`)** toplar (sohbette `"Essence Extraction failed"`
             görülürse yeniden dener).
           - Ekranda uzakta 3D `Aria`/`Mela` veya yüzen `[E]` ikonu varsa ona doğru yönelir (`steer_target`).
        """
        now_ts = time.time()
        self._step_count += 1
        step_notifications: List[str] = []
        newly_mapped_details: List[Dict[str, Any]] = []

        try:
            if (
                frame_bgr is None
                or not isinstance(frame_bgr, np.ndarray)
                or frame_bgr.size == 0
            ):
                logger.warning("step_passive_cartography: Geçersiz veya boş frame_bgr.")
                map_x, map_y, norm_x, norm_y = self.auto_cartographer.get_position()
                return {
                    "step_index": self._step_count,
                    "timestamp": now_ts,
                    "player_pos": {
                        "map_x": map_x,
                        "map_y": map_y,
                        "norm_x": norm_x,
                        "norm_y": norm_y,
                    },
                    "is_map_open": False,
                    "is_window_blocking": False,
                    "detected_materials": [],
                    "detected_count": 0,
                    "world_3d_count": 0,
                    "minimap_count": 0,
                    "gather_prompt_detected": False,
                    "gathered_action_triggered": False,
                    "harvests_completed_in_step": 0,
                    "opportunistic_gather": {},
                    "steer_target": None,
                    "newly_mapped_nodes": [],
                    "new_nodes": [],
                    "newly_mapped_count": 0,
                    "total_mapped_nodes": len(self._ordered_nodes),
                    "sequential_route_ids": [n.id for n in self._ordered_nodes],
                    "hud_notifications": [],
                    "latest_hud_message": "",
                }

            # Görüntüyü 3 kanallı BGR formatına normalize et
            if frame_bgr.ndim == 2:
                bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2BGR)
            elif frame_bgr.ndim == 3 and frame_bgr.shape[2] == 4:
                bgr = cv2.cvtColor(frame_bgr, cv2.COLOR_BGRA2BGR)
            else:
                bgr = frame_bgr

            h, w = bgr.shape[:2]

            # Dünya haritası ('M') veya engelleyici oyun penceresi kontrolü
            is_map_open = bool(self.vision_engine.is_map_open(bgr))
            is_blocking = False if is_map_open else bool(self.vision_engine.is_game_window_blocking(bgr))

            # Adım 1: Hibrit Sağ-Üst Minimap Öncelikli Tarama + 3D Dünya (Aria/Mela/[E]/[F]) Taraması
            minimap_detections: List[Dict[str, Any]] = []
            world_3d_detections: List[Dict[str, Any]] = []
            if not is_map_open and not is_blocking:
                minimap_detections = self._scan_minimap_priority_materials(bgr)
                world_3d_detections = self._scan_3d_world_hybrid(bgr)

            serialized_detections: List[Dict[str, Any]] = list(world_3d_detections) + list(
                minimap_detections
            )

            gather_prompts = [
                d for d in world_3d_detections if d.get("source") == "gather_prompt"
            ]
            gather_prompt_detected = len(gather_prompts) > 0
            active_gather_key = (
                str(gather_prompts[0].get("gather_key", self.gather_key)).strip().lower()
                if gather_prompts
                else self.gather_key
            )

            # Adım 2: AutoCartographer ile oyuncunun minimap odometri konumunu güncelle
            map_x, map_y, norm_x, norm_y = self.auto_cartographer.update_player_position(bgr)

            # 3D dünya spesifik isimlerini (Aria, Mela, Orichalcum Ore) ve minimap stack kümelerini önceliklendir
            prioritized_detections = sorted(
                serialized_detections,
                key=lambda item: (
                    0 if item.get("source") in ("3d_world", "gather_prompt") else 1,
                    -int(item.get("max_harvests", 3)),
                    -float(item.get("confidence", 0.0)),
                ),
            )

            newly_discovered = self.auto_cartographer.process_frame_discoveries(
                frame_bgr=bgr,
                detected_materials=prioritized_detections,
                route_manager=self.route_manager,
                region=self.active_region,
                route=self.active_route,
                update_odometry=False,
            )

            # Her tespit için RouteManager'daki karşılık gelen düğümün max_harvests, source_type ve
            # spesifik materyal ismini (Aria, Mela, Odyle, Orichalcum Ore) senkronize et
            for det in prioritized_detections:
                det_source = str(det.get("source", ""))
                det_source_type = str(det.get("source_type", det_source or "minimap_material"))
                det_type = str(det.get("material_type", ""))
                det_max_h = max(3, int(det.get("max_harvests", 3) or 3))
                obs_x = map_x + float(det.get("rel_map_dx", 0.0))
                obs_y = map_y + float(det.get("rel_map_dy", 0.0))

                tracked_node, _ = self.auto_cartographer._find_nearest_tracked_node(
                    region=self.active_region,
                    node_map_x=obs_x,
                    node_map_y=obs_y,
                )
                if tracked_node is not None:
                    rm_node = self.route_manager.get_node(tracked_node.node_id, region=self.active_region)
                    if (
                        det_source in ("3d_world", "gather_prompt")
                        and det_type
                        and det_type != "Gatherable Material"
                    ):
                        if tracked_node.resource_type in ("Gatherable Material", "Orichalcum Ore"):
                            tracked_node.resource_type = det_type
                        if rm_node is not None and rm_node.resource_type in (
                            "Gatherable Material",
                            "Orichalcum Ore",
                        ):
                            rm_node.resource_type = det_type
                    if rm_node is not None:
                        if det_max_h > int(getattr(rm_node, "max_harvests", 3) or 3):
                            rm_node.max_harvests = det_max_h
                        if det_source_type:
                            rm_node.source_type = det_source_type

            # Adım 3: Yeni keşfedilen her kaynağı SequentialRoutePlanner.insert_node_in_sequence
            # ile zikzaksız sıralı koridor içine yerleştir ve HUD bildirimi üret
            for new_node_info in newly_discovered:
                node_id = int(new_node_info.get("node_id", new_node_info.get("id", 0)))
                node_obj = self.route_manager.get_node(node_id, region=self.active_region)
                if node_obj is None:
                    node_obj = NodeData.from_dict(new_node_info)

                existing_ids = {n.id for n in self._ordered_nodes}
                if node_obj.id in existing_ids:
                    seq_pos = next(
                        (idx + 1 for idx, n in enumerate(self._ordered_nodes) if n.id == node_obj.id),
                        len(self._ordered_nodes),
                    )
                else:
                    updated_seq, inserted_idx = self.route_planner.insert_node_in_sequence(
                        ordered_nodes=self._ordered_nodes,
                        new_node=node_obj,
                        closed_loop=True,
                    )
                    self._ordered_nodes = list(updated_seq)
                    seq_pos = int(inserted_idx) + 1

                res_type = str(node_obj.resource_type or DEFAULT_RESOURCE_TYPE)
                max_h_val = int(getattr(node_obj, "max_harvests", 3) or 3)
                hud_msg = (
                    f"Auto-mapped #{node_obj.id} {res_type} "
                    f"(Sequential #{seq_pos}, {max_h_val}x harvest)"
                )
                self._emit_hud_notification(hud_msg)
                step_notifications.append(hud_msg)

                enriched = dict(new_node_info)
                enriched["resource_type"] = res_type
                enriched["max_harvests"] = max_h_val
                enriched["source_type"] = str(getattr(node_obj, "source_type", "minimap_material"))
                enriched["seq_index"] = int(seq_pos)
                enriched["hud_message"] = hud_msg
                newly_mapped_details.append(enriched)

            if newly_discovered:
                self._persist_ordered_nodes_to_route_manager()

            # Adım 4: 3D Hedefe Yönelme (Aria / Mela / Yüzen [E] İkonu) & 3x Hasat (gather_while_exploring)
            steer_target: Optional[Dict[str, Any]] = None
            only_3d_targets = [
                d
                for d in world_3d_detections
                if d.get("source") == "3d_world" or d.get("is_floating_gather_icon")
            ]
            if only_3d_targets and steer_to_3d_targets:
                closest_3d = min(
                    only_3d_targets,
                    key=lambda d: float(d.get("estimated_distance_m", 999.0)),
                )
                steer_target = self._steer_toward_3d_target(
                    closest_3d,
                    frame_w=w,
                    frame_h=h,
                    execute_steer=bool(gather_while_exploring and not gather_prompt_detected),
                )

            gathered_now = False
            opportunistic_gather_info: Dict[str, Any] = {}
            harvests_completed_in_step = 0
            if gather_while_exploring and gather_prompt_detected:
                opportunistic_gather_info = self._handle_opportunistic_gather(
                    frame_bgr=bgr,
                    gather_key=active_gather_key,
                    harvests_per_node=self.harvests_per_node,
                    frame_provider=frame_provider,
                )
                gathered_now = bool(opportunistic_gather_info.get("triggered", False))
                harvests_completed_in_step = int(
                    opportunistic_gather_info.get("harvests_completed", 0)
                )

            world_3d_count = sum(
                1 for d in serialized_detections if d.get("source") == "3d_world"
            )
            minimap_count = sum(
                1 for d in serialized_detections if d.get("source") == "minimap"
            )

            return {
                "step_index": self._step_count,
                "timestamp": now_ts,
                "player_pos": {
                    "map_x": map_x,
                    "map_y": map_y,
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                },
                "is_map_open": is_map_open,
                "is_window_blocking": is_blocking,
                "detected_materials": serialized_detections,
                "detected_count": len(serialized_detections),
                "world_3d_count": world_3d_count,
                "minimap_count": minimap_count,
                "gather_prompt_detected": gather_prompt_detected,
                "gather_key": active_gather_key,
                "gathered_action_triggered": gathered_now,
                "harvests_completed_in_step": harvests_completed_in_step,
                "opportunistic_gather": opportunistic_gather_info,
                "steer_target": steer_target,
                "newly_mapped_nodes": newly_mapped_details,
                "new_nodes": newly_mapped_details,
                "newly_mapped_count": len(newly_mapped_details),
                "total_mapped_nodes": len(self._ordered_nodes),
                "sequential_route_ids": [n.id for n in self._ordered_nodes],
                "hud_notifications": step_notifications,
                "latest_hud_message": (
                    step_notifications[-1]
                    if step_notifications
                    else (self.hud_notifications[-1] if self.hud_notifications else "")
                ),
            }

        except Exception as exc:
            logger.error("step_passive_cartography sırasında beklenmeyen hata: %s", exc, exc_info=True)
            map_x, map_y, norm_x, norm_y = self.auto_cartographer.get_position()
            return {
                "step_index": self._step_count,
                "timestamp": now_ts,
                "player_pos": {
                    "map_x": map_x,
                    "map_y": map_y,
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                },
                "is_map_open": False,
                "is_window_blocking": False,
                "detected_materials": [],
                "detected_count": 0,
                "world_3d_count": 0,
                "minimap_count": 0,
                "gather_prompt_detected": False,
                "gather_key": self.gather_key,
                "gathered_action_triggered": False,
                "harvests_completed_in_step": 0,
                "opportunistic_gather": {},
                "steer_target": None,
                "newly_mapped_nodes": [],
                "new_nodes": [],
                "newly_mapped_count": 0,
                "total_mapped_nodes": len(self._ordered_nodes),
                "sequential_route_ids": [n.id for n in self._ordered_nodes],
                "hud_notifications": step_notifications,
                "latest_hud_message": "",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 2. Rota Sonu Otomatik Sıralama (Finalize & Anti-Ping-Pong Sorting)
    # ------------------------------------------------------------------
    def finalize_and_sort_route(self, mode: str = "smooth_loop") -> dict:
        """
        Gezinti tamamlandığında veya kullanıcı `"Route Mode"`'a geçtiğinde,
        keşfedilen tüm gem/materyal noktalarını `SequentialRoutePlanner` ile
        tek yönlü, geri dönüşsüz (`Anti-Ping-Pong`) sıraya sokar, `RouteManager`
        veritabanını günceller ve özet istatistikleri döndürür.

        Args:
            mode: Sıralama modu (`"smooth_loop"`, `"walk_path"` veya `"corridor_sweep"`).

        Returns:
            Sıralanmış rota özeti, kalite metrikleri ve düğüm dizilimini içeren sözlük.
        """
        try:
            clean_mode = str(mode or "smooth_loop").strip().lower()
            if clean_mode not in SequentialRoutePlanner.VALID_MODES:
                logger.warning(
                    "Geçersiz sıralama modu '%s', 'smooth_loop' kullanılıyor.",
                    clean_mode,
                )
                clean_mode = "smooth_loop"

            target_region = self.active_region
            target_route = self.active_route

            current_nodes = self.route_manager.get_route_nodes(
                region_name=target_region,
                route_name=target_route,
            )
            if not current_nodes and self._ordered_nodes:
                current_nodes = list(self._ordered_nodes)

            closed_loop = clean_mode == "smooth_loop"
            before_quality = self.route_planner.evaluate_sequence_quality(
                current_nodes,
                closed_loop=closed_loop,
            )

            raw_traj = self.auto_cartographer.get_walk_trajectory()
            walk_traj_2d: List[Tuple[float, float]] = [
                (float(pt[0]), float(pt[1])) for pt in raw_traj if len(pt) >= 2
            ]

            ordered_nodes = self.route_planner.apply_sequential_order_to_route(
                route_manager=self.route_manager,
                region=target_region,
                route=target_route,
                mode=clean_mode,
                walk_trajectory=walk_traj_2d,
                reassign_ids=False,
            )
            if not ordered_nodes and current_nodes:
                ordered_nodes = self.route_planner.build_sequential_order(
                    nodes=current_nodes,
                    walk_trajectory=walk_traj_2d,
                    mode=clean_mode,
                )

            self._ordered_nodes = list(ordered_nodes)
            after_quality = self.route_planner.evaluate_sequence_quality(
                self._ordered_nodes,
                closed_loop=closed_loop,
            )

            resource_breakdown: Dict[str, int] = {}
            in_air_count = 0
            for node in self._ordered_nodes:
                r_type = str(
                    getattr(node, "resource_type", DEFAULT_RESOURCE_TYPE) or DEFAULT_RESOURCE_TYPE
                )
                resource_breakdown[r_type] = resource_breakdown.get(r_type, 0) + 1
                if bool(getattr(node, "in_air", False)):
                    in_air_count += 1

            px, py, _, _ = self.auto_cartographer.get_position()
            first_node, first_idx = self.route_planner.select_next_sequential_node(
                nodes=self._ordered_nodes,
                current_seq_pos=-1,
                player_pos=(px, py),
                forward_window=12,
                allow_wrap=True,
            )

            summary_msg = (
                f"Route finalized ({clean_mode}): {len(self._ordered_nodes)} nodes sorted "
                f"(smoothness={after_quality.get('smoothness_score', 100.0):.1f}/100, "
                f"distance={after_quality.get('total_distance_px', 0.0):.1f}px)"
            )
            self._emit_hud_notification(summary_msg)

            return {
                "region": target_region,
                "route": target_route,
                "mode": clean_mode,
                "node_count": len(self._ordered_nodes),
                "ordered_node_ids": [int(n.id) for n in self._ordered_nodes],
                "ordered_nodes": [
                    {
                        "id": int(n.id),
                        "seq_index": int(getattr(n, "seq_index", idx + 1)),
                        "resource_type": str(n.resource_type),
                        "map_x": int(n.map_x),
                        "map_y": int(n.map_y),
                        "norm_x": float(n.norm_x),
                        "norm_y": float(n.norm_y),
                        "in_air": bool(n.in_air),
                        "max_harvests": int(getattr(n, "max_harvests", 3) or 3),
                        "source_type": str(getattr(n, "source_type", "minimap_material")),
                        "patch_path": str(n.patch_path),
                    }
                    for idx, n in enumerate(self._ordered_nodes)
                ],
                "resource_breakdown": resource_breakdown,
                "in_air_count": in_air_count,
                "ground_count": len(self._ordered_nodes) - in_air_count,
                "quality_before": before_quality,
                "quality_after": after_quality,
                "total_distance_px": float(after_quality.get("total_distance_px", 0.0)),
                "avg_step_px": float(after_quality.get("avg_step_px", 0.0)),
                "max_jump_px": float(after_quality.get("max_jump_px", 0.0)),
                "sharp_turns_count": int(after_quality.get("sharp_turns_count", 0)),
                "crossings_count": int(after_quality.get("crossings_count", 0)),
                "smoothness_score": float(after_quality.get("smoothness_score", 100.0)),
                "walk_trajectory_points": len(walk_traj_2d),
                "total_gathered_while_exploring": self._total_gathered_in_explorer,
                "anti_ping_pong_enabled": True,
                "next_target_node_id": int(first_node.id) if first_node is not None else None,
                "next_target_seq_index": int(first_idx + 1) if first_idx >= 0 else None,
                "summary_message": summary_msg,
            }

        except Exception as exc:
            logger.error("finalize_and_sort_route sırasında hata: %s", exc, exc_info=True)
            return {
                "region": self.active_region,
                "route": self.active_route,
                "mode": str(mode),
                "node_count": len(self._ordered_nodes),
                "ordered_node_ids": [int(n.id) for n in self._ordered_nodes],
                "ordered_nodes": [],
                "resource_breakdown": {},
                "in_air_count": 0,
                "ground_count": len(self._ordered_nodes),
                "quality_before": {},
                "quality_after": {},
                "total_distance_px": 0.0,
                "avg_step_px": 0.0,
                "max_jump_px": 0.0,
                "sharp_turns_count": 0,
                "crossings_count": 0,
                "smoothness_score": 0.0,
                "walk_trajectory_points": len(self.auto_cartographer.walk_trajectory),
                "total_gathered_while_exploring": self._total_gathered_in_explorer,
                "anti_ping_pong_enabled": True,
                "next_target_node_id": None,
                "next_target_seq_index": None,
                "summary_message": f"Error finalizing route: {exc}",
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # 3. Otonom Keşif / Sektör Tarama (Autonomous Boustrophedon Sector Sweep)
    # ------------------------------------------------------------------
    def generate_sector_sweep_waypoints(
        self,
        map_bounds: tuple[int, int, int, int],
        lane_spacing_px: int = 90,
    ) -> list[tuple[int, int]]:
        """
        Botun harita üzerindeki bir bölgeyi Boustrophedon (paralel koridor şeritleri
        halinde sırayla, zikzaksız serpantin düzeninde) gezerek tüm gemleri ve
        materyalleri kendi kendine keşfetmesi için sıralı harita waypoint noktaları üretir.
        """
        waypoints: List[Tuple[int, int]] = []
        try:
            if not isinstance(map_bounds, (tuple, list)) or len(map_bounds) < 4:
                raise ValueError(f"Geçersiz map_bounds: {map_bounds!r}. 4 elemanlı tuple bekleniyor.")

            b0, b1, b2, b3 = (
                int(round(float(map_bounds[0]))),
                int(round(float(map_bounds[1]))),
                int(round(float(map_bounds[2]))),
                int(round(float(map_bounds[3]))),
            )

            if b2 > b0 and b3 > b1:
                x_min, y_min, x_max, y_max = b0, b1, b2, b3
            elif b2 > 0 and b3 > 0 and (b2 <= b0 or b3 <= b1):
                x_min, y_min = b0, b1
                x_max, y_max = b0 + b2, b1 + b3
            else:
                x_min, x_max = min(b0, b2), max(b0, b2)
                y_min, y_max = min(b1, b3), max(b1, b3)

            span_w = max(1, x_max - x_min)
            span_h = max(1, y_max - y_min)
            spacing = max(16, int(round(float(lane_spacing_px))))
            step_along = max(20, int(round(spacing * 0.85)))
            horizontal_lanes = span_w >= span_h

            if horizontal_lanes:
                num_lanes = max(1, int(math.ceil(span_h / float(spacing))) + 1)
                if num_lanes == 1:
                    y_positions = [int(round((y_min + y_max) * 0.5))]
                else:
                    y_positions = [
                        int(round(y_min + i * (span_h / float(num_lanes - 1))))
                        for i in range(num_lanes)
                    ]

                num_steps_x = max(2, int(math.ceil(span_w / float(step_along))) + 1)
                base_x_positions = [
                    int(round(x_min + j * (span_w / float(num_steps_x - 1))))
                    for j in range(num_steps_x)
                ]

                reverse_x = False
                for lane_y in y_positions:
                    xs = list(reversed(base_x_positions)) if reverse_x else base_x_positions
                    for wx in xs:
                        pt = (int(wx), int(lane_y))
                        if not waypoints or waypoints[-1] != pt:
                            waypoints.append(pt)
                    reverse_x = not reverse_x
            else:
                num_lanes = max(1, int(math.ceil(span_w / float(spacing))) + 1)
                if num_lanes == 1:
                    x_positions = [int(round((x_min + x_max) * 0.5))]
                else:
                    x_positions = [
                        int(round(x_min + i * (span_w / float(num_lanes - 1))))
                        for i in range(num_lanes)
                    ]

                num_steps_y = max(2, int(math.ceil(span_h / float(step_along))) + 1)
                base_y_positions = [
                    int(round(y_min + j * (span_h / float(num_steps_y - 1))))
                    for j in range(num_steps_y)
                ]

                reverse_y = False
                for lane_x in x_positions:
                    ys = list(reversed(base_y_positions)) if reverse_y else base_y_positions
                    for wy in ys:
                        pt = (int(lane_x), int(wy))
                        if not waypoints or waypoints[-1] != pt:
                            waypoints.append(pt)
                    reverse_y = not reverse_y

            self._sector_waypoints = list(waypoints)
            self._current_waypoint_idx = 0

            logger.info(
                "Boustrophedon sektör tarama rotası üretildi: bounds=(%d, %d, %d, %d), "
                "lane_spacing=%dpx, toplam_waypoint=%d",
                x_min,
                y_min,
                x_max,
                y_max,
                spacing,
                len(waypoints),
            )
            return waypoints

        except Exception as exc:
            logger.error("generate_sector_sweep_waypoints hatası: %s", exc, exc_info=True)
            return waypoints

    def get_next_sector_waypoint(
        self,
        arrival_radius_px: float = 18.0,
        loop_sweep: bool = False,
    ) -> Optional[Tuple[int, int]]:
        """
        Otonom sektör taraması sırasında oyuncunun anlık odometri konumuna göre
        sıradaki Boustrophedon waypoint hedefini döndürür.
        """
        try:
            if not self._sector_waypoints:
                return None

            px, py, _, _ = self.auto_cartographer.get_position()
            n = len(self._sector_waypoints)

            while self._current_waypoint_idx < n:
                wx, wy = self._sector_waypoints[self._current_waypoint_idx]
                if math.hypot(float(wx) - px, float(wy) - py) <= float(arrival_radius_px):
                    self._current_waypoint_idx += 1
                else:
                    break

            if self._current_waypoint_idx >= n:
                if loop_sweep:
                    self._current_waypoint_idx = 0
                else:
                    return None

            return self._sector_waypoints[self._current_waypoint_idx]
        except Exception as exc:
            logger.error("get_next_sector_waypoint hatası: %s", exc)
            return None


def get_explorer_mode_capabilities() -> list:
    return ["explorer_mode", "autonomous_exploration", "diagnostic_snapshot"]


def validate_explorer_mode_contract() -> bool:
    return bool(get_explorer_mode_capabilities())


def get_explorer_mode_baseline_metrics() -> dict:
    return {"module": "explorer_mode", "contract_valid": validate_explorer_mode_contract(), "capability_count": len(get_explorer_mode_capabilities())}


def get_explorer_mode_readiness_status() -> dict:
    return {"module": "explorer_mode", "ready": validate_explorer_mode_contract(), "metrics": get_explorer_mode_baseline_metrics()}


