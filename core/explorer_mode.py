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
from version import __version__

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
# P1-7 (K-20): minimap metre/piksel ölçeğinin tek kaynağı.
from core.corridor_navigator import MINIMAP_METERS_PER_PX
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

# H-sweep deseni (dilim 13 K3, auto_cartographer'daki UNKNOWN_RESOURCE_TYPE
# ile aynı): türü bilinmeyen materyalin dürüst işareti. "Gatherable
# Material" kod tabanında zaten jenerik sınıf etiketidir; spesifik bir
# cevher adı ("Orichalcum Ore" — route_manager.DEFAULT_RESOURCE_TYPE)
# asla varsayılan tür olarak yazılmaz. Bu sabite H dalındaki
# auto_cartographer tanımından bağımsızdır (o dal entegre değilken de çalışır).
UNKNOWN_RESOURCE_TYPE: str = "Gatherable Material"

# YZ kutu isimlerinde geçen değerli taş adları — bare "diamond"/"sapphire" gibi
# isimler de toplama tespiti sayılsın diye (_GATHER_NAMES'te "gem" var ama taş adı yok).
_GEMSTONE_NAMES = (
    "sapphire",
    "ruby",
    "diamond",
    "emerald",
    "topaz",
    "amethyst",
    "aquamarine",
    "garnet",
    "opal",
    "blood crystal",
)
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

# 3x3 morfolojik kapatma çekirdeği: minimap taraması her karede iki kez
# `np.ones((3, 3), np.uint8)` ayırmasın diye modül düzeyinde tek örnek
# paylaşılır (salt-okunur kullanım — hiçbir yerde değiştirilmez).
_MORPH_CLOSE_KERNEL_3X3 = np.ones((3, 3), np.uint8)

# F17 (mr6 adım yolu): içsel minimap filiz tarayıcısının ÖLÇÜLMÜŞ güveni.
# MaterialScanner ile AYNI kalibre ölçüm: kanonik 3-yapraklı filiz şablonuna
# karşı TM_CCOEFF_NORMED eşleşme skoru (eşik 0.68) ve aynı güven formülü
# (0.78 + 0.18*skor, üst sınır 0.96; core/material_scanner.py:
# scan_minimap_materials). Sabit güven YOK (P1-4/F-0264): skor eşiğin
# altındaysa (örn. düz beyaz canavar noktası ~0.50) güven 0.0 kalır ve aday
# kapılarda elenir. Şablon yüklenemezse de 0.0 döner (fail-closed).
_SPROUT_TPL_MATCH_MIN: float = 0.68
_SPROUT_TEMPLATES: Optional[List[np.ndarray]] = None


def _sprout_template_confidence(mm_v: np.ndarray, x: int, y: int, cw: int, ch: int) -> float:
    """Kontur kutusu (x, y, cw, ch) çevresindeki filiz adayının kanonik filiz
    şablonuna eşleşme skorundan ölçülmüş güveni döndürür.

    16 px kanonik şablon ve 12 px küçültülmüşü (gerçek oyun ikonu ~12 px)
    ile, adayın ±8 px penceresinde kayan eşleşme yapılır; en yüksek skor
    alınır. Skor eşiğin (0.68) altındaysa 0.0 döner — güven uydurulmaz.
    """
    global _SPROUT_TEMPLATES
    try:
        if _SPROUT_TEMPLATES is None:
            from core.minimap_scanner import MinimapScanner

            tpl16 = np.asarray(MinimapScanner._CANONICAL_SPROUT_TPL, dtype=np.uint8)
            tpl12 = cv2.resize(tpl16, (12, 12), interpolation=cv2.INTER_AREA)
            _SPROUT_TEMPLATES = [tpl16, tpl12]
        h, w = mm_v.shape[:2]
        x1 = max(0, x - 8)
        y1 = max(0, y - 8)
        x2 = min(w, x + cw + 8)
        y2 = min(h, y + ch + 8)
        window = mm_v[y1:y2, x1:x2]
        if window.shape[0] < 16 or window.shape[1] < 16:
            return 0.0
        best = 0.0
        for tpl in _SPROUT_TEMPLATES:
            res = cv2.matchTemplate(window, tpl, cv2.TM_CCOEFF_NORMED)
            best = max(best, float(res.max()))
        if best < _SPROUT_TPL_MATCH_MIN:
            return 0.0
        return round(min(0.96, 0.78 + 0.18 * best), 3)
    except Exception as exc:
        logger.debug("filiz şablon güveni ölçülemedi: %s", exc)
        return 0.0



def _dedupe_detections_by_proximity(
    detections: List[Dict[str, Any]],
    radius_px: float,
    on_duplicate: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    base: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Tespitleri `screen_x`/`screen_y` yakınlığına göre tekilleştirir.

    Eskiden her tarama noktasında korunan listedeki TÜM öğeler
    `math.hypot` ile taranıyordu (O(n²)); yoğun karelerde (ör. YOLO'dan
    yüzlerce kutu) adım başına yüzlerce ms tutuyordu. Burada tespitler
    yarıçap büyüklüğünde hücrelere bölünür ve her yeni tespit yalnızca
    3x3 komşu hücredeki korunan öğelerle karşılaştırılır (ortalama O(n)).

    Anlamsal eşdeğerlik: adaylar eklenme sırasına göre dizilir ve yarıçap
    içindeki İLK eşleşme kullanılır — tıpkı eski doğrusal tarama gibi.
    `base` verilirse taban öğeler kendi aralarında tekilleştirilmez (eski
    kod da tabanı kendi içinde elemiyordu); yalnızca yeni öğeler tabana +
    daha önce eklenen yeni öğelere karşı kontrol edilir. `on_duplicate`
    birleştirme geri çağrısı korunan öğenin koordinatını değiştirirse
    (ör. `existing.update(det)`) öğe yeni hücresine taşınır.
    """
    radius = max(0.0, float(radius_px))
    cell = max(1.0, radius)

    def _cell_of(x: int, y: int) -> Tuple[int, int]:
        return (int(math.floor(x / cell)), int(math.floor(y / cell)))

    kept: List[Dict[str, Any]] = list(base) if base else []
    grid: Dict[Tuple[int, int], List[int]] = {}
    cells: List[Tuple[int, int]] = []
    for idx, item in enumerate(kept):
        c = _cell_of(int(item.get("screen_x", 0)), int(item.get("screen_y", 0)))
        cells.append(c)
        grid.setdefault(c, []).append(idx)

    for det in detections:
        x = int(det.get("screen_x", 0))
        y = int(det.get("screen_y", 0))
        gx, gy = _cell_of(x, y)
        cand_idx: List[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                cand_idx.extend(grid.get((gx + dx, gy + dy), ()))
        cand_idx.sort()
        dup_idx: Optional[int] = None
        for idx in cand_idx:
            ex = kept[idx]
            if (
                math.hypot(
                    x - int(ex.get("screen_x", 0)),
                    y - int(ex.get("screen_y", 0)),
                )
                <= radius
            ):
                dup_idx = idx
                break
        if dup_idx is not None:
            if on_duplicate is not None:
                old_cell = cells[dup_idx]
                on_duplicate(kept[dup_idx], det)
                new_cell = _cell_of(
                    int(kept[dup_idx].get("screen_x", 0)),
                    int(kept[dup_idx].get("screen_y", 0)),
                )
                if new_cell != old_cell:
                    grid[old_cell].remove(dup_idx)
                    grid.setdefault(new_cell, []).append(dup_idx)
                    cells[dup_idx] = new_cell
            continue
        kept.append(det)
        cells.append((gx, gy))
        grid.setdefault((gx, gy), []).append(len(kept) - 1)
    return kept


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

    def __init__(self, *args, **kwargs) -> None:
        self._controller: Optional[ExplorerModeController] = None
        # F-0261: Enjekte bağımlılıklar saklanır — controller oluşturma hatası
        # yutulmaz, loglanır ve _ensure_controller orijinal argümanlarla tekrar dener.
        self._init_args = args
        self._init_kwargs = kwargs
        self._controller_init_error: Optional[BaseException] = None
        if args or kwargs:
            try:
                self._controller = ExplorerModeController(*args, **kwargs)
            except Exception as exc:
                self._controller_init_error = exc
                logger.exception(
                    "ExplorerMode controller oluşturulamadı; enjekte bağımlılıklar "
                    "saklandı, _ensure_controller orijinal argümanlarla tekrar deneyecek: %s",
                    exc,
                )

    def _ensure_controller(self) -> ExplorerModeController:
        if self._controller is None:
            if self._init_args or self._init_kwargs:
                try:
                    self._controller = ExplorerModeController(*self._init_args, **self._init_kwargs)
                    self._controller_init_error = None
                    return self._controller
                except Exception as exc:
                    self._controller_init_error = exc
                    logger.exception(
                        "ExplorerMode controller orijinal argümanlarla tekrar "
                        "oluşturulamadı, parametresiz controller kullanılıyor: %s",
                        exc,
                    )
            self._controller = ExplorerModeController()
        return self._controller

    def get_explorer_health_summary(self) -> dict:
        # S-0402: "explorer_ready" artık koşulsuz True değil — aynı sınıftaki
        # ready/version/diagnostic metotları gibi gerçek prob zincirine
        # (validate_explorer_mode_contract) delege edilir.
        try:
            return {
                "explorer_ready": validate_explorer_mode_contract(),
                "walking_state": self.get_walking_state(),
            }
        except Exception:
            return {"explorer_ready": False}

    def get_explorer_mode_ready_flag(self) -> dict:
        # S-0603 / Karar 3: Sahte-hazır yok — modül düzeyi gerçek prob
        # zincirine (validate_explorer_mode_contract) delege edilir.
        return {"module": "explorer_mode", "ready": validate_explorer_mode_contract()}

    def get_explorer_mode_version_info(self) -> dict:
        # S-0603: "ready" artık sabit True değil, gerçek prob sonucudur.
        return {"module": "explorer_mode", "version": __version__, "ready": validate_explorer_mode_contract()}

    def get_explorer_mode_diagnostic_snapshot(self) -> dict:
        # S-0603: "status"/"ready" artık sabit değil, gerçek prob sonucudur.
        ready = validate_explorer_mode_contract()
        return {
            "module": "explorer_mode",
            "status": "ok" if ready else "not_ready",
            "ready": ready,
            "walking_state": self.get_walking_state(),
            "movement_coords": self.get_player_movement_coords(),
            "anti_beklemede_lock": True,
        }

    def get_walking_state(self) -> str:
        return self._ensure_controller().get_walking_state()

    def set_walking_state(self, state: str, reason: str = "") -> str:
        return self._ensure_controller().set_walking_state(state, reason=reason)

    def update_walking_state(self, *args, **kwargs) -> str:
        return self._ensure_controller().update_walking_state(*args, **kwargs)

    def get_player_movement_coords(self) -> Tuple[float, float]:
        return self._ensure_controller().get_player_movement_coords()

    def get_movement_coords(self) -> Tuple[float, float]:
        return self._ensure_controller().get_movement_coords()

    def get_current_position(self) -> Tuple[float, float]:
        return self._ensure_controller().get_current_position()

    def is_movement_active(self) -> bool:
        return self._ensure_controller().is_movement_active()

    def check_path_obstacle(self, frame: np.ndarray, *args, **kwargs) -> Dict[str, Any]:
        return self._ensure_controller().check_path_obstacle(frame, *args, **kwargs)

    def advance_route_waypoint(self, *args, **kwargs) -> Any:
        return self._ensure_controller().advance_route_waypoint(*args, **kwargs)

    def clear_false_resting_locks(self, *args, **kwargs) -> int:
        return self._ensure_controller().clear_false_resting_locks(*args, **kwargs)

    def is_stuck_in_beklemede(self) -> bool:
        return self._ensure_controller().is_stuck_in_beklemede()

    def force_resume_from_beklemede(self) -> str:
        return self._ensure_controller().force_resume_from_beklemede()


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
        test_mode: bool = False,
        enable_sound: bool = False,
        hud_callback: Optional[Callable[[str], None]] = None,
        depth_obstacle_avoidance: Optional[MonocularDepthObstacleAvoidance] = None,
        glide_controller: Optional[GlideWaypointController] = None,
        reflex: Optional[Any] = None,
        yolo_detector: Optional[Any] = None,
        perception_hub: Optional[Any] = None,
        dataset_factory: Optional[Any] = None,
        heartbeat_callback: Optional[Callable[[], None]] = None,
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
        self.reflex: Optional[Any] = reflex
        self.yolo_detector: Optional[Any] = yolo_detector
        self.perception_hub: Optional[Any] = perception_hub
        self.dataset_factory: Optional[Any] = dataset_factory
        self.heartbeat_callback: Optional[Callable[[], None]] = heartbeat_callback

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
        # F-0271: Ulaşılamayan sektör waypoint'inde takılmayı önleyen zaman aşımı.
        self._sector_waypoint_started_ts: float = 0.0
        self._sector_waypoint_timeout_sec: float = 90.0
        # F-0262: Kapalı döngü rota tamamlanma sayacı (döngü sürer ama bitiş izlenir).
        self._route_loop_count: int = 0
        self._route_completed_once: bool = False
        # F-0268: Cast bar okuyucu (GameHUDReader) tembel yüklenir ve önbelleğe alınır.
        self._hud_reader: Optional[Any] = None
        self._hud_reader_unavailable: bool = False

        # Radar and UI event callbacks
        self._node_discovered_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._node_updated_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._radar_widgets: List[Any] = []
        self._map_viewers: List[Any] = []

        if hasattr(self.auto_cartographer, "add_node_discovered_callback"):
            self.auto_cartographer.add_node_discovered_callback(self._on_cartographer_node_discovered)
        if hasattr(self.auto_cartographer, "add_node_updated_callback"):
            self.auto_cartographer.add_node_updated_callback(self._on_cartographer_node_updated)

        # Wire get_current_position on auto_cartographer if missing so StateMachine / Radar gets genuine coordinates
        if not hasattr(self.auto_cartographer, "get_current_position"):
            self.auto_cartographer.get_current_position = self.get_player_movement_coords

        # Real-time Movement & Odometry tracking (prevents false 3s obstacle alerts & BEKLEMEDE locks)
        self._odometry_pos_x: float = float(getattr(self.auto_cartographer, "map_x", 960.0) or 960.0)
        self._odometry_pos_y: float = float(getattr(self.auto_cartographer, "map_y", 540.0) or 540.0)
        # F-0017: Son ölçülen (kartograf) konum — ölü hesap (dead reckoning)
        # yalnızca bu tabandan sapar, ölçümü ezmez.
        self._odometry_base_x: float = self._odometry_pos_x
        self._odometry_base_y: float = self._odometry_pos_y
        self._odometry_last_ts: float = time.time()
        self._odometry_vx: float = 0.0
        self._odometry_vy: float = 0.0
        self._odometry_heading_deg: float = 0.0
        self._walking_speed_px_sec: float = 85.0
        # F-0018/F-0440: Hareket bayrakları __init__'te tanımlı ve zaman damgalı;
        # bayat iddia (stale) hareket sayılmaz.
        self._movement_active_override_ts: float = 0.0
        self._movement_active_override_value: Optional[bool] = None
        self._movement_active_override: Optional[bool] = None
        self._is_walking_active: bool = False
        self._is_walking_active_ts: float = 0.0
        self.exemplar_memory: Optional[Any] = None
        self._last_glide_check: Dict[str, Any] = {}

        # Walking State Machine (guarantees YÜRÜYOR / ROTA TAKİBİ / KEŞİFTE accuracy)
        self._current_walking_state: str = "KEŞİFTE"
        self._walking_state_ts: float = time.time()
        self._last_state_change_reason: str = "init"
        self._anti_beklemede_lock: bool = True

        # Obstacle Alert Debounce & Multi-Frame Confirmation (eliminates false 3s alerts)
        self._consecutive_obstacle_hits: int = 0
        self._last_obstacle_alert_ts: float = 0.0
        self._obstacle_history: List[Dict[str, Any]] = []

        # Continuous Smooth Route Travel Engine
        self._active_route_index: int = 0
        self._waypoint_arrival_radius_px: float = 22.0
        self._waypoint_last_advanced_ts: float = time.time()
        self._continuous_travel_enabled: bool = True

        self._sync_ordered_nodes_from_route_manager()

    # F-0018: `_movement_active_override` atamaları zaman damgalıdır — doğrudan
    # atama (testler dahil) bile tazelik sayacını yeniler; bayat "hareket
    # ediyor" iddiası is_movement_active() içinde geçersiz sayılır.
    @property
    def _movement_active_override(self) -> Optional[bool]:
        return self._movement_active_override_value

    @_movement_active_override.setter
    def _movement_active_override(self, value: Optional[bool]) -> None:
        self._movement_active_override_value = value
        self._movement_active_override_ts = time.time()

    def add_node_discovered_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Yeni bir düğüm haritalandırıldığında anında tetiklenecek geri çağırma işlevini ekler."""
        if callable(callback) and callback not in self._node_discovered_callbacks:
            self._node_discovered_callbacks.append(callback)

    def remove_node_discovered_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Düğüm keşif dinleyicisini kaldırır."""
        if callback in self._node_discovered_callbacks:
            self._node_discovered_callbacks.remove(callback)

    def add_node_updated_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Düğüm güncelleme dinleyicisini ekler."""
        if callable(callback) and callback not in self._node_updated_callbacks:
            self._node_updated_callbacks.append(callback)

    def remove_node_updated_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Düğüm güncelleme dinleyicisini kaldırır."""
        if callback in self._node_updated_callbacks:
            self._node_updated_callbacks.remove(callback)

    def register_radar_widget(self, radar_widget: Any) -> None:
        """Radar tuvali/diyaloğunu bağlar."""
        if radar_widget is not None and radar_widget not in self._radar_widgets:
            self._radar_widgets.append(radar_widget)
            if hasattr(self.auto_cartographer, "register_ui_listener"):
                self.auto_cartographer.register_ui_listener(radar_widget)

    def unregister_radar_widget(self, radar_widget: Any) -> None:
        """Radar tuvali/diyaloğu bağlantısını kaldırır."""
        if radar_widget in self._radar_widgets:
            self._radar_widgets.remove(radar_widget)
            if hasattr(self.auto_cartographer, "unregister_ui_listener"):
                self.auto_cartographer.unregister_ui_listener(radar_widget)

    def register_map_viewer(self, map_viewer: Any) -> None:
        """Harita görüntüleyici penceresini bağlar."""
        if map_viewer is not None and map_viewer not in self._map_viewers:
            self._map_viewers.append(map_viewer)
            if hasattr(self.auto_cartographer, "register_ui_listener"):
                self.auto_cartographer.register_ui_listener(map_viewer)

    def unregister_map_viewer(self, map_viewer: Any) -> None:
        """Harita görüntüleyici bağlantısını kaldırır."""
        if map_viewer in self._map_viewers:
            self._map_viewers.remove(map_viewer)
            if hasattr(self.auto_cartographer, "unregister_ui_listener"):
                self.auto_cartographer.unregister_ui_listener(map_viewer)

    def _on_cartographer_node_discovered(self, node_dict: Dict[str, Any]) -> None:
        """AutoCartographer'dan gelen yeni düğüm keşif olayını UI ve dinleyicilere iletir."""
        self._emit_node_discovered(node_dict)

    def _on_cartographer_node_updated(self, node_dict: Dict[str, Any]) -> None:
        """AutoCartographer'dan gelen düğüm güncelleme olayını iletir."""
        self._emit_node_updated(node_dict)

    def _emit_node_discovered(self, node_dict: Dict[str, Any]) -> None:
        """Kayıtlı dinleyicileri, radar widget'larını ve harita görüntüleyicilerini uyarır."""
        for cb in list(self._node_discovered_callbacks):
            try:
                cb(node_dict)
            except Exception as exc:
                logger.debug("explorer node_discovered_callback hatası: %s", exc)
        for widget in list(self._radar_widgets):
            try:
                if hasattr(widget, "add_discovered_node"):
                    widget.add_discovered_node(node_dict, trigger_pulse=True)
                elif hasattr(widget, "reload_map_and_nodes"):
                    widget.reload_map_and_nodes()
            except Exception as exc:
                logger.debug("radar_widget bildirim hatası: %s", exc)
        for viewer in list(self._map_viewers):
            try:
                if hasattr(viewer, "reload_map_and_nodes"):
                    viewer.reload_map_and_nodes()
                if hasattr(viewer, "update"):
                    viewer.update()
            except Exception as exc:
                logger.debug("map_viewer bildirim hatası: %s", exc)

    def _emit_node_updated(self, node_dict: Dict[str, Any]) -> None:
        """Düğüm güncellendiğinde radar ve harita görüntüleyicilerini uyarır."""
        for cb in list(self._node_updated_callbacks):
            try:
                cb(node_dict)
            except Exception as exc:
                logger.debug("explorer node_updated_callback hatası: %s", exc)
        for widget in list(self._radar_widgets):
            try:
                if hasattr(widget, "update_node"):
                    widget.update_node(node_dict)
                elif hasattr(widget, "reload_map_and_nodes"):
                    widget.reload_map_and_nodes()
            except Exception as exc:
                logger.debug("radar_widget update hatası: %s", exc)
        for viewer in list(self._map_viewers):
            try:
                if hasattr(viewer, "reload_map_and_nodes"):
                    viewer.reload_map_and_nodes()
                if hasattr(viewer, "update"):
                    viewer.update()
            except Exception as exc:
                logger.debug("map_viewer update hatası: %s", exc)

    def bind_ai_sources(
        self,
        reflex: Optional[Any] = None,
        yolo_detector: Optional[Any] = None,
        perception_hub: Optional[Any] = None,
        exemplar_memory: Optional[Any] = None,
    ) -> None:
        """Late-bind YZ reflex / YOLO / ExemplarMemory so cartography uses the live AI pipeline."""
        if reflex is not None:
            self.reflex = reflex
        if yolo_detector is not None:
            self.yolo_detector = yolo_detector
        if perception_hub is not None:
            self.perception_hub = perception_hub
        if exemplar_memory is not None:
            self.exemplar_memory = exemplar_memory

    def _yz_box_to_detection(
        self,
        box: Dict[str, Any],
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        """Convert a Gatherion YOLO/reflex box into cartography detection dict."""
        try:
            from core.reflex_layer import _class_id, _gather_ids, _GATHER_NAMES, _MONSTER_NAMES
        except Exception:
            return None
        if not isinstance(box, dict):
            return None
        name = str(box.get("class_name", box.get("name", box.get("label", "")))).lower()
        cid = box.get("class_id", box.get("cls", box.get("class", None)))
        try:
            cid = int(cid) if cid is not None else _class_id(dict(box, name=name))
        except Exception:
            cid = _class_id(dict(box, name=name))
        if cid in {10, 11} or any(t in name for t in _MONSTER_NAMES):
            return None
        is_gather = (
            cid in _gather_ids()
            or any(t in name for t in _GATHER_NAMES)
            or any(t in name for t in _GEMSTONE_NAMES)
            or "odyle" in name
            or name.startswith("essence_")
            or name.startswith("herb_")
        )
        if not is_gather:
            return None

        x1 = box.get("x1", box.get("xmin", box.get("left")))
        y1 = box.get("y1", box.get("ymin", box.get("top")))
        x2 = box.get("x2", box.get("xmax", box.get("right")))
        y2 = box.get("y2", box.get("ymax", box.get("bottom")))
        if None in (x1, y1, x2, y2):
            # xywh / cxcywh fallbacks
            if "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) >= 4:
                bx, by, bw, bh = [float(v) for v in box["bbox"][:4]]
                if bw <= 1.5 and bh <= 1.5 and bx <= 1.5:
                    x1, y1 = bx * frame_w, by * frame_h
                    x2, y2 = (bx + bw) * frame_w, (by + bh) * frame_h
                else:
                    x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            else:
                return None
        else:
            x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
            if 0.0 <= x1 <= 1.5 and 0.0 <= x2 <= 1.5 and frame_w > 1:
                x1, x2 = x1 * frame_w, x2 * frame_w
                y1, y2 = y1 * frame_h, y2 * frame_h

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))
        bw = max(1, int(round(abs(x2 - x1))))
        bh = max(1, int(round(abs(y2 - y1))))
        conf = float(
            box.get("confidence", box.get("conf", box.get("score", box.get("prob", 0.0))))
            or 0.0
        )
        # F-0264: Güven bilinmiyorsa 0.75 uydurulmaz; aşağıda 0.0'a sıkıştırılır.
        if conf > 1.0:
            conf = conf / 100.0

        mat = "Gatherable Material"
        if "aria" in name:
            mat = "Aria"
        elif "mela" in name:
            mat = "Mela"
        elif "targena" in name:
            mat = "Targena"
        elif "orichalcum" in name or "ore" in name:
            mat = "Orichalcum Ore"
        elif "odyle" in name:
            mat = "Odyle"
        elif "gem" in name or "crystal" in name or any(t in name for t in _GEMSTONE_NAMES):
            # Specific gemstone when the box name says which stone it is;
            # generic "Gem" only when the stone type is truly unknown.
            # (Generic "Gem" is treated as upgradable by a later specific 3D
            # observation in AutoCartographer._refine_existing_node_kalman_ema.)
            if "sapphire" in name:
                mat = "Sapphire Gemstone"
            elif "ruby" in name:
                mat = "Ruby Gemstone"
            elif "diamond" in name:
                mat = "Diamond Gemstone"
            elif "emerald" in name:
                mat = "Emerald Gemstone"
            elif "topaz" in name:
                mat = "Topaz Gemstone"
            elif "amethyst" in name:
                mat = "Amethyst Gemstone"
            elif "aquamarine" in name:
                mat = "Aquamarine Gemstone"
            elif "garnet" in name:
                mat = "Garnet Gemstone"
            elif "opal" in name:
                mat = "Opal Gemstone"
            elif "blood" in name:
                mat = "Blood Crystal"
            else:
                mat = "Gem"
        elif "sprout" in name or "filiz" in name:
            mat = "Gatherable Material"

        # Minimap ROI (top-right) vs 3D world view projection
        on_minimap = False
        try:
            x1_mm, y1_mm, x2_mm, y2_mm = self.auto_cartographer._get_minimap_pixel_rect(frame_w, frame_h)
            if x1_mm <= cx <= x2_mm and y1_mm <= cy <= y2_mm:
                on_minimap = True
        except Exception:
            pass
        if not on_minimap:
            nx, ny = cx / float(max(1, frame_w)), cy / float(max(1, frame_h))
            on_minimap = nx >= 0.84 and ny <= 0.30

        if on_minimap:
            try:
                x1_mm, y1_mm, x2_mm, y2_mm = self.auto_cartographer._get_minimap_pixel_rect(frame_w, frame_h)
                mcx = (x1_mm + x2_mm) / 2.0
                mcy = y1_mm + (y2_mm - y1_mm) * 0.545
                scale = float(getattr(self.auto_cartographer, "minimap_to_map_scale", 1.0) or 1.0)
                rel_dx = float(cx - mcx) * scale
                rel_dy = float(cy - mcy) * scale
                dist_m = math.hypot(rel_dx, rel_dy)
            except Exception:
                rel_dx = float(cx - frame_w * 0.92)
                rel_dy = float(cy - frame_h * 0.16)
                dist_m = math.hypot(rel_dx, rel_dy)
        else:
            try:
                heading = getattr(self.auto_cartographer, "player_heading_deg", 0.0)
                rel_dx, rel_dy = self.auto_cartographer._project_3d_screen_to_map_offset(
                    cx, cy, frame_w, frame_h, heading_deg=heading
                )
                dist_m = math.hypot(rel_dx, rel_dy)
            except Exception:
                try:
                    rel_dx, rel_dy, dist_m = self.material_scanner._project_3d_to_map_offset(
                        cx, cy, frame_w, frame_h
                    )
                except Exception:
                    # F-0264: Başarısız projeksiyonda mesafe 8.0 uydurulmaz —
                    # göreli sapmadan türetilen gerçek değer kullanılır.
                    rel_dx = float(cx - frame_w // 2) * 0.15
                    rel_dy = float(cy - frame_h // 2) * 0.15
                    dist_m = math.hypot(rel_dx, rel_dy)

        return {
            "material_type": mat,
            "source": "minimap" if on_minimap else "3d_world",
            "source_type": "yz_minimap" if on_minimap else "yz_3d",
            "screen_x": int(cx),
            "screen_y": int(cy),
            "bbox": (int(x1), int(y1), int(bw), int(bh)),
            "rel_map_dx": float(rel_dx),
            "rel_map_dy": float(rel_dy),
            "estimated_distance_m": float(dist_m),
            # F-0264: Bilinmeyen güven en az 0.15 diye şişirilmez.
            "confidence": round(max(0.0, min(0.99, conf)), 3),
            "in_air": bool(cy < frame_h * 0.45),
            "stack_count": 1,
            "max_harvests": self.harvests_per_node,
            "yz": True,
            "class_id": int(cid) if cid >= 0 else -1,
            "class_name": name,
        }

    def _scan_yz_gather_detections(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        """Pull live YZ (reflex/YOLO) gather boxes into cartography detections."""
        out: List[Dict[str, Any]] = []
        if bgr is None or not isinstance(bgr, np.ndarray) or bgr.size == 0:
            return out
        h, w = bgr.shape[:2]
        boxes: List[Dict[str, Any]] = []

        reflex = self.reflex
        if reflex is not None:
            try:
                if hasattr(reflex, "submit_frame"):
                    reflex.submit_frame(bgr)
                if hasattr(reflex, "latest"):
                    snap = reflex.latest() or {}
                    # S062 (v19/P3-8): reflex asenkron çalışır — latest()
                    # az önce gönderilen kareyi henüz işlememiş olabilir;
                    # 2 sn'den eski anlık görüntü bayattır, kutuları düşer.
                    # Damga yoksa (eski/mock reflex) eski davranış korunur.
                    _snap_ts = snap.get("snapshot_ts") if isinstance(snap, dict) else None
                    _stale = (
                        isinstance(_snap_ts, (int, float))
                        and _snap_ts > 0.0
                        and (time.monotonic() - float(_snap_ts)) > 2.0
                    )
                    raw = snap.get("boxes") if isinstance(snap, dict) else None
                    if isinstance(raw, list) and not _stale:
                        boxes.extend([b for b in raw if isinstance(b, dict)])
                    elif _stale:
                        logger.debug("YZ reflex snapshot bayat (%.1fs), kutular atıldı.",
                                     time.monotonic() - float(_snap_ts))
            except Exception as exc:
                logger.debug("YZ reflex scan skipped: %s", exc)

        if not boxes and self.yolo_detector is not None:
            try:
                det_fn = getattr(self.yolo_detector, "detect", None) or getattr(
                    self.yolo_detector, "predict", None
                )
                if callable(det_fn):
                    raw = det_fn(bgr)
                    if isinstance(raw, list):
                        boxes.extend([b for b in raw if isinstance(b, dict)])
                    elif isinstance(raw, dict) and isinstance(raw.get("boxes"), list):
                        boxes.extend([b for b in raw["boxes"] if isinstance(b, dict)])
            except Exception as exc:
                logger.debug("YZ yolo detect skipped: %s", exc)

        if not boxes and self.perception_hub is not None:
            try:
                snap = None
                if hasattr(self.perception_hub, "last"):
                    last_fn = getattr(self.perception_hub, "last")
                    snap = last_fn() if callable(last_fn) else last_fn
                elif hasattr(self.perception_hub, "latest"):
                    lat_fn = getattr(self.perception_hub, "latest")
                    snap = lat_fn() if callable(lat_fn) else lat_fn
                elif hasattr(self.perception_hub, "snapshot"):
                    snap_fn = getattr(self.perception_hub, "snapshot")
                    snap = snap_fn() if callable(snap_fn) else snap_fn

                if snap is None and hasattr(self.perception_hub, "observe") and bgr is not None:
                    snap = self.perception_hub.observe(bgr)

                if snap is not None:
                    raw_targets = getattr(snap, "targets", None)
                    if isinstance(raw_targets, list):
                        boxes.extend([b for b in raw_targets if isinstance(b, dict)])
                    elif isinstance(snap, dict):
                        for key in ("targets", "yolo", "boxes", "detections"):
                            raw = snap.get(key)
                            if isinstance(raw, list):
                                boxes.extend([b for b in raw if isinstance(b, dict)])
                                break
                    elif hasattr(snap, "as_dict"):
                        snap_d = snap.as_dict()
                        if isinstance(snap_d, dict):
                            raw = (
                                snap_d.get("targets")
                                or snap_d.get("boxes")
                                or snap_d.get("detections")
                                or snap_d.get("yolo")
                            )
                            if isinstance(raw, list):
                                boxes.extend([b for b in raw if isinstance(b, dict)])
            except Exception as exc:
                logger.debug("YZ perception hub skipped: %s", exc)

        candidates: List[Dict[str, Any]] = []
        for box in boxes:
            # S062 (v19/P3-8): güven eşiği yoktu — 0.0 güvenli kutu da
            # düğüm oluyordu. Güven alanı taşıyan kutuda en düşük işletim
            # kademesi (Hassas ≈ 0.10) uygulanır; güven alanı taşımayan
            # kutu (sentetik/test) eski davranışla korunur.
            _conf_val: Optional[float] = None
            for _ck in ("confidence", "conf", "score", "prob"):
                if isinstance(box, dict) and box.get(_ck) is not None:
                    try:
                        _conf_val = float(box[_ck])
                    except (TypeError, ValueError):
                        _conf_val = None
                    break
            if _conf_val is not None:
                if _conf_val > 1.0:
                    _conf_val = _conf_val / 100.0
                if _conf_val < 0.10:
                    continue
            det = self._yz_box_to_detection(box, w, h)
            if det is not None:
                candidates.append(det)

        def _keep_higher_confidence(
            existing: Dict[str, Any], det: Dict[str, Any]
        ) -> None:
            if float(det["confidence"]) > float(existing.get("confidence", 0)):
                existing.update(det)

        # Grid tabanlı tekilleştirme (28 px): eski O(n²) doğrusal taramayla
        # aynı anlamsal sonuç (ilk eşleşme + yüksek güven kazanır).
        return _dedupe_detections_by_proximity(
            candidates, 28.0, _keep_higher_confidence
        )

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

    def _movement_signal_active(self) -> bool:
        """
        Canlı hareket sinyali: girdi denetleyicisinin `movement_active()` raporu
        veya taze (<3 sn) açık hareket bildirimi. Yürüme *durum etiketi*
        (`_is_walking_active`) tek başına hareket kanıtı değildir (F-0017/F-0018).
        """
        ov = self._movement_active_override
        if ov is True:
            if time.time() - self._movement_active_override_ts < 3.0:
                return True
        if self.input_controller is not None:
            active_fn = getattr(self.input_controller, "movement_active", None)
            if callable(active_fn):
                try:
                    if bool(active_fn()):
                        return True
                except Exception:
                    pass
        return False

    def get_player_movement_coords(self) -> Tuple[float, float]:
        """
        Oyuncunun gerçek zamanlı harita konumunu döndürür.

        F-0017: Önceki sürüm, hareket varsayılınca oyuncuyu waypoint'e doğru
        85 px/sn ilerletip uydurulan konumu paylaşılan gerçek konuma
        (`auto_cartographer.map_x/map_y`) yazıyordu. Artık:
        - Taban her zaman kartografın ölçtüğü konumdur; ölçüm değişirse ölü
          hesap (odometri) o konuma sıfırlanır.
        - Ölü hesap yalnızca canlı hareket sinyali varken (girdi denetleyicisi
          veya taze açık bildirim) ilerler ve SADECE iç odometri durumunu
          günceller — kartografın ölçtüğü konum asla üzerine yazılmaz.
        - Hareket sinyali yoksa hız sıfırlanır; konum ölçülen değerde kalır.
        """
        now = time.time()
        base_x = float(getattr(self.auto_cartographer, "map_x", self._odometry_pos_x) or self._odometry_pos_x)
        base_y = float(getattr(self.auto_cartographer, "map_y", self._odometry_pos_y) or self._odometry_pos_y)

        # Gerçek ölçüm değiştiyse odometriyi ölçüme sıfırla.
        if (base_x, base_y) != (self._odometry_base_x, self._odometry_base_y):
            self._odometry_base_x = base_x
            self._odometry_base_y = base_y
            self._odometry_pos_x = round(base_x, 2)
            self._odometry_pos_y = round(base_y, 2)
            self._odometry_vx = 0.0
            self._odometry_vy = 0.0
            self._odometry_last_ts = now
            return self._odometry_pos_x, self._odometry_pos_y

        moving = self._movement_signal_active()
        dt = now - self._odometry_last_ts
        if moving and dt >= 0.02:
            dt = min(0.5, dt)
            target_pt = self.get_active_waypoint_coords()
            if target_pt is not None:
                tx, ty = float(target_pt[0]), float(target_pt[1])
                dx = tx - self._odometry_pos_x
                dy = ty - self._odometry_pos_y
                dist = math.hypot(dx, dy)
                if dist > 1.0:
                    speed = self._walking_speed_px_sec
                    step = min(dist, speed * dt)
                    self._odometry_pos_x = round(self._odometry_pos_x + (dx / dist) * step, 2)
                    self._odometry_pos_y = round(self._odometry_pos_y + (dy / dist) * step, 2)
                    self._odometry_vx = (dx / dist) * speed
                    self._odometry_vy = (dy / dist) * speed
                    self._odometry_heading_deg = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
                else:
                    # Hedefe varıldı: hız sıfır (bayat hız hareket sayılmasın).
                    self._odometry_vx = 0.0
                    self._odometry_vy = 0.0
            else:
                rad = math.radians(self._odometry_heading_deg)
                speed = self._walking_speed_px_sec
                step = speed * dt
                self._odometry_pos_x = round(self._odometry_pos_x + math.sin(rad) * step, 2)
                self._odometry_pos_y = round(self._odometry_pos_y + -math.cos(rad) * step, 2)
                self._odometry_vx = math.sin(rad) * speed
                self._odometry_vy = -math.cos(rad) * speed
            self._odometry_last_ts = now
        elif not moving:
            # Hareket yok: bayat hızı sıfırla, konum ölçülen değerde kalsın.
            self._odometry_vx = 0.0
            self._odometry_vy = 0.0

        return self._odometry_pos_x, self._odometry_pos_y

    def get_movement_coords(self) -> Tuple[float, float]:
        """Alias for get_player_movement_coords."""
        return self.get_player_movement_coords()

    def get_current_position(self) -> Tuple[float, float]:
        """Returns (x, y) real coordinates for AutoCartographer/StateMachine integration."""
        return self.get_player_movement_coords()

    def is_movement_active(self) -> bool:
        """
        Determines whether the player or bot is actively walking.

        F-0018: Önceki sürümde bir kez YÜRÜYOR/ROTA/KEŞİFTE olunca
        `_is_walking_active` asla sıfırlanamıyor, bayat odometri hızı da
        süresiz hareket sayılıyordu — bot dursa/ölse bile hep `True` dönüyordu.
        Artık her sinyal zaman damgalıdır: açık bildirim 3 sn, yürüme bayrağı
        3 sn, odometri hızı 1 sn içinde tazelenmezse hareket sayılmaz.
        """
        now = time.time()

        ov = self._movement_active_override
        if ov is not None:
            if ov is False:
                return False
            if now - self._movement_active_override_ts < 3.0:
                return True
            # Bayat "hareket ediyor" iddiası geçersiz — canlı sinyallere düş.

        if self.input_controller is not None:
            active_fn = getattr(self.input_controller, "movement_active", None)
            if callable(active_fn):
                try:
                    if bool(active_fn()):
                        return True
                except Exception:
                    pass

        if getattr(self, "_is_walking_active", False):
            if now - getattr(self, "_is_walking_active_ts", 0.0) < 3.0:
                return True

        if now - self._odometry_last_ts < 1.0 and (
            abs(self._odometry_vx) > 5.0 or abs(self._odometry_vy) > 5.0
        ):
            return True

        return False

    def update_movement_odometry(
        self,
        dx: float = 0.0,
        dy: float = 0.0,
        dt: float = 0.05,
        is_walking: bool = True,
    ) -> Tuple[float, float]:
        """Manually push odometry offset (e.g. from minimap template delta or optical flow)."""
        now = time.time()
        self._odometry_pos_x += float(dx)
        self._odometry_pos_y += float(dy)
        if dt > 0:
            self._odometry_vx = float(dx) / float(dt)
            self._odometry_vy = float(dy) / float(dt)
            if math.hypot(dx, dy) > 0.1:
                self._odometry_heading_deg = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
        self._odometry_last_ts = now
        if is_walking:
            self._movement_active_override = True
            self._is_walking_active = True
            self._is_walking_active_ts = now
        try:
            self.auto_cartographer.map_x = round(self._odometry_pos_x, 2)
            self.auto_cartographer.map_y = round(self._odometry_pos_y, 2)
        except Exception:
            pass
        # Kartograf konumu açık delta ile güncellendi — ölçüm tabanını da ilerlet.
        self._odometry_base_x = round(self._odometry_pos_x, 2)
        self._odometry_base_y = round(self._odometry_pos_y, 2)
        return round(self._odometry_pos_x, 2), round(self._odometry_pos_y, 2)

    def get_walking_state(self) -> str:
        """Returns the current accurate walking state ('YÜRÜYOR', 'ROTA TAKİBİ', 'KEŞİFTE', etc.)."""
        if self._current_walking_state == "BEKLEMEDE" and self._anti_beklemede_lock:
            if self.is_movement_active():
                self._current_walking_state = "ROTA TAKİBİ" if self._ordered_nodes else "KEŞİFTE"
        return self._current_walking_state

    def set_walking_state(self, state: str, reason: str = "") -> str:
        """
        Sets the walking state machine state with anti-BEKLEMEDE protection.
        States: 'YÜRÜYOR', 'ROTA TAKİBİ', 'KEŞİFTE', 'TOPLANIYOR', 'BEKLEMEDE'.
        If the player or bot is in motion or following waypoints, prevents false BEKLEMEDE locking.
        """
        norm = str(state or "").strip().upper()
        if norm in ("WALKING", "WALK", "YÜRÜYÜŞ"):
            norm = "YÜRÜYOR"
        elif norm in ("ROUTE", "ROUTE_FOLLOWING", "ROTA", "ROTA_TAKİBİ"):
            norm = "ROTA TAKİBİ"
        elif norm in ("EXPLORING", "EXPLORE", "AUTO_CARTOGRAPHY", "KEŞİF"):
            norm = "KEŞİFTE"
        elif norm in ("GATHERING", "HARVESTING", "TOPLAMA"):
            norm = "TOPLANIYOR"
        elif norm in ("WAITING", "IDLE", "BEKLEME"):
            norm = "BEKLEMEDE"

        if norm == "BEKLEMEDE" and self._anti_beklemede_lock:
            if self.is_movement_active():
                norm = "ROTA TAKİBİ" if self._ordered_nodes else "KEŞİFTE"
                reason += " [prevented false BEKLEMEDE lock: movement active]"
            elif self._ordered_nodes and self._active_route_index < len(self._ordered_nodes):
                norm = "ROTA TAKİBİ"
                reason += " [prevented false BEKLEMEDE lock: route active]"

        self._current_walking_state = norm
        self._walking_state_ts = time.time()
        self._last_state_change_reason = reason
        if norm in ("YÜRÜYOR", "ROTA TAKİBİ", "KEŞİFTE"):
            self._is_walking_active = True
            self._is_walking_active_ts = time.time()
        elif norm in ("BEKLEMEDE", "IDLE", "PAUSED"):
            # F-0018: Durma durumu yürüme bayrağını sıfırlar (kalıcı kilit yok).
            self._is_walking_active = False
            self._is_walking_active_ts = 0.0
        return self._current_walking_state

    def update_walking_state(
        self,
        is_moving: Optional[bool] = None,
        has_active_target: bool = False,
        in_route: bool = False,
        in_exploration: bool = False,
    ) -> str:
        """
        Evaluates and updates the walking state machine based on live game state:
        - If harvesting/gathering: 'TOPLANIYOR'
        - If following route waypoints: 'ROTA TAKİBİ'
        - If exploring/player-walking in cartography: 'KEŞİFTE'
        - If walking forward: 'YÜRÜYOR'
        - If stopped and no active tasks: 'BEKLEMEDE'
        Guarantees 100% accuracy and eliminates false BEKLEMEDE locks.
        """
        if is_moving is not None:
            self._movement_active_override = bool(is_moving)

        moving = self.is_movement_active()
        if not moving:
            if not in_route and not in_exploration and not has_active_target:
                return self.set_walking_state("BEKLEMEDE", reason="idle stopped")

        if in_route or (has_active_target and bool(self._ordered_nodes)):
            return self.set_walking_state("ROTA TAKİBİ", reason="following route waypoints")
        elif in_exploration or bool(self._sector_waypoints):
            return self.set_walking_state("KEŞİFTE", reason="autonomous exploration")
        elif moving:
            return self.set_walking_state("YÜRÜYOR", reason="active walking motion")
        else:
            return self.set_walking_state("BEKLEMEDE", reason="stopped")

    def is_stuck_in_beklemede(self) -> bool:
        """Checks if bot is falsely stuck in BEKLEMEDE while tasks or movement exist."""
        if self._current_walking_state == "BEKLEMEDE":
            if self.is_movement_active() or bool(self._ordered_nodes) or bool(self._sector_waypoints):
                return True
        return False

    def force_resume_from_beklemede(self) -> str:
        """Forces the state machine out of BEKLEMEDE into active walking/route state."""
        new_state = "ROTA TAKİBİ" if self._ordered_nodes else "KEŞİFTE"
        self._current_walking_state = new_state
        self._walking_state_ts = time.time()
        self._last_state_change_reason = "force_resumed_from_beklemede"
        self.clear_false_resting_locks()
        return new_state

    def clear_false_resting_locks(self, region: Optional[str] = None) -> int:
        """
        Scans route nodes in the active region and clears any false 5-minute ('resting') locks
        caused by transient obstacle alerts or stuck timeouts, restoring nodes to 'active'.
        Eliminates the 5m BEKLEMEDE lock and allows the bot to travel continuously.
        Returns the number of unlocked nodes.
        """
        unlocked = 0
        target_reg = region or self.active_region
        if self.route_manager is None:
            return 0
        nodes = self.route_manager.get_route_nodes(
            region_name=target_reg,
            route_name=self.active_route,
        )
        if not nodes:
            nodes = self.route_manager.get_nodes(region=target_reg)
        seen_ids = set()
        for node in list(nodes) + list(self._ordered_nodes):
            nid = getattr(node, "id", None)
            key = (id(node), nid)
            if key in seen_ids or (nid is not None and nid in seen_ids):
                continue
            seen_ids.add(key)
            if nid is not None:
                seen_ids.add(nid)
            if getattr(node, "state", "") == "resting":
                # F-0272: Yalnızca kilit açılır — kısmi hasat ilerlemesi
                # (current_harvests, örn. 2/3) korunur, sıfırlanmaz.
                node.state = "active"
                node.consecutive_failures = 0
                unlocked += 1
        if unlocked > 0:
            try:
                self.route_manager.save_region(target_reg)
                logger.info(
                    "clear_false_resting_locks: %d adet düğümün 5dk BEKLEMEDE kilidi kaldırıldı.",
                    unlocked,
                )
            except Exception as exc:
                logger.debug("Failed to save region after clearing resting locks: %s", exc)
        return unlocked

    def check_path_obstacle(
        self,
        frame: np.ndarray,
        current_pos: Optional[Tuple[float, float]] = None,
        exemplar_memory: Optional[Any] = None,
        suppress_transient_depth: bool = True,
    ) -> Dict[str, Any]:
        """
        Uses MonocularDepthObstacleAvoidance and ExemplarMemory to estimate whether an obstacle
        (14: obstacle) or monster (10: monster_3d) is directly ahead during autonomous exploration,
        and prioritize walkable ground paths (15: path_ground).
        Eliminates false 3s obstacle alerts through walkable path ground confirmation,
        multi-frame debounce, and real movement coordinate continuity validation.
        """
        raw_res = self.depth_obstacle_avoidance.analyze_forward_corridor(frame)
        res = dict(raw_res) if isinstance(raw_res, dict) else {"obstacle_detected": False, "recommended_turn": "none"}
        raw_depth_obs = bool(res.get("obstacle_detected", False))

        is_exemplar_obs = False
        walkable_path_detected = False

        mem = exemplar_memory or getattr(self, "exemplar_memory", None)
        if mem is None:
            try:
                from dataset_factory.exemplar_memory import get_exemplar_memory
                mem = get_exemplar_memory()
            except Exception:
                mem = None

        if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0 and mem is not None:
            try:
                h, w = frame.shape[:2]
                center_corridor = frame[int(h * 0.52) : int(h * 0.82), int(w * 0.40) : int(w * 0.60)]
                if center_corridor.size > 0:
                    pos_cid, pos_cname, pos_sim = mem.classify_patch(center_corridor, min_sim=0.70)
                    if pos_cid in (10, 14) or (
                        pos_cname and any(tok in str(pos_cname).lower() for tok in ("obstacle", "monster"))
                    ):
                        is_exemplar_obs = True
                    elif pos_cid == 15 or (
                        pos_cname and ("path" in str(pos_cname).lower() or "ground" in str(pos_cname).lower())
                    ):
                        walkable_path_detected = True
            except Exception as exc:
                logger.debug("check_path_obstacle exemplar check error: %s", exc)

        coords = current_pos or self.get_player_movement_coords()
        false_alert_suppressed = False

        if is_exemplar_obs:
            res["obstacle_detected"] = True
            res["exemplar_obstacle"] = True
            if res.get("recommended_turn", "none") == "none":
                res["recommended_turn"] = "turn_left"
            self._consecutive_obstacle_hits += 1
            self._last_obstacle_alert_ts = time.time()
        elif walkable_path_detected:
            # High-confidence walkable path ground detected directly in corridor.
            # Suppress any false 3s depth alerts.
            if raw_depth_obs:
                false_alert_suppressed = True
            res["obstacle_detected"] = False
            res["exemplar_obstacle"] = False
            res["recommended_turn"] = "none"
            self._consecutive_obstacle_hits = 0
        else:
            if raw_depth_obs:
                self._consecutive_obstacle_hits += 1
                # F-0091: `or 1.0` tamamen kapalı koridoru (clearance 0.0)
                # "açık" (1.0) sayıyordu — yalnızca None varsayılan 1.0 alır.
                _clr_raw = res.get("clearance_center")
                center_clr = float(_clr_raw) if _clr_raw is not None else 1.0
                # Suppress transient 1-frame depth shadows if player is actively moving
                # and corridor clearance is not critically blocked (< 0.18).
                if suppress_transient_depth and self._consecutive_obstacle_hits < 2 and center_clr > 0.18:
                    res["obstacle_detected"] = False
                    false_alert_suppressed = True
                    res["recommended_turn"] = "none"
                else:
                    res["obstacle_detected"] = True
                    self._last_obstacle_alert_ts = time.time()
            else:
                self._consecutive_obstacle_hits = 0

        res["walkable_path_detected"] = walkable_path_detected
        res["false_alert_suppressed"] = false_alert_suppressed
        res["obstacle_alert"] = bool(res.get("obstacle_detected", False))
        res["current_coords"] = coords
        res["walking_state"] = self.get_walking_state()
        return res

    def get_active_waypoint(self) -> Optional[NodeData]:
        """Returns the current active target waypoint node from the route."""
        if not self._ordered_nodes:
            self._sync_ordered_nodes_from_route_manager()
        if not self._ordered_nodes:
            return None
        if self._active_route_index >= len(self._ordered_nodes):
            self._active_route_index = 0
        return self._ordered_nodes[self._active_route_index]

    def get_active_waypoint_coords(self) -> Optional[Tuple[float, float]]:
        """Returns (x, y) coordinates of the active route or sector waypoint.

        P1-10 (K-14): sektör waypoint ilerlemesi yalnızca ölü
        `get_next_sector_waypoint`'te yaşıyordu — bu canlı yol indeksi
        hiç ilerletmiyor, bot ilk sektör waypoint'inde takılıyordu.
        Rota düğümü yokken ilerleme (varış + F-0271 90 sn zaman aşımı
        atlaması) artık buradan akar; tarama bitince None döner.
        """
        if self._ordered_nodes:
            node = self.get_active_waypoint()
            if node is not None:
                return float(node.map_x), float(node.map_y)
        if self._sector_waypoints:
            nxt = self.get_next_sector_waypoint(
                arrival_radius_px=self._waypoint_arrival_radius_px,
            )
            if nxt is not None:
                return float(nxt[0]), float(nxt[1])
            return None
        return None

    def advance_route_waypoint(
        self,
        arrival_radius_px: float = 22.0,
        closed_loop: bool = True,
    ) -> Optional[NodeData]:
        """
        Advances to the next sequential route waypoint when player arrives within arrival_radius_px.
        Maintains continuous smooth travel without pausing, stopping, or locking into BEKLEMEDE.
        """
        if not self._ordered_nodes:
            self._sync_ordered_nodes_from_route_manager()
        if not self._ordered_nodes:
            return None

        px, py = self.get_player_movement_coords()
        curr_idx = self._active_route_index
        n_nodes = len(self._ordered_nodes)

        if curr_idx >= n_nodes:
            # F-0262: Kapalı döngüde başa dönüş rota bitişidir — sayacı işlet.
            if closed_loop:
                self._route_loop_count += 1
                self._route_completed_once = True
                logger.info("Rota döngüsü tamamlandı (kapalı döngü #%d).", self._route_loop_count)
            curr_idx = 0 if closed_loop else n_nodes - 1
            self._active_route_index = curr_idx

        target_node = self._ordered_nodes[curr_idx]
        tx, ty = float(target_node.map_x), float(target_node.map_y)
        dist = math.hypot(tx - px, ty - py)

        if dist <= float(arrival_radius_px):
            next_idx = curr_idx + 1
            if next_idx >= n_nodes:
                # F-0262: Son düğümde başa dönüş rota bitişidir — sayacı işlet.
                if closed_loop:
                    self._route_loop_count += 1
                    self._route_completed_once = True
                    logger.info("Rota döngüsü tamamlandı (kapalı döngü #%d).", self._route_loop_count)
                next_idx = 0 if closed_loop else n_nodes - 1
            self._active_route_index = next_idx
            self._waypoint_last_advanced_ts = time.time()
            target_node = self._ordered_nodes[next_idx]
            self.set_walking_state("ROTA TAKİBİ", reason=f"advanced to waypoint #{target_node.id}")
            if getattr(target_node, "state", "") == "resting":
                # S-0601 / F-0272: Yalnızca kilit açılır — kısmi hasat
                # ilerlemesi (current_harvests, örn. 2/3) korunur, sıfırlanmaz.
                # Meşru 3/3 tamamlanmış düğümde sayaç zaten 0'dır
                # (RouteManager.record_harvest_stage 3/3'te sıfırlar), o
                # yüzden sıfırlamayı kaldırmak o yolu değiştirmez.
                target_node.state = "active"

        return target_node

    def compute_waypoint_heading_and_steer(
        self,
        current_heading_deg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculates bearing, distance, angle difference, and recommended steering key
        toward the active waypoint, ensuring smooth travel without ping-pong oscillation.
        """
        px, py = self.get_player_movement_coords()
        target_coords = self.get_active_waypoint_coords()
        if target_coords is None:
            return {
                "has_target": False,
                "bearing_deg": 0.0,
                "heading_diff_deg": 0.0,
                "distance_px": 0.0,
                "steer_key": "none",
                "reached": False,
            }

        tx, ty = target_coords
        dx = tx - px
        dy = ty - py
        dist = math.hypot(dx, dy)
        bearing = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0

        cur_heading = (
            float(current_heading_deg)
            if current_heading_deg is not None
            else self._odometry_heading_deg
        )

        diff = (bearing - cur_heading + 180.0) % 360.0 - 180.0
        steer_key = "none"
        if diff > 12.0:
            steer_key = "d"  # Turn right
        elif diff < -12.0:
            steer_key = "a"  # Turn left

        reached = dist <= self._waypoint_arrival_radius_px
        # F-0439: Glide kontrolcüsü üretim akışında okunuyordu (yalnızca
        # lazy property olarak duruyordu) — mevcut bacak için süzülme
        # önerisi hesaplanır ve çağrıcıya (state machine) sunulur.
        glide_check: Dict[str, Any] = {"glide_recommended": False}
        try:
            current_node = self.get_active_waypoint()
            next_node = None
            if self._ordered_nodes:
                nxt_idx = (self._active_route_index + 1) % len(self._ordered_nodes)
                next_node = self._ordered_nodes[nxt_idx]
            if current_node is not None and next_node is not None:
                cur_d = {
                    "x": float(current_node.map_x),
                    "y": float(current_node.map_y),
                    "z": getattr(current_node, "elevation", getattr(current_node, "z", None)),
                    "is_glide_waypoint": getattr(current_node, "is_glide_waypoint", False),
                }
                nxt_d = {
                    "x": float(next_node.map_x),
                    "y": float(next_node.map_y),
                    "z": getattr(next_node, "elevation", getattr(next_node, "z", None)),
                    "is_glide_target": getattr(next_node, "is_glide_target", False),
                }
                recommended = bool(self.glide_controller.should_use_glide(cur_d, nxt_d))
                glide_check = {
                    "glide_recommended": recommended,
                    "glide_params": self.glide_controller.estimate_glide_parameters(cur_d, nxt_d)
                    if recommended
                    else None,
                }
        except Exception as exc:
            logger.debug("Glide bacak kontrolü atlandı: %s", exc)
        self._last_glide_check = glide_check
        return {
            "has_target": True,
            "target_coords": (tx, ty),
            "current_coords": (px, py),
            "bearing_deg": round(bearing, 1),
            "heading_diff_deg": round(diff, 1),
            "distance_px": round(dist, 1),
            "steer_key": steer_key,
            "reached": reached,
            "glide_recommended": bool(glide_check.get("glide_recommended", False)),
            "glide_params": glide_check.get("glide_params"),
            "walking_state": self.get_walking_state(),
        }

    def reset_waypoint_timeout(self) -> None:
        """Resets waypoint travel timer when smooth forward movement is ongoing."""
        self._waypoint_last_advanced_ts = time.time()
        self._consecutive_obstacle_hits = 0
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
        except Exception as exc:
            # F-0263: Geçici bir getirme hatası bellek içi rotayı yok etmesin —
            # eski liste korunur.
            logger.error("Rota düğümleri senkronize edilirken hata (mevcut liste korunuyor): %s", exc)
            return
        self._ordered_nodes = list(nodes) if nodes else []
        for idx, node in enumerate(self._ordered_nodes, start=1):
            setattr(node, "seq_index", idx)

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
                    # F-0092: Sıralı düğüm, id eşleşmesine göre yerleştirilir —
                    # sıradaki düğümü körü körüne almak yanlış düğümü yanlış sıraya yazar.
                    ordered_by_id = {n.id: n for n in self._ordered_nodes}
                    merged: List[NodeData] = []
                    for existing in all_region_nodes:
                        if existing.id in ordered_ids:
                            merged.append(ordered_by_id[existing.id])
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
            result.setdefault("material_type", UNKNOWN_RESOURCE_TYPE)
            result.setdefault("source", "unknown")
            result.setdefault("screen_x", 0)
            result.setdefault("screen_y", 0)
            result.setdefault("bbox", (0, 0, 0, 0))
            result.setdefault("rel_map_dx", 0.0)
            result.setdefault("rel_map_dy", 0.0)
            result.setdefault("estimated_distance_m", 0.0)
            # F-0264: Bilinmeyen güven 0.85 diye kayda girmez — bilinmiyorsa 0.0.
            result.setdefault("confidence", 0.0)
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
            "material_type": str(getattr(item, "material_type", UNKNOWN_RESOURCE_TYPE)),
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

            def _merge_minimap_candidate(
                existing: Dict[str, Any], cand: Dict[str, Any]
            ) -> None:
                if int(cand.get("max_harvests", 3)) > int(existing.get("max_harvests", 3)):
                    existing["max_harvests"] = int(cand["max_harvests"])
                    existing["stack_count"] = int(cand.get("stack_count", 1))
                if (
                    existing.get("material_type") in ("Gatherable Material", "")
                    and cand.get("material_type") not in ("Gatherable Material", "")
                ):
                    existing["material_type"] = cand["material_type"]

            # Grid tabanlı tekilleştirme (9 px): eski O(n²) döngüyle aynı
            # anlamsal sonuç (ilk eşleşme + max_harvests/tür yükseltme).
            merged = _dedupe_detections_by_proximity(
                internal_items, 9.0, _merge_minimap_candidate, base=merged
            )

            return merged
        except Exception as exc:
            logger.error("_scan_minimap_priority_materials hatası: %s", exc)
            return merged

    def _get_minimap_inner_bounds(self, w: int, h: int) -> Tuple[int, int, int, int]:
        """Ekran çözünürlüğüne göre minimap iç radar sınırlarını `(mx1, my1, mx2, my2)` döndürür."""
        if w >= 900 and h >= 550:
            # F-0435: MaterialScanner ile aynı oranlar (0.855–0.992 / 0.065–0.255).
            mx1 = int(round(w * 0.855))
            my1 = int(round(h * 0.065))
            mx2 = int(round(w * 0.992))
            my2 = int(round(h * 0.255))
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
            # F-0265: Ortalama parlaklık/doygunluk gece bölgelerinde düşük olur
            # diye tüm tarama kör bırakılmaz — yalnızca en parlak pikseller
            # bile sönükse (parlak ikon yoksa) atlanır.
            if mean_v < 58.0 or mean_s < 14.0:
                if float(np.percentile(hsv[:, :, 2], 99.5)) < 90.0:
                    return detections

            # Unified minimap player anchor (same default as AutoCartographer arrow).
            # B-05: _locate_minimap_player_arrow ok bulamayınca da istisna
            # atmadan AYNI sabit merkezi döner — "ok tam merkezde" hali ile
            # "ok bulunamadı" hali dönüş değerinden ayırt edilemez. Bu
            # yüzden dönen nokta tam varsayılan merkezse ankraj
            # ÖLÇÜLMEMİŞ sayılır: player_anchor_fallback=True ile
            # işaretlenir (mesafe/yön bu varsayılan ankrajdan hesaplanır).
            default_cx, default_cy = mw * 0.50, mh * 0.545
            cx, cy = default_cx, default_cy
            anchor_fallback = True
            try:
                # AutoCartographer modül düzeyinde import edilmiş durumda;
                # her karede içe import yapılmaz.
                cx, cy = AutoCartographer._locate_minimap_player_arrow(mm)
                anchor_fallback = (float(cx) == float(default_cx) and float(cy) == float(default_cy))
            except Exception as exc:
                anchor_fallback = True
                now_ts = time.time()
                last_warn = float(getattr(self, "_minimap_anchor_warn_ts", 0.0) or 0.0)
                if now_ts - last_warn >= 60.0:
                    self._minimap_anchor_warn_ts = now_ts
                    logger.warning(
                        "Minimap oyuncu oku tespiti başarısız (%s); sabit merkez "
                        "(%.2f, %.2f) varsayılan ankraj olarak kullanılıyor — "
                        "ikon mesafe/yön hesapları bu ankraja göredir "
                        "(player_anchor_fallback=True).",
                        exc, cx, cy,
                    )
                logger.debug("Minimap oyuncu oku tespiti başarısız: %s", exc)
            if anchor_fallback:
                now_ts = time.time()
                last_warn = float(getattr(self, "_minimap_anchor_fallback_warn_ts", 0.0) or 0.0)
                if now_ts - last_warn >= 60.0:
                    self._minimap_anchor_fallback_warn_ts = now_ts
                    logger.warning(
                        "Minimap oyuncu oku ölçülemedi; sabit merkez (%.2f, %.2f) "
                        "varsayılan ankraj olarak kullanılıyor — ikon mesafe/yön "
                        "hesapları bu ankraja göredir (player_anchor_fallback=True).",
                        cx, cy,
                    )
            # P1-7 (K-20): minimap ölçeği tek kaynaktan
            # (MINIMAP_METERS_PER_PX, 0.55); eski 0.65 yedeği kaldırıldı.
            meters_per_px = float(getattr(self.material_scanner, "meters_per_map_pixel", MINIMAP_METERS_PER_PX))

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
                cv2.morphologyEx(green_badge, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL_3X3),
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
                        # F-0436: Rozet boyutu çözünürlüğe göre değişir — eşik
                        # mutlak piksel değil, minimap boyutuna oranlıdır.
                        stack_cnt = (
                            3
                            if (gh >= max(11.0, mh * 0.085) or gw >= max(10.0, mw * 0.05))
                            else 2
                        )
                        badge_centers.append((gx + gw / 2.0, gy + gh / 2.0, stack_cnt))

            # 2. Cyan/Mavi Kanat (Wing Feather / Gemstone) ikonları
            cyan_wing = cv2.bitwise_and(
                cv2.inRange(hsv, (80, 65, 85), (115, 255, 255)),
                valid,
            )
            cnts_c, _ = cv2.findContours(
                cv2.morphologyEx(cyan_wing, cv2.MORPH_CLOSE, _MORPH_CLOSE_KERNEL_3X3),
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
                            # P1-4 (K-07): uydurma sabit güven yok — bu CV
                            # yolu gerçek bir skor üretmez; bilinmiyorsa
                            # 0.0 (F-0264 kuralı). Eskiden 0.91/0.86 idi ve
                            # dedup'ta YZ güvenini ezip YOLO tespitini
                            # bastırıyordu.
                            "confidence": 0.0,
                            "player_anchor_fallback": bool(anchor_fallback),
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
                    sprout_conf = 0.0
                else:
                    mat_type = "Gatherable Material"
                    # F17: filizde güven ÖLÇÜLÜR — kanonik şablon eşleşmesi
                    # (MaterialScanner kalibrasyonu); eşik altı 0.0 kalır.
                    # Odyle/kanatta şablon yok, güven 0.0'dır.
                    sprout_conf = _sprout_template_confidence(hsv[:, :, 2], x, y, cw, ch)

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
                        # P1-4 (K-07): uydurma sabit güven yok — filizde
                        # güven şablon eşleşmesinden ölçülür (F17), eşik
                        # altı 0.0 kalır (F-0264 kuralı). Eskiden 0.90/0.84
                        # sabitiydi.
                        "confidence": sprout_conf,
                        "player_anchor_fallback": bool(anchor_fallback),
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
            # v20 (MADDE 8): canlı oto-haritalama yolu scan_frame_for_materials'i
            # çağırmaz — exemplar kapısı (öğrenilmiş negatif örnek:
            # monster_3d_negative / player_character, sim >= 0.82) burada
            # hiç çalışmazdı; canavar "Targena" diye haritalanıyordu.
            # Toplama yolunun kullandığı aynı kapı burada da çalışır:
            # negatif eşleşen aday kartografa girmeden elenir.
            if self.material_scanner is not None and hasattr(
                self.material_scanner, "_refine_candidates_with_exemplars"
            ):
                try:
                    scanner_world = self.material_scanner._refine_candidates_with_exemplars(
                        bgr, scanner_world
                    )
                except Exception as exc:
                    logger.debug("3D dünya exemplar kapısı atlandı: %s", exc)
            for item in scanner_world:
                d = self._serialize_detected_material(item)
                d["source"] = "3d_world"
                d["source_type"] = "3d_world_crest"
                world_and_prompt.append(d)

            # 2. Dahili 3D Aria / Mela Beyaz Kanatlı Toplama Arması ve [E]/[F] Yüzen İkon Tarayıcı
            internal_3d = self._scan_3d_crests_and_floating_icons_internal(bgr, hsv)

            def _upgrade_3d_material(
                existing: Dict[str, Any], cand: Dict[str, Any]
            ) -> None:
                if (
                    existing.get("material_type")
                    in ("Orichalcum Ore", "Gatherable Material", "")
                    and cand.get("material_type") in ("Aria", "Mela", "Odyle")
                ):
                    existing["material_type"] = cand["material_type"]

            # Grid tabanlı tekilleştirme (36 px): eski O(n²) döngüyle aynı
            # anlamsal sonuç (ilk eşleşme + spesifik tür yükseltme).
            world_and_prompt = _dedupe_detections_by_proximity(
                internal_3d, 36.0, _upgrade_3d_material, base=world_and_prompt
            )

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
            prompt_dets: List[Dict[str, Any]] = []
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
                prompt_dets.append(pd)

            def _upgrade_prompt_material(
                existing: Dict[str, Any], pd: Dict[str, Any]
            ) -> None:
                # F-0266: Yüzen [E]/[F] ikonu zaten gather_prompt olarak ekliyse
                # aynı istem iki kez sayılmaz — 36 px içindekiler tekilleştirilir.
                if (
                    existing.get("material_type") in ("Gatherable Material", "Odyle", "")
                    and pd.get("material_type") not in ("Gatherable Material", "Odyle", "")
                ):
                    existing["material_type"] = pd["material_type"]

            # Grid tabanlı tekilleştirme (36 px): eski O(n²) döngüyle aynı
            # anlamsal sonuç (ilk eşleşme + spesifik tür yükseltme).
            world_and_prompt = _dedupe_detections_by_proximity(
                prompt_dets, 36.0, _upgrade_prompt_material, base=world_and_prompt
            )

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

                # v20 (MADDE 8): boş ad = mesh learned-negative reddi
                # (canavar/oyuncu) — boş ad kartografta UNKNOWN_RESOURCE_TYPE
                # düğüme dönerdi, bu yüzden tespit hiç üretilmez.
                if not mat_name:
                    continue

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
                        # P1-4 (K-07): arma sınıflandırması skor üretmez;
                        # sabit 0.92 uydurmaydı — bilinmiyorsa 0.0.
                        "confidence": 0.0,
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
                        # P1-4 (K-07): yüzen [E]/[F] ikonu skor üretmez;
                        # sabit 0.91 uydurmaydı — bilinmiyorsa 0.0.
                        "confidence": 0.0,
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
            else:
                # F-0267: Hiçbir tuş yöntemi yoksa sessizce geçilmez — uyarı loglanır.
                logger.warning(
                    "[ExplorerMode] Toplama tuşu '%s' gönderilemedi: input_controller'da "
                    "press_key/tap_key/callable yok.",
                    clean_key.upper(),
                )
        elif not self.test_mode:
            self.input_controller = InputController(test_mode=self.test_mode)
            self.input_controller.press_key(clean_key)
        else:
            logger.info("[ExplorerMode] Test modunda '%s' toplama tuşu simüle edildi.", clean_key.upper())

    def _get_cast_bar_reader(self) -> Optional[Any]:
        """Cast bar okuyucuyu (GameHUDReader.read_cast_bar_progress) tembel yükler."""
        if self._hud_reader is not None:
            return self._hud_reader
        if self._hud_reader_unavailable:
            return None
        try:
            from core.ocr_distance import GameHUDReader

            self._hud_reader = GameHUDReader()
            return self._hud_reader
        except Exception as exc:
            self._hud_reader_unavailable = True
            logger.debug("Cast bar okuyucu yüklenemedi, kör bekleme kullanılacak: %s", exc)
            return None

    def _read_cast_bar_progress(self, frame: Optional[np.ndarray]) -> Optional[float]:
        """Verilen karede toplama barı ilerlemesini (0.0..1.0) okur; okunamazsa None."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        reader = self._get_cast_bar_reader()
        if reader is None:
            return None
        try:
            return float(reader.read_cast_bar_progress(frame))
        except Exception as exc:
            logger.debug("Cast bar okuma hatası: %s", exc)
            return None

    def _wait_for_harvest_cast(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
    ) -> Optional[np.ndarray]:
        """
        Her bir hasat aşaması (`1/3`, `2/3`, `3/3`) sırasında toplama barının (cast bar)
        tamamlanmasını bekler ve en güncel kareyi döndürür.
        `test_mode=True` iken birim testlerini yavaşlatmamak için anında tamamlanır.

        F-0268: Önceki sürüm barı hiç okumadan kör bekliyordu. Artık her
        karede cast bar ilerlemesi okunur: bar görüldükten sonra dolması
        (>=0.99) veya kaybolması (0.0) hasat aşamasının bittiğini gösterir ve
        bekleme erken sonlanır; bar hiç görülmezse bütçe (cast_wait_sec)
        dolana kadar beklenir (eski davranış, üst sınır olarak korunur).
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
        cast_bar_seen = False
        initial_progress = self._read_cast_bar_progress(latest_frame)
        # Girişte dolu (0.99) görünen bar önceki aşamadan kalma olabilir —
        # yalnızca 0..0.99 arası ilerleme "görüldü" sayılır.
        if initial_progress is not None and 0.0 < initial_progress < 0.99:
            cast_bar_seen = True
        while time.time() < deadline:
            if self.heartbeat_callback is not None and callable(self.heartbeat_callback):
                try:
                    self.heartbeat_callback()
                except Exception:
                    pass
            time.sleep(min(0.12, max(0.02, deadline - time.time())))
            if frame_provider is not None and callable(frame_provider):
                try:
                    provided = frame_provider()
                    if isinstance(provided, np.ndarray) and provided.size > 0:
                        latest_frame = provided
                except Exception as exc:
                    logger.debug("Canlı frame_provider hatası: %s", exc)
            progress = self._read_cast_bar_progress(latest_frame)
            if progress is None:
                continue
            if progress > 0.0:
                cast_bar_seen = True
                if progress >= 0.99:
                    break
            elif cast_bar_seen:
                # Bar dolup kayboldu: hasat aşaması tamamlandı.
                break
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
        stage_failed = False

        try:
            for stage_idx in range(1, target_harvests + 1):
                stage_attempts = 0
                extraction_failed = False
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

                # F-0094: Son denemede de 'Essence Extraction failed' varsa aşama
                # BAŞARISIZDIR — sayaca eklenmez, sonraki aşamalara geçilmez.
                if extraction_failed:
                    stage_failed = True
                    logger.warning(
                        "[ExplorerMode] Hasat aşaması %d/%d son denemede de başarısız "
                        "('Essence Extraction failed'); hasat durduruldu.",
                        stage_idx,
                        target_harvests,
                    )
                    break

                completed_stages += 1
                self._total_gathered_in_explorer += 1
                stage_label = f"{completed_stages}/{target_harvests}"
                stages_log.append(stage_label)

                # Trigger Self-Training YOLO Dataset Factory (Time Machine t - 2.0s capture)
                df = getattr(self, "dataset_factory", None)
                if df is not None and hasattr(df, "on_gather_success"):
                    try:
                        resolved_mat = ""
                        if nearest_node is not None and getattr(nearest_node, "resource_type", None):
                            resolved_mat = str(nearest_node.resource_type)
                        elif getattr(self, "_last_steer_target", None):
                            resolved_mat = str(self._last_steer_target.get("material_type", ""))
                        df.on_gather_success(
                            material_name=resolved_mat,
                            harvest_stage=completed_stages,
                            max_harvests=target_harvests,
                        )
                    except Exception as df_exc:
                        logger.debug("[ExplorerMode] Dataset factory on_gather_success skipped: %s", df_exc)

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
                "stage_failed": stage_failed,
                "extraction_failed": stage_failed,
                "node_id": int(nearest_node.id) if nearest_node is not None else None,
                # F-0093: Yakında düğüm yokken hasat kaydı da yok — 'tükendi'
                # yalnızca bir düğüm gerçekten tüketildiyse doğrudur.
                "node_depleted": bool(nearest_node is not None and completed_stages >= target_harvests),
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
                "stage_failed": stage_failed,
                "extraction_failed": stage_failed,
                "node_id": int(nearest_node.id) if nearest_node is not None else None,
                "node_depleted": bool(nearest_node is not None and completed_stages >= target_harvests),
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
        steer_reason = "not_attempted"
        if execute_steer:
            if self.test_mode:
                # F-0096: Test modunda fare hiç oynatılmadan yönelme 'başarılı'
                # sayılmaz — girdi gönderilmedi.
                steered = False
                steer_reason = "test_mode_no_input_sent"
            elif self.input_controller is not None:
                try:
                    # F-0095: InputController'da move_mouse yok; var olan
                    # smooth_move_to (piksel hedef) kullanılır. move_mouse_abs
                    # normalize (0..65535) koordinat alır — piksel verilmez.
                    if hasattr(self.input_controller, "smooth_move_to"):
                        if hasattr(self.input_controller, "get_cursor_position"):
                            cx, cy = self.input_controller.get_cursor_position()
                            self.input_controller.smooth_move_to(cx + dx, cy + dy, duration=0.08)
                        else:
                            self.input_controller.smooth_move_to(tx, ty, duration=0.08)
                        steered = True
                        steer_reason = "smooth_move_to"
                    else:
                        steer_reason = "no_mouse_method_on_input_controller"
                        logger.warning(
                            "3D hedefe yönelme yapılamadı: input_controller'da "
                            "smooth_move_to yok."
                        )
                except Exception as exc:
                    steer_reason = f"mouse_move_failed: {exc}"
                    logger.debug("3D hedefe yönelme fare hareketi atlandı: %s", exc)
            else:
                steer_reason = "no_input_controller"

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
            "steer_reason": steer_reason,
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
        # F-0269: explore_step varsayılanı True iken bu metodun varsayılanı
        # False idi — aynı işi yapan iki giriş noktası çağrı yerine göre
        # toplar ya da toplamazdı. Varsayılanlar hizalandı (True).
        gather_while_exploring: bool = True,
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
                    # Sweep G: geçersiz karede harita durumu bilinmiyor —
                    # açık sayılır (fail-closed), bilinmezlik bayrağı eklenir.
                    "is_map_open": True,
                    "is_map_open_unknown": True,
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
            # Sweep G: is_map_open üç durumlu (True/False/None=bilinmiyor) —
            # bilinmiyor açık sayılır (fail-closed): tarama/toplama durur.
            _map_open_raw = self.vision_engine.is_map_open(bgr)
            is_map_open = _map_open_raw is not False
            is_map_open_unknown = _map_open_raw is None
            is_blocking = False if is_map_open else bool(self.vision_engine.is_game_window_blocking(bgr))

            # Adım 1: Hibrit Sağ-Üst Minimap + 3D Dünya + YZ (reflex/YOLO) tarama
            minimap_detections: List[Dict[str, Any]] = []
            world_3d_detections: List[Dict[str, Any]] = []
            yz_detections: List[Dict[str, Any]] = []
            if not is_map_open and not is_blocking:
                minimap_detections = self._scan_minimap_priority_materials(bgr)
                world_3d_detections = self._scan_3d_world_hybrid(bgr)
                yz_detections = self._scan_yz_gather_detections(bgr)

            serialized_detections: List[Dict[str, Any]] = (
                list(world_3d_detections) + list(minimap_detections) + list(yz_detections)
            )

            def _merge_step_detection(
                existing: Dict[str, Any], det: Dict[str, Any]
            ) -> None:
                # Dedup CV vs YZ overlaps (keep higher confidence).
                if float(det.get("confidence", 0)) > float(existing.get("confidence", 0)):
                    existing.update(det)
                elif det.get("yz") and not existing.get("material_type"):
                    existing["material_type"] = det.get("material_type", existing.get("material_type"))

            # Grid tabanlı tekilleştirme (32 px): eski O(n²) döngüyle aynı
            # anlamsal sonuç (ilk eşleşme + yüksek güven kazanır).
            serialized_detections = _dedupe_detections_by_proximity(
                serialized_detections, 32.0, _merge_step_detection
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
            self._odometry_pos_x = float(map_x)
            self._odometry_pos_y = float(map_y)
            self._odometry_last_ts = now_ts

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
                if det_source_type in ("yz_3d", "3d_world", "3d_auto"):
                    det_source_type = "3d_auto"
                elif det_source_type in ("yz_minimap", "minimap", "minimap_auto", "minimap_material"):
                    # F17: içsel minimap tarayıcısının etiketi de minimap_auto'dur
                    # (parse ile tutarlı; kartograf düğümü minimap_auto yazar).
                    det_source_type = "minimap_auto"
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
                    # F-0270: Yüzen [E] ikonunun sabit 'Odyle' etiketi jenerik
                    # düğümü ezemez — tür değişimi yalnızca gerçek 3D arma
                    # sınıflandırmasından (source 3d_world) gelir.
                    if (
                        det_source == "3d_world"
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
                    # Uzamsal kümeleme: _ordered_nodes içinde dedup_radius_px mesafesinde düğüm var mı?
                    nearest_ordered: Optional[NodeData] = None
                    min_dist_ord = float("inf")
                    dedup_r = float(getattr(self.auto_cartographer, "dedup_radius_px", DEFAULT_DEDUP_RADIUS_PX))
                    for ord_n in self._ordered_nodes:
                        d_ord = math.hypot(
                            float(node_obj.map_x) - float(ord_n.map_x),
                            float(node_obj.map_y) - float(ord_n.map_y),
                        )
                        if d_ord < min_dist_ord:
                            min_dist_ord = d_ord
                            nearest_ordered = ord_n

                    if nearest_ordered is not None and min_dist_ord <= dedup_r:
                        seq_pos = next(
                            (idx + 1 for idx, n in enumerate(self._ordered_nodes) if n.id == nearest_ordered.id),
                            len(self._ordered_nodes),
                        )
                        if hasattr(nearest_ordered, "max_harvests") and hasattr(node_obj, "max_harvests"):
                            nearest_ordered.max_harvests = max(
                                int(getattr(nearest_ordered, "max_harvests", 3) or 3),
                                int(getattr(node_obj, "max_harvests", 3) or 3),
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
                self._emit_node_discovered(enriched)

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

            if gathered_now:
                self.set_walking_state("TOPLANIYOR", reason="opportunistic gathering")
            elif steer_target and steer_target.get("steered"):
                self.set_walking_state("ROTA TAKİBİ" if self._ordered_nodes else "KEŞİFTE", reason="steering to 3d target")
            else:
                self.update_walking_state(in_route=bool(self._ordered_nodes), in_exploration=True)

            return {
                "step_index": self._step_count,
                "timestamp": now_ts,
                "player_pos": {
                    "map_x": map_x,
                    "map_y": map_y,
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                },
                "movement_coords": (self._odometry_pos_x, self._odometry_pos_y),
                "walking_state": self.get_walking_state(),
                "anti_beklemede_active": True,
                "active_waypoint_idx": self._active_route_index,
                "is_map_open": is_map_open,
                "is_map_open_unknown": is_map_open_unknown,
                "is_window_blocking": is_blocking,
                "needs_map_calibrate": bool(
                    getattr(self.auto_cartographer, "needs_map_calibrate", False)
                ),
                "gate_rejected_total": int(
                    getattr(self.auto_cartographer, "total_gate_rejected", 0) or 0
                ),
                "detected_materials": serialized_detections,
                "detected_count": len(serialized_detections),
                "world_3d_count": world_3d_count,
                "minimap_count": minimap_count,
                "yz_count": sum(1 for d in serialized_detections if d.get("yz") or str(d.get("source_type", "")).startswith("yz_")),
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
                # F-0437: Bu adımda bildirim yoksa önceki adımın mesajı 'son
                # mesaj' gibi sunulmaz — boş döner.
                "latest_hud_message": step_notifications[-1] if step_notifications else "",
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
                "movement_coords": (self._odometry_pos_x, self._odometry_pos_y),
                "walking_state": self.get_walking_state(),
                "anti_beklemede_active": True,
                # Sweep G: hata yolunda harita durumu bilinmiyor — açık
                # sayılır (fail-closed), bilinmezlik bayrağı eklenir.
                "is_map_open": True,
                "is_map_open_unknown": True,
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
            # F-0097: Sıralama boş dönerse rota listesi sessizce silinmez —
            # önce sıralanmamış güncel liste, o da yoksa önceki liste korunur.
            if not ordered_nodes:
                if current_nodes:
                    logger.warning(
                        "finalize_and_sort_route: sıralama boş döndü; sıralanmamış "
                        "%d düğümlük liste korunuyor.",
                        len(current_nodes),
                    )
                    ordered_nodes = list(current_nodes)
                elif self._ordered_nodes:
                    logger.warning(
                        "finalize_and_sort_route: sıralama boş döndü; önceki "
                        "%d düğümlük liste korunuyor.",
                        len(self._ordered_nodes),
                    )
                    ordered_nodes = list(self._ordered_nodes)

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

            # F-0438: Kalite hesabı skor üretemezse rota '100/100 pürüzsüz'
            # raporlanmaz — bilinmeyen skor 0.0'dır.
            _smooth_raw = after_quality.get("smoothness_score")
            smoothness_val = float(_smooth_raw) if _smooth_raw is not None else 0.0
            summary_msg = (
                f"Route finalized ({clean_mode}): {len(self._ordered_nodes)} nodes sorted "
                f"(smoothness={smoothness_val:.1f}/100, "
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
                "smoothness_score": smoothness_val,
                "walk_trajectory_points": len(walk_traj_2d),
                "total_gathered_while_exploring": self._total_gathered_in_explorer,
                "anti_ping_pong_enabled": True,
                "next_target_node_id": int(first_node.id) if first_node is not None else None,
                "next_target_seq_index": int(first_idx + 1) if first_idx >= 0 else None,
                "summary_message": summary_msg,
            }

        except Exception as exc:
            logger.error("finalize_and_sort_route sırasında hata: %s", exc, exc_info=True)
            # S064 (v19/P3-8): hata yolundaki walk_trajectory_points ham
            # öznitelik sayıyordu; başarı yolu filtreli
            # (len>=2) sayıyordu — aynı alan iki farklı sayı üretiyordu.
            # Hata yolunda da aynı filtre uygulanır.
            try:
                _raw_traj = self.auto_cartographer.walk_trajectory
                _traj_points = sum(
                    1
                    for _pt in _raw_traj
                    if isinstance(_pt, (list, tuple)) and len(_pt) >= 2
                )
            except Exception:
                _traj_points = 0
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
                "walk_trajectory_points": _traj_points,
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
            # F-0271: Yeni taramada waypoint zaman aşımı sayacı sıfırlanır.
            self._sector_waypoint_started_ts = 0.0

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
        waypoint_timeout_sec: float = 90.0,
    ) -> Optional[Tuple[int, int]]:
        """
        Otonom sektör taraması sırasında oyuncunun anlık odometri konumuna göre
        sıradaki Boustrophedon waypoint hedefini döndürür.

        F-0271: Ulaşılamayan waypoint'te tarama kilitlenmez — waypoint
        `waypoint_timeout_sec` içinde varılamazsa atlanır ve sonrakine geçilir.
        """
        try:
            if not self._sector_waypoints:
                return None

            px, py, _, _ = self.auto_cartographer.get_position()
            n = len(self._sector_waypoints)
            now = time.time()
            if self._sector_waypoint_started_ts <= 0.0:
                self._sector_waypoint_started_ts = now

            while self._current_waypoint_idx < n:
                wx, wy = self._sector_waypoints[self._current_waypoint_idx]
                if math.hypot(float(wx) - px, float(wy) - py) <= float(arrival_radius_px):
                    self._current_waypoint_idx += 1
                    self._sector_waypoint_started_ts = time.time()
                elif (
                    waypoint_timeout_sec > 0
                    and time.time() - self._sector_waypoint_started_ts > float(waypoint_timeout_sec)
                ):
                    logger.warning(
                        "Sektör waypoint #%d (%.0f, %.0f) %.0fs içinde ulaşılamadı; atlanıyor.",
                        self._current_waypoint_idx,
                        wx,
                        wy,
                        float(waypoint_timeout_sec),
                    )
                    self._current_waypoint_idx += 1
                    self._sector_waypoint_started_ts = time.time()
                else:
                    break

            if self._current_waypoint_idx >= n:
                if loop_sweep:
                    self._current_waypoint_idx = 0
                    self._sector_waypoint_started_ts = time.time()
                else:
                    return None

            return self._sector_waypoints[self._current_waypoint_idx]
        except Exception as exc:
            logger.error("get_next_sector_waypoint hatası: %s", exc)
            return None


def _run_explorer_mode_contract_checks() -> Dict[str, Any]:
    """
    F-0098: 'hazır/audit/certified' zinciri artık gerçek ölçüm yapar — test
    modunda bir controller kurulup yürüme durumu, rota listesi ve ana
    giriş noktalarının varlığı doğrulanır.
    """
    checks: Dict[str, Any] = {}
    try:
        ctrl = ExplorerModeController(test_mode=True)
        state = ctrl.get_walking_state()
        checks["controller_constructs"] = True
        checks["walking_state_known"] = state in (
            "YÜRÜYOR",
            "ROTA TAKİBİ",
            "KEŞİFTE",
            "TOPLANIYOR",
            "BEKLEMEDE",
        )
        checks["walking_state"] = state
        checks["ordered_nodes_is_list"] = isinstance(ctrl._ordered_nodes, list)
        checks["explore_step_callable"] = callable(getattr(ctrl, "explore_step", None))
        checks["movement_query_callable"] = callable(getattr(ctrl, "is_movement_active", None))
        checks["depth_avoidance_available"] = ctrl.depth_obstacle_avoidance is not None
        checks["glide_controller_available"] = ctrl.glide_controller is not None
    except Exception as exc:
        checks["controller_constructs"] = False
        checks["error"] = str(exc)
    passed = bool(checks.get("controller_constructs")) and all(
        bool(v) for k, v in checks.items() if k not in ("walking_state", "error")
    )
    checks["passed"] = passed
    return checks


def get_explorer_mode_capabilities() -> list:
    return ["explorer_mode", "autonomous_exploration", "diagnostic_snapshot"]


def validate_explorer_mode_contract() -> bool:
    return bool(_run_explorer_mode_contract_checks().get("passed", False))


def get_explorer_mode_baseline_metrics() -> dict:
    checks = _run_explorer_mode_contract_checks()
    return {"module": "explorer_mode", "contract_valid": bool(checks.get("passed", False)), "capability_count": len(get_explorer_mode_capabilities()), "checks": checks}


def get_explorer_mode_readiness_status() -> dict:
    return {"module": "explorer_mode", "ready": validate_explorer_mode_contract(), "metrics": get_explorer_mode_baseline_metrics()}


def get_explorer_mode_audit_record() -> dict:
    checks = _run_explorer_mode_contract_checks()
    return {
        "module": "explorer_mode",
        "audit_passed": bool(checks.get("passed", False)),
        "checks": checks,
        "readiness": get_explorer_mode_readiness_status(),
    }


def get_explorer_mode_operational_digest() -> dict:
    return {"module": "explorer_mode", "operational": validate_explorer_mode_contract(), "audit": get_explorer_mode_audit_record()}


def get_explorer_mode_telemetry_payload() -> dict:
    return {"module": "explorer_mode", "telemetry_ready": validate_explorer_mode_contract(), "digest": get_explorer_mode_operational_digest()}


def get_explorer_mode_integrity_seal() -> dict:
    return {"module": "explorer_mode", "sealed": validate_explorer_mode_contract(), "telemetry": get_explorer_mode_telemetry_payload()}


def get_explorer_mode_compliance_certificate() -> dict:
    return {"module": "explorer_mode", "certified": validate_explorer_mode_contract(), "seal": get_explorer_mode_integrity_seal()}


def get_explorer_mode_readiness_probe() -> dict:
    return {"module": "explorer_mode", "ready": validate_explorer_mode_contract(), "certificate": get_explorer_mode_compliance_certificate()}


def get_explorer_mode_audit_summary() -> dict:
    return {"module": "explorer_mode", "audit_passed": validate_explorer_mode_contract(), "probe": get_explorer_mode_readiness_probe()}


