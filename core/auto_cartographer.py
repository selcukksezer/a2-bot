# -*- coding: utf-8 -*-
"""
Sürekli SLAM, Görsel Odometri ve 3-Kaynaklı (Minimap + 3D Crest + F Gather)
Otomatik Materyal Haritalandırıcı (core/auto_cartographer.py).

Kullanıcının haritada gezerken tek tek manuel haritalandırmayla (F8 veya M harita
aç-kapa ile) uğraşmaması için, oyuncu gezdikçe gördüğü tüm gemleri, bitkileri
(`Aria`, `Mela`) ve materyalleri (`Orichalcum Ore`, `Odyle`, 3 yapraklı filizler)
arka planda kesintisiz haritalandıran ve her kaynağın 3 kez toplanabilme
(`max_harvests = 3 * stack_count`) kapasitesini takip eden AutoCartographer motoru.

Çalışma Prensibi:
CANLI YOL (H-sweep, dilim 13 Y5): üretimde çalışan tek kayıt yolu
`process_frame_discoveries`'dir (explorer_mode.py:2851 çağırır).
`record_from_minimap` / `record_from_3d_world` / `record_harvest_event` /
`ingest_detections` üretimde çağrılmaz — yalnızca testler ve eski
çağıranlar için korunan legacy giriş noktalarıdır; docstring'lerindeki
"3-Kaynaklı" anlatım canlı mimariyi değil, bu legacy yolları anlatır.
Bu dört fonksiyondaki düzeltmeler üretime etki etmez.
1. Real-Time Minimap Odometry (`VisualOdometryTracker` — Dead-Reckoning + Minimap SLAM):
   - Sağ üstteki Minimap bölgesini (`Pilgrim's Ridge`, `x: 0.855..0.992, y: 0.065..0.255`)
     her karede izler.
   - Ardışık minimap kareleri arasındaki doku kaymasını (`cv2.phaseCorrelate` ve
     optik akış / ORB özellik eşleştirme ile) hesaplayarak oyuncunun harita açmasına
     gerek kalmadan dünya/bölge haritası üzerindeki konumunu `(map_x, map_y, norm_x, norm_y)`
     sürekli günceller.
   - Dünya haritası (`M`) açık olduğunda konumunu mutlak harita üzerinde sıfır hatayla
     kalibre eder (`calibrate_on_world_map`).
   - Oyuncunun gezdiği ayak izini (`walk_trajectory: list[tuple[float, float, float]]`)
     `(map_x, map_y, timestamp)` olarak kaydeder.
2. 3 Eşzamanlı Otomatik Haritalandırma Kaynağı (3-Source Fusion):
   - Kaynak 1 — Sağ Üst Minimap Kaydı (`record_from_minimap`, `source_type="minimap_auto"`):
     Sağ üstteki minimap üzerinde SADECE **3 yapraklı beyaz bitki/filiz ikonlarını**
     (`Gatherable Material` — Minimap Filter Settings'deki bitki ikonu) tespit eder;
     canavar noktalarını (`• Monster`) ve tüy/mavi kanat ikonlarını (tüy toplanabilir
     materyal değildir!) kesin olarak filtreler. Yeşil yığın rozeti (`2` veya `3`)
     varsa `stack_count` değerini okuyarak `max_harvests = 3 * stack_count` (`3`, `6`
     veya `9`) atar ve 80x80 görsel şablonunu (`routes/<region>/node_XX.png` ve
     `routes/<region>/patches/node_XX.png`) kaydeder.
   - Kaynak 2 — 3D Dünya Beyaz Toplama Arması & `[E]` İkonları (`record_from_3d_world`, `source_type="3d_auto"`):
     Minimap'te görünmeyen `Aria` (mor kristal bitki) ve `Mela` (kırmızı meyveli çalı)
     gibi kaynakları 3D dünyadaki beyaz kanatlı toplama armasından ve yüzen `[E]`
     bitki ikonundan tespit eder; oyuncu ayak noktasına `(w*0.50, h*0.54)` göre perspektif
     izdüşümüyle harita koordinatına çevirerek `max_harvests = 3` olarak kaydeder.
   - Kaynak 3 — Aktif `[F Gather]` / `[E]` Toplama Olayı (`record_harvest_event`, `source_type="harvest_auto"`):
     Oyuncu veya bot bir kaynağın yanına gelip `[F Gather]` istemini gördüğünde veya
     toplama yaptığında, düğüm henüz rotada yoksa anında oyuncunun konumunu
     `max_harvests = 3` olarak rotaya ekler.
3. 3x Toplama Kapasitesi ve Uzamsal Tekilleştirme (Spatial Deduplication):
   - Her materyal düğümü varsayılan olarak 3 kez toplanabilir (`max_harvests = 3`,
     `current_harvests = 0`). Minimap üzerinde yeşil `2` veya `3` yığın rozeti varsa
     `max_harvests = 6` veya `9` olur.
   - `dedup_radius_px = 22.0` yarıçapı içinde kayıtlı bir düğüm varsa yeni düğüm
     oluşturmaz; mevcut düğümün koordinatını Kalman / EMA ağırlıklı ortalamayla
     iyileştirir ve `sightings_count` değerini artırır.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from core.route_manager import (
    DEFAULT_REGION,
    DEFAULT_RESOURCE_TYPE,
    DEFAULT_REST_SEC,
    DEFAULT_ROUTE,
    KNOWN_CANONICAL_REGIONS,
    PATCH_SIZE,
    NodeData,
    RouteManager,
    resolve_canonical_region,
)
from core.node_quality_gate import MODEL_CONF_MIN, evaluate_node_candidate
from core.vision_filters import (
    FeatherNegativeFilter,
    MonsterCircularityFilter,
    Pyramid3DCrestDetector,
    NameplateHSVIsolator,
    DynamicInteractionKeyReader,
    is_parchment_scroll_patch,
)
from version import __version__

try:
    from dataset_factory.exemplar_memory import get_exemplar_memory
except Exception:
    get_exemplar_memory = None

# Minimap sprout confidence floor aligned with MaterialScanner template score (≥0.68).
MINIMAP_SPROUT_CONF_MIN: float = 0.68
# 3D world detections below this without a harvest prompt are not persisted.
WORLD_3D_CONF_MIN: float = 0.55
# P1-3 (K-07): harvest_auto / radar_auto / unknown güven tabanı.
# 3d_auto (0.55) ve minimap_auto (0.68) tabanları varken bu kaynaklar
# kapısızdı — dış girdide 0.05'e kırpılmış güven bile düğüme dönerdi.
# İçsel [F] istemi zaten 0.60+ üretir (F-0056), 3D tespit 0.55+
# üretir; 0.55 tabanı gerçek üreticiyi geçirir, ölçülmemiş düşüğü eler.
HARVEST_RADAR_CONF_MIN: float = 0.55
# H-sweep (dilim 13 K3): türü bilinmeyen materyalin dürüst işareti.
# "Gatherable Material" kod tabanında zaten jenerik sınıf etiketidir
# (minimap filizleri, kapıdaki generic_names); spesifik bir cevher adı
# ("Orichalcum Ore") asla varsayılan tür olarak yazılmaz.
UNKNOWN_RESOURCE_TYPE: str = "Gatherable Material"

# v20-targena-3dauto: jenerik kayıt adları. 3D'den gelen spesifik bir
# materyal adı (Targena, Mela, Orichalcum Ore ...) asla bunlarla
# ezilmez; 3d_auto adayda ad belirlenemediyse (bu listedeyse) düğüm
# yazılmaz — yanlış jenerik kayıt yok.
_GENERIC_RESOURCE_NAMES = frozenset(
    {
        "",
        "unknown",
        "resource",
        "gatherable material",
        "gatherable plant",
        "gem",
        "gemstone",
        "world resource",
        "3d crest",
    }
)


def _is_generic_resource_name(name: Any) -> bool:
    """Ad jenerik/belirsiz mi (spesifik materyal adı değil mi)?"""
    if name is None:
        return True
    return str(name).strip().lower() in _GENERIC_RESOURCE_NAMES


def _is_specific_resource_name(name: Any) -> bool:
    """Ad spesifik bir materyal adı mı?"""
    return not _is_generic_resource_name(name)


def _resolve_mapped_resource_name(
    parsed_name: Any, verified_name: Any, source_type: str
) -> Tuple[str, bool]:
    """Kayıt adı kararı (v20-targena-3dauto). Döner (final_name, drop).

    - 3d_auto + spesifik parsed ad -> (parsed, False): kapının jenerik
      doğrulaması ("Gem" vb.) 3D mesh'ten gelen spesifik adı ezemez.
    - 3d_auto + jenerik parsed ad + spesifik verified -> (verified, False).
    - 3d_auto + ad hâlâ jenerik/belirsiz -> (parsed, True): yanlış
      jenerik kayıt yazılmaz, aday atılır.
    - Diğer kaynaklar: eski davranış (verified varsa onu al).
    """
    parsed = str(parsed_name or "")
    verified = str(verified_name or "") if verified_name else ""
    if str(source_type or "").strip().lower() == "3d_auto":
        if _is_specific_resource_name(parsed):
            return parsed, False
        if verified and _is_specific_resource_name(verified):
            return verified, False
        return parsed, True
    if verified:
        return verified, False
    return parsed, False


# Drift / time that sets needs_map_calibrate for a world-map fix.
ODOMETRY_CALIBRATE_DRIFT_PX: float = 48.0
ODOMETRY_CALIBRATE_INTERVAL_SEC: float = 90.0

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

# Sağ üst minimap ROI: x: 0.855..0.992 (w=0.137), y: 0.065..0.255 (h=0.190)
MINIMAP_NORM_RECT: Tuple[float, float, float, float] = (0.855, 0.065, 0.137, 0.190)
# Minimap→harita ölçeği: minimap ROI harita genişliğinin ~0.137'si kadardır
# (~263 px / 1920 px), yani 1 minimap pikseli ~7.30 harita pikselidir.
# Varsayılan 1.0, göreli ofsetleri ~7 kat küçük hesaplayıp düğümleri oyuncu
# etrafına yığıyordu (F-0003).
DEFAULT_MINIMAP_TO_MAP_SCALE: float = 1.0 / MINIMAP_NORM_RECT[2]
# Yakın-çekim (closeup) referans çerçevesinde minimap'in kapladığı göreli
# dikdörtgen (x, y, w, h); eskiden _get_minimap_pixel_rect içine gömülüydü (F-0408).
CLOSEUP_MINIMAP_NORM_RECT: Tuple[float, float, float, float] = (0.498, 0.200, 0.466, 0.712)
DEFAULT_DEDUP_RADIUS_PX: float = 22.0
DEDUP_RADIUS_MIN_PX: float = 15.0
DEDUP_RADIUS_MAX_PX: float = 25.0
# Kaynak güvenilirlik önceliği (cluster_detections): her çağrıda yeniden
# kurulmaması için modül düzeyinde tek örnek (D02).
CLUSTER_SOURCE_PRIORITY: Dict[str, int] = {
    "harvest_auto": 4,
    "minimap_auto": 3,
    "3d_auto": 2,
    "radar_auto": 1,
}
DEFAULT_EMA_ALPHA: float = 0.35
REF_MAP_WIDTH: int = 1920
REF_MAP_HEIGHT: int = 1080
DEFAULT_MAX_HARVESTS: int = 3

# Canlı bölge tespiti (Görev 4 / harita-istatistik).
# - Minimap tespiti en fazla 5 sn'de bir çalışır (başlık şablon eşleşmesi pahalıdır).
# - Açık dünya haritası (M) ORB eşleşmesi en fazla 30 sn'de bir çalışır.
# - Güven eşiği: minimap başlık-eşleşmesinin "eşleşme sayılır" sınırı (0.45).
REGION_DETECT_MIN_CONFIDENCE: float = 0.45
REGION_DETECT_INTERVAL_SEC: float = 5.0
REGION_WORLD_MAP_MATCH_INTERVAL_SEC: float = 30.0
# Dünya haritası ORB kararı için asgari iyi eşleşme sayısı ve en iyi/ikinci
# oran farkı. Canlı kareyle doğrulanmadı; eşik bilerek katı tutuldu.
REGION_WORLD_MAP_MIN_GOOD_MATCHES: int = 15
REGION_WORLD_MAP_MIN_MARGIN_RATIO: float = 2.0


def _is_normalized_coord_pair(sx: float, sy: float) -> bool:
    """`(sx, sy)` normalize (0..1) koordinat çifti mi?

    0..1.5 aralığı piksel `(1, 1)` ile normalize `(1.0, 1.0)` ayrımını yapamaz
    (F-0212): yalnızca kesirli 0..1 çiftleri normalize sayılır; tam sayı
    çiftler (0/1) piksel koordinatıdır. `(0, 0)` her iki yorumda da aynıdır.
    """
    if not (0.0 <= float(sx) <= 1.0 and 0.0 <= float(sy) <= 1.0):
        return False
    return not (float(sx).is_integer() and float(sy).is_integer())


@dataclass
class TrackedResourceNode:
    """
    AutoCartographer tarafından uzamsal olarak takip edilen, tekilleştirilen ve
    3x toplama kapasitesi (`max_harvests = 3 * stack_count`) yönetilen kaynak düğümü kaydı.
    """

    node_id: int
    region: str = DEFAULT_REGION
    routes: List[str] = field(default_factory=list)
    resource_type: str = DEFAULT_RESOURCE_TYPE
    # Ölçülmemiş düğüm merkez (960, 540) + güven 1.0 ile doğmuyor (F-0209):
    # konum bilinmiyorsa (0, 0) ve güven 0.0; gerçek ölçümle dolar.
    map_x: float = 0.0
    map_y: float = 0.0
    norm_x: float = 0.0
    norm_y: float = 0.0
    in_air: bool = False
    sightings_count: int = 1
    confidence: float = 0.0
    # Tüketim/dinlenme durumu (F-0061): "active" | "resting".
    state: str = "active"
    rest_started_timestamp: float = 0.0
    kalman_variance: float = 16.0
    kalman_variance_x: float = 16.0
    kalman_variance_y: float = 16.0
    patch_path: str = ""
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    patch_img: Optional[np.ndarray] = field(default=None, repr=False)
    stack_count: int = 1
    max_harvests: int = DEFAULT_MAX_HARVESTS
    current_harvests: int = 0
    source_type: str = "minimap_auto"
    last_rel_dx: float = 0.0
    last_rel_dy: float = 0.0
    rel_track_valid: bool = False
    node_type: str = "resource"
    connected_waypoints: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.stack_count = max(1, int(self.stack_count))
        if int(self.max_harvests) <= 0:
            self.max_harvests = DEFAULT_MAX_HARVESTS * self.stack_count
        else:
            self.max_harvests = max(1, int(self.max_harvests))
        self.current_harvests = max(0, int(self.current_harvests))
        self.source_type = str(self.source_type or "minimap_auto")
        self.node_type = str(self.node_type or "resource")
        self.state = str(self.state or "active").strip().lower() or "active"
        self.rest_started_timestamp = float(self.rest_started_timestamp or 0.0)
        if not hasattr(self, "kalman_variance_x"):
            self.kalman_variance_x = float(self.kalman_variance)
        if not hasattr(self, "kalman_variance_y"):
            self.kalman_variance_y = float(self.kalman_variance)

    @property
    def remaining_harvests(self) -> int:
        """Bu düğümden kalan toplanabilir materyal hakkını döndürür."""
        if str(self.state) == "resting":
            return 0
        return max(0, int(self.max_harvests) - int(self.current_harvests))

    @property
    def is_depleted(self) -> bool:
        """Düğümün 3x (veya `max_harvests`) toplama hakkının bitip bitmediğini döndürür."""
        return str(self.state) == "resting" or int(self.current_harvests) >= int(self.max_harvests)

    def to_dict(self) -> Dict[str, Any]:
        """Düğüm bilgilerini sözlük olarak döndürür."""
        return {
            "id": int(self.node_id),
            "node_id": int(self.node_id),
            "region": str(self.region),
            "routes": list(self.routes),
            "resource_type": str(self.resource_type),
            "map_x": int(round(self.map_x)),
            "map_y": int(round(self.map_y)),
            "exact_map_x": round(float(self.map_x), 3),
            "exact_map_y": round(float(self.map_y), 3),
            "norm_x": round(float(self.norm_x), 4),
            "norm_y": round(float(self.norm_y), 4),
            "in_air": bool(self.in_air),
            "sightings_count": int(self.sightings_count),
            "confidence": round(float(self.confidence), 4),
            "kalman_variance": round(float(self.kalman_variance), 4),
            "kalman_variance_x": round(float(getattr(self, "kalman_variance_x", self.kalman_variance)), 4),
            "kalman_variance_y": round(float(getattr(self, "kalman_variance_y", self.kalman_variance)), 4),
            "patch_path": str(self.patch_path),
            "first_seen_timestamp": float(self.first_seen_timestamp),
            "last_seen_timestamp": float(self.last_seen_timestamp),
            "stack_count": int(self.stack_count),
            "max_harvests": int(self.max_harvests),
            "current_harvests": int(self.current_harvests),
            "remaining_harvests": int(self.remaining_harvests),
            "is_depleted": bool(self.is_depleted),
            "state": str(self.state),
            "rest_started_timestamp": float(self.rest_started_timestamp),
            "source_type": str(self.source_type),
            "node_type": str(getattr(self, "node_type", "resource")),
            "connected_waypoints": list(getattr(self, "connected_waypoints", [])),
        }


@dataclass
class GraphWaypoint:
    """Otonom keşif rotasındaki yürüyüş ve gezinme ara noktası (breadcrumb / waypoint)."""

    waypoint_id: int
    map_x: float
    map_y: float
    norm_x: float = 0.5
    norm_y: float = 0.5
    timestamp: float = field(default_factory=time.time)
    connected_node_ids: List[int] = field(default_factory=list)
    seq_index: int = 0
    is_breadcrumb: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waypoint_id": int(self.waypoint_id),
            "map_x": round(float(self.map_x), 2),
            "map_y": round(float(self.map_y), 2),
            "norm_x": round(float(self.norm_x), 4),
            "norm_y": round(float(self.norm_y), 4),
            "timestamp": float(self.timestamp),
            "connected_node_ids": list(self.connected_node_ids),
            "seq_index": int(self.seq_index),
            "is_breadcrumb": bool(self.is_breadcrumb),
        }


@dataclass
class GraphEdge:
    """Rota çizgesindeki iki nokta arasındaki bağlantı ve mesafe bilgisi."""

    from_id: int
    to_id: int
    from_type: str = "waypoint"  # "waypoint" | "node"
    to_type: str = "waypoint"  # "waypoint" | "node"
    distance_px: float = 0.0
    heading_deg: float = 0.0
    bidirectional: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_id": int(self.from_id),
            "to_id": int(self.to_id),
            "from_type": str(self.from_type),
            "to_type": str(self.to_type),
            "distance_px": round(float(self.distance_px), 2),
            "heading_deg": round(float(self.heading_deg), 1),
            "bidirectional": bool(self.bidirectional),
        }


class RouteGraph:
    """
    Otonom keşif sırasında yürüyüş ayak izleri (breadcrumbs / waypoints) ile
    keşfedilen materyal düğümlerini (resource nodes) birbirine bağlayan yönlü/yönsüz rota çizgesi.
    """

    def __init__(self, region: str = DEFAULT_REGION) -> None:
        self.region: str = str(region or DEFAULT_REGION)
        self.waypoints: Dict[int, GraphWaypoint] = {}
        self.edges: List[GraphEdge] = []
        # P1-6 (K-06): kenar dedup endeksi — build_route_graph her
        # çağrıda aynı kenarı yeniden ekleyip total_distance'ı
        # şişirmesin diye add_edge idempotent'tir.
        self._edge_index: Dict[Tuple[Any, ...], GraphEdge] = {}
        self._next_wp_id: int = 1
        self._adjacency: Dict[Tuple[str, int], List[Tuple[str, int, float]]] = {}

    @property
    def total_distance(self) -> float:
        """Çizgedeki toplam rota uzunluğunu piksel cinsinden döndürür."""
        return sum(float(e.distance_px) for e in self.edges)

    @property
    def node_count(self) -> int:
        """Çizgeye bağlı farklı kaynak düğümü sayısı (waypoint değil, F-0218)."""
        return len({nid for wp in self.waypoints.values() for nid in wp.connected_node_ids})

    @property
    def waypoint_count(self) -> int:
        """Çizgedeki yürüyüş ara noktası (waypoint) sayısı."""
        return len(self.waypoints)

    @property
    def edge_count(self) -> int:
        return len(self.edges)

    def add_waypoint(
        self,
        map_x: float,
        map_y: float,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
        map_width: int = REF_MAP_WIDTH,
        map_height: int = REF_MAP_HEIGHT,
        is_breadcrumb: bool = True,
        timestamp: Optional[float] = None,
    ) -> GraphWaypoint:
        """Yeni bir rota ara noktası (waypoint/breadcrumb) ekler."""
        nx = float(norm_x if norm_x is not None else map_x / float(max(1, map_width)))
        ny = float(norm_y if norm_y is not None else map_y / float(max(1, map_height)))
        wp_id = self._next_wp_id
        self._next_wp_id += 1
        seq = len(self.waypoints) + 1
        wp = GraphWaypoint(
            waypoint_id=wp_id,
            map_x=float(map_x),
            map_y=float(map_y),
            norm_x=round(nx, 4),
            norm_y=round(ny, 4),
            timestamp=float(timestamp or time.time()),
            seq_index=seq,
            is_breadcrumb=is_breadcrumb,
        )
        self.waypoints[wp_id] = wp
        return wp

    def add_edge(
        self,
        from_id: int,
        to_id: int,
        from_type: str = "waypoint",
        to_type: str = "waypoint",
        distance_px: Optional[float] = None,
        heading_deg: float = 0.0,
        bidirectional: bool = True,
    ) -> GraphEdge:
        """İki nokta arasında bağlantı kenarı ekler.

        P1-6 (K-06): idempotent — aynı uç çiftine sahip kenar zaten
        varsa mevcut kenar döndürülür, liste/bitişiklik/total_distance
        değişmez. Çift yönlü kenarda uç sırası önemsizdir.
        """
        # Dedup anahtarı mesafeden bağımsızdır: aynı uç çifti aynı
        # geometridir (waypoint/node koordinatları sabittir).
        end_a = (str(from_type), int(from_id))
        end_b = (str(to_type), int(to_id))
        if bidirectional:
            edge_key: Tuple[Any, ...] = (
                "bidir",
                end_a if end_a <= end_b else end_b,
                end_b if end_a <= end_b else end_a,
            )
        else:
            edge_key = ("dir", end_a, end_b)
        existing = self._edge_index.get(edge_key)
        if existing is not None:
            return existing

        if distance_px is None:
            p1 = self.waypoints.get(from_id) if from_type == "waypoint" else None
            p2 = self.waypoints.get(to_id) if to_type == "waypoint" else None
            if p1 and p2:
                distance_px = math.hypot(p2.map_x - p1.map_x, p2.map_y - p1.map_y)
                heading_deg = (
                    math.degrees(math.atan2(p2.map_x - p1.map_x, -(p2.map_y - p1.map_y)))
                ) % 360.0
            else:
                distance_px = 0.0

        edge = GraphEdge(
            from_id=int(from_id),
            to_id=int(to_id),
            from_type=str(from_type),
            to_type=str(to_type),
            distance_px=float(distance_px),
            heading_deg=float(heading_deg),
            bidirectional=bool(bidirectional),
        )
        self.edges.append(edge)
        self._edge_index[edge_key] = edge

        k1 = (from_type, int(from_id))
        k2 = (to_type, int(to_id))
        self._adjacency.setdefault(k1, []).append((to_type, int(to_id), float(distance_px)))
        if bidirectional:
            self._adjacency.setdefault(k2, []).append((from_type, int(from_id), float(distance_px)))

        return edge

    def get_nearest_waypoint(
        self,
        map_x: float,
        map_y: float,
    ) -> Tuple[Optional[GraphWaypoint], float]:
        """`(map_x, map_y)` konumuna en yakın ara noktayı bulur.

        Kare mesafe karşılaştırması yapılır (sıralama `math.hypot` ile aynı,
        kareköksüz); tek `sqrt` yalnızca dönen mesafe için alınır (D02).
        """
        best: Optional[GraphWaypoint] = None
        best_d2 = float("inf")
        for wp in self.waypoints.values():
            dx = wp.map_x - map_x
            dy = wp.map_y - map_y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = wp
        return best, math.sqrt(best_d2) if best is not None else float("inf")

    def connect_node_to_nearest_waypoint(
        self,
        node_id: int,
        node_map_x: float,
        node_map_y: float,
        max_connection_dist_px: float = 120.0,
    ) -> Optional[GraphEdge]:
        """Bir materyal düğümünü en yakın yürüyüş ara noktasına bağlar."""
        nearest_wp, dist = self.get_nearest_waypoint(node_map_x, node_map_y)
        if nearest_wp is not None and dist <= max_connection_dist_px:
            if node_id not in nearest_wp.connected_node_ids:
                nearest_wp.connected_node_ids.append(node_id)
            return self.add_edge(
                from_id=nearest_wp.waypoint_id,
                to_id=node_id,
                from_type="waypoint",
                to_type="node",
                distance_px=dist,
                bidirectional=True,
            )
        return None

    def build_connected_sequence(self, loop: bool = True) -> List[Dict[str, Any]]:
        """
        Graf üzerindeki tüm ara noktaları ve kaynak düğümlerini
        kesintisiz, oynanabilir bir navigasyon rotasına dönüştürür.
        """
        sequence: List[Dict[str, Any]] = []
        sorted_wps = sorted(self.waypoints.values(), key=lambda w: w.seq_index)
        for wp in sorted_wps:
            item = wp.to_dict()
            item["target_type"] = "waypoint"
            sequence.append(item)
            for nid in wp.connected_node_ids:
                sequence.append({
                    "target_type": "node",
                    "node_id": nid,
                    "connected_from_wp": wp.waypoint_id,
                })
        # loop=True: rota son ara noktadan ilk ara noktaya kapanır (F-0218).
        # Sıraya ek öğe konmaz; kapanış bilgisi son waypoint öğesine işaretlenir.
        if loop and len(sorted_wps) >= 2:
            for item in reversed(sequence):
                if item.get("target_type") == "waypoint":
                    item["loop_close"] = True
                    item["loop_to_waypoint_id"] = int(sorted_wps[0].waypoint_id)
                    break
        return sequence

    def to_dict(self) -> Dict[str, Any]:
        """Çizgeyi JSON uyumlu sözlük olarak serileştirir."""
        return {
            "region": self.region,
            "waypoints": [
                wp.to_dict() for wp in sorted(self.waypoints.values(), key=lambda w: w.waypoint_id)
            ],
            "edges": [e.to_dict() for e in self.edges],
            "total_distance_px": round(self.total_distance, 2),
            "waypoint_count": len(self.waypoints),
            "edge_count": len(self.edges),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RouteGraph:
        """Sözlükten RouteGraph örneği oluşturur."""
        g = cls(region=data.get("region", DEFAULT_REGION))
        for wp_d in data.get("waypoints", []):
            wp = GraphWaypoint(
                waypoint_id=int(wp_d["waypoint_id"]),
                map_x=float(wp_d["map_x"]),
                map_y=float(wp_d["map_y"]),
                norm_x=float(wp_d.get("norm_x", 0.5)),
                norm_y=float(wp_d.get("norm_y", 0.5)),
                timestamp=float(wp_d.get("timestamp", 0.0)),
                connected_node_ids=list(wp_d.get("connected_node_ids", [])),
                seq_index=int(wp_d.get("seq_index", 0)),
                is_breadcrumb=bool(wp_d.get("is_breadcrumb", True)),
            )
            g.waypoints[wp.waypoint_id] = wp
            if wp.waypoint_id >= g._next_wp_id:
                g._next_wp_id = wp.waypoint_id + 1
        for e_d in data.get("edges", []):
            g.add_edge(
                from_id=int(e_d["from_id"]),
                to_id=int(e_d["to_id"]),
                from_type=str(e_d.get("from_type", "waypoint")),
                to_type=str(e_d.get("to_type", "waypoint")),
                distance_px=float(e_d.get("distance_px", 0.0)),
                heading_deg=float(e_d.get("heading_deg", 0.0)),
                bidirectional=bool(e_d.get("bidirectional", True)),
            )
        return g


class VisualOdometryTracker:
    """
    Minimap doku kayması (`cv2.phaseCorrelate` + Lucas-Kanade Optik Akış + ORB)
    ve dead-reckoning yer değiştirme üzerinden oyuncunun dünya/bölge haritası koordinatlarını
    `(map_x, map_y)` takip eden görsel odometri sınıfı.
    """

    def __init__(
        self,
        initial_map_x: float = 960.0,
        initial_map_y: float = 540.0,
        map_width: int = REF_MAP_WIDTH,
        map_height: int = REF_MAP_HEIGHT,
        # Doğrudan kurulumda harita uzayı varsayılır; AutoCartographer minimap
        # ölçeğini (DEFAULT_MINIMAP_TO_MAP_SCALE) kendi dönüşümlerinde uygular (F-0003).
        minimap_to_map_scale: float = 1.0,
        max_trajectory_len: int = 10000,
    ) -> None:
        self.map_width: int = max(160, int(map_width))
        self.map_height: int = max(120, int(map_height))
        self.minimap_to_map_scale: float = float(minimap_to_map_scale)
        self.max_trajectory_len: int = max(100, int(max_trajectory_len))

        self.map_x: float = float(initial_map_x)
        self.map_y: float = float(initial_map_y)
        self.norm_x: float = round(self.map_x / float(self.map_width), 4)
        self.norm_y: float = round(self.map_y / float(self.map_height), 4)
        self.walk_trajectory: List[Tuple[float, float, float]] = []

        self._prev_minimap_clean: Optional[np.ndarray] = None
        self._hann_window: Optional[np.ndarray] = None
        self._orb: Optional[Any] = None
        self.cumulative_drift_px: float = 0.0

        # Dead-reckoning ve hız takibi
        self.velocity_x: float = 0.0
        self.velocity_y: float = 0.0
        self.dead_reckoning_dx: float = 0.0
        self.dead_reckoning_dy: float = 0.0
        self.last_update_ts: float = time.time()

        try:
            self._orb = cv2.ORB_create(nfeatures=120, fastThreshold=12)
        except Exception as exc:
            logger.debug("VisualOdometryTracker ORB ilklendirilemedi: %s", exc)
            self._orb = None

    def get_position(self) -> Tuple[float, float, float, float]:
        """Güncel `(map_x, map_y, norm_x, norm_y)` konumunu döndürür."""
        self.norm_x = round(float(self.map_x) / float(max(1, self.map_width)), 4)
        self.norm_y = round(float(self.map_y) / float(max(1, self.map_height)), 4)
        return (
            round(float(self.map_x), 3),
            round(float(self.map_y), 3),
            float(self.norm_x),
            float(self.norm_y),
        )

    def set_position(
        self,
        map_x: float,
        map_y: float,
        map_width: Optional[int] = None,
        map_height: Optional[int] = None,
        record_trajectory: bool = True,
    ) -> Tuple[float, float, float, float]:
        """Konumu doğrudan ayarlar ve yürüyüş izine ekler."""
        if map_width is not None and map_width >= 160:
            self.map_width = int(map_width)
        if map_height is not None and map_height >= 120:
            self.map_height = int(map_height)
        self.map_x = float(np.clip(map_x, 0.0, float(self.map_width - 1)))
        self.map_y = float(np.clip(map_y, 0.0, float(self.map_height - 1)))
        self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
        self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
        self.dead_reckoning_dx = 0.0
        self.dead_reckoning_dy = 0.0
        if record_trajectory:
            self.append_trajectory()
        return self.get_position()

    def update_displacement(
        self,
        dx: float,
        dy: float,
        dt: Optional[float] = None,
        is_minimap_shift: bool = True,
    ) -> Tuple[float, float]:
        """
        Yer değiştirmeyi odometriye ekler, hızı günceller ve dead-reckoning durumunu yeniler.
        """
        now = time.time()
        elapsed = dt if (dt is not None and dt > 0) else max(0.01, now - self.last_update_ts)
        self.last_update_ts = now

        scale = self.minimap_to_map_scale if is_minimap_shift else 1.0
        step_x = float(dx) * scale
        step_y = float(dy) * scale

        self.map_x = float(np.clip(self.map_x + step_x, 0.0, float(self.map_width - 1)))
        self.map_y = float(np.clip(self.map_y + step_y, 0.0, float(self.map_height - 1)))
        self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
        self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)

        if elapsed > 0:
            self.velocity_x = step_x / elapsed
            self.velocity_y = step_y / elapsed
        # Dead-reckoning birikimi burada silinmez (F-0220): apply_dead_reckoning
        # ile biriken değer, offset_to_world_coords tüketene kadar korunur;
        # yalnızca reset()/set_position() sıfırlar.
        return step_x, step_y

    def apply_dead_reckoning(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """
        Görsel odometri kaybolduğunda (örtülme, karanlık vb.) hız tahminine göre
        dead-reckoning yer değiştirmesini hesaplar ve harita konumuna ekler.
        """
        now = time.time()
        elapsed = dt if (dt is not None and dt > 0) else max(0.01, now - self.last_update_ts)
        self.last_update_ts = now

        if (abs(self.velocity_x) > 0.05 or abs(self.velocity_y) > 0.05) and elapsed < 3.0:
            step_x = self.velocity_x * elapsed
            step_y = self.velocity_y * elapsed
            self.dead_reckoning_dx += step_x
            self.dead_reckoning_dy += step_y
            self.map_x = float(np.clip(self.map_x + step_x, 0.0, float(self.map_width - 1)))
            self.map_y = float(np.clip(self.map_y + step_y, 0.0, float(self.map_height - 1)))
            self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
            self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
            self.velocity_x *= 0.96
            self.velocity_y *= 0.96
            return step_x, step_y
        return 0.0, 0.0

    def append_trajectory(self, timestamp: Optional[float] = None) -> None:
        """Güncel konumu zaman damgasıyla birlikte yürüyüş izine ekler."""
        ts = float(timestamp if timestamp is not None else time.time())
        entry = (round(float(self.map_x), 3), round(float(self.map_y), 3), ts)
        self.walk_trajectory.append(entry)
        if len(self.walk_trajectory) > self.max_trajectory_len:
            excess = len(self.walk_trajectory) - self.max_trajectory_len
            del self.walk_trajectory[:excess]

    def offset_to_world_coords(
        self,
        dx: float,
        dy: float,
        is_map_offset: bool = True,
        dead_reckoning_dx: float = 0.0,
        dead_reckoning_dy: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        Minimap veya radar merkezindeki oyuncu okuna göre `(dx, dy)` ofsetini
        dünya/harita koordinatlarına `(world_x, world_y, norm_x, norm_y)` dönüştürür.
        Varsayılan `is_map_offset=True` (AutoCartographer ile aynı, F-0221):
        `dx, dy` doğrudan harita piksel ofsetidir; `is_map_offset=False` ise
        minimap piksel ofsetidir ve `minimap_to_map_scale` ile çarpılır.
        Dead-reckoning yer değiştirmesi varsa ofsete dahil edilir.
        """
        scale = 1.0 if is_map_offset is True else self.minimap_to_map_scale
        world_x = float(
            np.clip(
                self.map_x + float(dx) * scale + float(dead_reckoning_dx),
                0.0,
                float(self.map_width - 1),
            )
        )
        world_y = float(
            np.clip(
                self.map_y + float(dy) * scale + float(dead_reckoning_dy),
                0.0,
                float(self.map_height - 1),
            )
        )
        nx = round(world_x / float(max(1, self.map_width)), 4)
        ny = round(world_y / float(max(1, self.map_height)), 4)
        return round(world_x, 3), round(world_y, 3), float(nx), float(ny)

    def reset(
        self,
        map_x: float = 960.0,
        map_y: float = 540.0,
        clear_trajectory: bool = False,
    ) -> None:
        """Odometri durumunu sıfırlar."""
        self.map_x = float(map_x)
        self.map_y = float(map_y)
        self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
        self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
        self._prev_minimap_clean = None
        self.cumulative_drift_px = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.dead_reckoning_dx = 0.0
        self.dead_reckoning_dy = 0.0
        self.last_update_ts = time.time()
        if clear_trajectory:
            self.walk_trajectory.clear()


def _essence_name_near_box(frame_bgr: np.ndarray, screen_x: float, screen_y: float) -> str:
    """Name the essence under a floating gather icon. Falls back to a plant label."""
    height, width = frame_bgr.shape[:2]
    x1 = max(0, int(screen_x) - 48)
    y1 = max(0, int(screen_y) - 20)
    x2 = min(width, int(screen_x) + 48)
    y2 = min(height, int(screen_y) + 90)
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return "Gatherable Plant"
    try:
        from config.materials_db import classify_3d_material_patch

        info = classify_3d_material_patch(patch)
        name = str(info.get("name") or "").strip()
    except Exception:
        name = ""
    return name or "Gatherable Plant"


class AutoCartographer:
    """
    Gerçek Zamanlı Minimap Görsel Odometri (SLAM) ve 3-Kaynaklı (Minimap + 3D Crest + F Gather)
    Otomatik Materyal Haritalandırıcı.
    """

    # 3D ekran referans çözünürlüğü ve oyuncu ayak noktası (perspektif izdüşümü için)
    REF_SCREEN_WIDTH: int = 1280
    REF_SCREEN_HEIGHT: int = 720
    PLAYER_FEET_X_RATIO: float = 0.50
    PLAYER_FEET_Y_RATIO: float = 0.54
    PERSPECTIVE_SCALE_X: float = 0.085
    PERSPECTIVE_SCALE_Y: float = 0.115

    def __init__(
        self,
        route_manager: Optional[RouteManager] = None,
        vision_engine: Optional[Any] = None,
        minimap_norm_rect: Tuple[float, float, float, float] = MINIMAP_NORM_RECT,
        dedup_radius_px: float = DEFAULT_DEDUP_RADIUS_PX,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        initial_map_x: float = 960.0,
        initial_map_y: float = 540.0,
        map_width: int = REF_MAP_WIDTH,
        map_height: int = REF_MAP_HEIGHT,
        minimap_to_map_scale: float = DEFAULT_MINIMAP_TO_MAP_SCALE,
        max_trajectory_len: int = 10000,
        exemplar_memory: Optional[Any] = None,
    ) -> None:
        self.route_manager: Optional[RouteManager] = route_manager
        self.vision_engine: Optional[Any] = vision_engine
        self.exemplar_memory: Optional[Any] = exemplar_memory
        self.minimap_norm_rect: Tuple[float, float, float, float] = (
            float(minimap_norm_rect[0]),
            float(minimap_norm_rect[1]),
            float(minimap_norm_rect[2]),
            float(minimap_norm_rect[3]),
        )
        self.dedup_radius_px: float = float(
            np.clip(dedup_radius_px, DEDUP_RADIUS_MIN_PX, DEDUP_RADIUS_MAX_PX)
        )
        self.ema_alpha: float = float(np.clip(ema_alpha, 0.05, 0.95))

        self.map_width: int = max(160, int(map_width))
        self.map_height: int = max(120, int(map_height))
        self.minimap_to_map_scale: float = float(minimap_to_map_scale)
        self.max_trajectory_len: int = max(100, int(max_trajectory_len))

        # Görsel odometri takipçisi (VisualOdometryTracker) harita uzayında
        # çalışır (ölçeği 1.0): minimap→harita dönüşümü AutoCartographer'ın
        # kendi yollarında (tespit/odometri/ofset) self.minimap_to_map_scale
        # ile yapılır; tracker'a ölçek geçirilmez (F-0003).
        self.odometry_tracker: VisualOdometryTracker = VisualOdometryTracker(
            initial_map_x=initial_map_x,
            initial_map_y=initial_map_y,
            map_width=self.map_width,
            map_height=self.map_height,
            max_trajectory_len=self.max_trajectory_len,
        )

        # Oyuncunun anlık harita koordinatları (Dead-Reckoning + Minimap SLAM)
        self.map_x: float = float(initial_map_x)
        self.map_y: float = float(initial_map_y)
        self.norm_x: float = round(self.map_x / float(self.map_width), 4)
        self.norm_y: float = round(self.map_y / float(self.map_height), 4)

        # Oyuncunun gezdiği ayak izi: list[(map_x, map_y, timestamp)]
        self.walk_trajectory: List[Tuple[float, float, float]] = self.odometry_tracker.walk_trajectory

        # Görsel odometri iç durum değişkenleri
        self._prev_minimap_clean: Optional[np.ndarray] = None
        self._last_valid_minimap_bgr: Optional[np.ndarray] = None
        self._hann_window: Optional[np.ndarray] = None
        self._odometry_Search_orb: Optional[Any] = self.odometry_tracker._orb
        # Yön açısı için tek MinimapScanner örneği (her karede yeniden kurulmaz, F-0411).
        self._heading_scanner: Optional[Any] = None
        # Canlı bölge tespiti (Görev 4): MinimapScanner örneği + son tespit.
        # _detected_region yalnızca güvenilir tespitle dolar; boşken "bilinmiyor"
        # demektir ve asla varsayılan bölge sayılmaz.
        self._region_scanner: Optional[Any] = None
        self._detected_region: Optional[Dict[str, Any]] = None
        self._last_region_detect_ts: float = 0.0
        self._last_world_map_region_ts: float = 0.0
        self._world_map_orb_cache: Optional[Dict[str, Any]] = None
        self._is_calibrated_on_world_map: bool = False
        self._last_calibration_ts: float = 0.0
        # H-sweep (dilim 13 Y7): kümülatif drift HİÇ ölçülmez — bu
        # sayaç yalnızca sıfırlanır (F-0410: gerçek konum hatası
        # ölçülmeden birikim yapılmaz). 0.0 değeri ölçüm değildir;
        # raporlayanlar None + drift_not_measured döndürür. Alan
        # geriye uyum için (sıfırlama çağrıları) korunur.
        self._cumulative_drift_px: float = 0.0
        # H-sweep (dilim 13 O2): son kayma güveni (update_player_position
        # ölçer); düşük güvende dead-reckoning durumu dışarı çıkar.
        self._last_odometry_confidence: Optional[float] = None

        # Bölge bazlı takip edilen kaynak düğümleri: region -> List[TrackedResourceNode]
        self._tracked_nodes: Dict[str, List[TrackedResourceNode]] = {}

        # Otonom Rota Çizgesi ve Breadcrumb Üreteci (AUTO CARTOGRAPHY / Oto-Keşif)
        self.auto_cartography_enabled: bool = False
        self.auto_mapped_count: int = 0
        self.last_mapped_info: Optional[Dict[str, Any]] = None
        self.breadcrumb_interval_px: float = 30.0
        self._last_breadcrumb_x: Optional[float] = None
        self._last_breadcrumb_y: Optional[float] = None
        self._last_breadcrumb_wp_id: Optional[int] = None
        self._active_region: str = DEFAULT_REGION
        self._route_graphs: Dict[str, RouteGraph] = {}
        self.route_graph: RouteGraph = self.get_route_graph(DEFAULT_REGION)

        # Kaynak bazlı haritalandırma istatistik sayaçları
        # P1-6 (K-06): her kaynak türü yalnızca kendi sayacını artırır
        # (tek kaynak); radar_auto'nun da kendi sayacı vardır.
        self.total_minimap_recorded: int = 0
        self.total_3d_recorded: int = 0
        self.total_harvest_recorded: int = 0
        self.total_radar_recorded: int = 0
        # Envanter kapanış animasyonu boyunca materyal kaydını tut (saniye).
        self._inventory_block_until: float = 0.0
        self._orbit_rel_step_px: float = 36.0
        self._orbit_rel_max_age_sec: float = 3.5

        # Vision filter instances for enhanced confidence safeguard
        self.feather_negative_filter: FeatherNegativeFilter = FeatherNegativeFilter()
        self.monster_circularity_filter: MonsterCircularityFilter = MonsterCircularityFilter()
        self.pyramid_3d_crest_detector: Pyramid3DCrestDetector = Pyramid3DCrestDetector()
        self.nameplate_isolator: NameplateHSVIsolator = NameplateHSVIsolator()
        self.key_reader: DynamicInteractionKeyReader = DynamicInteractionKeyReader()

        # Lazy Gatherion ONNX gate (CPU). Loaded on first new-node candidate.
        self._quality_detector: Any = None
        self._quality_detector_failed: bool = False
        self.total_gate_rejected: int = 0
        self.needs_map_calibrate: bool = False

        # Dead-reckoning yer değiştirme ve hız takibi
        self.velocity_x: float = 0.0
        self.velocity_y: float = 0.0
        self.dead_reckoning_dx: float = 0.0
        self.dead_reckoning_dy: float = 0.0
        self._last_odometry_update_ts: float = time.time()
        self.player_heading_deg: float = 0.0

        # Gerçek zamanlı radar ve harita olay bildirim dinleyicileri (callbacks / listeners)
        self._node_discovered_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._node_updated_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._radar_event_callbacks: List[Callable[[str, Dict[str, Any]], None]] = []
        self._ui_listeners: List[Any] = []

        if self.route_manager is not None:
            self._sync_existing_nodes_from_route_manager(self.route_manager)
            if getattr(self.route_manager, "active_region", None):
                self._active_region = str(self.route_manager.active_region)
                self.route_graph = self.get_route_graph(self._active_region)

    @property
    def active_region(self) -> str:
        """Kullanılan aktif bölge adı."""
        if self.route_manager is not None and getattr(self.route_manager, "active_region", None):
            return str(self.route_manager.active_region)
        return getattr(self, "_active_region", DEFAULT_REGION)

    @active_region.setter
    def active_region(self, val: str) -> None:
        self._active_region = str(val or DEFAULT_REGION)
        if self.route_manager is not None:
            self.route_manager.active_region = self._active_region
        self.route_graph = self.get_route_graph(self._active_region)

    @property
    def auto_cartography_mode(self) -> bool:
        """Otonom haritalandırma modu açık mı?"""
        return bool(self.auto_cartography_enabled)

    @auto_cartography_mode.setter
    def auto_cartography_mode(self, enabled: bool) -> None:
        self.enable_auto_cartography(enabled)

    def enable_auto_cartography(self, enabled: bool = True) -> bool:
        """AUTO CARTOGRAPHY modunu açar veya kapatır."""
        self.auto_cartography_enabled = bool(enabled)
        target_reg = str(self.active_region or DEFAULT_REGION)
        self.route_graph = self.get_route_graph(target_reg)
        logger.info(
            "AutoCartographer: AUTO CARTOGRAPHY modu %s (Bölge: %s)",
            "aktif" if self.auto_cartography_enabled else "pasif",
            target_reg,
        )
        return self.auto_cartography_enabled

    def set_auto_cartography_mode(self, enabled: bool) -> None:
        """AUTO CARTOGRAPHY modunu ayarlar."""
        self.enable_auto_cartography(enabled)

    def is_auto_cartography_active(self) -> bool:
        """AUTO CARTOGRAPHY modunun aktif olup olmadığını döndürür."""
        return bool(self.auto_cartography_enabled)

    def get_route_graph(self, region: Optional[str] = None) -> RouteGraph:
        """Belirtilen bölgenin rota çizgesini (RouteGraph) döndürür."""
        target_reg = str(region or self.active_region or DEFAULT_REGION)
        if target_reg not in self._route_graphs:
            self._route_graphs[target_reg] = RouteGraph(region=target_reg)
        return self._route_graphs[target_reg]

    def build_route_graph(self, region: Optional[str] = None) -> RouteGraph:
        """Tüm kayıtlı düğümleri ve ara noktaları çizgeye entegre eder.

        P1-6 (K-06): idempotent — `RouteGraph.add_edge` aynı kenarı
        ikinci kez eklemez, o yüzden tekrar çağrı `total_distance`'ı
        şişirmez.
        """
        target_reg = str(region or self.active_region or DEFAULT_REGION)
        rg = self.get_route_graph(target_reg)
        for tracked in self._tracked_nodes.get(target_reg, []):
            rg.connect_node_to_nearest_waypoint(
                node_id=tracked.node_id,
                node_map_x=tracked.map_x,
                node_map_y=tracked.map_y,
            )
        return rg

    def export_route_graph(self, region: Optional[str] = None) -> Dict[str, Any]:
        """Bölgenin rota çizgesini sözlük olarak dışa aktarır."""
        return self.build_route_graph(region).to_dict()

    def get_navigation_sequence(
        self, region: Optional[str] = None, loop: bool = True
    ) -> List[Dict[str, Any]]:
        """Otonom keşif için sıralı gezinme listesini döndürür."""
        rg = self.build_route_graph(region)
        return rg.build_connected_sequence(loop=loop)

    def _record_breadcrumb_if_moved(
        self,
        map_x: float,
        map_y: float,
        timestamp: Optional[float] = None,
        region: Optional[str] = None,
    ) -> Optional[GraphWaypoint]:
        """
        AUTO CARTOGRAPHY modundayken, oyuncu her `breadcrumb_interval_px`
        yürüdüğünde yeni bir navigasyon ara noktası (waypoint) oluşturur ve önceki noktayla kenar kurar.

        H-sweep (dilim 13 O11): ilk breadcrumb kalibre edilmemiş köken
        (harita merkezi 960, 540) olabilir — başlangıç konumu dünya
        haritasıyla kalibre edilene kadar varsayılandır; last_mapped_info
        bu yüzden 'calibrated' bayrağı taşır.
        """
        if not self.auto_cartography_enabled:
            return None

        target_reg = str(region or self.active_region or DEFAULT_REGION)
        rg = self.get_route_graph(target_reg)
        ts = float(timestamp or time.time())

        if self._last_breadcrumb_x is None or self._last_breadcrumb_y is None:
            wp = rg.add_waypoint(
                map_x=map_x,
                map_y=map_y,
                norm_x=self.norm_x,
                norm_y=self.norm_y,
                map_width=self.map_width,
                map_height=self.map_height,
                timestamp=ts,
            )
            self._last_breadcrumb_x = float(map_x)
            self._last_breadcrumb_y = float(map_y)
            self._last_breadcrumb_wp_id = wp.waypoint_id
            self.auto_mapped_count += 1
            self.last_mapped_info = {
                "type": "waypoint",
                "waypoint_id": wp.waypoint_id,
                "map_x": round(float(map_x), 2),
                "map_y": round(float(map_y), 2),
                "timestamp": ts,
                # H-sweep (dilim 13 O11): hangi bölge + kalibre durumu.
                "region": target_reg,
                "calibrated": bool(self._is_calibrated_on_world_map),
            }
            return wp

        dist = math.hypot(map_x - self._last_breadcrumb_x, map_y - self._last_breadcrumb_y)
        if dist >= self.breadcrumb_interval_px:
            wp = rg.add_waypoint(
                map_x=map_x,
                map_y=map_y,
                norm_x=self.norm_x,
                norm_y=self.norm_y,
                map_width=self.map_width,
                map_height=self.map_height,
                timestamp=ts,
            )
            if self._last_breadcrumb_wp_id is not None:
                rg.add_edge(
                    from_id=self._last_breadcrumb_wp_id,
                    to_id=wp.waypoint_id,
                    from_type="waypoint",
                    to_type="waypoint",
                    distance_px=dist,
                    bidirectional=True,
                )
            self._last_breadcrumb_x = float(map_x)
            self._last_breadcrumb_y = float(map_y)
            self._last_breadcrumb_wp_id = wp.waypoint_id
            self.auto_mapped_count += 1
            self.last_mapped_info = {
                "type": "waypoint",
                "waypoint_id": wp.waypoint_id,
                "map_x": round(float(map_x), 2),
                "map_y": round(float(map_y), 2),
                "timestamp": ts,
                # H-sweep (dilim 13 O11): hangi bölge + kalibre durumu.
                "region": target_reg,
                "calibrated": bool(self._is_calibrated_on_world_map),
            }
            return wp

        return None

    def persist_to_region_data(
        self,
        region: Optional[str] = None,
        base_dir: str = "routes",
    ) -> bool:
        """
        Otonom haritalandırma verilerini (düğümler, rota çizgesi ve istatistikler)
        doğrudan `routes/<region>/region_data.json` dosyasına atomik olarak yazar.
        """
        try:
            target_region = str(region or self.active_region or DEFAULT_REGION)
            reg_dir = os.path.join(base_dir, target_region)
            os.makedirs(reg_dir, exist_ok=True)
            patches_dir = os.path.join(reg_dir, "patches")
            os.makedirs(patches_dir, exist_ok=True)

            if self.route_manager is not None:
                self.route_manager.save_region(target_region)

            json_path = os.path.join(reg_dir, "region_data.json")
            existing_data: Dict[str, Any] = {}
            if os.path.isfile(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as f:
                        existing_data = json.load(f)
                except Exception:
                    existing_data = {}

            rg = self.build_route_graph(target_region)
            nodes_data = existing_data.get("nodes", [])
            if not nodes_data:
                nodes_data = [t.to_dict() for t in self._tracked_nodes.get(target_region, [])]

            payload = {
                "region": target_region,
                "routes": existing_data.get("routes", [DEFAULT_ROUTE]),
                "nodes": nodes_data,
                "route_graph": rg.to_dict(),
                "cartography_stats": self.get_cartography_stats(target_region),
                "updated_at": time.time(),
            }

            tmp_path = json_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except (AttributeError, OSError):
                    pass

            for _ in range(5):
                try:
                    os.replace(tmp_path, json_path)
                    return True
                except PermissionError:
                    time.sleep(0.02)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            return False
        except Exception as exc:
            logger.error("persist_to_region_data hatası: %s", exc)
            return False

    @property
    def cluster_radius(self) -> float:
        """Uzamsal kümeleme ve tekilleştirme yarıçapı (15-25 piksel)."""
        return float(np.clip(self.dedup_radius_px, DEDUP_RADIUS_MIN_PX, DEDUP_RADIUS_MAX_PX))

    @cluster_radius.setter
    def cluster_radius(self, radius: float) -> None:
        self.dedup_radius_px = float(np.clip(radius, DEDUP_RADIUS_MIN_PX, DEDUP_RADIUS_MAX_PX))

    def set_position(
        self,
        map_x: float,
        map_y: float,
        map_width: Optional[int] = None,
        map_height: Optional[int] = None,
        record_trajectory: bool = True,
    ) -> Tuple[float, float, float, float]:
        """Oyuncunun harita konumunu ayarlar ve odometri takipçisini senkronize eder."""
        if map_width is not None and map_width >= 160:
            self.map_width = int(map_width)
            self.odometry_tracker.map_width = int(map_width)
        if map_height is not None and map_height >= 120:
            self.map_height = int(map_height)
            self.odometry_tracker.map_height = int(map_height)
        self.map_x = float(np.clip(map_x, 0.0, float(self.map_width - 1)))
        self.map_y = float(np.clip(map_y, 0.0, float(self.map_height - 1)))
        self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
        self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
        self.odometry_tracker.map_x = float(self.map_x)
        self.odometry_tracker.map_y = float(self.map_y)
        self.odometry_tracker.norm_x = float(self.norm_x)
        self.odometry_tracker.norm_y = float(self.norm_y)
        self.dead_reckoning_dx = 0.0
        self.dead_reckoning_dy = 0.0
        if record_trajectory:
            self._append_trajectory()
        return self.get_position()

    def offset_to_world_coords(
        self,
        dx: float,
        dy: float,
        is_map_offset: bool = True,
        dead_reckoning_dx: float = 0.0,
        dead_reckoning_dy: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        """
        Harita veya radar ofsetini güncel oyuncu konumuna ve odometriye göre
        dünya koordinatlarına `(world_x, world_y, norm_x, norm_y)` dönüştürür.
        Merkeze sabitlenmeyi (center-sticking) önlemek için odometri takipçisiyle
        tam senkronize çalışır ve dead-reckoning yer değiştirmesini hesaba katar.
        `is_map_offset=False` ise ofset minimap pikselidir ve
        `minimap_to_map_scale` ile harita pikseline çevrilir (F-0003).
        """
        if not is_map_offset:
            dx = float(dx) * self.minimap_to_map_scale
            dy = float(dy) * self.minimap_to_map_scale
            is_map_offset = True
        self.odometry_tracker.map_x = float(self.map_x)
        self.odometry_tracker.map_y = float(self.map_y)
        self.odometry_tracker.map_width = int(self.map_width)
        self.odometry_tracker.map_height = int(self.map_height)
        return self.odometry_tracker.offset_to_world_coords(
            dx=dx,
            dy=dy,
            is_map_offset=is_map_offset,
            dead_reckoning_dx=float(dead_reckoning_dx or self.dead_reckoning_dx),
            dead_reckoning_dy=float(dead_reckoning_dy or self.dead_reckoning_dy),
        )

    # ------------------------------------------------------------------
    # Radar & Harita Olay Bildirim Sistemi (Callbacks & UI Listeners)
    # ------------------------------------------------------------------
    def add_node_discovered_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Yeni bir düğüm haritalandırıldığında anında tetiklenecek geri çağırma işlevini ekler."""
        if callable(callback) and callback not in self._node_discovered_callbacks:
            self._node_discovered_callbacks.append(callback)

    def remove_node_discovered_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Düğüm keşif dinleyicisini kaldırır."""
        if callback in self._node_discovered_callbacks:
            self._node_discovered_callbacks.remove(callback)

    def add_node_updated_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Mevcut düğüm yeni gözlemle güncellendiğinde/tekilleştirildiğinde tetiklenecek dinleyiciyi ekler."""
        if callable(callback) and callback not in self._node_updated_callbacks:
            self._node_updated_callbacks.append(callback)

    def remove_node_updated_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Düğüm güncelleme dinleyicisini kaldırır."""
        if callback in self._node_updated_callbacks:
            self._node_updated_callbacks.remove(callback)

    def add_radar_event_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Genel radar olayları (node_discovered, node_updated vb.) için dinleyici ekler."""
        if callable(callback) and callback not in self._radar_event_callbacks:
            self._radar_event_callbacks.append(callback)

    def remove_radar_event_callback(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        """Genel radar olay dinleyicisini kaldırır."""
        if callback in self._radar_event_callbacks:
            self._radar_event_callbacks.remove(callback)

    def register_ui_listener(self, listener: Any) -> None:
        """Radar tuvali (SequentialMapCanvas/CartographyRadarDialog) veya Harita (MapViewerWindow) bağlar."""
        if listener is not None and listener not in self._ui_listeners:
            self._ui_listeners.append(listener)

    def unregister_ui_listener(self, listener: Any) -> None:
        """UI dinleyicisini kaldırır."""
        if listener in self._ui_listeners:
            self._ui_listeners.remove(listener)

    def _emit_node_discovered(self, node_dict: Dict[str, Any]) -> None:
        """Yeni düğüm keşfedildiğinde tüm kayıtlı dinleyicileri ve UI bileşenlerini anında haberdar eder."""
        for cb in list(self._node_discovered_callbacks):
            try:
                cb(node_dict)
            except Exception as exc:
                logger.debug("node_discovered_callback hatası: %s", exc)
        for listener in list(self._ui_listeners):
            try:
                if hasattr(listener, "add_discovered_node"):
                    listener.add_discovered_node(node_dict, trigger_pulse=True)
                elif hasattr(listener, "reload_map_and_nodes"):
                    listener.reload_map_and_nodes()
                elif hasattr(listener, "emit") and callable(listener.emit):
                    listener.emit(node_dict)
            except Exception as exc:
                logger.debug("UI listener node_discovered hatası: %s", exc)
        self._emit_radar_event("node_discovered", node_dict)

    def _emit_node_updated(self, node_dict: Dict[str, Any]) -> None:
        """Mevcut düğüm güncellendiğinde UI'ı ve dinleyicileri anında günceller."""
        for cb in list(self._node_updated_callbacks):
            try:
                cb(node_dict)
            except Exception as exc:
                logger.debug("node_updated_callback hatası: %s", exc)
        for listener in list(self._ui_listeners):
            try:
                if hasattr(listener, "update_node"):
                    listener.update_node(node_dict)
                elif hasattr(listener, "reload_map_and_nodes"):
                    listener.reload_map_and_nodes()
            except Exception as exc:
                logger.debug("UI listener node_updated hatası: %s", exc)
        self._emit_radar_event("node_updated", node_dict)

    def _emit_radar_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Genel radar olayını yayınlar."""
        for cb in list(self._radar_event_callbacks):
            try:
                cb(event_type, data)
            except Exception as exc:
                logger.debug("radar_event_callback hatası: %s", exc)

    def cluster_detections(
        self,
        parsed_items: List[Dict[str, Any]],
        cluster_radius: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Aynı kare veya ardışık akış içindeki birden fazla materyal tespitini
        belirtilen `cluster_radius` (15-25 px aralığına kenetlenmiş) içinde
        güven ağırlıklı merkez hesaplamasıyla (confidence-weighted centroid) kümeleyerek
        üst üste binen sahte düğüm ve mükerrer yığılma (duplicate piling) üretimini önler.
        """
        raw_r = cluster_radius if cluster_radius is not None else self.dedup_radius_px
        # Skaler np.clip yerine min/max: np.clip skalerde dizi kurar (D02).
        radius = min(DEDUP_RADIUS_MAX_PX, max(DEDUP_RADIUS_MIN_PX, float(raw_r)))
        if not parsed_items or len(parsed_items) <= 1:
            return list(parsed_items or [])

        SOURCE_PRIORITY = CLUSTER_SOURCE_PRIORITY

        clusters: List[Dict[str, Any]] = []
        # Küme merkezlerinin önbelleği: iç döngüde her karşılaştırmada
        # dict.get + float() tekrarı yapılmaz (D02). Değerler sözlükteki
        # yuvarlanmış değerlerle birebir aynı tutulur.
        _cl_dx: List[float] = []
        _cl_dy: List[float] = []
        _cl_ex: List[Optional[float]] = []
        _cl_ey: List[Optional[float]] = []
        for item in parsed_items:
            ix = float(item.get("rel_map_dx", 0.0))
            iy = float(item.get("rel_map_dy", 0.0))
            item_exact_x = item.get("exact_map_x")
            item_exact_y = item.get("exact_map_y")
            item_conf = min(1.0, max(0.05, float(item.get("confidence", 0.5))))
            item_source = str(item.get("source_type", "minimap_auto"))

            matched_cluster: Optional[Dict[str, Any]] = None
            matched_idx: int = -1
            min_d = float("inf")

            for j, cl in enumerate(clusters):
                # Mutlak koordinat varsa onu da uzamsal mesafede kullan
                if item_exact_x is not None and _cl_ex[j] is not None:
                    d = math.hypot(float(item_exact_x) - _cl_ex[j], float(item_exact_y) - _cl_ey[j])
                else:
                    d = math.hypot(ix - _cl_dx[j], iy - _cl_dy[j])

                if d <= radius and d < min_d:
                    min_d = d
                    matched_cluster = cl
                    matched_idx = j

            if matched_cluster is not None:
                cnt = int(matched_cluster.get("_cluster_count", 1))
                new_cnt = cnt + 1
                matched_cluster["_cluster_count"] = new_cnt

                # Güven ağırlıklı merkez (confidence-weighted centroid)
                cl_conf = min(1.0, max(0.05, float(matched_cluster.get("confidence", 0.5))))
                w_old = cl_conf * float(cnt)
                w_new = item_conf
                w_sum = max(0.001, w_old + w_new)

                matched_cluster["rel_map_dx"] = round(
                    (w_old * float(matched_cluster["rel_map_dx"]) + w_new * ix) / w_sum, 3
                )
                matched_cluster["rel_map_dy"] = round(
                    (w_old * float(matched_cluster["rel_map_dy"]) + w_new * iy) / w_sum, 3
                )
                _cl_dx[matched_idx] = float(matched_cluster["rel_map_dx"])
                _cl_dy[matched_idx] = float(matched_cluster["rel_map_dy"])

                if item_exact_x is not None and matched_cluster.get("exact_map_x") is not None:
                    matched_cluster["exact_map_x"] = round(
                        (w_old * float(matched_cluster["exact_map_x"]) + w_new * float(item_exact_x)) / w_sum, 3
                    )
                    matched_cluster["exact_map_y"] = round(
                        (w_old * float(matched_cluster["exact_map_y"]) + w_new * float(item_exact_y)) / w_sum, 3
                    )
                    _cl_ex[matched_idx] = float(matched_cluster["exact_map_x"])
                    _cl_ey[matched_idx] = float(matched_cluster["exact_map_y"])

                if "screen_x" in item and "screen_x" in matched_cluster:
                    matched_cluster["screen_x"] = int(round(
                        (w_old * float(matched_cluster["screen_x"]) + w_new * float(item["screen_x"])) / w_sum
                    ))
                if "screen_y" in item and "screen_y" in matched_cluster:
                    matched_cluster["screen_y"] = int(round(
                        (w_old * float(matched_cluster["screen_y"]) + w_new * float(item["screen_y"])) / w_sum
                    ))

                matched_cluster["confidence"] = round(max(cl_conf, item_conf), 3)
                matched_cluster["stack_count"] = max(
                    int(matched_cluster.get("stack_count", 1)),
                    int(item.get("stack_count", 1)),
                )
                matched_cluster["max_harvests"] = max(
                    int(matched_cluster.get("max_harvests", DEFAULT_MAX_HARVESTS)),
                    int(item.get("max_harvests", DEFAULT_MAX_HARVESTS)),
                )

                # Kaynak tipi belirleme: genel isim yerine daima spesifik ismi koru
                item_res = str(item.get("resource_type", "")).strip()
                cur_res = str(matched_cluster.get("resource_type", "")).strip()
                generic_names = (DEFAULT_RESOURCE_TYPE, "Gatherable Material", "resource", "unknown", "")
                if item_res and item_res not in generic_names:
                    matched_cluster["resource_type"] = item_res
                elif cur_res in generic_names and item_res:
                    matched_cluster["resource_type"] = item_res

                # Kaynak kaynağı güvenilirlik önceliği
                cl_source = str(matched_cluster.get("source_type", "minimap_auto"))
                if SOURCE_PRIORITY.get(item_source, 0) > SOURCE_PRIORITY.get(cl_source, 0):
                    matched_cluster["source_type"] = item_source
            else:
                new_cl = dict(item)
                new_cl["_cluster_count"] = 1
                clusters.append(new_cl)
                _cl_dx.append(float(new_cl.get("rel_map_dx", 0.0)))
                _cl_dy.append(float(new_cl.get("rel_map_dy", 0.0)))
                _cl_ex.append(
                    float(new_cl["exact_map_x"]) if new_cl.get("exact_map_x") is not None else None
                )
                _cl_ey.append(
                    float(new_cl["exact_map_y"]) if new_cl.get("exact_map_y") is not None else None
                )

        return clusters

    def _get_quality_detector(self) -> Any:
        """Lazy-load gatherion CPU detector for the pre-save quality gate."""
        if self._quality_detector is not None:
            return self._quality_detector
        if self._quality_detector_failed:
            return None
        try:
            from core.wrong_node_scan import load_gatherion_detector

            self._quality_detector = load_gatherion_detector()
        except Exception as exc:
            self._quality_detector_failed = True
            logger.info("Haritalama AI kapısı modeli yüklenemedi (CV-only devam): %s", exc)
            self._quality_detector = None
        return self._quality_detector

    def _get_exemplar_memory(self) -> Optional[Any]:
        """Lazy-load or return ExemplarMemory singleton instance."""
        if getattr(self, "exemplar_memory", None) is not None:
            return self.exemplar_memory
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory
            return get_exemplar_memory()
        except Exception as exc:
            logger.debug("Failed to get ExemplarMemory in AutoCartographer: %s", exc)
            return None

    def evaluate_node_candidate(
        self,
        patch: Optional[np.ndarray],
        source_type: str = "minimap_auto",
        resource_type: str = "",
        conf_min: float = MODEL_CONF_MIN,
        exemplar_memory: Optional[Any] = None,
        **kwargs: Any,
    ) -> GateVerdict:
        """Evaluates a candidate node patch against the multi-signal gate including ExemplarMemory."""
        mem = exemplar_memory or self._get_exemplar_memory()
        return evaluate_node_candidate(
            patch,
            source_type=source_type,
            resource_type=resource_type,
            detector=self._get_quality_detector(),
            conf_min=conf_min,
            exemplar_memory=mem,
            check_feather=kwargs.get("check_feather", True),
            check_monster=kwargs.get("check_monster", source_type.startswith("minimap")),
            check_exemplars=kwargs.get("check_exemplars", True),
            check_terrain=kwargs.get("check_terrain", True),
        )

    def _refresh_calibration_flag(self) -> None:
        """Set needs_map_calibrate when drift or wall-clock interval warrants a world-map fix."""
        now = time.time()
        stale = (now - float(self._last_calibration_ts or 0.0)) >= ODOMETRY_CALIBRATE_INTERVAL_SEC
        # H-sweep (dilim 13 Y7): drift dalı ÖLÜYDÜ — _cumulative_drift_px
        # hiç artırılmadığı için her zaman 0.0'dı; bayrağa sahte bir
        # 'drift yok' güvencesi katıyordu. Kaldırıldı; yalnızca kalibre
        # yokluğu ve 90 sn zaman aşımı kalır.
        uncalibrated = not bool(self._is_calibrated_on_world_map)
        self.needs_map_calibrate = bool(uncalibrated or stale)

    # ------------------------------------------------------------------
    # Yardımcı Görüntü & ROI Metodları
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_bgr(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Gelen görüntüyü güvenli biçimde 3 kanallı BGR uint8 dizisine dönüştürür."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        try:
            if frame.ndim == 2:
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 3 and frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            if frame.ndim == 3 and frame.shape[2] == 3:
                return frame
        except Exception as exc:
            logger.error("Görüntü BGR formatına dönüştürülemedi: %s", exc)
        return None

    def _get_minimap_pixel_rect(self, frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """
        `norm_rect = (0.855, 0.065, 0.137, 0.190)` (`x: 0.855..0.992, y: 0.065..0.255`)
        normalize koordinatlarını piksel `(x1, y1, x2, y2)` sınırlarına çevirir.
        Eğer verilen görüntü zaten kırpılmış bir minimap/yakın çekim ise uygun alt bölgeyi döndürür.
        """
        if frame_w < 800 and frame_h < 500:
            # Yakın çekim (closeup) çerçevelerinde minimap sağ yarıdadır;
            # dikdörtgen CLOSEUP_MINIMAP_NORM_RECT ile tanımlanır (F-0408).
            if frame_w >= 600 and frame_h <= 420:
                crx, cry, crw, crh = CLOSEUP_MINIMAP_NORM_RECT
                rect = (
                    int(round(frame_w * crx)),
                    int(round(frame_h * cry)),
                    int(round(frame_w * (crx + crw))),
                    int(round(frame_h * (cry + crh))),
                )
                if rect[2] - rect[0] >= 32 and rect[3] - rect[1] >= 32:
                    return rect
            return 0, 0, frame_w, frame_h

        nx, ny, nw, nh = self.minimap_norm_rect
        x1 = int(np.clip(round(frame_w * nx), 0, max(0, frame_w - 16)))
        y1 = int(np.clip(round(frame_h * ny), 0, max(0, frame_h - 16)))
        x2 = int(np.clip(round(frame_w * (nx + nw)), x1 + 16, frame_w))
        y2 = int(np.clip(round(frame_h * (ny + nh)), y1 + 16, frame_h))
        return x1, y1, x2, y2

    def extract_minimap_roi(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Sağ üstteki minimap bölgesini (`x: 0.855..0.992, y: 0.065..0.255`) BGR olarak kırpar."""
        bgr = self._ensure_bgr(frame_bgr)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = self._get_minimap_pixel_rect(w, h)
        roi = bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi.copy()

    @staticmethod
    def _locate_minimap_player_arrow(minimap_bgr: np.ndarray) -> Tuple[float, float]:
        """
        Minimap kesiti içindeki oyuncu okunun `(arrow_cx, arrow_cy)` piksel merkezini bulur.
        Başlık çubuğu (`Pilgrim's Ridge`, üst %15) nedeniyle radar dairesinin merkezi
        genellikle `(mw * 0.50, mh * 0.55)` civarındadır.
        """
        mh, mw = minimap_bgr.shape[:2]
        default_cx = mw * 0.50
        default_cy = mh * 0.545

        try:
            hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
            sx1 = max(0, int(round(mw * 0.40)))
            sx2 = min(mw, int(round(mw * 0.60)))
            sy1 = max(0, int(round(mh * 0.42)))
            sy2 = min(mh, int(round(mh * 0.68)))

            sub_hsv = hsv[sy1:sy2, sx1:sx2]
            if sub_hsv.size == 0:
                return default_cx, default_cy

            # Oyuncu oku parlak beyaz/cyan iç dolgulu ve koyu dış konturludur
            arrow_mask = cv2.inRange(sub_hsv, (0, 0, 205), (180, 45, 255))
            num, _, stats, cents = cv2.connectedComponentsWithStats(arrow_mask)
            best_cx, best_cy = default_cx, default_cy
            best_score = -1.0

            for idx in range(1, num):
                bx, by, bw, bh, area = stats[idx]
                if area < 10 or area > 130 or bw < 4 or bh < 4 or bw > 24 or bh > 24:
                    continue
                cand_cx = float(sx1) + float(cents[idx][0])
                cand_cy = float(sy1) + float(cents[idx][1])
                dist_to_expected = math.hypot(cand_cx - default_cx, cand_cy - default_cy)
                score = float(area) - dist_to_expected * 1.8
                if score > best_score:
                    best_score = score
                    best_cx, best_cy = cand_cx, cand_cy

            return best_cx, best_cy
        except Exception as exc:
            logger.debug("Minimap oyuncu oku merkezi hesaplanamadı: %s", exc)
            return default_cx, default_cy

    def _is_world_map_open(self, frame_bgr: np.ndarray) -> Optional[bool]:
        """Dünya haritasının (`M`) açık olup olmadığını kontrol eder.

        Üç durumlu (H-sweep ek): True = açık, False = kapalı,
        None = bilinmiyor. Vision açıkça None döndürürse bool() ile
        False'a indirilmez, None yayılır; yerel sezgisel yedek yalnızca
        vision'da is_map_open yoksa devreye girer. Hata da bilinmiyor
        sayılır (eskiden False döner, fail-open olurdu).
        """
        try:
            if self.vision_engine is not None and hasattr(self.vision_engine, "is_map_open"):
                result = self.vision_engine.is_map_open(frame_bgr)
                if result is None:
                    return None
                return bool(result)

            h, w = frame_bgr.shape[:2]
            if h < 120 or w < 160:
                return False

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            # Normal oyun ekranındaki alt HP/MP bar kontrolü
            hp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.36) : int(w * 0.47)]
            mp_roi = hsv[int(h * 0.87) : int(h * 0.92), int(w * 0.52) : int(w * 0.63)]
            hp_red = float(
                np.mean(
                    ((hp_roi[:, :, 0] < 12) | (hp_roi[:, :, 0] > 168))
                    & (hp_roi[:, :, 1] > 110)
                    & (hp_roi[:, :, 2] > 90)
                )
            )
            mp_blue = float(
                np.mean(
                    (mp_roi[:, :, 0] > 85)
                    & (mp_roi[:, :, 0] < 115)
                    & (mp_roi[:, :, 1] > 110)
                    & (mp_roi[:, :, 2] > 90)
                )
            )
            if hp_red > 0.06 or mp_blue > 0.06:
                return False

            left_hsv = hsv[int(h * 0.22) : int(h * 0.70), int(w * 0.01) : int(w * 0.18)]
            left_gray = gray[int(h * 0.22) : int(h * 0.70), int(w * 0.01) : int(w * 0.18)]
            canvas_hsv = hsv[int(h * 0.15) : int(h * 0.70), int(w * 0.25) : int(w * 0.92)]
            x_gray = gray[int(h * 0.018) : int(h * 0.062), int(w * 0.965) : int(w * 0.994)]

            return bool(
                float(np.mean(left_hsv[:, :, 1])) < 38.0
                and 25.0 <= float(np.mean(left_hsv[:, :, 2])) <= 90.0
                and float(np.mean(left_gray > 145)) > 0.015
                and float(np.mean(canvas_hsv[:, :, 1])) < 48.0
                and float(np.mean(canvas_hsv[:, :, 2])) > 72.0
                and float(np.mean(x_gray > 150)) > 0.015
            )
        except Exception as exc:
            logger.debug("Dünya haritası kontrolünde hata: %s", exc)
            # Hata = bilinmiyor (None), kapalı (False) değil.
            return None

    def _is_minimap_occluded(self, frame_bgr: np.ndarray, minimap_bgr: np.ndarray) -> bool:
        """
        Sağ üst menü / envanter / Cube penceresinin minimap'i kapatıp kapatmadığını doğrular.
        """
        try:
            if self.vision_engine is not None and hasattr(self.vision_engine, "is_game_window_blocking"):
                if bool(self.vision_engine.is_game_window_blocking(frame_bgr)):
                    return True

            if minimap_bgr is None or minimap_bgr.size == 0:
                return True

            hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
            mean_val = float(np.mean(hsv[:, :, 2]))
            std_val = float(np.std(hsv[:, :, 2]))
            # Sağ menü açıkken minimap bölgesi koyu düz bir panelle kaplanır (mean_val ~ 22-45)
            # R14b: std eşiği 8.0, dümdüz ama geçerli bir minimap'i (tek renk
            # zemin + küçük ikon: V std ~6.5) örtülü sayıyordu; patch çıkarma
            # o zaman sıfır patch döndürüyor, sıfır patch de kapıda öğrenilmiş
            # saf siyah negatif exemplare (benzerlik 1.0) takılıp gerçek
            # tespiti düşürüyordu. Düz örtü panelleri std ~0-2 bandındadır;
            # eşik 5.0 o korumayı korurken düz zeminli minimap'i geçirir.
            if mean_val < 52.0 or std_val < 5.0:
                return True
            return False
        except Exception as exc:
            logger.debug("Minimap kapanma kontrolü hatası: %s", exc)
            # Hata durumunda güvenli yön kapalı saymaktır (F-0210): şüpheli
            # kare atlanır, kör odometri/düğüm yazılmaz.
            return True

    @staticmethod
    def _clean_minimap_for_odometry(minimap_bgr: np.ndarray) -> np.ndarray:
        """
        Minimap üzerindeki sabit oyuncu okunu, kenar çerçevelerini ve parlak UI
        ikonlarını maskeleyip inpaint ederek saf harita dokusunu (gri tonlamalı) döndürür.
        """
        try:
            gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
            mh, mw = gray.shape[:2]
            cx, cy = mw // 2, int(round(mh * 0.545))

            mask = np.zeros((mh, mw), dtype=np.uint8)
            arrow_radius = max(7, int(round(min(mw, mh) * 0.08)))
            cv2.circle(mask, (cx, cy), arrow_radius, 255, -1)

            hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
            center_zone = np.zeros((mh, mw), dtype=np.uint8)
            cv2.circle(center_zone, (cx, cy), max(12, int(round(min(mw, mh) * 0.14))), 255, -1)
            arrow_pixels = (
                ((hsv[:, :, 1] > 115) & (hsv[:, :, 2] > 140)) | (hsv[:, :, 2] > 230)
            ).astype(np.uint8) * 255
            mask = cv2.bitwise_or(mask, cv2.bitwise_and(arrow_pixels, center_zone))

            cleaned = cv2.inpaint(gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return cleaned
        except Exception as exc:
            logger.debug("Minimap odometri temizleme hatası: %s", exc)
            return cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # 1. Görsel Odometri ve Konum Takibi (Minimap SLAM + World Map Calibration)
    # ------------------------------------------------------------------
    def _append_trajectory(self, timestamp: Optional[float] = None) -> None:
        """Oyuncunun güncel konumunu `walk_trajectory` listesine ekler."""
        self.odometry_tracker.map_x = float(self.map_x)
        self.odometry_tracker.map_y = float(self.map_y)
        self.odometry_tracker.map_width = int(self.map_width)
        self.odometry_tracker.map_height = int(self.map_height)
        self.odometry_tracker.append_trajectory(timestamp=timestamp)
        if self.auto_cartography_enabled:
            self._record_breadcrumb_if_moved(self.map_x, self.map_y, timestamp=timestamp)

    def calibrate_on_world_map(
        self,
        open_map_frame: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Dünya haritası (`M`) açıkken oyuncunun mutlak harita üzerindeki konumunu
        sıfır hatayla kalibre eder ve `(map_x, map_y, norm_x, norm_y)` döndürür.
        """
        try:
            bgr = self._ensure_bgr(open_map_frame)
            if bgr is None:
                return self.get_position()

            h, w = bgr.shape[:2]
            # Harita uzayı korunur (F-0053): dünya haritası karesi (örn. 1280x720)
            # harita uzayını (1920x1080) değiştirmez; oyuncu pikseli harita
            # uzayına oranla çevrilir.
            scale_to_map_x = float(self.map_width) / float(max(1, w))
            scale_to_map_y = float(self.map_height) / float(max(1, h))

            player_x: Optional[float] = None
            player_y: Optional[float] = None

            # 1. VisionEngine üzerinden oyuncu okunu bulmayı dene
            if self.vision_engine is not None and hasattr(self.vision_engine, "find_player_on_map"):
                try:
                    pos = self.vision_engine.find_player_on_map(bgr)
                    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                        px, py = float(pos[0]), float(pos[1])
                        if 0 <= px < w and 0 <= py < h:
                            player_x, player_y = px, py
                except Exception as exc:
                    logger.debug("vision_engine.find_player_on_map hatası: %s", exc)

            # 2. HSV tabanlı cyan/sarı/beyaz oyuncu oku tespiti
            if player_x is None or player_y is None or (
                abs(player_x - w / 2.0) < 1.0 and abs(player_y - h / 2.0) < 1.0
            ):
                hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                cx, cy = w // 2, h // 2
                r = max(32, int(round(min(w, h) * 0.12)))
                x1, y1 = max(0, cx - r), max(0, cy - r)
                x2, y2 = min(w, cx + r), min(h, cy + r)
                roi_hsv = hsv[y1:y2, x1:x2]
                cyan_mask = (
                    (roi_hsv[:, :, 0] >= 85)
                    & (roi_hsv[:, :, 0] <= 135)
                    & (roi_hsv[:, :, 1] > 95)
                    & (roi_hsv[:, :, 2] > 145)
                )
                ys, xs = np.where(cyan_mask)
                if len(xs) >= 6:
                    player_x = float(x1) + float(np.mean(xs))
                    player_y = float(y1) + float(np.mean(ys))

            # 3. Son geçerli minimap dokusu varsa açık dünya haritası üzerinde şablon eşleştirme ile doğrula
            if self._prev_minimap_clean is not None and self._prev_minimap_clean.size > 0:
                try:
                    map_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                    mh, mw = self._prev_minimap_clean.shape[:2]
                    half = min(PATCH_SIZE // 2, mh // 2 - 4, mw // 2 - 4)
                    if half >= 20:
                        mcx, mcy = mw // 2, mh // 2
                        tpl = self._prev_minimap_clean[mcy - half : mcy + half, mcx - half : mcx + half]
                        sx1, sy1 = int(w * 0.20), int(h * 0.10)
                        sx2, sy2 = int(w * 0.94), int(h * 0.88)
                        search_area = map_gray[sy1:sy2, sx1:sx2]
                        if search_area.shape[0] > tpl.shape[0] and search_area.shape[1] > tpl.shape[1]:
                            res = cv2.matchTemplate(search_area, tpl, cv2.TM_CCOEFF_NORMED)
                            _, max_val, _, max_loc = cv2.minMaxLoc(res)
                            if float(max_val) >= 0.72:
                                matched_x = float(sx1 + max_loc[0] + tpl.shape[1] // 2)
                                matched_y = float(sy1 + max_loc[1] + tpl.shape[0] // 2)
                                if player_x is None or float(max_val) >= 0.82:
                                    player_x, player_y = matched_x, matched_y
                except Exception as exc:
                    logger.debug("Dünya haritası üzerinde minimap şablon kalibrasyonu atlandı: %s", exc)

            if player_x is None or player_y is None:
                # Kalibrasyon bulunamadı: ekran merkezi "sıfır hata" değildir
                # (F-0001). Konum değiştirilmez, kalibre sayılmaz.
                logger.warning(
                    "Dünya haritası üzerinde oyuncu oku bulunamadı; kalibrasyon yapılmadı."
                )
                self._is_calibrated_on_world_map = False
                self.needs_map_calibrate = True
                return self.get_position()

            self.map_x = float(
                np.clip(player_x * scale_to_map_x, 0.0, float(self.map_width - 1))
            )
            self.map_y = float(
                np.clip(player_y * scale_to_map_y, 0.0, float(self.map_height - 1))
            )
            self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
            self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)

            self.odometry_tracker.map_x = float(self.map_x)
            self.odometry_tracker.map_y = float(self.map_y)
            self.odometry_tracker.norm_x = float(self.norm_x)
            self.odometry_tracker.norm_y = float(self.norm_y)
            self.velocity_x = 0.0
            self.velocity_y = 0.0
            self.dead_reckoning_dx = 0.0
            self.dead_reckoning_dy = 0.0

            now_cal = time.time()
            prev_log_x = getattr(self, "_last_logged_cal_x", -999.0)
            prev_log_y = getattr(self, "_last_logged_cal_y", -999.0)
            prev_log_ts = getattr(self, "_last_logged_cal_ts", 0.0)

            self._is_calibrated_on_world_map = True
            self._last_calibration_ts = now_cal
            self._cumulative_drift_px = 0.0
            self.odometry_tracker.cumulative_drift_px = 0.0
            self.needs_map_calibrate = False
            self._append_trajectory(self._last_calibration_ts)

            if (
                math.hypot(self.map_x - prev_log_x, self.map_y - prev_log_y) >= 6.0
                or (now_cal - prev_log_ts) >= 12.0
            ):
                self._last_logged_cal_x = self.map_x
                self._last_logged_cal_y = self.map_y
                self._last_logged_cal_ts = now_cal
                logger.info(
                    "Dünya haritası üzerinde konum kalibre edildi: map=(%.1f, %.1f), norm=(%.4f, %.4f)",
                    self.map_x,
                    self.map_y,
                    self.norm_x,
                    self.norm_y,
                )

            # Görev 4: açık dünya haritası hangi bölgeninse ORB ile tespit et
            # (Verteron'u canlı doğrulayabilen tek yol); kısılmış (30 sn).
            try:
                if now_cal - float(self._last_world_map_region_ts or 0.0) >= REGION_WORLD_MAP_MATCH_INTERVAL_SEC:
                    self._last_world_map_region_ts = now_cal
                    self.identify_region_from_world_map(bgr)
            except Exception as exc:
                logger.debug("Dünya haritası bölge tespiti atlandı: %s", exc)
            return self.get_position()
        except Exception as exc:
            logger.error("calibrate_on_world_map hatası: %s", exc)
            return self.get_position()

    def _estimate_minimap_shift(
        self,
        prev_clean: np.ndarray,
        curr_clean: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        İki ardışık temizlenmiş minimap karesi arasındaki doku kaymasını `(dx_terrain, dy_terrain, confidence)`
        olarak `cv2.phaseCorrelate` ve özellik eşleştirme / optik akış ile hesaplar.
        """
        try:
            h, w = prev_clean.shape[:2]
            if curr_clean.shape[:2] != (h, w) or h < 32 or w < 32:
                return 0.0, 0.0, 0.0

            mx = max(8, int(round(w * 0.12)))
            my = max(8, int(round(h * 0.12)))
            roi_prev = prev_clean[my : h - my, mx : w - mx]
            roi_curr = curr_clean[my : h - my, mx : w - mx]
            rh, rw = roi_prev.shape[:2]

            if self._hann_window is None or self._hann_window.shape[:2] != (rh, rw):
                self._hann_window = cv2.createHanningWindow((rw, rh), cv2.CV_32F)

            # 1. Yöntem: cv2.phaseCorrelate
            f_prev = np.float32(roi_prev)
            f_curr = np.float32(roi_curr)
            (pc_dx, pc_dy), pc_resp = cv2.phaseCorrelate(f_prev, f_curr, self._hann_window)
            pc_dx = float(pc_dx)
            pc_dy = float(pc_dy)
            pc_resp = float(pc_resp)

            # 2. Yöntem: Shi-Tomasi Köşe Özellikleri + Lucas-Kanade Piramidal Optik Akış
            feat_dx: Optional[float] = None
            feat_dy: Optional[float] = None
            feat_conf: float = 0.0

            feat_mask = np.full((rh, rw), 255, dtype=np.uint8)
            cv2.circle(feat_mask, (rw // 2, rh // 2), max(8, int(min(rw, rh) * 0.12)), 0, -1)

            pts_prev = cv2.goodFeaturesToTrack(
                roi_prev,
                maxCorners=80,
                qualityLevel=0.02,
                minDistance=5,
                mask=feat_mask,
            )
            if pts_prev is not None and len(pts_prev) >= 4:
                pts_curr, status, _ = cv2.calcOpticalFlowPyrLK(
                    roi_prev,
                    roi_curr,
                    pts_prev,
                    None,
                    winSize=(15, 15),
                    maxLevel=2,
                )
                if pts_curr is not None and status is not None:
                    good_prev = pts_prev[status.flatten() == 1].reshape(-1, 2)
                    good_curr = pts_curr[status.flatten() == 1].reshape(-1, 2)
                    if len(good_prev) >= 4:
                        deltas = good_curr - good_prev
                        med_dx = float(np.median(deltas[:, 0]))
                        med_dy = float(np.median(deltas[:, 1]))
                        residuals = np.hypot(deltas[:, 0] - med_dx, deltas[:, 1] - med_dy)
                        inliers = deltas[residuals < 3.5]
                        if len(inliers) >= 3:
                            feat_dx = float(np.mean(inliers[:, 0]))
                            feat_dy = float(np.mean(inliers[:, 1]))
                            feat_conf = min(1.0, float(len(inliers)) / 12.0)

            # 3. ORB Eşleştirme desteği
            if feat_dx is None and self._odometry_Search_orb is not None:
                kp1, des1 = self._odometry_Search_orb.detectAndCompute(roi_prev, feat_mask)
                kp2, des2 = self._odometry_Search_orb.detectAndCompute(roi_curr, feat_mask)
                if des1 is not None and des2 is not None and len(kp1) >= 4 and len(kp2) >= 4:
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    if len(matches) >= 4:
                        dxs = [kp2[m.trainIdx].pt[0] - kp1[m.queryIdx].pt[0] for m in matches]
                        dys = [kp2[m.trainIdx].pt[1] - kp1[m.queryIdx].pt[1] for m in matches]
                        feat_dx = float(np.median(dxs))
                        feat_dy = float(np.median(dys))
                        feat_conf = min(1.0, float(len(matches)) / 15.0)

            max_step = float(min(rw, rh) * 0.42)
            if abs(pc_dx) > max_step or abs(pc_dy) > max_step:
                pc_dx, pc_dy, pc_resp = 0.0, 0.0, 0.0

            if feat_dx is not None and feat_dy is not None and abs(feat_dx) <= max_step and abs(feat_dy) <= max_step:
                if pc_resp >= 0.25:
                    w_pc = pc_resp / (pc_resp + feat_conf + 1e-6)
                    final_dx = w_pc * pc_dx + (1.0 - w_pc) * feat_dx
                    final_dy = w_pc * pc_dy + (1.0 - w_pc) * feat_dy
                    confidence = max(pc_resp, feat_conf)
                else:
                    final_dx = feat_dx
                    final_dy = feat_dy
                    confidence = feat_conf
            else:
                final_dx = pc_dx if pc_resp >= 0.15 else 0.0
                final_dy = pc_dy if pc_resp >= 0.15 else 0.0
                confidence = pc_resp

            if math.hypot(final_dx, final_dy) < 0.04:
                final_dx, final_dy = 0.0, 0.0

            return float(final_dx), float(final_dy), float(np.clip(confidence, 0.0, 1.0))
        except Exception as exc:
            logger.error("_estimate_minimap_shift hatası: %s", exc)
            return 0.0, 0.0, 0.0

    def update_player_position(
        self,
        frame_bgr: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Sağ üstteki Minimap bölgesini (`Pilgrim's Ridge`, `x: 0.855..0.992, y: 0.065..0.255`)
        takip ederek oyuncunun harita üzerindeki `(map_x, map_y, norm_x, norm_y)` konumunu günceller.
        Odometri kaybolduğunda dead-reckoning yer değiştirmesini devreye alır.
        Dünya haritası (`M`) açıksa `calibrate_on_world_map` ile sıfır hatayla kalibre eder.
        """
        try:
            bgr = self._ensure_bgr(frame_bgr)
            if bgr is None:
                return self.get_position()

            h, w = bgr.shape[:2]
            # Harita uzayı kanoniktir (F-0053): ilk kare 1280x720 bile olsa
            # harita uzayı 1920x1080'den 1280x720'ye döndürülmez; kare boyutu
            # map_width/map_height'i ve map_x/map_y'yi değiştirmez.
            # calibrate_on_world_map kare pikselini zaten harita uzayına
            # oranla çevirir; eski blok scale_to_map_x'i 1.0 yapıp düğümleri
            # ekran uzayında kaydettiriyordu.

            # 1. Dünya haritası ('M') açıksa mutlak kalibrasyon yap.
            # Bilinmiyor (None) = açık say: kalibrasyon denemesi içeride
            # korumalıdır (oyuncu oku bulunamazsa konum değişmez, F-0001);
            # kapalı saymak odometriyi harita karesi üzerinde kör
            # çalıştırırdı. Bedeli: bilinmiyor durumunda odometri zinciri
            # bir kareliğine sıfırlanır (aşağıdaki _prev_minimap_clean).
            if w >= 800 and self._is_world_map_open(bgr) is not False:
                calibrated = self.calibrate_on_world_map(bgr)
                # Dünya haritası karesinden önceki minimap dokusu bayattır (F-0054).
                self._prev_minimap_clean = None
                return calibrated

            # 1b. Canlı bölge tespiti (Görev 4, kısılmış): minimap açıktayken
            # oyuncu konumundan alt bölge/bölge çözümlenir; güvenilir tespit
            # RouteManager'a iletilir (uyumsuzluk koruması).
            self._maybe_detect_region(bgr)

            # 2. Sağ üst minimap bölgesini kırp
            minimap_bgr = self.extract_minimap_roi(bgr)
            if minimap_bgr is None:
                return self.get_position()

            if self._is_minimap_occluded(bgr, minimap_bgr):
                logger.debug("Minimap kapalı/örtülü, mevcut odometri konumu korunuyor.")
                self._append_trajectory()
                return self.get_position()

            self._last_valid_minimap_bgr = minimap_bgr.copy()
            curr_clean = self._clean_minimap_for_odometry(minimap_bgr)

            now_t = time.time()
            dt = max(0.01, now_t - float(self._last_odometry_update_ts or 0.0)) if self._last_odometry_update_ts > 0 else 0.1
            self._last_odometry_update_ts = now_t

            if self._prev_minimap_clean is not None and self._prev_minimap_clean.shape == curr_clean.shape:
                dx_terrain, dy_terrain, conf = self._estimate_minimap_shift(
                    self._prev_minimap_clean,
                    curr_clean,
                )
                # H-sweep (dilim 13 O2): son kayma güveni izlenir;
                # get_position_status() düşük güvende kör
                # dead-reckoning'i 'lost' bayrağıyla dışarı çıkarır.
                self._last_odometry_confidence = float(conf)
                if conf >= 0.15:
                    player_dx = -dx_terrain * self.minimap_to_map_scale
                    player_dy = -dy_terrain * self.minimap_to_map_scale

                    self.map_x = float(np.clip(self.map_x + player_dx, 0.0, float(self.map_width - 1)))
                    self.map_y = float(np.clip(self.map_y + player_dy, 0.0, float(self.map_height - 1)))
                    self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
                    self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
                    # "Drift" hareketin %2'si sayılmaz (F-0410): gerçek konum
                    # hatası ölçülmeden birikim yapılmaz; birikim yalnızca
                    # kalibrasyonda sıfırlanır.

                    if dt > 0:
                        self.velocity_x = player_dx / dt
                        self.velocity_y = player_dy / dt
                    self.dead_reckoning_dx = 0.0
                    self.dead_reckoning_dy = 0.0
                else:
                    # Düşük güvende dead-reckoning ile konum devamlılığı sağla
                    if (abs(self.velocity_x) > 0.05 or abs(self.velocity_y) > 0.05) and dt < 2.5:
                        dr_x = self.velocity_x * dt
                        dr_y = self.velocity_y * dt
                        self.dead_reckoning_dx += dr_x
                        self.dead_reckoning_dy += dr_y
                        self.map_x = float(np.clip(self.map_x + dr_x, 0.0, float(self.map_width - 1)))
                        self.map_y = float(np.clip(self.map_y + dr_y, 0.0, float(self.map_height - 1)))
                        self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
                        self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
                        self.velocity_x *= 0.95
                        self.velocity_y *= 0.95

            # Odometri takipçisi konum ve durumunu senkronize tut
            self.odometry_tracker.map_x = float(self.map_x)
            self.odometry_tracker.map_y = float(self.map_y)
            self.odometry_tracker.norm_x = float(self.norm_x)
            self.odometry_tracker.norm_y = float(self.norm_y)
            self.odometry_tracker.velocity_x = float(self.velocity_x)
            self.odometry_tracker.velocity_y = float(self.velocity_y)
            self.odometry_tracker.dead_reckoning_dx = float(self.dead_reckoning_dx)
            self.odometry_tracker.dead_reckoning_dy = float(self.dead_reckoning_dy)

            # Minimap üzerinden yön açısını güncelle (tek örnek yeniden kullanılır, F-0411)
            try:
                from core.minimap_scanner import MinimapScanner

                if self._heading_scanner is None:
                    self._heading_scanner = MinimapScanner()
                ok_h, h_val = self._heading_scanner.detect_player_heading(bgr)
                if ok_h:
                    self.player_heading_deg = float(h_val)
            except Exception as exc:
                logger.debug("Minimap yön açısı okunamadı: %s", exc)

            self._prev_minimap_clean = curr_clean
            self._append_trajectory()
            self._refresh_calibration_flag()
            return self.get_position()
        except Exception as exc:
            logger.error("update_player_position hatası: %s", exc)
            return self.get_position()

    def get_position(self) -> Tuple[float, float, float, float]:
        """Güncel `(map_x, map_y, norm_x, norm_y)` tuple'ını döndürür."""
        self.norm_x = round(float(self.map_x) / float(max(1, self.map_width)), 4)
        self.norm_y = round(float(self.map_y) / float(max(1, self.map_height)), 4)
        self.odometry_tracker.map_x = float(self.map_x)
        self.odometry_tracker.map_y = float(self.map_y)
        self.odometry_tracker.norm_x = float(self.norm_x)
        self.odometry_tracker.norm_y = float(self.norm_y)
        return (
            round(float(self.map_x), 3),
            round(float(self.map_y), 3),
            float(self.norm_x),
            float(self.norm_y),
        )

    def get_position_status(self) -> Dict[str, Any]:
        """Konum + odometri güven durumunu döndürür.

        H-sweep (dilim 13 O2): düşük güvenli odometride kör
        dead-reckoning dışarıdan görünür olsun diye. get_position()
        tuple dönüşü korunur; bu metod yalnızca durum sözlüğü verir.
        'lost': son kayma güveni 0.15'in altındaysa (odometri kapısı)
        VE dead-reckoning birikimi sürüyorsa konum güvenilmez sayılır.
        """
        conf = self._last_odometry_confidence
        dead_reckoning = bool(
            abs(float(self.dead_reckoning_dx)) > 0.0 or abs(float(self.dead_reckoning_dy)) > 0.0
        )
        lost = bool(conf is not None and conf < 0.15 and dead_reckoning)
        return {
            "position": self.get_position(),
            "odometry_confidence": conf,
            "dead_reckoning": dead_reckoning,
            "dead_reckoning_dx": float(self.dead_reckoning_dx),
            "dead_reckoning_dy": float(self.dead_reckoning_dy),
            "calibrated": bool(self._is_calibrated_on_world_map),
            "needs_map_calibrate": bool(self.needs_map_calibrate),
            "lost": lost,
        }

    # ------------------------------------------------------------------
    # Canlı bölge tespiti (Görev 4 / harita-istatistik)
    #
    # İki kaynak vardır; ikisi de "bilinmiyor"u varsayılan bölgeye çevirmez:
    # 1. Minimap: MinimapScanner.identify_subregion — SADECE "spatial_bbox"
    #    kaynağı RouteManager'a iletilir. Başlık şablon listesi yalnızca
    #    Altgard alt bölgelerini içerir (minimap_scanner.py kapsam dışı),
    #    ayrıca Verteron alt bölgelerinin "bbox"u None'dur; yani başlık
    #    eşleşmesi Verteron'u asla doğrulayamaz, yalnızca Altgard'ı
    #    doğrulayabilir. Bu yüzden başlık sonucu yalnız görüntü amaçlıdır.
    # 2. Açık dünya haritası (M): ORB ile assets/maps/*.png karşılaştırması.
    #    Verteron'u canlı tespit edebilen tek yol budur; canlı kareyle
    #    doğrulanmadı, eşik bilerek katı tutuldu (bkz. identify_region_from_world_map).
    # ------------------------------------------------------------------
    def get_detected_region(self) -> Optional[Dict[str, Any]]:
        """Son güvenilir canlı bölge tespitinin kopyası; yoksa None."""
        if not isinstance(self._detected_region, dict):
            return None
        return dict(self._detected_region)

    def _forward_detected_region(self, detection: Dict[str, Any]) -> None:
        """Tespiti RouteManager'a iletir (uyumsuzluk korumasını besler)."""
        try:
            rm = getattr(self, "route_manager", None)
            note = getattr(rm, "note_detected_region", None) if rm is not None else None
            if callable(note):
                note(
                    detection.get("canonical_region") or detection.get("region"),
                    confidence=float(detection.get("confidence", 0.0) or 0.0),
                    source=str(detection.get("source", "") or ""),
                )
        except Exception as exc:
            logger.debug("Bölge tespiti RouteManager'a iletilemedi: %s", exc)

    def detect_region_from_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        player_map_pos: Optional[Tuple[float, float]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Minimap karesinden canlı bölge tespiti yapar.

        Yalnızca mekansal sınırlayıcı-kutu ("spatial_bbox") tespiti güvenilir
        sayılır ve saklanır/iletilir; diğer kaynaklar None döndürür.
        Dönüş: {"subregion", "canonical_region", "confidence", "source"} ya da None.
        """
        try:
            bgr = self._ensure_bgr(frame_bgr)
            if bgr is None:
                return None
            minimap_bgr = self.extract_minimap_roi(bgr)
            if minimap_bgr is None or minimap_bgr.size == 0:
                return None
            if self._region_scanner is None:
                from core.minimap_scanner import MinimapScanner

                self._region_scanner = MinimapScanner()
            pos = player_map_pos
            if pos is None:
                try:
                    cur = self.get_position()
                    pos = (float(cur[0]), float(cur[1]))
                except Exception:
                    pos = None
            info = self._region_scanner.identify_subregion(minimap_bgr, pos)
            if not isinstance(info, dict):
                return None
            # Yalnızca geometri tabanlı tespit RouteManager korumasını besler.
            if str(info.get("source", "")) != "spatial_bbox":
                return None
            conf = float(info.get("confidence", 0.0) or 0.0)
            if conf < REGION_DETECT_MIN_CONFIDENCE:
                return None
            canon = resolve_canonical_region(info.get("canonical_region"))
            if canon not in KNOWN_CANONICAL_REGIONS:
                return None
            detection = {
                "subregion": str(info.get("subregion", "") or ""),
                "canonical_region": canon,
                "confidence": conf,
                "source": "spatial_bbox",
            }
            self._detected_region = dict(detection)
            self._forward_detected_region(detection)
            return dict(detection)
        except Exception as exc:
            logger.debug("Bölge tespiti hatası: %s", exc)
            return None

    def _maybe_detect_region(self, frame_bgr: Optional[np.ndarray]) -> None:
        """Kısılmış (5 sn) minimap bölge tespiti; update_player_position çağırır."""
        now = time.time()
        if now - float(self._last_region_detect_ts or 0.0) < REGION_DETECT_INTERVAL_SEC:
            return
        self._last_region_detect_ts = now
        try:
            self.detect_region_from_frame(frame_bgr)
        except Exception as exc:
            logger.debug("Kısılmış bölge tespiti atlandı: %s", exc)

    def identify_region_from_world_map(
        self,
        frame_bgr: Optional[np.ndarray],
    ) -> Optional[Dict[str, Any]]:
        """Açık dünya haritası karesini assets/maps/*.png ile ORB eşleştirir.

        Ekrandaki harita hangi bölgeninse onu döndürür; karar için en iyi
        bölgenin iyi eşleşme sayısı >= 15 VE ikinciye oranı >= 2 olmalıdır.
        Karar verilemezse None döner (bilinmiyor — varsayılan bölge yok).
        NOT: canlı oyun karesiyle doğrulanmadı; sentetik kırpma testleriyle
        yalnızca eşleştirici mekaniği doğrulandı.
        """
        try:
            bgr = self._ensure_bgr(frame_bgr)
            if bgr is None or bgr.size == 0:
                return None
            if self._world_map_orb_cache is None:
                self._world_map_orb_cache = self._build_world_map_orb_cache()
            cache = self._world_map_orb_cache or {}
            if not cache:
                return None
            orb = cv2.ORB_create(nfeatures=1000)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
            if gray.shape[1] > 1280:
                scale = 1280.0 / float(gray.shape[1])
                gray = cv2.resize(
                    gray,
                    (1280, max(1, int(round(gray.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            _, frame_des = orb.detectAndCompute(gray, None)
            if frame_des is None or len(frame_des) < 8:
                return None
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            scores: Dict[str, int] = {}
            for region, entry in cache.items():
                ref_des = entry.get("descriptors")
                if ref_des is None or len(ref_des) < 8:
                    continue
                try:
                    knn = bf.knnMatch(frame_des, ref_des, k=2)
                except Exception:
                    continue
                good = 0
                for pair in knn:
                    if len(pair) != 2:
                        continue
                    m, n = pair
                    if m.distance < 0.75 * n.distance:
                        good += 1
                scores[str(region)] = int(good)
            if not scores:
                return None
            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_region, best_good = ranked[0]
            second_good = ranked[1][1] if len(ranked) > 1 else 0
            if best_good < REGION_WORLD_MAP_MIN_GOOD_MATCHES:
                return None
            if second_good > 0 and best_good < REGION_WORLD_MAP_MIN_MARGIN_RATIO * second_good:
                return None
            # H-sweep (dilim 13 Y1): güven sabit 0.85 değil, eşleşmeden
            # türetilir — taban 0.55 + iyi eşleşme sayısı (40'ta doyar) +
            # ikinciye marj (4x'te doyar); üst sınır 0.95. Kapılar
            # (>=15 eşleşme, >=2x marj) zaten geçildiği için en düşük
            # güven ~0.72 çıkar. Formül canlı kareyle doğrulanmadı;
            # yalnızca göreli ayrımı dürüst kılar.
            match_term = min(int(best_good), 40) / 40.0
            margin_term = min(float(best_good) / max(float(second_good), 1.0), 4.0) / 4.0
            orb_confidence = round(min(0.95, 0.55 + 0.25 * match_term + 0.15 * margin_term), 3)
            detection = {
                "subregion": best_region,
                "canonical_region": best_region,
                "confidence": orb_confidence,
                "source": "world_map_orb",
                "good_matches": int(best_good),
                "second_matches": int(second_good),
            }
            self._detected_region = dict(detection)
            self._forward_detected_region(detection)
            return dict(detection)
        except Exception as exc:
            logger.debug("Dünya haritası bölge eşleşmesi hatası: %s", exc)
            return None

    def _build_world_map_orb_cache(self) -> Dict[str, Any]:
        """assets/maps/<bölge>.png dosyalarının ORB tanımlayıcılarını çıkarır (bir kez)."""
        cache: Dict[str, Any] = {}
        try:
            orb = cv2.ORB_create(nfeatures=1000)
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            maps_dir = os.path.join(repo_root, "assets", "maps")
            for region in sorted(KNOWN_CANONICAL_REGIONS):
                path = os.path.join(maps_dir, f"{region.lower()}.png")
                if not os.path.isfile(path):
                    continue
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None or img.size == 0:
                    continue
                if img.shape[1] > 1280:
                    scale = 1280.0 / float(img.shape[1])
                    img = cv2.resize(
                        img,
                        (1280, max(1, int(round(img.shape[0] * scale)))),
                        interpolation=cv2.INTER_AREA,
                    )
                _, des = orb.detectAndCompute(img, None)
                if des is not None and len(des) >= 8:
                    cache[region] = {"path": path, "descriptors": des}
        except Exception as exc:
            logger.debug("Dünya haritası ORB önbelleği kurulamadı: %s", exc)
        return cache

    # ------------------------------------------------------------------
    # 2. Doğrudan Minimap Üzerinden 80x80 Temiz Harita Kesiti (Patch) Kırpma
    # ------------------------------------------------------------------
    def extract_clean_minimap_patch(
        self,
        frame_bgr: np.ndarray,
        rel_map_dx: float = 0.0,
        rel_map_dy: float = 0.0,
    ) -> np.ndarray:
        """
        Haritayı (`M`) açıp oyuncuyu durdurmadan, doğrudan sağ üst minimap üzerindeki
        ilgili `(rel_map_dx, rel_map_dy)` konumundan 80x80 gri harita kesitini (`patch_img`)
        oyuncu okunu ve çerçeveyi temizleyerek otomatik kırpar.
        """
        try:
            bgr = self._ensure_bgr(frame_bgr)
            # Dünya haritası dalı yalnızca KESİN açıkken (is True):
            # _extract_patch_from_world_map doğrulama yapmaz, oyun karesini
            # de kör kırpar — bilinmiyor durumunda sahte harita kesiti
            # üretilmesin diye None minimap yoluna düşer.
            if bgr is not None and bgr.shape[1] >= 800 and self._is_world_map_open(bgr) is True:
                target_x = int(round(self.map_x + rel_map_dx))
                target_y = int(round(self.map_y + rel_map_dy))
                return self._extract_patch_from_world_map(bgr, target_x, target_y)

            minimap_bgr: Optional[np.ndarray] = None
            if bgr is not None:
                candidate = self.extract_minimap_roi(bgr)
                if candidate is not None and not self._is_minimap_occluded(bgr, candidate):
                    minimap_bgr = candidate
                    self._last_valid_minimap_bgr = candidate.copy()

            if minimap_bgr is None and self._last_valid_minimap_bgr is not None:
                minimap_bgr = self._last_valid_minimap_bgr.copy()

            if minimap_bgr is None or minimap_bgr.size == 0:
                return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

            mh, mw = minimap_bgr.shape[:2]
            gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

            # 1. Minimap merkezindeki oyuncu okunu ve dış çerçeve/UI halkasını maskele
            clean_mask = np.zeros((mh, mw), dtype=np.uint8)
            arrow_cx, arrow_cy = self._locate_minimap_player_arrow(minimap_bgr)
            mcx, mcy = int(round(arrow_cx)), int(round(arrow_cy))
            cv2.circle(clean_mask, (mcx, mcy), max(8, int(round(min(mw, mh) * 0.085))), 255, -1)

            arrow_zone = np.zeros((mh, mw), dtype=np.uint8)
            cv2.circle(arrow_zone, (mcx, mcy), max(13, int(round(min(mw, mh) * 0.14))), 255, -1)
            arrow_px = (
                ((hsv[:, :, 1] > 110) & (hsv[:, :, 2] > 140)) | (hsv[:, :, 2] > 232)
            ).astype(np.uint8) * 255
            clean_mask = cv2.bitwise_or(clean_mask, cv2.bitwise_and(arrow_px, arrow_zone))

            border_px = 4
            clean_mask[:border_px, :] = 255
            clean_mask[mh - border_px :, :] = 255
            clean_mask[:, :border_px] = 255
            clean_mask[:, mw - border_px :] = 255

            inpainted_minimap = cv2.inpaint(gray, clean_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # 2. Kaynağın minimap üzerindeki piksel merkezini hesapla
            inv_scale = 1.0 / max(0.1, float(self.minimap_to_map_scale))
            target_mx = int(round(arrow_cx + float(rel_map_dx) * inv_scale))
            target_my = int(round(arrow_cy + float(rel_map_dy) * inv_scale))

            half = PATCH_SIZE // 2
            pad = PATCH_SIZE
            padded = cv2.copyMakeBorder(
                inpainted_minimap,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_REPLICATE,
            )

            px = int(np.clip(target_mx, 0, mw)) + pad
            py = int(np.clip(target_my, 0, mh)) + pad

            patch_80 = padded[py - half : py + half, px - half : px + half].copy()
            if patch_80.shape[0] != PATCH_SIZE or patch_80.shape[1] != PATCH_SIZE:
                patch_80 = cv2.resize(patch_80, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

            return np.asarray(patch_80, dtype=np.uint8)
        except Exception as exc:
            logger.error("extract_clean_minimap_patch hatası: %s", exc)
            return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    @staticmethod
    def _extract_patch_from_world_map(
        world_map_bgr: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """Açık dünya haritası üzerinden oyuncu okunu temizleyerek 80x80 gri kesit kırpar."""
        try:
            gray = cv2.cvtColor(world_map_bgr, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            half = PATCH_SIZE // 2
            cx = int(np.clip(center_x, half, max(half, w - half)))
            cy = int(np.clip(center_y, half, max(half, h - half)))
            patch_gray = gray[cy - half : cy + half, cx - half : cx + half].copy()
            if patch_gray.shape != (PATCH_SIZE, PATCH_SIZE):
                patch_gray = cv2.resize(patch_gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

            mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
            cv2.circle(mask, (half, half), 9, 255, -1)
            cleaned = cv2.inpaint(patch_gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            return np.asarray(cleaned, dtype=np.uint8)
        except Exception as exc:
            logger.error("_extract_patch_from_world_map hatası: %s", exc)
            return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

    @staticmethod
    def _save_direct_region_node_patch(
        base_dir: str,
        region: str,
        node_id: int,
        patch_img: np.ndarray,
    ) -> str:
        """
        80x80 görsel şablonunu hem `routes/<region>/patches/node_<id>.png` hem de
        doğrudan `routes/<region>/node_<02d>.png` / `routes/<region>/node_<id>.png`
        olarak kaydeder.
        """
        try:
            reg_dir = os.path.join(base_dir, str(region))
            patches_dir = os.path.join(reg_dir, "patches")
            os.makedirs(patches_dir, exist_ok=True)

            direct_formatted = os.path.join(reg_dir, f"node_{int(node_id):02d}.png")
            direct_simple = os.path.join(reg_dir, f"node_{int(node_id)}.png")
            patch_sub = os.path.join(patches_dir, f"node_{int(node_id)}.png")

            ok = True
            for p in (direct_formatted, direct_simple, patch_sub):
                if not cv2.imwrite(p, patch_img):
                    ok = False
            if not ok:
                # Yazım başarısızsa başarı yolu döndürülmez (F-0211).
                logger.debug("Doğrudan node patch kaydı başarısız: %s", region)
                return ""

            base_name = os.path.basename(os.path.normpath(base_dir)) or "routes"
            return f"{base_name}/{region}/node_{int(node_id):02d}.png".replace("\\", "/")
        except Exception as exc:
            logger.debug("Doğrudan node patch kaydı hatası: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # 3. 3 Eşzamanlı Otomatik Tarayıcı (Minimap 3-Leaf Sprout + 3D Crest + F Gather)
    # ------------------------------------------------------------------
    def _detect_minimap_gatherables(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Sağ üstteki minimap (`Pilgrim's Ridge`, `x: 0.855..0.992, y: 0.065..0.255`)
        üzerindeki toplanabilir kaynak ikonlarını tespit eder:
        - SADECE **3 Yapraklı Beyaz Filiz İkonlarını** (`Gatherable Material` — Minimap
          Filter Settings'deki bitki/filiz sembolü) kaydeder.
        - **Beyaz tüy (feather) ve mavi kanat ikonlarını KESİNLİKLE kaydetmez** (tüy Odyle
          veya toplanabilir materyal değildir!).
        - Canavar noktalarını (`• Monster`: küçük dairesel 5x5..7x7 beyaz noktalar),
          sarı görüş konisi halkasını, saat/kanal rozetlerini ve büyüteç ikonunu filtreler.
        - Filiz ikonunun yanında yeşil yığın rozeti (`2` veya `3`) varsa `stack_count=2|3`
          (`max_harvests = 3 * stack_count`) olarak kaydeder.
        """
        detections: List[Dict[str, Any]] = []
        try:
            minimap_bgr = self.extract_minimap_roi(frame_bgr)
            if minimap_bgr is None or minimap_bgr.size == 0:
                return detections
            if self._is_minimap_occluded(frame_bgr, minimap_bgr):
                return detections

            mh, mw = minimap_bgr.shape[:2]
            if mh < 32 or mw < 32:
                return detections

            hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)
            arrow_cx, arrow_cy = self._locate_minimap_player_arrow(minimap_bgr)
            frame_h, frame_w = frame_bgr.shape[:2]
            x1_mm, y1_mm, _, _ = self._get_minimap_pixel_rect(frame_w, frame_h)

            # 3 Yapraklı Beyaz Filiz (Gatherable Material Sprout) İkonu Tespiti:
            # - Canavar noktalarını (5x5..7x7 dairesel yüksek doluluklu noktalar) eler.
            # - Tüy (feather) ve mavi kanat ikonlarını (uzun diyagonal şekil veya mavi kanat komşuluğu) eler.
            white_mask = cv2.inRange(hsv, (0, 0, 192), (180, 46, 255))
            white_mask[: int(mh * 0.18), :] = 0
            white_mask[int(mh * 0.80) :, int(mw * 0.56) :] = 0
            white_mask[: int(mh * 0.34), int(mw * 0.78) :] = 0
            white_mask[:, : int(mw * 0.05)] = 0
            white_mask[:, int(mw * 0.95) :] = 0
            # Merkezdeki oyuncu okunu maskele
            cv2.circle(
                white_mask,
                (int(round(arrow_cx)), int(round(arrow_cy))),
                max(11, int(round(min(mw, mh) * 0.095))),
                0,
                -1,
            )

            num_w, _, stats_w, cents_w = cv2.connectedComponentsWithStats(white_mask)
            for idx in range(1, num_w):
                bx, by, bw, bh, area = stats_w[idx]
                # 3 yapraklı filiz ikonu kompakt ve simetriktir (8..16 px genişlik/yükseklik);
                # uzun tüy ikonları (bh >= 18 veya bw >= 18) burada otomatik reddedilir
                if bw < 8 or bh < 8 or bw > 16 or bh > 16 or area < 24 or area > 88:
                    continue
                if abs(int(bw) - int(bh)) > 4:
                    continue

                fill_ratio = float(area) / float(max(1, bw * bh))
                # Canavar noktaları küçük tam dairelerdir (fill_ratio >= 0.64);
                # 3 yapraklı filiz ikonlarının yaprak araları boşluklu olduğu için fill_ratio 0.30..0.62 aralığındadır
                if fill_ratio < 0.30 or fill_ratio > 0.62:
                    continue

                px1, py1 = max(0, bx - 2), max(0, by - 2)
                px2, py2 = min(mw, bx + bw + 2), min(mh, by + bh + 2)
                patch_v = hsv[py1:py2, px1:px2, 2]
                min_v = float(np.min(patch_v))
                contrast = float(np.max(patch_v)) - min_v
                # Filiz ikonunun koyu dış konturu vardır (min_v <= 85, contrast >= 115)
                if min_v > 85.0 or contrast < 115.0:
                    continue

                # Tüy / mavi kanat (feather / blue wing) filtresi:
                # Eğer beyaz bileşenin hemen etrafında mavi kanat pikselleri (H: 90..118, S >= 70, V >= 115)
                # yoğun ise bu bir tüy/kanat ikonudur, kesinlikle kaydetme!
                wx1, wy1 = max(0, bx - 6), max(0, by - 8)
                wx2, wy2 = min(mw, bx + bw + 6), min(mh, by + bh + 6)
                wing_blue_px = int(
                    np.count_nonzero(cv2.inRange(hsv[wy1:wy2, wx1:wx2], (92, 70, 115), (118, 255, 255)))
                )
                if wing_blue_px >= 8:
                    continue

                icx = float(cents_w[idx][0])
                icy = float(cents_w[idx][1])

                # Feather + monster circularity safeguards (required on save path)
                try:
                    icon_patch_bgr = minimap_bgr[
                        max(0, by - 2) : min(mh, by + bh + 2),
                        max(0, bx - 2) : min(mw, bx + bw + 2),
                    ]
                    if icon_patch_bgr.size > 0 and hasattr(self, "feather_negative_filter"):
                        if self.feather_negative_filter.is_feather_icon(icon_patch_bgr):
                            continue
                    if icon_patch_bgr.size > 0 and hasattr(self, "monster_circularity_filter"):
                        if self.monster_circularity_filter.is_monster_dot(icon_patch_bgr):
                            continue
                except Exception as exc_vf:
                    logger.debug("vision_filters minimap safeguard skipped: %s", exc_vf)

                # Yanında yeşil yığın rozeti ('2' veya '3') var mı kontrol et (H: 68..90, S >= 75, V >= 115).
                # Çevredeki dağınık yeşil arazi pikselleri rozet sayılmaz (F-0055):
                # yalnızca kompakt, rozet boyutunda yeşil bileşenler sayılır.
                bx1, by1 = max(0, bx - 4), max(0, by - 4)
                bx2, by2 = min(mw, bx + bw + 8), min(mh, by + bh + 6)
                badge_mask = cv2.inRange(hsv[by1:by2, bx1:bx2], (68, 75, 115), (90, 255, 255))
                badge_px = 0
                num_b, _, stats_b, _ = cv2.connectedComponentsWithStats(badge_mask)
                for b_idx in range(1, num_b):
                    _bx, _by, _bw, _bh, _area = stats_b[b_idx]
                    if _area < 8 or _bw < 3 or _bh < 3 or _bw > 24 or _bh > 24:
                        continue
                    if float(_area) / float(max(1, _bw * _bh)) < 0.30:
                        continue
                    badge_px += int(_area)
                stack_count = 3 if badge_px >= 28 else (2 if badge_px >= 10 else 1)

                rel_dx = (icx - arrow_cx) * self.minimap_to_map_scale
                rel_dy = (icy - arrow_cy) * self.minimap_to_map_scale

                detections.append(
                    {
                        "rel_map_dx": round(float(rel_dx), 3),
                        "rel_map_dy": round(float(rel_dy), 3),
                        "minimap_x": round(icx, 2),
                        "minimap_y": round(icy, 2),
                        "screen_x": int(round(x1_mm + icx)),
                        "screen_y": int(round(y1_mm + icy)),
                        "resource_type": "Gatherable Material",
                        "in_air": False,
                        "confidence": round(
                            max(
                                MINIMAP_SPROUT_CONF_MIN,
                                min(0.95, 0.76 + (contrast - 115.0) / 500.0),
                            ),
                            3,
                        ),
                        "stack_count": int(stack_count),
                        "max_harvests": int(DEFAULT_MAX_HARVESTS * stack_count),
                        "current_harvests": 0,
                        "source_type": "minimap_auto",
                    }
                )

            return detections
        except Exception as exc:
            logger.error("_detect_minimap_gatherables hatası: %s", exc)
            return detections

    def _project_3d_screen_to_map_offset(
        self,
        screen_x: float,
        screen_y: float,
        frame_w: int,
        frame_h: int,
        heading_deg: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        3D oyun ekranındaki `(screen_x, screen_y)` noktasını oyuncunun ayak merkezine
        `(frame_w * 0.50, frame_h * 0.54)` göre perspektif düzeltmeli ve oyuncu yön açısına
        (`heading_deg`) göre döndürülmüş harita ofsetine `(rel_map_dx, rel_map_dy)` dönüştürür.
        Normalize edilmiş koordinatlar (0..1) otomatik olarak piksel koordinatlarına çevrilir.
        """
        fw = max(1, int(frame_w))
        fh = max(1, int(frame_h))
        sx = float(screen_x)
        sy = float(screen_y)

        # Normalize koordinatları piksel koordinatlarına dönüştür (merkeze yapışmayı önler).
        # 0..1.5 aralığı piksel (1, 1) ile normalize (1.0, 1.0) ayrımını yapamaz (F-0212).
        if _is_normalized_coord_pair(sx, sy) and fw > 10 and fh > 10:
            sx = sx * float(fw)
            sy = sy * float(fh)

        player_x = float(fw) * self.PLAYER_FEET_X_RATIO
        player_y = float(fh) * self.PLAYER_FEET_Y_RATIO

        raw_dx = sx - player_x
        raw_dy = sy - player_y

        norm_dx = raw_dx * (float(self.REF_SCREEN_WIDTH) / float(fw))
        norm_dy = raw_dy * (float(self.REF_SCREEN_HEIGHT) / float(fh))

        vertical_ratio = max(0.35, min(1.45, (player_y / float(max(1.0, sy))) ** 1.15))
        base_dx = norm_dx * self.PERSPECTIVE_SCALE_X * vertical_ratio
        base_dy = norm_dy * self.PERSPECTIVE_SCALE_Y * vertical_ratio

        # Oyuncu yön açısına göre dünya haritası yönüne döndür
        # Kuzey = 0°, Doğu = 90°, Güney = 180°, Batı = 270°
        h_deg = heading_deg if heading_deg is not None else getattr(self, "player_heading_deg", 0.0)
        if abs(h_deg) > 0.01:
            rad = math.radians(float(h_deg))
            cos_a = math.cos(rad)
            sin_a = math.sin(rad)
            forward_map = -base_dy
            lateral_map = base_dx
            rot_dx = lateral_map * cos_a + forward_map * sin_a
            rot_dy = lateral_map * sin_a - forward_map * cos_a
            return float(round(rot_dx, 3)), float(round(rot_dy, 3))

        return float(round(base_dx, 3)), float(round(base_dy, 3))

    def _mesh_matches_monster_negative(
        self, mesh_bgr: np.ndarray
    ) -> Optional[Tuple[str, float]]:
        """3D mesh öğrenilmiş canavar/oyuncu negatif örneğine uyuyor mu?

        v20-targena-3dauto: renk sınıflandırması (sarı+yeşil Targena
        kuralı) mob gövdesini Targena sayabilir. v20 tarayıcı kapısı
        (material_scanner._classify_3d_material, sim >= 0.82) bu yolda
        hiç çalışmıyor — kartograf kendi 3D tespitinde mesh'i kendisi
        kontrol eder. Eşleşme varsa (örnek_adı, sim) döner, yoksa None.
        """
        try:
            mem = self._get_exemplar_memory()
        except Exception:
            return None
        if mem is None or not hasattr(mem, "is_negative_patch"):
            return None
        try:
            is_neg, neg_sim, neg_ex = mem.is_negative_patch(mesh_bgr, threshold=0.82)
        except Exception:
            return None
        if not is_neg or neg_ex is None:
            return None
        neg_name = str(neg_ex.get("class_name", "") or "")
        lowered = neg_name.lower()
        if "monster" not in lowered and "player" not in lowered:
            return None
        return neg_name, float(neg_sim)

    def _classify_3d_mesh(
        self, mesh_bgr: np.ndarray, label_width_px: int
    ) -> Optional[str]:
        """3D mesh'i renk kurallarıyla sınıflandır; canavar mesh None döner.

        Döner: materyal adı; sınıflandırılamazsa UNKNOWN_RESOURCE_TYPE;
        mesh canavar/oyuncu negatif örneğine uyuyorsa None (aday
        haritalanmaz — tek satır tanı logu burada basılır).
        """
        try:
            from config.materials_db import classify_3d_material_patch

            classified = classify_3d_material_patch(
                mesh_bgr,
                label_width_px=int(label_width_px),
            )
            name = str(classified.get("name") or "")
        except Exception as classify_exc:
            logger.debug("3B essence sınıfı okunamadı: %s", classify_exc)
            return UNKNOWN_RESOURCE_TYPE
        if not name or name == UNKNOWN_RESOURCE_TYPE:
            return UNKNOWN_RESOURCE_TYPE
        monster = self._mesh_matches_monster_negative(mesh_bgr)
        if monster is not None:
            neg_name, neg_sim = monster
            logger.info(
                "3d_auto %s adayı canavar filtresine takıldı (örnek=%s, sim=%.2f), haritalanmadı",
                name,
                neg_name,
                neg_sim,
            )
            return None
        return name

    def _detect_3d_world_gatherables(
        self,
        frame_bgr: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Minimap'te gösterilmeyen `Aria` (mor kristal bitki), `Mela` (kırmızı meyveli çalı),
        `Orichalcum Ore` ve yüzen `[E]` toplama ikonlarını 3D oyun dünyasında tespit eder
        (`source_type="3d_auto"`).
        """
        detections: List[Dict[str, Any]] = []
        try:
            h, w = frame_bgr.shape[:2]
            if w < 640 or h < 360:
                return detections

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            scale_x = float(w) / float(self.REF_SCREEN_WIDTH)
            scale_y = float(h) / float(self.REF_SCREEN_HEIGHT)

            # 1. Beyaz Kanatlı Toplama Arması + Cyan İsim Etiketi ('Aria', 'Mela', 'Orichalcum')
            cyan_mask = cv2.inRange(hsv, (74, 68, 118), (105, 255, 255))
            cyan_mask[: int(h * 0.16), :] = 0
            cyan_mask[int(h * 0.76) :, :] = 0
            cyan_mask[:, : int(w * 0.12)] = 0
            cyan_mask[:, int(w * 0.80) :] = 0
            # Oyuncu karakterinin kendi zırh/silah parlamasını maskele
            cyan_mask[int(h * 0.51) : int(h * 0.66), int(w * 0.475) : int(w * 0.525)] = 0

            kw = max(8, int(round(12 * scale_x)))
            kh = max(3, int(round(4 * scale_y)))
            closed = cv2.morphologyEx(
                cyan_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
            )
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in cnts:
                x, y, cw, ch = cv2.boundingRect(cnt)
                if cw < int(round(14 * scale_x)) or cw > int(round(145 * scale_x)):
                    continue
                if ch < int(round(5 * scale_y)) or ch > int(round(28 * scale_y)):
                    continue

                # Etiketin hemen üstündeki beyaz kanatlı toplama armasını (crest) doğrula
                cy1 = max(0, y - int(round(46 * scale_y)))
                cy2 = max(0, y - int(round(2 * scale_y)))
                cx_mid = x + cw // 2
                half_cw = max(14, int(round(24 * scale_x)))
                cx1 = max(0, cx_mid - half_cw)
                cx2 = min(w, cx_mid + half_cw)
                if cy2 <= cy1 or cx2 <= cx1:
                    continue

                crest_roi = hsv[cy1:cy2, cx1:cx2]
                crest_white = int(
                    np.count_nonzero(cv2.inRange(crest_roi, (0, 0, 172), (180, 52, 255)))
                )
                min_crest_px = max(16, int(round(28 * scale_x * scale_y)))
                
                # Optional pyramid_3d_crest_detector safeguard check if available
                pyramid_detected = False
                if hasattr(self, "pyramid_3d_crest_detector"):
                    crest_bgr_patch = frame_bgr[cy1:cy2, cx1:cx2]
                    if crest_bgr_patch.size > 0:
                        p_dets = self.pyramid_3d_crest_detector.detect(crest_bgr_patch, use_viewport_roi=False)
                        if p_dets:
                            pyramid_detected = True

                if crest_white < min_crest_px and not pyramid_detected:
                    continue

                # Etiketin altındaki 3D bitki/cevher dokusundan materyal türünü sınıflandır
                my1 = min(h - 1, y + ch + 4)
                my2 = min(h, y + ch + int(round(95 * scale_y)))
                mx1 = max(0, x - int(round(28 * scale_x)))
                mx2 = min(w, x + cw + int(round(28 * scale_x)))
                mesh_hsv = hsv[my1:my2, mx1:mx2]

                # H-sweep (dilim 13 K3): sınıflandırılamazsa tür jenerik
                # kalır; "Orichalcum Ore" varsayılmaz.
                resource_type = UNKNOWN_RESOURCE_TYPE
                mesh_bgr = frame_bgr[my1:my2, mx1:mx2]
                if mesh_bgr.size > 0:
                    # v20-targena-3dauto: canavar gövdesi renk kuralıyla
                    # Targena sayılmasın — None dönerse aday atılır.
                    classified_name = self._classify_3d_mesh(mesh_bgr, int(cw))
                    if classified_name is None:
                        continue
                    resource_type = str(classified_name or resource_type)

                node_sx = float(cx_mid)
                node_sy = float(min(h - 1, y + ch + int(round(26 * scale_y))))
                rel_dx, rel_dy = self._project_3d_screen_to_map_offset(node_sx, node_sy, w, h)

                # Güven ölçümden türetilir (F-0056): arma beyaz piksel gücü +
                # piramit doğrulaması; sabit 0.92 değil.
                crest_strength = min(
                    1.0, float(crest_white) / float(max(1, 2 * min_crest_px))
                )
                crest_confidence = round(
                    float(
                        np.clip(
                            0.55
                            + 0.35 * crest_strength
                            + (0.05 if pyramid_detected else 0.0),
                            0.55,
                            0.95,
                        )
                    ),
                    3,
                )
                detections.append(
                    {
                        "rel_map_dx": rel_dx,
                        "rel_map_dy": rel_dy,
                        "screen_x": int(round(node_sx)),
                        "screen_y": int(round(node_sy)),
                        "resource_type": resource_type,
                        # Etiket yüksekliği irtifa kanıtı değildir (F-0412).
                        "in_air": False,
                        "confidence": crest_confidence,
                        "stack_count": 1,
                        "max_harvests": DEFAULT_MAX_HARVESTS,
                        "current_harvests": 0,
                        "source_type": "3d_auto",
                    }
                )

            # 2. 3D Dünyada Yüzen Kare `[E]` Toplama İkonu (örn. pilgrims_ridge_1.jpg içindeki Odyle `[E]` kutusu)
            vx1, vy1 = int(w * 0.20), int(h * 0.22)
            vx2, vy2 = int(w * 0.78), int(h * 0.72)
            view_hsv = hsv[vy1:vy2, vx1:vx2]
            if view_hsv.size > 0:
                warm_core = cv2.inRange(view_hsv, (8, 95, 125), (28, 255, 255))
                cnts_e, _ = cv2.findContours(warm_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in cnts_e:
                    ex, ey, ew, eh = cv2.boundingRect(cnt)
                    if ew < 6 or ew > 40 or eh < 6 or eh > 40:
                        continue
                    bx1 = max(0, ex - int(round(16 * scale_x)))
                    by1 = max(0, ey - int(round(16 * scale_y)))
                    bx2 = min(view_hsv.shape[1], ex + ew + int(round(16 * scale_x)))
                    by2 = min(view_hsv.shape[0], ey + eh + int(round(16 * scale_y)))
                    box_hsv = view_hsv[by1:by2, bx1:bx2]
                    white_e = int(
                        np.count_nonzero(cv2.inRange(box_hsv, (0, 0, 190), (180, 55, 255)))
                    )
                    green_frame = int(
                        np.count_nonzero(cv2.inRange(box_hsv, (45, 65, 80), (88, 255, 255)))
                    )
                    warm_px = int(
                        np.count_nonzero(cv2.inRange(box_hsv, (8, 95, 125), (28, 255, 255)))
                    )
                    if white_e >= 15 and green_frame >= 180 and warm_px >= 60:
                        abs_sx = float(vx1 + bx1 + (bx2 - bx1) / 2.0)
                        abs_sy = float(vy1 + by1 + (by2 - by1) / 2.0)
                        rel_dx, rel_dy = self._project_3d_screen_to_map_offset(abs_sx, abs_sy, w, h)
                        detections.append(
                            {
                                "rel_map_dx": rel_dx,
                                "rel_map_dy": rel_dy,
                                "screen_x": int(round(abs_sx)),
                                "screen_y": int(round(abs_sy)),
                                "resource_type": _essence_name_near_box(
                                    frame_bgr, abs_sx, abs_sy
                                ),
                                "in_air": False,
                                # Güven ölçümden türetilir (F-0056): beyaz çekirdek /
                                # yeşil çerçeve / sıcak piksel güçleri.
                                "confidence": round(
                                    float(
                                        np.clip(
                                            0.55
                                            + 0.35
                                            * (
                                                min(1.0, white_e / 45.0)
                                                + min(1.0, green_frame / 400.0)
                                                + min(1.0, warm_px / 150.0)
                                            )
                                            / 3.0,
                                            0.55,
                                            0.95,
                                        )
                                    ),
                                    3,
                                ),
                                "stack_count": 1,
                                "max_harvests": DEFAULT_MAX_HARVESTS,
                                "current_harvests": 0,
                                "source_type": "3d_auto",
                            }
                        )

            return detections
        except Exception as exc:
            logger.error("_detect_3d_world_gatherables hatası: %s", exc)
            return detections

    def _detect_active_harvest_prompt(
        self,
        frame_bgr: np.ndarray,
        world_3d_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Oyuncu bir kaynağın (`Aria`, `Mela`, `Orichalcum Ore`) yanına geldiğinde beliren
        `[F Gather]` etkileşim istemini tespit eder (`source_type="harvest_auto"`).
        """
        prompts: List[Dict[str, Any]] = []
        try:
            h, w = frame_bgr.shape[:2]
            if w < 640 or h < 360:
                return prompts

            hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
            scale_x = float(w) / float(self.REF_SCREEN_WIDTH)
            scale_y = float(h) / float(self.REF_SCREEN_HEIGHT)

            rx1, ry1 = int(w * 0.48), int(h * 0.52)
            rx2, ry2 = int(w * 0.66), int(h * 0.65)
            roi_hsv = hsv[ry1:ry2, rx1:rx2]
            if roi_hsv.size == 0:
                return prompts

            white_mask = cv2.inRange(roi_hsv, (0, 0, 145), (180, 48, 255))
            kw = max(12, int(round(20 * scale_x)))
            kh = max(4, int(round(6 * scale_y)))
            closed = cv2.morphologyEx(
                white_mask,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh)),
            )
            cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_w = max(44, int(round(56 * scale_x)))
            max_w = max(85, int(round(145 * scale_x)))
            min_h = max(8, int(round(11 * scale_y)))
            max_h = max(24, int(round(36 * scale_y)))
            icon_split = max(16, int(round(24 * scale_x)))

            for cnt in cnts:
                fx, fy, fcw, fch = cv2.boundingRect(cnt)
                if fcw < min_w or fcw > max_w or fch < min_h or fch > max_h:
                    continue

                sub_mask = white_mask[fy : fy + fch, fx : fx + fcw]
                left_white = int(np.count_nonzero(sub_mask[:, :icon_split]))
                right_white = int(np.count_nonzero(sub_mask[:, icon_split:]))
                if left_white < int(round(25 * scale_x * scale_y)):
                    continue
                if right_white < int(round(25 * scale_x * scale_y)):
                    continue

                # 'Gather' yazısının arkasındaki koyu yarı-saydam kapsül arka planı doğrula
                pill_v = roi_hsv[fy : fy + fch, fx + icon_split : fx + fcw, 2]
                dark_pixels = pill_v[pill_v < 130]
                if dark_pixels.size == 0:
                    continue
                dark_mean = float(np.mean(dark_pixels))
                if not (18.0 <= dark_mean <= 82.0):
                    continue

                # Yakındaki 3D tespitten materyal adını ('Aria', 'Mela', vb.) miras al.
                # En yakın, oyuncuya en yakın olan değil, İSTEME ekranda en yakın
                # olandır (F-0057): oyuncuya yakınlık yanlış düğüm kaydeder.
                # H-sweep (dilim 13 K3): yakın 3D tespit yoksa / adı yoksa
                # tür jenerik kalır, "Orichalcum Ore" varsayılmaz.
                resource_type = UNKNOWN_RESOURCE_TYPE
                rel_dx, rel_dy = 0.0, 0.0
                prompt_cx = float(rx1 + fx + fcw // 2)
                prompt_cy = float(ry1 + fy + fch // 2)
                if world_3d_detections:
                    def _prompt_screen_dist(d: Dict[str, Any]) -> float:
                        dsx = d.get("screen_x")
                        dsy = d.get("screen_y")
                        if dsx is None or dsy is None:
                            return math.hypot(
                                float(d.get("rel_map_dx", 0.0)),
                                float(d.get("rel_map_dy", 0.0)),
                            )
                        return math.hypot(float(dsx) - prompt_cx, float(dsy) - prompt_cy)

                    nearest_3d = min(world_3d_detections, key=_prompt_screen_dist)
                    resource_type = str(
                        nearest_3d.get("resource_type") or UNKNOWN_RESOURCE_TYPE
                    )
                    rel_dx = float(nearest_3d.get("rel_map_dx", 0.0))
                    rel_dy = float(nearest_3d.get("rel_map_dy", 0.0))

                # Güven ölçümden türetilir (F-0056): sol/sağ beyaz parça güçleri.
                prompt_confidence = round(
                    float(
                        np.clip(
                            0.60
                            + 0.35
                            * (
                                min(1.0, left_white / max(1.0, 50.0 * scale_x * scale_y))
                                + min(1.0, right_white / max(1.0, 50.0 * scale_x * scale_y))
                            )
                            / 2.0,
                            0.60,
                            0.97,
                        )
                    ),
                    3,
                )

                prompts.append(
                    {
                        "rel_map_dx": rel_dx,
                        "rel_map_dy": rel_dy,
                        "screen_x": int(rx1 + fx + fcw // 2),
                        "screen_y": int(ry1 + fy + fch // 2),
                        "resource_type": resource_type,
                        "in_air": False,
                        "confidence": prompt_confidence,
                        "stack_count": 1,
                        "max_harvests": DEFAULT_MAX_HARVESTS,
                        "current_harvests": 0,
                        "source_type": "harvest_auto",
                    }
                )

            return prompts
        except Exception as exc:
            logger.error("_detect_active_harvest_prompt hatası: %s", exc)
            return prompts

    # ------------------------------------------------------------------
    # 4. Otomatik Materyal Haritalandırma ve Uzamsal Tekilleştirme
    # ------------------------------------------------------------------
    def _sync_existing_nodes_from_route_manager(
        self,
        route_manager: RouteManager,
        region: Optional[str] = None,
    ) -> None:
        """
        `RouteManager` içinde önceden kayıtlı olan düğümleri uzamsal tekilleştirme
        havuzuna (`_tracked_nodes`) senkronize eder.
        """
        try:
            regions_to_sync = [region] if region else route_manager.get_regions()
            for reg_name in regions_to_sync:
                if not reg_name:
                    continue
                try:
                    existing_nodes = route_manager.get_nodes(region=reg_name)
                except Exception:
                    existing_nodes = route_manager.get_route_nodes(region_name=reg_name)
                tracked_list = self._tracked_nodes.setdefault(reg_name, [])
                tracked_ids = {t.node_id for t in tracked_list}
                for n in existing_nodes:
                    if n.id not in tracked_ids:
                        max_h = int(getattr(n, "max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS)
                        curr_h = int(getattr(n, "current_harvests", 0) or 0)
                        st_cnt = int(getattr(n, "stack_count", max(1, max_h // DEFAULT_MAX_HARVESTS)) or 1)
                        src_t = str(getattr(n, "source_type", "unknown") or "unknown")
                        setattr(n, "max_harvests", max_h)
                        setattr(n, "current_harvests", curr_h)
                        setattr(n, "stack_count", st_cnt)
                        setattr(n, "source_type", src_t)
                        rm_state = str(getattr(n, "state", "active") or "active")
                        tracked_list.append(
                            TrackedResourceNode(
                                node_id=n.id,
                                region=n.region,
                                routes=list(n.routes),
                                resource_type=n.resource_type,
                                map_x=float(n.map_x),
                                map_y=float(n.map_y),
                                norm_x=float(n.norm_x),
                                norm_y=float(n.norm_y),
                                in_air=bool(n.in_air),
                                # H-sweep (dilim 13 O7): gathered_count
                                # hasat sayısıdır, görülme sayısı değil —
                                # RouteManager verisinde görülme sayacı
                                # yok; uydurma sayı (max(1, gathered))
                                # yerine 1 yazılır.
                                sightings_count=1,
                                patch_path=n.patch_path,
                                stack_count=st_cnt,
                                max_harvests=max_h,
                                current_harvests=curr_h,
                                source_type=src_t,
                                state=rm_state,
                                rest_started_timestamp=(
                                    float(getattr(n, "last_visited_timestamp", 0.0) or 0.0)
                                    if rm_state == "resting"
                                    else 0.0
                                ),
                            )
                        )
        except Exception as exc:
            logger.debug("RouteManager düğüm senkronizasyonu hatası: %s", exc)

    def _parse_material_detection(
        self,
        item: Any,
        frame_w: int,
        frame_h: int,
    ) -> Optional[Dict[str, Any]]:
        """
        `MaterialScanner`, radar veya harici YZ dedektörleri tarafından üretilen materyal/gem tespit
        öğesini (dict, nesne veya tuple) standart sözlük yapısına
        (`rel_map_dx`, `rel_map_dy`, `resource_type`, `in_air`, `confidence`,
        `stack_count`, `max_harvests`, `current_harvests`, `source_type`) çevirir.
        Merkeze sabitlenmeyi (center-sticking) önler; radar kutupsal (açı/mesafe) koordinatlarını,
        normalize edilmiş YZ kutularını ve mutlak harita koordinatlarını hatasız ayrıştırır.
        """
        try:
            if item is None:
                return None

            raw: Dict[str, Any] = {}
            if isinstance(item, dict):
                raw = dict(item)
            elif hasattr(item, "to_dict") and callable(item.to_dict):
                raw = dict(item.to_dict())
            elif hasattr(item, "__dict__"):
                raw = dict(vars(item))
            elif isinstance(item, (tuple, list)):
                if len(item) >= 2 and isinstance(item[0], (int, float)) and isinstance(item[1], (int, float)):
                    raw = {
                        "rel_map_dx": float(item[0]),
                        "rel_map_dy": float(item[1]),
                        "resource_type": str(item[2]) if len(item) >= 3 else None,
                    }
                else:
                    return None
            else:
                return None

            # Kaynak türü (minimap_auto, 3d_auto, harvest_auto, radar_auto) belirleme
            raw_source = str(raw.get("source_type") or raw.get("source") or "").strip().lower()
            # F17: "minimap_material" içsel minimap tarayıcısının etiketidir
            # (explorer_mode._scan_minimap_icons_internal /
            # MaterialScanner serileştirmesi) — birinci taraf minimap
            # üreticisidir, dış girdi değildir. Eskiden tanınmayıp
            # "unknown"a düşüyor, güven 0.05'e kırpılıyor ve P1-3 kapısında
            # (0.05 < 0.55) adım yolundaki filiz reddediliyordu.
            if raw_source in ("minimap", "minimap_auto", "yz_minimap", "minimap_material"):
                source_type = "minimap_auto"
            elif raw_source in ("gather_prompt", "harvest", "harvest_auto", "f_gather"):
                source_type = "harvest_auto"
            elif raw_source in ("radar", "radar_auto"):
                source_type = "radar_auto"
            elif raw_source in ("3d_world", "3d_auto", "3d", "world", "yz_3d"):
                source_type = "3d_auto"
            else:
                # Bilinmeyen kaynak minimap sayılmaz (F-0058); koordinat
                # kanıtı olan dallar aşağıda kaynağı yeniden sınıflandırır.
                source_type = "unknown"

            rel_dx: Optional[float] = None
            rel_dy: Optional[float] = None
            exact_map_x: Optional[float] = None
            exact_map_y: Optional[float] = None

            # 1. Mutlak harita koordinatları doğrudan verilmişse (exact_map_x / map_x)
            if "exact_map_x" in raw and "exact_map_y" in raw and raw["exact_map_x"] is not None and raw["exact_map_y"] is not None:
                exact_map_x = float(raw["exact_map_x"])
                exact_map_y = float(raw["exact_map_y"])
                rel_dx = exact_map_x - float(self.map_x)
                rel_dy = exact_map_y - float(self.map_y)
            elif "map_x" in raw and "map_y" in raw and raw["map_x"] is not None and raw["map_y"] is not None:
                exact_map_x = float(raw["map_x"])
                exact_map_y = float(raw["map_y"])
                rel_dx = exact_map_x - float(self.map_x)
                rel_dy = exact_map_y - float(self.map_y)
            elif "norm_x" in raw and "norm_y" in raw and raw["norm_x"] is not None and raw["norm_y"] is not None:
                exact_map_x = float(raw["norm_x"]) * float(self.map_width)
                exact_map_y = float(raw["norm_y"]) * float(self.map_height)
                rel_dx = exact_map_x - float(self.map_x)
                rel_dy = exact_map_y - float(self.map_y)

            # 2. Radar kutupsal koordinatları (polar coordinates: mesafe ve açı)
            if (rel_dx is None or rel_dy is None) and (
                any(k in raw for k in ("bearing_deg", "angle_deg", "bearing", "radar_angle"))
                and any(k in raw for k in ("distance", "dist", "range", "radar_distance"))
            ):
                bearing_val = float(
                    raw.get("bearing_deg", raw.get("angle_deg", raw.get("bearing", raw.get("radar_angle", 0.0))))
                )
                dist_val = float(
                    raw.get("distance", raw.get("dist", raw.get("range", raw.get("radar_distance", 0.0))))
                )
                rad = math.radians(bearing_val)
                # Kuzey = 0° (dy < 0), Doğu = 90° (dx > 0)
                rel_dx = dist_val * math.sin(rad)
                rel_dy = -dist_val * math.cos(rad)
                if not raw_source:
                    source_type = "radar_auto"

            # 3. Doğrudan göreli harita ofseti (rel_map_dx, rel_map_dy) kontrolü
            if rel_dx is None or rel_dy is None:
                for kx, ky in (
                    ("rel_map_dx", "rel_map_dy"),
                    ("map_dx", "map_dy"),
                    ("rel_dx", "rel_dy"),
                    ("dx", "dy"),
                    ("offset_x", "offset_y"),
                    ("radar_dx", "radar_dy"),
                ):
                    if kx in raw and ky in raw and raw[kx] is not None and raw[ky] is not None:
                        rel_dx = float(raw[kx])
                        rel_dy = float(raw[ky])
                        break

            # 4. Minimap içi koordinat verildiyse minimap merkezine göre ofseti hesapla
            if rel_dx is None or rel_dy is None:
                x1, y1, x2, y2 = self._get_minimap_pixel_rect(frame_w, frame_h)
                mcx = max(1.0, float(x2 - x1)) * 0.50
                mcy = max(1.0, float(y2 - y1)) * 0.545
                if "minimap_x" in raw and "minimap_y" in raw and raw["minimap_x"] is not None:
                    mx = float(raw["minimap_x"])
                    my = float(raw["minimap_y"])
                    if mx >= x1 and my >= y1:
                        mx -= float(x1)
                        my -= float(y1)
                    rel_dx = (mx - mcx) * self.minimap_to_map_scale
                    rel_dy = (my - mcy) * self.minimap_to_map_scale
                    # Minimap oyuncu oku merkez artefaktını reddet (merkeze yapışmayı önler)
                    if math.hypot(rel_dx, rel_dy) < 3.5:
                        return None
                    source_type = "minimap_auto"

            # 5. 3D Dünya ekran koordinatı (screen_x, screen_y veya bbox / cx, cy) verildiyse
            if rel_dx is None or rel_dy is None:
                sx: Optional[float] = None
                sy: Optional[float] = None
                if "screen_x" in raw and "screen_y" in raw:
                    sx = float(raw["screen_x"])
                    sy = float(raw["screen_y"])
                elif "cx" in raw and "cy" in raw:
                    sx = float(raw["cx"])
                    sy = float(raw["cy"])
                elif "center_x" in raw and "center_y" in raw:
                    sx = float(raw["center_x"])
                    sy = float(raw["center_y"])
                elif "x" in raw and "y" in raw:
                    sx = float(raw["x"])
                    sy = float(raw["y"])
                elif "bbox" in raw and isinstance(raw["bbox"], (tuple, list)) and len(raw["bbox"]) >= 4:
                    bx, by, bw, bh = [float(v) for v in raw["bbox"][:4]]
                    sx = bx + bw / 2.0
                    sy = by + bh / 2.0
                elif any(k in raw for k in ("x1", "xmin", "left")) and any(k in raw for k in ("x2", "xmax", "right")):
                    bx1 = float(raw.get("x1", raw.get("xmin", raw.get("left", 0.0))))
                    by1 = float(raw.get("y1", raw.get("ymin", raw.get("top", 0.0))))
                    bx2 = float(raw.get("x2", raw.get("xmax", raw.get("right", 0.0))))
                    by2 = float(raw.get("y2", raw.get("ymax", raw.get("bottom", 0.0))))
                    sx = (bx1 + bx2) / 2.0
                    sy = (by1 + by2) / 2.0

                if sx is not None and sy is not None:
                    # Normalize koordinatları (0.0..1.0) piksele çevir (merkeze yapışmayı önler, F-0212).
                    if _is_normalized_coord_pair(sx, sy) and frame_w > 10 and frame_h > 10:
                        sx *= float(frame_w)
                        sy *= float(frame_h)

                    x1, y1, x2, y2 = self._get_minimap_pixel_rect(frame_w, frame_h)
                    if x1 <= sx <= x2 and y1 <= sy <= y2:
                        # Minimap ROI içinde bir tespit (sprout vb.)
                        mcx = (x1 + x2) / 2.0
                        mcy = y1 + (y2 - y1) * 0.545
                        rel_dx = (sx - mcx) * self.minimap_to_map_scale
                        rel_dy = (sy - mcy) * self.minimap_to_map_scale
                        if math.hypot(rel_dx, rel_dy) < 3.5:
                            return None
                        source_type = "minimap_auto"
                    else:
                        # 3D kamera görünümü tespiti (yön açısı ve perspektif foreshortening ile)
                        heading = raw.get("heading_deg", raw.get("player_heading", self.player_heading_deg))
                        rel_dx, rel_dy = self._project_3d_screen_to_map_offset(
                            sx, sy, frame_w, frame_h, heading_deg=heading
                        )
                        if source_type in ("minimap_auto", "unknown") and not raw_source:
                            source_type = "3d_auto"

            # 6. Dead-reckoning yer değiştirme ofsetini ekle
            dr_x = float(raw.get("dead_reckoning_dx", raw.get("dr_dx", 0.0)))
            dr_y = float(raw.get("dead_reckoning_dy", raw.get("dr_dy", 0.0)))

            if rel_dx is None or rel_dy is None:
                # H-sweep (dilim 13 K3): koordinatsız tespit elenir —
                # harvest_auto için bile oyuncu ayağına (0, 0) düğüm
                # yazılmaz. İçsel [F] istemleri koordinatlarıyla
                # üretilir ve _parse'a uğramaz, bu dal onları etkilemez.
                return None

            rel_dx += dr_x
            rel_dy += dr_y

            # 3D dünya tespitlerinde oyuncunun doğrudan ayak ucuna düşen sahte merkez kutuları ele
            if source_type == "3d_auto" and math.hypot(rel_dx, rel_dy) < 3.0:
                return None

            # H-sweep (dilim 13 K3): materyal adı yoksa spesifik cevher
            # ("Orichalcum Ore") varsayılmaz; tür bilinmiyor diye jenerik
            # işaret (UNKNOWN_RESOURCE_TYPE) yazılır, kapı sonra
            # isimlendirebilir.
            resource_type = str(
                raw.get("resource_type")
                or raw.get("material_type")
                or raw.get("name")
                or raw.get("label")
                or raw.get("category")
                or UNKNOWN_RESOURCE_TYPE
            ).strip()
            if not resource_type or resource_type == "None":
                resource_type = UNKNOWN_RESOURCE_TYPE

            # Tüy (feather) ve kanat (wing) ikonlarını kesinlikle reddet ("tüy odyle değil tüy toplamayacağız")
            rt_lower = resource_type.lower()
            if any(bad_tok in rt_lower for bad_tok in ("feather", "tüy", "tuy", "wing")):
                return None
            if source_type == "minimap_auto" and "odyle" in rt_lower:
                return None

            in_air = bool(raw.get("in_air", False))
            # H-sweep (dilim 13 K3): güven yoksa 1.0 varsayılmaz.
            # Ölçülmemiş güven "mükemmel" değildir — tespit elenir.
            # İçsel dedektörler (minimap formülü, 3D arma, [F] istemi)
            # her zaman ölçülmüş güven taşır; bu dal yalnızca dış
            # girdileri etkiler.
            conf_raw = raw.get("confidence")
            if conf_raw is None:
                conf_raw = raw.get("score")
            if conf_raw is None:
                logger.debug(
                    "Materyal tespiti güveni yok, elendi (kaynak=%s)",
                    source_type,
                )
                return None
            try:
                confidence = float(conf_raw)
            except (TypeError, ValueError):
                # P1-3 (K-07): ayrıştırılamayan güven 0.5 varsayılmaz —
                # güveni yok/bozuk tespit bilinmiyor sayılır ve elenir
                # (güvensiz tespit dalındaki `return None` ile aynı kural).
                logger.debug(
                    "Materyal tespiti güveni ayrıştırılamadı (%r), elendi (kaynak=%s)",
                    conf_raw,
                    source_type,
                )
                return None
            if confidence > 1.0:
                confidence = confidence / 100.0
            confidence = float(np.clip(confidence, 0.05, 1.0))

            stack_count = max(1, int(raw.get("stack_count", raw.get("stack", 1)) or 1))
            max_harvests = int(
                raw.get("max_harvests")
                or (DEFAULT_MAX_HARVESTS * stack_count)
            )
            if max_harvests <= 0:
                max_harvests = DEFAULT_MAX_HARVESTS * stack_count
            current_harvests = max(0, int(raw.get("current_harvests", 0) or 0))

            parsed_res: Dict[str, Any] = {
                "rel_map_dx": float(rel_dx),
                "rel_map_dy": float(rel_dy),
                "resource_type": resource_type,
                "in_air": in_air,
                "confidence": confidence,
                "stack_count": int(stack_count),
                "max_harvests": int(max_harvests),
                "current_harvests": int(current_harvests),
                "source_type": str(source_type),
            }
            # F-1389/F-1390: dedektörün sınıf kimliğini koru — yoksa
            # _register_parsed_detections bunu kalite kapısına model_verdict
            # olarak iletemez ve canavar (11) tespiti kapıdan kaçardı.
            for _cid_key in ("class_id", "cls", "class"):
                if raw.get(_cid_key) is not None:
                    try:
                        _cid = int(raw[_cid_key])
                    except (TypeError, ValueError):
                        _cid = -1
                    if _cid >= 0:
                        parsed_res["class_id"] = _cid
                    break
            if exact_map_x is not None and exact_map_y is not None:
                parsed_res["exact_map_x"] = round(float(exact_map_x), 3)
                parsed_res["exact_map_y"] = round(float(exact_map_y), 3)
                parsed_res["has_absolute_map_coords"] = True

            return parsed_res
        except Exception as exc:
            logger.error("Materyal tespiti ayrıştırılırken hata: %s", exc)
            return None

    def _inventory_recording_blocked(self, frame_bgr: np.ndarray) -> bool:
        """
        Çanta açıksa veya yeni kapanmışsa materyal kaydı yapma.
        Kapanıştan sonra kısa süre (geçiş kareleri) de bloklu kalır.
        """
        now = time.time()
        open_now = False
        try:
            if self.vision_engine is not None and hasattr(self.vision_engine, "is_inventory_open"):
                open_now = bool(self.vision_engine.is_inventory_open(frame_bgr))
            else:
                from core.vision import detect_inventory_panel

                open_now = bool(detect_inventory_panel(frame_bgr))
        except Exception as exc:
            logger.debug("Envanter kapısı okunamadı: %s", exc)
            open_now = False
        if open_now:
            self._inventory_block_until = now + 0.9
            return True
        return now < float(self._inventory_block_until)

    def _note_rel_track(
        self,
        tracked: TrackedResourceNode,
        rel_dx: float,
        rel_dy: float,
        count_sighting: bool,
    ) -> None:
        """Kamerayla dönen aynı ikonun göreli ofsetini günceller. Konumu sürüklemez."""
        if count_sighting:
            tracked.sightings_count += 1
            tracked.last_seen_timestamp = time.time()
        tracked.last_rel_dx = float(rel_dx)
        tracked.last_rel_dy = float(rel_dy)
        tracked.rel_track_valid = True

    def _find_orbit_continuation(
        self,
        region: str,
        rel_dx: float,
        rel_dy: float,
    ) -> Optional[TrackedResourceNode]:
        """
        Aynı materyalin etrafında tur atınca ekran/minimap ofseti kayar ve dünya
        koordinatı ``dedup_radius_px`` (22) dışına düşer. Son görülen göreli ofsete
        yakın ve taze bir iz aynı ziyarettir; yeni düğüm açılmaz.
        """
        now = time.time()
        best: Optional[TrackedResourceNode] = None
        best_d = float("inf")
        for tracked in self._tracked_nodes.get(region, []):
            if not tracked.rel_track_valid:
                continue
            if (now - float(tracked.last_seen_timestamp)) > self._orbit_rel_max_age_sec:
                continue
            dist = math.hypot(rel_dx - tracked.last_rel_dx, rel_dy - tracked.last_rel_dy)
            if dist <= self._orbit_rel_step_px and dist < best_d:
                best = tracked
                best_d = dist
        return best

    def _find_nearest_tracked_node(
        self,
        region: str,
        node_map_x: float,
        node_map_y: float,
        resource_type: Optional[str] = None,
    ) -> Tuple[Optional[TrackedResourceNode], float]:
        """
        Belirtilen bölgede `(node_map_x, node_map_y)` koordinatına `dedup_radius_px`
        (15-25px) yarıçapı içindeki en yakın kayıtlı düğümü bulur.
        Sınır bölgelerinde (22-25px) mükerrer yığılmayı (duplicate piling) önler.
        """
        candidates = self._tracked_nodes.get(region, [])
        if not candidates and self.route_manager is not None:
            self._sync_existing_nodes_from_route_manager(self.route_manager, region=region)
            candidates = self._tracked_nodes.get(region, [])

        effective_radius = min(DEDUP_RADIUS_MAX_PX, max(DEDUP_RADIUS_MIN_PX, float(self.dedup_radius_px)))
        nearest: Optional[TrackedResourceNode] = None
        min_dist = float("inf")
        # Sınır bölgesi (effective_radius..25px) adayı: eski kod ikinci tam
        # taramada arıyordu; tek geçişte aynı anda izlenir (D02). İkinci
        # tarama yalnızca ilkinde eşleşme yoksa çalışırdı, o yüzden sınır
        # adayı yarıçap dışındaki düğümlerden seçilir — sonuç aynıdır.
        border_best: Optional[TrackedResourceNode] = None
        border_dist = float("inf")
        border_applies = effective_radius < DEDUP_RADIUS_MAX_PX

        for tracked in candidates:
            dx = float(node_map_x) - tracked.map_x
            dy = float(node_map_y) - tracked.map_y
            dist = math.hypot(dx, dy)
            if dist <= effective_radius:
                if dist < min_dist:
                    min_dist = dist
                    nearest = tracked
            elif border_applies and dist <= DEDUP_RADIUS_MAX_PX and dist < border_dist:
                # Sınır bölgesinde yalnızca aynı tür tekilleştirilir (F-0213):
                # az görülmüş olması farklı türü aynı düğüm saymaz.
                if (
                    resource_type is None
                    or resource_type in (DEFAULT_RESOURCE_TYPE, "Gatherable Material")
                    or tracked.resource_type == resource_type
                ):
                    border_dist = dist
                    border_best = tracked

        if nearest is None and border_best is not None:
            nearest = border_best
            min_dist = border_dist

        return nearest, min_dist

    def _refine_existing_node_kalman_ema(
        self,
        tracked: TrackedResourceNode,
        obs_map_x: float,
        obs_map_y: float,
        obs_confidence: float,
        route_manager: Optional[RouteManager] = None,
        route: Optional[str] = None,
        obs_stack_count: int = 1,
        obs_max_harvests: Optional[int] = None,
        obs_resource_type: Optional[str] = None,
    ) -> None:
        """
        Mevcut düğümün koordinatını 2D Kalman kazancı ve EMA (Exponential Moving Average)
        ağırlıklı ortalamasıyla daha hassas hale getirir; aykırı gözlemleri (outliers) sınırlayarak
        merkeze yapışmayı (center-sticking) ve ani koordinat sıçramalarını önler.
        """
        try:
            tracked.sightings_count += 1
            tracked.last_seen_timestamp = time.time()

            # Aykırı gözlem filtreleme ve sıçrama sınırlaması (outlier gating)
            dx = float(obs_map_x) - tracked.map_x
            dy = float(obs_map_y) - tracked.map_y
            dist_obs = math.hypot(dx, dy)
            if dist_obs > DEDUP_RADIUS_MAX_PX:
                scale = 4.0 / dist_obs
                clamped_obs_x = tracked.map_x + dx * scale
                clamped_obs_y = tracked.map_y + dy * scale
            else:
                clamped_obs_x = float(obs_map_x)
                clamped_obs_y = float(obs_map_y)

            # 2D Kalman filtresi durum güncellemesi
            measurement_var = max(1.5, 12.0 / max(0.15, float(obs_confidence)))
            process_var = 0.25
            var_x = getattr(tracked, "kalman_variance_x", tracked.kalman_variance)
            var_y = getattr(tracked, "kalman_variance_y", tracked.kalman_variance)

            pred_var_x = var_x + process_var
            pred_var_y = var_y + process_var
            gain_x = pred_var_x / (pred_var_x + measurement_var)
            gain_y = pred_var_y / (pred_var_y + measurement_var)

            # EMA ağırlığı ile Kalman kazancını birleştir
            ema_weight = self.ema_alpha / math.sqrt(max(1, tracked.sightings_count))
            # Skaler np.clip yerine min/max (D02).
            eff_weight_x = min(0.55, max(0.04, 0.5 * gain_x + 0.5 * ema_weight))
            eff_weight_y = min(0.55, max(0.04, 0.5 * gain_y + 0.5 * ema_weight))

            tracked.map_x = (1.0 - eff_weight_x) * tracked.map_x + eff_weight_x * clamped_obs_x
            tracked.map_y = (1.0 - eff_weight_y) * tracked.map_y + eff_weight_y * clamped_obs_y
            tracked.norm_x = round(tracked.map_x / float(max(1, self.map_width)), 4)
            tracked.norm_y = round(tracked.map_y / float(max(1, self.map_height)), 4)

            tracked.kalman_variance_x = (1.0 - gain_x) * pred_var_x
            tracked.kalman_variance_y = (1.0 - gain_y) * pred_var_y
            tracked.kalman_variance = round(0.5 * (tracked.kalman_variance_x + tracked.kalman_variance_y), 4)

            # Güven tek yönlü şişirilmez (F-0059): gözlemlerin çalışan
            # ortalamasıdır; her gözlem +0.02 ekleyip 1.0'a tırmandırmaz.
            tracked.confidence = round(
                min(
                    1.0,
                    max(
                        0.0,
                        (
                            float(tracked.confidence)
                            * max(0, tracked.sightings_count - 1)
                            + float(obs_confidence)
                        )
                        / max(1, tracked.sightings_count),
                    ),
                ),
                4,
            )

            if obs_stack_count > tracked.stack_count:
                tracked.stack_count = int(obs_stack_count)
                tracked.max_harvests = max(tracked.max_harvests, DEFAULT_MAX_HARVESTS * tracked.stack_count)
            if obs_max_harvests is not None and int(obs_max_harvests) > tracked.max_harvests:
                tracked.max_harvests = int(obs_max_harvests)
            if (
                obs_resource_type
                and obs_resource_type not in (DEFAULT_RESOURCE_TYPE, "Gatherable Material")
                and tracked.resource_type in (DEFAULT_RESOURCE_TYPE, "Gatherable Material", "Gem")
            ):
                tracked.resource_type = str(obs_resource_type)

            if route and route != DEFAULT_ROUTE and route not in tracked.routes:
                tracked.routes.append(route)

            # H-sweep (dilim 13 O6): last_mapped_info son haritalamanın
            # kaydıdır, AUTO modunun değil — AUTO kapalıyken yazılan
            # düğüm de panelde görünür; bölge bilgisi de eklenir.
            # P0-9 (K-15): buradaki bölge düğümün kendi bölgesidir
            # (tracked.region) — bu fonksiyonda `target_region` diye bir
            # isim yoktu; NameError her mevcut-düğüm iyileştirmesini
            # patlatıp aşağıdaki except'e yutuyor, RouteManager
            # senkronizasyonu + save_region hiç çalışmıyordu. Aynı
            # değer iki satır aşağıda rm.get_node / rm.save_region
            # çağrılarında zaten tracked.region olarak kullanılıyor;
            # çağıranlar da düğümü bölge anahtarlı havuzdan bulur
            # (_find_nearest_tracked_node), yani tracked.region
            # çağıranın bölgesiyle aynıdır.
            self.last_mapped_info = {
                "type": "node_update",
                "node_id": tracked.node_id,
                "resource_type": tracked.resource_type,
                "region": tracked.region,
                "map_x": int(round(tracked.map_x)),
                "map_y": int(round(tracked.map_y)),
                "timestamp": time.time(),
                "sightings": tracked.sightings_count,
            }

            rm = route_manager or self.route_manager
            if rm is not None:
                node_obj = rm.get_node(tracked.node_id, region=tracked.region)
                if node_obj is not None:
                    node_obj.map_x = int(round(tracked.map_x))
                    node_obj.map_y = int(round(tracked.map_y))
                    node_obj.norm_x = tracked.norm_x
                    node_obj.norm_y = tracked.norm_y
                    node_obj.resource_type = tracked.resource_type
                    setattr(node_obj, "max_harvests", int(tracked.max_harvests))
                    setattr(node_obj, "current_harvests", int(tracked.current_harvests))
                    setattr(node_obj, "stack_count", int(tracked.stack_count))
                    setattr(node_obj, "source_type", str(tracked.source_type))
                    if route and route != DEFAULT_ROUTE and route not in node_obj.routes:
                        node_obj.routes.append(route)
                    rm.save_region(tracked.region)
        except NameError:
            # P0-9 (K-15): programlama hatası (tanımsız isim) yutulmaz —
            # sessizce yutmak disk senkronizasyonunu her düğümde
            # öldürüyordu. Logla ve fail-closed: yukarı fırlat.
            logger.exception("Düğüm Kalman/EMA iyileştirmede tanımsız isim (P0-9):")
            raise
        except Exception as exc:
            logger.error("Düğüm Kalman/EMA koordinat iyileştirme hatası: %s", exc)

    def _register_parsed_detections(
        self,
        bgr: np.ndarray,
        parsed_items: List[Dict[str, Any]],
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Ayrıştırılmış materyal tespitlerini uzamsal tekilleştirme (`dedup_radius_px`)
        ile kontrol eder; yeni düğümler için 80x80 minimap kesitini kırparak
        `RouteManager` ve `_tracked_nodes` içine `max_harvests = 3 * stack_count`,
        `current_harvests = 0` olarak kaydeder.
        """
        newly_discovered_nodes: List[Dict[str, Any]] = []
        if not parsed_items:
            return newly_discovered_nodes

        rm = route_manager or self.route_manager
        target_region = str(
            region
            or (rm.active_region if rm is not None else DEFAULT_REGION)
            or DEFAULT_REGION
        ).strip()
        target_route = str(
            route
            or (rm.active_route if rm is not None else DEFAULT_ROUTE)
            or DEFAULT_ROUTE
        ).strip()

        if rm is not None:
            self._sync_existing_nodes_from_route_manager(rm, region=target_region)

        # 1. Aynı kare/tespit grubu içindeki tespitleri dedup_radius_px ile kümele
        clustered_items = self.cluster_detections(parsed_items, cluster_radius=self.dedup_radius_px)

        # Odometri takipçisi konumunu güncel tut
        self.odometry_tracker.map_x = float(self.map_x)
        self.odometry_tracker.map_y = float(self.map_y)
        self.odometry_tracker.map_width = int(self.map_width)
        self.odometry_tracker.map_height = int(self.map_height)
        self.odometry_tracker.dead_reckoning_dx = float(self.dead_reckoning_dx)
        self.odometry_tracker.dead_reckoning_dy = float(self.dead_reckoning_dy)

        for parsed in clustered_items:
            # H-sweep (dilim 13 K3): güveni/isimsiz-koordinatsız tespit
            # kapıda elenir; float(None) / "None" adı rotaya girmez.
            if parsed.get("confidence") is None or not parsed.get("resource_type"):
                logger.info(
                    "Tespit eksik güven/isim nedeniyle elendi (kaynak=%s)",
                    parsed.get("source_type", "?"),
                )
                self.total_gate_rejected += 1
                continue
            rel_dx = float(parsed["rel_map_dx"])
            rel_dy = float(parsed["rel_map_dy"])
            resource_type = str(parsed["resource_type"])
            in_air = bool(parsed["in_air"])
            confidence = float(parsed["confidence"])
            stack_count = max(1, int(parsed.get("stack_count", 1)))
            max_harvests = max(
                DEFAULT_MAX_HARVESTS * stack_count,
                int(parsed.get("max_harvests", DEFAULT_MAX_HARVESTS * stack_count)),
            )
            current_harvests = max(0, int(parsed.get("current_harvests", 0)))
            source_type = str(parsed.get("source_type", "minimap_auto"))

            if parsed.get("has_absolute_map_coords") and parsed.get("exact_map_x") is not None and parsed.get("exact_map_y") is not None:
                node_map_x = float(parsed["exact_map_x"])
                node_map_y = float(parsed["exact_map_y"])
                norm_x = round(node_map_x / float(max(1, self.map_width)), 4)
                norm_y = round(node_map_y / float(max(1, self.map_height)), 4)
            else:
                node_map_x, node_map_y, norm_x, norm_y = self.odometry_tracker.offset_to_world_coords(
                    rel_dx,
                    rel_dy,
                    is_map_offset=True,
                    dead_reckoning_dx=self.dead_reckoning_dx,
                    dead_reckoning_dy=self.dead_reckoning_dy,
                )

            # Yeni kaynak veya aday için patch kırp ve kapıdan geçir
            patch_80 = self.extract_clean_minimap_patch(
                frame_bgr=bgr,
                rel_map_dx=rel_dx,
                rel_map_dy=rel_dy,
            )
            if is_parchment_scroll_patch(patch_80):
                logger.info(
                    "Parşömen ikonu toplanabilir değil, düğüm yazılmadı (%s, konum=%.0f,%.0f)",
                    resource_type,
                    node_map_x,
                    node_map_y,
                )
                self.total_gate_rejected += 1
                continue

            # Low-confidence 3D projections without a harvest prompt are too noisy.
            if source_type == "3d_auto" and confidence < WORLD_3D_CONF_MIN:
                logger.info(
                    "3D aday düşük güven (%.2f < %.2f), düğüm yazılmadı (%s)",
                    confidence,
                    WORLD_3D_CONF_MIN,
                    resource_type,
                )
                self.total_gate_rejected += 1
                continue

            # H-sweep (dilim 13 K3): minimap kapısı yoktu — 3D için
            # WORLD_3D_CONF_MIN varken minimap_auto her güvenle
            # yazılıyordu. İçsel minimap dedektörleri zaten >=0.68
            # üretir (formül tabanı / MaterialScanner 0.78+); altı
            # ölçülmemiş/uydurma güvendir.
            if source_type == "minimap_auto" and confidence < MINIMAP_SPROUT_CONF_MIN:
                logger.info(
                    "Minimap aday düşük güven (%.2f < %.2f), düğüm yazılmadı (%s)",
                    confidence,
                    MINIMAP_SPROUT_CONF_MIN,
                    resource_type,
                )
                self.total_gate_rejected += 1
                continue

            # P1-3 (K-07): harvest_auto / radar_auto / unknown kapısızdı —
            # 3d_auto ve minimap_auto tabanları varken bu kaynaklar her
            # güvenle kalite kapısına ulaşırdı. Dış girdide 0.05'e
            # kırpılmış güven bile düğüme dönerdi. İçsel [F] istemi
            # 0.60+ ürettiği için 0.55 tabanı gerçek istemi geçirir.
            if source_type in ("harvest_auto", "radar_auto", "unknown") and confidence < HARVEST_RADAR_CONF_MIN:
                logger.info(
                    "Harvest/radar aday düşük güven (%.2f < %.2f), düğüm yazılmadı (%s, kaynak=%s)",
                    confidence,
                    HARVEST_RADAR_CONF_MIN,
                    resource_type,
                    source_type,
                )
                self.total_gate_rejected += 1
                continue

            # F-1389/F-1390 (R11): tespitin dedektör sınıfını model_verdict
            # olarak kapıya ilet — yoksa kapı canavar sınıfını (11) bilemez
            # ve canavar düğümü kabul ederdi.
            model_verdict = None
            try:
                if parsed.get("class_id") is not None:
                    model_verdict = (int(parsed["class_id"]), float(confidence))
            except (TypeError, ValueError):
                model_verdict = None

            gate = evaluate_node_candidate(
                patch_80,
                source_type=source_type,
                resource_type=resource_type,
                detector=self._get_quality_detector(),
                conf_min=MODEL_CONF_MIN,
                model_verdict=model_verdict,
                check_feather=source_type.startswith("minimap"),
                check_monster=source_type.startswith("minimap"),
                check_terrain=False,
                exemplar_memory=self._get_exemplar_memory(),
            )
            if gate.reject:
                self.total_gate_rejected += 1
                logger.info(
                    "AI/CV/Exemplar kapısı reddetti [%s]: reasons=%s class=%s conf=%s (%s @ %.0f,%.0f)",
                    source_type,
                    gate.reasons,
                    gate.class_id,
                    None if gate.confidence is None else round(float(gate.confidence), 3),
                    resource_type,
                    node_map_x,
                    node_map_y,
                )
                continue

            verified = getattr(gate, "verified_resource_type", None)
            # v20-targena-3dauto (R1+R2): 3D'den gelen spesifik ad
            # (Targena vb.) kapının jenerik doğrulamasıyla ("Gem" vb.)
            # ezilmez; 3D adayda ad belirlenemediyse yanlış jenerik
            # kayıt yazılmaz — aday atılır.
            resource_type, _drop_unnamed = _resolve_mapped_resource_name(
                resource_type, verified, source_type
            )
            if _drop_unnamed:
                logger.info(
                    "3D aday adı belirlenemedi, düğüm yazılmadı (%s)",
                    resource_type,
                )
                self.total_gate_rejected += 1
                continue
            # H-sweep (dilim 13 K4): güven yalnızca yukarı şişirilmez.
            # exemplar_sim ayrı bir kanıttır; ölçülmüş güveni yükseltmez
            # (düşük güvenli tespit exemplare benziyor diye yüksek güvenli
            # sayılmaz). Benzerlik kapı kararında zaten kullanılır.

            existing_node, dist_px = self._find_nearest_tracked_node(
                region=target_region,
                node_map_x=node_map_x,
                node_map_y=node_map_y,
                resource_type=resource_type,
            )

            if existing_node is not None:
                self._refine_existing_node_kalman_ema(
                    tracked=existing_node,
                    obs_map_x=node_map_x,
                    obs_map_y=node_map_y,
                    obs_confidence=confidence,
                    route_manager=rm,
                    route=target_route,
                    obs_stack_count=stack_count,
                    obs_max_harvests=max_harvests,
                    obs_resource_type=resource_type,
                )
                self._note_rel_track(existing_node, rel_dx, rel_dy, count_sighting=False)
                self._emit_node_updated(existing_node.to_dict())
                if self.auto_cartography_enabled:
                    self.persist_to_region_data(target_region)
                logger.debug(
                    "Mevcut kaynak tekilleştirildi (ID=#%d, mesafe=%.2f px <= %.1f px): "
                    "güncel konum=(%.2f, %.2f), sightings=%d, max_harvests=%d",
                    existing_node.node_id,
                    dist_px,
                    self.dedup_radius_px,
                    existing_node.map_x,
                    existing_node.map_y,
                    existing_node.sightings_count,
                    existing_node.max_harvests,
                )
                continue

            orbit_node = self._find_orbit_continuation(target_region, rel_dx, rel_dy)
            if orbit_node is not None:
                self._note_rel_track(orbit_node, rel_dx, rel_dy, count_sighting=True)
                self._emit_node_updated(orbit_node.to_dict())
                logger.debug(
                    "Aynı ziyaret (tur) tekilleştirildi ID=#%d rel=(%.1f, %.1f) -> yeni kayıt yok",
                    orbit_node.node_id,
                    rel_dx,
                    rel_dy,
                )
                continue

            routes_list = [target_route] if target_route and target_route != DEFAULT_ROUTE else []
            created_node: Optional[NodeData] = None

            if rm is not None:
                # max_harvests/current_harvests/source_type add_node'a verilir:
                # add_node içindeki save_region diske bunlarla yazar (F-0002).
                # Sonradan setattr ile düzeltmek, AUTO kapalıyken diske hiç
                # yansımıyordu.
                created_node = rm.add_node(
                    region=target_region,
                    route=target_route,
                    patch_img=patch_80,
                    in_air=in_air,
                    map_x=int(round(node_map_x)),
                    map_y=int(round(node_map_y)),
                    norm_x=norm_x,
                    norm_y=norm_y,
                    resource_type=resource_type,
                    max_harvests=int(max_harvests),
                    current_harvests=int(current_harvests),
                    source_type=str(source_type),
                )
                if created_node is not None:
                    setattr(created_node, "max_harvests", int(max_harvests))
                    setattr(created_node, "current_harvests", int(current_harvests))
                    setattr(created_node, "stack_count", int(stack_count))
                    setattr(created_node, "source_type", str(source_type))
                    # Bellek üstü alanlar da diske yansısın.
                    rm.save_region(target_region)
                    self._save_direct_region_node_patch(
                        base_dir=rm.base_dir,
                        region=target_region,
                        node_id=created_node.id,
                        patch_img=patch_80,
                    )

            # RouteManager yokken ID len()+1 ile çakışır (F-0214): en büyük
            # mevcut ID + 1 kullanılır.
            assigned_id = (
                int(created_node.id)
                if created_node is not None
                else max(
                    [t.node_id for t in self._tracked_nodes.get(target_region, [])],
                    default=0,
                )
                + 1
            )
            if created_node is not None:
                # RouteManager patch'i diske yazdı; gerçek yol kullanılır.
                patch_path = created_node.patch_path
            else:
                # S-1101 (S-0503 deseni): add_node None döndüyse (kara liste
                # reddi) veya RouteManager yoksa patch diske yazılmamıştır;
                # yazılmamış dosya yolu kaydedilmez. Doğrudan yazım denenir;
                # yazım başarısızsa _save_direct_region_node_patch "" döner
                # (F-0211) ve patch_path boş kalır.
                patch_path = self._save_direct_region_node_patch(
                    base_dir=rm.base_dir if rm is not None else "routes",
                    region=target_region,
                    node_id=assigned_id,
                    patch_img=patch_80,
                )

            tracked_new = TrackedResourceNode(
                node_id=assigned_id,
                region=target_region,
                routes=routes_list,
                resource_type=resource_type,
                map_x=node_map_x,
                map_y=node_map_y,
                norm_x=norm_x,
                norm_y=norm_y,
                in_air=in_air,
                sightings_count=1,
                confidence=confidence,
                patch_path=patch_path,
                patch_img=patch_80,
                stack_count=stack_count,
                max_harvests=max_harvests,
                current_harvests=current_harvests,
                source_type=source_type,
            )
            self._note_rel_track(tracked_new, rel_dx, rel_dy, count_sighting=False)
            self._tracked_nodes.setdefault(target_region, []).append(tracked_new)

            # P1-6 (K-06): tek kaynak — her yeni düğüm yalnızca kendi
            # türünün sayacını artırır. Eskiden harvest_auto hem kendi
            # sayacını hem total_3d_recorded'ı artırıyor (çift sayım),
            # radar_auto ise hiçbir sayaca girmiyordu.
            if source_type == "minimap_auto":
                self.total_minimap_recorded += 1
            elif source_type == "3d_auto":
                self.total_3d_recorded += 1
            elif source_type == "harvest_auto":
                self.total_harvest_recorded += 1
            elif source_type == "radar_auto":
                self.total_radar_recorded += 1

            node_dict = tracked_new.to_dict()
            newly_discovered_nodes.append(node_dict)
            self._emit_node_discovered(node_dict)

            if self.auto_cartography_enabled:
                self.auto_mapped_count += 1
                rg = self.get_route_graph(target_region)
                rg.connect_node_to_nearest_waypoint(assigned_id, node_map_x, node_map_y)
                self.persist_to_region_data(target_region)
            # H-sweep (dilim 13 O6): last_mapped_info AUTO'dan bağımsız
            # güncellenir (yukarıdaki node_update dalıyla aynı kural).
            self.last_mapped_info = {
                "type": "node",
                "node_id": assigned_id,
                "resource_type": resource_type,
                "region": target_region,
                "map_x": int(round(node_map_x)),
                "map_y": int(round(node_map_y)),
                "timestamp": time.time(),
                "source_type": source_type,
                "confidence": round(confidence, 3),
            }

            logger.info(
                "Yeni materyal otomatik haritalandırıldı [%s]: Bölge=%s, Rota=%s, ID=#%d (%s) "
                "konum=(%d, %d), max_harvests=%d (stack=%d), patch=%dx%d -> %s",
                source_type,
                target_region,
                target_route,
                assigned_id,
                resource_type,
                int(round(node_map_x)),
                int(round(node_map_y)),
                max_harvests,
                stack_count,
                patch_80.shape[1],
                patch_80.shape[0],
                patch_path,
            )

        return newly_discovered_nodes

    def record_from_minimap(
        self,
        frame: np.ndarray,
        detections: Optional[List[Any]] = None,
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        update_odometry: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Sağ üstteki minimap ROI'sini (`Pilgrim's Ridge`, `x: 0.855..0.992, y: 0.065..0.255`)
        çıkartır, canavar noktalarını filtreleyerek toplanabilir materyal ikonlarını
        (yalnızca 3 yapraklı filizler + yeşil `2`/`3` yığın rozetleri) tespit eder,
        `VisualOdometryTracker` ile dünya koordinatlarını hesaplar ve
        80x80 görsel şablonunu kaydederek `max_harvests = 3 * stack_count` ile rotaya ekler.
        Kanatlı Odyle/tüy ikonları tespit edilip eklenmez; kesin olarak reddedilir (F-0413).

        LEGACY (H-sweep, dilim 13 Y5): üretimde çağrılmaz — canlı yol
        `process_frame_discoveries`'dir (explorer_mode.py:2851). Bu
        fonksiyon yalnızca testler / eski çağıranlar için korunur.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return []

            if self._inventory_recording_blocked(bgr):
                return []

            if update_odometry or not self.walk_trajectory:
                self.update_player_position(bgr)

            frame_h, frame_w = bgr.shape[:2]
            parsed_list: List[Dict[str, Any]] = []

            # 1. Doğrudan minimap görüntü işleme ile materyal ikonlarını ve yeşil yığın rozetlerini tara
            auto_mm = self._detect_minimap_gatherables(bgr)
            parsed_list.extend(auto_mm)

            # 2. Dışarıdan geçirilen minimap tespitleri varsa onları da birleştir
            if detections:
                for item in detections:
                    parsed = self._parse_material_detection(item, frame_w=frame_w, frame_h=frame_h)
                    if parsed is not None:
                        parsed["source_type"] = "minimap_auto"
                        parsed_list.append(parsed)

            if not parsed_list:
                return []

            return self._register_parsed_detections(
                bgr=bgr,
                parsed_items=parsed_list,
                route_manager=route_manager,
                region=region,
                route=route,
            )
        except Exception as exc:
            logger.error("record_from_minimap hatası: %s", exc)
            return []

    def record_from_3d_world(
        self,
        frame: np.ndarray,
        detections: Optional[List[Any]] = None,
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        update_odometry: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Minimap'te görünmeyen ancak 3D oyun dünyasında beyaz kanatlı toplama arması (`Aria`, `Mela`)
        veya yüzen `[E]` ikonu ile görünen kaynakları oyuncu ayak konumuna `(w*0.5, h*0.54)` göre
        harita koordinatlarına izdüşürerek `source_type="3d_auto"` ve `max_harvests=3` ile kaydeder.

        LEGACY (H-sweep, dilim 13 Y5): üretimde çağrılmaz — canlı yol
        `process_frame_discoveries`'dir. Yalnızca testler / eski
        çağıranlar için korunur.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return []

            frame_h, frame_w = bgr.shape[:2]
            # record_from_minimap gibi odometri güncellenir (F-0060).
            if update_odometry or not self.walk_trajectory:
                self.update_player_position(bgr)

            parsed_list: List[Dict[str, Any]] = []

            auto_3d = self._detect_3d_world_gatherables(bgr)
            parsed_list.extend(auto_3d)

            if detections:
                for item in detections:
                    parsed = self._parse_material_detection(item, frame_w=frame_w, frame_h=frame_h)
                    if parsed is not None and parsed.get("source_type") == "3d_auto":
                        parsed_list.append(parsed)

            if not parsed_list:
                return []

            return self._register_parsed_detections(
                bgr=bgr,
                parsed_items=parsed_list,
                route_manager=route_manager,
                region=region,
                route=route,
            )
        except Exception as exc:
            logger.error("record_from_3d_world hatası: %s", exc)
            return []

    def record_harvest_event(
        self,
        frame: np.ndarray,
        material_type: Optional[str] = None,
        detections: Optional[List[Any]] = None,
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        update_odometry: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Oyuncu veya bot bir materyal üzerinde `[F Gather]` / `[E]` etkileşiminde bulunduğunda
        veya ekranda `[F Gather]` belirdiğinde, düğüm henüz rotada yoksa o konumu
        `source_type="harvest_auto"` ve `max_harvests=3` olarak kaydeder.

        LEGACY (H-sweep, dilim 13 Y5): üretimde çağrılmaz — canlı yol
        `process_frame_discoveries`'dir. Yalnızca testler / eski
        çağıranlar için korunur.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return []

            frame_h, frame_w = bgr.shape[:2]
            # record_from_minimap gibi odometri güncellenir (F-0060).
            if update_odometry or not self.walk_trajectory:
                self.update_player_position(bgr)

            world_3d = self._detect_3d_world_gatherables(bgr)
            prompts = self._detect_active_harvest_prompt(bgr, world_3d_detections=world_3d)

            if material_type:
                if prompts:
                    for p in prompts:
                        p["resource_type"] = str(material_type)
                # İstem yokken oyuncu konumunda düğüm uydurulmaz (F-0215):
                # [F Gather] görülmediyse kayıt yapılmaz.

            if detections:
                for item in detections:
                    parsed = self._parse_material_detection(item, frame_w=frame_w, frame_h=frame_h)
                    if parsed is not None and parsed.get("source_type") == "harvest_auto":
                        prompts.append(parsed)

            if not prompts:
                return []

            return self._register_parsed_detections(
                bgr=bgr,
                parsed_items=prompts,
                route_manager=route_manager,
                region=region,
                route=route,
            )
        except Exception as exc:
            logger.error("record_harvest_event hatası: %s", exc)
            return []

    def ingest_detections(
        self,
        frame: np.ndarray,
        detections: Optional[List[Any]] = None,
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        update_odometry: bool = True,
        include_minimap: bool = True,
        include_3d: bool = True,
        include_harvest_prompt: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Her 3 otomatik haritalandırma kaynağını (`minimap_auto`, `3d_auto`, `harvest_auto`)
        ve varsa dışarıdan verilen `detections` listesini tek geçişte birleştirerek
        aktif rotaya `max_harvests = 3 * stack_count` (`current_harvests = 0`) ile kaydeder.

        LEGACY (H-sweep, dilim 13 Y5): üretimde çağrılmaz — canlı yol
        `process_frame_discoveries`'dir. Yalnızca testler / eski
        çağıranlar için korunur.
        """
        newly_discovered: List[Dict[str, Any]] = []
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return newly_discovered

            if update_odometry or not self.walk_trajectory:
                self.update_player_position(bgr)

            frame_h, frame_w = bgr.shape[:2]
            combined_parsed: List[Dict[str, Any]] = []

            # 1. Dışarıdan verilen tespitleri ayrıştır
            if detections:
                for item in detections:
                    parsed = self._parse_material_detection(item, frame_w=frame_w, frame_h=frame_h)
                    if parsed is not None:
                        combined_parsed.append(parsed)

            # 2. Sağ üst Minimap otomatik taraması (3 yapraklı filiz + Odyle + yeşil 2/3 rozetleri)
            if include_minimap:
                combined_parsed.extend(self._detect_minimap_gatherables(bgr))

            # 3. 3D Dünya beyaz toplama arması (Aria, Mela) ve yüzen [E] ikonu taraması
            world_3d: List[Dict[str, Any]] = []
            if include_3d:
                world_3d = self._detect_3d_world_gatherables(bgr)
                combined_parsed.extend(world_3d)

            # 4. Aktif [F Gather] toplama istemi taraması
            if include_harvest_prompt:
                combined_parsed.extend(
                    self._detect_active_harvest_prompt(bgr, world_3d_detections=world_3d)
                )

            if not combined_parsed:
                return newly_discovered

            return self._register_parsed_detections(
                bgr=bgr,
                parsed_items=combined_parsed,
                route_manager=route_manager,
                region=region,
                route=route,
            )
        except Exception as exc:
            logger.error("ingest_detections hatası: %s", exc)
            return newly_discovered

    def process_frame_discoveries(
        self,
        frame_bgr: np.ndarray,
        detected_materials: Optional[List[Any]] = None,
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        update_odometry: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        `MaterialScanner` tarafından ekranda veya minimap'te görülen her gem/materyal için:
        1. Oyuncunun anlık `(map_x, map_y)` konumuna materyalin göreli harita ofsetini
           (`rel_map_dx, rel_map_dy`) ekleyerek kaynağın gerçek harita koordinatını
           `(node_map_x, node_map_y)` hesaplar.
        2. `dedup_radius_px = 22.0` yarıçapı içinde daha önce kayıtlı bir node varsa:
           - Yeni node oluşturmaz; mevcut node'un koordinatını ağırlıklı ortalamayla
             (Kalman / EMA refinement) daha hassas hale getirir ve `sightings_count` artırır.
        3. `dedup_radius_px` içinde kayıtlı node yoksa:
           - Haritayı (`M`) açıp oyuncuyu durdurmadan, doğrudan sağ üst minimap üzerindeki
             ilgili konumdan 80x80 gri harita kesitini (`patch_img`) otomatik kırpar.
           - `route_manager.add_node` çağırarak yeni kaynağı `max_harvests=3` (veya yığın
             rozetine göre `6`/`9`) ile anında ilgili bölge ve rotaya kaydeder.
        """
        try:
            bgr = self._ensure_bgr(frame_bgr)
            if bgr is None:
                return []

            if self._inventory_recording_blocked(bgr):
                return []

            if update_odometry or not self.walk_trajectory:
                self.update_player_position(bgr)

            if not detected_materials:
                return []

            frame_h, frame_w = bgr.shape[:2]
            parsed_items: List[Dict[str, Any]] = []
            for item in detected_materials:
                parsed = self._parse_material_detection(item, frame_w=frame_w, frame_h=frame_h)
                if parsed is not None:
                    parsed_items.append(parsed)

            if not parsed_items:
                return []

            return self._register_parsed_detections(
                bgr=bgr,
                parsed_items=parsed_items,
                route_manager=route_manager,
                region=region,
                route=route,
            )
        except Exception as exc:
            logger.error("process_frame_discoveries hatası: %s", exc)
            return []

    def record_node(
        self,
        patch_or_frame: Optional[np.ndarray] = None,
        *args: Any,
        patch: Optional[np.ndarray] = None,
        frame: Optional[np.ndarray] = None,
        map_x: Optional[float] = None,
        map_y: Optional[float] = None,
        resource_type: Optional[str] = None,
        route_manager: Optional[RouteManager] = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        source_type: str = "manual_auto",
        exemplar_memory: Optional[Any] = None,
        in_air: bool = False,
        stack_count: int = 1,
        max_harvests: Optional[int] = None,
        # H-sweep (dilim 13 K3/K4): varsayılan güven 0.95'ti — ölçülmemiş
        # güven "neredeyse kesin" sayılıyordu. Varsayılan None'dur;
        # güven verilmezse kayıt yapılmaz.
        confidence: Optional[float] = None,
        class_id: Optional[int] = None,
        model_verdict: Optional[Tuple[int, float]] = None,
        **kwargs: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Directly inspects and records a candidate node patch or position:
        1. Inspects candidate patch against ExemplarMemory (and multi-signal gate):
           - Drops candidate patches matching negative exemplars immediately so false
             nodes never enter RouteManager or the map.
           - Tags candidate patches matching positive exemplars with their verified material class.
        2. Performs spatial deduplication and Kalman/EMA coordinate refinement if existing
           node within dedup_radius_px.
        3. Persists clean nodes to RouteManager and notifies radar/UI listeners.
        """
        try:
            target_patch = patch
            target_frame = frame
            if target_patch is None and patch_or_frame is not None:
                if isinstance(patch_or_frame, np.ndarray):
                    if patch_or_frame.ndim in (2, 3) and max(patch_or_frame.shape[:2]) <= 120:
                        target_patch = patch_or_frame
                    else:
                        target_frame = patch_or_frame

            node_map_x = float(map_x if map_x is not None else self.map_x)
            node_map_y = float(map_y if map_y is not None else self.map_y)

            if target_patch is None:
                if target_frame is not None:
                    target_patch = self.extract_clean_minimap_patch(
                        frame_bgr=target_frame,
                        rel_map_dx=0.0,
                        rel_map_dy=0.0,
                    )
                else:
                    # Patch/frame verilmeden düz gri sahte patch ile düğüm
                    # kaydedilmez (F-0216).
                    logger.info(
                        "AutoCartographer.record_node: patch/frame verilmedi, düğüm kaydedilmedi"
                    )
                    return None

            mem = exemplar_memory or self._get_exemplar_memory()

            # H-sweep (dilim 13 K3): güven verilmediyse kayıt yok —
            # 0.95 varsayımı kaldırıldı.
            if confidence is None:
                logger.info(
                    "AutoCartographer.record_node: güven verilmedi, düğüm kaydedilmedi"
                )
                return None

            # F-1389/F-1390: verilen dedektör sınıfını model_verdict olarak
            # kapıya ilet (class_id -> (class_id, confidence)); yoksa kapı
            # canavar sınıfını (11) bilemez.
            record_model_verdict = model_verdict
            if record_model_verdict is None and class_id is not None:
                try:
                    record_model_verdict = (int(class_id), float(confidence))
                except (TypeError, ValueError):
                    record_model_verdict = None

            gate = evaluate_node_candidate(
                target_patch,
                source_type=source_type,
                resource_type=resource_type or UNKNOWN_RESOURCE_TYPE,
                detector=self._get_quality_detector(),
                conf_min=MODEL_CONF_MIN,
                model_verdict=record_model_verdict,
                exemplar_memory=mem,
                check_feather=kwargs.get("check_feather", source_type.startswith("minimap")),
                check_monster=kwargs.get("check_monster", source_type.startswith("minimap")),
                check_exemplars=kwargs.get("check_exemplars", True),
                check_terrain=kwargs.get("check_terrain", False),
            )
            if gate.reject:
                self.total_gate_rejected += 1
                logger.info(
                    "AutoCartographer.record_node: candidate patch rejected [%s]: reasons=%s class=%s conf=%s",
                    source_type,
                    gate.reasons,
                    gate.class_id,
                    None if gate.confidence is None else round(float(gate.confidence), 3),
                )
                return None

            # v20-targena-3dauto (R1+R2): _register_parsed_detections
            # ile aynı kural — spesifik 3D adı jenerik doğrulamayla
            # ezilmez; 3D adayda ad belirlenemediyse kayıt yok.
            final_res, _drop_unnamed = _resolve_mapped_resource_name(
                resource_type, gate.verified_resource_type, source_type
            )
            if _drop_unnamed:
                logger.info(
                    "AutoCartographer.record_node: 3D aday adı belirlenemedi, düğüm kaydedilmedi"
                )
                return None
            final_res = str(final_res or UNKNOWN_RESOURCE_TYPE)
            # H-sweep (dilim 13 K4): güven yalnızca yukarı şişirilmez.
            final_conf = float(confidence)
            st_cnt = max(1, int(stack_count))
            max_h = max(DEFAULT_MAX_HARVESTS * st_cnt, int(max_harvests or (DEFAULT_MAX_HARVESTS * st_cnt)))

            rm = route_manager or self.route_manager
            target_region = str(region or (rm.active_region if rm else DEFAULT_REGION) or DEFAULT_REGION).strip()
            target_route = str(route or (rm.active_route if rm else DEFAULT_ROUTE) or DEFAULT_ROUTE).strip()

            # Kara liste (Blacklist) kontrolü: Engellenmiş koordinata kayıt yapılmaz
            try:
                from core.blacklist_manager import get_blacklist_manager
                bl_mgr = get_blacklist_manager()
                is_bl, _ = bl_mgr.is_blacklisted(target_region, node_map_x, node_map_y)
                if is_bl:
                    logger.debug(
                        "[AutoCartographer] Kara listede bulunan konum atlandı: %s (%.1f, %.1f)",
                        target_region,
                        node_map_x,
                        node_map_y,
                    )
                    return None
            except Exception as exc:
                # S-1102: fail-open YOK — kara liste kontrolü hata verirse
                # kayıt yapılmaz; hata yutulup kara listedeki konumun
                # kaydedilmesine izin verilmez.
                logger.warning(
                    "[AutoCartographer] Kara liste kontrolü başarısız, kayıt reddedildi: %s (%.1f, %.1f): %s",
                    target_region,
                    node_map_x,
                    node_map_y,
                    exc,
                )
                return None

            if rm is not None:
                self._sync_existing_nodes_from_route_manager(rm, region=target_region)

            existing_node, dist_px = self._find_nearest_tracked_node(
                region=target_region,
                node_map_x=node_map_x,
                node_map_y=node_map_y,
            )

            if existing_node is not None:
                self._refine_existing_node_kalman_ema(
                    tracked=existing_node,
                    obs_map_x=node_map_x,
                    obs_map_y=node_map_y,
                    obs_confidence=final_conf,
                    route_manager=rm,
                    route=target_route,
                    obs_stack_count=st_cnt,
                    obs_max_harvests=max_h,
                    obs_resource_type=final_res,
                )
                self._emit_node_updated(existing_node.to_dict())
                return existing_node.to_dict()

            norm_x = round(node_map_x / float(max(1, self.map_width)), 4)
            norm_y = round(node_map_y / float(max(1, self.map_height)), 4)
            routes_list = [target_route] if target_route and target_route != DEFAULT_ROUTE else []
            created_node: Optional[NodeData] = None

            if rm is not None:
                # Değerler add_node'a verilir; add_node içindeki save_region
                # diske bunlarla yazar (F-0002).
                created_node = rm.add_node(
                    region=target_region,
                    route=target_route,
                    patch_img=target_patch,
                    in_air=in_air,
                    map_x=int(round(node_map_x)),
                    map_y=int(round(node_map_y)),
                    norm_x=norm_x,
                    norm_y=norm_y,
                    resource_type=final_res,
                    max_harvests=int(max_h),
                    current_harvests=0,
                    source_type=str(source_type),
                )
                if created_node is not None:
                    setattr(created_node, "max_harvests", int(max_h))
                    setattr(created_node, "current_harvests", 0)
                    setattr(created_node, "stack_count", int(st_cnt))
                    setattr(created_node, "source_type", str(source_type))
                    rm.save_region(target_region)
                    self._save_direct_region_node_patch(
                        base_dir=rm.base_dir,
                        region=target_region,
                        node_id=created_node.id,
                        patch_img=target_patch,
                    )

            assigned_id = (
                int(created_node.id)
                if created_node is not None
                else max(
                    [t.node_id for t in self._tracked_nodes.get(target_region, [])],
                    default=0,
                )
                + 1
            )
            if created_node is not None:
                patch_path = created_node.patch_path
            else:
                # RouteManager yokken / add_node başarısızken patch diske
                # yazılmamıştır (S-0503): yazılmamış dosya yolu kaydedilmez;
                # patch _register_parsed_detections'taki gibi doğrudan
                # kaydedilir; yazım başarısızsa "" döner (F-0211).
                patch_path = self._save_direct_region_node_patch(
                    base_dir=rm.base_dir if rm is not None else "routes",
                    region=target_region,
                    node_id=assigned_id,
                    patch_img=target_patch,
                )

            tracked_new = TrackedResourceNode(
                node_id=assigned_id,
                region=target_region,
                routes=routes_list,
                resource_type=final_res,
                map_x=node_map_x,
                map_y=node_map_y,
                norm_x=norm_x,
                norm_y=norm_y,
                in_air=in_air,
                sightings_count=1,
                confidence=final_conf,
                patch_path=patch_path,
                patch_img=target_patch,
                stack_count=st_cnt,
                max_harvests=max_h,
                current_harvests=0,
                source_type=source_type,
            )
            self._tracked_nodes.setdefault(target_region, []).append(tracked_new)
            node_dict = tracked_new.to_dict()
            self._emit_node_discovered(node_dict)
            return node_dict
        except Exception as exc:
            logger.error("AutoCartographer.record_node error: %s", exc)
            return None

    def record_node_harvest(
        self,
        node_id: Optional[int] = None,
        region: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Bir kaynak düğümü toplandığında `current_harvests` sayacını 1 artırır.
        Her materyal 3 kez (`max_harvests = 3 * stack_count`) toplanabildiği için,
        düğüm ancak `current_harvests >= max_harvests` olduğunda tükenmiş (`is_depleted=True`)
        sayılır ve dinlenme (cooldown) süresine girer.
        """
        try:
            rm = self.route_manager
            target_region = str(
                region
                or (rm.active_region if rm is not None else DEFAULT_REGION)
                or DEFAULT_REGION
            ).strip()

            candidates = self._tracked_nodes.get(target_region, [])
            target_node: Optional[TrackedResourceNode] = None

            if node_id is not None:
                for t in candidates:
                    if t.node_id == int(node_id):
                        target_node = t
                        break
            else:
                target_node, _ = self._find_nearest_tracked_node(
                    region=target_region,
                    node_map_x=self.map_x,
                    node_map_y=self.map_y,
                )

            if target_node is None:
                return None

            if rm is not None:
                existing = rm.get_node(target_node.node_id, region=target_region)
                if existing is not None and bool(getattr(existing, "not_gatherable", False)):
                    logger.info(
                        "Düğüm #%d parşömen, hasat sayılmadı",
                        int(target_node.node_id),
                    )
                    return None

            now_h = time.time()
            # Dinlenmedeki düğüm süre dolmadan hasat sayılmaz (F-0061).
            if target_node.state == "resting":
                ready = False
                rm_node = rm.get_node(target_node.node_id, region=target_region) if rm is not None else None
                if rm_node is not None:
                    ready = bool(rm_node.is_ready(now=now_h))
                elif target_node.rest_started_timestamp > 0.0:
                    ready = (now_h - target_node.rest_started_timestamp) >= float(DEFAULT_REST_SEC)
                else:
                    ready = True
                if not ready:
                    return target_node.to_dict()
                target_node.state = "active"
                target_node.current_harvests = 0
                target_node.rest_started_timestamp = 0.0
                if rm_node is not None:
                    rm_node.state = "active"
                    rm_node.current_harvests = 0

            target_node.current_harvests = min(
                target_node.max_harvests,
                target_node.current_harvests + 1,
            )
            target_node.last_seen_timestamp = now_h

            # Tükenen düğüm sonsuza dek tükenmiş kalmaz (F-0061): dinlenmeye
            # girer (state="resting"); süre dolunca yukarıdaki uyandırma
            # yoluyla yeniden toplanabilir olur. Sayaç dinlenme boyunca
            # max_harvests'ta KALIR (sıfırlanmaz): tükenmiş düğüm
            # current_harvests=max, remaining_harvests=0, is_depleted=True
            # olarak gözlenir. Sıfırlama yalnızca uyandırmada yapılır
            # (yukarıdaki resting dalı): D10'un tükenme anındaki
            # sıfırlaması sayacı gözlenemez yapıyordu (6/6 -> 0/6) ve
            # remaining_harvests tutarlılığı için property'de
            # resting kısa devresi gerektiriyordu.
            if target_node.is_depleted:
                target_node.state = "resting"
                target_node.rest_started_timestamp = now_h

            if rm is not None:
                node_obj = rm.get_node(target_node.node_id, region=target_region)
                if node_obj is not None:
                    setattr(node_obj, "current_harvests", int(target_node.current_harvests))
                    setattr(node_obj, "max_harvests", int(target_node.max_harvests))
                    node_obj.gathered_count += 1
                    if target_node.state == "resting":
                        node_obj.state = "resting"
                        node_obj.last_visited_timestamp = now_h
                        node_obj.consecutive_failures = 0
                    rm.save_region(target_region)

            return target_node.to_dict()
        except Exception as exc:
            logger.error("record_node_harvest hatası: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Durum ve İstatistik Sorgulama Metodları
    # ------------------------------------------------------------------
    def get_cartographer_health_summary(self) -> dict:
        """Gerçek durum kontrollerine dayanan sağlık özeti (F-0217; totoloji değil)."""
        try:
            stats = self.get_cartography_stats()
            checks = {
                "odometry_tracker": self.odometry_tracker is not None,
                "map_dimensions_valid": int(self.map_width) >= 160
                and int(self.map_height) >= 120,
                "dedup_radius_valid": DEDUP_RADIUS_MIN_PX
                <= float(self.dedup_radius_px)
                <= DEDUP_RADIUS_MAX_PX,
                "scale_valid": float(self.minimap_to_map_scale) > 0.0,
                "stats_ok": isinstance(stats, dict) and "total_nodes" in stats,
            }
            ready = bool(all(checks.values()))
            return {
                "cartographer_ready": ready,
                "checks": checks,
                "calibrated": bool(self._is_calibrated_on_world_map),
                "needs_map_calibrate": bool(self.needs_map_calibrate),
                "tracked_nodes": int(stats.get("total_nodes", 0)),
                # H-sweep (dilim 13 Y7): drift ölçülmüyor — 0.0 raporlamak
                # yalan olur; None + bayrak döndürülür.
                "cumulative_drift_px": None,
                "drift_not_measured": True,
            }
        except Exception as exc:
            return {"cartographer_ready": False, "error": str(exc)}

    def get_cartography_stats(self, region: Optional[str] = None) -> Dict[str, Any]:
        """
        HUD ve Cartography Radar için haritalandırma istatistiklerini döndürür.
        `total_minimap_recorded`, `total_3d_recorded`,
        `total_harvest_recorded` ve `total_radar_recorded` sayaçlarını içerir.
        """
        try:
            nodes = self.get_tracked_nodes(region=region)
            target_region = str(region or self.active_region or DEFAULT_REGION)
            rg = self.get_route_graph(target_region)
            # Üç ayrı sum() yerine tek geçiş (D02); toplamlar tam sayıdır,
            # sıralama sonucu değiştirmez.
            total_max_harvests = 0
            total_current_harvests = 0
            total_available_harvests = 0
            for n in nodes:
                total_max_harvests += int(n.get("max_harvests", DEFAULT_MAX_HARVESTS))
                total_current_harvests += int(n.get("current_harvests", 0))
                total_available_harvests += int(n.get("remaining_harvests", DEFAULT_MAX_HARVESTS))
            return {
                "total_nodes": len(nodes),
                "auto_mapped": int(self.auto_mapped_count),
                "last_mapped": dict(self.last_mapped_info) if self.last_mapped_info else None,
                "auto_cartography_mode": bool(self.auto_cartography_enabled),
                "graph_nodes_count": len(nodes),
                "graph_waypoints_count": len(rg.waypoints),
                "graph_edges_count": len(rg.edges),
                "graph_total_distance_px": round(float(rg.total_distance), 2),
                "total_minimap_recorded": int(self.total_minimap_recorded),
                "total_3d_recorded": int(self.total_3d_recorded),
                "total_harvest_recorded": int(self.total_harvest_recorded),
                "total_radar_recorded": int(self.total_radar_recorded),
                "total_max_harvests": int(total_max_harvests),
                "total_current_harvests": int(total_current_harvests),
                "total_available_harvests": int(total_available_harvests),
                "player_map_x": round(float(self.map_x), 3),
                "player_map_y": round(float(self.map_y), 3),
                "player_norm_x": float(self.norm_x),
                "player_norm_y": float(self.norm_y),
                "walk_trajectory_points": len(self.walk_trajectory),
                "is_calibrated_on_world_map": bool(self._is_calibrated_on_world_map),
                # H-sweep (dilim 13 Y7): drift ölçülmüyor (bkz. readiness).
                "cumulative_drift_px": None,
                "drift_not_measured": True,
            }
        except Exception as exc:
            logger.error("get_cartography_stats hatası: %s", exc)
            return {
                "total_nodes": 0,
                "auto_mapped": int(self.auto_mapped_count),
                "last_mapped": dict(self.last_mapped_info) if self.last_mapped_info else None,
                "auto_cartography_mode": bool(self.auto_cartography_enabled),
                "graph_nodes_count": 0,
                "graph_waypoints_count": 0,
                "graph_edges_count": 0,
                "graph_total_distance_px": 0.0,
                "total_minimap_recorded": int(self.total_minimap_recorded),
                "total_3d_recorded": int(self.total_3d_recorded),
                "total_harvest_recorded": int(self.total_harvest_recorded),
                "total_radar_recorded": int(self.total_radar_recorded),
                "total_max_harvests": 0,
                "total_current_harvests": 0,
                "total_available_harvests": 0,
                "player_map_x": round(float(self.map_x), 3),
                "player_map_y": round(float(self.map_y), 3),
                "player_norm_x": float(self.norm_x),
                "player_norm_y": float(self.norm_y),
                "walk_trajectory_points": len(self.walk_trajectory),
                "is_calibrated_on_world_map": bool(self._is_calibrated_on_world_map),
                "cumulative_drift_px": None,
                "drift_not_measured": True,
            }

    def get_tracked_nodes(self, region: Optional[str] = None) -> List[Dict[str, Any]]:
        """Takip edilen tekilleştirilmiş tüm düğümlerin sözlük listesini döndürür."""
        try:
            if region is not None:
                return [n.to_dict() for n in self._tracked_nodes.get(region, [])]
            result: List[Dict[str, Any]] = []
            for reg_nodes in self._tracked_nodes.values():
                for n in reg_nodes:
                    result.append(n.to_dict())
            return result
        except Exception as exc:
            logger.error("get_tracked_nodes hatası: %s", exc)
            return []

    def get_walk_trajectory(self) -> List[Tuple[float, float, float]]:
        """Oyuncunun yürüyüş izini `list[(map_x, map_y, timestamp)]` olarak döndürür."""
        return list(self.walk_trajectory)

    def reset_odometry(
        self,
        map_x: float = 960.0,
        map_y: float = 540.0,
        clear_trajectory: bool = False,
    ) -> None:
        """Görsel odometri referansını ve isteğe bağlı olarak yürüyüş izini sıfırlar."""
        self.map_x = float(map_x)
        self.map_y = float(map_y)
        self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
        self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
        self._prev_minimap_clean = None
        self._cumulative_drift_px = 0.0
        # Kalibre bayrağı ve hız da sıfırlanır (F-0219): sıfırlanmış odometri
        # kalibre sayılmaz.
        self._is_calibrated_on_world_map = False
        self._last_calibration_ts = 0.0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.dead_reckoning_dx = 0.0
        self.dead_reckoning_dy = 0.0
        self.odometry_tracker.reset(
            map_x=self.map_x,
            map_y=self.map_y,
            clear_trajectory=clear_trajectory,
        )
        self._refresh_calibration_flag()

    def get_auto_cartographer_ready_flag(self) -> dict:
        """Returns the auto cartographer ready flag (gerçek sağlık özetinden, F-0217)."""
        return {
            "module": "auto_cartographer",
            "ready": bool(self.get_cartographer_health_summary().get("cartographer_ready")),
        }

    def get_auto_cartographer_version_info(self) -> dict:
        return {
            "module": "auto_cartographer",
            "version": __version__,
            "ready": bool(self.get_cartographer_health_summary().get("cartographer_ready")),
        }

    def get_auto_cartographer_diagnostic_snapshot(self) -> dict:
        health = self.get_cartographer_health_summary()
        ready = bool(health.get("cartographer_ready"))
        return {
            "module": "auto_cartographer",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "health": health,
        }


def get_auto_cartographer_capabilities() -> list:
    return [
        "auto_cartographer",
        "minimap_mapping",
        "diagnostic_snapshot",
        "autonomous_route_graph",
        "spatial_clustering_15_25px",
        "kalman_ema_smoothing",
        "exemplar_learning_filter",
    ]


def validate_auto_cartographer_contract() -> bool:
    # Totoloji değil (F-0217): yetenek listesi + değişmez (invariant) kontrolleri.
    return bool(get_auto_cartographer_capabilities()) and bool(
        DEDUP_RADIUS_MIN_PX <= DEFAULT_DEDUP_RADIUS_PX <= DEDUP_RADIUS_MAX_PX
        and DEFAULT_MINIMAP_TO_MAP_SCALE > 0.0
        and 0.0 < MINIMAP_SPROUT_CONF_MIN <= 1.0
    )


def get_auto_cartographer_baseline_metrics() -> dict:
    return {"module": "auto_cartographer", "contract_valid": validate_auto_cartographer_contract(), "capability_count": len(get_auto_cartographer_capabilities())}


def get_auto_cartographer_readiness_status() -> dict:
    return {
        "module": "auto_cartographer",
        "ready": validate_auto_cartographer_contract(),
        "metrics": get_auto_cartographer_baseline_metrics(),
    }


def get_auto_cartographer_audit_record() -> dict:
    return {
        "module": "auto_cartographer",
        "audit_passed": validate_auto_cartographer_contract(),
        "readiness": get_auto_cartographer_readiness_status(),
    }


def get_auto_cartographer_operational_digest() -> dict:
    return {
        "module": "auto_cartographer",
        "operational": validate_auto_cartographer_contract(),
        "audit": get_auto_cartographer_audit_record(),
    }


def get_auto_cartographer_telemetry_payload() -> dict:
    return {
        "module": "auto_cartographer",
        "telemetry_ready": validate_auto_cartographer_contract(),
        "digest": get_auto_cartographer_operational_digest(),
    }


def get_auto_cartographer_integrity_seal() -> dict:
    return {
        "module": "auto_cartographer",
        "sealed": validate_auto_cartographer_contract(),
        "telemetry": get_auto_cartographer_telemetry_payload(),
    }


def get_auto_cartographer_compliance_certificate() -> dict:
    return {
        "module": "auto_cartographer",
        "certified": validate_auto_cartographer_contract(),
        "seal": get_auto_cartographer_integrity_seal(),
    }


def get_auto_cartographer_readiness_probe() -> dict:
    return {
        "module": "auto_cartographer",
        "ready": validate_auto_cartographer_contract(),
        "certificate": get_auto_cartographer_compliance_certificate(),
    }


def get_auto_cartographer_audit_summary() -> dict:
    return {
        "module": "auto_cartographer",
        "audit_passed": validate_auto_cartographer_contract(),
        "probe": get_auto_cartographer_readiness_probe(),
    }


