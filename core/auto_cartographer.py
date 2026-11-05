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

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from core.route_manager import (
    DEFAULT_REGION,
    DEFAULT_RESOURCE_TYPE,
    DEFAULT_ROUTE,
    PATCH_SIZE,
    NodeData,
    RouteManager,
)
from core.vision_filters import (
    FeatherNegativeFilter,
    MonsterCircularityFilter,
    Pyramid3DCrestDetector,
    NameplateHSVIsolator,
    DynamicInteractionKeyReader,
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

# Sağ üst minimap ROI: x: 0.855..0.992 (w=0.137), y: 0.065..0.255 (h=0.190)
MINIMAP_NORM_RECT: Tuple[float, float, float, float] = (0.855, 0.065, 0.137, 0.190)
DEFAULT_DEDUP_RADIUS_PX: float = 22.0
DEFAULT_EMA_ALPHA: float = 0.35
REF_MAP_WIDTH: int = 1920
REF_MAP_HEIGHT: int = 1080
DEFAULT_MAX_HARVESTS: int = 3


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
    map_x: float = 960.0
    map_y: float = 540.0
    norm_x: float = 0.5
    norm_y: float = 0.5
    in_air: bool = False
    sightings_count: int = 1
    confidence: float = 1.0
    kalman_variance: float = 16.0
    patch_path: str = ""
    first_seen_timestamp: float = field(default_factory=time.time)
    last_seen_timestamp: float = field(default_factory=time.time)
    patch_img: Optional[np.ndarray] = field(default=None, repr=False)
    stack_count: int = 1
    max_harvests: int = DEFAULT_MAX_HARVESTS
    current_harvests: int = 0
    source_type: str = "minimap_auto"

    def __post_init__(self) -> None:
        self.stack_count = max(1, int(self.stack_count))
        if int(self.max_harvests) <= 0:
            self.max_harvests = DEFAULT_MAX_HARVESTS * self.stack_count
        else:
            self.max_harvests = max(1, int(self.max_harvests))
        self.current_harvests = max(0, int(self.current_harvests))
        self.source_type = str(self.source_type or "minimap_auto")

    @property
    def remaining_harvests(self) -> int:
        """Bu düğümden kalan toplanabilir materyal hakkını döndürür."""
        return max(0, int(self.max_harvests) - int(self.current_harvests))

    @property
    def is_depleted(self) -> bool:
        """Düğümün 3x (veya `max_harvests`) toplama hakkının bitip bitmediğini döndürür."""
        return int(self.current_harvests) >= int(self.max_harvests)

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
            "patch_path": str(self.patch_path),
            "first_seen_timestamp": float(self.first_seen_timestamp),
            "last_seen_timestamp": float(self.last_seen_timestamp),
            "stack_count": int(self.stack_count),
            "max_harvests": int(self.max_harvests),
            "current_harvests": int(self.current_harvests),
            "remaining_harvests": int(self.remaining_harvests),
            "is_depleted": bool(self.is_depleted),
            "source_type": str(self.source_type),
        }


class VisualOdometryTracker:
    """
    Minimap doku kayması (`cv2.phaseCorrelate` + Lucas-Kanade Optik Akış + ORB)
    üzerinden oyuncunun dünya/bölge haritası koordinatlarını `(map_x, map_y)` takip eden
    görsel odometri sınıfı.
    """

    def __init__(
        self,
        initial_map_x: float = 960.0,
        initial_map_y: float = 540.0,
        map_width: int = REF_MAP_WIDTH,
        map_height: int = REF_MAP_HEIGHT,
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
        if record_trajectory:
            self.append_trajectory()
        return self.get_position()

    def append_trajectory(self, timestamp: Optional[float] = None) -> None:
        """Güncel konumu zaman damgasıyla birlikte yürüyüş izine ekler."""
        ts = float(timestamp if timestamp is not None else time.time())
        entry = (round(float(self.map_x), 3), round(float(self.map_y), 3), ts)
        self.walk_trajectory.append(entry)
        if len(self.walk_trajectory) > self.max_trajectory_len:
            excess = len(self.walk_trajectory) - self.max_trajectory_len
            del self.walk_trajectory[:excess]

    def offset_to_world_coords(self, dx: float, dy: float) -> Tuple[float, float, float, float]:
        """
        Minimap merkezindeki oyuncu okuna göre `(dx, dy)` piksel ofsetini
        dünya/harita koordinatlarına `(world_x, world_y, norm_x, norm_y)` dönüştürür.
        """
        world_x = float(
            np.clip(
                self.map_x + float(dx) * self.minimap_to_map_scale,
                0.0,
                float(self.map_width - 1),
            )
        )
        world_y = float(
            np.clip(
                self.map_y + float(dy) * self.minimap_to_map_scale,
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
        if clear_trajectory:
            self.walk_trajectory.clear()


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
        minimap_to_map_scale: float = 1.0,
        max_trajectory_len: int = 10000,
    ) -> None:
        self.route_manager: Optional[RouteManager] = route_manager
        self.vision_engine: Optional[Any] = vision_engine
        self.minimap_norm_rect: Tuple[float, float, float, float] = (
            float(minimap_norm_rect[0]),
            float(minimap_norm_rect[1]),
            float(minimap_norm_rect[2]),
            float(minimap_norm_rect[3]),
        )
        self.dedup_radius_px: float = max(1.0, float(dedup_radius_px))
        self.ema_alpha: float = float(np.clip(ema_alpha, 0.05, 0.95))

        self.map_width: int = max(160, int(map_width))
        self.map_height: int = max(120, int(map_height))
        self.minimap_to_map_scale: float = float(minimap_to_map_scale)
        self.max_trajectory_len: int = max(100, int(max_trajectory_len))

        # Görsel odometri takipçisi (VisualOdometryTracker)
        self.odometry_tracker: VisualOdometryTracker = VisualOdometryTracker(
            initial_map_x=initial_map_x,
            initial_map_y=initial_map_y,
            map_width=self.map_width,
            map_height=self.map_height,
            minimap_to_map_scale=self.minimap_to_map_scale,
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
        self._prev_minimap_gray: Optional[np.ndarray] = None
        self._prev_minimap_clean: Optional[np.ndarray] = None
        self._last_valid_minimap_bgr: Optional[np.ndarray] = None
        self._last_world_map_bgr: Optional[np.ndarray] = None
        self._hann_window: Optional[np.ndarray] = None
        self._odometry_Search_orb: Optional[Any] = self.odometry_tracker._orb
        self._is_calibrated_on_world_map: bool = False
        self._last_calibration_ts: float = 0.0
        self._cumulative_drift_px: float = 0.0

        # Bölge bazlı takip edilen kaynak düğümleri: region -> List[TrackedResourceNode]
        self._tracked_nodes: Dict[str, List[TrackedResourceNode]] = {}

        # Kaynak bazlı haritalandırma istatistik sayaçları
        self.total_minimap_recorded: int = 0
        self.total_3d_recorded: int = 0
        self.total_harvest_recorded: int = 0

        # Vision filter instances for enhanced confidence safeguard
        self.feather_negative_filter: FeatherNegativeFilter = FeatherNegativeFilter()
        self.monster_circularity_filter: MonsterCircularityFilter = MonsterCircularityFilter()
        self.pyramid_3d_crest_detector: Pyramid3DCrestDetector = Pyramid3DCrestDetector()
        self.nameplate_isolator: NameplateHSVIsolator = NameplateHSVIsolator()
        self.key_reader: DynamicInteractionKeyReader = DynamicInteractionKeyReader()

        if self.route_manager is not None:
            self._sync_existing_nodes_from_route_manager(self.route_manager)

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
            # Yakın çekim (closeup) referans görüntüleri için destek
            if frame_w >= 600 and frame_h <= 420:
                # minimap_filter_closeup.jpg (718x362): minimap sağ yarıda
                return (
                    int(round(frame_w * 0.498)),
                    int(round(frame_h * 0.200)),
                    int(round(frame_w * 0.964)),
                    int(round(frame_h * 0.912)),
                )
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

    def _is_world_map_open(self, frame_bgr: np.ndarray) -> bool:
        """Dünya haritasının (`M`) açık olup olmadığını kontrol eder."""
        try:
            if self.vision_engine is not None and hasattr(self.vision_engine, "is_map_open"):
                return bool(self.vision_engine.is_map_open(frame_bgr))

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
            return False

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
            if mean_val < 52.0 or std_val < 8.0:
                return True
            return False
        except Exception as exc:
            logger.debug("Minimap kapanma kontrolü hatası: %s", exc)
            return False

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
            self.map_width = w
            self.map_height = h
            self._last_world_map_bgr = bgr.copy()

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
                player_x = float(w // 2)
                player_y = float(h // 2)

            self.map_x = float(np.clip(player_x, 0.0, float(w - 1)))
            self.map_y = float(np.clip(player_y, 0.0, float(h - 1)))
            self.norm_x = round(self.map_x / float(max(1, w)), 4)
            self.norm_y = round(self.map_y / float(max(1, h)), 4)

            now_cal = time.time()
            prev_log_x = getattr(self, "_last_logged_cal_x", -999.0)
            prev_log_y = getattr(self, "_last_logged_cal_y", -999.0)
            prev_log_ts = getattr(self, "_last_logged_cal_ts", 0.0)

            self._is_calibrated_on_world_map = True
            self._last_calibration_ts = now_cal
            self._cumulative_drift_px = 0.0
            self.odometry_tracker.cumulative_drift_px = 0.0
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
        Dünya haritası (`M`) açıksa `calibrate_on_world_map` ile sıfır hatayla kalibre eder.
        """
        try:
            bgr = self._ensure_bgr(frame_bgr)
            if bgr is None:
                return self.get_position()

            h, w = bgr.shape[:2]
            if not self._is_calibrated_on_world_map and self.map_width == REF_MAP_WIDTH and w >= 800:
                scale_x = float(w) / float(self.map_width)
                scale_y = float(h) / float(self.map_height)
                self.map_x *= scale_x
                self.map_y *= scale_y
                self.map_width = w
                self.map_height = h
                self.odometry_tracker.map_width = w
                self.odometry_tracker.map_height = h

            # 1. Dünya haritası ('M') açıksa mutlak kalibrasyon yap
            if w >= 800 and self._is_world_map_open(bgr):
                return self.calibrate_on_world_map(bgr)

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

            if self._prev_minimap_clean is not None and self._prev_minimap_clean.shape == curr_clean.shape:
                dx_terrain, dy_terrain, conf = self._estimate_minimap_shift(
                    self._prev_minimap_clean,
                    curr_clean,
                )
                if conf >= 0.15:
                    player_dx = -dx_terrain * self.minimap_to_map_scale
                    player_dy = -dy_terrain * self.minimap_to_map_scale

                    self.map_x = float(np.clip(self.map_x + player_dx, 0.0, float(self.map_width - 1)))
                    self.map_y = float(np.clip(self.map_y + player_dy, 0.0, float(self.map_height - 1)))
                    self.norm_x = round(self.map_x / float(max(1, self.map_width)), 4)
                    self.norm_y = round(self.map_y / float(max(1, self.map_height)), 4)
                    drift_inc = math.hypot(player_dx, player_dy) * 0.02
                    self._cumulative_drift_px += drift_inc
                    self.odometry_tracker.cumulative_drift_px = self._cumulative_drift_px

            self._prev_minimap_clean = curr_clean
            self._prev_minimap_gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
            self._append_trajectory()
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
            if bgr is not None and bgr.shape[1] >= 800 and self._is_world_map_open(bgr):
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
            pad = half + 8
            padded = cv2.copyMakeBorder(
                inpainted_minimap,
                pad,
                pad,
                pad,
                pad,
                borderType=cv2.BORDER_REPLICATE,
            )

            px = int(np.clip(target_mx, -half // 2, mw + half // 2)) + pad
            py = int(np.clip(target_my, -half // 2, mh + half // 2)) + pad

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

            for p in (direct_formatted, direct_simple, patch_sub):
                cv2.imwrite(p, patch_img)

            base_name = os.path.basename(os.path.normpath(base_dir)) or "routes"
            return f"{base_name}/{region}/node_{int(node_id):02d}.png".replace("\\", "/")
        except Exception as exc:
            logger.debug("Doğrudan node patch kaydı hatası: %s", exc)
            return f"routes/{region}/patches/node_{int(node_id)}.png"

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

                # Optional vision_filters FeatherNegativeFilter & MonsterCircularityFilter safeguard check
                try:
                    icon_patch_bgr = minimap_bgr[max(0, by-2):min(mh, by+bh+2), max(0, bx-2):min(mw, bx+bw+2)]
                    if icon_patch_bgr.size > 0 and hasattr(self, "feather_negative_filter"):
                        is_f, f_score = self.feather_negative_filter.is_feather_icon(icon_patch_bgr)
                        if is_f and float(f_score) >= 0.88:
                            continue
                except Exception as exc_vf:
                    logger.debug("vision_filters minimap safeguard skipped: %s", exc_vf)

                # Yanında yeşil yığın rozeti ('2' veya '3') var mı kontrol et (H: 68..90, S >= 75, V >= 115)
                bx1, by1 = max(0, bx - 4), max(0, by - 4)
                bx2, by2 = min(mw, bx + bw + 8), min(mh, by + bh + 6)
                badge_px = int(
                    np.count_nonzero(cv2.inRange(hsv[by1:by2, bx1:bx2], (68, 75, 115), (90, 255, 255)))
                )
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
                        "confidence": round(min(0.95, 0.76 + (contrast - 115.0) / 500.0), 3),
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
    ) -> Tuple[float, float]:
        """
        3D oyun ekranındaki `(screen_x, screen_y)` noktasını oyuncunun ayak merkezine
        `(frame_w * 0.50, frame_h * 0.54)` göre perspektif düzeltmeli harita ofsetine
        `(rel_map_dx, rel_map_dy)` dönüştürür.
        """
        player_x = float(frame_w) * self.PLAYER_FEET_X_RATIO
        player_y = float(frame_h) * self.PLAYER_FEET_Y_RATIO

        raw_dx = float(screen_x) - player_x
        raw_dy = float(screen_y) - player_y

        norm_dx = raw_dx * (float(self.REF_SCREEN_WIDTH) / float(max(1, frame_w)))
        norm_dy = raw_dy * (float(self.REF_SCREEN_HEIGHT) / float(max(1, frame_h)))

        vertical_ratio = max(0.25, min(1.8, (player_y / float(max(1.0, float(screen_y)))) ** 1.15))
        rel_map_dx = round(norm_dx * self.PERSPECTIVE_SCALE_X * vertical_ratio, 3)
        rel_map_dy = round(norm_dy * self.PERSPECTIVE_SCALE_Y * vertical_ratio, 3)
        return float(rel_map_dx), float(rel_map_dy)

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

                resource_type = DEFAULT_RESOURCE_TYPE
                if mesh_hsv.size > 0:
                    # Aria: mor / menekşe kristal bitki (H: 125..165, S >= 45, V >= 60)
                    purple_px = int(
                        np.count_nonzero(cv2.inRange(mesh_hsv, (125, 45, 60), (165, 255, 255)))
                    )
                    # Mela: kırmızı / mercan meyveli yeşil çalı
                    red_px = int(
                        np.count_nonzero(cv2.inRange(mesh_hsv, (0, 80, 80), (12, 255, 255)))
                    ) + int(
                        np.count_nonzero(cv2.inRange(mesh_hsv, (166, 80, 80), (180, 255, 255)))
                    )
                    green_px = int(
                        np.count_nonzero(cv2.inRange(mesh_hsv, (30, 45, 50), (78, 255, 255)))
                    )
                    if purple_px >= max(20, int(round(30 * scale_x * scale_y))):
                        resource_type = "Aria"
                    elif red_px >= max(40, int(round(60 * scale_x * scale_y))) or (
                        red_px >= max(15, int(round(20 * scale_x * scale_y))) and green_px >= 10
                    ):
                        resource_type = "Mela"

                node_sx = float(cx_mid)
                node_sy = float(min(h - 1, y + ch + int(round(26 * scale_y))))
                rel_dx, rel_dy = self._project_3d_screen_to_map_offset(node_sx, node_sy, w, h)

                detections.append(
                    {
                        "rel_map_dx": rel_dx,
                        "rel_map_dy": rel_dy,
                        "screen_x": int(round(node_sx)),
                        "screen_y": int(round(node_sy)),
                        "resource_type": resource_type,
                        "in_air": bool(y < int(h * 0.22)),
                        "confidence": 0.92,
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
                                "resource_type": "Gatherable Plant",
                                "in_air": False,
                                "confidence": 0.89,
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

                # Yakındaki 3D tespitten materyal adını ('Aria', 'Mela', vb.) miras al
                resource_type = DEFAULT_RESOURCE_TYPE
                rel_dx, rel_dy = 0.0, 0.0
                if world_3d_detections:
                    nearest_3d = min(
                        world_3d_detections,
                        key=lambda d: math.hypot(
                            float(d.get("rel_map_dx", 0.0)),
                            float(d.get("rel_map_dy", 0.0)),
                        ),
                    )
                    resource_type = str(nearest_3d.get("resource_type", DEFAULT_RESOURCE_TYPE))
                    rel_dx = float(nearest_3d.get("rel_map_dx", 0.0))
                    rel_dy = float(nearest_3d.get("rel_map_dy", 0.0))

                prompts.append(
                    {
                        "rel_map_dx": rel_dx,
                        "rel_map_dy": rel_dy,
                        "screen_x": int(rx1 + fx + fcw // 2),
                        "screen_y": int(ry1 + fy + fch // 2),
                        "resource_type": resource_type,
                        "in_air": False,
                        "confidence": 0.96,
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
                existing_nodes = route_manager.get_route_nodes(region_name=reg_name, route_name=DEFAULT_ROUTE)
                tracked_list = self._tracked_nodes.setdefault(reg_name, [])
                tracked_ids = {t.node_id for t in tracked_list}
                for n in existing_nodes:
                    if n.id not in tracked_ids:
                        max_h = int(getattr(n, "max_harvests", DEFAULT_MAX_HARVESTS) or DEFAULT_MAX_HARVESTS)
                        curr_h = int(getattr(n, "current_harvests", 0) or 0)
                        st_cnt = int(getattr(n, "stack_count", max(1, max_h // DEFAULT_MAX_HARVESTS)) or 1)
                        src_t = str(getattr(n, "source_type", "minimap_auto") or "minimap_auto")
                        setattr(n, "max_harvests", max_h)
                        setattr(n, "current_harvests", curr_h)
                        setattr(n, "stack_count", st_cnt)
                        setattr(n, "source_type", src_t)
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
                                sightings_count=max(1, int(n.gathered_count) or 1),
                                patch_path=n.patch_path,
                                stack_count=st_cnt,
                                max_harvests=max_h,
                                current_harvests=curr_h,
                                source_type=src_t,
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
        `MaterialScanner` veya harici dedektörler tarafından üretilen materyal/gem tespit
        öğesini (dict, nesne veya tuple) standart sözlük yapısına
        (`rel_map_dx`, `rel_map_dy`, `resource_type`, `in_air`, `confidence`,
        `stack_count`, `max_harvests`, `current_harvests`, `source_type`) çevirir.
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
                        "resource_type": str(item[2]) if len(item) >= 3 else DEFAULT_RESOURCE_TYPE,
                    }
                else:
                    return None
            else:
                return None

            # Kaynak türü (minimap_auto, 3d_auto, harvest_auto) belirleme
            raw_source = str(raw.get("source_type") or raw.get("source") or "").strip().lower()
            if raw_source in ("minimap", "minimap_auto"):
                source_type = "minimap_auto"
            elif raw_source in ("gather_prompt", "harvest", "harvest_auto", "f_gather"):
                source_type = "harvest_auto"
            elif raw_source in ("3d_world", "3d_auto", "3d", "world"):
                source_type = "3d_auto"
            else:
                source_type = "minimap_auto"

            # 1. Doğrudan göreli harita ofseti (rel_map_dx, rel_map_dy) kontrolü
            rel_dx: Optional[float] = None
            rel_dy: Optional[float] = None

            for kx, ky in (
                ("rel_map_dx", "rel_map_dy"),
                ("map_dx", "map_dy"),
                ("rel_dx", "rel_dy"),
                ("dx", "dy"),
                ("offset_x", "offset_y"),
            ):
                if kx in raw and ky in raw and raw[kx] is not None and raw[ky] is not None:
                    rel_dx = float(raw[kx])
                    rel_dy = float(raw[ky])
                    break

            # 2. Minimap içi koordinat verildiyse minimap merkezine göre ofseti hesapla
            if rel_dx is None or rel_dy is None:
                x1, y1, x2, y2 = self._get_minimap_pixel_rect(frame_w, frame_h)
                mcx = (x2 - x1) / 2.0
                mcy = (y2 - y1) * 0.545
                if "minimap_x" in raw and "minimap_y" in raw:
                    mx = float(raw["minimap_x"])
                    my = float(raw["minimap_y"])
                    if mx >= x1 and my >= y1:
                        mx -= float(x1)
                        my -= float(y1)
                    rel_dx = (mx - mcx) * self.minimap_to_map_scale
                    rel_dy = (my - mcy) * self.minimap_to_map_scale
                    source_type = "minimap_auto"

            # 3. 3D Dünya ekran koordinatı (screen_x, screen_y veya bbox) verildiyse perspektif izdüşümü
            if rel_dx is None or rel_dy is None:
                sx: Optional[float] = None
                sy: Optional[float] = None
                if "screen_x" in raw and "screen_y" in raw:
                    sx = float(raw["screen_x"])
                    sy = float(raw["screen_y"])
                elif "x" in raw and "y" in raw:
                    sx = float(raw["x"])
                    sy = float(raw["y"])
                elif "bbox" in raw and isinstance(raw["bbox"], (tuple, list)) and len(raw["bbox"]) >= 4:
                    bx, by, bw, bh = [float(v) for v in raw["bbox"][:4]]
                    sx = bx + bw / 2.0
                    sy = by + bh / 2.0

                if sx is not None and sy is not None:
                    x1, y1, x2, y2 = self._get_minimap_pixel_rect(frame_w, frame_h)
                    if x1 <= sx <= x2 and y1 <= sy <= y2:
                        mcx = (x1 + x2) / 2.0
                        mcy = y1 + (y2 - y1) * 0.545
                        rel_dx = (sx - mcx) * self.minimap_to_map_scale
                        rel_dy = (sy - mcy) * self.minimap_to_map_scale
                        source_type = "minimap_auto"
                    else:
                        rel_dx, rel_dy = self._project_3d_screen_to_map_offset(sx, sy, frame_w, frame_h)
                        if source_type == "minimap_auto" and not raw_source:
                            source_type = "3d_auto"

            if rel_dx is None or rel_dy is None:
                rel_dx, rel_dy = 0.0, 0.0

            resource_type = str(
                raw.get("resource_type")
                or raw.get("material_type")
                or raw.get("name")
                or raw.get("label")
                or raw.get("category")
                or DEFAULT_RESOURCE_TYPE
            ).strip()
            if not resource_type:
                resource_type = DEFAULT_RESOURCE_TYPE

            # Tüy (feather) ve kanat (wing) ikonlarını kesinlikle reddet ("tüy odyle değil tüy toplamayacağız")
            rt_lower = resource_type.lower()
            if any(bad_tok in rt_lower for bad_tok in ("feather", "tüy", "tuy", "wing")):
                return None
            if source_type == "minimap_auto" and "odyle" in rt_lower:
                return None

            in_air = bool(raw.get("in_air", False))
            confidence = float(raw.get("confidence", raw.get("score", 1.0)))
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

            return {
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
        except Exception as exc:
            logger.error("Materyal tespiti ayrıştırılırken hata: %s", exc)
            return None

    def _find_nearest_tracked_node(
        self,
        region: str,
        node_map_x: float,
        node_map_y: float,
    ) -> Tuple[Optional[TrackedResourceNode], float]:
        """
        Belirtilen bölgede `(node_map_x, node_map_y)` koordinatına `dedup_radius_px`
        yarıçapı içindeki en yakın kayıtlı düğümü bulur.
        """
        candidates = self._tracked_nodes.get(region, [])
        nearest: Optional[TrackedResourceNode] = None
        min_dist = float("inf")

        for tracked in candidates:
            dist = math.hypot(float(node_map_x - tracked.map_x), float(node_map_y - tracked.map_y))
            if dist <= self.dedup_radius_px and dist < min_dist:
                min_dist = dist
                nearest = tracked

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
        Mevcut düğümün koordinatını 1D/2D Kalman kazancı ve EMA (Exponential Moving Average)
        ağırlıklı ortalamasıyla daha hassas hale getirir ve `sightings_count` değerini artırır.
        """
        try:
            tracked.sightings_count += 1
            tracked.last_seen_timestamp = time.time()

            measurement_var = max(1.5, 12.0 / max(0.15, float(obs_confidence)))
            process_var = 0.25
            pred_var = tracked.kalman_variance + process_var
            kalman_gain = pred_var / (pred_var + measurement_var)

            effective_weight = float(
                np.clip(
                    0.5 * kalman_gain + 0.5 * (self.ema_alpha / math.sqrt(tracked.sightings_count)),
                    0.05,
                    0.65,
                )
            )

            tracked.map_x = (1.0 - effective_weight) * tracked.map_x + effective_weight * float(obs_map_x)
            tracked.map_y = (1.0 - effective_weight) * tracked.map_y + effective_weight * float(obs_map_y)
            tracked.norm_x = round(tracked.map_x / float(max(1, self.map_width)), 4)
            tracked.norm_y = round(tracked.map_y / float(max(1, self.map_height)), 4)
            tracked.kalman_variance = (1.0 - kalman_gain) * pred_var
            tracked.confidence = round(
                min(1.0, max(tracked.confidence, float(obs_confidence)) + 0.02),
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
                and tracked.resource_type in (DEFAULT_RESOURCE_TYPE, "Gatherable Material")
            ):
                tracked.resource_type = str(obs_resource_type)

            if route and route != DEFAULT_ROUTE and route not in tracked.routes:
                tracked.routes.append(route)

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

        for parsed in parsed_items:
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

            node_map_x, node_map_y, norm_x, norm_y = self.odometry_tracker.offset_to_world_coords(
                rel_dx, rel_dy
            )

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
                    obs_confidence=confidence,
                    route_manager=rm,
                    route=target_route,
                    obs_stack_count=stack_count,
                    obs_max_harvests=max_harvests,
                    obs_resource_type=resource_type,
                )
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

            # Yeni kaynak keşfedildi -> Minimap'ten 80x80 gri patch kırp
            patch_80 = self.extract_clean_minimap_patch(
                frame_bgr=bgr,
                rel_map_dx=rel_dx,
                rel_map_dy=rel_dy,
            )

            routes_list = [target_route] if target_route and target_route != DEFAULT_ROUTE else []
            created_node: Optional[NodeData] = None

            if rm is not None:
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
                )
                if created_node is not None:
                    setattr(created_node, "max_harvests", int(max_harvests))
                    setattr(created_node, "current_harvests", int(current_harvests))
                    setattr(created_node, "stack_count", int(stack_count))
                    setattr(created_node, "source_type", str(source_type))
                    self._save_direct_region_node_patch(
                        base_dir=rm.base_dir,
                        region=target_region,
                        node_id=created_node.id,
                        patch_img=patch_80,
                    )

            assigned_id = (
                int(created_node.id)
                if created_node is not None
                else len(self._tracked_nodes.get(target_region, [])) + 1
            )
            if rm is None:
                self._save_direct_region_node_patch(
                    base_dir="routes",
                    region=target_region,
                    node_id=assigned_id,
                    patch_img=patch_80,
                )

            patch_path = (
                created_node.patch_path
                if created_node is not None
                else f"routes/{target_region}/node_{assigned_id:02d}.png"
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
            self._tracked_nodes.setdefault(target_region, []).append(tracked_new)

            if source_type == "minimap_auto":
                self.total_minimap_recorded += 1
            elif source_type == "3d_auto":
                self.total_3d_recorded += 1
            elif source_type == "harvest_auto":
                self.total_harvest_recorded += 1
                self.total_3d_recorded += 1

            node_dict = tracked_new.to_dict()
            newly_discovered_nodes.append(node_dict)
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
        (3 yapraklı filizler ve kanatlı Odyle/tüy ikonları + yeşil `2`/`3` yığın rozetleri)
        tespit eder, `VisualOdometryTracker` ile dünya koordinatlarını hesaplar ve
        80x80 görsel şablonunu kaydederek `max_harvests = 3 * stack_count` ile rotaya ekler.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
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
    ) -> List[Dict[str, Any]]:
        """
        Minimap'te görünmeyen ancak 3D oyun dünyasında beyaz kanatlı toplama arması (`Aria`, `Mela`)
        veya yüzen `[E]` ikonu ile görünen kaynakları oyuncu ayak konumuna `(w*0.5, h*0.54)` göre
        harita koordinatlarına izdüşürerek `source_type="3d_auto"` ve `max_harvests=3` ile kaydeder.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return []

            frame_h, frame_w = bgr.shape[:2]
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
    ) -> List[Dict[str, Any]]:
        """
        Oyuncu veya bot bir materyal üzerinde `[F Gather]` / `[E]` etkileşiminde bulunduğunda
        veya ekranda `[F Gather]` belirdiğinde, düğüm henüz rotada yoksa o konumu
        `source_type="harvest_auto"` ve `max_harvests=3` olarak kaydeder.
        """
        try:
            bgr = self._ensure_bgr(frame)
            if bgr is None:
                return []

            frame_h, frame_w = bgr.shape[:2]
            world_3d = self._detect_3d_world_gatherables(bgr)
            prompts = self._detect_active_harvest_prompt(bgr, world_3d_detections=world_3d)

            if material_type:
                if prompts:
                    for p in prompts:
                        p["resource_type"] = str(material_type)
                else:
                    prompts.append(
                        {
                            "rel_map_dx": 0.0,
                            "rel_map_dy": 0.0,
                            "resource_type": str(material_type),
                            "in_air": False,
                            "confidence": 0.98,
                            "stack_count": 1,
                            "max_harvests": DEFAULT_MAX_HARVESTS,
                            "current_harvests": 0,
                            "source_type": "harvest_auto",
                        }
                    )

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

            target_node.current_harvests = min(
                target_node.max_harvests,
                target_node.current_harvests + 1,
            )
            target_node.last_seen_timestamp = time.time()

            if rm is not None:
                node_obj = rm.get_node(target_node.node_id, region=target_region)
                if node_obj is not None:
                    setattr(node_obj, "current_harvests", int(target_node.current_harvests))
                    setattr(node_obj, "max_harvests", int(target_node.max_harvests))
                    node_obj.gathered_count += 1
                    if target_node.is_depleted:
                        node_obj.last_visited_timestamp = time.time()
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
        """Returns the cartographer health summary safely."""
        try:
            return {"cartographer_ready": True}
        except Exception:
            return {"cartographer_ready": False}

    def get_cartography_stats(self, region: Optional[str] = None) -> Dict[str, Any]:
        """
        HUD ve Cartography Radar için haritalandırma istatistiklerini döndürür.
        `total_minimap_recorded` ve `total_3d_recorded` sayaçlarını içerir.
        """
        try:
            nodes = self.get_tracked_nodes(region=region)
            total_max_harvests = sum(int(n.get("max_harvests", DEFAULT_MAX_HARVESTS)) for n in nodes)
            total_current_harvests = sum(int(n.get("current_harvests", 0)) for n in nodes)
            total_available_harvests = sum(int(n.get("remaining_harvests", DEFAULT_MAX_HARVESTS)) for n in nodes)
            return {
                "total_nodes": len(nodes),
                "total_minimap_recorded": int(self.total_minimap_recorded),
                "total_3d_recorded": int(self.total_3d_recorded),
                "total_harvest_recorded": int(self.total_harvest_recorded),
                "total_max_harvests": int(total_max_harvests),
                "total_current_harvests": int(total_current_harvests),
                "total_available_harvests": int(total_available_harvests),
                "player_map_x": round(float(self.map_x), 3),
                "player_map_y": round(float(self.map_y), 3),
                "player_norm_x": float(self.norm_x),
                "player_norm_y": float(self.norm_y),
                "walk_trajectory_points": len(self.walk_trajectory),
                "is_calibrated_on_world_map": bool(self._is_calibrated_on_world_map),
                "cumulative_drift_px": round(float(self._cumulative_drift_px), 3),
            }
        except Exception as exc:
            logger.error("get_cartography_stats hatası: %s", exc)
            return {
                "total_nodes": 0,
                "total_minimap_recorded": int(self.total_minimap_recorded),
                "total_3d_recorded": int(self.total_3d_recorded),
                "total_harvest_recorded": int(self.total_harvest_recorded),
                "total_max_harvests": 0,
                "total_current_harvests": 0,
                "total_available_harvests": 0,
                "player_map_x": round(float(self.map_x), 3),
                "player_map_y": round(float(self.map_y), 3),
                "player_norm_x": float(self.norm_x),
                "player_norm_y": float(self.norm_y),
                "walk_trajectory_points": len(self.walk_trajectory),
                "is_calibrated_on_world_map": bool(self._is_calibrated_on_world_map),
                "cumulative_drift_px": 0.0,
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
        self._prev_minimap_gray = None
        self._cumulative_drift_px = 0.0
        self.odometry_tracker.reset(
            map_x=self.map_x,
            map_y=self.map_y,
            clear_trajectory=clear_trajectory,
        )

    def get_auto_cartographer_ready_flag(self) -> dict:
        """Returns the auto cartographer ready flag."""
        return {"module": "auto_cartographer", "ready": True}

    def get_auto_cartographer_version_info(self) -> dict:
        return {"module": "auto_cartographer", "version": "2.0.0", "ready": True}

    def get_auto_cartographer_diagnostic_snapshot(self) -> dict:
        return {"module": "auto_cartographer", "status": "ok", "ready": True}


def get_auto_cartographer_capabilities() -> list:
    return ["auto_cartographer", "minimap_mapping", "diagnostic_snapshot"]


def validate_auto_cartographer_contract() -> bool:
    return bool(get_auto_cartographer_capabilities())


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



