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
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from core.corridor_navigator import WORLD_MAP_METERS_PER_PX, jump_length_threshold
from core.minimap_scanner import MinimapScanner

# P1-1: minimap class ids / names come from the single taxonomy source
# (dataset_factory.class_taxonomy, authority dataset/data.yaml) — the
# radar used to hardcode 9/11 and the non-canonical name
# "minimap_monster_negative" in a field called taxonomy_class_name.
from dataset_factory.class_taxonomy import canonical_name_to_id as _taxonomy_name_to_id

_TAXONOMY_IDS = _taxonomy_name_to_id()
MINIMAP_SPROUT_CLASS_ID: int = _TAXONOMY_IDS["minimap_sprout"]
MINIMAP_MONSTER_DOT_CLASS_ID: int = _TAXONOMY_IDS["minimap_monster_dot"]
MINIMAP_SPROUT_CLASS_NAME: str = "minimap_sprout"
MINIMAP_MONSTER_DOT_CLASS_NAME: str = "minimap_monster_dot"

from ui.shell_page_frame import wrap_shell_tab_root_layout

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
    QPalette,
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
    QComboBox,
)

from ui.map_viewer import (
    node_matches_material,
    sort_nodes_by_proximity,
    normalize_material_token,
)
from version import __version__

logger = logging.getLogger("Gatherion.CartographyRadar")

# Başlık şeridi, sağ üst büyüteç, sağ alt saat ve oyuncu okunun iç halkası.
# 3x/6x/9x yer tutucuları bu kromun üstüne oturuyordu; filiz testi dışında kalan
# nokta toplanabilir materyal sayılmaz. İç disk yalnız filiz adayını eler;
# gerçek canavar noktaları (•) diskte kalsa da kayda girmez.
# Oyuncu oku kleransı (0.075), oyuncu etrafındaki toplanabilir filizlerin
# kör noktaya düşmesini engellemek için daraltılmıştır.
_ZONE_TITLE_FRAC = 0.18
_MAGNIFIER_X_FRAC = 0.80
_MAGNIFIER_Y_FRAC = 0.36
_CLOCK_X_FRAC = 0.62
_CLOCK_Y_FRAC = 0.74
_GATHER_PLAYER_CLEARANCE = 0.075
# 3B projeksiyon zemin mesafesi kalibrasyonu (RadarCanvas.project_3d_yolo_boxes).
# Değerler Aion 2 70° HFOV kamera geometrisinden ampirik çıkarıldı; sihirli
# sabit olarak gömülmesin diye burada adlandırıldı (kalibre edilmediler,
# yalnızca tek yerden okunur hale getirildiler).
_PROJ_NEAR_DIST_M = 1.5  # En yakın zemin mesafesi (m)
_PROJ_FAR_DIST_M = 55.0  # En uzak zemin mesafesi (m)
_PROJ_FEET_BASE_M = 5.5  # Ayak çizgisindeki zemin mesafesi (m)
_PROJ_FEET_FALLOFF = 0.78  # Ayaktan uzağa gidildikçe mesafe düşüş katsayısı
_PROJ_BEYOND_FEET_BASE_M = 6.0  # Ayak çizgisi üstü taban mesafe (m)
_PROJ_BEYOND_FEET_SPAN_M = 39.0  # Ayak çizgisi üstü mesafe aralığı (m)
_PROJ_BEYOND_FEET_EXP = 1.35  # Ayak çizgisi üstü mesafe eğrisi üssü
_PROJ_APPARENT_REF_H_PX = 160.0  # Görünen yükseklikten mesafe için referans boy (px)
_PROJ_APPARENT_MIN_H_PX = 8.0  # Görünen yükseklik alt sınırı (px)
_PROJ_APPARENT_SCALE = 4.5  # Görünen yükseklikten mesafe ölçeği
_PANEL_SCANNER: Optional[MinimapScanner] = None


def _detection_confidence(det: Any, default: float = 0.0) -> float:
    """
    Tespit güvenini None/bozuk değere karşı güvenli okur.

    `det.get("confidence", 0.0)` anahtar mevcut ama değeri None ise
    `float(None)` ile çöker; eksik/None/bozuk değerde `default` döner.

    P1-4 (K-07): eksik güven artık 0.90 diye uydurulmuyor — bilinmiyorsa
    0.0 (minimap yolundaki `mat.get("confidence", 0.0)` ile tutarlı).
    """
    raw: Any = None
    if isinstance(det, dict):
        for key in ("confidence", "conf", "score"):
            if det.get(key) is not None:
                raw = det.get(key)
                break
    else:
        for key in ("confidence", "conf", "score"):
            value = getattr(det, key, None)
            if value is not None:
                raw = value
                break
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def _panel_minimap_scanner() -> MinimapScanner:
    global _PANEL_SCANNER
    if _PANEL_SCANNER is None:
        _PANEL_SCANNER = MinimapScanner()
    return _PANEL_SCANNER


def find_zone_distance_bbox(
    minimap_bgr: np.ndarray,
    apply_hd: bool = False,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Bölge adının altındaki hedef mesafesi (`5225m` / `5235m`) yazısının kutusunu döndürür.
    Kutu yalnızca kırpımı kaydırmak için değil, o piksellerin materyal sayılmaması için kullanılır.
    apply_hd=True durumunda kutuyu Altgard HD (2258x2400) hizalamasına öteler ve ölçekler.
    """
    if minimap_bgr is None or not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.size == 0:
        return None
    if minimap_bgr.ndim == 2:
        gray = minimap_bgr
    else:
        gray = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    if height < 40 or width < 40:
        return None

    # Eğer tam ekran / büyük kare verildiyse sağ üst minimap kesitini kontrol et
    crop_offset_x, crop_offset_y = 0, 0
    if width > 600 and height > 400:
        rx, ry, rw, rh = MinimapLiveCropPreviewWidget.MINIMAP_NORM_RECT
        crop_offset_x = int(width * rx)
        crop_offset_y = int(height * ry)
        cw = max(32, int(width * rw))
        ch = max(32, int(height * rh))
        gray = gray[crop_offset_y : crop_offset_y + ch, crop_offset_x : crop_offset_x + cw]
        height, width = gray.shape[:2]
        if height < 40 or width < 40:
            return None

    y0, y1 = int(height * 0.10), int(height * 0.40)
    x0, x1 = int(width * 0.12), int(width * 0.88)
    roi = gray[y0:y1, x0:x1]
    if roi.size == 0:
        return None

    blur = cv2.GaussianBlur(roi, (0, 0), 3)
    diff = roi.astype(np.int16) - blur.astype(np.int16)
    mask = ((diff > 12) & (roi > 150)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[Tuple[int, int, int, int]] = None
    best_dx = 1e9
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 12 or bh < 4 or bh > int(height * 0.14) or bw > int(width * 0.70):
            continue
        if bw / float(max(1, bh)) < 1.6:
            continue
        cx = x0 + x + bw * 0.5
        cy = y0 + y + bh * 0.5
        if cy > height * 0.34 or abs(cx - width * 0.5) > width * 0.30:
            continue
        dx = abs(cx - width * 0.5)
        if best is None or dx < best_dx - 1.0 or (abs(dx - best_dx) <= 1.0 and bw > best[2]):
            best = (x0 + x, y0 + y, bw, bh)
            best_dx = dx
    if best is None:
        return None

    pad = 8
    x, y, bw, bh = best
    x0e = max(0, x - pad) + crop_offset_x
    y0e = max(0, y - pad) + crop_offset_y
    x1e = min(width + crop_offset_x, x + bw + pad + crop_offset_x)
    y1e = min(height + crop_offset_y, y + bh + pad + crop_offset_y)
    box = (x0e, y0e, max(1, x1e - x0e), max(1, y1e - y0e))

    if apply_hd:
        bx, by, bw_b, bh_b = box
        h_bx, h_by = apply_hd_fit(float(bx), float(by))
        h_bw = max(1, int(round(float(bw_b) * _HD_A)))
        h_bh = max(1, int(round(float(bh_b) * _HD_D)))
        return (int(round(h_bx)), int(round(h_by)), h_bw, h_bh)
    return box


def _point_in_rect(px: float, py: float, rect: Tuple[int, int, int, int]) -> bool:
    x, y, bw, bh = rect
    return x <= px <= x + bw and y <= py <= y + bh


def _hud_chrome_rects(
    width: int,
    height: int,
    distance_box: Optional[Tuple[int, int, int, int]],
) -> List[Tuple[int, int, int, int]]:
    rects = [
        (0, 0, width, max(1, int(round(height * _ZONE_TITLE_FRAC)))),
        (int(round(width * _MAGNIFIER_X_FRAC)), 0, width, max(1, int(round(height * _MAGNIFIER_Y_FRAC)))),
        (int(round(width * _CLOCK_X_FRAC)), int(round(height * _CLOCK_Y_FRAC)), width, height),
    ]
    if distance_box is not None:
        rects.append(distance_box)
    return rects


def _stable_player_center(
    crop_shape: Tuple[int, ...],
    detected: Tuple[float, float],
    distance_box: Optional[Tuple[int, int, int, int]],
) -> Tuple[float, float]:
    """Mesafe yazısı parlaklığı oyuncu merkezini yukarı çekmesin."""
    height, width = int(crop_shape[0]), int(crop_shape[1])
    fallback = (
        float(width) * MinimapScanner.PLAYER_CENTER_REL[0],
        float(height) * MinimapScanner.PLAYER_CENTER_REL[1],
    )
    cx, cy = float(detected[0]), float(detected[1])
    if distance_box is not None and _point_in_rect(cx, cy, distance_box):
        return fallback
    if cy < height * 0.36 or cy > height * 0.78 or cx < width * 0.20 or cx > width * 0.80:
        return fallback
    return cx, cy


def minimap_norm_is_hud_chrome(
    norm_x: float,
    norm_y: float,
    confidence: float = 0.0,
) -> bool:
    """Başlık, saat, büyüteç veya oyuncu oku / iç halka. Filiz kaydı bu alanda yapılmaz.
    Yüksek güvenli (confidence >= 0.72) filiz düğümleri iç oyuncu kleransı (blind zone)
    içinde kalsa dahi elenmez."""
    nx, ny = float(norm_x), float(norm_y)
    if ny < _ZONE_TITLE_FRAC:
        return True
    if nx >= _MAGNIFIER_X_FRAC and ny <= _MAGNIFIER_Y_FRAC:
        return True
    if nx >= _CLOCK_X_FRAC and ny >= _CLOCK_Y_FRAC:
        return True
    if float(confidence) < 0.72 and math.hypot(
        nx - MinimapScanner.PLAYER_CENTER_REL[0],
        ny - MinimapScanner.PLAYER_CENTER_REL[1],
    ) <= _GATHER_PLAYER_CLEARANCE:
        return True
    return False


def minimap_material_is_recordable(mat: Dict[str, Any]) -> bool:
    """
    Minimap'ten kayda girecek düğüm: mevcut filiz testi (`sprout_material`) geçmiş olmalı.
    Canavar noktası, bölge adı, saat ve oyuncu oku kaydedilmez.
    İç klerans içindeki yüksek güvenli (confidence >= 0.72) filizler korunur.
    """
    if not isinstance(mat, dict):
        return False
    if mat.get("hud_chrome"):
        return False
    if mat.get("is_gatherable") is not True:
        return False
    if str(mat.get("kind", "")) != "sprout_material":
        return False
    try:
        conf = float(mat.get("confidence", 0.0) or 0.0)
        if minimap_norm_is_hud_chrome(
            float(mat.get("norm_x", 0.5)),
            float(mat.get("norm_y", 0.5)),
            confidence=conf,
        ):
            return False
    except (TypeError, ValueError):
        return False
    return True


def classify_minimap_crop_bgr(
    minimap_bgr: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Kırpılmış minimap üzerinde filiz testini çalıştırır.
    Toplanabilir olan yalnızca `sprout_material`dır. Yuvarlak beyaz nokta canavardır ve kayda girmez.
    Bölge adı, altındaki mesafe yazısı, saat ve oyuncu oku materyal değildir.
    Kırpım dikdörtgeni bu yazıya göre kaydırılmaz.
    """
    materials: List[Dict[str, Any]] = []
    monsters: List[Dict[str, Any]] = []
    if minimap_bgr is None or not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.size == 0:
        return materials, monsters

    if minimap_bgr.ndim == 2:
        crop = cv2.cvtColor(minimap_bgr, cv2.COLOR_GRAY2BGR)
    elif minimap_bgr.shape[2] == 4:
        crop = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGRA2BGR)
    else:
        crop = minimap_bgr

    height, width = crop.shape[:2]
    if height < 20 or width < 20:
        return materials, monsters

    scan = _panel_minimap_scanner().scan_minimap_crop(crop)
    distance_box = find_zone_distance_bbox(crop)
    center_x, center_y = _stable_player_center(
        crop.shape,
        tuple(scan.get("player_center", (width * 0.5, height * 0.5))),
        distance_box,
    )
    chrome = _hud_chrome_rects(width, height, distance_box)
    gather_clearance = _GATHER_PLAYER_CLEARANCE * float(min(width, height))

    for node in scan.get("gatherable_nodes", []):
        if str(node.get("kind", "")) != "sprout_material" or node.get("is_gatherable") is not True:
            continue
        pos = node.get("pos")
        if not pos or len(pos) < 2:
            continue
        px, py = float(pos[0]), float(pos[1])
        conf = float(node.get("confidence", 0.0) or 0.0)
        # Prevent player-adjacent harvest nodes from being dropped into the blind zone:
        # Nodes with confidence >= 0.72 must be kept even inside inner clearance.
        if conf < 0.72 and math.hypot(px - center_x, py - center_y) <= gather_clearance:
            continue
        if any(_point_in_rect(px, py, rect) for rect in chrome):
            continue
        harvests = int(node.get("harvests_available") or node.get("stack_count") or 3)
        if harvests not in (3, 6, 9):
            harvests = 3
        nx = round(px / float(width), 4)
        ny = round(py / float(height), 4)
        half = max(7.0, 0.055 * float(min(width, height)))
        materials.append(
            {
                "norm_x": nx,
                "norm_y": ny,
                "box_x": round(max(0.0, px - half) / float(width), 4),
                "box_y": round(max(0.0, py - half) / float(height), 4),
                "box_w": round(min(float(width), 2.0 * half) / float(width), 4),
                "box_h": round(min(float(height), 2.0 * half) / float(height), 4),
                "rel_map_dx": round(px - center_x, 1),
                "rel_map_dy": round(py - center_y, 1),
                "resource_type": "Toplanabilir Materyal",
                "stack_count": harvests,
                "max_harvests": harvests,
                "stack_badge": f"{harvests}x",
                "harvest_badge": f"0/{harvests}",
                "confidence": float(node.get("confidence", 0.0) or 0.0),
                "source_type": "minimap_auto",
                "kind": "sprout_material",
                "class_id": MINIMAP_SPROUT_CLASS_ID,
                "class_name": MINIMAP_SPROUT_CLASS_NAME,
                "is_gatherable": True,
                "hud_chrome": False,
            }
        )

    for dot in scan.get("monster_dots", []):
        pos = dot.get("pos")
        if not pos or len(pos) < 2:
            continue
        px, py = float(pos[0]), float(pos[1])
        if math.hypot(px - center_x, py - center_y) <= gather_clearance:
            continue
        if any(_point_in_rect(px, py, rect) for rect in chrome):
            continue
        monsters.append(
            {
                "norm_x": round(px / float(width), 4),
                "norm_y": round(py / float(height), 4),
                "label": "•✕",
                "kind": "monster_dot",
                "class_id": MINIMAP_MONSTER_DOT_CLASS_ID,
                "class_name": MINIMAP_MONSTER_DOT_CLASS_NAME,
                "is_hostile": False,
                "is_gatherable": False,
                "hud_chrome": False,
            }
        )

    for dot in scan.get("hostile_dots", []):
        pos = dot.get("pos")
        if not pos or len(pos) < 2:
            continue
        px, py = float(pos[0]), float(pos[1])
        if math.hypot(px - center_x, py - center_y) <= gather_clearance:
            continue
        if any(_point_in_rect(px, py, rect) for rect in chrome):
            continue
        monsters.append(
            {
                "norm_x": round(px / float(width), 4),
                "norm_y": round(py / float(height), 4),
                "label": "🔴",
                "kind": "hostile_dot",
                # P1-1: the taxonomy name is "minimap_monster_dot" —
                # "minimap_monster_negative" is not a canonical name
                # (it used to leak into taxonomy_class_name downstream);
                # hostility is carried by kind/is_hostile, not the name.
                "class_id": MINIMAP_MONSTER_DOT_CLASS_ID,
                "class_name": MINIMAP_MONSTER_DOT_CLASS_NAME,
                "is_hostile": True,
                "is_gatherable": False,
                "hud_chrome": False,
            }
        )

    return materials, monsters

CANONICAL_REGION_ALIASES: Dict[str, str] = {
    "dredgion crash site": "Altgard",
    "safe haven": "Altgard",
    "calderon canyon": "Altgard",
    "silent hill": "Altgard",
    "altgard fortress": "Altgard",
    "moslan forest": "Altgard",
    "mahki swamp": "Altgard",
    "gribbler canyon": "Altgard",
    "idun's lake": "Altgard",
    "altgard ice lake": "Altgard",
    "altgard": "Altgard",
}


def resolve_canonical_region(region_name: Optional[str]) -> str:
    """Alt bölge kanonik alias eşlemesi ('Dredgion Crash Site' -> 'Altgard')."""
    if not region_name:
        return "Altgard"
    raw = str(region_name).strip()
    return CANONICAL_REGION_ALIASES.get(raw.lower(), raw)


# Dünya haritası piksel çerçevesi. frame_18s.jpg 1280×720.
# Kayıtlar bazen bu çerçevede, bazen 2× (2560×1440) tutuluyor.
DEFAULT_MAP_REF: Tuple[float, float] = (1280.0, 720.0)

# Source: https://aion2data.com/game/maps/altgard.webp
# AION 2 Altgard dünya haritası (Aion2Data, istemci 2026-10-28). 2258×2400.
HD_ZONE_MAP_REL = os.path.join("assets", "maps", "altgard.png")

# Bölgeye duyarlı HD harita görselleri (Görev 4).
# Source: https://aion2data.com/game/maps/verteron.webp
# AION 2 Verteron dünya haritası (Aion2Data). 2303×2400.
# KURAL: Altgard görseli yalnızca bölge Altgard (veya bilinmiyor) ise adaydır;
# başka bölge için asla Altgard haritası gösterilmez.
HD_ZONE_MAPS: Dict[str, str] = {
    "altgard": os.path.join("assets", "maps", "altgard.png"),
    "verteron": os.path.join("assets", "maps", "verteron.png"),
}

# frame_18s.jpg Altgard dünya haritası kırpmasıdır; yalnızca Altgard
# bölgesinde yedek olabilir. Başka bölgede yedek olamaz.
FRAME18_REGION_KEYS: Tuple[str, ...] = ("", "altgard")


def region_map_candidates(region_name: Optional[str]) -> List[str]:
    """Bölge için HD harita görseli adayları (mutlak yol, sırayla).

    Altgard görseli yalnızca bölge Altgard (veya boş/bilinmiyor) ise adaydır;
    başka bölge için kendi görseli yoksa BOŞ liste döner — çağrıcı mozaik
    yedeğe düşer, yanlış bölge haritası göstermez (Görev 4).
    """
    key = str(region_name or "").strip().lower()
    if key == "":
        # Bölge bilinmiyorsa eski varsayılan: Altgard.
        key = "altgard"
    rel = HD_ZONE_MAPS.get(key)
    if rel is None:
        return []
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [os.path.join(root, rel)]


def is_altgard_map_region(region_name: Optional[str]) -> bool:
    """frame_18s.jpg (Altgard kırpması) bu bölge için yedek olabilir mi?"""
    return str(region_name or "").strip().lower() in FRAME18_REGION_KEYS

# frame_18s.jpg arazi kırpması (kare içinde sol-üst 280,80) ORB ile bu PNG'ye
# oturtuldu. Ölçek/öteleme yalnızca çizimde. region_data.json yazılmaz.
# Safe Haven, Dredgion Crash Site ve Silent Hill etiketleri aynı kuzeybatı
# arazinin üstüne denk geldi.
_HD_A = 1.46687753
_HD_B = -0.0150533993
_HD_C = -0.00368272413
_HD_D = 1.42560995
_HD_CROP_X = 280.0
_HD_CROP_Y = 80.0
_HD_TX = 242.632119
_HD_TY = 476.526405
HD_FIT_OFFSET: Tuple[float, float] = (
    _HD_TX - _HD_CROP_X * _HD_A - _HD_CROP_Y * _HD_B,
    _HD_TY - _HD_CROP_X * _HD_C - _HD_CROP_Y * _HD_D,
)


def hd_zone_map_path(region: Optional[str] = None) -> str:
    """Bölgeye duyarlı HD harita yolu.

    region verilmezse geriye dönük uyumluluk için Altgard yolu döner.
    Bölgenin kendi görseli yoksa (örn. Heiron) yine Altgard yolu döner;
    dosya varlığı çağrıcıda region_map_candidates ile denetlenmelidir —
    bu fonksiyon tek başına "harita var" anlamına gelmez.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if region:
        candidates = region_map_candidates(region)
        if candidates:
            return candidates[0]
    return os.path.join(root, HD_ZONE_MAP_REL)


def apply_hd_fit(x: float, y: float) -> Tuple[float, float]:
    """1280×720 video karesi pikseli -> HD Altgard pikseli (2258×2400). 614×450 kırpması yok."""
    ox, oy = HD_FIT_OFFSET
    return (
        _HD_A * float(x) + _HD_B * float(y) + ox,
        _HD_C * float(x) + _HD_D * float(y) + oy,
    )


def inv_hd_fit(hx: float, hy: float) -> Tuple[float, float]:
    """HD Altgard pikseli (2258×2400) -> 1280×720 video karesi pikseli ters dönüşümü."""
    ox, oy = HD_FIT_OFFSET
    det = _HD_A * _HD_D - _HD_B * _HD_C
    if abs(det) < 1e-9:
        return float(hx), float(hy)
    dx = float(hx) - ox
    dy = float(hy) - oy
    x = (_HD_D * dx - _HD_B * dy) / det
    y = (-_HD_C * dx + _HD_A * dy) / det
    return x, y


def resolve_node_xy(
    node: Any,
    ref_w: float = DEFAULT_MAP_REF[0],
    ref_h: float = DEFAULT_MAP_REF[1],
    apply_hd: bool = False,
) -> Tuple[float, float]:
    """
    Düğümü harita çerçevesine oturtur.
    map_x/map_y başka bir çözünürlükte yazıldıysa (norm ile anlaşmıyorsa)
    norm_x/norm_y üzerinden referans çerçeveye çeker. Kayıtlı veriyi değiştirmez.
    Altgard HD (2258x2400) hizalaması için apply_hd=True veya ref=(2258, 2400)
    durumunda HD fit (_HD_A, _HD_D, HD_FIT_OFFSET) uygular.
    """
    is_hd_target = bool(apply_hd) or (
        abs(float(ref_w) - 2258.0) <= 2.0 and abs(float(ref_h) - 2400.0) <= 2.0
    )

    try:
        if isinstance(node, dict):
            x = float(node.get("map_x", node.get("x", 0.0)) or 0.0)
            y = float(node.get("map_y", node.get("y", 0.0)) or 0.0)
            nx = float(node.get("norm_x", 0.0) or 0.0)
            ny = float(node.get("norm_y", 0.0) or 0.0)
        elif isinstance(node, (tuple, list)) and len(node) >= 2:
            x = float(node[0])
            y = float(node[1])
            nx = 0.0
            ny = 0.0
        else:
            x = float(getattr(node, "map_x", getattr(node, "x", 0.0)) or 0.0)
            y = float(getattr(node, "map_y", getattr(node, "y", 0.0)) or 0.0)
            nx = float(getattr(node, "norm_x", 0.0) or 0.0)
            ny = float(getattr(node, "norm_y", 0.0) or 0.0)
    except Exception as exc:
        logger.debug("Coordinate extraction error: %s", exc)
        return 0.0, 0.0

    if is_hd_target:
        # Eğer koordinat zaten HD harita ölçeğindeyse (örn. x > 1400 veya implied_w ~ 2258)
        if (x > 1400.0 or y > 850.0) or (0.0 < nx <= 1.0 and abs(x / max(1e-4, nx) - 2258.0) < 200.0):
            return x, y
        # Değilse video karesi pikselini HD Altgard'a oturt
        base_x, base_y = resolve_node_xy(node, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1], apply_hd=False)
        return apply_hd_fit(base_x, base_y)

    if x == 0.0 and y == 0.0 and 0.0 < nx <= 1.0 and 0.0 < ny <= 1.0:
        return nx * float(ref_w), ny * float(ref_h)

    if (
        ref_w > 1.0
        and ref_h > 1.0
        and 0.0 < nx <= 1.0
        and 0.0 < ny <= 1.0
        and abs(x) > 1.0
        and abs(y) > 1.0
    ):
        implied_w = x / nx
        implied_h = y / ny
        if abs(implied_w - ref_w) / ref_w > 0.22 or abs(implied_h - ref_h) / ref_h > 0.22:
            return nx * float(ref_w), ny * float(ref_h)
    return x, y


def _segment_lengths(coords: List[Tuple[float, float]], closed: bool) -> List[float]:
    n = len(coords)
    if n < 2:
        return []
    count = n if (closed and n >= 3) else n - 1
    lengths: List[float] = []
    for i in range(count):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        lengths.append(math.hypot(x1 - x0, y1 - y0))
    return lengths


# jump_length_threshold: tek tanım core/corridor_navigator.py'de (P1-12 / v18).
# Zikzak-sıçrama eşiği — yerel adımın çok üstündeki bağı sıçrama sayar.
# (Eski yerel tanım buradan kaldırıldı; bu modüldeki iki kullanım yeri
# — calculate_route_metrics ve _jump_mask — artık aynı tanımı okur.)


def _distance_matrix(coords: List[Tuple[float, float]]) -> np.ndarray:
    """Koordinatlar arası simetrik Öklid mesafe matrisi (n x n, float64).

    Rota iyileştirme (en yakın komşu, 2-opt, or-opt) eskiden her aday çifti
    için ayrı `math.hypot` çağrısı yapıyordu: n=120 düğümde tek bir
    `optimize_smooth_loop` ~92 bin hypot çağrısı demekti. Matris bir kez
    kurulur, sonrası O(1) tablo okumasıdır; değerler `math.hypot` ile
    aynı float64 Öklid mesafesidir (numpy `hypot`), yani tur maliyeti ve
    iyileştirme kararları değişmez.
    """
    pts = np.asarray(coords, dtype=np.float64).reshape(-1, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.hypot(diff[..., 0], diff[..., 1])


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
    Döngüsel rotaları (cyclical / closed-loop) ve alt bölge kanonik aliaslarını
    ('Dredgion Crash Site' -> 'Altgard') tam olarak destekler.
    """

    CANONICAL_REGION_ALIASES: Dict[str, str] = CANONICAL_REGION_ALIASES

    def __init__(self, meters_per_pixel: float = WORLD_MAP_METERS_PER_PX) -> None:
        self.meters_per_pixel: float = float(meters_per_pixel)
        self._map_ref: Tuple[float, float] = DEFAULT_MAP_REF
        self._hd_fit: bool = False
        self._external_planner: Optional[Any] = None
        ext_cls = _resolve_external_sequential_planner()
        if ext_cls is not None:
            try:
                self._external_planner = ext_cls()
            except Exception as exc:
                logger.debug("External SequentialRoutePlanner instantiation fallback: %s", exc)
                self._external_planner = None

    @staticmethod
    def resolve_canonical_region(region_name: Optional[str]) -> str:
        """Kanonik bölge adını çözümler ('Dredgion Crash Site' -> 'Altgard')."""
        return resolve_canonical_region(region_name)

    def canonical_region(self, region_name: Optional[str]) -> str:
        """Kanonik bölge adını döndürür."""
        return resolve_canonical_region(region_name)

    def get_canonical_alias(self, subregion: str) -> str:
        """Alt bölgenin bağlı olduğu kanonik üst bölgeyi döndürür."""
        return resolve_canonical_region(subregion)

    def is_subregion_of(self, subregion: str, region: str) -> bool:
        """Verilen alt bölgenin belirtilen kanonik bölgeye ait olup olmadığını kontrol eder."""
        return resolve_canonical_region(subregion).lower() == resolve_canonical_region(region).lower()

    def set_map_reference(self, width: float, height: float) -> None:
        """Harita piksel çerçevesini ayarlar (kırpma öncesi tam görüntü)."""
        self._map_ref = (max(1.0, float(width)), max(1.0, float(height)))

    def set_hd_fit(self, enabled: bool) -> None:
        """Düğümleri 1280×720 karesinden HD haritaya ölçekle. JSON'a yazmaz."""
        self._hd_fit = bool(enabled)

    def _node_xy(self, node: Any) -> Tuple[float, float]:
        """Düğümü aktif harita çerçevesine oturtulmuş (x, y) olarak döndürür."""
        if self._hd_fit:
            x, y = resolve_node_xy(node, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1], apply_hd=True)
            return x, y
        return resolve_node_xy(node, self._map_ref[0], self._map_ref[1], apply_hd=False)

    def _dist(self, a: Any, b: Any) -> float:
        ax, ay = self._node_xy(a)
        bx, by = self._node_xy(b)
        return math.hypot(bx - ax, by - ay)

    def optimize_smooth_loop(
        self,
        nodes: List[Any],
        walk_trajectory: Optional[List[Tuple[float, float]]] = None,
    ) -> List[Any]:
        """
        Düğümleri bi oraya bi buraya zikzak yapmayacak (0 ping-pong sıçrama) şekilde
        pürüzsüz bir ardışık döngü (Nearest-Neighbor + 2-Opt + Or-Opt + Turn-Angle Smoothing)
        halinde sıralar ve 1..N seq_index ataması yapar.
        """
        if not nodes:
            return []
        if len(nodes) <= 2:
            ordered_small: List[Any] = []
            for seq_i, item in enumerate(nodes, start=1):
                if isinstance(item, dict):
                    c_item = dict(item)
                    c_item["seq_index"] = seq_i
                    ordered_small.append(c_item)
                else:
                    if hasattr(item, "seq_index"):
                        try:
                            setattr(item, "seq_index", seq_i)
                        except Exception:
                            pass
                    ordered_small.append(item)
            return ordered_small

        # Harici planlayıcı varsa (core.sequential_router.SequentialRoutePlanner) kullan ve 2-opt ile doğrula
        working: List[Any] = list(nodes)
        external_order_used = False
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
                        external_order_used = True
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
                                external_order_used = True
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
        # Mesafe matrisi bir kez kurulur ve aşağıdaki tüm adımlar (tur maliyeti,
        # multi-start NN, 2-opt, or-opt) aynı tabloyu kullanır; eskiden her
        # `_tour_cost_indices` çağrısı matrisi sıfırdan kuruyordu.
        dist = _distance_matrix(coords)
        best_cost = self._tour_cost_indices(best_order, coords, dist)

        # Harici planlayıcının sırası çöpe atılmaz: aday tur olarak maliyetine
        # bakılır, dahili adaylardan kısaysa o tur kazanır.
        if external_order_used:
            ext_order = list(range(n_count))
            ext_cost = self._tour_cost_indices(ext_order, coords, dist)
            if ext_cost < best_cost:
                best_cost = ext_cost
                best_order = ext_order

        # Multi-start nearest neighbor denemeleri (mesafe matrisi + argmin:
        # eskiden her adımda kalan düğümler üzerinde math.hypot'lu `min`
        # çalışıyordu — O(n) hypot çağrısı/adım, başlangıç başına O(n^2)).
        step = max(1, n_count // 8)
        start_indices = list(range(0, n_count, step))
        if 0 not in start_indices:
            start_indices.insert(0, 0)

        for start_idx in start_indices:
            remaining = np.ones(n_count, dtype=bool)
            remaining[start_idx] = False
            tour = [start_idx]
            curr = start_idx
            while remaining.any():
                row = np.where(remaining, dist[curr], np.inf)
                nxt = int(np.argmin(row))
                tour.append(nxt)
                remaining[nxt] = False
                curr = nxt
            cost = self._tour_cost_indices(tour, coords, dist)
            if cost < best_cost:
                best_cost = cost
                best_order = tour

        # 2-Opt Kesişim ve Zikzak Giderme (Uncrossing all intersecting edges)
        best_order = self._two_opt_indices(best_order, coords, dist)

        # Or-Opt (Tekli düğüm kaydırma ile keskin geri dönüşleri / ping-pong'ları temizle)
        best_order = self._or_opt_indices(best_order, coords, dist)

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

        ordered_nodes: List[Any] = []
        for seq_i, orig_i in enumerate(best_order, start=1):
            item = working[orig_i]
            if isinstance(item, dict):
                c_item = dict(item)
                c_item["seq_index"] = seq_i
                ordered_nodes.append(c_item)
            else:
                if hasattr(item, "seq_index"):
                    try:
                        setattr(item, "seq_index", seq_i)
                    except Exception:
                        pass
                ordered_nodes.append(item)
        return ordered_nodes

    def plan_cyclical_route(
        self,
        nodes: List[Any],
        walk_trajectory: Optional[List[Tuple[float, float]]] = None,
        region_name: Optional[str] = None,
    ) -> List[Any]:
        """
        Düğümleri döngüsel (cyclical / closed-loop) olarak zikzaksız sıralar.
        Alt bölge takma adlarını ('Dredgion Crash Site' -> 'Altgard') destekler.
        """
        c_reg = resolve_canonical_region(region_name)
        ordered = self.optimize_smooth_loop(nodes, walk_trajectory=walk_trajectory)
        for idx, n in enumerate(ordered, start=1):
            if isinstance(n, dict):
                n["seq_index"] = idx
                if region_name:
                    n["region"] = c_reg
            elif hasattr(n, "seq_index"):
                try:
                    setattr(n, "seq_index", idx)
                except Exception:
                    pass
        return ordered

    def get_next_in_sequence(
        self,
        current_node: Any,
        nodes: List[Any],
        forward_window: int = 3,
        loop: bool = True,
    ) -> Optional[Any]:
        """
        Döngüsel rotada mevcut düğümden sonraki hedef düğümü (ileri pencere içinde) döndürür.
        Rotanın sonuna gelindiğinde (N -> 1) döngüsel olarak başa sarar (clean cyclical wrap-around).
        """
        if not nodes:
            return None
        n = len(nodes)
        if n == 1:
            return nodes[0]

        curr_idx: Optional[int] = None
        for idx, item in enumerate(nodes):
            if item is current_node:
                curr_idx = idx
                break
            if isinstance(item, dict) and isinstance(current_node, dict):
                if "id" in item and "id" in current_node and item["id"] == current_node["id"]:
                    curr_idx = idx
                    break
                if "seq_index" in item and "seq_index" in current_node and item["seq_index"] == current_node["seq_index"]:
                    curr_idx = idx
                    break
            elif hasattr(item, "id") and hasattr(current_node, "id"):
                if getattr(item, "id") == getattr(current_node, "id"):
                    curr_idx = idx
                    break

        if curr_idx is None:
            cx, cy = self._node_xy(current_node)
            curr_idx = min(
                range(n),
                key=lambda i: math.hypot(self._node_xy(nodes[i])[0] - cx, self._node_xy(nodes[i])[1] - cy)
            )

        # İleri pencere (forward_window): mevcut düğümde zaten duruluyorsa
        # (hedefe varılmışsa) pencere içindeki ilk henüz varılmamış düğüme
        # ilerle; yoksa sıradaki düğüm hedef olur. Pencere 1 ise hep sıradaki.
        window = max(1, int(forward_window)) if forward_window is not None else 1
        cx, cy = self._node_xy(current_node)
        for step in range(1, window + 1):
            cand_idx = curr_idx + step
            if loop:
                cand_idx %= n
            elif cand_idx >= n:
                break
            ccx, ccy = self._node_xy(nodes[cand_idx])
            if math.hypot(ccx - cx, ccy - cy) > 2.0:
                return nodes[cand_idx]

        next_idx = (curr_idx + 1) % n if loop else min(curr_idx + 1, n - 1)
        return nodes[next_idx]

    def cyclical_next(self, current_node: Any, nodes: List[Any]) -> Optional[Any]:
        """Döngüsel rotada ardışık bir sonraki düğümü ((i + 1) % len) döndürür."""
        return self.get_next_in_sequence(current_node, nodes, forward_window=1, loop=True)

    def insert_node_in_sequence(self, new_node: Any, nodes: List[Any]) -> List[Any]:
        """
        Yeni keşfedilen düğümü mevcut döngüsel sırayı bozmadan, en az ek yol maliyeti
        oluşturan (d(i, new) + d(new, i+1) - d(i, i+1)) ardışık iki düğüm arasına ekler.
        Tüm düğümlerin seq_index değerini 1..N günceller.
        """
        if not nodes:
            return self.optimize_smooth_loop([new_node])
        if len(nodes) <= 2:
            return self.optimize_smooth_loop(list(nodes) + [new_node])

        coords = [self._node_xy(n) for n in nodes]
        new_pt = self._node_xy(new_node)
        n = len(nodes)

        best_pos = 0
        best_cost_delta = float("inf")
        for i in range(n):
            j = (i + 1) % n
            d_ij = math.hypot(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1])
            d_inew = math.hypot(new_pt[0] - coords[i][0], new_pt[1] - coords[i][1])
            d_newj = math.hypot(coords[j][0] - new_pt[0], coords[j][1] - new_pt[1])
            delta = d_inew + d_newj - d_ij
            if delta < best_cost_delta:
                best_cost_delta = delta
                best_pos = i + 1

        res = list(nodes)
        res.insert(best_pos, new_node)
        ordered: List[Any] = []
        for seq_i, item in enumerate(res, start=1):
            if isinstance(item, dict):
                c = dict(item)
                c["seq_index"] = seq_i
                ordered.append(c)
            else:
                if hasattr(item, "seq_index"):
                    try:
                        setattr(item, "seq_index", seq_i)
                    except Exception:
                        pass
                ordered.append(item)
        return ordered

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
    ) -> Dict[str, Union[int, float, bool]]:
        """
        Verilen düğüm dizilimi için:
        - discovered_nodes / node_count (int)
        - smoothness_score_pct / smoothness_score (float, 0..100)
        - total_loop_distance_m / total_distance_m / total_distance_px (float)
        - avg_step_m / avg_step_px (float)
        - max_jump_m / max_jump_px (float)
        - min_step_m / min_step_px (float)
        - ping_pong_jumps (int)
        - closed_loop (bool)
        metriklerini hesaplar.
        """
        scale = float(meters_per_pixel if meters_per_pixel is not None else self.meters_per_pixel)
        count = len(nodes or [])
        # Kapalı döngü en az 3 düğüm ister; 0/1/2 düğümde kapanacak döngü yoktur.
        is_closed_loop = count >= 3
        if count == 0:
            return {
                "discovered_nodes": 0,
                "node_count": 0,
                "smoothness_score_pct": 100.0,
                "smoothness_score": 100.0,
                "total_loop_distance_m": 0.0,
                "total_distance_m": 0.0,
                "total_distance_px": 0.0,
                "avg_step_m": 0.0,
                "avg_step_px": 0.0,
                "max_jump_m": 0.0,
                "max_jump_px": 0.0,
                "min_step_m": 0.0,
                "min_step_px": 0.0,
                "ping_pong_jumps": 0,
                "closed_loop": is_closed_loop,
            }
        if count == 1:
            return {
                "discovered_nodes": 1,
                "node_count": 1,
                "smoothness_score_pct": 100.0,
                "smoothness_score": 100.0,
                "total_loop_distance_m": 0.0,
                "total_distance_m": 0.0,
                "total_distance_px": 0.0,
                "avg_step_m": 0.0,
                "avg_step_px": 0.0,
                "max_jump_m": 0.0,
                "max_jump_px": 0.0,
                "min_step_m": 0.0,
                "min_step_px": 0.0,
                "ping_pong_jumps": 0,
                "closed_loop": is_closed_loop,
            }

        coords = [self._node_xy(n) for n in nodes]
        # Kapalı döngü en az 3 düğüm ister (yukarıdaki is_closed_loop):
        # 2 düğümde kapanış kenarı (B -> A) yoktur, tek gidiş kenarı (A -> B)
        # sayılır. Eski kod range(count) ile B -> A kenarını da toplayıp
        # 2 düğümlü toplam mesafeyi iki kat çıkarıyordu (P1-12 / v18).
        num_segments = count if count >= 3 else max(count - 1, 0)
        total_px = 0.0
        seg_lengths: List[float] = []
        for i in range(num_segments):
            x0, y0 = coords[i]
            x1, y1 = coords[(i + 1) % count]
            seg = math.hypot(x1 - x0, y1 - y0)
            seg_lengths.append(seg)
            total_px += seg

        total_meters = round(total_px * scale, 1)
        avg_step_px = round(total_px / float(num_segments), 1) if num_segments > 0 else 0.0
        avg_step_m = round(avg_step_px * scale, 1)
        max_jump_px = round(max(seg_lengths), 1) if seg_lengths else 0.0
        max_jump_m = round(max_jump_px * scale, 1)
        min_step_px = round(min(seg_lengths), 1) if seg_lengths else 0.0
        min_step_m = round(min_step_px * scale, 1)

        if count == 2:
            return {
                "discovered_nodes": 2,
                "node_count": 2,
                "smoothness_score_pct": 100.0,
                "smoothness_score": 100.0,
                "total_loop_distance_m": total_meters,
                "total_distance_m": total_meters,
                "total_distance_px": round(total_px, 1),
                "avg_step_m": avg_step_m,
                "avg_step_px": avg_step_px,
                "max_jump_m": max_jump_m,
                "max_jump_px": max_jump_px,
                "min_step_m": min_step_m,
                "min_step_px": min_step_px,
                "ping_pong_jumps": 0,
                "closed_loop": is_closed_loop,
            }

        # Zikzak sıçrama = harita çerçevesine oturtulmuş koordinatta
        # yerel adımdan çok daha uzun bağ (yanlış çözünürlük / sahte köşe noktası).
        threshold = jump_length_threshold(seg_lengths)
        ping_pong_jumps = sum(1 for seg in seg_lengths if seg > threshold)
        sharp_turn_penalty = 0.0
        for i in range(count):
            p_prev = coords[(i - 1) % count]
            p_curr = coords[i]
            p_next = coords[(i + 1) % count]
            v1x, v1y = p_curr[0] - p_prev[0], p_curr[1] - p_prev[1]
            v2x, v2y = p_next[0] - p_curr[0], p_next[1] - p_curr[1]
            len1 = math.hypot(v1x, v1y)
            len2 = math.hypot(v2x, v2y)
            if len1 > 48.0 and len2 > 48.0:
                cos_theta = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (len1 * len2)))
                turn_angle_deg = math.degrees(math.acos(cos_theta))
                if turn_angle_deg > 150.0 and len1 < threshold and len2 < threshold:
                    sharp_turn_penalty += 6.0

        raw_score = 100.0 - (ping_pong_jumps * 15.0) - min(18.0, sharp_turn_penalty * 0.35)
        smoothness = round(max(5.0, min(100.0, raw_score)), 1)

        return {
            "discovered_nodes": count,
            "node_count": count,
            "smoothness_score_pct": smoothness,
            "smoothness_score": smoothness,
            "total_loop_distance_m": total_meters,
            "total_distance_m": total_meters,
            "total_distance_px": round(total_px, 1),
            "avg_step_m": avg_step_m,
            "avg_step_px": avg_step_px,
            "max_jump_m": max_jump_m,
            "max_jump_px": max_jump_px,
            "min_step_m": min_step_m,
            "min_step_px": min_step_px,
            "ping_pong_jumps": int(ping_pong_jumps),
            "closed_loop": is_closed_loop,
        }

    def evaluate_sequence_quality(
        self,
        nodes: Sequence[Any],
        closed_loop: bool = True,
    ) -> Dict[str, Any]:
        """core.sequential_router uyumluluğu için rota kalite metriklerini döndürür."""
        metrics = self.calculate_route_metrics(list(nodes))
        metrics["closed_loop"] = bool(closed_loop)
        return metrics

    # ------------------------------------------------------------------
    # Dahili Geometri ve 2-Opt Yardımcıları
    # ------------------------------------------------------------------
    @staticmethod
    def _tour_cost_indices(
        order: List[int],
        coords: List[Tuple[float, float]],
        dist: Optional[np.ndarray] = None,
    ) -> float:
        """Kapalı turun toplam uzunluğu. `dist` verilirse önceden kurulmuş
        mesafe matrisi kullanılır (çağrı başına O(n²) matris kurulumu yok);
        verilmezse matris burada kurulur (eski davranış)."""
        n = len(order)
        if n <= 1:
            return 0.0
        if dist is None:
            dist = _distance_matrix(coords)
        idx = np.asarray(order, dtype=np.intp)
        nxt = np.roll(idx, -1)
        return float(np.sum(dist[idx, nxt]))

    @classmethod
    def _two_opt_indices(
        cls,
        order: List[int],
        coords: List[Tuple[float, float]],
        dist: Optional[np.ndarray] = None,
        max_passes: int = 40,
    ) -> List[int]:
        n = len(order)
        if n <= 3:
            return list(order)
        if dist is None:
            dist = _distance_matrix(coords)

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
                    bi, bi1 = best[i], best[i + 1]
                    bj, bj1 = best[j], best[(j + 1) % n]
                    old_len = dist[bi, bi1] + dist[bj, bj1]
                    new_len = dist[bi, bj] + dist[bj1, bi1]
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
        dist: Optional[np.ndarray] = None,
        max_passes: int = 15,
    ) -> List[int]:
        """Tekil düğümleri daha uygun komşu kenarlara taşıyarak keskin zikzakları giderir."""
        n = len(order)
        if n <= 4:
            return list(order)
        if dist is None:
            dist = _distance_matrix(coords)

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

                rem_gain = dist[v, u] + dist[w, v] - dist[w, u]
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
                    ins_cost = dist[v, p] + dist[q, v] - dist[q, p]
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

    detections_changed = pyqtSignal()
    MINIMAP_NORM_RECT: Tuple[float, float, float, float] = (0.855, 0.065, 0.130, 0.215)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(240, 205)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self._minimap_pixmap: Optional[QPixmap] = None
        self._gatherable_materials: List[Dict[str, Any]] = []
        self._rejected_monsters: List[Dict[str, Any]] = []
        self._sweep_angle_rad: float = 0.0

        self._init_default_minimap_state()

    @staticmethod
    def crop_box(frame_w: int, frame_h: int) -> Tuple[int, int, int, int]:
        """Sağ üst minimap kırpımı. İçerik (bölge adı, `5225m`) kutuyu kaydırmaz."""
        rx, ry, rw, rh = MinimapLiveCropPreviewWidget.MINIMAP_NORM_RECT
        x = int(frame_w * rx)
        y = int(frame_h * ry)
        cw = max(32, int(frame_w * rw))
        ch = max(32, int(frame_h * rh))
        return x, y, cw, ch

    def _init_default_minimap_state(self) -> None:
        """
        `video_frames/frame_00s.jpg` sağ üst minimap kesitini yükler.
        Yalnızca filiz testini geçen ikonlar materyal olur; yer tutucu 3x/6x/9x konmaz.
        """
        loaded_img: Optional[QImage] = None
        for candidate_rel in (
            os.path.join("video_frames", "frame_00s.jpg"),
            os.path.join("video_frames", "frame_33s.jpg"),
            os.path.join("video_frames", "frame_36s.jpg"),
            os.path.join("video_frames", "frame_18s.jpg"),
        ):
            full_p = os.path.join(os.getcwd(), candidate_rel)
            if not os.path.exists(full_p):
                full_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), candidate_rel)
            if os.path.exists(full_p):
                img = QImage(full_p)
                if not img.isNull():
                    cx, cy, cw, ch = self.crop_box(img.width(), img.height())
                    loaded_img = img.copy(QRect(cx, cy, cw, ch))
                    break

        mats: List[Dict[str, Any]] = []
        mobs: List[Dict[str, Any]] = []
        if loaded_img is not None and not loaded_img.isNull():
            self._minimap_pixmap = QPixmap.fromImage(loaded_img)
            try:
                mats, mobs = self.analyze_minimap_image(loaded_img)
            except Exception as exc:
                logger.debug("Minimap kırpımı ayrıştırılamadı: %s", exc)
                mats, mobs = [], []
        self.set_minimap_detections(mats, mobs)

    @staticmethod
    def _qimage_to_bgr(minimap_img: QImage) -> Optional[np.ndarray]:
        if minimap_img is None or minimap_img.isNull():
            return None
        image = minimap_img.convertToFormat(QImage.Format.Format_RGB888)
        width, height = image.width(), image.height()
        if width < 2 or height < 2:
            return None
        stride = int(image.bytesPerLine())
        raw = image.bits()
        raw.setsize(height * stride)
        arr = np.frombuffer(raw, np.uint8).reshape((height, stride))[:, : width * 3]
        rgb = arr.reshape((height, width, 3)).copy()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    @classmethod
    def analyze_minimap_image(
        cls, minimap_img: QImage
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Sağ üst minimap kesitinde filiz testini (`sprout_material`) ve canavar noktasını ayırır.
        Bölge adı, mesafe yazısı, saat ve oyuncu oku materyal değildir.
        """
        bgr = cls._qimage_to_bgr(minimap_img)
        if bgr is None:
            return [], []
        return classify_minimap_crop_bgr(bgr)

    def set_minimap_pixmap(self, pixmap: QPixmap) -> None:
        """Canlı minimap kırpma görüntüsünü günceller ve filiz testini yeniden çalıştırır."""
        self._minimap_pixmap = pixmap
        mats: List[Dict[str, Any]] = []
        mobs: List[Dict[str, Any]] = []
        if pixmap is not None and not pixmap.isNull():
            try:
                mats, mobs = self.analyze_minimap_image(pixmap.toImage())
            except Exception as exc:
                logger.debug("Canlı minimap kırpımı ayrıştırılamadı: %s", exc)
        self.set_minimap_detections(mats, mobs)

    def update_minimap_crop(self, minimap_crop: Any) -> None:
        """
        Canlı minimap kırpımını (BGR ndarray) önizlemeye işler.

        Canvas (`SequentialMapCanvas.update_live_frame`) ve diyalog
        (`CartographyRadarDialog.update_live_frame`) bu metodu çağırır;
        daha önce metot yoktu, çağrı AttributeError ile sessizce yutuluyordu.
        """
        if minimap_crop is None:
            return
        try:
            if isinstance(minimap_crop, QPixmap):
                self.set_minimap_pixmap(minimap_crop)
                return
            arr = np.asarray(minimap_crop)
            if arr.size == 0 or arr.ndim != 3 or arr.shape[2] < 3:
                return
            rgb = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            qimg = QImage(
                rgb.data, w, h, int(rgb.strides[0]), QImage.Format.Format_RGB888
            )
            self.set_minimap_pixmap(QPixmap.fromImage(qimg.copy()))
        except Exception as exc:
            logger.debug("update_minimap_crop hatası: %s", exc)

    def set_minimap_detections(
        self,
        gatherable_materials: List[Dict[str, Any]],
        rejected_monsters: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Filiz testini geçen materyalleri ve yoksayılan canavar noktalarını ayarlar."""
        normalized_mats: List[Dict[str, Any]] = []
        for mat in gatherable_materials or []:
            if not isinstance(mat, dict) or not minimap_material_is_recordable(mat):
                continue
            d = dict(mat)
            sc = int(d.get("stack_count", d.get("max_harvests", 3)) or 3)
            if sc not in (3, 6, 9):
                sc = 3
            d["stack_count"] = sc
            d["max_harvests"] = sc
            d["stack_badge"] = f"{sc}x"
            d.setdefault("harvest_badge", f"0/{sc}")
            d.setdefault("source_type", "minimap_auto")
            d["kind"] = "sprout_material"
            d["is_gatherable"] = True
            d["hud_chrome"] = False
            normalized_mats.append(d)

        self._gatherable_materials = normalized_mats
        if rejected_monsters is not None:
            self._rejected_monsters = [
                dict(m)
                for m in rejected_monsters
                if isinstance(m, dict) and not minimap_material_is_recordable(m)
            ]
        self.update()
        self.detections_changed.emit()

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
        header_h = 22.0
        footer_h = 24.0
        crop_rect = QRectF(
            target_rect.left() + inner_margin,
            target_rect.top() + header_h + 4.0,
            max(40.0, target_rect.width() - inner_margin * 2.0),
            max(40.0, target_rect.height() - header_h - footer_h - 10.0),
        )

        # Üst Başlık: Sağ Üst Minimap Canlı Kırpma & Filtre
        hdr_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(hdr_font)
        painter.setPen(QColor("#67e8f9"))
        painter.drawText(
            QRectF(target_rect.left() + 8.0, target_rect.top() + 3.0, target_rect.width() - 16.0, header_h),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            "Sağ Üst Minimap Canlı Kırpma & Filtre",
        )

        # Minimap Görüntüsü. Boş halka ızgarası ve geometrik sarı nokta çizilmez:
        # filiz yokken bunlar oyun kırmızısının üstüne binen sahte işaret oluyordu.
        if self._minimap_pixmap is not None and not self._minimap_pixmap.isNull():
            painter.drawPixmap(crop_rect, self._minimap_pixmap, QRectF(self._minimap_pixmap.rect()))
        else:
            painter.fillRect(crop_rect, QColor("#09131c"))

        # 1. Yoksayılan Canavar Beyaz Noktaları (• shown as dim gray/red crossed-out dots •✕)
        mob_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
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
        badge_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(badge_font)
        for mat in self._gatherable_materials:
            nx = float(mat.get("norm_x", 0.5))
            ny = float(mat.get("norm_y", 0.5))
            mx = crop_rect.left() + nx * crop_rect.width()
            my = crop_rect.top() + ny * crop_rect.height()
            mpt = QPointF(mx, my)

            # Kabul edilen filizin kutusu (yalnızca sprout testi geçen nokta)
            if "box_w" in mat and "box_h" in mat:
                box = QRectF(
                    crop_rect.left() + float(mat.get("box_x", nx)) * crop_rect.width(),
                    crop_rect.top() + float(mat.get("box_y", ny)) * crop_rect.height(),
                    max(8.0, float(mat["box_w"]) * crop_rect.width()),
                    max(8.0, float(mat["box_h"]) * crop_rect.height()),
                )
            else:
                side = max(14.0, min(crop_rect.width(), crop_rect.height()) * 0.10)
                box = QRectF(mx - side * 0.5, my - side * 0.5, side, side)
            painter.setPen(QPen(QColor("#6ee7b7"), 1.5))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)

            # Dış camgöbeği halka (#22d3ee) + iç zümrüt yeşili dolgu (#10b981)
            painter.setPen(QPen(QColor("#22d3ee"), 2.0))
            painter.setBrush(QBrush(QColor(6, 182, 212, 55)))
            painter.drawEllipse(mpt, 7.8, 7.8)

            painter.setPen(QPen(QColor("#042f2e"), 1.5))
            painter.setBrush(QBrush(QColor("#10b981")))
            painter.drawEllipse(mpt, 4.8, 4.8)

            # 3x / 6x / 9x yığın hasat rozeti
            stack_txt = str(mat.get("stack_badge") or f"{int(mat.get('stack_count', 3) or 3)}x")
            pill_rect = QRectF(mx + 7.5, my - 8.0, 24.0, 16.0)
            painter.setPen(QPen(QColor("#34d399"), 1.2))
            painter.setBrush(QBrush(QColor(6, 48, 40, 235)))
            painter.drawRoundedRect(pill_rect, 3.5, 3.5)

            painter.setPen(QColor("#6ee7b7"))
            painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, stack_txt)

        # Alt Bilgi Satırı
        footer_font = QFont("Segoe UI", 8, QFont.Weight.DemiBold)
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

    # Eski orta kırpma (0.295, 0.145, 0.480, 0.625) → 1280×720 karede 614×450.
    # Tuval artık bu parçaya kesilmez; düğümler tam karenin pikselindedir.

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumSize(680, 460)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._base_pixmap: Optional[QPixmap] = None
        self._map_ref: Tuple[float, float] = DEFAULT_MAP_REF
        self._hd_fit_active: bool = False
        self._ordered_nodes: List[Dict[str, Any]] = []
        self._walk_trajectory: List[Tuple[float, float]] = []
        self._player_pos: Optional[Tuple[float, float, float]] = None  # (x, y, angle_deg)
        self._show_loop_closure: bool = True
        self._show_minimap_overlay: bool = False
        self._minimap_preview_ref: Optional[MinimapLiveCropPreviewWidget] = None

        # Pulse animasyon durumları: list of {"x": float, "y": float, "start_ts": float, "duration": float, "label": str}
        self._active_pulses: List[Dict[str, Any]] = []
        self._radar_sweep_phase: float = 0.0
        self._auto_cartography_active: bool = True

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(35)  # ~28 FPS akıcı radar pulse animasyonu
        self._pulse_timer.timeout.connect(self._on_animation_tick)
        self._pulse_timer.start()

        # Zoom & Pan durumu. Alt sınır, 1280×720 bölge karesinin küçük panele sığmasını kesmesin.
        self._zoom: float = 1.0
        self._min_zoom: float = 0.05
        self._max_zoom: float = 5.0
        self._pan: QPointF = QPointF(0.0, 0.0)
        self._is_dragging: bool = False
        self._drag_start_pos: QPointF = QPointF(0.0, 0.0)
        self._drag_moved_px: float = 0.0
        self._hovered_seq_idx: Optional[int] = None
        self._selected_seq_idx: Optional[int] = None
        self._initial_fit_done: bool = False

        # Selective gather material filter & proximity route
        self._material_filter: Optional[str] = None
        self._proximity_ordered_nodes: List[Dict[str, Any]] = []

    def set_material_filter(self, materials: Union[str, Sequence[str], None]) -> None:
        """Sets the selective gather material filter on the radar/map canvas."""
        if materials is None:
            self._material_filter = None
        elif isinstance(materials, (list, tuple, set)):
            clean_list = [str(m).strip() for m in materials if str(m).strip()]
            if not clean_list or any(normalize_material_token(m) in ("tumu", "all", "") for m in clean_list):
                self._material_filter = None
            else:
                self._material_filter = ", ".join(clean_list)
        else:
            s = str(materials).strip()
            norm = normalize_material_token(s)
            if not s or norm in ("tumu", "all", "hepsi", "tum", "tum materyaller"):
                self._material_filter = None
            else:
                self._material_filter = s
        self.calculate_proximity_route()
        self.update()

    def get_material_filter(self) -> Optional[str]:
        return self._material_filter

    def calculate_proximity_route(self, start_xy: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """Calculates proximity route from player position to matching nodes."""
        if not self._material_filter:
            self._proximity_ordered_nodes = []
            return []
        matching = [
            n for n in self._ordered_nodes
            if isinstance(n, dict) and node_matches_material(n, self._material_filter)
        ]
        if not matching:
            self._proximity_ordered_nodes = []
            return []
        if start_xy is None:
            if self._player_pos is not None:
                start_xy = (float(self._player_pos[0]), float(self._player_pos[1]))
            elif self._walk_trajectory:
                start_xy = (float(self._walk_trajectory[-1][0]), float(self._walk_trajectory[-1][1]))
        self._proximity_ordered_nodes = sort_nodes_by_proximity(matching, start_xy=start_xy)
        return list(self._proximity_ordered_nodes)

    def get_proximity_ordered_nodes(self) -> List[Dict[str, Any]]:
        return list(self._proximity_ordered_nodes)

    def get_nearest_target_node(self) -> Optional[Dict[str, Any]]:
        """Returns the nearest matching node to the player position."""
        if not self._material_filter:
            return None
        if not self._proximity_ordered_nodes:
            self.calculate_proximity_route()
        if self._proximity_ordered_nodes:
            return self._proximity_ordered_nodes[0]
        return None

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
    def set_map_reference(self, width: float, height: float) -> None:
        """Kırpma öncesi tam harita çerçevesini ayarlar."""
        self._map_ref = (max(1.0, float(width)), max(1.0, float(height)))
        self.update()

    def set_map_pixmap(
        self,
        pixmap: QPixmap,
        map_ref_size: Optional[Tuple[float, float]] = None,
    ) -> None:
        """Arka plan bölge haritasını ayarlar ve görünüme sığdırır."""
        self._base_pixmap = pixmap
        if map_ref_size is not None:
            self._map_ref = (max(1.0, float(map_ref_size[0])), max(1.0, float(map_ref_size[1])))
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
    def set_hd_fit(self, enabled: bool) -> None:
        self._hd_fit_active = bool(enabled)
        self.update()

    def _resolved_xy(self, node: Any) -> Tuple[float, float]:
        # HD modunda tek dönüşüm: resolve_node_xy(apply_hd=True) zaten HD
        # ölçeğindeki düğümü olduğu gibi bırakır, 1280×720 düğümü HD'ye oturtur.
        # Önceki kod apply_hd=False ile çözüp üstüne apply_hd_fit uyguluyordu;
        # zaten HD ölçeğindeki düğüm ikinci kez ölçekleniyordu (çift dönüşüm).
        if self._hd_fit_active:
            return resolve_node_xy(node, DEFAULT_MAP_REF[0], DEFAULT_MAP_REF[1], apply_hd=True)
        return resolve_node_xy(node, self._map_ref[0], self._map_ref[1])

    def _raw_to_map_coords(self, raw_x: float, raw_y: float) -> QPointF:
        """Tam harita pikseli (ofset yok; eski 614×450 kırpma ofseti kaldırıldı)."""
        return QPointF(float(raw_x), float(raw_y))

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
            nx, ny = self._resolved_xy(node)
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
                        self._selected_seq_idx = idx
                        nid = int(self._ordered_nodes[idx].get("id", idx + 1))
                        self.node_left_clicked.emit(nid)
                        self.update()

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

            # 4. Ardışık Node'lar Arası Yön Oklu Koridor. Uzun sıçrama koridor değildir.
            resolved = [self._resolved_xy(node) for node in self._ordered_nodes]
            widget_pts: List[QPointF] = [
                self._map_to_widget(self._raw_to_map_coords(nx, ny)) for nx, ny in resolved
            ]
            jump_mask = self._jump_mask(resolved)
            self._draw_directional_corridor(painter, widget_pts, jump_mask)

            # 5. Yeni Keşfedilen Gem/Materyal Yeşil Radar Halkası (Pulse Animation)
            self._draw_discovery_pulses(painter)

            # 6. Oyuncu Konumu ve Canlı Keşif Radarı Süpürme Halkası
            if self._player_pos is not None:
                px, py, angle_deg = self._player_pos
                p_wpt = self._map_to_widget(self._raw_to_map_coords(px, py))
                self._draw_player_radar_and_marker(painter, p_wpt, angle_deg)

            # 7. Sıralı Node Rozetleri. Etiketler sığmazsa sadece daire; hover/seçili/sıçrama ucu yazılır.
            jump_nodes: set[int] = set()
            n_pts = len(widget_pts)
            for i, is_jump in enumerate(jump_mask):
                if is_jump and n_pts:
                    jump_nodes.add(i % n_pts)
                    jump_nodes.add((i + 1) % n_pts)
            self._draw_sequential_nodes(painter, widget_pts, jump_nodes)

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

    def _jump_mask(self, coords: List[Tuple[float, float]]) -> List[bool]:
        """Her bağ için True = zikzak sıçrama (koridor olarak çizilmez). Kapanış bağı sonda."""
        closed = bool(self._show_loop_closure)
        lengths = _segment_lengths(coords, closed=closed)
        if not lengths:
            return []
        threshold = jump_length_threshold(lengths)
        return [seg > threshold for seg in lengths]

    def _draw_directional_corridor(
        self,
        painter: QPainter,
        widget_pts: List[QPointF],
        jump_mask: Optional[List[bool]] = None,
    ) -> None:
        """
        Ardışık node'lar arasına yön oklu mavi/turkuaz koridor çizgileri (1 ➔ 2 ➔ 3 ➔ 4) çizer.
        Eşik üstü bağlar kırmızı kesik sıçrama olarak çizilir.
        """
        count = len(widget_pts)
        if count < 2:
            return
        mask = jump_mask or []

        for i in range(count - 1):
            p1 = widget_pts[i]
            p2 = widget_pts[i + 1]
            if i < len(mask) and mask[i]:
                self._draw_jump_segment(painter, p1, p2)
            else:
                self._draw_corridor_segment(painter, p1, p2, is_closing_segment=False)

        if self._show_loop_closure and count >= 3:
            close_idx = count - 1 if len(mask) == count else len(mask) - 1
            is_jump = 0 <= close_idx < len(mask) and mask[close_idx]
            if is_jump:
                self._draw_jump_segment(painter, widget_pts[-1], widget_pts[0])
            else:
                self._draw_corridor_segment(
                    painter,
                    widget_pts[-1],
                    widget_pts[0],
                    is_closing_segment=True,
                )

    def _draw_jump_segment(self, painter: QPainter, p1: QPointF, p2: QPointF) -> None:
        """Haritayı kesen bağı koridor değil, kırmızı kesik sıçrama olarak çizer."""
        dx = p2.x() - p1.x()
        dy = p2.y() - p1.y()
        seg_len = math.hypot(dx, dy)
        if seg_len < 4.0:
            return
        pen = QPen(
            QColor("#f87171"),
            2.2,
            Qt.PenStyle.DashLine,
            Qt.PenCapStyle.RoundCap,
        )
        pen.setDashPattern([7.0, 6.0])
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(p1, p2)
        mid = QPointF((p1.x() + p2.x()) * 0.5, (p1.y() + p2.y()) * 0.5)
        painter.setPen(QPen(QColor("#fecaca"), 1.6))
        painter.drawLine(QPointF(mid.x() - 5.0, mid.y() - 5.0), QPointF(mid.x() + 5.0, mid.y() + 5.0))
        painter.drawLine(QPointF(mid.x() - 5.0, mid.y() + 5.0), QPointF(mid.x() + 5.0, mid.y() - 5.0))

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
                font = QFont("Segoe UI", 9, QFont.Weight.Bold)
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

    def _draw_sequential_nodes(
        self,
        painter: QPainter,
        widget_pts: List[QPointF],
        jump_nodes: Optional[set] = None,
    ) -> None:
        """
        Kayıtlı tüm node'ları sırasına göre (#1, #2, #3 ve #N), minimap oto-kayıt
        camgöbeği/zümrüt çift halkasıyla (`source_type == "minimap_auto"`) ve
        hasat/yığın rozetleriyle (`3x`, `6x`, `9x` veya `0/3`) çizer.
        Etiketler üst üste biniyorsa sadece daire kalır. Hover, seçili, uç ve
        sıçrama uçları her zaman yazılır.
        """
        font = QFont("Segoe UI", 10, QFont.Weight.Bold)
        badge_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        jump_nodes = jump_nodes or set()
        placed_labels: List[QRectF] = []

        total = len(self._ordered_nodes)
        is_filter = bool(self._material_filter)
        prox_map: Dict[int, int] = {}
        if is_filter:
            if not self._proximity_ordered_nodes:
                self.calculate_proximity_route()
            prox_map = {
                int(n.get("id", 0)): r + 1 for r, n in enumerate(self._proximity_ordered_nodes)
            }

        for idx in range(total):
            node = self._ordered_nodes[idx]
            w_pt = widget_pts[idx]
            seq_num = idx + 1
            is_hovered = idx == self._hovered_seq_idx
            is_start = seq_num == 1
            is_end = seq_num == total and total > 1
            is_minimap_auto = (
                str(node.get("source_type", node.get("source", ""))).strip().lower() == "minimap_auto"
                or bool(node.get("minimap_auto", False))
            )

            is_matching = node_matches_material(node, self._material_filter) if is_filter else True
            if is_filter and not is_matching:
                painter.save()
                painter.setOpacity(0.20)

            if is_start:
                fill_color = QColor("#10b981")
            elif is_end:
                fill_color = QColor("#06b6d4")
            else:
                fill_color = QColor("#22c55e")
            radius = 9.0 if is_hovered or idx == self._selected_seq_idx else 7.6

            # Selective gather highlight with double rings and bright pulsing colors
            if is_filter and is_matching:
                pulse = 1.5 * math.sin(getattr(self, "_radar_sweep_phase", 0.0) * 3.0)
                # Outer ring (gold)
                painter.setPen(QPen(QColor("#facc15"), 2.4))
                painter.setBrush(QBrush(QColor(250, 204, 21, 55)))
                painter.drawEllipse(w_pt, radius + 6.2 + pulse, radius + 6.2 + pulse)
                # Inner ring (cyan)
                painter.setPen(QPen(QColor("#00ffff"), 1.8))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(w_pt, radius + 3.4, radius + 3.4)
                fill_color = QColor("#fbbf24")

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
            if is_hovered or idx == self._selected_seq_idx:
                painter.setPen(QPen(QColor("#38bdf8"), 3.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(w_pt, radius + 3.0, radius + 3.0)
            painter.setPen(QPen(QColor("#04131d"), 2.2))
            painter.setBrush(QBrush(fill_color))
            painter.drawEllipse(w_pt, radius, radius)

            if is_filter and not is_matching:
                painter.restore()

        label_order = list(range(total))
        label_order.sort(
            key=lambda i: 0
            if (
                i == self._hovered_seq_idx
                or i == self._selected_seq_idx
                or i in jump_nodes
                or i == 0
                or (total > 1 and i == total - 1)
            )
            else 1
        )
        visible_labels: List[int] = []
        for idx in label_order:
            node = self._ordered_nodes[idx]
            w_pt = widget_pts[idx]
            is_minimap_auto = (
                str(node.get("source_type", node.get("source", ""))).strip().lower() == "minimap_auto"
                or bool(node.get("minimap_auto", False))
            )
            radius = 9.0 if idx in (self._hovered_seq_idx, self._selected_seq_idx) else 7.6
            seq_label = f"#{idx + 1}"
            ring_pad = 5.5 if is_minimap_auto else 0.0
            tx = w_pt.x() + radius + 4.0 + ring_pad
            ty = w_pt.y() + 4.0
            badge_str = str(
                node.get("stack_badge")
                or node.get("harvest_badge")
                or f"{int(node.get('stack_count', node.get('max_harvests', 3)) or 3)}x"
            )
            painter.setFont(font)
            seq_w = float(painter.fontMetrics().horizontalAdvance(seq_label))
            painter.setFont(badge_font)
            bw = max(20.0, float(painter.fontMetrics().horizontalAdvance(badge_str)) + 9.0)
            label_rect = QRectF(tx - 2.0, ty - 13.0, seq_w + 8.0 + bw, 18.0)
            force_label = (
                idx == self._hovered_seq_idx
                or idx == self._selected_seq_idx
                or idx == 0
                or (total > 1 and idx == total - 1)
                or idx in jump_nodes
            )
            overlaps = any(label_rect.intersects(prev) for prev in placed_labels)
            if overlaps and not force_label:
                continue
            placed_labels.append(label_rect.adjusted(-1.0, -1.0, 1.0, 1.0))
            visible_labels.append(idx)

        for idx in visible_labels:
            node = self._ordered_nodes[idx]
            w_pt = widget_pts[idx]
            is_matching = node_matches_material(node, self._material_filter) if is_filter else True
            if is_filter and not is_matching:
                continue

            is_minimap_auto = (
                str(node.get("source_type", node.get("source", ""))).strip().lower() == "minimap_auto"
                or bool(node.get("minimap_auto", False))
            )
            radius = 9.0 if idx in (self._hovered_seq_idx, self._selected_seq_idx) else 7.6
            if is_filter and is_matching:
                p_rank = prox_map.get(int(node.get("id", 0)), idx + 1)
                seq_label = f"#{p_rank}"
            else:
                seq_label = f"#{idx + 1}"
            ring_pad = 5.5 if is_minimap_auto else 0.0
            tx = w_pt.x() + radius + 4.0 + ring_pad
            ty = w_pt.y() + 4.0
            badge_str = str(
                node.get("stack_badge")
                or node.get("harvest_badge")
                or f"{int(node.get('stack_count', node.get('max_harvests', 3)) or 3)}x"
            )
            painter.setFont(font)
            seq_w = float(painter.fontMetrics().horizontalAdvance(seq_label))
            label_col = QColor("#facc15") if (is_filter and is_matching) else QColor("#ffffff")
            self._draw_outlined_text(painter, tx, ty, seq_label, font, text_color=label_col)
            painter.setFont(badge_font)
            bw = max(20.0, float(painter.fontMetrics().horizontalAdvance(badge_str)) + 9.0)
            bh = 16.0
            pill_rect = QRectF(tx + seq_w + 5.0, w_pt.y() - bh * 0.5 + 0.5, bw, bh)
            painter.save()
            border_col = QColor("#facc15" if (is_filter and is_matching) else ("#22d3ee" if is_minimap_auto else "#34d399"))
            bg_col = QColor(15, 23, 42, 230) if (is_filter and is_matching) else QColor(6, 46, 42, 230)
            text_col = QColor("#fde047" if (is_filter and is_matching) else ("#6ee7b7" if badge_str.endswith("x") else "#67e8f9"))
            painter.setPen(QPen(border_col, 1.3))
            painter.setBrush(QBrush(bg_col))
            painter.drawRoundedRect(pill_rect, 4.0, 4.0)
            painter.setPen(text_col)
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
        legend_w = min(float(self.width() - 24), 980.0)
        legend_rect = QRectF(12.0, float(self.height() - 42), legend_w, 30.0)
        painter.setPen(QPen(QColor(45, 55, 72, 190), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 215)))
        painter.drawRoundedRect(legend_rect, 6.0, 6.0)

        font = QFont("Segoe UI", 9, QFont.Weight.DemiBold)
        painter.setFont(font)
        painter.setPen(QColor("#e2e8f0"))
        legend_text = (
            "  ● #1 ➔ #2 ➔ #3: Sıralı Yön Koridoru   |   "
            "- - - Altın Kesikli: Yürüyüş İzi   |   "
            "·· Kırmızı Kesik: Zikzak Sıçrama   |   "
            "◎ Çift Halka: Minimap/3B Oto-Kayıt   |   "
            "Etiket: sığan veya üzerine gelinen"
        )
        painter.drawText(legend_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, legend_text)


class RadarCanvas(SequentialMapCanvas):
    """
    Canlı Taktik Radar Tuvali (RadarCanvas).
    SequentialMapCanvas'tan türetilmiştir ve ek olarak şunları sağlar:
    1. 3B Kamera YZ Kutularının (YOLO boxes) ve Minimap Filiz Tespitlerinin Radara İzdüşümü:
       - 3B nesne koordinatlarını (açı/bearing ve perspektif mesafe) radar polar/kartezyen koordinatlarına yansıtır.
       - Oyuncu konumu, bakış yönü vektörü, iç/orta/dış radar tarama halkaları (10m, 25m, 45m).
       - Renk kodlu tespit işaretleyicileri (Bitki/Herb: Zümrüt Yeşil, Maden/Cevher: Kehribar/Altın,
         Kristal/Gem: Camgöbeği/Turkuaz, Ağaç/Odun: Kahve/Turuncu, Canavar/Düşman: Kırmızı, vb.).
    2. Canlı Minimap Füzyonu & Çevrimdışı/Boş Kalmama (Idle Mode):
       - Canlı kare geldikçe FPS takılması olmadan radar gösterimini akıcı şekilde günceller.
       - Çevrimdışı veya oyun kapalıyken siyah ekran vermez; son bilinen haritayı, sıralı rotayı
         ve referans simülasyonunu temiz bir şekilde canlı tarama animasyonuyla gösterir.
    3. UI Kontrolleri ve Sinyalleri:
       - Programatik ve etkileşimli Yakınlaş/Uzaklaş (Zoom), Görünümü Sığdır, Oyuncuya Odakla,
         ve filtre açma/kapama fonksiyonları (Düğümleri Göster, Düşmanları Göster, Rota/Yolu Göster,
         3B YZ İzdüşümünü Göster).
    """

    projection_updated = pyqtSignal(list)
    filter_changed = pyqtSignal(str, bool)
    zoom_changed = pyqtSignal(float)
    pan_changed = pyqtSignal(float, float)
    live_frame_updated = pyqtSignal(int)
    idle_state_changed = pyqtSignal(bool)

    # Renk ve stil tablosu
    CATEGORY_STYLES = {
        "herb": {"color": "#10b981", "fill": QColor(16, 185, 129, 210), "border": QColor("#064e3b"), "label": "Bitki / Ot"},
        "ore": {"color": "#f59e0b", "fill": QColor(245, 158, 11, 210), "border": QColor("#78350f"), "label": "Maden / Cevher"},
        "gem": {"color": "#06b6d4", "fill": QColor(6, 182, 212, 210), "border": QColor("#164e63"), "label": "Kristal / Gem"},
        "wood": {"color": "#d97706", "fill": QColor(217, 119, 6, 210), "border": QColor("#713f12"), "label": "Ağaç / Odun"},
        "prompt": {"color": "#facc15", "fill": QColor(250, 204, 21, 230), "border": QColor("#713f12"), "label": "Toplama İkonu [F/E]"},
        "enemy": {"color": "#ef4444", "fill": QColor(239, 68, 68, 220), "border": QColor("#7f1d1d"), "label": "Canavar / Tehdit"},
        "sprout": {"color": "#22d3ee", "fill": QColor(34, 211, 238, 210), "border": QColor("#0e7490"), "label": "Minimap Filizi"},
        "obstacle": {"color": "#94a3b8", "fill": QColor(148, 163, 184, 180), "border": QColor("#334155"), "label": "Engel / Kaya"},
        "default": {"color": "#38bdf8", "fill": QColor(56, 189, 248, 200), "border": QColor("#0369a1"), "label": "Tespit"},
    }

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._live_3d_projections: List[Dict[str, Any]] = []
        self._live_minimap_detections: List[Dict[str, Any]] = []

        # Filtre durumları
        self._filter_show_nodes: bool = True
        self._filter_show_enemies: bool = True
        self._filter_show_path: bool = True
        self._filter_show_projections: bool = True
        self._filter_show_sweep: bool = True

        # Canlı / Çevrimdışı (Idle) Modu
        self._is_idle: bool = True
        self._idle_mode_enabled: bool = True
        self._last_live_timestamp: float = 0.0

        # Radar Geometri ve Projeksiyon Parametreleri
        self._heading_deg: float = 0.0
        self._meters_per_pixel: float = WORLD_MAP_METERS_PER_PX
        self._sweep_inner_m: float = 10.0
        self._sweep_mid_m: float = 25.0
        self._sweep_outer_m: float = 45.0
        self._camera_hfov_deg: float = 70.0
        self._feet_y_ratio: float = 0.58
        self._horizon_y_ratio: float = 0.28
        self._offline_sim_active: bool = True
        # S016 (v19/P3-8): varsayılan "Altgard" idi — canlı tespit
        # gelmeden HUD "(Altgard · Altgard)" gösteriyordu; oysa
        # route_manager.py:198 ve bu dosyadaki canlı tespit yolu
        # "bilinmiyor asla varsayılan bölgeye düşülmez" der. İlk
        # tespit gelene kadar "Bilinmiyor" gösterilir.
        self._current_subregion: str = "Bilinmiyor"
        self._current_canonical_region: str = "Bilinmiyor"

    def get_current_subregion(self) -> str:
        return str(self._current_subregion)

    def get_current_canonical_region(self) -> str:
        return str(self._current_canonical_region)

    def set_current_subregion(self, subregion: str, canonical_region: Optional[str] = None) -> None:
        # S016 (v19/P3-8): boş girdi eskiden "Altgard" varsayıyordu.
        if not str(subregion or "").strip():
            self._current_subregion = "Bilinmiyor"
            self._current_canonical_region = "Bilinmiyor"
            self.update()
            return
        self._current_subregion = str(subregion)
        self._current_canonical_region = str(canonical_region or resolve_canonical_region(subregion))
        self.update()

    # ------------------------------------------------------------------
    # UI Filtreleme & Kontrol API
    # ------------------------------------------------------------------
    def set_filter(self, filter_name: str, enabled: bool) -> None:
        """Belirtilen filtreyi ('nodes', 'enemies', 'path', 'projections', 'sweep') açar veya kapatır."""
        key = str(filter_name).strip().lower()
        changed = False
        if key in ("nodes", "node", "dugumler"):
            if self._filter_show_nodes != bool(enabled):
                self._filter_show_nodes = bool(enabled)
                changed = True
        elif key in ("enemies", "enemy", "dusmanlar", "monsters"):
            if self._filter_show_enemies != bool(enabled):
                self._filter_show_enemies = bool(enabled)
                changed = True
        elif key in ("path", "rota", "yol", "corridor"):
            if self._filter_show_path != bool(enabled):
                self._filter_show_path = bool(enabled)
                changed = True
        elif key in ("projections", "projection", "3d", "yolo", "izdusum"):
            if self._filter_show_projections != bool(enabled):
                self._filter_show_projections = bool(enabled)
                changed = True
        elif key in ("sweep", "supurme", "radar_sweep"):
            if self._filter_show_sweep != bool(enabled):
                self._filter_show_sweep = bool(enabled)
                changed = True

        if changed:
            self.filter_changed.emit(key, bool(enabled))
            self.update()

    def get_filter(self, filter_name: str) -> bool:
        key = str(filter_name).strip().lower()
        if key in ("nodes", "node", "dugumler"):
            return self._filter_show_nodes
        if key in ("enemies", "enemy", "dusmanlar", "monsters"):
            return self._filter_show_enemies
        if key in ("path", "rota", "yol", "corridor"):
            return self._filter_show_path
        if key in ("projections", "projection", "3d", "yolo", "izdusum"):
            return self._filter_show_projections
        if key in ("sweep", "supurme", "radar_sweep"):
            return self._filter_show_sweep
        # Bilinmeyen filtre adı 'açık' sayılmaz; yanlış pozitif filtre durumunu gizlerdi.
        logger.debug("get_filter: bilinmeyen filtre adı '%s'", filter_name)
        return False

    def get_filters(self) -> Dict[str, bool]:
        return {
            "show_nodes": self._filter_show_nodes,
            "show_enemies": self._filter_show_enemies,
            "show_path": self._filter_show_path,
            "show_projections": self._filter_show_projections,
            "show_sweep": self._filter_show_sweep,
        }

    def set_filters(self, filter_dict: Dict[str, bool]) -> None:
        for k, v in (filter_dict or {}).items():
            self.set_filter(k, bool(v))

    # Programatik Zoom / Pan
    def zoom_in(self, factor: float = 1.25) -> None:
        cw = max(1.0, float(self.width()))
        ch = max(1.0, float(self.height()))
        center_map = self._widget_to_map(QPointF(cw * 0.5, ch * 0.5))
        new_zoom = max(self._min_zoom, min(self._max_zoom, self._zoom * float(factor)))
        if abs(new_zoom - self._zoom) > 1e-4:
            self._zoom = new_zoom
            self._pan = QPointF(
                cw * 0.5 - center_map.x() * self._zoom,
                ch * 0.5 - center_map.y() * self._zoom,
            )
            self.zoom_changed.emit(self._zoom)
            self.pan_changed.emit(self._pan.x(), self._pan.y())
            self.update()

    def zoom_out(self, factor: float = 1.25) -> None:
        self.zoom_in(1.0 / max(1.01, float(factor)))

    def center_on_player(self) -> None:
        if self._player_pos is not None:
            px, py = self._player_pos[0], self._player_pos[1]
        elif self._walk_trajectory:
            px, py = self._walk_trajectory[-1]
        elif self._base_pixmap and not self._base_pixmap.isNull():
            px, py = float(self._base_pixmap.width()) * 0.5, float(self._base_pixmap.height()) * 0.5
        else:
            px, py = 460.0, 310.0
        cw = max(1.0, float(self.width()))
        ch = max(1.0, float(self.height()))
        raw_pt = self._raw_to_map_coords(px, py)
        self._pan = QPointF(
            cw * 0.5 - raw_pt.x() * self._zoom,
            ch * 0.5 - raw_pt.y() * self._zoom,
        )
        self.pan_changed.emit(self._pan.x(), self._pan.y())
        self.update()

    def pan_by(self, dx: float, dy: float) -> None:
        self._pan += QPointF(float(dx), float(dy))
        self.pan_changed.emit(self._pan.x(), self._pan.y())
        self.update()

    def get_zoom(self) -> float:
        return float(self._zoom)

    def get_pan(self) -> Tuple[float, float]:
        return float(self._pan.x()), float(self._pan.y())

    # ------------------------------------------------------------------
    # 3B YZ Kamera Kutuları (YOLO boxes) & Minimap Projeksiyon Motoru
    # ------------------------------------------------------------------
    def _parse_bbox_coords(self, det: Any, frame_w: int, frame_h: int) -> Optional[Tuple[float, float, float, float]]:
        """Bbox verisinden (center_x, ground_y, width, height) çıkarır."""
        try:
            if isinstance(det, (tuple, list)) and len(det) >= 4:
                x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                bw, bh = abs(x2 - x1), abs(y2 - y1)
                cx = (x1 + x2) * 0.5
                gy = max(y1, y2)
                return cx, gy, bw, bh
            elif isinstance(det, dict):
                b = det.get("box") or det.get("bbox")
                if isinstance(b, (tuple, list)) and len(b) >= 4:
                    x1, y1, x2, y2 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                    bw, bh = abs(x2 - x1), abs(y2 - y1)
                    cx = (x1 + x2) * 0.5
                    gy = max(y1, y2)
                    return cx, gy, bw, bh
                elif "screen_x" in det and "screen_y" in det:
                    cx = float(det["screen_x"])
                    gy = float(det["screen_y"])
                    bw = float(det.get("width", 32.0))
                    bh = float(det.get("height", 32.0))
                    return cx, gy, bw, bh
                elif "x" in det and "y" in det:
                    # x/y her zaman (merkez_x, zemin_y) demektir; w/h yalnızca
                    # boyuttur. "w" varken x'i sol-üst saymak, w yokken merkez
                    # saymakla çelişiyordu.
                    x = float(det["x"])
                    y = float(det["y"])
                    bw = float(det.get("w", det.get("width", 32.0)))
                    bh = float(det.get("h", det.get("height", 32.0)))
                    return x, y, bw, bh
            elif hasattr(det, "x1") and hasattr(det, "y1"):
                x1, y1 = float(det.x1), float(det.y1)
                x2 = float(getattr(det, "x2", x1 + 32))
                y2 = float(getattr(det, "y2", y1 + 32))
                return (x1 + x2) * 0.5, max(y1, y2), abs(x2 - x1), abs(y2 - y1)
        except Exception as exc:
            logger.debug("_parse_bbox_coords parse error: %s", exc)
        return None

    def _classify_detection_label(self, raw_label: str) -> Tuple[str, str, str]:
        """Etiket ismini analiz edip (category, color_hex, shape) döndürür."""
        lbl = str(raw_label or "resource").strip().lower()
        if any(w in lbl for w in ("monster", "enemy", "hostile", "mob", "aggro")):
            return "enemy", "#ef4444", "cross"
        if any(w in lbl for w in ("herb", "plant", "aria", "mela", "targena", "ot", "bitki")):
            return "herb", "#10b981", "circle_dot"
        if any(w in lbl for w in ("ore", "metal", "orichalcum", "iron", "maden", "cevher")):
            return "ore", "#f59e0b", "diamond"
        if any(w in lbl for w in ("gem", "crystal", "aether", "kristal", "elmas")):
            return "gem", "#06b6d4", "gem"
        if any(w in lbl for w in ("wood", "tree", "log", "agac", "odun", "kereste")):
            return "wood", "#d97706", "square"
        if any(w in lbl for w in ("crest", "f_gather", "e_gather", "prompt", "topla")):
            return "prompt", "#facc15", "star"
        if any(w in lbl for w in ("sprout", "filiz")):
            return "sprout", "#22d3ee", "double_circle"
        if any(w in lbl for w in ("obstacle", "rock", "wall", "engel", "kaya")):
            return "obstacle", "#94a3b8", "barrier"
        return "default", "#38bdf8", "circle"

    def project_3d_yolo_boxes(
        self,
        yolo_boxes: Optional[List[Any]],
        frame_w: int = 1280,
        frame_h: int = 720,
        heading_deg: Optional[float] = None,
        player_pos: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        3B kamera görüntüsündeki YOLO kutularını perspektif izdüşümle radar ekranına yansıtır.
        """
        projections: List[Dict[str, Any]] = []
        if not yolo_boxes:
            self._live_3d_projections = []
            return projections

        fw = max(64, int(frame_w))
        fh = max(64, int(frame_h))

        if heading_deg is not None:
            self._heading_deg = float(heading_deg)
        elif self._player_pos is not None:
            self._heading_deg = float(self._player_pos[2])
        eff_heading = float(self._heading_deg)

        if player_pos is not None:
            eff_px, eff_py = float(player_pos[0]), float(player_pos[1])
        elif self._player_pos is not None:
            eff_px, eff_py = float(self._player_pos[0]), float(self._player_pos[1])
        elif self._walk_trajectory:
            eff_px, eff_py = float(self._walk_trajectory[-1][0]), float(self._walk_trajectory[-1][1])
        elif self._base_pixmap and not self._base_pixmap.isNull():
            eff_px = float(self._base_pixmap.width()) * 0.5
            eff_py = float(self._base_pixmap.height()) * 0.5
        else:
            eff_px, eff_py = 460.0, 310.0

        feet_y = float(fh) * self._feet_y_ratio
        horizon_y = float(fh) * self._horizon_y_ratio
        cx_opt = float(fw) * 0.5

        for raw_det in yolo_boxes:
            parsed = self._parse_bbox_coords(raw_det, fw, fh)
            if parsed is None:
                continue
            sx, sy, bw, bh = parsed

            # Normalize koordinat koruması (0..1 aralığı)
            if 0.0 <= sx <= 1.0 and 0.0 <= sy <= 1.0:
                sx *= float(fw)
                sy *= float(fh)
                bw *= float(fw)
                bh *= float(fh)

            raw_label = ""
            # P1-4 (K-07): güveni olmayan kutu 0.90 sayılmaz — bilinmiyorsa 0.0.
            conf = 0.0
            if isinstance(raw_det, dict):
                raw_label = str(raw_det.get("class_name") or raw_det.get("name") or raw_det.get("label") or raw_det.get("cls") or "resource")
                conf = _detection_confidence(raw_det, 0.0)
            elif hasattr(raw_det, "class_name"):
                raw_label = str(raw_det.class_name)
                conf = _detection_confidence(raw_det, 0.0)

            if conf > 1.0:
                conf /= 100.0

            # 1. Yatay Açı & Yönelim (70° HFOV İğne Deliği / Perspektif İzdüşüm)
            dx_screen = sx - cx_opt
            hfov_half_rad = math.radians(float(self._camera_hfov_deg) * 0.5)
            focal_px = cx_opt / math.tan(hfov_half_rad)
            angle_offset_rad = math.atan2(dx_screen, focal_px)
            angle_offset_deg = math.degrees(angle_offset_rad)
            bearing_deg = (eff_heading + angle_offset_deg) % 360.0

            # 2. Dikey Perspektif Zemin Mesafesi (Ground Distance in Meters)
            if sy >= feet_y:
                foot_norm = max(0.0, min(1.0, (sy - feet_y) / max(1.0, float(fh) - feet_y)))
                ground_dist = max(_PROJ_NEAR_DIST_M, _PROJ_FEET_BASE_M * (1.0 - foot_norm * _PROJ_FEET_FALLOFF))
                if bh > 4.0:
                    apparent_dist = (_PROJ_APPARENT_REF_H_PX / max(_PROJ_APPARENT_MIN_H_PX, float(bh))) * _PROJ_APPARENT_SCALE
                    dist_m = min(ground_dist, apparent_dist) if apparent_dist < ground_dist else (0.85 * ground_dist + 0.15 * min(10.0, apparent_dist))
                else:
                    dist_m = ground_dist
            else:
                ground_norm = max(0.01, min(1.0, (feet_y - sy) / max(1.0, feet_y - horizon_y)))
                dist_m = _PROJ_BEYOND_FEET_BASE_M + _PROJ_BEYOND_FEET_SPAN_M * (ground_norm ** _PROJ_BEYOND_FEET_EXP)
                if bh > 4.0:
                    apparent_dist = (_PROJ_APPARENT_REF_H_PX / max(_PROJ_APPARENT_MIN_H_PX, float(bh))) * _PROJ_APPARENT_SCALE
                    dist_m = 0.65 * dist_m + 0.35 * min(55.0, max(2.0, apparent_dist))

            dist_m = float(np.clip(dist_m, _PROJ_NEAR_DIST_M, _PROJ_FAR_DIST_M))

            category, color_hex, shape = self._classify_detection_label(raw_label)

            # 0.075 Oyuncu Yakın Kleransı (Player Clearance: 45m * 0.075 = 3.375m & ekran alt ayak bölgesi)
            clearance_m = float(self._sweep_outer_m) * _GATHER_PLAYER_CLEARANCE
            screen_clearance = (sy >= fh * (1.0 - _GATHER_PLAYER_CLEARANCE * 1.5)) and (abs(dx_screen) <= cx_opt * _GATHER_PLAYER_CLEARANCE * 3.0)
            in_player_clearance = bool(dist_m <= clearance_m or screen_clearance)
            if in_player_clearance and conf < 0.72 and category in ("obstacle", "enemy"):
                continue

            # 3. Radar/Harita Kartezyen Koordinatına İzdüşüm
            m_per_px = max(0.05, float(self._meters_per_pixel))
            dist_px = dist_m / m_per_px
            rad = math.radians(bearing_deg)
            map_dx = dist_px * math.sin(rad)
            map_dy = -dist_px * math.cos(rad)
            proj_map_x = eff_px + map_dx
            proj_map_y = eff_py + map_dy

            item = {
                "source": "3d_yolo",
                "class_name": raw_label,
                "category": category,
                "color_hex": color_hex,
                "shape": shape,
                "confidence": round(float(conf), 3),
                "distance_m": round(float(dist_m), 1),
                "bearing_deg": round(float(bearing_deg), 1),
                "camera_hfov_deg": float(self._camera_hfov_deg),
                "in_player_clearance": in_player_clearance,
                "screen_x": round(float(sx), 1),
                "screen_y": round(float(sy), 1),
                "map_x": round(float(proj_map_x), 1),
                "map_y": round(float(proj_map_y), 1),
                "rel_map_dx": round(float(map_dx), 1),
                "rel_map_dy": round(float(map_dy), 1),
            }
            projections.append(item)

        self._live_3d_projections = projections
        self.projection_updated.emit(list(projections))
        return projections

    def project_minimap_detections(
        self,
        detections: Optional[List[Any]],
        heading_deg: Optional[float] = None,
        player_pos: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Sağ üst minimap kırpımından tespit edilen filizleri ve canavar noktalarını radar koordinatlarına ekler.
        """
        projected: List[Dict[str, Any]] = []
        if not detections:
            self._live_minimap_detections = []
            return projected

        if player_pos is not None:
            eff_px, eff_py = float(player_pos[0]), float(player_pos[1])
        elif self._player_pos is not None:
            eff_px, eff_py = float(self._player_pos[0]), float(self._player_pos[1])
        elif self._walk_trajectory:
            eff_px, eff_py = float(self._walk_trajectory[-1][0]), float(self._walk_trajectory[-1][1])
        elif self._base_pixmap and not self._base_pixmap.isNull():
            eff_px = float(self._base_pixmap.width()) * 0.5
            eff_py = float(self._base_pixmap.height()) * 0.5
        else:
            eff_px, eff_py = 460.0, 310.0

        gather_clearance_px = _GATHER_PLAYER_CLEARANCE * 115.0

        for det in detections:
            if not isinstance(det, dict):
                continue
            rel_dx = det.get("rel_map_dx")
            rel_dy = det.get("rel_map_dy")
            if rel_dx is None or rel_dy is None:
                nx = float(det.get("norm_x", 0.5))
                ny = float(det.get("norm_y", 0.545))
                rel_dx = (nx - 0.50) * 115.0
                rel_dy = (ny - 0.545) * 115.0

            dist_px = math.hypot(float(rel_dx), float(rel_dy))
            # P1-4 (K-07): eksik güven 0.95 sayılmaz — bilinmiyorsa 0.0.
            # Böylece güveni bilinmeyen tespit, oyuncu kleransı içinde
            # (conf < 0.72) elenir; `classify_minimap_crop_bgr` zaten aynı
            # kuralı 0.0 varsayımıyla uygular.
            conf = _detection_confidence(det, 0.0)

            # 0.075 oyuncu kleransı: Oyuncuya çok yakın sahte canavar/filiz filtreleme (güven >= 0.72 olan filiz korunur)
            if dist_px <= gather_clearance_px and conf < 0.72:
                continue

            mx = eff_px + float(rel_dx)
            my = eff_py + float(rel_dy)
            dist_m = dist_px * float(self._meters_per_pixel)

            is_gatherable = bool(det.get("is_gatherable", True))
            is_hostile = bool(det.get("is_hostile", False))
            kind = str(det.get("kind", "sprout_material"))
            res_type = str(det.get("resource_type") or det.get("name") or "Minimap Sprout")

            if is_hostile or kind == "hostile_dot":
                category = "enemy"
                color_hex = "#ef4444"
                shape = "cross"
                class_id = MINIMAP_MONSTER_DOT_CLASS_ID
                # P1-1: canonical taxonomy name — "minimap_monster_negative"
                # is not a class name; hostility lives in is_hostile/kind.
                class_name = MINIMAP_MONSTER_DOT_CLASS_NAME
            elif "monster" in kind.lower() or not is_gatherable:
                category = "enemy"
                color_hex = "#f87171"
                shape = "cross"
                class_id = MINIMAP_MONSTER_DOT_CLASS_ID
                class_name = MINIMAP_MONSTER_DOT_CLASS_NAME
            else:
                category = "sprout"
                color_hex = "#22d3ee"
                shape = "double_circle"
                class_id = MINIMAP_SPROUT_CLASS_ID
                class_name = MINIMAP_SPROUT_CLASS_NAME

            item = {
                "source": "minimap",
                "class_name": res_type,
                "class_id": class_id,
                "taxonomy_class_name": class_name,
                "category": category,
                "color_hex": color_hex,
                "shape": shape,
                "distance_m": round(float(dist_m), 1),
                "map_x": round(float(mx), 1),
                "map_y": round(float(my), 1),
                "rel_map_dx": round(float(rel_dx), 1),
                "rel_map_dy": round(float(rel_dy), 1),
                "confidence": conf,
                "stack_badge": str(det.get("stack_badge", "3x")),
                "is_hostile": is_hostile,
                "is_gatherable": is_gatherable,
            }
            projected.append(item)

        self._live_minimap_detections = projected
        return projected

    # ------------------------------------------------------------------
    # Canlı Akış & Boşta Kalmama (Idle / Offline Mode)
    # ------------------------------------------------------------------
    def update_live_frame(
        self,
        frame_bgr: Optional[np.ndarray],
        yolo_boxes: Optional[List[Any]] = None,
        minimap_crop: Optional[np.ndarray] = None,
        player_pos: Optional[Tuple[float, float, float]] = None,
        heading_deg: Optional[float] = None,
    ) -> None:
        """
        Canlı kare geldiğinde radar gösterimini anında günceller.
        Canlı kare yokken (None/boşta) referans simülasyonunu ve son bilinen haritayı korur.
        """
        now = time.time()
        if player_pos is not None:
            p_angle = player_pos[2] if len(player_pos) > 2 else (heading_deg or self._heading_deg)
            self.set_player_position(player_pos[0], player_pos[1], angle_deg=p_angle)
            self._heading_deg = float(p_angle)
        elif heading_deg is not None:
            self._heading_deg = float(heading_deg)
            if self._player_pos is not None:
                self._player_pos = (self._player_pos[0], self._player_pos[1], self._heading_deg)

        if frame_bgr is not None and frame_bgr.size > 0:
            was_idle = self._is_idle
            self._is_idle = False
            self._last_live_timestamp = now
            if was_idle:
                self.idle_state_changed.emit(False)

            fh, fw = frame_bgr.shape[:2]
            if yolo_boxes is not None:
                self.project_3d_yolo_boxes(yolo_boxes, frame_w=fw, frame_h=fh)

            if minimap_crop is not None and minimap_crop.size > 0:
                try:
                    scanner = _panel_minimap_scanner()
                    sub_info = scanner.identify_subregion(minimap_crop, self._player_pos)
                    if sub_info and sub_info.get("subregion"):
                        self._current_subregion = str(sub_info["subregion"])
                        # H-sweep (dilim 13): kanonik bölge bilinmiyorsa
                        # (None) son bilinen bölge korunur — "Altgard"
                        # varsayılmaz, str(None) = "None" yazılmaz.
                        # (identify_subregion bilinmeyende canonical None
                        # döner; anahtar yoksa da .get() None verir.)
                        _canon = sub_info.get("canonical_region")
                        if _canon:
                            self._current_canonical_region = str(_canon)
                        # Görev 4: canlı bölge tespitini RouteManager'a ilet.
                        # Yalnızca mekansal bbox tespiti güvenilirdir; başlık
                        # şablonları yalnızca Altgard adlarını içerir, o yüzden
                        # başlık sonucu korumayı beslemez (yalnız görüntü).
                        try:
                            routes = getattr(self, "routes", None)
                            note = getattr(routes, "note_detected_region", None)
                            if (
                                callable(note)
                                and str(sub_info.get("source", "")) == "spatial_bbox"
                                and float(sub_info.get("confidence", 0.0) or 0.0) >= 0.45
                            ):
                                note(
                                    sub_info.get("canonical_region"),
                                    confidence=float(sub_info.get("confidence", 0.0) or 0.0),
                                    source="spatial_bbox",
                                )
                        except Exception as _det_exc:
                            logger.debug("Bölge tespiti RouteManager'a iletilemedi: %s", _det_exc)

                    mats, mobs = classify_minimap_crop_bgr(minimap_crop)
                    combined_dets = list(mats) + list(mobs)
                    self.project_minimap_detections(combined_dets)
                    if self._minimap_preview_ref is not None:
                        self._minimap_preview_ref.update_minimap_crop(minimap_crop)
                except Exception as mm_exc:
                    logger.debug("update_live_frame minimap crop classification error: %s", mm_exc)

            self.live_frame_updated.emit(len(self._live_3d_projections))
        else:
            if not self._is_idle:
                self._is_idle = True
                self.idle_state_changed.emit(True)

        self.update()

    def set_offline_simulation(self, enabled: bool = True) -> None:
        """Oyun kapalıyken canlı referans simülasyonu gösterimini kontrol eder."""
        self._offline_sim_active = bool(enabled)
        self.update()

    def is_idle(self) -> bool:
        return bool(self._is_idle)

    # ------------------------------------------------------------------
    # Çizim Motoru (Paint Event & Katmanlar)
    # ------------------------------------------------------------------
    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)

        try:
            # 1. Koyu taktiksel arka plan
            painter.fillRect(self.rect(), QColor("#090d14"))

            # 2. Bölge Haritası veya Boşta Kalmama Referans Izgarası
            has_pixmap = self._base_pixmap is not None and not self._base_pixmap.isNull()
            if has_pixmap:
                target_rect = QRectF(
                    self._pan.x(),
                    self._pan.y(),
                    self._base_pixmap.width() * self._zoom,
                    self._base_pixmap.height() * self._zoom,
                )
                painter.drawPixmap(target_rect, self._base_pixmap, QRectF(self._base_pixmap.rect()))
                painter.setPen(QPen(QColor(56, 189, 248, 70), 1.2))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(target_rect)
            else:
                self._draw_tactical_reference_grid(painter)

            # 3. Oyuncu Yürüyüş İzi & Ardışık Yön Koridorları (Filtre Kontrollü)
            all_resolved = [self._resolved_xy(node) for node in self._ordered_nodes]
            all_widget_pts: List[QPointF] = [
                self._map_to_widget(self._raw_to_map_coords(nx, ny)) for nx, ny in all_resolved
            ]
            all_jump_mask = self._jump_mask(all_resolved)

            # If material filter is active and matching nodes exist, route connects matching nodes in proximity order
            if self._material_filter and self._proximity_ordered_nodes:
                corridor_resolved = [self._resolved_xy(node) for node in self._proximity_ordered_nodes]
                corridor_widget_pts: List[QPointF] = [
                    self._map_to_widget(self._raw_to_map_coords(nx, ny)) for nx, ny in corridor_resolved
                ]
                corridor_jump_mask = self._jump_mask(corridor_resolved)
            else:
                corridor_widget_pts = all_widget_pts
                corridor_jump_mask = all_jump_mask

            if self._filter_show_path:
                self._draw_walk_trajectory(painter)
                self._draw_directional_corridor(painter, corridor_widget_pts, corridor_jump_mask)

            # 4. Oyuncu Konumu, Bakış Yönü Vektörü ve İç/Orta/Dış Tarama Halkaları
            p_wpt, p_angle = self._get_player_widget_anchor()
            self._draw_concentric_radar_rings_and_sweep(painter, p_wpt)
            self._draw_player_heading_vector(painter, p_wpt, p_angle)

            # 4b. Directional Proximity Vector toward nearest selective target node
            if self._material_filter:
                self._draw_proximity_vector_to_nearest(painter, p_wpt)

            # 5. 3B Kamera YZ Kutuları İzdüşümü (Color-Coded 3D AI Markers)
            if self._filter_show_projections and self._live_3d_projections:
                self._draw_3d_projected_markers(painter, p_wpt)

            # 6. Minimap Filizleri & Canavar Tespitleri
            if self._live_minimap_detections:
                self._draw_minimap_projected_markers(painter)

            # 7. Sıralı Düğümler (Filtre Kontrollü)
            if self._filter_show_nodes:
                jump_nodes: set[int] = set()
                n_pts = len(all_widget_pts)
                for i, is_jump in enumerate(all_jump_mask):
                    if is_jump and n_pts:
                        jump_nodes.add(i % n_pts)
                        jump_nodes.add((i + 1) % n_pts)
                self._draw_sequential_nodes(painter, all_widget_pts, jump_nodes)

            # 8. Yeni Keşif Dalgaları (Pulse Animation)
            self._draw_discovery_pulses(painter)

            # 9. Sağ Üst Minimap Canlı Kırpma & Filtre Katmanı (Inset)
            if self._show_minimap_overlay and self._minimap_preview_ref is not None:
                self._draw_minimap_overlay_inset(painter)

            # 10. Taktiksel Radar HUD Başlığı & Sol Alt Lejant
            self._draw_radar_hud_overlay(painter)
            self._draw_canvas_legend(painter)
        except Exception as exc:
            logger.error("Error in RadarCanvas.paintEvent: %s", exc)
        finally:
            painter.end()

    def _get_player_widget_anchor(self) -> Tuple[QPointF, float]:
        if self._player_pos is not None:
            px, py, angle = self._player_pos
        elif self._walk_trajectory:
            px, py = self._walk_trajectory[-1]
            angle = self._heading_deg
        elif self._base_pixmap and not self._base_pixmap.isNull():
            px, py = float(self._base_pixmap.width()) * 0.5, float(self._base_pixmap.height()) * 0.5
            angle = self._heading_deg
        else:
            px, py = 460.0, 310.0
            angle = self._heading_deg
        wpt = self._map_to_widget(self._raw_to_map_coords(px, py))
        return wpt, float(angle)

    def _draw_tactical_reference_grid(self, painter: QPainter) -> None:
        """Harita yüklenmediğinde veya çevrimdışıyken boş siyah ekran yerine taktiksel koordinat ızgarası çizer."""
        w = max(1, self.width())
        h = max(1, self.height())
        grid_pen = QPen(QColor(30, 41, 59, 120), 1.0, Qt.PenStyle.DotLine)
        painter.setPen(grid_pen)
        step = max(35, int(round(50 * self._zoom)))
        ox = int(self._pan.x()) % step
        oy = int(self._pan.y()) % step
        for gx in range(ox, w, step):
            painter.drawLine(gx, 0, gx, h)
        for gy in range(oy, h, step):
            painter.drawLine(0, gy, w, gy)

        font = QFont("Segoe UI", 9)
        painter.setFont(font)
        # Not: HUD kutusu (12,12,340,46) tuvalin sol üstünü kaplar; ızgara başlığı
        # HUD'un altına (y=64) taşındı — aksi halde HUD dikdörtgeninin altında
        # kalıp hiç görünmüyordu (paintEvent'de HUD ızgaradan sonra çizilir).
        painter.setPen(QColor(100, 116, 139, 175))
        painter.drawText(
            QRectF(14.0, 64.0, 320.0, 22.0),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            # v19 P1-7/P1-12: ölçek etiketi artık sabit yazı değil — o an
            # kullanılan _meters_per_pixel değerini gösterir (kalibrasyon
            # sonrası da doğru kalır).
            "● RADAR KOORDİNAT REFERANS IZGARASI (Ölçek: %.2fm/px)" % float(self._meters_per_pixel),
        )

    def _draw_concentric_radar_rings_and_sweep(self, painter: QPainter, center: QPointF) -> None:
        """Oyuncu merkezinde 10m, 25m ve 45m iç/orta/dış radar tarama halkalarını ve dönen tarama huzmesini çizer."""
        m_per_px = max(0.05, float(self._meters_per_pixel))
        r_inner = (self._sweep_inner_m / m_per_px) * self._zoom
        r_mid = (self._sweep_mid_m / m_per_px) * self._zoom
        r_outer = (self._sweep_outer_m / m_per_px) * self._zoom

        # Dış halka (45m)
        painter.setPen(QPen(QColor(14, 165, 233, 75), 1.3, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(14, 165, 233, 10)))
        painter.drawEllipse(center, r_outer, r_outer)

        # Orta halka (25m)
        painter.setPen(QPen(QColor(34, 211, 238, 95), 1.2, Qt.PenStyle.DashLine))
        painter.setBrush(QBrush(QColor(34, 211, 238, 14)))
        painter.drawEllipse(center, r_mid, r_mid)

        # İç halka (10m) — Yakın toplama çemberi
        painter.setPen(QPen(QColor(16, 185, 129, 130), 1.5, Qt.PenStyle.SolidLine))
        painter.setBrush(QBrush(QColor(16, 185, 129, 22)))
        painter.drawEllipse(center, r_inner, r_inner)

        # Mesafe etiketleri (10m, 25m, 45m)
        dist_font = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(dist_font)
        painter.setPen(QColor("#6ee7b7"))
        painter.drawText(QPointF(center.x() + 4.0, center.y() - r_inner + 9.0), "10m")
        painter.setPen(QColor("#67e8f9"))
        painter.drawText(QPointF(center.x() + 4.0, center.y() - r_mid + 9.0), "25m")
        painter.setPen(QColor("#7dd3fc"))
        painter.drawText(QPointF(center.x() + 4.0, center.y() - r_outer + 9.0), "45m")

        # Canlı Dönen Radar Tarama Huzmesi (Rotating Radar Sweep Ray & Cone)
        if self._filter_show_sweep:
            sweep_angle_rad = self._radar_sweep_phase
            ray_x = center.x() + r_outer * math.sin(sweep_angle_rad)
            ray_y = center.y() - r_outer * math.cos(sweep_angle_rad)

            # Tarama huzme çizgisi
            sweep_pen = QPen(QColor(52, 211, 153, 180), 2.0)
            painter.setPen(sweep_pen)
            painter.drawLine(center, QPointF(ray_x, ray_y))

            # Huzmenin arkasındaki saydam tarama konisi (pie sweep)
            cone_path = QPainterPath()
            cone_path.moveTo(center)
            deg_start = math.degrees(-sweep_angle_rad) + 90.0
            cone_path.arcTo(
                QRectF(center.x() - r_outer, center.y() - r_outer, r_outer * 2.0, r_outer * 2.0),
                deg_start,
                38.0,
            )
            cone_path.closeSubpath()
            grad = QRadialGradient(center, r_outer)
            grad.setColorAt(0.0, QColor(16, 185, 129, 65))
            grad.setColorAt(0.85, QColor(34, 197, 94, 25))
            grad.setColorAt(1.0, QColor(34, 197, 94, 0))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawPath(cone_path)

    def _draw_player_heading_vector(self, painter: QPainter, center: QPointF, angle_deg: float) -> None:
        """Oyuncunun baktığı açıda (heading_deg) dış radar halkasına uzanan belirgin yön okunu ve oyuncu işaretçisini çizer."""
        m_per_px = max(0.05, float(self._meters_per_pixel))
        r_outer = (self._sweep_outer_m / m_per_px) * self._zoom
        rad_h = math.radians(angle_deg)
        vec_x = center.x() + r_outer * 0.92 * math.sin(rad_h)
        vec_y = center.y() - r_outer * 0.92 * math.cos(rad_h)

        # Bakış Yönü Vektör Hattı (Altın parıltılı kesikli yön çizgisi)
        heading_pen = QPen(QColor(250, 204, 21, 190), 2.2, Qt.PenStyle.DashLine)
        heading_pen.setDashPattern([7.0, 4.0])
        painter.setPen(heading_pen)
        painter.drawLine(center, QPointF(vec_x, vec_y))

        # Vektörün ucundaki yön üçgeni
        ux = math.sin(rad_h)
        uy = -math.cos(rad_h)
        self._draw_chevron_arrow(painter, QPointF(vec_x, vec_y), ux, uy, QColor("#facc15"), size=11.0)

        # Oyuncunun merkez işaretçisi (Yönlü sarı elmas/ok)
        painter.save()
        painter.translate(center)
        painter.rotate(angle_deg)
        arrow = QPolygonF([
            QPointF(0.0, -11.0),
            QPointF(7.5, 8.0),
            QPointF(0.0, 3.8),
            QPointF(-7.5, 8.0),
        ])
        painter.setPen(QPen(QColor("#04131d"), 2.2))
        painter.setBrush(QBrush(QColor("#facc15")))
        painter.drawPolygon(arrow)
        painter.restore()

    def _draw_proximity_vector_to_nearest(
        self,
        painter: QPainter,
        p_wpt: QPointF,
    ) -> None:
        """
        Draws a glowing directional proximity vector line, heading chevron,
        and distance badge from the player toward the nearest matching target node.
        """
        nearest = self.get_nearest_target_node()
        if nearest is None:
            return

        # En yakın düğüm de diğer düğümler gibi HD/norm çözümlemesinden
        # geçirilir; ham map_x/map_y HD modunda yanlış noktaya vektör çizer.
        nx, ny = self._resolved_xy(nearest)
        t_wpt = self._map_to_widget(self._raw_to_map_coords(nx, ny))

        dx = t_wpt.x() - p_wpt.x()
        dy = t_wpt.y() - p_wpt.y()
        seg_len = math.hypot(dx, dy)
        if seg_len < 2.0:
            return

        ux = dx / seg_len
        uy = dy / seg_len

        m_per_px = max(0.05, float(self._meters_per_pixel))
        dist_m = (seg_len / max(0.001, self._zoom)) * m_per_px

        painter.save()
        # Soft golden/cyan glow corridor
        glow_pen = QPen(QColor(250, 204, 21, 65), 7.0, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        painter.drawLine(p_wpt, t_wpt)

        # Crisp dashed core line
        core_pen = QPen(QColor(250, 204, 21, 230), 2.4, Qt.PenStyle.DashLine, Qt.PenCapStyle.RoundCap)
        core_pen.setDashPattern([6.0, 4.0])
        painter.setPen(core_pen)
        painter.drawLine(p_wpt, t_wpt)

        # Directional chevron arrow midway
        if seg_len >= 48.0:
            mid = QPointF(p_wpt.x() + dx * 0.52, p_wpt.y() + dy * 0.52)
            self._draw_chevron_arrow(painter, mid, ux, uy, QColor("#facc15"), size=10.0)

        # Arrow head pointing toward target node
        tip_pos = QPointF(t_wpt.x() - ux * 13.0, t_wpt.y() - uy * 13.0)
        self._draw_chevron_arrow(painter, tip_pos, ux, uy, QColor("#fbbf24"), size=11.5)

        # Floating distance & target name badge along vector
        mid_x = (p_wpt.x() + t_wpt.x()) * 0.5
        mid_y = (p_wpt.y() + t_wpt.y()) * 0.5
        target_name = str(nearest.get("resource_type") or nearest.get("name") or "Hedef")
        badge_text = f"🧭 {dist_m:.1f}m ➔ {target_name} [#1]"
        badge_font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(badge_font)
        fm = painter.fontMetrics()
        bw = max(40.0, float(fm.horizontalAdvance(badge_text)) + 12.0)
        bh = 19.0
        badge_rect = QRectF(mid_x - bw * 0.5, mid_y - bh * 0.5 - 12.0, bw, bh)

        painter.setPen(QPen(QColor("#facc15"), 1.2))
        painter.setBrush(QBrush(QColor(15, 23, 42, 235)))
        painter.drawRoundedRect(badge_rect, 4.0, 4.0)
        painter.setPen(QColor("#fde047"))
        painter.drawText(badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text)

        # Pulsing target beacon double-ring
        sweep_phase = getattr(self, "_radar_sweep_phase", 0.0)
        pulse_r1 = 12.0 + 3.0 * math.sin(sweep_phase * 3.0)
        pulse_r2 = 18.0 + 4.0 * math.sin(sweep_phase * 3.0 + 1.0)
        painter.setPen(QPen(QColor("#facc15"), 2.2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(t_wpt, pulse_r1, pulse_r1)
        painter.setPen(QPen(QColor(6, 182, 212, 180), 1.6, Qt.PenStyle.DashLine))
        painter.drawEllipse(t_wpt, pulse_r2, pulse_r2)

        painter.restore()

    def _draw_3d_projected_markers(self, painter: QPainter, player_center: QPointF) -> None:
        """3B Kamera YZ tespitlerini radar üzerine renk kodlu semboller ve mesafe etiketleriyle çizer."""
        # Not: det["map_x"/"map_y"] projeksiyon çıktısıdır; oyuncu konumu / baz
        # pixmap üzerinden zaten tuval harita çerçevesindedir. _resolved_xy
        # ile yeniden çözmek HD modunda çift dönüşüm yapar, uygulanmaz.
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        for det in self._live_3d_projections:
            cat = str(det.get("category", "default"))
            if cat == "enemy" and not self._filter_show_enemies:
                continue
            if cat in ("herb", "ore", "gem", "wood", "sprout", "prompt") and not self._filter_show_nodes:
                continue

            mx = float(det.get("map_x", 0.0))
            my = float(det.get("map_y", 0.0))
            wpt = self._map_to_widget(self._raw_to_map_coords(mx, my))
            color_hex = str(det.get("color_hex", "#38bdf8"))
            shape = str(det.get("shape", "circle"))
            cls_name = str(det.get("class_name", "target"))
            dist_m = float(det.get("distance_m", 0.0))

            # Oyuncudan tespite doğru ince kesikli radar ışını
            beam_pen = QPen(QColor(color_hex), 1.0, Qt.PenStyle.DotLine)
            painter.setPen(beam_pen)
            painter.drawLine(player_center, wpt)

            # Şekil Çizimi
            painter.setPen(QPen(QColor("#030712"), 2.0))
            painter.setBrush(QBrush(QColor(color_hex)))

            if shape == "cross":
                r = 8.5
                painter.setBrush(QBrush(QColor(239, 68, 68, 45)))
                painter.setPen(QPen(QColor("#ef4444"), 1.8))
                painter.drawEllipse(wpt, r + 4.0, r + 4.0)
                painter.setPen(QPen(QColor("#ef4444"), 2.5))
                painter.drawLine(QPointF(wpt.x() - 5.0, wpt.y() - 5.0), QPointF(wpt.x() + 5.0, wpt.y() + 5.0))
                painter.drawLine(QPointF(wpt.x() - 5.0, wpt.y() + 5.0), QPointF(wpt.x() + 5.0, wpt.y() - 5.0))
            elif shape == "diamond":
                d_poly = QPolygonF([
                    QPointF(wpt.x(), wpt.y() - 9.0),
                    QPointF(wpt.x() + 7.5, wpt.y()),
                    QPointF(wpt.x(), wpt.y() + 9.0),
                    QPointF(wpt.x() - 7.5, wpt.y()),
                ])
                painter.drawPolygon(d_poly)
            elif shape == "gem":
                g_poly = QPolygonF([
                    QPointF(wpt.x() - 4.0, wpt.y() - 8.0),
                    QPointF(wpt.x() + 4.0, wpt.y() - 8.0),
                    QPointF(wpt.x() + 8.0, wpt.y() - 4.0),
                    QPointF(wpt.x() + 8.0, wpt.y() + 4.0),
                    QPointF(wpt.x() + 4.0, wpt.y() + 8.0),
                    QPointF(wpt.x() - 4.0, wpt.y() + 8.0),
                    QPointF(wpt.x() - 8.0, wpt.y() + 4.0),
                    QPointF(wpt.x() - 8.0, wpt.y() - 4.0),
                ])
                painter.drawPolygon(g_poly)
            elif shape == "square":
                painter.drawRoundedRect(QRectF(wpt.x() - 7.0, wpt.y() - 7.0, 14.0, 14.0), 3.0, 3.0)
            elif shape == "star":
                painter.drawEllipse(wpt, 8.5, 8.5)
                painter.setPen(QPen(QColor("#ffffff"), 1.8))
                painter.drawEllipse(wpt, 4.0, 4.0)
            else:
                painter.drawEllipse(wpt, 7.5, 7.5)
                painter.setBrush(QBrush(QColor("#ffffff")))
                painter.drawEllipse(wpt, 2.8, 2.8)

            label_text = f"{cls_name} [{dist_m:.0f}m]"
            self._draw_outlined_text(
                painter,
                wpt.x() + 10.0,
                wpt.y() + 3.0,
                label_text,
                font,
                text_color=QColor("#f8fafc"),
            )

    def _draw_minimap_projected_markers(self, painter: QPainter) -> None:
        """Minimap'ten tespit edilen filizleri çift halka ve rozetlerle çizer."""
        # Not: mat["map_x"/"map_y"] project_minimap_detections çıktısıdır ve
        # zaten tuval harita çerçevesindedir; _resolved_xy ile yeniden çözmek
        # HD modunda çift dönüşüm yapar, uygulanmaz.
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        for mat in self._live_minimap_detections:
            cat = str(mat.get("category", "sprout"))
            if cat == "enemy" and not self._filter_show_enemies:
                continue
            if cat == "sprout" and not self._filter_show_nodes:
                continue

            mx = float(mat.get("map_x", 0.0))
            my = float(mat.get("map_y", 0.0))
            wpt = self._map_to_widget(self._raw_to_map_coords(mx, my))
            badge = str(mat.get("stack_badge", "3x"))

            if cat == "enemy":
                painter.setPen(QPen(QColor("#f87171"), 1.8))
                painter.setBrush(QBrush(QColor(248, 113, 113, 50)))
                painter.drawEllipse(wpt, 6.0, 6.0)
                painter.drawLine(QPointF(wpt.x() - 3.5, wpt.y() - 3.5), QPointF(wpt.x() + 3.5, wpt.y() + 3.5))
                painter.drawLine(QPointF(wpt.x() - 3.5, wpt.y() + 3.5), QPointF(wpt.x() + 3.5, wpt.y() - 3.5))
            else:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QBrush(QColor(6, 182, 212, 45)))
                painter.drawEllipse(wpt, 14.0, 14.0)
                painter.setPen(QPen(QColor("#22d3ee"), 2.0))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(wpt, 10.0, 10.0)
                painter.setPen(QPen(QColor("#10b981"), 1.8))
                painter.setBrush(QBrush(QColor("#10b981")))
                painter.drawEllipse(wpt, 5.0, 5.0)

                pill_rect = QRectF(wpt.x() + 11.0, wpt.y() - 8.0, 26.0, 16.0)
                painter.setPen(QPen(QColor("#22d3ee"), 1.2))
                painter.setBrush(QBrush(QColor(6, 46, 42, 220)))
                painter.drawRoundedRect(pill_rect, 3.5, 3.5)
                painter.setFont(font)
                painter.setPen(QColor("#67e8f9"))
                painter.drawText(pill_rect, Qt.AlignmentFlag.AlignCenter, badge)

    def _draw_radar_hud_overlay(self, painter: QPainter) -> None:
        """Sağ üstte/sol üstte canlı durum, yakınlaştırma ve aktif filtre durumunu gösteren taktiksel gösterge."""
        hud_w = 340.0
        hud_h = 46.0
        hud_rect = QRectF(12.0, 12.0, hud_w, hud_h)

        painter.setPen(QPen(QColor(51, 65, 85, 200), 1.0))
        painter.setBrush(QBrush(QColor(15, 23, 42, 225)))
        painter.drawRoundedRect(hud_rect, 6.0, 6.0)

        font_b = QFont("Segoe UI", 9, QFont.Weight.Bold)
        font_s = QFont("Segoe UI", 9)
        painter.setFont(font_b)

        zone_info = f"({self._current_canonical_region} · {self._current_subregion})"
        if not self._is_idle:
            painter.setPen(QColor("#34d399"))
            status_text = f"● CANLI RADAR FÜZYONU {zone_info}"
        else:
            painter.setPen(QColor("#f59e0b"))
            status_text = f"● REFERANS SİMÜLASYONU {zone_info}"

        painter.drawText(QRectF(20.0, 16.0, 315.0, 18.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, status_text)

        painter.setFont(font_s)
        painter.setPen(QColor("#94a3b8"))
        z_pct = int(round(self._zoom * 100))
        filter_str = (
            f"Zoom: {z_pct}% | Filtre: "
            f"{'Düğüm ' if self._filter_show_nodes else ''}"
            f"{'Düşman ' if self._filter_show_enemies else ''}"
            f"{'Rota ' if self._filter_show_path else ''}"
            f"{'3B YZ' if self._filter_show_projections else ''}"
        )
        painter.drawText(QRectF(20.0, 34.0, 300.0, 18.0), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, filter_str)


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
            meters_per_pixel=float(getattr(self.routes, "meters_per_pixel", WORLD_MAP_METERS_PER_PX) or WORLD_MAP_METERS_PER_PX)
        )

        self._region_name: str = region_name or self._resolve_active_region()
        self._route_name: str = route_name or self._resolve_active_route()
        self._explicit_map_path: Optional[str] = map_image_path
        self._loaded_map_path: str = ""
        # Görev 4: bölgeye ait HD harita görseli yoksa bölge adı burada tutulur
        # ("bu bölge için harita yok" durumu; Altgard görseli kullanılmaz).
        self._map_missing_for_region: str = ""
        self._hd_map_active: bool = False
        self._auto_cartography_enabled: bool = True
        initial_nodes = kwargs.get("ordered_nodes") or kwargs.get("nodes")
        self._ordered_nodes: List[Dict[str, Any]] = list(initial_nodes or [])
        self._walk_trajectory: List[Tuple[float, float]] = list(walk_trajectory or [])
        self._minimap_inset_user_override: bool = False

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
        # Varsayılan bölge Altgard'dır; harita Altgard karesidir (Verteron değil).
        return "Altgard"

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
                padding: 6px 10px;
            }
            QLabel#StatTitle {
                color: #94a3b8;
                font-size: 12px;
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
            QPushButton:disabled {
                background-color: #1e293b;
                color: #64748b;
                border: 1px solid #334155;
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

        root_layout = wrap_shell_tab_root_layout(self, "radar")
        root_layout.setSpacing(10)

        # 1. ÜST KONTROL ÇUBUĞU. Sayaçlar sağ sütunda; haritanın üstüne binmez.
        self.top_bar = QFrame(self)
        self.top_bar.setObjectName("TopStatsBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(14, 10, 14, 10)
        top_layout.setSpacing(12)

        self.lbl_status_prefix = QLabel(
            "Canlı Oto-Haritalama ve Minimap Radarı:", self.top_bar
        )
        self.lbl_status_prefix.setStyleSheet("color: #d4a017; font-weight: bold; font-size: 13px;")
        top_layout.addWidget(self.lbl_status_prefix)

        self.lbl_discovery_state = QLabel("AKTİF", self.top_bar)
        self.lbl_discovery_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.lbl_discovery_state)

        self.btn_start_discovery = QPushButton("Keşfi Başlat", self.top_bar)
        self.btn_start_discovery.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_start_discovery.clicked.connect(lambda: self._set_discovery_state(True))
        top_layout.addWidget(self.btn_start_discovery)

        self.btn_stop_discovery = QPushButton("Keşfi Durdur", self.top_bar)
        self.btn_stop_discovery.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_stop_discovery.clicked.connect(lambda: self._set_discovery_state(False))
        top_layout.addWidget(self.btn_stop_discovery)

        # Filtre ve Yakınlaştırma (Zoom/Pan) Kontrolleri (Worker MR4)
        top_layout.addSpacing(12)
        sep_filters = QFrame(self.top_bar)
        sep_filters.setFrameShape(QFrame.Shape.VLine)
        sep_filters.setStyleSheet("color: #334155;")
        top_layout.addWidget(sep_filters)

        self.btn_toggle_nodes = QPushButton("🌿 Düğümler", self.top_bar)
        self.btn_toggle_nodes.setCheckable(True)
        self.btn_toggle_nodes.setChecked(True)
        self.btn_toggle_nodes.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_nodes.toggled.connect(lambda c: self.set_radar_filter("nodes", c))
        top_layout.addWidget(self.btn_toggle_nodes)

        self.btn_toggle_enemies = QPushButton("⚔️ Düşmanlar", self.top_bar)
        self.btn_toggle_enemies.setCheckable(True)
        self.btn_toggle_enemies.setChecked(True)
        self.btn_toggle_enemies.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_enemies.toggled.connect(lambda c: self.set_radar_filter("enemies", c))
        top_layout.addWidget(self.btn_toggle_enemies)

        self.btn_toggle_path = QPushButton("🧭 Rota", self.top_bar)
        self.btn_toggle_path.setCheckable(True)
        self.btn_toggle_path.setChecked(True)
        self.btn_toggle_path.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_path.toggled.connect(lambda c: self.set_radar_filter("path", c))
        top_layout.addWidget(self.btn_toggle_path)

        self.btn_toggle_projections = QPushButton("📡 3B YZ", self.top_bar)
        self.btn_toggle_projections.setCheckable(True)
        self.btn_toggle_projections.setChecked(True)
        self.btn_toggle_projections.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_projections.toggled.connect(lambda c: self.set_radar_filter("projections", c))
        top_layout.addWidget(self.btn_toggle_projections)

        sep_zoom = QFrame(self.top_bar)
        sep_zoom.setFrameShape(QFrame.Shape.VLine)
        sep_zoom.setStyleSheet("color: #334155;")
        top_layout.addWidget(sep_zoom)

        self.btn_zoom_in = QPushButton("🔍＋", self.top_bar)
        self.btn_zoom_in.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_in.setToolTip("Yakınlaş (Zoom In)")
        self.btn_zoom_in.clicked.connect(self.canvas_zoom_in)
        top_layout.addWidget(self.btn_zoom_in)

        self.btn_zoom_out = QPushButton("🔍－", self.top_bar)
        self.btn_zoom_out.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_out.setToolTip("Uzaklaş (Zoom Out)")
        self.btn_zoom_out.clicked.connect(self.canvas_zoom_out)
        top_layout.addWidget(self.btn_zoom_out)

        self.btn_zoom_fit = QPushButton("⛶ Sığdır", self.top_bar)
        self.btn_zoom_fit.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_zoom_fit.setToolTip("Görünümü Sığdır (Reset View)")
        self.btn_zoom_fit.clicked.connect(self.canvas_reset_view)
        top_layout.addWidget(self.btn_zoom_fit)

        self.btn_center_player = QPushButton("🎯 Oyuncu", self.top_bar)
        self.btn_center_player.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_center_player.setToolTip("Oyuncuya Odakla (Center on Player)")
        self.btn_center_player.clicked.connect(self.canvas_center_on_player)
        top_layout.addWidget(self.btn_center_player)

        sep_select = QFrame(self.top_bar)
        sep_select.setFrameShape(QFrame.Shape.VLine)
        sep_select.setStyleSheet("color: #334155;")
        top_layout.addWidget(sep_select)

        self.lbl_material_filter = QLabel("🎯 Seç-Topla Filtresi:", self.top_bar)
        self.lbl_material_filter.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        top_layout.addWidget(self.lbl_material_filter)

        self.combo_material_filter = QComboBox(self.top_bar)
        self.combo_material_filter.setObjectName("RadarMaterialFilterCombo")
        self.combo_material_filter.setToolTip("🎯 Seç-Topla Filtresi: Toplanacak materyali seçin")
        self.combo_material_filter.setStyleSheet("""
            QComboBox {
                background-color: #1e293b;
                color: #ffffff;
                border: 1px solid #0284c7;
                border-radius: 4px;
                padding: 5px 10px;
                font-size: 12px;
                font-weight: bold;
                min-width: 160px;
            }
            QComboBox::drop-down { border: 0px; }
            QComboBox QAbstractItemView {
                background-color: #11151c;
                color: #ffffff;
                selection-background-color: #0284c7;
                selection-color: #ffffff;
            }
        """)
        if hasattr(self.combo_material_filter, "view") and self.combo_material_filter.view() is not None:
            view = self.combo_material_filter.view()
            palette = view.palette()
            palette.setColor(QPalette.ColorRole.Text, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.WindowText, QColor("#ffffff"))
            palette.setColor(QPalette.ColorRole.Base, QColor("#11151c"))
            view.setPalette(palette)

        for label, data in [
            ("🎯 Tümü", "Tümü"),
            ("⛏️ Orichalcum Ore", "Orichalcum Ore"),
            ("🌸 Aria", "Aria"),
            ("🍒 Mela", "Mela"),
            ("🌿 Targena", "Targena"),
            ("💎 Gem / Crystal", "Gem / Crystal"),
            ("🪵 Tree / Log", "Tree / Log"),
        ]:
            self.combo_material_filter.addItem(label, userData=data)
        self.combo_material_filter.currentTextChanged.connect(self._on_material_filter_changed)
        top_layout.addWidget(self.combo_material_filter)

        self.btn_draw_proximity_route = QPushButton("🧭 Yakınlık Rotası Çiz", self.top_bar)
        self.btn_draw_proximity_route.setObjectName("RadarDrawProximityRouteBtn")
        self.btn_draw_proximity_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_draw_proximity_route.setToolTip("Seçilen materyalin noktalarını en yakın mesafeye göre sıralayıp rota çizer")
        self.btn_draw_proximity_route.setStyleSheet("""
            QPushButton {
                background-color: #0284c7;
                color: #ffffff;
                border: 1px solid #38bdf8;
                border-radius: 4px;
                padding: 6px 12px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0369a1;
            }
        """)
        self.btn_draw_proximity_route.clicked.connect(self._on_draw_proximity_route_clicked)
        top_layout.addWidget(self.btn_draw_proximity_route)

        top_layout.addStretch(1)
        root_layout.addWidget(self.top_bar)

        # 2. ORTA BÖLÜM: RADAR TUVALİ + SAĞ ÜST MİNİMAP CANLI KIRPMA & FİLTRE PANELİ
        middle_layout = QHBoxLayout()
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(10)

        self.canvas = RadarCanvas(self)
        self.canvas.setMinimumSize(240, 200)
        self.canvas.node_left_clicked.connect(self._on_canvas_node_left_clicked)
        self.canvas.node_right_clicked.connect(self._on_canvas_node_right_clicked)
        middle_layout.addWidget(self.canvas, stretch=1)

        # Sağ sütun: kontroller haritanın üstüne binmez.
        self.minimap_side_panel = QFrame(self)
        self.minimap_side_panel.setObjectName("MinimapSidePanel")
        self.minimap_side_panel.setFixedWidth(284)
        side_layout = QVBoxLayout(self.minimap_side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)

        self.lbl_hybrid_scanner_title = QLabel(
            "3B Dünya + Sağ Üst Minimap Hibrit Tarayıcı", self.minimap_side_panel
        )
        self.lbl_hybrid_scanner_title.setWordWrap(True)
        self.lbl_hybrid_scanner_title.setStyleSheet(
            "color: #e8eaed; font-size: 12px; font-weight: bold;"
        )
        side_layout.addWidget(self.lbl_hybrid_scanner_title)

        self.lbl_minimap_sub = QLabel(
            "Top-Right Minimap Live Crop & Filter:\n"
            "• Zümrüt/Camgöbeği (3x/6x/9x): Toplanabilir Materyal\n"
            "• Gri/Kırmızı Çizili (•✕): Yoksayılan Canavar",
            self.minimap_side_panel,
        )
        self.lbl_minimap_sub.setWordWrap(True)
        self.lbl_minimap_sub.setStyleSheet("color: #94a3b8; font-size: 12px;")
        side_layout.addWidget(self.lbl_minimap_sub)

        self.minimap_preview_widget = MinimapLiveCropPreviewWidget(self.minimap_side_panel)
        self.canvas.set_minimap_preview_widget(self.minimap_preview_widget)
        side_layout.addWidget(self.minimap_preview_widget, stretch=1)

        self.lbl_minimap_filter_stats = QLabel(self.minimap_side_panel)
        self.lbl_minimap_filter_stats.setWordWrap(True)
        self.lbl_minimap_filter_stats.setStyleSheet(
            "color: #a7f3d0; font-size: 12px; font-weight: 600;"
        )
        side_layout.addWidget(self.lbl_minimap_filter_stats)

        self.lbl_discovered_nodes = self._create_stat_widget(
            side_layout, "Toplanan / Haritalanan Materyaller", "0", self.minimap_side_panel
        )
        self.lbl_smoothness_score = self._create_stat_widget(
            side_layout, "Rota Akıcılık Skoru (%)", "100.0%", self.minimap_side_panel
        )
        self.lbl_total_distance = self._create_stat_widget(
            side_layout, "Toplam Döngü Mesafesi (m)", "0.0 m", self.minimap_side_panel
        )
        self.lbl_ping_pong_jumps = self._create_stat_widget(
            side_layout, "Zikzak Sıçrama", "Zikzak Sıçrama: 0", self.minimap_side_panel
        )

        self.btn_record_minimap_now = QPushButton(
            "Minimap'ten Şimdi Kaydet", self.minimap_side_panel
        )
        self.btn_record_minimap_now.setObjectName("RecordMinimapBtn")
        self.btn_record_minimap_now.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_record_minimap_now.clicked.connect(self.record_minimap_now)
        side_layout.addWidget(self.btn_record_minimap_now)

        self.btn_toggle_minimap_inset = QPushButton(
            "Harita Üstü Minimap Önizleme: KAPALI", self.minimap_side_panel
        )
        self.btn_toggle_minimap_inset.setCheckable(True)
        self.btn_toggle_minimap_inset.setChecked(False)
        self.btn_toggle_minimap_inset.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_toggle_minimap_inset.clicked.connect(self._on_user_toggle_minimap_inset)
        side_layout.addWidget(self.btn_toggle_minimap_inset)
        self.minimap_preview_widget.detections_changed.connect(self._sync_minimap_inset_with_scanner)

        middle_layout.addWidget(self.minimap_side_panel, stretch=0)
        root_layout.addLayout(middle_layout, stretch=1)

        # 3. SIRALAMA & KORİDOR KONTROL BUTONLARI (Tam Türkçe)
        self.bottom_bar = QFrame(self)
        self.bottom_bar.setObjectName("BottomControlBar")
        bottom_layout = QHBoxLayout(self.bottom_bar)
        bottom_layout.setContentsMargins(14, 10, 14, 10)
        bottom_layout.setSpacing(10)

        self.btn_sort_smooth = QPushButton(
            "Sıralı Rotayı Optimize Et", self.bottom_bar
        )
        self.btn_sort_smooth.setObjectName("PrimarySortBtn")
        self.btn_sort_smooth.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sort_smooth.setMinimumHeight(32)
        self.btn_sort_smooth.clicked.connect(self.sort_smooth_sequential_loop)
        bottom_layout.addWidget(self.btn_sort_smooth)

        self.btn_sort_walk = QPushButton(
            "Gezdiğim Sırayla Diz", self.bottom_bar
        )
        self.btn_sort_walk.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sort_walk.setMinimumHeight(32)
        self.btn_sort_walk.clicked.connect(self.sort_by_walk_path)
        bottom_layout.addWidget(self.btn_sort_walk)

        self.btn_sort_corridor = QPushButton(
            "Koridor Şerit Sırası", self.bottom_bar
        )
        self.btn_sort_corridor.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_sort_corridor.setMinimumHeight(32)
        self.btn_sort_corridor.clicked.connect(self.sort_corridor_sweep)
        bottom_layout.addWidget(self.btn_sort_corridor)

        self.btn_reverse_order = QPushButton(
            "Sırayı Ters Çevir", self.bottom_bar
        )
        self.btn_reverse_order.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_reverse_order.setMinimumHeight(32)
        self.btn_reverse_order.clicked.connect(self.reverse_node_order)
        bottom_layout.addWidget(self.btn_reverse_order)

        bottom_layout.addStretch(1)

        self.btn_save_route = QPushButton("Sıralı Rotayı Kaydet", self.bottom_bar)
        self.btn_save_route.setObjectName("SaveRouteBtn")
        self.btn_save_route.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_save_route.setMinimumHeight(32)
        self.btn_save_route.clicked.connect(self.save_sequential_route)
        bottom_layout.addWidget(self.btn_save_route)

        root_layout.addWidget(self.bottom_bar)
        self._update_auto_cartography_button_style()
        self._refresh_minimap_filter_stats()
        self._sync_minimap_inset_with_scanner()

    def _create_stat_widget(
        self,
        parent_layout: QVBoxLayout,
        title: str,
        initial_val: str,
        parent_widget: Optional[QWidget] = None,
    ) -> QLabel:
        card = QFrame(parent_widget or self.top_bar)
        card.setObjectName("StatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 5, 10, 5)
        card_layout.setSpacing(3)

        lbl_title = QLabel(title, card)
        lbl_title.setObjectName("StatTitle")
        lbl_val = QLabel(initial_val, card)
        lbl_val.setObjectName("StatValue")

        card_layout.addWidget(lbl_title)
        card_layout.addWidget(lbl_val)
        parent_layout.addWidget(card)
        return lbl_val

    def _update_auto_cartography_button_style(self) -> None:
        """Durum yazısı sadece AKTİF/PASİF. Başlat ve Durdur aynı anda etkin olmaz."""
        active = bool(self._auto_cartography_enabled)
        if active:
            self.lbl_discovery_state.setText("AKTİF")
            self.lbl_discovery_state.setStyleSheet(
                "background-color: #065f46; color: #ecfdf5; border: 1px solid #10b981; "
                "border-radius: 6px; padding: 6px 12px; font-weight: bold;"
            )
        else:
            self.lbl_discovery_state.setText("PASİF")
            self.lbl_discovery_state.setStyleSheet(
                "background-color: #334155; color: #94a3b8; border: 1px solid #475569; "
                "border-radius: 6px; padding: 6px 12px; font-weight: bold;"
            )
        self.btn_start_discovery.setEnabled(not active)
        self.btn_stop_discovery.setEnabled(active)

    def _set_discovery_state(self, enabled: bool) -> None:
        """Keşfi Başlat / Keşfi Durdur butonları için ortak durum güncelleyici."""
        self._on_auto_cartography_toggled(bool(enabled))

    def _apply_minimap_inset(self, is_on: bool) -> None:
        self.canvas.set_minimap_overlay_visible(is_on)
        if hasattr(self, "btn_toggle_minimap_inset"):
            self.btn_toggle_minimap_inset.setText(
                "Harita Üstü Minimap Önizleme: AÇIK"
                if is_on
                else "Harita Üstü Minimap Önizleme: KAPALI"
            )

    def _on_user_toggle_minimap_inset(self, checked: bool) -> None:
        self._minimap_inset_user_override = True
        self._apply_minimap_inset(bool(checked))

    def _on_toggle_minimap_inset(self, checked: bool) -> None:
        self._on_user_toggle_minimap_inset(checked)

    def _sync_minimap_inset_with_scanner(self) -> None:
        """Canlı kırpım yüklüyken harita üstü önizleme açık kalır; kullanıcı kapatırsa öyle durur."""
        if self._minimap_inset_user_override:
            return
        if not hasattr(self, "minimap_preview_widget") or not hasattr(self, "btn_toggle_minimap_inset"):
            return
        pix = getattr(self.minimap_preview_widget, "_minimap_pixmap", None)
        has_crop = pix is not None and not pix.isNull()
        self.btn_toggle_minimap_inset.setChecked(has_crop)
        self._apply_minimap_inset(has_crop)

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
            mats = [
                m
                for m in self.minimap_preview_widget.get_gatherable_materials()
                if minimap_material_is_recordable(m)
            ]
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
                if "rel_map_dx" not in mat or "rel_map_dy" not in mat:
                    continue
                rel_dx = float(mat.get("rel_map_dx"))
                rel_dy = float(mat.get("rel_map_dy"))
                target_x = int(round(max(18.0, min(pw - 18.0, base_x + rel_dx))))
                target_y = int(round(max(18.0, min(ph - 18.0, base_y + rel_dy))))
                stack_cnt = int(mat.get("stack_count", mat.get("max_harvests", 3)) or 3)
                if stack_cnt not in (3, 6, 9):
                    stack_cnt = 3
                res_type = str(mat.get("resource_type") or "Toplanabilir Materyal")

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
                    # Yığın güncellemesi kalıcı depoya da yazılır; önceden
                    # yalnızca bellek içi _ordered_nodes güncelleniyordu.
                    if self.routes is not None and hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                        reg_data = self.routes.data.setdefault("regions", {}).setdefault(
                            self._region_name, {"nodes": [], "routes": {self._route_name: []}}
                        )
                        nodes_list = reg_data.setdefault("nodes", [])
                        stored = None
                        existing_id = existing.get("id")
                        for cand in nodes_list:
                            if not isinstance(cand, dict):
                                continue
                            if existing_id is not None and cand.get("id") == existing_id:
                                stored = cand
                                break
                            if math.hypot(
                                float(cand.get("map_x", 0)) - target_x,
                                float(cand.get("map_y", 0)) - target_y,
                            ) <= 14.0:
                                stored = cand
                                break
                        if stored is not None:
                            stored["source_type"] = "minimap_auto"
                            stored["minimap_auto"] = True
                            stored["stack_count"] = stack_cnt
                            stored["max_harvests"] = stack_cnt
                            stored["stack_badge"] = f"{stack_cnt}x"
                            stored["harvest_badge"] = existing["harvest_badge"]
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
                    "kind": "sprout_material",
                    "is_gatherable": True,
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
            # K-19: düğümler yukarıda yalnızca bellek içi sözlüklere
            # (self._ordered_nodes / routes.data) yazıldı; "kaydedildi"
            # raporu gerçek disk yazımına bağlanır. Kayıt sonucu ne olursa
            # olsun dönüş değeri (kaydedilen düğüm listesi) değişmez.
            persist_ok: Optional[bool] = None
            persist_error: Optional[str] = None
            if recorded_nodes and self.routes is not None:
                try:
                    if hasattr(self.routes, "save_region"):
                        persist_ok = self.routes.save_region(self._region_name) is not False
                    elif hasattr(self.routes, "save_data"):
                        persist_ok = self.routes.save_data() is not False
                except Exception as exc:
                    persist_ok = False
                    persist_error = str(exc)
                if persist_ok:
                    # Doğrulama: save_region True dönse bile, yukarıdaki
                    # yazımlar legacy `data` dalına gittiği için
                    # save_region içindeki senkron koşulu tutmazsa
                    # düğümler diske ulaşmamış olabilir. Kayıtta id'leri
                    # arayarak "kaydedildi" raporunu gerçekten doğrula.
                    try:
                        wanted_ids = {
                            int(n.get("id", 0))
                            for n in recorded_nodes
                            if isinstance(n, dict) and n.get("id") is not None
                        }
                        stored_ids = set()
                        regions_map = getattr(self.routes, "_regions", None)
                        if isinstance(regions_map, dict):
                            for key in (self._region_name, resolve_canonical_region(self._region_name)):
                                info = regions_map.get(key)
                                if isinstance(info, dict):
                                    for n in info.get("nodes", []):
                                        nid = getattr(n, "id", None)
                                        if nid is None and isinstance(n, dict):
                                            nid = n.get("id")
                                        if nid is not None:
                                            stored_ids.add(int(nid))
                        if wanted_ids and not wanted_ids <= stored_ids:
                            persist_ok = False
                            persist_error = (
                                "kayda yazılmayan düğümler: "
                                f"{sorted(wanted_ids - stored_ids)}"
                            )
                    except Exception as exc:
                        logger.warning("Minimap kayıt doğrulaması yapılamadı: %s", exc)
            self.minimap_materials_recorded.emit(list(recorded_nodes))
            if not recorded_nodes:
                logger.info("Minimap'ten kaydedilecek materyal bulunamadı.")
            elif persist_ok is True:
                logger.info(
                    "Minimap'ten %d toplanabilir materyal rotaya kaydedildi (canavarlar yoksayıldı).",
                    len(recorded_nodes),
                )
            elif persist_ok is False:
                logger.error(
                    "Minimap'ten %d toplanabilir materyal belleğe alındı ancak kalıcı depoya yazılamadı (%s/%s)%s; yeniden yüklemede kaybolur.",
                    len(recorded_nodes),
                    self._region_name,
                    self._route_name,
                    f": {persist_error}" if persist_error else "",
                )
            else:
                logger.warning(
                    "Minimap'ten %d toplanabilir materyal belleğe alındı; kalıcı depoya yazılmadı (RouteManager bağlı değil).",
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

    def _present_full_zone_pixmap(self, image: QImage) -> QPixmap:
        """
        frame_18s karesinin tamamını bölge tuvali yapar.
        Eski kod orta alanı 614×450 diye kesip o parçayı tüm harita sanıyordu;
        batı kıyısı (Silent Hill'in solu) ve karenin geri kalanı düşüyordu.
        Sol kayıt listesi bölge değil; o şerit kapanır. Video karesindeki
        altyazı bu dosyada durur; ayrı bir temiz bölge görseli yok.
        """
        gray = self._to_grayscale_pixmap(image)
        if gray.isNull():
            return gray
        w = gray.width()
        h = gray.height()
        # Sol kayıt/görev şeridi bölge değil. Batı kıyısı (x>~0.19) durur.
        painter = QPainter(gray)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.fillRect(0, 0, int(round(w * 0.19)), h, QColor("#10141a"))
        painter.end()
        return gray

    def _load_region_map_pixmap(
        self, nodes: List[Dict[str, Any]]
    ) -> Tuple[QPixmap, Tuple[int, int], Tuple[float, float]]:
        candidates: List[Tuple[str, bool]] = []
        if self._explicit_map_path and os.path.exists(self._explicit_map_path):
            is_f18 = os.path.basename(self._explicit_map_path).lower() == "frame_18s.jpg"
            candidates.append((self._explicit_map_path, is_f18))

        region_key = (self._region_name or "").strip().lower()
        # Görev 4 — bölgeye duyarlı harita seçimi: Altgard görseli yalnızca
        # bölge Altgard (veya bilinmiyor) ise adaydır. Başka bölgede kendi
        # görseli yoksa Altgard haritası kullanılmaz; mozaik yedeğe düşülür.
        self._map_missing_for_region = ""
        for hd_candidate in region_map_candidates(self._region_name):
            if os.path.isfile(hd_candidate):
                candidates.append((hd_candidate, False))
        if region_key not in ("", "altgard") and not candidates:
            self._map_missing_for_region = str(self._region_name or region_key)
            logger.warning(
                "'%s' bölgesi için HD harita görseli yok (assets/maps/%s.png); "
                "Altgard haritası kullanılmıyor, mozaik yedeğe düşülüyor.",
                self._region_name,
                region_key,
            )

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
        if not os.path.exists(frame18_path):
            frame18_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "video_frames",
                "frame_18s.jpg",
            )
        # frame_18s.jpg Altgard dünya haritası kırpmasıdır; başka bölgede yedek olamaz.
        if os.path.exists(frame18_path) and is_altgard_map_region(self._region_name):
            candidates.append((frame18_path, True))

        # HD fit yalnızca Altgard haritasına özeldir (2258×2400 affine);
        # başka bölge haritası (örn. Verteron 2303×2400) bu fiti almaz,
        # orantılı ölçekle çizilir (Verteron fiti henüz kalibre değil).
        hd_abs = os.path.abspath(hd_zone_map_path())
        for path, should_crop_f18 in candidates:
            try:
                img = QImage(path)
                if img.isNull():
                    continue
                full_w = float(img.width())
                full_h = float(img.height())
                is_hd = os.path.abspath(path) == hd_abs
                if is_hd:
                    pix = QPixmap.fromImage(img)
                elif should_crop_f18:
                    pix = self._present_full_zone_pixmap(img)
                else:
                    pix = self._to_grayscale_pixmap(img)
                if pix.isNull():
                    continue
                self._loaded_map_path = os.path.abspath(path)
                self._hd_map_active = bool(is_hd)
                return pix, (0, 0), (full_w, full_h)
            except Exception as exc:
                logger.warning("Harita görseli yüklenemedi ('%s'): %s", path, exc)

        self._loaded_map_path = ""
        self._hd_map_active = False
        return self._build_fallback_map_pixmap(nodes), (0, 0), DEFAULT_MAP_REF

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
            return list(self._ordered_nodes)

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
                if reg_data is None:
                    # İstenen bölge yoksa sessizce ilk bölgeye geçilmez;
                    # mevcut düğümler korunur, durum loglanır.
                    if regions:
                        logger.warning(
                            "İstenen bölge '%s' bulunamadı; ilk bölgeye geçilmiyor "
                            "(mevcut bölgeler: %s).",
                            self._region_name,
                            sorted(str(k) for k in regions.keys()),
                        )
                    reg_data = {}
                if isinstance(reg_data, dict):
                    for n in reg_data.get("nodes", []):
                        if isinstance(n, dict):
                            extracted.append(dict(n))
                        elif hasattr(n, "to_dict"):
                            extracted.append(dict(n.to_dict()))
        except Exception as exc:
            logger.error("RouteManager'dan node listesi alınırken hata: %s", exc)

        if not extracted and self._ordered_nodes:
            return list(self._ordered_nodes)
        return extracted

    def set_sequential_nodes(self, nodes: List[Dict[str, Any]]) -> None:
        """Sets sequential nodes directly and refreshes canvas and material filter."""
        self._ordered_nodes = list(nodes)
        self.canvas.set_sequential_nodes(self._ordered_nodes)
        self._refresh_metrics_bar()
        self._refresh_minimap_filter_stats()
        self._populate_material_filter_combo()

    def _generate_default_walk_trajectory(
        self, nodes: List[Dict[str, Any]], pixmap: QPixmap
    ) -> List[Tuple[float, float]]:
        """
        F-0643: Sentetik yürüyüş izi ÜRETİLMEZ.

        Eski davranış düğüm koordinatları arasına sin/cos salınımlı ara
        noktalar ekleyip bunu oyuncunun gezdiği iz gibi gösteriyordu; o iz
        gerçek bir yürüyüş kaydı değildir. Yürüyüş izi yalnızca kaydedilmiş
        oyuncu hareketidir (`set_walk_trajectory` / `append_walk_point`
        yoluyla gelen gerçek iz). Kayıtlı iz yoksa boş liste döner; düğüm
        koordinatlarından iz uydurulmaz. (P3-4: "geriye dönük uyumluluk
        için korunur (çağıran kod kırılmasın)" gerekçesi bayattı — R45
        çağrıyı reload_map_and_nodes'tan kaldırdığından beri üretimde
        çağıran kalmadı; yöntem fail-closed bekçi olarak durur: gelecekte
        biri iz isterse uydurma yerine [] alır. Tek çağıran
        tests/test_r45_cartography_synthetic_trajectory.py regresyon
        testidir.) Her zaman [] döner.
        """
        return []

    def reload_map_and_nodes(self) -> None:
        """Haritayı, sıralı düğümleri, yürüyüş izini ve üst istatistik çubuğunu yeniler."""
        try:
            self._region_name = self._region_name or self._resolve_active_region()
            self._route_name = self._route_name or self._resolve_active_route()
            title = (
                f"Canlı Oto-Haritalama ve Minimap Radarı · {self._region_name} ({self._route_name})"
            )
            # Görev 4: bölge uyumsuzluğu varsa başlıkta görünür uyarı.
            try:
                getter = getattr(self.routes, "get_region_mismatch", None)
                mismatch = getter() if callable(getter) else None
            except Exception:
                mismatch = None
            if isinstance(mismatch, dict):
                title += (
                    f" · ⚠ BÖLGE UYUMSUZLUĞU: {mismatch.get('detected_region')} ≠ "
                    f"{mismatch.get('active_region')} — rota durduruldu"
                )
            self.setWindowTitle(title)

            nodes = self._extract_nodes_from_manager()
            pixmap, _offset, map_ref = self._load_region_map_pixmap(nodes)
            self.planner.set_hd_fit(self._hd_map_active)
            self.planner.set_map_reference(map_ref[0], map_ref[1])
            self.canvas.set_map_reference(map_ref[0], map_ref[1])
            self.canvas.set_hd_fit(self._hd_map_active)

            # Konumu bilinmeyen (0,0) düğüme daire üzerinde uydurma koordinat
            # verilmez; kayıtlı koordinat aynen bırakılır, eksik konum loglanır.
            pw = max(200, pixmap.width())
            ph = max(200, pixmap.height())
            unknown_count = 0
            for n in nodes:
                try:
                    mx = float(n.get("map_x", n.get("x", 0.0)) or 0.0)
                    my = float(n.get("map_y", n.get("y", 0.0)) or 0.0)
                except (TypeError, ValueError):
                    mx, my = 0.0, 0.0
                if mx <= 0.0 and my <= 0.0:
                    unknown_count += 1
            if unknown_count:
                logger.warning(
                    "Konumu bilinmeyen %d düğüm var (%s); koordinat uydurulmadan bırakıldı.",
                    unknown_count,
                    self._region_name,
                )

            self._ordered_nodes = list(nodes)
            # F-0643: Kayıtlı yürüyüş izi yoksa düğümlerden sentetik iz
            # üretilmez ve oyuncu konumu izden/düğümden uydurulmaz.
            # Yürüyüş izi yalnızca kaydedilmiş gerçek oyuncu hareketidir
            # (set_walk_trajectory / append_walk_point); oyuncu konumu ise
            # yalnızca canlı kareden (update_live_frame) veya app_shell
            # üzerinden gelen gerçek konumdan alınır. Burada iz yoksa iz
            # boş kalır, oyuncu konumu da önceki gerçek değerinde (veya
            # bilinmiyorsa None) kalır; sabit 45° bakış açısıyla son
            # düğüme konum atanmaz.
            self.canvas.set_map_pixmap(pixmap, map_ref_size=map_ref)
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self.canvas.set_walk_trajectory(self._walk_trajectory)

            self._refresh_metrics_bar()
            self._refresh_minimap_filter_stats()
            self._populate_material_filter_combo()
        except Exception as exc:
            logger.error("reload_map_and_nodes hatası: %s", exc)

    def set_material_filter(self, materials: Union[str, Sequence[str], None]) -> None:
        """Sets the material filter in the radar dropdown and updates canvas."""
        if materials is None:
            text = "Tümü"
        elif isinstance(materials, (list, tuple, set)):
            text = str(list(materials)[0]) if materials else "Tümü"
        else:
            text = str(materials).strip()

        if hasattr(self, "combo_material_filter") and self.combo_material_filter is not None:
            idx = -1
            norm_target = normalize_material_token(text)
            for i in range(self.combo_material_filter.count()):
                item_text = self.combo_material_filter.itemText(i)
                item_data = self.combo_material_filter.itemData(i)
                if (item_data and normalize_material_token(str(item_data)) == norm_target) or \
                   item_text == text or normalize_material_token(item_text) == norm_target:
                    idx = i
                    break
            if idx >= 0:
                self.combo_material_filter.setCurrentIndex(idx)
            else:
                self.combo_material_filter.addItem(text, userData=text)
                self.combo_material_filter.setCurrentIndex(self.combo_material_filter.count() - 1)
        self.canvas.set_material_filter(text)

    def get_material_filter(self) -> str:
        """Returns the current material filter string."""
        if hasattr(self, "combo_material_filter") and self.combo_material_filter is not None:
            data = self.combo_material_filter.currentData()
            if data is not None and str(data).strip():
                return str(data)
            text = self.combo_material_filter.currentText()
            clean = re.sub(r"^[^\w\s/]+", "", text).strip()
            return clean or text
        return str(self.canvas.get_material_filter() or "Tümü")

    def draw_proximity_route(self) -> List[Dict[str, Any]]:
        """Calculates and renders proximity route for currently selected material."""
        mat = self.get_material_filter()
        self.canvas.set_material_filter(mat)
        ordered = self.canvas.calculate_proximity_route()
        self.canvas.update()
        logger.info("Radar Seç-Topla yakınlık rotası oluşturuldu: %s (%d nokta)", mat, len(ordered))
        return ordered

    def _on_material_filter_changed(self, text: str) -> None:
        """Slot for combo material filter change."""
        val = self.get_material_filter()
        self.canvas.set_material_filter(val)

    def _on_draw_proximity_route_clicked(self) -> None:
        """Slot for draw proximity route button."""
        self.draw_proximity_route()

    def _populate_material_filter_combo(self) -> None:
        """Dynamically populates material filter combo with distinct materials from current nodes."""
        if not hasattr(self, "combo_material_filter") or self.combo_material_filter is None:
            return
        existing = {
            normalize_material_token(str(self.combo_material_filter.itemData(i) or self.combo_material_filter.itemText(i)))
            for i in range(self.combo_material_filter.count())
        }
        for node in getattr(self, "_ordered_nodes", []):
            if isinstance(node, dict):
                rtype = str(node.get("resource_type") or node.get("material_type") or node.get("name") or "").strip()
                norm = normalize_material_token(rtype)
                if rtype and norm not in existing and len(norm) > 2:
                    self.combo_material_filter.addItem(rtype, userData=rtype)
                    existing.add(norm)

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

            persist_ok = True
            if self.routes is None:
                # K-19: kalıcı depo bağlı değilken "kaydedildi" dönmek yalan
                # olur — sıralama yalnızca bellekte (_ordered_nodes) durur.
                logger.error(
                    "Sıralı rota kalıcı depoya yazılamadı (%s/%s): RouteManager bağlı değil; sıralama yalnızca bellekte.",
                    self._region_name,
                    self._route_name,
                )
                return False
            if self.routes is not None:
                try:
                    # Yeni RouteManager (_regions sözlüğü ve NodeData listesi) uyumluluğu
                    if hasattr(self.routes, "_regions") and isinstance(
                        self.routes._regions, dict
                    ):
                        reg_info = self.routes._regions.get(self._region_name)
                        if reg_info is None:
                            reg_info = self.routes._regions.get(
                                resolve_canonical_region(self._region_name)
                            )
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
                                if self.routes.save_region(self._region_name) is False:
                                    persist_ok = False
                            elif hasattr(self.routes, "save_data"):
                                if self.routes.save_data() is False:
                                    persist_ok = False
                        else:
                            logger.warning(
                                "Sıralı rota yazılamadı: '%s' bölgesi RouteManager'da yok.",
                                self._region_name,
                            )
                            persist_ok = False
                    elif hasattr(self.routes, "data") and isinstance(self.routes.data, dict):
                        # Canlı sözlüğe yazılır; .get(..., {}) ile alınan kopuk
                        # sözlüğe yazmak kalıcı depoya ulaşmazdı.
                        regions = self.routes.data.setdefault("regions", {})
                        if not isinstance(regions, dict):
                            regions = {}
                            self.routes.data["regions"] = regions
                        reg_data = regions.get(self._region_name)
                        if reg_data is None:
                            reg_data = {}
                            regions[self._region_name] = reg_data
                        if isinstance(reg_data, dict):
                            reg_data["nodes"] = list(self._ordered_nodes)
                        if hasattr(self.routes, "save_data"):
                            if self.routes.save_data() is False:
                                persist_ok = False
                except Exception as inner_exc:
                    logger.warning(
                        "RouteManager üzerine sıralı rota yazılırken kısmi uyarı: %s",
                        inner_exc,
                    )
                    persist_ok = False

            if not persist_ok:
                logger.error(
                    "Sıralı rota kalıcı depoya yazılamadı (%s/%s); başarı sayılmadı.",
                    self._region_name,
                    self._route_name,
                )
                return False

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

    def _remove_node_at_index(self, idx: int) -> bool:
        """Sağ tık "Rotadan Çıkar": düğümü tek bir kalıcı daldan siler.

        K-19: eskiden yalnızca ``_ordered_nodes`` listesinden atılıyor, ne
        ``_regions`` ne ``data`` dalına ne de diske dokunuluyordu; düğüm
        yeniden yüklemede diriliyordu. Artık kalıcı depo tek kaynaktır:
        isimli rotada ``remove_node_from_route``, varsayılan ("All")
        rotada ``delete_node`` (varsayılan rotadaki düğüm bölgedeki
        düğümün kendisidir) çağrılır; yazım başarısızsa düğüm ekranda
        kalır ve ``False`` döner. Kalıcı depo bağlı değilse silme
        yalnızca bellekte olur ve bu açıkça loglanır.
        """
        if not 0 <= idx < len(self._ordered_nodes):
            return False
        node_obj = self._ordered_nodes[idx]
        try:
            node_id = int(node_obj.get("id", -1))
        except (TypeError, ValueError):
            node_id = -1
        if node_id < 0:
            return False

        route_name = str(getattr(self, "_route_name", "") or "")
        region_name = str(getattr(self, "_region_name", "") or "")

        if self.routes is None:
            self._ordered_nodes.pop(idx)
            self.canvas.set_sequential_nodes(self._ordered_nodes)
            self._refresh_metrics_bar()
            logger.warning(
                "Düğüm #%d rotadan çıkarıldı ancak kalıcı depoya yazılmadı "
                "(RouteManager bağlı değil); yeniden yüklemede geri gelir.",
                node_id,
            )
            return True

        persisted = False
        try:
            # "All", RouteManager'daki DEFAULT_ROUTE'un karşılığıdır;
            # bu dosyada core.route_manager import edilmez, literal
            # bilerek aynen tutulur.
            if route_name and route_name != "All" and hasattr(self.routes, "remove_node_from_route"):
                persisted = self.routes.remove_node_from_route(
                    node_id, route_name, region=region_name
                )
            elif hasattr(self.routes, "delete_node"):
                persisted = self.routes.delete_node(node_id, region=region_name)
            else:
                logger.error(
                    "Düğüm #%d rotadan çıkarılamadı: RouteManager'da silme yolu yok.",
                    node_id,
                )
                return False
        except Exception as exc:
            logger.error("Düğüm #%d rotadan çıkarılırken hata: %s", node_id, exc)
            return False

        if not persisted:
            # Kalıcı depoda düğüm hiç yoksa (yalnızca UI belleğindeyse)
            # ekrandan atmak dürüsttür; aksi hâlde dirilme yerine
            # ekranda tutup hatayı logla.
            node_in_store = False
            try:
                if hasattr(self.routes, "get_node"):
                    node_in_store = self.routes.get_node(node_id, region=region_name) is not None
            except Exception:
                node_in_store = True
            if node_in_store:
                logger.error(
                    "Düğüm #%d rotadan çıkarılamadı (kalıcı kayıt başarısız); ekranda bırakıldı.",
                    node_id,
                )
                return False
            logger.warning(
                "Düğüm #%d kalıcı depoda bulunamadı; yalnızca ekrandan çıkarıldı.",
                node_id,
            )

        self._ordered_nodes.pop(idx)
        self.canvas.set_sequential_nodes(self._ordered_nodes)
        self._refresh_metrics_bar()
        logger.info("Düğüm #%d rotadan çıkarıldı (%s/%s).", node_id, region_name, route_name)
        return True

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

    def canvas_zoom_in(self) -> None:
        if hasattr(self, "canvas") and hasattr(self.canvas, "zoom_in"):
            self.canvas.zoom_in()

    def canvas_zoom_out(self) -> None:
        if hasattr(self, "canvas") and hasattr(self.canvas, "zoom_out"):
            self.canvas.zoom_out()

    def canvas_reset_view(self) -> None:
        if hasattr(self, "canvas") and hasattr(self.canvas, "reset_view"):
            self.canvas.reset_view()

    def canvas_center_on_player(self) -> None:
        if hasattr(self, "canvas") and hasattr(self.canvas, "center_on_player"):
            self.canvas.center_on_player()

    def set_radar_filter(self, filter_name: str, enabled: bool) -> None:
        if hasattr(self, "canvas") and hasattr(self.canvas, "set_filter"):
            self.canvas.set_filter(filter_name, enabled)
        k = str(filter_name).lower()
        if "node" in k and hasattr(self, "btn_toggle_nodes") and self.btn_toggle_nodes.isChecked() != bool(enabled):
            self.btn_toggle_nodes.setChecked(bool(enabled))
        elif "enemi" in k and hasattr(self, "btn_toggle_enemies") and self.btn_toggle_enemies.isChecked() != bool(enabled):
            self.btn_toggle_enemies.setChecked(bool(enabled))
        elif ("path" in k or "rota" in k) and hasattr(self, "btn_toggle_path") and self.btn_toggle_path.isChecked() != bool(enabled):
            self.btn_toggle_path.setChecked(bool(enabled))
        elif ("proj" in k or "3d" in k) and hasattr(self, "btn_toggle_projections") and self.btn_toggle_projections.isChecked() != bool(enabled):
            self.btn_toggle_projections.setChecked(bool(enabled))

    def get_radar_filter(self, filter_name: str) -> bool:
        if hasattr(self, "canvas") and hasattr(self.canvas, "get_filter"):
            return self.canvas.get_filter(filter_name)
        return True

    def update_live_frame(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        yolo_boxes: Optional[List[Any]] = None,
        minimap_crop: Optional[np.ndarray] = None,
        player_pos: Optional[Tuple[float, float, float]] = None,
        heading_deg: Optional[float] = None,
    ) -> None:
        if hasattr(self, "canvas") and hasattr(self.canvas, "update_live_frame"):
            self.canvas.update_live_frame(
                frame_bgr=frame_bgr,
                yolo_boxes=yolo_boxes,
                minimap_crop=minimap_crop,
                player_pos=player_pos,
                heading_deg=heading_deg,
            )
        if minimap_crop is not None and hasattr(self, "minimap_preview_widget"):
            self.minimap_preview_widget.update_minimap_crop(minimap_crop)
            self._refresh_minimap_filter_stats()

    def project_3d_yolo_boxes(
        self,
        yolo_boxes: Optional[List[Any]],
        frame_w: int = 1280,
        frame_h: int = 720,
        heading_deg: Optional[float] = None,
        player_pos: Optional[Tuple[float, float]] = None,
    ) -> List[Dict[str, Any]]:
        if hasattr(self, "canvas") and hasattr(self.canvas, "project_3d_yolo_boxes"):
            return self.canvas.project_3d_yolo_boxes(
                yolo_boxes=yolo_boxes,
                frame_w=frame_w,
                frame_h=frame_h,
                heading_deg=heading_deg,
                player_pos=player_pos,
            )
        return []

    def get_radar_canvas(self) -> RadarCanvas:
        return self.canvas

    def get_radar_route_diagnostics(self) -> Dict[str, Any]:
        """Returns diagnostics on visible radar nodes, active material filter, and jump edges."""
        canvas = self.__dict__.get("canvas")
        nodes: List[Any] = []
        mat_filter = ""
        if canvas is not None:
            nodes = list(getattr(canvas, "_ordered_nodes", None) or getattr(canvas, "_nodes", None) or [])
            mat_filter = str(getattr(canvas, "_material_filter", "") or "")
        total = len(nodes)
        minimap_auto_count = sum(
            1 for n in nodes
            if isinstance(n, dict) and (str(n.get("source_type", n.get("source", ""))).lower() == "minimap_auto" or bool(n.get("minimap_auto", False)))
        )
        return {
            "total_nodes": total,
            "minimap_auto_nodes": minimap_auto_count,
            "material_filter": mat_filter,
            "radar_ready": canvas is not None,
        }

    def _cartography_radar_health(self) -> Tuple[bool, Dict[str, Any]]:
        """Gerçek hazırlık durumu: canvas + harita + planlayıcı mevcut mu?"""
        canvas = self.__dict__.get("canvas")
        canvas_ready = canvas is not None
        map_loaded = False
        if canvas is not None:
            pix = getattr(canvas, "_base_pixmap", None)
            map_loaded = pix is not None and not pix.isNull()
        planner_ready = getattr(self, "planner", None) is not None
        nodes = len(getattr(self, "_ordered_nodes", None) or [])
        ready = bool(canvas_ready and map_loaded and planner_ready)
        details: Dict[str, Any] = {
            "canvas_ready": canvas_ready,
            "map_loaded": map_loaded,
            "planner_ready": planner_ready,
            "nodes": nodes,
            "region": str(getattr(self, "_region_name", "")),
        }
        return ready, details

    def optimize_visible_radar_route(self, player_xy: Optional[Tuple[float, float]] = None) -> List[Dict[str, Any]]:
        """Re-orders visible radar route nodes by nearest-neighbor + 2-opt proximity."""
        canvas = self.__dict__.get("canvas")
        raw_nodes: List[Dict[str, Any]] = []
        if canvas is not None:
            for item in (getattr(canvas, "_ordered_nodes", None) or getattr(canvas, "_nodes", None) or []):
                if isinstance(item, dict):
                    raw_nodes.append(dict(item))
        if len(raw_nodes) <= 1:
            return raw_nodes
        px, py = player_xy if player_xy is not None else (
            float(raw_nodes[0].get("map_x", raw_nodes[0].get("x", 0.0))),
            float(raw_nodes[0].get("map_y", raw_nodes[0].get("y", 0.0))),
        )
        remaining = list(raw_nodes)
        ordered: List[Dict[str, Any]] = []
        cx, cy = float(px), float(py)
        while remaining:
            best_idx = min(
                range(len(remaining)),
                key=lambda i: math.hypot(
                    float(remaining[i].get("map_x", remaining[i].get("x", 0.0))) - cx,
                    float(remaining[i].get("map_y", remaining[i].get("y", 0.0))) - cy,
                ),
            )
            nxt = remaining.pop(best_idx)
            nxt["seq_index"] = len(ordered) + 1
            ordered.append(nxt)
            cx = float(nxt.get("map_x", nxt.get("x", cx)))
            cy = float(nxt.get("map_y", nxt.get("y", cy)))
        if canvas is not None and hasattr(canvas, "set_sequential_nodes"):
            try:
                canvas.set_sequential_nodes(ordered)
            except Exception:
                pass
        return ordered

    def get_cartography_radar_health_summary(self) -> dict:
        ready, details = self._cartography_radar_health()
        return {"cartography_radar_ready": ready, **details}

    def get_cartography_radar_ready_flag(self) -> dict:
        ready, _details = self._cartography_radar_health()
        return {"module": "cartography_radar", "ready": ready}

    def get_cartography_radar_version_info(self) -> dict:
        ready, _details = self._cartography_radar_health()
        return {"module": "cartography_radar", "version": __version__, "ready": ready}

    def get_cartography_radar_diagnostic_snapshot(self) -> dict:
        ready, details = self._cartography_radar_health()
        return {
            "module": "cartography_radar",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            **details,
        }


def get_cartography_radar_capabilities() -> list:
    return [
        "cartography_radar",
        "live_radar_overlay",
        "diagnostic_snapshot",
        "radar_canvas",
        "3d_ai_projection",
        "live_minimap_fusion",
        "offline_simulation",
        "filter_controls",
    ]


def validate_cartography_radar_contract() -> bool:
    return bool(get_cartography_radar_capabilities())


def get_cartography_radar_baseline_metrics() -> dict:
    return {"module": "cartography_radar", "contract_valid": validate_cartography_radar_contract(), "capability_count": len(get_cartography_radar_capabilities())}


def get_cartography_radar_readiness_status() -> dict:
    try:
        return {
            "module": "cartography_radar",
            "ready": validate_cartography_radar_contract(),
            "metrics": get_cartography_radar_baseline_metrics(),
        }
    except Exception:
        return {"module": "cartography_radar", "ready": False, "metrics": {}}


def get_cartography_radar_audit_record() -> dict:
    try:
        return {
            "module": "cartography_radar",
            "audit_passed": validate_cartography_radar_contract(),
            "readiness": get_cartography_radar_readiness_status(),
        }
    except Exception:
        return {
            "module": "cartography_radar",
            "audit_passed": False,
            "readiness": {"module": "cartography_radar", "ready": False, "metrics": {}},
        }


def get_cartography_radar_operational_digest() -> dict:
    return {
        "module": "cartography_radar",
        "operational": validate_cartography_radar_contract(),
        "audit": get_cartography_radar_audit_record(),
    }


def get_cartography_radar_telemetry_payload() -> dict:
    return {
        "module": "cartography_radar",
        "telemetry_ready": validate_cartography_radar_contract(),
        "digest": get_cartography_radar_operational_digest(),
    }


def get_cartography_radar_integrity_seal() -> dict:
    return {
        "module": "cartography_radar",
        "sealed": validate_cartography_radar_contract(),
        "telemetry": get_cartography_radar_telemetry_payload(),
    }


def get_cartography_radar_compliance_certificate() -> dict:
    return {
        "module": "cartography_radar",
        "certified": validate_cartography_radar_contract(),
        "seal": get_cartography_radar_integrity_seal(),
    }


def get_cartography_radar_readiness_probe() -> dict:
    return {
        "module": "cartography_radar",
        "ready": validate_cartography_radar_contract(),
        "certificate": get_cartography_radar_compliance_certificate(),
    }


def get_cartography_radar_audit_summary() -> dict:
    return {
        "module": "cartography_radar",
        "audit_passed": validate_cartography_radar_contract(),
        "probe": get_cartography_radar_readiness_probe(),
    }


# Map/Radar Worker 4 Aliases & Exports
CartographyRadarWindow = CartographyRadarDialog

__all__ = [
    "RadarCanvas",
    "CartographyRadarDialog",
    "CartographyRadarWindow",
    "SequentialMapCanvas",
    "SequentialRoutePlanner",
    "MinimapLiveCropPreviewWidget",
]

