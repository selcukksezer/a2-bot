# -*- coding: utf-8 -*-
"""
Koridor Navigasyon, 3B Engel Aşma, Uçurum Filtresi, Cooldown Öncesi Erken Yürüyüş,
Dinamik Merkez Kayması Düzeltme ve Rakip Oyuncu (Contested Node) Tespit Modülü.

50-Maddelik Spesifikasyon — Maddeler 26..30:
- Madde 26 (`PhaseCorrelateUnstuckManeuver`):
  Karakter yürürken (`W` basılı) sağ üst minimap arazi kırpımları üzerinde `cv2.phaseCorrelate`
  çalıştırır; `2.5 saniye` boyunca minimap arazi kayması `< 1.2 px` kalırsa (kaya/ağaç takılması),
  sırasıyla `Space` (Zıpla) ➔ `A + W` (`1.2 sn` ileri-sol strafe, ardışık tetiklemelerde `D + W`)
  ➔ hedefe yeniden hizalanma manevrasını uygular.
- Madde 27 (`ZAxisCliffFilter`):
  2B minimap üzerinde çok yakın (`< 25 px`) görünmesine rağmen ardışık `2` yaklaşma denemesinde
  etkileşime girilemeyen (uçurum tepesinde veya vadi tabanında kalan) düğümleri
  `unreachable_cliff = True` ve `altitude_band = "cliff_mismatch"` olarak etiketler; uçuş modu
  (`is_flying`) aktif olmadıkça veya aynı irtifa bandında bulunulmadıkça yer koridorundan dışlar.
- Madde 28 (`PredictivePreWalkScheduler`):
  5 dakikalık (`300 sn`) dinlenme süresinin bitmesine `<= 8 sn` kalan ve tahmini yürüyüş süresi
  `~10 sn` olan (`rest_remaining_sec <= estimated_walk_sec + 2.0`) düğümleri erken seçip yürüyüşü
  önceden başlatır; karakter materyal tam doğduğu saniyede düğümün başında olur.
- Madde 29 (`CentroidDriftCorrector`):
  Uzaktan (3B tepe ikonu veya minimap kenarından) otomatik haritalanan bir düğüme varılıp
  gerçekten `[F Gather]` yapıldığında, düğümün kayıtlı `(x, y)` koordinatını üstel hareketli
  ortalama (EMA) / gerçek oyuncu konumu (`refine_node_coordinates_on_gather`) ile milimetrik günceller.
- Madde 30 (`ContestedNodeDetector`):
  Hedef düğümün tam üzerinde başka bir oyuncu karakterinin / isimliğinin (`nameplate`) toplama
  animasyonu yaptığını tespit eder (`is_node_contested(frame, node_screen_xy) -> bool`) ve
  vakit kaybetmeden doğrudan sıradaki (`Next in Sequence`) düğüme geçer.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
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

Point2D = Tuple[float, float]


# --- Tek doğruluk kaynağı: metre/piksel ölçekleri (P1-7 / K-20 kararı) ---
# İki farklı piksel uzayı, iki farklı ölçek vardır; tek sayıya indirgenemezler:
#  * MINIMAP_METERS_PER_PX — sağ-üst minimap pikseli -> metre: 0.55.
#    material_scanner/explorer yolundaki 0.65 değeri bu sabite bağlandı.
#    (P1-12 doğrulaması: 0.65'in minimap uzayı olduğu kanıtlı —
#    scan_minimap_materials, minimap kırpıntısı merkezine göre göreli
#    piksel — yani bağlama doğru uzaya yapıldı; değerin kendisi hâlâ
#    canlı ölçümle kalibre edilmedi.)
#  * WORLD_MAP_METERS_PER_PX — açık dünya haritası (M) pikseli -> metre: 0.28.
#    vision.py'deki eski 0.25 değeri bu sabite bağlandı.
# ÖLÇEK KARARI — DOĞRULANAMADI: iki değer de canlı ölçümle kalibre edilmiş
# DEĞİLDİR (oyun açıkken bilinen mesafede minimap/dünya-harita pikseli
# ölçülmedi). 0.55 seçildi çünkü minimap yolunda 3 modülde (navigator.py
# PIXEL_TO_METER_DEFAULT, sequential_router, navigation_intel) zaten
# kullanılan değerdir; 0.65 yalnızca material_scanner/explorer yolundaydı.
# 0.28 seçildi çünkü dünya-harita yolunda route_manager / economy_broker /
# cartography_radar üçlüsünde zaten kullanılan değerdir; 0.25 yalnızca
# vision.py'deydi ve route_manager artık onu okumuyordu. Seçim gerekçesi
# ÇOĞUNLUK + TEK-KAYNAK tutarlılığıdır; kalibrasyon açık soru olarak
# kalır — bu sabitler "ölçüldü" diye okunmamalıdır.
MINIMAP_METERS_PER_PX: float = 0.55
WORLD_MAP_METERS_PER_PX: float = 0.28

# --- Kalibrasyon kancası (v19, P1-7/P1-12) ---
# Yukarıdaki iki değer CANLI ÖLÇÜMLE kalibre edilmiş DEĞİLDİR — çoğunluk +
# tek-kaynak tutarlılığıyla seçilmiş varsayılanlardır, "ölçüldü" diye
# okunmamalıdır. Gerçek kalibrasyon PC'de, oyun açıkken yapılır ve sonucu
# `config/settings.json` -> `route.minimap_meters_per_px` /
# `route.world_map_meters_per_px` anahtarlarına yazılır
# (bkz. `config/schema.py::RouteSettings`, `None` = kalibre edilmedi).
# `resolve_*_meters_per_px(settings)` config geçersiz kılmasını okur,
# yoksa yerleşik sabiti döndürür; hiçbir çağrı yerinde artık literal
# 0.55/0.65/0.28/0.25 yazılmaz.
SCALE_CALIBRATION_MIN_M_PER_PX: float = 0.05
SCALE_CALIBRATION_MAX_M_PER_PX: float = 5.0


def calibrate_meters_per_px(
    known_distance_m: float,
    pixel_distance: float,
) -> float:
    """Ölçülmüş bir çiftinden metre/piksel ölçeği hesaplar.

    Kalibrasyon ölçüm fonksiyonu — saf fonksiyondur, disk/ayar okumaz.
    `known_distance_m`: oyunda bilinen gerçek mesafe (metre, örn. iki
    düğüm arası tabelâ mesafesi veya karakter adım sayacından mesafe);
    `pixel_distance`: aynı iki nokta arası minimap / dünya-haritası
    piksel mesafesi (aynı karede, aynı yakınlaştırma düzeyinde ölçülür).
    Döndürür: `known_distance_m / pixel_distance` (m/px).
    Girdi geçersizse (sıfır/negatif/sonsuz) ValueError fırlatır — uydurma
    ölçek üretilemez.
    """
    try:
        known = float(known_distance_m)
        pixels = float(pixel_distance)
    except (TypeError, ValueError):
        raise ValueError(
            "Kalibrasyon girdileri sayı olmalı: known_distance_m=%r, pixel_distance=%r"
            % (known_distance_m, pixel_distance)
        )
    if not (math.isfinite(known) and math.isfinite(pixels)):
        raise ValueError("Kalibrasyon girdileri sonlu olmalı.")
    if known <= 0 or pixels <= 0:
        raise ValueError(
            "Kalibrasyon girdileri pozitif olmalı: known_distance_m=%r, pixel_distance=%r"
            % (known_distance_m, pixel_distance)
        )
    value = known / pixels
    if not (SCALE_CALIBRATION_MIN_M_PER_PX <= value <= SCALE_CALIBRATION_MAX_M_PER_PX):
        raise ValueError(
            "Hesaplanan ölçek %.4f m/px makul aralık dışında (%.2f..%.2f); "
            "ölçümü tekrarlayın." % (value, SCALE_CALIBRATION_MIN_M_PER_PX, SCALE_CALIBRATION_MAX_M_PER_PX)
        )
    return value


def _resolve_scale_override(settings: Any, key: str) -> Optional[float]:
    """`settings` içinden `route.<key>` geçersiz kılmasını okur.

    `settings` BotSettings, RouteSettings, dict veya `route` özniteliği
    taşıyan herhangi bir nesne olabilir. Geçersiz kılma yoksa / sayı
    değilse / aralık dışındaysa None döner (yerleşik sabit kullanılır);
    aralık dışı değer loglanır, sessizce kabul edilmez.
    """
    raw: Any = None
    try:
        if settings is None:
            return None
        route = settings
        if isinstance(settings, dict):
            route = settings.get("route", settings)
        else:
            route = getattr(settings, "route", settings)
        if isinstance(route, dict):
            raw = route.get(key)
        else:
            raw = getattr(route, key, None)
    except Exception:
        return None
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ölçek geçersiz kılması sayı değil (%s=%r); yok sayılıyor.", key, raw)
        return None
    if not math.isfinite(value):
        logger.warning("Ölçek geçersiz kılması sonlu değil (%s=%r); yok sayılıyor.", key, raw)
        return None
    if not (SCALE_CALIBRATION_MIN_M_PER_PX <= value <= SCALE_CALIBRATION_MAX_M_PER_PX):
        logger.warning(
            "Ölçek geçersiz kılması aralık dışı (%s=%r, aralık %.2f..%.2f); yok sayılıyor.",
            key,
            raw,
            SCALE_CALIBRATION_MIN_M_PER_PX,
            SCALE_CALIBRATION_MAX_M_PER_PX,
        )
        return None
    return value


def resolve_minimap_meters_per_px(settings: Any = None) -> float:
    """Minimap m/px: config geçersiz kılması varsa onu, yoksa `MINIMAP_METERS_PER_PX`."""
    override = _resolve_scale_override(settings, "minimap_meters_per_px")
    return override if override is not None else MINIMAP_METERS_PER_PX


def resolve_world_map_meters_per_px(settings: Any = None) -> float:
    """Dünya-haritası m/px: config geçersiz kılması varsa onu, yoksa `WORLD_MAP_METERS_PER_PX`."""
    override = _resolve_scale_override(settings, "world_map_meters_per_px")
    return override if override is not None else WORLD_MAP_METERS_PER_PX


def scale_calibration_status(settings: Any = None) -> Dict[str, Any]:
    """Ölçek kalibrasyon durumunu raporlar.

    `calibrated` yalnızca config'te geçerli bir geçersiz kılma varken
    True'dur; yerleşik sabitler kalibre edilmiş sayılmaz.
    """
    minimap_override = _resolve_scale_override(settings, "minimap_meters_per_px")
    world_override = _resolve_scale_override(settings, "world_map_meters_per_px")
    return {
        "minimap_meters_per_px": minimap_override if minimap_override is not None else MINIMAP_METERS_PER_PX,
        "minimap_calibrated": minimap_override is not None,
        "minimap_source": "config-override" if minimap_override is not None else "builtin-uncalibrated",
        "world_map_meters_per_px": world_override if world_override is not None else WORLD_MAP_METERS_PER_PX,
        "world_map_calibrated": world_override is not None,
        "world_map_source": "config-override" if world_override is not None else "builtin-uncalibrated",
    }


def jump_length_threshold(lengths: List[float]) -> float:
    """Zikzak-sıçrama eşiği — TEK TANIM (P1-12 / v18).

    "Ping-pong jump" (zikzak sıçrama) = yerel adımın çok üstündeki bağ
    (yanlış çözünürlük / sahte köşe noktası). Bu fonksiyon eşiğin tek
    tanımıdır; `ui/cartography_radar.py` (calculate_route_metrics,
    _jump_mask) ve `core/sequential_router.py` (calculate_route_metrics)
    aynı tanımı buradan okur. Önceden iki farklı `ping_pong_jumps`
    tanımı vardı: radar uzun-bağ sayıyordu, sequential_router ise
    `sharp_turns_count` (> 90° dönüş) sayısını aynı anahtara yazıyordu —
    keskin dönüş ayrı bir metriktir (`sharp_turns_count`), sıçrama değil.
    """
    if not lengths:
        return 280.0
    core = sorted(lengths)[: max(1, int(math.ceil(len(lengths) * 0.8)))]
    typical = core[len(core) // 2]
    return max(typical * 6.0, 280.0)
# --- Tek doğruluk kaynağı: manevra / dönüş varsayılanları (P3-3) ---
# `__init__` ve fonksiyon varsayılanlarında tekrar eden literal'lar; değerler aynı.
# Not: `UNSTUCK_STUCK_THRESHOLD_PX` (1.2 px) ile `UNSTUCK_STEP_DURATION_SEC`
# (1.2 sn) aynı sayı ama farklı anlam/birimdir; bilerek ayrı sabittirler.
UNSTUCK_STUCK_THRESHOLD_PX: float = 1.2
UNSTUCK_STUCK_DURATION_SEC: float = 2.5
UNSTUCK_STEP_DURATION_SEC: float = 1.2
TURN_MIN_DURATION_SEC: float = 0.04
TURN_MAX_DURATION_SEC: float = 0.35


def _get_node_attr(node: Any, key: str, default: Any = None) -> Any:
    """Düğüm nesnesinden (`dict` veya nesne) güvenli öznitelik okur."""
    if isinstance(node, dict):
        return node.get(key, default)
    return getattr(node, key, default)


def _set_node_attr(node: Any, key: str, value: Any) -> None:
    """Düğüm nesnesine (`dict` veya nesne) güvenli öznitelik yazar."""
    if isinstance(node, dict):
        node[key] = value
    elif not isinstance(node, (tuple, list)):
        try:
            setattr(node, key, value)
        except Exception as exc:
            logger.debug("Düğüm özniteliği yazılamadı (%s=%r): %s", key, value, exc)


def _node_has_coordinates(node: Any) -> bool:
    """Düğümün gerçek harita koordinatı var mı?

    `NodeData` varsayılanları (`map_x=0, map_y=0, norm_x=0.5, norm_y=0.5`)
    "koordinat yok" demektir; `(0, 0)` + norm `(0.5, 0.5)` bir düğümün
    harita merkezinde (960, 540) olduğu anlamına gelmez (dilim 12 K-3).
    Tam merkezde (norm tam 0.5, 0.5) gerçekten haritalanmış bir düğüm bu
    testte koordinatsız sayılır — merkezdeki düğüm yeniden haritalanmalıdır.
    """
    try:
        if isinstance(node, (tuple, list, np.ndarray)) and len(node) >= 2:
            return bool(float(node[0]) != 0.0 or float(node[1]) != 0.0)
        if isinstance(node, dict):
            mx = node.get("map_x", node.get("x"))
            my = node.get("map_y", node.get("y"))
            if mx is not None and my is not None and (float(mx) != 0.0 or float(my) != 0.0):
                return True
            nx = node.get("norm_x")
            ny = node.get("norm_y")
            if nx is None or ny is None:
                return False
            return bool(abs(float(nx) - 0.5) > 1e-9 or abs(float(ny) - 0.5) > 1e-9)
        mx = getattr(node, "map_x", getattr(node, "x", None))
        my = getattr(node, "map_y", getattr(node, "y", None))
        if mx is not None and my is not None and (float(mx) != 0.0 or float(my) != 0.0):
            return True
        nx = getattr(node, "norm_x", None)
        ny = getattr(node, "norm_y", None)
        if nx is None or ny is None:
            return False
        return bool(abs(float(nx) - 0.5) > 1e-9 or abs(float(ny) - 0.5) > 1e-9)
    except (TypeError, ValueError):
        return False


def _extract_node_xy_or_none(
    node: Any,
    fallback_width: Optional[float] = None,
    fallback_height: Optional[float] = None,
) -> Optional[Point2D]:
    """Koordinatsız düğümde `None` döner; uydurma merkez koordinat üretmez.

    `_extract_node_xy` geriye dönük uyumluluk için `(0, 0)` / `(960, 540)`
    döndürmeye devam eder; rota kararı veren yeni kod bu varyantı kullanır.
    Boyut verilmezse norm dönüşümü harita kanvası 1280x720 ile yapılır
    (B-01: eski sessiz 1920x1080 varsayılanı kaldırıldı).
    """
    if not _node_has_coordinates(node):
        return None
    return _extract_node_xy(node, fallback_width=fallback_width, fallback_height=fallback_height)


# B-01 (sweep-slice30): norm_x/norm_y HARİTA KANVASI (1280x720) uzayında
# tanımlıdır (kanonik değer core/route_manager.py MAP_CANVAS_*).
MAP_CANVAS_WIDTH = 1280.0
MAP_CANVAS_HEIGHT = 720.0
_NORM_CANVAS_FALLBACK_WARNED = False


def _warn_norm_canvas_fallback_once() -> None:
    """Norm dalı boyutsuz çağrıldı: kanvas varsayıldı — süreçte bir kez uyar."""
    global _NORM_CANVAS_FALLBACK_WARNED
    if not _NORM_CANVAS_FALLBACK_WARNED:
        _NORM_CANVAS_FALLBACK_WARNED = True
        logger.warning(
            "Norm-koordinat dönüşümünde boyut verilmedi; harita kanvası "
            "1280x720 varsayıldı (eski sessiz 1920x1080 yanlış ölçekti). "
            "Çağıran gerçek boyutu geçirmeli."
        )


def _norm_xy_to_px(
    nx: Any,
    ny: Any,
    fallback_width: Optional[float] = None,
    fallback_height: Optional[float] = None,
) -> Optional[Point2D]:
    """norm_x/norm_y -> piksel. Boyut yoksa kanvas varsayılır (bir kez uyarılır).

    (0.5, 0.5) yer tutucu merkezdir, gerçek koordinat değil — reddedilir
    (None döner). Ölçülemeyen norm da None döner.
    """
    if nx is None or ny is None:
        return None
    try:
        fx = float(nx)
        fy = float(ny)
    except (TypeError, ValueError):
        return None
    if fx == 0.5 and fy == 0.5:
        return None
    if fallback_width is None or fallback_height is None:
        _warn_norm_canvas_fallback_once()
    width = MAP_CANVAS_WIDTH if fallback_width is None else float(fallback_width)
    height = MAP_CANVAS_HEIGHT if fallback_height is None else float(fallback_height)
    return fx * width, fy * height


def _extract_node_xy(
    node: Any,
    fallback_width: Optional[float] = None,
    fallback_height: Optional[float] = None,
) -> Point2D:
    """Düğüm nesnesinden `(x, y)` 2B harita koordinatını güvenli şekilde çıkarır.

    Boyut verilmezse norm dönüşümü harita kanvası 1280x720 ile yapılır
    (B-01: eski sessiz 1920x1080 varsayılanı kaldırıldı) ve bir kez uyarı
    loglanır. (0.5, 0.5) norm yer tutucusu her dalda reddedilir.
    """
    try:
        if isinstance(node, (tuple, list, np.ndarray)) and len(node) >= 2:
            return float(node[0]), float(node[1])
        if isinstance(node, dict):
            mx = node.get("map_x", node.get("x", 0.0))
            my = node.get("map_y", node.get("y", 0.0))
            if (mx is None or float(mx) == 0.0) and (my is None or float(my) == 0.0):
                px = _norm_xy_to_px(
                    node.get("norm_x"), node.get("norm_y"),
                    fallback_width, fallback_height,
                )
                if px is not None:
                    return px
            return float(mx or 0.0), float(my or 0.0)

        mx = getattr(node, "map_x", getattr(node, "x", 0.0))
        my = getattr(node, "map_y", getattr(node, "y", 0.0))
        if (mx is None or float(mx) == 0.0) and (my is None or float(my) == 0.0):
            px = _norm_xy_to_px(
                getattr(node, "norm_x", None), getattr(node, "norm_y", None),
                fallback_width, fallback_height,
            )
            if px is not None:
                return px
        return float(mx or 0.0), float(my or 0.0)
    except Exception as exc:
        logger.debug("Koordinat çıkarılamadı (%r): %s", node, exc)
        return 0.0, 0.0


def _get_node_z(node: Any) -> Optional[float]:
    """Düğümün Z/irtifa değerini okur; yoksa veya 0.0 (ayarlanmamış) ise None döner.

    Gerçek rota düğümlerinde (NodeData) z/altitude_z alanı yoktur; bu yüzden
    altitude/elevation gibi diğer adlar da yoklanır. Bu kod tabanında 0.0
    "ayarlanmamış" anlamındadır (map_x/map_y ile aynı kural); gerçek irtifa
    gibi yorumlanıp düğümün yanlış dışlanmasına yol açardı (F-0249).
    """
    for key in ("z", "altitude_z", "altitude", "elevation", "height", "pos_z"):
        val = _get_node_attr(node, key, None)
        if val is None:
            continue
        try:
            z = float(val)
        except (TypeError, ValueError):
            continue
        if z == 0.0:
            continue
        return z
    return None


def _node_identifier(node: Any) -> Union[int, str]:
    """Düğüm için benzersiz izleme anahtarı üretir."""
    if isinstance(node, dict):
        if "id" in node and node["id"] is not None:
            return f"id:{node['id']}"
        if "seq_index" in node and node["seq_index"] is not None:
            return f"seq:{node['seq_index']}"
    else:
        node_id = getattr(node, "id", None)
        if node_id is not None:
            return f"id:{node_id}"
        seq_idx = getattr(node, "seq_index", None)
        if seq_idx is not None:
            return f"seq:{seq_idx}"
    return id(node)


# ==============================================================================
# HEADING VS BEARING ALIGNMENT & TURN DAMPING (OSCILLATION PREVENTION)
# ==============================================================================
DEFAULT_DEADBAND_DEG: float = 6.0
MIN_DEADBAND_DEG: float = 5.0
MAX_DEADBAND_DEG: float = 8.0


def compute_bearing(from_pos: Point2D, to_pos: Point2D) -> float:
    """
    Minimap / pusula koordinatlarında (0° = Kuzey, 90° = Doğu, 180° = Güney, 270° = Batı)
    hedef kerteriz açısını (bearing) hesaplar.
    Ekran koordinatlarında Y aşağı doğru büyüdüğünden, Kuzey = -Y yönüdür.
    Böylece Kartezyen (0° = Doğu) ile Minimap ok yönü (0° = Kuzey) arasındaki
    180° terslik (inversion) hatasını tamamen ortadan kaldırır.
    """
    dx = float(to_pos[0]) - float(from_pos[0])
    dy = float(to_pos[1]) - float(from_pos[1])
    if abs(dx) <= 1e-6 and abs(dy) <= 1e-6:
        return 0.0
    angle = math.degrees(math.atan2(dx, -dy))
    return round((angle + 360.0) % 360.0, 2)


def compute_angle_diff(target_bearing: float, player_heading: float) -> float:
    """
    Oyuncu yönünden (player_heading) hedef açıya (target_bearing) en kısa
    dönüş yolunu (shortest rotational path) [-180.0°, +180.0°] aralığında hesaplar:
        angle_diff = (target_bearing - player_heading + 180) % 360 - 180
    Pozitif (+) değerler sağa (saat yönü), negatif (-) değerler sola dönüşü gösterir.
    """
    diff = (float(target_bearing) - float(player_heading) + 180.0) % 360.0 - 180.0
    return round(diff, 2)


def apply_turn_damping(
    angle_diff: float,
    deadband_deg: float = DEFAULT_DEADBAND_DEG,
    min_turn_sec: float = TURN_MIN_DURATION_SEC,
    max_turn_sec: float = TURN_MAX_DURATION_SEC,
    max_error_deg: float = 180.0,
) -> Dict[str, Any]:
    """
    Ölü bölge (deadband zone: ±5° ila ±8°) uygulayarak hedef kerterize yakınken
    sağ-sol hızlı yalpalanmayı (oscillation) engeller. Açı hatasına göre dönüş
    süresini pürüzsüz biçimde enterpole eder.
    """
    clamped_deadband = max(MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg)))
    abs_err = abs(float(angle_diff))
    in_deadband = bool(abs_err <= clamped_deadband)

    if in_deadband:
        return {
            "angle_diff": round(float(angle_diff), 2),
            "abs_error": round(abs_err, 2),
            "in_deadband": True,
            "turn_needed": False,
            "turn_direction": "straight",
            "turn_duration_sec": 0.0,
            "deadband_deg": clamped_deadband,
        }

    direction = "right" if angle_diff > 0.0 else "left"
    span = max(1.0, float(max_error_deg) - clamped_deadband)
    norm = min(1.0, max(0.0, (abs_err - clamped_deadband) / span))
    smooth_t = norm * norm * (3.0 - 2.0 * norm)
    turn_duration = float(min_turn_sec) + smooth_t * (float(max_turn_sec) - float(min_turn_sec))

    return {
        "angle_diff": round(float(angle_diff), 2),
        "abs_error": round(abs_err, 2),
        "in_deadband": False,
        "turn_needed": True,
        "turn_direction": direction,
        "turn_duration_sec": round(turn_duration, 4),
        "deadband_deg": clamped_deadband,
    }


class TurnDampingController:
    """
    Yalpalanmayı önleyici durum denetleyicisi: ±5°..±8° ölü bölge uygular
    ve dönüş sürelerini yumuşak enterpolasyonla sınırlar.
    """
    def __init__(
        self,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        min_turn_sec: float = TURN_MIN_DURATION_SEC,
        max_turn_sec: float = TURN_MAX_DURATION_SEC,
        reversal_cooldown_sec: float = 0.12,
    ):
        self.deadband_deg = max(MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg)))
        self.min_turn_sec = float(min_turn_sec)
        self.max_turn_sec = float(max_turn_sec)
        self.reversal_cooldown_sec = float(reversal_cooldown_sec)
        self.last_turn_direction: Optional[str] = None
        self.last_turn_ts: float = 0.0

    def calculate_turn(
        self,
        target_bearing: float,
        player_heading: float,
        now: Optional[float] = None,
    ) -> Dict[str, Any]:
        curr_ts = time.time() if now is None else float(now)
        diff = compute_angle_diff(target_bearing, player_heading)
        res = apply_turn_damping(
            angle_diff=diff,
            deadband_deg=self.deadband_deg,
            min_turn_sec=self.min_turn_sec,
            max_turn_sec=self.max_turn_sec,
        )
        if res["turn_needed"]:
            if (
                self.last_turn_direction is not None
                and self.last_turn_direction != res["turn_direction"]
                and (curr_ts - self.last_turn_ts) < self.reversal_cooldown_sec
            ):
                res["turn_duration_sec"] = round(res["turn_duration_sec"] * 0.65, 4)
                res["reversal_damped"] = True
            else:
                res["reversal_damped"] = False
            self.last_turn_direction = res["turn_direction"]
            self.last_turn_ts = curr_ts
        else:
            res["reversal_damped"] = False
        return res


# ==============================================================================
# MADDE 26 — 3B ENGEL AŞMA (PhaseCorrelate Jump + Strafe Unstuck Maneuver)
# ==============================================================================
class PhaseCorrelateUnstuckManeuver:
    """
    Madde 26 — 3B Engel Aşma (`PhaseCorrelateUnstuckManeuver`):

    Karakter ileri yürürken (`W` basılı) sağ üst minimap arazi kırpımları üzerinde
    `cv2.phaseCorrelate` çalıştırır. Eğer `2.5 saniye` boyunca minimap arazi kayması
    `< 1.2 px` ise (karakter kaya, ağaç kökü veya çite takılmışsa), şu kurtulma
    dizisini çalıştırır:
        `Space` (Zıpla) ➔ `A + W` (`1.2 sn` ileri-sol strafe veya dönüşümlü `D + W`)
        ➔ hedefe yeniden hizalanma (`re-align toward target`).
    """

    def __init__(
        self,
        stuck_threshold_px: float = UNSTUCK_STUCK_THRESHOLD_PX,
        stuck_duration_sec: float = UNSTUCK_STUCK_DURATION_SEC,
        strafe_duration_sec: float = UNSTUCK_STEP_DURATION_SEC,
        reverse_duration_sec: float = UNSTUCK_STEP_DURATION_SEC,
        reverse_pivot_duration_sec: float = 0.4,
        optical_flow_threshold_px: float = 0.6,
        minimap_roi: Tuple[int, int, int, int] = (1640, 40, 240, 240),
        center_mask_radius: int = 16,
    ) -> None:
        self.stuck_threshold_px: float = float(max(0.1, stuck_threshold_px))
        self.stuck_duration_sec: float = float(max(0.5, stuck_duration_sec))
        self.strafe_duration_sec: float = float(max(0.2, strafe_duration_sec))
        self.reverse_duration_sec: float = float(max(0.2, reverse_duration_sec))
        self.reverse_pivot_duration_sec: float = float(max(0.1, reverse_pivot_duration_sec))
        self.optical_flow_threshold_px: float = float(max(0.1, optical_flow_threshold_px))
        self.minimap_roi: Tuple[int, int, int, int] = minimap_roi
        self.center_mask_radius: int = int(max(0, center_mask_radius))

        self._prev_terrain_f32: Optional[np.ndarray] = None
        self._prev_viewport_gray: Optional[np.ndarray] = None
        self._hann_window: Optional[np.ndarray] = None
        self._stuck_start_ts: Optional[float] = None
        self._last_update_ts: Optional[float] = None
        self._unstuck_count: int = 0
        self._last_shift_px: float = 0.0
        self._last_flow_px: float = 0.0
        self._recent_shifts: List[float] = []

    def reset(self) -> None:
        """Faz korelasyon geçmişini, optik akış referansını ve takılma zamanlayıcısını sıfırlar."""
        self._prev_terrain_f32 = None
        self._prev_viewport_gray = None
        self._stuck_start_ts = None
        self._last_update_ts = None
        self._last_shift_px = 0.0
        self._last_flow_px = 0.0
        self._recent_shifts.clear()
        self._unstuck_count = 0

    def extract_minimap_terrain_gray(
        self,
        frame_or_minimap: np.ndarray,
        minimap_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Tam ekran kareden veya doğrudan verilmiş minimap görüntüsünden arazi
        gri-seviye (`float32`) matrisini çıkarır. Merkezdeki sabit oyuncu okunu
        yumuşak biçimde maskeler ki `cv2.phaseCorrelate` yalnızca akan araziyi ölçsün.
        """
        if frame_or_minimap is None or not isinstance(frame_or_minimap, np.ndarray):
            return None
        if frame_or_minimap.size == 0:
            return None

        h_img, w_img = frame_or_minimap.shape[:2]

        # Eğer tam ekran kare verilmişse sağ üst minimap bölgesini kırp.
        # Varsayılan ROI 1920x1080 içindir (F-0247): başka çözünürlükte
        # orantılı ölçeklenir. Zaten minimap kırpımı olan küçük karelere
        # ROI uygulanmaz (tam ekran varsayımı yalnızca >= 800x450 karede).
        if minimap_roi is not None:
            roi = minimap_roi
            roi_is_fullscreen = True
        else:
            sx = w_img / 1920.0
            sy = h_img / 1080.0
            rx0, ry0, rw0, rh0 = self.minimap_roi
            roi = (
                int(round(rx0 * sx)),
                int(round(ry0 * sy)),
                int(round(rw0 * sx)),
                int(round(rh0 * sy)),
            )
            roi_is_fullscreen = w_img >= 800 and h_img >= 450
        rx, ry, rw, rh = roi
        if (
            roi_is_fullscreen
            and w_img > rw + 60
            and h_img > rh + 60
            and rx + rw <= w_img
            and ry + rh <= h_img
        ):
            crop = frame_or_minimap[ry : ry + rh, rx : rx + rw]
        else:
            crop = frame_or_minimap

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        gray_f32 = gray.astype(np.float32)
        ch, cw = gray_f32.shape[:2]
        if ch < 16 or cw < 16:
            return None

        # Sabit merkez oyuncu ikonunu yerel medyan/ortalama ile nötralize et
        if self.center_mask_radius > 0 and min(ch, cw) > self.center_mask_radius * 3:
            cy, cx = ch // 2, cw // 2
            mean_val = float(np.mean(gray_f32))
            cv2.circle(gray_f32, (cx, cy), self.center_mask_radius, mean_val, -1)

        return gray_f32

    def compute_terrain_shift(
        self,
        prev_terrain_f32: np.ndarray,
        curr_terrain_f32: np.ndarray,
    ) -> Tuple[float, Tuple[float, float], float]:
        """
        İki ardışık minimap arazi kırpımı arasındaki alt-piksel (`sub-pixel`) kaymayı
        `cv2.phaseCorrelate` ile hesaplar.

        Dönüş:
            `(shift_magnitude_px, (dx, dy), response)`
        """
        if (
            prev_terrain_f32 is None
            or curr_terrain_f32 is None
            or prev_terrain_f32.shape != curr_terrain_f32.shape
        ):
            return 0.0, (0.0, 0.0), 0.0

        h, w = curr_terrain_f32.shape[:2]
        if self._hann_window is None or self._hann_window.shape != (h, w):
            self._hann_window = cv2.createHanningWindow((w, h), cv2.CV_32F)

        (dx, dy), response = cv2.phaseCorrelate(
            prev_terrain_f32,
            curr_terrain_f32,
            self._hann_window,
        )
        shift_px = float(math.hypot(dx, dy))
        return shift_px, (float(dx), float(dy)), float(response)

    def compute_viewport_optical_flow(
        self,
        frame: Optional[np.ndarray],
        viewport_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> float:
        """
        Karakter 3B dünyada ileri yürürken ekranın alt-orta koridorundaki
        (yürünen zemin) optik akış (optical flow) büyüklüğünü hesaplar.
        Karakter engele (kaya/ağaç/duvar) takıldığında zemin pikselleri akmayı durdurur.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return 0.0
        h, w = frame.shape[:2]
        if h < 80 or w < 120:
            return 0.0

        if viewport_roi is not None:
            vx, vy, vw, vh = viewport_roi
            crop = frame[max(0, vy) : min(h, vy + vh), max(0, vx) : min(w, vx + vw)]
        else:
            # 3B alt koridor / zemin bölgesi
            crop = frame[int(h * 0.55) : int(h * 0.85), int(w * 0.35) : int(w * 0.65)]

        if crop.size == 0:
            return 0.0

        if crop.ndim == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Düşük hesaplama yükü için 80x45 boyutuna küçült
        small_gray = cv2.resize(gray, (80, 45), interpolation=cv2.INTER_AREA)

        if self._prev_viewport_gray is None or self._prev_viewport_gray.shape != small_gray.shape:
            self._prev_viewport_gray = small_gray
            return 0.0  # İlk kare referansı: henüz ölçüm yok (F-0428)

        try:
            flow = cv2.calcOpticalFlowFarneback(
                self._prev_viewport_gray,
                small_gray,
                None,
                pyr_scale=0.5,
                levels=2,
                winsize=11,
                iterations=2,
                poly_n=5,
                poly_sigma=1.1,
                flags=0,
            )
            mag = float(np.mean(np.hypot(flow[..., 0], flow[..., 1])))
        except Exception:
            # Yedek: mutlak kare farkı üzerinden kaba hareket kestirimi
            diff = cv2.absdiff(self._prev_viewport_gray, small_gray)
            mag = float(np.mean(diff)) / 12.0

        self._prev_viewport_gray = small_gray
        self._last_flow_px = mag
        return mag

    def update_walking_frame(
        self,
        frame_or_minimap: np.ndarray,
        is_walking: bool = True,
        now: Optional[float] = None,
        minimap_roi: Optional[Tuple[int, int, int, int]] = None,
        viewport_roi: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Karakter yürürken (`is_walking=True`, `W` basılı) her karede çağrılır.
        Minimap phaseCorrelate kayması (< 1.2 px) ve 3B koridor optik akışı
        birlikte değerlendirilir; < 2.5s (varsayılan 2.5s) boyunca takılma sürerse
        `is_stuck=True` döndürür.
        """
        curr_ts = time.time() if now is None else float(now)

        if not is_walking:
            self.reset()
            return {
                "is_stuck": False,
                "shift_px": 0.0,
                "flow_px": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "stuck_elapsed_sec": 0.0,
                "phase_response": 0.0,
            }

        curr_terrain = self.extract_minimap_terrain_gray(
            frame_or_minimap, minimap_roi=minimap_roi
        )
        if curr_terrain is None:
            # F-0427: arazi kırpımı yokken takılma zamanlayıcısını sıfırla;
            # aksi halde bir sonraki geçerli karede şişmiş elapsed ile
            # erken takılma ilan edilebilir.
            self._stuck_start_ts = None
            return {
                "is_stuck": False,
                "shift_px": 0.0,
                "flow_px": 0.0,
                "dx": 0.0,
                "dy": 0.0,
                "stuck_elapsed_sec": 0.0,
                "phase_response": 0.0,
            }

        # Optik akış analizi (eğer tam ekran kare verilmişse)
        flow_px = 0.0
        if frame_or_minimap.ndim >= 2 and frame_or_minimap.shape[1] >= 300 and frame_or_minimap.shape[0] >= 200:
            flow_px = self.compute_viewport_optical_flow(frame_or_minimap, viewport_roi=viewport_roi)

        if self._prev_terrain_f32 is None or self._prev_terrain_f32.shape != curr_terrain.shape:
            self._prev_terrain_f32 = curr_terrain
            self._last_update_ts = curr_ts
            self._stuck_start_ts = curr_ts
            return {
                "is_stuck": False,
                "shift_px": 0.0,
                "flow_px": round(flow_px, 4),
                "dx": 0.0,
                "dy": 0.0,
                "stuck_elapsed_sec": 0.0,
                "phase_response": 0.0,  # F-0428: henüz kayma ölçülmedi
            }

        shift_px, (dx, dy), response = self.compute_terrain_shift(
            self._prev_terrain_f32, curr_terrain
        )
        self._prev_terrain_f32 = curr_terrain
        self._last_update_ts = curr_ts
        self._last_shift_px = shift_px

        self._recent_shifts.append(shift_px)
        if len(self._recent_shifts) > 30:
            self._recent_shifts.pop(0)

        # Takılma tespiti (F-0079, F-0426): minimap kayması tek başına
        # yetmez. Son 3 karenin medyan kayması eşiğin altındaysa minimap
        # takılıdır; tam ekran karede optik akış da ölçüldüyse akışın da
        # durmuş olması gerekir (optical_flow_threshold_px artık ölü bir
        # sabit değil). Minimap kırpımı verildiğinde akış ölçülemez; o
        # durumda yalnızca minimap kararı geçerlidir.
        recent_window = self._recent_shifts[-3:]
        smoothed_shift_px = float(np.median(recent_window)) if recent_window else shift_px
        minimap_stuck = smoothed_shift_px < self.stuck_threshold_px
        flow_measured = bool(
            frame_or_minimap.ndim >= 2
            and frame_or_minimap.shape[1] >= 300
            and frame_or_minimap.shape[0] >= 200
        )
        flow_stuck = bool(flow_measured and flow_px < self.optical_flow_threshold_px)
        stuck_candidate = bool(minimap_stuck and flow_stuck) if flow_measured else bool(minimap_stuck)

        if stuck_candidate:
            if self._stuck_start_ts is None:
                self._stuck_start_ts = curr_ts
            stuck_elapsed = max(0.0, curr_ts - self._stuck_start_ts)
        else:
            self._stuck_start_ts = None
            stuck_elapsed = 0.0

        is_stuck = bool(stuck_elapsed >= self.stuck_duration_sec)
        return {
            "is_stuck": is_stuck,
            "shift_px": round(shift_px, 4),
            "shift_px_smoothed": round(smoothed_shift_px, 4),
            "flow_px": round(flow_px, 4),
            "flow_measured": flow_measured,
            "minimap_stuck": bool(minimap_stuck),
            "flow_stuck": flow_stuck,
            "dx": round(dx, 4),
            "dy": round(dy, 4),
            "stuck_elapsed_sec": round(stuck_elapsed, 3),
            "phase_response": round(response, 4),
        }

    @staticmethod
    def _press_key_hold(input_controller: Any, key: str, duration: float) -> bool:
        """
        S-2801: press_key-only kontrolcüde süreli basılı tutma.

        `InputDriver.press_key` varsayılan 40-90 ms tap'tir; süreli tutma
        yalnızca `hold_sec` verilirse olur (bkz. core/input_driver.py:3340
        ve combat.py LoS kurtarma). `hold_sec` desteklemeyen kontrolcüde
        tap + boş bekleme sahte manevradır — o durumda False döner;
        çağıran `executed: True` raporlamaz. press_key-only API ile iki
        tuş aynı anda tutulamaz; A+W strafe burada ardışık tutuşlarla
        yapılır (combat.py'daki LoS yaklaşımıyla aynı yaklaşıklık).
        """
        try:
            input_controller.press_key(key, hold_sec=float(duration))
        except TypeError:
            return False
        return True

    def execute_jump(
        self,
        input_controller: Any = None,
        wait_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Fiziksel zıplama (Space) komutunu yürütür."""
        # F-0011: kontrolcü yokken hiçbir tuşa basılmadı; True dönmek üst
        # katmana manevra uygulandı yalanını söyler.
        if input_controller is None:
            return False
        # F-0012: bekleme fonksiyonu verilmediyse gerçek bekleme
        # (time.sleep) kullanılır; no-op varsayılan süreli manevrayı
        # ~0 ms yapıyordu.
        fn = wait_fn if callable(wait_fn) else time.sleep
        if hasattr(input_controller, "tap_key") and callable(input_controller.tap_key):
            input_controller.tap_key("space")
            return True
        if hasattr(input_controller, "press_key") and callable(input_controller.press_key):
            input_controller.press_key("space")
            return True
        if hasattr(input_controller, "key_down") and hasattr(input_controller, "key_up"):
            input_controller.key_down("space")
            fn(0.06)
            input_controller.key_up("space")
            return True
        return False

    def execute_lateral_strafe(
        self,
        strafe_key: str = "a",
        duration: Optional[float] = None,
        input_controller: Any = None,
        wait_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Yanal sıyrılma (A+W veya D+W strafe) manevrasını yürütür."""
        # F-0011: kontrolcü yokken strafe uygulanmadı.
        if input_controller is None:
            return False
        dur = float(duration if duration is not None else self.strafe_duration_sec)
        # F-0012: bekleme verilmediyse gerçek bekleme kullanılır.
        fn = wait_fn if callable(wait_fn) else time.sleep
        k = strafe_key.lower()
        if hasattr(input_controller, "hold_keys") and callable(input_controller.hold_keys):
            input_controller.hold_keys([k, "w"], duration=dur)
            return True
        if hasattr(input_controller, "key_down") and hasattr(input_controller, "key_up"):
            input_controller.key_down("w")
            input_controller.key_down(k)
            fn(dur)
            input_controller.key_up(k)
            input_controller.key_up("w")
            return True
        if hasattr(input_controller, "press_key") and callable(input_controller.press_key):
            # S-2801: press_key varsayılan 40-90 ms tap'tir; hold_sec
            # verilmeden 1.2 sn'lik A+W strafe gerçekleşmezdi, kod boş
            # bekleyip True dönerdi. hold_sec desteklenmiyorsa False
            # döner (sahte True yok); ardışık tutuş, press_key-only
            # API'de mümkün olan en yakın karşılıktır.
            strafe_held = self._press_key_hold(input_controller, k, dur)
            forward_held = self._press_key_hold(input_controller, "w", dur)
            return bool(strafe_held and forward_held)
        return False

    def execute_reverse_backout(
        self,
        duration: Optional[float] = None,
        pivot_direction: str = "a",
        input_controller: Any = None,
        wait_fn: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """
        Fiziksel geri çekilme (reverse-backout) manevrası:
        1. W tuşunu bırak.
        2. S tuşu ile geri geri yürü (1.0 - 1.2 sn).
        3. S + A/D ile kuyruk açısı vererek engelin köşesinden çık (0.4 sn).
        4. Space ile zıpla (tümsek/kök kurtarma).
        """
        # F-0011: kontrolcü yokken hiçbir tuşa basılmadı.
        if input_controller is None:
            return False
        dur = float(duration if duration is not None else self.reverse_duration_sec)
        pivot_dur = float(self.reverse_pivot_duration_sec)
        # F-0012: bekleme verilmediyse gerçek bekleme kullanılır.
        fn = wait_fn if callable(wait_fn) else time.sleep
        pk = pivot_direction.lower()
        acted = False
        # S-2801: manevrayı tanımlayan geri hareketin kendisi (adım 2/3)
        # gerçekten uygulandı mı? Yalnızca W bırakma + Space zıplama
        # gerçekleştiyse geri çekilme olmadı; True dönmek sahte olur.
        reversed_ok = False

        # 1. W tuşunu güvenli bırak
        if hasattr(input_controller, "release_key") and callable(input_controller.release_key):
            input_controller.release_key("w")
            acted = True
        elif hasattr(input_controller, "key_up") and callable(input_controller.key_up):
            input_controller.key_up("w")
            acted = True

        # 2. S ile geri çık
        if hasattr(input_controller, "hold_keys") and callable(input_controller.hold_keys):
            input_controller.hold_keys(["s"], duration=dur)
            acted = True
            reversed_ok = True
        elif hasattr(input_controller, "key_down") and hasattr(input_controller, "key_up"):
            input_controller.key_down("s")
            fn(dur)
            input_controller.key_up("s")
            acted = True
            reversed_ok = True
        elif hasattr(input_controller, "press_key") and callable(input_controller.press_key):
            # S-2801: hold_sec verilmeden S tap'i 40-90 ms'de bırakılır,
            # geri çekilme gerçekleşmezdi. Desteklenmiyorsa acted False
            # kalır (sahte True yok).
            if self._press_key_hold(input_controller, "s", dur):
                acted = True
                reversed_ok = True
        else:
            fn(dur)

        # 3. Yanal geri dönüş (S + A veya S + D)
        if hasattr(input_controller, "hold_keys") and callable(input_controller.hold_keys):
            input_controller.hold_keys(["s", pk], duration=pivot_dur)
            acted = True
            reversed_ok = True
        elif hasattr(input_controller, "key_down") and hasattr(input_controller, "key_up"):
            input_controller.key_down("s")
            input_controller.key_down(pk)
            fn(pivot_dur)
            input_controller.key_up(pk)
            input_controller.key_up("s")
            acted = True
            reversed_ok = True
        elif hasattr(input_controller, "press_key") and callable(input_controller.press_key):
            # S-2801: hold_sec verilmeden S+A tap'i pivot dönüşü
            # yapmazdı. Desteklenmiyorsa acted değişmez (sahte True yok).
            if self._press_key_hold(input_controller, "s", pivot_dur):
                acted = True
                reversed_ok = True
            if self._press_key_hold(input_controller, pk, pivot_dur):
                acted = True
                reversed_ok = True
        else:
            fn(pivot_dur)

        # 4. Zıpla
        if self.execute_jump(input_controller=input_controller, wait_fn=fn):
            acted = True

        # S-2801: geri hareket hiç uygulanmadıysa (yalnızca W bırakma /
        # Space) manevra gerçekleşmedi sayılır.
        return bool(acted and reversed_ok)

    def execute_unstuck_sequence(
        self,
        input_controller: Any = None,
        target_angle_deg: Optional[float] = None,
        current_angle_deg: Optional[float] = None,
        realign_callback: Optional[Callable[[Optional[float]], None]] = None,
        sleep_fn: Optional[Callable[[float], None]] = None,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        maneuver_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Takılma tespit edildiğinde fiziksel kurtulma manevralarını uygular:
        - maneuver_type="jump": Sadece Zıpla (Space).
        - maneuver_type="lateral_strafe": Yanal strafe (A+W veya D+W).
        - maneuver_type="reverse_backout": Geri çekilme (Stop W -> S -> S+A/D -> Space -> Realign).
        - maneuver_type=None (otomatik kademelendirme):
            1. deneme: Space -> A + W (1.2s) -> Realign
            2. deneme: Space -> D + W (1.2s) -> Realign
            3. deneme: Reverse-Backout (Stop W -> S 1.2s -> S+A 0.4s -> Space -> Realign)
            4. deneme: Extended Reverse + Request Repath
        """
        tier = (self._unstuck_count) % 4
        self._unstuck_count += 1
        # F-0012: sleep_fn verilmediyse gerçek bekleme kullanılır; no-op
        # varsayılan, süreli manevraları ~0 ms yapıyordu.
        wait_fn = sleep_fn if callable(sleep_fn) else time.sleep

        # Manevra türü belirleme
        chosen_maneuver = maneuver_type
        if chosen_maneuver is None:
            if tier == 0:
                chosen_maneuver = "jump_strafe"
                strafe_key = "a"
            elif tier == 1:
                chosen_maneuver = "jump_strafe"
                strafe_key = "d"
            elif tier == 2:
                chosen_maneuver = "reverse_backout"
                strafe_key = "a"
            else:
                chosen_maneuver = "repath"
                strafe_key = "d"
        else:
            strafe_key = "a" if (self._unstuck_count % 2 == 1) else "d"

        actions_taken: List[str] = []
        combo_label = f"{strafe_key.upper()}+W"
        request_repath = False
        maneuver_ok = False

        if chosen_maneuver == "jump":
            actions_taken.append("SPACE")
            maneuver_ok = self.execute_jump(input_controller=input_controller, wait_fn=wait_fn)

        elif chosen_maneuver == "lateral_strafe":
            actions_taken.append(combo_label)
            maneuver_ok = self.execute_lateral_strafe(
                strafe_key=strafe_key,
                duration=self.strafe_duration_sec,
                input_controller=input_controller,
                wait_fn=wait_fn,
            )

        elif chosen_maneuver == "reverse_backout":
            actions_taken.extend(["STOP_W", "REVERSE_S", "REVERSE_STRAFE", "SPACE"])
            maneuver_ok = self.execute_reverse_backout(
                duration=self.reverse_duration_sec,
                pivot_direction=strafe_key,
                input_controller=input_controller,
                wait_fn=wait_fn,
            )

        elif chosen_maneuver == "repath":
            request_repath = True
            actions_taken.extend(["STOP_W", "EXTENDED_REVERSE", "PIVOT_90", "SPACE", "REQUEST_REPATH"])
            maneuver_ok = self.execute_reverse_backout(
                duration=self.reverse_duration_sec * 1.3,
                pivot_direction=strafe_key,
                input_controller=input_controller,
                wait_fn=wait_fn,
            )
            if input_controller is not None and hasattr(input_controller, "turn_by_angle") and callable(input_controller.turn_by_angle):
                turn_deg = 90.0 if strafe_key == "d" else -90.0
                try:
                    input_controller.turn_by_angle(turn_deg, duration=0.25)
                except TypeError:
                    input_controller.turn_by_angle(turn_deg)

        else:
            # Default "jump_strafe" (Tier 0 & 1): Space -> A+W/D+W
            actions_taken.append("SPACE")
            jump_ok = self.execute_jump(input_controller=input_controller, wait_fn=wait_fn)

            actions_taken.append(combo_label)
            strafe_ok = self.execute_lateral_strafe(
                strafe_key=strafe_key,
                duration=self.strafe_duration_sec,
                input_controller=input_controller,
                wait_fn=wait_fn,
            )
            maneuver_ok = bool(jump_ok and strafe_ok)

        # Hedefe yeniden hizalanma (Re-align toward target via shortest rotational path + deadband)
        actions_taken.append("REALIGN")
        angle_delta = 0.0
        damped_turn: Optional[Dict[str, Any]] = None
        realigned_flag = False

        if target_angle_deg is not None and current_angle_deg is not None:
            angle_delta = ((float(target_angle_deg) - float(current_angle_deg) + 180.0) % 360.0) - 180.0
            damped_turn = apply_turn_damping(angle_delta, deadband_deg=deadband_deg)

        if callable(realign_callback) and target_angle_deg is not None:
            # F-0080: callback'in dönüş değeri kontrol edilir.
            cb_result = realign_callback(target_angle_deg)
            realigned_flag = cb_result is not False
        elif input_controller is not None and target_angle_deg is not None:
            if damped_turn is None or damped_turn["turn_needed"]:
                if hasattr(input_controller, "realign_to_target") and callable(
                    input_controller.realign_to_target
                ):
                    cb_result = input_controller.realign_to_target(target_angle_deg)
                    realigned_flag = cb_result is not False
                elif hasattr(input_controller, "turn_by_angle") and callable(
                    input_controller.turn_by_angle
                ):
                    turn_dur = damped_turn.get("turn_duration_sec") if damped_turn else None
                    try:
                        turn_result = input_controller.turn_by_angle(angle_delta, duration=turn_dur)
                    except TypeError:
                        turn_result = input_controller.turn_by_angle(angle_delta)
                    realigned_flag = turn_result is not False
            else:
                # Ölü bölgede: dönüş gerekmez, hedefe zaten hizalı sayılır.
                realigned_flag = True
        else:
            # F-0080: ne callback ne kontrolcü ne de hedef açı var — hiçbir
            # hizalama yapılmadı, bayrak True olamaz.
            realigned_flag = False

        # Takılma sayacını sıfırla ki yeni manevra sonrası temiz ölçüm başlasın
        self._stuck_start_ts = None

        logger.info(
            "Fiziksel Kurtulma (#%d, manevra=%s) uygulandı: %s (strafe=%.2fs, açı_farkı=%.1f°, ölü_bölge=%s, repath=%s)",
            self._unstuck_count,
            chosen_maneuver,
            " -> ".join(actions_taken),
            self.strafe_duration_sec,
            angle_delta,
            bool(damped_turn and damped_turn.get("in_deadband")),
            request_repath,
        )

        return {
            # F-0011: executed yalnızca manevra gerçekten uygulandıysa True.
            "executed": bool(maneuver_ok),
            "maneuver_type": chosen_maneuver,
            "sequence": actions_taken,
            "strafe_key": strafe_key,
            "strafe_combo": combo_label,
            "strafe_duration_sec": self.strafe_duration_sec,
            "reverse_duration_sec": self.reverse_duration_sec if "REVERSE" in str(actions_taken) else 0.0,
            "request_repath": bool(request_repath),
            "realigned": bool(realigned_flag),
            "angle_delta_deg": round(angle_delta, 2),
            "in_deadband": bool(damped_turn.get("in_deadband", False)) if damped_turn else False,
            "turn_needed": bool(damped_turn.get("turn_needed", True)) if damped_turn else True,
            "turn_duration_sec": float(damped_turn.get("turn_duration_sec", 0.0)) if damped_turn else 0.0,
            "unstuck_count": self._unstuck_count,
        }


# ==============================================================================
# MADDE 27 — YÜKSEKLİK / UÇURUM FARKI TESPİTİ (ZAxisCliffFilter)
# ==============================================================================
class ZAxisCliffFilter:
    """
    Madde 27 — Yükseklik / Uçurum Farkı Tespiti (`ZAxisCliffFilter`):

    2B minimap üzerinde çok yakın (`< 25 px`) görünmesine rağmen ardışık `2` yaklaşma
    denemesinde ulaşılamayan veya etkileşime girilemeyen (uçurumun tepesinde ya da
    kanyonun dibinde kalan) düğümleri:
        - `unreachable_cliff = True`
        - `altitude_band = "cliff_mismatch"`
    olarak etiketler ve uçuş modu (`is_flying=True`) aktif olmadıkça veya aynı irtifa
    bandında bulunulmadıkça yer koridoru seçiminden dışlar.
    """

    def __init__(
        self,
        close_minimap_radius_px: float = 25.0,
        max_failed_approaches: int = 2,
        z_elevation_tolerance: float = 12.0,
    ) -> None:
        self.close_minimap_radius_px: float = float(max(1.0, close_minimap_radius_px))
        self.max_failed_approaches: int = int(max(1, max_failed_approaches))
        self.z_elevation_tolerance: float = float(max(1.0, z_elevation_tolerance))
        self._approach_failures: Dict[Union[int, str], int] = {}

    def record_approach_attempt(
        self,
        node: Any,
        minimap_dist_px: Optional[float] = None,
        interaction_succeeded: bool = False,
        now: Optional[float] = None,
    ) -> bool:
        """
        Bir düğüme yapılan yaklaşma denemesinin sonucunu kaydeder.
        Eğer `minimap_dist_px < 25.0` olduğu halde ardışık `2` denemede etkileşim
        başarısız olursa düğümü `unreachable_cliff = True` ve
        `altitude_band = "cliff_mismatch"` olarak etiketler ve `True` döndürür.
        """
        if node is None:
            return False

        key = _node_identifier(node)
        check_ts = time.time() if now is None else float(now)

        if interaction_succeeded:
            self._approach_failures[key] = 0
            _set_node_attr(node, "failed_approach_attempts", 0)
            _set_node_attr(node, "unreachable_cliff", False)
            if _get_node_attr(node, "altitude_band") == "cliff_mismatch":
                _set_node_attr(node, "altitude_band", "ground")
            return False

        # Yalnızca 2B minimap üzerinde yakın (< 25 px) göründüğü halde ulaşılamayan düğümleri say;
        # mesafe bilinmiyorsa (None) uçurum atfı yapılmaz.
        if minimap_dist_px is None:
            return bool(_get_node_attr(node, "unreachable_cliff", False))
        if float(minimap_dist_px) < self.close_minimap_radius_px:
            prev_fails = int(
                _get_node_attr(
                    node,
                    "failed_approach_attempts",
                    self._approach_failures.get(key, 0),
                )
                or 0
            )
            curr_fails = prev_fails + 1
            self._approach_failures[key] = curr_fails
            _set_node_attr(node, "failed_approach_attempts", curr_fails)
            _set_node_attr(node, "last_cliff_check_ts", check_ts)

            if curr_fails >= self.max_failed_approaches:
                self.mark_node_as_cliff(node)
                logger.warning(
                    "Uçurum/İrtifa farkı tespit edildi: düğüm=%r (2B mesafe=%.1f px, başarısız_deneme=%d) -> unreachable_cliff=True",
                    key,
                    float(minimap_dist_px),
                    curr_fails,
                )
                return True

        return bool(_get_node_attr(node, "unreachable_cliff", False))

    def mark_node_as_cliff(
        self,
        node: Any,
        altitude_band: str = "cliff_mismatch",
        fly_unreachable: bool = False,
    ) -> None:
        """Düğümü uçurum / irtifa uyumsuzluğu olarak etiketler."""
        if node is None:
            return
        _set_node_attr(node, "unreachable_cliff", True)
        _set_node_attr(node, "altitude_band", str(altitude_band or "cliff_mismatch"))
        if fly_unreachable:
            _set_node_attr(node, "fly_unreachable", True)

    def clear_cliff_tag(self, node: Any) -> None:
        """Düğüm üzerindeki uçurum etiketini ve başarısız yaklaşma sayacını temizler."""
        if node is None:
            return
        key = _node_identifier(node)
        self._approach_failures.pop(key, None)
        _set_node_attr(node, "failed_approach_attempts", 0)
        _set_node_attr(node, "unreachable_cliff", False)
        if _get_node_attr(node, "altitude_band") == "cliff_mismatch":
            _set_node_attr(node, "altitude_band", "ground")

    def is_node_excluded_by_cliff(
        self,
        node: Any,
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        player_z: Optional[float] = None,
    ) -> bool:
        """
        Düğümün uçurum/irtifa filtresi nedeniyle mevcut koridor seçiminden
        dışlanıp dışlanmayacağını (`True` = dışla / seçme) döndürür.
        """
        if node is None:
            return True

        fly_unreachable = bool(_get_node_attr(node, "fly_unreachable", False))
        if fly_unreachable:
            return True

        unreachable_cliff = bool(_get_node_attr(node, "unreachable_cliff", False))
        node_band = str(_get_node_attr(node, "altitude_band", "ground") or "ground").strip().lower()

        # Açık Z-koordinatı verilmişse ve irtifa farkı toleransı aşıyorsa
        node_z = _get_node_z(node)
        if player_z is not None and node_z is not None and not is_flying:
            try:
                if abs(float(player_z) - float(node_z)) > self.z_elevation_tolerance:
                    return True
            except (TypeError, ValueError):
                # B-14 (fail-closed): Z karşılaştırması yapılamıyorsa düğüm
                # "uçurum değil" sayılmaz — irtifa uyumu kanıtlanamadığı
                # için düğüm dışlanır.
                logger.warning(
                    "Uçurum filtresi Z ayrıştırma hatası (player_z=%r, node_z=%r); "
                    "düğüm irtifa uyumu kanıtlanamadığı için dışlandı.",
                    player_z, node_z,
                )
                return True

        if not unreachable_cliff and node_band != "cliff_mismatch":
            # Eğer oyuncunun irtifa bandı belirtilmişse ve düğüm farklı bir özel irtifa bandındaysa
            if (
                player_altitude_band is not None
                and node_band not in ("", "ground", "default", "normal")
                and node_band != str(player_altitude_band).strip().lower()
                and not is_flying
            ):
                return True
            return False

        # Uçurum olarak etiketlenmiş düğüm:
        # 1) Uçuş modundaysak (ve fly_unreachable değilse) ulaşılabilir
        if is_flying:
            return False

        # 2) Oyuncu düğümle birebir aynı özel irtifa bandındaysa ("cliff_mismatch" hariç)
        if (
            player_altitude_band is not None
            and node_band not in ("", "cliff_mismatch")
            and str(player_altitude_band).strip().lower() == node_band
        ):
            return False

        # 3) Oyuncu Z yüksekliği düğüm Z yüksekliği ile eşleşiyorsa
        if player_z is not None and node_z is not None:
            try:
                if abs(float(player_z) - float(node_z)) <= self.z_elevation_tolerance:
                    return False
            except (TypeError, ValueError):
                # B-14: Z eşleşmesi kanıtlanamadı — akış zaten aşağıdaki
                # "dışla" (return True) dalına düşer; arıza görünür olsun.
                logger.warning(
                    "Uçurum filtresi Z eşleşme hatası (player_z=%r, node_z=%r); "
                    "düğüm dışlandı.",
                    player_z, node_z,
                )

        return True

    def filter_ground_corridor_nodes(
        self,
        nodes: Sequence[Any],
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        player_z: Optional[float] = None,
    ) -> List[Any]:
        """Verilen düğüm listesinden uçurum/irtifa uyumsuz olanları ayıklar."""
        if not nodes:
            return []
        return [
            n
            for n in nodes
            if not self.is_node_excluded_by_cliff(
                n,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                player_z=player_z,
            )
        ]


# ==============================================================================
# MADDE 28 — DİNLENME SÜRESİ TAHMİNLEYİCİ (PredictivePreWalkScheduler)
# ==============================================================================
class PredictivePreWalkScheduler:
    """
    Madde 28 — Dinlenme Süresi (Cooldown) Tahminleyici (`PredictivePreWalkScheduler`):

    5 dakikalık (`300 sn`) dinlenme süresinin bitmesine `<= 8 sn` kalan ve o düğüme
    tahmini yürüyüş süresi `~10 sn` olan (`rest_remaining_sec <= estimated_walk_sec + 2.0`)
    düğümleri henüz cooldown bitmeden erken seçer ve yürüyüşü başlatır; böylece karakter
    materyalin yeniden doğduğu (`respawn`) tam saniyede kaynağın başında olur.

    H-sweep (dilim 12 O-4) — AÇIK SORU, değerler kalibre edilmedi: 28 px/sn
    x 0.55 m/px = 15.4 m/sn "yürüme" hızı binek mertebesindedir; gerçek
    yürüme hızı, respawn süresi (300 sn varsayımı) ve konumsuz 10 sn
    varsayımı oyun içi ölçümle doğrulanmadan değiştirilmedi. Yanlış hız
    yalnızca erken-yürüyüş zamanlamasını kaydırır (toplama kaybı değil,
    zaman kaybı).
    """

    def __init__(
        self,
        default_rest_duration_sec: int = 300,
        max_prewalk_remaining_sec: float = 8.0,
        arrival_margin_sec: float = 2.0,
        walk_speed_px_per_sec: float = 28.0,
        default_walk_sec: float = 10.0,
    ) -> None:
        self.default_rest_duration_sec: int = int(max(1, default_rest_duration_sec))
        self.max_prewalk_remaining_sec: float = float(max(0.5, max_prewalk_remaining_sec))
        self.arrival_margin_sec: float = float( arrival_margin_sec )
        self.walk_speed_px_per_sec: float = float(max(1.0, walk_speed_px_per_sec))
        self.default_walk_sec: float = float(max(1.0, default_walk_sec))

    def remaining_rest_seconds(
        self,
        node: Any,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
    ) -> float:
        """Düğümün 5 dakikalık (`300 sn`) dinlenme süresinden kalan saniyeyi hesaplar."""
        if node is None:
            return 0.0
        try:
            if hasattr(node, "remaining_rest_sec") and callable(node.remaining_rest_sec):
                return max(
                    0.0,
                    float(node.remaining_rest_sec(now=now, rest_duration_sec=rest_duration_sec)),
                )
            explicit_rem = _get_node_attr(node, "rest_remaining_sec", None)
            if explicit_rem is not None:
                return max(0.0, float(explicit_rem))

            last_ts = float(
                _get_node_attr(
                    node,
                    "last_visited_timestamp",
                    _get_node_attr(node, "last_visited", 0.0),
                )
                or 0.0
            )
            if last_ts <= 0.0:
                return 0.0
            curr = time.time() if now is None else float(now)
            rest = float(
                rest_duration_sec
                if rest_duration_sec is not None
                else (
                    _get_node_attr(node, "rest_duration_sec", self.default_rest_duration_sec)
                    or self.default_rest_duration_sec
                )
            )
            return max(0.0, rest - (curr - last_ts))
        except Exception as exc:
            logger.debug("remaining_rest_seconds hatası: %s", exc)
            return 0.0

    def estimate_walk_time_sec(
        self,
        player_pos: Optional[Point2D],
        node: Any,
        walk_speed_px_per_sec: Optional[float] = None,
    ) -> float:
        """Oyuncunun mevcut konumundan hedef düğüme tahmini yürüyüş süresini (sn) döndürür."""
        if node is None:
            return self.default_walk_sec

        explicit_walk = _get_node_attr(node, "estimated_walk_sec", None)
        if explicit_walk is not None:
            try:
                return max(0.5, float(explicit_walk))
            except (TypeError, ValueError):
                # B-14: kayıtlı yürüyüş süresi ayrıştırılamadı — varsayılan
                # süreye sessizce düşülmez, arıza loglanır.
                logger.warning(
                    "Tahmini yürüyüş süresi ayrıştırılamadı "
                    "(estimated_walk_sec=%r); varsayılan süre kullanılıyor.",
                    explicit_walk,
                )

        if player_pos is None:
            return self.default_walk_sec

        # P1-7 (K-20): koordinatsız düğümün yürüme süresi (0,0)
        # hayaletine göre ölçülmez — varsayılan süre döner.
        _node_xy = _extract_node_xy_or_none(node)
        if _node_xy is None:
            logger.warning(
                "Tahmini yürüyüş süresi: koordinatsız düğüm; varsayılan "
                "süre (%.1f sn) kullanılıyor.",
                self.default_walk_sec,
            )
            return self.default_walk_sec
        nx, ny = _node_xy
        px, py = float(player_pos[0]), float(player_pos[1])
        dist_px = math.hypot(nx - px, ny - py)
        speed = float(walk_speed_px_per_sec or self.walk_speed_px_per_sec)
        return max(0.5, dist_px / max(1.0, speed))

    def evaluate_prewalk_candidate(
        self,
        node: Any,
        player_pos: Optional[Point2D] = None,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Bir düğümün erken yürüyüş (`pre-walk`) için uygun olup olmadığını değerlendirir.

        Koşul:
        - Düğüm şu anda dinleniyor olmalı (`rest_remaining_sec > 0`)
        - `rest_remaining_sec <= estimated_walk_sec + 2.0` (veya kalan süre `<= 8.0 sn`
          olup yürüyüş süresiyle uyumlu) olmalı.
        """
        rem_sec = self.remaining_rest_seconds(
            node, now=now, rest_duration_sec=rest_duration_sec
        )
        walk_sec = self.estimate_walk_time_sec(player_pos, node)
        threshold_sec = walk_sec + self.arrival_margin_sec

        is_resting = rem_sec > 0.0
        within_walk_window = rem_sec <= threshold_sec
        # F-0073: 8 sn sınırı gerçek bir sınır olmalı; max(8, threshold)
        # her zaman threshold'a eşit/büyük olduğundan eski cap koşulu,
        # walk-window sağlandığında otomatik doğruydu (ölü kod).
        within_prewalk_cap = rem_sec <= self.max_prewalk_remaining_sec

        eligible = bool(is_resting and within_walk_window and within_prewalk_cap)
        arrival_delta_sec = walk_sec - rem_sec  # >= 0 ise varıldığında kaynak çoktan doğmuş olur

        return {
            "eligible": eligible,
            "rest_remaining_sec": round(rem_sec, 3),
            "estimated_walk_sec": round(walk_sec, 3),
            "threshold_sec": round(threshold_sec, 3),
            "arrival_delta_sec": round(arrival_delta_sec, 3),
            "within_walk_window": bool(within_walk_window),
            "within_prewalk_cap": bool(within_prewalk_cap),
            "max_prewalk_remaining_sec": round(self.max_prewalk_remaining_sec, 3),
        }

    def should_pre_walk_to_node(
        self,
        node: Any,
        player_pos: Optional[Point2D] = None,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
    ) -> bool:
        """Düğüme erken yürüyüş (`Predictive Pre-Walk`) başlatılması gerekiyorsa `True` döndürür."""
        info = self.evaluate_prewalk_candidate(
            node=node,
            player_pos=player_pos,
            now=now,
            rest_duration_sec=rest_duration_sec,
        )
        return bool(info["eligible"])


# ==============================================================================
# MADDE 29 — DİNAMİK DÜĞÜM KOORDİNATI İYİLEŞTİRME (CentroidDriftCorrector)
# ==============================================================================
class CentroidDriftCorrector:
    """
    Madde 29 — Dinamik Düğüm Koordinatı İyileştirme (`CentroidDriftCorrector`):

    Başlangıçta uzaktan (3B tepe ikonu veya minimap kenarından) otomatik haritalanan
    bir düğüm ziyaret edilip kaynağın tam başında `[F Gather]` tuşuna basıldığında,
    düğümün kayıtlı `(x, y)` koordinatını üstel hareketli ortalama (EMA) veya gerçek
    oyuncu konumu ile günceller (`refine_node_coordinates_on_gather(node, actual_x, actual_y)`).
    Böylece sonraki turlarda karakter kaynağın tam milimetrik merkezine yürür.
    """

    def __init__(
        self,
        # F-0239: oyuncu düğümün tam üstünde değil, etkileşim mesafesinde
        # durur; düğümü oyuncu konumuna %85 çekmek kayıtlı koordinatı bozar.
        # Muhafazakâr EMA: ilk uzaktan-haritalı hasatta %50, sonra %35.
        ema_alpha: float = 0.35,
        far_mapped_first_gather_alpha: float = 0.50,
        map_width: float = MAP_CANVAS_WIDTH,
        map_height: float = MAP_CANVAS_HEIGHT,
    ) -> None:
        self.ema_alpha: float = float(min(1.0, max(0.05, ema_alpha)))
        self.far_mapped_first_gather_alpha: float = float(
            min(1.0, max(0.1, far_mapped_first_gather_alpha))
        )
        self.map_width: float = float(max(1.0, map_width))
        self.map_height: float = float(max(1.0, map_height))

    def refine_node_coordinates_on_gather(
        self,
        node: Any,
        actual_x: float,
        actual_y: float,
        alpha: Optional[float] = None,
        exact_snap: bool = False,
    ) -> Dict[str, Any]:
        """
        `[F Gather]` etkileşimi gerçekleştiğinde düğümün `(x, y)` koordinatlarını
        gerçek oyuncu konumuna (`actual_x`, `actual_y`) göre iyileştirir.
        """
        if node is None:
            return {
                "updated": False,
                "old_x": 0.0,
                "old_y": 0.0,
                "new_x": float(actual_x),
                "new_y": float(actual_y),
                "drift_px": 0.0,
                "alpha_used": 0.0,
            }

        old_x, old_y = _extract_node_xy(
            node, fallback_width=self.map_width, fallback_height=self.map_height
        )

        # H-sweep (dilim 12 O-1 + K-3): koordinatsız düğümün eski konumu
        # yoktur; oyuncu konumuna "exact_snap" yapmak yer tutucu (0, 0)
        # koordinatı kalıcı "gerçek" konuma çevirirdi. İyileştirme yok.
        if not _node_has_coordinates(node):
            return {
                "updated": False,
                "old_x": float(old_x),
                "old_y": float(old_y),
                "new_x": float(old_x),
                "new_y": float(old_y),
                "drift_px": 0.0,
                "alpha_used": 0.0,
                "reason": "no_coordinates",
            }

        target_x = float(actual_x)
        target_y = float(actual_y)

        refinement_count = int(_get_node_attr(node, "gather_refinement_count", 0) or 0)
        source_tag = str(
            _get_node_attr(node, "discovery_source", _get_node_attr(node, "source", "")) or ""
        ).strip().lower()

        if exact_snap:
            eff_alpha = 1.0
        elif alpha is not None:
            eff_alpha = float(min(1.0, max(0.05, alpha)))
        elif refinement_count == 0 and source_tag in (
            "3d_crest",
            "minimap_edge",
            "auto_far",
            "auto_discovery",
            "crest",
        ):
            eff_alpha = self.far_mapped_first_gather_alpha
        else:
            eff_alpha = self.ema_alpha

        new_x = (1.0 - eff_alpha) * old_x + eff_alpha * target_x
        new_y = (1.0 - eff_alpha) * old_y + eff_alpha * target_y
        drift_px = math.hypot(new_x - old_x, new_y - old_y)

        # Hem x/y hem map_x/map_y alanlarını senkron güncelle
        if isinstance(node, dict):
            if "x" in node or "map_x" not in node:
                node["x"] = round(new_x, 3)
            if "y" in node or "map_y" not in node:
                node["y"] = round(new_y, 3)
            if "map_x" in node:
                node["map_x"] = round(new_x, 3)
            if "map_y" in node:
                node["map_y"] = round(new_y, 3)
            if "norm_x" in node and self.map_width > 0:
                node["norm_x"] = round(new_x / self.map_width, 6)
            if "norm_y" in node and self.map_height > 0:
                node["norm_y"] = round(new_y / self.map_height, 6)
        else:
            if hasattr(node, "x"):
                setattr(node, "x", round(new_x, 3))
            if hasattr(node, "y"):
                setattr(node, "y", round(new_y, 3))
            if hasattr(node, "map_x"):
                setattr(node, "map_x", round(new_x, 3))
            if hasattr(node, "map_y"):
                setattr(node, "map_y", round(new_y, 3))
            if hasattr(node, "norm_x") and self.map_width > 0:
                setattr(node, "norm_x", round(new_x / self.map_width, 6))
            if hasattr(node, "norm_y") and self.map_height > 0:
                setattr(node, "norm_y", round(new_y / self.map_height, 6))

        _set_node_attr(node, "gather_refinement_count", refinement_count + 1)
        _set_node_attr(node, "last_drift_correction_px", round(drift_px, 3))

        logger.debug(
            "Düğüm koordinatı iyileştirildi (EMA alpha=%.2f): (%.2f, %.2f) -> (%.2f, %.2f), kayma=%.2f px",
            eff_alpha,
            old_x,
            old_y,
            new_x,
            new_y,
            drift_px,
        )

        return {
            "updated": True,
            "old_x": round(old_x, 3),
            "old_y": round(old_y, 3),
            "new_x": round(new_x, 3),
            "new_y": round(new_y, 3),
            "drift_px": round(drift_px, 3),
            "alpha_used": round(eff_alpha, 3),
            "refinement_count": refinement_count + 1,
        }


def refine_node_coordinates_on_gather(
    node: Any,
    actual_x: float,
    actual_y: float,
    alpha: Optional[float] = None,
    exact_snap: bool = False,
) -> Dict[str, Any]:
    """
    Madde 29 için doğrudan çağrılabilir yardımcı fonksiyon:
    `[F Gather]` basıldığında düğümün `(x, y)` koordinatını EMA / gerçek konumla günceller.
    """
    corrector = CentroidDriftCorrector()
    return corrector.refine_node_coordinates_on_gather(
        node=node,
        actual_x=actual_x,
        actual_y=actual_y,
        alpha=alpha,
        exact_snap=exact_snap,
    )


# ==============================================================================
# MADDE 30 — RAKİP OYUNCU TESPİTİ (ContestedNodeDetector)
# ==============================================================================
# İsimlik (nameplate) yatay-şekil kuralı: genişlik >= 18 px, yükseklik >= 3 px,
# en/boy >= 2.0, alan >= 24 px. Gri ve renkli dallarda aynı kural geçerliydi;
# iki ayrı kopya yerine tek yardımcıda tanımlanır (D07).
_NAMEPLATE_MIN_WIDTH_PX = 18
_NAMEPLATE_MIN_HEIGHT_PX = 3
_NAMEPLATE_MIN_ASPECT = 2.0
_NAMEPLATE_MIN_AREA_PX = 24
# Yatay morfolojik kapama çekirdeği (7x2): isimlik metni/barı yatay süreklidir.
_NAMEPLATE_HORIZ_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 2))


def _has_valid_nameplate_blob(num_labels: int, stats: np.ndarray) -> bool:
    """`connectedComponentsWithStats` çıktısında yatay isimlik bloğu var mı?"""
    for lbl in range(1, num_labels):
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        if (
            bw >= _NAMEPLATE_MIN_WIDTH_PX
            and bh >= _NAMEPLATE_MIN_HEIGHT_PX
            and bw >= bh * _NAMEPLATE_MIN_ASPECT
            and area >= _NAMEPLATE_MIN_AREA_PX
        ):
            return True
    return False


class ContestedNodeDetector:
    """
    Madde 30 — Rakip Oyuncu (`ContestedNodeDetector`):

    Hedef düğümün ekran koordinatı (`node_screen_xy`) üzerinde başka bir oyuncu
    karakterinin / isimliğinin (`nameplate`) durup toplama animasyonu (`gather animation`)
    yaptığını tespit eder (`is_node_contested(frame, node_screen_xy) -> bool`).
    Eğer düğüm rakip oyuncu tarafından toplanıyorsa vakit kaybetmeden doğrudan
    sıradaki (`Next in Sequence`) düğüme atlar.
    """

    def __init__(
        self,
        nameplate_roi_half_w: int = 90,
        nameplate_roi_top_offset: int = 125,
        gather_aura_radius_px: int = 60,
        min_nameplate_pixels: int = 42,
        min_gather_glow_pixels: int = 28,
        contested_cooldown_sec: float = 45.0,
    ) -> None:
        self.nameplate_roi_half_w: int = int(max(25, nameplate_roi_half_w))
        self.nameplate_roi_top_offset: int = int(max(35, nameplate_roi_top_offset))
        self.gather_aura_radius_px: int = int(max(20, gather_aura_radius_px))
        self.min_nameplate_pixels: int = int(max(10, min_nameplate_pixels))
        self.min_gather_glow_pixels: int = int(max(8, min_gather_glow_pixels))
        self.contested_cooldown_sec: float = float(max(5.0, contested_cooldown_sec))

    def is_node_contested(
        self,
        frame: Optional[np.ndarray],
        node_screen_xy: Optional[Tuple[float, float]],
        prev_frame: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Hedef düğümün ekran konumunda (`node_screen_xy`) başka bir oyuncunun isimliği
        (`nameplate` / sağlık çubuğu) ve toplama animasyonu (`gather glow` / karakter
        hareketi) olup olmadığını tespit eder.
        """
        if frame is None or node_screen_xy is None:
            return False
        if not isinstance(frame, np.ndarray) or frame.size == 0 or frame.ndim < 2:
            return False

        h, w = frame.shape[:2]
        sx = int(round(float(node_screen_xy[0])))
        sy = int(round(float(node_screen_xy[1])))

        if sx < 0 or sx >= w or sy < 0 or sy >= h:
            return False

        # H-sweep (dilim 12 O-2): ROI sabitleri 1920x1080 içindir; canlı
        # çözünürlük (2560x1440) farklıysa orantılı ölçeklenir, yoksa 1440p'de
        # isimlik ROI dışında kalıp contested tespiti hep False dönerdi.
        sx_scale = float(w) / 1920.0
        sy_scale = float(h) / 1080.0
        np_half_w = max(25, int(round(self.nameplate_roi_half_w * sx_scale)))
        np_top_off = max(35, int(round(self.nameplate_roi_top_offset * sy_scale)))
        ga_radius = max(20, int(round(self.gather_aura_radius_px * sx_scale)))

        # 1) Düğümün hemen üstündeki Nameplate (Oyuncu İsimliği / Bar) ROI analizi
        np_x1 = max(0, sx - np_half_w)
        np_x2 = min(w, sx + np_half_w)
        np_y1 = max(0, sy - np_top_off)
        np_y2 = min(h, max(np_y1 + 12, sy - 10))

        # 2) Düğümün tam üzerindeki Karakter + Gather Animasyon Parıltısı ROI analizi
        ga_r = ga_radius
        ga_x1 = max(0, sx - ga_r)
        ga_x2 = min(w, sx + ga_r)
        ga_y1 = max(0, sy - ga_r)
        ga_y2 = min(h, sy + ga_r // 2 + 10)

        if (np_x2 - np_x1) < 16 or (np_y2 - np_y1) < 8 or (ga_x2 - ga_x1) < 16 or (ga_y2 - ga_y1) < 12:
            return False

        if frame.ndim == 2:
            # Gri-seviye kare verilmişse yüksek kontrastlı yatay isimlik + merkez aktivite kontrolü
            np_patch = frame[np_y1:np_y2, np_x1:np_x2]
            ga_patch = frame[ga_y1:ga_y2, ga_x1:ga_x2]
            bright_np = int(np.count_nonzero(np_patch > 215))
            # Kısa devre (D07): geçerli isimlik bloğu en az 24 px alan ister;
            # daha az parlak piksel varsa ne şekil ne kenar sonucu True yapamaz
            # (sonuç zaten `bright_np >= min_nameplate_pixels` gerektirir).
            if bright_np < _NAMEPLATE_MIN_AREA_PX:
                return False
            # F-0074: parlak piksel sayısına ek olarak isimliğin yatay şekli
            # de doğrulanır (renkli daldaki kuralın gri karşılığı:
            # genişlik >= 18 px, en/boy >= 2.0, alan >= 24).
            bright_mask = (np_patch > 215).astype(np.uint8)
            closed_np = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, _NAMEPLATE_HORIZ_KERNEL)
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed_np, connectivity=8)
            # Kısa devre (D07): geçerli isimlik yoksa sonuç False'dur
            # (son dönüş `has_valid_nameplate and ...` idi); Canny boşa çalışmaz.
            if not _has_valid_nameplate_blob(num_labels, stats):
                return False
            edges_ga = cv2.Canny(ga_patch, 80, 180)
            edge_cnt = int(np.count_nonzero(edges_ga))
            return bool(
                bright_np >= self.min_nameplate_pixels
                and edge_cnt >= self.min_gather_glow_pixels * 2
            )

        np_bgr = frame[np_y1:np_y2, np_x1:np_x2]
        ga_bgr = frame[ga_y1:ga_y2, ga_x1:ga_x2]

        np_hsv = cv2.cvtColor(np_bgr, cv2.COLOR_BGR2HSV)
        ga_hsv = cv2.cvtColor(ga_bgr, cv2.COLOR_BGR2HSV)

        # Oyuncu isimliği renk maskeleri:
        # - Parlak beyaz/açık mavi oyuncu adı metni: düşük doygunluk, çok yüksek parlaklık
        mask_white_text = cv2.inRange(
            np_hsv,
            np.array([0, 0, 215], dtype=np.uint8),
            np.array([180, 55, 255], dtype=np.uint8),
        )
        # - Dost/Grup/Lejyon yeşil-turkuaz isimlik veya HP barı
        mask_ally_bar = cv2.inRange(
            np_hsv,
            np.array([38, 110, 165], dtype=np.uint8),
            np.array([105, 255, 255], dtype=np.uint8),
        )
        # - Rakip ırk (Asmodian/Elyos) kırmızı-pembe-turuncu isimlik
        mask_hostile_1 = cv2.inRange(
            np_hsv,
            np.array([0, 130, 165], dtype=np.uint8),
            np.array([12, 255, 255], dtype=np.uint8),
        )
        mask_hostile_2 = cv2.inRange(
            np_hsv,
            np.array([165, 130, 165], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8),
        )

        nameplate_mask = cv2.bitwise_or(
            cv2.bitwise_or(mask_white_text, mask_ally_bar),
            cv2.bitwise_or(mask_hostile_1, mask_hostile_2),
        )

        # Kısa devre (D07): geçerli isimlik bloğu en az 24 px alan ister;
        # daha az isimlik pikseli varsa şekil aramak boşa morfoloji +
        # connectedComponents maliyetidir.
        total_np_pixels = int(np.count_nonzero(nameplate_mask))
        if total_np_pixels < _NAMEPLATE_MIN_AREA_PX:
            return False

        # İsimlik yatay süreklilik doğrulaması (morfolojik yatay kapama ile yatay metin/bar bloğu bul)
        closed_np = cv2.morphologyEx(nameplate_mask, cv2.MORPH_CLOSE, _NAMEPLATE_HORIZ_KERNEL)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed_np, connectivity=8)

        # Kısa devre (D07): geçerli isimlik yoksa son dönüş
        # (`has_valid_nameplate and has_gather_activity`) zaten False'dur;
        # gather maskeleri + hareket farkı boşa çalışmaz. Eski koşul
        # (`not has_valid and total < min*2`) yalnızca küçük toplamda
        # erken dönerdi, büyük toplamda aynı False'a pahalı yoldan varırdı.
        if not _has_valid_nameplate_blob(num_labels, stats):
            return False

        # Gather animasyonu (parlak mavi/camgöbeği/altın toplama efektleri + karakter silüeti/hareketi)
        mask_gather_cyan = cv2.inRange(
            ga_hsv,
            np.array([82, 115, 185], dtype=np.uint8),
            np.array([132, 255, 255], dtype=np.uint8),
        )
        mask_gather_gold = cv2.inRange(
            ga_hsv,
            np.array([16, 120, 195], dtype=np.uint8),
            np.array([36, 255, 255], dtype=np.uint8),
        )
        gather_glow_pixels = int(
            np.count_nonzero(cv2.bitwise_or(mask_gather_cyan, mask_gather_gold))
        )

        # Karakter silüeti / animasyon hareketi kontrolü
        motion_pixels = 0
        if (
            prev_frame is not None
            and isinstance(prev_frame, np.ndarray)
            and prev_frame.shape == frame.shape
        ):
            prev_ga_bgr = prev_frame[ga_y1:ga_y2, ga_x1:ga_x2]
            diff = cv2.absdiff(
                cv2.cvtColor(ga_bgr, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_ga_bgr, cv2.COLOR_BGR2GRAY),
            )
            motion_pixels = int(np.count_nonzero(diff > 22))

        has_gather_activity = bool(
            gather_glow_pixels >= self.min_gather_glow_pixels
            or motion_pixels >= self.min_gather_glow_pixels
        )

        # Geçerli isimlik yukarıda doğrulandı (yoksa erken False dönüldü).
        return bool(has_gather_activity)

    def mark_node_contested(
        self,
        node: Any,
        now: Optional[float] = None,
        cooldown_sec: Optional[float] = None,
    ) -> None:
        """Düğümü geçici süreyle rakip oyuncu tarafından tutulmuş (`contested`) olarak işaretler."""
        if node is None:
            return
        curr_ts = time.time() if now is None else float(now)
        cd = float(cooldown_sec if cooldown_sec is not None else self.contested_cooldown_sec)
        _set_node_attr(node, "is_contested", True)
        _set_node_attr(node, "contested_until_ts", curr_ts + cd)

    def is_node_currently_contested_flag(
        self,
        node: Any,
        now: Optional[float] = None,
    ) -> bool:
        """Düğüm üzerindeki geçici `contested_until_ts` kilidinin aktif olup olmadığını kontrol eder."""
        if node is None:
            return False
        curr_ts = time.time() if now is None else float(now)
        until_ts = float(_get_node_attr(node, "contested_until_ts", 0.0) or 0.0)
        if until_ts > curr_ts:
            return True
        if until_ts > 0.0 and until_ts <= curr_ts:
            _set_node_attr(node, "is_contested", False)
            _set_node_attr(node, "contested_until_ts", 0.0)
            return False
        return bool(_get_node_attr(node, "is_contested", False))


def is_node_contested(
    frame: Optional[np.ndarray],
    node_screen_xy: Optional[Tuple[float, float]],
    prev_frame: Optional[np.ndarray] = None,
) -> bool:
    """
    Madde 30 için doğrudan çağrılabilir fonksiyon:
    Hedef düğüm üzerinde başka bir oyuncu toplama animasyonu yapıyorsa `True` döndürür.
    """
    detector = ContestedNodeDetector()
    return detector.is_node_contested(
        frame=frame,
        node_screen_xy=node_screen_xy,
        prev_frame=prev_frame,
    )


# ==============================================================================
# BİRLEŞİK KORİDOR NAVİGATÖRÜ (Maddeler 26..30 Orkestratörü)
# ==============================================================================
class CorridorNavigator:
    """
    Maddeler 26..30'u tek bir üretim-hazır (production-ready) navigasyon
    katmanında birleştiren ana sınıf:
    - `unstuck_maneuver` (`PhaseCorrelateUnstuckManeuver`)
    - `cliff_filter` (`ZAxisCliffFilter`)
    - `prewalk_scheduler` (`PredictivePreWalkScheduler`)
    - `drift_corrector` (`CentroidDriftCorrector`)
    - `contested_detector` (`ContestedNodeDetector`)
    """

    def __init__(
        self,
        route_planner: Optional[Any] = None,
        unstuck_maneuver: Optional[PhaseCorrelateUnstuckManeuver] = None,
        cliff_filter: Optional[ZAxisCliffFilter] = None,
        prewalk_scheduler: Optional[PredictivePreWalkScheduler] = None,
        drift_corrector: Optional[CentroidDriftCorrector] = None,
        contested_detector: Optional[ContestedNodeDetector] = None,
        turn_damper: Optional[TurnDampingController] = None,
    ) -> None:
        self.route_planner = route_planner
        self.unstuck_maneuver = unstuck_maneuver or PhaseCorrelateUnstuckManeuver()
        self.cliff_filter = cliff_filter or ZAxisCliffFilter()
        self.prewalk_scheduler = prewalk_scheduler or PredictivePreWalkScheduler()
        self.drift_corrector = drift_corrector or CentroidDriftCorrector()
        self.contested_detector = contested_detector or ContestedNodeDetector()
        self.turn_damper = turn_damper or TurnDampingController()
        # ExemplarMemory tutamacı önbelleği (D07): `get_exemplar_memory()`
        # singleton döndürür; her karede yeniden import + kilit + çağrı
        # yapmak yerine tutamaç bir kez çözülür. Yükleme başarısız olursa
        # her karede yeniden denenip uyarı seli üretilmez; yalnızca
        # `exemplar_memory_retry_sec` geçtikten sonra bir kez daha denenir.
        # (v2r-R04: bu 4 satır D07'de `attach_to_sequential_planner`
        # içindeki `return True` sonrasına düşmüş, ölü kod olmuştu;
        # `__init__` hiç çalıştırmadığı için `_get_exemplar_memory`
        # her `evaluate_corridor_path` çağrısında AttributeError veriyordu.
        # v2r-R02 aynı kök nedeni bağımsız doğruladı — 8 koridor testi
        # bu blokla onarılır. v2r-R05 de aynı düzeltmeyi yaptı; birleşmede
        # bloğun ikinci kopyası oluştuğu için temizlendi. v2r-R10a da
        # aynı düzeltmeyi byte-birebir yaptı — 3 d07 testi bu blokla onarılır.)
        self._exemplar_memory: Any = None
        self._exemplar_memory_loaded: bool = False
        self._exemplar_memory_failed_at: float = 0.0
        self.exemplar_memory_retry_sec: float = 60.0
        # F-0075 groundwork: route_planner verildiyse planner'ın kendi
        # alt nesneleri benimsenir (tek durum; çift filtre/zamanlayıcı
        # yok) ve navigator planner'a kaydedilir. isinstance kontrolü
        # Mock planner'ların otomatik özniteliklerini benimsemeyi
        # engeller. Bu tek başına üretimi bağlamaz — üretimde
        # CorridorNavigator kurulması (main.py / state_machine.py)
        # dilim dışıdır; bkz. attach_to_sequential_planner.
        if route_planner is not None:
            self._adopt_planner_subobjects(route_planner)
            try:
                route_planner.corridor_navigator = self
            except Exception:
                pass

    def _adopt_planner_subobjects(self, planner: Any) -> None:
        """Planner'ın paylaşımlı alt nesnelerini benimser (yalnızca
        gerçek sınıf örnekleri; Mock öznitelikler benimsenmez)."""
        for attr, cls in (
            ("cliff_filter", ZAxisCliffFilter),
            ("prewalk_scheduler", PredictivePreWalkScheduler),
            ("contested_detector", ContestedNodeDetector),
            # H-sweep (dilim 12 D-3): eskiden yalnızca 3 alt nesne benimseniyordu;
            # planner'ın unstuck/drift/turn-damper ayarları yok sayılıp navigator
            # kendi varsayılanlarını kuruyor, iki ayrı "gerçek" oluşuyordu.
            ("unstuck_maneuver", PhaseCorrelateUnstuckManeuver),
            ("drift_corrector", CentroidDriftCorrector),
            ("turn_damper", TurnDampingController),
        ):
            try:
                candidate = getattr(planner, attr, None)
            except Exception:
                continue
            if isinstance(candidate, cls):
                try:
                    setattr(self, attr, candidate)
                except Exception:
                    pass

    def attach_to_sequential_planner(self, planner: Any) -> bool:
        """
        F-0075 groundwork — CorridorNavigator'ı üretimdeki
        SequentialRoutePlanner örneğine bağlar: planner'ın alt
        nesneleri benimsenir ve ``planner.corridor_navigator = self``
        kurulur; üretim döngüsü artık
        ``planner.corridor_navigator.step_navigation(...)`` çağırabilir
        (contested/cliff/unstuck tek noktadan çalışır).

        Kalan adım dilim dışıdır: üretimde CorridorNavigator
        kurulmadığı için bu metodun çağıranı yok — main.py'da
        ``SequentialRoutePlanner()`` kurulumunun yanına tek satır
        (``CorridorNavigator(route_planner=planner)``) ve
        state_machine.py düğüm seçim döngüsünde step_navigation
        çağrısı gerekir. O yüzden F-0075 bu dilimde fixed değil,
        blocked — cross-slice kalır. İdempotenttir.
        """
        if planner is None:
            return False
        self.route_planner = planner
        self._adopt_planner_subobjects(planner)
        try:
            planner.corridor_navigator = self
        except Exception as exc:
            logger.warning("CorridorNavigator: planner'a kaydedilemedi: %s", exc)
            return False
        return True

    def _get_exemplar_memory(self) -> Tuple[Any, bool]:
        """Önbelleğe alınmış ExemplarMemory tutamacını döndürür.

        Dönüş: `(memory_or_None, load_failed)`. Başarılı yükleme (None
        dönmesi dahil — örn. test mock'u) bir kez çözülür ve saklanır;
        başarısızlık 60 sn boyunca hatırlanır, bu sürede yeniden deneme
        ve uyarı günlüğü yapılmaz.
        """
        if self._exemplar_memory_loaded:
            return self._exemplar_memory, False
        if self._exemplar_memory_failed_at > 0.0 and (
            time.time() - self._exemplar_memory_failed_at
        ) < self.exemplar_memory_retry_sec:
            return None, True
        try:
            from dataset_factory.exemplar_memory import get_exemplar_memory

            self._exemplar_memory = get_exemplar_memory()
            self._exemplar_memory_loaded = True
            self._exemplar_memory_failed_at = 0.0
            return self._exemplar_memory, False
        except Exception as exc:
            # F-0076: yükleme hatası sessizce "temiz koridor" sayılmamalı.
            self._exemplar_memory_failed_at = time.time()
            logger.warning("ExemplarMemory yüklenemedi; koridor temiz sayılmayacak: %s", exc)
            return None, True

    def compute_bearing(self, from_pos: Point2D, to_pos: Point2D) -> float:
        """Calculates compass bearing (0°=North, 90°=East) from from_pos to to_pos."""
        return compute_bearing(from_pos, to_pos)

    def compute_angle_diff(self, target_bearing: float, player_heading: float) -> float:
        """Calculates shortest rotational angle difference in [-180°..+180°]."""
        return compute_angle_diff(target_bearing, player_heading)

    def apply_turn_damping(
        self,
        angle_diff: float,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        min_turn_sec: float = TURN_MIN_DURATION_SEC,
        max_turn_sec: float = TURN_MAX_DURATION_SEC,
    ) -> Dict[str, Any]:
        """Applies deadband zone (±5° to ±8°) and smooth turn duration interpolation."""
        return apply_turn_damping(
            angle_diff=angle_diff,
            deadband_deg=deadband_deg,
            min_turn_sec=min_turn_sec,
            max_turn_sec=max_turn_sec,
        )

    def calculate_turn(
        self,
        player_heading: float,
        target_bearing: float,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
    ) -> Dict[str, Any]:
        """Calculates damped steering turn command towards target bearing."""
        # F-0240: durum taşıyan turn_damper kullanılır; stateless
        # apply_turn_damping tersine-dönüş sönümlemesini (reversal damping)
        # hiç uygulamıyordu.
        self.turn_damper.deadband_deg = max(
            MIN_DEADBAND_DEG, min(MAX_DEADBAND_DEG, float(deadband_deg))
        )
        return self.turn_damper.calculate_turn(
            target_bearing=target_bearing,
            player_heading=player_heading,
        )

    def evaluate_corridor_path(
        self,
        frame: Optional[np.ndarray],
        yolo_detections: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates the forward walking corridor in the lower-middle 3D viewport
        (y: 0.52..0.82, x: 0.28..0.72) across 3 sub-lanes (Left, Center, Right):
        - Compares patches against learned obstacles (14: obstacle) and monsters (10: monster_3d)
        - Compares patches against learned walkable ground paths (15: path_ground)
        - Incorporates live YOLO detections (class 14, 10, 15) when available
        - Calculates steering adjustment (left bias, right bias, or center clear).
        Returns: {
            "obstacle_detected": bool,
            "monster_detected": bool,
            "path_detected": bool,
            "steering_bias": str ("clear", "steer_left", "steer_right"),
            "recommended_turn_offset_deg": float,
            "lane_scores": {"left": float, "center": float, "right": float},
        }
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return {
                "obstacle_detected": False,
                "monster_detected": False,
                "path_detected": False,
                "steering_bias": "clear",
                "recommended_turn_offset_deg": 0.0,
                "lane_scores": {"left": 0.5, "center": 0.5, "right": 0.5},
                "clear_terrain_found": False,
                "exemplar_memory_available": False,
                "exemplar_memory_error": False,
                "yolo_detections_used": 0,
                "yolo_detections_ignored": 0,
            }

        # D07: tutamaç navigator başına bir kez çözülür
        # (`_get_exemplar_memory`); başarısızlık 60 sn hatırlanır.
        mem, mem_load_failed = self._get_exemplar_memory()

        h, w = frame.shape[:2]
        y1, y2 = int(h * 0.52), int(h * 0.82)
        lanes = {
            "left": (int(w * 0.28), int(w * 0.44)),
            "center": (int(w * 0.44), int(w * 0.56)),
            "right": (int(w * 0.56), int(w * 0.72)),
        }

        lane_scores = {"left": 0.50, "center": 0.50, "right": 0.50}
        obstacle_detected = False
        monster_detected = False
        path_detected = False

        # 1. Visual Patch Scanning via ExemplarMemory
        for lane_name, (lx1, lx2) in lanes.items():
            patch = frame[y1:y2, lx1:lx2]
            if patch.size == 0:
                continue

            score = 0.50

            if mem is not None:
                # Negative exemplar check (rejected false targets / impassable rocks)
                is_neg, neg_sim, _ = mem.is_negative_patch(patch, threshold=0.72)
                if is_neg:
                    obstacle_detected = True
                    score -= 0.40 * float(neg_sim)

                # Class-specific positive exemplar checks:
                # 14: obstacle
                matched_specific = False
                obs_sim, _ = mem.match_class(patch, class_id=14)
                if obs_sim >= 0.70:
                    obstacle_detected = True
                    score -= 0.50 * float(obs_sim)
                    matched_specific = True

                # 10: monster_3d (steer around aggro/patrol monsters while pathing)
                mob_sim, _ = mem.match_class(patch, class_id=10)
                if mob_sim >= 0.70:
                    monster_detected = True
                    score -= 0.60 * float(mob_sim)
                    matched_specific = True

                # 15: path_ground (walkable ground corridor)
                ground_sim, _ = mem.match_class(patch, class_id=15)
                if ground_sim >= 0.68:
                    path_detected = True
                    score += 0.45 * float(ground_sim)
                    matched_specific = True

                # General positive exemplar classification fallback — yalnızca
                # sınıf-özel eşleşme bulunamadıysa (F-0241: aynı yama iki kez
                # cezalandırılmasın / ödüllendirilmesin).
                pos_cid, pos_cname, pos_sim = (None, None, 0.0)
                if not matched_specific:
                    pos_cid, pos_cname, pos_sim = mem.classify_patch(patch, min_sim=0.70)
                if pos_cid is not None:
                    cname_low = str(pos_cname or "").lower()
                    if pos_cid == 14 or "obstacle" in cname_low or "engel" in cname_low:
                        obstacle_detected = True
                        score -= 0.50 * float(pos_sim)
                    elif pos_cid == 10 or "monster" in cname_low or "mob" in cname_low:
                        monster_detected = True
                        score -= 0.60 * float(pos_sim)
                    elif pos_cid == 15 or "path" in cname_low or "ground" in cname_low:
                        path_detected = True
                        score += 0.40 * float(pos_sim)

            lane_scores[lane_name] = round(float(np.clip(score, 0.05, 0.95)), 3)

        # 2. Live YOLO Detections Integration (class 14, 10, 15)
        yolo_used = 0
        yolo_ignored = 0
        if yolo_detections:
            for det in yolo_detections:
                if not isinstance(det, dict):
                    continue
                cid = det.get("class_id")
                cname = str(det.get("class_name", "")).lower()
                conf = float(det.get("confidence", 0.70))

                cx, cy = None, None
                if "cx" in det and "cy" in det:
                    cx, cy = float(det["cx"]), float(det["cy"])
                elif "bbox" in det and len(det["bbox"]) == 4:
                    # F-0077: bbox xyxy'dir; merkezi sol-üst köşe değil,
                    # köşelerin ortalamasıdır (bbox_xyxy dalıyla aynı kural).
                    b = det["bbox"]
                    if max(b) <= 1.0:
                        cx = (float(b[0]) + float(b[2])) / 2.0
                        cy = (float(b[1]) + float(b[3])) / 2.0
                    else:
                        cx = (float(b[0]) + float(b[2])) / (2.0 * float(w))
                        cy = (float(b[1]) + float(b[3])) / (2.0 * float(h))
                elif "bbox_xyxy" in det and len(det["bbox_xyxy"]) == 4:
                    b = det["bbox_xyxy"]
                    cx = (float(b[0]) + float(b[2])) / (2.0 * float(w))
                    cy = (float(b[1]) + float(b[3])) / (2.0 * float(h))
                elif "bbox_xywh" in det and len(det["bbox_xywh"]) == 4:
                    b = det["bbox_xywh"]
                    cx = (float(b[0]) + 0.5 * float(b[2])) / float(w)
                    cy = (float(b[1]) + 0.5 * float(b[3])) / float(h)

                if cx is None or cy is None or not (0.30 <= cy <= 0.95):
                    # F-0242: şerit aralığı dışı tespit sessizce atılmaz;
                    # sayılır ve sonuçta raporlanır.
                    yolo_ignored += 1
                    continue

                det_lane = None
                if 0.20 <= cx < 0.44:
                    det_lane = "left"
                elif 0.44 <= cx <= 0.56:
                    det_lane = "center"
                elif 0.56 < cx <= 0.80:
                    det_lane = "right"

                if not det_lane:
                    yolo_ignored += 1
                    continue

                yolo_used += 1

                is_obstacle = (cid == 14) or ("obstacle" in cname) or ("engel" in cname)
                is_monster = (cid == 10) or ("monster" in cname) or ("mob" in cname)
                is_path = (cid == 15) or ("path_ground" in cname) or ("path" in cname and "ground" in cname)

                if is_obstacle:
                    obstacle_detected = True
                    lane_scores[det_lane] = round(float(np.clip(lane_scores[det_lane] - 0.45 * conf, 0.05, 0.95)), 3)
                elif is_monster:
                    monster_detected = True
                    lane_scores[det_lane] = round(float(np.clip(lane_scores[det_lane] - 0.55 * conf, 0.05, 0.95)), 3)
                elif is_path:
                    path_detected = True
                    lane_scores[det_lane] = round(float(np.clip(lane_scores[det_lane] + 0.40 * conf, 0.05, 0.95)), 3)

        c_score = lane_scores["center"]
        l_score = lane_scores["left"]
        r_score = lane_scores["right"]

        steering_bias = "clear"
        turn_offset = 0.0

        # Avoid hazards in center corridor or reverse if all lanes are blocked
        if c_score < 0.35 and l_score < 0.35 and r_score < 0.35:
            steering_bias = "reverse_needed"
            turn_offset = 180.0
        elif c_score < 0.42 or (obstacle_detected and c_score < max(l_score, r_score)) or (monster_detected and c_score < 0.45):
            if l_score >= r_score:
                steering_bias = "steer_left"
                turn_offset = -20.0 if monster_detected else -18.0
            else:
                steering_bias = "steer_right"
                turn_offset = 20.0 if monster_detected else 18.0
        elif path_detected:
            # Walkable ground preference: steer towards the best ground corridor
            best_lane = max(lane_scores, key=lambda k: lane_scores[k])
            if best_lane == "left" and l_score > c_score + 0.12:
                steering_bias = "steer_left"
                turn_offset = -10.0
            elif best_lane == "right" and r_score > c_score + 0.12:
                steering_bias = "steer_right"
                turn_offset = 10.0

        if yolo_ignored:
            logger.debug(
                "Koridor dışı %d YOLO tespiti şerit aralığı dışında bırakıldı (kullanılan: %d)",
                yolo_ignored,
                yolo_used,
            )
        return {
            "obstacle_detected": obstacle_detected,
            "monster_detected": monster_detected,
            "path_detected": path_detected,
            "steering_bias": steering_bias,
            "recommended_turn_offset_deg": turn_offset,
            "lane_scores": lane_scores,
            # F-0076: nötr 0.50 şerit skoru tek başına "temiz" sayılmaz;
            # exemplar memory yoksa ve YOLO patika kanıtı da yoksa temiz
            # ilan edilmez.
            "clear_terrain_found": bool(
                max(lane_scores.values()) >= 0.45 and (mem is not None or path_detected)
            ),
            "exemplar_memory_available": bool(mem is not None),
            "exemplar_memory_error": bool(mem_load_failed),
            "yolo_detections_used": yolo_used,
            "yolo_detections_ignored": yolo_ignored,
        }

    def steer_step(
        self,
        current_pos: Point2D,
        target_pos: Point2D,
        player_heading: float,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        frame: Optional[np.ndarray] = None,
        yolo_detections: Optional[Sequence[Dict[str, Any]]] = None,
        live_detections: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """Calculates bearing, shortest angular error, and damped turn command with obstacle avoidance."""
        bearing = compute_bearing(current_pos, target_pos)
        dets = yolo_detections if yolo_detections is not None else live_detections
        corridor_eval = (
            self.evaluate_corridor_path(frame, yolo_detections=dets)
            if frame is not None
            else None
        )
        adjusted_bearing = bearing
        if corridor_eval and corridor_eval.get("recommended_turn_offset_deg"):
            adjusted_bearing = (bearing + corridor_eval["recommended_turn_offset_deg"] + 360.0) % 360.0

        turn_data = self.calculate_turn(
            player_heading=player_heading,
            target_bearing=adjusted_bearing,
            deadband_deg=deadband_deg,
        )
        return {
            "target_bearing": bearing,
            "adjusted_bearing": adjusted_bearing,
            "player_heading": round(float(player_heading), 2),
            "corridor_eval": corridor_eval,
            **turn_data,
        }

    def step_navigation(
        self,
        frame: Optional[np.ndarray],
        nodes: Sequence[Any],
        current_seq_pos: int,
        player_pos: Optional[Point2D] = None,
        target_screen_xy: Optional[Tuple[float, float]] = None,
        is_walking: bool = True,
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        input_controller: Any = None,
        target_angle_deg: Optional[float] = None,
        current_angle_deg: Optional[float] = None,
        now: Optional[float] = None,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
        yolo_detections: Optional[Sequence[Dict[str, Any]]] = None,
        live_detections: Optional[Sequence[Any]] = None,
        force_unstick_maneuver: Optional[str] = None,
        prev_frame: Optional[np.ndarray] = None,
        player_z: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Her navigasyon adımında:
        1. Uçurum/irtifa filtresi (`ZAxisCliffFilter`) aktif düğümü dışlıyorsa
           sıradaki uygun düğüme geçer.
        2. Hedef düğümde rakip oyuncu (`ContestedNodeDetector`) varsa anında
           sıradaki (`Next in Sequence`) düğüme geçer.
        3. Yürürken (`W` basılı) minimap `cv2.phaseCorrelate` + optik akış ile `< 2.5 sn`
           kayma tespit edilirse fiziksel kurtulma (jump, lateral strafe, reverse-backout) uygular.
        4. Açısal hizalama ve ±5°..±8° ölü bölge sönümlemesiyle yalpalanmayı önler.
        5. İleri koridor görsel analizi (engel/canavar kaçınma, patika zemin önceliği).
        6. Erken yürüyüş (`PredictivePreWalkScheduler`) değerlendirmesi sonuca eklenir.
        Not: drift düzeltmesi burada değil, gerçek toplama anında
        `on_gather_success` üzerinden çağrılır (gerçek oyuncu konumu gerekir).
        """
        curr_ts = time.time() if now is None else float(now)
        node_list = list(nodes) if nodes else []
        n = len(node_list)
        active_idx = int(current_seq_pos) % n if (n > 0 and current_seq_pos >= 0) else 0
        active_node = node_list[active_idx] if n > 0 else None

        # H-sweep (dilim 12 K-3): koordinatsız düğüm harita merkezi (960, 540)
        # sayılmaz — hedef yapılmaz, sıradaki seçilebilir düğüme geçilir.
        if active_node is not None and not _node_has_coordinates(active_node):
            next_node, next_idx = self._advance_to_next_node(
                node_list,
                active_idx,
                player_pos=player_pos,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                now=curr_ts,
                player_z=player_z,
            )
            return {
                "status": "node_no_coordinates",
                "contested": False,
                "cliff_excluded": False,
                "no_coordinates": True,
                "unstuck_triggered": False,
                "selected_node": next_node,
                "selected_index": next_idx,
                "no_alternative_node": bool(next_node is None),
                "turn_evaluation": None,
                "prewalk": None,
            }

        # Turn evaluation if angles are provided
        turn_eval: Optional[Dict[str, Any]] = None
        if target_angle_deg is not None and current_angle_deg is not None:
            turn_eval = self.calculate_turn(
                player_heading=current_angle_deg,
                target_bearing=target_angle_deg,
                deadband_deg=deadband_deg,
            )

        # Madde 28: erken yürüyüş değerlendirmesi (F-0078).
        prewalk_info: Optional[Dict[str, Any]] = None
        if active_node is not None:
            prewalk_info = self.prewalk_scheduler.evaluate_prewalk_candidate(
                active_node, player_pos=player_pos, now=curr_ts
            )

        # Madde 27: uçurum/irtifa filtresi (F-0078) — aktif düğüm dışlanmışsa
        # sıradaki uygun düğüme geç.
        if (
            active_node is not None
            and self.cliff_filter.is_node_excluded_by_cliff(
                active_node,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                player_z=player_z,
            )
        ):
            next_node, next_idx = self._advance_to_next_node(
                node_list,
                active_idx,
                player_pos=player_pos,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                now=curr_ts,
                player_z=player_z,
            )
            return {
                "status": "cliff_skipped",
                "contested": False,
                "cliff_excluded": True,
                "unstuck_triggered": False,
                "selected_node": next_node,
                "selected_index": next_idx,
                "no_alternative_node": bool(next_node is None),
                "turn_evaluation": turn_eval,
                "prewalk": prewalk_info,
            }

        # Madde 30a (P1-11): 45 sn contested kilidi ana yolda okunur.
        # Düğüm önceki tick'te mark_node_contested ile kilitlendiyse,
        # bu karede görsel tespit ateşlenmese (kare yok / rakip ROI
        # dışında) bile kilit süresince hedef yapılmaz — kilit
        # tanımlıydı ama ana yol okumuyordu; _advance_to_next_node
        # zaten kilitli adayları atlar (F-0244), o yüzden sıradaki
        # uygun düğüme geçilir. Görsel tespit (Madde 30) bunun ardından
        # yeni kilitleri yakalamaya devam eder.
        if (
            active_node is not None
            and self.contested_detector.is_node_currently_contested_flag(active_node, now=curr_ts)
        ):
            next_node, next_idx = self._advance_to_next_node(
                node_list,
                active_idx,
                player_pos=player_pos,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                now=curr_ts,
                player_z=player_z,
            )
            return {
                "status": "contested_skipped",
                "contested": True,
                "unstuck_triggered": False,
                "selected_node": next_node,
                "selected_index": next_idx,
                "no_alternative_node": bool(next_node is None),
                "turn_evaluation": turn_eval,
                "prewalk": prewalk_info,
            }

        # Madde 30: Rakip oyuncu kontrolü (F-0078: prev_frame artık geçiliyor)
        if (
            frame is not None
            and target_screen_xy is not None
            and active_node is not None
            and self.contested_detector.is_node_contested(frame, target_screen_xy, prev_frame=prev_frame)
        ):
            self.contested_detector.mark_node_contested(active_node, now=curr_ts)
            next_node, next_idx = self._advance_to_next_node(
                node_list,
                active_idx,
                player_pos=player_pos,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                now=curr_ts,
                player_z=player_z,
            )
            return {
                "status": "contested_skipped",
                "contested": True,
                "unstuck_triggered": False,
                "selected_node": next_node,
                "selected_index": next_idx,
                "no_alternative_node": bool(next_node is None),
                "turn_evaluation": turn_eval,
                "prewalk": prewalk_info,
            }

        # Madde 26: 3B Engel Aşma & Fiziksel Kurtulma (PhaseCorrelate + Optical Flow Unstuck)
        unstuck_result: Optional[Dict[str, Any]] = None
        if force_unstick_maneuver:
            unstuck_result = self.unstuck_maneuver.execute_unstuck_sequence(
                input_controller=input_controller,
                target_angle_deg=target_angle_deg,
                current_angle_deg=current_angle_deg,
                deadband_deg=deadband_deg,
                maneuver_type=force_unstick_maneuver,
            )
        elif frame is not None and is_walking:
            walk_state = self.unstuck_maneuver.update_walking_frame(
                frame, is_walking=True, now=curr_ts
            )
            if walk_state["is_stuck"]:
                unstuck_result = self.unstuck_maneuver.execute_unstuck_sequence(
                    input_controller=input_controller,
                    target_angle_deg=target_angle_deg,
                    current_angle_deg=current_angle_deg,
                    deadband_deg=deadband_deg,
                )

        dets = yolo_detections if yolo_detections is not None else live_detections
        corridor_eval = (
            self.evaluate_corridor_path(frame, yolo_detections=dets)
            if frame is not None
            else None
        )

        # Eğer koridor tamamen tıkalıysa ve yürünüyorsa acil reverse-backout tetikle
        if (
            corridor_eval
            and corridor_eval.get("steering_bias") == "reverse_needed"
            and not unstuck_result
            and is_walking
        ):
            unstuck_result = self.unstuck_maneuver.execute_unstuck_sequence(
                input_controller=input_controller,
                target_angle_deg=target_angle_deg,
                current_angle_deg=current_angle_deg,
                deadband_deg=deadband_deg,
                maneuver_type="reverse_backout",
            )

        if (
            corridor_eval
            and corridor_eval.get("recommended_turn_offset_deg")
            and target_angle_deg is not None
            and current_angle_deg is not None
            and not unstuck_result
        ):
            adjusted_target_angle = (target_angle_deg + corridor_eval["recommended_turn_offset_deg"] + 360.0) % 360.0
            turn_eval = self.calculate_turn(
                player_heading=current_angle_deg,
                target_bearing=adjusted_target_angle,
                deadband_deg=deadband_deg,
            )
            turn_eval["corridor_adjusted"] = True
            turn_eval["original_target_bearing"] = target_angle_deg

        repath_requested = bool(unstuck_result.get("request_repath", False)) if unstuck_result else False
        selected_node = active_node
        selected_index = active_idx if n > 0 else -1

        if repath_requested:
            next_node, next_idx = self._advance_to_next_node(
                node_list,
                active_idx,
                player_pos=player_pos,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                now=curr_ts,
                player_z=player_z,
            )
            selected_node = next_node
            selected_index = next_idx

        unstuck_executed = bool(unstuck_result and unstuck_result.get("executed"))
        return {
            "status": (
                "unstuck_executed"
                if unstuck_executed
                else "unstuck_failed"
                if unstuck_result
                else "navigating"
            ),
            "contested": False,
            "unstuck_triggered": bool(unstuck_result is not None),
            "unstuck_info": unstuck_result,
            "selected_node": selected_node,
            "selected_index": selected_index,
            "no_alternative_node": bool(repath_requested and selected_node is None),
            "turn_evaluation": turn_eval,
            "corridor_eval": corridor_eval,
            "repath_requested": repath_requested,
            "prewalk": prewalk_info,
        }

    def execute_physical_unstick(
        self,
        maneuver_type: str = "reverse_backout",
        input_controller: Any = None,
        target_angle_deg: Optional[float] = None,
        current_angle_deg: Optional[float] = None,
        deadband_deg: float = DEFAULT_DEADBAND_DEG,
    ) -> Dict[str, Any]:
        """Fiziksel kurtulma manevrasını (jump, lateral strafe, reverse-backout) doğrudan çalıştırır."""
        return self.unstuck_maneuver.execute_unstuck_sequence(
            input_controller=input_controller,
            target_angle_deg=target_angle_deg,
            current_angle_deg=current_angle_deg,
            deadband_deg=deadband_deg,
            maneuver_type=maneuver_type,
        )

    def is_corridor_clear(
        self,
        frame: Optional[np.ndarray],
        live_detections: Optional[Sequence[Any]] = None,
    ) -> bool:
        """İleri koridorun engellerden ve canavarlardan tamamen temiz olup olmadığını döndürür."""
        eval_res = self.evaluate_corridor_path(frame, yolo_detections=live_detections)
        return bool(
            eval_res.get("steering_bias") == "clear"
            and not eval_res.get("obstacle_detected")
            and not eval_res.get("monster_detected")
            and eval_res.get("clear_terrain_found", False)
        )

    def get_corridor_steering_recommendation(
        self,
        frame: Optional[np.ndarray],
        live_detections: Optional[Sequence[Any]] = None,
    ) -> Dict[str, Any]:
        """İleri koridor için sapma açısı ve şerit puanlarını döndürür."""
        return self.evaluate_corridor_path(frame, yolo_detections=live_detections)


    def on_gather_success(
        self,
        node: Any,
        actual_x: float,
        actual_y: float,
        now: Optional[float] = None,
        minimap_dist_px: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        `[F Gather]` başarılı olduğunda:
        - Madde 27: Yaklaşma başarısızlık sayacını sıfırlar.
        - Madde 29: Düğümün `(x, y)` merkezini gerçek oyuncu konumuyla (EMA) iyileştirir.
        """
        # F-0243: başarı dalında mesafe bilinmiyorsa 0.0 uydurulmaz;
        # record_approach_attempt başarıda mesafeyi zaten kullanmaz.
        self.cliff_filter.record_approach_attempt(
            node=node,
            minimap_dist_px=minimap_dist_px,
            interaction_succeeded=True,
            now=now,
        )
        return self.drift_corrector.refine_node_coordinates_on_gather(
            node=node,
            actual_x=actual_x,
            actual_y=actual_y,
        )

    def on_approach_failed(
        self,
        node: Any,
        minimap_dist_px: float,
        now: Optional[float] = None,
    ) -> bool:
        """
        Düğüme yaklaşma denemesi etkileşim olmadan başarısız olduğunda çağrılır (Madde 27).
        `minimap_dist_px < 25px` iken 2. kez başarısız olursa `unreachable_cliff = True` işaretler.
        """
        return self.cliff_filter.record_approach_attempt(
            node=node,
            minimap_dist_px=minimap_dist_px,
            interaction_succeeded=False,
            now=now,
        )

    def _advance_to_next_node(
        self,
        nodes: Sequence[Any],
        current_seq_pos: int,
        player_pos: Optional[Point2D] = None,
        is_flying: bool = False,
        player_altitude_band: Optional[str] = None,
        now: Optional[float] = None,
        player_z: Optional[float] = None,
    ) -> Tuple[Optional[Any], int]:
        """Rakip oyuncu veya uçurum durumunda sıradaki uygun düğüme ilerler."""
        if self.route_planner is not None and hasattr(
            self.route_planner, "select_next_sequential_node"
        ):
            return self.route_planner.select_next_sequential_node(
                nodes=nodes,
                current_seq_pos=current_seq_pos,
                player_pos=player_pos,
                now=now,
                stay_on_partial_harvest=False,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
            )
        n = len(nodes)
        if n == 0:
            return None, -1
        if n == 1:
            # F-0245: tek düğümlü rotada "sıradaki" aynı düğümdür; contested /
            # uçurum atlaması aynı düğüme dönmemeli.
            return None, -1
        # F-0244: yedek yol uçurum filtresini ve contested kilidini yok
        # saymamalı — sıradaki uygun düğüm bulunana kadar tara.
        for step in range(1, n):
            idx = (int(current_seq_pos) + step) % n
            cand = nodes[idx]
            if self.cliff_filter.is_node_excluded_by_cliff(
                cand,
                is_flying=is_flying,
                player_altitude_band=player_altitude_band,
                player_z=player_z,
            ):
                continue
            if self.contested_detector.is_node_currently_contested_flag(cand, now=now):
                continue
            return cand, idx
        return None, -1

    def get_corridor_navigator_health_summary(self) -> dict:
        ready, checks = _corridor_navigator_self_check()
        return {"module": "corridor_navigator", "ready": ready, "checks": checks}

    def get_corridor_status_summary(self) -> dict:
        ready, checks = _corridor_navigator_self_check()
        return {"corridor_ready": ready, "metrics": checks}

    def get_corridor_navigator_ready_flag(self) -> dict:
        ready, checks = _corridor_navigator_self_check()
        return {"module": "corridor_navigator", "ready": ready, "checks": checks}

    def get_corridor_navigator_version_info(self) -> dict:
        ready, checks = _corridor_navigator_self_check()
        return {"module": "corridor_navigator", "version": __version__, "ready": ready, "checks": checks}

    def get_corridor_navigator_diagnostic_snapshot(self) -> dict:
        ready, checks = _corridor_navigator_self_check()
        return {
            "module": "corridor_navigator",
            "status": "ok" if ready else "degraded",
            "ready": ready,
            "checks": checks,
        }

def get_corridor_navigator_capabilities() -> list:
    return [
        "corridor_navigator",
        "corridor_steering",
        "heading_alignment",
        "turn_damping",
        "diagnostic_snapshot",
        "obstacle_avoidance",
        "monster_avoidance",
        "ground_path_steering",
        "optical_flow_stuck_detection",
        "physical_unstick_recovery",
        "reverse_backout",
    ]


CorridorStuckDetector = PhaseCorrelateUnstuckManeuver


# H-sweep (dilim 12 K-1): üretim bağlantı bayrağı. Varsayılan False'dur;
# yalnızca install_corridor_navigator() True yapar. "ready" bayrakları
# öz-deneyi anlatır, üretime bağlılığı anlatmaz — ikisi karıştırılmaz.
_PRODUCTION_WIRED: bool = False


def install_corridor_navigator(planner: Any = None, **kwargs: Any) -> "CorridorNavigator":
    """Üretim kurulumu (state_machine / main tarafı çağırır — C grubu işi).

    `CorridorNavigator` kurar, verilirse `SequentialRoutePlanner`'a bağlar
    (`attach_to_sequential_planner`) ve `_PRODUCTION_WIRED` bayrağını
    kaldırır. Çağrılmadığı sürece readiness "production_wired: False"
    döner; sınıfın kurulabiliyor olması üretime bağlı olduğu anlamına gelmez.
    """
    global _PRODUCTION_WIRED
    nav = CorridorNavigator(**kwargs)
    if planner is not None:
        nav.attach_to_sequential_planner(planner)
    _PRODUCTION_WIRED = True
    return nav


def _corridor_navigator_self_check() -> Tuple[bool, Dict[str, Any]]:
    """Alt bileşenleri ve kritik yöntemleri gerçekten yoklayarak hazır durumu ölçer (F-0246)."""
    try:
        nav = CorridorNavigator()
        checks: Dict[str, Any] = {
            "unstuck_maneuver": nav.unstuck_maneuver is not None,
            "cliff_filter": nav.cliff_filter is not None,
            "prewalk_scheduler": nav.prewalk_scheduler is not None,
            "drift_corrector": nav.drift_corrector is not None,
            "contested_detector": nav.contested_detector is not None,
            "turn_damper": nav.turn_damper is not None,
            "methods": all(
                callable(getattr(nav, name, None))
                for name in (
                    "step_navigation",
                    "evaluate_corridor_path",
                    "steer_step",
                    "execute_physical_unstick",
                    "on_gather_success",
                    "calculate_turn",
                )
            ),
            "capabilities": len(get_corridor_navigator_capabilities()) > 0,
        }
        return bool(all(checks.values())), checks
    except Exception as exc:
        return False, {"error": str(exc)}


def validate_corridor_navigator_contract() -> bool:
    # F-0246: yetenek listesinin boş olmaması yetmez; navigatör gerçekten
    # kurulabilmeli ve tüm alt bileşenleri + yöntemleri mevcut olmalı.
    ready, _checks = _corridor_navigator_self_check()
    return bool(ready and get_corridor_navigator_capabilities())


def get_corridor_navigator_baseline_metrics() -> dict:
    return {"module": "corridor_navigator", "contract_valid": validate_corridor_navigator_contract(), "capability_count": len(get_corridor_navigator_capabilities())}


def get_corridor_navigator_readiness_status() -> dict:
    # H-sweep (dilim 12 K-1): "ready" yalnızca modül öz-deneyidir (sınıf
    # kurulabiliyor + alt bileşenler mevcut). Üretime bağlı olup olmadığı
    # ayrı bayraktır — ölü modül "hazır" görünmesin diye production_wired
    # yalnızca install_corridor_navigator() çağrılınca True olur.
    return {
        "module": "corridor_navigator",
        "ready": validate_corridor_navigator_contract(),
        "production_wired": bool(_PRODUCTION_WIRED),
        "metrics": get_corridor_navigator_baseline_metrics(),
    }


def get_corridor_navigator_audit_record() -> dict:
    return {
        "module": "corridor_navigator",
        "audit_passed": validate_corridor_navigator_contract(),
        "readiness": get_corridor_navigator_readiness_status(),
    }


def get_corridor_navigator_operational_digest() -> dict:
    return {
        "module": "corridor_navigator",
        "operational": validate_corridor_navigator_contract(),
        "audit": get_corridor_navigator_audit_record(),
    }


def get_corridor_navigator_telemetry_payload() -> dict:
    return {
        "module": "corridor_navigator",
        "payload": get_corridor_navigator_operational_digest(),
    }


def get_corridor_navigator_integrity_seal() -> dict:
    return {
        "module": "corridor_navigator",
        "sealed": validate_corridor_navigator_contract(),
        "telemetry": get_corridor_navigator_telemetry_payload(),
    }


def get_corridor_navigator_compliance_certificate() -> dict:
    return {
        "module": "corridor_navigator",
        "certified": validate_corridor_navigator_contract(),
        "seal": get_corridor_navigator_integrity_seal(),
    }


def get_corridor_navigator_readiness_probe() -> dict:
    return {
        "module": "corridor_navigator",
        "ready": validate_corridor_navigator_contract(),
        "certificate": get_corridor_navigator_compliance_certificate(),
    }


def get_corridor_navigator_audit_summary() -> dict:
    return {
        "module": "corridor_navigator",
        "audit_passed": validate_corridor_navigator_contract(),
        "probe": get_corridor_navigator_readiness_probe(),
    }

