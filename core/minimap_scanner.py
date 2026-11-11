# -*- coding: utf-8 -*-
"""
Gerçek Zamanlı Minimap Materyal (3 Yapraklı Filiz) ve Radar Tarayıcı Motoru.

Oyun ekranının sağ üst köşesindeki minimap bölgesini (`norm_rect = (0.855, 0.065, 0.137, 0.190)`,
yani `x: 0.855..0.992, y: 0.065..0.255`) veya doğrudan kırpılmış minimap yakın çekimlerini
analiz ederek:
1. Oyuncu yön açısını (`detect_player_heading`: 0°-360°, Kuzey=0°, Doğu=90°),
2. Yalnızca gerçek toplanabilir materyal ikonu olan **beyaz 3 yapraklı filiz (`sprout_material`)**
   düğümlerini (`gatherable_nodes`),
3. Küçük düz yuvarlak beyaz canavar noktalarını (`monster_dots`, `•` - materyal olarak REDDEDİLİR),
4. Beyaz tüy (`minimap_feather_odyle.png`), mavi kanat ve yeşil yığın rozetlerini (`rejected_icons` -
   materyal olarak REDDEDİLİR, toplanmaz),
5. Kırmızı düşman noktalarını (`hostile_dots`),
6. Minimap solundaki `Minimap Filter Settings` açılır penceresinin durumunu (`detect_minimap_filter_state`),
7. Ardışık kareler arasındaki yürüyüş ilerlemesini (`detect_movement_progress`)
yüksek doğrulukla tespit eder.
"""

from __future__ import annotations

import copy
import itertools
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

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


# config/calibrated_rois.json disk okuma önbelleği: mutlak dosya yolu ->
# (mtime, parse edilmiş JSON dict). Her MinimapScanner() yapımı dosyayı
# diskten yeniden okuyordu (açılışta 8 kez ölçüldü); dosya değişmediyse
# önbellekten dönülür. Veri instance başına deepcopy ile verilir, böylece
# bir instance'ın dict'i diğerininkini etkilemez. Dosya değişirse (mtime
# farklı) diskten yeniden okunur ve önbellek tazelenir.
_CALIBRATED_ROIS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


def _get_sprout_monster_discriminator():
    """`MinimapSproutMonsterDiscriminator` örneğini tembel yükle.

    Modül düzeyinde import döngüsünden kaçınmak için ilk kullanımda yüklenir.
    Yüklenemezse `None` döner; çağıran o durumda diskriminatörsüz
    (yalnızca NCC + kromatik doğrulamalı) yola düşer ve bunu bayraklar.
    """
    try:
        from dataset_factory.minimap_sprout_monster_discriminator import (
            MinimapSproutMonsterDiscriminator,
        )

        return MinimapSproutMonsterDiscriminator()
    except Exception as exc:  # pragma: no cover - import ortamı hatası
        logger.debug("Sprout/monster discriminator yüklenemedi: %s", exc)
        return None


class MinimapScanner:
    """
    Minimap 3-Yapraklı Filiz (`sprout_material`) Tarayıcı, Canavar/Tüy Filtreleyici,
    Oyuncu Yönü (Heading) ve Yürüyüş İlerleme Analizörü.
    """

    # Sağ üst minimap kutusu: x in [0.855..0.992], y in [0.065..0.255]
    DEFAULT_NORM_RECT: Tuple[float, float, float, float] = (0.855, 0.065, 0.137, 0.190)
    PLAYER_CENTER_REL: Tuple[float, float] = (0.502, 0.545)
    PLAYER_CLEARANCE_NORM: float = 0.075

    # YOLO ve Algı Sınıfları (Filiz vs Düşman Noktası Ayrımı)
    CLASS_ID_SPROUT: int = 9
    CLASS_ID_MONSTER_DOT: int = 11
    CLASS_NAME_SPROUT: str = "minimap_sprout"
    CLASS_NAME_HOSTILE_DOT: str = "minimap_monster_negative"
    CLASS_NAME_MONSTER_DOT: str = "monster_dot"
    CLASS_NAME_MINIMAP_MONSTER: str = "minimap_monster_dot"

    # Sıkı Ayrım Kriterleri ve Eşik Değerleri (Strict Separation Criteria)
    SPROUT_NCC_MIN_THRESHOLD: float = 0.88
    # Tam-kare NCC ölçekleri: 5 yerine 3 (0.95/1.00/1.05) — 25 yerine en
    # fazla 20 tam-kare _masked_ncc (2 kanonik + 3 dış şablon x 4 ölçek:
    # çarpanlardan gelen 3 boyut + kanonik şablonun kendi 16px boyutu).
    SPROUT_NCC_SCALES: Tuple[float, ...] = (0.95, 1.00, 1.05)
    HOSTILE_RED_CIRCULARITY_MIN: float = 0.65
    CENTER_EXCLUSION_RADIUS_PX: float = 14.0

    # 3D Dünya Odyle Küresi (P2-10 / v19) — kompakt blob gürültü kapıları.
    # Bunlar "orb imzası" değil, yalnızca benek/çizgi reddi için gevşek
    # kapılardır; asıl kanıt `classify_3d_material_patch` "Odyle" kararı +
    # `ODYLE_ORB_MIN_RATIO`'dur (aşağıda). Değerler bu dosyadaki mevcut
    # kompakt-blob disiplininden alınır (hostile: alan 3..95, dairesellik
    # >= 0.65; monster: alan 7..85, dairesellik >= 0.62, solidity >= 0.80,
    # en-boy 0.72..1.38) — orb parlayan bir leke olduğu için kapılar
    # bilerek daha gevşektir.
    # `ODYLE_ORB_MIN_RATIO`: config/materials_db.py `map_descriptors_to_taxonomy`
    # satır ~1787'deki Odyle kanıt kuralı (`odyle_orb >= 0.045`).
    ODYLE_ORB_MIN_RATIO: float = 0.045
    ODYLE_ORB_MIN_AREA_REF_PX: float = 40.0  # 2000x1125 referans karede
    ODYLE_ORB_MIN_AREA_ABS_PX: float = 25.0  # küçük karelerde mutlak taban
    ODYLE_ORB_MAX_AREA_FRAC: float = 0.02  # kare alanının %2'si (arazi yıkaması reddi)
    ODYLE_ORB_MAX_BBOX_FRAC: float = 0.25  # kenar başına kare boyutunun %25'i
    ODYLE_ORB_CIRCULARITY_MIN: float = 0.50
    ODYLE_ORB_SOLIDITY_MIN: float = 0.70
    ODYLE_ORB_REF_AREA_PX: float = 2000.0 * 1125.0  # alan ölçek referansı

    # 3D Dünya ve Minimap Sınıflarının Birbirine Karışmasını Engelleyen Taksonomi
    MINIMAP_CLASS_IDS: frozenset[int] = frozenset({9, 11})
    WORLD_3D_CLASS_IDS: frozenset[int] = frozenset(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}
    )

    # Kanonik Bölge ve Alt Bölge Eşleme Haritası — TEK KAYNAK:
    # `core/route_manager.py:SUBREGION_TO_CANONICAL` (orada da
    # `SUBREGION_CANONICAL_MAP` adıyla ihraç edilir). Bu sınıftaki sözlük
    # yalnızca route_manager import edilemezse kullanılan yedektir ve
    # onunla aynı değerleri taşımalıdır (H-sweep dilim 13 K1: scanner
    # "pilgrim's ridge" -> Altgard diyordu, tek kaynak Verteron diyor).
    # Yeni alt bölge eklenecekse SADECE route_manager'a eklenir.
    SUBREGION_CANONICAL_MAP: Dict[str, str] = {
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
        "basfelt ruins": "Altgard",
        "black plains": "Altgard",
        "heart of the dragon": "Altgard",
        "pilgrim's ridge": "Verteron",
        "altgard": "Altgard",
    }

    @classmethod
    def _canonical_region_for(cls, subregion_name: Optional[str]) -> Optional[str]:
        """Alt bölge adından kanonik bölge; bilinmeyende None döner.

        Önce route_manager'daki tek kaynağa bakılır (Title-Case anahtarlar
        küçük harfe çevrilerek); import başarısız olursa yukarıdaki
        yedek sözlük kullanılır. Asla varsayılan bölge uydurulmaz.
        """
        if not subregion_name:
            return None
        key = str(subregion_name).strip().lower()
        try:
            from core.route_manager import SUBREGION_TO_CANONICAL

            lowered = {str(k).lower(): v for k, v in SUBREGION_TO_CANONICAL.items()}
            if key in lowered:
                return str(lowered[key])
        except Exception:
            pass
        return cls.SUBREGION_CANONICAL_MAP.get(key)

    # 332x257 referans minimap ölçeğinde kanonik 16x16 beyaz 3-yapraklı filiz (sprout) şablonu
    _CANONICAL_SPROUT_TPL: np.ndarray = np.array(
        [
            [81, 82, 84, 85, 77, 84, 62, 122, 92, 69, 84, 81, 88, 87, 83, 78],
            [83, 90, 88, 79, 87, 66, 90, 230, 159, 53, 83, 81, 80, 81, 81, 79],
            [83, 83, 82, 79, 83, 34, 157, 232, 200, 80, 73, 93, 85, 85, 85, 84],
            [84, 76, 80, 90, 59, 85, 240, 210, 227, 150, 56, 86, 92, 89, 87, 87],
            [87, 86, 86, 86, 44, 184, 245, 214, 227, 202, 83, 73, 90, 87, 85, 84],
            [85, 94, 82, 58, 93, 236, 235, 185, 214, 238, 145, 38, 86, 89, 90, 87],
            [81, 85, 73, 54, 181, 255, 255, 165, 216, 255, 199, 35, 73, 87, 95, 90],
            [84, 73, 73, 82, 229, 255, 232, 155, 195, 255, 248, 121, 50, 74, 90, 86],
            [87, 81, 46, 175, 249, 254, 228, 143, 202, 245, 255, 220, 53, 77, 88, 84],
            [81, 70, 57, 166, 245, 254, 244, 114, 190, 255, 246, 214, 65, 40, 86, 93],
            [79, 53, 38, 94, 185, 245, 234, 116, 173, 255, 205, 93, 29, 69, 71, 79],
            [66, 102, 146, 160, 122, 183, 234, 89, 179, 210, 131, 182, 139, 97, 78, 48],
            [220, 253, 246, 254, 227, 96, 174, 72, 140, 136, 214, 246, 254, 246, 206, 102],
            [67, 152, 251, 252, 241, 218, 80, 45, 87, 200, 243, 253, 237, 175, 84, 68],
            [67, 63, 131, 182, 141, 78, 40, 44, 43, 78, 124, 196, 142, 69, 78, 85],
            [100, 76, 50, 60, 78, 84, 100, 93, 91, 85, 61, 47, 64, 78, 85, 88],
        ],
        dtype=np.uint8,
    )

    # Alternatif kanonik 16x16 filiz şablonu (farklı ışık/zemin koşulları için)
    _CANONICAL_SPROUT_TPL_ALT: np.ndarray = np.array(
        [
            [120, 120, 123, 131, 122, 101, 105, 151, 100, 106, 128, 129, 140, 123, 129, 134],
            [121, 123, 124, 132, 123, 85, 154, 211, 134, 75, 129, 127, 129, 135, 122, 131],
            [122, 126, 123, 140, 99, 101, 189, 206, 177, 76, 103, 134, 116, 136, 129, 137],
            [123, 127, 128, 131, 58, 146, 207, 184, 214, 126, 54, 132, 122, 132, 137, 137],
            [122, 125, 129, 93, 75, 192, 206, 174, 216, 188, 63, 102, 128, 130, 135, 130],
            [120, 123, 120, 57, 137, 222, 192, 164, 188, 226, 126, 69, 120, 129, 131, 129],
            [112, 120, 99, 80, 198, 210, 188, 138, 197, 218, 190, 78, 100, 132, 118, 134],
            [116, 114, 47, 148, 211, 220, 180, 139, 196, 213, 211, 130, 77, 121, 115, 113],
            [105, 84, 69, 190, 209, 199, 175, 107, 197, 194, 205, 178, 57, 84, 125, 116],
            [107, 86, 60, 108, 209, 204, 176, 103, 180, 214, 203, 114, 52, 98, 115, 108],
            [80, 62, 64, 83, 133, 192, 167, 91, 176, 188, 124, 69, 70, 83, 84, 80],
            [112, 137, 160, 165, 119, 139, 147, 71, 160, 139, 136, 170, 162, 117, 101, 43],
            [140, 184, 199, 189, 192, 120, 104, 71, 104, 143, 189, 192, 192, 173, 129, 68],
            [70, 110, 164, 176, 147, 116, 46, 36, 74, 124, 164, 180, 153, 92, 81, 102],
            [85, 72, 75, 82, 79, 76, 83, 87, 79, 70, 80, 80, 82, 89, 98, 104],
            [100, 92, 83, 91, 92, 92, 100, 104, 100, 98, 95, 95, 96, 99, 103, 105],
        ],
        dtype=np.uint8,
    )

    _CANONICAL_SPROUT_MASK: np.ndarray = np.array(
        [
            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0],
            [1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1],
            [1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    def __init__(
        self,
        norm_rect: Tuple[float, float, float, float] = DEFAULT_NORM_RECT,
        movement_shift_threshold: float = 0.22,
        # P1-13: varsayılan 2.2 — önceden gömülü 2.2 sabitiydi; parametre
        # hiç okunmuyordu. Varsayılan eski sabite eşit, davranış değişmez.
        movement_diff_threshold: float = 2.2,
        templates_dir: Optional[str] = None,
        config_dir: Optional[str] = None,
    ) -> None:
        self.norm_rect: Tuple[float, float, float, float] = (
            float(norm_rect[0]),
            float(norm_rect[1]),
            float(norm_rect[2]),
            float(norm_rect[3]),
        )
        self.player_center_rel: Tuple[float, float] = (
            float(self.PLAYER_CENTER_REL[0]),
            float(self.PLAYER_CENTER_REL[1]),
        )
        self.movement_shift_threshold: float = float(movement_shift_threshold)
        self.movement_diff_threshold: float = float(movement_diff_threshold)

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.templates_dir: str = (
            templates_dir if templates_dir is not None else os.path.join(base_dir, "templates")
        )
        self.config_dir: str = (
            config_dir if config_dir is not None else os.path.join(base_dir, "config")
        )

        self.calibrated_rois: Dict[str, Any] = {}
        self._calibrated_rois_mtime: float = 0.0
        # norm_rect'in alındığı minimap girdisinin kayıtlı çözünürlüğü
        # (w, h) — None ise çözünürlük kaydı yok demektir.
        self._calibrated_minimap_resolution: Optional[Tuple[int, int]] = None

        self._external_sprout_templates: List[Tuple[np.ndarray, np.ndarray]] = []
        self._external_arrow_templates: List[Tuple[np.ndarray, np.ndarray]] = []
        self._player_arrow_template: Optional[np.ndarray] = None
        self._player_arrow_mask: Optional[np.ndarray] = None
        # Yön (heading) şablon önbelleği: döndürülmüş şablon/maske çiftleri
        # yalnızca (_player_arrow_template, _player_arrow_mask) ikilisine
        # bağlıdır; şablon değişirse kimlik kontrolüyle sıfırlanır.
        self._heading_cache_tpl: Optional[np.ndarray] = None
        self._heading_cache_mask: Optional[np.ndarray] = None
        self._heading_base_heading: Optional[float] = None
        self._heading_rot_cache: Dict[float, Tuple[np.ndarray, np.ndarray, int]] = {}

        self._load_calibrated_rois()
        self._load_optional_templates()

    def _load_calibrated_rois(self) -> None:
        """Diskte mevcutsa config/calibrated_rois.json dosyasından kalibre edilmiş ROI sınırlarını yükler."""
        self.calibrated_rois = {}
        self._calibrated_minimap_resolution = None
        calibrated_file = os.path.join(self.config_dir, "calibrated_rois.json")
        if not os.path.isfile(calibrated_file):
            return
        try:
            mtime = os.path.getmtime(calibrated_file)
            self._calibrated_rois_mtime = mtime
            # Modül düzeyi önbellek: dosya değişmediyse diskten yeniden
            # okuma, önbellekten dön (her yapımda 1 disk okuma + json
            # parse ödeniyordu). Anahtar (mutlak yol, mtime).
            cache_key = os.path.abspath(calibrated_file)
            cached = _CALIBRATED_ROIS_CACHE.get(cache_key)
            if cached is not None and cached[0] == mtime:
                data = copy.deepcopy(cached[1])
            else:
                with open(calibrated_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    _CALIBRATED_ROIS_CACHE[cache_key] = (
                        mtime,
                        copy.deepcopy(data),
                    )
            if isinstance(data, dict):
                self.calibrated_rois = data
                # Global norm_rect kontrolü. Gerçek dosya şeması
                # (config/calibrated_rois.json) üst anahtar olarak
                # "minimap_norm" taşımaz; minimap rect'i "rois"/"minimap"
                # (veya üst düzey "minimap") girdisinde, normalized_roi
                # [x, y, w, h] olarak durur — o da okunur.
                norm = data.get("minimap_norm") or data.get("minimap_norm_rect")
                minimap_entry: Optional[Dict[str, Any]] = None
                if not norm:
                    minimap_entry = self._find_calibrated_minimap_entry(data)
                    if isinstance(minimap_entry, dict):
                        norm = minimap_entry.get("normalized_roi") or minimap_entry.get("norm_rect")
                if norm and isinstance(norm, (list, tuple)) and len(norm) == 4:
                    self.norm_rect = (
                        float(norm[0]),
                        float(norm[1]),
                        float(norm[2]),
                        float(norm[3]),
                    )
                    # Bu norm_rect hangi çözünürlük için kaydedildiyse
                    # yalnızca o çözünürlükte geçerlidir: çözünürlük
                    # kaydı olmayan (örn. 1920x1080 tohumu) bir girdinin
                    # yakın çekime uygulanması regresyona yol açar
                    # (get_minimap_pixel_bounds adım 2).
                    self._calibrated_minimap_resolution = self._parse_entry_resolution(
                        minimap_entry
                    )
                # Global player_center_rel kontrolü (yalnızca dosya
                # açıkça player_center_rel verirse uygulanır).
                p_rel = data.get("player_center_rel")
                if p_rel and isinstance(p_rel, (list, tuple)) and len(p_rel) == 2:
                    self.player_center_rel = (float(p_rel[0]), float(p_rel[1]))
                logger.info("Kalibre edilmiş ROI'ler yüklendi: %s", calibrated_file)
        except Exception as exc:
            logger.warning("Kalibre edilmiş ROI dosyası yüklenemedi (%s): %s", calibrated_file, exc)

    @staticmethod
    def _find_calibrated_minimap_entry(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Kalibrasyon JSON'undan minimap rect girdisini bulur."""
        rois = data.get("rois") if isinstance(data.get("rois"), dict) else {}
        for container in (rois, data):
            for key in ("minimap", "minimap_roi", "minimap_bounds", "minimap_rect"):
                entry = container.get(key)
                if isinstance(entry, dict):
                    return entry
        return None

    @staticmethod
    def _parse_entry_resolution(entry: Any) -> Optional[Tuple[int, int]]:
        """Kalibre girdisinin kayıtlı ``resolution`` alanını (w, h) olarak çözer.

        Kayıt yoksa / çözümlenemezse None döner; None, eski davranışın
        (çözünürlük kontrolsüz uygulama) korunması anlamına gelir.
        """
        if not isinstance(entry, dict):
            return None
        res = entry.get("resolution")
        if res is None and isinstance(entry.get("minimap"), dict):
            res = entry["minimap"].get("resolution")
        if not isinstance(res, (list, tuple)) or len(res) != 2:
            return None
        try:
            return (int(res[0]), int(res[1]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _calibrated_entry_matches_resolution(entry: Any, w: int, h: int) -> bool:
        """Jenerik kalibre girdisi bu kareye uygulanabilir mi?

        Girdi çözünürlük kaydı taşıyor ve kare o çözünürlükte değilse
        False döner (örn. 1920x1080 tohumu 365x309 yakın çekime
        uygulanmaz). Kayıt yoksa True döner — eski davranış korunur.
        """
        res = MinimapScanner._parse_entry_resolution(entry)
        if res is None:
            return True
        return res == (w, h)

    def _check_and_reload_calibrated_rois(self) -> None:
        """Gerektiğinde kalibre edilmiş ROI dosyasını dinamik olarak yeniden yükler."""
        calibrated_file = os.path.join(self.config_dir, "calibrated_rois.json")
        if os.path.isfile(calibrated_file):
            try:
                mtime = os.path.getmtime(calibrated_file)
                if mtime > getattr(self, "_calibrated_rois_mtime", 0.0):
                    self._load_calibrated_rois()
            except Exception:
                pass

    def _parse_roi_bounds(self, entry: Any, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
        """Verilen JSON girdisinden (x1, y1, x2, y2) piksel sınırlarını çözer."""
        try:
            if isinstance(entry, dict):
                if "normalized_roi" in entry:
                    return self._parse_roi_bounds(entry["normalized_roi"], w, h)
                if "x1" in entry and "y1" in entry and "x2" in entry and "y2" in entry:
                    x1 = int(round(float(entry["x1"])))
                    y1 = int(round(float(entry["y1"])))
                    x2 = int(round(float(entry["x2"])))
                    y2 = int(round(float(entry["y2"])))
                    return max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
                if "x" in entry and "y" in entry and "w" in entry and "h" in entry:
                    x1 = int(round(float(entry["x"])))
                    y1 = int(round(float(entry["y"])))
                    x2 = x1 + int(round(float(entry["w"])))
                    y2 = y1 + int(round(float(entry["h"])))
                    return max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
                if "left" in entry and "top" in entry and "right" in entry and "bottom" in entry:
                    x1 = int(round(float(entry["left"])))
                    y1 = int(round(float(entry["top"])))
                    x2 = int(round(float(entry["right"])))
                    y2 = int(round(float(entry["bottom"])))
                    return max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
                if "minimap" in entry:
                    return self._parse_roi_bounds(entry["minimap"], w, h)
                if "bbox" in entry:
                    return self._parse_roi_bounds(entry["bbox"], w, h)
            elif isinstance(entry, (list, tuple)) and len(entry) == 4:
                vals = [float(v) for v in entry]
                if all(0.0 <= v <= 1.0 for v in vals):
                    # S105 (v19/P3-8): normalize 4'lü her zaman
                    # (x, y, w, h)'dir — sözleşme config/
                    # calibrated_rois.json'daki 11 ROI'nin tamamında bu
                    # (norm_x/norm_y/norm_w/norm_h anahtarları da aynı).
                    # Eski kod 3./4. değer 1./2.'den büyükse köşe
                    # (x1,y1,x2,y2) sanıyordu; w>x ve h>y olan bir
                    # (x,y,w,h) ROI yanlış okunurdu. Köşe biçimi
                    # yalnızca dict dalındaki açık x1/y1/x2/y2
                    # anahtarlarıyla verilir.
                    nx, ny, nw, nh = vals
                    x1 = int(round(w * nx))
                    y1 = int(round(h * ny))
                    x2 = int(round(w * (nx + nw)))
                    y2 = int(round(h * (ny + nh)))
                    return max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
                else:
                    x1, y1, x2, y2 = int(round(vals[0])), int(round(vals[1])), int(round(vals[2])), int(round(vals[3]))
                    return max(0, min(x1, w - 1)), max(0, min(y1, h - 1)), max(x1 + 1, min(x2, w)), max(y1 + 1, min(y2, h))
        except Exception as exc:
            logger.debug("_parse_roi_bounds hatası: %s", exc)
        return None

    def _load_optional_templates(self) -> None:
        """
        Diskte mevcutsa filiz (sprout) ve oyuncu oku (player arrow) şablonlarını
        Unicode-güvenli olarak yükler.
        Desteklenen şablonlar:
        - Filiz: minimap_sprout_node.png, minimap_gatherable_sprout_icon.png, minimap_sprout_template.png
        - Oyuncu oku: player_arrow_template.png, player_arrow.png
        """
        self._external_sprout_templates.clear()
        self._external_arrow_templates = []
        self._player_arrow_template = None
        self._player_arrow_mask = None

        # 1. Filiz Şablonları (Sprout Templates)
        sprout_names = (
            "minimap_sprout_node.png",
            "minimap_gatherable_sprout_icon.png",
            "minimap_sprout_template.png",
        )
        for fname in sprout_names:
            fpath = os.path.join(self.templates_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                raw_bytes = np.fromfile(fpath, dtype=np.uint8)
                if raw_bytes.size == 0:
                    continue
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                th, tw = gray.shape[:2]
                if 10 <= tw <= 36 and 10 <= th <= 36:
                    med = float(np.median(gray))
                    dev = (
                        ((gray < max(68.0, med - 15.0)) | (gray > min(180.0, med + 25.0))).astype(
                            np.uint8
                        )
                        * 255
                    )
                    # Şekilsiz eşik blob'u yerine ikon şekli: sapma
                    # maskesindeki en büyük bağlı bileşen + kapatma.
                    num, labels, stats, _ = cv2.connectedComponentsWithStats(dev, 8)
                    if num > 1:
                        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
                        mask = (labels == largest).astype(np.uint8)
                        mask = cv2.morphologyEx(
                            mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
                        )
                    else:
                        mask = (dev > 0).astype(np.uint8)
                    if int(mask.sum()) >= 20:
                        self._external_sprout_templates.append((gray, mask))
            except Exception as exc:
                logger.debug("Filiz şablon yükleme atlandı (%s): %s", fname, exc)

        # 2. Oyuncu Oku Şablonları (Player Arrow Templates)
        arrow_names = (
            "player_arrow_template.png",
            "player_arrow.png",
        )
        for fname in arrow_names:
            fpath = os.path.join(self.templates_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                raw_bytes = np.fromfile(fpath, dtype=np.uint8)
                if raw_bytes.size == 0:
                    continue
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                if img is None or img.size == 0:
                    continue
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                th, tw = gray.shape[:2]
                if 8 <= tw <= 48 and 8 <= th <= 48:
                    raw_mask = (gray >= 180).astype(np.uint8)
                    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, 8)
                    arrow_mask = np.zeros_like(raw_mask)
                    for idx in range(1, int(count)):
                        bx, by, bw, bh, area = (int(v) for v in stats[idx])
                        if area < 6 or area > 120:
                            continue
                        if bx <= 0 or by <= 0 or bx + bw >= tw - 1 or by + bh >= th - 1:
                            continue
                        arrow_mask[labels == idx] = 255

                    if int(np.sum(arrow_mask > 0)) < 6:
                        arrow_mask = raw_mask

                    self._external_arrow_templates.append((gray, arrow_mask))
                    if self._player_arrow_template is None:
                        self._player_arrow_template = gray
                        self._player_arrow_mask = arrow_mask
            except Exception as exc:
                logger.debug("Oyuncu oku şablon yükleme atlandı (%s): %s", fname, exc)

    # ------------------------------------------------------------------
    # Geometrik ve Görüntü Kesit Yardımcıları
    # ------------------------------------------------------------------
    def get_minimap_pixel_bounds(self, frame_shape: Tuple[int, ...]) -> Tuple[int, int, int, int]:
        """
        Ekran veya yakın çekim boyutuna göre minimap (x1, y1, x2, y2) piksel sınırlarını döndürür.
        1. Varsa config/calibrated_rois.json dosyasından kalibre edilmiş sınırları okur.
        2. Yakın çekimlerde (w < 500 veya 500 <= w < 1000) uygun kutu koordinatlarını döndürür.
        3. Tam ekranda görünüm alanı (viewport) boyutlarına göre dinamik olarak hesaplar.
        """
        h, w = int(frame_shape[0]), int(frame_shape[1])
        res_key = f"{w}x{h}"
        self._last_viewport_key = res_key

        # 1. Kalibre edilmiş ROI kontrolü
        self._check_and_reload_calibrated_rois()
        if self.calibrated_rois:
            calib_entry = None
            resolution_keyed = False
            if res_key in self.calibrated_rois:
                calib_entry = self.calibrated_rois[res_key]
                resolution_keyed = True
            elif f"{w}_{h}" in self.calibrated_rois:
                calib_entry = self.calibrated_rois[f"{w}_{h}"]
                resolution_keyed = True
            elif "minimap" in self.calibrated_rois:
                calib_entry = self.calibrated_rois["minimap"]
            elif "minimap_roi" in self.calibrated_rois:
                calib_entry = self.calibrated_rois["minimap_roi"]
            elif "minimap_bounds" in self.calibrated_rois:
                calib_entry = self.calibrated_rois["minimap_bounds"]
            elif isinstance(self.calibrated_rois.get("rois"), dict):
                calib_entry = self._find_calibrated_minimap_entry(self.calibrated_rois)

            # Jenerik ("minimap" vb.) girdi yalnızca kayıtlı çözünürlüğü
            # kareyle eşleşiyorsa uygulanır: 1920x1080 tohumunun 365x309
            # yakın çekime uygulanması filizleri sıfırlıyordu.
            # Çözünürlük anahtarlı ("1920x1080") girdiler anahtarın
            # kendisiyle eşleştiği için korumadan muaftır.
            if calib_entry is not None and (
                resolution_keyed
                or self._calibrated_entry_matches_resolution(calib_entry, w, h)
            ):
                parsed = self._parse_roi_bounds(calib_entry, w, h)
                if parsed is not None:
                    return parsed

        # 2. Yakın çekimler
        # norm_rect kalibre girdiden geldiyse ve o girdinin çözünürlüğü
        # bu kareyle eşleşmiyorsa yakın-çekim geometrisi uygulanır
        # (kalibre ROI yalnızca kendi çözünürlüğünde geçerlidir).
        calib_res = getattr(self, "_calibrated_minimap_resolution", None)
        calib_applies = calib_res is None or tuple(calib_res) == (w, h)
        if self.norm_rect == self.DEFAULT_NORM_RECT or not calib_applies:
            if w < 500 and h < 450:
                if 340 <= w <= 390 and 290 <= h <= 335:
                    return (
                        int(round(w * 0.019)),
                        int(round(h * 0.107)),
                        int(round(w * 0.929)),
                        int(round(h * 0.939)),
                    )
                return 0, 0, w, h

            if 500 <= w < 1000 and h < 500:
                return (
                    int(round(w * 0.498)),
                    int(round(h * 0.201)),
                    int(round(w * 0.961)),
                    int(round(h * 0.915)),
                )

            # 16:9 ref_h düzeltmesi KALDIRILDI: auto_cartographer gerçek
            # frame_h kullanır; iki modül aynı karede farklı bölgeye
            # bakıyordu (2000x981'de ~9px y kayması). Gerçek h kullanılır.
            nx, ny, nw, nh = self.norm_rect
            x1 = max(0, min(int(round(w * nx)), w - 1))
            y1 = max(0, min(int(round(h * ny)), h - 1))
            x2 = max(x1 + 1, min(int(round(w * (nx + nw))), w))
            y2 = max(y1 + 1, min(int(round(h * (ny + nh))), h))
            return x1, y1, x2, y2

        nx, ny, nw, nh = self.norm_rect
        x1 = max(0, min(int(round(w * nx)), w - 1))
        y1 = max(0, min(int(round(h * ny)), h - 1))
        x2 = max(x1 + 1, min(int(round(w * (nx + nw))), w))
        y2 = max(y1 + 1, min(int(round(h * (ny + nh))), h))
        return x1, y1, x2, y2

    def extract_minimap(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Kareden minimap alt görüntüsünü (BGR) ve ekran koordinat sınırlarını çıkarır."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Geçersiz veya boş görüntü karesi (frame).")
        if frame.ndim not in (2, 3):
            raise ValueError(f"Geçersiz kare boyutu: ndim={frame.ndim} (2 veya 3 beklenir).")
        if frame.ndim == 3 and frame.shape[2] not in (3, 4):
            raise ValueError(f"Geçersiz kanal sayısı: {frame.shape[2]} (3 veya 4 beklenir).")

        if frame.ndim == 2:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr_frame = frame

        x1, y1, x2, y2 = self.get_minimap_pixel_bounds(bgr_frame.shape)
        crop = bgr_frame[y1:y2, x1:x2]
        return crop, (x1, y1, x2, y2)

    def get_minimap_player_center(
        self,
        crop_shape: Tuple[int, ...],
        crop_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Minimap kesiti içerisindeki oyuncu okunun merkez (cx, cy) koordinatını döndürür.
        Görünüm alanı boyutlarına göre dinamik olarak hesaplar; varsa kalibre edilmiş merkez ve
        crop_bgr üzerinden parlak oyuncu okunun gerçek ağırlık merkezini / şablon konumunu bulur.
        """
        ch, cw = int(crop_shape[0]), int(crop_shape[1])
        res_key = f"{cw}x{ch}"

        # 1. Dinamik temel merkez koordinatları
        default_cx = float(cw) * float(self.player_center_rel[0])
        default_cy = float(ch) * float(self.player_center_rel[1])

        # Kalibrasyon dosyasından merkez kontrolü
        self._check_and_reload_calibrated_rois()
        if self.calibrated_rois:
            p_entry = None
            if res_key in self.calibrated_rois and isinstance(self.calibrated_rois[res_key], dict):
                p_entry = self.calibrated_rois[res_key].get("player_center") or self.calibrated_rois[res_key].get("player_center_rel")
            if p_entry is None and getattr(self, "_last_viewport_key", None) in self.calibrated_rois:
                v_entry = self.calibrated_rois[self._last_viewport_key]
                if isinstance(v_entry, dict):
                    p_entry = v_entry.get("player_center") or v_entry.get("player_center_rel")
            if p_entry is None:
                for _k, _v in self.calibrated_rois.items():
                    if isinstance(_v, dict):
                        _m_roi = _v.get("minimap") or _v.get("roi") or _v.get("bbox_px")
                        if isinstance(_m_roi, (list, tuple)) and len(_m_roi) == 4:
                            if int(_m_roi[2] - _m_roi[0]) == cw and int(_m_roi[3] - _m_roi[1]) == ch:
                                p_entry = _v.get("player_center") or _v.get("player_center_rel")
                                if p_entry is not None:
                                    break
            if p_entry is None:
                p_entry = self.calibrated_rois.get("player_center") or self.calibrated_rois.get("player_center_rel")

            if p_entry and isinstance(p_entry, (list, tuple)) and len(p_entry) == 2:
                v0, v1 = float(p_entry[0]), float(p_entry[1])
                if v0 <= 1.0 and v1 <= 1.0:
                    default_cx = float(cw) * v0
                    default_cy = float(ch) * v1
                else:
                    default_cx = v0
                    default_cy = v1

        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return default_cx, default_cy

        try:
            # Oyuncu oku arama penceresi (temel merkeze göre dinamik)
            half_w = max(6, int(round(cw * 0.055)))
            half_h = max(6, int(round(ch * 0.055)))
            rx1 = max(0, int(round(default_cx - half_w)))
            ry1 = max(0, int(round(default_cy - half_h)))
            rx2 = min(cw, int(round(default_cx + half_w + 1)))
            ry2 = min(ch, int(round(default_cy + half_h + 1)))

            roi = crop_bgr[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                return default_cx, default_cy

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

            # 2. Şablon eşleme ile hassas oyuncu merkezi arama
            if self._player_arrow_template is not None and self._player_arrow_mask is not None:
                tpl_g = self._player_arrow_template
                tpl_m = self._player_arrow_mask
                th, tw = tpl_g.shape[:2]
                if roi.shape[0] >= th and roi.shape[1] >= tw:
                    res = cv2.matchTemplate(gray, tpl_g, cv2.TM_CCOEFF_NORMED, mask=tpl_m)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if float(max_val) >= 0.55:
                        cx = float(rx1 + max_loc[0]) + float(tw) * 0.5
                        cy = float(ry1 + max_loc[1]) + float(th) * 0.5
                        return cx, cy

            # 3. Moment analizi ile parlak oyuncu oku / dairesi merkezi
            _, bw = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY)
            cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                _, bw = cv2.threshold(gray, 185, 255, cv2.THRESH_BINARY)
                cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if cnts:
                best_cnt = max(cnts, key=cv2.contourArea)
                best_area = float(cv2.contourArea(best_cnt))
                bx, by, bw, bh = cv2.boundingRect(best_cnt)
                # En büyük parlak kontur her zaman ok değildir (parlak
                # filiz/ikon da girebilir): ok-benzeri boyut/oran şartı
                # sağlanmazsa kalibre varsayılan merkeze düşülür.
                arrow_plausible = (
                    3.0 <= best_area <= 600.0
                    and 4 <= bw <= 48
                    and 4 <= bh <= 48
                    and 0.4 <= (bw / float(max(1, bh))) <= 2.5
                )
                if arrow_plausible:
                    m = cv2.moments(best_cnt)
                    if m["m00"] > 1e-5:
                        cx = float(rx1) + float(m["m10"] / m["m00"])
                        cy = float(ry1) + float(m["m01"] / m["m00"])
                        return cx, cy

        except Exception as exc:
            logger.debug("get_minimap_player_center dinamik iyileştirme hatası: %s", exc)

        return default_cx, default_cy

    @staticmethod
    def _vector_to_compass_deg(dx: float, dy: float) -> float:
        """
        Ekran koordinat farkını (dx sağa pozitif, dy aşağı pozitif)
        pusula açısına (0.0° - 360.0°, Kuzey=0°, Doğu=90°, Güney=180°, Batı=270°) dönüştürür.
        """
        angle = math.degrees(math.atan2(float(dx), -float(dy)))
        return round((angle + 360.0) % 360.0, 2)

    def _get_template_base_heading(self, tpl: np.ndarray, mask: np.ndarray) -> float:
        """Şablonun kendi doğal yön açısını (North=0°) hesaplar."""
        try:
            th, tw = tpl.shape[:2]
            scale = 4
            up_mask = cv2.resize(mask, (tw * scale, th * scale), interpolation=cv2.INTER_NEAREST)
            cnts, _ = cv2.findContours(up_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                return 0.0
            best_c = max(cnts, key=cv2.contourArea)
            peri = cv2.arcLength(best_c, True)
            approx = cv2.approxPolyDP(best_c, 0.04 * peri, True).reshape(-1, 2)

            if len(approx) >= 3:
                candidates = []
                for p1, p2, p3 in itertools.combinations(approx, 3):
                    d12 = float(np.linalg.norm(p1 - p2))
                    d23 = float(np.linalg.norm(p2 - p3))
                    d31 = float(np.linalg.norm(p3 - p1))
                    candidates.append((d12 + d23 + d31, (p1, p2, p3)))

                if candidates:
                    candidates.sort(key=lambda x: x[0], reverse=True)
                    p1, p2, p3 = candidates[0][1]
                    uh, uw = up_mask.shape[:2]
                    for apex, w1, w2 in [(p1, p2, p3), (p2, p3, p1), (p3, p1, p2)]:
                        mid = 0.5 * (w1 + w2)
                        mx = int(np.clip(round(mid[0]), 0, uw - 1))
                        my = int(np.clip(round(mid[1]), 0, uh - 1))
                        if up_mask[my, mx] == 0:
                            dx = float(apex[0] - mid[0])
                            dy = float(apex[1] - mid[1])
                            if math.hypot(dx, dy) > 1e-3:
                                return self._vector_to_compass_deg(dx, dy)

            m = cv2.moments(best_c)
            if m["m00"] > 1e-5:
                mcx = m["m10"] / m["m00"]
                mcy = m["m01"] / m["m00"]
                pts = best_c.reshape(-1, 2)
                dists = np.hypot(pts[:, 0] - mcx, pts[:, 1] - mcy)
                tip_pt = pts[int(np.argmax(dists))]
                dx = float(tip_pt[0] - mcx)
                dy = float(tip_pt[1] - mcy)
                if math.hypot(dx, dy) > 1e-3:
                    return self._vector_to_compass_deg(dx, dy)
        except Exception as exc:
            logger.debug("_get_template_base_heading hatası: %s", exc)
        return 0.0

    def _get_cached_template_base_heading(self, tpl: np.ndarray, mask: np.ndarray) -> float:
        """Şablonun doğal yön açısını döndürür; sonuç yalnızca (tpl, mask)
        ikilisine bağlı olduğundan örnek başına bir kez hesaplanıp saklanır.
        Şablon/maske kimliği değişirse önbellek sıfırlanır."""
        if (
            self._heading_cache_tpl is tpl
            and self._heading_cache_mask is mask
            and self._heading_base_heading is not None
        ):
            return self._heading_base_heading
        heading = self._get_template_base_heading(tpl, mask)
        self._heading_cache_tpl = tpl
        self._heading_cache_mask = mask
        self._heading_base_heading = heading
        self._heading_rot_cache = {}
        return heading

    def _get_heading_rotation(
        self, deg: float, tpl: np.ndarray, mask: np.ndarray, base_heading: float
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """`deg` açısına döndürülmüş (şablon, maske, maske piksel sayısı)
        üçlüsünü döndürür; döndürme yalnızca (şablon, maske, açı) üçlüsüne
        bağlı olduğundan kareler arasında yeniden kullanılır. Şablon/maske
        kimliği değişirse önbellek sıfırlanır."""
        if self._heading_cache_tpl is not tpl or self._heading_cache_mask is not mask:
            self._heading_cache_tpl = tpl
            self._heading_cache_mask = mask
            self._heading_base_heading = None
            self._heading_rot_cache = {}
        cached = self._heading_rot_cache.get(float(deg))
        if cached is not None:
            return cached
        th, tw = tpl.shape[:2]
        center = (tw / 2.0, th / 2.0)
        rot_deg = float(deg) - base_heading
        M = cv2.getRotationMatrix2D(center, -rot_deg, 1.0)
        rot_tpl = cv2.warpAffine(
            tpl, M, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        rot_m = cv2.warpAffine(
            mask, M, (tw, th), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        entry = (rot_tpl, rot_m, int(np.sum(rot_m > 0)))
        self._heading_rot_cache[float(deg)] = entry
        return entry

    def _detect_heading_via_template(self, gray_roi: np.ndarray) -> Optional[float]:
        """Şablon döndürme ve eşleme ile oyuncu ok açısını güvenilir şekilde tespit eder."""
        if self._player_arrow_template is None or self._player_arrow_mask is None:
            return None

        try:
            tpl_g = self._player_arrow_template
            tpl_m = self._player_arrow_mask
            th, tw = tpl_g.shape[:2]
            rh, rw = gray_roi.shape[:2]
            if rh < th or rw < tw:
                return None

            base_heading = self._get_cached_template_base_heading(tpl_g, tpl_m)

            best_score = -1.0
            best_angle = 0.0

            # 1. Kaba arama: 10 derecelik adımlarla
            for deg in range(0, 360, 10):
                rot_tpl, rot_m, mask_px = self._get_heading_rotation(float(deg), tpl_g, tpl_m, base_heading)
                if mask_px < 6:
                    continue
                res = cv2.matchTemplate(gray_roi, rot_tpl, cv2.TM_CCOEFF_NORMED, mask=rot_m)
                _, max_v, _, _ = cv2.minMaxLoc(res)
                if float(max_v) > best_score:
                    best_score = float(max_v)
                    best_angle = float(deg)

            # Eşik kontrolü: Eğer kaba eşleme yeterince yüksekse ince ayar yap
            if best_score >= 0.52:
                refined_angle = best_angle
                for delta in (-8, -6, -4, -2, 2, 4, 6, 8):
                    test_deg = (best_angle + float(delta) + 360.0) % 360.0
                    rot_tpl, rot_m, mask_px = self._get_heading_rotation(test_deg, tpl_g, tpl_m, base_heading)
                    if mask_px < 6:
                        continue
                    res = cv2.matchTemplate(gray_roi, rot_tpl, cv2.TM_CCOEFF_NORMED, mask=rot_m)
                    _, max_v, _, _ = cv2.minMaxLoc(res)
                    if float(max_v) > best_score:
                        best_score = float(max_v)
                        refined_angle = test_deg
                return round((refined_angle + 360.0) % 360.0, 2)
        except Exception as exc:
            logger.debug("_detect_heading_via_template hatası: %s", exc)

        return None

    def _detect_heading_via_moments(self, gray_roi: np.ndarray) -> Tuple[bool, float]:
        """Kontur köşe, tepe/çentik ve moment analizi ile yön açısını hesaplar."""
        try:
            scale = 8
            up_gray = cv2.resize(
                gray_roi,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )

            best_contour = None
            # 8x büyütülmüş uzayda eşik de 8^2 ile ölçeklenir: orijinalde
            # >= 10 px^2 alan şartı (önceki 40 px^2, büyütülmüş uzayda
            # ~0.6 px^2'e denk gelip parlak gürültüyü geçiriyordu).
            min_area_upscaled = 10.0 * float(scale) * float(scale)
            for thresh_val in (205, 195, 185, 175):
                _, bw = cv2.threshold(up_gray, thresh_val, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                valid_cnts = [c for c in contours if cv2.contourArea(c) >= min_area_upscaled]
                if valid_cnts:
                    best_contour = max(valid_cnts, key=cv2.contourArea)
                    break

            if best_contour is None:
                return False, 0.0

            moments = cv2.moments(best_contour)
            if moments["m00"] <= 1e-5:
                return False, 0.0
            mcx = moments["m10"] / moments["m00"]
            mcy = moments["m01"] / moments["m00"]

            peri = cv2.arcLength(best_contour, True)
            approx = cv2.approxPolyDP(best_contour, 0.03 * peri, True).reshape(-1, 2)

            if len(approx) >= 3:
                best_tri: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
                max_perim = -1.0
                for p1, p2, p3 in itertools.combinations(approx, 3):
                    d12 = float(np.linalg.norm(p1 - p2))
                    d23 = float(np.linalg.norm(p2 - p3))
                    d31 = float(np.linalg.norm(p3 - p1))
                    tri_p = d12 + d23 + d31
                    if tri_p > max_perim:
                        max_perim = tri_p
                        best_tri = (p1, p2, p3)

                if best_tri is not None:
                    p1, p2, p3 = best_tri
                    candidates = [(p1, p2, p3), (p2, p3, p1), (p3, p1, p2)]
                    chosen_apex = p1
                    chosen_base_mid = 0.5 * (p2 + p3)
                    lowest_notch_intensity = 999.0

                    uh, uw = up_gray.shape[:2]
                    for apex, wing1, wing2 in candidates:
                        mid_wings = 0.5 * (wing1 + wing2)
                        mx = int(np.clip(round(mid_wings[0]), 0, uw - 1))
                        my = int(np.clip(round(mid_wings[1]), 0, uh - 1))
                        notch_val = float(up_gray[my, mx])
                        if notch_val < lowest_notch_intensity:
                            lowest_notch_intensity = notch_val
                            chosen_apex = apex
                            chosen_base_mid = mid_wings

                    # Mutlak koyuluk şartı: üç aday çentik de parlaksa
                    # (arka plan koyu olmalı) tepe seçimi geçersizdir;
                    # moment/uç geri dönüşüne düşülür.
                    if lowest_notch_intensity < 160.0:
                        dx = float(chosen_apex[0] - chosen_base_mid[0])
                        dy = float(chosen_apex[1] - chosen_base_mid[1])
                        if math.hypot(dx, dy) > 1e-3:
                            heading = self._vector_to_compass_deg(dx, dy)
                            return True, heading

            pts = best_contour.reshape(-1, 2)
            dists = np.hypot(pts[:, 0] - mcx, pts[:, 1] - mcy)
            tip_pt = pts[int(np.argmax(dists))]
            dx = float(tip_pt[0] - mcx)
            dy = float(tip_pt[1] - mcy)
            if math.hypot(dx, dy) <= 1e-3:
                return False, 0.0

            heading = self._vector_to_compass_deg(dx, dy)
            return True, heading
        except Exception as exc:
            logger.debug("_detect_heading_via_moments hatası: %s", exc)
            return False, 0.0

    # ------------------------------------------------------------------
    # 1. Oyuncu Yön Açısı (Player Heading) Tespiti
    # ------------------------------------------------------------------
    def detect_player_heading(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Minimap merkezindeki beyaz/parlak oyuncu okunun baktığı yön açısını
        (0.0° - 360.0°, Kuzey=0°, Doğu=90°) şablon eşleme ve moment/kontur analizi ile hesaplar.
        """
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return False, 0.0

            crop, _ = self.extract_minimap(frame)
            ch, cw = crop.shape[:2]
            if ch < 20 or cw < 20:
                return False, 0.0

            cx_f, cy_f = self.get_minimap_player_center(crop.shape, crop_bgr=crop)
            ax = int(round(cx_f))
            ay = int(round(cy_f))
            half_win = max(6, int(round(min(cw, ch) * 0.045)))
            rx1 = max(0, ax - half_win)
            ry1 = max(0, ay - half_win)
            rx2 = min(cw, ax + half_win + 1)
            ry2 = min(ch, ay + half_win + 1)

            roi = crop[ry1:ry2, rx1:rx2]
            if roi.size == 0:
                return False, 0.0

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

            # 1. Şablon eşleme ile yön tespiti (eğer şablon mevcut ve güvenilirse)
            tpl_heading = self._detect_heading_via_template(gray_roi)
            if tpl_heading is not None:
                return True, tpl_heading

            # 2. Geri dönüş (Fallback): Moment ve kontur köşe analizi
            return self._detect_heading_via_moments(gray_roi)

        except Exception as exc:
            logger.error("detect_player_heading hatası: %s", exc)
            return False, 0.0

    # ------------------------------------------------------------------
    # 2. Minimap Filter Settings Açılır Penceresi Durum Tespiti
    # ------------------------------------------------------------------
    def detect_minimap_filter_state(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Minimap solundaki `Minimap Filter Settings` açılır penceresinin açık olup olmadığını,
        `Monster` ve `Gatherable M...` onay kutularının seçili olup olmadığını tespit eder.
        """
        result: Dict[str, Any] = {
            "popup_open": False,
            "quest_checked": False,
            "monster_checked": False,
            "gatherable_checked": False,
            "npc_checked": False,
        }
        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return result

            bgr = (
                cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                if frame.ndim == 2
                else (
                    cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                    if frame.shape[2] == 4
                    else frame
                )
            )
            h, w = bgr.shape[:2]
            if w < 500 or h < 250:
                return result

            if w >= 1000:
                # Gerçek frame_h kullanılır (16:9 ref_h düzeltmesi kaldırıldı;
                # bkz. get_minimap_pixel_bounds).
                px1 = int(round(w * 0.723))
                py1 = int(round(h * 0.066))
                px2 = int(round(w * 0.842))
                py2 = min(h, int(round(h * 0.248)))
            else:
                px1 = int(round(w * 0.020))
                py1 = int(round(h * 0.200))
                px2 = int(round(w * 0.433))
                py2 = int(round(h * 0.920))

            if px2 <= px1 + 20 or py2 <= py1 + 20:
                return result

            panel = bgr[py1:py2, px1:px2]
            hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
            ph, pw = panel.shape[:2]

            # İç arka plan (metin ve onay kutusu dışındaki koyu arduvaz zemin)
            bg_strip = hsv[int(ph * 0.22) : int(ph * 0.90), int(pw * 0.65) : int(pw * 0.78)]
            if bg_strip.size == 0:
                return result

            mean_v = float(np.mean(bg_strip[:, :, 2]))
            std_v = float(np.std(bg_strip[:, :, 2]))
            mean_s = float(np.mean(bg_strip[:, :, 1]))

            if not (22.0 <= mean_v <= 68.0 and std_v <= 28.0 and 15.0 <= mean_s <= 90.0):
                return result

            cb_col = hsv[:, int(pw * 0.84) : int(pw * 0.96), 2]
            if cb_col.size == 0:
                return result

            def _band_checked(y0: float, y1: float) -> bool:
                # Tek parlak piksel yetmez: bandın %90 persentili >= 170
                # VE en az 3 piksel >= 170 olmalı (onay işareti bir leke).
                band = cb_col[int(ph * y0) : int(ph * y1), :]
                if band.size == 0:
                    return False
                p90 = float(np.percentile(band, 90))
                bright_px = int(np.sum(band >= 170))
                return bool(p90 >= 170.0 and bright_px >= 3)

            quest = _band_checked(0.22, 0.39)
            monster = _band_checked(0.41, 0.57)
            gatherable = _band_checked(0.59, 0.75)
            npc = _band_checked(0.77, 0.93)

            if not (quest or monster or gatherable or npc):
                return result

            result["popup_open"] = True
            result["quest_checked"] = quest
            result["monster_checked"] = monster
            result["gatherable_checked"] = gatherable
            result["npc_checked"] = npc
            return result

        except Exception as exc:
            logger.debug("detect_minimap_filter_state hatası: %s", exc)
            return result

    # ------------------------------------------------------------------
    # 3. Maskeli Normalize Çapraz Korelasyon (3-Yapraklı Filiz Tespiti)
    # ------------------------------------------------------------------
    @staticmethod
    def _masked_ncc_local_std(
        gray_img: np.ndarray,
        mask: np.ndarray,
        min_local_contrast: float = 26.0,
    ) -> Optional[np.ndarray]:
        """
        Maskeli NCC'nin yalnızca maskeye/görüntüye bağlı kısmını (yerel
        standart sapma haritasını) hesaplar. Aynı (görüntü, maske) ikilisi
        birden çok şablonla kullanıldığında (ör. iki kanonik filiz şablonu
        aynı maskeyi paylaşır) bu iki `filter2D` çağrısı şablon başına
        tekrarlanmak yerine bir kez yapılıp paylaşılır.
        Maske pikseli < 12 ise `None` döner.
        """
        m = (mask > 0).astype(np.float32)
        n = float(np.sum(m))
        if n < 12.0:
            return None

        img_f = gray_img.astype(np.float32)
        loc_sum = cv2.filter2D(img_f, cv2.CV_32F, m)
        loc_sq_sum = cv2.filter2D(img_f * img_f, cv2.CV_32F, m)
        loc_var = np.maximum(loc_sq_sum - (loc_sum * loc_sum) / n, 1.0)
        return np.sqrt(loc_var)

    @staticmethod
    def _masked_ncc(
        gray_img: np.ndarray,
        tpl: np.ndarray,
        mask: np.ndarray,
        min_local_contrast: float = 26.0,
        loc_std: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Yalnızca ikon + koyu dış kontur maskesi (`mask > 0`) üzerinde sıfır-ortalama
        birim-varyanslı çapraz korelasyon hesaplar; böylece minimap zemin parlaklığından
        (parlak sarı radar halkası veya koyu çim zemin) bağımsız olarak 3-yapraklı
        filiz ikonunu yüksek hassasiyetle yakalar.

        `loc_std` verilirse (aynı görüntü+maske için
        `_masked_ncc_local_std` ile önceden hesaplanmış) yerel istatistik
        `filter2D` çağrıları atlanır; verilmezse eskisi gibi yerinde hesaplanır.
        `loc_std` yalnızca bu çağrıdaki `gray_img`/`mask` için hesaplanmış
        bir harita olmalıdır.
        """
        m = (mask > 0).astype(np.float32)
        n = float(np.sum(m))
        if n < 12.0:
            return np.zeros(gray_img.shape[:2], dtype=np.float32)

        t = tpl.astype(np.float32)
        t_mean = float(np.sum(t * m)) / n
        t_zero = (t - t_mean) * m
        t_norm = float(np.sqrt(np.sum(t_zero * t_zero))) + 1e-6
        t_kernel = t_zero / t_norm

        img_f = gray_img.astype(np.float32)
        if loc_std is None:
            loc_std = MinimapScanner._masked_ncc_local_std(gray_img, mask, min_local_contrast)
            if loc_std is None:
                return np.zeros(gray_img.shape[:2], dtype=np.float32)

        corr = cv2.filter2D(img_f, cv2.CV_32F, t_kernel) / loc_std
        contrast = loc_std / math.sqrt(n)
        corr[contrast < float(min_local_contrast)] = 0.0
        return corr

    @classmethod
    def is_minimap_class(cls, class_id: int) -> bool:
        """Sınıf kimliğinin (class_id) yalnızca minimap alanına (9: filiz, 11: canavar noktası) ait olduğunu denetler."""
        return int(class_id) in cls.MINIMAP_CLASS_IDS

    @classmethod
    def is_world_3d_class(cls, class_id: int) -> bool:
        """Sınıf kimliğinin 3D dünya / HUD alanına ait olduğunu ve asla minimap sınıfı olamayacağını denetler."""
        return int(class_id) in cls.WORLD_3D_CLASS_IDS

    @classmethod
    def sanitize_minimap_detection(cls, det: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimap tespitlerinin asla 3D dünya sınıflarıyla karıştırılmamasını garanti eder.
        Taksonomi izolasyonu ve değişmez sınıf etiketlerini zorunlu kılar.
        """
        cid = int(det.get("class_id", -1))
        if cid == cls.CLASS_ID_SPROUT:
            det["class_id"] = 9
            det["class_name"] = cls.CLASS_NAME_SPROUT
            det["is_gatherable"] = True
            det["is_hostile"] = False
            det["is_minimap"] = True
            det["is_world_3d"] = False
        elif cid == cls.CLASS_ID_MONSTER_DOT:
            det["class_id"] = 11
            det["is_gatherable"] = False
            det["is_minimap"] = True
            det["is_world_3d"] = False
            if det.get("is_hostile"):
                det["class_name"] = cls.CLASS_NAME_HOSTILE_DOT
            else:
                det["class_name"] = cls.CLASS_NAME_MONSTER_DOT
        elif cls.is_world_3d_class(cid):
            raise ValueError(
                f"Taksonomi İhlali: 3D Dünya sınıfı (class_id={cid}) minimap tarayıcısına sızamaz!"
            )
        else:
            raise ValueError(
                f"Bilinmeyen class_id={cid}: minimap sınıfı (9/11) değil; "
                "sessizce minimap diye damgalanamaz."
            )
        det["is_minimap"] = True
        det["is_world_3d"] = False
        return det

    def _is_cardinal_compass_marker(
        self,
        px: float,
        py: float,
        cx: float,
        cy: float,
        cw: int,
        ch: int,
    ) -> bool:
        """
        Aday (px, py) noktasının minimap radar çevresindeki statik ana yön
        işaretçilerine (Kuzey-N, Güney-S, Doğu-E, Batı-W) denk gelip gelmediğini doğrular.
        """
        # Kuzey kutbu (North: üst başlık kenarı ve merkez hizası)
        if abs(px - cx) <= max(12.0, cw * 0.045) and py <= (ch * 0.18):
            return True
        # Güney kutbu (South: alt sınır ve merkez hizası)
        if abs(px - cx) <= max(12.0, cw * 0.045) and py >= (ch * 0.92):
            return True
        # Batı kutbu (West: sol sınır ve merkez hizası)
        if abs(py - cy) <= max(12.0, ch * 0.045) and px <= (cw * 0.06):
            return True
        # Doğu kutbu (East: sağ dış sınır ve merkez hizası)
        if abs(py - cy) <= max(12.0, ch * 0.045) and px >= (cw * 0.94):
            return True
        return False

    def _check_sprout_chromatic_density(
        self,
        patch_hsv: np.ndarray,
        scale: float = 1.0,
    ) -> Tuple[bool, float]:
        """
        Yeşil 3-yapraklı filiz için yeşil/sarı-yeşil ve kanonik gümüş/beyaz HSV kromatik yoğunluğunu denetler.
        Hostile kırmızı canavar pikselleri veya mavi kanat baskınlığı varsa anında elenir.
        """
        if patch_hsv is None or patch_hsv.size == 0:
            return False, 0.0

        total_pixels = float(patch_hsv.shape[0] * patch_hsv.shape[1])
        if total_pixels <= 0.0:
            return False, 0.0

        # 1. Kırmızı canavar noktası kontrolü: Kırmızı pikseller varsa ASLA filiz değildir!
        red_p1 = (patch_hsv[:, :, 0] <= 14) & (patch_hsv[:, :, 1] >= 85) & (patch_hsv[:, :, 2] >= 85)
        red_p2 = (patch_hsv[:, :, 0] >= 166) & (patch_hsv[:, :, 1] >= 85) & (patch_hsv[:, :, 2] >= 85)
        red_mask = red_p1 | red_p2
        if float(np.mean(red_mask)) > 0.08 or int(np.sum(red_mask)) >= 4:
            return False, 0.0

        # 2. Yeşil HSV kromatikliği (Hue: 32..85, Sat >= 35, Val >= 70)
        green_sprout_mask = (
            (patch_hsv[:, :, 0] >= 32)
            & (patch_hsv[:, :, 0] <= 85)
            & (patch_hsv[:, :, 1] >= 35)
            & (patch_hsv[:, :, 2] >= 70)
        )

        # 3. Beyaz/Gümüş kanonik filiz yaprakları (Val >= 170, Sat <= 70)
        white_mask = (patch_hsv[:, :, 2] >= 170) & (patch_hsv[:, :, 1] <= 70)

        # 4. Sarı/Açık yeşil filiz pikselleri (Hue: 16..35, Sat >= 50, Val >= 85)
        yellow_sprout_mask = (
            (patch_hsv[:, :, 0] >= 16)
            & (patch_hsv[:, :, 0] <= 35)
            & (patch_hsv[:, :, 1] >= 50)
            & (patch_hsv[:, :, 2] >= 85)
        )

        sprout_color_mask = green_sprout_mask | white_mask | yellow_sprout_mask
        chromatic_count = int(np.sum(sprout_color_mask))
        chromatic_density = float(chromatic_count) / total_pixels

        min_required_pixels = max(8, int(round(14.0 * scale * scale)))
        if chromatic_density < 0.06 or chromatic_count < min_required_pixels:
            return False, chromatic_density

        # 5. Yüksek doygunluklu mavi/camgöbeği kanat kontrolü
        cyan_colored = (patch_hsv[:, :, 1] >= 85) & (patch_hsv[:, :, 0] >= 82) & (patch_hsv[:, :, 0] <= 125)
        if float(np.mean(cyan_colored)) > 0.20:
            return False, chromatic_density

        return True, chromatic_density

    def _verify_sprout_patch_structure(
        self,
        gray_crop: np.ndarray,
        hsv_crop: np.ndarray,
        cx: int,
        cy: int,
        scale: float,
    ) -> bool:
        """
        Aday (cx, cy) noktasının gerçekten toplanabilir filiz (`sprout_material`,
        yeşil veya beyaz/gümüş 3 yapraklı filiz) olduğunu doğrular.
        """
        ok, _ = self._verify_sprout_patch_structure_with_density(
            gray_crop, hsv_crop, cx, cy, scale
        )
        return ok

    def _verify_sprout_patch_structure_with_density(
        self,
        gray_crop: np.ndarray,
        hsv_crop: np.ndarray,
        cx: int,
        cy: int,
        scale: float,
    ) -> Tuple[bool, float]:
        """
        Aday (cx, cy) noktasının yeşil HSV kromatik yoğunluğu, kontrastı ve yapısal konturlarını
        analiz ederek doğrular ve kromatik yoğunluk skorunu döndürür.
        """
        ch, cw = gray_crop.shape[:2]
        rad = max(5, int(round(8.0 * scale)))
        x1 = max(0, cx - rad)
        y1 = max(0, cy - rad)
        x2 = min(cw, cx + rad + 1)
        y2 = min(ch, cy + rad + 1)

        patch_g = gray_crop[y1:y2, x1:x2]
        patch_hsv = hsv_crop[y1:y2, x1:x2]
        ph, pw = patch_g.shape[:2]
        if ph < 9 or pw < 9:
            return False, 0.0

        # 1. Yeşil/Kromatik HSV yoğunluk denetimi
        ok_chrom, chromatic_density = self._check_sprout_chromatic_density(patch_hsv, scale)
        if not ok_chrom:
            return False, chromatic_density

        # 2. Koyu dış sınır / kontrast kontrolü
        p_min = float(np.min(patch_g))
        p_max = float(np.max(patch_g))
        if (p_max - p_min) < 60.0 or p_min > 135.0:
            return False, chromatic_density

        # 3. Yuvarlak canavar noktası kontrolü: eğer aday küçük dairesel bir canavar noktasıysa (area <= 85, circ > 0.74, solidity > 0.86) reddet
        white_m = (patch_hsv[:, :, 2] >= 170) & (patch_hsv[:, :, 1] <= 70)
        green_m = (
            (patch_hsv[:, :, 0] >= 32)
            & (patch_hsv[:, :, 0] <= 85)
            & (patch_hsv[:, :, 1] >= 35)
            & (patch_hsv[:, :, 2] >= 70)
        )
        yellow_m = (
            (patch_hsv[:, :, 0] >= 16)
            & (patch_hsv[:, :, 0] <= 35)
            & (patch_hsv[:, :, 1] >= 50)
            & (patch_hsv[:, :, 2] >= 85)
        )
        sprout_mask = white_m | green_m | yellow_m
        bw_local = sprout_mask.astype(np.uint8) * 255
        cnts, _ = cv2.findContours(bw_local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            main_c = max(cnts, key=cv2.contourArea)
            area = float(cv2.contourArea(main_c))
            peri = float(cv2.arcLength(main_c, True))
            circ = 4.0 * math.pi * area / (peri * peri + 1e-6)
            hull_area = float(cv2.contourArea(cv2.convexHull(main_c)))
            solidity = area / (hull_area + 1e-6)
            if area <= 85.0 and circ > 0.74 and solidity > 0.86:
                return False, chromatic_density

        return True, chromatic_density

    @staticmethod
    def _measure_stack_badge(
        hsv_crop: np.ndarray, cx: int, cy: int, scale: float = 1.0
    ) -> Tuple[int, int]:
        """Filiz ikonunun yanındaki yeşil yığın rozetini (`2`/`3`) okur (F-0130).

        `core/auto_cartographer.py::_detect_minimap_gatherables` ile aynı kanıtlanmış
        mantık: rozet HSV aralığı (H: 68..90, S >= 75, V >= 115), yalnızca kompakt
        rozet boyutunda bileşenler sayılır (dağınık yeşil arazi pikselleri sayılmaz,
        F-0055); `badge_px >= 28` -> 3, `>= 10` -> 2, yoksa 1. Eşikler referans
        ölçeğe (332px minimap) göre `scale` ile ölçeklenir.

        Döndürür: (stack_count, badge_px). Rozet yoksa stack_count=1 döner — bu
        artık uydurma bir varsayılan değil, o karede rozet aranıp bulunamadığının
        ölçülmüş sonucudur; çağıran `stack_count_measured=True` yazmalıdır.
        Herhangi bir hatada fail-safe (1, 0) döner.
        """
        try:
            ch, cw = hsv_crop.shape[:2]
            s = max(0.5, float(scale))
            half = 8.0 * s
            x1 = max(0, int(math.floor(cx - half - 4.0 * s)))
            y1 = max(0, int(math.floor(cy - half - 4.0 * s)))
            x2 = min(cw, int(math.ceil(cx + half + 8.0 * s)))
            y2 = min(ch, int(math.ceil(cy + half + 6.0 * s)))
            if x2 <= x1 or y2 <= y1:
                return 1, 0
            badge_mask = cv2.inRange(
                hsv_crop[y1:y2, x1:x2], (68, 75, 115), (90, 255, 255)
            )
            min_area = 8.0 * s * s
            min_side = max(2.0, 3.0 * s)
            max_side = 24.0 * s
            badge_px = 0
            num_b, _, stats_b, _ = cv2.connectedComponentsWithStats(badge_mask)
            for b_idx in range(1, num_b):
                _bx, _by, _bw, _bh, _area = stats_b[b_idx]
                if (
                    _area < min_area
                    or _bw < min_side
                    or _bh < min_side
                    or _bw > max_side
                    or _bh > max_side
                ):
                    continue
                if float(_area) / float(max(1, _bw * _bh)) < 0.30:
                    continue
                badge_px += int(_area)
            if badge_px >= int(round(28.0 * s * s)):
                return 3, int(badge_px)
            if badge_px >= int(round(10.0 * s * s)):
                return 2, int(badge_px)
            return 1, int(badge_px)
        except Exception:
            return 1, 0

    # ------------------------------------------------------------------
    # 4. Ana Minimap Kesit Analizörü (3-Yapraklı Filiz vs Canavar Noktası)
    # ------------------------------------------------------------------
    def scan_minimap_crop(
        self,
        crop_bgr: np.ndarray,
        offset_xy: Tuple[int, int] = (0, 0),
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Doğrudan bir minimap BGR kesiti (`crop_bgr`) üzerinde çalışarak:
        - `gatherable_nodes`: YALNIZCA beyaz 3-yapraklı filiz (`kind="sprout_material"`) düğümlerini,
        - `monster_dots`: Küçük yuvarlak beyaz canavar noktalarını (`•`),
        - `hostile_dots`: Kırmızı düşman noktalarını,
        - `rejected_icons`: Beyaz tüy (`odyle_feather`), mavi kanat (`cyan_wing`) ve yeşil rozetleri
        birbirinden ayırarak döndürür.
        """
        gatherable_nodes: List[Dict[str, Any]] = []
        monster_dots: List[Dict[str, Any]] = []
        hostile_dots: List[Dict[str, Any]] = []
        rejected_icons: List[Dict[str, Any]] = []

        if crop_bgr is None or not isinstance(crop_bgr, np.ndarray) or crop_bgr.size == 0:
            return {
                "gatherable_nodes": gatherable_nodes,
                "monster_dots": monster_dots,
                "hostile_dots": hostile_dots,
                "rejected_icons": rejected_icons,
                "player_center": (0.0, 0.0),
                "player_center_found": False,
            }

        if crop_bgr.ndim == 2:
            crop = cv2.cvtColor(crop_bgr, cv2.COLOR_GRAY2BGR)
        elif crop_bgr.shape[2] == 4:
            crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGRA2BGR)
        else:
            crop = crop_bgr

        ch, cw = crop.shape[:2]
        if ch < 20 or cw < 20:
            return {
                "gatherable_nodes": gatherable_nodes,
                "monster_dots": monster_dots,
                "hostile_dots": hostile_dots,
                "rejected_icons": rejected_icons,
                "player_center": (0.0, 0.0),
                "player_center_found": False,
            }

        ox, oy = int(offset_xy[0]), int(offset_xy[1])
        full_h = int(frame_shape[0]) if frame_shape is not None and len(frame_shape) >= 2 else ch
        full_w = int(frame_shape[1]) if frame_shape is not None and len(frame_shape) >= 2 else cw

        center_x, center_y = self.get_minimap_player_center(crop.shape, crop_bgr=crop)
        clearance_px = float(min(cw, ch)) * self.PLAYER_CLEARANCE_NORM
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # UI kenar maskesi: üst başlık ("Pilgrim's Ridge" / "Western Altamia Highland"),
        # sağ üst büyüteç ("Q"), sağ alt saat ("10:09" / "10:16") ve merkez oyuncu oku
        valid_mask = np.zeros((ch, cw), dtype=np.uint8)
        top_margin = int(round(ch * 0.16))
        bottom_margin = int(round(ch * 0.96))
        side_margin = int(round(cw * 0.04))
        valid_mask[top_margin:bottom_margin, side_margin : cw - side_margin] = 255

        # Sağ üst büyüteç ikonu maskesi
        valid_mask[: int(round(ch * 0.32)), int(round(cw * 0.82)) :] = 0
        # Sağ alt saat ve mesafe ("1km / 10:16") maskesi
        valid_mask[int(round(ch * 0.76)) :, int(round(cw * 0.64)) :] = 0

        # Merkezdeki oyuncu okunu maskele (radius <= 14px ve klerans koruması)
        arrow_mask_rad = max(14, int(round(min(cw, ch) * 0.055)))
        cv2.circle(
            valid_mask,
            (int(round(center_x)), int(round(center_y))),
            arrow_mask_rad,
            0,
            -1,
        )

        # Ölçek faktörü (332px referans minimap genişliğine göre)
        scale_ref = float(cw) / 332.0

        # ------------------------------------------------------------------
        # A) Kırmızı Düşman Noktaları (hostile_dots, class 11)
        # Sıkı kriter: Hostile red HSV (H: 0..10 | 170..180, S >= 120, V >= 120)
        #              VE dairesellik circularity >= 0.65
        #              Merkezdeki oyuncu oku (radius <= 14px) ve sabit pusula (N,S,E,W) dışlanır
        # ------------------------------------------------------------------
        red_valid = valid_mask.copy()
        red_valid[int(round(ch * 0.89)) :, :] = 0
        red_low1 = cv2.inRange(hsv, np.array([0, 120, 120]), np.array([10, 255, 255]))
        red_low2 = cv2.inRange(hsv, np.array([170, 120, 120]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_and(cv2.bitwise_or(red_low1, red_low2), red_valid)

        red_cnts, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in red_cnts:
            area = float(cv2.contourArea(cnt))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if area < 3.0 or area > 95.0 or bw > 16 or bh > 16:
                continue

            # Hostile red circularity >= 0.65 zorunlu dairesellik şartı
            peri = float(cv2.arcLength(cnt, True))
            circularity = 4.0 * math.pi * area / (peri * peri + 1e-6)
            if circularity < self.HOSTILE_RED_CIRCULARITY_MIN:
                continue

            ix = float(bx) + float(bw) * 0.5
            iy = float(by) + float(bh) * 0.5
            dx = round(ix - center_x, 2)
            dy = round(iy - center_y, 2)
            dist_px = round(math.hypot(dx, dy), 2)

            # Merkezdeki oyuncu oku (radius <= 14px) ve 0.075 oyuncu kleransı
            if dist_px <= self.CENTER_EXCLUSION_RADIUS_PX or dist_px <= clearance_px:
                continue

            # Sabit pusula / ana yön işaretçilerini (N, S, E, W) dışla
            if self._is_cardinal_compass_marker(ix, iy, center_x, center_y, cw, ch):
                continue

            angle_deg = self._vector_to_compass_deg(dx, dy)
            sx = int(round(ox + ix))
            sy = int(round(oy + iy))

            h_item = {
                "type": "hostile_dot",
                "category": "hostile_dots",
                "kind": "hostile_dot",
                "class_id": self.CLASS_ID_MONSTER_DOT,
                "class_name": self.CLASS_NAME_HOSTILE_DOT,
                "dx": dx,
                "dy": dy,
                "dist_px": dist_px,
                "angle_deg": angle_deg,
                "pos": (int(round(ix)), int(round(iy))),
                "screen_pos": (sx, sy),
                "screen_x": sx,
                "screen_y": sy,
                "area": round(area, 1),
                "circularity": round(circularity, 3),
                "is_hostile": True,
                "is_odyle": False,
                "is_gatherable": False,
                "is_minimap": True,
                "is_world_3d": False,
                "minimap_monster_dot": True,
            }
            hostile_dots.append(self.sanitize_minimap_detection(h_item))

        # ------------------------------------------------------------------
        # B) Küçük Yuvarlak Beyaz Canavar Noktaları (monster_dots, `•` - REDDEDİLİR!)
        # ------------------------------------------------------------------
        white_mask = ((hsv[:, :, 2] > 190) & (hsv[:, :, 1] < 60)).astype(np.uint8) * 255
        white_mask = cv2.bitwise_and(white_mask, valid_mask)
        white_cnts, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        for cnt in white_cnts:
            area = float(cv2.contourArea(cnt))
            bx, by, bw, bh = cv2.boundingRect(cnt)

            if area < 7.0 or area > 85.0 or bw < 4 or bh < 4 or bw > 12 or bh > 12:
                continue

            aspect = float(bw) / float(max(1, bh))
            if not (0.72 <= aspect <= 1.38):
                continue

            peri = float(cv2.arcLength(cnt, True))
            circularity = 4.0 * math.pi * area / (peri * peri + 1e-6)
            hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
            solidity = area / (hull_area + 1e-6)

            if circularity < 0.62 or solidity < 0.80:
                continue

            mask_c = np.zeros_like(gray)
            cv2.drawContours(mask_c, [cnt], -1, 255, -1)
            ring = cv2.dilate(mask_c, np.ones((5, 5), dtype=np.uint8)) - mask_c
            ring_vals = gray[ring > 0]
            if len(ring_vals) == 0 or float(np.mean(ring_vals < 115)) < 0.26:
                continue

            ix = float(bx) + float(bw) * 0.5
            iy = float(by) + float(bh) * 0.5
            dx = round(ix - center_x, 2)
            dy = round(iy - center_y, 2)
            dist_px = round(math.hypot(dx, dy), 2)

            # Merkezdeki oyuncu oku (radius <= 14px) ve 0.075 oyuncu kleransı
            if dist_px <= self.CENTER_EXCLUSION_RADIUS_PX or dist_px <= clearance_px:
                continue

            # Sabit pusula / ana yön işaretçilerini (N, S, E, W) dışla
            if self._is_cardinal_compass_marker(ix, iy, center_x, center_y, cw, ch):
                continue

            angle_deg = self._vector_to_compass_deg(dx, dy)
            sx = int(round(ox + ix))
            sy = int(round(oy + iy))

            m_item = {
                "type": "monster_dot",
                "category": "monster_dots",
                "kind": "monster_dot",
                "class_id": self.CLASS_ID_MONSTER_DOT,
                "class_name": self.CLASS_NAME_MONSTER_DOT,
                "dx": dx,
                "dy": dy,
                "dist_px": dist_px,
                "angle_deg": angle_deg,
                "pos": (int(round(ix)), int(round(iy))),
                "screen_x": sx,
                "screen_y": sy,
                "circularity": round(circularity, 3),
                "area": round(area, 1),
                "is_hostile": False,
                "is_odyle": False,
                "is_gatherable": False,
                "is_minimap": True,
                "is_world_3d": False,
            }
            monster_dots.append(self.sanitize_minimap_detection(m_item))

        # ------------------------------------------------------------------
        # C) Mavi Kanat / Beyaz Tüy İkonlarını Tespit Et ve Materyal Olarak REDDET!
        # ("tüy odyle değil tüy toplamayacağız")
        # ------------------------------------------------------------------
        cyan_mask = cv2.inRange(hsv, np.array([82, 68, 125]), np.array([115, 255, 255]))
        cyan_mask = cv2.bitwise_and(cyan_mask, valid_mask)
        cyan_mask = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
        cyan_cnts, _ = cv2.findContours(cyan_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cyan_cnts:
            area = float(cv2.contourArea(cnt))
            if area >= 6.0:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                ix = float(bx) + float(bw) * 0.5
                iy = float(by) + float(bh) * 0.5
                rejected_icons.append(
                    {
                        "kind": "rejected_cyan_wing",
                        "dx": round(ix - center_x, 2),
                        "dy": round(iy - center_y, 2),
                        "pos": (int(round(ix)), int(round(iy))),
                        "area": round(area, 1),
                    }
                )

        # Uzun diyagonal beyaz tüy konturlarını da rejected_icons listesine kaydet
        # (Radar çemberi yaylarını hariç tutmak için solidity >= 0.55 ve koyu çerçeve şartı aranır)
        for cnt in white_cnts:
            area = float(cv2.contourArea(cnt))
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if area >= 35.0 and (bh >= 15 or bw >= 15) and bw <= 26 and bh <= 32:
                hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
                solidity = area / (hull_area + 1e-6)
                if solidity < 0.55:
                    continue
                mask_c = np.zeros_like(gray)
                cv2.drawContours(mask_c, [cnt], -1, 255, -1)
                ring = cv2.dilate(mask_c, np.ones((5, 5), dtype=np.uint8)) - mask_c
                ring_vals = gray[ring > 0]
                if len(ring_vals) == 0 or float(np.mean(ring_vals < 90)) < 0.40:
                    continue
                ix = float(bx) + float(bw) * 0.5
                iy = float(by) + float(bh) * 0.5
                rejected_icons.append(
                    {
                        "kind": "rejected_feather",
                        "dx": round(ix - center_x, 2),
                        "dy": round(iy - center_y, 2),
                        "pos": (int(round(ix)), int(round(iy))),
                        "area": round(area, 1),
                    }
                )

        # ------------------------------------------------------------------
        # D) Gerçek Toplanabilir Materyal: Beyaz/Yeşil 3-Yapraklı Filiz (`sprout_material`)
        # Sıkı kriter: Maskeli NCC template match >= 0.88 VE yeşil HSV kromatik yoğunluk denetimi
        # ------------------------------------------------------------------
        templates_to_eval: List[Tuple[np.ndarray, np.ndarray]] = [
            (self._CANONICAL_SPROUT_TPL, self._CANONICAL_SPROUT_MASK),
            (self._CANONICAL_SPROUT_TPL_ALT, self._CANONICAL_SPROUT_MASK),
        ] + self._external_sprout_templates

        combined_score_map = np.zeros((ch, cw), dtype=np.float32)

        # NCC yalnızca geçerli minimap dikdörtgeninde anlamlıdır
        # (valid_mask dışı zaten sıfırlanır): filter2D'leri bbox +
        # çekirdek yarıçapı payı (pad) kırpımında çalıştır, sonucu bbox'a
        # geri yerleştir. Her çıktı pikselinin ks×ks penceresi kırpım
        # içinde gerçek piksellerle kaplı olduğundan (pad, kırpımın görüntü
        # kenarına dayandığı yerde görüntü kenarıyla çakışır ve yansıma
        # aynı kalır) bbox içi skorlar tam-kare hesabıyla aynıdır — yalnızca
        # float32 toplama sırası farkından <1e-6 sapma olabilir.
        bbox_y1, bbox_y2 = top_margin, bottom_margin
        bbox_x1, bbox_x2 = side_margin, cw - side_margin
        bbox_h, bbox_w = bbox_y2 - bbox_y1, bbox_x2 - bbox_x1
        # Değerlendirilecek şablon boyutları: ölçek çarpanlarından gelenler +
        # kanonik şablonun kendi 16px boyutu. scale_ref kırpım genişliğinden
        # türetilir; yakın çekimde (365px) gerçek 16px ikon hiçbir çarpan
        # ölçeğinde (17/18px) yakalanamaz — W37: iki gerçek filizden biri
        # en fazla 0.8468'de kalıp 0.88 eşiğini geçemedi, öteki 0.8845 ile
        # ancak geçti. Yerel boyut her zaman denendiği için 332px referans
        # kırpımlarda küme değişmez ({15, 16, 17}).
        ks_values = sorted(
            {max(9, int(round(16.0 * scale_ref * mult))) for mult in self.SPROUT_NCC_SCALES}
            | {16}
        )
        max_ks = max(ks_values)
        pad = max_ks // 2 + 1
        ncc_y1 = max(0, bbox_y1 - pad)
        ncc_y2 = min(ch, bbox_y2 + pad)
        ncc_x1 = max(0, bbox_x1 - pad)
        ncc_x2 = min(cw, bbox_x2 + pad)
        gray_ncc = gray[ncc_y1:ncc_y2, ncc_x1:ncc_x2]
        off_y, off_x = bbox_y1 - ncc_y1, bbox_x1 - ncc_x1

        # Aynı maskeyi paylaşan şablonlar (iki kanonik şablon) aynı ölçekte
        # aynı yerel istatistiği kullanır: iki filter2D şablon başına değil,
        # (maske, ölçek) başına bir kez çalışır.
        stats_cache: Dict[Tuple[bytes, Tuple[int, int], int], Optional[np.ndarray]] = {}
        for base_tpl, base_mask in templates_to_eval:
            for ks in ks_values:
                t_scaled = cv2.resize(base_tpl, (ks, ks), interpolation=cv2.INTER_AREA)
                m_scaled = (
                    cv2.resize(
                        base_mask.astype(np.float32),
                        (ks, ks),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    > 0.40
                ).astype(np.uint8)
                stats_key = (base_mask.tobytes(), base_mask.shape, ks)
                if stats_key not in stats_cache:
                    stats_cache[stats_key] = self._masked_ncc_local_std(gray_ncc, m_scaled)
                loc_std = stats_cache[stats_key]
                if loc_std is None:
                    continue
                score_map = self._masked_ncc(
                    gray_ncc, t_scaled, m_scaled, min_local_contrast=26.0, loc_std=loc_std
                )
                region = score_map[off_y : off_y + bbox_h, off_x : off_x + bbox_w]
                np.maximum(
                    combined_score_map[bbox_y1:bbox_y2, bbox_x1:bbox_x2],
                    region,
                    out=combined_score_map[bbox_y1:bbox_y2, bbox_x1:bbox_x2],
                )

        # Geçerli minimap alanı dışını sıfırla
        combined_score_map[valid_mask == 0] = 0.0

        # Reddedilen mavi kanat ve tüy ikonlarının yakın çevresini sıfırla
        for rej in rejected_icons:
            rx, ry = rej["pos"]
            cv2.circle(
                combined_score_map,
                (int(rx), int(ry)),
                max(10, int(round(18.0 * scale_ref))),
                0.0,
                -1,
            )

        # Beyaz monster dot merkezlerinin tam üzerini sıfırla — yalnızca
        # noktanın kendisi (yarıçap ~3px); geniş bastırma bitişikteki
        # gerçek filizin NCC tepesini de silerdi. Tepe döngüsündeki
        # yakınlık kontrolü geri kalan ayrımı yapar.
        for mdot in monster_dots:
            mx, my = mdot["pos"]
            cv2.circle(
                combined_score_map,
                (int(mx), int(my)),
                max(3, int(round(3.0 * scale_ref))),
                0.0,
                -1,
            )

        # Kırmızı hostile dot (canavar) merkezlerinin yakın çevresini sıfırla (asla filiz ile karışmasın!)
        for hdot in hostile_dots:
            hx, hy = hdot["pos"]
            cv2.circle(
                combined_score_map,
                (int(hx), int(hy)),
                max(8, int(round(12.0 * scale_ref))),
                0.0,
                -1,
            )

        # Tepe noktalarını (NCC >= 0.88) çıkar ve yapısal/yeşil kromatik doğrulamadan geçir
        work_map = combined_score_map.copy()
        nms_rad = max(8, int(round(12.0 * scale_ref)))

        while True:
            _, max_val, _, max_loc = cv2.minMaxLoc(work_map)
            # Sıkı kriter: NCC template match >= 0.88 zorunluluğu
            if float(max_val) < self.SPROUT_NCC_MIN_THRESHOLD:
                break

            px, py = int(max_loc[0]), int(max_loc[1])
            cv2.circle(work_map, (px, py), nms_rad, 0.0, -1)

            dx = round(float(px) - center_x, 2)
            dy = round(float(py) - center_y, 2)
            dist_px = round(math.hypot(dx, dy), 2)

            # Merkezdeki oyuncu oku (radius <= 14px) dışla
            if dist_px <= self.CENTER_EXCLUSION_RADIUS_PX:
                continue

            # Sabit pusula / ana yön işaretçilerini (N, S, E, W) dışla
            if self._is_cardinal_compass_marker(float(px), float(py), center_x, center_y, cw, ch):
                continue

            # Kırmızı canavar noktaları ve beyaz monster dot'lara yakınlık kontrolü (asla yanlış sınıflandırma)
            if any(math.hypot(px - h["pos"][0], py - h["pos"][1]) < max(8.0, 14.0 * scale_ref) for h in hostile_dots):
                continue
            if any(math.hypot(px - m["pos"][0], py - m["pos"][1]) < max(6.0, 10.0 * scale_ref) for m in monster_dots):
                continue

            # Kırmızı canavar pikselleri kontrolü (canavarla asla karışmasın)
            pr = max(3, int(round(5.0 * scale_ref)))
            py1_c, py2_c = max(0, py - pr), min(ch, py + pr + 1)
            px1_c, px2_c = max(0, px - pr), min(cw, px + pr + 1)
            patch_hsv_small = hsv[py1_c:py2_c, px1_c:px2_c]
            if patch_hsv_small.size > 0:
                red1 = (patch_hsv_small[:, :, 0] <= 14) & (patch_hsv_small[:, :, 1] >= 85) & (patch_hsv_small[:, :, 2] >= 85)
                red2 = (patch_hsv_small[:, :, 0] >= 166) & (patch_hsv_small[:, :, 1] >= 85) & (patch_hsv_small[:, :, 2] >= 85)
                if float(np.mean(red1 | red2)) > 0.08 or int(np.sum(red1 | red2)) >= 4:
                    continue

            # Yeşil HSV kromatik yoğunluk ve yapısal kontur denetimi
            ok_sprout, chrom_density = self._verify_sprout_patch_structure_with_density(
                gray, hsv, px, py, scale_ref
            )
            if not ok_sprout:
                continue

            if dist_px < 5.0:
                continue

            # 0.075 oyuncu kleransı
            if float(max_val) < 0.90 and dist_px <= clearance_px:
                continue

            angle_deg = self._vector_to_compass_deg(dx, dy)
            sx = int(round(ox + px))
            sy = int(round(oy + py))
            norm_x = round(float(sx) / float(max(1, full_w)), 4)
            norm_y = round(float(sy) / float(max(1, full_h)), 4)
            conf = round(min(1.0, max(0.0, float(max_val))), 3)

            # Ölçülen filiz alanı: tepe çevresindeki parlak/kromatik
            # maskenin en büyük kontur alanı (formül değil, ölçüm).
            measured_area: Optional[float] = None
            try:
                ar = max(4, int(round(9.0 * scale_ref)))
                ay1, ay2 = max(0, py - ar), min(ch, py + ar + 1)
                ax1, ax2 = max(0, px - ar), min(cw, px + ar + 1)
                patch_hsv_m = hsv[ay1:ay2, ax1:ax2]
                patch_gray_m = gray[ay1:ay2, ax1:ax2]
                if patch_hsv_m.size > 0:
                    bright_m = ((patch_hsv_m[:, :, 2] >= 170) & (patch_hsv_m[:, :, 1] <= 70)).astype(np.uint8) * 255
                    m_cnts, _ = cv2.findContours(bright_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if m_cnts:
                        measured_area = float(cv2.contourArea(max(m_cnts, key=cv2.contourArea)))
            except Exception:
                measured_area = None

            # Yığın rozeti ölçümü (F-0130): filiz ikonunun yanındaki yeşil
            # rozet (`2`/`3`) bu karede gerçekten okunur; rozet yoksa
            # stack_count=1 ölçülmüş değerdir (uydurma varsayılan değil).
            # harvests_available = 3 * stack_count, o da ölçülmüştür.
            stack_count, _badge_px = self._measure_stack_badge(
                hsv, px, py, scale_ref
            )

            s_item = {
                "dx": dx,
                "dy": dy,
                "dist_px": dist_px,
                "angle_deg": angle_deg,
                "kind": "sprout_material",
                "class_id": self.CLASS_ID_SPROUT,
                "class_name": self.CLASS_NAME_SPROUT,
                # stack/harvest sayıları bu karede rozetten ÖLÇÜLÜR
                # (_measure_stack_badge); downstream max() birleştirmesi
                # (auto_cartographer) artık ölçülmüş değerle çalışır.
                "stack_count": int(stack_count),
                "stack_count_measured": True,
                "harvests_available": int(3 * stack_count),
                "harvests_available_measured": True,
                "screen_x": sx,
                "screen_y": sy,
                "norm_x": norm_x,
                "norm_y": norm_y,
                "confidence": conf,
                "ncc_score": round(float(max_val), 4),
                "chromatic_density": round(chrom_density, 3),
                "type": "resource",
                "category": "resource_icon",
                "pos": (px, py),
                "screen_pos": (sx, sy),
                "area": round(measured_area, 1) if measured_area is not None else round(16.0 * scale_ref * scale_ref, 1),
                "area_measured": bool(measured_area is not None),
                "is_hostile": False,
                "is_odyle": False,
                "is_gatherable": True,
                "is_minimap": True,
                "is_world_3d": False,
            }
            gatherable_nodes.append(self.sanitize_minimap_detection(s_item))
        gatherable_nodes.sort(key=lambda d: float(d["dist_px"]))
        monster_dots.sort(key=lambda d: float(d["dist_px"]))
        hostile_dots.sort(key=lambda d: float(d["dist_px"]))

        return {
            "gatherable_nodes": gatherable_nodes,
            "monster_dots": monster_dots,
            "hostile_dots": hostile_dots,
            "rejected_icons": rejected_icons,
            "player_center": (round(center_x, 2), round(center_y, 2)),
            "player_center_found": True,
            "player_clearance_norm": self.PLAYER_CLEARANCE_NORM,
            "player_clearance_px": round(clearance_px, 2),
        }

    # ------------------------------------------------------------------
    # 5. Tam Kare veya Yakın Çekim Materyal ve Odyle Tarama Metodları
    # ------------------------------------------------------------------
    def scan_gatherable_materials(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Tam ekran veya yakın çekim (`minimap_sprout_closeup.jpg`, `minimap_filter_closeup.jpg`)
        karesinde minimap'i çıkarır ve:
        - `gatherable_nodes` (yalnızca `sprout_material` 3-yapraklı filiz ikonları),
        - `monster_dots` (reddedilen yuvarlak beyaz canavar noktaları `•`),
        - `hostile_dots` (kırmızı düşman noktaları),
        - `rejected_icons` (reddedilen beyaz tüy ve mavi kanat ikonları),
        - `filter_state` (`Minimap Filter Settings` penceresi durumu),
        - `minimap_bounds` (`(x1, y1, x2, y2)`)
        bilgilerini döndürür.
        """
        try:
            crop, (x1, y1, x2, y2) = self.extract_minimap(frame)
            scan_res = self.scan_minimap_crop(
                crop_bgr=crop,
                offset_xy=(x1, y1),
                frame_shape=frame.shape,
            )
            scan_res["minimap_bounds"] = (x1, y1, x2, y2)
            scan_res["filter_state"] = self.detect_minimap_filter_state(frame)
            # H-sweep (dilim 13 O10): filtre durumu artık yalnızca
            # döndürülmüyor, yorumlanıyor da. Ayar penceresi açıkken
            # "Gatherable" işaretsizse minimap'te filiz görünmez ve
            # "düğüm yok" sonucu yanlıştır — tüketici uyarıyı görür.
            fs = scan_res.get("filter_state")
            if (
                isinstance(fs, dict)
                and fs.get("popup_open")
                and not fs.get("gatherable_checked")
            ):
                scan_res["minimap_filter_warning"] = "gatherable_unchecked"
            else:
                scan_res["minimap_filter_warning"] = None
            return scan_res
        except Exception as exc:
            logger.error("scan_gatherable_materials hatası: %s", exc)
            return {
                "gatherable_nodes": [],
                "monster_dots": [],
                "hostile_dots": [],
                "rejected_icons": [],
                "player_center": (0.0, 0.0),
                "player_center_found": False,
                "minimap_bounds": (0, 0, 0, 0),
                "filter_state": self.detect_minimap_filter_state(frame)
                if isinstance(frame, np.ndarray)
                else {},
                "minimap_filter_warning": None,
            }

    def scan_sprout_nodes(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Yalnızca gerçek toplanabilir materyal (`sprout_material` 3-yapraklı filiz)
        düğümlerini döndürür. Küçük yuvarlak beyaz canavar noktalarını (`•`) ve
        beyaz tüy (`minimap_feather_odyle.png`) ikonlarını ASLA döndürmez.
        """
        res = self.scan_gatherable_materials(frame)
        return list(res.get("gatherable_nodes", []))

    def scan_odyle_dots(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        DEPRECATED ad: Odyle (uçan tüy) döndürmez; `scan_sprout_nodes` ile
        aynıdır — yalnızca `sprout_material` filiz düğümleri döner.
        Geriye dönük uyumluluk için korunur; yeni kod `scan_sprout_nodes`
        kullanmalıdır.
        """
        return self.scan_sprout_nodes(frame)

    def get_nearest_sprout_offset(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """
        En yakın gerçek toplanabilir materyalin (`sprout_material`) oyuncuya göre
        `(dx, dy, dist_px)` ofsetini döndürür; yoksa `None` döner.
        """
        nodes = self.scan_sprout_nodes(frame)
        if not nodes:
            return None
        nearest = nodes[0]
        return (
            float(nearest["dx"]),
            float(nearest["dy"]),
            float(nearest["dist_px"]),
        )

    def get_nearest_odyle_offset(
        self, frame: np.ndarray
    ) -> Optional[Tuple[float, float, float]]:
        """DEPRECATED ad: `get_nearest_sprout_offset` ile aynıdır."""
        return self.get_nearest_sprout_offset(frame)

    def _scan_3d_odyle_orbs(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        3D dünyada süzülen GERÇEK Odyle enerji kürelerini tespit eder.

        Kök neden (P2-10 / v19): minimap'te Odyle ikonu YOKTUR — katalog
        (`config/materials_db.py` Odyle spec: `shows_on_minimap=False`,
        `in_air=True`, "NEVER the white feather on the minimap") ve yerleşik
        kararlar (F-0413: beyaz tüy / mavi kanat kesin ret;
        `auto_cartographer.py:3390`: minimap-kaynaklı "odyle" reddedilir)
        bunu kapatır. Dolayısıyla `scan_minimap_crop` hiçbir zaman
        `is_odyle=True` üretemez — bu minimap yolu için DOĞRU davranıştır.
        Odyle yalnızca 3D dünyada görünür: süzülen turkuaz/yeşil enerji
        küresi + [E]/[F] etkileşim kutusu. Bu metot o kanıtı tam karede
        arar; `scan_minimap_resources_and_odyle` yalnızca bu yolla
        `is_odyle=True` / `type="odyle_air"` üretebilir.

        Kanıt zinciri (uydurma eşik yok — tamamı mevcut kural):
        1. Kanonik `ODYLE_HSV_RANGE` (`config/materials_db.py:81`,
           P3-3 tek kaynak — aralık burada kopyalanmaz, import edilir).
        2. 3D dünya maskesi: minimap dikdörtgeni + üst bar + alt skill/HP
           barı + sağ görev paneli + sol-alt sohbet hariç tutulur (yalnızca
           3D dünya pikselleri kanıt sayılır; tüy/kanat zaten minimap'te).
        3. Kompakt blob kapıları (sınıf sabitleri — benek/çizgi reddi).
        4. Blob-maskeli yama (yalnızca lekenin kendi pikselleri) üzerinde
           `classify_3d_material_patch` kararı `"Odyle"` olmalı; maskesiz
           bbox köşelerindeki arazi (bronz kaya) taksonomi oyunu
           Orichalcum Ore'a çeviriyordu (ölçüldü: vein 0.985).
        5. `odyle_orb >= 0.045` ve `targena_gold_green < 0.15`
           (`materials_db.py:1786-1788` bonus koşulu, burada sert kapı).

        Döndürülen isabetlerde `dist_px`, EKRAN merkezinden ekran
        pikselidir (`dist_basis="screen_center_px"`) — 3D küre için minimap
        pikseli ölçülemez (derinlik yok). Tüketici
        (`behavioral_guard.should_fly_to_odyle_node`, 260 px kapısı) bu
        değeri merkezi görüş-alanı kapısı olarak uygular: yalnızca
        bakılan/merkezdeki küre uçuşu onaylar; kenardaki küre muhafazakâr
        biçimde reddedilir (uçuş iptali = güvenli varsayılan).
        `sanitize_minimap_detection` BU isabetlere uygulanmaz: o metot
        minimap taksonomi güvenlik duvarıdır (`is_minimap=True` dayatır,
        9/11 dışı class_id reddeder); küre 3D dünya tespitidir.

        Kanıt yoksa / katalog import edilemezse boş liste döner (fail-closed).
        """
        try:
            from config.materials_db import (
                ODYLE_HSV_RANGE,
                classify_3d_material_patch,
            )
        except Exception as exc:  # pragma: no cover - import ortamı hatası
            logger.warning("Odyle kataloğu yüklenemedi, 3D küre taraması atlandı: %s", exc)
            return []

        try:
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                return []
            if frame.ndim == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 3:
                bgr = frame
            else:
                return []

            fh, fw = int(bgr.shape[0]), int(bgr.shape[1])
            if fh < 40 or fw < 40:
                return []
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

            # 3D dünya maskesi: UI chrome hariç (yalnızca 3D dünya kanıttır).
            world_mask = np.full((fh, fw), 255, dtype=np.uint8)
            try:
                mx1, my1, mx2, my2 = self.get_minimap_pixel_bounds(bgr.shape)
            except Exception:
                mx1 = int(round(fw * self.DEFAULT_NORM_RECT[0]))
                my1 = int(round(fh * self.DEFAULT_NORM_RECT[1]))
                mx2 = int(round(fw * (self.DEFAULT_NORM_RECT[0] + self.DEFAULT_NORM_RECT[2])))
                my2 = int(round(fh * (self.DEFAULT_NORM_RECT[1] + self.DEFAULT_NORM_RECT[3])))
            world_mask[my1:my2, mx1:mx2] = 0  # minimap
            world_mask[: int(round(fh * 0.06)), :] = 0  # üst bar (para birimleri)
            world_mask[int(round(fh * 0.86)) :, :] = 0  # alt skill/HP barı
            world_mask[:, int(round(fw * 0.85)) :] = 0  # sağ görev paneli şeridi
            world_mask[int(round(fh * 0.70)) :, : int(round(fw * 0.30))] = 0  # sol-alt sohbet

            lo = np.array(ODYLE_HSV_RANGE[0], dtype=np.uint8)
            hi = np.array(ODYLE_HSV_RANGE[1], dtype=np.uint8)
            teal_mask = cv2.inRange(hsv, lo, hi)
            teal_mask = cv2.bitwise_and(teal_mask, world_mask)
            teal_mask = cv2.morphologyEx(
                teal_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
            )

            cnts, _ = cv2.findContours(
                teal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            frame_area = float(fw * fh)
            area_min = max(
                self.ODYLE_ORB_MIN_AREA_ABS_PX,
                self.ODYLE_ORB_MIN_AREA_REF_PX * frame_area / self.ODYLE_ORB_REF_AREA_PX,
            )
            area_max = self.ODYLE_ORB_MAX_AREA_FRAC * frame_area
            cx0, cy0 = float(fw) * 0.5, float(fh) * 0.5

            hits: List[Dict[str, Any]] = []
            for cnt in cnts:
                area = float(cv2.contourArea(cnt))
                if area < area_min or area > area_max:
                    continue
                bx, by, bw, bh = cv2.boundingRect(cnt)
                if bw > self.ODYLE_ORB_MAX_BBOX_FRAC * fw or bh > self.ODYLE_ORB_MAX_BBOX_FRAC * fh:
                    continue
                aspect = float(bw) / float(max(1, bh))
                if not (0.5 <= aspect <= 2.0):
                    continue
                peri = float(cv2.arcLength(cnt, True))
                circularity = 4.0 * math.pi * area / (peri * peri + 1e-6)
                hull_area = float(cv2.contourArea(cv2.convexHull(cnt)))
                solidity = area / (hull_area + 1e-6)
                if circularity < self.ODYLE_ORB_CIRCULARITY_MIN:
                    continue
                if solidity < self.ODYLE_ORB_SOLIDITY_MIN:
                    continue

                # Blob-maskeli yama: yalnızca lekenin kendi pikselleri
                # kanıttır (bbox köşelerindeki arazi oyu çevirir).
                patch = bgr[by : by + bh, bx : bx + bw].copy()
                blob_in_patch = teal_mask[by : by + bh, bx : bx + bw]
                patch[blob_in_patch == 0] = 0
                try:
                    verdict = classify_3d_material_patch(patch)
                except Exception as exc:
                    logger.debug("Odyle yama sınıflandırması atlandı: %s", exc)
                    continue
                color_ratios = verdict.get("color_ratios", {}) or {}
                odyle_ratio = float(color_ratios.get("odyle_orb", 0.0) or 0.0)
                targena_ratio = float(color_ratios.get("targena_gold_green", 0.0) or 0.0)
                if verdict.get("unknown", True):
                    continue
                if str(verdict.get("chosen_name")) != "Odyle":
                    continue
                if odyle_ratio < self.ODYLE_ORB_MIN_RATIO:
                    continue
                if targena_ratio >= 0.15:
                    continue

                ix = float(bx) + float(bw) * 0.5
                iy = float(by) + float(bh) * 0.5
                dx = round(ix - cx0, 2)
                dy = round(iy - cy0, 2)
                dist_px = round(math.hypot(dx, dy), 2)
                conf_raw = verdict.get("confidence")
                try:
                    confidence = round(min(1.0, max(0.0, float(conf_raw))), 3)
                except (TypeError, ValueError):
                    confidence = 0.0

                hits.append(
                    {
                        "type": "odyle_air",
                        "category": "odyle_air",
                        "kind": "odyle_air",
                        "class_name": "odyle_air",
                        "dx": dx,
                        "dy": dy,
                        "dist_px": dist_px,
                        # 3D kürede mesafe ekran pikselidir (minimap px değil);
                        # tüketici kapıyı merkezi görüş-alanı olarak uygular.
                        "dist_basis": "screen_center_px",
                        "angle_deg": self._vector_to_compass_deg(dx, dy),
                        "pos": (int(round(ix)), int(round(iy))),
                        "screen_pos": (int(round(ix)), int(round(iy))),
                        "screen_x": int(round(ix)),
                        "screen_y": int(round(iy)),
                        "area": round(area, 1),
                        "circularity": round(circularity, 3),
                        "solidity": round(solidity, 3),
                        "confidence": confidence,
                        "odyle_orb_ratio": round(odyle_ratio, 4),
                        "targena_ratio": round(targena_ratio, 4),
                        "odyle_evidence": "orb_blob+taxonomy",
                        "is_hostile": False,
                        "is_odyle": True,
                        "is_gatherable": True,
                        "is_minimap": False,
                        "is_world_3d": True,
                        "in_air": True,
                    }
                )

            hits.sort(key=lambda d: float(d["dist_px"]))
            return hits
        except Exception as exc:
            logger.error("_scan_3d_odyle_orbs hatası: %s", exc)
            return []

    def scan_minimap_resources_and_odyle(
        self,
        frame: np.ndarray,
        check_odyle_air: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Minimap üzerindeki gerçek toplanabilir materyal ikonlarını (`sprout_material`) ve
        kırmızı düşman noktalarını (`hostile_dots`) tespit eder.
        Küçük yuvarlak beyaz canavar noktalarını (`•`) ve beyaz tüy (`minimap_feather_odyle.png`)
        ikonlarını kesin olarak filtreler.

        P2-10 (v19): `check_odyle_air=True` iken ayrıca 3D dünyadaki GERÇEK
        Odyle enerji küreleri taranır (`_scan_3d_odyle_orbs`) ve
        `is_odyle=True` / `type="odyle_air"` isabetleri döndürülür —
        minimap'te Odyle ikonu yoktur (katalog `shows_on_minimap=False`;
        tüy/kanat F-0413 ile ret), bu yüzden minimap yolu tek başına
        asla `is_odyle=True` üretemezdi. Küre isabetlerinde `dist_px`
        EKRAN merkezinden ekran pikselidir (`dist_basis="screen_center_px"`);
        tüketici kapıyı merkezi görüş-alanı olarak uygular.
        `check_odyle_air=False` ise havadaki Odyle isabetlerini eler
        (artık ölü kod değil — eskiden elenecek isabet hiç yoktu).
        """
        detections: List[Dict[str, Any]] = []
        try:
            res = self.scan_gatherable_materials(frame)
            detections.extend(res.get("hostile_dots", []))
            detections.extend(res.get("gatherable_nodes", []))
            if check_odyle_air:
                detections.extend(self._scan_3d_odyle_orbs(frame))
            if not check_odyle_air:
                # check_odyle_air=False: havadaki Odyle işaretli tespitler
                # elenir (False geçen çağıran "hava kontrolü yok" sanmamalı).
                detections = [
                    d
                    for d in detections
                    if not bool(d.get("is_odyle", False))
                    and d.get("type") not in ("odyle_air",)
                    and d.get("category") not in ("odyle_air",)
                ]
            detections.sort(key=lambda d: float(d.get("dist_px", 9999.0)))
            return detections
        except Exception as exc:
            logger.error("scan_minimap_resources_and_odyle hatası: %s", exc)
            return detections

    # ------------------------------------------------------------------
    # 6. Minimap Dokusu Kayması ile Hareket / Takılma (Stuck) Tespiti
    # ------------------------------------------------------------------
    def detect_movement_progress(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray,
    ) -> Tuple[bool, float]:
        """
        Minimap dokusunun ardışık kareler arasındaki kaymasını (`cv2.phaseCorrelate`
        ve yoğunluk farkı ile) ölçerek karakterin gerçekten yürüyüp yürümediğini
        veya bir duvara/kayaya takılıp takılmadığını (`is_moving: bool, speed_score: float`)
        tespit eder. Yoğunluk farkı eşiği `movement_diff_threshold` (varsayılan 2.2).
        """
        try:
            if prev_frame is None or curr_frame is None:
                return False, 0.0

            crop1, _ = self.extract_minimap(prev_frame)
            crop2, _ = self.extract_minimap(curr_frame)

            if crop1.shape != crop2.shape:
                crop2 = cv2.resize(crop2, (crop1.shape[1], crop1.shape[0]))

            ch, cw = crop1.shape[:2]
            top_pad = int(round(ch * 0.16))
            bot_pad = int(round(ch * 0.88))
            side_pad = int(round(cw * 0.08))

            g1 = cv2.cvtColor(crop1, cv2.COLOR_BGR2GRAY)[top_pad:bot_pad, side_pad : cw - side_pad]
            g2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2GRAY)[top_pad:bot_pad, side_pad : cw - side_pad]

            if g1.size == 0 or g2.size == 0:
                return False, 0.0

            f1 = g1.astype(np.float32)
            f2 = g2.astype(np.float32)

            ih, iw = f1.shape[:2]
            arrow_x = int(round(cw * self.PLAYER_CENTER_REL[0])) - side_pad
            arrow_y = int(round(ch * self.PLAYER_CENTER_REL[1])) - top_pad
            mean1 = float(np.mean(f1))
            mean2 = float(np.mean(f2))
            cv2.circle(f1, (arrow_x, arrow_y), 8, mean1, -1)
            cv2.circle(f2, (arrow_x, arrow_y), 8, mean2, -1)

            window = cv2.createHanningWindow((iw, ih), cv2.CV_32F)
            (shift_x, shift_y), _ = cv2.phaseCorrelate(f1, f2, window)
            shift_mag = float(math.hypot(shift_x, shift_y))

            mean_abs_diff = float(np.mean(np.abs(f1 - f2)))

            # P1-13: yoğunluk farkı eşiği kurucu parametresinden okunur
            # (movement_diff_threshold); önceden 2.2 gömülüydü.
            diff_contrib = max(0.0, (mean_abs_diff - self.movement_diff_threshold) * 0.35)
            speed_score = round(shift_mag + diff_contrib, 4)

            # OR mantığı kaldırıldı: minimaptaki hareketli canavar/NPC/
            # animasyon tek başına yoğunluk farkı üretip oyuncu duvara
            # takılıyken bile "hareket ediyor" dedirtiyordu. Gerçek
            # yürüyüş hem doku kayması hem yoğunluk değişimi ister.
            is_moving = bool(
                shift_mag >= self.movement_shift_threshold
                and mean_abs_diff >= self.movement_diff_threshold
            )
            return is_moving, speed_score

        except Exception as exc:
            logger.error("detect_movement_progress hatası: %s", exc)
            return False, 0.0

    def get_minimap_orientation_status(self, frame=None) -> dict:
        """
        Minimap oryantasyon koruması ve canavar dairesellik filtresi durumunu döndürür.
        """
        north_up_verified = False
        circularity_filter_ready = False
        reason = "not_verified"
        try:
            from core.patch_normalizer import MinimapOrientationGuard
            from core.vision_filters import MonsterCircularityFilter

            _filter = MonsterCircularityFilter()
            circularity_filter_ready = bool(
                hasattr(_filter, "is_monster_dot") and hasattr(_filter, "is_sprout_geometry")
            )
            guard = MinimapOrientationGuard()

            if frame is None:
                reason = "no_frame"
            elif hasattr(guard, "verify_north_up"):
                res = guard.verify_north_up(frame)
                if isinstance(res, dict):
                    north_up_verified = bool(res.get("is_fixed_north", res.get("north_up_verified", False)))
                elif isinstance(res, bool):
                    north_up_verified = bool(res)
                reason = "verified" if north_up_verified else "rotating_or_unverified"
            elif hasattr(guard, "verify_fixed_north_minimap"):
                res = guard.verify_fixed_north_minimap(frame)
                if isinstance(res, dict):
                    north_up_verified = bool(res.get("is_fixed_north", False))
                    reason = "verified" if north_up_verified else "rotating_or_unverified"
        except Exception as exc:
            logger.debug("get_minimap_orientation_status hatası: %s", exc)
            north_up_verified = False
            circularity_filter_ready = False
            reason = f"error: {exc}"

        return {
            "north_up_verified": bool(north_up_verified),
            "circularity_filter_ready": bool(circularity_filter_ready),
            "reason": reason,
        }

    def scan_minimap_with_strict_discrimination(
        self,
        minimap_bgr: np.ndarray,
    ) -> Dict[str, Any]:
        """
        `MinimapSproutMonsterDiscriminator` ile entegre çalışarak minimap kırpımındaki:
        1. Gerçek 3-yapraklı filizleri (`minimap_sprout`, class_id=9),
        2. Dairesel canavar noktalarını (`minimap_monster_dot`, class_id=11 - beyaz ve kırmızı düşman noktaları),
        3. HUD krom bölgelerini (`header` başlık, `distance_text` mesafe yazısı, `magnifier` büyüteç, `clock` saat)
        kesin olarak ayrıştırır.
        """
        empty_chrome = {
            "header": (0, 0, 0, 0),
            "distance_text": (0, 0, 0, 0),
            "magnifier": (0, 0, 0, 0),
            "clock": (0, 0, 0, 0),
            "player_arrow": (0, 0, 0, 0),
        }
        if minimap_bgr is None or not isinstance(minimap_bgr, np.ndarray) or minimap_bgr.size == 0:
            return {
                "minimap_sprout": [],
                "minimap_monster_dot": [],
                "gatherable_nodes": [],
                "monster_dots": [],
                "hostile_dots": [],
                "rejected_icons": [],
                "filtered_chrome_artifacts": [],
                "hud_chrome_regions": empty_chrome,
                "chrome_regions": empty_chrome,
                "subregion_info": {
                    "subregion": "Altgard",
                    "canonical_region": "Altgard",
                    "confidence": 0.0,
                    "source": "empty_input",
                },
                "player_center": (0.0, 0.0),
                "player_center_found": False,
                "discriminator_used": False,
                "strict_discrimination": True,
            }

        # Tam ekran kare verilmişse önce minimap kırpımını çıkar
        offset_xy = (0, 0)
        frame_shape = minimap_bgr.shape
        if minimap_bgr.shape[1] >= 600 and minimap_bgr.shape[0] >= 400:
            try:
                crop, (x1, y1, _, _) = self.extract_minimap(minimap_bgr)
                offset_xy = (x1, y1)
            except Exception:
                crop = minimap_bgr
        else:
            crop = minimap_bgr

        if crop.ndim == 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        elif crop.shape[2] == 4:
            crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2BGR)

        ch, cw = crop.shape[:2]
        center_x, center_y = self.get_minimap_player_center(crop.shape, crop_bgr=crop)
        arrow_rad = max(14, int(round(min(cw, ch) * 0.055)))

        # HUD krom bölgeleri (başlık, mesafe yazısı, büyüteç, saat, merkez oyuncu oku)
        header_rect = (0, 0, int(round(cw * 0.82)), int(round(ch * 0.16)))
        magnifier_rect = (int(round(cw * 0.82)), 0, cw, int(round(ch * 0.32)))
        distance_rect = (int(round(cw * 0.64)), int(round(ch * 0.76)), int(round(cw * 0.82)), ch)
        clock_rect = (int(round(cw * 0.82)), int(round(ch * 0.76)), cw, ch)
        player_arrow_rect = (
            max(0, int(round(center_x)) - arrow_rad),
            max(0, int(round(center_y)) - arrow_rad),
            min(cw, int(round(center_x)) + arrow_rad),
            min(ch, int(round(center_y)) + arrow_rad),
        )
        hud_chrome_regions: Dict[str, Tuple[int, int, int, int]] = {
            "header": header_rect,
            "distance_text": distance_rect,
            "magnifier": magnifier_rect,
            "clock": clock_rect,
            "player_arrow": player_arrow_rect,
        }

        def _in_chrome_region(px: float, py: float) -> Optional[str]:
            for r_name, (rx1, ry1, rx2, ry2) in hud_chrome_regions.items():
                if r_name == "player_arrow":
                    if math.hypot(px - center_x, py - center_y) <= float(arrow_rad):
                        return r_name
                elif rx1 <= px <= rx2 and ry1 <= py <= ry2:
                    return r_name
            return None

        # MinimapSproutMonsterDiscriminator entegrasyonu. Diskriminatörün
        # `discriminate_patch` diye bir metodu yoktur; gerçek API
        # `classify_patch(patch_bgr, candidate_center_xy, minimap_center_xy)`
        # -> 9 (filiz) / 11 (canavar) / None (red) döndürür.
        discriminator = _get_sprout_monster_discriminator()
        discriminator_available = discriminator is not None

        base_scan = self.scan_minimap_crop(crop, offset_xy=offset_xy, frame_shape=frame_shape)
        subregion_info = self.identify_subregion(crop)

        strict_sprouts: List[Dict[str, Any]] = []
        strict_monster_dots: List[Dict[str, Any]] = []
        filtered_chrome_artifacts: List[Dict[str, Any]] = []

        # 1. Aday filizleri (minimap_sprout) HUD krom maskesi ve MinimapSproutMonsterDiscriminator ile doğrula
        for node in base_scan.get("gatherable_nodes", []):
            px, py = node.get("pos", (0, 0))
            chrome_hit = _in_chrome_region(float(px), float(py))
            if chrome_hit is not None:
                filtered_chrome_artifacts.append(
                    {"pos": (int(px), int(py)), "chrome_region": chrome_hit, "original_kind": "sprout_candidate"}
                )
                continue

            rad = max(6, int(round(8.0 * (float(cw) / 332.0))))
            patch = crop[max(0, int(py) - rad) : min(ch, int(py) + rad + 1), max(0, int(px) - rad) : min(cw, int(px) + rad + 1)]
            if discriminator_available:
                disc_class = discriminator.classify_patch(
                    patch,
                    candidate_center_xy=(float(px), float(py)),
                    minimap_center_xy=(float(center_x), float(center_y)),
                )
                if disc_class == self.CLASS_ID_MONSTER_DOT:
                    m_copy = dict(node)
                    m_copy["class_id"] = self.CLASS_ID_MONSTER_DOT
                    m_copy["class_name"] = self.CLASS_NAME_MINIMAP_MONSTER
                    m_copy["kind"] = self.CLASS_NAME_MINIMAP_MONSTER
                    m_copy["is_gatherable"] = False
                    m_copy["discriminator_label"] = "minimap_monster_dot"
                    strict_monster_dots.append(m_copy)
                    continue
                if disc_class != self.CLASS_ID_SPROUT:
                    filtered_chrome_artifacts.append(
                        {"pos": (int(px), int(py)), "chrome_region": "discriminator_reject", "original_kind": "sprout_candidate"}
                    )
                    continue
                s_copy = dict(node)
                s_copy["discriminator_label"] = "minimap_sprout"
            else:
                # Diskriminatör yüklenemedi: aday zaten NCC>=0.88 +
                # kromatik/yapısal doğrulamadan geçti; bayraklı kabul edilir.
                s_copy = dict(node)
                s_copy["discriminator_label"] = "unavailable_ncc_only"
            s_copy["class_id"] = self.CLASS_ID_SPROUT
            s_copy["class_name"] = self.CLASS_NAME_SPROUT
            s_copy["three_leaf_verified"] = True
            strict_sprouts.append(s_copy)

        # 2. Dairesel canavar noktalarını (beyaz ve kırmızı) doğrula
        raw_monsters = list(base_scan.get("monster_dots", [])) + list(base_scan.get("hostile_dots", []))
        for mdot in raw_monsters:
            px, py = mdot.get("pos", (0, 0))
            chrome_hit = _in_chrome_region(float(px), float(py))
            if chrome_hit is not None:
                filtered_chrome_artifacts.append(
                    {"pos": (int(px), int(py)), "chrome_region": chrome_hit, "original_kind": "monster_candidate"}
                )
                continue

            m_copy = dict(mdot)
            m_copy["class_id"] = self.CLASS_ID_MONSTER_DOT
            m_copy["class_name"] = self.CLASS_NAME_MINIMAP_MONSTER
            m_copy["discriminator_label"] = "minimap_monster_dot"
            m_copy["is_gatherable"] = False
            strict_monster_dots.append(m_copy)

        strict_sprouts.sort(key=lambda d: float(d.get("dist_px", 9999.0)))
        strict_monster_dots.sort(key=lambda d: float(d.get("dist_px", 9999.0)))

        return {
            "minimap_sprout": strict_sprouts,
            "minimap_monster_dot": strict_monster_dots,
            "gatherable_nodes": strict_sprouts,
            "sprouts": strict_sprouts,
            "monster_dots": [m for m in strict_monster_dots if not m.get("is_hostile", False)],
            "hostile_dots": [m for m in strict_monster_dots if m.get("is_hostile", False)],
            "rejected_icons": list(base_scan.get("rejected_icons", [])),
            "filtered_chrome_artifacts": filtered_chrome_artifacts,
            "hud_chrome_regions": hud_chrome_regions,
            "chrome_regions": hud_chrome_regions,
            "subregion_info": subregion_info,
            "player_center": base_scan.get("player_center", (round(center_x, 2), round(center_y, 2))),
            "player_center_found": base_scan.get("player_center_found", True),
            "discriminator_used": bool(discriminator_available),
            "strict_discrimination": True,
        }

    def identify_subregion(
        self,
        frame_or_crop: np.ndarray,
        player_map_pos: Optional[Tuple[float, float]] = None,
    ) -> Dict[str, Any]:
        """
        Minimap üst başlık şeridi ve/veya oyuncu harita koordinatlarından
        bölge ve alt bölgeyi otomatik tespit eder ('Dredgion Crash Site' -> 'Altgard').
        """
        if frame_or_crop is None or not isinstance(frame_or_crop, np.ndarray) or frame_or_crop.size == 0:
            # H-sweep (dilim 13 K1): girdi yokken "Altgard" uydurulmaz —
            # bilinmiyor None'dur, confidence 0.0'dır.
            return {
                "subregion": None,
                "canonical_region": None,
                "confidence": 0.0,
                "source": "no_input",
            }

        # 1. Oyuncu koordinatları verilmişse mekansal sınırlayıcı kutu (bounding box) kontrolü
        if player_map_pos is not None and len(player_map_pos) >= 2:
            try:
                from core.route_manager import SUBREGION_BOUNDING_AREAS
                px, py = float(player_map_pos[0]), float(player_map_pos[1])
                for sub_key, sub_data in SUBREGION_BOUNDING_AREAS.items():
                    bbox = sub_data.get("bbox")
                    if bbox and len(bbox) == 4:
                        bx1, by1, bx2, by2 = bbox
                        if bx1 <= px <= bx2 and by1 <= py <= by2:
                            c_reg = sub_data.get("canonical", "Altgard")
                            title_name = " ".join(w.capitalize() for w in sub_key.split())
                            return {
                                "subregion": title_name,
                                "canonical_region": c_reg,
                                "confidence": 0.95,
                                "source": "spatial_bbox",
                            }
            except Exception as exc:
                logger.debug("identify_subregion spatial check skipped: %s", exc)

        # 2. Minimap kesitini çıkar veya hazır kesiti al
        crop = frame_or_crop
        if frame_or_crop.shape[1] > 500 and frame_or_crop.shape[0] > 300:
            try:
                crop, _ = self.extract_minimap(frame_or_crop)
            except Exception:
                crop = frame_or_crop

        ch, cw = crop.shape[:2]
        if ch < 24 or cw < 40:
            return {
                "subregion": None,
                "canonical_region": None,
                "confidence": 0.0,
                "source": "too_small",
            }

        # Üst başlık şeridi (y: 0..ch*0.22, x: 0..cw*0.98)
        top_h = max(16, int(round(ch * 0.22)))
        right_w = max(32, int(round(cw * 0.98)))
        left_w = 0
        header = crop[0:top_h, left_w:right_w]
        gray_strip = cv2.cvtColor(header, cv2.COLOR_BGR2GRAY) if header.ndim == 3 else header

        # Şeritteki metin piksel genişliğini tespit et
        p90 = np.percentile(gray_strip, 90) if gray_strip.size > 0 else 128
        _, th_img = cv2.threshold(gray_strip, max(120, int(p90 * 0.8)), 255, cv2.THRESH_BINARY)
        pts = cv2.findNonZero(th_img)
        text_w = 0
        if pts is not None:
            _, _, rw, _ = cv2.boundingRect(pts)
            if rw >= 15:
                text_w = rw

        candidates = [
            "Dredgion Crash Site",
            "Safe Haven",
            "Calderon Canyon",
            "Silent Hill",
            "Altgard Fortress",
            "Moslan Forest",
            "Mahki Swamp",
            "Gribbler Canyon",
            "Idun's Lake",
            "Altgard Ice Lake",
            "Basfelt Ruins",
            "Black Plains",
            "Heart of the Dragon",
            "Pilgrim's Ridge",
            "Altgard",
        ]

        best_score = -1.0
        best_name = "Altgard"

        # Şerit üzerinde aday metin şablonu eşleme
        for cand in candidates:
            cand_best = -1.0
            for scale in (0.40, 0.48, 0.55, 0.62):
                (tw, th), _ = cv2.getTextSize(cand, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                # Şablon (th+6, tw+6) boyutundadır; kesitten büyük şablon
                # cv2.matchTemplate'de çöker (önceki th+4/tw+4 kontrolü
                # 2px eksikti ve küçük kesitlerde çöküyordu).
                if th + 6 > gray_strip.shape[0] or tw + 6 > gray_strip.shape[1]:
                    continue
                tpl = np.zeros((th + 6, tw + 6), dtype=np.uint8)
                cv2.putText(tpl, cand, (3, th + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, 255, 1, cv2.LINE_AA)
                res = cv2.matchTemplate(gray_strip, tpl, cv2.TM_CCOEFF_NORMED)
                _, max_v, _, _ = cv2.minMaxLoc(res)
                v = float(max_v)
                if text_w > 0:
                    ratio = min(tw, text_w) / max(tw, text_w)
                    v = v * (0.6 + 0.4 * ratio)
                if v > cand_best:
                    cand_best = v
            if cand_best > best_score:
                best_score = cand_best
                best_name = cand

        # H-sweep (dilim 13 K1): kanonik bölge tek kaynaktan çözülür
        # (route_manager.SUBREGION_TO_CANONICAL); bilinmeyen ad None'dur.
        # Eşleşme yoksa en yüksek skorlu aday da isim olarak döndürülmez —
        # "no_match" bilinmiyor demektir.
        canonical = self._canonical_region_for(best_name)
        if best_score > 0.45:
            conf = float(np.clip(best_score, 0.50, 0.99))
            source = "header_template_ncc"
            out_name: Optional[str] = best_name
        else:
            # Hiçbir şablon eşleşmedi: uydurma 0.65 güven ve rastgele
            # aday adı yerine skora bağlı düşük güven + no_match kaynağı.
            conf = float(np.clip(best_score, 0.0, 0.45))
            source = "no_match"
            out_name = None
            canonical = None

        return {
            "subregion": out_name,
            "canonical_region": canonical,
            "confidence": round(conf, 3),
            "source": source,
        }

    def detect_subregion(
        self,
        frame_or_crop: np.ndarray,
        player_map_pos: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Minimap karesinden tespit edilen alt bölge adını döndürür.

        Bilinmiyorsa "" döner (eskiden "Altgard" uyduruyordu).
        """
        info = self.identify_subregion(frame_or_crop, player_map_pos)
        return str(info.get("subregion") or "")

    def classify_minimap_targets(
        self,
        crop_bgr: np.ndarray,
        offset_xy: Tuple[int, int] = (0, 0),
        frame_shape: Optional[Tuple[int, ...]] = None,
    ) -> Dict[str, Any]:
        """
        Minimap üzerindeki hedefleri 3 yapraklı filizler (`sprout_material`, class 9)
        ve kırmızı düşman/canavar noktaları (`hostile_dot`, class 11) olarak kesin bir
        şekilde ayrıştırır ve 0.075 oyuncu kleransı uygular.
        """
        scan_res = self.scan_minimap_crop(crop_bgr, offset_xy=offset_xy, frame_shape=frame_shape)
        sprouts = list(scan_res.get("gatherable_nodes", []))
        hostiles = list(scan_res.get("hostile_dots", []))
        monsters = list(scan_res.get("monster_dots", []))
        subregion_info = self.identify_subregion(crop_bgr)

        return {
            "sprouts": sprouts,
            "hostile_dots": hostiles,
            "monster_dots": monsters,
            "all_targets": sprouts + hostiles + monsters,
            "player_center": scan_res.get("player_center", (0.0, 0.0)),
            "player_clearance_norm": self.PLAYER_CLEARANCE_NORM,
            "player_clearance_px": scan_res.get("player_clearance_px", 0.0),
            # H-sweep (dilim 13 K1): bilinmeyen bölge "Altgard" diye
            # geçilmez; None tüketicide "bilinmiyor" demektir.
            "subregion": subregion_info.get("subregion"),
            "canonical_region": subregion_info.get("canonical_region"),
            "subregion_info": subregion_info,
        }

    def get_minimap_scanner_health_summary(self) -> dict:
        """Gerçek sağlık özeti: sözleşme duman testi çalıştırılır."""
        try:
            contract_ok = validate_minimap_scanner_contract()
            return {
                "minimap_scanner_ready": bool(contract_ok),
                "contract_valid": bool(contract_ok),
            }
        except Exception as exc:
            return {"minimap_scanner_ready": False, "error": str(exc)}

    def get_minimap_scanner_ready_flag(self) -> dict:
        # H-sweep (dilim 13 O3): eskiden ölçüm yapmadan {"ready": True}
        # dönüyordu. Artık gerçek sözleşme yoklamasına bağlanır.
        return {"module": "minimap_scanner", "ready": bool(validate_minimap_scanner_contract())}

    def get_minimap_scanner_version_info(self) -> dict:
        return {"module": "minimap_scanner", "version": __version__, "ready": bool(validate_minimap_scanner_contract())}

    def get_minimap_scanner_diagnostic_snapshot(self) -> dict:
        ready = bool(validate_minimap_scanner_contract())
        return {"module": "minimap_scanner", "status": "ok" if ready else "degraded", "ready": ready}


def get_minimap_scanner_capabilities() -> list:
    return ["minimap_scanner", "compass_tracking", "diagnostic_snapshot"]


def validate_minimap_scanner_contract() -> bool:
    """Gerçek sözleşme kontrolü: tarayıcı kurulur ve sentetik bir
    minimap kesiti üzerinde beklenen anahtarlarla tarama döner."""
    try:
        scanner = MinimapScanner()
        probe = np.zeros((120, 120, 3), dtype=np.uint8)
        res = scanner.scan_minimap_crop(probe)
        if not isinstance(res, dict):
            return False
        if "gatherable_nodes" not in res or "player_center" not in res:
            return False
        if not isinstance(res.get("player_center"), tuple):
            return False
        return True
    except Exception as exc:
        logger.debug("Minimap scanner sözleşme kontrolü başarısız: %s", exc)
        return False


def _minimap_scanner_baseline_metrics(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "contract_valid": bool(ok),
        "capability_count": len(get_minimap_scanner_capabilities()),
    }


def get_minimap_scanner_baseline_metrics() -> dict:
    return _minimap_scanner_baseline_metrics(validate_minimap_scanner_contract())


def _minimap_scanner_readiness_status(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "ready": bool(ok),
        "metrics": _minimap_scanner_baseline_metrics(ok),
    }


def get_minimap_scanner_readiness_status() -> dict:
    try:
        return _minimap_scanner_readiness_status(validate_minimap_scanner_contract())
    except Exception:
        return {
            "module": "minimap_scanner",
            "ready": False,
            "metrics": {},
        }


def _minimap_scanner_audit_record(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "audit_passed": bool(ok),
        "readiness": _minimap_scanner_readiness_status(ok),
    }


def get_minimap_scanner_audit_record() -> dict:
    return _minimap_scanner_audit_record(validate_minimap_scanner_contract())


def _minimap_scanner_operational_digest(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "operational": bool(ok),
        "audit": _minimap_scanner_audit_record(ok),
    }


def get_minimap_scanner_operational_digest() -> dict:
    return _minimap_scanner_operational_digest(validate_minimap_scanner_contract())


def _minimap_scanner_telemetry_payload(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "payload": _minimap_scanner_operational_digest(ok),
    }


def get_minimap_scanner_telemetry_payload() -> dict:
    return _minimap_scanner_telemetry_payload(validate_minimap_scanner_contract())


def _minimap_scanner_integrity_seal(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "sealed": bool(ok),
        "telemetry": _minimap_scanner_telemetry_payload(ok),
    }


def get_minimap_scanner_integrity_seal() -> dict:
    return _minimap_scanner_integrity_seal(validate_minimap_scanner_contract())


def _minimap_scanner_compliance_certificate(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "certified": bool(ok),
        "seal": _minimap_scanner_integrity_seal(ok),
    }


def get_minimap_scanner_compliance_certificate() -> dict:
    return _minimap_scanner_compliance_certificate(validate_minimap_scanner_contract())


def _minimap_scanner_readiness_probe(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "ready": bool(ok),
        "certificate": _minimap_scanner_compliance_certificate(ok),
    }


def get_minimap_scanner_readiness_probe() -> dict:
    return _minimap_scanner_readiness_probe(validate_minimap_scanner_contract())


def _minimap_scanner_audit_summary(ok: bool) -> dict:
    return {
        "module": "minimap_scanner",
        "audit_passed": bool(ok),
        "probe": _minimap_scanner_readiness_probe(ok),
    }


def get_minimap_scanner_audit_summary() -> dict:
    return _minimap_scanner_audit_summary(validate_minimap_scanner_contract())

