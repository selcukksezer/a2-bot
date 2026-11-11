# -*- coding: utf-8 -*-
"""
Aion 2 Odyle & Materyal Toplama Botu — Duman Testi Paketi Bölüm 1 (Madde 1–30)
===============================================================================
Kapsam:
  1. Bölüm (Madde 1–10): Görsel Algılama, Minimap Beyaz Tüy Reddi (`FeatherNegativeFilter`),
     5-Ölçekli 3B Materyal Arması (`Pyramid3DCrestDetector`), Dinamik `[F]`/`[E]` Tuş
     Okuyucu (`DynamicInteractionKeyReader`), Yuvarlak Canavar Noktası (`•`) Dairesellik
     Filtresi (`MonsterCircularityFilter`), HSV İsim Plakası İzolatörü (`NameplateHSVIsolator`)
     & 100+ Materyal DB (`AION2_MATERIALS_CATALOG`), Sabit Kuzey Minimap Koruması
     (`MinimapOrientationGuard`), Sobel+CLAHE Yama Normalizasyonu (`SobelClahePatchNormalizer`),
     2560x1440 ROI Kırpıcı (`HighResROICropper`), Watershed Küme Ayırıcı
     (`WatershedClusterSeparator`) ve Laplacian Hareket Bulanıklığı Kapısı (`MotionBlurGate`).
     (`reference_screenshots/` altındaki tüm 7 gerçek oyun ekran görüntüsü dahil).
  2. Bölüm (Madde 11–20): Kararlılık & Log Hata Düzeltmeleri; `(1280, 720)` 2K Kamera
     Merkez Yakalama İstisnası (`CameraLockExceptionRule` & `PhysicalInputGuard`),
     `GetCursorInfo` İmleç Görünürlük Denetimi (`CursorVisibilityAutomator`), 5-Kare +
     Gri Ekran + Dirilme Penceresi Ölüm Koruması (`FiveFrameDeathConfirmationGuard`),
     Çoklu Kriterli (`>= 2/3`) Harita Açılış Doğrulayıcı (`MultiCriteriaMapVerifier`),
     250ms Öz-Girdi Beyaz Listesi (`SelfInputWhitelistWindow`), UAC Yönetici Yükseltici
     (`uac_elevator`), Watchdog `PAUSED` Dondurma Koruması (`BotWatchdog`), `< 100ms`
     Sohbet Sistem Uyarı Ayrıştırıcı (`GameHUDReader.detect_chat_system_alerts` &
     `parse_chat_feedback_fast`), Üst Hedef Barı (`detect_top_target_bar`) ve
     Cast/Vital Bar Okuyucu (`read_cast_bar_progress` & `read_vital_bars`).
  3. Bölüm (Madde 21–30): 3x Hasat Döngüsü (`1/3 -> 2/3 -> 3/3`, `ThreeStageHarvestController`),
     Yarım Kaynak (`PartialNodeDetector`) 400ms Erken Tükenme, `"Essence Extraction failed"`
     Hak Düşmeme ve Yeniden Deneme (`handle_extraction_failure`), `> 35m` Otomatik Binek
     (`AutoMountManager`), `< 40m` Haritasız Minimap Mikro-Navigasyonu (`MaplessMicroNavigator`),
     `cv2.phaseCorrelate` Sıkışma Kurtarma (`PhaseCorrelateUnstuckManeuver`), Z-Ekseni
     Uçurum Filtresi (`ZAxisCliffFilter`), Dinlenme Süresi Tahminleyici Öngörülü Ön-Yürüyüş
     (`PredictivePreWalkScheduler`), EMA Düğüm Koordinat Kalibrasyonu (`CentroidDriftCorrector`)
     ve Rakip Oyuncu (`ContestedNodeDetector`) Atlama Mekanizması.

Çalıştırma:
    python tools/smoke_test_vision_and_3x.py [--only 7,9] [--list]
                                            [--json-out sonuc.json]
                                            [--timeout 120 | --no-timeout]
                                            [--self-test]

    --only: yalnızca verilen maddeleri koşturur (hızlı tek-madde yineleme);
            rapor, seçmeli koşuyu açıkça belirtir. --list: madde listesini
            yazdırıp çıkar. --json-out: sonuçları JSON dosyaya yazar.
            Her maddenin varsayılan 120sn zaman aşımı vardır (takılan madde
            tüm paketi kilitlemez; POSIX/SIGALRM yoksa uygulanmaz).
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from version import __version__

# Proje kök dizinini sys.path'e ekle ve çalışma dizinini sabitle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
try:
    os.chdir(PROJECT_ROOT)
except OSError as _chdir_err:
    _ = _chdir_err

# UTF-8 konsol çıktısı (Windows PowerShell / CMD Türkçe karakter uyumu)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception as _reconf_err:
        _ = _reconf_err

from config.materials_db import (
    AION2_MATERIALS_CATALOG,
    MATERIAL_CATEGORIES,
    REJECTED_NON_MATERIAL_KEYWORDS,
    classify_3d_material_patch,
    get_material_spec,
    get_region_expected_materials,
    is_rejected_non_material,
)
from core.corridor_navigator import (
    CentroidDriftCorrector,
    ContestedNodeDetector,
    CorridorNavigator,
    PhaseCorrelateUnstuckManeuver,
    PredictivePreWalkScheduler,
    ZAxisCliffFilter,
    is_node_contested,
    refine_node_coordinates_on_gather,
)
from core.input_driver import InputDriver
from core.material_scanner import MaterialScanner
from core.minimap_scanner import MinimapScanner
from core.navigation_intel import (
    AutoMountManager,
    MaplessMicroNavigator,
    PartialNodeDetector,
    ThreeStageHarvestController,
    handle_extraction_failure,
)
from core.ocr_distance import GameHUDReader
from core.patch_normalizer import (
    HighResROICropper,
    MinimapOrientationGuard,
    MotionBlurGate,
    SobelClahePatchNormalizer,
    WatershedClusterSeparator,
)
from core.route_manager import NodeData, RouteManager
from core.sequential_router import SequentialRoutePlanner
from core.stability_guards import (
    CameraLockExceptionRule,
    CursorVisibilityAutomator,
    FiveFrameDeathConfirmationGuard,
    MultiCriteriaMapVerifier,
    SelfInputWhitelistWindow,
)
from core.uac_elevator import (
    LAST_ELEVATION_REQUEST,
    build_elevation_payload,
    ensure_admin_elevation,
    is_admin,
    resolve_venv_python,
)
from core.vision import VisionEngine
from core.watchdog import BotWatchdog
from core.windmouse_humanizer import PhysicalInputGuard

# core/vision_filters.py alt modülü yüklemesi
VISION_FILTERS_MODULE: Optional[Any] = None
if importlib.util.find_spec("core.vision_filters") is not None:
    try:
        VISION_FILTERS_MODULE = importlib.import_module("core.vision_filters")
    except Exception:
        VISION_FILTERS_MODULE = None


REFERENCE_SCREENSHOT_NAMES: Tuple[str, ...] = (
    "aria_3d_crest_far.jpg",
    "aria_f_gather_not_on_minimap.jpg",
    "mela_f_gather_not_on_minimap.jpg",
    "minimap_filter_closeup.jpg",
    "minimap_sprout_closeup.jpg",
    "pilgrims_ridge_1.jpg",
    "pilgrims_ridge_2.jpg",
)


def _load_image_unicode_safe(path: str) -> Optional[np.ndarray]:
    """Türkçe karakter ('ç', 'ğ', 'ş' vb.) içeren dosya yollarından güvenli BGR görüntü yükler."""
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _load_all_reference_screenshots() -> Dict[str, np.ndarray]:
    """`reference_screenshots/` dizinindeki 7 gerçek oyun ekran görüntüsünü belleğe yükler.

    Eksik/bozuk bir referans ekran görüntüsü sessizce yutulmaz: hangi dosyaların
    yüklenemediğini listeleyen `FileNotFoundError` fırlatılır (F-1279).
    """
    screenshots_dir = os.path.join(PROJECT_ROOT, "reference_screenshots")
    loaded: Dict[str, np.ndarray] = {}
    missing: List[str] = []
    for fname in REFERENCE_SCREENSHOT_NAMES:
        fpath = os.path.join(screenshots_dir, fname)
        img = _load_image_unicode_safe(fpath)
        if img is not None and img.size > 0:
            loaded[fname] = img
        else:
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"Eksik/okunamayan referans ekran görüntüsü ({len(missing)}): {missing} "
            f"(dizin: {screenshots_dir})"
        )
    return loaded


class _MockInputRecorder:
    """Birim/Duman testlerinde donanım girdisi göndermeden tuş ve fare çağrılarını kaydeder."""

    def __init__(self) -> None:
        self.pressed_keys: List[str] = []
        self.mouse_moves: List[Tuple[int, int]] = []
        self.key_downs: List[str] = []
        self.key_ups: List[str] = []
        self.realigned_angles: List[Optional[float]] = []

    def press_key(self, key: str, duration: float = 0.05) -> bool:
        self.pressed_keys.append(str(key).lower())
        return True

    def tap_key(self, key: str) -> bool:
        self.pressed_keys.append(str(key).lower())
        return True

    def send_key(self, key: str) -> bool:
        self.pressed_keys.append(str(key).lower())
        return True

    def key_down(self, key: str) -> bool:
        self.key_downs.append(str(key).lower())
        return True

    def key_up(self, key: str) -> bool:
        self.key_ups.append(str(key).lower())
        return True

    def move_mouse(self, x: int, y: int) -> bool:
        self.mouse_moves.append((int(x), int(y)))
        return True

    def smooth_move_to(self, x: int, y: int) -> bool:
        self.mouse_moves.append((int(x), int(y)))
        return True

    def realign_to_target(self, angle_deg: Optional[float]) -> bool:
        self.realigned_angles.append(angle_deg)
        return True


def _build_temp_route_manager() -> Tuple[RouteManager, str]:
    """Geçici dizin üzerinde izole bir `RouteManager` örneği oluşturur."""
    temp_dir = tempfile.mkdtemp(prefix="aion2_smoke_routes_")
    rm = RouteManager(base_dir=temp_dir, default_rest_sec=300)
    rm.create_region("Pilgrim's Ridge")
    rm.active_region = "Pilgrim's Ridge"
    return rm, temp_dir


def _cleanup_temp_dir(path: str) -> None:
    """Geçici test dizinini güvenli biçimde siler."""
    if path and os.path.exists(path):
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError as _rm_err:
            _ = _rm_err


# ==============================================================================
# BÖLÜM 1 (MADDE 1–10): GÖRSEL ALGILAMA, FİLTRELER VE YAMA NORMALİZASYONU
# ==============================================================================


def _test_item_01_minimap_feather_and_sprout(
    screenshots: Dict[str, np.ndarray],
    mm_scanner: MinimapScanner,
    vision_engine: VisionEngine,
) -> str:
    """
    Madde 1: Minimap Beyaz Tüy (`minimap_filter_closeup.jpg`) %100 Reddi ve
    3-Yapraklı Beyaz Filiz (`minimap_sprout_closeup.jpg`) Tespiti.
    """
    assert len(screenshots) == 7, f"7 referans ekran görüntüsü bekleniyordu, bulunan: {len(screenshots)}"
    feather_img = screenshots["minimap_filter_closeup.jpg"]
    sprout_img = screenshots["minimap_sprout_closeup.jpg"]

    # 1) minimap_filter_closeup.jpg üzerindeki beyaz tüy (feather) kesinlikle reddedilmeli
    feather_scan = mm_scanner.scan_gatherable_materials(feather_img)
    feather_odyle = mm_scanner.scan_odyle_dots(feather_img)
    filter_ui = vision_engine.detect_minimap_filter_settings(feather_img)

    rejected_kinds = [str(r.get("kind", "")) for r in feather_scan.get("rejected_icons", [])]
    # F-1169: ret koşulu OR ile gevşetilemez — tüyün reddedilenler listesinde
    # açıkça 'rejected_feather' olarak görünmesi zorunludur; yedek (fallback)
    # bir UI bayrağı ile geçmek yasaktır.
    has_rejected_feather = "rejected_feather" in rejected_kinds
    assert len(feather_odyle) == 0, f"Beyaz tüy görüntüsünde yanlış pozitif filiz bulundu: {feather_odyle}"
    assert has_rejected_feather, (
        f"Beyaz tüy ikonu reddedilenler listesinde görünmedi: {rejected_kinds} / {feather_scan}"
    )
    assert isinstance(filter_ui, dict) and filter_ui.get("is_open") is True, (
        f"Minimap filtre paneli doğrulanamadı: {filter_ui}"
    )

    # 2) minimap_sprout_closeup.jpg üzerindeki 3-yapraklı beyaz filiz (sprout) kesinlikle tespit edilmeli
    sprout_scan = mm_scanner.scan_gatherable_materials(sprout_img)
    sprout_odyle = mm_scanner.scan_odyle_dots(sprout_img)
    gatherable_nodes = sprout_scan.get("gatherable_nodes", [])
    assert len(gatherable_nodes) >= 1 and len(sprout_odyle) >= 1, (
        f"3-yapraklı filiz tespit edilemedi: gatherable={len(gatherable_nodes)}, odyle={len(sprout_odyle)}"
    )
    assert all(n.get("kind") == "sprout_material" for n in gatherable_nodes), (
        f"Filiz harici düğüm sızdı: {gatherable_nodes}"
    )

    # 3) FeatherNegativeFilter (core/vision_filters.py) doğrulaması
    # F-1170: modül/filtre yoksa bu adım sessizce atlanamaz — yokluğu testin
    # başarısızlığıdır.
    assert VISION_FILTERS_MODULE is not None, "core.vision_filters modülü yüklenemedi"
    assert hasattr(VISION_FILTERS_MODULE, "FeatherNegativeFilter"), (
        "FeatherNegativeFilter core/vision_filters.py içinde bulunamadı"
    )
    fn_filter = VISION_FILTERS_MODULE.FeatherNegativeFilter()
    assert fn_filter.feather_template is not None and fn_filter.reject_mask is not None
    assert fn_filter.is_feather_icon( fn_filter.feather_template ) is True
    assert fn_filter.sprout_templates, "FeatherNegativeFilter sprout şablonu içermiyor"
    assert fn_filter.is_feather_icon(fn_filter.sprout_templates[0]) is False
    vf_info = " + FeatherNegativeFilter(Hu+NCC)=Doğrulandı"

    return (
        f"Tüy reddedildi (odyle=0, rejected={len(feather_scan.get('rejected_icons', []))}); "
        f"3-yapraklı filiz yakalandı (sprout={len(gatherable_nodes)}){vf_info}"
    )


def _test_item_02_five_scale_3d_crest_pyramid(
    screenshots: Dict[str, np.ndarray],
    mat_scanner: MaterialScanner,
    vision_engine: VisionEngine,
) -> str:
    """
    Madde 2: 5-Ölçekli (`0.6, 0.8, 1.0, 1.2, 1.4`) 3B Materyal Arması (`3D Gather Crest`)
    Piramidi; Minimap'te görünmeyen `Aria` ve `Mela` 3B armalarının tespiti.
    """
    aria_far = screenshots["aria_3d_crest_far.jpg"]
    aria_f = screenshots["aria_f_gather_not_on_minimap.jpg"]
    mela_f = screenshots["mela_f_gather_not_on_minimap.jpg"]

    # Pyramid3DCrestDetector (core/vision_filters.py) + MaterialScanner + VisionEngine
    # F-1171: ölçek listesi testten önce varsayılan atanamazdı — dedektör yoksa
    # varsayılan (0.6..1.4) ile geçmek yasaktır; ölçekler dedektörden okunur.
    assert VISION_FILTERS_MODULE is not None, "core.vision_filters modülü yüklenemedi"
    assert hasattr(VISION_FILTERS_MODULE, "Pyramid3DCrestDetector"), (
        "Pyramid3DCrestDetector core/vision_filters.py içinde bulunamadı"
    )
    pyr_det = VISION_FILTERS_MODULE.Pyramid3DCrestDetector()
    scales = tuple(float(s) for s in pyr_det.scales)
    assert len(scales) == 5 and abs(scales[0] - 0.6) < 1e-4 and abs(scales[-1] - 1.4) < 1e-4, (
        f"Pyramid3DCrestDetector 5-ölçekli piramit (0.6..1.4) eksik: {scales}"
    )
    pyr_hits = pyr_det.detect(aria_far)
    assert len(pyr_hits) >= 1, "Pyramid3DCrestDetector aria_3d_crest_far.jpg üzerinde arma bulamadı!"
    assert 5.0 <= float(pyr_hits[0]["estimated_dist_m"]) <= 40.0

    # F-1242: özel (private) _scan_3d_gather_crests yerine herkese açık
    # scan_frame_for_materials kullanılır; arma, Aria tespitleriyle kanıtlanır.
    public_dets = mat_scanner.scan_frame_for_materials(aria_far)
    crests_far = [
        m for m in public_dets
        if "Aria" in str(getattr(m, "material_type", getattr(m, "material_name", "")))
    ]
    vision_crests_far = vision_engine.detect_3d_gather_crests(aria_far)
    assert len(crests_far) >= 1 and len(vision_crests_far) >= 1, (
        "aria_3d_crest_far.jpg üzerinde 3B materyal arması bulunamadı!"
    )

    aria_all = mat_scanner.scan_frame_for_materials(aria_f)
    mela_all = mat_scanner.scan_frame_for_materials(mela_f)
    assert len(aria_all) >= 1, "aria_f_gather_not_on_minimap.jpg üzerinde Aria materyali tespit edilemedi!"
    assert len(mela_all) >= 1, "mela_f_gather_not_on_minimap.jpg üzerinde Mela materyali tespit edilemedi!"

    aria_names = [str(getattr(m, "material_type", getattr(m, "material_name", ""))) for m in aria_all]
    mela_names = [str(getattr(m, "material_type", getattr(m, "material_name", ""))) for m in mela_all]
    assert any("Aria" in n for n in aria_names), f"Aria adı eşleşmedi, bulunanlar: {aria_names}"
    assert any("Mela" in n for n in mela_names), f"Mela adı eşleşmedi, bulunanlar: {mela_names}"

    return (
        f"5 ölçek {tuple(scales)} doğrulandı; Uzak 3B arma={len(crests_far)}, "
        f"Minimap dışı Aria ({aria_names[0]}) & Mela ({mela_names[0]}) yakalandı"
    )


def _test_item_03_dynamic_f_and_e_gather_key_reader(
    screenshots: Dict[str, np.ndarray],
    mat_scanner: MaterialScanner,
    vision_engine: VisionEngine,
) -> str:
    """
    Madde 3: Dinamik `[F Gather]` ve `[E]` Etkileşim Tuşu Okuyucu
    (`DynamicInteractionKeyReader`, `MaterialScanner`, `VisionEngine`).
    """
    aria_f = screenshots["aria_f_gather_not_on_minimap.jpg"]
    mela_f = screenshots["mela_f_gather_not_on_minimap.jpg"]
    pr1 = screenshots["pilgrims_ridge_1.jpg"]

    # 1) MaterialScanner & VisionEngine üzerinde gerçek ekran görüntüleri
    # F-1172: last_detected_gather_key bayat varsayılanla ("f"/"e") geçemez —
    # her okumadan önce nöbetçi (sentinel) değere sıfırlanır; detect çağrısı
    # tuşu gerçekten yazmazsa assert düşer.
    assert hasattr(vision_engine, "last_detected_gather_key"), (
        "VisionEngine.last_detected_gather_key özniteliği yok"
    )
    vision_engine.last_detected_gather_key = ""
    det_aria_list = mat_scanner.scan_gather_prompt(aria_f)
    vis_aria = vision_engine.detect_gather_prompt(aria_f)
    vis_aria_ok = bool(vis_aria[0]) if isinstance(vis_aria, tuple) else bool(vis_aria)
    key_aria_vis = vision_engine.last_detected_gather_key
    assert len(det_aria_list) >= 1 and vis_aria_ok is True and key_aria_vis == "f", (
        f"aria_f_gather_not_on_minimap.jpg [F] istemi okunamadı: {det_aria_list}, vis={vis_aria}"
    )

    vision_engine.last_detected_gather_key = ""
    det_mela_list = mat_scanner.scan_gather_prompt(mela_f)
    vis_mela = vision_engine.detect_gather_prompt(mela_f)
    vis_mela_ok = bool(vis_mela[0]) if isinstance(vis_mela, tuple) else bool(vis_mela)
    key_mela_vis = vision_engine.last_detected_gather_key
    assert len(det_mela_list) >= 1 and vis_mela_ok is True and key_mela_vis == "f", (
        f"mela_f_gather_not_on_minimap.jpg [F] istemi okunamadı: {det_mela_list}, vis={vis_mela}"
    )

    # 2) pilgrims_ridge_1.jpg üzerindeki yüzen [E] kutusu ('e' tuşu) tespiti
    # F-1242: özel _scan_floating_gather_boxes yerine herkese açık
    # scan_frame_for_materials kullanılır (kaynak: '3d_floating_icon').
    public_pr1 = mat_scanner.scan_frame_for_materials(pr1)
    e_boxes = [m for m in public_pr1 if str(getattr(m, "source", "")) == "3d_floating_icon"]
    vision_engine.last_detected_gather_key = ""
    vis_pr1 = vision_engine.detect_gather_prompt(pr1)
    vis_pr1_ok = bool(vis_pr1[0]) if isinstance(vis_pr1, tuple) else bool(vis_pr1)
    key_pr1_vis = vision_engine.last_detected_gather_key
    assert len(e_boxes) >= 1 and vis_pr1_ok is True and key_pr1_vis == "e", (
        f"pilgrims_ridge_1.jpg üzerinde yüzen [E] kutusu bulunamadı: {e_boxes}, key={key_pr1_vis}"
    )

    # 3) DynamicInteractionKeyReader (core/vision_filters.py) doğrudan doğrulaması
    # Yoksa sessizce atlanamaz.
    assert VISION_FILTERS_MODULE is not None, "core.vision_filters modülü yüklenemedi"
    assert hasattr(VISION_FILTERS_MODULE, "DynamicInteractionKeyReader"), (
        "DynamicInteractionKeyReader core/vision_filters.py içinde bulunamadı"
    )
    key_reader = VISION_FILTERS_MODULE.DynamicInteractionKeyReader()
    k_aria = key_reader.decide_interaction_key(aria_f)
    k_mela = key_reader.decide_interaction_key(mela_f)
    k_pr1 = key_reader.decide_interaction_key(pr1)
    assert k_aria == "f" and k_mela == "f" and k_pr1 == "e", (
        f"DynamicInteractionKeyReader hatalı tuş döndürdü: aria={k_aria}, mela={k_mela}, pr1={k_pr1}"
    )

    return (
        f"Aria=[{key_aria_vis.upper()}] (güven={det_aria_list[0].confidence:.2f}), "
        f"Mela=[{key_mela_vis.upper()}] (güven={det_mela_list[0].confidence:.2f}), "
        f"PilgrimsRidge1=[{key_pr1_vis.upper()}] ({e_boxes[0].material_type}) doğrulandı"
    )


def _test_item_04_monster_circularity_filter(
    screenshots: Dict[str, np.ndarray],
    mm_scanner: MinimapScanner,
) -> str:
    """
    Madde 4: Minimap Yuvarlak Beyaz Canavar Noktası (`•`) Dairesellik (`circularity >= 0.82`)
    Filtresi (`MonsterCircularityFilter` & `MinimapScanner`).
    """
    dot_cnt = np.array([[[16, 11]], [[21, 16]], [[16, 21]], [[11, 16]]], dtype=np.int32)
    sprout_cnt = np.array([[[16, 6]], [[23, 20]], [[16, 13]], [[9, 20]]], dtype=np.int32)

    # S-4601: sınıf yoksa sentetik geometri assert'leri sessizce atlanamaz —
    # yokluğu testin başarısızlığıdır (Madde 1'deki sert assert deseni).
    assert VISION_FILTERS_MODULE is not None, "core.vision_filters modülü yüklenemedi"
    assert hasattr(VISION_FILTERS_MODULE, "MonsterCircularityFilter"), (
        "MonsterCircularityFilter core/vision_filters.py içinde bulunamadı"
    )
    mc_filter = VISION_FILTERS_MODULE.MonsterCircularityFilter()
    dot_metrics = mc_filter.compute_geometry_metrics(dot_cnt)
    sprout_metrics = mc_filter.compute_geometry_metrics(sprout_cnt)
    assert mc_filter.is_monster_dot(dot_cnt) is True, f"Yuvarlak nokta canavar olarak tanınmadı: {dot_metrics}"
    assert mc_filter.is_sprout_geometry(dot_cnt) is False
    assert mc_filter.is_monster_dot(sprout_cnt) is False, f"Filiz konturu canavar sanıldı: {sprout_metrics}"
    assert mc_filter.is_sprout_geometry(sprout_cnt) is True, f"Filiz konturu reddedildi: {sprout_metrics}"

    # Gerçek ekran görüntülerinde (pilgrims_ridge_1.jpg & pilgrims_ridge_2.jpg) canavar noktalarının
    # asla 'sprout_material' listesine karışmadığını doğrula
    total_monsters = 0
    for name in ("pilgrims_ridge_1.jpg", "pilgrims_ridge_2.jpg"):
        scan = mm_scanner.scan_gatherable_materials(screenshots[name])
        gatherables = scan.get("gatherable_nodes", [])
        monster_dots = scan.get("monster_dots", [])
        # F-1173: canavar sayısı doğrulanmadan "%100 dışlandı" iddiası boş
        # geçemez — her karede en az 1 canavar noktası bulunması zorunludur.
        assert len(monster_dots) >= 1, (
            f"{name}: hiç canavar noktası bulunamadı, dışlama iddiası boş geçemez: {scan}"
        )
        total_monsters += len(monster_dots)
        for g in gatherables:
            assert g.get("kind") == "sprout_material", f"{name}: Gatherable içine yabancı tür sızdı: {g}"
        for m in monster_dots:
            # S-4602: başlıktaki eşik (circularity >= 0.82) ile tutarlı — bu aynı
            # zamanda MonsterCircularityFilter.circularity_threshold varsayılanıdır;
            # gerçek karelerdeki canavar noktaları 0.857–0.923 aralığındadır.
            assert float(m.get("circularity", 0.0)) >= 0.82, f"{name}: Monster daireselliği düşük: {m}"

    return (
        f"Yuvarlak canavar noktası (•) is_monster_dot=True, 3-yapraklı filiz is_sprout_geometry=True; "
        f"PR1+PR2 üzerinde {total_monsters} canavar noktası gatherable listesinden %100 dışlandı"
    )


def _test_item_05_hsv_nameplate_and_100plus_materials_db(
    screenshots: Dict[str, np.ndarray],
    mat_scanner: MaterialScanner,
) -> str:
    """
    Madde 5: 3B Dünya İsim Plakası (`NameplateHSVIsolator`) & 100+ Aion 2 Materyal Veritabanı.
    """
    catalog_size = len(AION2_MATERIALS_CATALOG)
    assert catalog_size >= 100, f"Materyal veritabanı en az 100 öğe içermeli, mevcut: {catalog_size}"
    assert len(MATERIAL_CATEGORIES) == 6, f"6 ana materyal kategorisi bekleniyordu: {MATERIAL_CATEGORIES}"

    categories = {spec.category for spec in AION2_MATERIALS_CATALOG.values()}
    assert len(categories) == 6, f"Katalogda 6 kategori aktif olmalı, bulunan: {categories}"

    # Temel Aion 2 materyallerinin varlığı
    for req_name in (
        "Aria",
        "Fresh Aria",
        "Mela",
        "Fresh Mela",
        "Odyle",
        "Pure Odyle",
        "Orichalcum Ore",
        "Targena",
        "Azpha",
        "Yggdrasil Log",
        "Sapphire Gemstone",
    ):
        spec = get_material_spec(req_name)
        assert spec is not None, f"Kritik materyal veritabanında bulunamadı: {req_name}"

    # Bölge bazlı materyal beklentisi ve materyal olmayan tüy/canavar reddi
    pr_mats = get_region_expected_materials("Pilgrim's Ridge")
    assert len(pr_mats) >= 5, f"Pilgrim's Ridge bölge materyalleri eksik: {len(pr_mats)}"
    assert is_rejected_non_material("feather") is True
    assert is_rejected_non_material("monster_white_dot") is True
    assert is_rejected_non_material("Aria") is False
    # [O10] D52/F-1101 kelime-sınırı koruması: "mobile" içinde "mob",
    # "request" içinde "quest" eşleşmemeli.
    assert is_rejected_non_material("mobile") is False
    assert is_rejected_non_material("request") is False
    assert len(REJECTED_NON_MATERIAL_KEYWORDS) >= 8

    # NameplateHSVIsolator (core/vision_filters.py) doğrulaması
    # S-4601: sınıf yoksa sentetik isim plakası assert'i sessizce atlanamaz —
    # yokluğu testin başarısızlığıdır (Madde 1'deki sert assert deseni).
    assert VISION_FILTERS_MODULE is not None, "core.vision_filters modülü yüklenemedi"
    assert hasattr(VISION_FILTERS_MODULE, "NameplateHSVIsolator"), (
        "NameplateHSVIsolator core/vision_filters.py içinde bulunamadı"
    )
    isolator = VISION_FILTERS_MODULE.NameplateHSVIsolator()
    # Sentetik koyu gölgeli turkuaz/yeşil '‹ Aria ›' isim plakası
    np_canvas = np.full((36, 120, 3), 25, dtype=np.uint8)
    cv2.putText(np_canvas, "< Aria >", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 235, 110), 1)
    np_res = isolator.isolate_gatherable_nameplate(np_canvas)
    assert np_res["is_gatherable_nameplate"] is True and np_res["is_monster_nameplate"] is False

    # 3B yama sınıflandırma ve gerçek ekran görüntüsü isim plakası taraması
    dummy_patch = np.zeros((32, 32, 3), dtype=np.uint8)
    dummy_patch[:, :] = (180, 220, 120)
    matched_info = classify_3d_material_patch(dummy_patch, region_name="Pilgrim's Ridge")
    assert isinstance(matched_info, dict) and bool(matched_info.get("name"))

    nameplates_aria = mat_scanner.scan_3d_world_nameplates(screenshots["aria_f_gather_not_on_minimap.jpg"])
    # F-1174: boş liste ile geçmek yasaktır — Aria karesinde en az 1 isim
    # plakası bulunması zorunludur.
    assert isinstance(nameplates_aria, list) and len(nameplates_aria) >= 1, (
        f"aria_f_gather_not_on_minimap.jpg üzerinde isim plakası bulunamadı: {nameplates_aria}"
    )

    return (
        f"Katalog={catalog_size} materyal (6 kategori), NameplateHSVIsolator=Doğrulandı, "
        f"Pilgrim's Ridge={len(pr_mats)} bölge materyali"
    )


def _test_item_06_fixed_north_minimap_orientation_guard(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 6: Sabit Kuzey (`Fixed-North`) Minimap Oryantasyon Koruması (`MinimapOrientationGuard`).
    """
    guard = MinimapOrientationGuard()
    pr1 = screenshots["pilgrims_ridge_1.jpg"]
    pr2 = screenshots["pilgrims_ridge_2.jpg"]

    res1 = guard.verify_fixed_north_minimap(pr1)
    res2 = guard.verify_fixed_north_minimap(pr2)
    assert isinstance(res1, dict) and "is_fixed_north" in res1 and "confidence" in res1
    assert bool(res1["is_fixed_north"]) is True, f"pilgrims_ridge_1.jpg Sabit Kuzey doğrulanamadı: {res1}"
    assert bool(res2["is_fixed_north"]) is True, f"pilgrims_ridge_2.jpg Sabit Kuzey doğrulanamadı: {res2}"

    # 80x80 düğüm yaması Kuzey-Yukarı (North-Up) normalizasyon testi
    # F-1281: yama dilimi sabit piksel varsayamaz — kare boyutundan göreli
    # hesaplanır (orijinal 2000x1124 karede x=1750..1830, y=80..160).
    pr_h, pr_w = pr1.shape[:2]
    patch_x0 = int(round(pr_w * 0.875))
    patch_y0 = int(round(pr_h * 0.0712))
    assert patch_x0 + 80 <= pr_w and patch_y0 + 80 <= pr_h, (
        f"Kare çok küçük ({pr_w}x{pr_h}), 80x80 yama sığmıyor"
    )
    sample_patch = pr1[patch_y0:patch_y0 + 80, patch_x0:patch_x0 + 80].copy()
    unrotated_patch = guard.normalize_patch_to_north(sample_patch, camera_yaw_deg=0.0)
    rotated_patch = guard.normalize_patch_to_north(sample_patch, camera_yaw_deg=90.0)
    assert unrotated_patch.shape == (80, 80, 3) and rotated_patch.shape == (80, 80, 3)
    assert np.array_equal(unrotated_patch, sample_patch), "0° yaw açısında 80x80 yama bozulmamalı!"

    return (
        f"Sabit Kuzey doğrulandı (PR1 güven={res1['confidence']:.2f}, "
        f"PR2 güven={res2['confidence']:.2f}), 80x80 Kuzey-Yukarı telafisi aktif"
    )


def _test_item_07_sobel_clahe_patch_normalizer(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 7: Sobel Kenar + CLAHE Aydınlatma Bağımsız Yama Normalizasyonu (`SobelClahePatchNormalizer`).
    """
    normalizer = SobelClahePatchNormalizer()
    sprout_img = screenshots["minimap_sprout_closeup.jpg"]

    # Gündüz referans yaması (48x48) ve yapay gece/parlama bozulmalı yamalar
    day_patch = cv2.resize(sprout_img, (48, 48), interpolation=cv2.INTER_AREA)
    night_patch = np.clip(day_patch.astype(np.float32) * 0.35 + 5.0, 0, 255).astype(np.uint8)
    glare_patch = np.clip(day_patch.astype(np.float32) * 1.45 + 35.0, 0, 255).astype(np.uint8)

    norm_day = normalizer.normalize_patch(day_patch)
    norm_night = normalizer.normalize_patch(night_patch)
    assert norm_day.shape == (48, 48) and norm_day.dtype == np.uint8
    assert norm_night.shape == (48, 48) and norm_night.dtype == np.uint8

    # Büyük arama alanında gündüz şablonunu gece görüntüsü içinde eşleştir
    # Yama sol-üst=(42, 36), boyut=(48, 48) -> beklenen merkez=(42 + 24, 36 + 24) = (66, 60)
    search_canvas = np.full((120, 120, 3), 18, dtype=np.uint8)
    search_canvas[36:84, 42:90] = night_patch
    score_night, center_night = normalizer.match_normalized_patch(search_canvas, day_patch)
    score_glare, _ = normalizer.match_normalized_patch(glare_patch, day_patch)

    assert score_night >= 0.80, f"Gece/Gündüz Sobel+CLAHE eşleşme skoru düşük: {score_night:.3f}"
    assert math.hypot(center_night[0] - 66, center_night[1] - 60) <= 3.0, (
        f"Eşleşme merkez konumu sapması: beklenen=(66, 60), bulunan={center_night}"
    )
    assert score_glare >= 0.85, f"Parlama (glare) Sobel+CLAHE eşleşme skoru düşük: {score_glare:.3f}"

    # F-1243: gece/parlama kanıtı tek bir gündüz yamasından türetilemez —
    # bağımsız ikinci kaynak (pilgrims_ridge_2.jpg, farklı kare ve farklı
    # kırpım) kendi gece türeviyle ayrıca doğrulanır.
    pr2_img = screenshots["pilgrims_ridge_2.jpg"]
    day_patch_2 = cv2.resize(pr2_img, (48, 48), interpolation=cv2.INTER_AREA)
    night_patch_2 = np.clip(day_patch_2.astype(np.float32) * 0.35 + 5.0, 0, 255).astype(np.uint8)
    search_canvas_2 = np.full((120, 120, 3), 18, dtype=np.uint8)
    search_canvas_2[36:84, 42:90] = night_patch_2
    score_night_2, center_night_2 = normalizer.match_normalized_patch(search_canvas_2, day_patch_2)
    assert score_night_2 >= 0.80, (
        f"Bağımsız 2. kaynakta Gece/Gündüz eşleşme skoru düşük: {score_night_2:.3f}"
    )
    assert math.hypot(center_night_2[0] - 66, center_night_2[1] - 60) <= 3.0, (
        f"Bağımsız 2. kaynakta eşleşme merkezi sapması: beklenen=(66, 60), bulunan={center_night_2}"
    )

    return (
        f"CLAHE(%65)+Sobel(%35) normalizasyonu başarılı; "
        f"Gece/Gündüz skoru={score_night:.3f} (merkez={center_night}), Parlama skoru={score_glare:.3f}, "
        f"Bağımsız 2. kaynak gece skoru={score_night_2:.3f}"
    )


def _test_item_08_high_res_2560x1440_roi_cropper(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 8: 2560x1440 (2K QHD) Yüksek Çözünürlüklü Statik ROI Kırpıcı (`HighResROICropper`).
    """
    cropper = HighResROICropper()
    # F-1244: test tamamen boş kare üzerinde yapılamaz — gerçek referans
    # ekran görüntüsü (pilgrims_ridge_1.jpg) 2560x1440'a ölçeklenir.
    frame_2k = cv2.resize(
        screenshots["pilgrims_ridge_1.jpg"], (2560, 1440), interpolation=cv2.INTER_AREA
    )
    assert frame_2k.shape == (1440, 2560, 3) and float(np.mean(frame_2k)) > 0.0, (
        "ROI testi için gerçek içerikli 2560x1440 kare gerekli"
    )

    f_crop, f_meta = cropper.crop_f_gather_roi(frame_2k, return_meta=True)
    crest_crop, crest_meta = cropper.crop_3d_crest_horizon_roi(frame_2k, return_meta=True)
    mm_crop, mm_meta = cropper.crop_minimap_roi(frame_2k, return_meta=True)
    mm_inner, mm_inner_meta = cropper.crop_with_meta(
        frame_2k, cropper.MINIMAP_INNER_RECT, name="minimap_inner_roi"
    )

    # Sıfır-kopya (NumPy view) doğrulaması + kırpımların gerçek içerik taşıması
    assert float(np.mean(f_crop)) > 0.0 and float(np.mean(mm_crop)) > 0.0, (
        "ROI kırpımları boş kareden gelmemeli (gerçek içerik bekleniyordu)"
    )
    assert np.shares_memory(f_crop, frame_2k), "crop_f_gather_roi sıfır-kopya (NumPy view) olmalı!"
    assert f_crop.size > 0 and f_meta.pixel_reduction_ratio >= 0.90, (
        f"[F Gather] ROI piksel tasarrufu yetersiz: {f_meta.pixel_reduction_ratio:.3f}"
    )
    assert crest_crop.size > 0 and crest_meta.pixel_reduction_ratio >= 0.60, (
        f"3D Crest ROI piksel tasarrufu yetersiz: {crest_meta.pixel_reduction_ratio:.3f}"
    )
    assert mm_crop.size > 0 and mm_meta.pixel_reduction_ratio >= 0.95, (
        f"Minimap ROI piksel tasarrufu yetersiz: {mm_meta.pixel_reduction_ratio:.3f}"
    )
    assert mm_inner.size > 0 and mm_inner_meta.pixel_reduction_ratio >= 0.96

    # Yerel ROI koordinatından tam ekran 2560x1440 koordinatına geri dönüşüm doğruluğu
    gx, gy = f_meta.to_full_screen(25, 15)
    assert gx == f_meta.x_offset + 25 and gy == f_meta.y_offset + 15

    return (
        f"2560x1440 sıfır-kopya ROI tasarrufu: FGather=%{f_meta.pixel_reduction_ratio * 100:.1f}, "
        f"Minimap=%{mm_meta.pixel_reduction_ratio * 100:.1f}, "
        f"3DCrest=%{crest_meta.pixel_reduction_ratio * 100:.1f}"
    )


def _test_item_09_watershed_cluster_separator() -> str:
    """
    Madde 9: Bitişik Minimap İkonlarını Ayıran Mesafe Dönüşümü + Watershed Küme Ayırıcı
    (`WatershedClusterSeparator`).
    """
    separator = WatershedClusterSeparator(min_peak_distance_px=5, expected_icon_radius_px=5.5)

    # Birbirine değen / üst üste binen 2 ve 3 beyaz filiz ikonu oluştur
    canvas = np.zeros((100, 100), dtype=np.uint8)
    true_centers = [(30, 45), (41, 45), (72, 65)]
    for cx, cy in true_centers:
        cv2.circle(canvas, (cx, cy), 6, 255, -1)

    num_cc, _ = cv2.connectedComponents(canvas)
    assert num_cc - 1 == 2, f"Test maskesinde 2 bağlı bileşen bekleniyordu, bulunan: {num_cc - 1}"

    centers = separator.separate_touching_icons(canvas)
    assert len(centers) == 3, f"Watershed bitişik kümeden 3 merkez çıkarmalıydı, bulunan ({len(centers)}): {centers}"

    for tx, ty in true_centers:
        nearest_dist = min(math.hypot(cx - tx, cy - ty) for cx, cy in centers)
        assert nearest_dist <= 3.5, f"Gerçek merkez ({tx},{ty}) ile ayrıştırılan merkez sapması yüksek: {nearest_dist:.2f}px"

    return (
        f"2 bitişik leke -> {len(centers)} bağımsız ikon merkezine ayrıştırıldı: {centers}"
    )


def _test_item_10_motion_blur_gate(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 10: Hareket Bulanıklığı (Motion Blur) Laplacian Varyans Kapısı (`MotionBlurGate`).
    """
    gate = MotionBlurGate(default_threshold=45.0)
    sharp_frame = screenshots["pilgrims_ridge_1.jpg"]

    is_blurry_sharp, var_sharp = gate.is_frame_blurry(sharp_frame, threshold=45.0)
    assert is_blurry_sharp is False and var_sharp >= 45.0, (
        f"Net referans ekran görüntüsü (pilgrims_ridge_1.jpg) bulanık sanıldı: var={var_sharp:.2f}"
    )

    ksize = 35
    motion_kernel = np.zeros((ksize, ksize), dtype=np.float32)
    motion_kernel[ksize // 2, :] = 1.0 / ksize
    blurred_frame = cv2.filter2D(sharp_frame, -1, motion_kernel)
    blurred_frame = cv2.GaussianBlur(blurred_frame, (15, 15), 5.0)

    is_blurry_blur, var_blur = gate.is_frame_blurry(blurred_frame, threshold=45.0)
    assert is_blurry_blur is True and var_blur < 45.0, (
        f"Bulanıklaştırılmış kare MotionBlurGate tarafından yakalanamadı: var={var_blur:.2f}"
    )

    return (
        f"Net kare (PR1) var={var_sharp:.1f} -> blurry=False; "
        f"Hareket bulanık kare var={var_blur:.2f} (<45.0) -> blurry=True"
    )


# ==============================================================================
# BÖLÜM 2 (MADDE 11–20): KARARLILIK KORUMALARI VE LOG HATA DÜZELTMELERİ
# ==============================================================================


def _test_item_11_camera_center_snap_2560x1440_whitelist() -> str:
    """
    Madde 11: 2560x1440 Oyun Kamerası Merkez Yakalama `(1280, 720)` Yanlış Fare Müdahalesi İstisnası.
    """
    cam_rule = CameraLockExceptionRule(tolerance_px=40.0)
    input_guard = PhysicalInputGuard(cursor_deviation_threshold_px=350.0)

    # 1) (1280, 720) 2K QHD ve (960, 540) 1080p merkez noktaları kamera kilidi olarak tanınmalı
    assert cam_rule.is_camera_lock_position(1280, 720) is True
    assert cam_rule.is_camera_lock_position(1288, 714) is True
    assert cam_rule.is_camera_lock_position(960, 540) is True
    assert input_guard.is_game_center_snap(1280, 720) is True

    # 2) Bot fareyi (1850, 420) noktasına götürdüğünde oyun kamerası imleci (1280, 720)'ye
    # kilitlerse (delta=644px > 25px), bu durum ASLA kullanıcı müdahalesi / abort sayılmamalı!
    jump_info = cam_rule.classify_cursor_jump(
        expected_xy=(1850, 420),
        actual_xy=(1280, 720),
        threshold_px=25.0,
        cursor_visible=False,
    )
    assert jump_info["is_camera_lock"] is True, f"Kamera kilidi algılanmadı: {jump_info}"
    assert jump_info["should_abort"] is False, "(1280, 720) kamera kilidi yanlış abort tetikledi!"

    # 3) Gerçek kullanıcı fareyi rastgele bir kenara (300, 900) çekerse müdahale tespit edilmeli
    real_user_jump = cam_rule.classify_cursor_jump(
        expected_xy=(1850, 420),
        actual_xy=(300, 900),
        threshold_px=25.0,
        cursor_visible=True,
    )
    assert real_user_jump["should_abort"] is True and real_user_jump["is_camera_lock"] is False

    return (
        f"(1280, 720) 2K merkez yakalama -> '{jump_info['classification']}' "
        f"(should_abort=False, delta={jump_info['delta_px']:.1f}px yoksayıldı)"
    )


def _test_item_12_cursor_visibility_automator() -> str:
    """
    Madde 12: Win32 `GetCursorInfo` İmleç Görünürlük Denetimi ve Otomatik `Alt` Kurtarma
    (`CursorVisibilityAutomator`).
    """
    # F-1175: kurtarma mock'un kendisi tarafından yapılamaz — tuş basıcı
    # yalnızca basımı kaydeder; görünürlük, benzetilmiş işletim sistemi
    # tepkisi olarak okuyucuda (getter) türetilir: Alt basıldıktan sonraki
    # sorguda imleç görünür döner. Böylece SUT'un (automator) gönderim +
    # yeniden-sorgu + sayaç mantığı test edilir, mock kurtarmayı üstlenmez.
    alt_presses: List[str] = []

    def _mock_getter() -> Dict[str, Any]:
        visible = len(alt_presses) >= 1  # benzetilmiş OS: Alt imleci açar
        return {
            "visible": visible,
            "flags": 1 if visible else 0,
            "x": 1280,
            "y": 720,
        }

    def _mock_key_presser(key: str) -> bool:
        alt_presses.append(str(key).lower())
        return True

    automator = CursorVisibilityAutomator(
        toggle_key="alt",
        cursor_info_getter=_mock_getter,
        key_presser=_mock_key_presser,
        settle_delay_sec=0.005,
    )
    assert automator.is_cursor_visible() is False
    recovered = automator.ensure_cursor_visible(max_attempts=2)
    assert recovered is True and automator.is_cursor_visible() is True
    assert alt_presses == ["alt"] and automator.reveal_count == 1
    assert automator.last_pressed_key == "alt"

    # Gerçek Win32 GetCursorInfo API çağrısının hatasız çalıştığını doğrula
    real_automator = CursorVisibilityAutomator()
    real_vis = real_automator.is_cursor_visible()
    assert isinstance(real_vis, bool)

    return (
        f"Win32 GetCursorInfo(visible={real_vis}), Gizli imleçte otomatik 'alt' basımı ile "
        f"imleç açıldı (reveal_count={automator.reveal_count})"
    )


def _test_item_13_five_frame_death_confirmation_guard(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 13: 5 Ardışık Kare + Gri Ekran + Dirilme Penceresi (`Revive Dialog`) Ölüm Doğrulama
    Koruması (`FiveFrameDeathConfirmationGuard`).
    """
    guard = FiveFrameDeathConfirmationGuard(required_frames=5)
    pr1 = screenshots["pilgrims_ridge_1.jpg"]

    # KURAL 1: pilgrims_ridge_1.jpg üzerinde (hp_ratio=0.000 okunsa bile) check_character_dead == False olmalı!
    for _ in range(6):
        is_dead_pr1 = guard.check_character_dead(frame=pr1, hp_ratio=0.0)
        assert is_dead_pr1 is False, "pilgrims_ridge_1.jpg üzerinde yanlış pozitif ölüm algılandı!"

    eval_pr1 = guard.evaluate_frame(frame=pr1, hp_ratio=0.0)
    assert eval_pr1["is_dead"] is False and eval_pr1["zero_hp_false_positive_blocked"] is True

    # KURAL 2: Gerçek ölüm durumu (Gri ekran + Koyu dirilme modal penceresi + Kırmızı/Altın dirilme butonu)
    guard.reset()
    gray_dead_frame = cv2.cvtColor(cv2.cvtColor(pr1, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    fh, fw = gray_dead_frame.shape[:2]
    # Merkezde koyu dirilme modal penceresi (V=28) + alt orta kısımda kırmızı/altın dirilme butonu
    mx1, my1, mx2, my2 = int(fw * 0.34), int(fh * 0.30), int(fw * 0.66), int(fh * 0.70)
    cv2.rectangle(gray_dead_frame, (mx1, my1), (mx2, my2), (28, 28, 28), -1)
    cv2.rectangle(gray_dead_frame, (mx1, my1), (mx2, my2), (160, 160, 160), 2)
    # Dirilme butonu: BGR=(20, 40, 210) -> HSV H~4 (kırmızı/altın), S>200, V=210
    cv2.rectangle(
        gray_dead_frame,
        (int(fw * 0.42), int(fh * 0.56)),
        (int(fw * 0.58), int(fh * 0.65)),
        (20, 40, 210),
        -1,
    )

    states: List[bool] = []
    for _ in range(5):
        states.append(guard.check_character_dead(frame=gray_dead_frame, hp_ratio=0.0))

    assert states[:4] == [False, False, False, False], f"5 kareden önce erken ölüm tetiklendi: {states}"
    assert states[4] is True, f"5. ardışık ölüm karesinde ölüm onaylanmadı: {states}"

    return (
        f"check_character_dead(pilgrims_ridge_1.jpg, hp=0.0)==False (0-HP koruma aktif); "
        f"5-kare dizisi={states}"
    )


def _test_item_14_multi_criteria_map_open_verifier(
    screenshots: Dict[str, np.ndarray],
    vision_engine: VisionEngine,
) -> str:
    """
    Madde 14: Çoklu Kriterli (`>= 2/3`) Dünya Haritası (`M`) Açılış Doğrulayıcı
    (`MultiCriteriaMapVerifier` & `VisionEngine.is_map_open`).
    """
    verifier = MultiCriteriaMapVerifier(min_criteria=2)

    for name in (
        "pilgrims_ridge_1.jpg",
        "pilgrims_ridge_2.jpg",
        "aria_f_gather_not_on_minimap.jpg",
        "mela_f_gather_not_on_minimap.jpg",
        "aria_3d_crest_far.jpg",
    ):
        frame = screenshots[name]
        res = verifier.verify_map_state(frame)
        vis_open = vision_engine.is_map_open(frame)
        assert res["is_open"] is False and vis_open is False, (
            f"3B dünya görüntüsünde ({name}) harita açık sanıldı: {res}"
        )

    # F-1245: "gerçek" pozitif örnek elle çizilmiş boş sentetik harita olamaz —
    # pozitif örnek, gerçek bir ekran görüntüsü (pilgrims_ridge_1.jpg) tabanı
    # üzerine kurulur: harita açıkken arayüz panelleri düz koyu olur, dünya
    # karesi altta kalır. Kriter bölgelerinin (üst şerit, sol panel) üstüne
    # düz koyu panel çizilir; karenin geri kalanı gerçek oyun görüntüsüdür.
    # Koyu harita görünümü için gerçek kare %35 parlaklığa indirilir.
    synth_map = (
        cv2.resize(screenshots["pilgrims_ridge_1.jpg"], (1920, 1080), interpolation=cv2.INTER_AREA)
        .astype(np.float32)
        * 0.35
    ).astype(np.uint8)
    cv2.rectangle(synth_map, (0, 0), (1920, 150), (28, 28, 28), -1)
    cv2.rectangle(synth_map, (0, 200), (430, 1080), (28, 28, 28), -1)
    assert float(np.mean(synth_map != 28)) > 0.5, (
        "Pozitif harita örneği gerçek kare içeriği taşımalı (düz zemin olamaz)"
    )
    cv2.line(synth_map, (1855, 28), (1880, 53), (235, 235, 235), 2)
    cv2.line(synth_map, (1880, 28), (1855, 53), (235, 235, 235), 2)
    cv2.putText(synth_map, "PILGRIM'S RIDGE - WORLD MAP", (60, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (220, 220, 220), 2)
    for row_y in range(280, 680, 45):
        cv2.putText(synth_map, "Filter Node", (40, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    open_res = verifier.verify_map_state(synth_map)
    assert open_res["is_open"] is True and open_res["matched_count"] >= 2, (
        f"Çoklu kriterli harita doğrulaması açık haritayı tanıyamadı: {open_res}"
    )

    return (
        f"5 gerçek 3B ekran görüntüsünde is_map_open=False ('Kapalı'); "
        f"Harita ekranında {open_res['matched_count']}/3 kriter ile '{open_res['state_label']}'"
    )


def _test_item_15_self_input_whitelist_250ms_window() -> str:
    """
    Madde 15: 250ms Öz-Girdi Beyaz Liste Penceresi (`SelfInputWhitelistWindow` & `InputDriver`).
    """
    whitelist = SelfInputWhitelistWindow(default_window_ms=250.0)
    t0 = 1000.0

    # Botun kendi bastığı tuşlar (W, F, E, Alt, Space, M) 250ms içinde öz-girdi olarak tanınmalı
    for k in ("w", "f", "e", "alt", "space", "m"):
        whitelist.record_self_key(k, timestamp=t0)
        assert whitelist.is_within_ignore_window(k, now=t0 + 0.120) is True, (
            f"Tuş '{k}' 120ms içinde beyaz listede bulunamadı!"
        )

    # 250ms penceresi geçtikten sonra (örn. +310ms) pencere dışı (False) dönmeli
    assert whitelist.is_within_ignore_window("w", now=t0 + 0.310) is False

    # Botun hiç basmadığı yabancı bir tuş (örn. 'p') anında False dönmeli
    assert whitelist.is_within_ignore_window("p", now=t0 + 0.050) is False

    return (
        "250ms öz-girdi beyaz listesi (W/F/E/Alt/Space/M) aktif: "
        "t+120ms=Yoksayıldı (Bot), t+310ms=Süre Doldu (Kullanıcı Müdahalesi)"
    )


def _test_item_16_uac_admin_elevator_venv_preservation() -> str:
    """
    Madde 16: Sanal Ortam (`venv`) Korumalı UAC Yönetici İmtiyazı Yükseltici (`core/uac_elevator.py`).
    """
    admin_flag = is_admin()
    assert isinstance(admin_flag, bool)

    venv_py = resolve_venv_python(preserve_venv=True)
    assert os.path.isabs(venv_py) and "python" in os.path.basename(venv_py).lower()

    payload = build_elevation_payload(
        preserve_venv=True,
        argv=["main.py", "--profile", "Pilgrim's Ridge", "--test-mode"],
        working_dir=PROJECT_ROOT,
    )
    assert payload["lpVerb"] == "runas"
    assert os.path.isabs(payload["lpFile"])
    assert "--test-mode" in payload["lpParameters"]
    assert payload["lpDirectory"] == os.path.abspath(PROJECT_ROOT)

    # F-1176: dry_run=True ile yükseltme hiç yapılmaz — gerçek (dry_run=False)
    # yol, ShellExecuteW yerine enjekte edilen bir stub üzerinden çalıştırılır:
    # stub çağrıyı kaydeder ve başarı kodu (33) döner; böylece UAC istemi
    # açılmadan gerçek yükseltme gönderim yolu doğrulanır.
    shell_calls: List[Tuple[Any, ...]] = []

    def _stub_shell_execute(*args: Any) -> int:
        shell_calls.append(tuple(args))
        return 33  # ShellExecuteW başarı eşiği (> 32)

    elevated_env_before = os.environ.get("GATHERION_UAC_ELEVATED")
    try:
        real_ok = ensure_admin_elevation(
            preserve_venv=True,
            argv=["main.py", "--test-mode"],
            working_dir=PROJECT_ROOT,
            dry_run=False,
            shell_execute_fn=_stub_shell_execute,
        )
    finally:
        if elevated_env_before is None:
            os.environ.pop("GATHERION_UAC_ELEVATED", None)
        else:
            os.environ["GATHERION_UAC_ELEVATED"] = elevated_env_before
    assert real_ok is True, f"Gerçek yükseltme yolu (stub ShellExecuteW) başarısız: {LAST_ELEVATION_REQUEST}"
    if admin_flag:
        # [O10] D24/F-1931 (merged): zaten yöneticiyken yeniden yükseltme
        # denenmez — mock ShellExecuteW hiç çağrılmaz; erken dönüş True +
        # already_admin=True kanıtlanır. (Bu VM root ile çalışır.)
        assert len(shell_calls) == 0, (
            f"Zaten yöneticiyken ShellExecuteW çağrılmamalıydı: {shell_calls}"
        )
        assert LAST_ELEVATION_REQUEST.get("already_admin") is True
        assert LAST_ELEVATION_REQUEST.get("elevated") is True
    else:
        assert len(shell_calls) == 1, f"ShellExecuteW tam 1 kez çağrılmalıydı: {shell_calls}"
        assert shell_calls[0][1] == "runas" and shell_calls[0][2] == payload["lpFile"], (
            f"Yükseltme çağrısı 'runas' + payload exe olmalıydı: {shell_calls[0]}"
        )
        assert LAST_ELEVATION_REQUEST.get("elevated") is True
        assert LAST_ELEVATION_REQUEST.get("return_code") == 33

    return (
        f"is_admin={admin_flag}, venv_python='{os.path.basename(venv_py)}', "
        f"ShellExecuteW(lpVerb='runas', dry_run=False, stub rc=33) doğrulandı"
    )


def _test_item_17_watchdog_paused_state_freeze() -> str:
    """
    Madde 17: Watchdog `PAUSED` / `IDLE` / `STOPPED` Durumu Zaman Aşımı Dondurma Koruması (`BotWatchdog`).
    """
    wd = BotWatchdog(stall_timeout_sec=0.06, freeze_watchdog_on_pause=True)
    wd.record_heartbeat(state="RUNNING", mode="route")

    # PAUSED durumuna alındığında süre aşımı geçse bile ASLA stall tetiklenmemeli!
    wd.set_paused(True, state="PAUSED")
    time.sleep(0.08)
    health_paused = wd.check_health()
    assert wd.is_stalled() is False, "Watchdog PAUSED durumundayken stall tetikledi!"
    assert health_paused["stalled"] is False and health_paused["frozen"] is True

    # PAUSED kaldırılıp RUNNING durumuna geçildiğinde ve heartbeat gelmediğinde stall tespit edilmeli
    # F-1246: stall, özel alan (_last_heartbeat_ts) yazılarak zorlanamaz —
    # set_paused(False) heartbeat'i tazeler; ardından zaman aşımı (0.06sn)
    # gerçekten beklenir ve stall herkese açık API ile gözlenir.
    wd.set_paused(False, state="RUNNING")
    time.sleep(0.08)
    assert wd.is_stalled() is True
    health_stalled = wd.check_health()
    assert health_stalled["stalled"] is True

    return (
        f"PAUSED durumunda stalled={health_paused['stalled']}, frozen={health_paused['frozen']}; "
        f"Aktif zaman aşımında stalled={health_stalled['stalled']}"
    )


_CHAT_PARSER_BUDGET_MS = 100.0


def _test_item_18_fast_sub5ms_chat_system_alerts_parser(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 18: `< 100ms` Sol-Alt Sohbet Sistem Uyarıları Hızlı Görsel/Metin Ayrıştırıcı
    (`GameHUDReader.detect_chat_system_alerts` & `parse_chat_feedback_fast`).

    F-1247: `< 5ms` iddiası makineye bağımlıydı (bu VM'de bile ~15-45ms
    ölçülüyor) — bütçe 100ms'ye çekildi ve hem duvar saati hem de
    ayrıştırıcının bildirdiği süre ayrı ayrı doğrulanır.
    """
    hud = GameHUDReader()
    pr1 = screenshots["pilgrims_ridge_1.jpg"]

    frame_2k = cv2.resize(pr1, (2560, 1440), interpolation=cv2.INTER_NEAREST)
    dark_2k = np.full((1440, 2560, 3), 18, dtype=np.uint8)

    # Isınma çağrısı
    hud.detect_chat_system_alerts(frame_2k)
    hud.parse_chat_feedback_fast(frame_2k)

    t_start = time.perf_counter()
    alerts_pr1 = hud.detect_chat_system_alerts(frame_2k)
    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    fast_pr1 = hud.parse_chat_feedback_fast(frame_2k)
    assert elapsed_ms < _CHAT_PARSER_BUDGET_MS and fast_pr1["elapsed_ms"] < _CHAT_PARSER_BUDGET_MS, (
        f"Sohbet ayrıştırıcı {_CHAT_PARSER_BUDGET_MS:.0f}ms bütçesini aştı: "
        f"{elapsed_ms:.3f}ms / {fast_pr1['elapsed_ms']:.3f}ms"
    )
    assert isinstance(alerts_pr1, dict) and "extraction_failed" in alerts_pr1 and "cannot_mount" in alerts_pr1

    alerts_clean = hud.detect_chat_system_alerts(dark_2k)
    assert (
        alerts_clean["extraction_failed"] is False
        and alerts_clean["cannot_mount"] is False
        and alerts_clean["invalid_target"] is False
    ), f"Boş/temiz sohbet kutusunda yanlış pozitif uyarı: {alerts_clean}"

    # Sistem uyarı metinleri ve aksiyon yönlendirmeleri
    a_fail = hud.parse_chat_feedback_fast("System: Essence Extraction failed.")
    a_mount = hud.parse_chat_feedback_fast("You cannot mount in current state.")
    a_inv = hud.parse_chat_feedback_fast("Invalid target.")

    assert a_fail["extraction_failed"] is True and a_fail["recommended_action"] == "retry_gather"
    assert a_mount["cannot_mount"] is True and a_mount["recommended_action"] == "skip_mount"
    assert a_inv["invalid_target"] is True and a_inv["recommended_action"] == "retarget"

    return (
        f"2560x1440 sohbet tarama süresi={elapsed_ms:.3f}ms (<{_CHAT_PARSER_BUDGET_MS:.0f}ms); "
        f"ExtractionFailed->'{a_fail['recommended_action']}', CannotMount->'{a_mount['recommended_action']}'"
    )


def _test_item_19_top_target_bar_classifier(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 19: Üst-Orta Hedef Çubuğu (`Target Bar`) Materyal vs. Canavar Sınıflandırıcı
    (`GameHUDReader.detect_top_target_bar`).
    """
    hud = GameHUDReader()
    aria_f = screenshots["aria_f_gather_not_on_minimap.jpg"]
    mela_f = screenshots["mela_f_gather_not_on_minimap.jpg"]

    bar_aria = hud.detect_top_target_bar(aria_f)
    bar_mela = hud.detect_top_target_bar(mela_f)
    assert isinstance(bar_aria, dict) and bar_aria["is_monster_target"] is False
    assert isinstance(bar_mela, dict) and bar_mela["is_monster_target"] is False

    # Sentetik canavar hedef barı testi (x=0.422..0.571, y=0.048..0.059 + koyu çerçeve + kırmızı/zeytin HP)
    monster_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bx1, bx2 = int(round(1920 * 0.422)), int(round(1920 * 0.571))
    by1, by2 = int(round(1080 * 0.048)), int(round(1080 * 0.059))
    cv2.rectangle(monster_frame, (bx1, by1), (bx2, by2), (20, 25, 220), -1)

    bar_monster = hud.detect_top_target_bar(monster_frame)
    assert bar_monster["has_target"] is True and bar_monster["is_monster_target"] is True, (
        f"Canavar hedef çubuğu algılanamadı: {bar_monster}"
    )
    assert bar_monster["is_gather_target"] is False

    return (
        f"Aria/Mela hedef barında is_monster_target=False; Sentetik düşman hedef barında "
        f"is_monster_target=True (hp={bar_monster['target_hp_ratio']:.2f})"
    )


def _test_item_20_cast_bar_and_vital_bars_reader(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 20: Toplama İlerleme Çubuğu (`Cast Bar`) ve HP/MP Hayati Çubuk Okuyucu
    (`GameHUDReader.read_cast_bar_progress` & `read_vital_bars`).
    """
    hud = GameHUDReader()
    pr1 = screenshots["pilgrims_ridge_1.jpg"]

    # 1) pilgrims_ridge_1.jpg üzerinde oyuncu HP ve MP çubukları (%100) okunmalı
    vitals = hud.read_vital_bars(pr1)
    assert vitals["hp_ratio"] >= 0.85 and vitals["mp_ratio"] >= 0.85, (
        f"pilgrims_ridge_1.jpg üzerinde hatalı HP/MP okuması: {vitals}"
    )

    # 2) Sentetik %94 dolu turkuaz/mavi toplama ilerleme çubuğu (Cast Bar) testi
    cast_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bx1, bx2 = int(round(1920 * 0.422)), int(round(1920 * 0.571))
    by1, by2 = int(round(1080 * 0.048)), int(round(1080 * 0.059))
    fill_x2 = bx1 + int(round((bx2 - bx1) * 0.92))
    # BGR=(235, 190, 20) -> HSV H~96 (cyan/blue), S>200, V=235
    cv2.rectangle(cast_frame, (bx1, by1), (fill_x2, by2), (235, 190, 20), -1)

    cast_ratio = hud.read_cast_bar_progress(cast_frame)
    assert cast_ratio >= 0.88, f"Cast bar ilerleme oranı okunamadı: {cast_ratio}"

    return (
        f"PR1 Vital HP={vitals['hp_pct']:.1f}%, MP={vitals['mp_pct']:.1f}%; "
        f"Cast Bar ilerlemesi={cast_ratio * 100:.1f}%"
    )


# ==============================================================================
# BÖLÜM 3 (MADDE 21–30): 3x HASAT DÖNGÜSÜ VE KORİDOR NAVİGASYON ZEKASI
# ==============================================================================


def _test_item_21_three_stage_harvest_controller() -> str:
    """
    Madde 21: 3 Aşamalı (`1/3 -> 2/3 -> 3/3`) Hasat Kontrolcüsü ve 5 Dakika
    `RESTING` Döngüsü (`ThreeStageHarvestController`).

    F-1168: başlık "is_anchored=True" ile çelişiyordu — sabitleme yalnızca
    aşamalar sırasındadır: start_node / aşama geçişinde is_anchored=True,
    3/3 tamamlanıp düğüm RESTING'e geçince is_anchored=False olur. Her iki
    durum da aşağıda ayrı ayrı doğrulanır.
    """
    rm, tmp_dir = _build_temp_route_manager()
    try:
        node = rm.add_node(
            region="Pilgrim's Ridge",
            map_x=500,
            map_y=600,
            resource_type="Odyle",
            max_harvests=3,
        )
        node2 = rm.add_node(
            region="Pilgrim's Ridge",
            map_x=520,
            map_y=620,
            resource_type="Odyle",
            max_harvests=3,
        )
        recorder = _MockInputRecorder()
        # F-1128: test_mode=True senaryosu verilen simülasyondur — üretim
        # modunda (test_mode=False) çalıştırılır; tuşlar kayıtçı üzerinden
        # gerçekten gönderilir ve sayılır.
        ctrl = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=recorder,
            reprompt_wait_sec=0.01,
            rest_minutes=5.0,
            sleep_fn=lambda _: None,
        )

        # Aşama sırasındaki sabitleme: start + 1/3 tamamlanınca is_anchored=True
        ctrl_mid = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=_MockInputRecorder(),
            reprompt_wait_sec=0.01,
            rest_minutes=5.0,
            sleep_fn=lambda _: None,
        )
        started = ctrl_mid.start_node(node2.id, gather_key="f", region="Pilgrim's Ridge")
        assert started["is_anchored"] is True, f"start_node düğüme sabitlemeliydi: {started}"
        mid = ctrl_mid.on_cast_completed(
            node_or_id=node2.id,
            prompt_reappeared=True,
            gather_key="f",
            region="Pilgrim's Ridge",
        )
        assert mid["is_anchored"] is True and mid["status"] == "CONTINUE_NEXT_STAGE", (
            f"Aşama geçişinde düğüm sabit kalmalıydı: {mid}"
        )

        res = ctrl.execute_harvest_loop(
            node_or_id=node.id,
            reprompt_sequence=[True, True],
            gather_key="f",
            region="Pilgrim's Ridge",
        )
        assert res["stages_completed"] == ["1/3", "2/3", "3/3"]
        assert res["harvests_completed"] == 3
        assert res["partial_node_detected"] is False
        assert res["node_state"] == "resting"
        # 3/3 tamamlanınca sabitleme kalkar (is_anchored=False) — başlıktaki
        # "is_anchored=True" yalnızca aşama sırası için geçerlidir.
        assert res["is_anchored"] is False
        # Üretim modunda 3 F basımı kayıtçıya düşmeli: ilk + 2 aşama geçişi.
        assert recorder.pressed_keys == ["f", "f", "f"], (
            f"Hasat döngüsünde 3x [F] basımı bekleniyordu: {recorder.pressed_keys}"
        )

        updated_node = rm.get_node(node.id, region="Pilgrim's Ridge")
        assert updated_node is not None and updated_node.state == "resting"
        assert updated_node.remaining_rest_sec() >= 290

        return (
            f"Düğüm #{node.id} aşamalar={res['stages_completed']} tamamlandı -> "
            f"durum='{updated_node.state}' (kalan={updated_node.remaining_rest_sec()}sn)"
        )
    finally:
        _cleanup_temp_dir(tmp_dir)


def _test_item_22_partial_node_early_depletion() -> str:
    """
    Madde 22: Yarım Bırakılmış Kaynak (`Partial Node`) 400ms Yeniden İstem (`Re-Prompt`)
    Kontrolü ve Erken `RESTING` Geçişi (`PartialNodeDetector`).
    """
    rm, tmp_dir = _build_temp_route_manager()
    try:
        node = rm.add_node(
            region="Pilgrim's Ridge",
            map_x=540,
            map_y=620,
            resource_type="Aria",
            max_harvests=3,
        )
        recorder = _MockInputRecorder()
        waited_durations: List[float] = []

        def _mock_sleep(sec: float) -> None:
            waited_durations.append(float(sec))

        partial_det = PartialNodeDetector(reprompt_wait_sec=0.40, sleep_fn=_mock_sleep)
        # F-1128: üretim modunda çalıştırılır (test_mode verilmez).
        ctrl = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=recorder,
            partial_detector=partial_det,
            reprompt_wait_sec=0.40,
            cast_duration_sec=0.0,
            rest_minutes=5.0,
            sleep_fn=_mock_sleep,
        )

        # 1/3 başarılı, ancak başka bir oyuncu daha önce 2/3 topladığı için 400ms sonra [F] tekrar çıkmaz!
        res = ctrl.execute_harvest_loop(
            node_or_id=node.id,
            reprompt_sequence=[False],
            gather_key="f",
            region="Pilgrim's Ridge",
        )
        assert res["stages_completed"] == ["1/3"], f"Erken tükenen düğüm 1/3'te durmalıydı: {res}"
        assert res["partial_node_detected"] is True
        assert res["node_state"] == "resting"
        # F-1238: bekleme süresi iki kaynaktan yedeklenemez — kontrolcünün
        # kendi raporladığı reprompt_waits_sec zorunludur; mock sleep
        # kayıtları yalnızca çapraz doğrulama içindir.
        assert isinstance(res.get("reprompt_waits_sec"), list) and len(res["reprompt_waits_sec"]) >= 1, (
            f"Kontrolcü bekleme süresini raporlamadı: {res}"
        )
        waits_recorded = list(res["reprompt_waits_sec"])
        assert abs(float(waits_recorded[0]) - 0.40) < 1e-3
        assert waited_durations and abs(float(waited_durations[0]) - 0.40) < 1e-3
        # Yarım düğümde yalnızca ilk [F] basılır; 2./3. aşama basılmaz.
        assert recorder.pressed_keys == ["f"], (
            f"Yarım düğümde yalnızca 1x [F] bekleniyordu: {recorder.pressed_keys}"
        )

        updated_node = rm.get_node(node.id, region="Pilgrim's Ridge")
        assert updated_node is not None and updated_node.state == "resting"

        return (
            f"1/3 sonrası {float(waits_recorded[0]) * 1000:.0f}ms beklendi, [F] çıkmayınca "
            f"3. toplama beklenmeden derhal '{updated_node.state}' durumuna alındı"
        )
    finally:
        _cleanup_temp_dir(tmp_dir)


def _test_item_23_extraction_failure_non_decrement_retry() -> str:
    """
    Madde 23: `"Essence Extraction failed"` Durumunda Hak Düşmeme (`current_harvests` Korunumu)
    ve Anında Yeniden Deneme (`handle_extraction_failure`).
    """
    rm, tmp_dir = _build_temp_route_manager()
    try:
        node = rm.add_node(
            region="Pilgrim's Ridge",
            map_x=580,
            map_y=640,
            resource_type="Mela",
            max_harvests=3,
        )
        recorder = _MockInputRecorder()
        # F-1128: üretim modunda çalıştırılır (test_mode verilmez).
        ctrl = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=recorder,
            reprompt_wait_sec=0.01,
            rest_minutes=5.0,
            sleep_fn=lambda _: None,
        )

        ctrl.start_node(node.id, gather_key="f", region="Pilgrim's Ridge")
        # 1. denemede "Essence Extraction failed" -> current_harvests 0 kalmalı, 1/3 tekrarlanmalı!
        step_fail = ctrl.on_cast_completed(
            node_or_id=node.id,
            chat_text="Essence Extraction failed.",
            gather_key="f",
            region="Pilgrim's Ridge",
        )
        assert step_fail["status"] == "RETRY_STAGE"
        assert step_fail["current_harvests"] == 0, f"Başarısız toplamada hak düştü: {step_fail}"
        assert step_fail["retry_stage"] == "1/3"
        assert step_fail["re_pressed"] is True
        assert step_fail["is_anchored"] is True

        # Ardından 1/3, 2/3, 3/3 başarılı tamamlanır
        s1 = ctrl.on_cast_completed(
            node_or_id=node.id, prompt_reappeared=True, gather_key="f", region="Pilgrim's Ridge"
        )
        s2 = ctrl.on_cast_completed(
            node_or_id=node.id, prompt_reappeared=True, gather_key="f", region="Pilgrim's Ridge"
        )
        s3 = ctrl.on_cast_completed(
            node_or_id=node.id, prompt_reappeared=False, gather_key="f", region="Pilgrim's Ridge"
        )
        assert s1["stage"] == "1/3" and s2["stage"] == "2/3" and s3["stage"] == "3/3"
        assert s3["status"] == "COMPLETED" and s3["node_depleted"] is True
        # Üretim modunda her yeniden basım kayıtçıya düşmeli: başarısız
        # deneme + 1/3 + 2/3 geçişi = 3x [F].
        assert recorder.pressed_keys == ["f", "f", "f"], (
            f"Yeniden deneme akışında 3x [F] bekleniyordu: {recorder.pressed_keys}"
        )

        return (
            f"'Essence Extraction failed' -> current_harvests=0 korundu, "
            f"retry_stage='{step_fail['retry_stage']}' için anında F basıldı ve 3/3 tamamlandı"
        )
    finally:
        _cleanup_temp_dir(tmp_dir)


def _test_item_24_auto_mount_manager_35m_threshold() -> str:
    """
    Madde 24: Mesafe Bazlı (`> 35m`) Otomatik Binek (`AutoMountManager`) ve `<= 8m` /
    `[F Gather]` Otomatik İnme + Toplama.
    """
    recorder = _MockInputRecorder()
    # F-1128: üretim modunda çalıştırılır (test_mode verilmez).
    mount_mgr = AutoMountManager(
        input_controller=recorder,
        mount_distance_m=35.0,
        dismount_distance_m=8.0,
        mount_key="c",
        gather_key="f",
    )

    r_mount = mount_mgr.update(distance_m=52.0, in_combat=False, cannot_mount_alert=False)
    assert r_mount["action"] == "MOUNT" and r_mount["is_mounted"] is True

    r_dismount = mount_mgr.update(distance_m=6.5, in_combat=False, gather_prompt_visible=True, gather_key="f")
    assert r_dismount["action"] == "DISMOUNT_AND_GATHER"
    assert r_dismount["is_mounted"] is False and r_dismount["started_gathering"] is True
    # Binek (c) ve toplama (f) tuşları kayıtçıya gerçekten düşmeli.
    assert "c" in recorder.pressed_keys, f"Binek tuşu [C] basılmadı: {recorder.pressed_keys}"
    assert "f" in recorder.pressed_keys, f"Toplama tuşu [F] basılmadı: {recorder.pressed_keys}"

    r_combat = mount_mgr.update(distance_m=60.0, in_combat=True)
    r_cannot = mount_mgr.update(distance_m=60.0, in_combat=False, chat_text="Cannot mount in current state.")
    assert r_combat["mounted"] is False and r_combat["action"] == "BLOCKED_COMBAT"
    assert r_cannot["mounted"] is False and r_cannot["action"] == "BLOCKED_CANNOT_MOUNT"

    return (
        f"52m -> '{r_mount['action']}', 6.5m+[F] -> '{r_dismount['action']}', "
        f"Savaş/CannotMount engeli doğrulandı"
    )


def _test_item_25_mapless_micro_navigator_under_40m() -> str:
    """
    Madde 25: `< 40m` Haritasız (`M` Açmadan) Minimap Üzerinden Doğrudan Vektörel
    Mikro-Navigasyon (`MaplessMicroNavigator`).
    """
    recorder = _MockInputRecorder()
    # F-1128: üretim modunda çalıştırılır (test_mode verilmez) — üretim
    # yolunda fare smooth_move_to ile yönelir.
    micro_nav = MaplessMicroNavigator(
        input_controller=recorder,
        max_micro_nav_distance_m=40.0,
    )

    res = micro_nav.navigate_to_node(
        minimap_offset=(18.0, -18.0),
        distance_m=24.0,
        visible_on_minimap=True,
        player_heading_deg=0.0,
    )
    assert res["used_mapless"] is True
    assert res["map_opened"] is False, "Haritasız mikro-navigasyonda M haritası açıldı!"
    assert res["steered"] is True, f"Mikro-navigasyon yönlendirme yapmadı: {res}"
    assert abs(res["heading_angle_deg"] - 45.0) <= 0.5, f"Pusula açısı 45° bekleniyordu: {res['heading_angle_deg']}"
    # F-1239: basılan tuşlar neredeyse hiç doğrulanmıyordu — yalnızca "m
    # basılmadı" negatif iddiası yetmez: üretim yolunun fare hareketi
    # (smooth_move_to) kayıtçıya düşmeli ve harita tuşu basılmamalı.
    assert len(recorder.mouse_moves) >= 1, (
        f"Mikro-navigasyonda fare yönlendirmesi kaydedilmedi: {recorder.mouse_moves}"
    )
    assert "m" not in recorder.pressed_keys
    assert recorder.pressed_keys == [], (
        f"Mikro-navigasyonda hiç tuş basılmamalıydı: {recorder.pressed_keys}"
    )
    moves_after_near = len(recorder.mouse_moves)

    res_far = micro_nav.navigate_to_node(
        minimap_offset=(90.0, -90.0),
        distance_m=65.0,
        visible_on_minimap=False,
    )
    assert res_far["used_mapless"] is False and res_far["map_opened"] is False
    assert len(recorder.mouse_moves) == moves_after_near, (
        "Uzak hedefte (65m) haritasız yönlendirme fare hareketi yapmamalıydı"
    )

    return (
        f"24m (<40m) hedefte map_opened={res['map_opened']}, "
        f"vektör={res['steering_vector']}, açı={res['heading_angle_deg']}° ile doğrudan yönelindi"
    )


def _test_item_26_phase_correlate_unstuck_maneuver() -> str:
    """
    Madde 26: `cv2.phaseCorrelate` Optik Akış Sıkışma Tespiti ve `Space -> A+W/D+W -> Realign`
    Kurtarma Manevrası (`PhaseCorrelateUnstuckManeuver`).
    """
    unstuck = PhaseCorrelateUnstuckManeuver(
        stuck_threshold_px=1.2,
        stuck_duration_sec=1.5,
        strafe_duration_sec=0.05,
    )
    recorder = _MockInputRecorder()

    rng = np.random.RandomState(42)
    static_minimap = rng.randint(40, 200, size=(200, 200, 3), dtype=np.uint8)

    def _shifted(frame: np.ndarray, dx: float, dy: float) -> np.ndarray:
        # F-1240: sıkışma aynı karenin iki kopyasıyla "tespit" edilemez —
        # ardışık kareler, aynı sahnenin alt-piksel kaydırılmış FARKLI
        # kareleridir (0.3px / 0.6px — eşik 1.2px'in altı, yani takılı).
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(
            frame, mat, (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
        )

    t0 = 2000.0
    unstuck.update_walking_frame(static_minimap, is_walking=True, now=t0)
    st1 = unstuck.update_walking_frame(_shifted(static_minimap, 0.3, 0.2), is_walking=True, now=t0 + 0.6)
    st2 = unstuck.update_walking_frame(_shifted(static_minimap, 0.6, 0.4), is_walking=True, now=t0 + 1.8)

    assert st1["is_stuck"] is False
    assert st2["is_stuck"] is True and st2["shift_px"] < 0.5
    # Karşıt kontrol: gerçekten hareket eden kareler (15px kayma) takılı sayılmaz.
    moving = PhaseCorrelateUnstuckManeuver(
        stuck_threshold_px=1.2,
        stuck_duration_sec=1.5,
        strafe_duration_sec=0.05,
    )
    moving.update_walking_frame(static_minimap, is_walking=True, now=t0)
    mv1 = moving.update_walking_frame(_shifted(static_minimap, 15.0, 0.0), is_walking=True, now=t0 + 0.6)
    mv2 = moving.update_walking_frame(_shifted(static_minimap, 30.0, 0.0), is_walking=True, now=t0 + 1.8)
    assert mv1["shift_px"] > 5.0 and mv1["is_stuck"] is False
    assert mv2["is_stuck"] is False, f"Hareket eden kare takılı sayıldı: {mv2}"

    m1 = unstuck.execute_unstuck_sequence(
        input_controller=recorder,
        target_angle_deg=90.0,
        current_angle_deg=0.0,
        sleep_fn=lambda _: None,
    )
    m2 = unstuck.execute_unstuck_sequence(
        input_controller=recorder,
        target_angle_deg=90.0,
        current_angle_deg=0.0,
        sleep_fn=lambda _: None,
    )

    assert m1["executed"] is True and m1["sequence"] == ["SPACE", "A+W", "REALIGN"]
    assert m2["executed"] is True and m2["sequence"] == ["SPACE", "D+W", "REALIGN"]

    return (
        f"phaseCorrelate kayma={st2['shift_px']:.2f}px (<1.2px) -> Sıkışma tespit edildi; "
        f"Manevra #1={m1['sequence']}, Manevra #2={m2['sequence']}"
    )


def _test_item_27_z_axis_cliff_filter() -> str:
    """
    Madde 27: Z-Ekseni / Uçurum (`Cliff / Altitude Anomaly`) Filtresi (`ZAxisCliffFilter`).
    """
    rm, tmp_dir = _build_temp_route_manager()
    try:
        node = rm.add_node(
            region="Pilgrim's Ridge",
            map_x=610,
            map_y=710,
            resource_type="Orichalcum Ore",
            max_harvests=3,
        )
        cliff_filter = ZAxisCliffFilter(close_minimap_radius_px=25.0, max_failed_approaches=2)

        # Minimap'te düğüm < 25px dibimizde görünmesine rağmen ardışık 2 yaklaşmada etkileşim yok -> Uçurum!
        c1 = cliff_filter.record_approach_attempt(node, minimap_dist_px=4.0, interaction_succeeded=False)
        c2 = cliff_filter.record_approach_attempt(node, minimap_dist_px=3.5, interaction_succeeded=False)
        assert c1 is False and c2 is True
        assert getattr(node, "unreachable_cliff", False) is True
        assert getattr(node, "altitude_band", "") == "cliff_mismatch"

        # Yer koridoru seçiminde dışlanmalı, uçuş modunda (is_flying=True) erişilebilir olmalı
        assert cliff_filter.is_node_excluded_by_cliff(node, is_flying=False) is True
        assert cliff_filter.is_node_excluded_by_cliff(node, is_flying=True) is False

        filtered = cliff_filter.filter_ground_corridor_nodes([node], is_flying=False)
        assert len(filtered) == 0

        return (
            f"Düğüm #{node.id} (2B mesafe=3.5px < 25px, 2 başarısız yaklaşma) -> "
            f"unreachable_cliff=True, yer koridorundan dışlandı"
        )
    finally:
        _cleanup_temp_dir(tmp_dir)


def _test_item_28_predictive_pre_walk_scheduler() -> str:
    """
    Madde 28: Dinlenme Süresi (`Cooldown`) Tahminleyici Öngörülü Ön-Yürüyüş
    (`PredictivePreWalkScheduler` & `SequentialRoutePlanner`).
    """
    pre_walk = PredictivePreWalkScheduler(
        default_rest_duration_sec=300,
        max_prewalk_remaining_sec=8.0,
        arrival_margin_sec=2.0,
        walk_speed_px_per_sec=28.0,
    )
    planner = SequentialRoutePlanner()

    now_ts = 5000.0
    # 1) Dinlenme süresinin bitmesine 120 sn kalan düğüm -> erken yürüyüşe uygun DEĞİL
    node_early = {
        "id": 1,
        "map_x": 280,
        "map_y": 0,
        "state": "resting",
        "last_visited_timestamp": now_ts - 180.0,  # 120 sn kaldı
        "rest_duration_sec": 300,
    }
    eval_early = pre_walk.evaluate_prewalk_candidate(node_early, player_pos=(0.0, 0.0), now=now_ts)
    assert eval_early["eligible"] is False

    # 2) Dinlenme süresinin bitmesine 6.5 sn kalan ve tahmini yürüyüş süresi ~10 sn olan düğüm -> UYGUN!
    node_ready_soon = {
        "id": 2,
        "map_x": 280,
        "map_y": 0,
        "state": "resting",
        "last_visited_timestamp": now_ts - 293.5,  # 6.5 sn kaldı (<= 8.0 sn ve <= 10.0 + 2.0 sn)
        "rest_duration_sec": 300,
    }
    eval_soon = pre_walk.evaluate_prewalk_candidate(node_ready_soon, player_pos=(0.0, 0.0), now=now_ts)
    assert eval_soon["eligible"] is True
    assert pre_walk.should_pre_walk_to_node(node_ready_soon, player_pos=(0.0, 0.0), now=now_ts) is True
    assert planner._is_ready_or_predictive_prewalk(node_ready_soon, player_pos=(0.0, 0.0), now=now_ts) is True

    return (
        f"120sn kalan düğüm eligible=False; {eval_soon['rest_remaining_sec']:.1f}sn kalan düğüm "
        f"(yürüyüş={eval_soon['estimated_walk_sec']:.1f}sn) -> pre_walk eligible=True"
    )


def _test_item_29_ema_centroid_drift_correction() -> str:
    """
    Madde 29: Başarılı Toplama Anında EMA Düğüm Koordinat Kalibrasyonu
    (`CentroidDriftCorrector` & `refine_node_coordinates_on_gather`).
    """
    rm, tmp_dir = _build_temp_route_manager()
    try:
        node = rm.add_node(
            region="Pilgrim's Ridge",
            map_x=1000,
            map_y=2000,
            resource_type="Aria",
            max_harvests=3,
        )
        corrector = CentroidDriftCorrector(ema_alpha=0.20)

        # Eski=(1000, 2000), Yeni=(1020, 1980), alpha=0.20 -> %80 Eski + %20 Yeni = (1004.0, 1996.0)
        ema_res = corrector.refine_node_coordinates_on_gather(
            node=node,
            actual_x=1020.0,
            actual_y=1980.0,
            alpha=0.20,
        )
        assert ema_res["updated"] is True
        assert abs(ema_res["new_x"] - 1004.0) < 1e-3 and abs(ema_res["new_y"] - 1996.0) < 1e-3
        assert abs(float(node.map_x) - 1004.0) < 1e-3 and abs(float(node.map_y) - 1996.0) < 1e-3

        # Doğrudan yardımcı fonksiyon testi
        dict_node = {"id": 99, "map_x": 500.0, "map_y": 500.0, "discovery_source": "3d_crest"}
        fn_res = refine_node_coordinates_on_gather(dict_node, actual_x=520.0, actual_y=510.0)
        assert fn_res["updated"] is True and fn_res["drift_px"] > 0.0

        return (
            f"EMA Kalibrasyonu: (1000, 2000) + (1020, 1980) -> ({ema_res['new_x']:.1f}, {ema_res['new_y']:.1f}), "
            f"kayma={ema_res['drift_px']:.2f}px düzeltildi"
        )
    finally:
        _cleanup_temp_dir(tmp_dir)


def _test_item_30_contested_node_detector_and_skip() -> str:
    """
    Madde 30: Rakip Oyuncu (`Contested Node`) Tespiti ve Vakit Kaybetmeden Sıradaki Düğüme
    Geçiş (`ContestedNodeDetector` & `CorridorNavigator.step_navigation`).
    """
    detector = ContestedNodeDetector()
    nav = CorridorNavigator(contested_detector=detector)

    # Sentetik 3B sahne: Düğüm (960, 560) üzerinde başka bir oyuncunun parlak isimliği + toplama parıltısı
    contested_frame = np.full((1080, 1920, 3), 24, dtype=np.uint8)
    sx, sy = 960, 560
    # 1) Düğümün üstünde yatay oyuncu isimliği (beyaz metin + yeşil/turkuaz HP barı)
    cv2.rectangle(contested_frame, (sx - 45, sy - 65), (sx + 45, sy - 57), (245, 245, 245), -1)
    cv2.rectangle(contested_frame, (sx - 40, sy - 52), (sx + 40, sy - 46), (180, 235, 60), -1)
    # 2) Düğümün tam üzerinde camgöbeği/altın toplama animasyonu parıltısı (gather aura)
    cv2.circle(contested_frame, (sx, sy - 10), 22, (240, 210, 30), -1)

    assert detector.is_node_contested(contested_frame, (sx, sy)) is True
    assert is_node_contested(contested_frame, (sx, sy)) is True

    # CorridorNavigator.step_navigation rakip düğümü gördüğü anda sıradaki düğüme (index 1) atlamalı!
    nodes = [
        {"id": 1, "name": "Contested Aria", "map_x": 300, "map_y": 400, "seq_index": 1},
        {"id": 2, "name": "Free Mela", "map_x": 340, "map_y": 430, "seq_index": 2},
    ]
    step_res = nav.step_navigation(
        frame=contested_frame,
        nodes=nodes,
        current_seq_pos=0,
        player_pos=(300.0, 400.0),
        target_screen_xy=(sx, sy),
        is_walking=True,
    )
    assert step_res["status"] == "contested_skipped" and step_res["contested"] is True
    assert step_res["selected_index"] == 1 and step_res["selected_node"]["id"] == 2

    return (
        f"is_node_contested=True -> Düğüm #1 ('Contested Aria') atlandı ve "
        f"anında Düğüm #{step_res['selected_node']['id']} ('{step_res['selected_node']['name']}') seçildi"
    )


# ==============================================================================
# ANA DUMAN TESTİ KOŞUCUSU VE TÜRKÇE RAPOR TABLOSU
# ==============================================================================


# Seçmeli koşuda (--only) hangi maddelerin hangi paylaşımlı kuruluma ihtiyaç
# duyduğu. Madde -> kullandığı kurulum eşlemesi _test_item_* imzalarından
# çıkarılmıştır (ekran görüntüsü alanlar / mm/mat/vision tarayıcısı alanlar).
_SCREENSHOT_ITEM_NUMBERS: frozenset = frozenset({1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 18, 19, 20})
_SCANNER_ITEM_NUMBERS: frozenset = frozenset({1, 2, 3, 4, 5, 14})

# Madde başına varsayılan zaman aşımı (sn). En yavaş madde bu VM'de ~19sn
# ölçülmüştür; 120sn ~6x pay bırakır, takılan madde ise paketi kilitlemez.
_DEFAULT_PER_ITEM_TIMEOUT_SEC = 120.0

# P2-5 / K-22 — AÇIK-KIRMIZI (bilinen başarısız) maddeler.
# Bu maddeler üretim davranışını değil, bilinen bir açığı kilitler;
# koşucu bunları sıradan başarısızlıktan ayırır (sonuçta
# `known_failure=True` + raporda ayrı satır) ama geçti saymaz —
# sayı (passed/total) değişmez, `ready` bayrağı da False kalır.
# Madde yeşile dönünce buradan silinir ve
# _self_test_smoke_runner_helpers() beklentisi güncellenir.
KNOWN_FAILING_ITEMS: Dict[int, str] = {
    21: (
        "3 aşamalı hasat kontrolcüsü — nedeni üretim/test ayrımıyla "
        "henüz teşhis edilmedi (self-test docstring'indeki "
        "create_region/add_node uyumsuzluğu iddiası kod iziyle "
        "doğrulanamadı); hedefli koşuda traceback ile teşhis edilmeli."
    ),
}


class _ItemTimeoutError(TimeoutError):
    """Tek maddenin per-item timeout sınırını aşması."""


def _run_item_with_timeout(fn: Callable[[], str], timeout_sec: Optional[float]) -> str:
    """`fn()` çağrısını tek-madde zaman aşımıyla çalıştırır.

    POSIX'te (SIGALRM + setitimer varsa) alarm kurulur ve süre aşımında
    `_ItemTimeoutError` fırlatılır; alarm her durumda sıfırlanır ve eski
    sinyal işleyici geri yüklenir. SIGALRM yoksa (Windows) ya da çağrı ana
    iş parçacığında değilse (`signal.signal` ValueError verir) zaman aşımı
    uygulanmadan doğrudan koşulur — sessizce geçmek yerine sınırsız koşu
    tercih edilir, çünkü bu ortamda sinyal zamanlayıcısı yoktur.
    """
    if timeout_sec is None or timeout_sec <= 0:
        return fn()
    sigalrm = getattr(signal, "SIGALRM", None)
    setitimer = getattr(signal, "setitimer", None)
    if sigalrm is None or setitimer is None:
        return fn()

    def _handler(signum: int, frame: Any) -> None:
        raise _ItemTimeoutError(f"madde {float(timeout_sec):g}sn sınırını aştı")

    try:
        old_handler = signal.signal(sigalrm, _handler)
    except (ValueError, OSError, RuntimeError):
        return fn()
    try:
        signal.setitimer(signal.ITIMER_REAL, float(timeout_sec))
        try:
            return fn()
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(sigalrm, old_handler)


def _parse_only_spec(spec: str) -> set:
    """`--only` değerini madde numarası kümesine çevirir.

    Kabul: "7", "7,9,14", "7 9 14". Geçersiz belirteç ya da 1–30 dışı
    numara `ValueError` verir (sessizce yutulmaz, boş kümeye inmez).
    """
    parts = [p for p in spec.replace(",", " ").split() if p]
    if not parts:
        raise ValueError("--only boş bırakılamaz (örn. --only 7,9)")
    try:
        nums = {int(p) for p in parts}
    except ValueError:
        raise ValueError(f"--only sayısal madde listesi olmalı: {spec!r}")
    unknown = nums - set(range(1, 31))
    if unknown:
        raise ValueError(f"--only bilinmeyen madde(ler): {sorted(unknown)} (geçerli: 1-30)")
    return nums


def _write_results_json(
    path: str,
    passed: int,
    total: int,
    results: List[dict],
    total_wall_ms: float,
    only: Optional[set],
) -> str:
    """Koşu sonucunu makinece okunur JSON olarak yazar (CI / karşılaştırma için)."""
    payload = {
        "module": "smoke_test_vision_and_3x",
        "passed": int(passed),
        "total": int(total),
        "total_wall_ms": float(total_wall_ms),
        "only": sorted(only) if only is not None else None,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    return path


def _build_test_specs(
    screenshots: Dict[str, np.ndarray],
    mm_scanner: Optional[Any],
    mat_scanner: Optional[Any],
    vision_engine: Optional[Any],
) -> List[Tuple[int, str, str, Callable[[], str]]]:
    """30 maddenin (no, kategori, baslik, calistirici) listesini kurar.

    Liste kurmak ucuzdur (lambda'lar ancak cagrilinca calisir); paylasimli
    kurulumlar (ekran goruntuleri, tarayicilar) burada YUKLENMEZ, yalnizca
    lambda'lara kapanir. ``--list`` bu sayede hicbir kurulum yapmadan
    madde basliklarini yazdirabilir.
    """
    test_specs: List[Tuple[int, str, str, Callable[[], str]]] = [
        # Bölüm 1: Madde 1–10
        (
            1,
            "Görsel & Filtre",
            "Minimap Beyaz Tüy (%100 Red) ve 3-Yapraklı Filiz Tespiti",
            lambda: _test_item_01_minimap_feather_and_sprout(screenshots, mm_scanner, vision_engine),
        ),
        (
            2,
            "Görsel & Filtre",
            "5-Ölçekli (0.6x..1.4x) 3B Materyal Arması (Aria & Mela) Piramidi",
            lambda: _test_item_02_five_scale_3d_crest_pyramid(screenshots, mat_scanner, vision_engine),
        ),
        (
            3,
            "Görsel & Filtre",
            "Dinamik [F Gather] ve [E] Etkileşim Tuşu Okuyucu",
            lambda: _test_item_03_dynamic_f_and_e_gather_key_reader(screenshots, mat_scanner, vision_engine),
        ),
        (
            4,
            "Görsel & Filtre",
            "Minimap Yuvarlak Beyaz Canavar Noktası (•) Dairesellik Filtresi",
            lambda: _test_item_04_monster_circularity_filter(screenshots, mm_scanner),
        ),
        (
            5,
            "Görsel & Filtre",
            "HSV İsim Plakası İzolatörü & 100+ Aion 2 Materyal Veritabanı",
            lambda: _test_item_05_hsv_nameplate_and_100plus_materials_db(screenshots, mat_scanner),
        ),
        (
            6,
            "Görsel & Filtre",
            "Sabit Kuzey (Fixed-North) Minimap Oryantasyon ve Yaw Koruması",
            lambda: _test_item_06_fixed_north_minimap_orientation_guard(screenshots),
        ),
        (
            7,
            "Görsel & Filtre",
            "Sobel Kenar + CLAHE Aydınlatma Bağımsız Yama Normalizasyonu",
            lambda: _test_item_07_sobel_clahe_patch_normalizer(screenshots),
        ),
        (
            8,
            "Görsel & Filtre",
            "2560x1440 (2K QHD) Yüksek Çözünürlüklü Sıfır-Kopya ROI Kırpıcı",
            lambda: _test_item_08_high_res_2560x1440_roi_cropper(screenshots),
        ),
        (
            9,
            "Görsel & Filtre",
            "Mesafe Dönüşümü + Watershed Bitişik Minimap İkon Ayırıcı",
            lambda: _test_item_09_watershed_cluster_separator(),
        ),
        (
            10,
            "Görsel & Filtre",
            "Hareket Bulanıklığı (Motion Blur) Laplacian Varyans Kapısı",
            lambda: _test_item_10_motion_blur_gate(screenshots),
        ),
        # Bölüm 2: Madde 11–20
        (
            11,
            "Kararlılık & Log",
            "(1280, 720) 2560x1440 Kamera Merkez Yakalama Fare İstisnası",
            lambda: _test_item_11_camera_center_snap_2560x1440_whitelist(),
        ),
        (
            12,
            "Kararlılık & Log",
            "Win32 GetCursorInfo İmleç Görünürlük Denetimi ve Otomatik Alt",
            lambda: _test_item_12_cursor_visibility_automator(),
        ),
        (
            13,
            "Kararlılık & Log",
            "5-Kare + Gri Ekran + Dirilme Penceresi Ölüm Doğrulama Koruması",
            lambda: _test_item_13_five_frame_death_confirmation_guard(screenshots),
        ),
        (
            14,
            "Kararlılık & Log",
            "Çoklu Kriterli (>= 2/3) Dünya Haritası (M) Açılış Doğrulayıcı",
            lambda: _test_item_14_multi_criteria_map_open_verifier(screenshots, vision_engine),
        ),
        (
            15,
            "Kararlılık & Log",
            "250ms Öz-Girdi Beyaz Liste Penceresi (SelfInputWhitelistWindow)",
            lambda: _test_item_15_self_input_whitelist_250ms_window(),
        ),
        (
            16,
            "Kararlılık & Log",
            "Sanal Ortam (venv) Korumalı UAC Yönetici İmtiyazı Yükseltici",
            lambda: _test_item_16_uac_admin_elevator_venv_preservation(),
        ),
        (
            17,
            "Kararlılık & Log",
            "Watchdog PAUSED / IDLE Durumu Zaman Aşımı Dondurma Koruması",
            lambda: _test_item_17_watchdog_paused_state_freeze(),
        ),
        (
            18,
            "Kararlılık & Log",
            "< 100ms Sol-Alt Sohbet Sistem Uyarıları Hızlı Ayrıştırıcı",
            lambda: _test_item_18_fast_sub5ms_chat_system_alerts_parser(screenshots),
        ),
        (
            19,
            "Kararlılık & Log",
            "Üst-Orta Hedef Çubuğu (Target Bar) Materyal vs Canavar Ayrımı",
            lambda: _test_item_19_top_target_bar_classifier(screenshots),
        ),
        (
            20,
            "Kararlılık & Log",
            "Toplama İlerleme Çubuğu (Cast Bar) ve HP/MP Hayati Çubuk Okuyucu",
            lambda: _test_item_20_cast_bar_and_vital_bars_reader(screenshots),
        ),
        # Bölüm 3: Madde 21–30
        (
            21,
            "3x Hasat & Rota",
            "3 Aşamalı (1/3 -> 2/3 -> 3/3) Hasat (aşamalar boyu sabit, bitince RESTING)",
            lambda: _test_item_21_three_stage_harvest_controller(),
        ),
        (
            22,
            "3x Hasat & Rota",
            "Yarım Kaynak (Partial Node) 400ms Re-Prompt ve Erken RESTING",
            lambda: _test_item_22_partial_node_early_depletion(),
        ),
        (
            23,
            "3x Hasat & Rota",
            "'Essence Extraction failed' Hak Düşmeme ve Anında Yeniden Deneme",
            lambda: _test_item_23_extraction_failure_non_decrement_retry(),
        ),
        (
            24,
            "3x Hasat & Rota",
            "> 35m Otomatik Binek (AutoMount) ve <= 8m / [F] Otomatik İnme",
            lambda: _test_item_24_auto_mount_manager_35m_threshold(),
        ),
        (
            25,
            "3x Hasat & Rota",
            "< 40m Haritasız (M Açmadan) Minimap Doğrudan Mikro-Navigasyon",
            lambda: _test_item_25_mapless_micro_navigator_under_40m(),
        ),
        (
            26,
            "3x Hasat & Rota",
            "PhaseCorrelate Optik Akış Sıkışma Tespiti ve Space+Strafe Kurtarma",
            lambda: _test_item_26_phase_correlate_unstuck_maneuver(),
        ),
        (
            27,
            "3x Hasat & Rota",
            "Z-Ekseni / Uçurum (Cliff) Filtresi ve Yer Koridoru Dışlama",
            lambda: _test_item_27_z_axis_cliff_filter(),
        ),
        (
            28,
            "3x Hasat & Rota",
            "Dinlenme Süresi (Cooldown <= 8sn) Öngörülü Ön-Yürüyüş (Pre-Walk)",
            lambda: _test_item_28_predictive_pre_walk_scheduler(),
        ),
        (
            29,
            "3x Hasat & Rota",
            "%80 Eski + %20 Yeni EMA Düğüm Koordinat Kalibrasyonu (Drift Fix)",
            lambda: _test_item_29_ema_centroid_drift_correction(),
        ),
        (
            30,
            "3x Hasat & Rota",
            "Rakip Oyuncu (Contested Node) Tespiti ve Sıradaki Düğüme Atlama",
            lambda: _test_item_30_contested_node_detector_and_skip(),
        ),
    ]

    return test_specs


def run_vision_and_3x_smoke_tests(
    print_report: bool = True,
    only: Optional[set] = None,
    per_item_timeout_sec: Optional[float] = 120.0,
    json_out: Optional[str] = None,
) -> Tuple[int, int, List[dict]]:
    """
    Madde 1–30 arasındaki tüm Görsel Algılama, Kararlılık Korumaları ve 3x Hasat / Koridor
    Navigasyon duman testlerini (`reference_screenshots/` altındaki 7 gerçek ekran görüntüsü
    dahil) çalıştırır.

    Parametreler:
        print_report: Türkçe konsol raporu yazdırılsın mı.
        only: yalnızca bu madde numaraları koşsun (örn. {7, 9}); None = 30 maddenin
            tamamı. Seçmeli koşuda dönen `total_count` seçili madde sayısıdır ve
            rapor bunu açıkça "SEÇMELİ KOŞU" diye belirtir — kısmi yeşil, tam
            yeşil gibi okunamaz.
        per_item_timeout_sec: madde başına zaman aşımı (sn); aşan madde
            BAŞARISIZ sayılır ve koşu sürer. None / <= 0 = sınırsız.
            POSIX'te SIGALRM ile uygulanır; SIGALRM yoksa (Windows) ya da
            çağrı ana iş parçacığında değilse zaman aşımı uygulanmaz.
        json_out: verilirse sonuçlar (passed/total/results + duvar süresi +
            seçim) bu JSON dosyaya yazılır.

    Dönüş:
        `(passed_count, total_count, results_list)`
    """
    if only is not None:
        only = set(only)
        unknown = only - set(range(1, 31))
        if unknown:
            raise ValueError(f"bilinmeyen madde(ler): {sorted(unknown)} (geçerli: 1-30)")
        if not only:
            raise ValueError("only boş olamaz (en az 1 madde seçilmeli)")
    run_t0 = time.perf_counter()
    # Seçmeli koşuda gerekmeyen paylaşımlı kurulum atlanır: ekran görüntüsü
    # yükleme ~1,7sn, tarayıcı kurulumu ~1,3sn (bu VM'de ölçülmüştür).
    needs_screenshots = only is None or bool(only & _SCREENSHOT_ITEM_NUMBERS)
    needs_scanners = only is None or bool(only & _SCANNER_ITEM_NUMBERS)
    screenshots = _load_all_reference_screenshots() if needs_screenshots else {}
    mm_scanner = MinimapScanner() if needs_scanners else None
    mat_scanner = MaterialScanner() if needs_scanners else None
    vision_engine = VisionEngine() if needs_scanners else None

    test_specs = _build_test_specs(screenshots, mm_scanner, mat_scanner, vision_engine)
    selected_specs = test_specs if only is None else [s for s in test_specs if s[0] in only]

    results: List[dict] = []
    passed_count = 0
    total_count = len(selected_specs)

    for item_no, category, title, fn in selected_specs:
        t0 = time.perf_counter()
        timed_out = False
        try:
            detail_msg = _run_item_with_timeout(fn, per_item_timeout_sec)
            passed = True
            status_txt = "BAŞARILI"
            passed_count += 1
        except _ItemTimeoutError as exc:
            passed = False
            timed_out = True
            status_txt = "BAŞARISIZ"
            detail_msg = (
                f"TimeoutError: madde #{item_no} {per_item_timeout_sec:g}sn sınırını aştı; "
                f"koşu yarıda kesildi, sonraki maddeler çalışmaya devam eder ({exc})"
            )
        except Exception as exc:
            passed = False
            status_txt = "BAŞARISIZ"
            # F-1280: koşucu hata detayını kısaltamaz — tam traceback
            # (son 12 satır) rapora girer; tek satırlık konum bilgisi yetmez.
            tb_lines = traceback.format_exc().strip().splitlines()
            detail_msg = (
                f"{type(exc).__name__}: {exc}\n"
                + "\n".join(tb_lines[-12:])
            )
        duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        results.append(
            {
                "id": item_no,
                "item_no": item_no,
                "category": category,
                "name": title,
                "title": title,
                "passed": passed,
                "status": status_txt,
                "timeout": timed_out,
                "known_failure": bool(item_no in KNOWN_FAILING_ITEMS),
                "duration_ms": duration_ms,
                "details": detail_msg,
            }
        )

    total_wall_ms = round((time.perf_counter() - run_t0) * 1000.0, 2)
    subset_note = ""
    if only is not None:
        subset_note = (
            f"SEÇMELİ KOŞU: yalnızca {sorted(only)} maddeleri koştu — bu TAM koşu değildir, "
            f"{30 - total_count} madde çalıştırılmadı"
        )
    if print_report:
        _print_turkish_smoke_report(
            passed_count, total_count, results, screenshots,
            total_wall_ms=total_wall_ms, subset_note=subset_note,
        )
    if json_out is not None:
        _write_results_json(json_out, passed_count, total_count, results, total_wall_ms, only)

    return passed_count, total_count, results


def _print_turkish_smoke_report(
    passed_count: int,
    total_count: int,
    results: List[dict],
    screenshots: Dict[str, np.ndarray],
    total_wall_ms: Optional[float] = None,
    subset_note: str = "",
) -> None:
    """Sonuçları ayrıntılı ve okunaklı bir Türkçe tablo halinde konsola yazdırır."""
    sep = "=" * 132
    thin = "-" * 132
    print(sep)
    print(
        "AION 2 ODYLE & MATERYAL TOPLAMA BOTU — DUMAN TESTİ RAPORU (BÖLÜM 1: MADDE 1–30)"
    )
    if screenshots:
        shots_txt = f"Referans Ekran Görüntüleri: {len(screenshots)}/7 yüklendi"
    else:
        shots_txt = "Referans Ekran Görüntüleri: yüklenmedi (seçili maddeler gerektirmedi)"
    print(
        f"{shots_txt} | "
        f"Toplam Maddeler: {total_count} | Başarılı: {passed_count} | Başarısız: {total_count - passed_count}"
    )
    if subset_note:
        print(subset_note)
    print(sep)
    header = f"{'No':<4} | {'Kategori':<16} | {'Durum':<10} | {'Süre(ms)':<8} | {'Test Başlığı ve Doğrulama Özeti'}"
    print(header)
    print(thin)

    for row in results:
        no_str = f"#{row['item_no']:02d}"
        cat_str = str(row["category"])[:16]
        stat_str = "GEÇTİ" if row["passed"] else "KALDI"
        dur_str = f"{row['duration_ms']:7.2f}"
        title_str = str(row["title"])
        detail_str = str(row["details"])
        print(f"{no_str:<4} | {cat_str:<16} | {stat_str:<10} | {dur_str:<8} | {title_str}")
        print(f"{'':<4} | {'':<16} | {'':<10} | {'':<8} |   -> {detail_str}")
        print(thin)

    rate = (passed_count / float(max(1, total_count))) * 100.0
    overall = "TÜM TESTLER BAŞARIYLA GEÇTİ (%100)" if passed_count == total_count else "BAZI TESTLER BAŞARISIZ OLDU"
    print(f"SONUÇ: {passed_count}/{total_count} ({rate:.1f}%) — {overall}")
    # P2-5/K-22: açık-kırmızı maddeler sıradan başarısızlıktan ayrılır —
    # geçti sayılmazlar, yalnızca bilinen açık oldukları raporda görünür.
    known_failed = [r for r in results if r.get("known_failure") and not r.get("passed")]
    if known_failed:
        for row in known_failed:
            print(
                f"BİLİNEN-BAŞARISIZ (açık-kırmızı, geçti sayılmaz): "
                f"#{int(row['item_no']):02d} — {KNOWN_FAILING_ITEMS.get(int(row['item_no']), '')}"
            )
    # Duvar süresi + en yavaş maddeler: yavaşlayan maddeyi tek bakışta gösterir.
    item_total_ms = sum(float(r.get("duration_ms", 0.0)) for r in results)
    wall_ms = total_wall_ms if total_wall_ms is not None else item_total_ms
    slowest = sorted(results, key=lambda r: -float(r.get("duration_ms", 0.0)))[:5]
    print(f"Toplam süre: {wall_ms / 1000.0:.2f}sn (madde süreleri toplamı: {item_total_ms / 1000.0:.2f}sn)")
    if slowest:
        slowest_txt = "; ".join(
            f"#{int(r['item_no']):02d} {float(r['duration_ms']):.0f}ms" for r in slowest
        )
        print(f"En yavaş maddeler: {slowest_txt}")
    print(sep)


# F-1129 / F-1241: hazır/audit bayrakları hiçbir test çalıştırmadan
# "ready: True" dönemez. Tek kaynak, 30 maddenin gerçek koşusudur
# (önbellekli — aynı süreçte bir kez koşulur, sonrası aynı sonucu verir).
# P0-8 (K-16): bu kural SORGU yolunu kapsamaz — version/readiness/heartbeat
# gibi sorgular SAF OKUMADIR ve tam süiti tetiklemez. Ağır 30 maddelik koşu
# YALNIZCA açık çağrıyla tetiklenir: `get_vision_3x_smoke_summary()` /
# `run_vision_and_3x_smoke_tests()` / `__main__`. Diğer bütün get_*
# sorguları `_get_cached_smoke_ready_result()` ile yalnızca önbelleğe bakar:
# özet henüz hesaplanmadıysa `ready: None` + `smoke_computed: False` döner
# (hardcoded True/False YOK), hesaplandıysa önbellekteki gerçek sonucu verir.
_SMOKE_READY_CACHE: Optional[Dict[str, Any]] = None


def _get_smoke_ready_result() -> Dict[str, Any]:
    """30 maddelik duman testini gerçekten koşup hazır sonucunu döndürür.

    AÇIK ÇAĞRI yoludur — yalnızca `get_vision_3x_smoke_summary()` ve
    `run_vision_and_3x_smoke_tests()` üzerinden tetiklenir. Sorgu/getter
    fonksiyonları bunu çağırmaz.
    """
    global _SMOKE_READY_CACHE
    if _SMOKE_READY_CACHE is None:
        try:
            passed, total, results = run_vision_and_3x_smoke_tests(print_report=False)
            failed_items = [int(r["item_no"]) for r in results if not r.get("passed")]
            _SMOKE_READY_CACHE = {
                "module": "smoke_test_vision_and_3x",
                "ready": bool(passed == total and total > 0),
                "passed": int(passed),
                "total": int(total),
                "failed_items": failed_items,
            }
        except Exception as exc:  # noqa: BLE001 — koşu çökerse hazır değildir
            _SMOKE_READY_CACHE = {
                "module": "smoke_test_vision_and_3x",
                "ready": False,
                "passed": 0,
                "total": 0,
                "failed_items": [],
                "error": str(exc),
            }
    return dict(_SMOKE_READY_CACHE)


def _get_cached_smoke_ready_result() -> Optional[Dict[str, Any]]:
    """Önbellekteki vision/3x hazır sonucunu döndürür; henüz yoksa None.

    P0-8 SAF OKUMA: bu yardımcı asla test çalıştırmaz, disk yazmaz.
    """
    return _SMOKE_READY_CACHE


def _cached_vision_ready() -> Optional[bool]:
    """Önbellekteki hazır bayrağı; özet henüz hesaplanmadıysa None."""
    res = _get_cached_smoke_ready_result()
    if res is None:
        return None
    return bool(res["ready"])


def get_vision_3x_smoke_ready_flag() -> dict:
    """SAF OKUMA (P0-8): önbellekteki hazır sonucunu döndürür, test çalıştırmaz.

    Özet henüz açık çağrıyla hesaplanmadıysa `ready` None'dır.
    """
    res = _get_cached_smoke_ready_result()
    if res is None:
        return {
            "module": "smoke_test_vision_and_3x",
            "ready": None,
            "passed": None,
            "total": None,
            "failed_items": [],
            "smoke_computed": False,
        }
    return dict(res, smoke_computed=True)


def get_vision_3x_smoke_summary() -> dict:
    """30 maddelik duman testini gerçekten koşup özetini döndürür.

    AÇIK ÇAĞRI: özeti hesaplamanın (ve önbelleğe almanın) tek getter yoludur.
    """
    res = _get_smoke_ready_result()
    return {
        "vision_3x_smoke_ready": bool(res["ready"]),
        "passed": res["passed"],
        "total": res["total"],
        "failed_items": res["failed_items"],
    }


def get_smoke_test_vision_and_3x_version_info() -> dict:
    """SAF OKUMA (P0-8): sürüm + önbellekteki hazır bayrağı; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "version": __version__, "ready": _cached_vision_ready(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_diagnostic_snapshot() -> dict:
    """SAF OKUMA (P0-8): önbellekteki özetten tanı görüntüsü; tam süiti koşturmaz."""
    res = _get_cached_smoke_ready_result()
    ready = _cached_vision_ready()
    return {
        "module": "smoke_test_vision_and_3x",
        "status": "ok" if ready else ("failing" if ready is False else "not_computed"),
        "ready": ready,
        "passed": res["passed"] if res is not None else None,
        "total": res["total"] if res is not None else None,
        "failed_items": res["failed_items"] if res is not None else [],
        "smoke_computed": res is not None,
    }


def get_smoke_test_vision_and_3x_capabilities() -> list:
    return ["smoke_test_vision_and_3x", "diagnostic_snapshot", "version_info"]


def validate_smoke_test_vision_and_3x_contract() -> bool:
    return bool(get_smoke_test_vision_and_3x_capabilities())


def get_smoke_test_vision_and_3x_baseline_metrics() -> dict:
    # F-1241: metrics her zaman boş / sahte olamaz — gerçek koşu sayıları girer.
    # P0-8: SAF OKUMA — sayılar yalnızca önbellekten gelir; özet henüz
    # hesaplanmadıysa sayılar None'dır (sahte 0 yazılmaz).
    res = _get_cached_smoke_ready_result()
    return {
        "module": "smoke_test_vision_and_3x",
        "contract_valid": validate_smoke_test_vision_and_3x_contract(),
        "capability_count": len(get_smoke_test_vision_and_3x_capabilities()),
        "smoke_passed": res["passed"] if res is not None else None,
        "smoke_total": res["total"] if res is not None else None,
        "smoke_ready": _cached_vision_ready(),
        "failed_items": res["failed_items"] if res is not None else [],
        "smoke_computed": res is not None,
    }


def get_smoke_test_vision_readiness_status() -> dict:
    """SAF OKUMA (P0-8): önbellekteki hazır durumu; tam süiti koşturmaz."""
    return {
        "module": "smoke_test_vision_and_3x",
        "ready": _cached_vision_ready(),
        "metrics": get_smoke_test_vision_and_3x_baseline_metrics(),
        "smoke_computed": _get_cached_smoke_ready_result() is not None,
    }


def get_smoke_test_vision_audit_record() -> dict:
    """SAF OKUMA (P0-8): önbellekteki özetten denetim kaydı; tam süiti koşturmaz."""
    return {
        "module": "smoke_test_vision_and_3x",
        "audit_passed": _cached_vision_ready(),
        "readiness": get_smoke_test_vision_readiness_status(),
        "smoke_computed": _get_cached_smoke_ready_result() is not None,
    }


def get_smoke_test_vision_and_3x_audit_record() -> dict:
    return get_smoke_test_vision_audit_record()


def get_smoke_test_vision_and_3x_operational_digest() -> dict:
    """SAF OKUMA (P0-8): önbellekten işletim özeti; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "operational": _cached_vision_ready(), "audit": get_smoke_test_vision_and_3x_audit_record(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_telemetry_payload() -> dict:
    """SAF OKUMA (P0-8): önbellekten telemetri; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "telemetry_ready": _cached_vision_ready(), "digest": get_smoke_test_vision_and_3x_operational_digest(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_integrity_seal() -> dict:
    """SAF OKUMA (P0-8): önbellekten bütünlük mührü; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "sealed": _cached_vision_ready(), "telemetry": get_smoke_test_vision_and_3x_telemetry_payload(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_compliance_certificate() -> dict:
    """SAF OKUMA (P0-8): önbellekten uygunluk sertifikası; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "certified": _cached_vision_ready(), "seal": get_smoke_test_vision_and_3x_integrity_seal(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_readiness_probe() -> dict:
    """SAF OKUMA (P0-8): önbellekten hazırlık probu; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "ready": _cached_vision_ready(), "certificate": get_smoke_test_vision_and_3x_compliance_certificate(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_audit_summary() -> dict:
    """SAF OKUMA (P0-8): önbellekten denetim özeti; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "audit_passed": _cached_vision_ready(), "probe": get_smoke_test_vision_and_3x_readiness_probe(), "smoke_computed": _get_cached_smoke_ready_result() is not None}


def get_smoke_test_vision_and_3x_heartbeat_snapshot() -> dict:
    """SAF OKUMA (P0-8): önbellekten canlılık görüntüsü; tam süiti koşturmaz."""
    return {"module": "smoke_test_vision_and_3x", "alive": _cached_vision_ready(), "audit": get_smoke_test_vision_and_3x_audit_summary(), "smoke_computed": _get_cached_smoke_ready_result() is not None}





def _list_test_items() -> str:
    """30 maddenin numara/kategori/başlık listesini döndürür (koşu yapmaz)."""
    specs = _build_test_specs({}, None, None, None)
    lines = ["Madde  Kategori         Başlık"]
    for item_no, category, title, _fn in specs:
        lines.append(f"#{item_no:02d}   {category:<16} {title}")
    return "\n".join(lines)


def _self_test_smoke_runner_helpers() -> None:
    """`--self-test`: koşucu yardımcılarını kilitleyen dosya-içi doğrulama.

    Koşar: (1) timeout tetiklenir, (2) hızlı çağrı timeout'dan etkilenmez,
    (3) --only ayrıştırıcı geçerli/geçersiz girdiyi doğru ayırır,
    (4) rapor altbilgisi (toplam süre + en yavaş + seçmeli notu) yazılır,
    (5) seçmeli koşu + JSON çıktısı uçtan uca çalışır,
    (6) ekran görüntüsü gerektiren madde seçilince görüntüler yüklenir,
    (7) bilinen başarısız madde seçmeli koşuda da dürüstçe başarısız
    döner ve açık-kırmızı (known_failure=True, KNOWN_FAILING_ITEMS)
    olarak işaretlenir — sayısı değişmez, düzeltme değildir.
    """
    import contextlib
    import io

    # 1) Timeout tetiklenir.
    try:
        _run_item_with_timeout(lambda: time.sleep(0.5), 0.05)
        raise AssertionError("self-test: 0.05sn timeout tetiklenmedi")
    except _ItemTimeoutError:
        pass
    # 2) Hızlı çağrı etkilenmez.
    assert _run_item_with_timeout(lambda: "ok", 5.0) == "ok"
    assert _run_item_with_timeout(lambda: "ok", None) == "ok"
    # 3) --only ayrıştırıcı.
    assert _parse_only_spec("7,9,14") == {7, 9, 14}
    assert _parse_only_spec("7 9") == {7, 9}
    for bad in ("99", "0", "abc", ""):
        try:
            _parse_only_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"self-test: --only {bad!r} ValueError vermeliydi")
    # 4) Rapor altbilgisi.
    fake = [
        {"item_no": 7, "category": "X", "title": "T", "passed": True,
         "duration_ms": 12.5, "details": "d"},
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _print_turkish_smoke_report(
            1, 1, fake, {}, total_wall_ms=1234.0, subset_note="SEÇMELİ KOŞU: test",
        )
    out = buf.getvalue()
    assert "Toplam süre: 1.23sn" in out, out
    assert "En yavaş maddeler: #07 12ms" in out, out
    assert "SEÇMELİ KOŞU: test" in out, out
    # 5-7) Uçtan uca seçmeli koşular (gerçek maddeler, hızlı olanlar).
    tmp_dir = tempfile.mkdtemp(prefix="d76_selftest_")
    try:
        jpath = os.path.join(tmp_dir, "out.json")
        passed, total, results = run_vision_and_3x_smoke_tests(
            print_report=False, only={9}, per_item_timeout_sec=60.0, json_out=jpath,
        )
        assert (passed, total) == (1, 1), f"self-test: only={{9}} -> {passed}/{total}"
        with open(jpath, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["passed"] == 1 and payload["total"] == 1 and payload["only"] == [9]
        assert len(payload["results"]) == 1 and payload["results"][0]["item_no"] == 9
        assert payload["results"][0]["timeout"] is False
        # 6) Ekran görüntüsü gerektiren madde (yükleme yolu da test edilir).
        passed7, total7, _ = run_vision_and_3x_smoke_tests(
            print_report=False, only={7}, per_item_timeout_sec=60.0,
        )
        assert (passed7, total7) == (1, 1), f"self-test: only={{7}} -> {passed7}/{total7}"
        # 7) Bilinen başarısız madde dürüstçe başarısız döner ve
        # açık-kırmızı (known_failure) olarak işaretlenir.
        passed21, total21, res21 = run_vision_and_3x_smoke_tests(
            print_report=False, only={21}, per_item_timeout_sec=60.0,
        )
        assert (passed21, total21) == (0, 1), f"self-test: only={{21}} -> {passed21}/{total21}"
        assert res21[0]["passed"] is False
        assert res21[0]["known_failure"] is True, (
            "self-test: madde 21 açık-kırmızı işaretlenmeliydi (KNOWN_FAILING_ITEMS)"
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    print("D76 self-test: 7/7 kontrol geçti (timeout, only-parse, rapor, JSON, seçmeli koşu)")


def _main(argv: Optional[List[str]] = None) -> int:
    """CLI giriş noktası. Dönüş: 0 = koşulan maddeler geçti, 1 = kalan var, 2 = kullanım hatası."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Aion 2 görme + 3x hasat duman testi (madde 1-30).",
    )
    parser.add_argument("--only", default=None,
                        help="yalnızca bu maddeleri koştur: 7 veya 7,9,14")
    parser.add_argument("--list", action="store_true",
                        help="madde listesini yazdır, koşturmadan çık")
    parser.add_argument("--json-out", default=None,
                        help="sonuçları bu JSON dosyaya yaz")
    parser.add_argument("--timeout", type=float, default=_DEFAULT_PER_ITEM_TIMEOUT_SEC,
                        help="madde başına zaman aşımı sn (varsayılan: 120)")
    parser.add_argument("--no-timeout", action="store_true",
                        help="madde başına zaman aşımını kapat")
    parser.add_argument("--self-test", action="store_true",
                        help="koşucu yardımcılarının dosya-içi doğrulamasını çalıştır")
    args = parser.parse_args(argv)

    if args.self_test:
        _self_test_smoke_runner_helpers()
        return 0
    if args.list:
        print(_list_test_items())
        return 0
    try:
        only = _parse_only_spec(args.only) if args.only else None
    except ValueError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 2
    timeout = None if args.no_timeout else args.timeout
    try:
        passed, total, _ = run_vision_and_3x_smoke_tests(
            print_report=True, only=only,
            per_item_timeout_sec=timeout, json_out=args.json_out,
        )
    except ValueError as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 2
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(_main())
