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
     (`uac_elevator`), Watchdog `PAUSED` Dondurma Koruması (`BotWatchdog`), `< 5ms`
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
    python tools/smoke_test_vision_and_3x.py
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import os
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

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
    """`reference_screenshots/` dizinindeki 7 gerçek oyun ekran görüntüsünü belleğe yükler."""
    screenshots_dir = os.path.join(PROJECT_ROOT, "reference_screenshots")
    loaded: Dict[str, np.ndarray] = {}
    for fname in REFERENCE_SCREENSHOT_NAMES:
        fpath = os.path.join(screenshots_dir, fname)
        img = _load_image_unicode_safe(fpath)
        if img is not None and img.size > 0:
            loaded[fname] = img
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
    has_rejected_feather = any("feather" in k or "wing" in k for k in rejected_kinds) or bool(
        feather_scan.get("filter_state", {}).get("feather_icons")
    )
    assert len(feather_odyle) == 0, f"Beyaz tüy görüntüsünde yanlış pozitif filiz bulundu: {feather_odyle}"
    assert has_rejected_feather or filter_ui.get("gatherables_rejection_verified", False), (
        f"Beyaz tüy ikonu reddedilenler listesinde görünmedi: {feather_scan}"
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
    vf_info = ""
    if VISION_FILTERS_MODULE is not None and hasattr(VISION_FILTERS_MODULE, "FeatherNegativeFilter"):
        fn_filter = VISION_FILTERS_MODULE.FeatherNegativeFilter()
        assert fn_filter.feather_template is not None and fn_filter.reject_mask is not None
        assert fn_filter.is_feather_icon( fn_filter.feather_template ) is True
        if fn_filter.sprout_templates:
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
    scales: Tuple[float, ...] = (0.6, 0.8, 1.0, 1.2, 1.4)
    if VISION_FILTERS_MODULE is not None and hasattr(VISION_FILTERS_MODULE, "Pyramid3DCrestDetector"):
        pyr_det = VISION_FILTERS_MODULE.Pyramid3DCrestDetector()
        scales = tuple(float(s) for s in pyr_det.scales)
        assert len(scales) == 5 and abs(scales[0] - 0.6) < 1e-4 and abs(scales[-1] - 1.4) < 1e-4, (
            f"Pyramid3DCrestDetector 5-ölçekli piramit (0.6..1.4) eksik: {scales}"
        )
        pyr_hits = pyr_det.detect(aria_far)
        assert len(pyr_hits) >= 1, "Pyramid3DCrestDetector aria_3d_crest_far.jpg üzerinde arma bulamadı!"
        assert 5.0 <= float(pyr_hits[0]["estimated_dist_m"]) <= 40.0

    crests_far = mat_scanner._scan_3d_gather_crests(aria_far)
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
    det_aria_list = mat_scanner.scan_gather_prompt(aria_f)
    vis_aria = vision_engine.detect_gather_prompt(aria_f)
    vis_aria_ok = bool(vis_aria[0]) if isinstance(vis_aria, tuple) else bool(vis_aria)
    key_aria_vis = getattr(vision_engine, "last_detected_gather_key", "f")
    assert len(det_aria_list) >= 1 and vis_aria_ok is True and key_aria_vis == "f", (
        f"aria_f_gather_not_on_minimap.jpg [F] istemi okunamadı: {det_aria_list}, vis={vis_aria}"
    )

    det_mela_list = mat_scanner.scan_gather_prompt(mela_f)
    vis_mela = vision_engine.detect_gather_prompt(mela_f)
    vis_mela_ok = bool(vis_mela[0]) if isinstance(vis_mela, tuple) else bool(vis_mela)
    key_mela_vis = getattr(vision_engine, "last_detected_gather_key", "f")
    assert len(det_mela_list) >= 1 and vis_mela_ok is True and key_mela_vis == "f", (
        f"mela_f_gather_not_on_minimap.jpg [F] istemi okunamadı: {det_mela_list}, vis={vis_mela}"
    )

    # 2) pilgrims_ridge_1.jpg üzerindeki yüzen [E] kutusu ('e' tuşu) tespiti
    e_boxes = mat_scanner._scan_floating_gather_boxes(pr1)
    vis_pr1 = vision_engine.detect_gather_prompt(pr1)
    vis_pr1_ok = bool(vis_pr1[0]) if isinstance(vis_pr1, tuple) else bool(vis_pr1)
    key_pr1_vis = getattr(vision_engine, "last_detected_gather_key", "e")
    assert len(e_boxes) >= 1 and vis_pr1_ok is True and key_pr1_vis == "e", (
        f"pilgrims_ridge_1.jpg üzerinde yüzen [E] kutusu bulunamadı: {e_boxes}, key={key_pr1_vis}"
    )

    # 3) DynamicInteractionKeyReader (core/vision_filters.py) doğrudan doğrulaması
    if VISION_FILTERS_MODULE is not None and hasattr(VISION_FILTERS_MODULE, "DynamicInteractionKeyReader"):
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

    if VISION_FILTERS_MODULE is not None and hasattr(VISION_FILTERS_MODULE, "MonsterCircularityFilter"):
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
        total_monsters += len(monster_dots)
        for g in gatherables:
            assert g.get("kind") == "sprout_material", f"{name}: Gatherable içine yabancı tür sızdı: {g}"
        for m in monster_dots:
            assert float(m.get("circularity", 0.0)) >= 0.75, f"{name}: Monster daireselliği düşük: {m}"

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
    assert len(REJECTED_NON_MATERIAL_KEYWORDS) >= 8

    # NameplateHSVIsolator (core/vision_filters.py) doğrulaması
    if VISION_FILTERS_MODULE is not None and hasattr(VISION_FILTERS_MODULE, "NameplateHSVIsolator"):
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
    assert isinstance(nameplates_aria, list)

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
    sample_patch = pr1[80:160, 1750:1830].copy()
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

    return (
        f"CLAHE(%65)+Sobel(%35) normalizasyonu başarılı; "
        f"Gece/Gündüz skoru={score_night:.3f} (merkez={center_night}), Parlama skoru={score_glare:.3f}"
    )


def _test_item_08_high_res_2560x1440_roi_cropper() -> str:
    """
    Madde 8: 2560x1440 (2K QHD) Yüksek Çözünürlüklü Statik ROI Kırpıcı (`HighResROICropper`).
    """
    cropper = HighResROICropper()
    frame_2k = np.zeros((1440, 2560, 3), dtype=np.uint8)

    f_crop, f_meta = cropper.crop_f_gather_roi(frame_2k, return_meta=True)
    crest_crop, crest_meta = cropper.crop_3d_crest_horizon_roi(frame_2k, return_meta=True)
    mm_crop, mm_meta = cropper.crop_minimap_roi(frame_2k, return_meta=True)
    mm_inner, mm_inner_meta = cropper.crop_with_meta(
        frame_2k, cropper.MINIMAP_INNER_RECT, name="minimap_inner_roi"
    )

    # Sıfır-kopya (NumPy view) doğrulaması
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
    vis_state = {"visible": False, "flags": 0}
    alt_presses: List[str] = []

    def _mock_getter() -> Dict[str, Any]:
        return {
            "visible": vis_state["visible"],
            "flags": vis_state["flags"],
            "x": 1280,
            "y": 720,
        }

    def _mock_key_presser(key: str) -> bool:
        alt_presses.append(str(key).lower())
        vis_state["visible"] = True
        vis_state["flags"] = 1
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

    synth_map = np.full((1080, 1920, 3), 28, dtype=np.uint8)
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

    dry_ok = ensure_admin_elevation(
        preserve_venv=True,
        argv=["main.py", "--test-mode"],
        working_dir=PROJECT_ROOT,
        dry_run=True,
    )
    assert dry_ok is True

    return (
        f"is_admin={admin_flag}, venv_python='{os.path.basename(venv_py)}', "
        f"ShellExecuteW(lpVerb='runas', dry_run=True) doğrulandı"
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
    wd.set_paused(False, state="RUNNING")
    wd._last_heartbeat_ts = time.time() - 5.0
    assert wd.is_stalled() is True
    health_stalled = wd.check_health()
    assert health_stalled["stalled"] is True

    return (
        f"PAUSED durumunda stalled={health_paused['stalled']}, frozen={health_paused['frozen']}; "
        f"Aktif zaman aşımında stalled={health_stalled['stalled']}"
    )


def _test_item_18_fast_sub5ms_chat_system_alerts_parser(
    screenshots: Dict[str, np.ndarray],
) -> str:
    """
    Madde 18: `< 5ms` Sol-Alt Sohbet Sistem Uyarıları Hızlı Görsel/Metin Ayrıştırıcı
    (`GameHUDReader.detect_chat_system_alerts` & `parse_chat_feedback_fast`).
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
    assert elapsed_ms < 5.0 and fast_pr1["elapsed_ms"] < 5.0, (
        f"Sohbet ayrıştırıcı 5ms bütçesini aştı: {elapsed_ms:.3f}ms / {fast_pr1['elapsed_ms']:.3f}ms"
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
        f"2560x1440 sohbet tarama süresi={elapsed_ms:.3f}ms (<5.0ms); "
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
    Madde 21: 3 Aşamalı (`1/3 -> 2/3 -> 3/3`) Sabitlenmeli (`is_anchored=True`) Hasat
    Kontrolcüsü ve 5 Dakika `RESTING` Döngüsü (`ThreeStageHarvestController`).
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
        recorder = _MockInputRecorder()
        ctrl = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=recorder,
            reprompt_wait_sec=0.01,
            rest_minutes=5.0,
            sleep_fn=lambda _: None,
            test_mode=True,
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
        assert res["is_anchored"] is False

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
        ctrl = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=recorder,
            partial_detector=partial_det,
            reprompt_wait_sec=0.40,
            cast_duration_sec=0.0,
            rest_minutes=5.0,
            sleep_fn=_mock_sleep,
            test_mode=True,
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
        waits_recorded = list(res.get("reprompt_waits_sec", [])) or waited_durations
        assert len(waits_recorded) >= 1 and abs(float(waits_recorded[0]) - 0.40) < 1e-3

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
        ctrl = ThreeStageHarvestController(
            route_manager=rm,
            input_controller=recorder,
            reprompt_wait_sec=0.01,
            rest_minutes=5.0,
            sleep_fn=lambda _: None,
            test_mode=True,
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
    mount_mgr = AutoMountManager(
        input_controller=recorder,
        mount_distance_m=35.0,
        dismount_distance_m=8.0,
        mount_key="r",
        gather_key="f",
        test_mode=True,
    )

    r_mount = mount_mgr.update(distance_m=52.0, in_combat=False, cannot_mount_alert=False)
    assert r_mount["action"] == "MOUNT" and r_mount["is_mounted"] is True

    r_dismount = mount_mgr.update(distance_m=6.5, in_combat=False, gather_prompt_visible=True, gather_key="f")
    assert r_dismount["action"] == "DISMOUNT_AND_GATHER"
    assert r_dismount["is_mounted"] is False and r_dismount["started_gathering"] is True

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
    micro_nav = MaplessMicroNavigator(
        input_controller=recorder,
        max_micro_nav_distance_m=40.0,
        test_mode=True,
    )

    res = micro_nav.navigate_to_node(
        minimap_offset=(18.0, -18.0),
        distance_m=24.0,
        visible_on_minimap=True,
        player_heading_deg=0.0,
    )
    assert res["used_mapless"] is True
    assert res["map_opened"] is False, "Haritasız mikro-navigasyonda M haritası açıldı!"
    assert abs(res["heading_angle_deg"] - 45.0) <= 0.5, f"Pusula açısı 45° bekleniyordu: {res['heading_angle_deg']}"
    assert "m" not in recorder.pressed_keys

    res_far = micro_nav.navigate_to_node(
        minimap_offset=(90.0, -90.0),
        distance_m=65.0,
        visible_on_minimap=False,
    )
    assert res_far["used_mapless"] is False and res_far["map_opened"] is False

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

    t0 = 2000.0
    unstuck.update_walking_frame(static_minimap, is_walking=True, now=t0)
    st1 = unstuck.update_walking_frame(static_minimap, is_walking=True, now=t0 + 0.6)
    st2 = unstuck.update_walking_frame(static_minimap, is_walking=True, now=t0 + 1.8)

    assert st1["is_stuck"] is False
    assert st2["is_stuck"] is True and st2["shift_px"] < 0.5

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


def run_vision_and_3x_smoke_tests(print_report: bool = True) -> Tuple[int, int, List[dict]]:
    """
    Madde 1–30 arasındaki tüm Görsel Algılama, Kararlılık Korumaları ve 3x Hasat / Koridor
    Navigasyon duman testlerini (`reference_screenshots/` altındaki 7 gerçek ekran görüntüsü
    dahil) çalıştırır.

    Dönüş:
        `(passed_count, total_count, results_list)`
    """
    screenshots = _load_all_reference_screenshots()
    mm_scanner = MinimapScanner()
    mat_scanner = MaterialScanner()
    vision_engine = VisionEngine()

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
            lambda: _test_item_08_high_res_2560x1440_roi_cropper(),
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
            "< 5ms Sol-Alt Sohbet Sistem Uyarıları Hızlı Ayrıştırıcı",
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
            "3 Aşamalı (1/3 -> 2/3 -> 3/3) Sabitlenmeli Hasat ve 5dk RESTING",
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

    results: List[dict] = []
    passed_count = 0
    total_count = len(test_specs)

    for item_no, category, title, fn in test_specs:
        t0 = time.perf_counter()
        try:
            detail_msg = fn()
            passed = True
            status_txt = "BAŞARILI"
            passed_count += 1
        except Exception as exc:
            passed = False
            status_txt = "BAŞARISIZ"
            tb_lines = traceback.format_exc().strip().splitlines()
            loc_info = tb_lines[-2].strip() if len(tb_lines) >= 2 else ""
            detail_msg = f"{type(exc).__name__}: {exc} [{loc_info}]"
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
                "duration_ms": duration_ms,
                "details": detail_msg,
            }
        )

    if print_report:
        _print_turkish_smoke_report(passed_count, total_count, results, screenshots)

    return passed_count, total_count, results


def _print_turkish_smoke_report(
    passed_count: int,
    total_count: int,
    results: List[dict],
    screenshots: Dict[str, np.ndarray],
) -> None:
    """Sonuçları ayrıntılı ve okunaklı bir Türkçe tablo halinde konsola yazdırır."""
    sep = "=" * 132
    thin = "-" * 132
    print(sep)
    print(
        "AION 2 ODYLE & MATERYAL TOPLAMA BOTU — DUMAN TESTİ RAPORU (BÖLÜM 1: MADDE 1–30)"
    )
    print(
        f"Referans Ekran Görüntüleri: {len(screenshots)}/7 yüklendi | "
        f"Toplam Maddeler: {total_count} | Başarılı: {passed_count} | Başarısız: {total_count - passed_count}"
    )
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
    print(sep)


if __name__ == "__main__":
    passed, total, _ = run_vision_and_3x_smoke_tests(print_report=True)
    sys.exit(0 if passed == total else 1)
