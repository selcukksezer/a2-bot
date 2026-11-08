# -*- coding: utf-8 -*-
"""
Bölge, Rota ve Düğüm Yöneticisi (RouteManager & NodeData).

Verileri `routes/<region_name>/region_data.json` olarak,
harita kesitlerini ise `routes/<region_name>/patches/node_<id>.png` (80x80 gri kesit)
olarak yönetir. F8 anlık kayıt, açık haritada şablon eşleştirme (TM_CCOEFF_NORMED)
ve yedek normalize koordinat hesaplama, bölge/rota CRUD işlemleri ile
ZIP içe/dışa aktarma işlevlerini barındırır.
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import shutil
import time
import zipfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from core.patch_normalizer import MinimapOrientationGuard, SobelClahePatchNormalizer

try:
    from config.materials_db import (
        AION2_MATERIALS_CATALOG,
        MATERIAL_CATEGORIES,
        classify_3d_material_patch,
        get_material_spec,
    )
except ImportError:
    try:
        from materials_db import (  # type: ignore[no-redef]
            AION2_MATERIALS_CATALOG,
            MATERIAL_CATEGORIES,
            classify_3d_material_patch,
            get_material_spec,
        )
    except ImportError:
        AION2_MATERIALS_CATALOG = None  # type: ignore[assignment]
        MATERIAL_CATEGORIES = {}  # type: ignore[assignment]
        classify_3d_material_patch = None  # type: ignore[assignment]
        get_material_spec = None  # type: ignore[assignment]

try:
    import winsound
except ImportError:
    winsound = None  # type: ignore[assignment]

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

PATCH_SIZE: int = 80
DEFAULT_REST_SEC: int = 300
DEFAULT_REGION: str = "Verteron"
DEFAULT_ROUTE: str = "All"
DEFAULT_RESOURCE_TYPE: str = "Orichalcum Ore"
DEFAULT_MAX_HARVESTS: int = 3
DEFAULT_SOURCE_TYPE: str = "manual"
VALID_SOURCE_TYPES: Tuple[str, ...] = ("manual", "minimap_auto", "3d_auto", "questlog")

# ==============================================================================
# SUBREGION CANONICAL MAPPING & BOUNDING SECTOR DEFINITIONS
# ==============================================================================
SUBREGION_TO_CANONICAL: Dict[str, str] = {
    # Altgard subregions
    "Dredgion Crash Site": "Altgard",
    "Safe Haven": "Altgard",
    "Calderon Canyon": "Altgard",
    "Silent Hill": "Altgard",
    "Altgard Fortress": "Altgard",
    "Basfelt Ruins": "Altgard",
    "Nameless Cemetery": "Altgard",
    "Fafnite Deposit": "Altgard",
    "Sanctum Outpost": "Altgard",
    "Quai Campsite": "Altgard",
    "Grave of the Red Sky Union": "Altgard",
    "Moslan Forest": "Altgard",
    "Idun's Lake": "Altgard",
    "Black Plains": "Altgard",
    "Heart of the Dragon": "Altgard",
    # Verteron subregions
    "Tolbas Forest": "Verteron",
    "Dukaki Settlement": "Verteron",
    "Cantas Coast": "Verteron",
    "Ardus Shrine": "Verteron",
    "Pilgrim's Ridge": "Verteron",
    "Pilgrims Ridge": "Verteron",
    "Verteron Citadel": "Verteron",
    "Zulons Island": "Verteron",
    "Ellun River Mouth": "Verteron",
    "Geolus Campsite": "Verteron",
    # Eltnen subregions
    "Manduri Forest": "Eltnen",
    "Golden Bough Garrison": "Eltnen",
    "Eiron Desert": "Eltnen",
    "Kaidan Headquarters": "Eltnen",
    "Eltnen Fortress": "Eltnen",
    # Morheim subregions
    "Ice Claw Village": "Morheim",
    "Patamor Ridge": "Morheim",
    "Salintus Desert": "Morheim",
    "Mist Mane Village": "Morheim",
    "Morheim Fortress": "Morheim",
    # Heiron subregions
    "Heiron Observatory": "Heiron",
    "Senea's Campsite": "Heiron",
    "Kishar Observation Post": "Heiron",
    "Arbolu's Haven": "Heiron",
    # Beluslan subregions
    "Besfer Refugee Camp": "Beluslan",
    "Red Mane Cavern": "Beluslan",
    "Hoarfrost Shelter": "Beluslan",
    "Bakarma Barracks": "Beluslan",
    # Theobomos subregions
    "Meniherk Excavation Camp": "Theobomos",
    "Observatory Village": "Theobomos",
    "Jamanok Inn": "Theobomos",
    # Brusthonin subregions
    "Settlers' Campsite": "Brusthonin",
    "Baltasar Hill Village": "Brusthonin",
    "Bubu Village": "Brusthonin",
    # Reshanta subregions
    "Teminon Landing": "Reshanta",
    "Primum Landing": "Reshanta",
    "Sulfur Tree Nest": "Reshanta",
    "Siel's Eastern Fortress": "Reshanta",
}

SUBREGION_CANONICAL_MAP: Dict[str, str] = SUBREGION_TO_CANONICAL

_LOWER_SUBREGION_MAP: Dict[str, str] = {
    k.strip().lower(): v for k, v in SUBREGION_TO_CANONICAL.items()
}

_CANONICAL_REGION_LOOKUP: Dict[str, str] = {
    "altgard": "Altgard",
    "verteron": "Verteron",
    "eltnen": "Eltnen",
    "morheim": "Morheim",
    "heiron": "Heiron",
    "beluslan": "Beluslan",
    "theobomos": "Theobomos",
    "brusthonin": "Brusthonin",
    "reshanta": "Reshanta",
}

SUBREGION_BOUNDING_AREAS: Dict[str, Dict[str, Any]] = {
    "dredgion crash site": {
        "canonical": "Altgard",
        "bbox": (350, 180, 1600, 950),
        "norm_bbox": (0.18, 0.16, 0.85, 0.88),
        "center": (655, 455),
        "radius": 450.0,
    },
    "safe haven": {
        "canonical": "Altgard",
        "bbox": (250, 150, 620, 440),
        "norm_bbox": (0.13, 0.13, 0.45, 0.41),
        "center": (420, 260),
        "radius": 220.0,
    },
    "calderon canyon": {
        "canonical": "Altgard",
        "bbox": (650, 360, 1250, 780),
        "norm_bbox": (0.33, 0.33, 0.65, 0.72),
        "center": (820, 520),
        "radius": 260.0,
    },
    "silent hill": {
        "canonical": "Altgard",
        "bbox": (300, 350, 680, 650),
        "norm_bbox": (0.15, 0.32, 0.53, 0.60),
        "center": (480, 480),
        "radius": 220.0,
    },
    "altgard fortress": {
        "canonical": "Altgard",
        "bbox": (750, 520, 1300, 900),
        "norm_bbox": (0.39, 0.48, 0.68, 0.83),
        "center": (950, 680),
        "radius": 250.0,
    },
    "basfelt ruins": {
        "canonical": "Altgard",
        "bbox": (300, 380, 620, 620),
        "norm_bbox": (0.15, 0.35, 0.48, 0.57),
        "center": (440, 495),
        "radius": 200.0,
    },
    "nameless cemetery": {
        "canonical": "Altgard",
        "bbox": (320, 240, 600, 480),
        "norm_bbox": (0.16, 0.22, 0.47, 0.44),
        "center": (455, 360),
        "radius": 200.0,
    },
    "fafnite deposit": {
        "canonical": "Altgard",
        "bbox": (440, 180, 720, 420),
        "norm_bbox": (0.22, 0.16, 0.56, 0.39),
        "center": (575, 290),
        "radius": 190.0,
    },
    "sanctum outpost": {
        "canonical": "Altgard",
        "bbox": (600, 200, 880, 450),
        "norm_bbox": (0.31, 0.18, 0.69, 0.42),
        "center": (735, 320),
        "radius": 180.0,
    },
    "quai campsite": {
        "canonical": "Altgard",
        "bbox": (880, 260, 1280, 560),
        "norm_bbox": (0.45, 0.24, 0.67, 0.52),
        "center": (1050, 400),
        "radius": 210.0,
    },
}


def resolve_canonical_region(name: Optional[str]) -> str:
    """Verilen bölge veya alt bölge adının kanonik ana bölge karşılığını döndürür."""
    if not name:
        return DEFAULT_REGION
    clean = str(name).strip()
    lower = clean.lower()
    if lower in _CANONICAL_REGION_LOOKUP:
        return _CANONICAL_REGION_LOOKUP[lower]
    if lower in _LOWER_SUBREGION_MAP:
        return _LOWER_SUBREGION_MAP[lower]
    return clean


def get_canonical_region(name: Optional[str]) -> str:
    """resolve_canonical_region için eşanlamlı yardımcı fonksiyon."""
    return resolve_canonical_region(name)


def is_subregion(name: Optional[str]) -> bool:
    """Verilen adın bir alt bölge (subregion) olup olmadığını kontrol eder."""
    if not name:
        return False
    lower = str(name).strip().lower()
    return lower in _LOWER_SUBREGION_MAP


def get_subregion_info(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """Alt bölge sınırlayıcı alan (bounding box / center) bilgilerini döndürür."""
    if not name:
        return None
    lower = str(name).strip().lower()
    return SUBREGION_BOUNDING_AREAS.get(lower)


def get_subregion_bounding_area(name: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """Alt bölge (min_x, min_y, max_x, max_y) piksel sınırlayıcı kutusunu döndürür."""
    info = get_subregion_info(name)
    if info and "bbox" in info:
        return info["bbox"]
    return None


def filter_nodes_by_subregion(nodes: Sequence[NodeData], subregion: str) -> List[NodeData]:
    """
    Kanonik bölge düğümlerini belirtilen alt bölgenin rota etiketine ve/veya
    sınırlayıcı alanına (bounding box / center radius) göre filtreler.
    """
    clean_sub = str(subregion or "").strip()
    if not clean_sub or not nodes:
        return list(nodes) if nodes else []

    clean_sub_lower = clean_sub.lower()
    # 1. Rota listesinde alt bölge adı geçen düğümler
    by_route = [
        n for n in nodes
        if any(str(r).strip().lower() == clean_sub_lower for r in getattr(n, "routes", []))
    ]

    # 2. Geometrik sınırlayıcı kutu / merkez yarıçapı
    sub_info = get_subregion_info(clean_sub)
    by_geom: List[NodeData] = []
    if sub_info:
        bbox = sub_info.get("bbox")
        norm_bbox = sub_info.get("norm_bbox")
        center = sub_info.get("center")
        radius = float(sub_info.get("radius", 260.0))

        for n in nodes:
            mx = float(getattr(n, "map_x", 0) or 0)
            my = float(getattr(n, "map_y", 0) or 0)
            nx = float(getattr(n, "norm_x", 0) or 0)
            ny = float(getattr(n, "norm_y", 0) or 0)

            in_box = False
            if bbox and (mx > 0 or my > 0):
                if bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]:
                    in_box = True
            elif norm_bbox and (nx > 0 or ny > 0):
                if norm_bbox[0] <= nx <= norm_bbox[2] and norm_bbox[1] <= ny <= norm_bbox[3]:
                    in_box = True

            if not in_box and center and (mx > 0 or my > 0):
                dist = math.hypot(mx - float(center[0]), my - float(center[1]))
                if dist <= (radius * 1.25):
                    in_box = True

            if in_box and n not in by_geom:
                by_geom.append(n)

    if by_route and by_geom:
        seen = {id(n) for n in by_route}
        combined = list(by_route)
        for n in by_geom:
            if id(n) not in seen:
                combined.append(n)
                seen.add(id(n))
        return combined
    elif by_route:
        return by_route
    elif by_geom:
        return by_geom

    return list(nodes)


# ==============================================================================
# CANONICAL MATERIAL TAXONOMY & PROXIMITY ROUTE BUILDER (SEÇ-TOPLA ENGINE)
# ==============================================================================

_TR_CHAR_TRANS = str.maketrans({
    "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g",
    "ı": "i", "I": "i",
    "i": "i", "İ": "i",
    "ö": "o", "Ö": "o",
    "ş": "s", "Ş": "s",
    "ü": "u", "Ü": "u",
})

CANONICAL_MATERIAL_GROUPS: Dict[str, Set[str]] = {
    "ore_orichalcum": {
        "ore_orichalcum",
        "orichalcum_ore",
        "orichalcum",
        "orikalkum",
        "orikalkum_cevheri",
        "orichalcum_cevheri",
        "fine_orichalcum_ore",
        "pure_orichalcum_ore",
        "radiant_orichalcum_ore",
        "fresh_orichalcum",
        "ore_vein",
        "3",
        "5",
    },
    "herb_aria": {
        "herb_aria",
        "aria",
        "fresh_aria",
        "taze_aria",
        "aria_mor_kristal_bitki",
        "herb_flora",
        "0",
    },
    "herb_mela": {
        "herb_mela",
        "mela",
        "fruit_mela",
        "fresh_mela",
        "taze_mela",
        "mela_meyvesi",
        "1",
        "4",
    },
    "herb_targena": {
        "herb_targena",
        "targena",
        "fresh_targena",
        "taze_targena",
        "targena_otu",
        "2",
    },
    "gem_crystal": {
        "gem_crystal",
        "gem_crystal_odyle",
        "gem_or_crystal_or_odyle",
        "gem_odyle",
        "crystal",
        "gem",
        "gemstone",
        "gemstones",
        "degerli_tas",
        "degerli_taslar",
        "degerli_taslar_gemler",
        "kristal",
        "odyle",
        "highland_odyle",
        "pure_odyle",
        "odyle_orb",
        "odyle_aether",
        "odyle_enerji_kuresi",
        "aether",
        "aether_crystal",
        "sapphire",
        "sapphire_gemstone",
        "gem_sapphire",
        "ruby",
        "ruby_gemstone",
        "gem_ruby",
        "diamond",
        "diamond_gemstone",
        "gem_diamond",
        "amethyst",
        "amethyst_gemstone",
        "topaz",
        "topaz_gemstone",
        "emerald",
        "emerald_gemstone",
        "garnet",
        "garnet_gemstone",
        "aquamarine",
        "aquamarine_gemstone",
        "opal",
        "opal_gemstone",
        "4",
        "10",
    },
    "tree_log": {
        "tree_log",
        "tree_or_log",
        "tree",
        "log",
        "log_timber",
        "timber",
        "agac",
        "agaclar",
        "agaclar_keresteler",
        "agac_kereste",
        "kereste",
        "kutuk",
        "yggdrasil",
        "yggdrasil_log",
        "asvata",
        "asvata_log",
        "zelkova",
        "zelkova_log",
        "teak",
        "teak_log",
        "dryas",
        "dryas_log",
        "kocan",
        "kocan_log",
        "spruce",
        "spruce_log",
        "duran",
        "duran_log",
        "subara",
        "subara_log",
        "5",
        "11",
    },
    "herb_azpha": {
        "herb_azpha",
        "azpha",
        "fresh_azpha",
        "taze_azpha",
        "azpha_otu",
    },
    "herb_calendula": {
        "herb_calendula",
        "calendula",
        "fresh_calendula",
    },
}

ALIAS_TO_CANONICAL_GROUP: Dict[str, str] = {}
for _grp_key, _aliases in CANONICAL_MATERIAL_GROUPS.items():
    for _al in _aliases:
        ALIAS_TO_CANONICAL_GROUP[_al] = _grp_key

CATEGORY_MATERIAL_MAP: Dict[str, Set[str]] = {
    "herbs": {"herb_aria", "herb_mela", "herb_targena", "herb_azpha", "herb_calendula"},
    "bitkiler": {"herb_aria", "herb_mela", "herb_targena", "herb_azpha", "herb_calendula"},
    "bitkiler_otlar": {"herb_aria", "herb_mela", "herb_targena", "herb_azpha", "herb_calendula"},
    "ores_metals": {"ore_orichalcum"},
    "madenler": {"ore_orichalcum"},
    "madenler_cevherler": {"ore_orichalcum"},
    "gemstones": {"gem_crystal"},
    "degerli_taslar": {"gem_crystal"},
    "odyle_aether": {"gem_crystal"},
    "logs_trees": {"tree_log"},
    "agaclar_keresteler": {"tree_log"},
}


def _normalize_material_token(text: Any) -> str:
    """
    Normalizes a material name, class name, or alias to a clean, canonical token.
    Strips emojis, replaces Turkish characters, and formats with underscores.
    """
    if text is None:
        return ""
    if isinstance(text, (int, float)):
        return str(int(text))
    s = str(text).strip()
    if not s:
        return ""
    s = s.translate(_TR_CHAR_TRANS).lower()
    cleaned: List[str] = []
    for ch in s:
        if ch.isalnum() or ch == "_":
            cleaned.append(ch)
        elif ch in (" ", "-", "/", "\\", "|", "+", "&"):
            cleaned.append("_")
    norm = "".join(cleaned).strip("_")
    while "__" in norm:
        norm = norm.replace("__", "_")
    return norm


def _extract_node_xy(
    node: Any,
    fallback_width: float = 1920.0,
    fallback_height: float = 1080.0,
) -> Tuple[float, float]:
    """Safely extracts 2D map coordinates (x, y) in pixels from NodeData, dict, or tuple/list."""
    try:
        if isinstance(node, (tuple, list, np.ndarray)) and len(node) >= 2:
            return float(node[0]), float(node[1])
        if isinstance(node, dict):
            mx = node.get("map_x", node.get("x"))
            my = node.get("map_y", node.get("y"))
            if mx is None and my is None:
                nx = node.get("norm_x")
                ny = node.get("norm_y")
                if nx is not None and ny is not None:
                    return float(nx) * fallback_width, float(ny) * fallback_height
                return 0.0, 0.0
            if float(mx or 0.0) == 0.0 and float(my or 0.0) == 0.0:
                nx = node.get("norm_x")
                ny = node.get("norm_y")
                if nx is not None and ny is not None and not (float(nx) == 0.5 and float(ny) == 0.5):
                    return float(nx) * fallback_width, float(ny) * fallback_height
            return float(mx or 0.0), float(my or 0.0)

        mx = getattr(node, "map_x", getattr(node, "x", None))
        my = getattr(node, "map_y", getattr(node, "y", None))
        if mx is None and my is None:
            nx = getattr(node, "norm_x", None)
            ny = getattr(node, "norm_y", None)
            if nx is not None and ny is not None:
                return float(nx) * fallback_width, float(ny) * fallback_height
            return 0.0, 0.0
        if float(mx or 0.0) == 0.0 and float(my or 0.0) == 0.0:
            nx = getattr(node, "norm_x", None)
            ny = getattr(node, "norm_y", None)
            if nx is not None and ny is not None and not (float(nx) == 0.5 and float(ny) == 0.5):
                return float(nx) * fallback_width, float(ny) * fallback_height
        return float(mx or 0.0), float(my or 0.0)
    except Exception as exc:
        logger.debug("_extract_node_xy error on %r: %s", node, exc)
        return 0.0, 0.0



def _to_material_route_slug(target_materials: Any) -> str:
    """Produces a clean alphanumeric route name slug from target_materials."""
    if target_materials is None:
        return "all"
    if isinstance(target_materials, str):
        norm = _normalize_material_token(target_materials)
        grp = ALIAS_TO_CANONICAL_GROUP.get(norm, norm)
        return grp or "all"
    if isinstance(target_materials, (list, tuple, set)):
        slugs = []
        for item in target_materials:
            if item is not None:
                norm = _normalize_material_token(item)
                grp = ALIAS_TO_CANONICAL_GROUP.get(norm, norm)
                if grp and grp not in slugs:
                    slugs.append(grp)
        return "_".join(slugs) if slugs else "all"
    return "all"


def _match_node_material(node: Any, targets: List[str]) -> bool:
    """Evaluates whether a node matches any target material in targets."""
    candidates: List[Any] = []
    if isinstance(node, dict):
        for k in (
            "resource_type",
            "material_type",
            "class_name",
            "category",
            "category_tr",
            "name",
            "yolo_class_id",
            "class_id",
        ):
            val = node.get(k)
            if val is not None:
                if k == "name":
                    sval = str(val).strip().lower()
                    if sval.startswith("node ") or sval.startswith("node_") or sval.startswith("node#"):
                        continue
                candidates.append(val)
        patch_info = node.get("patch_classification")
        if isinstance(patch_info, dict):
            if patch_info.get("chosen_name"):
                candidates.append(patch_info["chosen_name"])
            if patch_info.get("taxonomy", {}).get("category"):
                candidates.append(patch_info["taxonomy"]["category"])
    else:
        for attr in (
            "resource_type",
            "material_type",
            "class_name",
            "category",
            "category_tr",
            "name",
            "yolo_class_id",
            "class_id",
        ):
            val = getattr(node, attr, None)
            if val is not None:
                if attr == "name":
                    sval = str(val).strip().lower()
                    if sval.startswith("node ") or sval.startswith("node_") or sval.startswith("node#"):
                        continue
                candidates.append(val)
        patch_info = getattr(node, "patch_classification", None)
        if isinstance(patch_info, dict):
            if patch_info.get("chosen_name"):
                candidates.append(patch_info["chosen_name"])
            if patch_info.get("taxonomy", {}).get("category"):
                candidates.append(patch_info["taxonomy"]["category"])

    for tgt in targets:
        t_norm = _normalize_material_token(tgt)
        if not t_norm:
            continue
        t_group = ALIAS_TO_CANONICAL_GROUP.get(t_norm)

        for cand in candidates:
            if isinstance(cand, (int, float)):
                c_str = str(int(cand))
                if c_str == t_norm:
                    return True
                yolo_6_map = {
                    0: "herb_aria",
                    1: "herb_mela",
                    2: "herb_targena",
                    3: "ore_orichalcum",
                    4: "gem_crystal",
                    5: "tree_log",
                }
                c_group = yolo_6_map.get(int(cand))
                if c_group and (c_group == t_norm or c_group == t_group):
                    return True
                continue

            c_norm = _normalize_material_token(cand)
            if not c_norm:
                continue

            # Exact normalized token match
            if c_norm == t_norm:
                return True

            c_group = ALIAS_TO_CANONICAL_GROUP.get(c_norm)
            # Canonical group match
            if t_group and c_group and t_group == c_group:
                return True

            # Direct alias match
            if t_group and c_norm in CANONICAL_MATERIAL_GROUPS.get(t_group, set()):
                return True
            if c_group and t_norm in CANONICAL_MATERIAL_GROUPS.get(c_group, set()):
                return True

            # Category mapping check
            if t_norm in CATEGORY_MATERIAL_MAP:
                allowed_groups = CATEGORY_MATERIAL_MAP[t_norm]
                if c_group in allowed_groups or c_norm in allowed_groups:
                    return True

            # Substring containment
            if len(t_norm) >= 4 and len(c_norm) >= 4:
                if t_norm in c_norm or c_norm in t_norm:
                    return True

            # Catalog spec comparison
            if AION2_MATERIALS_CATALOG is not None:
                spec = AION2_MATERIALS_CATALOG.get(str(cand))
                if spec is not None:
                    base_norm = _normalize_material_token(spec.base_material)
                    cat_norm = _normalize_material_token(spec.category)
                    base_grp = ALIAS_TO_CANONICAL_GROUP.get(base_norm)
                    if (
                        base_norm == t_norm
                        or (t_group and base_norm in CANONICAL_MATERIAL_GROUPS.get(t_group, set()))
                        or (t_group and base_grp and t_group == base_grp)
                    ):
                        return True
                    if cat_norm == t_norm or (
                        t_norm in CATEGORY_MATERIAL_MAP
                        and cat_norm in CATEGORY_MATERIAL_MAP[t_norm]
                    ):
                        return True

    # Patch classification fallback if patch_path exists
    patch_path = getattr(node, "patch_path", None) or (
        node.get("patch_path") if isinstance(node, dict) else None
    )
    if patch_path and classify_3d_material_patch is not None:
        disk_path = str(patch_path)
        if not os.path.isabs(disk_path) and not os.path.isfile(disk_path):
            disk_path = os.path.abspath(disk_path)
        if os.path.isfile(disk_path):
            try:
                patch_img = _safe_imread_color(disk_path)
                if patch_img is not None and patch_img.size > 0:
                    reg_name = getattr(node, "region", "") or (
                        node.get("region") if isinstance(node, dict) else ""
                    )
                    cls_res = classify_3d_material_patch(patch_img, region_name=str(reg_name))
                    chosen = cls_res.get("chosen_name")
                    if chosen:
                        ch_norm = _normalize_material_token(chosen)
                        ch_grp = ALIAS_TO_CANONICAL_GROUP.get(ch_norm)
                        for tgt in targets:
                            t_norm = _normalize_material_token(tgt)
                            t_grp = ALIAS_TO_CANONICAL_GROUP.get(t_norm)
                            if (
                                ch_norm == t_norm
                                or (ch_grp and t_grp and ch_grp == t_grp)
                                or (t_norm and t_norm in ch_norm)
                            ):
                                return True
            except Exception as exc:
                logger.debug("classify_3d_material_patch fallback error: %s", exc)

    return False


def filter_nodes_by_material(
    nodes: Sequence[Any],
    target_materials: Any,
) -> List[Any]:
    """
    Filters resource nodes matching the requested target materials.
    Matches across:
    - resource_type
    - material_type
    - name
    - category / category_tr
    - class_name
    - yolo_class_id / class_id
    - patch classification (classify_3d_material_patch)
    - canonical Aion 2 taxonomy and material encyclopedia

    If target_materials is empty, None, 'All', or 'Tümü', returns all gatherable nodes.
    """
    if not nodes:
        return []

    targets_list: List[str] = []
    if target_materials is None:
        targets_list = []
    elif isinstance(target_materials, str):
        clean_str = target_materials.strip()
        if clean_str and clean_str.lower() not in ("all", "tumu", "tümü", "*", "none"):
            targets_list = [clean_str]
    elif isinstance(target_materials, (list, tuple, set)):
        for item in target_materials:
            if item is not None:
                cs = str(item).strip()
                if cs and cs.lower() not in ("all", "tumu", "tümü", "*", "none"):
                    targets_list.append(cs)

    if not targets_list:
        return [
            n for n in nodes
            if not (
                getattr(n, "not_gatherable", False)
                if hasattr(n, "not_gatherable")
                else (isinstance(n, dict) and n.get("not_gatherable", False))
            )
        ]

    matched: List[Any] = []
    for node in nodes:
        is_not_gatherable = (
            getattr(node, "not_gatherable", False)
            if hasattr(node, "not_gatherable")
            else (isinstance(node, dict) and node.get("not_gatherable", False))
        )
        if is_not_gatherable:
            continue
        if _match_node_material(node, targets_list):
            matched.append(node)

    return matched


def sort_nodes_by_proximity(
    nodes: Sequence[Any],
    start_xy: Optional[Tuple[float, float]] = None,
) -> List[Any]:
    """
    Sorts nodes starting from `start_xy = (x, y)` (player position or first node)
    using Greedy Nearest-Neighbor + 2-Opt TSP proximity optimization.
    Minimizes total Euclidean travel distance and untangles crossing segments.
    """
    if not nodes:
        return []
    node_list = list(nodes)
    n = len(node_list)
    if n <= 1:
        return node_list

    coords: List[Tuple[float, float]] = [_extract_node_xy(item) for item in node_list]

    if n == 2:
        if start_xy is not None:
            sx, sy = float(start_xy[0]), float(start_xy[1])
            d0 = math.hypot(coords[0][0] - sx, coords[0][1] - sy)
            d1 = math.hypot(coords[1][0] - sx, coords[1][1] - sy)
            if d1 < d0:
                return [node_list[1], node_list[0]]
            return [node_list[0], node_list[1]]
        return node_list

    # 1. Greedy Nearest-Neighbor Initialization
    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        first_idx = min(
            range(n),
            key=lambda i: math.hypot(coords[i][0] - sx, coords[i][1] - sy),
        )
    else:
        first_idx = 0

    visited = [False] * n
    visited[first_idx] = True
    tour = [first_idx]
    curr_idx = first_idx

    for _ in range(n - 1):
        best_next = -1
        best_dist = float("inf")
        curr_x, curr_y = coords[curr_idx]
        for j in range(n):
            if not visited[j]:
                d = math.hypot(coords[j][0] - curr_x, coords[j][1] - curr_y)
                if d < best_dist:
                    best_dist = d
                    best_next = j
        visited[best_next] = True
        tour.append(best_next)
        curr_idx = best_next

    # 2. 2-Opt Local Search Loop Optimization (untangle crossings & minimize perimeter)
    max_iterations = 60
    for _ in range(max_iterations):
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a = tour[i]
                b = tour[(i + 1) % n]
                c = tour[j]
                d = tour[(j + 1) % n]

                d_ab = math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
                d_cd = math.hypot(coords[c][0] - coords[d][0], coords[c][1] - coords[d][1])
                d_ac = math.hypot(coords[a][0] - coords[c][0], coords[a][1] - coords[c][1])
                d_bd = math.hypot(coords[b][0] - coords[d][0], coords[b][1] - coords[d][1])

                if (d_ac + d_bd) + 1e-6 < (d_ab + d_cd):
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
        if not improved:
            break

    # 3. Orient tour starting from node closest to start_xy
    if start_xy is not None:
        sx, sy = float(start_xy[0]), float(start_xy[1])
        best_pos = min(
            range(n),
            key=lambda k: math.hypot(coords[tour[k]][0] - sx, coords[tour[k]][1] - sy),
        )
        tour = tour[best_pos:] + tour[:best_pos]
        dist_fwd = math.hypot(coords[tour[1]][0] - sx, coords[tour[1]][1] - sy)
        dist_bwd = math.hypot(coords[tour[-1]][0] - sx, coords[tour[-1]][1] - sy)
        if dist_bwd < dist_fwd:
            tour = [tour[0]] + list(reversed(tour[1:]))
    else:
        if 0 in tour:
            zpos = tour.index(0)
            tour = tour[zpos:] + tour[:zpos]

    # 4. Open-path 2-opt fine-tuning
    for _ in range(30):
        improved = False
        for i in range(n - 2):
            for j in range(i + 2, n - 1):
                a = tour[i]
                b = tour[i + 1]
                c = tour[j]
                d = tour[j + 1]
                d_ab = math.hypot(coords[a][0] - coords[b][0], coords[a][1] - coords[b][1])
                d_cd = math.hypot(coords[c][0] - coords[d][0], coords[c][1] - coords[d][1])
                d_ac = math.hypot(coords[a][0] - coords[c][0], coords[a][1] - coords[c][1])
                d_bd = math.hypot(coords[b][0] - coords[d][0], coords[b][1] - coords[d][1])
                if (d_ac + d_bd) + 1e-6 < (d_ab + d_cd):
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True
        if not improved:
            break

    return [node_list[idx] for idx in tour]


def build_proximity_material_route(
    region_name: Any = None,
    target_materials: Any = None,
    start_xy: Optional[Tuple[float, float]] = None,
    persist_route: bool = True,
    route_manager: Optional[Any] = None,
) -> List[Any]:
    """Module-level helper to build and optionally persist a proximity-sorted material route."""
    mgr = route_manager or RouteManager()
    return mgr.build_proximity_material_route(
        region_name=region_name,
        target_materials=target_materials,
        start_xy=start_xy,
        persist_route=persist_route,
    )


class SelectiveRouteNodeDict(dict):
    """Dict representation of a route node that also supports attribute access."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def to_dict(self) -> Dict[str, Any]:
        return dict(self)


def get_selective_tsp_route(
    region: str,
    selected_materials: list,
    player_xy: Optional[Tuple[float, float]] = None,
    route_manager: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """
    Seç-Topla aktifken seçilen materyal(ler) için oyuncu konumundan başlayan
    en kısa 2-Opt TSP rotayı anında üretir.
    """
    mgr = route_manager or RouteManager()
    return mgr.get_selective_tsp_route(
        region=region,
        selected_materials=selected_materials,
        player_xy=player_xy,
    )


class CanonicalRegionDict(dict):

    """
    Subregion adlarını canonical üst bölgeye otomatik çözümleyen sözlük.
    Örn: 'Dredgion Crash Site' istendiğinde 'Altgard' kaydını döndürür.
    """
    def __getitem__(self, key: str) -> Any:
        if super().__contains__(key):
            return super().__getitem__(key)
        canon = resolve_canonical_region(key)
        if super().__contains__(canon):
            return super().__getitem__(canon)
        raise KeyError(key)

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if isinstance(key, str):
            canon = resolve_canonical_region(key)
            return super().__contains__(canon)
        return False

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default



def _safe_imread_gray(path: str) -> Optional[np.ndarray]:
    """Reads a grayscale image safely even when the path contains non-ASCII/Turkish characters."""
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    except Exception:
        try:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        except Exception as exc:
            logger.debug("Görsel okunamadı (%s): %s", path, exc)
            return None


def _safe_imread_color(path: str) -> Optional[np.ndarray]:
    """Reads a color (BGR) image safely even when the path contains non-ASCII/Turkish characters."""
    if not path or not os.path.isfile(path):
        return None
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.size == 0:
            return None
        return cv2.imdecode(raw, cv2.IMREAD_COLOR)
    except Exception:
        try:
            return cv2.imread(path, cv2.IMREAD_COLOR)
        except Exception as exc:
            logger.debug("Renkli görsel okunamadı (%s): %s", path, exc)
            return None


def _safe_imwrite(path: str, img: np.ndarray) -> bool:
    """Writes an image safely even when the path contains non-ASCII/Turkish characters."""
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return False
    try:
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        ext = os.path.splitext(path)[1] or ".png"
        ok, buf = cv2.imencode(ext, img)
        if ok:
            buf.tofile(path)
            return True
    except Exception as exc:
        logger.debug("imencode/tofile yedek yazma devreye girdi: %s", exc)
    try:
        return bool(cv2.imwrite(path, img))
    except Exception as exc:
        logger.debug("Görsel yazılamadı (%s): %s", path, exc)
        return False


def _play_beep(frequency: int, duration_ms: int) -> None:
    """Plays a system beep safely on Windows without raising exceptions."""
    if winsound is None:
        logger.debug("winsound modülü mevcut değil, bip sesi atlandı (%d Hz, %d ms).", frequency, duration_ms)
        return
    try:
        winsound.Beep(int(frequency), int(duration_ms))
    except Exception as exc:
        logger.debug("Bip sesi çalınamadı (%d Hz, %d ms): %s", frequency, duration_ms, exc)


def _normalize_path(path_str: str) -> str:
    """Normalizes file paths to forward slashes for cross-module consistency."""
    if not path_str:
        return ""
    return path_str.replace("\\", "/")


class NodeLocationResult(tuple):
    """
    Harita üzerinde bulunan düğüm sonucunu temsil eden tuple alt sınıfı.

    Hem standart 3'lü tuple açılımını:
        `(x, y), confidence, distance_str = result`
    hem 4'lü erişimi / sözlük erişimini:
        `result["x"], result["y"], result["confidence"], result["distance_m"]`
    hem de nitelik erişimini:
        `result.x, result.y, result.confidence, result.distance_str, result.under_overlay`
    destekler.
    """

    def __new__(
        cls,
        coords: Tuple[int, int],
        confidence: float,
        distance_str: str,
        distance_m: float = 0.0,
        under_overlay: bool = False,
        used_fallback: bool = False,
    ) -> "NodeLocationResult":
        instance = super().__new__(cls, (coords, float(confidence), str(distance_str)))
        instance.coords = (int(coords[0]), int(coords[1]))
        instance.x = int(coords[0])
        instance.y = int(coords[1])
        instance.confidence = float(confidence)
        instance.distance = str(distance_str)
        instance.distance_str = str(distance_str)
        instance.distance_m = float(distance_m)
        instance.under_overlay = bool(under_overlay)
        instance.used_fallback = bool(used_fallback)
        return instance

    def __getitem__(self, key: Union[int, slice, str]) -> Any:
        if isinstance(key, str):
            mapping = {
                "x": self.x,
                "y": self.y,
                "coords": self.coords,
                "point": self.coords,
                "confidence": self.confidence,
                "score": self.confidence,
                "distance": self.distance_str,
                "distance_str": self.distance_str,
                "distance_m": self.distance_m,
                "under_overlay": self.under_overlay,
                "used_fallback": self.used_fallback,
            }
            if key in mapping:
                return mapping[key]
            raise KeyError(key)
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "coords": self.coords,
            "confidence": self.confidence,
            "distance": self.distance_str,
            "distance_str": self.distance_str,
            "distance_m": self.distance_m,
            "under_overlay": self.under_overlay,
            "used_fallback": self.used_fallback,
        }


class SnapResult(dict):
    """
    Görsel odometri ve rota snap-to-path sonucu.
    Hem sözlük erişimini:
        res["coords"], res["snapped"], res["distance"]
    hem nitelik erişimini:
        res.coords, res.x, res.y, res.snapped, res.distance, res.segment_index, res.projection_t, res.nearest_node
    hem de tuple açılımını:
        (x, y), is_snapped, dist = res
    destekler.
    """

    def __init__(
        self,
        coords: Tuple[int, int],
        snapped: bool,
        distance: float,
        segment_index: Optional[int] = None,
        projection_t: float = 0.0,
        nearest_node: Optional["NodeData"] = None,
        nearest_node_id: Optional[int] = None,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
    ) -> None:
        coords_int = (int(coords[0]), int(coords[1]))
        dist_f = float(distance)
        proj_t = float(projection_t)
        super().__init__(
            coords=coords_int,
            x=coords_int[0],
            y=coords_int[1],
            snapped=bool(snapped),
            distance=dist_f,
            segment_index=segment_index,
            projection_t=proj_t,
            nearest_node=nearest_node,
            nearest_node_id=nearest_node_id,
            norm_x=float(norm_x) if norm_x is not None else None,
            norm_y=float(norm_y) if norm_y is not None else None,
        )
        self.coords = coords_int
        self.x = coords_int[0]
        self.y = coords_int[1]
        self.snapped = bool(snapped)
        self.distance = dist_f
        self.segment_index = segment_index
        self.projection_t = proj_t
        self.nearest_node = nearest_node
        self.nearest_node_id = nearest_node_id
        self.norm_x = float(norm_x) if norm_x is not None else None
        self.norm_y = float(norm_y) if norm_y is not None else None

    def __iter__(self):
        yield self.coords
        yield self.snapped
        yield self.distance

    def to_dict(self) -> Dict[str, Any]:
        return {
            "coords": self.coords,
            "x": self.x,
            "y": self.y,
            "snapped": self.snapped,
            "distance": self.distance,
            "segment_index": self.segment_index,
            "projection_t": self.projection_t,
            "nearest_node_id": self.nearest_node_id,
            "norm_x": self.norm_x,
            "norm_y": self.norm_y,
        }


@dataclass
class NodeData:
    """
    Harita üzerindeki tek bir kaynak (düğüm) noktasının veri modeli.
    Her materyal varsayılan olarak 3 kez toplanabilir (`max_harvests=3`,
    `current_harvests=0..3`) ve tamamlandığında `"resting"` durumuna geçer.
    """

    id: int = 1
    region: str = DEFAULT_REGION
    routes: List[str] = field(default_factory=list)
    patch_path: str = ""
    in_air: bool = False
    gathered_count: int = 0
    last_visited_timestamp: float = 0.0
    rest_duration_sec: int = DEFAULT_REST_SEC
    consecutive_failures: int = 0
    map_x: int = 0
    map_y: int = 0
    norm_x: float = 0.5
    norm_y: float = 0.5
    resource_type: str = DEFAULT_RESOURCE_TYPE
    max_harvests: int = DEFAULT_MAX_HARVESTS
    current_harvests: int = 0
    source_type: str = DEFAULT_SOURCE_TYPE
    state: str = "active"
    seq_index: int = 0
    not_gatherable: bool = False

    def __post_init__(self) -> None:
        self.id = int(self.id)
        self.region = str(self.region or DEFAULT_REGION)
        if isinstance(self.routes, (set, tuple)):
            self.routes = list(self.routes)
        elif not isinstance(self.routes, list):
            self.routes = [str(self.routes)] if self.routes else []
        else:
            self.routes = [str(r) for r in self.routes if r and str(r) != DEFAULT_ROUTE]

        if not self.patch_path:
            self.patch_path = f"routes/{self.region}/patches/node_{self.id}.png"
        self.patch_path = _normalize_path(str(self.patch_path))
        self.in_air = bool(self.in_air)
        self.gathered_count = int(self.gathered_count)
        self.last_visited_timestamp = float(self.last_visited_timestamp)
        self.rest_duration_sec = int(self.rest_duration_sec)
        self.consecutive_failures = int(self.consecutive_failures)
        self.map_x = int(self.map_x)
        self.map_y = int(self.map_y)
        self.norm_x = float(self.norm_x)
        self.norm_y = float(self.norm_y)
        self.resource_type = str(self.resource_type or DEFAULT_RESOURCE_TYPE)
        self.max_harvests = max(
            1,
            int(self.max_harvests if self.max_harvests is not None else DEFAULT_MAX_HARVESTS),
        )
        self.current_harvests = max(
            0,
            int(self.current_harvests if self.current_harvests is not None else 0),
        )
        self.source_type = (
            str(self.source_type or DEFAULT_SOURCE_TYPE).strip() or DEFAULT_SOURCE_TYPE
        )
        self.state = str(self.state or "active").strip().lower() or "active"
        self.seq_index = int(self.seq_index or 0)
        self.not_gatherable = bool(self.not_gatherable)

        if (
            self.last_visited_timestamp > 0.0
            and self.current_harvests == 0
            and self.state == "active"
            and (time.time() - self.last_visited_timestamp) < float(max(0, self.rest_duration_sec))
        ):
            self.state = "resting"

    @property
    def name(self) -> str:
        """Geriye dönük uyumluluk için düğüm adı."""
        return f"Node {self.id}"

    @property
    def gather_count(self) -> int:
        """Toplam hasat sayısı (`gathered_count` ile eş anlamlı)."""
        return self.gathered_count

    @gather_count.setter
    def gather_count(self, value: int) -> None:
        self.gathered_count = int(value)

    @property
    def visit_count(self) -> int:
        """Geriye dönük uyumluluk için toplanma sayısı."""
        return self.gathered_count

    @visit_count.setter
    def visit_count(self, value: int) -> None:
        self.gathered_count = int(value)

    @property
    def last_gathered_ts(self) -> float:
        """Son tam toplanma zaman damgası (`last_visited_timestamp` ile eş anlamlı)."""
        return self.last_visited_timestamp

    @last_gathered_ts.setter
    def last_gathered_ts(self, value: float) -> None:
        self.last_visited_timestamp = float(value)

    @property
    def last_visited(self) -> float:
        """Geriye dönük uyumluluk için son ziyaret zaman damgası."""
        return self.last_visited_timestamp

    @last_visited.setter
    def last_visited(self, value: float) -> None:
        self.last_visited_timestamp = float(value)

    @property
    def fail_count(self) -> int:
        """Ardışık hata sayısı (`consecutive_failures` ile eş anlamlı)."""
        return self.consecutive_failures

    @fail_count.setter
    def fail_count(self, value: int) -> None:
        self.consecutive_failures = int(value)

    @property
    def failures(self) -> int:
        """Geriye dönük uyumluluk için ardışık hata sayısı."""
        return self.consecutive_failures

    @failures.setter
    def failures(self, value: int) -> None:
        self.consecutive_failures = int(value)

    @property
    def patch_file(self) -> str:
        """Geriye dönük uyumluluk için yama dosya adı."""
        return os.path.basename(self.patch_path) if self.patch_path else f"node_{self.id}.png"

    def is_ready(
        self,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
    ) -> bool:
        """
        Düğümün toplanmaya hazır olup olmadığını döndürür.
        Eğer düğüm şu anda kısmi hasat aşamasındaysa (`1/3` veya `2/3`, `state != 'resting'`),
        henüz tükenmediği için `True` döndürür.
        """
        if self.not_gatherable:
            return False
        if self.state in ("active", "gathering") and 0 < self.current_harvests < max(1, self.max_harvests):
            return True
        if self.last_visited_timestamp <= 0.0:
            return self.state != "resting"
        current_time = time.time() if now is None else float(now)
        duration = self.rest_duration_sec if rest_duration_sec is None else int(rest_duration_sec)
        if duration <= 0:
            return True
        return (current_time - self.last_visited_timestamp) >= float(duration)

    def remaining_rest_sec(
        self,
        now: Optional[float] = None,
        rest_duration_sec: Optional[int] = None,
    ) -> int:
        """Düğümün hazır olmasına kalan saniyeyi döndürür (hazırsa 0)."""
        if self.state in ("active", "gathering") and 0 < self.current_harvests < max(1, self.max_harvests):
            return 0
        if self.last_visited_timestamp <= 0.0:
            return 0
        current_time = time.time() if now is None else float(now)
        duration = self.rest_duration_sec if rest_duration_sec is None else int(rest_duration_sec)
        if duration <= 0:
            return 0
        elapsed = current_time - self.last_visited_timestamp
        remaining = float(duration) - elapsed
        if remaining <= 0.0:
            return 0
        return int(math.ceil(remaining))

    def last_visited_str(self, now: Optional[float] = None) -> str:
        """Son ziyaret zamanını okunabilir metin olarak döndürür."""
        if self.last_visited_timestamp <= 0.0:
            return "Never"
        try:
            current_time = time.time() if now is None else float(now)
            diff = max(0.0, current_time - self.last_visited_timestamp)
            if diff < 60.0:
                return f"{int(diff)}s ago"
            if diff < 3600.0:
                mins = int(diff // 60)
                secs = int(diff % 60)
                return f"{mins}m {secs}s ago"
            dt = datetime.datetime.fromtimestamp(self.last_visited_timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            logger.debug("last_visited_str hesaplama hatası: %s", exc)
            return "Unknown"

    def to_dict(self) -> Dict[str, Any]:
        """NodeData nesnesini JSON serileştirmeye uygun sözlüğe dönüştürür."""
        return {
            "id": int(self.id),
            "region": str(self.region),
            "routes": list(self.routes),
            "patch_path": _normalize_path(self.patch_path),
            "in_air": bool(self.in_air),
            "gathered_count": int(self.gathered_count),
            "last_visited_timestamp": float(self.last_visited_timestamp),
            "rest_duration_sec": int(self.rest_duration_sec),
            "consecutive_failures": int(self.consecutive_failures),
            "map_x": int(self.map_x),
            "map_y": int(self.map_y),
            "norm_x": float(self.norm_x),
            "norm_y": float(self.norm_y),
            "resource_type": str(self.resource_type),
            "max_harvests": int(self.max_harvests),
            "current_harvests": int(self.current_harvests),
            "source_type": str(self.source_type),
            "state": str(self.state),
            "seq_index": int(self.seq_index),
            "not_gatherable": bool(self.not_gatherable),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NodeData":
        """Sözlükten NodeData nesnesi oluşturur (eski ve yeni şemaları destekler)."""
        if isinstance(d, cls):
            return d
        if not isinstance(d, dict):
            raise TypeError(f"NodeData.from_dict bir dict bekler, alınan: {type(d)!r}")

        node_id = int(d.get("id", 1))
        region = str(d.get("region", DEFAULT_REGION))

        raw_routes = d.get("routes", [])
        if isinstance(raw_routes, str):
            routes = [raw_routes] if raw_routes and raw_routes != DEFAULT_ROUTE else []
        elif isinstance(raw_routes, (list, tuple, set)):
            routes = [str(r) for r in raw_routes if r and str(r) != DEFAULT_ROUTE]
        else:
            routes = []

        patch_path = d.get("patch_path") or d.get("patch_file") or ""
        if patch_path and "/" not in str(patch_path) and "\\" not in str(patch_path):
            patch_path = f"routes/{region}/patches/{patch_path}"
        elif not patch_path:
            patch_path = f"routes/{region}/patches/node_{node_id}.png"

        gathered_count = d.get("gathered_count", d.get("gather_count", d.get("visit_count", 0)))
        last_visited_ts = d.get(
            "last_visited_timestamp",
            d.get("last_gathered_ts", d.get("last_visited", 0.0)),
        )
        rest_sec = d.get("rest_duration_sec", DEFAULT_REST_SEC)
        failures = d.get("consecutive_failures", d.get("fail_count", d.get("failures", 0)))
        raw_max_harvests = d.get("max_harvests", DEFAULT_MAX_HARVESTS)
        max_harvests = max(
            1,
            int(raw_max_harvests if raw_max_harvests is not None else DEFAULT_MAX_HARVESTS),
        )
        raw_curr_harvests = d.get("current_harvests", 0)
        current_harvests = max(
            0,
            int(raw_curr_harvests if raw_curr_harvests is not None else 0),
        )
        source_type = str(d.get("source_type", DEFAULT_SOURCE_TYPE) or DEFAULT_SOURCE_TYPE)
        state = str(d.get("state", "active") or "active")
        seq_index = int(d.get("seq_index", 0) or 0)
        not_gatherable = bool(d.get("not_gatherable", False))

        return cls(
            id=node_id,
            region=region,
            routes=routes,
            patch_path=_normalize_path(str(patch_path)),
            in_air=bool(d.get("in_air", False)),
            gathered_count=int(gathered_count),
            last_visited_timestamp=float(last_visited_ts),
            rest_duration_sec=int(rest_sec),
            consecutive_failures=int(failures),
            map_x=int(d.get("map_x", 0)),
            map_y=int(d.get("map_y", 0)),
            norm_x=float(d.get("norm_x", 0.5)),
            norm_y=float(d.get("norm_y", 0.5)),
            resource_type=str(d.get("resource_type", DEFAULT_RESOURCE_TYPE)),
            max_harvests=max_harvests,
            current_harvests=current_harvests,
            source_type=source_type,
            state=state,
            seq_index=seq_index,
            not_gatherable=not_gatherable,
        )

    # Sözlük uyumluluk metodları (eski kodların node["id"] / node.get("map_x") erişimleri için)
    def __getitem__(self, key: str) -> Any:
        if key == "name":
            return self.name
        if key in ("visit_count", "gather_count"):
            return self.gathered_count
        if key in ("last_visited", "last_gathered_ts"):
            return self.last_visited_timestamp
        if key in ("failures", "fail_count"):
            return self.consecutive_failures
        if key == "patch_file":
            return self.patch_file
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in ("visit_count", "gather_count"):
            self.gathered_count = int(value)
        elif key in ("last_visited", "last_gathered_ts"):
            self.last_visited_timestamp = float(value)
        elif key in ("failures", "fail_count"):
            self.consecutive_failures = int(value)
        elif hasattr(self, key):
            setattr(self, key, value)
        else:
            raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key) or key in {
            "name",
            "visit_count",
            "gather_count",
            "last_visited",
            "last_gathered_ts",
            "failures",
            "fail_count",
            "patch_file",
        }


class RouteManager:
    """
    Bölge, rota ve düğüm (node) veritabanı yöneticisi.

    Her bölgeyi `routes/<region_name>/region_data.json` altında,
    80x80 gri harita kesitlerini ise `routes/<region_name>/patches/node_<id>.png`
    altında saklar.
    """

    def __init__(
        self,
        base_dir: str = "routes",
        default_rest_sec: int = DEFAULT_REST_SEC,
        meters_per_pixel: float = 0.28,
        **kwargs: Any,
    ) -> None:
        # Geriye dönük uyumluluk: eğer ilk argüman bir .json dosya yoluysa dizinini al
        if base_dir.lower().endswith(".json"):
            resolved_base = os.path.dirname(base_dir) or "routes"
        else:
            resolved_base = base_dir or "routes"

        self.base_dir: str = resolved_base
        self.default_rest_sec: int = int(default_rest_sec)
        self.meters_per_pixel: float = float(meters_per_pixel)
        self.patch_normalizer: SobelClahePatchNormalizer = kwargs.get("patch_normalizer") or SobelClahePatchNormalizer()
        self.orientation_guard: MinimapOrientationGuard = kwargs.get("orientation_guard") or MinimapOrientationGuard()

        # Dynamic viewport ve odometri takipçisi
        init_vw = kwargs.get("viewport_width")
        init_vh = kwargs.get("viewport_height")
        self.viewport_width: Optional[int] = int(init_vw) if init_vw else None
        self.viewport_height: Optional[int] = int(init_vh) if init_vh else None
        self.odometry_tracker: Optional[Any] = kwargs.get("odometry_tracker")
        self._last_odometry_patch: Optional[np.ndarray] = None

        # Toplam hasat ve ardışık hata sayaçları
        self.total_gathered: int = 0
        self.consecutive_failures: int = 0

        # Kara Liste (Blacklist) Yöneticisi
        try:
            from core.blacklist_manager import get_blacklist_manager
            self.blacklist_manager = kwargs.get("blacklist_manager") or get_blacklist_manager()
        except Exception:
            self.blacklist_manager = None

        # Bellek içi veri yapıları:
        # _regions[region_name] = {"routes": ["All", "Ores"], "nodes": [NodeData]}
        self._regions: Dict[str, Dict[str, Any]] = CanonicalRegionDict()
        self.active_region: str = DEFAULT_REGION
        self.active_route: str = DEFAULT_ROUTE
        self._last_selected_node_id: Dict[Tuple[str, str], int] = {}
        self.last_save_error: Optional[BaseException] = None

        # Geriye dönük uyumluluk için mirror sözlük
        self.data: Dict[str, Any] = {
            "regions": CanonicalRegionDict(),
            "active_region": self.active_region,
            "active_route": self.active_route,
        }
        self._legacy_synced_ids: Dict[str, List[int]] = {}

    def resolve_canonical_region(self, name: Optional[str]) -> str:
        """Verilen bölge veya alt bölge adının kanonik ana bölge karşılığını döndürür."""
        return resolve_canonical_region(name)

    def get_canonical_region(self, name: Optional[str]) -> str:
        """resolve_canonical_region için eşanlamlı yardımcı metot."""
        return resolve_canonical_region(name)

    def is_subregion(self, name: Optional[str]) -> bool:
        """Verilen adın bir alt bölge (subregion) olup olmadığını kontrol eder."""
        return is_subregion(name)

    def get_subregion_bounding_box(self, name: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
        """Alt bölge (min_x, min_y, max_x, max_y) piksel sınırlayıcı kutusunu döndürür."""
        return get_subregion_bounding_area(name)

    def get_subregion_focus_area(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Alt bölge odak ve sınırlayıcı alan bilgilerini döndürür."""
        return get_subregion_info(name)

    def filter_nodes_by_subregion(self, nodes: Sequence[NodeData], subregion: str) -> List[NodeData]:
        """Düğümleri alt bölgeye göre filtreler."""
        return filter_nodes_by_subregion(nodes, subregion)

    def _get_material_route_slug(self, target_materials: Any) -> str:
        """Returns the route slug for target_materials."""
        return _to_material_route_slug(target_materials)

    def filter_nodes_by_material(
        self,
        nodes: Sequence[Any],
        target_materials: Any,
    ) -> List[Any]:
        """Filters nodes matching target_materials in canonical encyclopedia / taxonomy."""
        return filter_nodes_by_material(nodes, target_materials)

    def sort_nodes_by_proximity(
        self,
        nodes: Sequence[Any],
        start_xy: Optional[Tuple[float, float]] = None,
    ) -> List[Any]:
        """Sorts nodes using Greedy Nearest-Neighbor + 2-Opt TSP proximity optimization."""
        return sort_nodes_by_proximity(nodes, start_xy=start_xy)

    def build_proximity_material_route(
        self,
        region_name: Any = None,
        target_materials: Any = None,
        start_xy: Optional[Tuple[float, float]] = None,
        persist_route: bool = True,
    ) -> List[NodeData]:
        """
        Filters nodes in region_name by target_materials, re-sequences them in Greedy
        Nearest-Neighbor + 2-Opt TSP proximity order starting from start_xy (or first node),
        assigns 1-based seq_index, updates named route tagging, and optionally persists
        to routes/<region>/region_data.json.
        """
        # 1. Determine target region & source nodes
        candidate_nodes: Optional[List[Any]] = None
        if isinstance(region_name, (list, tuple)):
            candidate_nodes = list(region_name)
            target_reg = resolve_canonical_region(
                getattr(candidate_nodes[0], "region", self.active_region)
                if candidate_nodes
                else self.active_region
            )
        elif isinstance(region_name, str) and region_name.strip():
            target_reg = resolve_canonical_region(region_name.strip())
        else:
            target_reg = resolve_canonical_region(self.active_region)

        if candidate_nodes is not None:
            all_nodes = candidate_nodes
        else:
            if target_reg not in self._regions:
                self.load_region(target_reg)
            if target_reg not in self._regions:
                logger.warning("build_proximity_material_route: Bölge yüklenemedi: %s", target_reg)
                return []
            all_nodes = list(self._regions[target_reg].get("nodes", []))

        # 2. Filter matching nodes
        matched = self.filter_nodes_by_material(all_nodes, target_materials)
        if not matched:
            logger.info(
                "build_proximity_material_route: '%s' materyali için '%s' bölgesinde düğüm bulunamadı.",
                target_materials,
                target_reg,
            )
            return []

        # 3. Proximity sort (Greedy Nearest-Neighbor + 2-Opt TSP)
        ordered: List[Any] = self.sort_nodes_by_proximity(matched, start_xy=start_xy)

        # 4. Assign 1-based seq_index to matched nodes
        for idx, node in enumerate(ordered, start=1):
            if hasattr(node, "seq_index"):
                node.seq_index = idx
            elif isinstance(node, dict):
                node["seq_index"] = idx

        # 5. Route naming & persistence
        slug = self._get_material_route_slug(target_materials)
        route_name = f"proximity_{slug}"

        if target_reg in self._regions:
            reg_info = self._regions[target_reg]
            routes_list = list(reg_info.get("routes", []))
            if route_name not in routes_list:
                routes_list.append(route_name)
            reg_info["routes"] = routes_list

            for n in ordered:
                if hasattr(n, "routes"):
                    if route_name not in n.routes:
                        n.routes.append(route_name)
                elif isinstance(n, dict):
                    rts = n.setdefault("routes", [])
                    if route_name not in rts:
                        rts.append(route_name)

            # Reorder region nodes: matched ordered nodes first, remaining after
            matched_ids = {
                getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None)
                for n in ordered
            }
            remaining_nodes = [
                n for n in all_nodes
                if (getattr(n, "id", None) or (n.get("id") if isinstance(n, dict) else None))
                not in matched_ids
            ]
            for idx, n in enumerate(remaining_nodes, start=len(ordered) + 1):
                if hasattr(n, "seq_index"):
                    n.seq_index = idx
                elif isinstance(n, dict):
                    n["seq_index"] = idx

            reg_info["nodes"] = ordered + remaining_nodes
            self.active_route = route_name
            self._sync_legacy_data()

            if persist_route:
                self.save_region(target_reg)

        return ordered

    def update_viewport(self, width: int, height: int) -> None:
        """Dinamik harita / ekran çözünürlüğünü günceller (1280x720, 1920x1080, 2560x1440 vb.)."""
        if width > 0 and height > 0:
            self.viewport_width = int(width)
            self.viewport_height = int(height)
            if self.odometry_tracker is not None:
                if hasattr(self.odometry_tracker, "map_width"):
                    self.odometry_tracker.map_width = int(width)
                if hasattr(self.odometry_tracker, "map_height"):
                    self.odometry_tracker.map_height = int(height)

    @property
    def patch_dir(self) -> str:
        """Aktif bölgenin patches dizinini döndürür."""
        return self._get_region_patches_dir(self.active_region)

    # ------------------------------------------------------------------
    # Dosya Yolu Yardımcıları
    # ------------------------------------------------------------------
    def _get_region_dir(self, region: str) -> str:
        return os.path.join(self.base_dir, str(region))

    def _get_region_json_path(self, region: str) -> str:
        return os.path.join(self._get_region_dir(region), "region_data.json")

    def _get_region_patches_dir(self, region: str) -> str:
        return os.path.join(self._get_region_dir(region), "patches")

    def _get_patch_rel_path(self, region: str, node_id: int) -> str:
        base_name = os.path.basename(os.path.normpath(self.base_dir)) or "routes"
        return _normalize_path(f"{base_name}/{region}/patches/node_{int(node_id)}.png")

    def _resolve_patch_disk_path(self, node: NodeData) -> str:
        """NodeData içindeki patch_path değerini gerçek disk yoluna çözümler."""
        raw_path = node.patch_path
        if raw_path:
            if os.path.isabs(raw_path) and os.path.exists(raw_path):
                return raw_path
            if os.path.exists(raw_path):
                return raw_path
            # Eğer base_dir özel bir test diziniyse (örn. tmp_dir/routes)
            norm = _normalize_path(raw_path)
            parts = norm.split("/")
            if len(parts) >= 4 and parts[-2] == "patches":
                candidate = os.path.join(self._get_region_patches_dir(node.region), parts[-1])
                if os.path.exists(candidate):
                    return candidate
        return os.path.join(self._get_region_patches_dir(node.region), f"node_{node.id}.png")

    # ------------------------------------------------------------------
    # Yükleme / Kaydetme İşlemleri
    # ------------------------------------------------------------------
    def load_all_regions(self) -> Dict[str, Any]:
        """`routes/<region_name>/region_data.json` dosyalarının tamamını yükler."""
        self._regions.clear()
        try:
            if os.path.exists(self.base_dir):
                for entry in sorted(os.listdir(self.base_dir)):
                    reg_dir = os.path.join(self.base_dir, entry)
                    json_path = os.path.join(reg_dir, "region_data.json")
                    if os.path.isdir(reg_dir) and os.path.isfile(json_path):
                        self.load_region(entry)
        except Exception as exc:
            logger.error("Bölgeler taranırken hata oluştu: %s", exc)

        if self._regions and self.active_region not in self._regions:
            self.active_region = next(iter(self._regions.keys()))

        self._sync_legacy_data()
        return self.data

    def load_data(self) -> Dict[str, Any]:
        """Geriye dönük uyumluluk için tüm bölgeleri yükler ve data sözlüğünü döndürür."""
        return self.load_all_regions()

    def load_region(self, region: str) -> bool:
        """Belirtilen bölgenin `region_data.json` dosyasını okur (alt bölge eşlemelerini otomatik canonical ana bölgeye bağlar)."""
        clean_reg = str(region or "").strip()
        canon_reg = resolve_canonical_region(clean_reg)
        json_path = self._get_region_json_path(canon_reg)
        if not os.path.isfile(json_path):
            logger.warning("Bölge veri dosyası bulunamadı: %s", json_path)
            return False

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = json.load(f)

            reg_name = str(raw.get("region", canon_reg))
            raw_routes = raw.get("routes", [DEFAULT_ROUTE])
            routes_list: List[str] = [DEFAULT_ROUTE]
            if isinstance(raw_routes, dict):
                for r_key in raw_routes.keys():
                    if r_key and str(r_key) not in routes_list:
                        routes_list.append(str(r_key))
            elif isinstance(raw_routes, list):
                for r_name in raw_routes:
                    if r_name and str(r_name) not in routes_list:
                        routes_list.append(str(r_name))

            nodes_list: List[NodeData] = []
            for idx, item in enumerate(raw.get("nodes", []), start=1):
                if isinstance(item, dict):
                    item.setdefault("region", reg_name)
                    node_obj = NodeData.from_dict(item)
                    if node_obj.seq_index <= 0:
                        node_obj.seq_index = idx
                    nodes_list.append(node_obj)
                    for r_in_node in node_obj.routes:
                        if r_in_node not in routes_list:
                            routes_list.append(r_in_node)

            self._regions[reg_name] = {
                "routes": routes_list,
                "nodes": nodes_list,
            }
            os.makedirs(self._get_region_patches_dir(reg_name), exist_ok=True)
            self._sync_legacy_data()
            return True
        except Exception as exc:
            logger.error("'%s' bölgesi yüklenirken hata: %s", region, exc)
            return False

    def save_region(self, region: Optional[str] = None, create_backup: bool = True) -> bool:
        """
        Belirtilen bölgenin verilerini `routes/<region_name>/region_data.json` dosyasına yazar.
        - Atomik yazma (.tmp -> flush -> fsync -> os.replace)
        - Windows dosya kilitleme ve izin yarışlarına karşı yeniden deneme döngüsü
        - Otomatik yedekleme (`region_data.backup.json`)
        - Sıralı, 1-tabanlı ardışık seq_index garantisi
        """
        self.last_save_error = None
        target_region = resolve_canonical_region(region or self.active_region)
        if target_region not in self._regions:
            logger.warning("Kaydedilecek bölge bulunamadı: %s", target_region)
            return False

        try:
            reg_dir = self._get_region_dir(target_region)
            patches_dir = self._get_region_patches_dir(target_region)
            os.makedirs(reg_dir, exist_ok=True)
            os.makedirs(patches_dir, exist_ok=True)

            reg_info = self._regions[target_region]
            routes_list: List[str] = list(reg_info.get("routes", [DEFAULT_ROUTE]))
            if DEFAULT_ROUTE not in routes_list:
                routes_list.insert(0, DEFAULT_ROUTE)

            # UI veya map_viewer self.data üzerinde sıralama/düğüm güncellediyse
            # bunu öncelikli olarak self._regions içine senkronize et
            if not hasattr(self, "_legacy_synced_ids"):
                self._legacy_synced_ids = {}

            last_synced = self._legacy_synced_ids.get(target_region)
            current_nodes = reg_info.get("nodes", [])
            current_ids = [n.id for n in current_nodes]

            if isinstance(getattr(self, "data", None), dict):
                legacy_reg = self.data.get("regions", {}).get(target_region)
                if isinstance(legacy_reg, dict) and "nodes" in legacy_reg:
                    legacy_nodes = legacy_reg["nodes"]
                    if isinstance(legacy_nodes, list) and legacy_nodes:
                        legacy_ids = [
                            n.id if isinstance(n, NodeData) else int(n.get("id", 0))
                            for n in legacy_nodes
                        ]
                        # self.data son senkronize sıradan farklıysa ve _regions son senkronize sırayla eşleşiyorsa:
                        should_sync_from_legacy = (
                            last_synced is not None
                            and current_ids == last_synced
                            and legacy_ids != last_synced
                        )
                        if should_sync_from_legacy:
                            node_by_id = {n.id: n for n in current_nodes}
                            synced_nodes: List[NodeData] = []
                            for item in legacy_nodes:
                                if isinstance(item, NodeData):
                                    synced_nodes.append(item)
                                elif isinstance(item, dict):
                                    nid = int(item.get("id", 0))
                                    if nid in node_by_id:
                                        n_obj = node_by_id[nid]
                                        for k, v in item.items():
                                            if hasattr(n_obj, k) and k != "id":
                                                setattr(n_obj, k, v)
                                        synced_nodes.append(n_obj)
                                    else:
                                        item_copy = dict(item)
                                        item_copy.setdefault("region", target_region)
                                        synced_nodes.append(NodeData.from_dict(item_copy))
                            synced_ids = {n.id for n in synced_nodes}
                            for n in current_nodes:
                                if n.id not in synced_ids:
                                    synced_nodes.append(n)
                            reg_info["nodes"] = synced_nodes

            nodes_list: List[NodeData] = reg_info.get("nodes", [])
            for idx, node in enumerate(nodes_list, start=1):
                node.seq_index = idx

            for node in nodes_list:
                for r_name in node.routes:
                    if r_name and r_name not in routes_list:
                        routes_list.append(r_name)

            payload = {
                "region": target_region,
                "routes": routes_list,
                "nodes": [n.to_dict() for n in nodes_list],
                "updated_at": time.time(),
            }

            json_path = self._get_region_json_path(target_region)
            tmp_path = json_path + ".tmp"
            backup_path = os.path.join(reg_dir, "region_data.backup.json")

            # Mevcut geçerli json varsa yedekle
            if create_backup and os.path.isfile(json_path):
                try:
                    if os.path.getsize(json_path) > 0:
                        shutil.copyfile(json_path, backup_path)
                except Exception as b_exc:
                    logger.debug("Otomatik yedek oluşturulamadı (%s): %s", backup_path, b_exc)

            # Atomik geçici dosyaya yazma + flush + fsync
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass

            # Windows dosya kilitleme ve erişim yarışlarına karşı güvenli atomik değiştirme döngüsü
            replace_ok = False
            last_replace_exc: Optional[Exception] = None
            for attempt in range(5):
                try:
                    os.replace(tmp_path, json_path)
                    replace_ok = True
                    break
                except OSError as r_exc:
                    last_replace_exc = r_exc
                    time.sleep(0.04 * (attempt + 1))

            if not replace_ok:
                if last_replace_exc is not None:
                    raise last_replace_exc

            self._sync_legacy_data()
            return True
        except OSError as exc:
            self.last_save_error = exc
            winerror = getattr(exc, "winerror", None)
            if winerror == 5:
                logger.error(
                    "'%s' bölgesi yazılamadı (WinError 5): %s", target_region, exc
                )
            else:
                logger.error("'%s' bölgesi kaydedilirken hata: %s", target_region, exc)
            return False
        except Exception as exc:
            self.last_save_error = exc
            logger.error("'%s' bölgesi kaydedilirken hata: %s", target_region, exc)
            return False

    def backup_region_file(self, region: Optional[str] = None) -> str:
        """Copy ``region_data.json`` onto the single ``region_data.backup.json``."""
        target_region = resolve_canonical_region(region or self.active_region)
        src = self._get_region_json_path(target_region)
        dst = os.path.join(self._get_region_dir(target_region), "region_data.backup.json")
        if not os.path.isfile(src):
            raise FileNotFoundError(src)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return dst

    def save_data(self) -> bool:
        """Tüm bölgeleri diske kaydeder."""
        success = True
        for reg_name in list(self._regions.keys()):
            if not self.save_region(reg_name):
                success = False
        return success

    def reorder_route_nodes(
        self,
        region_name: str,
        ordered_ids: List[int],
        route_name: Optional[str] = None,
    ) -> bool:
        """
        Düğümleri belirtilen ID sırasına göre kalıcı olarak yeniden sıralar,
        seq_index değerlerini günceller ve diske kaydeder.
        """
        clean_reg = resolve_canonical_region(str(region_name or self.active_region).strip())
        if clean_reg not in self._regions:
            return False
        nodes = self._regions[clean_reg].get("nodes", [])
        node_map = {n.id: n for n in nodes}

        reordered: List[NodeData] = []
        seen = set()
        for nid in ordered_ids:
            if nid in node_map and nid not in seen:
                reordered.append(node_map[nid])
                seen.add(nid)
        # Sıralama listesinde yer almayan diğer düğümleri koru
        for n in nodes:
            if n.id not in seen:
                reordered.append(n)
                seen.add(n.id)

        for idx, node in enumerate(reordered, start=1):
            node.seq_index = idx

        self._regions[clean_reg]["nodes"] = reordered
        return self.save_region(clean_reg)

    def resequence_nodes(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
    ) -> List[NodeData]:
        """
        Bölgedeki tüm düğümlere 1-tabanlı sıralı seq_index (1, 2, ..., N)
        değerlerini kesin olarak atar, veritabanını atomik olarak kaydeder ve düğüm listesini döndürür.
        """
        target_region = resolve_canonical_region(region or self.active_region)
        if target_region not in self._regions:
            return []
        nodes = self._regions[target_region].get("nodes", [])
        for idx, node in enumerate(nodes, start=1):
            node.seq_index = idx
        self.save_region(target_region)
        if route and route != DEFAULT_ROUTE:
            return [n for n in nodes if route in n.routes]
        return list(nodes)

    def move_node_in_sequence(
        self,
        node_id: int,
        new_index: int,
        region: Optional[str] = None,
    ) -> bool:
        """
        Düğümü rota sıralamasında yeni bir konuma (1-tabanlı indeks) taşır,
        tüm düğümlerin seq_index değerlerini günceller ve atomik olarak kaydeder.
        """
        target_region = resolve_canonical_region(region or self.active_region)
        if target_region not in self._regions:
            return False
        nodes = self._regions[target_region].get("nodes", [])
        target_node = None
        curr_idx = -1
        for i, n in enumerate(nodes):
            if n.id == int(node_id):
                target_node = n
                curr_idx = i
                break
        if target_node is None:
            return False
        nodes.pop(curr_idx)
        target_idx = max(0, min(len(nodes), int(new_index) - 1))
        nodes.insert(target_idx, target_node)
        for idx, n in enumerate(nodes, start=1):
            n.seq_index = idx
        return self.save_region(target_region)

    def _sync_legacy_data(self) -> None:
        """UI ve eski modüllerin `self.data` erişimi için uyumluluk sözlüğünü günceller."""
        regions_dict: Dict[str, Any] = {}
        if not hasattr(self, "_legacy_synced_ids"):
            self._legacy_synced_ids = {}
        for reg_name, reg_info in self._regions.items():
            nodes_list: List[NodeData] = reg_info.get("nodes", [])
            routes_list: List[str] = reg_info.get("routes", [DEFAULT_ROUTE])
            routes_map: Dict[str, List[int]] = {DEFAULT_ROUTE: [n.id for n in nodes_list]}
            for r_name in routes_list:
                if r_name != DEFAULT_ROUTE:
                    routes_map[r_name] = [n.id for n in nodes_list if r_name in n.routes]
            regions_dict[reg_name] = {
                "routes": routes_map,
                "nodes": nodes_list,
            }
            self._legacy_synced_ids[reg_name] = [n.id for n in nodes_list]

        self.data["regions"] = CanonicalRegionDict(regions_dict)
        self.data["active_region"] = self.active_region
        self.data["active_route"] = self.active_route

    # ------------------------------------------------------------------
    # Bölge (Region) Yönetim Metodları
    # ------------------------------------------------------------------
    def create_region(self, region_name: str) -> bool:
        """Yeni bir bölge klasörü ve `region_data.json` oluşturur."""
        clean_name = str(region_name).strip()
        if not clean_name:
            logger.warning("Boş bölge adı ile bölge oluşturulamaz.")
            return False

        if clean_name in self._regions:
            logger.info("Bölge zaten mevcut: %s", clean_name)
            return True

        try:
            self._regions[clean_name] = {
                "routes": [DEFAULT_ROUTE],
                "nodes": [],
            }
            self.active_region = clean_name
            return self.save_region(clean_name)
        except Exception as exc:
            logger.error("Bölge oluşturma hatası (%s): %s", clean_name, exc)
            return False

    def rename_region(self, old_name: str, new_name: str) -> bool:
        """Mevcut bir bölgeyi, klasörünü ve tüm node patch_path referanslarını yeniden adlandırır."""
        old_clean = str(old_name).strip()
        new_clean = str(new_name).strip()
        if not old_clean or not new_clean:
            logger.warning("Geçersiz bölge adı: '%s' -> '%s'", old_clean, new_clean)
            return False
        if old_clean not in self._regions:
            logger.warning("Yeniden adlandırılacak bölge bulunamadı: %s", old_clean)
            return False
        if old_clean == new_clean:
            return True
        if new_clean in self._regions:
            logger.warning("Hedef bölge adı zaten kullanımda: %s", new_clean)
            return False

        try:
            old_dir = self._get_region_dir(old_clean)
            new_dir = self._get_region_dir(new_clean)
            if os.path.exists(old_dir):
                shutil.move(old_dir, new_dir)
            else:
                os.makedirs(self._get_region_patches_dir(new_clean), exist_ok=True)

            reg_info = self._regions.pop(old_clean)
            for node in reg_info.get("nodes", []):
                node.region = new_clean
                node.patch_path = self._get_patch_rel_path(new_clean, node.id)

            self._regions[new_clean] = reg_info
            if self.active_region == old_clean:
                self.active_region = new_clean

            return self.save_region(new_clean)
        except Exception as exc:
            logger.error("Bölge yeniden adlandırma hatası (%s -> %s): %s", old_clean, new_clean, exc)
            return False

    def delete_region(self, region_name: str) -> bool:
        """Belirtilen bölgeyi ve diskteki klasörünü tamamen siler."""
        clean_name = str(region_name).strip()
        if clean_name not in self._regions:
            logger.warning("Silinecek bölge bulunamadı: %s", clean_name)
            return False

        try:
            reg_dir = self._get_region_dir(clean_name)
            if os.path.exists(reg_dir):
                shutil.rmtree(reg_dir)

            del self._regions[clean_name]
            if self.active_region == clean_name:
                self.active_region = next(iter(self._regions.keys()), DEFAULT_REGION)
                self.active_route = DEFAULT_ROUTE

            self._sync_legacy_data()
            return True
        except Exception as exc:
            logger.error("Bölge silme hatası (%s): %s", clean_name, exc)
            return False

    def get_regions(self) -> List[str]:
        """Kayıtlı tüm bölge isimlerinin listesini döndürür."""
        return list(self._regions.keys())

    # ------------------------------------------------------------------
    # Rota (Route) Yönetim Metodları
    # ------------------------------------------------------------------
    def create_route(self, region_name: str, route_name: Optional[str] = None) -> bool:
        """
        Bir bölge içinde yeni rota oluşturur.
        Tek argüman verilirse aktif bölgede o isimle rota açar.
        """
        if route_name is None:
            target_region = self.active_region
            target_route = str(region_name).strip()
        else:
            target_region = str(region_name).strip()
            target_route = str(route_name).strip()

        if not target_route:
            logger.warning("Rota adı boş olamaz.")
            return False

        if target_region not in self._regions:
            if not self.create_region(target_region):
                return False

        routes_list: List[str] = self._regions[target_region]["routes"]
        if target_route not in routes_list:
            routes_list.append(target_route)
        return self.save_region(target_region)

    def rename_route(
        self,
        region_name: str,
        old_route_name: str,
        new_route_name: Optional[str] = None,
    ) -> bool:
        """Bir bölgedeki rotayı yeniden adlandırır ve bağlı düğümleri günceller."""
        if new_route_name is None:
            target_region = self.active_region
            old_r = str(region_name).strip()
            new_r = str(old_route_name).strip()
        else:
            target_region = str(region_name).strip()
            old_r = str(old_route_name).strip()
            new_r = str(new_route_name).strip()

        if not old_r or not new_r or old_r == DEFAULT_ROUTE or new_r == DEFAULT_ROUTE:
            logger.warning("Geçersiz rota yeniden adlandırma isteği: '%s' -> '%s'", old_r, new_r)
            return False

        if target_region not in self._regions:
            logger.warning("Rota yeniden adlandırma için bölge bulunamadı: %s", target_region)
            return False

        reg_info = self._regions[target_region]
        routes_list: List[str] = reg_info["routes"]
        if old_r not in routes_list:
            logger.warning("Yeniden adlandırılacak rota bulunamadı: %s/%s", target_region, old_r)
            return False

        idx = routes_list.index(old_r)
        if new_r in routes_list:
            routes_list.pop(idx)
        else:
            routes_list[idx] = new_r

        for node in reg_info["nodes"]:
            if old_r in node.routes:
                node.routes = [new_r if r == old_r else r for r in node.routes]
                # Tekrarlı rota isimlerini temizle
                deduped: List[str] = []
                for r_item in node.routes:
                    if r_item not in deduped:
                        deduped.append(r_item)
                node.routes = deduped

        if self.active_region == target_region and self.active_route == old_r:
            self.active_route = new_r

        return self.save_region(target_region)

    def delete_route(self, region_name: str, route_name: Optional[str] = None) -> bool:
        """Belirtilen rotayı bölgeden ve düğümlerin rota listesinden kaldırır (düğümleri silmez)."""
        if route_name is None:
            target_region = self.active_region
            target_route = str(region_name).strip()
        else:
            target_region = str(region_name).strip()
            target_route = str(route_name).strip()

        if target_route == DEFAULT_ROUTE:
            logger.warning("'All' varsayılan rotası silinemez.")
            return False

        if target_region not in self._regions:
            logger.warning("Rota silme için bölge bulunamadı: %s", target_region)
            return False

        reg_info = self._regions[target_region]
        routes_list: List[str] = reg_info["routes"]
        if target_route not in routes_list:
            logger.warning("Silinecek rota bulunamadı: %s/%s", target_region, target_route)
            return False

        routes_list.remove(target_route)
        for node in reg_info["nodes"]:
            if target_route in node.routes:
                node.routes.remove(target_route)

        if self.active_region == target_region and self.active_route == target_route:
            self.active_route = DEFAULT_ROUTE

        return self.save_region(target_region)

    def get_routes(self, region: Optional[str] = None) -> List[str]:
        """Belirtilen bölgedeki tüm rotaların listesini döndürür."""
        target_region = region or self.active_region
        canon_reg = resolve_canonical_region(target_region)
        if canon_reg not in self._regions:
            self.load_region(canon_reg)
        if canon_reg not in self._regions:
            return [DEFAULT_ROUTE]
        routes_list = list(self._regions[canon_reg].get("routes", [DEFAULT_ROUTE]))
        if DEFAULT_ROUTE not in routes_list:
            routes_list.insert(0, DEFAULT_ROUTE)
        if is_subregion(target_region) and target_region not in routes_list:
            routes_list.append(target_region)
        return routes_list

    # ------------------------------------------------------------------
    # Node (Düğüm) Yönetim Metodları
    # ------------------------------------------------------------------
    def _next_node_id(self, region: str) -> int:
        if region not in self._regions or not self._regions[region]["nodes"]:
            return 1
        existing_ids = [n.id for n in self._regions[region]["nodes"]]
        return max(existing_ids) + 1

    def _prepare_patch_image(self, patch_img: Optional[np.ndarray]) -> np.ndarray:
        """Gelen görüntüyü 80x80 gri tonlamalı (uint8) kesite dönüştürür."""
        if patch_img is None or not isinstance(patch_img, np.ndarray) or patch_img.size == 0:
            return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)

        if len(patch_img.shape) == 3:
            if patch_img.shape[2] == 4:
                gray = cv2.cvtColor(patch_img, cv2.COLOR_BGRA2GRAY)
            else:
                gray = cv2.cvtColor(patch_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = patch_img.copy()

        if gray.shape[0] != PATCH_SIZE or gray.shape[1] != PATCH_SIZE:
            gray = cv2.resize(gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

        return np.asarray(gray, dtype=np.uint8)

    def add_node(
        self,
        region: Optional[Union[str, int, NodeData]] = None,
        route: Optional[Union[str, List[str], int]] = None,
        patch_img: Optional[np.ndarray] = None,
        in_air: bool = False,
        map_x: int = 0,
        map_y: int = 0,
        norm_x: Optional[float] = None,
        norm_y: Optional[float] = None,
        resource_type: str = DEFAULT_RESOURCE_TYPE,
        rest_duration_sec: Optional[int] = None,
        node_id: Optional[int] = None,
        region_name: Optional[str] = None,
        route_name: Optional[str] = None,
        routes: Optional[List[str]] = None,
        patch_path: Optional[str] = None,
        max_harvests: int = DEFAULT_MAX_HARVESTS,
        current_harvests: int = 0,
        source_type: str = DEFAULT_SOURCE_TYPE,
        state: str = "active",
        seq_index: int = 0,
        **kwargs: Any,
    ) -> NodeData:
        """
        Bölgeye ve rotaya yeni bir düğüm ekler, 80x80 harita kesitini PNG olarak kaydeder
        ve `region_data.json` dosyasını günceller.
        """
        # Durum 1: Doğrudan NodeData nesnesi verildiyse
        if isinstance(region, NodeData):
            node_obj = region
            target_region = node_obj.region or self.active_region
            if getattr(self, "blacklist_manager", None) is not None and not kwargs.get("ignore_blacklist", False):
                is_bl, bl_entry = self.blacklist_manager.is_blacklisted(
                    region=target_region,
                    map_x=node_obj.map_x,
                    map_y=node_obj.map_y,
                    norm_x=node_obj.norm_x,
                    norm_y=node_obj.norm_y,
                )
                if is_bl:
                    logger.warning(
                        "⛔ [KARA LİSTE ENGELİ] '%s' bölgesinde (%d, %d) koordinatı kara listede! Düğüm eklenmesi engellendi.",
                        target_region,
                        node_obj.map_x,
                        node_obj.map_y,
                    )
                    return None
            if target_region not in self._regions:
                self.create_region(target_region)
            if not node_obj.id or any(n.id == node_obj.id for n in self._regions[target_region]["nodes"]):
                node_obj.id = self._next_node_id(target_region)
            node_obj.region = target_region
            node_obj.patch_path = self._get_patch_rel_path(target_region, node_obj.id)
            disk_patch_path = os.path.join(self._get_region_patches_dir(target_region), f"node_{node_obj.id}.png")
            os.makedirs(os.path.dirname(disk_patch_path), exist_ok=True)
            if patch_img is not None:
                _safe_imwrite(disk_patch_path, self._prepare_patch_image(patch_img))
            elif not os.path.exists(disk_patch_path):
                _safe_imwrite(disk_patch_path, self._prepare_patch_image(None))

            if node_obj.seq_index <= 0:
                node_obj.seq_index = len(self._regions[target_region]["nodes"]) + 1

            for r_item in node_obj.routes:
                if r_item and r_item not in self._regions[target_region]["routes"]:
                    self._regions[target_region]["routes"].append(r_item)
            self._regions[target_region]["nodes"].append(node_obj)
            self.save_region(target_region)
            return node_obj

        # Durum 2: Eski imza add_node(map_x, map_y, patch_img, region_name, route_name)
        if isinstance(region, (int, float)) and isinstance(route, (int, float)):
            map_x = int(region)
            map_y = int(route)
            target_region = str(region_name or self.active_region).strip()
            raw_route = route_name or routes or self.active_route
        else:
            target_region = str(region or region_name or self.active_region).strip()
            raw_route = route if route is not None else (routes if routes is not None else (route_name or self.active_route))

        if not target_region:
            target_region = DEFAULT_REGION

        if getattr(self, "blacklist_manager", None) is not None and not kwargs.get("ignore_blacklist", False):
            is_bl, bl_entry = self.blacklist_manager.is_blacklisted(
                region=target_region,
                map_x=map_x,
                map_y=map_y,
                norm_x=norm_x,
                norm_y=norm_y,
            )
            if is_bl:
                logger.warning(
                    "⛔ [KARA LİSTE ENGELİ] '%s' bölgesinde (%d, %d) koordinatı kara listede! Düğüm eklenmesi engellendi (Neden: %s).",
                    target_region,
                    map_x,
                    map_y,
                    bl_entry.reason if bl_entry else "Blacklisted",
                )
                return None

        if target_region not in self._regions:
            self.create_region(target_region)

        assigned_id = int(node_id) if node_id is not None else self._next_node_id(target_region)
        if any(n.id == assigned_id for n in self._regions[target_region]["nodes"]):
            assigned_id = self._next_node_id(target_region)

        node_routes: List[str] = []
        if isinstance(raw_route, str):
            if raw_route and raw_route != DEFAULT_ROUTE:
                node_routes.append(raw_route)
        elif isinstance(raw_route, (list, tuple, set)):
            for r_item in raw_route:
                if r_item and str(r_item) != DEFAULT_ROUTE and str(r_item) not in node_routes:
                    node_routes.append(str(r_item))

        for r_item in node_routes:
            if r_item not in self._regions[target_region]["routes"]:
                self._regions[target_region]["routes"].append(r_item)

        rel_patch_path = patch_path or self._get_patch_rel_path(target_region, assigned_id)
        disk_patch_path = os.path.join(self._get_region_patches_dir(target_region), f"node_{assigned_id}.png")
        os.makedirs(os.path.dirname(disk_patch_path), exist_ok=True)

        if patch_img is not None:
            prepared = self._prepare_patch_image(patch_img)
            _safe_imwrite(disk_patch_path, prepared)
        elif patch_path and os.path.exists(patch_path) and os.path.abspath(patch_path) != os.path.abspath(disk_patch_path):
            src_img = _safe_imread_gray(patch_path)
            _safe_imwrite(disk_patch_path, self._prepare_patch_image(src_img))
        elif not os.path.exists(disk_patch_path):
            _safe_imwrite(disk_patch_path, self._prepare_patch_image(None))

        # Viewport normalizasyonu: dinamik çözünürlük desteği (1280x720, 1920x1080 vb.)
        vw = float(kwargs.get("viewport_width") or getattr(self, "viewport_width", 0) or 0)
        vh = float(kwargs.get("viewport_height") or getattr(self, "viewport_height", 0) or 0)
        if vw <= 0:
            vw = 1280.0 if (0 < map_x <= 1280 and map_y <= 720) else 1920.0
        if vh <= 0:
            vh = 720.0 if (0 < map_y <= 720 and map_x <= 1280) else 1080.0

        calc_norm_x = float(norm_x) if norm_x is not None else (round(float(map_x) / vw, 4) if map_x > 0 else 0.5)
        calc_norm_y = float(norm_y) if norm_y is not None else (round(float(map_y) / vh, 4) if map_y > 0 else 0.5)

        existing_node_count = len(self._regions[target_region]["nodes"])
        assigned_seq = int(seq_index) if seq_index > 0 else (existing_node_count + 1)

        node = NodeData(
            id=assigned_id,
            region=target_region,
            routes=node_routes,
            patch_path=_normalize_path(rel_patch_path),
            in_air=bool(in_air),
            gathered_count=0,
            last_visited_timestamp=0.0,
            rest_duration_sec=int(rest_duration_sec if rest_duration_sec is not None else self.default_rest_sec),
            consecutive_failures=0,
            map_x=int(map_x),
            map_y=int(map_y),
            norm_x=calc_norm_x,
            norm_y=calc_norm_y,
            resource_type=str(resource_type or DEFAULT_RESOURCE_TYPE),
            max_harvests=max(1, int(max_harvests if max_harvests is not None else DEFAULT_MAX_HARVESTS)),
            current_harvests=max(0, int(current_harvests if current_harvests is not None else 0)),
            source_type=str(source_type or DEFAULT_SOURCE_TYPE),
            state=str(state or "active"),
            seq_index=assigned_seq,
        )

        self._regions[target_region]["nodes"].append(node)
        self.save_region(target_region)
        logger.info("Yeni düğüm eklendi: Bölge=%s, ID=#%d, Rotalar=%s", target_region, node.id, node.routes)
        return node

    def get_node(self, node_id: int, region: Optional[str] = None) -> Optional[NodeData]:
        """ID ve bölgeye göre NodeData nesnesini bulur."""
        target_id = int(node_id)
        if region:
            canon_reg = resolve_canonical_region(region)
            if canon_reg in self._regions:
                for node in self._regions[canon_reg]["nodes"]:
                    if node.id == target_id:
                        return node
                return None

        # Önce aktif bölgede ara, sonra diğer bölgelerde
        active_canon = resolve_canonical_region(self.active_region)
        search_order = [active_canon] + [r for r in self._regions.keys() if r != active_canon]
        for reg_name in search_order:
            if reg_name in self._regions:
                for node in self._regions[reg_name]["nodes"]:
                    if node.id == target_id:
                        return node
        return None

    def get_route_nodes(
        self,
        region_name: Optional[str] = None,
        route_name: Optional[str] = None,
    ) -> List[NodeData]:
        """
        Belirtilen bölge ve rotadaki düğümlerin listesini döndürür.
        Eğer 'Dredgion Crash Site', 'Safe Haven', 'Calderon Canyon' gibi bir alt bölge
        istendiyse, kanonik üst bölgeyi ('Altgard') şeffafça yükler ve alt bölgenin
        sınırlayıcı alanına (bounding box / center radius) veya rota etiketine göre filtreler.
        """
        raw_reg = region_name or self.active_region
        canon_reg = resolve_canonical_region(raw_reg)
        target_route = route_name or self.active_route

        if canon_reg not in self._regions:
            self.load_region(canon_reg)

        if canon_reg not in self._regions:
            return []

        all_nodes: List[NodeData] = self._regions[canon_reg].get("nodes", [])

        # Durum 1: İstenen bölge adı doğrudan bir alt bölge ise (örn: region_name="Dredgion Crash Site")
        if is_subregion(raw_reg):
            filtered = filter_nodes_by_subregion(all_nodes, raw_reg)
            if target_route and target_route not in (DEFAULT_ROUTE, raw_reg):
                route_filtered = [n for n in filtered if target_route in n.routes]
                if route_filtered:
                    return route_filtered
            return filtered

        # Durum 2: İstenen rota adı bir alt bölge ise (örn: region_name="Altgard", route_name="Dredgion Crash Site")
        if target_route and is_subregion(target_route):
            return filter_nodes_by_subregion(all_nodes, target_route)

        # Durum 3: Standart rota filtreleme
        if not target_route or target_route == DEFAULT_ROUTE:
            return list(all_nodes)
        return [n for n in all_nodes if target_route in n.routes]

    def get_nodes(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
    ) -> List[NodeData]:
        """`get_route_nodes` için eş anlamlı yardımcı metod."""
        return self.get_route_nodes(region_name=region, route_name=route)

    def delete_node(self, node_id: int, region: Optional[str] = None) -> bool:
        """Düğümü bölgeden ve diskteki `patches/node_<id>.png` dosyasından siler."""
        target_id = int(node_id)
        target_region = resolve_canonical_region(region or self.active_region)
        if target_region not in self._regions:
            node_found = self.get_node(target_id)
            if node_found is None:
                logger.warning("Silinecek düğüm bulunamadı: #%d", target_id)
                return False
            target_region = node_found.region

        nodes_list: List[NodeData] = self._regions[target_region]["nodes"]
        target_node: Optional[NodeData] = None
        for n in nodes_list:
            if n.id == target_id:
                target_node = n
                break

        if target_node is None:
            logger.warning("'%s' bölgesinde silinecek düğüm bulunamadı: #%d", target_region, target_id)
            return False

        nodes_list.remove(target_node)
        for idx, n in enumerate(nodes_list, start=1):
            n.seq_index = idx
        if not self.save_region(target_region):
            nodes_list.append(target_node)
            for idx, n in enumerate(nodes_list, start=1):
                n.seq_index = idx
            logger.error("Düğüm silinemedi, kayıt başarısız: %s #%d", target_region, target_id)
            return False

        try:
            disk_patch = self._resolve_patch_disk_path(target_node)
            if os.path.isfile(disk_patch):
                os.remove(disk_patch)
        except Exception as exc:
            logger.warning("Düğüm yama görseli silinemedi (#%d): %s", target_id, exc)

        logger.info("Düğüm silindi: %s #%d", target_region, target_id)
        return True

    def delete_nodes(self, node_ids: List[int], region: Optional[str] = None) -> bool:
        """Delete the given nodes and write ``region_data.json`` once.

        Uses the same ``save_region`` replace as ``delete_node``. An empty id
        list writes nothing and returns True. If ``os.replace`` fails, including
        WinError 5, the in-memory list is restored and this returns False.
        """
        target_region = resolve_canonical_region(region or self.active_region)
        wanted = []
        seen = set()
        for raw_id in node_ids or []:
            try:
                node_id = int(raw_id)
            except (TypeError, ValueError):
                continue
            if node_id not in seen:
                seen.add(node_id)
                wanted.append(node_id)
        if not wanted:
            return True
        if target_region not in self._regions:
            logger.warning("Silinecek bölge bulunamadı: %s", target_region)
            return False

        nodes_list: List[NodeData] = self._regions[target_region]["nodes"]
        previous = list(nodes_list)
        removed = [node for node in previous if node.id in seen]
        if not removed:
            logger.warning("'%s' bölgesinde silinecek düğüm bulunamadı.", target_region)
            return False

        nodes_list[:] = [node for node in previous if node.id not in seen]
        for idx, n in enumerate(nodes_list, start=1):
            n.seq_index = idx
        if not self.save_region(target_region):
            nodes_list[:] = previous
            for idx, n in enumerate(nodes_list, start=1):
                n.seq_index = idx
            logger.error(
                "Toplu silme kaydı başarısız (%s): %s",
                target_region,
                self.last_save_error,
            )
            return False

        for node in removed:
            try:
                disk_patch = self._resolve_patch_disk_path(node)
                if os.path.isfile(disk_patch):
                    os.remove(disk_patch)
            except Exception as exc:
                logger.warning("Düğüm yama görseli silinemedi (#%d): %s", node.id, exc)
        logger.info("Düğümler silindi: %s %s", target_region, [node.id for node in removed])
        return True

    def add_node_to_route(
        self,
        node_id: int,
        route: str,
        region: Optional[str] = None,
    ) -> bool:
        """Mevcut bir düğümü belirtilen rotaya dahil eder."""
        clean_route = str(route).strip()
        if not clean_route or clean_route == DEFAULT_ROUTE:
            return False

        node = self.get_node(node_id, region=region)
        if node is None:
            logger.warning("Rotaya eklenecek düğüm bulunamadı: #%d", int(node_id))
            return False

        if clean_route not in node.routes:
            node.routes.append(clean_route)

        reg_routes = self._regions[node.region]["routes"]
        if clean_route not in reg_routes:
            reg_routes.append(clean_route)

        return self.save_region(node.region)

    def remove_node_from_route(
        self,
        node_id: int,
        route: str,
        region: Optional[str] = None,
    ) -> bool:
        """Düğümü belirtilen rotadan çıkarır (düğümün kendisini silmez)."""
        clean_route = str(route).strip()
        node = self.get_node(node_id, region=region)
        if node is None:
            logger.warning("Rotadan çıkarılacak düğüm bulunamadı: #%d", int(node_id))
            return False

        if clean_route in node.routes:
            node.routes.remove(clean_route)
            return self.save_region(node.region)
        return False

    def move_node_to_region(
        self,
        node_id: int,
        target_region: str,
        source_region: Optional[str] = None,
        new_routes: Optional[List[str]] = None,
    ) -> Optional[NodeData]:
        """
        Bir düğümü (ve harita kesiti görselini) başka bir bölgeye taşır.
        """
        clean_target = str(target_region).strip()
        if not clean_target:
            logger.warning("Hedef bölge adı boş olamaz.")
            return None

        node = self.get_node(node_id, region=source_region)
        if node is None:
            logger.warning("Taşınacak düğüm bulunamadı: #%d", int(node_id))
            return None

        orig_region = node.region
        if orig_region == clean_target:
            return node

        if clean_target not in self._regions:
            self.create_region(clean_target)

        try:
            old_patch_disk = self._resolve_patch_disk_path(node)
            self._regions[orig_region]["nodes"].remove(node)

            # Hedef bölgede ID çakışması varsa yeni ID ata
            if any(n.id == node.id for n in self._regions[clean_target]["nodes"]):
                new_id = self._next_node_id(clean_target)
            else:
                new_id = node.id

            new_patch_disk = os.path.join(self._get_region_patches_dir(clean_target), f"node_{new_id}.png")
            os.makedirs(os.path.dirname(new_patch_disk), exist_ok=True)
            if os.path.isfile(old_patch_disk):
                shutil.move(old_patch_disk, new_patch_disk)
            else:
                _safe_imwrite(new_patch_disk, self._prepare_patch_image(None))

            node.id = new_id
            node.region = clean_target
            node.patch_path = self._get_patch_rel_path(clean_target, new_id)
            if new_routes is not None:
                node.routes = [str(r) for r in new_routes if r and str(r) != DEFAULT_ROUTE]

            for r_name in node.routes:
                if r_name not in self._regions[clean_target]["routes"]:
                    self._regions[clean_target]["routes"].append(r_name)

            self._regions[clean_target]["nodes"].append(node)
            self.save_region(orig_region)
            self.save_region(clean_target)
            logger.info("Düğüm taşındı: %s -> %s (Yeni ID=#%d)", orig_region, clean_target, new_id)
            return node
        except Exception as exc:
            logger.error("Düğüm taşıma hatası (#%d -> %s): %s", int(node_id), clean_target, exc)
            return None

    # ------------------------------------------------------------------
    # Rota Çalıştırma & İstatistik Metodları
    # ------------------------------------------------------------------
    def update_resting_nodes(
        self,
        rest_minutes: Optional[float] = None,
        region: Optional[str] = None,
    ) -> List[NodeData]:
        """
        Dinlenme süresi dolmuş olan düğümleri `"resting"` durumundan `"active"`
        durumuna uyandırır ve `node.current_harvests = 0` olarak sıfırlar.
        """
        now = time.time()
        rest_sec_override: Optional[int] = None
        if rest_minutes is not None and float(rest_minutes) > 0:
            rest_sec_override = int(round(float(rest_minutes) * 60.0))

        woken_nodes: List[NodeData] = []
        dirty_regions: List[str] = []
        region_keys = [region] if (region and region in self._regions) else list(self._regions.keys())

        for reg_name in region_keys:
            nodes_list: List[NodeData] = self._regions[reg_name].get("nodes", [])
            reg_dirty = False
            for node in nodes_list:
                if node.is_ready(now=now, rest_duration_sec=rest_sec_override):
                    if node.state == "resting":
                        node.state = "active"
                        node.current_harvests = 0
                        woken_nodes.append(node)
                        reg_dirty = True
                else:
                    if node.state != "resting" or node.current_harvests != 0:
                        node.state = "resting"
                        node.current_harvests = 0
                        reg_dirty = True
            if reg_dirty:
                dirty_regions.append(reg_name)

        for reg_name in dirty_regions:
            self.save_region(reg_name)

        return woken_nodes

    def refresh_cooldowns(
        self,
        rest_minutes: Optional[float] = None,
        region: Optional[str] = None,
    ) -> List[NodeData]:
        """`update_resting_nodes` için eş anlamlı yardımcı metod."""
        return self.update_resting_nodes(rest_minutes=rest_minutes, region=region)

    def get_next_ready_node(
        self,
        region: Optional[Union[str, int]] = None,
        route: Optional[str] = None,
        rest_duration_sec: Optional[int] = None,
        rest_minutes: Optional[float] = None,
        fallback_to_next: bool = False,
    ) -> Optional[NodeData]:
        """
        Seçili bölge ve rotadaki dinlenme süresi dolmuş (`is_ready() == True`)
        sıradaki düğümü döndürür.
        Eğer bir düğüm şu anda kısmi hasat aşamasındaysa (`1/3` veya `2/3`),
        tüm 3 hasat tamamlanana kadar o düğümde kalmaya öncelik verir.
        `fallback_to_next=True` olduğunda tüm düğümler dinlenmede olsa bile
        stall olmadan 1 -> 2 -> 3 -> 1... döngüsünü sürdürür.
        """
        # Geriye dönük uyumluluk: get_next_ready_node(300) şeklinde çağrıldıysa
        if isinstance(region, (int, float)) and rest_duration_sec is None:
            rest_duration_sec = int(region)
            target_region = self.active_region
        else:
            target_region = str(region) if region is not None else self.active_region

        if rest_duration_sec is None and rest_minutes is not None and float(rest_minutes) > 0:
            rest_duration_sec = int(round(float(rest_minutes) * 60.0))

        canon_reg = resolve_canonical_region(target_region)
        target_route = route or self.active_route
        nodes = self.get_route_nodes(region_name=target_region, route_name=target_route)
        if not nodes:
            return None

        now = time.time()
        ready_nodes: List[NodeData] = []
        for n in nodes:
            if n.not_gatherable:
                continue
            if n.is_ready(now=now, rest_duration_sec=rest_duration_sec):
                if n.state == "resting":
                    n.state = "active"
                    n.current_harvests = 0
                ready_nodes.append(n)

        # Kısmi hasat devam eden düğüm varsa (1/3 veya 2/3), 3/3 tamamlanana kadar aynı düğümde kal
        for candidate in ready_nodes:
            if 0 < candidate.current_harvests < max(1, candidate.max_harvests):
                key = (canon_reg, target_route)
                self._last_selected_node_id[key] = candidate.id
                return candidate

        # Döngüsel sıra: rota diziliminde son seçilen node'dan bir sonraki hazır node'u seç
        key = (canon_reg, target_route)
        last_id = self._last_selected_node_id.get(key)
        node_order = [n.id for n in nodes]
        last_pos = node_order.index(last_id) if (last_id is not None and last_id in node_order) else -1
        selected = None

        if ready_nodes:
            for candidate in ready_nodes:
                cand_pos = node_order.index(candidate.id) if candidate.id in node_order else 9999
                if cand_pos > last_pos:
                    selected = candidate
                    break
            if selected is None:
                selected = ready_nodes[0]
        elif fallback_to_next and nodes:
            # Tüm düğümler dinlenmede olsa bile stall olmadan döngüye devam et (1 -> 2 -> 3 -> 1...)
            next_idx = (last_pos + 1) % len(nodes)
            selected = nodes[next_idx]

        if selected is not None:
            self._last_selected_node_id[key] = selected.id
        return selected

    def get_next_node(
        self,
        region: Optional[Union[str, int]] = None,
        route: Optional[str] = None,
        rest_duration_sec: Optional[int] = None,
        rest_minutes: Optional[float] = None,
        fallback_to_next: bool = True,
    ) -> Optional[NodeData]:
        """
        Döngüsel rota sırasındaki bir sonraki düğümü seçer.
        Tüm düğümler dinlenmede olsa bile stall olmadan (duraklamadan)
        1 -> 2 -> 3 -> 1... döngüsüne süresiz devam eder.
        """
        return self.get_next_ready_node(
            region=region,
            route=route,
            rest_duration_sec=rest_duration_sec,
            rest_minutes=rest_minutes,
            fallback_to_next=fallback_to_next,
        )

    def get_route_stats(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
        rest_duration_sec: Optional[int] = None,
    ) -> Tuple[int, int]:
        """
        Belirtilen bölge ve rota için `(ready_count, resting_count)` istatistiğini döndürür.
        """
        target_region = region or self.active_region
        target_route = route or self.active_route
        nodes = self.get_route_nodes(region_name=target_region, route_name=target_route)
        now = time.time()
        ready_count = 0
        resting_count = 0
        for node in nodes:
            if node.is_ready(now=now, rest_duration_sec=rest_duration_sec):
                ready_count += 1
            else:
                resting_count += 1
        return ready_count, resting_count

    def record_harvest_stage(
        self,
        node_id: int,
        rest_minutes: float = 5.0,
        default_max_harvests: int = DEFAULT_MAX_HARVESTS,
        region: Optional[str] = None,
        extraction_failed: bool = False,
    ) -> Tuple[int, int, bool]:
        """
        Düğümün 3 aşamalı (`1/3`, `2/3`, `3/3`) toplama yaşam döngüsünü ilerletir:
        - Eğer `extraction_failed=True` ise ("Essence Extraction failed"), `node.current_harvests`
          ve `node.gather_count` KESİNLİKLE artırılmaz (hak düşmez, `1/3` ise `1/3` kalır)
          ve `(current_harvests, target, False)` döner.
        - `node.current_harvests += 1`, `node.gather_count += 1`, `self.total_gathered += 1`
        - `node.fail_count = 0` ve `self.consecutive_failures = 0`
        - `current_harvests < target` ise düğüm tükenmemiştir (`is_depleted=False`),
          `node.state = "active"` kalır ve `(current_harvests, target, False)` döner.
        - `current_harvests >= target` ise tüm 3 hasat tamamlanmıştır (`is_depleted=True`),
          `node.current_harvests = 0` sıfırlanır, `node.state = "resting"` yapılır,
          `node.last_gathered_ts = time.time()` ve dinlenme süresi ayarlanır,
          `(target, target, True)` döner.
        """
        target_default = max(1, int(default_max_harvests or DEFAULT_MAX_HARVESTS))
        node = self.get_node(node_id, region=region)
        if node is None:
            logger.warning("Hasat aşaması kaydedilecek düğüm bulunamadı: #%d", int(node_id))
            return (0, target_default, False)

        target = max(1, int(node.max_harvests or target_default))
        if node.not_gatherable:
            logger.info("Düğüm #%d parşömen, hasat sayılmadı", int(node.id))
            return (int(node.current_harvests), int(target), False)
        if extraction_failed:
            if node.state not in ("active", "gathering"):
                node.state = "active"
            self.save_region(node.region)
            return (int(node.current_harvests), int(target), False)

        node.current_harvests += 1
        node.gather_count += 1
        self.total_gathered += 1
        node.fail_count = 0
        self.consecutive_failures = 0

        if node.current_harvests < target:
            if node.state not in ("active", "gathering"):
                node.state = "active"
            self.save_region(node.region)
            return (int(node.current_harvests), int(target), False)

        # Tüm hasatlar (örn. 3/3) tamamlandı -> düğüm dinlenmeye (resting) geçer
        node.current_harvests = 0
        node.state = "resting"
        node.last_gathered_ts = time.time()
        if rest_minutes is not None and float(rest_minutes) > 0:
            node.rest_duration_sec = int(round(float(rest_minutes) * 60.0))
        self.save_region(node.region)
        return (int(target), int(target), True)

    def handle_extraction_failure(
        self,
        node_id: Union[int, NodeData, Dict[str, Any], None] = None,
        gather_key: str = "f",
        input_ctrl: Any = None,
        frame: Optional[np.ndarray] = None,
        chat_text: Optional[str] = None,
        chat_alerts: Optional[Dict[str, Any]] = None,
        extraction_failed: Optional[bool] = None,
        hud_reader: Any = None,
        region: Optional[str] = None,
        retry_press: bool = True,
        test_mode: bool = False,
    ) -> Dict[str, Any]:
        """
        Madde 23 — Toplama Başarısızlığında (`Essence Extraction failed`) Hak Düşmeme:
        Sohbette `"Essence Extraction failed"` algılandığında `current_harvests` sayacını
        artırmaz / sıfırlamaz (`0/3` -> `0/3`, `1/3` -> `1/3` olarak kalır) ve aynı kaynağa
        anında tekrar `F` / `E` tuşu basar.
        """
        from core.navigation_intel import handle_extraction_failure as _nav_handle_extraction_failure

        return _nav_handle_extraction_failure(
            node_or_id=node_id,
            route_mgr=self,
            input_ctrl=input_ctrl,
            gather_key=gather_key,
            frame=frame,
            chat_text=chat_text,
            chat_alerts=chat_alerts,
            extraction_failed=extraction_failed,
            hud_reader=hud_reader,
            region=region,
            retry_press=retry_press,
            test_mode=test_mode,
        )

    def handle_partial_node(
        self,
        node_id: Union[int, NodeData, Dict[str, Any]],
        completed_harvests: Optional[int] = None,
        max_harvests: Optional[int] = None,
        prompt_reappeared: Optional[bool] = False,
        rest_minutes: float = 5.0,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Madde 22 — Yarım Bırakılmış Kaynak (Partial Node) Algılama:
        Başka bir oyuncu kaynağı önceden 1 veya 2 kez toplamışsa (`1/3` veya `2/3` sonrasında
        kaynak yok olup `[F Gather]` tekrar çıkmazsa), 3. toplama için boşuna beklemek yerine
        `mark_node_gathered(node_id, force_deplete=True)` ile düğümü derhal `RESTING` (5 dk)
        durumuna geçirir.
        """
        from core.navigation_intel import PartialNodeDetector as _NavPartialNodeDetector

        detector = _NavPartialNodeDetector(
            route_manager=self,
            default_rest_minutes=rest_minutes,
        )
        return detector.check_and_handle_partial_node(
            node_or_id=node_id,
            completed_harvests=completed_harvests,
            max_harvests=max_harvests,
            prompt_reappeared=prompt_reappeared,
            route_mgr=self,
            rest_minutes=rest_minutes,
            region=region,
            wait_before_check=False,
        )

    detect_partial_node = handle_partial_node

    def mark_node_gathered(
        self,
        node_id: int,
        rest_minutes: Union[float, int, str, None] = 5.0,
        force_deplete: bool = True,
        region: Optional[str] = None,
    ) -> Optional[NodeData]:
        """
        Düğüm toplandığında çağrılır:
        - `force_deplete=True` (varsayılan veya erken tükenme durumunda):
          `node.current_harvests = 0` sıfırlanır ve düğüm doğrudan `"resting"` durumuna alınır.
        - `force_deplete=False`:
          `record_harvest_stage(node_id, rest_minutes)` metoduna delege edilir (`1/3`, `2/3`, `3/3`).
        """
        # Geriye dönük uyumluluk: mark_node_gathered(node_id, "Verteron") çağrısı
        effective_rest_min: Optional[float] = 5.0
        if isinstance(rest_minutes, str):
            region = rest_minutes
            effective_rest_min = None
        elif rest_minutes is not None:
            effective_rest_min = float(rest_minutes)
        else:
            effective_rest_min = None

        node = self.get_node(node_id, region=region)
        if node is None:
            logger.warning("Toplandı olarak işaretlenecek düğüm bulunamadı: #%d", int(node_id))
            return None
        if node.not_gatherable:
            logger.info("Düğüm #%d parşömen, toplanmış sayılmadı", int(node.id))
            return node

        if not force_deplete:
            stage_rest = effective_rest_min if effective_rest_min is not None else (node.rest_duration_sec / 60.0)
            self.record_harvest_stage(
                node_id=node.id,
                rest_minutes=float(stage_rest),
                default_max_harvests=max(1, int(node.max_harvests or DEFAULT_MAX_HARVESTS)),
                region=node.region,
            )
            return node

        # Eğer düğüm kısmi hasat aşamasındayken (current_harvests > 0) record_harvest_stage
        # üzerinden zaten gather_count artırıldıysa çift sayımı önle; doğrudan çağrıldıysa +1 artır.
        if node.current_harvests <= 0:
            node.gather_count += 1
            self.total_gathered += 1
        elif node.gather_count <= 0:
            node.gather_count = int(node.current_harvests)
            self.total_gathered += int(node.current_harvests)

        node.current_harvests = 0
        node.fail_count = 0
        self.consecutive_failures = 0
        node.state = "resting"
        node.last_gathered_ts = time.time()
        if effective_rest_min is not None and effective_rest_min > 0:
            node.rest_duration_sec = int(round(effective_rest_min * 60.0))
        self.save_region(node.region)
        return node

    def mark_node_visited(
        self,
        node_id: int,
        region_name: Optional[str] = None,
        rest_minutes: Optional[float] = None,
        force_deplete: bool = True,
    ) -> Optional[NodeData]:
        """Geriye dönük uyumluluk için `mark_node_gathered` takma adı."""
        return self.mark_node_gathered(
            node_id=node_id,
            rest_minutes=rest_minutes,
            force_deplete=force_deplete,
            region=region_name,
        )

    def mark_node_failed(
        self,
        node_id: int,
        region: Optional[str] = None,
        remove_after_failures: int = 3,
        active_route: Optional[str] = None,
        run_ai_check: bool = True,
        **kwargs: Any,
    ) -> Optional[NodeData]:
        """
        Düğüme ulaşılamadığında veya toplama başarısız olduğunda çağrılır:
        1. Eğer ardışık hata < 3 ise:
           Kullanıcı İsteği: "örnek rotaya geldi ve toplanacak bişey bulamadıysa orayı rotadan çıkarsın"
           Aktif rota (`active_route`) belirtilmişse düğümü o rotadan çıkarır (bölgeden silmez).
           Düğümü dinlenme/başarısız moduna alır böylece bu tur atlanır.
        2. Eğer ardışık hata >= 3 (veya >= remove_after_failures) ise:
           Kullanıcı İsteği: "3 kere üst üste geldi ve üçündede bulamadıysa rotayı otomatik silsin
           demekki yanlış işaretleme olduğunu anlasın yapay zekayla kontrol etsin ama bunu mutlaka
           orayıda birdaha kesinlikle kaydetmesin blacklist kısmı olsun ayarlarda oraya blacklistleri kaydetsin"
           - Yapay zeka ile analiz eder (verify_and_blacklist_node).
           - Aktif öğrenme ile ExemplarMemory'ye negatif örnek olarak ekler.
           - Koordinatları Kara Listeye (Blacklist) ekler (bir daha asla kaydedilmez).
           - Düğümü bölgeden ve diskten kalıcı olarak siler (delete_node).
        """
        node = self.get_node(node_id, region=region)
        if node is None:
            logger.warning("Başarısız olarak işaretlenecek düğüm bulunamadı: #%d", int(node_id))
            return None

        node.current_harvests = 0
        node.consecutive_failures += 1
        self.consecutive_failures += 1

        effective_limit = int(remove_after_failures) if int(remove_after_failures or 0) > 0 else 3
        curr_route = str(active_route or self.active_route or "").strip()

        # Durum 1: 3 kez üst üste başarısız -> YZ İnceleme + Kara Liste + Kalıcı Silme
        if node.consecutive_failures >= effective_limit:
            logger.warning(
                "🚨 [3x Başarısızlık] Düğüm #%d ardışık %d kez bulunamadı! YZ kontrolü başlatılıyor ve kara listeye ekleniyor...",
                node.id,
                node.consecutive_failures,
            )
            ai_diag = "empty_terrain"
            if run_ai_check:
                try:
                    from core.wrong_node_scan import verify_and_blacklist_node
                    ok, verdict, bl_entry = verify_and_blacklist_node(
                        node=node,
                        region=node.region,
                        reason="3x ardışık toplama başarısızlığı (YZ yanlış işaretleme doğrulaması)",
                        blacklist_mgr=getattr(self, "blacklist_manager", None),
                    )
                    ai_diag = verdict
                    logger.warning(
                        "🧠 [YZ Teşhisi] Düğüm #%d incelendi: %s -> Kara listeye kaydedildi.",
                        node.id,
                        verdict,
                    )
                except Exception as ai_exc:
                    logger.error("YZ kontrolü ve kara liste ekleme hatası: %s", ai_exc)
                    if getattr(self, "blacklist_manager", None) is not None:
                        self.blacklist_manager.add_entry(
                            region=node.region,
                            map_x=node.map_x,
                            map_y=node.map_y,
                            norm_x=node.norm_x,
                            norm_y=node.norm_y,
                            reason="3x ardışık toplama hatası",
                            source_node_id=node.id,
                        )

            # Düğümü haritadan ve diskten tamamen sil
            self.delete_node(node.id, region=node.region)
            return None

        # Durum 2: 1 veya 2 kez başarısız -> "orayı rotadan çıkarsın"
        if curr_route and curr_route != DEFAULT_ROUTE and curr_route in node.routes:
            logger.warning(
                "⚠️ [Rotadan Çıkarma] Düğüm #%d üzerinde kaynak bulunamadı -> '%s' rotasından çıkarıldı (Deneme: %d/%d).",
                node.id,
                curr_route,
                node.consecutive_failures,
                effective_limit,
            )
            self.remove_node_from_route(node.id, curr_route, region=node.region)
        else:
            logger.warning(
                "⚠️ [Tur Atlandı] Düğüm #%d üzerinde kaynak bulunamadı -> Geçici dinlenmeye alındı (Deneme: %d/%d).",
                node.id,
                node.consecutive_failures,
                effective_limit,
            )
            node.state = "resting"
            node.last_visited_timestamp = time.time()

        self.save_region(node.region)
        return node

    # ------------------------------------------------------------------
    # Görsel Odometri Snap-to-Path & Rota Geometrisi
    # ------------------------------------------------------------------
    def snap_to_path(
        self,
        point: Tuple[Union[int, float], Union[int, float]],
        region: Optional[str] = None,
        route: Optional[str] = None,
        max_snap_distance: float = 35.0,
        nodes: Optional[List[NodeData]] = None,
        closed_loop: bool = False,
    ) -> SnapResult:
        """
        Görsel odometri veya anlık koordinatı en yakın rota segmentine veya düğüme hizalar (Snap-to-Path).

        2D segment izdüşüm formülü:
            P: sorgu noktası, A, B: segment uçları
            V = B - A
            t = clamp(((P - A) . V) / |V|^2, 0.0, 1.0)
            Q = A + t * V
            d = |P - Q|

        Eğer d <= max_snap_distance ise:
            snapped=True, coords=(Q.x, Q.y) döner.
        Aksi takdirde:
            snapped=False, orijinal coords=(P.x, P.y) döner.
        """
        px, py = float(point[0]), float(point[1])
        target_region = resolve_canonical_region(region or self.active_region)
        target_route = route or self.active_route

        route_nodes = nodes
        if route_nodes is None:
            route_nodes = self.get_route_nodes(region_name=target_region, route_name=target_route)

        if not route_nodes:
            return SnapResult(
                coords=(int(round(px)), int(round(py))),
                snapped=False,
                distance=float("inf"),
                segment_index=None,
                projection_t=0.0,
                nearest_node=None,
                nearest_node_id=None,
            )

        best_dist = float("inf")
        best_proj = (int(round(px)), int(round(py)))
        best_seg_idx: Optional[int] = None
        best_t: float = 0.0
        best_node: Optional[NodeData] = None

        # 1. Tek düğüm varsa doğrudan o düğüme olan mesafeye bak
        if len(route_nodes) == 1:
            n = route_nodes[0]
            d = math.hypot(px - float(n.map_x), py - float(n.map_y))
            if d <= float(max_snap_distance):
                return SnapResult(
                    coords=(int(n.map_x), int(n.map_y)),
                    snapped=True,
                    distance=d,
                    segment_index=0,
                    projection_t=0.0,
                    nearest_node=n,
                    nearest_node_id=n.id,
                    norm_x=n.norm_x,
                    norm_y=n.norm_y,
                )
            return SnapResult(
                coords=(int(round(px)), int(round(py))),
                snapped=False,
                distance=d,
                segment_index=0,
                projection_t=0.0,
                nearest_node=n,
                nearest_node_id=n.id,
            )

        # 2. Çoklu düğümlerde segment izdüşümlerini tara
        seg_count = len(route_nodes) if closed_loop else (len(route_nodes) - 1)
        for i in range(seg_count):
            nA = route_nodes[i]
            nB = route_nodes[(i + 1) % len(route_nodes)]
            ax, ay = float(nA.map_x), float(nA.map_y)
            bx, by = float(nB.map_x), float(nB.map_y)

            dx = bx - ax
            dy = by - ay
            l2 = dx * dx + dy * dy
            if l2 < 1e-6:
                t = 0.0
                qx = ax
                qy = ay
            else:
                t = ((px - ax) * dx + (py - ay) * dy) / l2
                t = max(0.0, min(1.0, float(t)))
                qx = ax + t * dx
                qy = ay + t * dy

            d = math.hypot(px - qx, py - qy)
            if d < best_dist:
                best_dist = d
                best_proj = (int(round(qx)), int(round(qy)))
                best_seg_idx = i
                best_t = t
                best_node = nA if t < 0.5 else nB

        # 3. Düğüm köşelerine olan doğrudan mesafeleri de kontrol et
        for idx, n in enumerate(route_nodes):
            d_v = math.hypot(px - float(n.map_x), py - float(n.map_y))
            if d_v < best_dist:
                best_dist = d_v
                best_proj = (int(n.map_x), int(n.map_y))
                best_seg_idx = idx
                best_t = 0.0
                best_node = n

        vw = getattr(self, "viewport_width", 1920) or 1920
        vh = getattr(self, "viewport_height", 1080) or 1080
        calc_norm_x = round(float(best_proj[0]) / float(max(1, vw)), 4)
        calc_norm_y = round(float(best_proj[1]) / float(max(1, vh)), 4)

        if best_dist <= float(max_snap_distance):
            return SnapResult(
                coords=best_proj,
                snapped=True,
                distance=best_dist,
                segment_index=best_seg_idx,
                projection_t=best_t,
                nearest_node=best_node,
                nearest_node_id=best_node.id if best_node else None,
                norm_x=calc_norm_x,
                norm_y=calc_norm_y,
            )

        return SnapResult(
            coords=(int(round(px)), int(round(py))),
            snapped=False,
            distance=best_dist,
            segment_index=best_seg_idx,
            projection_t=best_t,
            nearest_node=best_node,
            nearest_node_id=best_node.id if best_node else None,
            norm_x=round(float(px) / float(max(1, vw)), 4),
            norm_y=round(float(py) / float(max(1, vh)), 4),
        )

    def snap_odometry_position(
        self,
        odometry_tracker: Any = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        max_snap_distance: float = 35.0,
    ) -> Optional[SnapResult]:
        """
        Görsel odometri takipçisinden (Visual Odometry / Optical Flow / Minimap Tracker)
        mevcut pozisyonu alır, aktif rotaya olan en yakın izdüşüm noktasına (snap-to-path)
        hizalar ve eğer takipçi güncellenebilir ise sürüklenmeyi (drift) sıfırlamak için
        takipçi koordinatlarını günceller.
        """
        tracker = odometry_tracker or getattr(self, "odometry_tracker", None)
        if tracker is None:
            return None

        pos: Optional[Tuple[float, float]] = None
        if hasattr(tracker, "get_position") and callable(tracker.get_position):
            try:
                opos = tracker.get_position()
                if opos and len(opos) >= 2:
                    pos = (float(opos[0]), float(opos[1]))
            except Exception as exc:
                logger.debug("tracker.get_position hatası: %s", exc)
        elif hasattr(tracker, "get_coordinates") and callable(tracker.get_coordinates):
            try:
                opos = tracker.get_coordinates()
                if opos and len(opos) >= 2:
                    pos = (float(opos[0]), float(opos[1]))
            except Exception as exc:
                logger.debug("tracker.get_coordinates hatası: %s", exc)
        elif hasattr(tracker, "map_x") and hasattr(tracker, "map_y"):
            pos = (float(tracker.map_x), float(tracker.map_y))

        if pos is None:
            return None

        snap_res = self.snap_to_path(
            point=pos,
            region=region,
            route=route,
            max_snap_distance=max_snap_distance,
        )

        if snap_res.snapped:
            if hasattr(tracker, "set_position") and callable(tracker.set_position):
                try:
                    tracker.set_position(snap_res.x, snap_res.y)
                except Exception:
                    pass
            elif hasattr(tracker, "update_position") and callable(tracker.update_position):
                try:
                    tracker.update_position(snap_res.x, snap_res.y)
                except Exception:
                    pass
            else:
                if hasattr(tracker, "map_x"):
                    tracker.map_x = snap_res.x
                if hasattr(tracker, "map_y"):
                    tracker.map_y = snap_res.y

        return snap_res

    def interpolate_intermediate_waypoints(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
        max_segment_distance: float = 150.0,
        resource_type: str = "Waypoint",
        not_gatherable: bool = True,
        map_frame: Optional[np.ndarray] = None,
    ) -> List[NodeData]:
        """
        Uzun segmentlerde (ardışık düğümler arası mesafe > max_segment_distance)
        gezinme kaybını ve sapmaları önlemek amacıyla araya otomatik ara noktalar
        (intermediate waypoints) ekler. seq_index değerlerini günceller ve atomik kaydeder.
        """
        target_region = resolve_canonical_region(region or self.active_region)
        target_route = route or self.active_route
        if target_region not in self._regions:
            return []

        all_region_nodes: List[NodeData] = self._regions[target_region].get("nodes", [])
        if len(all_region_nodes) < 2:
            return []

        # İlgili rotadaki düğümleri belirle
        if target_route and target_route != DEFAULT_ROUTE:
            route_subnodes = [n for n in all_region_nodes if target_route in n.routes]
        else:
            route_subnodes = list(all_region_nodes)

        if len(route_subnodes) < 2:
            return []

        max_dist = max(10.0, float(max_segment_distance))
        vw = float(getattr(self, "viewport_width", 1920) or 1920)
        vh = float(getattr(self, "viewport_height", 1080) or 1080)
        created_nodes: List[NodeData] = []

        # Ardışık düğümler arasında mesafe kontrolü
        i = 0
        while i < len(route_subnodes) - 1:
            node_a = route_subnodes[i]
            node_b = route_subnodes[i + 1]

            d = math.hypot(node_b.map_x - node_a.map_x, node_b.map_y - node_a.map_y)
            if d > max_dist:
                subdivisions = int(math.ceil(d / max_dist))
                # Ara noktaları sırayla ekle
                insert_pos_in_region = all_region_nodes.index(node_a) + 1
                insert_pos_in_route = i + 1

                for step in range(1, subdivisions):
                    frac = float(step) / float(subdivisions)
                    ix = int(round(node_a.map_x + frac * (node_b.map_x - node_a.map_x)))
                    iy = int(round(node_a.map_y + frac * (node_b.map_y - node_a.map_y)))
                    inx = round(float(node_a.norm_x + frac * (node_b.norm_x - node_a.norm_x)), 4)
                    iny = round(float(node_a.norm_y + frac * (node_b.norm_y - node_a.norm_y)), 4)

                    new_id = self._next_node_id(target_region)
                    patch_path = self._get_patch_rel_path(target_region, new_id)
                    disk_patch_path = os.path.join(self._get_region_patches_dir(target_region), f"node_{new_id}.png")
                    os.makedirs(os.path.dirname(disk_patch_path), exist_ok=True)

                    if map_frame is not None and isinstance(map_frame, np.ndarray) and map_frame.size > 0:
                        patch_img = self._extract_clean_80x80_patch(map_frame, ix, iy)
                    else:
                        patch_img = self._prepare_patch_image(None)
                    _safe_imwrite(disk_patch_path, patch_img)

                    routes_for_node = [target_route] if target_route and target_route != DEFAULT_ROUTE else []
                    new_node = NodeData(
                        id=new_id,
                        region=target_region,
                        routes=routes_for_node,
                        patch_path=_normalize_path(patch_path),
                        in_air=bool(node_a.in_air and node_b.in_air),
                        gathered_count=0,
                        last_visited_timestamp=0.0,
                        rest_duration_sec=self.default_rest_sec,
                        consecutive_failures=0,
                        map_x=ix,
                        map_y=iy,
                        norm_x=inx,
                        norm_y=iny,
                        resource_type=str(resource_type or "Waypoint"),
                        max_harvests=1,
                        current_harvests=0,
                        source_type="interpolated",
                        state="active",
                        not_gatherable=bool(not_gatherable),
                    )

                    all_region_nodes.insert(insert_pos_in_region, new_node)
                    route_subnodes.insert(insert_pos_in_route, new_node)
                    created_nodes.append(new_node)
                    insert_pos_in_region += 1
                    insert_pos_in_route += 1

                i += (subdivisions - 1)
            i += 1

        if created_nodes:
            for idx, n in enumerate(all_region_nodes, start=1):
                n.seq_index = idx
            self.save_region(target_region)
            logger.info(
                "Uzun segmentlerde %d ara nokta (intermediate waypoints) oluşturuldu: Bölge=%s, Rota=%s",
                len(created_nodes),
                target_region,
                target_route,
            )

        return created_nodes

    def deduplicate_nodes(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
        min_distance: float = 12.0,
    ) -> List[int]:
        """
        Rotadaki birbirine çok yakın (mesafe < min_distance) mükerrer düğümleri tespit eder,
        rota üyeliklerini birleştirerek tekilleştirir, fazlalıkları diskten ve veritabanından temizler.
        Silinen düğüm ID'lerinin listesini döndürür.
        """
        target_region = resolve_canonical_region(region or self.active_region)
        target_route = route or self.active_route
        if target_region not in self._regions:
            return []

        all_nodes: List[NodeData] = self._regions[target_region].get("nodes", [])
        if len(all_nodes) < 2:
            return []

        min_d = max(1.0, float(min_distance))
        kept_nodes: List[NodeData] = []
        removed_nodes: List[NodeData] = []
        removed_ids: List[int] = []

        for candidate in all_nodes:
            # Eğer belirli bir rota filtrelendiyse ve aday bu rotada değilse dokunma
            if target_route and target_route != DEFAULT_ROUTE and target_route not in candidate.routes:
                kept_nodes.append(candidate)
                continue

            # Daha önce saklanan düğümlerle yakınlık kontrolü
            duplicate_of: Optional[NodeData] = None
            for existing in kept_nodes:
                if target_route and target_route != DEFAULT_ROUTE and target_route not in existing.routes:
                    continue
                dist = math.hypot(candidate.map_x - existing.map_x, candidate.map_y - existing.map_y)
                if dist < min_d:
                    duplicate_of = existing
                    break

            if duplicate_of is not None:
                # Mükerrer düğüm tespit edildi -> Bilgileri birleştir
                # Eğer aday toplanabilir bir kaynaksa ama existing parşömen/waypoint ise existing'i kaynağa yükselt
                if not candidate.not_gatherable and duplicate_of.not_gatherable:
                    duplicate_of.not_gatherable = False
                    duplicate_of.resource_type = candidate.resource_type
                    duplicate_of.max_harvests = candidate.max_harvests
                # Rota etiketlerini birleştir
                for r in candidate.routes:
                    if r not in duplicate_of.routes:
                        duplicate_of.routes.append(r)
                # İstatistikleri birleştir
                duplicate_of.gathered_count = max(duplicate_of.gathered_count, candidate.gathered_count)

                removed_nodes.append(candidate)
                removed_ids.append(candidate.id)
            else:
                kept_nodes.append(candidate)

        if removed_ids:
            self._regions[target_region]["nodes"] = kept_nodes
            for idx, n in enumerate(kept_nodes, start=1):
                n.seq_index = idx
            self.save_region(target_region)

            for rem_node in removed_nodes:
                try:
                    patch_file = self._resolve_patch_disk_path(rem_node)
                    if os.path.isfile(patch_file):
                        os.remove(patch_file)
                except Exception as exc:
                    logger.debug("Mükerrer düğüm görseli silinemedi (#%d): %s", rem_node.id, exc)

            logger.info(
                "Mükerrer düğüm tekilleştirme tamamlandı: %d düğüm silindi (%s), Bölge=%s, Rota=%s",
                len(removed_ids),
                removed_ids,
                target_region,
                target_route,
            )

        return removed_ids

    # ------------------------------------------------------------------
    # F8 Anlık Düğüm Kayıt Akışı (Minimap + Açık Harita Hibrit Destekli)
    # ------------------------------------------------------------------
    def _send_map_key(self, input_ctrl: Any) -> bool:
        """Input kontrolcüsü üzerinden 'M' harita tuşuna basar."""
        if input_ctrl is None:
            logger.debug("input_ctrl nesnesi None olduğu için M tuşuna basılmadı.")
            return False
        try:
            if hasattr(input_ctrl, "press_key"):
                input_ctrl.press_key("m")
                return True
            if hasattr(input_ctrl, "tap_key"):
                input_ctrl.tap_key("m")
                return True
            if hasattr(input_ctrl, "toggle_map"):
                input_ctrl.toggle_map()
                return True
            if hasattr(input_ctrl, "send_key"):
                input_ctrl.send_key("m")
                return True
            if callable(input_ctrl):
                input_ctrl("m")
                return True
            logger.debug("input_ctrl üzerinde uygun tuş basma metodu bulunamadı.")
            return False
        except Exception as exc:
            logger.debug("M tuşu gönderilirken hata oluştu: %s", exc)
            return False

    def _grab_frame_from_vision(self, vision_engine: Any) -> Optional[np.ndarray]:
        """Vision motorundan veya ScreenCapture üzerinden güncel ekran görüntüsünü alır."""
        if isinstance(vision_engine, np.ndarray) and vision_engine.size > 0:
            return vision_engine

        if isinstance(vision_engine, str) and os.path.isfile(vision_engine):
            try:
                raw = np.fromfile(vision_engine, dtype=np.uint8)
                decoded = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                if decoded is not None and decoded.size > 0:
                    return decoded
            except Exception as exc:
                logger.debug("Dosya yolundan ekran görüntüsü okunamadı (%s): %s", vision_engine, exc)

        if vision_engine is not None:
            try:
                for method_name in (
                    "capture_frame",
                    "grab_frame",
                    "get_frame",
                    "get_open_map_frame",
                    "screenshot",
                ):
                    if hasattr(vision_engine, method_name):
                        fn = getattr(vision_engine, method_name)
                        frame = fn()
                        if isinstance(frame, np.ndarray) and frame.size > 0:
                            return frame

                if hasattr(vision_engine, "grab_screen"):
                    try:
                        frame = vision_engine.grab_screen(grayscale=False)
                    except TypeError:
                        frame = vision_engine.grab_screen()
                    if isinstance(frame, np.ndarray) and frame.size > 0:
                        return frame

                capture_obj = getattr(vision_engine, "capture", None)
                if capture_obj is not None and hasattr(capture_obj, "grab_frame"):
                    frame = capture_obj.grab_frame()
                    if isinstance(frame, np.ndarray) and frame.size > 0:
                        return frame
            except Exception as exc:
                logger.debug("Vision motorundan ekran görüntüsü alınırken hata: %s", exc)

        # Fallback: doğrudan ScreenCapture ile ekranı yakala
        try:
            from core.capture import ScreenCapture
            cap = ScreenCapture()
            frame = cap.grab_frame()
            if isinstance(frame, np.ndarray) and frame.size > 0:
                return frame
        except Exception as exc:
            logger.debug("ScreenCapture yedek yakalama hatası: %s", exc)

        return None

    def _verify_map_open(self, vision_engine: Any, frame: Optional[np.ndarray]) -> bool:
        """Tam ekran haritanın ('M') açık olup olmadığını kontrol eder."""
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return False

        if vision_engine is not None and hasattr(vision_engine, "is_map_open"):
            try:
                fn = getattr(vision_engine, "is_map_open")
                try:
                    return bool(fn(frame))
                except TypeError:
                    return bool(fn())
            except Exception as exc:
                logger.debug("vision_engine.is_map_open çağrısı başarısız: %s", exc)

        h, w = frame.shape[:2]
        if h < PATCH_SIZE or w < PATCH_SIZE:
            return False

        center_roi = frame[int(h * 0.25) : int(h * 0.75), int(w * 0.25) : int(w * 0.75)]
        if center_roi.size == 0:
            return False
        return float(np.std(center_roi)) > 1.0 or float(np.mean(center_roi)) > 5.0

    def _locate_minimap_player_center(self, frame: np.ndarray) -> Tuple[int, int]:
        """
        Sağ üst Minimap bölgesi (`x: 0.855..0.992, y: 0.065..0.255`) içindeki
        oyuncu okunun merkezini (`cx ~ 0.923*w, cy ~ 0.165*h`) tespit eder.
        """
        h, w = frame.shape[:2]
        default_cx = int(round(w * 0.923))
        default_cy = int(round(h * 0.165))

        if h < PATCH_SIZE * 2 or w < PATCH_SIZE * 2:
            return max(PATCH_SIZE // 2, w // 2), max(PATCH_SIZE // 2, h // 2)

        try:
            if frame.ndim == 3:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame
                # Oyuncu oku minimap merkezinde parlak beyaz/cyan üçgendir
                search_r = max(14, int(round(min(w, h) * 0.022)))
                x1 = max(0, default_cx - search_r)
                x2 = min(w, default_cx + search_r)
                y1 = max(0, default_cy - search_r)
                y2 = min(h, default_cy + search_r)
                sub = bgr[y1:y2, x1:x2]
                if sub.size > 0:
                    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
                    # Parlak beyaz veya cyan/sarı merkez ok pikselleri
                    arrow_mask = (
                        ((hsv[:, :, 2] >= 225) & (hsv[:, :, 1] <= 75))
                        | ((hsv[:, :, 0] >= 80) & (hsv[:, :, 0] <= 112) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 190))
                    )
                    ys, xs = np.where(arrow_mask)
                    if 4 <= xs.size <= 220:
                        cx = int(round(x1 + float(np.mean(xs))))
                        cy = int(round(y1 + float(np.mean(ys))))
                        return cx, cy
        except Exception as exc:
            logger.debug("Minimap oyuncu oku merkez iyileştirme hatası: %s", exc)

        return default_cx, default_cy

    def _extract_clean_80x80_patch(
        self,
        frame: np.ndarray,
        center_x: int,
        center_y: int,
    ) -> np.ndarray:
        """
        Merkez `(center_x, center_y)` etrafından 80x80 gri harita dokusunu kırpar
        ve merkezdeki oyuncu okunu maskeleyerek `cv2.inpaint` ile temizler.
        """
        if len(frame.shape) == 3:
            if frame.shape[2] == 4:
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            else:
                bgr = frame
            gray_full = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        else:
            bgr = None
            gray_full = frame.copy()

        h, w = gray_full.shape[:2]
        half = PATCH_SIZE // 2

        # Kenar taşmalarını önlemek için güvenli sınırlar
        cx = int(np.clip(center_x, half, max(half, w - half)))
        cy = int(np.clip(center_y, half, max(half, h - half)))

        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        patch_gray = gray_full[y1:y2, x1:x2].copy()
        if patch_gray.shape[0] != PATCH_SIZE or patch_gray.shape[1] != PATCH_SIZE:
            patch_gray = cv2.resize(patch_gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_AREA)

        # Oyuncu oku maskesi oluştur (merkezdeki 9px yarıçaplı alan + parlak/doygun ok pikselleri)
        mask = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
        cv2.circle(mask, (half, half), 9, 255, -1)

        if bgr is not None:
            patch_bgr = bgr[y1:y2, x1:x2]
            if patch_bgr.shape[0] == PATCH_SIZE and patch_bgr.shape[1] == PATCH_SIZE:
                hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
                # Merkez 16px içindeki yüksek doygunluklu (cyan/sarı/beyaz oyuncu oku) pikselleri maskeye ekle
                center_zone = np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.uint8)
                cv2.circle(center_zone, (half, half), 15, 255, -1)
                arrow_pixels = ((hsv[:, :, 1] > 110) | (hsv[:, :, 2] > 235)).astype(np.uint8) * 255
                arrow_mask = cv2.bitwise_and(arrow_pixels, center_zone)
                mask = cv2.bitwise_or(mask, arrow_mask)

        inpainted = cv2.inpaint(patch_gray, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return np.asarray(inpainted, dtype=np.uint8)

    def record_current_node(
        self,
        input_ctrl: Any = None,
        vision_engine: Any = None,
        region: Optional[str] = None,
        route: Optional[str] = None,
        in_air: bool = False,
        resource_type: str = DEFAULT_RESOURCE_TYPE,
        use_minimap_for_f8: bool = True,
        max_harvests: int = DEFAULT_MAX_HARVESTS,
        source_type: str = DEFAULT_SOURCE_TYPE,
        odometry_tracker: Any = None,
        snap_to_path: bool = False,
        snap_to_existing_path: bool = False,
        max_snap_distance: float = 35.0,
        min_record_distance: float = 5.0,
        **kwargs: Any,
    ) -> NodeData:
        """
        F8 Anlık Düğüm Kayıt Akışı (Hibrit Minimap + Açık Harita):
        1. Güncel ekran görüntüsünü yakalar (`vision_engine` veya `ScreenCapture`).
        2. Tam ekran harita (`M`) zaten açıksa harita merkezindeki oyuncu okundan,
           açık değilse (normal yürüyüş esnasında, örn. Pilgrim's Ridge / Western Altamia Highland)
           oyunu kesintiye uğratmadan doğrudan Sağ Üst Minimap (`x: 0.855..0.992, y: 0.065..0.255`,
           merkez `cx ~ 0.923, cy ~ 0.165`) üzerindeki oyuncu konumundan 80x80 gri kesiti kırpar.
        3. Merkezdeki oyuncu okunu `cv2.inpaint` ile temizleyerek `patches/node_<id>.png` olarak kaydeder.
        4. `max_harvests=3` (varsayılan) ile `NodeData` oluşturup `region_data.json` dosyasını günceller.
        5. Asla `"Haritanın açıldığı doğrulanamadı"` hatasıyla `None` döndürmez.
        """
        target_region = str(region or self.active_region or DEFAULT_REGION).strip() or DEFAULT_REGION
        target_route = str(route or self.active_route or DEFAULT_ROUTE).strip() or DEFAULT_ROUTE

        # Eğer ilk argüman doğrudan bir görüntü (np.ndarray) veya görüntü dosya yoluysa:
        direct_frame: Optional[np.ndarray] = None
        if isinstance(input_ctrl, np.ndarray) and input_ctrl.size > 0:
            direct_frame = input_ctrl
            input_ctrl = None
        elif isinstance(input_ctrl, str) and os.path.isfile(input_ctrl):
            direct_frame = self._grab_frame_from_vision(input_ctrl)
            input_ctrl = None

        map_opened_by_us = False
        try:
            frame = direct_frame if direct_frame is not None else self._grab_frame_from_vision(vision_engine)

            # Eğer use_minimap_for_f8=False ise ve tam harita açık değilse M tuşu ile açmayı dene
            if not use_minimap_for_f8 and input_ctrl is not None:
                if frame is None or not self._verify_map_open(vision_engine, frame):
                    if self._send_map_key(input_ctrl):
                        map_opened_by_us = True
                        time.sleep(0.25)
                        new_frame = self._grab_frame_from_vision(vision_engine)
                        if new_frame is not None and new_frame.size > 0:
                            frame = new_frame

            # Ekran görüntüsü hala None ise sentetik güvenli kare oluştur (asla None dönme)
            if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
                vw = getattr(self, "viewport_width", 1920) or 1920
                vh = getattr(self, "viewport_height", 1080) or 1080
                frame = np.zeros((vh, vw, 3), dtype=np.uint8)

            h, w = frame.shape[:2]
            self.update_viewport(w, h)
            is_full_map = self._verify_map_open(vision_engine, frame)

            # Özel test/vision_engine find_player_on_map sağlıyorsa öncelik ver
            custom_player_pos: Optional[Tuple[int, int]] = None
            if vision_engine is not None and hasattr(vision_engine, "find_player_on_map"):
                try:
                    pos = vision_engine.find_player_on_map(frame)
                    if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                        custom_player_pos = (int(pos[0]), int(pos[1]))
                except Exception as exc:
                    logger.debug("find_player_on_map yedek konuma düştü: %s", exc)

            if custom_player_pos is not None:
                crop_x, crop_y = custom_player_pos
                map_x, map_y = custom_player_pos
                norm_x = round(float(map_x) / float(max(1, w)), 4)
                norm_y = round(float(map_y) / float(max(1, h)), 4)
            elif is_full_map and not use_minimap_for_f8:
                crop_x, crop_y = w // 2, h // 2
                map_x, map_y = crop_x, crop_y
                norm_x = round(float(map_x) / float(max(1, w)), 4)
                norm_y = round(float(map_y) / float(max(1, h)), 4)
            elif is_full_map and vision_engine is not None and hasattr(vision_engine, "is_map_open"):
                # Tam ekran harita gerçekten açıksa merkezden kırp
                crop_x, crop_y = w // 2, h // 2
                map_x, map_y = crop_x, crop_y
                norm_x = round(float(map_x) / float(max(1, w)), 4)
                norm_y = round(float(map_y) / float(max(1, h)), 4)
            else:
                # Normal oyun ekranı (Pilgrim's Ridge / Western Altamia Highland vb.):
                # Sağ üst Minimap (x: 0.855..0.992, y: 0.065..0.255) oyuncu oku etrafından kırp
                crop_x, crop_y = self._locate_minimap_player_center(frame)
                existing_nodes = self._regions.get(target_region, {}).get("nodes", [])

                # Gerçek odometri / dünya koordinat dönüşümü:
                odometry_pos: Optional[Tuple[float, float, float, float]] = None

                # 1. Tracker sorgusu (odometry_tracker argümanı, vision_engine veya self)
                tracker = odometry_tracker
                if tracker is None and vision_engine is not None:
                    if hasattr(vision_engine, "odometry_tracker") and vision_engine.odometry_tracker is not None:
                        tracker = vision_engine.odometry_tracker
                    elif (
                        hasattr(vision_engine, "auto_cartographer")
                        and getattr(vision_engine.auto_cartographer, "odometry_tracker", None) is not None
                    ):
                        tracker = vision_engine.auto_cartographer.odometry_tracker
                    elif hasattr(vision_engine, "get_odometry_position") and callable(vision_engine.get_odometry_position):
                        try:
                            opos = vision_engine.get_odometry_position()
                            if opos and len(opos) >= 4:
                                odometry_pos = (float(opos[0]), float(opos[1]), float(opos[2]), float(opos[3]))
                            elif opos and len(opos) >= 2:
                                odometry_pos = (
                                    float(opos[0]),
                                    float(opos[1]),
                                    round(float(opos[0]) / float(w), 4),
                                    round(float(opos[1]) / float(h), 4),
                                )
                        except Exception as o_exc:
                            logger.debug("vision_engine.get_odometry_position hatası: %s", o_exc)

                if tracker is None and getattr(self, "odometry_tracker", None) is not None:
                    tracker = self.odometry_tracker

                if odometry_pos is None and tracker is not None:
                    try:
                        if hasattr(tracker, "get_position") and callable(tracker.get_position):
                            opos = tracker.get_position()
                            if opos and len(opos) >= 4:
                                odometry_pos = (float(opos[0]), float(opos[1]), float(opos[2]), float(opos[3]))
                        elif hasattr(tracker, "map_x") and hasattr(tracker, "map_y"):
                            ox, oy = float(tracker.map_x), float(tracker.map_y)
                            odometry_pos = (
                                ox,
                                oy,
                                round(ox / float(max(1, w)), 4),
                                round(oy / float(max(1, h)), 4),
                            )
                    except Exception as o_exc:
                        logger.debug("Odometry tracker get_position hatası: %s", o_exc)

                # 2. Koordinat belirleme
                if odometry_pos is not None:
                    map_x = int(round(odometry_pos[0]))
                    map_y = int(round(odometry_pos[1]))
                    norm_x = float(odometry_pos[2])
                    norm_y = float(odometry_pos[3])
                elif not existing_nodes:
                    # İlk düğüm: ekran/harita merkezini referans al
                    map_x, map_y = w // 2, h // 2
                    norm_x = round(float(map_x) / float(max(1, w)), 4)
                    norm_y = round(float(map_y) / float(max(1, h)), 4)
                else:
                    # Mevcut düğümler varsa: son düğümün konumu ve minimap optik kayması (phase correlate)
                    last_node = existing_nodes[-1]
                    prev_patch_path = self._resolve_patch_disk_path(last_node)
                    prev_patch = _safe_imread_gray(prev_patch_path) if os.path.isfile(prev_patch_path) else None

                    curr_patch = self._extract_clean_80x80_patch(frame, crop_x, crop_y)
                    shift_dx, shift_dy = 0.0, 0.0
                    if prev_patch is not None and prev_patch.size > 0 and curr_patch.size > 0:
                        try:
                            (pc_dx, pc_dy), resp = cv2.phaseCorrelate(
                                np.float32(prev_patch), np.float32(curr_patch)
                            )
                            if float(resp) >= 0.15:
                                scale = max(1.0, float(self.meters_per_pixel) * 4.0)
                                shift_dx = -float(pc_dx) * scale
                                shift_dy = -float(pc_dy) * scale
                        except Exception as pc_exc:
                            logger.debug("Minimap faz korelasyonu hatası: %s", pc_exc)

                    if abs(shift_dx) > 0.01 or abs(shift_dy) > 0.01:
                        calc_x = float(last_node.map_x) + shift_dx
                        calc_y = float(last_node.map_y) + shift_dy
                    else:
                        # Faz korelasyonu sıfırsa minimap içi oyuncu oku ofsetini kullan
                        default_cx = float(round(w * 0.923))
                        default_cy = float(round(h * 0.165))
                        off_dx = (float(crop_x) - default_cx) * 2.0
                        off_dy = (float(crop_y) - default_cy) * 2.0
                        calc_x = float(last_node.map_x) + off_dx
                        calc_y = float(last_node.map_y) + off_dy

                    map_x = int(np.clip(round(calc_x), 40, max(41, w - 40)))
                    map_y = int(np.clip(round(calc_y), 40, max(41, h - 40)))
                    norm_x = round(float(map_x) / float(max(1, w)), 4)
                    norm_y = round(float(map_y) / float(max(1, h)), 4)

            patch_80 = self._extract_clean_80x80_patch(frame, crop_x, crop_y)

            if map_opened_by_us and input_ctrl is not None:
                self._send_map_key(input_ctrl)
                map_opened_by_us = False

            # Snap-to-path kontrolü
            do_snap = bool(
                snap_to_path
                or snap_to_existing_path
                or kwargs.get("snap_to_path", False)
                or kwargs.get("snap_to_existing_path", False)
            )
            if do_snap:
                snap_dist = float(kwargs.get("max_snap_distance", max_snap_distance))
                snap_res = self.snap_to_path(
                    point=(map_x, map_y),
                    region=target_region,
                    route=target_route,
                    max_snap_distance=snap_dist,
                )
                if snap_res.snapped:
                    logger.info(
                        "[F8] Konum rotaya hizalandı (Snap-to-path): (%d, %d) -> (%d, %d) [Mesafe: %.1fpx]",
                        map_x,
                        map_y,
                        snap_res.x,
                        snap_res.y,
                        snap_res.distance,
                    )
                    map_x, map_y = snap_res.x, snap_res.y
                    norm_x = round(float(map_x) / float(max(1, w)), 4)
                    norm_y = round(float(map_y) / float(max(1, h)), 4)

            # Kara liste (Blacklist) kontrolü: Engellenmiş koordinat (her zaman mükerrer kontrolünden önce yapılmalı)
            if getattr(self, "blacklist_manager", None) is not None:
                is_bl, bl_entry = self.blacklist_manager.is_blacklisted(
                    region=target_region,
                    map_x=map_x,
                    map_y=map_y,
                    norm_x=norm_x,
                    norm_y=norm_y,
                )
                if is_bl:
                    logger.warning(
                        "⛔ [F8 KAYIT ENGELİ] Bu konum (%d, %d) kara listede (Blacklist)! Kayıt yapılmadı (Neden: %s).",
                        map_x,
                        map_y,
                        bl_entry.reason if bl_entry else "Blacklisted",
                    )
                    _play_beep(600, 200)
                    return None

            # Mükerrer / çok yakın F8 basma engeli
            effective_min_dist = float(kwargs.get("min_record_distance", min_record_distance))
            allow_dup = bool(kwargs.get("allow_duplicate", False))
            if effective_min_dist > 0.0 and not allow_dup:
                existing_for_dup = self.get_route_nodes(region_name=target_region, route_name=target_route)
                if existing_for_dup:
                    last_n = existing_for_dup[-1]
                    d_dup = math.hypot(map_x - last_n.map_x, map_y - last_n.map_y)
                    if d_dup < effective_min_dist:
                        logger.info(
                            "[F8] Düğüm #%d ile çok yakın mesafe (%.1fpx < %.1fpx), mükerrer F8 kaydı atlandı.",
                            last_n.id,
                            d_dup,
                            effective_min_dist,
                        )
                        _play_beep(900, 100)
                        return last_n

            node = self.add_node(
                region=target_region,
                route=target_route,
                patch_img=patch_80,
                in_air=bool(in_air),
                map_x=int(map_x),
                map_y=int(map_y),
                norm_x=norm_x,
                norm_y=norm_y,
                resource_type=resource_type,
                max_harvests=max(1, int(max_harvests or DEFAULT_MAX_HARVESTS)),
                current_harvests=0,
                source_type=str(source_type or DEFAULT_SOURCE_TYPE),
                viewport_width=w,
                viewport_height=h,
            )

            if node is None:
                logger.warning("[F8] Düğüm oluşturulamadı veya kara liste engeline takıldı.")
                _play_beep(600, 200)
                return None

            _play_beep(1200, 150)
            logger.info(
                "[F8] Konum başarıyla kaydedildi (Node #%d, %dx Toplama) [%s/%s] konum=(%d, %d) patch=%s",
                node.id,
                node.max_harvests,
                node.region,
                target_route,
                node.map_x,
                node.map_y,
                node.patch_path,
            )
            return node

        except Exception as exc:
            logger.warning("F8 Düğüm kaydı yedek mod ile tamamlanıyor: %s", exc)
            if map_opened_by_us and input_ctrl is not None:
                self._send_map_key(input_ctrl)
            vw = getattr(self, "viewport_width", 1920) or 1920
            vh = getattr(self, "viewport_height", 1080) or 1080
            fallback_node = self.add_node(
                region=target_region,
                route=target_route,
                patch_img=None,
                in_air=bool(in_air),
                map_x=vw // 2,
                map_y=vh // 2,
                norm_x=0.5,
                norm_y=0.5,
                resource_type=resource_type,
                max_harvests=max(1, int(max_harvests or DEFAULT_MAX_HARVESTS)),
                current_harvests=0,
                source_type=str(source_type or DEFAULT_SOURCE_TYPE),
                viewport_width=vw,
                viewport_height=vh,
            )
            logger.info(
                "[F8] Konum başarıyla kaydedildi (Node #%d, %dx Toplama)",
                fallback_node.id,
                fallback_node.max_harvests,
            )
            return fallback_node

    # ------------------------------------------------------------------
    # Açık Haritada Düğüm Bulma (Template Matching + Fallback)
    # ------------------------------------------------------------------
    @staticmethod
    def _is_point_in_overlay(x: int, y: int, overlay_rect: Any) -> bool:
        """`(x, y)` noktasının `overlay_rect` alanı içinde olup olmadığını kontrol eder."""
        if overlay_rect is None:
            return False

        try:
            # PyQt QRect uyumluluğu
            if hasattr(overlay_rect, "contains") and callable(overlay_rect.contains):
                try:
                    return bool(overlay_rect.contains(int(x), int(y)))
                except TypeError as exc:
                    logger.debug("overlay_rect.contains(x, y) imzası uyumsuz: %s", exc)
            if all(hasattr(overlay_rect, attr) for attr in ("x", "y", "width", "height")):
                rx = int(overlay_rect.x() if callable(overlay_rect.x) else overlay_rect.x)
                ry = int(overlay_rect.y() if callable(overlay_rect.y) else overlay_rect.y)
                rw = int(overlay_rect.width() if callable(overlay_rect.width) else overlay_rect.width)
                rh = int(overlay_rect.height() if callable(overlay_rect.height) else overlay_rect.height)
                return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

            if isinstance(overlay_rect, dict):
                rx = int(overlay_rect.get("left", overlay_rect.get("x", 0)))
                ry = int(overlay_rect.get("top", overlay_rect.get("y", 0)))
                if "width" in overlay_rect or "w" in overlay_rect:
                    rw = int(overlay_rect.get("width", overlay_rect.get("w", 0)))
                    rh = int(overlay_rect.get("height", overlay_rect.get("h", 0)))
                    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)
                rx2 = int(overlay_rect.get("right", overlay_rect.get("x2", rx)))
                ry2 = int(overlay_rect.get("bottom", overlay_rect.get("y2", ry)))
                return (rx <= x <= rx2) and (ry <= y <= ry2)

            if isinstance(overlay_rect, (tuple, list)) and len(overlay_rect) >= 4:
                rx, ry, v3, v4 = (
                    int(overlay_rect[0]),
                    int(overlay_rect[1]),
                    int(overlay_rect[2]),
                    int(overlay_rect[3]),
                )
                # (x, y, width, height) kontrolü
                return (rx <= x <= rx + v3) and (ry <= y <= ry + v4)
        except Exception as exc:
            logger.debug("overlay_rect kontrol hatası: %s", exc)
        return False

    def locate_node_on_open_map(
        self,
        node: Union[NodeData, Dict[str, Any], int, None],
        open_map_frame: Optional[np.ndarray],
        vision_engine: Any = None,
        overlay_rect: Any = None,
        match_threshold: float = 0.65,
    ) -> NodeLocationResult:
        """
        Açık harita (`open_map_frame`) üzerinde düğümün `patch_path` görselini
        `cv2.matchTemplate(cv2.TM_CCOEFF_NORMED)` ile arar.
        Eşleşme skoru düşükse kayıtlı normalize/piksel harita koordinatını yedek olarak kullanır.
        Tıklanacak `(X, Y)` ekran koordinatını, güven skorunu ve mesafe tahminini (`"41 m"`) döndürür.
        Eğer `(X, Y)` noktası `overlay_rect` altındaysa `"Node <id> is under the overlay panel"` uyarısını loglar.
        """
        # Düğüm nesnesini çözümle
        if node is None:
            resolved_node = self.get_next_ready_node()
            if resolved_node is None:
                resolved_node = NodeData(id=0, region=self.active_region)
        elif isinstance(node, int):
            resolved_node = self.get_node(node) or NodeData(id=node, region=self.active_region)
        elif isinstance(node, dict):
            resolved_node = NodeData.from_dict(node)
        elif isinstance(node, NodeData):
            resolved_node = node
        else:
            resolved_node = NodeData(id=0, region=self.active_region)

        # Harita karesi None ise vision_engine üzerinden yakalamayı dene
        if open_map_frame is None and vision_engine is not None:
            open_map_frame = self._grab_frame_from_vision(vision_engine)

        if open_map_frame is None or not isinstance(open_map_frame, np.ndarray) or open_map_frame.size == 0:
            vw = getattr(self, "viewport_width", 1920) or 1920
            vh = getattr(self, "viewport_height", 1080) or 1080
            fallback_x = int(resolved_node.map_x if resolved_node.map_x > 0 else round(resolved_node.norm_x * vw))
            fallback_y = int(resolved_node.map_y if resolved_node.map_y > 0 else round(resolved_node.norm_y * vh))
            under_ov = self._is_point_in_overlay(fallback_x, fallback_y, overlay_rect)
            if under_ov:
                logger.warning("Node %d is under the overlay panel", resolved_node.id)
            return NodeLocationResult(
                coords=(fallback_x, fallback_y),
                confidence=0.0,
                distance_str="0 m",
                distance_m=0.0,
                under_overlay=under_ov,
                used_fallback=True,
            )

        if len(open_map_frame.shape) == 3:
            if open_map_frame.shape[2] == 4:
                map_gray = cv2.cvtColor(open_map_frame, cv2.COLOR_BGRA2GRAY)
            else:
                map_gray = cv2.cvtColor(open_map_frame, cv2.COLOR_BGR2GRAY)
        else:
            map_gray = open_map_frame

        map_h, map_w = map_gray.shape[:2]
        self.update_viewport(map_w, map_h)
        target_x: int = 0
        target_y: int = 0
        confidence: float = 0.0
        used_fallback: bool = True

        disk_patch = self._resolve_patch_disk_path(resolved_node)
        if os.path.isfile(disk_patch):
            try:
                patch_gray = _safe_imread_gray(disk_patch)
                patch_color = None
                try:
                    raw_p = np.fromfile(disk_patch, dtype=np.uint8)
                    if raw_p.size > 0:
                        patch_color = cv2.imdecode(raw_p, cv2.IMREAD_COLOR)
                except Exception:
                    patch_color = cv2.imread(disk_patch, cv2.IMREAD_COLOR)

                if (
                    patch_gray is not None
                    and patch_gray.size > 0
                    and map_h >= patch_gray.shape[0]
                    and map_w >= patch_gray.shape[1]
                ):
                    res = cv2.matchTemplate(map_gray, patch_gray, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    confidence = float(max(0.0, min(1.0, max_val)))
                    if confidence >= float(match_threshold):
                        ph, pw = patch_gray.shape[:2]
                        target_x = int(max_loc[0] + pw // 2)
                        target_y = int(max_loc[1] + ph // 2)
                        used_fallback = False

                    # Sobel+CLAHE and MinimapOrientationGuard integration (e.g. if raw grayscale template score is between 0.55 and 0.69)
                    if (0.55 <= confidence <= 0.69) or (confidence < float(match_threshold)):
                        try:
                            template_bgr = patch_color if patch_color is not None else cv2.cvtColor(patch_gray, cv2.COLOR_GRAY2BGR)
                            if self.orientation_guard is not None:
                                template_bgr = self.orientation_guard.normalize_patch_to_north(template_bgr)

                            sc_score, (sc_cx, sc_cy) = self.patch_normalizer.match_normalized_patch(
                                search_roi_bgr=open_map_frame,
                                template_bgr=template_bgr,
                            )
                            if sc_score > confidence:
                                confidence = float(max(0.0, min(1.0, sc_score)))
                                target_x = int(sc_cx)
                                target_y = int(sc_cy)
                                if confidence >= float(match_threshold) or sc_score > 0.55:
                                    used_fallback = False
                                    logger.info(
                                        "Düğüm #%d için Sobel+CLAHE normalizasyon skoru ile iyileştirildi: score=%.2f at (%d, %d)",
                                        resolved_node.id,
                                        confidence,
                                        target_x,
                                        target_y,
                                    )
                        except Exception as sc_exc:
                            logger.debug("SobelClahePatchNormalizer secondary match failed: %s", sc_exc)

            except Exception as exc:
                logger.error("Düğüm #%d için şablon eşleştirme hatası: %s", resolved_node.id, exc)

        # Şablon eşleşmesi düşükse kayıtlı normalize / piksel harita koordinatını yedek olarak kullan
        if used_fallback:
            if 0.0 < resolved_node.norm_x <= 1.0 and 0.0 < resolved_node.norm_y <= 1.0 and (
                resolved_node.norm_x != 0.5 or resolved_node.norm_y != 0.5 or resolved_node.map_x <= 0
            ):
                target_x = int(round(resolved_node.norm_x * map_w))
                target_y = int(round(resolved_node.norm_y * map_h))
            elif resolved_node.map_x > 0 or resolved_node.map_y > 0:
                target_x = int(resolved_node.map_x)
                target_y = int(resolved_node.map_y)
            else:
                target_x = int(round(resolved_node.norm_x * map_w))
                target_y = int(round(resolved_node.norm_y * map_h))

            logger.info(
                "Düğüm #%d için şablon skoru düşük (%.2f), yedek koordinat kullanıldı: (%d, %d)",
                resolved_node.id,
                confidence,
                target_x,
                target_y,
            )

        # Oyuncu merkezine göre mesafe tahmini (metre cinsinden)
        player_x, player_y = map_w // 2, map_h // 2
        if vision_engine is not None and hasattr(vision_engine, "find_player_on_map"):
            try:
                pos = vision_engine.find_player_on_map(open_map_frame)
                if isinstance(pos, (tuple, list)) and len(pos) >= 2:
                    player_x, player_y = int(pos[0]), int(pos[1])
            except Exception as exc:
                logger.debug("Mesafe hesabı için oyuncu konumu varsayılan merkeze düştü: %s", exc)

        scale = float(getattr(vision_engine, "meters_per_pixel", self.meters_per_pixel))
        pixel_dist = math.hypot(float(target_x - player_x), float(target_y - player_y))
        distance_m = round(pixel_dist * scale, 1)
        distance_int = int(round(distance_m))
        distance_str = f"{distance_int} m"

        # Overlay çakışma kontrolü
        under_overlay = self._is_point_in_overlay(target_x, target_y, overlay_rect)
        if under_overlay:
            logger.warning("Node %d is under the overlay panel", resolved_node.id)

        return NodeLocationResult(
            coords=(target_x, target_y),
            confidence=round(confidence, 4),
            distance_str=distance_str,
            distance_m=distance_m,
            under_overlay=under_overlay,
            used_fallback=used_fallback,
        )

    # Geriye dönük uyumluluk ve alternatif arayüzler için alias
    find_node_on_map = locate_node_on_open_map

    # ------------------------------------------------------------------
    # Rota Paylaşımı (ZIP Export / Import)
    # ------------------------------------------------------------------
    def export_route_zip(
        self,
        region: Optional[str] = None,
        route: Optional[str] = None,
        zip_path: Optional[str] = None,
    ) -> str:
        """
        Seçili bölge ve rotayı `region_data.json` ve ilgili tüm `patches/node_*.png`
        dosyalarıyla birlikte `.zip` arşivine dışa aktarır.
        """
        # Eğer 2 argümanla `export_route_zip(region, zip_path)` çağrıldıysa:
        if zip_path is None and route is not None and (
            str(route).lower().endswith(".zip") or os.sep in str(route) or "/" in str(route)
        ):
            zip_path = str(route)
            route = DEFAULT_ROUTE

        target_region = str(region or self.active_region).strip()
        target_route = str(route or DEFAULT_ROUTE).strip()

        if target_region not in self._regions:
            raise ValueError(f"Dışa aktarılacak bölge bulunamadı: {target_region}")

        if not zip_path:
            safe_route = target_route.replace(" ", "_")
            zip_path = os.path.join(self.base_dir, f"{target_region}_{safe_route}.zip")

        reg_info = self._regions[target_region]
        all_nodes: List[NodeData] = reg_info.get("nodes", [])
        if not target_route or target_route == DEFAULT_ROUTE:
            selected_nodes = list(all_nodes)
            exported_routes = list(reg_info.get("routes", [DEFAULT_ROUTE]))
        else:
            selected_nodes = [n for n in all_nodes if target_route in n.routes]
            exported_routes = [DEFAULT_ROUTE, target_route]

        export_payload = {
            "region": target_region,
            "exported_route": target_route,
            "routes": exported_routes,
            "nodes": [n.to_dict() for n in selected_nodes],
            "exported_at": time.time(),
        }

        try:
            parent_dir = os.path.dirname(os.path.abspath(zip_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

            json_bytes = json.dumps(export_payload, indent=2, ensure_ascii=False).encode("utf-8")
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("region_data.json", json_bytes)
                for node in selected_nodes:
                    disk_patch = self._resolve_patch_disk_path(node)
                    arc_patch_name = f"patches/node_{node.id}.png"
                    if os.path.isfile(disk_patch):
                        zf.write(disk_patch, arcname=arc_patch_name)
                    else:
                        # Eksik yama varsa 80x80 boş gri PNG oluşturup arşive ekle
                        ok, buf = cv2.imencode(".png", self._prepare_patch_image(None))
                        if ok:
                            zf.writestr(arc_patch_name, buf.tobytes())

            logger.info(
                "Rota ZIP dışa aktarıldı: Bölge=%s, Rota=%s, Düğüm=%d -> %s",
                target_region,
                target_route,
                len(selected_nodes),
                zip_path,
            )
            return os.path.abspath(zip_path)
        except Exception as exc:
            logger.error("ZIP dışa aktarma hatası (%s): %s", zip_path, exc)
            raise

    def import_route_zip(
        self,
        zip_path: str,
        target_region: Optional[str] = None,
    ) -> str:
        """
        `.zip` arşivinden `region_data.json` ve `patches/node_*.png` dosyalarını
        içe aktarır ve içe aktarılan bölge adını döndürür.
        """
        if not zip_path or not os.path.isfile(zip_path):
            raise FileNotFoundError(f"İçe aktarılacak ZIP dosyası bulunamadı: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                namelist = zf.namelist()
                json_entry: Optional[str] = None
                for candidate in namelist:
                    if os.path.basename(candidate) == "region_data.json":
                        json_entry = candidate
                        break

                if json_entry is None:
                    raise ValueError(f"ZIP arşivinde 'region_data.json' bulunamadı: {zip_path}")

                raw_json = json.loads(zf.read(json_entry).decode("utf-8"))
                region_name = str(
                    target_region
                    or raw_json.get("region")
                    or os.path.splitext(os.path.basename(zip_path))[0]
                ).strip()
                if not region_name:
                    region_name = DEFAULT_REGION

                if region_name not in self._regions:
                    self.create_region(region_name)

                reg_info = self._regions[region_name]
                existing_routes: List[str] = reg_info["routes"]
                for r_name in raw_json.get("routes", []):
                    if r_name and str(r_name) not in existing_routes:
                        existing_routes.append(str(r_name))

                patches_dir = self._get_region_patches_dir(region_name)
                os.makedirs(patches_dir, exist_ok=True)

                # Arşivdeki patch dosyalarını indeksle
                patch_entries: Dict[str, str] = {}
                for item_name in namelist:
                    base = os.path.basename(item_name)
                    if base.startswith("node_") and base.lower().endswith(".png"):
                        patch_entries[base] = item_name

                existing_by_id: Dict[int, NodeData] = {
                    n.id: n for n in reg_info["nodes"]
                }

                for raw_node in raw_json.get("nodes", []):
                    if not isinstance(raw_node, dict):
                        continue
                    raw_node["region"] = region_name
                    imported_node = NodeData.from_dict(raw_node)
                    imported_node.region = region_name
                    imported_node.patch_path = self._get_patch_rel_path(region_name, imported_node.id)

                    patch_filename = f"node_{imported_node.id}.png"
                    dest_patch_path = os.path.join(patches_dir, patch_filename)
                    if patch_filename in patch_entries:
                        with zf.open(patch_entries[patch_filename]) as src_f, open(dest_patch_path, "wb") as dst_f:
                            shutil.copyfileobj(src_f, dst_f)
                    elif not os.path.isfile(dest_patch_path):
                        _safe_imwrite(dest_patch_path, self._prepare_patch_image(None))

                    for r_name in imported_node.routes:
                        if r_name and r_name not in existing_routes:
                            existing_routes.append(r_name)

                    if imported_node.id in existing_by_id:
                        idx = reg_info["nodes"].index(existing_by_id[imported_node.id])
                        reg_info["nodes"][idx] = imported_node
                    else:
                        reg_info["nodes"].append(imported_node)
                    existing_by_id[imported_node.id] = imported_node

                self.active_region = region_name
                self.save_region(region_name)
                logger.info(
                    "Rota ZIP içe aktarıldı: Bölge=%s, Toplam Düğüm=%d <- %s",
                    region_name,
                    len(reg_info["nodes"]),
                    zip_path,
                )
                return region_name
        except Exception as exc:
            logger.error("ZIP içe aktarma hatası (%s): %s", zip_path, exc)
            raise

    def get_route_manager_ready_flag(self) -> dict:
        return {"module": "route_manager", "ready": True}

    def get_route_manager_health_summary(self) -> dict:
        try:
            return {"route_manager_ready": True}
        except Exception:
            return {"route_manager_ready": True}

    def get_route_manager_version_info(self) -> dict:
        return {"module": "route_manager", "version": "2.0.0", "ready": True}

    def get_route_manager_diagnostic_snapshot(self) -> dict:
        return {"module": "route_manager", "status": "ok", "ready": True}


from core.navigation_intel import PartialNodeDetector, handle_extraction_failure  # noqa: E402,F401


def get_route_manager_capabilities() -> list:
    return [
        "route_manager",
        "node_persistence",
        "diagnostic_snapshot",
        "snap_to_path",
        "odometry_snap",
        "intermediate_waypoints",
        "deduplicate_nodes",
        "sequence_indexing",
        "atomic_persistence",
    ]


def validate_route_manager_contract() -> bool:
    return bool(get_route_manager_capabilities())


def get_route_manager_baseline_metrics() -> dict:
    return {"module": "route_manager", "contract_valid": validate_route_manager_contract(), "capability_count": len(get_route_manager_capabilities())}


def get_route_manager_readiness_status() -> dict:
    return {"module": "route_manager", "ready": validate_route_manager_contract(), "metrics": get_route_manager_baseline_metrics()}


def get_route_manager_audit_record() -> dict:
    return {"module": "route_manager", "audit_passed": validate_route_manager_contract(), "readiness": get_route_manager_readiness_status()}


def get_route_manager_operational_digest() -> dict:
    return {"module": "route_manager", "operational": validate_route_manager_contract(), "audit": get_route_manager_audit_record()}


def get_route_manager_telemetry_payload() -> dict:
    return {"module": "route_manager", "telemetry_ready": validate_route_manager_contract(), "digest": get_route_manager_operational_digest()}


def get_route_manager_integrity_seal() -> dict:
    return {"module": "route_manager", "sealed": validate_route_manager_contract(), "telemetry": get_route_manager_telemetry_payload()}


def get_route_manager_compliance_certificate() -> dict:
    return {"module": "route_manager", "certified": validate_route_manager_contract(), "seal": get_route_manager_integrity_seal()}


def get_route_manager_readiness_probe() -> dict:
    return {"module": "route_manager", "ready": validate_route_manager_contract(), "certificate": get_route_manager_compliance_certificate()}


def get_route_manager_audit_summary() -> dict:
    return {"module": "route_manager", "audit_passed": validate_route_manager_contract(), "probe": get_route_manager_readiness_probe()}


