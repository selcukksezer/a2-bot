# -*- coding: utf-8 -*-
"""Aion 2 & Aion Complete Gatherable Materials Encyclopedia & 3D Visual Classifier.

Provides the complete structured catalog of gatherable materials across all 6 official
Aion / Aion 2 Essence Extraction & Vitality Extraction categories:
1. Herbs (Bitkiler / Otlar - Alchemy)
2. Ores & Metals (Madenler / Cevherler - Blacksmithing / Armorsmithing)
3. Gemstones (Değerli Taşlar / Gemler - Handicrafting)
4. Logs & Trees (Ağaçlar / Keresteler - Woodcutting)
5. Cooking Ingredients, Fruits, Vegetables, Fibers & Aquatic (Meyve / Sebze / Deniz Ürünleri)
6. Odyle & Aether (Odyle / Enerji Küresi - Flight / Ground Extraction)

Also provides HSV-based 3D mesh visual classification (`classify_3d_material_patch`),
regional material lookup (`get_region_expected_materials`), and strict non-material
rejection (`is_rejected_non_material`) to guarantee feathers, monsters, NPCs, and quests
are never treated as gatherable materials.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from version import __version__

logger = logging.getLogger(__name__)

DEFAULT_MAX_HARVESTS: int = 3

HSVBound = Tuple[int, int, int]
HSVRange = Tuple[HSVBound, HSVBound]

MATERIAL_CATEGORIES: Dict[str, str] = {
    "Herbs": "Bitkiler / Otlar",
    "Ores & Metals": "Madenler / Cevherler",
    "Gemstones": "Değerli Taşlar / Gemler",
    "Logs & Trees": "Ağaçlar / Keresteler",
    "Cooking & Fibers": "Meyve / Sebze / Deniz Ürünleri",
    "Odyle & Aether": "Odyle / Enerji Küresi",
}

MATERIAL_GRADES: Tuple[str, ...] = (
    "Standard",
    "Fine",
    "Pure",
    "Radiant",
    "Fresh",
)

# Single source of truth for Aria's HSV range (F-1089): the catalog, the
# color-ratio masks in compute_hsv_color_distribution, and the taxonomy
# hierarchy previously carried three different values for the same material.
ARIA_HSV_RANGE: HSVRange = ((125, 65, 75), (165, 255, 255))

# --- HSV aralıkları: tek tanım (P3-3; Aria F-1089 deseninin genelleştirilmesi) ---
# Eskiden her birincil materyalin HSV aralığı üç yerde ayrı literal olarak
# duruyordu: katalog (`_build_materials_catalog` yerelleri), tespit maskeleri
# (`_compute_hsv_color_distribution_from_hsv`) ve `AION2_TAXONOMY_HIERARCHY`.
# Kopyalar zamanla kaydı (örn. Targena katalogda (22,90,120) iken maskede
# (22,65,65); Yggdrasil kabuğu katalogda V=145 iken maskede V=140).
# Katalog ailesi tek doğruluk kaynağıdır; hiyerarşi katalogdan türer.
# Maske ailesi tespit için daha gevşek tutulan AYRI değerlerdir — değerleri
# değiştirilmedi, yalnızca adlandırıldı (testler oran eşiklerine dayanır:
# aria_purple > 0.20, targena_gold_green > 0.15, bark_trunk > 0.50).
# Katalog-maske farkı açık kalibrasyon sorusudur (P1-12 benzeri); burada
# bir taraf seçilip değer değiştirilmedi.
# v19 P3-3 yeniden doğrulaması — KARAR: kökeni KAYMA (drift), ama değerler
# donduruldu, birleştirilmedi. Kanıt: fark tek yönlü değil — Targena maskesi
# katalogdan gevşek (S/V alt sınır 65/65 ↔ 90/120) ama Orichalcum bronz
# maskesi katalogdan sıkı (üst sınır (26,230,225) ↔ (26,255,245)) ve kabuk
# maskesi üst V'de sıkı (140 ↔ 145); ayrıca `_compute_hsv_..._from_hsv`
# içinde bazı oranlar maske, bazıları katalog aralığıyla sayılır
# (calendula/sapphire/azpha/diamond/odyle katalogdan). Tasarlanmış "gevşek
# tespit maskesi" olsaydı fark tek yönlü olurdu. Oran eşikleri +
# sınıflandırıcı skorları mevcut değerlere dayandığı için tekleştirme ancak
# canlı kalibrasyon verisiyle yapılır; sessizce eşitlemeyin. Not:
# `core/material_scanner.py:3339` civarında Aria için üçüncü bir literal
# ((125,65,70)-(164,255,255)) durur — ayrı tespit yolu, bu maddenin dışı.
# Katalog ailesi (12 birincil materyal):
TARGENA_HSV_RANGE: HSVRange = ((22, 90, 120), (45, 255, 255))
AZPHA_HSV_RANGE: HSVRange = ((105, 70, 90), (135, 255, 255))
CALENDULA_HSV_RANGE: HSVRange = ((10, 120, 140), (24, 255, 255))
SAPPHIRE_HSV_RANGE: HSVRange = ((95, 120, 140), (125, 255, 255))
RUBY_HSV_RANGES: Tuple[HSVRange, ...] = (
    ((0, 140, 140), (10, 255, 255)),
    ((168, 140, 140), (180, 255, 255)),
)
DIAMOND_HSV_RANGE: HSVRange = ((85, 10, 200), (115, 65, 255))
ODYLE_HSV_RANGE: HSVRange = ((58, 105, 145), (94, 255, 255))
ORICHALCUM_HSV_RANGES: Tuple[HSVRange, ...] = (
    ((6, 65, 55), (26, 255, 245)),
    ((72, 45, 70), (98, 210, 225)),
)
YGGDRASIL_LOG_HSV_RANGE: HSVRange = ((8, 45, 35), (22, 165, 145))
LOG_CANOPY_HSV_RANGE: HSVRange = ((35, 55, 45), (78, 255, 215))
ASVATA_LOG_HSV_RANGE: HSVRange = ((10, 50, 30), (24, 175, 135))
MELA_HSV_RANGES: Tuple[HSVRange, ...] = (
    ((0, 100, 85), (12, 255, 255)),
    ((168, 100, 85), (180, 255, 255)),
)
MELA_LEAF_HSV_RANGE: HSVRange = ((35, 50, 40), (75, 255, 235))
# Maske ailesi (tespit maskeleri; katalogdan farklı değerler aynen korunur):
TARGENA_MASK_HSV_RANGE: HSVRange = ((22, 65, 65), (45, 255, 255))
ORICHALCUM_BRONZE_MASK_HSV_RANGE: HSVRange = ((6, 65, 55), (26, 230, 225))
ORICHALCUM_TEAL_MASK_HSV_RANGE: HSVRange = ((72, 45, 65), (98, 210, 225))
BARK_TRUNK_MASK_HSV_RANGE: HSVRange = ((8, 45, 35), (22, 165, 140))
GREEN_LEAF_MASK_HSV_RANGE: HSVRange = ((35, 50, 40), (78, 255, 240))
GREEN_BROAD_MASK_HSV_RANGE: HSVRange = ((26, 40, 35), (82, 255, 245))
DARK_ROCK_MASK_HSV_RANGE: HSVRange = ((0, 0, 20), (180, 65, 115))

# Canonical 12 primary materials (F-1085/F-1088): the classifier scores exactly
# these — one entry per primary material in AION2_TAXONOMY_HIERARCHY — instead
# of a hardcoded 11-entry dict that merged Yggdrasil/Asvata into a single slot.
# Specific catalog essences beyond these 12 are reached through the regional
# refinement in _more_specific_catalog_material, never by raw HSV alone.
CANONICAL_PRIMARY_MATERIALS: Tuple[str, ...] = (
    "Aria",
    "Mela",
    "Ruby Gemstone",
    "Sapphire Gemstone",
    "Diamond Gemstone",
    "Odyle",
    "Targena",
    "Calendula",
    "Azpha",
    "Yggdrasil Log",
    "Asvata Log",
    "Orichalcum Ore",
)

OFFICIAL_REGIONS: Tuple[str, ...] = (
    "Pilgrim's Ridge",
    "Western Altamia Highland",
    "Verteron",
    "Altgard",
    "Eltnen",
    "Morheim",
    "Heiron",
    "Beluslan",
    "Poeta",
    "Ishalgen",
    "Inggison",
    "Gelkmaros",
    "Dredgion Crash Site",
    "Ellun River Midstream",
    "Scarlet Forest Observation Site",
)

# Strictly rejected non-material tokens (feathers on minimap, monsters, NPCs, quests)
REJECTED_NON_MATERIAL_KEYWORDS: Tuple[str, ...] = (
    "feather",
    "white feather",
    "minimap feather",
    "odyle feather",
    "wing icon",
    "blue wing",
    "tüy",
    "beyaz tüy",
    "kanat",
    "monster",
    "monster dot",
    "canavar",
    "mob",
    "hostile",
    "agro",
    "yellow-maned zaif",
    "zaif",
    "highland odyle sparkie",
    "sparkie",
    "functional npc",
    "npc",
    "vendor",
    "merchant",
    "teleporter",
    "flight transporter",
    "kisk",
    "obelisk",
    "quest",
    "quest giver",
    "quest marker",
    "görev",
    "player",
    "party member",
    "ally",
)


@dataclass
class MaterialSpec:
    """Complete specification for a gatherable resource node in Aion 2 / Aion."""

    name: str
    name_tr: str
    category: str
    category_tr: str
    profession: str = "Essence Extraction"
    grade: str = "Standard"
    base_material: str = ""
    grades_available: Tuple[str, ...] = ("Standard",)
    max_harvests: int = DEFAULT_MAX_HARVESTS
    regions: Tuple[str, ...] = ()
    hsv_ranges: Tuple[HSVRange, ...] = ()
    secondary_hsv_ranges: Tuple[HSVRange, ...] = ()
    shows_on_minimap: bool = True
    visible_in_3d_crest: bool = True
    visible_in_floating_box: bool = True
    in_air: bool = False
    gather_keys: Tuple[str, ...] = ("f", "e")
    visual_description: str = ""

    def __post_init__(self) -> None:
        self.name = str(self.name).strip()
        self.name_tr = str(self.name_tr or self.name).strip()
        self.category = str(self.category).strip()
        self.category_tr = str(
            self.category_tr or MATERIAL_CATEGORIES.get(self.category, self.category)
        ).strip()
        self.grade = str(self.grade or "Standard").strip()
        if not self.base_material:
            self.base_material = self.name
        self.max_harvests = max(1, int(self.max_harvests or DEFAULT_MAX_HARVESTS))
        self.regions = tuple(str(r).strip() for r in self.regions if str(r).strip())
        self.grades_available = tuple(
            str(g).strip() for g in self.grades_available if str(g).strip()
        ) or (self.grade,)

    def __hash__(self) -> int:
        return hash(self.name.lower())

    def __eq__(self, other: object) -> bool:
        # Hash/eq contract (F-1098): equality is defined only between two
        # MaterialSpec objects, by case-insensitive canonical name — the same
        # value __hash__ is built from. Comparing a spec to a bare string
        # ("Aria" == spec) used to return True while hash(spec) !=
        # hash("Aria"), breaking set/dict invariants. Name lookup by string
        # belongs to MaterialCatalog.get / get_material_spec, not __eq__.
        if isinstance(other, MaterialSpec):
            return self.name.lower() == other.name.lower()
        return NotImplemented

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the material specification to a dictionary."""
        return asdict(self)


class MaterialCatalog(list):
    """Hybrid list/dict catalog of MaterialSpec objects supporting both sequence and mapping access."""

    def __init__(self, specs: Sequence[MaterialSpec]) -> None:
        super().__init__(specs)
        self._by_key: Dict[str, MaterialSpec] = {}
        self._ordered_keys: List[str] = []
        for spec in specs:
            key_lower = spec.name.lower()
            if key_lower not in self._by_key:
                self._ordered_keys.append(spec.name)
            self._by_key[key_lower] = spec
            self._by_key[spec.name_tr.lower()] = spec

    def __getitem__(self, key: Union[int, slice, str]) -> Any:  # type: ignore[override]
        if isinstance(key, (int, slice)):
            return super().__getitem__(key)
        if isinstance(key, str):
            norm = key.strip().lower()
            if norm in self._by_key:
                return self._by_key[norm]
            raise KeyError(f"Material not found in AION2_MATERIALS_CATALOG: {key!r}")
        raise TypeError(f"Invalid key type for MaterialCatalog: {type(key)!r}")

    def __contains__(self, item: object) -> bool:
        if isinstance(item, str):
            return item.strip().lower() in self._by_key
        if isinstance(item, MaterialSpec):
            return item.name.lower() in self._by_key
        return super().__contains__(item)

    def get(self, key: str, default: Optional[MaterialSpec] = None) -> Optional[MaterialSpec]:
        """Return MaterialSpec by English or Turkish name (case-insensitive)."""
        if not isinstance(key, str):
            return default
        return self._by_key.get(key.strip().lower(), default)

    def keys(self) -> List[str]:
        """Return canonical material names in catalog order."""
        return list(self._ordered_keys)

    def values(self) -> List[MaterialSpec]:
        """Return all MaterialSpec entries in catalog order."""
        return list(self)

    def items(self) -> List[Tuple[str, MaterialSpec]]:
        """Return (name, MaterialSpec) pairs in catalog order."""
        return [(spec.name, spec) for spec in self]

    def to_dict(self) -> Dict[str, MaterialSpec]:
        """Return a standard dict mapping canonical material name to MaterialSpec."""
        return {spec.name: spec for spec in self}


def _build_materials_catalog() -> MaterialCatalog:
    """Build the complete Aion 2 & Aion gatherable materials encyclopedia."""
    specs: List[MaterialSpec] = []

    def add_spec(spec: MaterialSpec) -> None:
        specs.append(spec)

    # =========================================================================
    # CATEGORY 1: HERBS (Bitkiler / Otlar - Alchemy)
    # =========================================================================
    aria_hsv: Tuple[HSVRange, ...] = (ARIA_HSV_RANGE,)
    aria_regions = (
        "Western Altamia Highland",
        "Verteron",
        "Poeta",
        "Pilgrim's Ridge",
        "Eltnen",
    )
    for grade, name, name_tr in (
        ("Standard", "Aria", "Aria (Mor Kristal Bitki)"),
        ("Fresh", "Fresh Aria", "Taze Aria"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade=grade,
                base_material="Aria",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=aria_regions,
                hsv_ranges=aria_hsv,
                shows_on_minimap=False,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Purple/violet crystal-petaled herb below white winged gather crest.",
            )
        )

    targena_hsv: Tuple[HSVRange, ...] = (TARGENA_HSV_RANGE,)
    targena_regions = (
        "Verteron",
        "Pilgrim's Ridge",
        "Western Altamia Highland",
        "Eltnen",
        "Heiron",
    )
    for grade, name, name_tr in (
        ("Standard", "Targena", "Targena Otu"),
        ("Fresh", "Fresh Targena", "Taze Targena Otu"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade=grade,
                base_material="Targena",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=targena_regions,
                hsv_ranges=targena_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Golden-yellow and bright green leafy herb with [E]/[F] interaction icon.",
            )
        )

    azpha_hsv: Tuple[HSVRange, ...] = (AZPHA_HSV_RANGE,)
    azpha_regions = ("Altgard", "Ishalgen", "Morheim", "Beluslan")
    for grade, name, name_tr in (
        ("Standard", "Azpha", "Azpha Otu"),
        ("Fresh", "Fresh Azpha", "Taze Azpha Otu"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade=grade,
                base_material="Azpha",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=azpha_regions,
                hsv_ranges=azpha_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Blue-violet Asmodian highland herb.",
            )
        )

    calendula_hsv: Tuple[HSVRange, ...] = (CALENDULA_HSV_RANGE,)
    calendula_regions = ("Altgard", "Verteron", "Pilgrim's Ridge", "Eltnen", "Morheim")
    for grade, name, name_tr in (
        ("Standard", "Calendula", "Calendula (Aynısafa Otu)"),
        ("Fresh", "Fresh Calendula", "Taze Calendula"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade=grade,
                base_material="Calendula",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=calendula_regions,
                hsv_ranges=calendula_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Vibrant orange-gold marigold blossom herb.",
            )
        )

    extended_herbs: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[HSVRange, ...]], ...] = (
        ("Lura", "Lura Otu", ("Pilgrim's Ridge", "Western Altamia Highland", "Verteron"), (((30, 75, 90), (65, 255, 245)),)),
        ("Gravi's Root", "Gravi Kökü", ("Pilgrim's Ridge", "Western Altamia Highland", "Altgard"), (((16, 65, 70), (34, 220, 210)),)),
        ("Purifying Forest Large Leaf", "Arındırıcı Orman Büyük Yaprağı", ("Pilgrim's Ridge", "Scarlet Forest Observation Site", "Verteron"), (((35, 80, 95), (72, 255, 245)),)),
        ("Withered Branch", "Kurumuş Dal", ("Pilgrim's Ridge", "Western Altamia Highland", "Dredgion Crash Site"), (((10, 45, 55), (24, 170, 175)),)),
        ("Miana", "Miana Otu", ("Poeta", "Verteron", "Ishalgen"), (((28, 70, 100), (55, 240, 240)),)),
        ("Pella", "Pella Otu", ("Verteron", "Altgard", "Eltnen"), (((135, 65, 95), (168, 245, 245)),)),
        ("Rina", "Rina Otu", ("Verteron", "Eltnen", "Morheim"), (((20, 85, 115), (42, 255, 255)),)),
        ("Mandragora", "Adamotu (Mandragora)", ("Poeta", "Ishalgen", "Verteron", "Altgard"), (((32, 65, 80), (68, 235, 220)),)),
        ("Atri", "Atri Otu", ("Eltnen", "Morheim"), (((115, 70, 95), (145, 250, 250)),)),
        ("Kalgelly", "Kalgelly Otu", ("Eltnen", "Heiron", "Morheim"), (((24, 80, 110), (48, 255, 250)),)),
        ("Lolo", "Lolo Otu", ("Eltnen", "Morheim", "Beluslan"), (((14, 95, 120), (30, 255, 255)),)),
        ("Nezekan's Tear", "Nezekan'ın Gözyaşı", ("Eltnen", "Heiron", "Pilgrim's Ridge"), (((95, 90, 130), (125, 255, 255)),)),
        ("Marle", "Marle Otu", ("Heiron", "Beluslan"), (((35, 75, 100), (65, 250, 245)),)),
        ("Amilis", "Amilis Otu", ("Heiron", "Beluslan"), (((125, 65, 100), (158, 250, 250)),)),
        ("Valerian", "Kediotu (Valerian)", ("Heiron", "Beluslan", "Inggison"), (((130, 60, 110), (165, 235, 255)),)),
        ("Caprauna", "Caprauna Otu", ("Heiron", "Beluslan", "Gelkmaros"), (((22, 85, 110), (44, 250, 250)),)),
        ("Afari", "Afari Otu", ("Inggison", "Gelkmaros"), (((12, 100, 125), (28, 255, 255)),)),
        ("Lumesia", "Lumesia Otu", ("Inggison", "Gelkmaros"), (((88, 70, 140), (118, 245, 255)),)),
        ("Wiki", "Wiki Otu", ("Inggison", "Gelkmaros", "Eltnen"), (((34, 80, 105), (66, 255, 245)),)),
        ("Theonia", "Theonia Otu", ("Inggison", "Gelkmaros"), (((140, 75, 110), (172, 255, 255)),)),
        ("Senor", "Senor Otu", ("Inggison", "Gelkmaros"), (((20, 90, 120), (40, 255, 255)),)),
        ("Jai", "Jai Otu", ("Inggison", "Gelkmaros"), (((100, 75, 110), (130, 250, 250)),)),
        ("Ratal", "Ratal Otu", ("Morheim", "Beluslan", "Gelkmaros"), (((8, 95, 110), (22, 255, 250)),)),
        ("Blicora", "Blicora Otu", ("Eltnen", "Heiron", "Inggison"), (((118, 70, 100), (150, 255, 250)),)),
        ("Xilix", "Xilix Otu", ("Heiron", "Beluslan", "Inggison"), (((38, 85, 110), (70, 255, 250)),)),
        ("Krotan", "Krotan Otu", ("Inggison", "Gelkmaros"), (((0, 110, 120), (14, 255, 255)),)),
        ("Pyre", "Pyre Ateş Otu", ("Morheim", "Beluslan", "Gelkmaros"), (((4, 130, 135), (18, 255, 255)),)),
        ("Diaphanous Herb", "Yarı Saydam Ot", ("Pilgrim's Ridge", "Western Altamia Highland", "Inggison"), (((85, 30, 165), (120, 140, 255)),)),
        ("Penac", "Penac Otu", ("Verteron", "Altgard", "Eltnen"), (((30, 75, 100), (58, 245, 245)),)),
        ("Ormea", "Ormea Otu", ("Eltnen", "Morheim", "Heiron"), (((18, 90, 115), (36, 255, 250)),)),
        ("Hasia", "Hasia Otu", ("Heiron", "Beluslan"), (((110, 75, 105), (140, 250, 250)),)),
        ("Oriel", "Oriel Otu", ("Pilgrim's Ridge", "Western Altamia Highland", "Inggison"), (((24, 85, 125), (46, 255, 255)),)),
        ("Hanate", "Hanate Otu", ("Inggison", "Gelkmaros"), (((128, 70, 105), (160, 250, 255)),)),
        ("Atreian Herb", "Atreia Otu", ("Pilgrim's Ridge", "Western Altamia Highland", "Verteron", "Altgard"), (((32, 80, 105), (68, 255, 250)),)),
        ("Lunime", "Lunime Kristal Otu", ("Pilgrim's Ridge", "Western Altamia Highland", "Inggison", "Gelkmaros"), (((92, 80, 140), (128, 255, 255)),)),
        ("Cenna", "Cenna Otu", ("Poeta", "Verteron", "Altgard", "Ishalgen"), (((34, 75, 100), (64, 245, 245)),)),
    )
    for herb_name, herb_tr, herb_regions, herb_hsv in extended_herbs:
        add_spec(
            MaterialSpec(
                name=herb_name,
                name_tr=herb_tr,
                category="Herbs",
                category_tr="Bitkiler / Otlar",
                profession="Alchemy",
                grade="Standard",
                base_material=herb_name,
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=herb_regions,
                hsv_ranges=herb_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description=f"{herb_name} alchemy herb gatherable node.",
            )
        )

    # =========================================================================
    # CATEGORY 2: ORES & METALS (Madenler / Cevherler - Blacksmithing/Armorsmithing)
    # =========================================================================
    orichalcum_hsv: Tuple[HSVRange, ...] = ORICHALCUM_HSV_RANGES
    orichalcum_regions = (
        "Verteron",
        "Pilgrim's Ridge",
        "Altgard",
        "Dredgion Crash Site",
        "Western Altamia Highland",
        "Eltnen",
        "Morheim",
    )
    for grade, name, name_tr in (
        ("Standard", "Orichalcum Ore", "Orichalcum Cevheri"),
        ("Fine", "Fine Orichalcum Ore", "Kaliteli Orichalcum Cevheri"),
        ("Pure", "Pure Orichalcum Ore", "Saf Orichalcum Cevheri"),
        ("Radiant", "Radiant Orichalcum Ore", "Parlak Orichalcum Cevheri"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Ores & Metals",
                category_tr="Madenler / Cevherler",
                profession="Blacksmithing / Armorsmithing",
                grade=grade,
                base_material="Orichalcum Ore",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=orichalcum_regions,
                hsv_ranges=orichalcum_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Metallic bronze, amber-gold, and oxidized teal rock vein.",
            )
        )

    extended_ores: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[HSVRange, ...]], ...] = (
        ("Iron Ore", "Demir Cevheri", ("Poeta", "Verteron", "Altgard", "Ishalgen", "Pilgrim's Ridge"), (((8, 40, 50), (22, 165, 190)),)),
        ("Copper Ore", "Bakır Cevheri", ("Poeta", "Verteron", "Altgard", "Ishalgen"), (((8, 95, 75), (20, 245, 225)),)),
        ("Silver Ore", "Gümüş Cevheri", ("Verteron", "Altgard", "Eltnen", "Morheim"), (((0, 0, 145), (180, 42, 240)),)),
        ("Gold Ore", "Altın Cevheri", ("Eltnen", "Morheim", "Heiron", "Beluslan", "Pilgrim's Ridge"), (((16, 110, 115), (28, 255, 255)),)),
        ("Platinum Ore", "Platin Cevheri", ("Heiron", "Beluslan", "Inggison", "Gelkmaros"), (((90, 10, 160), (125, 55, 250)),)),
        ("Titanium Ore", "Titanyum Cevheri", ("Eltnen", "Morheim", "Heiron", "Beluslan"), (((95, 20, 130), (120, 75, 235)),)),
        ("Adamantium", "Adamantiyum Cevheri", ("Heiron", "Beluslan", "Inggison", "Gelkmaros"), (((100, 45, 85), (128, 165, 210)),)),
        ("Starmetal", "Yıldız Metali (Starmetal)", ("Heiron", "Beluslan", "Inggison", "Gelkmaros"), (((105, 65, 135), (138, 210, 255)),)),
        ("Mithril Ore", "Mithril Cevheri", ("Inggison", "Gelkmaros", "Heiron", "Beluslan"), (((88, 35, 150), (115, 130, 255)),)),
        ("Thallium", "Talyum Cevheri", ("Eltnen", "Morheim", "Heiron"), (((32, 45, 80), (62, 160, 200)),)),
        ("Dark Iron", "Kara Demir (Dark Iron)", ("Morheim", "Beluslan", "Dredgion Crash Site"), (((0, 0, 30), (180, 50, 95)),)),
        ("Boon Ore", "Boon Cevheri", ("Pilgrim's Ridge", "Western Altamia Highland", "Verteron"), (((12, 70, 80), (26, 220, 220)),)),
        ("Hamerium", "Hamerium Cevheri", ("Verteron", "Altgard", "Eltnen", "Morheim"), (((10, 80, 70), (24, 230, 215)),)),
        ("Lyrion", "Lyrion Cevheri", ("Eltnen", "Morheim", "Heiron"), (((85, 50, 100), (112, 185, 230)),)),
        ("Blue Titanium", "Mavi Titanyum", ("Heiron", "Beluslan", "Pilgrim's Ridge"), (((98, 85, 115), (122, 235, 245)),)),
        ("Silicate", "Silikat Minerali", ("Pilgrim's Ridge", "Western Altamia Highland", "Eltnen"), (((80, 20, 140), (110, 90, 245)),)),
        ("Magma Ore", "Magma Cevheri", ("Morheim", "Beluslan", "Dredgion Crash Site"), (((0, 135, 110), (14, 255, 255)),)),
        ("Katalium", "Katalium Cevheri", ("Inggison", "Gelkmaros", "Pilgrim's Ridge"), (((128, 70, 100), (155, 235, 245)),)),
        ("Anarium", "Anarium Cevheri", ("Inggison", "Gelkmaros"), (((18, 85, 120), (34, 240, 250)),)),
        ("Ceramium", "Seramyum (Ceramium)", ("Inggison", "Gelkmaros"), (((95, 70, 125), (120, 225, 250)),)),
        ("Gravi", "Gravi Cevheri", ("Pilgrim's Ridge", "Western Altamia Highland", "Dredgion Crash Site"), (((14, 60, 75), (28, 200, 210)),)),
    )
    for ore_name, ore_tr, ore_regions, ore_hsv in extended_ores:
        add_spec(
            MaterialSpec(
                name=ore_name,
                name_tr=ore_tr,
                category="Ores & Metals",
                category_tr="Madenler / Cevherler",
                profession="Blacksmithing / Armorsmithing",
                grade="Standard",
                base_material=ore_name,
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=ore_regions,
                hsv_ranges=ore_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description=f"{ore_name} metallic rock vein.",
            )
        )

    # =========================================================================
    # CATEGORY 3: GEMSTONES (Değerli Taşlar / Gemler - Handicrafting)
    # =========================================================================
    sapphire_hsv: Tuple[HSVRange, ...] = (SAPPHIRE_HSV_RANGE,)
    sapphire_regions = (
        "Pilgrim's Ridge",
        "Ellun River Midstream",
        "Scarlet Forest Observation Site",
        "Western Altamia Highland",
        "Verteron",
        "Eltnen",
    )
    for grade, name, name_tr in (
        ("Standard", "Sapphire Gemstone", "Safir Değerli Taş"),
        ("Fine", "Fine Sapphire Gemstone", "Kaliteli Safir Değerli Taş"),
        ("Pure", "Pure Sapphire Gemstone", "Saf Safir Değerli Taş"),
        ("Radiant", "Radiant Sapphire Gemstone", "Parlak Safir Değerli Taş"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Gemstones",
                category_tr="Değerli Taşlar / Gemler",
                profession="Handicrafting",
                grade=grade,
                base_material="Sapphire Gemstone",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=sapphire_regions,
                hsv_ranges=sapphire_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Deep glowing blue/cyan crystal cluster.",
            )
        )

    ruby_hsv: Tuple[HSVRange, ...] = RUBY_HSV_RANGES
    ruby_regions = (
        "Pilgrim's Ridge",
        "Western Altamia Highland",
        "Verteron",
        "Altgard",
        "Eltnen",
        "Morheim",
    )
    for grade, name, name_tr in (
        ("Standard", "Ruby Gemstone", "Yakut Değerli Taş"),
        ("Fine", "Fine Ruby Gemstone", "Kaliteli Yakut Değerli Taş"),
        ("Pure", "Pure Ruby Gemstone", "Saf Yakut Değerli Taş"),
        ("Radiant", "Radiant Ruby Gemstone", "Parlak Yakut Değerli Taş"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Gemstones",
                category_tr="Değerli Taşlar / Gemler",
                profession="Handicrafting",
                grade=grade,
                base_material="Ruby Gemstone",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=ruby_regions,
                hsv_ranges=ruby_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Crimson/red glowing crystal spire without green foliage.",
            )
        )

    diamond_hsv: Tuple[HSVRange, ...] = (DIAMOND_HSV_RANGE,)
    diamond_regions = (
        "Pilgrim's Ridge",
        "Western Altamia Highland",
        "Eltnen",
        "Morheim",
        "Heiron",
        "Beluslan",
        "Inggison",
        "Gelkmaros",
    )
    for grade, name, name_tr in (
        ("Standard", "Diamond Gemstone", "Elmas Değerli Taş"),
        ("Fine", "Fine Diamond Gemstone", "Kaliteli Elmas Değerli Taş"),
        ("Pure", "Pure Diamond Gemstone", "Saf Elmas Değerli Taş"),
        ("Radiant", "Radiant Diamond Gemstone", "Parlak Elmas Değerli Taş"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Gemstones",
                category_tr="Değerli Taşlar / Gemler",
                profession="Handicrafting",
                grade=grade,
                base_material="Diamond Gemstone",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=diamond_regions,
                hsv_ranges=diamond_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Brilliant white/ice crystalline formation.",
            )
        )

    extended_gems: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[HSVRange, ...]], ...] = (
        ("Rose Quartz", "Pembe Kuvars (Rose Quartz)", ("Verteron", "Altgard", "Pilgrim's Ridge"), (((155, 55, 160), (176, 185, 255)),)),
        ("Crystal", "Kristal", ("Poeta", "Ishalgen", "Verteron", "Altgard"), (((88, 25, 175), (112, 110, 255)),)),
        ("Amethyst", "Ametist", ("Verteron", "Altgard", "Eltnen", "Morheim"), (((130, 95, 120), (158, 255, 255)),)),
        ("Aquamarine", "Akuamarin", ("Verteron", "Ellun River Midstream", "Eltnen", "Heiron"), (((82, 95, 145), (98, 255, 255)),)),
        ("Garnet", "Lal Taşı (Garnet)", ("Altgard", "Morheim", "Eltnen"), (((0, 150, 95), (8, 255, 215)),)),
        ("Leucoxene", "Lökoksen (Leucoxene)", ("Eltnen", "Morheim", "Heiron"), (((18, 45, 150), (35, 145, 245)),)),
        ("Zircon", "Zirkon", ("Eltnen", "Morheim", "Heiron", "Beluslan"), (((14, 105, 145), (28, 245, 255)),)),
        ("Topaz", "Topaz", ("Eltnen", "Morheim", "Heiron", "Beluslan", "Pilgrim's Ridge"), (((20, 130, 155), (32, 255, 255)),)),
        ("Emerald", "Zümrüt (Emerald)", ("Heiron", "Beluslan", "Inggison", "Gelkmaros", "Pilgrim's Ridge"), (((52, 125, 120), (78, 255, 255)),)),
        ("Opal", "Opal", ("Heiron", "Beluslan", "Inggison"), (((85, 35, 185), (145, 125, 255)),)),
        ("Jacinth", "Jasen (Jacinth)", ("Heiron", "Beluslan", "Inggison", "Gelkmaros"), (((8, 145, 150), (20, 255, 255)),)),
        ("Sardon", "Sardon", ("Inggison", "Gelkmaros"), (((4, 130, 115), (16, 250, 235)),)),
        ("Black Gemstone", "Siyah Değerli Taş", ("Dredgion Crash Site", "Beluslan", "Gelkmaros"), (((0, 0, 20), (180, 55, 75)),)),
        ("Hyacinth", "Sümbül Taşı (Hyacinth)", ("Inggison", "Gelkmaros", "Pilgrim's Ridge"), (((118, 105, 135), (142, 255, 255)),)),
    )
    for gem_name, gem_tr, gem_regions, gem_hsv in extended_gems:
        add_spec(
            MaterialSpec(
                name=gem_name,
                name_tr=gem_tr,
                category="Gemstones",
                category_tr="Değerli Taşlar / Gemler",
                profession="Handicrafting",
                grade="Standard",
                base_material=gem_name,
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=gem_regions,
                hsv_ranges=gem_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description=f"{gem_name} glowing crystal formation.",
            )
        )

    # =========================================================================
    # CATEGORY 4: LOGS & TREES (Ağaçlar / Keresteler - Woodcutting)
    # =========================================================================
    log_hsv: Tuple[HSVRange, ...] = (YGGDRASIL_LOG_HSV_RANGE,)
    log_canopy_hsv: Tuple[HSVRange, ...] = (LOG_CANOPY_HSV_RANGE,)

    yggdrasil_regions = (
        "Verteron",
        "Pilgrim's Ridge",
        "Western Altamia Highland",
        "Scarlet Forest Observation Site",
        "Poeta",
        "Eltnen",
        "Heiron",
    )
    for grade, name, name_tr in (
        ("Standard", "Yggdrasil Log", "Yggdrasil Kerestesi"),
        ("Fine", "Fine Yggdrasil Log", "Kaliteli Yggdrasil Kerestesi"),
        ("Pure", "Pure Yggdrasil Log", "Saf Yggdrasil Kerestesi"),
        ("Radiant", "Radiant Yggdrasil Log", "Parlak Yggdrasil Kerestesi"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Logs & Trees",
                category_tr="Ağaçlar / Keresteler",
                profession="Woodcutting",
                grade=grade,
                base_material="Yggdrasil Log",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=yggdrasil_regions,
                hsv_ranges=log_hsv,
                secondary_hsv_ranges=log_canopy_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Elyos sacred timber node with brown bark trunk and leafy canopy.",
            )
        )

    asvata_regions = ("Altgard", "Morheim", "Ishalgen", "Beluslan", "Gelkmaros")
    # F-1091: Asvata keeps its own bark range (darker, redder Asmodian
    # timber) instead of sharing Yggdrasil's byte-for-byte; the family
    # trunk mask in compute_hsv_color_distribution still covers both.
    asvata_hsv: Tuple[HSVRange, ...] = (ASVATA_LOG_HSV_RANGE,)
    for grade, name, name_tr in (
        ("Standard", "Asvata Log", "Asvata Kerestesi"),
        ("Fine", "Fine Asvata Log", "Kaliteli Asvata Kerestesi"),
        ("Pure", "Pure Asvata Log", "Saf Asvata Kerestesi"),
        ("Radiant", "Radiant Asvata Log", "Parlak Asvata Kerestesi"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Logs & Trees",
                category_tr="Ağaçlar / Keresteler",
                profession="Woodcutting",
                grade=grade,
                base_material="Asvata Log",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=asvata_regions,
                hsv_ranges=asvata_hsv,
                secondary_hsv_ranges=log_canopy_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Asmodian timber node with rugged brown bark trunk.",
            )
        )

    extended_logs: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
        ("Tecoma Log", "Tecoma Kerestesi", ("Poeta", "Verteron", "Pilgrim's Ridge")),
        ("Betua Log", "Betua Huş Kerestesi", ("Ishalgen", "Altgard", "Morheim")),
        ("Dryad Log", "Dryad Kerestesi", ("Verteron", "Eltnen", "Heiron")),
        ("Birch Log", "Huş Ağacı Kerestesi", ("Altgard", "Morheim", "Beluslan")),
        ("Pine Log", "Çam Kerestesi", ("Verteron", "Altgard", "Pilgrim's Ridge", "Western Altamia Highland")),
        ("Fir Log", "Göknar Kerestesi", ("Morheim", "Beluslan", "Altgard")),
        ("Bamboo Log", "Bambu Gövdesi", ("Heiron", "Inggison", "Ellun River Midstream")),
        ("Oak Log", "Meşe Kerestesi", ("Verteron", "Eltnen", "Heiron", "Pilgrim's Ridge")),
        ("Teak Log", "Tik Ağacı Kerestesi", ("Eltnen", "Heiron", "Inggison")),
        ("Wrud Log", "Wrud Kerestesi", ("Morheim", "Beluslan", "Gelkmaros")),
        ("Koa Log", "Koa Kerestesi", ("Heiron", "Inggison", "Scarlet Forest Observation Site")),
        ("Ulmus Log", "Karaağaç (Ulmus) Kerestesi", ("Eltnen", "Heiron", "Beluslan")),
        ("Maple Log", "Akçaağaç Kerestesi", ("Scarlet Forest Observation Site", "Pilgrim's Ridge", "Heiron", "Beluslan")),
        ("Ash Log", "Dişbudak Kerestesi", ("Verteron", "Altgard", "Eltnen", "Morheim")),
        ("Palm Log", "Palmiye Kerestesi", ("Eltnen", "Ellun River Midstream", "Inggison")),
        ("Cedar Log", "Sedir Kerestesi", ("Pilgrim's Ridge", "Western Altamia Highland", "Heiron", "Beluslan")),
        ("Ebony Log", "Abanoz Kerestesi", ("Heiron", "Beluslan", "Inggison", "Gelkmaros")),
        ("Sandalwood Log", "Sandal Ağacı Kerestesi", ("Inggison", "Gelkmaros")),
        ("Ancient Log", "Kadim Ağaç Kerestesi", ("Pilgrim's Ridge", "Western Altamia Highland", "Inggison", "Gelkmaros")),
        ("Heartwood", "Öz Odun (Heartwood)", ("Scarlet Forest Observation Site", "Heiron", "Beluslan", "Inggison")),
        ("Ironwood", "Demir Ağacı (Ironwood)", ("Beluslan", "Gelkmaros", "Dredgion Crash Site")),
    )
    # F-1091: all 29 extended logs previously shared one identical HSV range,
    # so no two log species could ever be told apart. Each species now gets a
    # distinct (still approximate, bark-family) range derived deterministically
    # from its index; the two canonical primaries (Yggdrasil/Asvata) keep the
    # shared family range above.
    for log_idx, (log_name, log_tr, log_regions) in enumerate(extended_logs):
        _log_hue_lo = 5 + (log_idx % 7)
        _log_hue_hi = 19 + (log_idx % 7)
        _log_hsv: Tuple[HSVRange, ...] = (
            (
                (_log_hue_lo, 45, 35),
                (_log_hue_hi, 150 + (log_idx % 5) * 5, 135 + (log_idx % 4) * 5),
            ),
        )
        add_spec(
            MaterialSpec(
                name=log_name,
                name_tr=log_tr,
                category="Logs & Trees",
                category_tr="Ağaçlar / Keresteler",
                profession="Woodcutting",
                grade="Standard",
                base_material=log_name,
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=log_regions,
                hsv_ranges=_log_hsv,
                secondary_hsv_ranges=log_canopy_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description=f"{log_name} harvestable tree trunk.",
            )
        )

    # =========================================================================
    # CATEGORY 5: COOKING INGREDIENTS, FRUITS, VEGETABLES, FIBERS & AQUATIC
    # =========================================================================
    mela_hsv: Tuple[HSVRange, ...] = MELA_HSV_RANGES
    mela_leaf_hsv: Tuple[HSVRange, ...] = (MELA_LEAF_HSV_RANGE,)
    mela_regions = (
        "Western Altamia Highland",
        "Pilgrim's Ridge",
        "Verteron",
        "Poeta",
        "Eltnen",
    )
    for grade, name, name_tr in (
        ("Standard", "Mela", "Mela (Kırmızı Meyveli Çalı)"),
        ("Fresh", "Fresh Mela", "Taze Mela"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Cooking & Fibers",
                category_tr="Meyve / Sebze / Deniz Ürünleri",
                profession="Cooking & Tailoring",
                grade=grade,
                base_material="Mela",
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=mela_regions,
                hsv_ranges=mela_hsv,
                secondary_hsv_ranges=mela_leaf_hsv,
                shows_on_minimap=False,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description="Red-berry fruit bush with green leaves below white gather crest.",
            )
        )

    cooking_and_fibers: Tuple[Tuple[str, str, Tuple[str, ...], Tuple[HSVRange, ...], str], ...] = (
        ("Conide", "Conide (Kabuklu Deniz Ürünü)", ("Verteron", "Ellun River Midstream", "Poeta"), (((85, 35, 95), (115, 150, 220)),), "Shellfish aquatic node."),
        ("Coriolus", "Coriolus Mantarı", ("Poeta", "Verteron", "Altgard", "Scarlet Forest Observation Site"), (((12, 55, 85), (26, 185, 220)),), "Forest mushroom cluster."),
        ("Cypri", "Cypri Balığı", ("Verteron", "Ellun River Midstream", "Heiron", "Altgard"), (((90, 50, 100), (118, 190, 235)),), "Freshwater/coastal fish school."),
        ("Inina", "Inina İstiridyesi", ("Altgard", "Ishalgen", "Morheim", "Ellun River Midstream"), (((88, 30, 105), (115, 145, 230)),), "Shellfish gathering node."),
        ("Kukuru", "Kukuru Sebzesi", ("Poeta", "Verteron", "Altgard", "Pilgrim's Ridge"), (((22, 85, 110), (42, 245, 245)),), "Wild root/bulb vegetable plant."),
        ("Morfa", "Morfa Tahılı", ("Verteron", "Altgard", "Eltnen", "Morheim"), (((18, 95, 125), (32, 245, 250)),), "Golden grain stalk bundle."),
        ("Ohkra", "Ohkra (Bamya Sebzesi)", ("Verteron", "Eltnen", "Pilgrim's Ridge", "Western Altamia Highland"), (((36, 80, 90), (68, 250, 235)),), "Green vegetable pod plant."),
        ("Pujery", "Pujery Balığı", ("Altgard", "Morheim", "Beluslan", "Ellun River Midstream"), (((92, 55, 100), (122, 195, 235)),), "Cold-water fish node."),
        ("Raydam", "Raydam Meyvesi", ("Poeta", "Verteron", "Pilgrim's Ridge", "Western Altamia Highland"), (((4, 115, 110), (18, 255, 250)),), "Red-orange orchard fruit bush."),
        ("Ruko", "Ruko (Anathe Lifi)", ("Verteron", "Poeta", "Eltnen", "Pilgrim's Ridge"), (((28, 45, 130), (52, 165, 245)),), "Anathe tailoring fiber plant."),
        ("Tikel", "Tikel Lifi", ("Altgard", "Ishalgen", "Morheim", "Beluslan"), (((25, 45, 120), (50, 160, 240)),), "Asmodian tailoring fiber plant."),
        ("Briga", "Briga Meyvesi", ("Eltnen", "Heiron", "Inggison"), (((12, 110, 115), (26, 250, 250)),), "Aromatic cooking fruit shrub."),
        ("Liguri", "Liguri Sebzesi", ("Morheim", "Beluslan", "Gelkmaros"), (((34, 70, 95), (64, 240, 235)),), "Hardy highland cooking vegetable."),
    )
    for cf_name, cf_tr, cf_regions, cf_hsv, cf_desc in cooking_and_fibers:
        add_spec(
            MaterialSpec(
                name=cf_name,
                name_tr=cf_tr,
                category="Cooking & Fibers",
                category_tr="Meyve / Sebze / Deniz Ürünleri",
                profession="Cooking & Tailoring",
                grade="Standard",
                base_material=cf_name,
                grades_available=("Standard", "Fresh"),
                max_harvests=3,
                regions=cf_regions,
                hsv_ranges=cf_hsv,
                shows_on_minimap=True,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                visual_description=cf_desc,
            )
        )

    # =========================================================================
    # CATEGORY 6: ODYLE & AETHER (Odyle / Enerji Küresi - Flight/Ground Extraction)
    # =========================================================================
    odyle_hsv: Tuple[HSVRange, ...] = (ODYLE_HSV_RANGE,)
    odyle_regions = OFFICIAL_REGIONS
    for grade, name, name_tr in (
        ("Standard", "Odyle", "Odyle Enerji Küresi"),
        ("Fine", "Fine Odyle", "Kaliteli Odyle"),
        ("Pure", "Pure Odyle", "Saf Odyle"),
        ("Radiant", "Radiant Odyle", "Parlak Odyle"),
    ):
        add_spec(
            MaterialSpec(
                name=name,
                name_tr=name_tr,
                category="Odyle & Aether",
                category_tr="Odyle / Enerji Küresi",
                profession="Flight / Ground Extraction",
                grade=grade,
                base_material="Odyle",
                grades_available=("Standard", "Fine", "Pure", "Radiant"),
                max_harvests=3,
                regions=odyle_regions,
                hsv_ranges=odyle_hsv,
                shows_on_minimap=False,
                visible_in_3d_crest=True,
                visible_in_floating_box=True,
                in_air=True,
                visual_description=(
                    "Floating green/teal energy orb in 3D world with [E]/[F] interaction box; "
                    "NEVER the white feather on the minimap."
                ),
            )
        )

    # F-1090: grades_available must only promise grades that actually exist as
    # specs for that base material. Extended entries were built with
    # ("Standard", "Fine", "Pure", "Radiant") / ("Standard", "Fresh") while
    # only the Standard spec exists, so trim each base to its real grades
    # (in MATERIAL_GRADES order) instead of advertising phantom variants.
    _grades_by_base: Dict[str, set] = {}
    for _spec in specs:
        _grades_by_base.setdefault(_spec.base_material, set()).add(_spec.grade)
    for _spec in specs:
        _present = _grades_by_base[_spec.base_material]
        _spec.grades_available = tuple(
            g for g in MATERIAL_GRADES if g in _present
        ) or (_spec.grade,)

    return MaterialCatalog(specs)


AION2_MATERIALS_CATALOG: MaterialCatalog = _build_materials_catalog()


def _build_materials_by_name() -> Dict[str, MaterialSpec]:
    """Build the by-name index keyed by English canonical name, Turkish name,
    and their lowercase forms (F-1116): the old index only carried the
    canonical English name, so Turkish / case-variant lookups missed."""
    by_name: Dict[str, MaterialSpec] = {}
    for spec in AION2_MATERIALS_CATALOG:
        for key in (spec.name, spec.name_tr, spec.name.lower(), spec.name_tr.lower()):
            by_name.setdefault(key, spec)
    return by_name


AION2_MATERIALS_BY_NAME: Dict[str, MaterialSpec] = _build_materials_by_name()


def _build_region_index() -> Dict[str, List[MaterialSpec]]:
    """Normalized-region -> specs index, built once at import.

    get_region_expected_materials used to rescan the whole catalog and
    re-normalize every spec region on every call (~2 ms), and it is called
    up to 3x per classify_3d_material_patch call. The index answers the
    common exact-region case in O(1); partial/substring region spellings
    still fall through to the original scan, so matching behaviour is
    unchanged. A spec is listed once per normalized region key even if
    two of its raw region strings normalize identically.
    """
    index: Dict[str, List[MaterialSpec]] = {}
    for spec in AION2_MATERIALS_CATALOG:
        seen_keys: set = set()
        for reg in spec.regions:
            key = _normalize_region_token(reg)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            index.setdefault(key, []).append(spec)
    return index


# Built after get_region_expected_materials below (it needs
# _normalize_region_token, defined there); see _REGION_INDEX assignment.


def get_material_spec(name: str) -> Optional[MaterialSpec]:
    """Look up a MaterialSpec by its English or Turkish name (case-insensitive)."""
    if not name or not isinstance(name, str):
        return None
    return AION2_MATERIALS_CATALOG.get(name)


def is_rejected_non_material(name_or_kind: str) -> bool:
    """Return True if the label/kind is a non-material (feather, monster, NPC, quest).

    Strictly rejects:
    - Feathers ("feather", "tüy", "white feather", "minimap feather")
    - Monsters ("monster", "yellow-maned zaif", "highland odyle sparkie", "canavar")
    - NPCs and Quests ("functional npc", "quest", "npc")

    The word-boundary patterns are compiled once at import
    (_REJECTED_PATTERNS) instead of rebuilt with re.escape + re.search on
    every call: this runs on the per-detection hot path and the old code
    spent ~0.29 ms per label just (re)compiling ~37 patterns.
    """
    if not name_or_kind or not isinstance(name_or_kind, str):
        return False

    # [O10] D52/F-1101 (word-boundary) × D92/#05: "_" bir kelime karakteri
    # olduğundan "monster_white_dot" tek kelime sayılıyor ve "monster"
    # anahtarı eşleşmiyordu. Alt çizgi ayraçtır (etiket/kind token'ları
    # underscore-joined gelir) — boşluğa çevrilir; F-1101'in koruması
    # bozulmaz ("mobile" hâlâ tek kelime, "mob" eşleşmez; "request"te
    # "quest" eşleşmez). Katalogda "_" içeren isim yok, erken dönüş etkilenmez.
    norm = " ".join(name_or_kind.strip().lower().replace("_", " ").split())
    if not norm:
        return False

    # If it exactly matches a known valid gatherable material in the catalog, never reject
    if norm in AION2_MATERIALS_CATALOG or norm in (
        "gatherable material",
        "gatherable_node",
        "gatherable",
        "toplanabilir materyal",
    ):
        return False

    # F-1101: match keywords on word boundaries, not as raw substrings.
    # Substring matching rejected innocent labels ("mob" inside "mobile",
    # "npc" inside "npcs"-style tokens, "quest" inside "request"). A
    # single-word keyword may carry a trailing plural "s".
    for pattern in _REJECTED_PATTERNS:
        if pattern.search(norm):
            return True

    return False


def _compile_rejected_patterns() -> Tuple["re.Pattern[str]", ...]:
    """Compile the F-1101 word-boundary patterns once (see is_rejected_non_material)."""
    compiled = []
    for token in REJECTED_NON_MATERIAL_KEYWORDS:
        source = r"(?<!\w)" + re.escape(token) + (r"s?(?!\w)" if " " not in token else r"(?!\w)")
        compiled.append(re.compile(source))
    return tuple(compiled)


_REJECTED_PATTERNS: Tuple["re.Pattern[str]", ...] = _compile_rejected_patterns()


def _normalize_region_token(text: str) -> str:
    """Normalize region string by stripping apostrophes and extra whitespace."""
    return " ".join(
        text.strip().lower().replace("'", "").replace("’", "").replace("_", " ").split()
    )


def _region_tokens_contain(haystack: str, needle: str) -> bool:
    """S121 (v19/P3-8): normalize edilmiş bölge adında `needle`'ın
    belirteçleri bitişik alt dizi mi? Ham alt-dize yerine kullanılır."""
    hay = haystack.split()
    nee = needle.split()
    if not hay or not nee or len(nee) > len(hay):
        return False
    return any(hay[i:i + len(nee)] == nee for i in range(len(hay) - len(nee) + 1))


def get_region_expected_materials(region_name: str) -> List[MaterialSpec]:
    """Return the known gatherable MaterialSpec entries for a given Aion 2 / Aion region.

    Supports partial/case-insensitive matching for region names such as
    "Pilgrim's Ridge", "Western Altamia Highland", "Verteron", "Altgard", etc.

    F-1092: an empty/blank region and an unknown region now behave the same —
    both return [] (no region is known, so nothing can be expected there).
    The old code returned the *whole catalog* for an empty region and []
    for an unknown one, which silently turned "no region given" into
    "every material is expected here".
    """
    if not region_name or not isinstance(region_name, str):
        return []

    target = _normalize_region_token(region_name)
    if not target:
        return []

    # Fast path: exact normalized region -> prebuilt index (O(1)). This
    # returns the same specs, in the same catalog order, as the scan below
    # would for an exact match (target == reg_low). Anything else
    # (partial/substring spelling) falls through to the original scan.
    exact = _REGION_INDEX.get(target)
    if exact is not None:
        return list(exact)

    matched: List[MaterialSpec] = []
    for spec in AION2_MATERIALS_CATALOG:
        for reg in spec.regions:
            reg_low = _normalize_region_token(reg)
            # S121 (v19/P3-8): ham alt-dize ("ridge" ⊂ her "* Ridge")
            # fazla eşleşiyordu; belirteç sınırında alt dizi aranır:
            # "altgard" ⊂ "altgard depths" evet, "ash" ⊂ "ashen" hayır.
            if (
                target == reg_low
                or _region_tokens_contain(reg_low, target)
                or _region_tokens_contain(target, reg_low)
            ):
                matched.append(spec)
                break

    return matched


_REGION_INDEX: Dict[str, List[MaterialSpec]] = _build_region_index()


def _mask_ratio(hsv: np.ndarray, lower: HSVBound, upper: HSVBound) -> float:
    """Compute the fraction of pixels in `hsv` falling within `[lower, upper]`."""
    total_px = float(max(1, hsv.shape[0] * hsv.shape[1]))
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return float(np.count_nonzero(mask)) / total_px


def _mask_count(hsv: np.ndarray, lower: HSVBound, upper: HSVBound) -> int:
    """Count pixels in `hsv` falling within `[lower, upper]`."""
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return int(np.count_nonzero(mask))


def _more_specific_catalog_material(
    hsv: np.ndarray,
    chosen_name: str,
    coarse_ratio: float,
    region_name: str,
    ore_ratio: float,
) -> Optional[str]:
    """Prefer a catalog essence when its color is clearer than the coarse family."""
    # Only allow refinement when chosen_name is a coarse fallback (e.g. weak Orichalcum Ore)
    if chosen_name != "Orichalcum Ore":
        return None

    chosen_spec = AION2_MATERIALS_CATALOG.get(chosen_name)
    if not chosen_spec:
        return None
    target_category = chosen_spec.category

    # If Orichalcum has strong vein signal, keep Orichalcum Ore
    if ore_ratio >= 0.04:
        return None

    # F-1102: refinement is region-gated. Without a known region the regional
    # constraint used to silently lift and any-region essences could win; an
    # unknown/empty region now means "no refinement", keeping the coarse
    # primary instead of guessing a specific essence.
    if not region_name:
        return None
    region_specs = get_region_expected_materials(region_name)
    if not region_specs:
        return None
    region_bases = {spec.base_material.lower() for spec in region_specs}
    best_name = ""
    best_score = 0.0
    seen = set()
    # NOTE (D79): an earlier draft of this change iterated region_specs
    # instead of the catalog here, but that saves nothing measurable — the
    # region_bases filter below already runs BEFORE any _mask_ratio /
    # inRange call, so non-region specs never cost a mask; the loop only
    # skips ~84 trivial iterations. Reverted to the original catalog scan.
    for spec in AION2_MATERIALS_CATALOG:
        # Category boundary guard: NEVER cross categories (e.g. Gemstone -> Ore, Tree -> Ore)
        if spec.category != target_category:
            continue
        base_key = spec.base_material.strip().lower()
        if base_key in seen or spec.grade not in ("Standard", ""):
            continue
        seen.add(base_key)
        if spec.name == chosen_name or spec.base_material == chosen_name:
            continue
        # If region is specified, require regional presence
        if region_bases and base_key not in region_bases:
            continue
        ratio = 0.0
        for lower, upper in spec.hsv_ranges:
            ratio = max(ratio, _mask_ratio(hsv, lower, upper))
        if ratio < 0.08:
            continue
        score = ratio + (0.02 if base_key in region_bases else 0.0)
        if score > best_score:
            best_score = score
            best_name = spec.name
    if not best_name:
        return None
    if best_score >= float(coarse_ratio) + 0.15:
        return best_name
    return None


@dataclass
class MaterialVisualDescriptor:
    """Real-time visual feature descriptors extracted from a 3D resource mesh patch.

    Captures genuine computer vision features without synthetic shortcuts:
    - Multi-bin HSV color distributions & statistical color masks
    - 3D texture energy, directional edge gradients, and specular highlights
    - Morphological aspect ratios and spatial profiles
    - Bilateral symmetry and brightness of 3D winged crests
    """

    # 1. HSV Color Distributions
    hue_histogram: List[float] = field(default_factory=lambda: [0.0] * 16)
    dominant_hue: float = 0.0
    mean_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    std_hsv: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    color_ratios: Dict[str, float] = field(default_factory=dict)

    # 2. 3D Texture & Crest Features
    texture_contrast: float = 0.0
    gradient_magnitude_mean: float = 0.0
    vertical_edge_ratio: float = 0.5
    horizontal_edge_ratio: float = 0.5
    specular_highlight_ratio: float = 0.0
    spot_cluster_count: int = 0
    crest_symmetry_score: float = 0.0
    crest_brightness: float = 0.0

    # 3. Geometric & Aspect Ratio Features
    aspect_ratio: float = 1.0  # width / height
    height_width_ratio: float = 1.0  # height / width
    morphological_profile: str = "balanced"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize descriptor to plain dictionary."""
        return asdict(self)


def _empty_hsv_distribution() -> Dict[str, Any]:
    """Zero-valued HSV distribution returned for invalid/empty input."""
    return {
        "hue_histogram": [0.0] * 16,
        "dominant_hue": 0.0,
        "mean_hsv": (0.0, 0.0, 0.0),
        "std_hsv": (0.0, 0.0, 0.0),
        "color_ratios": {},
        "total_px": 0,
    }


def _compute_hsv_color_distribution_from_hsv(hsv: Optional[np.ndarray]) -> Dict[str, Any]:
    """HSV distribution from an already-converted HSV array.

    Split out of compute_hsv_color_distribution so a caller that already
    holds the HSV image (extract_visual_descriptor / classify) does not pay
    for a second cv2.cvtColor. Behaviour is identical to the public
    function for a valid input; None/empty input yields the zero dict.
    """
    if (
        hsv is None
        or not isinstance(hsv, np.ndarray)
        or hsv.size == 0
        or hsv.ndim != 3
        or hsv.shape[2] < 3
    ):
        return _empty_hsv_distribution()

    total_px = float(max(1, hsv.shape[0] * hsv.shape[1]))

    # Compute 16-bin Hue histogram for chromatic pixels (S >= 25, 20 <= V <= 250)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    chroma_mask = (sat >= 25) & (val >= 20) & (val <= 250)
    chroma_count = int(np.count_nonzero(chroma_mask))

    if chroma_count > 0:
        hist = cv2.calcHist(
            [hsv], [0], chroma_mask.astype(np.uint8), [16], [0, 180]
        ).flatten()
        norm_hist = [float(c) / float(chroma_count) for c in hist]
        dom_bin = int(np.argmax(hist))
        dominant_hue = float(dom_bin * 11.25 + 5.625)
    else:
        norm_hist = [0.0] * 16
        dominant_hue = 0.0

    mean_hsv = (
        float(np.mean(hsv[:, :, 0])),
        float(np.mean(hsv[:, :, 1])),
        float(np.mean(hsv[:, :, 2])),
    )
    std_hsv = (
        float(np.std(hsv[:, :, 0])),
        float(np.std(hsv[:, :, 1])),
        float(np.std(hsv[:, :, 2])),
    )

    # Specific color masks for taxonomy categories (Aria uses the single
    # canonical range from ARIA_HSV_RANGE; see F-1089).
    aria_px = _mask_count(hsv, ARIA_HSV_RANGE[0], ARIA_HSV_RANGE[1])
    red_berry_px = _mask_count(hsv, MELA_HSV_RANGES[0][0], MELA_HSV_RANGES[0][1]) + _mask_count(
        hsv, MELA_HSV_RANGES[1][0], MELA_HSV_RANGES[1][1]
    )
    ruby_crystal_px = _mask_count(hsv, RUBY_HSV_RANGES[0][0], RUBY_HSV_RANGES[0][1]) + _mask_count(
        hsv, RUBY_HSV_RANGES[1][0], RUBY_HSV_RANGES[1][1]
    )
    green_leaf_px = _mask_count(hsv, GREEN_LEAF_MASK_HSV_RANGE[0], GREEN_LEAF_MASK_HSV_RANGE[1])
    green_broad_px = _mask_count(hsv, GREEN_BROAD_MASK_HSV_RANGE[0], GREEN_BROAD_MASK_HSV_RANGE[1])
    targena_px = _mask_count(hsv, TARGENA_MASK_HSV_RANGE[0], TARGENA_MASK_HSV_RANGE[1])
    calendula_px = _mask_count(hsv, CALENDULA_HSV_RANGE[0], CALENDULA_HSV_RANGE[1])
    sapphire_px = _mask_count(hsv, SAPPHIRE_HSV_RANGE[0], SAPPHIRE_HSV_RANGE[1])
    azpha_px = _mask_count(hsv, AZPHA_HSV_RANGE[0], AZPHA_HSV_RANGE[1])
    diamond_px = _mask_count(hsv, DIAMOND_HSV_RANGE[0], DIAMOND_HSV_RANGE[1])
    odyle_px = _mask_count(hsv, ODYLE_HSV_RANGE[0], ODYLE_HSV_RANGE[1])
    bronze_ore_px = _mask_count(hsv, ORICHALCUM_BRONZE_MASK_HSV_RANGE[0], ORICHALCUM_BRONZE_MASK_HSV_RANGE[1])
    teal_ore_px = _mask_count(hsv, ORICHALCUM_TEAL_MASK_HSV_RANGE[0], ORICHALCUM_TEAL_MASK_HSV_RANGE[1])
    dark_rock_px = _mask_count(hsv, DARK_ROCK_MASK_HSV_RANGE[0], DARK_ROCK_MASK_HSV_RANGE[1])
    bark_px = _mask_count(hsv, BARK_TRUNK_MASK_HSV_RANGE[0], BARK_TRUNK_MASK_HSV_RANGE[1])

    color_ratios = {
        "aria_purple": round(aria_px / total_px, 4),
        "red_berry": round(red_berry_px / total_px, 4),
        "ruby_crystal": round(ruby_crystal_px / total_px, 4),
        "green_leaf": round(green_leaf_px / total_px, 4),
        "green_broad": round(green_broad_px / total_px, 4),
        "targena_gold_green": round(targena_px / total_px, 4),
        "calendula_orange": round(calendula_px / total_px, 4),
        "sapphire_blue": round(sapphire_px / total_px, 4),
        "azpha_blue_violet": round(azpha_px / total_px, 4),
        "diamond_ice": round(diamond_px / total_px, 4),
        "odyle_orb": round(odyle_px / total_px, 4),
        "orichalcum_vein": round((bronze_ore_px + teal_ore_px) / total_px, 4),
        "bronze_ore": round(bronze_ore_px / total_px, 4),
        "teal_ore": round(teal_ore_px / total_px, 4),
        "dark_rock": round(dark_rock_px / total_px, 4),
        "bark_trunk": round(bark_px / total_px, 4),
    }

    return {
        "hue_histogram": norm_hist,
        "dominant_hue": round(dominant_hue, 2),
        "mean_hsv": tuple(round(v, 2) for v in mean_hsv),
        "std_hsv": tuple(round(v, 2) for v in std_hsv),
        "color_ratios": color_ratios,
        "total_px": int(total_px),
    }


def compute_hsv_color_distribution(mesh_bgr: np.ndarray) -> Dict[str, Any]:
    """Compute multi-bin HSV color distributions and statistical color masks."""
    if (
        mesh_bgr is None
        or not isinstance(mesh_bgr, np.ndarray)
        or mesh_bgr.size == 0
        or mesh_bgr.ndim != 3
        or mesh_bgr.shape[2] < 3
    ):
        return _empty_hsv_distribution()

    hsv = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2HSV)
    return _compute_hsv_color_distribution_from_hsv(hsv)


def _empty_crest_texture() -> Dict[str, Any]:
    """Zero-valued crest/texture dict returned for invalid/empty input."""
    return {
        "texture_contrast": 0.0,
        "gradient_magnitude_mean": 0.0,
        "vertical_edge_ratio": 0.5,
        "horizontal_edge_ratio": 0.5,
        "specular_highlight_ratio": 0.0,
        "spot_cluster_count": 0,
        "crest_symmetry_score": 0.0,
        "crest_brightness": 0.0,
    }


def _compute_3d_crest_texture_from_hsv(
    mesh_bgr: np.ndarray,
    hsv: Optional[np.ndarray],
    crest_bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Crest/texture descriptors from an already-converted HSV array.

    Split out of compute_3d_crest_texture so a caller that already holds
    the HSV image does not pay for a second cv2.cvtColor. Behaviour is
    identical to the public function for a valid input; invalid input (or
    a missing HSV array) yields the zero dict.
    """
    if (
        mesh_bgr is None
        or not isinstance(mesh_bgr, np.ndarray)
        or mesh_bgr.size == 0
        or mesh_bgr.ndim != 3
        or mesh_bgr.shape[2] < 3
        or hsv is None
        or not isinstance(hsv, np.ndarray)
        or hsv.size == 0
    ):
        return _empty_crest_texture()

    gray = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    total_px = float(max(1, h * w))

    contrast = float(np.std(gray)) / 255.0

    # Sobel directional gradients (ksize=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grad_mean = float(np.mean(mag)) / 255.0

    sum_gx = float(np.sum(np.abs(gx)))
    sum_gy = float(np.sum(np.abs(gy)))
    total_g = sum_gx + sum_gy + 1e-6
    vert_ratio = sum_gx / total_g
    horiz_ratio = sum_gy / total_g

    # Specular highlights (characteristic of crystal facets and reflective ore)
    # `hsv` is the shared conversion passed in by the caller (see
    # _compute_3d_crest_texture_from_hsv) — no second cvtColor here.
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    spec_mask = (val >= 210) & ((sat <= 50) | ((val >= 240) & (sat >= 90)))
    spec_ratio = float(np.count_nonzero(spec_mask)) / total_px

    # Connected component spot clustering (characteristic of Mela red berries)
    red_mask = cv2.inRange(hsv, MELA_HSV_RANGES[0][0], MELA_HSV_RANGES[0][1]) | cv2.inRange(
        hsv, MELA_HSV_RANGES[1][0], MELA_HSV_RANGES[1][1]
    )
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        red_mask.astype(np.uint8), connectivity=8
    )
    spot_cluster_count = sum(
        1 for i in range(1, num_labels) if 3 <= stats[i, cv2.CC_STAT_AREA] <= 180
    )

    # 3D Winged Crest symmetry and brightness
    crest_crop = crest_bgr
    if crest_crop is None and h >= 24:
        crest_crop = mesh_bgr[: max(8, int(h * 0.25)), :]

    crest_symmetry = 0.0
    crest_bright = 0.0
    if crest_crop is not None and crest_crop.size > 0:
        cg = cv2.cvtColor(crest_crop, cv2.COLOR_BGR2GRAY)
        crest_bright = float(np.mean(cg)) / 255.0
        cw = cg.shape[1]
        mid = cw // 2
        if mid >= 4:
            left = cg[:, :mid].astype(float)
            right = np.fliplr(cg[:, mid + (cw % 2) :]).astype(float)
            diff = float(np.mean(np.abs(left - right))) / 255.0
            crest_symmetry = max(0.0, 1.0 - diff * 2.2)

    return {
        "texture_contrast": round(contrast, 4),
        "gradient_magnitude_mean": round(grad_mean, 4),
        "vertical_edge_ratio": round(vert_ratio, 4),
        "horizontal_edge_ratio": round(horiz_ratio, 4),
        "specular_highlight_ratio": round(spec_ratio, 4),
        "spot_cluster_count": int(spot_cluster_count),
        "crest_symmetry_score": round(crest_symmetry, 4),
        "crest_brightness": round(crest_bright, 4),
    }


def compute_3d_crest_texture(
    mesh_bgr: np.ndarray, crest_bgr: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """Compute 3D crest and mesh texture descriptors: directional edge gradients, specular highlights, and symmetry."""
    if (
        mesh_bgr is None
        or not isinstance(mesh_bgr, np.ndarray)
        or mesh_bgr.size == 0
        or mesh_bgr.ndim != 3
        or mesh_bgr.shape[2] < 3
    ):
        return _empty_crest_texture()

    hsv = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2HSV)
    return _compute_3d_crest_texture_from_hsv(mesh_bgr, hsv, crest_bgr)


def compute_aspect_ratio_features(
    mesh_bgr: np.ndarray,
    bbox_aspect_ratio: Optional[float] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, Any]:
    """Compute aspect ratio and morphological profile of the resource node."""
    if bbox_aspect_ratio is not None and bbox_aspect_ratio > 0:
        ar = float(bbox_aspect_ratio)
    elif bbox is not None and len(bbox) == 4 and bbox[3] > 0:
        ar = float(bbox[2]) / float(bbox[3])
    elif (
        mesh_bgr is not None
        and isinstance(mesh_bgr, np.ndarray)
        and mesh_bgr.ndim >= 2
        and mesh_bgr.shape[0] > 0
    ):
        ar = float(mesh_bgr.shape[1]) / float(mesh_bgr.shape[0])
    else:
        ar = 1.0

    ar = round(max(0.05, min(10.0, ar)), 3)
    hw_ratio = round(1.0 / max(0.01, ar), 3)

    if hw_ratio >= 1.35:
        profile = "vertical_trunk"
    elif ar >= 1.35:
        profile = "horizontal_vein"
    elif ar >= 1.08:
        profile = "squat_bush"
    elif hw_ratio >= 1.10:
        profile = "upright_herb"
    else:
        profile = "balanced"

    return {
        "aspect_ratio": ar,
        "height_width_ratio": hw_ratio,
        "morphological_profile": profile,
    }


def _extract_visual_descriptor_with_hsv(
    mesh_bgr: np.ndarray,
    bbox_aspect_ratio: Optional[float] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    crest_bgr: Optional[np.ndarray] = None,
) -> Tuple[MaterialVisualDescriptor, Optional[np.ndarray]]:
    """Extract the visual descriptor plus the shared HSV conversion.

    classify_3d_material_patch used to convert BGR->HSV three times per
    patch (once in each of compute_hsv_color_distribution and
    compute_3d_crest_texture, once more for the refinement step). This
    converts once and shares the array; the public extract function below
    keeps its old signature and discards the array.
    """
    hsv: Optional[np.ndarray] = None
    if (
        mesh_bgr is not None
        and isinstance(mesh_bgr, np.ndarray)
        and mesh_bgr.size > 0
        and mesh_bgr.ndim == 3
        and mesh_bgr.shape[2] >= 3
    ):
        hsv = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2HSV)

    hsv_data = _compute_hsv_color_distribution_from_hsv(hsv)
    tex_data = _compute_3d_crest_texture_from_hsv(mesh_bgr, hsv, crest_bgr)
    ar_data = compute_aspect_ratio_features(
        mesh_bgr, bbox_aspect_ratio=bbox_aspect_ratio, bbox=bbox
    )

    return (
        MaterialVisualDescriptor(
            hue_histogram=hsv_data["hue_histogram"],
            dominant_hue=hsv_data["dominant_hue"],
            mean_hsv=hsv_data["mean_hsv"],
            std_hsv=hsv_data["std_hsv"],
            color_ratios=hsv_data["color_ratios"],
            texture_contrast=tex_data["texture_contrast"],
            gradient_magnitude_mean=tex_data["gradient_magnitude_mean"],
            vertical_edge_ratio=tex_data["vertical_edge_ratio"],
            horizontal_edge_ratio=tex_data["horizontal_edge_ratio"],
            specular_highlight_ratio=tex_data["specular_highlight_ratio"],
            spot_cluster_count=tex_data["spot_cluster_count"],
            crest_symmetry_score=tex_data["crest_symmetry_score"],
            crest_brightness=tex_data["crest_brightness"],
            aspect_ratio=ar_data["aspect_ratio"],
            height_width_ratio=ar_data["height_width_ratio"],
            morphological_profile=ar_data["morphological_profile"],
        ),
        hsv,
    )


def extract_visual_descriptor(
    mesh_bgr: np.ndarray,
    bbox_aspect_ratio: Optional[float] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    crest_bgr: Optional[np.ndarray] = None,
) -> MaterialVisualDescriptor:
    """Extract full real-time visual feature descriptors from a 3D resource mesh patch."""
    desc, _ = _extract_visual_descriptor_with_hsv(
        mesh_bgr, bbox_aspect_ratio=bbox_aspect_ratio, bbox=bbox, crest_bgr=crest_bgr
    )
    return desc


AION2_TAXONOMY_HIERARCHY: Dict[str, Dict[str, Any]] = {
    "Herbs": {
        "category_tr": "Bitkiler / Otlar",
        "profession": "Alchemy",
        "harvest_type": "Vitality Extraction",
        "interaction_keys": ("f", "e"),
        "primary_materials": {
            "Aria": {
                "name_tr": "Aria (Mor Kristal Bitki)",
                "visual_profile": "purple_violet_crystal_herb",
                "hsv_range": ARIA_HSV_RANGE,
                "expected_aspect": "upright_herb",
                "typical_aspect_ratio": 0.85,
            },
            "Targena": {
                "name_tr": "Targena Otu",
                "visual_profile": "golden_yellow_green_herb",
                "hsv_range": TARGENA_HSV_RANGE,
                "expected_aspect": "upright_herb",
                "typical_aspect_ratio": 0.90,
            },
            "Calendula": {
                "name_tr": "Calendula (Aynısafa Otu)",
                "visual_profile": "orange_gold_marigold",
                "hsv_range": CALENDULA_HSV_RANGE,
                "expected_aspect": "upright_herb",
                "typical_aspect_ratio": 0.95,
            },
            "Azpha": {
                "name_tr": "Azpha Otu",
                "visual_profile": "blue_violet_highland_herb",
                "hsv_range": AZPHA_HSV_RANGE,
                "expected_aspect": "upright_herb",
                "typical_aspect_ratio": 0.88,
            },
        },
    },
    "Cooking & Fibers": {
        "category_tr": "Meyve / Sebze / Deniz Ürünleri",
        "profession": "Cooking & Tailoring",
        "harvest_type": "Vitality Extraction",
        "interaction_keys": ("f", "e"),
        "primary_materials": {
            "Mela": {
                "name_tr": "Mela (Kırmızı Meyveli Çalı)",
                "visual_profile": "red_berries_green_bush",
                "hsv_range": MELA_HSV_RANGES[0],
                "secondary_hsv_range": MELA_LEAF_HSV_RANGE,
                "expected_aspect": "squat_bush",
                "typical_aspect_ratio": 1.45,
            },
        },
    },
    "Ores & Metals": {
        "category_tr": "Madenler / Cevherler",
        "profession": "Blacksmithing / Armorsmithing",
        "harvest_type": "Vitality Extraction",
        "interaction_keys": ("f", "e"),
        "primary_materials": {
            "Orichalcum Ore": {
                "name_tr": "Orichalcum Cevheri",
                "visual_profile": "bronze_teal_gold_vein",
                "hsv_range": ORICHALCUM_HSV_RANGES[0],
                "secondary_hsv_range": ORICHALCUM_HSV_RANGES[1],
                "expected_aspect": "horizontal_vein",
                "typical_aspect_ratio": 1.55,
            },
        },
    },
    "Gemstones": {
        "category_tr": "Değerli Taşlar / Gemler",
        "profession": "Handicrafting",
        "harvest_type": "Vitality Extraction",
        "interaction_keys": ("f", "e"),
        "primary_materials": {
            "Sapphire Gemstone": {
                "name_tr": "Safir Değerli Taş",
                "visual_profile": "deep_cyan_blue_crystal",
                "hsv_range": SAPPHIRE_HSV_RANGE,
                "expected_aspect": "crystal_spire",
                "typical_aspect_ratio": 0.80,
            },
            "Ruby Gemstone": {
                "name_tr": "Yakut Değerli Taş",
                "visual_profile": "crimson_crystal_spire",
                "hsv_range": RUBY_HSV_RANGES[0],
                "expected_aspect": "crystal_spire",
                "typical_aspect_ratio": 0.70,
            },
            "Diamond Gemstone": {
                "name_tr": "Elmas Değerli Taş",
                "visual_profile": "brilliant_white_ice_crystal",
                "hsv_range": DIAMOND_HSV_RANGE,
                "expected_aspect": "crystal_spire",
                "typical_aspect_ratio": 0.85,
            },
        },
    },
    "Logs & Trees": {
        "category_tr": "Ağaçlar / Keresteler",
        "profession": "Woodcutting",
        "harvest_type": "Vitality Extraction",
        "interaction_keys": ("f", "e"),
        "primary_materials": {
            "Yggdrasil Log": {
                "name_tr": "Yggdrasil Kerestesi",
                "visual_profile": "brown_bark_trunk_canopy",
                "hsv_range": YGGDRASIL_LOG_HSV_RANGE,
                "secondary_hsv_range": LOG_CANOPY_HSV_RANGE,
                "expected_aspect": "vertical_trunk",
                "typical_aspect_ratio": 0.55,
            },
            "Asvata Log": {
                "name_tr": "Asvata Kerestesi",
                "visual_profile": "asmodian_rugged_bark_trunk",
                "hsv_range": ASVATA_LOG_HSV_RANGE,
                "secondary_hsv_range": LOG_CANOPY_HSV_RANGE,
                "expected_aspect": "vertical_trunk",
                "typical_aspect_ratio": 0.50,
            },
        },
    },
    "Odyle & Aether": {
        "category_tr": "Odyle / Enerji Küresi",
        "profession": "Flight / Ground Extraction",
        "harvest_type": "Aether Extraction",
        "interaction_keys": ("f", "e"),
        "primary_materials": {
            "Odyle": {
                "name_tr": "Odyle Enerji Küresi",
                "visual_profile": "floating_teal_green_energy_orb",
                "hsv_range": ODYLE_HSV_RANGE,
                "expected_aspect": "balanced",
                "typical_aspect_ratio": 1.0,
            },
        },
    },
}


def map_descriptors_to_taxonomy(
    descriptor: MaterialVisualDescriptor,
    region_name: str = "",
    label_width_px: int = 0,
) -> Dict[str, Any]:
    """Map visual feature descriptors (HSV distributions, 3D textures, aspect ratios) to exact Aion 2 material taxonomy."""
    cr = descriptor.color_ratios
    reg_low = (region_name or "").strip().lower()
    is_asmodian_region = any(
        r in reg_low for r in ("altgard", "morheim", "ishalgen", "beluslan", "gelkmaros")
    )

    # 1. Aria: Purple/violet crystal plant
    aria_score = cr.get("aria_purple", 0.0) * 3.8
    if cr.get("aria_purple", 0.0) >= 0.025 and cr.get("aria_purple", 0.0) > cr.get("red_berry", 0.0) * 1.3:
        aria_score += 0.45
    if descriptor.aspect_ratio <= 1.35:
        aria_score += 0.05

    # 2. Mela: Red berries + green foliage bush
    mela_score = cr.get("red_berry", 0.0) * 2.8 + cr.get("green_leaf", 0.0) * 1.2
    if cr.get("red_berry", 0.0) >= 0.035 and (
        cr.get("green_leaf", 0.0) >= 0.02
        or cr.get("green_broad", 0.0) >= 0.04
        or descriptor.spot_cluster_count >= 1
    ):
        mela_score += 0.44
    if descriptor.aspect_ratio >= 1.05:
        mela_score += 0.08
    if cr.get("green_leaf", 0.0) < 0.015 and descriptor.specular_highlight_ratio >= 0.02:
        mela_score -= 0.35  # Distinguish from Ruby Gemstone

    # 3. Ruby Gemstone: Crimson crystal spire, specular highlights, zero/negligible foliage
    ruby_score = cr.get("ruby_crystal", 0.0) * 3.4 + descriptor.specular_highlight_ratio * 1.5
    if cr.get("ruby_crystal", 0.0) >= 0.03 and cr.get("green_leaf", 0.0) < 0.02:
        ruby_score += 0.44
    if descriptor.height_width_ratio >= 1.15:
        ruby_score += 0.08
    if cr.get("green_leaf", 0.0) >= 0.035:
        ruby_score -= 0.40  # Foliage indicates fruit bush, not crystal

    # 4. Sapphire Gemstone: Deep glowing blue/cyan crystal
    sapphire_score = cr.get("sapphire_blue", 0.0) * 3.4 + descriptor.specular_highlight_ratio * 1.4
    if cr.get("sapphire_blue", 0.0) >= 0.03 and cr.get("green_leaf", 0.0) < 0.06:
        sapphire_score += 0.42

    # 5. Diamond Gemstone: Brilliant white/ice crystal
    diamond_score = cr.get("diamond_ice", 0.0) * 3.0 + descriptor.specular_highlight_ratio * 1.6
    if cr.get("diamond_ice", 0.0) >= 0.05 and cr.get("green_leaf", 0.0) < 0.08:
        diamond_score += 0.40

    # 6. Odyle: Floating teal/green energy orb
    odyle_score = cr.get("odyle_orb", 0.0) * 3.4
    if cr.get("odyle_orb", 0.0) >= 0.045 and cr.get("targena_gold_green", 0.0) < 0.15:
        odyle_score += 0.42

    # 7. Targena: Golden-yellow & green leafy herb
    targena_score = cr.get("targena_gold_green", 0.0) * 2.6 + cr.get("green_leaf", 0.0) * 0.8
    if cr.get("targena_gold_green", 0.0) >= 0.04 or (
        cr.get("green_leaf", 0.0) >= 0.22 and cr.get("targena_gold_green", 0.0) >= 0.01
    ):
        targena_score += 0.40
    if descriptor.aspect_ratio <= 1.40:
        targena_score += 0.05

    # 8. Calendula: Orange-gold marigold blossom
    calendula_score = cr.get("calendula_orange", 0.0) * 3.2
    if cr.get("calendula_orange", 0.0) >= 0.04 and cr.get("green_leaf", 0.0) >= 0.02:
        calendula_score += 0.38

    # 9. Azpha: Blue-violet Asmodian herb
    azpha_score = cr.get("azpha_blue_violet", 0.0) * 3.0 + cr.get("green_leaf", 0.0) * 0.8
    if cr.get("azpha_blue_violet", 0.0) >= 0.035 and cr.get("green_leaf", 0.0) >= 0.04:
        azpha_score += 0.36
    if is_asmodian_region:
        azpha_score += 0.04

    # 10. Tree Logs: Yggdrasil / Asvata (tall vertical trunk aspect ratio, vertical bark texture).
    # F-1085/F-1088: the two log primaries are scored separately (they were
    # merged into one slot, so only 11 names were ever scored). Both share
    # the trunk evidence; the region breaks the tie (Asmodian regions favour
    # Asvata, Elyos regions favour Yggdrasil) and without a region the
    # hierarchy order (Yggdrasil first) decides the tie deterministically.
    tree_score = cr.get("bark_trunk", 0.0) * 3.4
    if cr.get("bark_trunk", 0.0) >= 0.14:
        tree_score += 0.35
    if descriptor.height_width_ratio >= 1.25:
        tree_score += 0.25  # Tall vertical aspect ratio
    if descriptor.vertical_edge_ratio >= 0.52:
        tree_score += 0.15  # Vertical bark striations
    if cr.get("green_leaf", 0.0) >= 0.05:
        tree_score += 0.10  # Canopy foliage
    if cr.get("dark_rock", 0.0) >= 0.35:
        tree_score -= 0.25
    yggdrasil_score = tree_score + (0.03 if not is_asmodian_region else 0.0)
    asvata_score = tree_score + (0.03 if is_asmodian_region else 0.0)

    # 11. Orichalcum Ore: Metallic bronze/teal/gold vein on dark rock
    ore_score = cr.get("orichalcum_vein", 0.0) * 3.0 + cr.get("dark_rock", 0.0) * 0.7
    if cr.get("orichalcum_vein", 0.0) >= 0.015 or cr.get("dark_rock", 0.0) >= 0.20:
        ore_score += 0.35
    if descriptor.aspect_ratio >= 1.15:
        ore_score += 0.12  # Horizontal rock vein
    if descriptor.height_width_ratio >= 1.30:
        ore_score -= 0.60  # Vertical columns are trees or spires, NOT horizontal ore veins!
    if descriptor.gradient_magnitude_mean >= 0.12:
        ore_score += 0.06  # Rough rock surface texture

    # F-1099: label_width_px used to add +0.35 to the ore score all by itself,
    # so any wide nameplate made a patch "ore". Label width is not ore
    # evidence; the parameter is kept for API compatibility but no longer
    # contributes to any score.
    _ = label_width_px

    # Score exactly the 12 canonical primaries, in hierarchy order.
    scores: Dict[str, float] = {
        "Aria": round(float(aria_score), 4),
        "Mela": round(float(mela_score), 4),
        "Ruby Gemstone": round(float(ruby_score), 4),
        "Sapphire Gemstone": round(float(sapphire_score), 4),
        "Diamond Gemstone": round(float(diamond_score), 4),
        "Odyle": round(float(odyle_score), 4),
        "Targena": round(float(targena_score), 4),
        "Calendula": round(float(calendula_score), 4),
        "Azpha": round(float(azpha_score), 4),
        "Yggdrasil Log": round(float(yggdrasil_score), 4),
        "Asvata Log": round(float(asvata_score), 4),
        "Orichalcum Ore": round(float(ore_score), 4),
    }
    assert tuple(scores) == CANONICAL_PRIMARY_MATERIALS

    # Region prior slight boost
    if region_name:
        reg_specs = get_region_expected_materials(region_name)
        reg_bases = {s.base_material.lower() for s in reg_specs}
        for mat_name in list(scores.keys()):
            spec = AION2_MATERIALS_CATALOG.get(mat_name)
            if spec and spec.base_material.lower() in reg_bases:
                scores[mat_name] = round(scores[mat_name] + 0.03, 4)

    sorted_scores = sorted(scores.items(), key=lambda it: it[1], reverse=True)
    best_candidate, best_val = sorted_scores[0]
    # The two log primaries (Yggdrasil/Asvata) share one trunk-evidence
    # score and differ only by the region nudge, so the runner-up must not
    # be the winner's own twin: a 0.06 "margin" against Asvata would report
    # a clear Yggdrasil patch as low-confidence. The margin is measured
    # against the best genuinely different rival.
    _LOG_TWINS = {"Yggdrasil Log": "Asvata Log", "Asvata Log": "Yggdrasil Log"}
    _twin = _LOG_TWINS.get(best_candidate)
    _rivals = [v for k, v in scores.items() if k != best_candidate and k != _twin]
    second_val = max(_rivals) if _rivals else 0.0

    margin = best_val - second_val
    # F-1082: confidence is calibrated from the evidence (margin + best
    # score) with NO floor and NO 0.74 base — a weak/ambiguous patch must
    # come out with a low number, not a confident-looking 0.74+.
    confidence = min(
        0.95,
        0.38 + min(0.35, margin * 0.45) + min(0.25, best_val * 0.12),
    )

    # F-1083: no signal -> "unknown", never a confident Orichalcum Ore.
    # A patch whose best score stays below the evidence threshold carries no
    # material claim at all; callers must check `unknown` / the low
    # confidence instead of receiving a fabricated ore.
    if best_val < 0.20:
        return {
            "chosen_name": None,
            "confidence": round(float(min(0.49, 0.10 + best_val * 0.5)), 3),
            "scores": scores,
            "unknown": True,
            "taxonomy": {
                "category": "Unknown",
                "category_tr": "Bilinmiyor",
                "profession": "Unknown",
                "harvest_type": "Unknown",
                "candidate": None,
            },
        }

    category = "Ores & Metals"
    profession = "Blacksmithing / Armorsmithing"
    harvest_type = "Vitality Extraction"
    for cat_name, cat_data in AION2_TAXONOMY_HIERARCHY.items():
        if best_candidate in cat_data.get("primary_materials", {}):
            category = cat_name
            profession = cat_data.get("profession", profession)
            harvest_type = cat_data.get("harvest_type", harvest_type)
            break

    return {
        "chosen_name": best_candidate,
        "confidence": round(float(confidence), 3),
        "scores": scores,
        "unknown": False,
        "taxonomy": {
            "category": category,
            "category_tr": MATERIAL_CATEGORIES.get(category, category),
            "profession": profession,
            "harvest_type": harvest_type,
            "candidate": best_candidate,
        },
    }


def classify_3d_material_patch(
    mesh_bgr: np.ndarray,
    region_name: str = "",
    label_width_px: int = 0,
    aspect_ratio: Optional[float] = None,
    bbox: Optional[Tuple[int, int, int, int]] = None,
    crest_patch_bgr: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Classify a 3D world resource mesh patch below the white gather crest using real-time visual descriptors.

    Analyzes multi-channel HSV color distributions, 3D crest textures, directional gradients,
    specular highlights, and morphological aspect ratios, mapping directly to Aion 2 taxonomy:
    - `Aria`: Purple/violet crystal-petaled herb below winged gather crest (Alchemy)
    - `Mela`: Red berries + green bush foliage with squat dome aspect ratio (Cooking & Tailoring)
    - `Targena`: Golden-yellow and green herb with delicate leaf texture (Alchemy)
    - `Calendula`: Orange-gold marigold blossom herb (Alchemy)
    - `Azpha`: Blue-violet highland herb in Asmodian territories (Alchemy)
    - `Sapphire Gemstone`: Deep glowing cyan/blue crystal with faceted specularity (Handicrafting)
    - `Ruby Gemstone`: Crimson crystal spire with vertical aspect ratio and zero foliage (Handicrafting)
    - `Diamond Gemstone`: Brilliant white/ice crystal cluster (Handicrafting)
    - `Yggdrasil Log` / `Asvata Log`: Tall vertical trunk aspect ratio with directional bark grain (Woodcutting)
    - `Orichalcum Ore`: Metallic bronze/teal/gold vein on dark rock strata (Blacksmithing / Armorsmithing)
    - `Odyle`: Floating glowing teal/green energy orb in 3D world (Flight / Ground Extraction)

    Args:
        mesh_bgr: BGR image crop of the 3D resource mesh below the crest or around the [E]/[F] icon.
        region_name: Optional active region name for regional prior disambiguation.
        label_width_px: Optional width in pixels of the cyan 3D nameplate above the mesh.
        aspect_ratio: Optional bounding box aspect ratio (width / height).
        bbox: Optional (x, y, w, h) bounding box coordinates.
        crest_patch_bgr: Optional separate crop of the 3D winged gather crest or nameplate.

    Returns:
        Enriched dictionary containing canonical material names, categories, calibrated confidence,
        diagnostic color ratios, visual descriptors, texture features, aspect ratio features, and taxonomy.
    """
    # P1-4 (K-07): Orichalcum varsayılanı YOK. Katalogda karşılığı
    # olmayan bir isim "Orichalcum Ore" diye dönmez; aşağıda katalog
    # ıskalaması "Unknown" (güven 0.0) döner. Geçersiz/kanıtı olmayan
    # girdiler zaten yukarıda "Unknown" döner (F-1083/F-1084).
    # Last-resort guard only: at this point chosen_name is already a validated
    # canonical primary, so this is never reached as a classification result
    # (invalid / no-signal inputs return "Unknown" above).
    # F-1084: invalid input is "Unknown" with confidence 0.0 — never a
    # fabricated Orichalcum Ore at 0.50. Callers gate on confidence and on
    # the `unknown` flag; a None/empty/2-D array carries no material claim.
    if (
        mesh_bgr is None
        or not isinstance(mesh_bgr, np.ndarray)
        or mesh_bgr.size == 0
        or mesh_bgr.ndim != 3
        or mesh_bgr.shape[2] < 3
    ):
        return {
            "name": "Unknown",
            "name_tr": "Bilinmiyor",
            "category": "Unknown",
            "category_tr": "Bilinmiyor",
            "grade": None,
            "max_harvests": 0,
            "confidence": 0.0,
            "unknown": True,
            "chosen_name": "Unknown",
            "region_matched": False,
            "color_ratios": {},
            "visual_descriptor": MaterialVisualDescriptor().to_dict(),
            "texture_features": {},
            "aspect_ratio_features": {
                "aspect_ratio": 1.0,
                "height_width_ratio": 1.0,
                "morphological_profile": "balanced",
            },
            "taxonomy": {
                "category": "Unknown",
                "category_tr": "Bilinmiyor",
                "profession": "Unknown",
                "candidate": None,
            },
            "scores": {},
        }

    # Extract real-time visual feature descriptors (single shared BGR->HSV
    # conversion; the HSV array is reused for the refinement step below).
    desc, hsv = _extract_visual_descriptor_with_hsv(
        mesh_bgr,
        bbox_aspect_ratio=aspect_ratio,
        bbox=bbox,
        crest_bgr=crest_patch_bgr,
    )

    # Map descriptors to Aion 2 material taxonomy
    mapped = map_descriptors_to_taxonomy(
        descriptor=desc,
        region_name=region_name,
        label_width_px=label_width_px,
    )

    # No-signal patch: report "Unknown" instead of forcing a material.
    if mapped.get("unknown") or not mapped.get("chosen_name"):
        return {
            "name": "Unknown",
            "name_tr": "Bilinmiyor",
            "category": "Unknown",
            "category_tr": "Bilinmiyor",
            "grade": None,
            "max_harvests": 0,
            "confidence": round(float(mapped.get("confidence", 0.0)), 3),
            "unknown": True,
            "chosen_name": "Unknown",
            "region_matched": False,
            "color_ratios": desc.color_ratios,
            "visual_descriptor": desc.to_dict(),
            "texture_features": {
                "texture_contrast": desc.texture_contrast,
                "gradient_magnitude_mean": desc.gradient_magnitude_mean,
                "vertical_edge_ratio": desc.vertical_edge_ratio,
                "horizontal_edge_ratio": desc.horizontal_edge_ratio,
                "specular_highlight_ratio": desc.specular_highlight_ratio,
                "spot_cluster_count": desc.spot_cluster_count,
                "crest_symmetry_score": desc.crest_symmetry_score,
                "crest_brightness": desc.crest_brightness,
            },
            "aspect_ratio_features": {
                "aspect_ratio": desc.aspect_ratio,
                "height_width_ratio": desc.height_width_ratio,
                "morphological_profile": desc.morphological_profile,
            },
            "taxonomy": mapped.get("taxonomy", {}),
            "scores": mapped.get("scores", {}),
        }

    chosen_name = mapped["chosen_name"]
    confidence = float(mapped["confidence"])
    scores = mapped["scores"]
    taxonomy_info = mapped["taxonomy"]

    # Catalog essence refinement if a specific regional essence has higher match.
    # `hsv` is the shared conversion from the descriptor extraction above
    # (previously a third cv2.cvtColor per patch); it is not None here
    # because invalid input already returned above — the fallback only
    # guards against a future caller path, it is not a second conversion
    # on the normal path.
    if hsv is None:
        hsv = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2HSV)
    # F-1100: the per-name coarse_ratio branches below were dead code —
    # _more_specific_catalog_material only refines the coarse Orichalcum Ore
    # fallback, so every non-ore branch was computed and discarded. Only
    # the ore ratio is computed now.
    coarse_ratio = desc.color_ratios.get("orichalcum_vein", 0.0)

    specific_name = _more_specific_catalog_material(
        hsv,
        chosen_name=chosen_name,
        coarse_ratio=float(coarse_ratio),
        region_name=region_name,
        ore_ratio=float(desc.color_ratios.get("orichalcum_vein", 0.0)),
    )
    if specific_name:
        chosen_name = specific_name
        confidence = min(0.95, max(confidence, 0.78 + min(0.14, coarse_ratio)))

    spec = AION2_MATERIALS_CATALOG.get(chosen_name)
    if spec is None:
        # P1-4 (K-07): katalog ıskalaması Orichalcum Ore varsayılanına
        # düşmez — türsüz sonuç "Unknown" döner, güven 0.0. Normal yolda
        # erişilemez (12 kanonik primerin tamamı katalogda); katalog
        # değişirse kanıtsız materyal iddiası üretilmez.
        return {
            "name": "Unknown",
            "name_tr": "Bilinmiyor",
            "category": "Unknown",
            "category_tr": "Bilinmiyor",
            "grade": None,
            "max_harvests": 0,
            "confidence": 0.0,
            "unknown": True,
            "chosen_name": "Unknown",
            "region_matched": False,
            "color_ratios": desc.color_ratios,
            "visual_descriptor": desc.to_dict(),
            "texture_features": {
                "texture_contrast": desc.texture_contrast,
                "gradient_magnitude_mean": desc.gradient_magnitude_mean,
                "vertical_edge_ratio": desc.vertical_edge_ratio,
                "horizontal_edge_ratio": desc.horizontal_edge_ratio,
                "specular_highlight_ratio": desc.specular_highlight_ratio,
                "spot_cluster_count": desc.spot_cluster_count,
                "crest_symmetry_score": desc.crest_symmetry_score,
                "crest_brightness": desc.crest_brightness,
            },
            "aspect_ratio_features": {
                "aspect_ratio": desc.aspect_ratio,
                "height_width_ratio": desc.height_width_ratio,
                "morphological_profile": desc.morphological_profile,
            },
            "taxonomy": taxonomy_info,
            "scores": scores,
        }
    expected_in_region = True
    if region_name:
        reg_specs = get_region_expected_materials(region_name)
        expected_in_region = any(s.base_material == spec.base_material for s in reg_specs)

    return {
        "name": spec.name,
        "name_tr": spec.name_tr,
        "category": spec.category,
        "category_tr": spec.category_tr,
        "grade": spec.grade,
        "max_harvests": spec.max_harvests,
        "confidence": round(float(confidence), 3),
        "unknown": False,
        "chosen_name": spec.name,
        "region_matched": bool(expected_in_region),
        "color_ratios": desc.color_ratios,
        "visual_descriptor": desc.to_dict(),
        "texture_features": {
            "texture_contrast": desc.texture_contrast,
            "gradient_magnitude_mean": desc.gradient_magnitude_mean,
            "vertical_edge_ratio": desc.vertical_edge_ratio,
            "horizontal_edge_ratio": desc.horizontal_edge_ratio,
            "specular_highlight_ratio": desc.specular_highlight_ratio,
            "spot_cluster_count": desc.spot_cluster_count,
            "crest_symmetry_score": desc.crest_symmetry_score,
            "crest_brightness": desc.crest_brightness,
        },
        "aspect_ratio_features": {
            "aspect_ratio": desc.aspect_ratio,
            "height_width_ratio": desc.height_width_ratio,
            "morphological_profile": desc.morphological_profile,
        },
        "taxonomy": taxonomy_info,
        "scores": scores,
    }


def get_materials_db_summary() -> dict:
    """Return summary dictionary for the materials database (real counts,
    not a hardcoded ready flag)."""
    contract_ok = validate_materials_db_contract()
    return {
        "materials_db_ready": bool(contract_ok),
        "material_count": len(AION2_MATERIALS_CATALOG),
        "category_count": len(MATERIAL_CATEGORIES),
        "taxonomy_primary_count": len(CANONICAL_PRIMARY_MATERIALS),
    }


def get_materials_db_health_summary() -> dict:
    """Return health summary for the materials_db module (grounded in the
    real contract check, not a constant True)."""
    return {
        "module": "materials_db",
        "ready": bool(validate_materials_db_contract()),
        "material_count": len(AION2_MATERIALS_CATALOG),
    }


def get_materials_db_ready_flag() -> dict:
    """Return ready flag for the materials_db module (grounded in the real
    contract check)."""
    return {"module": "materials_db", "ready": bool(validate_materials_db_contract())}


__all__ = [
    "DEFAULT_MAX_HARVESTS",
    "ARIA_HSV_RANGE",
    "CANONICAL_PRIMARY_MATERIALS",
    "MATERIAL_CATEGORIES",
    "MATERIAL_GRADES",
    "OFFICIAL_REGIONS",
    "REJECTED_NON_MATERIAL_KEYWORDS",
    "MaterialSpec",
    "MaterialCatalog",
    "AION2_MATERIALS_CATALOG",
    "AION2_MATERIALS_BY_NAME",
    "get_material_spec",
    "get_region_expected_materials",
    "is_rejected_non_material",
    "MaterialVisualDescriptor",
    "compute_hsv_color_distribution",
    "compute_3d_crest_texture",
    "compute_aspect_ratio_features",
    "extract_visual_descriptor",
    "AION2_TAXONOMY_HIERARCHY",
    "map_descriptors_to_taxonomy",
    "classify_3d_material_patch",
    "get_materials_db_summary",
    "get_materials_db_health_summary",
    "get_materials_db_ready_flag",
    "get_materials_db_version_info",
    "get_materials_db_diagnostic_snapshot",
    "get_materials_db_capabilities",
    "validate_materials_db_contract",
    "get_materials_db_baseline_metrics",
    "get_materials_db_readiness_status",
    "get_materials_db_audit_record",
    "get_materials_db_operational_digest",
    "get_materials_db_telemetry_payload",
    "get_materials_db_integrity_seal",
    "get_materials_db_compliance_certificate",
    "get_materials_db_readiness_probe",
    "get_materials_db_audit_summary",
]


def get_materials_db_version_info() -> dict:
    """Return version information for the materials_db module."""
    return {"module": "materials_db", "version": __version__, "ready": True}


def get_materials_db_diagnostic_snapshot() -> dict:
    """Return diagnostic snapshot for the materials_db module."""
    return {
        "module": "materials_db",
        "status": "ok" if validate_materials_db_contract() else "failed",
        "ready": bool(validate_materials_db_contract()),
        "material_count": len(AION2_MATERIALS_CATALOG),
    }


def get_materials_db_capabilities() -> list:
    """Return capabilities list for the materials_db module."""
    return ["materials_db", "gatherable_catalog", "diagnostic_snapshot"]


def validate_materials_db_contract() -> bool:
    """Validate the contract for the materials_db module against the real
    catalog (F-1114): the old version only checked that a hardcoded
    capability list was non-empty, i.e. it validated itself. This checks
    that the catalog is non-empty, every spec's category is a known
    category, every canonical primary exists in the catalog, every
    grades_available entry is a real grade, and every HSV range is
    well-formed (lower <= upper per channel, values in OpenCV HSV range).
    """
    try:
        if len(AION2_MATERIALS_CATALOG) == 0:
            return False
        for spec in AION2_MATERIALS_CATALOG:
            if spec.category not in MATERIAL_CATEGORIES:
                return False
            if not spec.name or not spec.base_material:
                return False
            if any(g not in MATERIAL_GRADES for g in spec.grades_available):
                return False
            for ranges in (spec.hsv_ranges, spec.secondary_hsv_ranges):
                for lower, upper in ranges:
                    if len(lower) != 3 or len(upper) != 3:
                        return False
                    for lo, hi in zip(lower, upper):
                        if not (0 <= lo <= hi <= 255):
                            return False
        for primary in CANONICAL_PRIMARY_MATERIALS:
            if AION2_MATERIALS_CATALOG.get(primary) is None:
                return False
        for cat_data in AION2_TAXONOMY_HIERARCHY.values():
            for primary in cat_data.get("primary_materials", {}):
                if AION2_MATERIALS_CATALOG.get(primary) is None:
                    return False
        return True
    except Exception:
        return False


def get_materials_db_baseline_metrics(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return baseline metrics for the materials_db module.

    `_contract_ok` is an internal pass-through: a caller that already
    validated the contract passes the result down instead of every nested
    getter re-scanning the whole catalog.
    """
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "contract_valid": ok,
        "capability_count": len(get_materials_db_capabilities()),
    }


def get_materials_db_readiness_status(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return readiness status for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "ready": ok,
        "metrics": get_materials_db_baseline_metrics(_contract_ok=ok),
    }


def get_materials_db_audit_record(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return audit record for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "audit_passed": ok,
        "readiness": get_materials_db_readiness_status(_contract_ok=ok),
    }


def get_materials_db_operational_digest(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return operational digest for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "operational": ok,
        "audit": get_materials_db_audit_record(_contract_ok=ok),
    }


def get_materials_db_telemetry_payload(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return telemetry payload for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "payload": get_materials_db_operational_digest(_contract_ok=ok),
    }


def get_materials_db_integrity_seal(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return integrity seal for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "sealed": ok,
        "telemetry": get_materials_db_telemetry_payload(_contract_ok=ok),
    }


def get_materials_db_compliance_certificate(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return compliance certificate for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "certified": ok,
        "seal": get_materials_db_integrity_seal(_contract_ok=ok),
    }


def get_materials_db_readiness_probe(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return readiness probe for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "ready": ok,
        "certificate": get_materials_db_compliance_certificate(_contract_ok=ok),
    }


def get_materials_db_audit_summary(
    *, _contract_ok: Optional[bool] = None
) -> dict:
    """Return audit summary for the materials_db module."""
    ok = validate_materials_db_contract() if _contract_ok is None else bool(_contract_ok)
    return {
        "module": "materials_db",
        "audit_passed": ok,
        "probe": get_materials_db_readiness_probe(_contract_ok=ok),
    }


