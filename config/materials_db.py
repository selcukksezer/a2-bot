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
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

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
        if isinstance(other, MaterialSpec):
            return self.name.lower() == other.name.lower()
        if isinstance(other, str):
            norm = other.strip().lower()
            return norm == self.name.lower() or norm == self.name_tr.lower()
        return False

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
    aria_hsv: Tuple[HSVRange, ...] = (((125, 70, 90), (165, 255, 255)),)
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

    targena_hsv: Tuple[HSVRange, ...] = (((22, 90, 120), (45, 255, 255)),)
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

    azpha_hsv: Tuple[HSVRange, ...] = (((105, 70, 90), (135, 255, 255)),)
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

    calendula_hsv: Tuple[HSVRange, ...] = (((10, 120, 140), (24, 255, 255)),)
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
    orichalcum_hsv: Tuple[HSVRange, ...] = (
        ((6, 65, 55), (26, 255, 245)),
        ((72, 45, 70), (98, 210, 225)),
    )
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
    sapphire_hsv: Tuple[HSVRange, ...] = (((95, 120, 140), (125, 255, 255)),)
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

    ruby_hsv: Tuple[HSVRange, ...] = (
        ((0, 140, 140), (10, 255, 255)),
        ((168, 140, 140), (180, 255, 255)),
    )
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

    diamond_hsv: Tuple[HSVRange, ...] = (((85, 10, 200), (115, 65, 255)),)
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
        ("Fine", "Diamond Gemstone (Fine)" if False else "Fine Diamond Gemstone", "Kaliteli Elmas Değerli Taş"),
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
    log_hsv: Tuple[HSVRange, ...] = (((8, 45, 35), (22, 165, 145)),)
    log_canopy_hsv: Tuple[HSVRange, ...] = (((35, 55, 45), (78, 255, 215)),)

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
                hsv_ranges=log_hsv,
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
    for log_name, log_tr, log_regions in extended_logs:
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
                hsv_ranges=log_hsv,
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
    mela_hsv: Tuple[HSVRange, ...] = (
        ((0, 100, 85), (12, 255, 255)),
        ((168, 100, 85), (180, 255, 255)),
    )
    mela_leaf_hsv: Tuple[HSVRange, ...] = (((35, 50, 40), (75, 255, 235)),)
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
    odyle_hsv: Tuple[HSVRange, ...] = (((58, 105, 145), (94, 255, 255)),)
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

    return MaterialCatalog(specs)


AION2_MATERIALS_CATALOG: MaterialCatalog = _build_materials_catalog()
AION2_MATERIALS_BY_NAME: Dict[str, MaterialSpec] = AION2_MATERIALS_CATALOG.to_dict()


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
    """
    if not name_or_kind or not isinstance(name_or_kind, str):
        return False

    norm = " ".join(name_or_kind.strip().lower().split())
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

    for token in REJECTED_NON_MATERIAL_KEYWORDS:
        if token in norm:
            return True

    return False


def _normalize_region_token(text: str) -> str:
    """Normalize region string by stripping apostrophes and extra whitespace."""
    return " ".join(
        text.strip().lower().replace("'", "").replace("’", "").replace("_", " ").split()
    )


def get_region_expected_materials(region_name: str) -> List[MaterialSpec]:
    """Return the known gatherable MaterialSpec entries for a given Aion 2 / Aion region.

    Supports partial/case-insensitive matching for region names such as
    "Pilgrim's Ridge", "Western Altamia Highland", "Verteron", "Altgard", etc.
    """
    if not region_name or not isinstance(region_name, str):
        return list(AION2_MATERIALS_CATALOG)

    target = _normalize_region_token(region_name)
    if not target:
        return list(AION2_MATERIALS_CATALOG)

    matched: List[MaterialSpec] = []
    for spec in AION2_MATERIALS_CATALOG:
        for reg in spec.regions:
            reg_low = _normalize_region_token(reg)
            if target == reg_low or target in reg_low or reg_low in target:
                matched.append(spec)
                break

    return matched


def _mask_ratio(hsv: np.ndarray, lower: HSVBound, upper: HSVBound) -> float:
    """Compute the fraction of pixels in `hsv` falling within `[lower, upper]`."""
    total_px = float(max(1, hsv.shape[0] * hsv.shape[1]))
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return float(np.count_nonzero(mask)) / total_px


def _mask_count(hsv: np.ndarray, lower: HSVBound, upper: HSVBound) -> int:
    """Count pixels in `hsv` falling within `[lower, upper]`."""
    mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    return int(np.count_nonzero(mask))


def classify_3d_material_patch(
    mesh_bgr: np.ndarray,
    region_name: str = "",
    label_width_px: int = 0,
) -> Dict[str, Any]:
    """Classify a 3D world plant/crystal/ore/tree/orb mesh patch below the white gather crest.

    Analyzes HSV color distributions of the 3D mesh patch:
    - Purple/violet crystal foliage (H=125..165, S=70..255, V=90..255) -> `Aria`
    - Red berries (H=0..12 / 168..180) + green bush foliage (H=26..80) -> `Mela`
    - Deep glowing blue/cyan crystal (H=95..125, S=120..255, V=140..255) -> `Sapphire Gemstone`
    - Crimson/red glowing crystal (H=0..10 / 168..180, S=140..255, V=140..255, low green) -> `Ruby Gemstone`
    - Brilliant white/ice crystal (H=85..115, S=10..65, V=200..255) -> `Diamond Gemstone`
    - Metallic bronze/teal/gold rock vein -> `Orichalcum Ore`
    - Golden-yellow/green herb (H=22..45, S=90..255, V=120..255) -> `Targena`
    - Orange-gold marigold herb (H=10..24, S=120..255, V=140..255) -> `Calendula`
    - Blue-violet herb in Asmodian regions (H=105..135, S=70..255, V=90..255) -> `Azpha`
    - Brown bark trunk + green canopy -> `Yggdrasil Log` / `Asvata Log`
    - Glowing green/teal orb (H=58..94, S=105..255, V=145..255) -> `Odyle`

    Args:
        mesh_bgr: BGR image crop of the 3D resource mesh below the crest or around the [E]/[F] icon.
        region_name: Optional active region name for regional prior disambiguation.
        label_width_px: Optional width in pixels of the cyan 3D nameplate above the mesh.

    Returns:
        Dictionary with keys:
        - `name`: Canonical English material name (e.g. "Aria", "Mela", "Targena", "Orichalcum Ore")
        - `name_tr`: Turkish material name
        - `category`: English category name
        - `category_tr`: Turkish category name
        - `max_harvests`: Default harvest capacity (3)
        - `confidence`: Classification confidence in [0.0, 0.99]
        - `color_ratios`: Dict of diagnostic HSV color ratios
    """
    default_spec = AION2_MATERIALS_CATALOG["Orichalcum Ore"]
    if (
        mesh_bgr is None
        or not isinstance(mesh_bgr, np.ndarray)
        or mesh_bgr.size == 0
        or mesh_bgr.ndim != 3
        or mesh_bgr.shape[2] < 3
    ):
        return {
            "name": default_spec.name,
            "name_tr": default_spec.name_tr,
            "category": default_spec.category,
            "category_tr": default_spec.category_tr,
            "max_harvests": DEFAULT_MAX_HARVESTS,
            "confidence": 0.50,
            "color_ratios": {},
        }

    hsv = cv2.cvtColor(mesh_bgr, cv2.COLOR_BGR2HSV)
    total_px = float(max(1, hsv.shape[0] * hsv.shape[1]))

    # 1. Aria: Purple/violet crystal plant (H=125..165, S=65..255, V=75..255)
    aria_px = _mask_count(hsv, (125, 65, 75), (165, 255, 255))
    aria_ratio = aria_px / total_px

    # 2. Red berries / Ruby crystal
    red_berry_px = _mask_count(hsv, (0, 100, 85), (12, 255, 255)) + _mask_count(
        hsv, (168, 100, 85), (180, 255, 255)
    )
    red_berry_ratio = red_berry_px / total_px

    ruby_crystal_px = _mask_count(hsv, (0, 140, 140), (10, 255, 255)) + _mask_count(
        hsv, (168, 140, 140), (180, 255, 255)
    )
    ruby_crystal_ratio = ruby_crystal_px / total_px

    # 3. Green bush leaves (for Mela and herbs)
    green_leaf_px = _mask_count(hsv, (35, 50, 40), (78, 255, 240))
    green_broad_px = _mask_count(hsv, (26, 40, 35), (82, 255, 245))
    green_leaf_ratio = green_leaf_px / total_px

    # 4. Golden-yellow/green herb (Targena: H=22..45, S=65..255, V=65..255)
    targena_px = _mask_count(hsv, (22, 65, 65), (45, 255, 255))
    targena_ratio = targena_px / total_px

    # 5. Orange-gold marigold herb (Calendula: H=10..24, S=120..255, V=140..255)
    calendula_px = _mask_count(hsv, (10, 120, 140), (24, 255, 255))
    calendula_ratio = calendula_px / total_px

    # 6. Sapphire Gemstone (Deep glowing blue/cyan crystal: H=95..125, S=120..255, V=140..255)
    sapphire_px = _mask_count(hsv, (95, 120, 140), (125, 255, 255))
    sapphire_ratio = sapphire_px / total_px

    # 7. Azpha (Blue-violet herb: H=105..135, S=70..255, V=90..255)
    azpha_px = _mask_count(hsv, (105, 70, 90), (135, 255, 255))
    azpha_ratio = azpha_px / total_px

    # 8. Diamond Gemstone (Brilliant white/ice crystal: H=85..115, S=10..65, V=200..255)
    diamond_px = _mask_count(hsv, (85, 10, 200), (115, 65, 255))
    diamond_ratio = diamond_px / total_px

    # 9. Glowing green/teal orb (Odyle: H=58..94, S=105..255, V=145..255)
    odyle_px = _mask_count(hsv, (58, 105, 145), (94, 255, 255))
    odyle_ratio = odyle_px / total_px

    # 10. Metallic bronze/teal/gold rock vein (Orichalcum Ore)
    bronze_ore_px = _mask_count(hsv, (6, 65, 55), (26, 230, 225))
    teal_ore_px = _mask_count(hsv, (72, 45, 65), (98, 210, 225))
    dark_rock_px = _mask_count(hsv, (0, 0, 20), (180, 65, 115))
    ore_ratio = (bronze_ore_px + teal_ore_px) / total_px
    dark_rock_ratio = dark_rock_px / total_px

    # 11. Brown bark trunk (Yggdrasil / Asvata Log: H=8..22, S=45..165, V=35..140)
    bark_px = _mask_count(hsv, (8, 45, 35), (22, 165, 140))
    bark_ratio = bark_px / total_px

    color_ratios: Dict[str, float] = {
        "aria_purple": round(aria_ratio, 4),
        "red_berry": round(red_berry_ratio, 4),
        "ruby_crystal": round(ruby_crystal_ratio, 4),
        "green_leaf": round(green_leaf_ratio, 4),
        "targena_gold_green": round(targena_ratio, 4),
        "calendula_orange": round(calendula_ratio, 4),
        "sapphire_blue": round(sapphire_ratio, 4),
        "azpha_blue_violet": round(azpha_ratio, 4),
        "diamond_ice": round(diamond_ratio, 4),
        "odyle_orb": round(odyle_ratio, 4),
        "orichalcum_vein": round(ore_ratio, 4),
        "bark_trunk": round(bark_ratio, 4),
    }

    reg_low = (region_name or "").strip().lower()
    is_asmodian_region = any(
        r in reg_low for r in ("altgard", "morheim", "ishalgen", "beluslan", "gelkmaros")
    )

    chosen_name = "Orichalcum Ore"
    confidence = 0.72

    # Priority 1: Aria (distinctive purple/violet crystal petals)
    if aria_px >= 90 and aria_ratio >= 0.025 and aria_px > red_berry_px * 1.5:
        chosen_name = "Aria"
        confidence = min(0.98, 0.84 + min(0.14, aria_ratio * 1.8))

    # Priority 2: Mela (red berries + green/olive bush leaves)
    elif red_berry_px >= 100 and red_berry_ratio >= 0.04 and (green_leaf_px >= 25 or green_broad_px >= 60):
        chosen_name = "Mela"
        confidence = min(0.98, 0.85 + min(0.13, red_berry_ratio * 0.4))

    # Priority 3: Sapphire Gemstone (deep glowing blue/cyan crystal)
    elif sapphire_px >= 80 and sapphire_ratio >= 0.035 and sapphire_px >= aria_px:
        if is_asmodian_region and azpha_px > sapphire_px * 1.25 and green_leaf_ratio > 0.08:
            chosen_name = "Azpha"
            confidence = min(0.94, 0.80 + min(0.14, azpha_ratio * 1.2))
        else:
            chosen_name = "Sapphire Gemstone"
            confidence = min(0.97, 0.84 + min(0.13, sapphire_ratio * 1.5))

    # Priority 4: Ruby Gemstone (crimson/red glowing crystal with negligible green foliage)
    elif ruby_crystal_px >= 80 and ruby_crystal_ratio >= 0.035 and green_leaf_ratio < 0.015:
        chosen_name = "Ruby Gemstone"
        confidence = min(0.96, 0.83 + min(0.13, ruby_crystal_ratio * 1.4))

    # Priority 5: Diamond Gemstone (brilliant white/ice crystal cluster)
    elif diamond_px >= 110 and diamond_ratio >= 0.06 and green_leaf_ratio < 0.08:
        chosen_name = "Diamond Gemstone"
        confidence = min(0.95, 0.82 + min(0.13, diamond_ratio * 1.2))

    # Priority 6: Odyle (floating glowing green/teal energy orb)
    elif odyle_px >= 90 and odyle_ratio >= 0.05 and targena_ratio < 0.15:
        chosen_name = "Odyle"
        confidence = min(0.96, 0.83 + min(0.13, odyle_ratio * 1.3))

    # Priority 7: Targena / Calendula (golden-yellow & lush green herb patch, e.g. Pilgrim's Ridge [E] plant)
    elif (
        (targena_px >= 100 and targena_ratio >= 0.05)
        or (green_leaf_ratio >= 0.25 and targena_ratio >= 0.01)
        or (green_leaf_ratio >= 0.45 and bark_ratio < 0.15)
    ):
        if calendula_ratio > targena_ratio * 1.25 and calendula_px >= 100:
            chosen_name = "Calendula"
            confidence = min(0.95, 0.82 + min(0.13, calendula_ratio * 1.2))
        else:
            chosen_name = "Targena"
            confidence = min(0.97, 0.84 + min(0.13, (targena_ratio + green_leaf_ratio) * 0.12))

    # Priority 8: Calendula (vivid orange-gold marigold blossom)
    elif calendula_px >= 95 and calendula_ratio >= 0.045 and green_leaf_ratio >= 0.03:
        chosen_name = "Calendula"
        confidence = min(0.94, 0.81 + min(0.13, calendula_ratio * 1.3))

    # Priority 9: Azpha (blue-violet herb)
    elif azpha_px >= 85 and azpha_ratio >= 0.035:
        chosen_name = "Azpha"
        confidence = min(0.93, 0.80 + min(0.13, azpha_ratio * 1.3))

    # Priority 10: Yggdrasil Log / Asvata Log (wooden bark trunk with canopy foliage and long label)
    elif bark_ratio >= 0.18 and green_leaf_ratio >= 0.08 and dark_rock_ratio < 0.35:
        chosen_name = "Asvata Log" if is_asmodian_region else "Yggdrasil Log"
        confidence = min(0.92, 0.78 + min(0.14, bark_ratio * 0.4))

    # Priority 11: Orichalcum Ore (metallic bronze/teal/gold vein on rock)
    elif ore_ratio >= 0.015 or dark_rock_ratio >= 0.20 or label_width_px >= 75:
        chosen_name = "Orichalcum Ore"
        confidence = min(0.95, 0.80 + min(0.15, ore_ratio * 1.2 + dark_rock_ratio * 0.2))

    spec = AION2_MATERIALS_CATALOG.get(chosen_name, default_spec) or default_spec
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
        "region_matched": bool(expected_in_region),
        "color_ratios": color_ratios,
    }


def get_materials_db_summary() -> dict:
    """Return summary dictionary for the materials database."""
    try:
        return {"materials_db_ready": True}
    except Exception:
        return {"materials_db_ready": False}


def get_materials_db_health_summary() -> dict:
    """Return health summary for the materials_db module."""
    return {"module": "materials_db", "ready": True}


def get_materials_db_ready_flag() -> dict:
    """Return ready flag for the materials_db module."""
    return {"module": "materials_db", "ready": True}


__all__ = [
    "DEFAULT_MAX_HARVESTS",
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
    "classify_3d_material_patch",
    "get_materials_db_summary",
    "get_materials_db_health_summary",
    "get_materials_db_ready_flag",
    "get_materials_db_version_info",
]


def get_materials_db_version_info() -> dict:
    """Return version information for the materials_db module."""
    return {"module": "materials_db", "version": "2.0.0", "ready": True}

