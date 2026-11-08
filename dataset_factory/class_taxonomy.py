# -*- coding: utf-8 -*-
"""dataset_factory/class_taxonomy.py
===================================
Canonical YOLO Class Taxonomy, Validation & Sanitization for Aion 2.

Defines and strictly enforces the canonical 22 classes (0..21) plus consumable
classes (22..24: hp_potion, mp_potion, buff_scroll).

Strictly blocks non-canonical classes (such as class_id: 98, essence_*, or legacy
omni IDs) from entering YOLO training `.txt` files or review `.json` files.
Provides super-category helpers for 3D world materials, minimap symbols, HUD elements,
navigable elements, and consumables.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from config.materials_db import AION2_MATERIALS_CATALOG

# =============================================================================
# Canonical Class Definitions (0..21 Core + 22..24 Consumables)
# =============================================================================

CORE_22_CANONICAL_NAMES: Dict[int, str] = {
    0: "herb_aria",
    1: "herb_mela",
    2: "herb_targena",
    3: "ore_orichalcum",
    4: "gem_crystal",
    5: "tree_log",
    6: "herb_edana",
    7: "herb_lumina",
    8: "ore_obsidian",
    9: "minimap_sprout",
    10: "monster_3d",
    11: "minimap_monster_dot",
    12: "crystal_blood",
    13: "skill_ui_prompt",
    14: "obstacle",
    15: "path_ground",
    16: "player_hp_bar",
    17: "player_mp_bar",
    18: "target_hp_bar",
    19: "skill_bar",
    20: "skill_slot_ready",
    21: "skill_slot_cooldown",
}

CONSUMABLE_CANONICAL_NAMES: Dict[int, str] = {
    22: "hp_potion",
    23: "mp_potion",
    24: "buff_scroll",
}

CURRENCY_CANONICAL_NAMES: Dict[int, str] = {
    25: "kinah",
}

PLAYER_CANONICAL_NAMES: Dict[int, str] = {
    26: "player_character",
}

CANONICAL_NAMES: Dict[int, str] = {
    **CORE_22_CANONICAL_NAMES,
    **CONSUMABLE_CANONICAL_NAMES,
    **CURRENCY_CANONICAL_NAMES,
    **PLAYER_CANONICAL_NAMES,
}

CANONICAL_NAME_TO_ID: Dict[str, int] = {name: cid for cid, name in CANONICAL_NAMES.items()}

# Backward-compatibility alias for STABLE_NAMES
STABLE_NAMES: Dict[int, str] = CANONICAL_NAMES

CANONICAL_LABELS_TR: Dict[int, str] = {
    0: "Aria (Mor Kristal Bitki)",
    1: "Mela (Kırmızı Meyveli Çalı)",
    2: "Targena (Azpha/Calendula)",
    3: "Orichalcum (Maden Cevheri)",
    4: "Gem/Kristal (Safir/Yakut/Elmas)",
    5: "Ağaç/Kereste (Yggdrasil/Asvata)",
    6: "Edana (Mor/Pembe Bitki)",
    7: "Lumina (Sarı/Altın Parıltılı Bitki)",
    8: "Obsidyen (Koyu Kristal Maden) / Kombo İkonu (Q/E)",
    9: "Minimap Filiz (3 Yapraklı Yeşil)",
    10: "3B Canavar",
    11: "Minimap Canavar Noktası",
    12: "Kan Kristali (Kırmızı Kristal)",
    13: "Skill/Arayüz İstemi",
    14: "Engel",
    15: "Yol/Zemin",
    16: "Oyuncu Can Barı (HP)",
    17: "Oyuncu Mana Barı (MP)",
    18: "Hedef Can Barı",
    19: "Skill Barı",
    20: "Hazır Skill Yuvası",
    21: "Beklemede Skill Yuvası (CD)",
    22: "Can İksiri",
    23: "Mana İksiri",
    24: "Parşömen",
    25: "Kinah (Altın / Para)",
    26: "Karakterim (Oyuncu Karakteri)",
}

CANONICAL_SHORT_LABELS_TR: Dict[int, str] = {
    0: "Aria",
    1: "Mela",
    2: "Targena",
    3: "Orichalcum",
    4: "Gem/Kristal",
    5: "Ağaç/Kereste",
    6: "Edana",
    7: "Lumina",
    8: "Obsidyen",
    9: "Minimap Filiz",
    10: "Canavar",
    11: "Minimap Canavar",
    12: "Kan Kristali",
    13: "Skill/Arayüz",
    14: "Engel",
    15: "Yol/Zemin",
    16: "Oyuncu HP",
    17: "Oyuncu MP",
    18: "Hedef HP",
    19: "Skill Barı",
    20: "Skill Hazır",
    21: "Skill CD",
    22: "Can İksiri",
    23: "Mana İksiri",
    24: "Parşömen",
    25: "Kinah",
    26: "Karakterim",
}

# Backward-compatibility alias for STABLE_LABELS_TR
STABLE_LABELS_TR: Dict[int, str] = CANONICAL_LABELS_TR

# Palette colors for UI bounding boxes and charts
CANONICAL_PALETTE_HEX: Dict[int, str] = {
    0: "#ec4899",   # herb_aria: magenta/pink
    1: "#ef4444",   # herb_mela: coral red
    2: "#10b981",   # herb_targena: emerald green
    3: "#f59e0b",   # ore_orichalcum: amber/orange
    4: "#06b6d4",   # gem_crystal: cyan/sapphire
    5: "#8b5cf6",   # tree_log: violet/bronze
    6: "#a855f7",   # herb_edana: purple
    7: "#eab308",   # herb_lumina: golden yellow
    8: "#6366f1",   # ore_obsidian: dark indigo/obsidian
    9: "#22c55e",   # minimap_sprout: bright lime green
    10: "#dc2626",  # monster_3d: deep crimson
    11: "#f43f5e",  # minimap_monster_dot: hot red-pink
    12: "#991b1b",  # crystal_blood: dark ruby red
    13: "#c084fc",  # skill_ui_prompt: soft purple
    14: "#fb7185",  # obstacle: rose/salmon
    15: "#facc15",  # path_ground: sand gold
    16: "#4ade80",  # player_hp_bar: health bright green
    17: "#38bdf8",  # player_mp_bar: mana sky blue
    18: "#fb923c",  # target_hp_bar: target orange
    19: "#2dd4bf",  # skill_bar: teal
    20: "#34d399",  # skill_slot_ready: electric green
    21: "#94a3b8",  # skill_slot_cooldown: slate gray
    22: "#f87171",  # hp_potion: vibrant potion red
    23: "#60a5fa",  # mp_potion: vibrant potion blue
    24: "#fbbf24",  # buff_scroll: amber scroll parchment
    25: "#eab308",  # kinah: shiny gold / currency
    26: "#00d2ff",  # player_character: vibrant electric cyan
}

CANONICAL_PALETTE_BGR: Dict[int, Tuple[int, int, int]] = {
    cid: (
        int(hex_c[5:7], 16),
        int(hex_c[3:5], 16),
        int(hex_c[1:3], 16),
    )
    for cid, hex_c in CANONICAL_PALETTE_HEX.items()
}

# =============================================================================
# Super-Category Definitions
# =============================================================================

MATERIAL_3D_CLASSES: FrozenSet[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 12})
MINIMAP_SYMBOL_CLASSES: FrozenSet[int] = frozenset({9, 11})
MONSTER_CLASSES: FrozenSet[int] = frozenset({10})
NAVIGABLE_ELEMENT_CLASSES: FrozenSet[int] = frozenset({14, 15})
HUD_VITALS_CLASSES: FrozenSet[int] = frozenset({16, 17, 18})
HUD_SKILL_CLASSES: FrozenSet[int] = frozenset({13, 19, 20, 21})
CONSUMABLE_CLASSES: FrozenSet[int] = frozenset({22, 23, 24})
CURRENCY_CLASSES: FrozenSet[int] = frozenset({25})
PLAYER_CLASSES: FrozenSet[int] = frozenset({26})
HUD_ELEMENT_CLASSES: FrozenSet[int] = frozenset({13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25})

CORE_22_CLASSES: FrozenSet[int] = frozenset(range(22))
ALL_CANONICAL_CLASSES: FrozenSet[int] = frozenset(range(27))

# =============================================================================
# Internal Caches and Build System
# =============================================================================

_NAMES: Dict[int, str] = {}
_LABELS_TR: Dict[int, str] = {}
_ALIASES: Dict[str, int] = {}
_RESOURCE_NAMES: Dict[int, str] = {}
_READY = False


def _slug(text: str) -> str:
    """Produce a normalized lowercase ASCII slug from text."""
    ascii_text = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_text.lower()).strip("_")
    return slug or "unknown"


def _add_alias(text: str, class_id: int) -> None:
    """Register normalized variations of text pointing to class_id."""
    raw = str(text or "").strip().lower()
    if not raw:
        return
    cleaned = " ".join(raw.split())
    slugged = _slug(cleaned)
    cid = int(class_id)
    _ALIASES[cleaned] = cid
    if slugged and slugged not in _ALIASES:
        _ALIASES[slugged] = cid


def _build_taxonomy() -> None:
    global _READY
    if _READY:
        return

    _NAMES.update(CANONICAL_NAMES)
    _LABELS_TR.update(CANONICAL_LABELS_TR)

    # 1. Register canonical names and labels
    for cid, name in CANONICAL_NAMES.items():
        _add_alias(name, cid)
        _add_alias(_LABELS_TR.get(cid, ""), cid)
        _add_alias(CANONICAL_SHORT_LABELS_TR.get(cid, ""), cid)

    # 2. Resource display names
    _RESOURCE_NAMES.update(
        {
            0: "Aria",
            1: "Mela",
            2: "Targena",
            3: "Orichalcum Ore",
            4: "Gem / Crystal",
            5: "Tree Log",
            6: "Edana",
            7: "Lumina",
            8: "Obsidian Ore",
            9: "Minimap Sprout",
            10: "Monster",
            11: "Minimap Monster Dot",
            12: "Blood Crystal",
            13: "Skill UI Prompt",
            14: "Obstacle",
            15: "Path Ground",
            16: "Player HP Bar",
            17: "Player MP Bar",
            18: "Target HP Bar",
            19: "Skill Bar",
            20: "Ready Skill Slot",
            21: "Cooldown Skill Slot",
            22: "HP Potion",
            23: "MP Potion",
            24: "Buff Scroll",
            25: "Kinah",
            26: "Player Character",
        }
    )

    # 3. Legacy aliases from previous model layouts and labeling experiments
    legacy_mappings: Dict[str, int] = {
        # Player character
        "player_character": 26,
        "player": 26,
        "character": 26,
        "karakter": 26,
        "karakterim": 26,
        "oyuncu": 26,
        "oyuncu_karakteri": 26,
        "player_3d": 26,
        "self_character": 26,
        "avatar": 26,
        "hero": 26,
        "ben": 26,
        # Old tree names
        "wood_tree": 5,
        "log_timber": 5,
        "tree": 5,
        "log": 5,
        "timber": 5,
        "wood": 5,
        "tree_log": 5,
        # Old monster names
        "monster_3d_negative": 10,
        "monster_3d": 10,
        "monster": 10,
        "mob_guardian": 10,
        "mob": 10,
        # Old minimap names
        "minimap_monster_negative": 11,
        "minimap_monster_dot": 11,
        "minimap_monster": 11,
        "minimap_sprout": 9,
        "sprout": 9,
        # Old prompt / crest names
        "world_gather_crest": 13,
        "crest": 13,
        "f_gather_prompt": 13,
        "f_gather": 13,
        "e_gather_icon": 8,
        "e_gather": 8,
        "e_floating_icon": 8,
        "combo_skill_prompt": 8,
        "combo_icon": 8,
        "q_combo_icon": 8,
        "e_combo_icon": 8,
        "kombo": 8,
        "kombo_ikonu": 8,
        "combo": 8,
        "skill_ui_prompt": 13,
        # Old generic world material
        "world_material": 12,
        "crystal_blood": 12,
        "blood_crystal": 12,
        "blood crystal": 12,
        # Old herb variants
        "herb_azpha": 2,
        "azpha": 2,
        "herb_calendula": 2,
        "calendula": 2,
        "fruit_mela": 1,
        "fresh aria": 0,
        "fresh mela": 1,
        "fresh targena": 2,
        # Old ores and gems
        "ore_iron_silver": 3,
        "gem_sapphire": 4,
        "gem_ruby": 4,
        "gem_diamond": 4,
        "crystal_node": 4,
        "obsidian": 8,
        "ore_obsidian": 8,
        "obsidian ore": 8,
        "edana": 6,
        "herb_edana": 6,
        "lumina": 7,
        "herb_lumina": 7,
        # Potions and scrolls (Consumables)
        "hp_potion": 22,
        "can iksiri": 22,
        "can_iksiri": 22,
        "health_potion": 22,
        "potion_hp": 22,
        "mp_potion": 23,
        "mana iksiri": 23,
        "mana_iksiri": 23,
        "mana_potion": 23,
        "potion_mp": 23,
        "buff_scroll": 24,
        "parşömen": 24,
        "parsomen": 24,
        "scroll": 24,
        "buff scroll": 24,
        "buff": 24,
        "scroll_buff": 24,
        # Skill slot variants
        "skill_slot": 20,
        "skill_slot_ready": 20,
        "skill_slot_cooldown": 21,
        "skill_slot_cd": 21,
        "skill_cd": 21,
        "skill_ready": 20,
        "skill_bar": 19,
        # Currency / Kinah
        "kinah": 25,
        "kina": 25,
        "kinah_display": 25,
        "kinah_ui": 25,
        "altin": 25,
        "gold": 25,
        "para": 25,
        "coin": 25,
        "coins": 25,
        "currency": 25,
    }
    for alias_text, target_cid in legacy_mappings.items():
        _add_alias(alias_text, target_cid)

    # 4. Map entire Aion 2 catalog of 140+ essence items onto the canonical classes (0..24).
    # STRICT MANDATE: NEVER assign IDs >= 25 to catalog essences!
    for spec in AION2_MATERIALS_CATALOG:
        name_lower = str(spec.name).lower()
        base_lower = str(spec.base_material).lower()
        cat = spec.category

        # Determine target canonical class ID
        if "aria" in base_lower or "aria" in name_lower:
            cid = 0
        elif "mela" in base_lower or "mela" in name_lower:
            cid = 1
        elif any(k in base_lower or k in name_lower for k in ("targena", "azpha", "calendula")):
            cid = 2
        elif "edana" in base_lower or "edana" in name_lower:
            cid = 6
        elif "lumina" in base_lower or "lumina" in name_lower:
            cid = 7
        elif "obsidian" in base_lower or "obsidian" in name_lower:
            cid = 8
        elif "blood" in base_lower and "crystal" in base_lower:
            cid = 12
        elif "blood" in name_lower and "crystal" in name_lower:
            cid = 12
        elif "orichalcum" in base_lower or "orichalcum" in name_lower:
            cid = 3
        elif cat == "Herbs":
            cid = 2
        elif cat == "Ores & Metals":
            cid = 3
        elif cat == "Gemstones":
            cid = 4
        elif cat == "Logs & Trees":
            cid = 5
        elif cat == "Cooking & Fibers":
            cid = 1
        elif cat == "Odyle & Aether":
            cid = 4
        else:
            cid = 2

        # Register aliases for catalog entries
        base_slug = _slug(spec.base_material)
        name_slug = _slug(spec.name)
        _add_alias(f"essence_{base_slug}", cid)
        _add_alias(f"essence_{name_slug}", cid)
        _add_alias(spec.base_material, cid)
        _add_alias(spec.name, cid)
        _add_alias(spec.name_tr, cid)

    _READY = True


# =============================================================================
# Validation and Query Predicates
# =============================================================================

def is_valid_canonical_class_id(class_id: Any) -> bool:
    """Return True if class_id is an integer in the canonical range (0..24)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in ALL_CANONICAL_CLASSES


def is_valid_core_22_class_id(class_id: Any) -> bool:
    """Return True if class_id is an integer in the core 22 classes (0..21)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in CORE_22_CLASSES


def is_3d_world_material(class_id: Any) -> bool:
    """Return True if class_id represents a 3D gatherable world resource (0..5, 6..8, 12)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in MATERIAL_3D_CLASSES


def is_minimap_symbol(class_id: Any) -> bool:
    """Return True if class_id represents a minimap symbol (9: sprout, 11: monster dot)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in MINIMAP_SYMBOL_CLASSES


def is_monster_element(class_id: Any) -> bool:
    """Return True if class_id represents a monster (10: 3D monster, 11: minimap monster dot)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in {10, 11}


def is_navigable_element(class_id: Any) -> bool:
    """Return True if class_id represents terrain navigation (14: obstacle, 15: path_ground)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in NAVIGABLE_ELEMENT_CLASSES


def is_consumable(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name represents a consumable item (hp_potion, mp_potion, buff_scroll)."""
    if isinstance(class_id_or_name, int):
        return class_id_or_name in CONSUMABLE_CLASSES
    try:
        cid = int(class_id_or_name)
        if cid in CONSUMABLE_CLASSES:
            return True
    except (TypeError, ValueError):
        pass

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in CONSUMABLE_CLASSES
    return any(k in text for k in ("potion", "iksir", "scroll", "parşömen", "parsomen", "buff"))


def is_skill_element(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name is a skill slot or skill bar element (strictly separated from consumables)."""
    if isinstance(class_id_or_name, int):
        return class_id_or_name in HUD_SKILL_CLASSES
    try:
        cid = int(class_id_or_name)
        if cid in HUD_SKILL_CLASSES:
            return True
    except (TypeError, ValueError):
        pass

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in HUD_SKILL_CLASSES
    return any(k in text for k in ("skill", "yetenek")) and not is_consumable(text)


def is_currency(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name represents currency / Kinah (class 25)."""
    if isinstance(class_id_or_name, int):
        return class_id_or_name in CURRENCY_CLASSES
    try:
        cid = int(class_id_or_name)
        if cid in CURRENCY_CLASSES:
            return True
    except (TypeError, ValueError):
        pass

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in CURRENCY_CLASSES
    return any(k in text for k in ("kinah", "kina", "altin", "gold", "coin", "para", "currency"))


def is_hud_element(class_id: Any) -> bool:
    """Return True if class_id represents a HUD element (vitals, skills, prompts, consumables, or currency)."""
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return False
    return cid in HUD_ELEMENT_CLASSES


def is_player_character(class_id_or_name: Any) -> bool:
    """Return True if class_id represents the player character (26)."""
    if isinstance(class_id_or_name, int):
        return class_id_or_name in PLAYER_CLASSES
    try:
        cid = int(class_id_or_name)
        if cid in PLAYER_CLASSES:
            return True
    except (TypeError, ValueError):
        pass

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in PLAYER_CLASSES
    return any(k in text for k in ("player", "character", "karakter", "oyuncu"))


def get_super_category(class_id_or_name: Any) -> Optional[str]:
    """Return the canonical super-category name for a class id or class name."""
    try:
        cid = int(class_id_or_name)
    except (TypeError, ValueError):
        sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=str(class_id_or_name or ""))
        if sanitized is None:
            return None
        cid = sanitized[0]

    if cid in PLAYER_CLASSES:
        return "player"
    if cid in MATERIAL_3D_CLASSES:
        return "3d_material"
    if cid in MONSTER_CLASSES:
        return "monster"
    if cid in MINIMAP_SYMBOL_CLASSES:
        return "minimap"
    if cid in NAVIGABLE_ELEMENT_CLASSES:
        return "navigable"
    if cid in HUD_VITALS_CLASSES:
        return "hud_vitals"
    if cid in HUD_SKILL_CLASSES:
        return "hud_skill"
    if cid in CONSUMABLE_CLASSES:
        return "consumable"
    if cid in CURRENCY_CLASSES:
        return "currency"
    return None


def is_same_super_category(class_id_a: Any, class_id_b: Any) -> bool:
    """Return True if both class IDs belong to the same super-category."""
    cat_a = get_super_category(class_id_a)
    cat_b = get_super_category(class_id_b)
    return cat_a is not None and cat_a == cat_b


# =============================================================================
# Sanitization & Anti-Corruption Engine
# =============================================================================

def sanitize_class_id_and_name(
    class_id: Any,
    class_name: Optional[str] = "",
    allow_fallback: bool = False,
    default_fallback: Optional[int] = None,
) -> Tuple[int, str]:
    """Validate and sanitize a class ID and class name against canonical taxonomy.

    Resolves:
    - Canonical names and labels (e.g. 'herb_aria', 'Aria' -> (0, 'herb_aria'))
    - Legacy aliases (e.g. 'wood_tree' -> (5, 'tree_log'))
    - Catalog essence slugs (e.g. 'essence_yggdrasil_log' -> (5, 'tree_log'))
    - Legacy class index collisions (e.g. class_id: 18 with class_name: 'hp_potion' -> (22, 'hp_potion'))
    - Consumable scrolls and potions (e.g. 'buff_scroll' -> (24, 'buff_scroll'))

    Raises:
        ValueError: If neither class_id nor class_name can be resolved to a canonical
                    class (0..24) and allow_fallback is False.
    """
    _build_taxonomy()

    clean_name = str(class_name or "").strip()
    norm_name = " ".join(clean_name.lower().split()) if clean_name else ""
    slug_name = _slug(norm_name) if norm_name else ""

    # Parse raw integer class_id if possible
    try:
        raw_cid: Optional[int] = int(class_id)
    except (TypeError, ValueError):
        raw_cid = None

    # Priority 1: Exact match on canonical names
    if norm_name in CANONICAL_NAME_TO_ID:
        resolved_id = CANONICAL_NAME_TO_ID[norm_name]
        return resolved_id, CANONICAL_NAMES[resolved_id]
    if slug_name in CANONICAL_NAME_TO_ID:
        resolved_id = CANONICAL_NAME_TO_ID[slug_name]
        return resolved_id, CANONICAL_NAMES[resolved_id]

    # Priority 2: Alias dictionary lookup
    if norm_name in _ALIASES:
        resolved_id = _ALIASES[norm_name]
        return resolved_id, CANONICAL_NAMES[resolved_id]
    if slug_name in _ALIASES:
        resolved_id = _ALIASES[slug_name]
        return resolved_id, CANONICAL_NAMES[resolved_id]

    # Priority 3: Legacy collision disambiguation
    # If raw_cid was 18 or 19 from old data.yaml (where 18 was hp_potion, 19 was mp_potion)
    if raw_cid == 18 and any(k in norm_name for k in ("potion", "iksir", "hp_pot")):
        return 22, CANONICAL_NAMES[22]
    if raw_cid == 19 and any(k in norm_name for k in ("potion", "iksir", "mp_pot")):
        return 23, CANONICAL_NAMES[23]

    # Priority 4: Valid canonical class_id with no name or matching name
    if raw_cid is not None and raw_cid in CANONICAL_NAMES:
        return raw_cid, CANONICAL_NAMES[raw_cid]

    # Priority 5: Keyword-based heuristic resolution
    if norm_name:
        if "aria" in norm_name:
            return 0, CANONICAL_NAMES[0]
        if "mela" in norm_name:
            return 1, CANONICAL_NAMES[1]
        if any(k in norm_name for k in ("targena", "azpha", "calendula")):
            return 2, CANONICAL_NAMES[2]
        if "orichalcum" in norm_name:
            return 3, CANONICAL_NAMES[3]
        if "edana" in norm_name:
            return 6, CANONICAL_NAMES[6]
        if "lumina" in norm_name:
            return 7, CANONICAL_NAMES[7]
        if "obsidian" in norm_name:
            return 8, CANONICAL_NAMES[8]
        if "blood" in norm_name and "crystal" in norm_name:
            return 12, CANONICAL_NAMES[12]
        if "gem" in norm_name or "sapphire" in norm_name or "ruby" in norm_name or "diamond" in norm_name:
            return 4, CANONICAL_NAMES[4]
        if any(k in norm_name for k in ("tree", "log", "wood", "timber", "yggdrasil", "asvata")):
            return 5, CANONICAL_NAMES[5]
        if "sprout" in norm_name or "filiz" in norm_name:
            return 9, CANONICAL_NAMES[9]
        if "monster" in norm_name or "canavar" in norm_name:
            return (11, CANONICAL_NAMES[11]) if ("minimap" in norm_name or "dot" in norm_name) else (10, CANONICAL_NAMES[10])
        if "potion" in norm_name or "iksir" in norm_name:
            return (22, CANONICAL_NAMES[22]) if "hp" in norm_name or "can" in norm_name else (23, CANONICAL_NAMES[23])
        if "scroll" in norm_name or "parşömen" in norm_name or "parsomen" in norm_name:
            return 24, CANONICAL_NAMES[24]
        if "hp_bar" in norm_name or "can bar" in norm_name:
            return (18, CANONICAL_NAMES[18]) if "target" in norm_name or "hedef" in norm_name else (16, CANONICAL_NAMES[16])
        if "mp_bar" in norm_name or "mana bar" in norm_name:
            return 17, CANONICAL_NAMES[17]
        if "skill_bar" in norm_name:
            return 19, CANONICAL_NAMES[19]
        if "skill" in norm_name or "yuva" in norm_name:
            return (21, CANONICAL_NAMES[21]) if ("cd" in norm_name or "cooldown" in norm_name or "bekleme" in norm_name) else (20, CANONICAL_NAMES[20])
        if "obstacle" in norm_name or "engel" in norm_name:
            return 14, CANONICAL_NAMES[14]
        if "path" in norm_name or "ground" in norm_name or "yol" in norm_name or "zemin" in norm_name:
            return 15, CANONICAL_NAMES[15]
        if "prompt" in norm_name or "crest" in norm_name or "icon" in norm_name:
            return 13, CANONICAL_NAMES[13]
        if any(k in norm_name for k in ("kinah", "kina", "altin", "gold", "coin", "para", "currency")):
            return 25, CANONICAL_NAMES[25]

    # Priority 6: Fallback or Error
    if allow_fallback and default_fallback is not None and default_fallback in CANONICAL_NAMES:
        return default_fallback, CANONICAL_NAMES[default_fallback]

    raise ValueError(
        f"Non-canonical class cannot be resolved: class_id={class_id!r}, class_name={class_name!r}"
    )


def try_sanitize_class_id_and_name(
    class_id: Any,
    class_name: Optional[str] = "",
) -> Optional[Tuple[int, str]]:
    """Safely sanitize class_id and class_name. Returns None if unresolvable."""
    try:
        return sanitize_class_id_and_name(class_id, class_name)
    except (TypeError, ValueError):
        return None


def sanitize_yolo_box(
    box: Dict[str, Any],
    strict_drop_non_canonical: bool = True,
) -> Optional[Dict[str, Any]]:
    """Sanitize a bounding box dict in-place or via copy.

    Enforces that 'class_id' is in 0..24 and 'class_name' matches canonical name.
    If the box has an unresolvable non-canonical class (e.g. 98 or unresolvable essence):
      - Returns None if strict_drop_non_canonical is True (dropping the poisoned box).
      - Returns unmodified box if strict_drop_non_canonical is False.
    """
    if not isinstance(box, dict):
        return None
    raw_cid = box.get("class_id")
    raw_cname = box.get("class_name", "")

    sanitized = try_sanitize_class_id_and_name(raw_cid, raw_cname)
    if sanitized is None:
        if strict_drop_non_canonical:
            return None
        return box

    cid, cname = sanitized
    box["class_id"] = cid
    box["class_name"] = cname
    return box


def filter_canonical_review_boxes(
    boxes: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Filter a sequence of review boxes, dropping any unresolvable non-canonical classes."""
    filtered: List[Dict[str, Any]] = []
    for b in boxes:
        if not isinstance(b, dict):
            continue
        sanitized = sanitize_yolo_box(dict(b), strict_drop_non_canonical=True)
        if sanitized is not None:
            filtered.append(sanitized)
    return filtered


def format_yolo_txt_line(
    class_id: Any,
    cx: float,
    cy: float,
    w: float,
    h: float,
    class_name: Optional[str] = None,
) -> Optional[str]:
    """Format a canonical YOLO line: '<class_id> <cx> <cy> <w> <h>'.

    Strictly prevents non-canonical classes (like 98 or unresolvable) from entering .txt.
    Returns None if class is invalid or box coordinates are invalid/degenerate.
    """
    try:
        sanitized_id, _ = sanitize_class_id_and_name(class_id, class_name)
    except (TypeError, ValueError):
        return None

    if not is_valid_canonical_class_id(sanitized_id):
        return None

    try:
        fcx, fcy = float(cx), float(cy)
        fw, fh = float(w), float(h)
    except (TypeError, ValueError):
        return None

    # Clamp normalized coordinates to [0.0, 1.0]
    fcx = max(0.0, min(1.0, fcx))
    fcy = max(0.0, min(1.0, fcy))
    fw = max(0.0, min(1.0, fw))
    fh = max(0.0, min(1.0, fh))

    if fw <= 1e-6 or fh <= 1e-6:
        return None

    return f"{sanitized_id} {fcx:.6f} {fcy:.6f} {fw:.6f} {fh:.6f}"


def parse_and_sanitize_yolo_txt_line(
    line: str,
) -> Optional[Tuple[int, float, float, float, float, str]]:
    """Parse a YOLO txt line, sanitize class_id, and return (canonical_id, cx, cy, w, h, name).

    Returns None if line is empty, comment, malformed, or has an unresolvable class ID.
    """
    clean_line = str(line or "").strip()
    if not clean_line or clean_line.startswith("#"):
        return None
    parts = clean_line.split()
    if len(parts) < 5:
        return None
    try:
        raw_cid = int(parts[0])
        cx = float(parts[1])
        cy = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])
    except (TypeError, ValueError):
        return None

    sanitized = try_sanitize_class_id_and_name(raw_cid)
    if sanitized is None:
        return None
    cid, cname = sanitized
    return (cid, cx, cy, w, h, cname)


# =============================================================================
# Backward Compatibility & Export Functions
# =============================================================================

def dataset_class_names() -> Dict[int, str]:
    """Return dictionary of all canonical class IDs and names (0..26)."""
    _build_taxonomy()
    return {cid: name for cid, name in _NAMES.items() if cid in ALL_CANONICAL_CLASSES}


def canonical_22_class_names() -> Dict[int, str]:
    """Return dictionary of the core 22 canonical class IDs and names (0..21)."""
    return dict(CORE_22_CANONICAL_NAMES)


def dataset_labels_tr() -> Dict[int, str]:
    """Return dictionary of all Turkish labels for canonical classes."""
    _build_taxonomy()
    return {cid: lbl for cid, lbl in _LABELS_TR.items() if cid in ALL_CANONICAL_CLASSES}


def alias_map() -> Dict[str, int]:
    """Return mapping of all text aliases and catalog slugs to canonical class IDs."""
    _build_taxonomy()
    return {alias: target_cid for alias, target_cid in _ALIASES.items() if target_cid in ALL_CANONICAL_CLASSES}


def class_id_for_material_text(text: str) -> Optional[int]:
    """Exact material, Turkish name, or class key lookup. Returns canonical class ID or None."""
    _build_taxonomy()
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0]
    return None


def material_name_for_class(class_id: int) -> str:
    """Gather resource name for a class, or the Turkish label."""
    _build_taxonomy()
    cid = int(class_id)
    if cid in _RESOURCE_NAMES:
        return _RESOURCE_NAMES[cid]
    return _LABELS_TR.get(cid, _NAMES.get(cid, str(cid)))


def gather_class_ids() -> FrozenSet[int]:
    """Classes the bot may walk to and extract (3D materials + minimap sprout)."""
    return frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12})


def gather_3d_material_class_ids() -> FrozenSet[int]:
    """3D gatherable world materials only."""
    return MATERIAL_3D_CLASSES


def cognition_class_ids() -> FrozenSet[int]:
    """Bars, potions, scrolls, skills, obstacles, roads, currency, and player character."""
    return frozenset({13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26})


def ui_class_specs() -> List[dict]:
    """UI chart and combobox specs for non-base classes (classes >= 16: vitals, skills, consumables, kinah, player).

    Ensures perfect compatibility with `YOLO_12_CLASS_SPECS.extend(_ui_class_specs())`.
    """
    _build_taxonomy()
    rows: List[dict] = []
    for cid in sorted(class_id for class_id in _NAMES if class_id >= 16 and class_id in ALL_CANONICAL_CLASSES):
        hex_color = CANONICAL_PALETTE_HEX.get(cid, "#94a3b8")
        bgr_color = CANONICAL_PALETTE_BGR.get(cid, (180, 180, 180))
        rows.append(
            {
                "id": cid,
                "key": _NAMES[cid],
                "label_tr": f"{cid}: {_LABELS_TR[cid]}",
                "full_tr": f"{cid}: {_LABELS_TR[cid]} ({_NAMES[cid]})",
                "color_hex": hex_color,
                "color_bgr": bgr_color,
            }
        )
    return rows


def all_ui_class_specs() -> List[dict]:
    """UI specs for all canonical classes (0..26)."""
    _build_taxonomy()
    rows: List[dict] = []
    for cid in sorted(class_id for class_id in _NAMES if class_id in ALL_CANONICAL_CLASSES):
        hex_color = CANONICAL_PALETTE_HEX.get(cid, "#94a3b8")
        bgr_color = CANONICAL_PALETTE_BGR.get(cid, (180, 180, 180))
        rows.append(
            {
                "id": cid,
                "key": _NAMES[cid],
                "label_tr": f"{cid}: {_LABELS_TR[cid]}",
                "full_tr": f"{cid}: {_LABELS_TR[cid]} ({_NAMES[cid]})",
                "color_hex": hex_color,
                "color_bgr": bgr_color,
            }
        )
    return rows
