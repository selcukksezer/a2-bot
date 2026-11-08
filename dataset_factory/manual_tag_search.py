# -*- coding: utf-8 -*-
"""dataset_factory/manual_tag_search.py
=======================================
Interactive Manual Tag Search & Human-Approved Annotation Box Builder for Aion 2.

Provides:
- `SearchableTagRegistry`: Bilingual (Turkish & English) instant search and
  filtering registry over all canonical YOLO classes (0..24: 22 core classes +
  3 consumable classes: hp_potion, mp_potion, buff_scroll) and the 140+ Aion 2
  materials catalog.
- `search_label_classes(query: str) -> List[Dict[str, Any]]`: Instant search
  helper for UI search bars, combobox filters, and quick-tag hotkeys (supports
  queries like "aria", "orichalcum", "pot", "iksir", "scroll", "parşömen",
  "canavar", "skill", "filiz", "engel", "yol", etc.).
- `create_manual_annotation_box(class_id, cx, cy, w, h) -> Dict[str, Any]`:
  Constructs a canonical, sanitized manual bounding box dictionary with
  `trusted=True, weak=False, hand_label=True, human_approved=True, confidence=1.0`.
"""

from __future__ import annotations

import re
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from config.materials_db import AION2_MATERIALS_CATALOG
from dataset_factory.class_taxonomy import (
    ALL_CANONICAL_CLASSES,
    CANONICAL_LABELS_TR,
    CANONICAL_NAMES,
    CANONICAL_PALETTE_BGR,
    CANONICAL_PALETTE_HEX,
    CANONICAL_SHORT_LABELS_TR,
    CONSUMABLE_CLASSES,
    CORE_22_CLASSES,
    HUD_ELEMENT_CLASSES,
    HUD_SKILL_CLASSES,
    HUD_VITALS_CLASSES,
    MATERIAL_3D_CLASSES,
    MINIMAP_SYMBOL_CLASSES,
    MONSTER_CLASSES,
    NAVIGABLE_ELEMENT_CLASSES,
    format_yolo_txt_line,
    gather_class_ids,
    get_super_category,
    is_valid_canonical_class_id,
    material_name_for_class,
    sanitize_class_id_and_name,
    try_sanitize_class_id_and_name,
)

# =============================================================================
# Turkish & ASCII Text Normalization Utilities
# =============================================================================

_TR_CHAR_MAP = str.maketrans(
    {
        "ı": "i",
        "İ": "i",
        "I": "i",
        "ğ": "g",
        "Ğ": "g",
        "ü": "u",
        "Ü": "u",
        "ş": "s",
        "Ş": "s",
        "ö": "o",
        "Ö": "o",
        "ç": "c",
        "Ç": "c",
        "â": "a",
        "Â": "a",
        "î": "i",
        "Î": "i",
        "û": "u",
        "Û": "u",
    }
)


def _normalize_tr(text: Any) -> str:
    """Lowercase text while preserving Turkish characters accurately."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    # Replace Turkish capital I/İ before standard lower()
    raw = raw.replace("İ", "i").replace("I", "ı")
    cleaned = " ".join(raw.lower().split())
    return cleaned


def _fold_ascii(text: Any) -> str:
    """Fold Turkish and accented characters into plain lowercase ASCII."""
    raw = str(text or "").strip()
    if not raw:
        return ""
    translated = raw.translate(_TR_CHAR_MAP).lower()
    decomposed = unicodedata.normalize("NFKD", translated).encode("ascii", "ignore").decode("ascii")
    cleaned = re.sub(r"[^a-z0-9\s_-]+", " ", decomposed)
    return " ".join(cleaned.split())


def _tokenize_search_text(text: Any) -> List[str]:
    """Split normalized search text into alphanumeric tokens."""
    folded = _fold_ascii(text)
    if not folded:
        return []
    return [tok for tok in re.split(r"[\s_\-/,:;()]+", folded) if tok]


# =============================================================================
# Curated Bilingual Keywords & Search Synonyms per Canonical Class (0..24)
# =============================================================================

_SUPER_CATEGORY_LABELS_TR: Dict[str, str] = {
    "3d_material": "3B Dünya Materyali / Kaynak",
    "minimap": "Minimap Sembolü",
    "monster": "Canavar / Düşman",
    "navigable": "Navigasyon / Arazi",
    "hud_vitals": "HUD Can / Mana Barı",
    "hud_skill": "HUD Skill / Arayüz",
    "consumable": "Tüketilebilir Eşya",
    "currency": "Para / Kinah Göstergesi",
}

_CURATED_CLASS_KEYWORDS: Dict[int, Tuple[str, ...]] = {
    0: (
        "aria",
        "herb_aria",
        "herb aria",
        "fresh aria",
        "pure aria",
        "mor kristal bitki",
        "mor bitki",
        "mor çiçek",
        "mor cicek",
        "bitki",
        "ot",
        "çiçek",
        "cicek",
        "herb",
        "plant",
        "flower",
        "gather",
        "toplama",
        "materyal",
        "material",
    ),
    1: (
        "mela",
        "herb_mela",
        "fruit_mela",
        "fresh mela",
        "kırmızı meyveli çalı",
        "kirmizi meyveli cali",
        "kırmızı bitki",
        "kirmizi bitki",
        "meyve",
        "çalı",
        "cali",
        "berry",
        "fruit",
        "herb",
        "bitki",
        "ot",
        "gather",
        "toplama",
        "materyal",
    ),
    2: (
        "targena",
        "herb_targena",
        "azpha",
        "herb_azpha",
        "calendula",
        "herb_calendula",
        "fresh targena",
        "yeşil bitki",
        "yesil bitki",
        "şifalı ot",
        "sifali ot",
        "bitki",
        "ot",
        "herb",
        "plant",
        "gather",
        "toplama",
        "materyal",
    ),
    3: (
        "orichalcum",
        "ore_orichalcum",
        "orichalcum ore",
        "orikalkum",
        "maden",
        "cevher",
        "maden cevheri",
        "ore",
        "metal",
        "iron",
        "demir",
        "silver",
        "gümüş",
        "gumus",
        "gold",
        "altın",
        "altin",
        "titanium",
        "titanyum",
        "adamantium",
        "copper",
        "bakır",
        "bakir",
        "platinum",
        "platin",
        "kaya madeni",
        "mining",
        "kazı",
        "kazi",
        "gather",
        "materyal",
    ),
    4: (
        "gem",
        "crystal",
        "gem_crystal",
        "kristal",
        "mücevher",
        "mucevher",
        "değerli taş",
        "degerli tas",
        "safir",
        "sapphire",
        "yakut",
        "ruby",
        "elmas",
        "diamond",
        "zümrüt",
        "zumrut",
        "emerald",
        "ametist",
        "amethyst",
        "topaz",
        "opal",
        "garnet",
        "lal",
        "aquamarine",
        "odyle",
        "aether",
        "eter",
        "mavi kristal",
        "gather",
        "materyal",
    ),
    5: (
        "tree",
        "log",
        "tree_log",
        "wood",
        "wood_tree",
        "timber",
        "log_timber",
        "ağaç",
        "agac",
        "kereste",
        "odun",
        "kütük",
        "kutuk",
        "gövde",
        "govde",
        "yggdrasil",
        "asvata",
        "draupnir",
        "bambu",
        "bamboo",
        "lumber",
        "kesim",
        "gather",
        "materyal",
    ),
    6: (
        "edana",
        "herb_edana",
        "fresh edana",
        "mor pembe bitki",
        "pembe bitki",
        "pembe çiçek",
        "pembe cicek",
        "edana otu",
        "herb",
        "bitki",
        "ot",
        "gather",
        "toplama",
        "materyal",
    ),
    7: (
        "lumina",
        "herb_lumina",
        "fresh lumina",
        "sarı altın parıltılı bitki",
        "sari altin pariltili bitki",
        "sarı bitki",
        "sari bitki",
        "altın bitki",
        "altin bitki",
        "parlak bitki",
        "ışıklı bitki",
        "isikli bitki",
        "lumina otu",
        "herb",
        "bitki",
        "ot",
        "gather",
        "toplama",
        "materyal",
    ),
    8: (
        "obsidian",
        "ore_obsidian",
        "obsidian ore",
        "obsidyen",
        "koyu kristal maden",
        "siyah kristal",
        "kara maden",
        "kara taş",
        "kara tas",
        "volkanik cam",
        "maden",
        "cevher",
        "ore",
        "dark crystal",
        "gather",
        "materyal",
    ),
    9: (
        "filiz",
        "minimap_sprout",
        "minimap filiz",
        "sprout",
        "minimap sprout",
        "3 yapraklı yeşil",
        "3 yaprakli yesil",
        "yaprak",
        "yeşil filiz",
        "yesil filiz",
        "harita filiz",
        "minimap bitki",
        "minimap kaynak",
        "radar filiz",
        "fide",
        "minimap",
        "harita",
    ),
    10: (
        "canavar",
        "monster",
        "monster_3d",
        "3b canavar",
        "3d monster",
        "mob",
        "mob_guardian",
        "düşman",
        "dusman",
        "yaratık",
        "yaratik",
        "boss",
        "elit",
        "elite",
        "hedef canavar",
        "enemy",
        "hostile",
        "creature",
        "agresif",
        "saldırgan",
        "saldirgan",
    ),
    11: (
        "minimap canavar",
        "minimap_monster_dot",
        "minimap canavar noktası",
        "minimap canavar noktasi",
        "minimap_monster",
        "canavar noktası",
        "canavar noktasi",
        "kırmızı nokta",
        "kirmizi nokta",
        "red dot",
        "monster dot",
        "radar canavar",
        "harita canavar",
        "canavar",
        "monster",
        "mob",
        "minimap",
        "harita",
    ),
    12: (
        "kan kristali",
        "crystal_blood",
        "blood_crystal",
        "blood crystal",
        "kırmızı kristal",
        "kirmizi kristal",
        "kan taşı",
        "kan tasi",
        "kızıl kristal",
        "kizil kristal",
        "red crystal",
        "crimson crystal",
        "kristal",
        "crystal",
        "gather",
        "materyal",
    ),
    13: (
        "skill",
        "skill_ui_prompt",
        "skill/arayüz istemi",
        "skill arayuz istemi",
        "arayüz istemi",
        "arayuz istemi",
        "etkileşim",
        "etkilesim",
        "toplama ikonu",
        "f tuşu",
        "f tusu",
        "e tuşu",
        "e tusu",
        "f_gather_prompt",
        "e_gather_icon",
        "world_gather_crest",
        "crest",
        "prompt",
        "buton",
        "ikon",
        "icon",
        "ui",
        "hud",
        "yetenek",
    ),
    14: (
        "engel",
        "obstacle",
        "duvar",
        "kaya",
        "taş engel",
        "tas engel",
        "bariyer",
        "barikat",
        "çit",
        "cit",
        "uçurum",
        "ucurum",
        "su engeli",
        "geçilmez",
        "gecilmez",
        " çarpışma",
        "carpisma",
        "collision",
        "wall",
        "rock",
        "barrier",
        "block",
        "blocked",
        "terrain obstacle",
        "navigasyon",
    ),
    15: (
        "yol",
        "path",
        "path_ground",
        "zemin",
        "yol/zemin",
        "yol zemin",
        "ground",
        "road",
        "patika",
        "geçit",
        "gecit",
        "yürünebilir",
        "yurunebilir",
        "yürüme yolu",
        "yurume yolu",
        "rota",
        "rota zemini",
        "walkable",
        "terrain",
        "floor",
        "trail",
        "way",
        "koridor",
        "köprü",
        "kopru",
        "navigasyon",
    ),
    16: (
        "oyuncu can barı",
        "oyuncu can bari",
        "player_hp_bar",
        "oyuncu hp",
        "player hp",
        "hp bar",
        "can barı",
        "can bari",
        "can çubuğu",
        "can cubugu",
        "sağlık barı",
        "saglik bari",
        "health bar",
        "hp",
        "can",
        "vitals",
        "hud",
    ),
    17: (
        "oyuncu mana barı",
        "oyuncu mana bari",
        "player_mp_bar",
        "oyuncu mp",
        "player mp",
        "mp bar",
        "mana barı",
        "mana bari",
        "mana çubuğu",
        "mana cubugu",
        "mana",
        "mp",
        "büyü gücü",
        "buyu gucu",
        "vitals",
        "hud",
    ),
    18: (
        "hedef can barı",
        "hedef can bari",
        "target_hp_bar",
        "hedef hp",
        "target hp",
        "target health",
        "canavar can barı",
        "canavar can bari",
        "düşman can barı",
        "dusman can bari",
        "mob hp",
        "boss hp",
        "hedef barı",
        "hedef bari",
        "hp bar",
        "can barı",
        "hp",
        "vitals",
        "hud",
    ),
    19: (
        "skill",
        "skill_bar",
        "skill barı",
        "skill bari",
        "skill bar",
        "yetenek barı",
        "yetenek bari",
        "yetenek çubuğu",
        "yetenek cubugu",
        "beceri çubuğu",
        "beceri cubugu",
        "action bar",
        "hotbar",
        "quickbar",
        "kısayol çubuğu",
        "kisayol cubugu",
        "yetenek",
        "beceri",
        "hud",
    ),
    20: (
        "skill",
        "skill_slot_ready",
        "hazır skill yuvası",
        "hazir skill yuvasi",
        "skill hazır",
        "skill hazir",
        "ready skill slot",
        "skill_ready",
        "skill_slot",
        "aktif skill",
        "kullanılabilir skill",
        "kullanilabilir skill",
        "hazır yetenek",
        "hazir yetenek",
        "yetenek yuvası",
        "yetenek yuvasi",
        "skill yuvası",
        "skill yuvasi",
        "slot",
        "yuva",
        "yetenek",
        "beceri",
        "ready",
        "hud",
    ),
    21: (
        "skill",
        "skill_slot_cooldown",
        "beklemede skill yuvası",
        "beklemede skill yuvasi",
        "skill cd",
        "skill_cd",
        "skill_slot_cd",
        "cooldown skill slot",
        "cooldown",
        "cd",
        "bekleme süresi",
        "bekleme suresi",
        "dolum süresi",
        "dolum suresi",
        "kapalı skill",
        "kapali skill",
        "pasif skill",
        "bekleyen yetenek",
        "yetenek cd",
        "slot",
        "yuva",
        "yetenek",
        "beceri",
        "hud",
    ),
    22: (
        "pot",
        "iksir",
        "can iksiri",
        "hp_potion",
        "hp potion",
        "health potion",
        "health_potion",
        "potion_hp",
        "can potu",
        "hp pot",
        "hp potu",
        "kırmızı iksir",
        "kirmizi iksir",
        "kırmızı pot",
        "kirmizi pot",
        "sağlık iksiri",
        "saglik iksiri",
        "şifa iksiri",
        "sifa iksiri",
        "healing potion",
        "red potion",
        "potion",
        "şişe",
        "sise",
        "tüketilebilir",
        "tuketilebilir",
        "consumable",
    ),
    23: (
        "pot",
        "iksir",
        "mana iksiri",
        "mp_potion",
        "mp potion",
        "mana potion",
        "mana_potion",
        "potion_mp",
        "mana potu",
        "mp pot",
        "mp potu",
        "mavi iksir",
        "mavi pot",
        "büyü iksiri",
        "buyu iksiri",
        "blue potion",
        "potion",
        "şişe",
        "sise",
        "tüketilebilir",
        "tuketilebilir",
        "consumable",
    ),
    24: (
        "scroll",
        "parşömen",
        "parsomen",
        "buff_scroll",
        "buff scroll",
        "scroll_buff",
        "buff",
        "buff parşömeni",
        "buff parsomeni",
        "büyü parşömeni",
        "buyu parsomeni",
        "güçlendirme parşömeni",
        "guclendirme parsomeni",
        "hız parşömeni",
        "hiz parsomeni",
        "saldırı parşömeni",
        "saldiri parsomeni",
        "savunma parşömeni",
        "savunma parsomeni",
        "kağıt",
        "kagit",
        "büyü kağıdı",
        "buyu kagidi",
        "parchment",
        "tüketilebilir",
        "tuketilebilir",
        "consumable",
    ),
    25: (
        "kinah",
        "kina",
        "altin",
        "altın",
        "gold",
        "para",
        "coin",
        "coins",
        "currency",
        "para birimi",
        "bakiye",
        "kinah göstergesi",
        "kinah gostergesi",
        "kinah display",
        "altın göstergesi",
        "altin gostergesi",
        "para göstergesi",
        "para gostergesi",
        "hud",
        "economy",
        "kazanç",
        "kazanc",
    ),
}


# =============================================================================
# SearchableTagRegistry
# =============================================================================

class SearchableTagRegistry:
    """Bilingual (Turkish & English) searchable registry for all canonical YOLO classes.

    Provides instant filtering across class IDs, canonical keys, Turkish display names,
    English resource names, super-categories, curated game terminology, and the 140+
    Aion 2 materials catalog.
    """

    def __init__(self) -> None:
        self._entries_by_id: Dict[int, Dict[str, Any]] = {}
        self._search_blobs_tr: Dict[int, str] = {}
        self._search_blobs_ascii: Dict[int, str] = {}
        self._keyword_sets_tr: Dict[int, Set[str]] = {}
        self._keyword_sets_ascii: Dict[int, Set[str]] = {}
        self._token_sets_ascii: Dict[int, Set[str]] = {}
        self._build_registry()

    def _build_registry(self) -> None:
        """Populate registry metadata and search indices for classes 0..24."""
        gather_ids = gather_class_ids()

        # Collect catalog material names mapped to their canonical class IDs
        catalog_keywords_by_cid: Dict[int, List[str]] = {cid: [] for cid in ALL_CANONICAL_CLASSES}
        for spec in AION2_MATERIALS_CATALOG:
            sanitized = try_sanitize_class_id_and_name(-1, spec.name)
            if sanitized is None:
                sanitized = try_sanitize_class_id_and_name(-1, spec.base_material)
            if sanitized is not None:
                cid = sanitized[0]
                if cid in catalog_keywords_by_cid:
                    for term in (spec.name, spec.name_tr, spec.base_material, spec.category):
                        if term:
                            catalog_keywords_by_cid[cid].append(str(term))

        for cid in sorted(ALL_CANONICAL_CLASSES):
            cname = CANONICAL_NAMES[cid]
            label_tr = CANONICAL_LABELS_TR.get(cid, cname)
            short_tr = CANONICAL_SHORT_LABELS_TR.get(cid, label_tr)
            label_en = material_name_for_class(cid)
            super_cat = get_super_category(cid) or "other"
            category_tr = _SUPER_CATEGORY_LABELS_TR.get(super_cat, super_cat)
            color_hex = CANONICAL_PALETTE_HEX.get(cid, "#94a3b8")
            color_bgr = CANONICAL_PALETTE_BGR.get(cid, (180, 180, 180))

            curated = list(_CURATED_CLASS_KEYWORDS.get(cid, ()))
            catalog_terms = catalog_keywords_by_cid.get(cid, [])

            raw_keywords: List[str] = [
                str(cid),
                cname,
                cname.replace("_", " "),
                label_tr,
                short_tr,
                label_en,
                super_cat,
                category_tr,
                *curated,
                *catalog_terms,
            ]

            # Deduplicate keywords while preserving order
            dedup_keywords: List[str] = []
            seen_kw: Set[str] = set()
            for kw in raw_keywords:
                norm_kw = _normalize_tr(kw)
                if norm_kw and norm_kw not in seen_kw:
                    seen_kw.add(norm_kw)
                    dedup_keywords.append(str(kw).strip())

            kw_set_tr: Set[str] = set()
            kw_set_ascii: Set[str] = set()
            tok_set_ascii: Set[str] = set()

            for kw in dedup_keywords:
                ntr = _normalize_tr(kw)
                nascii = _fold_ascii(kw)
                if ntr:
                    kw_set_tr.add(ntr)
                if nascii:
                    kw_set_ascii.add(nascii)
                    for tok in _tokenize_search_text(nascii):
                        tok_set_ascii.add(tok)

            display_label = f"{cid}: {label_tr} ({cname})"
            hotkey = str(cid + 1) if cid < 9 else ("0" if cid == 9 else "")

            entry: Dict[str, Any] = {
                "id": cid,
                "class_id": cid,
                "key": cname,
                "class_name": cname,
                "name": cname,
                "label_tr": label_tr,
                "short_label_tr": short_tr,
                "label_en": label_en,
                "display_label": display_label,
                "full_tr": display_label,
                "super_category": super_cat,
                "category_tr": category_tr,
                "color_hex": color_hex,
                "color_bgr": color_bgr,
                "shortcut_key": hotkey,
                "is_core_22": cid in CORE_22_CLASSES,
                "is_consumable": cid in CONSUMABLE_CLASSES,
                "is_gatherable": cid in gather_ids,
                "is_3d_material": cid in MATERIAL_3D_CLASSES,
                "is_minimap": cid in MINIMAP_SYMBOL_CLASSES,
                "is_monster": cid in MONSTER_CLASSES or cid == 11,
                "is_navigable": cid in NAVIGABLE_ELEMENT_CLASSES,
                "is_hud": cid in HUD_ELEMENT_CLASSES,
                "is_skill": cid in HUD_SKILL_CLASSES,
                "is_vitals": cid in HUD_VITALS_CLASSES,
                "keywords": dedup_keywords,
                "aliases": list(curated),
            }

            self._entries_by_id[cid] = entry
            self._keyword_sets_tr[cid] = kw_set_tr
            self._keyword_sets_ascii[cid] = kw_set_ascii
            self._token_sets_ascii[cid] = tok_set_ascii
            self._search_blobs_tr[cid] = " | ".join(sorted(kw_set_tr))
            self._search_blobs_ascii[cid] = " | ".join(sorted(kw_set_ascii))

    def get_all_classes(self, include_consumables: bool = True) -> List[Dict[str, Any]]:
        """Return all canonical class metadata dictionaries ordered by class_id."""
        results: List[Dict[str, Any]] = []
        for cid in sorted(self._entries_by_id.keys()):
            if not include_consumables and cid in CONSUMABLE_CLASSES:
                continue
            item = dict(self._entries_by_id[cid])
            item["score"] = 1.0
            results.append(item)
        return results

    def get_by_id(self, class_id: Any) -> Optional[Dict[str, Any]]:
        """Return class metadata dictionary for a canonical class_id (0..24), or None."""
        try:
            cid = int(class_id)
        except (TypeError, ValueError):
            return None
        entry = self._entries_by_id.get(cid)
        if entry is None:
            return None
        result = dict(entry)
        result["score"] = 1.0
        return result

    def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Resolve a class name or alias to its canonical class metadata dictionary."""
        sanitized = try_sanitize_class_id_and_name(-1, name)
        if sanitized is not None:
            return self.get_by_id(sanitized[0])
        matches = self.search(name, limit=1)
        return matches[0] if matches else None

    def get_class(self, class_id_or_name: Any) -> Optional[Dict[str, Any]]:
        """Return canonical class metadata by integer ID or string name/query."""
        if isinstance(class_id_or_name, int):
            return self.get_by_id(class_id_or_name)
        try:
            cid = int(str(class_id_or_name).strip())
            if cid in self._entries_by_id:
                return self.get_by_id(cid)
        except (TypeError, ValueError):
            pass
        return self.get_by_name(str(class_id_or_name or ""))

    def resolve_class_id(self, query_or_id: Any) -> Optional[int]:
        """Resolve any integer ID, class name, or search term to a canonical class ID (0..24)."""
        entry = self.get_class(query_or_id)
        if entry is not None:
            return int(entry["class_id"])
        return None

    def _score_entry(self, cid: int, query_tr: str, query_ascii: str, query_tokens: List[str]) -> float:
        """Compute relevance score (>= 0.0) for class `cid` against a search query."""
        entry = self._entries_by_id[cid]
        cname_ascii = _fold_ascii(entry["class_name"])
        short_tr = _normalize_tr(entry["short_label_tr"])
        short_ascii = _fold_ascii(entry["short_label_tr"])
        label_tr = _normalize_tr(entry["label_tr"])
        label_ascii = _fold_ascii(entry["label_tr"])
        label_en_ascii = _fold_ascii(entry["label_en"])

        kw_tr = self._keyword_sets_tr[cid]
        kw_ascii = self._keyword_sets_ascii[cid]
        tokens_ascii = self._token_sets_ascii[cid]
        blob_tr = self._search_blobs_tr[cid]
        blob_ascii = self._search_blobs_ascii[cid]

        score = 0.0

        # 1. Exact numeric ID match
        if query_ascii.isdigit() and int(query_ascii) == cid:
            return 1000.0

        # 2. Exact match on canonical name, short Turkish label, or English resource name
        if query_tr in (cname_ascii, short_tr, label_tr) or query_ascii in (
            cname_ascii,
            short_ascii,
            label_ascii,
            label_en_ascii,
        ):
            score = max(score, 950.0)

        # 3. Exact match on any curated keyword / alias
        if query_tr in kw_tr or query_ascii in kw_ascii:
            score = max(score, 850.0)

        # 4. Exact token match in primary fields (e.g., "aria", "orichalcum", "pot", "iksir", "scroll", "parşömen", "canavar", "skill", "filiz", "engel", "yol")
        primary_tokens = set(
            _tokenize_search_text(
                f"{entry['class_name']} {entry['short_label_tr']} {entry['label_tr']} {entry['label_en']}"
            )
        )
        if query_ascii in primary_tokens:
            score = max(score, 820.0)
        elif query_ascii in tokens_ascii:
            score = max(score, 760.0)

        # 5. Prefix match on primary labels or keywords
        if (
            cname_ascii.startswith(query_ascii)
            or short_ascii.startswith(query_ascii)
            or label_ascii.startswith(query_ascii)
            or label_en_ascii.startswith(query_ascii)
            or short_tr.startswith(query_tr)
            or label_tr.startswith(query_tr)
        ):
            score = max(score, 720.0)
        elif any(k.startswith(query_tr) for k in kw_tr) or any(k.startswith(query_ascii) for k in kw_ascii):
            score = max(score, 650.0)
        elif any(tok.startswith(query_ascii) for tok in tokens_ascii):
            score = max(score, 600.0)

        # 6. Multi-token AND matching (all query tokens must match when multiple tokens are given)
        if len(query_tokens) > 1:
            matched_tokens = 0
            token_bonus = 0.0
            for qtok in query_tokens:
                if qtok in primary_tokens:
                    matched_tokens += 1
                    token_bonus += 120.0
                elif qtok in tokens_ascii:
                    matched_tokens += 1
                    token_bonus += 90.0
                elif any(t.startswith(qtok) for t in tokens_ascii):
                    matched_tokens += 1
                    token_bonus += 70.0
                elif qtok in blob_ascii:
                    matched_tokens += 1
                    token_bonus += 50.0
            if matched_tokens == len(query_tokens):
                score = max(score, 500.0 + token_bonus)

        # 7. Substring match in primary names or keyword blobs (for queries >= 4 chars, or primary fields)
        if score == 0.0:
            if (
                query_tr in short_tr
                or query_tr in label_tr
                or query_ascii in cname_ascii
                or query_ascii in short_ascii
                or query_ascii in label_ascii
                or query_ascii in label_en_ascii
            ):
                score = max(score, 520.0)
            elif len(query_ascii) >= 4 and (query_tr in blob_tr or query_ascii in blob_ascii):
                score = max(score, 400.0)

        # 8. Priority boost for primary canonical target of iconic queries
        if score > 0.0:
            # Prefer 3D monster (10) slightly above minimap monster dot (11) when query is just "canavar" or "monster"
            if query_ascii in ("canavar", "monster", "mob") and cid == 10:
                score += 25.0
            # Prefer HP potion (22) slightly above MP potion (23) when generic "pot" or "iksir" is searched
            if query_ascii in ("pot", "potion", "iksir") and cid in (22, 23):
                score += 30.0 if cid == 22 else 25.0
            # Prefer buff_scroll (24) when "scroll", "parsomen", "buff" is searched
            if query_ascii in ("scroll", "parsomen", "buff") and cid == 24:
                score += 40.0
            # Prefer ore_orichalcum (3) when "orichalcum" or "maden" is searched
            if query_ascii in ("orichalcum", "maden", "cevher", "ore") and cid == 3:
                score += 35.0

        return score

    def search(
        self,
        query: str = "",
        limit: Optional[int] = None,
        category: Optional[str] = None,
        include_consumables: bool = True,
    ) -> List[Dict[str, Any]]:
        """Filter and rank canonical classes matching the Turkish or English query.

        Args:
            query: Search string (e.g. "aria", "orichalcum", "pot", "iksir",
                "scroll", "parşömen", "canavar", "skill", "filiz", "engel", "yol").
                If empty, returns all classes in canonical order.
            limit: Optional maximum number of results to return.
            category: Optional super-category filter (e.g. "3d_material", "consumable", "hud_skill").
            include_consumables: Whether to include classes 22..24 (default True).

        Returns:
            List of matching class metadata dicts, sorted by descending relevance score
            and ascending class_id.
        """
        raw_query = str(query or "").strip()
        cat_filter = _fold_ascii(category) if category else ""

        if not raw_query:
            base = self.get_all_classes(include_consumables=include_consumables)
            if cat_filter:
                base = [
                    item
                    for item in base
                    if _fold_ascii(item["super_category"]) == cat_filter
                    or cat_filter in _fold_ascii(item["category_tr"])
                ]
            if limit is not None and limit > 0:
                return base[: int(limit)]
            return base

        query_tr = _normalize_tr(raw_query)
        query_ascii = _fold_ascii(raw_query)
        query_tokens = _tokenize_search_text(raw_query)

        scored_results: List[Tuple[float, int, Dict[str, Any]]] = []

        for cid in sorted(self._entries_by_id.keys()):
            if not include_consumables and cid in CONSUMABLE_CLASSES:
                continue
            entry = self._entries_by_id[cid]
            if cat_filter:
                if (
                    _fold_ascii(entry["super_category"]) != cat_filter
                    and cat_filter not in _fold_ascii(entry["category_tr"])
                ):
                    continue

            score = self._score_entry(cid, query_tr, query_ascii, query_tokens)
            if score > 0.0:
                item = dict(entry)
                item["score"] = round(float(score), 2)
                scored_results.append((score, cid, item))

        # Sort by descending score, then ascending canonical class_id
        scored_results.sort(key=lambda pair: (-pair[0], pair[1]))
        results = [item for _, _, item in scored_results]

        if limit is not None and limit > 0:
            return results[: int(limit)]
        return results

    # Alias for UI callers expecting filter_classes
    filter_classes = search

    def get_combobox_items(
        self,
        query: str = "",
        include_consumables: bool = True,
    ) -> List[Tuple[int, str]]:
        """Return `(class_id, display_label)` pairs matching `query` for UI dropdowns."""
        matches = self.search(query=query, include_consumables=include_consumables)
        return [(int(m["class_id"]), str(m["display_label"])) for m in matches]

    def create_manual_box(
        self,
        class_id: Any,
        cx: float,
        cy: float,
        w: float,
        h: float,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a manual human-approved annotation box via this registry."""
        return create_manual_annotation_box(class_id, cx, cy, w, h, **kwargs)


# =============================================================================
# Module-Level Singleton & Helper Functions
# =============================================================================

_DEFAULT_REGISTRY: Optional[SearchableTagRegistry] = None


def get_tag_registry() -> SearchableTagRegistry:
    """Return the singleton `SearchableTagRegistry` instance."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = SearchableTagRegistry()
    return _DEFAULT_REGISTRY


def search_label_classes(
    query: str = "",
    limit: Optional[int] = None,
    category: Optional[str] = None,
    include_consumables: bool = True,
) -> List[Dict[str, Any]]:
    """Instantly search and filter canonical label classes by Turkish/English query.

    Supports queries like:
    "aria", "orichalcum", "pot", "iksir", "scroll", "parşömen",
    "canavar", "skill", "filiz", "engel", "yol", as well as numeric IDs (0..24)
    and any Aion 2 catalog material name.

    Args:
        query: User search query in Turkish or English.
        limit: Optional cap on number of returned class dicts.
        category: Optional super-category filter.
        include_consumables: Include consumable classes 22..24 (default True).

    Returns:
        List of matching class dictionaries ordered by relevance score.
    """
    return get_tag_registry().search(
        query=query,
        limit=limit,
        category=category,
        include_consumables=include_consumables,
    )


def create_manual_annotation_box(
    class_id: Any,
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    class_name: Optional[str] = None,
    image_width: Optional[ int] = None,
    image_height: Optional[int] = None,
    source: str = "manual_hand_label",
    clamp: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a canonical manual annotation bounding box dictionary.

    Guarantees that manual annotations drawn or edited by the user are marked with:
        `trusted=True, weak=False, hand_label=True, human_approved=True, confidence=1.0`
    so they are never purged by weak auto-label filters and are immediately eligible
    for the approved training pool and golden benchmark anchor bank.

    Args:
        class_id: Canonical integer class ID (0..24) or class name / search alias
            (e.g., 0, "herb_aria", "aria", 22, "hp_potion", "can iksiri").
        cx: Box center X coordinate (normalized 0..1, or pixel coordinate if image_width given).
        cy: Box center Y coordinate (normalized 0..1, or pixel coordinate if image_height given).
        w: Box width (normalized 0..1, or pixel width if image_width given).
        h: Box height (normalized 0..1, or pixel height if image_height given).
        class_name: Optional explicit class name hint for sanitization.
        image_width: Optional image width in pixels if cx, w are given in pixels.
        image_height: Optional image height in pixels if cy, h are given in pixels.
        source: Provenance tag (defaults to "manual_hand_label").
        clamp: Clamp normalized coordinates to [0.0, 1.0] (default True).
        extra_metadata: Optional additional key-value pairs to merge into the box dict.

    Returns:
        Canonical manual annotation box dictionary.

    Raises:
        ValueError: If `class_id` / `class_name` cannot be resolved to a valid canonical
            class (0..24) or if coordinates are non-numeric.
    """
    registry = get_tag_registry()

    # 1. Resolve canonical class_id and class_name
    resolved_id: Optional[int] = None
    resolved_name: Optional[str] = None

    sanitized = try_sanitize_class_id_and_name(class_id, class_name or "")
    if sanitized is not None:
        resolved_id, resolved_name = sanitized
    else:
        # Try registry search fallback if a Turkish/English search keyword was passed as class_id
        query_str = str(class_name if class_name else class_id or "").strip()
        if query_str:
            found = registry.get_class(query_str)
            if found is not None:
                resolved_id = int(found["class_id"])
                resolved_name = str(found["class_name"])

    if resolved_id is None or resolved_name is None or not is_valid_canonical_class_id(resolved_id):
        # Raise descriptive ValueError via canonical sanitizer
        sanitize_class_id_and_name(class_id, class_name or "")
        raise ValueError(f"Invalid canonical class_id={class_id!r}, class_name={class_name!r}")

    # 2. Parse and normalize coordinates
    try:
        fcx = float(cx)
        fcy = float(cy)
        fw = abs(float(w))
        fh = abs(float(h))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid box coordinates: cx={cx!r}, cy={cy!r}, w={w!r}, h={h!r}") from exc

    if image_width is not None and int(image_width) > 0 and (fcx > 1.0 or fw > 1.0):
        fcx = fcx / float(image_width)
        fw = fw / float(image_width)
    if image_height is not None and int(image_height) > 0 and (fcy > 1.0 or fh > 1.0):
        fcy = fcy / float(image_height)
        fh = fh / float(image_height)

    if clamp:
        fcx = max(0.0, min(1.0, fcx))
        fcy = max(0.0, min(1.0, fcy))
        fw = max(1e-4, min(1.0, fw))
        fh = max(1e-4, min(1.0, fh))
        # Ensure box edges stay within [0.0, 1.0]
        half_w = min(fw * 0.5, fcx, 1.0 - fcx)
        half_h = min(fh * 0.5, fcy, 1.0 - fcy)
        if half_w > 1e-4:
            fw = half_w * 2.0
        if half_h > 1e-4:
            fh = half_h * 2.0

    fcx = round(fcx, 6)
    fcy = round(fcy, 6)
    fw = round(fw, 6)
    fh = round(fh, 6)

    entry = registry.get_by_id(resolved_id) or {}
    label_tr = entry.get("label_tr", CANONICAL_LABELS_TR.get(resolved_id, resolved_name))
    short_tr = entry.get("short_label_tr", CANONICAL_SHORT_LABELS_TR.get(resolved_id, label_tr))
    super_cat = entry.get("super_category", get_super_category(resolved_id) or "other")
    color_hex = entry.get("color_hex", CANONICAL_PALETTE_HEX.get(resolved_id, "#94a3b8"))
    color_bgr = entry.get("color_bgr", CANONICAL_PALETTE_BGR.get(resolved_id, (180, 180, 180)))
    yolo_line = format_yolo_txt_line(resolved_id, fcx, fcy, fw, fh, class_name=resolved_name)

    box: Dict[str, Any] = {
        "class_id": int(resolved_id),
        "class_name": str(resolved_name),
        "label_tr": str(label_tr),
        "short_label_tr": str(short_tr),
        "super_category": str(super_cat),
        "cx": fcx,
        "cy": fcy,
        "w": fw,
        "h": fh,
        "confidence": 1.0,
        "conf": 1.0,
        "trusted": True,
        "weak": False,
        "hand_label": True,
        "human_approved": True,
        "manual": True,
        "is_manual": True,
        "approved": True,
        "verified": True,
        "source": str(source or "manual_hand_label"),
        "color_hex": str(color_hex),
        "color_bgr": tuple(color_bgr),
        "yolo_line": yolo_line,
        "created_at": round(time.time(), 3),
    }

    if isinstance(extra_metadata, dict) and extra_metadata:
        for k, v in extra_metadata.items():
            if k not in (
                "class_id",
                "class_name",
                "cx",
                "cy",
                "w",
                "h",
                "trusted",
                "weak",
                "hand_label",
                "human_approved",
                "confidence",
            ):
                box[k] = v

    return box


__all__ = [
    "SearchableTagRegistry",
    "get_tag_registry",
    "search_label_classes",
    "create_manual_annotation_box",
]
