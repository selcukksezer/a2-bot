# -*- coding: utf-8 -*-
"""dataset_factory/class_taxonomy.py
===================================
Canonical YOLO Class Taxonomy, Validation & Sanitization for Aion 2.

Defines and strictly enforces the canonical 27 classes (0..26): core 22
(0..21), consumable classes (22..24: hp_potion, mp_potion, buff_scroll),
currency (25: kinah) and player character (26: player_character).

Strictly blocks non-canonical classes (such as class_id: 98, essence_*, or legacy
omni IDs) from entering YOLO training `.txt` files or review `.json` files.
Provides super-category helpers for 3D world materials, minimap symbols, HUD elements,
navigable elements, and consumables.
"""

from __future__ import annotations

import logging
import numbers
import re
import threading
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union

from config.materials_db import AION2_MATERIALS_CATALOG

logger = logging.getLogger(__name__)

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

# v20 F4 (Selçuk 11.11): canlı log kanıtıyla jenerik sınıflar — yalnızca
# görülen katalog maddelerine alias (other_herb YOK; Dark Iron→27, Odyle→28).
GENERIC_F4_CANONICAL_NAMES: Dict[int, str] = {
    27: "other_ore",
    28: "other_odyle",
}

# MaterialScanner / auto_cartographer satırları (logs/, 2026-11-10..11).
F4_LIVE_GENERIC_MATERIAL_MAP: Dict[str, int] = {
    "Dark Iron": 27,
    "Odyle": 28,
}

CANONICAL_NAMES: Dict[int, str] = {
    **CORE_22_CANONICAL_NAMES,
    **CONSUMABLE_CANONICAL_NAMES,
    **CURRENCY_CANONICAL_NAMES,
    **PLAYER_CANONICAL_NAMES,
    **GENERIC_F4_CANONICAL_NAMES,
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
    8: "Obsidyen (Koyu Kristal Maden)",
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
    27: "Diğer Cevher (Jenerik)",
    28: "Odyle (Jenerik)",
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
    27: "Diğer Cevher",
    28: "Odyle",
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
    27: "#78716c",  # other_ore: neutral stone
    28: "#a78bfa",  # other_odyle: soft violet
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

MATERIAL_3D_CLASSES: FrozenSet[int] = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 27, 28})
# v20-loop: 3D materyal AİLELERİ (data.yaml kanonik isimlerinden).
# "3d_material" süper-kategorisi bitki+cevher+mücevher+ağacı tek kapta
# topluyordu; repair kapısı 2 (herb_targena) -> 3 (ore_orichalcum)
# dönüşümüne izin veriyordu (canlı 16:56:00). Aile kilidi: repair
# yalnızca AYNI aile içinde olur — bitki bitkiye, cevher cevhere.
# tree_log (5) tekildir: başka sınıfa repair edilemez.
MATERIAL_FAMILY_HERBS: FrozenSet[int] = frozenset({0, 1, 2, 6, 7})
MATERIAL_FAMILY_ORES: FrozenSet[int] = frozenset({3, 8, 27})
MATERIAL_FAMILY_GEMS: FrozenSet[int] = frozenset({4, 12})
MATERIAL_FAMILY_TREES: FrozenSet[int] = frozenset({5})
MATERIAL_FAMILIES: Tuple[FrozenSet[int], ...] = (
    MATERIAL_FAMILY_HERBS,
    MATERIAL_FAMILY_ORES,
    MATERIAL_FAMILY_GEMS,
    MATERIAL_FAMILY_TREES,
)
MINIMAP_SYMBOL_CLASSES: FrozenSet[int] = frozenset({9, 11})
MONSTER_CLASSES: FrozenSet[int] = frozenset({10})
NAVIGABLE_ELEMENT_CLASSES: FrozenSet[int] = frozenset({14, 15})
HUD_VITALS_CLASSES: FrozenSet[int] = frozenset({16, 17, 18})
HUD_SKILL_CLASSES: FrozenSet[int] = frozenset({13, 19, 20, 21})
CONSUMABLE_CLASSES: FrozenSet[int] = frozenset({22, 23, 24})
CURRENCY_CLASSES: FrozenSet[int] = frozenset({25})
PLAYER_CLASSES: FrozenSet[int] = frozenset({26})
HUD_ELEMENT_CLASSES: FrozenSet[int] = frozenset({13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26})

CORE_22_CLASSES: FrozenSet[int] = frozenset(range(22))
ALL_CANONICAL_CLASSES: FrozenSet[int] = frozenset(range(29))


# =============================================================================
# Canonical taxonomy — single source (P0-3, v18; extended P1-1, v18)
# =============================================================================
# AUTHORITY: ``dataset/data.yaml`` is the authority for the canonical
# taxonomy (``nc`` -> ids 0..nc-1, ``names`` -> id -> name; today nc=27 ->
# 0..26). Every resolver in the codebase reads the taxonomy through
# :func:`load_canonical_taxonomy` / :func:`canonical_names` /
# :func:`resolve_canonical_max_class_id` in this module — no other file
# may keep its own copy of the id/name table (K-01). The in-code
# ``CANONICAL_NAMES`` table below is only the FALLBACK used when
# data.yaml cannot be read or parsed (logged, never silent); when the
# file loads, its names WIN and any disagreement with the in-code table
# is warning-logged (the authority decision itself — which side wins a
# real conflict — belongs to Selçuk, contract items 23/24/26/29/31; this
# module only makes the disagreement visible instead of silently
# picking a side per file).
#
# Both ``dataset_factory/pre_training_hygiene_gate.py`` (gate default)
# and ``dataset_factory/synthetic_augmentor.py`` (range check) resolve
# through :func:`resolve_canonical_max_class_id`, so the two can never
# disagree again (K-01/K-10). ``dataset_factory/dataset_hygiene_gate.py``,
# ``core/node_quality_gate.py``, ``core/skill_bar_reader.py`` and
# ``ui/cartography_radar.py`` read their class sets / names / labels
# from here as well (P1-1).
#
# Cache: the file parse lives behind ONE lru_cache
# (:func:`load_canonical_taxonomy`). :func:`resolve_canonical_max_class_id`
# is a thin wrapper over it whose ``cache_clear``/``cache_info`` ARE the
# loader's, so a test that rewrites data.yaml mid-process invalidates
# everything with a single ``resolve_canonical_max_class_id.cache_clear()``
# call — there is no second cache that could go stale.


@dataclass(frozen=True)
class CanonicalTaxonomy:
    """Canonical taxonomy as read from the authority (``dataset/data.yaml``).

    ``nc`` is the class count (ids 0..nc-1); ``names`` maps every id in
    that range to its canonical name. ``source`` records where the data
    came from (``"data.yaml"`` or ``"fallback:KANONIK_TABLO"``) so a
    fallback is visible in logs / debugging instead of silent.
    """

    nc: int
    names: Dict[int, str]
    source: str


def _default_data_yaml_path() -> Path:
    """Repo-root ``dataset/data.yaml`` next to this package."""
    return Path(__file__).resolve().parents[1] / "dataset" / "data.yaml"


def _parse_data_yaml_taxonomy(
    text: str,
) -> Optional[Tuple[int, Dict[int, str]]]:
    """Parse ``(nc, names)`` out of data.yaml text; None when unusable.

    Prefers PyYAML; falls back to a line scan (``nc: <int>`` plus the
    ``names:`` block) so the taxonomy still loads where yaml is absent.
    Returns None when no usable ``nc`` is found or the ``names`` block
    does not cover exactly 0..nc-1 (a partial names block is worse than
    no block — the caller falls back to the in-code table and warns).
    """
    nc: Any = None
    names: Any = None
    try:
        import yaml  # type: ignore[import-not-found]  # PyYAML, repo'da kullanimda

        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            nc = loaded.get("nc")
            names = loaded.get("names")
    except Exception:
        nc = None
        names = None
    if nc is None:
        # yaml yoksa / parse edilemezse: tek satirlik `nc: <int>` tara.
        match = re.search(r"(?m)^\s*nc\s*:\s*(\d+)\s*(?:#.*)?$", text)
        if match:
            nc = int(match.group(1))
    if not isinstance(nc, int) or isinstance(nc, bool) or nc < 1:
        return None
    if names is None:
        # yaml'siz yedek: `names:` blogundaki `id: isim` satirlarini tara.
        names = {}
        in_names = False
        for line in text.splitlines():
            if re.match(r"^names\s*:\s*(?:#.*)?$", line):
                in_names = True
                continue
            if in_names:
                m = re.match(r"^\s*(\d+)\s*:\s*([A-Za-z0-9_]+)\s*(?:#.*)?$", line)
                if m:
                    names[int(m.group(1))] = m.group(2)
                    continue
                if line.strip() and not line.startswith((" ", "\t")):
                    break
        if not names:
            names = None
    if not isinstance(names, dict) or not names:
        return None
    try:
        parsed_names = {int(k): str(v) for k, v in names.items()}
    except (TypeError, ValueError):
        return None
    if set(parsed_names) != set(range(nc)):
        return None
    return nc, parsed_names


@lru_cache(maxsize=8)
def load_canonical_taxonomy(
    data_yaml_path: Optional[Union[str, Path]] = None,
) -> CanonicalTaxonomy:
    """Return the canonical taxonomy from the authority (``dataset/data.yaml``).

    ``data.yaml`` wins: when it loads, its ``names`` are returned as-is
    and any id whose name differs from the in-code fallback table is
    warning-logged. When the file is missing, unreadable, or has no
    usable ``nc``/``names``, the in-code table is returned instead
    (warning-logged, ``source="fallback:KANONIK_TABLO"``) — a missing
    file must not break the gate/augmentor. Cached: the taxonomy is a
    dataset fact, not per-call input — call
    ``resolve_canonical_max_class_id.cache_clear()`` (which clears this
    cache too) in tests that rewrite data.yaml mid-process.
    """
    fallback = CanonicalTaxonomy(
        nc=len(CANONICAL_NAMES),
        names=dict(CANONICAL_NAMES),
        source="fallback:KANONIK_TABLO",
    )
    path = Path(data_yaml_path) if data_yaml_path is not None else _default_data_yaml_path()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "P1-1: data.yaml okunamadi (%s): %s — kanonik taksonomi "
            "kod-ici yedek tabloya dusuyor (nc=%d)",
            path,
            exc,
            fallback.nc,
        )
        return fallback

    parsed = _parse_data_yaml_taxonomy(text)
    if parsed is None:
        logger.warning(
            "P1-1: data.yaml'da gecerli 'nc'/'names' bulunamadi (%s) — "
            "kanonik taksonomi kod-ici yedek tabloya dusuyor (nc=%d)",
            path,
            fallback.nc,
        )
        return fallback

    nc, names = parsed
    differing = sorted(
        cid for cid, name in names.items() if CANONICAL_NAMES.get(cid) != name
    )
    if differing:
        logger.warning(
            "P1-1: data.yaml <-> kod-ici taksonomi celisiyor (otorite "
            "data.yaml; karar dosyasi 23/24/26/29/31): farkli id'ler %s — "
            "data.yaml isimleri esas aliniyor: %s",
            differing,
            {cid: names[cid] for cid in differing},
        )
    if nc != fallback.nc:
        logger.warning(
            "P1-1: data.yaml nc=%d (ust sinir %d) kod-ici taksonomi ust "
            "siniri %d ile celisiyor — data.yaml esas aliniyor",
            nc,
            nc - 1,
            fallback.nc - 1,
        )
    return CanonicalTaxonomy(nc=nc, names=dict(names), source="data.yaml")


def resolve_canonical_max_class_id(
    data_yaml_path: Optional[Union[str, Path]] = None,
) -> int:
    """Return the canonical max class id (``nc - 1`` from ``dataset/data.yaml``).

    Thin wrapper over :func:`load_canonical_taxonomy` — the ceiling and
    the names can never disagree because they come from the same parse.
    Falls back to ``max(ALL_CANONICAL_CLASSES)`` (26) with a warning log
    when data.yaml cannot be read (see :func:`load_canonical_taxonomy`).
    Cached via the loader: call ``resolve_canonical_max_class_id.cache_clear()``
    in tests that rewrite data.yaml mid-process.
    """
    return load_canonical_taxonomy(data_yaml_path).nc - 1


# One cache behind both entry points (see the module header): clearing
# the resolver's cache clears the loader's, so no stale taxonomy can
# survive a test that rewrites data.yaml.
resolve_canonical_max_class_id.cache_clear = load_canonical_taxonomy.cache_clear  # type: ignore[attr-defined]
resolve_canonical_max_class_id.cache_info = load_canonical_taxonomy.cache_info  # type: ignore[attr-defined]


def canonical_names(
    data_yaml_path: Optional[Union[str, Path]] = None,
) -> Dict[int, str]:
    """Return the authority id -> canonical-name map (``dataset/data.yaml``).

    A copy: callers must not mutate the cached taxonomy. Falls back to
    the in-code table (logged) when data.yaml cannot be read.
    """
    return dict(load_canonical_taxonomy(data_yaml_path).names)


def canonical_name_to_id(
    data_yaml_path: Optional[Union[str, Path]] = None,
) -> Dict[str, int]:
    """Return the authority canonical-name -> id map (``dataset/data.yaml``)."""
    return {name: cid for cid, name in canonical_names(data_yaml_path).items()}


def is_valid_canonical_name(name: Any) -> bool:
    """Return True if ``name`` is a canonical class name in the authority map."""
    if not isinstance(name, str):
        return False
    return name.strip() in canonical_name_to_id()


# =============================================================================
# Internal (non-canonical) slot-state ids — separate namespace (P1-1)
# =============================================================================
# Skill-bar slot states ("empty" / "automatic") are NOT classes: there is
# no "empty" or "automatic" in data.yaml (25 is kinah, 26 is
# player_character). They used to reuse 25/26 as fake class ids, so an
# "empty"-slot exemplar was stored — and read back — as a KINAH exemplar
# (Y-3). These ids live in their own namespace, far below the canonical
# range, so no resolver, gate, or exemplar matcher can mistake them for
# a canonical class: ``is_valid_canonical_class_id(-100)`` is False,
# ``sanitize_*`` rejects them, and the exemplar memory only admits them
# on the NEGATIVE list (a rejected slot crop is suppression signal, never
# a positive prototype). -1 is NOT used: it already means "no class /
# missing" in exemplar_memory.
SLOT_STATE_EMPTY_ID: int = -100
SLOT_STATE_AUTOMATIC_ID: int = -101
SLOT_STATE_ID_TO_NAME: Dict[int, str] = {
    SLOT_STATE_EMPTY_ID: "empty",
    SLOT_STATE_AUTOMATIC_ID: "automatic",
}
INTERNAL_SLOT_STATE_IDS: FrozenSet[int] = frozenset(SLOT_STATE_ID_TO_NAME)


def is_internal_slot_state_id(value: Any) -> bool:
    """Return True if ``value`` is an internal slot-state id (not a class id)."""
    cid = _parse_class_id(value)
    if cid is None:
        return False
    return cid in INTERNAL_SLOT_STATE_IDS

# =============================================================================
# Legacy YOLO scheme -> canonical translation (F-1968 / F-0802)
# =============================================================================
# The old 23-class YOLO scheme used by YOLOAutoLabeler/Factory assigns several
# ids a different meaning than the canonical 0..26 taxonomy (6/7/8 were UI
# icons, 12 was a generic "world material", 18..22 were shifted consumables /
# bars). This map translates a legacy id to its canonical id; ids absent here
# mean the same thing in both schemes. It must ONLY be applied to annotations
# known to come from the legacy scheme -- never to already-canonical ids
# (e.g. canonical 6 is herb_edana, legacy 6 is a gather crest).
LEGACY_YOLO_CLASS_TO_CANONICAL: Dict[int, int] = {
    6: 13,   # world_gather_crest -> skill_ui_prompt
    7: 13,   # f_gather_prompt -> skill_ui_prompt
    8: 13,   # e_floating_icon / combo icon -> skill_ui_prompt
    18: 22,  # hp_potion (legacy) -> hp_potion (canonical)
    19: 23,  # mp_potion (legacy) -> mp_potion (canonical)
    20: 18,  # target_hp_bar (legacy) -> target_hp_bar (canonical)
    21: 20,  # skill_slot (legacy) -> skill_slot_ready (canonical)
    22: 24,  # buff_scroll (legacy) -> buff_scroll (canonical)
}


LEGACY_AMBIGUOUS_IDS: FrozenSet[int] = frozenset(LEGACY_YOLO_CLASS_TO_CANONICAL)
"""Ids whose meaning differs between the legacy and canonical schemes.

A bare (nameless) id in this set cannot be disambiguated on its own:
legacy 6 is a gather crest, canonical 6 is herb_edana. See
:func:`sanitize_class_id_and_name` (``scheme=``) and the P1-1 provenance
rule below.
"""

# Scheme labels for :func:`sanitize_class_id_and_name` / TXT parsing.
SCHEME_CANONICAL: str = "canonical"
SCHEME_LEGACY: str = "legacy"
SCHEME_LEGACY_YOLO: str = "legacy_yolo"
_LEGACY_SCHEMES: FrozenSet[str] = frozenset({SCHEME_LEGACY, SCHEME_LEGACY_YOLO})

# Ids already warned about once (bounded: at most len(LEGACY_AMBIGUOUS_IDS)
# entries) so a dataset full of bare herb_edana boxes logs the ambiguity
# once per id instead of once per box.
_AMBIGUOUS_BARE_ID_WARNED: Set[int] = set()


def is_legacy_ambiguous_id(class_id: Any) -> bool:
    """Return True if ``class_id`` means different things in each scheme."""
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in LEGACY_AMBIGUOUS_IDS


def _warn_ambiguous_bare_id(cid: int) -> None:
    """Log once per id that a bare legacy-ambiguous id resolved as canonical.

    P1-1 provenance rule: a bare id in ``LEGACY_AMBIGUOUS_IDS`` with no
    name is STILL resolved as canonical by default (live producers emit
    canonical ids — P1-14; pinned by test_w67
    ``sanitize(18, "") == (18, "target_hp_bar")``), but the ambiguity is
    no longer silent. Callers that KNOW the origin must pass
    ``scheme="legacy"`` (migration) or ``scheme="canonical"`` (dataset /
    live) explicitly; the default stays canonical until Selçuk decides
    contract items 23/24/26/29/31.
    """
    if cid in _AMBIGUOUS_BARE_ID_WARNED:
        return
    _AMBIGUOUS_BARE_ID_WARNED.add(cid)
    legacy_cid = LEGACY_YOLO_CLASS_TO_CANONICAL.get(cid, cid)
    logger.warning(
        "P1-1: ciplak legacy-belirsiz id %d isimsiz girdide kanonik "
        "sayildi (%d, %r) — legacy semada %r olurdu; kokeni biliniyorsa "
        "scheme='legacy'/'canonical' belirtin (sozlesme 23/24/26/29/31 "
        "karari bekleniyor)",
        cid,
        cid,
        CANONICAL_NAMES.get(cid),
        CANONICAL_NAMES.get(legacy_cid),
    )


def translate_legacy_class_id(class_id: Any, *, provenance: str) -> int:
    """Translate a legacy-scheme class id to its canonical id (strict).

    P1-1 provenance gate: the ``provenance`` label is MANDATORY —
    ``provenance`` must be ``"legacy"`` or ``"legacy_yolo"`` (the input is
    explicitly marked legacy-scheme data, e.g. a pre-F-0824 on-disk review
    JSON / TXT set being migrated, with its origin recorded). Any other
    value (including a missing one) raises ``ValueError`` and logs: a
    provenance-less legacy input is REJECTED, never silently translated
    and never silently read as canonical. Ids absent from
    ``LEGACY_YOLO_CLASS_TO_CANONICAL`` mean the same in both schemes and
    pass through unchanged (after canonical validation).

    Raises:
        ValueError: unparseable id, missing/wrong provenance, or a
            translated id outside the canonical range.
    """
    if provenance not in _LEGACY_SCHEMES:
        logger.warning(
            "P1-1: legacy ceviri reddedildi — koken etiketi yok/yanlis "
            "(provenance=%r, class_id=%r); beklenen 'legacy'/'legacy_yolo'",
            provenance,
            class_id,
        )
        raise ValueError(
            "Legacy class translation requires provenance='legacy' "
            f"(got provenance={provenance!r}, class_id={class_id!r})"
        )
    cid = _parse_class_id(class_id)
    if cid is None:
        raise ValueError(f"Legacy class id parse edilemedi: {class_id!r}")
    translated = LEGACY_YOLO_CLASS_TO_CANONICAL.get(cid, cid)
    if translated not in CANONICAL_NAMES:
        raise ValueError(
            f"Legacy class id {cid!r} kanonik aralik disina cevriliyor: {translated!r}"
        )
    return translated


def translate_legacy_yolo_class_id(class_id: Any) -> Optional[int]:
    """Translate a legacy-scheme class id to its canonical id (None if unparseable).

    P1-14 (K-36): MIGRATION-ONLY helper — it must NEVER be applied to live
    annotations. Both live producers (``YOLOAutoLabeler`` and the curator
    fallback labeler) emit canonical ids (0..26) today, so blindly
    translating a live id corrupts it (canonical 6 herb_edana would become
    13 skill_ui_prompt). Only data EXPLICITLY marked as legacy scheme
    (e.g. a pre-F-0824 on-disk review JSON / TXT set being migrated, with
    its origin recorded) may pass through this map; a bare id with no
    provenance is canonical by definition. Named legacy boxes do not need
    this map either: ``sanitize_class_id_and_name`` resolves them by name
    (``crest`` -> 13, ``hp_potion`` -> 22).

    P1-1: this stays as the backward-compatible entry point (its bare-id
    behaviour is pinned by tests); new code must use the provenance-gated
    :func:`translate_legacy_class_id` instead.
    """
    cid = _parse_class_id(class_id)
    if cid is None:
        return None
    return LEGACY_YOLO_CLASS_TO_CANONICAL.get(cid, cid)


# Precompiled once at import: _parse_class_id() is called on every predicate /
# sanitize call (hot path for YOLO/review pipelines), so the digit pattern
# must not be re-compiled (or re-looked-up in the re cache) per call.
_DIGIT_RE = re.compile(r"[+-]?\d+")


def _parse_class_id(value: Any) -> Optional[int]:
    """Strictly parse a class id.

    Accepts real integers (incl. ``numbers.Integral`` such as numpy ints) and
    digit-only strings (``"5"``, ``" 12 "``). Rejects booleans (``int(True) == 1``
    would silently turn ``True`` into class 1), floats (``5.0``) and float-like
    strings (``"5.0"``). Returns ``None`` when the value is not a class id.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if _DIGIT_RE.fullmatch(text):
            try:
                return int(text)
            except ValueError:
                return None
    return None

# =============================================================================
# Internal Caches and Build System
# =============================================================================

_NAMES: Dict[int, str] = {}
_LABELS_TR: Dict[int, str] = {}
_ALIASES: Dict[str, int] = {}
_RESOURCE_NAMES: Dict[int, str] = {}
_READY = False
# F4: catalog specs that could NOT be mapped onto a canonical class (no
# name match and no generic class for their category). They get NO alias —
# a wrong specific class is worse than an unmapped name — and are listed
# here for the taxonomy decision (do Lura / Miana / Iron Ore / ... exist
# in Aion 2, and if so, under which canonical id?). Populated by
# _build_taxonomy_locked(); empty until the first build runs.
# v19 F4 verdict (2026-11-11, tests/test_f4_taxonomy_unmapped.py pins it):
# 0 of the 74 CAN be assigned — data.yaml (nc=27) has no generic herb
# class (only 0/1/2/6/7 specific herbs), no generic ore class (only
# 3 orichalcum, 8 obsidian), no cooking/fiber class and no odyle class
# (Herbs 36 / Ores & Metals 21 / Cooking & Fibers 13 / Odyle & Aether 4).
# Assigning any of them would be a blind translation (P1-14). The only
# honest path is a taxonomy extension — new generic classes in
# dataset/data.yaml, which changes nc and the model — a Selçuk decision,
# NOT taken here; data.yaml is untouched. The warning below stays
# (log-greppable) until that decision lands.
UNMAPPED_CATALOG_MATERIALS: List[str] = []
# Guards the lazy first build: without it two threads can observe the alias
# tables half-built (or build them twice). See _build_taxonomy().
_BUILD_LOCK = threading.Lock()


@lru_cache(maxsize=4096)
def _slug(text: str) -> str:
    """Produce a normalized lowercase ASCII slug from text.

    Pure function of its input, and the same strings recur constantly
    (canonical names, catalog base materials shared by many essences, alias
    keys re-slugged on registration), so results are cached. The cache is
    bounded (4096 entries) so arbitrary label strings from data files cannot
    grow it without limit.
    """
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
    """Build the alias/resource tables once, lazily (thread-safe).

    Double-checked locking: the common path (already built) is a single flag
    read with no lock acquisition; concurrent first calls serialize on
    ``_BUILD_LOCK`` so the tables are never observed half-built and the
    143-entry catalog walk never runs twice.
    """
    global _READY
    if _READY:
        return
    with _BUILD_LOCK:
        if _READY:
            return
        _build_taxonomy_locked()
        _READY = True


def _build_taxonomy_locked() -> None:
    """Populate _NAMES/_LABELS_TR/_ALIASES/_RESOURCE_NAMES (caller holds _BUILD_LOCK).

    P1-1: the name table is seeded from the authority
    (:func:`canonical_names` — ``dataset/data.yaml``), not from the
    in-code fallback, so the alias table (Priority 2 in
    :func:`sanitize_class_id_and_name`) can never disagree with the
    dataset file. Labels / palettes are display data with no data.yaml
    counterpart, so they stay in-code, keyed by id.
    """
    authority_names = canonical_names()
    # S201 (v19/P3-8): bu fonksiyon tek başına idempotent değildi —
    # UNMAPPED_CATALOG_MATERIALS.extend() ikinci çalışmada listeyi
    # ikiye katlıyordu (dış _READY bayrağı koruyordu ama fonksiyonun
    # kendisi tekrar-çalıştırmaya güvenli değildi). Baştan temizlenir;
    # _ALIASES/_NAMES/_LABELS_TR dict-update ile zaten idempotent.
    del UNMAPPED_CATALOG_MATERIALS[:]
    _NAMES.update(authority_names)
    _LABELS_TR.update(CANONICAL_LABELS_TR)

    # 1. Register canonical names and labels
    for cid, name in authority_names.items():
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
            27: "Other Ore",
            28: "Other Odyle",
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
        "e_gather_icon": 13,
        "e_gather": 13,
        "e_floating_icon": 13,
        "combo_skill_prompt": 13,
        "combo_icon": 13,
        "q_combo_icon": 13,
        "e_combo_icon": 13,
        "kombo": 13,
        "kombo_ikonu": 13,
        "combo": 13,
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
        # P1-14 (F4): "ore_iron_silver" was removed — an iron/silver ore is
        # NOT orichalcum (21 ores are not orichalcum); it must stay unmapped.
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
        # P1-1: "yetenek" (Turkish for skill) used to resolve via the
        # is_skill_element() substring tail; that tail is gone (single
        # resolver), so the word lives here now, pointing at the same
        # class the old heuristic implied (20 skill_slot_ready).
        "yetenek": 20,
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

    # 4. Map entire Aion 2 catalog of 140+ essence items onto the canonical classes (0..26).
    # STRICT MANDATE: NEVER assign IDs >= 25 to catalog essences!
    # F4: a category fallback may only point at a GENERIC class (Gemstones
    # -> 4 gem_crystal, Logs & Trees -> 5 tree_log). It must NEVER assert a
    # SPECIFIC material identity: 36 real herbs (Lura, Miana, Pella, ...)
    # are not targena, 21 ores (Iron, Copper, Silver, ...) are not
    # orichalcum, cooking ingredients are not mela, odyle is not a gem.
    # Specs with no name match and no generic class get NO alias at all
    # (they resolve as unknown instead of as a false targena/orichalcum)
    # and are collected in UNMAPPED_CATALOG_MATERIALS for the taxonomy
    # decision. The old `else -> 2` branch never fired for any current
    # spec, but it is removed too — an unmapped spec must stay unmapped.
    # v19 F4 verdict (2026-11-11): the decision is RECORDED, not pending —
    # 0/74 assignable within nc=27 (see the UNMAPPED_CATALOG_MATERIALS
    # comment above and tests/test_f4_taxonomy_unmapped.py, which pins
    # the exact 74 names + groups). Only a data.yaml taxonomy extension
    # (Selçuk decision) can change this; until then the warning below is
    # the honest state, not an open TODO to guess from.
    _unmapped_specs: List[str] = []
    for spec in AION2_MATERIALS_CATALOG:
        name_lower = str(spec.name).lower()
        base_lower = str(spec.base_material).lower()
        cat = spec.category

        # Determine target canonical class ID (None = no honest mapping)
        if "aria" in base_lower or "aria" in name_lower:
            cid: Optional[int] = 0
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
        elif cat == "Gemstones":
            cid = 4
        elif cat == "Logs & Trees":
            cid = 5
        elif str(spec.name) in F4_LIVE_GENERIC_MATERIAL_MAP:
            cid = F4_LIVE_GENERIC_MATERIAL_MAP[str(spec.name)]
        else:
            # No name match and no generic class for this category (Herbs,
            # Ores & Metals, Cooking & Fibers, Odyle & Aether, or an
            # unknown category): do NOT guess a specific material.
            cid = None

        if cid is None:
            _unmapped_specs.append(str(spec.name))
            continue

        # Register aliases for catalog entries
        base_slug = _slug(spec.base_material)
        name_slug = _slug(spec.name)
        _add_alias(f"essence_{base_slug}", cid)
        _add_alias(f"essence_{name_slug}", cid)
        _add_alias(spec.base_material, cid)
        _add_alias(spec.name, cid)
        _add_alias(spec.name_tr, cid)

    if _unmapped_specs:
        UNMAPPED_CATALOG_MATERIALS.extend(_unmapped_specs)
        logger.warning(
            "F4: %d catalog specs have no canonical class and got no alias "
            "(they no longer resolve as targena/orichalcum/mela); taxonomy "
            "decision pending: %s",
            len(_unmapped_specs),
            ", ".join(_unmapped_specs),
        )


# =============================================================================
# Validation and Query Predicates
# =============================================================================

def is_valid_canonical_class_id(class_id: Any) -> bool:
    """Return True if class_id is an integer in the canonical range.

    P1-1: the range comes from the authority
    (:func:`load_canonical_taxonomy` — ``dataset/data.yaml`` ``nc``),
    not from the hardcoded ``range(27)`` — so a dataset whose ``nc``
    changes cannot silently disagree with the gate/augmentor ceiling.
    Today nc=27, so this is exactly 0..26.
    """
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in load_canonical_taxonomy().names


def is_valid_core_22_class_id(class_id: Any) -> bool:
    """Return True if class_id is an integer in the core 22 classes (0..21)."""
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in CORE_22_CLASSES


def is_3d_world_material(class_id: Any) -> bool:
    """Return True if class_id represents a 3D gatherable world resource (0..5, 6..8, 12)."""
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in MATERIAL_3D_CLASSES


def is_minimap_symbol(class_id: Any) -> bool:
    """Return True if class_id represents a minimap symbol (9: sprout, 11: monster dot)."""
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in MINIMAP_SYMBOL_CLASSES


def is_monster_element(class_id: Any) -> bool:
    """Return True if class_id represents a 3D monster (10: monster_3d).

    Class 11 (minimap_monster_dot) is a *minimap symbol* of a monster, not a
    monster element itself: ``get_super_category(11) == "minimap"`` and
    ``is_minimap_symbol(11)`` is True, so it must not also count as a monster
    element (previously ``is_monster_element(11)`` was True, contradicting the
    super-category). Use :func:`is_minimap_symbol` for class 11.
    """
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in MONSTER_CLASSES


def is_navigable_element(class_id: Any) -> bool:
    """Return True if class_id represents terrain navigation (14: obstacle, 15: path_ground)."""
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in NAVIGABLE_ELEMENT_CLASSES


def is_consumable(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name represents a consumable item (hp_potion, mp_potion, buff_scroll).

    P1-1: the name path resolves through the single resolver
    (:func:`try_sanitize_class_id_and_name`) — the old substring tail
    (``"potion" in text`` ...) is gone, so an unresolvable name is
    NOT a consumable instead of guessing one.
    """
    cid = _parse_class_id(class_id_or_name)
    if cid is not None:
        return cid in CONSUMABLE_CLASSES

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in CONSUMABLE_CLASSES
    return False


def is_skill_element(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name is a skill slot or skill bar element (strictly separated from consumables).

    P1-1: the name path resolves through the single resolver
    (:func:`try_sanitize_class_id_and_name`); the old substring tail
    (``"skill" in text`` matched ``"unskilled"`` too) is gone. The
    Turkish ``"yetenek"`` still resolves — via the alias table
    (``"yetenek" -> 20``), not a substring.
    """
    cid = _parse_class_id(class_id_or_name)
    if cid is not None:
        return cid in HUD_SKILL_CLASSES

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in HUD_SKILL_CLASSES
    return False


def is_currency(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name represents currency / Kinah (class 25).

    P1-1: the name path resolves through the single resolver
    (:func:`try_sanitize_class_id_and_name`); the old substring tail
    (``"gold" in text``) made ``"gold ore"`` kinah, undoing the F4
    decision that an ore is not currency. P1-8 keeps a token-based
    fallback (whole tokens only, no substrings): an unmapped compound
    such as ``"gold ore"`` stays unmapped instead of becoming currency.
    """
    cid = _parse_class_id(class_id_or_name)
    if cid is not None:
        return cid in CURRENCY_CLASSES

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in CURRENCY_CLASSES
    # P1-8 (K-21): token-based, mirroring sanitize Priority 5 (F4).
    # Substring matching resurrected the F4 decision this file documents:
    # "gold ore" contains "gold", so an ore was classified as Kinah.
    # Bare "gold" already resolves to kinah via the alias table
    # (Priority 2) before this fallback runs, so "gold" is deliberately
    # NOT a fallback token — only whole tokens that are unambiguously
    # currency count. An unmapped compound ("gold ore", "gold coins")
    # stays unmapped instead of becoming currency.
    tokens = set(_slug(text).split("_"))
    return bool(tokens & {"kinah", "kina", "altin", "coin", "para", "currency"})


def is_hud_element(class_id: Any) -> bool:
    """Return True if class_id represents a HUD element (vitals, skills, prompts, consumables, currency, or player)."""
    cid = _parse_class_id(class_id)
    if cid is None:
        return False
    return cid in HUD_ELEMENT_CLASSES


def is_player_character(class_id_or_name: Any) -> bool:
    """Return True if class_id_or_name represents the player character (26).

    P1-1: the name path resolves through the single resolver
    (:func:`try_sanitize_class_id_and_name`); the old substring tail is
    gone, so an unresolvable name is NOT the player character.
    """
    cid = _parse_class_id(class_id_or_name)
    if cid is not None:
        return cid in PLAYER_CLASSES

    text = str(class_id_or_name or "").strip().lower()
    if not text:
        return False
    sanitized = try_sanitize_class_id_and_name(class_id=-1, class_name=text)
    if sanitized is not None:
        return sanitized[0] in PLAYER_CLASSES
    return False


def get_super_category(class_id_or_name: Any) -> Optional[str]:
    """Return the canonical super-category name for a class id or class name."""
    cid = _parse_class_id(class_id_or_name)
    if cid is None:
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


def is_same_material_family(class_id_a: Any, class_id_b: Any) -> bool:
    """Return True if two 3D-material classes belong to the same family.

    v20-loop: repair kapısının ikinci kilidi. Süper-kategori tek
    başına yeterli değildi — herb_targena (2) ile ore_orichalcum (3)
    aynı "3d_material" kabında olduğu için repair bitkiyi cevhere
    çeviriyordu (canlı 16:56:00, sim=0.89). Aileler: bitkiler
    {0,1,2,6,7}, cevherler {3,8}, mücevherler {4,12}, ağaç {5}.
    3D materyal olmayan çiftlerde kilit uygulanmaz (True döner) —
    o sınıfları süper-kategori kapısı yönetir.
    """
    try:
        a = int(class_id_a)
        b = int(class_id_b)
    except (TypeError, ValueError):
        return False
    if a not in MATERIAL_3D_CLASSES or b not in MATERIAL_3D_CLASSES:
        return True
    for family in MATERIAL_FAMILIES:
        if a in family and b in family:
            return True
    return False


# =============================================================================
# Sanitization & Anti-Corruption Engine
# =============================================================================

def sanitize_class_id_and_name(
    class_id: Any,
    class_name: Optional[str] = "",
    allow_fallback: bool = False,
    default_fallback: Optional[int] = None,
    scheme: Optional[str] = None,
) -> Tuple[int, str]:
    """Validate and sanitize a class ID and class name against canonical taxonomy.

    Resolves:
    - Canonical names and labels (e.g. 'herb_aria', 'Aria' -> (0, 'herb_aria'))
    - Legacy aliases (e.g. 'wood_tree' -> (5, 'tree_log'))
    - Catalog essence slugs (e.g. 'essence_yggdrasil_log' -> (5, 'tree_log'))
    - Legacy class index collisions (e.g. class_id: 18 with class_name: 'hp_potion' -> (22, 'hp_potion'))
    - Consumable scrolls and potions (e.g. 'buff_scroll' -> (24, 'buff_scroll'))

    ``scheme`` (P1-1 provenance rule) declares the input's origin:
    ``"canonical"`` (dataset / live producers — ids mean what data.yaml
    says) or ``"legacy"`` / ``"legacy_yolo"`` (old-scheme data being
    migrated — a bare id in ``LEGACY_AMBIGUOUS_IDS`` is translated via
    ``LEGACY_YOLO_CLASS_TO_CANONICAL`` instead of being read as its
    canonical namesake). ``None`` (default) keeps the live default: a
    bare id is canonical (P1-14; pinned by test_w67
    ``sanitize(18, "") == (18, "target_hp_bar")``) — but a bare
    legacy-ambiguous id is now WARNING-logged once per id instead of
    silently resolved, and named input always lets the NAME win over
    either scheme. The default stays canonical until Selçuk decides
    contract items 23/24/26/29/31; only the mechanism (explicit scheme
    + visible ambiguity) is built here, no side of the conflict is
    picked.

    Raises:
        ValueError: If neither class_id nor class_name can be resolved to a canonical
                    class (0..26) and allow_fallback is False; also for an
                    unknown ``scheme`` value.

    Uyarı (B-11): allow_fallback=True + default_fallback verilirse çözülemeyen
    her girdi sorgusuz o sınıfa çevrilir (warning loglanır). Etiket verisinde
    kullanmayın — bilinmeyen sınıf sessizce yanlış sınıfa dönmesin diye
    varsayılan False'tur (hata fırlatır).
    """
    if scheme is not None and scheme not in (SCHEME_CANONICAL, *_LEGACY_SCHEMES):
        raise ValueError(
            f"Bilinmeyen taksonomi scheme: {scheme!r} "
            f"(beklenen 'canonical'/'legacy'/'legacy_yolo')"
        )
    _build_taxonomy()

    clean_name = str(class_name or "").strip()
    norm_name = " ".join(clean_name.lower().split()) if clean_name else ""

    # Parse raw integer class_id strictly: bools, floats and float-like strings
    # ("5.0") are NOT class ids (int(True) == 1 and int(5.0) == 5 used to slip
    # through and silently resolve to the wrong class).
    raw_cid: Optional[int] = _parse_class_id(class_id)

    # P1-1: bare legacy id with a declared legacy scheme — translate it
    # here, before the fast path can read it as its canonical namesake
    # (legacy 6 is a crest, not herb_edana). A NAME, when present, is
    # stronger evidence than the scheme and runs through the priorities
    # below instead.
    if (
        scheme in _LEGACY_SCHEMES
        and raw_cid is not None
        and raw_cid in LEGACY_AMBIGUOUS_IDS
        and not norm_name
    ):
        translated = LEGACY_YOLO_CLASS_TO_CANONICAL[raw_cid]
        return translated, CANONICAL_NAMES[translated]

    # Fast path (hot path for YOLO/review pipelines, which mostly pass an int
    # id with no name or with its own canonical name): resolves without slug
    # computation or alias lookups. Name-vs-id precedence is preserved -- any
    # other name falls through to the priority chain below, where the name
    # still wins over the id (Priority 1/2) and legacy collisions still apply
    # (Priority 3).
    if raw_cid is not None and raw_cid in CANONICAL_NAMES:
        if not norm_name or norm_name == CANONICAL_NAMES[raw_cid]:
            if not norm_name and raw_cid in LEGACY_AMBIGUOUS_IDS:
                # P1-1: bare legacy-ambiguous id, no declared scheme —
                # resolved as canonical (live default) but NOT silently.
                _warn_ambiguous_bare_id(raw_cid)
            return raw_cid, CANONICAL_NAMES[raw_cid]

    slug_name = _slug(norm_name) if norm_name else ""

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
        if scheme in _LEGACY_SCHEMES and raw_cid in LEGACY_AMBIGUOUS_IDS:
            # Named input whose name did not resolve anywhere above: the
            # id is the only evidence left, and its declared scheme is
            # legacy — translate instead of reading the canonical namesake.
            translated = LEGACY_YOLO_CLASS_TO_CANONICAL[raw_cid]
            return translated, CANONICAL_NAMES[translated]
        if not norm_name and raw_cid in LEGACY_AMBIGUOUS_IDS:
            # P1-1: bare legacy-ambiguous id, no declared scheme — same
            # live default (canonical) and same once-per-id warning as
            # the fast path above.
            _warn_ambiguous_bare_id(raw_cid)
        return raw_cid, CANONICAL_NAMES[raw_cid]

    # Priority 5: Keyword-based heuristic resolution.
    # Token-based on purpose: substring matching ("skill" in name, "yuva" in
    # name, "mela" in "caramel", ...) used to mint wrong classes out of
    # unrelated words. Only whole underscore/space-separated tokens count, so
    # e.g. "unskilled" or "yuva" no longer resolve to a skill slot.
    if norm_name:
        tokens = set(slug_name.split("_")) if slug_name else set()
        if "aria" in tokens:
            return 0, CANONICAL_NAMES[0]
        if "mela" in tokens:
            return 1, CANONICAL_NAMES[1]
        if tokens & {"targena", "azpha", "calendula"}:
            return 2, CANONICAL_NAMES[2]
        if "orichalcum" in tokens:
            return 3, CANONICAL_NAMES[3]
        if "edana" in tokens:
            return 6, CANONICAL_NAMES[6]
        if "lumina" in tokens:
            return 7, CANONICAL_NAMES[7]
        if "obsidian" in tokens:
            return 8, CANONICAL_NAMES[8]
        if "blood" in tokens and "crystal" in tokens:
            return 12, CANONICAL_NAMES[12]
        if tokens & {"gem", "gemstone", "sapphire", "ruby", "diamond"}:
            return 4, CANONICAL_NAMES[4]
        if tokens & {"tree", "log", "wood", "timber", "yggdrasil", "asvata"}:
            return 5, CANONICAL_NAMES[5]
        if tokens & {"sprout", "filiz"}:
            return 9, CANONICAL_NAMES[9]
        if tokens & {"monster", "canavar"}:
            return (11, CANONICAL_NAMES[11]) if tokens & {"minimap", "dot"} else (10, CANONICAL_NAMES[10])
        if tokens & {"potion", "iksir"}:
            return (22, CANONICAL_NAMES[22]) if tokens & {"hp", "can"} else (23, CANONICAL_NAMES[23])
        if tokens & {"scroll", "parşömen", "parsomen"}:
            return 24, CANONICAL_NAMES[24]
        if "hp_bar" in slug_name or "can_bar" in slug_name:
            return (18, CANONICAL_NAMES[18]) if tokens & {"target", "hedef"} else (16, CANONICAL_NAMES[16])
        if "mp_bar" in slug_name or "mana_bar" in slug_name:
            return 17, CANONICAL_NAMES[17]
        if "skill_bar" in slug_name:
            return 19, CANONICAL_NAMES[19]
        if tokens & {"prompt", "crest", "icon"}:
            return 13, CANONICAL_NAMES[13]
        if "skill" in tokens:
            return (21, CANONICAL_NAMES[21]) if tokens & {"cd", "cooldown", "bekleme"} else (20, CANONICAL_NAMES[20])
        if tokens & {"obstacle", "engel"}:
            return 14, CANONICAL_NAMES[14]
        if tokens & {"path", "ground", "yol", "zemin"}:
            return 15, CANONICAL_NAMES[15]
        # F4: "gold" is deliberately NOT in this set. Bare "gold" already
        # resolves to kinah via the alias table (Priority 2), but the token
        # also matched compound material names — "Gold Ore" fell through to
        # kinah once its (wrong) orichalcum alias was removed. An ore is
        # not currency; an unmapped "Gold Ore" must stay unmapped.
        if tokens & {"kinah", "kina", "altin", "coin", "para", "currency"}:
            return 25, CANONICAL_NAMES[25]

    # Priority 6: Fallback or Error
    # B-11 koruması: yedek mekanizması duruyor ama artık SESSİZ değil.
    # Çözülemeyen girdi varsayılan sınıfa çevrildiğinde warning loglanır —
    # yarın biri default_fallback=10 (canavar) geçirirse çözülemeyen her
    # etiket sessizce canavar olmaz; logda izi kalır. Etiket verisi için
    # allow_fallback=True kullanılmamalı (varsayılan False = hata fırlatır);
    # üretimde hiçbir çağıran bu dalı kullanmıyor (yalnızca testler).
    if allow_fallback and default_fallback is not None:
        fallback_cid = _parse_class_id(default_fallback)
        if fallback_cid is not None and fallback_cid in CANONICAL_NAMES:
            logger.warning(
                "Sınıf yedeği tetiklendi: çözülemeyen girdi "
                "class_id=%r, class_name=%r varsayılan sınıfa çevrildi "
                "(%d, %r) — yanlış sınıf ataması olabilir, girdiyi düzeltin.",
                class_id, class_name, fallback_cid, CANONICAL_NAMES[fallback_cid],
            )
            return fallback_cid, CANONICAL_NAMES[fallback_cid]

    raise ValueError(
        f"Non-canonical class cannot be resolved: class_id={class_id!r}, class_name={class_name!r}"
    )


def try_sanitize_class_id_and_name(
    class_id: Any,
    class_name: Optional[str] = "",
    scheme: Optional[str] = None,
) -> Optional[Tuple[int, str]]:
    """Safely sanitize class_id and class_name. Returns None if unresolvable.

    ``scheme`` is passed through to :func:`sanitize_class_id_and_name`
    (P1-1 provenance rule — see there).
    """
    try:
        return sanitize_class_id_and_name(class_id, class_name, scheme=scheme)
    except (TypeError, ValueError):
        return None


def sanitize_yolo_box(
    box: Dict[str, Any],
    strict_drop_non_canonical: bool = True,
) -> Optional[Dict[str, Any]]:
    """Sanitize a bounding box dict in-place or via copy.

    Enforces that 'class_id' is in 0..26 and 'class_name' matches canonical name.
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
    Out-of-range coordinates are REJECTED (None), never silently clamped: a box
    with cx > 1.0 is corrupt input, and clamping it would move the box onto a
    different object in the training data.
    """
    try:
        sanitized_id, _ = sanitize_class_id_and_name(class_id, class_name)
    except (TypeError, ValueError):
        return None
    # No separate is_valid_canonical_class_id() check here: every return path
    # of sanitize_class_id_and_name() yields a member of CANONICAL_NAMES, so
    # sanitized_id is canonical by construction (the extra check was dead).

    try:
        fcx, fcy = float(cx), float(cy)
        fw, fh = float(w), float(h)
    except (TypeError, ValueError):
        return None

    if not (0.0 <= fcx <= 1.0 and 0.0 <= fcy <= 1.0):
        return None
    if not (0.0 < fw <= 1.0 and 0.0 < fh <= 1.0):
        return None
    if fw <= 1e-6 or fh <= 1e-6:
        return None

    return f"{sanitized_id} {fcx:.6f} {fcy:.6f} {fw:.6f} {fh:.6f}"


def parse_and_sanitize_yolo_txt_line(
    line: str,
    scheme: str = SCHEME_CANONICAL,
) -> Optional[Tuple[int, float, float, float, float, str]]:
    """Parse a YOLO txt line, sanitize class_id, and return (canonical_id, cx, cy, w, h, name).

    Returns None if line is empty, comment, malformed, has an unresolvable class ID,
    or has out-of-range / degenerate coordinates (cx, cy must be in [0, 1], w, h
    in (0, 1]). Coordinates are validated, never silently accepted.

    ``scheme`` (P1-1 provenance rule): TXT lines in the dataset are
    canonical by construction, so the default is ``"canonical"`` —
    declared here, not silently assumed. Pass ``"legacy"`` only when
    parsing a known legacy-scheme TXT set (migration); a bare legacy
    id then translates via ``LEGACY_YOLO_CLASS_TO_CANONICAL`` instead
    of being read as its canonical namesake.
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

    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return None

    # Fast path: sanitize(raw_cid) with no name resolves a canonical id to
    # itself via Priority 4, so a direct membership test is equivalent and
    # skips the alias/slug machinery for every parsed line. It only runs
    # for the declared-canonical scheme: under a legacy scheme a bare
    # legacy-ambiguous id must translate, not pass through as canonical.
    if raw_cid in CANONICAL_NAMES and not (
        scheme in _LEGACY_SCHEMES and raw_cid in LEGACY_AMBIGUOUS_IDS
    ):
        return (raw_cid, cx, cy, w, h, CANONICAL_NAMES[raw_cid])
    sanitized = try_sanitize_class_id_and_name(raw_cid, scheme=scheme)
    if sanitized is None:
        return None
    cid, cname = sanitized
    return (cid, cx, cy, w, h, cname)


# =============================================================================
# Backward Compatibility & Export Functions
# =============================================================================

def dataset_class_names() -> Dict[int, str]:
    """Return dictionary of all canonical class IDs and names (0..26).

    P1-1: read from the authority (:func:`canonical_names` —
    ``dataset/data.yaml``), not from the alias-built table.
    """
    return canonical_names()


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
