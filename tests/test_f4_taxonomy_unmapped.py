# -*- coding: utf-8 -*-
"""tests/test_f4_taxonomy_unmapped.py
=====================================
v19 F4 — the 74 catalog specs with no canonical class (P1-14).

Verdict (2026-11-11, reproduced from the v18 smoke logs AND the live
build — the exact 74-name list matches
``~/workspace/a2-bot-audit/bulk-onay-purge-only2.log`` line 6):

* **0 of 74 can be assigned a canonical class.** ``dataset/data.yaml``
  (``nc=27``) has NO generic herb class (only 0 herb_aria, 1 herb_mela,
  2 herb_targena, 6 herb_edana, 7 herb_lumina), NO generic ore class
  (only 3 ore_orichalcum, 8 ore_obsidian), NO cooking/fiber class and
  NO odyle class. Assigning any of the 74 to one of the 27 would assert
  a false specific identity — exactly what P1-14 forbids ("kör çeviri
  YOK", "36 herbs are not targena, 21 ores are not orichalcum, odyle
  is not a gem, cooking ingredients are not mela").
* Breakdown of the 74 (by catalog category): Herbs 36, Ores & Metals 21,
  Cooking & Fibers 13, Odyle & Aether 4.
* The ONLY way to give these items a canonical class is a taxonomy
  extension — new generic classes in ``dataset/data.yaml`` (which
  changes ``nc`` and the model). That is a Selçuk decision; it is NOT
  done here and ``data.yaml`` is untouched by this change.

This file pins the verdict so a future catalog / mapping change fails
loudly instead of silently re-mapping (or un-mapping) an item: update
``F4_UNMAPPED`` deliberately, with a taxonomy decision, if the catalog
or the authority ever changes.

Related fix pinned here: ``dataset_factory/yolo_auto_labeler.py`` used
to map "Withered Branch" -> 5 (tree_log) through its ``"branch"``
substring keyword — the only catalog item containing "branch", and it
is an Alchemy herb, not a log. The keyword is removed; all 74 now fall
through to the generic world-material bucket (12) in the labeler.
"""

import re
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.materials_db import AION2_MATERIALS_CATALOG
from dataset_factory.class_taxonomy import (
    UNMAPPED_CATALOG_MATERIALS,
    alias_map,
    canonical_names,
    class_id_for_material_text,
    is_consumable,
    is_currency,
    is_player_character,
    is_skill_element,
    load_canonical_taxonomy,
    try_sanitize_class_id_and_name,
)
from dataset_factory.yolo_auto_labeler import map_material_name_to_yolo_class_id

# The 74 catalog specs with no canonical class (F4 warning, v18 smoke
# logs). Grouped by catalog category; the grouping is the reason no
# honest mapping exists for that group.
F4_UNMAPPED_HERBS = [
    "Lura", "Gravi's Root", "Purifying Forest Large Leaf", "Withered Branch",
    "Miana", "Pella", "Rina", "Mandragora", "Atri", "Kalgelly", "Lolo",
    "Nezekan's Tear", "Marle", "Amilis", "Valerian", "Caprauna", "Afari",
    "Lumesia", "Wiki", "Theonia", "Senor", "Jai", "Ratal", "Blicora",
    "Xilix", "Krotan", "Pyre", "Diaphanous Herb", "Penac", "Ormea", "Hasia",
    "Oriel", "Hanate", "Atreian Herb", "Lunime", "Cenna",
]
F4_UNMAPPED_ORES = [
    "Iron Ore", "Copper Ore", "Silver Ore", "Gold Ore", "Platinum Ore",
    "Titanium Ore", "Adamantium", "Starmetal", "Mithril Ore", "Thallium",
    "Dark Iron", "Boon Ore", "Hamerium", "Lyrion", "Blue Titanium",
    "Silicate", "Magma Ore", "Katalium", "Anarium", "Ceramium", "Gravi",
]
F4_UNMAPPED_COOKING = [
    "Conide", "Coriolus", "Cypri", "Inina", "Kukuru", "Morfa", "Ohkra",
    "Pujery", "Raydam", "Ruko", "Tikel", "Briga", "Liguri",
]
F4_UNMAPPED_ODYLE = [
    "Odyle", "Fine Odyle", "Pure Odyle", "Radiant Odyle",
]

F4_UNMAPPED = (
    F4_UNMAPPED_HERBS + F4_UNMAPPED_ORES + F4_UNMAPPED_COOKING + F4_UNMAPPED_ODYLE
)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _build_and_collect():
    # alias_map() triggers the lazy taxonomy build.
    alias_map()
    return list(UNMAPPED_CATALOG_MATERIALS)


def test_f4_unmapped_set_is_exactly_the_74():
    """v20: 74 katalog spec'in 72'si hâlâ UNMAPPED; canlı kanıtlı 2 si jenerik."""
    from dataset_factory.class_taxonomy import F4_LIVE_GENERIC_MATERIAL_MAP

    assert len(F4_UNMAPPED) == 74
    assert len(set(F4_UNMAPPED)) == 74
    live = set(F4_LIVE_GENERIC_MATERIAL_MAP.keys())
    assert live <= set(F4_UNMAPPED)
    expected_unmapped = sorted(set(F4_UNMAPPED) - live)
    assert sorted(_build_and_collect()) == sorted(expected_unmapped)


def test_f4_unmapped_group_counts_match_catalog_categories():
    """Each group is exactly the unmapped remainder of its catalog category."""
    by_name = {spec.name: spec for spec in AION2_MATERIALS_CATALOG}
    # 147, not 143: materials_db.py defines 143 static specs; importing
    # dataset_factory.yolo_auto_labeler pulls in core.material_scanner,
    # whose module-level ensure_aion2_materials_catalog_specs() registers
    # the 4 remaining main materials (Herb Edana, Herb Lumina, Obsidian
    # Ore, Blood Crystal — canonical classes 6, 7, 8, 12) at import time.
    # All 4 are mapped, so the unmapped 74 below is unaffected.
    assert len(AION2_MATERIALS_CATALOG) == 147
    assert len(by_name) == 147
    groups = {
        "Herbs": F4_UNMAPPED_HERBS,
        "Ores & Metals": F4_UNMAPPED_ORES,
        "Cooking & Fibers": F4_UNMAPPED_COOKING,
        "Odyle & Aether": F4_UNMAPPED_ODYLE,
    }
    for category, names in groups.items():
        assert names, category
        for name in names:
            assert by_name[name].category == category, name
    # Every other catalog spec of these categories IS mapped (73 mapped).
    mapped = [s for s in AION2_MATERIALS_CATALOG if s.name not in set(F4_UNMAPPED)]
    assert len(mapped) == 147 - 74


def test_f4_authority_has_no_generic_class_for_any_group():
    """v20: nc=29 — yalnızca other_ore / other_odyle jenerikleri; other_herb yok."""
    from dataset_factory.class_taxonomy import F4_LIVE_GENERIC_MATERIAL_MAP

    taxonomy = load_canonical_taxonomy()
    assert taxonomy.nc == 29
    names = canonical_names()
    base = {
        0: "herb_aria", 1: "herb_mela", 2: "herb_targena",
        3: "ore_orichalcum", 4: "gem_crystal", 5: "tree_log",
        6: "herb_edana", 7: "herb_lumina", 8: "ore_obsidian",
        9: "minimap_sprout", 10: "monster_3d", 11: "minimap_monster_dot",
        12: "crystal_blood", 13: "skill_ui_prompt", 14: "obstacle",
        15: "path_ground", 16: "player_hp_bar", 17: "player_mp_bar",
        18: "target_hp_bar", 19: "skill_bar", 20: "skill_slot_ready",
        21: "skill_slot_cooldown", 22: "hp_potion", 23: "mp_potion",
        24: "buff_scroll", 25: "kinah", 26: "player_character",
        27: "other_ore", 28: "other_odyle",
    }
    assert names == base
    herb_classes = {"herb_aria", "herb_mela", "herb_targena", "herb_edana", "herb_lumina"}
    ore_classes = {"ore_orichalcum", "ore_obsidian", "other_ore"}
    assert set(names.values()) & herb_classes == herb_classes
    assert set(names.values()) & ore_classes == ore_classes
    assert "other_herb" not in names.values()
    assert len(F4_LIVE_GENERIC_MATERIAL_MAP) == 2
    assert not any("cook" in n or "fiber" in n for n in names.values())


def test_f4_none_of_the_74_resolves_to_a_canonical_class():
    """UNMAPPED kalanlar çözülmez; canlı kanıtlı 2 madde jeneriğe map'lenir."""
    from dataset_factory.class_taxonomy import F4_LIVE_GENERIC_MATERIAL_MAP

    _build_and_collect()
    aliases = alias_map()
    by_name = {spec.name: spec for spec in AION2_MATERIALS_CATALOG}
    live = set(F4_LIVE_GENERIC_MATERIAL_MAP.keys())
    # Odyle grades share essence_odyle from the base ``Odyle`` jenerik kaydı (Selçuk 11.11).
    odyle_grades = {"Fine Odyle", "Pure Odyle", "Radiant Odyle"}
    for name in live:
        assert class_id_for_material_text(name) == F4_LIVE_GENERIC_MATERIAL_MAP[name]
    for name in set(F4_UNMAPPED) - live - odyle_grades:
        assert try_sanitize_class_id_and_name(class_id=-1, class_name=name) is None, name
        assert class_id_for_material_text(name) is None, name
        assert name.lower() not in aliases, name
        assert _slug(name) not in aliases, name
        assert "essence_" + _slug(name) not in aliases, name
        assert "essence_" + _slug(by_name[name].base_material) not in aliases, name
        name_tr = by_name[name].name_tr
        assert name_tr.lower() not in aliases, name_tr
        assert _slug(name_tr) not in aliases, name_tr
        # No predicate may re-classify them either ("gold ore" is not kinah).
        assert is_currency(name) is False, name
        assert is_consumable(name) is False, name
        assert is_skill_element(name) is False, name
        assert is_player_character(name) is False, name


def test_f4_labeler_falls_through_to_generic_bucket_12():
    """Auto-labeler: UNMAPPED kalanlar kova 12; canlı jenerik eşlemeler 27/28."""
    from dataset_factory.class_taxonomy import F4_LIVE_GENERIC_MATERIAL_MAP

    _build_and_collect()
    live = set(F4_LIVE_GENERIC_MATERIAL_MAP.keys())
    for name in set(F4_UNMAPPED) - live:
        assert map_material_name_to_yolo_class_id(name) == 12, name
    for name, cid in F4_LIVE_GENERIC_MATERIAL_MAP.items():
        assert map_material_name_to_yolo_class_id(name) == cid
    assert map_material_name_to_yolo_class_id("Withered Branch") == 12


def test_f4_gold_ore_is_not_currency():
    """Spot check of the stale ultra-tarama claim: v18 P1-8 already fixed
    ``is_currency("gold ore")`` — an ore is not Kinah (25)."""
    _build_and_collect()
    assert is_currency("gold ore") is False
    assert is_currency("Gold Ore") is False
    assert is_currency("gold") is True
