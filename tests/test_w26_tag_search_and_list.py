# -*- coding: utf-8 -*-
"""tests/test_w26_tag_search_and_list.py
========================================
Unit tests for Worker W26 verifying the tag list and bilingual search system
(`dataset_factory.manual_tag_search`) for queries:
"aria", "orichalcum", "pot", "iksir", "scroll", "parşömen", "canavar", "skill", "filiz".
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.class_taxonomy import CANONICAL_NAMES
from dataset_factory.manual_tag_search import (
    SearchableTagRegistry,
    create_manual_annotation_box,
    get_tag_registry,
    search_label_classes,
)


def test_tag_list_full_canonical_catalog_and_combobox_items() -> None:
    """Verify the tag list returns all 25 canonical classes (0..24) with exact class_id and class_name."""
    registry = SearchableTagRegistry()
    all_classes = registry.get_all_classes(include_consumables=True)
    assert len(all_classes) == 25

    for idx, entry in enumerate(all_classes):
        assert entry["class_id"] == idx
        assert entry["id"] == idx
        assert entry["class_name"] == CANONICAL_NAMES[idx]
        assert entry["key"] == CANONICAL_NAMES[idx]
        assert f"{idx}:" in entry["display_label"]
        assert CANONICAL_NAMES[idx] in entry["display_label"]

    core_only = registry.get_all_classes(include_consumables=False)
    assert len(core_only) == 22
    assert [c["class_id"] for c in core_only] == list(range(22))

    combo_items = registry.get_combobox_items()
    assert len(combo_items) == 25
    assert combo_items[0][0] == 0 and "herb_aria" in combo_items[0][1]
    assert combo_items[3][0] == 3 and "ore_orichalcum" in combo_items[3][1]
    assert combo_items[9][0] == 9 and "minimap_sprout" in combo_items[9][1]
    assert combo_items[22][0] == 22 and "hp_potion" in combo_items[22][1]
    assert combo_items[23][0] == 23 and "mp_potion" in combo_items[23][1]
    assert combo_items[24][0] == 24 and "buff_scroll" in combo_items[24][1]


def test_search_gather_materials_aria_and_orichalcum() -> None:
    """Verify 'aria' and 'orichalcum' queries return (0, 'herb_aria') and (3, 'ore_orichalcum')."""
    aria_results = search_label_classes("aria")
    assert len(aria_results) >= 1
    assert aria_results[0]["class_id"] == 0
    assert aria_results[0]["class_name"] == "herb_aria"

    orichalcum_results = search_label_classes("orichalcum")
    assert len(orichalcum_results) >= 1
    assert orichalcum_results[0]["class_id"] == 3
    assert orichalcum_results[0]["class_name"] == "ore_orichalcum"

    registry = get_tag_registry()
    assert registry.resolve_class_id("aria") == 0
    assert registry.resolve_class_id("orichalcum") == 3


def test_search_potions_pot_and_iksir() -> None:
    """Verify 'pot' and 'iksir' queries return hp_potion (22) and mp_potion (23) with exact class_id and class_name."""
    for query in ("pot", "iksir"):
        results = search_label_classes(query)
        assert len(results) >= 2
        top_two_pairs = [(r["class_id"], r["class_name"]) for r in results[:2]]
        assert (22, "hp_potion") in top_two_pairs
        assert (23, "mp_potion") in top_two_pairs
        assert results[0]["class_id"] == 22
        assert results[0]["class_name"] == "hp_potion"
        assert results[1]["class_id"] == 23
        assert results[1]["class_name"] == "mp_potion"


def test_search_scrolls_scroll_and_parsomen() -> None:
    """Verify 'scroll' and 'parşömen' queries return (24, 'buff_scroll')."""
    for query in ("scroll", "parşömen"):
        results = search_label_classes(query)
        assert len(results) >= 1
        assert results[0]["class_id"] == 24
        assert results[0]["class_name"] == "buff_scroll"

        box = create_manual_annotation_box(query, 0.5, 0.5, 0.1, 0.1)
        assert box["class_id"] == 24
        assert box["class_name"] == "buff_scroll"
        assert box["trusted"] is True
        assert box["hand_label"] is True


def test_search_canavar_and_filiz_matches() -> None:
    """Verify 'canavar' returns monster classes (10: monster_3d, 11: minimap_monster_dot) and 'filiz' returns (9, 'minimap_sprout')."""
    canavar_results = search_label_classes("canavar")
    assert len(canavar_results) >= 2
    assert canavar_results[0]["class_id"] == 10
    assert canavar_results[0]["class_name"] == "monster_3d"

    canavar_map = {r["class_id"]: r["class_name"] for r in canavar_results}
    assert canavar_map.get(10) == "monster_3d"
    assert canavar_map.get(11) == "minimap_monster_dot"

    filiz_results = search_label_classes("filiz")
    assert len(filiz_results) >= 1
    assert filiz_results[0]["class_id"] == 9
    assert filiz_results[0]["class_name"] == "minimap_sprout"


def test_search_skill_classes_and_all_nine_queries_end_to_end() -> None:
    """Verify 'skill' returns all canonical skill classes (13, 19, 20, 21) and validate all 9 target queries."""
    skill_results = search_label_classes("skill")
    skill_pairs = {(r["class_id"], r["class_name"]) for r in skill_results}
    expected_skill_pairs = {
        (13, "skill_ui_prompt"),
        (19, "skill_bar"),
        (20, "skill_slot_ready"),
        (21, "skill_slot_cooldown"),
    }
    assert expected_skill_pairs.issubset(skill_pairs)

    expected_top_matches = {
        "aria": (0, "herb_aria"),
        "orichalcum": (3, "ore_orichalcum"),
        "pot": (22, "hp_potion"),
        "iksir": (22, "hp_potion"),
        "scroll": (24, "buff_scroll"),
        "parşömen": (24, "buff_scroll"),
        "canavar": (10, "monster_3d"),
        "filiz": (9, "minimap_sprout"),
    }
    for query, (exp_id, exp_name) in expected_top_matches.items():
        res = search_label_classes(query)
        assert res, f"Query {query!r} returned empty results"
        assert (res[0]["class_id"], res[0]["class_name"]) == (exp_id, exp_name)
