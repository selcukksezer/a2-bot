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

import pytest

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
    """Verify aggregate sizes: 27 canonical classes, 24 core-only, 27 combobox items."""
    registry = SearchableTagRegistry()
    all_classes = registry.get_all_classes(include_consumables=True)
    assert len(all_classes) == 27
    assert len(CANONICAL_NAMES) == 27

    core_only = registry.get_all_classes(include_consumables=False)
    assert len(core_only) == 24
    assert [c["class_id"] for c in core_only] == list(range(22)) + [25, 26]

    combo_items = registry.get_combobox_items()
    assert len(combo_items) == 27


@pytest.mark.parametrize("class_id", range(27))
def test_tag_list_entry_matches_canonical_catalog(class_id: int) -> None:
    """Verify each single catalog entry (id, name, key, display label, combobox item).

    Eskiden 27 girdilik tek döngüydü: ilk hatalı girdide döngü kırılır,
    kalan girdiler hiç denetlenmezdi. Parametreli hali her girdiyi ayrı
    test vakası yapar; bir girdideki hata diğer 26'sını gizlemez.
    """
    registry = SearchableTagRegistry()
    all_classes = registry.get_all_classes(include_consumables=True)
    entry = all_classes[class_id]
    assert entry["class_id"] == class_id
    assert entry["id"] == class_id
    assert entry["class_name"] == CANONICAL_NAMES[class_id]
    assert entry["key"] == CANONICAL_NAMES[class_id]
    assert f"{class_id}:" in entry["display_label"]
    assert CANONICAL_NAMES[class_id] in entry["display_label"]

    combo_items = registry.get_combobox_items()
    assert combo_items[class_id][0] == class_id
    assert CANONICAL_NAMES[class_id] in combo_items[class_id][1]


@pytest.mark.parametrize(
    "query,expected_id,expected_name",
    [("aria", 0, "herb_aria"), ("orichalcum", 3, "ore_orichalcum")],
)
def test_search_gather_materials_aria_and_orichalcum(query: str, expected_id: int, expected_name: str) -> None:
    """Verify 'aria' and 'orichalcum' queries return (0, 'herb_aria') and (3, 'ore_orichalcum')."""
    results = search_label_classes(query)
    assert len(results) >= 1
    assert results[0]["class_id"] == expected_id
    assert results[0]["class_name"] == expected_name

    registry = get_tag_registry()
    assert registry.resolve_class_id(query) == expected_id


@pytest.mark.parametrize("query", ["pot", "iksir"])
def test_search_potions_pot_and_iksir(query: str) -> None:
    """Verify 'pot' and 'iksir' queries return hp_potion (22) and mp_potion (23) with exact class_id and class_name."""
    results = search_label_classes(query)
    assert len(results) >= 2
    top_two_pairs = [(r["class_id"], r["class_name"]) for r in results[:2]]
    assert (22, "hp_potion") in top_two_pairs
    assert (23, "mp_potion") in top_two_pairs
    assert results[0]["class_id"] == 22
    assert results[0]["class_name"] == "hp_potion"
    assert results[1]["class_id"] == 23
    assert results[1]["class_name"] == "mp_potion"


@pytest.mark.parametrize("query", ["scroll", "parşömen"])
def test_search_scrolls_scroll_and_parsomen(query: str) -> None:
    """Verify 'scroll' and 'parşömen' queries return (24, 'buff_scroll')."""
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


def test_search_skill_classes() -> None:
    """Verify 'skill' returns all canonical skill classes (13, 19, 20, 21)."""
    skill_results = search_label_classes("skill")
    skill_pairs = {(r["class_id"], r["class_name"]) for r in skill_results}
    expected_skill_pairs = {
        (13, "skill_ui_prompt"),
        (19, "skill_bar"),
        (20, "skill_slot_ready"),
        (21, "skill_slot_cooldown"),
    }
    assert expected_skill_pairs.issubset(skill_pairs)


@pytest.mark.parametrize(
    "query,expected_id,expected_name",
    [
        ("aria", 0, "herb_aria"),
        ("orichalcum", 3, "ore_orichalcum"),
        ("pot", 22, "hp_potion"),
        ("iksir", 22, "hp_potion"),
        ("scroll", 24, "buff_scroll"),
        ("parşömen", 24, "buff_scroll"),
        ("canavar", 10, "monster_3d"),
        ("filiz", 9, "minimap_sprout"),
    ],
)
def test_search_all_target_queries_end_to_end(query: str, expected_id: int, expected_name: str) -> None:
    """Validate all 8 target queries end to end: each query's top match is exact.

    Eskiden tek testteki döngüydü: ilk hatalı sorguda kalan sorgular hiç
    koşmazdı. Parametreli hali her sorguyu ayrı vaka yapar.
    """
    res = search_label_classes(query)
    assert res, f"Query {query!r} returned empty results"
    assert (res[0]["class_id"], res[0]["class_name"]) == (expected_id, expected_name)
