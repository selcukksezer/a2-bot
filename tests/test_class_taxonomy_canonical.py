# -*- coding: utf-8 -*-
"""tests/test_class_taxonomy_canonical.py
==========================================
Comprehensive verification test suite for canonical class taxonomy, validation,
sanitization, consumables, and super-category helpers in dataset_factory/class_taxonomy.py.
"""

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.class_taxonomy import (
    ALL_CANONICAL_CLASSES,
    CANONICAL_NAMES,
    CANONICAL_NAME_TO_ID,
    CONSUMABLE_CLASSES,
    CORE_22_CANONICAL_NAMES,
    CORE_22_CLASSES,
    HUD_ELEMENT_CLASSES,
    HUD_SKILL_CLASSES,
    HUD_VITALS_CLASSES,
    MATERIAL_3D_CLASSES,
    MINIMAP_SYMBOL_CLASSES,
    NAVIGABLE_ELEMENT_CLASSES,
    PLAYER_CLASSES,
    STABLE_NAMES,
    alias_map,
    all_ui_class_specs,
    class_id_for_material_text,
    cognition_class_ids,
    dataset_class_names,
    dataset_labels_tr,
    filter_canonical_review_boxes,
    format_yolo_txt_line,
    gather_3d_material_class_ids,
    gather_class_ids,
    get_super_category,
    is_3d_world_material,
    is_consumable,
    is_hud_element,
    is_minimap_symbol,
    is_monster_element,
    is_navigable_element,
    is_player_character,
    is_same_super_category,
    is_skill_element,
    is_valid_canonical_class_id,
    is_valid_core_22_class_id,
    material_name_for_class,
    parse_and_sanitize_yolo_txt_line,
    sanitize_class_id_and_name,
    sanitize_yolo_box,
    try_sanitize_class_id_and_name,
    ui_class_specs,
)


def test_core_22_classes_exactness():
    """Verify that the core 22 classes match the dispatch requirements exactly."""
    expected_core = {
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
    assert len(CORE_22_CANONICAL_NAMES) == 22
    for cid, name in expected_core.items():
        assert CORE_22_CANONICAL_NAMES[cid] == name
        assert CANONICAL_NAMES[cid] == name
        assert is_valid_core_22_class_id(cid) is True


def test_consumable_classes_defined():
    """Verify that consumable classes 22..24 (hp_potion, mp_potion, buff_scroll) are defined."""
    assert CANONICAL_NAMES[22] == "hp_potion"
    assert CANONICAL_NAMES[23] == "mp_potion"
    assert CANONICAL_NAMES[24] == "buff_scroll"
    assert CANONICAL_NAMES[25] == "kinah"
    assert CANONICAL_NAMES[26] == "player_character"
    assert len(CANONICAL_NAMES) == 27
    assert len(ALL_CANONICAL_CLASSES) == 27


def test_is_valid_canonical_class_id():
    """Verify that canonical validation accepts 0..26 and rejects alien IDs."""
    for cid in range(27):
        assert is_valid_canonical_class_id(cid) is True
        assert is_valid_canonical_class_id(str(cid)) is True

    # Alien / non-canonical IDs
    assert is_valid_canonical_class_id(-1) is False
    assert is_valid_canonical_class_id(27) is False
    assert is_valid_canonical_class_id(98) is False  # Legacy essence_yggdrasil_log ID
    assert is_valid_canonical_class_id(135) is False
    assert is_valid_canonical_class_id("invalid") is False
    assert is_valid_canonical_class_id(None) is False


def test_super_categories():
    """Verify super-category memberships and predicates."""
    # 3D Materials
    expected_3d = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12}
    assert MATERIAL_3D_CLASSES == expected_3d
    for cid in expected_3d:
        assert is_3d_world_material(cid) is True
        assert get_super_category(cid) == "3d_material"
    assert is_3d_world_material(10) is False  # monster is not a material
    assert is_3d_world_material(9) is False   # minimap sprout is not 3D material

    # Minimap
    assert MINIMAP_SYMBOL_CLASSES == {9, 11}
    assert is_minimap_symbol(9) is True
    assert is_minimap_symbol(11) is True
    assert is_minimap_symbol(10) is False

    # Navigable
    assert NAVIGABLE_ELEMENT_CLASSES == {14, 15}
    assert is_navigable_element(14) is True
    assert is_navigable_element(15) is True
    assert is_navigable_element(0) is False

    # Consumables vs Skills vs HUD
    assert is_consumable(22) is True
    assert is_consumable(23) is True
    assert is_consumable(24) is True
    assert is_consumable("hp_potion") is True
    assert is_consumable("mp_potion") is True
    assert is_consumable("buff_scroll") is True
    assert is_consumable("Parşömen") is True
    assert is_consumable("Can İksiri") is True
    assert is_consumable("skill_slot_ready") is False

    # Skills separated from consumables
    assert is_skill_element(19) is True
    assert is_skill_element(20) is True
    assert is_skill_element(21) is True
    assert is_skill_element(13) is True
    assert is_skill_element("skill_bar") is True
    assert is_skill_element("skill_slot_ready") is True
    assert is_skill_element("hp_potion") is False
    assert is_skill_element("buff_scroll") is False

    # HUD Elements include vitals, skills, prompts, consumables
    for cid in (13, 16, 17, 18, 19, 20, 21, 22, 23, 24):
        assert is_hud_element(cid) is True
    assert is_hud_element(0) is False
    assert is_hud_element(10) is False


def test_is_same_super_category():
    """Verify category locking for ExemplarMemory and label trust."""
    # 3D materials with 3D materials
    assert is_same_super_category(0, 1) is True
    assert is_same_super_category(0, 5) is True
    assert is_same_super_category(3, 8) is True
    assert is_same_super_category(4, 12) is True

    # 3D material cannot cross to HUD or Monster
    assert is_same_super_category(0, 10) is False
    assert is_same_super_category(0, 16) is False
    assert is_same_super_category(0, 20) is False

    # Minimap symbols match minimap symbols
    assert is_same_super_category(9, 11) is True
    assert is_same_super_category(9, 0) is False

    # Consumables match consumables
    assert is_same_super_category(22, 23) is True
    assert is_same_super_category(22, 24) is True
    assert is_same_super_category(22, 20) is False  # Potion vs skill


def test_sanitize_class_id_and_name_success():
    """Verify sanitization of valid, alias, and legacy class names."""
    # Canonical direct
    assert sanitize_class_id_and_name(0, "herb_aria") == (0, "herb_aria")
    assert sanitize_class_id_and_name(6, "herb_edana") == (6, "herb_edana")
    assert sanitize_class_id_and_name(7, "herb_lumina") == (7, "herb_lumina")
    assert sanitize_class_id_and_name(8, "ore_obsidian") == (8, "ore_obsidian")
    assert sanitize_class_id_and_name(12, "crystal_blood") == (12, "crystal_blood")
    assert sanitize_class_id_and_name(24, "buff_scroll") == (24, "buff_scroll")

    # Legacy aliases
    assert sanitize_class_id_and_name(5, "wood_tree") == (5, "tree_log")
    assert sanitize_class_id_and_name(10, "monster_3d_negative") == (10, "monster_3d")
    assert sanitize_class_id_and_name(11, "minimap_monster_negative") == (11, "minimap_monster_dot")
    assert sanitize_class_id_and_name(20, "skill_slot") == (20, "skill_slot_ready")

    # Catalog essence slugs resolved to canonical class IDs
    assert sanitize_class_id_and_name(98, "essence_yggdrasil_log") == (5, "tree_log")
    assert sanitize_class_id_and_name(98, "essence_asvata_log") == (5, "tree_log")
    assert sanitize_class_id_and_name(100, "essence_aria") == (0, "herb_aria")
    assert sanitize_class_id_and_name(100, "essence_orichalcum") == (3, "ore_orichalcum")
    assert sanitize_class_id_and_name(100, "essence_blood_crystal") == (12, "crystal_blood")
    assert sanitize_class_id_and_name(100, "essence_obsidian") == (8, "ore_obsidian")

    # Disambiguation of old data.yaml 18/19
    assert sanitize_class_id_and_name(18, "hp_potion") == (22, "hp_potion")
    assert sanitize_class_id_and_name(18, "target_hp_bar") == (18, "target_hp_bar")
    assert sanitize_class_id_and_name(19, "mp_potion") == (23, "mp_potion")
    assert sanitize_class_id_and_name(19, "skill_bar") == (19, "skill_bar")

    # Valid ID with empty name
    assert sanitize_class_id_and_name(3, "") == (3, "ore_orichalcum")
    assert sanitize_class_id_and_name(24, "") == (24, "buff_scroll")


def test_sanitize_class_id_and_name_rejection():
    """Verify that unresolvable non-canonical classes raise ValueError."""
    with pytest.raises(ValueError):
        # Alien ID with no name
        sanitize_class_id_and_name(98, "")

    with pytest.raises(ValueError):
        # Invalid ID and unknown junk name
        sanitize_class_id_and_name(-1, "completely_unknown_gibberish_12345")

    with pytest.raises(ValueError):
        # Alien ID 135
        sanitize_class_id_and_name(135, "")


def test_try_sanitize_class_id_and_name():
    """Verify try_sanitize returns None instead of raising."""
    assert try_sanitize_class_id_and_name(98, "") is None
    assert try_sanitize_class_id_and_name(-1, "unknown") is None
    assert try_sanitize_class_id_and_name(0, "herb_aria") == (0, "herb_aria")
    assert try_sanitize_class_id_and_name(98, "essence_yggdrasil_log") == (5, "tree_log")


def test_sanitize_yolo_box():
    """Verify box dictionary sanitization and dropping of poisoned boxes."""
    # Good box
    box_good = {"class_id": 0, "class_name": "herb_aria", "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    res = sanitize_yolo_box(box_good)
    assert res is not None
    assert res["class_id"] == 0
    assert res["class_name"] == "herb_aria"

    # Box with legacy alias that gets repaired
    box_repair = {"class_id": 98, "class_name": "essence_yggdrasil_log", "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    res = sanitize_yolo_box(box_repair)
    assert res is not None
    assert res["class_id"] == 5
    assert res["class_name"] == "tree_log"

    # Poisoned box with non-canonical class and no resolvable name -> dropped
    box_poison = {"class_id": 98, "class_name": "", "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1}
    assert sanitize_yolo_box(box_poison, strict_drop_non_canonical=True) is None


def test_filter_canonical_review_boxes():
    """Verify filtering a batch of boxes drops invalid and repairs aliases."""
    boxes = [
        {"class_id": 0, "class_name": "herb_aria"},
        {"class_id": 98, "class_name": ""},  # Unresolvable -> DROP
        {"class_id": 98, "class_name": "essence_yggdrasil_log"},  # Repaired to 5
        {"class_id": 24, "class_name": "buff_scroll"},
    ]
    filtered = filter_canonical_review_boxes(boxes)
    assert len(filtered) == 3
    assert filtered[0]["class_id"] == 0
    assert filtered[1]["class_id"] == 5
    assert filtered[1]["class_name"] == "tree_log"
    assert filtered[2]["class_id"] == 24


def test_format_and_parse_yolo_txt_line():
    """Verify formatting and parsing of YOLO txt lines with canonical enforcement."""
    # Valid line
    line = format_yolo_txt_line(0, 0.5, 0.5, 0.2, 0.2, "herb_aria")
    assert line == "0 0.500000 0.500000 0.200000 0.200000"

    # Alien unresolvable class -> returns None (strictly blocks poison!)
    assert format_yolo_txt_line(98, 0.5, 0.5, 0.2, 0.2, "") is None
    assert format_yolo_txt_line(-1, 0.5, 0.5, 0.2, 0.2, "unknown") is None

    # Alien repaired class -> formatted with canonical ID
    line_repaired = format_yolo_txt_line(98, 0.5, 0.5, 0.2, 0.2, "essence_yggdrasil_log")
    assert line_repaired == "5 0.500000 0.500000 0.200000 0.200000"

    # Parsing
    parsed = parse_and_sanitize_yolo_txt_line("0 0.500000 0.500000 0.200000 0.200000")
    assert parsed is not None
    assert parsed[0] == 0
    assert parsed[5] == "herb_aria"

    # Parsing non-canonical 98 line without name -> None
    assert parse_and_sanitize_yolo_txt_line("98 0.500000 0.500000 0.200000 0.200000") is None


def test_ui_class_specs_and_backwards_compatibility():
    """Verify that ui_class_specs produces specs for classes >= 16 and contains consumables, kinah, and player_character."""
    specs = ui_class_specs()
    spec_ids = {s["id"] for s in specs}
    assert 16 in spec_ids
    assert 17 in spec_ids
    assert 18 in spec_ids
    assert 19 in spec_ids
    assert 20 in spec_ids
    assert 21 in spec_ids
    assert 22 in spec_ids
    assert 23 in spec_ids
    assert 24 in spec_ids
    assert 25 in spec_ids
    assert 26 in spec_ids
    assert max(spec_ids) == 26

    all_specs = all_ui_class_specs()
    assert len(all_specs) == 27


def test_dataset_class_names_and_alias_map():
    """Verify that dataset_class_names and alias_map are correctly built with 27 canonical classes."""
    names = dataset_class_names()
    assert len(names) == 27
    assert max(names.keys()) == 26

    aliases = alias_map()
    assert "essence_yggdrasil_log" in aliases
    assert aliases["essence_yggdrasil_log"] == 5
    assert aliases["buff_scroll"] == 24
    assert aliases["parşömen"] == 24
    assert aliases["kinah"] == 25
    assert aliases["player_character"] == 26
    assert aliases["karakter"] == 26
    assert aliases["karakterim"] == 26
    assert aliases["oyuncu"] == 26
    assert aliases["player"] == 26
    assert max(aliases.values()) <= 26


def test_player_character_class():
    """Verify player character (26: player_character) is properly recognized and categorized."""
    assert is_player_character(26) is True
    assert is_player_character("player_character") is True
    assert is_player_character("karakter") is True
    assert is_player_character("karakterim") is True
    assert is_player_character(10) is False
    assert get_super_category(26) == "player"
    assert get_super_category("player_character") == "player"
    cid, cname = sanitize_class_id_and_name(26, "player_character")
    assert cid == 26
    assert cname == "player_character"
