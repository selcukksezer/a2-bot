# -*- coding: utf-8 -*-
"""Unit tests for dataset_factory/class_taxonomy.py consistency (W67)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dataset_factory.class_taxonomy import (
    ALL_CANONICAL_CLASSES,
    CANONICAL_LABELS_TR,
    CANONICAL_NAME_TO_ID,
    CANONICAL_NAMES,
    CANONICAL_SHORT_LABELS_TR,
    CONSUMABLE_CANONICAL_NAMES,
    CORE_22_CANONICAL_NAMES,
    dataset_class_names,
    dataset_labels_tr,
    is_valid_canonical_class_id,
    sanitize_class_id_and_name,
)


def test_canonical_classes_0_to_24_exact_mapping() -> None:
    """Verify 0..24 classes from herb_aria to buff_scroll and core/consumable splits."""
    expected_names = {
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
        22: "hp_potion",
        23: "mp_potion",
        24: "buff_scroll",
        25: "kinah",
        26: "player_character",
    }
    assert CANONICAL_NAMES == expected_names
    assert dataset_class_names() == expected_names
    assert set(CANONICAL_NAMES.keys()) == set(range(27))
    assert ALL_CANONICAL_CLASSES == frozenset(range(27))
    assert set(CORE_22_CANONICAL_NAMES.keys()) == set(range(22))
    assert set(CONSUMABLE_CANONICAL_NAMES.keys()) == {22, 23, 24}
    assert CANONICAL_NAMES[0] == "herb_aria"
    assert CANONICAL_NAMES[24] == "buff_scroll"

    # Bidirectional mapping check
    for cid, name in expected_names.items():
        assert CANONICAL_NAME_TO_ID[name] == cid
        assert is_valid_canonical_class_id(cid) is True
    assert is_valid_canonical_class_id(-1) is False
    assert is_valid_canonical_class_id(25) is True
    assert is_valid_canonical_class_id(26) is True
    assert is_valid_canonical_class_id(27) is False
    assert is_valid_canonical_class_id(98) is False


def test_turkish_labels_completeness_and_consistency() -> None:
    """Verify Turkish full and short labels exist for all classes and resolve back."""
    labels_tr = dataset_labels_tr()
    assert labels_tr == CANONICAL_LABELS_TR
    assert set(CANONICAL_LABELS_TR.keys()) == set(range(27))
    assert set(CANONICAL_SHORT_LABELS_TR.keys()) == set(range(27))

    assert CANONICAL_LABELS_TR[0] == "Aria (Mor Kristal Bitki)"
    assert CANONICAL_LABELS_TR[22] == "Can İksiri"
    assert CANONICAL_LABELS_TR[23] == "Mana İksiri"
    assert CANONICAL_LABELS_TR[24] == "Parşömen"

    for cid in range(25):
        full_tr = CANONICAL_LABELS_TR[cid]
        short_tr = CANONICAL_SHORT_LABELS_TR[cid]
        assert isinstance(full_tr, str) and len(full_tr.strip()) > 0
        assert isinstance(short_tr, str) and len(short_tr.strip()) > 0

        # Both Turkish full label and short label must sanitize to the exact canonical (cid, name)
        assert sanitize_class_id_and_name(-1, full_tr) == (cid, CANONICAL_NAMES[cid])
        assert sanitize_class_id_and_name(-1, short_tr) == (cid, CANONICAL_NAMES[cid])


def test_sanitize_class_id_and_name_canonical_roundtrip() -> None:
    """Verify sanitize_class_id_and_name resolves all 0..24 IDs, names, and formatting variants."""
    for cid, cname in CANONICAL_NAMES.items():
        # By ID only
        assert sanitize_class_id_and_name(cid, "") == (cid, cname)
        assert sanitize_class_id_and_name(str(cid), None) == (cid, cname)
        # By canonical name only (with invalid ID)
        assert sanitize_class_id_and_name(-1, cname) == (cid, cname)
        assert sanitize_class_id_and_name(None, f"  {cname.upper()}  ") == (cid, cname)
        # By both matching ID and name
        assert sanitize_class_id_and_name(cid, cname) == (cid, cname)


def test_sanitize_class_id_and_name_aliases_and_legacy_collisions() -> None:
    """Verify legacy aliases, Turkish aliases, and legacy ID 18/19 potion collisions."""
    # Legacy class index collision disambiguation (old 18=hp_potion, 19=mp_potion vs vitals/skill_bar)
    assert sanitize_class_id_and_name(18, "hp_potion") == (22, "hp_potion")
    assert sanitize_class_id_and_name(18, "can iksiri") == (22, "hp_potion")
    assert sanitize_class_id_and_name(18, "") == (18, "target_hp_bar")
    assert sanitize_class_id_and_name(19, "mp_potion") == (23, "mp_potion")
    assert sanitize_class_id_and_name(19, "mana iksiri") == (23, "mp_potion")
    assert sanitize_class_id_and_name(19, "") == (19, "skill_bar")

    # Legacy aliases and Turkish terms
    assert sanitize_class_id_and_name(98, "wood_tree") == (5, "tree_log")
    assert sanitize_class_id_and_name(98, "world_gather_crest") == (13, "skill_ui_prompt")
    assert sanitize_class_id_and_name(-1, "blood_crystal") == (12, "crystal_blood")
    assert sanitize_class_id_and_name(-1, "parsomen") == (24, "buff_scroll")
    assert sanitize_class_id_and_name(-1, "parşömen") == (24, "buff_scroll")


def test_sanitize_class_id_and_name_rejects_non_canonical_and_supports_fallback() -> None:
    """Verify unresolvable classes (e.g. 98) raise ValueError unless valid fallback is enabled."""
    with pytest.raises(ValueError, match="Non-canonical class cannot be resolved"):
        sanitize_class_id_and_name(98, "")

    with pytest.raises(ValueError, match="Non-canonical class cannot be resolved"):
        sanitize_class_id_and_name(-1, "completely_unknown_nonexistent_class_xyz")

    with pytest.raises(ValueError, match="Non-canonical class cannot be resolved"):
        sanitize_class_id_and_name(999, "invalid_item", allow_fallback=True, default_fallback=98)

    # Valid fallback when allow_fallback=True and default_fallback is in 0..24
    assert sanitize_class_id_and_name(
        98, "completely_unknown_xyz", allow_fallback=True, default_fallback=0
    ) == (0, "herb_aria")
    assert sanitize_class_id_and_name(
        None, "", allow_fallback=True, default_fallback=24
    ) == (24, "buff_scroll")
