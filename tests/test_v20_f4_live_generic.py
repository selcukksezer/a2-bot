# -*- coding: utf-8 -*-
"""v20 F4 — yalnızca canlı log kanıtlı jenerik eşleme (Dark Iron, Odyle)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dataset_factory.class_taxonomy import (
    F4_LIVE_GENERIC_MATERIAL_MAP,
    UNMAPPED_CATALOG_MATERIALS,
    alias_map,
    class_id_for_material_text,
    load_canonical_taxonomy,
)

from tests.test_f4_taxonomy_unmapped import F4_UNMAPPED


def test_f4_generic_classes_in_data_yaml():
    tax = load_canonical_taxonomy()
    assert tax.nc == 29
    assert tax.names[27] == "other_ore"
    assert tax.names[28] == "other_odyle"


def test_f4_live_mapped_materials_resolve():
    alias_map()
    for name, cid in F4_LIVE_GENERIC_MATERIAL_MAP.items():
        assert class_id_for_material_text(name) == cid, name


def test_f4_unmapped_is_74_minus_live_mapped():
    alias_map()
    live = set(F4_LIVE_GENERIC_MATERIAL_MAP.keys())
    still_unmapped = sorted(set(UNMAPPED_CATALOG_MATERIALS))
    assert len(still_unmapped) == len(F4_UNMAPPED) - len(live)
    for name in live:
        assert name not in still_unmapped
    assert "Iron Ore" in still_unmapped
    assert "Lura" in still_unmapped


def test_f4_no_alias_for_unseen_unmapped():
    alias_map()
    assert class_id_for_material_text("Iron Ore") is None
    assert class_id_for_material_text("Withered Branch") is None
