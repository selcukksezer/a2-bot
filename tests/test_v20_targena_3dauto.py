# -*- coding: utf-8 -*-
"""
v20-targena-3dauto düzeltmeleri — birim testleri (fix/v20-targena-3dauto).

Canlı kanıt (11 Kasım 2026, v20, PC):
- #297 (Targena) [3d_auto] haritalandı ama tarayıcı aynı görünümlü
  adayları canavar (monster_3d_negative, sim 0.83-0.90) / Mela /
  Orichalcum diye işaretliyor — v20 düzeltmesi (e205d1e) yalnızca
  material_scanner yolunu kapattı; kartografın kendi 3D tespiti
  (_detect_3d_world_gatherables) renk kurallarıyla (sarı+yeşil)
  mob gövdesini Targena saymaya devam etti.
- #292/#294/#295 (Gem) [3d_auto] — muhtemelen Targena'ydı ama kayıt
  "Gem" düştü; #297 (Targena) doğru yazıldı, tutarsız. Kapının
  verified_resource_type'ı spesifik 3D adını jenerik "Gem" ile eziyordu.

NOT: Bu dosya YAZILIR, KOŞULMAZ — testleri ana ajan koşar.
Mock'lanmış tarayıcı/3D aday ile çalışır; gerçek CV çalıştırılmaz.
"""

import unittest
from unittest import mock

import numpy as np

from core.auto_cartographer import (
    UNKNOWN_RESOURCE_TYPE,
    AutoCartographer,
    _is_generic_resource_name,
    _is_specific_resource_name,
    _resolve_mapped_resource_name,
)


class _StubMemory:
    """is_negative_patch'i hazır cevapla döndüren sahte exemplar belleği."""

    def __init__(self, verdict):
        # verdict: (is_neg, sim, ex_dict) — ex_dict {"class_name": ...}
        self._verdict = verdict
        self.calls = 0

    def is_negative_patch(self, crop, threshold=None, **kwargs):
        self.calls += 1
        return self._verdict


def _mesh():
    return np.zeros((10, 10, 3), dtype=np.uint8)


class TestMonsterNegativeGate(unittest.TestCase):
    """(1) 3D mesh canavar negatif örneğine uyuyorsa aday atılır."""

    def test_monster_match_returns_name_and_sim(self):
        mem = _StubMemory(
            (True, 0.88, {"class_name": "monster_3d_negative"})
        )
        cart = AutoCartographer(exemplar_memory=mem)
        self.assertEqual(
            cart._mesh_matches_monster_negative(_mesh()),
            ("monster_3d_negative", 0.88),
        )

    def test_player_match_returns_name_and_sim(self):
        mem = _StubMemory((True, 0.85, {"class_name": "player_character"}))
        cart = AutoCartographer(exemplar_memory=mem)
        self.assertEqual(
            cart._mesh_matches_monster_negative(_mesh()),
            ("player_character", 0.85),
        )

    def test_non_monster_negative_does_not_block(self):
        mem = _StubMemory((True, 0.90, {"class_name": "ground_rock"}))
        cart = AutoCartographer(exemplar_memory=mem)
        self.assertIsNone(cart._mesh_matches_monster_negative(_mesh()))

    def test_no_match_returns_none(self):
        mem = _StubMemory((False, 0.30, None))
        cart = AutoCartographer(exemplar_memory=mem)
        self.assertIsNone(cart._mesh_matches_monster_negative(_mesh()))

    def test_no_memory_returns_none(self):
        cart = AutoCartographer(exemplar_memory=object())
        # object() üzerinde is_negative_patch yok.
        self.assertIsNone(cart._mesh_matches_monster_negative(_mesh()))

    def test_memory_loader_none_returns_none(self):
        cart = AutoCartographer()
        with mock.patch.object(
            AutoCartographer, "_get_exemplar_memory", return_value=None
        ):
            self.assertIsNone(cart._mesh_matches_monster_negative(_mesh()))


class TestClassify3dMesh(unittest.TestCase):
    """(1b) Renk sınıflandırması + canavar kapısı birlikte."""

    def _cart(self, verdict):
        return AutoCartographer(exemplar_memory=_StubMemory(verdict))

    def test_monster_mesh_returns_none(self):
        cart = self._cart((True, 0.89, {"class_name": "monster_3d_negative"}))
        with mock.patch(
            "config.materials_db.classify_3d_material_patch",
            return_value={"name": "Targena"},
        ):
            self.assertIsNone(cart._classify_3d_mesh(_mesh(), 40))

    def test_clean_mesh_returns_name(self):
        cart = self._cart((False, 0.20, None))
        with mock.patch(
            "config.materials_db.classify_3d_material_patch",
            return_value={"name": "Targena"},
        ):
            self.assertEqual(cart._classify_3d_mesh(_mesh(), 40), "Targena")

    def test_no_memory_returns_name(self):
        cart = AutoCartographer()
        with mock.patch.object(
            AutoCartographer, "_get_exemplar_memory", return_value=None
        ), mock.patch(
            "config.materials_db.classify_3d_material_patch",
            return_value={"name": "Targena"},
        ):
            self.assertEqual(cart._classify_3d_mesh(_mesh(), 40), "Targena")

    def test_unclassified_returns_unknown(self):
        cart = self._cart((False, 0.20, None))
        with mock.patch(
            "config.materials_db.classify_3d_material_patch",
            return_value={"name": ""},
        ):
            self.assertEqual(
                cart._classify_3d_mesh(_mesh(), 40), UNKNOWN_RESOURCE_TYPE
            )

    def test_classifier_error_returns_unknown(self):
        cart = self._cart((False, 0.20, None))
        with mock.patch(
            "config.materials_db.classify_3d_material_patch",
            side_effect=RuntimeError("boom"),
        ):
            self.assertEqual(
                cart._classify_3d_mesh(_mesh(), 40), UNKNOWN_RESOURCE_TYPE
            )


class TestGenericNames(unittest.TestCase):
    """(2) Jenerik / spesifik ad ayrımı."""

    def test_specific_names(self):
        for name in (
            "Targena",
            "Mela",
            "Aria",
            "Orichalcum Ore",
            "Sapphire Gemstone",
            "Diamond Gemstone",
            "Yggdrasil Log",
            "Odyle",
        ):
            self.assertTrue(
                _is_specific_resource_name(name), f"{name} spesifik olmalı"
            )
            self.assertFalse(_is_generic_resource_name(name))

    def test_generic_names(self):
        for name in (
            "",
            None,
            "unknown",
            "Gatherable Material",
            "Gatherable Plant",
            "Gem",
            "Gemstone",
            "World Resource",
            "Resource",
        ):
            self.assertTrue(
                _is_generic_resource_name(name), f"{name!r} jenerik olmalı"
            )
            self.assertFalse(_is_specific_resource_name(name))


class TestResolveMappedName(unittest.TestCase):
    """(2b) Kayıt adı kararı — Targena "Gem" diye yazılmaz."""

    def test_targena_kept_over_gem(self):
        # Görev tanımı: Targena adayı -> kayıt adı "Targena", "Gem" değil.
        final, drop = _resolve_mapped_resource_name("Targena", "Gem", "3d_auto")
        self.assertEqual(final, "Targena")
        self.assertFalse(drop)

    def test_specific_never_overwritten_by_specific(self):
        final, drop = _resolve_mapped_resource_name("Targena", "Mela", "3d_auto")
        self.assertEqual(final, "Targena")
        self.assertFalse(drop)

    def test_generic_filled_by_specific_verification(self):
        final, drop = _resolve_mapped_resource_name(
            "Gatherable Material", "Targena", "3d_auto"
        )
        self.assertEqual(final, "Targena")
        self.assertFalse(drop)

    def test_unnamed_3d_candidate_dropped(self):
        # Ad belirlenemediyse yanlış jenerik kayıt yok.
        final, drop = _resolve_mapped_resource_name(
            "Gatherable Material", "Gem", "3d_auto"
        )
        self.assertTrue(drop)
        final, drop = _resolve_mapped_resource_name(
            "Gatherable Material", None, "3d_auto"
        )
        self.assertTrue(drop)
        final, drop = _resolve_mapped_resource_name("", "", "3d_auto")
        self.assertTrue(drop)

    def test_minimap_behavior_unchanged(self):
        # Minimap yolunda kapı doğrulaması eskisi gibi adı alır.
        final, drop = _resolve_mapped_resource_name(
            "Targena", "Gem", "minimap_auto"
        )
        self.assertEqual(final, "Gem")
        self.assertFalse(drop)
        final, drop = _resolve_mapped_resource_name(
            "Gatherable Material", "Gem", "minimap_auto"
        )
        self.assertEqual(final, "Gem")
        self.assertFalse(drop)

    def test_other_sources_unchanged(self):
        final, drop = _resolve_mapped_resource_name(
            "Targena", None, "harvest_auto"
        )
        self.assertEqual(final, "Targena")
        self.assertFalse(drop)


if __name__ == "__main__":
    unittest.main()
