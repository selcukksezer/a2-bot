# -*- coding: utf-8 -*-
"""v20: MaterialScanner exemplar refine must not cross material families.

Live bug: Targena candidates were remapped to Orichalcum Ore / Mela via
learned positive exemplars (sim 0.71–0.91) because
_refine_candidates_with_exemplars called unrestricted classify_patch.

Family lock (class_taxonomy): herbs {0,1,2,6,7} / ores {3,8} /
gems {4,12} / trees {5}.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.material_scanner import DetectedMaterial, MaterialScanner
from dataset_factory.class_taxonomy import (
    MATERIAL_FAMILY_HERBS,
    MATERIAL_FAMILY_ORES,
    is_same_material_family,
)
from dataset_factory.exemplar_memory import ExemplarMemory, extract_crop_features


ROOT = Path(__file__).resolve().parent.parent


def _textured_crop(seed: int = 7, h: int = 48, w: int = 48) -> np.ndarray:
    rng = np.random.default_rng(seed)
    crop = rng.integers(40, 90, size=(h, w, 3), dtype=np.uint8)
    crop[:, :, 1] = np.clip(crop[:, :, 1].astype(np.int16) + 80, 0, 255).astype(np.uint8)
    noise = rng.integers(0, 40, size=crop.shape, dtype=np.uint8)
    return np.clip(crop.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _candidate(name: str = "Targena") -> DetectedMaterial:
    return DetectedMaterial(
        material_type=name,
        source="3d_world",
        screen_x=100,
        screen_y=100,
        bbox=(76, 76, 48, 48),
        rel_map_dx=0.0,
        rel_map_dy=0.0,
        estimated_distance_m=5.0,
        confidence=0.55,
    )


def test_resolve_material_family_cid_targena_and_ore() -> None:
    assert MaterialScanner._resolve_material_family_cid("Targena") == 2
    assert MaterialScanner._resolve_material_family_cid("Orichalcum Ore") == 3
    assert MaterialScanner._resolve_material_family_cid("Mela") == 1
    assert 2 in MATERIAL_FAMILY_HERBS
    assert 3 in MATERIAL_FAMILY_ORES
    assert is_same_material_family(2, 1) is True
    assert is_same_material_family(2, 3) is False


def test_refine_blocks_targena_to_orichalcum_cross_family(tmp_path: Path) -> None:
    """Strong ore exemplar must not rename a Targena candidate.

    Source is not in the 3d discrimination set so the candidate is not
    dropped for authenticity — this isolates the refine family lock.
    """
    ex_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=ex_path)
    crop = _textured_crop(seed=11)
    feats = extract_crop_features(crop)
    assert feats is not None
    mem.add_positive_exemplar(crop, class_id=3, class_name="ore_orichalcum")

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[76:124, 76:124] = crop

    scanner = MaterialScanner()
    cand = _candidate("Targena")
    cand.source = "nameplate"  # skip discriminate_3d_material drop path
    out = scanner._refine_candidates_with_exemplars(frame, [cand], exemplar_memory=mem)
    assert len(out) == 1
    assert out[0].material_type == "Targena"


def test_refine_allows_same_family_herb_to_herb(tmp_path: Path) -> None:
    """Within herbs, positive exemplar may refine Aria -> Mela."""
    ex_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=ex_path)
    crop = _textured_crop(seed=22)
    mem.add_positive_exemplar(crop, class_id=1, class_name="herb_mela")

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    frame[76:124, 76:124] = crop

    scanner = MaterialScanner()
    cand = _candidate("Aria")
    out = scanner._refine_candidates_with_exemplars(frame, [cand], exemplar_memory=mem)
    assert len(out) == 1
    assert out[0].material_type == "Mela"


def test_classify_patch_respects_allowed_classes(tmp_path: Path) -> None:
    ex_path = tmp_path / "learned_exemplars.json"
    mem = ExemplarMemory(json_path=ex_path)
    crop = _textured_crop(seed=33)
    mem.add_positive_exemplar(crop, class_id=3, class_name="ore_orichalcum")
    mem.add_positive_exemplar(crop, class_id=2, class_name="herb_targena")

    cid, cname, sim = mem.classify_patch(
        crop, min_sim=0.50, allowed_classes=frozenset(MATERIAL_FAMILY_HERBS)
    )
    assert cid == 2
    assert "targena" in str(cname).lower()
    assert sim >= 0.50
