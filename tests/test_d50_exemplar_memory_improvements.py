# -*- coding: utf-8 -*-
"""
tests/test_d50_exemplar_memory_improvements.py
==============================================
DEV D50 verification suite for the exemplar_memory improvements:

1. Vectorized super-category mask in match_positive() — the precomputed
   per-exemplar code table must agree with get_super_category() for every
   canonical class, and category-locked matching must return only
   same-category classes.
2. Vectorized rounding in extract_crop_features() — np.round on float64
   must produce exactly the old per-element round(float(v), n) values.
3. feats= reuse in match_negative()/match_positive() — passing
   pre-extracted features must give byte-identical results to
   re-extraction, and refine_candidate() must extract features exactly
   once per candidate (was: 3x).
4. sanitize() positive purge shares the single _exemplar_is_clean()
   predicate with _sanitize_classes_dict() — purge decisions must agree
   record by record, including every poison shape.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

import dataset_factory.exemplar_memory as em
from dataset_factory.exemplar_memory import (
    ExemplarMemory,
    _exemplar_is_clean,
    extract_crop_features,
    get_super_category,
)


def _rng_crop(seed: int, size: int = 96) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def _exemplar(class_id: int, seed: int, source: str = "ui_approval") -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "class_id": class_id,
        "class_name": f"class_{class_id}",
        "source": source,
        "hsv_hist": rng.random(432).tolist(),
        "aspect_ratio": 1.0,
        "texture": [0.2, 0.2, 0.2, 0.2, 0.2],
        "mean_hsv": [90.0, 120.0, 150.0],
        "timestamp": 1700000000.0 + seed,
    }


def _seed_memory(tmp_path: Path, n_pos: int = 44, n_neg: int = 10) -> ExemplarMemory:
    """Seed via the public JSON persistence format only."""
    pos = [_exemplar(i % 22, 1000 + i) for i in range(n_pos)]
    neg = [_exemplar(9 + (i % 2), 2000 + i, source="ui_reject") for i in range(n_neg)]
    jp = tmp_path / "learned_exemplars.json"
    jp.write_text(json.dumps({"classes": {}, "positive": pos, "negative": neg}))
    mem = ExemplarMemory(json_path=jp)
    assert mem.load()
    return mem


# --- 1. Super-category code table -------------------------------------------

def test_super_category_code_table_matches_get_super_category() -> None:
    # O-2: 8 üst kategori (22..26 artık HUD sayılmıyor: consumables 5,
    # currency 6, player 7); tablo tüm kanonik 0..26 sınıflarını kapsar.
    for cid in range(27):
        name = get_super_category(cid)
        assert name is not None
        assert em._SUPER_CATEGORY_CODE_BY_CLASS[cid] == em._SUPER_CATEGORY_CODES[name]
    # No canonical class is left unmapped (-1).
    assert not (em._SUPER_CATEGORY_CODE_BY_CLASS < 0).any()
    # Distinct names map to distinct codes.
    assert len(set(em._SUPER_CATEGORY_CODES.values())) == 8


def test_match_positive_category_lock_returns_only_same_category(tmp_path: Path) -> None:
    mem = _seed_memory(tmp_path)
    crop = _rng_crop(77)
    # class 3 is a 3d_material; the lock must never return a HUD/minimap class.
    sim, best_cid, _, _ = mem.match_positive(crop, same_category_as=3)
    assert best_cid is not None
    assert get_super_category(best_cid) == get_super_category(3)
    assert sim > 0.0
    # class 17 is HUD; the lock must return a HUD class only.
    sim, best_cid, _, _ = mem.match_positive(crop, same_category_as=17)
    assert best_cid is not None
    assert get_super_category(best_cid) == "hud"


# --- 2. Vectorized rounding -------------------------------------------------

def test_vectorized_rounding_matches_per_element_round() -> None:
    rng = np.random.default_rng(123)
    hist = rng.random(432, dtype=np.float32)
    h1 = rng.random(16, dtype=np.float32)
    old_hist = [round(float(v), 5) for v in hist.flatten()]
    new_hist = np.round(hist.flatten().astype(np.float64), 5).tolist()
    assert old_hist == new_hist
    old_h = [round(float(v) / 3.7, 4) for v in h1]
    new_h = np.round(h1.astype(np.float64) / 3.7, 4).tolist()
    assert old_h == new_h


def test_extract_crop_features_values_are_rounded_deterministically() -> None:
    crop = _rng_crop(5)
    f1 = extract_crop_features(crop)
    f2 = extract_crop_features(crop)
    assert f1 is not None and f1 == f2
    assert len(f1["hsv_hist"]) == 432
    # Every histogram value is rounded to <= 5 decimals (vectorized np.round).
    for v in f1["hsv_hist"]:
        assert v == round(v, 5)
    for key in ("hist_h", "hist_s", "hist_v"):
        for v in f1[key]:
            assert v == round(v, 4)


# --- 3. feats= reuse + single extraction in refine_candidate ----------------

def test_match_feats_reuse_identical_to_reextraction(tmp_path: Path) -> None:
    mem = _seed_memory(tmp_path)
    crop = _rng_crop(31)
    feats = extract_crop_features(crop)
    assert feats is not None
    assert mem.match_positive(crop) == mem.match_positive(crop, feats=feats)
    assert mem.match_negative(crop) == mem.match_negative(crop, feats=feats)
    # feats=None still re-extracts (old behaviour preserved).
    assert mem.match_positive(crop, feats=None) == mem.match_positive(crop)


def test_refine_candidate_extracts_features_once(tmp_path: Path, monkeypatch) -> None:
    jp = tmp_path / "learned_exemplars.json"
    jp.write_text(json.dumps({"classes": {}, "positive": [], "negative": []}))
    mem = ExemplarMemory(json_path=jp)
    assert mem.load()
    frame = _rng_crop(41, size=480)
    crop = _rng_crop(1000)
    frame[100:196, 100:196] = crop
    # Register the very same crop as a positive exemplar through the public
    # path, so refine_candidate() takes the full reinforce branch.
    assert mem.add_positive_exemplar(crop, 0, "class_0") is True
    cand = {"class_id": 0, "class_name": "class_0", "confidence": 0.8,
            "bbox_xyxy": [100, 100, 196, 196]}
    calls = {"n": 0}
    orig = em.extract_crop_features

    def counting(crop_arr):
        calls["n"] += 1
        return orig(crop_arr)

    monkeypatch.setattr(em, "extract_crop_features", counting)
    keep, refined = mem.refine_candidate(frame, cand)
    assert calls["n"] == 1, f"expected exactly 1 extraction, got {calls['n']}"
    assert keep is True
    assert refined.get("exemplar_reinforced") is True


# --- 4. Shared purge predicate ----------------------------------------------

# NOTE (R49, e04f561 — v1 ve v2'de var): 22..26 meşru kanonik sınıftır
# (22..24 consumable, 25 currency/kinah, 26 player); zehir yalnızca
# essence_* adıyla belirlenir, ID aralığıyla değil. Bu yüzden
# {"class_id": 25, "class_name": "currency"} zehir DEĞİLDİR — D50'nin
# yazıldığı tabandaki predicate `cid >= 22` kuralını taşıyordu
# (2e939ac), R49 o kuralı kaldırdı. Currency burada _POISON'da değil,
# aşağıda _exemplar_is_clean için True beklentiyle kilitlenir.
_POISON: List[Any] = [
    {"class_id": 3, "class_name": "essence_fire", "source": "ui_approval"},
    {"class_id": -1, "class_name": "class_-1", "source": "ui_approval"},
    {"class_id": 17, "class_name": "class_17", "source": "3d_world"},
    {"class_id": 16, "class_name": "class_16", "source": "crest"},
    {"class_id": "xx", "class_name": "class_x", "source": "ui_approval"},
    "not-a-dict",
    None,
]


def test_sanitize_positive_purge_agrees_with_shared_predicate(tmp_path: Path) -> None:
    clean = [_exemplar(i % 22, 3000 + i) for i in range(10)]
    poison = [dict(p) if isinstance(p, dict) else p for p in _POISON]
    jp = tmp_path / "learned_exemplars.json"
    jp.write_text(json.dumps({"classes": {}, "positive": clean + poison, "negative": []}))
    mem = ExemplarMemory(json_path=jp)
    assert mem.load()
    # load() already purged the dict-shaped poison; the non-dict entry is
    # dropped by load() before sanitize (by design, see load() comment).
    # Every remaining positive must satisfy the shared predicate.
    assert len(mem._positive) == len(clean)
    assert all(_exemplar_is_clean(ex) for ex in mem._positive)
    res = mem.sanitize(persist=False)
    assert res["remaining_positive"] == len(clean)
    # load() already purged the 5 dict-shaped poison records, but sanitize()
    # folds that load-time count into the first explicit call (by design);
    # the second explicit call then reports 0.
    assert res["purged_positive"] == 5
    assert mem.sanitize(persist=False)["purged_positive"] == 0


def test_exemplar_is_clean_matches_inline_rules() -> None:
    for p in _POISON:
        assert _exemplar_is_clean(p) is False
    # R49 (e04f561): class 25 currency/kinah is a legitimate canonical
    # class, not poison — poison is by essence_* name only, never by
    # ID range (see _is_poisoned_essence_name "Decision 1").
    assert _exemplar_is_clean(
        {"class_id": 25, "class_name": "currency", "source": "ui_approval"}
    ) is True
    assert _exemplar_is_clean(_exemplar(3, 1)) is True
    assert _exemplar_is_clean(_exemplar(17, 1, source="ui_approval")) is True
    # Negative-only extra rule: 3D material negatives are unclean.
    assert _exemplar_is_clean(_exemplar(3, 1), negative=True) is False
    assert _exemplar_is_clean(_exemplar(9, 1), negative=True) is True


# --- 5. Masked dot == full dot + where --------------------------------------
# The optimization dots only the mask-passing rows. These tests lock the
# equivalence with an independent full-matrix oracle (the old computation).

def _oracle_match_positive(mem: ExemplarMemory, crop: np.ndarray, mask) -> tuple:
    feats = extract_crop_features(crop)
    assert feats is not None
    q = em.feature_dict_to_normalized_vector(feats)
    sims = np.dot(mem._pos_matrix, q)
    sims = np.where(mask, sims, -1.0)
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim < 0.0:
        return (0.0, None, None, None)
    ex = mem._positive[best_idx]
    return (round(best_sim, 4), int(ex["class_id"]), str(ex["class_name"]), ex)


def test_match_positive_masked_dot_matches_full_dot_oracle(tmp_path: Path) -> None:
    mem = _seed_memory(tmp_path)
    crop = _rng_crop(77)
    for anchor in (0, 3, 9, 10, 14, 17):
        name = get_super_category(anchor)
        assert name is not None
        mask = np.array(
            [get_super_category(int(c)) == name for c in mem._pos_cids], dtype=bool
        )
        assert mem.match_positive(crop, same_category_as=anchor) == \
            _oracle_match_positive(mem, crop, mask)
    for allowed in ({0, 1, 2}, {9, 11}, {16, 17, 18}, {10}):
        mask = np.isin(mem._pos_cids, list(allowed))
        assert mem.match_positive(crop, allowed_classes=allowed) == \
            _oracle_match_positive(mem, crop, mask)
    # Unmasked path is untouched: still the plain full dot.
    assert mem.match_positive(crop) == _oracle_match_positive(
        mem, crop, np.ones(len(mem._pos_cids), dtype=bool)
    )


def test_match_class_masked_dot_matches_full_dot_oracle(tmp_path: Path) -> None:
    mem = _seed_memory(tmp_path)
    crop = _rng_crop(77)
    feats = extract_crop_features(crop)
    assert feats is not None
    q = em.feature_dict_to_normalized_vector(feats)
    for cid in (0, 3, 9, 10, 14, 17):
        mask = mem._pos_cids == cid
        sims = np.where(mask, np.dot(mem._pos_matrix, q), -1.0)
        best_idx = int(np.argmax(sims))
        expected = (round(float(sims[best_idx]), 4), mem._positive[best_idx])
        assert mem.match_class(crop, class_id=cid) == expected
    # Unknown class: no rows -> (0.0, None).
    assert mem.match_class(crop, class_id=99) == (0.0, None)


# --- 5. K-19 (P1-15): remove_* exemplar — yanlış öğretimi geri alma ------

def test_remove_negative_exemplar_undoes_wrong_teach(tmp_path: Path) -> None:
    # Yanlış negatif öğretimi geri alınabilir: kayıt listeden çıkar,
    # matrisler yeniden kurulur, disk yazılır. Aralık dışı indeks False
    # döner, istisna fırlatmaz.
    mem = _seed_memory(tmp_path, n_pos=2, n_neg=3)
    assert len(mem.get_negative_exemplars()) == 3
    assert mem.remove_negative_exemplar(1) is True
    assert len(mem.get_negative_exemplars()) == 2
    # Disk de güncellendi: yeniden yükleyince 2 negatif kalır.
    mem2 = ExemplarMemory(json_path=tmp_path / "learned_exemplars.json")
    assert mem2.load()
    assert len(mem2.get_negative_exemplars()) == 2
    assert mem.remove_negative_exemplar(99) is False
    assert mem.remove_negative_exemplar(-1) is False
    assert len(mem.get_negative_exemplars()) == 2
    # Pozitif için simetrik yol.
    assert mem.remove_positive_exemplar(0) is True
    assert len(mem.get_positive_exemplars()) == 1
    assert mem.remove_positive_exemplar(99) is False
