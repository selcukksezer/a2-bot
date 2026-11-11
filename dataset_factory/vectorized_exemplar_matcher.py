# -*- coding: utf-8 -*-
"""
dataset_factory/vectorized_exemplar_matcher.py
==============================================
Vectorized Positive/Negative Exemplar Matcher for 438-Dimensional
L2-Normalized Visual Feature Vectors with Super-Category Masking.

Architecture & Guarantees:
1. 438-Dimensional L2-Normalized Feature Space:
   - 432 bins: 3D HSV Histogram (12 Hue x 6 Sat x 6 Val), L2-normalized, weighted by sqrt(0.70)
   - 1 bin:    Aspect Ratio (clamped [0.05, 10.0] / 10.0), weighted by sqrt(0.15)
   - 5 bins:   Texture Descriptor [mean_edge, std_edge, laplacian, mean_gray, std_gray],
               L2-normalized, weighted by sqrt(0.15)
   - Combined vector is L2-normalized to unit length (||v||_2 == 1.0) so that
     a single `np.dot(matrix, query)` directly computes exact cosine similarity in [-1.0, 1.0].
2. Vectorized Matrix Multiplication (one BLAS call, O(N * 438) work for N exemplars):
   - Pre-allocates contiguous `np.float32` matrices of shape `(N, 438)` for both
     positive and negative exemplars.
   - Pre-caches integer `class_id` arrays `(N,)` and integer `super_category_id` arrays `(N,)`
     so super-category locking (`self._pos_super_cats == target_cat_id`) executes in pure C/NumPy
     without Python-level loops.
3. Strict Super-Category Isolation:
   - Prevents cross-category class pollution (e.g., 3D world materials 0..8, 12 can never
     match or repair into HUD classes 13, 16..21 or minimap classes 9, 11).
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

FEATURE_VECTOR_DIM: int = 438
HSV_HIST_BINS: int = 432
ASPECT_RATIO_BINS: int = 1
TEXTURE_BINS: int = 5

WEIGHT_HSV: float = 0.70
WEIGHT_ASPECT_RATIO: float = 0.15
WEIGHT_TEXTURE: float = 0.15

# Pre-computed sqrt weights (D42): feature_dict_to_normalized_vector used to
# call np.sqrt() on these constants for every exemplar vectorized; the values
# never change, so compute them once at import time.
_SQRT_WEIGHT_HSV: np.float32 = np.sqrt(WEIGHT_HSV, dtype=np.float32)
_SQRT_WEIGHT_ASPECT_RATIO: np.float32 = np.sqrt(WEIGHT_ASPECT_RATIO, dtype=np.float32)
_SQRT_WEIGHT_TEXTURE: np.float32 = np.sqrt(WEIGHT_TEXTURE, dtype=np.float32)

# Canonical YOLO class IDs (0..24: core 0..21 + consumables 22..24
# hp_potion/mp_potion/buff_scroll, cf. dataset_factory/class_taxonomy.py
# CONSUMABLE_CANONICAL_NAMES and scroll_potion_classifier.py
# CANONICAL_HUD_CLASS_IDS). Currency (25) and player (26) are canonical
# YOLO classes but are not visual exemplars for this matcher.
CANONICAL_CLASSES: Set[int] = set(range(25))

# Super-Category Sets — MUST mirror dataset_factory/class_taxonomy.py:
# MATERIAL_3D_CLASSES is {0..8, 12}: classes 6/7/8 (herb_edana,
# herb_lumina, ore_obsidian) are 3d_materials, NOT a separate "3d_crests"
# category (F-1355; D42 merged them into 3d_materials in exemplar_memory.py).
SUPER_CATEGORY_3D_MATERIALS: Set[int] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 12}
SUPER_CATEGORY_MINIMAP: Set[int] = {9, 11}
SUPER_CATEGORY_HUD: Set[int] = {13, 16, 17, 18, 19, 20, 21}
SUPER_CATEGORY_3D_MONSTERS: Set[int] = {10}
SUPER_CATEGORY_3D_PATHS: Set[int] = {14, 15}
SUPER_CATEGORY_CONSUMABLES: Set[int] = {22, 23, 24}

# Integer codes for vectorized NumPy super-category masking.
# NOTE: id 4 (former "3d_crests") is intentionally left unassigned after
# F-1355 merged {6, 7, 8} into 3d_materials; the remaining ids are unchanged
# so persisted/masked codes keep their meaning.
SUPER_CAT_ID_NONE: int = -1
SUPER_CAT_ID_3D_MATERIALS: int = 1
SUPER_CAT_ID_MINIMAP: int = 2
SUPER_CAT_ID_HUD: int = 3
SUPER_CAT_ID_3D_MONSTERS: int = 5
SUPER_CAT_ID_3D_PATHS: int = 6
SUPER_CAT_ID_CONSUMABLES: int = 7

SUPER_CATEGORY_NAME_TO_ID: Dict[str, int] = {
    "3d_materials": SUPER_CAT_ID_3D_MATERIALS,
    "minimap": SUPER_CAT_ID_MINIMAP,
    "hud": SUPER_CAT_ID_HUD,
    "3d_monsters": SUPER_CAT_ID_3D_MONSTERS,
    "3d_paths": SUPER_CAT_ID_3D_PATHS,
    "consumables": SUPER_CAT_ID_CONSUMABLES,
}

SUPER_CATEGORY_ID_TO_NAME: Dict[int, str] = {
    v: k for k, v in SUPER_CATEGORY_NAME_TO_ID.items()
}


def get_super_category(class_id: int) -> Optional[str]:
    """
    Map canonical class_id to its super-category name:
    - '3d_materials': 3D world gatherable resources (0..8, 12 — taxonomy
      MATERIAL_3D_CLASSES; classes 6/7/8 are materials, not crests, F-1355)
    - 'minimap': Minimap symbols (9, 11)
    - 'hud': HUD and cognition symbols (13, 16..21)
    - '3d_monsters': 3D monsters (10)
    - '3d_paths': Walkable path & obstacles (14, 15)
    - 'consumables': HUD consumables (22 hp_potion, 23 mp_potion, 24 buff_scroll)
    Returns None for non-canonical classes.
    """
    try:
        cid = int(class_id)
    except (TypeError, ValueError):
        return None
    if cid in SUPER_CATEGORY_3D_MATERIALS:
        return "3d_materials"
    if cid in SUPER_CATEGORY_MINIMAP:
        return "minimap"
    if cid in SUPER_CATEGORY_HUD:
        return "hud"
    if cid in SUPER_CATEGORY_3D_MONSTERS:
        return "3d_monsters"
    if cid in SUPER_CATEGORY_3D_PATHS:
        return "3d_paths"
    if cid in SUPER_CATEGORY_CONSUMABLES:
        return "consumables"
    return None


def get_super_category_id(class_id: int) -> int:
    """Return integer super-category code for vectorized array comparisons."""
    cat_name = get_super_category(class_id)
    if cat_name is None:
        return SUPER_CAT_ID_NONE
    return SUPER_CATEGORY_NAME_TO_ID.get(cat_name, SUPER_CAT_ID_NONE)


def is_same_super_category(cid1: int, cid2: int) -> bool:
    """Return True if and only if cid1 and cid2 belong to the same valid super-category."""
    cat1 = get_super_category_id(cid1)
    cat2 = get_super_category_id(cid2)
    return cat1 != SUPER_CAT_ID_NONE and cat1 == cat2


def _coerce_class_id(exemplar: Dict[str, Any]) -> int:
    """Best-effort int coercion of an exemplar's class_id; -1 when absent/non-numeric."""
    try:
        return int(exemplar.get("class_id", -1))
    except (TypeError, ValueError):
        return -1


def _coerce_timestamp(exemplar: Dict[str, Any]) -> float:
    """Best-effort float coercion of an exemplar's timestamp; 0.0 when absent/non-numeric."""
    try:
        return float(exemplar.get("timestamp", 0.0))
    except (TypeError, ValueError):
        return 0.0


def normalize_l2_vector(vec: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
    """
    Ensure a 1D vector of length 438 is float32 and L2-normalized to unit length.
    Raises ValueError for a non-empty vector whose length differs from 438:
    silently padding/truncating would corrupt cosine similarities with a
    vector that is not what the caller supplied. An empty input still yields
    the zero vector (no data to match).
    """
    arr = np.asarray(vec, dtype=np.float32).ravel()
    if arr.size != FEATURE_VECTOR_DIM:
        if arr.size == 0:
            return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
        raise ValueError(
            f"feature vector must have length {FEATURE_VECTOR_DIM}, got {arr.size}"
        )
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9 or not np.isfinite(norm):
        return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def feature_dict_to_normalized_vector(feat: Dict[str, Any]) -> np.ndarray:
    """
    Convert a visual feature dictionary or pre-computed vector dict into a
    438-dimensional L2-normalized float32 vector (`||v||_2 = 1.0`).

    If `feat` already contains `"feature_vector"` or `"vector"` of length 438,
    it is directly L2-normalized and returned. Otherwise builds the 438-D vector from:
    - `hsv_hist`: 432 bins (L2-normalized, weighted by sqrt(0.70))
    - `aspect_ratio`: 1 bin (clamped [0.05, 10.0] / 10.0, weighted by sqrt(0.15))
      so the component stays in [0, 1] and can contribute at most its design
      weight (0.15) of the squared norm; the old / 2.0 scaling let it reach
      5.0 and dominate the HSV histogram (up to ~79% of the energy).
    - `texture`: 5 bins (L2-normalized, weighted by sqrt(0.15))
    """
    if not feat:
        return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)

    if "feature_vector" in feat and feat["feature_vector"] is not None:
        return normalize_l2_vector(feat["feature_vector"])
    if "vector" in feat and feat["vector"] is not None:
        return normalize_l2_vector(feat["vector"])

    # S-2204: a present-but-wrong-size hsv_hist is corrupt input, not data to
    # silently repair. Padding with zeros / truncating (the old behaviour)
    # manufactured a plausible-looking 432-bin histogram from a broken one
    # and then normalized it, so a bad feature dict produced a confident
    # vector. Raise instead, mirroring normalize_l2_vector (F-1013).
    # A missing key keeps the old zero-component default.
    if "hsv_hist" in feat:
        hsv = np.asarray(feat["hsv_hist"], dtype=np.float32).ravel()
        if hsv.size != HSV_HIST_BINS:
            raise ValueError(
                f"hsv_hist must have length {HSV_HIST_BINS}, got {hsv.size}"
            )
    else:
        hsv = np.zeros(HSV_HIST_BINS, dtype=np.float32)
    hsv_norm = hsv / (float(np.linalg.norm(hsv)) + 1e-9)

    ar = 1.0
    try:
        ar = float(feat.get("aspect_ratio", 1.0))
    except (TypeError, ValueError):
        ar = 1.0
    if not np.isfinite(ar):
        ar = 1.0
    ar_val = np.array([float(np.clip(ar, 0.05, 10.0)) / 10.0], dtype=np.float32)

    # S-2204: same rule for texture — a present-but-wrong-size texture
    # descriptor is corrupt input; do not pad/truncate it into 5 bins.
    if "texture" in feat:
        tex = np.asarray(feat["texture"], dtype=np.float32).ravel()
        if tex.size != TEXTURE_BINS:
            raise ValueError(
                f"texture must have length {TEXTURE_BINS}, got {tex.size}"
            )
    else:
        tex = np.zeros(TEXTURE_BINS, dtype=np.float32)
    tex_norm = tex / (float(np.linalg.norm(tex)) + 1e-9)

    combined = np.concatenate(
        [
            _SQRT_WEIGHT_HSV * hsv_norm,
            _SQRT_WEIGHT_ASPECT_RATIO * ar_val,
            _SQRT_WEIGHT_TEXTURE * tex_norm,
        ]
    )
    norm = float(np.linalg.norm(combined))
    if norm <= 1e-9 or not np.isfinite(norm):
        return np.zeros(FEATURE_VECTOR_DIM, dtype=np.float32)
    return (combined / norm).astype(np.float32)


def coerce_query_vector(query: Union[np.ndarray, Dict[str, Any], Sequence[float]]) -> np.ndarray:
    """Convert a query (feature dict, 1D array, or sequence) to a 438-D L2-normalized float32 vector."""
    if isinstance(query, dict):
        return feature_dict_to_normalized_vector(query)
    return normalize_l2_vector(query)


@dataclass(frozen=True)
class ExemplarMatchResult:
    """Structured result of a vectorized exemplar match."""

    matched: bool
    similarity: float
    class_id: Optional[int]
    class_name: Optional[str]
    super_category: Optional[str]
    exemplar_index: int
    exemplar: Optional[Dict[str, Any]]


class VectorizedExemplarMatcher:
    """
    Vectorized Positive/Negative Exemplar Matcher operating on 438-dimensional
    L2-normalized feature vectors via a single `np.dot(matrix, query)` call and
    vectorized super-category boolean masking. Each match costs O(N * 438)
    work for N indexed exemplars (one BLAS call, linear in N).
    """

    def __init__(
        self,
        positive_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
        negative_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
        suppress_neg_threshold: float = 0.82,
        reinforce_pos_threshold: float = 0.70,
        repair_pos_threshold: float = 0.88,
        cross_class_neg_scale: float = 0.90,
    ) -> None:
        self.suppress_neg_threshold = float(suppress_neg_threshold)
        self.reinforce_pos_threshold = float(reinforce_pos_threshold)
        self.repair_pos_threshold = float(repair_pos_threshold)
        self.cross_class_neg_scale = float(cross_class_neg_scale)
        for _name, _value in (
            ("suppress_neg_threshold", self.suppress_neg_threshold),
            ("reinforce_pos_threshold", self.reinforce_pos_threshold),
            ("repair_pos_threshold", self.repair_pos_threshold),
            ("cross_class_neg_scale", self.cross_class_neg_scale),
        ):
            # Cosine thresholds/scales only mean something in [0, 1]; NaN
            # also fails the chained comparison and is rejected here.
            if not 0.0 <= _value <= 1.0:
                raise ValueError(f"{_name} must be in [0.0, 1.0], got {_value}")

        self._lock = threading.RLock()
        self._positive: List[Dict[str, Any]] = []
        self._negative: List[Dict[str, Any]] = []

        self._pos_matrix: np.ndarray = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
        self._pos_cids: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._pos_super_cats: np.ndarray = np.zeros((0,), dtype=np.int32)

        self._neg_matrix: np.ndarray = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
        self._neg_cids: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._neg_super_cats: np.ndarray = np.zeros((0,), dtype=np.int32)

        if positive_exemplars or negative_exemplars:
            self.rebuild_matrices(
                positive_exemplars=positive_exemplars or [],
                negative_exemplars=negative_exemplars or [],
            )

    @property
    def feature_dim(self) -> int:
        """Return feature vector dimensionality (438)."""
        return FEATURE_VECTOR_DIM

    @property
    def positive_count(self) -> int:
        """Return number of indexed positive exemplars."""
        with self._lock:
            return int(self._pos_matrix.shape[0])

    @property
    def negative_count(self) -> int:
        """Return number of indexed negative exemplars."""
        with self._lock:
            return int(self._neg_matrix.shape[0])

    @property
    def positive_matrix(self) -> np.ndarray:
        """Return read-only view of the (N_pos, 438) L2-normalized positive matrix."""
        with self._lock:
            view = self._pos_matrix.view()
            view.flags.writeable = False
            return view

    @property
    def negative_matrix(self) -> np.ndarray:
        """Return read-only view of the (N_neg, 438) L2-normalized negative matrix."""
        with self._lock:
            view = self._neg_matrix.view()
            view.flags.writeable = False
            return view

    def rebuild_matrices(
        self,
        positive_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
        negative_exemplars: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """
        Build contiguous `(N, 438)` L2-normalized float32 matrices and pre-computed
        class/super-category index arrays for vectorized matching.
        Exemplars whose feature vector is all zeros (no usable visual signal)
        are dropped from the index instead of entering it as a silent
        zero row: a zero row can never match, yet it occupies an index slot
        and its class/super-category entries would pollute masks.
        """
        with self._lock:
            if positive_exemplars is not None:
                self._positive = [dict(ex) for ex in positive_exemplars if isinstance(ex, dict)]
            if negative_exemplars is not None:
                self._negative = [dict(ex) for ex in negative_exemplars if isinstance(ex, dict)]

            if self._positive:
                pos_vecs_all = [feature_dict_to_normalized_vector(ex) for ex in self._positive]
                kept = [i for i, v in enumerate(pos_vecs_all) if np.any(v)]
                self._positive = [self._positive[i] for i in kept]
                pos_vecs = [pos_vecs_all[i] for i in kept]
                if not pos_vecs:
                    self._pos_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                    self._pos_cids = np.zeros((0,), dtype=np.int32)
                    self._pos_super_cats = np.zeros((0,), dtype=np.int32)
                else:
                    pos_mat = np.vstack(pos_vecs).astype(np.float32, copy=False)
                    pos_norms = np.linalg.norm(pos_mat, axis=1, keepdims=True)
                    self._pos_matrix = np.ascontiguousarray(pos_mat / np.maximum(pos_norms, 1e-9), dtype=np.float32)
                    self._pos_cids = np.array(
                        [_coerce_class_id(ex) for ex in self._positive],
                        dtype=np.int32,
                    )
                    self._pos_super_cats = np.array(
                        [get_super_category_id(_coerce_class_id(ex)) for ex in self._positive],
                        dtype=np.int32,
                    )
            else:
                self._pos_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._pos_cids = np.zeros((0,), dtype=np.int32)
                self._pos_super_cats = np.zeros((0,), dtype=np.int32)

            if self._negative:
                neg_vecs_all = [feature_dict_to_normalized_vector(ex) for ex in self._negative]
                kept = [i for i, v in enumerate(neg_vecs_all) if np.any(v)]
                self._negative = [self._negative[i] for i in kept]
                neg_vecs = [neg_vecs_all[i] for i in kept]
                if not neg_vecs:
                    self._neg_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                    self._neg_cids = np.zeros((0,), dtype=np.int32)
                    self._neg_super_cats = np.zeros((0,), dtype=np.int32)
                else:
                    neg_mat = np.vstack(neg_vecs).astype(np.float32, copy=False)
                    neg_norms = np.linalg.norm(neg_mat, axis=1, keepdims=True)
                    self._neg_matrix = np.ascontiguousarray(neg_mat / np.maximum(neg_norms, 1e-9), dtype=np.float32)
                    self._neg_cids = np.array(
                        [_coerce_class_id(ex) for ex in self._negative],
                        dtype=np.int32,
                    )
                    self._neg_super_cats = np.array(
                        [get_super_category_id(_coerce_class_id(ex)) for ex in self._negative],
                        dtype=np.int32,
                    )
            else:
                self._neg_matrix = np.zeros((0, FEATURE_VECTOR_DIM), dtype=np.float32)
                self._neg_cids = np.zeros((0,), dtype=np.int32)
                self._neg_super_cats = np.zeros((0,), dtype=np.int32)

    # ------------------------------------------------------------------
    # Incremental index maintenance (D42)
    #
    # add_positive_exemplar / add_negative_exemplar used to call the full
    # rebuild_matrices() (re-vectorize EVERY exemplar, both polarities) for
    # every single add — O(N) Python-level work per add, O(N**2) for a
    # bulk import of N exemplars (200 one-by-one adds: ~0.7-2.9s on this
    # VM, min of 3 timeit runs, load avg 20-40). The helpers
    # below append/replace a single already-normalized row instead; the
    # full rebuild is kept only for the rare eviction path (row removal
    # shifts indices) and for explicit rebuild_matrices() calls.
    # All helpers assume `_lock` is held and keep `self._positive` /
    # `self._negative` aligned with their matrix rows, class-id arrays and
    # super-category arrays.
    # ------------------------------------------------------------------
    def _append_positive_row_locked(
        self, exemplar: Dict[str, Any], q_vec: np.ndarray
    ) -> None:
        self._positive.append(dict(exemplar))
        cid = _coerce_class_id(exemplar)
        self._pos_matrix = np.ascontiguousarray(
            np.vstack([self._pos_matrix, q_vec]), dtype=np.float32
        )
        self._pos_cids = np.append(self._pos_cids, np.int32(cid))
        self._pos_super_cats = np.append(
            self._pos_super_cats, np.int32(get_super_category_id(cid))
        )

    def _append_negative_row_locked(
        self, exemplar: Dict[str, Any], q_vec: np.ndarray
    ) -> None:
        self._negative.append(dict(exemplar))
        cid = _coerce_class_id(exemplar)
        self._neg_matrix = np.ascontiguousarray(
            np.vstack([self._neg_matrix, q_vec]), dtype=np.float32
        )
        self._neg_cids = np.append(self._neg_cids, np.int32(cid))
        self._neg_super_cats = np.append(
            self._neg_super_cats, np.int32(get_super_category_id(cid))
        )

    def _update_positive_row_locked(
        self, index: int, exemplar: Dict[str, Any], q_vec: np.ndarray
    ) -> None:
        """Replace one indexed positive row in place (dedup-hit path)."""
        self._positive[index].update(exemplar)
        if not self._pos_matrix.flags.writeable:
            self._pos_matrix = self._pos_matrix.copy()
        self._pos_matrix[index] = q_vec
        cid = _coerce_class_id(exemplar)
        self._pos_cids[index] = cid
        self._pos_super_cats[index] = get_super_category_id(cid)

    def _update_negative_row_locked(
        self, index: int, exemplar: Dict[str, Any], q_vec: np.ndarray
    ) -> None:
        """Replace one indexed negative row in place (dedup-hit path)."""
        self._negative[index].update(exemplar)
        if not self._neg_matrix.flags.writeable:
            self._neg_matrix = self._neg_matrix.copy()
        self._neg_matrix[index] = q_vec
        cid = _coerce_class_id(exemplar)
        self._neg_cids[index] = cid
        self._neg_super_cats[index] = get_super_category_id(cid)

    def _best_positive_index(
        self, q_vec: np.ndarray, valid_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float]:
        """Shared dot + mask + argmax core for the positive matchers (D42).

        Assumes `_lock` is held. Returns `(-1, -1.0)` when the positive
        index is empty; otherwise `(best_idx, raw_best_similarity)` with
        masked-out rows forced to -1.0. Callers keep their own threshold /
        empty-mask early returns, so each method's observable behaviour is
        unchanged — only the triplicated dot/where/argmax block lives here.
        """
        if self._pos_matrix.shape[0] == 0:
            return -1, -1.0
        sims = np.dot(self._pos_matrix, q_vec)
        if valid_mask is not None:
            sims = np.where(valid_mask, sims, -1.0)
        best_idx = int(np.argmax(sims))
        return best_idx, float(sims[best_idx])

    def build_super_category_mask(
        self,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
        allowed_classes: Optional[Collection[int]] = None,
        for_negative: bool = False,
    ) -> np.ndarray:
        """
        Compute a boolean mask of shape `(N,)` using vectorized NumPy comparisons
        over pre-cached super-category IDs and class IDs.
        """
        with self._lock:
            cids = self._neg_cids if for_negative else self._pos_cids
            scats = self._neg_super_cats if for_negative else self._pos_super_cats
            n = cids.shape[0]
            if n == 0:
                return np.zeros((0,), dtype=bool)

            mask = np.ones((n,), dtype=bool)

            if same_category_as is not None:
                target_cat_id = get_super_category_id(int(same_category_as))
                if target_cat_id == SUPER_CAT_ID_NONE:
                    return np.zeros((n,), dtype=bool)
                mask &= (scats == target_cat_id)

            if super_category is not None:
                cat_key = str(super_category).strip().lower()
                target_cat_id = SUPER_CATEGORY_NAME_TO_ID.get(cat_key, SUPER_CAT_ID_NONE)
                if target_cat_id == SUPER_CAT_ID_NONE:
                    return np.zeros((n,), dtype=bool)
                mask &= (scats == target_cat_id)

            if allowed_classes is not None:
                allowed_arr = np.fromiter((int(c) for c in allowed_classes), dtype=np.int32)
                if allowed_arr.size == 0:
                    return np.zeros((n,), dtype=bool)
                mask &= np.isin(cids, allowed_arr)

            return mask

    def match_positive(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> Tuple[float, Optional[int], Optional[str], Optional[Dict[str, Any]]]:
        """
        Vectorized positive exemplar match via single `np.dot(matrix, query)` and
        super-category masking.

        Only a best similarity at or above `min_similarity` (default:
        `reinforce_pos_threshold`) counts as a match; below it this returns
        `(0.0, None, None, None)` instead of a thresholdless "best" exemplar
        that may not match at all. Pass `min_similarity=0.0` explicitly to
        recover the old best-regardless behaviour.

        Returns:
            `(max_similarity, best_class_id, best_class_name, best_exemplar)`
        """
        threshold = float(self.reinforce_pos_threshold if min_similarity is None else min_similarity)
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return 0.0, None, None, None

        with self._lock:
            if self._pos_matrix.shape[0] == 0:
                return 0.0, None, None, None

            valid_mask: Optional[np.ndarray] = None
            if same_category_as is not None or super_category is not None or allowed_classes is not None:
                valid_mask = self.build_super_category_mask(
                    same_category_as=same_category_as,
                    super_category=super_category,
                    allowed_classes=allowed_classes,
                    for_negative=False,
                )
                if not np.any(valid_mask):
                    return 0.0, None, None, None

            best_idx, best_sim = self._best_positive_index(q_vec, valid_mask)
            if best_sim < threshold:
                return 0.0, None, None, None

            best_sim = float(np.clip(best_sim, 0.0, 1.0))
            best_ex = self._positive[best_idx]
            best_cid = int(best_ex.get("class_id", 0))
            best_cname = str(best_ex.get("class_name", f"class_{best_cid}"))
            return round(best_sim, 4), best_cid, best_cname, best_ex

    def match_positive_detailed(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
        min_similarity: Optional[float] = None,
    ) -> ExemplarMatchResult:
        """Return a structured `ExemplarMatchResult` for a positive exemplar query."""
        threshold = float(self.reinforce_pos_threshold if min_similarity is None else min_similarity)
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

        with self._lock:
            if self._pos_matrix.shape[0] == 0:
                return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

            valid_mask: Optional[np.ndarray] = None
            if same_category_as is not None or super_category is not None or allowed_classes is not None:
                valid_mask = self.build_super_category_mask(
                    same_category_as=same_category_as,
                    super_category=super_category,
                    allowed_classes=allowed_classes,
                    for_negative=False,
                )
                if not np.any(valid_mask):
                    return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

            best_idx, best_sim = self._best_positive_index(q_vec, valid_mask)
            if best_sim < 0.0:
                return ExemplarMatchResult(False, 0.0, None, None, None, -1, None)

            best_sim = round(float(np.clip(best_sim, 0.0, 1.0)), 4)
            best_ex = self._positive[best_idx]
            best_cid = int(best_ex.get("class_id", 0))
            best_cname = str(best_ex.get("class_name", f"class_{best_cid}"))
            best_scat = get_super_category(best_cid)
            return ExemplarMatchResult(
                matched=(best_sim >= threshold),
                similarity=best_sim,
                class_id=best_cid,
                class_name=best_cname,
                super_category=best_scat,
                exemplar_index=best_idx,
                exemplar=best_ex,
            )

    def match_positive_for_repair(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        allowed_classes: Optional[Collection[int]] = None,
        same_category_as: Optional[int] = None,
        super_category: Optional[str] = None,
    ) -> ExemplarMatchResult:
        """Repair-grade positive match: only a best similarity at or above
        `repair_pos_threshold` (default 0.88) counts as a repair candidate.
        This is the read site for `repair_pos_threshold`; below the threshold
        the result reports `matched=False` so callers never repair a box
        class on a weak similarity."""
        return self.match_positive_detailed(
            query,
            allowed_classes=allowed_classes,
            same_category_as=same_category_as,
            super_category=super_category,
            min_similarity=self.repair_pos_threshold,
        )

    def match_negative(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        class_id: Optional[int] = None,
        same_category_only: bool = False,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Vectorized negative exemplar match via single `np.dot(matrix, query)`.
        If `class_id` is given, cross-class negative matches are scaled by `cross_class_neg_scale` (0.90).
        If `same_category_only` is True, masks out negatives from unrelated super-categories.

        Returns:
            `(max_similarity, best_matching_negative_exemplar)`
        """
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return 0.0, None

        with self._lock:
            if self._neg_matrix.shape[0] == 0:
                return 0.0, None

            # Single matrix-vector multiplication: (N, 438) @ (438,) -> (N,)
            sims = np.dot(self._neg_matrix, q_vec)

            if class_id is not None:
                cid = int(class_id)
                if same_category_only:
                    valid_mask = self.build_super_category_mask(
                        same_category_as=cid,
                        for_negative=True,
                    )
                    if not np.any(valid_mask):
                        return 0.0, None
                    sims = np.where(valid_mask, sims, -1.0)
                scale = np.where(self._neg_cids == cid, 1.0, self.cross_class_neg_scale)
                sims = np.where(sims >= 0.0, sims * scale, -1.0)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])
            if best_sim <= 0.0:
                return 0.0, None

            best_sim = round(float(np.clip(best_sim, 0.0, 1.0)), 4)
            return best_sim, self._negative[best_idx]

    def match_class(
        self,
        query: Union[np.ndarray, Dict[str, Any], Sequence[float]],
        class_id: int,
        min_similarity: Optional[float] = None,
    ) -> Tuple[float, Optional[Dict[str, Any]]]:
        """
        Vectorized match against positive exemplars belonging specifically to `class_id`.
        Only a best similarity at or above `min_similarity` (default:
        `reinforce_pos_threshold`) counts as a match.
        """
        threshold = float(self.reinforce_pos_threshold if min_similarity is None else min_similarity)
        q_vec = coerce_query_vector(query)
        if not np.any(q_vec):
            return 0.0, None

        cid = int(class_id)
        with self._lock:
            if self._pos_matrix.shape[0] == 0:
                return 0.0, None

            mask = (self._pos_cids == cid)
            if not np.any(mask):
                return 0.0, None

            # Dot only the rows of this class: the old code dotted the whole
            # (N, 438) matrix and then masked the result, paying O(N*438)
            # for O(k*438) work (k = rows of this class). argmax over the
            # submatrix is identical to argmax over the masked full sims.
            idx = np.flatnonzero(mask)
            sub_sims = np.dot(self._pos_matrix[idx], q_vec)
            best_local = int(np.argmax(sub_sims))
            best_sim = float(sub_sims[best_local])
            if best_sim < threshold:
                return 0.0, None

            best_sim = round(float(np.clip(best_sim, 0.0, 1.0)), 4)
            return best_sim, self._positive[int(idx[best_local])]

    def match_batch(
        self,
        queries: Sequence[Union[np.ndarray, Dict[str, Any], Sequence[float]]],
        candidate_class_ids: Optional[Sequence[int]] = None,
        enforce_super_category: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Batch vectorized positive & negative matching for `M` queries simultaneously.
        Computes `(N_pos, 438) @ (438, M)` and `(N_neg, 438) @ (438, M)` in two BLAS calls.
        """
        if not queries:
            return []

        q_vecs = [coerce_query_vector(q) for q in queries]
        # Shape: (438, M)
        query_mat_t = np.column_stack(q_vecs).astype(np.float32, copy=False)
        m_count = len(q_vecs)

        with self._lock:
            has_pos = self._pos_matrix.shape[0] > 0
            has_neg = self._neg_matrix.shape[0] > 0

            # Single matrix-matrix multiply for all M queries: (N, 438) @ (438, M) -> (N, M)
            pos_sims_mat = (
                np.dot(self._pos_matrix, query_mat_t)
                if has_pos
                else np.zeros((0, m_count), dtype=np.float32)
            )
            neg_sims_mat = (
                np.dot(self._neg_matrix, query_mat_t)
                if has_neg
                else np.zeros((0, m_count), dtype=np.float32)
            )

            results: List[Dict[str, Any]] = []
            for j in range(m_count):
                cand_cid = (
                    int(candidate_class_ids[j])
                    if candidate_class_ids is not None and j < len(candidate_class_ids)
                    else None
                )

                # Negative evaluation for query j: a negative exemplar is only
                # reported when its best similarity is strictly positive. An
                # argmax over an all-negative (or all-zero) column is not a
                # match — reporting it would fabricate a negative exemplar
                # with similarity 0.0.
                neg_sim = 0.0
                neg_ex: Optional[Dict[str, Any]] = None
                if has_neg:
                    col_neg = neg_sims_mat[:, j]
                    if cand_cid is not None:
                        scale = np.where(self._neg_cids == cand_cid, 1.0, self.cross_class_neg_scale)
                        col_neg = col_neg * scale
                    neg_idx = int(np.argmax(col_neg))
                    raw_neg = float(col_neg[neg_idx])
                    if raw_neg > 0.0:
                        neg_sim = round(float(np.clip(raw_neg, 0.0, 1.0)), 4)
                        neg_ex = self._negative[neg_idx]

                # Positive evaluation for query j
                pos_sim = 0.0
                pos_cid: Optional[int] = None
                pos_cname: Optional[str] = None
                pos_ex: Optional[Dict[str, Any]] = None
                if has_pos:
                    col_pos = pos_sims_mat[:, j]
                    if enforce_super_category and cand_cid is not None:
                        target_cat = get_super_category_id(cand_cid)
                        if target_cat == SUPER_CAT_ID_NONE:
                            col_pos = np.full_like(col_pos, -1.0)
                        else:
                            col_pos = np.where(self._pos_super_cats == target_cat, col_pos, -1.0)
                    pos_idx = int(np.argmax(col_pos))
                    raw_best = float(col_pos[pos_idx])
                    # S-2203: strictly positive, mirroring the negative side
                    # (F-0904). `>= 0.0` reported a positive exemplar with
                    # similarity 0.0 — an argmax over an all-zero column is
                    # not a match.
                    if raw_best > 0.0:
                        pos_sim = round(float(np.clip(raw_best, 0.0, 1.0)), 4)
                        pos_ex = self._positive[pos_idx]
                        pos_cid = int(pos_ex.get("class_id", 0))
                        pos_cname = str(pos_ex.get("class_name", f"class_{pos_cid}"))

                results.append(
                    {
                        "query_index": j,
                        "candidate_class_id": cand_cid,
                        "is_suppressed": neg_sim >= self.suppress_neg_threshold,
                        "negative_similarity": neg_sim,
                        "negative_exemplar": neg_ex,
                        "positive_similarity": pos_sim,
                        "positive_class_id": pos_cid,
                        "positive_class_name": pos_cname,
                        "positive_exemplar": pos_ex,
                    }
                )
            return results

    def _append_index_row(
        self, vec: np.ndarray, cid: int, for_negative: bool = False
    ) -> None:
        """
        Append one already-L2-normalized row to the index without a full
        rebuild. The old add_* path called rebuild_matrices() per exemplar,
        which re-parsed every stored feature dict and re-normalized both
        matrices: bulk-adding N exemplars one by one cost O(N^2) parses
        (~127-168x slower than one bulk constructor call for N=200/500).
        Appending one row is a memcpy plus two small int appends; the
        eviction and dedup-hit paths still go through rebuild_matrices().
        Callers must hold self._lock and guarantee vec is non-zero and
        L2-normalized (add_* checks both before calling).
        """
        row = np.ascontiguousarray(vec.reshape(1, -1), dtype=np.float32)
        cat_id = np.int32(get_super_category_id(cid))
        if for_negative:
            self._neg_matrix = np.ascontiguousarray(
                np.vstack([self._neg_matrix, row]), dtype=np.float32
            )
            self._neg_cids = np.append(self._neg_cids, np.int32(cid))
            self._neg_super_cats = np.append(self._neg_super_cats, cat_id)
        else:
            self._pos_matrix = np.ascontiguousarray(
                np.vstack([self._pos_matrix, row]), dtype=np.float32
            )
            self._pos_cids = np.append(self._pos_cids, np.int32(cid))
            self._pos_super_cats = np.append(self._pos_super_cats, cat_id)

    def add_positive_exemplar(
        self,
        exemplar: Dict[str, Any],
        dedup_threshold: float = 0.96,
        max_per_class: int = 40,
    ) -> bool:
        """
        Add a positive exemplar dictionary and index it incrementally
        (single-row append / in-place update; only the quota-eviction path
        falls back to a full matrix rebuild). Rejects non-canonical classes
        (`class_id` outside `0..24`) and non-numeric `class_id` values
        (returns False instead of raising). A dedup hit updates the stored
        exemplar AND the indexed row, so the matrix reflects the new feature
        vector instead of going stale.
        """
        if not isinstance(exemplar, dict):
            return False
        cid = _coerce_class_id(exemplar)
        cname = str(exemplar.get("class_name", f"class_{cid}"))
        if cid not in CANONICAL_CLASSES or "essence" in cname.lower():
            return False

        q_vec = feature_dict_to_normalized_vector(exemplar)
        if not np.any(q_vec):
            return False

        with self._lock:
            if self._pos_matrix.shape[0] > 0:
                mask = (self._pos_cids == cid)
                if np.any(mask):
                    sims = np.dot(self._pos_matrix, q_vec)
                    sims = np.where(mask, sims, -1.0)
                    best_idx = int(np.argmax(sims))
                    if float(sims[best_idx]) >= dedup_threshold:
                        # Dedup hit: replace the single indexed row in place
                        # instead of re-vectorizing every exemplar (D42).
                        self._update_positive_row_locked(best_idx, exemplar, q_vec)
                        return True

            if int(np.count_nonzero(self._pos_cids == cid)) >= max_per_class:
                # Quota eviction removes a row, shifting indices: the rare
                # path that still needs a full rebuild to stay consistent.
                self._positive.append(dict(exemplar))
                cls_entries = [e for e in self._positive if _coerce_class_id(e) == cid]
                # Evict the genuinely oldest: smallest timestamp, with
                # insertion order breaking ties (timestamp-less exemplars
                # all default to 0.0, so plain min() picked arbitrarily).
                oldest_pos = min(
                    range(len(cls_entries)),
                    key=lambda i: (_coerce_timestamp(cls_entries[i]), i),
                )
                self._positive.remove(cls_entries[oldest_pos])
                self.rebuild_matrices()
                return True

            # Dedup hit in the normal (non-eviction) path: the exemplar dict
            # MUST be appended to the stored list as well as the matrix row.
            # _append_index_row() alone only appends the matrix row/cids, so
            # calling it here would desync the list from the matrix (the
            # list-append lives inside _append_positive_row_locked).
            self._append_positive_row_locked(exemplar, q_vec)
            return True

    def add_negative_exemplar(
        self,
        exemplar: Dict[str, Any],
        dedup_threshold: float = 0.96,
        max_negatives: int = 150,
    ) -> bool:
        """
        Add a negative exemplar dictionary and index it incrementally
        (single-row append / in-place update; only the FIFO-eviction path
        falls back to a full matrix rebuild). Rejects non-canonical classes
        and non-numeric `class_id` values (returns False instead of raising).
        A dedup hit updates the stored exemplar AND the indexed row, so the
        matrix reflects the new feature vector instead of going stale.
        Dedup is class-masked (S-5401): only negatives of the same
        `class_id` may dedup-hit, so a new negative can never overwrite
        another class's stored exemplar.
        """
        if not isinstance(exemplar, dict):
            return False
        cid = _coerce_class_id(exemplar)
        cname = str(exemplar.get("class_name", f"class_{cid}"))
        if cid not in CANONICAL_CLASSES or "essence" in cname.lower():
            return False

        q_vec = feature_dict_to_normalized_vector(exemplar)
        if not np.any(q_vec):
            return False

        with self._lock:
            if self._neg_matrix.shape[0] > 0:
                # S-5401: dedup is class-masked, mirroring add_positive_exemplar.
                # A global argmax could match another class's negative exemplar
                # (>= threshold) and .update() it into this class, silently
                # erasing that class's negative coverage (and flipping its
                # class_id on rebuild). Only same-class rows may dedup-hit;
                # a cross-class near-duplicate is appended as its own entry
                # instead.
                mask = (self._neg_cids == cid)
                if np.any(mask):
                    sims = np.dot(self._neg_matrix, q_vec)
                    sims = np.where(mask, sims, -1.0)
                    best_idx = int(np.argmax(sims))
                    if float(sims[best_idx]) >= dedup_threshold:
                        # Dedup hit: replace the single indexed row in place
                        # instead of re-vectorizing every exemplar (D42).
                        self._update_negative_row_locked(best_idx, exemplar, q_vec)
                        return True

            if len(self._negative) >= max_negatives:
                # FIFO eviction removes a row, shifting indices: the rare
                # path that still needs a full rebuild to stay consistent.
                self._negative.append(dict(exemplar))
                self._negative.pop(0)
                self.rebuild_matrices()
                return True

            # Same as the positive path: the exemplar dict MUST be appended
            # to the stored list as well as the matrix row; _append_index_row()
            # alone only appends the matrix row/cids and would desync them.
            self._append_negative_row_locked(exemplar, q_vec)
            return True

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics for the vectorized matcher."""
        with self._lock:
            return {
                "feature_dim": FEATURE_VECTOR_DIM,
                "positive_count": int(self._pos_matrix.shape[0]),
                "negative_count": int(self._neg_matrix.shape[0]),
                "positive_matrix_shape": list(self._pos_matrix.shape),
                "negative_matrix_shape": list(self._neg_matrix.shape),
                "suppress_neg_threshold": self.suppress_neg_threshold,
                "reinforce_pos_threshold": self.reinforce_pos_threshold,
                "repair_pos_threshold": self.repair_pos_threshold,
                "cross_class_neg_scale": self.cross_class_neg_scale,
            }


__all__ = [
    "FEATURE_VECTOR_DIM",
    "HSV_HIST_BINS",
    "ASPECT_RATIO_BINS",
    "TEXTURE_BINS",
    "CANONICAL_CLASSES",
    "SUPER_CATEGORY_3D_MATERIALS",
    "SUPER_CATEGORY_MINIMAP",
    "SUPER_CATEGORY_HUD",
    "SUPER_CATEGORY_3D_MONSTERS",
    "SUPER_CATEGORY_3D_PATHS",
    "SUPER_CATEGORY_CONSUMABLES",
    "ExemplarMatchResult",
    "VectorizedExemplarMatcher",
    "get_super_category",
    "get_super_category_id",
    "is_same_super_category",
    "normalize_l2_vector",
    "feature_dict_to_normalized_vector",
    "coerce_query_vector",
]
